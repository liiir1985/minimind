import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset, PackedPretrainDataset, build_in_memory_validation_dataset, filter_validation_indices, shard_dataset_indices
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, build_dsv4_mini_config

warnings.filterwarnings('ignore')


def build_optimizer(model, args):
    if args.optimizer == 'adamw':
        Logger('Optimizer: torch.optim.AdamW (fp32 states)')
        return optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.optimizer == 'adamw8bit':
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as exc:
            raise ImportError(
                "未安装 bitsandbytes，无法使用 --optimizer adamw8bit。"
                "请在训练环境安装 bitsandbytes，或显式传 --optimizer adamw。"
            ) from exc
        Logger('Optimizer: bitsandbytes AdamW8bit (8-bit states)')
        return AdamW8bit(model.parameters(), lr=args.learning_rate)

    raise ValueError(f"未知优化器: {args.optimizer}")


@torch.no_grad()
def evaluate_validation_loss(val_loader, max_batches):
    if val_loader is None or max_batches <= 0:
        return None

    was_training = model.training
    model.eval()
    total = torch.tensor([0.0, 0.0], device=args.device)
    for batch_idx, (input_ids, labels) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            aux_loss = res.aux_loss if res.aux_loss is not None else 0.0
            loss = res.loss + aux_loss
        total[0] += loss.detach()
        total[1] += 1
        del input_ids, labels, res, loss

    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    return (total[0] / total[1]).item() if total[1].item() > 0 else None


def train_epoch(epoch, loader, val_loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    prof = None
    if args.profile == 1:
        from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
        )
        prof.start()
        Logger(f'Profiler enabled: writing to {args.profile_dir}. Will stop after wait+warmup+active=8 steps.')
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate,
                    schedule=args.lr_schedule, warmup_steps=args.warmup_steps, decay_ratio=args.decay_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if prof is not None:
            prof.step()
            if step - start_step >= 8:
                prof.stop()
                Logger(f'Profiler finished. Run: tensorboard --logdir {args.profile_dir}')
                prof = None
                return

        do_save = step % args.save_interval == 0 or step == iters
        do_log = step % args.log_interval == 0 or step == iters or do_save
        if do_log:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            val_batches = args.val_save_batches if do_save else args.val_log_batches
            val_loss = evaluate_validation_loss(val_loader, val_batches)
            val_msg = f', val_loss({val_batches}b): {val_loss:.4f}' if val_loss is not None else ''
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}{val_msg}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                log_data = {"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min}
                if val_loss is not None:
                    log_data["val_loss"] = val_loss
                wandb.log(log_data)

        if do_save and is_main_process():
            model.eval()
            moe_suffix = '_moe' if getattr(lm_config, 'use_moe', False) else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "wsd"],
                        help="LR schedule: cosine 或 wsd(Warmup-Stable-Decay)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="WSD warmup 步数 (仅 --lr_schedule wsd 时有效)")
    parser.add_argument("--decay_ratio", type=float, default=0.1, help="WSD decay 占总步数比例 (仅 --lr_schedule wsd 时有效)")
    parser.add_argument("--optimizer", type=str, default="adamw8bit", choices=["adamw8bit", "adamw"],
                        help="优化器: adamw8bit 使用 bitsandbytes 8-bit AdamW 以节省显存; adamw 使用 PyTorch fp32 AdamW")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=None, type=int, help="隐藏层维度（默认：minimind=768，dsv4_mini=1536）")
    parser.add_argument('--num_hidden_layers', default=None, type=int, help="隐藏层数量（默认：minimind=8，dsv4_mini=25）")
    parser.add_argument('--max_seq_len', default=4096, type=int, help="训练的最大上下文长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径 (jsonl 单文件走 PretrainDataset, 目录走 PackedPretrainDataset 读 pack_pretrain_parquet.py 产物)")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument('--model_type', default='minimind', type=str, choices=['minimind', 'dsv4_mini'], help="模型类型")
    parser.add_argument('--attn_chunk_size', default=0, type=int, help="dsv4_mini HCA 训练 chunk 大小 (0=自动使用window_size; 长上下文可显式设1024/2048)")
    parser.add_argument('--ce_chunk_size', default=0, type=int, help="dsv4_mini 交叉熵 seqlen 分块大小 (0=不分块, 走原版; 长上下文时设 1024/2048)")
    parser.add_argument('--rope_theta', default=10000.0, type=float, help="dsv4_mini 纯滑窗层 RoPE theta")
    parser.add_argument('--compress_rope_theta', default=160000.0, type=float, help="dsv4_mini HCA 压缩层 RoPE theta")
    parser.add_argument("--profile", default=0, type=int, choices=[0, 1], help="启用PyTorch Profiler，在前8个step采样并输出trace到--profile_dir（用tensorboard查看）")
    parser.add_argument("--profile_dir", default="../prof_log", type=str, help="profiler trace输出目录")
    parser.add_argument("--packed_chunk_size", type=int, default=512, help="PackedPretrainDataset 内 chunk 打乱粒度 (行数), 越小 domain 混合越均匀 (0=按整 rg)")
    parser.add_argument("--packed_mix_rgs", type=int, default=4, help="PackedPretrainDataset 同时交错的 row_group 数量, cache_row_groups 需 >= 此值")
    parser.add_argument("--val_ratio", type=float, default=0.001, help="从训练数据中固定抽取的验证集比例 (默认0.1%%; <=0关闭)")
    parser.add_argument("--val_seed", type=int, default=2024, help="验证集抽样随机种子")
    parser.add_argument("--val_log_batches", type=int, default=1, help="普通loss日志时计算验证集loss的batch数")
    parser.add_argument("--val_save_batches", type=int, default=20, help="save_interval/epoch末尾时计算验证集loss的batch数")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    if args.model_type == 'dsv4_mini':
        lm_config = build_dsv4_mini_config(args)
    else:
        args.hidden_size = args.hidden_size if args.hidden_size is not None else 768
        args.num_hidden_layers = args.num_hidden_layers if args.num_hidden_layers is not None else 8
        lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device, model_type=args.model_type)
    if os.path.isdir(args.data_path):
        train_ds = PackedPretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len,
                                         cache_row_groups=max(2, args.packed_mix_rgs))
        Logger(f'使用 PackedPretrainDataset: {args.data_path} ({len(train_ds):,} 条 packed 样本), '
               f'chunk_size={args.packed_chunk_size}, mix_rgs={args.packed_mix_rgs}')
    else:
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    val_ds, val_index_set = build_in_memory_validation_dataset(train_ds, args.val_ratio, args.val_seed, Logger)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=(device_type == "cuda")) if val_ds is not None else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model, args)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    is_packed = isinstance(train_ds, PackedPretrainDataset)
    for epoch in range(start_epoch, args.epochs):
        setup_seed(42 + epoch)
        if is_packed:
            # packed parquet: 按 row_group 聚簇 shuffle, 避免 DataLoader worker
            # 随机 index 打散 row_group 命中, 每次 __getitem__ 都解压 256MB
            indices = train_ds.rowgroup_shuffled_indices(
                seed=42 + epoch,
                chunk_size=args.packed_chunk_size,
                mix_rgs=args.packed_mix_rgs)
        else:
            indices = torch.randperm(len(train_ds)).tolist()
        indices = filter_validation_indices(indices, val_index_set)
        indices = shard_dataset_indices(indices, args.batch_size)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, val_loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, val_loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
