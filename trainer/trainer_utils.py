"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM
from model.model_dsv4_mini import DeepSeekV4MiniConfig, DeepSeekV4MiniForCausalLM

def _count_minimind_params(model, config):
    """MiniMind: dense or MoE. Params live at `model.layers.X.mlp.experts.Y.*`."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    if not getattr(config, 'use_moe', False):
        return total, total
    n_routed = config.num_experts
    n_active = config.num_experts_per_tok
    n_layers = config.num_hidden_layers
    routed_all = sum(p.numel() for n, p in model.named_parameters() if '.experts.0.' in n) / 1e6
    expert = routed_all / max(n_layers, 1)
    total_expert = expert * n_routed * n_layers
    active_expert = expert * n_active * n_layers
    active = (total - total_expert) + active_expert
    return total, active


def _count_dsv4_mini_params(model, config):
    """dsv4_mini: MoE with shared expert + MTP block (skipped for inference active)."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_active = config.num_activated_experts
    n_layers = config.num_hidden_layers
    # Exclude MTP block (used only during training aux loss, not standard inference).
    routed_all = sum(p.numel() for n, p in model.named_parameters()
                     if '.experts.0.' in n and not n.startswith('mtp.')) / 1e6
    shared_all = sum(p.numel() for n, p in model.named_parameters()
                     if '.shared_experts.' in n and not n.startswith('mtp.')) / 1e6
    all_moe = sum(p.numel() for n, p in model.named_parameters()
                  if ('.experts.' in n or '.shared_experts.' in n) and not n.startswith('mtp.')) / 1e6
    expert = routed_all / max(n_layers, 1)
    active_expert = expert * n_active * n_layers + shared_all
    active = (total - all_moe) + active_expert
    return total, active


def get_model_params(model, config):
    if getattr(config, 'model_type', None) == 'dsv4_mini':
        total, active = _count_dsv4_mini_params(model, config)
    else:
        total, active = _count_minimind_params(model, config)
    if 0 < active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def build_dsv4_mini_config(args, inference: bool = False):
    is_micro = args.hidden_size < 768
    inference_rope_scaling = bool(getattr(args, "inference_rope_scaling", False)) if inference else False
    rope_factor = getattr(args, "rope_factor", 16.0)
    trained_max_seq_len = args.max_seq_len
    max_seq_len = int(trained_max_seq_len * rope_factor) if inference_rope_scaling else trained_max_seq_len

    lm_config = DeepSeekV4MiniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.hidden_size // 128 if args.hidden_size % 128 == 0 else 4,
        moe_inter_dim=math.ceil(args.hidden_size * math.pi / 64) * 64,
        num_routed_experts=1,
        num_shared_experts=0,
        max_seq_len=max_seq_len,
        attn_chunk_size=getattr(args, "attn_chunk_size", 0) if getattr(args, "attn_chunk_size", 0) > 0 else None,
        ce_chunk_size=getattr(args, "ce_chunk_size", 0) if getattr(args, "ce_chunk_size", 0) > 0 else None,
        inference_rope_scaling=inference_rope_scaling,
        rope_factor=rope_factor,
        original_seq_len=trained_max_seq_len,
    )

    if getattr(args, "use_moe", 0) == 1:
        lm_config.num_routed_experts = 16
        lm_config.num_shared_experts = 1
    else:
        lm_config.num_routed_experts = 1
        lm_config.num_shared_experts = 0

    return lm_config


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if getattr(lm_config, 'use_moe', False) else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda', model_type='minimind'):
    if model_type == 'dsv4_mini' and tokenizer_path == '../model':
        tokenizer_path = '../model/tokenizer_dsv4m'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if model_type == 'dsv4_mini':
        model = DeepSeekV4MiniForCausalLM(lm_config)
    else:
        model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if getattr(lm_config, 'use_moe', False) else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
