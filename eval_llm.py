import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer_path = args.load_from
    if tokenizer_path == 'model' and getattr(args, 'model_type', 'minimind') == 'dsv4_mini':
        tokenizer_path = 'model/tokenizer_dsv4m'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if 'model' in args.load_from:
        if args.model_type == 'dsv4_mini':
            from model.model_dsv4_mini import DeepSeekV4MiniConfig, DeepSeekV4MiniForCausalLM
            is_micro = args.hidden_size < 768
            # max_seq_len is the trained context length. When YaRN is enabled,
            # the effective context extends to max_seq_len * rope_factor.
            effective_max = int(args.max_seq_len * args.rope_factor) if args.inference_rope_scaling else args.max_seq_len
            lm_config = DeepSeekV4MiniConfig(
                hidden_size=args.hidden_size, 
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.hidden_size // 128 if args.hidden_size % 128 == 0 else 4,
                moe_inter_dim=args.hidden_size,
                q_lora_rank=(args.hidden_size // 3 // 16 * 16) if is_micro else 256,
                o_lora_rank=(args.hidden_size // 3 // 16 * 16) if is_micro else 256,
                num_routed_experts=8 if is_micro else 16,
                hc_mult=2 if is_micro else 4,
                n_mtp_layers=0,
                max_seq_len=effective_max,
                inference_rope_scaling=args.inference_rope_scaling,
                rope_factor=args.rope_factor,
                original_seq_len=args.max_seq_len,
            )
            model = DeepSeekV4MiniForCausalLM(lm_config)
            moe_suffix = ''
        else:
            model = MiniMindForCausalLM(MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling
            ))
            moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    model = model.to(args.device).eval()
    # Each model type knows its own inference dtype recipe internally.
    if hasattr(model, 'to_inference_dtype'):
        model = model.to_inference_dtype()
    else:
        model = model.half()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（YaRN，训练长度→更长）")
    parser.add_argument('--max_seq_len', default=2048, type=int, help="dsv4_mini专用：训练时的最大上下文长度（YaRN外推起点，开启外推后实际最大长度=max_seq_len*rope_factor）")
    parser.add_argument('--rope_factor', default=16.0, type=float, help="dsv4_mini专用：YaRN外推倍数")
    parser.add_argument('--max_new_tokens', default=2000, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--model_type', default='minimind', type=str, choices=['minimind', 'dsv4_mini'], help="模型类型")
    parser.add_argument("--profile", default=0, type=int, choices=[0, 1], help="启用PyTorch Profiler，采样decode阶段热点")
    parser.add_argument("--profile_dir", default="./prof_log_infer", type=str, help="profiler trace输出目录")
    parser.add_argument("--profile_warmup", default=50, type=int, help="profile前跳过的forward步数（等待KV cache稳态）")
    parser.add_argument("--profile_active", default=30, type=int, help="profile采样的forward步数")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    if args.profile == 1:
        # Warm up KV cache / triton autotune with one short run before profiling.
        print(f'[Profile] Warming up ({args.profile_warmup} tokens)...')
        warm_ids = tokenizer('你好', return_tensors='pt').input_ids.to(args.device)
        with torch.inference_mode():
            model.generate(inputs=warm_ids, max_new_tokens=args.profile_warmup, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=None)
        torch.cuda.synchronize() if 'cuda' in args.device else None
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        activities = [ProfilerActivity.CPU]
        if 'cuda' in args.device:
            activities.append(ProfilerActivity.CUDA)
        prof_ctx = profile(
            activities=activities,
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
        )
        print(f'[Profile] Recording {args.profile_active} tokens to {args.profile_dir}')
        prompt = '请简单介绍一下你自己'
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        with prof_ctx:
            with torch.inference_mode():
                model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                               max_new_tokens=args.profile_active, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=None)
            torch.cuda.synchronize() if 'cuda' in args.device else None
        print(f'[Profile] Done. Run: tensorboard --logdir {args.profile_dir}')
        return

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()