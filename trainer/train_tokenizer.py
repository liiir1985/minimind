# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
import glob
import gc
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pds
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

DATA_DIR = r'Y:\AI\Pretrain'  # 读取该目录下所有 parquet 文件
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 32000
SPECIAL_TOKENS_NUM = 36
TARGET_CHARS = 1_000_000_000  # 目标训练语料总字符数（约 1GB 文本），据此反推采样率
BATCH_SIZE = 1000  # 流式读取批次行数，控制内存峰值
SEED = 42  # 随机种子，保证抽样结果可复现
MAX_SEQ_CHARS = 5000  # 单条文本最多送入训练器的字符数。超长文本命中后随机截取一段该长度的
                       # 连续窗口（原文每个字节都有被采样的机会），既避免巨文本单序列在 BPE
                       # 预分词阶段内存爆炸，也让长/短文本贡献的采样长度相当
FILE_RATE_MULTIPLIERS = {'minimind_pretrain.parquet': 0.03}  # 指定文件的采样倍率（相对正常 rate），
                                                             # 用于给海量短文本文件（如 minimind_pretrain）降权

def _find_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.parquet'), recursive=True))
    if not files:
        raise FileNotFoundError(f'目录 {data_dir} 下未找到任何 parquet 文件')
    return files

def _read_batches(f, batch_size):
    """按 row group 流式产出 text 列批次。

    用 pyarrow.dataset 而不是 ParquetFile.iter_batches：iter_batches 会跨行组合并
    凑满 batch_size 行，行数少的文件会被整文件一次性读进内存（实测峰值 7GB）；
    dataset 扫描器按行组逐个读取，内存只占一个 row group，不会 OOM。
    """
    ds = pds.dataset(f)
    for batch in ds.to_batches(columns=['text'], batch_size=batch_size):
        yield batch

def collect_stats(data_dir, batch_size, max_seq_chars):
    """第一遍流式遍历：只统计每行文本长度（不加载文本内容），按文件聚合截断后的字符量。

    内存占用仅约 8 字节/行（只存长度数组），不会 OOM。返回值用于反推采样率。
    """
    file_capped = {}  # 文件名 -> Σ min(文本长度, max_seq_chars)
    total_rows = 0
    total_chars = 0
    files = _find_files(data_dir)
    print(f'共发现 {len(files)} 个 parquet 文件：')
    for f in files:
        print(' -', f)
    for f in files:
        print(f'统计长度: {f}')
        name = os.path.basename(f)
        capped = 0
        for batch in _read_batches(f, batch_size):
            lens = pc.fill_null(pc.utf8_length(batch['text']), -1).to_numpy(zero_copy_only=False)
            lens = lens[lens > 0]
            if len(lens):
                total_rows += len(lens)
                total_chars += int(lens.sum())
                capped += int(np.minimum(lens, max_seq_chars).sum())
        file_capped[name] = capped
    print(f'总行数: {total_rows:,} | 总字符数: {total_chars:,} | 平均长度: {total_chars / max(total_rows, 1):.1f}')
    for name, capped in file_capped.items():
        mult = FILE_RATE_MULTIPLIERS.get(name, 1.0)
        print(f' - {name}: 截断后 {capped:,} 字符, 倍率 {mult}')
    return file_capped

def compute_sampling_rate(file_capped, target_chars):
    """按目标总字符数反推统一采样率：P(保留) = target_chars / Σ(倍率 × 各文件截断字符量)。

    每条文本被采样到的概率相同（与长度无关）；FILE_RATE_MULTIPLIERS 指定的文件按倍率降权；
    期望保留的总字符数 ≈ target_chars。
    """
    effective = 0.0
    for name, capped in file_capped.items():
        effective += FILE_RATE_MULTIPLIERS.get(name, 1.0) * capped
    if effective <= 0 or target_chars >= effective:
        return 1.0  # 目标超出总量，全部保留
    rate = target_chars / effective
    print(f'反推采样率: rate = {rate:.6f}（预计保留约 {min(target_chars, effective) / 1e6:.2f}M 字符）')
    return rate

def get_texts(data_dir, batch_size, rate, max_seq_chars):
    """第二遍流式遍历：按统一概率随机抽样，命中文本逐条产出。

    FILE_RATE_MULTIPLIERS 指定的文件按其倍率缩小采样概率（如 minimind_pretrain 用 rate×0.05）。
    内存控制：
    - 用 pyarrow 按批次流式读取，只投影 text 列，不一次性载入整个文件；
    - 逐元素（.as_py()）取出抽样命中的文本，避免整批转成 Python 字符串造成内存尖峰；
    - 超长文本随机截取一段 max_seq_chars 的连续窗口（而非固定取开头），使原文每个字节
      都有被采样的机会，同时避免巨文本单序列在 BPE 预分词阶段内存爆炸。
    """
    rng = np.random.default_rng(SEED)
    stats = {'rows_read': 0, 'rows_sampled': 0, 'chars': 0}
    for f in _find_files(data_dir):
        file_rate = rate * FILE_RATE_MULTIPLIERS.get(os.path.basename(f), 1.0)
        print(f'开始读取: {f}（rate={file_rate:.6f}）')
        for batch in _read_batches(f, batch_size):
            arr = batch['text']
            n = len(arr)
            lens = pc.fill_null(pc.utf8_length(arr), -1).to_numpy(zero_copy_only=False)
            valid = lens > 0
            keep = (rng.random(n) < file_rate) & valid  # 统一概率，与文本长度无关
            stats['rows_read'] += n
            for i in range(n):
                if not keep[i]:
                    continue
                text = arr[i].as_py()
                if not text:
                    continue
                if len(text) > max_seq_chars:
                    # 随机取一段连续窗口，让长文本的每个字节都有机会被采样到
                    start = rng.integers(0, len(text) - max_seq_chars + 1)
                    text = text[start:start + max_seq_chars]
                stats['rows_sampled'] += 1
                stats['chars'] += len(text)
                yield text
        print(f'完成: {f}')
    print(f'总计读取 {stats["rows_read"]:,} 行，采样后保留 {stats["rows_sampled"]:,} 条文本，共 {stats["chars"]:,} 字符')

def train_tokenizer(data_dir, tokenizer_dir, vocab_size, target_chars, batch_size,
                    rate=None, max_seq_chars=MAX_SEQ_CHARS, special_tokens_num=SPECIAL_TOKENS_NUM):
    """完整训练流程：第一遍统计长度 -> 反推统一采样率 -> 第二遍抽样训练。

    未指定 rate 时自动按 target_chars 反推；已知采样率可传入（--rate）跳过第一遍统计。
    内存不足时自动降低数据量重试。
    """
    if rate is None:
        file_capped = collect_stats(data_dir, batch_size, max_seq_chars)  # 第一遍：统计文本长度（按文件聚合）

    def reduce_data(msg):
        """内存不足时降低数据量：给定 rate 则减半 rate，否则减半目标字符数。"""
        nonlocal rate, target_chars
        if rate is not None:
            if rate < 1e-7:
                return False
            rate /= 2
            print(f'[警告] {msg}，采样率降至 {rate:.6f} 后重试')
        else:
            if target_chars < 1_000_000:
                return False
            target_chars //= 2
            print(f'[警告] {msg}，目标总字符数降至 {target_chars / 1e6:.0f}M 后重试')
        gc.collect()
        return True

    while True:
        try:
            if rate is None:
                rate = compute_sampling_rate(file_capped, target_chars)  # 反推采样率
            else:
                print(f'跳过统计：使用给定采样率 rate={rate:.6f}')
            _train_once(data_dir, tokenizer_dir, vocab_size, rate, batch_size, max_seq_chars, special_tokens_num)
            return
        except KeyboardInterrupt:
            raise
        except MemoryError:
            if not reduce_data('内存不足'):
                raise
        except BaseException as e:
            # tokenizers(Rust) 内存分配失败会直接 panic（PanicException 继承自 BaseException，
            # 不会抛 Python 的 MemoryError），同样按内存不足处理
            if 'memory' not in str(e).lower():
                raise
            del e  # 释放异常 traceback 对失败帧（tokenizer/trainer 等 Rust 大对象）的引用，
                   # 否则重试期间旧内存一直不释放，导致反复 OOM
            if not reduce_data('tokenizer 内存分配失败'):
                raise

def _train_once(data_dir, tokenizer_dir, vocab_size, rate, batch_size, max_seq_chars, special_tokens_num):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)] # 预留一定数量的token位置
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens
    )
    texts = get_texts(data_dir, batch_size, rate, max_seq_chars)  # 第二遍：流式抽样训练
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_list)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True if token in special_tokens_list else False
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")

def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'},
        {"role": "assistant", "content": '「只看到坠毁的敌军部队烧成一团大火，尸体根本看不出是几岁。厉害的飞机变成散落四处的碎片，被压毁的民宅也是支离破碎，根本不知道死者是大人还是小孩，每个人都像木炭人偶般在地上滚来滚去。」看起来像是木炭人偶的物体不断干烧，淡淡的烟雾渗透到空气中，被烟雾迷蒙的双眼掉下眼泪。'},
        {"role": "user", "content": '今やかつての浮航都市(フローティング・シティ)群も土台となる岩礁に癒着されて動けない単なる辺境の孤島と化してしまっていた。'},
        
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)
    print('-'*100)
    print('tokenizer词表长度：', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")
    print('-'*100)
    print('压缩率测试（Chars/Tokens）：')
    test_texts = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",        
    ]
    
    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")
    
    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print('-'*100)
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练 MiniMind BPE tokenizer')
    parser.add_argument('--rate', type=float, default=0.3,
                        help='直接指定采样率并跳过第一遍统计（未指定时按 --target-chars 自动反推）')
    parser.add_argument('--target-chars', type=int, default=TARGET_CHARS,
                        help=f'目标总字符数（默认 {TARGET_CHARS}），仅在未指定 --rate 时使用')
    parser.add_argument('--max-seq-chars', type=int, default=MAX_SEQ_CHARS,
                        help=f'单条文本截断长度（默认 {MAX_SEQ_CHARS}）：长文本命中后只取前 N 字符')
    args = parser.parse_args()
    #train_tokenizer(DATA_DIR, TOKENIZER_DIR, VOCAB_SIZE, args.target_chars, BATCH_SIZE,
    #                rate=args.rate, max_seq_chars=args.max_seq_chars)
    eval_tokenizer(TOKENIZER_DIR)
