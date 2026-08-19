from torch.utils.data import Dataset
import torch
import torch.distributed as dist
import json
import os
import re
import bisect
import random
from datasets import load_dataset, Features, Sequence, Value
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_CACHE"] = "e:/myexe/minimind/dataset/hf_cache"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        cache_path = data_path + f".packed_{max_length}_vocab{len(tokenizer)}.npy"
        
        import torch.distributed as dist
        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        
        if rank == 0:
            if not os.path.exists(cache_path):
                print(f"Building packed dataset to {cache_path}...")
                raw_samples = load_dataset('json', data_files=data_path, split='train')
                
                def tokenize_fn(examples):
                    texts = [str(t) for t in examples['text']]
                    batch_encoded = tokenizer(texts, add_special_tokens=False)
                    batch_tokens = []
                    batch_lengths = []
                    
                    for tokens in batch_encoded.input_ids:
                        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
                        if len(tokens) > max_length:
                            stride = max_length
                            for i in range(0, len(tokens), stride):
                                chunk = tokens[i:i + max_length]
                                batch_tokens.append(chunk)
                                batch_lengths.append(len(chunk))
                        else:
                            batch_tokens.append(tokens)
                            batch_lengths.append(len(tokens))
                    return {'tokens': batch_tokens, 'length': batch_lengths}
                
                print("Tokenizing and chunking dataset...")
                tokenized_dataset = raw_samples.map(
                    tokenize_fn, 
                    batched=True, 
                    batch_size=1000, 
                    remove_columns=raw_samples.column_names,
                    desc="Tokenizing dataset"
                )
                
                print("Sorting sequences for optimal packing (First-Fit Decreasing)...")
                lengths_and_indices = [(length, i) for i, length in enumerate(tokenized_dataset['length'])]
                lengths_and_indices.sort(key=lambda x: x[0], reverse=True)
                
                print("Packing sequences into bins (Bucketed Best-Fit) with Numpy Memmap...")
                from tqdm import tqdm
                bins = [] 
                bin_sums = [] 
                pad_token = tokenizer.pad_token_id
                
                # 为极大提升打包速度，当bin的剩余空间小于100时不再继续匹配
                min_seq_length = lengths_and_indices[-1][0] if lengths_and_indices else 0
                drop_threshold = max(100, min_seq_length)
                
                # 按剩余容量将 bin 分桶，实现 O(N) 极速匹配
                bins_by_capacity = [[] for _ in range(max_length + 1)]
                
                for length, idx in tqdm(lengths_and_indices, desc="Packing sequences"):
                    tokens = tokenized_dataset[idx]['tokens']
                    placed = False
                    
                    # 寻找能装下该长度的最佳 bin (容量从刚好等于 length 开始往上找)
                    for c in range(length, max_length + 1):
                        if bins_by_capacity[c]:
                            bin_idx = bins_by_capacity[c].pop()
                            
                            start = bin_sums[bin_idx]
                            bins[bin_idx][start:start+length] = tokens
                            bin_sums[bin_idx] += length
                            placed = True
                            
                            new_c = c - length
                            if new_c >= drop_threshold:
                                bins_by_capacity[new_c].append(bin_idx)
                            break
                            
                    if not placed:
                        # 新开一个 bin，直接用 numpy 数组
                        new_bin = np.full(max_length, pad_token, dtype=np.uint32)
                        new_bin[0:length] = tokens
                        bins.append(new_bin)
                        bin_sums.append(length)
                        
                        new_c = max_length - length
                        if new_c >= drop_threshold:
                            bins_by_capacity[new_c].append(len(bins) - 1)
                        
                print(f"Packed {len(tokenized_dataset)} sequences into {len(bins)} bins.")
                bins_np = np.stack(bins)
                np.save(cache_path, bins_np)
                print(f"Saved packed dataset to {cache_path}")
        
        if is_dist:
            dist.barrier()
            
        print(f"Loading packed dataset from {cache_path}")
        self.samples = np.load(cache_path, mmap_mode='r')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        packed_ids = self.samples[index]
        input_ids = torch.tensor(packed_ids.astype(np.int64), dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class PackedPretrainDataset(Dataset):
    """读取 pack_pretrain_parquet.py 生成的 packed parquet 目录

    - 只在初始化时读每个 part-*.parquet 的 metadata (行数 + schema),
      不加载 input_ids 数据本体
    - __getitem__ 时按 (file_idx, row_group_idx) 惰性读取 row group,
      读进来的 block 缓存以服务同 rg 内后续 __getitem__ (顺序 / 按 rg 聚簇
      shuffle 时命中率 ~100%; 全局 shuffle 时命中率 ~0, 可将
      cache_row_groups 设为 0 关掉)
    - schema 校验 input_ids 是 fixed_size_list<int32>[max_length],
      与传入的 max_length 一致
    """

    _PART_RE = re.compile(r"part-(\d{5})\.parquet$")

    def __init__(self, data_dir, tokenizer, max_length, cache_row_groups=2):
        super().__init__()
        import pyarrow.parquet as pq

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.cache_row_groups = cache_row_groups

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"packed parquet 目录不存在: {data_dir}")

        files = []
        for fn in os.listdir(data_dir):
            m = self._PART_RE.search(fn)
            if m:
                files.append((int(m.group(1)), os.path.join(data_dir, fn)))
        if not files:
            raise FileNotFoundError(f"目录下没有 part-*.parquet: {data_dir}")
        files.sort()
        self.files = [p for _, p in files]

        # 只读元数据 (num_rows / schema / row_group 行数), 不加载 input_ids
        self.file_rows = []            # 每个文件的总行数
        self.file_rg_cumrows = []      # 每个文件: [rg_i 之前累计行数], 用于二分查找 rg
        self.file_num_row_groups = []
        for fp in self.files:
            pf = pq.ParquetFile(fp)
            schema = pf.schema_arrow
            if "input_ids" not in schema.names:
                raise ValueError(f"{fp} 缺少 input_ids 列, 实际列: {schema.names}")
            field = schema.field("input_ids")
            t = field.type
            if not (pa_is_fixed_size_list(t) and t.list_size == max_length):
                raise ValueError(
                    f"{fp}: input_ids 类型不匹配, 期望 fixed_size_list<int32>[{max_length}], "
                    f"实际 {t}")
            nrg = pf.metadata.num_row_groups
            cum = [0] * (nrg + 1)
            for i in range(nrg):
                cum[i + 1] = cum[i] + pf.metadata.row_group(i).num_rows
            self.file_rows.append(cum[-1])
            self.file_rg_cumrows.append(cum)
            self.file_num_row_groups.append(nrg)

        # 全局行 -> 文件二分
        self.file_cumrows = [0]
        for n in self.file_rows:
            self.file_cumrows.append(self.file_cumrows[-1] + n)
        self.total_rows = self.file_cumrows[-1]

        # 惰性状态: ParquetFile 句柄 + row group 缓存 (每 worker 独立创建)
        # DataLoader num_workers > 0 时进程内独占, 无需锁
        self._pf_cache = {}            # file_idx -> ParquetFile
        self._rg_cache = {}            # (file_idx, rg_idx) -> np.ndarray[num_rows, max_length]
        self._rg_lru = []              # 顺序记录, 超过 cache_row_groups 时淘汰最早

    def __len__(self):
        return self.total_rows

    def rowgroup_shuffled_indices(self, seed, chunk_size=0, mix_rgs=1):
        """返回按 row_group 聚簇 shuffle 的全局 index 序列

        参数:
          chunk_size: 每个 rg 内进一步切成 chunk 的行数 (0 = 不切, 整 rg 作一个 chunk)
          mix_rgs:    同时交错的 rg 数量 (>=1); 训练时 cache_row_groups 必须 >= mix_rgs
                      否则 LRU 会不停淘汰导致重复解压

        算法:
          - rg 全局顺序打乱
          - 每 mix_rgs 个 rg 组成一个 band, band 内每个 rg 切成 chunk,
            所有 chunk 混合打乱, 但 chunk 内行保持连续 (=chunk 内也可再 shuffle)
          - band 之间按 shuffle 后的 rg 顺序处理

        效果:
          - domain 切换粒度 = chunk_size 行 (远小于 rg 大小)
          - cache 命中: 一个 band 内的 chunk 只在 mix_rgs 个 rg 之间跳,
            cache_row_groups >= mix_rgs 时命中率 100%

        seed 需与训练 epoch 关联 (通常 = base_seed + epoch)。
        """
        g = np.random.default_rng(seed)
        # 收集所有 (rg 全局起始, rg 长度)
        rg_spans = []
        for fi, cum in enumerate(self.file_rg_cumrows):
            base = self.file_cumrows[fi]
            for rg in range(len(cum) - 1):
                rg_spans.append((base + cum[rg], cum[rg + 1] - cum[rg]))
        g.shuffle(rg_spans)

        out = np.empty(self.total_rows, dtype=np.int64)
        pos = 0
        # 每个 band 处理 mix_rgs 个 rg
        for band_start in range(0, len(rg_spans), max(mix_rgs, 1)):
            band = rg_spans[band_start:band_start + max(mix_rgs, 1)]
            # 生成 band 内所有 chunk 的 (start, length)
            chunks = []
            for lo, n in band:
                if chunk_size <= 0:
                    chunks.append((lo, n))
                else:
                    for c_start in range(0, n, chunk_size):
                        chunks.append((lo + c_start, min(chunk_size, n - c_start)))
            g.shuffle(chunks)
            for c_lo, c_n in chunks:
                rows = np.arange(c_lo, c_lo + c_n, dtype=np.int64)
                g.shuffle(rows)
                out[pos:pos + c_n] = rows
                pos += c_n
        return out

    def _get_row_group(self, file_idx, rg_idx):
        key = (file_idx, rg_idx)
        cached = self._rg_cache.get(key)
        if cached is not None:
            return cached

        import pyarrow.parquet as pq
        pf = self._pf_cache.get(file_idx)
        if pf is None:
            pf = pq.ParquetFile(self.files[file_idx])
            self._pf_cache[file_idx] = pf

        table = pf.read_row_group(rg_idx, columns=["input_ids"])
        arr = table.column("input_ids")
        # ChunkedArray -> 单一 FixedSizeListArray; row_group 通常单 chunk,
        # combine_chunks 在此情况下零拷贝
        if hasattr(arr, "combine_chunks"):
            arr = arr.combine_chunks()
        values = arr.values.to_numpy(zero_copy_only=True)
        block = values.reshape(-1, self.max_length)

        self._rg_cache[key] = block
        self._rg_lru.append(key)
        while len(self._rg_lru) > self.cache_row_groups:
            old = self._rg_lru.pop(0)
            self._rg_cache.pop(old, None)
        return block

    def __getitem__(self, index):
        if index < 0:
            index += self.total_rows
        if index < 0 or index >= self.total_rows:
            raise IndexError(index)

        file_idx = bisect.bisect_right(self.file_cumrows, index) - 1
        local = index - self.file_cumrows[file_idx]
        cum = self.file_rg_cumrows[file_idx]
        rg_idx = bisect.bisect_right(cum, local) - 1
        row_in_rg = local - cum[rg_idx]

        block = self._get_row_group(file_idx, rg_idx)
        row = block[row_in_rg]

        input_ids = torch.from_numpy(row.astype(np.int64, copy=True))
        labels = input_ids.clone()
        if self.pad_token_id is not None:
            labels[input_ids == self.pad_token_id] = -100
        return input_ids, labels


def pa_is_fixed_size_list(t):
    import pyarrow as pa
    return pa.types.is_fixed_size_list(t)


class InMemoryTupleDataset(Dataset):
    """把固定验证样本常驻在 CPU Tensor 中，适用于返回 tuple[tensor, ...] 的 LM 数据集。"""

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("InMemoryTupleDataset 至少需要一个 tensor")
        first_len = tensors[0].size(0)
        if any(t.size(0) != first_len for t in tensors):
            raise ValueError("所有 tensor 的第 0 维长度必须一致")
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)


def build_in_memory_validation_dataset(dataset, val_ratio=0.001, val_seed=2024, logger=print):
    """从任意 tuple[tensor, ...] 数据集中固定抽样，构建常驻 CPU 内存的验证集。"""
    if val_ratio <= 0:
        return None, set()

    val_size = max(1, int(len(dataset) * val_ratio))
    rng = random.Random(val_seed)
    val_indices = rng.sample(range(len(dataset)), min(val_size, len(dataset)))
    val_index_set = set(val_indices)

    # packed parquet 按全局行号排序读取，避免抽验证集时频繁解压不同 row_group。
    load_indices = sorted(val_indices) if isinstance(dataset, PackedPretrainDataset) else val_indices
    logger(f'抽取验证集到内存: {len(load_indices):,} 条 ({val_ratio:.4%})')

    columns = None
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(val_seed)
    np.random.seed(val_seed)
    torch.manual_seed(val_seed)
    try:
        for idx in load_indices:
            sample = dataset[idx]
            if not isinstance(sample, tuple):
                sample = (sample,)
            if columns is None:
                columns = [[] for _ in sample]
            if len(sample) != len(columns):
                raise ValueError("数据集 __getitem__ 返回的字段数量不稳定")
            for col, value in zip(columns, sample):
                col.append(value)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    if columns is None:
        raise ValueError("无法从空数据集抽取验证集")

    tensors = [torch.stack(col).contiguous() for col in columns]
    g = torch.Generator().manual_seed(val_seed)
    perm = torch.randperm(tensors[0].size(0), generator=g)
    val_ds = InMemoryTupleDataset(*(t[perm] for t in tensors))
    logger(f'验证集已常驻CPU内存: {len(val_ds):,} 条')
    return val_ds, val_index_set


def filter_validation_indices(indices, val_index_set):
    if not val_index_set:
        return indices
    if isinstance(indices, np.ndarray):
        val_indices = np.fromiter(val_index_set, dtype=indices.dtype, count=len(val_index_set))
        return indices[~np.isin(indices, val_indices)]
    return [int(i) for i in indices if int(i) not in val_index_set]


def shard_dataset_indices(indices, batch_size):
    """DDP 下按 rank 切分 index 序列，并补齐到 global batch 的整数倍。"""
    if not dist.is_initialized():
        return indices

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    total = len(indices)
    if total == 0:
        return indices
    samples_per_global_batch = batch_size * world_size
    padded_total = ((total + samples_per_global_batch - 1) // samples_per_global_batch) * samples_per_global_batch
    if padded_total > total:
        pad = padded_total - total
        if isinstance(indices, np.ndarray):
            indices = np.concatenate([indices, indices[np.arange(pad) % total]])
        else:
            indices = list(indices)
            indices.extend(indices[i % total] for i in range(pad))
    return indices[rank:padded_total:world_size]


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, list_name='conversations', file_format='json'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.list_name = list_name
        features = Features({list_name: [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset(file_format, data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample[self.list_name])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
