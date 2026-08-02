#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_pretrain_parquet.py

递归扫描输入目录下所有 .parquet 文件（含子目录，按自然排序顺序处理），
使用 dataset/lm_dataset.py 中 PretrainDataset 的 packing 方法
（bos+eos 包裹、超长按 max_length stride 切块、长度降序 FFD +
Bucketed Best-Fit 定长 bin 填充）生成 packed 数据集，
输出到另一目录，按逻辑数据量（行数 x max_length x 4B）约 max-file-size
一个 parquet 文件切分。

输出 schema: 每行一条 packed sample, 单列 input_ids (list<int32>, 长度 = max_length,
pad 位置为 tokenizer.pad_token_id)。labels 可在加载时低成本重构:
    labels = input_ids.clone(); labels[input_ids == pad_token_id] = -100

内存控制（避免全量加载）:
  - pyarrow iter_batches 流式读取输入 parquet
  - tokenize 后的序列累积在待打包队列，token 总量达 --mem-budget 才触发块内打包
  - 活跃 bin 池跨块复用（未满 bin 保留继续接收新序列），近似全局 Best-Fit 填充率
  - 装满（剩余 < drop_threshold）的 bin 立即出池进写入缓冲
  - bin 池上限 --max-open-bins，超限时优先释放剩余空间最小的 bin
  - 写入缓冲攒够 --flush-rows 行写一个 row group

峰值内存 ≈ mem_budget + flush_rows*max_length*4 + max_open_bins*max_length*4 + tokenizer
64GB 机器默认参数下约 2GB 以内。

依赖: transformers, pyarrow, numpy（不需要 torch / datasets）
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time

# 强制离线: worker 子进程 from_pretrained 时不得访问 HF hub (网络超时会拖垮启动)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------- utils ----

def natural_key(path: str):
    """自然排序 key: shard_2.parquet < shard_10.parquet"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def find_parquet_files(input_dir: str):
    files = []
    for root, _dirs, fnames in os.walk(input_dir):
        for fn in fnames:
            if fn.lower().endswith(".parquet"):
                files.append(os.path.join(root, fn))
    files.sort(key=natural_key)
    return files


# ---------------------------------------------------------------- tokenize workers ----

_WORKER_TOK = None  # worker 进程内的全局 tokenizer


def _init_worker(tokenizer_dir: str):
    global _WORKER_TOK
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_dir)


def _tokenize_batch(texts: list):
    """worker 进程执行: 批量 tokenize, 返回 input_ids 列表 (无 special tokens)"""
    return _WORKER_TOK(texts, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------- packer ----

class BinPacker:
    """
    与原 PretrainDataset 完全一致的 Bucketed Best-Fit 打包器,
    但支持"未满 bin 跨批复用" + 池上限控制, 以支持流式/分块处理。
    """

    def __init__(self, max_length: int, pad_token: int, max_open_bins: int):
        self.max_length = max_length
        self.pad_token = pad_token
        self.max_open_bins = max_open_bins
        self.bins_by_capacity = [[] for _ in range(max_length + 1)]  # 剩余容量 -> bin 索引
        self.bins = []        # bin 索引 -> np.ndarray (uint32, max_length), 已出池为 None
        self.bin_sums = []    # bin 索引 -> 当前已用长度, 已出池为 None
        self.finalized = []   # 已出池、待写入的 bin 数组
        self.drop_threshold = 100
        self.next_idx = 0

    def set_drop_threshold(self, min_seq_length: int):
        """与 lm_dataset.py 一致: drop_threshold = max(100, 最短序列长度)"""
        self.drop_threshold = max(100, min_seq_length)

    def pack(self, tokens: np.ndarray, length: int):
        placed = False
        for c in range(length, self.max_length + 1):
            if self.bins_by_capacity[c]:
                bin_idx = self.bins_by_capacity[c].pop()
                bin_arr = self.bins[bin_idx]
                start = self.bin_sums[bin_idx]
                bin_arr[start:start + length] = tokens
                used = start + length
                self.bin_sums[bin_idx] = used
                placed = True
                new_c = self.max_length - used
                if new_c >= self.drop_threshold:
                    self.bins_by_capacity[new_c].append(bin_idx)
                else:
                    self._finalize(bin_idx)
                break

        if not placed:
            new_bin = np.full(self.max_length, self.pad_token, dtype=np.uint32)
            new_bin[0:length] = tokens
            self.bins.append(new_bin)
            self.bin_sums.append(length)
            bin_idx = self.next_idx
            self.next_idx += 1
            new_c = self.max_length - length
            if new_c >= self.drop_threshold:
                self.bins_by_capacity[new_c].append(bin_idx)
            else:
                self._finalize(bin_idx)

        self._trim_pool()

    def _finalize(self, bin_idx: int):
        self.finalized.append(self.bins[bin_idx])
        self.bins[bin_idx] = None
        self.bin_sums[bin_idx] = None

    def _trim_pool(self):
        """池超限时从剩余空间最小的桶开始释放, 直到降到 80% 水位"""
        active = sum(len(b) for b in self.bins_by_capacity)
        if active <= self.max_open_bins:
            return
        target = int(self.max_open_bins * 0.8)
        for c in range(self.max_length + 1):
            if active <= target:
                break
            while self.bins_by_capacity[c] and active > target:
                self._finalize(self.bins_by_capacity[c].pop())
                active -= 1

    def flush_all(self):
        for bucket in self.bins_by_capacity:
            while bucket:
                self._finalize(bucket.pop())


# ---------------------------------------------------------------- writer ----

class OutputWriter:
    """按逻辑数据量切分输出 parquet 文件, 每个 ≤ max_file_size"""

    def __init__(self, out_dir: str, max_length: int, max_file_size: int, start_idx: int = 0):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.max_length = max_length
        self.max_file_size = max_file_size
        self.file_idx = start_idx
        self.logical_bytes = 0
        self.writer = None
        self.schema = pa.schema([("input_ids", pa.list_(pa.int32(), max_length))])
        self.row_bytes = max_length * 4  # 单列 int32

    def write(self, bins: list):
        """bins: list[np.ndarray (max_length,) uint32]"""
        if not bins:
            return
        if self.writer is None:
            path = os.path.join(self.out_dir, f"part-{self.file_idx:05d}.parquet")
            self.writer = pq.ParquetWriter(path, self.schema, compression="zstd", compression_level=9)

        values = np.stack(bins).astype(np.int32).reshape(-1)
        arr = pa.FixedSizeListArray.from_arrays(pa.array(values), self.max_length)
        self.writer.write_table(pa.table({"input_ids": arr}))

        self.logical_bytes += len(bins) * self.row_bytes
        if self.logical_bytes >= self.max_file_size:
            self.close_current()
            self.file_idx += 1
            self.logical_bytes = 0

    def close_current(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self):
        self.close_current()


# ---------------------------------------------------------------- main ----

def parse_args():
    p = argparse.ArgumentParser(description="Pretrain parquet -> packed parquet (流式, 低内存)")
    p.add_argument("--input-dir", required=True, help="输入目录, 递归扫描所有 .parquet")
    p.add_argument("--output-dir", required=True, help="输出目录")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help=f"tokenizer 目录 (默认 {DEFAULT_TOKENIZER})")
    p.add_argument("--max-length", type=int, default=2000, help="packed 序列长度 (默认 2000, 同 train_pretrain)")
    p.add_argument("--text-column", default="text", help="输入 parquet 的文本列名 (默认 text)")
    p.add_argument("--max-file-size", type=str, default="4GiB", help="单输出文件逻辑大小上限 (默认 4GiB)")
    p.add_argument("--mem-budget", type=str, default="200M", help="待打包序列 token 总量预算 (默认 200M token)")
    p.add_argument("--max-open-bins", type=int, default=4096, help="活跃 bin 池上限 (默认 4096)")
    p.add_argument("--flush-rows", type=int, default=100000, help="写入缓冲行数, 攒够写一个 row group (默认 100000)")
    p.add_argument("--batch-size", type=int, default=8192, help="读取输入 parquet 的 batch 行数 (默认 8192)")
    p.add_argument("--num-workers", type=int, default=1, help="tokenize 并行进程数 (默认 1。本机单核, 多进程无收益反而更慢; 多核机器可调大)")
    p.add_argument("--resume", action="store_true", help="断点续跑 (输出目录 state.json)")
    return p.parse_args()


def parse_size(s: str) -> int:
    s = s.strip().upper()
    m = re.fullmatch(r"(\d+)([KMGT]?I?B?)", s)
    if not m:
        raise ValueError(f"无法解析大小: {s}")
    val, unit = int(m.group(1)), m.group(2)
    mult = {"": 1, "B": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40,
            "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
            "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    if unit not in mult:
        raise ValueError(f"未知单位: {unit}")
    return val * mult[unit]


STATE_FILE = "state.json"

# tokenizer 默认取 minimind/model/ (脚本所在目录的上一级), 不依赖 cwd
DEFAULT_TOKENIZER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"
)


def load_state(out_dir: str):
    path = os.path.join(out_dir, STATE_FILE)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done_inputs": [], "closed_files": 0}


def save_state(out_dir: str, state: dict):
    tmp = os.path.join(out_dir, STATE_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, os.path.join(out_dir, STATE_FILE))


def cleanup_incomplete(out_dir: str, closed_files: int):
    """删除中断时未完成 close 的 part 文件 (index >= closed_files)"""
    if not os.path.isdir(out_dir):
        return
    for fn in os.listdir(out_dir):
        m = re.fullmatch(r"part-(\d{5})\.parquet", fn)
        if m and int(m.group(1)) >= closed_files:
            os.remove(os.path.join(out_dir, fn))
            print(f"  清理未完成文件: {fn}")


def main():
    args = parse_args()
    max_file_size = parse_size(args.max_file_size)
    mem_budget = parse_size(args.mem_budget)

    if not os.path.isdir(args.input_dir):
        sys.exit(f"输入目录不存在: {args.input_dir}")

    print("加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else 1
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 2
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    print(f"  bos={bos_id} eos={eos_id} pad={pad_id} vocab={tok.vocab_size}")

    files = find_parquet_files(args.input_dir)
    if not files:
        sys.exit(f"输入目录下没有找到 .parquet 文件: {args.input_dir}")
    print(f"找到 {len(files)} 个 parquet 文件")

    state = load_state(args.output_dir)
    if args.resume and state["done_inputs"]:
        done = set(state["done_inputs"])
        skip = [f for f in files if f in done]
        files = [f for f in files if f not in done]
        print(f"断点续跑: 跳过 {len(skip)} 个已处理文件, 剩余 {len(files)} 个")
        if files:
            cleanup_incomplete(args.output_dir, state["closed_files"])
    else:
        state = {"done_inputs": [], "closed_files": 0}
        # 非 resume 模式下清空输出目录的旧 part 文件, 避免混入
        cleanup_incomplete(args.output_dir, 0)

    packer = BinPacker(args.max_length, pad_id, args.max_open_bins)
    writer = OutputWriter(args.output_dir, args.max_length, max_file_size,
                          start_idx=state["closed_files"])

    # 多进程 tokenize 池 (BPE 受 GIL 限制, 必须用进程而非线程)
    pool = None
    if args.num_workers > 1:
        print(f"启动 {args.num_workers} 个 tokenize worker 进程 ...")
        pool = mp.Pool(args.num_workers, initializer=_init_worker, initargs=(args.tokenizer,))

    pending = []          # list[np.ndarray (n,) uint32]
    pending_tokens = 0
    total_texts = 0
    total_raw_tokens = 0
    total_packed = 0
    t_start = time.time()

    def flush_pending():
        nonlocal pending, pending_tokens
        if not pending:
            return
        # 块内按长度降序 (FFD), 与原版全局排序语义一致
        order = sorted(range(len(pending)), key=lambda i: -len(pending[i]))
        min_len = len(pending[order[-1]]) if order else 0
        packer.set_drop_threshold(min_len)
        for i in order:
            arr = pending[i]
            packer.pack(arr, len(arr))
        pending = []
        pending_tokens = 0

    def flush_writer_buf():
        nonlocal total_packed
        if len(packer.finalized) >= args.flush_rows:
            writer.write(packer.finalized)
            total_packed += len(packer.finalized)
            packer.finalized.clear()

    def process_ids(ids):
        """单条文本的 input_ids -> 加 bos/eos -> 切块 -> 进待打包队列"""
        nonlocal pending, pending_tokens, total_texts, total_raw_tokens
        toks = [bos_id] + ids + [eos_id]
        n = len(toks)
        total_raw_tokens += n
        total_texts += 1
        if n > args.max_length:
            # 与原版一致: stride = max_length 切块, 每块独立参与打包
            for i in range(0, n, args.max_length):
                chunk = toks[i:i + args.max_length]
                arr = np.asarray(chunk, dtype=np.uint32)
                pending.append(arr)
                pending_tokens += len(arr)
                if pending_tokens >= mem_budget:
                    flush_pending()
                    flush_writer_buf()
        else:
            arr = np.asarray(toks, dtype=np.uint32)
            pending.append(arr)
            pending_tokens += n
            if pending_tokens >= mem_budget:
                flush_pending()
                flush_writer_buf()

    print("开始处理 ...")
    for fi, fpath in enumerate(files, 1):
        pf = pq.ParquetFile(fpath)
        nrows = pf.metadata.num_rows
        nbatches = max(1, (nrows + args.batch_size - 1) // args.batch_size)
        fname = os.path.relpath(fpath, args.input_dir)
        pbar = tqdm(total=nrows, desc=f"[{fi}/{len(files)}] {fname}", unit="row",
                    leave=False, dynamic_ncols=True)

        if pool is not None:
            # 并行路径: 主进程读 parquet, worker 池 tokenize (imap 保持顺序、流水线)
            batch_iter = (
                [str(t) for t in b.column(args.text_column).to_pylist()]
                for b in pf.iter_batches(batch_size=args.batch_size, columns=[args.text_column])
            )
            for ids_list in pool.imap(_tokenize_batch, batch_iter, chunksize=1):
                for ids in ids_list:
                    process_ids(ids)
                pbar.update(len(ids_list))
        else:
            for batch in pf.iter_batches(batch_size=args.batch_size, columns=[args.text_column]):
                texts = [str(t) for t in batch.column(args.text_column).to_pylist()]
                enc = tok(texts, add_special_tokens=False)
                for ids in enc["input_ids"]:
                    process_ids(ids)
                pbar.update(len(texts))
        pbar.close()

        # 该文件处理完: 清空待打包队列, 强制把 finalized 缓冲落盘, 更新 state
        # (part 文件边界只由逻辑大小决定, 与输入文件边界无关, writer 保持打开)
        flush_pending()
        if packer.finalized:
            writer.write(packer.finalized)
            total_packed += len(packer.finalized)
            packer.finalized.clear()
        state["done_inputs"].append(fpath)
        state["closed_files"] = writer.file_idx
        save_state(args.output_dir, state)

    # 收尾: 池中所有 bin 落盘
    flush_pending()
    packer.flush_all()
    writer.write(packer.finalized)
    total_packed += len(packer.finalized)
    packer.finalized.clear()
    writer.close()

    if pool is not None:
        pool.close()
        pool.join()

    state["closed_files"] = writer.file_idx
    save_state(args.output_dir, state)

    elapsed = time.time() - t_start
    # 填充率统计: 非 pad token 占比
    out_files = [f for f in os.listdir(args.output_dir) if re.fullmatch(r"part-\d{5}\.parquet", f)]
    print(f"\n完成! 用时 {elapsed/60:.1f} 分钟")
    print(f"  输入文本: {total_texts:,} 条, 原始 token: {total_raw_tokens:,}")
    print(f"  输出 packed samples: {total_packed:,}  (每条 {args.max_length} token)")
    print(f"  输出文件: {len(out_files)} 个 ({args.output_dir})")
    for f in sorted(out_files):
        fp = os.path.join(args.output_dir, f)
        print(f"    {f}: {os.path.getsize(fp)/2**30:.2f} GiB (物理)")


if __name__ == "__main__":
    main()
