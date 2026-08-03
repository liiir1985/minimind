#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_large_rowgroups.py — 把"存在超大 row group"的 parquet 重写为小 row group

为什么需要:
  pyarrow 的 iter_batches 对非字典编码的列, 会一次性解压整个 row group 的
  列 chunk 再按 batch_size 切片返回 —— batch_size 只控制返回粒度, 不控制
  解压内存。单 row group 的大文本文件 (如 HF 下载的 305 行整本书合集,
  Harem_CN.parquet text 列未压缩 1.8GB) 读取即吃 1.6GB+ 内存。
  重写为 ~64MB/row group 后, 逐组流式解压, 读取内存降至 1/8。

判断规则 (对每个 parquet 文件):
  - 任一 row group 未压缩字节 > --threshold → 重写该文件 (拆成 ~--max-rg-size 的 RG)
  - 所有 row group ≤ --threshold   → 跳过 (符合规则的不动)
  - 重写只拆 row group, 不改变文件内容与总大小; threshold 是"拆不拆"的门槛,
    --max-rg-size 是拆的目标粒度

重写方式 (字节驱动, 不做平均行大小换算):
  - iter_batches(batch_size=128) 流式读, 逐批累积字节
  - 单批 nbytes 未超预算 → 攒批; 单批本身就超预算 (行超大) → 按行拆
  - 累积 ≥ 预算才 write_table 一次 (一个 row group), 预算唯一决定 RG 大小

用法:
  python dataset/split_large_rowgroups.py --input-dir /data/pretrain \
      --output-dir /data/pretrain_rg [--threshold 256M] [--max-rg-size 64M]

参数:
  --input-dir       输入目录, 递归扫描所有 .parquet (必须)
  --output-dir      输出目录, 保持相对路径 (必须)
  --threshold       文件处理门槛: 所有 RG 未压缩总字节 ≤ 此值直接跳过 (默认 256M)
  --max-rg-size     重写目标: 单个 row group 未压缩字节上限 (默认 64M, 支持 K/M/G 后缀)
  --compression     重写压缩算法 (默认 zstd, 中间文件不需要极高压缩)
  --compression-level 压缩级别 (默认 3)
"""

import argparse
import os
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


def natural_key(path: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def find_parquet_files(input_dir: str):
    files = []
    for root, _dirs, fnames in os.walk(input_dir):
        for fn in fnames:
            if fn.lower().endswith(".parquet"):
                files.append(os.path.join(root, fn))
    files.sort(key=natural_key)
    return files


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


def row_group_uncompressed(pf, rg_idx: int) -> int:
    """单个 row group 所有列的未压缩字节总和"""
    rg = pf.metadata.row_group(rg_idx)
    return sum(rg.column(c).total_uncompressed_size for c in range(rg.num_columns))


def rewrite_byte_driven(src_path: str, dst_path: str, rg_bytes: int,
                        compression: str, compression_level: int):
    """字节驱动流式重写: 累积到字节预算才写一个 row group。

    返回 (原行数, 新 row group 数, 用时秒)
    """
    src = pq.ParquetFile(src_path)
    nrows = src.metadata.num_rows
    t0 = time.time()
    writer = pq.ParquetWriter(
        dst_path, src.schema_arrow, compression=compression,
        compression_level=compression_level)
    buffer = []
    current_bytes = 0
    total_rows = 0

    def flush():
        nonlocal buffer, current_bytes
        if buffer:
            writer.write_table(pa.Table.from_batches(buffer))
            buffer = []
            current_bytes = 0

    for batch in src.iter_batches(batch_size=128):
        total_rows += batch.num_rows
        if batch.nbytes < rg_bytes:
            # 常规: 攒批直到字节预算
            buffer.append(batch)
            current_bytes += batch.nbytes
            if current_bytes >= rg_bytes:
                flush()
        else:
            # 单批本身就超预算 (行超大): 按行拆, 预算仍唯一决定 RG 大小
            for i in range(batch.num_rows):
                buffer.append(batch.slice(i, 1))
                current_bytes += batch.slice(i, 1).nbytes
                if current_bytes >= rg_bytes:
                    flush()
    flush()
    writer.close()
    out = pq.ParquetFile(dst_path)
    return total_rows, out.metadata.num_row_groups, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="把存在超大 row group 的 parquet 重写为小 row group")
    ap.add_argument("--input-dir", required=True, help="输入目录, 递归扫描所有 .parquet")
    ap.add_argument("--output-dir", required=True, help="输出目录 (保持相对路径; 符合规则的文件跳过不复制)")
    ap.add_argument("--threshold", type=str, default="256M",
                    help="重写门槛: 单个 row group 未压缩字节 > 此值才重写 (默认 256M)")
    ap.add_argument("--max-rg-size", type=str, default="64M",
                    help="重写目标: 单个 row group 未压缩字节上限 (默认 64M)")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--compression-level", type=int, default=3)
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"输入目录不存在: {args.input_dir}")
    threshold = parse_size(args.threshold)
    rg_bytes = parse_size(args.max_rg_size)
    os.makedirs(args.output_dir, exist_ok=True)

    files = find_parquet_files(args.input_dir)
    print(f"找到 {len(files)} 个 parquet 文件, 门槛 {threshold/2**20:.0f} MB, "
          f"目标 {rg_bytes/2**20:.0f} MB/row group")

    rewritten, skipped = 0, 0
    for fpath in files:
        base = os.path.basename(fpath)
        try:
            pf = pq.ParquetFile(fpath)
        except Exception as e:
            print(f"  [跳过] {base}: 无法打开 ({e})")
            skipped += 1
            continue
        nrows = pf.metadata.num_rows
        nrg = pf.metadata.num_row_groups
        rg_sizes = [row_group_uncompressed(pf, i) for i in range(nrg)]
        max_rg = max(rg_sizes) if rg_sizes else 0

        if max_rg <= threshold:
            print(f"  [跳过] {base}: {nrows} 行, {nrg} RG, 最大 {max_rg/2**20:.0f} MB (<= 门槛 {threshold/2**20:.0f} MB)")
            skipped += 1
            continue

        rel = os.path.relpath(fpath, args.input_dir)
        dst = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        dst_tmp = dst + ".rgtmp.parquet"

        print(f"  [重写] {base}: {nrows} 行, {nrg} RG (最大 {max_rg/2**20:.0f} MB) -> 目标 ~{rg_bytes/2**20:.0f} MB/RG")
        try:
            n, new_nrg, elapsed = rewrite_byte_driven(
                fpath, dst_tmp, rg_bytes, args.compression, args.compression_level)
        except Exception as e:
            print(f"    [失败] {e}")
            if os.path.exists(dst_tmp):
                os.remove(dst_tmp)
            skipped += 1
            continue
        os.replace(dst_tmp, dst)
        print(f"    完成: {n} 行 -> {new_nrg} RG, {elapsed:.0f}s")
        rewritten += 1

    print(f"\n完成: 重写 {rewritten} 个, 跳过 {skipped} 个")
    print("符合规则 (RG ≤ 阈值) 的文件未复制, output-dir 只含重写后的文件。")


if __name__ == "__main__":
    main()
