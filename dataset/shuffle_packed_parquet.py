#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shuffle_packed_parquet.py

把 pack_pretrain_parquet.py 生成的 packed parquet 数据集离线打散，输出仍为
PackedPretrainDataset 可直接读取的 schema:

    input_ids: fixed_size_list<int32>[max_length]

为什么不直接全量读内存:
  packed 数据通常几十到上百 GB，无法一次性载入内存。脚本采用外部 shuffle:

  1. 顺序读取原始 part-*.parquet，把每行随机分配到临时 bucket
  2. 每个 bucket 是全局随机样本子集，可单独载入内存
  3. 逐 bucket 内部完全 shuffle 后写入最终输出 parquet

这样训练时可以顺序读取打散后的输出目录，最大化 row group cache 命中，
同时排除原始 pack 顺序导致的 dataset/domain 成块问题。

注意:
  - 非删除模式峰值约为 原始 + 临时 bucket + 最终输出，通常接近 3x
  - 若磁盘只允许 2.0~2.3x，使用 --delete-input-after-bucketize:
      phase1 完成后临时 bucket 已经是全量数据副本，脚本会删除原输入目录，
      再写最终输出。峰值通常约为 max(原始+临时, 临时+输出) ~= 2x
  - 输出文件使用 zstd level 9
"""

import argparse
import math
import os
import re
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


STATE_FILE = "_shuffle_state.json"


def natural_key(path: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def find_part_files(data_dir: str):
    files = []
    for root, _dirs, names in os.walk(data_dir):
        for name in names:
            if re.fullmatch(r"part-\d{5}\.parquet", name):
                files.append(os.path.join(root, name))
    files.sort(key=natural_key)
    return files


def dir_size(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def is_relative_to(path: str, parent: str) -> bool:
    path = os.path.abspath(path)
    parent = os.path.abspath(parent)
    try:
        common = os.path.commonpath([path, parent])
    except ValueError:
        return False
    return common == parent


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


def inspect_dataset(files):
    if not files:
        raise FileNotFoundError("输入目录下没有找到 part-xxxxx.parquet")

    total_rows = 0
    max_length = None
    schema = None
    row_groups = 0

    for path in files:
        pf = pq.ParquetFile(path)
        if "input_ids" not in pf.schema_arrow.names:
            raise ValueError(f"{path} 缺少 input_ids 列")
        field = pf.schema_arrow.field("input_ids")
        t = field.type
        if not pa.types.is_fixed_size_list(t):
            raise ValueError(f"{path}: input_ids 必须是 fixed_size_list, 实际 {t}")
        if max_length is None:
            max_length = t.list_size
            schema = pa.schema([("input_ids", pa.list_(pa.int32(), max_length))])
        elif t.list_size != max_length:
            raise ValueError(f"{path}: max_length 不一致, 期望 {max_length}, 实际 {t.list_size}")
        total_rows += pf.metadata.num_rows
        row_groups += pf.metadata.num_row_groups

    return total_rows, max_length, schema, row_groups


def fixed_list_to_numpy(array, max_length: int):
    """FixedSizeListArray / ChunkedArray -> np.ndarray [rows, max_length] int32."""
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    return values.reshape(-1, max_length)


def rows_to_table(rows: np.ndarray, max_length: int):
    rows = np.asarray(rows, dtype=np.int32)
    values = rows.reshape(-1)
    arr = pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.int32()), max_length)
    return pa.table({"input_ids": arr})


class OutputWriter:
    def __init__(self, out_dir: str, schema, max_length: int, rows_per_rg: int,
                 max_file_size: int, compression_level: int = 9):
        self.out_dir = out_dir
        self.schema = schema
        self.max_length = max_length
        self.rows_per_rg = rows_per_rg
        self.max_file_size = max_file_size
        self.compression_level = compression_level
        self.file_idx = 0
        self.writer = None
        self.path = None
        self.buffer = []
        self.buffer_rows = 0
        self.total_rows = 0

    def _open(self):
        if self.writer is None:
            self.path = os.path.join(self.out_dir, f"part-{self.file_idx:05d}.parquet")
            self.writer = pq.ParquetWriter(
                self.path, self.schema, compression="zstd",
                compression_level=self.compression_level)

    def _close_current(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.file_idx += 1

    def _write_rows(self, rows: np.ndarray):
        if len(rows) == 0:
            return
        self._open()
        self.writer.write_table(rows_to_table(rows, self.max_length))
        self.total_rows += len(rows)
        if os.path.getsize(self.path) >= self.max_file_size:
            self._close_current()

    def write(self, rows: np.ndarray):
        if len(rows) == 0:
            return
        self.buffer.append(rows)
        self.buffer_rows += len(rows)
        while self.buffer_rows >= self.rows_per_rg:
            merged = np.concatenate(self.buffer, axis=0)
            emit = merged[:self.rows_per_rg]
            rest = merged[self.rows_per_rg:]
            self._write_rows(emit)
            self.buffer = [rest] if len(rest) else []
            self.buffer_rows = len(rest)

    def close(self):
        if self.buffer_rows:
            merged = np.concatenate(self.buffer, axis=0)
            self._write_rows(merged)
            self.buffer = []
            self.buffer_rows = 0
        self._close_current()


class BucketWriter:
    def __init__(self, temp_dir: str, schema, max_length: int, buffer_rows: int,
                 compression_level: int):
        self.temp_dir = temp_dir
        self.schema = schema
        self.max_length = max_length
        self.buffer_rows = buffer_rows
        self.compression_level = compression_level
        self.buffers = {}
        self.counts = {}
        self.part_counts = {}
        self.rows_written = {}

    def add(self, bucket: int, rows: np.ndarray):
        if len(rows) == 0:
            return
        self.buffers.setdefault(bucket, []).append(rows.copy())
        self.counts[bucket] = self.counts.get(bucket, 0) + len(rows)
        if self.counts[bucket] >= self.buffer_rows:
            self.flush_bucket(bucket)

    def flush_bucket(self, bucket: int):
        count = self.counts.get(bucket, 0)
        if count == 0:
            return
        rows = np.concatenate(self.buffers[bucket], axis=0)
        bucket_dir = os.path.join(self.temp_dir, f"bucket-{bucket:05d}")
        os.makedirs(bucket_dir, exist_ok=True)
        part_idx = self.part_counts.get(bucket, 0)
        path = os.path.join(bucket_dir, f"part-{part_idx:05d}.parquet")
        pq.write_table(
            rows_to_table(rows, self.max_length),
            path,
            compression="zstd",
            compression_level=self.compression_level,
        )
        self.part_counts[bucket] = part_idx + 1
        self.rows_written[bucket] = self.rows_written.get(bucket, 0) + len(rows)
        self.buffers[bucket] = []
        self.counts[bucket] = 0

    def close(self):
        for bucket in list(self.buffers.keys()):
            self.flush_bucket(bucket)


def make_bucket_plan(total_rows: int, max_length: int, bucket_target_bytes: int,
                     num_buckets: int):
    row_bytes = max_length * 4
    if num_buckets <= 0:
        num_buckets = max(1, math.ceil(total_rows * row_bytes / bucket_target_bytes))
    bucket_target_rows = max(1, math.ceil(total_rows / num_buckets))
    return num_buckets, bucket_target_rows


def parse_args():
    p = argparse.ArgumentParser(description="Shuffle packed parquet dataset with external buckets")
    p.add_argument("--input-dir", required=True, help="pack_pretrain_parquet.py 输出目录")
    p.add_argument("--output-dir", required=True, help="打散后的输出目录")
    p.add_argument("--temp-dir", default=None, help="临时 bucket 目录 (默认 output-dir + .shuffle_tmp)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--row-group-size", default="64M", help="最终输出 row group 未压缩目标大小 (默认 64M)")
    p.add_argument("--max-file-size", default="4GiB", help="最终输出单 parquet 文件压缩后大小上限")
    p.add_argument("--bucket-target-size", default="512M",
                   help="单个临时 bucket 目标未压缩大小; 越大随机性相同但 phase2 内存越高 (默认 512M)")
    p.add_argument("--num-buckets", type=int, default=0,
                   help="临时 bucket 数量 (默认按 bucket-target-size 自动计算)")
    p.add_argument("--read-batch-rows", type=int, default=4096,
                   help="读取输入 parquet 的 batch 行数 (默认 4096)")
    p.add_argument("--bucket-buffer-size", default="16M",
                   help="每个 bucket 的内存缓冲未压缩目标大小 (默认 16M)")
    p.add_argument("--temp-compression-level", type=int, default=9,
                   help="临时 bucket parquet 的 zstd 压缩级别 (默认 9, 用于控制磁盘峰值)")
    p.add_argument("--output-compression-level", type=int, default=9,
                   help="最终输出 parquet 的 zstd 压缩级别 (默认 9)")
    p.add_argument("--max-disk-multiplier", type=float, default=2.3,
                   help="相对输入目录物理大小的峰值磁盘倍率保护 (默认 2.3; <=0 表示关闭检查)")
    p.add_argument("--delete-input-after-bucketize", action="store_true",
                   help="危险: phase1 完成并校验行数后删除 input-dir, 再写最终输出, 以把磁盘峰值控制在约 2x")
    p.add_argument("--overwrite", action="store_true",
                   help="允许删除已有 output-dir/temp-dir")
    return p.parse_args()


def prepare_dirs(out_dir: str, temp_dir: str, overwrite: bool):
    for path in (out_dir, temp_dir):
        if os.path.exists(path):
            if not overwrite:
                raise FileExistsError(f"目录已存在: {path} (需要覆盖请加 --overwrite)")
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def phase1_bucketize(files, temp_dir, schema, max_length, num_buckets,
                     bucket_buffer_rows, read_batch_rows, seed, temp_compression_level):
    print("\n[1/2] 随机分桶写临时 parquet ...")
    rng = np.random.default_rng(seed)
    writer = BucketWriter(temp_dir, schema, max_length, bucket_buffer_rows, temp_compression_level)
    total = 0
    for path in tqdm(files, desc="bucketize files", unit="file"):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=read_batch_rows, columns=["input_ids"]):
            rows = fixed_list_to_numpy(batch.column("input_ids"), max_length)
            buckets = rng.integers(0, num_buckets, size=len(rows), dtype=np.int32)
            for bucket in np.unique(buckets):
                writer.add(int(bucket), rows[buckets == bucket])
            total += len(rows)
    writer.close()
    print(f"  分桶完成: {total:,} 行, {num_buckets} buckets")
    return total


def phase2_write_output(temp_dir, out_dir, schema, max_length, num_buckets,
                        rows_per_rg, max_file_size, seed, output_compression_level):
    print("\n[2/2] 逐 bucket 内部 shuffle 后写最终 parquet ...")
    out = OutputWriter(out_dir, schema, max_length, rows_per_rg, max_file_size, output_compression_level)
    bucket_order = np.arange(num_buckets, dtype=np.int32)
    np.random.default_rng(seed + 1).shuffle(bucket_order)

    for bucket in tqdm(bucket_order, desc="write buckets", unit="bucket"):
        bucket_dir = os.path.join(temp_dir, f"bucket-{int(bucket):05d}")
        if not os.path.isdir(bucket_dir):
            continue
        parts = [os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir)
                 if f.endswith(".parquet")]
        parts.sort(key=natural_key)
        if not parts:
            continue

        chunks = []
        for part in parts:
            pf = pq.ParquetFile(part)
            for rg in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg, columns=["input_ids"])
                chunks.append(fixed_list_to_numpy(table.column("input_ids"), max_length))
        rows = np.concatenate(chunks, axis=0)
        order = np.random.default_rng(seed + 1009 + int(bucket)).permutation(len(rows))
        rows = rows[order]

        for start in range(0, len(rows), rows_per_rg):
            out.write(rows[start:start + rows_per_rg])

    out.close()
    print(f"  输出完成: {out.total_rows:,} 行, {out.file_idx} 个 parquet 文件")
    return out.total_rows


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    temp_dir = os.path.abspath(args.temp_dir or (output_dir + ".shuffle_tmp"))
    if output_dir == input_dir:
        raise ValueError("output-dir 不能与 input-dir 相同")
    if temp_dir == input_dir:
        raise ValueError("temp-dir 不能与 input-dir 相同")
    if is_relative_to(output_dir, input_dir) or is_relative_to(temp_dir, input_dir):
        raise ValueError("output-dir/temp-dir 不能放在 input-dir 内部, 否则删除输入时会误删输出或临时数据")

    files = find_part_files(input_dir)
    total_rows, max_length, schema, row_groups = inspect_dataset(files)
    input_bytes = dir_size(input_dir)
    original_input_bytes = input_bytes
    row_group_bytes = parse_size(args.row_group_size)
    max_file_size = parse_size(args.max_file_size)
    bucket_target_bytes = parse_size(args.bucket_target_size)
    bucket_buffer_bytes = parse_size(args.bucket_buffer_size)

    rows_per_rg = max(1, row_group_bytes // (max_length * 4))
    bucket_buffer_rows = max(1, bucket_buffer_bytes // (max_length * 4))
    num_buckets, bucket_target_rows = make_bucket_plan(
        total_rows, max_length, bucket_target_bytes, args.num_buckets)

    print(f"输入文件: {len(files)} 个, row groups: {row_groups:,}")
    print(f"输入行数: {total_rows:,}, max_length={max_length}")
    print(f"输出 row group: ~{row_group_bytes/2**20:.0f} MB = {rows_per_rg:,} 行")
    print(f"bucket 数: {num_buckets}, 目标每 bucket ~{bucket_target_rows:,} 行 "
          f"(~{bucket_target_rows * max_length * 4 / 2**20:.0f} MB 未压缩)")
    print(f"临时目录: {temp_dir}")
    print(f"输出目录: {output_dir}")
    print(f"输入物理大小: {input_bytes/2**30:.2f} GiB")
    if args.delete_input_after_bucketize:
        print("磁盘模式: phase1 后删除 input-dir, 控制峰值约 2x (中断后临时 bucket 是唯一副本)")
    else:
        print("磁盘模式: 保留 input-dir, 峰值通常接近 3x")

    prepare_dirs(output_dir, temp_dir, args.overwrite)
    t0 = time.time()
    n1 = phase1_bucketize(
        files, temp_dir, schema, max_length, num_buckets,
        bucket_buffer_rows, args.read_batch_rows, args.seed, args.temp_compression_level)
    if n1 != total_rows:
        raise RuntimeError(f"phase1 行数不一致: input={total_rows}, bucketized={n1}")

    temp_bytes = dir_size(temp_dir)
    phase1_peak = input_bytes + temp_bytes
    print(f"  phase1 后临时 bucket 大小: {temp_bytes/2**30:.2f} GiB, "
          f"当前峰值约 {phase1_peak / max(input_bytes, 1):.2f}x")
    if args.max_disk_multiplier > 0 and phase1_peak > input_bytes * args.max_disk_multiplier:
        raise RuntimeError(
            f"phase1 磁盘峰值 {phase1_peak / max(input_bytes, 1):.2f}x "
            f"超过限制 {args.max_disk_multiplier:.2f}x。可减小输入、提高压缩、或增大限制。"
        )

    if args.delete_input_after_bucketize:
        print(f"  删除输入目录以释放空间: {input_dir}")
        shutil.rmtree(input_dir)
        input_bytes = 0
    elif args.max_disk_multiplier > 0:
        projected_peak = input_bytes + temp_bytes + max(input_bytes, temp_bytes)
        if projected_peak > (input_bytes * args.max_disk_multiplier):
            raise RuntimeError(
                f"保留 input-dir 时预计峰值约 {projected_peak / max(input_bytes, 1):.2f}x，"
                f"超过限制 {args.max_disk_multiplier:.2f}x。"
                "请加 --delete-input-after-bucketize 或调大 --max-disk-multiplier。"
            )

    n2 = phase2_write_output(
        temp_dir, output_dir, schema, max_length, num_buckets,
        rows_per_rg, max_file_size, args.seed, args.output_compression_level)

    if n1 != total_rows or n2 != total_rows:
        raise RuntimeError(f"行数不一致: input={total_rows}, bucketized={n1}, output={n2}")

    output_bytes = dir_size(output_dir)
    phase2_peak = input_bytes + temp_bytes + output_bytes
    print(f"  输出物理大小: {output_bytes/2**30:.2f} GiB, "
          f"phase2 峰值约 {phase2_peak / max(original_input_bytes, 1):.2f}x")
    if args.max_disk_multiplier > 0 and phase2_peak > original_input_bytes * args.max_disk_multiplier:
        raise RuntimeError(
            f"phase2 磁盘峰值 {phase2_peak / max(original_input_bytes, 1):.2f}x "
            f"超过限制 {args.max_disk_multiplier:.2f}x。"
        )

    shutil.rmtree(temp_dir)
    elapsed = time.time() - t0
    print(f"\n完成! 用时 {elapsed/60:.1f} 分钟")
    print(f"输出: {output_dir}")


if __name__ == "__main__":
    main()
