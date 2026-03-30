"""Experiment 12: Micro-Anatomy of a v4 Read.

Where does the time go in read_columns()?

The v4 reader (2.9ms for 12.6MB = 4.3 GB/s) leaves ~40% of NVMe bandwidth
on the table. This experiment decomposes the read into individual operations
to identify exactly which operations are the bottleneck and what a compiled
(Rust/C++) reader should optimize.

Discovery: read_columns() opens the file TWICE — once in read_header()
for Block 0 + directory, then again for column data. This experiment also
tests a single-open alternative.

Questions:
  1. What fraction of read time is file I/O vs Python overhead?
  2. How much does the double file open cost?
  3. Is seek-per-column slower than one bulk read?
  4. What's the theoretical floor for a compiled reader?
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import v4 from fintek
sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.format import (
    BLOCK_SIZE, COL_DIR_ENTRY_SIZE, DTYPE_TO_NUMPY,
    MKTFHeader, UpstreamFingerprint,
    unpack_block0, unpack_column_entry,
)
from trunk.backends.mktf.writer import write_mktf as write_v4
from trunk.backends.mktf.reader import read_columns as read_columns_v4, read_header as read_header_v4

# Import v3 source data loader
sys.path.insert(0, str(Path("R:/winrapids/research/20260327-mktf-format")))
from mktf_v3 import AAPL_PATH, COL_MAP, CONDITION_BITS


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_source_columns() -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(AAPL_PATH))
    raw = {new: tbl.column(old).to_numpy() for old, new in COL_MAP.items()}
    n = len(raw["price"])
    cols = {}
    cols["price"] = raw["price"].astype(np.float32)
    cols["size"] = raw["size"].astype(np.float32)
    cols["timestamp"] = raw["timestamp"].astype(np.int64)
    cols["exchange"] = raw["exchange"].astype(np.uint8)
    bitmasks = np.zeros(n, dtype=np.uint32)
    for i, s in enumerate(raw["conditions"]):
        if s and isinstance(s, str):
            for code in s.split(","):
                code = code.strip()
                if code:
                    bit = CONDITION_BITS.get(int(code))
                    if bit is not None:
                        bitmasks[i] |= 1 << bit
    cols["conditions"] = bitmasks
    return cols


def timed_ns(fn, n_runs: int) -> dict:
    gc.disable()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e3)  # us
    gc.enable()
    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.median(arr)),
    }


def fmt_us(stats: dict) -> str:
    return f"{stats['mean']:.1f}us ±{stats['std']:.1f}"


def fmt_ms(stats: dict) -> str:
    return f"{stats['mean']/1000:.3f}ms ±{stats['std']/1000:.3f}"


# ═══════════════════════════════════════════════════════════════
# SINGLE-OPEN READ (avoids the double file open)
# ═══════════════════════════════════════════════════════════════

def read_columns_single_open(path: str) -> tuple[MKTFHeader, dict[str, np.ndarray]]:
    """Read header + all columns in a single file open."""
    with open(path, "rb") as f:
        # Block 0
        block0 = f.read(BLOCK_SIZE)
        header = unpack_block0(block0)

        # Column directory
        if header.dir_entries > 0 and header.dir_offset > 0:
            f.seek(header.dir_offset)
            dir_bytes = f.read(header.dir_entries * COL_DIR_ENTRY_SIZE)
            for i in range(header.dir_entries):
                entry = unpack_column_entry(dir_bytes, i * COL_DIR_ENTRY_SIZE)
                header.columns.append(entry)

        # Column data (same file handle, no re-open)
        columns: dict[str, np.ndarray] = {}
        for entry in header.columns:
            np_dtype = DTYPE_TO_NUMPY[entry.dtype_code]
            f.seek(entry.data_offset)
            raw = f.read(entry.data_nbytes)
            columns[entry.name] = np.frombuffer(raw, dtype=np_dtype)

    return header, columns


def read_columns_bulk(path: str) -> tuple[MKTFHeader, dict[str, np.ndarray]]:
    """Read entire file in one shot, slice columns from buffer."""
    with open(path, "rb") as f:
        data = f.read()

    header = unpack_block0(data[:BLOCK_SIZE])

    if header.dir_entries > 0 and header.dir_offset > 0:
        for i in range(header.dir_entries):
            off = header.dir_offset + i * COL_DIR_ENTRY_SIZE
            entry = unpack_column_entry(data, off)
            header.columns.append(entry)

    columns: dict[str, np.ndarray] = {}
    for entry in header.columns:
        np_dtype = DTYPE_TO_NUMPY[entry.dtype_code]
        raw = data[entry.data_offset:entry.data_offset + entry.data_nbytes]
        columns[entry.name] = np.frombuffer(raw, dtype=np_dtype)

    return header, columns


def read_columns_mmap(path: str) -> tuple[MKTFHeader, dict[str, np.ndarray]]:
    """Memory-map the file, parse header from buffer, slice columns."""
    import mmap
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    header = unpack_block0(mm[:BLOCK_SIZE])

    if header.dir_entries > 0 and header.dir_offset > 0:
        for i in range(header.dir_entries):
            off = header.dir_offset + i * COL_DIR_ENTRY_SIZE
            entry = unpack_column_entry(mm, off)
            header.columns.append(entry)

    columns: dict[str, np.ndarray] = {}
    for entry in header.columns:
        np_dtype = DTYPE_TO_NUMPY[entry.dtype_code]
        columns[entry.name] = np.frombuffer(
            mm, dtype=np_dtype, count=entry.n_elements, offset=entry.data_offset
        )

    # Keep mm alive — numpy arrays reference it
    columns["_mmap"] = mm  # type: ignore
    return header, columns


# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

def main():
    N = 50  # runs per micro-benchmark

    print("=" * 78)
    print("EXPERIMENT 12: MICRO-ANATOMY OF A v4 READ")
    print("=" * 78)
    print(f"Runs per benchmark: {N}")
    print()

    cols = load_source_columns()
    n_rows = len(cols["price"])
    data_bytes = sum(arr.nbytes for arr in cols.values())

    tmpdir = tempfile.mkdtemp(prefix="mktf_anatomy_")
    v4_path = os.path.join(tmpdir, "test.mktf")
    write_v4(v4_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
             upstream=[UpstreamFingerprint(leaf_id="ingest", write_ts_ns=time.time_ns())])

    file_size = os.path.getsize(v4_path)
    print(f"Data: {n_rows} rows, {len(cols)} columns, {data_bytes/1e6:.2f} MB")
    print(f"File: {file_size/1e6:.2f} MB")
    print()

    # Pre-read to ensure warm cache
    for _ in range(5):
        _ = read_columns_v4(v4_path)

    # ── Phase 1: Decompose the Production Read ──────────────────
    print("PHASE 1: DECOMPOSE PRODUCTION read_columns()")
    print("-" * 60)
    print("Current code: read_header() [open→read→close] + open→seek×N→read×N→close")
    print()

    # 1a: File open + close (empty)
    stats_open_close = timed_ns(lambda: open(v4_path, "rb").close(), N)
    print(f"  file open+close:        {fmt_us(stats_open_close)}")

    # 1b: File open + read Block 0 (4096 bytes) + close
    def read_block0_only():
        with open(v4_path, "rb") as f:
            _ = f.read(BLOCK_SIZE)
    stats_block0_io = timed_ns(read_block0_only, N)
    print(f"  open + read(4096) + close: {fmt_us(stats_block0_io)}")

    # 1c: unpack_block0 (pure CPU, no I/O)
    with open(v4_path, "rb") as f:
        block0_bytes = f.read(BLOCK_SIZE)
    stats_unpack = timed_ns(lambda: unpack_block0(block0_bytes), N)
    print(f"  unpack_block0 (CPU):    {fmt_us(stats_unpack)}")

    # 1d: Directory read (seek + read, file already open)
    header = read_header_v4(v4_path)
    dir_size = header.dir_entries * COL_DIR_ENTRY_SIZE

    def read_dir_only():
        with open(v4_path, "rb") as f:
            f.seek(header.dir_offset)
            _ = f.read(dir_size)
    stats_dir_io = timed_ns(read_dir_only, N)
    print(f"  open + seek + read({dir_size}) dir + close: {fmt_us(stats_dir_io)}")

    # 1e: Directory unpack (pure CPU)
    with open(v4_path, "rb") as f:
        f.seek(header.dir_offset)
        dir_bytes = f.read(dir_size)
    def unpack_dir():
        for i in range(header.dir_entries):
            unpack_column_entry(dir_bytes, i * COL_DIR_ENTRY_SIZE)
    stats_dir_unpack = timed_ns(unpack_dir, N)
    print(f"  unpack directory (CPU):  {fmt_us(stats_dir_unpack)}")

    # 1f: Per-column seek + read (file already open with header parsed)
    def read_data_only():
        with open(v4_path, "rb") as f:
            for entry in header.columns:
                f.seek(entry.data_offset)
                _ = f.read(entry.data_nbytes)
    stats_data_io = timed_ns(read_data_only, N)
    print(f"  open + 5×(seek+read) data + close: {fmt_us(stats_data_io)}")

    # 1g: np.frombuffer only (from pre-read buffers)
    raw_bufs = {}
    with open(v4_path, "rb") as f:
        for entry in header.columns:
            f.seek(entry.data_offset)
            raw_bufs[entry.name] = (f.read(entry.data_nbytes), DTYPE_TO_NUMPY[entry.dtype_code])
    def frombuffer_only():
        for name, (raw, dt) in raw_bufs.items():
            _ = np.frombuffer(raw, dtype=dt)
    stats_frombuf = timed_ns(frombuffer_only, N)
    print(f"  5× np.frombuffer (CPU): {fmt_us(stats_frombuf)}")

    # 1h: Full production read (baseline)
    stats_full = timed_ns(lambda: read_columns_v4(v4_path), N)
    print(f"  ---")
    print(f"  FULL read_columns():    {fmt_us(stats_full)}")

    # Sum components
    # Production path: open1→read(4096)→unpack→seek→read(dir)→unpack_dir→close1
    #                 + open2→N×(seek+read)→N×frombuffer→close2
    header_phase = stats_block0_io["mean"] + stats_unpack["mean"] + stats_dir_unpack["mean"]
    # directory I/O is part of the first open, but we measure it with its own open
    # Actually we need to account for the EXTRA open in read_header
    data_phase = stats_data_io["mean"] + stats_frombuf["mean"]
    estimated_total = header_phase + data_phase
    overhead = stats_full["mean"] - estimated_total

    print()
    print(f"  Component sum estimate: {estimated_total:.0f}us")
    print(f"  Measured total:         {stats_full['mean']:.0f}us")
    print(f"  Unaccounted (Python/dict/overhead): {overhead:.0f}us ({overhead/stats_full['mean']*100:.0f}%)")
    print()

    # Breakdown pie
    total = stats_full["mean"]
    print(f"  TIME BUDGET (% of {total:.0f}us):")
    print(f"    I/O (block0+dir+data):  {stats_block0_io['mean'] + stats_data_io['mean']:.0f}us"
          f"  ({(stats_block0_io['mean'] + stats_data_io['mean'])/total*100:.0f}%)")
    print(f"    CPU (unpack+frombuf):   {stats_unpack['mean'] + stats_dir_unpack['mean'] + stats_frombuf['mean']:.0f}us"
          f"  ({(stats_unpack['mean'] + stats_dir_unpack['mean'] + stats_frombuf['mean'])/total*100:.0f}%)")
    print(f"    2nd file open penalty:  ~{stats_open_close['mean']:.0f}us"
          f"  ({stats_open_close['mean']/total*100:.0f}%)")
    print(f"    Unaccounted:            {overhead:.0f}us ({overhead/total*100:.0f}%)")
    bw = file_size / (total / 1e6) / 1e9
    bw_io_only = file_size / ((stats_block0_io['mean'] + stats_data_io['mean']) / 1e6) / 1e9
    print(f"  Effective BW: {bw:.2f} GB/s (total), {bw_io_only:.2f} GB/s (I/O only)")
    print()

    # ── Phase 2: Read Strategy Comparison ──────────────────────
    print("PHASE 2: READ STRATEGY COMPARISON")
    print("-" * 60)

    # Warmup all strategies
    _ = read_columns_v4(v4_path)
    _ = read_columns_single_open(v4_path)
    _ = read_columns_bulk(v4_path)
    _ = read_columns_mmap(v4_path)

    stats_prod = timed_ns(lambda: read_columns_v4(v4_path), N)
    stats_single = timed_ns(lambda: read_columns_single_open(v4_path), N)
    stats_bulk = timed_ns(lambda: read_columns_bulk(v4_path), N)
    stats_mmap = timed_ns(lambda: read_columns_mmap(v4_path), N)

    strategies = [
        ("Production (2 opens, seek/col)", stats_prod),
        ("Single open (1 open, seek/col)", stats_single),
        ("Bulk read (1 open, 1 read)",     stats_bulk),
        ("Memory-mapped (mmap + slice)",   stats_mmap),
    ]

    print(f"  {'Strategy':<40} {'Mean':>10} {'p50':>10} {'Min':>10} {'Speedup':>10}")
    print(f"  {'-'*80}")
    baseline = stats_prod["mean"]
    for name, s in strategies:
        speedup = baseline / s["mean"]
        bw = file_size / (s["mean"] / 1e6) / 1e9
        print(f"  {name:<40} {fmt_us(s):>18} {s['p50']:>8.0f}us {s['min']:>8.0f}us {speedup:>8.2f}x  {bw:.1f} GB/s")

    print()

    # ── Phase 3: I/O Pattern Analysis ──────────────────────────
    print("PHASE 3: I/O PATTERN ANALYSIS")
    print("-" * 60)

    # What's the cost of seek vs sequential?
    # Read all column data sequentially (one big read from data_start)
    total_data_bytes = sum(e.data_nbytes for e in header.columns)
    def read_sequential():
        with open(v4_path, "rb") as f:
            f.seek(header.data_start)
            _ = f.read(total_data_bytes + 4096 * len(header.columns))  # overread to include padding
    stats_seq = timed_ns(read_sequential, N)

    def read_seek_per_col():
        with open(v4_path, "rb") as f:
            for entry in header.columns:
                f.seek(entry.data_offset)
                _ = f.read(entry.data_nbytes)
    stats_seek = timed_ns(read_seek_per_col, N)

    # What about reading the entire file?
    def read_entire_file():
        with open(v4_path, "rb") as f:
            _ = f.read()
    stats_entire = timed_ns(read_entire_file, N)

    print(f"  seek-per-column (5 seeks):  {fmt_us(stats_seek)}")
    print(f"  sequential from data_start: {fmt_us(stats_seq)}")
    print(f"  entire file (1 read):       {fmt_us(stats_entire)}")
    print(f"  seek overhead: {stats_seek['mean'] - stats_seq['mean']:.0f}us "
          f"({(stats_seek['mean']/stats_seq['mean'] - 1)*100:.0f}% slower)")
    print()

    # ── Phase 4: What Would a Compiled Reader Look Like? ────────
    print("PHASE 4: COMPILED READER FLOOR ESTIMATE")
    print("-" * 60)

    # Theoretical: single read + zero-copy cast
    # The minimum is: open + read(entire) + close
    # A Rust reader using read_exact or mmap would eliminate all Python overhead
    io_floor = stats_entire["mean"]
    cpu_overhead = stats_full["mean"] - io_floor
    bw_floor = file_size / (io_floor / 1e6) / 1e9

    print(f"  I/O floor (single read):    {io_floor:.0f}us = {io_floor/1000:.3f}ms")
    print(f"  Python overhead above I/O:  {cpu_overhead:.0f}us ({cpu_overhead/stats_full['mean']*100:.0f}%)")
    print(f"  I/O bandwidth at floor:     {bw_floor:.2f} GB/s")
    print(f"  NVMe theoretical max:       ~7.0 GB/s")
    print(f"  I/O floor / NVMe:           {bw_floor/7.0*100:.0f}%")
    print()

    # What a Rust reader gains
    rust_estimate = io_floor * 1.05  # 5% overhead for struct copy + pointer setup
    bw_rust = file_size / (rust_estimate / 1e6) / 1e9
    speedup_rust = stats_full["mean"] / rust_estimate

    print(f"  Estimated Rust reader:      {rust_estimate:.0f}us = {rust_estimate/1000:.3f}ms")
    print(f"  Rust speedup vs Python:     {speedup_rust:.2f}x")
    print(f"  Rust BW:                    {bw_rust:.2f} GB/s")
    print()

    # Universe projections
    n_tickers = 4604
    print(f"  UNIVERSE PROJECTIONS ({n_tickers} tickers):")
    print(f"    Python read_columns:  {stats_full['mean']*n_tickers/1e6:.1f}s")
    print(f"    Single-open Python:   {stats_single['mean']*n_tickers/1e6:.1f}s")
    print(f"    Bulk-read Python:     {stats_bulk['mean']*n_tickers/1e6:.1f}s")
    print(f"    Mmap Python:          {stats_mmap['mean']*n_tickers/1e6:.1f}s")
    print(f"    Estimated Rust:       {rust_estimate*n_tickers/1e6:.1f}s")
    print(f"    I/O floor:            {io_floor*n_tickers/1e6:.1f}s")
    print()

    # ── Phase 5: unpack_block0 Detailed Breakdown ──────────────
    print("PHASE 5: HEADER UNPACK COST BREAKDOWN")
    print("-" * 60)

    # How much of unpack_block0 is struct.unpack_from vs dataclass creation?
    def just_struct_unpack():
        struct.unpack_from("<4s H I H H", block0_bytes, 0)
        struct.unpack_from("<32s 16s 10s B B 16s 16s I", block0_bytes, 16)
        struct.unpack_from("<B B B B B B B B B B B B I B B B", block0_bytes, 128)
        struct.unpack_from("<Q H I I I Q Q", block0_bytes, 176)
        struct.unpack_from("<q q q q q", block0_bytes, 240)
        struct.unpack_from("<Q Q Q I d d Q", block0_bytes, 288)
        struct.unpack_from("<q I I I I Q 16s I I", block0_bytes, 368)
        struct.unpack_from("<Q Q Q Q Q", block0_bytes, 448)
        struct.unpack_from("<H B B I I q", block0_bytes, 512)
        struct.unpack_from("<d d d d d d d d", block0_bytes, 1600)
        struct.unpack_from("<I I", block0_bytes, 1728)
        for j in range(8):
            struct.unpack_from("<d", block0_bytes, 1728 + 8 + j * 8)

    stats_raw_struct = timed_ns(just_struct_unpack, N)
    stats_full_unpack = timed_ns(lambda: unpack_block0(block0_bytes), N)
    dataclass_overhead = stats_full_unpack["mean"] - stats_raw_struct["mean"]

    print(f"  Raw struct.unpack_from (×12):  {fmt_us(stats_raw_struct)}")
    print(f"  Full unpack_block0:            {fmt_us(stats_full_unpack)}")
    print(f"  Dataclass + string decode:     {dataclass_overhead:.1f}us "
          f"({dataclass_overhead/stats_full_unpack['mean']*100:.0f}%)")
    print()

    # ── Phase 6: Verify Correctness ────────────────────────────
    print("PHASE 6: CORRECTNESS VERIFICATION")
    print("-" * 60)

    # Verify all strategies produce identical results
    _, cols_prod = read_columns_v4(v4_path)
    _, cols_single = read_columns_single_open(v4_path)
    _, cols_bulk = read_columns_bulk(v4_path)
    h_mmap, cols_mmap = read_columns_mmap(v4_path)

    all_ok = True
    for name in cols_prod:
        for label, other in [("single", cols_single), ("bulk", cols_bulk), ("mmap", cols_mmap)]:
            if name.startswith("_"):
                continue
            if name not in other:
                print(f"  FAIL: {label} missing column {name}")
                all_ok = False
                continue
            if not np.array_equal(cols_prod[name], other[name]):
                print(f"  FAIL: {label}.{name} differs from production")
                all_ok = False

    if all_ok:
        print(f"  All 4 strategies produce identical results for all {len(cols_prod)} columns.")
    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
