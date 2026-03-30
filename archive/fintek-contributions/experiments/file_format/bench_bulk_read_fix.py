"""Experiment 21: Bulk Read Fix — single-read + buffer-slicing vs production reader.

Experiment 20 found bulk reads are 46x slower than single-file warm reads.
Root cause: read_columns() opens file twice + 25 per-column seeks.

This experiment tests a simple fix: read the entire file into a buffer with
one f.read(), then parse header + extract columns from the buffer. No file
modification — the fast reader is standalone in this benchmark.

Expected: 10-30x improvement for small KO05 files (14.9 KB at align=64).
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_columns
from trunk.backends.mktf.format import (
    BLOCK_SIZE, ENTRY_SIZE, DTYPE_TO_NUMPY, unpack_block0, unpack_byte_entry,
)


def fast_read_columns(path: str | Path) -> tuple:
    """Single-read fast path: one f.read() + buffer slicing."""
    with open(path, "rb") as f:
        buf = f.read()  # One syscall, entire file

    # Parse header from buffer
    header = unpack_block0(buf[:BLOCK_SIZE])

    # Parse directory from buffer
    if header.dir_entries > 0 and header.dir_offset > 0:
        for i in range(header.dir_entries):
            offset = header.dir_offset + i * ENTRY_SIZE
            entry = unpack_byte_entry(buf, offset)
            header.columns.append(entry)

    # Extract columns from buffer (no seeks, no additional I/O)
    columns = {}
    for entry in header.columns:
        np_dtype = DTYPE_TO_NUMPY[entry.dtype_code]
        columns[entry.name] = np.frombuffer(
            buf, dtype=np_dtype,
            offset=entry.data_offset,
            count=entry.n_elements,
        ).copy()  # copy() to decouple from buffer

    return header, columns


def make_ko05_columns(n_bins: int, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    stat_names = ["sum", "sum_sq", "min", "max", "count"]
    data_names = ["price", "size", "timestamp", "exchange", "conditions"]
    cols = {}
    for dname in data_names:
        for sname in stat_names:
            cols[f"{dname}_{sname}"] = rng.normal(100, 10, n_bins).astype(np.float32)
    return cols


def main():
    print("=" * 78)
    print("EXPERIMENT 21: BULK READ FIX (single-read + buffer-slicing)")
    print("=" * 78)
    print()

    n_bins = 78
    alignment = 64
    N_FILES = 2000
    BATCH_SIZE = 100

    # Create test files
    tmpdir = Path(os.environ.get("TEMP", "C:/Temp")) / "mktf_readfix_test"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)

    print(f"Config: {N_FILES} files, {n_bins} bins, alignment={alignment}")
    print("Creating test files...", end=" ", flush=True)

    for i in range(N_FILES):
        cols = make_ko05_columns(n_bins, seed=i)
        fpath = tmpdir / f"T{i:04d}.KO05.mktf"
        write_mktf(fpath, cols, leaf_id=f"T{i:04d}.K02.KO05",
                   safe=False, alignment=alignment)
    print("done.")

    file_list = sorted(tmpdir.glob("T*.KO05.mktf"))
    print(f"Files created: {len(file_list)}, size: {file_list[0].stat().st_size / 1024:.1f} KB each")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Correctness — verify fast reader matches production
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: CORRECTNESS")
    print("-" * 60)

    test_file = file_list[0]
    h_prod, cols_prod = read_columns(test_file)
    h_fast, cols_fast = fast_read_columns(test_file)

    all_match = True
    for name in cols_prod:
        if name not in cols_fast:
            print(f"  MISSING in fast reader: {name}")
            all_match = False
        elif not np.array_equal(cols_prod[name], cols_fast[name]):
            print(f"  MISMATCH: {name}")
            all_match = False

    if all_match and len(cols_prod) == len(cols_fast):
        print(f"  All {len(cols_prod)} columns match. Fast reader is correct.")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Single-file warm comparison (baseline)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: SINGLE-FILE WARM (same file, 30 runs)")
    print("-" * 60)

    N_WARM = 30
    for label, read_fn in [("production", read_columns), ("fast", fast_read_columns)]:
        # Warm
        for _ in range(5):
            read_fn(test_file)

        gc.disable()
        times = []
        for _ in range(N_WARM):
            t0 = time.perf_counter_ns()
            read_fn(test_file)
            times.append((time.perf_counter_ns() - t0) / 1e3)
        gc.enable()

        arr = np.array(times)
        print(f"  {label:>12}: {np.mean(arr):.0f}us ±{np.std(arr):.0f} "
              f"(min={np.min(arr):.0f}, p50={np.median(arr):.0f})")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Bulk read comparison (the real test)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: BULK READ (2000 unique files)")
    print("-" * 60)

    for label, read_fn in [("production", read_columns), ("fast", fast_read_columns)]:
        # Light warm (first 5 files)
        for f in file_list[:5]:
            read_fn(f)

        batch_times = []
        gc.disable()

        idx = 0
        while idx < N_FILES:
            batch_end = min(idx + BATCH_SIZE, N_FILES)
            batch_n = batch_end - idx

            t0 = time.perf_counter_ns()
            for i in range(idx, batch_end):
                read_fn(file_list[i])
            elapsed_us = (time.perf_counter_ns() - t0) / 1e3
            per_file = elapsed_us / batch_n
            batch_times.append((idx, elapsed_us, per_file))
            idx = batch_end

        gc.enable()

        total_us = sum(t[1] for t in batch_times)
        overall_per_file = total_us / N_FILES

        print(f"  {label}:")
        # Show first, middle, last batches
        for bt in [batch_times[0], batch_times[len(batch_times)//2], batch_times[-1]]:
            print(f"    batch {bt[0]:>5}: {bt[2]/1000:.3f}ms/file")
        print(f"    TOTAL: {total_us/1e6:.2f}s, {overall_per_file:.0f}us/file")
        print()

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Universe projections with fix
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: UNIVERSE PROJECTIONS")
    print("-" * 60)

    # Re-measure both for clean numbers
    results = {}
    for label, read_fn in [("production", read_columns), ("fast", fast_read_columns)]:
        gc.disable()
        t0 = time.perf_counter_ns()
        for f in file_list:
            read_fn(f)
        total_us = (time.perf_counter_ns() - t0) / 1e3
        gc.enable()
        per_file = total_us / N_FILES
        results[label] = per_file

    speedup = results["production"] / results["fast"]
    print(f"  Production: {results['production']:.0f}us/file")
    print(f"  Fast:       {results['fast']:.0f}us/file")
    print(f"  Speedup:    {speedup:.1f}x")
    print()

    tickers = 4604
    for n_cad in [8, 10]:
        total = tickers * n_cad
        prod_s = total * results["production"] / 1e6
        fast_s = total * results["fast"] / 1e6
        print(f"  {n_cad} cadences ({total:,} files):")
        print(f"    Production: {prod_s:.1f}s")
        print(f"    Fast:       {fast_s:.1f}s")
    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
