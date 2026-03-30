"""Experiment 20: Bulk Write Scaling — does per-file overhead hold at N=1000+?

All prior experiments measured single-file writes (30 runs of the same file).
The pipeline model multiplies per-file time × file count and assumes linearity.
This experiment tests that assumption by writing many DISTINCT files in sequence
and checking whether per-file time stays constant or drifts.

Potential effects at scale:
  - NTFS metadata overhead (directory entries grow)
  - NVMe command queue depth / coalescing
  - OS file cache warming
  - Python GC pressure from accumulated objects
  - File handle churn

If per-file time is constant: pipeline model is validated.
If it degrades: model is optimistic.
If it improves: model is pessimistic.
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_columns


def make_ko05_columns(n_bins: int, seed: int = 42) -> dict[str, np.ndarray]:
    """25 stat columns: 5 data cols * 5 stats."""
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
    print("EXPERIMENT 20: BULK WRITE SCALING")
    print("=" * 78)
    print()

    n_bins = 78  # 5min cadence, representative
    alignment = 64  # KO05 production setting

    # Pre-generate column data for multiple "tickers" (different seeds = different data)
    # Use 2000 files — enough to see trends, ~half a cadence slice
    N_FILES = 2000
    BATCH_SIZE = 100  # measure per-file time in batches of 100

    print(f"Config: {N_FILES} files, {n_bins} bins, alignment={alignment}, safe=False")
    print(f"Batch size for timing: {BATCH_SIZE}")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Bulk write — track per-batch timing
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: BULK WRITE (2000 files into one directory)")
    print("-" * 60)

    tmpdir = Path(os.environ.get("TEMP", "C:/Temp")) / "mktf_bulk_test"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)

    # Pre-generate all column data to exclude generation time
    print("  Pre-generating column data...", end=" ", flush=True)
    all_cols = []
    for i in range(N_FILES):
        all_cols.append(make_ko05_columns(n_bins, seed=i))
    print("done.")
    print()

    # Warm the writer (3 throws)
    warm_path = tmpdir / "warm.mktf"
    for _ in range(3):
        write_mktf(warm_path, all_cols[0], leaf_id="X", safe=False, alignment=alignment)
    warm_path.unlink(missing_ok=True)

    # Write N_FILES, measuring each batch
    batch_times = []  # (batch_index, total_us, per_file_us)
    gc.disable()

    file_idx = 0
    while file_idx < N_FILES:
        batch_end = min(file_idx + BATCH_SIZE, N_FILES)
        batch_n = batch_end - file_idx

        t0 = time.perf_counter_ns()
        for i in range(file_idx, batch_end):
            fpath = tmpdir / f"T{i:04d}.KO05.mktf"
            write_mktf(fpath, all_cols[i], leaf_id=f"TICKER{i:04d}.K02.KO05",
                       ticker=f"T{i:04d}", day="2025-09-02",
                       safe=False, alignment=alignment)
        elapsed_us = (time.perf_counter_ns() - t0) / 1e3
        per_file = elapsed_us / batch_n

        batch_times.append((file_idx, elapsed_us, per_file))
        file_idx = batch_end

    gc.enable()

    # Report
    print(f"  {'Batch':>10} | {'Files':>8} | {'Batch Time':>12} | {'Per-File':>10} | {'Drift':>8}")
    print(f"  {'-----':>10} | {'-----':>8} | {'----------':>12} | {'--------':>10} | {'-----':>8}")

    baseline = batch_times[0][2]
    for start_idx, total_us, per_file_us in batch_times:
        drift = per_file_us / baseline
        print(f"  {start_idx:>5}-{start_idx + BATCH_SIZE - 1:<4} | {BATCH_SIZE:>8} | "
              f"{total_us / 1000:>9.2f}ms | {per_file_us / 1000:>7.3f}ms | {drift:>7.2f}x")

    # Overall
    total_time_us = sum(t[1] for t in batch_times)
    overall_per_file = total_time_us / N_FILES
    print()
    print(f"  Total: {total_time_us / 1e6:.2f}s for {N_FILES} files")
    print(f"  Overall per-file: {overall_per_file / 1000:.3f}ms")
    print(f"  Exp 19 single-file: ~1.38ms (align=64, 78 bins)")
    print(f"  Ratio (bulk/single): {overall_per_file / 1380:.2f}x")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Directory size effect — 1 dir vs many subdirs
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: DIRECTORY STRUCTURE (flat vs sharded)")
    print("-" * 60)

    # Flat: all 500 files in one directory
    # Sharded: 500 files across 10 subdirectories (50 each)
    N_PHASE2 = 500

    for label, shard_count in [("flat (1 dir)", 1), ("sharded (10 dirs)", 10)]:
        subdir = tmpdir / f"phase2_{label.split()[0]}"
        if subdir.exists():
            shutil.rmtree(subdir)

        if shard_count == 1:
            subdir.mkdir(parents=True)
        else:
            for s in range(shard_count):
                (subdir / f"shard{s:02d}").mkdir(parents=True)

        gc.disable()
        t0 = time.perf_counter_ns()
        for i in range(N_PHASE2):
            if shard_count == 1:
                fpath = subdir / f"T{i:04d}.KO05.mktf"
            else:
                shard = i % shard_count
                fpath = subdir / f"shard{shard:02d}" / f"T{i:04d}.KO05.mktf"

            write_mktf(fpath, all_cols[i], leaf_id=f"T{i:04d}.K02.KO05",
                       safe=False, alignment=alignment)
        elapsed_us = (time.perf_counter_ns() - t0) / 1e3
        gc.enable()

        per_file = elapsed_us / N_PHASE2
        print(f"  {label:>20}: {elapsed_us / 1e6:.2f}s total, {per_file / 1000:.3f}ms/file")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Bulk read scaling
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: BULK READ (2000 files from Phase 1 directory)")
    print("-" * 60)

    # The 2000 files from Phase 1 are still on disk
    file_list = sorted(tmpdir.glob("T*.KO05.mktf"))
    n_read = len(file_list)
    print(f"  Found {n_read} files to read")

    # Warm
    for f in file_list[:5]:
        read_columns(f)

    # Read in batches
    read_batch_times = []
    gc.disable()

    idx = 0
    while idx < n_read:
        batch_end = min(idx + BATCH_SIZE, n_read)
        batch_n = batch_end - idx

        t0 = time.perf_counter_ns()
        for i in range(idx, batch_end):
            header, cols = read_columns(file_list[i])
        elapsed_us = (time.perf_counter_ns() - t0) / 1e3
        per_file = elapsed_us / batch_n

        read_batch_times.append((idx, elapsed_us, per_file))
        idx = batch_end

    gc.enable()

    print(f"  {'Batch':>10} | {'Batch Time':>12} | {'Per-File':>10} | {'Drift':>8}")
    print(f"  {'-----':>10} | {'----------':>12} | {'--------':>10} | {'-----':>8}")

    read_baseline = read_batch_times[0][2]
    for start_idx, total_us, per_file_us in read_batch_times:
        drift = per_file_us / read_baseline
        print(f"  {start_idx:>5}-{start_idx + BATCH_SIZE - 1:<4} | "
              f"{total_us / 1000:>9.2f}ms | {per_file_us / 1000:>7.3f}ms | {drift:>7.2f}x")

    total_read_us = sum(t[1] for t in read_batch_times)
    read_per_file = total_read_us / n_read
    print()
    print(f"  Total: {total_read_us / 1e6:.2f}s for {n_read} files")
    print(f"  Overall per-file: {read_per_file:.0f}us")
    print(f"  Exp 19 single-file: ~105us (align=64, 78 bins)")
    print(f"  Ratio (bulk/single): {read_per_file / 105:.2f}x")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Universe extrapolation
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: UNIVERSE EXTRAPOLATION")
    print("-" * 60)

    tickers = 4604
    for n_cad in [8, 10]:
        total_files = tickers * n_cad
        write_s = total_files * (overall_per_file / 1e6)
        read_s = total_files * (read_per_file / 1e6)
        print(f"  {n_cad} cadences: {total_files:,} files")
        print(f"    Write: {overall_per_file / 1000:.3f}ms/file × {total_files:,} = {write_s:.1f}s")
        print(f"    Read:  {read_per_file:.0f}us/file × {total_files:,} = {read_s:.1f}s")
        print(f"    (Exp 19 projections: write={total_files * 1.54 / 1000:.1f}s, read={total_files * 0.105 / 1000:.1f}s)")
    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
