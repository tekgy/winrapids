"""Experiment 18: Column Bundling — 25 narrow vs 5 wide columns.

KO05 files have 25 stat columns (5 data cols × 5 stats) with 4096-byte alignment
each = 110 KB minimum. Bundling into 5 wide columns (one per data col, shape
(n_bins, 5)) cuts alignment from 25×4096 to 5×4096.

Questions:
  1. How much faster is writing 5 wide columns vs 25 narrow?
  2. Where does the per-file time floor live (file open/close/rename)?
  3. What's the realistic universe projection for KO05 writes?
  4. Does column count affect read speed proportionally?
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_columns, read_header


def timed_us(fn, n_runs: int) -> dict:
    gc.disable()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e3)
    gc.enable()
    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.median(arr)),
    }


def make_narrow_columns(n_bins: int, n_data_cols: int = 5) -> dict[str, np.ndarray]:
    """25 narrow columns: {price_sum, price_sum_sq, price_min, price_max, price_count, size_sum, ...}"""
    rng = np.random.default_rng(42)
    stat_names = ["sum", "sum_sq", "min", "max", "count"]
    data_names = ["price", "size", "timestamp", "exchange", "conditions"]
    cols = {}
    for dname in data_names[:n_data_cols]:
        for sname in stat_names:
            cols[f"{dname}_{sname}"] = rng.normal(100, 10, n_bins).astype(np.float32)
    return cols


def make_wide_columns(n_bins: int, n_data_cols: int = 5) -> dict[str, np.ndarray]:
    """5 wide columns: {price_stats: (n_bins, 5), size_stats: (n_bins, 5), ...}"""
    rng = np.random.default_rng(42)
    data_names = ["price", "size", "timestamp", "exchange", "conditions"]
    cols = {}
    for dname in data_names[:n_data_cols]:
        cols[f"{dname}_stats"] = rng.normal(100, 10, (n_bins, 5)).astype(np.float32)
    return cols


def make_single_column(n_bins: int, n_data_cols: int = 5) -> dict[str, np.ndarray]:
    """1 column: {all_stats: (n_bins, 25)}"""
    rng = np.random.default_rng(42)
    cols = {"all_stats": rng.normal(100, 10, (n_bins, n_data_cols * 5)).astype(np.float32)}
    return cols


def main():
    N = 30

    print("=" * 78)
    print("EXPERIMENT 18: COLUMN BUNDLING (25 narrow vs 5 wide vs 1 mega)")
    print("=" * 78)
    print(f"Runs per benchmark: {N}")
    print()

    tmpdir = tempfile.mkdtemp(prefix="mktf_bundle_")

    # Test at multiple bin counts
    bin_counts = [1, 13, 78, 390, 780, 2340]

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Write speed comparison
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: WRITE SPEED")
    print("-" * 60)

    configs = [
        ("25 narrow", make_narrow_columns),
        ("5 wide", make_wide_columns),
        ("1 mega", make_single_column),
    ]

    print(f"  {'Bins':>6} | ", end="")
    for label, _ in configs:
        print(f"{'Write':>10} {'Size':>8} | ", end="")
    print("Speedup (25→5)")

    print(f"  {'----':>6} | ", end="")
    for _ in configs:
        print(f"{'-----':>10} {'----':>8} | ", end="")
    print("-" * 14)

    for n_bins in bin_counts:
        results = []
        for label, make_fn in configs:
            cols = make_fn(n_bins)
            fpath = Path(tmpdir) / f"{label.replace(' ', '_')}_{n_bins}.mktf"

            # Warm
            for _ in range(3):
                write_mktf(fpath, cols, leaf_id="AAPL.K02.KO05", ticker="AAPL",
                           day="2025-09-02", safe=False)

            t = timed_us(lambda c=cols, p=fpath: write_mktf(
                p, c, leaf_id="AAPL.K02.KO05", ticker="AAPL",
                day="2025-09-02", safe=False
            ), N)

            file_size = fpath.stat().st_size
            results.append((t, file_size))

        speedup = results[0][0]["mean"] / results[1][0]["mean"]

        print(f"  {n_bins:>6} | ", end="")
        for t, sz in results:
            print(f"{t['mean']/1000:>8.2f}ms {sz/1024:>6.1f}KB | ", end="")
        print(f"{speedup:>10.1f}x")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Write time decomposition (78 bins = 5min cadence)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: WRITE TIME DECOMPOSITION (78 bins)")
    print("-" * 60)

    for label, make_fn in configs:
        cols = make_fn(78)
        n_cols = len(cols)
        data_bytes = sum(a.nbytes for a in cols.values())
        aligned_bytes = n_cols * 4096  # minimum from alignment

        fpath = Path(tmpdir) / f"decomp_{label.replace(' ', '_')}.mktf"

        # Measure individual components
        # 1. Just file open/close/write header (minimal file)
        min_cols = {"x": np.zeros(1, dtype=np.float32)}
        t_minimal = timed_us(lambda: write_mktf(
            Path(tmpdir) / "minimal.mktf", min_cols,
            leaf_id="X", safe=False
        ), N)

        # 2. Full write
        t_full = timed_us(lambda c=cols, p=fpath: write_mktf(
            p, c, leaf_id="AAPL.K02.KO05", ticker="AAPL",
            day="2025-09-02", safe=False
        ), N)

        overhead_us = t_full["mean"] - t_minimal["mean"]
        print(f"  {label:>12}: {t_full['mean']/1000:.2f}ms total, "
              f"{t_minimal['mean']/1000:.2f}ms floor, "
              f"{overhead_us/1000:.2f}ms overhead | "
              f"{n_cols} cols, {data_bytes}B data, {aligned_bytes}B aligned")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Read speed comparison
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: READ SPEED (78 bins)")
    print("-" * 60)

    for label, make_fn in configs:
        cols = make_fn(78)
        fpath = Path(tmpdir) / f"read_{label.replace(' ', '_')}.mktf"
        write_mktf(fpath, cols, leaf_id="AAPL.K02.KO05", ticker="AAPL",
                   day="2025-09-02", safe=False)

        # Warm
        for _ in range(5):
            read_columns(fpath)

        t = timed_us(lambda p=fpath: read_columns(fpath), N)
        file_size = fpath.stat().st_size
        print(f"  {label:>12}: {t['mean']:.0f}us ±{t['std']:.0f} ({file_size/1024:.1f} KB)")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Universe projections
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: UNIVERSE PROJECTIONS")
    print("-" * 60)

    # Use 78 bins (5min cadence) as representative
    cadence_counts = [8, 10]
    tickers = 4604

    for label, make_fn in configs:
        cols = make_fn(78)
        fpath = Path(tmpdir) / f"proj_{label.replace(' ', '_')}.mktf"

        # Measure write time
        for _ in range(3):
            write_mktf(fpath, cols, leaf_id="X", safe=False)
        t = timed_us(lambda c=cols, p=fpath: write_mktf(
            p, c, leaf_id="X", safe=False
        ), N)

        per_file_ms = t["mean"] / 1000
        for n_cad in cadence_counts:
            total_files = tickers * n_cad
            total_s = total_files * per_file_ms / 1000
            print(f"  {label:>12} × {n_cad} cad: {per_file_ms:.2f}ms/file × {total_files:,} = {total_s:.1f}s")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Correctness — verify data roundtrips
    # ═══════════════════════════════════════════════════════════
    print("PHASE 5: CORRECTNESS")
    print("-" * 60)

    for label, make_fn in configs:
        cols = make_fn(78)
        fpath = Path(tmpdir) / f"correct_{label.replace(' ', '_')}.mktf"
        write_mktf(fpath, cols, leaf_id="X", safe=False)
        read_back = read_columns(fpath)

        all_match = True
        for name in cols:
            original = cols[name].ravel()
            recovered = read_back[name]
            if not np.array_equal(original, recovered):
                print(f"  MISMATCH: {label} / {name}")
                all_match = False
        if all_match:
            print(f"  {label:>12}: all {len(cols)} columns roundtrip correctly")

    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
