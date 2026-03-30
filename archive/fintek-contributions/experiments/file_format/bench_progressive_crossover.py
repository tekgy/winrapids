"""Experiment 16: Progressive Level Read Crossover.

At what bin count does reading a progressive level become more expensive than
reading the full column data? Experiment 15 showed:
  - 13 bins (30min):     926us
  - 78 bins (5min):    1,162us
  - 23,400 bins (1s): 61,012us
  - Full column read:  4,072us

The crossover determines when the daemon should use progressive stats vs
reading raw column data. This experiment sweeps bin counts from 1 to 50,000
to find the exact crossover and characterize the cost curve.
"""

from __future__ import annotations

import gc
import io
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.format import (
    UpstreamFingerprint, pack_progressive_section,
    unpack_progressive_directory, unpack_progressive_level,
)
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import (
    read_columns, read_header, read_progressive_summary,
    read_progressive_level_data,
)

sys.path.insert(0, str(Path("R:/winrapids/research/20260327-mktf-format")))
from mktf_v3 import AAPL_PATH, COL_MAP, CONDITION_BITS


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


def make_single_level_stats(
    col_names: list[str], n_bins: int, cadence_ms: int, n_rows: int,
) -> list[tuple[int, dict[str, np.ndarray]]]:
    """Generate progressive stats with exactly one level at given bin count."""
    rng = np.random.default_rng(42)
    col_stats = {}
    ticks_per_bin = max(1, n_rows // n_bins)
    for name in col_names:
        arr = np.zeros((n_bins, 5), dtype=np.float32)
        arr[:, 0] = rng.normal(100, 10, n_bins).astype(np.float32)
        arr[:, 1] = rng.normal(10000, 100, n_bins).astype(np.float32)
        arr[:, 2] = rng.uniform(80, 95, n_bins).astype(np.float32)
        arr[:, 3] = rng.uniform(105, 120, n_bins).astype(np.float32)
        arr[:, 4] = np.full(n_bins, ticks_per_bin, dtype=np.float32)
        col_stats[name] = arr
    return [(cadence_ms, col_stats)]


def make_multi_level_stats(
    col_names: list[str], target_bins: int, n_rows: int,
) -> list[tuple[int, dict[str, np.ndarray]]]:
    """Generate progressive stats where the target cadence has the given bin count,
    plus a coarser companion level. This matches realistic usage (always at least 2 levels)."""
    rng = np.random.default_rng(42)
    session_ms = 6.5 * 3600 * 1000

    # Target level
    target_cad_ms = max(1, int(session_ms / target_bins))
    # Coarser companion (10x fewer bins)
    coarse_bins = max(1, target_bins // 10)
    coarse_cad_ms = max(1, int(session_ms / coarse_bins))

    levels = []
    for cad_ms, n_bins in [(coarse_cad_ms, coarse_bins), (target_cad_ms, target_bins)]:
        col_stats = {}
        ticks_per_bin = max(1, n_rows // n_bins)
        for name in col_names:
            arr = np.zeros((n_bins, 5), dtype=np.float32)
            arr[:, 0] = rng.normal(100, 10, n_bins).astype(np.float32)
            arr[:, 1] = rng.normal(10000, 100, n_bins).astype(np.float32)
            arr[:, 2] = rng.uniform(80, 95, n_bins).astype(np.float32)
            arr[:, 3] = rng.uniform(105, 120, n_bins).astype(np.float32)
            arr[:, 4] = np.full(n_bins, ticks_per_bin, dtype=np.float32)
            col_stats[name] = arr
        levels.append((cad_ms, col_stats))

    return levels


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


def main():
    N = 20  # runs per benchmark point

    print("=" * 78)
    print("EXPERIMENT 16: PROGRESSIVE LEVEL READ CROSSOVER")
    print("=" * 78)
    print(f"Runs per benchmark: {N}")
    print()

    cols = load_source_columns()
    col_names = list(cols.keys())
    n_rows = len(cols["price"])
    n_cols = len(col_names)

    tmpdir = tempfile.mkdtemp(prefix="mktf_cross_")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Full column read baseline
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: FULL COLUMN READ BASELINE")
    print("-" * 60)

    # Write a file without progressive (baseline)
    base_path = Path(tmpdir) / "baseline.mktf"
    write_mktf(base_path, cols, leaf_id="AAPL.K01", ticker="AAPL",
               day="2025-09-02", safe=False)

    # Warm read
    for _ in range(3):
        read_columns(base_path)

    full_read = timed_us(lambda: read_columns(base_path), N)
    print(f"  Full column read: {full_read['mean']:.0f}us +/- {full_read['std']:.0f}")
    baseline_us = full_read["mean"]
    print(f"  Baseline for crossover: {baseline_us:.0f}us")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Progressive level read sweep
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: PROGRESSIVE LEVEL READ vs BIN COUNT")
    print("-" * 60)

    # Sweep: logarithmically spaced bin counts
    bin_counts = [5, 10, 20, 50, 100, 200, 500, 780, 1000, 2000,
                  5000, 10000, 20000, 50000]

    session_ms = 6.5 * 3600 * 1000

    results = []
    print(f"  {'Bins':>7} {'Cadence':>10} {'Read (us)':>12} {'Std':>8} {'vs Full':>8} {'Data (KB)':>10}")
    print(f"  {'----':>7} {'-------':>10} {'---------':>12} {'---':>8} {'-------':>8} {'---------':>10}")

    for n_bins in bin_counts:
        cad_ms = max(1, int(session_ms / n_bins))

        # Generate stats and write file
        prog_stats = make_single_level_stats(col_names, n_bins, cad_ms, n_rows)
        fpath = Path(tmpdir) / f"prog_{n_bins}.mktf"
        write_mktf(fpath, cols, leaf_id="AAPL.K01", ticker="AAPL",
                   day="2025-09-02", progressive_stats=prog_stats, safe=False)

        # Warm reads
        for _ in range(3):
            read_progressive_level_data(fpath, cad_ms, col_names)

        # Benchmark
        t = timed_us(lambda p=fpath, c=cad_ms: read_progressive_level_data(p, c, col_names), N)

        data_kb = n_bins * n_cols * 5 * 4 / 1024  # float32 stats
        ratio = t["mean"] / baseline_us

        results.append({
            "bins": n_bins,
            "cad_ms": cad_ms,
            "read_us": t["mean"],
            "read_std": t["std"],
            "ratio": ratio,
            "data_kb": data_kb,
        })

        cad_label = f"{cad_ms}ms" if cad_ms < 1000 else f"{cad_ms/1000:.0f}s" if cad_ms < 60000 else f"{cad_ms/60000:.0f}min"
        marker = " <-- CROSSOVER" if 0.9 < ratio < 1.1 else (" *** SLOWER" if ratio > 1.0 else "")
        print(f"  {n_bins:>7} {cad_label:>10} {t['mean']:>10.0f}us {t['std']:>6.0f} {ratio:>7.2f}x {data_kb:>9.1f}{marker}")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Find exact crossover via interpolation
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: CROSSOVER ANALYSIS")
    print("-" * 60)

    # Find where ratio crosses 1.0
    crossover_bins = None
    for i in range(len(results) - 1):
        if results[i]["ratio"] < 1.0 and results[i + 1]["ratio"] >= 1.0:
            # Linear interpolation
            r0 = results[i]
            r1 = results[i + 1]
            frac = (1.0 - r0["ratio"]) / (r1["ratio"] - r0["ratio"])
            crossover_bins = r0["bins"] + frac * (r1["bins"] - r0["bins"])
            crossover_data_kb = r0["data_kb"] + frac * (r1["data_kb"] - r0["data_kb"])
            print(f"  Crossover at ~{crossover_bins:.0f} bins ({crossover_data_kb:.1f} KB)")
            print(f"  Between {r0['bins']} bins ({r0['read_us']:.0f}us, {r0['ratio']:.2f}x)")
            print(f"  and     {r1['bins']} bins ({r1['read_us']:.0f}us, {r1['ratio']:.2f}x)")
            break

    if crossover_bins is None:
        # Check if always above or always below
        if all(r["ratio"] < 1.0 for r in results):
            print("  No crossover found — progressive is ALWAYS cheaper (up to 50K bins)")
        elif all(r["ratio"] >= 1.0 for r in results):
            print("  No crossover found — progressive is ALWAYS more expensive (from 5 bins)")
        else:
            print("  Non-monotonic — crossover region is complex")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Progressive summary (directory only) as reference
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: PROGRESSIVE SUMMARY (directory only) REFERENCE")
    print("-" * 60)

    # Use the largest progressive file for summary read
    largest_path = Path(tmpdir) / f"prog_{bin_counts[-1]}.mktf"
    for _ in range(3):
        read_progressive_summary(largest_path)

    summary_t = timed_us(lambda: read_progressive_summary(largest_path), N)
    print(f"  Summary read (50K-bin file): {summary_t['mean']:.0f}us +/- {summary_t['std']:.0f}")
    print(f"  vs full column read: {summary_t['mean']/baseline_us:.3f}x")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Cost model
    # ═══════════════════════════════════════════════════════════
    print("PHASE 5: COST MODEL")
    print("-" * 60)

    # Fit: read_us = a + b * n_bins (linear model)
    bins_arr = np.array([r["bins"] for r in results], dtype=np.float64)
    reads_arr = np.array([r["read_us"] for r in results], dtype=np.float64)

    # Linear regression
    A = np.vstack([np.ones_like(bins_arr), bins_arr]).T
    (intercept, slope), residuals, _, _ = np.linalg.lstsq(A, reads_arr, rcond=None)

    print(f"  Linear model: read_us = {intercept:.1f} + {slope:.4f} * n_bins")
    print(f"  Fixed overhead: {intercept:.0f}us (file open + header + directory parse)")
    print(f"  Per-bin cost: {slope:.4f}us ({slope*1000:.2f}ns)")
    print(f"  Per-bin data: {n_cols * 5 * 4} bytes ({n_cols} cols x 5 stats x 4B)")
    print(f"  Implied bandwidth: {n_cols * 5 * 4 / (slope / 1e6) / 1e9:.2f} GB/s")
    print()

    # Predicted crossover from model
    if slope > 0:
        model_crossover = (baseline_us - intercept) / slope
        model_crossover_kb = model_crossover * n_cols * 5 * 4 / 1024
        print(f"  Model-predicted crossover: {model_crossover:.0f} bins ({model_crossover_kb:.1f} KB)")
        print(f"  At that point: progressive read = full column read = {baseline_us:.0f}us")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: Decision table for the daemon
    # ═══════════════════════════════════════════════════════════
    print("PHASE 6: DECISION TABLE")
    print("-" * 60)
    print(f"  Full column read baseline: {baseline_us:.0f}us")
    print()
    print(f"  {'Cadence':>10} {'Bins':>7} {'Prog Read':>10} {'Strategy':>15}")
    print(f"  {'-------':>10} {'----':>7} {'---------':>10} {'--------':>15}")

    cadence_labels = [
        ("30min", 13), ("20min", 19), ("15min", 26), ("10min", 39),
        ("5min", 78), ("1min", 390), ("30s", 780), ("10s", 2340),
        ("5s", 4680), ("1s", 23400),
    ]

    for label, bins in cadence_labels:
        predicted_us = intercept + slope * bins
        strategy = "PROGRESSIVE" if predicted_us < baseline_us else "FULL READ"
        print(f"  {label:>10} {bins:>7} {predicted_us:>8.0f}us {'  <-- ' + strategy:>15}")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 7: Write cost vs bin count
    # ═══════════════════════════════════════════════════════════
    print("PHASE 7: WRITE COST vs BIN COUNT (batch mode)")
    print("-" * 60)

    write_bins = [13, 78, 390, 780, 2340, 4680, 23400]
    print(f"  {'Bins':>7} {'Write (ms)':>12} {'Prog Size (KB)':>15} {'% of File':>10}")
    print(f"  {'----':>7} {'---------':>12} {'--------------':>15} {'---------':>10}")

    for n_bins in write_bins:
        cad_ms = max(1, int(session_ms / n_bins))
        prog_stats = make_single_level_stats(col_names, n_bins, cad_ms, n_rows)
        fpath = Path(tmpdir) / f"write_{n_bins}.mktf"

        # Warm
        for _ in range(3):
            write_mktf(fpath, cols, leaf_id="AAPL.K01", ticker="AAPL",
                       day="2025-09-02", progressive_stats=prog_stats, safe=False)

        t = timed_us(lambda n=n_bins, c=cad_ms: write_mktf(
            Path(tmpdir) / f"write_{n}.mktf", cols, leaf_id="AAPL.K01", ticker="AAPL",
            day="2025-09-02", progressive_stats=make_single_level_stats(col_names, n, c, n_rows),
            safe=False
        ), N)

        file_size_kb = fpath.stat().st_size / 1024
        prog_kb = n_bins * n_cols * 5 * 4 / 1024
        pct = prog_kb / file_size_kb * 100

        print(f"  {n_bins:>7} {t['mean']/1000:>10.2f}ms {prog_kb:>13.1f}KB {pct:>9.1f}%")

    print()

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
