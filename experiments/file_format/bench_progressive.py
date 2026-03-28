"""Experiment 15: Progressive Section + safe=False Benchmark.

Pathmaker has wired progressive stats and dual write mode into production v4.
This experiment validates:
  1. Write overhead of progressive section (tiny vs source-scale)
  2. safe=False speedup matches Experiment 14's prediction (62% for small, 91% grid)
  3. Progressive summary read cost (directory only, no stat data)
  4. Progressive level data read cost (one cadence)
  5. Write+read roundtrip correctness
  6. File size overhead from progressive section
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
from trunk.backends.mktf.format import (
    UpstreamFingerprint, pack_progressive_section,
    unpack_progressive_directory, unpack_progressive_level,
)
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import (
    read_columns, read_header, read_progressive_summary,
    read_progressive_level_data, verify_checksum,
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


def make_progressive_stats(
    col_names: list[str], cadences_ms: list[int], n_rows: int
) -> list[tuple[int, dict[str, np.ndarray]]]:
    """Generate synthetic progressive stats for benchmarking."""
    rng = np.random.default_rng(42)
    session_ms = 6.5 * 3600 * 1000  # 6.5 hours

    stats = []
    for cad_ms in sorted(cadences_ms, reverse=True):  # coarsest first
        n_bins = max(1, int(session_ms / cad_ms))
        col_stats = {}
        for name in col_names:
            arr = np.zeros((n_bins, 5), dtype=np.float32)
            ticks_per_bin = max(1, n_rows // n_bins)
            arr[:, 0] = rng.normal(100, 10, n_bins).astype(np.float32)  # sum
            arr[:, 1] = rng.normal(10000, 100, n_bins).astype(np.float32)  # sum_sq
            arr[:, 2] = rng.uniform(80, 95, n_bins).astype(np.float32)  # min
            arr[:, 3] = rng.uniform(105, 120, n_bins).astype(np.float32)  # max
            arr[:, 4] = np.full(n_bins, ticks_per_bin, dtype=np.float32)  # count
            col_stats[name] = arr
        stats.append((cad_ms, col_stats))
    return stats


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


def fmt_us(s: dict) -> str:
    return f"{s['mean']:.0f}us ±{s['std']:.0f}"

def fmt_ms(s: dict) -> str:
    return f"{s['mean']/1000:.2f}ms ±{s['std']/1000:.2f}"


def main():
    N = 20

    print("=" * 78)
    print("EXPERIMENT 15: PROGRESSIVE SECTION + DUAL WRITE MODE")
    print("=" * 78)
    print(f"Runs per benchmark: {N}")
    print()

    cols = load_source_columns()
    col_names = list(cols.keys())
    n_rows = len(cols["price"])

    # Cadence grid matching the detrended spectral findings
    cadences_ms = [
        1000,    # 1s
        5000,    # 5s
        10000,   # 10s
        30000,   # 30s
        60000,   # 1min
        300000,  # 5min (the real institutional boundary)
        600000,  # 10min
        900000,  # 15min
        1200000, # 20min
        1800000, # 30min
    ]

    prog_stats = make_progressive_stats(col_names, cadences_ms, n_rows)
    mi_scores = [0.85, 0.72, 0.68, 0.55, 0.42, 0.38, 0.31, 0.28, 0.22]  # synthetic

    tmpdir = tempfile.mkdtemp(prefix="mktf_prog_")

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Write comparison (with vs without progressive, safe vs unsafe)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: WRITE COST COMPARISON")
    print("-" * 60)

    configs = [
        ("No progressive, safe=True",   None,       None,      True),
        ("No progressive, safe=False",  None,       None,      False),
        ("With progressive, safe=True",  prog_stats, mi_scores, True),
        ("With progressive, safe=False", prog_stats, mi_scores, False),
    ]

    print(f"  {'Config':<42} {'Write':>12} {'File Size':>10} {'Overhead':>10}")
    print(f"  {'-'*74}")

    sizes = {}
    write_stats = {}
    for label, pstats, pmi, safe in configs:
        p = os.path.join(tmpdir, f"bench_{label.replace(' ', '_').replace(',','')}.mktf")

        # Warmup
        write_mktf(p, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
                   progressive_stats=pstats, progressive_mi=pmi, safe=safe)

        stats = timed_us(
            lambda p=p, ps=pstats, pm=pmi, s=safe: write_mktf(
                p, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
                progressive_stats=ps, progressive_mi=pm, safe=s),
            N)

        fsize = os.path.getsize(p)
        sizes[label] = fsize
        write_stats[label] = stats

        overhead = ""
        if "No progressive, safe=True" in sizes and label != "No progressive, safe=True":
            base = sizes["No progressive, safe=True"]
            overhead = f"+{(fsize-base)/1024:.1f}KB"

        print(f"  {label:<42} {fmt_ms(stats):>18} {fsize/1024:.1f}KB {overhead:>10}")

    # Speedups
    print()
    base_safe = write_stats["No progressive, safe=True"]["mean"]
    for label, stats in write_stats.items():
        if label != "No progressive, safe=True":
            speedup = base_safe / stats["mean"]
            print(f"  {label:<42} {speedup:.2f}x vs baseline")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Read cost comparison
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: READ COST COMPARISON")
    print("-" * 60)

    # Write a file with progressive stats for read benchmarks
    prog_path = os.path.join(tmpdir, "with_progressive.mktf")
    write_mktf(prog_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
               progressive_stats=prog_stats, progressive_mi=mi_scores, safe=False)

    noprog_path = os.path.join(tmpdir, "no_progressive.mktf")
    write_mktf(noprog_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
               safe=False)

    # Warmup
    read_columns(prog_path)
    read_columns(noprog_path)
    read_progressive_summary(prog_path)
    read_progressive_level_data(prog_path, 300000, col_names)

    # 2a: Full column read (should be unaffected by progressive section)
    stats_read_prog = timed_us(lambda: read_columns(prog_path), N * 2)
    stats_read_noprog = timed_us(lambda: read_columns(noprog_path), N * 2)

    print(f"  Full read (no progressive):   {fmt_us(stats_read_noprog)}")
    print(f"  Full read (with progressive): {fmt_us(stats_read_prog)}")
    delta = stats_read_prog["mean"] - stats_read_noprog["mean"]
    print(f"  Progressive read overhead:     {delta:.0f}us ({delta/stats_read_noprog['mean']*100:.1f}%)")
    print()

    # 2b: Progressive summary (directory only)
    stats_summary = timed_us(lambda: read_progressive_summary(prog_path), N * 2)
    print(f"  Progressive summary (dir only):  {fmt_us(stats_summary)}")

    # 2c: Progressive level data (one cadence)
    stats_5min = timed_us(
        lambda: read_progressive_level_data(prog_path, 300000, col_names), N * 2)
    stats_1s = timed_us(
        lambda: read_progressive_level_data(prog_path, 1000, col_names), N * 2)
    stats_30min = timed_us(
        lambda: read_progressive_level_data(prog_path, 1800000, col_names), N * 2)

    print(f"  Level data (5min, 78 bins):      {fmt_us(stats_5min)}")
    print(f"  Level data (1s, 23400 bins):     {fmt_us(stats_1s)}")
    print(f"  Level data (30min, 13 bins):     {fmt_us(stats_30min)}")
    print()

    # 2d: Header read (should be same either way — 3 new fields in Layout)
    stats_hdr_prog = timed_us(lambda: read_header(prog_path), N * 2)
    stats_hdr_noprog = timed_us(lambda: read_header(noprog_path), N * 2)
    print(f"  Header only (no progressive):    {fmt_us(stats_hdr_noprog)}")
    print(f"  Header only (with progressive):  {fmt_us(stats_hdr_prog)}")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Correctness verification
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: CORRECTNESS VERIFICATION")
    print("-" * 60)

    # 3a: Column data unaffected
    _, cols_prog = read_columns(prog_path)
    _, cols_noprog = read_columns(noprog_path)
    all_ok = True
    for name in cols_noprog:
        if not np.array_equal(cols_prog[name], cols_noprog[name]):
            print(f"  FAIL: column {name} differs with progressive section")
            all_ok = False
    if all_ok:
        print(f"  Column data: IDENTICAL (progressive section doesn't affect columns)")

    # 3b: Checksum passes
    ok1 = verify_checksum(prog_path)
    ok2 = verify_checksum(noprog_path)
    print(f"  Checksum (with progressive): {'PASS' if ok1 else 'FAIL'}")
    print(f"  Checksum (no progressive):   {'PASS' if ok2 else 'FAIL'}")

    # 3c: Progressive stats roundtrip
    summary = read_progressive_summary(prog_path)
    if summary is not None:
        n_levels, n_cols_s, levels, mi = summary
        print(f"  Progressive summary: {n_levels} levels, {n_cols_s} cols")
        print(f"  MI scores: {[f'{m:.2f}' for m in mi]}")
        print(f"  Cadences: {[f'{l.cadence_ms}ms ({l.bin_count} bins)' for l in levels]}")

        # Verify one level roundtrips
        level_data = read_progressive_level_data(prog_path, 300000, col_names)
        if level_data is not None:
            ref_stats = dict(prog_stats)[300000]
            match = all(
                np.allclose(level_data[name], ref_stats[name], atol=1e-6)
                for name in col_names
            )
            print(f"  5min level roundtrip: {'MATCH' if match else 'MISMATCH'}")
    else:
        print(f"  FAIL: read_progressive_summary returned None")

    # 3d: No progressive section on non-progressive file
    null_summary = read_progressive_summary(noprog_path)
    print(f"  Non-progressive file summary: {'None (correct)' if null_summary is None else 'ERROR'}")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Progressive section size analysis
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: SIZE ANALYSIS")
    print("-" * 60)

    prog_size = sizes["With progressive, safe=True"]
    noprog_size = sizes["No progressive, safe=True"]
    delta_kb = (prog_size - noprog_size) / 1024

    print(f"  Without progressive: {noprog_size/1024:.1f} KB")
    print(f"  With progressive:    {prog_size/1024:.1f} KB")
    print(f"  Progressive overhead: {delta_kb:.1f} KB ({delta_kb/noprog_size*1024*100:.1f}%)")
    print()

    # Breakdown
    # Section header: 8 bytes
    # MI scores: 9 × 4 = 36 bytes
    # Level directory: 10 × 24 = 240 bytes
    # Level data: sum of bins × cols × 5 stats × 4 bytes
    session_ms = 6.5 * 3600 * 1000
    total_bins = sum(max(1, int(session_ms / c)) for c in cadences_ms)
    data_bytes = total_bins * len(col_names) * 5 * 4
    overhead_bytes = 8 + 36 + 240
    print(f"  Bins across all cadences: {total_bins}")
    print(f"  Stats data: {data_bytes/1024:.1f} KB ({total_bins} bins × {len(col_names)} cols × 5 stats × 4B)")
    print(f"  Directory overhead: {overhead_bytes} bytes")
    print(f"  Total theoretical: {(data_bytes+overhead_bytes)/1024:.1f} KB")
    print(f"  Actual delta: {delta_kb:.1f} KB (includes 4096-byte alignment padding)")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Universe projections
    # ═══════════════════════════════════════════════════════════
    print("PHASE 5: UNIVERSE PROJECTIONS (4604 tickers)")
    print("-" * 60)
    n_tickers = 4604

    print(f"  {'Operation':<42} {'Per-file':>10} {'Universe':>10}")
    print(f"  {'-'*62}")

    ops = [
        ("Write (safe, no progressive)", write_stats["No progressive, safe=True"]),
        ("Write (safe, with progressive)", write_stats["With progressive, safe=True"]),
        ("Write (batch, no progressive)", write_stats["No progressive, safe=False"]),
        ("Write (batch, with progressive)", write_stats["With progressive, safe=False"]),
        ("Full column read", stats_read_prog),
        ("Progressive summary only", stats_summary),
        ("Progressive 5min level", stats_5min),
        ("Progressive 30min level", stats_30min),
    ]

    for label, s in ops:
        per_ms = s["mean"] / 1000
        univ_s = per_ms * n_tickers / 1000
        print(f"  {label:<42} {per_ms:>7.2f}ms {univ_s:>7.1f}s")

    print()

    # Coarse K04 from progressive stats
    print(f"  COARSE K04 VIA PROGRESSIVE (vs full column read):")
    prog_read_s = stats_summary["mean"] / 1000 * n_tickers / 1000
    full_read_s = stats_read_prog["mean"] / 1000 * n_tickers / 1000
    print(f"    Progressive summary read: {prog_read_s:.1f}s")
    print(f"    Full column read:         {full_read_s:.1f}s")
    print(f"    Speedup: {full_read_s/prog_read_s:.0f}x")
    print()

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    main()
