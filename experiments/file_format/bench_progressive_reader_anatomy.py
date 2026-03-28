"""Experiment 17: Progressive Reader Anatomy.

The progressive reader's 0.04 GB/s bandwidth (Experiment 16) is 99% Python overhead.
Root cause: nested Python loop in unpack_progressive_level (format.py:815-820).
  for b in range(n_bins):          # 23,400 iterations for 1s cadence
      for name in col_names:       # × 5 columns
          np.frombuffer(...)       # = 117,000 Python calls

This experiment:
  1. Measures each component of the progressive read path
  2. Benchmarks an alternative column-major reader (5 np.frombuffer calls)
  3. Benchmarks raw byte read (no unpacking) as GPU-transfer ceiling
  4. Characterizes the crossover shift if the reader were optimized
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
    ProgressiveLevel, PROG_STAT_FIELDS, PROG_STAT_BYTES,
    pack_progressive_section, unpack_progressive_directory,
    unpack_progressive_level,
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


def fast_unpack_level_bulk(data: bytes, level: ProgressiveLevel, col_names: list[str]) -> dict[str, np.ndarray]:
    """Optimized reader: single np.frombuffer + reshape + transpose.
    No Python per-bin loop."""
    n_bins = level.bin_count
    n_cols = len(col_names)
    total_floats = n_bins * n_cols * PROG_STAT_FIELDS

    # Read entire level as one flat array
    raw = np.frombuffer(
        data[level.offset:level.offset + total_floats * 4],
        dtype=np.float32,
    )

    # Layout is bin-major: [bin0_col0_stats, bin0_col1_stats, ..., bin1_col0_stats, ...]
    # Reshape to (n_bins, n_cols, STAT_FIELDS) then extract per-column
    shaped = raw.reshape(n_bins, n_cols, PROG_STAT_FIELDS)

    return {name: shaped[:, i, :].copy() for i, name in enumerate(col_names)}


def fast_unpack_level_contiguous(data: bytes, level: ProgressiveLevel, col_names: list[str]) -> dict[str, np.ndarray]:
    """Even faster: single frombuffer, reshape, return views (no copy)."""
    n_bins = level.bin_count
    n_cols = len(col_names)
    total_floats = n_bins * n_cols * PROG_STAT_FIELDS

    raw = np.frombuffer(
        data[level.offset:level.offset + total_floats * 4],
        dtype=np.float32,
    )
    shaped = raw.reshape(n_bins, n_cols, PROG_STAT_FIELDS)
    return {name: shaped[:, i, :] for i, name in enumerate(col_names)}


def raw_byte_read(data: bytes, level: ProgressiveLevel) -> bytes:
    """Minimum possible work: just slice the bytes. GPU-transfer ceiling."""
    n_cols = 5
    total = level.bin_count * n_cols * PROG_STAT_FIELDS * 4
    return data[level.offset:level.offset + total]


def main():
    N = 50  # more runs for micro-benchmarks

    print("=" * 78)
    print("EXPERIMENT 17: PROGRESSIVE READER ANATOMY")
    print("=" * 78)
    print(f"Runs per benchmark: {N}")
    print()

    cols = load_source_columns()
    col_names = list(cols.keys())
    n_cols = len(col_names)
    n_rows = len(cols["price"])

    tmpdir = tempfile.mkdtemp(prefix="mktf_reader_")
    session_ms = 6.5 * 3600 * 1000

    # Test at multiple bin counts
    test_cases = [
        ("30min", 13, 1800000),
        ("5min", 78, 300000),
        ("1min", 390, 60000),
        ("30s", 780, 30000),
        ("10s", 2340, 10000),
        ("5s", 4680, 5000),
        ("1s", 23400, 1000),
    ]

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Full column read baseline
    # ═══════════════════════════════════════════════════════════
    print("PHASE 1: BASELINE")
    print("-" * 60)

    base_path = Path(tmpdir) / "baseline.mktf"
    write_mktf(base_path, cols, leaf_id="AAPL.K01", ticker="AAPL",
               day="2025-09-02", safe=False)
    for _ in range(3):
        read_columns(base_path)
    full_read = timed_us(lambda: read_columns(base_path), N)
    baseline_us = full_read["mean"]
    print(f"  Full column read: {baseline_us:.0f}us")
    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Read path decomposition for each cadence
    # ═══════════════════════════════════════════════════════════
    print("PHASE 2: READER DECOMPOSITION")
    print("-" * 60)

    headers = [
        "Cadence", "Bins", "Data KB",
        "Production", "Bulk+reshape", "Bulk+view", "Raw bytes",
        "Speedup",
    ]
    print(f"  {headers[0]:>7} {headers[1]:>6} {headers[2]:>8} "
          f"{headers[3]:>12} {headers[4]:>14} {headers[5]:>12} {headers[6]:>10} "
          f"{headers[7]:>8}")
    print(f"  {'---':>7} {'---':>6} {'---':>8} "
          f"{'---':>12} {'---':>14} {'---':>12} {'---':>10} "
          f"{'---':>8}")

    for label, n_bins, cad_ms in test_cases:
        # Create progressive stats
        rng = np.random.default_rng(42)
        col_stats = {}
        for name in col_names:
            arr = np.zeros((n_bins, 5), dtype=np.float32)
            arr[:, 0] = rng.normal(100, 10, n_bins).astype(np.float32)
            arr[:, 1] = rng.normal(10000, 100, n_bins).astype(np.float32)
            arr[:, 2] = rng.uniform(80, 95, n_bins).astype(np.float32)
            arr[:, 3] = rng.uniform(105, 120, n_bins).astype(np.float32)
            arr[:, 4] = np.full(n_bins, max(1, n_rows // n_bins), dtype=np.float32)
            col_stats[name] = arr
        prog_stats = [(cad_ms, col_stats)]

        fpath = Path(tmpdir) / f"test_{n_bins}.mktf"
        write_mktf(fpath, cols, leaf_id="AAPL.K01", ticker="AAPL",
                   day="2025-09-02", progressive_stats=prog_stats, safe=False)

        # Read the progressive section bytes once (warm cache)
        header = read_header(fpath)
        with open(fpath, "rb") as f:
            f.seek(header.progressive_offset)
            prog_data = f.read(header.progressive_size)

        _, _, levels, _ = unpack_progressive_directory(prog_data)
        level = levels[0]
        data_kb = n_bins * n_cols * 5 * 4 / 1024

        # Warm all paths
        for _ in range(3):
            unpack_progressive_level(prog_data, level, col_names)
            fast_unpack_level_bulk(prog_data, level, col_names)
            fast_unpack_level_contiguous(prog_data, level, col_names)
            raw_byte_read(prog_data, level)

        # Benchmark: production reader (per-bin loop)
        t_prod = timed_us(lambda: unpack_progressive_level(prog_data, level, col_names), N)

        # Benchmark: bulk reshape (single frombuffer + reshape + copy)
        t_bulk = timed_us(lambda: fast_unpack_level_bulk(prog_data, level, col_names), N)

        # Benchmark: view-only (no copy)
        t_view = timed_us(lambda: fast_unpack_level_contiguous(prog_data, level, col_names), N)

        # Benchmark: raw bytes (minimum I/O)
        t_raw = timed_us(lambda: raw_byte_read(prog_data, level), N)

        speedup = t_prod["mean"] / t_bulk["mean"] if t_bulk["mean"] > 0 else 0

        print(f"  {label:>7} {n_bins:>6} {data_kb:>7.1f} "
              f"{t_prod['mean']:>10.0f}us {t_bulk['mean']:>12.0f}us "
              f"{t_view['mean']:>10.0f}us {t_raw['mean']:>8.0f}us "
              f"{speedup:>7.1f}x")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Full pipeline comparison (file open + header + unpack)
    # ═══════════════════════════════════════════════════════════
    print("PHASE 3: FULL PIPELINE (file I/O + unpack)")
    print("-" * 60)

    print(f"  {'Cadence':>7} {'Bins':>6} "
          f"{'Production':>12} {'Optimized':>12} {'vs Full Read':>12} {'New Crossover':>14}")
    print(f"  {'---':>7} {'---':>6} "
          f"{'---':>12} {'---':>12} {'---':>12} {'---':>14}")

    for label, n_bins, cad_ms in test_cases:
        fpath = Path(tmpdir) / f"test_{n_bins}.mktf"

        # Production: full read_progressive_level_data (includes file I/O + header)
        for _ in range(3):
            read_progressive_level_data(fpath, cad_ms, col_names)
        t_prod = timed_us(lambda: read_progressive_level_data(fpath, cad_ms, col_names), N)

        # Optimized: file I/O + header + bulk unpack
        def optimized_read():
            header = read_header(fpath)
            with open(fpath, "rb") as f:
                f.seek(header.progressive_offset)
                prog_data = f.read(header.progressive_size)
            _, _, levels, _ = unpack_progressive_directory(prog_data)
            for lv in levels:
                if lv.cadence_ms == cad_ms:
                    return fast_unpack_level_bulk(prog_data, lv, col_names)

        for _ in range(3):
            optimized_read()
        t_opt = timed_us(optimized_read, N)

        ratio = t_opt["mean"] / baseline_us
        crossover = "PROGRESSIVE" if ratio < 1.0 else "FULL READ"

        print(f"  {label:>7} {n_bins:>6} "
              f"{t_prod['mean']:>10.0f}us {t_opt['mean']:>10.0f}us "
              f"{ratio:>10.2f}x {'  <-- ' + crossover:>14}")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Correctness check
    # ═══════════════════════════════════════════════════════════
    print("PHASE 4: CORRECTNESS")
    print("-" * 60)

    # Verify fast reader matches production reader
    fpath_5min = Path(tmpdir) / "test_78.mktf"
    header = read_header(fpath_5min)
    with open(fpath_5min, "rb") as f:
        f.seek(header.progressive_offset)
        prog_data = f.read(header.progressive_size)
    _, _, levels, _ = unpack_progressive_directory(prog_data)
    level = levels[0]

    prod_result = unpack_progressive_level(prog_data, level, col_names)
    fast_result = fast_unpack_level_bulk(prog_data, level, col_names)

    all_match = True
    for name in col_names:
        if not np.array_equal(prod_result[name], fast_result[name]):
            print(f"  MISMATCH in column {name}")
            all_match = False
    if all_match:
        print(f"  All columns MATCH between production and optimized reader")

    # Also check 1s cadence
    fpath_1s = Path(tmpdir) / "test_23400.mktf"
    header = read_header(fpath_1s)
    with open(fpath_1s, "rb") as f:
        f.seek(header.progressive_offset)
        prog_data = f.read(header.progressive_size)
    _, _, levels, _ = unpack_progressive_directory(prog_data)
    level = levels[0]

    prod_result = unpack_progressive_level(prog_data, level, col_names)
    fast_result = fast_unpack_level_bulk(prog_data, level, col_names)

    all_match = True
    for name in col_names:
        if not np.array_equal(prod_result[name], fast_result[name]):
            print(f"  MISMATCH in column {name} (1s cadence)")
            all_match = False
    if all_match:
        print(f"  1s cadence (23,400 bins): MATCH")

    print()

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: Summary
    # ═══════════════════════════════════════════════════════════
    print("PHASE 5: SUMMARY")
    print("-" * 60)
    print("  The production progressive reader uses a nested Python loop")
    print("  (n_bins × n_cols iterations). The optimized version uses a single")
    print("  np.frombuffer + reshape — same result, no Python per-bin overhead.")
    print()
    print("  This is a pure Python optimization, not a Rust/C rewrite.")
    print("  If adopted, the crossover shifts dramatically upward,")
    print("  potentially making progressive faster than full reads at ALL cadences.")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
