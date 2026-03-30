"""Experiment 9: MKTF v4 (FinTek Production Spec) Benchmark.

Benchmarks the production v4 reader/writer from R:/fintek against the
v3 prototype. V4 adds:
  - Crash-safe atomic writes (2x fsync + .tmp rename)
  - Data checksum (SHA-256 over data region, written post-data)
  - EOF status bytes (is_complete + is_dirty mirrored at header[4094] and EOF[-2])
  - Richer header (13 sections: Tree, Temporal, Asset, Statistics, Spatial, ...)
  - verify_checksum() integrity verification path
  - scan_dirty / scan_incomplete daemon paths (1 byte read per file)

Questions:
  1. How much does crash safety (fsync + rename) cost vs v3's direct write?
  2. Does the richer header slow down unpack_block0?
  3. How fast is the EOF status scan vs v3's 4096-byte header scan?
  4. What does verify_checksum cost for real data?
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

# Import v4 from fintek
sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.format import MKTFHeader, UpstreamFingerprint
from trunk.backends.mktf.writer import write_mktf as write_v4, flip_dirty
from trunk.backends.mktf.reader import (
    read_header as read_header_v4,
    read_columns as read_columns_v4,
    read_selective as read_selective_v4,
    read_status as read_status_v4,
    is_dirty as is_dirty_v4,
    is_complete as is_complete_v4,
    verify_checksum as verify_checksum_v4,
    scan_dirty as scan_dirty_v4,
    scan_incomplete as scan_incomplete_v4,
)

# Import v3 from winrapids research
sys.path.insert(0, str(Path("R:/winrapids/research/20260327-mktf-format")))
from mktf_v3 import (
    write_mktf as write_v3,
    read_data as read_data_v3,
    read_header as read_header_v3,
    AAPL_PATH, COL_MAP, CONDITION_BITS,
)

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def load_source_columns() -> dict[str, np.ndarray]:
    """Load 5 source columns from real AAPL data."""
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
    """Run fn n_runs times, return timing stats in microseconds."""
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
        "max": float(np.max(arr)),
        "p50": float(np.median(arr)),
    }


def timed_ms(fn, n_runs: int) -> dict:
    """Run fn n_runs times, return timing stats in milliseconds."""
    gc.disable()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)  # ms
    gc.enable()
    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.median(arr)),
    }


def fmt_us(stats: dict) -> str:
    return f"{stats['mean']:.1f}us +/- {stats['std']:.1f}"


def fmt_ms(stats: dict) -> str:
    return f"{stats['mean']:.2f}ms +/- {stats['std']:.2f}"


# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

def main():
    N_WRITE = 10
    N_READ = 30
    N_SCAN = 50
    N_FILES_SCAN = 200  # for daemon scan benchmarks

    print("=" * 78)
    print("EXPERIMENT 9: MKTF v4 (FINTEK PRODUCTION) vs v3 (WINRAPIDS PROTOTYPE)")
    print("=" * 78)
    print(f"Write runs: {N_WRITE}, Read runs: {N_READ}, Scan runs: {N_SCAN}")
    print()

    cols = load_source_columns()
    n_rows = len(cols["price"])
    data_bytes = sum(arr.nbytes for arr in cols.values())
    print(f"Data: {n_rows} rows, {len(cols)} columns, {data_bytes/1e6:.2f} MB")
    print()

    tmpdir = tempfile.mkdtemp(prefix="mktf_v4_bench_")
    v3_path = os.path.join(tmpdir, "test_v3.mktf")
    v4_path = os.path.join(tmpdir, "test_v4.mktf")

    # ── Phase 1: Write comparison ────────────────────────────
    print("PHASE 1: WRITE COMPARISON (v3 vs v4)")
    print("-" * 60)

    # Warmup
    write_v3(v3_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02")
    write_v4(v4_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
             upstream=[UpstreamFingerprint(leaf_id="ingest", write_ts_ns=time.time_ns())])

    v3_size = os.path.getsize(v3_path)
    v4_size = os.path.getsize(v4_path)

    stats_w3 = timed_ms(
        lambda: write_v3(v3_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02"),
        N_WRITE)
    stats_w4 = timed_ms(
        lambda: write_v4(v4_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02",
                         upstream=[UpstreamFingerprint(leaf_id="ingest", write_ts_ns=time.time_ns())]),
        N_WRITE)

    print(f"  v3 write: {fmt_ms(stats_w3)}  file: {v3_size/1e6:.2f} MB")
    print(f"  v4 write: {fmt_ms(stats_w4)}  file: {v4_size/1e6:.2f} MB")
    print(f"  v4 overhead: {stats_w4['mean'] - stats_w3['mean']:.2f}ms "
          f"({stats_w4['mean']/stats_w3['mean']:.2f}x)")
    print(f"  v4 extras: 2x fsync, SHA-256 data checksum, atomic .tmp rename")
    print()

    # ── Phase 2: Read comparison ─────────────────────────────
    print("PHASE 2: READ COMPARISON (full read)")
    print("-" * 60)

    # Warmup
    _ = read_data_v3(v3_path)
    _ = read_columns_v4(v4_path)

    stats_r3 = timed_ms(lambda: read_data_v3(v3_path), N_READ)
    stats_r4 = timed_ms(lambda: read_columns_v4(v4_path), N_READ)

    print(f"  v3 full read: {fmt_ms(stats_r3)}")
    print(f"  v4 full read: {fmt_ms(stats_r4)}")
    print(f"  Delta: {stats_r4['mean'] - stats_r3['mean']:.2f}ms "
          f"({stats_r4['mean']/stats_r3['mean']:.2f}x)")
    print()

    # ── Phase 3: Header-only read ────────────────────────────
    print("PHASE 3: HEADER-ONLY READ (v3 vs v4)")
    print("-" * 60)

    # Warmup
    _ = read_header_v3(v3_path)
    _ = read_header_v4(v4_path)

    stats_h3 = timed_ns(lambda: read_header_v3(v3_path), N_READ)
    stats_h4 = timed_ns(lambda: read_header_v4(v4_path), N_READ)

    print(f"  v3 header: {fmt_us(stats_h3)}")
    print(f"  v4 header: {fmt_us(stats_h4)}")
    print(f"  v4 parses: 13 sections (Format, Identity, Tree, Dims, Temporal,")
    print(f"              Quality, Provenance, Layout, Asset, Upstream x16,")
    print(f"              Statistics, Spatial, Status)")
    print()

    # ── Phase 4: Selective read ──────────────────────────────
    print("PHASE 4: SELECTIVE READ (price + size only)")
    print("-" * 60)

    stats_sel4 = timed_ms(lambda: read_selective_v4(v4_path, ["price", "size"]), N_READ)
    print(f"  v4 selective (2 cols): {fmt_ms(stats_sel4)}")
    print()

    # ── Phase 5: EOF Status Fast Path ────────────────────────
    print("PHASE 5: EOF STATUS FAST PATH")
    print("-" * 60)

    # Warmup
    read_status_v4(v4_path)
    is_dirty_v4(v4_path)

    stats_status = timed_ns(lambda: read_status_v4(v4_path), N_SCAN)
    stats_isdirty = timed_ns(lambda: is_dirty_v4(v4_path), N_SCAN)
    stats_iscomplete = timed_ns(lambda: is_complete_v4(v4_path), N_SCAN)

    print(f"  read_status (2 bytes at EOF):  {fmt_us(stats_status)}")
    print(f"  is_dirty (1 byte at EOF):      {fmt_us(stats_isdirty)}")
    print(f"  is_complete (1 byte at EOF-1):  {fmt_us(stats_iscomplete)}")
    print(f"  vs v3 header scan:              {fmt_us(stats_h3)}")
    print(f"  Speedup: {stats_h3['mean']/stats_isdirty['mean']:.1f}x "
          f"(EOF byte vs full header parse)")
    print()

    # ── Phase 6: Data Integrity Verification ─────────────────
    print("PHASE 6: DATA INTEGRITY VERIFICATION")
    print("-" * 60)

    # Warmup
    verify_checksum_v4(v4_path)

    stats_verify = timed_ms(lambda: verify_checksum_v4(v4_path), N_READ)
    print(f"  verify_checksum: {fmt_ms(stats_verify)}")
    print(f"  (re-reads all data + SHA-256 hash + compare)")
    print(f"  Passes: {verify_checksum_v4(v4_path)}")
    print()

    # ── Phase 7: Daemon Scan at Scale ────────────────────────
    print(f"PHASE 7: DAEMON SCAN ({N_FILES_SCAN} files)")
    print("-" * 60)

    # Create N_FILES_SCAN v4 files
    scan_paths = []
    for i in range(N_FILES_SCAN):
        p = os.path.join(tmpdir, f"scan_{i:04d}.mktf")
        write_v4(p, cols, leaf_id="K01P01", ticker=f"T{i:04d}", day="2025-09-02")
        scan_paths.append(p)

    # Mark some as dirty for scan_dirty benchmark
    for p in scan_paths[:20]:
        flip_dirty(p, dirty=True)

    # Warmup
    scan_dirty_v4(scan_paths)
    scan_incomplete_v4(scan_paths)

    stats_scan_dirty = timed_ms(lambda: scan_dirty_v4(scan_paths), N_SCAN)
    stats_scan_incomplete = timed_ms(lambda: scan_incomplete_v4(scan_paths), N_SCAN)

    # Also benchmark header scan for comparison (v4 reader)
    stats_header_scan = timed_ms(
        lambda: [read_header_v4(p) for p in scan_paths], 5)

    n_dirty = len(scan_dirty_v4(scan_paths))
    n_incomplete = len(scan_incomplete_v4(scan_paths))

    print(f"  scan_dirty ({N_FILES_SCAN} files):      {fmt_ms(stats_scan_dirty)} "
          f"  ({stats_scan_dirty['mean']/N_FILES_SCAN*1000:.0f}us/file)")
    print(f"  scan_incomplete ({N_FILES_SCAN} files): {fmt_ms(stats_scan_incomplete)} "
          f"  ({stats_scan_incomplete['mean']/N_FILES_SCAN*1000:.0f}us/file)")
    print(f"  header_scan ({N_FILES_SCAN} files):     {fmt_ms(stats_header_scan)} "
          f"  ({stats_header_scan['mean']/N_FILES_SCAN*1000:.0f}us/file)")
    print(f"  Dirty found: {n_dirty}, Incomplete found: {n_incomplete}")
    print()

    # ── Phase 8: Universe Projections ────────────────────────
    print("PHASE 8: UNIVERSE PROJECTIONS (4,604 tickers)")
    print("=" * 60)
    n_tickers = 4604

    scan_dirty_per = stats_scan_dirty["mean"] / N_FILES_SCAN
    scan_inc_per = stats_scan_incomplete["mean"] / N_FILES_SCAN
    header_per = stats_header_scan["mean"] / N_FILES_SCAN
    read_per = stats_r4["mean"]

    print(f"  scan_dirty:       {scan_dirty_per*n_tickers:.0f}ms "
          f"= {scan_dirty_per*n_tickers/1000:.2f}s")
    print(f"  scan_incomplete:  {scan_inc_per*n_tickers:.0f}ms "
          f"= {scan_inc_per*n_tickers/1000:.2f}s")
    print(f"  header_scan:      {header_per*n_tickers:.0f}ms "
          f"= {header_per*n_tickers/1000:.2f}s")
    print(f"  full_read:        {read_per*n_tickers:.0f}ms "
          f"= {read_per*n_tickers/1000:.1f}s")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
