"""Experiment 11: MKTF v4 vs Parquet — The Production Comparison.

Team-lead's request:
  1. MKTF v4 write vs parquet write
  2. MKTF selective read vs polars scan_parquet with projection pushdown
  3. Header-only read timing (Block 0 + directory)
  4. Status byte read timing (EOF-1, cached)

This is the benchmark that justifies the format migration.
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

# fintek v4
sys.path.insert(0, "R:/fintek")
from trunk.backends.mktf.writer import write_mktf as write_v4
from trunk.backends.mktf.reader import (
    read_header as read_header_v4,
    read_columns as read_columns_v4,
    read_selective as read_selective_v4,
    read_status as read_status_v4,
    is_dirty as is_dirty_v4,
    is_complete as is_complete_v4,
    verify_checksum as verify_checksum_v4,
)

# winrapids data source
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


def timed(fn, n_runs: int, unit: str = "ms") -> dict:
    gc.disable()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        fn()
        ns = time.perf_counter_ns() - t0
        if unit == "us":
            times.append(ns / 1e3)
        else:
            times.append(ns / 1e6)
    gc.enable()
    arr = np.array(times)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.median(arr)),
        "unit": unit,
    }


def fmt(stats: dict) -> str:
    u = stats["unit"]
    return f"{stats['mean']:.1f}{u} +/- {stats['std']:.1f}"


def main():
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    N_WRITE = 10
    N_READ = 30
    N_FAST = 100  # for sub-microsecond ops

    print("=" * 78)
    print("EXPERIMENT 11: MKTF v4 vs PARQUET — PRODUCTION COMPARISON")
    print("=" * 78)
    print(f"Polars {pl.__version__}, PyArrow {pa.__version__}")
    print()

    cols = load_source_columns()
    n_rows = len(cols["price"])
    data_bytes = sum(arr.nbytes for arr in cols.values())
    print(f"Data: {n_rows:,} rows, {len(cols)} columns, {data_bytes/1e6:.2f} MB")
    print()

    tmpdir = tempfile.mkdtemp(prefix="mktf_vs_pq_")
    mktf_path = os.path.join(tmpdir, "test.mktf")
    pq_none_path = os.path.join(tmpdir, "test_none.parquet")
    pq_snappy_path = os.path.join(tmpdir, "test_snappy.parquet")
    pq_zstd_path = os.path.join(tmpdir, "test_zstd.parquet")

    # Build pyarrow table from our columns
    pa_table = pa.table({
        name: pa.array(arr) for name, arr in cols.items()
    })

    # ── Phase 1: Write comparison ────────────────────────────
    print("PHASE 1: WRITE SPEED")
    print("-" * 60)

    # Warmup
    write_v4(mktf_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02")
    pq.write_table(pa_table, pq_none_path, compression="none")
    pq.write_table(pa_table, pq_snappy_path, compression="snappy")
    pq.write_table(pa_table, pq_zstd_path, compression="zstd")

    stats_w_mktf = timed(
        lambda: write_v4(mktf_path, cols, leaf_id="K01P01", ticker="AAPL", day="2025-09-02"),
        N_WRITE)
    stats_w_pq_none = timed(
        lambda: pq.write_table(pa_table, pq_none_path, compression="none"),
        N_WRITE)
    stats_w_pq_snappy = timed(
        lambda: pq.write_table(pa_table, pq_snappy_path, compression="snappy"),
        N_WRITE)
    stats_w_pq_zstd = timed(
        lambda: pq.write_table(pa_table, pq_zstd_path, compression="zstd"),
        N_WRITE)

    mktf_size = os.path.getsize(mktf_path)
    pq_none_size = os.path.getsize(pq_none_path)
    pq_snappy_size = os.path.getsize(pq_snappy_path)
    pq_zstd_size = os.path.getsize(pq_zstd_path)

    print(f"  {'Format':<25} {'Write':>20} {'Size':>12}")
    print(f"  {'-'*25} {'-'*20} {'-'*12}")
    print(f"  {'MKTF v4 (crash-safe)':<25} {fmt(stats_w_mktf):>20} {mktf_size/1e6:>10.2f} MB")
    print(f"  {'Parquet (none)':<25} {fmt(stats_w_pq_none):>20} {pq_none_size/1e6:>10.2f} MB")
    print(f"  {'Parquet (snappy)':<25} {fmt(stats_w_pq_snappy):>20} {pq_snappy_size/1e6:>10.2f} MB")
    print(f"  {'Parquet (zstd)':<25} {fmt(stats_w_pq_zstd):>20} {pq_zstd_size/1e6:>10.2f} MB")
    print()
    print(f"  Note: MKTF v4 includes 2x fsync + SHA-256 checksum + atomic rename.")
    print(f"  Parquet writes are NOT crash-safe (no fsync, no atomic rename).")
    print()

    # ── Phase 2: Full read comparison ────────────────────────
    print("PHASE 2: FULL READ")
    print("-" * 60)

    # Warmup
    _ = read_columns_v4(mktf_path)
    _ = pq.read_table(pq_none_path)
    _ = pl.read_parquet(pq_none_path)

    stats_r_mktf = timed(lambda: read_columns_v4(mktf_path), N_READ)
    stats_r_pq_none = timed(lambda: pq.read_table(pq_none_path), N_READ)
    stats_r_pq_snappy = timed(lambda: pq.read_table(pq_snappy_path), N_READ)
    stats_r_pq_zstd = timed(lambda: pq.read_table(pq_zstd_path), N_READ)
    stats_r_pl_none = timed(lambda: pl.read_parquet(pq_none_path), N_READ)
    stats_r_pl_snappy = timed(lambda: pl.read_parquet(pq_snappy_path), N_READ)

    print(f"  {'Reader':<30} {'Read':>20}")
    print(f"  {'-'*30} {'-'*20}")
    print(f"  {'MKTF v4 (numpy arrays)':<30} {fmt(stats_r_mktf):>20}")
    print(f"  {'PyArrow parquet (none)':<30} {fmt(stats_r_pq_none):>20}")
    print(f"  {'PyArrow parquet (snappy)':<30} {fmt(stats_r_pq_snappy):>20}")
    print(f"  {'PyArrow parquet (zstd)':<30} {fmt(stats_r_pq_zstd):>20}")
    print(f"  {'Polars parquet (none)':<30} {fmt(stats_r_pl_none):>20}")
    print(f"  {'Polars parquet (snappy)':<30} {fmt(stats_r_pl_snappy):>20}")
    print()
    print(f"  MKTF speedup vs PyArrow(none): {stats_r_pq_none['mean']/stats_r_mktf['mean']:.1f}x")
    print(f"  MKTF speedup vs Polars(none):  {stats_r_pl_none['mean']/stats_r_mktf['mean']:.1f}x")
    print()

    # ── Phase 3: Selective read (projection pushdown) ────────
    print("PHASE 3: SELECTIVE READ — price + size (2 of 5 columns)")
    print("-" * 60)

    sel_cols = ["price", "size"]

    # Warmup
    _ = read_selective_v4(mktf_path, sel_cols)
    _ = pq.read_table(pq_none_path, columns=sel_cols)
    _ = pl.scan_parquet(pq_none_path).select(sel_cols).collect()

    stats_sel_mktf = timed(lambda: read_selective_v4(mktf_path, sel_cols), N_READ)
    stats_sel_pq_none = timed(
        lambda: pq.read_table(pq_none_path, columns=sel_cols), N_READ)
    stats_sel_pq_snappy = timed(
        lambda: pq.read_table(pq_snappy_path, columns=sel_cols), N_READ)
    stats_sel_pl_scan = timed(
        lambda: pl.scan_parquet(pq_none_path).select(sel_cols).collect(), N_READ)
    stats_sel_pl_scan_snappy = timed(
        lambda: pl.scan_parquet(pq_snappy_path).select(sel_cols).collect(), N_READ)

    print(f"  {'Reader':<35} {'Read':>20}")
    print(f"  {'-'*35} {'-'*20}")
    print(f"  {'MKTF v4 selective (seek per col)':<35} {fmt(stats_sel_mktf):>20}")
    print(f"  {'PyArrow parquet(none) columns=':<35} {fmt(stats_sel_pq_none):>20}")
    print(f"  {'PyArrow parquet(snappy) columns=':<35} {fmt(stats_sel_pq_snappy):>20}")
    print(f"  {'Polars scan_parquet(none) .select':<35} {fmt(stats_sel_pl_scan):>20}")
    print(f"  {'Polars scan_parquet(snappy) .sel':<35} {fmt(stats_sel_pl_scan_snappy):>20}")
    print()
    print(f"  MKTF speedup vs PyArrow sel:  {stats_sel_pq_none['mean']/stats_sel_mktf['mean']:.1f}x")
    print(f"  MKTF speedup vs Polars scan:  {stats_sel_pl_scan['mean']/stats_sel_mktf['mean']:.1f}x")
    print()

    # ── Phase 4: Header-only read ────────────────────────────
    print("PHASE 4: HEADER-ONLY READ (Block 0 + directory)")
    print("-" * 60)

    # Warmup
    read_header_v4(mktf_path)

    stats_header = timed(lambda: read_header_v4(mktf_path), N_FAST, unit="us")
    # Also time parquet metadata read for comparison
    stats_pq_meta = timed(lambda: pq.read_metadata(pq_none_path), N_FAST, unit="us")

    print(f"  MKTF v4 header:       {fmt(stats_header)}")
    print(f"  Parquet metadata:     {fmt(stats_pq_meta)}")
    print(f"  Speedup: {stats_pq_meta['mean']/stats_header['mean']:.1f}x")
    print()

    h = read_header_v4(mktf_path)
    print(f"  Header contents: {h.n_rows} rows, {h.n_cols} cols, "
          f"ticker={h.ticker}, day={h.day}")
    print(f"  Columns: {[c.name for c in h.columns]}")
    print(f"  is_complete={h.is_complete}, is_dirty={h.is_dirty}")
    print()

    # ── Phase 5: Status byte read (the daemon fast path) ────
    print("PHASE 5: STATUS BYTE READ (EOF seek, zero parsing)")
    print("-" * 60)

    stats_status = timed(lambda: read_status_v4(mktf_path), N_FAST, unit="us")
    stats_dirty = timed(lambda: is_dirty_v4(mktf_path), N_FAST, unit="us")
    stats_complete = timed(lambda: is_complete_v4(mktf_path), N_FAST, unit="us")

    print(f"  read_status (2 bytes):  {fmt(stats_status)}")
    print(f"  is_dirty (1 byte):      {fmt(stats_dirty)}")
    print(f"  is_complete (1 byte):   {fmt(stats_complete)}")
    print()

    # Compare: how many status checks per second?
    checks_per_sec = 1_000_000 / stats_dirty["mean"]  # us → checks/sec
    universe_scan = stats_dirty["mean"] * 4604 / 1e6  # seconds
    print(f"  Throughput: {checks_per_sec:,.0f} status checks/second")
    print(f"  Full universe (4604): {universe_scan*1000:.0f}ms = {universe_scan:.2f}s")
    print()

    # ── Phase 6: GPU Pipeline comparison ─────────────────────
    print("PHASE 6: GPU PIPELINE (read → H2D → fused kernel)")
    print("-" * 60)

    import cupy as cp

    _FUSED_PW_F32 = cp.RawKernel(r'''
    extern "C" __global__
    void fused_pw_f32(
        const float* __restrict__ price,
        const float* __restrict__ size,
        float* __restrict__ ln_price,
        float* __restrict__ sqrt_price,
        float* __restrict__ recip_price,
        float* __restrict__ notional,
        float* __restrict__ ln_notional,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        float p = price[idx];
        float s = size[idx];
        ln_price[idx] = logf(fmaxf(p, 1e-38f));
        sqrt_price[idx] = sqrtf(fmaxf(p, 0.0f));
        recip_price[idx] = 1.0f / p;
        float ntnl = p * s;
        notional[idx] = ntnl;
        ln_notional[idx] = logf(fmaxf(ntnl, 1e-38f));
    }
    ''', 'fused_pw_f32')

    def mktf_gpu_pipeline():
        _, data = read_columns_v4(mktf_path)
        n = len(data["price"])
        p = cp.asarray(data["price"])
        s = cp.asarray(data["size"])
        ln_p = cp.empty(n, dtype=cp.float32)
        sqrt_p = cp.empty(n, dtype=cp.float32)
        recip_p = cp.empty(n, dtype=cp.float32)
        ntnl = cp.empty(n, dtype=cp.float32)
        ln_ntnl = cp.empty(n, dtype=cp.float32)
        threads = 256
        blocks = (n + threads - 1) // threads
        _FUSED_PW_F32((blocks,), (threads,), (p, s, ln_p, sqrt_p, recip_p, ntnl, ln_ntnl, n))

    def parquet_gpu_pipeline():
        tbl = pq.read_table(pq_none_path)
        price = tbl.column("price").to_numpy()
        size = tbl.column("size").to_numpy()
        n = len(price)
        p = cp.asarray(price)
        s = cp.asarray(size)
        ln_p = cp.empty(n, dtype=cp.float32)
        sqrt_p = cp.empty(n, dtype=cp.float32)
        recip_p = cp.empty(n, dtype=cp.float32)
        ntnl = cp.empty(n, dtype=cp.float32)
        ln_ntnl = cp.empty(n, dtype=cp.float32)
        threads = 256
        blocks = (n + threads - 1) // threads
        _FUSED_PW_F32((blocks,), (threads,), (p, s, ln_p, sqrt_p, recip_p, ntnl, ln_ntnl, n))

    # Warmup
    mktf_gpu_pipeline(); cp.cuda.Stream.null.synchronize()
    parquet_gpu_pipeline(); cp.cuda.Stream.null.synchronize()

    stats_mktf_gpu = timed(lambda: (mktf_gpu_pipeline(), cp.cuda.Stream.null.synchronize()), N_READ)
    stats_pq_gpu = timed(lambda: (parquet_gpu_pipeline(), cp.cuda.Stream.null.synchronize()), N_READ)

    print(f"  MKTF v4 → GPU:    {fmt(stats_mktf_gpu)}")
    print(f"  Parquet → GPU:    {fmt(stats_pq_gpu)}")
    print(f"  Speedup: {stats_pq_gpu['mean']/stats_mktf_gpu['mean']:.1f}x")
    print()

    # Universe projections
    print("UNIVERSE PROJECTIONS (4604 tickers)")
    print("=" * 60)
    n = 4604
    print(f"  MKTF v4 full read:     {stats_r_mktf['mean']*n/1000:.1f}s")
    print(f"  Parquet(none) read:    {stats_r_pq_none['mean']*n/1000:.1f}s")
    print(f"  Polars(none) read:     {stats_r_pl_none['mean']*n/1000:.1f}s")
    print(f"  MKTF v4 → GPU:        {stats_mktf_gpu['mean']*n/1000:.1f}s")
    print(f"  Parquet → GPU:         {stats_pq_gpu['mean']*n/1000:.1f}s")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
