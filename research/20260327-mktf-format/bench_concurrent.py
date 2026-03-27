"""Concurrent MKTF reader — saturating NVMe bandwidth.

Navigator insight: We're at 2.5 GB/s actual vs 7 GB/s NVMe theoretical.
3-4 concurrent reads via ThreadPoolExecutor should close the gap.

This benchmark:
1. Creates N synthetic MKTF files (same real AAPL data, different paths)
2. Reads them sequentially (baseline)
3. Reads them concurrently with 1, 2, 4, 8, 16 workers
4. Adds GPU pipeline: concurrent read + H2D + fused kernel
5. Projects to full universe (4604 tickers)

The dream: full universe in ~10s instead of 22s.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cupy as cp
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mktf_v3 import write_mktf, read_data, AAPL_PATH, COL_MAP, CONDITION_BITS


# ══════════════════════════════════════════════════════════════════
# DATA & SETUP
# ══════════════════════════════════════════════════════════════════

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


def gpu_pipeline(data: dict[str, np.ndarray]) -> None:
    """Full GPU pipeline: H2D + fused kernel."""
    n = len(data["price"])
    p = cp.asarray(data["price"])
    s = cp.asarray(data["size"])
    # Allocate outputs
    ln_p = cp.empty(n, dtype=cp.float32)
    sqrt_p = cp.empty(n, dtype=cp.float32)
    recip_p = cp.empty(n, dtype=cp.float32)
    ntnl = cp.empty(n, dtype=cp.float32)
    ln_ntnl = cp.empty(n, dtype=cp.float32)
    threads = 256
    blocks = (n + threads - 1) // threads
    _FUSED_PW_F32((blocks,), (threads,), (p, s, ln_p, sqrt_p, recip_p, ntnl, ln_ntnl, n))


# ══════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("CONCURRENT MKTF READER BENCHMARK")
    print("=" * 78)
    print()

    # Create synthetic ticker files
    N_FILES = 100  # Simulate 100 tickers
    cols = load_source_columns()
    data_size = sum(arr.nbytes for arr in cols.values())

    tmpdir = tempfile.mkdtemp(prefix="mktf_bench_")
    paths = []
    for i in range(N_FILES):
        p = os.path.join(tmpdir, f"ticker_{i:04d}.mktf")
        write_mktf(p, cols,
                    leaf_id="K01P01", ticker=f"T{i:04d}", day="2025-09-02",
                    metadata={"mode": "source-only"})
        paths.append(p)

    file_size = os.path.getsize(paths[0])
    total_data = file_size * N_FILES
    print(f"Created {N_FILES} MKTF files, {file_size/1e6:.1f}MB each, "
          f"{total_data/1e6:.0f}MB total")
    print()

    # ── Sequential baseline ────────────────────────────────────
    # Warmup: read all files once to populate OS cache
    for p in paths:
        _ = read_data(p)

    print("SEQUENTIAL READ (baseline)")
    print("-" * 50)
    t0 = time.perf_counter()
    for p in paths:
        _ = read_data(p)
    t_seq = time.perf_counter() - t0
    bw_seq = total_data / t_seq / 1e9
    print(f"  {N_FILES} files: {t_seq*1000:.1f}ms ({t_seq/N_FILES*1000:.2f}ms/file)")
    print(f"  Bandwidth: {bw_seq:.2f} GB/s")
    print()

    # ── Concurrent reads ───────────────────────────────────────
    print("CONCURRENT READS (ThreadPoolExecutor)")
    print("-" * 50)

    for n_workers in [1, 2, 4, 8, 12, 16]:
        # Warmup
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            list(exe.map(read_data, paths))

        # Timed — 3 iterations for stability
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_workers) as exe:
                list(exe.map(read_data, paths))
            times.append(time.perf_counter() - t0)

        t_avg = np.mean(times)
        bw = total_data / t_avg / 1e9
        speedup = t_seq / t_avg
        print(f"  {n_workers:2d} workers: {t_avg*1000:>7.1f}ms ({t_avg/N_FILES*1000:.2f}ms/file) "
              f" BW: {bw:.2f} GB/s  Speedup: {speedup:.2f}x")

    print()

    # ── Sequential read + GPU pipeline ─────────────────────────
    print("SEQUENTIAL: Read + GPU Pipeline")
    print("-" * 50)

    # Warmup
    for p in paths[:5]:
        data = read_data(p)
        gpu_pipeline(data)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for p in paths:
        data = read_data(p)
        gpu_pipeline(data)
    cp.cuda.Stream.null.synchronize()
    t_seq_gpu = time.perf_counter() - t0
    print(f"  {N_FILES} files: {t_seq_gpu*1000:.1f}ms ({t_seq_gpu/N_FILES*1000:.2f}ms/file)")
    print()

    # ── Concurrent read + sequential GPU ───────────────────────
    # Pattern: prefetch next batch while GPU processes current batch
    print("PIPELINED: Concurrent prefetch + GPU")
    print("-" * 50)

    for n_workers in [2, 4, 8]:
        # Warmup
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            futs = [exe.submit(read_data, p) for p in paths[:10]]
            for f in futs:
                gpu_pipeline(f.result())
        cp.cuda.Stream.null.synchronize()

        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_workers) as exe:
                futs = [exe.submit(read_data, p) for p in paths]
                for f in as_completed(futs):
                    gpu_pipeline(f.result())
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)

        t_avg = np.mean(times)
        speedup = t_seq_gpu / t_avg
        print(f"  {n_workers:2d} prefetch workers: {t_avg*1000:>7.1f}ms "
              f"({t_avg/N_FILES*1000:.2f}ms/file)  Speedup: {speedup:.2f}x")

    print()

    # ── Universe projections ───────────────────────────────────
    print("UNIVERSE PROJECTIONS (4,604 tickers)")
    print("-" * 60)

    # Find best concurrent read time
    best_conc_per_file = None
    for n_workers in [4, 8]:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_workers) as exe:
                list(exe.map(read_data, paths))
            times.append(time.perf_counter() - t0)
        per_file = np.mean(times) / N_FILES
        if best_conc_per_file is None or per_file < best_conc_per_file:
            best_conc_per_file = per_file

    seq_per_file = t_seq / N_FILES
    seq_gpu_per_file = t_seq_gpu / N_FILES

    n_tickers = 4604
    print(f"  Sequential read:           {seq_per_file*n_tickers:.1f}s "
          f"= {seq_per_file*n_tickers/60:.1f}min")
    print(f"  Sequential read+GPU:       {seq_gpu_per_file*n_tickers:.1f}s "
          f"= {seq_gpu_per_file*n_tickers/60:.1f}min")
    print(f"  Concurrent read (best):    {best_conc_per_file*n_tickers:.1f}s "
          f"= {best_conc_per_file*n_tickers/60:.1f}min")

    # Theoretical NVMe-saturated
    nvme_bw = 7e9  # 7 GB/s theoretical
    t_nvme = file_size * n_tickers / nvme_bw
    print(f"  NVMe theoretical ({nvme_bw/1e9:.0f}GB/s): {t_nvme:.1f}s "
          f"= {t_nvme/60:.1f}min")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
