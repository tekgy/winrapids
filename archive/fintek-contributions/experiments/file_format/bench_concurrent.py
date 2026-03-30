"""Experiment 7: Concurrent MKTF Reader — NVMe Saturation Benchmark.

Validates pathmaker's concurrent reader prototype with:
- Statistical significance (multiple runs, mean/std/p50/p95)
- Cold cache vs warm cache
- Worker sweep (1-16 threads)
- Pipelined read+GPU vs sequential
- Full universe projections

Navigator insight: 2.5 GB/s actual vs 7 GB/s NVMe theoretical.
Target: saturate NVMe with concurrent reads, full universe < 10s.
"""

from __future__ import annotations

import gc
import io
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cupy as cp
import numpy as np

# UTF-8 stdout for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "20260327-mktf-format"))
from mktf_v3 import write_mktf, read_data, AAPL_PATH, COL_MAP, CONDITION_BITS


# ═══════════════════════════════════════════════════════════════
# DATA SETUP
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
    ln_p = cp.empty(n, dtype=cp.float32)
    sqrt_p = cp.empty(n, dtype=cp.float32)
    recip_p = cp.empty(n, dtype=cp.float32)
    ntnl = cp.empty(n, dtype=cp.float32)
    ln_ntnl = cp.empty(n, dtype=cp.float32)
    threads = 256
    blocks = (n + threads - 1) // threads
    _FUSED_PW_F32((blocks,), (threads,), (p, s, ln_p, sqrt_p, recip_p, ntnl, ln_ntnl, n))


def flush_os_cache():
    """Pressure-based OS cache eviction on Windows."""
    try:
        _ = np.ones(int(2e9 / 8), dtype=np.float64)  # 2 GB allocation
        del _
    except MemoryError:
        pass


def timed_runs(fn, n_runs: int) -> dict:
    """Run fn n_runs times, return timing stats."""
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
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

def main():
    N_FILES = 100
    N_WARM = 10
    N_COLD = 5
    N_GPU = 10

    print("=" * 78)
    print("EXPERIMENT 7: CONCURRENT MKTF READER — NVMe SATURATION")
    print("=" * 78)
    print(f"Files: {N_FILES}, Warm runs: {N_WARM}, Cold runs: {N_COLD}, GPU runs: {N_GPU}")
    print()

    # Create synthetic ticker files
    cols = load_source_columns()
    tmpdir = tempfile.mkdtemp(prefix="mktf_bench_conc_")
    paths = []
    for i in range(N_FILES):
        p = os.path.join(tmpdir, f"ticker_{i:04d}.mktf")
        write_mktf(p, cols,
                    leaf_id="K01P01", ticker=f"T{i:04d}", day="2025-09-02",
                    metadata={"mode": "source-only"})
        paths.append(p)

    file_size = os.path.getsize(paths[0])
    total_bytes = file_size * N_FILES
    print(f"Created {N_FILES} files, {file_size/1e6:.2f} MB each, {total_bytes/1e6:.0f} MB total")
    print()

    # ── Phase 1: Sequential baseline (warm) ──────────────────
    print("PHASE 1: SEQUENTIAL BASELINE (WARM)")
    print("-" * 60)

    # Warmup
    for p in paths:
        _ = read_data(p)

    stats_seq = timed_runs(lambda: [read_data(p) for p in paths], N_WARM)
    bw_seq = total_bytes / (stats_seq["mean"] / 1000) / 1e9
    print(f"  Mean: {stats_seq['mean']:.1f}ms +/- {stats_seq['std']:.1f}ms "
          f"({stats_seq['mean']/N_FILES:.2f} ms/file)")
    print(f"  p50: {stats_seq['p50']:.1f}ms  p95: {stats_seq['p95']:.1f}ms  "
          f"min: {stats_seq['min']:.1f}ms  max: {stats_seq['max']:.1f}ms")
    print(f"  Bandwidth: {bw_seq:.2f} GB/s")
    print()

    # ── Phase 2: Concurrent read sweep (warm) ────────────────
    print("PHASE 2: CONCURRENT READ SWEEP (WARM)")
    print("-" * 60)

    worker_results = {}
    for n_w in [1, 2, 4, 8, 12, 16]:
        # Warmup
        with ThreadPoolExecutor(max_workers=n_w) as exe:
            list(exe.map(read_data, paths))

        def _conc_read(nw=n_w):
            with ThreadPoolExecutor(max_workers=nw) as exe:
                list(exe.map(read_data, paths))

        stats = timed_runs(_conc_read, N_WARM)
        bw = total_bytes / (stats["mean"] / 1000) / 1e9
        speedup = stats_seq["mean"] / stats["mean"]
        worker_results[n_w] = stats
        print(f"  {n_w:2d} workers: {stats['mean']:>7.1f}ms +/- {stats['std']:.1f}ms "
              f"({stats['mean']/N_FILES:.2f} ms/file)  "
              f"BW: {bw:.2f} GB/s  Speedup: {speedup:.2f}x")

    print()

    # ── Phase 3: Cold cache ──────────────────────────────────
    print("PHASE 3: COLD CACHE (memory-pressure eviction)")
    print("-" * 60)

    cold_seq_times = []
    cold_4w_times = []
    for i in range(N_COLD):
        # Sequential cold
        flush_os_cache()
        gc.disable()
        t0 = time.perf_counter_ns()
        for p in paths:
            _ = read_data(p)
        cold_seq_times.append((time.perf_counter_ns() - t0) / 1e6)
        gc.enable()

        # Concurrent cold (4 workers — sweet spot from warm results)
        flush_os_cache()
        gc.disable()
        t0 = time.perf_counter_ns()
        with ThreadPoolExecutor(max_workers=4) as exe:
            list(exe.map(read_data, paths))
        cold_4w_times.append((time.perf_counter_ns() - t0) / 1e6)
        gc.enable()

    seq_cold = np.array(cold_seq_times)
    conc_cold = np.array(cold_4w_times)
    print(f"  Sequential cold: {np.mean(seq_cold):.1f}ms +/- {np.std(seq_cold):.1f}ms "
          f"({np.mean(seq_cold)/N_FILES:.2f} ms/file)")
    print(f"  4-worker cold:   {np.mean(conc_cold):.1f}ms +/- {np.std(conc_cold):.1f}ms "
          f"({np.mean(conc_cold)/N_FILES:.2f} ms/file)")
    print(f"  Cold speedup:    {np.mean(seq_cold)/np.mean(conc_cold):.2f}x")
    print(f"  Cold penalty (seq):  {np.mean(seq_cold)/stats_seq['mean']:.2f}x vs warm")
    print(f"  Cold penalty (conc): {np.mean(conc_cold)/worker_results[4]['mean']:.2f}x vs warm")
    print()

    # ── Phase 4: Pipelined read + GPU ────────────────────────
    print("PHASE 4: PIPELINED READ + GPU (prefetch + fused kernel)")
    print("-" * 60)

    # Warmup GPU
    for p in paths[:5]:
        gpu_pipeline(read_data(p))
    cp.cuda.Stream.null.synchronize()

    # Sequential read + GPU baseline
    def _seq_gpu():
        for p in paths:
            gpu_pipeline(read_data(p))
        cp.cuda.Stream.null.synchronize()

    stats_seq_gpu = timed_runs(_seq_gpu, N_GPU)
    print(f"  Sequential read+GPU: {stats_seq_gpu['mean']:.1f}ms +/- {stats_seq_gpu['std']:.1f}ms "
          f"({stats_seq_gpu['mean']/N_FILES:.2f} ms/file)")

    for n_w in [2, 4, 8]:
        # Warmup
        with ThreadPoolExecutor(max_workers=n_w) as exe:
            for f in as_completed([exe.submit(read_data, p) for p in paths[:10]]):
                gpu_pipeline(f.result())
        cp.cuda.Stream.null.synchronize()

        def _pipe_gpu(nw=n_w):
            with ThreadPoolExecutor(max_workers=nw) as exe:
                for f in as_completed([exe.submit(read_data, p) for p in paths]):
                    gpu_pipeline(f.result())
            cp.cuda.Stream.null.synchronize()

        stats = timed_runs(_pipe_gpu, N_GPU)
        speedup = stats_seq_gpu["mean"] / stats["mean"]
        print(f"  {n_w:2d}-worker pipeline: {stats['mean']:>7.1f}ms +/- {stats['std']:.1f}ms "
              f"({stats['mean']/N_FILES:.2f} ms/file)  Speedup: {speedup:.2f}x")

    print()

    # ── Phase 5: Universe projections ────────────────────────
    print("PHASE 5: UNIVERSE PROJECTIONS (4,604 tickers)")
    print("=" * 60)

    n_tickers = 4604
    seq_pf = stats_seq["mean"] / N_FILES
    best_w = min(worker_results, key=lambda k: worker_results[k]["mean"])
    conc_pf = worker_results[best_w]["mean"] / N_FILES
    seq_gpu_pf = stats_seq_gpu["mean"] / N_FILES
    cold_seq_pf = np.mean(seq_cold) / N_FILES
    cold_conc_pf = np.mean(conc_cold) / N_FILES
    nvme_theoretical = file_size * n_tickers / 7e9  # 7 GB/s

    print(f"  Sequential warm read:      {seq_pf*n_tickers/1000:.1f}s")
    print(f"  Concurrent warm ({best_w}w):     {conc_pf*n_tickers/1000:.1f}s")
    print(f"  Sequential cold read:      {cold_seq_pf*n_tickers/1000:.1f}s")
    print(f"  Concurrent cold (4w):      {cold_conc_pf*n_tickers/1000:.1f}s")
    print(f"  Sequential warm read+GPU:  {seq_gpu_pf*n_tickers/1000:.1f}s")
    print(f"  NVMe theoretical (7GB/s):  {nvme_theoretical:.1f}s")
    print()
    print(f"  Best concurrent per-file:  {conc_pf:.3f}ms ({best_w} workers)")
    print(f"  Sequential per-file:       {seq_pf:.3f}ms")
    print(f"  Speedup:                   {seq_pf/conc_pf:.2f}x")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
