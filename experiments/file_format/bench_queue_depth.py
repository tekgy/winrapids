"""Experiment 8: NVMe Queue Depth Sweep — Finding the Saturation Knee.

Naturalist insight: NVMe throughput is heavily queue-depth dependent.
Experiment 7 only tested 1-16 workers. Theory says QD16-32 captures
77-85% of peak Gen5 bandwidth, but our actual drive may differ.

Sweep: 1, 2, 4, 8, 16, 32, 64 concurrent readers.
Goal: find the saturation knee for our specific NVMe drive.
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "20260327-mktf-format"))
from mktf_v3 import write_mktf, read_data, AAPL_PATH, COL_MAP, CONDITION_BITS


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


def main():
    # Use 200 files to give higher worker counts enough I/O to overlap
    N_FILES = 200
    N_RUNS = 5

    print("=" * 78)
    print("EXPERIMENT 8: NVMe QUEUE DEPTH SWEEP")
    print("=" * 78)
    print(f"Files: {N_FILES}, Runs per config: {N_RUNS}")
    print()

    cols = load_source_columns()
    tmpdir = tempfile.mkdtemp(prefix="mktf_qd_")
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

    # ── Phase 1: Read-only sweep ─────────────────────────────
    print("PHASE 1: READ-ONLY QUEUE DEPTH SWEEP")
    print("-" * 60)
    print(f"{'Workers':>8} {'Mean (ms)':>12} {'Std':>8} {'ms/file':>10} "
          f"{'BW (GB/s)':>12} {'Speedup':>10}")
    print("-" * 60)

    # Warmup all files into OS cache
    for p in paths:
        _ = read_data(p)

    # Sequential baseline
    gc.disable()
    seq_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter_ns()
        for p in paths:
            _ = read_data(p)
        seq_times.append((time.perf_counter_ns() - t0) / 1e6)
    gc.enable()
    seq_mean = np.mean(seq_times)
    seq_std = np.std(seq_times)
    seq_bw = total_bytes / (seq_mean / 1000) / 1e9
    print(f"{'seq':>8} {seq_mean:>12.1f} {seq_std:>8.1f} {seq_mean/N_FILES:>10.2f} "
          f"{seq_bw:>12.2f} {'1.00x':>10}")

    results = {}
    for n_w in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]:
        # Warmup
        with ThreadPoolExecutor(max_workers=n_w) as exe:
            list(exe.map(read_data, paths))

        gc.disable()
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=n_w) as exe:
                list(exe.map(read_data, paths))
            times.append((time.perf_counter_ns() - t0) / 1e6)
        gc.enable()

        mean_t = np.mean(times)
        std_t = np.std(times)
        bw = total_bytes / (mean_t / 1000) / 1e9
        speedup = seq_mean / mean_t
        results[n_w] = {"mean": mean_t, "std": std_t, "bw": bw, "speedup": speedup}
        print(f"{n_w:>8} {mean_t:>12.1f} {std_t:>8.1f} {mean_t/N_FILES:>10.2f} "
              f"{bw:>12.2f} {speedup:>9.2f}x")

    print()

    # ── Phase 2: Pipelined read+GPU sweep ────────────────────
    print("PHASE 2: PIPELINED READ+GPU SWEEP")
    print("-" * 60)

    # Warmup GPU
    for p in paths[:5]:
        gpu_pipeline(read_data(p))
    cp.cuda.Stream.null.synchronize()

    # Sequential baseline
    gc.disable()
    seq_gpu_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter_ns()
        for p in paths:
            gpu_pipeline(read_data(p))
        cp.cuda.Stream.null.synchronize()
        seq_gpu_times.append((time.perf_counter_ns() - t0) / 1e6)
    gc.enable()
    seq_gpu_mean = np.mean(seq_gpu_times)

    print(f"{'Workers':>8} {'Mean (ms)':>12} {'Std':>8} {'ms/file':>10} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'seq':>8} {seq_gpu_mean:>12.1f} {np.std(seq_gpu_times):>8.1f} "
          f"{seq_gpu_mean/N_FILES:>10.2f} {'1.00x':>10}")

    for n_w in [2, 4, 8, 12, 16, 24, 32, 48, 64]:
        # Warmup
        with ThreadPoolExecutor(max_workers=n_w) as exe:
            for f in as_completed([exe.submit(read_data, p) for p in paths[:10]]):
                gpu_pipeline(f.result())
        cp.cuda.Stream.null.synchronize()

        gc.disable()
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter_ns()
            with ThreadPoolExecutor(max_workers=n_w) as exe:
                for f in as_completed([exe.submit(read_data, p) for p in paths]):
                    gpu_pipeline(f.result())
            cp.cuda.Stream.null.synchronize()
            times.append((time.perf_counter_ns() - t0) / 1e6)
        gc.enable()

        mean_t = np.mean(times)
        std_t = np.std(times)
        speedup = seq_gpu_mean / mean_t
        print(f"{n_w:>8} {mean_t:>12.1f} {std_t:>8.1f} {mean_t/N_FILES:>10.2f} "
              f"{speedup:>9.2f}x")

    print()

    # ── Phase 3: Universe projections ────────────────────────
    print("PHASE 3: UNIVERSE PROJECTIONS (4,604 tickers)")
    print("=" * 60)
    n_tickers = 4604

    print(f"  Sequential read:       {seq_mean/N_FILES*n_tickers/1000:.1f}s")
    best_w = min(results, key=lambda k: results[k]["mean"])
    best_pf = results[best_w]["mean"] / N_FILES
    print(f"  Best concurrent ({best_w}w):  {best_pf*n_tickers/1000:.1f}s "
          f"@ {results[best_w]['bw']:.2f} GB/s")
    print(f"  NVMe theoretical (7GB/s): {file_size*n_tickers/7e9:.1f}s")
    print()

    # Saturation analysis
    print("SATURATION ANALYSIS")
    print("-" * 60)
    prev_bw = seq_bw
    for n_w in sorted(results.keys()):
        r = results[n_w]
        marginal = (r["bw"] - prev_bw) / prev_bw * 100 if prev_bw > 0 else 0
        pct_theoretical = r["bw"] / 7.0 * 100
        print(f"  {n_w:>3}w: {r['bw']:.2f} GB/s ({pct_theoretical:.0f}% theoretical) "
              f"  marginal: {marginal:+.1f}%")
        prev_bw = r["bw"]

    shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
