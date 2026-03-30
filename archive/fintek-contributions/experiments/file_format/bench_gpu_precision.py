"""Observer Experiment 5: GPU Precision & Throughput — FP64 vs FP32 vs INT encoding.

Benchmarks:
1. Fused pointwise kernel: FP64 vs FP32 (log, sqrt, reciprocal, abs, sign, etc.)
2. Precision validation: max abs/rel error of FP32 vs FP64 on real AAPL data
3. "Don't store derivables" pipeline: store base cols + GPU recompute vs store all
4. Kahan compensated summation: FP32 naive vs FP32 Kahan vs FP64
5. Integer-only pipeline: int64 scaled + GPU compute without float conversion

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q
"""
from __future__ import annotations

import gc
import io
import os
import sys
import tempfile
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "research", "20260327-mktf-format"))

import numpy as np
import cupy as cp

import pyarrow.parquet as pq
from mktf_v2 import write_mktf, read_mktf, load_and_encode, PRICE_SCALE


# ══════════════════════════════════════════════════════════════════
# CUDA KERNELS
# ══════════════════════════════════════════════════════════════════

# Fused pointwise kernel — computes ALL derived columns in one launch
# Inputs: price, size, timestamp (3 arrays)
# Outputs: notional, ln_price, sqrt_price, recip_price, abs_return,
#          sign_return, ln_size, sqrt_size, recip_size,
#          price_x_sqrt_size, vwap_contrib, spread_proxy, momentum
# That's 13 outputs from 3 inputs — matches the K01 fused kernel spec

FUSED_POINTWISE_FP64 = cp.RawKernel(r"""
extern "C" __global__
void fused_pointwise_fp64(
    const double* __restrict__ price,
    const double* __restrict__ size,
    const long long* __restrict__ timestamp,
    double* __restrict__ notional,
    double* __restrict__ ln_price,
    double* __restrict__ sqrt_price,
    double* __restrict__ recip_price,
    double* __restrict__ abs_return,
    double* __restrict__ sign_return,
    double* __restrict__ ln_size,
    double* __restrict__ sqrt_size,
    double* __restrict__ recip_size,
    double* __restrict__ price_x_sqrt_size,
    double* __restrict__ vwap_contrib,
    double* __restrict__ spread_proxy,
    double* __restrict__ momentum,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double p = price[idx];
    double s = size[idx];

    notional[idx] = p * s;
    ln_price[idx] = log(p);
    sqrt_price[idx] = sqrt(p);
    recip_price[idx] = 1.0 / p;

    // Return vs previous (idx > 0)
    double ret = (idx > 0) ? (p - price[idx - 1]) / price[idx - 1] : 0.0;
    abs_return[idx] = fabs(ret);
    sign_return[idx] = (ret > 0.0) ? 1.0 : ((ret < 0.0) ? -1.0 : 0.0);

    ln_size[idx] = log(s + 1.0);  // +1 to avoid log(0)
    sqrt_size[idx] = sqrt(s);
    recip_size[idx] = 1.0 / (s + 1.0);

    price_x_sqrt_size[idx] = p * sqrt(s);
    vwap_contrib[idx] = p * s;  // same as notional but semantically different
    spread_proxy[idx] = fabs(p - price[max(0, idx - 1)]);
    momentum[idx] = (idx >= 5) ? p - price[idx - 5] : 0.0;
}
""", "fused_pointwise_fp64")


FUSED_POINTWISE_FP32 = cp.RawKernel(r"""
extern "C" __global__
void fused_pointwise_fp32(
    const float* __restrict__ price,
    const float* __restrict__ size,
    const long long* __restrict__ timestamp,
    float* __restrict__ notional,
    float* __restrict__ ln_price,
    float* __restrict__ sqrt_price,
    float* __restrict__ recip_price,
    float* __restrict__ abs_return,
    float* __restrict__ sign_return,
    float* __restrict__ ln_size,
    float* __restrict__ sqrt_size,
    float* __restrict__ recip_size,
    float* __restrict__ price_x_sqrt_size,
    float* __restrict__ vwap_contrib,
    float* __restrict__ spread_proxy,
    float* __restrict__ momentum,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float p = price[idx];
    float s = size[idx];

    notional[idx] = p * s;
    ln_price[idx] = logf(p);
    sqrt_price[idx] = sqrtf(p);
    recip_price[idx] = 1.0f / p;

    float ret = (idx > 0) ? (p - price[idx - 1]) / price[idx - 1] : 0.0f;
    abs_return[idx] = fabsf(ret);
    sign_return[idx] = (ret > 0.0f) ? 1.0f : ((ret < 0.0f) ? -1.0f : 0.0f);

    ln_size[idx] = logf(s + 1.0f);
    sqrt_size[idx] = sqrtf(s);
    recip_size[idx] = 1.0f / (s + 1.0f);

    price_x_sqrt_size[idx] = p * sqrtf(s);
    vwap_contrib[idx] = p * s;
    spread_proxy[idx] = fabsf(p - price[max(0, idx - 1)]);
    momentum[idx] = (idx >= 5) ? p - price[idx - 5] : 0.0f;
}
""", "fused_pointwise_fp32")


# Kahan summation kernel
KAHAN_SUM_FP32 = cp.RawKernel(r"""
extern "C" __global__
void kahan_sum_fp32(const float* __restrict__ data, double* __restrict__ result, int n) {
    // Each block computes a partial Kahan sum, then atomicAdd to result
    __shared__ double block_sum[1];
    __shared__ double block_comp[1];

    if (threadIdx.x == 0) {
        block_sum[0] = 0.0;
        block_comp[0] = 0.0;
    }
    __syncthreads();

    // Thread-local Kahan sum
    double local_sum = 0.0;
    double local_comp = 0.0;

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < n; i += stride) {
        double y = (double)data[i] - local_comp;
        double t = local_sum + y;
        local_comp = (t - local_sum) - y;
        local_sum = t;
    }

    // Reduce within block using atomicAdd (simplified — warp shuffle would be better)
    atomicAdd(&block_sum[0], local_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum[0]);
    }
}
""", "kahan_sum_fp32")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def stats(times):
    s = sorted(times)
    n = len(s)
    m = sum(s) / n
    sd = (sum((x - m) ** 2 for x in s) / max(n - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "min": s[0], "max": s[-1],
            "p50": s[n // 2], "p95": s[int(n * 0.95)], "n": n}


def load_real_prices():
    """Load real AAPL prices from parquet."""
    tbl = pq.read_table("W:/fintek/data/fractal/K01/2025-09-02/AAPL/K01P01.TI00TO00.parquet")
    price = tbl.column("K01P01.DI01DO01").to_numpy().astype(np.float64)
    size = tbl.column("K01P01.DI01DO02").to_numpy().astype(np.float64)
    timestamp = tbl.column("K01P01.DI01DO03").to_numpy().astype(np.int64)
    return price, size, timestamp


def run_fused_fp64(price_gpu, size_gpu, ts_gpu, n):
    """Run FP64 fused pointwise kernel, return output arrays."""
    outputs = [cp.empty(n, dtype=cp.float64) for _ in range(13)]
    threads = 256
    blocks = (n + threads - 1) // threads
    FUSED_POINTWISE_FP64(
        (blocks,), (threads,),
        (price_gpu, size_gpu, ts_gpu,
         *outputs, np.int32(n))
    )
    return outputs


def run_fused_fp32(price_gpu, size_gpu, ts_gpu, n):
    """Run FP32 fused pointwise kernel, return output arrays."""
    outputs = [cp.empty(n, dtype=cp.float32) for _ in range(13)]
    threads = 256
    blocks = (n + threads - 1) // threads
    FUSED_POINTWISE_FP32(
        (blocks,), (threads,),
        (price_gpu, size_gpu, ts_gpu,
         *outputs, np.int32(n))
    )
    return outputs


OUTPUT_NAMES = [
    "notional", "ln_price", "sqrt_price", "recip_price",
    "abs_return", "sign_return", "ln_size", "sqrt_size", "recip_size",
    "price_x_sqrt_size", "vwap_contrib", "spread_proxy", "momentum",
]


# ══════════════════════════════════════════════════════════════════
# PHASE 1: FP64 vs FP32 kernel throughput
# ══════════════════════════════════════════════════════════════════

def phase1_kernel_throughput():
    print("=" * 80)
    print("PHASE 1: FP64 vs FP32 Fused Pointwise Kernel Throughput")
    print("=" * 80)
    print()

    price, size, timestamp = load_real_prices()
    n = len(price)
    print(f"  Data: {n:,} ticks, real AAPL prices")
    print()

    # FP64 path
    p64 = cp.asarray(price)
    s64 = cp.asarray(size)
    ts = cp.asarray(timestamp)

    # FP32 path
    p32 = cp.asarray(price.astype(np.float32))
    s32 = cp.asarray(size.astype(np.float32))

    # Warmup
    for _ in range(10):
        run_fused_fp64(p64, s64, ts, n)
        run_fused_fp32(p32, s32, ts, n)
    cp.cuda.Device(0).synchronize()

    n_runs = 200

    # FP64
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(n_runs):
        run_fused_fp64(p64, s64, ts, n)
    cp.cuda.Device(0).synchronize()
    t_fp64 = (time.perf_counter_ns() - t0) / n_runs / 1e3  # microseconds

    # FP32
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(n_runs):
        run_fused_fp32(p32, s32, ts, n)
    cp.cuda.Device(0).synchronize()
    t_fp32 = (time.perf_counter_ns() - t0) / n_runs / 1e3  # microseconds

    speedup = t_fp64 / t_fp32
    print(f"  FP64 fused pointwise (13 outputs): {t_fp64:8.1f} us")
    print(f"  FP32 fused pointwise (13 outputs): {t_fp32:8.1f} us")
    print(f"  Speedup: {speedup:.1f}x")
    print()

    # Bandwidth analysis
    input_bytes_64 = n * (8 + 8 + 8)  # 2x float64 + 1x int64
    output_bytes_64 = n * 8 * 13       # 13 float64 outputs
    total_bytes_64 = input_bytes_64 + output_bytes_64
    bw_64 = total_bytes_64 / (t_fp64 / 1e6) / 1e9

    input_bytes_32 = n * (4 + 4 + 8)  # 2x float32 + 1x int64
    output_bytes_32 = n * 4 * 13       # 13 float32 outputs
    total_bytes_32 = input_bytes_32 + output_bytes_32
    bw_32 = total_bytes_32 / (t_fp32 / 1e6) / 1e9

    print(f"  FP64 bandwidth: {bw_64:.0f} GB/s ({total_bytes_64/1e6:.1f} MB)")
    print(f"  FP32 bandwidth: {bw_32:.0f} GB/s ({total_bytes_32/1e6:.1f} MB)")
    print()

    # Individual op benchmarks
    print("  Individual operations (200 runs each):")
    print(f"  {'Op':<20s} {'FP64 (us)':>10s} {'FP32 (us)':>10s} {'Speedup':>8s}")
    print(f"  {'─'*50}")

    individual_ops = {
        "log": (lambda: cp.log(p64), lambda: cp.log(p32)),
        "sqrt": (lambda: cp.sqrt(p64), lambda: cp.sqrt(p32)),
        "reciprocal": (lambda: 1.0 / p64, lambda: 1.0 / p32),
        "multiply": (lambda: p64 * s64, lambda: p32 * s32),
        "abs": (lambda: cp.abs(p64), lambda: cp.abs(p32)),
        "diff": (lambda: cp.diff(p64), lambda: cp.diff(p32)),
        "sum": (lambda: cp.sum(p64), lambda: cp.sum(p32)),
        "mean": (lambda: cp.mean(p64), lambda: cp.mean(p32)),
        "std": (lambda: cp.std(p64), lambda: cp.std(p32)),
        "sort": (lambda: cp.sort(p64.copy()), lambda: cp.sort(p32.copy())),
        "cumsum": (lambda: cp.cumsum(p64), lambda: cp.cumsum(p32)),
        "where": (lambda: cp.where(p64 > 230, p64, s64),
                  lambda: cp.where(p32 > 230, p32, s32)),
    }

    for name, (fn64, fn32) in individual_ops.items():
        # Warmup
        for _ in range(10):
            fn64()
            fn32()
        cp.cuda.Device(0).synchronize()

        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(n_runs):
            fn64()
        cp.cuda.Device(0).synchronize()
        us64 = (time.perf_counter_ns() - t0) / n_runs / 1e3

        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(n_runs):
            fn32()
        cp.cuda.Device(0).synchronize()
        us32 = (time.perf_counter_ns() - t0) / n_runs / 1e3

        sp = us64 / us32 if us32 > 0 else 0
        print(f"  {name:<20s} {us64:>8.1f}us {us32:>8.1f}us {sp:>6.1f}x")

    return p64, s64, ts, p32, s32, n


# ══════════════════════════════════════════════════════════════════
# PHASE 2: Precision validation
# ══════════════════════════════════════════════════════════════════

def phase2_precision(p64, s64, ts, p32, s32, n):
    print()
    print("=" * 80)
    print("PHASE 2: Precision Validation — FP32 vs FP64 on Real AAPL Data")
    print("=" * 80)
    print()

    out64 = run_fused_fp64(p64, s64, ts, n)
    out32 = run_fused_fp32(p32, s32, ts, n)
    cp.cuda.Device(0).synchronize()

    print(f"  {'Column':<22s} {'Max AbsErr':>12s} {'Max RelErr':>12s} {'Mean RelErr':>12s} {'Range':>20s}")
    print(f"  {'─'*80}")

    for i, name in enumerate(OUTPUT_NAMES):
        ref = out64[i].get().astype(np.float64)
        test = out32[i].get().astype(np.float64)

        abs_err = np.abs(ref - test)
        # Relative error (avoid div by zero)
        mask = np.abs(ref) > 1e-15
        rel_err = np.zeros_like(abs_err)
        rel_err[mask] = abs_err[mask] / np.abs(ref[mask])

        max_abs = np.max(abs_err)
        max_rel = np.max(rel_err[mask]) if np.any(mask) else 0.0
        mean_rel = np.mean(rel_err[mask]) if np.any(mask) else 0.0
        val_range = f"[{np.min(ref):.4g}, {np.max(ref):.4g}]"

        print(f"  {name:<22s} {max_abs:>12.6g} {max_rel:>12.6g} {mean_rel:>12.6g} {val_range:>20s}")

    # Summary statistics
    print()
    print("  Key observations:")

    # Check if any relative error exceeds thresholds
    all_within_1e4 = True
    all_within_1e6 = True
    for i, name in enumerate(OUTPUT_NAMES):
        ref = out64[i].get().astype(np.float64)
        test = out32[i].get().astype(np.float64)
        mask = np.abs(ref) > 1e-15
        if np.any(mask):
            max_rel = np.max(np.abs(ref[mask] - test[mask]) / np.abs(ref[mask]))
            if max_rel > 1e-4:
                all_within_1e4 = False
            if max_rel > 1e-6:
                all_within_1e6 = False

    print(f"  All columns within 1e-6 relative error: {'YES' if all_within_1e6 else 'NO'}")
    print(f"  All columns within 1e-4 relative error: {'YES' if all_within_1e4 else 'NO'}")

    # Price-specific: how many decimal places are preserved?
    p64_host = p64.get()
    p32_host = p32.get().astype(np.float64)
    price_err = np.max(np.abs(p64_host - p32_host))
    print(f"  Price roundtrip error (fp64->fp32->fp64): max ${price_err:.10f}")
    print(f"  Price decimal places preserved: {-np.log10(price_err / np.mean(p64_host)):.1f}")


# ══════════════════════════════════════════════════════════════════
# PHASE 3: "Don't store derivables" pipeline
# ══════════════════════════════════════════════════════════════════

def phase3_dont_store_derivables():
    print()
    print("=" * 80)
    print("PHASE 3: Store-All vs Recompute-on-GPU Pipeline Comparison")
    print("=" * 80)
    print()

    price_np, size_np, ts_np = load_real_prices()
    n = len(price_np)

    # Path A: Store all 16 columns (3 base + 13 derived) as float32
    print("  Path A: Store all 16 columns in MKTF (float32)")
    p32_np = price_np.astype(np.float32)
    s32_np = size_np.astype(np.float32)

    # Compute derived on CPU for storage
    all_cols = {
        "price": p32_np,
        "size": s32_np,
        "timestamp": ts_np,
        "notional": (p32_np * s32_np).astype(np.float32),
        "ln_price": np.log(p32_np).astype(np.float32),
        "sqrt_price": np.sqrt(p32_np).astype(np.float32),
        "recip_price": (1.0 / p32_np).astype(np.float32),
        "ln_size": np.log(s32_np + 1).astype(np.float32),
        "sqrt_size": np.sqrt(s32_np).astype(np.float32),
        "recip_size": (1.0 / (s32_np + 1)).astype(np.float32),
        "price_x_sqrt_size": (p32_np * np.sqrt(s32_np)).astype(np.float32),
        "vwap_contrib": (p32_np * s32_np).astype(np.float32),
        "spread_proxy": np.abs(np.diff(p32_np, prepend=p32_np[0])).astype(np.float32),
        "momentum": np.zeros(n, dtype=np.float32),
    }
    meta_a = {"encoding": "float32", "pipeline": "store_all"}
    path_a = tempfile.mktemp(suffix=".mktf")
    write_mktf(path_a, all_cols, meta_a)
    size_a = os.path.getsize(path_a) / 1e6
    print(f"    File size: {size_a:.2f} MB ({len(all_cols)} columns)")

    # Path B: Store only 3 base columns (price fp32, size fp32, timestamp int64)
    base_cols = {
        "price": p32_np,
        "size": s32_np,
        "timestamp": ts_np,
    }
    meta_b = {"encoding": "float32", "pipeline": "recompute"}
    path_b = tempfile.mktemp(suffix=".mktf")
    write_mktf(path_b, base_cols, meta_b)
    size_b = os.path.getsize(path_b) / 1e6
    print(f"    Path B file size: {size_b:.2f} MB ({len(base_cols)} columns)")
    print(f"    Size reduction: {(1 - size_b/size_a)*100:.0f}%")
    print()

    n_runs = 30

    # ── Path A: Read all → H2D ──
    def pipeline_a():
        d = read_mktf(path_a)
        gpu = {k: cp.asarray(v) for k, v in d.items()}
        cp.cuda.Device(0).synchronize()
        return gpu

    # Warmup
    pipeline_a()
    pipeline_a()

    times_a = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        pipeline_a()
        t1 = time.perf_counter_ns()
        gc.enable()
        times_a.append((t1 - t0) / 1e6)

    sa = stats(times_a)
    print(f"  Path A (read 16 cols + H2D):      {sa['mean']:7.2f}ms +/- {sa['std']:.2f}")

    # ── Path B: Read base → H2D → GPU recompute ──
    def pipeline_b():
        d = read_mktf(path_b)
        p_gpu = cp.asarray(d["price"])
        s_gpu = cp.asarray(d["size"])
        ts_gpu = cp.asarray(d["timestamp"])
        outputs = run_fused_fp32(p_gpu, s_gpu, ts_gpu, n)
        cp.cuda.Device(0).synchronize()
        return p_gpu, s_gpu, ts_gpu, outputs

    # Warmup
    pipeline_b()
    pipeline_b()

    times_b = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        pipeline_b()
        t1 = time.perf_counter_ns()
        gc.enable()
        times_b.append((t1 - t0) / 1e6)

    sb = stats(times_b)
    print(f"  Path B (read 3 cols + H2D + GPU): {sb['mean']:7.2f}ms +/- {sb['std']:.2f}")

    # Breakdown for Path B
    # Read only
    times_read = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        d = read_mktf(path_b)
        t1 = time.perf_counter_ns()
        gc.enable()
        times_read.append((t1 - t0) / 1e6)

    # H2D only
    d = read_mktf(path_b)
    arrays = [np.ascontiguousarray(v) for v in d.values()]
    times_h2d = []
    for _ in range(n_runs):
        gc.disable()
        t0 = time.perf_counter_ns()
        gpu_arrs = [cp.asarray(a) for a in arrays]
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter_ns()
        gc.enable()
        times_h2d.append((t1 - t0) / 1e6)

    # Compute only (already on GPU)
    p_gpu = cp.asarray(d["price"])
    s_gpu = cp.asarray(d["size"])
    ts_gpu = cp.asarray(d["timestamp"])
    cp.cuda.Device(0).synchronize()

    times_compute = []
    for _ in range(200):
        gc.disable()
        t0 = time.perf_counter_ns()
        run_fused_fp32(p_gpu, s_gpu, ts_gpu, n)
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter_ns()
        gc.enable()
        times_compute.append((t1 - t0) / 1e6)

    sr = stats(times_read)
    sh = stats(times_h2d)
    sc = stats(times_compute)
    print()
    print(f"  Path B breakdown:")
    print(f"    Read 3 cols:    {sr['mean']:7.2f}ms")
    print(f"    H2D transfer:   {sh['mean']:7.2f}ms")
    print(f"    GPU compute:    {sc['mean']:7.4f}ms  ({sc['mean']*1000:.1f}us)")
    print(f"    Sum:            {sr['mean'] + sh['mean'] + sc['mean']:7.2f}ms")
    print()

    winner = "B (recompute)" if sb['mean'] < sa['mean'] else "A (store all)"
    diff = abs(sa['mean'] - sb['mean'])
    print(f"  WINNER: Path {winner} by {diff:.2f}ms")
    print(f"  Path B saves {(1 - size_b/size_a)*100:.0f}% disk space")
    per_ticker_save = size_a - size_b
    print(f"  Universe savings: {per_ticker_save * 4604:.0f} MB = {per_ticker_save * 4604 / 1000:.1f} GB per day")
    print()

    # Universe projections
    print("  Universe projection (4604 tickers):")
    print(f"    Path A: {sa['mean'] * 4604 / 1000:.1f}s")
    print(f"    Path B: {sb['mean'] * 4604 / 1000:.1f}s")

    os.unlink(path_a)
    os.unlink(path_b)


# ══════════════════════════════════════════════════════════════════
# PHASE 4: Kahan summation
# ══════════════════════════════════════════════════════════════════

def phase4_kahan():
    print()
    print("=" * 80)
    print("PHASE 4: Kahan Compensated Summation — FP32 Naive vs Kahan vs FP64")
    print("=" * 80)
    print()

    price, _, _ = load_real_prices()
    n = len(price)

    p64 = cp.asarray(price)
    p32 = cp.asarray(price.astype(np.float32))

    # Ground truth: FP64 sum (numpy for max precision)
    truth = np.sum(price)  # float64 on CPU
    print(f"  Ground truth (numpy FP64): {truth:.10f}")
    print()

    # FP64 GPU sum
    gpu_64_sum = float(cp.sum(p64))
    err_64 = abs(gpu_64_sum - truth) / abs(truth)
    print(f"  CuPy FP64 sum:            {gpu_64_sum:.10f}  (rel err: {err_64:.2e})")

    # FP32 naive GPU sum
    gpu_32_sum = float(cp.sum(p32))
    err_32 = abs(gpu_32_sum - truth) / abs(truth)
    print(f"  CuPy FP32 naive sum:      {gpu_32_sum:.10f}  (rel err: {err_32:.2e})")

    # FP32 Kahan GPU sum
    result = cp.zeros(1, dtype=cp.float64)
    threads = 256
    blocks = min(256, (n + threads - 1) // threads)
    KAHAN_SUM_FP32((blocks,), (threads,), (p32, result, np.int32(n)))
    cp.cuda.Device(0).synchronize()
    kahan_sum = float(result[0])
    err_kahan = abs(kahan_sum - truth) / abs(truth)
    print(f"  FP32 Kahan sum:           {kahan_sum:.10f}  (rel err: {err_kahan:.2e})")

    # FP32 double-precision accumulator (upcast in reduction)
    # CuPy doesn't do this natively, so let's test manual approach
    p32_as_64 = p32.astype(cp.float64)
    upcast_sum = float(cp.sum(p32_as_64))
    err_upcast = abs(upcast_sum - truth) / abs(truth)
    print(f"  FP32->FP64 upcast sum:    {upcast_sum:.10f}  (rel err: {err_upcast:.2e})")

    print()

    # Mean comparison
    truth_mean = np.mean(price)
    gpu_64_mean = float(cp.mean(p64))
    gpu_32_mean = float(cp.mean(p32))
    kahan_mean = kahan_sum / n

    print(f"  Mean comparison:")
    print(f"    Truth (numpy FP64):     {truth_mean:.10f}")
    print(f"    CuPy FP64 mean:        {gpu_64_mean:.10f}  (err: {abs(gpu_64_mean - truth_mean):.2e})")
    print(f"    CuPy FP32 mean:        {gpu_32_mean:.10f}  (err: {abs(gpu_32_mean - truth_mean):.2e})")
    print(f"    Kahan FP32 mean:       {kahan_mean:.10f}  (err: {abs(kahan_mean - truth_mean):.2e})")
    print()

    # Std comparison
    truth_std = np.std(price)
    gpu_64_std = float(cp.std(p64))
    gpu_32_std = float(cp.std(p32))

    print(f"  Std comparison:")
    print(f"    Truth (numpy FP64):     {truth_std:.10f}")
    print(f"    CuPy FP64 std:         {gpu_64_std:.10f}  (err: {abs(gpu_64_std - truth_std):.2e})")
    print(f"    CuPy FP32 std:         {gpu_32_std:.10f}  (err: {abs(gpu_32_std - truth_std):.2e})")
    print()

    # Practical significance test: does the error matter for signal detection?
    # Typical signal: price mean over different windows
    windows = [100, 1000, 10000, 100000]
    print(f"  Rolling mean accuracy (FP32 vs FP64, practical significance):")
    for w in windows:
        # Compare rolling means
        roll_64 = np.convolve(price, np.ones(w)/w, mode='valid')
        roll_32 = np.convolve(price.astype(np.float32), np.ones(w, dtype=np.float32)/w, mode='valid')
        max_err = np.max(np.abs(roll_64 - roll_32.astype(np.float64)))
        max_rel = max_err / np.mean(np.abs(roll_64))
        print(f"    Window {w:>6d}: max err ${max_err:.8f}, rel err {max_rel:.2e}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print()
    print("OBSERVER EXPERIMENT 5: GPU Precision & Throughput")
    print("Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q")
    dev = cp.cuda.Device(0)
    print(f"GPU Memory: {dev.mem_info[1] / 1e9:.1f} GB")
    print()

    p64, s64, ts, p32, s32, n = phase1_kernel_throughput()
    phase2_precision(p64, s64, ts, p32, s32, n)
    phase3_dont_store_derivables()
    phase4_kahan()

    print()
    print("=" * 80)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
