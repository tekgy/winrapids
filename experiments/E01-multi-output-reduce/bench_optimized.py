"""
E01 Phase 2 — Optimized fused kernel + fair comparison

Phase 1 revealed:
  - CuPy's std() is 250x slower than sum/min/max/mean (83ms vs 0.3ms at 100M)
  - The naive fused kernel (17ms) beats "separate" only because std() is bad
  - Individual CuPy ops achieve 1200+ GB/s; fused kernel only 23 GB/s

This phase:
  1. Write an optimized fused kernel (grid-stride, no per-element division)
  2. Compare fairly: fused vs (4 CuPy ops + custom std)
  3. Measure the ACTUAL read count via bandwidth analysis
"""

import time
import numpy as np
import cupy as cp


# ============================================================
# Optimized fused kernel: grid-stride, simple accumulators
# ============================================================

OPTIMIZED_FUSED = cp.RawKernel(r"""
extern "C" __global__
void opt_fused_stats(const float* __restrict__ data,
                     double* out_sum, double* out_min, double* out_max,
                     double* out_sq_sum,
                     int n) {
    // Grid-stride loop: each thread processes many elements
    // Accumulate: sum, min, max, sum_of_squares
    // Mean = sum/n, Var = sq_sum/n - mean^2, Std = sqrt(Var)

    extern __shared__ char smem[];
    int num_warps = blockDim.x >> 5;
    double* s_sum = (double*)smem;
    double* s_min = s_sum + num_warps;
    double* s_max = s_min + num_warps;
    double* s_sq  = s_max + num_warps;

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int stride = gridDim.x * blockDim.x;

    double t_sum = 0.0;
    double t_min = 1e308;
    double t_max = -1e308;
    double t_sq = 0.0;

    // Grid-stride loop
    for (unsigned int i = gid; i < n; i += stride) {
        double v = (double)data[i];
        t_sum += v;
        t_sq += v * v;
        if (v < t_min) t_min = v;
        if (v > t_max) t_max = v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        t_sum += __shfl_down_sync(0xFFFFFFFF, t_sum, offset);
        t_sq += __shfl_down_sync(0xFFFFFFFF, t_sq, offset);
        double o_min = __shfl_down_sync(0xFFFFFFFF, t_min, offset);
        double o_max = __shfl_down_sync(0xFFFFFFFF, t_max, offset);
        if (o_min < t_min) t_min = o_min;
        if (o_max > t_max) t_max = o_max;
    }

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        s_sum[warp_id] = t_sum;
        s_min[warp_id] = t_min;
        s_max[warp_id] = t_max;
        s_sq[warp_id] = t_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        t_sum = (lane < num_warps) ? s_sum[lane] : 0.0;
        t_min = (lane < num_warps) ? s_min[lane] : 1e308;
        t_max = (lane < num_warps) ? s_max[lane] : -1e308;
        t_sq = (lane < num_warps) ? s_sq[lane] : 0.0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            t_sum += __shfl_down_sync(0xFFFFFFFF, t_sum, offset);
            t_sq += __shfl_down_sync(0xFFFFFFFF, t_sq, offset);
            double o_min = __shfl_down_sync(0xFFFFFFFF, t_min, offset);
            double o_max = __shfl_down_sync(0xFFFFFFFF, t_max, offset);
            if (o_min < t_min) t_min = o_min;
            if (o_max > t_max) t_max = o_max;
        }
        if (lane == 0) {
            out_sum[blockIdx.x] = t_sum;
            out_min[blockIdx.x] = t_min;
            out_max[blockIdx.x] = t_max;
            out_sq_sum[blockIdx.x] = t_sq;
        }
    }
}
""", "opt_fused_stats")


# ============================================================
# Custom single-pass std kernel (for fair comparison)
# ============================================================

CUSTOM_STD = cp.RawKernel(r"""
extern "C" __global__
void custom_std(const float* __restrict__ data,
                double* out_sum, double* out_sq_sum,
                int n) {
    extern __shared__ char smem[];
    int num_warps = blockDim.x >> 5;
    double* s_sum = (double*)smem;
    double* s_sq = s_sum + num_warps;

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int stride = gridDim.x * blockDim.x;

    double t_sum = 0.0;
    double t_sq = 0.0;

    for (unsigned int i = gid; i < n; i += stride) {
        double v = (double)data[i];
        t_sum += v;
        t_sq += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        t_sum += __shfl_down_sync(0xFFFFFFFF, t_sum, offset);
        t_sq += __shfl_down_sync(0xFFFFFFFF, t_sq, offset);
    }

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        s_sum[warp_id] = t_sum;
        s_sq[warp_id] = t_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        t_sum = (lane < num_warps) ? s_sum[lane] : 0.0;
        t_sq = (lane < num_warps) ? s_sq[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            t_sum += __shfl_down_sync(0xFFFFFFFF, t_sum, offset);
            t_sq += __shfl_down_sync(0xFFFFFFFF, t_sq, offset);
        }
        if (lane == 0) {
            out_sum[blockIdx.x] = t_sum;
            out_sq_sum[blockIdx.x] = t_sq;
        }
    }
}
""", "custom_std")


def opt_fused_reduce(data: cp.ndarray):
    """Optimized: grid-stride, 128 blocks, sum+min+max+sq_sum in one pass."""
    n = len(data)
    threads = 256
    blocks = 128  # Fixed block count, grid-stride handles the rest
    num_warps = threads // 32
    smem = num_warps * 4 * 8  # 4 arrays of doubles

    p_sum = cp.empty(blocks, dtype=cp.float64)
    p_min = cp.empty(blocks, dtype=cp.float64)
    p_max = cp.empty(blocks, dtype=cp.float64)
    p_sq = cp.empty(blocks, dtype=cp.float64)

    OPTIMIZED_FUSED((blocks,), (threads,),
                    (data, p_sum, p_min, p_max, p_sq, n),
                    shared_mem=smem)

    total_sum = float(cp.sum(p_sum))
    total_min = float(cp.min(p_min))
    total_max = float(cp.max(p_max))
    total_sq = float(cp.sum(p_sq))

    mean = total_sum / n
    variance = total_sq / n - mean * mean
    std = max(0.0, variance) ** 0.5

    return total_sum, total_min, total_max, mean, std


def custom_std_reduce(data: cp.ndarray):
    """Custom std kernel (single pass, sum + sum_of_squares)."""
    n = len(data)
    threads = 256
    blocks = 128
    num_warps = threads // 32
    smem = num_warps * 2 * 8

    p_sum = cp.empty(blocks, dtype=cp.float64)
    p_sq = cp.empty(blocks, dtype=cp.float64)

    CUSTOM_STD((blocks,), (threads,),
               (data, p_sum, p_sq, n),
               shared_mem=smem)

    total_sum = float(cp.sum(p_sum))
    total_sq = float(cp.sum(p_sq))
    mean = total_sum / n
    variance = total_sq / n - mean * mean
    return max(0.0, variance) ** 0.5


def cupy_4ops_custom_std(data: cp.ndarray):
    """4 CuPy ops (sum/min/max/mean) + custom std. Reads data ~5 times but
    std is fast."""
    s = float(cp.sum(data))
    mn = float(cp.min(data))
    mx = float(cp.max(data))
    mean = float(cp.mean(data))
    std = custom_std_reduce(data)
    return s, mn, mx, mean, std


def separate_cupy(data: cp.ndarray):
    """5 separate CuPy reductions (baseline with CuPy std)."""
    s = float(cp.sum(data))
    mn = float(cp.min(data))
    mx = float(cp.max(data))
    mean = float(cp.mean(data))
    std = float(cp.std(data))
    return s, mn, mx, mean, std


def bench(fn, data, warmup=3, runs=20, label=""):
    for _ in range(warmup):
        fn(data)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn(data)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    avg = sum(times) / len(times)
    return {"p50": p50, "p99": p99, "mean": avg, "label": label}


def main():
    print("=" * 70)
    print("E01 Phase 2 -- Optimized Fused + Fair Comparison")
    print("=" * 70)

    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    size_labels = ["1M (4MB)", "10M (40MB)", "50M (200MB)", "100M (400MB)"]

    for size, label in zip(sizes, size_labels):
        print(f"\n--- {label} ({size:,} float32) ---")

        data = cp.random.uniform(-100.0, 100.0, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()

        # Correctness
        ref = separate_cupy(data)
        opt = opt_fused_reduce(data)
        print(f"  Correctness vs CuPy:")
        for name, rv, ov in zip(["sum","min","max","mean","std"], ref, opt):
            err = abs(rv - ov) / max(abs(rv), 1e-10)
            print(f"    {name:>5}: ref={rv:12.4f}  opt={ov:12.4f}  err={err:.2e}")

        # Three approaches:
        # A) 5 separate CuPy (includes bad std)
        # B) 4 CuPy + custom std (fair multi-read comparison)
        # C) Optimized fused (single read)
        a = bench(separate_cupy, data, label="A: 5x CuPy (incl std)")
        b = bench(cupy_4ops_custom_std, data, label="B: 4x CuPy + custom std")
        c = bench(opt_fused_reduce, data, label="C: optimized fused")

        # Also time just the custom std alone
        std_times = []
        for _ in range(3):
            custom_std_reduce(data)
        cp.cuda.Stream.null.synchronize()
        for _ in range(20):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            custom_std_reduce(data)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            std_times.append((t1 - t0) * 1000)
        std_times.sort()
        std_p50 = std_times[len(std_times) // 2]

        # Time CuPy's std alone
        cupy_std_times = []
        for _ in range(3):
            float(cp.std(data))
        cp.cuda.Stream.null.synchronize()
        for _ in range(20):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            float(cp.std(data))
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            cupy_std_times.append((t1 - t0) * 1000)
        cupy_std_times.sort()
        cupy_std_p50 = cupy_std_times[len(cupy_std_times) // 2]

        data_mb = size * 4 / 1e6
        print(f"\n  Results (ms):")
        print(f"    {'':>35}  {'p50':>8}  {'p99':>8}  {'mean':>8}")
        print(f"    {'A: 5x CuPy (incl bad std)':>35}  {a['p50']:8.3f}  {a['p99']:8.3f}  {a['mean']:8.3f}")
        print(f"    {'B: 4x CuPy + custom std':>35}  {b['p50']:8.3f}  {b['p99']:8.3f}  {b['mean']:8.3f}")
        print(f"    {'C: optimized fused (1 read)':>35}  {c['p50']:8.3f}  {c['p99']:8.3f}  {c['mean']:8.3f}")
        print(f"    {'CuPy std alone':>35}  {cupy_std_p50:8.3f}")
        print(f"    {'Custom std alone':>35}  {std_p50:8.3f}")
        print(f"\n  Speedups (p50):")
        print(f"    C vs A (fused vs all-CuPy): {a['p50']/c['p50']:.2f}x")
        print(f"    C vs B (fused vs fair):     {b['p50']/c['p50']:.2f}x")
        print(f"    Custom std vs CuPy std:     {cupy_std_p50/std_p50:.1f}x")

        # Bandwidth
        c_bw = (data_mb / 1e3) / (c['p50'] / 1000)
        b_bw = (data_mb * 5 / 1e3) / (b['p50'] / 1000)
        print(f"\n  Bandwidth:")
        print(f"    C (1 read): {c_bw:.0f} GB/s  ({data_mb:.0f} MB)")
        print(f"    B (5 reads): {b_bw:.0f} GB/s  ({data_mb*5:.0f} MB total)")

        del data
        cp.get_default_memory_pool().free_all_blocks()

    print(f"\n{'=' * 70}")
    print("E01 Phase 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
