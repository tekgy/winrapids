"""
E05 -- JIT Compilation Overhead: Pre-built vs JIT vs Composed

Key question (from team lead):
  What's the JIT compilation overhead -- is 40ms realistic or optimistic?

The compiler vision proposes three tiers:
  1. Pre-built: compiled at install time, zero JIT overhead
  2. JIT: compiled at first use, cached for reuse
  3. Composed: assembled from pre-built blocks at runtime

This experiment measures:
  - NVRTC compilation time for various kernel complexities
  - CuPy's RawKernel JIT overhead (first call vs cached)
  - Cost of composing a pipeline from pre-built blocks vs monolithic JIT
  - The cache hit path: how fast is kernel reuse?

Kernel complexities tested:
  - Trivial: single arithmetic op
  - Simple: 5-op fused expression
  - Medium: 20-op expression with branches
  - Complex: full pipeline kernel (rolling stats + z-score + output)
"""

import time
import hashlib
import cupy as cp


def measure_compilation(name, kernel_src, kernel_name, warmup=False):
    """Measure NVRTC compilation time for a kernel.

    Only measures compilation (RawKernel creation + forced compile),
    not execution (since kernels have different signatures).
    """
    unique_tag = hashlib.md5(f"{name}{time.perf_counter()}".encode()).hexdigest()[:8]
    modified_src = kernel_src.replace(kernel_name, f"{kernel_name}_{unique_tag}")
    modified_name = f"{kernel_name}_{unique_tag}"

    # CuPy lazy-compiles: RawKernel() just stores source.
    # Compilation happens on first .kernel access or __call__.
    # We use .attributes to force compilation without needing correct args.
    t0 = time.perf_counter()
    kernel = cp.RawKernel(modified_src, modified_name)
    t_create = time.perf_counter() - t0

    # Force compilation via accessing .kernel (triggers NVRTC)
    t0 = time.perf_counter()
    _ = kernel.kernel  # This triggers the actual NVRTC compile
    t_compile = time.perf_counter() - t0

    # Second access (cached PTX, no recompilation)
    t0 = time.perf_counter()
    _ = kernel.kernel
    t_cached = time.perf_counter() - t0

    return {
        "create_ms": t_create * 1000,
        "first_call_ms": t_compile * 1000,
        "cached_ms": t_cached * 1000,
        "total_jit_ms": (t_create + t_compile) * 1000
    }


# ============================================================
# Kernel templates at varying complexity
# ============================================================

TRIVIAL_KERNEL = r"""
extern "C" __global__
void trivial(const float* a, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (double)a[idx] + 1.0;
}
"""

SIMPLE_KERNEL = r"""
extern "C" __global__
void simple(const float* a, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double v = (double)a[idx];
        out[idx] = (v * v + v * 2.0 - 1.0) / (v + 1.0001) + sqrt(fabs(v));
    }
}
"""

MEDIUM_KERNEL = r"""
extern "C" __global__
void medium(const float* a, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double v = (double)a[idx];
        double r = v * v;
        r = r + v * 3.0;
        r = r - sqrt(fabs(v) + 0.001);
        r = r * (v > 0.0 ? 1.0 : -1.0);
        double s = log(fabs(v) + 1.0);
        s = s * exp(-v * v * 0.01);
        s = s + pow(fabs(v), 0.3333);
        r = r + s;
        r = r / (1.0 + fabs(r) * 0.001);
        double t = sin(v * 3.14159) * cos(v * 1.57);
        t = t + tanh(v * 0.5);
        r = r + t * 0.1;
        r = r * (1.0 + 0.01 * v);
        r = fmin(fmax(r, -1e6), 1e6);
        out[idx] = r;
    }
}
"""

COMPLEX_KERNEL = r"""
extern "C" __global__
void complex_pipeline(const float* prices, const float* volumes,
                      const double* price_cumsum, const double* price_sq_cumsum,
                      const double* vol_cumsum,
                      double* vwap_z, double* vol_ratio,
                      int n, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data_idx = idx + window - 1;

    if (data_idx < n) {
        // Rolling VWAP
        double sum_pv = 0.0;
        double sum_v = vol_cumsum[data_idx + 1] - vol_cumsum[data_idx + 1 - window];
        // Approximate VWAP as price_mean * vol_ratio
        double price_sum = price_cumsum[data_idx + 1] - price_cumsum[data_idx + 1 - window];
        double price_mean = price_sum / (double)window;

        // Rolling price std
        double price_sq = price_sq_cumsum[data_idx + 1] - price_sq_cumsum[data_idx + 1 - window];
        double price_var = price_sq / (double)window - price_mean * price_mean;
        if (price_var < 0.0) price_var = 0.0;
        double price_std = sqrt(price_var);

        // Z-score of current price vs rolling mean
        double z = (price_std > 1e-10) ?
            ((double)prices[data_idx] - price_mean) / price_std : 0.0;

        // Volume ratio: current volume / rolling average volume
        double vol_avg = sum_v / (double)window;
        double vr = (vol_avg > 1e-10) ?
            (double)volumes[data_idx] / vol_avg : 1.0;

        vwap_z[idx] = z;
        vol_ratio[idx] = vr;
    }
}
"""

# A very large kernel (simulating a production leaf with many computations)
LARGE_KERNEL = r"""
extern "C" __global__
void large_production(const float* a, const float* b, const float* c,
                      const float* d, const float* e,
                      double* out1, double* out2, double* out3,
                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double va = (double)a[idx];
        double vb = (double)b[idx];
        double vc = (double)c[idx];
        double vd = (double)d[idx];
        double ve = (double)e[idx];

        // 30+ operations simulating a real leaf computation
        double r1 = va * vb + vc;
        double r2 = vd * ve - va;
        double r3 = sqrt(fabs(r1) + 0.001);
        double r4 = log(fabs(r2) + 1.0);
        double r5 = exp(-va * va * 0.001);
        double r6 = r3 * r4 * r5;
        double r7 = (r1 > 0.0 ? r1 : -r1);
        double r8 = fmin(r7, 1e6);
        double r9 = pow(fabs(vb), 0.5);
        double r10 = tanh(vc * 0.1);
        double r11 = r6 + r9 * r10;
        double r12 = r11 / (1.0 + fabs(r11) * 0.0001);
        double r13 = va * vd + vb * ve;
        double r14 = sin(r13 * 0.01) + cos(r13 * 0.02);
        double r15 = r12 + r14 * 0.01;

        out1[idx] = r15;
        out2[idx] = r6;
        out3[idx] = r8 + r10;
    }
}
"""


def main():
    print("=" * 70)
    print("E05 -- JIT Compilation Overhead: Three Tiers Benchmarked")
    print("=" * 70)

    # ── Compilation time by kernel complexity ────────────────────
    print("\n--- Compilation time by kernel complexity ---\n")

    kernels = [
        ("trivial (1 op)", TRIVIAL_KERNEL, "trivial"),
        ("simple (5 ops)", SIMPLE_KERNEL, "simple"),
        ("medium (20 ops, branches, transcendentals)", MEDIUM_KERNEL, "medium"),
        ("complex (pipeline: VWAP + z-score + vol)", COMPLEX_KERNEL, "complex_pipeline"),
        ("large (30+ ops, 5 inputs, 3 outputs)", LARGE_KERNEL, "large_production"),
    ]

    print(f"  {'Kernel':>45}  {'Create':>8}  {'1st Call':>8}  {'Cached':>8}  {'Total JIT':>10}")
    print(f"  {'':>45}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>10}")

    for name, src, kname in kernels:
        # Run 5 times, take median
        results = []
        for _ in range(5):
            r = measure_compilation(name, src, kname)
            results.append(r)

        # Sort by total JIT time, take median
        results.sort(key=lambda x: x["total_jit_ms"])
        median = results[len(results) // 2]

        print(f"  {name:>45}  {median['create_ms']:8.1f}  "
              f"{median['first_call_ms']:8.1f}  {median['cached_ms']:8.3f}  "
              f"{median['total_jit_ms']:10.1f}")

    # ── Composed pipeline: multiple pre-built kernels vs monolithic ──
    print("\n--- Composed vs Monolithic Pipeline ---\n")

    n = 10_000_000
    prices = cp.random.uniform(50, 150, n).astype(cp.float32)
    volumes = cp.random.uniform(100, 10000, n).astype(cp.float32)
    window = 60

    # Pre-compute cumsums (shared infrastructure)
    price_cs = cp.cumsum(prices.astype(cp.float64))
    price_cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), price_cs])
    price_sq_cs = cp.cumsum((prices.astype(cp.float64)) ** 2)
    price_sq_cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), price_sq_cs])
    vol_cs = cp.cumsum(volumes.astype(cp.float64))
    vol_cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), vol_cs])
    cp.cuda.Stream.null.synchronize()

    n_out = n - window + 1

    # Path A: Composed from CuPy primitives (pre-built blocks)
    def composed_pipeline():
        # Rolling mean/std
        price_mean = (price_cs[window:] - price_cs[:-window]) / window
        price_sq_mean = (price_sq_cs[window:] - price_sq_cs[:-window]) / window
        price_var = cp.maximum(price_sq_mean - price_mean * price_mean, 0)
        price_std = cp.sqrt(price_var)

        # Z-score
        current_prices = prices[window - 1:].astype(cp.float64)
        z = (current_prices - price_mean) / cp.maximum(price_std, 1e-10)

        # Volume ratio
        vol_mean = (vol_cs[window:] - vol_cs[:-window]) / window
        current_vols = volumes[window - 1:].astype(cp.float64)
        vr = current_vols / cp.maximum(vol_mean, 1e-10)

        return z, vr

    # Path B: Monolithic JIT kernel
    # First, compile the complex kernel
    complex_kernel = cp.RawKernel(COMPLEX_KERNEL, "complex_pipeline")

    def monolithic_pipeline():
        vwap_z = cp.empty(n_out, dtype=cp.float64)
        vol_ratio = cp.empty(n_out, dtype=cp.float64)
        threads = 256
        blocks = (n_out + threads - 1) // threads
        complex_kernel((blocks,), (threads,),
                       (prices, volumes, price_cs, price_sq_cs, vol_cs,
                        vwap_z, vol_ratio, n, window))
        return vwap_z, vol_ratio

    # Correctness check
    z_comp, vr_comp = composed_pipeline()
    z_mono, vr_mono = monolithic_pipeline()
    cp.cuda.Stream.null.synchronize()
    z_err = float(cp.max(cp.abs(z_comp - z_mono)))
    vr_err = float(cp.max(cp.abs(vr_comp - vr_mono)))
    print(f"  Correctness: z_err={z_err:.2e}, vr_err={vr_err:.2e}")

    # Benchmark
    def bench(fn, warmup=5, runs=30):
        for _ in range(warmup):
            fn()
        cp.cuda.Stream.null.synchronize()
        times = []
        for _ in range(runs):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            fn()
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        times.sort()
        return times[len(times) // 2], times[int(len(times) * 0.99)], sum(times) / len(times)

    comp_p50, comp_p99, comp_mean = bench(composed_pipeline)
    mono_p50, mono_p99, mono_mean = bench(monolithic_pipeline)

    print(f"\n  Pipeline execution (10M rows, window=60):")
    print(f"    {'':>25}  {'p50':>8}  {'p99':>8}  {'mean':>8}")
    print(f"    {'Composed (CuPy blocks)':>25}  {comp_p50:8.3f}  {comp_p99:8.3f}  {comp_mean:8.3f}")
    print(f"    {'Monolithic (JIT kernel)':>25}  {mono_p50:8.3f}  {mono_p99:8.3f}  {mono_mean:8.3f}")
    print(f"    {'speedup (mono/comp)':>25}  {comp_p50/mono_p50:8.2f}x")

    # Count kernel launches in each path
    # Composed: ~10 CuPy ops (slice, subtract, divide, maximum, sqrt, slice, subtract, divide, slice, divide)
    # Monolithic: 1 kernel launch
    print(f"\n  Kernel launches: composed ~10, monolithic 1")
    print(f"  At 0.07 ms/launch overhead: composed overhead ~0.7 ms")

    del prices, volumes, price_cs, price_sq_cs, vol_cs
    cp.get_default_memory_pool().free_all_blocks()

    # ── Cache behavior: compilation amortization ──────────────────
    print("\n--- JIT Cache Behavior ---\n")

    # Simulate the lifecycle: compile once, use many times
    # What's the break-even point?
    tag = hashlib.md5(str(time.perf_counter()).encode()).hexdigest()[:8]
    kname = f"cache_test_{tag}"
    src = SIMPLE_KERNEL.replace("void simple(", f"void {kname}(")

    data = cp.random.uniform(-1, 1, 1_000_000).astype(cp.float32)
    out = cp.empty(1_000_000, dtype=cp.float64)
    threads = 256
    blocks = (1_000_000 + threads - 1) // threads

    t0 = time.perf_counter()
    kernel = cp.RawKernel(src, kname)
    # First call (compilation + execution)
    kernel((blocks,), (threads,), (data, out, 1_000_000))
    cp.cuda.Stream.null.synchronize()
    t_first = (time.perf_counter() - t0) * 1000

    # Subsequent calls (cached)
    cached_times = []
    for _ in range(100):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        kernel((blocks,), (threads,), (data, out, 1_000_000))
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        cached_times.append((t1 - t0) * 1000)

    cached_times.sort()
    cached_p50 = cached_times[50]

    # Break-even: after how many calls does JIT overhead amortize to <1% of total?
    jit_overhead = t_first - cached_p50
    breakeven_1pct = int(jit_overhead / (cached_p50 * 0.01)) if cached_p50 > 0 else 0

    print(f"  First call (compile + execute): {t_first:.1f} ms")
    print(f"  Cached call (execute only):     {cached_p50:.3f} ms")
    print(f"  JIT overhead:                   {jit_overhead:.1f} ms")
    print(f"  Break-even (1% amortization):   {breakeven_1pct} calls")
    print(f"  At 11K queries/sec:             {breakeven_1pct/11000*1000:.0f} ms")

    print(f"\n{'=' * 70}")
    print("E05 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
