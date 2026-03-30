"""
E03 -- Cross-Algorithm Sharing: rolling -> z_score -> PCA

Key question (from team lead):
  Does cross-algorithm sharing produce measurably different performance,
  or is the overhead of tracking shared state worse than recomputing?

The compiler vision proposes that algorithms can share intermediate results:
  - rolling_mean feeds z_score (z = (x - rolling_mean) / rolling_std)
  - rolling_std feeds z_score
  - z_score feeds PCA (PCA needs centered/standardized data)
  - rolling_mean and rolling_std share a rolling window scan

If the compiler detects these shared dependencies, it can eliminate
redundant computation. But tracking shared state has overhead:
  - Memory to store intermediates
  - Pipeline stalls waiting for shared results
  - Complexity in the scheduler

This experiment compares:
  A) Independent computation: each algorithm runs from scratch
  B) Shared computation: intermediates are reused across algorithms
  C) Fully fused: hand-written pipeline that combines everything

Sizes represent realistic signal farm workloads:
  - 1M rows = ~1 day of 1-second bars for one ticker
  - 10M rows = ~10 days or 1 day at 100ms resolution
"""

import time
import numpy as np
import cupy as cp


# ============================================================
# Rolling statistics (shared primitive)
# ============================================================

def rolling_mean_naive(data, window):
    """Rolling mean via cumsum trick."""
    cs = cp.cumsum(data)
    cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
    return (cs[window:] - cs[:-window]) / window


def rolling_std_naive(data, window):
    """Rolling std via cumsum of squares trick."""
    n = len(data)
    cs = cp.cumsum(data)
    cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
    cs2 = cp.cumsum(data * data)
    cs2 = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs2])

    mean = (cs[window:] - cs[:-window]) / window
    mean_sq = (cs2[window:] - cs2[:-window]) / window
    var = mean_sq - mean * mean
    # Clamp negative variance from floating point
    var = cp.maximum(var, 0)
    return cp.sqrt(var)


def rolling_mean_std_shared(data, window):
    """Rolling mean AND std from shared cumsums (one pass)."""
    cs = cp.cumsum(data)
    cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
    cs2 = cp.cumsum(data * data)
    cs2 = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs2])

    mean = (cs[window:] - cs[:-window]) / window
    mean_sq = (cs2[window:] - cs2[:-window]) / window
    var = cp.maximum(mean_sq - mean * mean, 0)
    std = cp.sqrt(var)
    return mean, std


# ============================================================
# Z-score
# ============================================================

def z_score_independent(data, window):
    """Z-score computing its own rolling stats from scratch."""
    rm = rolling_mean_naive(data, window)
    rs = rolling_std_naive(data, window)
    trimmed = data[window - 1:]
    z = (trimmed - rm) / cp.maximum(rs, 1e-10)
    return z


def z_score_from_shared(data_trimmed, rm, rs):
    """Z-score using pre-computed rolling stats."""
    return (data_trimmed - rm) / cp.maximum(rs, 1e-10)


# ============================================================
# Simple PCA-like operation (covariance on standardized data)
# ============================================================

def pca_independent(data_cols, window):
    """PCA-like: z-score each column independently, then covariance."""
    z_cols = []
    for col in data_cols:
        z = z_score_independent(col, window)
        z_cols.append(z)

    # Covariance matrix (simplified: just cross-products)
    n_cols = len(z_cols)
    n_rows = len(z_cols[0])
    cov = cp.empty((n_cols, n_cols), dtype=cp.float64)
    for i in range(n_cols):
        for j in range(i, n_cols):
            c = float(cp.sum(z_cols[i] * z_cols[j])) / n_rows
            cov[i, j] = c
            cov[j, i] = c
    return cov


def pca_shared(data_cols, window):
    """PCA-like: shared rolling stats -> z-score -> covariance."""
    z_cols = []
    for col in data_cols:
        rm, rs = rolling_mean_std_shared(col, window)
        trimmed = col[window - 1:]
        z = z_score_from_shared(trimmed, rm, rs)
        z_cols.append(z)

    n_cols = len(z_cols)
    n_rows = len(z_cols[0])
    cov = cp.empty((n_cols, n_cols), dtype=cp.float64)
    for i in range(n_cols):
        for j in range(i, n_cols):
            c = float(cp.sum(z_cols[i] * z_cols[j])) / n_rows
            cov[i, j] = c
            cov[j, i] = c
    return cov


# ============================================================
# Fully fused: single pass through data per column
# ============================================================

FUSED_ROLLING_ZSCORE = cp.RawKernel(r"""
extern "C" __global__
void fused_rolling_zscore(const float* data, float* z_out,
                          const double* cumsum, const double* cumsum_sq,
                          int n, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_idx = idx;  // output starts at 0
    int data_idx = idx + window - 1;  // data index for the "current" value

    if (data_idx < n) {
        // Rolling mean and std from precomputed cumsums
        double sum_w = cumsum[data_idx + 1] - cumsum[data_idx + 1 - window];
        double sq_w = cumsum_sq[data_idx + 1] - cumsum_sq[data_idx + 1 - window];
        double mean = sum_w / (double)window;
        double mean_sq = sq_w / (double)window;
        double var = mean_sq - mean * mean;
        if (var < 0.0) var = 0.0;
        double std = sqrt(var);

        double val = (double)data[data_idx];
        double z = (std > 1e-10) ? (val - mean) / std : 0.0;
        z_out[out_idx] = (float)z;
    }
}
""", "fused_rolling_zscore")


def pca_fused(data_cols, window):
    """Fully fused: precompute cumsums, then fused kernel for z-score."""
    z_cols = []
    for col in data_cols:
        # Precompute cumsums (these could also be fused but cumsum is
        # already efficient as a CuPy primitive)
        cs = cp.cumsum(col.astype(cp.float64))
        cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), cs])
        cs2 = cp.cumsum((col.astype(cp.float64)) ** 2)
        cs2 = cp.concatenate([cp.zeros(1, dtype=cp.float64), cs2])

        n_out = len(col) - window + 1
        z = cp.empty(n_out, dtype=cp.float32)
        threads = 256
        blocks = (n_out + threads - 1) // threads
        FUSED_ROLLING_ZSCORE((blocks,), (threads,),
                             (col, z, cs, cs2, len(col), window))
        z_cols.append(z)

    # Covariance
    n_cols = len(z_cols)
    n_rows = len(z_cols[0])
    cov = cp.empty((n_cols, n_cols), dtype=cp.float64)
    for i in range(n_cols):
        for j in range(i, n_cols):
            c = float(cp.sum(z_cols[i].astype(cp.float64) * z_cols[j].astype(cp.float64))) / n_rows
            cov[i, j] = c
            cov[j, i] = c
    return cov


def bench(fn, *args, warmup=3, runs=15, label=""):
    for _ in range(warmup):
        fn(*args)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn(*args)
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
    print("E03 -- Cross-Algorithm Sharing: rolling -> z_score -> PCA")
    print("=" * 70)

    window = 60  # 60-second rolling window (realistic for 1s bars)
    n_cols = 5   # 5 features (like OHLCV or 5 tickers)

    sizes = [100_000, 1_000_000, 10_000_000]
    size_labels = ["100K", "1M", "10M"]

    # First: measure the building blocks independently
    print(f"\n--- Building block costs (single column, window={window}) ---\n")
    for size, label in zip(sizes, size_labels):
        data = cp.random.uniform(-1, 1, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()

        ops = {
            "rolling_mean (naive)": lambda: rolling_mean_naive(data, window),
            "rolling_std (naive)": lambda: rolling_std_naive(data, window),
            "rolling_mean+std (shared)": lambda: rolling_mean_std_shared(data, window),
            "z_score (independent)": lambda: z_score_independent(data, window),
        }

        print(f"  {label}:")
        for op_name, op_fn in ops.items():
            r = bench(op_fn, label=op_name)
            print(f"    {op_name:>30}: {r['p50']:.3f} ms")

        # Shared savings
        rm_naive = bench(lambda: rolling_mean_naive(data, window))
        rs_naive = bench(lambda: rolling_std_naive(data, window))
        shared = bench(lambda: rolling_mean_std_shared(data, window))
        savings_pct = (1 - shared['p50'] / (rm_naive['p50'] + rs_naive['p50'])) * 100
        print(f"    {'shared savings':>30}: {savings_pct:.0f}% vs computing separately")

        del data
        cp.get_default_memory_pool().free_all_blocks()

    # Main experiment: full pipeline
    print(f"\n{'=' * 70}")
    print(f"Full pipeline: {n_cols} columns, rolling z-score -> covariance")
    print(f"{'=' * 70}")

    for size, label in zip(sizes, size_labels):
        print(f"\n--- {label} rows x {n_cols} columns ---")

        data_cols = [cp.random.uniform(-1, 1, size).astype(cp.float32)
                     for _ in range(n_cols)]
        cp.cuda.Stream.null.synchronize()

        # Correctness
        cov_indep = pca_independent(data_cols, window)
        cov_shared = pca_shared(data_cols, window)
        cov_fused = pca_fused(data_cols, window)

        max_err_shared = float(cp.max(cp.abs(cov_indep - cov_shared)))
        max_err_fused = float(cp.max(cp.abs(cov_indep - cov_fused)))
        print(f"  Correctness: shared_err={max_err_shared:.2e}, fused_err={max_err_fused:.2e}")

        # Benchmark all three
        a = bench(pca_independent, data_cols, window, label="independent")
        b = bench(pca_shared, data_cols, window, label="shared intermediates")
        c = bench(pca_fused, data_cols, window, label="fused kernel")

        print(f"\n  Results (ms, p50):")
        print(f"    {'':>30}  {'p50':>8}  {'p99':>8}  {'mean':>8}")
        print(f"    {'A: independent':>30}  {a['p50']:8.3f}  {a['p99']:8.3f}  {a['mean']:8.3f}")
        print(f"    {'B: shared intermediates':>30}  {b['p50']:8.3f}  {b['p99']:8.3f}  {b['mean']:8.3f}")
        print(f"    {'C: fused kernel':>30}  {c['p50']:8.3f}  {c['p99']:8.3f}  {c['mean']:8.3f}")
        print(f"\n  Speedups (p50):")
        print(f"    B vs A (shared vs independent): {a['p50']/b['p50']:.2f}x")
        print(f"    C vs A (fused vs independent):  {a['p50']/c['p50']:.2f}x")
        print(f"    C vs B (fused vs shared):       {b['p50']/c['p50']:.2f}x")

        # Memory analysis
        data_bytes = size * 4 * n_cols
        # Independent: each z_score_independent computes 2 cumsums + 2 cumsum_sq
        # = 4 cumsum per column, 5 columns = 20 cumsums
        # Shared: 2 cumsums per column (sum + sq), 5 columns = 10 cumsums
        # Fused: same as shared (cumsums precomputed, but z-score is fused)
        print(f"\n  Intermediate memory:")
        print(f"    Data: {data_bytes/1e6:.0f} MB")
        print(f"    Independent: ~{n_cols * 4 * size * 8 / 1e6:.0f} MB (4 cumsum arrays x {n_cols} cols x f64)")
        print(f"    Shared: ~{n_cols * 2 * size * 8 / 1e6:.0f} MB (2 cumsum arrays x {n_cols} cols x f64)")
        print(f"    Fused: ~{n_cols * 2 * size * 8 / 1e6:.0f} MB (same cumsums + fused z-score)")

        del data_cols
        cp.get_default_memory_pool().free_all_blocks()

    # Isolate the overhead of tracking shared state
    print(f"\n{'=' * 70}")
    print("Overhead test: dict lookup + memory management cost")
    print("=" * 70)

    # Simulate the compiler's intermediate cache
    data = cp.random.uniform(-1, 1, 10_000_000).astype(cp.float32)
    cache = {}

    def with_cache(data, window):
        """Simulate shared computation with a cache dict."""
        key = ("rolling_mean_std", id(data), window)
        if key not in cache:
            cache[key] = rolling_mean_std_shared(data, window)
        return cache[key]

    def without_cache(data, window):
        """Direct computation."""
        return rolling_mean_std_shared(data, window)

    # Benchmark cache lookup overhead
    cache.clear()
    cached = bench(with_cache, data, window)
    direct = bench(without_cache, data, window)
    overhead_ms = cached["p50"] - direct["p50"]
    print(f"\n  Cache lookup overhead at 10M: {overhead_ms:.3f} ms "
          f"({overhead_ms/direct['p50']*100:.1f}% of computation)")

    print(f"\n{'=' * 70}")
    print("E03 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
