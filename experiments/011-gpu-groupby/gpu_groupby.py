"""
WinRapids Experiment 011: GPU GroupBy

The expedition log called out the gap: "You can launch CUDA kernels on Windows
today. You just can't do df.groupby('category').sum() on a GPU DataFrame."

This experiment builds GPU groupby using two approaches:
  1. CuPy sort-based: sort by keys, find group boundaries, segmented reduce
  2. CuPy hash-based: atomic scatter into hash buckets (for low-cardinality keys)
  3. Custom CUDA kernel: fused sort + segmented warp-shuffle reduction

Both compared against pandas groupby.

Architecture:
  - Sort-based is better for high-cardinality keys (many groups)
  - Hash-based is better for low-cardinality keys (few groups)
  - The sort step dominates cost — segmented reduction is fast once sorted

Test cases:
  - 10M rows, 100 groups (typical: category, state, etc.)
  - 10M rows, 10K groups (moderate: zip code, product ID)
  - 10M rows, 1M groups (high: user ID, session ID)
"""

from __future__ import annotations

import time
import numpy as np
import cupy as cp
import pandas as pd


# ============================================================
# GPU GroupBy: Sort-based approach
# ============================================================

def groupby_sum_sort(keys: cp.ndarray, values: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    GroupBy sum using sort + segmented reduction.

    1. Sort keys, get sort order
    2. Reorder values by sort order
    3. Find group boundaries (where keys change)
    4. Segmented sum between boundaries

    Returns (unique_keys, group_sums).
    """
    # Sort keys and get permutation
    sort_idx = cp.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_vals = values[sort_idx]

    # Find group boundaries
    # diff != 0 means key changed; prepend True for first group
    boundaries = cp.concatenate([
        cp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1]
    ])
    boundary_idx = cp.where(boundaries)[0]

    # Unique keys at boundary positions
    unique_keys = sorted_keys[boundary_idx]

    # Segmented sum using cumsum trick:
    # cumsum of values, then diff at boundaries
    cumsum = cp.cumsum(sorted_vals)

    # Group sums = cumsum[end] - cumsum[start-1]
    n_groups = len(boundary_idx)
    # End indices: next boundary - 1, or last element
    end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([len(values) - 1])])

    group_sums = cumsum[end_idx].copy()
    # Subtract cumsum[start-1] for all groups except the first
    group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

    return unique_keys, group_sums


def groupby_mean_sort(keys: cp.ndarray, values: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """GroupBy mean: sum / count per group."""
    sort_idx = cp.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_vals = values[sort_idx]

    boundaries = cp.concatenate([
        cp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1]
    ])
    boundary_idx = cp.where(boundaries)[0]
    unique_keys = sorted_keys[boundary_idx]

    cumsum = cp.cumsum(sorted_vals)
    end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([len(values) - 1])])

    group_sums = cumsum[end_idx].copy()
    group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

    # Count per group
    group_counts = cp.diff(cp.concatenate([boundary_idx, cp.array([len(values)])]))

    return unique_keys, group_sums / group_counts


# ============================================================
# GPU GroupBy: Hash-based approach (low cardinality)
# ============================================================

# Atomic add kernel for hash-based groupby
_atomic_sum_kernel = cp.RawKernel(r"""
extern "C" __global__
void atomic_groupby_sum(const int* keys, const double* values,
                        double* group_sums, int* group_counts,
                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int key = keys[idx];
        atomicAdd(&group_sums[key], values[idx]);
        atomicAdd(&group_counts[key], 1);
    }
}
""", "atomic_groupby_sum")


def groupby_sum_hash(keys: cp.ndarray, values: cp.ndarray,
                     n_groups: int) -> tuple[cp.ndarray, cp.ndarray]:
    """
    GroupBy sum using atomic operations.

    Requires keys to be contiguous integers [0, n_groups).
    Fast for low cardinality (few groups), but atomic contention
    hurts with many threads hitting the same bucket.
    """
    group_sums = cp.zeros(n_groups, dtype=cp.float64)
    group_counts = cp.zeros(n_groups, dtype=cp.int32)

    n = len(keys)
    threads = 256
    blocks = (n + threads - 1) // threads

    _atomic_sum_kernel(
        (blocks,), (threads,),
        (keys.astype(cp.int32), values, group_sums, group_counts, n)
    )

    # Return only non-empty groups
    mask = group_counts > 0
    return cp.arange(n_groups)[mask], group_sums[mask]


def groupby_mean_hash(keys: cp.ndarray, values: cp.ndarray,
                      n_groups: int) -> tuple[cp.ndarray, cp.ndarray]:
    """GroupBy mean using atomic scatter."""
    group_sums = cp.zeros(n_groups, dtype=cp.float64)
    group_counts = cp.zeros(n_groups, dtype=cp.int32)

    n = len(keys)
    threads = 256
    blocks = (n + threads - 1) // threads

    _atomic_sum_kernel(
        (blocks,), (threads,),
        (keys.astype(cp.int32), values, group_sums, group_counts, n)
    )

    mask = group_counts > 0
    unique_keys = cp.arange(n_groups)[mask]
    means = group_sums[mask] / group_counts[mask].astype(cp.float64)
    return unique_keys, means


# ============================================================
# Benchmarks
# ============================================================

def bench(name, fn, warmup=2, runs=10):
    for _ in range(warmup):
        fn()
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def benchmark_groupby(n: int, n_groups: int):
    """Benchmark groupby sum at a given cardinality."""
    print(f"\n=== GroupBy Sum: {n:,} rows, {n_groups:,} groups ===\n")

    rng = np.random.default_rng(42)
    keys_np = rng.integers(0, n_groups, size=n).astype(np.int64)
    values_np = rng.standard_normal(n).astype(np.float64)

    # --- pandas baseline ---
    pdf = pd.DataFrame({"key": keys_np, "value": values_np})
    _ = pdf.groupby("key")["value"].sum()

    t0 = time.perf_counter()
    for _ in range(5):
        pd_result = pdf.groupby("key")["value"].sum()
    t_pandas = (time.perf_counter() - t0) / 5 * 1000

    # --- GPU sort-based ---
    keys_gpu = cp.asarray(keys_np)
    values_gpu = cp.asarray(values_np)
    cp.cuda.Device(0).synchronize()

    ms_sort = bench("sort", lambda: groupby_sum_sort(keys_gpu, values_gpu))

    # --- GPU hash-based ---
    ms_hash = bench("hash", lambda: groupby_sum_hash(keys_gpu, values_gpu, n_groups))

    # Verify
    gpu_keys_sort, gpu_sums_sort = groupby_sum_sort(keys_gpu, values_gpu)
    gpu_keys_hash, gpu_sums_hash = groupby_sum_hash(keys_gpu, values_gpu, n_groups)

    # Compare against pandas
    pd_sorted = pd_result.sort_index()
    gpu_sorted_sort = cp.asnumpy(gpu_sums_sort[cp.argsort(gpu_keys_sort)])
    gpu_sorted_hash = cp.asnumpy(gpu_sums_hash[cp.argsort(gpu_keys_hash)])

    max_err_sort = np.max(np.abs(pd_sorted.values - gpu_sorted_sort))
    max_err_hash = np.max(np.abs(pd_sorted.values - gpu_sorted_hash))

    print(f"  pandas:          {t_pandas:8.2f} ms")
    print(f"  GPU sort-based:  {ms_sort:8.2f} ms  (speedup: {t_pandas/ms_sort:.1f}x)  err={max_err_sort:.2e}")
    print(f"  GPU hash-based:  {ms_hash:8.2f} ms  (speedup: {t_pandas/ms_hash:.1f}x)  err={max_err_hash:.2e}")
    print()

    return t_pandas, ms_sort, ms_hash


def benchmark_groupby_mean(n: int, n_groups: int):
    """Benchmark groupby mean."""
    print(f"\n=== GroupBy Mean: {n:,} rows, {n_groups:,} groups ===\n")

    rng = np.random.default_rng(42)
    keys_np = rng.integers(0, n_groups, size=n).astype(np.int64)
    values_np = rng.standard_normal(n).astype(np.float64)

    # pandas
    pdf = pd.DataFrame({"key": keys_np, "value": values_np})
    _ = pdf.groupby("key")["value"].mean()

    t0 = time.perf_counter()
    for _ in range(5):
        pd_result = pdf.groupby("key")["value"].mean()
    t_pandas = (time.perf_counter() - t0) / 5 * 1000

    # GPU sort-based
    keys_gpu = cp.asarray(keys_np)
    values_gpu = cp.asarray(values_np)
    cp.cuda.Device(0).synchronize()

    ms_sort = bench("sort", lambda: groupby_mean_sort(keys_gpu, values_gpu))

    # GPU hash-based
    ms_hash = bench("hash", lambda: groupby_mean_hash(keys_gpu, values_gpu, n_groups))

    print(f"  pandas:          {t_pandas:8.2f} ms")
    print(f"  GPU sort-based:  {ms_sort:8.2f} ms  (speedup: {t_pandas/ms_sort:.1f}x)")
    print(f"  GPU hash-based:  {ms_hash:8.2f} ms  (speedup: {t_pandas/ms_hash:.1f}x)")
    print()


def benchmark_multi_agg(n: int, n_groups: int):
    """Benchmark groupby with multiple aggregations (sum + mean + count)."""
    print(f"\n=== GroupBy Multi-Agg (sum, mean, count): {n:,} rows, {n_groups:,} groups ===\n")

    rng = np.random.default_rng(42)
    keys_np = rng.integers(0, n_groups, size=n).astype(np.int64)
    values_np = rng.standard_normal(n).astype(np.float64)

    # pandas
    pdf = pd.DataFrame({"key": keys_np, "value": values_np})
    _ = pdf.groupby("key")["value"].agg(["sum", "mean", "count"])

    t0 = time.perf_counter()
    for _ in range(5):
        _ = pdf.groupby("key")["value"].agg(["sum", "mean", "count"])
    t_pandas = (time.perf_counter() - t0) / 5 * 1000

    # GPU: sort once, compute sum + mean from same sorted data
    keys_gpu = cp.asarray(keys_np)
    values_gpu = cp.asarray(values_np)
    cp.cuda.Device(0).synchronize()

    def gpu_multi_agg():
        sort_idx = cp.argsort(keys_gpu)
        sorted_keys = keys_gpu[sort_idx]
        sorted_vals = values_gpu[sort_idx]

        boundaries = cp.concatenate([
            cp.array([True]),
            sorted_keys[1:] != sorted_keys[:-1]
        ])
        boundary_idx = cp.where(boundaries)[0]
        end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([len(values_gpu) - 1])])

        cumsum = cp.cumsum(sorted_vals)
        group_sums = cumsum[end_idx].copy()
        group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

        group_counts = cp.diff(cp.concatenate([boundary_idx, cp.array([len(values_gpu)])]))
        group_means = group_sums / group_counts

        return sorted_keys[boundary_idx], group_sums, group_means, group_counts

    ms_gpu = bench("multi", gpu_multi_agg)

    print(f"  pandas multi-agg:    {t_pandas:8.2f} ms")
    print(f"  GPU sort+multi-agg:  {ms_gpu:8.2f} ms  (speedup: {t_pandas/ms_gpu:.1f}x)")
    print(f"  Note: GPU sorts once, derives sum+mean+count from same sorted data")
    print()


def main():
    print("WinRapids Experiment 011: GPU GroupBy")
    print("=" * 60)

    n = 10_000_000

    # Test at different cardinalities
    benchmark_groupby(n, 100)        # Low: categories, states
    benchmark_groupby(n, 10_000)     # Medium: zip codes, products
    benchmark_groupby(n, 1_000_000)  # High: user IDs

    # Mean aggregation
    benchmark_groupby_mean(n, 100)

    # Multi-aggregation
    benchmark_multi_agg(n, 100)

    print("=" * 60)
    print("Experiment 011 complete.")


if __name__ == "__main__":
    main()
