"""
WinRapids Experiment 013: GPU Hash Join

Joins are the other fundamental relational operation alongside groupby.
This experiment implements GPU inner join using three approaches:

1. Sort-merge join: sort both tables by key, binary search
2. Hash join via direct indexing: for dense integer keys, just use array indexing
3. CuPy-based hash join: searchsorted on sorted keys

Test case: classic star schema join
- Fact table: 10M rows (orders with product_id, quantity, price)
- Dimension table: 10K-1M rows (products with product_id, category)
- Join: fact.product_id = dim.product_id (inner join)
"""

from __future__ import annotations

import time
import numpy as np
import cupy as cp
import pandas as pd


# ============================================================
# GPU Join: Sort-merge via searchsorted
# ============================================================

def gpu_sort_merge_join(fact_keys: cp.ndarray, dim_keys: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    GPU sort-merge join using CuPy's searchsorted.

    For unique dimension keys (typical star schema):
    1. Sort dimension keys, keep permutation
    2. Binary search each fact key in sorted dimension keys
    3. Filter matches
    """
    dim_sort_idx = cp.argsort(dim_keys)
    sorted_dim_keys = dim_keys[dim_sort_idx]

    positions = cp.searchsorted(sorted_dim_keys, fact_keys)

    valid = positions < len(dim_keys)
    valid_pos = cp.where(valid, positions, 0)
    matches = valid & (sorted_dim_keys[valid_pos] == fact_keys)

    fact_idx = cp.where(matches)[0]
    dim_idx = dim_sort_idx[valid_pos[fact_idx]]

    return fact_idx.astype(cp.int32), dim_idx.astype(cp.int32)


# ============================================================
# GPU Join: Direct index (for dense integer keys 0..N-1)
# ============================================================

def gpu_direct_join(fact_keys: cp.ndarray, dim_keys: cp.ndarray,
                    n_dim: int) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Direct-index join for dense integer keys [0, n_dim).

    Instead of building a hash table, use the key itself as the index
    into a lookup array. O(1) per lookup, no hash collisions.

    This is the fastest possible join for integer keys.
    """
    # Build lookup: key -> dim row index
    # For arange keys, lookup[key] = key. But this works for any
    # unique integer keys in [0, max_key].
    lookup = cp.full(n_dim, -1, dtype=cp.int32)
    lookup[dim_keys.astype(cp.int64)] = cp.arange(len(dim_keys), dtype=cp.int32)

    # Probe: each fact key looks up its dim index
    fact_keys_bounded = cp.clip(fact_keys, 0, n_dim - 1)
    dim_indices = lookup[fact_keys_bounded.astype(cp.int64)]

    # Filter: only keep matches (dim_indices != -1 AND key was in range)
    valid = (fact_keys >= 0) & (fact_keys < n_dim) & (dim_indices >= 0)
    fact_idx = cp.where(valid)[0].astype(cp.int32)
    dim_idx = dim_indices[fact_idx]

    return fact_idx, dim_idx


# ============================================================
# GPU Join: Custom CUDA hash table (for arbitrary keys)
# ============================================================

_hash_join_kernel = cp.RawKernel(r"""
extern "C" __global__
void hash_join_probe(
    const long long* fact_keys,
    const long long* ht_keys,     // hash table slots (key)
    const int* ht_values,          // hash table slots (dim row idx)
    int* match_dim_idx,            // output: dim index per fact row (-1 = no match)
    int n_fact,
    int ht_mask                    // ht_size - 1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_fact) return;

    long long key = fact_keys[idx];
    // Fibonacci hash
    unsigned long long h = (unsigned long long)key * 0x9E3779B97F4A7C15ULL;
    int slot = (int)(h >> 33) & ht_mask;

    for (int probe = 0; probe <= ht_mask; probe++) {
        long long slot_key = ht_keys[slot];
        if (slot_key == key) {
            match_dim_idx[idx] = ht_values[slot];
            return;
        }
        if (slot_key == -1LL) {
            match_dim_idx[idx] = -1;
            return;
        }
        slot = (slot + 1) & ht_mask;
    }
    match_dim_idx[idx] = -1;
}
""", "hash_join_probe")


def gpu_hash_join(fact_keys: cp.ndarray, dim_keys: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    GPU hash join for arbitrary int64 keys.

    Build phase: done on CPU (small dim table), uploaded to GPU.
    Probe phase: massively parallel on GPU.
    """
    n_fact = len(fact_keys)
    n_dim = len(dim_keys)

    # Hash table size: 2x dim, power of 2
    ht_size = 1
    while ht_size < n_dim * 2:
        ht_size *= 2
    ht_mask = ht_size - 1

    # Build hash table ON CPU (dim table is small, avoid GPU atomicCAS issues)
    dim_keys_np = cp.asnumpy(dim_keys)
    ht_keys_np = np.full(ht_size, -1, dtype=np.int64)
    ht_values_np = np.full(ht_size, -1, dtype=np.int32)

    for i in range(n_dim):
        key = int(dim_keys_np[i])
        h = (key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        slot = int((h >> 33) & ht_mask)
        while ht_keys_np[slot] != -1:
            slot = (slot + 1) & ht_mask
        ht_keys_np[slot] = key
        ht_values_np[slot] = i

    # Upload hash table to GPU
    ht_keys_gpu = cp.asarray(ht_keys_np)
    ht_values_gpu = cp.asarray(ht_values_np)

    # Probe on GPU (massively parallel)
    match_dim_idx = cp.full(n_fact, -1, dtype=cp.int32)
    threads = 256
    blocks = (n_fact + threads - 1) // threads

    _hash_join_kernel(
        (blocks,), (threads,),
        (fact_keys, ht_keys_gpu, ht_values_gpu, match_dim_idx, n_fact, ht_mask)
    )

    # Filter matches
    valid = match_dim_idx >= 0
    fact_idx = cp.where(valid)[0].astype(cp.int32)
    dim_idx = match_dim_idx[fact_idx]

    return fact_idx, dim_idx


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


def benchmark_join(n_fact: int, n_dim: int, match_rate: float = 1.0):
    """Benchmark inner join at given sizes."""
    print(f"\n=== Inner Join: {n_fact:,} fact x {n_dim:,} dim (match_rate={match_rate}) ===\n")

    rng = np.random.default_rng(42)
    dim_keys_np = np.arange(n_dim, dtype=np.int64)

    if match_rate >= 1.0:
        fact_keys_np = rng.integers(0, n_dim, size=n_fact).astype(np.int64)
    else:
        n_matching = int(n_fact * match_rate)
        n_nonmatching = n_fact - n_matching
        matching_keys = rng.integers(0, n_dim, size=n_matching).astype(np.int64)
        nonmatching_keys = rng.integers(n_dim, n_dim * 2, size=n_nonmatching).astype(np.int64)
        fact_keys_np = np.concatenate([matching_keys, nonmatching_keys])
        rng.shuffle(fact_keys_np)

    # pandas
    fact_df = pd.DataFrame({"key": fact_keys_np, "val": rng.standard_normal(n_fact)})
    dim_df = pd.DataFrame({"key": dim_keys_np, "dim_val": rng.standard_normal(n_dim)})

    _ = pd.merge(fact_df, dim_df, on="key", how="inner")
    t0 = time.perf_counter()
    for _ in range(5):
        pd_result = pd.merge(fact_df, dim_df, on="key", how="inner")
    t_pandas = (time.perf_counter() - t0) / 5 * 1000
    n_pd = len(pd_result)

    # GPU
    fact_keys_gpu = cp.asarray(fact_keys_np)
    dim_keys_gpu = cp.asarray(dim_keys_np)
    cp.cuda.Device(0).synchronize()

    # Sort-merge
    ms_sort = bench("sort", lambda: gpu_sort_merge_join(fact_keys_gpu, dim_keys_gpu))
    fi_s, di_s = gpu_sort_merge_join(fact_keys_gpu, dim_keys_gpu)

    # Direct index (only for dense integer keys)
    ms_direct = bench("direct", lambda: gpu_direct_join(fact_keys_gpu, dim_keys_gpu, n_dim))
    fi_d, di_d = gpu_direct_join(fact_keys_gpu, dim_keys_gpu, n_dim)

    # Hash join (CPU build + GPU probe)
    ms_hash = bench("hash", lambda: gpu_hash_join(fact_keys_gpu, dim_keys_gpu))
    fi_h, di_h = gpu_hash_join(fact_keys_gpu, dim_keys_gpu)

    # Verify
    def verify(name, fi, di):
        fk = cp.asnumpy(fact_keys_gpu[fi])
        dk = cp.asnumpy(dim_keys_gpu[di])
        n_match = len(fi)
        n_correct = int(np.sum(fk == dk))
        return n_match, n_correct

    n_s, nc_s = verify("sort", fi_s, di_s)
    n_d, nc_d = verify("direct", fi_d, di_d)
    n_h, nc_h = verify("hash", fi_h, di_h)

    print(f"  pandas:              {t_pandas:8.2f} ms  ({n_pd:,} matches)")
    print(f"  GPU sort-merge:      {ms_sort:8.2f} ms  ({n_s:,} matches, {nc_s:,} correct)  {t_pandas/ms_sort:.1f}x")
    print(f"  GPU direct-index:    {ms_direct:8.2f} ms  ({n_d:,} matches, {nc_d:,} correct)  {t_pandas/ms_direct:.1f}x")
    print(f"  GPU hash (CPU+GPU):  {ms_hash:8.2f} ms  ({n_h:,} matches, {nc_h:,} correct)  {t_pandas/ms_hash:.1f}x")
    print()


def benchmark_join_then_groupby(n_fact: int, n_dim: int, n_categories: int):
    """End-to-end: join + groupby. Classic analytics pipeline."""
    print(f"\n=== Join + GroupBy: {n_fact:,} fact x {n_dim:,} dim -> groupby({n_categories} categories) ===\n")

    rng = np.random.default_rng(42)
    dim_keys_np = np.arange(n_dim, dtype=np.int64)
    dim_categories_np = rng.integers(0, n_categories, size=n_dim).astype(np.int64)
    fact_keys_np = rng.integers(0, n_dim, size=n_fact).astype(np.int64)
    fact_amounts_np = rng.standard_normal(n_fact).astype(np.float64)

    # pandas
    fact_df = pd.DataFrame({"product_id": fact_keys_np, "amount": fact_amounts_np})
    dim_df = pd.DataFrame({"product_id": dim_keys_np, "category": dim_categories_np})

    _ = pd.merge(fact_df, dim_df, on="product_id").groupby("category")["amount"].sum()
    t0 = time.perf_counter()
    for _ in range(5):
        result = pd.merge(fact_df, dim_df, on="product_id").groupby("category")["amount"].sum()
    t_pandas = (time.perf_counter() - t0) / 5 * 1000

    # GPU: direct join + scatter groupby
    fact_keys = cp.asarray(fact_keys_np)
    dim_keys = cp.asarray(dim_keys_np)
    dim_categories = cp.asarray(dim_categories_np)
    fact_amounts = cp.asarray(fact_amounts_np)
    cp.cuda.Device(0).synchronize()

    def gpu_pipeline():
        # Join
        fi, di = gpu_direct_join(fact_keys, dim_keys, n_dim)
        # Gather
        joined_amounts = fact_amounts[fi]
        joined_categories = dim_categories[di]
        # GroupBy sum via bincount
        weights = cp.zeros(n_categories, dtype=cp.float64)
        # Use advanced indexing + bincount approach
        for cat in range(n_categories):
            mask = joined_categories == cat
            if cp.any(mask):
                weights[cat] = float(cp.sum(joined_amounts[mask]))
        return weights

    # Faster version: use sort-based groupby
    def gpu_pipeline_fast():
        fi, di = gpu_direct_join(fact_keys, dim_keys, n_dim)
        joined_amounts = fact_amounts[fi]
        joined_categories = dim_categories[di]

        # Sort by category and cumsum reduce
        sort_idx = cp.argsort(joined_categories)
        sorted_cats = joined_categories[sort_idx]
        sorted_vals = joined_amounts[sort_idx]

        boundaries = cp.concatenate([cp.array([True]), sorted_cats[1:] != sorted_cats[:-1]])
        boundary_idx = cp.where(boundaries)[0]
        cumsum = cp.cumsum(sorted_vals)
        end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([len(sorted_vals) - 1])])
        group_sums = cumsum[end_idx].copy()
        group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

        return sorted_cats[boundary_idx], group_sums

    ms_gpu = bench("gpu_fast", gpu_pipeline_fast)

    # Verify
    gpu_cats, gpu_sums = gpu_pipeline_fast()
    gpu_result = dict(zip(cp.asnumpy(gpu_cats), cp.asnumpy(gpu_sums)))
    pd_result = result.sort_index()

    max_err = 0
    for cat in pd_result.index:
        if cat in gpu_result:
            err = abs(pd_result[cat] - gpu_result[cat])
            max_err = max(max_err, err)

    print(f"  pandas (join+groupby):  {t_pandas:8.2f} ms")
    print(f"  GPU (join+groupby):     {ms_gpu:8.2f} ms  ({t_pandas/ms_gpu:.1f}x)")
    print(f"  Max error:              {max_err:.2e}")
    print()


def main():
    print("WinRapids Experiment 013: GPU Hash Join")
    print("=" * 60)

    benchmark_join(10_000_000, 10_000)
    benchmark_join(10_000_000, 100_000)
    benchmark_join(10_000_000, 1_000_000)
    benchmark_join(10_000_000, 10_000, match_rate=0.5)

    benchmark_join_then_groupby(10_000_000, 10_000, 50)

    print("=" * 60)
    print("Experiment 013 complete.")


if __name__ == "__main__":
    main()
