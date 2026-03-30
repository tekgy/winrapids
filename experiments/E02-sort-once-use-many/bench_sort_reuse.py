"""
E02 -- Sort-Once-Use-Many: Does reusing a single sort beat independent sorts?

Key question (from team lead):
  Is sort-once-use-many faster, or does CuPy's internal caching already
  handle it?

What we test:
  Given an array that needs: groupby, rank, dedup, and percentile --
  all of which require or benefit from sorted order --
  is it faster to:
    (A) Sort once, feed sorted result to all four operations
    (B) Let each operation sort independently (CuPy's default)

Why this matters for the compiler:
  If sort-reuse is a significant win, the pipeline generator should detect
  when multiple downstream operations share a sort dependency and lift
  the sort to a common ancestor node. If CuPy already caches this (or the
  cost difference is negligible), the compiler optimization is wasted complexity.

Additional sub-experiment:
  (C) Does CuPy cache sort results? (Test by sorting the same array twice
      and checking if the second sort is faster)
"""

import time
import numpy as np
import cupy as cp


# ============================================================
# Operations that consume sorted data
# ============================================================

def op_groupby_sorted(sorted_keys, sorted_vals):
    """GroupBy sum using sorted keys: find group boundaries, then cumsum."""
    # Group boundaries where key changes
    changes = cp.zeros(len(sorted_keys), dtype=cp.bool_)
    changes[0] = True
    changes[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = cp.nonzero(changes)[0]

    # Cumulative sum of values, then diff at group boundaries
    cumsum = cp.cumsum(sorted_vals)
    # Group sums = cumsum at group ends - cumsum at group starts
    group_ends = cp.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:] - 1
    group_ends[-1] = len(sorted_keys) - 1

    group_sums = cumsum[group_ends]
    group_sums[1:] -= cumsum[group_starts[1:] - 1]
    return group_sums


def op_rank(sorted_indices, n):
    """Rank: given argsort indices, produce ranks (1-based)."""
    ranks = cp.empty(n, dtype=cp.int64)
    ranks[sorted_indices] = cp.arange(1, n + 1, dtype=cp.int64)
    return ranks


def op_dedup(sorted_data):
    """Dedup: unique values from sorted array."""
    if len(sorted_data) == 0:
        return sorted_data
    mask = cp.ones(len(sorted_data), dtype=cp.bool_)
    mask[1:] = sorted_data[1:] != sorted_data[:-1]
    return sorted_data[mask]


def op_percentile(sorted_data, percentiles):
    """Percentile: direct index into sorted array."""
    n = len(sorted_data)
    indices = cp.array([(p / 100.0) * (n - 1) for p in percentiles], dtype=cp.float64)
    lower = cp.floor(indices).astype(cp.int64)
    upper = cp.minimum(lower + 1, n - 1)
    frac = indices - lower.astype(cp.float64)
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


# ============================================================
# Path A: Sort once, reuse for all operations
# ============================================================

def path_sort_once(keys, vals, percentiles):
    """Sort once, feed to all downstream operations."""
    sorted_indices = cp.argsort(keys)
    sorted_keys = keys[sorted_indices]
    sorted_vals = vals[sorted_indices]

    group_sums = op_groupby_sorted(sorted_keys, sorted_vals)
    ranks = op_rank(sorted_indices, len(keys))
    unique = op_dedup(sorted_keys)
    pcts = op_percentile(sorted_vals, percentiles)

    return group_sums, ranks, unique, pcts


# ============================================================
# Path B: Each operation sorts independently
# ============================================================

def path_independent_sorts(keys, vals, percentiles):
    """Each operation sorts independently."""
    # GroupBy: sort keys and vals together
    idx1 = cp.argsort(keys)
    sk1 = keys[idx1]
    sv1 = vals[idx1]
    group_sums = op_groupby_sorted(sk1, sv1)

    # Rank: separate argsort
    idx2 = cp.argsort(keys)
    ranks = op_rank(idx2, len(keys))

    # Dedup: sort keys separately
    sorted_keys = cp.sort(keys)
    unique = op_dedup(sorted_keys)

    # Percentile: sort vals separately
    sorted_vals = cp.sort(vals)
    pcts = op_percentile(sorted_vals, percentiles)

    return group_sums, ranks, unique, pcts


# ============================================================
# Sub-experiment C: Does CuPy cache sort results?
# ============================================================

def test_sort_caching():
    """Test if CuPy caches sort results by sorting the same array twice."""
    print("\n--- Sub-experiment C: Does CuPy cache sort results? ---\n")

    for size, label in [(1_000_000, "1M"), (10_000_000, "10M"), (50_000_000, "50M")]:
        data = cp.random.uniform(0, 1, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()

        # Warmup (different data to avoid any warmup artifacts)
        warmup = cp.random.uniform(0, 1, size).astype(cp.float32)
        for _ in range(3):
            cp.sort(warmup)
        cp.cuda.Stream.null.synchronize()
        del warmup

        # First sort
        times_first = []
        for _ in range(10):
            data_copy = data.copy()
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            cp.sort(data_copy)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times_first.append((t1 - t0) * 1000)
            del data_copy

        # Second sort (same data, already "seen")
        times_second = []
        for _ in range(10):
            # Sort the exact same array object again
            t0 = time.perf_counter()
            cp.sort(data)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times_second.append((t1 - t0) * 1000)

        times_first.sort()
        times_second.sort()
        p50_first = times_first[len(times_first) // 2]
        p50_second = times_second[len(times_second) // 2]

        print(f"  {label}: first sort p50={p50_first:.3f} ms, "
              f"second sort p50={p50_second:.3f} ms, "
              f"ratio={p50_second/p50_first:.2f}x")

        # Also test argsort
        times_argsort1 = []
        times_argsort2 = []
        data2 = cp.random.uniform(0, 1, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()

        for _ in range(10):
            data_copy = data2.copy()
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            cp.argsort(data_copy)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times_argsort1.append((t1 - t0) * 1000)
            del data_copy

        for _ in range(10):
            t0 = time.perf_counter()
            cp.argsort(data2)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times_argsort2.append((t1 - t0) * 1000)

        times_argsort1.sort()
        times_argsort2.sort()
        print(f"  {label}: first argsort p50={times_argsort1[len(times_argsort1)//2]:.3f} ms, "
              f"second argsort p50={times_argsort2[len(times_argsort2)//2]:.3f} ms, "
              f"ratio={times_argsort2[len(times_argsort2)//2]/times_argsort1[len(times_argsort1)//2]:.2f}x")

        del data, data2
        cp.get_default_memory_pool().free_all_blocks()


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


def time_sort_alone(data, warmup=3, runs=15):
    """Time just the sort operation."""
    for _ in range(warmup):
        cp.sort(data.copy())
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(runs):
        d = data.copy()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        cp.argsort(d)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        del d

    times.sort()
    return times[len(times) // 2]


def main():
    print("=" * 70)
    print("E02 -- Sort-Once-Use-Many: Reuse vs Independent Sorts")
    print("=" * 70)

    # Sub-experiment C: caching test
    test_sort_caching()

    # Main experiment
    percentiles = [25, 50, 75, 90, 95, 99]
    sizes = [1_000_000, 10_000_000, 50_000_000]
    n_groups = 1000  # Number of unique keys

    print(f"\n{'=' * 70}")
    print(f"Main experiment: sort-once vs independent sorts")
    print(f"Operations: groupby-sum, rank, dedup, percentile")
    print(f"Key cardinality: {n_groups}")
    print(f"{'=' * 70}")

    for size in sizes:
        print(f"\n--- {size/1e6:.0f}M rows ({size*8/1e6:.0f} MB: keys+vals) ---")

        # Generate data: integer keys (for groupby), float values
        keys = cp.random.randint(0, n_groups, size).astype(cp.int32)
        vals = cp.random.uniform(0, 100, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()

        # Correctness check
        r_once = path_sort_once(keys, vals, percentiles)
        r_indep = path_independent_sorts(keys, vals, percentiles)

        # Group sums should match
        gs_err = float(cp.max(cp.abs(r_once[0] - r_indep[0])))
        # Ranks should match exactly
        rank_match = bool(cp.all(r_once[1] == r_indep[1]))
        # Unique counts should match
        uniq_match = len(r_once[2]) == len(r_indep[2])
        # Percentiles should match
        pct_err = float(cp.max(cp.abs(r_once[3] - r_indep[3])))

        print(f"  Correctness: group_sum_err={gs_err:.2e}, "
              f"rank_match={rank_match}, uniq_match={uniq_match}, "
              f"pct_err={pct_err:.2e}")

        # Time sort alone
        sort_p50 = time_sort_alone(keys)
        print(f"  Sort alone (argsort, p50): {sort_p50:.3f} ms")

        # Benchmark both paths
        a = bench(path_sort_once, keys, vals, percentiles, label="sort-once")
        b = bench(path_independent_sorts, keys, vals, percentiles, label="independent")

        speedup = b["p50"] / a["p50"] if a["p50"] > 0 else float("inf")
        sort_savings = (b["p50"] - a["p50"])  # ms saved
        # Path B does 4 sorts (2 argsort + 2 sort), path A does 1 argsort
        # Savings should be ~3 * sort_time

        print(f"\n  Results (ms, p50):")
        print(f"    {'':>25}  {'p50':>8}  {'p99':>8}  {'mean':>8}")
        print(f"    {'A: sort-once':>25}  {a['p50']:8.3f}  {a['p99']:8.3f}  {a['mean']:8.3f}")
        print(f"    {'B: independent sorts':>25}  {b['p50']:8.3f}  {b['p99']:8.3f}  {b['mean']:8.3f}")
        print(f"    {'speedup (A over B)':>25}  {speedup:8.2f}x")
        print(f"    {'time saved':>25}  {sort_savings:8.3f} ms")
        print(f"    {'sort cost (1x argsort)':>25}  {sort_p50:8.3f} ms")
        print(f"    {'expected savings (3x sort)':>25}  {3*sort_p50:8.3f} ms")

        # Break down: time each downstream op individually (on pre-sorted data)
        sorted_idx = cp.argsort(keys)
        sorted_keys = keys[sorted_idx]
        sorted_vals = vals[sorted_idx]
        cp.cuda.Stream.null.synchronize()

        print(f"\n  Downstream op timing (on pre-sorted data):")
        for op_name, op_fn in [
            ("groupby", lambda: op_groupby_sorted(sorted_keys, sorted_vals)),
            ("rank", lambda: op_rank(sorted_idx, len(keys))),
            ("dedup", lambda: op_dedup(sorted_keys)),
            ("percentile", lambda: op_percentile(sorted_vals, percentiles)),
        ]:
            op_times = []
            for _ in range(3):
                op_fn()
            cp.cuda.Stream.null.synchronize()
            for _ in range(15):
                cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                op_fn()
                cp.cuda.Stream.null.synchronize()
                t1 = time.perf_counter()
                op_times.append((t1 - t0) * 1000)
            op_times.sort()
            print(f"    {op_name:>12}: {op_times[len(op_times)//2]:.3f} ms")

        del keys, vals, sorted_idx, sorted_keys, sorted_vals
        cp.get_default_memory_pool().free_all_blocks()

    print(f"\n{'=' * 70}")
    print("E02 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
