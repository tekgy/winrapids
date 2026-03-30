"""
E06 -- Resident Query Latency: GPU-persistent data, repeated queries

Key question:
  When data is already resident on GPU, what is the actual query latency?
  Phase 1 Entry 025 showed second-query latency at ~5 ms for the analytics
  pipeline. Can we do better? What's the floor?

This experiment measures:
  1. First query (cold: data on CPU, must transfer)
  2. Second query (warm: data on GPU from first query)
  3. Repeated queries on resident data (steady-state latency)
  4. Multiple different queries on the same resident data
  5. How latency scales with data size (resident)

This validates the "GPU as persistent database" concept — if resident query
latency is sub-millisecond, the persistent store becomes viable.
"""

import time
import numpy as np
import cupy as cp


# ============================================================
# Query implementations (all operate on GPU-resident data)
# ============================================================

def query_sum(cols):
    """Simple aggregation: sum of all columns."""
    return [float(cp.sum(c)) for c in cols]


def query_filtered_sum(cols, mask):
    """Filtered aggregation: sum where mask is true."""
    return [float(cp.sum(c[mask])) for c in cols]


def query_groupby_sum(vals, keys, n_groups):
    """GroupBy sum using sort-based approach."""
    idx = cp.argsort(keys)
    sorted_keys = keys[idx]
    sorted_vals = vals[idx]
    changes = cp.zeros(len(keys), dtype=cp.bool_)
    changes[0] = True
    changes[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = cp.nonzero(changes)[0]
    cumsum = cp.cumsum(sorted_vals)
    group_ends = cp.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:] - 1
    group_ends[-1] = len(keys) - 1
    group_sums = cumsum[group_ends]
    group_sums[1:] -= cumsum[group_starts[1:] - 1]
    return group_sums


def query_expression(a, b, c):
    """Fused arithmetic expression: (a * b + c) / (a + 1)."""
    return (a * b + c) / (a + 1)


def query_rolling_mean(data, window=60):
    """Rolling mean via cumsum."""
    cs = cp.cumsum(data)
    cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
    return (cs[window:] - cs[:-window]) / window


def bench_latency(fn, warmup=5, runs=50, label=""):
    """High-precision latency benchmark."""
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
    p50 = times[len(times) // 2]
    p01 = times[0]  # best case
    p99 = times[int(len(times) * 0.99)]
    avg = sum(times) / len(times)
    return {"p50": p50, "p01": p01, "p99": p99, "mean": avg, "all": times}


def main():
    print("=" * 70)
    print("E06 -- Resident Query Latency")
    print("=" * 70)

    # ── Test 1: Cold vs Warm vs Steady-State ──────────────────────
    print("\n--- Test 1: Cold vs Warm vs Steady-State (10M rows, 5 cols) ---\n")

    n = 10_000_000
    n_cols = 5
    n_groups = 1000

    # Create CPU data
    np_cols = [np.random.uniform(0, 100, n).astype(np.float32) for _ in range(n_cols)]
    np_keys = np.random.randint(0, n_groups, n).astype(np.int32)
    np_mask = np.random.random(n) > 0.5

    # Cold query: transfer + compute
    cp.cuda.Stream.null.synchronize()
    cold_times = []
    for _ in range(5):
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        gpu_cols = [cp.asarray(c) for c in np_cols]
        result = query_sum(gpu_cols)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        cold_times.append((t1 - t0) * 1000)
        # Don't delete — keep for warm test
    cold_times.sort()

    # Warm query: data already on GPU
    warm = bench_latency(lambda: query_sum(gpu_cols), label="warm sum")
    print(f"  Cold query (H2D + sum):     p50={cold_times[len(cold_times)//2]:.3f} ms")
    print(f"  Warm query (sum only):      p50={warm['p50']:.3f} ms, "
          f"best={warm['p01']:.3f} ms, p99={warm['p99']:.3f} ms")

    # ── Test 2: Multiple Query Types on Resident Data ─────────────
    print("\n--- Test 2: Query Latency on Resident Data (10M rows) ---\n")

    gpu_keys = cp.asarray(np_keys)
    gpu_mask = cp.asarray(np_mask)
    cp.cuda.Stream.null.synchronize()

    queries = {
        "sum (5 cols)": lambda: query_sum(gpu_cols),
        "filtered sum (5 cols)": lambda: query_filtered_sum(gpu_cols, gpu_mask),
        "groupby sum (1K groups)": lambda: query_groupby_sum(gpu_cols[0], gpu_keys, n_groups),
        "expression (a*b+c)/(a+1)": lambda: query_expression(gpu_cols[0], gpu_cols[1], gpu_cols[2]),
        "rolling mean (w=60)": lambda: query_rolling_mean(gpu_cols[0]),
    }

    for qname, qfn in queries.items():
        r = bench_latency(qfn, label=qname)
        print(f"  {qname:>30}: p50={r['p50']:.3f} ms, "
              f"best={r['p01']:.3f} ms, p99={r['p99']:.3f} ms")

    del gpu_cols, gpu_keys, gpu_mask
    cp.get_default_memory_pool().free_all_blocks()

    # ── Test 3: Latency vs Data Size (Resident) ──────────────────
    print("\n--- Test 3: Latency vs Data Size (resident, sum query) ---\n")

    sizes = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000]

    for size in sizes:
        data = cp.random.uniform(0, 100, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()

        r = bench_latency(lambda: float(cp.sum(data)))
        data_mb = size * 4 / 1e6
        throughput = data_mb / r['p50'] * 1000  # MB/s -> GB/s
        print(f"  {size:>12,} ({data_mb:>6.1f} MB): p50={r['p50']:.3f} ms, "
              f"best={r['p01']:.3f} ms ({throughput:.0f} GB/s)")

        del data
    cp.get_default_memory_pool().free_all_blocks()

    # ── Test 4: Kernel Launch Overhead Floor ──────────────────────
    print("\n--- Test 4: Kernel Launch Overhead (empty kernel baseline) ---\n")

    # Measure the absolute minimum latency: launch an empty kernel
    EMPTY_KERNEL = cp.RawKernel(r"""
    extern "C" __global__
    void empty() {}
    """, "empty")

    # Warmup
    for _ in range(10):
        EMPTY_KERNEL((1,), (1,), ())
    cp.cuda.Stream.null.synchronize()

    empty_times = []
    for _ in range(100):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        EMPTY_KERNEL((1,), (1,), ())
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        empty_times.append((t1 - t0) * 1000)

    empty_times.sort()
    print(f"  Empty kernel:    p50={empty_times[50]:.3f} ms, "
          f"best={empty_times[0]:.3f} ms, p99={empty_times[99]:.3f} ms")

    # Measure CuPy overhead: just calling cp.sum on a tiny array
    tiny = cp.array([1.0], dtype=cp.float32)
    tiny_r = bench_latency(lambda: float(cp.sum(tiny)), runs=100)
    print(f"  CuPy sum(1 elem): p50={tiny_r['p50']:.3f} ms, "
          f"best={tiny_r['p01']:.3f} ms, p99={tiny_r['p99']:.3f} ms")

    # Python function call overhead (no GPU)
    py_times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        py_times.append((t1 - t0) * 1000)
    py_times.sort()
    print(f"  Python timer:    p50={py_times[500]:.4f} ms (measurement floor)")

    # ── Test 5: Sequential Query Throughput ──────────────────────
    print("\n--- Test 5: Sequential Query Throughput (10M, resident) ---\n")

    data = cp.random.uniform(0, 100, 10_000_000).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()

    # How many sum queries per second on resident data?
    n_queries = 1000
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_queries):
        float(cp.sum(data))
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    qps = n_queries / (t1 - t0)
    print(f"  Sum queries:    {qps:.0f} queries/sec on 10M resident float32")

    # Rolling mean queries
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        query_rolling_mean(data)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    qps_roll = 100 / (t1 - t0)
    print(f"  Rolling mean:   {qps_roll:.0f} queries/sec on 10M resident float32")

    del data
    cp.get_default_memory_pool().free_all_blocks()

    print(f"\n{'=' * 70}")
    print("E06 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
