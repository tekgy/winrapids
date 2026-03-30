"""
Observer: WDDM scheduling jitter profile.

Phase 2 noted WDDM jitter in p99 latencies. This experiment characterizes:
1. How bad is the tail at different array sizes?
2. Is it correlated with array size or random?
3. What does the full latency distribution look like?
4. How does it compare across op types (scan, reduce, element-wise)?

This matters for the Rust compiler's cost model: if p99 is 10x p50,
the cost model needs to account for tail variance.
"""
import cupy as cp
import numpy as np
import time

def profile_latency(fn, runs=200, label=""):
    """Collect 200 latency samples for distribution analysis."""
    # Warmup
    for _ in range(5):
        fn()
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)  # us

    times_arr = np.array(times)
    p50 = np.percentile(times_arr, 50)
    p90 = np.percentile(times_arr, 90)
    p95 = np.percentile(times_arr, 95)
    p99 = np.percentile(times_arr, 99)
    p999 = np.percentile(times_arr, 99.9)
    max_t = np.max(times_arr)

    if label:
        print(f"  {label:40s}  p50={p50:7.1f}  p90={p90:7.1f}  p95={p95:7.1f}  "
              f"p99={p99:7.1f}  max={max_t:7.1f}  ratio={max_t/p50:.1f}x")

    return times_arr


def main():
    print("=" * 70)
    print("Observer: WDDM Jitter Profile")
    print("=" * 70)

    print("\n-- Cumsum (prefix scan) across sizes --\n")
    print(f"  {'Operation':40s}  {'p50':>7}  {'p90':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}  {'ratio':>7}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    for n in [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
        d = cp.random.randn(n, dtype=cp.float32)
        profile_latency(lambda: cp.cumsum(d), runs=200,
                       label=f"cumsum n={n:>10,}")

    print("\n-- Different op types at n=100K --\n")
    print(f"  {'Operation':40s}  {'p50':>7}  {'p90':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}  {'ratio':>7}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    n = 100_000
    d = cp.random.randn(n, dtype=cp.float32)
    d2 = cp.random.randn(n, dtype=cp.float32)

    profile_latency(lambda: cp.cumsum(d), label="scan: cumsum")
    profile_latency(lambda: cp.sum(d), label="reduce: sum")
    profile_latency(lambda: d + d2, label="element-wise: add")
    profile_latency(lambda: d * d2 + d, label="element-wise: fma")
    profile_latency(lambda: cp.sort(d.copy()), label="sort")
    profile_latency(lambda: cp.argsort(d), label="argsort")

    print("\n-- Rapid-fire (10 ops chained, simulates pipeline) --\n")
    print(f"  {'Operation':40s}  {'p50':>7}  {'p90':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}  {'ratio':>7}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    for n in [50_000, 100_000, 500_000]:
        d = cp.random.randn(n, dtype=cp.float32)

        def pipeline():
            cs = cp.cumsum(d)
            cs2 = cp.cumsum(d * d)
            cnt = cp.arange(1, n + 1, dtype=cp.float32)
            mean = cs / cnt
            var = cs2 / cnt - mean * mean
            std = cp.sqrt(cp.maximum(var, 0))
            z = (d - mean) / (std + 1e-8)
            return z

        profile_latency(pipeline, label=f"10-op pipeline n={n:>10,}")

    print(f"\n{'=' * 70}")
    print("Jitter profile complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
