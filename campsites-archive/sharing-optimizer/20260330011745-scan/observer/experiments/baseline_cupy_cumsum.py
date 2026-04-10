"""
Observer baseline: CuPy cumsum performance and correctness reference.

Phase 3 Task #2 (winrapids-scan GPU launch) must match these numbers.
Establishes the target for Rust scan implementation.

Standing methodology: 3 warmup, 20 timed runs, p50/p99/mean.
"""
import cupy as cp
import numpy as np
import time

def bench(fn, warmup=3, runs=20, label=""):
    """Benchmark with warm cache. Returns dict of p50/p99/mean in microseconds."""
    for _ in range(warmup):
        fn()
        cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter_ns()
        result = fn()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)  # nanoseconds -> microseconds

    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    mean = sum(times) / len(times)

    if label:
        print(f"  {label:30s}  p50={p50:8.1f} us  p99={p99:8.1f} us  mean={mean:8.1f} us")

    return {"p50": p50, "p99": p99, "mean": mean, "result": result}


def correctness_reference(n):
    """Generate a deterministic test case and its exact cumsum for cross-language verification."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(n).astype(np.float32)

    # CPU reference (float64 accumulation for ground truth)
    cpu_cumsum_f64 = np.cumsum(data.astype(np.float64))
    cpu_cumsum_f32 = cpu_cumsum_f64.astype(np.float32)

    # GPU cumsum
    d_data = cp.asarray(data)
    d_cumsum = cp.cumsum(d_data)
    gpu_cumsum = cp.asnumpy(d_cumsum)

    # Compare GPU vs CPU-f64 reference
    max_abs_err = float(np.max(np.abs(gpu_cumsum - cpu_cumsum_f32)))
    max_rel_err = float(np.max(np.abs(gpu_cumsum - cpu_cumsum_f32) / (np.abs(cpu_cumsum_f32) + 1e-30)))

    return {
        "n": n,
        "max_abs_error": max_abs_err,
        "max_rel_error": max_rel_err,
        "first_5": gpu_cumsum[:5].tolist(),
        "last_5": gpu_cumsum[-5:].tolist(),
        "checksum": float(gpu_cumsum[-1]),  # last element = total sum
    }


def main():
    print("=" * 70)
    print("Observer Baseline: CuPy cumsum")
    print("=" * 70)

    # -- Correctness --
    print("\n-- Correctness (GPU float32 vs CPU float64 reference) --\n")
    for n in [1_000, 50_000, 100_000, 500_000, 1_000_000, 10_000_000]:
        ref = correctness_reference(n)
        status = "PASS" if ref["max_rel_error"] < 1e-4 else "FAIL"
        print(f"  n={ref['n']:>10,}  max_abs={ref['max_abs_error']:.2e}  "
              f"max_rel={ref['max_rel_error']:.2e}  sum={ref['checksum']:+.4f}  {status}")

    # -- Performance --
    print("\n-- Performance (FinTek-realistic + large sizes) --\n")
    sizes = [50_000, 100_000, 500_000, 900_000, 1_000_000, 5_000_000, 10_000_000]

    for n in sizes:
        d_data = cp.random.randn(n, dtype=cp.float32)
        data_mb = n * 4 / 1e6

        result = bench(lambda: cp.cumsum(d_data), label=f"cumsum n={n:>10,} ({data_mb:.1f} MB)")

        # Bandwidth: read n + write n float32
        bw_gb = (n * 4 * 2) / (result["p50"] * 1e-6) / 1e9
        print(f"  {'':30s}  bandwidth={bw_gb:.0f} GB/s")
        print()

    # -- Dispatch overhead (minimal array) --
    print("-- Dispatch overhead (n=1, measures pure launch cost) --\n")
    d_tiny = cp.ones(1, dtype=cp.float32)
    bench(lambda: cp.cumsum(d_tiny), label="cumsum n=1")

    # -- Welford-style running mean+var (what WelfordOp must match) --
    print("\n-- Running mean+variance via CuPy (WelfordOp reference) --\n")
    for n in [50_000, 100_000, 1_000_000]:
        d_data = cp.random.randn(n, dtype=cp.float32)

        def welford_cupy():
            cs = cp.cumsum(d_data)
            cs2 = cp.cumsum(d_data ** 2)
            counts = cp.arange(1, n + 1, dtype=cp.float32)
            mean = cs / counts
            var = cs2 / counts - mean ** 2
            return mean, var

        bench(welford_cupy, label=f"welford n={n:>10,}")

    print("\n" + "=" * 70)
    print("Baseline complete. Rust scan must match correctness and beat performance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
