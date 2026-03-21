"""
WinRapids Experiment 010: CuPy comparison for kernel fusion benchmarks.

Runs the same expressions as fused_ops.cu using CuPy (separate kernel launches)
to measure the actual abstraction cost of non-fused execution.
"""

import time
import numpy as np
import cupy as cp


def bench(name, fn, warmup=2, runs=20):
    """Benchmark a function with warmup and timing."""
    for _ in range(warmup):
        fn()
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    cp.cuda.Device(0).synchronize()
    avg_ms = (time.perf_counter() - t0) / runs * 1000
    return avg_ms


def main():
    print("WinRapids Experiment 010: CuPy Comparison (unfused)")
    print("=" * 60)

    n = 10_000_000
    print(f"\nData size: {n:,} elements ({n * 8 / 1e6:.1f} MB per column)\n")

    np.random.seed(42)
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal(n)
    b_np = rng.standard_normal(n)
    c_np = rng.standard_normal(n)

    a = cp.asarray(a_np)
    b = cp.asarray(b_np)
    c = cp.asarray(c_np)
    cp.cuda.Device(0).synchronize()

    # Test 1: a * b + c
    print("=== Test 1: a * b + c ===")
    ms = bench("a*b+c", lambda: a * b + c)
    bw = 4.0 * n * 8 / (ms * 1e6)
    print(f"  CuPy (2 kernels):  {ms:.4f} ms ({bw:.1f} GB/s)")
    print()

    # Test 2: a*b + c*c - a/b
    print("=== Test 2: a*b + c*c - a/b ===")
    ms = bench("complex", lambda: a * b + c * c - a / b)
    bw = 4.0 * n * 8 / (ms * 1e6)
    print(f"  CuPy (5 kernels):  {ms:.4f} ms ({bw:.1f} GB/s)")
    print()

    # Test 3: where(a > 0, b * c, -b * c)
    print("=== Test 3: where(a > 0, b * c, -b * c) ===")
    ms = bench("where", lambda: cp.where(a > 0, b * c, -b * c))
    bw = 4.0 * n * 8 / (ms * 1e6)
    print(f"  CuPy (5 kernels):  {ms:.4f} ms ({bw:.1f} GB/s)")
    print()

    # Test 4: sum(a * b + c)
    print("=== Test 4: sum(a * b + c) ===")
    ms = bench("sum_fma", lambda: float(cp.sum(a * b + c)))
    bw = 3.0 * n * 8 / (ms * 1e6)
    print(f"  CuPy (3 kernels):  {ms:.4f} ms ({bw:.1f} GB/s)")
    result = float(cp.sum(a * b + c))
    print(f"  Result:            {result:.6f}")
    print()

    # Test 5: sqrt(abs(a*b + c*c - a))
    print("=== Test 5: sqrt(abs(a*b + c*c - a)) ===")
    ms = bench("deep_chain", lambda: cp.sqrt(cp.abs(a * b + c * c - a)))
    bw = 4.0 * n * 8 / (ms * 1e6)
    print(f"  CuPy (6 kernels):  {ms:.4f} ms ({bw:.1f} GB/s)")
    print()

    print("=" * 60)
    print("CuPy comparison complete.")


if __name__ == "__main__":
    main()
