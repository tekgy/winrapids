"""Observer benchmark: plan-to-dispatch wall time.

Measures the end-to-end execute() path:
1. plan() compilation
2. Provenance probing (world state lookups)
3. Mock kernel dispatch per MISS step
4. Store registration per MISS step

Scenarios:
A) Cold run (NullWorld, all misses) — baseline
B) Warm run (GpuStore, all hits) — the 865x path
C) Plan-to-first-dispatch latency

Standing methodology: 3 warmup, 20 timed, p50/p99/mean.
"""

import time
import statistics

def bench_cold_execute():
    """Full execute path, all misses (NullWorld)."""
    print("\n--- Cold Execute (NullWorld, All Misses) ---\n")

    from _winrapids_core import Pipeline

    pipe = Pipeline()
    pipe.add("rolling_zscore", data="price", window=20)
    pipe.add("rolling_std", data="price", window=20)

    data = {"price": (0x100, 8000)}

    # Warmup
    for _ in range(3):
        pipe.execute(data, use_store=False)

    # Timed
    times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        result = pipe.execute(data, use_store=False)
        times.append((time.perf_counter_ns() - t0) / 1000.0)

    times.sort()
    p50 = times[10]
    p99 = times[19]
    mean = statistics.mean(times)

    stats = result["stats"]
    print(f"  E04 pipeline (2 specialists, 4 CSE nodes):")
    print(f"  execute() p50={p50:.1f} us  p99={p99:.1f} us  mean={mean:.1f} us")
    print(f"  Stats: hits={stats['hits']} misses={stats['misses']} hit_rate={stats['hit_rate']:.2f}")
    print(f"  Breakdown: plan ~11us + {stats['misses']} mock dispatches + store overhead")

    return p50


def bench_warm_execute():
    """Execute with GpuStore — second run should be all hits."""
    print("\n--- Warm Execute (GpuStore, Second Run All Hits) ---\n")

    from _winrapids_core import Pipeline

    pipe = Pipeline()
    pipe.add("rolling_zscore", data="price", window=20)
    pipe.add("rolling_std", data="price", window=20)

    data = {"price": (0x100, 8000)}

    # First run: populate store (all misses)
    result1 = pipe.execute(data, use_store=True)
    stats1 = result1["stats"]
    print(f"  First run:  hits={stats1['hits']} misses={stats1['misses']} hit_rate={stats1['hit_rate']:.2f}")

    # Second run: all hits
    # Note: GpuStore is created per execute() call in the current PyO3 binding,
    # so we need to check if the store persists across calls.
    result2 = pipe.execute(data, use_store=True)
    stats2 = result2["stats"]
    print(f"  Second run: hits={stats2['hits']} misses={stats2['misses']} hit_rate={stats2['hit_rate']:.2f}")

    if stats2["hit_rate"] == 0.0:
        print("  NOTE: GpuStore is per-execute() in current binding — store doesn't persist.")
        print("  This means use_store=True creates a fresh store each call.")
        print("  The warm-path test requires a persistent store (future API).")
        return None

    # If store persists, benchmark the warm path
    times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        result = pipe.execute(data, use_store=True)
        times.append((time.perf_counter_ns() - t0) / 1000.0)

    times.sort()
    p50 = times[10]
    stats = result["stats"]
    print(f"  Warm execute() p50={p50:.1f} us  hit_rate={stats['hit_rate']:.2f}")
    return p50


def bench_scaling():
    """Execute path scaling with pipeline size."""
    print("\n--- Execute Scaling (Cold, NullWorld) ---\n")

    from _winrapids_core import Pipeline

    specialists = ["rolling_mean", "rolling_std", "rolling_zscore"]
    data = {"price": (0x100, 8000)}

    for n_calls in [1, 2, 5, 10, 30, 50]:
        pipe = Pipeline()
        for i in range(n_calls):
            pipe.add(specialists[i % 3], data="price", window=20)

        # Warmup
        for _ in range(3):
            pipe.execute(data, use_store=False)

        # Timed
        times = []
        for _ in range(20):
            t0 = time.perf_counter_ns()
            result = pipe.execute(data, use_store=False)
            times.append((time.perf_counter_ns() - t0) / 1000.0)

        times.sort()
        p50 = times[10]
        stats = result["stats"]
        plan = result["plan"]

        print(f"  n_calls={n_calls:>3}: execute_p50={p50:7.1f} us  "
              f"steps={stats['hits']+stats['misses']}  "
              f"misses={stats['misses']}  "
              f"CSE={plan.eliminated}/{plan.original_nodes}")


def bench_compile_vs_execute():
    """Compare compile-only vs full execute."""
    print("\n--- Compile-Only vs Full Execute ---\n")

    from _winrapids_core import Pipeline

    pipe = Pipeline()
    pipe.add("rolling_zscore", data="price", window=20)
    pipe.add("rolling_std", data="price", window=20)

    data = {"price": (0x100, 8000)}

    # compile() only
    for _ in range(3):
        pipe.compile()
    compile_times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        pipe.compile()
        compile_times.append((time.perf_counter_ns() - t0) / 1000.0)
    compile_times.sort()

    # execute()
    for _ in range(3):
        pipe.execute(data, use_store=False)
    execute_times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        pipe.execute(data, use_store=False)
        execute_times.append((time.perf_counter_ns() - t0) / 1000.0)
    execute_times.sort()

    c_p50 = compile_times[10]
    e_p50 = execute_times[10]
    diff = e_p50 - c_p50

    print(f"  compile() only: p50={c_p50:.1f} us")
    print(f"  execute() full: p50={e_p50:.1f} us")
    print(f"  Execute overhead: {diff:.1f} us (provenance probes + mock dispatch + store)")
    print(f"  Per-step overhead: {diff / 4:.1f} us (4 CSE nodes)")


if __name__ == "__main__":
    print("=" * 70)
    print("Observer Benchmark: Execute Path + Plan-to-Dispatch Latency")
    print("=" * 70)

    bench_cold_execute()
    bench_warm_execute()
    bench_scaling()
    bench_compile_vs_execute()

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)
