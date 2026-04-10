"""Observer benchmark: PyO3 boundary crossing cost.

Measures:
1. Pipeline.add() call overhead (Python -> Rust struct push)
2. Pipeline.compile() call overhead (Python -> Rust compiler -> Python)
3. Breakdown: how much of compile() is boundary vs computation
4. Round-trip vs pure Rust compile time (from Entry 010: 11us)

Standing methodology: 3 warmup, 20 timed, p50/p99/mean.
"""

import time
import statistics

def bench_add_call():
    """Measure Pipeline.add() boundary crossing cost."""
    print("\n--- Pipeline.add() Boundary Cost ---\n")

    from _winrapids_core import Pipeline

    # Warmup
    for _ in range(3):
        p = Pipeline()
        for _ in range(100):
            p.add("rolling_zscore", data="price", window=20)

    # Timed: single add
    times = []
    for _ in range(20):
        p = Pipeline()
        t0 = time.perf_counter_ns()
        p.add("rolling_zscore", data="price", window=20)
        times.append((time.perf_counter_ns() - t0) / 1000.0)

    times.sort()
    p50 = times[10]
    p99 = times[19]
    mean = statistics.mean(times)
    print(f"  Single add():  p50={p50:.1f} us  p99={p99:.1f} us  mean={mean:.1f} us")

    # Timed: 100 adds
    times = []
    for _ in range(20):
        p = Pipeline()
        t0 = time.perf_counter_ns()
        for _ in range(100):
            p.add("rolling_zscore", data="price", window=20)
        elapsed = (time.perf_counter_ns() - t0) / 1000.0
        times.append(elapsed / 100.0)  # per-call

    times.sort()
    p50 = times[10]
    p99 = times[19]
    mean = statistics.mean(times)
    print(f"  Per-add (100x): p50={p50:.1f} us  p99={p99:.1f} us  mean={mean:.1f} us")


def bench_compile():
    """Measure Pipeline.compile() round-trip cost."""
    print("\n--- Pipeline.compile() Round-Trip Cost ---\n")
    print("  Pure Rust plan() baseline (Entry 010): 11.0 us p50")
    print("  Overhead above that = PyO3 boundary crossing\n")

    from _winrapids_core import Pipeline

    # E04 pipeline: 2 specialists
    def make_pipe():
        p = Pipeline()
        p.add("rolling_zscore", data="price", window=20)
        p.add("rolling_std", data="price", window=20)
        return p

    # Warmup
    for _ in range(3):
        make_pipe().compile()

    # Timed
    times = []
    for _ in range(20):
        p = make_pipe()
        t0 = time.perf_counter_ns()
        plan = p.compile()
        times.append((time.perf_counter_ns() - t0) / 1000.0)

    times.sort()
    p50 = times[10]
    p99 = times[19]
    mean = statistics.mean(times)
    print(f"  E04 compile():  p50={p50:.1f} us  p99={p99:.1f} us  mean={mean:.1f} us")
    print(f"  Pure Rust:      p50=11.0 us")
    print(f"  PyO3 overhead:  ~{p50 - 11.0:.1f} us ({(p50 - 11.0) / 11.0:.1f}x boundary tax)")

    # Verify correctness
    plan = make_pipe().compile()
    stats = plan.cse_stats
    print(f"\n  CSE verification: {stats}")
    assert stats["eliminated"] == 2, f"Expected 2 eliminated, got {stats['eliminated']}"
    assert stats["original_nodes"] == 6
    print("  CSE: PASS")

    return p50


def bench_compile_scaling():
    """Measure compile() scaling with pipeline size."""
    print("\n--- compile() Scaling ---\n")

    from _winrapids_core import Pipeline

    specialists = ["rolling_mean", "rolling_std", "rolling_zscore"]

    for n_calls in [1, 2, 5, 10, 30, 50]:
        def make_pipe():
            p = Pipeline()
            for i in range(n_calls):
                p.add(specialists[i % 3], data="price", window=20)
            return p

        # Warmup
        for _ in range(3):
            make_pipe().compile()

        # Timed
        times = []
        for _ in range(20):
            p = make_pipe()
            t0 = time.perf_counter_ns()
            plan = p.compile()
            times.append((time.perf_counter_ns() - t0) / 1000.0)

        times.sort()
        p50 = times[10]
        elim = plan.eliminated
        orig = plan.original_nodes
        pct = 100 * elim / orig if orig > 0 else 0
        print(f"  n_calls={n_calls:>3}: compile_p50={p50:7.1f} us  CSE={elim}/{orig} ({pct:.0f}%)")


def bench_list_specialists():
    """Measure utility function boundary cost."""
    print("\n--- Utility Functions ---\n")

    from _winrapids_core import list_specialists, specialist_dag

    # list_specialists
    times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        names = list_specialists()
        times.append((time.perf_counter_ns() - t0) / 1000.0)

    times.sort()
    print(f"  list_specialists(): p50={times[10]:.1f} us  result={names}")

    # specialist_dag
    times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        dag = specialist_dag("rolling_zscore")
        times.append((time.perf_counter_ns() - t0) / 1000.0)

    times.sort()
    print(f"  specialist_dag('rolling_zscore'): p50={times[10]:.1f} us  steps={len(dag)}")


if __name__ == "__main__":
    print("=" * 70)
    print("Observer Benchmark: PyO3 Boundary Crossing Cost")
    print("=" * 70)

    bench_add_call()
    bench_compile()
    bench_compile_scaling()
    bench_list_specialists()

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)
