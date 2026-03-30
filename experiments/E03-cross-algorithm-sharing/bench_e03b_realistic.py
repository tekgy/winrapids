"""
E03b -- Cross-Algorithm Sharing Retest at FinTek-Realistic Sizes

Navigator's hypothesis: E03 at 10M rows was GPU-compute-dominated, hiding
Python dispatch overhead. At FinTek-realistic sizes (50K-900K), the GPU
finishes in ~0.07ms (L2-bound, near kernel launch floor). If each CuPy
call has ~70us Python overhead, 10 CuPy calls = 0.7ms overhead on 0.07ms
compute. A monolithic kernel reduces those 10 calls to 1.

Two benchmark variants:
  Variant 1 -- Tight loop (original E03 methodology)
  Variant 2 -- Realistic pipeline overhead (CPU-side work between GPU ops)

Also measures:
  - GPU→CPU sync overhead in isolation
  - Python dispatch overhead (no-op function calls)
  - Monolithic JIT kernel (compile-once, run-forever path)
"""

import time
import numpy as np
import cupy as cp


# ============================================================
# Rolling statistics (from E03, unchanged)
# ============================================================

def rolling_mean_naive(data, window):
    cs = cp.cumsum(data)
    cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
    return (cs[window:] - cs[:-window]) / window


def rolling_std_naive(data, window):
    cs = cp.cumsum(data)
    cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
    cs2 = cp.cumsum(data * data)
    cs2 = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs2])
    mean = (cs[window:] - cs[:-window]) / window
    mean_sq = (cs2[window:] - cs2[:-window]) / window
    var = cp.maximum(mean_sq - mean * mean, 0)
    return cp.sqrt(var)


def rolling_mean_std_shared(data, window):
    cs = cp.cumsum(data)
    cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
    cs2 = cp.cumsum(data * data)
    cs2 = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs2])
    mean = (cs[window:] - cs[:-window]) / window
    mean_sq = (cs2[window:] - cs2[:-window]) / window
    var = cp.maximum(mean_sq - mean * mean, 0)
    std = cp.sqrt(var)
    return mean, std


def z_score_from_shared(data_trimmed, rm, rs):
    return (data_trimmed - rm) / cp.maximum(rs, 1e-10)


# ============================================================
# Simulated pipeline overhead functions
# ============================================================

_validation_log = []
_metric_log = []


def _validate(arr):
    """Lightweight validation — real work but no GPU."""
    if arr is None:
        raise ValueError("null result")
    _validation_log.append(len(arr))


def _log_metric(name, arr):
    """Lightweight metric logging — real work but no GPU."""
    _metric_log.append((name, time.perf_counter()))


# ============================================================
# Path A: Independent computation (each algo from scratch)
# ============================================================

def path_a_tight(data, window):
    """Tight loop: no overhead between CuPy ops."""
    rm = rolling_mean_naive(data, window)
    rs = rolling_std_naive(data, window)
    trimmed = data[window - 1:]
    z = (trimmed - rm) / cp.maximum(rs, 1e-10)
    return z


def path_a_overhead(data, window):
    """Realistic pipeline: CPU work between each GPU op."""
    rm = rolling_mean_naive(data, window)
    _validate(rm)
    _log_metric("rolling_mean", rm)

    rs = rolling_std_naive(data, window)
    _validate(rs)
    _log_metric("rolling_std", rs)

    trimmed = data[window - 1:]
    z = (trimmed - rm) / cp.maximum(rs, 1e-10)
    _validate(z)
    _log_metric("z_score", z)
    return z


# ============================================================
# Path B: Shared intermediates (shared cumsums)
# ============================================================

def path_b_tight(data, window):
    """Shared cumsums, tight loop."""
    rm, rs = rolling_mean_std_shared(data, window)
    trimmed = data[window - 1:]
    z = z_score_from_shared(trimmed, rm, rs)
    return z


def path_b_overhead(data, window):
    """Shared cumsums, realistic pipeline overhead."""
    rm, rs = rolling_mean_std_shared(data, window)
    _validate(rm)
    _validate(rs)
    _log_metric("rolling_mean_std_shared", rm)

    trimmed = data[window - 1:]
    z = z_score_from_shared(trimmed, rm, rs)
    _validate(z)
    _log_metric("z_score_shared", z)
    return z


# ============================================================
# Path C: Monolithic fused CUDA kernel
# ============================================================

FUSED_ROLLING_ZSCORE = cp.RawKernel(r"""
extern "C" __global__
void fused_rolling_zscore(const float* data, float* z_out,
                          const double* cumsum, const double* cumsum_sq,
                          int n, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data_idx = idx + window - 1;

    if (data_idx < n) {
        double sum_w = cumsum[data_idx + 1] - cumsum[data_idx + 1 - window];
        double sq_w = cumsum_sq[data_idx + 1] - cumsum_sq[data_idx + 1 - window];
        double mean = sum_w / (double)window;
        double mean_sq = sq_w / (double)window;
        double var = mean_sq - mean * mean;
        if (var < 0.0) var = 0.0;
        double std_val = sqrt(var);
        double val = (double)data[data_idx];
        double z = (std_val > 1e-10) ? (val - mean) / std_val : 0.0;
        z_out[idx] = (float)z;
    }
}
""", "fused_rolling_zscore")


def path_c_tight(data, window):
    """Monolithic fused kernel, tight loop."""
    cs = cp.cumsum(data.astype(cp.float64))
    cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), cs])
    cs2 = cp.cumsum((data.astype(cp.float64)) ** 2)
    cs2 = cp.concatenate([cp.zeros(1, dtype=cp.float64), cs2])

    n_out = len(data) - window + 1
    z = cp.empty(n_out, dtype=cp.float32)
    threads = 256
    blocks = (n_out + threads - 1) // threads
    FUSED_ROLLING_ZSCORE((blocks,), (threads,),
                         (data, z, cs, cs2, len(data), window))
    return z


def path_c_overhead(data, window):
    """Monolithic fused kernel, realistic pipeline overhead."""
    cs = cp.cumsum(data.astype(cp.float64))
    cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), cs])
    cs2 = cp.cumsum((data.astype(cp.float64)) ** 2)
    cs2 = cp.concatenate([cp.zeros(1, dtype=cp.float64), cs2])

    n_out = len(data) - window + 1
    z = cp.empty(n_out, dtype=cp.float32)
    threads = 256
    blocks = (n_out + threads - 1) // threads
    FUSED_ROLLING_ZSCORE((blocks,), (threads,),
                         (data, z, cs, cs2, len(data), window))
    _validate(z)
    _log_metric("fused_zscore", z)
    return z


# ============================================================
# Benchmarking
# ============================================================

def bench(fn, warmup=5, runs=30, label=""):
    """Higher iteration count for small arrays (more variance)."""
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
    p99 = times[int(len(times) * 0.99)]
    best = times[0]
    return {"p50": p50, "p99": p99, "best": best, "label": label}


def main():
    print("=" * 70)
    print("E03b -- Cross-Algorithm Sharing Retest (FinTek-Realistic Sizes)")
    print("=" * 70)

    window = 60

    # ── Preliminary: measure overhead floors ──────────────────
    print("\n--- Preliminary: Overhead Floors ---\n")

    # 1. GPU sync overhead
    sync_times = []
    cp.cuda.Stream.null.synchronize()
    for _ in range(1000):
        t0 = time.perf_counter()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        sync_times.append((t1 - t0) * 1e6)  # microseconds
    sync_times.sort()
    print(f"  GPU sync (idle stream):  p50={sync_times[500]:.1f} us, "
          f"best={sync_times[0]:.1f} us, p99={sync_times[990]:.1f} us")

    # 2. Python no-op function call overhead
    noop_times = []
    dummy = cp.zeros(10, dtype=cp.float32)
    for _ in range(10000):
        t0 = time.perf_counter()
        _validate(dummy)
        _log_metric("dummy", dummy)
        t1 = time.perf_counter()
        noop_times.append((t1 - t0) * 1e6)
    noop_times.sort()
    print(f"  Python validate+log:     p50={noop_times[5000]:.2f} us, "
          f"best={noop_times[0]:.2f} us, p99={noop_times[9900]:.2f} us")
    _validation_log.clear()
    _metric_log.clear()

    # 3. Minimal CuPy kernel launch (empty-ish)
    tiny = cp.zeros(1, dtype=cp.float32)
    def cupy_noop():
        return cp.sum(tiny)
    cupy_noop()  # warmup
    cp.cuda.Stream.null.synchronize()
    cupy_times = []
    for _ in range(500):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        cupy_noop()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        cupy_times.append((t1 - t0) * 1e6)
    cupy_times.sort()
    print(f"  CuPy sum(1 elem):        p50={cupy_times[250]:.1f} us, "
          f"best={cupy_times[0]:.1f} us, p99={cupy_times[495]:.1f} us")

    # ── Main experiment: FinTek-realistic sizes ───────────────
    sizes = [50_000, 100_000, 500_000, 900_000]
    size_labels = ["50K", "100K", "500K", "900K"]

    # Also include 10M for reference (to match original E03)
    sizes.append(10_000_000)
    size_labels.append("10M")

    print(f"\n{'=' * 70}")
    print(f"Variant 1: Tight Loop (original E03 methodology)")
    print(f"{'=' * 70}")

    print(f"\n  {'Size':>6}  {'A:Independent':>14}  {'B:Shared':>10}  {'C:Fused':>10}  "
          f"{'B/A':>6}  {'C/A':>6}  {'C/B':>6}")

    tight_results = {}
    for size, slabel in zip(sizes, size_labels):
        data = cp.random.uniform(-1, 1, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()

        a = bench(lambda: path_a_tight(data, window), label="A tight")
        b = bench(lambda: path_b_tight(data, window), label="B tight")
        c = bench(lambda: path_c_tight(data, window), label="C tight")

        print(f"  {slabel:>6}  {a['p50']:11.3f} ms  {b['p50']:7.3f} ms  {c['p50']:7.3f} ms  "
              f"{a['p50']/b['p50']:5.2f}x  {a['p50']/c['p50']:5.2f}x  {b['p50']/c['p50']:5.2f}x")

        tight_results[slabel] = {"A": a, "B": b, "C": c}
        del data
        cp.get_default_memory_pool().free_all_blocks()

    print(f"\n{'=' * 70}")
    print(f"Variant 2: Realistic Pipeline Overhead")
    print(f"{'=' * 70}")

    print(f"\n  {'Size':>6}  {'A:Independent':>14}  {'B:Shared':>10}  {'C:Fused':>10}  "
          f"{'B/A':>6}  {'C/A':>6}  {'C/B':>6}")

    overhead_results = {}
    for size, slabel in zip(sizes, size_labels):
        data = cp.random.uniform(-1, 1, size).astype(cp.float32)
        cp.cuda.Stream.null.synchronize()
        _validation_log.clear()
        _metric_log.clear()

        a = bench(lambda: path_a_overhead(data, window), label="A overhead")
        b = bench(lambda: path_b_overhead(data, window), label="B overhead")
        c = bench(lambda: path_c_overhead(data, window), label="C overhead")

        print(f"  {slabel:>6}  {a['p50']:11.3f} ms  {b['p50']:7.3f} ms  {c['p50']:7.3f} ms  "
              f"{a['p50']/b['p50']:5.2f}x  {a['p50']/c['p50']:5.2f}x  {b['p50']/c['p50']:5.2f}x")

        overhead_results[slabel] = {"A": a, "B": b, "C": c}
        del data
        cp.get_default_memory_pool().free_all_blocks()

    # ── Variant comparison: overhead impact ────────────────────
    print(f"\n{'=' * 70}")
    print(f"Overhead Impact: Tight vs Pipeline (p50 ms)")
    print(f"{'=' * 70}")

    print(f"\n  {'Size':>6}  {'A-tight':>8}  {'A-pipe':>8}  {'overhead':>9}  "
          f"{'C-tight':>8}  {'C-pipe':>8}  {'overhead':>9}")
    for slabel in size_labels:
        at = tight_results[slabel]["A"]["p50"]
        ap = overhead_results[slabel]["A"]["p50"]
        ct = tight_results[slabel]["C"]["p50"]
        cp_ = overhead_results[slabel]["C"]["p50"]
        print(f"  {slabel:>6}  {at:8.3f}  {ap:8.3f}  {ap-at:+8.3f}  "
              f"{ct:8.3f}  {cp_:8.3f}  {cp_-ct:+8.3f}")

    # ── Per-CuPy-call overhead measurement ─────────────────────
    print(f"\n{'=' * 70}")
    print(f"Per-CuPy-Call Overhead (dispatch + sync)")
    print(f"{'=' * 70}")

    print(f"\n  Measuring individual CuPy op latency at each size...\n")
    print(f"  {'Size':>6}  {'cumsum':>10}  {'concat':>10}  {'slice-div':>10}  "
          f"{'multiply':>10}  {'sqrt':>10}")

    for size, slabel in zip(sizes, size_labels):
        data = cp.random.uniform(-1, 1, size).astype(cp.float32)
        cs_pre = cp.cumsum(data)
        cs_pre = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs_pre])
        cp.cuda.Stream.null.synchronize()

        t_cumsum = bench(lambda: cp.cumsum(data), runs=50)
        t_concat = bench(lambda: cp.concatenate([cp.zeros(1, dtype=data.dtype), cs_pre]), runs=50)
        t_slice = bench(lambda: (cs_pre[window:] - cs_pre[:-window]) / window, runs=50)
        t_mul = bench(lambda: data * data, runs=50)
        t_sqrt = bench(lambda: cp.sqrt(data), runs=50)

        print(f"  {slabel:>6}  {t_cumsum['p50']*1000:8.1f} us  {t_concat['p50']*1000:8.1f} us  "
              f"{t_slice['p50']*1000:8.1f} us  {t_mul['p50']*1000:8.1f} us  "
              f"{t_sqrt['p50']*1000:8.1f} us")

        del data, cs_pre
        cp.get_default_memory_pool().free_all_blocks()

    # ── CuPy call count analysis ───────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"CuPy Call Count Analysis")
    print(f"{'=' * 70}")

    print(f"""
  Path A (independent): rolling_mean + rolling_std + z_score
    rolling_mean:  cumsum, concat, slice-sub-div          = 3 CuPy calls
    rolling_std:   cumsum, concat, cumsum, concat,
                   slice, slice, sub, mul, max, sqrt      = 10 CuPy calls
    z_score:       slice, sub, max, div                   = 4 CuPy calls
    TOTAL:         ~17 CuPy calls

  Path B (shared):   rolling_mean_std_shared + z_score
    shared:        cumsum, concat, cumsum, concat,
                   slice, slice, sub, mul, max, sqrt      = 10 CuPy calls
    z_score:       slice, sub, max, div                   = 4 CuPy calls
    TOTAL:         ~14 CuPy calls

  Path C (fused):    cumsums + 1 RawKernel launch
    cumsums:       cumsum, concat, cumsum, concat, cast,
                   cast, sq                               = 7 CuPy calls
    fused kernel:  1 RawKernel launch
    TOTAL:         ~8 CuPy calls

  Call reduction: A→C = {17-8} fewer calls = {(17-8)/17*100:.0f}% fewer dispatches
""")

    # ── Correctness check ──────────────────────────────────────
    print("--- Correctness Check ---\n")
    data = cp.random.uniform(-1, 1, 100_000).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()

    za = path_a_tight(data, window)
    zb = path_b_tight(data, window)
    zc = path_c_tight(data, window)

    err_ab = float(cp.max(cp.abs(za - zb)))
    err_ac = float(cp.max(cp.abs(za - zc)))
    print(f"  A vs B max error: {err_ab:.2e}")
    print(f"  A vs C max error: {err_ac:.2e}")
    print(f"  Status: {'PASS' if err_ab < 1e-5 and err_ac < 1e-3 else 'FAIL'}")

    del data
    cp.get_default_memory_pool().free_all_blocks()

    print(f"\n{'=' * 70}")
    print("E03b COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
