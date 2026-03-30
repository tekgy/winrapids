"""Tensor Core exploration — 1000+ TFLOPS untapped.

Our RTX 6000 Pro Blackwell has:
  FP64: 1.95 TFLOPS (CUDA cores, 2 ALUs/SM)
  FP32: 125 TFLOPS (CUDA cores)
  FP16->FP32: ~1000 TFLOPS (Tensor Cores, via cuBLAS GEMM)
  FP8->FP32:  ~2000 TFLOPS (Tensor Cores, 5th gen)

We're currently at 125 TFLOPS (FP32 CUDA cores). Can we reach Tensor Core territory?

Three hypotheses to test:
  1. Column decode: integer->float as FP16 diagonal GEMM (Tensor Cores)
  2. Bin stats: bin membership as sparse matrix × data vector = GEMM
  3. Cross-ticker correlation: X^T @ X = textbook Tensor Core GEMM

Key questions:
  - Does CuPy's @ operator route through cuBLAS -> Tensor Cores?
  - What's the actual measured TFLOPS on each approach?
  - Is the data large enough to saturate Tensor Cores? (GEMM needs large M,N,K)
  - FP16 precision: do we lose anything meaningful for market data?
"""

from __future__ import annotations

import time
import numpy as np
import cupy as cp
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# DATA SETUP
# ══════════════════════════════════════════════════════════════════

def load_source_gpu():
    """Load real AAPL data to GPU."""
    import pyarrow.parquet as pq
    AAPL_PATH = Path("W:/fintek/data/fractal/K01/2025-09-02/AAPL/K01P01.TI00TO00.parquet")
    tbl = pq.read_table(str(AAPL_PATH))
    price = tbl.column("K01P01.DI01DO01").to_numpy().astype(np.float32)
    size = tbl.column("K01P01.DI01DO02").to_numpy().astype(np.float32)
    ts = tbl.column("K01P01.DI01DO03").to_numpy().astype(np.int64)
    return cp.asarray(price), cp.asarray(size), cp.asarray(ts), len(price)


def time_gpu(fn, n_warmup=20, n_iters=200):
    """Time a GPU function with warmup."""
    for _ in range(n_warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / n_iters


# ══════════════════════════════════════════════════════════════════
# HYPOTHESIS 1: Column decode as GEMM
# ══════════════════════════════════════════════════════════════════

def bench_column_decode(price_f32, size_f32, n):
    """Can we use GEMM for column scaling/transformation?

    Integer decode: int_col * scale_factor -> float_col
    This is a diagonal matrix multiply: data @ diag(scales)

    For Tensor Cores: need FP16 input, FP32 accumulator.
    cuBLAS GEMM automatically routes FP16 matmul through Tensor Cores.
    """
    print("HYPOTHESIS 1: Column Decode as GEMM (Tensor Cores)")
    print("-" * 60)

    # Simulate integer-encoded columns (3 cols)
    # In real pipeline: price_int * 1e-8, size * 1.0, notional_int * 1e-8
    int_prices = (price_f32 * 1e4).astype(cp.int32)  # Simulate int32 encoded prices
    int_sizes = size_f32.astype(cp.int32)

    # ── Approach A: Scalar multiply (current, CUDA cores) ──
    def scalar_decode():
        p = int_prices.astype(cp.float32) * 1e-4
        s = int_sizes.astype(cp.float32)
        return p, s

    t_scalar = time_gpu(scalar_decode)

    # ── Approach B: GEMM decode (N×3 @ 3×3 diagonal) ──
    # Stack columns as (N, 3) matrix, multiply by diagonal scale matrix
    int_matrix = cp.stack([
        int_prices.astype(cp.float32),
        int_sizes.astype(cp.float32),
        (int_prices.astype(cp.float32) * int_sizes.astype(cp.float32)),  # notional
    ], axis=1)  # (N, 3)

    scale_diag = cp.array([[1e-4, 0, 0],
                           [0, 1.0, 0],
                           [0, 0, 1e-4]], dtype=cp.float32)  # (3, 3)

    def gemm_decode_f32():
        return int_matrix @ scale_diag

    t_gemm_f32 = time_gpu(gemm_decode_f32)

    # ── Approach C: FP16 GEMM (Tensor Core path) ──
    int_matrix_f16 = int_matrix.astype(cp.float16)
    scale_diag_f16 = scale_diag.astype(cp.float16)

    def gemm_decode_f16():
        return int_matrix_f16 @ scale_diag_f16  # cuBLAS routes FP16 GEMM to Tensor Cores

    t_gemm_f16 = time_gpu(gemm_decode_f16)

    # ── Approach D: Direct elementwise (baseline, what we actually do) ──
    scales = cp.array([1e-4, 1.0, 1e-4], dtype=cp.float32)

    def elementwise_decode():
        return int_matrix * scales[None, :]

    t_elem = time_gpu(elementwise_decode)

    print(f"  Scalar multiply:     {t_scalar*1e6:>8.1f}us")
    print(f"  Elementwise (broad): {t_elem*1e6:>8.1f}us")
    print(f"  GEMM FP32:           {t_gemm_f32*1e6:>8.1f}us")
    print(f"  GEMM FP16 (TC):      {t_gemm_f16*1e6:>8.1f}us")
    print()

    # Check if FP16 GEMM actually uses Tensor Cores
    # A matrix with N=598K, K=3, M=3 is too thin for Tensor Cores to help
    # Tensor Cores need M,N,K multiples of 16 and large enough to saturate
    print(f"  Matrix shape: ({n}, 3) @ (3, 3) — K=3 is too small for TC saturation")
    print(f"  Tensor Cores need M,N,K >= 16 and large enough GEMM to amortize overhead")
    print()

    # ── Approach E: Wide GEMM (what if we have many more columns?) ──
    # Simulate K01P02 with 20 derived columns from 5 source columns
    # Source (N, 5) @ Transform (5, 20) -> Derived (N, 20)
    src_matrix = cp.random.randn(n, 5, dtype=cp.float32)
    transform = cp.random.randn(5, 20, dtype=cp.float32)

    src_f16 = src_matrix.astype(cp.float16)
    transform_f16 = transform.astype(cp.float16)

    def wide_gemm_f32():
        return src_matrix @ transform

    def wide_gemm_f16():
        return src_f16 @ transform_f16

    t_wide_f32 = time_gpu(wide_gemm_f32)
    t_wide_f16 = time_gpu(wide_gemm_f16)

    print(f"  Wide GEMM ({n}x5 @ 5x20):")
    print(f"    FP32:  {t_wide_f32*1e6:>8.1f}us")
    print(f"    FP16:  {t_wide_f16*1e6:>8.1f}us")
    print(f"    Ratio: {t_wide_f32/t_wide_f16:.1f}x")
    print()

    return {
        "scalar": t_scalar, "elementwise": t_elem,
        "gemm_f32": t_gemm_f32, "gemm_f16": t_gemm_f16,
        "wide_f32": t_wide_f32, "wide_f16": t_wide_f16,
    }


# ══════════════════════════════════════════════════════════════════
# HYPOTHESIS 2: Bin stats as matrix multiply
# ══════════════════════════════════════════════════════════════════

def bench_bin_stats_gemm(price_f32, n):
    """Can bin-sum be expressed as sparse_matrix @ data_vector?

    Bin membership matrix B (n_bins × n_ticks): B[i,j] = 1 if tick j is in bin i
    Bin sums: B @ data = sum of data per bin

    For mean: B @ data / bin_counts
    For sum_sq: B @ (data * data)

    This is SpMV (sparse matrix × dense vector). cuSPARSE does this.
    But also: if B is dense (which it is for contiguous bins), it's a GEMM.
    """
    print("HYPOTHESIS 2: Bin Stats as Matrix Multiply")
    print("-" * 60)

    # Create realistic bin boundaries (1437 bins for a trading day)
    n_bins = 1437
    boundaries = np.sort(np.random.choice(n, n_bins + 1, replace=False))
    boundaries[0] = 0
    boundaries[-1] = n
    boundaries_gpu = cp.asarray(boundaries.astype(np.int64))

    # ── Approach A: Our fused CUDA kernel (current) ──
    # Simplified: just compute bin sums
    def cuda_bin_sums():
        sums = cp.zeros(n_bins, dtype=cp.float32)
        for i in range(n_bins):
            lo, hi = int(boundaries[i]), int(boundaries[i + 1])
            if hi > lo:
                sums[i] = cp.sum(price_f32[lo:hi])
        return sums

    # This is too slow in Python loop — use reduceat
    def reduceat_bin_sums():
        # CuPy doesn't have reduceat, use cumsum + diff
        cs = cp.cumsum(price_f32)
        sums = cp.empty(n_bins, dtype=cp.float32)
        # Vectorized: sums[i] = cs[hi-1] - cs[lo-1]
        lo = boundaries_gpu[:-1]
        hi = boundaries_gpu[1:]
        sums = cp.where(hi > lo,
                        cp.where(lo > 0, cs[hi - 1] - cs[lo - 1], cs[hi - 1]),
                        0.0)
        return sums

    t_reduceat = time_gpu(reduceat_bin_sums)

    # ── Approach B: Dense bin membership matrix @ data ──
    # B is (n_bins, n_ticks) — this is HUGE (1437 × 598057 = 859M elements)
    # Even as float16, that's 1.7GB. Too large.
    print(f"  Dense B matrix: ({n_bins}, {n}) = {n_bins*n/1e9:.1f}B elements = "
          f"{n_bins*n*2/1e9:.1f}GB (FP16) — TOO LARGE")
    print()

    # ── Approach C: Segment reduction via cumsum (vectorized) ──
    t_cumsum = time_gpu(reduceat_bin_sums)

    # ── Approach D: cupy.add.reduceat equivalent ──
    # Actually, let's test if matmul of thin matrices works
    # Idea: reshape data into (n_bins, max_bin_size) and sum along axis=1
    # But bins have different sizes, so padding needed

    # ── Approach E: Multiple columns at once via matmul ──
    # If we have K columns and N bins, and all bins are same size:
    # Reshape data from (N_ticks, K) to (n_bins, bin_size, K)
    # Sum along axis=1 -> (n_bins, K)
    # This IS a valid use of GEMM: (n_bins, bin_size) × (bin_size, 1) for each col
    # Or better: stack all columns and do one batched GEMM

    # For uniform bins (equal-size), this works perfectly:
    uniform_bin_size = n // n_bins
    n_uniform = uniform_bin_size * n_bins
    data_uniform = price_f32[:n_uniform].reshape(n_bins, uniform_bin_size)

    def uniform_bin_sum():
        return cp.sum(data_uniform, axis=1)

    t_uniform = time_gpu(uniform_bin_sum)

    # Multi-column uniform bins via GEMM
    # (n_bins, bin_size) @ ones(bin_size, 1) = bin sums
    ones_vec = cp.ones((uniform_bin_size, 1), dtype=cp.float32)
    ones_f16 = cp.ones((uniform_bin_size, 1), dtype=cp.float16)
    data_uniform_f16 = data_uniform.astype(cp.float16)

    def gemm_bin_sum_f32():
        return data_uniform @ ones_vec  # (n_bins, 1)

    def gemm_bin_sum_f16():
        return data_uniform_f16 @ ones_f16  # Tensor Core path

    t_gemm_f32 = time_gpu(gemm_bin_sum_f32)
    t_gemm_f16 = time_gpu(gemm_bin_sum_f16)

    # Multi-column: 5 data columns at once
    multi_data = cp.random.randn(n_bins, uniform_bin_size, 5, dtype=cp.float32)
    multi_reshaped = multi_data.reshape(n_bins * 5, uniform_bin_size)  # batch the GEMM
    ones_wide = cp.ones((uniform_bin_size, 1), dtype=cp.float32)

    def gemm_multi_col_f32():
        return multi_reshaped @ ones_wide  # (n_bins*5, 1)

    t_multi_f32 = time_gpu(gemm_multi_col_f32)

    print(f"  Cumsum + diff (variable bins): {t_cumsum*1e6:>8.1f}us")
    print(f"  Uniform reshape + sum:         {t_uniform*1e6:>8.1f}us")
    print(f"  GEMM FP32 (uniform bins):      {t_gemm_f32*1e6:>8.1f}us")
    print(f"  GEMM FP16 (TC, uniform bins):  {t_gemm_f16*1e6:>8.1f}us")
    print(f"  GEMM FP32 (5 cols, uniform):   {t_multi_f32*1e6:>8.1f}us")
    print(f"  Bin size: {uniform_bin_size} ticks/bin, {n_bins} bins")
    print()

    return {
        "cumsum": t_cumsum, "uniform_sum": t_uniform,
        "gemm_f32": t_gemm_f32, "gemm_f16": t_gemm_f16,
        "multi_f32": t_multi_f32,
    }


# ══════════════════════════════════════════════════════════════════
# HYPOTHESIS 3: Cross-ticker correlation via GEMM
# ══════════════════════════════════════════════════════════════════

def bench_correlation_gemm():
    """X^T @ X for cross-ticker correlation matrix.

    This is THE textbook Tensor Core workload.
    X shape: (n_timepoints, n_tickers) — e.g., (1437 bins × 4604 tickers)
    X^T @ X -> (4604, 4604) correlation matrix
    """
    print("HYPOTHESIS 3: Cross-Ticker Correlation via GEMM")
    print("-" * 60)

    # Realistic dimensions
    n_bins = 1437    # time bins per day
    n_tickers = 4604  # full universe

    # Smaller test first (memory constraints)
    for n_t in [100, 500, 1000, 4604]:
        X_f32 = cp.random.randn(n_bins, n_t, dtype=cp.float32)
        X_f16 = X_f32.astype(cp.float16)

        def corr_f32():
            return X_f32.T @ X_f32  # (n_t, n_t)

        def corr_f16():
            return X_f16.T @ X_f16

        t_f32 = time_gpu(corr_f32, n_warmup=10, n_iters=50)
        t_f16 = time_gpu(corr_f16, n_warmup=10, n_iters=50)

        # FLOPS: 2 * n_bins * n_t * n_t (for X^T @ X)
        flops = 2.0 * n_bins * n_t * n_t
        tflops_f32 = flops / t_f32 / 1e12
        tflops_f16 = flops / t_f16 / 1e12

        ratio = t_f32 / t_f16
        print(f"  {n_t:>5d} tickers: FP32 {t_f32*1000:>7.2f}ms ({tflops_f32:>5.1f} TFLOPS)  "
              f"FP16 {t_f16*1000:>7.2f}ms ({tflops_f16:>5.1f} TFLOPS)  "
              f"ratio: {ratio:.1f}x")

    print()

    # The real deal: full universe correlation
    print("  Full universe correlation (1437 × 4604):")
    X_full_f32 = cp.random.randn(n_bins, n_tickers, dtype=cp.float32)
    X_full_f16 = X_full_f32.astype(cp.float16)

    t_full_f32 = time_gpu(lambda: X_full_f32.T @ X_full_f32, n_warmup=5, n_iters=20)
    t_full_f16 = time_gpu(lambda: X_full_f16.T @ X_full_f16, n_warmup=5, n_iters=20)

    flops_full = 2.0 * n_bins * n_tickers * n_tickers
    print(f"    FP32: {t_full_f32*1000:.2f}ms ({flops_full/t_full_f32/1e12:.1f} TFLOPS)")
    print(f"    FP16: {t_full_f16*1000:.2f}ms ({flops_full/t_full_f16/1e12:.1f} TFLOPS)")
    print(f"    Ratio: {t_full_f32/t_full_f16:.1f}x")
    print(f"    Output: ({n_tickers}, {n_tickers}) = {n_tickers**2/1e6:.1f}M elements")
    print()

    # FP16 precision check — with normalized data (like real returns/z-scores)
    X_norm_f32 = cp.random.randn(n_bins, n_tickers, dtype=cp.float32)
    X_norm_f32 = X_norm_f32 / cp.sqrt(cp.sum(X_norm_f32**2, axis=0, keepdims=True))  # unit norm
    X_norm_f16 = X_norm_f32.astype(cp.float16)

    result_f32 = (X_norm_f32.T @ X_norm_f32).get()
    result_f16 = (X_norm_f16.T @ X_norm_f16).get().astype(np.float32)
    rel_err = np.max(np.abs(result_f32 - result_f16) / (np.abs(result_f32) + 1e-10))
    print(f"  FP16 vs FP32 max relative error (normalized): {rel_err:.2e}")
    mean_err = np.mean(np.abs(result_f32 - result_f16) / (np.abs(result_f32) + 1e-10))
    print(f"  FP16 vs FP32 mean relative error (normalized): {mean_err:.2e}")

    return {
        "full_f32": t_full_f32, "full_f16": t_full_f16,
        "tflops_f32": flops_full / t_full_f32 / 1e12,
        "tflops_f16": flops_full / t_full_f16 / 1e12,
    }


# ══════════════════════════════════════════════════════════════════
# BONUS: What's our actual Tensor Core throughput?
# ══════════════════════════════════════════════════════════════════

def bench_peak_tflops():
    """Measure peak TFLOPS by running large GEMM."""
    print("PEAK TFLOPS MEASUREMENT (large square GEMM)")
    print("-" * 60)

    for M in [4096, 8192, 16384]:
        A_f32 = cp.random.randn(M, M, dtype=cp.float32)
        A_f16 = A_f32.astype(cp.float16)

        t_f32 = time_gpu(lambda: A_f32 @ A_f32, n_warmup=5, n_iters=10)
        t_f16 = time_gpu(lambda: A_f16 @ A_f16, n_warmup=5, n_iters=10)

        flops = 2.0 * M * M * M
        tflops_f32 = flops / t_f32 / 1e12
        tflops_f16 = flops / t_f16 / 1e12

        print(f"  {M}x{M}: FP32 {t_f32*1000:>7.1f}ms ({tflops_f32:>6.1f} TFLOPS)  "
              f"FP16 {t_f16*1000:>7.1f}ms ({tflops_f16:>6.1f} TFLOPS)  "
              f"ratio: {t_f32/t_f16:.1f}x")

        del A_f32, A_f16
        cp.get_default_memory_pool().free_all_blocks()

    print()


def main():
    print("=" * 78)
    print("TENSOR CORE EXPLORATION — RTX 6000 Pro Blackwell")
    print("=" * 78)
    print()

    # GPU info
    dev = cp.cuda.Device(0)
    try:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    except Exception:
        name = "Unknown"
    print(f"GPU: {name}")
    mem = dev.mem_info
    print(f"VRAM: {mem[1]/1e9:.1f}GB total, {mem[0]/1e9:.1f}GB free")
    print()

    # Peak measurement first
    bench_peak_tflops()

    # Load real data
    price_f32, size_f32, ts_i64, n = load_source_gpu()
    cp.cuda.Stream.null.synchronize()
    print(f"Data: {n:,} ticks loaded to GPU")
    print()

    # Run hypotheses
    r1 = bench_column_decode(price_f32, size_f32, n)
    print()
    r2 = bench_bin_stats_gemm(price_f32, n)
    print()
    r3 = bench_correlation_gemm()

    # ── Summary ────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY — Where Tensor Cores Help")
    print("=" * 78)
    print()
    print("1. Column decode (int->float scaling):")
    print(f"   GEMM is SLOWER than elementwise. Tensor Cores need large K dimension.")
    print(f"   Winner: elementwise multiply ({r1['elementwise']*1e6:.1f}us)")
    print()
    print("2. Bin stats (bin sums):")
    print(f"   Cumsum+diff wins for variable-size bins ({r2['cumsum']*1e6:.1f}us)")
    print(f"   GEMM only viable for uniform bins, and still slower")
    print()
    print("3. Cross-ticker correlation (X^T @ X):")
    print(f"   THIS IS THE TENSOR CORE WORKLOAD.")
    print(f"   FP32: {r3['full_f32']*1000:.1f}ms ({r3['tflops_f32']:.1f} TFLOPS)")
    print(f"   FP16: {r3['full_f16']*1000:.1f}ms ({r3['tflops_f16']:.1f} TFLOPS)")
    print(f"   {r3['tflops_f16']/r3['tflops_f32']:.0f}x more TFLOPS with FP16 Tensor Cores")
    print(f"   Full 4604-ticker correlation in {r3['full_f16']*1000:.1f}ms")
    print()
    print("VERDICT:")
    print("  - Column decode / bin stats: stay on CUDA cores (too narrow for TC)")
    print("  - Cross-ticker correlation (K04): USE TENSOR CORES (textbook GEMM)")
    print("  - Any future NxN or NxK GEMM with large dims: USE TENSOR CORES")


if __name__ == "__main__":
    main()
