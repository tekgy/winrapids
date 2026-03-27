"""Fused bin statistics — one CUDA kernel, all stats, all bins.

This is the core K02 acceleration for Market Atlas. Instead of launching
separate CuPy kernels for mean, std, min, max, sum, count, first, last
(each with ~10μs launch overhead × N_bins), we launch ONE kernel that
processes the entire tick array and writes all statistics for all bins
in a single pass.

Architecture (from winrapids Experiment 012 "fused groupby"):
  "Fuse the computation, don't fuse the reduction."

But for bin stats, the reduction IS simple enough to fuse — we're computing
independent reductions per bin with known boundaries (not hash-based groupby).
Each thread block handles one or more bins. No atomics needed because bins
don't overlap.

Expected speedup vs CuPy wrapper: 10-100x for typical bin counts (1000-10000 bins).
The wrapper version was 0.33ms (SLOWER than CPU's 0.02ms due to kernel launch overhead).
This fused version should be <0.01ms for the same workload.
"""

from __future__ import annotations

import cupy as cp
import numpy as np


# The CUDA kernel: one pass over data, all stats for all bins
_FUSED_BIN_STATS_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void fused_bin_stats(
    const double* __restrict__ data,
    const long long* __restrict__ boundaries,
    double* __restrict__ out_sum,
    double* __restrict__ out_sum_sq,
    double* __restrict__ out_min,
    double* __restrict__ out_max,
    double* __restrict__ out_first,
    double* __restrict__ out_last,
    int* __restrict__ out_count,
    int n_bins
) {
    // Each thread handles one bin
    int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= n_bins) return;

    long long lo = boundaries[bin_idx];
    long long hi = boundaries[bin_idx + 1];
    int count = (int)(hi - lo);

    out_count[bin_idx] = count;

    if (count == 0) {
        // NaN for empty bins
        double nan_val = __longlong_as_double(0x7FF8000000000000LL);
        out_sum[bin_idx] = nan_val;
        out_sum_sq[bin_idx] = nan_val;
        out_min[bin_idx] = nan_val;
        out_max[bin_idx] = nan_val;
        out_first[bin_idx] = nan_val;
        out_last[bin_idx] = nan_val;
        return;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    double mn = data[lo];
    double mx = data[lo];
    double first = data[lo];
    double last = data[hi - 1];

    for (long long i = lo; i < hi; i++) {
        double val = data[i];
        sum += val;
        sum_sq += val * val;
        if (val < mn) mn = val;
        if (val > mx) mx = val;
    }

    out_sum[bin_idx] = sum;
    out_sum_sq[bin_idx] = sum_sq;
    out_min[bin_idx] = mn;
    out_max[bin_idx] = mx;
    out_first[bin_idx] = first;
    out_last[bin_idx] = last;
}
''', 'fused_bin_stats')


def fused_bin_stats(
    data: cp.ndarray,
    boundaries: cp.ndarray,
) -> dict[str, cp.ndarray]:
    """Compute all basic bin statistics in ONE kernel launch.

    Args:
        data: 1D float64 GPU array of tick values.
        boundaries: 1D int64 GPU array of bin boundary indices.
                    Length = n_bins + 1. boundaries[i]:boundaries[i+1] = bin i.

    Returns:
        Dict with GPU arrays, each of length n_bins:
            count, sum, mean, variance, std, min, max, first, last
    """
    n_bins = len(boundaries) - 1

    # Allocate output arrays on GPU
    out_sum = cp.empty(n_bins, dtype=cp.float64)
    out_sum_sq = cp.empty(n_bins, dtype=cp.float64)
    out_min = cp.empty(n_bins, dtype=cp.float64)
    out_max = cp.empty(n_bins, dtype=cp.float64)
    out_first = cp.empty(n_bins, dtype=cp.float64)
    out_last = cp.empty(n_bins, dtype=cp.float64)
    out_count = cp.empty(n_bins, dtype=cp.int32)

    # Launch: one thread per bin, 256 threads per block
    threads_per_block = 256
    blocks = (n_bins + threads_per_block - 1) // threads_per_block

    _FUSED_BIN_STATS_KERNEL(
        (blocks,), (threads_per_block,),
        (data, boundaries,
         out_sum, out_sum_sq, out_min, out_max, out_first, out_last, out_count,
         n_bins)
    )

    # Derive mean, variance, std from sum and sum_sq
    count_f = out_count.astype(cp.float64)
    mean = cp.where(count_f > 0, out_sum / count_f, cp.nan)
    variance = cp.where(count_f > 1, out_sum_sq / count_f - mean ** 2, cp.nan)
    variance = cp.maximum(variance, 0.0)  # clamp float noise

    return {
        "count": out_count,
        "sum": out_sum,
        "mean": mean,
        "variance": variance,
        "std": cp.sqrt(variance),
        "min": out_min,
        "max": out_max,
        "first": out_first,
        "last": out_last,
    }


def fused_multi_column_bin_stats(
    columns: dict[str, cp.ndarray],
    boundaries: cp.ndarray,
) -> dict[str, dict[str, cp.ndarray]]:
    """Compute fused bin stats for multiple columns.

    Args:
        columns: {name: gpu_array} for each column.
        boundaries: Shared bin boundaries.

    Returns:
        {col_name: {stat_name: gpu_array}} nested dict.
    """
    return {name: fused_bin_stats(arr, boundaries) for name, arr in columns.items()}
