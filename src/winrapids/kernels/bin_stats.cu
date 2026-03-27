/**
 * Fused bin statistics kernel — the CUDA C specification.
 *
 * This kernel computes sum, sum_sq, min, max, first, last, count
 * for all bins in a single launch. One thread per bin.
 *
 * This file is the SPECIFICATION extracted from the CuPy RawKernel
 * prototype. The winrapids C++ build system should compile this
 * into a shared library callable from Python via pybind11/ctypes.
 *
 * Performance notes (from benchmarking):
 * - CuPy RawKernel version: 0.163 ms for 598K ticks, 1437 bins
 * - CPU numpy prefix sums: 0.014 ms (still faster due to launch overhead)
 * - The GPU wins at SCALE: many columns × many cadences × batched
 *   (one launch for everything, not per-column per-cadence)
 *
 * Future optimizations:
 * - Multi-column version: process price + size + notional in one launch
 * - Multi-cadence version: all 31 cadences in one launch
 * - Shared memory for bins that fit in smem
 * - Warp-level reductions for large bins (>32 ticks)
 * - Higher-order moments (skewness, kurtosis) in the same pass
 */

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
    int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= n_bins) return;

    long long lo = boundaries[bin_idx];
    long long hi = boundaries[bin_idx + 1];
    int count = (int)(hi - lo);

    out_count[bin_idx] = count;

    if (count == 0) {
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
    out_first[bin_idx] = data[lo];
    out_last[bin_idx] = data[hi - 1];
}


/**
 * Multi-column version — process N columns in one launch.
 * Each thread handles one (bin, column) pair.
 * Grid: (n_bins * n_columns + 255) / 256 blocks.
 */
extern "C" __global__
void fused_bin_stats_multi(
    const double* __restrict__ data,  /* packed: col0[N], col1[N], ... */
    const long long* __restrict__ boundaries,
    double* __restrict__ out_sum,     /* packed: col0[n_bins], col1[n_bins], ... */
    double* __restrict__ out_sum_sq,
    double* __restrict__ out_min,
    double* __restrict__ out_max,
    double* __restrict__ out_first,
    double* __restrict__ out_last,
    int* __restrict__ out_count,
    int n_bins,
    int n_columns,
    long long n_ticks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_bins * n_columns;
    if (idx >= total) return;

    int bin_idx = idx / n_columns;
    int col_idx = idx % n_columns;

    long long lo = boundaries[bin_idx];
    long long hi = boundaries[bin_idx + 1];
    int count = (int)(hi - lo);

    /* Column data starts at data + col_idx * n_ticks */
    const double* col_data = data + (long long)col_idx * n_ticks;

    /* Output offset: col_idx * n_bins + bin_idx */
    int out_idx = col_idx * n_bins + bin_idx;

    if (col_idx == 0) {
        out_count[bin_idx] = count;
    }

    if (count == 0) {
        double nan_val = __longlong_as_double(0x7FF8000000000000LL);
        out_sum[out_idx] = nan_val;
        out_sum_sq[out_idx] = nan_val;
        out_min[out_idx] = nan_val;
        out_max[out_idx] = nan_val;
        out_first[out_idx] = nan_val;
        out_last[out_idx] = nan_val;
        return;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    double mn = col_data[lo];
    double mx = col_data[lo];

    for (long long i = lo; i < hi; i++) {
        double val = col_data[i];
        sum += val;
        sum_sq += val * val;
        if (val < mn) mn = val;
        if (val > mx) mx = val;
    }

    out_sum[out_idx] = sum;
    out_sum_sq[out_idx] = sum_sq;
    out_min[out_idx] = mn;
    out_max[out_idx] = mx;
    out_first[out_idx] = col_data[lo];
    out_last[out_idx] = col_data[hi - 1];
}
