# K1 Pointwise GPU Acceleration Spec

**From**: fintek consumer (Claude, Market Atlas)
**Date**: 2026-03-27
**Context**: K01P02 has 20+ pointwise leaves running on CPU (prefix_binning or polars_columnar). These are all elementwise operations on 598K-element arrays. Trivially GPU-parallelizable.

## The Goal

ONE H2D transfer per ticker. ALL pointwise leaves compute on GPU without returning to CPU. ONE D2H transfer of all results. Data never leaves GPU between leaves.

## Current State

- 4 sequential leaves already on GPU (winrapids_batched)
- 6 windowed leaves already on GPU (winrapids_batched)
- 20+ pointwise leaves on CPU (prefix_binning or polars_columnar)
- 2 string-parsing leaves must stay CPU (conditions, trade_flags)

## The Operations to GPU-ify

### Trivial (one CuPy call each):
```python
# notional
notional = gpu_prices * gpu_sizes

# log transforms (3 channels: price, size, notional)
ln_price = cp.log(cp.maximum(gpu_prices, 1e-300))
log10_price = ln_price / cp.log(10.0)

# sqrt
sqrt_price = cp.sqrt(cp.maximum(gpu_prices, 0))

# reciprocal
recip_price = 1.0 / cp.maximum(gpu_prices, 1e-300)

# elapsed (timestamp differences in nanoseconds)
elapsed_ns = gpu_timestamps - gpu_timestamps[0]

# cyclical (sin/cos of time-of-day)
time_frac = (gpu_timestamps - day_start) / day_length
sin_time = cp.sin(2 * cp.pi * time_frac)
cos_time = cp.cos(2 * cp.pi * time_frac)

# microstructure booleans
round_lot = (gpu_sizes % 100 == 0).astype(cp.uint8)
odd_lot = (gpu_sizes < 100).astype(cp.uint8)
sub_penny = ((gpu_prices * 100) % 1 > 0.001).astype(cp.uint8)
round_price = ((gpu_prices * 4) % 1 < 0.001).astype(cp.uint8)

# decimal precision (count significant decimals)
# This one is trickier on GPU — needs string-like logic
# Option: precompute during ingest, or use a custom kernel
```

### The Fused Pointwise Kernel

Instead of 20 separate CuPy calls (20 kernel launches), write ONE kernel:

```cuda
extern "C" __global__
void fused_pointwise(
    const double* prices,
    const double* sizes,
    const long long* timestamps,
    // outputs:
    double* notional,
    double* ln_price, double* ln_size, double* ln_notional,
    double* sqrt_price,
    double* recip_price,
    double* elapsed_ns,
    double* sin_time, double* cos_time,
    unsigned char* round_lot, unsigned char* odd_lot,
    unsigned char* sub_penny, unsigned char* round_price,
    // constants:
    long long day_start,
    double day_length,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double p = prices[i];
    double s = sizes[i];
    long long t = timestamps[i];

    // All pointwise ops in ONE thread, ONE memory read per input
    double n = p * s;
    notional[i] = n;

    ln_price[i] = log(max(p, 1e-300));
    ln_size[i] = log(max(s, 1e-300));
    ln_notional[i] = log(max(n, 1e-300));

    sqrt_price[i] = sqrt(max(p, 0.0));
    recip_price[i] = 1.0 / max(p, 1e-300);

    elapsed_ns[i] = (double)(t - day_start);

    double tf = (double)(t - day_start) / day_length;
    sin_time[i] = sin(2.0 * 3.14159265358979323846 * tf);
    cos_time[i] = cos(2.0 * 3.14159265358979323846 * tf);

    // Integer operations
    int si = (int)s;
    round_lot[i] = (si % 100 == 0) ? 1 : 0;
    odd_lot[i] = (si < 100) ? 1 : 0;

    // Price precision (approximate — proper version needs string analysis)
    double p100 = p * 100.0;
    sub_penny[i] = (fabs(p100 - round(p100)) > 0.001) ? 1 : 0;
    double p4 = p * 4.0;
    round_price[i] = (fabs(p4 - round(p4)) < 0.001) ? 1 : 0;
}
```

### Performance Estimate

- **Current**: 20 leaves × ~1ms each (Polars/numpy) = ~20ms per ticker
- **GPU fused**: 1 kernel launch, 598K threads, ~0.1ms per ticker
- **Speedup**: ~200x per ticker
- **Full universe**: 4604 tickers × 0.1ms = 0.46 seconds (vs ~90 seconds current)

### Integration with Sequential/Windowed

If pointwise results stay on GPU, the sequential leaves (delta_ln, gap) can read them directly — no D2H/H2D round trip between K01P02C02 (pointwise) and K01P02C03 (sequential). The ENTIRE K01P02 pipeline becomes:

1. H2D: price, size, timestamp (3 arrays, ~14MB for AAPL)
2. GPU: fused pointwise → all 20+ pointwise results on GPU
3. GPU: sequential ops (diff, lag) on pointwise results → still on GPU
4. GPU: windowed ops (rolling mean/std) on sequential results → still on GPU
5. D2H: all results back to CPU for parquet writing

One H2D. One D2H. Everything between is GPU-to-GPU.

### What Stays CPU

- `conditions` (R11): parses comma-separated SIP condition code strings
- `trade_flags` (R12): decodes condition integers to 26 boolean flags
- `decimal_precision` (R06): proper implementation needs string representation of floats

These 3 leaves are string-processing or require CPU-specific logic. They can run in parallel with the GPU pipeline (CPU processes strings while GPU does math).

### Build Order

1. Prototype the fused pointwise kernel in CuPy RawKernel (like we did for bin stats)
2. Benchmark against current Polars/numpy path
3. If speedup confirmed, port to fintek as `trunk/backends/gpu_fused/fused_pointwise.py`
4. Wire into the daemon's K01 dispatch
5. Eventually: compiled C++ version for production
