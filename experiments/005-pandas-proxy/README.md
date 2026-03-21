# Experiment 005: Pandas GPU Proxy Pattern

## Hypothesis

A proxy wrapper around pandas can transparently GPU-accelerate numeric operations while falling back to CPU for unsupported operations, without changing user code beyond a single `gpu_accelerate()` call.

## Method

Built `GpuAcceleratedSeries` and `GpuAcceleratedDataFrame` proxy classes (~350 lines Python) that:
- Wrap pandas objects, maintaining a lazy GPU copy via CuPy
- Intercept numeric operations (sum, mean, arithmetic, comparison) and run on GPU
- Fall back to pandas via `__getattr__` for unsupported operations
- Log all fallbacks for visibility

Tested correctness against pandas 3.0.1, then benchmarked on 10M rows.

## Results

### Correctness

All tests pass:
- Aggregations (sum, mean, min, max): match pandas to < 0.01
- Arithmetic (a * b + a): match pandas
- Comparison + filtering (flag == 1): match pandas
- DataFrame-level sum: match pandas
- Fallback (describe): works via delegation

### Transparent Usage

Real pandas code works with only one change: `gdf = gpu_accelerate(df)`:
```python
gdf = gpu_accelerate(df)
avg_price = gdf["price"].mean()        # GPU
total_volume = gdf["volume"].sum()     # GPU
high_volume = gdf[gdf["volume"] > 5000]  # GPU comparison, pandas filter
gdf["notional"] = gdf["price"] * gdf["volume"]  # GPU arithmetic
```

### Benchmarks (10M rows)

| Operation | pandas (ms) | Proxy (ms) | Speedup |
|-----------|------------|-----------|---------|
| sum | 7.9 | 0.13 | 61x |
| mean | 12.0 | 0.12 | 103x |
| filtered sum | 30.2 | 59.2 | 0.5x (SLOWER) |

## Analysis

### What Works

Simple aggregations (sum, mean, min, max) get 60-100x speedup. The proxy overhead is negligible — the GPU computation dominates. This is the cudf.pandas sweet spot.

### What Doesn't Work (Yet)

**Filtered sum is 2x SLOWER than pandas.** The proxy's filtering path materializes the boolean mask to CPU (D2H), applies it via pandas indexing, then transfers the filtered result back to GPU (H2D). The roundtrip kills the benefit.

The fix is clear: keep boolean masks on GPU and use them directly for GPU-side filtering. This requires the proxy to track which results are GPU-resident and avoid unnecessary D2H materialization. The Experiment 004 GpuFrame does this correctly (filtered_sum = 0.38 ms = 92x faster than pandas).

### The Fundamental Tension

The proxy pattern works by maintaining dual representations (CPU + GPU). Operations that touch only one domain are fast. Operations that chain across domains (GPU comparison -> CPU filter -> GPU sum) pay the PCIe tax on every boundary crossing.

cudf.pandas solves this by having EVERYTHING on GPU and only falling back to CPU as a last resort. Our current proxy takes the opposite approach: everything on CPU with GPU acceleration for individual operations. For simple aggregations this is fine. For complex workflows, it degrades.

### The Right Architecture

1. **For simple acceleration** (sum, mean, arithmetic): the proxy pattern is perfect. 60-100x speedup with zero code changes.

2. **For complex workflows**: use GpuFrame (Experiment 004) where data stays on GPU by default. The proxy pattern isn't the right abstraction here.

3. **The middle ground**: a proxy that is lazily GPU-resident and avoids materialization when consecutive operations can fuse on GPU. This is what cudf.pandas does, and it requires a full GPU DataFrame implementation underneath.

## Conclusions

1. **The proxy pattern works for simple ops.** 60-103x speedup on aggregations with one line of code change.

2. **Chained operations expose the proxy's weakness.** D2H + H2D roundtrips on every intermediate result kill performance.

3. **The proxy NEEDS a GPU DataFrame underneath** to avoid unnecessary materialization. Our GpuFrame from Experiment 004 is the right foundation.

4. **Fallback mechanism works.** `describe()` gracefully falls back to pandas, logged for visibility.

5. **Path forward:** Build a proper GPU-resident pandas-compatible API (like cudf.pandas) on top of GpuFrame, not as a thin proxy over pandas.

## Files

- `pandas_proxy.py` — GpuAcceleratedSeries, GpuAcceleratedDataFrame, tests and benchmarks
