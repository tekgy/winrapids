"""
WinRapids Experiment 007: Polars GPU Acceleration on Windows

Questions:
1. Does Polars GPU engine work on Windows? (Expected: No)
2. How is the cuDF backend integrated? (Pluggable or compiled-in?)
3. What interface would a custom GPU backend need?
4. How does Polars CPU compare to our GPU DataFrame?
"""

import time
import numpy as np
import polars as pl

print(f"Polars version: {pl.__version__}")
print()

# ============================================================
# Test 1: Does Polars GPU engine work?
# ============================================================

print("=== Test 1: Polars GPU Engine ===\n")

n = 10_000_000
df = pl.DataFrame({
    "value": np.random.randn(n).astype(np.float64),
    "flag": np.random.randint(0, 2, n).astype(np.int8),
})

# Try to use GPU engine
try:
    result = df.lazy().select(pl.col("value").sum()).collect(engine="gpu")
    print(f"  GPU engine: WORKS (result={result})")
except Exception as e:
    error_str = str(e)
    # Truncate long error messages
    if len(error_str) > 200:
        error_str = error_str[:200] + "..."
    print(f"  GPU engine: NOT AVAILABLE")
    print(f"  Error: {error_str}")
print()

# ============================================================
# Test 2: Polars CPU baseline benchmarks
# ============================================================

print("=== Test 2: Polars CPU Benchmarks ===\n")

# Sum
_ = df.select(pl.col("value").sum())  # warmup
t0 = time.perf_counter()
for _ in range(10):
    _ = df.select(pl.col("value").sum())
t_sum = (time.perf_counter() - t0) / 10

# Mean
t0 = time.perf_counter()
for _ in range(10):
    _ = df.select(pl.col("value").mean())
t_mean = (time.perf_counter() - t0) / 10

# Filtered sum (lazy)
_ = df.lazy().filter(pl.col("flag") == 1).select(pl.col("value").sum()).collect()
t0 = time.perf_counter()
for _ in range(5):
    _ = df.lazy().filter(pl.col("flag") == 1).select(pl.col("value").sum()).collect()
t_filtered = (time.perf_counter() - t0) / 5

# Column arithmetic
t0 = time.perf_counter()
for _ in range(5):
    _ = df.select((pl.col("value") * 2 + 1).alias("result"))
t_arith = (time.perf_counter() - t0) / 5

print(f"  sum (10M):          {t_sum*1000:8.3f} ms")
print(f"  mean (10M):         {t_mean*1000:8.3f} ms")
print(f"  filtered sum (10M): {t_filtered*1000:8.3f} ms")
print(f"  arithmetic (10M):   {t_arith*1000:8.3f} ms")
print()

# ============================================================
# Test 3: Polars vs our GPU DataFrame
# ============================================================

print("=== Test 3: Comparison with WinRapids GPU DataFrame ===\n")

try:
    import cupy as cp

    # GPU DataFrame (from experiment 004)
    gpu_values = cp.asarray(df["value"].to_numpy())
    gpu_flags = cp.asarray(df["flag"].to_numpy())
    cp.cuda.Device(0).synchronize()

    # GPU sum
    _ = float(cp.sum(gpu_values))
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = float(cp.sum(gpu_values))
    cp.cuda.Device(0).synchronize()
    t_gpu_sum = (time.perf_counter() - t0) / 10

    # GPU filtered sum
    _ = float(cp.sum(gpu_values[gpu_flags == 1]))
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        _ = float(cp.sum(gpu_values[gpu_flags == 1]))
    cp.cuda.Device(0).synchronize()
    t_gpu_filtered = (time.perf_counter() - t0) / 5

    print(f"  {'Operation':<25s} {'Polars CPU':>12s} {'GPU':>12s} {'Speedup':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'sum':<25s} {t_sum*1000:>11.3f}ms {t_gpu_sum*1000:>11.3f}ms {t_sum/t_gpu_sum:>9.1f}x")
    print(f"  {'filtered sum':<25s} {t_filtered*1000:>11.3f}ms {t_gpu_filtered*1000:>11.3f}ms {t_filtered/t_gpu_filtered:>9.1f}x")
    print()
except ImportError:
    print("  CuPy not available — skipping GPU comparison")

# ============================================================
# Test 4: Polars Arrow interop
# ============================================================

print("=== Test 4: Polars Arrow Interop ===\n")

import pyarrow as pa

# Polars -> Arrow
t0 = time.perf_counter()
arrow_table = df.to_arrow()
t_to_arrow = time.perf_counter() - t0

# Arrow -> Polars
t0 = time.perf_counter()
df_back = pl.from_arrow(arrow_table)
t_from_arrow = time.perf_counter() - t0

print(f"  Polars -> Arrow: {t_to_arrow*1000:.3f} ms")
print(f"  Arrow -> Polars: {t_from_arrow*1000:.3f} ms")
print(f"  Zero-copy:       Both should be near-zero for numeric columns")
print()

# ============================================================
# Test 5: Check GPU backend architecture
# ============================================================

print("=== Test 5: GPU Backend Architecture ===\n")

# Check what Polars exposes about GPU
print("  Checking Polars GPU-related attributes...")

gpu_attrs = [attr for attr in dir(pl) if 'gpu' in attr.lower() or 'cuda' in attr.lower()]
print(f"  GPU-related top-level attributes: {gpu_attrs if gpu_attrs else 'None'}")

# Check collect engine options
try:
    engines = ["cpu", "gpu", "streaming"]
    for eng in engines:
        try:
            _ = df.lazy().select(pl.col("value").sum()).collect(engine=eng)
            print(f"  Engine '{eng}': available")
        except Exception as e:
            err = str(e)[:100]
            print(f"  Engine '{eng}': {err}")
except Exception as e:
    print(f"  Engine check failed: {e}")

print()

# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("Summary:")
print()
print(f"  Polars CPU sum:          {t_sum*1000:.3f} ms")
print(f"  Polars CPU filtered sum: {t_filtered*1000:.3f} ms")
if 't_gpu_sum' in dir():
    print(f"  GPU sum:                 {t_gpu_sum*1000:.3f} ms ({t_sum/t_gpu_sum:.0f}x faster)")
    print(f"  GPU filtered sum:        {t_gpu_filtered*1000:.3f} ms ({t_filtered/t_gpu_filtered:.0f}x faster)")
print()
print("  Polars GPU engine on Windows: NOT AVAILABLE (requires cuDF/Linux)")
print("  Integration path: Polars <-> Arrow (zero-copy) <-> GPU (CuPy)")
