"""
Scale Ladder Benchmark: Descriptive Statistics
Compare numpy/scipy vs analytical on scale tiers.

For each tier: wall clock for mean+std+skew+kurtosis, peak memory.
This measures the PYTHON side. Rust side runs as a Rust binary.

Usage:
    python research/gold_standard/scale_ladder_descriptive.py
"""

import time
import sys
import numpy as np
from scipy import stats

scales = [
    ("10", 10),
    ("1K", 1_000),
    ("100K", 100_000),
    ("1M", 1_000_000),
    ("10M", 10_000_000),
    ("100M", 100_000_000),
    ("1B", 1_000_000_000),
]

print("=" * 80)
print("SCALE LADDER: Descriptive Statistics (numpy/scipy)")
print("=" * 80)
print(f"{'Scale':>8}  {'Alloc(s)':>10}  {'Mean(s)':>10}  {'Std(s)':>10}  {'Skew(s)':>10}  {'Kurt(s)':>10}  {'Total(s)':>10}  {'MB':>8}")
print("-" * 80)

for label, n in scales:
    try:
        # Allocation
        t0 = time.perf_counter()
        data = np.random.randn(n).astype(np.float64)
        t_alloc = time.perf_counter() - t0

        mb = data.nbytes / 1e6

        # Mean
        t0 = time.perf_counter()
        m = np.mean(data)
        t_mean = time.perf_counter() - t0

        # Std
        t0 = time.perf_counter()
        s = np.std(data)
        t_std = time.perf_counter() - t0

        # Skewness (scipy)
        t0 = time.perf_counter()
        sk = stats.skew(data)
        t_skew = time.perf_counter() - t0

        # Kurtosis (scipy)
        t0 = time.perf_counter()
        ku = stats.kurtosis(data)
        t_kurt = time.perf_counter() - t0

        t_total = t_alloc + t_mean + t_std + t_skew + t_kurt

        print(f"{label:>8}  {t_alloc:>10.4f}  {t_mean:>10.4f}  {t_std:>10.4f}  {t_skew:>10.4f}  {t_kurt:>10.4f}  {t_total:>10.4f}  {mb:>8.1f}")
        print(f"         values: mean={m:.6f}, std={s:.6f}, skew={sk:.6f}, kurt={ku:.6f}")

        del data

    except MemoryError:
        print(f"{label:>8}  *** MemoryError: cannot allocate {n * 8 / 1e9:.1f} GB ***")
        break
    except Exception as e:
        print(f"{label:>8}  *** Error: {e} ***")
        break

print("=" * 80)
