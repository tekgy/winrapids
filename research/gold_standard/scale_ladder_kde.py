"""
Scale Ladder Benchmark: KDE (scipy)

Measures scipy.stats.gaussian_kde at increasing data sizes.
scipy uses FFT internally for evaluation, so it should scale better
than tambear's current O(n*m) naive KDE.

Usage:
    python research/gold_standard/scale_ladder_kde.py
"""

import time
import numpy as np
from scipy.stats import gaussian_kde

m = 1000  # eval points
eval_pts = np.linspace(-4.0, 4.0, m)
bw = 0.5  # fixed bandwidth

scales = [
    ("100",     100),
    ("1K",      1_000),
    ("10K",     10_000),
    ("100K",    100_000),
    ("1M",      1_000_000),
    # 10M+ omitted: takes >5min with naive O(n*m)
    # ("10M",     10_000_000),
    # ("100M",    100_000_000),
]

print("=" * 70)
print(f"SCALE LADDER: KDE (scipy gaussian_kde, m={m} eval points)")
print("=" * 70)
print(f"{'Scale':>8}  {'Alloc(s)':>10}  {'Fit(s)':>10}  {'Eval(s)':>10}  {'Total(s)':>10}  {'MB':>8}")
print("-" * 70)

for label, n in scales:
    try:
        # Allocation
        t0 = time.perf_counter()
        np.random.seed(42)
        data = np.random.randn(n)
        t_alloc = time.perf_counter() - t0
        mb = data.nbytes / 1e6

        # Fit KDE
        t0 = time.perf_counter()
        kernel = gaussian_kde(data, bw_method=bw)
        t_fit = time.perf_counter() - t0

        # Evaluate
        t0 = time.perf_counter()
        density = kernel(eval_pts)
        t_eval = time.perf_counter() - t0

        t_total = t_alloc + t_fit + t_eval

        dx = 8.0 / (m - 1)
        integral = np.sum(density) * dx

        print(f"{label:>8}  {t_alloc:>10.4f}  {t_fit:>10.4f}  {t_eval:>10.4f}  {t_total:>10.4f}  {mb:>8.1f}")
        print(f"         peak={density.max():.6f}, integral~={integral:.4f}")

        del data, kernel

    except MemoryError:
        print(f"{label:>8}  *** MemoryError ***")
        break
    except Exception as e:
        print(f"{label:>8}  *** Error: {e} ***")
        break

print("=" * 70)
print()
print("scipy.gaussian_kde uses direct evaluation: O(n*m).")
print("At large n, both scipy and tambear naive are O(n*m).")
print("FFT-KDE would be O(n + m*log(m)) — the build item for tambear.")
