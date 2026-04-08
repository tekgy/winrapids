"""
Scale Ladder Benchmark: KDE (sklearn KernelDensity)

Measures sklearn's KernelDensity at increasing data sizes.
sklearn uses KD-tree or ball-tree internally — O(n log n) fit, O(m log n) eval.
But the constant factors are large and memory grows fast.

Compare with tambear's kde_fft() — O(n) bin + O(m log m) convolve.

Usage:
    python research/gold_standard/scale_ladder_kde_fft.py
"""

import time
import sys
import numpy as np
from sklearn.neighbors import KernelDensity

m = 1024  # eval grid
bw = 0.3

scales = [
    ("1K",      1_000),
    ("10K",     10_000),
    ("100K",    100_000),
    ("1M",      1_000_000),
    ("10M",     10_000_000),
    ("100M",    100_000_000),
]

print("=" * 80)
print(f"SCALE LADDER: KDE (sklearn KernelDensity, m={m} eval points, bw={bw})")
print("=" * 80)
print(f"{'Scale':>8}  {'Fit(s)':>10}  {'Eval(s)':>10}  {'Total(s)':>10}  {'MB':>8}")
print("-" * 80)

for label, n in scales:
    try:
        # Generate data
        np.random.seed(42)
        t0 = time.perf_counter()
        data = np.random.randn(n).reshape(-1, 1)
        t_alloc = time.perf_counter() - t0
        mb = data.nbytes / 1e6

        grid = np.linspace(-4, 4, m).reshape(-1, 1)

        # Fit KDE
        t0 = time.perf_counter()
        kde = KernelDensity(bandwidth=bw, kernel='gaussian').fit(data)
        t_fit = time.perf_counter() - t0

        # Evaluate (score_samples returns log-density)
        t0 = time.perf_counter()
        log_density = kde.score_samples(grid)
        density = np.exp(log_density)
        t_eval = time.perf_counter() - t0

        t_total = t_fit + t_eval

        dx = 8.0 / (m - 1)
        integral = np.sum(density) * dx

        print(f"{label:>8}  {t_fit:>10.4f}  {t_eval:>10.4f}  {t_total:>10.4f}  {mb:>8.1f}")
        print(f"         peak={density.max():.6f}, integral~={integral:.4f}")

        del data, kde

    except MemoryError:
        print(f"{label:>8}  *** MemoryError: cannot allocate {n * 8 / 1e9:.1f} GB ***")
        break
    except Exception as e:
        print(f"{label:>8}  *** Error: {e} ***")
        break

print("=" * 80)
print()
print("sklearn KDE: KD-tree fit O(n log n), eval O(m log n).")
print("tambear FFT KDE: bin O(n), convolve O(m log m).")
print("At large n, tambear's FFT approach dominates because it decouples n from m.")
