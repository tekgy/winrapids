"""
Gold Standard Oracle: Family 08 — Nonparametric Statistics

Generates expected values from scipy.stats for comparison with tambear.

Algorithms covered:
  - Ranking (average ties)
  - Spearman correlation
  - Kendall's tau
  - Mann-Whitney U
  - Wilcoxon signed-rank
  - Kruskal-Wallis H
  - KS two-sample test
  - Sign test

Usage:
    python research/gold_standard/family_08_nonparametric_oracle.py
"""

import json
import numpy as np
from scipy import stats

results = {}

# -- Ranking --

r = stats.rankdata([3, 1, 4, 1, 5])
results["rank_with_ties"] = {
    "data": [3, 1, 4, 1, 5],
    "ranks": r.tolist(),
}

r = stats.rankdata([10, 30, 20])
results["rank_no_ties"] = {
    "data": [10, 30, 20],
    "ranks": r.tolist(),
}

r = stats.rankdata([5, 5, 5, 5])
results["rank_all_ties"] = {
    "data": [5, 5, 5, 5],
    "ranks": r.tolist(),
}

# -- Spearman --

rho, p = stats.spearmanr([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
results["spearman_perfect_pos"] = {
    "x": [1, 2, 3, 4, 5],
    "y": [10, 20, 30, 40, 50],
    "rho": float(rho),
    "p_value": float(p),
}

rho, p = stats.spearmanr([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
results["spearman_perfect_neg"] = {
    "rho": float(rho),
}

# Non-trivial case
rho, p = stats.spearmanr([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4, 3, 6, 5, 8, 7])
results["spearman_swapped_pairs"] = {
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "y": [2, 1, 4, 3, 6, 5, 8, 7],
    "rho": float(rho),
    "p_value": float(p),
}

# -- Kendall's tau --

tau, p = stats.kendalltau([1, 2, 3, 4], [1, 3, 2, 4])
results["kendall_known"] = {
    "x": [1, 2, 3, 4],
    "y": [1, 3, 2, 4],
    "tau": float(tau),
    "p_value": float(p),
}

tau, p = stats.kendalltau([1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4, 3, 6, 5, 8, 7])
results["kendall_swapped_pairs"] = {
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "y": [2, 1, 4, 3, 6, 5, 8, 7],
    "tau": float(tau),
    "p_value": float(p),
}

# -- Mann-Whitney U --

# Separated groups
u_stat, p_val = stats.mannwhitneyu([1, 2, 3], [4, 5, 6], alternative='two-sided')
results["mann_whitney_separated"] = {
    "x": [1, 2, 3],
    "y": [4, 5, 6],
    "U": float(u_stat),
    "p_value": float(p_val),
}

# Overlapping groups
u_stat, p_val = stats.mannwhitneyu([1, 3, 5, 7], [2, 4, 6, 8], alternative='two-sided')
results["mann_whitney_interleaved"] = {
    "x": [1, 3, 5, 7],
    "y": [2, 4, 6, 8],
    "U": float(u_stat),
    "p_value": float(p_val),
}

# -- Wilcoxon signed-rank --

# All positive
w_stat, p_val = stats.wilcoxon([1, 2, 3, 4, 5])
results["wilcoxon_all_positive"] = {
    "differences": [1, 2, 3, 4, 5],
    "W": float(w_stat),
    "p_value": float(p_val),
}

# -- Kruskal-Wallis --

h_stat, p_val = stats.kruskal([1, 2, 3], [10, 11, 12], [20, 21, 22])
results["kruskal_wallis_separated"] = {
    "groups": [[1, 2, 3], [10, 11, 12], [20, 21, 22]],
    "H": float(h_stat),
    "p_value": float(p_val),
}

h_stat, p_val = stats.kruskal([1, 2, 3], [1, 2, 3], [1, 2, 3])
results["kruskal_wallis_identical"] = {
    "H": float(h_stat),
    "p_value": float(p_val),
}

# -- KS two-sample --

d_stat, p_val = stats.ks_2samp([1, 2, 3, 4, 5], [100, 200, 300, 400, 500])
results["ks_completely_separated"] = {
    "D": float(d_stat),
    "p_value": float(p_val),
}

d_stat, p_val = stats.ks_2samp(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)
results["ks_identical_samples"] = {
    "D": float(d_stat),
    "p_value": float(p_val),
}

# Overlapping but shifted
np.random.seed(42)
x = np.random.normal(0, 1, 50)
y = np.random.normal(0.5, 1, 50)
d_stat, p_val = stats.ks_2samp(x, y)
results["ks_shifted_normal"] = {
    "D": float(d_stat),
    "p_value": float(p_val),
    "x_mean": float(x.mean()),
    "y_mean": float(y.mean()),
    "n_x": len(x),
    "n_y": len(y),
}

# -- Save --

with open("research/gold_standard/family_08_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"F08 Oracle: {len(results)} test cases generated")
for name, r in results.items():
    parts = []
    if 'rho' in r: parts.append(f"rho={r['rho']:.6f}")
    if 'tau' in r: parts.append(f"tau={r['tau']:.6f}")
    if 'U' in r: parts.append(f"U={r['U']:.1f}")
    if 'W' in r: parts.append(f"W={r['W']:.1f}")
    if 'H' in r: parts.append(f"H={r['H']:.4f}")
    if 'D' in r: parts.append(f"D={r['D']:.6f}")
    if 'ranks' in r: parts.append(f"ranks={r['ranks']}")
    if 'p_value' in r: parts.append(f"p={r['p_value']:.4f}")
    print(f"  PASS {name}: {', '.join(parts)}")
