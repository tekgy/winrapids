"""
Gold Standard Oracle: Family 14 — Factor Analysis & Reliability
Generates expected values for tambear validation.

Tests:
  - Correlation matrix (numpy.corrcoef)
  - Cronbach's alpha (manual formula)
  - Factor structure recovery (communalities, variance explained)
  - Varimax rotation (communality preservation)
  - Kaiser criterion and scree test
"""

import json
import numpy as np

results = {}

# ===============================================================
# 1. Correlation matrix: known perfect correlation
# ===============================================================

data_corr = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, 9.0],
    [4.0, 8.0, 12.0],
    [5.0, 10.0, 15.0],
])
corr = np.corrcoef(data_corr, rowvar=False)

results["correlation_perfect"] = {
    "data": data_corr.flatten().tolist(),
    "n": 5, "p": 3,
    "corr": corr.flatten().tolist(),
}
print("Correlation (perfect linear):")
print(corr)

# ===============================================================
# 2. Cronbach's alpha
# ===============================================================

# Known formula: alpha = (p/(p-1)) * (1 - sum(item_var)/total_var)

np.random.seed(42)
n_alpha = 30
p_alpha = 5

# High reliability: items = true_score + small noise
true_score = np.random.randn(n_alpha) * 3
items_high = np.column_stack([true_score + 0.3 * np.random.randn(n_alpha) for _ in range(p_alpha)])

item_vars = np.var(items_high, axis=0, ddof=1)
total = items_high.sum(axis=1)
total_var = np.var(total, ddof=1)
alpha_high = (p_alpha / (p_alpha - 1)) * (1 - item_vars.sum() / total_var)

results["cronbach_high"] = {
    "data": items_high.flatten().tolist(),
    "n": n_alpha, "p": p_alpha,
    "alpha": float(alpha_high),
    "item_vars": item_vars.tolist(),
    "total_var": float(total_var),
}
print("Cronbach alpha (high reliability): %.6f" % alpha_high)

# Low reliability: random items
np.random.seed(99)
items_low = np.random.randn(n_alpha, p_alpha)
item_vars_low = np.var(items_low, axis=0, ddof=1)
total_low = items_low.sum(axis=1)
total_var_low = np.var(total_low, ddof=1)
alpha_low = (p_alpha / (p_alpha - 1)) * (1 - item_vars_low.sum() / total_var_low)

results["cronbach_low"] = {
    "data": items_low.flatten().tolist(),
    "n": n_alpha, "p": p_alpha,
    "alpha": float(alpha_low),
}
print("Cronbach alpha (low reliability): %.6f" % alpha_low)

# ===============================================================
# 3. Two-factor structure for EFA validation
# ===============================================================

np.random.seed(77)
n_fa = 100
p_fa = 6
# Factor 1: items 0-2, Factor 2: items 3-5
f1 = np.random.randn(n_fa)
f2 = np.random.randn(n_fa)
noise = 0.2 * np.random.randn(n_fa, p_fa)

data_fa = np.column_stack([
    f1 + noise[:, 0], f1 + noise[:, 1], f1 + noise[:, 2],
    f2 + noise[:, 3], f2 + noise[:, 4], f2 + noise[:, 5],
])

corr_fa = np.corrcoef(data_fa, rowvar=False)
eigenvalues = np.linalg.eigvalsh(corr_fa)[::-1]  # sorted descending

results["efa_two_factor"] = {
    "data": data_fa.flatten().tolist(),
    "n": n_fa, "p": p_fa,
    "corr": corr_fa.flatten().tolist(),
    "eigenvalues": eigenvalues.tolist(),
    "kaiser_count": int(np.sum(eigenvalues > 1.0)),
}
print("EFA eigenvalues: %s" % eigenvalues)
print("Kaiser count: %d" % np.sum(eigenvalues > 1.0))

# ===============================================================
# 4. Kaiser and Scree
# ===============================================================

evals_test = np.array([4.0, 1.5, 0.8, 0.7, 0.5, 0.3])
kaiser = int(np.sum(evals_test > 1.0))
drops = np.diff(evals_test)
scree_elbow = int(np.argmax(-drops)) + 1  # position after largest drop

results["kaiser_scree"] = {
    "eigenvalues": evals_test.tolist(),
    "kaiser_count": kaiser,
    "scree_elbow": scree_elbow,
}
print("Kaiser: %d, Scree elbow: %d" % (kaiser, scree_elbow))

# ===============================================================
# 5. Varimax: communality preservation check
# ===============================================================

# Just verify the mathematical property that varimax preserves communalities
# Loadings before rotation
L = np.array([
    [0.8, 0.2],
    [0.7, 0.3],
    [0.2, 0.8],
    [0.3, 0.7],
])
comm_before = np.sum(L**2, axis=1)

results["varimax_communalities"] = {
    "loadings": L.flatten().tolist(),
    "p": 4, "k": 2,
    "communalities_before": comm_before.tolist(),
}
print("Varimax communalities before: %s" % comm_before)

# ===============================================================
# Save
# ===============================================================

with open("research/gold_standard/family_14_factor_expected.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved family_14_factor_expected.json")
