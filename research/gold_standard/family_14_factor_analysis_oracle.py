"""
Gold Standard Oracle: Family 14 — Factor Analysis & SEM

Generates expected values from factor_analyzer (Python) and psych (R concept).

Tests covered:
  - Correlation matrix from data (exact, matches numpy)
  - EFA via principal axis factoring (communalities, eigenvalues)
  - Varimax rotation (preserves communalities)
  - Cronbach's alpha (reliability)
  - Kaiser criterion, scree plot

Reference: factor_analyzer 0.5+, numpy, analytical formulas
Usage:
    python research/gold_standard/family_14_factor_analysis_oracle.py
"""

import json
import numpy as np

results = {}

# ── Correlation matrix (numpy ground truth) ──────────────────────────────
np.random.seed(42)
n, p = 100, 4

# Generate data with known factor structure:
# 2 factors: F1 → vars 0,1 (loading ~0.8); F2 → vars 2,3 (loading ~0.8)
f1 = np.random.randn(n)
f2 = np.random.randn(n)
noise = np.random.randn(n, p) * 0.3
data = np.column_stack([
    0.8 * f1 + noise[:, 0],
    0.8 * f1 + noise[:, 1],
    0.8 * f2 + noise[:, 2],
    0.8 * f2 + noise[:, 3],
])

corr_numpy = np.corrcoef(data, rowvar=False)

results["correlation_matrix"] = {
    "n": n,
    "p": p,
    "data_seed": 42,
    "corr_matrix": [[float(v) for v in row] for row in corr_numpy],
    "diagonal_all_ones": bool(np.allclose(np.diag(corr_numpy), 1.0)),
    "symmetric": bool(np.allclose(corr_numpy, corr_numpy.T)),
    "note": "numpy.corrcoef ground truth. Diagonal=1, symmetric."
}

# ── Factor analysis via factor_analyzer ──────────────────────────────────
try:
    from factor_analyzer import FactorAnalyzer

    fa = FactorAnalyzer(n_factors=2, rotation=None, method='principal')
    fa.fit(data)

    results["efa_principal_axis_2factors"] = {
        "loadings": [[float(v) for v in row] for row in fa.loadings_],
        "communalities": [float(v) for v in fa.get_communalities()],
        "eigenvalues": [float(v) for v in fa.get_eigenvalues()[0]],  # original eigenvalues
        "variance_explained": [float(v) for v in fa.get_factor_variance()[1]],  # proportion
        "uniquenesses": [float(v) for v in fa.get_uniquenesses()],
        "note": "factor_analyzer PAF with 2 factors, no rotation"
    }

    # Varimax rotation
    fa_rot = FactorAnalyzer(n_factors=2, rotation='varimax', method='principal')
    fa_rot.fit(data)

    comm_unrot = fa.get_communalities()
    comm_rot = fa_rot.get_communalities()

    results["varimax_rotation"] = {
        "loadings_rotated": [[float(v) for v in row] for row in fa_rot.loadings_],
        "communalities_preserved": bool(np.allclose(comm_unrot, comm_rot, atol=1e-6)),
        "communalities_unrotated": [float(v) for v in comm_unrot],
        "communalities_rotated": [float(v) for v in comm_rot],
        "note": "Varimax must preserve communalities (row sums of squared loadings)"
    }

except ImportError:
    print("factor_analyzer not available — using analytical values")
    # Kaiser criterion: eigenvalue > 1 → retain
    eigenvalues = np.linalg.eigvalsh(corr_numpy)[::-1]
    results["eigenvalues_for_kaiser"] = {
        "eigenvalues": [float(v) for v in eigenvalues],
        "kaiser_count": int(np.sum(eigenvalues > 1.0)),
        "note": "Eigenvalues of correlation matrix. Kaiser: retain if > 1."
    }

# ── Cronbach's alpha (analytical) ────────────────────────────────────────
# α = (p/(p-1)) × (1 - Σvar_j / var_total)
# For perfectly correlated items: α → 1
# For uncorrelated items: α → 0

# High reliability data (correlated items)
high_rel = data  # our factor-structured data
item_vars = np.var(high_rel, axis=0, ddof=1)
total_var = np.var(np.sum(high_rel, axis=1), ddof=1)
alpha_high = (p / (p - 1)) * (1 - np.sum(item_vars) / total_var)

results["cronbach_alpha_high"] = {
    "alpha": float(alpha_high),
    "n_items": p,
    "sum_item_var": float(np.sum(item_vars)),
    "total_var": float(total_var),
    "note": "Factor-structured data: α should be > 0.7"
}

# Low reliability (random items)
np.random.seed(99)
low_rel = np.random.randn(100, 4)
item_vars_low = np.var(low_rel, axis=0, ddof=1)
total_var_low = np.var(np.sum(low_rel, axis=1), ddof=1)
alpha_low = (p / (p - 1)) * (1 - np.sum(item_vars_low) / total_var_low)

results["cronbach_alpha_low"] = {
    "alpha": float(alpha_low),
    "n_items": p,
    "note": "Random data: α should be near 0 or negative"
}

# ── Scree / Kaiser ───────────────────────────────────────────────────────
eigenvalues = np.sort(np.linalg.eigvalsh(corr_numpy))[::-1]
results["kaiser_criterion"] = {
    "eigenvalues": [float(v) for v in eigenvalues],
    "n_above_1": int(np.sum(eigenvalues > 1.0)),
    "note": "Kaiser: retain factors with eigenvalue > 1"
}

# ── Write output ─────────────────────────────────────────────────────────
output_path = "research/gold_standard/family_14_expected.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Family 14 oracle: {len(results)} test groups written to {output_path}")
for key in results:
    print(f"  {key}")
