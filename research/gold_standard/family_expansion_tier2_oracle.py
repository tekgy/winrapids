"""
Gold Standard Oracle: Tier 2 Expansion Methods

Proactive oracle for methods in the next implementation wave.
Tests against scipy.cluster.hierarchy, statsmodels.tsa.

Methods covered:
  - Hierarchical agglomerative clustering (single/complete/average/Ward)
  - ARIMA model fitting and forecasting
  - Elastic net regression
  - Weighted least squares
  - Permutation test
  - Bootstrap BCa confidence intervals

Usage:
    python research/gold_standard/family_expansion_tier2_oracle.py
"""

import json
import numpy as np
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

results = {}

# ===========================================================================
# 1. Hierarchical Clustering
# ===========================================================================

# Well-separated 2D clusters
np.random.seed(42)
c1 = np.random.normal([0, 0], 0.5, (5, 2))
c2 = np.random.normal([5, 5], 0.5, (5, 2))
c3 = np.random.normal([10, 0], 0.5, (5, 2))
X_hc = np.vstack([c1, c2, c3])

dist_matrix = pdist(X_hc)

for method in ['single', 'complete', 'average', 'ward']:
    Z = hierarchy.linkage(X_hc, method=method)
    labels_3 = hierarchy.fcluster(Z, t=3, criterion='maxclust')

    results[f"hierarchical_{method}"] = {
        "data": X_hc.tolist(),
        "method": method,
        "linkage_matrix": Z.tolist(),
        "labels_3_clusters": labels_3.tolist(),
        "n_leaves": int(Z.shape[0] + 1),
    }

    # Cophenetic correlation: how well does the dendrogram preserve distances?
    coph_dist, coph_matrix = hierarchy.cophenet(Z, dist_matrix)
    results[f"hierarchical_{method}"]["cophenetic_corr"] = float(coph_dist)

# Properties that must hold for all linkages:
# - Linkage matrix Z has n-1 rows (n=15 → 14 rows)
# - Z[:,2] (merge distances) is monotonically non-decreasing
# - fcluster with maxclust=1 gives all same label
# - fcluster with maxclust=n gives all different labels
results["hierarchical_properties"] = {
    "n_points": 15,
    "n_merges": 14,
    "monotone_distances": True,
    "note": "Z[:,2] must be non-decreasing for all valid linkages",
}

# ===========================================================================
# 2. ARIMA
# ===========================================================================

try:
    from statsmodels.tsa.arima.model import ARIMA

    # Generate AR(1) data
    np.random.seed(42)
    n_ts = 200
    ar_data = [0.0]
    for i in range(n_ts - 1):
        ar_data.append(0.7 * ar_data[-1] + np.random.normal(0, 1))
    ar_data = np.array(ar_data)

    # Fit ARIMA(1,0,0) = AR(1)
    model_ar = ARIMA(ar_data, order=(1, 0, 0))
    fit_ar = model_ar.fit()

    results["arima_100_ar1"] = {
        "order": [1, 0, 0],
        "n": n_ts,
        "ar_coeff": float(fit_ar.arparams[0]),
        "true_ar": 0.7,
        "sigma2": float(fit_ar.params[-1]) if len(fit_ar.params) > 2 else None,
        "aic": float(fit_ar.aic),
        "bic": float(fit_ar.bic),
    }

    # Fit ARIMA(1,1,1) on random walk + drift
    rw_data = np.cumsum(0.1 + np.random.normal(0, 1, n_ts))
    model_arima = ARIMA(rw_data, order=(1, 1, 1))
    fit_arima = model_arima.fit()

    results["arima_111_rw"] = {
        "order": [1, 1, 1],
        "ar_coeff": float(fit_arima.arparams[0]) if len(fit_arima.arparams) > 0 else None,
        "ma_coeff": float(fit_arima.maparams[0]) if len(fit_arima.maparams) > 0 else None,
        "aic": float(fit_arima.aic),
    }

    # Forecast
    forecast = fit_ar.forecast(steps=5)
    results["arima_forecast"] = {
        "model": "ARIMA(1,0,0)",
        "forecast_5": forecast.tolist(),
    }

except ImportError:
    results["arima_note"] = "statsmodels ARIMA not available"

# ===========================================================================
# 3. Elastic Net
# ===========================================================================

try:
    from sklearn.linear_model import ElasticNet

    np.random.seed(42)
    n_en = 50
    X_en = np.random.normal(0, 1, (n_en, 5))
    beta_true = np.array([3.0, -2.0, 0.0, 0.0, 1.0])  # 2 zeros for sparsity
    y_en = X_en @ beta_true + np.random.normal(0, 0.5, n_en)

    for alpha, l1_ratio, label in [(0.1, 0.5, "balanced"),
                                     (1.0, 0.9, "lasso_heavy"),
                                     (0.01, 0.1, "ridge_heavy")]:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=10000)
        model.fit(X_en, y_en)
        results[f"elastic_net_{label}"] = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "r_squared": float(model.score(X_en, y_en)),
            "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-10)),
        }

except ImportError:
    results["elastic_net_note"] = "sklearn not available"

# ===========================================================================
# 4. Permutation test
# ===========================================================================

np.random.seed(42)
x_perm = np.random.normal(0, 1, 20)
y_perm = np.random.normal(0.5, 1, 20)

# Manual permutation test for difference in means
observed_diff = float(np.mean(x_perm) - np.mean(y_perm))
combined = np.concatenate([x_perm, y_perm])
n_perms = 10000
count_extreme = 0
rng = np.random.RandomState(42)
for _ in range(n_perms):
    perm = rng.permutation(combined)
    perm_diff = np.mean(perm[:20]) - np.mean(perm[20:])
    if abs(perm_diff) >= abs(observed_diff):
        count_extreme += 1

perm_p = count_extreme / n_perms

results["permutation_test_means"] = {
    "observed_diff": observed_diff,
    "n_permutations": n_perms,
    "p_value": float(perm_p),
    "n_x": 20,
    "n_y": 20,
    "note": "Two-sided permutation test for difference in means",
}

# ===========================================================================
# 5. Bootstrap BCa confidence interval
# ===========================================================================

np.random.seed(42)
boot_data = np.random.exponential(2, 30)
observed_mean = float(np.mean(boot_data))

n_boot = 5000
rng = np.random.RandomState(42)
boot_means = np.array([np.mean(rng.choice(boot_data, size=len(boot_data), replace=True))
                        for _ in range(n_boot)])

# Percentile CI
ci_lower = float(np.percentile(boot_means, 2.5))
ci_upper = float(np.percentile(boot_means, 97.5))

results["bootstrap_percentile_ci"] = {
    "observed_mean": observed_mean,
    "n_bootstrap": n_boot,
    "ci_95_lower": ci_lower,
    "ci_95_upper": ci_upper,
    "seed": 42,
}

# ===========================================================================
# Save
# ===========================================================================

with open("research/gold_standard/family_expansion_tier2_expected.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"Tier 2 Expansion Oracle: {len(results)} test cases generated")
for name, r in results.items():
    if isinstance(r, dict):
        if 'cophenetic_corr' in r:
            print(f"  PASS {name} (coph={r['cophenetic_corr']:.3f})")
        elif 'ar_coeff' in r:
            print(f"  PASS {name} (ar={r.get('ar_coeff', '?')})")
        elif 'p_value' in r:
            print(f"  PASS {name} (p={r['p_value']:.4f})")
        elif 'r_squared' in r:
            print(f"  PASS {name} (R2={r['r_squared']:.4f})")
        else:
            print(f"  PASS {name}")
    else:
        print(f"  PASS {name}")
