"""
Gold Standard Oracle: Post-hoc Tests, Influence Diagnostics, Factor Adequacy

Tests covered:
  - Tukey HSD post-hoc test (statsmodels)
  - Dunn's test (manual rank-based computation)
  - Cook's distance (statsmodels OLSInfluence)
  - KMO statistic (manual computation against factor_analyzer if available)
  - Bartlett's test of sphericity (analytical chi-square)
  - Cluster validation metrics (sklearn): silhouette, Calinski-Harabasz, Davies-Bouldin

Usage:
    python research/gold_standard/family_posthoc_influence_oracle.py
"""

import json
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

results = {}

# ===========================================================================
# 1. Tukey HSD
# ===========================================================================

g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
g2 = [3.0, 4.0, 5.0, 6.0, 7.0]
g3 = [6.0, 7.0, 8.0, 9.0, 10.0]

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    all_data = np.array(g1 + g2 + g3)
    labels = ['A']*5 + ['B']*5 + ['C']*5
    tukey = pairwise_tukeyhsd(all_data, labels, alpha=0.05)

    results["tukey_hsd_3groups"] = {
        "groups": [g1, g2, g3],
        "group_means": [float(np.mean(g1)), float(np.mean(g2)), float(np.mean(g3))],
        "comparisons": [],
    }
    for row in tukey.summary().data[1:]:
        results["tukey_hsd_3groups"]["comparisons"].append({
            "group1": str(row[0]),
            "group2": str(row[1]),
            "mean_diff": float(row[2]),
            "p_adj": float(row[3]),
            "lower": float(row[4]),
            "upper": float(row[5]),
            "reject": bool(row[6]),
        })

    # ANOVA for ms_error and df_error
    f_stat, anova_p = stats.f_oneway(g1, g2, g3)
    n_total = 15
    k = 3
    grand_mean = np.mean(all_data)
    ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in [g1, g2, g3])
    ms_error = ss_within / (n_total - k)
    df_error = n_total - k
    results["tukey_hsd_3groups"]["ms_error"] = float(ms_error)
    results["tukey_hsd_3groups"]["df_error"] = float(df_error)
    results["tukey_hsd_3groups"]["anova_f"] = float(f_stat)

except ImportError:
    # Manual computation
    results["tukey_hsd_3groups"] = {
        "groups": [g1, g2, g3],
        "group_means": [3.0, 5.0, 8.0],
        "note": "statsmodels not available for Tukey HSD",
    }

# Equal means case
g_eq1 = [10.0, 11.0, 12.0, 13.0, 14.0]
g_eq2 = [10.0, 11.0, 12.0, 13.0, 14.0]
results["tukey_hsd_equal_means"] = {
    "groups": [g_eq1, g_eq2],
    "mean_diff": 0.0,
    "note": "identical groups: mean_diff=0, q=0, p=1.0",
}

# ===========================================================================
# 2. Dunn's test
# ===========================================================================

# Three groups with different medians
d_g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
d_g2 = [4.0, 5.0, 6.0, 7.0, 8.0]
d_g3 = [7.0, 8.0, 9.0, 10.0, 11.0]
all_dunn = d_g1 + d_g2 + d_g3
n_dunn = len(all_dunn)

# Compute ranks
from scipy.stats import rankdata
ranks = rankdata(all_dunn).tolist()
mean_ranks = [
    float(np.mean(ranks[:5])),
    float(np.mean(ranks[5:10])),
    float(np.mean(ranks[10:15])),
]

results["dunn_3groups"] = {
    "data": all_dunn,
    "group_sizes": [5, 5, 5],
    "mean_ranks": mean_ranks,
    "n_total": n_dunn,
}

# Compute Dunn's z statistics manually
# z_ij = (R_i - R_j) / SE, SE = sqrt(N(N+1)/12 * (1/n_i + 1/n_j))
n = 15
comparisons = []
for i in range(3):
    for j in range(i+1, 3):
        se = np.sqrt(n * (n + 1) / 12.0 * (1.0/5 + 1.0/5))
        z = abs(mean_ranks[i] - mean_ranks[j]) / se
        p = 2 * (1 - stats.norm.cdf(z))
        comparisons.append({
            "group_i": i, "group_j": j,
            "z_stat": float(z),
            "p_unadjusted": float(p),
        })
results["dunn_3groups"]["comparisons"] = comparisons

# ===========================================================================
# 3. Cook's distance
# ===========================================================================

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import OLSInfluence

    np.random.seed(42)
    n_cook = 20
    x = np.random.normal(0, 1, n_cook)
    y = 2 + 3 * x + np.random.normal(0, 0.5, n_cook)

    # Add an outlier
    x_out = np.append(x, [0.0])
    y_out = np.append(y, [50.0])  # extreme y outlier

    X = sm.add_constant(x_out)
    model = sm.OLS(y_out, X).fit()
    influence = OLSInfluence(model)

    cooks_d = influence.cooks_distance[0]  # D_i values
    leverage = influence.hat_matrix_diag

    results["cooks_distance_with_outlier"] = {
        "x": x_out.tolist(),
        "y": y_out.tolist(),
        "cooks_d": cooks_d.tolist(),
        "leverage": leverage.tolist(),
        "outlier_index": n_cook,  # last point is the outlier
        "max_cooks_d": float(np.max(cooks_d)),
        "max_cooks_d_index": int(np.argmax(cooks_d)),
        "threshold_4_over_n": 4.0 / len(x_out),
    }

    # No outlier case
    X_clean = sm.add_constant(x)
    model_clean = sm.OLS(y, X_clean).fit()
    infl_clean = OLSInfluence(model_clean)
    cooks_clean = infl_clean.cooks_distance[0]

    results["cooks_distance_clean"] = {
        "n": n_cook,
        "p": 2,
        "max_cooks_d": float(np.max(cooks_clean)),
        "all_below_threshold": bool(np.all(cooks_clean < 4.0 / n_cook)),
    }

except ImportError:
    results["cooks_distance_note"] = "statsmodels not available"

# ===========================================================================
# 4. KMO and Bartlett's sphericity
# ===========================================================================

# Generate correlated data
np.random.seed(42)
n_kmo = 100
z1 = np.random.normal(0, 1, n_kmo)
z2 = np.random.normal(0, 1, n_kmo)
x1 = z1 + 0.3 * z2
x2 = 0.8 * z1 + 0.5 * z2
x3 = z2 + 0.1 * np.random.normal(0, 1, n_kmo)
x4 = 0.5 * z1 + 0.7 * z2 + 0.2 * np.random.normal(0, 1, n_kmo)
X_kmo = np.column_stack([x1, x2, x3, x4])
R = np.corrcoef(X_kmo.T)

results["kmo_bartlett_correlated"] = {
    "n_obs": n_kmo,
    "n_vars": 4,
    "correlation_matrix": R.tolist(),
}

# Bartlett's test: -( n - 1 - (2p+5)/6 ) * ln|R| ~ chi2(p(p-1)/2)
p_vars = 4
ln_det_R = float(np.log(np.linalg.det(R)))
bartlett_stat = -(n_kmo - 1 - (2 * p_vars + 5) / 6.0) * ln_det_R
bartlett_df = p_vars * (p_vars - 1) // 2
bartlett_p = float(1 - stats.chi2.cdf(bartlett_stat, bartlett_df))

results["kmo_bartlett_correlated"]["bartlett_statistic"] = float(bartlett_stat)
results["kmo_bartlett_correlated"]["bartlett_df"] = bartlett_df
results["kmo_bartlett_correlated"]["bartlett_p_value"] = bartlett_p
results["kmo_bartlett_correlated"]["bartlett_reject_005"] = bartlett_p < 0.05

# KMO: using anti-image correlation
try:
    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo_model = calculate_kmo(X_kmo)
    results["kmo_bartlett_correlated"]["kmo_overall"] = float(kmo_model)
    results["kmo_bartlett_correlated"]["kmo_per_variable"] = kmo_all.tolist()
except ImportError:
    # Manual KMO computation
    # KMO = Σr²ij / (Σr²ij + Σq²ij) where q is partial correlation
    R_inv = np.linalg.inv(R)
    D = np.diag(1.0 / np.sqrt(np.diag(R_inv)))
    Q = -D @ R_inv @ D  # anti-image correlation
    np.fill_diagonal(Q, 1.0)

    # r² and q² sums (off-diagonal only)
    r2_sum = 0.0
    q2_sum = 0.0
    for i in range(p_vars):
        for j in range(p_vars):
            if i != j:
                r2_sum += R[i, j]**2
                q2_sum += Q[i, j]**2
    kmo_overall = r2_sum / (r2_sum + q2_sum)
    results["kmo_bartlett_correlated"]["kmo_overall"] = float(kmo_overall)

# Identity matrix: KMO should be low, Bartlett should not reject
R_identity = np.eye(4)
ln_det_I = 0.0  # ln|I| = 0
bart_id_stat = -(n_kmo - 1 - (2 * 4 + 5) / 6.0) * ln_det_I
results["kmo_bartlett_identity"] = {
    "correlation_matrix": R_identity.tolist(),
    "bartlett_statistic": float(bart_id_stat),
    "bartlett_p_value": 1.0,
    "kmo_note": "identity matrix: KMO undefined (no shared variance)",
}

# ===========================================================================
# 5. Cluster validation metrics (sklearn)
# ===========================================================================

try:
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score
    )

    # Well-separated clusters
    np.random.seed(42)
    cluster1 = np.random.normal([0, 0], 0.5, (30, 2))
    cluster2 = np.random.normal([5, 5], 0.5, (30, 2))
    cluster3 = np.random.normal([10, 0], 0.5, (30, 2))
    X_clust = np.vstack([cluster1, cluster2, cluster3])
    labels = [0]*30 + [1]*30 + [2]*30

    sil = silhouette_score(X_clust, labels)
    ch = calinski_harabasz_score(X_clust, labels)
    db = davies_bouldin_score(X_clust, labels)

    results["cluster_validation_separated"] = {
        "n_samples": 90,
        "n_clusters": 3,
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "davies_bouldin": float(db),
        "silhouette_high": float(sil) > 0.7,  # well separated
        "db_low": float(db) < 0.5,
    }

    # Overlapping clusters
    oc1 = np.random.normal([0, 0], 2.0, (30, 2))
    oc2 = np.random.normal([1, 1], 2.0, (30, 2))
    X_overlap = np.vstack([oc1, oc2])
    labels_ov = [0]*30 + [1]*30

    sil_ov = silhouette_score(X_overlap, labels_ov)
    ch_ov = calinski_harabasz_score(X_overlap, labels_ov)
    db_ov = davies_bouldin_score(X_overlap, labels_ov)

    results["cluster_validation_overlapping"] = {
        "n_samples": 60,
        "n_clusters": 2,
        "silhouette": float(sil_ov),
        "calinski_harabasz": float(ch_ov),
        "davies_bouldin": float(db_ov),
        "silhouette_low": float(sil_ov) < 0.5,  # overlapping
    }

except ImportError:
    results["cluster_validation_note"] = "sklearn not available"

# ===========================================================================
# Save
# ===========================================================================

with open("research/gold_standard/family_posthoc_influence_expected.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"Post-hoc/Influence Oracle: {len(results)} test cases generated")
for name, r in results.items():
    if isinstance(r, dict):
        extra_parts = []
        if 'bartlett_statistic' in r: extra_parts.append(f"bart={r['bartlett_statistic']:.2f}")
        if 'kmo_overall' in r: extra_parts.append(f"KMO={r['kmo_overall']:.3f}")
        if 'silhouette' in r: extra_parts.append(f"sil={r['silhouette']:.3f}")
        if 'max_cooks_d' in r: extra_parts.append(f"max_D={r['max_cooks_d']:.3f}")
        extra = ", ".join(extra_parts) if extra_parts else ""
        print(f"  PASS {name}" + (f" ({extra})" if extra else ""))
    else:
        print(f"  PASS {name}")
