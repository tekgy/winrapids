"""
Gold Standard Oracle: Family 33 — Multivariate Statistics
Generates exact expected values from scipy/sklearn for tambear validation.

Tests:
  - Correlation matrix (numpy)
  - Hotelling's T² one-sample (manual + scipy.stats.f)
  - Hotelling's T² two-sample
  - LDA classification (sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
  - Mardia's multivariate normality (pingouin-style manual computation)
  - CCA (sklearn.cross_decomposition.CCA)
"""

import json
import numpy as np
from scipy import stats
from scipy.linalg import inv as scipy_inv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA

results = {}

# ═══════════════════════════════════════════════════════════════════════
# 1. Correlation matrix — Gold standard: numpy.corrcoef
# ═══════════════════════════════════════════════════════════════════════

data_3x4 = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 5.0, 7.0],
    [3.0, 5.0, 4.0, 2.0],
])
# numpy computes rowwise, we want columnwise (variables as columns)
# So transpose: 4 observations x 3 variables → nah, let's use a standard dataset
# 5 observations, 3 variables
data_corr = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 7.0],
    [2.0, 3.0, 5.0],
    [5.0, 6.0, 4.0],
])
corr_matrix = np.corrcoef(data_corr, rowvar=False)
results["correlation_matrix"] = {
    "data": data_corr.flatten().tolist(),
    "n": 5, "p": 3,
    "corr": corr_matrix.flatten().tolist(),
}
print(f"Correlation matrix (5x3):\n{corr_matrix}")

# ═══════════════════════════════════════════════════════════════════════
# 2. Hotelling's T² — one-sample
# ═══════════════════════════════════════════════════════════════════════
# H₀: μ = μ₀ = [0, 0]
# Test stat: T² = n * (x̄ - μ₀)' S⁻¹ (x̄ - μ₀)
# F = T² * (n-p) / (p*(n-1)) ~ F(p, n-p)

np.random.seed(42)
n_hot1 = 20
p_hot1 = 2
x_hot1 = np.random.randn(n_hot1, p_hot1) + np.array([1.0, 2.0])  # shifted from 0
mu0_hot1 = np.array([0.0, 0.0])

x_bar = x_hot1.mean(axis=0)
s_cov = np.cov(x_hot1, rowvar=False)
diff = x_bar - mu0_hot1
t2_hot1 = n_hot1 * diff @ scipy_inv(s_cov) @ diff
f_hot1 = t2_hot1 * (n_hot1 - p_hot1) / (p_hot1 * (n_hot1 - 1))
p_val_hot1 = 1 - stats.f.cdf(f_hot1, p_hot1, n_hot1 - p_hot1)

results["hotelling_one_sample"] = {
    "data": x_hot1.flatten().tolist(),
    "n": n_hot1, "p": p_hot1,
    "mu0": mu0_hot1.tolist(),
    "x_bar": x_bar.tolist(),
    "t2": float(t2_hot1),
    "f_statistic": float(f_hot1),
    "df1": p_hot1, "df2": n_hot1 - p_hot1,
    "p_value": float(p_val_hot1),
}
print("Hotelling one-sample: T2=%.6f, F=%.6f, p=%.6e" % (t2_hot1, f_hot1, p_val_hot1))

# ═══════════════════════════════════════════════════════════════════════
# 3. Hotelling's T² — two-sample
# ══════════════════════════════════════════════════════════════���════════

np.random.seed(123)
n1_hot2, n2_hot2, p_hot2 = 15, 15, 2
x1_hot2 = np.random.randn(n1_hot2, p_hot2)
x2_hot2 = np.random.randn(n2_hot2, p_hot2) + np.array([1.5, 1.0])

x1_bar = x1_hot2.mean(axis=0)
x2_bar = x2_hot2.mean(axis=0)
s1 = np.cov(x1_hot2, rowvar=False)
s2 = np.cov(x2_hot2, rowvar=False)
s_pooled = ((n1_hot2 - 1) * s1 + (n2_hot2 - 1) * s2) / (n1_hot2 + n2_hot2 - 2)
diff2 = x1_bar - x2_bar
t2_hot2 = (n1_hot2 * n2_hot2) / (n1_hot2 + n2_hot2) * diff2 @ scipy_inv(s_pooled) @ diff2
n_total = n1_hot2 + n2_hot2
f_hot2 = t2_hot2 * (n_total - p_hot2 - 1) / (p_hot2 * (n_total - 2))
p_val_hot2 = 1 - stats.f.cdf(f_hot2, p_hot2, n_total - p_hot2 - 1)

results["hotelling_two_sample"] = {
    "data_x1": x1_hot2.flatten().tolist(),
    "data_x2": x2_hot2.flatten().tolist(),
    "n1": n1_hot2, "n2": n2_hot2, "p": p_hot2,
    "t2": float(t2_hot2),
    "f_statistic": float(f_hot2),
    "df1": p_hot2, "df2": n_total - p_hot2 - 1,
    "p_value": float(p_val_hot2),
}
print("Hotelling two-sample: T2=%.6f, F=%.6f, p=%.6e" % (t2_hot2, f_hot2, p_val_hot2))

# ═══════════════════════════════════════════════════════════════════════
# 4. LDA — Gold standard: sklearn.discriminant_analysis
# ═══════════════════════════════════════════════════════════════════════

np.random.seed(77)
n_lda = 60
# 3 groups of 20 in 2D, well-separated
x_lda = np.vstack([
    np.random.randn(20, 2) + [0, 0],
    np.random.randn(20, 2) + [5, 0],
    np.random.randn(20, 2) + [2.5, 4],
])
y_lda = np.array([0]*20 + [1]*20 + [2]*20)

clf = LinearDiscriminantAnalysis()
clf.fit(x_lda, y_lda)
predictions = clf.predict(x_lda)
accuracy = np.mean(predictions == y_lda)
# Group means
group_means = np.array([x_lda[y_lda == g].mean(axis=0) for g in range(3)])

results["lda"] = {
    "data": x_lda.flatten().tolist(),
    "labels": y_lda.tolist(),
    "n": n_lda, "p": 2, "n_groups": 3,
    "accuracy": float(accuracy),
    "group_means": group_means.flatten().tolist(),
    "predictions": predictions.tolist(),
}
print(f"\nLDA accuracy: {accuracy:.4f}")
print(f"LDA group means:\n{group_means}")

# ═══════════════════════════════════════════════════════════════════════
# 5. CCA — Gold standard: sklearn.cross_decomposition.CCA
# ═══════════════════════════════════════════════════════════════════════

np.random.seed(55)
n_cca = 30
# X and Y with known correlation structure
z = np.random.randn(n_cca)
x_cca = np.column_stack([z + 0.1*np.random.randn(n_cca), np.random.randn(n_cca)])
y_cca = np.column_stack([z + 0.1*np.random.randn(n_cca), np.random.randn(n_cca)])

cca_model = CCA(n_components=1)
cca_model.fit(x_cca, y_cca)
x_c, y_c = cca_model.transform(x_cca, y_cca)
canon_corr = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]

results["cca"] = {
    "x_data": x_cca.flatten().tolist(),
    "y_data": y_cca.flatten().tolist(),
    "n": n_cca, "px": 2, "py": 2,
    "canonical_correlation": float(canon_corr),
}
print(f"\nCCA canonical correlation: {canon_corr:.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 6. Mardia's multivariate normality test
# ═══════════════════════════════════════════════════════════════════════
# Mardia's skewness: b₁,p = (1/n²) Σᵢ Σⱼ mᵢⱼ³
# where mᵢⱼ = (xᵢ - x̄)' S⁻¹ (xⱼ - x̄)
# Mardia's kurtosis: b₂,p = (1/n) Σᵢ mᵢᵢ²

np.random.seed(42)
n_mardia = 50
p_mardia = 3
x_mardia = np.random.randn(n_mardia, p_mardia)  # standard normal → should be normal

x_bar_m = x_mardia.mean(axis=0)
s_cov_m = np.cov(x_mardia, rowvar=False, ddof=0)  # population cov
s_inv_m = scipy_inv(s_cov_m)

# Mahalanobis-like matrix
centered = x_mardia - x_bar_m
m_mat = centered @ s_inv_m @ centered.T  # n×n matrix of mᵢⱼ

# Mardia skewness
b1p = np.mean(m_mat ** 3)
skew_stat = n_mardia * b1p / 6  # chi-squared with p(p+1)(p+2)/6 df
skew_df = p_mardia * (p_mardia + 1) * (p_mardia + 2) / 6
skew_p = 1 - stats.chi2.cdf(skew_stat, skew_df)

# Mardia kurtosis
b2p = np.mean(np.diag(m_mat) ** 2)
kurt_expected = p_mardia * (p_mardia + 2)
kurt_stat = (b2p - kurt_expected) / np.sqrt(8 * p_mardia * (p_mardia + 2) / n_mardia)
kurt_p = 2 * (1 - stats.norm.cdf(abs(kurt_stat)))

results["mardia_normality"] = {
    "data": x_mardia.flatten().tolist(),
    "n": n_mardia, "p": p_mardia,
    "skewness": float(b1p),
    "skewness_stat": float(skew_stat),
    "skewness_p": float(skew_p),
    "kurtosis": float(b2p),
    "kurtosis_stat": float(kurt_stat),
    "kurtosis_p": float(kurt_p),
}
print("Mardia skewness: b1p=%.6f, stat=%.6f, p=%.6f" % (b1p, skew_stat, skew_p))
print("Mardia kurtosis: b2p=%.6f, stat=%.6f, p=%.6f" % (b2p, kurt_stat, kurt_p))

# ═══════════════════════════════════════════════════════════════════════
# 7. Simple known-answer tests for Hotelling
# ═══════════════════════════════════════════════════════════════════════
# Small hand-computable example: 3 observations, 2 vars, test mu=(0,0)
x_small = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 6.0], [4.0, 1.0], [5.0, 3.0]])
mu0_small = np.array([0.0, 0.0])
n_s, p_s = 5, 2
x_bar_s = x_small.mean(axis=0)  # [2, 3]
s_cov_s = np.cov(x_small, rowvar=False)  # sample cov
diff_s = x_bar_s - mu0_small
t2_s = n_s * diff_s @ scipy_inv(s_cov_s) @ diff_s
f_s = t2_s * (n_s - p_s) / (p_s * (n_s - 1))
p_val_s = 1 - stats.f.cdf(f_s, p_s, n_s - p_s)

results["hotelling_one_sample_small"] = {
    "data": x_small.flatten().tolist(),
    "n": n_s, "p": p_s,
    "mu0": mu0_small.tolist(),
    "x_bar": x_bar_s.tolist(),
    "t2": float(t2_s),
    "f_statistic": float(f_s),
    "df1": p_s, "df2": n_s - p_s,
    "p_value": float(p_val_s),
    "cov_matrix": s_cov_s.flatten().tolist(),
}
print("Small Hotelling: x_bar=%s, T2=%.6f, F=%.6f, p=%.6e" % (x_bar_s, t2_s, f_s, p_val_s))
print(f"Cov matrix:\n{s_cov_s}")

# ═══════════════════════════════════════════════════════════════���═══════
# Save
# ═══════════════════════════════════════════════════════════════════════

with open("research/gold_standard/family_33_expected.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved family_33_expected.json")
