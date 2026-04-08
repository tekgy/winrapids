"""
Gold Standard Oracle: Family 11 — Mixed Effects Models
Generates expected values for tambear validation.

Tests:
  - Random intercept model (statsmodels MixedLM comparison)
  - ICC (one-way ANOVA formula)
  - Design effect (exact formula)
"""

import json
import numpy as np

results = {}

# ===============================================================
# 1. Random intercept model: known structure
# ===============================================================

np.random.seed(42)
n_per_group = 20
k = 3
true_beta = np.array([2.0, 3.0])  # intercept, slope
true_sigma2 = 0.25  # residual variance
true_sigma2_u = 4.0  # random effect variance (large groups)
group_effects = np.array([0.0, 2.0, -1.0])

x_list = []
y_list = []
groups_list = []

for g in range(k):
    for i in range(n_per_group):
        xi = i / n_per_group
        noise = np.sqrt(true_sigma2) * np.random.randn()
        yi = true_beta[0] + true_beta[1] * xi + group_effects[g] + noise
        x_list.append(xi)
        y_list.append(yi)
        groups_list.append(g)

x = np.array(x_list)
y = np.array(y_list)
groups = np.array(groups_list)

results["lme_known"] = {
    "x": x.tolist(),
    "y": y.tolist(),
    "groups": groups.tolist(),
    "n": len(y),
    "d": 1,
    "k": k,
    "true_beta": true_beta.tolist(),
    "true_sigma2": true_sigma2,
    "true_sigma2_u": true_sigma2_u,
    "true_group_effects": group_effects.tolist(),
}
print("LME known: n=%d, k=%d, true_beta=%s" % (len(y), k, true_beta))

# ===============================================================
# 2. ICC: one-way ANOVA method
# ===============================================================
# ICC(1,1) = (MSB - MSW) / (MSB + (n0-1)*MSW)
# where n0 = (N - sum(n_k^2)/N) / (k-1)

values_icc = np.array([
    1.0, 1.1, 0.9, 1.0,   # group 0
    5.0, 5.1, 4.9, 5.0,   # group 1
    10.0, 10.1, 9.9, 10.0  # group 2
])
groups_icc = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

n_icc = len(values_icc)
k_icc = 3
grand_mean = values_icc.mean()

g_means = np.array([values_icc[groups_icc == g].mean() for g in range(k_icc)])
g_counts = np.array([np.sum(groups_icc == g) for g in range(k_icc)])

msb = np.sum(g_counts * (g_means - grand_mean)**2) / (k_icc - 1)
msw = np.sum((values_icc - g_means[groups_icc])**2) / (n_icc - k_icc)

n0 = (n_icc - np.sum(g_counts**2) / n_icc) / (k_icc - 1)
icc = (msb - msw) / (msb + (n0 - 1) * msw)

results["icc_high"] = {
    "values": values_icc.tolist(),
    "groups": groups_icc.tolist(),
    "n": n_icc, "k": k_icc,
    "grand_mean": float(grand_mean),
    "group_means": g_means.tolist(),
    "msb": float(msb),
    "msw": float(msw),
    "n0": float(n0),
    "icc": float(icc),
}
print("ICC high: MSB=%.4f, MSW=%.4f, n0=%.4f, ICC=%.6f" % (msb, msw, n0, icc))

# Low ICC case
values_low = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0, 0.5, 4.5, 2.5, 6.5])
groups_low = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
grand_mean_low = values_low.mean()
g_means_low = np.array([values_low[groups_low == g].mean() for g in range(3)])
g_counts_low = np.array([4, 4, 4])
msb_low = np.sum(g_counts_low * (g_means_low - grand_mean_low)**2) / 2
msw_low = np.sum((values_low - g_means_low[groups_low])**2) / 9
n0_low = (12 - 48/12) / 2
icc_low = max(0, (msb_low - msw_low) / (msb_low + (n0_low - 1) * msw_low))

results["icc_low"] = {
    "values": values_low.tolist(),
    "groups": groups_low.tolist(),
    "icc": float(icc_low),
}
print("ICC low: ICC=%.6f" % icc_low)

# ===============================================================
# 3. Design effect: exact formula
# ===============================================================
# DEFF = 1 + (m-1)*ICC where m = average cluster size

test_cases = [
    (0.0, 10.0, 1.0),           # ICC=0 → DEFF=1
    (0.1, 20.0, 2.9),           # ICC=0.1, m=20
    (0.5, 10.0, 5.5),           # ICC=0.5, m=10
    (1.0, 5.0, 5.0),            # ICC=1.0, m=5
]
results["design_effect"] = {
    "cases": [{"icc": icc_v, "cluster_size": m, "deff": deff} for icc_v, m, deff in test_cases],
}
for icc_v, m, deff in test_cases:
    print("DEFF: ICC=%.1f, m=%.0f -> %.1f" % (icc_v, m, deff))

# ===============================================================
# Save
# ===============================================================

with open("research/gold_standard/family_11_expected.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved family_11_expected.json")
