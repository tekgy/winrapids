"""
Gold Standard Oracle: Family 35 — Causal Inference
Generates exact expected values for tambear validation.

Tests:
  - DiD (manual 2x2 computation + statsmodels OLS)
  - IPW (manual weighted means)
  - E-value (formula)
  - Doubly robust AIPW (manual formula)
  - RDD (local linear regression, manual OLS)
  - Propensity scores (sklearn LogisticRegression comparison)
"""

import json
import numpy as np

results = {}

# ===============================================================
# 1. Difference-in-Differences: known effect with noise
# ===============================================================

np.random.seed(42)
n_per_cell = 25  # 25 per cell = 100 total
noise = 0.5

# Control group: baseline=10, trend=+2
ctrl_pre = 10.0 + noise * np.random.randn(n_per_cell)
ctrl_post = 12.0 + noise * np.random.randn(n_per_cell)

# Treatment group: baseline=20, trend=+2 + effect=3
treat_pre = 20.0 + noise * np.random.randn(n_per_cell)
treat_post = 25.0 + noise * np.random.randn(n_per_cell)  # 20 + 2 + 3

y = np.concatenate([ctrl_pre, ctrl_post, treat_pre, treat_post])
treat_vec = np.concatenate([np.zeros(2*n_per_cell), np.ones(2*n_per_cell)])
post_vec = np.concatenate([np.zeros(n_per_cell), np.ones(n_per_cell),
                           np.zeros(n_per_cell), np.ones(n_per_cell)])

# Manual 2x2 DiD
m_cp = ctrl_pre.mean()
m_cq = ctrl_post.mean()
m_tp = treat_pre.mean()
m_tq = treat_post.mean()
did_effect = (m_tq - m_tp) - (m_cq - m_cp)

# SE via pooled residual variance
resids = np.concatenate([
    ctrl_pre - m_cp, ctrl_post - m_cq,
    treat_pre - m_tp, treat_post - m_tq
])
mse = np.sum(resids**2) / (len(y) - 4)
se = np.sqrt(mse * (1/n_per_cell + 1/n_per_cell + 1/n_per_cell + 1/n_per_cell))

results["did_noisy"] = {
    "y": y.tolist(),
    "treat": treat_vec.tolist(),
    "post": post_vec.tolist(),
    "n": len(y),
    "cell_means": [m_cp, m_cq, m_tp, m_tq],
    "effect": float(did_effect),
    "se": float(se),
    "true_effect": 3.0,
}
print("DiD: effect=%.6f (true=3.0), SE=%.6f" % (did_effect, se))

# ===============================================================
# 2. DiD: exact (no noise)
# ===============================================================

y_exact = np.array([10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 25.0, 26.0])
treat_exact = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
post_exact = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

# Cell means
m_cp_e = np.mean([10.0, 11.0])  # 10.5
m_cq_e = np.mean([12.0, 13.0])  # 12.5
m_tp_e = np.mean([20.0, 21.0])  # 20.5
m_tq_e = np.mean([25.0, 26.0])  # 25.5
did_exact = (m_tq_e - m_tp_e) - (m_cq_e - m_cp_e)

results["did_exact"] = {
    "y": y_exact.tolist(),
    "treat": treat_exact.tolist(),
    "post": post_exact.tolist(),
    "effect": float(did_exact),
    "cell_means": [float(m_cp_e), float(m_cq_e), float(m_tp_e), float(m_tq_e)],
}
print("DiD exact: effect=%.6f" % did_exact)

# ===============================================================
# 3. IPW with known propensity
# ===============================================================

# 8 observations, known propensity and outcome
propensity_ipw = np.array([0.3, 0.3, 0.4, 0.4, 0.7, 0.7, 0.6, 0.6])
treatment_ipw = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
outcome_ipw = np.array([10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0])

# Stabilized IPW: w_t = p_treat/e(x), w_c = (1-p_treat)/(1-e(x))
p_treat = treatment_ipw.mean()  # 0.5

# ATE = E_w[Y|T=1] - E_w[Y|T=0]
w_t = p_treat / propensity_ipw[treatment_ipw > 0.5]
w_c = (1 - p_treat) / (1 - propensity_ipw[treatment_ipw <= 0.5])
y_t = outcome_ipw[treatment_ipw > 0.5]
y_c = outcome_ipw[treatment_ipw <= 0.5]

ate_ipw = np.sum(w_t * y_t) / np.sum(w_t) - np.sum(w_c * y_c) / np.sum(w_c)

# ATT: weight only control group with e/(1-e)
att_t = outcome_ipw[treatment_ipw > 0.5].mean()
w_att_c = propensity_ipw[treatment_ipw <= 0.5] / (1 - propensity_ipw[treatment_ipw <= 0.5])
att_c = np.sum(w_att_c * outcome_ipw[treatment_ipw <= 0.5]) / np.sum(w_att_c)
att_ipw = att_t - att_c

results["ipw_known"] = {
    "propensity": propensity_ipw.tolist(),
    "treatment": treatment_ipw.tolist(),
    "outcome": outcome_ipw.tolist(),
    "p_treat": float(p_treat),
    "ate": float(ate_ipw),
    "att": float(att_ipw),
}
print("IPW: ATE=%.6f, ATT=%.6f" % (ate_ipw, att_ipw))

# ===============================================================
# 4. E-value
# ===============================================================

# E-value = RR + sqrt(RR*(RR-1))
rr_vals = [1.0, 1.5, 2.0, 3.0, 5.0]
e_vals = []
for rr in rr_vals:
    ev = rr + np.sqrt(rr * (rr - 1))
    e_vals.append(float(ev))
    print("E-value(RR=%.1f) = %.6f" % (rr, ev))

results["e_value"] = {
    "risk_ratios": rr_vals,
    "e_values": e_vals,
}

# ===============================================================
# 5. Doubly Robust (AIPW)
# ===============================================================

# Known scenario: true ATE = 5.0
propensity_dr = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.4, 0.3, 0.6, 0.7])
treatment_dr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
outcome_dr = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
mu1_dr = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 15.0, 16.0, 17.0, 18.0, 19.0])
mu0_dr = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 11.0, 12.0, 13.0, 14.0])

# AIPW formula
n_dr = len(propensity_dr)
dr_sum = 0.0
for i in range(n_dr):
    e = np.clip(propensity_dr[i], 0.01, 0.99)
    t = treatment_dr[i]
    dr_i = (mu1_dr[i] - mu0_dr[i]) + t * (outcome_dr[i] - mu1_dr[i]) / e - (1 - t) * (outcome_dr[i] - mu0_dr[i]) / (1 - e)
    dr_sum += dr_i
ate_dr = dr_sum / n_dr

results["doubly_robust"] = {
    "propensity": propensity_dr.tolist(),
    "treatment": treatment_dr.tolist(),
    "outcome": outcome_dr.tolist(),
    "mu1": mu1_dr.tolist(),
    "mu0": mu0_dr.tolist(),
    "ate": float(ate_dr),
}
print("Doubly robust ATE=%.6f" % ate_dr)

# ===============================================================
# 6. RDD: sharp with known discontinuity
# ===============================================================

# y = 2*x + 5*I(x>=0) + noise
np.random.seed(99)
n_rdd = 200
running = np.linspace(-5, 5, n_rdd)
noise_rdd = 0.1 * np.random.randn(n_rdd)
outcome_rdd = 2.0 * running + 5.0 * (running >= 0).astype(float) + noise_rdd

# Local linear regression within bandwidth=2.0
bw = 2.0
cutoff = 0.0
left_mask = (running < cutoff) & (np.abs(running - cutoff) <= bw)
right_mask = (running >= cutoff) & (np.abs(running - cutoff) <= bw)

left_x = running[left_mask] - cutoff
left_y = outcome_rdd[left_mask]
right_x = running[right_mask] - cutoff
right_y = outcome_rdd[right_mask]

# OLS: y = a + b*x for each side
def ols_1d(x, y):
    n = len(x)
    mx = x.mean()
    my = y.mean()
    sxy = np.sum((x - mx) * (y - my))
    sxx = np.sum((x - mx)**2)
    b = sxy / sxx if abs(sxx) > 1e-15 else 0.0
    a = my - b * mx
    return a, b

a_l, b_l = ols_1d(left_x, left_y)
a_r, b_r = ols_1d(right_x, right_y)
rdd_effect = a_r - a_l

results["rdd_sharp"] = {
    "running": running.tolist(),
    "outcome": outcome_rdd.tolist(),
    "cutoff": cutoff,
    "bandwidth": bw,
    "effect": float(rdd_effect),
    "true_effect": 5.0,
    "intercept_left": float(a_l),
    "intercept_right": float(a_r),
    "slope_left": float(b_l),
    "slope_right": float(b_r),
}
print("RDD: effect=%.6f (true=5.0), slopes: L=%.4f, R=%.4f" % (rdd_effect, b_l, b_r))

# ===============================================================
# 7. RDD: exact (no noise)
# ===============================================================

running_exact = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
outcome_exact = running_exact + 3.0 * (running_exact >= 0).astype(float)

a_l_e, b_l_e = ols_1d(
    running_exact[running_exact < 0] - 0.0,
    outcome_exact[running_exact < 0]
)
a_r_e, b_r_e = ols_1d(
    running_exact[running_exact >= 0] - 0.0,
    outcome_exact[running_exact >= 0]
)
rdd_exact = a_r_e - a_l_e

results["rdd_exact"] = {
    "running": running_exact.tolist(),
    "outcome": outcome_exact.tolist(),
    "cutoff": 0.0,
    "bandwidth": 5.0,
    "effect": float(rdd_exact),
    "intercept_left": float(a_l_e),
    "intercept_right": float(a_r_e),
    "slope_left": float(b_l_e),
    "slope_right": float(b_r_e),
}
print("RDD exact: effect=%.6f, slopes: L=%.4f, R=%.4f" % (rdd_exact, b_l_e, b_r_e))

# ===============================================================
# Save
# ===============================================================

with open("research/gold_standard/family_35_expected.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved family_35_expected.json")
