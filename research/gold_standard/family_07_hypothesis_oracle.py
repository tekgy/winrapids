"""
Gold standard oracle for Family 07: Hypothesis Testing
Compares tambear hypothesis.rs against scipy.stats.

Every value here becomes a hardcoded expected in Rust tests.
"""
import json
import numpy as np
from scipy import stats

# Deterministic seed
rng = np.random.default_rng(42)

results = {}

# ─── Test data ───────────────────────────────────────────────────────
group1 = rng.normal(loc=5.0, scale=2.0, size=30).tolist()
group2 = rng.normal(loc=6.5, scale=2.5, size=35).tolist()
arr1 = np.array(group1)
arr2 = np.array(group2)

# ─── 1. One-sample t-test ────────────────────────────────────────────
t_stat, p_val = stats.ttest_1samp(arr1, popmean=5.0)
n = len(arr1)
d = (np.mean(arr1) - 5.0) / np.std(arr1, ddof=1)  # Cohen's d
results["one_sample_t"] = {
    "data": group1,
    "mu": 5.0,
    "t_statistic": float(t_stat),
    "p_value": float(p_val),
    "df": float(n - 1),
    "cohens_d": float(d),
    "mean": float(np.mean(arr1)),
    "std_sample": float(np.std(arr1, ddof=1)),
}

# ─── 2. Two-sample t-test (equal variance) ───────────────────────────
t_stat, p_val = stats.ttest_ind(arr1, arr2, equal_var=True)
n1, n2 = len(arr1), len(arr2)
# Pooled std
sp = np.sqrt(((n1-1)*np.var(arr1, ddof=1) + (n2-1)*np.var(arr2, ddof=1)) / (n1+n2-2))
d = (np.mean(arr1) - np.mean(arr2)) / sp
results["two_sample_t"] = {
    "data1": group1,
    "data2": group2,
    "t_statistic": float(t_stat),
    "p_value": float(p_val),
    "df": float(n1 + n2 - 2),
    "cohens_d": float(d),
}

# ─── 3. Welch's t-test (unequal variance) ────────────────────────────
t_stat, p_val = stats.ttest_ind(arr1, arr2, equal_var=False)
# Welch-Satterthwaite df
s1sq, s2sq = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
num = (s1sq/n1 + s2sq/n2)**2
den = (s1sq/n1)**2/(n1-1) + (s2sq/n2)**2/(n2-1)
welch_df = num / den
results["welch_t"] = {
    "data1": group1,
    "data2": group2,
    "t_statistic": float(t_stat),
    "p_value": float(p_val),
    "df": float(welch_df),
}

# ─── 4. Paired t-test ────────────────────────────────────────────────
# Need equal-length arrays
paired_a = rng.normal(loc=10.0, scale=3.0, size=25).tolist()
paired_b = (np.array(paired_a) + rng.normal(loc=1.5, scale=1.0, size=25)).tolist()
arr_pa = np.array(paired_a)
arr_pb = np.array(paired_b)
t_stat, p_val = stats.ttest_rel(arr_pa, arr_pb)
diffs = arr_pb - arr_pa
d_paired = np.mean(diffs) / np.std(diffs, ddof=1)
results["paired_t"] = {
    "data_a": paired_a,
    "data_b": paired_b,
    "t_statistic": float(t_stat),
    "p_value": float(p_val),
    "df": float(len(paired_a) - 1),
    "mean_diff": float(np.mean(diffs)),
    "std_diff": float(np.std(diffs, ddof=1)),
}

# ─── 5. One-way ANOVA ────────────────────────────────────────────────
group_a = rng.normal(loc=5.0, scale=1.5, size=20).tolist()
group_b = rng.normal(loc=6.0, scale=1.5, size=25).tolist()
group_c = rng.normal(loc=7.5, scale=1.5, size=22).tolist()
arr_a, arr_b, arr_c = np.array(group_a), np.array(group_b), np.array(group_c)

f_stat, p_val = stats.f_oneway(arr_a, arr_b, arr_c)

# Compute ANOVA table manually
all_data = np.concatenate([arr_a, arr_b, arr_c])
grand_mean = np.mean(all_data)
N = len(all_data)
k = 3

ss_between = (len(arr_a) * (np.mean(arr_a) - grand_mean)**2 +
              len(arr_b) * (np.mean(arr_b) - grand_mean)**2 +
              len(arr_c) * (np.mean(arr_c) - grand_mean)**2)
ss_within = (np.sum((arr_a - np.mean(arr_a))**2) +
             np.sum((arr_b - np.mean(arr_b))**2) +
             np.sum((arr_c - np.mean(arr_c))**2))
ss_total = np.sum((all_data - grand_mean)**2)

df_between = k - 1
df_within = N - k
ms_between = ss_between / df_between
ms_within = ss_within / df_within
eta_sq = ss_between / ss_total
omega_sq = (ss_between - (k-1)*ms_within) / (ss_total + ms_within)

results["one_way_anova"] = {
    "group_a": group_a,
    "group_b": group_b,
    "group_c": group_c,
    "f_statistic": float(f_stat),
    "p_value": float(p_val),
    "df_between": float(df_between),
    "df_within": float(df_within),
    "ss_between": float(ss_between),
    "ss_within": float(ss_within),
    "ss_total": float(ss_total),
    "ms_between": float(ms_between),
    "ms_within": float(ms_within),
    "eta_squared": float(eta_sq),
    "omega_squared": float(omega_sq),
}

# ─── 6. Chi-square goodness of fit ───────────────────────────────────
observed = [16.0, 18.0, 22.0, 14.0, 30.0]
expected = [20.0, 20.0, 20.0, 20.0, 20.0]
chi2, p_val = stats.chisquare(observed, expected)
results["chi2_gof"] = {
    "observed": observed,
    "expected": expected,
    "statistic": float(chi2),
    "p_value": float(p_val),
    "df": float(len(observed) - 1),
}

# ─── 7. Chi-square independence ───────────────────────────────────────
# 2x3 contingency table
table = np.array([[10, 20, 30],
                  [6, 9, 17]])
chi2, p_val, dof, expected_freq = stats.chi2_contingency(table, correction=False)
n_obs = table.sum()
min_dim = min(table.shape) - 1
cramers_v = np.sqrt(chi2 / (n_obs * min_dim))
results["chi2_independence"] = {
    "table": table.flatten().tolist(),
    "n_rows": 2,
    "n_cols": 3,
    "statistic": float(chi2),
    "p_value": float(p_val),
    "df": float(dof),
    "cramers_v": float(cramers_v),
}

# ─── 8. Effect sizes ─────────────────────────────────────────────────
# Cohen's d (already computed in two_sample_t, but let's also compute standalone)
sp_standalone = np.sqrt(((n1-1)*np.var(arr1, ddof=1) + (n2-1)*np.var(arr2, ddof=1)) / (n1+n2-2))
cohens_d = float((np.mean(arr1) - np.mean(arr2)) / sp_standalone)

# Hedges' g (bias-corrected)
hedges_g = cohens_d * (1 - 3/(4*(n1+n2) - 9))

# Glass's delta (using group2 as control)
glass_delta = float((np.mean(arr1) - np.mean(arr2)) / np.std(arr2, ddof=1))

# Point-biserial r
rpb = cohens_d / np.sqrt(cohens_d**2 + (n1+n2)**2/(n1*n2))

results["effect_sizes"] = {
    "data1": group1,
    "data2": group2,
    "cohens_d": cohens_d,
    "hedges_g": float(hedges_g),
    "glass_delta": glass_delta,
    "point_biserial_r": float(rpb),
}

# ─── 9. Odds ratio ───────────────────────────────────────────────────
table_2x2 = [30.0, 10.0, 20.0, 40.0]  # [a, b, c, d]
a, b, c, d = table_2x2
or_val = (a * d) / (b * c)
log_or = np.log(or_val)
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
results["odds_ratio"] = {
    "table": table_2x2,
    "odds_ratio": float(or_val),
    "log_odds_ratio": float(log_or),
    "log_odds_ratio_se": float(se_log_or),
}

# ─── 10. Multiple testing corrections ────────────────────────────────
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.013, 0.029, 0.04, 0.05, 0.07, 0.10, 0.20, 0.50, 0.80]

_, bonf, _, _ = multipletests(p_values, method='bonferroni')
_, holm, _, _ = multipletests(p_values, method='holm')
_, bh, _, _ = multipletests(p_values, method='fdr_bh')

results["multiple_testing"] = {
    "p_values": p_values,
    "bonferroni": [float(x) for x in bonf],
    "holm": [float(x) for x in holm],
    "benjamini_hochberg": [float(x) for x in bh],
}

# ─── 11. Proportion z-tests ──────────────────────────────────────────
# One-proportion z-test: 60 successes out of 100, test against p0=0.5
from statsmodels.stats.proportion import proportions_ztest
z_stat, p_val = proportions_ztest(60, 100, value=0.5)
# Cohen's h = 2*arcsin(sqrt(p)) - 2*arcsin(sqrt(p0))
p_hat = 60/100
p0 = 0.5
cohens_h_1prop = 2*np.arcsin(np.sqrt(p_hat)) - 2*np.arcsin(np.sqrt(p0))
results["one_proportion_z"] = {
    "successes": 60.0,
    "n": 100.0,
    "p0": 0.5,
    "z_statistic": float(z_stat),
    "p_value": float(p_val),
    "cohens_h": float(cohens_h_1prop),
}

# Two-proportion z-test: 45/100 vs 60/120
z_stat, p_val = proportions_ztest([45, 60], [100, 120])
p1, p2 = 45/100, 60/120
cohens_h_2prop = 2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2))
results["two_proportion_z"] = {
    "successes1": 45.0,
    "n1": 100.0,
    "successes2": 60.0,
    "n2": 120.0,
    "z_statistic": float(z_stat),
    "p_value": float(p_val),
    "cohens_h": float(cohens_h_2prop),
}

# ─── Write output ────────────────────────────────────────────────────
with open("R:/winrapids/research/gold_standard/family_07_hypothesis_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote {len(results)} test cases")
for k, v in results.items():
    vals = {kk: vv for kk, vv in v.items()
            if kk not in ("data", "data1", "data2", "data_a", "data_b",
                          "group_a", "group_b", "group_c", "sorted")}
    print(f"  {k}: {vals}")
