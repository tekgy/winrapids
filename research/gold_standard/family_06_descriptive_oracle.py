"""
Gold standard oracle for Family 06: Descriptive Statistics
Compares tambear descriptive.rs against scipy, numpy.

Every value here becomes a hardcoded expected in Rust tests.
"""
import json
import numpy as np
from scipy import stats

# Deterministic seed
rng = np.random.default_rng(42)

# ─── Test data ───────────────────────────────────────────────────────
data = rng.normal(loc=5.0, scale=2.0, size=50).tolist()
arr = np.array(data)

results = {}

# ─── 1. Basic moments ────────────────────────────────────────────────
results["basic_moments"] = {
    "data": data,
    "n": len(data),
    "mean": float(np.mean(arr)),
    "var_pop": float(np.var(arr, ddof=0)),
    "var_sample": float(np.var(arr, ddof=1)),
    "std_pop": float(np.std(arr, ddof=0)),
    "std_sample": float(np.std(arr, ddof=1)),
    "min": float(np.min(arr)),
    "max": float(np.max(arr)),
    "sum": float(np.sum(arr)),
    "range": float(np.ptp(arr)),
}

# ─── 2. Skewness ─────────────────────────────────────────────────────
results["skewness"] = {
    "data": data,
    "biased": float(stats.skew(arr, bias=True)),
    "unbiased": float(stats.skew(arr, bias=False)),
}

# ─── 3. Kurtosis ─────────────────────────────────────────────────────
results["kurtosis"] = {
    "data": data,
    "excess_biased": float(stats.kurtosis(arr, bias=True, fisher=True)),
    "excess_unbiased": float(stats.kurtosis(arr, bias=False, fisher=True)),
    "raw_biased": float(stats.kurtosis(arr, bias=True, fisher=False)),
}

# ─── 4. Quantiles (linear = numpy default) ───────────────────────────
sorted_data = sorted(data)
results["quantiles"] = {
    "data": data,
    "sorted": sorted_data,
    "median": float(np.median(arr)),
    "q25_linear": float(np.quantile(arr, 0.25, method='linear')),
    "q75_linear": float(np.quantile(arr, 0.75, method='linear')),
    "q10_linear": float(np.quantile(arr, 0.10, method='linear')),
    "q90_linear": float(np.quantile(arr, 0.90, method='linear')),
    "iqr": float(stats.iqr(arr)),
}

# ─── 5. Quantile methods (R types) ───────────────────────────────────
# numpy methods map to R types:
# inverted_cdf = R type 1, linear (type 4 in R) = method='interpolated_inverted_cdf'
# hazen (type 5) = method='hazen', weibull (type 6) = method='weibull'
# linear (type 7) = method='linear', median_unbiased (type 8) = method='median_unbiased'
# normal_unbiased (type 9) = method='normal_unbiased'
results["quantile_methods"] = {
    "data": data,
    "q": 0.25,
    "inverted_cdf": float(np.quantile(arr, 0.25, method='inverted_cdf')),        # R type 1
    "hazen": float(np.quantile(arr, 0.25, method='hazen')),                       # R type 5
    "weibull": float(np.quantile(arr, 0.25, method='weibull')),                   # R type 6
    "linear": float(np.quantile(arr, 0.25, method='linear')),                     # R type 7
    "median_unbiased": float(np.quantile(arr, 0.25, method='median_unbiased')),   # R type 8
    "normal_unbiased": float(np.quantile(arr, 0.25, method='normal_unbiased')),   # R type 9
}

# ─── 6. Central tendency variants ────────────────────────────────────
positive_data = np.abs(arr) + 0.1  # ensure positive for geo/harmonic mean
results["central_tendency"] = {
    "data": positive_data.tolist(),
    "geometric_mean": float(stats.gmean(positive_data)),
    "harmonic_mean": float(stats.hmean(positive_data)),
    "trimmed_mean_10": float(stats.trim_mean(arr, 0.10)),   # trim 10% each side
    "trimmed_mean_25": float(stats.trim_mean(arr, 0.25)),   # trim 25% each side
}

# ─── 7. Winsorized mean ──────────────────────────────────────────────
from scipy.stats.mstats import winsorize
wins_10 = winsorize(arr, limits=[0.10, 0.10])
wins_25 = winsorize(arr, limits=[0.25, 0.25])
results["winsorized"] = {
    "data": data,
    "winsorized_mean_10": float(np.mean(wins_10)),
    "winsorized_mean_25": float(np.mean(wins_25)),
}

# ─── 8. MAD (unscaled, consistent with scipy default) ────────────────
results["mad"] = {
    "data": data,
    "mad_unscaled": float(stats.median_abs_deviation(arr, scale=1.0)),
}

# ─── 9. Gini coefficient ─────────────────────────────────────────────
# Manual computation: Gini = (Σ Σ |xᵢ - xⱼ|) / (2 * n² * mean)
pos_sorted = np.sort(positive_data)
n = len(pos_sorted)
gini_num = 0.0
for i in range(n):
    for j in range(n):
        gini_num += abs(pos_sorted[i] - pos_sorted[j])
gini = gini_num / (2 * n * n * np.mean(pos_sorted))
results["gini"] = {
    "data": pos_sorted.tolist(),
    "gini": float(gini),
}

# ─── 10. MomentStats merge parity ────────────────────────────────────
# Split data in two, verify merged stats equal full-data stats
split = 20
part_a = arr[:split]
part_b = arr[split:]
results["merge_parity"] = {
    "data": data,
    "split_at": split,
    "full_mean": float(np.mean(arr)),
    "full_var_pop": float(np.var(arr, ddof=0)),
    "full_var_sample": float(np.var(arr, ddof=1)),
    "full_skew_biased": float(stats.skew(arr, bias=True)),
    "full_kurt_excess_biased": float(stats.kurtosis(arr, bias=True, fisher=True)),
    "part_a_mean": float(np.mean(part_a)),
    "part_a_var_pop": float(np.var(part_a, ddof=0)),
    "part_b_mean": float(np.mean(part_b)),
    "part_b_var_pop": float(np.var(part_b, ddof=0)),
}

# ─── 11. SEM (standard error of mean) ────────────────────────────────
results["sem"] = {
    "data": data,
    "sem": float(stats.sem(arr, ddof=1)),
}

# ─── 12. CV (coefficient of variation) ───────────────────────────────
results["cv"] = {
    "data": data,
    "cv": float(stats.variation(arr, ddof=1)),
}

# ─── 13. Bowley skewness ─────────────────────────────────────────────
q1 = float(np.quantile(arr, 0.25, method='linear'))
q2 = float(np.quantile(arr, 0.50, method='linear'))
q3 = float(np.quantile(arr, 0.75, method='linear'))
bowley = (q3 + q1 - 2*q2) / (q3 - q1) if q3 != q1 else 0.0
results["bowley_skewness"] = {
    "data": data,
    "bowley": bowley,
    "q1": q1,
    "q2": q2,
    "q3": q3,
}

# ─── 14. Pearson first skewness ──────────────────────────────────────
pfs = 3.0 * (float(np.mean(arr)) - float(np.median(arr))) / float(np.std(arr, ddof=1))
results["pearson_first_skewness"] = {
    "data": data,
    "pearson_first": pfs,
}

# ─── Write output ────────────────────────────────────────────────────
with open("R:/winrapids/research/gold_standard/family_06_descriptive_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote {len(results)} test cases")
for k, v in results.items():
    vals = {kk: vv for kk, vv in v.items() if kk != "data" and kk != "sorted"}
    print(f"  {k}: {vals}")
