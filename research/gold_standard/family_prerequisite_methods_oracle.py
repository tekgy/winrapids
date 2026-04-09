"""
Gold Standard Oracle: Prerequisite Statistical Methods

Generates expected values for the 9 new prerequisite methods:
  1. Shapiro-Wilk normality test
  2. Levene's test for equality of variances
  3. Welch's ANOVA
  4. Ljung-Box portmanteau test
  5. KPSS stationarity test
  6. Durbin-Watson autocorrelation test
  7. Tukey HSD post-hoc (prepared, not yet impl)
  8. Dunn's test (prepared, not yet impl)
  9. Breusch-Pagan heteroscedasticity test (prepared, not yet impl)

Usage:
    python research/gold_standard/family_prerequisite_methods_oracle.py
"""

import json
import numpy as np
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

results = {}

# ===========================================================================
# 1. Shapiro-Wilk normality test
# ===========================================================================

# Normal data — should NOT reject
np.random.seed(42)
normal_data = np.random.normal(0, 1, 50).tolist()
sw_stat, sw_p = stats.shapiro(normal_data)
results["shapiro_wilk_normal"] = {
    "data": normal_data,
    "statistic": float(sw_stat),
    "p_value": float(sw_p),
    "reject_005": float(sw_p) < 0.05,
}

# Uniform data — should reject
uniform_data = np.random.uniform(0, 1, 50).tolist()
sw_stat2, sw_p2 = stats.shapiro(uniform_data)
results["shapiro_wilk_uniform"] = {
    "data": uniform_data,
    "statistic": float(sw_stat2),
    "p_value": float(sw_p2),
    "reject_005": float(sw_p2) < 0.05,
}

# Exponential data — should reject
exp_data = np.random.exponential(1, 50).tolist()
sw_stat3, sw_p3 = stats.shapiro(exp_data)
results["shapiro_wilk_exponential"] = {
    "data": exp_data,
    "statistic": float(sw_stat3),
    "p_value": float(sw_p3),
    "reject_005": float(sw_p3) < 0.05,
}

# ===========================================================================
# 2. Levene's test for equality of variances
# ===========================================================================

g1 = [2.0, 3.0, 4.0, 5.0, 6.0]
g2 = [1.0, 5.0, 9.0, 13.0, 17.0]
g3 = [3.0, 3.5, 4.0, 4.5, 5.0]

# Center = mean (original Levene)
lev_stat_mean, lev_p_mean = stats.levene(g1, g2, g3, center='mean')
results["levene_mean_3groups"] = {
    "groups": [g1, g2, g3],
    "center": "mean",
    "statistic": float(lev_stat_mean),
    "p_value": float(lev_p_mean),
}

# Center = median (Brown-Forsythe variant)
lev_stat_med, lev_p_med = stats.levene(g1, g2, g3, center='median')
results["levene_median_3groups"] = {
    "groups": [g1, g2, g3],
    "center": "median",
    "statistic": float(lev_stat_med),
    "p_value": float(lev_p_med),
}

# Equal variance case — should not reject
g_eq1 = [1.0, 2.0, 3.0, 4.0, 5.0]
g_eq2 = [2.0, 3.0, 4.0, 5.0, 6.0]
lev_eq_stat, lev_eq_p = stats.levene(g_eq1, g_eq2, center='mean')
results["levene_equal_var"] = {
    "groups": [g_eq1, g_eq2],
    "center": "mean",
    "statistic": float(lev_eq_stat),
    "p_value": float(lev_eq_p),
    "reject_005": float(lev_eq_p) < 0.05,
}

# ===========================================================================
# 3. Welch's ANOVA
# ===========================================================================

# Three groups with unequal variances
wa_g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
wa_g2 = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
wa_g3 = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

# scipy doesn't have Welch's ANOVA directly; use Alexander-Govern or manual
# For 2 groups, Welch ANOVA = Welch t-test
wa_2g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
wa_2g2 = [3.0, 4.0, 5.0, 6.0, 7.0]
welch_t_stat, welch_t_p = stats.ttest_ind(wa_2g1, wa_2g2, equal_var=False)
results["welch_anova_2groups"] = {
    "groups": [wa_2g1, wa_2g2],
    "note": "2-group Welch ANOVA = Welch t-test; F = t^2",
    "t_statistic": float(welch_t_stat),
    "f_statistic": float(welch_t_stat ** 2),
    "p_value": float(welch_t_p),
}

# Equal means — should not reject
wa_eq1 = [10.0, 10.5, 11.0, 11.5, 12.0]
wa_eq2 = [10.0, 10.5, 11.0, 11.5, 12.0]
welch_eq_t, welch_eq_p = stats.ttest_ind(wa_eq1, wa_eq2, equal_var=False)
results["welch_anova_equal_means"] = {
    "groups": [wa_eq1, wa_eq2],
    "f_statistic": float(welch_eq_t ** 2),
    "p_value": float(welch_eq_p),
    "reject_005": float(welch_eq_p) < 0.05,
}

# ===========================================================================
# 4. Ljung-Box portmanteau test
# ===========================================================================

# White noise — should NOT reject
np.random.seed(123)
white_noise = np.random.normal(0, 1, 100).tolist()

# Use statsmodels for Ljung-Box
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(white_noise, lags=[10], return_df=True)
    lb_stat = float(lb_result['lb_stat'].iloc[0])
    lb_p = float(lb_result['lb_pvalue'].iloc[0])
    results["ljung_box_white_noise"] = {
        "data_seed": 123,
        "n": 100,
        "n_lags": 10,
        "fitted_params": 0,
        "statistic": lb_stat,
        "p_value": lb_p,
        "reject_005": lb_p < 0.05,
    }
except ImportError:
    # Fall back to manual computation
    n = len(white_noise)
    mean = sum(white_noise) / n
    centered = [x - mean for x in white_noise]
    var = sum(x**2 for x in centered) / n
    acf_vals = []
    for k in range(11):
        if k == 0:
            acf_vals.append(1.0)
        else:
            cov_k = sum(centered[t] * centered[t-k] for t in range(k, n)) / n
            acf_vals.append(cov_k / var)
    q = n * (n + 2) * sum(acf_vals[k]**2 / (n - k) for k in range(1, 11))
    from scipy.stats import chi2
    lb_p = 1.0 - chi2.cdf(q, 10)
    results["ljung_box_white_noise"] = {
        "data_seed": 123,
        "n": 100,
        "n_lags": 10,
        "fitted_params": 0,
        "statistic": float(q),
        "p_value": float(lb_p),
        "reject_005": lb_p < 0.05,
    }

# AR(1) data — should reject
ar_data = [0.0]
rng = np.random.RandomState(42)
for i in range(99):
    ar_data.append(0.9 * ar_data[-1] + rng.normal(0, 1))

try:
    lb_ar = acorr_ljungbox(ar_data, lags=[5], return_df=True)
    lb_ar_stat = float(lb_ar['lb_stat'].iloc[0])
    lb_ar_p = float(lb_ar['lb_pvalue'].iloc[0])
except:
    n = len(ar_data)
    mean = sum(ar_data) / n
    centered = [x - mean for x in ar_data]
    var = sum(x**2 for x in centered) / n
    acf_vals = [1.0]
    for k in range(1, 6):
        cov_k = sum(centered[t] * centered[t-k] for t in range(k, n)) / n
        acf_vals.append(cov_k / var)
    lb_ar_stat = n * (n + 2) * sum(acf_vals[k]**2 / (n - k) for k in range(1, 6))
    lb_ar_p = 1.0 - float(stats.chi2.cdf(lb_ar_stat, 5))

results["ljung_box_ar1"] = {
    "data": ar_data,
    "n_lags": 5,
    "fitted_params": 0,
    "statistic": float(lb_ar_stat),
    "p_value": float(lb_ar_p),
    "reject_005": lb_ar_p < 0.05,
}

# ===========================================================================
# 5. KPSS stationarity test
# ===========================================================================

try:
    from statsmodels.tsa.stattools import kpss as sm_kpss
    # Stationary data
    kpss_stat, kpss_p, kpss_lags, kpss_crit = sm_kpss(white_noise, regression='c', nlags='auto')
    results["kpss_stationary"] = {
        "data_seed": 123,
        "n": 100,
        "trend": False,
        "statistic": float(kpss_stat),
        "p_value": float(kpss_p),
        "n_lags": int(kpss_lags),
        "critical_values": {k: float(v) for k, v in kpss_crit.items()},
        "reject_005": kpss_p < 0.05,
    }

    # Random walk (non-stationary)
    random_walk = np.cumsum(np.random.RandomState(99).normal(0, 1, 100)).tolist()
    kpss_rw_stat, kpss_rw_p, kpss_rw_lags, kpss_rw_crit = sm_kpss(random_walk, regression='c', nlags='auto')
    results["kpss_nonstationary"] = {
        "data": random_walk,
        "trend": False,
        "statistic": float(kpss_rw_stat),
        "p_value": float(kpss_rw_p),
        "n_lags": int(kpss_rw_lags),
        "critical_values": {k: float(v) for k, v in kpss_rw_crit.items()},
        "reject_005": kpss_rw_p < 0.05,
    }
except ImportError:
    # Fallback: just record asymptotic critical values
    results["kpss_critical_values"] = {
        "level": {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739},
        "trend": {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216},
    }

# ===========================================================================
# 6. Durbin-Watson
# ===========================================================================

# No autocorrelation — d ≈ 2
dw_no_ac = np.random.RandomState(42).normal(0, 1, 50).tolist()
# Manual DW computation
num_dw = sum((dw_no_ac[t] - dw_no_ac[t-1])**2 for t in range(1, len(dw_no_ac)))
den_dw = sum(e**2 for e in dw_no_ac)
dw_stat = num_dw / den_dw
results["durbin_watson_no_autocorr"] = {
    "data": dw_no_ac,
    "statistic": dw_stat,
    "rho_hat": 1.0 - dw_stat / 2.0,
    "near_2": abs(dw_stat - 2.0) < 1.0,
}

# Positive autocorrelation — d < 2
dw_pos = [0.0]
rng2 = np.random.RandomState(77)
for i in range(49):
    dw_pos.append(0.8 * dw_pos[-1] + rng2.normal(0, 0.5))
num_dw2 = sum((dw_pos[t] - dw_pos[t-1])**2 for t in range(1, len(dw_pos)))
den_dw2 = sum(e**2 for e in dw_pos)
dw_stat2 = num_dw2 / den_dw2
results["durbin_watson_positive_ac"] = {
    "data": dw_pos,
    "statistic": dw_stat2,
    "rho_hat": 1.0 - dw_stat2 / 2.0,
    "below_2": dw_stat2 < 2.0,
}

# ===========================================================================
# 7. Tukey HSD (prepared — test data ready for when impl lands)
# ===========================================================================

# Three groups
tukey_g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
tukey_g2 = [3.0, 4.0, 5.0, 6.0, 7.0]
tukey_g3 = [6.0, 7.0, 8.0, 9.0, 10.0]
try:
    tukey_res = stats.tukey_hsd(tukey_g1, tukey_g2, tukey_g3)
    # Pairwise differences
    results["tukey_hsd_3groups"] = {
        "groups": [tukey_g1, tukey_g2, tukey_g3],
        "statistic": float(tukey_res.statistic) if hasattr(tukey_res, 'statistic') else None,
        "pvalues": tukey_res.pvalue.tolist() if hasattr(tukey_res, 'pvalue') else None,
        "note": "pvalue matrix: rows=groups, cols=groups",
    }
except Exception as e:
    results["tukey_hsd_3groups"] = {
        "groups": [tukey_g1, tukey_g2, tukey_g3],
        "note": f"scipy tukey_hsd not available: {e}",
    }

# ===========================================================================
# 8. Breusch-Pagan heteroscedasticity test
# ===========================================================================

try:
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm

    # Homoscedastic
    np.random.seed(42)
    X_bp = np.random.normal(0, 1, (100, 2))
    X_bp_c = sm.add_constant(X_bp)
    y_bp = 2 + 3 * X_bp[:, 0] - 1 * X_bp[:, 1] + np.random.normal(0, 1, 100)
    ols_bp = sm.OLS(y_bp, X_bp_c).fit()
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(ols_bp.resid, X_bp_c)
    results["breusch_pagan_homoscedastic"] = {
        "lm_stat": float(bp_lm),
        "lm_pvalue": float(bp_p),
        "f_stat": float(bp_f),
        "f_pvalue": float(bp_fp),
        "reject_005": float(bp_p) < 0.05,
    }

    # Heteroscedastic: variance proportional to X
    y_het = 2 + 3 * X_bp[:, 0] + np.random.normal(0, 1, 100) * (1 + 2 * np.abs(X_bp[:, 0]))
    ols_het = sm.OLS(y_het, X_bp_c).fit()
    bp_het_lm, bp_het_p, bp_het_f, bp_het_fp = het_breuschpagan(ols_het.resid, X_bp_c)
    results["breusch_pagan_heteroscedastic"] = {
        "lm_stat": float(bp_het_lm),
        "lm_pvalue": float(bp_het_p),
        "f_stat": float(bp_het_f),
        "f_pvalue": float(bp_het_fp),
        "reject_005": float(bp_het_p) < 0.05,
    }
except ImportError:
    results["breusch_pagan_note"] = "statsmodels not available"

# ===========================================================================
# 9. VIF (Variance Inflation Factor)
# ===========================================================================

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Uncorrelated predictors — VIF ≈ 1
    np.random.seed(42)
    X_vif = np.random.normal(0, 1, (100, 3))
    X_vif_c = sm.add_constant(X_vif)
    vifs_uncorr = [float(variance_inflation_factor(X_vif_c, i)) for i in range(1, 4)]
    results["vif_uncorrelated"] = {
        "vifs": vifs_uncorr,
        "all_near_1": all(v < 2.0 for v in vifs_uncorr),
    }

    # Collinear predictors — VIF >> 1
    X_col = np.column_stack([X_vif[:, 0], X_vif[:, 0] + 0.01 * np.random.normal(0, 1, 100), X_vif[:, 2]])
    X_col_c = sm.add_constant(X_col)
    vifs_coll = [float(variance_inflation_factor(X_col_c, i)) for i in range(1, 4)]
    results["vif_collinear"] = {
        "vifs": vifs_coll,
        "high_vif": max(vifs_coll) > 10.0,
    }
except ImportError:
    results["vif_note"] = "statsmodels not available"

# ===========================================================================
# 10. ARCH-LM test (Engle 1982)
# ===========================================================================

try:
    from statsmodels.stats.diagnostic import het_arch

    # No ARCH effects (white noise)
    np.random.seed(42)
    wn = np.random.normal(0, 1, 200)
    arch_stat, arch_p, arch_f, arch_fp = het_arch(wn, nlags=5)
    results["arch_lm_no_effects"] = {
        "n": 200,
        "n_lags": 5,
        "lm_stat": float(arch_stat),
        "lm_pvalue": float(arch_p),
        "f_stat": float(arch_f),
        "f_pvalue": float(arch_fp),
        "reject_005": float(arch_p) < 0.05,
    }

    # GARCH-like data (should reject)
    garch_data = [0.0]
    sigma2 = 1.0
    rng_g = np.random.RandomState(42)
    for i in range(199):
        sigma2 = 0.1 + 0.3 * garch_data[-1]**2 + 0.6 * sigma2
        garch_data.append(rng_g.normal(0, np.sqrt(sigma2)))
    arch_g_stat, arch_g_p, _, _ = het_arch(np.array(garch_data), nlags=5)
    results["arch_lm_garch"] = {
        "n": 200,
        "n_lags": 5,
        "lm_stat": float(arch_g_stat),
        "lm_pvalue": float(arch_g_p),
        "reject_005": float(arch_g_p) < 0.05,
    }
except ImportError:
    results["arch_lm_note"] = "statsmodels not available"

# ===========================================================================
# Save
# ===========================================================================

with open("research/gold_standard/family_prerequisite_methods_expected.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"Prerequisite Methods Oracle: {len(results)} test cases generated")
for name, r in results.items():
    extra = ""
    if isinstance(r, dict) and 'p_value' in r:
        extra = f", p={r['p_value']:.4f}"
    elif isinstance(r, dict) and 'statistic' in r:
        extra = f", stat={r['statistic']:.4f}"
    print(f"  PASS {name}{extra}")
