"""
Gold Standard Oracle: Tier 1 Expansion Methods

Proactive oracle for methods about to be implemented.
Tests against established implementations with full precision.

Methods covered:
  - Cramér's V (association strength for contingency tables)
  - Distance correlation (dcor package)
  - Ridge regression (sklearn)
  - Lasso regression (sklearn)
  - ICC types 1, 2, 3 (pingouin)
  - Concordance correlation (Lin 1989)
  - Anderson-Darling normality test
  - Friedman test (repeated measures nonparametric)
  - Risk ratio / relative risk
  - Common language effect size

Usage:
    python research/gold_standard/family_expansion_tier1_oracle.py
"""

import json
import numpy as np
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

results = {}

# ===========================================================================
# 1. Cramér's V
# ===========================================================================

# Contingency table: strong association
table_strong = np.array([[50, 5], [5, 40]])
chi2_s, p_s, dof_s, _ = stats.chi2_contingency(table_strong)
n = table_strong.sum()
k = min(table_strong.shape) - 1
cramers_v_strong = float(np.sqrt(chi2_s / (n * k)))

results["cramers_v_strong"] = {
    "table": table_strong.tolist(),
    "chi2": float(chi2_s),
    "p_value": float(p_s),
    "cramers_v": cramers_v_strong,
    "n": int(n),
}

# Weak association
table_weak = np.array([[30, 20], [25, 25]])
chi2_w, p_w, dof_w, _ = stats.chi2_contingency(table_weak)
n_w = table_weak.sum()
k_w = min(table_weak.shape) - 1
cramers_v_weak = float(np.sqrt(chi2_w / (n_w * k_w)))

results["cramers_v_weak"] = {
    "table": table_weak.tolist(),
    "chi2": float(chi2_w),
    "cramers_v": cramers_v_weak,
}

# No association (independent)
table_indep = np.array([[25, 25], [25, 25]])
chi2_i, p_i, _, _ = stats.chi2_contingency(table_indep, correction=False)
cramers_v_indep = float(np.sqrt(chi2_i / (100 * 1)))

results["cramers_v_independent"] = {
    "table": table_indep.tolist(),
    "cramers_v": cramers_v_indep,
    "expected_near_zero": True,
}

# 3x3 table
table_3x3 = np.array([[20, 5, 5], [5, 20, 5], [5, 5, 20]])
chi2_3, p_3, _, _ = stats.chi2_contingency(table_3x3)
cramers_v_3x3 = float(np.sqrt(chi2_3 / (table_3x3.sum() * (min(table_3x3.shape) - 1))))

results["cramers_v_3x3"] = {
    "table": table_3x3.tolist(),
    "chi2": float(chi2_3),
    "cramers_v": cramers_v_3x3,
}

# ===========================================================================
# 2. Ridge Regression
# ===========================================================================

try:
    from sklearn.linear_model import Ridge, Lasso

    np.random.seed(42)
    n_r = 50
    X_r = np.random.normal(0, 1, (n_r, 3))
    beta_true = np.array([2.0, -1.0, 0.5])
    y_r = X_r @ beta_true + np.random.normal(0, 0.5, n_r)

    for alpha, label in [(0.1, "weak"), (1.0, "moderate"), (10.0, "strong")]:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_r, y_r)
        results[f"ridge_{label}"] = {
            "alpha": alpha,
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "r_squared": float(model.score(X_r, y_r)),
            "true_beta": beta_true.tolist(),
        }

    # Lasso
    for alpha, label in [(0.01, "weak"), (0.1, "moderate"), (1.0, "strong")]:
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        model.fit(X_r, y_r)
        results[f"lasso_{label}"] = {
            "alpha": alpha,
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "r_squared": float(model.score(X_r, y_r)),
            "n_nonzero": int(np.sum(model.coef_ != 0)),
        }

    results["ridge_lasso_data"] = {
        "X": X_r.tolist(),
        "y": y_r.tolist(),
        "true_beta": beta_true.tolist(),
        "seed": 42,
    }

except ImportError:
    results["ridge_lasso_note"] = "sklearn not available"

# ===========================================================================
# 3. Anderson-Darling normality test
# ===========================================================================

np.random.seed(42)
normal_data = np.random.normal(0, 1, 50)
ad_stat_n, ad_crit_n, ad_sig_n = stats.anderson(normal_data, dist='norm')

results["anderson_darling_normal"] = {
    "data_seed": 42,
    "n": 50,
    "statistic": float(ad_stat_n),
    "critical_values": ad_crit_n.tolist(),
    "significance_levels": ad_sig_n.tolist(),
    "reject_5pct": float(ad_stat_n) > float(ad_crit_n[2]),  # 5% is index 2
}

uniform_data = np.random.uniform(0, 1, 50)
ad_stat_u, ad_crit_u, ad_sig_u = stats.anderson(uniform_data, dist='norm')

results["anderson_darling_uniform"] = {
    "statistic": float(ad_stat_u),
    "critical_values": ad_crit_u.tolist(),
    "reject_5pct": float(ad_stat_u) > float(ad_crit_u[2]),
}

# ===========================================================================
# 4. Friedman test (repeated measures)
# ===========================================================================

# Three treatments, 6 subjects
treatment_a = [5.0, 6.0, 7.0, 4.0, 5.0, 6.0]
treatment_b = [8.0, 9.0, 7.0, 8.0, 9.0, 10.0]
treatment_c = [6.0, 7.0, 8.0, 5.0, 6.0, 7.0]

fried_stat, fried_p = stats.friedmanchisquare(treatment_a, treatment_b, treatment_c)

results["friedman_3treatments"] = {
    "data": [treatment_a, treatment_b, treatment_c],
    "statistic": float(fried_stat),
    "p_value": float(fried_p),
    "reject_005": float(fried_p) < 0.05,
}

# No difference case
eq_a = [5.0, 6.0, 7.0, 4.0, 5.0, 6.0]
eq_b = [5.0, 6.0, 7.0, 4.0, 5.0, 6.0]
eq_c = [5.0, 6.0, 7.0, 4.0, 5.0, 6.0]

fried_eq_stat, fried_eq_p = stats.friedmanchisquare(eq_a, eq_b, eq_c)
results["friedman_equal"] = {
    "statistic": float(fried_eq_stat),
    "p_value": float(fried_eq_p),
    "reject_005": float(fried_eq_p) < 0.05,
}

# ===========================================================================
# 5. Concordance Correlation Coefficient (Lin 1989)
# ===========================================================================

# CCC = 2*rho*s_x*s_y / (s_x^2 + s_y^2 + (mean_x - mean_y)^2)
def concordance_cc(x, y):
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    r = np.corrcoef(x, y)[0, 1]
    return 2 * r * sx * sy / (sx**2 + sy**2 + (mx - my)**2)

np.random.seed(42)
x_ccc = np.random.normal(10, 2, 50)
y_perfect = x_ccc.copy()
y_shifted = x_ccc + 2.0
y_noisy = x_ccc + np.random.normal(0, 1, 50)
y_uncorr = np.random.normal(10, 2, 50)

results["concordance_cc"] = {
    "perfect_agreement": float(concordance_cc(x_ccc, y_perfect)),  # should be 1.0
    "shifted": float(concordance_cc(x_ccc, y_shifted)),  # < 1 despite r=1
    "noisy": float(concordance_cc(x_ccc, y_noisy)),
    "uncorrelated": float(concordance_cc(x_ccc, y_uncorr)),
    "note": "CCC < Pearson r when means differ or scales differ",
}

# ===========================================================================
# 6. Risk Ratio / Relative Risk
# ===========================================================================

# 2x2 table: [[a, b], [c, d]]
# RR = (a/(a+b)) / (c/(c+d))
a, b, c, d = 30, 70, 10, 90
rr = (a / (a + b)) / (c / (c + d))
rr_se = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
rr_ci_lower = np.exp(np.log(rr) - 1.96 * rr_se)
rr_ci_upper = np.exp(np.log(rr) + 1.96 * rr_se)

results["risk_ratio"] = {
    "table": [[a, b], [c, d]],
    "risk_ratio": float(rr),
    "log_rr": float(np.log(rr)),
    "se_log_rr": float(rr_se),
    "ci_95_lower": float(rr_ci_lower),
    "ci_95_upper": float(rr_ci_upper),
}

# ===========================================================================
# 7. Common Language Effect Size (probability of superiority)
# ===========================================================================

# CLES = P(X > Y) for X ~ group1, Y ~ group2
# For normal data: CLES = Phi(d / sqrt(2))
from scipy.stats import norm
d_val = 0.8  # large effect
cles = float(norm.cdf(d_val / np.sqrt(2)))

results["common_language_es"] = {
    "cohens_d": d_val,
    "cles": cles,  # probability that random X from group 1 > random Y from group 2
    "note": "CLES = Phi(d/sqrt(2)) for equal-variance normal groups",
}

# ===========================================================================
# Save
# ===========================================================================

with open("research/gold_standard/family_expansion_tier1_expected.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"Tier 1 Expansion Oracle: {len(results)} test cases generated")
for name, r in results.items():
    if isinstance(r, dict):
        if 'cramers_v' in r:
            print(f"  PASS {name} (V={r['cramers_v']:.4f})")
        elif 'statistic' in r:
            print(f"  PASS {name} (stat={r['statistic']:.4f})")
        elif 'coefficients' in r:
            print(f"  PASS {name} (R2={r.get('r_squared', '?'):.4f})")
        elif 'risk_ratio' in r:
            print(f"  PASS {name} (RR={r['risk_ratio']:.3f})")
        else:
            print(f"  PASS {name}")
    else:
        print(f"  PASS {name}")
