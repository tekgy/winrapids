"""
Task #27: Systematic Precision Investigation

Compare scipy vs mpmath (50-digit ground truth) across all test families.
For every case where scipy deviates from mpmath by > 1e-15, document the error.

This identifies:
1. Where scipy has known precision limitations
2. Where tambear (with upgraded erfc) might be MORE accurate than scipy
3. Known scipy bugs in our test families

Output: precision_investigation_results.json
"""

import json
import math
import numpy as np
from scipy import stats, special
import mpmath

mpmath.mp.dps = 50  # 50 decimal digits

results = {}

# ===========================================================================
# 1. Error function / normal CDF — the foundation of all p-values
# ===========================================================================

print("=" * 70)
print("1. erf/erfc/normal_cdf precision vs mpmath")
print("=" * 70)

erf_tests = {}
for x in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 27.0]:
    mp_erf = float(mpmath.erf(x))
    mp_erfc = float(mpmath.erfc(x))
    scipy_erf = float(special.erf(x))
    scipy_erfc = float(special.erfc(x))

    erf_err = abs(scipy_erf - mp_erf)
    erfc_err = abs(scipy_erfc - mp_erfc)

    if erf_err > 1e-16 or erfc_err > 1e-16:
        erf_tests[f"x={x}"] = {
            "mpmath_erf": mp_erf,
            "scipy_erf": scipy_erf,
            "erf_error": erf_err,
            "mpmath_erfc": mp_erfc,
            "scipy_erfc": scipy_erfc,
            "erfc_error": erfc_err,
        }
        print(f"  x={x:5.1f}: erf_err={erf_err:.2e}, erfc_err={erfc_err:.2e}")

if not erf_tests:
    print("  All erf/erfc values match mpmath to < 1e-16")

results["erf_erfc"] = erf_tests if erf_tests else {"status": "all match to machine precision"}

# Normal CDF in tails
print()
ncdf_tests = {}
for x in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -3.0, -5.0, -8.0]:
    mp_val = float(mpmath.ncdf(x))
    scipy_val = float(stats.norm.cdf(x))
    err = abs(scipy_val - mp_val)
    rel_err = err / abs(mp_val) if mp_val != 0 else err

    if err > 1e-15:
        ncdf_tests[f"x={x}"] = {
            "mpmath": mp_val,
            "scipy": scipy_val,
            "abs_error": err,
            "rel_error": rel_err,
        }
        print(f"  Phi({x}): abs_err={err:.2e}, rel_err={rel_err:.2e}")

if not ncdf_tests:
    print("  All normal CDF values match mpmath to < 1e-15")

results["normal_cdf"] = ncdf_tests if ncdf_tests else {"status": "all match to machine precision"}

# ===========================================================================
# 2. Incomplete beta function (drives t-CDF, F-CDF)
# ===========================================================================

print()
print("=" * 70)
print("2. Regularized incomplete beta I_x(a,b) precision")
print("=" * 70)

ibeta_tests = {}
test_cases_ibeta = [
    (0.5, 0.5, 0.5),   # symmetric
    (1.0, 1.0, 0.5),   # uniform
    (0.5, 0.5, 0.01),  # near 0
    (0.5, 0.5, 0.99),  # near 1
    (100.0, 100.0, 0.5),  # large params
    (100.0, 1.0, 0.99),   # extreme asymmetry
    (0.01, 0.01, 0.5),    # very small params
    (1000.0, 1000.0, 0.5), # very large params
    (5.0, 10.0, 0.3),     # typical t-CDF scenario
    (2.5, 15.0, 0.1),     # typical F-CDF scenario
]

for a, b, x in test_cases_ibeta:
    mp_val = float(mpmath.betainc(a, b, 0, x, regularized=True))
    scipy_val = float(special.betainc(a, b, x))
    err = abs(scipy_val - mp_val)
    rel_err = err / abs(mp_val) if mp_val != 0 else err

    if err > 1e-12:
        ibeta_tests[f"I_{x}({a},{b})"] = {
            "mpmath": mp_val,
            "scipy": scipy_val,
            "abs_error": err,
            "rel_error": rel_err,
        }
        print(f"  I_{x}({a},{b}): abs_err={err:.2e}, rel_err={rel_err:.2e}")

if not ibeta_tests:
    print("  All regularized incomplete beta values match mpmath to < 1e-12")

results["incomplete_beta"] = ibeta_tests if ibeta_tests else {"status": "all match to < 1e-12"}

# ===========================================================================
# 3. Incomplete gamma (drives chi2-CDF)
# ===========================================================================

print()
print("=" * 70)
print("3. Regularized incomplete gamma Q(a,x) precision")
print("=" * 70)

igamma_tests = {}
test_cases_igamma = [
    (1.0, 1.0),    # exponential
    (0.5, 0.5),    # chi2 df=1
    (5.0, 3.0),    # typical chi2
    (100.0, 90.0), # large df
    (0.01, 0.001), # very small
    (50.0, 100.0), # tail: x >> a
    (2.5, 7.815),  # chi2(5) at 5% critical value
]

for a, x in test_cases_igamma:
    mp_val = float(mpmath.gammainc(a, x, mpmath.inf, regularized=True))
    scipy_val = float(special.gammaincc(a, x))
    err = abs(scipy_val - mp_val)
    rel_err = err / abs(mp_val) if mp_val != 0 else err

    if err > 1e-12:
        igamma_tests[f"Q({a},{x})"] = {
            "mpmath": mp_val,
            "scipy": scipy_val,
            "abs_error": err,
            "rel_error": rel_err,
        }
        print(f"  Q({a},{x}): abs_err={err:.2e}, rel_err={rel_err:.2e}")

if not igamma_tests:
    print("  All regularized incomplete gamma values match mpmath to < 1e-12")

results["incomplete_gamma"] = igamma_tests if igamma_tests else {"status": "all match to < 1e-12"}

# ===========================================================================
# 4. Log-gamma function
# ===========================================================================

print()
print("=" * 70)
print("4. Log-gamma ln(Gamma(x)) precision")
print("=" * 70)

lgamma_tests = {}
for x in [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 0.01, 0.001]:
    mp_val = float(mpmath.loggamma(x))
    scipy_val = float(special.gammaln(x))
    err = abs(scipy_val - mp_val)
    rel_err = err / abs(mp_val) if mp_val != 0 else err

    if err > 1e-12:
        lgamma_tests[f"x={x}"] = {
            "mpmath": mp_val,
            "scipy": scipy_val,
            "abs_error": err,
            "rel_error": rel_err,
        }
        print(f"  lgamma({x}): abs_err={err:.2e}, rel_err={rel_err:.2e}")

if not lgamma_tests:
    print("  All log-gamma values match mpmath to < 1e-12")

results["log_gamma"] = lgamma_tests if lgamma_tests else {"status": "all match to < 1e-12"}

# ===========================================================================
# 5. p-values at critical boundaries (most impactful precision zone)
# ===========================================================================

print()
print("=" * 70)
print("5. p-values near significance boundaries (most impactful)")
print("=" * 70)

pval_tests = {}

# t-distribution p-values
print("\n  t-distribution:")
for t_val, df in [(1.96, 30), (2.0, 10), (2.576, 5), (3.0, 100), (5.0, 3)]:
    # Two-tailed p-value
    mp_p = float(2 * (1 - mpmath.ncdf(t_val * mpmath.sqrt(mpmath.mpf(df) / (mpmath.mpf(df) - 2 + t_val**2)))))
    # Actually use the proper t-CDF via mpmath
    # mpmath doesn't have t-distribution directly, use betainc
    x = df / (df + t_val**2)
    mp_ibeta = float(mpmath.betainc(df/2, 0.5, 0, x, regularized=True))
    mp_t_p = mp_ibeta  # one-tail
    scipy_t_p = float(stats.t.cdf(-abs(t_val), df))  # left tail

    err = abs(scipy_t_p - mp_t_p)
    rel_err = err / abs(mp_t_p) if mp_t_p > 1e-300 else err

    if err > 1e-12:
        key = f"t({t_val},df={df})"
        pval_tests[key] = {
            "mpmath": mp_t_p,
            "scipy": scipy_t_p,
            "abs_error": err,
            "rel_error": rel_err,
        }
        print(f"    {key}: err={err:.2e}")

# chi2 p-values
print("\n  chi-squared:")
for chi2_val, df in [(3.841, 1), (5.991, 2), (11.07, 5), (50.0, 10), (100.0, 5)]:
    mp_q = float(mpmath.gammainc(df/2, chi2_val/2, mpmath.inf, regularized=True))
    scipy_q = float(stats.chi2.sf(chi2_val, df))
    err = abs(scipy_q - mp_q)
    rel_err = err / abs(mp_q) if mp_q > 1e-300 else err

    if err > 1e-12:
        key = f"chi2({chi2_val},df={df})"
        pval_tests[key] = {
            "mpmath": mp_q,
            "scipy": scipy_q,
            "abs_error": err,
            "rel_error": rel_err,
        }
        print(f"    {key}: err={err:.2e}")

if not pval_tests:
    print("  All p-values match mpmath to < 1e-12")

results["p_values"] = pval_tests if pval_tests else {"status": "all match to < 1e-12"}

# ===========================================================================
# 6. Descriptive statistics: variance computation
# ===========================================================================

print()
print("=" * 70)
print("6. Variance computation precision (mean-shifted vs naive)")
print("=" * 70)

# Test case: data with large mean, small variance (catastrophic cancellation test)
# This is where tambear's two-pass centered algorithm beats naive implementations
np.random.seed(42)
large_mean_data = 1e8 + np.random.normal(0, 1, 1000)

# Naive (one-pass): E[X^2] - E[X]^2 — catastrophic cancellation
mean_naive = np.mean(large_mean_data)
var_naive_onepass = np.mean(large_mean_data**2) - mean_naive**2

# numpy (two-pass): sum((x - mean)^2) / n
var_numpy = float(np.var(large_mean_data))

# mpmath (extended precision ground truth)
mp_data = [mpmath.mpf(x) for x in large_mean_data]
mp_mean = sum(mp_data) / len(mp_data)
mp_var = float(sum((x - mp_mean)**2 for x in mp_data) / len(mp_data))

naive_err = abs(var_naive_onepass - mp_var)
numpy_err = abs(var_numpy - mp_var)

results["variance_precision"] = {
    "data_description": "N(1e8, 1) n=1000 — catastrophic cancellation test",
    "mpmath_variance": mp_var,
    "numpy_variance": var_numpy,
    "naive_onepass_variance": var_naive_onepass,
    "numpy_error": numpy_err,
    "naive_error": naive_err,
    "naive_digits_lost": math.log10(naive_err / mp_var) if naive_err > 0 and mp_var > 0 else 0,
    "numpy_digits_lost": math.log10(numpy_err / mp_var) if numpy_err > 0 and mp_var > 0 else 0,
}

print(f"  Ground truth variance: {mp_var:.15e}")
print(f"  numpy (2-pass):  err={numpy_err:.2e} ({-math.log10(numpy_err/mp_var):.1f} digits)" if numpy_err > 0 else "  numpy: exact")
print(f"  naive (1-pass):  err={naive_err:.2e} ({-math.log10(naive_err/mp_var):.1f} digits lost)" if naive_err > 0 else "  naive: exact")
print(f"  NOTE: tambear uses Welford/two-pass centered. Should match numpy or better.")

# ===========================================================================
# 7. Series acceleration: known exact limits
# ===========================================================================

print()
print("=" * 70)
print("7. Series acceleration vs mpmath exact limits")
print("=" * 70)

# pi/4 via Leibniz at various term counts
for n_terms in [10, 20, 50, 100]:
    mp_sum = float(sum(mpmath.mpf((-1)**k) / (2*k + 1) for k in range(n_terms)))
    py_sum = sum((-1.0)**k / (2*k + 1) for k in range(n_terms))
    err = abs(py_sum - mp_sum)
    if err > 1e-15:
        print(f"  Leibniz {n_terms} terms: f64 err vs mpmath = {err:.2e}")

mp_pi4 = float(mpmath.pi / 4)
mp_pi2_6 = float(mpmath.pi**2 / 6)
mp_ln2 = float(mpmath.log(2))

print(f"  pi/4   mpmath = {mp_pi4:.20e}")
print(f"  pi²/6  mpmath = {mp_pi2_6:.20e}")
print(f"  ln(2)  mpmath = {mp_ln2:.20e}")
print(f"  pi/4   math   = {math.pi/4:.20e}  err={abs(math.pi/4 - mp_pi4):.2e}")

results["series_limits"] = {
    "pi_over_4": {"mpmath": mp_pi4, "math": math.pi/4, "error": abs(math.pi/4 - mp_pi4)},
    "pi2_over_6": {"mpmath": mp_pi2_6, "math": math.pi**2/6, "error": abs(math.pi**2/6 - mp_pi2_6)},
    "ln2": {"mpmath": mp_ln2, "math": math.log(2), "error": abs(math.log(2) - mp_ln2)},
}

# ===========================================================================
# 8. Black-Scholes: the case where erfc matters most
# ===========================================================================

print()
print("=" * 70)
print("8. Black-Scholes pricing at extended precision")
print("=" * 70)

S, K, T, r, sigma = mpmath.mpf(100), mpmath.mpf(100), mpmath.mpf(1), mpmath.mpf('0.05'), mpmath.mpf('0.2')
d1 = (mpmath.log(S/K) + (r + sigma**2/2)*T) / (sigma * mpmath.sqrt(T))
d2 = d1 - sigma * mpmath.sqrt(T)
mp_call = float(S * mpmath.ncdf(d1) - K * mpmath.exp(-r*T) * mpmath.ncdf(d2))

scipy_call = float(stats.norm.cdf(float(d1)) * 100 - 100 * math.exp(-0.05) * stats.norm.cdf(float(d2)))

print(f"  BS call (mpmath 50dp): {mp_call:.15e}")
print(f"  BS call (scipy f64):   {scipy_call:.15e}")
print(f"  Error:                 {abs(scipy_call - mp_call):.2e}")

results["black_scholes"] = {
    "mpmath_call": mp_call,
    "scipy_call": scipy_call,
    "error": abs(scipy_call - mp_call),
    "note": "tambear with upgraded erfc should match this to ~1e-13",
}

# ===========================================================================
# 9. Digamma/trigamma
# ===========================================================================

print()
print("=" * 70)
print("9. Digamma/trigamma precision")
print("=" * 70)

digamma_tests = {}
for x in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 0.01]:
    mp_val = float(mpmath.digamma(x))
    scipy_val = float(special.digamma(x))
    err = abs(scipy_val - mp_val)
    rel_err = err / abs(mp_val) if mp_val != 0 else err

    if rel_err > 1e-13:
        digamma_tests[f"x={x}"] = {
            "mpmath": mp_val, "scipy": scipy_val,
            "abs_error": err, "rel_error": rel_err,
        }
        print(f"  digamma({x}): rel_err={rel_err:.2e}")

if not digamma_tests:
    print("  All digamma values match mpmath to < 1e-13 relative error")

results["digamma"] = digamma_tests if digamma_tests else {"status": "all match"}

# ===========================================================================
# Summary
# ===========================================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

n_issues = sum(1 for v in results.values() if isinstance(v, dict) and "status" not in v and len(v) > 0)
print(f"  Families with precision issues: {n_issues}")
for family, data in results.items():
    if isinstance(data, dict) and "status" not in data and len(data) > 0:
        print(f"    {family}: {len(data)} cases")

# Save
with open("research/gold_standard/task27_precision_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to research/gold_standard/task27_precision_results.json")
