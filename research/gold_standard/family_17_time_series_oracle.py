"""
Gold Standard Oracle: Family 17 — Time Series Models

Generates expected values from statsmodels for comparison with tambear.

Tests covered:
  - AR(p) fitting via Yule-Walker (coefficient recovery, innovation variance, AIC)
  - ACF/PACF computation
  - ADF test (statistic, critical values)
  - Simple Exponential Smoothing level/forecast

Reference: statsmodels 0.14+, numpy
Usage:
    python research/gold_standard/family_17_time_series_oracle.py
"""

import json
import numpy as np

results = {}

# ── Deterministic test data (must match Rust LCG) ──────────────────────────
# Generate AR(1) data with known phi=0.7, seed=42
np.random.seed(42)
n = 2000
phi_true = 0.7
ar1_data = np.zeros(n)
for t in range(1, n):
    ar1_data[t] = phi_true * ar1_data[t-1] + np.random.randn()

# ── AR(1) via Yule-Walker ──────────────────────────────────────────────────
try:
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.regression.linear_model import yule_walker

    # Yule-Walker direct
    rho, sigma2 = yule_walker(ar1_data, order=1, method='mle')
    results["ar1_yule_walker"] = {
        "coefficient": float(rho[0]),
        "sigma2": float(sigma2),
        "n": n,
        "true_phi": phi_true,
        "note": "statsmodels yule_walker, method=mle"
    }

    # AR(2) on AR(1) data — phi2 should be ~0
    rho2, sigma2_2 = yule_walker(ar1_data, order=2, method='mle')
    results["ar2_overfitting_ar1"] = {
        "coefficients": [float(c) for c in rho2],
        "sigma2": float(sigma2_2),
        "note": "AR(2) fit to AR(1) data: phi2 should be ~0"
    }
except ImportError:
    print("statsmodels not available — using analytical values")
    # Fallback: analytical Yule-Walker for AR(1) with known autocorrelation
    # For AR(1): phi = r(1)/r(0), sigma2 = r(0)(1 - phi^2)
    results["ar1_yule_walker"] = {
        "coefficient": 0.7,  # true value; actual estimate will be close
        "note": "analytical (statsmodels not available)"
    }

# ── ACF ────────────────────────────────────────────────────────────────────
try:
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

    # White noise ACF — should be ~0 for lag > 0
    np.random.seed(99)
    white_noise = np.random.randn(500)
    acf_vals = sm_acf(white_noise, nlags=10, fft=True)
    results["acf_white_noise"] = {
        "values": [float(v) for v in acf_vals],
        "note": "ACF of white noise (n=500). lag-0=1, rest ~0"
    }

    # AR(1) ACF — should decay as phi^k
    acf_ar1 = sm_acf(ar1_data, nlags=10, fft=True)
    results["acf_ar1"] = {
        "values": [float(v) for v in acf_ar1],
        "note": "ACF of AR(1) with phi=0.7. Theoretical: phi^k"
    }

    # PACF of AR(1) — should cutoff after lag 1
    pacf_ar1 = sm_pacf(ar1_data, nlags=10)
    results["pacf_ar1"] = {
        "values": [float(v) for v in pacf_ar1],
        "note": "PACF of AR(1). Should have significant lag-1 only"
    }
except ImportError:
    print("statsmodels ACF not available")

# ── ADF test ───────────────────────────────────────────────────────────────
try:
    from statsmodels.tsa.stattools import adfuller

    # Stationary AR(1) — should reject
    np.random.seed(42)
    stationary = np.zeros(500)
    for t in range(1, 500):
        stationary[t] = 0.2 * stationary[t-1] + np.random.randn()
    adf_stat = adfuller(stationary, maxlag=2, regression='c', autolag=None)
    results["adf_stationary"] = {
        "statistic": float(adf_stat[0]),
        "pvalue": float(adf_stat[1]),
        "critical_1pct": float(adf_stat[4]['1%']),
        "critical_5pct": float(adf_stat[4]['5%']),
        "critical_10pct": float(adf_stat[4]['10%']),
        "note": "Stationary AR(1) phi=0.2, n=500, maxlag=2"
    }

    # Random walk — should fail to reject
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(200) * 0.25)
    adf_rw = adfuller(rw, maxlag=2, regression='c', autolag=None)
    results["adf_random_walk"] = {
        "statistic": float(adf_rw[0]),
        "pvalue": float(adf_rw[1]),
        "critical_1pct": float(adf_rw[4]['1%']),
        "critical_5pct": float(adf_rw[4]['5%']),
        "critical_10pct": float(adf_rw[4]['10%']),
        "note": "Random walk n=200. Should fail to reject unit root"
    }
except ImportError:
    print("statsmodels ADF not available")

# ── Simple Exponential Smoothing ───────────────────────────────────────────
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    data_ses = np.array([10.0, 12.0, 11.0, 13.0, 10.5, 11.5])
    model = SimpleExpSmoothing(data_ses).fit(smoothing_level=0.3, optimized=False)
    results["ses_alpha_03"] = {
        "fitted": [float(v) for v in model.fittedvalues],
        "forecast": float(model.forecast(1)[0]),
        "level": float(model.level),
        "alpha": 0.3,
        "note": "SES with alpha=0.3 on [10,12,11,13,10.5,11.5]"
    }
except ImportError:
    print("statsmodels SES not available")

# ── Differencing (analytical) ──────────────────────────────────────────────
results["differencing"] = {
    "linear_d1": {
        "input": [0.0, 2.0, 4.0, 6.0, 8.0],
        "output": [2.0, 2.0, 2.0, 2.0],
        "note": "d=1 of linear: constant"
    },
    "quadratic_d2": {
        "input": [0.0, 1.0, 4.0, 9.0, 16.0],
        "d1": [1.0, 3.0, 5.0, 7.0],
        "d2": [2.0, 2.0, 2.0],
        "note": "d=2 of quadratic: constant"
    }
}

# ── Write output ───────────────────────────────────────────────────────────
output_path = "research/gold_standard/family_17_expected.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Family 17 oracle: {len(results)} test groups written to {output_path}")
for key, val in results.items():
    n_vals = sum(1 for _ in str(val))  # rough count
    print(f"  {key}")
