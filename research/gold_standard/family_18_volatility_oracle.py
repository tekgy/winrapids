"""
Gold Standard Oracle: Family 18 — Volatility & Financial Time Series

Generates expected values from arch (Kevin Sheppard) for comparison with tambear.

Tests covered:
  - GARCH(1,1) parameter estimation
  - EWMA variance computation
  - Realized variance/volatility (analytical — exact sum of squared returns)
  - Bipower variation (analytical formula for constant returns)

Reference: arch 6.0+, numpy
Usage:
    python research/gold_standard/family_18_volatility_oracle.py
"""

import json
import numpy as np

results = {}

# ── GARCH(1,1) via arch library ───────────────────────────────────────────
try:
    from arch import arch_model

    # Generate returns with known GARCH structure
    np.random.seed(42)
    n = 2000
    omega_true, alpha_true, beta_true = 0.00001, 0.08, 0.90
    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[0] = omega_true / (1 - alpha_true - beta_true)
    for t in range(1, n):
        sigma2[t] = omega_true + alpha_true * returns[t-1]**2 + beta_true * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

    am = arch_model(returns * 100, vol='Garch', p=1, q=1, mean='Zero')
    res = am.fit(disp='off')

    results["garch11_arch_library"] = {
        "omega": float(res.params['omega']),
        "alpha": float(res.params['alpha[1]']),
        "beta": float(res.params['beta[1]']),
        "log_likelihood": float(res.loglikelihood),
        "note": "arch library GARCH(1,1) on simulated data (scaled by 100). True: omega=0.001, alpha=0.08, beta=0.90"
    }
except ImportError:
    print("arch library not available — using analytical values only")

# ── EWMA (analytical) ─────────────────────────────────────────────────────
# EWMA: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
lambda_val = 0.94
returns_ewma = [0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.02, -0.015]
sigma2_ewma = [sum(r**2 for r in returns_ewma) / len(returns_ewma)]  # unconditional
for t in range(1, len(returns_ewma)):
    sigma2_ewma.append(lambda_val * sigma2_ewma[-1] + (1 - lambda_val) * returns_ewma[t-1]**2)

results["ewma_lambda094"] = {
    "lambda": lambda_val,
    "returns": returns_ewma,
    "sigma2": [float(s) for s in sigma2_ewma],
    "note": "EWMA with λ=0.94, backcast initialization"
}

# ── Realized Variance (exact analytical) ──────────────────────────────────
returns_rv = [0.01, -0.02, 0.015, -0.005, 0.03]
rv = sum(r**2 for r in returns_rv)
rvol = np.sqrt(rv)

results["realized_variance_exact"] = {
    "returns": returns_rv,
    "rv": float(rv),
    "rvol": float(rvol),
    "note": "RV = Σr², RVol = √RV. Exact."
}

# ── Bipower Variation (analytical for constant |r|) ──────────────────────
c = 0.02
n_bp = 20
mu1 = np.sqrt(2 / np.pi)  # E[|Z|] for standard normal
bpv_constant = (n_bp - 1) * c * c / (mu1**2)

results["bipower_variation_constant"] = {
    "constant_return": c,
    "n": n_bp,
    "mu1_squared": float(mu1**2),
    "expected_bpv": float(bpv_constant),
    "note": "For constant |r|=c: BPV = (n-1)·c²/μ₁². μ₁ = √(2/π)"
}

# ── Kyle Lambda (exact linear) ───────────────────────────────────────────
# ΔP = λ · signed_volume → λ = Cov(ΔP, V) / Var(V)
lambda_true = 0.005
signed_volumes = [float(i * 100) for i in range(-10, 11)]
price_changes = [lambda_true * v for v in signed_volumes]

results["kyle_lambda_exact"] = {
    "lambda_true": lambda_true,
    "signed_volumes": signed_volumes,
    "price_changes": price_changes,
    "note": "Perfect linear: λ = Cov(ΔP,V)/Var(V) = 0.005 exactly"
}

# ── Amihud Illiquidity (exact) ───────────────────────────────────────────
returns_am = [0.02, -0.01, 0.03]
volumes_am = [1_000_000.0, 2_000_000.0, 500_000.0]
amihud = np.mean([abs(r) / v for r, v in zip(returns_am, volumes_am)])

results["amihud_exact"] = {
    "returns": returns_am,
    "volumes": volumes_am,
    "illiquidity": float(amihud),
    "note": "ILLIQ = (1/n) Σ |r_t|/V_t"
}

# ── Annualization ────────────────────────────────────────────────────────
daily_vol = 0.015
annual_vol = daily_vol * np.sqrt(252)

results["annualize_252"] = {
    "daily_vol": daily_vol,
    "trading_days": 252,
    "annual_vol": float(annual_vol),
    "note": "σ_annual = σ_daily × √252"
}

# ── Write output ─────────────────────────────────────────────────────────
output_path = "research/gold_standard/family_18_expected.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Family 18 oracle: {len(results)} test groups written to {output_path}")
for key in results:
    print(f"  {key}")
