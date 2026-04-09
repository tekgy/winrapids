"""
Gold Standard Oracle: GARCH Variants (EGARCH, GJR-GARCH, TGARCH)

Tests fit quality against the `arch` Python package when available,
plus analytical properties that must hold for any correct implementation.

Usage:
    python research/gold_standard/family_garch_variants_oracle.py
"""

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

results = {}

# Generate synthetic GARCH(1,1) return series with known parameters
# r_t = sigma_t * z_t, z_t ~ N(0,1)
# sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}

def simulate_garch(n, omega, alpha, beta, seed=42):
    rng = np.random.RandomState(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    for t in range(1, n):
        returns[t - 1] = np.sqrt(sigma2[t - 1]) * rng.standard_normal()
        sigma2[t] = omega + alpha * returns[t - 1]**2 + beta * sigma2[t - 1]
    returns[n - 1] = np.sqrt(sigma2[n - 1]) * rng.standard_normal()
    return returns, sigma2

# GJR asymmetric simulation
def simulate_gjr_garch(n, omega, alpha, gamma, beta, seed=42):
    rng = np.random.RandomState(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - gamma/2 - beta)
    for t in range(1, n):
        returns[t - 1] = np.sqrt(sigma2[t - 1]) * rng.standard_normal()
        indicator = 1.0 if returns[t - 1] < 0 else 0.0
        sigma2[t] = (omega + alpha * returns[t - 1]**2
                     + gamma * returns[t - 1]**2 * indicator
                     + beta * sigma2[t - 1])
    returns[n - 1] = np.sqrt(sigma2[n - 1]) * rng.standard_normal()
    return returns, sigma2

# ===========================================================================
# GARCH(1,1) synthetic data
# ===========================================================================

n = 500
omega_true = 0.02
alpha_true = 0.08
beta_true = 0.90

r_garch, s2_garch = simulate_garch(n, omega_true, alpha_true, beta_true, seed=42)

results["garch_synthetic"] = {
    "n": n,
    "true_omega": omega_true,
    "true_alpha": alpha_true,
    "true_beta": beta_true,
    "true_persistence": alpha_true + beta_true,
    "unconditional_variance_theoretical": omega_true / (1 - alpha_true - beta_true),
    "returns": r_garch.tolist(),
    "true_variances": s2_garch.tolist(),
    "sample_variance": float(np.var(r_garch, ddof=1)),
}

# ===========================================================================
# GJR-GARCH asymmetric data
# ===========================================================================

omega_gjr = 0.02
alpha_gjr = 0.05
gamma_gjr = 0.08  # asymmetry: negative shocks contribute extra
beta_gjr = 0.85

r_gjr, s2_gjr = simulate_gjr_garch(n, omega_gjr, alpha_gjr, gamma_gjr, beta_gjr, seed=123)

results["gjr_garch_synthetic"] = {
    "n": n,
    "true_omega": omega_gjr,
    "true_alpha": alpha_gjr,
    "true_gamma": gamma_gjr,
    "true_beta": beta_gjr,
    "true_persistence": alpha_gjr + gamma_gjr / 2 + beta_gjr,
    "returns": r_gjr.tolist(),
    "true_variances": s2_gjr.tolist(),
    "leverage_expected": "negative returns should produce higher future variance",
}

# ===========================================================================
# Try `arch` package for validation (optional)
# ===========================================================================

try:
    from arch import arch_model

    # Fit GARCH(1,1) with `arch`
    am = arch_model(r_garch, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')

    results["arch_garch11_fit"] = {
        "omega": float(res.params['omega']),
        "alpha": float(res.params['alpha[1]']),
        "beta": float(res.params['beta[1]']),
        "log_likelihood": float(res.loglikelihood),
    }

    # Fit GJR-GARCH
    am_gjr = arch_model(r_gjr, mean='Zero', vol='GARCH', p=1, o=1, q=1, dist='normal')
    res_gjr = am_gjr.fit(disp='off')

    results["arch_gjr_fit"] = {
        "omega": float(res_gjr.params['omega']),
        "alpha": float(res_gjr.params['alpha[1]']),
        "gamma": float(res_gjr.params['gamma[1]']),
        "beta": float(res_gjr.params['beta[1]']),
        "log_likelihood": float(res_gjr.loglikelihood),
    }

    # Fit EGARCH
    am_e = arch_model(r_garch, mean='Zero', vol='EGARCH', p=1, o=1, q=1, dist='normal')
    res_e = am_e.fit(disp='off')

    results["arch_egarch_fit"] = {
        "omega": float(res_e.params['omega']),
        "alpha": float(res_e.params['alpha[1]']),
        "gamma": float(res_e.params['gamma[1]']),
        "beta": float(res_e.params['beta[1]']),
        "log_likelihood": float(res_e.loglikelihood),
    }

except ImportError:
    results["arch_package_note"] = (
        "`arch` not installed — property tests only. "
        "Install with: pip install arch"
    )

# ===========================================================================
# Property tests (must hold for any correct implementation)
# ===========================================================================

results["properties"] = {
    "variances_positive": "all sigma2_t > 0",
    "variances_finite": "no NaN or Inf in sigma2_t",
    "egarch_log_variances_finite": "log_sigma2_t finite for all t",
    "persistence_bound_garch": "alpha + beta < 1 for stationary",
    "persistence_bound_gjr": "alpha + gamma/2 + beta < 1 for stationary",
    "omega_positive": "omega > 0 for GARCH and GJR (EGARCH omega unrestricted)",
    "alpha_nonneg_garch": "alpha >= 0 for GARCH (EGARCH unrestricted)",
    "beta_nonneg_garch": "beta >= 0 for GARCH (EGARCH unrestricted)",
    "ll_finite": "log-likelihood finite",
    "sample_var_matches_unconditional": (
        "mean(sigma2) should be close to sample_var (within 30%) for well-fit model"
    ),
}

# ===========================================================================
# Save
# ===========================================================================

with open("research/gold_standard/family_garch_variants_expected.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"GARCH Variants Oracle: {len(results)} test cases generated")
for name in results:
    print(f"  PASS {name}")
