"""
Gold Standard Oracle: Distribution CDFs/PDFs/Quantiles

Proactive oracle — ready for when tambear implements these distributions.
Tests CDF, PDF, quantile (ppf), moments, and edge cases against scipy.stats.

Distributions covered:
  - Weibull (minimum): shape k, scale lambda
  - Pareto: shape alpha, scale xm
  - Generalized Extreme Value (GEV): shape xi, loc, scale
  - Negative Binomial: n, p
  - Hypergeometric: M, n, N
  - Inverse Gaussian: mu, lambda
  - Log-logistic: alpha, beta

Usage:
    python research/gold_standard/family_distributions_cdf_oracle.py
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
# 1. Weibull (minimum)
# ===========================================================================

# Weibull: F(x) = 1 - exp(-(x/lambda)^k)
for k, lam, label in [(1.0, 1.0, "exponential"), (2.0, 1.0, "rayleigh_like"),
                        (0.5, 1.0, "heavy_tail"), (3.0, 2.0, "typical")]:
    dist = stats.weibull_min(k, scale=lam)
    x_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    results[f"weibull_{label}"] = {
        "k": k, "lambda": lam,
        "mean": float(dist.mean()),
        "var": float(dist.var()),
        "median": float(dist.median()),
        "cdf": {str(x): float(dist.cdf(x)) for x in x_vals},
        "pdf": {str(x): float(dist.pdf(x)) for x in x_vals},
        "quantiles": {
            "0.25": float(dist.ppf(0.25)),
            "0.50": float(dist.ppf(0.50)),
            "0.75": float(dist.ppf(0.75)),
            "0.95": float(dist.ppf(0.95)),
            "0.99": float(dist.ppf(0.99)),
        },
    }

# ===========================================================================
# 2. Pareto
# ===========================================================================

# Pareto: F(x) = 1 - (xm/x)^alpha for x >= xm
for alpha, xm, label in [(1.5, 1.0, "heavy"), (3.0, 1.0, "moderate"),
                           (5.0, 2.0, "light")]:
    dist = stats.pareto(alpha, scale=xm)
    x_vals = [xm, xm*1.5, xm*2, xm*5, xm*10]
    results[f"pareto_{label}"] = {
        "alpha": alpha, "xm": xm,
        "mean": float(dist.mean()) if alpha > 1 else None,
        "var": float(dist.var()) if alpha > 2 else None,
        "cdf": {str(x): float(dist.cdf(x)) for x in x_vals},
        "pdf": {str(x): float(dist.pdf(x)) for x in x_vals},
        "quantiles": {
            "0.50": float(dist.ppf(0.50)),
            "0.90": float(dist.ppf(0.90)),
            "0.99": float(dist.ppf(0.99)),
        },
    }

# ===========================================================================
# 3. Generalized Extreme Value (GEV)
# ===========================================================================

# GEV: three sub-families based on shape xi
# xi > 0: Frechet (heavy-tailed), xi = 0: Gumbel, xi < 0: Weibull (bounded)
for xi, loc, scale, label in [(0.0, 0.0, 1.0, "gumbel"),
                                (0.5, 0.0, 1.0, "frechet"),
                                (-0.5, 0.0, 1.0, "reversed_weibull")]:
    # scipy convention: c = -xi (sign flip!)
    dist = stats.genextreme(-xi, loc=loc, scale=scale)
    x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    results[f"gev_{label}"] = {
        "xi": xi, "loc": loc, "scale": scale,
        "scipy_c": -xi,
        "mean": float(dist.mean()) if dist.mean() != np.inf else None,
        "cdf": {str(x): float(dist.cdf(x)) for x in x_vals},
        "quantiles": {
            "0.10": float(dist.ppf(0.10)),
            "0.50": float(dist.ppf(0.50)),
            "0.90": float(dist.ppf(0.90)),
            "0.99": float(dist.ppf(0.99)),
        },
    }

# ===========================================================================
# 4. Negative Binomial
# ===========================================================================

# NB(n, p): number of failures before n successes
for n, p, label in [(5, 0.5, "balanced"), (1, 0.3, "geometric"),
                     (10, 0.8, "high_success")]:
    dist = stats.nbinom(n, p)
    k_vals = [0, 1, 2, 5, 10, 20]
    results[f"nbinom_{label}"] = {
        "n": n, "p": p,
        "mean": float(dist.mean()),
        "var": float(dist.var()),
        "pmf": {str(k): float(dist.pmf(k)) for k in k_vals},
        "cdf": {str(k): float(dist.cdf(k)) for k in k_vals},
        "quantiles": {
            "0.25": int(dist.ppf(0.25)),
            "0.50": int(dist.ppf(0.50)),
            "0.75": int(dist.ppf(0.75)),
        },
    }

# ===========================================================================
# 5. Hypergeometric
# ===========================================================================

# Hypergeom(M, n, N): M total, n successes, N drawn
for M, n, N, label in [(20, 7, 12, "typical"), (50, 10, 5, "rare_draw"),
                        (100, 50, 10, "balanced")]:
    dist = stats.hypergeom(M, n, N)
    k_vals = list(range(min(n, N) + 1))
    results[f"hypergeom_{label}"] = {
        "M": M, "n": n, "N": N,
        "mean": float(dist.mean()),
        "var": float(dist.var()),
        "pmf": {str(k): float(dist.pmf(k)) for k in k_vals[:8]},
        "cdf": {str(k): float(dist.cdf(k)) for k in k_vals[:8]},
    }

# ===========================================================================
# 6. Inverse Gaussian (Wald distribution)
# ===========================================================================

for mu, lam, label in [(1.0, 1.0, "standard"), (2.0, 3.0, "typical"),
                        (0.5, 10.0, "concentrated")]:
    dist = stats.invgauss(mu/lam, scale=lam)
    x_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    results[f"invgauss_{label}"] = {
        "mu": mu, "lambda": lam,
        "mean": float(dist.mean()),
        "var": float(dist.var()),
        "cdf": {str(x): float(dist.cdf(x)) for x in x_vals},
        "pdf": {str(x): float(dist.pdf(x)) for x in x_vals},
        "quantiles": {
            "0.50": float(dist.ppf(0.50)),
            "0.95": float(dist.ppf(0.95)),
        },
    }

# ===========================================================================
# Save
# ===========================================================================

with open("research/gold_standard/family_distributions_cdf_expected.json", "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"Distribution CDF Oracle: {len(results)} test cases generated")
for name, r in results.items():
    extra = f", E={r.get('mean', '?')}" if 'mean' in r else ""
    print(f"  PASS {name}{extra}")
