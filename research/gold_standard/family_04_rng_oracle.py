"""
Gold Standard Oracle: Family 04 — Random Number Generation & Distributions

Generates expected values from scipy.stats for comparison with tambear.

Tests covered:
  - Distribution moments (mean, variance) for Normal, Exponential, Gamma, Beta, Poisson
  - Quantile values (ppf) for parametric distributions
  - KS test critical values for goodness-of-fit
  - Theoretical CDF values at specific points

Usage:
    python research/gold_standard/family_04_rng_oracle.py
"""

import json
import numpy as np
from scipy import stats

results = {}

# -- Normal(0, 1) --

dist = stats.norm(0, 1)
results["normal_0_1"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "std": float(dist.std()),
    "quantiles": {
        "0.01": float(dist.ppf(0.01)),
        "0.05": float(dist.ppf(0.05)),
        "0.25": float(dist.ppf(0.25)),
        "0.50": float(dist.ppf(0.50)),
        "0.75": float(dist.ppf(0.75)),
        "0.95": float(dist.ppf(0.95)),
        "0.99": float(dist.ppf(0.99)),
    },
    "cdf_at": {
        "0.0": float(dist.cdf(0.0)),
        "1.0": float(dist.cdf(1.0)),
        "-1.0": float(dist.cdf(-1.0)),
        "1.96": float(dist.cdf(1.96)),
        "-1.96": float(dist.cdf(-1.96)),
    },
}

# -- Normal(5, 2) --

dist = stats.norm(5, 2)
results["normal_5_2"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "quantiles": {
        "0.50": float(dist.ppf(0.50)),
        "0.95": float(dist.ppf(0.95)),
    },
}

# -- Exponential(lambda=2) → scipy scale=1/lambda=0.5 --

dist = stats.expon(scale=0.5)
results["exponential_lambda2"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "quantiles": {
        "0.50": float(dist.ppf(0.50)),
        "0.90": float(dist.ppf(0.90)),
        "0.99": float(dist.ppf(0.99)),
    },
    "cdf_at": {
        "0.5": float(dist.cdf(0.5)),
        "1.0": float(dist.cdf(1.0)),
        "2.0": float(dist.cdf(2.0)),
    },
}

# -- Gamma(alpha=2, beta=1) → scipy: a=2, scale=1/beta=1 --

dist = stats.gamma(a=2, scale=1.0)
results["gamma_2_1"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "quantiles": {
        "0.50": float(dist.ppf(0.50)),
        "0.95": float(dist.ppf(0.95)),
    },
}

# -- Gamma(alpha=5, beta=2) → scipy: a=5, scale=0.5 --

dist = stats.gamma(a=5, scale=0.5)
results["gamma_5_2"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
}

# -- Beta(alpha=2, beta=5) --

dist = stats.beta(2, 5)
results["beta_2_5"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "quantiles": {
        "0.50": float(dist.ppf(0.50)),
    },
}

# -- Beta(alpha=0.5, beta=0.5) — U-shaped (Jeffreys prior) --

dist = stats.beta(0.5, 0.5)
results["beta_0.5_0.5"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
}

# -- Poisson(lambda=5) --

dist = stats.poisson(mu=5)
results["poisson_5"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "pmf_at": {
        "0": float(dist.pmf(0)),
        "5": float(dist.pmf(5)),
        "10": float(dist.pmf(10)),
    },
}

# -- Poisson(lambda=0.5) --

dist = stats.poisson(mu=0.5)
results["poisson_0.5"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "pmf_at": {
        "0": float(dist.pmf(0)),
        "1": float(dist.pmf(1)),
    },
}

# -- Uniform(0,1) --

results["uniform_0_1"] = {
    "mean": 0.5,
    "var": 1.0 / 12.0,
    "quantiles": {
        "0.25": 0.25,
        "0.50": 0.50,
        "0.75": 0.75,
    },
}

# -- KS test critical values --
# For n=100000 samples, alpha=0.05: D_crit = 1.36 / sqrt(n)

n_samples = 100000
results["ks_critical_values"] = {
    "n": n_samples,
    "alpha_0.05": 1.36 / np.sqrt(n_samples),
    "alpha_0.01": 1.63 / np.sqrt(n_samples),
}

# -- Chi-squared(k=3) --

dist = stats.chi2(df=3)
results["chi2_3"] = {
    "mean": float(dist.mean()),
    "var": float(dist.var()),
    "quantiles": {
        "0.95": float(dist.ppf(0.95)),
    },
}

# -- Save --

with open("research/gold_standard/family_04_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"F04 Oracle: {len(results)} test cases generated")
for name, r in results.items():
    parts = []
    if 'mean' in r:
        parts.append(f"E={r['mean']:.4f}")
    if 'var' in r:
        parts.append(f"V={r['var']:.4f}")
    print(f"  PASS {name}: {', '.join(parts)}")
