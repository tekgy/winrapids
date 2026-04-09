"""
Gold Standard Oracle: Shapiro-Wilk Comprehensive Verification

Tests Shapiro-Wilk W statistic AND p-value against scipy.stats.shapiro
across multiple sample sizes: n=10, 20, 30, 50, 100, 500, 1000, 5000.

For each n, generates:
  - Normal(0,1) data → p should be > 0.05 most of the time
  - Uniform(0,1) data → p should be < 0.05 for large n
  - Fixed-seed data for reproducible comparison

This oracle provides the ground truth for verifying the Royston 1995
p-value approximation after the sigma formula fix.

Usage:
    python research/gold_standard/shapiro_wilk_comprehensive_oracle.py
"""

import json
import numpy as np
from scipy import stats

results = {}

sample_sizes = [10, 20, 30, 50, 100, 500, 1000, 5000]

# ─── Normal data at each sample size ─────────────────────────────────────

for n in sample_sizes:
    np.random.seed(42 + n)  # reproducible per size
    data = np.random.normal(0, 1, n)
    w_stat, p_val = stats.shapiro(data)

    results[f"normal_n{n}"] = {
        "n": n,
        "distribution": "normal",
        "seed": 42 + n,
        "W": float(w_stat),
        "p_value": float(p_val),
        "reject_005": float(p_val) < 0.05,
        "data_first_10": data[:10].tolist(),
        "data_mean": float(np.mean(data)),
        "data_std": float(np.std(data, ddof=1)),
    }

# ─── Uniform data at each sample size ────────────────────────────────────

for n in sample_sizes:
    np.random.seed(100 + n)
    data = np.random.uniform(0, 1, n)
    w_stat, p_val = stats.shapiro(data)

    results[f"uniform_n{n}"] = {
        "n": n,
        "distribution": "uniform",
        "seed": 100 + n,
        "W": float(w_stat),
        "p_value": float(p_val),
        "reject_005": float(p_val) < 0.05,
        "data_first_10": data[:10].tolist(),
    }

# ─── Exponential data (strongly non-normal, right-skewed) ────────────────

for n in [20, 50, 100, 500]:
    np.random.seed(200 + n)
    data = np.random.exponential(1, n)
    w_stat, p_val = stats.shapiro(data)

    results[f"exponential_n{n}"] = {
        "n": n,
        "distribution": "exponential",
        "seed": 200 + n,
        "W": float(w_stat),
        "p_value": float(p_val),
        "reject_005": float(p_val) < 0.05,
    }

# ─── Known small-sample exact values ─────────────────────────────────────

# n=3: smallest valid case
data3 = [1.0, 2.0, 3.0]
w3, p3 = stats.shapiro(data3)
results["exact_n3"] = {
    "data": data3,
    "W": float(w3),
    "p_value": float(p3),
}

# n=5: known coefficients
data5 = [1.0, 2.0, 3.0, 4.0, 5.0]
w5, p5 = stats.shapiro(data5)
results["exact_n5"] = {
    "data": data5,
    "W": float(w5),
    "p_value": float(p5),
}

# n=5: perfectly normal-looking (symmetric)
data5_sym = [-2.0, -1.0, 0.0, 1.0, 2.0]
w5s, p5s = stats.shapiro(data5_sym)
results["exact_n5_symmetric"] = {
    "data": data5_sym,
    "W": float(w5s),
    "p_value": float(p5s),
}

# ─── W statistic properties ──────────────────────────────────────────────

# W for constant data should be 1.0 (degenerate)
results["constant_data"] = {
    "data": [5.0] * 20,
    "note": "constant data: W should be 1.0 (or NaN), p should be 1.0",
}

# W is always in (0, 1]
# W closer to 1 = more normal-like

# ─── Summary table ───────────────────────────────────────────────────────

summary = {}
for key, val in results.items():
    if isinstance(val, dict) and "W" in val and "p_value" in val:
        summary[key] = {
            "n": val.get("n", len(val.get("data", []))),
            "W": val["W"],
            "p": val["p_value"],
        }
results["summary"] = summary

# ─── Save ─────────────────────────────────────────────────────────────────

with open("research/gold_standard/shapiro_wilk_comprehensive_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Shapiro-Wilk Comprehensive Oracle: {len(results)} test cases generated")
print()
print(f"{'Test':<25} {'n':>5} {'W':>10} {'p-value':>12} {'Reject?':>8}")
print("-" * 65)
for key in sorted(results.keys()):
    val = results[key]
    if isinstance(val, dict) and "W" in val and "p_value" in val:
        n = val.get("n", len(val.get("data", [])))
        reject = "YES" if val.get("reject_005", val["p_value"] < 0.05) else "no"
        print(f"{key:<25} {n:>5} {val['W']:>10.6f} {val['p_value']:>12.6e} {reject:>8}")
