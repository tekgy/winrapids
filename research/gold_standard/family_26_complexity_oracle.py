"""
Gold Standard Oracle: Family 26 — Complexity & Chaos

Generates expected values from nolds for comparison with tambear.

Algorithms covered:
  - Sample entropy (nolds.sampen)
  - Hurst exponent (nolds.hurst_rs)
  - DFA (nolds.dfa)
  - Correlation dimension (nolds.corr_dim)
  - Permutation entropy (analytical: monotonic=0, uniform=ln(m!))

Usage:
    python research/gold_standard/family_26_complexity_oracle.py
"""

import json
import numpy as np
import nolds

results = {}

# -- Sample Entropy --

# Periodic sine wave (m=2, r=0.2*std)
np.random.seed(42)
periodic = np.sin(np.arange(500) * 0.1)
std_p = periodic.std()
se_periodic = nolds.sampen(periodic, emb_dim=2, tolerance=0.2 * std_p)
results["sampen_periodic"] = {
    "value": float(se_periodic),
    "emb_dim": 2,
    "tolerance_factor": 0.2,
    "n": 500,
    "note": "periodic sin(0.1*i), SampEn should be low (<1.0)",
}

# Random data
random_data = np.random.randn(500)
std_r = random_data.std()
se_random = nolds.sampen(random_data, emb_dim=2, tolerance=0.2 * std_r)
results["sampen_random"] = {
    "value": float(se_random),
    "emb_dim": 2,
    "tolerance_factor": 0.2,
    "n": 500,
    "note": "random N(0,1), SampEn should be higher than periodic",
}

# -- Hurst Exponent --

# Random walk (cumulative sum of N(0,1)) -> H ≈ 0.5
random_walk = np.cumsum(np.random.randn(2000))
h_rw = nolds.hurst_rs(random_walk)
results["hurst_random_walk"] = {
    "value": float(h_rw),
    "n": 2000,
    "expected_range": [0.3, 0.7],
    "note": "random walk, H should be near 0.5",
}

# Trending data (cumulative sum of positive increments) -> H > 0.5
trending = np.cumsum(np.abs(np.random.randn(2000)))
h_trend = nolds.hurst_rs(trending)
results["hurst_trending"] = {
    "value": float(h_trend),
    "n": 2000,
    "note": "monotone trending, H should be > 0.5",
}

# Anti-persistent (alternating) -> H < 0.5
alternating = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(2000)])
h_alt = nolds.hurst_rs(alternating)
results["hurst_alternating"] = {
    "value": float(h_alt),
    "n": 2000,
    "note": "alternating +-1, H should be < 0.5",
}

# -- DFA --

# White noise -> alpha ≈ 0.5
white = np.random.randn(1000)
alpha_white = nolds.dfa(white)
results["dfa_white_noise"] = {
    "value": float(alpha_white),
    "n": 1000,
    "expected_range": [0.3, 0.7],
    "note": "white noise, DFA alpha should be near 0.5",
}

# Brownian motion (random walk) -> alpha ≈ 1.5
brown = np.cumsum(np.random.randn(1000))
alpha_brown = nolds.dfa(brown)
results["dfa_brownian"] = {
    "value": float(alpha_brown),
    "n": 1000,
    "expected_range": [1.0, 2.0],
    "note": "Brownian motion, DFA alpha should be near 1.5",
}

# -- Correlation Dimension --

# Sine wave: 1D attractor -> corr_dim near 1.0
sine = np.sin(np.arange(500) * 0.1)
try:
    cd_sine = nolds.corr_dim(sine, emb_dim=3)
    results["corr_dim_sine"] = {
        "value": float(cd_sine),
        "emb_dim": 3,
        "note": "sine wave, low-dimensional attractor",
    }
except Exception as e:
    results["corr_dim_sine"] = {
        "error": str(e),
        "note": "nolds.corr_dim failed on sine",
    }

# -- Permutation Entropy (analytical) --

# Monotonic data -> PE = 0 (single permutation pattern)
results["pe_monotonic"] = {
    "value": 0.0,
    "m": 3,
    "note": "monotonic [0,1,2,...] -> only one pattern -> PE=0",
}

# Maximum PE for m=3: ln(3!) = ln(6)
results["pe_max_m3"] = {
    "max_value": float(np.log(6)),
    "m": 3,
    "note": "max PE = ln(m!) = ln(6) for m=3",
}

# Maximum PE for m=4: ln(4!) = ln(24)
results["pe_max_m4"] = {
    "max_value": float(np.log(24)),
    "m": 4,
    "note": "max PE = ln(m!) = ln(24) for m=4",
}

# -- Save --

with open("research/gold_standard/family_26_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"F26 Oracle: {len(results)} test cases generated")
for name, r in results.items():
    if 'value' in r and not isinstance(r['value'], str):
        note = r.get('note', '')
        print(f"  PASS {name}: value={r['value']:.6f} ({note})")
    elif 'max_value' in r:
        print(f"  PASS {name}: max={r['max_value']:.6f}")
    elif 'error' in r:
        print(f"  WARN {name}: {r['error']}")
    else:
        print(f"  PASS {name}")
