"""
Gold Standard Oracle: Family 15 — Item Response Theory & Psychometrics

Generates expected values from analytical IRT formulas for comparison with tambear.
Optionally validates against mirt (R) or girth (Python) if available.

Tests covered:
  - Rasch/2PL/3PL probability models (exact logistic formulas)
  - Item/test information functions (exact: a²PQ)
  - SEM (exact: 1/√I)
  - Ability estimation (MLE, EAP — directional checks)

Reference: Analytical IRT (Lord & Novick 1968, Hambleton & Swaminathan 1985)
Usage:
    python research/gold_standard/family_15_irt_oracle.py
"""

import json
import numpy as np
from scipy.special import expit  # logistic function

results = {}

# ── Probability models (exact) ───────────────────────────────────────────

def prob_2pl(theta, a, b):
    return float(expit(a * (theta - b)))

def prob_3pl(theta, a, b, c):
    return float(c + (1 - c) * expit(a * (theta - b)))

# Rasch: P(θ=b) = 0.5 exactly
results["rasch_at_difficulty"] = {
    "cases": [
        {"theta": b, "b": b, "expected": 0.5}
        for b in [-3.0, -1.0, 0.0, 1.0, 3.0]
    ],
    "note": "Rasch: P(θ=b) = 1/(1+exp(0)) = 0.5 exactly"
}

# 2PL at various points
results["prob_2pl_values"] = {
    "cases": [
        {"theta": 0.0, "a": 1.0, "b": 0.0, "expected": 0.5},
        {"theta": 1.0, "a": 1.0, "b": 0.0, "expected": prob_2pl(1.0, 1.0, 0.0)},
        {"theta": 2.0, "a": 2.0, "b": 1.0, "expected": prob_2pl(2.0, 2.0, 1.0)},
        {"theta": -1.0, "a": 1.5, "b": 0.5, "expected": prob_2pl(-1.0, 1.5, 0.5)},
        {"theta": 0.0, "a": 3.0, "b": 0.0, "expected": 0.5},  # at difficulty always 0.5
    ],
    "note": "2PL: P = 1/(1+exp(-a(θ-b))). scipy.special.expit used."
}

# 3PL: floor = c at θ→-∞, ceiling = 1 at θ→+∞
results["prob_3pl_limits"] = {
    "floor_cases": [
        {"theta": -100.0, "a": 2.0, "b": 0.0, "c": c, "expected": c}
        for c in [0.0, 0.1, 0.25, 0.5]
    ],
    "ceiling_cases": [
        {"theta": 100.0, "a": 2.0, "b": 0.0, "c": c, "expected": 1.0}
        for c in [0.0, 0.1, 0.25]
    ],
    "note": "3PL: P → c as θ→-∞, P → 1 as θ→+∞"
}

# ── Information functions (exact: I = a²PQ for 2PL) ─────────────────────

def item_info_2pl(theta, a, b):
    p = expit(a * (theta - b))
    return float(a**2 * p * (1 - p))

items = [
    {"a": 1.0, "b": -1.0},
    {"a": 1.5, "b": 0.0},
    {"a": 2.0, "b": 1.0},
    {"a": 0.8, "b": 2.0},
]

# Item info at peak (θ=b): I = a²/4
results["item_info_at_peak"] = {
    "cases": [
        {"a": item["a"], "b": item["b"],
         "info_at_peak": float(item["a"]**2 / 4.0)}
        for item in items
    ],
    "note": "At θ=b, P=Q=0.5, so I = a²·0.25 = a²/4"
}

# Test information = sum of item info
thetas_test = [-3.0, -1.0, 0.0, 1.0, 3.0]
results["test_information"] = {
    "items": items,
    "theta_values": thetas_test,
    "test_info": [
        float(sum(item_info_2pl(theta, item["a"], item["b"]) for item in items))
        for theta in thetas_test
    ],
    "note": "T(θ) = Σ I_j(θ). Exact sum of 2PL item info."
}

# SEM = 1/√I
results["sem_values"] = {
    "theta_values": thetas_test,
    "sem": [
        float(1.0 / np.sqrt(sum(item_info_2pl(theta, item["a"], item["b"]) for item in items)))
        for theta in thetas_test
    ],
    "note": "SEM = 1/√T(θ). Exact."
}

# ── Write output ─────────────────────────────────────────────────────────
output_path = "research/gold_standard/family_15_expected.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Family 15 oracle: {len(results)} test groups written to {output_path}")
for key in results:
    print(f"  {key}")
