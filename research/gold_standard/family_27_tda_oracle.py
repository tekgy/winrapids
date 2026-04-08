"""
Gold Standard Oracle: Family 27 — Topological Data Analysis

Generates expected values from ripser/gudhi for comparison with tambear.
Falls back to analytical values for simple cases.

Tests covered:
  - H₀ persistent homology (union-find): pair counts, birth/death values
  - Persistence diagram properties: entropy, total persistence
  - Betti curves: monotonicity, initial count
  - Bottleneck/Wasserstein distances: metric properties

Reference: ripser 0.6+, analytical topology
Usage:
    python research/gold_standard/family_27_tda_oracle.py
"""

import json
import numpy as np

results = {}

# ── H₀ analytical: collinear points ─────────────────────────────────────
# Points on a line: 0--1--2--3 with distances 1, 1, 1
# All merges at d=1 (adjacent edges). 3 finite pairs, 1 infinite.
dist_collinear = [
    [0.0, 1.0, 2.0, 3.0],
    [1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, 0.0, 1.0],
    [3.0, 2.0, 1.0, 0.0],
]

results["h0_collinear_4pt"] = {
    "dist_matrix": dist_collinear,
    "n": 4,
    "n_h0_pairs": 4,
    "n_finite": 3,
    "n_infinite": 1,
    "finite_deaths": [1.0, 1.0, 1.0],
    "note": "4 collinear points, step=1. All merges at d=1."
}

# ── H₀: two well-separated clusters ─────────────────────────────────────
# Cluster A: 3 points mutual dist 0.1
# Cluster B: 3 points mutual dist 0.1
# Inter-cluster: 10.0
results["h0_two_clusters"] = {
    "n": 6,
    "intra_dist": 0.1,
    "inter_dist": 10.0,
    "max_h0_persistence_expected": 10.0,
    "n_features_above_1": 1,
    "note": "Two tight clusters far apart. Max persistence ≈ inter-cluster gap."
}

# ── Persistence entropy (analytical) ─────────────────────────────────────
# n equal-persistence pairs → entropy = ln(n)
for n in [2, 3, 4, 8]:
    results[f"persistence_entropy_{n}_equal"] = {
        "n_pairs": n,
        "persistence_each": 1.0,
        "expected_entropy": float(np.log(n)),
        "note": f"{n} equal pairs → entropy = ln({n}) = {np.log(n):.10f}"
    }

# Single pair → entropy = 0
results["persistence_entropy_1_pair"] = {
    "n_pairs": 1,
    "expected_entropy": 0.0,
    "note": "Single pair: p=1, -1·ln(1)=0"
}

# ── Persistence statistics (exact) ───────────────────────────────────────
# Pairs with persistence 1, 3, 3 (finite only)
pers = [1.0, 3.0, 3.0]
results["persistence_statistics_exact"] = {
    "persistences": pers,
    "count": len(pers),
    "total": float(sum(pers)),
    "max": float(max(pers)),
    "mean": float(np.mean(pers)),
    "std": float(np.std(pers)),  # population std
    "note": "Exact statistics of [1, 3, 3]"
}

# ── Bottleneck distance properties (analytical) ─────────────────────────
results["bottleneck_metric_properties"] = {
    "self_distance": 0.0,
    "triangle_inequality": True,
    "symmetry": True,
    "note": "Bottleneck is a metric on persistence diagrams"
}

# ── Wasserstein distance properties ──────────────────────────────────────
results["wasserstein_metric_properties"] = {
    "self_distance": 0.0,
    "nonnegative": True,
    "note": "1-Wasserstein on persistence diagrams"
}

# ── ripser validation (if available) ─────────────────────────────────────
try:
    from ripser import ripser

    # 3 points: equilateral triangle (side 1)
    X_tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    rips = ripser(X_tri, maxdim=1)

    h0 = rips['dgms'][0]
    h1 = rips['dgms'][1]

    results["ripser_triangle_h0"] = {
        "n_pairs": len(h0),
        "births": [float(b) for b, d in h0],
        "deaths": [float(d) for b, d in h0],
        "note": "ripser H₀ for equilateral triangle"
    }
    if len(h1) > 0:
        results["ripser_triangle_h1"] = {
            "n_pairs": len(h1),
            "births": [float(b) for b, d in h1],
            "deaths": [float(d) for b, d in h1],
            "note": "ripser H₁ for equilateral triangle — should detect 1-cycle"
        }

    # 6-point two-cluster dataset
    X_clust = np.array([
        [0, 0], [0.05, 0.05], [0.1, 0],  # cluster A
        [10, 0], [10.05, 0.05], [10.1, 0],  # cluster B
    ])
    rips2 = ripser(X_clust, maxdim=0)
    h0_2 = rips2['dgms'][0]
    pers_2 = [float(d - b) for b, d in h0_2 if d != np.inf]
    results["ripser_two_clusters_h0"] = {
        "max_persistence": float(max(pers_2)) if pers_2 else 0.0,
        "n_finite_pairs": len(pers_2),
        "note": "Two tight clusters separated by ~10. Max persistence ≈ 10."
    }

except ImportError:
    print("ripser not available — analytical values only")

# ── Write output ─────────────────────────────────────────────────────────
output_path = "research/gold_standard/family_27_expected.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Family 27 oracle: {len(results)} test groups written to {output_path}")
for key in results:
    print(f"  {key}")
