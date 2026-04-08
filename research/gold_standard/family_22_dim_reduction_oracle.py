"""
Gold Standard Oracle: Family 22 — Dimensionality Reduction

Generates expected values from sklearn for comparison with tambear.

Tests covered:
  - PCA: explained variance ratio, components orthogonality, projection
  - Classical MDS: stress, distance preservation
  - NMF: non-negativity, reconstruction error

Reference: sklearn 1.4+, numpy
Note: t-SNE is stochastic — test properties (separation, KL) not exact values.
Usage:
    python research/gold_standard/family_22_dim_reduction_oracle.py
"""

import json
import numpy as np

results = {}

# ── PCA rank-1 data ──────────────────────────────────────────────────────
# Data: x_i = (i, 2i, 3i) for i=1..10. Rank-1.
data_rank1 = np.array([[i, 2*i, 3*i] for i in range(1, 11)], dtype=float)

try:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    transformed = pca.fit_transform(data_rank1)

    results["pca_rank1"] = {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "singular_values": [float(v) for v in pca.singular_values_],
        "components": [[float(v) for v in row] for row in pca.components_],
        "n": 10, "d": 3,
        "note": "Rank-1 data (i, 2i, 3i). First PC should explain ~100%."
    }

    # PCA on random data — variance ratios sum to 1
    np.random.seed(42)
    data_random = np.random.randn(50, 4)
    pca2 = PCA(n_components=4)
    pca2.fit(data_random)
    results["pca_random_4d"] = {
        "explained_variance_ratio": [float(v) for v in pca2.explained_variance_ratio_],
        "singular_values": [float(v) for v in pca2.singular_values_],
        "total_variance_ratio": float(sum(pca2.explained_variance_ratio_)),
        "note": "Random 50×4 data. Variance ratios sum to 1.0"
    }
except ImportError:
    # Analytical fallback
    results["pca_rank1"] = {
        "explained_variance_ratio": [1.0, 0.0, 0.0],
        "note": "analytical: rank-1 data, first PC=100%"
    }

# ── Classical MDS ────────────────────────────────────────────────────────
# 3 points in 2D: (0,0), (1,0), (0,1)
sqrt2 = float(np.sqrt(2))
dist_3pt = [
    [0.0, 1.0, 1.0],
    [1.0, 0.0, sqrt2],
    [1.0, sqrt2, 0.0],
]

try:
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, random_state=42)
    embedding = mds.fit_transform(np.array(dist_3pt))
    results["mds_3point_triangle"] = {
        "dist_matrix": dist_3pt,
        "embedding": [[float(v) for v in row] for row in embedding],
        "stress": float(mds.stress_),
        "note": "3 points forming right triangle. Stress should be ~0 for 2D embedding."
    }

    # Collinear points MDS k=1
    dist_line = [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ]
    mds1 = MDS(n_components=1, dissimilarity='precomputed', normalized_stress=False, random_state=42)
    emb1 = mds1.fit_transform(np.array(dist_line))
    results["mds_collinear_1d"] = {
        "dist_matrix": dist_line,
        "stress": float(mds1.stress_),
        "note": "4 collinear points. 1D MDS stress ~0."
    }
except ImportError:
    results["mds_3point_triangle"] = {
        "dist_matrix": dist_3pt,
        "stress_expected": 0.0,
        "note": "Exact Euclidean distances in 2D. Stress=0 analytically."
    }

# ── NMF ──────────────────────────────────────────────────────────────────
try:
    from sklearn.decomposition import NMF

    V = np.array([
        [3.0, 1.0, 0.5],
        [0.0, 2.0, 1.5],
        [1.0, 0.0, 3.0],
        [2.0, 1.0, 0.0],
    ])

    nmf_k2 = NMF(n_components=2, init='random', random_state=42, max_iter=500)
    W = nmf_k2.fit_transform(V)
    H = nmf_k2.components_
    error = nmf_k2.reconstruction_err_

    results["nmf_4x3_rank2"] = {
        "V": [[float(v) for v in row] for row in V],
        "reconstruction_error": float(error),
        "W_nonneg": bool(np.all(W >= 0)),
        "H_nonneg": bool(np.all(H >= 0)),
        "note": "NMF rank-2 of 4×3 matrix. All W,H >= 0."
    }

    # Rank-1 exact: V = [1,2,3]' × [1,1,1]
    V_rank1 = np.array([[1,1,1],[2,2,2],[3,3,3]], dtype=float)
    nmf_k1 = NMF(n_components=1, init='random', random_state=42, max_iter=500)
    nmf_k1.fit(V_rank1)
    results["nmf_rank1_exact"] = {
        "reconstruction_error": float(nmf_k1.reconstruction_err_),
        "note": "Rank-1 NMF of rank-1 data. Error should be ~0."
    }
except ImportError:
    pass

# ── Write output ─────────────────────────────────────────────────────────
output_path = "research/gold_standard/family_22_expected.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Family 22 oracle: {len(results)} test groups written to {output_path}")
for key in results:
    print(f"  {key}")
