"""
Scale Ladder Benchmark: DBSCAN + KNN (sklearn)

Measures the cost of DBSCAN alone, KNN alone, and DBSCAN+KNN in Python
using sklearn. This is the competitor baseline — sklearn cannot share
the distance matrix between algorithms without manual precomputation.

The "manual sharing" path precomputes the distance matrix once and passes
it to both algorithms — this is what a skilled Python user would do.
tambear does this automatically via TamSession.

Usage:
    python research/gold_standard/scale_ladder_dbscan_knn.py
"""

import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

scales = [
    ("100",    100),
    ("500",    500),
    ("1K",     1_000),
    ("2K",     2_000),
    ("5K",     5_000),
    ("10K",    10_000),
    ("20K",    20_000),
    ("50K",    50_000),
]

d = 3
k_clusters = 3
k_neighbors = 5
eps = 3.0
min_pts = 3

print("=" * 90)
print("SCALE LADDER: DBSCAN + KNN (sklearn)")
print("=" * 90)
print(f"{'Scale':>6}  {'DBSCAN(s)':>10}  {'KNN(s)':>10}  {'Manual(s)':>10}  {'Naive(s)':>10}  {'DistMB':>8}")
print("-" * 90)

for label, n in scales:
    try:
        # Generate blob data
        np.random.seed(42)
        centers = np.array([[i * 10.0] * d for i in range(k_clusters)])
        data = np.vstack([
            centers[i % k_clusters] + np.random.randn(d) for i in range(n)
        ])

        dist_mb = n * n * 8 / 1e6

        if dist_mb > 4000:
            print(f"{label:>6}  *** distance matrix would be {dist_mb:.0f} MB -- skipping ***")
            continue

        # --- Naive: DBSCAN alone (computes distance internally) ---
        t0 = time.perf_counter()
        db = DBSCAN(eps=eps, min_samples=min_pts, metric='euclidean').fit(data)
        t_dbscan = time.perf_counter() - t0

        # --- Naive: KNN alone (computes distance internally) ---
        t0 = time.perf_counter()
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(data)
        knn_dist, knn_idx = nn.kneighbors(data)
        t_knn = time.perf_counter() - t0

        t_naive = t_dbscan + t_knn

        # --- Manual sharing: precompute distance matrix, pass to both ---
        t0 = time.perf_counter()
        dist_matrix = pairwise_distances(data, metric='euclidean')
        db2 = DBSCAN(eps=eps, min_samples=min_pts, metric='precomputed').fit(dist_matrix)
        nn2 = NearestNeighbors(n_neighbors=k_neighbors, metric='precomputed').fit(dist_matrix)
        knn_dist2, knn_idx2 = nn2.kneighbors(dist_matrix)
        t_manual = time.perf_counter() - t0

        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

        print(f"{label:>6}  {t_dbscan:>10.4f}  {t_knn:>10.4f}  {t_manual:>10.4f}  {t_naive:>10.4f}  {dist_mb:>8.1f}")
        print(f"        clusters={n_clusters}, knn_check: point0={knn_idx[0][:3].tolist()}")

        del data, dist_matrix

    except MemoryError:
        print(f"{label:>6}  *** MemoryError: distance matrix = {n * n * 8 / 1e9:.1f} GB ***")
        break
    except Exception as e:
        print(f"{label:>6}  *** Error: {e} ***")
        break

print("=" * 90)
print()
print("Naive = DBSCAN + KNN computed independently (2x distance computation).")
print("Manual = precompute distance once, pass as precomputed metric.")
print("tambear automates the Manual pattern via TamSession.")
