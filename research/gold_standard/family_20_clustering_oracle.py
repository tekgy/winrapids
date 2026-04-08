"""
Gold Standard Oracle: Family 20 — Clustering (DBSCAN)
Generates expected values for tambear validation.

Tests:
  - Two well-separated clusters (sklearn DBSCAN comparison)
  - Three clusters with noise
  - Single cluster (all points close)
  - All noise (high min_samples)
  - Chain-shaped clusters (DBSCAN strength vs k-means)
"""

import json
import numpy as np
from sklearn.cluster import DBSCAN

results = {}

# ===============================================================
# 1. Two well-separated 2D clusters
# ===============================================================

np.random.seed(42)
cluster1 = np.random.randn(20, 2) * 0.3 + np.array([0.0, 0.0])
cluster2 = np.random.randn(20, 2) * 0.3 + np.array([5.0, 5.0])
data_2c = np.vstack([cluster1, cluster2])

eps_2c = 1.0
min_samples_2c = 3
db = DBSCAN(eps=eps_2c, min_samples=min_samples_2c, metric='euclidean')
labels_2c = db.fit_predict(data_2c)
n_clusters_2c = len(set(labels_2c) - {-1})
n_noise_2c = int(np.sum(labels_2c == -1))

results["two_clusters"] = {
    "data": data_2c.flatten().tolist(),
    "n": len(data_2c),
    "d": 2,
    "eps": eps_2c,
    "min_samples": min_samples_2c,
    "labels": labels_2c.tolist(),
    "n_clusters": n_clusters_2c,
    "n_noise": n_noise_2c,
}
print("Two clusters: n_clusters=%d, n_noise=%d" % (n_clusters_2c, n_noise_2c))

# ===============================================================
# 2. Three clusters with noise points
# ===============================================================

np.random.seed(99)
c1 = np.random.randn(15, 2) * 0.3 + np.array([0.0, 0.0])
c2 = np.random.randn(15, 2) * 0.3 + np.array([4.0, 0.0])
c3 = np.random.randn(15, 2) * 0.3 + np.array([2.0, 4.0])
noise_pts = np.random.uniform(-2, 7, (5, 2))  # scattered noise
data_3c = np.vstack([c1, c2, c3, noise_pts])

eps_3c = 1.0
min_samples_3c = 3
db3 = DBSCAN(eps=eps_3c, min_samples=min_samples_3c, metric='euclidean')
labels_3c = db3.fit_predict(data_3c)
n_clusters_3c = len(set(labels_3c) - {-1})
n_noise_3c = int(np.sum(labels_3c == -1))

results["three_clusters_noise"] = {
    "data": data_3c.flatten().tolist(),
    "n": len(data_3c),
    "d": 2,
    "eps": eps_3c,
    "min_samples": min_samples_3c,
    "labels": labels_3c.tolist(),
    "n_clusters": n_clusters_3c,
    "n_noise": n_noise_3c,
}
print("Three clusters + noise: n_clusters=%d, n_noise=%d" % (n_clusters_3c, n_noise_3c))

# ===============================================================
# 3. Single tight cluster
# ===============================================================

np.random.seed(55)
data_1c = np.random.randn(30, 2) * 0.2
eps_1c = 1.0
min_samples_1c = 3
db1 = DBSCAN(eps=eps_1c, min_samples=min_samples_1c, metric='euclidean')
labels_1c = db1.fit_predict(data_1c)
n_clusters_1c = len(set(labels_1c) - {-1})
n_noise_1c = int(np.sum(labels_1c == -1))

results["single_cluster"] = {
    "data": data_1c.flatten().tolist(),
    "n": len(data_1c),
    "d": 2,
    "eps": eps_1c,
    "min_samples": min_samples_1c,
    "labels": labels_1c.tolist(),
    "n_clusters": n_clusters_1c,
    "n_noise": n_noise_1c,
}
print("Single cluster: n_clusters=%d, n_noise=%d" % (n_clusters_1c, n_noise_1c))

# ===============================================================
# 4. All noise (min_samples too high)
# ===============================================================

np.random.seed(77)
data_noise = np.random.randn(10, 2) * 2.0
eps_n = 0.5
min_samples_n = 8  # too high for sparse data
db_n = DBSCAN(eps=eps_n, min_samples=min_samples_n, metric='euclidean')
labels_n = db_n.fit_predict(data_noise)
n_clusters_n = len(set(labels_n) - {-1})

results["all_noise"] = {
    "data": data_noise.flatten().tolist(),
    "n": len(data_noise),
    "d": 2,
    "eps": eps_n,
    "min_samples": min_samples_n,
    "labels": labels_n.tolist(),
    "n_clusters": n_clusters_n,
    "all_noise": bool(n_clusters_n == 0),
}
print("All noise: n_clusters=%d, all_noise=%s" % (n_clusters_n, n_clusters_n == 0))

# ===============================================================
# 5. Deterministic micro-dataset for exact label comparison
# ===============================================================
# Small enough that we can verify exact labels

data_micro = np.array([
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [5.0, 5.0],
    [5.1, 5.0],
    [5.0, 5.1],
])
eps_micro = 0.5
min_samples_micro = 2
db_micro = DBSCAN(eps=eps_micro, min_samples=min_samples_micro, metric='euclidean')
labels_micro = db_micro.fit_predict(data_micro)

# Points 0,1,2 should be one cluster, 3,4,5 should be another
# (labels may be 0,1 or 1,0 but must be two distinct non-noise clusters)
results["micro_exact"] = {
    "data": data_micro.flatten().tolist(),
    "n": 6,
    "d": 2,
    "eps": eps_micro,
    "min_samples": min_samples_micro,
    "labels": labels_micro.tolist(),
    "n_clusters": int(len(set(labels_micro) - {-1})),
    "cluster_a": labels_micro[0].tolist(),  # label for first group
    "cluster_b": labels_micro[3].tolist(),  # label for second group
}
print("Micro exact: labels=%s" % labels_micro.tolist())

# ===============================================================
# Save
# ===============================================================

with open("research/gold_standard/family_20_clustering_expected.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved family_20_clustering_expected.json")
