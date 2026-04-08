"""
Gold Standard Oracle: Existing Tambear Algorithms

Verifies algorithms already implemented in tambear against scipy/sklearn.
For each algorithm, generates expected outputs on identical synthetic data.

Algorithms covered:
  - Linear Regression (vs sklearn LinearRegression)
  - Logistic Regression (vs sklearn LogisticRegression)
  - DBSCAN (vs sklearn DBSCAN)
  - KMeans (vs sklearn KMeans)
  - KNN (vs sklearn NearestNeighbors)
  - Softmax (vs scipy softmax)
  - Matrix multiply / dot product (vs numpy)
  - L2 distance matrix (vs scipy cdist)

Usage:
    python research/gold_standard/existing_algorithms_oracle.py
"""

import json
import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist, squareform
from scipy.special import softmax as scipy_softmax
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import sys


def oracle_linear_regression():
    """Gold standard for tambear::train::linear::fit()"""
    np.random.seed(42)
    results = []

    # Test 1: Perfect linear (same as tambear's test)
    n, d = 100, 2
    x = np.zeros((n, d))
    y = np.zeros(n)
    for i in range(n):
        x1 = i / 10.0
        x2 = np.sin(float(i))
        x[i, 0] = x1
        x[i, 1] = x2
        y[i] = 3.0 * x1 + 2.0 * x2 + 1.0

    model = LinearRegression().fit(x, y)
    results.append({
        "test": "perfect_linear_n100_d2",
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(x, y)),
        "true_coefficients": [3.0, 2.0],
        "true_intercept": 1.0,
    })

    # Test 2: Noisy linear
    np.random.seed(42)
    n, d = 500, 3
    x = np.random.randn(n, d)
    true_beta = np.array([2.5, -1.3, 0.7])
    y = x @ true_beta + 4.2 + np.random.randn(n) * 0.5

    model = LinearRegression().fit(x, y)
    preds = model.predict(x)
    ss_res = np.sum((y - preds) ** 2)
    results.append({
        "test": "noisy_linear_n500_d3",
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(x, y)),
        "rmse_biased": float(np.sqrt(ss_res / n)),  # tambear uses n
        "rmse_unbiased": float(np.sqrt(ss_res / (n - d - 1))),  # sklearn convention
        "true_coefficients": true_beta.tolist(),
        "true_intercept": 4.2,
    })

    # Test 3: Multi-scale features (tests numerical conditioning)
    np.random.seed(42)
    n, d = 200, 3
    x = np.zeros((n, d))
    for i in range(n):
        x[i, 0] = i / 20.0          # range 0..10
        x[i, 1] = np.sin(i * 0.1)   # range -1..1
        x[i, 2] = np.cos(i * 0.07) * 500.0  # range -500..500
    y = 2.5 * x[:, 0] - 1.3 * x[:, 1] + 0.01 * x[:, 2] + 4.2

    model = LinearRegression().fit(x, y)
    results.append({
        "test": "multi_scale_n200_d3",
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(x, y)),
        "note": "same data as tambear fit_session_matches_fit test",
    })

    # Test 4: Near-collinear (adversarial — condition number ~1e8)
    np.random.seed(42)
    n, d = 100, 3
    x = np.random.randn(n, d)
    x[:, 2] = x[:, 0] + 1e-8 * np.random.randn(n)  # nearly collinear with col 0
    y = x[:, 0] + x[:, 1] + 1.0

    model = LinearRegression().fit(x, y)
    cond = np.linalg.cond(x.T @ x)
    results.append({
        "test": "near_collinear_n100_d3",
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "r_squared": float(model.score(x, y)),
        "xtx_condition_number": float(cond),
        "note": "near-collinear: col2 ≈ col0 + noise(1e-8). High condition number.",
    })

    return {"algorithm": "linear_regression", "oracle": "sklearn.LinearRegression", "tests": results}


def oracle_logistic_regression():
    """Gold standard for tambear::train::logistic::fit()"""
    results = []

    # Test 1: Linearly separable (same setup as tambear test)
    n, d = 200, 2
    x = np.zeros((n, d))
    y = np.zeros(n)
    for i in range(n):
        label = 0.0 if i < n // 2 else 1.0
        cx = -2.0 if label == 0.0 else 2.0
        offset = (i % (n // 2)) / (n / 2) - 0.5
        x[i, 0] = cx + offset * 0.5
        x[i, 1] = cx + offset * 0.3
        y[i] = label

    # sklearn with no regularization (C=very large)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    results.append({
        "test": "linearly_separable_n200_d2",
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "accuracy": float(model.score(x, y)),
        "mean_proba_class0": float(np.mean(proba[:n//2])),
        "mean_proba_class1": float(np.mean(proba[n//2:])),
        "note": "sklearn uses L-BFGS, tambear uses vanilla GD — coefficients may differ but accuracy should match",
    })

    # Test 2: 1D threshold
    n, d = 100, 1
    x = np.linspace(-5, 5, n).reshape(-1, 1)
    y = (x.ravel() >= 0).astype(float)

    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    results.append({
        "test": "1d_threshold_n100",
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "accuracy": float(model.score(x, y)),
        "proba_at_zero": float(proba[n//2]),
        "note": "probability at x=0 should be ~0.5",
    })

    return {"algorithm": "logistic_regression", "oracle": "sklearn.LogisticRegression", "tests": results}


def oracle_dbscan():
    """Gold standard for tambear::clustering::ClusteringEngine::dbscan()"""
    results = []

    # Test 1: Two tight clusters (same as tambear test)
    data = np.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
        [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],
    ])
    db = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(data)
    results.append({
        "test": "two_tight_clusters_n6",
        "labels": db.labels_.tolist(),
        "n_clusters": len(set(db.labels_)) - (1 if -1 in db.labels_ else 0),
        "n_noise": int(np.sum(db.labels_ == -1)),
        "n_core": len(db.core_sample_indices_),
        "note": "tambear uses L2Sq (squared distances), so eps comparison is on squared values",
    })

    # Test 2: Isolated point is noise
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [100.0, 100.0],
    ])
    db = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(data)
    results.append({
        "test": "isolated_noise_n3",
        "labels": db.labels_.tolist(),
        "n_clusters": len(set(db.labels_)) - (1 if -1 in db.labels_ else 0),
        "n_noise": int(np.sum(db.labels_ == -1)),
    })

    # Test 3: All one cluster
    data = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0],
    ])
    db = DBSCAN(eps=1.0, min_samples=2, metric='euclidean').fit(data)
    results.append({
        "test": "single_cluster_n4",
        "labels": db.labels_.tolist(),
        "n_clusters": len(set(db.labels_)) - (1 if -1 in db.labels_ else 0),
    })

    # Test 4: Larger dataset with noise
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) * 0.3
    cluster2 = np.random.randn(50, 2) * 0.3 + [5, 5]
    noise = np.random.uniform(-2, 8, (10, 2))
    data = np.vstack([cluster1, cluster2, noise])
    db = DBSCAN(eps=0.8, min_samples=5, metric='euclidean').fit(data)
    results.append({
        "test": "two_clusters_with_noise_n110",
        "n_clusters": len(set(db.labels_)) - (1 if -1 in db.labels_ else 0),
        "n_noise": int(np.sum(db.labels_ == -1)),
        "n_core": len(db.core_sample_indices_),
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(db.labels_, return_counts=True))},
    })

    return {"algorithm": "dbscan", "oracle": "sklearn.cluster.DBSCAN", "tests": results}


def oracle_kmeans():
    """Gold standard for tambear::kmeans::KMeansEngine::fit()"""
    results = []

    # Test 1: Two well-separated clusters
    data = np.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
        [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],
    ])
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(data)
    results.append({
        "test": "two_clusters_n6",
        "labels": km.labels_.tolist(),
        "centroids": km.cluster_centers_.tolist(),
        "inertia": float(km.inertia_),
        "n_clusters": 2,
        "note": "label assignment may differ (0/1 vs 1/0) — compare cluster membership, not label values",
    })

    return {"algorithm": "kmeans", "oracle": "sklearn.cluster.KMeans", "tests": results}


def oracle_knn():
    """Gold standard for tambear::knn::KnnEngine"""
    results = []

    # Test 1: Basic KNN (same as tambear test)
    data = np.array([
        [0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0],
    ])
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(data)
    distances, indices = nn.kneighbors(data)
    # Convert to squared L2 for comparison with tambear (which uses L2Sq)
    sq_distances = distances ** 2
    results.append({
        "test": "basic_knn_n4_k2",
        "indices": indices.tolist(),
        "distances_l2": distances.tolist(),
        "distances_l2sq": sq_distances.tolist(),
        "note": "tambear uses L2Sq metric internally",
    })

    return {"algorithm": "knn", "oracle": "sklearn.neighbors.NearestNeighbors", "tests": results}


def oracle_softmax():
    """Gold standard for tambear::accumulate::softmax()"""
    results = []

    # Test 1: Known values (same as tambear test)
    x = np.array([1.0, 2.0, 3.0])
    s = scipy_softmax(x)
    results.append({
        "test": "known_values_3",
        "input": x.tolist(),
        "output": s.tolist(),
        "sum": float(np.sum(s)),
    })

    # Test 2: Single element
    x = np.array([42.0])
    s = scipy_softmax(x)
    results.append({
        "test": "single_element",
        "input": x.tolist(),
        "output": s.tolist(),
    })

    # Test 3: Large values (numerical stability)
    x = np.array([1000.0, 1001.0, 1002.0])
    s = scipy_softmax(x)
    results.append({
        "test": "large_values",
        "input": x.tolist(),
        "output": s.tolist(),
        "sum": float(np.sum(s)),
        "note": "naive exp() would overflow; log-sum-exp trick required",
    })

    # Test 4: Negative large values
    x = np.array([-1000.0, -999.0, -998.0])
    s = scipy_softmax(x)
    results.append({
        "test": "large_negative_values",
        "input": x.tolist(),
        "output": s.tolist(),
        "sum": float(np.sum(s)),
    })

    # Test 5: All zeros (uniform)
    x = np.zeros(5)
    s = scipy_softmax(x)
    results.append({
        "test": "all_zeros_uniform",
        "input": x.tolist(),
        "output": s.tolist(),
        "expected_each": 0.2,
    })

    return {"algorithm": "softmax", "oracle": "scipy.special.softmax", "tests": results}


def oracle_dot_product():
    """Gold standard for TiledEngine DotProduct"""
    results = []

    # Test 1: 2x3 * 3x2 (same as tambear test)
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
    c = a @ b
    results.append({
        "test": "2x3_times_3x2",
        "a": a.tolist(),
        "b": b.tolist(),
        "result": c.tolist(),
    })

    # Test 2: Identity
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    I = np.eye(2)
    results.append({
        "test": "identity_multiply",
        "a": a.tolist(),
        "result": (a @ I).tolist(),
    })

    # Test 3: Larger random (for precision testing)
    np.random.seed(42)
    a = np.random.randn(50, 30)
    b = np.random.randn(30, 20)
    c = a @ b
    results.append({
        "test": "random_50x30_times_30x20",
        "a_flat": a.ravel().tolist(),
        "b_flat": b.ravel().tolist(),
        "result_flat": c.ravel().tolist(),
        "m": 50, "n": 20, "k": 30,
    })

    return {"algorithm": "dot_product", "oracle": "numpy.matmul", "tests": results}


def oracle_distance_matrix():
    """Gold standard for TiledEngine DistanceOp (L2 squared)"""
    results = []

    # Test 1: Self-distance (same as tambear test)
    data = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
    dist = cdist(data, data, metric='sqeuclidean')
    results.append({
        "test": "self_distance_3x2",
        "data": data.tolist(),
        "distance_matrix_l2sq": dist.tolist(),
        "diagonal_zeros": bool(np.allclose(np.diag(dist), 0)),
        "symmetric": bool(np.allclose(dist, dist.T)),
    })

    # Test 2: Different sets
    a = np.array([[0, 0], [1, 0]], dtype=np.float64)
    b = np.array([[0, 1], [1, 1], [2, 2]], dtype=np.float64)
    dist = cdist(a, b, metric='sqeuclidean')
    results.append({
        "test": "cross_distance_2x3",
        "a": a.tolist(),
        "b": b.tolist(),
        "distance_matrix_l2sq": dist.tolist(),
    })

    return {"algorithm": "l2_distance_matrix", "oracle": "scipy.spatial.distance.cdist", "tests": results}


def run_all_oracles():
    """Run all gold standard oracles."""
    all_results = [
        oracle_linear_regression(),
        oracle_logistic_regression(),
        oracle_dbscan(),
        oracle_kmeans(),
        oracle_knn(),
        oracle_softmax(),
        oracle_dot_product(),
        oracle_distance_matrix(),
    ]

    print(json.dumps(all_results, indent=2, allow_nan=True))

    # Summary
    total_tests = sum(len(r["tests"]) for r in all_results)
    print(f"\n{len(all_results)} algorithms, {total_tests} test cases", file=sys.stderr)
    for r in all_results:
        print(f"  {r['algorithm']}: {len(r['tests'])} tests (oracle: {r['oracle']})", file=sys.stderr)


if __name__ == "__main__":
    run_all_oracles()
