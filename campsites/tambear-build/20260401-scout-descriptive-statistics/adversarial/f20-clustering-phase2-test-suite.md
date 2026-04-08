# F20 Clustering — Adversarial Test Suite (Phase 2 Update)

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Files**: `src/clustering.rs`, `src/kmeans.rs`, `src/knn.rs`

---

## C1 [HIGH]: KMeans empty cluster centroid collapses to origin

**Location**: kmeans.rs:94-108 (CUDA kernel)

When a cluster gets zero points, `new_centroids` was zeroed (line 212), so the empty cluster centroid becomes (0,0,...,0). On next iteration this phantom origin-centroid attracts points.

**Test**: K=5 on data concentrated in 2 clusters. After a few iterations, >=1 cluster will be empty → its centroid goes to origin → steals points from real clusters.

**Fix**: Reinitialize empty clusters (pick farthest point from assigned centroid, or copy old centroid).

---

## C2 [HIGH]: KMeans k=0 panics with division by zero

**Location**: kmeans.rs:133, 155

`engine.fit(&data, 10, 2, 0, 100)` → `n / k` where k=0 → integer division panic.

**Fix**: `assert!(k >= 1, "k must be >= 1")`.

---

## C3 [MEDIUM]: Deterministic stride initialization produces duplicate centroids

**Location**: kmeans.rs:154-158

Picks points at indices `0, step, 2*step, ...`. If data is structured/sorted, multiple centroids land in the same region.

## C4 [MEDIUM]: DBSCAN self-distance inflates density by 1

**Location**: clustering.rs:346-354

Diagonal entry (distance to self = 0) always counted. With `min_samples=1`, every point is core. Undocumented.

## C5 [MEDIUM]: DBSCAN border point assignment by index, not by nearest distance

**Location**: clustering.rs:394-403

`break` on first core neighbor found. Different row ordering → different cluster assignment for border points.

## C6 [MEDIUM]: KNN manifold distance labeled as L2Sq

**Location**: knn.rs:182-183

On cache miss, `DistanceMatrix::from_vec(Metric::L2Sq, ...)` even when computing manifold distances.

## C7 [MEDIUM]: clustering_from_distance doesn't validate dist.len() == n*n

**Location**: clustering.rs:339

Public API accepts raw slice without dimension check. OOB or silent wrong results on mismatch.

---

## C8-C11 [LOW]: k=0 in knn_from_distance, negative epsilon, timing div-by-zero, no runtime guard on n
