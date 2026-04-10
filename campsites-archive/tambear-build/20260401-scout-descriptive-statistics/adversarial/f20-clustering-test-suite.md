# Family 20: Clustering — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/clustering.rs`, `crates/tambear/src/kmeans.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| DBSCAN (full) | clustering.rs:127-151 | OK |
| DBSCAN density | clustering.rs:346-354 | OK (self-count matches sklearn convention) |
| DBSCAN union-find | clustering.rs:360-372 | OK (path halving) |
| DBSCAN border assignment | clustering.rs:394-403 | OK (greedy = standard) |
| Session-aware DBSCAN | clustering.rs:278-306 | OK (sharing pattern correct) |
| KMeans assign kernel | kmeans.rs:41-69 | OK (fused distance+argmin) |
| KMeans update kernel | kmeans.rs:74-108 | OK (atomicAdd + normalize) |
| KMeans convergence | kmeans.rs:199-207 | OK (label stability, immune to atomicAdd noise) |
| KMeans initialization | kmeans.rs:154-158 | **MEDIUM** (sorted data vulnerability) |
| KMeans empty cluster | kmeans.rs:102-107 | **MEDIUM** (0-centroid trap) |

---

## Finding F20-1: KMeans Empty Cluster Degeneracy (MEDIUM)

**Bug**: When a cluster loses all points, its centroid stays at `(0,0,...,0)` from the zeroed accumulator buffer. If `count == 0`, the normalize kernel leaves the centroid at 0. In the next assignment step, this ghost centroid at origin may steal points from a legitimate cluster, causing oscillation.

**Impact**: For data not centered at origin, the empty centroid stays at origin forever (nothing attracts to it). For data centered near origin, oscillation between iterations.

**Fix**: Standard approaches:
- Re-initialize empty centroids from the farthest point in the largest cluster
- Split the largest cluster when one empties
- Or: detect and restart with a different initialization

---

## Finding F20-2: KMeans Initialization Not Robust (MEDIUM)

**Bug**: `step = n / k`, picks every `step`-th point. For sorted data (e.g., stock prices by time), initial centroids are all from the same value range. This causes:
- Many iterations to converge (or non-convergence within max_iter)
- Suboptimal local minimum
- Empty clusters (if multiple centroids start in the same natural cluster)

**Fix**: K-means++ initialization (O(nk) on CPU, then upload). Pick first centroid uniformly random, subsequent centroids proportional to squared distance from nearest existing centroid.

---

## Positive Findings

**Label-stability convergence is excellent.** Discrete comparison immune to f32 atomicAdd ordering noise. A point either changes cluster or it doesn't — no floating-point comparison involved. This is a genuinely clever design choice.

**DBSCAN is clean.** Union-find with path halving, proper core/border/noise classification, session-aware sharing. The distance matrix sharing infrastructure is architecturally sound.

**NaN handling is reasonable.** NaN distances → NaN ≤ ε is false → point is noise. Silent but arguably correct (NaN data shouldn't cluster).

---

## Test Vectors

### TV-F20-KM-01: Empty cluster trap
```
data = 3 clusters at (100,0), (0,100), (-100,-100)
k = 4 (one more than natural clusters)
Expected: 4th centroid should not stay at (0,0,0)
Currently: may trap at origin if initialization picks 4 points from 3 clusters
```

### TV-F20-KM-02: Sorted data initialization
```
data = [1.0, 1.1, 1.2, ..., 999.0, 999.1, 999.2] (sorted)
k = 3
Expected: converges to ~3 equal clusters spanning the range
Test: compare convergence speed with shuffled data
```

### TV-F20-DB-01: NaN in data
```
data = [[0,0], [0.1,0], [NaN,NaN], [5,5], [5.1,5]]
epsilon=0.5, min_samples=2
Expected: NaN point is noise, 2 clusters from clean points
```

### TV-F20-DB-02: All points identical
```
data = [[1.0, 2.0]; 100]
epsilon=0.01, min_samples=2
Expected: 1 cluster, all core points, 0 noise
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F20-1: Empty cluster | **MEDIUM** | Ghost centroid at origin | Re-init or split |
| F20-2: Initialization | **MEDIUM** | Slow/suboptimal for sorted data | K-means++ |
