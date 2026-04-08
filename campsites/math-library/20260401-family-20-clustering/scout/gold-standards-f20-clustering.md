# F20 Clustering — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 20 (Clustering: all methods + validation).
Prerequisite: F01 distance matrix (DBSCAN, silhouette) and F10/GramMatrix (spectral).
This family is the first heavy consumer of TamSession DistancePairs.

---

## Sharing Chain Through TamSession

```
DistancePairs (F01) ──→ DBSCAN
                    ──→ KNN (zero GPU cost)
                    ──→ KMeans E-step (nearest centroid)
                    ──→ Silhouette (per-cluster distance means)
                    ──→ Davies-Bouldin (inter/intra cluster distances)
                    ──→ HDBSCAN (core distances from KNN)
                    ──→ Correlation dimension (F26, chaos family)
                    ──→ RQA (F26, chaos family)

GramMatrix (F10) ──→ SpectralClustering (eigenmap of affinity)
                 ──→ KernelKMeans (kernel matrix → Euclidean embedding)
```

Every algorithm after the first DBSCAN call gets distances for free.

---

## sklearn.cluster — Complete API Reference

```python
from sklearn.cluster import (
    KMeans,                  # Lloyd's algorithm + k-means++
    MiniBatchKMeans,         # online k-means for large n
    DBSCAN,                  # density-based, requires eps + min_samples
    HDBSCAN,                 # hierarchical DBSCAN, auto selects eps
    AgglomerativeClustering, # hierarchical, bottom-up
    SpectralClustering,      # eigenmap → KMeans in embedding
    OPTICS,                  # like DBSCAN but auto selects eps range
    MeanShift,               # kernel density estimate peaks
    Birch,                   # online clustering with CF-tree
    AffinityPropagation,     # exemplar-based, no K needed
    GaussianMixture,         # EM-based (in sklearn.mixture, not .cluster)
)
```

---

## KMeans

```python
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3,
    init='k-means++',     # default since sklearn 0.24 — D²-weighted seeding
    n_init=10,            # number of random restarts (run 10, keep best)
    max_iter=300,
    tol=1e-4,             # convergence tolerance on inertia change
    random_state=42,
)
km.fit(X)

km.cluster_centers_    # K×d centroid matrix
km.labels_             # hard cluster assignments (n,)
km.inertia_            # total within-cluster sum of squares (WCSS)
km.n_iter_             # iterations taken

# Optimal K via elbow method:
wcss = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in range(1, 11)]
```

**Trap: `init='random'` vs `init='k-means++'`**:
- `'random'`: old default, random centroid initialization
- `'k-means++'`: D²-weighted seeding — much better convergence, now default
- For tambear validation: use `init='k-means++'` as oracle

**Trap: `n_init=10`** runs 10 times, keeps the best. Tambear should match this convention.

### R: kmeans

```r
km <- kmeans(X, centers=3, nstart=25, iter.max=300)
km$cluster      # hard labels
km$centers      # K×d centroids
km$withinss     # within-cluster SS per cluster
km$tot.withinss # total WCSS (= sklearn inertia_)
km$betweenss    # between-cluster SS
km$totss        # total SS = tot.withinss + betweenss
```

R uses `nstart=25` by default in recommendations (not the default of 1).

---

## DBSCAN

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=0.5,            # neighborhood radius — in same units as data
    min_samples=5,      # minimum points to form a core point
    metric='euclidean', # or 'precomputed' for tambear's cached distance matrix
    algorithm='auto',   # 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,
    n_jobs=-1,
)
db.fit(X)

db.labels_             # cluster labels; -1 = noise/outlier
db.core_sample_indices_  # indices of core points
(db.labels_ == -1).sum()  # number of noise points

# With precomputed distance matrix:
D = pairwise_distances(X)
db = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
db.fit(D)
```

**Critical trap: eps is NOT scale-invariant**. DBSCAN on unscaled data will cluster
by the feature with largest variance. Always standardize or choose eps based on
the k-NN distance distribution.

**Rule of thumb for eps**: plot k-NN distance (k=min_samples-1) sorted ascending;
eps ≈ knee of the curve. This requires the k-NN distances = slice of DistancePairs.

### R: dbscan package

```r
library(dbscan)
result <- dbscan(X, eps=0.5, minPts=5)
result$cluster    # labels; 0 = noise
result$isseed     # logical: is point a core seed?

# k-NN distance plot for eps selection:
kNNdistplot(X, k=4)  # plots sorted 4-NN distances; look for knee
abline(h=0.5, col="red")  # proposed eps
```

---

## HDBSCAN

HDBSCAN eliminates the eps parameter by extracting a hierarchy and using cluster stability.

```python
from sklearn.cluster import HDBSCAN

hdb = HDBSCAN(
    min_cluster_size=5,   # minimum points to form a cluster (replaces eps+min_samples)
    min_samples=None,     # if None, equals min_cluster_size (smoothing parameter)
    cluster_selection_method='eom',  # 'eom' (Excess of Mass) or 'leaf'
    metric='euclidean',   # or 'precomputed'
    allow_single_cluster=False,
)
hdb.fit(X)

hdb.labels_                    # cluster labels; -1 = noise
hdb.probabilities_             # soft cluster membership (0-1)
hdb.cluster_persistence_       # stability of each cluster
hdb.exemplars_                 # cluster exemplar points
```

```python
# Python hdbscan package (older, more features):
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
clusterer.fit(X)
clusterer.labels_
clusterer.probabilities_        # soft membership
clusterer.condensed_tree_.plot()  # cluster hierarchy visualization
clusterer.minimum_spanning_tree_.plot()  # MST visualization
```

```r
library(dbscan)
result <- hdbscan(X, minPts=5)
result$cluster
result$outlier_scores  # analogous to probabilities
```

**Key**: HDBSCAN's cluster_persistence_ = Campello stability = Σ(λ_p - λ_birth) = PHI_CENTERED_SUM accumulate (from the scout's naturalist notes).

---

## Cluster Validation Metrics

### Silhouette Score

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# With raw data (computes L2 distances internally):
s = silhouette_score(X, labels)          # mean silhouette across all points
s_per_sample = silhouette_samples(X, labels)  # per-point silhouette

# With precomputed distance matrix (reuses TamSession DistancePairs):
s = silhouette_score(D, labels, metric='precomputed')
s_per_sample = silhouette_samples(D, labels, metric='precomputed')
```

```r
library(cluster)
s <- silhouette(km$cluster, dist(X))   # cluster package
summary(s)$avg.width   # mean silhouette
plot(s)                # silhouette plot
```

**Formula**: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
- `a(i)` = mean distance to points in same cluster
- `b(i)` = mean distance to nearest other cluster

Range: [-1, +1]. Higher = better. Negative = misclassified.

**Tambear path**: pure CPU on cached DistancePairs. Row-wise scatter by label, compute means, then per-point formula. O(K·n) after distance matrix is available.

### Calinski-Harabasz (CH) Index

```python
from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(X, labels)  # higher = better
```

```r
library(fpc)
cluster.stats(dist(X), km$cluster)$ch
```

**Formula**: `CH = (SS_between / (K-1)) / (SS_within / (n-K))`

**Tambear path**: directly from MomentStats(ByKey) — no distance matrix needed.
SS_within = Σᵢ (nᵢ-1) · var_i and SS_between from group means vs grand mean.
Same extraction as F07 one-way ANOVA F-statistic (ANOVA F = CH up to a constant).

### Davies-Bouldin (DB) Index

```python
from sklearn.metrics import davies_bouldin_score
db = davies_bouldin_score(X, labels)  # lower = better
```

**Formula**: `DB = (1/K) · Σᵢ max_{j≠i} [(sᵢ + sⱼ) / dᵢⱼ]`
- `sᵢ` = mean intra-cluster distance = from DistancePairs
- `dᵢⱼ` = distance between cluster centroids = from DistancePairs on centroids

**Tambear**: requires DistancePairs, but only on K centroids after clustering.
K×K centroid distances = O(K²d) — trivially small for typical K.

### Gap Statistic

```r
library(cluster)
gap <- clusGap(X, FUN=kmeans, K.max=10, B=50, nstart=25)
plot(gap, main="Gap Statistic")
gap$Tab[,"gap"]    # gap statistic per K
which.max(gap$Tab[,"gap"])  # optimal K
```

```python
# sklearn does NOT have gap statistic — use manual implementation
# or gap_statistic package (gap_stat): not in core scipy/sklearn
```

**Tambear**: gap statistic requires B=50 random reference datasets + clustering each.
Computationally expensive. CPU-side is fine for moderate n.

---

## Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

# Single linkage (chaining):
ac = AgglomerativeClustering(n_clusters=3, linkage='single')
# Complete linkage (compact clusters):
ac = AgglomerativeClustering(n_clusters=3, linkage='complete')
# Average linkage (UPGMA):
ac = AgglomerativeClustering(n_clusters=3, linkage='average')
# Ward linkage (minimizes within-cluster variance):
ac = AgglomerativeClustering(n_clusters=3, linkage='ward')

ac.fit(X)
ac.labels_

# With precomputed distance matrix:
ac = AgglomerativeClustering(n_clusters=3, linkage='average', metric='precomputed')
ac.fit(D)
```

```r
# R base:
h <- hclust(dist(X), method="ward.D2")  # ward.D2 matches sklearn 'ward'
labels <- cutree(h, k=3)
plot(h)  # dendrogram

# Linkage methods: "ward.D2", "complete", "average", "single", "mcquitty", "centroid", "median"
# Note: R has "ward.D" and "ward.D2" — sklearn matches "ward.D2"
```

**Trap: R's ward.D vs ward.D2**: sklearn Ward uses squared distances internally (Ward.D2).
R's `hclust(method="ward.D")` uses distances (not squared). Use `ward.D2` in R to match sklearn.

---

## Spectral Clustering

```python
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(
    n_clusters=3,
    affinity='rbf',        # 'rbf', 'nearest_neighbors', 'precomputed', 'precomputed_nearest_neighbors'
    gamma=1.0,             # RBF kernel bandwidth
    assign_labels='kmeans',  # 'kmeans' or 'discretize'
    n_init=10,
    random_state=42,
)
sc.fit(X)
sc.labels_

# With precomputed affinity/kernel matrix:
K = rbf_kernel(X, gamma=1.0)
sc_pre = SpectralClustering(n_clusters=3, affinity='precomputed')
sc_pre.fit(K)
```

**Tambear path**:
1. KernelMatrix(RBF) from TamSession (= exp(-γ·DistancePairs))
2. EigenDecomposition of Laplacian L = D - K (or normalized)
3. KMeans on embedding (reuses tambear KMeans)

Spectral clustering = one-shot EigenDecomposition (F22 primitive) + KMeans.

---

## Validation Oracle Commands (confirm these before hardcoding)

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)
# Well-separated clusters:
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 0],
    np.random.randn(50, 2) + [2.5, 4],
])
true_labels = np.array([0]*50 + [1]*50 + [2]*50)

km = KMeans(n_clusters=3, n_init=10, random_state=42)
km.fit(X)

print("inertia:", km.inertia_)        # capture oracle value
print("silhouette:", silhouette_score(X, km.labels_))
# Expected silhouette > 0.8 for well-separated clusters

# With precomputed distances:
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
print("silhouette (precomputed):", silhouette_score(D, km.labels_, metric='precomputed'))
# Should match silhouette above to < 1e-10
```
