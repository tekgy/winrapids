# Cluster Validation Metrics — Gold Standard Scout Report

**What this covers**: Silhouette, Davies-Bouldin, Calinski-Harabasz, Gap Statistic.
All four validate clustering results (DBSCAN + KMeans + anything else).
**The headline**: three of four are pure operations on the distance matrix and/or sufficient
statistics already cached in TamSession. They come nearly for free.

---

## Why NOW

DBSCAN and KMeans are built. The clustering family (task #20) needs these four metrics to
be complete and to validate the algorithms. The sharing story is the whole point — DBSCAN
computes the distance matrix, registers it in TamSession, and then silhouette + DB + CH all
consume it at zero GPU cost. This is the architecture demonstrated, not just described.

---

## 1. Silhouette Score

**Gold standard**: `sklearn.metrics.silhouette_score` / `silhouette_samples`
**Reference**: Rousseeuw, 1987 — "Silhouettes: a graphical aid to the interpretation and
validation of cluster analysis."

### What it computes

For each point `i`:
```
a(i) = mean distance from i to ALL other points in same cluster
b(i) = mean distance from i to ALL points in nearest OTHER cluster
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Overall silhouette = mean(s(i)) across all i.

- `s(i)` near 1.0 → point is well-clustered
- `s(i)` near 0.0 → point is on a cluster boundary
- `s(i)` negative → point may be in the wrong cluster

### sklearn's actual algorithm

```python
# 1. chunked pairwise distance (or use precomputed matrix)
# 2. for each chunk of distances D[i, :]:
#    intra[i] += sum(D[i, j] for j in same cluster as i)
#    inter[k][i] += sum(D[i, j] for j in cluster k, k != cluster(i))
# 3. a[i] = intra[i] / (label_freq[cluster(i)] - 1)  # exclude self
# 4. b[i] = min over k != cluster(i) of inter[k][i] / label_freq[k]
# 5. s[i] = (b[i] - a[i]) / max(a[i], b[i])
# nan_to_num: handles singletons (a[i]=0, b[i]=0 → s=0)
```

Key: uses `np.bincount` per row to accumulate per-cluster distance sums.

### Tambear decomposition

INPUTS:
- `dist: Arc<DistanceMatrix>` — n×n from TamSession (already there after DBSCAN/KMeans!)
- `labels: &[i32]` — cluster assignments

STEPS (all CPU given precomputed distance matrix):

```
Step 1: For each row i, accumulate per-cluster distance sum and count:
  → scatter(dist[i, :], keys=labels, n_groups=K, op=Add)
  → this gives: intra_sum[i] = dist[i, same_cluster_j] summed
  → and per-cluster sums for b computation

Step 2: a[i] = intra_sum[i] / (cluster_size[labels[i]] - 1)

Step 3: b[i] = min over k ≠ labels[i] of (cluster_sum[i][k] / cluster_size[k])

Step 4: s[i] = (b[i] - a[i]) / max(a[i], b[i])
```

**The key insight**: because the distance matrix is n×n, and n ≤ 5000 (current range), all
of this is O(n²) CPU operations over an already-materialized matrix. No new GPU work.
The sharing infrastructure delivers this completely for free once DBSCAN/KMeans ran.

### Edge cases sklearn handles

- Singleton clusters (only one point): `a[i] = 0`, silhouette = 0 (via nan_to_num)
- Requires: 1 < n_clusters < n_samples
- With `sample_size` parameter: sub-sample for large n (but exact for small n)

### Sufficient stats approach

When the full n×n matrix is NOT available (large n, approximate case), silhouette can be
estimated from MSR: `{n_k, Σd_k, Σd_k²}` per cluster per point (distance moments). But
for n ≤ 5000, exact computation from cached matrix is preferred.

---

## 2. Davies-Bouldin Index

**Gold standard**: `sklearn.metrics.davies_bouldin_score`
**Reference**: Davies & Bouldin, 1979 — "A Cluster Separation Measure."

### What it computes

```
DB = (1/K) Σ_i max_{j≠i} (s_i + s_j) / d_ij
```

Where:
- `s_i` = average distance from points in cluster i to centroid i
- `d_ij` = distance between centroid i and centroid j
- Lower DB = better clustering (opposite convention from silhouette/CH)

### sklearn's actual algorithm

```python
# 1. centroids[k] = mean(X[labels == k], axis=0)  — one mean per cluster
# 2. intra_dists[k] = mean(pairwise_distances(X[labels==k], centroids[[k]]))
# 3. centroid_distances = pairwise_distances(centroids)
#    centroid_distances[centroid_distances == 0] = inf  # handle duplicates
# 4. combined = (intra_dists[:, None] + intra_dists[None, :]) / centroid_distances
#    combined[diag] = 0  (or masked)
# 5. scores[i] = max(combined[i, :])
# 6. return mean(scores)
```

### Tambear decomposition

INPUTS: `data: &[f64]` (n×d), `labels: &[i32]`, `n`, `d`, `K`

```
Step 1: Compute centroids
  centroids[k] = accumulate(data, ByKey{keys=labels, n_groups=K}, Expr::Value, Op::Add)
                 / accumulate(data, ByKey{...}, Expr::One, Op::Add)
  → scatter_multi_phi([1.0, x, x, ...], sum + count) → divide → K centroid vectors

Step 2: Compute intra-cluster scatter s_k
  For each point i: dist_to_centroid[i] = L2(data[i], centroids[labels[i]])
  → gather: for each i, look up centroids[labels[i]]  ← gather operation!
  → compute L2 (elementwise subtract, square, sum across d)
  → accumulate(dist_to_centroid, ByKey{keys=labels}, Expr::Value, Op::Add) / count
  → gives s[k] for each cluster

Step 3: K×K centroid distance matrix
  → tiled(DistanceOp, centroids, centroids_T, K, K, d)
  → K is small (2-50 typical), so this is tiny GPU work

Step 4: (s_i + s_j) / d_ij for all pairs → argmax per row → mean
  → pure CPU: K×K table lookup
```

**Insight**: step 1 (centroids) comes for FREE from KMeans output — KMeans already
computes centroids! If `KMeansResult.centroids` is exposed, DB only needs steps 2-4.
The centroid distance matrix in step 3 is K×K (small K), essentially free GPU work.

---

## 3. Calinski-Harabasz Index (Variance Ratio Criterion)

**Gold standard**: `sklearn.metrics.calinski_harabasz_score`
**Reference**: Caliński & Harabász, 1974 — "A dendrite method for cluster analysis."

### What it computes

```
CH = (SS_B / (K-1)) / (SS_W / (N-K))

SS_B = Σ_k n_k · ||c_k - c_global||²    — between-cluster dispersion
SS_W = Σ_k Σ_{i in k} ||x_i - c_k||²   — within-cluster dispersion
```

Higher CH = better clustering. Returns 1.0 if SS_W == 0.

### sklearn's actual algorithm

```python
mean = X.mean(axis=0)              # global centroid
extra_disp, intra_disp = 0.0, 0.0
for k in range(n_labels):
    cluster_k = X[labels == k]
    mean_k = cluster_k.mean(axis=0)
    extra_disp += len(cluster_k) * ((mean_k - mean)**2).sum()  # n_k * ||c_k - c||²
    intra_disp += ((cluster_k - mean_k)**2).sum()              # Σ||x - c_k||²
score = extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
```

### KILLER INSIGHT: CH from MSR sufficient stats alone

CH only needs: `{n_k, Σx_k, Σ(x_k - c_k)²}` per cluster, plus global `{n, Σx, Σx²}`.

The within-cluster SS: `Σ(x - c_k)² = Σx² - n_k·c_k² = Σx² - (Σx)²/n_k`
→ this is the **variance formula** from sufficient stats {n_k, Σx_k, Σx_k²}.

The between-cluster SS: `n_k·||c_k - c_global||² = n_k·(c_k - c_global)²`
→ `c_k = Σx_k/n_k`, `c_global = Σx/n`
→ also directly from {n_k, Σx_k}.

So **CH is computable from the 11-field MSR** without ever materializing the distance matrix
or computing centroids explicitly. One `accumulate(ByKey, [x, x², 1], Add)` pass → CH.
This means CH validation is a constant-time postprocessing step on the existing MSR.

### Tambear decomposition (exact)

```
Step 1: accumulate(data, All, [x, x², 1], Add) → global {Σx, Σx², n}
Step 2: accumulate(data, ByKey{labels, K}, [x, x², 1], Add) → per-cluster {Σx_k, Σx_k², n_k}
Step 3: c_k = Σx_k / n_k  (pointwise division)
Step 4: c_global = Σx / n
Step 5: SS_W = Σ_k (Σx_k² - (Σx_k)²/n_k)  — scalar
Step 6: SS_B = Σ_k n_k * ||c_k - c_global||²  — scalar
Step 7: CH = (SS_B / (K-1)) / (SS_W / (N-K))
```

Steps 1-2 are single accumulate passes. Steps 3-7 are scalar arithmetic.
**No distance matrix needed at all.** No GPU kernel after step 2.

---

## 4. Gap Statistic

**Gold standard**: R `cluster::clusGap`
**Reference**: Tibshirani, Walther & Hastie, 2001 — JRSS-B 63(2):411-423.

### What it computes

Estimates the optimal number of clusters K by comparing within-cluster dispersion to
what's expected under a null (uniform) distribution:

```
Gap(K) = E*[log(W_K)] - log(W_K)
```

Where:
- `W_K` = within-cluster sum of squared distances (= SS_W from CH)
- `E*` = expectation under reference distribution (bootstrapped uniform samples)
- Optimal K: smallest K where `Gap(K) ≥ Gap(K+1) - s(K+1)` (s = bootstrap std dev)

### Reference distribution

Two options (from the original paper):
1. **Uniform box**: generate B samples from Uniform(min_j, max_j) per feature j
2. **PCA-rotated box**: rotate data to PC coordinates, generate uniform in PC box,
   rotate back. Better for elongated distributions.

R's `clusGap` uses option 1 by default, option 2 available.

### Algorithm

```python
for k in 1..K_max:
    # Real data: cluster and compute W_k
    labels = kmeans(data, k)
    W_k = SS_W(data, labels, k)

    # Reference: B bootstrap samples
    W_k_star = []
    for b in 1..B:
        data_b = uniform_sample(data)  # from reference distribution
        labels_b = kmeans(data_b, k)
        W_k_star.append(SS_W(data_b, labels_b, k))

    Gap[k] = mean(log(W_k_star)) - log(W_k)
    s[k] = std(log(W_k_star)) * sqrt(1 + 1/B)  # corrected std

# Select K: first k where Gap[k] >= Gap[k+1] - s[k+1]
```

### Tambear decomposition

This is the EXPENSIVE one: B × K_max × n points of KMeans. Default: B=50, K_max=10.
Total: 500 KMeans runs. With GPU KMeans this is fast, but it's not sharable.

The SS_W computation (step inside the loop) IS from sufficient stats — same as CH formula.
`W_k = SS_W = Σ_k (Σx_k² - (Σx_k)²/n_k)`.

Opportunity: **batch the B reference runs**. All B samples for a given K can run
concurrently — they're independent KMeans instances on different data. GPU batching
with different starting seeds per stream could amortize the launch overhead.

### Note on log(W_k)

The paper uses log(W_k) not W_k directly — stabilizes variance across K values.
Log of the SS_W, not log applied elementwise. Easy post-hoc scalar op.

---

## Build Order Recommendation

From the perspective of the sharing architecture, the right order is:

1. **Calinski-Harabasz** — implement first. No distance matrix needed. Pure scatter + scalar
   arithmetic. Validates KMeans immediately and demonstrates MSR → metric output.

2. **Silhouette** — implement second. Consumes distance matrix from TamSession. Demonstrates
   cross-algorithm sharing: run DBSCAN/KMeans, then get silhouette for free.

3. **Davies-Bouldin** — implement third. Needs centroids (from KMeans) + intra-cluster
   scatter + K×K distance matrix (tiny). Moderate complexity.

4. **Gap Statistic** — implement fourth. Expensive, needs Monte Carlo. Implement as a
   multi-stream batch job.

---

## Sharing Graph: What Flows Into What

```
DBSCAN/KMeans run
    ↓
TamSession: DistanceMatrix(n×n)   KMeansResult.centroids(K×d)   labels(n)
    ↓                                  ↓                           ↓
Silhouette (row-wise groupby)    Davies-Bouldin (intra-scatter)   CH (scatter moments)
    ↑                                  ↑                           ↑
Zero GPU cost                    K² tiled tiny                    Sufficient stats only
```

The distance matrix → silhouette path is the canonical demonstration of the sharing
infrastructure paying off. One GPU computation enables three validation metrics at near-zero
incremental cost. This is the architecture promise kept.
