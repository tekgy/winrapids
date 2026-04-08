# F20 Sharing Surface: Clustering Validation as Distance Matrix Extractions

Created: 2026-04-01T06:09:03-05:00
By: navigator

Prerequisites: F01 complete (DistancePairs / DistanceMatrix), F20 partial (DBSCAN/KMeans cluster labels cached in TamSession).

---

## Core Insight: All Internal Validation Metrics Read the Cached Distance Matrix

The TamSession already has:
1. The distance matrix (from F01 DistancePairs — O(N²) computation, expensive)
2. Cluster labels (from KMeans, DBSCAN — O(N) lookup, cheap)

Every internal validation metric is a SUMMARY of distances within and between clusters.
They are all O(N²) reads of the existing matrix — no new GPU computation.

```
Validation metric = summarize(distances[i,j], labels[i], labels[j])
```

This is scatter_phi with keys = (label[i], label[j]) pair. Every metric below is a
different phi expression over the same distance matrix.

---

## Silhouette Coefficient

For each point i:
```
a(i) = mean distance to all other points in same cluster
     = mean {d(i,j) : label[j] == label[i], j != i}

b(i) = min over other clusters c of: mean distance to points in cluster c
     = min_c { mean {d(i,j) : label[j] == c} }

s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Overall silhouette score = mean s(i) over all i.

**Tambear decomposition**:

Step 1: `scatter_phi("sum(d)", keys=(i, label[j]))` — for each (point_i, cluster_k) pair, sum distances.
        This is a grouped accumulate on the distance matrix rows.
        Output: matrix D_grouped[i, k] = Σ_{j: label[j]=k} d(i,j), and count[i, k] = |cluster k|.

Step 2: `a(i) = D_grouped[i, label[i]] / (count[i, label[i]] - 1)` — within-cluster mean (exclude self)

Step 3: `b(i) = min_{k ≠ label[i]} D_grouped[i, k] / count[i, k]` — min over other-cluster means
        This is MinCombineOp over k ≠ label[i].

Step 4: `s(i) = (b(i) - a(i)) / max(a(i), b(i))` — scalar per point

Step 5: `mean(s)` — global score

**All steps are O(N·K) where K = number of clusters.** No new primitives — uses existing scatter
and MinCombineOp.

**For Phase 1**: precompute pairwise Silhouette only when N < 10000 (O(N²) matrix is 800MB at f64).
For large N: approximate Silhouette via sampling or minibatch.

---

## Davies-Bouldin Index

```
R_ij = (s_i + s_j) / d(c_i, c_j)
where:
  s_k = mean distance of points in cluster k to centroid c_k
  c_k = centroid of cluster k
  d(c_i, c_j) = distance between centroids

DB = (1/K) Σ_i max_{j≠i} R_ij
```

**Tambear decomposition**:

Step 1: Centroids c_k = `scatter_phi("mean(x)", keys=labels)` — F06 grouped mean, already in KMeans
Step 2: s_k = `scatter_phi("mean(dist(x, c_k))", keys=labels)` — intra-cluster scatter
        = `scatter_phi("mean(|v - ref_k|)", keys=labels, ref_values=centroids)`
        This is RefCenteredStats! Already designed as a multi-domain primitive.
Step 3: d(c_i, c_j) = pairwise distances between K centroids — K×K matrix, K small (K << N)
Step 4: R_ij = (s_i + s_j) / d(c_i, c_j) — scalar ops
Step 5: DB = mean over i of max_{j≠i} R_ij — MaxCombineOp on the K×K matrix

**Total new code: ~30 lines.** Centroids already exist in KMeans result (TamSession cache hit).

---

## Calinski-Harabasz Index (Variance Ratio Criterion)

```
CH = [SS_between / (K-1)] / [SS_within / (N-K)]
where:
  SS_between = Σ_k n_k · ||c_k - c_global||²   (between-cluster sum of squares)
  SS_within  = Σ_k Σ_{i in k} ||x_i - c_k||²   (within-cluster sum of squares)
```

**Tambear decomposition**:

SS_within = `scatter_phi("sum((v-r)²)", keys=labels, ref_values=centroids)` — RefCenteredStats.
SS_between = scatter of n_k · ||c_k - c_global||² over clusters — scalar ops.

**This is EXACTLY RefCenteredStats** — the same primitive identified in HDBSCAN stability,
financial excess moments, and variance computation. SS_within IS grouped variance.

**Total new code: ~15 lines.** Nearly free given RefCenteredStats and centroids.

---

## Additional Internal Metrics (Phase 2)

### Dunn Index
```
Dunn = min inter-cluster distance / max intra-cluster diameter
```
min and max over distance matrix subsets — Min/MaxCombineOp on DistancePairs. ~20 lines.

### Adjusted Rand Index (ARI) — External Validation
```
ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
```
Requires contingency table of (predicted_labels, true_labels) — CrosstabStats from F25/F07.
Nearly free once CrosstabStats exists.

### Normalized Mutual Information (NMI) — External Validation
```
NMI = 2 · I(Y_pred; Y_true) / (H(Y_pred) + H(Y_true))
```
F25 (Information Theory) extraction on CrosstabStats. Free after F25.

---

## MSR Types F20 Produces

```rust
pub struct ClusterValidation {
    pub n_obs: usize,
    pub n_clusters: usize,
    pub labels: Arc<Vec<usize>>,   // cluster assignments

    // Internal metrics (don't require ground truth):
    pub silhouette_mean: f64,
    pub silhouette_per_point: Option<Arc<Vec<f64>>>,  // optional, expensive for large N
    pub davies_bouldin: f64,
    pub calinski_harabasz: f64,
    pub dunn_index: Option<f64>,   // Phase 2

    // External metrics (require ground truth labels):
    pub adjusted_rand_index: Option<f64>,
    pub normalized_mutual_info: Option<f64>,
    pub v_measure: Option<f64>,
}
```

Add to `IntermediateTag`:
```rust
ClusterValidation {
    distance_id: DataId,   // which distance matrix
    labels_id: DataId,     // which clustering result
    ground_truth_id: Option<DataId>,   // for external metrics
},
```

---

## Build Order

1. **Silhouette coefficient** — grouped scatter on distance matrix rows, then MinCombineOp per point (~60 lines)
2. **Davies-Bouldin** — RefCenteredStats for s_k, centroids from KMeans cache, pairwise centroid distances (~30 lines)
3. **Calinski-Harabasz** — SS_within via RefCenteredStats, SS_between via grouped scatter (~15 lines)
4. **ClusterValidation struct** + `IntermediateTag` (~25 lines)
5. **Tests**: sklearn.metrics for all three metrics — exact numerical match to 6 decimal places.

**Total Phase 1: ~130 lines.** Essentially free given F01 DistancePairs and KMeans/DBSCAN labels.

**Key constraint**: Silhouette requires O(N²) distance matrix. Gate on N < threshold (e.g., 50000)
or add approximate Silhouette via sampling for large N.

---

## Sharing Architecture

```
TamSession contains:
  DistancePairs (F01)           → Silhouette (reads full matrix)
  ClusterLabels (KMeans/DBSCAN) → grouping key for all metrics
  Centroids (KMeans)            → Davies-Bouldin, Calinski-Harabasz (cache hit)
  CrosstabStats (F07/F25)       → ARI, NMI (external validation)
```

F20 is purely a READER of existing TamSession state. Zero new accumulate calls for DB and CH.
One new accumulate call for Silhouette (grouped distance sums, O(N·K)).

---

## The Lab Notebook Claim

> Clustering validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) are scatter_phi expressions over the cached distance matrix and cluster centroids. DB and CH are RefCenteredStats — the same multi-domain primitive identified in variance computation, HDBSCAN stability, and financial excess returns. Silhouette requires one new grouped scatter (O(N·K)) over the distance matrix rows. Total new code: ~130 lines. F20 is the payoff for having cached the distance matrix in TamSession from the start.
