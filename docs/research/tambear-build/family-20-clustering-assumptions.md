# Family 20: Clustering (All Methods + Validation) — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (distance matrix, validation), C (KMeans/EM iteration), Sequential (MST, OPTICS ordering)

---

## Core Insight: Distance Matrix as Shared Infrastructure

Nearly every clustering method starts with the **N x N distance matrix** — the shared foundational primitive:

1. TiledEngine computes D[i,j] = distance(xᵢ, xⱼ) for all pairs
2. DBSCAN, HDBSCAN, OPTICS, spectral, agglomerative, silhouette ALL consume this matrix
3. Via TamSession: compute once, feed 9+ consumers

**One GPU computation, nine consumers.** The sharing infrastructure thesis proven.

---

## 1. HDBSCAN (Campello, Moulavi, Sander 2013)

### Parameters
| Parameter | Symbol | Role |
|-----------|--------|------|
| `min_samples` | k | Core distance = k-th nearest neighbor distance. Controls density smoothing. |
| `min_cluster_size` | m | Minimum points for a group to be a cluster (condensed tree extraction). |
| `metric` | d(a,b) | Distance function. Default: Euclidean. |

**Critical distinction**: `min_samples` controls density estimation (core distance). `min_cluster_size` controls cluster extraction from hierarchy. They are **orthogonal**.

### Step 1: Core Distance
```
core_k(xᵢ) = d(xᵢ, NN_k(xᵢ))
```
k-th nearest neighbor distance. Low core distance = dense region. High = sparse.

### Step 2: Mutual Reachability Distance
```
d_mreach(a, b) = max(core_k(a), core_k(b), d(a, b))
```
Symmetric. Points in dense regions keep original distances; sparse points get "pushed apart."

**Theoretical justification**: Mutual reachability transform makes single-linkage clustering approximate the hierarchy of level sets of the true underlying density.

### Step 3: Minimum Spanning Tree
MST of complete graph weighted by d_mreach. For dense matrix: Prim's O(N²). For GPU parallelism: Boruvka's O(E log V).

### Step 4: Single-Linkage Hierarchy
Sort MST edges by weight → process in order with union-find → dendrogram.

### Step 5: Condensed Cluster Tree
Walk hierarchy top-down. At each split:
- Both children ≥ min_cluster_size → **true split** (new cluster nodes)
- One child < min_cluster_size → those points "fall out" (absorbed, with recorded λ_fall)
- Both children < min_cluster_size → cluster dies

**λ = 1/ε** (density parameter). Higher λ = higher density = more zoomed in.

### Step 6: Stability & Cluster Selection
```
S(C) = Σ_{p ∈ C} (λ_p - λ_birth(C))
```
where λ_p = density at which point p leaves cluster C.

**Selection** (bottom-up):
```
if S(parent) > Σ S(children): keep parent, deselect descendants
else: keep children, propagate Σ S(children) as parent's effective stability
```

### Step 7: Membership Probabilities
```
prob(p, C) = (λ_p - λ_birth(C)) / (λ_death(C) - λ_birth(C))
```

### GLOSH Outlier Scores
```
GLOSH(p) = 1 - λ_max(p, C) / λ_max(C)
```
Each point compared to its LOCAL cluster's peak density. Handles variable-density clusters.

### HDBSCAN Generalizes DBSCAN
DBSCAN at epsilon ε is equivalent to cutting the HDBSCAN hierarchy at λ = 1/ε. HDBSCAN computes ALL DBSCAN* clusterings simultaneously.

### Failure Modes
- **Curse of dimensionality** (d > 50): distances converge, density estimation meaningless. Use dimensionality reduction first.
- **Very small min_samples (1-2)**: noisy core distances, spurious clusters
- **Single large cluster**: EOM criterion biased against root. Use `allow_single_cluster` parameter.
- **Ties in MST weights**: non-unique MST. Rare for continuous data.

### GPU Decomposition
| Step | GPU/CPU | Primitive | Reuse |
|------|---------|-----------|-------|
| Distance matrix | GPU | TiledEngine | F01 DistancePairs |
| Core distance (k-th NN) | GPU | partial sort per row | KNN module |
| Mutual reachability | GPU | element-wise max(c[i], c[j], D[i,j]) | fused_expr |
| MST (Prim's, N < 10K) | CPU | Sequential, O(N²) | New |
| Condensed tree + selection | CPU | Sort + union-find + tree walk | New, O(N log N) |

---

## 2. OPTICS (Ankerst et al. 1999)

### Core Distance
Same as HDBSCAN: `core_dist(p) = d(p, NN_k(p))`.

### Reachability Distance (ASYMMETRIC — unlike HDBSCAN)
```
reach_dist(o, p) = max(core_dist(o), d(o, p))
```
From predecessor o to successor p. NOT the same as mutual reachability.

### Ordering Algorithm
Priority-queue traversal: start from arbitrary point, greedily add the point with smallest reachability distance from any already-processed point.

Output: ordered list of (point_index, reachability_distance). The "reachability plot" shows valleys for clusters.

### Cluster Extraction
**Xi method**: Detect steep-down / steep-up transitions:
```
steep_down: reach[i] ≥ reach[i+1] · (1 - ξ)
steep_up:   reach[i] · (1 - ξ) ≤ reach[i+1]
```
Clusters = matched (SDA_start, SUA_end) pairs.

**DBSCAN-like extraction**: For any ε, cut the reachability plot at height ε. Same as DBSCAN with that ε.

### Failure Modes
- Ordering is start-point dependent (though cluster structure is not)
- Xi parameter sensitivity: too small → over-splits, too large → under-splits
- Inherently sequential ordering → poor GPU parallelism

---

## 3. Spectral Clustering (Ng, Jordan, Weiss 2001)

### Pipeline
1. **Affinity matrix**: W[i,j] = exp(-||xᵢ - xⱼ||² / (2σ²))
2. **Degree matrix**: D[i,i] = Σⱼ W[i,j]
3. **Normalized Laplacian**: L_sym = I - D⁻¹/²WD⁻¹/²
4. **Eigendecomposition**: k smallest eigenvectors of L_sym → V (N × k)
5. **Row-normalize**: U[i,:] = V[i,:] / ||V[i,:]||
6. **KMeans** on rows of U → labels

### Three Laplacian Types
| Type | Formula | Properties |
|------|---------|------------|
| Unnormalized | L = D - W | PSD, k zero eigenvalues = k connected components |
| Symmetric normalized | L_sym = I - D⁻¹/²WD⁻¹/² | Eigenvalues ∈ [0,2], symmetric |
| Random walk | L_rw = I - D⁻¹W | NOT symmetric, same eigenvalues as L_sym |

### σ Selection
- **Local scaling** (Zelnik-Manor & Perona 2004): σᵢ = d(i, NN_k(i)), then W[i,j] = exp(-d²/(σᵢσⱼ))
- **Median heuristic**: σ = median(all pairwise distances) / √2
- Grid search with silhouette score

### Failure Modes
- σ too small → identity affinity → no structure
- σ too large → all-ones affinity → one component
- Disconnected graph → more zero eigenvalues than k → degenerate
- Repeated eigenvalues → eigenvector subspace not unique → KMeans sensitive

### CRITICAL: Spectral Clustering = Graph Laplacian Eigenmap (F29)
Pipeline is identical through step 4. Only the consumer differs (clustering vs embedding). **Structural rhyme confirmed.**

### GPU Decomposition
| Step | GPU/CPU | Reuse |
|------|---------|-------|
| Distance matrix | GPU | F01 DistancePairs |
| Affinity W = exp(-D/2σ²) | GPU | fused_expr on DistancePairs |
| Degree = row_sum(W) | GPU | reduce per row |
| L_sym | GPU | element-wise |
| Eigendecomposition | CPU | Lanczos (shared with F22 PCA) |
| KMeans on U | GPU | Existing KMeansEngine |

---

## 4. Agglomerative Clustering

### Lance-Williams Recurrence (Unified Update)
When clusters A, B merge into (AB), distance to any cluster C:
```
d((AB), C) = α_A·d(A,C) + α_B·d(B,C) + β·d(A,B) + γ·|d(A,C) - d(B,C)|
```

| Linkage | α_A | α_B | β | γ |
|---------|-----|-----|---|---|
| Single | 1/2 | 1/2 | 0 | -1/2 |
| Complete | 1/2 | 1/2 | 0 | 1/2 |
| Average (UPGMA) | nA/(nA+nB) | nB/(nA+nB) | 0 | 0 |
| Ward's | (nC+nA)/(nC+nAB) | (nC+nB)/(nC+nAB) | -nC/(nC+nAB) | 0 |
| Centroid | nA/(nA+nB) | nB/(nA+nB) | -nAnB/(nA+nB)² | 0 |
| Median | 1/2 | 1/2 | -1/4 | 0 |

### Key Insight
Lance-Williams means we NEVER re-access original data after computing the initial distance matrix. Each merge is O(N) updates. Total: O(N²).

### CRITICAL: Centroid/Median Linkage Can Produce Inversions
Merge distance can DECREASE — the dendrogram is NOT monotone. This violates the ultrametric property. Single, complete, average, and Ward's are always monotone.

### Ward's Requires Euclidean
Ward's formula uses centroids. Applying Ward's to non-Euclidean distances is undefined.

### Single Linkage = MST
The single-linkage dendrogram is equivalent to the MST cut at increasing thresholds. HDBSCAN's MST step IS single-linkage agglomerative clustering.

---

## 5. Gaussian Mixture Models (EM)

### Model
```
p(x) = Σ_{k=1}^K πₖ · N(x | μₖ, Σₖ)
```

### E-Step: Responsibilities (numerically stable, log-space)
```
log ρₙₖ = log πₖ - (d/2)log(2π) - (1/2)log|Σₖ| - (1/2)(xₙ-μₖ)'Σₖ⁻¹(xₙ-μₖ)
γₙₖ = exp(log ρₙₖ - log_sum_exp_k(log ρₙₖ))
```

### M-Step
```
Nₖ = Σₙ γₙₖ
μₖ = (1/Nₖ) Σₙ γₙₖ · xₙ
Σₖ = (1/Nₖ) Σₙ γₙₖ · (xₙ - μₖ)(xₙ - μₖ)'
πₖ = Nₖ / N
```

### CRITICAL: M-Step = Weighted Grouped Variance (scatter_multi_phi_weighted)
The M-step IS the same as F06 grouped descriptive statistics with soft (weighted) assignments:
```
scatter_multi_phi(phi=[1.0, v, (v-r)²], keys=argmax(γ), weights=γ)
```
**Zero new infrastructure needed if F06 supports weighted scatter.**

### Singularity Trap
If a component collapses to a single point: Σₖ → 0, likelihood → ∞.
**Fix**: Σₖ_reg = Σₖ + ε·I (regularization), or minimum effective count N_k ≥ d+1.

### Model Selection
```
BIC = -2·log L + p·log(N)
AIC = -2·log L + 2·p
```
where p = K·(d + d(d+1)/2 + 1) - 1 for full covariance.

### Failure Modes
- Local optima: run 10-20 random initializations, keep best log L
- Slow convergence near degenerate components
- Curse of dimensionality: responsibilities become hard (0/1) → degenerates to KMeans
- Empty components (Nₖ → 0): detect and remove/restart

### Kingdom: C (Iterative). Each iteration = Kingdom A E-step + M-step.

---

## 6. Cluster Validation Metrics

### 6a. Silhouette Coefficient
For point i with label lᵢ:
```
a(i) = (1/(|C_lᵢ|-1)) · Σ_{j∈C_lᵢ, j≠i} d(i,j)     (intra-cluster)
b(i) = min_{k≠lᵢ} (1/|Cₖ|) · Σ_{j∈Cₖ} d(i,j)        (nearest-cluster)
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Range: [-1, 1]. Higher is better.

**Edge cases**: K=1 → undefined (no other cluster). |C_lᵢ|=1 → a(i)=0, s(i)=0.

### 6b. Davies-Bouldin Index
```
DB = (1/K) · Σₖ max_{l≠k} (σₖ + σₗ) / d(cₖ, cₗ)
```
where σₖ = avg distance of points to centroid k. Lower is better.

### 6c. Calinski-Harabasz Index (Variance Ratio)
```
CH = (SS_between/(K-1)) / (SS_within/(N-K))
```
Higher is better. Same SS decomposition as ANOVA (F07). **Structural rhyme: CH index IS an F-statistic on cluster labels.**

### 6d. Adjusted Rand Index
```
ARI = (Σᵢⱼ C(nᵢⱼ,2) - [Σᵢ C(aᵢ,2)·Σⱼ C(bⱼ,2)]/C(N,2)) /
      (½[Σᵢ C(aᵢ,2) + Σⱼ C(bⱼ,2)] - [Σᵢ C(aᵢ,2)·Σⱼ C(bⱼ,2)]/C(N,2))
```
Requires contingency table of two clusterings. Range: [-1, 1]. ARI=1 → perfect, ARI=0 → random.

### 6e. Normalized Mutual Information
```
NMI = MI(U,V) / ((H(U) + H(V))/2)
```
Uses F25 (Information Theory) entropy/MI primitives. Range: [0, 1].

**Edge case**: H(U)=0 or H(V)=0 → 0/0 → return 0.

### 6f. Gap Statistic (Tibshirani, Walther, Hastie 2001)
```
Gap(K) = E_null[log W_K] - log W_K
```
Compare within-cluster dispersion to uniform null reference. Select smallest K where Gap(K) ≥ Gap(K+1) - s'_{K+1}.

### All Validation Metrics are Kingdom A
Scatter/reduce operations on DistancePairs + ClusterAssignment. Embarrassingly parallel.

---

## Sharing Surface

### From F01 (Distance & Similarity):
```
distance_matrix → DBSCAN, HDBSCAN, OPTICS, spectral, agglomerative, silhouette, Davies-Bouldin, gap
```

### From F06 (Descriptive Statistics):
```
MomentStats per cluster → Calinski-Harabasz (SS decomposition)
scatter_multi_phi → GMM M-step (weighted variant)
```

### From F07 (Hypothesis Testing):
```
ANOVA F-test ≡ Calinski-Harabasz index (same formula!)
```

### From F10 (Regression):
```
Cholesky → GP kernel solve, RBF interpolation
```

### From F22 (Dimensionality Reduction):
```
eigendecomposition → spectral clustering Laplacian
```

### From F25 (Information Theory):
```
entropy, MI → NMI validation metric
```

---

## New MSR Types Needed

| MSR Type | Produced By | Consumed By |
|----------|-------------|-------------|
| MinSpanTree | HDBSCAN/single-linkage | Stability extraction, dendrogram |
| CondensedClusterTree | HDBSCAN | Cluster selection, soft membership, GLOSH |
| OpticsOrdering | OPTICS | Xi extraction, DBSCAN-like extraction |
| Dendrogram | Agglomerative | Flat cut, visualization |
| SoftAssignment(N×K) | GMM E-step | M-step, soft silhouette, Bayesian posterior |
| EigenDecomposition | Spectral Laplacian | Shared with F22 PCA, F29 graph embedding |

---

## Implementation Priority

**Phase 1** — Complete existing + HDBSCAN:
1. KMeans++ initialization (gap in current impl)
2. Empty cluster handling (gap in current impl)
3. HDBSCAN (core distance + mutual reachability + MST + condensed tree + stability)
4. Silhouette coefficient (most-used validation metric)

**Phase 2** — Agglomerative:
5. Single linkage (= MST cut, reuses HDBSCAN MST)
6. Complete + Average + Ward's linkage (Lance-Williams)
7. Dendrogram data structure

**Phase 3** — Spectral + GMM:
8. Spectral clustering (RBF affinity + Laplacian + eigendecomp + KMeans)
9. GMM/EM (E-step + M-step via weighted scatter)
10. BIC/AIC model selection

**Phase 4** — Validation + OPTICS:
11. Calinski-Harabasz, Davies-Bouldin, Gap statistic
12. Adjusted Rand Index, NMI
13. OPTICS ordering + Xi extraction

---

## Composability Contract

```toml
[family_20]
name = "Clustering"
kingdom = "Mixed (A + C + Sequential)"

[family_20.shared_primitives]
distance_matrix = "tiled_accumulate(DistanceOp) — shared with F01"
scatter_by_label = "scatter_multi_phi(ByKey(label)) — shared with F06"
weighted_scatter = "scatter_multi_phi_weighted — shared with F06, needed for GMM"
eigendecomposition = "Lanczos on Laplacian — shared with F22"
union_find = "connected component labeling"

[family_20.reuses]
f01_distance_pairs = "TiledEngine DistanceOp — the core shared intermediate"
f06_moment_stats = "per-cluster descriptive statistics"
f07_anova = "Calinski-Harabasz IS an F-statistic"
f22_eigendecomp = "spectral clustering shares with PCA"
f25_entropy_mi = "NMI validation metric"

[family_20.session_intermediates]
distance_matrix = "DistanceMatrix(data_id, metric)"
cluster_assignment = "ClusterAssignment(data_id, method, params)"
dendrogram = "Dendrogram(data_id, linkage)"
soft_assignment = "SoftAssignment(data_id, K)"
mst = "MinSpanTree(data_id, metric)"
```
