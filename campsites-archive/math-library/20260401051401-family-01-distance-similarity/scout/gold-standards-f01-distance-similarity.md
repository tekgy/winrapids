# F01 Distance & Similarity — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load context for the pathmaker on Family 01 (Distance & Similarity).
Documents: which library implements which metric, formulas, and tambear decomposition.

---

## scipy.spatial.distance — Complete Metric List

All pairwise: `scipy.spatial.distance.cdist(XA, XB, metric='...')`.
Symmetric: `scipy.spatial.distance.pdist(X, metric='...')`.

### Continuous Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| `euclidean` | √Σ(uᵢ-vᵢ)² | L2 norm of difference |
| `sqeuclidean` | Σ(uᵢ-vᵢ)² | L2² — avoids sqrt, useful for clustering |
| `cityblock` | Σ|uᵢ-vᵢ| | L1, Manhattan |
| `chebyshev` | max|uᵢ-vᵢ| | L∞ — needed by sample entropy |
| `minkowski` | (Σ|uᵢ-vᵢ|^p)^(1/p) | Generalizes all Lp norms |
| `cosine` | 1 - (u·v)/(‖u‖‖v‖) | In [0,2]; 0=identical direction |
| `correlation` | 1 - Pearson_r(u, v) | Correlation distance ∈ [0,2] |
| `mahalanobis` | √((u-v)^T Σ^{-1} (u-v)) | Requires covariance matrix VI |
| `canberra` | Σ|uᵢ-vᵢ|/(|uᵢ|+|vᵢ|) | Weighted L1 |
| `braycurtis` | Σ|uᵢ-vᵢ| / Σ|uᵢ+vᵢ| | Bray-Curtis dissimilarity |
| `seuclidean` | √Σ(uᵢ-vᵢ)²/Vᵢ | Standardized Euclidean; Vᵢ = per-feature variance |
| `jensenshannon` | √(JSD(u‖v)) | √Jensen-Shannon divergence; symmetric KL |

### Binary/Set Metrics (Boolean arrays)

| Metric | Formula | Notes |
|--------|---------|-------|
| `hamming` | fraction of positions where uᵢ ≠ vᵢ | Works on any dtype |
| `jaccard` | 1 - |u∩v|/|u∪v| | Intersection over union |
| `dice` | 1 - 2|u∩v|/(|u|+|v|) | F1 complement |
| `rogerstanimoto` | 2·(u XOR v) / (u XOR v + n) | — |
| `russellrao` | (n - |u∩v|) / n | Proportion not in both |
| `sokalsneath` | 2·(u XOR v) / (2·(u XOR v) + n_11) | — |
| `yule` | (n00·n11 - 2·n01·n10) / (n00·n11 + n01·n10) | Yule Q |

**NOT in scipy.spatial.distance**: DTW (→ dtaidistance/tslearn), Wasserstein (→ scipy.stats), Earth Mover's Distance for 2D+ (→ POT library).

---

## sklearn.metrics.pairwise — Kernel and Distance Functions

```python
from sklearn.metrics.pairwise import (
    euclidean_distances,      # D[i,j] = ‖xᵢ - xⱼ‖²  (squared, not sqrt)
    manhattan_distances,      # L1 pairwise
    cosine_similarity,        # S[i,j] = (xᵢ·xⱼ)/(‖xᵢ‖‖xⱼ‖)
    cosine_distances,         # 1 - cosine_similarity
    pairwise_distances,       # wrapper; supports all scipy metrics + more
    linear_kernel,            # K[i,j] = xᵢ·xⱼ (dot product)
    polynomial_kernel,        # K[i,j] = (γ·xᵢ·xⱼ + coef0)^degree
    rbf_kernel,               # K[i,j] = exp(-γ·‖xᵢ-xⱼ‖²)  ← Gaussian RBF
    laplacian_kernel,         # K[i,j] = exp(-γ·‖xᵢ-xⱼ‖₁)  ← L1-based RBF
    sigmoid_kernel,           # K[i,j] = tanh(γ·xᵢ·xⱼ + coef0)
    chi2_kernel,              # K[i,j] = exp(-γ·Σ (xᵢₖ-xⱼₖ)²/(xᵢₖ+xⱼₖ))
    additive_chi2_kernel,     # K[i,j] = Σ (xᵢₖ-xⱼₖ)²/(xᵢₖ+xⱼₖ)
)
```

**Key observation**: `euclidean_distances()` returns SQUARED distances without sqrt.
This matches TiledEngine's DistanceOp formula `‖a-b‖² = ‖a‖² + ‖b‖² - 2⟨a,b⟩`.

---

## R: stats::dist and proxy::dist

### stats::dist (base R)

```r
stats::dist(X, method="euclidean")  # default
# methods: "euclidean", "maximum" (L∞), "manhattan", "canberra",
#          "binary" (Jaccard for binary), "minkowski" (+ p parameter)
```

Only computes the **lower triangle** as a `dist` object (compact storage).

### proxy::dist (proxy package — much more comprehensive)

```r
library(proxy)
proxy::dist(X, method="...")
# Available methods (selected):
# Continuous: Euclidean, Cosine, Correlation, Maximum, Manhattan,
#             Canberra, Minkowski, Hellinger, KL, JSD, Bray, Bray-Curtis
# Binary: Jaccard, Dice, Hamming
# Time series: DTW (calls dtw package)
# Custom: register your own distance function with pr_DB
proxy::pr_DB$get_entry("Cosine")  # inspect formula
```

**proxy::dist(method="DTW")** calls the `dtw` package internally.

---

## DTW (Dynamic Time Warping)

DTW is NOT in scipy.spatial.distance. Three main implementations:

### R: dtw package

```r
library(dtw)
dtw(x, y)                   # full DTW, symmetric step pattern
dtw(x, y)$distance          # scalar distance
dtw(x, y, window.type="sakoechiba", window.size=10)  # Sakoe-Chiba band
dtw(x, y, step.pattern=asymmetric)  # asymmetric slope constraint
dtw(x, y, open.end=TRUE)    # open-end DTW for partial matching

# Key step patterns (from dtw package):
# symmetric1       : classic Sakoe-Chiba, no slope constraint
# symmetric2       : symmetric, equal-cost weighting (standard)
# asymmetric       : asymmetric slope constraint (1 cell per step)
# rabinerJuangStepPattern(type, slope.weighting): SpeechRec patterns
```

### Python: dtaidistance

```python
from dtaidistance import dtw, dtw_ndim
dtw.distance(s1, s2)                     # scalar DTW distance, C-compiled
dtw.distance(s1, s2, window=10)          # with Sakoe-Chiba window
dtw_ndim.distance(s1, s2, ndim=3)       # multivariate DTW (per-dim L2)
dtw.warp(s1, s2)                         # warping path
dtw.warping_paths(s1, s2)               # full DP matrix

# Batch computation (vectorized):
dtw.distance_matrix_fast(series)         # C-compiled pairwise DTW
```

### Python: tslearn

```python
from tslearn.metrics import (
    dtw,                   # classic DTW distance
    soft_dtw,              # differentiable SoftDTW (Cuturi & Blondel 2017)
    dtw_path,              # DTW + warping path
    ctw_path,              # Canonical Time Warping
)
```

**SoftDTW** is the differentiable variant used for gradient-based DTW averaging.
Formula: `SoftDTW_γ(x,y) = -γ · log Σ exp(-cost/γ)` over all paths.

---

## Wasserstein / Earth Mover's Distance

### 1D Wasserstein (scipy)

```python
from scipy.stats import wasserstein_distance
wasserstein_distance(u_values, v_values)                     # p=1 Wasserstein
wasserstein_distance(u_values, v_values, u_weights, v_weights)  # weighted
# Algorithm: O(n log n) via sorted CDFs
# W₁(P,Q) = ∫|F_P(x) - F_Q(x)| dx = mean absolute difference of sorted values
```

For discrete distributions of equal mass: just `mean(|sorted(u) - sorted(v)|)`.

### 2D / High-Dimensional Wasserstein

**NOT in scipy** — requires separate library.

Primary Python option: `pot` (Python Optimal Transport):

```python
import ot  # pip install pot

# Exact EMD (Earth Mover's Distance):
M = ot.dist(a, b)                    # cost matrix (L2² by default)
T = ot.emd(a_weights, b_weights, M)  # transport plan, O(n³) LP
W2 = np.sum(T * M)                   # W2² distance

# Sinkhorn (regularized, much faster for large n):
T_reg = ot.sinkhorn(a_weights, b_weights, M, reg=0.1)
W2_approx = np.sum(T_reg * M)

# Sliced Wasserstein (fast approximation for high-d):
sw = ot.sliced_wasserstein_distance(a, b, n_projections=50)
```

**Sinkhorn complexity**: O(n² / ε²) iterations of O(n²) matrix ops.
**Sliced Wasserstein**: Projects to 1D random directions, averages W₁. O(n log n) per projection.

R equivalent: `transport` package — `wasserstein1d()`, `transport()`.

---

## Kernel Functions (for Kernel SVM / Kernel PCA)

These define similarity in feature space H without explicit computation.

| Kernel | Formula | Parameters |
|--------|---------|-----------|
| Linear | K(x,y) = x·y | — |
| Polynomial | K(x,y) = (γ·x·y + c)^d | γ, c (coef0), d (degree) |
| RBF / Gaussian | K(x,y) = exp(-γ·‖x-y‖²) | γ = 1/(2σ²) |
| Laplacian | K(x,y) = exp(-γ·‖x-y‖₁) | γ |
| Sigmoid | K(x,y) = tanh(γ·x·y + c) | γ, c |
| χ² | K(x,y) = exp(-γ·Σ(xᵢ-yᵢ)²/(xᵢ+yᵢ)) | γ |

Note: RBF kernel and Gaussian kernel are the same function.
`γ = 1/(2σ²)` — sklearn uses γ parameterization, papers often use σ.

**Key tambear observation**: RBF(x,y) = exp(-γ · ‖x-y‖²) = exp(-γ · sqeuclidean(x,y)).
Once the TiledEngine computes sqeuclidean distance matrix D, kernel matrix K = exp(-γ · D).
This is O(N²) element-wise operations on the cached DistancePairs MSR.
RBF kernel matrix is FREE once DistancePairs is computed.

---

## Tambear Decomposition by Metric Class

### Class 1: Derived from GramMatrix (fastest path)

L2 (Euclidean):
```
‖xᵢ - xⱼ‖² = GramMatrix[i,i] - 2·DotProduct[i,j] + GramMatrix[j,j]
```
Free from GramMatrix. One GPU pass for all N×N L2 distances.

Cosine:
```
cos(xᵢ, xⱼ) = DotProduct[i,j] / (‖xᵢ‖·‖xⱼ‖)
            = GramMatrix[i,j] / √(GramMatrix[i,i] · GramMatrix[j,j])
```
Also free from GramMatrix. Element-wise division on the matrix.

Pearson correlation (as distance):
```
corr(xᵢ, xⱼ) = centered cosine similarity (center each row first)
```
Requires centering pass before GramMatrix.

### Class 2: Direct TiledOp (no GramMatrix shortcut)

L1 (Manhattan), L∞ (Chebyshev), Lp (Minkowski p≠2):
- Cannot be derived from GramMatrix
- Require full O(N²·d) TiledOp with different combine function
- L1 combine: `|aᵢ - bⱼ|` — abs difference
- L∞ combine: `max|aᵢ - bⱼ|` — max-combine, not yet in TiledEngine

**L∞ is needed for sample entropy** (fintek) — see chaos family scout notes.

### Class 3: Non-geometric (CPU post-processing)

Mahalanobis: standard L2 after whitening transform. Compute covariance (GramMatrix), Cholesky, apply L transform to data, then standard L2.

Kernel matrix from distance: element-wise exp(-γ·D) — pure CPU or GPU kernel, no new accumulate.

### Class 4: Sequential / Order-Based (outside accumulate)

| Metric | What's needed | Tambear path |
|--------|--------------|-------------|
| DTW | DP table per pair | Custom tiled DP kernel |
| Wasserstein 1D | Sort + CDF comparison | Sequential scan |
| Wasserstein 2D+ | LP or Sinkhorn | Not in tambear primitives |
| Edit distance | DP table | String-specific; not in scope |

---

## Sharing Surface: What Consumers Share

When TamSession caches `Arc<DistanceMatrix>` keyed by (DataId, Metric):

| Consumer | Metric needed | Derives from |
|----------|-------------|-------------|
| DBSCAN | L2² | TiledEngine pass |
| KNN | L2² | same cached matrix |
| KMeans E-step | L2² (nearest centroid) | argmin of same matrix |
| Silhouette | L2² | same cached matrix |
| Davies-Bouldin | L2² | same cached matrix |
| Correlation Dimension | L2² | same cached matrix |
| RQA | L2² (binary threshold) | same cached matrix |
| Cosine similarity | cosine | derived from GramMatrix |
| RBF kernel | L2² → exp(-γ·D) | element-wise on cached L2² |
| Spectral clustering | L2² | same cached matrix |

**Zero additional GPU cost** for all consumers after the first DBSCAN/distance computation.

---

## Key Traps for Pathmaker

1. **sklearn euclidean_distances returns squared** (no sqrt). Check whether consumers expect d or d².

2. **scipy.spatial.distance.cdist returns full matrix** (N×M). For symmetric N×N: pdist returns upper triangle (N(N-1)/2 values). TiledEngine returns full N×N.

3. **DTW is not in scipy** — easy to assume it is since the rest of the distance zoo is there.

4. **Wasserstein 1D is in scipy.stats, not scipy.spatial.distance** — different submodule.

5. **Cosine distance ≠ 1 - cosine similarity always** — only true when similarity ∈ [0,1]. sklearn's `cosine_distances` = 1 - cos_sim, can be negative if vectors point opposite directions. For centered/unit-normed vectors this doesn't arise.

6. **L∞ (Chebyshev) in sample entropy** requires max-combine TiledOp — currently not in TiledEngine. This is the one new primitive the chaos family needs.
