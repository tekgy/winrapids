# PCA & Dimensionality Reduction — Gold Standard Scout Report

**What this covers**: PCA (full, truncated, randomized), the gateway to spectral clustering,
t-SNE, UMAP, and all eigenvector-based methods.
**The headline**: PCA's covariance matrix is a tiled_accumulate. It uses the SAME TiledEngine
primitive as DBSCAN's distance matrix — just DotProductOp instead of DistanceOp, and
transposed roles. No new GPU primitives needed for the covariance step.

---

## What PCA Computes

Given: X (n samples × d features)

1. Center: `X_c = X - mean(X, axis=0)`  ← subtract per-feature mean
2. Covariance: `C = X_c^T X_c / (n-1)` ← d×d matrix
3. Eigendecompose: `C = V Λ V^T`  ← eigenvalues Λ, eigenvectors V
4. Sort: `V[:, :k]` = top k eigenvectors by eigenvalue magnitude
5. Project: `Z = X_c @ V[:, :k]` ← n×k projected data

### The proportion of variance explained

```
explained_ratio[i] = eigenvalue[i] / sum(all eigenvalues)
cumulative_explained = cumsum(explained_ratio)
```
→ Select k such that cumulative_explained ≥ 0.95 (or whatever threshold).

---

## Gold Standard

- **sklearn**: `sklearn.decomposition.PCA(n_components=k)`
- **scipy**: `scipy.linalg.eigh(C)` for symmetric eigendecomposition (C is symmetric PSD)
- **numpy**: `np.linalg.eig(C)` (less stable), `np.linalg.svd(X_c)` (preferred)
- **R**: `prcomp(X)` (uses SVD), `princomp(X)` (uses eigendecomposition of cov)

### sklearn's actual algorithm

```python
# Full PCA via SVD (more numerically stable than explicit covariance + eig):
# X_c is (n × d), n ≥ d typical
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
# U (n×k), S (k,), Vt (k×d)
components = Vt  # principal components (rows)
explained_variance = (S ** 2) / (n - 1)
loadings = X_c @ Vt.T  # projection = Z
```

**R's prcomp()** also uses SVD. `princomp()` uses eigendecomposition.
SVD is numerically more stable than explicit covariance + eig for n >> d.
For d << n, covariance + eig is more efficient (d×d eig vs n×d SVD).

---

## Tambear Decomposition

### Step 1: Compute per-feature mean
```
means = accumulate(X, All, Expr::Value, Op::Add) / n
```
→ gives d-element mean vector

### Step 2: Center (compute X_c)
Option A: Scatter the means back and subtract (materialized X_c)
Option B: Use RefCentered approach — don't materialize, use refs in scatter_multi_phi

### Step 3: Covariance matrix (d×d)
```
C = tiled_accumulate(X_c^T, X_c, d, d, n) / (n-1)
```
Where:
- Left matrix: X_c^T (d×n) — each row is one centered feature
- Right matrix: X_c (n×d) — this is the transpose again (same as DBSCAN's pattern!)
- TiledEngine op: DotProductOp (Σ a_k * b_k)

The layout: `C[i,j] = Σ_t X_c[t,i] * X_c[t,j]` — feature cross-covariance.

**KEY**: The DBSCAN distance matrix was `dist[i,j] = Σ_k (a_k - b_k)²` (distance op on n×d).
The PCA covariance matrix is `C[i,j] = Σ_t x_t_i * x_t_j` (dot product on d×n).
**Same TiledEngine call, different op, transposed shape.** No new GPU primitive.

### Step 4: Eigendecomposition
For d×d covariance matrix (d = number of features, typically small: d ≤ 1000):
- Use CPU Jacobi iteration for symmetric matrices (O(d³) but d is small)
- Or: delegate to a Rust `eigh` implementation (can use the Power method for truncated)
- Or: CPU LAPACK via `ndarray-linalg` crate for full eigendecomposition

For d large (> 1000): Lanczos algorithm (truncated Krylov method) — O(kd²) for top k.

### Step 5: Projection (n×k output)
```
Z = tiled_accumulate(X_c, V_k, n, k, d)
```
- Left: X_c (n×d)
- Right: V_k (d×k, top k eigenvectors)
- DotProductOp
→ n×k output Z = X_c @ V_k

---

## What's Missing in Tambear's Current Primitives

1. **Eigendecomposition of d×d symmetric matrix**: Not yet in any module.
   For symmetric PSD: Jacobi iteration, Power method, or Lanczos.
   This is the ONLY new primitive needed for PCA.
   CPU implementation is fine for small d (≤ 1000 features is typical).

2. **Materialized centering step**: RefCenteredStatsEngine (stats.rs) is the scaffold.
   RefCentered approach computes {sum, sum_sq} → enables centered covariance computation
   without materializing X_c explicitly.

---

## MSR / Sufficient Stats for PCA

PCA's covariance can be expressed as:
```
C[i,j] = (Σ_t x_{t,i} * x_{t,j}) / (n-1) - n * mean_i * mean_j / (n-1)
       = (Σ_t x_{t,i} * x_{t,j} - n * mean_i * mean_j) / (n-1)
```
→ From sufficient stats: `{n, Σx_i, Σx_j, Σ(x_i * x_j)}` for each pair (i,j).

The MSR for PCA is the **cross-moment matrix**: `d(d+1)/2` cross-products.
For d=10 features: 55 cross-moments. For d=100: 5050. For d=1000: 500k.
This is expensive as a MSR approach for large d. For small d (financial features), manageable.

For the fintek case: the signal farm typically uses d=5..30 features → cross-moment MSR
is feasible. For d=30: 465 cross-products in one accumulate pass.

---

## Randomized PCA

For large n, large d: randomized SVD (Halko, Martinsson, Tropp 2011).
Algorithm:
```
1. Sketch: Y = X @ Omega (random Gaussian Omega, d×k+oversampling)
2. Orthogonalize: Q, _ = QR(Y)  ← Q is n×(k+oversampling)
3. Project: B = Q^T @ X  ← (k+oversampling)×d, small matrix
4. SVD(B): U_B, S, Vt
5. U = Q @ U_B  ← recover left singular vectors
```
Steps 1 and 3 are matrix multiplications → tiled_accumulate.
Step 2 is QR decomposition → new primitive needed (CPU for small matrices is fine).
Step 4 is SVD of a small matrix → CPU LAPACK or Jacobi.

Randomized PCA is what sklearn uses for `svd_solver='randomized'` — O(ndk) vs O(nd²).

---

## Connection to Other Algorithms

PCA is the gateway to:
- **Spectral clustering** (Laplacian eigenvectors, same eigendecomp primitive)
- **Kernel PCA** (uses distance matrix → kernel matrix → eigendecomp)
- **t-SNE** (uses PCA for initial dimensionality reduction)
- **LDA** (Linear Discriminant Analysis: eigendecomp of S_W^{-1} S_B)
- **Factor Analysis** (rotated PCA with noise model)
- **ICA** (non-Gaussian PCA variant)

**All share the covariance/kernel matrix eigendecomposition step.**
Implement it once, serve all.

---

## Singular vs Eigenvalue Approach

| Method | Computes | Best for | Numerically |
|--------|---------|---------|-------------|
| Covariance + eig | C = X^T X, then eig(C) | d << n | Less stable |
| SVD of X_c | Directly decomposes X_c | Any shape | More stable |
| Randomized SVD | Top k only | Large n,d | Same as SVD |

**Recommendation for tambear**: implement SVD for d ≤ 1000, randomized SVD for larger.
The covariance + eig path is faster for small d but SVD is more flexible and stable.

---

## Edge Cases Gold Standard Handles

- **n < d**: Covariance matrix is rank-deficient. Use min(n-1, d) components max.
  sklearn handles this automatically.
- **Constant features**: zero variance → divide by zero in normalization → zero component.
  Filter or replace with NaN.
- **All identical points**: covariance = 0 matrix → all zero eigenvalues → trivial projection.
- **n=1**: Can't compute covariance. Return error.
- **Duplicate features**: covariance is rank-deficient. Power method finds independent subspace.
