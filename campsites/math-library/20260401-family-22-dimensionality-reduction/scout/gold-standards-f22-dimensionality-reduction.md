# F22 Dimensionality Reduction — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 22 (Dimensionality Reduction).
Central primitive: EigenDecomposition of covariance/Gram matrix.
This is the Level-2 MSR that serves PCA, Factor Analysis (F14), spectral clustering (F20),
and graph Laplacian embedding (F29).

---

## PCA: The Gateway to the Eigendecomposition Kingdom

PCA is the first consumer of EigenDecomposition. Once implemented, the same eigendecomp
path serves F14, F20 spectral clustering, and F29 graph embedding for free.

### What PCA Computes

```
1. Center data: X_c = X - mean(X, axis=0)    [subtract column means]
2. Covariance:  C = (1/(n-1)) · X_c' · X_c  [p×p covariance matrix = GramMatrix]
3. Eigen:       C = V · Λ · V'               [V = eigenvectors, Λ = eigenvalues]
4. Project:     Z = X_c · V[:, :k]           [n×k scores matrix]
```

Step 2: `GramMatrix = accumulate(Tiled(p,p), DotProduct, Add) / (n-1)` on centered data.
Step 3: EigenDecomposition on p×p CPU matrix (p << n in typical use).
Step 4: `scores = accumulate(Tiled(n,k), DotProduct, Add)` — matrix multiply.

The covariance matrix is the SAME GramMatrix from F10, just on centered data.
If X is standardized (z-scored), covariance = correlation matrix.

---

## R: base and prcomp — The Canonical PCA

```r
# Base R PCA (uses SVD internally):
pca <- prcomp(X)                  # centers and scales by default? NO — center=TRUE, scale=FALSE
pca <- prcomp(X, center=TRUE, scale.=FALSE)  # center only (covariance PCA)
pca <- prcomp(X, center=TRUE, scale.=TRUE)   # standardize (correlation PCA)
pca <- prcomp(X, center=FALSE)               # no centering (works on raw Gram matrix)

# Output:
pca$rotation    # p×p matrix of eigenvectors (principal component loadings)
pca$x           # n×p scores matrix Z = X_c · V
pca$sdev        # sqrt(eigenvalues) = standard deviations of components
pca$sdev^2      # eigenvalues = variance explained per component
pca$center      # column means used for centering

# Proportion of variance explained:
cumsum(pca$sdev^2) / sum(pca$sdev^2)

# Alternative (uses spectral decomposition on covariance, slightly different for small n):
pca2 <- princomp(X, cor=FALSE)   # cov matrix, n denominator (not n-1!)
pca2 <- princomp(X, cor=TRUE)    # correlation matrix
```

**Trap: prcomp vs princomp**:
- `prcomp`: uses SVD of X_c directly. Numerically superior. Denominator = n-1.
- `princomp`: uses eigendecomposition of C = X_c'X_c/n. Denominator = n (population).
Eigenvalues differ by factor (n-1)/n. Use `prcomp` as the gold standard.

**Trap: scale. parameter**:
- `prcomp(X, scale.=TRUE)` = PCA of the correlation matrix (z-scored features)
- `prcomp(X, scale.=FALSE)` = PCA of the covariance matrix
- These give COMPLETELY DIFFERENT eigenvectors when features have different scales.
Document which convention tambear uses.

### R: Additional PCA Functionality

```r
# Biplot:
biplot(pca)

# Scree plot:
plot(pca, type="l")

# Prediction (project new data onto existing PCs):
predict(pca, newdata=X_new)   # = (X_new - pca$center) %*% pca$rotation

# Reconstruction:
X_reconstructed <- pca$x[, 1:k] %*% t(pca$rotation[, 1:k])  # k-component reconstruction
X_reconstructed <- sweep(X_reconstructed, 2, pca$center, "+")  # add back mean
```

---

## Python: sklearn.decomposition — Full Suite

```python
from sklearn.decomposition import (
    PCA,                  # standard PCA, SVD-based
    IncrementalPCA,       # online PCA for large data
    KernelPCA,            # nonlinear PCA via kernel trick
    SparsePCA,            # sparse loadings
    MiniBatchSparsePCA,   # mini-batch variant
    FactorAnalysis,       # probabilistic FA (F14 boundary)
    FastICA,              # Independent Component Analysis
    NMF,                  # Non-negative Matrix Factorization
    TruncatedSVD,         # SVD on sparse matrices (LSA)
    DictionaryLearning,   # sparse coding
)

# Standard PCA:
pca = PCA(n_components=k)
pca.fit(X)

# Results:
pca.components_              # k×p matrix of eigenvectors (k principal components)
pca.explained_variance_      # eigenvalues (variance per component)
pca.explained_variance_ratio_ # proportion of variance per component
pca.singular_values_         # singular values of X_c
pca.mean_                    # column means

Z = pca.transform(X)         # n×k scores = (X - mean) @ components_.T
X_back = pca.inverse_transform(Z)  # reconstruction

# Cumulative variance explained:
np.cumsum(pca.explained_variance_ratio_)

# For large matrices (randomized SVD):
pca_random = PCA(n_components=k, svd_solver='randomized', random_state=42)
```

**Trap: sklearn uses SVD, not eigendecomposition directly**.
`pca.components_` = rows of right singular matrix V (shape k×p, not p×k).
`pca.explained_variance_` = singular_values²/(n-1).
Eigenvectors from R's `prcomp$rotation` = `pca.components_.T` in sklearn (transposed!).

**Trap: sign ambiguity**.
PCA eigenvectors are defined up to sign flip. sklearn and R may return -v where the other
returns +v. Normalize by making the largest absolute component positive before comparing.

### Kernel PCA

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=k, kernel='rbf', gamma=0.1)
Z = kpca.fit_transform(X)

# Kernels: 'linear' (= standard PCA), 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'
# With 'precomputed': pass the kernel matrix directly
K_matrix = rbf_kernel(X, gamma=0.1)  # compute kernel matrix first
kpca2 = KernelPCA(n_components=k, kernel='precomputed')
Z2 = kpca2.fit_transform(K_matrix)
```

**Connection to tambear**: `kernel='precomputed'` means: compute `KernelMatrix(RBF)` via
TiledEngine (free once DistancePairs is cached), pass to KernelPCA's eigendecomposition.
Kernel PCA = EigenDecomposition of KernelMatrix — same primitive, different input.

---

## UMAP (not in sklearn core, separate library)

```python
import umap

reducer = umap.UMAP(
    n_components=2,           # target dimension
    n_neighbors=15,           # local neighborhood size (like k in KNN)
    min_dist=0.1,             # minimum distance in embedding
    metric='euclidean',       # or 'cosine', 'manhattan', etc.
    random_state=42,
)
Z = reducer.fit_transform(X)

# Note: UMAP is non-parametric and stochastic.
# Reproducibility requires random_state.
# NOT a linear projection — cannot use for reconstruction.
```

**Tambear observation**: UMAP's first step is KNN graph construction — same as KNN (F01).
Once TamSession has `Arc<KnnResult>`, UMAP can reuse it.
The remaining steps (fuzzy simplicial set, gradient descent optimization) are CPU-side.

---

## t-SNE

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30.0,           # relates to neighborhood size (~30-50 typical)
    learning_rate='auto',      # or 200.0
    n_iter=1000,
    metric='euclidean',        # or 'precomputed' for custom distances
    random_state=42,
    method='barnes_hut',       # O(n log n); 'exact' = O(n²) for small n
)
Z = tsne.fit_transform(X)
# OR with precomputed distance matrix:
Z = TSNE(metric='precomputed').fit_transform(D)  # D = n×n distance matrix
```

**Tambear observation**: `metric='precomputed'` means t-SNE can consume `Arc<DistanceMatrix>`
from TamSession directly. The expensive O(n²d) GPU compute is already done.
t-SNE then runs its gradient descent on the CPU side.

---

## EigenDecomposition: The Shared Primitive

All of these consume the same eigendecomposition:

| Algorithm | Input matrix | Output consumed |
|-----------|-------------|----------------|
| PCA | Covariance = X_c'X_c/(n-1) | Top-k eigenvectors |
| Factor Analysis | Correlation matrix | Specific rotation of eigenvectors |
| Spectral clustering (F20) | Graph Laplacian L = D - W | Bottom-k eigenvectors |
| Kernel PCA | Centered kernel matrix K_c | Top-k eigenvectors |
| UMAP (initial step) | KNN affinity matrix | Not eigendecomp — but graph-based |
| Graph Laplacian (F29) | Normalized L = D^{-1/2}LD^{-1/2} | Bottom-k eigenvectors |

For p×p matrices with p << n (typical): eigendecomposition is CPU-side O(p³).
For n×n kernel matrices (kernel PCA): if n is large, truncated eigendecomposition needed.

```python
import numpy as np
# Full eigendecomposition (p×p):
eigenvalues, eigenvectors = np.linalg.eigh(C)  # C = symmetric p×p covariance
# Note: eigh for symmetric, NOT eig (eigh is more stable for symmetric matrices)

# Truncated (top-k only, for large n×n):
from scipy.sparse.linalg import eigsh
eigenvalues, eigenvectors = eigsh(K, k=10, which='LM')  # k largest eigenvalues
```

**Trap: `eig` vs `eigh`**: for symmetric matrices, always use `eigh` (symmetric eigendecomposition).
`eig` may return complex eigenvalues for nearly-singular matrices due to floating point.
`eigh` guarantees real eigenvalues for symmetric input.

---

## Validation Targets

```python
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(42)
n, p = 100, 5
# Generate data with known covariance structure:
C_true = np.diag([5.0, 3.0, 1.0, 0.5, 0.2])  # 5 components with decreasing variance
X = np.random.randn(n, p) @ np.sqrt(C_true)

pca = PCA()
pca.fit(X)
print("Explained variance ratios:", pca.explained_variance_ratio_)
# Expected: ≈ [5/9.7, 3/9.7, 1/9.7, 0.5/9.7, 0.2/9.7] (proportional to diagonal values)
# Note: exact values depend on sample (not population) covariance

# Tambear oracle: compare eigenvalues from covariance GramMatrix against sklearn
C_sample = (X - X.mean(0)).T @ (X - X.mean(0)) / (n-1)
eigvals, eigvecs = np.linalg.eigh(C_sample)
eigvals_sorted = eigvals[::-1]   # sort descending (eigh gives ascending order)
print("Manual eigenvalues:", eigvals_sorted)
print("sklearn variance:  ", pca.explained_variance_)
# Should match to < 1e-10
```

**Key**: `np.linalg.eigh` returns eigenvalues in ascending order. sklearn sorts in descending.
Always flip: `eigvals[::-1]`, `eigvecs[:, ::-1]`.

---

## Memory Budget

For n=100,000 points, p=100 features:
- GramMatrix (p×p = 100×100): 80 KB — trivial
- Scores (n×k = 100,000×10): 8 MB — manageable

For kernel PCA with n=100,000:
- Kernel matrix (n×n): 80 GB — infeasible on GPU
- Must use Nyström approximation or random features instead

**Practical limit**: full kernel PCA feasible up to n ≈ 50,000 on 24 GB GPU (24GB/(8 bytes) = 3B elements → sqrt = 54,772 × 54,772 matrix).
