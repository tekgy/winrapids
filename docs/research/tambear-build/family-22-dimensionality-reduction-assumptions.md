# Family 22: Dimensionality Reduction — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (PCA/SVD/MDS), C (t-SNE/UMAP iterative optimization)

---

## Core Insight: Three Roads to Low Dimensions

All dimensionality reduction methods answer one question: how do I project high-dimensional data into a low-dimensional space that preserves structure? The three approaches are:

1. **Spectral** (Kingdom A): eigendecomposition of a matrix derived from data. PCA, MDS, Isomap, spectral embedding, LDA. One shot, no iteration.
2. **Neighbor-graph optimization** (Kingdom C): build a graph, optimize an embedding that preserves graph structure. t-SNE, UMAP. Iterative.
3. **Neural** (Kingdom C): learn an encoding function via gradient descent. Autoencoders. Uses F23/F24 infrastructure.

The spectral methods decompose into: `accumulate(Tiled, x_i·x_j, Add)` → eigensolve (F02). The optimization methods decompose into: `accumulate(ByKey(neighbors), attraction_repulsion, Add)` → gradient step (F05). The neural methods are F23/F24 compositions.

**Structural rhyme**: PCA = eigendecomposition of GramMatrix. Spectral clustering = eigendecomposition of graph Laplacian. Same F02 eigen call, different input matrix.

---

## 1. PCA (Principal Component Analysis)

### The Computation
Given data matrix X (n×d), find the k directions of maximum variance.

**Method 1: Covariance eigendecomposition**
```
X_centered = X - mean(X)                    # F06 mean, broadcast subtract
C = (1/(n-1)) · X_centered' · X_centered    # accumulate(Tiled, x_i·x_j, Add) = GramMatrix
eigendecompose(C) → (eigenvalues, V)         # F02 eigen
Z = X_centered · V[:,:k]                     # projection (Tiled accumulate)
```

**Method 2: SVD (preferred for numerical stability)**
```
X_centered = U · Σ · V'                      # F02 SVD
Z = U[:,:k] · Σ[:k,:k]                       # scores
loadings = V[:,:k]                            # loadings
explained_var = σ²_i / Σσ²_j                 # from singular values
```

SVD avoids forming X'X explicitly — better for ill-conditioned data.

### CRITICAL: Center First
PCA on uncentered data computes principal components of the RAW data, not the variance structure. The first component just points toward the mean. **Always center.**

For standardized PCA (correlation-based): center AND divide by std. This is appropriate when variables have different units.

### Variants

| Variant | Method | Use case |
|---------|--------|----------|
| Full PCA | SVD of X_centered | d < 1000 |
| Truncated PCA | Randomized SVD (Halko 2011) | d large, want k << d |
| Incremental PCA | Sequential SVD updates | Streaming / out-of-core |
| Kernel PCA | Eigendecompose kernel matrix K | Non-linear structure |
| Sparse PCA | L1-penalized loadings | Interpretability |
| Robust PCA | MCD covariance (F09) | Outlier resistance |

### Randomized SVD (Halko-Martinsson-Tropp 2011)
For truncated PCA when n,d are large:
```
1. Random projection: Y = X · Ω           # Ω is d×(k+p) random Gaussian
2. QR decomposition: Q, R = qr(Y)         # orthogonal basis for range(X)
3. Project: B = Q' · X                     # (k+p)×d
4. SVD of small matrix: U_B, Σ, V' = svd(B)
5. Recover: U = Q · U_B
```

Oversampling parameter p (typically 5-10) improves approximation. One or two power iterations improve accuracy for slowly decaying spectra.

### Explained Variance
```
explained_ratio_i = σ²_i / Σ_j σ²_j
cumulative_ratio_k = Σ_{i=1}^k σ²_i / Σ_j σ²_j
```

**Scree plot**: plot eigenvalues vs component number. Look for "elbow."
**Kaiser criterion**: keep components with eigenvalue > 1 (for correlation-based PCA).
**Parallel analysis**: compare eigenvalues against random permutation of data (better than Kaiser).

### GPU decomposition
- Centering: `accumulate(All, x_i, Welford)` → broadcast subtract
- Covariance: `accumulate(Tiled{X', X}, x_i·x_j, Add)` — this is GEMM (F02)
- SVD: F02 SVD
- Projection: `accumulate(Tiled{X, V}, x_i·v_j, Add)` — GEMM

---

## 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Algorithm (van der Maaten & Hinton 2008)
1. Compute pairwise affinities in high-D: p_{j|i} = exp(-||x_i-x_j||²/(2σ²_i)) / Σ_{k≠i} exp(-||x_i-x_k||²/(2σ²_i))
2. Symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2n)
3. Initialize low-D embedding Y (random or PCA)
4. Optimize KL(P||Q) where q_ij = (1 + ||y_i-y_j||²)^(-1) / Σ_{k≠l} (1 + ||y_k-y_l||²)^(-1)

### Perplexity → σ
Perplexity = 2^H(P_i) where H is Shannon entropy. Binary search for σ_i that achieves target perplexity.
Typical: perplexity ∈ [5, 50]. Higher = more global structure.

### CRITICAL: Hyperparameter Sensitivity
- **Perplexity**: Too low → disconnected clusters, too high → ball of mud
- **Learning rate**: Too low → poor convergence, too high → unstable
- **Early exaggeration**: Multiply p_ij by 4-12 for first 250 iterations to form tight clusters
- **Iterations**: 1000 minimum, often 2000-5000 needed

### Barnes-Hut Approximation (O(n log n) instead of O(n²))
Build quad-tree (2D) or oct-tree (3D). Approximate distant interactions using center-of-mass.
θ parameter controls accuracy/speed tradeoff (θ=0.5 default).

### GPU decomposition
- Pairwise distances: `accumulate(Tiled{X, X}, ||x_i-x_j||², Add)` — F01 distance matrix
- Affinities: elementwise exp + row normalization
- Gradient: for each point, sum attractive + repulsive forces (parallel per point)
- Update: gradient descent with momentum (F05)

**Kingdom C**: the optimization loop is iterative. Each iteration uses Kingdom A operations.

---

## 3. UMAP (Uniform Manifold Approximation and Projection)

### Algorithm (McInnes, Healy & Melville 2018)
1. Build k-NN graph (F01 distance → F29 kNN)
2. Compute fuzzy simplicial set (membership strengths from distances)
3. Initialize embedding (spectral initialization from graph Laplacian, or random)
4. Optimize cross-entropy loss via SGD:
   ```
   L = Σ_{(i,j)∈edges} [-p_ij·log(q_ij) - (1-p_ij)·log(1-q_ij)]
   ```
   where q_ij = (1 + a·||y_i-y_j||^(2b))^(-1)

### Advantages over t-SNE
- Preserves more global structure (due to cross-entropy loss vs KL)
- Faster (SGD on edges, not all pairs)
- Can project new data (parametric UMAP)
- Deterministic with fixed seed

### Key Parameters
- **n_neighbors** (k): analogous to perplexity. 15-200. Higher = more global.
- **min_dist**: minimum distance in embedding. Controls cluster tightness. 0.0-0.5.
- **metric**: any F01 distance. Default Euclidean.
- **(a, b)**: computed from min_dist by curve fitting. User doesn't set directly.

### GPU decomposition
- k-NN graph: F01 brute-force distance (for d ≤ 50) or approximate NN
- Edge sampling: negative sampling from non-edges (parallel)
- Gradient per edge: elementwise force computation (parallel per edge)
- Update: SGD step (F05)

---

## 4. Classical MDS (Multidimensional Scaling)

### Algorithm
Given distance matrix D (n×n):
```
1. Double centering: B = -½ · H · D² · H    where H = I - (1/n)·11'
2. Eigendecompose B → (λ_i, v_i)
3. Embedding: Y = V[:,:k] · diag(√λ[:k])
```

### Relationship to PCA
When D is Euclidean distance from X: classical MDS = PCA. Same output.

**Metric MDS**: minimize STRESS = √(Σ(d_ij - δ_ij)² / Σ d_ij²). Iterative (Kingdom C, uses F05 optimization).

**Non-metric MDS**: preserve only rank order of distances. Minimize Kruskal's STRESS with isotonic regression.

### GPU decomposition
- Double centering: row mean, grand mean → elementwise (parallel)
- Eigendecompose: F02

---

## 5. Isomap

### Algorithm (Tenenbaum, de Silva & Langford 2000)
1. Build k-NN graph (F01 → F29)
2. Compute geodesic distances: shortest paths on graph (F29 Dijkstra/Floyd-Warshall)
3. Apply classical MDS to geodesic distance matrix

**Isomap = F01 kNN → F29 shortest paths → F22 MDS.** Zero new primitives.

### Failure Mode: holes in the manifold create shortcut paths. Sensitive to k and noise.

---

## 6. LLE (Locally Linear Embedding)

### Algorithm (Roweis & Saul 2000)
1. Find k nearest neighbors for each point
2. Solve for reconstruction weights: min_W Σ ||x_i - Σ_{j∈N(i)} w_ij·x_j||² s.t. Σ_j w_ij = 1
3. Find embedding: min_Y Σ ||y_i - Σ_{j∈N(i)} w_ij·y_j||² s.t. Y'Y = I

Step 2: small linear system per point (F02 solve). Parallel across points.
Step 3: eigendecomposition of (I-W)'(I-W). Bottom k+1 eigenvectors (skip first).

### GPU decomposition
- Neighbors: F01 kNN
- Weights: batch F02 linear solve (parallel per point)
- Embedding: F02 sparse eigendecomposition

---

## 7. ICA (Independent Component Analysis)

### FastICA (Hyvärinen 1999)
```
for each component:
    w = random_init
    repeat:
        w_new = E[x·g(w'x)] - E[g'(w'x)]·w    # fixed-point iteration
        w_new = w_new / ||w_new||                 # normalize
        w_new = orthogonalize against previous w's
    until convergence
```

Where g is a nonlinearity (tanh, exp, cube).

### Kingdom C (iterative per component), each iteration uses:
- `accumulate(All, x_i · g(w'·x_i), Add)` — the expectation
- Gram-Schmidt orthogonalization (F02)

### Preprocessing: whiten data first (PCA to decorrelate, scale to unit variance).

---

## 8. NMF (Non-negative Matrix Factorization)

### Problem
Find W ≥ 0, H ≥ 0 such that X ≈ W·H, where X is n×d, W is n×k, H is k×d.

### Multiplicative Update Rules (Lee & Seung 2001)
```
H ← H · (W'X) / (W'WH)
W ← W · (XH') / (WHH')
```

Element-wise multiply and divide. Guaranteed to decrease the objective (Frobenius or KL).

### GPU decomposition
- W'X, W'W, XH', HH': matrix multiplications (F02 GEMM, Tiled accumulate)
- Element-wise operations: parallel

### Kingdom C: iterative (typically 200-500 iterations), each iteration is pure Kingdom A.

---

## 9. Spectral Embedding (Laplacian Eigenmaps)

### Algorithm (Belkin & Niyogi 2003)
1. Build adjacency graph (k-NN or ε-neighborhood)
2. Compute weight matrix W (heat kernel: w_ij = exp(-||x_i-x_j||²/t))
3. Graph Laplacian: L = D - W where D = diag(W·1)
4. Solve generalized eigenproblem: Ly = λDy
5. Embedding = eigenvectors for smallest non-zero eigenvalues

### Relationship to Spectral Clustering
**Same computation, different use.** Spectral clustering clusters the embedding. Spectral embedding IS the embedding.

### GPU decomposition
- Graph: F01 distance, F29 kNN
- Laplacian: sparse matrix construction (parallel)
- Eigenproblem: F02 sparse eigensolve (Lanczos)

---

## 10. Numerical Stability

### PCA
- **Use SVD, not eigendecomposition of X'X.** Computing X'X squares the condition number: κ(X'X) = κ(X)². SVD avoids this.
- **Center in double precision** when data has large offset (same as F06 RefCentered principle).
- **Truncated SVD for k << min(n,d)**: Halko randomized SVD is both faster and numerically stable.

### t-SNE
- **Affinities underflow**: use log-space computation. log p_{j|i} = -d²/(2σ²) - log(Σ exp(-d²/(2σ²))).
- **Denominator of q_ij → 0**: add ε to prevent log(0).
- **Gradient explosion early**: early exaggeration must decay, not persist.

### UMAP
- **Negative sampling ratio**: too few → poor repulsion, too many → slow. Default 5.
- **Learning rate decay**: reduce over epochs for stable convergence.

### General
- **Random initialization**: PCA initialization is more stable than random for t-SNE/UMAP (provides global structure seed).

---

## 11. Edge Cases

| Algorithm | Edge Case | Expected |
|-----------|----------|----------|
| PCA | k > rank(X) | Extra eigenvalues = 0, warn user |
| PCA | Constant feature (zero variance) | Remove before PCA or eigenvalue = 0 |
| PCA | n < d | Use SVD on X directly (not X'X which is d×d) |
| t-SNE | Duplicate points | Perplexity computation fails (distance = 0). Add tiny jitter. |
| t-SNE | n < perplexity×3 | Perplexity too high, reduce automatically |
| UMAP | Disconnected graph | Separate components embed independently |
| MDS | Non-Euclidean distances | Negative eigenvalues in B. Use |λ| or discard. |
| ICA | Gaussian sources | ICA fails (Gaussians are rotationally symmetric). Warn user. |
| NMF | Zero entries in X | Update rules handle naturally (0/0 set to 0) |
| All | k ≥ d | Embedding dimension ≥ input dimension. Pointless. Error. |

---

## Sharing Surface

### Reuses from Other Families
- **F02 (Linear Algebra)**: SVD, eigendecomposition, linear solve — the core computations
- **F01 (Distance)**: pairwise distances for t-SNE, UMAP, MDS, Isomap, spectral
- **F05 (Optimization)**: gradient descent for t-SNE, SGD for UMAP, NMF convergence
- **F06 (Descriptive)**: centering, standardization (MomentStats for mean/std)
- **F29 (Graph)**: k-NN graph, shortest paths (Isomap), Laplacian (spectral embedding)

### Provides to Other Families
- **F20 (Clustering)**: spectral embedding for spectral clustering
- **F33 (Multivariate)**: PCA scores, loadings for further analysis
- **F37 (Visualization)**: 2D/3D embeddings for scatter plots
- **F27 (TDA)**: Mapper filter functions (PCA projection)
- **F28 (Manifold)**: Isomap/LLE as manifold learning input

### Structural Rhymes
- **PCA covariance = F02 GramMatrix**: same tiled accumulate, different matrix
- **Spectral embedding = PCA on graph Laplacian**: same eigendecomposition, different input
- **MDS = PCA when distances are Euclidean**: formally equivalent
- **NMF update = multiplicative EM**: same alternating optimization structure as F16 GMM

---

## Implementation Priority

**Phase 1** — Core spectral methods (~120 lines):
1. PCA via SVD (full, with centering and explained variance)
2. Randomized/Truncated PCA (Halko et al.)
3. Classical MDS (double centering + eigen)
4. Incremental PCA (streaming SVD updates)

**Phase 2** — Neighbor-graph methods (~200 lines):
5. t-SNE (with Barnes-Hut, perplexity search, early exaggeration)
6. UMAP (k-NN graph, fuzzy simplicial set, SGD embedding)
7. Isomap (F01 kNN → F29 Dijkstra → MDS)
8. Spectral embedding / Laplacian eigenmaps

**Phase 3** — Additional methods (~150 lines):
9. FastICA
10. NMF (multiplicative updates + coordinate descent)
11. LLE
12. Kernel PCA (using F01 kernel functions)
13. Sparse PCA (L1-penalized via F05)

**Phase 4** — Diagnostics (~50 lines):
14. Explained variance plots (PCA)
15. Trustworthiness / continuity metrics
16. Reconstruction error
17. Parallel analysis for component selection

---

## Composability Contract

```toml
[family_22]
name = "Dimensionality Reduction"
kingdom = "Mixed — A (spectral), C (optimization-based)"

[family_22.shared_primitives]
pca = "SVD of centered data matrix → scores, loadings, explained variance"
tsne = "Pairwise affinity → KL-divergence optimization → 2D/3D embedding"
umap = "k-NN graph → cross-entropy optimization → embedding"
mds = "Distance matrix → double centering → eigendecomposition → embedding"
spectral = "Graph Laplacian → eigendecomposition → embedding"

[family_22.reuses]
f02_linear_algebra = "SVD, eigendecomposition, linear solve"
f01_distance = "Pairwise distances, k-NN, kernel matrices"
f05_optimization = "Gradient descent (t-SNE), SGD (UMAP)"
f06_descriptive = "Centering, standardization"
f29_graph = "k-NN graph, shortest paths, Laplacian"

[family_22.provides]
embeddings = "Low-dimensional coordinates (scores, embedding points)"
loadings = "PCA loadings (variable contributions)"
explained_variance = "Variance ratios per component"
transformed_distances = "Geodesic distances (Isomap)"

[family_22.consumers]
f20_clustering = "Spectral embedding for spectral clustering"
f33_multivariate = "PCA scores for further analysis"
f37_visualization = "2D/3D scatter plot coordinates"
f27_tda = "Filter functions for Mapper"
f28_manifold = "Manifold learning embeddings"
```
