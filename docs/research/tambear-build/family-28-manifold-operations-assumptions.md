# Family 28: Manifold Operations — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (distance/kernel computation) + C (iterative optimization for UMAP/t-SNE)

---

## Core Insight: Isomap:PCA :: GeodesicDistance:EuclideanDistance

Every manifold learning algorithm replaces Euclidean distance with a distance that respects the manifold structure. The computation decomposes into:
1. Build local neighborhood graph (F01 k-NN + F29 graph)
2. Compute manifold-aware distance/similarity
3. Embed via eigendecomposition (F22) or optimization (F05)

---

## 1. Isomap (Tenenbaum et al. 2000)

### Algorithm
```
1. Build k-NN graph from data (F01 distance → F29 graph)
2. Compute shortest paths between all pairs (F29 Dijkstra/Floyd-Warshall) → geodesic distance matrix D_G
3. Classical MDS on D_G: center D_G², eigendecompose → embedding (F22)
```

### Classical MDS (Multidimensional Scaling)
Given distance matrix D:
```
B = -½ H · D² · H    where H = I - (1/n)·11'    (centering matrix)
```
B should be PSD. Eigendecompose B = UΛU'. Embedding: X = U_d · Λ_d^{1/2}.

### Implementation
- F01: k-NN distance computation
- F29: Dijkstra from all sources (~80 lines as noted in task)
- F22: eigendecomposition of centered squared distance matrix

### Residual Variance
```
1 - R²(D_G, D_embed)
```
Use to select dimensionality d (elbow in residual variance plot).

### Edge Cases
- Disconnected graph: geodesic distance = ∞ between components. Either: increase k, or embed components separately.
- Short-circuit edges: if k too large, edges cut across the manifold → geodesic distances are too short. Keep k small.
- Non-convex manifold: Isomap fails (shortest path ≠ geodesic). Use LLE or UMAP instead.
- **Landmark Isomap**: For large n, compute Dijkstra from L << n landmark points only. Nyström extension for remaining points.

### Kingdom: A (distances) + C (all-pairs shortest paths) + A (eigendecomposition)

---

## 2. Locally Linear Embedding (LLE — Roweis & Saul 2000)

### Algorithm
```
1. Find k nearest neighbors for each point x_i
2. For each point, find weights W_ij minimizing:
   ε(W) = Σ_i ‖x_i - Σ_j W_{ij}·x_j‖²
   s.t.  Σ_j W_{ij} = 1,  W_{ij} = 0 if j not neighbor of i
3. Find embedding Y minimizing:
   Φ(Y) = Σ_i ‖y_i - Σ_j W_{ij}·y_j‖²
   s.t.  Y'Y = I    (centering and unit covariance)
```

### Step 2: Local Cholesky
For each point i with k neighbors, solve the k×k system:
```
C_i · w_i = 1_k    where C_{jl} = (x_i - x_j)'(x_i - x_l)
```
Then normalize: w_i = w_i / (1'w_i).

### Step 3: Sparse Eigenvalue Problem
```
M = (I - W)'(I - W)
```
Find bottom d+1 eigenvectors of M (skip the constant eigenvector λ=0).

### Implementation
- F01: k-NN
- Per-point Cholesky: k×k system per point (embarrassingly parallel across points)
- F22: sparse eigendecomposition of M

### Edge Cases
- k < d: underdetermined local reconstruction → regularize C_i + εI
- Regularization: ε = 10⁻³ · tr(C_i) standard

### Kingdom: A (k-NN, parallel per-point solves) + A (eigendecomposition)

---

## 3. Laplacian Eigenmaps (Belkin & Niyogi 2003)

### Algorithm
```
1. Build k-NN or ε-neighborhood graph
2. Weight edges: W_{ij} = exp(-‖x_i - x_j‖²/t)    (heat kernel)
   or W_{ij} = 1 (simple adjacency)
3. Compute graph Laplacian: L = D - W
4. Solve generalized eigenvalue problem: L·y = λ·D·y
5. Embedding: bottom d non-trivial eigenvectors
```

### Structural Rhyme with Spectral Clustering (F20)
Spectral clustering = Laplacian eigenmap + K-means. Same eigenvectors, different consumer.

### Implementation: F29 graph Laplacian + F22 generalized eigendecomposition.

### Edge Cases
- Heat kernel parameter t: use median of k-NN distances as heuristic
- Disconnected graph: embed components separately (otherwise Laplacian has multiple zero eigenvalues)

---

## 4. t-SNE (van der Maaten & Hinton 2008)

### Algorithm
```
1. Compute pairwise affinities in high-D:
   p_{j|i} = exp(-‖x_i-x_j‖²/(2σ²_i)) / Σ_{k≠i} exp(-‖x_i-x_k‖²/(2σ²_i))
   p_{ij} = (p_{j|i} + p_{i|j}) / (2n)

2. Initialize embedding Y (PCA or random)

3. Minimize KL divergence via gradient descent:
   C = KL(P || Q) = Σ_{ij} p_{ij} log(p_{ij}/q_{ij})
   where q_{ij} = (1 + ‖y_i-y_j‖²)⁻¹ / Σ_{k≠l} (1 + ‖y_k-y_l‖²)⁻¹
```

### Perplexity
σ_i is set so that perplexity of P_i matches target (typically 5-50):
```
Perp(P_i) = 2^{H(P_i)} = 2^{-Σ_j p_{j|i} log₂ p_{j|i}}
```
Binary search for σ_i that gives desired perplexity.

### Barnes-Hut Approximation
For large n: use tree-based approximation for repulsive forces (q_{ij} terms).
Complexity: O(n log n) instead of O(n²).

### Gradient
```
∂C/∂y_i = 4 Σ_j (p_{ij} - q_{ij})(y_i - y_j)(1 + ‖y_i - y_j‖²)⁻¹
```
= attractive forces (p_{ij} terms) - repulsive forces (q_{ij} terms).

### Early Exaggeration
Multiply p_{ij} by factor (4-12) for first 250 iterations → forces clusters to separate before fine-tuning.

### CRITICAL Pitfalls
- **Not distance-preserving**: cluster sizes and inter-cluster distances are NOT meaningful
- **Perplexity-dependent**: different perplexities show different structure
- **Non-convex optimization**: results depend on initialization. Run multiple times.
- **Crowding problem**: why Student-t distribution (heavy-tailed) is used for q, not Gaussian

### Kingdom: C (iterative gradient descent on KL divergence)

---

## 5. UMAP (McInnes et al. 2018)

### Algorithm (simplified)
```
1. Build fuzzy k-NN graph:
   For each point, find k nearest neighbors
   σ_i = distance to k-th nearest neighbor (adaptive bandwidth)
   w_{ij} = exp(-(d(x_i,x_j) - ρ_i)/σ_i)    where ρ_i = distance to nearest neighbor

2. Symmetrize: w̃_{ij} = w_{ij} + w_{ji} - w_{ij}·w_{ji}

3. Initialize embedding (spectral or random)

4. Optimize cross-entropy loss:
   C = Σ_{(i,j)∈edges} [-w̃_{ij}·log(q_{ij}) - (1-w̃_{ij})·log(1-q_{ij})]
   where q_{ij} = (1 + a·‖y_i-y_j‖^{2b})⁻¹
```

### Parameters a, b
Fit to match the fuzzy set structure. Default: a ≈ 1.929, b ≈ 0.7915 (for min_dist=0.1).

### Stochastic Gradient Descent
Sample positive edges (from k-NN graph) and negative edges (random pairs) per epoch. Much faster than t-SNE's full gradient.

### Advantages over t-SNE
- Faster (SGD on edges, not full pairwise)
- Better global structure preservation
- Theoretically grounded (category theory / fuzzy topology)
- Can transform new data points (parametric UMAP)

### Implementation
- F01: k-NN computation
- F05: SGD optimizer for embedding
- Sparse operations (edge sampling)

### Kingdom: C (iterative SGD). Phase 3 as noted in task (~more lines).

---

## 6. Diffusion Maps (Coifman & Lafon 2006)

### Algorithm
```
1. Build kernel matrix: K_{ij} = exp(-‖x_i-x_j‖²/ε)
2. Normalize: K̃ = D⁻¹K (row-normalize) → Markov matrix
3. Further normalize for density: K̃^(α)
4. Eigendecompose: K̃·ψ_k = λ_k·ψ_k
5. Diffusion coordinates: Ψ_t(x) = [λ₁ᵗψ₁(x), ..., λ_dᵗψ_d(x)]
```

### Diffusion Distance
```
D_t²(x_i, x_j) = Σ_k λ_k^{2t} (ψ_k(x_i) - ψ_k(x_j))²
```
t = diffusion time (scale parameter). Large t: global structure. Small t: local.

### Relation to Laplacian Eigenmaps
Diffusion maps with α=1 normalization are equivalent to normalized Laplacian eigenmaps as ε → 0.

### Implementation: F01 kernel matrix + eigendecomposition (F22).

---

## 7. Multidimensional Scaling (MDS)

### Classical MDS (Torgerson)
Already described under Isomap. Given distance matrix → centering → eigendecomposition.

### Metric MDS (Stress Minimization)
```
min_Y Stress = √(Σ_{i<j} (d_{ij} - ‖y_i - y_j‖)² / Σ_{i<j} d²_{ij})
```
SMACOF algorithm (Scaling by MAjorizing a COmplicated Function):
iterative update Y_new = n⁻¹ · B(Y_old) · Y_old where B is derived from distances.

### Non-Metric MDS
```
min_Y Stress = √(Σ_{i<j} (f(d_{ij}) - ‖y_i - y_j‖)² / Σ_{i<j} ‖y_i - y_j‖²)
```
where f is a monotone function (isotonic regression). Preserves only rank order of distances.

---

## 8. Geodesic Distance on Manifolds

### From k-NN Graph
Graph shortest path ≈ geodesic distance. Better with larger k but risk of short-circuits.

### Heat Method (Crane et al. 2017)
1. Solve heat equation: (M - t·L)u = δ_s (impulse at source)
2. Compute gradient: X = -∇u/|∇u| (normalized gradient points toward source)
3. Solve Poisson: L·φ = ∇·X (divergence of normalized gradient)
φ gives approximate geodesic distances. Works on meshes.

---

## Edge Cases (Common Across Methods)

- **Ambient dimension >> intrinsic dimension**: All methods need k large enough to capture local geometry but small enough to avoid short-circuits
- **Noise**: High noise → local neighborhoods are unreliable → increase k or use denoising
- **Non-uniform density**: Some methods (t-SNE, UMAP) adapt bandwidth per point. Others (Isomap, LLE) don't → underperform on varying density
- **Out-of-sample**: Isomap (Nyström), UMAP (parametric), t-SNE (parametric variant). LLE has no natural extension.

---

## Sharing Surface

### Reuse from Other Families
- **F01 (Distance)**: k-NN, distance matrices, kernel matrices
- **F22 (Dimensionality Reduction)**: PCA initialization, eigendecomposition
- **F29 (Graph Algorithms)**: Dijkstra (Isomap), graph Laplacian (Laplacian eigenmaps)
- **F05 (Optimization)**: SGD for UMAP, gradient descent for t-SNE
- **F20 (Clustering)**: Spectral clustering shares Laplacian eigenvectors

### Consumers of F28
- **F22 (Dimensionality Reduction)**: Manifold-aware embeddings
- **F27 (TDA)**: Manifold → point cloud → persistent homology
- **F20 (Clustering)**: Graph-based manifold clustering

### Structural Rhymes
- **Isomap = PCA on geodesic distances**: same eigendecomposition, different distance metric
- **LLE = local PCA + global consistency**: each neighborhood is linear, stitched together
- **Laplacian eigenmap = spectral clustering without K-means**: same eigenvectors
- **t-SNE = KL minimization with Student-t kernel**: same as VI (F34) with different KL direction
- **UMAP = cross-entropy on fuzzy simplicial sets**: categorical / topological foundation
- **Diffusion maps = Markov chain stationary analysis**: same as F17/F29 random walks

---

## Implementation Priority

**Phase 1** — Isomap + LLE (~80 lines each):
1. Isomap (F01 k-NN → F29 Dijkstra → F22 MDS)
2. LLE (k-NN → per-point Cholesky → sparse eigendecomposition)
3. Laplacian Eigenmaps (k-NN → heat kernel → generalized eigenvalue)
4. Classical MDS (distance matrix → centering → eigendecomposition)

**Phase 2** — Diffusion + MDS (~100 lines):
5. Diffusion Maps (kernel matrix → normalize → eigendecompose)
6. Metric MDS (SMACOF iteration)
7. Non-metric MDS (isotonic regression + SMACOF)

**Phase 3** — t-SNE + UMAP (~200 lines):
8. t-SNE (perplexity calibration → gradient descent → Barnes-Hut)
9. UMAP (fuzzy k-NN → cross-entropy SGD)
10. Parametric UMAP (neural network embedding)

---

## Composability Contract

```toml
[family_28]
name = "Manifold Operations"
kingdom = "A (distance/kernel) + C (iterative embedding optimization)"

[family_28.shared_primitives]
knn_graph = "k-NN graph from F01 distances"
geodesic_distance = "Shortest path on k-NN graph (F29 Dijkstra)"
local_reconstruction = "Per-point Cholesky for LLE weights"
embedding_optimization = "SGD/GD for t-SNE/UMAP (F05)"

[family_28.reuses]
f01_distance = "k-NN, distance matrices, kernel matrices"
f22_eigendecomp = "MDS, Laplacian eigenmaps, diffusion maps"
f29_graph = "Dijkstra for geodesic, graph Laplacian"
f05_optimizer = "SGD for UMAP, GD for t-SNE"
f20_clustering = "Spectral clustering shares Laplacian eigenvectors"

[family_28.provides]
isomap = "Isometric embedding via geodesic distances"
lle = "Locally Linear Embedding"
laplacian_eigenmap = "Laplacian eigenmap embedding"
tsne = "t-SNE embedding (2D/3D visualization)"
umap = "UMAP embedding (fast, preserves global structure)"
diffusion_map = "Diffusion map embedding"
mds = "Classical, metric, and non-metric MDS"

[family_28.session_intermediates]
knn_graph = "KNNGraph(data_id, k) — neighbor indices + distances"
geodesic_dist = "GeodesicDist(data_id, k) — all-pairs geodesic"
embedding = "Embedding(data_id, method, d) — low-dimensional coordinates"
```
