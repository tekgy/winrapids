# F28 Sharing Surface: Manifold Operations as Geodesic Distance + EigenDecomp

Created: 2026-04-01T06:28:21-05:00
By: navigator

Prerequisites: F01 complete (DistancePairs), F22 complete (EigenDecomposition).

---

## Core Insight: Manifold Learning = Geodesic Distance + EigenDecomp

All classical manifold learning algorithms:
1. Build a neighborhood graph from DistancePairs
2. Compute geodesic/manifold distances on the graph
3. EigenDecomp of a matrix derived from those distances

The "manifold learning" insight is that Euclidean distance ≠ geodesic distance on a curved
manifold. But once you have the geodesic distances (step 2), the problem is just F22 again.

---

## Isomap

```
1. KNN graph: edges to k nearest neighbors (from DistancePairs)
2. Geodesic distances: shortest paths on KNN graph (Dijkstra / Floyd-Warshall)
3. MDS embedding: PCA/EigenDecomp on geodesic distance matrix
```

**Step 1**: KNN on DistancePairs — F01/F21 infrastructure (zero new code).
**Step 2**: Dijkstra = graph shortest paths — new O(N² log N) algorithm (or Floyd-Warshall O(N³)).
**Step 3**: Classical MDS on geodesic distances = double-centering + EigenDecomp = F22.

```
Double centering: B = -0.5 · H D² H    where H = I - 1/N · 11'
Embedding: Y = V_k Λ_k^{1/2}
```

This is EigenDecomp of the double-centered geodesic distance matrix — **exactly F22's infrastructure** applied to a different input matrix.

**New code**: Dijkstra/Floyd-Warshall shortest paths (~80 lines). Everything else is F01+F22.

---

## Locally Linear Embedding (LLE)

```
1. KNN graph: k nearest neighbors (from DistancePairs)
2. Local reconstruction weights W: minimize ||x_i - Σ_j w_ij x_j||² per point
   subject to: w_ij = 0 if j not in N(i), Σ_j w_ij = 1
3. Embedding: eigenvectors of (I - W)'(I - W)
```

**Step 1**: KNN — F01 infrastructure.
**Step 2**: Local GramMatrix solve. For each point i:
  - Local GramMatrix C_i = (X_{N(i)} - x_i)' (X_{N(i)} - x_i) = RefCenteredStats on k neighbors
  - Constrained solve: w_i = C_i^{-1} 1 / (1' C_i^{-1} 1) — Cholesky on k×k matrix (F10)
**Step 3**: EigenDecomp of sparse (I-W)'(I-W) — F22 on sparse matrix.

**Tambear decomposition**:
- Step 2: K separate Cholesky solves on k×k matrices (k=10-50 typically)
- Step 3: Sparse EigenDecomp — need sparse matrix support or dense approximation

**New code**: ~100 lines. Most is local Cholesky solves (very fast, k<<N) + sparse assembly.

---

## UMAP (Uniform Manifold Approximation and Projection)

UMAP is fundamentally different from Isomap/LLE — it's NOT a linear eigendecomp. It uses:
1. KNN graph + fuzzy weights (from DistancePairs + exponential decay)
2. Low-dimensional graph optimization via gradient descent

```
High-dim fuzzy weights: w_ij = exp(-(d(x_i,x_j) - ρ_i) / σ_i)
where ρ_i = distance to nearest neighbor, σ_i = bandwidth
Low-dim repulsion: optimize y_i to minimize cross-entropy between high/low dim graphs
```

The optimization (step 2) is gradient descent via F05's GradientOracle.
The gradient is derived from the cross-entropy loss between high-dim and low-dim adjacency.

**Tambear path**:
- High-dim weights: `scatter_phi("exp(-(d-rho)/sigma)", ByKNN)` — exponential on distances
- Symmetrize: sparse matrix operation
- Optimization: F05 gradient descent on low-dim embedding coordinates
- Loss: binary cross-entropy between sparse adjacency matrices

UMAP requires sparse matrix operations — NOT currently in tambear's primitive set.
**This is the primary new capability F28 adds**: sparse CSR matrix-vector multiply.

**Phase scope**: Isomap Phase 1, LLE Phase 2, UMAP Phase 3 (requires sparse primitives).

---

## Metric Learning (Adjacent Topic)

Learning a Mahalanobis metric d_M(x, y) = sqrt((x-y)' M (x-y)) from labeled pairs.

**LMNN (Large Margin Nearest Neighbor)**:
```
Minimize: Σ target_neighbors ||x_i - x_j||²_M
  subject to: ||x_i - x_j||²_M < ||x_i - x_k||²_M + 1  (imposters kept far)
M ≽ 0 (positive semidefinite)
```

This is a semidefinite programming (SDP) problem on the metric matrix M.

**Tambear path**: M = L'L where L is learned transformation. Gradient of LMNN loss with respect to L = F05 GradientOracle. Constraint M ≽ 0 is enforced via retraction to PSD cone (eigendecomp + threshold).

**New code needed**: Riemannian gradient on PSD cone (~100 lines). Phase 3.

---

## Geodesic Distance via Dijkstra

The key new primitive for F28 is shortest-path computation on KNN graphs:

```rust
/// Compute all-pairs geodesic distances on the KNN graph.
/// Input: knn_indices[i] = k nearest neighbor indices of point i
///        knn_distances[i] = L2 distances to those neighbors
/// Output: geodesic_distances[N × N] — full shortest-path matrix
pub fn dijkstra_all_pairs(
    knn_indices: &[usize],
    knn_distances: &[f64],
    n: usize,
    k: usize,
) -> Vec<f64>
```

Standard Dijkstra: O(N²) space (output), O(N² log N) time.
For large N (> 10K), geodesic distances are approximate — use sparse approximation.

**GPU opportunity**: Dijkstra is not trivially GPU-parallelizable, but delta-stepping algorithms
exist. Phase 2 optimization. Phase 1: CPU Dijkstra (fast for N < 5000).

---

## MSR Types F28 Produces

```rust
pub struct ManifoldEmbedding {
    pub method: ManifoldMethod,
    pub n_obs: usize,
    pub n_components: usize,

    /// Low-dimensional embedding coordinates. Shape: (N, n_components).
    pub embedding: Arc<Vec<f64>>,

    /// Geodesic distances (if computed). Shape: (N, N).
    pub geodesic_distances: Option<Arc<Vec<f64>>>,

    /// Reconstruction error (for Isomap: fraction of variance preserved).
    pub reconstruction_error: f64,

    /// Neighborhood graph: indices per point. Shape: (N, k).
    pub knn_indices: Arc<Vec<usize>>,
}

pub enum ManifoldMethod {
    Isomap { k: usize },
    Lle { k: usize },
    Umap { n_neighbors: usize, min_dist: f64 },  // Phase 3
}
```

---

## Build Order

**Phase 1 (Isomap)**:
1. KNN graph from DistancePairs — ArgMinOp K times (F01 infrastructure, ~20 lines wiring)
2. Dijkstra all-pairs: standard CPU Dijkstra for N < 5000 (~80 lines)
3. Double-centering: `B = -0.5 H D² H` — matrix arithmetic (~15 lines)
4. EigenDecomp(B): reuse F22 infrastructure (~5 lines wiring)
5. `ManifoldEmbedding` struct (~20 lines)
6. Tests: `sklearn.manifold.Isomap` — match embeddings up to sign/rotation (Procrustes alignment needed for comparison)

**Phase 2 (LLE)**:
1. Local GramMatrix per point: RefCenteredStats on k neighbors (~40 lines)
2. Constrained solve: Cholesky on k×k, divide by 1'C^{-1}1 (~30 lines)
3. Sparse assembly of (I-W)'(I-W) — dense approximation: W is sparse, compute explicitly
4. EigenDecomp of assembled matrix — F22 on N×N matrix

**Phase 3 (UMAP)**:
- Requires sparse CSR operations — new capability
- F05 gradient descent on embedding coordinates

**Gold standards**:
- Python `sklearn.manifold.Isomap`, `sklearn.manifold.LocallyLinearEmbedding`
- Python `umap-learn` for UMAP
- Alignment: apply Procrustes rotation before comparing embeddings (sign ambiguity)

---

## Structural Rhyme

**Isomap : PCA :: Graph Distance : Euclidean Distance**

- PCA (F22): EigenDecomp of covariance matrix (Euclidean inner product structure)
- Isomap (F28): EigenDecomp of geodesic distance matrix (graph distance structure)
- Both produce orthogonal projections that maximize variance under their respective metrics
- The only difference: the distance metric used to build the kernel matrix

This confirms: F28 = F22 with a different distance metric. Once geodesic distances exist, the rest IS F22.

---

## The Lab Notebook Claim

> Manifold learning is F22 (EigenDecomposition) applied to a different distance matrix — geodesic instead of Euclidean. The geodesic distance computation (Dijkstra on KNN graph) is the only genuinely new code: ~80 lines for Phase 1. Everything after Dijkstra is double-centering + F22's EigenDecomp infrastructure. Isomap = geodesic distances + classical MDS. LLE = local Cholesky solves + sparse EigenDecomp. UMAP is genuinely different (gradient-based optimization on fuzzy adjacency) and requires sparse matrix primitives — a real new capability for Phase 3.
