# F22 Sharing Surface: EigenDecomposition as the Level-2 MSR Backbone

Created: 2026-04-01T05:53:57-05:00
By: navigator

Prerequisites: F10 complete (GramMatrix). `cholesky.rs` already exists.

---

## EigenDecomposition: The Critical Level-2 MSR

GramMatrix is Level-1 (raw accumulate). EigenDecomposition is Level-2 (derived from GramMatrix). It unlocks four families simultaneously:

| Consumer | What it uses | Status |
|---------|-------------|--------|
| F22 (PCA) | Top-k eigenvectors as projection matrix | This family |
| F14 (Factor Analysis) | Same correlation matrix eigendecomp + rotation | Depends on F22 |
| F20 (Spectral clustering) | Graph Laplacian eigendecomp → cluster in eigenvector space | Depends on F22 |
| F33 (CCA) | Generalized eigenvalue: Σ_xy·Σ_yy⁻¹·Σ_yx·a = λ·Σ_xx·a | Depends on F22 |

**EigenDecomposition is the single most important Level-2 MSR to implement.** One correct implementation propagates through 4+ families.

---

## PCA: The Clean Case

**What PCA computes**:
```
Given data X (N × p), centered (subtract column means):
1. Covariance matrix: S = X'X / (N-1) = GramMatrix / (N-1)
2. Eigendecompose: S = V Λ Vᵀ  where V = eigenvectors (p × p), Λ = diagonal eigenvalues
3. Project to k dimensions: Y = X · V[:, :k]  (N × k scores)
```

**What comes from F10 for free**:
- `GramMatrix` = X'X (already computed, centered)
- Column means (for uncentering predictions)
- Dividing by N-1 is element-wise

**New step**: eigendecompose the p × p symmetric matrix S.

---

## Eigendecomposition Algorithm

For symmetric positive semidefinite matrices (covariance matrices are always PSD):

### Option 1: LAPACK `dsyevd` (recommended for production)

`dsyevd` is the divide-and-conquer symmetric eigensolver. It's:
- O(p³) time, O(p²) memory
- More stable than QR iteration for nearly-degenerate eigenvalues
- Available via the `lapack-src` crate (links to OpenBLAS, MKL, or Netlib)
- Returns eigenvalues in ascending order (reverse for PCA: descending = most variance first)

```rust
// Rough call structure (pseudo-code):
let (eigenvalues, eigenvectors) = lapack::dsyevd(
    symmetric_matrix,  // p×p, row-major
    p,
    jobz: 'V',    // compute eigenvectors
    uplo: 'U',    // upper triangle
)?;
```

### Option 2: Pure Rust QR iteration with Householder tridiagonalization

For p < 50 (typical PCA use case with p = number of features):
1. Householder reduction to tridiagonal form: O(p²) work
2. QR iteration on tridiagonal: O(p²) per iteration, converges in O(p) iterations
3. Back-transform: O(p²)

Total: O(p³) but with small constant. Fast enough for p < 200 without LAPACK dependency.

**Phase 1**: pure Rust QR iteration (no LAPACK, no external dep)
**Phase 2**: LAPACK `dsyevd` for p > 200

### Option 3: Power iteration (for top-k only)

If only the top-k << p eigenvectors are needed:
```
For i = 1..k:
    v = random vector
    for iter in 1..max_iter:
        v = S·v  (matrix-vector multiply: O(p²))
        v = v / ‖v‖
    deflate S = S - λᵢ·vᵢ·vᵢᵀ
```

O(k × p² × iterations). Faster than full eigendecomp when k << p. Good for PCA with k = 2 or 3 visualization dimensions.

**Phase 1**: power iteration for common case (top-k with k << p)
**Phase 2**: full symmetric eigendecomp for Factor Analysis and CCA which need all eigenvalues

---

## EigenDecomposition MSR Type

```rust
/// Eigendecomposition of a symmetric matrix. Used by PCA, Factor Analysis, spectral clustering.
/// Eigenvalues in DESCENDING order (most variance = most important first).
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    pub dim: usize,
    /// Eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₚ.
    pub eigenvalues: Arc<Vec<f64>>,
    /// Eigenvectors in columns (column-major). Column j = eigenvector for λⱼ.
    /// Shape: (dim × dim), stored as flat Vec of length dim².
    pub eigenvectors: Arc<Vec<f64>>,
}

impl EigenDecomposition {
    /// Proportion of variance explained by first k components.
    pub fn explained_variance_ratio(&self, k: usize) -> f64 {
        let total: f64 = self.eigenvalues.iter().sum();
        self.eigenvalues[..k].iter().sum::<f64>() / total
    }

    /// Number of components needed to explain frac of variance.
    pub fn components_for_variance(&self, frac: f64) -> usize { ... }

    /// PCA projection matrix (top-k eigenvectors): shape (dim, k).
    pub fn projection_matrix(&self, k: usize) -> &[f64] {
        // Returns first k columns of eigenvectors matrix
    }
}
```

Add to `IntermediateTag`:
```rust
EigenDecomposition {
    gram_id: DataId,        // which GramMatrix this came from
    matrix_type: MatrixType, // Covariance | Correlation | Laplacian | Custom
},

pub enum MatrixType {
    Covariance,    // X'X / (N-1)
    Correlation,   // normalized to correlation matrix (diagonal = 1)
    Laplacian,     // graph Laplacian for spectral clustering
}
```

---

## PCA Extraction from EigenDecomposition

```rust
pub struct PcaResult {
    pub n_components: usize,
    pub components: Vec<f64>,          // top-k eigenvectors, shape (p, k) column-major
    pub explained_variance: Vec<f64>,  // λ₁, ..., λₖ
    pub explained_variance_ratio: Vec<f64>,  // λᵢ / Σλ
    pub singular_values: Vec<f64>,     // √(λᵢ * (N-1))
}
```

**PCA scores (projection)**: Y = X_centered · components. This is a matrix multiply:
```rust
// Projection: O(N × p × k)
// Use: scatter(Tiled(N, k), x[n,p] * component[p, c], Add)
// OR: standard BLAS dgemm if available
```

For typical k = 2 or 3 (visualization), this is ~O(N × p × 3) = very fast.

---

## What F14/F22/F33 Share with F22 PCA

### Factor Analysis (F14)

PCA + rotation: factor analysis is PCA on the correlation matrix (not covariance), followed by rotation (Varimax, Promax) to find interpretable loadings.

**Sharing with F22**:
1. Correlation matrix = GramMatrix normalized (same GramMatrix, different extraction)
2. EigenDecomposition on correlation matrix = same algorithm as PCA
3. Rotation (Varimax): post-processing of eigenvectors, pure CPU linear algebra
4. `FittedModel` for factor scores (optional)

F14 needs: EigenDecomposition(Correlation) + rotation algorithm. If F22 provides EigenDecomp, F14 adds ~100 lines of rotation code.

### Spectral Clustering (F20)

Spectral clustering = eigendecompose GRAPH LAPLACIAN, cluster in eigenvector space.

**Sharing with F22**:
1. Graph Laplacian L = D - W (degree matrix minus similarity matrix)
2. W = kernel similarity matrix (e.g., RBF kernel applied to DistancePairs)
3. EigenDecompose L → same algorithm as PCA, different input matrix
4. Cluster in eigenvector space using KMeans (F20 KMeans already implemented)

The only new thing: compute D - W from the similarity matrix. Then reuse EigenDecomposition.

### CCA (F33)

Canonical Correlation Analysis — generalized eigenvalue problem:
```
Σ_xy · Σ_yy⁻¹ · Σ_yx · a = λ · Σ_xx · a
```

Σ_xx, Σ_yy, Σ_xy are subblocks of the full GramMatrix (computed when F10 includes multiple response variables). The generalized eigenvalue solve is:
```
Cholesky(Σ_xx) → L  →  transform to standard eigen: L⁻¹ Σ_xy Σ_yy⁻¹ Σ_yx L⁻ᵀ
```

This is EigenDecomposition of a derived p×p matrix, using the same algorithm.

---

## t-SNE and UMAP: NOT derived from EigenDecomposition

t-SNE and UMAP are non-linear dimensionality reduction — they're NOT extractions from GramMatrix.

- **t-SNE**: compute pairwise distances (F01/DistancePairs), construct probability distributions P and Q, then gradient descent minimizing KL(P||Q). Kingdom C (iterative), depends on DistancePairs trunk, NOT GramMatrix/EigenDecomp trunk.
- **UMAP**: construct fuzzy simplicial set from k-NN graph, optimize layout. Kingdom C, depends on SortedPermutation (k-NN construction).

These require:
1. `DistancePairs` (from F01)
2. Gradient-based optimization (F05)
3. No EigenDecomposition

They belong to the DistancePairs trunk, not the GramMatrix trunk. Different Track.

---

## ISOMAP (geodesic PCA)

ISOMAP is in the bridge between both trunks:
1. Compute shortest paths on k-NN graph (Dijkstra's on `DistancePairs`): DistancePairs trunk
2. Classical MDS (multi-dimensional scaling) on geodesic distances = EigenDecompose centered distance matrix: GramMatrix/EigenDecomp trunk

ISOMAP is one of the few algorithms that spans both trunks.

---

## LDA (Linear Discriminant Analysis)

LDA needs:
- Between-class scatter matrix: S_B = Σ_k n_k (μ_k - μ)(μ_k - μ)ᵀ
- Within-class scatter matrix: S_W = Σ_k Σ_{x in k} (x - μ_k)(x - μ_k)ᵀ

S_W = sum of per-class covariance matrices × (n_k - 1) = RefCenteredStats per class!
S_B = function of group means = MomentStats(order=1, ByKey)!

Both are extractable from F06 infrastructure. Then solve generalized eigenvalue S_W⁻¹ S_B u = λ u.

LDA uses GramMatrix trunk for the solve but MomentStats trunk for S_B, S_W construction. Another algorithm that spans both trunks.

---

## Build Order

1. **Symmetric eigendecompose** — QR iteration or LAPACK `dsyevd` for p×p matrix
   - Start with pure Rust power iteration (top-k, simplest)
   - Add full symmetric eigendecomp for Factor Analysis needs
   - Store as `EigenDecomposition` in `intermediates.rs`

2. **PCA from GramMatrix** — divide by N-1, eigendecomp, return top-k components

3. **Verify against numpy**: `np.linalg.eigh(cov)` or R's `prcomp(x)$rotation`. Must match within 1e-8 (up to sign of eigenvectors — sign flip is valid).

4. **Note on sign ambiguity**: eigenvectors have arbitrary sign. Convention: largest absolute component is positive. This is what sklearn uses. Gold standard: `sklearn.decomposition.PCA`.

5. **PCA scores (projection)**: Y = X_centered · V[:, :k]. One matrix multiply.

6. **Cache EigenDecomposition in TamSession** with `IntermediateTag::EigenDecomposition { gram_id, matrix_type }`.

7. **F14 (Factor Analysis) Phase 1**: consume same EigenDecomposition with Varimax rotation (~150 lines).

---

## The Lab Notebook Claim

> PCA, Factor Analysis, Spectral Clustering, and CCA all share a single O(p³) computation: the eigendecomposition of a symmetric matrix derived from GramMatrix. Once GramMatrix is accumulated (O(N×p²)), the O(p³) eigendecomp is a negligible post-processing step. All four families are "free" in the sense that adding any one of them after the others costs only the last-mile arithmetic on the same EigenDecomposition.
