# Are Our Intermediates the Right Decomposition?

*2026-04-10 — Aristotle, first principles*

## The Question

The sharing infrastructure defines these intermediates: DistanceMatrix, SufficientStatistics (sum, m2, count), FFT, CovMatrix, SortedOrder, ACF. Are these the RIGHT decomposition of shared state? Or are they artifacts of the linear algebra / frequentist statistics tradition?

## The Aristotelian Approach

An intermediate is correct if it satisfies THREE conditions:
1. **Sufficient**: downstream consumers can derive everything they need from it
2. **Minimal**: no proper subset carries the same information
3. **Natural**: it corresponds to a real mathematical object, not an implementation convenience

## Current Intermediates, Examined

### DistanceMatrix — CORRECT but INCOMPLETE

- **Sufficient**: Yes for algorithms that need all pairwise distances (DBSCAN, KNN, silhouette)
- **Minimal**: Yes — you can't share less than the full matrix if someone needs all pairs
- **Natural**: Yes — a metric space structure on the point set

**But**: The distance matrix is O(n^2) space. For n > 50k, it doesn't fit in GPU memory. The natural intermediate for LARGE datasets is not the distance matrix but the **distance FUNCTION** (metric + data, lazily evaluated). For KNN you only need each point's k nearest neighbors, not all n^2 pairs. The framework pre-materializes what should be lazily computed.

**The deeper issue**: The 3-field sufficient stats {sq_norm_x, sq_norm_y, dot_prod} show that the REAL intermediate is not the distance matrix but the **Gram matrix** (X @ X^T for dot products) or the **three scalar fields**. From these, ALL inner-product distances are derivable. The DistanceMatrix is a DERIVED intermediate, not a primitive one.

**First-principles answer**: The true primitive intermediate for pairwise relationships is:
- **Small n**: Full Gram matrix (n x n dot products) — from which all inner-product distances follow
- **Large n**: Approximate nearest neighbor structure (VP-tree, locality-sensitive hash) — from which KNN and DBSCAN follow without O(n^2)
- **Both**: The 3 sufficient stat fields as the MSR

### SufficientStatistics (sum, m2, count) — CORRECT and COMPLETE

- **Sufficient**: Yes for mean, variance, std, z-score, normalization, t-test, ANOVA, F-test
- **Minimal**: Yes — removing any of the three fields loses information
- **Natural**: Yes — these are the sufficient statistics for a Gaussian model

**This is the gold standard of what an intermediate should look like.** Three scalars per group. Additive merge (Welford). From them, an entire family of computations is free.

**But**: only sufficient for the first two moments. Skewness needs M3. Kurtosis needs M4. The generalization is the k-th order Pebay accumulator. The question: should we store (sum, m2) or (n, m1, m2, m3, m4)? The answer depends on downstream demand. If skewness and kurtosis are commonly needed, the 5-field accumulator is the right MSR.

### FFT — CORRECT but MISNAMED

The FFT output is not "an FFT." It's a **frequency-domain representation** of the signal. Multiple downstream consumers use it: power spectral density, cross-spectral density, cepstrum, spectral coherence, Hilbert transform, analytic signal.

The natural name is not "FFT intermediate" but "spectral representation" — it's a change of basis from time to frequency. The sharing tag should reflect the mathematical object (spectrum of signal X), not the algorithm that produced it (FFT).

**Why this matters**: If someone computes the DFT via Goertzel algorithm (for specific frequencies) or via Chirp Z-transform, they produce a compatible intermediate. The sharing contract should be about the RESULT, not the METHOD.

### CovMatrix — CORRECT and RICH

The covariance matrix is the hub of multivariate statistics. From it: PCA, factor analysis, CCA, Mahalanobis distance, LDA, regression coefficients, partial correlations, test statistics.

**First-principles check**: The covariance matrix IS X'X/n (centered). It IS the second-moment sufficient statistic for a multivariate Gaussian. It IS the natural parameter space for the multivariate normal family. This intermediate is not an artifact — it's mathematically fundamental.

**But**: The full covariance matrix is O(d^2). For d > 10k features, it doesn't fit. The natural intermediate for high-dimensional data is a FACTORED representation: randomized SVD factors, sketch matrix, or random projection. The framework should support covariance-as-factors alongside covariance-as-matrix.

## What's MISSING from the Intermediate Catalog

### 1. Sorted Order (partially present)

Sort order is needed for quantiles, ranks, order statistics, nonparametric tests. The intermediate is the permutation, not the sorted values.

**First-principles**: The argsort permutation IS the natural intermediate. From it: any quantile, any rank, any order statistic. This is correct.

### 2. Group Index / Partition Structure

The GroupIndex (rows_by_group, group_offsets) is itself a sharable intermediate. Multiple algorithms that group by the same column can share the partition computation.

**Currently**: GroupIndex is built inside specific operations. It should be a first-class shareable intermediate.

### 3. Graph/Neighborhood Structure

KNN graph, epsilon-neighborhood graph, Delaunay triangulation — these are intermediates for graph-based methods (spectral clustering, manifold learning, graph neural networks, UMAP, t-SNE).

**Missing**: No graph intermediate type exists. When UMAP needs the KNN graph that DBSCAN already computed, there's no sharing path.

### 4. Basis Function Evaluations

For spline/polynomial/wavelet regression, the basis matrix B (each column = a basis function evaluated at all data points) is expensive to compute and shared across multiple model fits with different targets.

**Missing**: No BasisMatrix intermediate type.

### 5. Kernel Matrix (distinct from distance matrix)

For kernel methods (kernel PCA, kernel SVM, Gaussian process), the kernel matrix K[i,j] = k(x_i, x_j) is shared. The kernel matrix is NOT the distance matrix — exp(-d^2/2sigma^2) is a transform of the distance matrix, and different kernel widths produce different matrices.

**Missing**: KernelMatrix(kernel_type, params, data_id) as a shareable intermediate.

## The Deeper Question

Is the intermediate catalog a FLAT list or a DEPENDENCY GRAPH?

Currently: flat. DistanceMatrix, SufficientStatistics, FFT, CovMatrix are independent types.

But: CovMatrix = SufficientStatistics applied to all pairs of columns. DistanceMatrix = derived from CovMatrix (via Mahalanobis) or from Gram matrix. FFT = basis function evaluation (Fourier basis). They are connected.

**The dependency graph of intermediates:**
```
Raw data
  ├─ SortedOrder (argsort)
  │    └─ Quantiles, Ranks, Order statistics
  ├─ GroupIndex (partition)
  │    └─ Per-group SufficientStatistics
  │         └─ Moments, Tests, Normalization
  ├─ GramMatrix (X @ X^T or X^T @ X)
  │    ├─ CovMatrix (centered, normalized)
  │    │    ├─ PCA, FA, CCA, LDA
  │    │    └─ Mahalanobis → DistanceMatrix
  │    └─ DistanceMatrix (from Gram + norms)
  │         ├─ KNN graph
  │         │    └─ UMAP, t-SNE, spectral clustering
  │         └─ DBSCAN, silhouette, outlier detection
  ├─ SpectralRepresentation (DFT/DCT)
  │    └─ PSD, coherence, cepstrum, analytic signal
  └─ KernelMatrix (from DistanceMatrix + kernel function)
       └─ Kernel PCA, GP, SVM
```

The first-principles answer: the intermediate catalog should be organized as this dependency graph, with each node knowing its parents. When a consumer requests a DistanceMatrix but only a GramMatrix exists, the system can DERIVE the distance matrix from the Gram matrix without re-scanning the data.

## Recommendation

The current intermediates are mostly correct. The main gaps:
1. Rename FFT → SpectralRepresentation (tag by content, not algorithm)
2. Add GramMatrix as the primitive below CovMatrix and DistanceMatrix
3. Add KNNGraph for graph-based methods
4. Add KernelMatrix for kernel methods
5. Organize intermediates as a dependency graph so derived intermediates can be computed from primitives
6. Support factored representations (sketched covariance, approximate KNN) for high-dimensional data
