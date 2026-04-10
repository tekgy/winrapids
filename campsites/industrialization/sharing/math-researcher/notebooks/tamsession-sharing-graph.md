# TamSession Sharing Graph — Intermediate Phyla Map

## What TamSession Does

TamSession caches expensive intermediates with content-addressed tags. Two methods requesting
"the covariance matrix" share it IFF their requirements are compatible (same data, same ddof,
same assumptions about centering).

## The Phyla

Each phylum is a family of intermediates that share computation. The phylum name is the
IntermediateTag prefix. Compatibility is determined by the tag's metadata fields.

---

### Phylum 1: MomentStats

**Tag**: `moments(data_hash, ddof)`

**Computed by**: `moments_ungrouped`, `moments_session`

**Fields cached**: n, mean, variance, skewness, kurtosis, min, max, sum, sum_of_squares

**Consumers** (methods that pull from this):
- t-tests (all): need mean, variance, n
- ANOVA: need per-group moments
- effect sizes (Cohen's d, Hedges' g, Glass' delta): need mean, variance, n
- Pearson r: need mean, variance for normalization
- Box-Cox: need mean for transform
- Forecast metrics: need mean for centering
- Control charts: need mean, std for limits
- Shapiro-Wilk: needs mean for centering

**Compatibility**: Same data slice + same ddof. Most methods use ddof=1 (sample).

---

### Phylum 2: CovarianceMatrix

**Tag**: `covariance_matrix(data_hash, ddof, centered)`

**Computed by**: `covariance_matrix`, `col_means` + outer product

**Fields cached**: p×p covariance matrix, column means

**Consumers**:
- PCA: eigendecomposition of covariance
- LDA: within-class and between-class covariance
- CCA: cross-covariance matrices
- Mahalanobis distance: inv(cov) × centered data
- Ridge/Lasso: X'X is proportional to covariance
- VIF: diagonal of inv(correlation)
- Hotelling T²: needs covariance + mean
- MANOVA: within-group covariance
- Factor analysis: covariance or correlation
- Mixed effects: variance components
- GLS: error covariance
- Mardia normality: needs covariance for Mahalanobis
- MCD robust covariance: alternative computation
- Shrinkage (Ledoit-Wolf): regularized version

**Compatibility rules**:
- ddof must match
- `centered` flag: some methods center data themselves
- MCD covariance is NOT compatible with sample covariance (different assumptions)
- Shrinkage covariance is NOT compatible with sample covariance

---

### Phylum 3: Eigendecomposition

**Tag**: `eigen(matrix_hash, symmetric, sorted)`

**Computed by**: `sym_eigen` (symmetric), `general_eigen` (non-symmetric)

**Fields cached**: eigenvalues (Vec<f64>), eigenvectors (Mat)

**Consumers**:
- PCA: top-k eigenvectors of covariance
- Spectral clustering: bottom-k eigenvectors of Laplacian
- Effective rank: Shannon entropy of normalized eigenvalues
- Condition number: max/min eigenvalue ratio
- Definiteness test: sign of eigenvalues
- Matrix functions (exp, log, sqrt): via eigendecomposition
- Marchenko-Pastur: eigenvalue classification
- Principal angles: between subspaces

**Compatibility rules**:
- Symmetric eigendecomposition is NOT compatible with non-symmetric
- Sorted vs unsorted: trivially convertible, use sorted always
- Full vs truncated: truncated is a subset of full (compatible upward)

---

### Phylum 4: SVD

**Tag**: `svd(matrix_hash, full, economy)`

**Computed by**: `svd`

**Fields cached**: U, S (singular values), Vt

**Consumers**:
- Pseudoinverse: V Σ⁻¹ Uᵀ
- Least squares: via SVD
- Rank: count singular values above tolerance
- Condition number: max/min singular value
- Effective rank: entropy of normalized sv
- Low-rank approximation: truncated SVD
- Polar decomposition: U × Vᵀ
- Nuclear norm: Σ σᵢ
- PCA (alternative): SVD of centered data matrix
- Total least squares: uses right singular vectors

**Compatibility**: Full SVD is always compatible; economy is compatible for consumers that only need the slim matrices.

---

### Phylum 5: Rank (ordinal)

**Tag**: `rank(data_hash, tie_method)`

**Computed by**: `rank` in nonparametric

**Fields cached**: rank vector

**Consumers**:
- Spearman: Pearson on ranks
- Kendall: inversion count on ranks
- Mann-Whitney: rank sums
- Kruskal-Wallis: group rank sums
- Friedman: ranks within blocks
- Rank-based cost (changepoint)
- Rank von Neumann ratio
- Wilcoxon signed rank
- Dunn test

**Compatibility**: Tie-breaking method must match (average, min, max, ordinal, dense).
Most statistical tests use average ranks.

---

### Phylum 6: SortedData

**Tag**: `sorted(data_hash)`

**Computed by**: sort

**Fields cached**: sorted array, sort permutation (indices)

**Consumers**:
- Quantile/Median/IQR: direct lookup
- Trimmed/Winsorized mean: index-based trim
- Order statistics: direct access
- KS test: empirical CDF from sorted data
- Shapiro-Wilk: needs sorted data
- Box-counting: sorted for range queries
- Wasserstein 1D: sorted CDF difference
- Hill estimator: upper tail from sorted data

**Compatibility**: Always compatible (sort is deterministic for a given data slice).

---

### Phylum 7: DistanceMatrix

**Tag**: `distance(data_hash, metric, params)`

**Computed by**: pairwise distance computation

**Fields cached**: n×n distance matrix

**Consumers**:
- DBSCAN: neighborhood queries
- HDBSCAN: mutual reachability from kNN distances
- Hierarchical clustering: all linkages
- K-medoids: medoid assignment
- Silhouette score: intra/inter distances
- RQA: recurrence matrix (threshold on distances)
- Correlation dimension: count within radius
- Distance correlation: double-centered distances
- MDS: from distance matrix
- Kernel matrix: k(x,y) = f(d(x,y))
- OPTICS: reachability distances

**Compatibility rules**:
- Metric must match exactly (Euclidean ≠ Manhattan ≠ Mahalanobis)
- For Mahalanobis: the covariance matrix used must match
- Phase-space distances (from delay embedding) are NOT compatible with raw data distances
- Normalized distances are NOT compatible with unnormalized

---

### Phylum 8: FFT

**Tag**: `fft(data_hash, n_fft, window)`

**Computed by**: FFT implementation

**Fields cached**: complex spectrum, power spectral density, frequencies

**Consumers**:
- Spectral analysis (all): PSD, spectral features
- Welch method: averaged PSD from overlapping segments
- Cepstrum: IFFT(log(|FFT|²))
- Cross-spectrum: FFT(x) × conj(FFT(y))
- Coherence: cross-spectrum / (auto-spectra)
- Convolution: pointwise multiply in frequency domain
- Circulant solve: diagonalization via FFT
- KDE-FFT: fast kernel density estimation

**Compatibility rules**:
- Window function must match (rectangle ≠ Hann ≠ Hamming)
- n_fft must match (zero-padding affects resolution)
- Welch segments are NOT compatible with single-window FFT

---

### Phylum 9: ACF (Autocorrelation)

**Tag**: `acf(data_hash, max_lag)`

**Computed by**: `acf`

**Fields cached**: autocorrelation values r(0), r(1), ..., r(max_lag)

**Consumers**:
- PACF: Levinson-Durbin on ACF
- AR fit: Yule-Walker via ACF
- Ljung-Box: sum of squared ACF
- Box-Pierce: sum of squared ACF (simpler version)
- Durbin-Watson: related to r(1)
- Newey-West: long-run variance from ACF
- KPSS: uses partial sum of ACF
- Hurst exponent: ACF decay analysis
- DFA: related to ACF structure
- ESS (effective sample size): from ACF

**Compatibility**: max_lag must be ≥ required. Larger max_lag is compatible
(consumer takes what it needs).

---

### Phylum 10: DelayEmbedding

**Tag**: `delay_embed(data_hash, dim, tau)`

**Computed by**: `delay_embed`

**Fields cached**: embedded matrix (n-m×tau+1) × m

**Consumers**:
- Sample entropy: template matching on embedded vectors
- Approximate entropy: same
- Fuzzy entropy: same
- Correlation dimension: distances in embedded space
- Lyapunov exponents: neighbor divergence in embedded space
- RQA: recurrence matrix from embedded vectors
- FNN: testing embedding quality
- CCM: shadow manifold reconstruction

**Compatibility rules**:
- (dim, tau) must match exactly — this is critical
- Different (dim, tau) produce DIFFERENT embeddings, never compatible
- A method requesting (m=2, tau=1) CANNOT use a cached (m=3, tau=1)
  even though the m=3 embedding "contains" the m=2 information,
  because the distance structure changes with dimension

---

### Phylum 11: KernelMatrix

**Tag**: `kernel(data_hash, kernel_type, params)`

**Computed by**: kernel computation (Gaussian RBF, polynomial, etc.)

**Fields cached**: n×n kernel matrix K

**Consumers**:
- Kernel PCA: eigendecomposition of centered K
- Kernel ridge regression: (K + λI)⁻¹y
- SVM: quadratic programming on K
- HSIC: trace(KHLH)
- MMD: block structure of K
- Spectral clustering: similarity from K
- GP regression: K + σ²I

**Compatibility rules**:
- Kernel type AND parameters must match
- Gaussian(σ=1) ≠ Gaussian(σ=2)
- Centered kernel is NOT compatible with uncentered

---

## Cross-Phylum Dependencies

```
MomentStats ───→ CovarianceMatrix (uses means for centering)
                      │
                      ├──→ Eigendecomposition (PCA path)
                      │         │
                      │         └──→ Matrix functions
                      │
                      ├──→ SVD (alternative PCA path)
                      │
                      └──→ DistanceMatrix (Mahalanobis needs inv(cov))

SortedData ────→ Rank (rank is computed from sorted data + position)

ACF ───────────→ AR coefficients (Levinson-Durbin)
                      │
                      └──→ PSD (AR spectral density)

DelayEmbedding ─→ DistanceMatrix (in embedded space)
                      │
                      └──→ RecurrenceMatrix (threshold on distances)

FFT ───────────→ ACF (Wiener-Khinchin: ACF = IFFT(|FFT|²))
```

## The Compatibility Invariant

**The sharing contract**: if the cached intermediate was computed under assumptions
that don't match the consumer's method, it is NOT reusable. Sharing only happens
when the upstream intermediate is *provably correct for the downstream method*.

Key violation patterns to watch for:
1. Method A uses Euclidean distance; Method B needs Mahalanobis → NOT shareable
2. Method A computed covariance with ddof=0; Method B needs ddof=1 → NOT shareable
3. Method A embedded with (m=3, tau=2); Method B needs (m=2, tau=1) → NOT shareable
4. Method A used Hann window FFT; Method B needs rectangular → NOT shareable
5. Method A computed robust covariance (MCD); Method B expects sample covariance → NOT shareable

The IntermediateTag carries enough metadata to make these checks explicit.
