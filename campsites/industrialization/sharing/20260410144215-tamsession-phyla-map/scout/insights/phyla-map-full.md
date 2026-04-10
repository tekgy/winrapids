# TamSession Phyla Map — Full Audit
Written: 2026-04-10

## Current State (Actually Wired)

Only 6 intermediate types are currently wired into TamSession:

| Tag | Producer | Consumers | Value |
|-----|----------|-----------|-------|
| `DistanceMatrix` | clustering.rs, knn.rs, pipeline.rs | 4 | High (all clustering) |
| `MomentStats` | descriptive::moments_session | Underused | High potential |
| `SufficientStatistics` | hash_scatter.rs, train/linear.rs | 2 | Medium |
| `DataQualitySummary` | data_quality::from_session | ~18 validity predicates | High |
| `ClusterLabels` | pipeline.rs | 1 | Medium |
| `Sketch (HLL)` | sketches.rs | 1 | Low |

The `IntermediateTag` enum has 10 variants defined, but only 6 are used in production code.
The `Embedding`, `Centroids`, `TopKNeighbors`, `ManifoldDistanceMatrix`, `ManifoldMixtureDistance`
variants are defined but I found no producers for them in the current codebase.

## The Real Phyla (What Should Exist)

### Tier 1: High ROI — Multiple heavy consumers, O(n log n) or worse

**Phylum 1: SortedArray**
- Cost: O(n log n) — sort the input data
- Producers: any normality test, quantile, rank, KDE, KS test, order statistics
- Consumers: shapiro_wilk, dagostino_pearson, ks_test_normal, quantile (multiple q values),
  rank (Spearman, Kendall use it), inversion_count (takes sorted input), KDE
- Tag: `SortedArray { data_id: DataId }`
- Note: Sorting once and sharing is a big win when a pipeline runs shapiro_wilk + quantile + rank.

**Phylum 2: FFTOutput**
- Cost: O(n log n) — FFT computation
- Producers: periodogram, welch, spectrogram, stft
- Consumers: periodogram, welch, stft (all need the FFT), spectral entropy, spectral features
- Tag: `FFTOutput { data_id: DataId, n_fft: usize }`
- Note: signal_processing runs FFT independently in every spectral function; every fintek
  Family 6 leaf recomputes the FFT on the same bin.

**Phylum 3: CovarianceMatrix**
- Cost: O(n*d²) — covariance computation
- Producers: pca (implies covariance), factor_analysis, manova, lda
- Consumers: pca, factor_analysis, lda, cca (uses Cov(X)), manova, mahalanobis_distances,
  ridge regression (regularizes covariance), vif (via QR of design matrix)
- Tag: `CovarianceMatrix { data_id: DataId, ddof: usize }`
- Note: The fintek per-bin pipeline computes covariance in family13 (PCA), and it could be
  shared with correlation-based tests in family8.

**Phylum 4: PairwiseDistances**
- Cost: O(n²*d) — brute-force all-pairs
- Note: Already exists as `DistanceMatrix` in TamSession! This IS the tag.
  But it's only wired in clustering.rs and knn.rs. The following also need it:
  - `correlation_dimension` (complexity.rs) — computes pairwise L∞ distances inline
  - `largest_lyapunov` (complexity.rs) — same, L2 distances inline
  - `cluster_validation` (silhouette) — needs pairwise distances
  - `family15_manifold_topology` — delegates to `tambear::graph::pairwise_dists` correctly
  The fix: complexity.rs should call `session.get::<DistanceMatrix>` before recomputing.

### Tier 2: Significant ROI — Medium cost, 3-5 consumers

**Phylum 5: EigenDecomposition**
- Cost: O(d³) — eigenvalue computation of d×d matrix
- Tag: `EigenDecomposition { matrix_id: DataId }`
- Consumers: PCA (takes top-k eigenvectors), spectral clustering (graph Laplacian eigen),
  factor analysis (correlation matrix eigen), manova (Wilks' lambda via eigvals)
- Note: Computing eigen once when PCA runs and reusing for spectral clustering is a real win.

**Phylum 6: QRFactorization**
- Cost: O(n*d²) — QR of design matrix
- Tag: `QRFactorization { data_id: DataId }`
- Consumers: lstsq (OLS via QR), ols_normal_equations, regression diagnostics (leverage,
  Cook's D), VIF computation — all build the same X matrix and factor it.

**Phylum 7: DelayEmbedding**
- Cost: O(n*d) — Takens delay embedding
- Tag: `DelayEmbedding { data_id: DataId, dim: usize, tau: usize }`
- Note: `time_series::delay_embed` is already public. 
  `complexity::correlation_dimension` and `complexity::largest_lyapunov` both embed inline.
  `family24::ccm` embeds inline. `family15::delay_embed` wraps the public primitive.
  The first caller to embed should register; the rest should look up.

**Phylum 8: SymbolizedSeries**
- Cost: O(n log n) — sort + quantile assignment
- Tag: `SymbolizedSeries { data_id: DataId, n_symbols: usize }`
- Consumers: permutation_entropy, LZ complexity (binarize at median = 2 symbols),
  transfer entropy (quantize continuous), CCM (ordinal patterns), information_theory
- Note: This phylum doesn't exist at all yet. The symbolization primitive family
  (binarize_at_median, symbolize_quantile, symbolize_ordinal) needs to be extracted first,
  then TamSession sharing can be added.

### Tier 3: Moderate ROI — 2-3 consumers, medium cost

**Phylum 9: ACFVector**
- Cost: O(n*max_lag) — autocorrelation coefficients
- Tag: `ACFVector { data_id: DataId, max_lag: usize }`
- Consumers: ar_fit (uses r[0..p]), box_pierce, breusch_godfrey, PACF (builds from ACF),
  data_quality::lag1_autocorrelation, data_quality::acf_decay_exponent
- Note: time_series::acf is public. ar_fit recomputes ACF inline.

**Phylum 10: KernelDensityEstimate**
- Cost: O(n²) brute force, O(n log n) FFT variant — KDE computation
- Tag: `KDE { data_id: DataId, bandwidth: f64, kernel: KernelType }`
- Consumers: kde, kde_fft (same result if bandwidth matches), KDE-based tests

**Phylum 11: RunsTestResult**
- Cost: O(n) — runs computation
- Already separate but small — probably not worth caching unless used in bundles.

**Phylum 12: SortedRanks**
- Cost: O(n log n) — rank computation (sort + fractional rank)
- Tag: `Ranks { data_id: DataId }`
- Consumers: spearman (uses ranks of both columns), kendall (uses sorted ranks for
  inversion count), nonparametric tests that use ranks
- Observation: `rank()` is already public. Spearman + Kendall on the same data pair
  should share the rank computation.

**Phylum 13: LogReturnsSeries**
- Cost: O(n) — ln(p[i+1]/p[i])
- Tag: `LogReturns { data_id: DataId }`
- Consumers: volatility (GARCH), ARCH-LM test, autocorrelation of squared returns,
  DFA on returns, almost every fintek family that takes returns as input
- Note: The fintek pipeline computes log returns once per bin. This should register in
  TamSession so all families that need returns get them from cache.

### Tier 4: Low individual ROI, high aggregate

**Phylum 14: NormalityTestResult**
- Cost: O(n log n) — Shapiro-Wilk / D'Agostino
- Tag: `NormalityTest { data_id: DataId, method: NormalityMethod }`
- Note: The tbs_executor runs normality tests for EVERY auto-detect path. Same data,
  same test, multiple calls. Caching would eliminate 3-4 redundant normality tests per pipeline.

**Phylum 15: PrincipalComponents**
- Cost: O(n*d² + d³) — PCA
- Tag: `PCA { data_id: DataId, n_components: usize }`
- Consumers: spectral clustering (graph Laplacian PCA), UMAP initialization, t-SNE init

**Phylum 16: GramMatrix**
- Cost: O(n²) — X'X for normal equations
- Tag: `GramMatrix { data_id: DataId }`
- Consumers: OLS (builds gram, factorizes), any method needing X'X once

## The Critical Missing Infrastructure

Looking at this map, the biggest gap is not in the IntermediateTag variants — it's in the
**wiring**. The current producers and consumers are isolated islands:

- `complexity.rs` computes pairwise distances, eigendecompositions, delay embeddings —
  never through TamSession, always inline.
- `hypothesis.rs` computes the same normality tests multiple times in different code paths.
- `multivariate.rs` computes covariance matrices independently of `factor_analysis.rs`.

The IntermediateTag enum needs these variants added:
```rust
SortedArray { data_id: DataId },
FFTOutput { data_id: DataId, n_fft: usize },
CovarianceMatrix { data_id: DataId, ddof: u8 },
EigenDecomposition { matrix_id: DataId },
QRFactorization { data_id: DataId },
DelayEmbedding { data_id: DataId, dim: usize, tau: usize },
SymbolizedSeries { data_id: DataId, n_symbols: usize },
ACFVector { data_id: DataId, max_lag: usize },
Ranks { data_id: DataId },
LogReturns { data_id: DataId },
NormalityTest { data_id: DataId },
PrincipalComponents { data_id: DataId, n_components: usize },
```

Then producers need to register, and consumers need to check before computing.

## The Sharing Graph as Architecture Documentation

The phyla map is also a dependency graph for the codebase:
- If two functions share a phylum, they can cooperate via TamSession.
- If a function is the only consumer of a phylum, it should own that computation fully.
- If a phylum has 5+ consumers, it's a fundamental primitive that should be extracted.

The phyla aren't random groupings — they're the natural taxonomy of the math.
FFT, sort, covariance, eigendecomposition — these are the atoms of the mathematical universe.
Every method is built from them. TamSession is just the mechanism by which this structure
gets cached, not the structure itself.
