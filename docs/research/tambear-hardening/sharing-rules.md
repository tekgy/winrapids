# Sharing Compatibility Matrix

**Author**: math-researcher  
**Date**: 2026-04-08  
**Status**: Formal specification for auto-detection sharing decisions

---

## 1. IntermediateTag Registry

### Existing Tags

| Tag | Type stored | Key fields |
|-----|-------------|------------|
| `DistanceMatrix` | `DistanceMatrix` (n×n f64) | `metric: Metric`, `data_id` |
| `SufficientStatistics` | `SufficientStatistics` | `data_id`, `grouping_id` |
| `ClusterLabels` | `Vec<i32>` | `data_id` |
| `Centroids` | centroids (k×d) | `data_id`, `k` |
| `TopKNeighbors` | per-point neighbor lists | `k`, `metric`, `data_id` |
| `Embedding` | embedding vectors (n×dim) | `dim`, `data_id` |
| `ManifoldDistanceMatrix` | distance matrix | `manifold_name`, `data_id` |
| `ManifoldMixtureDistance` | combined distance matrix | `mix_id`, `data_id` |
| `MomentStats` | `MomentStats` {count,sum,min,max,m2,m3,m4} | `data_id` |
| `GroupedMomentStats` | `Vec<MomentStats>` | `data_id`, `groups_id` |

### Proposed Tags (Task #95, #98)

| Tag | Type stored | Key fields |
|-----|-------------|------------|
| `CorrelationMatrix` | `Mat` (p×p) | `data_id` |
| `InverseCorrelationMatrix` | `Mat` (p×p) | `data_id` |
| `OlsRegression` | residuals + hat diag + fitted | `data_id`, `response_id` |
| `AutocorrelationFunction` | `Vec<f64>` (lags 0..h) | `data_id`, `max_lag` |
| `CoxRiskSetSums` | S₀, S₁ at each event time | `data_id` |
| `GlobalRanks` | `Vec<f64>` (average-tie ranks) | `data_id` |
| `EigenDecomposition` | eigenvalues + eigenvectors | `data_id` |
| `QrFactorization` | Q, R matrices | `data_id` |

---

## 2. Compatibility Matrix: DistanceMatrix

### Producers

| Method | Module | Metric produced | Notes |
|--------|--------|----------------|-------|
| `dbscan_session` | clustering.rs | L2Sq | Pairwise distance, n² |
| `knn_session` | knn.rs | L2Sq | Same computation, different consumer |
| `TiledEngine::distance` | winrapids-tiled | L2Sq, Cosine, Dot | GPU-accelerated |
| `discover_clusters` | pipeline.rs | L2Sq (default) | Via TiledEngine |
| `ManifoldPipeline` | pipeline.rs | Manifold-specific | Poincaré, spherical, etc. |

### Consumers

| Method | Module | Metrics accepted | Sharing valid? | Conditions |
|--------|--------|-----------------|----------------|------------|
| DBSCAN | clustering.rs | L2Sq, L2 | **YES** | ε² if L2Sq, ε if L2 |
| KNN | knn.rs | Any | **YES** | Monotone metrics equivalent for ranking |
| Silhouette | clustering.rs | L2Sq, L2 | **YES** | Needs actual distances for averaging |
| Calinski-Harabasz | clustering.rs | L2Sq | **YES** | Uses within/between SS on distances |
| Davies-Bouldin | clustering.rs | L2Sq, L2 | **YES** | Uses centroid distances |
| Hopkins | clustering.rs | L2Sq, L2 | **YES** | NN distances |
| Moran's I | spatial.rs | L2 | **PARTIAL** | Spatial weights use distance threshold, need L2 not L2Sq |
| Kriging | spatial.rs | L2 | **NO** | Needs Euclidean for variogram, not squared |
| Spectral clustering | graph/spectral | Cosine, L2 | **YES** | Affinity = exp(-d²/σ²) |
| t-SNE | dim_reduction | L2Sq | **YES** | Uses squared distances in probability |

### CANNOT Share (even though it looks compatible)

| Consumer | Why not |
|----------|---------|
| Spearman correlation | Needs rank-transformed distances, not raw L2. Rank distances ≠ distances of ranks. |
| Robust PCA (MCD) | Needs Mahalanobis distance, not L2. The covariance matrix differs (MCD vs sample). |
| Manifold methods | Poincaré distance ≠ L2. Cannot substitute even on the same data. The tag `ManifoldDistanceMatrix` exists precisely for this. |
| Cosine-KNN from L2 | Cosine and L2 are NOT monotone-equivalent. cos_dist(a,b) ≠ f(L2(a,b)) for arbitrary a,b. Need separate computation. |

---

## 3. Compatibility Matrix: MomentStats

### Producers

| Method | Module | Notes |
|--------|--------|-------|
| `moments_ungrouped` | descriptive.rs | Single-pass Welford on raw data |
| `moments_session` | descriptive.rs | Session-aware, registers in TamSession |
| `DescriptiveEngine::moments` | descriptive.rs | GPU scatter-based |
| Hash scatter | hash_scatter.rs | GroupedMomentStats (per-key) |

### Consumers

| Method | Module | Sharing valid? | Conditions |
|--------|--------|----------------|------------|
| One-sample t-test | hypothesis.rs | **YES** | Uses mean, variance, count |
| Two-sample t-test | hypothesis.rs | **YES** | Two MomentStats |
| Paired t-test | hypothesis.rs | **YES** | MomentStats of differences |
| Welch's t-test | hypothesis.rs | **YES** | Same as two-sample |
| One-way ANOVA | hypothesis.rs | **YES** | GroupedMomentStats → F test |
| Welch's ANOVA | hypothesis.rs | **YES** | Uses group means and variances |
| Levene's test | hypothesis.rs | **INDIRECT** | Needs raw data to compute |x-median|, then ANOVA on z-scores |
| Pearson correlation | nonparametric.rs | **PARTIAL** | Needs two MomentStats + cross-product. Cross-product not in MomentStats. |
| Cohen's d | hypothesis.rs | **YES** | Mean difference / pooled SD |
| Shapiro-Wilk | nonparametric.rs | **NO** | Needs sorted raw data, not moments |
| KDE | nonparametric.rs | **NO** | Needs raw data |
| Bootstrap | nonparametric.rs | **NO** | Needs raw data for resampling |
| Skewness/Kurtosis | descriptive.rs | **YES** | Uses m3, m4 from MomentStats |
| Z-scoring | descriptive.rs | **YES** | (x - mean) / std, uses mean and var |

### CANNOT Share

| Consumer | Why not |
|----------|---------|
| Trimmed mean | Needs sorted raw data; trimming operates on order statistics, not moments |
| MAD (median absolute deviation) | Needs raw data for median, then absolute deviations |
| Quantiles/percentiles | Need sorted raw data |
| Rank-based tests (Mann-Whitney, Kruskal-Wallis) | Need ranks, not moments |
| Any bootstrap procedure | Needs raw data for resampling |

### Key Insight
MomentStats are the MSR for the **Gaussian world** — any statistic that depends only on mean, variance, skewness, kurtosis can be computed from MomentStats in O(1). Anything that depends on **order statistics** (ranks, quantiles, trimming) or **individual data points** (resampling, KDE) cannot.

---

## 4. Compatibility Matrix: CorrelationMatrix (proposed)

### Producers

| Method | Module | Notes |
|--------|--------|-------|
| `correlation_matrix` | multivariate.rs | Pearson R from raw data, O(np²) |
| PCA preprocessing | dim_reduction.rs | Computes R as a byproduct of centering + SVD |

### Consumers

| Method | Module | Sharing valid? | Conditions |
|--------|--------|----------------|------------|
| KMO | factor_analysis.rs | **YES** | Needs R and R⁻¹ |
| Bartlett's sphericity | factor_analysis.rs | **YES** | Needs eigenvalues of R (= det via log-sum) |
| VIF | multivariate.rs | **YES** | VIF_j = diag(R⁻¹)_j. If R⁻¹ already computed for KMO, reuse. |
| Factor analysis (PAF) | factor_analysis.rs | **YES** | Initial communalities from R |
| PCA | dim_reduction.rs | **YES** | Eigendecomposition of R (or covariance) |
| Partial correlation | (pending #80) | **YES** | partial_r_ij = -R⁻¹_ij / √(R⁻¹_ii R⁻¹_jj) |
| CCA | multivariate.rs | **YES** | Needs sub-blocks of R |
| Pearson r (pairwise) | nonparametric.rs | **YES** | R_ij is the answer |

### CANNOT Share

| Consumer | Why not |
|----------|---------|
| Spearman correlation matrix | Spearman uses ranks, not raw values. Pearson R on raw data ≠ Spearman ρ. Must rank-transform FIRST, then compute Pearson on ranks. |
| Kendall τ matrix | Completely different computation (concordant/discordant pairs). No relationship to Pearson R. |
| Polychoric/polyserial correlation | Assumes underlying bivariate normal latent variables. Different estimator. |
| Robust covariance (MCD, MVE) | Minimum covariance determinant uses a subset of points. Standard R uses all points. Different matrix. |
| Distance correlation | Uses centered distance matrices, not Pearson correlations. |

---

## 5. Compatibility Matrix: OlsRegression (proposed)

### Producers

| Method | Module | Components produced |
|--------|--------|-------------------|
| `linear_regression` | (various) | β̂, residuals ê, fitted ŷ |
| Cook's distance | hypothesis.rs | hat matrix diagonal h_ii, (X'X)⁻¹ |
| QR-based regression | linear_algebra.rs | Q, R factors |

### Consumers

| Method | Component needed | Sharing valid? | Conditions |
|--------|-----------------|----------------|------------|
| Breusch-Pagan | residuals ê, X | **YES** | Same regression, auxiliary regression on ê² |
| Durbin-Watson | residuals ê | **YES** | d = Σ(ê_t - ê_{t-1})² / Σ ê² |
| Cook's distance | hat diag h_ii, residuals ê, MSE | **YES** | D_i = ê²_i h_ii / (p·MSE·(1-h_ii)²) |
| VIF | (X'X)⁻¹ diagonal | **MAYBE** | Only if X is standardized. If not, need R⁻¹ separately. |
| Leverage plot | hat diagonal h_ii | **YES** | Direct reuse |
| Studentized residuals | ê, h_ii, MSE | **YES** | r_i = ê_i / (MSE·(1-h_ii))^½ |
| DFBETAS | (X'X)⁻¹, ê, h_ii | **YES** | Influence on each coefficient |
| Ramsey RESET | fitted ŷ | **YES** | Adds ŷ², ŷ³ to regression |
| Added variable plots | residuals from partial regressions | **NO** | Different regressions |

### CANNOT Share

| Consumer | Why not |
|----------|---------|
| Ridge regression (X'X + λI)⁻¹ | Different matrix — regularized. Cannot reuse OLS (X'X)⁻¹. |
| Weighted LS | Different objective: minimize Σ w_i ê²_i. Different residuals and hat matrix. |
| Robust regression (Huber/LMS) | Different estimator entirely. Residuals and leverage are not comparable. |
| Logistic regression | Different link function. Residuals are deviance or Pearson, not OLS. |

### Critical Sharing Rule
**Breusch-Pagan and Cook's distance compute DIFFERENT auxiliary regressions.** BP regresses ê² on X. Cook's uses the original regression's hat matrix. They share the original residuals ê and design matrix X, but NOT each other's internal computations.

---

## 6. Compatibility Matrix: AutocorrelationFunction (proposed)

### Producers

| Method | Module | Notes |
|--------|--------|-------|
| `acf` | time_series.rs | Sample ACF for lags 0..max_lag |

### Consumers

| Method | Module | Sharing valid? | Conditions |
|--------|--------|----------------|------------|
| Ljung-Box | time_series.rs | **YES** | Q = n(n+2) Σ ρ̂²_k/(n-k). Direct reuse of ACF. |
| PACF (Levinson-Durbin) | time_series.rs | **YES** | PACF is computed FROM ACF coefficients. Currently recomputes. |
| AR order selection | time_series.rs | **YES** | Uses ACF/PACF patterns (cutoff vs tailing) |
| Spectral density (via ACF) | signal_processing.rs | **YES** | PSD = FFT(ACF) (Wiener-Khinchin). |

### CANNOT Share

| Consumer | Why not |
|----------|---------|
| ACF of residuals (post-ARIMA) | Different series! ACF of raw data ≠ ACF of residuals. Must recompute on the residual series. |
| Cross-correlation (CCF) | Different computation: cross-covariance between TWO series, not auto-covariance of one. |
| Partial ACF (from raw data) | Actually CAN share — PACF is derived from ACF via Levinson-Durbin. But current implementation recomputes from raw data. This is a sharing opportunity. |

---

## 7. Compatibility Matrix: GlobalRanks (proposed)

### Producers

| Method | Module | Notes |
|--------|--------|-------|
| `rank` | nonparametric.rs | Average-tie ranks, NaN-safe |
| Kruskal-Wallis (internal) | nonparametric.rs | Computes global ranks, then group rank sums |

### Consumers

| Method | Module | Sharing valid? | Conditions |
|--------|--------|----------------|------------|
| Dunn's test | nonparametric.rs | **YES** | Uses same global ranks for pairwise comparisons |
| Spearman ρ | nonparametric.rs | **YES** | Spearman = Pearson on ranks. Need ranks of both x and y. |
| Friedman test | (pending) | **YES** | Within-block ranking (different from global, but same function) |
| Percentile computation | descriptive.rs | **PARTIAL** | Ranks give order, but percentiles need sorted values |

### CANNOT Share

| Consumer | Why not |
|----------|---------|
| Mann-Whitney U | U uses sum of ranks in one group from the COMBINED ranking. If groups change, ranks change. The ranks are data-dependent and group-dependent simultaneously. |
| Kendall τ | Kendall uses concordant/discordant PAIR comparisons, not ranks. |

### Key Insight
Ranks are shareable when the SAME data is ranked the SAME way. Adding or removing observations changes ALL ranks (because rank is relative). So ranks can only be shared within a single analysis, not across analyses on different subsets.

---

## 8. Compatibility Matrix: CoxRiskSetSums (proposed)

### Producers

| Method | Module | Notes |
|--------|--------|-------|
| `cox_ph` (gradient computation) | survival.rs | S₀(t_i), S₁(t_i), S₂(t_i) at each event time |

### Consumers

| Method | Module | Sharing valid? | Conditions |
|--------|--------|----------------|------------|
| Schoenfeld residuals | survival.rs | **YES** | r_ij = x_ij - S₁_j(t_i)/S₀(t_i). This IS the gradient contribution. |
| Martingale residuals | (pending) | **YES** | M_i = δ_i - Ĥ₀(t_i)·exp(x_i'β̂). Needs cumulative baseline hazard from S₀. |
| Deviance residuals | (pending) | **YES** | d_i = sign(M_i)·√(-2[M_i + δ_i·ln(δ_i - M_i)]). Derived from martingale. |
| Concordance index | (pending) | **PARTIAL** | Needs risk scores exp(x'β) at each time, but not S₀/S₁ directly. |

### CANNOT Share

| Consumer | Why not |
|----------|---------|
| Stratified Cox PH | Different risk sets per stratum. S₀, S₁ are stratum-specific. |
| Time-varying coefficients | β(t) varies, so the risk set sums have different weighting at each time point. |
| AFT model | Completely different parameterization. No risk sets. |

---

## 9. Compatibility Matrix: EigenDecomposition (proposed)

### Producers

| Method | Module | Notes |
|--------|--------|-------|
| `sym_eigen` | linear_algebra.rs | Eigenvalues + eigenvectors of symmetric matrix |
| PCA (via SVD) | dim_reduction.rs | Singular values ≈ √eigenvalues of X'X/n |
| Spectral clustering | graph.rs / spectral.rs | Eigenvalues of graph Laplacian |
| Arnoldi iteration | spectral_gap.rs | Top-k eigenvalues of large sparse matrix |

### Consumers

| Method | Module | Sharing valid? | Conditions |
|--------|--------|----------------|------------|
| Bartlett's sphericity | factor_analysis.rs | **YES** | Uses eigenvalues of R to compute ln(det(R)) = Σ ln(λ_i) |
| Kaiser criterion | factor_analysis.rs | **YES** | Count eigenvalues > 1 |
| Scree test | factor_analysis.rs | **YES** | Plot eigenvalues (needs eigenvalues) |
| Parallel analysis | factor_analysis.rs | **PARTIAL** | Needs eigenvalues of R, then compares to random. Sharing the real eigenvalues saves one decomposition. |
| Spectral gap | spectral_gap.rs | **YES** | λ₁ - λ₂ = community structure indicator |
| Condition number | linear_algebra.rs | **YES** | κ = λ_max / λ_min |

### CANNOT Share

| Consumer | Why not |
|----------|---------|
| PCA on covariance vs correlation | Eigendecomposition of covariance matrix ≠ correlation matrix. Different scaling. If you want both, compute both. |
| Sparse eigenvalues (Arnoldi) from dense | Arnoldi is for LARGE sparse matrices. Dense eigendecomposition is for small matrices. Different algorithms, different contexts. But if n < 1000, dense is fine for both. |
| Eigenvalues of different matrices | Obvious but worth stating: eigenvalues of R (correlation) ≠ eigenvalues of H (MANOVA hypothesis matrix) ≠ eigenvalues of L (graph Laplacian). The tag must include which matrix. |

---

## 10. Cross-Pipeline Sharing Rules

These rules govern when a PIPELINE (sequence of TBS steps) can share intermediates across steps.

### Rule 1: Same Data Invariant
Sharing is only valid when both producer and consumer operate on the **same data** (same `data_id`). If the data is transformed between steps (e.g., log-transform, standardize, filter), the intermediate is invalid.

**Exception**: MomentStats can survive additive shifts. If the data is shifted by a constant c, the new mean = old mean + c, and m2/m3/m4 are unchanged. But MomentStats currently don't track this.

### Rule 2: Metric Compatibility
Distance matrices are parameterized by metric. L2Sq and L2 are monotone-equivalent for nearest-neighbor queries (ranking preserved) but NOT for threshold-based operations (DBSCAN ε). The consumer must know which metric was used.

### Rule 3: Group Consistency
GroupedMomentStats are valid only for the same grouping. If groups change (merge, split, relabel), the stats are invalid. The `groups_id` field in the tag enforces this.

### Rule 4: Temporal Ordering
Some methods produce intermediates that are only valid BEFORE other methods run:
- ARCH-LM must run BEFORE GARCH (tests raw residuals, not GARCH-standardized)
- Shapiro-Wilk should run BEFORE parametric tests (normality is a prerequisite)
- ADF/KPSS should run BEFORE ARIMA fitting (stationarity determines differencing)

### Rule 5: Assumption Dependency
If a diagnostic test REJECTS an assumption, the intermediate from the method that assumed it is INVALID for consumers that need the assumption:
- If Shapiro-Wilk rejects normality → MomentStats-based t-test p-values are unreliable (but the MomentStats themselves are fine — it's the inference that's affected, not the computation)
- If Breusch-Pagan rejects homoscedasticity → OLS standard errors are wrong (but residuals and coefficients are still valid)

**Key distinction**: Sharing is about COMPUTATION, not INFERENCE. A shared intermediate is computationally valid even when the statistical inference built on it is questionable. The V columns (confidence metadata) track this distinction.

### Rule 6: No Cross-Subset Sharing
Ranks, distances, and moment statistics computed on a SUBSET of data cannot be shared with methods operating on the full dataset (or a different subset). This is because:
- Ranks are relative to the sample
- Distances may involve different point sets
- Moments of a subset ≠ moments of the whole

**Exception**: SufficientStatistics are DESIGNED to merge. Two SufficientStatistics from disjoint subsets can be combined algebraically (Welford merge). This is the whole point of the accumulate architecture.

---

## 11. Sharing Opportunity Summary

| Opportunity | Estimated savings | Difficulty |
|-------------|------------------|------------|
| ACF → Ljung-Box (avoid recompute) | O(n·h) per call | Low: add `ljung_box_from_acf` |
| Correlation matrix → KMO + Bartlett + VIF | O(np²) per matrix | Medium: new IntermediateTag |
| OLS regression → BP + DW + Cook's | O(np) per regression | Medium: new IntermediateTag |
| Cox S₀/S₁ → Schoenfeld | O(n·d) per risk set pass | Medium: refactor cox_ph |
| Kruskal-Wallis ranks → Dunn's | O(n log n) per ranking | Low: pass ranks explicitly |
| PCA eigenvalues → Bartlett + Kaiser | O(p³) per decomposition | Low: pass eigenvalues |
| PACF from ACF (Levinson-Durbin) | O(n·h) → O(h²) | Low: already uses ACF internally |

---

_This document is the formal specification for Task #23 (Formalize sharing rules for all intermediate types)._
_Updated as new IntermediateTag variants are added._
