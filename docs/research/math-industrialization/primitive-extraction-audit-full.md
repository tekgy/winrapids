# Full Primitive Extraction Audit

*2026-04-10 — full scan of every .rs file in crates/tambear/src/ and crates/tambear-fintek/src/*

The prior audit (2026-04-09) found ~20 violations. This audit found **102** confirmed violations,
grouped into four categories. Every finding includes file, line number (approximate), what the math
computes, and which other consumers would benefit.

---

## Category 1: Duplicated across modules (HIGHEST PRIORITY)

Same math written independently in multiple places. Every instance is a separate bug surface.

| # | Primitive | Locations | Canonical module |
|---|-----------|-----------|-----------------|
| 1 | `sigmoid(z) = 1/(1+exp(-z))` | `irt.rs:30` (delegates to special_functions), `train/logistic.rs:59` (delegates to neural), `neural.rs` (origin) — **three delegation chains** instead of one canonical call | `special_functions.rs` (already has `logistic`) |
| 2 | `ols_slope(x, y)` — simple 1D OLS | `complexity.rs:253` (delegates to linear_algebra), `data_quality.rs:trend_r2` (inline), `family20_visibility_graphs.rs:power_law_exponent` (inline), `family20_visibility_graphs.rs:compute_nvg_stats` (inline), `family12_causality_info.rs:ols_rss` (partial), `hypothesis.rs:ols_simple` (private fn), `hypothesis.rs:ols_two_predictor` (private fn) | `linear_algebra.rs` (already has `ols_slope` and `simple_linear_regression`) |
| 3 | `rolling_variance_prefix(returns, window)` — O(n) rolling variance via prefix sums | `family9_volatility.rs:243` (private fn), `family17_market_microstructure.rs:302` (private fn — **identical implementation**) | `time_series.rs` or new `rolling.rs` |
| 4 | `pearson_corr(x, y)` — standard Pearson correlation | `family24_manifold.rs:144` (private fn), `family8_correlation.rs` (inline variants), `data_quality.rs:lag1_autocorrelation` (inline specialized), `family11_tick_microstructure.rs:lag1_autocorr` (private fn, specialized variant), `family17_market_microstructure.rs` (inline) | `nonparametric.rs` or `descriptive.rs` (already has Pearson; call it) |
| 5 | `lag1_autocorr(x)` — lag-1 Pearson autocorrelation | `data_quality.rs:lag1_autocorrelation` (public), `family11_tick_microstructure.rs:lag1_autocorr` (private), `time_series.rs:acf` (general ACF, but lag-1 is the common case) | `data_quality.rs` already has the canonical one — fintek should call it |
| 6 | `erfc_approx(x)` — Abramowitz-Stegun polynomial approximation | `family12_causality_info.rs:49` (private fn), `family17_market_microstructure.rs:libm_erf` (private fn — different constants, same idea) | `special_functions.rs` already has `erfc` with higher precision — both should call it |
| 7 | `rfft_naive(x)` / `fft_full(x)` — O(n²) DFT | `family12_causality_info.rs:rfft_naive:16` (half-spectrum), `family7_wavelets.rs:fft_full:41` (full complex DFT) — same algorithm, different return types | `signal_processing.rs` (already has FFT infrastructure) |
| 8 | `median_from_sorted(sorted, n)` — inline median extraction from sorted slice | `data_quality.rs:longest_gap_ratio:197`, `data_quality.rs:jump_ratio_proxy:300`, `data_quality_catalog.rs:mad_raw:259`, `data_quality_catalog.rs:skewness_bowley:382`, `data_quality_catalog.rs:count_outliers_mad:184`, `data_quality_catalog.rs:count_outliers_iqr:148` — **6 independent copies** of `if n%2==0 { (sorted[n/2-1]+sorted[n/2])/2 } else { sorted[n/2] }` | `descriptive.rs` — a public `median_from_sorted` |
| 9 | `log_returns_from_prices(prices)` — `ln(p[i+1]/p[i])` | `family11_tick_microstructure.rs:61` (private fn), `family17_market_microstructure.rs` (inlined in multiple places), `family9_volatility.rs` (inlined) | `time_series.rs` already has `difference` — add `log_returns` |
| 10 | `variance_of(v)` — population or sample variance of a slice | `time_series.rs` (multiple inline), `data_quality.rs:split_variance_ratio:233` (lambda), `data_quality_catalog.rs:skewness_fisher` (inline), `complexity.rs:hurst_rs:227` (inline) | `descriptive.rs` (already has `moments_ungrouped` — many callers duplicate the math rather than calling it) |
| 11 | `mean_of(v)` — simple arithmetic mean | `data_quality.rs:price_cv:83` (inline), `data_quality.rs:lag1_autocorrelation:422` (inline), `data_quality.rs:has_vol_clustering:324` (inline), `complexity.rs:hurst_rs:215` (inline), `family11_tick_microstructure.rs:lag1_autocorr:49` (inline), `family17_market_microstructure.rs` (multiple inline), `time_series.rs` (many inline) — **>15 independent copies of `iter().sum() / n`** | `descriptive.rs::moments_ungrouped` — callers should use `.mean()` |
| 12 | `cv(values)` — coefficient of variation (std/|mean|) | `family17_market_microstructure.rs:221` (private fn), `data_quality.rs:price_cv:77` (standalone pub fn), `data_quality.rs:sampling_regularity_cv:166` (specialized timestamp variant) | `data_quality.rs` already has `price_cv` — fintek should call it |
| 13 | `inversion_count(x)` — merge-sort counting of inversions | `nonparametric.rs` (public — this IS the canonical), `data_quality_catalog.rs:count_inversions:212` (delegates correctly — OK), **but** `data_quality.rs` has inline pairwise versions in older functions that predate the public primitive | `nonparametric.rs` (already done — verify all callers route through it) |
| 14 | `ols_rss(y, x, nobs, ncols)` — OLS residual sum of squares via Gauss-Jordan | `family12_causality_info.rs:65` (private fn, Gauss-Jordan normal equations) — duplicates `linear_algebra::qr_solve` + RSS computation | `linear_algebra.rs` — expose `ols_rss` or have Granger call `qr_solve` |
| 15 | `pairwise_dists(mat, n, d)` — Euclidean pairwise distance matrix | `family15_manifold_topology.rs:36` (private fn), `complexity.rs:correlation_dimension:462` (inline), `complexity.rs:largest_lyapunov:541` (inline) | `spatial.rs` or `manifold.rs` — this is a fundamental geometric primitive |
| 16 | `delay_embed(returns, d, tau, max_pts)` — Takens delay embedding | `family15_manifold_topology.rs:18` (private fn), `complexity.rs:correlation_dimension:456` (inline), `complexity.rs:largest_lyapunov:526` (inline) | `complexity.rs` should export `delay_embed` publicly; fintek should call it |
| 17 | `knn_adjacency(dists, n, k)` — k-nearest-neighbor graph construction from distance matrix | `family15_manifold_topology.rs:54` (private fn) — foundational graph primitive | `spatial.rs` or `graph.rs` — used by all manifold/topological methods |
| 18 | `graph_laplacian(adj, n)` — degree-normalized Laplacian from adjacency | `family15_manifold_topology.rs:75` (private fn) — foundational graph primitive | `graph.rs` — used by spectral clustering, diffusion maps, spectral embedding |

---

## Category 2: Private functions that should be public

Correct implementation, wrong visibility. Each of these is general math that has independent value.

| # | Function | File & approx line | What it computes | Who else needs it |
|---|----------|--------------------|-----------------|-------------------|
| 19 | `arma_css_residuals(centered, ar, ma)` | `time_series.rs:331` | ARMA conditional residuals (sequential recursion) | ARMA, ARIMA, SARIMA, VARMA, Ljung-Box residual checking, impulse response functions |
| 20 | `forward_solve(l, b)` | `multivariate.rs:318` | Forward substitution: solves Lx=b where L is lower-triangular | Cholesky-based solvers everywhere — duplicates internal logic of `linear_algebra::cholesky_solve` |
| 21 | `back_solve_transpose(l, b)` | `multivariate.rs:427` | Back-substitution for L^T x=b | LDA, CCA, any Cholesky-based generalized eigenproblem |
| 22 | `sscp_matrices(x, groups)` | `multivariate.rs:91` | Between/within SSCP for grouped data | MANOVA, LDA, multivariate tests, mixed effects — called from both `manova` and `lda`, should be extracted and public |
| 23 | `mackinnon_adf_critical_values(n)` | `time_series.rs:918` | MacKinnon (2010) ADF response surface critical values | ADF test, PP test, DF-GLS, KPSS, any unit root test needing MacKinnon CVs |
| 24 | `interpret_bf(bf10)` | `hypothesis.rs:1176` | Jeffreys/Kass-Raftery verbal Bayes factor interpretation | Any Bayesian hypothesis test: t-test BF, correlation BF, ANOVA BF, mixed model BF |
| 25 | `ols_simple(x, y)` | `hypothesis.rs:1321` | 1-predictor OLS: returns (intercept, slope, residuals, se_slope) | Mediation analysis, simple regression diagnostics, any 1D regression caller |
| 26 | `ols_two_predictor(x, m, y)` | `hypothesis.rs:1343` | 2-predictor OLS via Cramer's rule (3×3 normal equations) | Mediation analysis (current consumer), any 2-predictor model — should generalize or call `qr_solve` |
| 27 | `count_matches(data, m, r)` | `complexity.rs:72` | Template matches for SampEn/ApEn: counts pairs within L∞ tolerance | SampEn, ApEn, cross-approximate entropy (XApEn), FuzzyEn — shared counting core |
| 28 | `phi_func(data, m, r)` | `complexity.rs:94` | ApEn φ function (with self-matches) | ApEn and its variants |
| 29 | `pattern_to_index(pattern, m)` | `complexity.rs:165` | Ordinal pattern → Lehmer code index | Permutation entropy, ordinal pattern analysis, symbolic dynamics — currently also inlined in `data_quality.rs:unique_ordinal_3` |
| 30 | `factorial(n)` | `complexity.rs:177` | Integer factorial | Permutation entropy normalization, combinatorics, binomial coefficients — used in multiple places |
| 31 | `linear_fit_segment(segment)` | `complexity.rs:325` | (intercept, slope) via centered OLS for a 0..n indexed segment | DFA (current consumer), Higuchi FD (uses OLS via `ols_slope`), any windowed linear detrending |
| 32 | `estimate_mean_period(data)` | `complexity.rs:580` | Mean period via zero-crossing count | Rosenstein Lyapunov (current consumer), minimum mutual information for embedding, any period estimation |
| 33 | `expected_mutual_info(a, b, n)` | `information_theory.rs:417` | Expected MI under hypergeometric model (exact AMI) | AMI score (current consumer), any adjusted clustering metric requiring E[MI] |
| 34 | `contingency_from_labels(a, b)` | `information_theory.rs:346` | Build contingency table from label vectors | MI score, NMI score, AMI score, VI, Fowlkes-Mallows — all clustering evaluation metrics |
| 35 | `p_log_p(p)` | `information_theory.rs:44` | Safe p·ln(p) with 0 for p≤0 | Every entropy/divergence computation — should be public utility |
| 36 | `p_log_p_over_q(p, q)` | `information_theory.rs:49` | Safe p·ln(p/q) | KL divergence, cross-entropy, any divergence computation |
| 37 | `ks_p_value(d, n)` | `nonparametric.rs` (prior audit) | Kolmogorov distribution p-value | KS test, Lilliefors, Anderson-Darling-related p-values |
| 38 | `shapiro_wilk_coefficients(n)` | `nonparametric.rs` (prior audit) | Expected normal order statistics for SW test | Shapiro-Wilk, Shapiro-Francia, any normality test using order stats |
| 39 | `morlet_fft(m, scale)` | `family7_wavelets.rs:88` | Morlet wavelet in frequency domain | CWT (current consumer), any Morlet-based time-frequency analysis — should live in `signal_processing.rs` |
| 40 | `fft_full(x)` | `family7_wavelets.rs:41` | O(n²) complex DFT | CWT, coherence, any small-n frequency analysis — should be `signal_processing::dft_naive` |
| 41 | `delay_covariance(mat, n, d)` | `family15_manifold_topology.rs:100` | Covariance matrix from delay-embedded rows | Any manifold/phase-space method needing the covariance of embedded coordinates |
| 42 | `top_k_right_sv(mat, n, d, k)` | `family15_manifold_topology.rs:292` | Top-k right singular vectors of (n×d) matrix | Grassmannian leaf, any SVD-based subspace method |
| 43 | `ami_at_lag(x, tau)` | `family15_manifold_topology.rs:460` | Average mutual information at a given lag | Optimal embedding delay selection (FNN), any AMI-based analysis |
| 44 | `fnn_frac(x, d, tau)` | `family15_manifold_topology.rs:496` | False nearest neighbor fraction for embedding | Optimal embedding dimension selection — general primitive for all chaotic system analysis |
| 45 | `ccm_predict(embed, target, lib_size)` | `family24_manifold.rs:96` | Convergent cross-mapping prediction via kNN | CCM causality test (current consumer), any manifold-based prediction |
| 46 | `hankel_r_stat(data, embed_dim)` | `family24_manifold.rs:258` | Wigner r-statistic from Hankel SVD | Harmonic leaf, RMT analysis, any random matrix theory application |
| 47 | `segment_cost(cumsum, cumsum2, start, end)` | `family21_changepoint.rs:30` | Squared-error segment cost for changepoint PELT | PELT changepoint (current consumer), BOCPD, binary segmentation — canonical cost function |
| 48 | `effective_rank_from_sv(sv)` | `family11_tick_microstructure.rs:1354` | Effective rank from singular values: exp(H(sv²/total)) | PCA rank selection, any effective dimension measure |
| 49 | `matrix_effective_rank(mat, m)` | `family11_tick_microstructure.rs:1367` | Effective rank of a matrix via SVD | Should compose `svd` + `effective_rank_from_sv` — the combination is a primitive |
| 50 | `build_compression_matrix(price, volume)` | `family11_tick_microstructure.rs:1311` | Lag-feature matrix for compression complexity | Compression leaf (current consumer) — general lag-feature construction |
| 51 | `histogram_entropy(values, n_bins)` | `family11_tick_microstructure.rs:23` (private) | Shannon entropy of a histogram | Any discrete entropy computation — should call `information_theory::entropy_histogram` |
| 52 | `power_law_exponent(log_k, log_pk)` | `family20_visibility_graphs.rs:295` | Negated OLS slope in log-log space | Visibility graph degree exponent (current consumer), any power-law fitting — should call `linear_algebra::ols_slope` |
| 53 | `polyfit_predict_error(ticks, order)` | `family23_taylor_fold.rs:146` | Polynomial fit prediction error | Taylor fold leaf (current consumer), any polynomial regression residual |
| 54 | `classify_regime(r)` | `family24_manifold.rs:304` | Classify r-statistic vs Poisson/GOE thresholds | Harmonic leaf (current consumer), any RMT regime classification |
| 55 | `regularize_interp/bin_mean/subsample` | `family7_wavelets.rs:209` (dispatches to signal_processing) | The dispatch wrapper already delegates — but the `RegStrategy` enum and `regularize` fn are private | `signal_processing.rs` should expose these directly |

---

## Category 3: Inline code blocks (>10 lines) that should be extracted

These are embedded computations inside pub functions that are non-trivially reusable.

| # | Location (file:approx_line) | What the inline block computes | Should become |
|---|-----------------------------|-------------------------------|---------------|
| 56 | `complexity.rs:hurst_rs:199–244` | R/S statistic for one block size: mean, cumulative deviations, range, std | `fn rescaled_range(block: &[f64]) -> f64` |
| 57 | `complexity.rs:correlation_dimension:456–499` | Time-delay embedding → L∞ pairwise distances → sorted distance vector | Call `delay_embed` + `pairwise_dists` (primitives that exist but aren't used here) |
| 58 | `complexity.rs:largest_lyapunov:526–577` | Same embedding + L2 distances + NN search | Same — call `delay_embed` + `pairwise_dists` |
| 59 | `complexity.rs:lempel_ziv_complexity:403–406` | Binarize at median | `fn binarize_at_median(data: &[f64]) -> Vec<bool>` — standalone primitive |
| 60 | `time_series.rs:ar_fit:36–44` | Autocorrelation computation r[0..p] | Should call `acf(data, p)` (already public) |
| 61 | `time_series.rs:arma_fit:424–439` | Numerical gradient (central differences) | `fn numerical_gradient(f: impl Fn(&[f64])->f64, x: &[f64], eps: f64) -> Vec<f64>` — should live in `numerical.rs` |
| 62 | `time_series.rs:adf_test:847–856` | OLS via normal equations (XTX, XTy build) | Should call `linear_algebra::qr_solve` or the existing `cholesky_solve` path |
| 63 | `time_series.rs:breusch_godfrey:2228–2239` | Same OLS normal equations build | Same — should call `qr_solve` |
| 64 | `time_series.rs:pacf:969–1005` | Levinson-Durbin recursion for PACF | This IS the Levinson-Durbin algorithm — should be `fn levinson_durbin(r: &[f64]) -> (Vec<f64>, f64)` |
| 65 | `multivariate.rs:cca:479–487` | Cross-covariance Σ_XY computation | Should be `fn cross_covariance_matrix(x: &Mat, y: &Mat, ddof: usize) -> Mat` — a public primitive |
| 66 | `multivariate.rs:manova:271–285` | L⁻¹·H·L⁻ᵀ similarity transform | Should be `fn cholesky_similarity(l: &Mat, h: &Mat) -> Mat` — used in both `manova` and `lda` |
| 67 | `hypothesis.rs:bayes_factor_t_one_sample:1219–1248` | Log-spaced Simpson integration | `fn log_spaced_simpson(f: impl Fn(f64)->f64, log_a: f64, log_b: f64, n: usize) -> f64` in `numerical.rs` |
| 68 | `hypothesis.rs:wls:1143–1153` | Weighted mean computation | `fn weighted_mean(x: &[f64], w: &[f64]) -> f64` — general primitive |
| 69 | `data_quality.rs:split_variance_ratio:233` | Lambda `var_half` that computes variance of a subslice | Should call `descriptive::moments_ungrouped` |
| 70 | `data_quality.rs:trend_r2:260–276` | OLS R² for linear trend vs index | Should call `linear_algebra::simple_linear_regression` (already public) |
| 71 | `data_quality.rs:has_vol_clustering:319–338` | Lag-1 autocorrelation of squared returns | Should call `data_quality::lag1_autocorrelation(r2)` (already public) |
| 72 | `data_quality_catalog.rs:sample_std:240–249` | Sample std of finite values | Should call `descriptive::moments_ungrouped` |
| 73 | `data_quality_catalog.rs:skewness_fisher:354–368` | Central moments m2, m3 | Should call `descriptive::moments_ungrouped` |
| 74 | `family9_volatility.rs:vol_dynamics` (inline) | AR(1) coefficient via simple OLS on rolling vol | Should call `linear_algebra::ols_slope` |
| 75 | `family12_causality_info.rs:te_histogram:763` | Transfer entropy from histogram of quantized values | `fn transfer_entropy_discrete(x: &[usize], y: &[usize]) -> f64` — standalone primitive |
| 76 | `family12_causality_info.rs:quantize:801` | Quantize continuous data into rank-based bins | `fn quantize_ranks(data: &[f64], n_bins: usize) -> Vec<usize>` — general primitive |
| 77 | `family11_tick_microstructure.rs:convex_hull_area:70–114` | 2D convex hull area via gift-wrapping + shoelace | `fn convex_hull_area(points: &[(f64,f64)]) -> f64` — should live in `spatial.rs` |
| 78 | `family15_manifold_topology.rs:diff_geometry:222–248` | Menger curvature at consecutive triplets | `fn menger_curvature(a: [f64;3], b: [f64;3], c: [f64;3]) -> f64` |
| 79 | `family15_manifold_topology.rs:diff_geometry:245–249` | Cross product of 3D vectors | `fn cross_product_3d(v1: [f64;3], v2: [f64;3]) -> [f64;3]` — should live in `linear_algebra.rs` |
| 80 | `family24_manifold.rs:ccm_predict` | Exponential kNN weighting: `exp(-d / (d_min + eps))` | `fn exponential_knn_weights(dists: &[f64], eps: f64) -> Vec<f64>` |
| 81 | `information_theory.rs:mutual_info_miller_madow:540–568` | Inline Pearson r from contingency table | Should call `descriptive` or `nonparametric` Pearson — not inline inside MI |

---

## Category 4: Hardcoded constants that should be parameters (Principle 4 violations)

| # | Location | Hardcoded constant | What it controls | Should be `using()` parameter with this default |
|---|-----------|--------------------|-----------------|--------------------------------------------------|
| 82 | `family17_market_microstructure.rs:vol_regime` | `const HIGH_VOL_THRESHOLD: f64 = 1.5` | Variance ratio above which regime is "high vol" | `high_vol_threshold: f64 = 1.5` |
| 83 | `family17_market_microstructure.rs:vol_regime` | `const SHORT: usize = 20`, `const LONG: usize = 100` | Short/long variance windows | `short_window: usize = 20`, `long_window: usize = 100` |
| 84 | `family9_volatility.rs:vol_regime` | `const HIGH_VOL_THRESH: f64 = 1.5` | Same hardcoded threshold as family17 (duplicate!) | `high_vol_threshold: f64 = 1.5` |
| 85 | `family9_volatility.rs:vol_regime` | `const SHORT: usize = 20`, `const LONG: usize = 100` | Same windows as family17 (duplicate!) | `short_window`, `long_window` |
| 86 | `family17_market_microstructure.rs:vpin_bvc` | `const N_BUCKETS: usize = 20` | Number of VPIN volume buckets | `n_buckets: usize = 20` |
| 87 | `family12_causality_info.rs` | `const GRANGER_MIN_OBS: usize = 30`, `const GRANGER_MAX_LAG: usize = 5` | Granger test minimum sample and maximum lag | `min_obs: usize = 30`, `max_lag: usize = 5` |
| 88 | `irt.rs:fit_2pl` | `theta.clamp(-6.0, 6.0)` — hardcoded ability range | Theta clamp prevents MLE boundary issues | `theta_min: f64 = -6.0`, `theta_max: f64 = 6.0` |
| 89 | `irt.rs:fit_2pl` | `a.clamp(0.1, 5.0)` — hardcoded discrimination range | Discrimination stability constraint | `disc_min: f64 = 0.1`, `disc_max: f64 = 5.0` |
| 90 | `irt.rs:fit_2pl` | `b.clamp(-5.0, 5.0)` — hardcoded difficulty range | Difficulty stability constraint | `diff_min: f64 = -5.0`, `diff_max: f64 = 5.0` |
| 91 | `complexity.rs:hurst_rs` | `let min_block = 10` — hardcoded minimum R/S block size | Controls minimum block size for R/S analysis | `min_block: usize = 10` |
| 92 | `complexity.rs:correlation_dimension` | `let n_r = 20` — number of r values for C(r) | Controls resolution of correlation integral | `n_radii: usize = 20` |
| 93 | `complexity.rs:correlation_dimension` | `r_min = distances[len/10]`, `r_max = distances[len*9/10]` | Percentile range for r values | `r_pct_lo: f64 = 0.10`, `r_pct_hi: f64 = 0.90` |
| 94 | `complexity.rs:largest_lyapunov` | `let max_diverge = n_vectors / 4` — divergence tracking length | Controls Lyapunov fit window | `diverge_fraction: f64 = 0.25` |
| 95 | `data_quality.rs:fft_is_valid` | `tick_count(x) < 64`, `nyquist_bins(x) < 32`, `cv >= 0.5`, `gap_ratio >= 10.0` | All four FFT validity thresholds | `min_ticks`, `min_nyquist_bins`, `max_sampling_cv`, `max_gap_ratio` — all hardcoded |
| 96 | `data_quality.rs:garch_is_valid` | `tick_count(returns) < 100`, `threshold = 0.05` | GARCH validity minimum and vol-clustering threshold | `min_ticks: usize = 100`, `vol_clustering_threshold: f64 = 0.05` |
| 97 | `family15_manifold_topology.rs:spectral_embedding` | `let d = 3`, `let k_nn = 5`, `let max_pts = 200` | Embedding dim, kNN count, point cap | `embed_dim`, `k_neighbors`, `max_points` — all hardcoded |
| 98 | `family15_manifold_topology.rs:diff_geometry` | `let d = 3`, `let max_pts = 2000` | Embedding dim, point cap | Same |
| 99 | `family7_wavelets.rs` | `const OMEGA_0: f64 = 6.0` — Morlet central frequency | Controls wavelet time-frequency tradeoff | `omega_0: f64 = 6.0` |
| 100 | `hypothesis.rs:bayes_factor_t_one_sample` | `g_min = 1e-8`, `g_max = 1e4`, `n_points = 4000` | Integration range and resolution | `g_min`, `g_max`, `n_integration_points` |
| 101 | `time_series.rs:newey_west_lrv` | `4.0 * (n/100)^(2/9)` — automatic bandwidth | Newey-West lag truncation rule (Andrews 1991) | `bandwidth_rule: BandwidthRule = Andrews1991` |
| 102 | `family24_manifold.rs` | `const HARMONIC_R_POISSON: f64 = 0.38629`, `const HARMONIC_R_GOE: f64 = 0.53590` | RMT classification thresholds | These are mathematically derived constants, not free parameters — but `classify_regime` should expose them as configurable for research |

---

## Priority Rankings

### Tier 1 — Fix now (highest blast radius, easiest wins)

These are pure duplicates with no architectural complexity:

1. **rolling_variance_prefix** — identical fn in family9 and family17, extract to `time_series.rs`
2. **median from sorted** — 6 inline copies, extract to `descriptive::median_from_sorted`
3. **mean_of(v)** — >15 inline copies, route all callers through `moments_ungrouped().mean()`
4. **log_returns_from_prices** — 3+ copies in fintek, extract once to `time_series.rs`
5. **lag1_autocorr** — fintek's private fn should call `data_quality::lag1_autocorrelation`
6. **erfc_approx / libm_erf** — both should call `special_functions::erfc`
7. **power_law_exponent** — should call `linear_algebra::ols_slope`
8. **data_quality::trend_r2** — should call `linear_algebra::simple_linear_regression`
9. **data_quality::has_vol_clustering** — should call `data_quality::lag1_autocorrelation`
10. **histogram_entropy in fintek** — should call `information_theory::entropy_histogram`

### Tier 2 — Extract before next method wave

These require a small function to be created and exposed:

- `delay_embed` → public in `complexity.rs`
- `pairwise_dists` → public in `spatial.rs`
- `knn_adjacency`, `graph_laplacian` → public in `graph.rs`
- `levinson_durbin` → extracted from `time_series::pacf`
- `cross_covariance_matrix` → extracted from `multivariate::cca`
- `cholesky_similarity` → extracted from `manova` and `lda`
- `numerical_gradient` → extracted from `arma_fit`, moved to `numerical.rs`

### Tier 3 — Parameter exposure (Principle 4)

Lower risk, affects only the using() surface:

- All hardcoded window sizes, thresholds, and integration constants should be optional parameters
- Priority: spectral_embedding constants (d=3, k=5), VPIN N_BUCKETS, GARCH thresholds

---

## Cross-cutting observation: the ols_slope convergence problem

Seven different places compute OLS in different ways:
- `linear_algebra::ols_slope` — global primitive (correct home)
- `linear_algebra::simple_linear_regression` — richer version with residuals (correct home)
- `family20_visibility_graphs::power_law_exponent` — reinvented OLS slope (should delegate)
- `family12_causality_info::ols_rss` — Gauss-Jordan, no Cholesky (should use `qr_solve`)
- `hypothesis::ols_simple` — private fn with residuals (should use `simple_linear_regression`)
- `hypothesis::ols_two_predictor` — 3×3 Cramer's rule (special case of `qr_solve`)
- `data_quality::trend_r2` — inline slope computation (should use `simple_linear_regression`)

This is the most concrete evidence of decomposition debt. Every OLS computation beyond the two
canonical functions in `linear_algebra.rs` is a violation of the Methods-Are-Compositions principle.

---

*Scanned files: all 97 .rs files in crates/tambear/src/ and crates/tambear-fintek/src/*
*Prior audit: ~/.claude/garden/2026-04-09-primitive-extraction-audit.md (20 findings)*
*This audit: 102 findings (82 new, all 20 prior confirmed and cross-referenced)*
