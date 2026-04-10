# TBS Surface Audit

Every pub fn in tambear examined for TBS expressibility, primitive composition,
using() coverage, missing flavors, and compound decomposition.

Gap taxonomy:
- **MISSING_PRIMITIVE**: math that should exist as a standalone fn but doesn't
- **MISSING_PARAMETER**: hardcoded value that should be a using() key
- **MISSING_COMPOSITION**: method that embeds sub-operations instead of calling primitives
- **MISSING_USING_KEY**: parameter exists on the fn but isn't wirable through using()
- **MISSING_FAMILY_TAG**: primitive that should be discoverable under multiple family names
- **COMPOUND_PRIMITIVE**: does more than one thing, should be decomposed
- **MISSING_FLAVOR**: alternative method/variant exists in literature but not implemented

---

## descriptive

### MomentStats (Level 0 Primitive)
TBS: `moments(col=0)`
Composes: single-pass Welford accumulator (mean, m2, m3, m4, min, max, count)
using() keys: ddof
Level 0: is a primitive (accumulate pattern)
Family tags: statistics, descriptive, sufficient_statistics
OK: clean primitive, mergeable, TamSession-shareable

### mean (Level 0 Primitive)
TBS: `mean(col=0)`
Composes: `moments(col=0)` -> extract mean
using() keys: method("arithmetic"|"trimmed"|"winsorized"|"geometric"|"harmonic"|"power"|"weighted")
Level 0: arithmetic mean exists via MomentStats.mean()
MISSING_FLAVOR: power_mean(p), Lehmer mean, contraharmonic mean, exponential mean,
  Frechet mean, interquartile mean, midrange (exists in data_quality), midhinge,
  truncated mean (different from trimmed — trimmed discards, truncated re-estimates)
GAP: geometric_mean, harmonic_mean, trimmed_mean, winsorized_mean exist as
  SEPARATE free functions, not as using() keys on a unified mean primitive.
  weighted_mean exists in numerical.rs, not linked.
MISSING_USING_KEY: fraction (for trimmed/winsorized), power (for power mean)

### variance / std (Level 0 Primitive)
TBS: `variance(col=0)` / `std(col=0)`
Composes: `moments(col=0)` -> extract variance
using() keys: ddof(0|1|n), method("standard"|"robust")
MISSING_PARAMETER: ddof is a function parameter, not a using() key
MISSING_FLAVOR: robust variance (MAD-based, Qn, Sn, tau-scale — these exist in robust.rs
  but not composable through using(method="mad"))

### skewness (Level 0 Primitive)
TBS: `skewness(col=0)`
using() keys: bias(true|false), method("fisher"|"bowley"|"pearson_first"|"kelly")
MISSING_COMPOSITION: bowley_skewness, pearson_first_skewness exist as separate fns,
  not routed through using()
MISSING_FLAVOR: Groeneveld-Meeden, L-moment skewness, Hosking-Wallis

### kurtosis (Level 0 Primitive)
TBS: `kurtosis(col=0)`
using() keys: excess(true|false), bias(true|false), method("fisher"|"crow_siddiqui")
MISSING_FLAVOR: L-kurtosis, percentile kurtosis (Moors), hogg kurtosis

### quantile (Level 0 Primitive)
TBS: `quantile(col=0, q=0.5)`
using() keys: method("linear"|"linear4"|"hazen"|"weibull"|"inverse_cdf"|"median_unbiased"|"normal_unbiased")
Level 0: proper primitive with 7 methods
Family tags: statistics, descriptive, order_statistics
OK: well parameterized. Has 7 of 9 scipy methods.
MISSING_FLAVOR: scipy types 1-9 naming (ours map but names differ),
  weighted quantile, exponentially weighted quantile

### median (Level 0+ Composition)
TBS: `median(col=0)` = `quantile(col=0, q=0.5)`
Level 0: exists as convenience fn. Calls quantile with hard-coded Linear method.
MISSING_PARAMETER: quantile method is hardcoded to midpoint average for even n;
  should delegate to quantile(q=0.5) with using(method=...) passthrough

### quartiles (Level 0+ Composition)
TBS: `quartiles(col=0)`
Composes: quantile(0.25) + quantile(0.50) + quantile(0.75), all hardcoded to Linear
MISSING_USING_KEY: quantile method not passable

### iqr (Level 0+ Composition)
TBS: `iqr(col=0)`
Composes: quartiles -> Q3-Q1
MISSING_USING_KEY: quantile method hardcoded

### geometric_mean (Level 0 Primitive)
TBS: `geometric_mean(col=0)` or `mean(col=0).using(method="geometric")`
Level 0: exists as standalone fn
Family tags: statistics, descriptive, central_tendency
OK as standalone. But also a composition: `log(col=0).mean().exp()`
MISSING_COMPOSITION: not wired as using() key on mean()

### harmonic_mean (Level 0 Primitive)
TBS: `harmonic_mean(col=0)` or `mean(col=0).using(method="harmonic")`
Same issues as geometric_mean.
Also a composition: `reciprocal(col=0).mean().reciprocal()`

### trimmed_mean (Level 0 Primitive)
TBS: `trimmed_mean(col=0, fraction=0.1)`
using() keys: fraction
MISSING_USING_KEY: not wirable through unified mean

### winsorized_mean (Level 0 Primitive)
TBS: `winsorized_mean(col=0, fraction=0.1)`
Same issues as trimmed_mean.

### mad (Level 0 Primitive)
TBS: `mad(col=0)`
Composes: median of |x - median(x)|
using() keys: center("median"|"mean"), scale_factor(1.4826 for normal consistency)
MISSING_PARAMETER: consistency constant 1.4826 not applied (raw MAD returned).
  R's mad() has constant=1.4826 by default. scipy has scale parameter.
MISSING_USING_KEY: center, constant

### gini (Level 0 Primitive)
TBS: `gini(col=0)`
Composes: sort -> pairwise absolute difference formula
using() keys: (none needed)
Level 0: proper primitive
Family tags: statistics, inequality, economics
MISSING_FLAVOR: concentration ratio, Lorenz curve (exists implicitly)

### qq_normal (Level 1 Method)
TBS: `qq(col=0, distribution="normal")`
Composes: sort -> normal_quantile for each rank
MISSING_PARAMETER: plotting position formula uses Blom (0.375, 0.25) hardcoded.
  R has type= parameter (Blom, Hazen, etc.)
MISSING_USING_KEY: distribution (only normal supported), position_formula
MISSING_FLAVOR: qq against t, uniform, exponential, arbitrary distribution

### forecast_metrics (Level 1 Method)
TBS: `forecast_metrics(actual=col0, predicted=col1)`
COMPOUND_PRIMITIVE: computes MAE + RMSE + MAPE + MASE simultaneously.
  Each should be independently callable: `mae(actual, predicted)`, etc.
MISSING_PRIMITIVE: mae, rmse, mape, mase as standalone primitives
MISSING_PARAMETER: MAPE zero-protection threshold hardcoded at 1e-15
MISSING_FLAVOR: sMAPE, WMAPE, RMSLE, MdAE, MRAE

### box_cox_fit (Level 1 Method)
TBS: `box_cox(col=0)` or `box_cox_fit(col=0)`
Composes: box_cox_transform + box_cox_log_likelihood + golden_section_search
using() keys: lambda_min(-2), lambda_max(2), method("golden_section")
Level 0 primitives needed: box_cox_transform OK, box_cox_log_likelihood OK,
  but uses internal golden section, not the public optimization::golden_section
MISSING_COMPOSITION: does not call public golden_section primitive
MISSING_FLAVOR: Yeo-Johnson (handles negative values), modulus transform
COMPOUND_PRIMITIVE: fit = search + transform + likelihood

### sorted_nan_free (Level 0 Primitive)
TBS: `sort(col=0).drop_nan()`
Composes: NaN filter + sort
OK: utility function, properly decomposable

### cv (Level 0+ Composition)
TBS: `cv(col=0)` = `std(col=0) / mean(col=0)`
Composes: std / mean from MomentStats
Level 0: on MomentStats
OK

### sem (Level 0+ Composition)
TBS: `sem(col=0)` = `std(col=0) / sqrt(n)`
Composes: std / sqrt(count)
OK

### describe (Level 1 Method)
TBS: `describe(col=0)`
COMPOUND_PRIMITIVE: returns (mean, std, min, max, skew, kurtosis, n) all at once.
  Each is independently callable. This is a convenience bundle.
  Acceptable as Level 1, but should be documented as a composition.

---

## nonparametric

### rank (Level 0 Primitive)
TBS: `rank(col=0)`
using() keys: method("average"|"min"|"max"|"ordinal"|"dense"), na_option("keep"|"first"|"last")
Level 0: proper primitive
Family tags: statistics, nonparametric, order_statistics
MISSING_PARAMETER: tie-breaking method is hardcoded to "average".
  R's rank() has ties.method=("average","first","last","random","max","min").
  scipy.stats.rankdata has method=("average","min","max","ordinal","dense").
MISSING_USING_KEY: ties_method, na_option

### spearman (Level 1 Method)
TBS: `spearman(col_x=0, col_y=1)`
Composes: rank(col_x) -> rank(col_y) -> pearson_on_ranks(rx, ry)
using() keys: (none beyond what flows through to rank)
Level 0 primitives needed: rank OK, pearson_on_ranks OK
OK: properly decomposed already. Model composition.
Family tags: statistics, correlation, nonparametric
MISSING_FLAVOR: Spearman footrule, Spearman rho with correction for ties

### kendall_tau (Level 1 Method)
TBS: `kendall_tau(col_x=0, col_y=1)`
Composes: co_sort -> inversion_count -> tie_count -> tau_b_formula
using() keys: variant("tau_a"|"tau_b"|"tau_c"), inversion_method("mergesort"|"fenwick")
Level 0 primitives needed: inversion_count OK (has mergesort variant)
MISSING_PARAMETER: variant hardcoded to tau_b. tau_a and tau_c not available.
MISSING_USING_KEY: variant, inversion_method (tau uses private kendall_merge_sort_count,
  not the public inversion_count primitive)
MISSING_COMPOSITION: tie counting is embedded (private loop), should be a
  separate tie_count primitive. Joint tie counting also private.
MISSING_FLAVOR: tau_a, tau_c, Stuart's tau_c, Goodman-Kruskal gamma
MISSING_PRIMITIVE: tie_count(col=0) as standalone

### inversion_count / inversion_count_mergesort (Level 0 Primitive)
TBS: `inversion_count(col=0)`
using() keys: method("mergesort"|"fenwick"|"naive")
Level 0: proper primitive, well documented
Family tags: combinatorics, nonparametric, order_statistics
MISSING_FLAVOR: fenwick tree variant (documented as desired but not implemented)

### pearson_on_ranks (Level 0 Primitive)
TBS: `pearson_on_ranks(col_x=0, col_y=1)`
Level 0: proper primitive. Is the Pearson correlation formula applied to pre-ranked data.
Family tags: statistics, correlation
OK

### pearson_r (Level 0 Primitive)
TBS: `pearson_r(col_x=0, col_y=1)`
using() keys: (none)
Level 0: proper primitive
Family tags: statistics, correlation, parametric
MISSING_USING_KEY: confidence_interval (should return CI), alternative("two_sided"|"less"|"greater")
MISSING_FLAVOR: weighted Pearson, partial Pearson (partial_correlation exists separately)

### partial_correlation (Level 1 Method)
TBS: `partial_correlation(col_x=0, col_y=1, covariates=[col2, col3])`
Composes: OLS residuals of x on covariates, OLS residuals of y on covariates,
  then pearson_r of residuals
using() keys: method("ols_residuals"|"recursive")
Level 0 primitives: ols_residuals not exposed as standalone
MISSING_PRIMITIVE: ols_residuals(col, covariates) as standalone primitive
COMPOUND_PRIMITIVE: partial_correlation_full returns r, p, t, df simultaneously

### mann_whitney_u (Level 1 Method)
TBS: `mann_whitney_u(col_x=0, col_y=1)`
Composes: combine -> rank -> sum_ranks_group1 -> U_formula -> normal_approx_z -> p_value
using() keys: alternative("two_sided"|"less"|"greater"), continuity_correction(true|false),
  exact(true|false)
Level 0 primitives: rank OK. Tie correction done inline.
MISSING_USING_KEY: alternative (hardcoded two-sided), continuity_correction,
  exact (no exact p-value, always normal approximation)
MISSING_COMPOSITION: tie counting is embedded (private loop), should use tie_count primitive
MISSING_PARAMETER: continuity correction not applied (scipy applies 0.5 correction by default)
MISSING_FLAVOR: exact U test for small samples (scipy method="exact")

### wilcoxon_signed_rank (Level 1 Method)
TBS: `wilcoxon_signed_rank(col=0)` or with `median0` parameter
Composes: filter_zeros -> abs -> rank -> signed_sum -> normal_approx
using() keys: alternative("two_sided"|"less"|"greater"), method("auto"|"approx"|"exact"),
  correction(true|false), zero_method("wilcox"|"pratt"|"zsplit")
MISSING_USING_KEY: alternative (hardcoded two-sided), method, correction, zero_method
MISSING_PARAMETER: continuity correction absent, zero handling is "wilcox" only
MISSING_COMPOSITION: tie correction is embedded private loop
MISSING_FLAVOR: Pratt's method (keeps zeros), z-split method

### kruskal_wallis (Level 1 Method)
TBS: `kruskal_wallis(col=0, groups=col1)`
Composes: rank -> group_rank_sums -> H_formula -> chi2_p_value
using() keys: tie_correction(true|false), nan_policy
MISSING_USING_KEY: nan_policy
Level 0 primitives: rank OK
OK for basic test but missing post-hoc:
MISSING_FLAVOR: Conover-Iman post-hoc (dunn_test exists but separate, not linked)

### dunn_test (Level 1 Method)
TBS: `dunn_test(col=0, groups=col1)`
Composes: kruskal_wallis ranks -> pairwise z-scores -> bonferroni correction
using() keys: correction("bonferroni"|"sidak"|"holm"|"bh"|"none")
MISSING_USING_KEY: correction method hardcoded to Bonferroni
MISSING_COMPOSITION: re-ranks data internally instead of reusing kruskal_wallis ranks

### shapiro_wilk (Level 1 Method)
TBS: `shapiro_wilk(col=0)`
Composes: sort -> shapiro_wilk_coefficients -> W_formula -> p_value (Royston approx)
using() keys: (none typical)
Level 0 primitives: shapiro_wilk_coefficients OK as standalone
Family tags: statistics, normality, goodness_of_fit
MISSING_FLAVOR: Chen-Shapiro (for non-normal), Lilliefors, Cramer-von Mises

### dagostino_pearson (Level 1 Method)
TBS: `dagostino_pearson(col=0)`
Composes: moments -> skewness_z -> kurtosis_z -> K2 = z_s^2 + z_k^2 -> chi2_p
using() keys: (none)
COMPOUND_PRIMITIVE: internally computes skewness and kurtosis z-scores which
  are useful independently

### jarque_bera (Level 1 Method, duplicate)
TBS: `jarque_bera(col=0)`
Note: exists in BOTH nonparametric.rs AND hypothesis.rs. Identical math.
  The hypothesis.rs version takes MomentStats, the nonparametric.rs version takes raw data.
GAP: duplicate implementation. Should be one primitive that accepts MomentStats,
  with a convenience wrapper.

### ks_test_normal (Level 1 Method)
TBS: `ks_test(col=0, distribution="normal")`
Composes: sort -> standardize -> normal_cdf -> D_statistic -> p_value
using() keys: distribution("normal"|"exponential"|...), alternative("two_sided"|"less"|"greater")
MISSING_USING_KEY: alternative (hardcoded two-sided)
MISSING_PARAMETER: distribution hardcoded to normal. No general distribution parameter.
MISSING_FLAVOR: Lilliefors (ks_test_normal_standardized exists, good),
  general K-S against arbitrary CDF

### ks_test_two_sample (Level 1 Method)
TBS: `ks_test(col_x=0, col_y=1)`
using() keys: alternative("two_sided"|"less"|"greater"), method("asymptotic"|"exact"|"auto")
MISSING_USING_KEY: alternative, method (always asymptotic approximation)

### anderson_darling (Level 1 Method)
TBS: `anderson_darling(col=0)`
Composes: sort -> standardize -> normal_cdf -> A2_formula -> p_value
using() keys: distribution("normal"|"exponential"|"logistic"|"gumbel"|"extreme1")
MISSING_USING_KEY: distribution (hardcoded to normal)
MISSING_FLAVOR: scipy supports 5+ distributions for AD test

### bootstrap_percentile (Level 1 Method)
TBS: `bootstrap(col=0, statistic="mean", n_resamples=10000)`
Composes: resample_with_replacement -> compute_statistic -> sort_bootstrap_stats -> CI
using() keys: statistic (fn pointer currently — should be a string key),
  n_resamples, alpha, seed, method("percentile"|"bca"|"basic"|"studentized")
MISSING_PARAMETER: alpha hardcoded as parameter, not using() wirable
MISSING_USING_KEY: method (only percentile CI, no BCa, basic, or studentized)
MISSING_FLAVOR: BCa bootstrap (bias-corrected accelerated), block bootstrap,
  double bootstrap, wild bootstrap, circular bootstrap
COMPOUND_PRIMITIVE: returns estimate + CI + SE all at once

### permutation_test_mean_diff (Level 1 Method)
TBS: `permutation_test(col_x=0, col_y=1, statistic="mean_diff")`
Composes: combine -> shuffle -> mean_diff -> count_extreme -> p_value
using() keys: statistic("mean_diff"|"median_diff"|"t_stat"|...), n_permutations, seed
MISSING_USING_KEY: statistic (hardcoded to mean difference)
MISSING_FLAVOR: permutation test for arbitrary statistics

### kde (Level 1 Method)
TBS: `kde(col=0)`
Composes: bandwidth_estimate -> kernel evaluation at each point
using() keys: kernel("gaussian"|"epanechnikov"|"uniform"|"triangular"|"biweight"|"cosine"),
  bandwidth, bandwidth_method("silverman"|"scott"|"sheather_jones"|"isj"|"cv")
Level 0 primitives: kernel_eval OK, silverman_bandwidth OK, scott_bandwidth OK
MISSING_FLAVOR: kernels (only 2 of ~8 standard kernels implemented),
  bandwidth methods (no Sheather-Jones plug-in, no cross-validated bandwidth, no ISJ)
MISSING_USING_KEY: bandwidth_method (only silverman default or explicit value)
MISSING_PRIMITIVE: sheather_jones_bandwidth

### kde_fft (Level 1 Method)
TBS: `kde(col=0).using(method="fft")`
Composes: binning -> FFT convolution -> kernel smoothing
using() keys: n_grid, bandwidth
OK but should be a using() variant of kde, not separate function

### ecdf (Level 0 Primitive)
TBS: `ecdf(col=0)`
Level 0: proper primitive
Family tags: statistics, nonparametric, distribution
OK

### histogram_auto (Level 1 Method)
TBS: `histogram(col=0)`
Composes: bin_count_rule -> bin_edges -> count_per_bin
using() keys: rule("sturges"|"scott"|"freedman_diaconis"|"doane"|"sqrt"|"rice")
Level 0 primitives: sturges_bins, scott_bins, freedman_diaconis_bins, doane_bins all OK
MISSING_USING_KEY: n_bins (explicit override), range
MISSING_FLAVOR: Knuth's rule, Bayesian blocks

### runs_test (Level 1 Method)
TBS: `runs_test(col=0)`
Composes: binarize -> count_runs -> normal_approx -> p_value
using() keys: cutpoint("median"|value)
MISSING_USING_KEY: cutpoint (runs_test_numeric exists but hardcoded to median)

### sign_test (Level 1 Method)
TBS: `sign_test(col=0, median0=0)`
Composes: count above/below -> binomial test
using() keys: median0, alternative("two_sided"|"less"|"greater")
MISSING_USING_KEY: alternative (hardcoded two-sided)
MISSING_FLAVOR: exact vs normal approximation

### phi_coefficient (Level 0 Primitive)
TBS: `phi(col_x=0, col_y=1)`
Level 0: proper primitive for 2x2 binary association
Family tags: statistics, correlation, categorical
OK

### point_biserial (Level 0 Primitive)
TBS: `point_biserial(binary=col0, continuous=col1)`
OK

### biserial_correlation (Level 0 Primitive)
TBS: `biserial(binary=col0, continuous=col1)`
OK. Distinct from point-biserial (assumes underlying continuous variable).

### rank_biserial (Level 0 Primitive)
TBS: `rank_biserial(col_x=0, col_y=1)`
OK

### tetrachoric (Level 0 Primitive)
TBS: `tetrachoric(table=[a,b,c,d])`
MISSING_FLAVOR: polychoric correlation (multi-category analog)
MISSING_PRIMITIVE: polychoric_correlation

### cramers_v (Level 0 Primitive)
TBS: `cramers_v(table, n_rows)`
Family tags: statistics, correlation, categorical, association
MISSING_FLAVOR: Tschuprow's T, contingency coefficient C, Goodman-Kruskal lambda/tau

### eta_squared (Level 0 Primitive)
TBS: `eta_squared(col=0, groups=col1)`
Family tags: statistics, effect_size, anova
OK

### distance_correlation (Level 0 Primitive)
TBS: `distance_correlation(col_x=0, col_y=1)`
Composes: pairwise_distances -> double_center -> dCov -> dCor formula
MISSING_COMPOSITION: entire distance matrix computation embedded inline.
  Should call pairwise_distance_matrix primitive.
Family tags: statistics, correlation, nonlinear, independence
MISSING_FLAVOR: partial distance correlation, MIC (maximal information coefficient),
  Hoeffding's D, Schweizer-Wolff sigma, Blomqvist's beta

### friedman_test (Level 1 Method)
TBS: `friedman(data, n_subjects, n_treatments)`
Composes: within-block ranking -> Q_formula -> chi2_p
using() keys: (none)
Level 0 primitives: rank OK (applied within blocks)
MISSING_FLAVOR: Iman-Davenport correction, Quade test
MISSING_USING_KEY: correction("none"|"iman_davenport")

### concordance_correlation (Level 0 Primitive)
TBS: `concordance_correlation(col_x=0, col_y=1)` (Lin's CCC)
OK

### dtw / dtw_banded (Level 0 Primitive)
TBS: `dtw(col_x=0, col_y=1)` / `dtw(col_x=0, col_y=1).using(window=10)`
using() keys: window (Sakoe-Chiba band), step_pattern, distance_metric
Level 0: proper primitives
MISSING_USING_KEY: step_pattern, distance_metric (hardcoded to absolute diff)
MISSING_FLAVOR: itakura parallelogram, multiscale DTW, derivative DTW,
  soft-DTW (differentiable)

### levenshtein (Level 0 Primitive)
TBS: `levenshtein(seq_a, seq_b)`
Family tags: string, distance, edit_distance
MISSING_FLAVOR: Damerau-Levenshtein, Jaro-Winkler, Hamming, LCS distance,
  Needleman-Wunsch, Smith-Waterman

### gutenberg_richter_fit / omori_fit / bath_law (Level 1 Methods)
TBS: `gutenberg_richter(magnitudes, m_min)` etc.
Family tags: seismology, power_law
COMPOUND_PRIMITIVE: each returns multiple values
OK for domain-specific methods, narrow use case

### sde_estimate (Level 1 Method)
TBS: `sde_estimate(prices, n_grid)`
COMPOUND_PRIMITIVE: estimates drift, diffusion, and stationary density simultaneously
Family tags: stochastic, diffusion, finance

---

## hypothesis

### one_sample_t (Level 1 Method)
TBS: `one_sample_t(col=0, mu=0)`
Composes: moments -> t_statistic -> p_value -> cohens_d -> CI
using() keys: alpha, alternative("two_sided"|"less"|"greater"), ci_level
Level 0 primitives: moments OK
MISSING_PARAMETER: CI level hardcoded at 0.95, alternative hardcoded to two-sided
MISSING_USING_KEY: alpha, alternative, ci_level
COMPOUND_PRIMITIVE: returns statistic + p + effect_size + CI all at once.
  Each is a separate computation.

### two_sample_t (Level 1 Method)
TBS: `two_sample_t(col_x=0, col_y=1)`
Composes: moments(x) + moments(y) -> pooled_variance -> t_stat -> p_value -> cohens_d -> CI
Same gaps as one_sample_t
MISSING_PARAMETER: CI level hardcoded at 0.95, alternative hardcoded to two-sided

### welch_t (Level 1 Method)
TBS: `welch_t(col_x=0, col_y=1)`
Composes: moments(x) + moments(y) -> Welch-Satterthwaite df -> t_stat -> p -> d -> CI
Same gaps as above
MISSING_PARAMETER: CI level 0.95 hardcoded

### paired_t (Level 0+ Composition)
TBS: `paired_t(col_x=0, col_y=1)` = `diff(col_x, col_y).one_sample_t(mu=0)`
Composes: differences -> one_sample_t(mu=0)
OK: properly delegates

### one_way_anova (Level 1 Method)
TBS: `anova(groups)`
Composes: per-group moments -> between_SS + within_SS -> F_ratio -> p_value
using() keys: (none)
COMPOUND_PRIMITIVE: returns F, p, eta2, partial_eta2, omega2, cohens_f all at once
MISSING_FLAVOR: two-way ANOVA, factorial ANOVA, repeated measures ANOVA,
  mixed ANOVA, ANCOVA
MISSING_PRIMITIVE: sum_of_squares_between, sum_of_squares_within as standalone

### welch_anova (Level 1 Method)
TBS: `welch_anova(groups)`
Composes: per-group moments -> Welch's F -> p_value
MISSING_PARAMETER: none (takes MomentStats), but there's a duplicate welch_anova
  that takes raw slices

### levene_test (Level 1 Method)
TBS: `levene(groups)`
Composes: center_transform -> one_way_anova_on_deviations
using() keys: center("median"|"mean"|"trimmed_mean")
Level 0: center parameter exists (LeveneCenter enum)
MISSING_USING_KEY: center is enum param, not wirable through using() bag
MISSING_FLAVOR: Brown-Forsythe (= Levene with median, which IS the default here),
  Bartlett's test, Fligner-Killeen

### chi2_goodness_of_fit (Level 1 Method)
TBS: `chi2_gof(observed, expected)`
Composes: chi2_statistic -> p_value
using() keys: (none)
Level 0: proper
MISSING_FLAVOR: G-test (log-likelihood ratio), multinomial exact test
MISSING_USING_KEY: lambda_param for power-divergence statistic family

### chi2_independence (Level 1 Method)
TBS: `chi2_independence(table, n_rows)`
Composes: expected_counts -> chi2_stat -> p_value -> effect_size(cramers_v)
using() keys: correction("yates"|"none")
MISSING_USING_KEY: correction (no Yates' correction implemented)
MISSING_FLAVOR: G-test, Fisher exact for r x c tables, exact multinomial

### one_proportion_z / two_proportion_z (Level 1 Methods)
TBS: `proportion_z(successes, n, p0)` / `proportion_z_2(s1, n1, s2, n2)`
using() keys: alternative, correction("continuity"|"none")
MISSING_USING_KEY: alternative (hardcoded two-sided), correction (hardcoded no correction)
MISSING_FLAVOR: Wilson score interval, Clopper-Pearson exact interval,
  Agresti-Coull interval

### Effect sizes: cohens_d / glass_delta / hedges_g / point_biserial_r (Level 0 Primitives)
TBS: `cohens_d(col_x=0, col_y=1)`, `hedges_g(col_x=0, col_y=1)` etc.
Level 0: proper primitives
Family tags: statistics, effect_size
MISSING_FLAVOR: Cliff's delta, CLES (common language effect size),
  eta squared (exists), omega squared (exists in ANOVA result but not standalone),
  epsilon squared

### odds_ratio / log_odds_ratio / log_odds_ratio_se (Level 0 Primitives)
TBS: `odds_ratio(table)`, `log_odds_ratio(table)`, `log_odds_ratio_se(table)`
Level 0: proper primitives
Family tags: statistics, effect_size, categorical, epidemiology
MISSING_FLAVOR: relative risk, NNT (number needed to treat), risk difference

### bonferroni / holm / benjamini_hochberg (Level 0 Primitives)
TBS: `p_adjust(p_values, method="bonferroni")` etc.
Level 0: each is a standalone fn
Family tags: statistics, multiple_comparisons
MISSING_FLAVOR: Sidak, Hommel, Hochberg, BY (Benjamini-Yekutieli),
  Bonferroni-Holm, Shaffer
MISSING_COMPOSITION: should be a single `p_adjust` with method parameter

### tukey_hsd (Level 1 Method)
TBS: `tukey_hsd(groups)`
Composes: MSE from ANOVA -> pairwise mean diffs -> q-statistic -> studentized range p
using() keys: alpha
MISSING_USING_KEY: alpha not wirable through using()
MISSING_FLAVOR: Games-Howell (unequal variances), Scheffe, Dunnett, Fisher LSD

### breusch_pagan (Level 1 Method)
TBS: `breusch_pagan(x_matrix, residuals)`
Composes: auxiliary_regression -> chi2_test
Family tags: statistics, regression_diagnostics
OK

### cooks_distance (Level 0 Primitive)
TBS: `cooks_distance(hat, residuals, mse, p)`
Level 0: proper primitive
Family tags: statistics, regression_diagnostics, outlier
MISSING_FLAVOR: DFFITS, DFBETAS, covariance ratio

### wls (Level 1 Method)
TBS: `wls(x, y, weights)`
Composes: weight_transform -> ols on transformed
Family tags: statistics, regression
NOTE: exists in BOTH hypothesis.rs AND multivariate.rs — duplicate

### bayes_factor_t_one_sample / bayes_factor_correlation (Level 1 Methods)
TBS: `bayes_factor(test="t", t_stat=2.5, n=30)` etc.
Composes: numerical integration (adaptive quadrature) -> BF10 + interpretation
using() keys: r (Cauchy prior scale, default sqrt(2)/2), prior("cauchy"|"normal")
MISSING_USING_KEY: r is a direct parameter but not wirable through using()
MISSING_FLAVOR: Bayes factor for ANOVA, chi-square, regression

### mediation (Level 2 Pipeline)
TBS: `mediation(x=col0, m=col1, y=col2)`
Composes: ols(y~x) -> ols(m~x) -> ols(y~x+m) -> Sobel test
using() keys: bootstrap(true|false), n_bootstrap, alpha
MISSING_USING_KEY: bootstrap option (only Sobel z-test, no bootstrap CI)
MISSING_COMPOSITION: uses private ols_simple/ols_two_predictor, not public OLS primitive
COMPOUND_PRIMITIVE: returns total, direct, indirect, Sobel z, p, proportion all at once
MISSING_FLAVOR: bootstrap mediation (Baron-Kenny with Preacher-Hayes bootstrap)

### moderation (Level 2 Pipeline)
TBS: `moderation(x=col0, z=col1, y=col2)`
Composes: interaction_term -> ols(y ~ x + z + xz) -> simple slopes
using() keys: alpha, centering("mean"|"none")
MISSING_USING_KEY: centering (variables not centered before interaction)
MISSING_COMPOSITION: uses qr_solve directly, not public OLS primitive
COMPOUND_PRIMITIVE: returns coefficients, se, t, p, simple slopes all at once
MISSING_FLAVOR: moderated mediation, polynomial moderation

### logistic_regression (Level 1 Method)
TBS: `logistic_regression(x, y)`
Composes: IRLS iterations -> coefficients -> std_errors -> Wald z -> p
using() keys: max_iter, tol, penalty("none"|"l1"|"l2"|"elastic_net"), lambda
MISSING_USING_KEY: penalty, lambda (no regularization)
MISSING_COMPOSITION: uses private IRLS, not public optimization primitives
COMPOUND_PRIMITIVE: returns coefficients, se, z, p, deviance, AIC all at once
MISSING_FLAVOR: probit regression, multinomial logistic, ordinal logistic,
  conditional logistic, Firth's penalized logistic

### glm_fit (Level 1 Method)
TBS: `glm(x, y, family="poisson")`
Composes: family-specific link -> IRLS -> coefficients -> deviance
using() keys: family("gaussian"|"poisson"|"binomial"|"gamma"|"inverse_gaussian"),
  link("identity"|"log"|"logit"|"inverse"|"probit"|"cloglog"),
  max_iter, tol
MISSING_USING_KEY: link function is hardcoded per family (should be overridable)
COMPOUND_PRIMITIVE: returns coefficients, se, z, p, deviance, AIC all at once

### two_group_comparison (Level 2 Pipeline)
TBS: `two_group_comparison(group1=col0, group2=col1)`
Composes: normality_test(shapiro_wilk|dagostino) -> levene -> [t_test|welch_t|mann_whitney]
  -> cohens_d -> hedges_g -> CI
using() keys: alpha, normality_test("shapiro"|"dagostino"|"lilliefors"),
  force_parametric(bool), force_nonparametric(bool)
Level 0 primitives: all subcomponents exist as standalone
MISSING_USING_KEY: alpha partially wired (function param, not using()),
  normality_test (hardcoded n < 5000 switch)
MISSING_PARAMETER: n threshold for normality test switch hardcoded at 5000
OK: this IS a Level 2 pipeline. The composition is correct. But
  auto-detection logic (normality -> variance -> test choice) is all inline.

### fisher_exact (Level 1 Method)
TBS: `fisher_exact(table=[a,b,c,d])`
using() keys: alternative("two_sided"|"less"|"greater")
MISSING_USING_KEY: alternative (hardcoded two-sided)

### mcnemar (Level 1 Method)
TBS: `mcnemar(table=[a,b,c,d])`
using() keys: continuity(true|false)
Level 0: continuity is a direct parameter
MISSING_USING_KEY: continuity not wirable through using()

### cochran_q (Level 1 Method)
TBS: `cochran_q(data, n_subjects, n_treatments)`
Family tags: statistics, nonparametric
OK

### Power analysis: power_one_sample_t / power_two_sample_t / sample_size_* (Level 0 Primitives)
TBS: `power(test="one_sample_t", effect_size=0.5, n=30, alpha=0.05)`
TBS: `sample_size(test="one_sample_t", effect_size=0.5, power=0.8, alpha=0.05)`
using() keys: two_sided(bool)
MISSING_USING_KEY: two_sided not wirable through using()
MISSING_FLAVOR: power for chi-square, proportions, correlation (power_correlation exists),
  McNemar, Wilcoxon, logistic regression, mixed models

---

## time_series

### levinson_durbin (Level 0 Primitive)
TBS: `levinson_durbin(acf_values)`
Level 0: proper primitive (returns AR coefficients, reflection coefficients, error variance)
Family tags: linear_algebra, time_series, autoregressive
COMPOUND_PRIMITIVE: returns (coefficients, kappas, sigma2) — but this IS
  the canonical Levinson-Durbin output. Acceptable.

### delay_embed (Level 0 Primitive)
TBS: `delay_embed(col=0, dim=3, tau=1)`
Level 0: proper primitive
Family tags: dynamical_systems, time_series, phase_space
OK

### acf (Level 0 Primitive)
TBS: `acf(col=0, max_lag=20)`
using() keys: method("biased"|"unbiased"|"fft"), demean(true|false)
Level 0: proper primitive
MISSING_USING_KEY: method (always biased estimator), demean (always demeaned)
MISSING_FLAVOR: FFT-based ACF (much faster for large n), unbiased ACF (divides by n-k)

### pacf (Level 0+ Composition)
TBS: `pacf(col=0, max_lag=20)`
Composes: acf -> levinson_durbin -> extract reflection coefficients
using() keys: method("levinson_durbin"|"ols"|"burg")
Level 0 primitives: acf OK, levinson_durbin OK
MISSING_USING_KEY: method (hardcoded to Levinson-Durbin)
MISSING_FLAVOR: OLS-based PACF

### ar_fit (Level 1 Method)
TBS: `ar_fit(col=0, order=4)`
Composes: moments -> center -> acf -> levinson_durbin -> AIC
using() keys: order, method("yule_walker"|"burg"|"ols"|"mle"), ic("aic"|"bic"|"hqic")
Level 0 primitives: moments OK, acf (recomputed inline), levinson_durbin OK
COMPOUND_PRIMITIVE: returns coefficients + sigma2 + AIC. Each sub-computation
  (ACF, Levinson-Durbin, AIC) should be independently callable.
MISSING_COMPOSITION: ACF computed inline (private loop), should call public acf()
MISSING_USING_KEY: method (hardcoded Yule-Walker), ic (hardcoded AIC)
Note: ar_burg_fit exists as SEPARATE function. Should be using(method="burg") on ar_fit.

### ar_burg_fit (Level 1 Method)
TBS: `ar_fit(col=0, order=4).using(method="burg")`
Should be a using() variant of ar_fit, not a separate function.
MISSING_COMPOSITION: exists as standalone rather than using() variant

### ar_predict (Level 0+ Composition)
TBS: `ar_predict(col=0, model, horizon=5)`
OK

### ar_psd / ar_psd_at (Level 0+ Composition)
TBS: `ar_psd(model, n_freqs=256)` / `ar_psd_at(model, freq=0.1)`
Composes: AR coefficients -> spectral density via transfer function
Family tags: spectral, time_series, parametric
OK

### arma_fit (Level 1 Method)
TBS: `arma_fit(col=0, p=2, q=1)`
Composes: center -> ar_fit(init) -> L-BFGS optimization of CSS -> coefficients
using() keys: p, q, max_iter, method("css"|"mle"|"css_mle"), optimizer("lbfgs"|"nelder_mead")
MISSING_USING_KEY: method (hardcoded CSS), optimizer (hardcoded L-BFGS)
MISSING_COMPOSITION: uses gradient via finite differences (inline), should use
  public derivative primitive. ar_fit used for initialization but not via using().
COMPOUND_PRIMITIVE: returns ar + ma + intercept + sigma2 + css + aic + bic + iterations + residuals

### arima_fit (Level 1 Method)
TBS: `arima_fit(col=0, p=1, d=1, q=1)`
Composes: difference(d times) -> arma_fit(p, q)
using() keys: p, d, q, max_iter
Level 0 primitives: difference OK, arma_fit OK
OK: properly composed

### arima_forecast (Level 0+ Composition)
TBS: `arima_forecast(model, horizon=10)`
Composes: arma_forecast on differenced -> undifference
OK

### auto_arima (Level 2 Pipeline)
TBS: `auto_arima(col=0, max_p=5, max_d=2, max_q=5)`
Composes: grid search over (p,d,q) -> arima_fit each -> select by AIC
using() keys: max_p, max_d, max_q, max_iter, ic("aic"|"bic"|"aicc"),
  stepwise(true|false), seasonal(true|false)
MISSING_USING_KEY: ic (hardcoded AIC), stepwise (brute-force grid search)
MISSING_FLAVOR: stepwise search (R's auto.arima uses Hyndman-Khandakar stepwise),
  SARIMA (seasonal), AICc
MISSING_PARAMETER: no stationarity/invertibility checking

### difference / undifference (Level 0 Primitives)
TBS: `difference(col=0, d=1)` / `undifference(col=0, initial=...)`
Level 0: proper primitives
Family tags: time_series, transform
MISSING_FLAVOR: seasonal differencing (period parameter)

### cusum_mean (Level 0 Primitive)
TBS: `cusum(col=0)`
Level 0: proper primitive
Family tags: time_series, changepoint, quality_control
OK

### cusum_binary_segmentation (Level 1 Method)
TBS: `changepoints(col=0, method="cusum_binary_segmentation")`
using() keys: threshold, min_segment_size, max_changepoints
MISSING_USING_KEY: parameters are direct args, not using() wirable

### pelt (Level 1 Method)
TBS: `changepoints(col=0, method="pelt")`
Composes: prefix_sums -> dynamic programming -> optimal partition
using() keys: min_seg, penalty, cost("normal_mean"|"normal_meanvar"|"poisson"|"exponential")
MISSING_USING_KEY: cost function (hardcoded to normal mean change)
MISSING_FLAVOR: PELT with different cost functions, kernel change point detection

### bocpd (Level 1 Method)
TBS: `changepoints(col=0, method="bocpd")`
Composes: online_bayesian -> run_length_posterior -> changepoint_extraction
using() keys: max_run, hazard, threshold, model("gaussian"|"student_t")
MISSING_USING_KEY: model (hardcoded Gaussian conjugate model)
MISSING_FLAVOR: BOCPD with different observation models

### simple_exponential_smoothing (Level 1 Method)
TBS: `ses(col=0, alpha=0.3)`
using() keys: alpha, optimize_alpha(true|false)
MISSING_USING_KEY: optimize_alpha (alpha must be provided, no auto-optimization)
MISSING_FLAVOR: Holt-Winters (additive/multiplicative seasonal),
  damped trend method, Croston's method

### holt_linear (Level 1 Method)
TBS: `holt(col=0, alpha=0.3, beta=0.1, horizon=10)`
using() keys: alpha, beta, damped(true|false), phi (damping factor)
MISSING_USING_KEY: damped, phi
MISSING_FLAVOR: damped trend, Holt-Winters triple smoothing

### adf_test (Level 1 Method)
TBS: `adf_test(col=0, n_lags=4)`
Composes: difference -> OLS regression -> t-statistic on gamma -> MacKinnon critical values
using() keys: n_lags, regression("c"|"ct"|"ctt"|"n"), autolag("aic"|"bic"|"t-stat"|"none")
Level 0 primitives: difference OK, ols used inline
MISSING_USING_KEY: regression type (hardcoded to "constant" model),
  autolag (no automatic lag selection)
MISSING_COMPOSITION: OLS done inline, not via public ols primitive
MISSING_PARAMETER: no p-value returned (only critical values)
MISSING_FLAVOR: DF-GLS, Elliott-Rothenberg-Stock

### kpss_test (Level 1 Method)
TBS: `kpss_test(col=0)`
using() keys: trend(true|false), n_lags, regression("c"|"ct")
MISSING_USING_KEY: n_lags auto-selection rule hardcoded to (4*(n/100)^0.25)
MISSING_COMPOSITION: Newey-West LRV computed inline, should call newey_west_lrv
MISSING_PARAMETER: critical values hardcoded to asymptotic (KPSS 1992 Table 1),
  no finite-sample correction

### pp_test (Level 1 Method)
TBS: `pp_test(col=0)`
Composes: DF regression -> Newey-West LRV -> Phillips-Perron correction
using() keys: n_lags, regression("c"|"ct")
MISSING_USING_KEY: regression (hardcoded "constant")
MISSING_COMPOSITION: Newey-West LRV done inline. There IS a newey_west_lrv fn,
  and a separate phillips_perron_test fn that may duplicate.
Note: pp_test and phillips_perron_test BOTH exist (L1342 and L2207) — possible duplicate

### variance_ratio_test (Level 1 Method)
TBS: `variance_ratio(col=0, q=2)`
Composes: variance at lag q / variance at lag 1
using() keys: q, overlap(true|false), heteroskedasticity_robust(true|false)
MISSING_USING_KEY: overlap, robust correction
MISSING_FLAVOR: multiple variance ratio test (Chow-Denning), wild bootstrap VR

### ljung_box / box_pierce (Level 1 Methods)
TBS: `ljung_box(col=0, n_lags=10)` / `box_pierce(col=0, n_lags=10)`
using() keys: n_lags, fitted_params
Family tags: time_series, diagnostics
OK

### durbin_watson (Level 0 Primitive)
TBS: `durbin_watson(residuals)`
Level 0: proper primitive
Family tags: time_series, regression_diagnostics
OK

### breusch_godfrey (Level 1 Method)
TBS: `breusch_godfrey(x, y, residuals, n_lags)`
Composes: auxiliary_regression -> chi2 test
Family tags: time_series, regression_diagnostics
OK

### zivot_andrews_test (Level 1 Method)
TBS: `zivot_andrews(col=0)`
Composes: grid search over break dates -> ADF at each break -> minimum t-stat
using() keys: model("intercept"|"trend"|"both"), trim(0.15)
MISSING_USING_KEY: model and trim not wirable through using()
MISSING_FLAVOR: Bai-Perron multiple structural breaks

### stl_decompose (Level 1 Method)
TBS: `stl(col=0, period=12)`
Composes: iterative LOESS smoothing -> trend + seasonal + remainder
using() keys: period, robust(true|false), n_outer, n_inner, seasonal_window,
  trend_window, seasonal_deg, trend_deg
MISSING_USING_KEY: only period and robust exposed; many STL parameters hardcoded
MISSING_FLAVOR: X-11, X-13, SEATS, MSTL (multiple seasonalities),
  classical decomposition (additive/multiplicative)

### Spectral time series features (Level 0 Primitives)
spectral_flatness, spectral_rolloff, spectral_centroid, spectral_bandwidth,
spectral_skewness, spectral_kurtosis, spectral_crest, spectral_slope,
spectral_fwhm, spectral_q_factor, spectral_flux, spectral_decrease,
spectral_contrast, dominant_frequency, peak_to_average_power_ratio,
spectral_peak_count
All Level 0: proper primitives operating on PSD
Family tags: spectral, signal_processing, audio
using() keys: most take (freqs, psd) directly
OK: good set of spectral features. Each is genuinely atomic.
MISSING_USING_KEY: spectral_rolloff has pct param but not using() wirable

### newey_west_lrv (Level 0 Primitive)
TBS: `newey_west_lrv(residuals, n_lags)`
Level 0: proper primitive
Family tags: time_series, econometrics, variance_estimation
OK

---

## linear_algebra

### Mat (Level 0 Primitive type)
Constructors: zeros, eye, from_vec, from_rows, col_vec, diag
Methods: get, set, t, trace, norm_fro, norm_inf, norm_1, diagonal, submat
All Level 0 primitives. Proper building blocks.

### mat_mul / mat_add / mat_sub / mat_scale (Level 0 Primitives)
TBS: `mat_mul(A, B)`, `mat_add(A, B)` etc.
Level 0: proper primitives
Family tags: linear_algebra, matrix_operations
OK

### dot / vec_norm / outer (Level 0 Primitives)
TBS: `dot(col_x=0, col_y=1)`, `norm(col=0)`, `outer(col_x=0, col_y=1)`
Level 0: proper primitives
Family tags: linear_algebra, vector_operations
MISSING_FLAVOR: norm — only L2 norm. Should support L1, Linf, Lp via using(p=...)

### lu / lu_solve (Level 0 Primitives)
TBS: `lu(matrix)`, `lu_solve(lu_result, b)`
Level 0: proper primitives
Family tags: linear_algebra, factorization
OK

### det / log_det (Level 0 Primitives)
TBS: `det(matrix)`, `log_det(matrix)`
Composes: lu -> product of diagonal -> sign
Level 0: proper primitives (internally call lu)
OK: composition is correct

### inv (Level 0 Primitive)
TBS: `inv(matrix)`
Composes: lu -> lu_solve for each column
Level 0: proper primitive
MISSING_FLAVOR: pseudoinverse (pinv exists separately — OK)

### cholesky / cholesky_solve (Level 0 Primitives)
TBS: `cholesky(matrix)`, `cholesky_solve(L, b)`
Level 0: proper primitives
Family tags: linear_algebra, factorization, positive_definite
OK

### forward_solve / back_solve_transpose (Level 0 Primitives)
TBS: `forward_solve(L, b)`, `back_solve(L, b)`
Level 0: proper primitives
OK

### qr / qr_solve (Level 0 Primitives)
TBS: `qr(matrix)`, `qr_solve(matrix, b)`
Level 0: proper primitives
Family tags: linear_algebra, factorization, least_squares
MISSING_FLAVOR: pivoted QR (column pivoting for rank-revealing)

### svd (Level 0 Primitive)
TBS: `svd(matrix)`
Level 0: proper primitive (Golub-Kahan bidiagonalization)
Family tags: linear_algebra, factorization, decomposition
MISSING_FLAVOR: truncated SVD (only compute first k), randomized SVD,
  economy SVD (already returns full)
COMPOUND_PRIMITIVE: returns U + sigma + Vt. But this IS the canonical SVD output.

### pinv (Level 0+ Composition)
TBS: `pinv(matrix)`
Composes: svd -> filter small singular values -> reconstruct
using() keys: rcond (threshold for zero singular values)
OK

### sym_eigen (Level 0 Primitive)
TBS: `eigendecomposition(matrix)` (symmetric)
Level 0: proper primitive (QR iteration)
Family tags: linear_algebra, eigenvalues, spectral
MISSING_FLAVOR: non-symmetric eigendecomposition, generalized eigendecomposition,
  sparse eigendecomposition (Lanczos, Arnoldi)
MISSING_PRIMITIVE: non-symmetric eigen, generalized eigen

### power_iteration (Level 0 Primitive)
TBS: `power_iteration(matrix, max_iter, tol)`
Level 0: proper primitive
Family tags: linear_algebra, eigenvalues, iterative
MISSING_USING_KEY: max_iter, tol not wirable through using()

### cond / rank (Level 0+ Compositions)
TBS: `cond(matrix)`, `matrix_rank(matrix, tol)`
Composes: svd -> sigma_max/sigma_min (for cond), count sigma > tol (for rank)
OK

### solve / solve_spd / lstsq (Level 0 Primitives / Compositions)
TBS: `solve(A, b)`, `solve_spd(A, b)`, `lstsq(A, b)`
solve: lu_solve if square
solve_spd: cholesky_solve
lstsq: qr_solve
OK: proper routing to primitives

### simple_linear_regression (Level 1 Method)
TBS: `slr(col_x=0, col_y=1)` or `regression(col_x=0, col_y=1)`
Composes: moments(x,y) -> slope + intercept + r_squared + se_slope
COMPOUND_PRIMITIVE: returns slope + intercept + r2 + se_slope + se_intercept + residual_se.
  Each is independently callable.
MISSING_PRIMITIVE: r_squared as standalone (exists as part of regression result)

### ols_slope (Level 0 Primitive)
TBS: `ols_slope(col_x=0, col_y=1)`
Level 0: proper primitive (just the slope)
OK

### ols_normal_equations (Level 0 Primitive)
TBS: `ols(x_matrix, y)`
Level 0: proper primitive
MISSING_FLAVOR: should be unified with qr_solve path. Two OLS implementations.

### sigmoid (Level 0 Primitive)
TBS: `sigmoid(x)`
Level 0: proper primitive (logistic function)
Family tags: activation, neural, special_functions
Note: also exists in special_functions.rs as `logistic()`. Documented as dedup candidate.

### effective_rank_from_sv (Level 0 Primitive)
TBS: `effective_rank(singular_values)`
Composes: entropy of normalized singular values
Family tags: linear_algebra, dimensionality
OK

### solve_tridiagonal / solve_tridiagonal_scan (Level 0 Primitives)
TBS: `solve_tridiagonal(a, b, c, d)`
Level 0: proper primitive (Thomas algorithm and parallel scan version)
Family tags: linear_algebra, sparse, tridiagonal
OK: two methods (serial Thomas, parallel scan). Should be using(method=...) variants.

---

## information_theory

### probabilities (Level 0 Primitive)
TBS: `probabilities(counts)`
Level 0: normalize counts to probabilities
OK

### shannon_entropy / shannon_entropy_from_counts (Level 0 Primitive)
TBS: `entropy(col=0)` or `entropy(probs)` or `entropy(counts)`
using() keys: base("e"|"2"|"10"), method("plugin"|"miller_madow"|"grassberger"|"chao_shen")
Level 0: proper primitive
MISSING_USING_KEY: base (hardcoded to natural log), method (only plugin estimator)
MISSING_FLAVOR: Grassberger entropy estimator, Chao-Shen estimator,
  NSB (Nemenman-Shafee-Bialek), jackknife entropy,
  KDE-based differential entropy

### renyi_entropy (Level 0 Primitive)
TBS: `renyi_entropy(probs, alpha=2)`
using() keys: alpha
Level 0: proper primitive with all special cases (0, 1, 2, inf)
Family tags: information_theory, entropy, generalized
OK

### tsallis_entropy (Level 0 Primitive)
TBS: `tsallis_entropy(probs, q=2)`
using() keys: q
Level 0: proper primitive
OK

### kl_divergence (Level 0 Primitive)
TBS: `kl_divergence(p, q)`
Level 0: proper primitive
Family tags: information_theory, divergence
Note: ALSO exists in distributional_distances.rs with UsingBag support
MISSING_FLAVOR: symmetrized KL (already have JS), reverse KL,
  alpha-divergence, f-divergence framework

### js_divergence (Level 0+ Composition)
TBS: `js_divergence(p, q)`
Composes: m = (p+q)/2 -> 0.5*kl(p,m) + 0.5*kl(q,m)
Level 0: proper primitive
Note: also in distributional_distances.rs with UsingBag
OK

### cross_entropy (Level 0 Primitive)
TBS: `cross_entropy(p, q)`
OK

### mutual_information (Level 0 Primitive)
TBS: `mutual_information(contingency_table, nx, ny)`
using() keys: method("plugin"|"miller_madow"|"ksg"|"gaussian"), base
Level 0: proper primitive
MISSING_USING_KEY: method, base
MISSING_FLAVOR: KSG estimator (k-nearest neighbor based, for continuous variables)

### normalized_mutual_information (Level 0+ Composition)
TBS: `nmi(contingency, nx, ny, method="arithmetic")`
Composes: MI + H(X) + H(Y) -> normalize
using() keys: method("arithmetic"|"geometric"|"min"|"max")
Level 0: method parameter exists as direct arg
MISSING_USING_KEY: method not wirable through using()
OK

### conditional_entropy / variation_of_information (Level 0+ Compositions)
TBS: `conditional_entropy(contingency, nx, ny)`
Composes: H(X,Y) - H(Y)
OK

### mutual_info_score / normalized_mutual_info_score / adjusted_mutual_info_score (Level 0 Primitives)
TBS: `ami(labels_true, labels_pred)` etc.
Family tags: information_theory, clustering, evaluation
OK: sklearn-compatible interface

### entropy_histogram (Level 1 Method)
TBS: `differential_entropy(col=0, n_bins=50)`
Composes: histogram -> probabilities -> Shannon entropy + log(bin_width)
using() keys: n_bins, method("histogram"|"knn"|"kernel")
MISSING_USING_KEY: method (only histogram method)
MISSING_FLAVOR: k-NN entropy estimator (Kozachenko-Leonenko),
  kernel density entropy estimator

### mutual_info_miller_madow (Level 1 Method)
TBS: `mi_corrected(contingency, nx, ny)`
Composes: MI -> Miller-Madow correction -> Gaussian MI -> nonlinear excess
COMPOUND_PRIMITIVE: returns (mi_corrected, nonlinear_excess, mi_normalized) —
  three distinct quantities
OK as Level 1 but each return value is independently meaningful

### fisher_information_histogram (Level 1 Method)
TBS: `fisher_information(col=0, n_bins=50)`
Composes: histogram -> density gradient -> Fisher info formula
COMPOUND_PRIMITIVE: returns (fisher_info, fisher_distance, gradient_norm)
MISSING_PARAMETER: Laplace smoothing constant hardcoded at 0.5

### transfer_entropy (Level 1 Method)
TBS: `transfer_entropy(col_x=0, col_y=1, n_bins=10)`
Composes: discretize -> joint histograms -> conditional MI
using() keys: n_bins, k (history length), method("binning"|"knn")
MISSING_USING_KEY: k (lag/history length), method
MISSING_FLAVOR: partial transfer entropy, conditional transfer entropy,
  KSG-based transfer entropy

### tfidf (Level 1 Method)
TBS: `tfidf(documents)`
Composes: term_frequency -> inverse_document_frequency -> multiply
Family tags: information_theory, nlp, text
MISSING_FLAVOR: BM25 (modern alternative), sublinear TF, various IDF variants

### cosine_similarity / cosine_similarity_matrix (Level 0 Primitives)
TBS: `cosine_similarity(a, b)` / `cosine_matrix(data)`
Level 0: proper primitives
Family tags: similarity, distance, vector
OK

---

## multivariate

### covariance_matrix (Level 0 Primitive)
TBS: `cov_matrix(data)`
using() keys: ddof(0|1), method("standard"|"robust"|"shrinkage")
Level 0: proper primitive, TamSession shareable
Family tags: statistics, multivariate, covariance
MISSING_USING_KEY: method (no robust or shrinkage covariance)
MISSING_FLAVOR: Ledoit-Wolf shrinkage, Oracle Approximating Shrinkage,
  robust covariance (MCD — exists in robust.rs but not via using()),
  Graphical Lasso (sparse precision matrix)
MISSING_PRIMITIVE: shrinkage_covariance, sparse_precision

### col_means (Level 0 Primitive)
TBS: `col_means(data)`
OK

### sscp_matrices (Level 0 Primitive)
TBS: `sscp(data, groups)`
Family tags: statistics, multivariate, anova
COMPOUND_PRIMITIVE: returns (between, within, group_means, group_sizes, total_mean)

### hotelling_one_sample / hotelling_two_sample (Level 1 Methods)
TBS: `hotelling_t2(data, mu0)` / `hotelling_t2(data1, data2)`
Composes: covariance_matrix -> Mahalanobis form -> F transform -> p_value
COMPOUND_PRIMITIVE: returns T2, F, df1, df2, p simultaneously
MISSING_FLAVOR: paired Hotelling T-squared

### manova (Level 1 Method)
TBS: `manova(data, groups)`
Composes: sscp -> eigendecomposition of E^{-1}H -> Wilks/Pillai/Hotelling-Lawley/Roy
COMPOUND_PRIMITIVE: returns Wilks, Pillai, Hotelling-Lawley, Roy all at once
MISSING_USING_KEY: which test statistic to use (returns all four)
MISSING_FLAVOR: MANCOVA

### lda (Level 1 Method)
TBS: `lda(data, groups)`
Composes: sscp -> eigendecomposition -> discriminant functions
using() keys: n_components, method("lda"|"qda"|"regularized"), shrinkage
MISSING_USING_KEY: method (no QDA, no regularized LDA)
COMPOUND_PRIMITIVE: returns coefficients + scalings + prior + transform + predict
MISSING_FLAVOR: QDA, regularized LDA (shrinkage), flexible DA

### cca (Level 1 Method)
TBS: `cca(data_x, data_y)`
Composes: covariance matrices -> generalized eigendecomposition
COMPOUND_PRIMITIVE: returns correlations + x_weights + y_weights + x_loadings + y_loadings
MISSING_FLAVOR: kernel CCA, sparse CCA, partial least squares

### mardia_normality (Level 1 Method)
TBS: `mardia_test(data)`
Composes: Mahalanobis distances -> skewness + kurtosis measures -> chi2/z p-values
COMPOUND_PRIMITIVE: returns skewness, kurtosis, each with stat + p

### vif (Level 0+ Composition)
TBS: `vif(x_matrix)`
Composes: for each column, regress on others, VIF = 1/(1-R^2)
Level 0 at concept level, but computed as n regressions
Family tags: regression_diagnostics, multivariate
OK

### mahalanobis_distances (Level 0 Primitive)
TBS: `mahalanobis(data)`
Composes: covariance_matrix -> inv -> d_i = sqrt((x_i - mu)' * S^{-1} * (x_i - mu))
using() keys: method("standard"|"robust_mcd"), cov_estimator("standard"|"mcd")
MISSING_USING_KEY: cov_estimator (no option for robust covariance)
MISSING_COMPOSITION: should share covariance matrix via TamSession

### ridge / lasso / elastic_net (Level 1 Methods)
TBS: `ridge(x, y, lambda=1)`, `lasso(x, y, lambda=1)`, `elastic_net(x, y, lambda=1, alpha=0.5)`
Composes: (ridge: (X'X + λI)^{-1} X'y), (lasso/elastic_net: coordinate descent)
using() keys: lambda, alpha (for elastic_net), max_iter, tol, standardize(true|false)
MISSING_USING_KEY: standardize (no automatic standardization),
  intercept (always includes intercept — should be optional)
MISSING_PARAMETER: cross-validation for lambda selection
MISSING_FLAVOR: group lasso, adaptive lasso, SCAD, MCP, Bayesian ridge
MISSING_PRIMITIVE: cross_validated_lambda (not implemented)
COMPOUND_PRIMITIVE: each returns coefficients + residuals + fitted + mse

### wls (duplicate)
Exists in BOTH hypothesis.rs AND multivariate.rs

---

## complexity

### sample_entropy (Level 0 Primitive)
TBS: `sample_entropy(col=0, m=2, r=0.2)`
using() keys: m, r, distance("chebyshev"|"euclidean"), normalize_r(true|false)
Level 0: proper primitive
Family tags: complexity, entropy, nonlinear
MISSING_USING_KEY: distance metric (hardcoded Chebyshev/L-inf),
  normalize_r (r is absolute, not relative to std)
MISSING_PARAMETER: r is absolute value. R/Python typically express r as r*std(data).
MISSING_FLAVOR: multiscale sample entropy, fuzzy entropy, conditional entropy (spectral),
  distribution entropy, bubble entropy, dispersion entropy

### approx_entropy (Level 0 Primitive)
TBS: `approx_entropy(col=0, m=2, r=0.2)`
Same gaps as sample_entropy regarding r normalization and distance metric

### permutation_entropy (Level 0 Primitive)
TBS: `permutation_entropy(col=0, m=3, tau=1)`
using() keys: m, tau, normalize(true|false)
Level 0: proper primitive
MISSING_USING_KEY: normalize (exists as separate fn normalized_permutation_entropy)
MISSING_FLAVOR: weighted permutation entropy, conditional permutation entropy,
  multivariate permutation entropy, multiscale permutation entropy

### hurst_rs (Level 0 Primitive)
TBS: `hurst(col=0, method="rs")`
using() keys: method("rs"|"dfa"|"wavelet")
Level 0: proper primitive (R/S method)
MISSING_USING_KEY: method (DFA gives Hurst via alpha, but not unified)
MISSING_FLAVOR: Whittle estimator, periodogram estimator

### dfa (Level 0 Primitive)
TBS: `dfa(col=0, min_box=4, max_box=n/4)`
Composes: cumulative_sum -> box_partition -> linear_detrend_per_box -> log-log regression
using() keys: min_box, max_box, detrend_order(1|2|3), scale_ratio(1.3)
Level 0: proper primitive
MISSING_USING_KEY: detrend_order (hardcoded to linear), scale_ratio (hardcoded 1.3)
MISSING_COMPOSITION: uses inline ols_slope, should call public primitive
MISSING_FLAVOR: DFA2 (quadratic detrending), DFA3, backward DFA, sign DFA

### higuchi_fd (Level 0 Primitive)
TBS: `higuchi_fd(col=0, k_max=10)`
using() keys: k_max
Level 0: proper primitive
OK

### lempel_ziv_complexity (Level 0 Primitive)
TBS: `lempel_ziv(col=0)`
using() keys: binarize_method("median"|"mean"|"threshold"), threshold
MISSING_USING_KEY: binarization method (hardcoded to median)
MISSING_FLAVOR: modified LZ (LZ76, LZ78), compression ratio

### correlation_dimension (Level 1 Method)
TBS: `correlation_dimension(col=0, m=3, tau=1)`
Composes: delay_embed -> pairwise_distances -> count_within_r -> log-log slope
using() keys: m, tau, r_range, n_r_values
MISSING_USING_KEY: r_range, n_r_values
MISSING_COMPOSITION: delay_embed called inline, not via public primitive

### largest_lyapunov (Level 1 Method)
TBS: `lyapunov(col=0, m=3, tau=1, dt=1)`
Composes: delay_embed -> nearest_neighbors -> divergence_rate
using() keys: m, tau, dt, method("rosenstein"|"kantz"|"wolf")
MISSING_USING_KEY: method (hardcoded to Rosenstein)
MISSING_FLAVOR: Kantz method, Wolf method, full Lyapunov spectrum estimation

### lyapunov_spectrum (Level 1 Method)
TBS: `lyapunov_spectrum(col=0, m=3, tau=1, dt=1)`
MISSING_FLAVOR: QR-based Lyapunov spectrum (more stable), Benettin method

### rqa (Level 1 Method)
TBS: `rqa(col=0, m=3, tau=1, epsilon=0.1, lmin=2)`
Composes: delay_embed -> distance_matrix -> recurrence_matrix -> line_statistics
using() keys: m, tau, epsilon, lmin, norm("euclidean"|"maximum"|"manhattan"),
  theiler_window, auto_epsilon("percentile"|"fan"|"fixed")
COMPOUND_PRIMITIVE: returns rr + det + lam + entr + lmax + l_avg + tt all at once.
  Each is independently meaningful.
MISSING_USING_KEY: norm (hardcoded Euclidean), theiler_window (no exclusion),
  auto_epsilon
MISSING_COMPOSITION: delay_embed done inline, distance matrix done inline,
  should reuse public primitives

### mfdfa (Level 1 Method)
TBS: `mfdfa(col=0, q_values, min_seg, max_seg)`
Composes: cumulative_sum -> segment -> detrend -> fluctuation per q -> h(q) -> tau(q) -> D(alpha)
COMPOUND_PRIMITIVE: returns h_q + tau_q + alpha + f_alpha
MISSING_USING_KEY: detrend_order (hardcoded to linear)

### ccm (Level 1 Method)
TBS: `ccm(col_x=0, col_y=1, embed_dim=3, tau=1, k=5)`
Composes: delay_embed -> nearest_neighbors -> cross_map -> correlation
using() keys: embed_dim, tau, k, convergence_test(true|false)
MISSING_USING_KEY: convergence_test
Family tags: causality, dynamical_systems, nonlinear
MISSING_FLAVOR: partial CCM, multivariate CCM

---

## special_functions

### Distribution functions (Level 0 Primitives)
Each distribution has: pdf/pmf, cdf, quantile (inverse CDF), sometimes sf (survival fn)

Normal: normal_cdf, normal_sf, normal_quantile OK
t: t_cdf, t_quantile OK
F: f_cdf, f_quantile OK
Chi-squared: chi2_cdf, chi2_sf, chi2_quantile OK
Weibull: weibull_cdf, weibull_pdf, weibull_quantile OK
Pareto: pareto_cdf, pareto_pdf, pareto_quantile OK
Exponential: exponential_cdf, exponential_pdf, exponential_quantile OK
Lognormal: lognormal_cdf, lognormal_pdf, lognormal_quantile OK
Beta: beta_pdf, beta_cdf OK
Gamma: gamma_pdf, gamma_cdf OK
Poisson: poisson_pmf, poisson_cdf OK
Binomial: binomial_pmf, binomial_cdf OK
Neg binomial: neg_binomial_pmf, neg_binomial_cdf OK
Cauchy: cauchy_cdf, cauchy_pdf, cauchy_quantile OK

MISSING_FLAVOR (distributions without implementations):
  - Hypergeometric (pmf, cdf)
  - Geometric (pmf, cdf — sampling exists but not pmf/cdf)
  - Multinomial (pmf)
  - Dirichlet (pdf)
  - Multivariate normal (pdf, cdf)
  - Multivariate t (pdf)
  - Inverse Gaussian / Wald (pdf, cdf)
  - Rayleigh (pdf, cdf)
  - Rice (pdf, cdf)
  - Gumbel / extreme value (pdf, cdf, quantile)
  - Logistic distribution (pdf exists as `logistic`, but not cdf/quantile)
  - Laplace / double exponential
  - Levy
  - Skew normal
  - Student t with location/scale
  - Truncated normal
  - Zipf / Zeta

MISSING_PRIMITIVE: normal_pdf (surprisingly absent — have cdf and quantile but not pdf)
MISSING_PRIMITIVE: t_pdf, f_pdf, chi2_pdf
MISSING_PRIMITIVE: distribution random sampling unified with rng.rs

### Special mathematical functions (Level 0 Primitives)
erf, erfc, log_gamma, gamma, log_beta, digamma, trigamma OK
regularized_incomplete_beta, regularized_gamma_p, regularized_gamma_q OK

MISSING_FLAVOR: polygamma (general), incomplete gamma, upper incomplete gamma,
  Hurwitz zeta, Riemann zeta, polylogarithm, elliptic integrals (K, E, Pi),
  hypergeometric functions (1F1, 2F1, pFq), Airy functions, Struve functions,
  spherical harmonics, associated Legendre, Jacobi polynomials, Gegenbauer polynomials

### Orthogonal polynomials (Level 0 Primitives)
chebyshev_t, chebyshev_u, legendre_p, hermite_he, laguerre_l OK
gauss_legendre_nodes_weights OK
MISSING_FLAVOR: associated Laguerre, Jacobi, Gegenbauer, Zernike

### Bessel functions (Level 0 Primitives)
bessel_j0, bessel_j1, bessel_jn, bessel_i0, bessel_i1 OK
MISSING_FLAVOR: Y0, Y1, Yn (Neumann), K0, K1, Kn (modified second kind),
  spherical Bessel, Airy functions

### Marchenko-Pastur (Level 0 Primitives)
marchenko_pastur_pdf, marchenko_pastur_bounds, marchenko_pastur_classify OK
Family tags: random_matrix_theory, statistics, PCA

### Activation / link functions
logistic, logit, softmax, log_softmax OK
sigmoid duplicate documented

---

## numerical

### Root finding (Level 0 Primitives)
bisection, newton, secant, brent, brent_expand, fixed_point — all OK
using() keys: tol, max_iter
Family tags: numerical, root_finding
MISSING_FLAVOR: Muller's method, Jenkins-Traub, Ridder's method,
  Illinois method, Regula Falsi

### Differentiation (Level 0 Primitives)
derivative_central, derivative2_central, derivative_richardson — OK
using() keys: h, n_steps
MISSING_FLAVOR: forward difference, complex step derivative,
  automatic differentiation (structural, not numerical)

### Integration (Level 0 Primitives)
simpson, gauss_legendre_5, adaptive_simpson, trapezoid — OK
using() keys: n, tol, max_depth
MISSING_FLAVOR: Gauss-Kronrod, Clenshaw-Curtis, Romberg, double exponential,
  Monte Carlo integration, Gauss-Laguerre (improper), Gauss-Hermite

### ODE solvers (Level 0 Primitives)
euler, rk4, rk45 (adaptive), rk4_system — OK
using() keys: dt, tol, max_steps
MISSING_FLAVOR: implicit methods (backward Euler, trapezoidal rule, BDF),
  symplectic integrators (Verlet, leapfrog), Dormand-Prince,
  stiff ODE solvers, Radau IIA
MISSING_PRIMITIVE: sde_euler_maruyama, sde_milstein (for stochastic DEs)

### Utility (Level 0 Primitives)
log_sum_exp, weighted_mean, weighted_variance — OK
Family tags: numerical, statistics

---

## series_accel

### Series acceleration (Level 0 Primitives)
partial_sums, cumsum, cesaro_sum, aitken_delta2, wynn_epsilon,
richardson_extrapolate, euler_transform, abel_sum, richardson_partial_sums,
euler_maclaurin_zeta — all OK
Level 0: proper primitives
Family tags: numerical, series, convergence_acceleration

### LevinU (Level 0 Primitive — streaming)
push, estimate, converged — streaming series accelerator
OK: proper streaming primitive

### detect_convergence (Level 0 Primitive)
TBS: `detect_convergence(terms)`
Returns ConvergenceType enum
OK

### accelerate (Level 1 Method)
TBS: `accelerate(terms)`
Composes: detect_convergence -> select best accelerator -> apply
OK: Level 1 auto-selection method

---

## robust

### Weight functions (Level 0 Primitives)
huber_weight, bisquare_weight, hampel_weight — OK
using() keys: k (tuning constant)
Family tags: robust_statistics, weight_functions

### M-estimators (Level 1 Methods)
huber_m_estimate, bisquare_m_estimate, hampel_m_estimate
TBS: `m_estimate(col=0, weight="huber")` or `m_estimate(col=0).using(weight="huber", k=1.345)`
Each should be using() variant of a single m_estimate, not 3 separate functions.
MISSING_COMPOSITION: exist as separate functions, not unified
MISSING_USING_KEY: weight function not selectable via using()
MISSING_FLAVOR: Andrew's wave, Cauchy (Lorentzian), Fair, logistic, Talwar, Welsch

### Scale estimators (Level 0 Primitives)
qn_scale, sn_scale, tau_scale — OK
Family tags: robust_statistics, scale_estimation
MISSING_FLAVOR: MADN (MAD with normal consistency), Hodges-Lehmann estimator

### lts_simple (Level 1 Method)
TBS: `lts(col_x=0, col_y=1)`
Composes: random subsets -> OLS each -> select by trimmed residuals
Family tags: robust_statistics, regression
MISSING_USING_KEY: coverage (proportion of observations), n_trials, seed

### mcd_2d (Level 1 Method)
TBS: `mcd(data)`
Family tags: robust_statistics, covariance
MISSING_FLAVOR: MCD for arbitrary dimensions (currently only 2D)

### medcouple (Level 0 Primitive)
TBS: `medcouple(col=0)`
Level 0: proper primitive (robust skewness measure)
Family tags: robust_statistics, skewness
OK

---

## optimization

### Line search (Level 0 Primitive)
backtracking_line_search — OK

### 1D optimization (Level 0 Primitive)
golden_section — OK
MISSING_FLAVOR: Brent's method for optimization

### Gradient-based (Level 0 Primitives)
gradient_descent, adam, adagrad, rmsprop, lbfgs — OK
using() keys: lr, max_iter, tol, beta1, beta2, epsilon, memory (for L-BFGS)
Family tags: optimization, gradient
MISSING_FLAVOR: SGD with momentum, AMSGrad, Lookahead, LAMB, Shampoo,
  natural gradient, conjugate gradient

### Derivative-free (Level 0 Primitives)
nelder_mead, nelder_mead_with_params, coordinate_descent — OK
MISSING_FLAVOR: Powell's method, CMA-ES, differential evolution, particle swarm,
  simulated annealing, Bayesian optimization
MISSING_PRIMITIVE: all evolutionary/metaheuristic optimizers

### Constrained (Level 0 Primitive)
projected_gradient — OK
MISSING_FLAVOR: augmented Lagrangian, SQP, interior point, barrier methods,
  linear programming (simplex), quadratic programming, ADMM

---

## clustering

### discover_clusters / discover_clusters_with_distance (Level 3 Discovery)
TBS: `discover_clusters(data)`
Composes: multiple clustering algorithms -> ClusterValidation -> best
using() keys: k_range, method, distance
OK: proper Level 3 discovery

### dbscan (Level 1 Method)
TBS: `dbscan(data, eps=0.5, min_pts=5)`
Composes: distance_matrix -> neighborhood_count -> core_points -> union_find -> labels
using() keys: eps, min_pts, metric("euclidean"|"manhattan"|"cosine")
MISSING_USING_KEY: metric (hardcoded via distance matrix)
MISSING_FLAVOR: HDBSCAN, OPTICS

### hierarchical_clustering (Level 1 Method)
TBS: `hierarchical(data, n_clusters, linkage="complete")`
using() keys: n_clusters, linkage("single"|"complete"|"average"|"ward")
MISSING_USING_KEY: linkage is a direct param
MISSING_FLAVOR: centroid linkage, median linkage, minimax linkage

### cluster_validation (Level 0 Primitive)
TBS: `cluster_validate(data, labels)`
COMPOUND_PRIMITIVE: returns silhouette + calinski_harabasz + davies_bouldin + gap_statistic
Each should be independently callable
MISSING_PRIMITIVE: silhouette_score, calinski_harabasz_score, davies_bouldin_score,
  gap_statistic as standalone

### hopkins_statistic (Level 0 Primitive)
TBS: `hopkins(data)`
Family tags: clustering, clusterability
OK

---

## dim_reduction

### pca (Level 1 Method)
TBS: `pca(data, n_components=2)`
Composes: center -> svd -> select_top_k -> project
using() keys: n_components, whiten(true|false), method("full"|"randomized"|"incremental")
Level 0 primitives: col_means OK, svd OK
MISSING_USING_KEY: whiten, method
MISSING_FLAVOR: kernel PCA, sparse PCA, robust PCA, incremental PCA, probabilistic PCA

### classical_mds (Level 1 Method)
TBS: `mds(dist_matrix, n_components=2)`
Composes: double_center -> eigendecomposition -> project
using() keys: n_components, metric(true|false)
MISSING_USING_KEY: metric
MISSING_FLAVOR: non-metric MDS, Sammon mapping, Isomap

### tsne (Level 1 Method)
TBS: `tsne(data, n_components=2, perplexity=30)`
using() keys: n_components, perplexity, learning_rate, max_iter, method("exact"|"barnes_hut")
MISSING_USING_KEY: method (always exact, no Barnes-Hut), early_exaggeration,
  initialization("random"|"pca")
MISSING_FLAVOR: UMAP, TriMap, PaCMAP, LargeVis

### nmf (Level 1 Method)
TBS: `nmf(data, k=5, max_iter=200)`
using() keys: k, max_iter, method("multiplicative"|"als"|"hals")
MISSING_USING_KEY: method (hardcoded multiplicative update)
MISSING_FLAVOR: sparse NMF, semi-NMF, convex NMF

---

## volatility

### garch11_fit (Level 1 Method)
TBS: `garch(col=0, p=1, q=1)`
Composes: moments -> log_likelihood -> L-BFGS optimization -> variance series
using() keys: max_iter, distribution("normal"|"student_t"|"ged"), optimizer
MISSING_USING_KEY: distribution (hardcoded normal), optimizer (hardcoded L-BFGS)
COMPOUND_PRIMITIVE: returns omega + alpha + beta + variances + log_likelihood + iterations
MISSING_COMPOSITION: uses internal sigmoid/logit, not public primitives
MISSING_PARAMETER: initialization parameters hardcoded (alpha=0.1, beta=0.8)
MISSING_FLAVOR: GARCH(p,q) for general p,q (only 1,1 implemented)

### egarch11_fit / gjr_garch11_fit / tgarch11_fit (Level 1 Methods)
TBS: `garch(col=0).using(variant="egarch")` etc.
Should be using() variants of unified garch, not separate functions.
Each only implements (1,1) order.
MISSING_USING_KEY: variant not wirable through using()

### ewma_variance (Level 0 Primitive)
TBS: `ewma_variance(returns, lambda=0.94)`
using() keys: lambda
OK

### Realized volatility measures (Level 0 Primitives)
realized_variance, realized_volatility, bipower_variation,
jump_test_bns, tripower_quarticity — OK
Family tags: volatility, microstructure, realized
OK: each is genuinely atomic

### Market microstructure (Level 0 Primitives)
roll_spread, kyle_lambda, amihud_illiquidity — OK
Family tags: microstructure, liquidity, market_quality

### Range-based volatility (Level 0 Primitives)
parkinson_variance, garman_klass_variance, rogers_satchell_variance,
yang_zhang_variance — OK
Family tags: volatility, range_based
OK: good coverage of range-based estimators

### hill_estimator / hill_tail_alpha (Level 0 Primitives)
TBS: `hill(col=0, k=10)`
Family tags: extreme_value, tail, heavy_tails
MISSING_FLAVOR: Pickands estimator, moment estimator, kernel Hill

### arch_lm_test (Level 1 Method)
TBS: `arch_lm(residuals, n_lags)`
Family tags: time_series, heteroskedasticity
OK

### vpin_bvc (Level 1 Method)
TBS: `vpin(prices, volumes, bucket_volume, n_avg)`
Family tags: microstructure, toxicity
COMPOUND_PRIMITIVE: returns VPIN series + mean + volatility

### Visibility graphs (Level 0 Primitives)
nvg_degree, hvg_degree, nvg_mean_degree, hvg_mean_degree — OK
Family tags: graph, time_series, complexity
OK

---

## survival

### kaplan_meier (Level 1 Method)
TBS: `kaplan_meier(times, events)`
Composes: sort_by_time -> at_risk_count -> conditional_survival -> Greenwood_SE
COMPOUND_PRIMITIVE: returns sequence of (time, n_risk, n_event, survival, se)
using() keys: ci_method("greenwood"|"exponential"|"log"|"log_log")
MISSING_USING_KEY: ci_method (hardcoded Greenwood)
MISSING_FLAVOR: Nelson-Aalen estimator, Fleming-Harrington

### km_median (Level 0+ Composition)
TBS: `km_median(km_result)`
OK

### log_rank_test (Level 1 Method)
TBS: `log_rank(times, events, groups)`
using() keys: weight("logrank"|"wilcoxon"|"tarone_ware"|"peto_prentice"|"fleming_harrington")
MISSING_USING_KEY: weight (hardcoded to standard log-rank)
MISSING_FLAVOR: Wilcoxon (Gehan-Breslow), Tarone-Ware, Peto-Prentice,
  Fleming-Harrington(p,q), stratified log-rank

### cox_ph (Level 1 Method)
TBS: `cox_ph(x, times, events)`
Composes: Newton-Raphson on partial likelihood -> coefficients + SE + hazard_ratios
using() keys: max_iter, tie_method("breslow"|"efron"), strata
MISSING_USING_KEY: tie_method (hardcoded Breslow), strata
MISSING_COMPOSITION: Newton-Raphson inline, not via public optimizer
COMPOUND_PRIMITIVE: returns beta + se + hazard_ratios + log_likelihood +
  schoenfeld_residuals all at once

### grambsch_therneau_test (Level 1 Method)
TBS: `ph_test(cox_result)`
Tests proportional hazards assumption
OK

---

## signal_processing

### FFT/IFFT (Level 0 Primitives)
fft, ifft, rfft, irfft, fft2d — OK
Family tags: signal_processing, spectral, transform
MISSING_FLAVOR: split-radix, Bluestein, chirp-z transform

### Window functions (Level 0 Primitives)
window_hann, window_hamming, window_blackman, window_bartlett,
window_kaiser, window_flat_top — OK
MISSING_FLAVOR: Tukey, Gaussian, Dolph-Chebyshev, Planck-taper, DPSS (Slepian)

### PSD estimation (Level 1 Methods)
periodogram, welch — OK
TBS: `psd(col=0)`, `psd(col=0).using(method="welch", segment=256, overlap=128)`
MISSING_USING_KEY: method not unified, window (hardcoded Hann for welch)
MISSING_FLAVOR: multitaper (exists in spectral.rs),
  Burg (exists via AR in time_series), Capon (MVDR)

### STFT / spectrogram (Level 1 Methods)
stft, spectrogram — OK
using() keys: window_len, hop_size, window("hann"|"hamming"|...)
MISSING_USING_KEY: window type (hardcoded Hann)

### Convolution / correlation (Level 0 Primitives)
convolve, cross_correlate, autocorrelation — OK
Family tags: signal_processing
OK

### DCT (Level 0 Primitives)
dct2, dct3 — OK
MISSING_FLAVOR: DCT-I, DCT-IV, DST (discrete sine transform)

### Filters (Level 0 Primitives)
fir_lowpass, fir_highpass, fir_bandpass, fir_filter — OK
Biquad, butterworth_lowpass_cascade — OK
moving_average, ema, savgol_filter, median_filter — OK
Family tags: signal_processing, filter
MISSING_FLAVOR: Chebyshev filter, elliptic filter, notch/bandstop,
  IIR filter design (currently only Butterworth)

### Hilbert / envelope / instantaneous_frequency (Level 0 Primitives)
hilbert, envelope, instantaneous_frequency — OK
Family tags: signal_processing, analytic_signal
OK

### Cepstrum (Level 0 Primitive)
real_cepstrum — OK
MISSING_FLAVOR: complex cepstrum, power cepstrum

### Wavelets (Level 0 Primitives)
haar_dwt, haar_idwt, haar_wavedec, haar_waverec, db4_dwt, db4_idwt — OK
morlet_wavelet, morlet_cwt — OK
MISSING_FLAVOR: Daubechies (general order), symlets, coiflets,
  Meyer wavelet, biorthogonal wavelets, wavelet packets,
  stationary wavelet transform, MODWT, synchrosqueezing

### Goertzel (Level 0 Primitive)
goertzel, goertzel_mag — OK
Family tags: signal_processing, spectral, single_frequency
OK

### Savitzky-Golay (Level 0 Primitive)
savitzky_golay (general with derivative order) — OK

### Regularization (Level 0 Primitives)
regularize_interp, regularize_bin_mean, regularize_subsample — OK

### Path signatures (Level 0 Primitives)
path_signature_2d, log_signature_2d — OK
Family tags: topology, path, signature
MISSING_FLAVOR: higher-order signatures, arbitrary dimension

### Wigner-Ville distribution (Level 1 Method)
wvd_features — OK
COMPOUND_PRIMITIVE: returns mean_freq + bandwidth + time_width + time_freq_spread +
  peak_magnitude + phase_coherence

### FastICA (Level 1 Method)
fast_ica — OK
MISSING_USING_KEY: n_components, algorithm("parallel"|"deflation"),
  nonlinearity("logcosh"|"exp"|"cube")
MISSING_FLAVOR: JADE, SOBI, kernel ICA

### EMD (Level 1 Method)
emd — OK
MISSING_USING_KEY: max_imfs, max_sift_iter, sift_threshold
MISSING_FLAVOR: EEMD, CEEMDAN, VMD

---

## bayesian

### metropolis_hastings (Level 0 Primitive)
TBS: `mcmc(log_posterior, proposal, n_samples, seed)`
using() keys: n_burnin, thin, proposal_scale
MISSING_USING_KEY: n_burnin, thin (not implemented)
MISSING_FLAVOR: HMC, NUTS, Gibbs sampler, slice sampling, MALA,
  parallel tempering, SMC, ABC

### bayesian_linear_regression (Level 1 Method)
TBS: `bayesian_regression(x, y)`
using() keys: prior_precision, noise_precision, n_samples
MISSING_FLAVOR: Bayesian logistic regression, Bayesian GP regression,
  variational Bayes, expectation propagation

### effective_sample_size / r_hat (Level 0 Primitives)
TBS: `ess(samples)` / `rhat(chains)`
Family tags: bayesian, diagnostics, mcmc
OK

---

## graph

### Graph algorithms (Level 0 Primitives)
bfs, dfs, topological_sort, connected_components — OK
dijkstra, bellman_ford, floyd_warshall — OK
kruskal, prim — OK
degree_centrality, closeness_centrality, pagerank — OK
label_propagation, modularity — OK
max_flow, diameter, density, clustering_coefficient — OK
Family tags: graph, algorithm

MISSING_FLAVOR: betweenness centrality, eigenvector centrality,
  community detection (Louvain, Girvan-Newman),
  A* search, Johnson's algorithm, bipartite matching,
  minimum vertex cover, graph coloring, isomorphism

### pairwise_dists / knn_adjacency / graph_laplacian (Level 0 Primitives)
OK: proper building blocks for spectral methods

---

## interpolation

### Polynomial interpolation (Level 0 Primitives)
lagrange, newton_divided_diff, newton_eval, neville — OK
Family tags: interpolation, approximation

### Linear / nearest (Level 0 Primitives)
lerp, nearest — OK

### Splines (Level 0 Primitives)
natural_cubic_spline, clamped_cubic_spline, monotone_hermite,
akima, pchip — OK
Family tags: interpolation, spline
MISSING_FLAVOR: B-spline (exists as bspline_basis/eval/uniform_knots — OK),
  tension spline, Hermite interpolation, NURBS

### Chebyshev approximation (Level 0 Primitives)
chebyshev_nodes, chebyshev_coefficients, chebyshev_eval,
chebyshev_approximate — OK
Family tags: approximation, polynomial

### Polynomial fitting (Level 1 Method)
polyfit — OK
using() keys: deg, method("ols"|"orthogonal")
MISSING_USING_KEY: method (hardcoded to Vandermonde system)

### RBF interpolation (Level 0 Primitive)
rbf_interpolate — OK
using() keys: kernel(RbfKernel enum)
MISSING_FLAVOR: thin plate spline, polyharmonic, compact support RBFs

### Barycentric rational (Level 0 Primitive)
barycentric_rational — OK

### GP regression (Level 1 Method)
gp_regression — OK
using() keys: kernel, length_scale, noise, n_restarts
MISSING_USING_KEY: most GP parameters hardcoded
MISSING_FLAVOR: sparse GP, GP classification

### Pade approximant (Level 0 Primitive)
pade — OK
Family tags: approximation, rational

---

## irt

### IRT models (Level 1 Methods)
rasch_prob (Level 0), prob_2pl (Level 0), prob_3pl (Level 0) — OK
fit_2pl (Level 1), ability_mle, ability_eap — OK
item_information, test_information, sem — OK
MISSING_FLAVOR: GRM (graded response model), PCM (partial credit),
  GPCM, nominal response model, MIRT (multidimensional IRT)
MISSING_PRIMITIVE: fit_1pl/fit_rasch (only 2PL fitting exists)

### mantel_haenszel_dif (Level 1 Method)
TBS: `dif(responses, group, items)`
Family tags: psychometrics, fairness
OK

---

## data_quality

This module is primarily fintek-facing (validity checks, counting primitives).
Most functions are Level 0 primitives.

### Counting primitives (Level 0)
count_finite, count_nan, count_inf, count_positive, count_negative,
count_zeros, count_above, count_below, count_in_range,
count_zero_crossings, count_sign_changes, count_peaks, count_troughs,
count_inflections, count_runs, count_outliers_mad, count_outliers_zscore,
count_inversions, count_ties — all OK
Family tags: data_quality, counting, descriptive

Note: many of these DUPLICATE functions in data_quality_catalog.rs.
Both modules have count_peaks, count_troughs, count_zeros, etc.

### Validity checks (Level 0)
fft_is_valid, garch_is_valid, rank_based_is_valid, etc. — OK
These are domain-specific guards, not TBS primitives.

### IAT statistics (Level 0 Primitives)
iat_mean, iat_median, iat_variance, iat_std, iat_mad, iat_skewness,
iat_kurtosis, iat_gini, iat_hill_tail_index, iat_lag1_autocorrelation,
iat_memory_coefficient, iat_burstiness, iat_entropy,
poisson_dispersion_index, fano_factor, iat_allan_variance,
iat_ks_exponential, iat_ks_uniform — all OK
Family tags: data_quality, timing, point_process

### Inequality measures (Level 0 Primitives)
theil_t, theil_l, atkinson_index, hoover_index, palma_ratio,
coefficient_of_dispersion, quartile_coefficient_of_dispersion — OK
Family tags: inequality, economics, descriptive

### DataQualityProfile (Level 2 Pipeline)
from_slice, from_slice_with_params, from_session — Level 2 pipelines
Composes: moments + many data quality metrics
COMPOUND_PRIMITIVE: massive profile object with ~40 fields

---

## stochastic

### Brownian motion / GBM / OU (Level 0 Primitives)
brownian_motion, brownian_bridge, geometric_brownian_motion,
ornstein_uhlenbeck — OK
Family tags: stochastic, brownian, sde

### Black-Scholes (Level 0 Primitive)
TBS: `black_scholes(s=100, k=105, t=0.25, r=0.05, sigma=0.2, call=true)`
OK
MISSING_FLAVOR: Greeks as separate primitives (delta, gamma, theta, vega, rho)
MISSING_PRIMITIVE: bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho

### Markov chains (Level 0 Primitives)
markov_n_step, stationary_distribution, mean_first_passage_time,
is_ergodic, mixing_time — OK
Family tags: stochastic, markov

### CTMC (Level 0 Primitives)
ctmc_transition_matrix, ctmc_stationary, ctmc_holding_time — OK

### Queuing theory (Level 0 Primitives)
birth_death_stationary, mm1_queue, erlang_c — OK
Family tags: queuing, operations_research
MISSING_FLAVOR: M/M/c, M/G/1, G/G/1, Jackson networks

### Random walks (Level 0 Primitives)
simple_random_walk, first_passage_time_cdf, return_probability_1d — OK

### Ito / Stratonovich (Level 0 Primitives)
ito_integral, stratonovich_integral, ito_lemma_verification — OK
Family tags: stochastic, calculus

---

## factor_analysis

### correlation_matrix (Level 0 Primitive)
TBS: `corr_matrix(data)`
using() keys: method("pearson"|"spearman"|"kendall"|"polychoric"|"tetrachoric")
MISSING_USING_KEY: method (only Pearson)
Family tags: statistics, correlation, multivariate

### principal_axis_factoring (Level 1 Method)
TBS: `efa(data, n_factors=3)`
Composes: correlation_matrix -> communality_estimation -> eigendecomposition -> iterate
using() keys: n_factors, max_iter, extraction("principal_axis"|"ml"|"minres"|"uls")
MISSING_USING_KEY: extraction method (only PAF)
MISSING_FLAVOR: maximum likelihood FA, MINRES, ULS, image factoring

### varimax (Level 0 Primitive)
TBS: `rotate(loadings, method="varimax")`
using() keys: method, max_iter, gamma (for Crawford-Ferguson family)
MISSING_USING_KEY: gamma
MISSING_FLAVOR: promax, oblimin, quartimin, equamax, parsimax, Crawford-Ferguson

### cronbachs_alpha / mcdonalds_omega (Level 0 Primitives)
TBS: `reliability(data, method="cronbach")` / `omega(loadings)`
Family tags: psychometrics, reliability
OK

### scree_elbow / kaiser_criterion (Level 0 Primitives)
TBS: `n_factors(eigenvalues, method="kaiser")` / `n_factors(eigenvalues, method="scree")`
MISSING_FLAVOR: parallel analysis (Horn's), MAP (Velicer's), very simple structure

### kmo_bartlett (Level 1 Method)
TBS: `kmo(data)` / `bartlett_sphericity(data)`
COMPOUND_PRIMITIVE: returns KMO + Bartlett chi2 + p together
Should be two separate primitives

---

## hmm (two implementations!)

Note: HMM functions exist in BOTH hmm.rs AND kalman.rs. Potential duplication.

### hmm_forward / hmm_backward / hmm_forward_backward (Level 0 Primitives)
TBS: `hmm_forward(model, observations)`
Family tags: probabilistic, sequence, hidden_markov
OK

### hmm_viterbi (Level 0 Primitive)
TBS: `hmm_viterbi(model, observations)`
OK

### hmm_baum_welch (Level 1 Method)
TBS: `hmm_fit(observations, n_states)`
Composes: random_init -> iterate(forward_backward -> re_estimate) -> converge
using() keys: max_iter, tol, n_restarts
MISSING_USING_KEY: n_restarts (for random initialization)

---

## kalman / state_space

### kalman_filter_scalar / kalman_filter_matrix (Level 0 Primitives)
TBS: `kalman(model, observations)`
Family tags: state_space, filtering, bayesian
OK: both scalar and matrix versions

### rts_smoother / rts_smoother_scalar (Level 0 Primitives)
TBS: `rts_smoother(model, filter_result)`
OK

### ssm_em (Level 1 Method)
TBS: `ssm_fit(observations, model_template, max_iter)`
Composes: E-step(kalman_filter + rts_smoother) -> M-step -> iterate
COMPOUND_PRIMITIVE: returns fitted model + log_likelihood + iterations

### particle_filter (Level 1 Method)
TBS: `particle_filter(transition, observation, n_particles)`
using() keys: n_particles, resampling("systematic"|"multinomial"|"stratified"|"residual")
MISSING_USING_KEY: resampling method (hardcoded systematic)
MISSING_FLAVOR: auxiliary particle filter, Rao-Blackwellized particle filter

---

## spatial

### Distance functions (Level 0 Primitives)
euclidean_2d, haversine — OK
MISSING_FLAVOR: Vincenty, great circle, Manhattan

### Variogram / Kriging (Level 1 Methods)
empirical_variogram, ordinary_kriging — OK
using() keys: model("spherical"|"exponential"|"gaussian"), n_bins, max_lag
MISSING_USING_KEY: model (parameter to kriging, not wirable via using())
MISSING_FLAVOR: universal kriging, indicator kriging, co-kriging

### Spatial statistics (Level 0 Primitives)
morans_i, gearys_c, ripleys_k, ripleys_l, clark_evans_r — OK
Family tags: spatial, statistics, point_process
OK: good coverage

### Computational geometry (Level 0 Primitives)
convex_hull_2d, polygon_area, polygon_perimeter — OK
MISSING_FLAVOR: Delaunay triangulation, Voronoi diagram, alpha shapes,
  point-in-polygon

---

## tda

### Persistent homology (Level 0 Primitives)
rips_h0, rips_h1 — OK
Family tags: topology, tda, persistent_homology
MISSING_FLAVOR: alpha complex, Cech complex, cubical complex,
  Vietoris-Rips with lazy witness

### Persistence distances (Level 0 Primitives)
bottleneck_distance, wasserstein_distance — OK

### Persistence statistics (Level 0 Primitives)
persistence_statistics, persistence_entropy, betti_curve — OK

---

## train

### Linear regression (train/linear.rs) (Level 1 Method)
TBS: `linear_model(x, y)`
Composes: column_stats -> center -> cholesky_solve -> coefficients
using() keys: standardize(true|false)
MISSING_USING_KEY: standardize, regularization

### Logistic regression (train/logistic.rs) (Level 1 Method)
TBS: `logistic_model(x, y)`
Composes: IRLS -> coefficients -> sigmoid -> predictions
MISSING_FLAVOR: multinomial logistic

### Gaussian Naive Bayes (train/naive_bayes.rs) (Level 1 Method)
TBS: `naive_bayes(x, y)`
using() keys: var_smoothing
MISSING_FLAVOR: multinomial NB, Bernoulli NB, complement NB

---

## scoring / hazard / predictive / distributional_distances

These are thin modules primarily serving BOCPD changepoint detection.

### scoring: score_max_posterior_drop, score_rl0_mass, threshold_fixed, threshold_adaptive
Level 0 primitives, using() bag aware. OK.

### hazard: hazard_constant, hazard_geometric, hazard_power_law
Level 0 primitives, using() bag aware. OK.

### predictive: predictive_gaussian, predictive_student_t
Level 0 primitives, using() bag aware.
MISSING_FLAVOR: Poisson predictive, negative binomial predictive

### distributional_distances: total_variation_distance, hellinger_distance, kl_divergence, js_divergence
Level 0 primitives, using() bag aware.
Note: kl_divergence duplicates information_theory::kl_divergence but takes UsingBag.
MISSING_FLAVOR: Wasserstein distance, energy distance, MMD (maximum mean discrepancy),
  Bhattacharyya distance, Mahalanobis between distributions,
  Cramer distance, Anderson-Darling distance

---

# GAP SUMMARY

## Structural / architectural gaps

1. **Duplicate implementations**: jarque_bera (nonparametric + hypothesis), wls (hypothesis +
   multivariate), kl_divergence/js_divergence (information_theory + distributional_distances),
   sigmoid/logistic (linear_algebra + special_functions), pp_test/phillips_perron_test
   (time_series), HMM (hmm.rs + kalman.rs), count_* functions (data_quality + data_quality_catalog)

2. **Separate functions that should be using() variants**: ar_fit vs ar_burg_fit,
   huber_m_estimate vs bisquare_m_estimate vs hampel_m_estimate,
   garch11_fit vs egarch11_fit vs gjr_garch11_fit vs tgarch11_fit,
   kde vs kde_fft, geometric_mean/harmonic_mean/trimmed_mean/winsorized_mean
   (should be mean with using(method=...))

3. **Compound primitives that need decomposition**: forecast_metrics (should yield mae, rmse,
   mape, mase individually), cluster_validation (should yield silhouette, CH, DB individually),
   describe (returns 6 things), TestResult objects (stat + p + effect + CI all at once),
   DataQualityProfile (~40 fields)

4. **using() bag not wired**: Most primitives take direct parameters but do not query UsingBag.
   Only scoring, hazard, predictive, and distributional_distances modules use UsingBag.
   Every parameter on every function is a MISSING_USING_KEY until the using() bag infrastructure
   is plumbed through.

## Most critical missing primitives

1. **normal_pdf, t_pdf, f_pdf, chi2_pdf** — have CDF and quantile but not PDF for main distributions
2. **tie_count** — used by Kendall, Mann-Whitney, Wilcoxon but not standalone
3. **mae, rmse, mape, mase, smape** — forecast error metrics as standalone
4. **silhouette_score, calinski_harabasz_score, davies_bouldin_score** — cluster validation as standalone
5. **ols_residuals** — used by partial correlation, ADF, KPSS but not standalone
6. **sheather_jones_bandwidth** — optimal KDE bandwidth
7. **polychoric_correlation** — for ordinal data
8. **seasonal_difference** — for SARIMA
9. **bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho** — option Greeks

## Most critical missing flavors

1. **Correlation**: MIC, Hoeffding's D, Schweizer-Wolff, Blomqvist beta, polychoric
2. **Bootstrap**: BCa, block, wild, circular
3. **Clustering**: HDBSCAN, OPTICS, spectral clustering (exists in spectral_clustering.rs),
   k-medoids, GMM EM
4. **Dimensionality reduction**: UMAP, kernel PCA, Isomap, diffusion maps
5. **Normality tests**: Lilliefors, Cramer-von Mises (general), Anderson-Darling for other distributions
6. **Multiple comparison**: Sidak, Hochberg, Hommel, Games-Howell, Scheffe, Dunnett
7. **GARCH family**: general GARCH(p,q), FIGARCH, component GARCH, multivariate GARCH (DCC, BEKK)
8. **Optimization**: CMA-ES, differential evolution, simulated annealing, Bayesian optimization,
   SQP, interior point
9. **MCMC**: HMC, NUTS, Gibbs, slice sampling
10. **Entropy**: Grassberger estimator, Chao-Shen, NSB, multiscale entropy, fuzzy entropy
11. **Distributions**: hypergeometric, Gumbel, Rayleigh, Laplace, inverse Gaussian, Dirichlet,
    multivariate normal
12. **Post-hoc**: Games-Howell, Scheffe, Dunnett
13. **Effect sizes**: Cliff's delta, CLES, epsilon squared

## using() coverage assessment

Of ~400 pub fns audited:
- **~10 functions** actually query UsingBag (scoring, hazard, predictive, distributional_distances)
- **~390 functions** take direct parameters with no using() wiring
- This means the TBS surface described in the architecture docs is ~97% aspirational.
  Plumbing using() through every primitive is a significant infrastructure task.

## Composition assessment

Of ~80 Level 1 methods audited:
- **~15** properly compose public Level 0 primitives (spearman, arima_fit, paired_t)
- **~40** embed sub-computations inline (kendall_tau embeds tie counting, mann_whitney embeds
  tie correction, adf_test embeds OLS, garch embeds optimization, etc.)
- **~25** call other module functions directly but not through TBS-composable interface
- This means ~50% of methods need primitive extraction to fully support the
  decomposition contract.
