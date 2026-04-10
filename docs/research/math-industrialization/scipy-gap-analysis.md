# Scipy Gap Analysis — Tambear Coverage

Generated 2026-04-10. Covers scipy 1.x public API across all major submodules.

Legend:
- **HAVE** = implemented, named function exists in tambear
- **PARTIAL** = partially covered (e.g., scalar version but not matrix version, or approximation rather than full implementation)
- **MISSING** = not found in any tambear module

---

## scipy.stats — Summary Statistics & Descriptive

| scipy function | tambear equivalent | status |
|---|---|---|
| `describe` | `descriptive::moments_ungrouped` | HAVE |
| `gmean` | `descriptive::geometric_mean` | HAVE |
| `hmean` | `descriptive::harmonic_mean` | HAVE |
| `pmean` | — | MISSING |
| `kurtosis` | `descriptive::MomentStats.excess_kurtosis` | HAVE |
| `mode` | — | MISSING |
| `moment` | `descriptive::MomentStats` (1st–4th moments) | PARTIAL |
| `lmoment` | — | MISSING |
| `expectile` | — | MISSING |
| `skew` | `descriptive::MomentStats.skewness` | HAVE |
| `kstat` | — | MISSING |
| `kstatvar` | — | MISSING |
| `tmean` | `descriptive::trimmed_mean` | HAVE |
| `tvar` | — | MISSING |
| `tmin` | — | MISSING |
| `tmax` | — | MISSING |
| `tstd` | — | MISSING |
| `tsem` | — | MISSING |
| `variation` | `descriptive::coefficient_of_variation` | HAVE |
| `rankdata` | `nonparametric::rank` | HAVE |
| `tiecorrect` | `nonparametric::tie_count` | HAVE |
| `trim_mean` | `descriptive::trimmed_mean` | HAVE |
| `gstd` | — | MISSING |
| `iqr` | `descriptive::iqr` | HAVE |
| `sem` | — | MISSING |
| `bayes_mvs` | — | MISSING |
| `mvsdist` | — | MISSING |
| `entropy` | `information_theory::shannon_entropy` | HAVE |
| `differential_entropy` | — | MISSING |
| `median_abs_deviation` | `descriptive::mad` | HAVE |

### scipy.stats — Gap Summary: Summary Statistics

**MISSING (moderate complexity):**

- `pmean` — Power/generalized mean with parameter p. Composes from `geometric_mean`, `harmonic_mean`, arithmetic mean. Trivial — just `(sum(x^p)/n)^(1/p)`.
- `mode` — Modal value of array (argmax of frequency). Trivial — needs a frequency-count accumulate.
- `lmoment` — L-moments (Hosking, 1990). Computes from order statistics. Moderate — needs sort + linear combinations of order stats.
- `expectile` — Asymmetric least squares analog of quantile. Moderate — iterative IRLS.
- `kstat` — Polykay/k-statistics (unbiased cumulant estimators). Moderate — algebraic expressions over central moments.
- `kstatvar` — Variance of k-statistics. Moderate — algebraic, builds on kstat.
- `tvar/tmin/tmax/tstd/tsem` — Trimmed versions of variance, min, max, std, std-error-of-mean. Trivial — slice after sort, then standard formulas.
- `gstd` — Geometric standard deviation: `exp(std(log(x)))`. Trivial composition.
- `sem` — Standard error of mean: `std / sqrt(n)`. Trivial.
- `differential_entropy` — Entropy of continuous distribution (KDE-based or parametric). Moderate — needs KDE + integration.
- `bayes_mvs` — Bayesian confidence intervals on mean, variance, std. Moderate — t and chi2 posterior forms.
- `mvsdist` — Returns frozen rv_continuous objects for mean/variance/std posterior. Complex — needs distribution objects.

---

## scipy.stats — Frequency Statistics

| scipy function | tambear equivalent | status |
|---|---|---|
| `cumfreq` | — | MISSING |
| `quantile` | `descriptive::quantile` | HAVE |
| `percentileofscore` | — | MISSING |
| `scoreatpercentile` | — | MISSING |
| `relfreq` | — | MISSING |
| `binned_statistic` | — | MISSING |
| `binned_statistic_2d` | — | MISSING |
| `binned_statistic_dd` | — | MISSING |

**MISSING:**

- `cumfreq` — Cumulative frequency histogram. Trivial — sort + cumsum.
- `percentileofscore` — Rank of a score within a dataset as percentile. Trivial — binary search + ECDF.
- `scoreatpercentile` — Inverse: value at given percentile. Trivial — calls quantile.
- `relfreq` — Relative frequency histogram. Trivial — counts / total.
- `binned_statistic` — Apply function to values grouped by 1D bins. Moderate — custom accumulate with bin-grouping.
- `binned_statistic_2d` — 2D version. Moderate.
- `binned_statistic_dd` — N-D version. Moderate.

---

## scipy.stats — Hypothesis Tests (One Sample / Paired)

| scipy function | tambear equivalent | status |
|---|---|---|
| `ttest_1samp` | `hypothesis::one_sample_t` | HAVE |
| `binomtest` | `hypothesis::one_proportion_z` | PARTIAL |
| `quantile_test` | — | MISSING |
| `skewtest` | — | MISSING |
| `kurtosistest` | — | MISSING |
| `normaltest` | `nonparametric::dagostino_pearson` | HAVE |
| `jarque_bera` | `nonparametric::jarque_bera` | HAVE |
| `shapiro` | `nonparametric::shapiro_wilk` | HAVE |
| `anderson` | `nonparametric::anderson_darling` | HAVE |
| `cramervonmises` | — | MISSING |
| `ks_1samp` | `nonparametric::ks_test_normal` | PARTIAL |
| `goodness_of_fit` | — | MISSING |
| `chisquare` | `hypothesis::chi2_goodness_of_fit` | HAVE |
| `power_divergence` | — | MISSING |
| `ttest_rel` | `hypothesis::paired_t` | HAVE |
| `wilcoxon` | `nonparametric::wilcoxon_signed_rank` | HAVE |

**MISSING:**

- `binomtest` — Exact binomial test (we have z-approximation, not exact). Moderate — needs exact binomial CDF.
- `quantile_test` — Tests whether population quantile equals given value. Moderate — sign test variant.
- `skewtest` — D'Agostino skewness test (separate from combined normaltest). Moderate — z-transform of sample skewness.
- `kurtosistest` — Anscombe-Glynn kurtosis test. Moderate — z-transform of excess kurtosis.
- `cramervonmises` — Cramér-von Mises goodness-of-fit vs normal (or any CDF). Moderate — sum of squared differences between ECDF and theoretical CDF.
- `ks_1samp` — We have normal-specific; need general CDF version. Moderate — accepts arbitrary CDF function.
- `goodness_of_fit` — Composite goodness-of-fit framework (fit + test). Complex.
- `power_divergence` — Generalized power divergence statistic (includes chi2, G-test, Freeman-Tukey, etc.). Moderate — one formula family.

---

## scipy.stats — Hypothesis Tests (Correlation / Association)

| scipy function | tambear equivalent | status |
|---|---|---|
| `linregress` | `linear_algebra::simple_linear_regression` | HAVE |
| `pearsonr` | `nonparametric::pearson_r` | HAVE |
| `spearmanr` | `nonparametric::spearman` | HAVE |
| `pointbiserialr` | `nonparametric::point_biserial` | HAVE |
| `kendalltau` | `nonparametric::kendall_tau` | HAVE |
| `chatterjeexi` | — | MISSING |
| `weightedtau` | — | MISSING |
| `somersd` | — | MISSING |
| `siegelslopes` | — | MISSING |
| `theilslopes` | — | MISSING |
| `page_trend_test` | — | MISSING |
| `multiscale_graphcorr` | — | MISSING |
| `chi2_contingency` | `hypothesis::chi2_independence` | HAVE |
| `fisher_exact` | `hypothesis::fisher_exact` | HAVE |
| `barnard_exact` | — | MISSING |
| `boschloo_exact` | — | MISSING |

**MISSING:**

- `chatterjeexi` — Chatterjee's xi correlation (2021, rank-based, asymmetric measure of dependence). Moderate — sort + rank + formula.
- `weightedtau` — Weighted Kendall tau with Knight's algorithm. Moderate — extends kendall_tau.
- `somersd` — Somers' D (asymmetric rank correlation). Moderate — based on concordant/discordant pairs.
- `siegelslopes` — Repeated medians regression (robust). Moderate — median of medians of slopes.
- `theilslopes` — Theil-Sen estimator (median of pairwise slopes). Moderate — O(n^2) naive, O(n log n) optimal.
- `page_trend_test` — Page's trend test for ordered alternatives in repeated measures. Moderate — rank sums with weights.
- `multiscale_graphcorr` — Distance/kernel-based independence test (MGC). Complex — requires repeated permutation over distance graphs.
- `barnard_exact` — Barnard's exact test for 2x2 contingency tables. Complex — nuisance parameter maximization.
- `boschloo_exact` — Boschloo's exact test (max of Fisher p-values). Complex — numerical maximization.

---

## scipy.stats — Hypothesis Tests (Two-sample / Multi-sample)

| scipy function | tambear equivalent | status |
|---|---|---|
| `ttest_ind` | `hypothesis::welch_t` | HAVE |
| `ttest_ind_from_stats` | `hypothesis::welch_t` (from MomentStats) | HAVE |
| `poisson_means_test` | — | MISSING |
| `mannwhitneyu` | `nonparametric::mann_whitney_u` | HAVE |
| `bws_test` | — | MISSING |
| `ranksums` | — | MISSING |
| `brunnermunzel` | — | MISSING |
| `mood` | — | MISSING |
| `ansari` | — | MISSING |
| `cramervonmises_2samp` | — | MISSING |
| `epps_singleton_2samp` | — | MISSING |
| `ks_2samp` | `nonparametric::ks_test_two_sample` | HAVE |
| `kstest` | `nonparametric::ks_test_normal` | PARTIAL |
| `f_oneway` | `hypothesis::one_way_anova` | HAVE |
| `tukey_hsd` | `hypothesis::tukey_hsd` | HAVE |
| `dunnett` | — | MISSING |
| `kruskal` | `nonparametric::kruskal_wallis` | HAVE |
| `alexandergovern` | — | MISSING |
| `fligner` | — | MISSING |
| `levene` | `hypothesis::levene_test` | HAVE |
| `bartlett` | — | MISSING |
| `median_test` | — | MISSING |
| `friedmanchisquare` | `nonparametric::friedman_test` | HAVE |
| `anderson_ksamp` | — | MISSING |

**MISSING:**

- `poisson_means_test` — E-test for equality of Poisson rates. Moderate — uses chi2 distribution.
- `bws_test` — Baumgartner-Weiss-Schindler test for distributional equality. Moderate — rank-based with special weights.
- `ranksums` — Wilcoxon rank-sum test (= Mann-Whitney formulation). Trivial — alias to mann_whitney_u output with z-score presentation.
- `brunnermunzel` — Brunner-Munzel test (non-parametric, doesn't assume equal variances). Moderate — rank-based with variance correction.
- `mood` — Mood's test for scale difference. Moderate — rank-based.
- `ansari` — Ansari-Bradley test for scale. Moderate — rank-based.
- `cramervonmises_2samp` — Two-sample CvM test. Moderate — compares ECDFs.
- `epps_singleton_2samp` — Epps-Singleton two-sample test (characteristic function-based). Complex — integrates characteristic functions.
- `dunnett` — Dunnett's test (many-to-one post-hoc). Complex — uses multivariate-t distribution.
- `alexandergovern` — Alexander-Govern test (heteroscedastic ANOVA alternative). Moderate — F-like with Welch correction.
- `fligner` — Fligner-Killeen test for variance homogeneity. Moderate — rank-based.
- `bartlett` — Bartlett's test for variance homogeneity (assumes normality). Moderate — chi2-based.
- `median_test` — Mood's median test. Moderate — contingency table chi2.
- `anderson_ksamp` — k-sample Anderson-Darling test. Complex — extends 2-sample AD to k groups.

---

## scipy.stats — Resampling Methods

| scipy function | tambear equivalent | status |
|---|---|---|
| `monte_carlo_test` | — | MISSING |
| `permutation_test` | `nonparametric::permutation_test_mean_diff` | PARTIAL |
| `bootstrap` | `nonparametric::bootstrap_percentile` | PARTIAL |
| `power` | `hypothesis::power_one_sample_t` etc | PARTIAL |

**MISSING / PARTIAL:**

- `monte_carlo_test` — Generic Monte Carlo hypothesis test (user-supplied statistic). Moderate — generic resampling loop.
- `permutation_test` — We have mean-diff only; need generic statistic version. Moderate — generalize existing.
- `bootstrap` — We have percentile CI only; need BCa, basic, studentized methods. Moderate — additional CI variants.
- `power` — We have analytical power for t/ANOVA/correlation; need general simulation-based power. Moderate.

---

## scipy.stats — Distributions (CDFs, PDFs, quantiles)

We have: normal, t, f, chi2, weibull, pareto, exponential, lognormal, beta, gamma, poisson, binomial, negative-binomial, cauchy, studentized_range.

**MISSING continuous distributions (all follow same CDF/PDF/quantile pattern — trivial to moderate per distribution):**

alpha, anglit, arcsine, argus, betaprime, bradford, burr, burr12, crystalball, dgamma, dweibull, erlang, exponnorm, exponweib, exponpow, fatiguelife, fisk, foldcauchy, foldnorm, genlogistic, gennorm, genpareto, genexpon, genextreme, gausshyper, gengamma, genhalflogistic, genhyperbolic, geninvgauss, gibrat, gompertz, gumbel_r, gumbel_l, halfcauchy, halflogistic, halfnorm, halfgennorm, hypsecant, invgamma, invgauss, invweibull, irwinhall, jf_skew_t, johnsonsb, johnsonsu, kappa4, kappa3, ksone, kstwo, kstwobign, landau, laplace, laplace_asymmetric, levy, levy_l, levy_stable, logistic, loggamma, loglaplace, loguniform, lomax, maxwell, mielke, moyal, nakagami, ncx2, ncf, nct, norminvgauss, pearson3, powerlaw, powerlognorm, powernorm, rdist, rayleigh, rel_breitwigner, rice, recipinvgauss, semicircular, skewcauchy, skewnorm, studentized_range (HAVE), trapezoid, triang, truncexpon, truncnorm, truncpareto, truncweibull_min, tukeylambda, uniform, vonmises, vonmises_line, wald, weibull_min (HAVE via weibull_cdf), weibull_max, wrapcauchy.

**MISSING discrete distributions:**

bernoulli, betabinom, betanbinom, boltzmann, dlaplace, geom, hypergeom, logser, nchypergeom_fisher, nchypergeom_wallenius, nhypergeom, planck, poisson_binom, randint, skellam, yulesimon, zipf, zipfian.

**MISSING multivariate distributions:**

multivariate_normal, matrix_normal, dirichlet, dirichlet_multinomial, wishart, invwishart, multinomial, special_ortho_group, ortho_group, unitary_group, random_correlation, multivariate_t, multivariate_hypergeom, normal_inverse_gamma, random_table, uniform_direction, vonmises_fisher, matrix_t.

---

## scipy.stats — Statistical Distances

| scipy function | tambear equivalent | status |
|---|---|---|
| `wasserstein_distance` | `information_theory::wasserstein_1d` | HAVE |
| `wasserstein_distance_nd` | — | MISSING |
| `energy_distance` | `information_theory::energy_distance` | HAVE |

**MISSING:**
- `wasserstein_distance_nd` — Multi-dimensional Wasserstein distance. Complex — requires optimal transport solver (e.g., Sinkhorn or linear programming).

---

## scipy.stats — Transformations

| scipy function | tambear equivalent | status |
|---|---|---|
| `boxcox` | `descriptive::box_cox_transform` | HAVE |
| `boxcox_normmax` | `descriptive::box_cox_fit` | HAVE |
| `boxcox_llf` | `descriptive::box_cox_log_likelihood` | HAVE |
| `yeojohnson` | — | MISSING |
| `yeojohnson_normmax` | — | MISSING |
| `yeojohnson_llf` | — | MISSING |
| `obrientransform` | — | MISSING |
| `sigmaclip` | — | MISSING |
| `trimboth` | — | MISSING |
| `trim1` | — | MISSING |
| `zmap` | — | MISSING |
| `zscore` | — | MISSING |
| `gzscore` | — | MISSING |

**MISSING:**
- `yeojohnson` / `yeojohnson_normmax` / `yeojohnson_llf` — Yeo-Johnson power transform (extends Box-Cox to negative values). Moderate — same structure as Box-Cox.
- `obrientransform` — O'Brien's transformation for variance homogeneity tests. Moderate — formula on each group.
- `sigmaclip` — Iterative sigma-clipping outlier removal. Trivial — loop: remove points > k*sigma from mean.
- `trimboth` / `trim1` — Trim both/one side of sorted array. Trivial — slice.
- `zmap` / `zscore` / `gzscore` — Standard score, reference-group z-score, geometric z-score. Trivial compositions.

---

## scipy.stats — Directional Statistics

| scipy function | tambear equivalent | status |
|---|---|---|
| `directional_stats` | — | MISSING |
| `circmean` | — | MISSING |
| `circvar` | — | MISSING |
| `circstd` | — | MISSING |

**MISSING — Circular/Directional Statistics family:**
All four are missing. These are moderate complexity — use trigonometric moments (mean resultant vector). Compose from `atan2`, `cos`, `sin`, `sqrt`. Critical for any angular/phase data.

---

## scipy.stats — Survival & Fitting

| scipy function | tambear equivalent | status |
|---|---|---|
| `fit` | — | MISSING |
| `ecdf` | `nonparametric::ecdf` | HAVE |
| `logrank` | — | MISSING |

**MISSING:**
- `fit` — MLE distribution fitting (general framework for all distributions). Complex — requires optimizer + distribution family.
- `logrank` — Log-rank test for survival curve comparison. Moderate — chi2-based test on hazard contributions.

---

## scipy.stats — Other

| scipy function | tambear equivalent | status |
|---|---|---|
| `gaussian_kde` | `nonparametric::kde` | HAVE |
| `ppcc_max` | — | MISSING |
| `ppcc_plot` | — | MISSING |
| `probplot` | — | MISSING |
| `sobol_indices` | — | MISSING |
| `combine_pvalues` | — | MISSING |
| `false_discovery_control` | `hypothesis::benjamini_hochberg` | HAVE |

**MISSING:**
- `ppcc_max` / `ppcc_plot` — Probability Plot Correlation Coefficient (for optimal Box-Cox lambda). Moderate.
- `probplot` — Q-Q plot data generation for any distribution. Trivial — quantile matching.
- `sobol_indices` — Variance-based sensitivity analysis (Saltelli method). Complex — requires quasi-Monte Carlo sampling.
- `combine_pvalues` — Meta-analytic p-value combination (Fisher, Pearson, Tippett, Stouffer, Mudholkar-George). Moderate — formula per method.

---

## scipy.linalg

| scipy function | tambear equivalent | status |
|---|---|---|
| `inv` | `linear_algebra::inv` | HAVE |
| `solve` | `linear_algebra::solve` | HAVE |
| `solve_banded` | — | MISSING |
| `solveh_banded` | — | MISSING |
| `solve_circulant` | — | MISSING |
| `solve_triangular` | `linear_algebra::forward_solve` + `back_solve_transpose` | PARTIAL |
| `solve_toeplitz` | — | MISSING |
| `matmul_toeplitz` | — | MISSING |
| `det` | `linear_algebra::det` | HAVE |
| `norm` | `linear_algebra::norm_fro` + `norm_inf` + `norm_1` | PARTIAL |
| `lstsq` | `linear_algebra::lstsq` | HAVE |
| `pinv` | `linear_algebra::pinv` | HAVE |
| `pinvh` | — | MISSING |
| `khatri_rao` | — | MISSING |
| `orthogonal_procrustes` | — | MISSING |
| `matrix_balance` | — | MISSING |
| `subspace_angles` | — | MISSING |
| `bandwidth` | — | MISSING |
| `eig` | `linear_algebra::sym_eigen` | PARTIAL |
| `eigvals` | — | MISSING |
| `eigh` | `linear_algebra::sym_eigen` | HAVE |
| `eig_banded` | — | MISSING |
| `eigh_tridiagonal` | — | MISSING |
| `lu` | `linear_algebra::lu` | HAVE |
| `lu_solve` | `linear_algebra::lu_solve` | HAVE |
| `svd` | `linear_algebra::svd` | HAVE |
| `svdvals` | — | MISSING |
| `diagsvd` | — | MISSING |
| `orth` | `linear_algebra::gram_schmidt` | PARTIAL |
| `null_space` | — | MISSING |
| `ldl` | — | MISSING |
| `cholesky` | `linear_algebra::cholesky` | HAVE |
| `cho_solve` | `linear_algebra::cholesky_solve` | HAVE |
| `polar` | — | MISSING |
| `qr` | `linear_algebra::qr` | HAVE |
| `qr_multiply` | — | MISSING |
| `qr_update` | — | MISSING |
| `qr_delete` | — | MISSING |
| `qr_insert` | — | MISSING |
| `rq` | — | MISSING |
| `qz` | — | MISSING |
| `schur` | — | MISSING |
| `hessenberg` | — | MISSING |
| `cossin` | — | MISSING |
| `expm` | `linear_algebra::matrix_exp` | HAVE |
| `logm` | `linear_algebra::matrix_log` | HAVE |
| `sqrtm` | `linear_algebra::matrix_sqrt` | HAVE |
| `cosm/sinm/tanm` | — | MISSING |
| `coshm/sinhm/tanhm` | — | MISSING |
| `signm` | — | MISSING |
| `funm` | — | MISSING |
| `expm_frechet` | — | MISSING |
| `fractional_matrix_power` | — | MISSING |
| `solve_sylvester` | — | MISSING |
| `solve_continuous_are` | — | MISSING |
| `solve_discrete_are` | — | MISSING |
| `solve_continuous_lyapunov` | — | MISSING |
| `solve_discrete_lyapunov` | — | MISSING |
| `clarkson_woodruff_transform` | — | MISSING |
| `block_diag` | — | MISSING |
| `circulant` | — | MISSING |
| `companion` | — | MISSING |
| `convolution_matrix` | — | MISSING |
| `hadamard` | — | MISSING |
| `hankel` | — | MISSING |
| `hilbert` (matrix) | — | MISSING |
| `toeplitz` | — | MISSING |

**Key MISSING linalg groups:**

1. **Banded/structured solvers** (solve_banded, solveh_banded, solve_toeplitz, solve_circulant): Moderate each — exploit matrix structure. We have tridiagonal, not general banded.
2. **General eigendecomposition** (eig for non-symmetric): Complex — QR iteration algorithm needed.
3. **Factorizations** (LDL, Schur, QZ, RQ, Hessenberg, polar): Moderate to complex each.
4. **Rank-k updates** (qr_update, qr_delete, qr_insert): Moderate — Givens rotations.
5. **Matrix equations** (Sylvester, Lyapunov, Riccati): Complex — these are critical for control theory.
6. **Matrix functions** (funm, matrix trig/hyperbolic, signm, fractional power, Frechet derivative): Moderate — Schur decomposition based.
7. **Special matrix constructors** (circulant, Hankel, Toeplitz, Hadamard, Hilbert): Trivial each.
8. **Null space, subspace angles, Procrustes, Khatri-Rao**: Moderate each.
9. **Clarkson-Woodruff sketch**: Moderate — random sparse projection.

---

## scipy.signal

| scipy function | tambear equivalent | status |
|---|---|---|
| `convolve` | `signal_processing::convolve` | HAVE |
| `correlate` | `signal_processing::cross_correlate` | HAVE |
| `fftconvolve` | — | MISSING |
| `oaconvolve` | — | MISSING |
| `convolve2d` | — | MISSING |
| `correlate2d` | — | MISSING |
| `hilbert` | `signal_processing::hilbert` | HAVE |
| `lfilter` | — | MISSING |
| `filtfilt` | — | MISSING |
| `sosfilt` | — | MISSING |
| `sosfiltfilt` | — | MISSING |
| `savgol_filter` | `signal_processing::savgol_filter` | HAVE |
| `medfilt` | `signal_processing::median_filter` | HAVE |
| `wiener` | — | MISSING |
| `decimate` | — | MISSING |
| `detrend` | — | MISSING |
| `resample` | `signal_processing::regularize_interp` | PARTIAL |
| `resample_poly` | — | MISSING |
| `firwin` | `signal_processing::fir_lowpass/highpass/bandpass` | PARTIAL |
| `freqz` | — | MISSING |
| `iirfilter` | `signal_processing::butterworth_lowpass_cascade` | PARTIAL |
| `butter` | — | MISSING |
| `periodogram` | `signal_processing::periodogram` | HAVE |
| `welch` | `signal_processing::welch` | HAVE |
| `csd` | `spectral::cross_spectral` | HAVE |
| `coherence` | — | MISSING |
| `spectrogram` | `signal_processing::spectrogram` | HAVE |
| `lombscargle` | `spectral::lomb_scargle` | HAVE |
| `stft` | `signal_processing::stft` | HAVE |
| `istft` | — | MISSING |
| `find_peaks` | — | MISSING |
| `peak_prominences` | — | MISSING |
| `peak_widths` | — | MISSING |
| `chirp` | — | MISSING |
| `gausspulse` | — | MISSING |
| `sawtooth` | — | MISSING |
| `square` | — | MISSING |
| `czt` | — | MISSING |
| `zoom_fft` | — | MISSING |

**Key MISSING signal groups:**

1. **2D convolution/correlation** (convolve2d, correlate2d): Moderate — extend 1D FFT-based approach.
2. **OA-overlap-add convolution** (oaconvolve): Moderate — partition-based FFT convolution.
3. **Forward digital filtering** (lfilter, filtfilt, sosfilt, sosfiltfilt): Moderate — IIR filter application. We have FIR only.
4. **IIR design** (butter, cheby1/2, ellip, bessel, full iirfilter): Complex — analog prototype + bilinear transform. We have Butterworth only.
5. **Frequency response** (freqz, freqs, sosfreqz, bode): Moderate — evaluate transfer function on unit circle.
6. **Peak finding** (find_peaks, peak_prominences, peak_widths): Moderate — local maxima with configurable conditions.
7. **Waveform generators** (chirp, gausspulse, sawtooth, square, sweep_poly): Trivial each.
8. **Inverse STFT** (istft): Moderate — overlap-add reconstruction.
9. **Coherence**: Trivial — |csd|^2 / (psd_x * psd_y).
10. **CZT / Zoom FFT**: Moderate — generalized z-transform evaluation.
11. **Wiener filter**: Moderate — power spectrum estimation + Wiener-Hopf.
12. **Decimate, detrend**: Trivial–moderate.

---

## scipy.fft

| scipy function | tambear equivalent | status |
|---|---|---|
| `fft` | `signal_processing::fft` | HAVE |
| `ifft` | `signal_processing::ifft` | HAVE |
| `fft2` | `signal_processing::fft2d` | HAVE |
| `ifft2` | — | MISSING |
| `fftn` | — | MISSING |
| `ifftn` | — | MISSING |
| `rfft` | `signal_processing::rfft` | HAVE |
| `irfft` | `signal_processing::irfft` | HAVE |
| `rfft2/irfft2/rfftn/irfftn` | — | MISSING |
| `hfft/ihfft/hfft2/ihfft2/hfftn/ihfftn` | — | MISSING |
| `dct` | `signal_processing::dct2` + `dct3` | PARTIAL |
| `idct` | — | MISSING |
| `dst` | — | MISSING |
| `idst` | — | MISSING |
| `fht/ifht` | — | MISSING |
| `fftshift/ifftshift/fftfreq/rfftfreq` | — | MISSING |
| `next_fast_len` | `signal_processing::next_pow2` | PARTIAL |

**Key MISSING fft groups:**

1. **Multi-dimensional inverse transforms** (ifft2, ifftn): Moderate — extend existing pattern.
2. **Hermitian transforms** (hfft family): Moderate — special case of FFT for Hermitian input.
3. **Real 2D/N-D transforms** (rfft2, rfftn etc): Moderate.
4. **DST all types** (dst, idst, dstn, idstn): Moderate — DCT variants.
5. **DCT types 1 and 4, and their inverses** (idct, dctn, idctn): Moderate.
6. **Fast Hankel Transform** (fht, ifht): Complex — bessel function-based transform.
7. **Frequency utilities** (fftshift, fftfreq, rfftfreq): Trivial.

---

## scipy.optimize

| scipy function | tambear equivalent | status |
|---|---|---|
| `minimize_scalar` | `numerical::bisection`, `optimization::golden_section` | PARTIAL |
| `minimize` | `optimization::gradient_descent`, `adam`, `lbfgs`, `nelder_mead` | PARTIAL |
| `basinhopping` | — | MISSING |
| `brute` | — | MISSING |
| `differential_evolution` | — | MISSING |
| `shgo` | — | MISSING |
| `dual_annealing` | — | MISSING |
| `direct` | — | MISSING |
| `least_squares` | — | MISSING |
| `nnls` | — | MISSING |
| `lsq_linear` | `multivariate::lasso` (regularized) | PARTIAL |
| `isotonic_regression` | — | MISSING |
| `curve_fit` | — | MISSING |
| `root_scalar` | `numerical::bisection`, `newton`, `secant`, `brent` | HAVE |
| `root` | — | MISSING |
| `linprog` | — | MISSING |
| `milp` | — | MISSING |
| `linear_sum_assignment` | — | MISSING |
| `quadratic_assignment` | — | MISSING |
| `approx_fprime` | `numerical::derivative_central` | HAVE |
| `check_grad` | — | MISSING |
| `fixed_point` | `numerical::fixed_point` | HAVE |

**Key MISSING optimize groups:**

1. **Global optimizers** (basinhopping, differential_evolution, dual_annealing, shgo, direct, brute): Moderate to complex each — stochastic/meta-heuristic methods.
2. **Nonlinear least squares** (least_squares, curve_fit): Moderate — Levenberg-Marquardt or trust-region.
3. **Constrained LS** (nnls, lsq_linear, isotonic_regression): Moderate — NNLS = active set, isotonic = pool adjacent violators.
4. **Multivariate root finding** (root): Complex — Newton/Krylov/Broyden for systems.
5. **Linear programming** (linprog, milp): Complex — simplex and interior-point methods.
6. **Assignment problems** (linear_sum_assignment, quadratic_assignment): Moderate to complex — Hungarian algorithm.

---

## scipy.interpolate

| scipy function | tambear equivalent | status |
|---|---|---|
| `CubicSpline` | `interpolation::natural_cubic_spline` | HAVE |
| `PchipInterpolator` | `interpolation::pchip` | HAVE |
| `Akima1DInterpolator` | `interpolation::akima` | HAVE |
| `BarycentricInterpolator` | `interpolation::barycentric_rational` | HAVE |
| `KroghInterpolator` | `interpolation::newton_divided_diff` + `newton_eval` | HAVE |
| `CubicHermiteSpline` | `interpolation::monotone_hermite` | HAVE |
| `FloaterHormannInterpolator` | — | MISSING |
| `BSpline` | `interpolation::bspline_eval` | HAVE |
| `make_interp_spline` | — | MISSING |
| `make_lsq_spline` | — | MISSING |
| `make_smoothing_spline` | — | MISSING |
| `PPoly` | — | MISSING |
| `BPoly` | — | MISSING |
| `RBFInterpolator` | `interpolation::rbf_interpolate` | HAVE |
| `LinearNDInterpolator` | — | MISSING |
| `NearestNDInterpolator` | — | MISSING |
| `CloughTocher2DInterpolator` | — | MISSING |
| `RegularGridInterpolator` | — | MISSING |
| `NdBSpline` | — | MISSING |
| `AAA` | — | MISSING |
| `splrep` / `splev` etc | — | MISSING |
| `pade` | `interpolation::pade` | HAVE |
| `lagrange` | `interpolation::lagrange` | HAVE |
| `approximate_taylor_polynomial` | — | MISSING |
| `interpn` / `griddata` | — | MISSING |

**Key MISSING interpolate groups:**

1. **Floater-Hormann rational interpolation**: Moderate — barycentric weights computed differently.
2. **FITPACK wrappers** (splrep, splev, splint etc): Complex — we have B-spline eval, need full fitting machinery.
3. **Multivariate interpolators** (LinearND, NearestND, CloughTocher, RegularGrid, NdBSpline): Complex — require spatial search structures.
4. **Smoothing splines** (make_smoothing_spline, make_lsq_spline): Moderate — optimization-based.
5. **AAA rational approximation**: Complex — adaptive Antoulas-Anderson algorithm.
6. **griddata / interpn**: Moderate — wrappers over LinearND/RegularGrid.
7. **Taylor polynomial approximation**: Trivial — Horner evaluation.

---

## scipy.integrate

| scipy function | tambear equivalent | status |
|---|---|---|
| `quad` | `numerical::adaptive_simpson` | PARTIAL |
| `dblquad` | — | MISSING |
| `tplquad` | — | MISSING |
| `nquad` | — | MISSING |
| `tanhsinh` | — | MISSING |
| `fixed_quad` | `numerical::gauss_legendre_5` | PARTIAL |
| `newton_cotes` | — | MISSING |
| `trapezoid` | `numerical::trapezoid` | HAVE |
| `cumulative_trapezoid` | — | MISSING |
| `simpson` | `numerical::simpson` | HAVE |
| `cumulative_simpson` | — | MISSING |
| `romb` | — | MISSING |
| `nsum` | — | MISSING |
| `solve_ivp` | `numerical::rk45` | PARTIAL |
| `odeint` | `numerical::rk4_system` | PARTIAL |
| `solve_bvp` | — | MISSING |
| `qmc_quad` | — | MISSING |
| `lebedev_rule` | — | MISSING |
| `cubature` | — | MISSING |

**Key MISSING integrate groups:**

1. **Double/triple/N-d quadrature** (dblquad, tplquad, nquad): Moderate — nested adaptive quadrature.
2. **Tanh-sinh quadrature** (tanhsinh): Moderate — variable substitution for endpoint singularities.
3. **Gauss-Legendre with n-point rules** (fixed_quad): Partial — we have 5-point only, need n-point from `special_functions::gauss_legendre_nodes_weights`.
4. **Newton-Cotes rules**: Trivial — weighted sum formulas.
5. **Cumulative versions** (cumulative_trapezoid, cumulative_simpson): Trivial — prefix sum versions.
6. **Romberg integration** (romb): Moderate — Richardson extrapolation of trapezoid estimates.
7. **Numerical summation** (nsum): Complex — Euler-Maclaurin, Levin transforms (partial: series_accel has pieces).
8. **BVP solver** (solve_bvp): Complex — collocation method.
9. **Quasi-Monte Carlo quadrature** (qmc_quad): Moderate — Sobol/Halton sequences + Monte Carlo.
10. **Lebedev quadrature on sphere** (lebedev_rule): Complex — precomputed rules for spherical integrals.
11. **Multidimensional cubature** (cubature): Complex — adaptive N-d integration.

---

## scipy.special

| scipy function | tambear equivalent | status |
|---|---|---|
| `erf` | `special_functions::erf` | HAVE |
| `erfc` | `special_functions::erfc` | HAVE |
| `erfinv` | — | MISSING |
| `erfcinv` | — | MISSING |
| `erfcx` | — | MISSING |
| `erfi` | — | MISSING |
| `wofz` | — | MISSING |
| `dawsn` | — | MISSING |
| `fresnel` | — | MISSING |
| `voigt_profile` | — | MISSING |
| `gamma` | `special_functions::gamma` | HAVE |
| `gammaln` | `special_functions::log_gamma` | HAVE |
| `loggamma` | `special_functions::log_gamma` | HAVE |
| `gammasgn` | — | MISSING |
| `gammainc` | `special_functions::regularized_gamma_p` | HAVE |
| `gammaincc` | `special_functions::regularized_gamma_q` | HAVE |
| `gammaincinv` | — | MISSING |
| `gammainccinv` | — | MISSING |
| `beta` | `special_functions::log_beta` (log form) | PARTIAL |
| `betaln` | `special_functions::log_beta` | HAVE |
| `betainc` | `special_functions::regularized_incomplete_beta` | HAVE |
| `betaincinv` | — | MISSING |
| `psi` / `digamma` | `special_functions::digamma` | HAVE |
| `polygamma` | `special_functions::trigamma` (order 1 only) | PARTIAL |
| `multigammaln` | — | MISSING |
| `poch` | — | MISSING |
| `airy` | — | MISSING |
| `jv` | `special_functions::bessel_jn` | PARTIAL |
| `j0/j1` | `special_functions::bessel_j0` + `bessel_j1` | HAVE |
| `yn/yv` | — | MISSING |
| `iv/ive` | `special_functions::bessel_i0` + `bessel_i1` | PARTIAL |
| `kn/kv` | — | MISSING |
| `spherical_jn` | — | MISSING |
| `struve` | — | MISSING |
| `legendre_p` | `special_functions::legendre_p` | HAVE |
| `lpmv` | — | MISSING |
| `sph_harm_y` | — | MISSING |
| `eval_legendre` | `special_functions::legendre_p` | HAVE |
| `eval_chebyt` | `special_functions::chebyshev_t` | HAVE |
| `eval_chebyu` | `special_functions::chebyshev_u` | HAVE |
| `eval_hermite` | `special_functions::hermite_he` | HAVE |
| `eval_laguerre` | `special_functions::laguerre_l` | HAVE |
| `eval_gegenbauer` | — | MISSING |
| `eval_jacobi` | — | MISSING |
| `hyp2f1` | — | MISSING |
| `hyp1f1` | — | MISSING |
| `hyperu` | — | MISSING |
| `hyp0f1` | — | MISSING |
| `ellipj` | — | MISSING |
| `ellipk` | — | MISSING |
| `ellipe` | — | MISSING |
| `elliprc/rd/rf/rg/rj` | — | MISSING |
| `lambertw` | — | MISSING |
| `wrightomega` | — | MISSING |
| `zeta` | — | MISSING |
| `zetac` | — | MISSING |
| `logsumexp` | `numerical::log_sum_exp` | HAVE |
| `logit` | `special_functions::logit` | HAVE |
| `expit` | `special_functions::logistic` | HAVE |
| `softmax` | `special_functions::softmax` | HAVE |
| `log_softmax` | `special_functions::log_softmax` | HAVE |
| `xlogy` | — | MISSING |
| `xlog1py` | — | MISSING |
| `entr` | `information_theory::p_log_p` | HAVE |
| `rel_entr` | `information_theory::p_log_p_over_q` | HAVE |
| `kl_div` | `information_theory::kl_divergence` | HAVE |
| `huber` | — | MISSING |
| `pseudo_huber` | — | MISSING |
| `comb` | — | MISSING |
| `perm` | — | MISSING |
| `stirling2` | — | MISSING |
| `bernoulli` | — | MISSING |
| `euler` | — | MISSING |
| `expn` | — | MISSING |
| `factorial` | `complexity::factorial` | HAVE |
| `factorial2` | — | MISSING |
| `sinc` | — | MISSING |
| `spence` | — | MISSING |
| `cbrt` | — | MISSING |
| `mathieu_a/b` | — | MISSING |
| `mathieu_cem/sem` | — | MISSING |
| `pbdv/pbvv` | — | MISSING |
| `pro_ang1/rad1` etc (spheroidal) | — | MISSING |
| `kelvin/ber/bei/ker/kei` | — | MISSING |
| `ndtr` | `special_functions::normal_cdf` | HAVE |
| `ndtri` | `special_functions::normal_quantile` | HAVE |
| `stdtr` | `special_functions::t_cdf` | HAVE |
| `chdtr` | `special_functions::chi2_cdf` | HAVE |
| `fdtr` | `special_functions::f_cdf` | HAVE |
| `pdtr` | `special_functions::poisson_cdf` | HAVE |
| `bdtr` | `special_functions::binomial_cdf` | HAVE |
| `roots_legendre` | `special_functions::gauss_legendre_nodes_weights` | HAVE |
| `roots_chebyt/u` | — | MISSING |
| `owens_t` | — | MISSING |
| `tklmbda` | — | MISSING |
| `agm` | — | MISSING |

**Key MISSING special groups:**

1. **Error function variants** (erfinv, erfcinv, erfcx, erfi, Faddeeva wofz, Dawson dawsn): Moderate each — series/rational approximations.
2. **Fresnel integrals** (fresnel): Moderate — power series + asymptotic.
3. **Voigt profile**: Moderate — convolution of Gaussian and Lorentzian (real part of wofz).
4. **Incomplete gamma inverse** (gammaincinv, gammainccinv): Moderate — Newton iteration on regularized_gamma.
5. **Beta function inverse** (betaincinv): Moderate — Newton on regularized_incomplete_beta.
6. **Polygamma order > 1**: Moderate — extend trigamma to arbitrary order.
7. **Bessel Y, K functions** (yn, yv, kn, kv): Moderate — series + recursion.
8. **Spherical Bessel functions**: Moderate.
9. **Struve functions**: Moderate — power series.
10. **Hypergeometric functions** (hyp2f1, hyp1f1, hyperu, hyp0f1): Complex — many special cases.
11. **Elliptic integrals and functions** (ellipj, ellipk, ellipe, Carlson forms): Moderate — AGM iteration.
12. **Lambert W** (lambertw): Moderate — Halley iteration.
13. **Riemann zeta** (zeta, zetac): Moderate — Euler-Maclaurin + Bernoulli numbers.
14. **Gegenbauer, Jacobi polynomials**: Moderate — recurrence relations.
15. **Associated Legendre, spherical harmonics**: Moderate — recursion.
16. **Parabolic cylinder functions, Mathieu functions, spheroidal wave functions, Kelvin functions**: Complex — ODEs with special structure.
17. **Combinatorial** (comb, perm, stirling2): Trivial.
18. **Bernoulli numbers, Euler numbers**: Moderate — recurrence.
19. **sinc, cbrt, expn, spence, agm**: Trivial to moderate.
20. **xlogy, xlog1py, huber, pseudo_huber**: Trivial — numerically safe wrappers.
21. **Owen's T function**: Moderate — integration of bivariate normal.

---

## scipy.spatial.distance

| scipy function | tambear equivalent | status |
|---|---|---|
| `euclidean` | `spatial::euclidean_2d` (2D only) | PARTIAL |
| `cityblock` | — | MISSING |
| `chebyshev` | — | MISSING |
| `minkowski` | — | MISSING |
| `cosine` | `information_theory::cosine_similarity` | HAVE |
| `correlation` | `nonparametric::pearson_r` | PARTIAL |
| `mahalanobis` | `multivariate::mahalanobis_distances` | PARTIAL |
| `seuclidean` | — | MISSING |
| `sqeuclidean` | — | MISSING |
| `braycurtis` | — | MISSING |
| `canberra` | — | MISSING |
| `jensenshannon` | `information_theory::js_divergence` | HAVE |
| `dice` | — | MISSING |
| `hamming` | `nonparametric::levenshtein` | PARTIAL |
| `jaccard` | — | MISSING |
| `rogerstanimoto` | — | MISSING |
| `russellrao` | — | MISSING |
| `sokalsneath` | — | MISSING |
| `yule` | — | MISSING |
| `pdist` | — | MISSING |
| `cdist` | — | MISSING |
| `directed_hausdorff` | — | MISSING |

**Key MISSING distance groups:**

1. **Vector distance primitives** (cityblock, chebyshev, minkowski, seuclidean, sqeuclidean, braycurtis, canberra): Trivial each — single formulas.
2. **Boolean distances** (dice, jaccard, rogerstanimoto, russellrao, sokalsneath, yule): Trivial each — set-intersection formulas.
3. **pdist / cdist** — Pairwise distance matrices for any metric. Moderate — n^2 accumulate pattern.
4. **directed_hausdorff** — Directed Hausdorff distance (max of nearest-neighbor distances). Moderate — O(n*m) worst case.

---

## scipy.spatial (geometry)

| scipy function | tambear equivalent | status |
|---|---|---|
| `ConvexHull` | `spatial::convex_hull_2d` | PARTIAL |
| `Delaunay` | — | MISSING |
| `Voronoi` | — | MISSING |
| `SphericalVoronoi` | — | MISSING |
| `KDTree / cKDTree` | — | MISSING |
| `distance_matrix` | — | MISSING |
| `procrustes` | — | MISSING |
| `geometric_slerp` | — | MISSING |
| `HalfspaceIntersection` | — | MISSING |

**Key MISSING geometry:**
- **KDTree**: Complex but critical — spatial indexing for k-NN, range queries.
- **Delaunay / Voronoi**: Complex — computational geometry algorithms.
- **General distance_matrix**: Moderate — pdist without metric restriction.
- **Procrustes**: Moderate — SVD-based alignment (relates to `linear_algebra::svd` which we have).
- **Spherical interpolation** (geometric_slerp): Moderate.

---

## scipy.cluster

| scipy function | tambear equivalent | status |
|---|---|---|
| `linkage` | `clustering::hierarchical_clustering` | PARTIAL |
| `single/complete/average/ward/weighted/centroid/median` | `clustering::hierarchical_clustering` (Ward) | PARTIAL |
| `fcluster` | — | MISSING |
| `dendrogram` | — | MISSING |
| `cophenet` | — | MISSING |
| `inconsistent` | — | MISSING |
| `leaves_list` | — | MISSING |
| `optimal_leaf_ordering` | — | MISSING |
| `DisjointSet` | `clustering::uf_new/uf_find/uf_union` | HAVE |

**Key MISSING cluster:**
- **Full linkage family** (single, complete, average, weighted, centroid, median): Moderate — we have Ward only.
- **Flat cluster extraction** (fcluster): Moderate — cut dendrogram at height or k.
- **Cophenetic correlation** (cophenet): Moderate — validate hierarchical structure.
- **Optimal leaf ordering**: Complex — TSP-like reordering.

---

## Priority Summary — Top Gaps by Impact

### Critical / High Priority (used everywhere)

| Function | Module | Complexity | Key Primitives |
|---|---|---|---|
| `erfinv` | special_functions | Moderate | series inversion of erf |
| `betaincinv` | special_functions | Moderate | Newton on regularized_incomplete_beta |
| `gammaincinv` | special_functions | Moderate | Newton on regularized_gamma_p |
| General `ks_1samp(cdf)` | nonparametric | Moderate | ecdf + any CDF function |
| `pdist` / `cdist` | spatial/multivariate | Moderate | accumulate over pairs |
| `theilslopes` | nonparametric | Moderate | sort + median-of-medians |
| `logrank` test | survival | Moderate | chi2 + hazard accumulation |
| Circular statistics (circmean/var/std) | descriptive | Moderate | trig moments |
| `combine_pvalues` | hypothesis | Moderate | Fisher/Stouffer/chi2 |
| `zscore` / `zmap` | descriptive | Trivial | (x - mean) / std |
| `sem` | descriptive | Trivial | std / sqrt(n) |
| `mode` | descriptive | Trivial | argmax of frequency counts |
| `gstd` | descriptive | Trivial | exp(std(log(x))) |
| `trim1`/`trimboth` | descriptive | Trivial | sort + slice |
| `sigmaclip` | robust | Trivial | iterative outlier removal |
| `yeojohnson` | descriptive | Moderate | Box-Cox extension to negatives |
| `bartlett` test | hypothesis | Moderate | chi2-based variance test |
| `fligner` test | hypothesis | Moderate | rank-based variance test |
| `ranksums` | nonparametric | Trivial | presentation alias for mann_whitney |
| `brunnermunzel` | nonparametric | Moderate | rank-based, unequal variances |
| `cramervonmises` (1-samp) | nonparametric | Moderate | ECDF vs CDF sum-of-squares |
| `cramervonmises_2samp` | nonparametric | Moderate | ECDF vs ECDF |
| `chatterjeexi` | nonparametric | Moderate | 2021 rank correlation |
| `somersd` | nonparametric | Moderate | concordant/discordant pairs |
| `power_divergence` | hypothesis | Moderate | G-test, CR stat, Freeman-Tukey |
| Boolean distances (jaccard, dice, hamming...) | distributional_distances | Trivial | set formulas |
| Vector distances (cityblock, chebyshev...) | distributional_distances | Trivial | Lp formulas |
| `hyp2f1` / `hyp1f1` | special_functions | Complex | Gauss hypergeometric |
| Elliptic integrals (ellipk, ellipe, ellipj) | special_functions | Moderate | AGM iteration |
| `lambertw` | special_functions | Moderate | Halley iteration |
| `zeta` | special_functions | Moderate | Euler-Maclaurin |
| `lfilter`/`sosfilt` (IIR apply) | signal_processing | Moderate | direct form II |
| Full IIR design (butter/cheby/ellip/bessel) | signal_processing | Complex | analog prototype + bilinear |
| `curve_fit` | optimization | Moderate | LM algorithm |
| `nnls` | optimization | Moderate | active-set |
| `linprog` | optimization | Complex | simplex/interior-point |
| `differential_evolution` | optimization | Moderate | metaheuristic |
| **linalg: Sylvester, Lyapunov, Riccati** | linear_algebra | Complex | Schur-based solvers |
| **linalg: Schur decomposition** | linear_algebra | Complex | QR iteration |
| **linalg: general eig** (non-symmetric) | linear_algebra | Complex | QR iteration |
| **linalg: LDL factorization** | linear_algebra | Moderate | pivoted Bunch-Kaufman |
| KDTree | spatial | Complex | kd-tree construction/query |
| Full linkage family (single/complete/average) | clustering | Moderate | distance accumulation |
| `fcluster` | clustering | Moderate | dendrogram cut |
| Multivariate distributions (multivariate_normal, Wishart etc.) | special_functions | Complex | matrix determinant + Cholesky |
| Most continuous distributions (~80 missing) | special_functions | Trivial-Moderate per | CDF/PDF/quantile from special funcs |

---

## Count Summary

| Module | Total scipy functions | Tambear HAVE | Tambear PARTIAL | MISSING |
|---|---|---|---|---|
| scipy.stats (tests + descriptive) | ~120 | ~45 | ~8 | ~67 |
| scipy.stats distributions (continuous) | ~100 | ~10 | ~2 | ~88 |
| scipy.stats distributions (discrete) | ~20 | ~5 | 0 | ~15 |
| scipy.stats distributions (multivariate) | ~18 | 0 | 0 | ~18 |
| scipy.linalg | ~80 | ~15 | ~5 | ~60 |
| scipy.signal | ~100 | ~20 | ~5 | ~75 |
| scipy.fft | ~35 | ~8 | ~2 | ~25 |
| scipy.optimize | ~40 | ~8 | ~3 | ~29 |
| scipy.interpolate | ~50 | ~12 | ~1 | ~37 |
| scipy.integrate | ~25 | ~5 | ~3 | ~17 |
| scipy.special | ~250 | ~35 | ~8 | ~207 |
| scipy.spatial.distance | ~25 | ~4 | ~3 | ~18 |
| scipy.spatial (geometry) | ~10 | ~1 | 0 | ~9 |
| scipy.cluster | ~20 | ~4 | ~2 | ~14 |
| **TOTAL** | **~893** | **~172** | **~42** | **~679** |

Tambear has approximately 19% of the scipy API (by function count), with another 5% partially covered. ~76% is missing — almost all of it is buildable from tambear's existing primitives with no new dependencies.
