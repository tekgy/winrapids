# R Stats Ecosystem — Gap Analysis vs Tambear

**Date**: 2026-04-10
**Method**: Grepped all `pub fn` in `crates/tambear/src/` and matched against R's core stats ecosystem.

---

## What tambear already has (confirmed by grep)

Noting these to establish coverage baseline before listing gaps.

### base R `stats::`
- t-tests: `one_sample_t`, `two_sample_t`, `welch_t`, `paired_t`
- ANOVA: `one_way_anova`, `welch_anova`, Tukey HSD
- Chi-square: `chi2_goodness_of_fit`, `chi2_independence`
- Fisher exact: `fisher_exact`
- Wilcoxon / Mann-Whitney: `wilcoxon_signed_rank`, `mann_whitney_u`
- Kruskal-Wallis: `kruskal_wallis`; Dunn post-hoc: `dunn_test`
- KS tests: `ks_test_normal`, `ks_test_two_sample`
- Shapiro-Wilk: `shapiro_wilk`
- Correlation: `pearson_r`, `spearman`, `kendall_tau`, `partial_correlation`
- Linear models: `simple_linear_regression`, `ols_normal_equations`, `lstsq`, WLS
- Logistic regression: `logistic_regression`, `glm_fit` (Poisson + NegBinomial via IRLS)
- GLM families in `glm_fit`: `GlmFamily::Poisson`, `GlmFamily::NegativeBinomial`
- Power / sample size: `power_one_sample_t`, `power_two_sample_t`, `sample_size_*`
- Effect sizes: `cohens_d`, `hedges_g`, `glass_delta`, `eta_squared`, `cramers_v`
- Multiple comparison: `bonferroni`, `holm`, `benjamini_hochberg`
- Mediation / moderation: full implementations
- ACF / PACF: `acf`, `pacf`
- Ljung-Box / Box-Pierce: `ljung_box`, `box_pierce`
- Durbin-Watson: `durbin_watson`
- ADF test: `adf_test` (with MacKinnon critical values)
- KPSS test: `kpss_test`
- Phillips-Perron: `pp_test`, `phillips_perron_test`
- ARMA/ARIMA/auto.arima: `arma_fit`, `arima_fit`, `auto_arima`
- Exponential smoothing: `simple_exponential_smoothing`, `holt_linear`
- STL decomposition: `stl_decompose`
- Changepoint: `cusum_mean`, `cusum_binary_segmentation`, `pelt`, `bocpd`
- KM survival: `kaplan_meier`, `log_rank_test`, `cox_ph`, `grambsch_therneau_test`
- PCA / MDS / t-SNE / NMF: in `dim_reduction.rs`
- Factor analysis: `principal_axis_factoring`, `varimax`, `cronbachs_alpha`, `mcdonalds_omega`
- LDA / CCA / MANOVA / Hotelling: in `multivariate.rs`
- Mixed effects: `lme_random_intercept`, ICC variants
- Panel: FE/RE/FD/TWFE, Hausman test, 2SLS, DiD
- Robust: Huber/Bisquare/Hampel M-estimates, LTS, MCD, Qn/Sn scale
- Bayesian: `metropolis_hastings`, `bayesian_linear_regression`, `r_hat`, ESS
- IRT: 1PL/2PL/3PL, `ability_mle`, `ability_eap`, DIF
- KDE: `kde`, `kde_fft`, Silverman/Scott bandwidth
- Bootstrap: `bootstrap_percentile`, permutation tests
- Residual diagnostics: `breusch_pagan`, `cooks_distance`, VIF, Breusch-Godfrey
- Bayes factors: `bayes_factor_t_one_sample`, `bayes_factor_correlation`
- GARCH family: GARCH(1,1), EGARCH, GJR-GARCH, TGARCH, EWMA
- Markov chains / CTMC / HMM: fully implemented
- Kalman filter / RTS smoother / particle filter: fully implemented

---

## Gaps — Functions in R Ecosystem Not Yet in Tambear

Each entry notes: **what it computes**, **difficulty** (Low/Medium/High), and **which R package**.

---

### 1. GLM — Missing Link Functions and Families

**Package**: `stats::glm`

Current `glm_fit` supports Poisson and NegBinomial only with log link. R's `glm()` supports:

| R family/link | Status | What it computes | Difficulty |
|---|---|---|---|
| `gaussian(link="log")` | MISSING | Log-linear Gaussian (log link, identity variance) | Low |
| `gaussian(link="inverse")` | MISSING | Inverse link Gaussian | Low |
| `Gamma(link="log")` | MISSING | Gamma regression — strictly positive continuous outcomes | Medium |
| `Gamma(link="inverse")` | MISSING | Canonical Gamma link | Medium |
| `Gamma(link="identity")` | MISSING | Identity-link Gamma | Low |
| `inverse.gaussian(link="1/mu^2")` | MISSING | Inverse Gaussian regression (heavy right tail) | Medium |
| `quasi*` families | MISSING | Quasi-likelihood (over/underdispersion correction) | Medium |
| `binomial(link="probit")` | MISSING | Probit regression (normal CDF link) | Low |
| `binomial(link="cloglog")` | MISSING | Complementary log-log link (extreme value distribution) | Low |
| `binomial(link="cauchit")` | MISSING | Cauchy CDF link | Low |

**Notes**: The IRLS skeleton in `hypothesis.rs:glm_fit` is parameterizable. Adding these is essentially writing the variance function, link function, and derivative for each family. Probit is the most-used missing link. Gamma regression is critical for modeling positive continuous data (claims, durations).

---

### 2. Ordinal and Nominal Regression

**Package**: `MASS::polr`, `nnet::multinom`

| Function | What it computes | Difficulty |
|---|---|---|
| `polr` (proportional odds) | Ordinal logistic regression — cumulative link model with logit/probit/cloglog | High |
| `multinom` | Multinomial logistic regression — softmax over K categories | Medium |
| `clmm` (from `ordinal` package) | Cumulative link mixed model (ordinal + random effects) | High |

**Notes**: `polr` is widely used in psychometrics, clinical trials, survey analysis. Uses iterative weighted least squares with a cumulative logit link. Harder than binomial logistic because the likelihood involves differences of CDFs. `multinom` is sofmax regression — the forward pass is straightforward, gradient via cross-entropy.

---

### 3. Zero-Inflated and Hurdle Models

**Package**: `pscl::zeroinfl`, `pscl::hurdle`

| Function | What it computes | Difficulty |
|---|---|---|
| `zeroinfl` | ZIP / ZINB — mixture of point mass at zero and Poisson/NegBin | High |
| `hurdle` | Two-part model: binary for zero/nonzero + truncated count model | High |

**Notes**: Both require EM or joint likelihood optimization. ZIP is Poisson likelihood on nonzero counts mixed with a logistic model for the zero-inflation probability. Common in ecology, health data, insurance claims. Medium-high because the likelihood is non-standard but the pieces (logistic, Poisson log-lik) exist.

---

### 4. Survival Analysis — Missing Pieces

**Package**: `survival::survfit`, `survival::survreg`, `survival::coxme`

| Function | What it computes | Difficulty |
|---|---|---|
| `survreg` (Weibull AFT) | Parametric AFT model — Weibull, log-normal, log-logistic | High |
| `survreg` (exponential) | Exponential AFT — special case of Weibull with shape=1 | Medium |
| `survreg` (log-logistic) | Log-logistic AFT — non-monotone hazard | High |
| `coxme` | Cox PH with frailty / random effects | Very High |
| `survfit` CI methods | Log-log CIs for KM; Hall-Wellner; Equal Precision | Medium |
| Competing risks (cmprsk) | Cause-specific hazards; Fine-Gray subdistribution hazard | High |
| Nelson-Aalen estimator | Cumulative hazard estimator (alternative to KM) | Low |
| Breslow-day test | Homogeneity of log-odds ratios across strata | Medium |

**Notes**: `survreg` fits log(T) = X'β + σε, where ε has extreme value / logistic / normal distribution. Requires maximizing a likelihood with both uncensored and censored contributions. Nelson-Aalen is trivial — cumulative sum of d_i/n_i. The competing risks gap is the largest missing block in the survival module.

---

### 5. Nonparametric Tests — Missing

**Package**: `stats`, `coin`, `NSM3`

| Function | What it computes | Difficulty |
|---|---|---|
| `fligner.test` | Fligner-Killeen test for homogeneity of variances (nonparametric alt to Levene) | Low |
| `mood.test` | Mood two-sample test for equality of scale | Low |
| `ansari.test` | Ansari-Bradley test (scale difference between two groups) | Low |
| `var.test` | F-test for equality of two variances | Low |
| `bartlett.test` | Bartlett's test for equal variances across k groups | Low |
| `oneway.test` | Welch one-way test (already have `welch_anova`) | Already covered |
| `pairwise.wilcox.test` | All pairwise Mann-Whitney with multiple comparison correction | Low |
| `pairwise.t.test` | All pairwise t-tests with correction | Low |
| `jonckheere.test` | Jonckheere-Terpstra ordered alternatives test | Medium |
| `Page.test` | Page's test for ordered alternatives in a block design | Medium |

**Notes**: `bartlett.test` and `fligner.test` are commonly paired with ANOVA as variance homogeneity checks. Both straightforward — Bartlett needs log of pooled variance, Fligner ranks absolute deviations. `var.test` is just the ratio of sample variances under F distribution.

---

### 6. Regression Diagnostics — Missing

**Package**: `stats`, `car`

| Function | What it computes | Difficulty |
|---|---|---|
| `influence.measures` | Hat values, DFFITS, DFBETAS, covratio | Medium |
| `hatvalues` | Leverage = diag(H) = diag(X(X'X)⁻¹X') | Low |
| `dffits` | Standardized change in fitted values when obs removed | Low (needs hat) |
| `dfbetas` | Change in each coefficient when obs removed | Medium |
| `covratio` | Change in generalized variance of betas when obs removed | Medium |
| `outlierTest` (car) | Bonferroni-adjusted outlier test using studentized residuals | Low |
| `spreadLevelPlot` (car) | Spread-level plot for non-constant variance | Low |
| `avPlots` (car) | Added-variable plots (partial regression plots) | Medium |
| `ceresPlots` (car) | CERES plots for nonlinearity | Medium |
| `resettest` (lmtest) | RESET test for functional form misspecification | Low |

**Notes**: Hat values are the diagonal of H = X(X'X)⁻¹X'. DFFITS = studentized_resid_i * sqrt(h_ii / (1-h_ii)). DFBETAS = change in j-th coefficient divided by its SE when obs i deleted. These all follow directly from QR decomposition. The `car` package diagnostics are heavily used by applied statisticians.

---

### 7. Time Series — Missing Pieces

**Package**: `forecast`, `tseries`, `strucchange`

| Function | What it computes | Difficulty |
|---|---|---|
| `ets` (exponential smoothing state space) | ETS(E,T,S): error/trend/season — full Holt-Winters state space | High |
| `holt_winters` (full seasonal) | Triple exponential smoothing with multiplicative/additive seasonality | Medium |
| `tbats` | BATS/TBATS: Box-Cox + ARMA + Trigonometric + Seasonal | Very High |
| `nnetar` | Neural network AR model | High |
| `VAR` (vars package) | Vector autoregression — multivariate AR | High |
| `VECM` (tsDyn) | Vector error correction model | Very High |
| `arch.test` (tseries) | Lagrange multiplier ARCH test (already have `arch_lm_test`) | Already covered |
| `garch` (tseries) | Basic GARCH(p,q) (already have GARCH(1,1)) | Partially covered |
| `po.test` | Phillips-Ouliaris cointegration test | Medium |
| `ca.jo` (urca) | Johansen cointegration test | High |
| `grangertest` (lmtest) | Granger causality test | Low |
| `sctest` (strucchange) | Structural change test (Chow, CUSUM, MOSUM) | Medium |
| `efp` (strucchange) | Empirical fluctuation process for structural change | High |
| Lomb-Scargle periodogram | Unevenly-spaced time series spectral analysis | Medium |
| `stl` with outlier detection | Robust STL already covered (robust=true flag) | Already covered |
| `msts` | Multiple seasonal time series | High |

**Notes**: `grangertest` is just fitting two AR models (restricted/unrestricted) and comparing log-likelihoods via F-test — very easy given `ar_fit`. Johansen cointegration is the most important missing time series test for financial data — determines number of cointegrating vectors via eigenvalue decomposition of the error-correction system. Lomb-Scargle is critical for unevenly-sampled tick data.

---

### 8. Bootstrap and Resampling

**Package**: `boot`

| Function | What it computes | Difficulty |
|---|---|---|
| `boot` (basic/normal/BCA CI) | BCa (bias-corrected, accelerated) bootstrap CI | Medium |
| `boot.ci` types | Normal, basic, studentized, percentile, BCa CIs | Medium |
| `boot` (two-sample) | Bootstrap for two-sample statistics | Low |
| Jackknife | Delete-one jackknife estimator and bias/variance | Low |
| `cv.glm` | Cross-validation for GLM — leave-one-out and k-fold | Medium |
| `tsboot` | Block bootstrap for time series (moving blocks, stationary) | Medium |

**Notes**: BCa bootstrap requires the acceleration constant (from jackknife) and the bias-correction constant. Currently tambear has `bootstrap_percentile` but not BCa. BCa is the gold standard CI method. Block bootstrap for time series is important for dependent data — can't use iid bootstrap on autocorrelated series.

---

### 9. Mixed Effects — Missing

**Package**: `lme4`

| Function | What it computes | Difficulty |
|---|---|---|
| `lmer` (full) | LME with multiple random effects, crossed/nested, REML | Very High |
| `glmer` | Generalized LME (Poisson/Binomial random effects via Laplace/AGQ) | Very High |
| `nlmer` | Nonlinear mixed effects | Very High |
| `lmerTest` | Satterthwaite df for t/F in LME | High |
| `VarCorr` | Variance components from fitted LME | Medium (after lmer) |
| `ranef` | BLUPs — predicted random effects | Medium (after lmer) |
| `fixef` | Fixed effects from LME | Low (after lmer) |

**Notes**: Current `lme_random_intercept` covers one narrow case (one grouping factor, random intercept only). Full `lmer` requires the full PLS (penalized least squares) or sparse Cholesky machinery. This is one of the most complex gaps — lme4 is a very sophisticated package. The key challenge is handling crossed vs nested random effects and computing the sparse relative covariance factor.

---

### 10. Causal Inference — Missing

**Package**: `MatchIt`, `Synth`, `rdrobust`, `rdd`

| Function | What it computes | Difficulty |
|---|---|---|
| Optimal matching (MatchIt) | Optimal/genetic/full matching for propensity score matching | High |
| `synth` | Synthetic control method | High |
| `rdrobust` | Robust RD estimator with optimal bandwidth (already have basic `rdd_sharp`) | Medium |
| `rdbwselect` | Optimal bandwidth selection for RD | Medium |
| `rdplot` | RD plot with polynomial fit and confidence bands | Low |
| Instrumental variables (full 2SLS diagnostics) | First-stage F, Sargan, Wu-Hausman | Medium |
| `ivreg` (AER package) | 2SLS with robust SEs, diagnostics | Medium |
| Covariate balance (MatchIt) | Standardized mean differences, Love plot data | Low |

**Notes**: `rdrobust` implements the Calonico-Cattaneo-Titiunik optimal bandwidth — more principled than the naive bandwidth in current `rdd_sharp`. The synthetic control gap is significant for policy analysis.

---

### 11. Cluster Analysis — Missing

**Package**: `cluster`

| Function | What it computes | Difficulty |
|---|---|---|
| `pam` | Partitioning Around Medoids — k-medoids clustering | Medium |
| `clara` | CLARA: PAM on large datasets via subsampling | Medium |
| `fanny` | Fuzzy analysis clustering — soft membership | Medium |
| `agnes` | Agglomerative nesting with dissimilarity matrix | Low (have hierarchical) |
| `diana` | Divisive analysis clustering — top-down | Medium |
| `mclust` (package) | Model-based clustering via Gaussian mixture EM | High |
| `dbscan` (dbscan package) | DBSCAN (have via `spectral_clustering.rs`?) | Check |
| `hdbscan` | HDBSCAN — hierarchical density-based | High |
| `kmeans++` initialization | k-means++ seeded initialization | Low |
| Dunn index | Another internal validation index | Low |

**Notes**: k-medoids (PAM) is important when cluster centers must be actual data points (outlier robustness). Fuzzy clustering (FANNY) assigns probabilistic memberships rather than hard assignments. Model-based clustering (mclust) is the most widely used for statisticians and is a significant gap.

---

### 12. Density Estimation — Missing

**Package**: `stats`, `KernSmooth`, `ks`

| Function | What it computes | Difficulty |
|---|---|---|
| `bkde` (KernSmooth) | Binned kernel density estimator | Low |
| `bkde2D` | 2D binned KDE | Medium |
| `dpik` | Plug-in bandwidth selector (Sheather-Jones) | Medium |
| Multivariate KDE | Full d-dimensional KDE with bandwidth matrix | High |
| `locpoly` (KernSmooth) | Local polynomial regression | Medium |
| `loess` (stats) | Local polynomial regression smoother (weighted) | Medium |
| `smooth.spline` | Smoothing spline with GCV bandwidth | High |

**Notes**: `loess` is heavily used for nonparametric smoothing. It fits local polynomial regressions in a sliding window with tricube weights. `smooth.spline` minimizes RSS + lambda * integral of (f'')^2. Both are important for EDA and signal analysis. The Sheather-Jones plug-in bandwidth is the gold standard for univariate KDE (better than Silverman/Scott in practice).

---

### 13. Extreme Value Theory

**Package**: `evd`, `ismev`, `extRemes`

| Function | What it computes | Difficulty |
|---|---|---|
| GEV distribution | Generalized extreme value (Gumbel/Fréchet/Weibull family) | Medium |
| GPD | Generalized Pareto distribution (threshold excess) | Medium |
| `fgev` | Fit GEV by MLE | Medium |
| `fpot` / `fevd` | Fit GPD above threshold (POT method) | Medium |
| Return levels | T-year return levels from fitted GEV/GPD | Low (after fit) |
| Block maxima | Extract block maxima for GEV fitting | Low |
| Hill estimator | Already have `hill_estimator` | Covered |

**Notes**: GEV/GPD are the foundational distributions for extremes. The MLE for GEV has a tricky boundary (shape parameter near zero) requiring careful numerical treatment. Return levels are a simple transformation of quantiles. The GPD threshold selection (mean excess plot, stability plots) is also missing.

---

### 14. Copulas

**Package**: `copula`, `VineCopula`

| Function | What it computes | Difficulty |
|---|---|---|
| Gaussian copula | Copula from multivariate normal | Medium |
| t-copula | Copula from multivariate-t | Medium |
| Clayton copula | Archimedean copula with lower tail dependence | Medium |
| Gumbel copula | Archimedean copula with upper tail dependence | Medium |
| Frank copula | Archimedean copula (symmetric tails) | Medium |
| Joe copula | Archimedean with strong upper tail dependence | Medium |
| `fitCopula` | MLE fitting of copula parameters | High |
| Vine copulas (VineCopula) | Pair-copula construction via R-vine / C-vine / D-vine | Very High |
| Tail dependence coefficients | Lambda_U (upper), Lambda_L (lower) from copula | Low |

**Notes**: Copulas are essential for multivariate financial modeling — they decouple marginals from dependence structure. Archimedean copulas (Clayton, Gumbel, Frank) are one-parameter families with closed-form CDFs and are straightforward to implement. Vine copulas are a major undertaking but are the state of the art for high-dimensional dependence.

---

### 15. Multiple Testing — Missing Methods

**Package**: `stats`, `multtest`, `qvalue`

| Function | What it computes | Difficulty |
|---|---|---|
| `p.adjust(method="BY")` | Benjamini-Yekutieli (arbitrary dependence) | Low |
| `q-value` (Storey) | q-value / FDR via estimate of pi_0 | Medium |
| `mt.rawp2adjp` (multtest) | Multiple test procedures including Westfall-Young | High |
| `simes.test` | Simes' global test of combined p-values | Low |
| `hommel` | Hommel's step-down procedure | Medium |

**Notes**: Benjamini-Yekutieli (BY) is a minor extension of BH that controls FDR under arbitrary dependence. Storey's q-value is widely used in genomics. Simes' test is a simple but useful global test. All are low-medium difficulty.

---

### 16. Psychometrics / IRT — Missing Models

**Package**: `ltm`, `mirt`, `psych`

| Function | What it computes | Difficulty |
|---|---|---|
| GRM (Graded Response Model) | IRT for ordered polytomous items | High |
| PCM (Partial Credit Model) | IRT for partial credit polytomous items | High |
| GPCM (Generalized PCM) | PCM with discrimination parameters | High |
| Nominal response model | IRT for unordered multicategory responses | High |
| `cfa` (lavaan) | Confirmatory factor analysis | Very High |
| `sem` (lavaan) | Structural equation modeling | Very High |
| `mediation` (full SEM path) | Mediation via SEM with bootstrapped CI | High |
| `omega` (psych) | McDonald's omega (already have `mcdonalds_omega`) | Covered |
| `testlet` models | Testlet IRT for correlated items | Very High |

**Notes**: Current IRT covers 1PL/2PL/3PL (dichotomous). GRM/PCM/GPCM for polytomous items are major extensions. SEM (lavaan-equivalent) is a very large gap — it would require FIML, model-implied covariance structures, and a path grammar.

---

### 17. Spatial Statistics

**Package**: `sp`, `spdep`, `spatstat`

| Function | What it computes | Difficulty |
|---|---|---|
| Moran's I | Spatial autocorrelation statistic | Medium |
| Geary's C | Alternative spatial autocorrelation | Medium |
| `lm.morantest` | Moran test on regression residuals | Medium |
| Kriging (ordinary) | Spatial interpolation via variogram model | High |
| Variogram estimation | Empirical and theoretical variogram | High |
| Ripley's K function | Point pattern analysis (clustering vs regularity) | Medium |
| `ppp` / Poisson point process | Spatial point process modeling | High |

**Notes**: Moran's I and Geary's C are relatively simple (weighted spatial lag correlation). Kriging requires variogram fitting and solving a kriging system — substantial but well-defined. Spatial stats are less relevant to the market atlas use case but are in scope per CLAUDE.md.

---

### 18. Generalized Additive Models (GAM)

**Package**: `mgcv`

| Function | What it computes | Difficulty |
|---|---|---|
| `gam` | GAM with penalized smoothing splines | Very High |
| `bam` | GAM for large datasets (fast mgcv) | Very High |
| `gamm` | GAM with mixed effects | Very High |
| Tensor product smooths | 2D/3D smooth surfaces | Very High |
| `gam.check` | Basis dimension adequacy diagnostics | High |

**Notes**: GAMs are a major gap. They require: automatic GCV/REML smoothing parameter selection, B-spline or thin-plate spline bases, iterative PIRLS fitting (like GLM but with penalized likelihood), and wood-style efficient QR approaches. This is probably the largest single package gap in terms of complexity.

---

### 19. Distributions — Missing Density/CDF/Quantile

**Package**: `stats`

R's `d/p/q/r` convention for every named distribution. Tambear has many; missing:

| Distribution | Missing functions | Difficulty |
|---|---|---|
| Hypergeometric | `dhyper`, `phyper`, `qhyper` | Low |
| Geometric | `dgeom`, `pgeom`, `qgeom` (already have via NegBin?) | Low |
| Logistic | `dlogis`, `plogis`, `qlogis` | Low |
| Uniform | `dunif`, `punif`, `qunif` | Trivial |
| Tukey's studentized range | `ptukey` (already have `studentized_range_cdf`) | Covered |
| F distribution | `df`, `pf`, `qf` (have `f_cdf`, `f_quantile`) | Covered |
| Non-central distributions | `pt(ncp=)`, `pf(ncp=)`, `pchisq(ncp=)` | Medium |
| Wilcoxon rank-sum distribution | `pwilcox`, `qwilcox` (exact distribution) | Medium |
| Wilcoxon signed-rank distribution | `psignrank` (exact) | Medium |
| Beta distribution | already have `beta_cdf`, `beta_pdf` | Covered |
| Multinomial | `dmultinom` | Low |
| Dirichlet | No standard R function; from `gtools` | Low |
| Truncated normal | `dtnorm` (from `msm`) | Low |
| GEV / GPD | See EVT section above | Medium |

**Notes**: Non-central distributions are needed for power calculations. Wilcoxon exact distributions are important for small samples. Logistic CDF/PDF/quantile are simple (sigmoid / logit).

---

### 20. Simulation and Monte Carlo

**Package**: `stats`, `SimDesign`

| Function | What it computes | Difficulty |
|---|---|---|
| `simulate` method for fitted models | Parametric bootstrap from fitted model | Medium |
| Antithetic variates | Variance reduction for MC simulation | Low |
| Control variates | MC variance reduction via correlated estimator | Low |
| Importance sampling | IS estimator with ESS diagnostics | Low |
| Quasi-Monte Carlo (Halton, Sobol) | Low-discrepancy sequences | Medium |

**Notes**: Sobol sequences for QMC are a clean, self-contained implementation. Antithetic variates are trivial (negate standard normals). These are useful across many primitives that use simulation.

---

## Priority Ranking

Based on frequency of use in applied statistics and relevance to tambear's scope:

### P1 — High Priority (most impactful gaps)
1. **Granger causality test** — trivial, extremely common in time series
2. **`var.test` / `bartlett.test` / `fligner.test`** — basic variance tests used daily
3. **`hatvalues` / DFFITS / DFBETAS** — core regression diagnostics
4. **`loess`** — nonparametric smoother used in almost every EDA
5. **Nelson-Aalen estimator** — trivial, completes survival family
6. **BCa bootstrap CI** — gold standard CI method, minor extension of existing bootstrap
7. **Logistic distribution** (d/p/q) — trivial, fills an obvious gap
8. **Probit / cloglog link in GLM** — minor extension of existing IRLS
9. **Gamma GLM** — log-link Gamma is extremely common for positive continuous outcomes
10. **Johansen cointegration** — critical for financial time series

### P2 — Medium Priority
11. **GEV / GPD distributions + fitting** — needed for extreme value / risk analysis
12. **Block bootstrap** — needed for time series resampling
13. **k-medoids (PAM)** — commonly used alternative to k-means
14. **Benjamini-Yekutieli** — minor BH extension
15. **`polr` (ordinal logistic)** — common in social science / clinical data
16. **`multinom` (multinomial logistic)** — common for K-category outcomes
17. **Gaussian / t-copula** — multivariate dependence modeling
18. **Moran's I** — spatial autocorrelation
19. **Competing risks** (Fine-Gray) — survival extension
20. **`smooth.spline`** — penalized spline smoother

### P3 — Lower Priority (complex or niche)
21. VAR / VECM (multivariate time series)
22. Full `lme4` equivalent (lmer/glmer)
23. GAM (mgcv)
24. Vine copulas
25. SEM / lavaan equivalent
26. Zero-inflated / hurdle models
27. Fuzzy clustering (FANNY)
28. Spatial kriging
29. Polytomous IRT (GRM/PCM/GPCM)
30. TBATS

---

## Implementation Notes

**Easiest wins** (< 50 lines each, direct from formula):
- `var.test`: F = s1^2/s2^2, F(n1-1, n2-1) p-value
- `bartlett.test`: Bartlett's chi-square statistic from pooled variance
- `fligner.test`: Rank absolute deviations around group median, chi-square on scores
- Nelson-Aalen: cumulative sum of d_i/n_i at event times
- Logistic distribution: CDF = sigmoid, PDF = sigmoid*(1-sigmoid), quantile = logit
- Granger causality: two AR models, F-test on additional lags
- `hatvalues`: diag(X * solve(X'X) * X') — can use existing QR
- BCa bootstrap: jackknife acceleration + normal bias correction on existing `bootstrap_percentile`
- Benjamini-Yekutieli: BH * (1 + 1/2 + ... + 1/n) correction factor

**Medium complexity** (100-300 lines):
- Probit / cloglog link in GLM: add to existing IRLS switch
- Gamma GLM: add Gamma variance/link to existing IRLS
- GEV/GPD: distributions straightforward; MLE needs careful numerical handling near shape=0
- k-medoids (PAM): swap step over all data×medoid pairs per iteration
- Block bootstrap: requires blockwise resampling infrastructure
- Moran's I: spatial weight matrix, z-score under normality assumption

**High complexity** (500+ lines, novel infrastructure):
- Johansen cointegration: VECM parameterization, eigenvalue of reduced rank matrix
- `loess`: tricube weights, local polynomial via WLS, span selection
- `polr`: cumulative link model, threshold parameters, Hessian for SEs
- Copulas: CDF/PDF/MLE for each family; sampling via conditional inversion
- VAR model: multivariate OLS per equation, information criteria for lag selection
