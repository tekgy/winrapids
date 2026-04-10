# Primitive Decomposition Map — Methods → Primitives

## The Core Thesis

Every method in tambear decomposes into:
1. **Global primitives** — math that exists independently
2. **A formula** — the thin orchestration unique to the method

If a method is >50 lines, it probably contains embedded primitives that should be extracted.

---

## Universal Primitives (shared across many methods)

These are the atoms that appear in 10+ method decompositions:

### Tier 0 — Used by almost everything
| Primitive | Current location | Methods that use it |
|---|---|---|
| `sort` | std | quantile, rank, median, trimmed_mean, KS-test, Hurst, bootstrap, order stats |
| `cumsum` | series_accel | CUSUM, DFA, Hurst, PELT, BOCPD, KPSS, PP, variance_ratio |
| `dot_product` | linear_algebra | pearson_r, OLS, cosine_sim, projections, PCA, CCA, GRAM |
| `mean` | descriptive | everything statistical (>100 methods) |
| `variance` | descriptive | everything statistical (>100 methods) |
| `rank` | nonparametric | spearman, kendall, kruskal_wallis, friedman, dunn, mann_whitney |

### Tier 1 — Used by 5-20 methods
| Primitive | Current location | Methods that use it |
|---|---|---|
| `delay_embed` | time_series | sample_entropy, approx_entropy, correlation_dim, lyapunov, RQA, FNN |
| `inversion_count` | nonparametric | kendall_tau, goodman_kruskal_gamma, somers_d |
| `tie_count` | nonparametric | kendall_tau_b, concordance_correlation |
| `ols_slope` | linear_algebra | DFA, Higuchi, Hurst, box_counting, regression diagnostics |
| `covariance_matrix` | multivariate | PCA, LDA, CCA, Mahalanobis, ridge, MANOVA, GLS |
| `eigendecomposition` | linear_algebra | PCA, spectral_clustering, matrix_exp, definiteness |
| `cholesky` | linear_algebra | solve_spd, Mahalanobis, gen_eigen, whitening |
| `svd` | linear_algebra | pinv, lstsq, rank, cond, effective_rank, polar, low_rank |
| `fft` | signal_processing | spectral (all), wavelet, convolution, circulant_solve |
| `histogram` | information_theory | entropy (all), MI, transfer_entropy, KDE |
| `distance_matrix` | clustering | DBSCAN, hierarchical, k-medoids, RQA, correlation_dim |
| `normal_cdf` | special_functions | p-values for: t-test, z-test, KS, SW, JB, DW, KPSS |
| `t_cdf` | special_functions | t-test (all), regression CIs, Bayesian factors |
| `log_gamma` | special_functions | beta, gamma distributions, factorials, Stirling |

### Tier 2 — Used by 3-5 methods
| Primitive | Current location | Methods that use it |
|---|---|---|
| `qr_factorization` | linear_algebra | lstsq, eigenvalue iteration, Gram-Schmidt |
| `lu_factorization` | linear_algebra | det, solve, inv |
| `quantile` | descriptive | quartiles, IQR, box_cox, trimmed_mean, percentile bootstrap |
| `ordinal_pattern` | complexity | permutation_entropy, forbidden_patterns, symbolic_TE |
| `levinson_durbin` | time_series | AR fit, PACF, Toeplitz solve |
| `union_find` | clustering | DBSCAN, HDBSCAN, MST-based clustering |
| `knn_search` | (needed) | KSG-MI, HDBSCAN, LOF, kNN classifier |

---

## Decomposition Details — Method by Method

### Correlation Family

```
spearman(x, y):
  primitives: rank(x) → rx, rank(y) → ry, pearson_on_ranks(rx, ry)
  formula: ρ = pearson_r(rx, ry)  [pure delegation]

kendall_tau(x, y):
  primitives: rank(y, ordered_by=x) → paired_ranks, inversion_count(paired_ranks), tie_count(x), tie_count(y)
  formula: τ_b = (n_concordant - n_discordant) / √((n₀ - n₁)(n₀ - n₂))

goodman_kruskal_gamma(x, y):         [MISSING — should exist]
  primitives: concordant_discordant_count(x, y) → (C, D)
  formula: γ = (C - D) / (C + D)

somers_d(x, y):                      [MISSING — should exist]
  primitives: concordant_discordant_count(x, y) → (C, D), tie_count(x)
  formula: d = (C - D) / (C + D + ties_on_x)

distance_correlation(x, y):
  primitives: distance_matrix(x), distance_matrix(y), double_center(A), double_center(B)
  formula: dCor = dCov(X,Y) / √(dVar(X) × dVar(Y))
```

### Hypothesis Testing Family

```
two_sample_t(s1, s2):
  primitives: moments_ungrouped(x) → s1, moments_ungrouped(y) → s2
  formula: t = (μ₁ - μ₂) / √(sp² (1/n₁ + 1/n₂))

welch_t(s1, s2):
  primitives: moments_ungrouped(x) → s1, moments_ungrouped(y) → s2
  formula: t = (μ₁ - μ₂) / √(s₁²/n₁ + s₂²/n₂), df = welch_satterthwaite_df(s1, s2)

one_way_anova(groups):
  primitives: moments_ungrouped(gᵢ) for each group
  formula: F = MS_between / MS_within

kruskal_wallis(data, groups):
  primitives: rank(data) → ranks, group_rank_sums(ranks, groups)
  formula: H = (12/(N(N+1))) × Σ Rᵢ²/nᵢ - 3(N+1)

shapiro_wilk(data):
  primitives: sort(data) → sorted, shapiro_wilk_coefficients(n) → a
  formula: W = (Σ aᵢ x_{(i)})² / Σ (xᵢ - x̄)²

two_group_comparison(g1, g2):         [Layer 2 pipeline]
  primitives: shapiro_wilk → normality, levene_test → equal_var
  formula: if normal+equal_var → t_test, else → welch or mann_whitney
```

### Time Series Family

```
ar_fit(data, p):
  primitives: acf(data, p) → autocorrelations, levinson_durbin(acf) → coefficients
  formula: pack into ArResult

adf_test(data, n_lags):
  primitives: difference(data) → Δy, lag_matrix(Δy, n_lags) → X, ols_normal_equations(X, Δy)
  formula: t_stat = β̂₁ / se(β̂₁), compare to ADF critical values

garch11_fit(returns):
  primitives: optimizer(nelder_mead or lbfgs), garch_log_likelihood
  formula: σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}, maximize LL

stl_decompose(data, period):
  primitives: moving_average(data, period), loess_smooth(detrended), 
  formula: iterative: trend = MA → seasonal = LOESS on detrended → remainder
```

### Complexity Family

```
sample_entropy(data, m, r):
  primitives: delay_embed(data, m, tau=1) → vectors_m, 
              delay_embed(data, m+1, tau=1) → vectors_m1,
              count_matches(vectors_m, r) → B,
              count_matches(vectors_m1, r) → A
  formula: SampEn = -ln(A/B)

dfa(data, min_box, max_box):
  primitives: cumsum(data - mean) → profile,
              segment(profile, box_sizes) → segments,
              ols_residuals(segment) for each → fluctuations,
              ols_slope(log(box_sizes), log(fluctuations)) → alpha
  formula: α = slope of log-log fit

hurst_rs(data):
  primitives: cumsum(data - mean) → cumulative_deviation,
              max(cumdev) - min(cumdev) → R,
              std(data) → S (each for different sub-series)
  formula: H from log(R/S) vs log(n) regression

mfdfa(data, q_values):
  primitives: dfa_at_multiple_q_orders → fluctuation functions F_q(s),
              ols_slope for each q → h(q) (generalized Hurst exponents)
  formula: τ(q) = q×h(q) - 1, f(α) = q×α - τ(q) (Legendre transform)
```

### Information Theory Family

```
mutual_information(contingency, nx, ny):
  primitives: marginal_sums(contingency) → row_sums, col_sums
  formula: MI = Σ p(i,j) × log(p(i,j) / (p(i) × p(j)))

normalized_mi(contingency, nx, ny, method):
  primitives: mutual_information → MI, shannon_entropy(row_probs) → H_X, 
              shannon_entropy(col_probs) → H_Y
  formula: NMI = MI / normalizer(H_X, H_Y, method)

transfer_entropy(x, y, n_bins):
  primitives: quantile_symbolize(x, n_bins), quantile_symbolize(y, n_bins),
              joint_histogram_3d(y_next, y_t, x_t)
  formula: TE = Σ p(y',y,x) × log(p(y'|y,x) / p(y'|y))
```

### Regression Family

```
ols_normal_equations(x, y):
  primitives: mat_mul(X^T, X) → gram, mat_mul(X^T, y) → Xty, 
              solve(gram, Xty) → coefficients
  formula: β = (X'X)⁻¹X'y

ridge(X, y, lambda):
  primitives: mat_mul(X^T, X) → gram, add_diagonal(gram, lambda),
              solve(gram + λI, X^T y)
  formula: β = (X'X + λI)⁻¹X'y

lasso(X, y, lambda):
  primitives: coordinate_descent with soft_threshold
  formula: β_j = soft_threshold(Σ xᵢⱼ(yᵢ - ŷᵢ₋ⱼ), λ) / Σ xᵢⱼ²

pca(X):
  primitives: covariance_matrix(X) → Σ, sym_eigen(Σ) → (λ, V)
  formula: select top-k eigenvectors from V, project X onto them
```

---

## Extracted Primitives That Don't Yet Exist Standalone

These are operations currently embedded inside methods but should be first-class:

1. **`concordant_discordant_count(x, y)`** — currently inside `kendall_tau`
   - Consumers: kendall, gamma, somers_d, tau-c
   
2. **`double_center(matrix)`** — currently inside `distance_correlation`
   - Consumers: distance_correlation, brownian_correlation, HSIC

3. **`count_matches(vectors, r)`** — currently inside `sample_entropy` / `approx_entropy`
   - Consumers: SampEn, ApEn, FuzzyEn, cross-SampEn

4. **`segment_fluctuation(profile, box_size)`** — currently inside `dfa`
   - Consumers: DFA, MFDFA, MF-DCCA

5. **`coarse_grain(data, scale)`** — needed for multiscale entropy
   - Not currently anywhere

6. **`ordinal_pattern(data, m, tau)`** — currently inside `permutation_entropy`
   - Consumers: permutation_entropy, forbidden_patterns, symbolic_TE

7. **`welch_satterthwaite_df(s1, s2)`** — currently inline in `welch_t`
   - Consumers: Welch t, Games-Howell, Welch ANOVA

8. **`soft_threshold(x, lambda)`** — currently inline in `lasso`
   - Consumers: LASSO, elastic net, proximal gradient, ADMM

9. **`log_sum_exp(x)`** — in `numerical.rs` already (good!)
   - Used by: attention, softmax, mixture models, BOCPD

10. **`moving_average(data, window)`** — partially in various places
    - Consumers: STL, SMA, trend estimation, coarse-graining
