# Social Sciences & Advanced Finance Math Gaps
## What tambear is missing — and what cross-field primitives unlock it

**Date**: 2026-04-10  
**Scope**: Econometrics, Psychometrics, Game Theory, Network Science, Market Microstructure, Risk, Spatial Stats

---

## 1. Audit Method

- Read all `pub fn` exports in `tambear/src/` (lib.rs, panel.rs, volatility.rs, irt.rs, graph.rs, time_series.rs, multivariate.rs, hypothesis.rs, causal.rs, survival.rs, factor_analysis.rs)
- Grepped for presence of specific named functions/models
- Searched current literature for canonical methods in each field

---

## 2. What We Already Have (Relevant Inventory)

### Econometrics
- Panel: `panel_fe`, `panel_re`, `panel_fd`, `panel_twfe`, `hausman_test`, `two_sls`, `did`, `breusch_pagan_re`
- Time series: `arma_fit`, `arima_fit`, `phillips_perron_test`, `breusch_godfrey`, `box_pierce`, `newey_west_lrv`
- OLS: `lstsq`, `ols_slope`, `ols_normal_equations`, `wls`, `ridge`, `lasso`, `elastic_net`

### Market Microstructure / Volatility
- GARCH: `garch11_fit`, `egarch11_fit`, `gjr_garch11_fit`, `tgarch11_fit`, `garch11_forecast`
- Realized: `realized_variance`, `bipower_variation`, `jump_test_bns`, `tripower_quarticity`
- Microstructure: `roll_spread`, `kyle_lambda`, `amihud_illiquidity`, `vpin_bvc`
- Range-based: `parkinson_variance`, `garman_klass_variance`, `rogers_satchell_variance`, `yang_zhang_variance`
- EVT adjacent: `hill_estimator`, `hill_tail_alpha`

### Psychometrics / IRT
- IRT: `rasch_prob`, `prob_2pl`, `prob_3pl`, `fit_2pl`, `ability_mle`, `ability_eap`, `test_information`, `sem`, `mantel_haenszel_dif`
- Factor: `principal_axis_factoring`, `varimax`, `cronbachs_alpha`, `mcdonalds_omega`, `kmo_bartlett`, `scree_elbow`, `kaiser_criterion`
- Multivariate: `hotelling_one_sample`, `hotelling_two_sample`, `manova`, `lda`, `cca`, `mardia_normality`, `vif`, `mahalanobis_distances`

### Graph / Networks
- Traversal: `bfs`, `dfs`, `dijkstra`, `bellman_ford`, `floyd_warshall`
- Centrality: `degree_centrality`, `closeness_centrality`, `pagerank`
- Community: `label_propagation`, `modularity`
- Graph properties: `diameter`, `density`, `clustering_coefficient`

### Game Theory / Causal
- Causal: `propensity_scores`, `psm_match`, `ipw`, `rdd_sharp`, `doubly_robust_ate`, `e_value`
- Stochastic: `stationary_distribution`, `mean_first_passage_time`, `mixing_time`

---

## 3. Confirmed Gaps — By Domain

---

### 3.1 Econometrics

#### GMM (Generalized Method of Moments)
**Status**: ABSENT — `mixture.rs` has GMM for Gaussian mixture estimation (EM); that is NOT the econometric GMM estimator.

**What econometric GMM needs**:
- `moment_conditions(theta, data) -> Vec<f64>` — user-specified moment vector
- `gmm_objective(theta, moments, W) -> f64` — weighted quadratic form m'Wm
- `optimal_weighting_matrix(moments_matrix) -> Mat` — S = (1/n) Σ g_i g_i' (two-step: use residuals from first stage)
- `gmm_fit(moment_fn, data, W_init, optimizer) -> GmmResult` — two-step GMM
- `gmm_hac_se(theta, moment_fn, data, W) -> Vec<f64>` — HAC-robust standard errors (delegates to `newey_west_lrv` we already have)
- `sargan_hansen_j_test(gmm_result) -> TestResult` — overidentification test (χ² with df = n_moments - n_params)

**Cross-field**: GMM serves econometrics, spatial stats (spatial GMM), psychometrics (moment-based IRT fitting), and structural finance models. The moment-condition infrastructure is universal.

**Shared primitive**: `quadratic_form(v, M) -> f64` (= v'Mv) is already composable from `mat_vec` + `dot` in `linear_algebra.rs`. The gap is the GMM-specific orchestration layer.

---

#### Dynamic Panel (Arellano-Bond / System GMM)
**Status**: ABSENT

**What's needed**:
- `arellano_bond_instruments(y, lags) -> Mat` — constructs instrument matrix Z from lagged levels
- `panel_gmm_ab(y, x, units, times, n_lags) -> DynPanelResult` — difference GMM (Arellano-Bond)
- `panel_gmm_bb(y, x, units, times, n_lags) -> DynPanelResult` — system GMM (Blundell-Bond, adds level equations)
- `windmeijer_correction(vcov_twostep, ...) -> Mat` — finite-sample corrected SEs for two-step GMM
- `sargan_test`, `arellano_bond_ar_test` — instrument validity and AR(2) test for residuals

**Primitives consumed**: `panel_fd` (we have), `two_sls` (we have), `hausman_test` (we have), `optimal_weighting_matrix` (new)

---

#### VAR / SVAR / Johansen Cointegration / VECM
**Status**: ABSENT (we have univariate ARIMA; multivariate VAR is missing entirely)

**What's needed**:
- `var_fit(Y, p) -> VarResult` — multivariate AR(p) via OLS (each equation = one `lstsq` call)
- `var_aic_bic(Y, max_p)` — lag selection
- `irf(var_result, horizon, shock_col) -> Vec<Vec<f64>>` — impulse response functions
- `fevd(var_result, horizon) -> Mat` — forecast error variance decomposition
- `svar_cholesky(var_result) -> SvarResult` — structural identification via Cholesky decomposition of residual covariance
- `johansen_cointegration(Y, p, r) -> JohansenResult` — eigendecomposition of the long-run matrix Π; trace statistic and max-eigenvalue statistic
- `vecm_fit(Y, r, p) -> VecmResult` — VECM after confirming cointegrating rank r

**Primitives consumed**: `sym_eigen` (we have), `lstsq` (we have), `chi2_cdf` (we have), `newey_west_lrv` (we have)

**Cross-field**: VAR is the workhorse of macro finance, systemic risk, and climate econometrics. Johansen cointegration is the foundation of pairs trading and relative value strategies in fintek.

---

#### 3SLS (Three-Stage Least Squares)
**Status**: ABSENT (we have 2SLS)

**What's needed**:
- `three_sls(equations, instruments) -> ThreeSLSResult` — extends 2SLS to system of simultaneous equations with GLS across equations

**Primitives consumed**: `two_sls` (we have), `inv` (we have), `wls` (we have)

---

### 3.2 Psychometrics

#### Polytomous IRT Models
**Status**: ABSENT — we have only dichotomous Rasch/2PL/3PL

**What's needed**:
- `grm_prob(theta, a, b_vec) -> Vec<f64>` — Graded Response Model (Samejima 1969): category probabilities for ordered polytomous items
- `pcm_prob(theta, b_vec) -> Vec<f64>` — Partial Credit Model (Masters 1982): Rasch extension
- `gpcm_prob(theta, a, b_vec) -> Vec<f64>` — Generalized PCM (Muraki 1992): adds discrimination
- `nrm_prob(theta, a_vec, c_vec) -> Vec<f64>` — Nominal Response Model (Bock 1972): unordered categories
- `fit_grm(responses, n_persons, n_items, n_cats) -> GrmResult`
- `fit_gpcm(responses, ...) -> GpcmResult`
- Category information functions for each model

**Cross-field**: Polytomous IRT is required for Likert-scale surveys in social science, customer satisfaction analysis, and educational assessment. The `prob_*` functions are all Kingdom A (closed-form). The `fit_*` functions are Kingdom C (EM/Newton).

---

#### Structural Equation Modeling (SEM / CFA)
**Status**: ABSENT — we have `factor_analysis.rs` (EFA/CFA-adjacent) but no full SEM

**What's needed**:

**CFA (measurement model)**:
- `cfa_fit(data, model_spec) -> CfaResult` — estimate factor loadings, intercepts, residual variances via ML (iterates on implied covariance matrix Σ(θ) = ΛΦΛ' + Θ)
- `model_implied_cov(loadings, factor_cov, residual_var) -> Mat` — Σ(θ) computation
- `cfa_fit_indices(result) -> CfaFitIndices` — CFI, TLI, RMSEA, SRMR
- `cfa_modification_indices(result) -> Vec<f64>` — expected improvement from freeing each fixed parameter
- `measurement_invariance_test(group_results) -> TestResult` — metric/scalar invariance tests

**SEM (structural model)**:
- `sem_fit(data, measurement_model, structural_model) -> SemResult` — combines CFA with path model
- `sem_indirect_effect(result, from, through, to) -> f64` — mediation through latent variable
- `sem_bootstrap_ci(result, n_boot) -> BootstrapResult` — delta-method vs bootstrap SE for indirect effects

**Primitives consumed**: `covariance_matrix` (we have), `sym_eigen` (we have), `inv` (we have), `lbfgs` (we have), `chi2_cdf` (we have), `pearson_r` (we have)

**Key insight**: SEM's core primitive is `kl_divergence(S, Sigma_theta)` where S = sample covariance and Sigma_theta = model-implied covariance. This delegates to information theory we already have. The ML fit function is: F = tr(S Σ⁻¹) - ln|S Σ⁻¹| - p, minimized over θ.

---

#### Reliability (GLB, Omega_h, CAT)
**Status**: Partial — we have `cronbachs_alpha` and `mcdonalds_omega` (total omega), missing:
- `omega_hierarchical` — omega_h (general factor saturation; requires bifactor model)
- `glb` — greatest lower bound (Woodhouse-Jackson 1977; SDP-based)
- `cat_step(items, administered, responses) -> usize` — Computer Adaptive Testing: maximum information item selection

---

### 3.3 Game Theory

#### Normal-Form Nash Equilibrium
**Status**: ABSENT (we have `stationary_distribution` for Markov chains, which is adjacent but distinct)

**What's needed**:
- `support_enumeration(A, B) -> Vec<NashEquilibrium>` — enumerate all mixed-strategy Nash equilibria of a bimatrix game by testing all (I, J) support pairs; for each: solve linear system, check best-response conditions
- `lemke_howson(A, B, initial_label) -> NashEquilibrium` — complementary pivoting; finds one Nash equilibrium; exponential worst-case but fast in practice
- `pure_strategy_nash(A, B) -> Vec<(usize, usize)>` — find all pure Nash equilibria
- `dominated_strategies_elimination(A, B) -> (Mat, Mat, Vec<usize>, Vec<usize>)` — iterated elimination of strictly dominated strategies
- `minimax(A) -> (f64, Vec<f64>)` — zero-sum minimax via LP

**Primitives consumed**: `lstsq` (we have), linear programming (missing — LP solver needed)

---

#### Cooperative Game Theory: Shapley / Banzhaf
**Status**: ABSENT

**What's needed**:
- `shapley_value(v_fn, n) -> Vec<f64>` — weighted average of marginal contributions over all orderings; O(n! ) exact or O(n·2^n) via dynamic programming
- `shapley_value_dp(v_fn, n) -> Vec<f64>` — O(n·2^n) DP implementation
- `shapley_value_sampling(v_fn, n, n_samples) -> (Vec<f64>, Vec<f64>)` — Monte Carlo approximation with error bars
- `banzhaf_index(v_fn, n) -> Vec<f64>` — proportion of coalitions where player i is pivotal; DP variant O(n·2^n)
- `nucleolus(v_fn, n) -> Vec<f64>` — minimum-excess allocation (requires LP — links to LP solver gap)
- `shapley_shubik_power(weights, quota) -> Vec<f64>` — weighted voting game specialization

**Cross-field**: Shapley value is the canonical "fair attribution" primitive in ML explainability (SHAP), finance (risk attribution), and economics. Our `factor_analysis.rs` explains variance; Shapley explains contribution. These converge.

---

#### Extensive-Form / CFR
**Status**: ABSENT

**What's needed**:
- `cfr_step(info_states, strategy, regrets) -> (strategy, regrets)` — one iteration of counterfactual regret minimization
- `cfr_train(game_tree, n_iter) -> CfrResult` — converges to Nash in two-player zero-sum extensive-form games
- `cfr_plus_step(...)` — CFR+ variant (only accumulates positive regrets)

**Note**: This requires a game tree representation. The primitives (regret accumulation) are Kingdom A; the tree traversal is Kingdom B (sequential). Low priority unless fintek needs poker/auction simulation.

---

#### Auctions and Mechanism Design
**Status**: ABSENT (tangential to fintek but relevant for market design)

**What's needed**:
- `vickrey_auction(bids) -> AuctionResult` — second-price sealed-bid; trivial
- `first_price_sealed_bid(bids) -> AuctionResult` — highest bid wins, pays bid
- `vcg_mechanism(valuations, allocation_fn) -> Vec<f64>` — Vickrey-Clarke-Groves payments

---

### 3.4 Network Science

#### Advanced Centrality (Missing from graph.rs)
**Status**: Partial — we have `degree_centrality`, `closeness_centrality`, `pagerank`. Missing:
- `betweenness_centrality(g) -> Vec<f64>` — Brandes algorithm O(VE); fraction of shortest paths through each node
- `eigenvector_centrality(g, tol, max_iter) -> Vec<f64>` — power iteration on adjacency matrix
- `katz_centrality(g, alpha) -> Vec<f64>` — (I - α·A)⁻¹ · 1; generalization of eigenvector centrality
- `harmonic_centrality(g) -> Vec<f64>` — sum of inverse distances; handles disconnected graphs
- `hub_authority_scores(g) -> (Vec<f64>, Vec<f64>)` — HITS algorithm; separates hub score from authority score

**Primitives consumed**: `floyd_warshall` (we have), `power_iteration` (we have), `sym_eigen` (we have for symmetric; need unsymmetric variant)

---

#### Community Detection (Missing from graph.rs)
**Status**: Partial — we have `label_propagation`. Missing:
- `louvain(g) -> Vec<usize>` — greedy modularity optimization with multi-level merging; O(n log n)
- `leiden(g) -> Vec<usize>` — improved Louvain that guarantees well-connected communities
- `spectral_bisection(g, k) -> Vec<usize>` — Fiedler eigenvector partitioning (we have `spectral_clustering` in spectral_clustering.rs — check if it covers this)
- `girvan_newman_step(g) -> (usize, usize)` — edge betweenness-based community detection (one step)

---

#### Network Topology / Small-World / Scale-Free
**Status**: ABSENT

**What's needed**:
- `small_world_metrics(g) -> SmallWorldResult` — Watts-Strogatz sigma and omega statistics; compare clustering_coefficient and path length to random graph baseline
- `is_scale_free(g) -> (f64, f64)` — fit power law to degree distribution; returns (alpha, KS_stat)
- `barabasi_albert_graph(n, m) -> Graph` — preferential attachment generator
- `erdos_renyi_graph(n, p) -> Graph` — random graph generator
- `watts_strogatz_graph(n, k, beta) -> Graph` — small-world graph generator
- `network_motif_count(g, motif_size) -> HashMap<u64, usize>` — count 3- and 4-node motifs; requires graph isomorphism (expensive; O(n^k))

---

### 3.5 Market Microstructure (Advanced / Missing)

#### Pre-Averaging / Realized Kernels
**Status**: Partial — we have `realized_variance`, `bipower_variation`. Missing:
- `realized_kernel(returns, kernel_fn, h) -> f64` — Barndorff-Nielsen et al. (2008); noise-robust realized variance via kernel-weighted autocovariances
- `pre_averaged_rv(prices, kn) -> f64` — Ait-Sahalia & Jacod pre-averaging; local averaging before computing realized variance
- `two_scale_rv(prices_sparse, prices_dense) -> f64` — Zhang, Mykland, Ait-Sahalia (2005); debiased via scale difference
- `multi_scale_rv(prices, scales) -> f64` — multiple time scale combination

**These are critical for fintek's high-frequency signal farm.** The Hill estimator we have is adjacent but these are the canonical noise-robust estimators.

---

#### Trade Classification
**Status**: We have `vpin_bvc` (BVC classifier). Missing:
- `lee_ready_classify(price, prev_price, mid) -> i8` — Lee-Ready tick+quote rule for trade direction
- `bulk_volume_classify(price_changes, volumes) -> Vec<f64>` — BVC via normal CDF (check: vpin_bvc may already do this internally)

---

#### PIN Model (Probability of Informed Trading)
**Status**: We have `vpin_bvc`. The original structural PIN model is absent:
- `pin_mle(buys, sells) -> PinResult` — Easley-Kiefer-O'Hara (1996) mixture model; EM estimation of (alpha, delta, mu, epsilon_b, epsilon_s)
- `pin_from_params(alpha, delta, mu, epsilon) -> f64` — closed form PIN = αμ / (αμ + ε_b + ε_s)

**Note**: VPIN (volume-synchronized) vs PIN (order-flow based) are distinct. We have VPIN; PIN requires EM over the order flow mixture model.

---

#### Coherent Risk Measures / EVT (Missing)
**Status**: We have `hill_estimator` and `hill_tail_alpha`. Missing the complete EVT toolkit:
- `gpd_fit(exceedances) -> GpdResult` — maximum likelihood fit of Generalized Pareto Distribution (shape ξ, scale σ)
- `gev_fit(block_maxima) -> GevResult` — GEV fit for block maxima
- `pot_threshold_select(data) -> f64` — mean excess plot and stability diagnostic for POT threshold
- `var_gpd(p, gpd_result, threshold, n, n_excess) -> f64` — VaR from fitted GPD
- `cvar_gpd(p, gpd_result, threshold) -> f64` — CVaR/ES from fitted GPD: CVaR = VaR + (σ + ξ·(VaR-u)) / (1-ξ)
- `spectral_risk_measure(returns, phi_fn) -> f64` — weighted average of sorted losses (Acerbi 2002); CVaR is a special case with uniform phi on [p, 1]
- `var_historical(returns, p) -> f64` — empirical VaR (simple quantile)
- `es_historical(returns, p) -> f64` — empirical CVaR
- `expected_shortfall` alias for `es_historical`

**Cross-field**: GPD/GEV are needed for climate risk, insurance, seismology (we have Gutenberg-Richter), and any heavy-tail application. The POT method is the standard statistical tool shared across all extreme-value domains.

---

### 3.6 Spatial Statistics (Partial — Check existing coverage)

**Status**: We have `morans_i`, `gearys_c`, `ordinary_kriging`, `ripleys_k`, `empirical_variogram`, `nn_distances`, `clark_evans_r`. 

Confirmed missing:
- `variogram_fit_wls(empirical, model) -> VariogramFitResult` — weighted least squares variogram model fitting (currently: `spherical_variogram`, `exponential_variogram`, `gaussian_variogram` exist as model evaluators but fitting is missing)
- `universal_kriging(points, values, covariate_fn) -> KrigingResult` — kriging with trend (generalization of ordinary kriging we have)
- `kriging_variance(points, query, variogram) -> f64` — kriging variance (uncertainty)
- `spatial_lag_model(y, W, x) -> SpatialLagResult` — spatial econometrics; y = ρWy + Xβ + ε
- `spatial_error_model(y, W, x) -> SpatialErrorResult` — y = Xβ + ε, ε = λWε + u
- `local_morans_i(data, W) -> Vec<f64>` — LISA statistic (local spatial autocorrelation)
- `getis_ord_g(data, W, i) -> f64` — Getis-Ord G* hotspot statistic
- `geary_local(data, W) -> Vec<f64>` — local Geary C

---

## 4. Cross-Field Primitives: The High-Leverage Targets

These primitives unlock multiple domains simultaneously:

| Primitive | Unlocks |
|-----------|---------|
| `gmm_econometric(moment_fn, data, W)` | Econometric GMM, spatial GMM, moment-based psychometrics |
| `quadratic_form(v, M)` | GMM objective, Wald test, Mahalanobis, quadratic programming |
| `var_fit(Y, p)` | VAR, VECM, Johansen, FEVD, IRF |
| `gpd_fit(exceedances)` | EVT, CVaR, POT, insurance, climate, seismology |
| `model_implied_cov(Lambda, Phi, Theta)` | SEM/CFA, FA extensions, measurement invariance |
| `shapley_value_dp(v_fn, n)` | ML explainability (SHAP), risk attribution, cooperative games |
| `betweenness_centrality(g)` | Network analysis, systemic risk, contagion modeling |
| `louvain(g)` | Community detection in financial networks, social networks |
| `realized_kernel(returns, kernel)` | Noise-robust HF realized variance |
| `grm_prob(theta, a, b_vec)` | Polytomous IRT → Likert surveys, partial credit scoring |

---

## 5. Priority Stack for Fintek Integration

**Tier 1 — Direct fintek value (signal farm immediately benefits)**:
1. `realized_kernel` + `two_scale_rv` + `pre_averaged_rv` — noise-robust realized variance for HF tick data
2. `gpd_fit` + `cvar_gpd` + `var_gpd` + `pot_threshold_select` — EVT-based risk measures from raw ticks
3. `johansen_cointegration` + `vecm_fit` + `var_fit` — pairs trading foundation; cross-ticker K04 structure
4. `lee_ready_classify` + `pin_mle` — trade direction and informed trading detection
5. `local_morans_i` — spatial structure in cross-ticker correlation matrices

**Tier 2 — Econometric completeness (needed for academic rigors)**:
6. `gmm_econometric` — moment-condition framework; Sargan-Hansen J-test
7. `arellano_bond` + `panel_gmm_bb` — dynamic panel for factor models
8. `irf` + `fevd` — VAR structural analysis
9. `spectral_risk_measure` — coherent risk beyond CVaR

**Tier 3 — Network science (needed for K04+ systemic risk)**:
10. `betweenness_centrality` + `eigenvector_centrality` + `katz_centrality`
11. `louvain` + `leiden`
12. `small_world_metrics` + `is_scale_free`

**Tier 4 — Psychometrics / Social science (needed for completeness; lower fintek urgency)**:
13. `grm_prob` + `gpcm_prob` + `pcm_prob` + `fit_grm` + `fit_gpcm`
14. `cfa_fit` + `cfa_fit_indices` + `model_implied_cov`
15. `shapley_value_dp` + `banzhaf_index`

**Tier 5 — Game theory (long-run; needed for mechanism design and auction modeling)**:
16. `support_enumeration` + `lemke_howson` + `pure_strategy_nash`
17. `cfr_step` + `cfr_train`
18. `vickrey_auction` + `vcg_mechanism`

---

## 6. Key Insight: GMM Disambiguation

The name "GMM" is overloaded in this codebase:
- `mixture.rs::gmm_em` = Gaussian Mixture Model (EM clustering) — **PRESENT**
- Econometric GMM (Generalized Method of Moments) — **ABSENT**

When implementing econometric GMM, name it clearly: `gmm_moment_fit` or `econo_gmm_fit` or follow the convention of naming by estimator class: `mm_fit` (method of moments) + `gmm_fit` with explicit disambiguation in docs.

---

## 7. Shared Primitive Infrastructure Needed

Three lower-level primitives are repeatedly blocked on:

**1. LP Solver** (`linear_programming.rs`) — needed by:
- Nash equilibrium (minimax, dominated strategy elimination)
- Nucleolus computation
- Constrained portfolio optimization (currently using `projected_gradient`)
- Spectral risk measure with general phi
- VaR as quantile LP

**2. Unsymmetric Eigendecomposition** (`linear_algebra.rs`) — needed by:
- HITS (hub/authority); works on general rectangular adjacency
- PCA on directed graphs (asymmetric adjacency)
- Non-symmetric state-space matrices

**3. Sparse Matrix Operations** — needed by:
- Louvain/Leiden (huge graphs; dense adjacency prohibitive)
- Realized kernel (Toeplitz structure)
- Johansen (VAR coefficient matrix)

Currently `linear_algebra.rs` has dense-only; `graph.rs` has sparse adjacency lists. The gap is sparse matrix-vector multiply and sparse eigendecomposition.
