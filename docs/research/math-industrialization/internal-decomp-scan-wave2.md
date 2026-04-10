# Internal Decomposition Scan — Wave 2

Scanned files: complexity.rs, spectral.rs, causal.rs, hmm.rs, kalman.rs, state_space.rs,
mixture.rs, survival.rs, panel.rs, train/logistic.rs, train/naive_bayes.rs, train/mod.rs,
tbs_executor.rs, tbs_lint.rs, pipeline.rs, superposition.rs, robust.rs

Wave 1 exhausted nonparametric.rs, hypothesis.rs, descriptive.rs, linear_algebra.rs,
time_series.rs, information_theory.rs, data_quality.rs.

---

## 1. Private functions that are standalone math primitives

### complexity.rs

**`kaplan_yorke` (line 783)**
- Computes the Kaplan-Yorke (Lyapunov) dimension from a sorted descending exponent vector.
- Formula: `D_KY = j + (Σλᵢ over positive prefix) / |λ_{j+1}|`
- Independently meaningful: fractal dimension estimator from Lyapunov spectra.
- Used in physics, chaos theory, turbulence analysis.
- Should be `pub fn kaplan_yorke_dimension(exponents: &[f64]) -> f64`.

**`rk4_step` (line 803)**
- Single RK4 integration step: `y_next = y + (h/6)(k1 + 2k2 + 2k3 + k4)`.
- The most fundamental ODE solver primitive — used by every dynamical systems method.
- Lives privately inside `lyapunov_spectrum`. Should be a top-level pub primitive in `numerical.rs` or a new `ode.rs`.
- Every DFA, Lorenz, chaos simulation, neural ODE, particle simulation needs this.

**`estimate_mean_period` (line 609)**
- Already `pub`. No action needed.

**`linear_fit_segment` (line 352)**
- Already `pub`. No action needed.

**`ols_slope` (line 274)**
- Private delegating wrapper to `linear_algebra::ols_slope`. The private wrapper is fine as a local alias — no duplication of math.

**`ccm_pearson` (line 1548)**
- Private Pearson correlation inside the CCM (Convergent Cross Mapping) implementation.
- Same math as the global `pearson_r` primitive in nonparametric/multivariate.
- Should delegate to the global primitive, not be a local copy.
- Violation: private copy of an existing global primitive.

### hmm.rs

**`safe_ln` (line 83)**
- `ln(x) = -∞ for x ≤ 0`, else `x.ln()`.
- A universal numerical primitive used in every probabilistic model.
- Appears privately in hmm.rs. Should be `pub fn safe_ln(x: f64) -> f64` in `special_functions.rs` or `numerical.rs`.
- Currently each module that needs it either reimplements it or has its own private copy.

**`log_sum_exp` (line 88)**
- Numerically stable `log(Σ exp(xᵢ))` via max-shift.
- This is arguably the most important numerical primitive in all of probabilistic computing.
- Lives privately in hmm.rs. Should be `pub fn log_sum_exp(values: &[f64]) -> f64` in `special_functions.rs`.
- Every HMM, mixture model, softmax, variational inference method needs this.
- Other modules (mixture.rs, superposition.rs, naive_bayes.rs) each implement their own inline version.

### state_space.rs

**`identity_minus` (line 449)**
- Computes `I - A` for a square matrix.
- This is a trivial linear algebra operation that belongs in `linear_algebra.rs` as `pub fn mat_identity_minus(a: &Mat) -> Mat`.
- Currently private inside state_space.rs.

**`log_det_sym` (line 464)**
- Log-determinant of a symmetric positive-definite matrix via Cholesky.
- `linear_algebra.rs` already has a `pub fn log_det(a: &Mat) -> f64`.
- This is a local reimplementation of an existing global primitive. Violation: should delegate to `log_det`.

**`outer_mat` (line 481)**
- Outer product `u vᵀ` as a Mat.
- `linear_algebra.rs` likely has `outer_product`. If not, this should be extracted as `pub fn outer_product(u: &[f64], v: &[f64]) -> Mat`.
- Used in particle filter variance updates — broadly needed.

**`symmetrise` (line 493)**
- `A → (A + Aᵀ) / 2` — symmetric part of a matrix.
- Needed everywhere covariance matrices accumulate floating-point asymmetry.
- Should be `pub fn mat_symmetrize(a: &Mat) -> Mat` in `linear_algebra.rs`.

**`sample_from_normal` (line 839)**
- Samples from N(mean, Cholesky(Σ)) using a Xoshiro256 RNG.
- A complete multivariate normal sampler. Broadly needed: GMM sampling, particle filter, Bayesian inference, bootstrap.
- Currently private inside state_space.rs. Should be `pub fn sample_multivariate_normal(mean: &[f64], l_chol: &Mat, rng: &mut Xoshiro256) -> Vec<f64>` in a sampling or stochastic module.

**`cholesky_solve_lower` (line 853)**
- Forward substitution for lower triangular system.
- `linear_algebra.rs` has `pub fn cholesky_solve` which does both forward and backward substitution. This is a half-step of that, re-implemented privately.
- Violation: duplicates logic from the global primitive.

### mixture.rs

**`log_gaussian_pdf` (line 166)**
- Log of multivariate Gaussian PDF: `-0.5 * (d·ln(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ))`.
- Used in GMM E-step. Needed by: MVN test, density estimation, anomaly detection, copula fitting, Kalman likelihood, particle filter weights.
- Should be `pub fn log_mvn_pdf(x: &[f64], mean: &[f64], cov: &Mat) -> f64` in `multivariate.rs` or `distributional_distances.rs`.
- This is a fundamental probability primitive, not a GMM implementation detail.

### causal.rs

**`sigmoid` (line 84)**
- Private re-export of `crate::neural::sigmoid`.
- Not a new primitive — correctly delegates. No violation.

### panel.rs

**`panel_fe_from_demeaned` (line 323)**
- Internal helper: OLS + clustered sandwich SE on already-demeaned data.
- Called by both `panel_fe` and `panel_twfe`. Not independently meaningful as a primitive — it's a composition step that only makes sense as part of FE estimation.
- No extraction warranted. This is correct internal sharing.

### pipeline.rs

**`rand_index_sampled` (line 575)**
- Rand Index between two label vectors, sampled for O(1) scalability.
- A standard clustering quality metric — directly related to `nonparametric` and agreement measures.
- Should be `pub fn rand_index(a: &[i32], b: &[i32]) -> f64` (full) and `pub fn rand_index_sampled(a: &[i32], b: &[i32], sample_n: usize) -> f64` in `clustering.rs` or a new `cluster_metrics.rs`.

**`pairwise_rand_index` (line 560)**
- Averages Rand Index over all pairs of views. Uses `rand_index_sampled`. Not a standalone primitive — it's an ensemble agreement metric. No extraction warranted.

**`compute_compactness` (line 592)**
- Computes mean intra-cluster squared distance / total data variance — a normalized within-cluster compactness metric (related to silhouette, Calinski-Harabasz).
- This is a cluster quality primitive: should be `pub fn cluster_compactness(data: &[f64], n: usize, d: usize, labels: &[i32]) -> f64` in `clustering.rs` or `cluster_metrics.rs`.

**`sorted_percentile` (line 642)**
- Percentile of a pre-sorted slice via linear interpolation.
- `descriptive.rs` or `nonparametric.rs` almost certainly has this. If not, it's a primitive.
- Should check for deduplication with `quantile_sorted` in nonparametric.rs (line 1467).

### superposition.rs

**`rand_index_sampled` (line 464)**
- Exact duplicate of the same function in pipeline.rs.
- Both files have their own private copy of this identical computation.
- Clear violation: two modules, same private function, same math. Extract once, pub in the right place, import everywhere.

**`pearson_corr` (line 552)**
- Local Pearson correlation for series agreement. Same math as `pearson_r` in nonparametric/multivariate.
- Violation: private copy of an existing global primitive.

**`label_agreement`, `modal_k_from_labels`, `variance_ratio_agreement`, `elbow_nc`, `series_agreement`**
- These are meaningful superposition-specific aggregation functions, not globally reusable primitives. No extraction warranted.

### robust.rs

**`m_estimate_irls` (line 129)**
- Generic IRLS M-estimation loop: given data and a weight function, iterates weighted mean to convergence.
- The IRLS framework is used in: M-estimation (many weight functions), IRLS logistic regression, IRLS WLS, robust regression generally.
- Should be `pub fn irls_weighted_mean(data: &[f64], weight_fn: impl Fn(f64) -> f64, max_iter: usize, tol: f64) -> MEstimateResult`.
- Currently only the specific weight-function variants (Huber, Bisquare, Hampel) are pub. The engine is private.

**`mad_from_sorted` (line 177)**
- MAD from already-sorted data (avoids re-sorting).
- `descriptive.rs` likely has a `pub fn mad(data: &[f64]) -> f64`. This is an optimization variant: MAD when input is already sorted.
- Should be `pub fn mad_sorted(sorted: &[f64], median: f64) -> f64` to expose the optimized path.

### train/naive_bayes.rs

No private functions with standalone math. The inline Gaussian log-PDF at line 101 duplicates `log_mvn_pdf` in 1D form — the same formula, written inline rather than calling a primitive. Minor violation; extract when `log_mvn_pdf` is promoted.

---

## 2. Inline loops >10 lines computing a reusable result

### complexity.rs — `hurst_rs` (line 208)
- Lines 236-248: manually computes mean, cumulative deviation, range, and sample std of a block.
- The mean and std computations are primitives (`moments_ungrouped`). The cumulative deviation is `prefix_sum(x - mean)`. Range is `max - min` of a prefix sum vector.
- Each block stat should call the global moment primitive; range is a `max_scan - min_scan` gather.

### complexity.rs — `lempel_ziv_complexity` (line 425)
- Lines 430-432: computes median by full sort. Should call `crate::descriptive::median` or `crate::nonparametric::quantile`.
- Currently re-sorts manually. Violation: uses sort-then-index-halfway instead of calling the global quantile primitive.

### complexity.rs — `largest_lyapunov` (line 545)
- Lines 563-567: manually computes L2 distance between embedded vectors inline.
- This exact computation (`||xᵢ - xⱼ||₂` between delay-embedded vectors) appears in correlation_dimension, RQA, and Lyapunov. Should be a shared `euclidean_distance(a: &[f64], b: &[f64]) -> f64` primitive — or reuse the global distance primitive if one exists.

### survival.rs — `cox_ph` (lines 192-302)
- Lines 196-209: manually computes `exp(x·β)` with log-sum-exp stabilization.
- This dot product + exp pattern (score = x·β, probability = sigmoid/exp(score)) appears in Cox PH, logistic regression, propensity scores, and GLM estimation. The stabilized `exp_xb` computation (shift by max, then exp) should be a primitive: `pub fn linear_predictor_exp(x: &[f64], n: usize, d: usize, beta: &[f64]) -> Vec<f64>`.
- Lines 213-269: S0/S1/S2 accumulation (risk set statistics) — this is an order-statistic accumulation pattern (reverse-order prefix scan over sorted event times). The skeleton appears in Kaplan-Meier, log-rank, Nelson-Aalen baseline hazard. Should be a `risk_set_accumulate` primitive.

### panel.rs — `panel_fe` and `panel_fe_from_demeaned`
- Both contain a ~20-line block computing the sandwich (clustered SE) estimator:
  `V = (X'X)⁻¹ B (X'X)⁻¹ * correction`
  where `B = Σ_u (X_u' û_u)(X_u' û_u)'`.
- This exact sandwich is also needed in 2SLS, robust regression, GEE. Should be `pub fn clustered_sandwich_se(x_dm: &[f64], residuals: &[f64], n: usize, d: usize, units: &[usize]) -> Vec<f64>`.

### mixture.rs — `gmm_em` (line 34)
- Lines 43-61: K-means++ style initialization (distance-based weighted sampling).
- This seeding algorithm is also used in kmeans.rs. Two files, same algorithm, unshared.
- Should be `pub fn kmeans_plus_plus_init(data: &[f64], n: usize, d: usize, k: usize, seed: u64) -> Vec<Vec<f64>>` in clustering.rs.

---

## 3. Functions >100 lines with multiple embedded sub-operations

### complexity.rs — `lyapunov_spectrum` (~130 lines, 651-778)
- Sub-operations: RK4 integration of combined ODE+variational system, QR decomposition, log accumulation.
- Already calls `qr` from linear_algebra. The RK4 step is private (`rk4_step`). The combined ODE + variational equation integration is a reusable pattern (appears in any tangent-flow computation).
- Estimated sub-operations: 3 (RK4 step, tangent-flow combine, QR accumulate).

### survival.rs — `cox_ph` (~220 lines, 181-398)
- Sub-operations: log-sum-exp stabilized linear predictor, S0/S1/S2 risk set scan, Newton step (Cholesky solve), Schoenfeld residuals, SE from Hessian inverse.
- Estimated sub-operations: 5.
- The final SE computation block (lines 330-396) re-runs the entire risk set scan — it repeats the same O(n²) work as the training loop. A single-pass implementation would compute residuals and information matrix simultaneously.

### causal.rs — `propensity_scores` (~55 lines, 25-79)
- Implements IRLS for logistic regression from scratch inside the function.
- Already exists as `train::logistic::fit`. Should call the global logistic regression primitive, not reimplement IRLS inline.
- Estimated sub-operations: 3 (IRLS loop, XTWX assembly, Cholesky solve).

### panel.rs — `panel_fe` (~120 lines, 83-201)
- Sub-operations: per-unit mean accumulation (demean), OLS via lstsq, R², sandwich SE.
- Estimated sub-operations: 4.

---

## 4. Duplicate private implementations of existing global primitives

| Location | Private fn | Existing global |
|---|---|---|
| complexity.rs:1548 | `ccm_pearson` | `nonparametric::pearson_r` |
| state_space.rs:464 | `log_det_sym` | `linear_algebra::log_det` |
| state_space.rs:853 | `cholesky_solve_lower` | `linear_algebra::cholesky_solve` |
| superposition.rs:464 | `rand_index_sampled` | (should be in clustering.rs) |
| superposition.rs:552 | `pearson_corr` | `nonparametric::pearson_r` |
| pipeline.rs:575 | `rand_index_sampled` | identical to superposition.rs:464 |
| pipeline.rs:642 | `sorted_percentile` | `nonparametric::quantile_sorted` (line 1467) |
| causal.rs:25-79 | IRLS logistic inline | `train::logistic::fit` |
| hmm.rs:83 | `safe_ln` | (should be in special_functions.rs) |
| hmm.rs:88 | `log_sum_exp` | (should be in special_functions.rs) |
| mixture.rs:43-61 | K-means++ init | (should be in clustering.rs, also used by kmeans.rs) |

---

## 5. Primitives that should be promoted to the global catalog

These are functions that exist in the scanned files but don't yet exist as standalone pub primitives anywhere:

| Primitive | Currently in | Priority |
|---|---|---|
| `log_sum_exp(values: &[f64]) -> f64` | hmm.rs private | **Critical** — needed by every probabilistic model |
| `safe_ln(x: f64) -> f64` | hmm.rs private | High — log of zero is universal |
| `log_mvn_pdf(x: &[f64], mean: &[f64], cov: &Mat) -> f64` | mixture.rs private | High — core density primitive |
| `rk4_step(f, t, y, h) -> Vec<f64>` | complexity.rs private | High — ODE solver primitive |
| `kaplan_yorke_dimension(exponents: &[f64]) -> f64` | complexity.rs private | Medium — chaos/fractal catalog |
| `rand_index(a: &[i32], b: &[i32]) -> f64` | pipeline.rs + superposition.rs private × 2 | Medium — clustering quality metric |
| `cluster_compactness(data, n, d, labels) -> f64` | pipeline.rs private | Medium — clustering quality metric |
| `outer_product(u: &[f64], v: &[f64]) -> Mat` | state_space.rs private | Medium — linear algebra |
| `mat_symmetrize(a: &Mat) -> Mat` | state_space.rs private | Medium — linear algebra |
| `irls_weighted_mean(data, weight_fn, max_iter, tol) -> MEstimateResult` | robust.rs private | Medium — M-estimation engine |
| `sample_multivariate_normal(mean, l_chol, rng) -> Vec<f64>` | state_space.rs private | Medium — sampling primitive |
| `clustered_sandwich_se(x_dm, residuals, n, d, units) -> Vec<f64>` | panel.rs embedded × 2 | Medium — regression SE primitive |
| `kmeans_plus_plus_init(data, n, d, k, seed) -> Vec<Vec<f64>>` | mixture.rs + kmeans.rs | Medium — initialization primitive |
| `euclidean_distance(a: &[f64], b: &[f64]) -> f64` | complexity.rs inline × 3 | Low — may already exist |
| `mad_sorted(sorted: &[f64], median: f64) -> f64` | robust.rs private | Low — optimization of existing mad |

---

## Summary

**Total private functions scanned**: ~90 across 17 files

**Clear violations** (private copies of existing globals): 10 instances

**Missing global primitives** (math that exists independently but isn't pub): 15 candidates

**Highest-priority extractions**:
1. `log_sum_exp` — the single most reused probabilistic primitive, currently private in hmm.rs, reimplemented inline in 4+ other files
2. `log_mvn_pdf` — used in GMM, Kalman, particle filter; currently private in mixture.rs
3. `rk4_step` — the fundamental ODE primitive; all dynamical systems methods need it
4. `rand_index` — duplicated identically in pipeline.rs and superposition.rs
5. `causal.rs:propensity_scores` IRLS inline — should call `train::logistic::fit`
6. `state_space.rs:log_det_sym` — duplicates `linear_algebra::log_det`
