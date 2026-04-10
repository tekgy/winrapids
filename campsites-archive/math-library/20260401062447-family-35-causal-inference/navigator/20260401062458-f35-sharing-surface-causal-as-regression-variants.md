# F35 Sharing Surface: Causal Inference as Regression Variants

Created: 2026-04-01T06:24:58-05:00
By: navigator

Prerequisites: F10 complete (OLS + IV), F11 complete (LME), F12 complete (panel FE).

---

## Core Insight: Every Causal Method Is a Regression with a Design Twist

Causal inference methods are NOT new statistical algorithms — they're F10 regression applied to:
1. A transformed dataset (matching, weighting, IV)
2. A structural restriction (DiD interaction, RDD window)
3. A sequencing convention (IV two-stage)

Every method in F35 reduces to: "run F10 OLS/GLM on some transformed or restricted dataset."

---

## Instrumental Variables (IV / 2SLS)

**Problem**: X is correlated with ε (endogeneity). Need instrument Z: correlated with X, uncorrelated with ε.

**Two-Stage Least Squares (2SLS)**:
```
Stage 1: X̂ = Z(Z'Z)^{-1}Z'X    ← OLS of X on Z
Stage 2: β̂_IV = (X̂'X̂)^{-1}X̂'y ← OLS of y on X̂ (predicted X)
```

Both stages = F10 OLS. The result:
```
β̂_IV = (X̂'X̂)^{-1}X̂'y
      = (X'P_Z X)^{-1} X'P_Z y    where P_Z = Z(Z'Z)^{-1}Z' (projection matrix)
```

**Tambear decomposition**:
1. Stage 1: `GramMatrix(Z, X)` → Cholesky solve → X̂ (F10)
2. Stage 2: `GramMatrix(X̂, y)` → Cholesky solve → β̂_IV (F10)
3. Standard errors: HC/cluster robust SEs (sandwich formula from F10/F12)

**Zero new primitives.** Two sequential F10 OLS calls.

**Generalized Method of Moments (GMM IV)**:
```
β̂_GMM = (X'Z W_n Z'X)^{-1} X'Z W_n Z'y
```
where W_n = optimal weighting matrix. Still two GramMatrix operations + Cholesky.

---

## Difference-in-Differences (DiD)

**Classic 2×2 DiD**:
```
y_it = β₀ + β₁·Post_t + β₂·Treated_i + β₃·(Post_t × Treated_i) + ε_it
ATT = β₃
```

This is OLS with an interaction term. F10 OLS with augmented X matrix.

**With unit fixed effects** (standard in modern DiD):
```
y_it = α_i + γ_t + β · (Treated_i × Post_t) + ε_it
```

This is F12 Two-Way Panel FE with an interaction column added to X.
ATT = β̂_within from panel FE on the interaction term.

**Staggered DiD** (Callaway-Sant'Anna, Sun-Abraham):
Multiple treatment cohorts, staggered rollout. Requires cohort-specific ATT estimation.
= Multiple DiD regressions, one per cohort × period pair. F12 repeated.

**Parallel trends test** = F07 pre-trend test: t-test on pre-treatment differences.
Plot pre-trends from F06/F07 group means. Zero new code.

---

## Regression Discontinuity Design (RDD)

**Sharp RDD**:
```
y_i = α + β·D_i + f(x_i) + ε_i   where D_i = 1{x_i ≥ c} (cutoff)
β = treatment effect at the discontinuity
```

**Local linear regression** (standard bandwidth h around cutoff c):
```
β̂_RDD = β̂ from OLS on observations with |x_i - c| < h, including:
- Separate polynomials on each side of cutoff
- Kernel weights w_i = K((x_i - c)/h) (triangular or Epanechnikov)
```

This is **weighted OLS on a filtered subset** — F10 WLS with kernel weights, applied to the
subset |x_i - c| < h.

**Tambear decomposition**:
1. Filter observations: `|x_i - c| < h` — gather operation (F01)
2. Kernel weights: triangular `w_i = 1 - |x_i - c|/h` — element-wise arithmetic
3. WLS: F10 weighted GramMatrix solve on filtered subset

**Bandwidth selection**: MSE-optimal bandwidth via cross-validation or IK estimator.
CV = repeated F10 OLS on different bandwidth values — Kingdom C outer loop.

**Fuzzy RDD** (discontinuous but non-sharp treatment takeup):
= IV where the instrument is `1{x_i ≥ c}`. Reduces to 2SLS (F35 IV above).

---

## Propensity Score Methods

**Propensity score** = P(D=1 | X) = predicted treatment probability.
Estimated via logistic regression on X: **F10 logistic GLM**.

**Propensity Score Matching**:
1. Estimate propensity score: F10 logistic
2. Match treated to controls: nearest-neighbor matching on p-score = KNN on scalar distance (F01)
3. Estimate ATT: F10 OLS on matched sample

**Inverse Probability Weighting (IPW)**:
1. Estimate propensity score: F10 logistic → ê(x_i)
2. Weights: `w_i = D_i/ê(x_i) + (1-D_i)/(1-ê(x_i))`
3. Weighted OLS: F10 WLS with w_i

**Doubly Robust (AIPW)**:
Combines outcome model + propensity model for robustness.
= F10 WLS + propensity correction term. Slightly more complex but still F10-based.

---

## Synthetic Control

**What it does**: creates a weighted combination of control units that matches the treated unit's pre-treatment outcomes.

```
Minimize: ||y_treated_pre - Σ_j w_j y_j_pre||²
subject to: w_j ≥ 0, Σ_j w_j = 1
```

This is a **constrained OLS** (non-negative weights summing to 1) = QP on GramMatrix.
Same structure as SVM dual (F21) — convex QP on a GramMatrix.

The synthetic control then estimates ATT by comparing treated post-period to weighted control post-period.

**Phase 2**: synthetic control requires the QP infrastructure from F05/F21.

---

## Local Average Treatment Effect (IV + Heterogeneous Effects)

LATE = E[Y₁ - Y₀ | complier] from Wald ratio:
```
LATE = (E[Y|Z=1] - E[Y|Z=0]) / (E[D|Z=1] - E[D|Z=0])
```
= ratio of two F06 group means. One scatter call, two extractions.

**Quantile Treatment Effects**: LATE at quantiles = quantile regression (F10) with IV.
Requires LP solver (F10 Phase 2, quantile regression). Phase 3.

---

## Randomization Inference

Test sharp null H₀: Y_i(1) = Y_i(0) for all i by permuting treatment assignments:
```
T_obs = observed test statistic (e.g., difference in means)
T_perm = T under random permutation of D_i
p-value = P(T_perm > T_obs)
```

Permutation test = F07 two-sample test with permutation distribution.
R permutations × one group mean calculation each = R × scatter_phi("mean", ByGroup) calls.
For R=9999, this is 9999 F06 grouped mean computations — very fast with batched accumulate.

---

## MSR Types F35 Produces

```rust
pub struct CausalEstimate {
    pub method: CausalMethod,
    pub n_obs: usize,
    pub n_treated: usize,

    /// Average Treatment Effect (ATE or ATT):
    pub ate: f64,
    pub ate_se: f64,
    pub ate_ci_lower: f64,
    pub ate_ci_upper: f64,

    /// First-stage statistics (for IV):
    pub first_stage_f: Option<f64>,   // F-stat > 10 = strong instrument
    pub first_stage_r2: Option<f64>,

    /// Diagnostics:
    pub bandwidth: Option<f64>,       // RDD bandwidth
    pub n_matched: Option<usize>,     // matching n
}

pub enum CausalMethod {
    Ols,
    Iv2Sls { n_instruments: usize },
    Did { n_periods: usize, n_units: usize },
    Rdd { cutoff: f64, bandwidth: f64, polynomial_order: u8 },
    Psm { n_neighbors: usize },
    Ipw,
    DoublyRobust,
    SyntheticControl { n_donors: usize },
}
```

---

## Build Order

**Phase 1 (IV, DiD, RDD)**:
1. `fn twosls(y, x_endogenous, z_instruments, x_exogenous) -> CausalEstimate` — two F10 OLS calls (~30 lines)
2. `fn did_twoway_fe(y, treated, post, unit_ids, time_ids) -> CausalEstimate` — F12 panel FE call + interaction column (~20 lines)
3. `fn rdd_local_linear(y, x, cutoff, bandwidth) -> CausalEstimate` — filter + WLS (F10) (~40 lines)
4. Tests: R `AER::ivreg()` for 2SLS, `plm` + manual interaction for DiD, `rdrobust` for RDD

**Phase 2 (Propensity score methods)**:
1. `fn psm(y, d, x) -> CausalEstimate` — F10 logistic + KNN matching + OLS (~50 lines)
2. `fn ipw(y, d, x) -> CausalEstimate` — F10 logistic + WLS (~30 lines)
3. Tests: R `MatchIt`, `WeightIt` packages

**Phase 3 (Synthetic Control, Quantile Treatment Effects)**:
- Require QP solver (F05/F21) and quantile regression (F10 Phase 2)

---

## The Lab Notebook Claim

> Causal inference (F35) is not a new statistical family — it is F10/F11/F12 regression with structural restrictions and design conventions applied. 2SLS = two F10 OLS calls. DiD = F12 two-way panel FE with an interaction term. RDD = F10 WLS on a filtered subset. Propensity score matching = F10 logistic + F01 KNN + F10 OLS. F35 adds ~90 lines for Phase 1 — three structural restrictions around existing regression infrastructure. No new primitives.
