# F12 Sharing Surface: Panel Data as RefCenteredStats + OLS

Created: 2026-04-01T06:11:51-05:00
By: navigator

Prerequisites: F10 complete (GramMatrix + OLS), F06 complete (MomentStats, ByKey grouping).

---

## Core Insight: Panel Fixed Effects IS Within-Group Centering

Panel data has repeated observations across units (firms, individuals, countries) over time.
The canonical model: `y_it = α_i + β x_it + ε_it` where α_i = unit-specific intercept (fixed effect).

**The "within" transformation** eliminates α_i by demeaning within each unit:
```
ÿ_it = y_it - ȳ_i   (demeaned outcome)
ẋ_it = x_it - x̄_i   (demeaned covariate)
Then: ÿ_it = β ẋ_it + ε_it   (OLS on demeaned data, no α_i)
```

**This IS RefCenteredStats.** `scatter_phi("v - r", keys=unit_id, ref_values=group_means)`
produces ẋ_it for every x simultaneously. One scatter call, all demeaned covariates.

Panel FE = RefCenteredStats (F06 grouped centering) + OLS (F10 GramMatrix on demeaned data).
Zero new primitives.

---

## Estimator Decomposition

### Fixed Effects (Within) Estimator

```
1. Compute group means: x̄_i = scatter_phi("mean(v)", ByKey(unit_id)) — F06 reuse
2. Demean: ẋ_it = x_it - x̄_i — RefCenteredStats scatter
3. OLS on demeaned: β̂_FE = GramMatrix(ẋ)^{-1} ẋ'ÿ — F10 reuse
4. Recover individual FEs: α̂_i = ȳ_i - x̄_i' β̂_FE — trivial
```

**Variance estimation**: `Var(β̂_FE) = σ² (ẋ'ẋ)^{-1}` where σ² = ||ÿ - ẋβ̂||² / (NT - N - K + 1)
GramMatrix(ẋ) already computed in step 3. Residuals from step 3. σ² = RefCenteredStats pass.

**Degrees of freedom correction**: N individual FEs consume N degrees of freedom.
`df_within = NT - N - K + 1` where N = units, T = avg time periods, K = covariates.

### Between Estimator

Regresses unit averages on covariate averages:
```
ȳ_i = α + β x̄_i + ε_i
```
β̂_between = OLS on (ȳ_i, x̄_i) — the group-level aggregates from MomentStats.
This is F10 OLS on the group-level MSR from F06. Zero new computation.

### Random Effects (RE) Estimator

GLS on a quasi-demeaned form:
```
(y_it - θȳ_i) = (1-θ)α + β(x_it - θx̄_i) + ε_it
where θ = 1 - sqrt(σ_ε² / (σ_ε² + T·σ_α²))
```

For θ=0: pooled OLS. For θ=1: FE within estimator. RE interpolates between them.
θ requires variance components σ_ε² and σ_α² — estimated via REML (same as F11 LME!).

**Structural rhyme**: RE panel estimator = LME with one random intercept per unit.
**Tambear path**: RE calls F11's variance component estimation.

### Hausman Test

Tests FE vs RE: `H = (β̂_FE - β̂_RE)'[Var(β̂_FE) - Var(β̂_RE)]^{-1}(β̂_FE - β̂_RE) ~ χ²_K`

This is a chi-square test (F07) on the difference of two estimators. The difference vector is
from GramMatrix computations above. The matrix inverse is O(K³) — trivial for K < 100.

**Gold standard**: R `plm::phtest()` for the Hausman test.

---

## Time-Series Extensions

### First-Differences Estimator

Another way to remove fixed effects:
```
Δy_it = y_it - y_{i,t-1}   (first difference)
β̂_FD = OLS(Δy_it, Δx_it)
```

`Δy_it` = AffineState with A=1, b=-y_{i,t-1} — a LAGGED DIFFERENCE, one step behind.
This is the Affine scan (F17, Kingdom B) applied within each unit.

**For unbalanced panels**: need to track unit × time indices. The lag operation is:
`gather(lag_indices, values)` where lag_indices maps (unit_i, t) → (unit_i, t-1) position.

### Two-Way FE

Adds time fixed effects: `y_it = α_i + γ_t + β x_it + ε_it`

"Within-within" transformation: demean by unit, demean by time, add back grand mean.
```
ÿ_it = y_it - ȳ_i - ȳ_t + ȳ   (Frisch-Waugh iterative projections)
```
Two RefCenteredStats passes (ByUnit then ByTime) — two scatter calls.
The alternating projection converges in 2-3 iterations for balanced panels.

---

## Panel-Robust Standard Errors

OLS SEs assume homoscedasticity. Panel data has clustered errors (within-unit correlation).
Clustered SEs (cluster-robust): `Var(β̂) = (X'X)^{-1} B (X'X)^{-1}` where

```
B = Σ_i X_i' ê_i ê_i' X_i   (sandwich meat)
```

X_i = rows of X belonging to unit i. ê_i = unit i's residuals.
`B` = scatter of outer products of (residual-weighted X) grouped by unit.

**Tambear**: `scatter_outer_product_weighted` — same primitive needed for MANOVA (F33).
Group by unit, accumulate ê_it² · x_it x_it' (weighted outer product).
This is the ONLY new primitive F12 needs (and F33 also needs it).

---

## Dynamic Panel Models (Phase 2)

### Arellano-Bond (GMM-IV)

When lagged dependent variable is a regressor: `y_it = ρ y_{i,t-1} + β x_it + α_i + ε_it`

FE is biased (Nickell bias). Solution: use lags as instruments (GMM).
`β̂_AB = (Z'ΔY)^{-1} Z'ΔX'ΔY` where Z = instrument matrix, Δ = first differences.

This requires IV/2SLS (F10 extension) and GMM weighting. Phase 2.

### Panel VAR

Vector autoregression for panel data. Requires F17 (Affine scan) + F11 (LME).
Phase 3.

---

## MSR Types F12 Produces

```rust
pub struct PanelModel {
    pub n_units: usize,     // N
    pub n_periods: usize,   // T (avg, may be unbalanced)
    pub n_obs: usize,       // NT
    pub n_params: usize,    // K

    /// Fixed effects estimates:
    pub beta: Vec<f64>,           // β̂ (pooled or within)
    pub beta_se: Vec<f64>,        // cluster-robust or homoscedastic
    pub unit_effects: Option<Vec<f64>>,  // α̂_i (FE only, may be large)

    pub estimator: PanelEstimator,
    pub r2_within: f64,     // R² from demeaned regression
    pub r2_between: f64,    // R² from group-mean regression
    pub r2_overall: f64,    // R² from pooled OLS

    /// Hausman test (if RE and FE both computed):
    pub hausman_chi2: Option<f64>,
    pub hausman_df: Option<usize>,
    pub hausman_p: Option<f64>,
}

pub enum PanelEstimator {
    Pooled,
    FixedEffects { two_way: bool },
    RandomEffects { theta: f64 },
    FirstDifferences,
}
```

---

## Build Order

**Phase 1 (one-way FE, between, RE)**:
1. `fn within_demean(data: &[f64], unit_ids: &[usize]) -> Vec<f64>` — scatter mean + subtract (~20 lines)
2. `fn fe_estimate(y: &[f64], x_matrix: &[f64], unit_ids: &[usize]) -> PanelModel` — call within_demean then F10 OLS (~40 lines)
3. Variance components for RE: call F11's `em_variance_components()` (~10 lines)
4. Hausman test: chi-square computation from FE and RE results (~30 lines)
5. `PanelModel` struct + `IntermediateTag` (~30 lines)
6. Tests: R `plm` package — `plm(y ~ x, data=df, model="within", index=c("unit","time"))`

**Phase 2 (two-way FE, clustered SEs, first differences)**:
1. Two-way demeaning: two RefCenteredStats passes with alternating projection
2. `scatter_outer_product_weighted` for cluster-robust Sandwich SEs
3. First-difference transform: lag-by-unit via gather(lag_indices)
4. Tests: match `plm(model="fd")` and `plm(model="within", effect="twoways")`

**Gold standards**:
- R `plm` package: `plm(y ~ x, model="within/random/between")`, `phtest()` for Hausman
- Python `linearmodels`: `PanelOLS`, `RandomEffects`, `BetweenOLS` — all match `plm`
- Stata: `xtreg y x, fe/re` — industry standard, use as final verification

---

## Structural Rhymes

**FE = Within-Group OLS = F07 ANOVA structure**:
SS_within from ANOVA = denominator of the FE estimator's variance. The demeaned data IS the within-group residuals from ANOVA. FE regression uses the within-group variation, just as ANOVA does.

**RE = LME with one random intercept per unit** (confirmed):
The RE estimator θ formula IS the LME variance ratio formula. `plm` RE and `lme4` LME give identical β̂ when the model matches. F12 Phase 1 RE calls F11 directly.

**Panel FD = F17 (first-difference = lag-1 Affine scan)**:
The first-difference operator is ΔA = A - L(A) where L = lag operator = the Affine scan with A=I, b=0. FD is an Affine scan applied within each unit.

---

## The Lab Notebook Claim

> Panel fixed effects estimation is RefCenteredStats (F06 grouped centering) applied to outcome and covariates, followed by OLS (F10 GramMatrix) on the demeaned data. The "within transformation" IS scatter_phi("v - group_mean", ByKey(unit_id)) — already designed as the multi-domain RefCenteredStats primitive. Panel random effects = LME with one random intercept per unit (F11). Panel first differences = Affine lag scan within units (F17). F12 adds ~120 lines for Phase 1 — all new code is wiring, not new math.
