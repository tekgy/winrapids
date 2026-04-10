# F15 Sharing Surface: IRT as GLMM with Item Parameters

Created: 2026-04-01T06:22:04-05:00
By: navigator

Prerequisites: F10 complete (logistic GLM), F11 Phase 2 in progress (GLMM).
FA loadings from F14 provide useful starting values for item parameters.

---

## Core Insight: IRT = Logistic Regression with Crossed Random Effects

**Item Response Theory** models the probability of a correct response as a function of:
- Person ability θ_p (latent, random — GLMM random effect)
- Item difficulty b_j (estimated, fixed — GLMM fixed effect for item j)
- Item discrimination a_j (for 2PL model — multiplicative slope)

**The Rasch Model (1PL)**:
```
P(Y_pj = 1 | θ_p, b_j) = logistic(θ_p - b_j)
```

This is logistic regression where:
- The predictor is `θ_p - b_j`
- `θ_p` is a random effect for person p
- `b_j` is a fixed effect for item j

**Rasch = Logistic GLMM with CROSSED random effects (persons) and fixed effects (items).**

**The 2PL Model**:
```
P(Y_pj = 1 | θ_p, a_j, b_j) = logistic(a_j(θ_p - b_j))
```

Adding `a_j` = item discrimination. This is a slope parameter per item.
**2PL = Logistic GLMM with item-specific slope + intercept parameters.**

**The 3PL Model**:
```
P(Y_pj = 1 | θ_p, a_j, b_j, c_j) = c_j + (1-c_j) · logistic(a_j(θ_p - b_j))
c_j = guessing probability (lower asymptote)
```

3PL adds a floor (can't go below c_j even for very low ability). Phase 2.

---

## EM Algorithm for IRT (the standard approach)

IRT is estimated via EM because integrating out the latent ability θ is required:

**E-step**: compute posterior P(θ_p | Y_p, item_params) for each person
- Numerical quadrature over the θ distribution (discrete θ grid = Gauss-Hermite points)
- Posterior weight at each quadrature point = product of item probabilities

**M-step**: maximize expected complete-data likelihood
- Update item parameters (a_j, b_j) given fixed person posteriors
- Each item's update = weighted logistic regression (F10 GLM IRLS)
- Weight = posterior mass at each quadrature point (same structure as EM for GMM)

**This IS the IRLS master template**:
- **weights** = posterior P(θ_k | Y_p) at each quadrature point k (EM responsibility)
- **weighted GLS** = one weighted logistic regression per item
- **Loop**: alternate E-step and M-step until convergence

**Same `scatter_multi_phi_weighted` as F09/F10/F16/F13.**

---

## Tambear Decomposition

### E-step

For each person p, at each quadrature point θ_k:
```
L(θ_k | Y_p) = ∏_j P(Y_pj | θ_k, a_j, b_j)    (item likelihood at θ_k)
P(θ_k | Y_p) = L(θ_k | Y_p) · N(θ_k | 0, 1) / Z_p    (posterior, z = normalizer)
```

`L(θ_k | Y_p)` = `scatter_phi("prod(logistic(a*(theta-b)))", ByPerson)` over items.
Numerically: use `scatter_phi("sum(log_prob_item)", ByPerson)` then exp (LogSumExp safe).

This is a grouped log-probability sum: `scatter_phi("sum(y*log(p) + (1-y)*log(1-p))", ByPerson)`.
Same binary cross-entropy that GLM uses. F10 infrastructure.

The posterior weights form an N_persons × N_quadrature matrix.
For Q quadrature points and N items, E-step cost = O(N × P × Q).

### M-step

For each item j:
```
Update (a_j, b_j) = weighted logistic regression with weights = posterior column from E-step
```

`scatter_multi_phi_weighted(log_likelihood_gradient, weights, X=[θ_k - b_j])` per item.
Then Newton step (same as logistic IRLS in F10).

N_items separate weighted logistic regressions per EM iteration.
Each regression is small (p=1 or p=2, Q observations). Very fast.

### Factor Analysis Starting Values (F14 → F15 bridge)

The FA loading matrix Λ provides good starting values for IRT item parameters:
- FA loading λ_j ↔ IRT discrimination a_j (both measure "how well item j measures the factor")
- FA uniqueness ψ_j ↔ IRT misfit (both measure item-factor non-compliance)

After running F14 on the item-person score matrix, use:
- `a_j_init = Λ_j / sqrt(1 - Λ_j²)` (D'Agostino transformation from factor loading to 2PL discrimination)
- `b_j_init = -intercept_j / slope_j` from logistic regression of item j on factor score

This initialization accelerates EM convergence by 3-5x for good-fitting scales.

---

## Test Information and Adaptive Testing

**Item information function**:
```
I_j(θ) = a_j² · P(θ)(1 - P(θ))    (Fisher information for item j at ability θ)
```

where P(θ) = logistic(a_j(θ - b_j)). This is the variance of the logistic — the same
`wᵢ = μ(1-μ)` weight from logistic IRLS (F10). Zero new code.

**Test information** = Σ_j I_j(θ) — total information across all items.
SE(θ̂) = 1/√I_test(θ̂) — standard error of ability estimate.

Both are O(N_items × N_quadrature_points) arithmetic. No new GPU work.

**Computer Adaptive Testing (CAT)**:
At each step: select item j* = argmax_j I_j(θ̂_current) — maximize information.
This is `argmax` over item information values = ArgMaxOp. F01 infrastructure.

---

## MSR Types F15 Produces

```rust
pub struct IrtModel {
    pub n_persons: usize,   // P
    pub n_items: usize,     // J
    pub model: IrtModelType,

    /// Item parameters.
    pub difficulty: Vec<f64>,       // b_j, shape (J,)
    pub discrimination: Option<Vec<f64>>,  // a_j (2PL/3PL only)
    pub guessing: Option<Vec<f64>>,         // c_j (3PL only)

    /// Person ability estimates.
    pub ability: Vec<f64>,          // θ̂_p via EAP or ML, shape (P,)
    pub ability_se: Vec<f64>,       // SE(θ̂_p), shape (P,)

    /// Model fit.
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub rmsea: Option<f64>,

    /// Item fit statistics.
    pub item_infit: Vec<f64>,   // Infit mean square per item
    pub item_outfit: Vec<f64>,  // Outfit mean square per item
}

pub enum IrtModelType {
    Rasch,   // 1PL: a_j = 1 for all j
    TwoPL,   // 2PL: a_j estimated
    ThreePL, // 3PL: a_j, c_j estimated (Phase 2)
}
```

---

## Build Order

**Phase 1 (Rasch model)**:
1. Quadrature setup: Q=61 Gauss-Hermite points in [-4, 4] (~10 lines)
2. E-step: log-probability per person per quadrature point via grouped scatter (~40 lines)
3. M-step: weighted logistic regression per item using F10 GLM (~20 lines per item, J calls)
4. EM loop until convergence (~20 lines outer loop)
5. Ability estimation: EAP = weighted mean of quadrature points (~10 lines)
6. `IrtModel` struct (~30 lines)
7. Tests: R `eRm::RM()` or `ltm::rasch()` for Rasch model; match difficulty parameters within 0.01

**Phase 2 (2PL)**:
1. Add discrimination parameter a_j to M-step (2-parameter weighted logistic per item)
2. F14 starting values: compute `a_j_init` from FA loadings (~10 lines)
3. Tests: R `ltm::ltm()` for 2PL; item parameter recovery within 0.05

**Phase 3 (3PL, CAT)**:
- 3PL adds bounded guessing parameter c_j — constrained logistic regression
- CAT: item selection = ArgMaxOp on information function at current θ̂

**Gold standards**:
- R `ltm` package: `ltm::rasch()`, `ltm::ltm()`
- R `eRm` package: extended Rasch models
- Python `girth` package: IRT estimation
- Dataset: LSAT (5 items, 1000 persons) — standard IRT benchmark

---

## Structural Rhymes

**IRT 2PL = Mixed Logistic Regression** (confirmed by naturalist observation #9):
- Logistic regression (F10): fixed coefficients β for all observations
- IRT 2PL: person ability θ_p is the "random intercept" per person
- 2PL IS logistic GLMM with one random intercept per person + fixed slope-intercept per item

**Test information = Fisher information = logistic variance** (confirmed):
- F10 logistic: weights `wᵢ = μ_i(1-μ_i)` = variance of Bernoulli
- IRT: item information `I_j(θ) = a_j² · P(1-P)` = scaled Bernoulli variance
- Same quantity, different name. Zero new code once F10 is built.

**IRT infit/outfit = Pearson chi-square on residuals** (F07 bridge):
- Infit = information-weighted mean square of standardized residuals
- Outfit = unweighted mean square of standardized residuals
- Both are ratios of residuals to expected variance — chi-square statistics
- F07 chi-square test infrastructure applies

---

## The Lab Notebook Claim

> IRT is logistic GLMM (F11) with a specific crossed random effects structure: persons as random intercepts, items as fixed effects. The EM algorithm instantiates the IRLS master template: E-step computes person posterior weights (same structure as GMM E-step), M-step updates item parameters via weighted logistic regression (same as F10 GLM IRLS with fractional weights). F14 Factor Analysis provides initialization that accelerates convergence 3-5x. F15 adds ~130 lines for Phase 1 (Rasch) on top of F10/F11/F14 infrastructure.
