# F11 Sharing Surface: Mixed Effects as F10 + Variance Components

Created: 2026-04-01T06:08:58-05:00
By: navigator

Prerequisites: F10 complete (GramMatrix + Cholesky), F05 Phase 1 complete (GradientDescent/Adam).

---

## Core Insight: LME4 = OLS on Augmented System

**Linear Mixed Effects (LME)**:
```
y = Xβ + Zb + ε
where:
  X = fixed effects design matrix (N×p)
  β = fixed effects coefficients (p×1)   ← what F10 estimates
  Z = random effects design matrix (N×q)  ← group-level covariates
  b ~ N(0, G)                             ← random effects, distributed
  ε ~ N(0, R)                             ← residuals, distributed
```

**Key structural insight**: the BLUP (Best Linear Unbiased Prediction) for β is:

```
β̂_GLS = (X' V^{-1} X)^{-1} X' V^{-1} y
where V = Z G Z' + R   (marginal covariance of y)
```

This is GLS (F10 Phase 2), specialized to structured V = ZGZ' + R.

For the simple case G = σ_b² I (scalar random intercepts), V = σ_b² ZZ' + σ_ε² I.
The mixed effects estimator iterates between:
1. Estimate β given (σ_b², σ_ε²) via GLS = GramMatrix solve with weights
2. Estimate (σ_b², σ_ε²) given β via REML

---

## The Henderson Mixed Model Equations

The combined system can be written as a single augmented GramMatrix solve:

```
[ X'R^{-1}X   X'R^{-1}Z  ] [ β̂ ]   [ X'R^{-1}y ]
[ Z'R^{-1}X   Z'R^{-1}Z + G^{-1} ] [ b̂ ]  =  [ Z'R^{-1}y ]
```

This is a (p+q)×(p+q) augmented GramMatrix system. For R = σ_ε² I and G = σ_b² I:

```
[ X'X/σ_ε²       X'Z/σ_ε²              ] [ β̂ ]   [ X'y/σ_ε² ]
[ Z'X/σ_ε²       Z'Z/σ_ε² + I/σ_b²    ] [ b̂ ]  =  [ Z'y/σ_ε² ]
```

This IS GramMatrix([X|Z]) with a regularizer on the Z block. GramMatrix from F10,
Cholesky solve from F10, regularization = add scalar to diagonal (same as Ridge in F10).

**Tambear path**: Henderson equations = F10 GramMatrix + Ridge-style diagonal regularization.
The only new code: constructing Z (random effects design matrix) and estimating σ_b², σ_ε².

---

## Variance Component Estimation

Given β̂ and b̂, estimating (σ_b², σ_ε²) is a REML problem:

### Method 1: Iterative REML (standard in lme4)

```
Profile likelihood over (σ_b², σ_ε²):
L_REML(σ_b², σ_ε²) = log|V| + log|X'V^{-1}X| + y'P y
where P = V^{-1} - V^{-1}X(X'V^{-1}X)^{-1}X'V^{-1}
```

This requires computing log|V| and the quadratic form y'Py. Both require V^{-1} which
requires Cholesky of V = ZGZ' + R.

**For Phase 1 (scalar random intercepts only)**: V = σ_b² ZZ' + σ_ε² I.
- Cholesky of V: use matrix determinant lemma (O(q³) not O(N³))
- Two-parameter optimization: grid search or Newton-Raphson on (σ_b², σ_ε²)

### Method 2: EM Algorithm for variance components

```
E-step:  Compute E[b|y, β, σ_b², σ_ε²] = posterior mean of b
M-step:  Update σ_b² = (||b̂||² + tr(Cov[b|y])) / q
         Update σ_ε² = (||y - Xβ̂ - Zb̂||² + tr(Z' Cov[b|y] Z)) / N
```

This is Kingdom C outer loop over a Kingdom A inner step. Same EM pattern as F16 (GMM).
Phase 1: EM (simpler to implement). Phase 2: REML via L-BFGS (more accurate for small N).

---

## Random Effects Design Matrix Z

Z encodes group membership. For random intercepts only:
- Z = indicator matrix for group membership (N × G where G = number of groups)
- This is a Boolean sparse matrix

**Key optimization**: when Z is a group indicator (most common case), ZZ' is block-diagonal.
Block-diagonal Cholesky = separate Cholesky per block = O(G · (N/G)³) instead of O(N³).

For random slopes: Z includes the group-level covariate columns — still sparse.

---

## MSR Types F11 Produces

```rust
pub struct MixedModel {
    pub n_obs: usize,          // N
    pub n_fixed: usize,        // p (fixed effects)
    pub n_random: usize,       // q (random effects total)
    pub n_groups: usize,       // G (number of level-2 units)

    /// Fixed effects estimates β̂ and SEs.
    pub fixed_effects: Vec<f64>,       // shape (p,)
    pub fixed_se: Vec<f64>,            // shape (p,)

    /// Random effects estimates b̂ (BLUPs).
    pub random_effects: Vec<f64>,      // shape (q,)

    /// Variance components.
    pub sigma_b2: Vec<f64>,    // random effects variance(s)
    pub sigma_e2: f64,         // residual variance

    /// Fit statistics.
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,

    /// Random effects covariance structure.
    pub re_structure: RandomEffectsStructure,
}

pub enum RandomEffectsStructure {
    ScalarIntercept,               // G scalars b_g ~ N(0, σ_b²)
    DiagonalSlopes { n_slopes: u32 }, // diagonal G structure
    UnstructuredG,                 // full G matrix (Phase 3)
}
```

---

## GLMM (Generalized Linear Mixed Models)

GLMM = GLM (F10 GLM) + random effects. The canonical example: logistic regression with random
intercepts (e.g., students nested within schools).

```
y_ij ~ Bernoulli(logistic(x_ij'β + b_j))
b_j ~ N(0, σ_b²)
```

This is no longer analytically tractable — the marginal likelihood requires integrating out b.

**Laplace approximation** (used by lme4::glmer, the standard):
```
L(β, σ_b²) ≈ (2π)^{q/2} |H|^{-1/2} exp(ℓ(β, b̂; y))
where b̂ = mode of p(b|y, β, σ_b²)
      H = -∂²log p(b|y) / ∂b∂b'  (Hessian at mode)
```

b̂ is found by inner Newton-Raphson (= GLM IRLS applied to the augmented system).
Outer loop: L-BFGS on (β, σ_b²) via GradientOracle.

**Phase scope**:
- Phase 1: LME (linear, scalar random intercepts) — no GLMM
- Phase 2: LME with diagonal G (random slopes) + Laplace-approximated GLMM binary
- Phase 3: Full GLMM with non-Gaussian likelihoods

---

## Build Order

**Phase 1 (scalar random intercepts, linear)**:
1. `fn build_z_indicator(group_ids: &[usize], n: usize, g: usize) -> SparseMatrix` — Z matrix (~20 lines)
2. Henderson equations = GramMatrix([X|Z]) + diagonal regularization → Cholesky solve (~30 lines using F10)
3. EM variance estimation: M-step update for σ_b², σ_ε² (~30 lines)
4. Outer EM loop: iterate until ||σ_new - σ_old|| < tol (~20 lines)
5. `MixedModel` struct + `IntermediateTag` (~30 lines)
6. Standard errors: diagonal of (X'V^{-1}X)^{-1} (~10 lines)

**Phase 2 (random slopes, REML)**:
1. Block-diagonal Cholesky for V = ZGZ' + R
2. REML likelihood via L-BFGS (GradientOracle, F05 Phase 2)
3. Wald tests, LR tests for fixed effects

**Gold standards**:
- R: `lme4::lmer(y ~ x + (1|group))` for random intercepts
- R: `lme4::lmer(y ~ x + (x|group))` for random slopes
- Python: `statsmodels.MixedLM` for linear mixed models
- Match: fixed effects β̂ within 1e-4, random effects variance components within 1e-3

---

## Structural Rhyme

**LME : OLS :: Ridge : OLS**

- OLS: minimize ||y - Xβ||² (F10 GramMatrix solve)
- Ridge: minimize ||y - Xβ||² + λ||β||² (same GramMatrix + diagonal regularization)
- LME: minimize ||y - Xβ - Zb||² + b'G^{-1}b (GramMatrix on augmented system + block regularization)

LME is Ridge regression on the augmented [X|Z] system where λ = σ_ε²/σ_b² (the variance ratio).
The only difference from Ridge: λ is ESTIMATED from data, not set by the user.

This means: **lme4 = self-tuning Ridge regression on the augmented design matrix.**

---

## The Lab Notebook Claim

> Linear Mixed Effects is Ridge regression where the regularization strength is estimated from data rather than set by the user. The Henderson mixed model equations are GramMatrix([X|Z]) with a diagonal block regularizer — identical structure to Ridge (F10) on the augmented system. F11 adds variance component estimation (EM for Phase 1, REML via L-BFGS for Phase 2) on top of F10's Cholesky infrastructure. Total new code: ~90 lines for Phase 1.
