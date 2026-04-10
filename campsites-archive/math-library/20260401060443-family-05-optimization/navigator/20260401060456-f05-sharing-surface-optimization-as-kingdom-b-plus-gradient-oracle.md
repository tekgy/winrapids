# F05 Sharing Surface: Optimization as Kingdom B + Gradient Oracle

Created: 2026-04-01T06:04:56-05:00
By: navigator

Prerequisites: F17 complete (Affine scan); gradient duality proven (March 31 session).

---

## Core Insight: Two Kingdoms, One Framework

**F05 spans two kingdoms:**

- **Kingdom B** (sequential/Affine scan): Adam optimizer — moment estimates ARE EWM of gradients
- **Kingdom C** (iterative outer loop): gradient descent — each outer iteration calls the gradient oracle

The gradient oracle itself is a backward accumulate pass — gradient duality. No separate autodiff engine needed.

---

## Adam = Affine Scan (Kingdom B)

Adam maintains two Affine recurrences:

```
First moment (mean of gradients):
  m_t = β₁ · m_{t-1} + (1-β₁) · g_t
  → A = β₁  (constant scalar)
  → b_t = (1-β₁) · g_t  (data-dependent)

Second moment (mean of squared gradients):
  v_t = β₂ · v_{t-1} + (1-β₂) · g_t²
  → A = β₂  (constant scalar)
  → b_t = (1-β₂) · g_t²  (data-dependent)
```

Both are scalar Affine scans over gradient sequence [g_1, g_2, ..., g_T]. This is IDENTICAL
to EWM in F17 (span-weighted moving average) and EWMV in F18 (variance of returns).

**Parameter update**:
```
m̂_t = m_t / (1 - β₁ᵗ)   (bias-corrected)
v̂_t = v_t / (1 - β₂ᵗ)   (bias-corrected)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

Bias correction is just scalar division — trivial. The scan IS the algorithm.

**Tambear path**: `AffineState` from F17 directly reusable. Two AffineState instances per parameter
dimension (one for m, one for v). No new infrastructure.

**AdaGrad, RMSProp**: same structure, simpler (one moment instead of two):
- AdaGrad: v_t = v_{t-1} + g_t² (Affine with A=1, cumulative sum)
- RMSProp: v_t = ρ·v_{t-1} + (1-ρ)·g_t² (EWM, A=ρ)

---

## Gradient Oracle: Backward Accumulate

**Proven in March 31 session (gradient duality)**:

Forward accumulate: `y = scatter_phi(expr, keys, values)`
Backward pass: `∂L/∂values[i] = ∂L/∂y[keys[i]] · ∂expr/∂values[i]`

This is ITSELF a scatter operation — the backward pass through accumulate IS accumulate with
transposed indices. The chain rule is closed under the accumulate primitive.

For any K-pass computation:
```
Forward:  state_1 = acc(data); state_2 = acc(state_1); ... → output
Backward: ∂output/∂data = backward_acc(∂L/∂output) through each layer, reversed
```

This means: **any tambear computation graph has an automatic gradient oracle.**
The oracle calls the same accumulate primitives in reverse order with transposed keys.

**For GARCH MLE** (F18): gradient of log-likelihood through Affine scan = reverse Affine scan.
**For GMM MLE** (F16): gradient of log-likelihood through E-step = backward scatter_multi_phi_weighted.
**For CFA/SEM** (F33): gradient of fit function through model-implied Σ(θ).

---

## L-BFGS: History-Based Curvature

L-BFGS approximates the inverse Hessian from history of parameter/gradient differences:

```
s_k = θ_k - θ_{k-1}     (parameter step)
y_k = g_k - g_{k-1}     (gradient difference)
```

The two-loop recursion computes `H_k^{-1} g_k` from m stored (s, y) pairs — typically m=10-20.
Each loop is O(m·d) where d = parameter dimension. For typical models (d < 1000), this is trivial CPU work.

**Tambear path**: L-BFGS stores m pairs of f64 vectors. No GPU needed for the curvature update —
the expensive part is gradient evaluation (GPU), the update is CPU-side (O(m·d) per step).

```rust
pub struct LbfgsState {
    pub m: usize,              // memory size (typically 10-20)
    pub s: Vec<Vec<f64>>,      // parameter differences, length m
    pub y: Vec<Vec<f64>>,      // gradient differences, length m
    pub rho: Vec<f64>,         // 1 / (y_k · s_k), length m
    pub iter: usize,           // current iteration
}
```

---

## MSR Types F05 Produces

```rust
/// Optimizer state for a single optimization run.
pub struct OptimizerState {
    pub params: Vec<f64>,          // current θ
    pub gradient: Vec<f64>,        // ∇L at current params
    pub loss: f64,                 // L(θ) at current params
    pub step: usize,               // iteration count
    pub converged: bool,
    pub optimizer_kind: OptimizerKind,
    pub moment_state: Option<MomentEstimates>,  // Adam: (m_t, v_t)
    pub lbfgs_state: Option<LbfgsState>,
}

pub enum OptimizerKind {
    Sgd { lr: f64, momentum: f64 },
    Adam { lr: f64, beta1: f64, beta2: f64, eps: f64 },
    Lbfgs { m: usize, c1: f64, c2: f64 },  // Wolfe conditions
    GradientDescent { lr: f64 },
}

/// Adam's moment estimates (Affine scan state).
pub struct MomentEstimates {
    pub m: Vec<f64>,   // first moment (AffineState output)
    pub v: Vec<f64>,   // second moment (AffineState output)
    pub t: usize,      // for bias correction
}
```

Add to `IntermediateTag`:
```rust
OptimizationResult {
    objective_id: DataId,   // hash of the objective function spec
    init_params_id: DataId, // hash of starting parameters
    optimizer: OptimizerKind,
},
```

---

## Gradient Oracle API

The gradient oracle is a trait — each family that needs MLE implements it:

```rust
pub trait GradientOracle {
    /// Evaluate objective and gradient at params.
    /// Returns: (loss, gradient)
    fn value_and_gradient(&self, params: &[f64]) -> (f64, Vec<f64>);

    /// Parameter dimension.
    fn dim(&self) -> usize;

    /// Optional: parameter bounds (for constrained optimization).
    fn bounds(&self) -> Option<Vec<(f64, f64)>> { None }
}
```

Implementations:
- `GarchOracle`: forward Affine scan + backward scan for gradient
- `GmmOracle`: scatter_multi_phi_weighted forward + backward for gradient
- `GlmOracle`: weighted GramMatrix + link function gradient
- `CfaOracle`: model-implied covariance gradient (F33/SEM)

---

## Build Order

**Phase 1 — Gradient Descent + Adam** (no L-BFGS):
1. `GradientOracle` trait (~10 lines)
2. `fn gradient_descent(oracle, lr, max_iter) -> OptimizerState` (~30 lines)
3. `fn adam(oracle, lr, beta1, beta2, eps, max_iter) -> OptimizerState` (~50 lines)
4. `OptimizerState` struct + `IntermediateTag::OptimizationResult`
5. Test: Rosenbrock function (gold standard for optimizer testing); match scipy.optimize.minimize

**Phase 2 — L-BFGS** (needed for GARCH, GMM MLE):
1. `LbfgsState` struct
2. Two-loop recursion for H^{-1}g product (~60 lines)
3. Wolfe line search (~40 lines)
4. `fn lbfgs(oracle, m, max_iter) -> OptimizerState` (~40 lines)
5. Test: match arch Python library GARCH parameter estimates (via GarchOracle)

**Phase 3 — Constrained optimization** (needed for CFA/SEM positive-definite constraints):
- Projected gradient (box constraints)
- Augmented Lagrangian (general linear constraints)
- Deferred: this is F33 territory

---

## What F05 Unlocks

| Family | What it needs | F05 component |
|--------|--------------|---------------|
| F18 GARCH | MLE of (ω, α, β) | L-BFGS + GarchOracle |
| F16 GMM | MLE of (μ_k, Σ_k, π_k) | L-BFGS + GmmOracle (or EM, which is already working) |
| F10 GLM | IRLS = implicit optimizer | Not needed (IRLS is its own convergent scheme) |
| F33 SEM/CFA | MLE of (Λ, Φ, Θ) | L-BFGS + CfaOracle |
| F34 Bayesian | Variational inference, HMC | Adam + gradient oracle for ELBO |
| F09 Robust | IRLS = implicit optimizer | Not needed (same as F10) |
| F14 Factor | MLE FA (Phase 2) | L-BFGS + FactorOracle |

**Important**: IRLS (F09, F10) does NOT need F05. IRLS is its own convergent iterative scheme —
it uses the natural gradient and converges in O(10) iterations without a line search.
F05 is for problems where IRLS doesn't apply (GARCH, GMM with collapsed clusters, SEM).

---

## The Lab Notebook Claim

> Family 05 (Optimization) is two things unified: Kingdom B for the parameter update rule (Adam = EWM on gradients, proven via Affine scan) and Kingdom C for the outer loop (iterate until convergence). The gradient oracle is a backward accumulate pass — gradient duality means no separate autodiff engine is ever needed. Every optimization problem in the math library reduces to: forward accumulate → objective + gradient → optimizer step → repeat. F05 is the backbone that completes Kingdom C for all maximum-likelihood families.
