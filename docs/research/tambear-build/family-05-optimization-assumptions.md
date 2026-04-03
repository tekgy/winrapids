# Family 05: Optimization — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — B (Adam/momentum = Affine scan), C (L-BFGS/Newton = iterative with line search)

---

## Core Insight: Adam IS an Affine Scan

The state update for Adam (and all momentum-based optimizers) is:
```
state_{t+1} = A · state_t + B · input_t
```
This is an **Affine(2,2)** operator — same as Kalman filter, same as EMA. Kingdom B.

The convergence loop (iterate until gradient small enough) is Kingdom C. But **within each iteration**, the state update is Kingdom B. This means: for fixed-iteration budgets (common in deep learning), the entire optimizer is a scan.

---

## 1. Gradient Descent (Vanilla)

### Update Rule
```
θ_{t+1} = θ_t - η · ∇f(θ_t)
```

### Convergence
- **Convex + L-smooth**: f(θ_t) - f* ≤ L‖θ₀ - θ*‖² / (2t) with η = 1/L
- **Strongly convex (μ > 0)**: ‖θ_t - θ*‖² ≤ (1 - μ/L)^t · ‖θ₀ - θ*‖² — linear convergence
- **Non-convex**: converges to stationary point (∇f = 0), not necessarily minimum

### Learning Rate
- Too large (η > 2/L): diverges
- Optimal for quadratic: η = 2/(λ_max + λ_min), convergence rate (κ-1)/(κ+1) where κ = λ_max/λ_min

### Edge Cases
- Saddle points: GD can get stuck (Hessian has negative eigenvalues). Add noise or use second-order methods.
- Plateau regions: gradient ≈ 0 but far from optimum. Momentum helps.

---

## 2. Momentum (Polyak Heavy Ball)

### Update Rule
```
v_{t+1} = β·v_t + ∇f(θ_t)
θ_{t+1} = θ_t - η·v_{t+1}
```

### As Affine Scan
State = [θ, v]. Update:
```
[v_{t+1}]   [β  0] [v_t]   [1]
[θ_{t+1}] = [-ηβ I] [θ_t] + [-η] · ∇f(θ_t)
```
This is Affine(state_dim, state_dim) where state_dim = 2p (p = parameter dimension).

### Convergence
- Optimal β = ((√κ - 1)/(√κ + 1))² for quadratic
- Convergence rate: (√κ - 1)/(√κ + 1) vs (κ-1)/(κ+1) for vanilla GD
- For κ = 100: momentum converges ~10x faster

### Edge Cases
- β too close to 1 with large η: oscillation/divergence
- Non-convex: momentum can overshoot minima

---

## 3. Nesterov Accelerated Gradient (NAG)

### Update Rule (Sutskever reparameterization for implementation)
```
v_{t+1} = β·v_t + ∇f(θ_t - η·β·v_t)      ← look-ahead gradient
θ_{t+1} = θ_t - η·v_{t+1}
```

Equivalent reformulation (easier to implement):
```
v_{t+1} = β·v_t - η·∇f(θ_t)
θ_{t+1} = θ_t + β²·v_t - (1+β)·η·∇f(θ_t)
```

### Convergence
- Convex: O(1/t²) — provably optimal for first-order methods (Nesterov 1983)
- Strongly convex: (√κ - 1)/(√κ + 1) — same as Polyak heavy ball on quadratics, but with guaranteed rate

### Why Better Than Momentum
Momentum uses gradient at current position. NAG uses gradient at anticipated next position (look-ahead). This prevents overshooting.

---

## 4. Adam (Adaptive Moment Estimation)

### Update Rule
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t           ← 1st moment (mean)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²           ← 2nd moment (uncentered variance)
m̂_t = m_t / (1-β₁ᵗ)                       ← bias correction
v̂_t = v_t / (1-β₂ᵗ)                       ← bias correction
θ_t = θ_{t-1} - η·m̂_t / (√v̂_t + ε)
```

### Default Hyperparameters (Kingma & Ba 2015)
- β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, η = 0.001

### As Affine Scan
State = [m, v, β₁ᵗ, β₂ᵗ]. The m and v updates are EXACTLY EMA (exponential moving average) — which is Affine(1,1). The bias correction is a scalar division by (1 - running_product). The parameter update uses the corrected moments.

For fixed iteration count T:
```
scan(t=1..T, state=[m,v,β₁ᵗ,β₂ᵗ], input=g_t):
  m' = β₁·m + (1-β₁)·g_t       ← Affine
  v' = β₂·v + (1-β₂)·g_t²      ← Affine
  β₁ᵗ' = β₁·β₁ᵗ                ← Affine
  β₂ᵗ' = β₂·β₂ᵗ                ← Affine
```
All four state channels are independent Affine(1,1) scans. The parameter update is a post-scan extraction.

### Convergence Issues
- **Original Adam can diverge** on simple convex problems (Reddi et al. 2018)
- AMSGrad fix: v̂_t = max(v̂_{t-1}, v_t/(1-β₂ᵗ)) — ensures non-increasing step size
- In practice, Adam works fine; AMSGrad rarely needed

### Edge Cases
- g_t = 0 for all t: m_t = 0, v_t = 0, update = 0/ε ≈ 0 (correct)
- Very large gradients: v_t grows, step size shrinks (adaptive clipping)
- ε too small: numerical instability when v̂_t ≈ 0
- t = 0: bias correction divides by 0 (start from t = 1)

---

## 5. AdaGrad

### Update Rule
```
G_t = G_{t-1} + g_t²                      ← accumulated squared gradients
θ_t = θ_{t-1} - η·g_t / (√G_t + ε)
```

### Key Property
Learning rate decays per-parameter proportional to historical gradient magnitude. Good for sparse features (NLP).

### Problem
G_t only grows → learning rate monotonically decreases → premature convergence. This is WHY RMSprop and Adam were invented.

### As Accumulate
G_t is simply `accumulate(All, g², Add)` — a running sum. Kingdom A within each step.

---

## 6. RMSprop (Hinton, unpublished lecture)

### Update Rule
```
v_t = β·v_{t-1} + (1-β)·g_t²              ← EMA of squared gradients
θ_t = θ_{t-1} - η·g_t / (√v_t + ε)
```

### Relation to Adam
Adam = RMSprop + momentum + bias correction. RMSprop is Adam with β₁ = 0.

### Default: β = 0.9, η = 0.001

---

## 7. AdamW (Decoupled Weight Decay)

### Update Rule
```
m_t, v_t = (same as Adam)
θ_t = (1 - λ·η)·θ_{t-1} - η·m̂_t / (√v̂_t + ε)
```

### CRITICAL: AdamW ≠ Adam + L2 regularization
- Adam + L2: adds λ·θ to gradient BEFORE adaptive scaling → penalty is scaled differently per parameter
- AdamW: applies weight decay AFTER adaptive step → uniform regularization
- Loshchilov & Hutter (2019) showed AdamW generalizes significantly better

### This is the standard optimizer for modern deep learning (transformers, LLMs)

---

## 8. L-BFGS (Limited-memory BFGS)

### Background: Newton's Method
```
θ_{t+1} = θ_t - H⁻¹·∇f(θ_t)
```
where H = ∇²f(θ_t) is the Hessian. Quadratic convergence near optimum. But H is p×p — too large to store/invert.

### BFGS Approximation
Maintain approximate inverse Hessian B⁻¹ updated via rank-2 corrections:
```
sₖ = θ_{k+1} - θ_k
yₖ = ∇f(θ_{k+1}) - ∇f(θ_k)
ρₖ = 1 / (yₖ'sₖ)

B⁻¹_{k+1} = (I - ρₖsₖyₖ')B⁻¹_k(I - ρₖyₖsₖ') + ρₖsₖsₖ'
```

### L-BFGS: Store Only m Recent {sₖ, yₖ} Pairs
Instead of maintaining full B⁻¹ (p×p), store the last m pairs {s, y} and reconstruct H⁻¹·g via the **two-loop recursion**:

```
Algorithm: L-BFGS Two-Loop Recursion
Input: g = ∇f(θ_current), pairs {sₖ, yₖ} for k = t-m..t-1

// Forward loop (newest to oldest)
q = g
for i = t-1 down to t-m:
    αᵢ = ρᵢ · sᵢ'q
    q = q - αᵢ · yᵢ

// Initial Hessian approximation
γ = s_{t-1}'y_{t-1} / (y_{t-1}'y_{t-1})
r = γ · q

// Backward loop (oldest to newest)
for i = t-m to t-1:
    βᵢ = ρᵢ · yᵢ'r
    r = r + sᵢ · (αᵢ - βᵢ)

return r    // This is H⁻¹·g
```

### Memory: 2mp floats (m pairs of p-dimensional vectors). Typical m = 5-20.

### Kingdom C: The two-loop recursion is inherently sequential — each iteration depends on the previous. The outer optimization loop is also iterative. Pure Kingdom C.

### Wolfe Line Search (REQUIRED for L-BFGS)

L-BFGS requires the step size α to satisfy the **Wolfe conditions**:
```
(i)  f(θ + α·d) ≤ f(θ) + c₁·α·g'd           (sufficient decrease / Armijo)
(ii) ∇f(θ + α·d)'d ≥ c₂·g'd                  (curvature condition)
```
where c₁ = 10⁻⁴, c₂ = 0.9 (standard for L-BFGS), d = -H⁻¹g (search direction).

**Strong Wolfe** replaces (ii) with |∇f(θ + α·d)'d| ≤ c₂·|g'd|.

**Implementation**: Moré-Thuente (1994) line search — robust, handles non-convex cases, 5-15 function evaluations typically.

### Convergence
- Superlinear convergence for smooth convex problems
- Much faster than first-order methods for well-conditioned problems
- Struggles with: non-smooth objectives, stochastic gradients, very ill-conditioned problems

### Edge Cases
- yₖ'sₖ ≤ 0: skip this pair (curvature condition violated — non-convex region)
- First iteration (no pairs): use steepest descent with Wolfe line search
- m = 0: reduces to steepest descent
- Stochastic gradients: L-BFGS is NOT designed for stochastic optimization. Use Adam.

---

## 9. Conjugate Gradient (Nonlinear)

### Update Rule
```
d₀ = -g₀
d_{t+1} = -g_{t+1} + β_t · d_t
θ_{t+1} = θ_t + α_t · d_t
```
where α_t found by line search.

### β Formulas
| Variant | Formula | Properties |
|---------|---------|------------|
| Fletcher-Reeves | β = g_{t+1}'g_{t+1} / (g_t'g_t) | Simplest, can cycle |
| Polak-Ribière | β = g_{t+1}'(g_{t+1}-g_t) / (g_t'g_t) | Better restart behavior |
| Hestenes-Stiefel | β = g_{t+1}'(g_{t+1}-g_t) / (d_t'(g_{t+1}-g_t)) | Automatic restart |
| Dai-Yuan | β = g_{t+1}'g_{t+1} / (d_t'(g_{t+1}-g_t)) | Sufficient descent |

**Recommendation**: Polak-Ribière+ (β = max(0, β_PR)) — automatic restart, good convergence

### Restart
Reset d = -g every n iterations or when |g_{t+1}'g_t| > 0.2·‖g_{t+1}‖² (Powell restart).

### For Quadratic f(x) = ½x'Ax - b'x
CG solves Ax = b in at most n iterations (exact arithmetic). This IS the linear CG method.

---

## 10. Newton's Method (Full)

### Update Rule
```
θ_{t+1} = θ_t - [∇²f(θ_t)]⁻¹ · ∇f(θ_t)
```

### Implementation: Solve ∇²f · Δθ = -∇f via Cholesky (if ∇²f is PD) or modified Newton.

### Modified Newton
If Hessian not PD, add regularization:
```
(H + τI)Δθ = -g
```
Start τ = 0, increase until H + τI is PD (check Cholesky succeeds).

### Convergence
- Quadratic convergence near optimum: ‖θ_{t+1} - θ*‖ ≤ C·‖θ_t - θ*‖²
- Global convergence with line search or trust region
- Requires: Hessian computation (O(p²) storage, O(p³) solve)

### When to Use
- p small (< 1000): Newton is king
- p moderate (1000-10000): L-BFGS
- p large (> 10000): Adam/SGD

---

## 11. Trust Region Methods

### Idea
Instead of line search (choose direction, find step size), trust region chooses both simultaneously:
```
min_Δθ  m(Δθ) = f + g'Δθ + ½Δθ'HΔθ
s.t.    ‖Δθ‖ ≤ Δ
```
where Δ is the trust region radius.

### Solving the Subproblem
**Cauchy point**: steepest descent within trust region (cheap, used as safeguard).
**Dogleg method**: interpolate between Cauchy point and Newton step (requires H PD).
**Steihaug-Toint CG**: truncated CG within trust region (handles indefinite H).

### Radius Update
```
ρ = (f(θ) - f(θ + Δθ)) / (m(0) - m(Δθ))       ← actual vs predicted reduction
```
- ρ > 0.75: expand Δ (model good)
- ρ < 0.25: shrink Δ (model poor)
- ρ < 0: reject step (worse than current)

### Edge Cases
- H indefinite: Steihaug-Toint handles this gracefully
- Δ → 0: effectively no step (stuck)
- ρ ≈ 1: model is excellent, safe to increase Δ aggressively

---

## 12. Constrained Optimization

### 12a. Barrier (Interior Point) Method
For: min f(x) s.t. gᵢ(x) ≤ 0

Reformulate: min f(x) - (1/t)·Σ ln(-gᵢ(x))

Increase t (barrier parameter) toward ∞. At each t, solve unconstrained problem (Newton/L-BFGS). The -ln(-g) term → ∞ as g → 0, keeping iterates strictly feasible.

### 12b. Penalty Method
min f(x) + (μ/2)·Σ max(0, gᵢ(x))²

Increase μ. Simple but ill-conditioned for large μ.

### 12c. Augmented Lagrangian (ALM)
```
L_A(x, λ, μ) = f(x) + Σ λᵢgᵢ(x) + (μ/2)Σ max(0, gᵢ(x) + λᵢ/μ)²
```
Update λ and μ between outer iterations. Better conditioning than pure penalty.

### 12d. Projected Gradient
For simple constraints (box, simplex):
```
θ_{t+1} = Proj_C(θ_t - η·∇f(θ_t))
```
Proj_C is projection onto constraint set. Trivial for box constraints: clamp.

---

## 13. Stochastic Gradient Descent (SGD) and Variants

### Mini-batch SGD
```
θ_{t+1} = θ_t - η · (1/|B|) Σ_{i∈B} ∇fᵢ(θ_t)
```
where B is a random mini-batch.

### Variance: E[‖g_B - ∇f‖²] = σ²/|B|
Larger batch → lower variance → can use larger η.

### Learning Rate Schedules
| Schedule | Formula | Use Case |
|----------|---------|----------|
| Constant | η_t = η₀ | Baseline |
| Step decay | η_t = η₀ · γ^⌊t/s⌋ | Classic deep learning |
| Cosine | η_t = η_min + ½(η₀-η_min)(1+cos(πt/T)) | Modern default |
| Linear warmup | η_t = η₀ · t/T_w for t < T_w | Transformers |
| 1/t decay | η_t = η₀ / (1 + λt) | Theory-optimal (convex) |
| OneCycleLR | Triangular with anneal | Smith (2019), very fast |

### Gradient Clipping
- By norm: g' = g · min(1, c/‖g‖) — standard for transformers
- By value: g' = clamp(g, -c, c) — simpler but changes direction

---

## GradientOracle Trait

The unifying abstraction across ALL optimization methods:

```rust
trait GradientOracle {
    type Params;
    fn eval(&self, θ: &Self::Params) -> f64;                    // f(θ)
    fn grad(&self, θ: &Self::Params) -> Self::Params;           // ∇f(θ)
    fn eval_grad(&self, θ: &Self::Params) -> (f64, Self::Params); // both (often cheaper)
}
```

Every optimizer calls the oracle; the oracle encapsulates the objective. This is how F05 serves F18 (GARCH MLE), F16 (GMM MLE), F34 (Bayesian MAP), F24 (training losses).

### Optional Extensions
```rust
trait HessianOracle: GradientOracle {
    fn hessian(&self, θ: &Self::Params) -> Matrix;              // ∇²f(θ)
    fn hessian_vector_product(&self, θ: &Self::Params, v: &Self::Params) -> Self::Params; // H·v
}
```
Newton and trust region need Hessian. CG and L-BFGS only need Hessian-vector products (can be approximated by finite differences).

---

## OptimizerState MSR

```rust
struct OptimizerState<P> {
    params: P,              // current θ
    step: u64,              // iteration count
    // First-order state (Adam/momentum):
    m: Option<P>,           // first moment
    v: Option<P>,           // second moment
    // L-BFGS state:
    history: Option<LbfgsHistory<P>>,
    // Convergence:
    grad_norm: f64,
    f_val: f64,
}

struct LbfgsHistory<P> {
    s: VecDeque<P>,         // position differences (most recent m)
    y: VecDeque<P>,         // gradient differences (most recent m)
    rho: VecDeque<f64>,     // 1/(y'·s) precomputed
    m: usize,               // history length
}
```

This IS the MSR for optimization. The OptimizerState at any point carries everything needed to continue optimization — it's the sufficient statistic of the optimization trajectory.

---

## Sharing Surface

### Kingdom Classification per Algorithm

| Algorithm | Kingdom | Why |
|-----------|---------|-----|
| Gradient Descent | B (fixed iter) / C (converge) | θ_{t+1} = θ_t - η·g is Affine(p,p) |
| Momentum | B / C | [θ,v] state is Affine scan |
| NAG | B / C | Same as momentum with look-ahead |
| Adam/AdamW | B / C | [m,v,β₁ᵗ,β₂ᵗ] = 4 independent Affine(1,1) scans |
| AdaGrad | A (accumulate) + step | G_t = running sum of g² |
| RMSprop | B / C | EMA = Affine(1,1) |
| L-BFGS | C | Two-loop recursion is sequential; outer loop iterative |
| Newton | C | Hessian solve per iteration |
| CG | C | Sequential direction updates |
| Trust Region | C | Sequential subproblem solves |
| Barrier/Penalty | C | Outer loop increases parameter |

### Reuse from Other Families
- **F10 (Regression)**: Cholesky solve for Newton step. X'X GramMatrix for Gauss-Newton.
- **F22 (Dimensionality Reduction)**: Eigendecomposition for trust region subproblem (Steihaug-Toint).
- **F32 (Numerical Methods)**: Root-finding for line search bracketing.

### Consumers of F05
- **F18 (Volatility/GARCH)**: MLE via L-BFGS with GradientOracle wrapping GARCH likelihood
- **F16 (Mixture Models)**: EM M-step sometimes needs constrained optimization; full MLE via Adam
- **F34 (Bayesian)**: MAP estimation = optimization of log-posterior
- **F24 (Training)**: ALL deep learning training loops use F05 optimizers
- **F14 (SEM)**: Model fitting via optimization of discrepancy function
- **F10 (Regression)**: Iteratively Reweighted Least Squares (IRLS) for GLMs

---

## Implementation Priority

**Phase 1** — First-order optimizers (~200 lines):
1. GradientOracle trait + OptimizerState struct
2. Gradient descent (vanilla + momentum + NAG)
3. Adam / AdamW (with bias correction, AMSGrad variant)
4. Learning rate schedules (constant, step, cosine, linear warmup)
5. Gradient clipping (by norm, by value)

**Phase 2** — Second-order + line search (~200 lines):
6. Wolfe line search (Moré-Thuente)
7. L-BFGS (two-loop recursion + history management)
8. Conjugate gradient (Polak-Ribière+)
9. Backtracking line search (Armijo only — simpler, for when Wolfe is overkill)

**Phase 3** — Newton + constrained (~150 lines):
10. Newton's method (with modified Newton for indefinite Hessian)
11. Trust region (Cauchy point + dogleg)
12. Projected gradient (box constraints)
13. Augmented Lagrangian (general constraints)

**Phase 4** — Stochastic extensions (~100 lines):
14. SGD with mini-batch support
15. AdaGrad, RMSprop (trivial once Adam exists)
16. Gradient noise injection (Langevin dynamics — connects to F34 Bayesian)

---

## Structural Rhymes

- **Adam state = 4 EMA channels**: same Affine(1,1) as F17 exponential smoothing
- **L-BFGS two-loop = sequential scan with bounded history**: structural rhyme with F17 ARIMA (bounded AR history)
- **Newton step = Cholesky solve**: same as F10 regression normal equations
- **GradientOracle = the universal interface**: every family that does MLE/MAP passes through F05
- **Momentum = leaky integrator**: same transfer function as F03 IIR filter
- **AdaGrad accumulation = running sum of squares**: same as Welford's M2 in F06

---

## Composability Contract

```toml
[family_05]
name = "Optimization"
kingdom = "B (first-order state updates) + C (convergence loops)"

[family_05.shared_primitives]
gradient_oracle = "GradientOracle trait — objective + gradient"
optimizer_state = "OptimizerState MSR — params + moments + history"
line_search = "Wolfe conditions (Moré-Thuente) or Armijo backtracking"
lr_schedule = "LearningRateSchedule trait — step → η"

[family_05.reuses]
f10_cholesky = "Newton step solve"
f06_welford = "AdaGrad accumulation ≈ Welford M2 pattern"
f32_root_finding = "Line search bracketing"

[family_05.provides]
gradient_oracle = "Universal interface for MLE/MAP/training"
optimizer_state = "Resumable optimization state"
first_order = "Adam, AdamW, SGD+momentum, NAG, RMSprop, AdaGrad"
second_order = "L-BFGS, Newton, CG, trust region"
constrained = "Projected gradient, barrier, augmented Lagrangian"

[family_05.consumers]
f18_garch = "GARCH MLE via L-BFGS"
f16_gmm = "GMM MLE via Adam or EM"
f34_bayesian = "MAP estimation"
f24_training = "All neural network training"
f14_sem = "SEM discrepancy optimization"

[family_05.session_intermediates]
optimizer_state = "OptimizerState(model_id) — resumable"
lbfgs_history = "LbfgsHistory(model_id) — last m {s,y} pairs"
```
