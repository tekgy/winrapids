# F34 Sharing Surface: Bayesian Inference as GradientOracle + Kingdom C

Created: 2026-04-01T06:11:51-05:00
By: navigator

Prerequisites: F05 Phase 1+2 complete (GradientOracle, Adam, L-BFGS); F10 (GramMatrix for priors).

---

## Core Insight: Bayesian = Kingdom C with GradientOracle

All Bayesian inference methods iterate an inner computation:

**Variational Inference**: maximize ELBO = E_q[log p(x,z)] - E_q[log q(z)] over variational params φ
- ELBO gradient = GradientOracle
- Optimizer = Adam (F05)
- Kingdom C outer loop

**HMC / NUTS**: simulate Hamiltonian dynamics on the log-posterior surface
- Leapfrog step: position update + momentum update = TWO Affine-like updates
- Gradient of log-posterior = GradientOracle
- Kingdom C outer loop (MCMC chain)

**Laplace Approximation**: fit Gaussian to posterior via Newton at the MAP
- MAP = L-BFGS on negative log-posterior (F05)
- Hessian at MAP = curvature = inferred precision matrix

All three reduce to: (1) compute log-posterior and its gradient via GradientOracle, (2) run a Kingdom C outer loop.

---

## Log-Posterior = Log-Likelihood + Log-Prior

```
log p(θ | x) = log p(x | θ) + log p(θ) - log p(x)
                likelihood      prior    normalizing constant (intractable)
```

**In tambear terms**:
- `log p(x | θ)` = GradientOracle.value_and_gradient() — the same oracle as F05 MLE
- `log p(θ)` = closed-form prior terms added to the oracle output
- The only new code: common prior distributions and their log-densities + gradients

Adding a prior to any MLE problem = adding a regularization term to the GradientOracle. This is
literally Ridge regression (L2 prior) and LASSO (L1 prior) in Bayesian clothing.

---

## Variational Inference (VI)

### Mean-Field VI (simplest)

Approximate posterior: `q(θ) = ∏_i q(θ_i)` (fully factored Gaussian)

Parameters: φ = {μ_i, log σ_i} per latent variable dimension.

ELBO gradient via reparameterization trick:
```
θ ~ q_φ(θ) = μ + σ · ε,  ε ~ N(0,1)   (reparameterized sample)
∂ELBO/∂φ = E_ε[∂log p(x,θ)/∂θ · ∂θ/∂φ] - ∂KL[q||p]/∂φ
```

The gradient through the likelihood is exactly GradientOracle (F05). The KL term has closed form
for Gaussian q and Gaussian prior p.

**Tambear path**:
- Sample ε ~ N(0,1): `cuRAND` or CPU RNG (F04 infrastructure)
- Reparameterize: `θ = μ + σ ⊙ ε` — element-wise
- GradientOracle.value_and_gradient(θ) → log-likelihood gradient
- KL gradient: closed-form for conjugate priors
- Adam step on φ (F05 AffineState)

This is the ELBO-based black-box VI used in PyMC, Stan's ADVI, and pyro.

### ELBO as MSR

```rust
pub struct ElboState {
    pub elbo: f64,              // current ELBO estimate
    pub kl_divergence: f64,     // KL[q||p] component
    pub expected_ll: f64,       // E_q[log p(x|θ)] component
    pub variational_params: Vec<f64>,  // μ, log σ concatenated
    pub step: usize,
    pub converged: bool,
}
```

---

## HMC (Hamiltonian Monte Carlo)

### The Leapfrog Integrator

HMC simulates Hamiltonian dynamics: `H(θ, r) = -log p(θ|x) + 0.5 r'M^{-1}r`

Leapfrog step (ε = step size, M = mass matrix):
```
r_{t+ε/2} = r_t + (ε/2) ∇log p(θ_t|x)        // half-step momentum
θ_{t+ε}   = θ_t + ε M^{-1} r_{t+ε/2}          // full-step position
r_{t+ε}   = r_{t+ε/2} + (ε/2) ∇log p(θ_{t+ε}|x) // half-step momentum
```

**This IS Affine scan** (Kingdom B) for fixed ε and M:
```
state = (θ, r)
A = [[I, εM^{-1}], [0, I]]    (position update)
b = (0, ε·∇log p)              (gradient kick)
```

But A and b change every step (because ∇log p depends on θ) — this is a NON-LINEAR Affine scan.
The gradient IS from GradientOracle. Each leapfrog call = one oracle evaluation.

**NUTS (No-U-Turn Sampler)**: adaptive leapfrog length via tree doubling. Kingdom C outer loop
around leapfrog steps. The tree doubling structure is CPU-side bookkeeping; all expensive work
is GradientOracle calls.

**Tambear path**:
- GradientOracle.value_and_gradient(θ) per leapfrog step
- Mass matrix M: diagonal (adaptive diagonal, AdaGrad-style)
- Metropolis acceptance: standard Bernoulli draw
- Chain sampling = Kingdom C outer loop (L leapfrog steps per proposal)

---

## Laplace Approximation

Fastest Bayesian method — the "fast path" before VI/HMC:

```
1. Find MAP: θ* = argmax log p(θ|x) via L-BFGS + GradientOracle (F05)
2. Hessian: H = -∂²log p(θ|x)/∂θ² evaluated at θ*
3. Posterior ≈ N(θ*, H^{-1})
4. Evidence approximation: log p(x) ≈ log p(x|θ*) + log p(θ*) + (p/2)log(2π) - 0.5 log|H|
```

Step 2 is the bottleneck: computing the full p×p Hessian costs p GradientOracle calls (finite
differences) or one reverse-mode AD call.

**For tambear**: diagonal Hessian approximation (AdaGrad accumulation of gradient squares).
This is `scatter_phi("g^2", all)` — a one-pass accumulate. Diagonal Laplace is cheap and
often sufficient for unimodal posteriors.

Full Hessian via Gauss-Newton: `H ≈ J'J` where J = Jacobian = GramMatrix of gradients.
`GramMatrix(gradient_matrix)` via F10's tiled accumulate.

---

## Common Priors (CPU-side additions to oracle)

```rust
pub enum Prior {
    Improper,           // log p(θ) = 0 (no regularization)
    Normal { mu: Vec<f64>, sigma: Vec<f64> },  // log p = -||θ-μ||²/(2σ²)
    Laplace { mu: Vec<f64>, b: Vec<f64> },     // log p = -|θ-μ|/b (LASSO prior)
    HalfNormal { sigma: f64 },                  // for positive parameters (scale, variance)
    Beta { a: f64, b: f64 },                    // for probability parameters
    Gamma { shape: f64, rate: f64 },            // for rate parameters
    Dirichlet { alpha: Vec<f64> },              // for simplex (mixture weights)
}
```

**Ridge = Normal prior** on β: `log p(β|σ_prior²) = -||β||²/(2σ_prior²)` adds `β/σ_prior²` to gradient.
**LASSO = Laplace prior** on β: adds `sign(β)/b` to gradient (subgradient for L1).

These are the Bayesian interpretations of regularized regression. Zero new infrastructure.

---

## MSR Types F34 Produces

```rust
pub struct PosteriorSummary {
    /// Point estimates:
    pub mean: Vec<f64>,          // E[θ|x]
    pub median: Vec<f64>,        // quantile 0.5
    pub mode: Vec<f64>,          // MAP estimate (from L-BFGS)

    /// Uncertainty:
    pub std: Vec<f64>,           // posterior std dev
    pub ci_lower: Vec<f64>,      // 95% credible interval lower
    pub ci_upper: Vec<f64>,      // 95% credible interval upper

    /// Diagnostics (MCMC only):
    pub r_hat: Option<Vec<f64>>,  // Gelman-Rubin convergence (multi-chain)
    pub ess: Option<Vec<f64>>,    // effective sample size

    /// ELBO history (VI only):
    pub elbo_history: Option<Vec<f64>>,

    pub method: InferenceMethod,
    pub n_params: usize,
}

pub enum InferenceMethod {
    Laplace,
    Variational { n_samples: usize, n_steps: usize },
    Hmc { n_chains: usize, n_warmup: usize, n_samples: usize, step_size: f64 },
}
```

Note: posterior SAMPLES are NOT stored in TamSession by default (too large).
PosteriorSummary is the MSR — downstream uses mean/CI/std, not raw samples.

---

## Build Order

**Phase 1 — Laplace + diagonal VI** (simplest Bayesian):
1. `Prior` enum + `fn log_prior_and_gradient(prior, params) -> (f64, Vec<f64>)` (~50 lines)
2. MAP = F05 L-BFGS on combined oracle (likelihood + prior) (~10 lines wrapping F05)
3. Diagonal Hessian via `scatter_phi("g²", all)` (~10 lines)
4. `PosteriorSummary` struct from Laplace approximation (~20 lines)
5. Tests: match PyMC's `pm.find_MAP()` and diagonal Laplace summary

**Phase 2 — Mean-field VI**:
1. Reparameterization gradient: sample ε, compute θ = μ + σ⊙ε, call oracle (~30 lines)
2. KL term for Normal q vs Normal prior (~20 lines closed form)
3. Adam on ELBO (reuse F05 Adam with new oracle) (~10 lines)
4. Tests: match Stan ADVI / PyMC `ADVI` on simple models

**Phase 3 — HMC (if needed)**:
1. Leapfrog step: oracle call + Affine updates (~40 lines)
2. NUTS tree doubling: CPU bookkeeping (~100 lines)
3. Adaptive step size (dual averaging) (~40 lines)
4. Tests: match Stan's HMC on canonical models (8-schools, stochastic volatility)

---

## What F34 Unlocks

| Family | How F34 helps |
|--------|--------------|
| F18 GARCH | Bayesian GARCH via HMC (posterior on (ω,α,β)) |
| F16 GMM | Bayesian GMM with Dirichlet prior on π (stick-breaking) |
| F11 LME | Bayesian mixed effects with hyperpriors on σ_b² |
| F10 GLM | Bayesian logistic with regularizing priors = MAP estimate |
| F33 SEM | Bayesian CFA with informative priors on loadings |

**Important**: for most fintek applications, F34 is NOT needed. MLE + confidence intervals is
sufficient. F34 is primarily for: small-n problems where priors matter, hierarchical models with
complex posterior geometry, and uncertainty quantification beyond confidence intervals.

---

## Gold Standards

```python
import pymc as pm
import arviz as az

# Simple Bayesian regression — Phase 1 target
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    mu = alpha + beta * X
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Laplace: find MAP
    map_estimate = pm.find_MAP()

    # VI: ADVI
    approx = pm.fit(method='advi', n=10000)

    # HMC: NUTS
    trace = pm.sample(1000, tune=1000)
    az.summary(trace)  # mean, std, hpd_3%, hpd_97%, r_hat, ess_bulk
```

Match target: PyMC's Laplace MAP within 1e-4. ADVI means within 1e-3 (stochastic). HMC R̂ < 1.01.

---

## The Lab Notebook Claim

> Bayesian inference is Kingdom C (outer MCMC/VI loop) around GradientOracle (log-posterior = log-likelihood + log-prior, same oracle as F05 MLE). Laplace approximation = MAP finding via L-BFGS (F05) + diagonal Hessian via one scatter_phi. Mean-field VI = Adam (F05 AffineState) on the ELBO, where the gradient passes through a reparameterized GradientOracle call. HMC leapfrog = data-driven Affine scan (Kingdom B with θ-dependent b). F34 adds ~100 lines for Phase 1 (Laplace + diagonal VI) on top of F05's infrastructure. The prior terms are regularization with a probability interpretation — Bayesian is just regularized MLE in a Kingdom C wrapper.
