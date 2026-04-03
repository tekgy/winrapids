# Family 34: Bayesian Methods — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — C (MCMC/HMC = iterative sampling), B (leapfrog = Affine scan), A (VI ELBO = accumulate)

---

## Core Insight: Three Roads to the Posterior

Every Bayesian computation approximates p(θ|y) ∝ p(y|θ)·p(θ):

1. **MAP (Maximum A Posteriori)**: Point estimate = mode of posterior. Just optimization (F05). Prior = regularization.
2. **Variational Inference (VI)**: Approximate posterior with tractable q(θ). Optimize ELBO via F05 Adam.
3. **MCMC (Markov Chain Monte Carlo)**: Sample from posterior. HMC uses GradientOracle + leapfrog integrator.

All three consume F05 (Optimization). The prior is just an additive term in the GradientOracle.

---

## 1. MAP Estimation

### Formula
```
θ_MAP = argmax_θ [log p(y|θ) + log p(θ)]
```

This IS optimization of log-posterior. The prior adds a regularization term:
- Normal prior N(0, σ²): adds -θ²/(2σ²) = L2 penalty (Ridge)
- Laplace prior: adds -|θ|/b = L1 penalty (Lasso)
- Cauchy prior: adds -log(1 + θ²/γ²) (robust shrinkage)

### Implementation: GradientOracle where eval/grad include prior term. Feed to F05 L-BFGS.

### Zero new infrastructure. MAP = F05 optimization with modified objective.

---

## 2. Variational Inference (VI)

### ELBO (Evidence Lower Bound)
```
ELBO(q) = E_q[log p(y,θ)] - E_q[log q(θ)]
         = E_q[log p(y|θ)] - KL(q || p(θ))
```

Maximize ELBO ≡ minimize KL(q(θ) || p(θ|y)).

### Mean-Field VI
Factorize: q(θ) = Π_i q_i(θ_i). Each factor updated via coordinate ascent:
```
log q_j(θ_j) = E_{q_{-j}}[log p(y, θ)] + const
```

For conjugate-exponential families: closed-form updates. For non-conjugate: use stochastic VI.

### Stochastic VI (ADVI — Kucukelbir et al. 2017)

1. Transform θ to unconstrained space ζ (log for positive, logit for [0,1], etc.)
2. Approximate posterior: q(ζ) = N(μ, Σ) — typically diagonal Σ = diag(σ²)
3. Reparameterization trick: ζ = μ + σ ⊙ ε, ε ~ N(0,I)
4. Estimate ELBO gradient via Monte Carlo:
```
∇_μ ELBO ≈ (1/S) Σ_{s=1}^{S} ∇_ζ [log p(y, g⁻¹(ζ_s)) + log|det J_{g⁻¹}(ζ_s)|] - ∇_μ E_q[log q(ζ)]
```
5. Optimize μ, log(σ) via Adam (F05)

### ELBO as Accumulate
For i.i.d. data: E_q[log p(y|θ)] = Σ_i E_q[log p(y_i|θ)] — sum over data points. This is `accumulate(All, elbo_term_i, Add)`. Kingdom A for fixed q parameters.

### Edge Cases
- Multimodal posterior: VI finds one mode (the one closest to the initialization). It cannot represent multimodality.
- Posterior variance underestimation: mean-field VI systematically underestimates posterior variance. Use full-rank Gaussian or normalizing flows for better coverage.
- ELBO can be positive or negative. Increasing ELBO = better approximation.

---

## 3. MCMC — Metropolis-Hastings

### Algorithm
```
1. Propose θ* ~ q(θ*|θ_t)
2. Compute α = min(1, p(θ*|y)·q(θ_t|θ*) / (p(θ_t|y)·q(θ*|θ_t)))
3. Accept with probability α: θ_{t+1} = θ* if accept, θ_t otherwise
```

### Random Walk Metropolis
q(θ*|θ) = N(θ, Σ_proposal). Symmetric → ratio q terms cancel:
```
α = min(1, p(θ*|y) / p(θ_t|y)) = min(1, exp(log p(y|θ*) + log p(θ*) - log p(y|θ_t) - log p(θ_t)))
```

### Tuning
- Proposal scale Σ: target acceptance rate ~23.4% for high-dimensional, ~44% for 1D (Roberts, Gelman, Gilks 1997)
- Adaptive MCMC: update Σ using running sample covariance (Haario, Saksman, Tamminen 2001)
- Burn-in: discard first ~1000 samples. Assess convergence via R̂ statistic.

### Kingdom: C (inherently sequential — each sample depends on previous)

---

## 4. HMC (Hamiltonian Monte Carlo)

### Key Idea
Augment θ with momentum p. Simulate Hamiltonian dynamics:
```
H(θ, p) = U(θ) + K(p) = -log p(θ|y) + ½p'M⁻¹p
```

### Leapfrog Integrator
```
For L steps with step size ε:
  p_{1/2} = p_0 - (ε/2)·∇U(θ_0)           ← half-step momentum
  for l = 1 to L-1:
    θ_l = θ_{l-1} + ε·M⁻¹·p_{l-1/2}        ← full-step position
    p_{l+1/2} = p_{l-1/2} - ε·∇U(θ_l)       ← full-step momentum
  θ_L = θ_{L-1} + ε·M⁻¹·p_{L-1/2}          ← final position
  p_L = p_{L-1/2} - (ε/2)·∇U(θ_L)           ← final half-step momentum
```

### Leapfrog as Affine Scan (Kingdom B)
Each leapfrog step is:
```
[θ_{l+1}]   [I   ε·M⁻¹] [θ_l]   [      0      ]
[p_{l+1/2}] = [0     I   ] [p_{l-1/2}] + [-ε·∇U(θ_l)]
```
This is an Affine map (linear in state, with input = gradient). For L leapfrog steps: a length-L scan.

**CRITICAL**: The gradient ∇U(θ_l) at each step depends on the current θ, so the scan is not a simple constant-coefficient affine — the input at each step depends on the state. This makes it Kingdom C in general, but the structure IS a scan applied to a sequence of gradient evaluations.

### NUTS (No-U-Turn Sampler — Hoffman & Gelman 2014)
Automatically tunes L (number of leapfrog steps) by detecting when the trajectory starts to turn back:
```
Stop when: θ_L · p_L < 0    (momentum dot position is negative → turning around)
```
Uses a doubling scheme: run 1 step, then 2, then 4, ... until U-turn detected. Multinomial sampling from the valid trajectory.

**NUTS is the default in Stan and PyMC.** Eliminates the need to tune L.

### Mass Matrix M
- **Unit**: M = I. Simple but can be slow for correlated parameters.
- **Diagonal**: M = diag(estimated marginal variances). Standard.
- **Dense**: M = estimated posterior covariance. Best for highly correlated parameters. Expensive for large p.

Adaptation: During warmup, estimate M from sample covariance of chain. Update periodically.

### Step Size ε
Dual averaging (Nesterov 2009): target acceptance probability δ = 0.8 for NUTS:
```
log ε_{m+1} = μ - √m / (γ · (m + t₀)) · H̄_m
```
where H̄_m is the running average of acceptance statistics.

### Edge Cases
- Divergent transitions: step size too large → Hamiltonian not conserved → proposal rejected. Reduce ε.
- Funnel geometries (e.g., hierarchical models): parameterize as non-centered (θ = μ + σ·z where z ~ N(0,1))
- Discrete parameters: HMC cannot handle (needs continuous gradient). Marginalize out or use Gibbs for discrete.

---

## 5. Gibbs Sampling

### Algorithm
For θ = (θ₁, ..., θ_K):
```
for each component k:
    θ_k^{(t+1)} ~ p(θ_k | θ_{-k}^{(current)}, y)
```

### When to Use
- Conditional distributions are available in closed form (conjugate models)
- Mixed continuous/discrete: Gibbs for discrete components, HMC for continuous

### Conjugate Pairs
| Prior | Likelihood | Posterior |
|-------|-----------|-----------|
| Normal | Normal | Normal |
| Gamma | Poisson | Gamma |
| Beta | Bernoulli/Binomial | Beta |
| Dirichlet | Multinomial | Dirichlet |
| Inverse-Gamma | Normal (variance) | Inverse-Gamma |
| Normal-Inverse-Gamma | Normal (μ,σ²) | Normal-Inverse-Gamma |
| Wishart | Normal (Σ) | Wishart |

### Kingdom: C (sequential — each component conditioned on others)

---

## 6. Laplace Approximation

### Formula
Approximate p(θ|y) ≈ N(θ_MAP, H⁻¹) where H = -∇²log p(θ|y)|_{θ_MAP}.

### Steps
1. Find MAP via F05 L-BFGS → θ_MAP
2. Compute Hessian H at θ_MAP (or approximate via finite differences)
3. Posterior ≈ N(θ_MAP, H⁻¹)
4. Evidence ≈ p(y|θ_MAP)·p(θ_MAP)·(2π)^{p/2}·|H|^{-1/2}

### When Good
- Unimodal, roughly Gaussian posterior (common for large n)
- Fast: only needs optimization + Hessian

### When Bad
- Skewed, multimodal, or heavy-tailed posteriors
- Small n or weakly identified parameters

### Implementation: F05 optimizer + diagonal/full Hessian. ~20 lines.

---

## 7. Prior Specification

### Weakly Informative Priors (Gelman et al.)
- **Coefficients**: N(0, 2.5) after standardizing predictors
- **Scale parameters**: Half-Cauchy(0, 2.5) or Half-Normal(0, σ)
- **Correlation matrices**: LKJ(η) — η=1 is uniform, η>1 favors identity

### Horseshoe Prior (for sparsity)
```
θ_j | τ, λ_j ~ N(0, τ²λ²_j)
λ_j ~ Cauchy⁺(0, 1)    (local shrinkage)
τ ~ Cauchy⁺(0, τ₀)     (global shrinkage)
```
τ₀ = (p₀/(p-p₀)) · σ/√n where p₀ = expected number of nonzero coefficients.

### Regularized Horseshoe (Piironen & Vehtari 2017)
Adds slab: λ̃_j² = c²λ²_j/(c² + τ²λ²_j). Prevents unreasonably large coefficients.

---

## 8. Diagnostics

### R̂ (Gelman-Rubin)
```
R̂ = √((n-1)/n + B/(n·W))
```
where B = between-chain variance, W = within-chain variance. R̂ < 1.01 for convergence (strict).

### ESS (Effective Sample Size)
```
ESS = N / (1 + 2·Σ_{k=1}^{K} ρ_k)
```
where ρ_k = lag-k autocorrelation. ESS < 100·n_chains is concerning.

### WAIC (Widely Applicable Information Criterion)
```
WAIC = -2(lppd - p_WAIC)
lppd = Σ_i log(1/S · Σ_s p(y_i|θ_s))
p_WAIC = Σ_i Var_s(log p(y_i|θ_s))
```

### LOO-CV (Leave-One-Out Cross-Validation via PSIS)
Pareto-smoothed importance sampling. k̂ diagnostic:
- k̂ < 0.5: reliable
- 0.5 < k̂ < 0.7: OK
- k̂ > 0.7: unreliable, refit without that observation

---

## Sharing Surface

### Reuse from Other Families
- **F05 (Optimization)**: GradientOracle for MAP + L-BFGS; Adam for VI ELBO optimization
- **F06 (Descriptive)**: Posterior summary statistics (mean, median, quantiles, HPD intervals)
- **F07 (Hypothesis Testing)**: Bayes factors (marginal likelihood ratios)
- **F10 (Regression)**: Bayesian regression = F10 with prior (MAP = Ridge/Lasso)
- **F04 (RNG)**: ALL MCMC methods need random number generation
- **F32 (Numerical)**: Numerical integration for marginal likelihood

### Consumers of F34
- **F11 (Mixed Effects)**: Bayesian LME via HMC
- **F13 (Survival)**: Bayesian survival models
- **F16 (Mixture)**: Bayesian mixture models (Dirichlet process)
- **F18 (Volatility)**: Bayesian GARCH, stochastic volatility via MCMC
- **F35 (Causal)**: Bayesian causal inference

### Structural Rhymes
- **MAP = F05 optimization with prior penalty**: Ridge ↔ Normal prior, Lasso ↔ Laplace prior
- **VI ELBO = accumulate over data**: same as F10 log-likelihood but with KL term
- **HMC leapfrog = Störmer-Verlet integrator**: same as molecular dynamics (symplectic)
- **Gibbs sampling = coordinate-wise conditional**: structural rhyme with EM (F16)
- **NUTS doubling = binary tree search**: structural rhyme with bisection (F32)
- **Laplace = mode + curvature**: same as Fisher information (F10)

---

## Implementation Priority

**Phase 1** — Core inference (~100 lines, as noted in task):
1. Prior distributions (Normal, Cauchy, Half-Cauchy, Laplace, Gamma, Beta)
2. BayesianOracle (wraps GradientOracle + prior log-density + prior gradient)
3. MAP estimation (BayesianOracle → F05 L-BFGS)
4. Laplace approximation (MAP + Hessian)

**Phase 2** — MCMC (~150 lines):
5. Random Walk Metropolis-Hastings
6. HMC (leapfrog integrator + Metropolis correction)
7. NUTS (U-turn criterion + doubling)
8. Step size adaptation (dual averaging)
9. Mass matrix adaptation (diagonal + dense)

**Phase 3** — VI (~100 lines):
10. Mean-field ADVI (automatic transforms + reparameterization)
11. Full-rank ADVI (dense Gaussian approximation)
12. ELBO computation + Adam optimization

**Phase 4** — Diagnostics + advanced (~100 lines):
13. R̂, ESS, trace plots
14. WAIC, LOO-CV (PSIS)
15. Posterior predictive checks
16. Horseshoe prior (regularized)

---

## Composability Contract

```toml
[family_34]
name = "Bayesian Methods"
kingdom = "C (MCMC sampling) + B (HMC leapfrog) + A (VI ELBO accumulate)"

[family_34.shared_primitives]
bayesian_oracle = "BayesianOracle = GradientOracle + prior term"
hmc_leapfrog = "Leapfrog integrator (L steps, step size ε)"
nuts = "NUTS with dual averaging + mass matrix adaptation"
vi_elbo = "ELBO estimation via reparameterization trick"
prior = "Prior distribution (density + gradient + sample)"

[family_34.reuses]
f05_optimizer = "L-BFGS for MAP, Adam for VI"
f04_rng = "Random sampling for MCMC proposals"
f06_descriptive = "Posterior summary statistics"
f07_hypothesis = "Bayes factors"
f10_regression = "Bayesian regression = F10 + prior"

[family_34.provides]
map = "Maximum a posteriori point estimates"
laplace = "Laplace approximation to posterior"
mcmc_samples = "Posterior samples via HMC/NUTS/MH"
vi_approximation = "Variational posterior q(θ)"

[family_34.consumers]
f11_mixed = "Bayesian LME"
f13_survival = "Bayesian survival"
f16_mixture = "Bayesian mixture / Dirichlet process"
f18_volatility = "Bayesian SV, GARCH"
f35_causal = "Bayesian causal inference"

[family_34.session_intermediates]
mcmc_chain = "MCMCChain(model_id) — samples + diagnostics"
vi_params = "VIParams(model_id) — μ, σ of variational posterior"
posterior_summary = "PosteriorSummary(model_id) — mean, sd, quantiles, ESS, R̂"
```
