# Challenge 34 — Particle Methods Are Kingdom A

**Date**: 2026-04-06  
**Type A: Classification Error — wrong kingdom from wrong level of abstraction**  
**Found in**: `bayesian.rs` module docstring: "MCMC = sequential sampling (Kingdom B)"

---

## The Traditional Assumption

All MCMC is sequential. The Bayesian module is Kingdom B by definition.

## Why It's Half-Wrong

Markov chain Monte Carlo (MH, Gibbs, HMC) IS sequential — each step requires the previous state. But **Sequential Monte Carlo (particle filters, importance sampling)** is NOT sequential. It's embarrassingly parallel.

The module conflates two fundamentally different computational structures under the label "MCMC."

---

## The Structure of Particle Methods

A particle filter maintains N particles `{x_i^(t)}_{i=1}^N`, each representing a hypothesis about the current state. One step:

1. **Propagate**: `∀i: x_i^(t) ~ p(x | x_i^(t-1))`  
   → `accumulate(All, transition_fn(particle_i), ...)` — N independent propagations
   
2. **Weight update**: `w_i ∝ p(observation | x_i^(t))`  
   → `accumulate(All, log_likelihood(observation, particle_i), Add)` + softmax normalization
   
3. **Resample**: draw N new particles according to weights  
   → `gather(particles, categorical_sample(weights))` — weighted random permutation
   
4. **Estimate**: `E[f] = Σ_i w_i · f(x_i^(t))`  
   → `accumulate(All, f(particle_i) * weight_i, Add)` — weighted mean

All 4 steps are `{accumulate, gather}`. The ONLY sequential component is the random draw in step 3 — and even that can be parallelized via systematic resampling (a deterministic scan over the cumulative weight vector).

---

## Why This Matters

N = 10,000 particles running on GPU = real-time Bayesian filtering. Current implementation: not implemented at all. Classification: wrong.

Particle methods are the GPU-native form of Bayesian inference. The conjugate update (Bayesian linear regression at line 110) is the ANALYTIC version. Particle methods are the NUMERICAL version that works when conjugacy fails.

---

## The Missing Taxonomy

The Bayesian module has three layers of inference, not two:

| Method | Kingdom | Correctness | Coverage |
|---|---|---|---|
| Conjugate updates (BLR) | A — closed form accumulate | Exact (under model) | Implemented |
| Particle filter / SMC | A — parallel accumulate + gather | Approximate but unbiased | NOT IMPLEMENTED |
| MH-MCMC | B — sequential Markov chain | Asymptotically exact | Implemented |
| HMC / NUTS | B — sequential gradient dynamics | Best practice for posteriors | NOT IMPLEMENTED |

The entire middle row (particle methods) is missing and misclassified.

---

## The Conjugate Update Is Also An Accumulate (bonus observation)

`bayesian_linear_regression` at lines 121-128:
```rust
for i in 0..n {
    for j in 0..d {
        for k in 0..d {
            lambda_n[j * d + k] += x[i * d + j] * x[i * d + k];
        }
    }
}
```

This computes `X'X = Σ_i x_i · x_i'` (sum of outer products). This IS:
```
accumulate(All, outer_product(x_i), Op::Add)
```

The sequential triple loop is concealing a parallel Gram matrix accumulation. Every conjugate family's sufficient statistics are `accumulate(All, ...)` patterns:

| Conjugate family | Sufficient stats | Accumulate operation |
|---|---|---|
| Normal-Normal | (Σx_i, n) | (accumulate Add, Count) |
| Normal-InvGamma | (Σx_i, Σx_i², n) | WelfordMerge state |
| Normal-Wishart | (Σx_i, X'X) | (accumulate Add, outer product Add) |
| Dirichlet-Multinomial | (count per category) | accumulate ByKey with Count |

When `Op::WelfordMerge` exists (challenge 32), `bayesian_linear_regression` reduces to: accumulate WelfordMerge for σ² update + accumulate outer-product for Λ_n update. One pass, fully parallel.

---

## Most Actionable

1. Add `Op::OuterProductAdd` — accumulates rank-1 outer products `x·x'` to a symmetric matrix. This makes the BLR inner loop parallel and GPU-ready.

2. Implement `particle_filter(initial_particles, transition, log_likelihood, observations)` using `accumulate(All, ...) + gather(resampling_permutation)`. The particle filter is literally accumulate + gather in sequence. No new infrastructure.

3. Fix the Kingdom classification: bayesian.rs is not "Kingdom B" — it's a module with all three kingdom types.
