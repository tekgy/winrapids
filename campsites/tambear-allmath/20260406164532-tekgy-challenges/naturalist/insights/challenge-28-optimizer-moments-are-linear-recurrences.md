# Challenge 28 — Optimizer Moment Estimates Are Linear Recurrences

**Date**: 2026-04-06  
**Type B: Parallelization Challenge**

---

## The Traditional Assumption

Gradient optimizers (Adam, RMSProp, AdaGrad, SGD with momentum) must run sequentially — each iteration updates state that the next iteration reads.

## Why It Partially Dissolves

The MODULE COMMENT already notes: "Adam's state (m, v per parameter) is the MSR of the gradient history... Both are accumulate patterns: running averages with decay."

This is true but incomplete. The key next step:

Adam's moment update is a 2×2 linear recurrence. For each parameter i:
```
m[i] = β₁ · m[i] + (1-β₁) · g[i]   ← first moment (EWMA of gradient)
v[i] = β₂ · v[i] + (1-β₂) · g²[i]  ← second moment (EWMA of gradient²)
```

Written as a matrix:
```
[m_t]   [β₁      0   ] [m_{t-1}]   [(1-β₁)·g_t]
[1  ] = [0       1   ] [1      ] + [0          ]
```

Matrix products are associative. Parallel prefix scan over these 2×2 matrices gives ALL moment states simultaneously in O(log T) GPU steps.

**This is IDENTICAL to challenge 13 (GARCH as matrix prefix scan).**

---

## What the Connection Is

GARCH(1,1): `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}` → 2×2 matrix linear recurrence
Adam m:     `m_t = β₁·m_{t-1} + (1-β₁)·g_t`   → 2×2 matrix linear recurrence
SGD+momentum: `v_t = μ·v_{t-1} + g_t`         → 2×2 matrix linear recurrence
RMSProp:    `v_t = γ·v_{t-1} + (1-γ)·g²_t`   → 2×2 matrix linear recurrence

All EWMA-based optimizers are the SAME ALGORITHM with different constants. All parallelizable via the same matrix prefix scan.

This is an instance of the **liftability theorem** (challenge 18): every state transition of the form `state = f(state, data)` where f is a semigroup homomorphism → matrix prefix scan.

---

## What This Enables

### 1. Post-hoc trace analysis (immediately actionable)
Given a fixed gradient sequence `g₁, ..., gT` (e.g., from a training log), you can compute ALL Adam states at ALL timesteps simultaneously:
```rust
// Currently: sequential loop
for t in 0..T {
    m = beta1 * m + (1-beta1) * g[t];
    ...
}

// After challenge 13+28: matrix prefix scan
let matrices = g.iter().map(|g_t| adam_matrix(g_t, beta1)).collect();
let all_states = matrix_prefix_scan(&matrices);
```

This is O(log T) vs O(T). For hyperparameter search over many training trajectories, this could be significant.

### 2. Hyperparameter sensitivity (via automatic differentiation of prefix scan)
The prefix scan is differentiable. ∂(final_state)/∂(beta1) can be computed via reverse-mode through the matrix product tree. This gives exact hyperparameter gradients for free.

### 3. Same GPU kernel for GARCH + Adam + ARMA + all EWMA
Once the matrix prefix scan primitive exists (challenge 13), ALL EWMA processes use the same kernel. GARCH σ², Adam m/v, ARMA errors, exponential smoothing — one kernel, multiple use cases.

---

## The Important Limitation

This parallelization applies to POST-HOC analysis (given a fixed gradient sequence) or to the INNER LOOP of a batch gradient computation where gradients are precomputed.

It does NOT parallelize the OUTER ITERATION of gradient descent (where each iteration's gradient depends on the previous iteration's x). That outer loop is genuinely sequential (the search direction depends on where you are).

But: the inner moment update loop (given g_t, update m and v) IS parallelizable over parameters i. This is embarrassingly parallel and already should be a GPU vector kernel, not a CPU loop. The sequential loop over `i in 0..n` in `adam()` is currently CPU-side and should be emitted as a map kernel.

---

## Connection to Challenge 13

Challenge 13 implementation directly enables challenge 28 for free. The matrix scan primitive is the same. The only difference is the 2×2 matrices involved:

- GARCH: `[[1, ω], [0, β]]·[[r²_t, 1], [0, 1]]` type construction
- Adam: `[[β₁, (1-β₁)·g_t], [0, 1]]` per step

Same primitive, different matrix entries.
