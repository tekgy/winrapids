# Joint Proposal: Kingdom Reclassification Audit

*Scout + math-researcher, 2026-04-10*

## The proposal

Campsite: `theory/kingdom-reclassification-audit`
Owners: scout + math-researcher

**Output**: a table of every recurrence in the codebase — current Kingdom label, the
data-vs-state test result, the correct Kingdom, and the representation that reveals it.

**Predicted result**: EGARCH, BOCPD, TAR, MCMC, Wynn's epsilon, and genuine
regime-switching models survive as Kingdom B. Most other "sequential" algorithms
dissolve to Kingdom A under the right representation.

---

## The ARMA innovations form — Kingdom A proof (math-researcher)

The direct CSS implementation of ARMA(p,q) uses explicit MA residual recursion where
ε_t appears to depend on previous ε_{t-j}. This LOOKS Kingdom B.

The innovations state-space form:
```
x_t = F x_{t-1} + K·e_t         where e_t = y_t - H·F·x_{t-1}
    = (F - K·H·F) x_{t-1} + K·y_t
    = A·x_{t-1} + b_t
```

- A = F - KHF: CONSTANT matrix (F is companion matrix, K is steady-state Kalman gain)
- b_t = K·y_t: DATA-DETERMINED (y_t is the observed series, not hidden state)

This IS an affine recurrence with constant coefficients over max(p,q)-dimensional state.
Kingdom A. The affine prefix scan lifts it.

**Recognition event**: the direct CSS form is a Kingdom B representation of Kingdom A
math. The innovations form reveals the parallelizable structure. The Fock boundary was
representational, not operational — it dissolved under the right algebraic form.

**Current codebase**: `time_series.rs:arma_css_residuals` uses the CSS form. Kingdom B
implementation of Kingdom A computation. Implementation debt: the innovations form would
be parallelizable. The outer MLE optimization wrapping it is Kingdom C (iterative fixed-point).

---

## The two-step test for the audit

**Step 1 — Affine augmentation**: Can you write s_{t+1} = A·s_t + B·data_t with
A, B constant matrices? If yes → Kingdom A (affine prefix scan).

**Step 2 — Finitely-representable semigroup**: Even if nonlinear, does there exist a
semiring under which the maps compose with O(1) representation? If yes → Kingdom A
over that semiring (tropical for DP, LogSumExp for probabilistic).

**Fail both tests** → genuine Kingdom B (map selection requires intermediate state
OR composition representation grows unboundedly).

---

## Predicted classification table (joint)

| Function | Current label | Step 1 result | Step 2 result | Correct Kingdom |
|----------|--------------|---------------|---------------|-----------------|
| garch11_filter | B (was wrong, now fixed) | A (3×3 companion) | — | A |
| arma_css_residuals | B | A (innovations form: A=F-KHF constant) | — | A (repr: innovations) |
| ewma_variance | B | A (EMA = 1D affine) | — | A |
| arma_fit outer MLE | C | — | — | C (iterative fixed-point) |
| egarch11_filter | B | Fails (z_t/σ_t in denominator) | Fails | B — confirmed |
| tar_model | B | Fails (branch selection is state-dep.) | Fails | B — canonical |
| bocpd | B | Fails (state grows with t) | Fails | B — confirmed |
| pelt DP recurrence | B/C | Fails step 1 | A (tropical min-plus) | A (tropical) |
| hmm_forward | A | A (matrix prefix product) | — | A — confirmed |
| kalman_filter | A | A (constant F, H, Q, R) | — | A — confirmed |
| viterbi | B | Fails step 1 | A (tropical max-plus) | A (tropical) |
| bellman_ford | B/C | Fails step 1 | A (tropical Graph) | A (tropical Graph) |
| smc_particle_filter | D | Fails | Fails | D (no exact combine) |
| mcmc (MH) | B | Fails (acceptance is state-dep.) | Fails | B — confirmed |
| wynn_epsilon | BC | Fails (tableau entries needed) | Fails | BC — confirmed |
| silhouette_score | B (was wrong) | A (all pairs independent) | — | A — fixed |
| mod_pow | B (was wrong, now fixed) | A (bits are data, squarings independent) | — | A |
| kaplan_meier | B (was wrong) | A (prefix product associative) | — | A |

---

## What the audit produces

1. Corrected Kingdom docstrings for every mislabeled function
2. List of "Kingdom A math, Kingdom B implementation" cases — implementation debt
3. List of which cases need tropical or AffineSemigroup Op variants to express as accumulate
4. Confirmation that EGARCH, TAR, BOCPD, MCMC remain genuine Kingdom B

The audit is the precondition for the Semiring trait design to be grounded in actual
codebase need rather than speculation.

---

## Sequencing note

The Semiring trait design should be co-developed with this audit:
1. Audit identifies which functions need tropical / LogSumExp / AffineSemigroup ops
2. Semiring trait designed to cover exactly those needs (not speculative)
3. Op enum extended with semiring instances (not flat variants)
4. Kingdom A reclassified functions can use the new ops as accumulate targets

This is the right order. The audit drives the trait design, which drives the Op extension.
