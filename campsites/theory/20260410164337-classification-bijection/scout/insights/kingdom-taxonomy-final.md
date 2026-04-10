# Kingdom Taxonomy — Final Session Results

*Scout, 2026-04-10 (session close)*

## The operational criterion

After seven independent convergences, the criterion is:

**Write the recurrence as s_{t+1} = M(data_t) · s_t.**

- If M is determined entirely by input data (not by current state s_t) → Kingdom A
- If M's SELECTION depends on current state s_t → Kingdom B

This is the single question to ask. Everything else follows.

---

## The canonical examples

**Kingdom A — GARCH filter**

Variance update: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

Write as companion matrix:
```
[σ²_t ]   [β  α] [σ²_{t-1}]   [ω]
[r²_t ] = [0  0] [r²_{t-1}] + [r²_t]
[1    ]   [0  0] [1        ]   [1  ]
```

M = fixed matrix. r²_t is input data, not current state. Kingdom A.
The sequential implementation is implementation debt — the model is a 3×3 affine prefix scan.

**Kingdom B — TAR model (Threshold AR)**

Regime selection: x_t = a·x_{t-1} + ε_t if x_{t-1} ≤ c, else x_t = b·x_{t-1} + ε_t

M switches between multiplier a and multiplier b based on x_{t-1} — the current state.
No companion matrix exists because the STRUCTURE of the map (which branch) depends on
the state value. Neither Test 1 nor Test 2 passes. Genuine Kingdom B.

TAR is canonical because it has no representational escape. It's not that we haven't
found the right algebraic form — the branch selection IS the self-reference.

---

## The complete classification table

| Function | M depends on data or state? | Sufficient statistic finite? | True Kingdom |
|----------|-----------------------------|------------------------------|--------------|
| garch11_filter | data (r²_t external) | — | A |
| egarch11_filter | state (z_t = r_t/σ_t couples state) | Yes | B |
| ewma_variance | data (r²_{t-1} external) | — | A |
| arma_css_residuals | state (MA residuals ε_{t-j} are computed state values, not data) | Yes | B |
| tar_model | state (branch selection: a vs b depends on x_{t-1}) | Yes | B — canonical |
| smc_particle_filter | state (resampling depends on accumulated history) | No | D |
| bocpd | state (per-run-length stats shift and extend each step; map not data-determined) | Yes (growing) | B |
| pelt_changepoint | split: DP recurrence = tropical A; pruning optimization = B (uses f[t] to prune candidates) | A+B hybrid | A (underlying) / B (optimized) |
| hmm_forward | data (transition matrix is fixed parameter) | — | A |
| kalman_filter | data (A, H, Q, R are parameters not state) | — | A |

---

## The key lemma discovered late

**Linear functions of state are still affine recurrences.**

If s_{t+1} = A · s_t + B · data_t where A and B are fixed parameters, this is always
Kingdom A via companion matrix augmentation, even if it "looks recursive" at first
glance. The state appears on the right-hand side, but the MAP itself (A, B) is fixed.

ARMA residuals: the AR part propagates y_t values via companion matrix — Kingdom A.
The MA part chains residuals ε_{t-j} where each residual is computed from previous
residuals. The residuals are state (computed values), not data (observed values). The
MA map at time t contains actual numerical residuals you can only know by running
prior steps sequentially. ARMA(p, q>0) = Kingdom B. Correction from scipy-gap-scan:
the companion matrix argument works for the AR part only.

EGARCH: z_t = r_t / σ_t couples the standardized residual to the current volatility.
The variance equation uses z_t rather than r_t directly. The MAP now requires a
division by the current state σ_t. The map ITSELF depends on the state value. Genuine
Kingdom B.

The lemma distinguishes: state appearing in the UPDATE TERM is Kingdom A. State
appearing in the MAP ITSELF is Kingdom B.

---

## The two failure modes for Kingdom B

From the session's internal-decomp-scan analysis:

**Failure mode 1 — Map not data-determined:**
Can't write s_{t+1} = M(data_t) · s_t because the map M requires state.
Examples: tanh-RNN (tanh introduces state-dependent nonlinearity), TAR (branch selection).

**Failure mode 2 — Composition not finitely closed:**
The map algebra (set of all possible M values) doesn't form a finitely-representable
semigroup closed under composition.
Examples: BOCPD (state space grows with t — each new timestep adds one dimension to the
transition matrix, so the set of possible matrices is unbounded).

TAR fails mode 1. BOCPD fails mode 2. Both are genuine Kingdom B but via different mechanisms.

---

## The representational vs operational distinction

From Aristotle's analysis:

A **representational** Fock boundary dissolves when you find the right algebraic form.
GARCH looked like Kingdom B (sequential variance recursion) until the companion matrix
representation revealed it as Kingdom A. The barrier was in the representation.

An **operational** Fock boundary is irreducible. Newton's method:
g(g(x)) requires evaluating f at g(x), which requires the numerical value of g(x).
No algebraic form can bypass this. TAR similarly: the threshold comparison x_{t-1} ≤ c
requires the numerical value, and the map branches on it. No algebraic shortcut.

**Practical rule**: When you see an apparent Kingdom B, ask "representational or operational?"
Most are representational. Exhaust algebraic representations first (companion matrix,
associative composition via Blelloch). If none work after systematic search → operational
Fock boundary → genuine Kingdom B.

---

## The semiring flag (architectural)

HMM forward, Viterbi, soft-Viterbi, and softmax are ALL the same prefix scan under
different semirings:
- HMM forward: LogSumExp semiring (log-domain sum-product)
- Viterbi: Tropical-max semiring (max-plus)
- Soft-Viterbi: same as HMM forward (expectation over paths)
- Softmax: same LogSumExp but with gather step

The `log_sum_exp` function is private inside `hmm.rs`. This hides the unity.

**Architectural flag**: Before pathmaker extends the Op enum with TropicalMinPlus and
TropicalMaxPlus variants, design a `Semiring<T>` trait. The semiring instances should
be parameters, not enum variants. Op::PrefixScan(semiring: &dyn Semiring<T>) is the
right shape. Separate enum variants for each semiring is the wrong shape — it scales
as O(semirings) instead of O(1).

This is the most important pending architectural decision from this session.

---

## The size criterion (sharpest formulation)

From scipy-gap-scan: **A computation is Kingdom B iff the state required to compose
two adjacent segment-summaries grows with segment length.**

- Affine recurrences: boundary state is k×k matrix regardless of segment length. Kingdom A.
- BOCPD: boundary state grows with t. Kingdom B.
- Sample entropy: combining segments requires all subsequences. Kingdom B.
- Exact median: no finite order-statistic summary. Kingdom B.
- DTW: boundary column grows with query length (Kingdom B in general); fixed-template
  windowed DTW has fixed boundary size (Kingdom A). Parameterization determines kingdom.

This criterion gives the most direct test when the algebraic form isn't obvious:
measure whether the boundary state object's size grows with segment length.
