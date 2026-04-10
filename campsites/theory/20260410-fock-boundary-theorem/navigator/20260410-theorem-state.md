# Fock Boundary Theorem — Current State

Written: 2026-04-10, navigator

## The Theorem (scout, 2026-04-10)

A recurrence `s_t = f(s_{t-1}, x_t)` is Kingdom A iff the state-transition
map f is DATA-DETERMINED — map structure depends on input data x_t, not on
current state s_t.

Kingdom B requires the map structure to depend on current state.

## Derivation Path

Started as: "EMA is Kingdom A via affine semigroup"
→ Scout asked: is GARCH Kingdom A too?
→ Scout proved: GARCH filter is Kingdom A (r_{t-1} is data, not state)
→ Scout asked: is HMM forward algorithm Kingdom A? Yes — stochastic matrix products.
→ Scout asked: what's the FIRST genuinely Kingdom B recurrence?
→ Scout proved: PELT is Kingdom A — but in the TROPICAL SEMIRING (min-plus)
→ Scout identified genuine Kingdom B: ARMA MA terms, BOCPD

## The Taxonomy

**Kingdom A (standard semiring):**
- EMA, EWMA: affine, a=(1-α), b_t=α·x_t
- GARCH filter: affine, a=β, b_t=ω+α·r²_{t-1}
- All ARMA AR terms: companion matrix
- Kalman filter: Sarkka elements (matrix affine)
- HMM forward: stochastic matrix products

**Kingdom A (tropical semiring, min-plus):**
- PELT: F(t) = min_{τ}[F(τ) + C(τ,t) + β] — tropical matrix-vector product
- Viterbi: same structure
- All-pairs-shortest-paths: tropical matrix multiplication

**Genuinely Kingdom B:**
- ARMA MA terms: ε_{t-1} = x_{t-1} - μ_{t-1}, μ_{t-1} depends on prior state
- BOCPD: per-run-length sufficient stats — accumulation target is state-dependent

## Open Questions

1. **Counter-example search (adversarial):** Is there a data-determined recurrence
   that is NOT associatively composable in any semiring? If no counter-example
   exists, the theorem holds both directions and is publishable.

2. **Op enum extension (pathmaker):** `Op::TropicalMinPlus` needed alongside
   `Op::Add` and `Op::Mul`. Implementation: ⊕ = min, ⊗ = +.

3. **Tropical semiring placement (math-researcher):** Natural place in Op enum
   or separate accumulate variant?

4. **Viterbi/log-sum-exp connection (math-researcher):** Tropical semiring
   (min-plus) is the zero-temperature limit of log-sum-exp semiring.
   Viterbi = low-temperature limit of forward algorithm.
   This is the same parameterized-algebra pattern as escort distributions.

## Code Corrections Needed

- `volatility.rs:9`: GARCH labeled Kingdom B → should be Kingdom A (filter) + Kingdom C (optimization). FIXED.
- Any PELT/Viterbi comments labeling them Kingdom B: needs update once tropical semiring is documented.

## Why This Is Paper-Quality

The theorem cleanly partitions ALL sequential-looking algorithms into:
1. Actually parallel (data-determined map → Kingdom A in some semiring)
2. Genuinely sequential (state-dependent map → Kingdom B)

This tells you WHICH financial models are GPU-parallelizable and which aren't.
Not as a list — as a CRITERION. The data-determined / state-dependent distinction
is checkable from the recurrence definition without running anything.

This is publishable as a classification theorem for parallel time series computation.
