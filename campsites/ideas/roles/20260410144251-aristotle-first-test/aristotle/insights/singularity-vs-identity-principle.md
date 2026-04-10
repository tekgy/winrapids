# The Singularity vs Identity Principle

*2026-04-10 — Aristotle, after adversarial wave 13*

## The Bug

`tridiagonal_scan_element(a, b=0, c)` returns the 3x3 identity matrix when the pivot b is zero. The comment says "avoid NaN propagation." But identity in a scan means "this element doesn't exist." A singular row is not non-existent — it's broken. The scan should FAIL, not silently skip the row.

## The General Principle

**In any scan-based system, the identity element must be reserved EXCLUSIVELY for structural purposes (padding, neutral element). It must NEVER be used as an error sentinel.**

Identity conflates two meanings:
- "I am the neutral element" (structural: padding in Blelloch tree)
- "I am broken and should be ignored" (semantic: singularity)

These are OPPOSITE meanings. The neutral element says "I contribute nothing." A singularity says "I make the whole system undefined." Mapping both to the same value is a type error — a semantic collision.

## The Affected Op Variants

Every Op with a degenerate case:

| Op | Degenerate Input | Current Risk | Correct Response |
|----|-----------------|-------------|-----------------|
| MatMulPrefix(3) | b_i = 0 (singular pivot) | Returns identity (BUG) | Return singularity sentinel |
| AffineCompose | A = 0 (non-invertible) | Not yet implemented | Must not return (1, 0) identity |
| SarkkaMerge | sigma = 0 (zero innovation) | Not yet implemented | Must not return identity 5-tuple |
| LogSumExpMerge | all inputs = -inf | Not yet implemented | Must not return (−inf, 0, 0) |
| WelfordMerge | n = 0 (empty partition) | Likely returns (0, 0, 0) identity | Correct IF identity is documented |

## The Design Rule

For every AssociativeOp:
1. Define the identity element (compose(I, x) = x = compose(x, I))
2. Define the singularity condition (when the input makes the computation undefined)
3. Ensure identity ≠ singularity response
4. The scan engine's extract phase must check for singularity markers in the output

This is a CORRECTNESS invariant for the parallel scan infrastructure. Sequential code (Thomas algorithm) catches singularities in its own path. Parallel code (Blelloch tree) will NOT catch them unless the scan elements carry the singularity signal through the combine chain.

## Connection to First Principles

From the debut deconstruction: "The framework says 'associative' everywhere. It means 'monoidal.'" The identity element is part of the monoidal structure. The adversarial's bug shows that getting the identity wrong doesn't just affect padding — it affects the MEANING of degenerate inputs in the entire parallel execution model.

The monoid axiom `compose(I, x) = x` is a PROMISE: applying the identity doesn't change the result. If singularity returns identity, then `compose(singularity, x) = x` — the singularity is invisible. The promise is kept but the semantics are violated.

## Wave 14 Confirmation: log_sum_exp

The adversarial applied this principle to `log_sum_exp` and found the EXACT same bug pattern:

`log_sum_exp([NaN]) = -Inf` because `f64::max(NEG_INFINITY, NaN) = NEG_INFINITY`. The fold identity (NEG_INFINITY) swallows the singularity (NaN) via IEEE 754's max semantics. The result is indistinguishable from the empty-input case.

Asymmetry: `log_sum_exp([1.0, NaN, 2.0])` correctly returns NaN because the infection propagates through `exp()`. The bug is ONLY in the all-NaN case — the degenerate case where no real values rescue the propagation.

Fix: NaN check before fold. Committed at a6fa8ce.

**Total bugs from this principle: 3** (1 tridiagonal identity, 2 log_sum_exp NaN-swallowing). All from the same root cause: identity element absorbing invalid inputs instead of propagating failure.

**SarkkaMerge test pre-written** by adversarial (wave 14): catastrophic cancellation in eta correction term. J_b=1e8, b_a=1e8, eta_b=1e16+1. Correct answer: 1.0. f64 answer: 0.0. Fix: Kahan compensated summation. Test awaits SarkkaMerge implementation.

## The Complete Adversarial Suite for New Op Variants

When any new Op is implemented, three tests should run:
1. **Singularity-vs-identity**: degenerate input must NOT return identity
2. **Associativity**: compose(compose(a,b),c) vs compose(a,compose(b,c)) for large dynamic range
3. **Non-power-of-2 padding**: n=17, 31, 100 must match sequential computation

Together these attack identity semantics, combine precision, and tree padding — the three failure modes of parallel scan operators.

## Status

Principle confirmed across 3 bugs in 2 waves (13, 14). Design rule validated. SarkkaMerge test pre-written. The principle is now battle-tested.
