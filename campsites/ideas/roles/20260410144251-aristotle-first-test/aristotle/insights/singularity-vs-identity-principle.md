<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

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


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

