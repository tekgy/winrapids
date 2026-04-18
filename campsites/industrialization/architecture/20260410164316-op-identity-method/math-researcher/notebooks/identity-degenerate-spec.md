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

# Op Identity Method — Design Specification

## Origin

Aristotle (deconstruction): "the framework says associative but requires monoid"
Adversarial (waves 13-15): found 7 bugs from conflating identity and degenerate
Math-researcher (this doc): formalized as two required trait methods

## The Design Rule

Every scannable Op must provide two methods:

```rust
/// The neutral element of the monoid.
/// Used for padding when n is not a power of 2.
/// MUST satisfy: identity ⊕ x = x ⊕ identity = x for all x.
fn identity(&self) -> State;

/// What to return for invalid/empty/degenerate input.
/// NOT part of the monoid — it signals computation failure.
/// MUST NOT be used for padding.
fn degenerate(&self) -> State;
```

## Why Two Methods, Not One

| Concept | Mathematical role | Used for |
|---|---|---|
| identity | Element OF the monoid | Padding in Blelloch tree |
| degenerate | Signal OUTSIDE the monoid | Empty input, NaN data, singular matrix |

Conflating them produces bugs:
- Padding Max with NaN (degenerate) instead of -inf (identity) → NaN propagates through entire scan
- Padding Welford with NaN-mean (degenerate) instead of zero-count (identity) → corrupted moments
- Returning -inf (identity) for "no data" instead of NaN (degenerate) → false results look valid

## Identity Values for All Ops

| Op | State type | identity() | degenerate() |
|---|---|---|---|
| Add | f64 | 0.0 | NaN |
| Max | f64 | f64::NEG_INFINITY | NaN |
| Min | f64 | f64::INFINITY | NaN |
| ArgMin | (f64, usize) | (f64::INFINITY, usize::MAX) | (NaN, usize::MAX) |
| ArgMax | (f64, usize) | (f64::NEG_INFINITY, usize::MAX) | (NaN, usize::MAX) |
| DotProduct | f64 | 0.0 | NaN |
| Distance | f64 | 0.0 | NaN |
| WelfordMerge | (n,mean,m2) | (0, 0.0, 0.0) | (0, NaN, NaN) |
| AffineMap | (a,b) | (1.0, 0.0) | (NaN, NaN) |
| MatMulPrefix(d) | [f64; d*d] | I_d (identity matrix) | [NaN; d*d] |
| SarkkaMerge | (A,b,C,eta,J) | (I, 0, 0, 0, 0) | (NaN matrices) |
| LogSumExpMerge | (max,sum,val) | (-inf, 0.0, 0.0) | (NaN, NaN, NaN) |

### Verification Property

For every Op, the following must be tested:
```
assert!(combine(identity, x) == x);    // left identity
assert!(combine(x, identity) == x);    // right identity
assert!(combine(identity, identity) == identity);  // idempotent
```

## Why identity() Returns State, Not Option<State>

If an Op doesn't have an identity, it cannot be used in a prefix scan.
Making identity() return Option<State> would:
1. Allow constructing scans over non-monoidal Ops (type-unsafe)
2. Defer the check to runtime (exactly the bug class found in waves 13-15)
3. Require every scan engine callsite to unwrap (boilerplate)

The type system should enforce: **scannable ⟹ monoidal**. A required `identity()` 
method does this at compile time.

## Implementation Path

1. Add `identity()` and `degenerate()` to the Op enum (or trait)
2. Update all scan engines to use `op.identity()` for padding
3. Update all empty-input paths to use `op.degenerate()` for return values
4. Add invariant tests: `combine(identity, x) == x` for random x, for each Op
5. Add adversarial test: all-negative vector with Op::Max, length 2^k-1

## Priority

HIGH. This prevents an entire class of bugs (7 already found empirically).
One-time implementation cost, permanent correctness benefit.


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

