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

# Pitfall: NaN Silently Ignored by Min/Max Operations

**Status: CONFIRMED BUG — tests `min_with_nan_is_nan` and `max_with_nan_is_nan` FAIL**

## What broke

`min_all([3.0, NaN, 1.0])` returns **1** (not NaN).
`max_all([1.0, NaN, 5.0])` returns **5** (not NaN).

NaN in the input is silently discarded. The result looks like a valid answer — but
it was computed from incomplete, contaminated data. This is a **silent failure**.

## Root cause

In `crates/tambear-primitives/src/accumulates/mod.rs`, `execute_pass_cpu`:

```rust
Op::Min => if val < accs[j] { accs[j] = val; },
Op::Max => if val > accs[j] { accs[j] = val; },
```

IEEE 754 defines: **any comparison involving NaN returns false**.
- `NaN < current_min` → false → NaN is NOT placed into the accumulator. Skipped.
- `NaN > current_max` → false → NaN is NOT placed into the accumulator. Skipped.

The accumulator only ever sees the non-NaN values. NaN disappears without a trace.

## Why this is dangerous

A financial dataset with a corrupt tick (NaN) should produce a detectable signal
that something is wrong, not a silently-clean result computed from the good ticks.
The consumer downstream gets a min/max that looks valid but was computed from
fewer data points than claimed.

Correct behavior: if any element evaluates to NaN, the accumulator should become NaN
and stay NaN for the rest of the pass (NaN is "sticky").

## The fix

Two approaches:

**Option A**: NaN-propagating comparisons:
```rust
Op::Min => {
    if val.is_nan() || val < accs[j] { accs[j] = val; }
}
Op::Max => {
    if val.is_nan() || val > accs[j] { accs[j] = val; }
}
```

This makes NaN sticky: once the accumulator is NaN, any comparison `NaN < x`
or `NaN > x` is false, but `NaN.is_nan()` is true, so NaN writes NaN back.

**Option B**: Pre-check at the element level (propagate earlier):
In `execute_pass_cpu`, after computing `val`, check:
```rust
let val = crate::tbs::eval(...);
if val.is_nan() {
    // Set all accumulators to NaN and break/continue
    for acc in accs.iter_mut() { *acc = f64::NAN; }
    break; // or: continue with NaN-sticky accumulators
}
```

Option A is more compositional (doesn't require special-casing per Op).
Option B is faster (short-circuits early on first NaN).

## Impact

Affects: `min_all`, `max_all`, `range_all`, `midrange`, `linf_norm` (uses Max),
and any other recipe using `Op::Min` or `Op::Max` when input contains NaN.

## Related: empty-input identity leaking

The same root (identity initialization) causes `min_empty` → +Inf and
`max_empty` → -Inf. When data is empty, the accumulator never gets written,
so the identity value is returned. This is also wrong (should be NaN for
an undefined result), documented separately in
`pitfalls/identity-value-leaked-on-empty-input.md`.

## Files

- **Failing tests**: `crates/tambear-primitives/tests/adversarial_baseline.rs`
  - `min_with_nan_is_nan`
  - `max_with_nan_is_nan`
- **Buggy code**: `crates/tambear-primitives/src/accumulates/mod.rs:146-148`


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

