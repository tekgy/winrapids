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

# Adversarial Rigor Gauntlet — Navigator Session Log

## What happened

The adversarial agent wrote `tests/adversarial_rigor_gauntlet.rs` (67 tests) that caught 8 real bugs. Fixed all 8. All 67 green.

## Bugs found and fixed

### tie_count — assumed sorted input
`tie_count(&[1.0, 2.0, 1.0])` returned 0 ties because the scan compared adjacent elements without sorting first. Fixed: sort + NaN-filter internally before scanning.

### blomqvist_beta — signum(0.0) = 1.0 in Rust
IEEE 754: `0.0f64.signum()` returns 1.0, not 0.0. Median-tied elements (dx = 0.0 or dy = 0.0) were being counted as concordant instead of excluded. Also: NaN inputs silently treated as discordant (-1). Fixed with raw `dx == 0.0` check + early NaN guard.

### hoeffdings_d — NaN silently dropped
NaN observations were excluded from ECDF counts without propagating NaN to the result. Fixed with early NaN guard.

### distance_correlation — overflow for extreme-scale inputs
`(1e300 - (-1e300)).abs()` = Inf. dCor is scale-invariant, so pre-normalize by max(|x|) before computing distances. Fixed.

### log_returns — Inf price passed the guard
`Inf <= 0.0` is false, so infinite prices slipped through. `ln(Inf/100) = Inf`. Fixed by adding `!is_finite()` check.

### adversarial_rigor_gauntlet test itself — wrong expectation
`blomqvist_perfect_positive` used odd n=9 with X=Y, expecting 1.0. With the signum fix, the median element (dx=0) is correctly excluded, giving 8/9. Fixed test to use even n=8.

## Commit

`0e9d42b` — "Rigor gauntlet: 8 bug fixes + 12 information theory primitives + 74 tests"


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

