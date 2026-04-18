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

# Pitfall: Variance Catastrophic Cancellation (One-Pass Formula)

**Status: CONFIRMED BUG — test `variance_catastrophic_cancellation_exposed` FAILS**

## What broke

The one-pass variance formula `(Σx² - (Σx)²/n) / (n-1)` catastrophically fails
when the mean is large relative to the standard deviation.

**Test data**: 1000 values `1e9, 1e9+1e-6, 1e9+2e-6, ..., 1e9+999e-6`
- True sample variance: ~8.34e-8
- Computed: **-4592.1** (negative! 55 billion times wrong)

**Second test** (`variance_welford_vs_onepass_stress`): 10000 values alternating
`1e8+1` and `1e8-1`. True variance: ~1.0. Computed: **0.0** exactly (total destruction).

## Root cause

The formula computes `Σx²` and `(Σx)²/n` as two large numbers, then subtracts them.
When mean >> std, these numbers agree to many significant digits, and the subtraction
destroys all signal through catastrophic cancellation.

Specifically: with mean = 1e9 and spread = 1e-3:
- `Σx²` ≈ n * (1e9)² = 1e21 (accumulated in 64-bit float)
- `(Σx)²/n` ≈ n * (1e9)² = same

The true difference is ~n * 1e-3 * 1e9 * (something small) ~ 1e3, but the
two terms each have magnitude 1e21. In f64 (53 bits of mantissa = ~15.9 decimal
digits), two 1e21 numbers that agree to 15 digits leave NO significant bits for
the difference.

**The negative result** comes from fp rounding: the accumulated `Σx²` is slightly
LESS than `(Σx)²/n` due to rounding direction during accumulation, producing a
negative "variance."

## The fix: Welford's online algorithm

Welford (1962) maintains running mean and sum of squared deviations:
```
M_1 = x_1;  S_1 = 0
M_k = M_{k-1} + (x_k - M_{k-1}) / k
S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
variance = S_n / (n - 1)
```

This is numerically stable because `(x_k - M_k)` and `(x_k - M_{k-1})` are
always small deviations from the running mean, never large absolute values.

## Impact

**Any recipe using variance/std_dev/covariance/pearson_r on data with large mean
and small spread returns garbage.** This covers essentially all financial data
(prices are large, movements are small).

## Accumulate decomposition challenge

The current accumulate+gather architecture assumes the accumulate step is a
simple elementwise-then-combine. Welford's algorithm is inherently sequential
(each step depends on the running mean from the previous step) — it's **Kingdom B**.

For the current architecture, the options are:
1. **Two-pass**: Pass 1 computes the mean exactly (sum/count). Pass 2 computes
   Σ(x - mean)² with the mean as a `reference` value. This IS expressible in
   accumulate+gather: a second `AccumulatePass` with `Grouping::All` and the
   reference set to the first-pass mean.
2. **Kahan-compensated one-pass**: Add error compensation terms (Kahan summation).
   Harder to express as accumulate+gather, requires multiple accumulators.
3. **Sort-based**: Sort first, compute from sorted order. Very different architecture.

**Recommended fix**: Two-pass variance. The architecture already supports multiple
passes. The first pass computes mean, the second pass computes sum of squared
deviations. This is correct, parallelizable (each element's contribution to
Σ(xᵢ-μ)² is independent), and expressible in the accumulate+gather framework.

## Files

- **Failing test**: `crates/tambear-primitives/tests/adversarial_baseline.rs`
  - `variance_catastrophic_cancellation_exposed`
  - `variance_welford_vs_onepass_stress`
- **Buggy formula**: `crates/tambear-primitives/src/recipes/mod.rs:183-189`
  (the `variance()` recipe gather expression)
- **Also affected**: `variance_biased()`, `std_dev()`, `covariance()`, `pearson_r()`

## Test that will pass when fixed

Both `variance_catastrophic_cancellation_exposed` and `variance_welford_vs_onepass_stress`
in `adversarial_baseline.rs`.


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

