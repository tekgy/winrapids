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

# pinv rcond Bug — Absolute vs Relative Threshold

## The Bug

`linear_algebra.rs:753`:
```rust
let rcond = rcond.unwrap_or(1e-12);
```

The default rcond is absolute (1e-12), but it should be relative to the matrix.

## Why This Is Wrong

For a 100×100 matrix with singular values [1e6, 5e5, ..., 1.0, 1e-3, 1e-10]:
- **Current behavior**: treats σ < 1e-12 as zero. All singular values survive. CORRECT for this matrix.
- But for a matrix with singular values [1e-6, 5e-7, ..., 1e-12, 1e-15, 1e-22]:
  - Current: treats σ < 1e-12 as zero. Keeps 1e-12, drops 1e-15. 
  - This means noise at 1e-15 is treated as zero, but signal at 1e-12 (which is 1e-6 of max σ) is kept.
  - **This is the wrong cutoff**: 1e-12 relative to max σ = 1e-6 is a ratio of 1e-6 — that's real signal, not noise. But 1e-15 relative to 1e-6 is a ratio of 1e-9 — also possibly signal.

The real problem: for a matrix with max σ = 1e6, the correct cutoff is:
```
100 * 1e6 * 2.2e-16 ≈ 2.2e-8
```
But the code uses 1e-12. So singular values between 1e-12 and 2.2e-8 are treated as nonzero when they're actually numerical noise. The pseudoinverse will amplify this noise by factors of 1/σ up to 1/1e-12 = 1e12.

## The Fix

```rust
let rcond = rcond.unwrap_or_else(|| {
    let max_dim = a.rows.max(a.cols) as f64;
    let max_sv = svd_res.sigma.first().copied().unwrap_or(0.0);
    max_dim * max_sv * f64::EPSILON
});
```

Note: `svd_res.sigma` must be sorted in descending order for `sigma[0]` to be the largest. Verify this is the case.

## Reference

numpy.linalg.pinv uses:
```python
rcond = len(a) * amax(s) * finfo(a.dtype).eps
```
where `s` is the singular value array, `len(a)` is max(m,n).

LAPACK's dgelss uses:
```
rcond = max(m,n) * ||A||_2 * eps  (when rcond < 0, meaning "use default")
```

## Impact

This affects any downstream method that uses `pinv` on matrices that aren't unit-scaled:
- `lstsq` (least squares via pinv path)
- Any covariance-based method where covariances are in squared units
- Financial data where prices are in thousands (covariance ~ 1e6)

## Severity

MEDIUM. Most uses in tambear work on standardized data (correlation matrices, normalized features) where max σ ≈ n and the absolute threshold is adequate. But the principle is wrong, and any user passing raw (non-standardized) data will get incorrect results for ill-conditioned systems.

## Note

The `rcond: Option<f64>` parameter already exists, so users CAN override. The issue is solely the default. This is a one-line fix with no API change.


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

