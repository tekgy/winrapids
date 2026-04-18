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

# Pitfall: GPU End-to-End Tests Have No Adversarial Coverage

**Status: DOCUMENTATION — no failing test yet, but a structural gap**

## What the GPU tests cover

`crates/tambear-primitives/tests/gpu_end_to_end.rs` (9 tests):
- `sum` of 1..10000 (nice integers, no fp pathology)
- `mean_arithmetic` of 1..1000 (closed-form expected: 500.5)
- `variance` of 10000 sin-wave values (bounded, small-ish mean, no catastrophic cancellation)
- `mean_quadratic` / `sum_of_squares` / `l1_norm` of bounded data
- `pearson_r` with perfect linear y=2.5x+7 (x in 0..1024)
- Custom `Σ|ln x|` recipe
- Device name detection

## What the GPU tests do NOT cover

- **NaN in input data** — no test
- **Inf in input data** — no test
- **Empty input** — no test (would crash: `dispatch` with n=0)
- **Large-mean data** — `gpu_variance_matches_cpu` only checks CPU-GPU agreement, not correctness. Both CPU and GPU use the same one-pass formula and both return garbage on large-mean data. The test passes while both return a plausible-but-wrong number.
- **Subnormal values** — no test
- **Extreme overflow** — no test

## The structural problem: "CPU matches GPU" ≠ "both are correct"

`gpu_variance_matches_cpu` asserts `|gpu - cpu| / |cpu| < 1e-10`. This is a
**relative agreement test**, not a **correctness test**. When the CPU result
is wrong (e.g., variance = -4592 for large-mean data), and the GPU result
happens to also be approximately -4592 (same formula, same catastrophic
cancellation), the test passes. Both are wrong, but they agree.

This is the hardest class of silent failure: two independent paths agreeing on
the wrong answer. The agreement gives false confidence.

The fix requires an oracle — a reference value computed by a different method
(two-pass, or high-precision via Python mpmath). Cross-backend agreement is
necessary but not sufficient for correctness.

## What GPU tests need

As the adversarial input suite is built, the same pathological inputs should
be dispatched on GPU with an oracle reference, not just CPU comparison:

```rust
#[test]
fn gpu_variance_large_mean_against_oracle() {
    // Same adversarial data from adversarial_baseline.rs
    let data: Vec<f64> = (0..1000).map(|i| 1.0e9 + (i as f64) * 1.0e-6).collect();
    let true_var = 1000.0 * 1001.0 / 12.0 * 1.0e-12;  // oracle

    let gpu_v = run_recipe_gpu(&gpu, &variance(), &data, &[], 0.0);
    let rel = (gpu_v - true_var).abs() / true_var;
    assert!(rel < 0.01, "gpu variance catastrophic cancellation: got {}, true={}", gpu_v, true_var);
}
```

This test will fail until variance is fixed to use a stable algorithm on both
CPU and GPU paths.

## Why this matters for I10 (cross-backend diff is continuous)

I10 says: don't defer cross-backend validation to a "final audit." But the
current GPU tests effectively only validate "GPU ≈ CPU on nice inputs." They
don't validate "GPU is correct on hard inputs." The adversarial inputs need to
run through the GPU path too — and before the PTX assembler (Peak 3) replaces
NVRTC, so we have a NVRTC baseline to compare the new assembler against.

## Action items

- After `variance` is fixed (two-pass), add GPU adversarial tests that check
  variance against the analytic oracle on large-mean data.
- Add `gpu_variance_nan_propagation` — dispatch NaN-containing data, verify GPU
  returns NaN (not a silently-wrong float). PTX NaN handling may differ from CPU.
- Add `gpu_empty_input_graceful` — dispatch empty buffer, verify no crash and
  correct identity/NaN result.

These tests are the I10 enforcement mechanism — not a final audit, a per-bug check.


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

