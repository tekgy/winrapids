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

# Variance Numerical Analysis — Gold Standard Scientist Report

**Date:** 2026-04-11  
**Author:** Test Oracle (scientist role)  
**Status:** Evidence for pathmaker's campsite 1.4 redesign

---

## The finding: the current variance_pass.tam is still broken

The current `peak1-tam-ir/programs/variance_pass.tam` accumulates `sum(x)`, `sum(x²)`, and `count` in one pass, with comments claiming the cancellation is avoided because "the subtraction happens at gather time."

**This is wrong.** The catastrophic cancellation is in the gather formula:

```
var_s = (out[1] - out[0]^2 / out[2]) / (out[2] - 1)
      = (sum_sq  -  sum^2  / n       ) / (n - 1)
```

When data values are near a large mean M (say M=1e9), the two terms being subtracted are:
- `sum_sq ≈ n * M²`  
- `sum²/n ≈ n * M²`

Both are ~n * 10¹⁸. Their difference is ~n * σ², where σ is the actual standard deviation (small). The subtraction of two nearly equal numbers of magnitude ~10¹⁸ loses all bits below the scale of the difference.

**Measured result** (seed=42, n=100, data ∈ [1e9, 1e9+1)):
- True variance (two-pass): `8.70e-02`
- One-pass formula: `0.0` (100% relative error — completely wrong)
- Welford streaming: `8.70e-02` (relative error ~2e-8 — correct)

The comment "the kernel is numerically clean" is accurate — the kernel IS clean. The gather formula is the bug.

---

## What "two-pass" actually means

The navigator check-in routes this to pathmaker: "variance_pass in .tam IR as two-pass from the start." Here's what that means in concrete `.tam` terms.

### Pass 1: compute mean

```
kernel variance_pass1(buf<f64> %data, buf<f64> %out) {
entry:
  %n    = bufsize %data
  %acc0 = const.f64 0.0
  %acc1 = const.f64 0.0
  loop_grid_stride %i in [0, %n) {
    %v     = load.f64 %data, %i
    %one   = const.f64 1.0
    %acc0' = fadd.f64 %acc0, %v
    %acc1' = fadd.f64 %acc1, %one
  }
  %s0 = const.i32 0
  %s1 = const.i32 1
  reduce_block_add.f64 %out, %s0, %acc0'  ; out[0] = sum(x)
  reduce_block_add.f64 %out, %s1, %acc1'  ; out[1] = count
}
; host computes: mean = out[0] / out[1]
```

### Pass 2: accumulate Σ(x - mean)²

```
kernel variance_pass2(buf<f64> %data, f64 %mean, buf<f64> %out) {
entry:
  %n    = bufsize %data
  %acc0 = const.f64 0.0
  %acc1 = const.f64 0.0
  loop_grid_stride %i in [0, %n) {
    %v    = load.f64 %data, %i
    %d    = fsub.f64 %v, %mean     ; x - mean
    %d2   = fmul.f64 %d, %d       ; (x - mean)^2
    %one  = const.f64 1.0
    %acc0' = fadd.f64 %acc0, %d2
    %acc1' = fadd.f64 %acc1, %one
  }
  %s0 = const.i32 0
  %s1 = const.i32 1
  reduce_block_add.f64 %out, %s0, %acc0'  ; out[0] = sum((x-mean)^2)
  reduce_block_add.f64 %out, %s1, %acc1'  ; out[1] = count
}
; host computes: var = out[0] / (out[1] - 1)   [sample]
;                      out[0] / out[1]          [population]
```

The gather formula is now just a single division — no subtraction of nearly-equal large numbers. This is the numerically stable design.

**Note for pathmaker:** pass 2 needs `f64 %mean` as a scalar parameter — a new parameter type. The current spec has `buf<f64>` and `i32` parameters. You'll need to decide: does `f64` become a valid scalar parameter type, or does mean get passed through a single-element buffer? The test oracle doesn't have authority here — it's your call as IR Architect. Just make it consistent.

---

## Why Welford isn't the answer here

Welford's online algorithm would also solve the problem, but it's **Kingdom B** (sequential recurrence). The update is:

```
m' = m + (x - m) / k    ; new mean
s' = s + (x - m) * (x - m')  ; note: uses BOTH m and m'
```

The second line uses the old `m` (before update) in one factor and the new `m'` in the other. This is a serial dependency: iteration k must complete before iteration k+1 begins. On GPU with thousands of threads, this is disqualifying — you'd have to run it single-threaded.

Two-pass is the right answer for a parallel accumulation framework. Welford is the right answer for a streaming single-pass constraint. We're building a parallel system, so two-pass.

**The tradeoff:** two-pass reads the data buffer twice. For memory-bound kernels, this doubles I/O. For the scale of data we process, the numerical correctness is worth it. This matches the tambear philosophy: run everything, V columns carry confidence — a numerically wrong answer with high "confidence" is worse than two reads.

---

## The parity table consequence

Until the two-pass variance lands and pathmaker's campsite 1.4 is accepted, the `variance` entry in the parity table has this note:

> **WARNING:** Current variance recipe produces 0.0 on data near large means (e.g. data ∈ [1e9, 1e9+1)). 100% relative error. R and numpy both compute the correct answer on this input. tambear currently disagrees with BOTH R and Python on the trap case. This is a known architectural bug routed to pathmaker; fix is two-pass variance kernel (campsite 1.4).

This is not a ULP disagreement. It's a correctness failure. The parity table row will be marked `BUG(tambear)` until fixed.

---

## Synthetic ground truth test design (for when two-pass lands)

When campsite 1.4 ships a corrected variance kernel, I'll add this test:

```rust
// Known parameters: Normal(μ=1e9, σ=0.1)
// Generate n=10000 samples
// Run two-pass variance kernel
// Assert |σ̂² - 0.01| < 1e-6   (i.e., recovered within ~1e-4 relative error)
// Assert σ̂ ≈ 0.1 to 3 significant figures
```

The current one-pass formula gives σ̂² = 0, which fails this test immediately. The two-pass formula should give σ̂² ≈ 0.01 to within floating-point limits. This is the acceptance criterion I'll use for sign-off.


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

