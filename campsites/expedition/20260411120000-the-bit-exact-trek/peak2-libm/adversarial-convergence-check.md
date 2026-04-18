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

# Adversarial Convergence Check — Seven Libm Review Docs

**Author:** Adversarial Mathematician  
**Date:** 2026-04-12  
**Method:** Structural table across seven adversarial review outputs, looking for rhymes.  
**Source docs:** adversarial-review-exp.md, adversarial-review-log.md, adversarial-review-sin-cos.md, adversarial-review-pow.md, adversarial-review-atan.md, adversarial-review-tan.md, adversarial-review-hyperbolic.md

---

## The table

| Function | Signed-zero bug | Threshold justification wrong | Precision claim unanalyzed | Missing IR op | Missing special-values entry | Phase boundary error |
|---|---|---|---|---|---|---|
| exp | B1 (−inf→−0 in underflow path) | — | — | — | — | — |
| log | B1 (log(−0) sign) | — | B2 (log1p regime boundary) | — | B3 (missing log column) | — |
| sin/cos | B1 (sin(−0)→+0 signed-zero bug) | — | — | — | B2 (missing tan/atan cols) | — |
| pow | B1 (pow(−0, odd+) table entry) | — | B3 (exp_dd lo bound) | B2 (TwoProd I3) | — | B4 (negative-base branch missing) |
| atan | B1 (atan(−0) sign) | B2 (2-ULP claim not analyzed) | — | — | B3 (missing atan column) | — |
| tan | B1 (signed-zero early return) | B2 (pole-exclusion missing) | — | — | B3 (missing tan column) | — |
| hyperbolic | — | B1 (e^{−22} < 2^{−64} false) | B2 (1/e_x vs exp(−x) wrong reasoning) | B3 (fabs.f64 unconfirmed) | — | B4 (tanh medium-regime ~2 ULP) |

---

## Convergences

### C1 — Signed-zero bugs appear in six of seven reviews

Every function with a zero-special-case has a signed-zero bug. The pattern is always the same: the design doc handles `x = 0` with an equality check (`if x == 0: return +0`), which catches both `+0` and `−0` and always returns `+0`. The fix is always the same: `return x` (pass-through, which preserves sign automatically).

**Functions affected:** exp (B1), log (B1), sin/cos (B1), pow (B1), atan (B1), tan (B1). Hyperbolic avoids it only because `fabs(x) < 2^{-28}` fires before the sign matters, and the formula `x + x³ * P(x²)` preserves sign through polynomial evaluation.

**Why this rhyme matters:** The signed-zero bug is not a careless mistake — it is a systematic design-doc pattern. Every math-researcher working from the same mental model of "handle zero early, return the constant" will introduce it. The fix requires knowing that `fcmp_eq(−0.0, 0.0) = true` in IEEE 754. This is not obvious. The bug survived review of six functions independently.

**Design rule derived:** In any libm function: never `return ±0_constant` in a zero-detection branch. Always `return x`. The only exception is when the output sign is genuinely independent of the input sign (e.g. `exp(−0) = 1.0`, `cosh(−0) = 1.0`) — those are correctly bit_exact_checks entries for the specific value, not early-return branches.

---

### C2 — Special-values matrix is always incomplete

Four of seven reviews flag a missing column in `special-values-matrix.md` (log B3, sin/cos B2, atan B3, tan B3). The pattern: math-researcher adds a new function, designs the special-value handling in the design doc, but doesn't add the column to the shared matrix.

**Why this rhymes:** The matrix is a shared artifact that no single function's design doc "owns." There's no forcing function that triggers updating it. Every new function adds a column that needs to be added, but the design doc workflow doesn't include that step.

**Design rule derived:** The campsite template for any new libm function should include an explicit checklist item: "Add column to `special-values-matrix.md` before campsite closes." This is not optional and not covered by the adversarial review — it should be a pre-commit check.

---

### C3 — Precision claims are asserted, not derived

Three reviews find that a precision claim ("1 ULP over the primary domain") is stated without error analysis: atan B2 (2-ULP claim, no analysis), hyperbolic B2 (1/e_x "safer" claim is wrong reasoning), hyperbolic B4 (tanh medium-regime formula structurally cannot hit 1 ULP near the boundary).

The rhyme: every case where a precision claim failed analysis, it failed in the same direction — the claimed bound was optimistic. No review found a case where the true error was smaller than claimed.

**Why this is structural, not coincidental:** When you write a design doc, the natural thing to do is assert the target precision ("we aim for 1 ULP") without working through whether the algorithm achieves it. The adversarial role's job is to force the derivation. But the pattern suggests the math-researcher role should own the derivation, not defer it to adversarial.

**Design rule derived:** Every precision claim in a design doc must have an explicit error budget derivation in the same section. "1 ULP" is not a claim — it is a conclusion. Show the derivation. If the derivation doesn't fit in the design doc, it doesn't belong as a claim.

---

### C4 — Threshold justifications are present but wrong, not absent

Two reviews find wrong threshold justifications: hyperbolic B1 (the `|x| = 22` threshold, where the claim `e^{−22} < 2^{−64}` is false — the correct argument is `e^{−44} < 2^{−53}`), and atan B2 (threshold for the large-argument regime, where the argument involves `atan(x) ≈ π/2 − 1/x` — needs to be shown to give 1 ULP in the transition region).

The rhyme: the threshold IS correct (the algorithm works), but the stated justification is wrong. This is subtler than a wrong threshold — the math-researcher has the right intuition but the wrong derivation. A future implementer reading the justification would trust it and not re-derive it.

**Why this matters more than it looks:** A wrong justification that accompanies a correct algorithm is a latent bug. The next person to touch the threshold (tuning for performance, extending the domain) will use the wrong reasoning and may break the threshold while believing they're making it better.

**Design rule derived:** Threshold justifications must be checkable in situ. The derivation should be short enough to verify by hand in 30 seconds. If it isn't, it's probably wrong. The formula `e^{−2x} < 2^{−53}` → `x > 53 × ln(2)/2 ≈ 18.4` is 30-second checkable. The formula `e^{−22} < 2^{−64}` takes 5 seconds to disprove.

---

### C5 — Missing IR ops are always `fabs.f64` or a related bit-manipulation op

One review explicitly flags a missing IR op (hyperbolic B3: `fabs.f64` unconfirmed). Pow B2 flags `TwoProd` as requiring I3-compliant implementation (two fmul + fsub, not a single fma). Tan implicitly uses `fabs` via `|x|` notation throughout.

The pattern: every time a design doc uses `|x|` without specifying the IR op, it's implicitly assuming `fabs.f64` exists. If it doesn't, the implementer will reach for `sqrt(x*x)` or a sign-bit mask or `fcmp_lt + fneg` — all of which have different semantics for `±0` and NaN.

**Design rule derived:** The notation `|x|` in a design doc must always be annotated with the IR op: `fabs.f64(x)`. Any other form is a latent divergence. Pathmaker should confirm `fabs.f64` is in the IR op set before any function that uses it opens for implementation.

---

## What the convergences say about the adversarial review process

The five convergences cluster into two categories:

**Category A — Design-doc conventions that propagate bugs:**
- Signed-zero early-return pattern (C1)
- Special-values matrix not updated (C2)
- Notation `|x|` without IR op specified (C5 partial)

These are conventions problems. The design doc template could prevent them with checklist items.

**Category B — Claimed results without derivations:**
- Precision claims asserted not derived (C3)
- Threshold justifications present but wrong (C4)

These are verification problems. The adversarial review catches them, but math-researcher should own them. The design doc template should require derivations to be present, not just claims.

**The meta-finding:** The adversarial role is currently catching two kinds of bugs: (A) systematic convention violations that a better template would prevent upstream, and (B) unverified mathematical claims that the math-researcher role should derive and that adversarial verifies. Category A bugs cost review time but are structurally preventable. Category B bugs are the ones that actually require the adversarial role's domain expertise.

**Proposed action:** Add a "pre-adversarial checklist" to the math-researcher campsite template:
1. All `|x|` occurrences annotated with IR op
2. All zero-special-cases use `return x` not `return ±0_constant`
3. All special-values columns added to `special-values-matrix.md`
4. All precision claims accompanied by error budget derivation
5. All threshold justifications checkable in 30 seconds

If math-researcher runs this checklist before submitting for adversarial review, the review time compresses and the adversarial role focuses entirely on the Category B problems — which is where the real mathematical risk lives.

---

## Blocker status as of session end (2026-04-12)

| Function | Open blockers | Status |
|---|---|---|
| exp | B1-B5 (in exp-design.md) | Campsite 2.6 blocked on math-researcher amendments |
| log | B1-B3 | Campsite 2.9 blocked |
| sin/cos | B1-B2 | Campsite 2.12 blocked |
| pow | B1, B4 (B2, B3 resolved in pow-design.md) | Campsite 2.17 blocked |
| atan | B1-B3 | Campsite 2.15 blocked |
| tan | B1-B3 | Campsite 2.20 blocked (new this session) |
| hyperbolic | B1, B2, B3 (fabs.f64), B4 (navigator ruled: two-call formula) | Campsite 2.18 blocked |

**Total outstanding blockers:** ~20 across 7 docs. All campsites held. No implementation may start until the relevant design doc amendments are reviewed and cleared.

**Next adversarial action after session end:** When math-researcher submits amendments, review each in turn. The signed-zero fixes (C1) are mechanical and fast to verify. The precision derivations (C3) require careful reading. The threshold justification fixes (C4) require numerical verification.


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

