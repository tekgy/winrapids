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

# Parity Table — Peak 4 Oracle

**Purpose:** Running record of what each reference implementation says for each recipe/function, and whether tambear agrees.

**Legend:**
- `?` — not yet measured
- `MATCH` — bit-exact agreement (or within documented ULP for transcendentals)
- `DIFF(n ULP)` — differs by n ULP — under investigation
- `BUG(impl)` — we found a bug in the reference; filed upstream

**The oracle:** mpmath at ≥50-digit precision is the ground truth (I9).
R and Python are peers, not authorities. When they disagree with each other
or with tambear, mpmath breaks the tie.

---

## Pure arithmetic (should be bit-exact across all backends)

| Recipe | tambear | R (`Rscript`) | Python (`numpy`) | mpmath (50-dig) | CPU↔GPU | Notes |
|--------|---------|---------------|------------------|-----------------|---------|-------|
| sum | ? | ? | ? | ? | ? | Pending Peak 5 |
| mean | ? | ? | ? | ? | ? | Pending Peak 5 |
| variance | **BUG(tambear)** | CORRECT | CORRECT | CORRECT | — | One-pass formula: 0.0 vs true 0.087 on data∈[1e9,1e9+1) — 100% error. Fix: two-pass .tam IR (pathmaker 1.4). See variance-numerical-analysis.md |
| sum_sq | ? | ? | ? | ? | ? | Pending Peak 5 |
| l1_norm | ? | ? | ? | ? | ? | Pending Peak 5 |
| rms | ? | ? | ? | ? | ? | Pending Peak 5 |
| pearson_r | ? | ? | ? | ? | ? | Pending Peak 5 |

## Transcendentals (ULP-bounded; mpmath is the oracle)

| Function | tambear max ULP | R max ULP | Python max ULP | mpmath ref | Notes |
|----------|-----------------|-----------|----------------|------------|-------|
| tam_sqrt | ? | ? | ? | — (IEEE exact) | Pending Peak 2 |
| tam_exp | ? | ? | ? | ? | Pending Peak 2 |
| tam_ln | ? | ? | ? | ? | Pending Peak 2 |
| tam_sin | ? | ? | ? | ? | Pending Peak 2 |
| tam_cos | ? | ? | ? | ? | Pending Peak 2 |
| tam_pow | ? | ? | ? | ? | Pending Peak 2 |
| tam_tanh | ? | ? | ? | ? | Pending Peak 2 |
| tam_atan | ? | ? | ? | ? | Pending Peak 2 |
| tam_atan2 | ? | ? | ? | ? | Pending Peak 2 |

## Special-value behavior

| Function | Input | Expected | tambear | Notes |
|----------|-------|----------|---------|-------|
| tam_ln | 0.0 | -inf | ? | Pending Peak 2 |
| tam_ln | -1.0 | NaN | ? | Pending Peak 2 |
| tam_ln | 1.0 | 0.0 (exact) | ? | Pending Peak 2 |
| tam_exp | 1000.0 | +inf | ? | Pending Peak 2 |
| tam_exp | -700.0 | subnormal | ? | must not flush to 0 |
| tam_pow | 0.0, 0.0 | 1.0 (convention) | ? | Pending Peak 2 |
| tam_atan2 | 0.0, 0.0 | 0.0 (convention) | ? | Pending Peak 2 |

## TBS expression evaluator NaN correctness (P16/P17/P18)

These are expression-evaluator bugs in the CPU reference implementation (`tbs::eval`),
independent of backends. All three were independently verified by Oracle against IEEE 754.

| Expr | Input | Expected (IEEE 754) | Old tambear | Fixed tambear | Status |
|------|-------|---------------------|-------------|---------------|--------|
| Min(NaN, 5.0) | NaN | NaN | 5.0 (wrong) | NaN | FIXED |
| Min(5.0, NaN) | NaN | NaN | NaN (happened to be OK) | NaN | was OK, now explicit |
| Max(NaN, 5.0) | NaN | NaN | 5.0 (wrong) | NaN | FIXED |
| Max(5.0, NaN) | NaN | NaN | NaN (happened to be OK) | NaN | was OK, now explicit |
| Sign(NaN) | NaN | NaN | 0.0 (wrong) | NaN | FIXED |
| Gt(NaN, 0.0) | NaN | NaN | 0.0 (wrong) | NaN | FIXED |
| Lt(NaN, 0.0) | NaN | NaN | 0.0 (wrong) | NaN | FIXED |
| Eq(NaN, x) | NaN | NaN | 0.0 (wrong, was 1e-15 epsilon) | NaN | FIXED |
| Eq(1.0, 1.0+1ULP) | near-1.0 | 0.0 (unequal) | 1.0 (wrong, 1e-15 epsilon) | 0.0 | FIXED |

**Oracle reasoning for Eq**: `==` (IEEE 754 equality) is correct; `to_bits()` equality is wrong
because it makes `0.0 != -0.0` which violates mathematical equality. NaN must be guarded
separately. Adversarial's `to_bits()` suggestion was partially right (fix the epsilon) but
used the wrong equality primitive.

---

## Update log

| Date | What changed |
|------|-------------|
| 2026-04-11 | TBS NaN correctness table added: P16/P17/P18 independently verified by Oracle against IEEE 754; all three fixed; Gt/Lt NaN propagation also fixed (same root cause as P18 but not in original adversarial report). |
| 2026-04-11 | Variance row updated: BUG(tambear) confirmed — one-pass formula gives 0.0 on near-1e9 data. See variance-numerical-analysis.md for full quantitative evidence and two-pass design. |
| 2026-04-11 | Table created; all entries pending (harness skeleton landed) |


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

