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

# Peak 2 — tambear-libm Phase 1

**Owner:** math-researcher (design docs) + pathmaker (implementation in .tam IR)

Our own transcendentals, from first principles, bit-exact across every backend.

## What lives here

- `accuracy-target.md` — Campsite 2.1. The ULP bound we commit to, and why.
- `gen-reference.py` — Campsite 2.2. mpmath reference generator.
- `<function>-design.md` — per-function algorithm design documents. Pathmaker implements from these.
- `logbook/` — per-function near-miss entries once implementation is underway.

## Invariants in force (see `../invariants.md`)

- **I1** — No vendor math library. Ever. Not glibc, not `__nv_*`, not `f64::sin` in the interpreter.
- **I3** — No FMA contraction. Every `a*b + c` is two ops (`fmul` then `fadd`). Horner's scheme is canonical.
- **I4** — No implicit reordering. Evaluation sequence is part of the contract.
- **I8** — First-principles only. Read papers. Do not look at glibc/musl/fdlibm/sun-libm source. The coefficients are ours.
- **I9** — mpmath at ≥50-digit precision is the oracle. Not another libm.

## Phase 1 function list

| Function | Campsite | Status |
|---|---|---|
| `tam_sqrt` | 2.4 | trivial (hardware fsqrt) |
| `tam_exp`  | 2.5–2.9 | design doc drafted |
| `tam_ln`   | 2.10–2.12 | design doc pending |
| `tam_sin`  | 2.13–2.14 | design doc pending |
| `tam_cos`  | 2.13, 2.15 | design doc pending |
| (big-arg trig) | 2.16 | deferred to Phase 2 (Payne-Hanek) |
| `tam_pow`  | 2.17 | `exp(b*log(a))` + specials |
| `tam_tanh`, `tam_sinh`, `tam_cosh` | 2.18 | from exp |
| `tam_atan`, `tam_asin`, `tam_acos` | 2.19 | atan is the base |
| `tam_atan2` | 2.20 | quadrant handling |

## The design → implementation pipeline

1. Math researcher writes `<function>-design.md`. Navigator + math researcher agree before any code.
2. Pathmaker implements as a `.tam` function once Peak 1 parser (1.7) is ready.
3. CPU interpreter (Peak 5) executes the `.tam` function op-by-op.
4. ULP harness (2.3) runs 1M samples against mpmath reference (2.2).
5. If `max_ulp > target`, diagnose. Do not relax the bound.


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

