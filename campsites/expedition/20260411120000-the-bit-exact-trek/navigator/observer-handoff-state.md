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

# Observer Handoff State
# Written at session end 2026-04-12 for the next observer to read first.
# This is NOT the lab notebook — it's the short version. Read this, then read lab-notebook.md.

## What I am actively watching (6 standing concerns)

| # | Concern | Trigger | Blocking |
|---|---------|---------|---------|
| SC-1 | Tolerance drift — every `assert!((a-b).abs() < 1e-X)` needs justification | Any new test with relative/absolute tolerance | Ongoing |
| SC-2 | Reduction determinism — `@xfail_nondeterministic` must not be removed before Peak 6 | Any removal of that annotation | Peaks 1-5 |
| SC-3 | One-pass variance trap — 5 recipes still use `(Σx² - (Σx)²/n)/(n-1)` | B1 fix landing | Summit tests |
| SC-4 | I1 creep — `eval_gather_expr` in tambear-tam calls f64::exp/ln; must not be reached by new path | Any new `use tambear_tam` in the trek crates | Peaks 4-5 |
| SC-5 | FMA contraction (I3) — PTX defaults contract-everywhere; SPIR-V needs NoContraction per op | Peak 3 PTX assembler landing; Peak 7 SPIR-V landing | Peaks 3, 7 |
| SC-6 | Oracle validation gap — cross-backend agreement is necessary but not sufficient; need mpmath check too | Peak 7 summit test design | Summit |

## 3 most urgent open items

1. **B1 variance** — `tambear-primitives` adversarial_baseline has 2 FAIL, both variance catastrophic cancellation. 5 recipes affected. Two-pass fix architecturally available via `DataSource::Reference`. Nobody has fixed this yet. Will corrupt real financial time series.

2. **tbs eval NaN/Eq** — 4 FAIL in adversarial_tbs_expr: B2 (Eq epsilon 1e-8), B4/B5 (NaN non-propagation through Min/Max in tbs string evaluator). Fixed at accumulates level (`ad84a51`) but tbs::eval layer unfixed.

3. **TMD corpus** — `exp-tmd.md` referenced in `tam_exp.toml` but not created. The `tmd_candidates` corpus has only 4 values. Needs systematic expansion before tam_exp claims faithful rounding is defensible.

## Last known test counts (2026-04-12, pre-shutdown)

| Suite | Passing | Failing | Ignored |
|-------|---------|---------|---------|
| `tambear-tam-ir` | 47+ | 0 | 0 |
| `tambear-tam-test-harness` (lib) | 65 | 0 | 2 (xfail, correct) |
| `tambear-tam-test-harness` (integration) | 5 | 0 | 1 |
| `tambear-tam-test-harness` (oracle_runner) | 5 | 0 | 1 |
| `tambear-primitives` adversarial_baseline | ? | 2 | ? |
| `tambear-primitives` adversarial_tbs_expr | ? | 4 | ? |

## Critical gotcha the next observer must know

**Cody-Waite corpus loading trap:** The `cody_waite_exact` inputs in `tam_exp.toml` are `(n as f64) * LN2_HI` where `LN2_HI = 0x3FE62E42FEFA3800`. The oracle runner must NOT re-evaluate these via mpmath. mpmath's `ln(2)` differs from `LN2_HI`, and that difference is what Cody-Waite tracks. Re-evaluating via mpmath turns "residual is exactly zero" tests into "general case" tests, which are weaker. The oracle TOML note [N2] documents this. Verify corpus loading code enforces it.

## What the lab notebook covers

Entries 001-007. Start with Entry 003 (Peak 6 RFA correction — big framing change) and Entry 005 (adversarial sweep — P19 structural finding that agreement ≠ correctness). The other entries are architectural observations and integration bug tracking.

Lab notebook path: `campsites/expedition/20260411120000-the-bit-exact-trek/lab-notebook.md`


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

