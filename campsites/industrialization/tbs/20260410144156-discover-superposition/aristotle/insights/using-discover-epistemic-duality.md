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

# Using/Discover: The Epistemic Duality

*2026-04-10 — Aristotle*

## The First-Principles Observation

`using()` and `discover()` are not just API methods. They are the TWO POSSIBLE EPISTEMIC STATES of a parameter:

- `using(method="pearson")` = "I KNOW the right answer here"
- `discover(method)` = "I DON'T KNOW — find the best one"

This is a **complete partition of epistemic space**. At each decision point, the user either knows or doesn't. There is no third option. The API encodes this partition directly.

## Why This Is Deep

Most APIs force the user into one mode:
- **Opinionated frameworks** (sklearn, tidyverse): choose FOR the user. No `using()` equivalent — the defaults ARE the answers.
- **Bare-metal libraries** (LAPACK, BLAS): force the user to choose EVERYTHING. No `discover()` equivalent — every parameter must be specified.

Tambear is the only framework I've seen that makes the epistemic state EXPLICIT and COMPOSABLE. You can know some things and not others within the same pipeline. The system fills in what you don't know without overriding what you do know.

## Connection to Holographic Error Correction

When `using()` and `discover()` coexist in the same pipeline:
- The `using()` parameters are FIXED (the user's knowledge constrains the computation)
- The `discover()` parameters are EXPLORED (the system runs multiple options)

The `view_agreement` across `discover()` options measures how ROBUST the answer is to the unknown parameters. High agreement = the unknown parameters don't matter (the answer is structurally determined by the known ones). Low agreement = the unknown parameters matter critically (the user NEEDS to know more).

This IS error correction in the epistemic sense: the redundancy from running multiple options detects when the user's partial knowledge is sufficient vs. insufficient.

## What Using() Should Flow Through

The CLAUDE.md contract says `using()` flows DOWN through compositions. When `kendall_tau` calls `inversion_count`, the user's `using(inversion_method="fenwick")` should reach it. This is currently incomplete (task #2 — ~400 pub fns need wiring).

From first principles: `using()` should flow through ANY composition boundary. If method A calls primitive B calls primitive C, the user's override should reach C without A or B knowing about it. This is dynamic scoping — the override is in scope for the duration of the call, not lexically.

The UsingBag is currently a HashMap<String, UsingValue>. This is the right data structure because:
- Open-ended keys (any method can define its own)
- Flat namespace (no nesting required — methods query for their keys and ignore the rest)
- Consumed after use (prevents stale overrides from leaking)

The one thing I'd question: should `using()` be consumed after the NEXT step, or should it persist through the pipeline? Current design: consumed. But if a user says `using(alpha=0.01)` they probably want alpha=0.01 for ALL hypothesis tests in the pipeline, not just the next one.

This is a design decision with no obviously right answer. Consumed is safer (prevents surprises). Persistent is more convenient (set once, use everywhere). The CLAUDE.md doesn't specify. Worth a conversation.


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

