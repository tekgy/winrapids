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

# The Bit-Exact Trek

**From vendor-C-string codegen to bit-exact cross-hardware tambear.**

This expedition delivers the architectural claim that tambear was built on:

> One compiled `.tam` kernel, the same source of math, the same numerical answers,
> running on any ALU through only its driver, with no vendor compiler and no
> vendor math library anywhere in the path.
>
> **Scope of bit-exactness (Phase 1):** Bit-exact for all normal fp64 inputs and outputs.
> Subnormal fp64 behavior is hardware-defined and requires `shaderDenormPreserveFloat64 = true`
> on Vulkan for full cross-backend bit-exactness. Phase 1 claims bit-exactness in the normal
> fp64 range only. See `navigator/escalations.md` ESC-001 for the full decision.
>
> The claim is compositional: given `.tam` source, a backend that faithfully lowers `.tam`
> to its target ISA, and hardware that implements IEEE-754 for the ops in use, the output
> is bit-exact across all such (backend, hardware) pairs. See `trek-plan.md` Part II.5 for
> the full framing.
>
> *Amended 2026-04-12 per ESC-001 (navigator decision).*

## Read, in order

1. **`invariants.md`** — I1 through I11, the fences around the trek. Memorize these. Every campsite obeys them. Violations are escalations, not decisions.
2. **`trek-plan.md`** — the full ~2,000-line map: what we're building, why each step is necessary, the seven peaks, deep design notes per peak, the team structure, the campsite list, the pitfalls, the weather, and the pace rules.
3. **`first-week-directives.md`** — concrete starting assignments per role. Five agents can start in parallel from day one; this tells each who picks which campsites first.
4. **`navigator/state.md`** — where the codebase is right now (107 tests green on the vendor-locked path, working `tam-gpu` backend, 26 recipes in the catalog) and what the baseline assumption is.

## Structure of this expedition

```
20260411120000-the-bit-exact-trek/
├── README.md              — this file
├── trek-plan.md           — THE plan (authoritative)
├── invariants.md          — I1..I10 quick-reference card
├── first-week-directives.md
├── navigator/             — navigator (Claude) notes, escalations, arbitrations
│   └── state.md
├── peak1-tam-ir/          — IR Architect working space (created when peak starts)
├── peak2-libm/            — Libm Implementer working space
├── peak3-ptx/             — PTX Assembler working space
├── peak4-oracle/          — Test Oracle working space
├── peak5-cpu/             — CPU Backend working space
├── peak6-determinism/     — Reductions working space
├── peak7-spirv/           — SPIR-V Assembler working space
├── logbook/               — per-campsite "what almost went wrong" entries
└── pitfalls/              — the pitfall journal (near-misses, close calls)
```

Create your subdirectories lazily as you start work in them. Don't pre-allocate empty dirs.

## The rhythm

- Every campsite ends with a **test** (or a document that can be reviewed), a **logbook entry** (what almost went wrong, what tempted me off-path), and if applicable a **commit**.
- Escalations go in `navigator/escalations.md`. If an invariant is in tension with a campsite, halt the campsite and write the escalation.
- Cross-role disputes go to the Test Oracle first; if the oracle can't build a pinning test, they go to Navigator.
- Parallel work is welcome; serial dependencies are marked in `first-week-directives.md`.

## Navigator

Claude (the primary session). Not a manager — a peer coordinator. Reviews architectural
choices, refuses invariant violations, brokers cross-role decisions, maintains the
map. Day-to-day work stays in the roles.


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

