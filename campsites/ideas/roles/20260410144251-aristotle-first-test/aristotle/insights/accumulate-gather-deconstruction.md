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

# Accumulate+Gather: 8-Phase First Principles Deconstruction

*2026-04-10 — Aristotle debut*

## Verdict

Accumulate+gather is the correct decomposition FOR SIMT hardware with flat memory and discrete arrays. The number "two" comes from hardware, not mathematics. Mathematics needs one (function application) or infinitely many. Hardware needs two because there are exactly two sources of parallelism: algebraic structure (associativity for tree combining) and flat memory (uniform access for random reads).

## 14 Assumptions Identified

1. **Data is arrays** — inherited from FORTRAN/C/CUDA tradition, not mathematical necessity
2. **Two verbs suffice** — map/transform, sort, and decision are hidden 3rd/4th/5th operations (absorbed, rewritten, externalized respectively)
3. **Associativity = parallelism** — actually needs full monoid (identity element). Also: f64 "associativity" is approximate
4. **Four menus are orthogonal** — they're a constrained product space (Tiled requires DotProduct/Distance; Prefix requires monoidal op)
5. **Gather is pure reading** — gather with interpolation/defaults/data-dependent addressing is computation
6. **Gather then accumulate** — real algorithms interleave them (SpMV, tree reduction, attention)
7. **8 grouping patterns exhaust space** — tree, graph, adaptive, circular groupings missing
8. **Ops are stateless** — true mathematically, false on hardware (shared mem, register pressure)
9. **f64 is universal** — engineering choice, architecture generalizes
10. **SIMT execution** — architecture is more general than current target
11. **Discovered, not designed** — both: Riesz/monoid/liftability were discovered, but privileging these over alternatives was designed
12. **(mine) Parallelism decomposition is the right frame** — information/transformation/decision is an alternative 3-op decomposition
13. **(mine) Grouping patterns are the frontier** — tree, probabilistic, recursive, dynamic groupings may need first-class support
14. **(mine) Algebra/topology split is clean** — it leaks: Sarkka encodes geometry in op; lag-gather encodes algebra in addressing

## 5 Irreducible Truths

1. Computation requires reading and combining (logically distinct)
2. Parallel combining requires algebraic structure (theorem, not choice)
3. Partition determines communication pattern (fundamental to any parallel model)
4. Data layout and computation are coupled (physical constraint)
5. Some computations are irreducibly sequential (Fock boundary is real)

## The Aristotelian Move

**Accumulate+gather decomposes PARALLELISM, not computation.** This reframes everything:
- Accumulate captures the algebraic dimension (associativity enables tree combining)
- Gather captures the topological dimension (flat memory enables random reads)
- Two operations because two sources of parallelism, not because math has two verbs

**Corollary**: The Riesz representation argument proves universality (every linear transform IS an accumulate) but not efficiency. Efficiency lives in the GROUPING PATTERNS, which encode the structure that makes computation faster than naive accumulate. The grouping patterns are where the real mathematical content lives. Accumulate is the canvas; grouping is the painting.

## What the Void Looks Like (Phase 8)

If accumulate+gather is WRONG, what replaces it? A computation framework parameterized by dependency structure, work function, communication pattern, and scheduling constraint. In that framework:
- Systolic arrays: shift+accumulate (not gather+accumulate)
- Diffusion/PDE solvers: iterative local exchange (not gather then accumulate)
- Auction/assignment: distributed game-theoretic process

All expressible AS compositions of accumulate+gather, but structurally different. The framework is CONTINGENT universal — correct given SIMT + arrays + static partitions + monoids.

## Full analysis

Garden entry: `~/.claude/garden/2026-04-10-aristotle-first-deconstruction.md`


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

