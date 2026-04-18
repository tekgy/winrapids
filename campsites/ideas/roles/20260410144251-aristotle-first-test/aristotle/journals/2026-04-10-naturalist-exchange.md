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

# The Naturalist Exchange

*2026-04-10, late in session*

The naturalist cross-referenced my four missing grouping classes against the codebase and came back with:
- Tree: MOST evidence (hierarchical_clustering, union-find, connected_components)
- Graph: STRONG evidence (pagerank, label_propagation)
- Circular: CLEAR evidence (Ising model, FFT zero-padding)
- Adaptive: WEAKEST evidence (BOCPD max_run only)

And a correction: phyla are `(grouping, op)` pairs, not grouping alone. FFT and MomentStats share `All` grouping but different ops and zero shared intermediates.

I accepted the correction and refined the theorem. The four-menu decomposition has two levels:
- Level 1: `(grouping, op)` = structural identity = phylum
- Level 2: `(addressing, expr)` = instance parameters

The most beautiful insight from the exchange: `accumulate(data, Circular(n), ComplexExpWeight(k)*x[j], Add)` IS the DFT definition. The butterfly is not a different algorithm — it's the Blelloch tree optimization of a circular accumulate, exploiting the periodicity of the complex exponential. If Circular were first-class, FFT would fall out of the existing compiler optimization pass.

This is deeper than my Riesz argument. Riesz says "FFT is a kernel integral." The circular grouping says "FFT is a circular accumulate and the butterfly is the COMPILER telling the HARDWARE how to schedule it efficiently." The mathematics (circular accumulate) and the optimization (butterfly tree) separate cleanly.

The collaboration worked as designed. The naturalist found the evidence. Aristotle provided the framework. The naturalist corrected the framework. Both got sharper.


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

