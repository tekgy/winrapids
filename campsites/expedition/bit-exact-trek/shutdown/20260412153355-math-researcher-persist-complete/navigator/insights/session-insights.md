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

# Navigator Session Insights — Bit-Exact Trek 2026-04-12

## The P2 tightening was the session's load-bearing event

The ESC-002 ruling (Option 3: never emit OpFMin/OpFMax) was the catalyst, but the real work was abstracting from that ruling to a general principle. The P2 tightening in guarantee-ledger.md — "the lowering must be a homomorphism to a semantically-pinned subset of the target ISA" — is what converts a one-time ruling into a reusable decision procedure. Every future VB entry can now be evaluated against the checklist rather than re-derived from first principles.

## The Bucket A/B finding emerged from the full VB-001 through VB-005 view

No single role had the view across all five vendor bugs simultaneously. Scout filed them, adversarial filed VB-004, the entries arrived as separate messages. The navigator was the only position that read them as a set and saw the convergence: all five are IEEE-754 corner case gaps, not core arithmetic errors. That structural observation is what makes the P2 audit checklist principled rather than ad hoc.

## The evidence chain gap (vulkaninfo JSON) is important but not urgent

The capability matrix has confirmed device properties without committed primary sources. This is acceptable for an expedition (the device is physical and re-queryable) but not for publication-grade rigor. Next session: commit the raw vulkaninfo JSON to an evidence directory, add citations to the capability matrix.

## The guarantee ledger is Kingdom A only — that's correct for now

Deliberate. Kingdom B invariants can't be written correctly until there's an implementation to constrain them. The Kingdom B extension campsite documents what's known about where the gaps will be, so next navigator doesn't have to re-derive that too.

## Crossovers produced richer reasoning trails than non-crossovers

Every crossover this session added reasoning, not confusion: scout's Option 1 entry made the Option 3 ruling explanation clearer; aristotle's four-option arc closure was better documentation than a simple SHA pointer. The anti-crossover move is "never silently overwrite" — preserve the earlier reasoning and make the supersession explicit.

## The coordination cost of large stale message batches is real but manageable

Eight stale messages took ~20 minutes to triage correctly. The protocol (inventory → git check → file state check → categorize → act) works. The prevention is better session-end-state handoffs; the fallback is the triage protocol.


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

