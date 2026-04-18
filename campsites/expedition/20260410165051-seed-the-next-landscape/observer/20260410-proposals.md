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

# Observer's Next-Landscape Proposals

Written: 2026-04-10

## oracle-coverage-map (verification family)

A systematic map of which primitives have been verified against high-precision
oracles vs. which haven't. Not "how many tests pass" but "what territory is
still unclaimed."

**Why this matters:** The Padé [6/6] failure wasn't found by reading the
implementation. It was found by stating a mathematical truth and seeing the
code violate it. `exp(t·I) = e^t·I` is a theorem — the test asserted the
theorem, the code falsified it. You can't find that class of bug by inspection.
You can only find it by systematic oracle coverage.

**Current state:** Two workup files exist (erfc.md, pearson_r.md) doing this
for two primitives. The coverage map extends that to all 120+ primitives.

**Format (navigator's suggestion):**
Three-tier status per primitive:
- Not verified (no oracle comparison exists)
- Spot-checked (verified on a handful of known values)
- Fully oracled (verified against mpmath at 50+ digits, adversarial edge cases,
  multiple scales)

**Open design question from navigator:** Track status only, or also track
which tests would be needed to reach full coverage? Status-only is auditable
right now. Test-specification is more useful but harder to maintain.

**Suggested owner:** Observer + scientist jointly — observer tracks coverage
state, scientist executes workups for uncovered primitives.

**Connection to KingdomProof idea (navigator garden):** The coverage map is
the same concept applied to mathematical correctness that KingdomProof applies
to kingdom classification. Both are systematic verification layers that
currently exist as informal convention.


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

