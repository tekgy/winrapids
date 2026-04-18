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

# Sign-Off Log — Peak 4 Oracle

**Protocol:** an entry here means the Test Oracle has personally verified:
1. Gold-standard comparison (tambear vs R vs Python vs mpmath)
2. Synthetic ground truth (known parameters recovered from known distribution)
3. Real data agreement (CPU interpreter vs GPU backend)
4. Cross-platform agreement (every available backend)
5. Hard-cases suite (all adversarial inputs pass)

No entry = not yet certified.

---

| Recipe / Function | Date | Evidence | Backend comparison | Notes |
|-------------------|------|----------|-------------------|-------|
| *(none yet)* | — | — | — | Harness skeleton landed 2026-04-11 |

---

## When to add an entry

An entry is added when ALL of the following are true:

- [ ] tambear result matches mpmath reference at ≤ documented ULP bound
- [ ] tambear result matches R result (or discrepancy is a filed R bug)
- [ ] tambear result matches Python/numpy/scipy result (or discrepancy filed)
- [ ] CPU interpreter agrees bit-exactly with GPU backend (for pure arithmetic)
- [ ] Every hard case from `hard_cases.rs` produces expected behavior (no panic, correct special values)
- [ ] Synthetic ground truth test: known-parameter synthetic data → algorithm → recovered parameters within expected tolerance

## Escalation trigger

If tambear's result disagrees with mpmath and tambear appears correct, the
Test Oracle files this as a potential mpmath bug (unlikely but possible) and
escalates to Navigator before signing off.  We do NOT silently accept a ULP
bound wider than documented.


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

