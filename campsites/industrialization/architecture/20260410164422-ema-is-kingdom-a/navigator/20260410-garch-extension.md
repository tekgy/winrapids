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

# GARCH Extension to Kingdom A Finding

Written: 2026-04-10, navigator

## GARCH is Kingdom A (not Kingdom B as labeled)

`volatility.rs:9` labels GARCH as Kingdom B. This is wrong.

GARCH(1,1): `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}`

Affine map form: `a = β` (constant), `b_t = ω + α·r²_{t-1}` (data-dependent).
`r_{t-1}` is the observed return series — DATA, not state. Maps compose.
This is Kingdom A.

Correct classification:
- GARCH filter: Kingdom A (affine prefix scan)
- Log-likelihood sum: Kingdom A (prefix sum)  
- Parameter optimization: Kingdom C (iterative MLE)

Source: scout garden note `~/.claude/garden/2026-04-10-garch-is-kingdom-a.md`

## Action required

Fix comment at `volatility.rs:9`. Code is correct; label is wrong.
Routed to pathmaker 2026-04-10.

## The broader theorem emerging

EMA, EWMA, all ARMA, GARCH filter, HMM forward algorithm, Kalman filter —
all Kingdom A via the affine semigroup or matrix product semigroup.

Genuine Kingdom B requires state-dependent maps (map structure depends on
the current hidden state, not the current data). Scout/aristotle/math-researcher
actively investigating where the true Fock boundary lies.

Conjecture: Kingdom B requires nonlinearity in the STATE variables specifically
(not in the data). Affine-in-state = Kingdom A regardless of data complexity.


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

