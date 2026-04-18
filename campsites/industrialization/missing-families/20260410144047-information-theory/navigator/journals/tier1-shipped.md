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

# Information Theory — Tier 1 Primitives Shipped

## Status: DONE (Tier 1)

12 new primitives added to `crates/tambear/src/information_theory.rs` and exported in `lib.rs`.
74 information theory tests green.

## What shipped

### f-divergence family
- `hellinger_distance_sq`, `hellinger_distance` — √(1 - BC(p,q)); range [0,1]
- `total_variation_distance` — ½ Σ|p-q|; range [0,1]
- `chi_squared_divergence` — Σ (p-q)²/q
- `renyi_divergence(p, q, alpha)` — special cases α=0 (support ratio), α=1 (KL), α=∞ (max ratio)
- `bhattacharyya_coefficient`, `bhattacharyya_distance` — -ln(BC); range [0,∞)
- `f_divergence(p, q, f)` — general framework via closure; subsumes all the above

### Joint entropy and PMI
- `joint_entropy` — H(X,Y) from joint probability matrix
- `pointwise_mutual_information(positive: bool)` — PMI or PPMI

### Sample-based divergences
- `wasserstein_1d` — O(n log n) via sorted CDF merge; handles unequal sizes
- `mmd_rbf` — Maximum Mean Discrepancy with Gaussian RBF kernel; median bandwidth heuristic
- `energy_distance` — V-statistic form; clamped to 0 for finite-sample bias

## Tier 2 remaining (not yet implemented)
- KSG mutual information estimator (k-NN based, for continuous data)
- NSB entropy estimator (Bayesian, for sparse discrete data)
- Chao-Shen entropy (small-sample corrected)
- Conditional MI: I(X;Y|Z)
- Directed information: Massey's I(X→Y)
- Blahut-Arimoto: channel capacity via alternating optimization

## Commit
`0e9d42b`


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

