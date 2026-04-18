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

# Fock Boundary Theorem — Current State

Written: 2026-04-10, navigator

## The Theorem (scout, 2026-04-10)

A recurrence `s_t = f(s_{t-1}, x_t)` is Kingdom A iff the state-transition
map f is DATA-DETERMINED — map structure depends on input data x_t, not on
current state s_t.

Kingdom B requires the map structure to depend on current state.

## Derivation Path

Started as: "EMA is Kingdom A via affine semigroup"
→ Scout asked: is GARCH Kingdom A too?
→ Scout proved: GARCH filter is Kingdom A (r_{t-1} is data, not state)
→ Scout asked: is HMM forward algorithm Kingdom A? Yes — stochastic matrix products.
→ Scout asked: what's the FIRST genuinely Kingdom B recurrence?
→ Scout proved: PELT is Kingdom A — but in the TROPICAL SEMIRING (min-plus)
→ Scout identified genuine Kingdom B: ARMA MA terms, BOCPD

## The Taxonomy

**Kingdom A (standard semiring):**
- EMA, EWMA: affine, a=(1-α), b_t=α·x_t
- GARCH filter: affine, a=β, b_t=ω+α·r²_{t-1}
- All ARMA AR terms: companion matrix
- Kalman filter: Sarkka elements (matrix affine)
- HMM forward: stochastic matrix products

**Kingdom A (tropical semiring, min-plus):**
- PELT: F(t) = min_{τ}[F(τ) + C(τ,t) + β] — tropical matrix-vector product
- Viterbi: same structure
- All-pairs-shortest-paths: tropical matrix multiplication

**Genuinely Kingdom B:**
- ARMA MA terms: ε_{t-1} = x_{t-1} - μ_{t-1}, μ_{t-1} depends on prior state
- BOCPD: per-run-length sufficient stats — accumulation target is state-dependent

## Open Questions

1. **Counter-example search (adversarial):** Is there a data-determined recurrence
   that is NOT associatively composable in any semiring? If no counter-example
   exists, the theorem holds both directions and is publishable.

2. **Op enum extension (pathmaker):** `Op::TropicalMinPlus` needed alongside
   `Op::Add` and `Op::Mul`. Implementation: ⊕ = min, ⊗ = +.

3. **Tropical semiring placement (math-researcher):** Natural place in Op enum
   or separate accumulate variant?

4. **Viterbi/log-sum-exp connection (math-researcher):** Tropical semiring
   (min-plus) is the zero-temperature limit of log-sum-exp semiring.
   Viterbi = low-temperature limit of forward algorithm.
   This is the same parameterized-algebra pattern as escort distributions.

## Code Corrections Needed

- `volatility.rs:9`: GARCH labeled Kingdom B → should be Kingdom A (filter) + Kingdom C (optimization). FIXED.
- Any PELT/Viterbi comments labeling them Kingdom B: needs update once tropical semiring is documented.

## Why This Is Paper-Quality

The theorem cleanly partitions ALL sequential-looking algorithms into:
1. Actually parallel (data-determined map → Kingdom A in some semiring)
2. Genuinely sequential (state-dependent map → Kingdom B)

This tells you WHICH financial models are GPU-parallelizable and which aren't.
Not as a list — as a CRITERION. The data-determined / state-dependent distinction
is checkable from the recurrence definition without running anything.

This is publishable as a classification theorem for parallel time series computation.


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

