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

# Scout's Next-Landscape Proposals

Written: 2026-04-10

## Priority order (scout's read): (3) → (5) → (4) → (2) → (1)
affine-prefix-scan is the foundation everything else builds on.

---

**1. `industrialization/architecture/tropical-semiring`**
Op::TropicalMinPlus as a new Op variant. PELT, Viterbi, all-pairs-shortest-paths
are Kingdom A in tropical semiring but labeled B — we lack the Op.
Once it exists, these admit GPU-parallel schedules via tropical matmul.
GPU shortest-path via tropical matmul already exists in literature.
Role: math-researcher (verify) + pathmaker (implement)

**2. `industrialization/architecture/kingdom-classification-audit`**
Systematic scan of all Kingdom B labels in codebase. GARCH filter, exponential
smoothing, PELT confirmed mislabeled. Audit produces corrections list +
reclassification spec. Mislabeled ops miss GPU parallelization.
Role: scout

**3. `industrialization/architecture/affine-prefix-scan`** ← FOUNDATION
`affine_prefix_scan(a: &[f64], b: &[f64], s0: f64) -> Vec<f64>` in signal_processing.rs.
EMA, GARCH filter, EWMA variance, all first-order IIR filters delegate to this.
GPU path = scan kernel replacing sequential loop.
Spec complete in EMA campsite (`campsites/industrialization/architecture/20260410164422-ema-is-kingdom-a/scout/insights/ema-kingdom-a-spec.md`).
Role: pathmaker

**4. `industrialization/sharing/gram-matrix-tier1`**
GramMatrix as Tier 1 intermediate in TamSession with derivation rules to
DistanceMatrix and CovarianceMatrix. Unlocks materializing-view pattern.
Spec: `~/.claude/garden/2026-04-10-intermediate-dependency-dag.md`
Requires: IntermediateTag::GramMatrix variant + derive_from() method.
Role: pathmaker

**5. `industrialization/architecture/garch-filter-extraction`** ← HIGH LEVERAGE
Extract `garch11_filter(returns, omega, alpha, beta) -> Vec<f64>` from garch11_fit
as standalone public primitive with correct Kingdom A classification.
MLE outer loop (Kingdom C) calls the filter (Kingdom A).
GPU benefit: 200 optimization iterations each call scan kernel vs sequential loop.
For tick-level data (millions of obs), this is the dominant cost.
Role: pathmaker (straightforward extraction, math done)


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

