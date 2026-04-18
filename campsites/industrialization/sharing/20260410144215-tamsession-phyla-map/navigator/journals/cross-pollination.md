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

# Cross-Pollination Log — Navigator Session

## Scout → Pathmaker (HIGH PRIORITY)

**EMA as Kingdom A via affine parallel scan** (from ta-decomposition.md)

The scout found this and buried it in the TA analysis:

> EMA is a first-order linear recurrence: s_t = α*x_t + (1-α)*s_{t-1}
> This IS a Kingdom A primitive when expressed as a parallel prefix over affine maps:
> (a, b) * (c, d) = (a*c, a*d + b)

This is the same semigroup trick that makes prefix sums Kingdom A. The Fock boundary
dissolves. ALL linear IIR filters have this structure: MACD = three EMA computations,
all liftable to Kingdom A. This changes the architecture for anything using running filters.

**What to do**: Tell the pathmaker. When they implement `ema_period`, implement it as
a parallel prefix scan, not a sequential loop. Document it as Kingdom A with the affine
map semigroup. This is publishable.

## Scout → Pathmaker (HIGH PRIORITY)

**Phantom fix — 6 complete modules were invisible** (from phantom-scan-complete.md)

The scout found: survival, panel, bayesian, irt, series_accel, train::naive_bayes had
ZERO pub use entries despite being fully implemented. Also complexity was missing ccm,
mfdfa, rqa, phase_transition, harmonic_r_stat, hankel_r_stat.

**Status**: FIXED in commit `f431f2e`. All now exported.

Side effect: fintek family22/24 reimplemented ccm, mfdfa, phase_transition from scratch
because they couldn't reach the tambear versions. Now that these are on the pub surface,
the bridges should be updated to delegate.

## Scout → Pathmaker (MEDIUM PRIORITY)

**Missing rolling_aggregate primitive** (from ta-decomposition.md)

`rolling_max`, `rolling_min`, `rolling_std` are all absent from signal_processing.rs.
The unifying primitive is `rolling_aggregate(data, window, op: fn(&[f64]) -> f64) -> Vec<f64>`.

This is a Kingdom B primitive (sequential window scan), but simpler than EMA since the
window just slides (no state carries forward beyond the window). Implement it as the base,
then ema_period on top, then the TA signals on top.

## Scout → Pathmaker (MEDIUM PRIORITY)

**`median_from_sorted` — 6 inline duplicates** (from compound-analysis.md)

Three lines of code, written 6 times across the codebase. The scout ranked this Priority 1.

Also: symbolization primitives (`symbolize_median`, `symbolize_quantile`, `symbolize_ordinal`)
appear in 4 different places and would unblock TamSession sharing for LZ complexity, permutation
entropy, transfer entropy, and quantile symbolize.

## Naturalist → Pathmaker (STRUCTURAL)

**TamSession phyla from spring simulation** (from atomic-industrialization/expedition-log.md)

The naturalist ran the spring topology simulation and derived 8 natural phyla:
1. MomentStats — spans descriptive + hypothesis + volatility
2. SortedOrder — spans descriptive + nonparametric
3. FFT — mostly contained in spectral
4. CovMatrix — spans multivariate + clustering
5. ACF — spans time_series + volatility
6. DistMatrix — spans clustering + complexity
7. OLS — spans linear_algebra + time_series + hypothesis
8. Eigendecomp — spans linear_algebra + multivariate + clustering

Key finding: "descriptive" and "time_series" are FALSE CLUSTERS — each splits into
2-3 natural sub-families by sharing structure. The phyla are the natural scheduler units,
not the family labels.

The scout confirmed: only 6 intermediates are currently wired (DistanceMatrix, MomentStats,
SufficientStatistics, DataQualitySummary, ClusterLabels, HLL Sketch). The 8 phyla above
are the roadmap for what to wire next.

## Navigator → Team

**Status snapshot**:
- adversarial gauntlet: 67/67 green (8 bugs fixed)
- information theory Tier 1: 74/74 green (12 new primitives)
- phantom surface: 6 complete modules now visible
- complexity: ccm/mfdfa/rqa/phase_transition now exported
- next terrain for pathmaker: rolling_aggregate + ema_period, fintek bridges delegating to tambear primitives, TamSession phyla wiring (SortedArray + FFTOutput)


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

