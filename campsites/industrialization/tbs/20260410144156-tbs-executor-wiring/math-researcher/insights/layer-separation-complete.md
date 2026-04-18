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

# Layer 1 Separation Complete

*2026-04-10 — math-researcher*

## What was done

The CLAUDE.md contract requires clean layer separation between:
- Layer 0 = math primitives
- Layer 1 = diagnostic-driven auto-selection
- Layer 2 = `using()` override transparency

Before this work, the auto-detection logic for `correlation`, `t_test_2`, `anova`, `regression`, `pca`, and `vol_analyze` was embedded inline inside the executor's match arms — 100-200 line blocks mixing diagnostic computation with dispatch logic.

## New files

**`tbs_advice.rs`** — holds the four advice types:
- `TbsDiagnostic` (test_name, result, conclusion)
- `TbsRecommendation` (method, reason)
- `TbsOverride` (method, key, warning)
- `TbsStepAdvice` (recommended, user_override, diagnostics)

Previously these lived in `tbs_executor.rs`, creating circular import issues. Moving them to a dedicated module breaks the cycle: `tbs_autodetect → tbs_advice ← tbs_executor`.

**`tbs_autodetect.rs`** — six Layer 1 functions:
- `normality_test(col, n_thresh)` — shared helper: SW vs D'Agostino-Pearson selection
- `autodetect_correlation(x, y, bag)` — binary/normality/pearson/spearman routing
- `autodetect_t_test_2(x, y, bag)` — normal+equal-var/welch/mann-whitney routing
- `autodetect_anova(group_vecs, bag)` — anova/welch-anova/kruskal routing
- `autodetect_pca_components(data, n, d, bag)` — KMO/Bartlett/Kaiser criterion
- `autodetect_regression_diagnostics(x_pred, x_aug, residuals, ...)` — VIF/normality/Breusch-Pagan/Cook
- `autodetect_volatility(col, bag)` → `VolatilityDecision` — ARCH-LM/GARCH/EWMA routing

## Executor arms after refactor

Each auto-detection match arm is now ~15-20 lines:
1. Extract columns from pipeline
2. Call `autodetect_*(...)` → returns `(auto_method, advice)`
3. Apply Layer 2 override: if `using_bag.method()` is set, wrap with `TbsStepAdvice::overridden`
4. Run the selected primitive
5. Set `step_advice = Some(adv)`

## What this enables

The autodetect functions are now independently callable:
```rust
use tambear::tbs_autodetect::autodetect_correlation;
let (val, method, advice) = autodetect_correlation(&x, &y, &bag);
```

The Layer 1 logic is testable without going through the executor. Future diagnostics modules (audit logs, method-selection reports) can call these functions directly.

## What remained inline (cluster_auto)

`cluster_auto` uses a silhouette sweep that mutates `pipeline` state (calls `pipeline.kmeans()`). Extracting it would require returning a mutated pipeline from the autodetect function, which breaks the pure-function contract. Left inline intentionally.

## `discover_*` status

All four `discover_*` dispatches were already wired before this session:
- `discover_correlation` → `sweep_correlation(x, y)` in superposition.rs
- `discover_regression` → `sweep_regression(x, y)`
- `discover_changepoint` → `sweep_changepoint(col)`
- `discover_stationarity` → `sweep_stationarity(col, alpha)`

All four have executor tests that pass.


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

