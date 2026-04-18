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

# Observer Seed Verification Report

Written: 2026-04-10
By: observer

Verified all agent seed proposals against current codebase state.
Reports: confirmed claims, stale claims, wrong oracles, and items already fixed.

---

## Confirmed fixes (already in code, proposals can mark CLOSED)

**1. log_gamma non-integer negatives** (scipy-gap-scan #2, social-finance-math-scan #2, scientist #2)
- Fixed in `special_functions.rs:160-167`
- Integer poles → `f64::INFINITY` (correct), non-integer negatives → reflection formula → finite
- 5 tests pass including oracle and reflection consistency
- **All three proposals targeting this bug are CLOSED**

**2. Kingdom A reclassifications already landed** (social-finance-math-scan #1, reclassification table)
- `kaplan_meier`: tbs_lint.rs:478 — Kingdom A (prior Kingdom B label fixed)
- `exp_smoothing`: tbs_lint.rs:482 — Kingdom A (prior Kingdom B label fixed)
- `panel_fd`: panel.rs:217 — Kingdom A (prior label fixed)
- `garch11_filter`: volatility.rs:55 — Kingdom A (prior label fixed)
- `silhouette_score`: clustering.rs:697 — Kingdom A (all pairs independent, no state)
- `mod_pow`: number_theory.rs:153 — Kingdom A (bits are data, already documented)
- `bellman_ford`: graph.rs:253 — Kingdom A tropical (already labeled, NOT B/C as table claims)
- `dijkstra`: graph.rs:218 — Kingdom A tropical (already labeled)
- `floyd_warshall`: graph.rs:292 — Kingdom A tropical (already labeled)

**3. renyi_entropy NaN fix** (wave 18 survivor)
- Fixed `information_theory.rs:108`: `fold(0.0f64, f64::max)` → `fold(f64::NEG_INFINITY, crate::numerical::nan_max)`
- 74 information_theory tests pass

**4. SVD section header corrected** (math-researcher correction accepted)
- `linear_algebra.rs:607`: was "Golub-Kahan bidiagonalization + QR iteration"
- Now: "one-sided Jacobi rotations"

**5. matrix_log docstring corrected**
- Removed false claim "Uses the Schur decomposition approach"
- Now accurately says "inverse scaling and squaring ... No Schur decomposition is used"

**6. covariance_matrix workup** (scientist item 4 targets this as zero-coverage)
- `crates/tambear/tests/workup_covariance_matrix.rs` — 10 tests, all green
- Covers: symmetry, diagonal=variance, shift invariance, scale covariance, NaN propagation
- **Scientist item 4 is CLOSED**

---

## Stale claims in proposals

**Joint reclassification table (classification-bijection campsite)**
- Lists `bellman_ford` current label as "B/C" — actual current label is Kingdom A tropical
- Lists `viterbi` (hmm_viterbi) current label as "B" — actual: unlabeled (no Kingdom doc)
- Lists `silhouette_score` current label as "B (was wrong)" — actually already A before this session
- The table documents *intended* corrections, not current state — risk of double-correction

**Math-researcher proposals item 8**
- Describes SVD as "Golub-Kahan bidiagonalization + QR iteration"
- Observer corrected this header this session — implementation is one-sided Jacobi
- SVD workup is still needed (accurate claim), algorithm description is now stale

**Scientist proposals item 2**
- "log_gamma(x≤0) returns INFINITY — known bug" — already fixed

**Scientist proposals item 4**
- "Zero direct oracle tests" for covariance_matrix — 10 tests exist in workup_covariance_matrix.rs

---

## Wrong oracle values corrected

**scipy-gap-scan proposals.md:71**
- Claimed: `log_gamma(-0.5) = ln(2√π) ≈ 1.7232658`
- Correct: `ln(2√π) = 0.5·ln(π) + ln(2) ≈ 1.2655`
- Confirmed by scipy (`gammaln(-0.5) = 1.2655...`) and first principles
- Proposals.md corrected in place
- Note: `log_gamma(-1.5) ≈ 0.8600472` was correct

---

## Genuine open gaps confirmed

These proposal items are accurate — nothing in the codebase covers them:

**SVD workup** (scientist #3, math-researcher #8)
- `crates/tambear/src/linear_algebra.rs` has `svd()` — no dedicated workup file
- Foundation for pinv, lstsq, rank, effective_rank
- Highest unverified primitive by fan-out after covariance_matrix (now covered)

**hmm_viterbi Kingdom classification** (joint reclassification table)
- `hmm.rs:253` — `hmm_viterbi` has no Kingdom annotation in its docstring
- Joint proposal argues Kingdom A via tropical max-plus
- Needs explicit classification before Op::TropicalMaxPlus exists

**Op::TropicalMinPlus / Op::TropicalMaxPlus** (scout #1, pure-math-scan #2, accumulate.rs:121 gap note)
- `accumulate.rs:121-124` documents the gap explicitly
- Not implemented — blocks PELT/Viterbi reclassification from documentation to code

**Validity policy declaration** (adversarial #1, social-finance-math-scan #4, biology-math-scan #2)
- No explicit per-function Propagate/Ignore/Error annotation exists
- Three agents converged on same proposal from different angles — high confidence it's real

**mass_action_rhs** (biology-math-scan #4)
- Genuinely absent from tambear
- Enables SIR/SEIR, Michaelis-Menten, pharmacokinetics

**Schur decomposition** (math-researcher #6)
- Genuinely absent — `linear_algebra.rs` docstring now explicitly says so
- Would improve numerical stability of matrix_log for ill-conditioned matrices

---

## Priority read from observer angle

Items where the verification gap is highest (most claimed but least tested):

1. **SVD** — all consumers (pinv, lstsq, rank, effective_rank) treated SVD as trusted black box
2. **hmm_viterbi kingdom classification** — unlabeled, used in HMM decoding
3. **Op::TropicalMinPlus/MaxPlus** — the gap that blocks three Kingdom A reclassifications from being expressible
4. **Validity policy** — most structural fix, prevents future bug classes

Observer will write the SVD workup as the next proactive action pending incoming verification requests.


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

