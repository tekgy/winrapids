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

# Scientist Proposals — Next Landscape Wave

## Experiments (runnable now)

### 1. Holographic error-correction experiment [CLEANEST, RUN FIRST]
**Claim**: discover_correlation's view_agreement drops when data is corrupted in a way that
affects covariance-sensitive methods (Pearson) but not rank-based methods (Kendall/Spearman).

**Setup**:
- Generate (x, y) with r≈0.7, n=100, deterministic seed
- Baseline: run discover_correlation → record view_agreement and per-method correlations
- Corrupt: multiply 3 extreme x values by 100 (influential outliers)
- Rerun: discover_correlation on corrupted data → measure view_agreement drop
- Expected: Pearson shifts dramatically, Kendall/Spearman stable, view_agreement drops

**Why it matters**: confirms discover() provides structural robustness via method ensemble
diversity, not just redundancy. The disagreement IS the error signal.
**Note**: the protection comes from diverse intermediate dependencies, NOT TamSession
shared intermediates (shared cache = all methods fail together = high view_agreement on
wrong answer). Data-level corruption is the right experimental lever.

---

### 2. Special-function poles diagnostic [FASTEST, 10 MINUTES]
**Claim**: gamma(-1), digamma(0), log_gamma(x≤0) return wrong values.

**Known bug**: log_gamma(x≤0) returns INFINITY — mathematically wrong for negative integers
where the sign alternates. gamma(-1) should return ±∞ (with sign); digamma(0) = -∞.

**Setup**: call each at classical poles, record what tambear returns, compare to mpmath.
Write failing tests first, then fix.
**Files**: `crates/tambear/src/special_functions.rs`

---

## Workups needed

### 3. workup/svd [HIGHEST LEVERAGE]
SVD is the untested foundation for pinv, lstsq, rank, effective_rank. I verified pinv
against oracles but treated SVD as a trusted black box. The erfc pattern applies: bugs in
intermediates are invisible when you only test endpoints.

**Oracle cases needed**:
- Known matrix with analytical SVD (e.g. diagonal, rank-1, Hilbert matrix)
- Nearly rank-deficient: one SV near machine epsilon × largest SV
- Very ill-conditioned: condition number ~1e15
- Matrices where bidiagonalization produces near-cancellation

**Reference**: LAPACK dgesvd (via scipy.linalg.svd), mpmath for small matrices at 50dp.

### 4. workup/covariance-matrix-direct-oracles [HIGH FAN-OUT]
Zero direct oracle tests. 8 consumers (Hotelling, MANOVA, PCA, LDA, Mahalanobis,
factor_analysis, etc.) all dark at the primitive level.

Four theorem tests that need no external tools:
- Symmetry: C = Cᵀ
- Diagonal consistency: C[i,i] = sample variance of column i
- Shift invariance: cov(x + c, y + d) = cov(x, y)
- Scale covariance: cov(ax, by) = ab·cov(x, y)

Oracle: mpmath at 50dp for small (3×3, 5×5) known matrices.

---

## Architecture proposals

### 5. workup/chain-workup-doctrine [NAMING THE PATTERN]
The erfc bug was invisible to the erfc oracle suite but visible to normal_cdf's workup
because 1.96 is a statistically meaningful input. This is a general pattern:

**Doctrine**: workup inputs should include inputs that are load-bearing to consumers, not
just mathematically interesting inputs. The consumer's domain tells you which inputs matter.

**Concrete artifact**: each workup doc should have a "Consumer-derived test inputs" section
documenting which inputs were added because of downstream consumers and why.

**Chain to document now**: erfc → normal_cdf → normal_quantile → hypothesis tests.
The chain flows upstream from meaning: "researchers care about p=0.025" → "normal_cdf
must be accurate at -1.96" → "erfc must be accurate at 1.386."

### 6. workup/gamma-beta-chain [UNTESTED INTERIOR LINKS]
chi2_cdf and t_cdf verified accurate. Both delegate to gamma_cdf / incomplete_beta.
The delegates haven't been directly oracle-tested. Risk: a bug in gamma_cdf could be
masked by lucky test inputs at the chi2/t level. Same erfc pattern.

---

## Priority ranking (scientist's view)

| Priority | Item | Why |
|----------|------|-----|
| 1 | Holographic experiment | Testable now, publishable claim, fast |
| 2 | Special-function poles | Known bug (log_gamma), 10-minute diagnostic |
| 3 | SVD workup | Foundation for pinv/lstsq/rank; highest unverified risk |
| 4 | Covariance matrix oracles | 8-consumer fan-out, zero direct coverage |
| 5 | Chain workup doctrine | Architecture, not code; names the pattern |
| 6 | Gamma/beta chain | Important but lower urgency than SVD |


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

