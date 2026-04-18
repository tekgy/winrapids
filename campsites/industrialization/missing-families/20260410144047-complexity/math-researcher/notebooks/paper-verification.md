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

# Complexity — Paper Verification of Promoted Primitives

Verified against original papers.
Date: 2026-04-10

---

### 1. count_matches — CORRECT

**Reference**: Richman & Moorman (2000), "Physiological time-series analysis using approximate entropy and sample entropy"

**Paper says**: "B^m(r) is the number of template vectors of length m that are within tolerance r of each other (excluding self-matches)."

**Implementation** (complexity.rs:77):
- Uses L∞ (Chebyshev) distance: CORRECT per Richman & Moorman
- Excludes self-matches (j > i, upper triangle): CORRECT for SampEn
- Returns raw pair count, not normalized: CORRECT (SampEn normalizes externally)

**Subtle point**: The original SampEn paper counts ordered pairs i≠j (both (i,j) and (j,i)), so the count is 2× our upper-triangle count. But since SampEn takes the ratio B/A, the factor cancels. CORRECT.

### 2. phi_func — CORRECT

**Reference**: Pincus (1991), "Approximate entropy as a measure of system complexity"

**Paper says**: φ^m(r) = (1/(N-m+1)) Σᵢ ln(C_i^m(r)) where C_i = (count including self)/N'

**Implementation** (complexity.rs:102):
- INCLUDES self-matches (j ranges over all templates including i): CORRECT per Pincus
- Normalizes count by n_templates: CORRECT (C_i = count/N')
- Takes log of normalized count: CORRECT
- Averages log values: CORRECT

**Key difference from count_matches**: phi_func INCLUDES self-matches (ApEn convention), count_matches EXCLUDES them (SampEn convention). This distinction is correct and matches the original papers.

### 3. pattern_to_index — CORRECT

**Reference**: Bandt & Pompe (2002), "Permutation entropy: a natural complexity measure for time series"

**Algorithm**: Lehmer code → factorial number system

**Implementation** (complexity.rs:177):
- Computes rank of each element in the ordinal pattern
- Converts to Lehmer code (factorial base index)
- This gives a unique index ∈ [0, m!) for each of the m! possible ordinal patterns

**Verification**: For m=3, the 6 patterns map to indices 0-5 bijectively.
Pattern (0,1,2) → index 0, (2,1,0) → index 5. CORRECT.

### 4. factorial — CORRECT

Straightforward: n! = 1×2×...×n

**Note**: Returns usize, so will overflow for n > 20 on 64-bit. This is fine for complexity measures where m ≤ 10 typically. Should document the overflow bound.

### 5. linear_fit_segment — CORRECT

**Reference**: Peng et al. (1994) for DFA, general OLS for detrending

**Implementation** (complexity.rs:347):
- Centers x at (n-1)/2
- Computes slope = Σ(x_c × y) / Σ(x_c²)
- Computes intercept from centered coordinates

**Verification**: This is OLS on {0, 1, ..., n-1} vs y. The centering avoids numerical issues with large indices. CORRECT.

**Note**: Could call `ols_slope` instead but the centered formulation is more numerically stable for DFA's use case (many small segments).

### 6. estimate_mean_period — CORRECT

**Reference**: Rosenstein et al. (1993), "A practical method for calculating largest Lyapunov exponents from small data sets"

**Paper says**: "The mean period of the data is estimated as the reciprocal of the mean frequency, obtained by counting zero-crossings."

**Implementation** (complexity.rs:602):
- Counts zero-crossings of mean-centered data
- Returns n / n_crossings as mean period
- Handles edge case: no crossings → returns n

CORRECT per Rosenstein's algorithm.

---

## Assumptions Documented

| Primitive | Key Assumption | Failure Mode |
|---|---|---|
| count_matches | r > 0 | r=0 → only exact matches (data-dependent) |
| count_matches | m ≥ 1 | m=0 → trivially all match |
| phi_func | n > m | n ≤ m → empty template set |
| pattern_to_index | No ties in pattern | Ties → ambiguous ordering (implementation handles via element comparison order) |
| factorial | n ≤ 20 (for usize) | n > 20 → overflow |
| linear_fit_segment | segment length ≥ 2 | length < 2 → no slope (returns 0) |
| estimate_mean_period | Data has oscillatory structure | Monotone data → 0 crossings → returns n |

## Still Missing Complexity Primitives

From my earlier catalog, not yet implemented:
- `false_nearest_neighbors` — CRITICAL for validating all embedding-based methods
- `multiscale_entropy` — most-cited after SampEn
- `fuzzy_entropy` — better than SampEn in nearly all settings
- `katz_fd` / `petrosian_fd` — fast fractal dimension
- `zero_one_test_chaos` — no embedding needed
- `dispersion_entropy` — O(n), parameter-light
- `bubble_entropy` — parameter-free
- `cross_rqa` — bivariate RQA
- `coarse_grain` — needed for MSE
- `forbidden_patterns` — determinism diagnostic


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

