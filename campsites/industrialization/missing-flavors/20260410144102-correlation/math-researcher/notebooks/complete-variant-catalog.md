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

# Correlation — Complete Variant Catalog

## What Exists

### In tambear::nonparametric
- `pearson_r(x, y)` — Pearson product-moment
- `spearman(x, y)` — Spearman rank
- `kendall_tau(x, y)` — Kendall tau-b (O(n log n) via merge-sort inversions)
- `partial_correlation(x, y, covariates)` / `partial_correlation_full`
- `phi_coefficient(x, y)` — for binary data
- `point_biserial(binary, continuous)` — binary × continuous
- `biserial_correlation(binary, continuous)` — assumes continuous latent
- `rank_biserial(x, y)` — effect size for Mann-Whitney
- `tetrachoric(table)` — 2×2 table → latent bivariate normal r
- `cramers_v(table, n_rows)` — chi-squared-based association
- `eta_squared(values, groups)` — ANOVA effect size
- `distance_correlation(x, y)` — Székely 2007
- `concordance_correlation(x, y)` — Lin 1989
- `hoeffdings_d(x, y)` — Hoeffding's D
- `blomqvist_beta(x, y)` — medial correlation
- `level_spacing_r_stat(sorted_values)` — Wigner-Dyson

### In tambear::nonparametric (distance)
- `dtw(x, y)` — dynamic time warping
- `dtw_banded(x, y, window)` — Sakoe-Chiba band

### In tambear::multivariate
- `covariance_matrix(x, ddof)` — sample covariance
- `cca(x, y)` — canonical correlation analysis

---

## What's MISSING — Complete Catalog

### A. Missing Correlation Coefficients

1. **Kendall tau-c** (Stuart's) — for rectangular contingency tables
   - τ_c = 2(n_c - n_d) × m / (n² × (m-1))
   - Parameters: `x`, `y` (ordinal)
   - Difference from tau-b: adjusts for table shape rather than ties

2. **Somers' d** — asymmetric version of Kendall tau
   - d(Y|X) = (concordant - discordant) / (concordant + discordant + ties on X)
   - Parameters: `x` (predictor), `y` (response)
   - Shares: concordant/discordant counts from Kendall

3. **Goodman-Kruskal gamma** — ignores ties entirely
   - γ = (n_c - n_d) / (n_c + n_d)
   - Parameters: `x`, `y`
   - Shares: concordant/discordant counts

4. **Polychoric correlation** — two ordinal variables
   - MLE of bivariate normal correlation from ordinal data
   - Parameters: `contingency_table`, `max_iter`, `tol`
   - Assumption: continuous latent bivariate normal, thresholds at category boundaries
   - Tetrachoric is the 2×2 special case (already have it)

5. **Polyserial correlation** — one continuous, one ordinal
   - MLE of bivariate normal correlation
   - Parameters: `continuous`, `ordinal_categories`
   - Assumption: continuous latent behind ordinal

6. **Intraclass correlation** (ICC) — agreement between raters/measures
   - ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k)
   - 6 forms from Shrout & Fleiss 1979
   - Parameters: `data: &Mat` (subjects × raters)
   - Primitives: one-way/two-way ANOVA components
   - Critical for: reliability studies, measurement agreement

7. **Cohen's kappa** — inter-rater agreement for categorical
   - κ = (p_o - p_e) / (1 - p_e)
   - Parameters: `rater1`, `rater2` (categorical labels)
   - Variants: weighted kappa (linear, quadratic weights)
   - Fleiss' kappa: multiple raters

8. **Weighted kappa** — accounts for degree of disagreement
   - Parameters: `rater1`, `rater2`, `weights` (linear or quadratic)

9. **Krippendorff's alpha** — generalized agreement for multiple raters
   - Handles: missing data, any number of raters, any scale type
   - Parameters: `data: &Mat` (raters × items, NaN for missing)

10. **Gwet's AC1/AC2** — paradox-resistant agreement coefficient
    - More stable than Cohen's kappa for high-agreement or skewed data
    - Parameters: `rater1`, `rater2`

11. **Mutual information correlation** — r_MI = √(1 - exp(-2MI))
    - Transforms MI into [0,1] scale comparable to Pearson r
    - Already have MI; this is a thin wrapper

12. **Maximal information coefficient** (MIC) — Reshef et al. 2011
    - Explores all 2D grid resolutions, maximizes normalized MI
    - Parameters: `x`, `y`, `alpha` (default 0.6), `c` (default 15)
    - Expensive: O(n^(1+alpha)) but captures any functional relationship

13. **Chatterjee's xi** (rank correlation) — Chatterjee 2021
    - ξₙ = 1 - n Σ|r_{i+1} - rᵢ| / (2Σrᵢ(1-rᵢ))
    - Parameters: `x`, `y`
    - Advantages: consistent test for independence, detects non-monotone

14. **Brownian correlation** — Székely 2009
    - Related to distance correlation but uses Brownian covariance
    - Parameters: `x`, `y`

15. **Hilbert-Schmidt Independence Criterion** (HSIC)
    - HSIC(X,Y) = trace(KHLH) / n² where K,L are kernel matrices
    - Parameters: `x`, `y`, `kernel` (default: Gaussian)
    - Shares: kernel matrix computation with MMD

16. **Randomized dependence coefficient** (RDC) — Lopez-Paz 2013
    - Random nonlinear projections → CCA on projections
    - Parameters: `x`, `y`, `k` (projections), `s` (scale)
    - O(n × k) — much faster than MIC

### B. Missing Correlation Matrix Variants

1. **Robust covariance** — Minimum Covariance Determinant (MCD)
   - Rousseeuw 1984: find h-subset with smallest determinant
   - Parameters: `data: &Mat`, `support_fraction` (default 0.5)
   - Breakdown point: 50% outliers
   - Fast-MCD: Rousseeuw & Van Driessen 1999

2. **Shrinkage covariance** — Ledoit-Wolf
   - Shrink sample covariance toward structured target (identity, diagonal)
   - Σ_shrunk = (1-α)Σ_sample + α×target
   - Automatic α from analytical formula
   - Parameters: `data: &Mat`, `target` (default: scaled identity)

3. **Oracle Approximating Shrinkage** (OAS) — Chen et al. 2010
   - Improved shrinkage estimator
   - Parameters: `data: &Mat`

4. **Graphical Lasso** (GLASSO) — Friedman et al. 2008
   - Sparse precision matrix estimation
   - max log det(Θ) - trace(SΘ) - λ||Θ||₁
   - Parameters: `data: &Mat`, `lambda`
   - Output: sparse precision matrix (inverse covariance)

5. **Partial correlation matrix** — from precision matrix
   - ρᵢⱼ|rest = -Θᵢⱼ / √(Θᵢᵢ Θⱼⱼ)
   - Already have bivariate partial; need full matrix version
   - Primitives: inv(covariance_matrix) → normalize

6. **Kendall rank correlation matrix** — pairwise Kendall for all pairs
   - Parameters: `data: &Mat` → `Mat`
   - O(p² × n log n) for p variables

7. **Heterogeneous correlation matrix** — mix of Pearson/polychoric/polyserial
   - Automatically picks appropriate correlation for each variable pair
   - Parameters: `data: &Mat`, `var_types: &[VarType]`

### C. Missing Similarity/Distance Measures

1. **Jaccard similarity** — |A∩B| / |A∪B| for sets
   - Parameters: `a: &[bool]`, `b: &[bool]`

2. **Dice coefficient** — 2|A∩B| / (|A|+|B|)
   - Parameters: `a: &[bool]`, `b: &[bool]`

3. **Adjusted Rand Index** — agreement between clusterings
   - Already have in information_theory (AMI); need ARI too
   - Parameters: `labels1`, `labels2`

4. **Rand Index** — fraction of pairs that agree
   - Parameters: `labels1`, `labels2`

5. **Fowlkes-Mallows Index** — geometric mean of precision and recall
   - Parameters: `labels1`, `labels2`

6. **Matthews Correlation Coefficient** (MCC) — balanced binary measure
   - MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
   - Parameters: `predicted`, `actual`

---

## Decomposition into Primitives

```
rank(data) ────────────┬── spearman (pearson on ranks)
                       ├── kendall_tau (inversion_count on ranks)
                       ├── somers_d
                       ├── goodman_kruskal_gamma
                       ├── chatterjee_xi
                       └── rank_biserial

concordant_discordant ─┬── kendall_tau
counts                 ├── kendall_tau_c
                       ├── somers_d
                       └── goodman_kruskal_gamma

covariance_matrix ─────┬── pearson_matrix
                       ├── partial_correlation_matrix (via inv)
                       ├── ridge / lasso covariance
                       └── cca

distance_matrix ───────┬── distance_correlation
                       ├── hsic (kernel matrix)
                       └── brownian_correlation

ANOVA components ──────── icc (all 6 forms)

contingency_table ─────┬── cramers_v
                       ├── cohens_kappa
                       ├── polychoric
                       └── gwets_ac
```

## Priority

**Tier 1** — Missing from current set that are widely needed:
1. `cohens_kappa` / `weighted_kappa` — fundamental agreement measure
2. `icc` (all 6 forms) — reliability, measurement studies
3. `polychoric` — ordinal data (tetrachoric generalization)
4. `partial_correlation_matrix` — full matrix version
5. `somers_d` / `goodman_kruskal_gamma` — trivial from existing Kendall internals
6. `chatterjee_xi` — modern, consistent, detects non-monotone
7. `shrinkage_covariance` (Ledoit-Wolf) — essential for high-dimensional

**Tier 2**:
8. `mcc` — balanced binary metric
9. `mic` — captures any functional relationship
10. `hsic` — kernel-based independence
11. `robust_covariance` (MCD) — outlier-robust
12. `graphical_lasso` — sparse precision matrix
13. `krippendorffs_alpha` — multi-rater agreement
14. `adjusted_rand_index` — clustering evaluation

**Tier 3**:
15-20: polyserial, RDC, Brownian, heterogeneous matrix, Kendall matrix


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

