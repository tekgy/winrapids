# Oracle Coverage Map — Tensor Structure

*2026-04-10 — observer (following r-gap-scan exchange)*

---

## The Tensor Frame

The seed document described coverage along the *input axis* — which inputs have been
checked against an external oracle. r-gap-scan added the missing second axis: *claim type*.

The full coverage map is a tensor: **(primitive, claim type, input region)**

### Three claim types

| Type | Description | Examples |
|------|-------------|---------|
| **Type 1 — Formula correctness** | Theorem assertions: linearity, symmetry, special cases, limiting behavior. Mathematical necessity — if code disagrees, the formula is wrong. | `exp(t·I) = e^t·I`, `τ = 1` for monotone, `r(x+c, y) = r(x, y)` |
| **Type 2 — Numerical precision** | Oracle comparison at high precision (mpmath 50dp, SymPy closed-form). Catches coefficient errors, wrong series, normalization mistakes. | `|erfc(0.5) - oracle| < 1e-14` |
| **Type 3 — Robustness** | Adversarial inputs: NaN, singular, collinear, zero variance, ties, extreme magnitudes. | NaN propagation waves 1-19, adversarial gauntlet |

A primitive can have any combination. The highest-leverage configuration is all three.
The adversarial wave sweep was excellent Type 3. The workup files give Type 1 + Type 2.
Type 1 alone (theorem assertions) requires no external tools — just mathematical knowledge.

---

## Fan-out Weighting

From r-gap-scan: coverage effort should weight by fan-out. A verified shared intermediate
propagates correctness to all consumers simultaneously.

Fan-out table (from `intermediates.rs` docstrings, verified 2026-04-10):

| Intermediate | Producer primitive | Direct consumers | Fan-out |
|---|---|---|---|
| `DataQuality` | `data_quality::from_slice` | 18+ (every validity predicate, auto-detect chain) | **highest** |
| `CovarianceMatrix` | `multivariate::covariance_matrix` | pca, factor_analysis, lda, cca, manova, mahalanobis_distances, ridge, vif | **8** |
| `FFTOutput` | spectral primitives | spectral entropy/centroid/rolloff/flux, fintek Family 6 leaves | **6+** |
| `SortedArray` | `descriptive::sorted_nan_free` | shapiro_wilk, dagostino_pearson, ks_test_normal, quantile, rank, inversion_count, KDE | **7** |
| `MomentStats` | `descriptive::moments_session` | t-tests, ANOVA, normalization, z-scoring, Pearson, regression | **5+** |
| `EigenDecomposition` | `linear_algebra::sym_eigen` | pca, spectral_clustering, factor_analysis, manova | **4** |
| `DelayEmbedding` | `complexity::delay_embed` | correlation_dimension, largest_lyapunov, ccm, family15 | **4** |
| `GramMatrix` | OLS/ridge | OLS, ridge, ridge cross-validation | **3** |
| `Ranks` | `nonparametric::rank` | Spearman, Kendall | **2** |

---

## Tensor Cells — Verified State (2026-04-10)

### `covariance_matrix` — fan-out 8, zero theorem coverage

| Claim type | State | Evidence |
|---|---|---|
| Type 1 (formula correctness) | **BLACK** | No symmetry test (`C[i,j] = C[j,i]`), no shift invariance (`cov(X+c) = cov(X)`), no PSD test, no `cov(αX) = α²cov(X)` |
| Type 2 (numerical precision) | **BLACK** | No workup file. No mpmath oracle. Not in gold_standard_parity.rs. |
| Type 3 (robustness) | **ORANGE** | Tested indirectly via Hotelling, MANOVA, PCA tests. No direct adversarial inputs (all-identical rows, rank-deficient X, single-row X). |

**Impact**: Any formula error in `covariance_matrix` propagates undetected to PCA, LDA, CCA,
Mahalanobis distances, ridge regression, VIF, factor analysis, MANOVA — 8 methods simultaneously.

The implementation (`multivariate.rs:46-71`) uses a two-pass algorithm (col_means then
centered products). Type 1 tests that could be added immediately with no external tools:
- `C[i,j] = C[j,i]` for all i,j (symmetry — mathematical necessity)
- `cov(X, ddof=0)[i,i] = var(X[:,i], ddof=0)` (diagonal consistency)
- `cov(X+c) = cov(X)` for column-wise shift (shift invariance)
- `cov(αX) = α² cov(X)` for scalar scale (scale covariance)
- `cov(n=1) = NaN or zero matrix` (degenerate case)

These are provable from the formula, require no external oracle, and would catch any wrong
implementation of the centered-product formula.

---

### `sorted_nan_free` / sort — fan-out 7

| Claim type | State | Evidence |
|---|---|---|
| Type 1 | **YELLOW** | Transitivity asserted implicitly (downstream rank tests). No explicit `sorted[i] ≤ sorted[i+1]` loop test. |
| Type 2 | **BLACK** | Sort is deterministic — oracle comparison would just be `===` reference. Unnecessary. |
| Type 3 | **ORANGE** | NaN stripping tested via `sorted_nan_free` name and downstream use. Not adversarially probed (all-NaN input, all-identical input, single element). |

Type 1 gap: the explicit `∀i: sorted[i] ≤ sorted[i+1]` assertion is a mathematical theorem
about the output that isn't directly tested. Easy to add.

---

### `MomentStats` — fan-out 5+

| Claim type | State | Evidence |
|---|---|---|
| Type 1 | **ORANGE** | Welford merge tested for associativity in adversarial wave 15 (Welford merges). Mean formula tested implicitly. |
| Type 2 | **BLACK** | No mpmath oracle for Welford accumulation. |
| Type 3 | **ORANGE** | NaN propagation tested. Single-element, all-identical tested via t-test boundary cases. |

Welford merge: `SarkkaMerge` (Op) was theorem-tested in adversarial wave 13 (scan
identity/associativity proof). So the scan operation is verified, but the final
extraction (converting Welford (n, mean, M2) to variance/skewness/kurtosis) is not
theorem-tested against mpmath.

---

### `EigenDecomposition` (sym_eigen) — fan-out 4

| Claim type | State | Evidence |
|---|---|---|
| Type 1 | **YELLOW** | `workup_pinv.rs` verifies SVD via reconstruction `A ≈ UΣVᵀ`. Eigendecomposition shares structure. `Av = λv` not directly tested. |
| Type 2 | **YELLOW** | pinv workup has mpmath oracle. SVD accuracy tested. |
| Type 3 | **ORANGE** | Singular/near-singular tested via pinv. |

Best-covered of the shared intermediates. But `sym_eigen` specifically (symmetric eigdecomp
used by PCA/spectral clustering) has no direct tests — only through downstream PCA variance
retention tests.

---

## Priority Order for Workup Effort

Sorted by: (fan-out × type1_gap × type2_gap):

1. **`covariance_matrix`** — fan-out 8, all types missing. One workup fills 8 downstream black cells.
2. **`DataQuality` per-field** — fan-out 18+, composite structure. One field at a time.
3. **`SortedArray`** — fan-out 7, Type 1 gap is a one-line assertion (`sorted[i] ≤ sorted[i+1]`).
4. **`MomentStats`** — fan-out 5+, Welford extraction needs oracle for skewness/kurtosis formulas.
5. **`FFTOutput`** — fan-out 6+, Parseval's theorem already asserted once. Needs per-region oracle.

---

## Updated Yellow Count (corrected measurement)

`gold_standard_parity.rs`: **826 tests** (not 471 — file grew during waves 16-23).

The seed document's estimate of "80-120 yellow primitives" should be revised upward.
826 tests across ~200+ functions in the parity file suggests ~150-200 yellow primitives.

---

## The r-gap-scan Insight (recorded)

From the garden entry `2026-04-10-the-tests-know-before-the-codebase.md`:

> "The codebase encodes structure; the tests encode semantics. The observer role is the bridge:
> reading the tests (semantic claims) against the code (structural implementation) and finding
> the gaps. That's not oversight in the management sense — it's a specific kind of knowing that
> neither tests nor code has alone."

The tensor structure is the concrete form of this: every (primitive, claim type, input region)
cell where the code has structure but the tests have no semantic claim is a gap where a
Padé-class error could hide.

---

*Measurement date: 2026-04-10. Next: systematic per-module scan to replace estimates with counts.*
