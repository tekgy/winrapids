# Phantom Scan — Complete Report
Written: 2026-04-10

## The Good News: No Compiler Phantoms

Both `tambear` and `tambear-fintek` compile clean. Every `crate::` reference resolves.
The "kmeans is a phantom" note was about `crate::kmeans::kmeans` (the function) — but that
was already fixed. The `kmeans` module exists and has `KMeansEngine`. No phantom functions
that the compiler catches.

## The Real Phantoms: Missing Public Surface

These are the **structural phantoms** — real, implemented primitives that are invisible at the
crate surface because they're not in `lib.rs pub use`. Users of the library can't reach them.

### 1. Entire modules with NO pub use

These modules are `pub mod` (accessible via full path) but NOTHING is `pub use`d from them:

| Module | Key public items |
|--------|-----------------|
| `survival` | `kaplan_meier`, `KmStep`, `log_rank_test`, `CoxResult`, `cox_ph`, `GrambschTherneauResult` |
| `panel` | `FeResult`, `FdResult`, `TwoSlsResult`, `DidResult`, `panel_fe`, `panel_re`, `panel_fd`, `panel_twfe`, `hausman_test`, `two_sls`, `did` |
| `bayesian` | `McmcChain`, `metropolis_hastings`, `BayesLinearResult`, `bayesian_linear_regression`, `effective_sample_size`, `r_hat` |
| `irt` | `rasch_prob`, `prob_2pl`, `prob_3pl`, `ItemParams`, `fit_2pl`, `ability_mle`, `ability_eap`, `mantel_haenszel_dif` |
| `series_accel` | `partial_sums`, `cesaro_sum`, `aitken_delta2`, `wynn_epsilon`, `euler_transform`, `richardson_extrapolate`, `abel_sum`, `accelerate`, `ConvergenceType`, `detect_convergence` |
| `volatility` | (entire family) |
| `causal` | (entire family) |

Also these modules have NO pub use and are largely internal machinery:
`dim_reduction`, `factor_analysis`, `spectral`, `spectral_clustering`, `tda`, `mixture`, `hmm`,
`sketches`, `stats`, `proof`, `spec_compiler`, `nan_guard`, `using`, `codegen`, `copa`

Some of these (dim_reduction, tda, factor_analysis) definitely have user-facing primitives.

### 2. Rich public APIs hidden inside modules with PARTIAL pub use

#### nonparametric.rs — many primitives NOT pub use'd:

- **Bandwidth rules**: `scott_bandwidth`, `sturges_bins`, `scott_bins`, `freedman_diaconis_bins`, `doane_bins`
- **Histogram**: `BinRule`, `Histogram`, `histogram_auto`
- **ECDF**: `Ecdf`, `ecdf`, `ecdf_confidence_band`
- **Correlation measures**: `phi_coefficient`, `point_biserial`, `biserial_correlation`, `rank_biserial`, `tetrachoric`, `cramers_v`, `eta_squared`, `distance_correlation`, `concordance_correlation`
- **Sequence similarity**: `dtw`, `dtw_banded`, `levenshtein`, `quantile_symbolize`, `edit_distance_on_series`

#### complexity.rs — missing from pub use:
- `RqaResult`, `rqa` (Recurrence Quantification Analysis)
- `MfdfaResult`, `mfdfa` (Multifractal DFA)
- `CcmResult`, `ccm` (Convergent Cross Mapping)
- `PhaseTransitionResult`, `phase_transition`
- `harmonic_r_stat`, `hankel_r_stat`

These 5 are **particularly critical**: they're the primitives that fintek's family21/22/24 bridges need,
but the bridges REIMPLEMENTED them from scratch instead of calling the tambear primitives!

#### information_theory.rs — missing from pub use:
- `histogram`, `joint_histogram` (primitive building blocks)
- `transfer_entropy`
- `TfidfResult`, `tfidf`
- `cosine_similarity`, `cosine_similarity_matrix`

#### train/naive_bayes.rs — entirely missing:
- `GaussianNB`, `gaussian_nb_fit`, `gaussian_nb_predict`, `gaussian_nb_predict_proba`

#### linear_algebra.rs — missing:
- `log_det` (pub fn, not in lib.rs)
- Matrix `rank()` function (conflicts with nonparametric `rank()` — needs resolution)

### 3. Gap statistic: inline kmeans_cpu_f64 closure

In `clustering.rs:1066`, `gap_statistic` contains an inline k-means implementation:
```rust
let kmeans_cpu_f64 = |pts: &[f64], k: usize, max_iter: usize, init_seed: u64| -> Vec<i32> { ... }
```
This doesn't use `crate::kmeans::KMeansEngine`. The note says "kmeans.rs uses f32 GPU paths;
we need f64 CPU k-means here." So there's a real gap: **no pub fn kmeans_f64** primitive exists.
The KMeansEngine is f32-GPU only. A f64 CPU k-means primitive should be extracted.

## The Structural Finding: Fintek Bridges Reimplementing Tambear Primitives

`family22_criticality.rs` and `family24_manifold.rs` contain FULL reimplementations of:
- `ccm` (Convergent Cross Mapping) — tambear::complexity::ccm exists
- `mfdfa` (Multifractal DFA) — tambear::complexity::mfdfa exists
- `phase_transition` — tambear::complexity::phase_transition exists

The bridges use different API shapes (fintek wants fixed output columns, tambear returns full spectra).
This is **intentional design** but creates a divergence where:
1. Two implementations of the same math exist
2. They can drift apart
3. The tambear primitives aren't being used (defeating the purpose)

The right fix is for fintek bridges to call tambear primitives and then extract/reformat their outputs.

## The sigmoid / logistic Duplication Web

`sigmoid` / `logistic` exists in:
- `special_functions::logistic` — CANONICAL
- `neural::sigmoid` → delegates to special_functions
- `linear_algebra::sigmoid` → delegates to neural::sigmoid (pub fn — unnecessary wrapper)
- `hypothesis::sigmoid` (private) → delegates to special_functions
- `irt::logistic` (private) → delegates to special_functions
- `causal::sigmoid` (private) → delegates to neural::sigmoid

These delegates are correct but the `linear_algebra::sigmoid` pub fn is noise — it should be removed
or kept only if linear_algebra users legitimately need sigmoid without importing neural.

## Two Logistic Regression Implementations

`hypothesis::logistic_regression` — IRLS (Newton-Raphson with Hessian inverse)
`train::logistic::fit` — gradient descent with SGD

Different algorithms, not bugs. But they should be clearly documented as distinct approaches
and ideally the hypothesis version should call the train version or vice versa.

## Priority Ranking for Pathmaker

1. **CRITICAL**: Add `pub use` for `complexity::{ccm, CcmResult, mfdfa, MfdfaResult, phase_transition, PhaseTransitionResult, harmonic_r_stat, hankel_r_stat, rqa, RqaResult}` — fintek family22/24 need these
2. **HIGH**: Add `pub use` for `survival::*`, `panel::*`, `bayesian::*`, `irt::*` — complete families with no surface
3. **HIGH**: Add `pub use` for `nonparametric` correlation primitives (phi_coefficient, point_biserial, cramers_v, etc.)
4. **HIGH**: Add `pub use` for `train::naive_bayes::{GaussianNB, gaussian_nb_fit, ...}`
5. **MEDIUM**: Extract `kmeans_cpu_f64` from gap_statistic into a standalone `pub fn kmeans_f64()`
6. **MEDIUM**: Have fintek family22/24 call tambear complexity primitives instead of own implementations
7. **LOW**: Remove `linear_algebra::sigmoid` pub fn (unnecessary wrapper)
8. **LOW**: Add `pub use` for `information_theory::{histogram, joint_histogram, transfer_entropy, cosine_similarity}`

## Patterns That Connect Distant Parts

The `using()` system references functions that are accessible via crate:: paths but not
at the crate surface. When the pathmaker wires `using()` through compositions, they need to
be able to call each primitive. The missing pub uses create invisible walls in the composition graph.

The accumulate+gather unification means that every missing pub fn is also a missing node in
the composition graph. You can't compose something you can't name at Level 0 TBS.
