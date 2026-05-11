# Recipe tree — correlations (bivariate and multivariate dependence)

**Status**: Third pilot of the catalog-as-tree pattern (`recipe-trees/README.md`). Mixed-topology: overlapping kernels at the moment-based core, disjoint kernels for concordance / energy / information families.

**Drafted**: 2026-05-10 by main-thread Claude with Tekgy. Awaiting math-researcher walk-through for completeness of the literature catalog and for ratification of the four-kernel split (especially: is `DistanceCorrelation` actually a moment kernel on a *distance-transformed* sample, collapsing into the moment family?); pathmaker walk-through for accumulate+gather decomposition validation.

**Anchor**: Naturalist's `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — *"names are parameter assignments on a graph; the graph is the actual catalog."* Briefing predicted "3-4 kernels along axes (rank-vs-raw, scale-invariance, tail-sensitivity)"; this draft lands at four. Pearson + Spearman do share a moment kernel parameterized by rank-transform, as suspected.

**Pre-flight read**: `recipe-trees/README.md`, `recipe-trees/means.md`, `recipe-trees/sketches.md`, `docs/architecture/holonomic-architecture.md`.

---

## TL;DR — four kernels, ~30 literature names

The correlations family resolves to **four kernels** distinguished by *what coupling structure they measure*:

| Kernel | Measures | Parameter axes | Named leaves it covers | Disjoint from |
|---|---|---|---|---|
| **MomentCorrelation** `M<T, R, P>` | linear coupling of (possibly transformed) values via covariance / variance ratio | value-transform `T` (identity / rank / category-code / box-cox), reweighting `R`, partialing-out set `P` | Pearson, Spearman ρ, point-biserial, phi (φ), polychoric, tetrachoric, biweight-midcorrelation, Theil-Sen-as-slope, partial, semi-partial, weighted-Pearson | concordance-counting, energy-distance-based, information-theoretic |
| **ConcordanceCorrelation** `C<S, A, B>` | pairwise concordant-vs-discordant orientation | tie-handling style `S`, asymmetry mode `A` (symmetric / directed), base-pair counter `B` (Kendall / Goodman-Kruskal) | Kendall τ-a, τ-b, τ-c, Somers' D (asymmetric variant), Goodman-Kruskal γ, Blomqvist β (median-quadrant-degenerate), Schweizer-Wolff σ (sup-norm variant of τ-shape) | moment-based, energy-based, information-based |
| **EnergyCorrelation** `E<D, W>` | distance-covariance / characteristic-function distance | distance metric `D` (Euclidean / Lp / Mahalanobis / kernel-induced), weighting / sample-pairing scheme `W` | distance correlation `dCor`, distance covariance `dCov`, HSIC (kernel-induced energy), partial distance correlation, ball covariance (Pan-Tian-Liu) | moment-based, concordance-based, info-based |
| **InformationCorrelation** `I<E, B, N>` | mutual-information / dependence-via-uncertainty-reduction | entropy estimator `E` (plug-in / KSG / KDE / spline / sketch), binning / partition scheme `B`, normalization `N` (MI / NMI / MIC / Hoeffding-D-shape / U-statistic-form) | MIC, Hoeffding's D, mutual information (Shannon / Rényi), NMI / AMI, transfer entropy (lagged), MIC variants (TIC, GMIC), copula entropy | moment-based, concordance-based, energy-based |

The **multivariate-extension axis** (canonical correlation analysis, multiple correlation R², partial correlation matrix) lives orthogonal to the four kernels above — it's a *composition pattern*: "apply a bivariate kernel pairwise over a vector / matrix, then solve a generalized eigenproblem / matrix inversion." CCA is `MomentCorrelation` lifted via generalized eigendecomposition; partial-correlation matrix is `MomentCorrelation` composed with matrix-inverse on the covariance. See `multivariate-extension axis` section below.

The **copula axis** (copula-based correlations: Spearman ρ from the copula directly, Kendall τ from the copula, copula entropy, tail-dependence) is also a composition pattern, not a kernel: "estimate the copula first (a separate recipe family), then apply a bivariate kernel to copula-rank-transformed samples." See `copula axis` section below.

**Suspected from briefing, confirmed**: Pearson + Spearman do share a moment kernel parameterized by rank-transform. Kendall is its own kernel (concordance-counting state shape doesn't overlap with moment-state shape). Distance correlation + MIC are different mathematical families entirely (energy and information, respectively). One refinement: biweight-midcorrelation, polychoric, point-biserial, and phi all fall in the moment family via different value-transform / reweighting combinations — wider than briefing implied.

---

## Kernel 1 — MomentCorrelation `M<T, R, P>`

The moment-based core of the correlations family. Every correlation measure that resolves to *"covariance of transformed values, scaled by variances of transformed values"* lives here. The Pearson formula is the kernel's unbranded form; everything else is a parameter assignment.

```
M(x, y; T, R, P) = Cov_R( T(x | P), T(y | P) ) / sqrt( Var_R(T(x | P)) · Var_R(T(y | P)) )
```

where:
- `T` is the value-transform applied per-sample to x and y independently (or jointly for categorical schemes)
- `R` is the reweighting scheme (uniform / robust / kernel-weighted / outlier-downweighted)
- `P` is the set of partialed-out covariates (empty for unconditional; non-empty for partial / semi-partial)

### Parameter axes

```rust
pub struct MomentCorrelation<T, R, P> {
    pub transform: T,         // ValueTransform
    pub reweight: R,          // ReweightScheme
    pub partial: P,           // PartialingSet (possibly empty)
}

pub enum ValueTransform {
    Identity,                                              // Pearson on raw values
    Rank { style: RankStyle, tie: TieHandling },           // Spearman ρ (avg ranks)
    BinaryIndicator { positive_class: f64 },               // point-biserial (one var binary)
    DichotomousPair { x_cut: f64, y_cut: f64 },            // phi (both binary)
    PolychoricLatent { x_thresholds: Vec<f64>,             // polychoric (ordinal-to-latent-Gaussian)
                       y_thresholds: Vec<f64>,
                       solver: LatentSolver },
    TetrachoricLatent { x_cut: f64, y_cut: f64,            // tetrachoric (binary-to-latent-Gaussian)
                        solver: LatentSolver },
    BoxCox { lambda_x: f64, lambda_y: f64 },               // Box-Cox-corrected Pearson
    Custom(Box<dyn Fn(&[f64]) -> Vec<f64>>),               // any T: ℝ^n → ℝ^n
}

pub enum RankStyle {
    AverageOnTies,    // Spearman canonical
    LowerRank,
    HigherRank,
    Ordinal,          // dense / no-tie-correction
    Random,           // random-among-ties
}

pub enum TieHandling {
    AverageRank,
    BrokenAtRandom,
    KeepDuplicates,
}

pub enum ReweightScheme {
    Uniform,
    BiweightMidvariance { c: f64 },                        // biweight-midcorrelation
    HuberM { k: f64 },                                     // M-estimator-weighted
    KernelWeighted { kernel: KernelType, bandwidth: f64 }, // local-window correlations
    OutlierTrimmed { fraction: f64 },                      // trimmed Pearson / Spearman
    Custom(Vec<f64>),                                      // user-supplied weights
}

pub enum PartialingSet {
    None,                                  // unconditional correlation
    Partial { covariates: Vec<usize> },    // both x and y residualized on Z
    SemiPartial { covariates: Vec<usize>,
                  which_residualized: SemiSide },  // only x or only y residualized
}

pub enum SemiSide { X, Y }

pub enum LatentSolver {
    TwoStepML,                  // Olsson's two-step
    JointML { iters: usize },   // full joint maximum-likelihood
    Polychoric { method: PolychoricMethod },
}
```

### Literature-named leaves

| Name | transform | reweight | partial |
|---|---|---|---|
| `pearson_r` | Identity | Uniform | None |
| `spearman_rho` | Rank{Average, AverageRank} | Uniform | None |
| `point_biserial` | BinaryIndicator | Uniform | None |
| `phi_coefficient` | DichotomousPair | Uniform | None |
| `polychoric_correlation` | PolychoricLatent | Uniform | None |
| `tetrachoric_correlation` | TetrachoricLatent | Uniform | None |
| `biweight_midcorrelation` | Identity | BiweightMidvariance{c=9} | None |
| `huber_correlation` | Identity | HuberM{k=1.345} | None |
| `weighted_pearson` | Identity | Custom | None |
| `kernel_local_correlation` | Identity | KernelWeighted | None |
| `box_cox_pearson` | BoxCox | Uniform | None |
| `partial_pearson` | Identity | Uniform | Partial |
| `semi_partial_pearson` | Identity | Uniform | SemiPartial |
| `partial_spearman` | Rank | Uniform | Partial |
| `weighted_spearman` | Rank | Custom | None |
| `trimmed_pearson(α)` | Identity | OutlierTrimmed{α} | None |
| `theil_sen_correlation` | Identity (via slope-from-pairwise-medians) | (special — see Note) | None |

**Note on Theil-Sen**: the Theil-Sen *estimator* produces a slope from pairwise sample-medians. The Theil-Sen *correlation* sense is two-valued: (a) sign of the slope as a robust direction indicator, or (b) the Pearson r between (x, ŷ_Theil-Sen) and (x, y). Sense (b) fits this kernel cleanly via Identity + a residual-based reweight. Sense (a) doesn't fit — it's a direction, not a coupling magnitude. Surface for math-researcher walk-through (open question #1).

### Gaps the literature has not named (anti-YAGNI candidates)

These parameter combinations are reachable but unnamed:
- `biweight_spearman` — Rank + BiweightMidvariance (rank-transform + robust reweight; standard separately, but the composition isn't a literature name)
- `partial_biweight_correlation` — Identity + BiweightMidvariance + Partial (robust partial correlation)
- `kernel_local_spearman` — Rank + KernelWeighted (rank correlation in a moving local window)
- `box_cox_spearman` — Box-Cox + Rank (probably redundant — rank is invariant under monotone transforms — but the kernel admits it)
- `polychoric_partial` — PolychoricLatent + Partial (partial correlation of ordinal variables via latent-Gaussian)
- `trimmed_polychoric` — PolychoricLatent + OutlierTrimmed (rare in literature but admissible)

Per anti-YAGNI: every combination is reachable through the kernel without per-name implementation. Named leaves are recipe wrappers (~20 lines each); unnamed combinations are reachable via direct kernel calls.

### Accumulate + gather decomposition

For unconditional case (P = None) with arbitrary `(T, R)`:

```
MomentCorrelation<T, R, None>(x, y):
  let x_t = gather(x, transform: T)                       // T-transform (rank/binary/latent/...)
  let y_t = gather(y, transform: T)
  let w   = gather(x_t, y_t, reweight: R)                 // R yields per-sample weight
  let w_sum   = accumulate(w, expr: w(i),                                   op: Add)
  let mean_x  = accumulate((x_t, w), expr: w(i) * x_t(i),                    op: Add) / w_sum
  let mean_y  = accumulate((y_t, w), expr: w(i) * y_t(i),                    op: Add) / w_sum
  let cov_xy  = accumulate((x_t, y_t, w), expr: w(i)*(x_t(i)-mean_x)*(y_t(i)-mean_y), op: Add) / w_sum
  let var_x   = accumulate((x_t, w), expr: w(i)*(x_t(i)-mean_x)^2,           op: Add) / w_sum
  let var_y   = accumulate((y_t, w), expr: w(i)*(y_t(i)-mean_y)^2,           op: Add) / w_sum
  return cov_xy / sqrt(var_x * var_y)
```

For partial case (P = Partial{Z}):

```
MomentCorrelation<T, R, Partial{Z}>(x, y, z_matrix):
  let x_resid = gather(x, residual_on: z_matrix)          // OLS residuals via QR or normal equations
  let y_resid = gather(y, residual_on: z_matrix)
  return MomentCorrelation<T, R, None>(x_resid, y_resid)   // delegate to unconditional
```

The residualization step is itself a recipe (`ols_residual`) consumed by this one — see the means tree's cross-tree-connection thread and the (future) regression tree.

For polychoric / tetrachoric (latent-Gaussian solver):

```
MomentCorrelation<PolychoricLatent, Uniform, None>(x, y):
  let contingency = accumulate((x, y), grouping: GroupBy((x_cat(i), y_cat(i))), expr: 1, op: Add)
  let (thresh_x, thresh_y, rho) = LatentSolver.fit(contingency)    // Kingdom B/C (iterative MLE)
  return rho
```

Polychoric / tetrachoric declare **Kingdom B/C** — the latent-Gaussian threshold + correlation MLE is iterative. TAM handles the inner loop; the kernel honestly declares the kingdom.

### Sharing opportunities via TamSession

- **`mean_x`, `mean_y`, `var_x`, `var_y`**: the four moments are shareable with the means tree's `GeneralizedMean<1, R, None>` and `GeneralizedMean<2, R, None>` results on the same `(x_t, w)` tuple. Tag: `(x_t_fingerprint, weight_id)`. A pipeline computing `pearson_r(x, y)` and `arithmetic_mean(x)` and `variance(x)` on the same x shares the `mean_x` and `var_x` intermediates.
- **`cov_xy`**: shareable across consumers asking for *any* moment correlation on `(x_t, y_t, w)` — Pearson and the Theil-Sen-correlation residual-form both pull from the same cov accumulator. Tag: `(x_t_fingerprint, y_t_fingerprint, weight_id)`.
- **Ranks**: when `T = Rank`, the rank vectors `x_t = rank(x)` and `y_t = rank(y)` are shareable with the concordance tree (Kendall consumes ranks too) and with any ranks-tree recipe. Tag: `(x_fingerprint, rank_style, tie_handling)`.
- **Polychoric contingency table**: shareable across consumers asking for polychoric or other discrete-bivariate measures (chi-square test, Cramér's V, Goodman-Kruskal λ) on the same `(x_cat, y_cat)` pair. Tag: `(x_cat_fingerprint, y_cat_fingerprint)`.
- **OLS residuals** for the partial / semi-partial branch: shareable with the regression tree. Tag: `(target_fingerprint, covariate_set_fingerprint)`.

The sharing contract enforces compatibility per Tambear Contract item 3: a downstream consumer asking for "covariance of (x, y) under uniform weight" must NOT reuse a cached covariance computed under biweight reweighting even though the shape matches. The `weight_id` in the tag includes the reweight scheme.

---

## Kernel 2 — ConcordanceCorrelation `C<S, A, B>`

Pairwise concordance counting. State is *pair-orientation tallies* — concordant pairs (sign(x_i - x_j) = sign(y_i - y_j)), discordant pairs (opposite signs), and ties. Structurally disjoint from `MomentCorrelation`: there is no covariance / variance ratio; the metric is a function of (n_C, n_D, n_T_x, n_T_y).

```
C(x, y; S, A, B) = f_B( n_C - n_D, n_C + n_D, n_T_x, n_T_y, n )
```

where `f_B` is the kernel form: Kendall's `(n_C - n_D) / denom_S`, Somers' `(n_C - n_D) / (n_C + n_D + n_T_y)`, Goodman-Kruskal's `(n_C - n_D) / (n_C + n_D)`, Blomqvist's median-quadrant variant.

### Parameter axes

```rust
pub struct ConcordanceCorrelation<S, A, B> {
    pub tie_handling: S,        // TieHandling
    pub asymmetry: A,           // AsymmetryMode
    pub counter: B,             // ConcordanceCounter
}

pub enum TieHandling {
    Strict,           // ties counted separately (τ-a: ignore ties; (n_C - n_D) / C(n, 2))
    TauB,             // ties penalize denominator: sqrt((P+T_x)(P+T_y)) where P = n_C + n_D
    TauC,             // Stuart's bias correction for non-square contingency tables
    GoodmanKruskal,   // ties dropped from numerator and denominator
    Blomqvist,        // pairs split by median-quadrant only
}

pub enum AsymmetryMode {
    Symmetric,                       // Kendall τ, Goodman-Kruskal γ
    DirectedXY { which_ties: SemiSide }, // Somers' D — y conditional on x (or vice versa)
}

pub enum ConcordanceCounter {
    Naive,                              // O(n²) pairwise loop — exact
    Fenwick,                            // O(n log n) via Fenwick tree on rank-sorted x
    MergeSort,                          // O(n log n) via inversion count during merge
    Knight,                             // Knight's algorithm (O(n log n) for τ with ties)
    ScratchpadGpu { tile_size: usize }, // tile-parallel inversion count
}
```

### Literature-named leaves

| Name | tie_handling | asymmetry | counter |
|---|---|---|---|
| `kendall_tau_a` | Strict | Symmetric | Fenwick (default) |
| `kendall_tau_b` | TauB | Symmetric | Fenwick |
| `kendall_tau_c` | TauC | Symmetric | Fenwick |
| `somers_d_xy` | TauB | DirectedXY{X} | Fenwick |
| `somers_d_yx` | TauB | DirectedXY{Y} | Fenwick |
| `goodman_kruskal_gamma` | GoodmanKruskal | Symmetric | Fenwick |
| `blomqvist_beta` | Blomqvist | Symmetric | Naive (only 4 quadrants) |
| `schweizer_wolff_sigma` | Strict (sup-norm form) | Symmetric | Naive (integral, not pair-count — see Note) |

**Note on Schweizer-Wolff σ**: literally defined as `12 ∫∫ |C(u,v) - uv| du dv` where C is the empirical copula. It's *not* a pairwise concordance count — it's a copula-based L1 deviation from independence. It rhymes with τ structurally (both are bounded in [-1, 1], both are concordance-flavored measures) but its kernel is closer to `InformationCorrelation` over the copula. Provisionally placed here for catalog-adjacency; final placement is open question #4.

### Gaps the literature has not named (anti-YAGNI candidates)

- `kendall_tau_a_directed` — Strict + DirectedXY (asymmetric τ-a; rarely named because Somers' D already occupies the niche, but the kernel admits it)
- `goodman_kruskal_directed_gamma` — GoodmanKruskal + DirectedXY (γ-style numerator with Somers-style asymmetry)
- `kendall_tau_b_with_weights` — weighted concordance; rare, but the counter could be extended with per-pair weights without breaking the algorithm
- `concordance_on_kernel_smoothed` — apply a kernel smoother to (x, y) and then count concordance on the smoothed signal; admissible but unnamed

### Accumulate + gather decomposition

For Fenwick-based τ-b:

```
ConcordanceCorrelation<TauB, Symmetric, Fenwick>(x, y):
  let perm   = gather(x, addressing: SortAscending, return_permutation: true)
  let y_perm = gather(y, addressing: ApplyPermutation(perm))
  let y_rank = gather(y_perm, transform: Rank)
  let inversions = accumulate(
    y_rank,
    grouping: PrefixWithFenwick,           // Fenwick tree maintains running rank-count
    expr: count_greater_than(y_rank(i)),
    op: Add,
  )
  let n_C = C(n, 2) - inversions - tie_count
  let n_D = inversions
  let n_T_x = accumulate(x, grouping: GroupBy(x(i)), expr: C(group_size, 2), op: Add)
  let n_T_y = accumulate(y, grouping: GroupBy(y(i)), expr: C(group_size, 2), op: Add)
  let denom = sqrt((n_C + n_D + n_T_x) * (n_C + n_D + n_T_y))
  return (n_C - n_D) / denom
```

The Fenwick tree update is **Kingdom B intrinsically** — each insert into the tree depends on the tree's state from prior inserts. TAM schedules the Fenwick passes; tile-parallel implementations push the work into per-block prefix counts and merge across blocks (Kingdom A within block, Kingdom B at merge points). See DEC-031 on scheduling.

For MergeSort-based τ (Knight's algorithm):

```
ConcordanceCorrelation<_, Symmetric, MergeSort>(x, y):
  let sorted_by_x = gather((x, y), addressing: SortAscendingBy(x))
  let inversions = MergeSortAndCount(sorted_by_x.y_column)   // recursive — Kingdom B
  ...
```

### Sharing opportunities via TamSession

- **Ranks**: `y_rank` (and analogous `x_rank` for any rank-based downstream) shareable with the moment tree's `T = Rank` branch and with any ranks-tree consumer. Tag: `(y_fingerprint, rank_style, tie_handling)`.
- **Tie counts**: `n_T_x` and `n_T_y` shareable across all concordance-family recipes on the same `(x, y)`. Tag: `(x_fingerprint, y_fingerprint, tie_handling)`.
- **Sort permutations**: the permutation that sorts x is shareable with any recipe that needs x in sorted order — sketch builds, rank-based recipes, order-statistics recipes. Tag: `(x_fingerprint, sort_style)`.
- **Inversion counts**: shareable across all τ variants (τ-a, τ-b, τ-c, γ, Somers' D) on the same `(x, y)` — they all consume the same n_C, n_D, n_T_x, n_T_y tuple and only differ in denominator. Tag: `(x_fingerprint, y_fingerprint, counter_id)`.

The five τ-and-friends variants on the same `(x, y)` share *one* inversion-count build via TamSession. A pipeline computing `kendall_tau_b(x, y)`, `somers_d_xy(x, y)`, and `goodman_kruskal_gamma(x, y)` pays for one Fenwick pass.

---

## Kernel 3 — EnergyCorrelation `E<D, W>`

The Székely-Rizzo distance-correlation family plus kernel-induced HSIC. Coupling is measured via *pairwise distances* — not pairwise values, not pairwise concordance, but pairwise inter-sample distances under a chosen metric. Structurally disjoint from `MomentCorrelation` and `ConcordanceCorrelation`: the state shape is an `n × n` distance matrix (or its kernel-Gram analog), not a sufficient-stats tuple, not a pair-orientation tally.

```
dCor(x, y; D, W) = dCov(x, y) / sqrt( dVar(x) · dVar(y) )

where:
  A_ij = D(x_i, x_j) - row_mean_i(A) - col_mean_j(A) + grand_mean(A)   (centered distance matrix for x)
  B_ij = D(y_i, y_j) - row_mean_i(B) - col_mean_j(B) + grand_mean(B)   (centered for y)
  dCov² = (1/n²) Σᵢⱼ A_ij · B_ij
  dVar²(x) = (1/n²) Σᵢⱼ A_ij²
```

### Parameter axes

```rust
pub struct EnergyCorrelation<D, W> {
    pub distance: D,            // DistanceMetric (cross-tree connection — see distances.md when it lands)
    pub weighting: W,           // EnergyWeighting
}

pub enum DistanceMetric {
    Euclidean,                                 // canonical dCor
    Manhattan,                                 // L1-based dCor
    Lp { p: f64 },                             // any Lp dCor
    Mahalanobis { precision: Matrix },         // metric-tensor variant
    KernelInduced { kernel: PdKernel },        // HSIC — distance via -log(kernel)
    BallCovariance,                            // Pan-Tian-Liu ball-based
    Custom(Box<dyn Fn(&[f64], &[f64]) -> f64>),
}

pub enum EnergyWeighting {
    Unweighted,                                // canonical Székely-Rizzo
    Random { subsample_size: usize },          // random-projection / subsampling-based dCor
    PartialDistance { covariates: Vec<usize>}, // partial distance correlation
    Adjusted,                                  // bias-corrected dCor for high-dim
}
```

### Literature-named leaves

| Name | distance | weighting |
|---|---|---|
| `distance_correlation` (dCor) | Euclidean | Unweighted |
| `distance_covariance` (dCov) | Euclidean | Unweighted (numerator-only return) |
| `dcor_manhattan` | Manhattan | Unweighted |
| `dcor_mahalanobis` | Mahalanobis | Unweighted |
| `hsic` (Hilbert-Schmidt Independence Criterion) | KernelInduced{Gaussian} | Unweighted |
| `hsic_with_kernel(K)` | KernelInduced{K} | Unweighted |
| `partial_distance_correlation` | Euclidean | PartialDistance |
| `ball_covariance` | BallCovariance | Unweighted |
| `adjusted_dcor` | Euclidean | Adjusted |
| `subsampled_dcor` | Euclidean | Random |

### Gaps the literature has not named (anti-YAGNI candidates)

- `hsic_partial` — KernelInduced + PartialDistance (partial HSIC via kernel-Gram residualization; some recent literature, not canonical)
- `adjusted_hsic` — KernelInduced + Adjusted
- `dcor_with_Lp(p)` for arbitrary p — the literature names L2 and L1; arbitrary Lp is admissible but unnamed
- `mahalanobis_hsic` — kernel-Gram on a Mahalanobis-corrected base; rare composition

### Accumulate + gather decomposition

```
EnergyCorrelation<D, Unweighted>(x, y):
  // Build the n×n distance matrices A and B (Kingdom A — parallelizable over (i, j))
  let A_raw = accumulate(
    (i, j) in n×n,
    expr: D(x(i), x(j)),
    op: Identity,                              // n² output
  )
  let B_raw = accumulate(
    (i, j) in n×n,
    expr: D(y(i), y(j)),
    op: Identity,
  )

  // Center each matrix (row mean, col mean, grand mean)
  let row_mean_A = accumulate(A_raw, grouping: GroupBy(row), expr: A_raw(i,j),     op: Add) / n
  let col_mean_A = accumulate(A_raw, grouping: GroupBy(col), expr: A_raw(i,j),     op: Add) / n
  let grand_mean_A = accumulate(A_raw,                       expr: A_raw(i,j),     op: Add) / (n*n)
  let A = gather(A_raw, transform: |i, j| A_raw(i,j) - row_mean_A(i) - col_mean_A(j) + grand_mean_A)
  // Same for B

  // dCov², dVar²x, dVar²y
  let dCov_sq  = accumulate((i, j), expr: A(i,j) * B(i,j), op: Add) / (n*n)
  let dVar_x_sq = accumulate((i, j), expr: A(i,j)^2,        op: Add) / (n*n)
  let dVar_y_sq = accumulate((i, j), expr: B(i,j)^2,        op: Add) / (n*n)

  return sqrt(dCov_sq / sqrt(dVar_x_sq * dVar_y_sq))
```

The `n × n` distance-matrix materialization is the cost driver — O(n²) memory and compute. For large n, **Kingdom A** with tiled compute and on-the-fly centering avoids the full materialization; literature on "fast dCor" reduces certain special cases (e.g., 1D Euclidean) to O(n log n) via sort-and-prefix-sum tricks. The kernel honestly declares the dense-O(n²) path; the fast-special-case path is a separate optimization at the IR layer (open question #5).

### Sharing opportunities via TamSession

- **Distance matrices A_raw, B_raw**: shareable across all energy-family recipes on the same `(x, distance)` pair. Tag: `(x_fingerprint, distance_id)`. Same x with same metric → one distance-matrix build, regardless of how many y-partners and how many energy-family recipes follow.
- **Cross-tree share**: the distance matrix is also the central object for the (future) distances tree's many-points pairwise queries, for some clustering recipes (hierarchical, spectral), and for some manifold-learning recipes (MDS, Isomap). Strong cross-tree sharing potential.
- **Centered matrices A, B**: shareable across recipes asking for dCov, dVar, dCor on the same `(x, y, D)` triple. Tag: `(x_fingerprint, y_fingerprint, distance_id)`.
- **Kernel Gram matrices** (for HSIC): shareable with kernel-PCA, kernel ridge regression, kernel SVM, and other kernel-methods recipes. Tag: `(x_fingerprint, kernel_id, bandwidth)`. This is the major cross-tree share for the kernels tree.

---

## Kernel 4 — InformationCorrelation `I<E, B, N>`

The MI / Hoeffding's D / MIC family. Coupling is measured via *information shared between x and y* — mutual information, normalized variants, the maximal-information-coefficient grid-search, Hoeffding's distribution-free D statistic. Structurally disjoint from the other three kernels: state shape is a *joint distribution estimate* (binned, kernel-estimated, sketch-based, or copula-based), not a moment tuple, not a pair-tally, not a distance matrix.

```
I(x, y; E, B, N) = N_form( E(joint_dist(x, y; B)), E(marginal_x), E(marginal_y) )
```

### Parameter axes

```rust
pub struct InformationCorrelation<E, B, N> {
    pub entropy_estimator: E,         // EntropyEstimator
    pub partitioning: B,              // PartitionScheme
    pub normalization: N,             // NormalizationForm
}

pub enum EntropyEstimator {
    PluginShannon,                          // -Σ p log p over bin counts
    KraskovStoegbauerGrassberger { k: usize }, // KSG MI estimator (k-NN-based)
    KernelDensity { kernel: KernelType, bw: f64 },
    Spline { order: usize },
    SketchBased { sketch: QuantileSketch }, // cross-tree connection to sketches.md
    Renyi { alpha: f64 },                   // Rényi entropy
    Tsallis { q: f64 },                     // Tsallis entropy
}

pub enum PartitionScheme {
    EqualWidth { bins: usize },
    EqualFrequency { bins: usize },
    AdaptiveGridSearch { max_bins: usize },     // MIC's signature characteristic-matrix search
    KdTree { max_leaf: usize },                 // for KSG variants
    CopulaRank,                                  // bin in copula-rank space (Hoeffding-D-friendly)
    None,                                        // continuous estimator, no binning
}

pub enum NormalizationForm {
    RawMI,                                  // bits / nats
    NormalizedMI { method: NmiMethod },     // NMI: 2·MI/(H_x + H_y) etc.
    AdjustedMI { null_model: NullModel },   // AMI (chance-adjusted)
    MicNormalization,                       // log_2(min(B_x, B_y))-normalized — MIC
    HoeffdingD,                              // (n / (n-3)(n-4)) · (... U-statistic ...)
    TIC,                                    // total information coefficient
    GMIC,                                   // generalized MIC
    TauStar,                                // ASLR-style transformation
}
```

### Literature-named leaves

| Name | entropy_estimator | partitioning | normalization |
|---|---|---|---|
| `mutual_information_shannon` | PluginShannon | EqualWidth or EqualFrequency | RawMI |
| `mi_ksg(k)` | KraskovStoegbauerGrassberger{k} | KdTree | RawMI |
| `mi_kde` | KernelDensity | None | RawMI |
| `normalized_mi_arithmetic` | PluginShannon | EqualWidth | NormalizedMI{Arithmetic} |
| `normalized_mi_geometric` | PluginShannon | EqualWidth | NormalizedMI{Geometric} |
| `adjusted_mi` | PluginShannon | EqualWidth | AdjustedMI |
| `mic` | PluginShannon | AdaptiveGridSearch | MicNormalization |
| `tic` | PluginShannon | AdaptiveGridSearch | TIC |
| `gmic` | PluginShannon | AdaptiveGridSearch | GMIC |
| `hoeffding_d` | PluginShannon (or U-statistic) | CopulaRank | HoeffdingD |
| `mi_renyi(α)` | Renyi{α} | EqualWidth | RawMI |
| `mi_tsallis(q)` | Tsallis{q} | EqualWidth | RawMI |
| `transfer_entropy(τ)` | PluginShannon | EqualWidth | RawMI (lagged variant) |
| `copula_entropy` | PluginShannon | CopulaRank | RawMI |

### Gaps the literature has not named (anti-YAGNI candidates)

- `mic_with_ksg_estimator` — AdaptiveGridSearch combined with KSG MI rather than plug-in; admissible, possibly more robust at small n, unnamed
- `mi_sketch_based` — SketchBased estimator with EqualFrequency; cross-tree composition with the sketches tree's DDSketch / KLL for streaming MI — unnamed but tambear's natural fit
- `renyi_normalized_mi` — Renyi entropy + NormalizedMI normalization; admissible
- `copula_rank_ksg` — CopulaRank + KSG; combines two robust-to-marginal-shape strategies
- `tsallis_mic` — Tsallis-MIC variant; rare, admissible

### Accumulate + gather decomposition

For plug-in Shannon with EqualWidth binning:

```
InformationCorrelation<PluginShannon, EqualWidth{B}, RawMI>(x, y):
  let x_bin = gather(x, transform: |v| floor((v - x_min) / bin_width))
  let y_bin = gather(y, transform: |v| floor((v - y_min) / bin_width))
  let joint_counts = accumulate(
    (x_bin, y_bin),
    grouping: GroupBy((x_bin(i), y_bin(i))),
    expr: 1,
    op: Add,
  )
  let marginal_x_counts = accumulate(x_bin, grouping: GroupBy(x_bin(i)), expr: 1, op: Add)
  let marginal_y_counts = accumulate(y_bin, grouping: GroupBy(y_bin(i)), expr: 1, op: Add)
  // MI = ΣΣ p_xy log(p_xy / (p_x p_y))
  let mi = accumulate(
    joint_counts.entries(),
    expr: (joint_counts(i,j)/n) * log( (joint_counts(i,j) * n) / (marginal_x_counts(i) * marginal_y_counts(j)) ),
    op: Add,
  )
  return mi
```

For MIC (adaptive grid search):

```
InformationCorrelation<PluginShannon, AdaptiveGridSearch{max_bins}, MicNormalization>(x, y):
  let mic_value = 0
  for (B_x, B_y) such that B_x * B_y <= max_bins:
    let mi_grid = MI_with_partition(x, y, B_x, B_y)        // recurse into plug-in path
    let normalized = mi_grid / log2(min(B_x, B_y))
    mic_value = max(mic_value, normalized)
  return mic_value
```

MIC declares **Kingdom A in the inner MI compute, Kingdom B over the grid-search loop** — each grid evaluation is independent, but the outer max is a reduction with early-termination opportunities. TAM schedules the grid fan-out.

For KSG MI estimator: **Kingdom A/B mixed** — the kNN search per sample is Kingdom A (parallel-over-i), but the kNN data structure build is Kingdom B (sequential KD-tree insert) or Kingdom A (parallel construction with synchronization overhead). The kernel declares which path it took.

### Sharing opportunities via TamSession

- **Marginal histograms**: shareable across all info-family recipes on the same `(x, partitioning)` — `mutual_information`, `entropy_x`, `entropy_y`, `NMI`, `MIC` all consume the same marginal counts. Tag: `(x_fingerprint, partition_id)`.
- **Joint histograms**: shareable across MI / NMI / AMI / MIC base-MI calls on the same `(x, y, partitioning)`. Tag: `(x_fingerprint, y_fingerprint, partition_id)`.
- **Cross-tree share with entropies tree (future)**: marginal-entropy values, joint-entropy values, conditional-entropy values are all reused across info-correlation recipes and standalone entropy queries.
- **Cross-tree share with copulas tree (future)**: when partitioning = CopulaRank, the empirical copula is the shared object — feeds Hoeffding's D, copula entropy, Schweizer-Wolff σ, copula-based tail dependence.
- **KSG kNN graph**: shareable with manifold-learning recipes (UMAP, t-SNE), with kNN classifiers/regressors, with density-estimation recipes. Tag: `(x_fingerprint, k, distance_id)`. Strong cross-tree share with the (future) neighbors tree.

---

## Multivariate-extension axis (not a kernel — a composition pattern)

Multivariate correlations are *compositions over the four kernels*, not a separate kernel:

| Name | Composition |
|---|---|
| `multiple_correlation_R2` | `MomentCorrelation` applied to `(y, ŷ_OLS)` where `ŷ_OLS = OLS(y ~ X)`. Equivalently: 1 - (residual SS / total SS). |
| `partial_correlation_matrix` | Element-wise `MomentCorrelation` with `Partial{rest}` over a (p × p) matrix. Equivalently: invert the covariance, normalize the inverse. |
| `canonical_correlation_analysis` (CCA) | Generalized eigenproblem on `(C_xx^{-1} C_xy C_yy^{-1} C_yx)`; the singular values are the canonical correlations. Each canonical correlation is structurally a `MomentCorrelation` on a projected pair `(α'X, β'Y)`. |
| `partial_distance_correlation_matrix` | Element-wise `EnergyCorrelation` with `PartialDistance` over a (p × p) matrix. |
| `regularized_CCA` | CCA with shrinkage on the covariance matrices before the eigenproblem; same composition shape. |
| `kernel_CCA` | CCA in a kernel-induced feature space — composition of `EnergyCorrelation{KernelInduced}` with the CCA pattern. |

These deserve their own (small) recipes that wrap the bivariate kernel + the composition machinery (matrix-inverse, eigenproblem, etc.). The bivariate kernels stay clean; the multivariate recipes compose them with the (future) linear-algebra tree's primitives.

---

## Copula axis (also a composition pattern, not a kernel)

Copula-based correlations are *kernel-applications-after-copula-extraction*:

| Name | Composition |
|---|---|
| `spearman_rho_from_copula` | `MomentCorrelation<Rank, ...>` applied to copula-rank-transformed samples. (Equivalent to ordinary Spearman ρ when copula is empirical.) |
| `kendall_tau_from_copula` | `ConcordanceCorrelation` applied to copula-rank samples. (Equivalent to ordinary Kendall τ on raw x, y, since τ is invariant under monotone marginals.) |
| `copula_entropy` | `InformationCorrelation<PluginShannon, CopulaRank, RawMI>` — already named above. |
| `upper_tail_dependence` | Limit-based statistic from the copula's upper-tail behavior; composes with the (future) tail-estimators tree on the copula transform. |
| `lower_tail_dependence` | Analog for lower tail. |
| `schweizer_wolff_sigma` | `12 ∫∫ |C(u,v) - uv| du dv` over the empirical copula — provisionally placed under `ConcordanceCorrelation` but its kernel is really an `InformationCorrelation`-flavored copula-distance. Open question #4. |

The copula itself is extracted by the (future) copulas tree — `EmpiricalCopula`, `GaussianCopula`, `ClaytonCopula`, `GumbelCopula`, `FrankCopula`, `tCopula`, `VineCopula`. The correlation recipes consume the copula recipe's output.

---

## Cross-kernel structural map

```
                              correlations family
                                       |
        +---------------+---------------+---------------+---------------+
        |               |               |               |               |
 MomentCorrelation  Concordance    Energy         Information    (compositions)
   M<T, R, P>        C<S, A, B>    E<D, W>        I<E, B, N>     (multivariate /
        |               |               |               |          copula axes)
   +----+----+    +-----+-----+    +---+----+      +---+----+
   |    |    |    |     |     |    |   |    |      |   |    |
 Pearson    polychoric  tau_a       dCor    HSIC   MI       MIC       Hoeffding-D
 Spearman   tetrachoric tau_b       dCov    ball-cov NMI     TIC
 phi        biweight    tau_c                       AMI      GMIC
 point-     huber       Somers' D                   copula-entropy
   biserial weighted    Goodman-                    transfer-entropy
 partial-r  trimmed     Kruskal γ
 semi-      box-cox     Blomqvist β
   partial              Schweizer-Wolff σ (provisional)

                          shared via TamSession:
                          - ranks (Moment{T=Rank} ↔ Concordance ↔ ranks-tree)
                          - moments (Moment ↔ means tree)
                          - tie counts (Concordance internal sharing)
                          - distance matrices (Energy ↔ distances tree)
                          - kernel Gram matrices (Energy{KernelInduced} ↔ kernels tree)
                          - histograms / partition counts (Information ↔ sketches tree)
                          - copulas (Information / Concordance / Energy ↔ copulas tree)

                                            multivariate compositions
                                            (R², partial-corr-matrix,
                                            CCA, kernel CCA, ...)

                                            copula compositions
                                            (Spearman-from-copula,
                                            tail dependence, ...)

                                            [unnamed gaps]
                                            biweight-spearman, partial-biweight,
                                            kernel-local-spearman, MIC-with-KSG,
                                            HSIC-partial, MI-from-sketch, ...
```

Mixed topology: overlapping inside `MomentCorrelation` (one kernel covers ~12 named leaves with very different parameterizations — Pearson, Spearman, polychoric, biweight, partial all in one kernel), disjoint across the four kernels at top level. This is the most structurally varied family among the three pilots so far (means: all overlapping; sketches: all disjoint; correlations: mixed).

---

## Open questions for math-researcher walk-through

1. **Theil-Sen correlation — sense (a) vs sense (b)?** The Theil-Sen slope estimator produces a robust slope from pairwise sample-medians. Two senses for the "correlation" question: (a) sign of the slope as a direction-only indicator (doesn't fit MomentCorrelation — it's not a coupling magnitude), (b) Pearson r between (x, ŷ_Theil-Sen-line), which fits Identity + a custom reweight. Which (or both) should appear as a named leaf? Or does Theil-Sen's robust-slope idea belong in a (future) regression tree, with no native correlation leaf?

2. **Polychoric / tetrachoric kingdom declaration**: the latent-Gaussian solver is Kingdom B/C. Does this mean polychoric should be a *separate kernel from the moment kernel*, since the solver state-shape is fundamentally non-moment-shaped? My current draft places it in MomentCorrelation via the `PolychoricLatent` value-transform variant, with the kernel honestly declaring the kingdom at the transform layer. Alternative: a fifth kernel `LatentGaussianCorrelation<Discretization, Solver>` that covers polychoric, tetrachoric, biserial-latent variants. Cleaner taxonomy, more kernels.

3. **Hoeffding's D — concordance or information?** Hoeffding's D is defined as a U-statistic over five-tuples that is positive when samples deviate from independence. Its computation is concordance-shaped (pairwise comparisons over rank tuples), but its semantic is information-shaped (deviation-from-independence over the joint distribution). I placed it in InformationCorrelation with CopulaRank partitioning, but it could equally well sit in ConcordanceCorrelation with a 5-tuple counter. Which kernel does it actually belong to? Or is this a sign that there's a fifth kernel — `RankDeviation` — that covers Hoeffding D, Schweizer-Wolff σ, and rank-based independence tests as a separate family?

4. **Schweizer-Wolff σ placement**: I placed it provisionally under ConcordanceCorrelation for catalog-adjacency with τ and γ, but its computation is `12 ∫∫ |C(u,v) - uv| du dv` over the empirical copula — structurally a copula-distance, not a concordance count. Should it move to InformationCorrelation under `CopulaRank` partitioning (sharing infrastructure with copula entropy)? Or is "copula-distance correlations" a fifth kernel of its own?

5. **EnergyCorrelation fast paths**: for 1D Euclidean dCor, the literature has O(n log n) algorithms using sort + prefix-sum tricks. For higher dimensions, the canonical algorithm is O(n²). Should the kernel expose a `fast: bool` knob, or is this an IR-layer optimization (the IR detects when the distance is 1D Euclidean and routes to the fast path automatically)? Per holonomic-architecture.md, this is a provenance-addressed concern (IR-tier), not a content-addressed concern (recipe-tier).

6. **MIC's adaptive-grid-search as Kingdom**: MIC's outer grid-search is Kingdom B with early termination, but the inner MI per grid is Kingdom A. Does this earn its own composite-kingdom declaration, or do we declare the worst-case (Kingdom B) and let TAM optimize? Implication for sharing: if Kingdom B is declared, every grid evaluation is sequentially-dependent; if Kingdom A is declared for the inner with B at the outer, the fan-out is parallel — exactly what we want, but the declaration needs to match.

---

## Implementation roadmap

This doc is the catalog tree. **Implementation is downstream and not blocked by this doc.** Suggested ordering when the team turns to implementation:

1. **Build `MomentCorrelation<T, R, P>` first** — covers the most named leaves with the cleanest accumulate+gather decomposition and the strongest cross-tree sharing (means tree, ranks tree). Start with `T = Identity | Rank` and `R = Uniform | BiweightMidvariance` and `P = None | Partial`; add other transforms (BinaryIndicator, latent variants) as recipes need them.

2. **Build the recipe wrappers** for each named leaf — ~20 lines each, just parameter assignments + a docstring + a link back to this tree. `pearson_r`, `spearman_rho`, `biweight_midcorrelation`, `partial_pearson` are the high-traffic recipes.

3. **Build `ConcordanceCorrelation<S, A, B>`** — Fenwick counter is the default; expose Naive for small-n / oracle-comparison usage. The five τ-and-friends recipes (`kendall_tau_a/b/c`, `somers_d`, `goodman_kruskal_gamma`) all wrap this one kernel.

4. **Build `InformationCorrelation<E, B, N>`** — start with `(PluginShannon, EqualWidth, RawMI)` for the canonical MI recipe and `(PluginShannon, AdaptiveGridSearch, MicNormalization)` for MIC. KSG, KDE, Sketch-based estimators come later; the structure is in place from day one.

5. **Build `EnergyCorrelation<D, W>` last** among the four** — the O(n²) memory budget is the cost driver, and the kernel benefits from being last in line so the distances tree and kernels tree are already shipping (sharing fires immediately).

6. **Multivariate compositions** wrap (1)-(4) — `multiple_correlation_R2`, `partial_correlation_matrix`, `cca`, `kernel_cca` each compose a bivariate kernel with the (future) linear-algebra primitives.

7. **Copula compositions** consume the (future) copulas tree's recipes and apply (1)-(4) on the rank-transformed samples.

Sharing across kernels happens through TamSession with compatibility tags per Tambear Contract item 3. Major cross-tree shares: ranks (with the ranks tree), moments (with the means tree), distance matrices (with the distances tree), kernel Gram matrices (with the kernels tree), histograms (with the sketches tree), copulas (with the copulas tree).

---

## What the tree teaches

Three things visible from this artifact that are NOT visible from the per-name recipe list:

1. **`MomentCorrelation` covers ~12 named correlations** with one kernel and ~12 thin wrappers, including some that look very different on the surface (Pearson is "correlation of raw values"; polychoric is "MLE of correlation of latent-Gaussian-after-discretization"; biweight-midcorrelation is "robust-reweighted Pearson on raw values"). The unification fires through the `ValueTransform` axis (rank, binary, latent, identity, Box-Cox) and the `ReweightScheme` axis. The catalog-collapse argument for the moment kernel is the largest of any pilot so far.

2. **The four kernels carve the design space along *what coupling structure they measure***: moment (linear-in-transformed-values), concordance (pairwise-orientation), energy (pairwise-distance), information (joint-distribution-shape). The kernel boundary is structural — what the *state shape* of the computation looks like — not historical. A user choosing among Pearson, Kendall, dCor, MIC isn't choosing among "named algorithms," they're choosing among coupling-structure measurements.

3. **The cross-tree sharing topology is the richest of the three pilots.** Correlations share ranks with concordance, moments with means, distance matrices with distances, kernel Grams with kernels, histograms with sketches, copulas with copulas. TamSession fan-out across trees is the load-bearing infrastructure that lets the catalog scale. A pipeline computing `pearson_r`, `spearman_rho`, `kendall_tau_b`, `dCor`, `MI`, and `MIC` on the same `(x, y)` pays for: one rank build (shared between Moment{Rank} and Concordance), one distance matrix (Energy), one joint histogram (Information / MIC reuse), one moment tuple (Moment / means tree). The naïve cost would be six independent computations; the kernel decomposition collapses to roughly two heavy intermediates plus six recipe wrappers.

The naturalist's claim was *the recipe catalog becomes a map, not a list*. For correlations, the map is a four-kernel tree with one richly-overlapping kernel (moment) and three structurally-disjoint kernels (concordance, energy, information), plus two cross-cutting composition patterns (multivariate, copula). Mixed topology, as predicted; the surprise was how wide MomentCorrelation became under one kernel.

---

## Threads downstream of this tree

- **Means tree** (`recipe-trees/means.md`): MomentCorrelation consumes `arithmetic_mean` and `variance` (special cases of GeneralizedMean). The cov/var-ratio formula at the heart of Pearson is built from the means tree's `GeneralizedMean<1, R, None>` and `GeneralizedMean<2, R, None>` outputs. Sharing flows through TamSession at the moment-tuple level.

- **Ranks tree** (`recipe-trees/ranks.md`, TBD): MomentCorrelation{T=Rank} and ConcordanceCorrelation both consume ranks. The ranks tree owns the rank-transform recipes (`rank_average`, `rank_dense`, `rank_random`, `rank_ordinal`) plus tie-handling primitives. Both correlation kernels are downstream consumers.

- **Distances tree** (`recipe-trees/distances.md`, TBD): EnergyCorrelation lives at the intersection — the bivariate `(x, y)` distance correlation needs a univariate (or multivariate) distance recipe. The distances tree owns the distance primitives; EnergyCorrelation wraps them in the centered-distance-matrix scheme.

- **Kernels tree** (`recipe-trees/kernels.md`, TBD): EnergyCorrelation{KernelInduced} = HSIC consumes the kernels tree's positive-definite kernel recipes (Gaussian / Matérn / polynomial / Bessel / Laplace). The kernels tree's `KdeKernel` and `MercerKernel` taxonomies feed directly into HSIC and (via ReweightScheme::KernelWeighted) into MomentCorrelation.

- **Sketches tree** (`recipe-trees/sketches.md`): InformationCorrelation{SketchBased} can run streaming MI on top of DDSketch / KLL / GK marginal sketches. This is currently an unnamed gap in the literature but a natural tambear composition. The sketches tree's `CompressedHistogram<B, C, I>` is the backing kernel.

- **Tail-estimators tree** (`recipe-trees/tail-estimators.md`, TBD): the copula-based tail-dependence variants compose with the tail-estimators tree's recipes after copula extraction.

- **Copulas tree** (`recipe-trees/copulas.md`, TBD): the copula-axis composition pattern depends entirely on the copulas tree. All five copula-flavored correlations (spearman-from-copula, kendall-from-copula, copula-entropy, tail-dependence, Schweizer-Wolff σ) sit at this seam.

- **Information / entropies tree** (`recipe-trees/information.md`, TBD): InformationCorrelation is the bivariate-coupling face of the entropies family. Marginal entropies, joint entropies, conditional entropies live in the entropies tree; the *coupling* recipes (MI, NMI, MIC, AMI) live here. Strong cross-tree share at every entropy-estimator call.

- **Regression tree** (TBD): partial / semi-partial correlations consume OLS residuals from the regression tree. The PartialingSet axis of MomentCorrelation invokes regression-tree primitives.

- **Linear-algebra tree** (TBD): multivariate compositions (CCA, partial-correlation-matrix) consume eigendecomposition / matrix-inverse / generalized-eigenproblem recipes from the (future) linear-algebra tree.

These are not assignments — they're invitations. The tree pattern propagates naturally as someone touches a family.
