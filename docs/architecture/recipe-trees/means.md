# Recipe tree — means (central tendency / centrality)

**Status**: First pilot of the catalog-as-tree pattern (`recipe-trees/README.md`).

**Drafted**: 2026-05-08 by main-thread Claude with Tekgy. Awaiting math-researcher walk-through for completeness; pathmaker walk-through for accumulate+gather decomposition validation.

**Anchor**: Naturalist's `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — *"names are parameter assignments on a graph; the graph is the actual catalog."*

---

## TL;DR — five kernels, ~30 literature names

The "mean" / "central tendency" family resolves to **five kernels** with overlapping subsets of literature-named leaves. Most named means are reachable from at least two of these kernels via different parameterizations. The overlap is structural information, not redundancy.

| Kernel | Parameter axes | Named leaves it covers | Disjoint from |
|---|---|---|---|
| **GeneralizedMean** `M_p` | power `p`, weight `w`, filter `f` | arithmetic, quadratic, harmonic, geometric (limit), max (limit), min (limit), trimmed, winsorized, weighted, EMA, kernel-smoothed, simple moving | rank-based, count-based |
| **TransformedMean** | transform-in `g`, transform-out `g⁻¹`, base mean | arithmetic, geometric, harmonic, quadratic, contraharmonic-like via custom g | rank-based, count-based |
| **LehmerMean** `L_p` | power `p`, weight `w` | arithmetic (p=1), harmonic (p=0), contraharmonic (p=2) | filter-axis variants, rank, count |
| **RankMean** | rank percentile `q`, optional symmetry | median, quartile, percentile, IQR-midpoint, Hodges-Lehmann | non-rank-based |
| **FréchetMean** | metric `d`, optimization scheme | arithmetic (Euclidean), geometric_median (L1), Karcher (Riemannian), spherical, hyperbolic | parametric / closed-form variants |

The "ModeMean" / "modal-tendency" family deliberately lives elsewhere — it's a count-based statistic via histogram or KDE, not a transformation of values. See `recipe-trees/modes.md` (TBD).

---

## Kernel 1 — GeneralizedMean `M_p<P, W, F>`

The Hölder generalized mean (also "power mean"):

```
M_p(x; w, f) = ( Σᵢ wᵢ · f(xᵢ)^p / Σᵢ wᵢ ) ^ (1/p)
```

with limit cases for `p ∈ {-∞, 0, +∞}`:
- `p = +∞`: `max(filter(x_i))`
- `p = 0`: `geometric_mean(filter(x_i))` — i.e. `exp(Σ wᵢ log(filter(xᵢ)) / Σ wᵢ)`
- `p = -∞`: `min(filter(x_i))`

### Parameter axes

```rust
pub struct GeneralizedMean<P, W, F> {
    pub power: P,         // f64 in (-∞, +∞), or sentinel for ±∞ / 0-limit
    pub weight: W,        // WeightFn
    pub filter: F,        // FilterFn
}

pub enum WeightFn {
    Uniform,
    Custom(Vec<f64>),
    Exponential { alpha: f64 },          // wᵢ = (1-α)^(n-i-1) · α
    Linear { decay: f64 },               // wᵢ = max(0, 1 - decay·(n-i-1))
    Kernel { kind: KernelType, bandwidth: f64 },
    InverseDistance { reference: Vec<f64>, p: f64 },  // wᵢ = 1 / d(xᵢ, ref)^p
}

pub enum FilterFn {
    None,
    Trim { low: f64, high: f64 },           // drop α_low and α_high fractions
    Winsorize { low: f64, high: f64 },      // clamp α_low and α_high fractions
    Window { start: usize, end: usize },    // index range
    Threshold { lower: Option<f64>, upper: Option<f64> },
}
```

### Literature-named leaves

| Name | Power | Weight | Filter |
|---|---|---|---|
| `arithmetic_mean` | 1 | Uniform | None |
| `quadratic_mean` (RMS) | 2 | Uniform | None |
| `harmonic_mean` | -1 | Uniform | None |
| `geometric_mean` | 0 (limit) | Uniform | None |
| `cubic_mean` | 3 | Uniform | None |
| `max` | +∞ | Uniform | None |
| `min` | -∞ | Uniform | None |
| `weighted_mean` | 1 | Custom | None |
| `weighted_geometric_mean` | 0 (limit) | Custom | None |
| `weighted_harmonic_mean` | -1 | Custom | None |
| `trimmed_mean(α)` | 1 | Uniform | Trim(α, α) |
| `winsorized_mean(α)` | 1 | Uniform | Winsorize(α, α) |
| `windsorized-then-trimmed_mean` | 1 | Uniform | Trim then Winsorize composition |
| `exponential_moving_average(α)` | 1 | Exponential(α) | None |
| `linear_decay_average(d)` | 1 | Linear(d) | None |
| `simple_moving_average(N)` | 1 | Uniform | Window(n-N, n) |
| `kernel_smoothed_mean(K, h)` | 1 | Kernel(K, h) | None |
| `inverse_distance_weighted` | 1 | InverseDistance | None |
| `clipped_mean(lo, hi)` | 1 | Uniform | Threshold(lo, hi) |
| `power_mean(p)` | p | Uniform | None |
| `weighted_power_mean(p)` | p | Custom | None |

### Gaps the literature has not named (anti-YAGNI candidates)

These parameter combinations are reachable but unnamed:
- `weighted_quadratic_mean` (p=2, weight=Custom)
- `kernel_weighted_geometric_mean` (p=0-limit, weight=Kernel)
- `trimmed_geometric_mean` (p=0-limit, filter=Trim)
- `exponential_weighted_quadratic_RMS` (p=2, weight=Exponential)
- All triple-combinations: power × Custom-weight × Trim-filter
- All `power_mean(p)` for `p ∈ ℝ` not in {1, 2, 3, -1, ±∞, 0-limit}

Per anti-YAGNI: every combination is reachable through the kernel without per-name implementation. The named leaves are *recipe wrappers* (~20 lines each); the unnamed combinations are reachable via direct kernel calls with the right parameters.

### Accumulate + gather decomposition

```
GeneralizedMean<p, w, f>(x):
  let filtered = gather(x | f)              // FilterFn applied
  let weighted_p_sum = accumulate(filtered, expr: w(i) * filtered(i)^p, op: Add)
  let weight_sum    = accumulate(filtered, expr: w(i),                  op: Add)
  let mean_p = weighted_p_sum / weight_sum
  return mean_p^(1/p)
```

For `p = 0` (geometric limit):
```
  let weighted_log_sum = accumulate(filtered, expr: w(i) * log(filtered(i)), op: Add)
  let weight_sum       = accumulate(filtered, expr: w(i),                    op: Add)
  return exp(weighted_log_sum / weight_sum)
```

For `p = ±∞` (max/min limits):
```
  return accumulate(filtered, expr: filtered(i), op: Max | Min)
```

**Sharing**: `weight_sum` is reusable across multiple p-power computations on the same `(x, w, f)` triple — e.g., a recipe computing `arithmetic_mean` and `quadratic_mean` simultaneously can register `weight_sum` once via TamSession. Tag it with the `(x, w, f)` fingerprint; both consumers pull from cache.

---

## Kernel 2 — TransformedMean `T<g, g⁻¹, M>`

```
T_g(x) = g⁻¹( M(g(x)) )
```

where `M` is a base mean (typically arithmetic) and `g, g⁻¹` are an inverse-paired transform.

### Parameter axes

```rust
pub struct TransformedMean<G, GInv, M> {
    pub transform_in: G,        // f64 -> f64, must be invertible on the data domain
    pub transform_out: GInv,    // inverse of transform_in
    pub base_mean: M,           // typically GeneralizedMean<1, Uniform, None>
}
```

### Literature-named leaves

| Name | transform_in | transform_out | base_mean |
|---|---|---|---|
| `arithmetic_mean` | identity | identity | M_1 |
| `geometric_mean` | log | exp | M_1 |
| `harmonic_mean` | reciprocal | reciprocal | M_1 |
| `quadratic_mean` | square | sqrt | M_1 |
| `logarithmic_mean` | identity | identity | (different formula — Stolarsky family — see Kernel 5b below) |
| `box-cox_mean(λ)` | box_cox(λ) | inverse_box_cox(λ) | M_1 |
| `back-transformed_mean<g>` | g | g⁻¹ | M_1 |

### Overlap with Kernel 1

`geometric_mean`, `harmonic_mean`, `quadratic_mean` are reachable from BOTH `GeneralizedMean` (via power parameter) and `TransformedMean` (via transform-pair). Mathematically they produce the same answer.

**Implementation note**: which kernel is "primary"?
- `GeneralizedMean` is more expressive (handles weights and filters in the same kernel).
- `TransformedMean` is more general for *non-power* transforms (Box-Cox, custom invertible g).

Recommendation: **`GeneralizedMean` is the primary kernel**, with `TransformedMean` as a secondary kernel for non-power transforms (Box-Cox specifically). The overlapping leaves (`geometric_mean`, `harmonic_mean`, `quadratic_mean`) live as recipe wrappers on `GeneralizedMean` because that path also gets weight/filter axes "for free."

### Accumulate + gather decomposition

```
TransformedMean<g, g_inv, M>(x):
  let transformed = gather(x, transform: g)        // gather with transform
  let mean_t = M(transformed)                       // base mean kernel
  return g_inv(mean_t)
```

The `gather(x, transform: g)` is itself an atom-with-expression — `gather(addressing: All, expr: g(x))`. The transform composition is in the gather, not in a separate primitive.

---

## Kernel 3 — LehmerMean `L_p<P, W>`

```
L_p(x; w) = ( Σᵢ wᵢ · xᵢ^p ) / ( Σᵢ wᵢ · xᵢ^(p-1) )
```

### Parameter axes

```rust
pub struct LehmerMean<P, W> {
    pub power: P,    // f64
    pub weight: W,   // WeightFn (same type as Kernel 1)
}
```

(No filter axis. Lehmer mean is rare in the literature with filters; if needed, the user composes via gather.)

### Literature-named leaves

| Name | Power | Weight |
|---|---|---|
| `arithmetic_mean` | 1 | Uniform |
| `harmonic_mean` | 0 | Uniform |
| `contraharmonic_mean` | 2 | Uniform |
| `weighted_contraharmonic_mean` | 2 | Custom |
| `lehmer_mean(p)` | p | Uniform |

### Overlap with Kernel 1

`arithmetic_mean` (Lehmer p=1 ≡ GeneralizedMean p=1) and `harmonic_mean` (Lehmer p=0 ≡ GeneralizedMean p=-1) are reachable from both. **The two parameterizations give the same answer for these leaves but differ off-the-named-set.** Specifically:
- `LehmerMean(p=2)` = `contraharmonic_mean` ≠ `GeneralizedMean(p=2)` = `quadratic_mean`. Same `p` value, different formula.
- The kernels are *genuinely distinct* despite overlapping leaves.

Implementation: **LehmerMean is its own kernel**, not a sub-case of GeneralizedMean. Named leaves that exist in both kernels become recipe wrappers on whichever one is more expressive — for arithmetic and harmonic, that's GeneralizedMean (richer parameterization). For contraharmonic and Lehmer-with-arbitrary-p, that's LehmerMean.

### Accumulate + gather decomposition

```
LehmerMean<p, w>(x):
  let num   = accumulate(x, expr: w(i) * x(i)^p,     op: Add)
  let denom = accumulate(x, expr: w(i) * x(i)^(p-1), op: Add)
  return num / denom
```

**Sharing opportunity**: when `p = 1`, `denom = Σ wᵢ` (the weight sum), shared with `GeneralizedMean` via TamSession. When `p > 1`, the denominator is a power-1-lower numerator of a different LehmerMean call on the same `(x, w)` — recipe pipelines computing `LehmerMean(p)` and `LehmerMean(p-1)` share the cross-power intermediate.

---

## Kernel 4 — RankMean `R<Q, S>`

Rank-based central tendency. Disjoint from Kernels 1-3 (no closed-form transformation of values; depends on order statistics).

```
R_q(x) = ordered_x[round(q · n)]   (or interpolated for non-integer index)
```

with optional symmetric variants (e.g., Hodges-Lehmann uses pairwise averages).

### Parameter axes

```rust
pub struct RankMean<Q, S> {
    pub quantile: Q,           // f64 in [0, 1], or vec for multi-quantile
    pub style: S,              // RankStyle: Lower, Higher, Linear, NearestEven, ...
    pub symmetry: Option<SymmetryFn>,  // None | HodgesLehmann | ...
}
```

Quantile-style enum maps to NumPy's quantile interpolation modes (linear, lower, higher, midpoint, nearest, etc.) — published convention.

### Literature-named leaves

| Name | Quantile | Style | Symmetry |
|---|---|---|---|
| `median` | 0.5 | Linear (or NearestEven) | None |
| `first_quartile` | 0.25 | Linear | None |
| `third_quartile` | 0.75 | Linear | None |
| `iqr_midpoint` | (0.25, 0.75) → midpoint | Linear | None |
| `percentile(p)` | p | Linear | None |
| `hodges_lehmann_estimator` | 0.5 | Linear | HodgesLehmann |
| `tukey_trimean` | (0.25, 0.5, 0.5, 0.75) → weighted | Linear | None |

### Composition with sketches

For streaming data, `RankMean` is realized through a quantile sketch (DDSketch by default per `vocabulary.md` 2026-05-08, with `using(sketch: ...)` overrides). The kernel + sketch compose: `RankMean<q>(stream)` calls `sketch.add(x)` for each element, then `sketch.quantile(q)` at the end.

This is the structural place where the means tree connects to the sketches tree — `RankMean` is a `using(sketch: ...)`-parameterized subset of the sketches family.

### Accumulate + gather decomposition

```
RankMean<q, style>(x):
  let sorted = gather(x, addressing: SortAscending)    // gather with order
  let idx = q · (n - 1)
  return interpolate(sorted, idx, style)
```

For streaming:
```
RankMean<q>(stream):
  let sketch = accumulate(stream, expr: x(i), op: AddToSketch{algorithm})
  return sketch.quantile(q, style)
```

`AddToSketch` is itself a `using(sketch: ...)`-parameterized accumulator, defaulting to DDSketch.

---

## Kernel 5 — FréchetMean `F<d, M>`

Generalizes arithmetic mean to metric spaces. Also called "centroid" or "barycenter."

```
F_d(x) = arg min_y Σᵢ d(xᵢ, y)²
```

For `d = L2`, this collapses to arithmetic mean. For `d = L1` (Manhattan / sum-absolute-deviation), this is the geometric median. For `d = Riemannian`, this is the Karcher mean on a manifold.

### Parameter axes

```rust
pub struct FrechetMean<D, S> {
    pub metric: D,            // distance function or metric tag
    pub solver: S,            // ClosedForm | GradientDescent { rate, iters } | Weiszfeld { iters }
}
```

### Literature-named leaves

| Name | Metric | Solver |
|---|---|---|
| `arithmetic_mean` | L2 | ClosedForm |
| `geometric_median` | L1 | Weiszfeld |
| `karcher_mean` (Riemannian) | Riemannian | GradientDescent |
| `spherical_mean` | spherical_geodesic | Spherical-specific solver |
| `hyperbolic_centroid` | Poincaré | Hyperbolic-specific solver |
| `tukey_depth_median` | Tukey-half-space | Half-space optimizer |
| `weber_problem_solution` | L1, with weights | Weiszfeld |

### Overlap with Kernel 1

`arithmetic_mean` (Fréchet under L2) ≡ `GeneralizedMean(p=1)`. Same answer, vastly different computation: closed-form sum/n vs an iterative optimizer.

Implementation: **arithmetic_mean is a recipe wrapper on `GeneralizedMean`**, NOT on `FrechetMean`. FrechetMean is the kernel for non-Euclidean metric spaces. The overlap exists for the Euclidean case but the closed-form path wins on cost.

### Accumulate + gather decomposition

For `d = L2` (closed-form case):
```
FrechetMean<L2, ClosedForm>(x):
  return GeneralizedMean<1, Uniform, None>(x)    // delegate
```

For general `d`:
```
FrechetMean<d, GradientDescent { rate, iters }>(x):
  let y = init(x)                                 // initial guess
  for _ in 0..iters:
    let grad = accumulate(x, expr: ∇_y d(x(i), y)², op: Add)
    y = y - rate · grad
  return y
```

This is **Kingdom B/C** (sequential / iterative) territory per DEC-031. TAM handles scheduling; the kernel is honest about its iteration structure.

---

## Cross-kernel structural map

Visual representation of the means tree's overlap structure:

```
                    arithmetic_mean
                  /       |       \
       GeneralizedMean   LehmerMean   FréchetMean
            |               |             |
            ├── geometric_mean (p=0-limit) ──── via TransformedMean (in=log, out=exp)
            ├── harmonic_mean (p=-1) ────────── via TransformedMean (in=1/x, out=1/x)
            │                                    via LehmerMean (p=0)
            ├── quadratic_mean (p=2) ────────── via TransformedMean (in=x², out=√x)
            ├── max / min (p=±∞ limits)
            ├── trimmed_mean (Filter=Trim)
            ├── winsorized_mean (Filter=Winsorize)
            ├── EMA / SMA / kernel-smoothed (various Weight)
            └── weighted variants

                                  RankMean
                                  /     \
                            median    quartiles    hodges_lehmann   trimean
                                                          ...

                                  FréchetMean
                                  /         \
                          geometric_median   karcher_mean   spherical_mean

                                 [unnamed gaps]
```

---

## Open questions for math-researcher walk-through

1. **Lehmer mean at non-integer p**: the formula `Σ x^p / Σ x^(p-1)` is well-defined for `p ∈ ℝ \ {0, 1}`. For `p = 0` and `p = 1` the formula collapses to special cases (harmonic and arithmetic). Is the "Lehmer mean of order p" worth shipping for arbitrary real p, or do we restrict to integer p with named leaves (1, 2 = contraharmonic) plus the limits?

2. **TransformedMean with non-monotone transforms**: the current parameterization assumes `g` is invertible and monotone. If `g` is non-monotone (e.g., `g = abs`), the inverse is not well-defined. Should the kernel reject non-invertible transforms at construction (F13 antibody), or accept any `(g, g_inv)` pair and let the user assert correctness?

3. **Hodges-Lehmann symmetry as a separate kernel?** The H-L estimator is `median(pairwise averages)` — it's structurally a composition of `RankMean` over a derived sequence, not a parameter assignment on RankMean. Should it be its own kernel or stay buried in `RankMean::symmetry`?

4. **Mode-finding / count-based central tendency**: deliberately deferred to `recipe-trees/modes.md`. Or should mode-as-mean live here for catalog completeness, just to surface the disjoint structure?

5. **Stolarsky / Lehmer-extended families**: there's a richer literature of "logarithmic mean," "identric mean," "Heronian mean" — these have their own parameterized form (Stolarsky mean `S_p,q`). Worth a sixth kernel, or fold into TransformedMean as "non-standard transforms"?

6. **Anti-YAGNI scope**: should every named gap above (e.g., `weighted_quadratic_mean`, `trimmed_geometric_mean`) become a recipe wrapper at the same time the kernel is implemented, or do we ship the kernel + the literature-named leaves and let users pull on `using()` for unnamed gaps?

---

## Implementation roadmap

This doc is the catalog tree. **Implementation is downstream and not blocked by this doc.** When pathmaker (or whoever) implements the means family:

1. Build `GeneralizedMean<P, W, F>` first — it covers the most named leaves with the cleanest accumulate+gather decomposition.
2. Build the recipe wrappers for each named leaf — ~20 lines each, just parameter assignments + a docstring + a link back to this tree.
3. Build `TransformedMean<G, GInv, M>` for the Box-Cox / custom-transform leaves; reuse `GeneralizedMean` as the base mean.
4. Build `LehmerMean<P, W>` for contraharmonic and arbitrary-p Lehmer; share the `Σ wᵢ` accumulator with `GeneralizedMean` via TamSession.
5. Build `RankMean<Q, S>` separately, threading through the sketch family for streaming.
6. Build `FrechetMean<D, S>` for non-Euclidean metric centroids, declaring Kingdom B/C.

Sharing across kernels happens through TamSession with compatibility tags (per Tambear Contract item 3): `Σ wᵢ` is reusable across `GeneralizedMean(p=1)` and `LehmerMean(p=1)` because both define the weight sum identically; not reusable across `GeneralizedMean(p=2)` and `LehmerMean(p=2)` because the "p=2" parameter means different things in each kernel.

---

## What the tree teaches

Three things visible from this artifact that are NOT visible from the per-name recipe list:

1. **The `GeneralizedMean` kernel covers ~15 named means.** One kernel + 15 wrappers ≪ 15 implementations. That's the catalog-collapse argument made concrete.

2. **Named leaves with multiple parameterizations** (geometric, harmonic, quadratic, arithmetic) are reachable through 2-3 different kernels. The user picks the path that fits their context (which kernel they already have a TamSession cache for, which axes they want to keep tunable, etc.). The literature naming is a cross-section, not a partition.

3. **Gaps are structural** — `weighted_quadratic_mean` and `trimmed_geometric_mean` aren't named in the literature but are reachable in the kernel for free. tambear exposes them via direct kernel calls; the lack of a name is a literature artifact, not a math fact.

The naturalist's claim was: *implementing-by-name doesn't scale to "all math"; implementing-by-graph is the only architecture that does.* For means alone, this saves ~14 implementations vs 1 + 15 thin wrappers, and exposes gaps the literature hasn't named. Multiply by every recipe family in tambear and the saved-work is days; the surfaced gaps are research opportunities; the catalog becomes a *map*, not a list.

---

## Threads downstream of this tree

- **Distances tree** (`recipe-trees/distances.md`, TBD): same pattern. Minkowski(p) is the GeneralizedMean's structural cousin (Lp = (Σ |xᵢ|^p)^(1/p)). Mahalanobis, Wasserstein, MMD live in the parametric-distance kernel; cosine-similarity in the inner-product kernel. Likely 4-5 kernels with overlap.

- **Correlations tree** (`recipe-trees/correlations.md`, TBD): rank-vs-raw axis, scale-vs-tail axis, dependency-shape axis. Pearson, Spearman, Kendall, distance correlation, MIC, Hoeffding's D — likely 3 kernels.

- **Sketches tree** (`recipe-trees/sketches.md`, TBD): the naturalist's worked example. `CompressedHistogram<B, C, I>` parameterized by (bucket, compress, interp). DDSketch / KLL / GK / t-digest as parameter assignments.

These are not assignments — they're invitations. The tree pattern propagates naturally as someone touches a family.
