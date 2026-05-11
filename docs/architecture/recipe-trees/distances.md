# Recipe tree — distances (dissimilarity / metric / divergence)

**Status**: Third recipe-tree draft, following the `means.md` (overlapping-kernels) and `sketches.md` (disjoint-kernels) pilots.

**Drafted**: 2026-05-10 by main-thread sub-agent under team-lead delegation. Awaiting math-researcher walk-through for completeness of the divergence family + the metric-vs-divergence boundary; pathmaker walk-through for accumulate+gather decomposition validation.

**Anchor**: Naturalist's `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — *"names are parameter assignments on a graph; the graph is the actual catalog."* The briefing at `R:\winrapids\docs\expedition\recipe-trees-continuation-briefing.md` suggested 4-5 kernels with mixed topology; the topology that emerged from literature is **six kernels with mixed topology** — some pairs overlap on named leaves (Euclidean = L2 = inner-product-derived), most are genuinely disjoint at the state-shape level (string-edit cannot be expressed in Lp space).

**Pre-flight reading (in order)**:
1. `recipe-trees/README.md` — pattern, six steps, two topologies
2. `recipe-trees/means.md` — overlapping pilot
3. `recipe-trees/sketches.md` — disjoint pilot
4. `holonomic-architecture.md` — catalog-tier placement
5. `expedition/recipe-trees-continuation-briefing.md` — § Distances
6. `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — pattern source

---

## TL;DR — six kernels, ~50+ literature names, mixed topology

The "distance" family is broader than means or sketches and resolves to **six kernels** along the dimension *"what underlying structure does the distance read?"* — coordinates, inner products, sets, probability distributions, sequences, or graph adjacency.

| Kernel | Reads | Parameter axes | Named leaves it covers | Topology relation |
|---|---|---|---|---|
| **MinkowskiNorm** `M<p, W, F>` | coordinate vectors | order `p`, weight `W`, filter `F` | Euclidean (L2), Manhattan (L1), Chebyshev (L∞), Minkowski-p, fractional-Lp, weighted-Lp, Canberra, Bray-Curtis | overlaps with InnerProductDistance at p=2 |
| **InnerProductDistance** `I<G, B, N>` | bilinear / inner products | Gram-form `G`, bias `B`, normalization `N` | cosine distance, angular distance, Mahalanobis, correlation distance, dot-product distance, Bregman-Mahalanobis | overlaps with MinkowskiNorm at L2; disjoint elsewhere |
| **SetDistance** `S<T, N>` | finite sets / binary vectors | set transform `T`, normalization `N` | Jaccard, Sørensen-Dice, Tversky, Hamming, Tanimoto, Ochiai, Overlap coeff, Kulczynski | disjoint from all others |
| **Divergence** `D<F, S>` | probability distributions | f-form `F`, symmetrization `S` | KL, JS, Hellinger, total variation, χ², α-divergence, Rényi, Bregman, Itakura-Saito, energy distance, MMD, Wasserstein-p, Sinkhorn, Cramér | overlaps with MinkowskiNorm via Wasserstein-p of discrete measures |
| **SequenceEdit** `E<C, A, N>` | strings / token sequences | cost matrix `C`, alignment `A`, normalization `N` | Levenshtein, Damerau-Levenshtein, Hamming-string, LCS, Jaro, Jaro-Winkler, Smith-Waterman, Needleman-Wunsch, OSA, edit-tree, normalized-edit | disjoint from all others |
| **GraphDistance** `G<W, T, S>` | graphs / nodes-in-graph | walk weighting `W`, traversal `T`, summary `S` | shortest-path (Dijkstra, Bellman-Ford, Floyd-Warshall), graph-edit, spectral, resistance, commute-time, diffusion, random-walk, Weisfeiler-Lehman | disjoint at state-shape; overlaps with MinkowskiNorm via spectral-embedding lifting |

**Cross-cutting axis** that is NOT itself a kernel but parameterizes *several* kernels: **PairwiseTopology** — "do I want the full N×N matrix, k-NN graph, ε-ball graph, or single-pair?" Lives as `using()` knob inside the kernel calls, dispatched by TAM scheduler.

**The topology distinction is mixed** in the sense the briefing predicted:
- MinkowskiNorm ↔ InnerProductDistance: **overlapping** at the L2 / cosine-and-dot-product seam (Euclidean = √(2-2·cos(θ)) on the unit sphere)
- SetDistance ↔ everything: **disjoint** (binary-vector state doesn't reduce to coordinate state without information loss)
- Divergence ↔ MinkowskiNorm: **overlapping at the Wasserstein-p ↔ Lp-of-CDF seam** for 1D (Wasserstein-1 of empirical distributions = L1 of sorted-values)
- SequenceEdit ↔ everything: **disjoint** (the state is a multi-step alignment lattice, not a vector)
- GraphDistance ↔ MinkowskiNorm: **partially overlapping** through spectral embedding (graph → eigenvector coords → Lp), but the *kernel* differs because the embedding step is lossy and parameter-dependent

This mirrors the briefing's hint: not pure overlapping (like means) and not pure disjoint (like sketches), but a **graph of partial overlaps** clustered into roughly two regions — the "geometric" cluster (MinkowskiNorm + InnerProductDistance + parts of Divergence) and the "combinatorial" cluster (SetDistance + SequenceEdit + GraphDistance).

---

## Kernel 1 — MinkowskiNorm `M<p, W, F>`

The Minkowski (Hölder) norm family. Structural cousin of `GeneralizedMean` in `means.md` — same `Σ wᵢ |xᵢ|^p` skeleton, different exponent role:

```
d_p(x, y; w, f) = ( Σᵢ wᵢ · |f(xᵢ) - f(yᵢ)|^p ) ^ (1/p)
```

with limit cases:
- `p = +∞`: `max_i |f(xᵢ) - f(yᵢ)|` (Chebyshev)
- `p = 1`: `Σᵢ wᵢ · |f(xᵢ) - f(yᵢ)|` (Manhattan / city-block / L1)
- `p = 2`: `√(Σᵢ wᵢ · (f(xᵢ) - f(yᵢ))²)` (Euclidean)
- `p < 1`: "fractional Lp" — not a metric (fails triangle inequality) but still useful for outlier-robust distance

### Parameter axes

```rust
pub struct MinkowskiNorm<P, W, F> {
    pub order: P,                // f64 in (0, +∞], or sentinel for ∞ / non-metric warnings
    pub weight: W,               // WeightFn — same enum vocabulary as GeneralizedMean
    pub filter: F,               // FilterFn — drop coords by index/threshold/mask
}

pub enum WeightFn {
    Uniform,
    Custom(Vec<f64>),
    InverseVariance { var: Vec<f64> },        // Mahalanobis-diagonal special case
    InverseRange { ranges: Vec<f64> },        // Gower-like coord normalization
    InverseScale { scales: Vec<f64> },        // for Canberra / Bray-Curtis
    Kernel { kind: KernelType, bw: f64 },     // local-weighted distance
}

pub enum FilterFn {
    None,
    Mask(BitVec),                             // skip masked coords (NA-handling)
    Threshold { tol: f64 },                   // ignore coord pairs where |xᵢ-yᵢ| < tol
    TopK { k: usize },                        // largest-k contributions only
}
```

### Literature-named leaves

| Name | order | weight | filter | Notes |
|---|---|---|---|---|
| `euclidean_distance` (L2) | 2 | Uniform | None | metric; overlaps with InnerProductDistance |
| `manhattan_distance` (L1) | 1 | Uniform | None | metric; "city block" |
| `chebyshev_distance` (L∞) | ∞ | Uniform | None | metric; "king-move" / max-norm |
| `minkowski_distance(p)` | p | Uniform | None | metric for `p ≥ 1` |
| `fractional_lp_distance(p)` | p<1 | Uniform | None | non-metric; outlier-robust |
| `weighted_euclidean` | 2 | Custom | None | metric |
| `weighted_manhattan` | 1 | Custom | None | metric |
| `mahalanobis_diagonal` | 2 | InverseVariance | None | special case of full Mahalanobis (Kernel 2) |
| `canberra_distance` | 1 | InverseScale (|xᵢ|+|yᵢ|) | None | metric; near-zero-sensitive |
| `bray_curtis_distance` | 1 | (Σ|xᵢ+yᵢ|)⁻¹ | None | non-metric; abundance ecology |
| `gower_distance` | 1 | InverseRange | None | mixed-type variables |
| `squared_euclidean` | 2 | Uniform | None | drop the outer √; Bregman-divergence form |
| `chord_distance` | 2 | Uniform (after L2-normalize) | None | √2·sin(θ/2); composition |
| `hellinger_via_sqrt` | 2 | Uniform (after √-transform) | None | crosses into Divergence; see Kernel 4 |
| `lp_norm(p)` of single vector | p | Uniform | None | `d(x, 0)` — magnitude |

### Gaps the literature has not named (anti-YAGNI candidates)

Reachable but unnamed parameter combinations:
- `weighted_chebyshev` (∞-order × Custom-weight) — max of weighted coord-differences
- `trimmed_euclidean` (2 × Uniform × TopK) — Euclidean over largest-k coordinate gaps
- `masked_minkowski(p, mask)` — Lp over a coord-subset; handy for NA-aware distance
- `inverse_scale_l3` (3 × InverseScale × None) — cubic analog of Canberra
- `tukey-weighted lp` (any p × custom kernel-weight × None) — robust regression flavor
- `triple-combo`: order p × Custom weight × TopK filter — domain-specific tuning
- Continuous order `p ∈ ℝ⁺ \ {1, 2, 3, ∞}` without naming

Per anti-YAGNI: every combination is reachable through the kernel without per-name implementation.

### Accumulate + gather decomposition

```
MinkowskiNorm<p, w, f>(x, y):
  let masked = gather((x, y) | f)                    // FilterFn applied per-coord
  let weighted_diff_p = accumulate(
    masked,
    expr: w(i) * |masked.x(i) - masked.y(i)|^p,
    op: Add,
  )
  return weighted_diff_p^(1/p)
```

For `p = ∞`:
```
  return accumulate(masked, expr: w(i) * |x(i) - y(i)|, op: Max)
```

For `p = 1`:
```
  return accumulate(masked, expr: w(i) * |x(i) - y(i)|, op: Add)
```

### Sharing opportunities via TamSession

- **`|xᵢ - yᵢ|` intermediate**: shareable across multiple `p` orders on the same `(x, y, w, f)`. A recipe computing `euclidean` and `manhattan` simultaneously builds the absolute-difference vector once.
- **`(x - y)²` intermediate**: shareable across `euclidean`, `squared_euclidean`, and the Bregman-divergence form. Tag: `(x_id, y_id, filter_id)`.
- **`weight_sum`**: same as the means tree — reusable wherever the weight-normalizer fires.
- **N×N pairwise**: when the consumer asks for the full pairwise distance matrix, the gather addressing changes (PairwiseAll), but the same kernel applies per-pair. Shareable across downstream consumers (clustering, MDS, k-NN) via the matrix-id tag.

---

## Kernel 2 — InnerProductDistance `I<G, B, N>`

Distances built from inner products — bilinear forms or kernel evaluations. The kernel reads coordinate vectors but compresses them through a Gram form before extracting distance.

```
d_G(x, y; B, N) = N⁻¹( B( ⟨x - y, x - y⟩_G ) )
```

where `⟨·,·⟩_G` is a Gram-form-defined inner product, `B` is a base distance transform (squared, square-rooted, normalized), and `N` is a normalization (unit-sphere for cosine, identity for raw).

### Parameter axes

```rust
pub struct InnerProductDistance<G, B, N> {
    pub gram: G,                  // GramForm
    pub base: B,                  // BaseTransform (Squared, Rooted, Identity)
    pub normalize: N,             // NormalizationMode
}

pub enum GramForm {
    Identity,                              // standard L2 inner product
    Diagonal { weights: Vec<f64> },        // weighted inner product
    FullPSD { matrix: Mat<f64> },          // Mahalanobis full-precision
    KernelMatrix { precomputed: Mat<f64> },// RKHS via stored Gram
    KernelFn { fn: KernelFn },             // RKHS via on-the-fly kernel eval
}

pub enum BaseTransform {
    Squared,                               // ⟨v, v⟩_G
    Rooted,                                // √⟨v, v⟩_G  (metric form)
    Arccos,                                // arccos(⟨x,y⟩/(‖x‖‖y‖)) for angular distance
    OneMinus,                              // 1 - ⟨x,y⟩/(‖x‖‖y‖)
}

pub enum NormalizationMode {
    Raw,                                   // no rescaling of x, y
    UnitSphere,                            // x ← x/‖x‖₂
    ZeroMean,                              // x ← x - mean(x)  (correlation distance)
    ZeroMeanUnitVariance,                  // standardize per-vector
}
```

### Literature-named leaves

| Name | gram | base | normalize |
|---|---|---|---|
| `euclidean_distance` | Identity | Rooted | Raw |
| `squared_euclidean` | Identity | Squared | Raw |
| `mahalanobis_distance` | FullPSD(Σ⁻¹) | Rooted | Raw |
| `cosine_distance` | Identity | OneMinus | UnitSphere |
| `cosine_similarity` (negated) | Identity | OneMinus (inverted) | UnitSphere |
| `angular_distance` | Identity | Arccos | UnitSphere |
| `correlation_distance` | Identity | OneMinus | ZeroMeanUnitVariance |
| `pearson_distance` | Identity | OneMinus | ZeroMean |
| `weighted_mahalanobis` | FullPSD(W) | Rooted | Raw |
| `kernel_distance(K)` | KernelFn(K) | Rooted | Raw |
| `RBF_kernel_distance(γ)` | KernelFn(RBF{γ}) | Rooted | Raw |
| `polynomial_kernel_distance(d, c)` | KernelFn(Poly{d,c}) | Rooted | Raw |
| `dot_product` (raw) | Identity | identity-of-⟨·,·⟩ | Raw |
| `bregman_mahalanobis` | FullPSD(H) | Squared | Raw |

### Gaps the literature has not named (anti-YAGNI candidates)

- `mahalanobis_unitsphere` (FullPSD × Rooted × UnitSphere) — angular-distance under arbitrary metric
- `correlation_distance_with_full_gram` — correlation under non-diagonal precision matrix
- `kernel_with_zero_mean_normalization` — centered kernel distance
- `arccos_with_full_gram` — generalized-angular under metric
- `squared_kernel_distance` — for Bregman-divergence-like uses of an RKHS

### Accumulate + gather decomposition

```
InnerProductDistance<G, B, N>(x, y):
  let (xn, yn) = gather((x, y), normalize: N)        // optional rescaling
  let inner_xx = accumulate(xn, expr: xn(i) * G(i,j) * xn(j), op: Add)   // for full-PSD
  let inner_yy = accumulate(yn, expr: yn(i) * G(i,j) * yn(j), op: Add)
  let inner_xy = accumulate((xn, yn), expr: xn(i) * G(i,j) * yn(j), op: Add)
  let d2 = inner_xx - 2·inner_xy + inner_yy           // = ⟨x-y, x-y⟩_G
  return B(d2 or inner_xy depending on form)
```

For diagonal Gram or identity-Gram, the three accumulators collapse to single-pass per-vector:
```
  let xx = accumulate(xn, expr: w(i)·xn(i)², op: Add)
  let xy = accumulate((xn, yn), expr: w(i)·xn(i)·yn(i), op: Add)
  let yy = accumulate(yn, expr: w(i)·yn(i)², op: Add)
  return B(xx - 2·xy + yy)
```

### Sharing opportunities via TamSession

- **The 3-field "moment trio" `{‖x‖², ⟨x,y⟩, ‖y‖²}`** is universal across every inner-product geometry per `project_poincare_kernel.md` — once cached for `(x, y, G)`, every distance/similarity in this kernel (Euclidean, cosine, angular, correlation, Mahalanobis, kernel-distance) reuses it.
- **Cross-tree sharing**: the same moment trio feeds Kernel 1's `squared_euclidean` (when `G = Identity`), the means tree's `quadratic_mean`, and the correlations tree's `pearson_r`.
- **The Gram matrix `G` itself** is shareable across distance calls on different `(x, y)` pairs from the same dataset — typically the precomputed Σ⁻¹ for Mahalanobis or the kernel-evaluation matrix for RKHS.

### Overlap with Kernel 1

`euclidean_distance` and `squared_euclidean` are reachable from BOTH MinkowskiNorm (via `p=2`) and InnerProductDistance (via `Identity` Gram + Rooted/Squared base). The math is identical; the natural axes differ.

**Implementation recommendation**: `MinkowskiNorm` is the primary kernel for `euclidean_distance` because it shares the weight/filter axes with the rest of the Lp family. `InnerProductDistance` becomes the primary kernel for everything that needs a Gram form (Mahalanobis, cosine, kernel-distance). The named-leaf wrappers route through whichever kernel preserves the most reusable axes for that name.

---

## Kernel 3 — SetDistance `S<T, N>`

Distances on finite sets, multisets, or binary vectors. State shape is fundamentally different from coordinate vectors — element membership, not magnitude. **Disjoint from all other kernels at the state-shape level**.

```
d(A, B; T, N) = N⁻¹( g(|T(A)|, |T(B)|, |T(A ∩ B)|, |T(A ∪ B)|, |T(A △ B)|) )
```

where `g` is the family-specific formula and `T` is an optional set transform (multiset-collapse, weighting, hashing).

### Parameter axes

```rust
pub struct SetDistance<T, N> {
    pub transform: T,             // SetTransform
    pub normalize: N,             // SetNormalization
    pub formula: SetFormula,      // pick the leaf form
}

pub enum SetTransform {
    Identity,                              // raw set/multiset
    MultisetCollapse,                      // dedupe to set
    HashSet { hash_fn: HashFn },           // approximate via MinHash / SimHash
    Weighted { weights: HashMap<E, f64> },  // weighted set elements
    Shingled { k: usize },                 // k-shingles of a sequence (string → set)
}

pub enum SetNormalization {
    Unnormalized,                          // raw count
    UnionNormalized,                       // /|A ∪ B|
    MinSizeNormalized,                     // /min(|A|, |B|)
    GeometricMeanNormalized,               // /√(|A|·|B|)
    ArithmeticMeanNormalized,              // /((|A|+|B|)/2)
}

pub enum SetFormula {
    Jaccard,                               // 1 - |A∩B|/|A∪B|
    Dice,                                  // 1 - 2|A∩B|/(|A|+|B|)
    Tversky { alpha: f64, beta: f64 },     // |A∩B|/(|A∩B|+α|A\B|+β|B\A|)
    Hamming,                               // |A △ B|
    Tanimoto,                              // same as Jaccard for binary; weighted differently for multiset
    Ochiai,                                // |A∩B|/√(|A|·|B|)
    Overlap,                               // |A∩B|/min(|A|,|B|)
    Kulczynski,                            // (|A∩B|/|A| + |A∩B|/|B|) / 2  (or inverted)
    SimpleMatchingCoefficient,             // for binary vectors with negatives counted
    Russell_Rao,                           // |A∩B| / N
}
```

### Literature-named leaves

| Name | transform | normalize | formula |
|---|---|---|---|
| `jaccard_distance` | Identity | UnionNormalized | Jaccard |
| `weighted_jaccard` | Weighted | UnionNormalized | Jaccard |
| `sorensen_dice_distance` | Identity | ArithmeticMeanNormalized | Dice |
| `tversky_index(α, β)` | Identity | (formula-internal) | Tversky{α,β} |
| `hamming_distance` (binary vectors) | Identity | Unnormalized | Hamming |
| `normalized_hamming` | Identity | (length-normalized) | Hamming |
| `tanimoto_coefficient` | Identity | UnionNormalized | Tanimoto |
| `ochiai_coefficient` | Identity | GeometricMeanNormalized | Ochiai |
| `overlap_coefficient` | Identity | MinSizeNormalized | Overlap |
| `kulczynski_dissimilarity` | Identity | (formula-internal) | Kulczynski |
| `simple_matching_distance` | Identity | (length-normalized) | SimpleMatchingCoefficient |
| `minhash_jaccard` (approximate) | HashSet(MinHash) | UnionNormalized | Jaccard |
| `simhash_distance` | HashSet(SimHash) | (hamming-on-fingerprint) | Hamming |
| `k_shingle_jaccard` | Shingled{k} | UnionNormalized | Jaccard |

### Gaps the literature has not named (anti-YAGNI candidates)

- `weighted_dice` (Weighted × ArithmeticMeanNormalized × Dice)
- `tversky_with_minhash_approx`
- `ochiai_with_weighted_sets`
- `multi-set Hamming with k-shingles` for sequence-similarity preprocessing
- Arbitrary `Tversky(α, β)` choices outside the literature defaults

### Accumulate + gather decomposition

```
SetDistance<T, N, F>(A, B):
  let (At, Bt) = gather((A, B), transform: T)
  let counts = accumulate(
    (At, Bt),
    grouping: GroupByElement,
    expr: |At ∩ {e}|, |Bt ∩ {e}|,
    op: PairSum,
  )
  let intersection_size = accumulate(counts, expr: min(a_e, b_e) where both > 0, op: Add)
  let union_size = accumulate(counts, expr: max(a_e, b_e), op: Add)
  let a_size = accumulate(counts, expr: a_e, op: Add)
  let b_size = accumulate(counts, expr: b_e, op: Add)
  return F.formula(intersection_size, union_size, a_size, b_size).normalize(N)
```

For binary vectors:
```
  let counts_11 = accumulate(zip(A, B), expr: A(i) & B(i), op: Add)
  let counts_10 = accumulate(zip(A, B), expr: A(i) & !B(i), op: Add)
  let counts_01 = accumulate(zip(A, B), expr: !A(i) & B(i), op: Add)
  let counts_00 = accumulate(zip(A, B), expr: !A(i) & !B(i), op: Add)
  return F.formula(counts_11, counts_10, counts_01, counts_00).normalize(N)
```

### Sharing opportunities via TamSession

- **The 4-count fingerprint `{|A|, |B|, |A∩B|, |A∪B|}`**: once cached for `(A, B, T)`, every SetFormula reuses it. Jaccard, Dice, Tversky, Tanimoto, Ochiai, Overlap, Kulczynski — all are different *formulas* over the same four counts.
- **MinHash signatures**: shareable across all Jaccard-approximation queries on the same hash family.
- **Element-count histograms**: shareable across multiset and weighted-set distances on the same `T`-transformed input.

---

## Kernel 4 — Divergence `D<F, S>`

Statistical "distances" between probability distributions. Most are NOT metrics (fail symmetry or triangle inequality) but they're called "distances" in the literature, so they live here.

```
D(p ‖ q; F, S) = S( ∫ F(p(x), q(x)) dx )
```

where `F` is the family-defining functional form (KL, χ², f-divergence, optimal transport, kernel-MMD) and `S` is an optional symmetrization wrapper.

### Parameter axes

```rust
pub struct Divergence<F, S> {
    pub form: F,                  // DivergenceForm
    pub symmetrize: S,             // SymmetrizationMode
    pub support: SupportMode,      // discrete vs continuous; how to handle p > 0 but q = 0
}

pub enum DivergenceForm {
    Kl,                                              // ∫ p log(p/q)
    FDivergence { f: FDivFn },                       // ∫ q · f(p/q); KL, χ², Hellinger², TV all here
    Renyi { alpha: f64 },                            // (1/(α-1)) log ∫ p^α q^(1-α)
    Bregman { potential: BregmanPotential },         // φ(p) - φ(q) - ⟨∇φ(q), p - q⟩
    AlphaDivergence { alpha: f64 },                  // generalized f-divergence with α-family
    ItakuraSaito,                                    // ∫ (p/q - log(p/q) - 1)
    Wasserstein { p: f64 },                          // (inf_γ ∫ |x-y|^p dγ)^(1/p)
    SinkhornEntropic { eps: f64, p: f64 },           // entropy-regularized Wasserstein
    MMD { kernel: KernelFn },                        // E_{x,x'} k(x,x') - 2 E_{x,y} k(x,y) + E_{y,y'} k(y,y')
    Energy,                                          // 2 E|X-Y| - E|X-X'| - E|Y-Y'|
    Cramer,                                          // L2 of CDF difference (= Wasserstein-2 in 1D up to sqrt)
}

pub enum FDivFn {
    Kl,           // f(t) = t log t
    ReverseKl,    // f(t) = -log t
    ChiSquared,   // f(t) = (t-1)²
    Hellinger,    // f(t) = (√t - 1)²
    TotalVariation,  // f(t) = (1/2)|t-1|
    JensenShannon,   // symmetric mix
    LeCam,           // f(t) = (1-t)²/(2(1+t))
    Triangular,      // f(t) = (t-1)²/(t+1)
    Custom(Box<dyn Fn(f64) -> f64>),
}

pub enum SymmetrizationMode {
    None,
    Mean,                          // (D(p‖q) + D(q‖p)) / 2
    JsStyle,                       // D(p‖m) + D(q‖m), m = (p+q)/2
    MaxOfPair,                     // max(D(p‖q), D(q‖p))
    SquareRoot,                    // for Hellinger, Wasserstein
}

pub enum SupportMode {
    Discrete,                                        // p, q are pmf vectors
    Continuous { quadrature: QuadratureRule },       // p, q are pdfs
    EmpiricalSamples,                                // {x_i} ~ p, {y_j} ~ q
    EmpiricalCdf,                                    // F_p, F_q via sketches
}
```

### Literature-named leaves

| Name | form | symmetrize | support |
|---|---|---|---|
| `kl_divergence` | Kl | None | Discrete or Continuous |
| `reverse_kl` | FDivergence{ReverseKl} | None | any |
| `js_divergence` | FDivergence{JensenShannon} | JsStyle | any |
| `hellinger_distance` | FDivergence{Hellinger} | SquareRoot | any |
| `chi_squared_divergence` | FDivergence{ChiSquared} | None | Discrete |
| `total_variation_distance` | FDivergence{TotalVariation} | None | any |
| `renyi_divergence(α)` | Renyi{α} | None | any |
| `f_divergence(custom_f)` | FDivergence{Custom} | None | any |
| `bregman_divergence(φ)` | Bregman{φ} | None | any |
| `alpha_divergence(α)` | AlphaDivergence{α} | None | any |
| `itakura_saito` | ItakuraSaito | None | Discrete (audio) |
| `wasserstein_1` (EMD) | Wasserstein{1} | None | EmpiricalSamples |
| `wasserstein_2` | Wasserstein{2} | None | EmpiricalSamples |
| `wasserstein_p(p)` | Wasserstein{p} | None | EmpiricalSamples |
| `sinkhorn_distance` | SinkhornEntropic{ε, 2} | None | EmpiricalSamples |
| `cramer_distance` | Cramer | None | EmpiricalCdf |
| `energy_distance` | Energy | None | EmpiricalSamples |
| `mmd_gaussian(γ)` | MMD{RBF(γ)} | None | EmpiricalSamples |
| `mmd_polynomial(d)` | MMD{Poly(d)} | None | EmpiricalSamples |

### Gaps the literature has not named

- `mean-symmetric KL` over discrete distributions — used in practice but rarely named
- `wasserstein_p` for `p ∈ ℝ⁺ \ {1, 2, ∞}` — fractional Wasserstein
- `renyi-symmetrized` (Mean wrapper on Rényi)
- `bregman-mahalanobis` with custom PSD potential — connects to Kernel 2
- `sinkhorn with α-divergence-cost` instead of L2-cost
- `mmd with mixture kernel`

### Accumulate + gather decomposition

For discrete f-divergences:
```
Divergence<FDivergence(f), None, Discrete>(p, q):
  return accumulate(zip(p, q), expr: q(i) * f(p(i)/q(i)), op: Add)
```

For empirical Wasserstein-1 in 1D (closed-form via sorted-CDF):
```
Wasserstein<1>(samples_p, samples_q):
  let sp = gather(samples_p, addressing: SortAscending)
  let sq = gather(samples_q, addressing: SortAscending)
  return accumulate(zip(sp, sq), expr: |sp(i) - sq(i)|, op: Add) / n
```

For MMD:
```
MMD<kernel k>(samples_p, samples_q):
  let pp = accumulate(pairs(samples_p), expr: k(x, x'), op: Add) / n²
  let qq = accumulate(pairs(samples_q), expr: k(y, y'), op: Add) / m²
  let pq = accumulate(cross(samples_p, samples_q), expr: k(x, y), op: Add) / (n·m)
  return pp - 2·pq + qq
```

For Wasserstein-p in higher dimensions: **Kingdom B/C** — Sinkhorn iteration, network simplex, or entropic regularization. Honestly declare the kingdom.

### Sharing opportunities via TamSession

- **Sorted empirical samples**: shareable across `wasserstein_p` for all `p`, `cramer_distance`, all CDF-based divergences.
- **Pairwise kernel matrices `k(xᵢ, xⱼ)`, `k(xᵢ, yⱼ)`, `k(yᵢ, yⱼ)`**: shareable across all MMD variants on the same `(samples, kernel)` triple.
- **`p log p` per-bin accumulators**: shareable across KL and entropy (cross-tree to the information-theoretic tree).
- **Bregman potentials**: shareable across multiple Bregman-divergence calls on the same potential `φ`.

### Overlap with Kernel 1 (MinkowskiNorm)

The **Wasserstein-p of empirical distributions in 1D** reduces to `Lp(sort(x) - sort(y)) / n` — provably equal to a MinkowskiNorm computation on sorted samples. **In 1D, Wasserstein is a parameter assignment on MinkowskiNorm with `SortAscending` gather.**

In higher dimensions, no such reduction exists — Wasserstein requires solving a linear assignment / transport problem. The kernels diverge at d ≥ 2.

**Implementation recommendation**: `wasserstein_1` in 1D should be a recipe wrapper that detects dimensionality and dispatches: `1D → MinkowskiNorm` (fast closed-form), `d ≥ 2 → Divergence{Wasserstein}` (iterative solver). The kernel boundary shifts at d=1 to d=2.

---

## Kernel 5 — SequenceEdit `E<C, A, N>`

Distances on strings / token sequences / time-aligned signals. The state is a 2D alignment lattice; cost is the minimum-cost path. **Disjoint from all coordinate-based kernels** because the operation is alignment, not magnitude.

```
d(s, t; C, A, N) = N⁻¹( min_{alignment α} Σ_{op ∈ α} C(op) )
```

### Parameter axes

```rust
pub struct SequenceEdit<C, A, N> {
    pub cost: C,                  // CostMatrix
    pub alignment: A,             // AlignmentMode
    pub normalize: N,             // EditNormalization
}

pub enum CostMatrix {
    UnitCost,                              // all ops cost 1 (Levenshtein)
    SymmetricCustom { sub: f64, ins: f64, del: f64 },
    AsymmetricCustom { sub_table: Mat<f64>, ins: Vec<f64>, del: Vec<f64> },
    Biological { matrix: BioMatrix },      // BLOSUM, PAM
    Hamming,                               // sub-only, no insert/delete
    LcsCost,                               // 0 for match, 1 for insert/delete, ∞ for sub
}

pub enum AlignmentMode {
    Global,                                // Needleman-Wunsch (full)
    Local,                                 // Smith-Waterman (best local sub-alignment)
    SemiGlobal,                            // free start/end gaps (sequence-in-text)
    OverlapAlignment,                      // end-of-s ↔ start-of-t
    GapPenalty { affine: AffineGapPenalty }, // for biological-style affine gaps
    DamerauTranspositions,                 // adjacent-swap as a single op
}

pub enum EditNormalization {
    Unnormalized,
    LengthMax,                             // / max(|s|, |t|)
    LengthSum,                             // / (|s| + |t|)
    LengthGeometric,                       // / √(|s|·|t|)
    JaroStyle,                             // Jaro's matching-and-transposition formula
    JaroWinklerStyle { prefix_p: f64 },    // Jaro + prefix bonus
}
```

### Literature-named leaves

| Name | cost | alignment | normalize |
|---|---|---|---|
| `levenshtein_distance` | UnitCost | Global | Unnormalized |
| `normalized_levenshtein` | UnitCost | Global | LengthMax |
| `damerau_levenshtein` | UnitCost | DamerauTranspositions | Unnormalized |
| `optimal_string_alignment` (OSA) | UnitCost | DamerauTranspositions (restricted) | Unnormalized |
| `hamming_distance` (sequence) | Hamming | Global | Unnormalized |
| `longest_common_subsequence` | LcsCost | Global | Unnormalized |
| `lcs_distance` | LcsCost | Global | LengthSum |
| `jaro_similarity` | UnitCost | (custom matching-window) | JaroStyle |
| `jaro_winkler_similarity` | UnitCost | (custom matching-window) | JaroWinklerStyle{0.1} |
| `needleman_wunsch_score` | Biological{BLOSUM62} | Global | Unnormalized |
| `smith_waterman_score` | Biological{BLOSUM62} | Local | Unnormalized |
| `affine_gap_edit` | SymmetricCustom + Affine | Global+GapPenalty | Unnormalized |
| `edit_tree_distance` | UnitCost | (tree-mode) | Unnormalized |

### Gaps the literature has not named

- `local_damerau_levenshtein` — Smith-Waterman with transpositions
- `weighted_lcs` with custom substitution cost (lifting LCS toward Needleman-Wunsch)
- `jaro_winkler_with_affine_gaps`
- Custom biological matrices outside BLOSUM/PAM (domain-specific)
- `unicode_grapheme_levenshtein` — operations on grapheme clusters rather than codepoints

### Accumulate + gather decomposition

```
SequenceEdit<C, A, N>(s, t):
  let dp = accumulate(
    rectangle(|s|, |t|),
    grouping: DiagonalWavefront,           // alignment-lattice diagonal scan
    expr: min(
      dp(i-1, j-1) + C.sub(s(i), t(j)),
      dp(i-1, j)   + C.del(s(i)),
      dp(i, j-1)   + C.ins(t(j)),
    ),
    op: MinUpdate,
  )
  return N⁻¹(dp(|s|, |t|))
```

**Kingdom B intrinsically** — every cell depends on three predecessors. Wavefront diagonalization parallelizes within-anti-diagonal but not across; TAM handles the wavefront scheduler.

For Hamming on equal-length strings, the alignment lattice collapses:
```
  return accumulate(zip(s, t), expr: s(i) != t(i), op: Add)
```
This is **Kingdom A** and is a degenerate fast-path.

### Sharing opportunities via TamSession

- **DP-table for `(s, t)` under cost C**: shareable across multiple alignment readout types (full alignment, score-only, traceback) on the same input.
- **Suffix arrays / suffix automata of `s`**: shareable across edit-distance queries with `s` fixed and varying `t`.
- **Biological cost matrices (BLOSUM, PAM)**: globally shareable; precomputed once per matrix family.

---

## Kernel 6 — GraphDistance `G<W, T, S>`

Distances on graphs — either *between nodes* in a graph (shortest path, commute time) or *between graphs* (graph edit, spectral). The state is an adjacency structure plus walk semantics. **Mostly disjoint** from coordinate kernels; **partially overlapping** with MinkowskiNorm through spectral embedding.

```
d(u, v; W, T, S)  [node-node case]
d(G₁, G₂; W, T, S)  [graph-graph case]
```

### Parameter axes

```rust
pub struct GraphDistance<W, T, S> {
    pub edge_weight: W,           // EdgeWeightMode
    pub traversal: T,             // TraversalMode
    pub summary: S,               // SummaryMode (for graph-graph)
}

pub enum EdgeWeightMode {
    Unit,                                  // unweighted graph
    Weighted { weights: Vec<f64> },        // explicit edge weights
    InverseDegree,                         // for random walks
    Heat { time: f64 },                    // e^(-tL) for diffusion distance
    Inverse,                               // 1/w for resistance distance
}

pub enum TraversalMode {
    ShortestPath { algorithm: ShortestPathAlgo },
    RandomWalk { steps: usize, restart: f64 },
    CommuteTime,                                     // expected meeting time
    Resistance,                                      // electrical-network analog
    Diffusion { t: f64 },                            // heat-kernel distance
    Spectral { laplacian: LaplacianMode, k: usize }, // embed via top-k eigenvectors
    WeisfeilerLehman { iterations: usize },          // for graph-graph isomorphism approx
    GraphEdit { node_cost: f64, edge_cost: f64 },    // align + edit on graphs
}

pub enum ShortestPathAlgo {
    Dijkstra,
    BellmanFord,
    FloydWarshall,                                   // all-pairs
    AStar { heuristic: HeuristicFn },
    Bidirectional,
}

pub enum LaplacianMode {
    Combinatorial,                                   // L = D - A
    Normalized,                                      // L = I - D^(-1/2) A D^(-1/2)
    RandomWalk,                                      // L = I - D^(-1) A
    Signless,                                        // L = D + A
}

pub enum SummaryMode {
    NodeFingerprint { hash: HashFn },                // per-node summary, then aggregate
    SpectralSorted,                                  // sorted eigenvalues
    GraphletDistribution,                            // count small subgraphs
    DegreeDistribution,
    EditOptimum,                                     // for graph edit distance
}
```

### Literature-named leaves

| Name | edge_weight | traversal | summary |
|---|---|---|---|
| `dijkstra_distance(u, v)` | Weighted | ShortestPath{Dijkstra} | n/a |
| `bellman_ford_distance` | Weighted | ShortestPath{BellmanFord} | n/a |
| `floyd_warshall_all_pairs` | Weighted | ShortestPath{FloydWarshall} | n/a |
| `astar_distance(u, v, h)` | Weighted | ShortestPath{AStar{h}} | n/a |
| `unweighted_shortest_path` (BFS) | Unit | ShortestPath{Dijkstra (degenerate to BFS)} | n/a |
| `commute_time_distance` | Weighted | CommuteTime | n/a |
| `resistance_distance` | Inverse | Resistance | n/a |
| `diffusion_distance(t)` | Heat{t} | Diffusion{t} | n/a |
| `spectral_distance(k)` | n/a | Spectral{Normalized, k} | SpectralSorted |
| `random_walk_distance(steps)` | InverseDegree | RandomWalk{steps, 0} | n/a |
| `personalized_pagerank_distance` | InverseDegree | RandomWalk{∞, α} | n/a |
| `weisfeiler_lehman_distance(k)` | Unit | WeisfeilerLehman{k} | NodeFingerprint |
| `graph_edit_distance` | Weighted | GraphEdit{1, 1} | EditOptimum |
| `graphlet_kernel_distance` | Unit | n/a | GraphletDistribution |
| `degree_sequence_distance` | n/a | n/a | DegreeDistribution |

### Gaps the literature has not named

- `astar_with_diffusion_heuristic` — heuristic from heat-kernel admissibility
- `restart_random_walk_with_heat_weights`
- `spectral_with_signless_laplacian` (rare but defensible for bipartite-like graphs)
- `commute-time over weighted+restart walks`
- `weisfeiler_lehman_with_edge_features`

### Accumulate + gather decomposition

For Dijkstra on a single source:
```
GraphDistance<W, Dijkstra>(graph, source):
  let dist = accumulate(
    nodes(graph),
    grouping: PriorityQueueExtractMin,    // Kingdom B intrinsically
    expr: min(dist(u), dist(u_prev) + w(u_prev, u)),
    op: MinRelax,
  )
  return dist
```

For Floyd-Warshall (all-pairs):
```
GraphDistance<W, FloydWarshall>(graph):
  let dist = accumulate(
    triples(nodes, nodes, nodes),
    grouping: KIntermediateScan,           // three-nested loop
    expr: min(dist(i, j), dist(i, k) + dist(k, j)),
    op: MinUpdate,
  )
  return dist
```

For spectral distance:
```
GraphDistance<W, Spectral{L, k}>(graph):
  let L = laplacian(graph, L_mode)
  let (eigvals, eigvecs) = eigendecomposition(L, top_k: k)
  let embedding = gather(nodes, expr: eigvecs(node, :k))
  return embedding   // delegate to MinkowskiNorm on rows
```

This last is where the **partial overlap with MinkowskiNorm** lives: graph → spectral embedding → coordinate vectors → Lp distance. The lift is parameter-dependent (which Laplacian, how many eigenvectors), so the kernels are not interchangeable, but `spectral_distance(k)` is a recipe wrapper that delegates to `MinkowskiNorm<2>` after the embedding.

**Kingdom B/C** throughout for shortest-path, walk-based, and iterative spectral methods. TAM scheduler handles the boundary.

### Sharing opportunities via TamSession

- **Adjacency / Laplacian matrices**: shareable across all GraphDistance queries on the same graph.
- **All-pairs shortest paths**: a single Floyd-Warshall computation feeds N² queries.
- **Top-k eigendecomposition of L**: shareable across spectral, diffusion, and commute-time distances.
- **Heat kernel `e^(-tL)`**: shareable across all diffusion-related distances at the same `t`.

---

## Cross-kernel structural map

```
                          distance family
                                |
        +---------------+-------+-------+----------------+
        |               |               |                |
   GEOMETRIC CLUSTER                     |    COMBINATORIAL CLUSTER
        |               |               |                |
   MinkowskiNorm    InnerProductDistance |   SetDistance  SequenceEdit  GraphDistance
   M<p, W, F>       I<G, B, N>           |   S<T, N>      E<C, A, N>    G<W, T, S>
        |               |                |       |            |              |
        |   +-----------+                |       |            |              |
        |   | overlap @ L2/Euclidean     |       |            |              |
        |   | (Mahalanobis diagonal also)|       |            |              |
        +---+                            |       |            |              |
        |                                |       |            |              |
        |        Divergence              |       |            |              |
        |        D<F, S>                 |       |            |              |
        |          |                     |       |            |              |
        +-- overlap @ Wasserstein-p in 1D|       |            |              |
            (sort + Lp)                  |       |            |              |
                                         |       |            |              |
                                         |       |            |              |
                          spectral-embedding lift (partial overlap)
                          GraphDistance → MinkowskiNorm via eigvec coords

   euclidean_distance ─── reachable from MinkowskiNorm(p=2) OR InnerProductDistance(Identity, Rooted, Raw)
   squared_euclidean  ─── reachable from MinkowskiNorm(p=2, drop √) OR InnerProductDistance(Identity, Squared, Raw)
   mahalanobis_diag   ─── reachable from MinkowskiNorm(p=2, w=1/σ²) OR InnerProductDistance(Diagonal, Rooted, Raw)
   wasserstein_1 (1D) ─── reachable from MinkowskiNorm(p=1) on sorted samples OR Divergence(Wasserstein{1})
   hellinger_distance ─── reachable from MinkowskiNorm(p=2) after √-transform OR Divergence(FDiv{Hellinger})

   [no overlap zone]
   SetDistance ←→ SequenceEdit ←→ GraphDistance
       (these three are pairwise disjoint at state-shape; nothing reduces between them)

                                                   [unnamed gaps clustered per-kernel]
```

The topology is **two overlap-clusters plus three isolated kernels**:
- *Geometric overlap-cluster*: {MinkowskiNorm, InnerProductDistance, Divergence (partially)} — share moment-statistics infrastructure, Lp-of-coords as substrate
- *Combinatorial isolated*: {SetDistance, SequenceEdit, GraphDistance} — each carries its own state-shape and accumulate decomposition

The bridge from Combinatorial to Geometric is **always an embedding** (spectral lift, MinHash → vector, k-shingle hashing), and the embedding is *parameter-dependent*. Embeddings are not part of any kernel; they're recipes that compose two kernels.

---

## Open questions for math-researcher walk-through

1. **Wasserstein-p as MinkowskiNorm wrapper in 1D vs Divergence kernel.** In 1D, `wasserstein_p` reduces to `Lp(sort(x) - sort(y))` and could be a thin recipe wrapper on MinkowskiNorm. In `d ≥ 2`, it needs the iterative-transport machinery in the Divergence kernel. Should `wasserstein_p` be a *single recipe* that dispatches by dimensionality, or *two recipes* (`wasserstein_p_1d` and `wasserstein_p_nd`) with the user choosing? My read: single recipe with dimensionality dispatch is cleaner, but the kingdom-boundary changes (A→C) which TAM cares about.

2. **The MinkowskiNorm/InnerProductDistance overlap split.** Both can compute Euclidean. Mahalanobis-diagonal is reachable from both. Mahalanobis-full is only in InnerProductDistance. Is the split where I drew it (MinkowskiNorm primary for diagonal-and-simpler; InnerProductDistance primary for Gram-form-required) the right place, or should the split go elsewhere — e.g., MinkowskiNorm covers everything diagonal-weighted, InnerProductDistance covers everything that needs cross-coordinate coupling?

3. **Bregman divergences span Divergence and InnerProductDistance.** A Bregman divergence with quadratic potential `φ(x) = x' H x / 2` reduces to `(x-y)' H (x-y)` = squared Mahalanobis. So Bregman lives in BOTH Divergence (general Bregman) and InnerProductDistance (quadratic-potential special case). Is there a sharper formulation that puts Bregman cleanly in one kernel with the other as a recipe wrapper?

4. **Is `correlation_distance` really an InnerProductDistance leaf, or does it belong in the correlations tree?** Pearson correlation is the cosine of mean-centered vectors — structurally `InnerProductDistance(Identity, OneMinus, ZeroMean)`. But the correlations tree (TBD) will own Pearson too. Cross-tree placement: correlations primary, distances secondary recipe-wrapper? Or vice versa?

5. **MMD with mixture kernels.** The Divergence kernel currently treats `MMD{kernel}` as one slot with one kernel function. Most modern MMD work uses *mixture kernels* (Σ_h α_h k_h). Should `kernel` itself become a parametric slot — `KernelFn::Mixture { kernels: Vec<KernelFn>, weights: Vec<f64> }` — or stay simple and let mixture-MMD be a recipe wrapper that runs MMD multiple times and combines?

6. **The PairwiseTopology meta-axis.** I named it but didn't make it a parameter on any kernel. Every distance call has the choice "single-pair / full-matrix / k-NN graph / ε-ball graph." Where does this live — as a `using()` knob on every distance call, as a separate kernel `PairwiseDispatcher<dist>`, or as an entirely separate concern in the IR/TAM layer? My intuition: TAM-layer concern, because it's about *what gather pattern wraps the accumulate*, not about the math. But it's worth ratifying.

---

## Implementation roadmap

This doc is the catalog tree. **Implementation is downstream and not blocked by this doc.** Ordering hint when implementation lands:

1. **MinkowskiNorm first** — covers the most named leaves with the cleanest accumulate+gather decomposition. Shares vocabulary with `GeneralizedMean`; same `WeightFn` / `FilterFn` enums can be reused.
2. **InnerProductDistance second** — needed for cosine + Mahalanobis + RKHS-based distances. Shares the 3-field moment trio infrastructure across the means and correlations trees via TamSession.
3. **SetDistance third** — independent state-shape, isolated implementation, broad coverage with one kernel.
4. **Divergence fourth** — biggest kernel by scope but compartmentalizes well (f-divergences via shared `f` form; Wasserstein with kingdom-honest declaration; MMD via kernel substitution). Build in stages by DivergenceForm.
5. **SequenceEdit fifth** — Kingdom B intrinsically; wavefront scheduler from TAM needs to be in place. Hamming-special-case fast-path first; full DP-lattice second.
6. **GraphDistance sixth** — Kingdom B/C throughout; depends on graph data structure conventions being settled. Shortest-path family before walk-based before spectral.

Cross-kernel sharing fires through TamSession with compatibility tags per Tambear Contract item 3. The two big shared structures:
- **Moment trio `{‖x‖², ⟨x,y⟩, ‖y‖²}`** — shared across MinkowskiNorm-L2, InnerProductDistance, the means tree's `quadratic_mean`, and the correlations tree's `pearson_r`.
- **Sorted empirical samples** — shared across Wasserstein-1D, Cramér distance, MinkowskiNorm on sorted vectors, KS statistic (correlations tree).

---

## Threads downstream of this tree

- **Means tree** (`recipe-trees/means.md`): `FrechetMean<D>` consumes a distance metric. Every distance in this tree is a candidate `D` for a Fréchet mean. The means tree's `FrechetMean(Euclidean) ≡ arithmetic_mean` is the canonical bridge. Other bridges: `FrechetMean(Manhattan) = geometric_median`, `FrechetMean(Mahalanobis) = whitened-Euclidean centroid`, `FrechetMean(Riemannian) = Karcher mean`.

- **Sketches tree** (`recipe-trees/sketches.md`): empirical-distribution distances (Wasserstein, Cramér, KS) compose with quantile sketches. Streaming-Wasserstein-1D = `CompressedHistogram → sorted-CDF → MinkowskiNorm(p=1)`. The Divergence kernel's `SupportMode::EmpiricalCdf` is exactly this composition.

- **Correlations tree** (TBD `recipe-trees/correlations.md`): `correlation_distance` = `1 - pearson_r`; cosine/angular distance shares vocabulary with the correlation-as-cosine framing. The two trees overlap at the seam between "distance from 1" and "correlation coefficient" — same math, different naming convention.

- **Clustering tree** (TBD `recipe-trees/clustering.md`): every clustering recipe (k-means, DBSCAN, hierarchical, spectral) takes a distance as a `using()` knob. This tree is the universe of valid choices. Sharing fires when multiple clustering recipes run on the same dataset — one distance matrix feeds all of them through TamSession.

- **Dimensionality-reduction tree** (TBD): MDS, Isomap, t-SNE, UMAP all take distance/dissimilarity matrices. Same sharing story as clustering.

- **Information-theoretic tree** (TBD `recipe-trees/information.md`): Divergence kernel here is a *subset* of the information-theoretic tree's scope. KL/JS/Rényi/MMD live in both; the distances tree treats them as distance-family leaves, the info-theoretic tree treats them as divergence-family leaves. Cross-tree linking, single canonical implementation.

- **Kernels tree** (TBD `recipe-trees/kernels.md`): the `KernelFn` parameter in InnerProductDistance and Divergence::MMD points into the kernels tree. RBF, polynomial, Matérn, etc., are all parameter assignments on that family.

These are not assignments — they're invitations. The tree pattern propagates naturally as someone touches a family.
