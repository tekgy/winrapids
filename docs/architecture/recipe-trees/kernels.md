# Recipe tree — kernels (smoothing kernels + positive-definite kernels)

**Status**: Third pilot of the catalog-as-tree pattern (`recipe-trees/README.md`).

**Drafted**: 2026-05-10 by main-thread Claude. Awaiting math-researcher walk-through for the double-meaning split (is the disjoint-pair topology correct, or is there a deeper common kernel?); pathmaker walk-through for accumulate+gather decomposition validation; naturalist walk-through for the cross-tree implications (KDE kernels live inside `kernel_smoothed_mean` in `means.md`).

**Anchor**: Naturalist's `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — *"names are parameter assignments on a graph; the graph is the actual catalog."* This is the family where the name *"kernel"* itself is overloaded — the literature uses one word for two structurally distinct mathematical objects. The tree separates them.

---

## TL;DR — two top-level kernel-of-kernels (disjoint), seven sub-kernels, ~40 literature names

The "kernel" family in mathematics carries a **double meaning** in the literature. They share a name and (sometimes) a functional form, but they are structurally disjoint as catalog objects — they answer different questions, parameterize different axes, and compose into different recipes.

| Top-level kernel-of-kernels | Question it answers | Sub-kernels | Disjoint from |
|---|---|---|---|
| **(a) SmoothingKernel** `S<shape, bandwidth, support>` | "How should I weight neighbors by distance when estimating a local quantity?" | `RadialShape`, `WaveletShape`, `OrderedSupport` | positive-definite Mercer kernels |
| **(b) PositiveDefiniteKernel** `K<form, bandwidth, hyperparameters>` | "What inner product on a feature space encodes my similarity notion?" | `StationaryPD`, `NonStationaryPD`, `CompositionalPD`, `StructuredPD` | smoothing kernels (different intent, different constraints) |

**The disjoint topology is confirmed**, but with a structural asterisk: both families share a *functional-form vocabulary* (Gaussian, Laplace, exponential, polynomial...). The same `exp(-r²/2h²)` curve appears as the Gaussian KDE kernel (smoothing) AND as the RBF Mercer kernel (PD). They are *not* the same recipe — the smoothing-Gaussian is normalized over its support to integrate to 1 (a probability density weighting); the PD-Gaussian is a similarity function on ℝᵈ × ℝᵈ that must satisfy Mercer's positivity condition. Same curve, different mathematical commitments, different consumers.

This is the cleanest case so far of "naming collision in the literature obscures structural difference." The tree's job is to make the difference visible.

**Wavelet kernels** (Morlet, Mexican-hat, Haar, Daubechies, Coiflet, Symlet, biorthogonal): I place these under (a) SmoothingKernel as `WaveletShape`. The case for putting them in their own top-level kernel-of-kernels is real (wavelets carry an admissibility condition that smoothing kernels don't require, and they're the basis of multi-resolution analysis, not just weighting); see open question #3.

**String kernels** (spectrum, mismatch, gappy) and **graph kernels** (diffusion, random-walk, Weisfeiler-Lehman): these are *positive-definite kernels on structured-input domains*, so they belong inside (b) PositiveDefiniteKernel as `StructuredPD`. The literature sometimes treats them as their own family because the inputs (strings, graphs) aren't ℝᵈ, but the Mercer-positivity property and consumer-pattern (SVM, GP, kernel-PCA) are identical.

**Convolution kernels** (signal-processing FIR filters, image-processing 2D filters): deliberately scoped to a future `signal-processing.md` tree. They're the *same functional forms* used in (a), but the consumer pattern is different (apply the kernel via FFT or direct convolution to *transform* a signal, not to *weight neighbors* for density estimation).

---

## Top-level (a) — SmoothingKernel `S<shape, bandwidth, support>`

The KDE family. Every literature-named smoothing kernel is a `(shape, bandwidth, support)` assignment on this kernel. Consumed by: `kernel_density_estimate`, `kernel_smoothed_mean` (in `means.md`), `nadaraya_watson_regression`, `local_polynomial_regression`, `kernel_regression`, kernel-weighted variants of tail estimators.

### Parameter axes

```rust
pub struct SmoothingKernel<Sh, Bw, Su> {
    pub shape: Sh,         // ShapeFn — how weight decays with normalized distance
    pub bandwidth: Bw,     // BandwidthRule — scale parameter for the kernel
    pub support: Su,       // Support — where the kernel is non-zero
}

pub enum ShapeFn {
    Gaussian,                              // K(u) = (2π)^(-1/2) · exp(-u²/2)
    Epanechnikov,                          // K(u) = 0.75 · (1 - u²)  for |u| ≤ 1
    Biweight,                              // K(u) = (15/16) · (1 - u²)²  for |u| ≤ 1
    Triweight,                             // K(u) = (35/32) · (1 - u²)³  for |u| ≤ 1
    Tricube,                               // K(u) = (70/81) · (1 - |u|³)³  for |u| ≤ 1
    Triangular,                            // K(u) = (1 - |u|)  for |u| ≤ 1
    Uniform,                               // K(u) = 0.5  for |u| ≤ 1   (box / rectangular)
    Cosine,                                // K(u) = (π/4) · cos(πu/2)  for |u| ≤ 1
    Logistic,                              // K(u) = 1 / (exp(u) + 2 + exp(-u))
    Exponential,                           // K(u) = 0.5 · exp(-|u|)
    Silverman,                             // K(u) = 0.5 · exp(-|u|/√2) · sin(|u|/√2 + π/4)
    Custom(Box<dyn Fn(f64) -> f64>),       // user-defined; will check unit-integral if Support is finite
}

pub enum BandwidthRule {
    Fixed(f64),                                            // h provided directly
    SilvermanRuleOfThumb,                                  // h = 1.06 · σ̂ · n^(-1/5)
    ScottsRule,                                            // h = 1.06 · σ̂ · n^(-1/(d+4))
    BiasedCrossValidation,                                 // BCV: minimize asymptotic MISE
    UnbiasedCrossValidation,                               // UCV: leave-one-out MISE
    PluginEstimator { order: u8 },                         // Sheather-Jones, AMISE, etc.
    BalloonAdaptive { k_neighbors: usize },                // h_i = distance to kth neighbor
    SampleSmoothing { alpha: f64, pilot: BandwidthRulePilot }, // Abramson's variable bandwidth
}

pub enum Support {
    Compact { radius: f64 },               // K(u) = 0 for |u| > radius (in u-units)
    Infinite,                              // K(u) > 0 everywhere (Gaussian, Logistic, Exponential)
}
```

### Sub-kernel 1 — RadialShape (point kernels)

These weight scalar/Euclidean distances. Most named KDE kernels live here.

| Name | shape | bandwidth | support |
|---|---|---|---|
| `gaussian_kernel` | Gaussian | Fixed(h) | Infinite |
| `epanechnikov_kernel` | Epanechnikov | Fixed(h) | Compact{1} |
| `biweight_kernel` (quartic) | Biweight | Fixed(h) | Compact{1} |
| `triweight_kernel` | Triweight | Fixed(h) | Compact{1} |
| `tricube_kernel` | Tricube | Fixed(h) | Compact{1} |
| `triangular_kernel` | Triangular | Fixed(h) | Compact{1} |
| `uniform_kernel` (box / rectangular) | Uniform | Fixed(h) | Compact{1} |
| `cosine_kernel` | Cosine | Fixed(h) | Compact{1} |
| `logistic_kernel` | Logistic | Fixed(h) | Infinite |
| `exponential_kernel` | Exponential | Fixed(h) | Infinite |
| `silverman_kernel` | Silverman | Fixed(h) | Infinite |
| `silverman_rule_kernel(K)` | any | SilvermanRuleOfThumb | shape's support |
| `scotts_kernel(K)` | any | ScottsRule | shape's support |
| `sheather_jones_kernel(K)` | any | PluginEstimator{order=2} | shape's support |
| `abramson_adaptive_kernel(K)` | any | SampleSmoothing{α=0.5} | shape's support |

**The `quartic_kernel` from the briefing is `biweight_kernel`** — same kernel, two literature names. Either name resolves to `shape: Biweight, bandwidth: Fixed(h), support: Compact{1}`.

### Sub-kernel 2 — WaveletShape (wavelet kernels)

Wavelets are smoothing kernels with **mother-wavelet ψ + scaling axis**. Distinct from RadialShape because they're parameterized by *(scale, translation)* pairs rather than a single bandwidth, and they carry an admissibility condition (`∫ ψ(u) du = 0`).

```rust
pub enum WaveletShape {
    Morlet { omega_0: f64 },               // complex sinusoid × Gaussian envelope
    MexicanHat,                            // -d²/du² Gaussian (Ricker wavelet)
    Haar,                                  // simplest, discontinuous
    Daubechies { order: u8 },              // db2, db4, db6, db8, ...
    Coiflet { order: u8 },                 // coif1, coif2, ...
    Symlet { order: u8 },                  // sym2, sym4, ... (least-asymmetric Daubechies)
    Biorthogonal { decomp: (u8,u8), recon: (u8,u8) },   // bior2.2, bior4.4, ...
    Meyer,
    Shannon,
}

pub struct WaveletKernel<W> {
    pub mother: W,
    pub scale: f64,            // a — dilation
    pub translation: f64,      // b — shift
}
```

| Name | mother | scale | translation |
|---|---|---|---|
| `morlet_wavelet` | Morlet{ω₀=5} | a | b |
| `ricker_wavelet` / `mexican_hat` | MexicanHat | a | b |
| `haar_wavelet` | Haar | a (dyadic) | b |
| `db4_wavelet` | Daubechies{4} | a (dyadic) | b |
| `db8_wavelet` | Daubechies{8} | a (dyadic) | b |
| `coif2_wavelet` | Coiflet{2} | a (dyadic) | b |
| `sym4_wavelet` | Symlet{4} | a (dyadic) | b |
| `bior2.2_wavelet` | Biorthogonal{(2,2),(2,2)} | a | b |

### Sub-kernel 3 — OrderedSupport (order-statistic kernels)

A small but real corner — kernels parameterized by **rank position** rather than value distance. The "kernel of an L-estimator" is a weighting function on order statistics. Includes Tukey's biweight as a robust-M-estimator weighting (separate from the biweight-as-KDE).

| Name | weighting | normalization |
|---|---|---|
| `tukey_biweight_psi(c)` | rank-based ψ-function | Compact{c} |
| `huber_psi(c)` | rank-bounded linear | Threshold{c} |
| `hampel_psi(a,b,c)` | three-part redescending | Compact{c} |

This sub-kernel exists to be honest about a structural distinction: Tukey's biweight as a robust-regression ψ-function (this sub-kernel) is *not* the same recipe as `biweight_kernel` (RadialShape, KDE) even though both involve `(1-u²)²`. Same expression, different role in the math. See open question #2.

### Gaps the literature has not named (anti-YAGNI candidates)

- `kernel_with_silverman_bandwidth_at_arbitrary_shape` — Silverman's rule is canonical with Gaussian, but the formula extends to any shape via the canonical-bandwidth-correction (Marron-Nolan).
- `epanechnikov_with_adaptive_bandwidth` (Epanechnikov + BalloonAdaptive) — rare in practice but reachable.
- `cosine_kernel_with_sheather_jones` — Sheather-Jones plug-in is canonically Gaussian; the rule generalizes.
- `custom_shape × any_bandwidth × any_support` triples — fully parameterizable.
- `quintic_kernel`, `sextic_kernel`, etc. — `(1-u²)^k` for `k > 3`. Triweight is k=3. Higher orders are obvious generalizations no literature has bothered to name.

Per anti-YAGNI: each combination is reachable through the kernel without per-name implementation.

### Accumulate + gather decomposition

For `kernel_density_estimate` at a query point `y` over a sample `x`:

```
SmoothingKernel<sh, bw, su> .at(y, x):
  let h = bw.resolve(x)                          // bandwidth selection (may itself be Kingdom B if iterative)
  let normalized_distances = gather(x, addressing: All, expr: (x(i) - y) / h)
  let kernel_values = gather(normalized_distances, addressing: All, expr: sh(u))
  let weight_sum = accumulate(kernel_values, expr: kernel_values(i), op: Add)
  return weight_sum / (n * h)                    // density estimate
```

For `nadaraya_watson_regression` (uses smoothing kernel as a weight in a weighted-mean):

```
NadarayaWatson(y_obs, x_obs, x_query, kernel):
  let weights = kernel.weights_at(x_query, x_obs)
  return GeneralizedMean<p=1, weight=weights, filter=None>(y_obs)
```

This is the **cross-tree connection to means.md**: `nadaraya_watson` is a `GeneralizedMean` recipe wrapper where the weight function is itself a `SmoothingKernel` parameter. The means tree's `WeightFn::Kernel { kind, bandwidth }` enum entry is exactly this composition.

### Sharing opportunities via TamSession

- **Bandwidth selection** (`bw.resolve(x)`) is shareable across all consumers asking for *any* shape on the same data: Silverman's `h = 1.06 · σ̂ · n^(-1/5)` only depends on `σ̂` and `n`. Tag: `(sample_fingerprint, bandwidth_rule_id)`. A user computing both `gaussian_kde` and `epanechnikov_kde` on the same sample shares one bandwidth computation.
- **Pilot bandwidth** (for plug-in estimators) is itself shareable.
- **Normalized distances** `(x(i) - y) / h` are shareable across multiple shape evaluations at the same query point and bandwidth. A user comparing `gaussian_kde` vs `epanechnikov_kde` at the same `(x, y, h)` shares the gather.
- **Distance matrix on a fixed sample** (for cross-validation bandwidth selection) is heavily shared — every shape and bandwidth candidate reuses it. This is the same `distance_matrix` intermediate that other recipes use; the sharing fires cross-tree.

---

## Top-level (b) — PositiveDefiniteKernel `K<form, bandwidth, hyperparameters>`

The Mercer kernel family. Every literature-named PD kernel is a `(form, bandwidth, hyperparameters)` assignment satisfying Mercer's condition (positive semi-definite Gram matrix for any finite set of inputs). Consumed by: SVM (every variant), Gaussian Processes (regression, classification), kernel-PCA, kernel-CCA, kernel-Ridge, kernel-Lasso, MMD distance estimation, energy distance, kernel two-sample testing.

### Parameter axes

```rust
pub struct PositiveDefiniteKernel<F, Bw, Hp> {
    pub form: F,           // FunctionalForm — the kernel function
    pub bandwidth: Bw,     // BandwidthRule (same enum as SmoothingKernel for shape-shared rules,
                           //                or PD-specific lengthscale)
    pub hyperparameters: Hp,  // FormSpecificHps — varies per form
}

pub enum FunctionalForm {
    // Stationary (depend on x - y only)
    Rbf,                                   // k(x,y) = exp(-||x-y||² / (2σ²))
    Laplace,                               // k(x,y) = exp(-||x-y|| / σ)
    Matern { nu: MaternNu },               // matérn family
    RationalQuadratic { alpha: f64 },      // k(x,y) = (1 + ||x-y||²/(2α σ²))^(-α)
    Periodic { period: f64 },              // k(x,y) = exp(-2 sin²(π||x-y||/p) / σ²)
    BesselJ { order: u8 },                 // k(x,y) = J_n(σ||x-y||)
    ExponentialPD,                         // k(x,y) = exp(-||x-y|| / σ)  (synonym of Laplace)
    // Non-stationary
    Polynomial { degree: u8, offset: f64 },     // k(x,y) = (x·y + c)^d
    Sigmoid { gain: f64, offset: f64 },         // k(x,y) = tanh(γ x·y + c)  (note: not always PSD!)
    Linear,                                // k(x,y) = x·y
    Quadratic,                             // Polynomial{degree: 2, offset: 1}
    // Compositional
    Sum(Vec<FunctionalForm>),              // K = K1 + K2 + ...
    Product(Vec<FunctionalForm>),          // K = K1 · K2 · ...
    LocallyPeriodic { period: f64, length_scale: f64 },  // Periodic · RBF
    // Structured-input
    Spectrum { k: usize },                 // string kernel — sum over k-mers
    Mismatch { k: usize, m: usize },       // spectrum + m mismatches
    Gappy { k: usize, max_gap: usize },    // spectrum + gap allowance
    Diffusion { steps: usize },            // graph kernel — heat diffusion
    RandomWalk { length: usize },          // graph kernel — bounded walk
    WeisfeilerLehman { iterations: usize },// graph kernel — WL refinement
    // White noise (for GP nugget term)
    WhiteNoise,                            // k(x,y) = σ² · δ(x = y)
    Custom(Box<dyn PsdKernel>),
}

pub enum MaternNu {
    Half,       // ν = 1/2 → Laplace
    ThreeHalves, // ν = 3/2
    FiveHalves,  // ν = 5/2
    General(f64), // arbitrary ν > 0 (uses modified Bessel function)
}
```

### Sub-kernel 1 — StationaryPD

`k(x, y) = f(||x - y||)` — depends on distance only.

| Name | form | bandwidth | hyperparameters |
|---|---|---|---|
| `rbf_kernel` / `gaussian_pd_kernel` / `squared_exponential` | Rbf | σ | — |
| `laplace_kernel` / `exponential_pd_kernel` | Laplace | σ | — |
| `matern_1/2_kernel` | Matern{Half} | σ | — |
| `matern_3/2_kernel` | Matern{ThreeHalves} | σ | — |
| `matern_5/2_kernel` | Matern{FiveHalves} | σ | — |
| `matern_general_kernel(ν)` | Matern{General(ν)} | σ | — |
| `rational_quadratic_kernel(α)` | RationalQuadratic{α} | σ | — |
| `periodic_kernel(p)` | Periodic{p} | σ | — |
| `bessel_j_kernel(n)` | BesselJ{n} | σ | — |

**Matérn ν=1/2 ≡ Laplace** — same recipe via two names, like quartic ≡ biweight in the SmoothingKernel family. The Matérn family is the canonical parent (Laplace = Matérn(1/2), RBF = Matérn(∞) limit). For implementation, **`Matern` is the primary form** with Laplace and RBF as recipe wrappers at fixed ν.

### Sub-kernel 2 — NonStationaryPD

`k(x, y)` depends on `x` and `y` individually, not just their difference.

| Name | form | hyperparameters |
|---|---|---|
| `linear_kernel` | Linear | — |
| `polynomial_kernel(d, c)` | Polynomial{d, c} | — |
| `quadratic_kernel` | Quadratic | — |
| `sigmoid_kernel(γ, c)` | Sigmoid{γ, c} | — (PSD only for specific (γ, c)!) |

The sigmoid kernel is a special case where Mercer-positivity is **not always satisfied** — it depends on parameter values. tambear should detect at construction (F13 antibody) and warn. See open question #4.

### Sub-kernel 3 — CompositionalPD

Kernels built by combining other PD kernels. Sums and products of PD kernels are PD; this sub-kernel exposes the composition.

| Name | form |
|---|---|
| `sum_kernel(K1, K2, ...)` | Sum([K1, K2, ...]) |
| `product_kernel(K1, K2, ...)` | Product([K1, K2, ...]) |
| `locally_periodic_kernel(p, l)` | LocallyPeriodic{p, l} (Periodic × RBF) |
| `additive_kernel(K_per_feature)` | Sum across feature dimensions |
| `automatic_relevance_determination` | Product of RBFs with per-dim lengthscale |

This is where the GP literature gets its expressive power. Open question #5: should `CompositionalPD` be a sub-kernel (recipe-level) or a kernel-algebra (a recursive type that any sub-kernel participates in)?

### Sub-kernel 4 — StructuredPD

PD kernels on non-vector inputs. String kernels and graph kernels live here.

| Name | form | hyperparameters |
|---|---|---|
| `spectrum_kernel(k)` | Spectrum{k} | — |
| `mismatch_kernel(k, m)` | Mismatch{k, m} | — |
| `gappy_kernel(k, g)` | Gappy{k, g} | — |
| `diffusion_kernel(steps)` | Diffusion{steps} | β (heat scale) |
| `random_walk_kernel(L)` | RandomWalk{L} | — |
| `weisfeiler_lehman_kernel(h)` | WeisfeilerLehman{h} | — |
| `white_noise_kernel(σ²)` | WhiteNoise | σ² |

### Gaps the literature has not named

- `matern_ν_with_period` — Matérn + periodic structure (compositional). Used informally; no canonical name.
- `polynomial_with_per_feature_offset` — Polynomial{d, c_i} where c varies per feature. Reachable but unnamed.
- `bessel_j_at_non_integer_order` — Bessel kernels are typically published for integer order; fractional order is reachable.
- `mismatch_with_gappy` — combine the mismatch tolerance with gap allowance. Each is named separately; the combination is reachable.
- `diffusion_random_walk_hybrid` — convex combination of diffusion and random-walk graph kernels. Common in practice; no name.

### Accumulate + gather decomposition

The fundamental operation on a PD kernel is **Gram matrix construction** — `G_ij = k(x_i, x_j)` for all pairs `(i, j)` in a training set:

```
GramMatrix(k, X):
  return gather(
    X × X,                                          // outer-product addressing
    addressing: AllPairs,
    expr: k(X(i), X(j))
  )
```

For SVM training, the optimization works on the Gram matrix directly — the kernel-tree contribution is the Gram matrix construction; the solver is downstream (separate `optimization.md` family).

For GP inference, the Gram matrix is augmented with a noise term (WhiteNoise kernel) and Cholesky-decomposed:

```
GP.posterior_mean(k, X_train, y_train, X_query):
  let K = GramMatrix(k, X_train)                    // accumulate+gather over pairs
  let K_q = gather(X_train × X_query, expr: k(X(i), X_q(j)))
  let L = cholesky(K)                               // Kingdom B/C — sequential factorization
  let alpha = L_T_solve(L, L_solve(L, y_train))     // triangular solves
  return K_q^T · alpha                              // matrix-vector product
```

Stationary kernels admit a faster path: `K_ij = f(||x_i - x_j||²)` depends only on pairwise squared distances, which is a *shareable intermediate*. Same `||x_i - x_j||²` matrix serves RBF (ν=∞), Laplace (ν=1/2), all Matérn ν, RationalQuadratic, Periodic.

### Sharing opportunities via TamSession

- **Pairwise squared-distance matrix** (`D_ij = ||x_i - x_j||²`) is shareable across all stationary PD kernels on the same training data. Tag: `(X_fingerprint, metric_id=L2)`. A user training an RBF-SVM and a Matérn-GP on the same data shares one distance-matrix computation. **This is one of the biggest sharing wins in the library.**
- **Inner-product matrix** (`I_ij = x_i · x_j`) is shareable across `Linear`, `Polynomial{d, c}`, `Quadratic`, `Sigmoid` — all non-stationary PD kernels that depend on dot products. Tag: `(X_fingerprint, kind=InnerProduct)`.
- **Cholesky factor** of the Gram matrix `K + σ²·I` is shareable across all GP queries with the same hyperparameters. Tag: `(X_fingerprint, kernel_id, noise_var, jitter)`.
- **K_train_query** matrix is *not* shareable across query sets (provenance differs) but IS shareable across multiple GP outputs (mean, variance, samples) on the same query set.

The pairwise-distance / pairwise-inner-product sharing fires across the kernels tree AND the distances tree — `pairwise_l2_distance` from `distances.md` (TBD) is the same intermediate. Same `D_ij` matrix serves both trees.

---

## Cross-kernel structural map

```
                          "kernel" family
                                |
        +----------------------(disjoint)----------------------+
        |                                                       |
  (a) SmoothingKernel                              (b) PositiveDefiniteKernel
   S<shape, bw, support>                            K<form, bw, hyperparams>
        |                                                       |
   +----+-----+--------+                          +---+----+--------+---------+
   |          |        |                          |   |    |        |         |
RadialShape Wavelet OrderedSupport      Stationary NonStat Composit  Structured
   |          |        |                          |   |    |        |
   |    Morlet,      Tukey ψ,                    RBF,Linear, Sum,    Spectrum,
   |   MexicanHat,   Huber ψ,                  Matérn,Poly, Product,Mismatch,
Gaussian,Epi,Haar,Daub,...   Hampel ψ                Laplace,Sigmoid,LocallyPeriodic,Gappy,
Biweight,Cosine,                                  RationalQuad,    ARD,        Diffusion,
Tricube,Logistic,                                 Periodic,       Additive   RandomWalk, WL,
Triangular,Box,                                    BesselJ,                   WhiteNoise
Exponential,                                       Quadratic
Silverman

   ┌─────────────── Shared functional-form vocabulary ──────────────┐
   │  Gaussian / RBF        ←─ same curve, different role:           │
   │      smoothing: ∫ K(u) du = 1                                   │
   │      PD: ⟨φ(x), φ(y)⟩ in some feature space                     │
   │  Laplace / Exponential ←─ both kernels named, both meanings     │
   │  Polynomial            ←─ only PD-side has it (no KDE analog)   │
   └─────────────────────────────────────────────────────────────────┘

                                                  [unnamed gaps]
                                                  quintic KDE kernels
                                                  matern + periodic
                                                  polynomial per-feature
                                                  bessel non-integer order
                                                  mismatch + gappy
                                                  ...
```

The two top-level kernel-of-kernels are **disjoint as catalog objects** but **share a functional-form vocabulary** (curves named "Gaussian" appear in both). The vocabulary overlap is a literature artifact; the structural difference is mathematical. The recipe API exposes both kernels separately:

```rust
// SmoothingKernel — for KDE / kernel regression
let s = gaussian_kernel(h=0.5);                    // returns SmoothingKernel<Gaussian, _, Infinite>
let density = kernel_density_estimate(x, s);

// PositiveDefiniteKernel — for SVM / GP
let k = rbf_kernel(sigma=0.5);                     // returns PositiveDefiniteKernel<Rbf, _, ()>
let gram = gram_matrix(X, k);
```

Both use `exp(-r² / 2σ²)` internally, but they're not interchangeable: passing a `SmoothingKernel` to a GP solver is a type error (no Mercer guarantee carried), and passing a `PositiveDefiniteKernel` to a KDE estimator is a type error (no unit-integral guarantee carried).

---

## Open questions for math-researcher walk-through

1. **Is the disjoint top-level split correct, or is there a hidden unifying kernel?** Both families share `(shape/form, scale, support/hyperparameters)` axes at first glance. Could there be a `MathematicalKernel<Form, Scale, Constraints>` where `Constraints` is `{UnitIntegral, PSD, AdmissibilityWavelet, ...}` and the two current top-level kernels are `Constraints=UnitIntegral` vs `Constraints=PSD`? My read: not yet useful. The consumers (KDE vs SVM/GP) ask such different questions that the unification doesn't pay rent at the recipe-tier API. But the structural rhyme is real — flag for naturalist.

2. **OrderedSupport sub-kernel — is it really in (a)?** Tukey biweight, Huber, Hampel as M-estimator ψ-functions are *robustness weights* on residuals — they're consumed by robust regression and robust mean estimation, not by KDE. They share *functional form* with KDE biweight but their role in the math is different. Should they live in their own top-level kernel (`RobustWeightingKernel`)? Or stay nested as a sub-kernel-of-(a) because they're still "smoothing-via-weighting"? I leaned toward the latter but I'm uncertain.

3. **Wavelet kernels — sub-kernel of (a), or third top-level kernel?** Wavelets carry an admissibility condition (`∫ ψ du = 0`) that no KDE kernel satisfies (KDE kernels integrate to 1). They're parameterized by (scale, translation) pairs, not a single bandwidth. Their consumers are multi-resolution analysis and continuous-wavelet-transform — not KDE. The case for a third top-level kernel is real. Counter-argument: when you use a wavelet as a smoothing kernel at a fixed scale, the admissibility condition doesn't bind, and the recipe surface looks like a KDE kernel. Naturalist's read welcome.

4. **Mercer-positivity as a parameter or a type witness?** I made `FunctionalForm::Sigmoid` a variant of `FunctionalForm` even though it's only PSD for some parameter values. Should construction *reject* Sigmoid kernels with PSD-failing parameters (F13 antibody), or accept them with a runtime warning? My intuition: detect-and-warn for Sigmoid (it's named in the literature and users expect it), but ship a `verify_psd: bool` knob that runs a finite-sample-PSD check on the actual training Gram matrix.

5. **CompositionalPD as kernel-algebra vs sub-kernel?** Sum, Product, LocallyPeriodic, ARD are *operations on kernels* — they take kernels as inputs and produce kernels as outputs. This is structurally a recursive type, not a flat enum. Should `FunctionalForm::Sum(Vec<FunctionalForm>)` and `Product(Vec<FunctionalForm>)` be in the same enum as `Rbf`, `Linear`, etc., or extracted into a kernel-algebra layer? My current sketch puts them in the same enum for simplicity; the recursive structure makes the type harder to serialize for cache keys.

6. **Cross-tree boundary with distances.md**: `pairwise_l2_distance` is the shared intermediate between this tree and the distances tree. Which tree owns the canonical recipe? My read: distances.md owns it (it's a distance, not a kernel), and PD kernels consume it via TamSession. But the cross-tree dependency means kernels.md cannot ship as a standalone family — it needs distances.md to land first (or in parallel). Surface for navigator.

---

## Implementation roadmap

This doc is the catalog tree. **Implementation is downstream and not blocked by this doc.** If/when the team builds the kernels family:

1. **Build `SmoothingKernel<Sh, Bw, Su>` first** — covers the KDE consumers (`kernel_density_estimate`, `kernel_smoothed_mean`, `nadaraya_watson_regression`, `local_polynomial_regression`). The means-tree's `WeightFn::Kernel { kind, bandwidth }` becomes a thin pointer to this kernel.
2. **Build the shape recipe wrappers** — `gaussian_kernel`, `epanechnikov_kernel`, `biweight_kernel` (alias `quartic_kernel`), etc. ~20 lines each, just parameter assignments + docstring + sibling pointers.
3. **Build the bandwidth-selection recipes** — Silverman, Scott, BCV, UCV, Sheather-Jones, Abramson. These are recipes that *return a bandwidth value*, consumed by the kernel via `BandwidthRule::resolve(x)`.
4. **Build `WaveletShape` as a sub-kernel** if/when wavelet recipes are pulled (`continuous_wavelet_transform`, `discrete_wavelet_transform`). Defer until consumer exists.
5. **Build `PositiveDefiniteKernel<F, Bw, Hp>` second** — covers SVM, GP, kernel-PCA, MMD. **Depends on distances.md** for the pairwise-distance shared intermediate.
6. **Build the Stationary recipe wrappers** — RBF, Matérn (and its fixed-ν children), Laplace, RationalQuadratic, Periodic, BesselJ.
7. **Build NonStationary** — Linear, Polynomial, Sigmoid (with PSD check).
8. **Build CompositionalPD** — Sum, Product, LocallyPeriodic, ARD. Recursive type construction.
9. **Build StructuredPD** when string/graph consumers exist (string-kernel SVM, graph-kernel SVM, WL graph classification).

Sharing across the two top-level kernels happens via TamSession with compatibility tags. The biggest wins are pairwise-distance sharing (across all Stationary PD kernels) and bandwidth-selection sharing (across all Smoothing kernels on the same sample). Both tagged at `(X_fingerprint, kind=…)`.

---

## What the tree teaches

Four things visible from this artifact that are NOT visible from the per-name recipe list:

1. **"Kernel" is two words in mathematics, not one.** The literature uses one term for the smoothing-kernel family (KDE) and the positive-definite-kernel family (Mercer), and the only way to see they're different objects is to lay out their parameter axes side by side. SmoothingKernel asks `(shape, bandwidth, support)`; PositiveDefiniteKernel asks `(form, bandwidth, hyperparameters)` plus a Mercer-positivity constraint. Different questions.

2. **Functional-form vocabulary is shared but the recipes are not.** A "Gaussian smoothing kernel" and a "Gaussian PD kernel" use the same curve `exp(-r² / 2σ²)` but with different normalization and different consumers. tambear should expose both, with the type system preventing mis-routing — passing a `SmoothingKernel` to a GP solver is a category error the API should catch.

3. **Big cross-tree sharing.** Pairwise distance matrix is the universal shared intermediate for stationary PD kernels — RBF, all Matérn, Laplace, RationalQuadratic, Periodic all consume the same `D_ij = ||x_i - x_j||²`. The sharing fires at the kernels-tree/distances-tree boundary via TamSession. For SVM-vs-GP comparison studies, this is the single biggest performance win in the library.

4. **Synonyms / aliases in literature are tree-collapsing structural information.** `quartic_kernel = biweight_kernel`, `gaussian_pd_kernel = rbf_kernel = squared_exponential_kernel`, `matérn_1/2 = laplace = exponential_pd_kernel`. Three names, one recipe. The tree makes this visible; the per-name catalog hides it. When a user reaches for `quartic_kernel` and gets a sibling-pointer to `biweight_kernel`, they learn the structure as they navigate.

The naturalist's claim was *names are parameter assignments on a graph; the graph is the actual catalog.* For kernels, the graph has two roots (disjoint top-level kernel-of-kernels) and a shared vocabulary at the leaves. The double meaning of "kernel" in the literature is the artifact of two distinct mathematical objects sharing a name; the tree makes the structure visible without forcing them into one kernel they're not.

---

## Threads downstream of this tree

- **Means tree** (`recipe-trees/means.md`): the `WeightFn::Kernel { kind: KernelType, bandwidth: f64 }` variant in `GeneralizedMean` is exactly a `SmoothingKernel` reference. The two trees connect at this seam — when a user says `mean(x).using(weight=Kernel { kind: Gaussian, h: 0.5 })`, the means tree's `GeneralizedMean` recipe wraps a kernels-tree `SmoothingKernel`. Sharing fires through TamSession (bandwidth selection, normalized distances).

- **Distances tree** (`recipe-trees/distances.md`, TBD): the pairwise-L2-distance matrix is the canonical shared intermediate consumed by every Stationary PD kernel. The distances tree owns the recipe; PositiveDefiniteKernel consumes it via TamSession. Cross-tree sharing at scale (`(X_fingerprint, metric_id=L2)`).

- **Regression / SVM / GP trees** (TBD): downstream consumers of the kernels family. SVM is a quadratic-programming optimization on a Gram matrix; GP is Cholesky + triangular solve on `K + σ²I`; kernel-PCA is eigendecomposition of the centered Gram matrix. Each lives in its own tree but consumes kernels.md via the Gram-matrix recipe.

- **Tail estimators tree** (`recipe-trees/tail-estimators.md`, TBD): kernel-weighted Hill (Aban-Meerschaert) uses a smoothing kernel on the order-statistic axis. Composes `SmoothingKernel<RadialShape>` with the tail-estimator's threshold-choice logic.

- **Signal processing tree** (TBD): convolution kernels (FIR, IIR, image filters) use the *same functional forms* as SmoothingKernel but the consumer pattern is different (FFT-based convolution to transform a signal, not to weight neighbors for density estimation). Whether to share the SmoothingKernel recipe surface or duplicate is a design question for that tree.

- **Information-theoretic tree** (`recipe-trees/information.md`, TBD): MMD (maximum mean discrepancy) and energy distance are kernel-based two-sample tests. They consume PositiveDefiniteKernel and compute distribution-level statistics. Cross-tree composition.

These are not assignments — they're invitations. The tree pattern propagates naturally as someone touches a family.
