# The Spine, Missing MSR Types, and Structural Rhymes

*Naturalist response to navigator's three questions*
*2026-04-01*

---

## Q1: Does the Sharing Tree Have a Spine?

**Yes.** Three passes unlock ~25 of 35 families:

### Pass 1: Moments + Extrema (unlocks Kingdom A scalar families)

```
accumulate(data, ByKey(groups),
    phi=[1, x, x², x³, x⁴, max, min],
    op=[Add×5, Max, Min])
```

**Produces:** MomentStats(order=4) + ExtremaStats per group.

**Unlocks fully:** F06, F07, F25(histogram part)
**Unlocks partially:** F08(moment-based tests), F09(starting values)

Cost: one scatter_multi_phi dispatch (~40μs JIT + O(N) data). Marginal cost of x³, x⁴ beyond x²: two extra multiplies per element in the already-open kernel.

### Pass 2: Cross-Products (unlocks Kingdom A matrix families)

```
accumulate(data, Tiled(p,p), xᵢ·xⱼ, Add)
```

**Produces:** GramMatrix(p×p) = X'X, X'y, y'y.

**Unlocks fully:** F10(regression), F14(factor analysis), F22(PCA), F33(multivariate)
**Unlocks partially:** F02(least squares)

Cost: one tiled accumulate, O(N × p²). For typical p < 100, this is fast. For p > 1000, tiling kicks in.

### Pass 3: Distances (unlocks metric families)

```
accumulate(data, Tiled(n,n), (a-b)², Add)
```

**Produces:** DistancePairs for L2 (and other metrics via ManifoldMixtureOp).

**Unlocks:** F01, F20(clustering), F21(KNN/SVM), F22(t-SNE/UMAP), F28(manifolds), F30(spatial)

Cost: O(N² × d). This is the expensive one. For large N, use approximate methods (ANN).

### The Spine Summary

| Pass | Cost | Families Unlocked |
|------|------|------------------|
| Moments | O(N) | 06, 07, 08p, 09p, 25p |
| Cross-products | O(N × p²) | 02p, 10, 14, 22, 33 |
| Distances | O(N² × d) | 01, 20, 21, 22, 28, 30 |
| **Total** | | **~20 of 35** |

The remaining ~15 need: Affine scan (F17, F18, F13), iteration envelope (F05, F09, F16, F20-iter, F34), FFT (F03, F19), sort (F08-rank), random generation (F04), graph structure (F29), TDA (F27).

**Should the compiler auto-run the spine?** I think yes for Pass 1 (nearly free). Maybe for Pass 2 (depends on p). No for Pass 3 (O(N²) is never free). The `tb.session(data).prime()` idea should be Pass 1 + optionally Pass 2.

---

## Q2: Missing MSR Types

Your 9 types cover all Kingdom A raw accumulations. What's missing are:

### Level 2: Derived Intermediates

These are produced by EXTRACTING from Level 1 accumulations. They're reusable across algorithms but aren't themselves accumulations:

| Type | Produced From | Consumed By |
|------|--------------|------------|
| **EigenDecomposition** | GramMatrix → eigendecompose | F14(factor loadings), F22(PCA components), F20(spectral clustering), F29(graph spectral) |
| **FittedModel** (β, ŷ, e) | GramMatrix → solve | F07(F-test), F09(robust starting point), F10(diagnostics), F35(residualization) |
| **FrequencyDomain** (FFT output) | Raw data → FFT | F19(PSD, coherence, transfer function), F03(filtering in freq domain), F26(spectral entropy) |

### Level 1: Missing Raw Accumulations

| Type | What It Is | Families |
|------|-----------|----------|
| **AffineState** | Scan state (A, b) from sequential pass | F17(AR/ARIMA/EWM), F18(GARCH), F13(cumulative hazard) |
| **GraphStructure** | Adjacency, degree, Laplacian | F29(PageRank, communities, centrality), F20(spectral clustering) |
| **SortedPermutation** | argsort indices | F08(all rank-based), F27(persistence via sorted edges), F06(exact quantiles) |

**AffineState** is the Kingdom B equivalent of MomentStats. If F17 runs an Affine scan, the resulting state (fitted AR coefficients, EWM values, Kalman smoothed estimates) should be deposited in the session for F18, F19, and diagnostics to consume.

**GraphStructure** has no analogue in the moment/distance world. Graph algorithms need topological structure — adjacency, not metric distance. This is genuinely separate.

**SortedPermutation** is the argsort result. Computing ranks (for Spearman, Wilcoxon, etc.) requires sorting. The sort itself produces a permutation that multiple rank-based algorithms share. This is a gather preprocessing step, not an accumulation — but it's expensive (O(N log N)) and shareable.

### Revised Count: 9 + 3 raw + 3 derived = 15 MSR types

---

## Q3: Structural Rhymes

### Confirmed Rhymes (same math, different name)

**ANOVA (F07) = Regression F-test (F10)**
- ANOVA with k groups is IDENTICAL to linear regression with k-1 dummy variables
- The between-group SS = regression SS when X is one-hot group indicators
- The F-statistic formula is identical: (SSR/df₁) / (SSE/df₂)
- **Implementation:** one regression with dummy variables gives ANOVA for free
- **Sharing:** both consume GramMatrix (if you include the intercept and dummies)

**KMeans E-step (F20) = KNN query (F21)**
- Both: given query points Q and reference points R, find nearest in R for each q
- KMeans: R = centroids (k points), query = data (N points), find 1-nearest
- KNN: R = training data (M points), query = test (N points), find k-nearest
- **Same:** tiled distance accumulate + argmin per row
- **Difference:** KNN needs top-k not argmin-1; KMeans uses result to update R
- **Sharing:** DistancePairs(Q vs R) is computed once, consumed by both

**PCA (F22) and Factor Analysis (F14)**
- PCA: eigendecompose Σ (covariance matrix)
- FA: eigendecompose R - Ψ (correlation matrix minus uniquenesses)
- **Shared:** both need the covariance/correlation matrix from GramMatrix
- **Different:** FA iterates (EM or principal axis) to estimate Ψ; PCA doesn't iterate
- **Sharing:** GramMatrix → correlation extraction is shared; FA adds an iteration envelope

**Kalman (F17) = Recursive Least Squares (F10)**
- Kalman with A=I, Q=0: P(t|t) = (X'X)⁻¹σ², K = P·x'/(x·P·x'+σ²)
- RLS with forgetting factor λ=1: identical update
- **Same:** Affine scan where state = (β, P) and combine = Bayesian update
- **Sharing:** Särkkä operator handles both. Set A=I and Q=0 for RLS.

### New Rhymes (not yet mentioned)

**Kernel SVM (F21) = GP regression (F34/F31)**
- Both build K(xᵢ, xⱼ) = kernel(xᵢ, xⱼ) matrix
- SVM: find α maximizing Σαᵢ - ½α'Kα subject to constraints → optimization (F05)
- GP: compute K⁻¹y for prediction, diag(K - K*K⁻¹K*') for variance → matrix solve (F02)
- **Sharing:** same kernel matrix K. One tiled accumulate with kernel expr.

**Spectral Clustering (F20) = Graph Laplacian Eigenmap (F29)**
- Spectral clustering: build similarity graph → compute Laplacian L → eigendecompose → cluster eigenvectors
- Graph spectral embedding: same L → same eigendecomposition → use eigenvectors as coordinates
- **Identical computation.** The only difference is what you DO with the eigenvectors (cluster vs. embed)

**CCA (F33) = Multivariate Regression**
- CCA: find directions maximizing cor(Xa, Yb)
- Equivalent to regression of Y on X, in the space of canonical directions
- Solves generalized eigenvalue: Σ_xy·Σ_yy⁻¹·Σ_yx·a = λ·Σ_xx·a
- **Sharing:** Σ_xx, Σ_yy, Σ_xy are ALL subblocks of the full GramMatrix

**IRT 2PL (F15) = Logistic Regression with Random Effects (F10+F11)**
- Both: P(y=1) = sigmoid(a·θ + b) where θ is a latent variable
- IRT: θ is "person ability" (random effect), (a,b) per item (fixed effects)
- Mixed logistic: same structure — random intercept per subject, fixed effects per predictor
- **Sharing:** same EM estimation (E-step integrates over random effects, M-step is logistic regression)

### The Rhyme Count

I've now identified **11 structural rhymes** where families that appear different are computationally identical or near-identical. These represent sharing opportunities that the compiler should detect:

| Rhyme | Families | Shared Computation |
|-------|----------|-------------------|
| ANOVA = Regression F-test | 07 ↔ 10 | GramMatrix → F-statistic |
| KMeans E-step = KNN | 20 ↔ 21 | DistancePairs → argmin/top-k |
| PCA ~ Factor Analysis | 22 ↔ 14 | GramMatrix → correlation → eigen |
| Kalman = Recursive LS | 17 ↔ 10 | AffineState with A=I |
| Kernel SVM = GP regression | 21 ↔ 34 | Kernel matrix K |
| Spectral clustering = Laplacian eigenmap | 20 ↔ 29 | Graph Laplacian → eigen |
| CCA = multivariate regression | 33 ↔ 10 | GramMatrix subblocks |
| IRT 2PL = mixed logistic | 15 ↔ 11 | EM over random effects |
| Chi-square = mutual information | 07 ↔ 25 | CrosstabStats (joint histogram) |
| Random forest splits = information gain | 21 ↔ 25 | Histogram → entropy |
| Transfer entropy = lagged MI = Granger | 25 ↔ 17 | Lagged gather → histogram/regression |

---

## The Revised Sharing Tree (with Levels)

```
RAW DATA
│
├─[LEVEL 1: Raw Accumulations — O(N)]─────────────────
│  ├── MomentStats        → F06,F07,F08,F09,F25
│  ├── ExtremaStats       → F06,F08,F09,F20
│  ├── WeightedMomentStats → F10(WLS),F07
│  ├── BivariateMomentStats → F10(bivar),F33
│  ├── RankStats           → F08(all rank-based)
│  ├── QuantileSketch      → F06,F08,F09
│  ├── CrosstabStats       → F07,F16,F25
│  ├── GramMatrix          → F02,F10,F14,F22,F33
│  ├── DistancePairs       → F01,F20,F21,F22,F28,F30
│  ├── AffineState  (NEW)  → F17,F18,F13
│  ├── GraphStructure (NEW) → F29,F20(spectral)
│  └── SortedPermutation (NEW) → F08,F27,F06(exact)
│
├─[LEVEL 2: Derived Intermediates — O(K²) or O(p³)]───
│  ├── EigenDecomposition  (from GramMatrix) → F14,F22,F20,F29
│  ├── FittedModel         (from GramMatrix) → F07,F09,F10,F35
│  └── FrequencyDomain     (from raw data via FFT) → F03,F19,F26
│
└─[LEVEL 3: Iterative State — multiple passes]────────
   ├── ConvergedCentroids   (from KMeans iter) → F20(eval),F21(init)
   ├── FactorLoadings       (from FA iter) → F14(CFA),F15(IRT)
   └── MixtureParameters    (from EM iter) → F16,F34(DPM)
```

Level 1 is where the sharing goldmine lives. Level 2 is where the rhymes live. Level 3 is where the iteration envelopes deposit their converged state.

---

## Addendum: Two Sharing Trunks (confirmed April 1)

The sharing tree has two main trunks, bridged by GramMatrix:

### Trunk 1: MomentStats (statistics chain)
```
F06 scatter_multi_phi → MomentStats(order=2..4)
    ├── F07: t-tests, ANOVA, effect sizes → ZERO new GPU ✓ (confirmed: 27 algorithms, ~930 lines, 0 GPU code)
    ├── F08: rank(x) → MomentStats(ranks) → Spearman, K-W ✓ (transform-reentry pattern)
    ├── F09: MAD/median → IRLS weight → weighted scatter ✓ (Kingdom C iterative)
    ├── F10: column means for centering before GramMatrix ✓
    └── F25: PHI_COUNT histogram → entropy ✓ (same JIT kernel)
```

### Trunk 2: DistancePairs (geometry chain)
```
TiledEngine → DistanceMatrix → TamSession
    ├── DBSCAN: registers in TamSession ✓
    ├── KNN: reads from TamSession, zero GPU cost ✓
    └── KMeans: tiled distance + argmin ✓
```

### Bridge: GramMatrix
```
GramMatrix (tiled accumulate, O(N×p²))
    ├── diagonal → variances → MomentStats (Trunk 1)
    ├── off-diagonal + norms → L2 DistancePairs (Trunk 2, derived for free)
    ├── Cholesky → regression F10
    └── eigendecomp → PCA F22, factor analysis F14
```

---

## Addendum: Operator Classification Update

### LogSumExpOp = Compound, Not Atom

The F16 navigator proposes LogSumExpOp for EM E-step. Decomposition: Max + element-wise exp + Add + element-wise log. This is a **performance fusion** of two existing operators, not a genuinely new mathematical operation.

**Taxonomy distinction:**
- **Atoms** (8, closed): Add, Welford, TiledAdd, Affine, Särkkä, Max/Min, ArgMin/ArgMax, SoftmaxWeighted
- **Compounds** (grows as needed): LogSumExp, (and arguably SoftmaxWeighted itself)

The 8-operator model holds. Compounds fuse atoms into single kernels for numerical stability or memory bandwidth.

---

## Addendum: Centering as Universal Kingdom A Requirement

**Proven by adversarial** (GramMatrix centering proof, April 1):

The centered GramMatrix condition number is INVARIANT to data offset: cond((X-μ)'(X-μ)) = 8.33 regardless of whether data is at offset 0, 1e4, or 1e6. The naive X'X has cond = ∞ at offset 1e6.

At offset 1e8: naive regression returns β₁ = **-2.0** instead of +3.0. The sign is wrong.

**Universal rule**: ALL Kingdom A accumulations must use the centered basis. `accumulate(grouping, (x-μ)^k, Add)`, where μ comes from Pass 1 (MomentStats). Centering is a correctness requirement, not an optimization.

---

## Addendum: Cross-Domain Unification — RefCenteredStats

`scatter_phi("v - r", keys, ref_values)` appears as a primitive across multiple domains:
- F06: centered variance = scatter_phi("(v-r)*(v-r)", keys, μ)
- F20: HDBSCAN stability = scatter on (point - centroid)
- F10: regression = GramMatrix on centered data
- Financial: excess returns = price - reference

All are the same: subtract a reference, accumulate the result. RefCenteredStats is the cross-domain unification. The reference is the only thing that varies.

---

## Addendum: Rhyme #12 — IRLS = EM

**IRLS** (F09, robust M-estimators) and **EM** (F16, mixture models) are the SAME iterative pattern:
```
iterate: weights = f(residual) → μ = Σwφ(x) / Σw
```
- IRLS: deterministic weight function (Huber ψ)
- EM: probabilistic weight function (posterior responsibility)

Both are Kingdom C: outer iteration over weighted accumulation (Kingdom A). The `scatter_multi_phi_weighted` extension serves both.

---

## Addendum: Rhyme #13 — The IRLS Master Template

**GLM IRLS = Robust IRLS = EM M-step** — all three are the SAME weighted scatter accumulate with different weight functions. This subsumes and extends Rhyme #12.

```
Iteration:
  weights wᵢ = f(μᵢ, yᵢ)           // differs by algorithm
  β̂_new = (X'WX)⁻¹ X'Wz           // weighted GramMatrix solve
  μ = g⁻¹(Xβ̂_new)                  // predict
```

| Domain | Family | Weight | What W Encodes |
|--------|--------|--------|----------------|
| Robust | F09 | ψ(ε/σ)/(ε/σ) | Outlier down-weighting |
| GLM logistic | F10 | μ(1-μ) | Binomial variance function |
| GLM Poisson | F10 | μ | Poisson variance function |
| GLM gamma | F10 | μ² | Gamma variance function |
| Mixture EM | F16 | r_nk (posterior) | Soft cluster membership |
| Mixed effects | F11 | (ZGZ'+R)⁻¹ diag | Random effect variance structure |

**One `scatter_multi_phi_weighted` call per iteration covers all six.** The weight function is the only domain-specific code. The inner accumulation (X'WX, X'Wz → solve) is universal.

IRLS is the MASTER TEMPLATE for Kingdom C weighted-iteration algorithms. Any algorithm that "assigns weights to observations and solves weighted least squares" belongs to this family. The number of families covered will grow as more are implemented — F11 (mixed effects), F12 (panel GEE), F13 (Cox PH via IRLS), F15 (IRT via EM-IRLS) all likely instantiate this template.

**Publishable framing**: "One weighted scatter primitive for robust regression, generalized linear models, and Gaussian mixture models — six algorithm families, one GPU kernel."

---

## Addendum: Rhymes #14–17 (from navigator F05/F33/F11/F20)

**Rhyme #14: MANOVA : ANOVA :: CCA : Regression**
All four = same GramMatrix computation, different extraction:
- ANOVA: scalar group means → F-test on SS_between/SS_within
- MANOVA: vector group means → eigenvalues of HE⁻¹
- Regression: β = Σ_xx⁻¹ Σ_xy (GramMatrix subblock solve)
- CCA: canonical weights = SVD of Σ_xx⁻½ Σ_xy Σ_yy⁻½ (GramMatrix subblock SVD)

The GramMatrix subblock structure is the unifying MSR for the multivariate family (F10, F33, F22).

**Rhyme #15: Adam : SGD :: ARIMA : AR**
Both add a momentum/memory layer via Affine scan on top of the base algorithm:
- Adam = SGD + first+second moment EWM layers
- ARIMA = AR + MA and integration layers
- Base algorithm = iterate + update; extension = Affine scan wraps the update

**Rhyme #16: LME = Self-Tuning Ridge Regression**
- Ridge: minimize ||y - Xβ||² + λ||β||² where λ is SET by user
- LME: minimize ||y - Xβ - Zb||² + b'G⁻¹b where λ = σ_ε²/σ_b² is ESTIMATED from data
- Henderson equations = GramMatrix on augmented [X|Z] + block diagonal regularization
- The variance ratio IS Ridge's λ, just estimated via REML

**Rhyme #17: Davies-Bouldin/Calinski-Harabasz = RefCenteredStats (5th domain)**
- SS_within (CH) = scatter_phi("(v-r)²", keys=labels, ref=centroids)
- s_k intra-scatter (DB) = scatter_phi("|v-r|", keys=labels, ref=centroids)
- RefCenteredStats now spans 5 domains: variance stats, HDBSCAN stability, financial excess moments, F10 regression centering, and cluster validation

**Rhyme #18: Bayesian prior = regularization**
- Ridge = Gaussian prior on β; LASSO = Laplace prior on β
- L2/L1 regularization IS Bayesian MAP estimation — literally the same optimization
- F34's `Prior` enum covers the regularization story for F10 also

**Rhyme #19: Panel FE = Within-Group ANOVA demeaning**
- Same RefCenteredStats scatter, same subtract-group-means operation
- ANOVA uses SS_within as denominator; FE uses within-group variation as signal

**Rhyme #20: Cox PH = Logistic Regression on Risk Sets**
- Same IRLS template, same softmax-weighted GramMatrix
- Cox = nested suffix risk sets (Kingdom B within Kingdom C)
- 8th family in the IRLS master template

**Rhyme #21: IRT information = Fisher information = IRLS weight = Bernoulli variance**
- All four: μ(1-μ) (for logistic link), or generally Var(Y|θ)
- Four names for one mathematical object. Four research communities. One line of code.

**Rhyme #22: LDA = Discriminant CCA**
- LDA (F21) IS CCA (F33) with Y = one-hot class indicators
- Both compute eigendecomp of scatter-matrix ratios

**Rhyme #23: 2SLS = Sequential OLS**
- Stage 1: OLS(endog on instruments). Stage 2: OLS(y on predicted). Two F10 calls.
- F35 (Causal Inference) is ~90 lines total. Zero new primitives.

**Total structural rhymes: 23.**

---

## Addendum: The Four Oracles (from navigator F05)

Every estimation method in the library routes through one of four oracle interfaces:

| Oracle | Kingdom | Families |
|--------|---------|----------|
| MomentStats extraction | A | F06, F07, F08p, F25 |
| IRLS (known-form weight update) | C over A | F09, F10 GLM, F11, F16 EM, F13 |
| Affine scan backward | B | F17 ARIMA, F18 GARCH, Kalman |
| GradientOracle (black-box gradient) | C over (A or B) | F05 Adam/L-BFGS, F16 MLE, F18 MLE, F34 MAP |

35 families × ~3 estimation methods each → every method routes through one of 4 oracles. The oracle determines the computational strategy; the family determines the mathematical domain.

---

## Addendum: F31 Interpolation — Second Composite Family

F31 (1648 lines, 18 functions, zero imports) is the second composite family after F26. Key findings:

1. **polyfit = F10 OLS** with polynomial basis (Vandermonde normal equations = GramMatrix)
2. **GP regression = F01 distance + F10 solve** (kernel matrix is a transformed distance matrix)
3. **Cubic splines = Kingdom B** (Thomas algorithm = sequential scan, not commutative)
4. **Three linear solvers**: Cholesky (train/cholesky.rs), Gauss elim (interpolation.rs:609), Thomas (interpolation.rs:274) — three implementations of one concept

---

## Addendum: Naive Formula Bug Class — Six Instances (from adversarial sweep)

The same E[x²]-E[x]² catastrophic cancellation appears in 6 locations:

| Severity | File | Line | Formula |
|----------|------|------|---------|
| **HIGH** | hash_scatter.rs | 193 | `(sq - s*s/c) / (c-1)` |
| **HIGH** | intermediates.rs | — | m2 construction from raw sums |
| **HIGH** | robust.rs | 381 | `n*sxx - sx*sx` (SINGULAR at 1e8) |
| **MEDIUM** | tambear-py/src/lib.rs | 101 | Python binding variance |
| **LOW** | complexity.rs | 348 | linear_fit_segment (x=integers) |
| **LOW** | main.rs | 98 | test reference uses buggy formula |

**Root cause**: GPU scatter accumulates `{count, sum, sum_sq}`. Deriving variance from these IS the naive formula. **Structural fix**: accumulate `{count, sum, m2}` using Welford/CLG merge — which `MomentStats` already does. The issue is bypass paths that recompute variance without going through MomentStats.

1 correct instance: complexity.rs:254 `ols_slope` — properly centered.

Full adversarial sweep: `campsites/.../adversarial/naive-formula-codebase-sweep.md`

---

## Addendum: F30 Spatial Statistics — Rhyme #24 and 7th Solver

### Rhyme #24: Kriging = GP Regression

Kriging (F30, spatial.rs:147) and Gaussian process regression (F31, interpolation.rs:1055) are the SAME algorithm:
- Geostatistics: build covariance matrix from variogram → solve with Lagrange constraint → weighted prediction
- Machine learning: build kernel matrix → add noise diagonal → solve → weighted prediction
- Nugget IS noise variance. Sill IS signal variance. Variogram model IS kernel function.
- Two families, two files, two Gaussian elimination solvers, zero cross-references

**Total structural rhymes: 24.**

### 7th Linear Solver

`solve_system` (spatial.rs:211): Gaussian elimination with partial pivoting, Vec<Vec<f64>>. Same layout and algorithm as interpolation.rs:609. The kriging matrix is SPD → should use Cholesky from linear_algebra.rs.

**Solver census: 7 implementations, 3 data layouts, 0 cross-references.**

### SpatialWeights = Graph Adjacency

`SpatialWeights { neighbors: Vec<Vec<(usize, f64)>> }` (spatial.rs:244) is isomorphic to `Graph { adj: Vec<Vec<Edge>> }` (graph.rs). Same SparseAdjacency grouping pattern, different types.

### Moran's I = Weighted Scatter over SparseAdjacency

`I = (n/S0) Σ w_ij (x_i - x̄)(x_j - x̄) / Σ (x_i - x̄)²` — center by mean, then cross-covariance weighted by spatial neighbors. Correctly applies centering principle (line 305). In accumulate terms: `accumulate(data, SparseAdjacency(W), centered_cross_product, Add)`.

### Point Pattern = DistancePairs

- Ripley's K (line 353): O(n²) pairwise distances → threshold count = DBSCAN ε-neighborhood
- nn_distances (line 385): nearest neighbor per point = KNN with k=1
- SpatialWeights::knn (line 252): k-nearest-neighbor search = knn.rs reimplemented
- Clark-Evans R (line 405): test statistic from mean NN distance = consumer of DistancePairs

### MSR Insight (from module header)

The variogram IS the spatial sufficient statistic: {nugget, sill, range} is the MSR of spatial structure. Once you have the variogram model, kriging needs no raw data.

---

## Addendum: F23 Neural Network Ops — 8-Operator Closure CONFIRMED

### The Largest Family Adds ZERO New Operators

neural.rs: 1901 lines, 56 tests, 73 public functions. First cross-module import (`crate::linear_algebra::{Mat, mat_mul}`, `crate::rng::TamRng`). Zero new mathematical operators.

### Conv2D = im2col (gather) + mat_mul (DotProduct)
- im2col (line 362-381): windowed gather reshaping input patches into columns
- GEMM (line 387): `mat_mul(&kernel_mat, &col_mat)` — DotProduct operator
- New ADDRESSING pattern (windowed/strided), not new operator

### Attention = Three DotProducts + Softmax
- `scores = Q @ K^T * scale` → DotProduct
- `weights = softmax(scores)` → element-wise (not an operator)
- `output = weights @ V` → DotProduct
- Multi-head adds 4 more mat_muls (Q/K/V/O projections). All DotProduct.

### Normalization = MomentStats + Affine (variation is ONLY in grouping)
- BatchNorm: ByKey(feature) across batch → affine. CORRECTLY CENTERS.
- LayerNorm: ByKey(sample) across features → affine. CORRECTLY CENTERS.
- RMSNorm: sum-of-squares only — INTENTIONAL no centering (Transformer convention).
- GroupNorm: Segmented groups of channels → affine.
- InstanceNorm: per-channel per-sample → affine.

### Operator Census After F23
8 operators confirmed for 31/35 families. Remaining 4 (F11, F24, F28, F33) predicted zero new operators.

**Total structural rhymes: 24. Total solver implementations: 7 (no new solver in F23 — it imports mat_mul!)**
