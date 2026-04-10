# Sharing Surface Status: 26+ Families

*Naturalist summary, April 1 2026 (updated continuously)*

---

## Completed Families and Sharing Verification

| Family | Status | New GPU Primitives | Sharing with |
|--------|--------|-------------------|-------------|
| **F01** Distance | ✅ | TiledEngine DistanceOp | → F20, F21 (distance matrix via TamSession) |
| **F06** Descriptive | ✅ | scatter_multi_phi (centered), MinMax | → F07, F08, F09, F10, F25 |
| **F07** Hypothesis | ✅ | **ZERO** | ← F06 MomentStats only |
| **F08** Non-parametric | ✅ | argsort/rank (CPU) | ← F06 via transform-reentry |
| **F09** Robust | ✅ | IRLS loop (CPU) | ← F06 median/MAD, shares pattern with F16 |
| **F10** Regression | ✅ | DotProductOp tiled accumulate | ← F06 MomentStats (centering), → F14, F22, F33 |
| **F13** Survival | ✅ | — | IRLS template (Rhyme #13) |
| **F14** Factor Analysis | ✅ | — | GramMatrix → EigenDecomp (predicted cheap, confirmed) |
| **F16** Mixture | ✅ | — | EM = IRLS (Rhyme #12), LogSumExpOp compound |
| **F17** Time Series | ✅ | Affine scan | Kingdom B first implementation |
| **F19** Spectral TS | ✅ | — | FrequencyDomain intermediate |
| **F20** Clustering | ✅ | — | DistancePairs from TamSession, validation zero new GPU |
| **F22** DimReduction | ✅ | — | GramMatrix → EigenDecomp (predicted cheap, confirmed) |
| **F25** Information | ✅ | PHI_COUNT histogram | Shares PHI_COUNT kernel with F06 |
| **F26** Complexity | ✅ | **ZERO** (reimplements) | Cross-kingdom composite, pattern-level sharing only |
| **F31** Interpolation | ✅ | **ZERO** (reimplements) | Second composite: polyfit=F10, GP=F01+F10, splines=Kingdom B |
| **F32** Numerical | ✅ | **ZERO** (pure CPU) | Root finding, quadrature, ODE solvers |
| **F02** Linear Algebra | ✅ | Mat struct, LU, Cholesky, QR, SVD, eigen | Foundation — but nobody imports from it yet |
| **F03** Signal Processing | ✅ | FFT (butterfly transform) | Cross-domain: EMA = F17, Welch = segmented A, DCT |
| **F04** Random Numbers | ✅ | xoshiro256**, distributions | Infrastructure — consumed by F34, F16, F08, F20 |
| **F05** Optimization | ✅ | Adam, L-BFGS, Nelder-Mead | 4th oracle; Adam = two EWM Affine scans (Rhyme #15) |
| **F29** Graph | 🔨 | Graph struct, sparse adjacency | SparseAdjacency grouping; PageRank = sparse Kingdom C |
| **F30** Spatial | ✅ | **ZERO** (reimplements) | Kriging=GP (Rhyme #24), 7th solver, SpatialWeights=Graph adj |
| **F23** Neural Ops | 🔨 | **ZERO** (im2col+GEMM, attention=DotProduct) | FIRST cross-import (Mat, mat_mul from F02). 8-operator closure CONFIRMED. |

## Verified Sharing Chains

### Chain 1: MomentStats Trunk (Pass 1)
```
F06 scatter_multi_phi → MomentStats(order=2..4)
    ├── F07: t-tests, ANOVA, effect sizes → ZERO new GPU ✓
    ├── F08: rank(x) → MomentStats(ranks) → Spearman, K-W ✓
    ├── F09: MAD/median → IRLS weight → weighted scatter (iterative)
    ├── F10: column means for centering before GramMatrix ✓
    └── F25: PHI_COUNT histogram → entropy ✓
```

### Chain 2: DistancePairs Trunk
```
TiledEngine → DistanceMatrix
    ├── DBSCAN: registers in TamSession ✓
    ├── KNN: reads from TamSession, zero GPU cost ✓
    └── KMeans: tiled distance + argmin ✓
```

### Chain 3: GramMatrix Backbone (Pass 2)
```
F10 DotProductOp → GramMatrix (centered via F06 means)
    ├── F10: Cholesky → FittedModel (β̂, R², F-stat)
    ├── F14: EigenDecomp → loadings (Factor Analysis)
    ├── F22: EigenDecomp → principal components (PCA)
    ├── F33: Subblock eigen → canonical correlations (CCA)
    ├── F01: to_l2_distance_matrix() → DistancePairs for free
    └── F07: ANOVA F-stat = Regression F-stat (same extraction)
```

## Structural Rhymes (23 total)

| # | Rhyme | Evidence |
|---|-------|---------|
| 1 | ANOVA = Regression F-test | F07/F10: same GramMatrix extraction |
| 2 | KMeans E-step = KNN query | kmeans.rs/knn.rs: same tiled distance + argmin |
| 3 | Kalman = Recursive LS | Affine scan with A=I |
| 4 | Spectral clustering = Laplacian eigenmap | Same Laplacian → eigen |
| 5 | EM M-step = Grouped variance | F16: same scatter_multi_phi |
| 6 | Random forest split = Mutual info | Histogram → entropy |
| 7 | Spearman = Pearson on ranks | nonparametric.rs:90 |
| 8 | Kruskal-Wallis = ANOVA on ranks | rank-sum extraction |
| 9 | L2 distance = GramMatrix extraction | norms[i] - 2K[i,j] + norms[j] |
| 10 | IRT = Mixed logistic regression | EM over random effects |
| 11 | Bayesian = everything with posterior | Prior × Likelihood |
| 12 | IRLS = EM | Same iterative weighted accumulation |
| 13 | IRLS Master Template (8 families) | GLM=Robust=EM=Cox: one weighted scatter |
| 14 | MANOVA:ANOVA :: CCA:Regression | Same GramMatrix, different extraction |
| 15 | Adam:SGD :: ARIMA:AR | Affine scan wrapping base |
| 16 | LME = Self-Tuning Ridge | σ_ε²/σ_b² IS Ridge's λ |
| 17 | DB/CH = RefCenteredStats | 6th domain for RefCenteredStats |
| 18 | Bayesian prior = regularization | Ridge=Gaussian, LASSO=Laplace |
| 19 | Panel FE = ANOVA demeaning | Same RefCenteredStats scatter |
| 20 | Cox PH = Logistic on risk sets | IRLS #8, softmax weights |
| 21 | IRT info = Fisher info = IRLS wt | μ(1-μ): 4 names, 1 formula |
| 22 | LDA = Discriminant CCA | Eigendecomp of scatter ratio |
| 23 | 2SLS = Sequential OLS | F35: two F10 calls, ~30 lines |
| 24 | Kriging = GP Regression | Same covariance solve; nugget=noise, sill=signal |

## Key Metrics

- **GPU primitives used**: scatter_multi_phi, DotProductOp, DistanceOp, MinMax, AffineOp, FFT — 6 primitive types
- **Of 8 operators**: Add, TiledAdd (Kingdom A), Max/Min, Affine (Kingdom B) confirmed. Särkkä pending (F17 implemented but not via scan framework yet).
- **Families fully implemented (code)**: 23 with source files (F01-F04, F06-F10, F13-F14, F16-F20, F22, F25-F26, F29-F32)
- **Families complete (sharing surface only)**: 4 (F05, F11-F12, F33)
- **Families in progress**: 7 (F11, F15, F21, F23, F27-F29, F33)
- **Families pending**: 1 (F24)
- **Structural rhymes**: 24 identified
- **Naive formula bugs found**: 6 — centering principle violations
- **Composite families**: 3 (F26, F30, F31)
- **Four estimation oracles**: MomentStats, IRLS, Affine scan, GradientOracle
- **Codebase**: ~32,000 lines Rust, ~620 tests
- **Sharing gap**: 7 independent linear solvers, 3 data layouts, 0 cross-references

## Predictions

**F11 (Mixed Effects) should confirm Rhyme #16** — self-tuning Ridge. The Henderson equations are GramMatrix on augmented [X|Z] + REML variance estimation. If it follows the IRLS master template, zero new accumulation primitives.

**F18 (Volatility) will be the Kingdom B stress test** — GARCH is Affine scan where the combine function involves conditional variance. If this reuses F17's Affine infrastructure cleanly, Kingdom B sharing is validated.

**F33 (Multivariate) confirms Rhyme #14** — MANOVA and CCA should be GramMatrix subblock extractions. Navigator's sharing surface says ~250 new lines on top of F10+F22.

**F05 (Optimization) is the 4th oracle** — GradientOracle trait. This is infrastructure that F16 MLE, F18 MLE, and F34 MAP will all consume. Watch for whether it actually reuses the IRLS iteration envelope or needs its own.

**F16 (EM/GMM) will be the FIRST KINGDOM C test** — iterative loop over weighted scatter. The `scatter_multi_phi_weighted` extension is the key new primitive.

---

*The sharing tree is real. Seven families, four GPU primitives. The genealogy predicted this structure before the code was written, and the code confirms it at every step.*
