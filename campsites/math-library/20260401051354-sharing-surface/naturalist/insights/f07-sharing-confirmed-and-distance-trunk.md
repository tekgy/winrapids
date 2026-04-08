# F07 Sharing Confirmed + Distance Trunk Identified

*Naturalist observation, April 1 2026*

---

## F07 Sharing Verification: CONFIRMED

**Prediction** (navigator): "Every hypothesis test in F07 is an EXTRACTION from MomentStats — not a new accumulation."

**Actual** (hypothesis.rs):
- Imports: `crate::descriptive::MomentStats` + `crate::special_functions::*`
- Zero imports of scatter_jit, accumulate, compute_engine, or any GPU infrastructure
- Every test function: `fn test(&MomentStats, ...) → TestResult` — pure CPU arithmetic
- HypothesisEngine wraps DescriptiveEngine — delegates ALL scatter work to F06

**Coverage**: 27 algorithms (18 tests + 6 effect sizes + 3 correction methods) in ~900 lines. Zero GPU code.

**Conclusion**: The sharing tree is not theoretical. F07 is a pure extraction layer on F06's accumulator fields. MomentStats IS the sharing surface between descriptive statistics and hypothesis testing.

---

## Two Sharing Trunks

The implemented code reveals two clear sharing trunks:

### Trunk 1: MomentStats (statistics chain)
```
F06 scatter → MomentStats(order=2..4)
    ├── F07: t-tests, ANOVA, chi-square, effect sizes (CONFIRMED: zero new GPU)
    ├── F08: will need SortedPermutation for non-parametric, but polynomial parts share
    ├── F09: will need Masked grouping for robust variants
    └── F25: histogram → entropy (CONFIRMED earlier: shares PHI_COUNT kernel)
```

### Trunk 2: DistancePairs (geometry chain)
```
TiledEngine → DistanceMatrix
    ├── DBSCAN: threshold → density → union-find → clusters (registers matrix in TamSession)
    ├── KNN: sorts per row → k-nearest (reads matrix from TamSession — zero GPU cost)
    ├── KMeans: tiled distance + argmin (same pipeline, different reference set)
    └── (future) Silhouette, spectral clustering, manifold learning
```

### The Bridge: GramMatrix
```
GramMatrix (tiled accumulate, O(N×p²))
    ├── diagonal → variances → MomentStats (Trunk 1)
    ├── off-diagonal + norms → L2 DistancePairs (Trunk 2)
    ├── Cholesky → regression (F10)
    └── eigendecomp → PCA (F22), factor analysis (F14)
```

---

## LogSumExpOp Classification

Navigator proposes LogSumExpOp for F16 E-step. Decomposition: Max + element-wise exp + Add + element-wise log. This is a **performance fusion** of two existing operators, not a new mathematical atom.

Proposed taxonomy:
- **Atoms** (8): Add, Welford, TiledAdd, Affine, Särkkä, Max/Min, ArgMin/ArgMax, SoftmaxWeighted
- **Compounds** (grows as needed): LogSumExp, (others as discovered)

The 8-operator model holds.

---

## Updated Sharing Surface Scorecard

| Family | Status | Sharing verification |
|--------|--------|---------------------|
| F06 | ✅ Complete | MomentStats, ExtremaStats, PHI expressions |
| F07 | ✅ Complete | **ZERO new GPU kernels** — pure extraction from F06 |
| F25 | ✅ Complete | Shares PHI_COUNT histogram kernel with F06 |
| F01 | ✅ Complete | DistancePairs via TiledEngine |
| F20 | 🔨 Partial | DBSCAN implemented, shares distance matrix via TamSession |
| F21 | 🔨 Partial | KNN implemented, consumes shared distance matrix |
| F16 | 📋 Designed | Navigator doc ready, scatter_multi_phi_weighted extension |
