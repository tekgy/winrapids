# MSR Types and Sharing Tree — 35-Family Math Library

Created: 2026-04-01T05:14:29-05:00
By: navigator

---

## The Core Question

What flows between algorithms through TamSession? What does each family deposit, and what does it consume? This is the sharing surface — the contracts that let the compiler eliminate redundant computation across algorithm boundaries.

---

## MSR Types

These are the typed intermediates in the TamSession marketplace. Every algorithm declares which it produces and which it consumes.

### MomentStats

```
MomentStats(order: u8, group: GroupKey)
  n:     u64    // count
  sum1:  f64    // Σx
  sum2:  f64    // Σx²
  sum3:  f64    // Σx³   (order ≥ 3)
  sum4:  f64    // Σx⁴   (order ≥ 4)
  sum5:  f64    // Σx⁵   (order ≥ 5)
  ...
  sum8:  f64    // Σx⁸   (order ≥ 8)
```

accumulate decomposition: `accumulate(All|ByKey, [x, x², x³, ..., x^order, 1], Add)`

Produced by: F06 (any descriptive stat computation)
Consumed by: F06, F07 (t-tests, ANOVA), F09 (trimmed moments), F10 (OLS diagonal), F11 (REML), F25 (entropy from moments)

### ExtremaStats

```
ExtremaStats(group: GroupKey)
  max:  f64
  min:  f64
```

accumulate decomposition: `accumulate(All|ByKey, identity, MinMax)` — single pass, custom operator
(MinMax is a new fused operator: state = (max, min), combine = (max(a,b), min(a,b)))

Produced by: F06 (any min/max/range computation)
Consumed by: F06, F08 (range normalization for KS), F20 (bounding box for init), F09 (winsorization bounds)

### WeightedMomentStats

```
WeightedMomentStats(order: u8, group: GroupKey)
  n:       u64
  sum_w:   f64   // Σw
  sum_wx:  f64   // Σ(w·x)
  sum_wx2: f64   // Σ(w·x²)
```

accumulate decomposition: `accumulate(All|ByKey, [w, w·x, w·x², 1], Add)`

Produced by: F06 (weighted mean/variance)
Consumed by: F10 (WLS), F07 (weighted t-tests), F06 (weighted stats)

### BivariateMomentStats

```
BivariateMomentStats(col_x: ColId, col_y: ColId, group: GroupKey)
  n:    u64
  sx:   f64    // Σx
  sy:   f64    // Σy
  sx2:  f64    // Σx²
  sy2:  f64    // Σy²
  sxy:  f64    // Σxy
```

accumulate decomposition: `accumulate(All|ByKey, [x, y, x², y², xy, 1], Add)`

Produced by: F10 (regression, computes covariance terms anyway)
Consumed by: F10 (OLS bivariate), F07 (t-test for correlation, F-test), F33 (CCA)

### RankStats

```
RankStats(col: ColId, group: GroupKey)
  ranks:          Vec<f64>   // dense ranks, ties averaged
  rank_sum:       f64        // Σ rank
  tie_correction: f64        // correction factor for tied ranks
```

Requires sort + scan — non-trivial parallel decomposition.
Ranks themselves are the MSR; individual tests consume them.

Produced by: F08 (any rank-based computation)
Consumed by: F08 (Mann-Whitney, Wilcoxon, Kruskal-Wallis, Spearman, Kendall)

### QuantileSketch

```
QuantileSketch(col: ColId, group: GroupKey)
  digest: TDigest   // mergeable quantile sketch, ~1% error
```

accumulate decomposition: `accumulate(All|ByKey, identity, TDigestMerge)`
TDigestMerge: new AssociativeOp — associative, approximate, O(1) state per centroid.

Produced by: F06 (median, IQR, percentile computation)
Consumed by: F06 (all quantile ops), F08 (KS test CDF), F09 (trimmed/winsorized bounds), F31

### DistancePairs

```
DistancePairs(metric: DistanceMetric, group: GroupKey)
  // Small n: Materialized n×n
  // Large n: ApproximateANN (HNSW)
  pairs: DistanceStructure
```

accumulate decomposition: `accumulate(Tiled(n,n), gather(Tiled), expr=(a-b)², Add)` per metric
This is the tiled accumulate gap from accumulate-unification.md — must be filled first.

Produced by: F01 (any distance computation)
Consumed by: F20 (KMeans, DBSCAN, hierarchical), F21 (KNN, SVM), F22 (t-SNE, UMAP, MDS), F28, F30

### GramMatrix

```
GramMatrix(col_ids: Vec<ColId>)
  XtX: p×p matrix   // X'X for p predictors
  Xty: p vector      // X'y
  yty: f64           // y'y
```

accumulate decomposition: `accumulate(Tiled(p,p), [xᵢ·xⱼ], Add)` — normal equations

Produced by: F10 (OLS normal equations)
Consumed by: F10 (all linear models), F02 (least squares), F22 (PCA from covariance)

### CrosstabStats

```
CrosstabStats(col_a: ColId, col_b: ColId)
  n_ij: HashMap<(A,B), u64>
  n_i:  HashMap<A, u64>
  n_j:  HashMap<B, u64>
  n:    u64
```

accumulate decomposition: `accumulate(ByKey((a,b)), identity, Add)` on indicator variable

Produced by: F07 (chi-square computation)
Consumed by: F07 (chi-square variants, McNemar), F16 (LCA)

---

## The Sharing Tree

```
RAW DATA
├── MomentStats(order=2) ──► mean, variance, std, SE
│   ├──► F07: one-sample t, two-sample t, ANOVA between-group
│   ├──► F10: OLS residual variance, R², F-stat
│   └──► F09: Winsorized mean (consumes bounds from ExtremaStats too)
│
├── MomentStats(order=4) ──► skewness, kurtosis
│   ├──► F06: all shape statistics
│   ├──► F07: D'Agostino-Pearson normality test
│   └──► F08: Jarque-Bera test
│
├── MomentStats(order=8) ──► moments 5-8
│   └──► F06: complete moment family, L-moment flavors
│
├── ExtremaStats ──► min, max, range
│   ├──► F06: range, CV denominator check
│   ├──► F08: KS statistic (range normalization)
│   └──► F09: winsorization clip bounds
│
├── QuantileSketch ──► approximate quantiles at any p
│   ├──► F06: median, IQR, quartiles, deciles, percentiles
│   ├──► F08: quantile-based non-parametric tests
│   └──► F09: trimmed mean (needs p% and (100-p)% quantiles)
│
├── RankStats ──► dense ranks with tie correction
│   └──► F08: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Spearman, Kendall
│
├── BivariateMomentStats ──► covariance terms
│   ├──► F10: OLS bivariate
│   └──► F33: CCA (scaled up to GramMatrix)
│
├── GramMatrix ──► X'X, X'y, y'y
│   ├──► F10: OLS, WLS, ridge, lasso, elastic net
│   ├──► F02: least squares via Cholesky on X'X
│   └──► F22: PCA (covariance = X'X/n - x̄x̄ᵀ)
│
├── CrosstabStats ──► joint/marginal counts
│   ├──► F07: chi-square (all variants)
│   └──► F16: LCA cross-tabs
│
└── DistancePairs ──► pairwise distances
    ├──► F20: KMeans (L2), DBSCAN (any), hierarchical
    ├──► F21: KNN (any metric), kernel SVM
    ├──► F22: t-SNE, UMAP, MDS, Isomap
    ├──► F28: Riemannian geodesics
    └──► F30: spatial autocorrelation (Moran's I)
```

---

## Key Design Decisions

### D1: Types are named and typed, not shaped

The compiler matches by semantic type (MomentStats, DistancePairs), not by tensor shape. Two f64 arrays with the same shape are NOT interchangeable if one is distances and one is variances. The type system is the sharing contract.

### D2: QuantileSketch is approximate

Exact quantiles require sorted data — O(n log n), not accumulate-friendly. T-Digest gives 1% error in O(n) time and is associative (mergeable across partitions). When exact quantiles are required, declare `requires_exact_quantiles = true` — tambear falls back to sort + indexed scan. Most downstream algorithms (median, IQR, percentiles for EDA) are fine with 1% approximation.

### D3: DistancePairs is polymorphic

Small n: full n×n materialized. Large n: ANN index (HNSW). Consumers declare what they accept. KMeans works with ANN (assignment step only needs nearest centroid, not all pairwise). Exact hierarchical clustering needs full matrix. The type carries this.

### D4: Anti-YAGNI on MomentStats order

When computing mean (needs order=1), we deposit MomentStats(order=2) for free — Σx² costs one multiply in the already-open accumulate kernel. When computing variance (needs order=2), we deposit order=4. The marginal cost of extra powers is one multiply each; the sharing value is potentially eliminating entire downstream accumulate passes.

**Rule**: always deposit one order higher than required.

### D5: GramMatrix subsumes BivariateMomentStats

For p predictors, GramMatrix(p×p) contains all bivariate pairs. So BivariateMomentStats is a specialization for p=1 (simple linear regression / Pearson r). Compiler optimization: if GramMatrix(p) exists, BivariateMomentStats for any column pair within those p columns is free by extraction.

---

## Optimal MSR for Family 06 (full accumulate)

Single pass, 11 scalar fields + 1 quantile sketch = entire descriptive family:

| Field | What it gives |
|-------|--------------|
| n | count, sample size |
| Σx | mean |
| Σx² | variance, std |
| Σx³ | Fisher skewness (g₁) |
| Σx⁴ | excess kurtosis (g₂) |
| Σx⁵ | 5th standardized moment |
| Σx⁶ | 6th standardized moment |
| Σx⁷ | 7th standardized moment |
| Σx⁸ | 8th standardized moment |
| max | maximum, range upper bound |
| min | minimum, range lower bound |
| TDigest | median, IQR, ANY percentile |

The Σx⁵...Σx⁸ additions are almost free: x is in register, one multiply each. The sketch is the expensive part (centroid management), but it's O(1) amortized.

---

## Open Questions for the Team

1. **math-researcher**: L-moment formulas — they're NOT equivalent to moment-based skewness/kurtosis. Hosking 1990. We need both implementations. Do they share the same MSR fields or need separate accumulators?

2. **adversarial**: What breaks T-Digest? Adversarial quantile sketching is real. Near-sorted data can concentrate centroids. Test: uniform, sorted, reverse-sorted, adversarial (data designed to maximize T-Digest error).

3. **naturalist**: The sharing tree has a clear root (MomentStats). Does it have a clear "spine" — a single accumulate pass that produces enough MSR to unblock most of the 35 families? If so, should that be the first thing tambear runs on any new dataset?

4. **pathmaker**: What's the Rust type for TamSession? A HashMap<TypeId, Box<dyn Any>>? Or something more structured? The compiler needs to inspect it statically at pipeline planning time.

5. **navigator (me)**: TamSession type registry design — next doc.
