# Three-Kingdom Taxonomy and Revised MSR Type Hierarchy

Created: 2026-04-01T05:32:51-05:00
By: navigator
Session: Day 1 team synthesis

---

## Discovery: Three Mathematical Kingdoms

Emerged from naturalist's cross-family analysis. Organizes all 35 families.

### Kingdom A — Commutative (GramMatrix / power sums)

Families: 06, 07, 10, 14, 20, 21, 22, 25, 33

Defining property: computation depends only on SUMS of products. Order of data doesn't matter. Parallel decomposition is trivially correct. The central MSR type is **GramMatrix**.

The polynomial theorem (naturalist): for polynomial extraction functions, MSR = power sums up to max degree.
- Degree 1: {n, Σx} → mean, VWAP
- Degree 2 in x: {n, Σx, Σx²} → variance, z-score, t-test, batch norm
- Degree 2 in (x,y): {n, Σx, Σy, Σx², Σy², Σxy} → Pearson r, OLS, L2 distance
- Degree 3: add Σx³ → Fisher skewness
- Degree 4: add Σx⁴ → kurtosis
- Degree 2 in p×p: GramMatrix → PCA, factor analysis, linear regression, canonical correlation

**Key sharing insight**: L2 distance and Pearson correlation share accumulator fields (both degree-2 in two variables). L2 distance is derivable from GramMatrix:
```
D[i,j] = ||xᵢ - xⱼ||² = ||xᵢ||² - 2⟨xᵢ,xⱼ⟩ + ||xⱼ||²
        = GramMatrix[i,i] - 2·DotProduct[i,j] + GramMatrix[j,j]
```
If GramMatrix (tiled accumulate) is computed, DistancePairs is FREE — no second GPU pass.

**8 operators, Kingdom A uses**: Add, TiledAdd

### Kingdom B — Sequential (Affine scan)

Families: 17, 18, 13, 23, 24

Defining property: each element depends on the prior element's accumulator. Not embarrassingly parallel — but the scan structure allows parallel prefix with O(log N) depth.

Algorithms: AR models, GARCH, EWM, Kalman filter, backprop, RNN/LSTM hidden state
All use: `accumulate(Prefix(forward), identity, Affine(Aₜ, bₜ))`
where Aₜ, bₜ may vary per timestep (time-varying Affine).

**Key insight**: these are ALL the same Affine scan operator, differently parameterized. Kalman with A=I = Recursive Least Squares. ARIMA is Affine with polynomial A matrix. Backprop is Affine (reverse scan) with A = Jacobian.

**8 operators, Kingdom B uses**: Affine, Särkkä (specialized Kalman), SoftmaxWeighted (attention)

### Kingdom C — Iterative (outer loop around Kingdom A)

Families: 05, 09, 16, 20(iter), 34

Defining property: outer optimization loop where each iteration IS a Kingdom A computation. The hard algorithms are easy algorithms repeated.

- KMeans: iteration over (distance + argmin + scatter_mean)
- EM/GMM: iteration over (scatter_multi_phi for M-step + tiled for E-step)
- IRLS (robust regression, M-estimators): iteration over weighted least squares
- HMC/NUTS (Bayesian sampling): iteration over gradient (= backprop = Kingdom B)
- Stochastic optimization: iteration over gradient descent (= scatter_multi_phi)

**EM M-step IS scatter_multi_phi** (naturalist finding):
```
N_k = scatter_multi_phi(keys=soft_assign, phi="1.0")        // counts
μ_k = scatter_multi_phi(keys=soft_assign, phi="v")           // sums
Σ_k = scatter_multi_phi(keys=soft_assign, phi="(v-r)*(v-r)") // sum_sq
```
One kernel, one pass. M-step shares IDENTICAL accumulator fields with Family 06 grouped variance. Family 16 is implementable with zero new infrastructure.

---

## Revised MSR Type Hierarchy

**15 types total** (original 9 + 6 from naturalist).

### Level 1 — Raw Accumulations (expensive, O(N) or O(N²))

**Kingdom A types:**
1. `MomentStats` — {n, Σx, Σx², ..., Σx^k} per group. Central type for F06, F07, F09.
2. `ExtremaStats` — {max, min} per group. Simple but not polynomial (MinMax op).
3. `WeightedMomentStats` — {Σw, Σ(wx), Σ(wx²)} per group. For WLS, weighted tests.
4. `GramMatrix` — {Σxᵢxⱼ for all i,j} + {Σxᵢyⱼ}. Central type for F10, F14, F22, F33. SUBSUMES BivariateMomentStats (p=1 special case) and the basis for DistancePairs.
5. `CrosstabStats` — {n_ij joint counts}. For chi-square, LCA.
6. `QuantileSketch` — T-Digest or DDSketch. For median, IQR, quantiles. OUTSIDE polynomial MSR.

**Kingdom B type:**
7. `AffineState` — scan output for sequential/recurrent algorithms. Holds (state, covariance) at each timestep for Kalman/AR/EWM. NOT a power sum — requires Affine scan operator.

**Cross-kingdom types:**
8. `RankStats` / `SortedPermutation` — argsort + rank vector. Required by F08 (non-parametric) and F09 (rank-based robust). Gather prerequisite, outside polynomial MSR.
9. `GraphStructure` — adjacency list, Laplacian, degree sequence. For F29 (graph algorithms). No polynomial analogue.

### Level 2 — Derived Extractions (cheaper, O(K²) or O(p³))

10. `EigenDecomposition` — eigenvectors/eigenvalues from GramMatrix. Shared by PCA (F22), Factor Analysis (F14), spectral clustering (F20), graph spectral embedding (F29).
11. `FittedModel` — {β̂, ŷ, residuals, σ², R²}. From GramMatrix + Cholesky solve. Shared by diagnostic algorithms (F10), robust starting points (F09), SEM measurement model (F14).
12. `FrequencyDomain` — FFT output. Consumed by spectral analysis (F19), Welch PSD, coherence, spectral entropy (F25).
13. `DistancePairs` — pairwise distances. DERIVED from GramMatrix for L2 (free). Independent accumulate for other metrics (Manhattan, Cosine). Shared by F20, F21, F22, F28, F30.
14. `ClusterAssignment` — soft or hard cluster labels. Produced by KMeans/DBSCAN, consumed by silhouette, within-group stats, EM E-step.
15. `KernelMatrix` — K[i,j] = kernel(xᵢ, xⱼ). For kernel SVM (F21), kernel PCA (F22), GP regression (F31). Computed via tiled accumulate with kernel expr.

---

## The Spine (3 passes unlock 25/35 families)

**Pass 1 — Moments** O(N): `scatter_multi_phi([x, x², x³, x⁴, 1], All)` + MinMax + QuantileSketch
→ Deposits MomentStats(order=4), ExtremaStats, QuantileSketch
→ Unlocks: F06, F07, F08 (partial), F09 (partial), F25 (partial)
→ Cost: one memory pass over data. Should AUTO-RUN on `tb.session(data).prime()`

**Pass 2 — Cross-products** O(N×p²): `accumulate(Tiled(p,p), xᵢxⱼ, Add)`
→ Deposits GramMatrix
→ Unlocks: F02 (partial), F10, F14, F22, F33, and via EigenDecomp: F14, F22
→ Cost: O(N×p²). For large p, expensive. Run lazily when needed.

**Pass 3 — Distances** O(N²×d): `accumulate(Tiled(N,N), (aᵢ-bⱼ)², Add)` OR derive from GramMatrix
→ Deposits DistancePairs
→ Unlocks: F01, F20, F21, F28, F30
→ Cost: O(N²). NEVER auto-run. User must explicitly request.

Key insight: Pass 3 from GramMatrix = Pass 2 + O(N²) element-wise arithmetic. For p >> 1 (many features), computing GramMatrix first and deriving distances is faster than direct O(N²×d) tiled distance.

---

## Structural Rhymes (selected from naturalist's 11)

These represent algorithms that are literally the same computation, extracted differently:

| Rhyme | Family A | Family B | Shared computation |
|-------|----------|----------|-------------------|
| ANOVA = Regression F-test | F07 | F10 | Same GramMatrix + F-statistic extraction |
| KMeans E-step = KNN query | F20 | F21 | Same tiled distance + argmin |
| Kalman = Recursive LS | F17 | F10 | Same Affine scan with A=I |
| Spectral clustering = Graph Laplacian eigenmap | F20 | F29 | Identical pipeline |
| EM M-step = Grouped variance | F16 | F06 | Same scatter_multi_phi |
| Random forest split = Mutual information | F21 | F25 | Same histogram→entropy |

**Navigator implication**: these rhymes are implementation hints, not coincidences. When building Family X, check whether a rhyming Family Y is already done — the implementation is the same accumulate with different last-mile math.

---

## Production Bug Log

**Critical bugs found by adversarial (2026-04-01):**

1. `intermediates.rs::SufficientStatistics::variance()` — naive formula `sum_sq/n - (sum/n)²`. Breaks at offset ~1e4 for kurtosis, ~1e8 for variance in f64. FIX: replace with centered formula once RefCenteredStats is implemented.

2. `hash_scatter.rs` — same naive variance. Status: scope under investigation by adversarial.

3. Third critical — pending adversarial report.

**Fix strategy**: RefCenteredStats (implementing now) is the correct long-term solution. Immediate fix for SufficientStatistics::variance(): add centering using the existing mean as reference before we have the full RefCentered infrastructure.

---

## Open Design Question

**Should DistancePairs be Level 1 or Level 2?**

If most consumers use L2 distance AND GramMatrix is already computed (from F10/F22 work), then DistancePairs = Level 2 derived, and the O(N²×d) direct computation is just a fallback for the non-L2 case.

But for pure distance-only workflows (no regression, no PCA), GramMatrix may not be computed, and DistancePairs is Level 1.

**Resolution**: DistancePairs has two backends — `DerivedFromGramMatrix` (O(N²) element-wise, fast) and `DirectTiled` (O(N²×d), accurate for non-L2 metrics). TamSession prefers the derived path when GramMatrix is available.
