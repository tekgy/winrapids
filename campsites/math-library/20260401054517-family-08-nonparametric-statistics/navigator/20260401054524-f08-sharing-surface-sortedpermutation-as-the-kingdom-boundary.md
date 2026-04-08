# F08 Sharing Surface: SortedPermutation as the Kingdom Boundary

Created: 2026-04-01T05:45:24-05:00
By: navigator

Prerequisite: F06 complete (MomentStats for some tests). SortedPermutation is the new primitive.

---

## The Structural Position of F08

Non-parametric statistics is where the polynomial MSR breaks down. All rank-based tests need ORDER STATISTICS — the sorted positions of data points. This is NOT a power sum. No scatter_multi_phi expression produces ranks.

F08 sits at the boundary between Kingdom A (commutative, polynomial) and Kingdom B (sequential). The sort itself is a one-time O(N log N) cost; everything after is pure extraction — O(N) scans over the sorted permutation.

**The principle**: Sort once. Deposit SortedPermutation in TamSession. Every rank-based test is an O(N) extraction.

---

## New MSR Type: SortedPermutation

```rust
/// The result of sorting a data column: indices that sort the data.
/// argsort(data)[i] = the original index of the i-th smallest value.
///
/// Deposited in TamSession after the first sort of a (DataId, column) pair.
/// All rank-based algorithms consume this — they never re-sort.
#[derive(Debug, Clone)]
pub struct SortedPermutation {
    pub n: usize,
    /// indices[i] = original position of the i-th smallest element.
    /// i.e., data[indices[i]] ≤ data[indices[i+1]] for all i.
    pub indices: Arc<Vec<u32>>,
    /// Ranks of original data: rank[j] = position of element j in sorted order.
    /// rank[indices[i]] = i. This is the inverse permutation.
    /// Stored alongside because most rank tests use it directly.
    pub ranks: Arc<Vec<f64>>,  // f64 for tie averaging
}

impl SortedPermutation {
    /// Average-rank tie correction: if k values tie, each gets rank (r + r+1 + ... + r+k-1)/k.
    pub fn rank_with_ties(data: &[f64]) -> SortedPermutation { ... }
}
```

Add to `IntermediateTag`:
```rust
SortedPermutation {
    data_id: DataId,
    column_id: DataId,  // which column was sorted
},
```

**Memory cost**: N × (u32 + f64) = N × 12 bytes. For N=1M: 12MB. Acceptable.

**GPU sort**: use parallel radix sort (via cuB's `DeviceRadixSort` on CUDA, or bitonic sort on WGSL). Output: indices. One additional O(N) scatter to build the inverse permutation (ranks).

---

## What Each Test Needs

### Tests That Need ONLY SortedPermutation (one sort → free)

**Wilcoxon Signed-Rank Test** (H₀: median = μ₀ for paired differences):
```
d_i = x_i - μ₀ (or x1_i - x2_i for paired)
Sort |d_i|, assign ranks. W+ = Σ(rank where d_i > 0). W- = Σ(rank where d_i < 0).
W = min(W+, W-). Compare to Wilcoxon table (or normal approximation for large n).
```
MSR: SortedPermutation of |d_i|. O(N) extraction.

**Mann-Whitney U Test** (H₀: stochastic equality of two distributions):
```
Pool x and y, sort combined array. For each x_i, count how many y_j < x_i.
U₁ = Σ_{i} (rank_i in pooled - i) [rank sum minus natural sequence]
    = R₁ - n₁(n₁+1)/2 where R₁ = sum of x ranks in pooled sort
```
MSR: SortedPermutation of pooled (x,y) array with group labels. O(N) rank-sum extraction.

**Kruskal-Wallis Test** (non-parametric ANOVA; H₀: all groups same distribution):
```
Pool all groups, sort. Compute R_g = rank sum for group g.
H = (12/(N(N+1))) · Σ_g R_g²/n_g - 3(N+1)
p = pchisq(H, df=k-1)
```
MSR: SortedPermutation of pooled data with group labels. Same permutation as Mann-Whitney.

**Friedman Test** (repeated measures non-parametric ANOVA):
```
For each subject i (row), rank the k treatments.
R_j = column rank sum for treatment j.
Q = 12/(nk(k+1)) · Σ_j R_j² - 3n(k+1)
```
MSR: SortedPermutation within each row (subject). Different from above — row-wise sort.

---

### Tests That Need SortedPermutation + QuantileSketch

**Sign Test** (H₀: median = μ₀):
```
S = number of x_i > μ₀. Under H₀, S ~ Binomial(n, 0.5).
```
Actually NO sort needed for sign test — just MomentStats(order=1, indicator=(x > μ₀)). But median estimation needs QuantileSketch.

**Runs Test** (H₀: sequence is random — tests for autocorrelation):
```
Dichotomize: each x_i gets +/- based on median.
Count runs (consecutive same-sign sequences).
Z = (R - (2n₁n₂/n + 1)) / std_R
```
MSR: QuantileSketch for median, then one-pass scan for run count. No full sort needed.

---

### Tests That Need FULL SORT but NOT SortedPermutation-based

**Kolmogorov-Smirnov Test** (H₀: data from distribution F₀; or two-sample):
```
One-sample: D = sup_x |F_n(x) - F₀(x)|
Two-sample: D = sup_x |F_n(x) - G_m(x)|
```
Requires computing the empirical CDF, which requires sorted data. SortedPermutation gives the ECDF directly: ECDF[i/n] = sorted_data[i].
MSR: SortedPermutation of x (or pooled x,y for two-sample).

**Anderson-Darling Test** (improved KS; more sensitive to tails):
```
A² = -n - (1/n) Σ_{i=1}^{n} (2i-1)[log(F₀(x_i)) + log(1 - F₀(x_{n+1-i}))]
where x_i are sorted observations
```
MSR: SortedPermutation. O(N) extraction after sort.

---

### Tests That Need Spearman/Kendall (rank correlations)

**Spearman's ρ** (rank correlation):
```
ρ = 1 - 6·Σd²/(n(n²-1)) where d_i = rank(x_i) - rank(y_i)
  = Pearson(rank(x), rank(y))  (exact equivalence)
```
MSR: SortedPermutation of x AND SortedPermutation of y → ranks → then MomentStats on rank vectors. Two sorts, then standard Pearson correlation extraction.

**Kendall's τ** (concordance measure):
```
τ = (C - D) / (n(n-1)/2)
where C = concordant pairs (both orders agree), D = discordant pairs
```
Requires counting concordant/discordant pairs — O(N log N) via merge sort or BIT (binary indexed tree). NOT simply extractable from SortedPermutation alone. Requires its own pass.

---

## The Sharing Tree for F08

**One SortedPermutation unlocks:**
- Wilcoxon signed-rank
- Mann-Whitney U
- Kruskal-Wallis (k groups)
- KS test (one and two sample)
- Anderson-Darling test
- Spearman ρ (from ranks array directly)

**Separate SortedPermutation (second sort) unlocks:**
- Friedman test (row-wise sort of a matrix)

**Needs its own O(N log N) pass:**
- Kendall's τ (concordance counting, merge sort trick)

**Needs QuantileSketch:**
- Sign test (median as threshold)
- Runs test (median for dichotomization)
- Bowley skewness (Q1, Q2, Q3)

---

## Tie Handling

Most non-parametric tests assume no ties. With ties, two strategies:

1. **Average ranks**: tied values get the mean of their ordinal ranks. Most common, supported by all R implementations.
2. **Mid-ranks**: same as average ranks.
3. **Min/max/random rank**: specialized use cases.

Tie-correction formulas exist for Wilcoxon, Mann-Whitney, Kruskal-Wallis. The T correction factor:
```
T = Σ_g (t_g³ - t_g)  where t_g = size of g-th tied group
```
This requires one additional O(N) scan over the sorted data to count tie groups.

---

## F08 Implementation Order

1. **GPU argsort** — either bitonic sort (N ≤ 2^20) or radix sort (unlimited). Produces SortedPermutation.
   - Phase 1: bitonic sort in WGSL/CUDA (already know how to do tiled kernels)
   - Phase 2: CUB radix sort for CUDA path (faster for large N)

2. **Rank assignment with ties** — O(N) scan over sorted indices to detect tie groups and assign average ranks. Produces SortedPermutation.ranks.

3. **Mann-Whitney U** — rank sum extraction. O(N) single pass over sorted combined array.

4. **Kruskal-Wallis** — same rank sum extraction with group labels.

5. **Wilcoxon signed-rank** — sort |d_i|, extract W+ and W-.

6. **KS test** — ECDF from SortedPermutation, one-pass computation of max deviation.

7. **Spearman ρ** — use ranks from step 2, then standard Pearson (from F06/F07 MomentStats).

8. **Anderson-Darling** — O(N) extraction from SortedPermutation.

9. **Kendall's τ** — separate merge-sort based algorithm (or O(N log N) BIT).

---

## Open Questions for Naturalist

1. Can Kendall's τ be expressed as an accumulate operation? There's a formulation using a modified merge sort — but is it expressible as `accumulate(Segmented, expr, op)`? This is the boundary test for the accumulate framework.

2. For Friedman test: the sort is row-wise (each row is a subject, columns are treatments). Is this a tiled sort? It seems like `accumulate(Tiled(n, k), argsort_row, identity)`. This is a new grouping pattern.

---

## The Structural Insight for Lab Notebook

> Non-parametric statistics is the family where the polynomial MSR breaks down — and that breakdown is INFORMATIVE. When you cannot compute a statistic from power sums, it means the statistic depends on ORDER STATISTICS, not just the distribution of values. This is the exact condition where ranks matter more than means. The SortedPermutation MSR type marks the "order statistics frontier" — the line where Kingdom A (commutative) methods end and Kingdom B (sequential) methods begin.

> The Spearman-Pearson connection is the rhyme: ρ_Spearman = r_Pearson applied to ranks. This means once you have ranks (from SortedPermutation), all of F07's correlation and regression infrastructure applies directly to ranked data. F08 adds exactly one new primitive (argsort) and inherits everything else.
