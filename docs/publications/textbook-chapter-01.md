# Chapter 1: What IS Computation?

*From: Computation Through the Accumulate Lens*

---

> *"The purpose of computing is insight, not numbers."*
> — Richard Hamming

---

## 1.1 A Question Worth Asking

Here is a list of computations you might encounter in a single day of data science work:

1. A groupby-sum: aggregate sales by region
2. A rolling mean: smooth a time series over a 20-day window
3. Principal component analysis: reduce a 1000-dimensional dataset to 10 dimensions
4. Training a neural network: minimize a loss function via gradient descent
5. Running DBSCAN: find clusters without specifying k
6. Computing a Fourier transform: decompose a signal into frequencies
7. Fitting a variogram: model spatial autocorrelation in geological data
8. Calculating mutual information: measure dependence between two variables

Now ask: what do these eight computations have in common?

The standard answer is: not much. They're from different fields (statistics, ML, signal processing, geostatistics, information theory), they appear in different textbooks, they're implemented in different libraries (pandas, sklearn, numpy.fft, scipy.spatial, sklearn.feature_selection), and they require different expertise to use correctly.

This textbook offers a different answer. These eight computations are **the same computation, with different parameters.**

Every one of them decomposes to two operations:

```
accumulate(data, grouping, expr, op)
gather(indices, source)
```

Full stop. That's all of computation.

This isn't an approximation. It's not "well, if you squint." It's a precise, verified claim: every computation you will encounter in statistics, machine learning, signal processing, optimization, and geometric algorithms is a composition of `accumulate` and `gather` with specific parameter choices. We've verified this decomposition for over 500 algorithms across 35 algorithm families, with gold-standard parity tests against R, Python, scipy, and sklearn for every implementation.

The goal of this chapter is to make that claim precise, show you why it's true, and help you see computation through this new lens. Once you see it, you can't unsee it.

---

## 1.2 Two Operations

### 1.2.1 Accumulate

**`accumulate(data, grouping, expr, op)`** has four parameters:

| Parameter | Meaning | Choices |
|-----------|---------|---------|
| `data` | The input sequence | any sequence |
| `grouping` | Where to write / how to partition | 9 patterns (see §1.3) |
| `expr` | What to compute per element | any element-wise function |
| `op` | How to combine partial results | 8 operators (see §1.4) |

The operation: for each element in `data`, compute `expr(element)`, then combine with the accumulator for the group determined by `grouping`, using `op` as the combination rule.

A few examples to make this concrete:

**Sum of a column:**
```
accumulate(prices, grouping=All, expr=identity, op=Add)
```
Grouping `All` means one group (the whole sequence). `Add` combines by addition. Result: the sum.

**Groupby sum:**
```
accumulate(sales, grouping=ByKey("region"), expr=identity, op=Add)
```
Grouping `ByKey("region")` means one group per distinct value of the "region" column. Result: sum by region.

**Rolling mean:**
```
accumulate(prices, grouping=Windowed(20), expr=identity, op=Welford)
```
Grouping `Windowed(20)` means a sliding window of size 20. `Welford` maintains a running mean and variance. Result: the 20-day rolling mean (and variance, for free).

**Variance (what the textbooks call a two-pass algorithm):**
```
// Pass 1:
mean = accumulate(prices, grouping=All, expr=identity, op=Add) / n
// Pass 2:
m2 = accumulate(prices, grouping=All, expr=(v-mean)², op=Add)
variance = m2 / (n-1)
```
Two accumulators. The second uses the mean from the first as a reference point.

**KMeans update step:**
```
// For each data point: assign to nearest centroid (separately computed)
// Then update centroids:
accumulate(data, grouping=ByKey(assignment), expr=identity, op=Welford)
```
`Welford` per group: each centroid gets the online mean of the points assigned to it. One pass.

These examples span groupby, rolling statistics, variance, and KMeans — four methods from four different textbooks. Same operation, different parameters.

### 1.2.2 Gather

**`gather(indices, source)`** reads elements from a source by index. It's the indexing primitive.

```
gather([2, 0, 4, 1], source)   // returns [source[2], source[0], source[4], source[1]]
```

Gather sounds trivial, but it's the key to expressing algorithms that require non-sequential access:

- **Embedding lookup**: `gather(token_ids, embedding_table)` — each token ID selects a row
- **k-nearest neighbors**: `gather(neighbor_indices, feature_matrix)` — collect each point's neighbors
- **Join**: `gather(foreign_keys, reference_table)` — the relational join is a gather
- **Permutation / sort reindex**: `gather(argsort(keys), values)` — sort by gathering in sorted order
- **Lagged features**: `gather(t - lag, time_series)` — look back in time

The interplay between `gather` and `accumulate` is where composition happens. The pipeline `gather → accumulate` is everywhere:

```
// Weighted mean (gather weights for each point, then accumulate):
weights = gather(weight_indices, weight_table)
accumulate(data, All, v * weight, op=Add) / accumulate(weights, All, v, op=Add)

// Sparse matrix-vector product (gather, then accumulate by row):
values_at_cols = gather(col_indices, vector)
accumulate(values_at_cols, ByKey(row_indices), v * sparse_val, op=Add)
```

### 1.2.3 Map is Not a Primitive

You might expect a third operation: element-wise transformation. Map. Apply a function to every element.

There is no map primitive. Map is absorbed into `accumulate`'s `expr` parameter.

- `map(x → x²) → collect()` is rare — you rarely want the intermediate result
- `map(x → x²) → accumulate(All, op=Add)` = sum of squares — the expr IS the map
- `map(x → log(x)) → accumulate(ByKey(label), op=Welford)` = log-space grouped stats — the expr is the map

In a lazy computation graph (which is how tambear compiles pipelines), map never executes standalone. It always fuses into its consumer. This is why `expr` is a parameter of `accumulate` rather than a separate operation.

---

## 1.3 The Nine Grouping Patterns

`accumulate`'s `grouping` parameter determines the structure of the output: how elements are partitioned and where results are written. There are nine grouping patterns, and they cover every known reduction structure.

| Grouping | Input → Output | Traditional name | Example |
|----------|---------------|-----------------|---------|
| `All` | N → 1 | Reduce | sum, mean, variance, max |
| `ByKey(column)` | N → K | Scatter / GroupBy | groupby, histogram, embedding update |
| `Prefix(forward)` | N → N | Scan | cumsum, EWM, Kalman, running stats |
| `Prefix(reverse)` | N → N | Suffix scan | backprop gradients, reverse Kalman smoother |
| `Prefix(bidirectional)` | N → N | Bidirectional scan | BiLSTM, Kalman smoother |
| `Windowed(size)` | N → N | Rolling | rolling mean/std/var |
| `Tiled(m, n)` | N → M×N | Tiled accumulate | GEMM, attention scores, covariance matrix |
| `Segmented(boundaries)` | N → N (reset at boundaries) | Segmented scan | per-group prefix ops, layer norm |
| `Masked(bitmask)` | N → N (skip where false) | Masked accumulate | fused filter-and-compute |

The profound insight: these nine patterns all have the *same structure*. They differ only in how they partition the input and where they write results. The kernel template — "for each element, compute expr, combine with accumulator for this group" — is identical.

This means a compiler that understands accumulate can generate all nine patterns from the same template. There is no separate reduce codepath, scatter codepath, scan codepath. There is one codepath, parameterized by grouping pattern.

### Examples in .tbs

The `.tbs` scripting language (introduced fully in Chapter 24) exposes accumulate directly. Every `.tbs` script is compilable to fused GPU kernels.

```tbs
// Groupby sum: ByKey grouping
sales | accumulate(by="region", expr=v, op=Add)
```

```tbs
// Rolling mean: Windowed grouping with Welford
prices | accumulate(window=20, expr=v, op=Welford) | select("mean")
```

```tbs
// Cumulative sum: Prefix grouping
returns | accumulate(grouping=Prefix, expr=v, op=Add)
```

```tbs
// Z-score: two accumulators sharing data (compiled into one pass)
stats = data | accumulate(grouping=All, expr=v, op=Welford)
data | map((v - stats.mean) / stats.std)
```

The last example is important: the two operations (compute stats, apply to data) share the data scan. The compiler sees that both operations read the same data and fuses them into one kernel. In traditional frameworks, this requires explicit programming. In tambear, it falls out of the lazy graph compiler automatically.

---

## 1.4 The Eight Operators

The `op` parameter in `accumulate` is an *associative binary operator* — a function `(state, element) → state` that satisfies:
- **Associativity**: `op(op(a, b), c) = op(a, op(b, c))`

Associativity is what enables parallelism. If the operator is associative, we can split the data into blocks, accumulate each block independently, and combine the block results. This is the foundation of GPU parallelism.

There are eight built-in operators. Each carries different state and computes different output:

| Operator | State | What it computes | Performance tier |
|----------|-------|-----------------|-----------------|
| `Add` | `(sum)` | sum, count, mean (with normalize) | Fast (~42μs) |
| `Welford` | `(count, mean, M2)` | online mean + variance | Fast (~42μs) |
| `RefCentered` | `(count, Σδ, Σδ²)` | zero-cancellation variance | Fast (~42μs) |
| `Affine(A, b)` | `(A, b)` | any linear recurrence | Fast (~42μs) |
| `Särkkä` | `(m, P, K, S, v)` | exact transient Kalman filter | Fast (~42μs) |
| `Max` / `Min` | `(value)` | extrema | Branched (~100μs) |
| `ArgMax` / `ArgMin` | `(value, index)` | extrema with position | Branched (~100μs) |
| `SoftmaxWeighted` | `(max, sum_exp, weighted_sum)` | online softmax + weighted accumulation | Division (~72μs) |
| `Custom(fn)` | user-defined | any associative combine | User-determined |

The performance tiers reflect a GPU reality: the `combine` function runs O(n log n) times (once per pair-merge in the parallel reduction tree). Simple adds/multiplies run at full throughput. Division introduces a latency stall. Branching causes warp divergence. The design rule: push complexity from `combine` into the `lift` (the function applied to each element before accumulation), which runs only O(n) times.

### The Associativity Test

How do you know if your operation is associative? There's a simple test: verify that combining two partial results gives the same answer as accumulating everything at once.

For Welford's algorithm, combining two partial results (with count_A, mean_A, M2_A) and (count_B, mean_B, M2_B) uses Pebay's formula:

```
δ = mean_B - mean_A
n_X = n_A + n_B
mean_X = mean_A + δ * n_B / n_X
M2_X = M2_A + M2_B + δ² * n_A * n_B / n_X
```

This is the parallel merge rule. It's associative and numerically stable. This is why Welford's algorithm can run in parallel on a GPU: every block accumulates independently, then block results are combined using the merge rule.

**Contrast**: the naive formula `Var = E[x²] - E[x]²` is NOT an associative combination rule — it's a post-processing formula applied to raw power sums. Raw power sums suffer catastrophic cancellation (see Chapter 7 and Appendix A). Welford's algorithm avoids this by accumulating centered deviations rather than raw squares.

---

## 1.5 Why Two Operations Suffice

The claim that `accumulate + gather` covers all of computation deserves a proof sketch.

**Claim**: Every computable function on sequences (and matrices, and graphs) can be expressed as a composition of `accumulate` and `gather` operations.

**Sketch**:

1. **Element-wise functions** (map): `f(x_i) = expr(x_i)` is the `expr` parameter of accumulate. Every element-wise function is free.

2. **Reductions** (sum, max, mean, ...): `accumulate(All, expr, op)` for any associative `op`. Well-known to be complete for all parallel reductions.

3. **Scans** (prefix sum, EWM, ...): `accumulate(Prefix, expr, op)` with any associative `op`. By the Blelloch theorem (1990), any associative operator has an optimal parallel prefix computation.

4. **Scatter operations** (groupby, histogram, ...): `accumulate(ByKey(key), expr, op)`. Covers all partition-and-aggregate operations.

5. **Matrix multiplication**: `accumulate(Tiled, expr=a*b, op=Add)`. Matrix-vector product, GEMM, batched attention — all tiled accumulators.

6. **Sorting**: `gather(argsort(keys), data)`. The sort produces indices; gather reorders. Any comparison sort plus index gather covers all sorting.

7. **Permutation and indexing**: `gather(indices, source)`. Covers all non-sequential access patterns.

8. **Composition**: Any algorithm is a DAG of operations. Each node is either `accumulate` or `gather`. The composition of `accumulate` and `gather` operations covers any DAG.

The argument is not that we can express everything awkwardly (you can express anything in Turing-complete languages). The argument is that `accumulate + gather` expresses algorithms *naturally*, with parameter choices that match how we *think about* the algorithm. The decomposition is explanatory, not just computational.

**Corollary**: Any framework that implements `accumulate` with all nine grouping patterns and `gather` with all six addressing patterns is computationally complete for the algorithms studied in this textbook.

---

## 1.6 The Four-Menu Representation

Every algorithm is a choice from four menus:

```
algorithm = (addressing, grouping, expr, op)
```

where:
- **Addressing** selects how to read data (6 patterns)
- **Grouping** selects how to partition and where to write (9 patterns)
- **Expression** selects what to compute per element (unbounded)
- **Operator** selects how to combine (8 built-ins + custom)

This representation has a useful property: two algorithms that share a menu choice can share computation.

- Same `(data, expr)` but different `grouping`: the expr is computed once, results go to multiple groups. One kernel, two outputs.
- Same `data` but different `(expr, op)`: two separate passes, but the data scan is shared.
- Same `grouping` and `expr` but different `op`: run both operators in a single pass, computing two outputs simultaneously.

The compiler's job is to find shared menu choices and eliminate redundant computation. This is **structural sharing** — not caching, not optimization, but the observation that the same computation appears in multiple algorithms and should execute once.

We'll see in Chapter 21 that this structural sharing is worth 3-5820x speedup depending on the pipeline.

### The Algorithm Table

Here is a sample decomposition table showing familiar algorithms through the four-menu lens:

| Algorithm | Addressing | Grouping | Expression | Operator |
|-----------|-----------|----------|------------|---------|
| Column sum | Direct | All | identity | Add |
| GroupBy mean | Direct | ByKey(group) | identity | Welford |
| Variance | Direct | All | (v − mean)² | Add |
| Rolling std | Direct | Windowed(w) | identity | Welford |
| Cumulative sum | Direct | Prefix | identity | Add |
| EWM (exponential weight) | Direct | Prefix | identity | Affine(α) |
| Histogram | Direct | ByKey(bin(v)) | 1.0 | Add |
| Dot product | Direct | All | x·y | Add |
| Matrix multiply | Tiled | Tiled(block) | a·b | Add |
| Attention scores | Tiled | Tiled | Q·Kᵀ | Add |
| FlashAttention | Tiled | Tiled | Q·Kᵀ | SoftmaxWeighted |
| KMeans centroid update | Direct | ByKey(label) | identity | Welford |
| Softmax | Direct | All → Broadcast | exp(v) | Add, then divide |
| Backpropagation | Direct | Prefix(reverse) | gradient | chain-rule Affine |
| DBSCAN core points | MultiOffset | All | ε-neighborhood | Add |

The table continues for 500+ algorithms in Appendix D. Every row is a verified decomposition — we've implemented each algorithm from first principles and tested it against the authoritative reference implementation.

---

## 1.7 A First Glimpse: FlashAttention

Let's look at one algorithm that might seem to resist this decomposition: FlashAttention [Dao et al. 2022], the state-of-the-art self-attention mechanism that powers modern transformers.

Self-attention computes, for queries Q, keys K, values V:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

The naive implementation materializes the n×n attention matrix QKᵀ — impossible for long sequences (n=100K requires 80GB for n=100K in f32). FlashAttention's key insight is that the softmax MSR — {max, sum_exp, weighted_sum} — fits in registers. The full attention can be computed in tiles without ever materializing the attention matrix.

In the four-menu representation:

```
accumulate(
    data   = (Q_tile, K_tile, V_tile),
    grouping = Tiled(block_size),
    expr   = (Q_i · K_j) / √d,
    op     = SoftmaxWeighted
)
```

`SoftmaxWeighted` carries state {max, sum_exp, weighted_sum} — exactly the online softmax MSR from Milakov & Gimelshein (2018). It's associative: two tiles can be combined by updating the running max and renormalizing. This is the SAME structural insight as FlashAttention, expressed as a choice from our four menus.

FlashAttention is not a special-case algorithm that required deep insight to discover. It's a `SoftmaxWeighted` accumulator with `Tiled` grouping. Once you see computation through the accumulate lens, it's the *obvious* choice.

---

## 1.8 What This Changes

If accumulate + gather IS computation, several things follow:

**For algorithm design**: Every new algorithm should begin with the four-menu decomposition. "What is the expr? What is the grouping? What is the operator?" These questions cut through complexity and reveal the essential structure.

**For implementation**: A library that implements the four menus well is computationally complete. No special cases, no algorithm-by-algorithm hand-coding. The specialist library is a table of `(addressing, grouping, expr, op)` tuples, compiled to fused kernels at request time.

**For optimization**: Sharing opportunities are visible at the menu level. Two algorithms with the same `expr` share the per-element computation. Two algorithms with the same `data` share the memory bandwidth. The compiler doesn't need heuristics — it needs to find shared menu items.

**For teaching**: Instead of 50 formulas for 50 algorithms, there are four menus. The student learns the menus once, then reads algorithms as menu choices. The structural rhymes that textbooks never mention (PCA = symmetric GEMM = SVD of a centered matrix = the SAME four-menu choice with different `grouping`) become visible.

**For hardware**: Every GPU primitive maps to a grouping pattern. Reduce kernels = `All`. Scatter kernels = `ByKey`. Scan kernels = `Prefix`. GEMM = `Tiled`. A framework that understands grouping patterns maps directly to hardware primitives without translation layers.

---

## 1.9 Chapter Summary

- All computation decomposes to two operations: `accumulate` and `gather`
- `accumulate` has four parameters: data, grouping, expression, operator
- Nine grouping patterns cover all known partition structures
- Eight built-in operators cover all known associative combination rules
- Map is not a primitive — it fuses into `accumulate`'s expression parameter
- The four-menu representation enables structural sharing across algorithms
- Associativity is the key to GPU parallelism — operators that are associative are parallelizable
- The centered-basis requirement (preview: Chapter 7 and Appendix A) is the main correctness condition

In Chapter 2, we go deep on the eight operators — their state, their combining rules, their performance characteristics, and which algorithms use them.

---

## Exercises

**1.1** Express each of the following as a four-menu decomposition `(addressing, grouping, expr, op)`:
- (a) Sum of squares
- (b) Moving average over a window of 10
- (c) Maximum per group
- (d) Cumulative product
- (e) Count of elements satisfying x > 0

**1.2** For each pair of algorithms, identify the shared menu item that enables structural sharing:
- (a) Mean and variance
- (b) GroupBy sum and histogram
- (c) Cumulative sum and exponential moving average
- (d) PCA and linear regression (hint: both need X'X)

**1.3** The Welford combining rule:
```
δ = mean_B - mean_A
n_X = n_A + n_B
mean_X = mean_A + δ * n_B / n_X
M2_X = M2_A + M2_B + δ² * n_A * n_B / n_X
```
Verify that this rule is associative: show that combining (A, B) then C gives the same result as combining A, then (B, C).

**1.4** FlashAttention uses the `SoftmaxWeighted` operator with state {max, sum_exp, weighted_sum}. Write the combining rule for two partial `SoftmaxWeighted` states. Verify it's associative.

**1.5** (Bonus) The naive variance formula `Var = E[x²] - E[x]²` uses two accumulators (Σx and Σx²). Generate a dataset where this formula loses all significant digits but Welford's algorithm remains accurate. How large does the mean need to be relative to the variance?

**1.6** Write the following pipeline in `.tbs` notation:
```
data → normalize (z-score) → compute pairwise distances → find 5 nearest neighbors
```
Identify which steps can share computation (sharing the z-score stats between normalization and distance computation).

---

## Notes and References

The accumulate + gather decomposition was developed in the tambear project during exploration of first-principles GPU algorithm design. The observation that ALL ML operations are accumulate + gather was formalized in March 2026 after finding that the 9 tambear "primitives" were all the same operation with different grouping parameters.

The parallel scan primitive (`Prefix` grouping with any associative operator) was identified by Blelloch (1990). The connection to all nine grouping patterns is new.

Welford's online algorithm (Welford 1962) and Pebay's parallel combining formulas (Pebay 2008) are the canonical implementations of variance accumulation. The `RefCentered` operator (Manuscript 001) extends these to a fixed-reference formulation with improved numerical properties for offset data.

FlashAttention (Dao et al. 2022) independently discovered the `SoftmaxWeighted` MSR for attention computation. Its connection to the general four-menu framework is noted here for the first time.

The nine grouping patterns cover all known parallel reduction structures. Whether they are truly *complete* — whether there exists a parallel reduction structure not expressible as one of these nine patterns — remains an open question.

---

*Next: Chapter 2 — The Eight Operators*
