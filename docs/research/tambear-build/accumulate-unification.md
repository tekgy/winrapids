# The Accumulate Unification

**Date**: 2026-03-31
**Status**: Architecture specification — discovered during first-principles decomposition of ML primitives

---

## The Core Insight

ALL computation primitives collapse to TWO fundamental operations:

1. **`accumulate(data, grouping, expr, op)`** — THE computation primitive
2. **`gather(indices, source)`** — THE read primitive

Every ML algorithm, every DataFrame operation, every signal processing pipeline is a composition of these two operations with different parameter choices.

---

## Accumulate Grouping Patterns

The "9 primitives" (reduce, scatter, scan, tiled_accumulate, etc.) are the SAME operation with different grouping:

| Grouping | Input → Output | What it was called | Example |
|----------|---------------|-------------------|---------|
| `All` | N → 1 | reduce | sum, mean, max, variance |
| `ByKey(column)` | N → K | scatter | groupby, histogram, embedding update |
| `Prefix(forward)` | N → N | scan | cumsum, EWM, Kalman, running stats |
| `Prefix(reverse)` | N → N | suffix scan | backprop gradients, reverse Kalman |
| `Prefix(bidirectional)` | N → N | bidirectional scan | BiLSTM, Kalman smoother |
| `Windowed(size)` | N → N | rolling | rolling mean/std (prefix subtraction trick) |
| `Tiled(m, n)` | N → M×N | tiled_accumulate | GEMM, attention scores, covariance |
| `Segmented(boundaries)` | N → N (reset at boundaries) | segmented scan | per-group prefix ops |
| `Masked(bitmask)` | N → N (skip where false) | masked accumulate | filter fused into computation |

### Key: fused_expr (map) is NOT a primitive

In a lazy pipeline, element-wise operations NEVER execute alone. They ALWAYS fuse into their consumer:

- `fused_expr → reduce` = expr computed INSIDE the reduce kernel
- `fused_expr → scatter` = expr computed INSIDE the scatter kernel
- `fused_expr → scan` = expr computed INSIDE the scan kernel
- `fused_expr → fused_expr` = CHAIN into one kernel
- `fused_expr → collect()` = only time it materializes standalone

Map is absorbed into accumulate's `expr` parameter. It's not a separate primitive.

---

## Accumulate Operators (the AssociativeOp)

| Operator | State | What it computes | Used by |
|----------|-------|-----------------|---------|
| `Add` | `(sum)` | sum, count, mean | Everywhere |
| `Welford` | `(count, mean, M2)` | online mean + variance | Batch norm, layer norm, running stats |
| `RefCentered` | `(count, sum_delta, sum_delta_sq)` | zero-division variance | Better Welford (Manuscript 001) |
| `Affine(A, b)` | `(A, b)` | any linear recurrence | EWM, Kalman, ARIMA, SSM |
| `Särkkä` | `(m, P, K, S, v)` | exact transient Kalman | Kalman from step 1 at machine epsilon |
| `Max` / `Min` | `(value)` | extrema | Max pooling, beam search |
| `ArgMax` / `ArgMin` | `(value, index)` | extrema with position | Assignment, top-k, selection |
| `SoftmaxWeighted` | `(max, sum_exp, weighted_sum)` | online softmax + weighted accumulation | FlashAttention |
| `Custom(user_fn)` | user-defined | any associative combine | User extensions |

### Performance tiers (measured):
- **Fast tier (~42μs)**: adds/muls only in combine (Add, Affine)
- **Division tier (~72μs)**: division in combine
- **Branch tier (~100μs+)**: branching in combine (Max, ArgMax)

**Design rule**: push complexity from combine (runs O(n log n) times) to lift/constructor (runs once).

---

## Gather Addressing Patterns

| Addressing | What it does | Example |
|-----------|-------------|---------|
| `Direct(indices)` | Simple index lookup | Embedding, join, permutation |
| `Strided(offset, stride)` | Regular pattern | Shift, downsample |
| `MultiOffset(offsets)` | Multiple offsets per position | local_context (gather at [-10,-5,-1,0,+1,+5,+10]) |
| `Broadcast(scalar)` | Repeat one value to all positions | Scalar division after reduce |
| `Masked(indices, mask)` | Gather only where mask is true | Filtered gather |
| `Tiled(tile_indices)` | Gather tiles for blocked ops | Tiled accumulate input staging |

---

## The Universal Composition

Every ML operation is:

```
result = accumulate(
    gather(source, addressing),  // HOW to read
    grouping,                     // WHERE to write / how to partition
    expr,                         // WHAT to compute per element
    op,                           // HOW to combine
)
```

A choice from **four menus**: addressing × grouping × expr × op.

### ML Algorithm Decompositions

| Algorithm | Gather | Grouping | Expr | Op |
|-----------|--------|----------|------|----|
| **KMeans distance** | Direct | All (per point) | `(a-b)²` | Add |
| **KMeans assign** | Direct | All (per point) | identity | ArgMin |
| **KMeans update** | Direct | ByKey(label) | identity | Welford |
| **GroupBy sum** | identity | ByKey("ticker") | `price*qty` | Add |
| **Rolling std** | identity | Windowed(20) | identity | Welford |
| **GEMM** | Tiled | Tiled(32,32) | `a*b` | Add |
| **Attention scores** | Tiled | Tiled | `Q*K` | Add |
| **Softmax** | identity | All (per row) | `exp(x)` | Add → then broadcast divide |
| **FlashAttention** | Tiled | Tiled | `Q*K` | SoftmaxWeighted |
| **Embedding lookup** | Direct(token_ids) | — | — | — (pure gather) |
| **Backprop** | identity | Prefix(reverse) | gradient | ChainRule |
| **Batch norm** | identity | ByKey(batch) | identity | Welford |
| **Layer norm** | identity | Segmented(layer_boundaries) | identity | Welford |
| **Random Forest split** | identity | ByKey(bin) | identity | Add (histogram) |
| **Linear regression** | identity | Tiled | `x*x` | Add (X'X normal equations) |

### The Sharing Insight

Two accumulates with the **same data and same expr** but different grouping patterns SHARE the data loading and expr evaluation:

```python
# These share the expr evaluation (price*qty computed ONCE):
total = accumulate(data, All, price*qty, Add)           # reduce
by_ticker = accumulate(data, ByKey("ticker"), price*qty, Add)  # scatter
```

The compiler detects shared `(data, expr)` pairs and fuses them into a single kernel with multiple accumulation targets.

---

## What This Means for tambear

### The Specialist Library IS the Menu Combinations

Each "specialist" (rolling_zscore, VWAP, KMeans, etc.) is a RECIPE — a specific choice from the four menus. The specialist registry is literally a table of (addressing, grouping, expr, op) tuples.

### The Compiler Sees ONE Operation

Instead of routing to different kernel generators for reduce vs scatter vs scan, the compiler generates accumulate kernels parameterized by grouping pattern. This unifies the codegen path.

### Fusion Becomes Trivial

Two adjacent accumulates with shared inputs → fuse. The grouping pattern determines the kernel structure; the expr and op are just parameters plugged in. Cross-algorithm fusion falls out for free.

---

## Implementation Gaps (as of 2026-03-31)

| Gap | What it unlocks | Effort | Status |
|-----|----------------|--------|--------|
| Distance as composed accumulate | KMeans, KNN, clustering, kernels | LOW — reduce + fused_expr exist | Not built |
| Softmax as reduce-broadcast-divide | Attention, classification | MEDIUM — needs broadcast pattern | Not built |
| Tiled accumulate GPU launch | GEMM, neural nets, attention, PCA | MEDIUM — trait exists, needs launch | Not built |
| Multi-target fused accumulate | Shared expr across groupings | MEDIUM — compiler optimization | Not built |
| Tambear clustering (density-based) | No k, no iteration, discovers structure | MEDIUM — uses local_context + scan | Not built |
| Unified accumulate API | Single entry point for all groupings | LOW — dispatch wrapper | Not built |
| ArgMin/ArgMax | Assignment, beam search, TopK | IN PROGRESS | Navigator claimed |

---

## The Tambear Clustering Algorithm (first principles)

Traditional KMeans assumptions questioned:

1. **"Requires iteration"** → What if we discover structure in ONE PASS via density?
2. **"Distance is O(n × k × d)"** → What if tile headers eliminate most distance computations?
3. **"Centroids are points"** → What if centroids are distributions (RefCenteredStatsOp)?
4. **"We need k"** → What if Tam discovers k from density structure?
5. **"Clustering is separate from the pipeline"** → What if `.kmeans()` is a pipeline node that composes and fuses?

Tambear clustering = `local_context → reduce(density) → fused_expr(mask cores) → scan(MinLabelOp)` — four primitives, one pass each, no iteration, no k.

---

## Open Questions

1. Can the compiler AUTO-DETECT when two accumulates share (data, expr) and fuse them?
2. Is SoftmaxWeighted the right operator for FlashAttention, or does it need a separate Tiled+Online pattern?
3. What's the right API surface? `tb.accumulate(...)` directly, or named methods that compile to accumulate?
4. Can Prefix(bidirectional) be a single kernel or must it be two passes?
5. What other grouping patterns exist beyond the 8 listed?
