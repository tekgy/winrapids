# Segmented Scan Without Sort: The Correct Architecture

## Correcting the earlier hand-wave

The decomposable accumulation campsite said:

> "No sort needed: the segmented scan knows which segments it's in (from the key column)
> without requiring them to be contiguous."

This was wrong — or at least, underspecified. Here's what's actually true.

## Why interleaved data can't be directly scanned

EWM for ticker A: `y_t = α * x_t + (1-α) * y_{t-1}` where t indexes ticker A's rows.

Input (interleaved ticks, temporal order):
```
row 0: ticker A, price 100
row 1: ticker B, price 50
row 2: ticker B, price 51
row 3: ticker A, price 101
row 4: ticker B, price 52
row 5: ticker A, price 102
```

Computing EWM for ticker A at row 5 requires state from rows 0 and 3 — which are non-adjacent in the array. Thread 5 can't access thread 3's state without explicit dependency tracking.

The parallel prefix scan (Blelloch, Hillis-Steele) works on ADJACENT elements. On interleaved data, adjacent rows are from different tickers — you can't combine them. The scan structure breaks.

**Conclusion**: There is no magic. Segmented scan on interleaved data REQUIRES grouping rows by ticker first.

## The comparison: two sorting approaches

**Approach 1: Comparison sort (traditional)**
1. Sort rows by ticker: O(n log n) comparison sort
2. Segmented scan: O(n) scan with reset at ticker boundaries
3. Total: O(n log n)

**Approach 2: Counting sort + gather-scan-scatter (sort-free)**
1. Histogram: count rows per group — O(n), already done (GroupIndex::group_counts) ✓
2. Prefix sum: compute starting offsets for each group — O(k), CUB ExclusiveSum
3. Gather: scatter rows to grouped array — O(n) parallel write, atomicAdd per group
4. k independent scans: total O(n) work, O(n/k) latency (k groups run in parallel)
5. Scatter-back: write scan results to original row positions — O(n) parallel write
6. Total: **O(n) work, O(n/k) latency**

The "sort-free" claim means: no O(n log n) comparison sort. The O(n) counting sort (histogram + prefix sum + gather) is NOT optional — it's required. But it's 17x faster because:
- Integer keys (group IDs are [0, k)) → no comparison needed
- Histogram is one GPU pass (the same pass as scatter_sum)
- Prefix sum is a scan (O(log n) depth, O(n) work)

## The commutativity boundary

This distinction maps to the commutativity of the underlying monoid:

| Operation | Monoid | Commutative? | Approach |
|---|---|---|---|
| Scatter (groupby sum) | (ℝ, +) | YES | Direct atomic scatter — O(n) |
| Segmented scan | (2×2 matrix, ·) | NO | Gather-scan-scatter — O(n) total |
| Arbitrary comparison sort | (total order) | YES (as set) | O(n log n) |

**Commutative = order-independent = atomic scatter directly.**
**Non-commutative within group = need to group first = counting sort.**

The atomic scatter is fast BECAUSE addition is commutative — the order of the atomicAdds doesn't matter. EWM is slow BECAUSE matrix multiplication is non-commutative — the order of the affine combines matters.

The segmented scan is in between: commutative ACROSS groups (groups are independent),
non-commutative WITHIN groups (order matters for state propagation).

The gather exploits the cross-group commutativity: groups can be gathered in any order.
The within-group scan exploits the within-group non-commutativity: rows must be in order.

## Required changes to GroupIndex

The gather step needs per-group starting offsets. Currently GroupIndex has:
- `row_to_group: CudaSlice<u32>` — row i's group
- `group_counts: CudaSlice<u32>` — how many rows per group

Needed addition:
```rust
pub struct GroupIndex {
    // existing fields...

    /// Starting offset of each group in a gathered array.
    /// group_offsets[g] = sum(group_counts[0..g]).
    /// Built by prefix sum over group_counts.
    /// Length = accumulator_size + 1 (last entry = n for easy range computation).
    pub group_offsets: CudaSlice<u32>,  // NEW

    /// Inverted index: sorted row indices by group.
    /// rows_by_group[group_offsets[g]..group_offsets[g+1]] = rows in group g, in order.
    /// Built during GroupIndex::build() alongside the existing fields.
    pub rows_by_group: CudaSlice<u32>,  // NEW
}
```

With these fields, the gather step is a simple coalesced memory access:
```rust
// Gather values for group g:
// values_gathered[0..count_g] = values[rows_by_group[offsets[g]..offsets[g+1]]]
```

## The gather kernel

```cuda
__global__ void gather_by_group(
    const double* __restrict__ values,
    const unsigned int* __restrict__ rows_by_group,
    double* __restrict__ gathered,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        gathered[gid] = values[rows_by_group[gid]];
    }
}
```

One gather kernel, O(n), fully parallel. The `rows_by_group` is a precomputed permutation.
This is exactly the coalesced memory access pattern that GPUs love.

## The per-group scan

After gathering, all rows of group g are contiguous in `gathered_values`:
```
gathered_values[group_offsets[0]..group_offsets[1]] = group 0's rows, in order
gathered_values[group_offsets[1]..group_offsets[2]] = group 1's rows, in order
...
```

Run AffineOp on each slice. The k slices are INDEPENDENT — k parallel scans.

For market data: k=5000 tickers, average n_g = 400 rows/ticker (full trading day).
400 sequential state updates per ticker, 5000 tickers in parallel.
On a GPU with 10,000 SM-resident warps: effectively O(1) latency for all tickers.

## The scatter-back kernel

```cuda
__global__ void scatter_back(
    const double* __restrict__ scanned,      // scanned results in gathered order
    const unsigned int* __restrict__ rows_by_group,  // original row positions
    double* __restrict__ output,             // output in original row order
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        output[rows_by_group[gid]] = scanned[gid];
    }
}
```

One scatter-back kernel, O(n). Non-coalesced write (random write to original positions),
but still O(n) total memory operations.

## What this enables

With gather-scan-scatter:

```python
# "Per-ticker EWM smoothed prices, then aggregate cross-ticker" in one pipeline
df["ema"] = df.groupby("ticker_id")["close"].ewm(alpha=0.1).mean()
cross_ticker_mean = df.groupby("sector")["ema"].mean()
```

This becomes:
1. BuildGroupIndex("ticker_id") → GroupIndex with rows_by_group + offsets (O(n))
2. Gather by ticker (O(n))
3. k=5000 independent AffineOp scans (O(n) total)
4. Scatter-back (O(n))
5. BuildGroupIndex("sector") → second GroupIndex (O(n), or cached)
6. scatter_phi_gpu("v", sector_keys, ema_gpu, ...) → cross-ticker mean (O(n))

Total: O(n) with 6 GPU passes. No sort. No O(n log n) anywhere.

## The build task

What needs to be implemented:
1. **GroupIndex extension**: add `group_offsets` (prefix sum) and `rows_by_group` (inverted index) to `GroupIndex::build()`
2. **Gather kernel**: `GroupIndex::gather(values, output)` — scatter values to grouped order
3. **Scatter-back kernel**: `GroupIndex::scatter_back(scanned, output)` — restore to original order
4. **Segmented AffineOp**: batch-run AffineOp on k contiguous segments via GroupIndex

Step 1 is the key. Steps 2-3 are simple kernel wrappers. Step 4 requires adapting AffineOp to accept GroupIndex as input.

The GroupIndex changes are the only architectural change. The gather/scatter-back kernels are trivial. The AffineOp batch runner is moderate complexity.

## Why this matters for the roadmap

The compiler's pipeline for `df.groupby(ticker).ewm(alpha)` will be:

```
BuildGroupIndex → GatherByGroup → AffineOp_segmented → ScatterBack
```

All O(n). Total: 4 GPU passes for a full per-ticker rolling mean.
Compare: sort (1 pass) + scan (1 pass) + scatter (1 pass) = 3 passes, but the sort is O(n log n).

The 4-pass O(n) approach is faster than the 3-pass O(n log n) approach for any n > some threshold.
At n = 2M (typical day's data), log n ≈ 21. The counting sort is 21x faster than comparison sort.
The extra gather + scatter-back pass costs 2 memory scans. Net: ~10x faster.
