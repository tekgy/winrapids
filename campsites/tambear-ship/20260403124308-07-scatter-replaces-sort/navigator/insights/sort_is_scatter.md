# Sort IS Scatter: The Proof Structure

## The Claim

Every sort in tambear can be expressed as `accumulate(scatter_to_buckets) + prefix_scan + gather`.
Therefore tambear needs exactly two primitives: accumulate and gather. No sort primitive.

## Why This Is True

Radix sort IS multi-pass scatter. Each pass:
1. **Scatter** elements to 256 buckets (one per byte value)
2. **Prefix scan** bucket counts to get output positions
3. **Gather** elements to their output positions

This is `accumulate(Bucketed(256), count) + prefix_scan + gather(permutation)`.
K passes for K-byte keys. For f64: 8 passes. For restricted ranges: fewer.

## The Sort Inventory (99 occurrences, 32 files)

### Category 1: Median/Quantile — SCATTER ELIMINATES SORT
- `descriptive.rs` sorted_nan_free → median/quantile/MAD
- `robust.rs` (14 sorts): medcouple, qn_scale, sn_scale, tau_scale, all via sorted_nan_free+median
- `nonparametric.rs`: bootstrap sort, KS test

**Scatter replacement**: Histogram scatter (accumulate to fixed bins) + prefix scan = approximate quantile in O(n). For exact: radix scatter to 2^16 bins → prefix scan → the k-th element is at the bin where cumulative count crosses k. One pass, no comparison sort needed.

### Category 2: Rank — SCATTER ELIMINATES SORT
- `nonparametric.rs`: Wilcoxon, Mann-Whitney rank computation
- Rank = prefix_scan(scatter(1, bucketed_by_value))

### Category 3: Selection (top-k) — SCATTER ELIMINATES SORT
- `knn.rs`: nearest neighbors
- `robust.rs`: LTS/LMS residual selection
- `survival.rs`: event time ordering

**Scatter replacement**: Scatter to buckets → partial prefix scan → only materialize top-k bucket.

### Category 4: Structural — SORT IS SCATTER IN DISGUISE
- `graph.rs`: Kruskal's edge ordering → radix sort = scatter
- `optimization.rs`: Nelder-Mead simplex → tiny n, irrelevant
- `tda.rs`: filtration → radix sort = scatter

## The Quantile Construction (exact, O(n))

Current code: `sorted_nan_free(data)` → `quantile(sorted, q)` → O(n log n).

Scatter replacement (exact):
```
1. min, max = moments_ungrouped(data).min, .max          // O(n), or free from MomentStats
2. buckets = accumulate(data, Bucketed(B, min, max), count) // O(n) scatter
3. cumsum = prefix_scan(buckets, sum)                      // O(B)
4. target_bucket = first where cumsum[b] >= q*n            // O(log B) binary search
5. recurse within target_bucket for exact answer           // O(n/B) filter + recurse
```

Total: O(n) per pass × O(log(range/eps)) passes. No comparison sort.
On GPU: each pass is massively parallel (scatter + scan = O(n/P)).

## The Radix Sort Identity

Comparison sort is O(n log n). Radix sort is O(n × k) where k = key bytes.
Radix sort IS multi-pass scatter:
```
for byte_pos in 0..8:  // 8 bytes for f64
    counts = accumulate(data, Bucketed(256, by=byte[byte_pos]), count)  // scatter
    offsets = prefix_scan(counts, sum)                                  // scan
    output = gather(data, offsets[bucket_of(key[byte_pos])]++)         // gather
```

8 passes × (scatter + scan + gather) = sort.
Even "real sort" is just accumulate + gather composed 8 times.

## The Punchline

Sort is not a third primitive. Sort is a *composition* of accumulate and gather:
```
sort(data) = gather(data, prefix_scan(accumulate(data, Bucketed(range), count)))
```

The two-operation principle holds. QED.

## 99 Sort Sites by Module

| Module | Count | Category | Scatter Path |
|--------|-------|----------|-------------|
| descriptive.rs | 7 | Median/Quantile | Histogram scatter |
| robust.rs | 14 | Median/Quantile | Via sorted_nan_free |
| nonparametric.rs | 9 | Rank + Bootstrap | Bucketed rank |
| tbs_executor.rs | 8 | Mixed | Dispatch to appropriate |
| equipartition.rs | 7 | Structural | Radix scatter |
| fold_irreversibility.rs | 4 | Structural | Radix scatter |
| spectral_gap.rs | 4 | Structural | Radix scatter |
| tda.rs | 4 | Structural | Radix scatter |
| survival.rs | 3 | Selection | Partial prefix |
| knn.rs | 3 | Selection | Partial prefix |
| rng.rs | 3 | Test only | N/A |
| complexity.rs | 3 | Structural | Radix scatter |
| Others (19 files) | 30 | Mixed | Case-by-case |
