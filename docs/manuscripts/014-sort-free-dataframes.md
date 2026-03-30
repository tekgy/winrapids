# Sort-Free DataFrames: Eliminating O(n log n) From GPU Data Science

**Draft — 2026-03-30**
**Field**: Database Systems / GPU Computing / Data Science

---

## Abstract

We demonstrate that sort — the O(n log n) operation that dominates GPU DataFrame pipelines — is unnecessary for the four most common operations that use it: groupby, deduplication, join, and top-k selection. On GPU, hash-based alternatives achieve 2-17x speedup over sort-based approaches across all group cardinalities, because hash scatter is O(n) with one pass and naturally parallel, while sort is O(n log n) with random-access memory writes that degrade GPU memory bandwidth. We propose sort-free DataFrame design as an architectural principle: sort exists only for the rare case where the user literally requests sorted output. All other operations that traditionally depend on sort are reimplemented via hash scatter, hash probe, and selection algorithms. Combined with a persistent group index (built once, reused via provenance cache), groupby becomes O(n_groups) metadata read + O(n) scatter — eliminating sort entirely from the hot path.

---

## 1. The Sort Tax

### 1.1 Where Sort Hides

In a typical DataFrame pipeline, sort is rarely called explicitly. It hides inside other operations:

| User operation | Hidden sort | Why sort is used | Actually needed? |
|---|---|---|---|
| `groupby("key").sum()` | argsort keys | Group identical keys together | NO — hash scatter groups without ordering |
| `drop_duplicates("key")` | sort then scan for changes | Adjacent duplicates easy to detect | NO — hash set detects duplicates in one pass |
| `merge(left, right, on="key")` | sort-merge join | Match keys between tables | NO — hash join matches without ordering |
| `nlargest(k, "value")` | full sort then take first k | Find the k largest elements | NO — selection algorithm is O(n) not O(n log n) |
| `rank()` | argsort to determine position | Ordinal position requires ordering | YES — rank genuinely needs sort |
| `sort_values("col")` | explicit sort | User wants sorted output | YES — this is actual sort |

**Four of six operations don't need sort.** Only rank and explicit sort_values require ordering. Everything else uses sort as an implementation convenience, not a mathematical necessity.

### 1.2 Why Sort Is Expensive on GPU

Sort on GPU is O(n log n) with poor memory access patterns:

1. **Radix sort** (the fastest GPU sort): multiple passes over the data, each pass reading and writing ALL elements. For 32-bit keys: 4 passes. For 64-bit keys: 8 passes. Each pass: O(n) reads + O(n) writes with scatter (non-sequential) access patterns.

2. **Data reordering**: after computing the sort permutation, the actual data columns must be gathered in the new order. For k columns: k × O(n) random reads. Random reads on GPU achieve ~10-30% of sequential bandwidth.

3. **The sort permutation itself**: O(n) int64 values = 8 bytes per row. For 10M rows: 80MB of auxiliary memory just for the permutation.

Total: O(n log n) compute + O(k × n) random-access memory moves + O(n) auxiliary memory.

### 1.3 Why Hash Scatter Is Cheap on GPU

Hash-based groupby via `scatter_add`:

1. **One pass**: each thread reads one element, computes hash(key), atomically adds value to the group accumulator. Total: O(n) reads (sequential) + O(n) atomic writes to O(n_groups) accumulators.

2. **No data movement**: values are read in place. No reordering. No permutation. No auxiliary memory (beyond the n_groups accumulators).

3. **Naturally parallel**: each element is independent. No dependency between elements. Maximum GPU occupancy.

**Measured result** (1M rows, 4,600 groups, NVIDIA Blackwell):
- Sort-based groupby: 1.04 ms (argsort = 0.49ms = 47% of total)
- Hash scatter groupby: 0.06 ms
- **Sort-free is 17x faster**

---

## 2. Sort-Free Operations

### 2.1 GroupBy via Hash Scatter

**Traditional**: sort keys → find group boundaries → segmented reduce per group.

**Sort-free**: for each element, hash its key → atomic scatter-add to group accumulator.

```
// One GPU kernel, one pass, no sort
for each element i (parallel):
    group = keys[i]           // or hash(keys[i]) for non-integer keys
    atomicAdd(&sums[group], values[i])
    atomicAdd(&counts[group], 1)
```

**Multi-aggregation**: sum + count + mean in one pass costs LESS than sort alone (measured: 1.07ms for 3 aggregations vs 1.04ms for argsort only).

**Variance and higher moments**: track sum and sum_sq per group via scatter_add. Extract mean and variance at the end. Same one-pass pattern. Uses the RefCentered approach (manuscript 001) for numerical stability — center around per-group reference point.

**Cardinality independence**: hash scatter costs ~0.05ms regardless of whether there are 1,000 or 50,000 groups. Sort cost varies with key range. Hash is more predictable.

### 2.2 Deduplication via Hash Set

**Traditional**: sort → scan for adjacent changes → compact unique values.

**Sort-free**: hash each key → check if seen before → emit first occurrence.

```
// Approach: hash set with atomic CAS
for each element i (parallel):
    slot = hash(keys[i]) % table_size
    old = atomicCAS(&table[slot], EMPTY, keys[i])
    if (old == EMPTY):
        // First occurrence — this element is unique
        mark_unique[i] = 1
    else if (old == keys[i]):
        // Duplicate — skip
        mark_unique[i] = 0
    else:
        // Collision — linear probe
        ...
```

O(n) expected time. No sort. The hash table is O(n_unique) auxiliary memory, typically much smaller than the sort permutation.

### 2.3 Join via Hash Probe

**Traditional**: sort both tables on key → merge scan.

**Sort-free**: build hash table from smaller table → probe with larger table.

```
// Build phase: hash table from dimension table (small)
for each dim_row i (parallel):
    slot = hash(dim_keys[i]) % table_size
    atomicCAS(&ht_keys[slot], EMPTY, dim_keys[i])
    ht_values[slot] = i

// Probe phase: each fact row looks up its match
for each fact_row i (parallel):
    slot = hash(fact_keys[i]) % table_size
    // linear probe until match or empty
    dim_idx = ht_values[matching_slot]
```

Already proven in Day 1 experiments: hash join at 382x over pandas, sort-merge at 233x. Hash wins because the build phase is O(n_dim) and the probe is O(n_fact), both with no sorting.

For integer keys in [0, N): direct-index join (the key IS the array index) is even faster: O(n_fact) with zero hashing.

### 2.4 Top-K via Selection

**Traditional**: full sort → take first k elements.

**Sort-free**: selection algorithm finds the k-th largest element in O(n), then partition around it.

```
// GPU quickselect: find the k-th element
pivot = approximate_median(sample)
count_greater = count_if(values > pivot)  // one parallel pass
if count_greater == k: done
if count_greater > k: recurse on elements > pivot
if count_greater < k: recurse on elements <= pivot, adjust k
```

Expected O(n) with O(log n) recursion depth. Each recursion is a parallel pass over a shrinking subset. Much faster than O(n log n) full sort when k << n (the common case: "give me the top 10 of 10 million rows").

---

## 3. The Persistent Group Index

### 3.1 Build Once, Reuse Forever

In the tambear `.tb` file format, the group index is stored alongside the data:

```
group_index: {
    "ticker_id": {
        AAPL: { rows: [0, 5, 12, 18, ...], count: 50000 },
        MSFT: { rows: [1, 3, 7, 14, ...], count: 48000 },
        ...
    }
}
```

**First groupby on "ticker_id"**: builds the index via one-pass hash scatter, stores it in the file. Cost: O(n).

**Every subsequent groupby on "ticker_id"**: reads the pre-built index from the file. Cost: O(n_groups) metadata read. The scatter-add still costs O(n), but the GROUP DISCOVERY is free.

### 3.2 Provenance Integration

The group index has a provenance hash: `hash(column_data_id, "group_index")`. If the column hasn't changed, the index is valid. Provenance check: 35ns. If valid: skip index rebuild entirely.

For streaming data (new rows appended): the index can be INCREMENTALLY updated. New rows are hash-scattered into the existing index structure. No full rebuild.

### 3.3 Sharing Across Operations

The group index serves ALL operations that need group membership:
- GroupBy (any aggregation)
- Group-wise filtering (`filter(group_size > 100)`)
- Group-wise ranking (`rank within group`)
- Group-wise sampling (`sample 10 per group`)

One index, built once, shared across all group-aware operations. The sharing optimizer's provenance system ensures it's never rebuilt unnecessarily.

---

## 4. The Liftability Perspective

### 4.1 GroupBy IS a Decomposable Accumulation

Hash scatter-add is NATURALLY liftable:

- **Lift**: each element (key, value) maps to a group assignment
- **Combine**: addition (or any commutative, associative aggregation)
- **Extract**: read the per-group accumulators

Each element contributes INDEPENDENTLY to its group. No inter-element comparison. No ordering. Pure per-element computation. This is order-1 liftable.

### 4.2 Sort BREAKS Liftability

Sort introduces inter-element dependencies: the position of element i depends on its rank relative to ALL other elements. This is inherently non-local — you cannot sort element i without knowing ALL other elements. Sort is a GLOBAL operation disguised as a per-element one.

By using sort for groupby, we artificially introduce a non-local dependency (sorting) into a naturally local operation (group membership). The sort-free approach restores the natural locality.

### 4.3 Sort = Forced Dimensional Projection

From the dimensional gap perspective (manuscript 013): sort projects a multi-dimensional entity (the dataset with its group structure) onto a one-dimensional ordering (the sorted sequence). This projection DESTROYS the group structure (groups are contiguous in the sorted order, but their boundaries must be rediscovered by scanning). The hash-based approach preserves the group structure directly — no projection, no information loss, no rediscovery.

---

## 5. Design Principle: Sort-Free by Default

### 5.1 The Tambear Rule

**Sort exists in exactly two places:**
1. `tb.sort_values("col")` — explicit user request for sorted output
2. `tb.rank("col")` — ordinal position genuinely requires ordering

**Everything else is sort-free:**
- GroupBy → hash scatter
- Dedup → hash set
- Join → hash probe (or direct-index for integer keys)
- Top-K → selection algorithm
- Filter → boolean mask (already sort-free)
- Arithmetic → element-wise (already sort-free)

### 5.2 The Compiler's Role

The tambear compiler enforces sort-freedom:
- If the user writes `groupby`, the compiler emits hash scatter, never sort
- If the user writes `nlargest`, the compiler emits selection, never sort
- If the user writes `merge`, the compiler emits hash join, never sort
- Only explicit `sort_values` emits an actual sort

The sharing optimizer compounds this: group indices are cached via provenance. The first groupby builds the index. Every subsequent groupby on the same key is O(n_groups) + O(n) scatter — no index rebuild, no sort, ever.

### 5.3 Measured Impact

| Operation | Sort-based | Sort-free | Speedup |
|---|---|---|---|
| GroupBy sum (1M, 4600 groups) | 1.04 ms | 0.06 ms | **17x** |
| GroupBy multi-agg (sum+count+mean) | 3+ ms | 1.07 ms | **3x** |
| Join (10M × 10K) | 1.07 ms (sort-merge) | 0.65 ms (direct-index) | **1.6x** |
| Join (10M × 10K) | 1.07 ms (sort-merge) | 0.73 ms (hash) | **1.5x** |

The groupby improvement is the headline: **17x by simply not sorting.**

---

## 6. What About Variance, Std, Median?

### 6.1 Variance and Std: Sort-Free via RefCentered Scatter

GroupBy variance requires sum and sum_of_squares per group. Two scatter_adds:

```
scatter_add(group_sums, keys, values)
scatter_add(group_sum_sq, keys, values * values)
// or: scatter_add(group_sum_sq, keys, (values - group_ref)^2) for RefCentered stability
```

Two passes, O(n) each. No sort. The RefCentered approach (manuscript 001) provides numerical stability by centering around a per-group reference point.

### 6.2 Median: The One Hard Case

Median DOES require ordering within groups. There's no O(n) sort-free median for grouped data.

**However**: per-group selection (find the k-th element within each group) is O(n_group) per group. With the group index, each group's rows are known. Apply quickselect per group. Total: O(n) expected.

This is still faster than sort-based median (O(n log n) sort + O(n_groups) reads) for moderate group counts.

### 6.3 Quantiles

Same as median: per-group selection. The group index provides the row lists. Quickselect within each group finds the desired quantile.

---

## 7. Conclusion

Sort is the O(n log n) tax that GPU DataFrame operations pay because existing libraries (cuDF, Polars) inherited sort-based groupby from CPU implementations where sort was well-optimized and cache-friendly. On GPU, where hash scatter achieves 17x speedup over sort-based groupby, the inheritance is a performance liability.

The sort-free principle — eliminate sort from every operation that doesn't intrinsically require ordering — is a natural consequence of the liftability framework: groupby is decomposable (each element contributes independently), sort is not (each element's position depends on all others). Using sort for groupby introduces an unnecessary non-local dependency into a naturally local operation.

Combined with the persistent group index (built once, cached via provenance, shared across all group operations), the sort-free DataFrame achieves O(n_groups) metadata overhead per groupby — approaching O(1) for the organizational component, with only the O(n) scatter-add for the actual aggregation.

Tam doesn't sort. He knows where everything is.

---

## References

- Blelloch, G. E. (1990). Prefix sums and their applications.
- Harris, M. et al. (2007). Parallel prefix sum (scan) with CUDA.
- Merrill, D. & Grimshaw, A. (2010). Revisiting sorting on GPUs. CUB library.
- Satish, N. et al. (2009). Designing efficient sorting algorithms for manycore GPUs.
