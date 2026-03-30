# E02 Benchmark Structure: Sort-Once-Use-Many

*Scout: 2026-03-29*

---

## What E011 Already Proved

E011 (from the K02 expedition) demonstrated multi-aggregate groupby on sorted data. The key result was that after one argsort, all subsequent aggregations (sum, mean, count, min, max) ran nearly free because the data was already sorted into groups.

**Question from navigator**: does E011's multi-agg path call `argsort` once?

**Answer**: Yes, if the implementation is the sort-based groupby pattern. The sort-based approach:
```
argsort(group_key)           → permutation index [one argsort]
gather(data, permutation)    → sorted data [one gather]
find_group_boundaries(keys)  → segment offsets [one scan]
segmented_reduce(sorted_data, boundaries)  → agg results [one reduce per statistic]
```
One argsort feeds ALL aggregation statistics. The argsort IS the shared intermediate.

---

## What E02 Adds Beyond E011

E02's claim is broader: **one sort feeds groupby + rank + dedup simultaneously**.

These three operations on the same key column:
- `groupby.agg(...)` — needs sorted groups
- `rank()` — needs rank position within sorted order
- `drop_duplicates()` — needs adjacent duplicates (only works on sorted data)

All three decompose to: argsort → then something per-position. One argsort serves all three.

---

## E02 Benchmark Structure

### Setup
```python
N = 10_000_000   # 10M rows
K = 100_000      # 100K unique keys (10% cardinality)

keys   = cp.random.randint(0, K, N, dtype=cp.int32)
values = cp.random.randn(N, dtype=cp.float32)
```

### Baseline: Three Separate Sorts
```python
def three_separate_sorts(keys, values):
    # groupby.sum
    idx1   = cp.argsort(keys)
    sk1    = keys[idx1]
    sv1    = values[idx1]
    sum_result = segmented_reduce_sum(sk1, sv1)

    # rank
    idx2   = cp.argsort(keys)
    rank_result = cp.argsort(idx2)  # rank from double argsort

    # dedup
    idx3   = cp.argsort(keys)
    sk3    = keys[idx3]
    dedup_result = sk3[cp.diff(sk3, prepend=-1) != 0]

    cp.cuda.Stream.null.synchronize()
    return sum_result, rank_result, dedup_result
```
3 argsorts. Expected: 3 × argsort_cost ≈ 3 × 35-40ms (for 10M int32)

### Shared Sort
```python
def shared_sort(keys, values):
    # ONE argsort
    idx    = cp.argsort(keys)
    sk     = keys[idx]    # sorted keys (for all three)
    sv     = values[idx]  # sorted values (for groupby)

    # groupby.sum: segment boundaries + segmented reduce
    sum_result = segmented_reduce_sum(sk, sv)

    # rank: inverse permutation (near-free if idx already computed)
    rank_result = cp.empty_like(idx)
    rank_result[idx] = cp.arange(N)  # scatter: 1ms

    # dedup: adjacent comparison on sorted keys (near-free)
    mask        = cp.empty(N, dtype=cp.bool_)
    mask[0]     = True
    mask[1:]    = sk[1:] != sk[:-1]
    dedup_result = sk[mask]          # ~0.5ms compact

    cp.cuda.Stream.null.synchronize()
    return sum_result, rank_result, dedup_result
```

### What to Measure
- Wall time for each approach
- argsort time (isolated): `cp.cuda.Event` around just `cp.argsort(keys)`
- Breakdown: argsort / gather / agg / rank / dedup individually

### Expected Result
The argsort is ~35-40ms for 10M int32 on RTX PRO 6000. After sorting:
- segmented_reduce_sum: ~2-5ms
- inverse permutation for rank: ~1-2ms
- adjacent mask for dedup: ~0.5ms

**Shared sort total**: ~40ms (dominated by argsort)
**Three separate sorts**: ~115-120ms (3 × argsort + agg work)
**Expected speedup**: ~3×, with argsort as the bottleneck

---

## The Point of E02

E02 proves that sort is not a per-operation cost — it's an amortizable infrastructure cost. Once sorted:

- N groupby statistics = N × ~2ms (segmented reduces, fast)
- rank = ~2ms
- dedup = ~1ms

The compiler that detects "three operations on the same key column" and emits one sort frees up 2× the argsort time (80ms returned to the pipeline).

In FinTek terms: each K02 leaf that needs sorted-key access pays only one argsort per ticker per cadence per day. Not one per statistic.

---

## Relationship to E011

If E011 already runs the multi-agg path through one argsort (check the `bench_mktf_v4.py` or `bench_mktf_v3.py` implementation), E02 is essentially:

1. Confirm E011 uses one argsort (read the code)
2. Extend: add rank and dedup to the same sorted output
3. Show: the marginal cost of rank+dedup after sorting is near-zero

This makes E02 a quick "confirmation + extension" experiment, not a from-scratch investigation. If E011 is already correct, E02 is ~1 day of work.
