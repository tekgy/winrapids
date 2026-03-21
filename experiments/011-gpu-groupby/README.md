# Experiment 011: GPU GroupBy

## Purpose

Fill the core gap identified in Day 1: "You can launch CUDA kernels on Windows today. You just can't do `df.groupby('category').sum()` on a GPU DataFrame."

This experiment builds GPU groupby from scratch, tests two algorithms across a wide cardinality range, and establishes the dispatch strategy for production.

## Files

- `gpu_groupby.py` — Sort-based and hash-based groupby implementations. Benchmarks at 100, 10K, 1M groups. Multi-aggregation (sum+mean+count).

## Algorithms

### Sort-Based
1. `cp.argsort(keys)` — radix sort, gets permutation
2. Reorder values by sort order
3. Find group boundaries (where sorted key changes)
4. Segmented cumsum between boundaries

Cost is dominated by argsort: O(N log N) in elements, not groups. Stable at ~3.3 ms regardless of cardinality.

### Hash-Based
Each thread atomically adds its value to the corresponding group bucket. Fast when collision probability is low (high cardinality). Slow when many threads target few buckets (low cardinality, high contention).

## Results (10M rows, float64)

| Groups | Sort (ms) | Sort speedup | Hash (ms) | Hash speedup | Pandas (ms) |
|--------|-----------|-------------|-----------|-------------|-------------|
| 100 | 3.3 | **21x** | 3.1 | **22x** | 69.7 |
| 10K | 3.4 | **26x** | 0.5 | **176x** | 87.7 |
| 1M | 3.4 | **113x** | 0.5 | **706x** | 378.3 |

### Multi-Aggregation (sort-based, 100 groups)

sum+mean+count in 3.4 ms vs pandas 98.6 ms — **29x**. Sort once, derive all aggregations.

## Key Findings

**Sort-based is stable, hash-based scales.** Sort argsort dominates cost at any cardinality (~3.3 ms floor). Hash-based contention drops as cardinality rises — at 1M groups, near-zero contention, near-linear parallelism.

**The crossover is ~1K-10K groups.** Below: either works, sort preferred for stability and multi-agg. Above: hash wins dramatically.

**Multi-aggregation is nearly free with sort-based.** Sort once, apply N aggregations via separate cumsum passes. The argsort dominates in all cases — additional aggregations are cheap.

**Sort-based produces sorted output.** Useful if downstream operations expect ordered groups (e.g., window functions, time-series joins).

## Architecture Decision

**Dual-dispatch groupby** by estimated cardinality:
- Estimate with `len(cp.unique(keys))` — near-zero cost relative to the aggregation
- Below ~1K groups: sort-based (stable cost, multi-agg friendly, sorted output)
- Above ~1K groups: hash-based (low contention, 100-700x over pandas)

## What's Next

Experiment 012 tests fusing expression evaluation into the groupby — `groupby("key").agg(sum(a * b + c))` without materializing the `a * b + c` intermediate. The hypothesis: eliminate the 80 MB intermediate VRAM round-trip from expression evaluation.

The challenge: Experiment 012's fused kernel uses atomic adds for group accumulation (like hash-based) even when cardinality is low — potentially creating contention that outweighs the fusion benefit. The correct from-scratch architecture would use the sort permutation + warp-shuffle segmented reduction (no atomics). This is the question Experiment 012/013 will answer.
