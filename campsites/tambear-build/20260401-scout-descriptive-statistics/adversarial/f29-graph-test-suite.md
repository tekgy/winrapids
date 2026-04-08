# Family 29: Graph Algorithms — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/graph.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| BFS | graph.rs:96-113 | OK |
| DFS | graph.rs:116-132 | OK |
| Topological sort (Kahn) | graph.rs:137-157 | OK |
| Connected components | graph.rs:162-182 | OK |
| Dijkstra | graph.rs:210-231 | OK (NaN-safe ordering) |
| Bellman-Ford | graph.rs:236-265 | OK |
| Floyd-Warshall | graph.rs:270-289 | OK |
| Kruskal MST | graph.rs:362-387 | OK (NaN-safe sort) |
| Prim MST | graph.rs:390-419 | **LOW** (-0.0 overflow) |
| Degree centrality | graph.rs:434-438 | OK |
| Closeness centrality | graph.rs:443-455 | OK |
| PageRank | graph.rs:462-487 | OK |
| Label propagation | graph.rs:495-518 | OK |
| Modularity | graph.rs:524-539 | OK |
| Max flow (Edmonds-Karp) | graph.rs:546-601 | OK |
| Clustering coefficient | graph.rs:629-650 | OK |

---

## Finding F29-1: Prim's neg_weight_key Overflow on -0.0 (LOW)

**Bug**: `neg_weight_key` at line 424 computes `-(w.to_bits() as i64)`. For w = -0.0, `to_bits()` = 0x8000000000000000, which casts to `i64::MIN`. Negating `i64::MIN` is integer overflow — panics in debug mode, wraps to `i64::MIN` in release.

**Impact**: Only triggered by -0.0 edge weights, which are extremely unlikely. For all positive weights (the normal MST case), the encoding is correct and monotone.

**Fix**: Use `-(w.to_bits() as i64).wrapping_neg()` or convert via float comparison directly in a custom Ord impl (like Dijkstra does).

---

## Positive Findings

**DijkstraState is NaN-safe.** Uses `unwrap_or(Ordering::Equal)` at line 203.

**Kruskal edge sort is NaN-safe.** Uses `unwrap_or(Ordering::Equal)` at line 374.

**PageRank handles dangling nodes.** Distributes rank evenly from nodes with no out-edges. Standard approach.

**Max flow is correct.** Edmonds-Karp (BFS augmenting paths) with proper residual graph management.

**Bellman-Ford negative cycle detection uses tolerance 1e-14.** Prevents false positives from floating-point noise.

---

## Test Vectors

### TV-F29-DIJKSTRA-01: Shortest path correctness
```
Graph: 0→1(1), 1→2(2), 0→2(10)
Expected: d(0,2) = 3 via path [0,1,2]
```

### TV-F29-BF-01: Negative cycle detection
```
Graph: 0→1(1), 1→2(-5), 2→0(1)
Expected: None (negative cycle detected)
```

### TV-F29-PR-01: PageRank sums to 1
```
Any connected graph
Expected: sum(pagerank) = 1.0 ± 1e-6
```

### TV-F29-MST-01: Kruskal = Prim
```
Any undirected connected graph
Expected: kruskal.total_weight == prim.total_weight
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F29-1: Prim -0.0 overflow | **LOW** | Debug panic on -0.0 | wrapping_neg or custom Ord |
