# F29 Graph Algorithms — Adversarial Test Suite (Phase 2)

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**File**: `src/graph.rs` (~28KB)

---

## G1 [CRITICAL]: Prim MST neg_weight_key overflow on negative/−0.0 weights

**Location**: graph.rs:422-428

**Bug**: `neg_weight_key(w)` computes `-(w.to_bits() as i64)`. For any negative weight, `to_bits() > i64::MAX`, so `as i64` wraps. Then negation of `i64::MIN` overflows (panic in debug, wrap in release).

**Test vectors**:
```rust
// -0.0: to_bits() = 0x8000_0000_0000_0000 → as i64 = i64::MIN → -i64::MIN = OVERFLOW
prim(&graph_with_edge(0, 1, -0.0)) // PANICS in debug

// -1.0: to_bits() = 0xBFF0... > i64::MAX → same overflow class
prim(&graph_with_edge(0, 1, -1.0)) // PANICS in debug

// Positive weights: safe
prim(&graph_with_edge(0, 1, 1.0)) // OK
```

**Severity UPGRADE**: Originally flagged as LOW (just −0.0). Now CRITICAL: affects ALL negative weights.

**Fix**: Standard float-to-orderable bijection: `if bits < 0 { !bits } else { bits }`.

---

## G2 [HIGH]: Dijkstra NaN poison via unwrap_or(Equal)

**Location**: graph.rs:203

NaN costs compare as Equal to everything → silently poisons entire distance array.

## G3 [HIGH]: Kruskal NaN sort via unwrap_or(Equal)

**Location**: graph.rs:374

NaN edges sort non-transitively → MST can include NaN-weight edges.

**Fix for both**: `a.w.total_cmp(&b.w)` — NaN sorts last, excluded naturally.

---

## G4 [MEDIUM]: Bellman-Ford n=0 underflow

**Location**: graph.rs:242

`0..n-1` where n=0 → usize underflow → ~2^64 iterations.

## G5 [MEDIUM]: Floyd-Warshall no negative cycle detection

**Location**: graph.rs:270-288

Silently produces wrong distances. Should check `dist[i][i] < 0.0` post-loop.

## G6 [MEDIUM]: reconstruct_path infinite loop on cyclic parent array

**Location**: graph.rs:292-305

Public function, no visited tracking, no iteration bound.

## G7 [MEDIUM]: Floyd-Warshall/max_flow O(V²) dense allocation

**Location**: graph.rs:272, 549

80GB for V=100K. No size guard.

---

## G8 [LOW]: PageRank absolute L1 convergence
## G9 [LOW]: label_propagation non-deterministic tie-breaking
## G10 [LOW]: modularity formula ambiguity (directed vs undirected)
## G11 [LOW]: Bellman-Ford 1e-14 epsilon misses tiny negative cycles
