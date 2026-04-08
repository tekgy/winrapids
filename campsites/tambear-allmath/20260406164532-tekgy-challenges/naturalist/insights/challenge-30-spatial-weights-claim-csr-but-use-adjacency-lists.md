# Challenge 30 — spatial.rs Claims CSR But Uses Adjacency Lists (+ Graph Duplication)

**Date**: 2026-04-06  
**Type A: Representation Challenge**

---

## The Bug

`spatial.rs` module docstring says:
> "Spatial weights matrices are sparse (compressed row format)."

The implementation:
```rust
pub struct SpatialWeights {
    // For each node: list of (neighbor_idx, weight).
    pub neighbors: Vec<Vec<(usize, f64)>>,
    pub n: usize,
}
```

This is `Vec<Vec<(usize, f64)>>` — adjacency list, NOT CSR. The comment and the code disagree.

**The author intended CSR but implemented adjacency lists.**

---

## The Duplication Problem

`SpatialWeights` and `Graph` are structurally identical:

```rust
// graph.rs
pub struct Graph {
    pub adj: Vec<Vec<Edge>>,  // Edge = {to: usize, weight: f64}
    pub n_nodes: usize,
}

// spatial.rs
pub struct SpatialWeights {
    pub neighbors: Vec<Vec<(usize, f64)>>,
    pub n: usize,
}
```

Two structs. Same shape. Different names. Both GPU-hostile. Neither uses CSR.

Moran's I, Geary's C, spatial lag operations — all loop over `weights.neighbors` in the same pattern as graph algorithms loop over `graph.adj`. These are the same operation with different labels.

---

## What Should Happen

Challenge 04 (CSR for graphs) applies equally here. The fix is:
1. Implement `CsrMatrix { row_ptr: Vec<usize>, col_idx: Vec<usize>, values: Vec<f64> }` as a shared type
2. Replace both `Graph::adj` and `SpatialWeights::neighbors` with CsrMatrix
3. The module comment in spatial.rs becomes true: CSR IS the representation

For GPU operations:
- `row_ptr = prefix_sum(degrees)` — Kingdom A (prefix scan)
- `col_idx = gathered neighbors` — gather operation
- Moran's I spatial lag = `accumulate(CsrRow, v[neighbor_j] * w_ij, Add)` — scatter over CSR rows

---

## Why This Matters Now (Not Just Architecture)

`morans_i` and `gearys_c` are O(n²) in the current form — they iterate over all pairs in `neighbors`. With CSR, both become Kingdom A operations (single-pass accumulate over the sparse structure) and GPU-parallelizable.

The spatial autocorrelation algorithms (Moran's I, Geary's C, Ripley's K) are all disguised sparse matrix-vector products. With CSR + accumulate, they're structurally identical to PageRank (also a sparse matrix-vector product over a graph).

---

## Immediate Action

1. Fix the spatial.rs docstring: change "compressed row format" to "adjacency list" so the claim matches the implementation
2. Log as tech debt: `SpatialWeights` and `Graph` should be unified into `CsrMatrix` as challenge 04 is addressed
3. Note: ordinary_kriging builds a DENSE covariance matrix (kriging matrix K). That's correct for kriging (you need all pairwise covariances). But the SPATIAL WEIGHTS (k-NN structure used for Moran's I) should be sparse/CSR.
