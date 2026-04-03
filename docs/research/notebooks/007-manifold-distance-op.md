# Notebook 007 — ManifoldDistanceOp: Closing the JIT Loop

*2026-03-31 | Navigator*

---

## Hypothesis

`ManifoldDistanceOp` — a struct that implements `TiledOp` using a `Manifold` parameter —
can close the loop between:
1. The type scaffolding built in notebook 005 (`Manifold` enum with JIT expression strings)
2. The actual kernel generator (`TiledEngine`) that compiles and dispatches CUDA

If this works, `AccumulateEngine` can be called with a manifold-parameterized distance,
producing a real non-Euclidean distance matrix. The session can then cache it under
`ManifoldDistanceMatrix { manifold_name, data_id }`.

---

## Design

`ManifoldDistanceOp` lives in `tambear/src/manifold.rs` — not in `winrapids-tiled/src/ops.rs`.
This is the correct side of the dependency boundary:
- `tambear` depends on `winrapids-tiled` (already true)
- `winrapids-tiled` must NOT depend on `tambear`
- `ManifoldDistanceOp` implements `TiledOp` (from tiled) using `Manifold` (from tambear)

```rust
pub struct ManifoldDistanceOp {
    pub manifold: Manifold,
}

impl TiledOp for ManifoldDistanceOp {
    fn name(&self) -> &'static str { "manifold_distance" }
    fn params_key(&self) -> String { self.manifold.name() }  // cache key differentiation

    fn cuda_accumulate_body(&self) -> String {
        match &self.manifold {
            Manifold::Euclidean =>
                // L2Sq: Σ(a - b)²
                "    double diff = a_val - b_val;\n    acc += diff * diff;",
            Manifold::Sphere { .. } =>
                // Dot product; extract converts to cosine distance
                // Requires pre-normalized input vectors
                "    acc += a_val * b_val;",
            Manifold::Poincare { .. } | ... =>
                // Falls back to Euclidean; Poincaré needs whole-vector ops
                "    double diff = a_val - b_val;\n    acc += diff * diff;",
        }
    }

    fn cuda_extract(&self) -> String {
        match &self.manifold {
            Manifold::Sphere { .. } => "(1.0 - acc)",  // cosine distance for unit vectors
            _ => "acc",
        }
    }
}
```

**Key API detail discovered**: `TiledEngine::run(op, A, B, m, n, k)` expects B in K×N format:
- For all-pairs distance: A is n×d (n points × d dims), B is d×n (B^T of the point matrix)
- This means `result[i*n+j] = dist(A_row_i, B_col_j)` where B_col_j = original point j

---

## Results

All 27 manifold tests passing (180 lib total, 198 total with doctests).

### Verified behaviors:

| Test | Result |
|------|--------|
| Euclidean via TiledEngine: L2Sq for (0,0)/(1,0)/(0,1) | ✓ |
| Sphere cosine via TiledEngine: unit vectors give dist=0 or 1 | ✓ |
| `params_key()` unique per manifold (kernel cache differentiation) | ✓ |
| `cuda_accumulate_body()` differs by geometry | ✓ |

### The B^T convention:

For `engine.run(&op, A, A_transposed, n, n, d)`:
```
d(p0, p1) = Σ_k (A[0,k] - A_t[k,1])² = (0-1)² + (0-0)² = 1  ✓
d(p1, p2) = Σ_k (A[1,k] - A_t[k,2])² = (1-0)² + (0-1)² = 2  ✓
```

The documentation comment makes this explicit for future callers.

---

## Discussion

### Where `tiled_dist_expr()` fits now

`manifold.rs` has two levels of JIT expression:

1. `tiled_dist_expr()` — strings like `"(a[k] - b[k]) * (a[k] - b[k])"` designed for a
   per-dimension loop JIT system (using array indexing notation)
2. `ManifoldDistanceOp::cuda_accumulate_body()` — strings using `a_val`/`b_val` (TiledOp style)

These are different abstractions at different levels. `tiled_dist_expr()` is still correct
as documentation of the per-dimension formula; `cuda_accumulate_body()` is what actually
compiles. Both should exist until the per-dimension loop JIT is designed — they document
the same geometry from two different perspectives.

### Poincaré is a real constraint

The Poincaré ball distance formula is:
```
d_H(x, y) = (2/√|c|) * arctanh(√|c| * ||(-x) ⊕_c y||)
```

Where `⊕_c` is the Möbius addition, which requires the FULL vectors x and y, not
per-dimension pairs. The current `TiledOp` accumulate structure only gives `a_val`
and `b_val` (one element each). A correct Poincaré kernel needs `a_ptr[k]` for all k,
i.e., access to the whole input rows.

This is a genuine kernel architecture change — not a parameter tweak. It would require
a different accumulate structure (or a separate kernel type outside TiledOp). The fallback
to Euclidean is correct until that work is done.

### What this means for the superposition architecture

Before this notebook: the superposition architecture was type-complete but computation-free.
`ManifoldMixture.combine()` could compute weighted sums of matrices, but those matrices
came from nowhere (Euclidean-only in practice).

After this notebook: for Euclidean and Sphere manifolds, we can actually compute real
distance matrices via `ManifoldDistanceOp + TiledEngine`. The superposition can now
mix Euclidean and cosine distances with real numbers.

The pipeline step `discover_clusters_mixture()` is the next piece — it would:
1. For each component `(manifold, weight)` in the mixture:
   a. Check session for `ManifoldDistanceMatrix { manifold_name, data_id }` (cache hit)
   b. If miss: compute via `TiledEngine::run(&ManifoldDistanceOp::new(manifold), ...)`
   c. Pre-normalize data first if `manifold.needs_prenormalization()`
2. Call `mix.combine(matrices)` → weighted distance matrix
3. Run DBSCAN on combined matrix
4. Cache under `ManifoldMixtureDistance { mix_id, data_id }`

The types, kernels, and cache slots all exist. The pipeline wiring is left.
