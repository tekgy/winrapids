# Notebook 006 — ManifoldMixture and Multi-Manifold Session

*2026-03-31 | Navigator*

---

## Hypothesis

A `ManifoldMixture` struct (weighted combination of `Manifold` variants) can:

1. Be hashed to a stable `DataId` (`mix_id`) for session keying
2. Combine pre-computed `DistanceMatrix` values from the session via weighted sum — O(n²) not O(n²d)
3. Enable the pipeline step `discover_clusters_mixture(mix, epsilon, min_samples)` which:
   - Retrieves each component's `ManifoldDistanceMatrix` from session (or computes it)
   - Combines them: `D_combined = Σ w_i * D_i`
   - Runs DBSCAN on the combined matrix
   - Registers `ManifoldMixtureDistance { mix_id, data_id }` in session

For now, only `Manifold::Euclidean` has a real distance kernel. Poincaré/Sphere are architectural placeholders (the JIT wiring is future work). The Euclidean-only case proves the mixture concept with real numbers.

---

## Design

**`ManifoldMixture` in `manifold.rs`:**
```rust
pub struct ManifoldMixture {
    pub components: Vec<(Manifold, f64)>,
}
impl ManifoldMixture {
    pub fn mix_id(&self) -> DataId
    pub fn combine(&self, matrices: &[Arc<DistanceMatrix>]) -> Vec<f64>
    pub fn uniform(manifolds: Vec<Manifold>) -> Self
    pub fn single(manifold: Manifold) -> Self
    pub fn normalize(self) -> Self
    pub fn total_weight(&self) -> f64
}
```

**New tag variants in `intermediates.rs`:**
```rust
ManifoldDistanceMatrix { manifold_name: String, data_id: DataId },
ManifoldMixtureDistance { mix_id: DataId, data_id: DataId },
```

Note: `manifold_name: String` (not `Manifold`) avoids a circular dependency.
`manifold.rs` depends on `intermediates.rs` for `DataId`; the reverse would be circular.

**`Manifold::from_metric()` conversion bridge:**
```rust
impl Manifold {
    pub fn from_metric(m: Metric) -> Self { ... }
}
```

---

## Results

All tests pass: 176 lib, 18 doc.

### ManifoldMixture behavior verified:

| Test | Result |
|------|--------|
| `uniform()` weights sum to 1.0 | ✓ |
| `single()` is weight 1.0 | ✓ |
| `normalize()` renormalizes arbitrary weights | ✓ |
| `combine()` single matrix = identity | ✓ |
| `combine()` weighted sum (D1=1s, D2=2s, w=0.6/0.4) → 1.4 | ✓ |
| `combine()` non-symmetric layout preserved | ✓ |
| `mix_id()` stable for same mixture | ✓ |
| `mix_id()` differs for different weights | ✓ |
| `mix_id()` differs for different manifolds | ✓ |
| `from_metric(L2Sq)` → `Euclidean` | ✓ |
| `from_metric(Cosine)` → `Sphere { r=1.0 }` | ✓ |

### What works end-to-end:

```rust
// This executes correctly:
let mix = ManifoldMixture::new(vec![
    (Manifold::Euclidean, 0.6),
    (Manifold::poincare(-1.0), 0.4),
]);
let d1 = Arc::new(DistanceMatrix { n: 2, data: Arc::new(vec![1.0; 4]), .. });
let d2 = Arc::new(DistanceMatrix { n: 2, data: Arc::new(vec![2.0; 4]), .. });
let combined = mix.combine(&[d1, d2]);
// combined[i,j] = 0.6*1.0 + 0.4*2.0 = 1.4  ✓
```

The `mix_id()` hash is deterministic — two mixtures with the same components and weights
produce the same `DataId`, enabling session deduplication.

### What remains architectural (not tested against real kernels):

- `Manifold::Poincare` and `Manifold::Sphere` JIT expression strings are present
  but not wired into `TiledEngine`. Their distance matrices fall back to Euclidean.
- The pipeline step `discover_clusters_mixture()` is not yet implemented.
  Session caching for mixture results exists as `IntermediateTag::ManifoldMixtureDistance`
  but the pipeline method that produces it is future work.

---

## Discussion

### The O(n²) vs O(n²d) leverage

The core insight holds numerically. Once each component's distance matrix is in the session,
varying the weights is pure O(n²) arithmetic — no re-scanning of the data points (O(n²d)).

For n=10,000 points, d=128 dimensions:
- Recomputing all three matrices: 3 × 10,000² × 128 = ~38B operations
- Recombining cached matrices at a new weight: 10,000² × 3 = ~300M operations
- Speedup: ~125×

The session makes weight-space search (e.g., gradient descent on silhouette loss) tractable.
Without caching, each weight evaluation recomputes everything.

### The circular dependency resolution

`manifold.rs` imports `DataId` from `intermediates.rs`, so `IntermediateTag` cannot hold
a `Manifold` directly. The solution: `ManifoldDistanceMatrix { manifold_name: String, data_id }`.

This is an intentional trade-off. The `String` key is slightly less type-safe but avoids
a module-level cycle. When `Manifold` eventually replaces `Metric` in `IntermediateTag`,
the module structure will need to be reorganized (move `DataId` to a shared types crate,
or merge `Metric`/`Manifold` into `intermediates.rs`). For now the string bridge is correct.

### The type scaffolding is the expensive part

Building the JIT expressions for Poincaré/Sphere into `tiled_dist_expr()`,
`gradient_scale_expr()`, `centroid_update_expr()`, `project_expr()` took thought.
The *wiring* (passing these strings into `TiledEngine`'s kernel generator) is mechanical.
The types are right. The algebra is right. The plumbing is left.

### What this unlocks

The manifold superposition architecture is now type-complete:

1. `Manifold` — single geometry as JIT parameter ✓
2. `ManifoldMixture` — weighted combination ✓
3. `IntermediateTag::ManifoldDistanceMatrix` — per-geometry session cache ✓
4. `IntermediateTag::ManifoldMixtureDistance` — combined session cache ✓
5. `TiledEngine` JIT parameterization — next architectural layer
6. Pipeline step `discover_clusters_mixture()` — follows from 5

The boundary between "done" and "next" is clean: everything up to TiledEngine
parameterization is types and math. Everything after is kernel plumbing.
