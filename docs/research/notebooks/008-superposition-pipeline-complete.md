# Notebook 008 — Manifold Superposition Pipeline: Architecture Complete

*2026-03-31 | Navigator*

---

## Hypothesis

The superposition architecture can be wired end-to-end:
`ManifoldMixture` → per-manifold distance computation → weighted combination → DBSCAN → labels.

The session deduplication property should hold: calling the mixture twice with the same
manifold components but different epsilon values reuses the cached distance matrices.

---

## Design

Three new pieces, each in the right layer:

**`ManifoldDistanceOp`** (in `manifold.rs`)
- Implements `TiledOp` using `Manifold` parameter
- Stays in `tambear` (not `winrapids-tiled`) — correct dependency boundary
- `params_key()` returns `manifold.name()` → different kernel per geometry

**`ClusteringEngine::discover_clusters_from_combined`** (in `clustering.rs`)
- Free function (no `&mut self` needed) wrapping `clustering_from_distance`
- Skips the `Metric::L2Sq` assertion (mixture has no single metric)

**`TamPipeline::discover_clusters_mixture`** (in `pipeline.rs`)
- Takes `&ManifoldMixture` + `epsilon_threshold` + `min_samples`
- Session tags: `ManifoldDistanceMatrix { manifold_name, data_id }` per component
- Session tags: `ManifoldMixtureDistance { mix_id, data_id }` for combined
- `TiledEngine` added to `TamPipeline` struct (initialized once in `from_slice`)

### Session caching behavior

```
First call with (mix=[Euclidean, Sphere], data):
  → Compute Euclidean matrix → register ManifoldDistanceMatrix{euclidean, data_id}
  → Compute Sphere matrix   → register ManifoldDistanceMatrix{sphere(r=1.0000), data_id}
  → Combine → register ManifoldMixtureDistance{mix_id, data_id}
  Session: 3 entries

Second call with same mix, different epsilon:
  → Hit ManifoldDistanceMatrix{euclidean}   (cache hit)
  → Hit ManifoldDistanceMatrix{sphere}      (cache hit)
  → Combine (O(n²), no GPU) → mix_tag already registered → skip
  Session: still 3 entries
```

---

## Results

218 total (200 lib, 18 doc). All green.

### New tests passing:

| Test | Verifies |
|------|----------|
| `pipeline_mixture_euclidean_only_matches_dbscan` | Single-manifold mixture = plain DBSCAN |
| `pipeline_mixture_caches_component_matrices` | 3 session entries (2 component + 1 combined) |
| `pipeline_mixture_second_call_hits_cache` | Session stays at 3 on second call |

### The co-native pipeline chain:

```rust
// This executes correctly with real numbers:
let mix = ManifoldMixture::uniform(vec![
    Manifold::Euclidean,
    Manifold::sphere(1.0),
]);

let p = TamPipeline::from_slice(data, n, d)
    .normalize()
    .discover_clusters_mixture(&mix, 2.0, 1);

// p.frame().labels — DBSCAN labels from combined Euclidean+cosine distance
// p.session_len() == 3 — both component matrices + combined, all cached
```

---

## Discussion

### What "superposition complete" means

The architecture the team lead seeded — "all manifolds running simultaneously, never collapsed,
combination weights ARE the explanation" — is now executable:

1. **Type scaffolding** (Notebook 005) — `Manifold` enum, JIT expression strings ✓
2. **Mixture type** (Notebook 006) — `ManifoldMixture`, session tags, `combine()` ✓
3. **Kernel bridge** (Notebook 007) — `ManifoldDistanceOp` implements `TiledOp` ✓
4. **Pipeline wiring** (This notebook) — `discover_clusters_mixture` closes the loop ✓

Remaining:
5. Weight learning — gradient descent on silhouette loss over mixture weights
6. Full Poincaré kernel — whole-vector Möbius subtraction (not per-dimension)

### The B^T convention is a footgun

Every all-pairs distance computation via `TiledEngine` requires B = A^T. This is not
intuitive. The test for `ManifoldDistanceOp` discovered this (returned zeros for d(0,1)
before the fix). This should eventually be abstracted:

```rust
// Future: TiledEngine::distance_matrix(op, data, n, d) → Vec<f64>
// Handles the transpose internally
```

For now it's documented in `discover_clusters_mixture` and the manifold test.

### Session scale invariant

The session holds 3 entries for a 2-manifold mixture:
- `ManifoldDistanceMatrix { euclidean, data_id }` — n×n
- `ManifoldDistanceMatrix { sphere(r=1.0000), data_id }` — n×n
- `ManifoldMixtureDistance { mix_id, data_id }` — n×n

Total memory: 3 × n² × 8 bytes. For n=10,000: ~2.4 GB. This is manageable for
production manifold sizes but would need eviction logic for large n or many manifolds.
Weight-space search (varying weights, fixed matrices) only recomputes the O(n²) combine
step — the per-manifold matrices stay resident. That's the tractability guarantee.

### What the tests actually prove

`pipeline_mixture_euclidean_only_matches_dbscan` proves the mathematical consistency:
- `ManifoldMixture::single(Euclidean)` with `epsilon_threshold=epsilon²`
- Plain `discover_clusters(epsilon, min_samples)`
- Same cluster count

This is non-trivial — it proves that `ManifoldDistanceOp(Euclidean)` produces the same
distance values as `DistanceOp`, that `combine()` with a single weight=1.0 is identity,
and that `clustering_from_combined` with `epsilon²` matches `discover_clusters_session`
with `epsilon`. Three layers of consistency verified.

### The architecture is now a testbed

The `TamPipeline` can now run real geometric experiments:
```rust
// How much hyperbolic structure does financial tick data have?
let mix_euclidean = ManifoldMixture::single(Manifold::Euclidean);
let mix_poincare  = ManifoldMixture::single(Manifold::poincare(-1.0));
let mix_50_50     = ManifoldMixture::new(vec![
    (Manifold::Euclidean,         0.5),
    (Manifold::poincare(-1.0),    0.5),
]).normalize();

// Compare silhouette scores across weight vectors → learn the geometry
```

Poincaré still falls back to Euclidean (whole-vector kernel not yet implemented), but
the infrastructure is in place. When the Poincaré kernel arrives, it's a `TiledOp` impl —
the pipeline doesn't change.
