# The Sharing Infrastructure Layer

## The Problem

Multiple algorithms need the same expensive intermediate. DBSCAN, KNN, outlier detection, silhouette scoring — all want an n×n distance matrix. Without sharing: O(n²d) GPU work, repeated per algorithm. With sharing: computed once, cloned in O(1).

The naive fix (just memoize) misses the structural point: sharing only works if you can PROVE two computations are the same. A distance matrix computed from two different Vec<f64> instances might be the same or different. The type system needs to know.

## The Three-Layer Answer

**Layer 1 — Metric tag**
`DistanceMatrix(L2Sq)` ≠ `DistanceMatrix(Cosine)`. The metric is part of the type, not the data. Consumers that need L2Sq cannot accidentally receive a cosine matrix.

**Layer 2 — DataId (content-addressed provenance)**
`IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: DataId }`. The `DataId` is a blake3 hash of the input data — same data, same hash, regardless of which Vec<f64> instance. Captures semantic identity, not pointer identity.

Implication: when two algorithms compute distances from the same source data (same MKTF blob at same version), they produce the same `DataId`. The session matches them.

**Layer 3 — TamSession (the registry)**
`HashMap<IntermediateTag, Arc<dyn Any + Send + Sync>>`. Type-erased at storage, downcast at retrieval. `register(tag, Arc<T>) -> bool` (first producer wins). `get::<T>(tag) -> Option<Arc<T>>`.

The type erasure is necessary — different intermediates have different concrete types (`DistanceMatrix`, `SufficientStatistics`, etc.) but must coexist in one registry.

## What's Now Wired

`ClusteringEngine` has two new entry points:
- `discover_clusters_with_distance(...)` → `(ClusterResult, Arc<DistanceMatrix>)` — producer: returns distance as byproduct
- `discover_clusters_from_distance(Arc<DistanceMatrix>, epsilon, min_samples)` → `ClusterResult` — consumer: skips GPU work

The CPU steps (density, union-find, border assignment) are extracted into `clustering_from_distance()` — shared by both paths, no duplication.

Demonstrated by test: DBSCAN with epsilon=0.25 → get distance matrix. DBSCAN with epsilon=60.0 → same n×n matrix, zero GPU cost.

## What's Declared But Not Yet Wired

`IntermediateTag` defines all the canonical shared intermediates:
- `DistanceMatrix { metric, data_id }` — ✓ wired in ClusteringEngine
- `SufficientStatistics { data_id, grouping_id }` — declared, not yet auto-registered
- `ClusterLabels { data_id }` — declared, not yet auto-registered
- `Centroids { data_id, k }` — declared, not yet auto-registered
- `TopKNeighbors { k, metric, data_id }` — declared, not yet auto-registered

The pattern is established. Wiring the others follows the same shape as ClusteringEngine.

## The Next Step: TamSession Integration

Currently, sharing is EXPLICIT: the caller holds `Arc<DistanceMatrix>` and passes it manually.

The next step: algorithms take `&mut TamSession` instead of `Option<Arc<...>>`. The session replaces manual threading:

```rust
// Algorithm declares what it consumes
fn needed(session: &TamSession, data: &[f64]) -> Option<Arc<DistanceMatrix>> {
    let tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: DataId::from_f64(data) };
    session.get::<DistanceMatrix>(&tag)
}

// Algorithm declares what it produces
fn register_distance(session: &mut TamSession, data: &[f64], dist: DistanceMatrix) {
    let tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id: DataId::from_f64(data) };
    session.register(tag, Arc::new(dist));
}
```

Then the session can be threaded through a pipeline of algorithms. The compiler step (analyzing the graph before execution) comes after this.

## The Deeper Insight

`SufficientStatistics` is the other canonical shared intermediate. It's the output of `ScatterJit::scatter_multi_phi([PHI_SUM, PHI_SUM_SQ, PHI_COUNT], ...)`. Consumers include:
- Normalization (needs mean = sum/count)
- Z-scoring (needs mean + std = sqrt(sum_sq/count - mean²))
- Pearson correlation (needs sum, sum_sq, count for both variables)
- Linear regression preprocessing (X'X partial sufficient statistics for incremental computation)

For market data: if you're computing per-minute bin statistics, you run `scatter_multi_phi` once and feed the `SufficientStatistics` to every downstream feature that needs group moments. Currently each feature recomputes from scratch.

**The pattern generalizes**: every primitive's output can be a shareable intermediate. The sharing infrastructure makes "compute once, consume many" automatic once all algorithms are wired.
