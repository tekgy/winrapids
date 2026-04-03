# Notebook 002 — TamPipeline: Fluent Chain API

*2026-03-31 | Navigator | retroactive*

---

## Hypothesis

The `.tbs` vocabulary sketch (garden: `tbs-vocabulary-sketch-2026-03-31.md`) describes a chain-style scripting language where each step transforms a `TamFrame`. The hypothesis: this chain structure can be implemented in Rust as a fluent builder (`TamPipeline`) that:

1. Threads the `TamSession` automatically through all steps — callers never touch engines or sessions directly
2. Passes all tests that verify shared intermediates are actually shared (not re-computed)
3. Demonstrates the co-native property: the Rust API reads like the `.tbs` vocabulary

The secondary hypothesis: a full chain (normalize → cluster → train) should register ≥2 intermediates in the session by end of execution.

---

## Design

### TamFrame
Single type flowing through the chain:
```rust
pub struct TamFrame {
    pub data: Vec<f64>,   // row-major, n×d
    pub n: usize,
    pub d: usize,
    pub labels: Option<Vec<i32>>,   // from clustering
    pub n_clusters: Option<usize>,
}
```

Rejected: state-machine types (`TamPipeline`, `TamPipelineWithClusters`, etc.) — adds type complexity without benefit at this stage. `Option` fields are the right granularity.

### TamPipeline
Holds:
- `TamFrame` (current state)
- `TamSession` (shared intermediates)
- `ClusteringEngine` (initialized once, reused across all cluster steps)

The `HashScatterEngine` was intentionally excluded — `normalize()` uses `train::linear::column_stats()` (pure CPU, no GPU dependency). This keeps the pipeline functional even without a second GPU allocation.

### normalize()
Uses `column_stats()` from `train::linear` directly. Not session-wired at this stage — column stats for normalization are on pre-normalized data, column stats for `fit_session` are on post-normalized data. Different `data_id` → no collision, no false sharing.

Decision: don't force sharing that doesn't exist. The session is for correct sharing, not superficial sharing.

### discover_clusters()
Calls `ClusteringEngine::discover_clusters_session()` — already session-wired. The distance matrix is registered under `IntermediateTag::DistanceMatrix { metric: L2Sq, data_id }`.

### train_linear()
Calls `train::linear::fit_session()` — already session-wired. Returns `(Self, LinearModel)` so the pipeline can continue after training (e.g., evaluate on test set, run clustering on residuals).

---

## Results

**All 7 new tests pass. 105 unit + 16 doctest = 121 total, all green.**

| Test | Verifies |
|------|---------|
| `pipeline_from_slice_shape` | Basic construction, n/d/len |
| `pipeline_normalize_changes_data` | Column 0 mean ≈ 0 after normalize |
| `pipeline_discover_clusters_finds_two` | 2 clusters in 6-point data, no noise |
| `pipeline_session_caches_distance_matrix` | session_len stays 1 across two cluster calls |
| `pipeline_train_linear_basic` | R² > 0.99 on perfect linear data |
| `pipeline_normalize_then_train` | R² > 0.99 after normalize → train |
| `pipeline_full_chain` | normalize → cluster → train; session_len ≥ 2 |

**Surprise**: `pipeline_session_caches_distance_matrix` required choosing `epsilon=3.0` (not `5.0` or `30.0`). The two clusters have intra-cluster max L2Sq ≈ 2, inter-cluster min L2Sq ≈ 162. Any epsilon between ~2 and ~162 gives 2 clean clusters. The earlier session summary showed this caused a test failure in a prior run — the lesson is to verify epsilon against actual distances before asserting cluster count.

**Surprise**: `discover_clusters()` panics via `expect()` instead of propagating errors. This was a deliberate choice — the pipeline chain syntax `.normalize().discover_clusters(...)` is cleaner without `?`, and clustering failures are almost always programmer errors (bad epsilon, n<2), not recoverable at runtime. `train_linear()` keeps `Result` because model failures (singular matrix, etc.) are legitimate runtime conditions.

---

## Discussion

The `.tbs` vocabulary maps cleanly to Rust method calls:

| .tbs | TamPipeline |
|------|-------------|
| `.normalize()` | `.normalize()` |
| `.discover_clusters(epsilon=0.5, min_samples=2)` | `.discover_clusters(0.5, 2)` |
| `.train.linear(target=...)` | `.train_linear(&y)` |

The `target=` argument is the one place where the mapping is lossy — `.tbs` can refer to a named column, Rust needs an explicit `&[f64]`. This will resolve when `TamFrame` gets named columns (DataFrame layer). For now, passing `y` separately is correct.

**What's missing before `.tbs` compiles:**
1. ~~TamPipeline (Rust fluent API)~~ — ✓ **DONE**
2. `.tbs` parser — text → `TamPipeline` calls
3. Manifold-parameterized JIT

Next: notebook 003 covers the parser.
