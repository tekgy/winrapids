# Lab Notebook 003: .tbs Stack End-to-End — Parser, Executor, Pipeline, Session

**Date**: 2026-03-31
**Author**: Pathmaker
**Branch**: main
**Status**: Active
**Hardware**: NVIDIA RTX PRO 6000 Blackwell, CUDA 13.1

---

## Context & Motivation

Three team members built three layers in the same session:
1. **tbs_parser.rs** — Recursive descent parser for `.tbs` chain syntax
2. **pipeline.rs** — `TamPipeline` fluent builder threading `TamSession`
3. **tbs_executor.rs** — Bridge: dispatches parsed steps to pipeline methods

I built the session-aware modules the pipeline consumes (clustering, linear regression, logistic regression, KNN). The question: does the full stack work end-to-end? Can you write `.tbs` text and get cross-algorithm sharing through all layers automatically?

---

## Experiment: Full Stack Integration

### Before

**Hypothesis**: The `.tbs` → executor → pipeline → session → GPU path is complete. Cross-algorithm sharing (proven earlier at the session level) works through all layers, including when expressed as `.tbs` text.

**Design**: Wire `train.logistic` and `knn` into pipeline + executor. Test chains that exercise sharing:

```
discover_clusters(epsilon=3.0, min_samples=1).knn(k=2)
```

If sharing works, the session holds exactly 1 intermediate (distance matrix) after both steps. Two algorithms, one GPU computation, one `.tbs` chain.

**What we're testing**:
1. Pipeline: `train_logistic()`, `knn()` methods work in the fluent builder
2. Executor: `train.logistic(lr=1.0)` and `knn(k=5)` parse and dispatch correctly
3. Cross-algorithm sharing survives the full stack (not just direct session API)
4. Error paths: missing `y`, missing `k`, unsupported operations

### Results

**Implementation**: Added to `pipeline.rs` (2 methods + 3 tests), `tbs_executor.rs` (3 match arms + 8 tests).

**Pipeline layer**:
- `train_logistic(y, lr, max_iter, tol)` → calls `logistic::fit()`, returns `(Self, LogisticModel)`
- `knn(k)` → calls `knn::knn_session()` through pipeline's `TamSession`, stores result in `TamFrame`
- `TamFrame` extended with `knn_result: Option<KnnResult>`

**Executor layer**:
- `("train", Some("logistic"))` → extracts `lr`, `max_iter`, `tol` with defaults (1.0, 500, 1e-8)
- `("knn", None)` → extracts `k` (required, named or positional)
- `TbsResult` now holds `linear_model` + `logistic_model` (renamed from single `model` field)

**Test: cross-algorithm sharing through .tbs**:
```rust
let chain = TbsChain::parse("discover_clusters(epsilon=3.0, min_samples=1).knn(k=2)").unwrap();
let result = execute(chain, data, n, d, None).unwrap();
assert_eq!(result.pipeline.session_len(), 1, "distance matrix shared, not duplicated");
```
Passes. The session holds exactly 1 entry after both steps. The DBSCAN step computes the distance matrix and registers it; the KNN step finds it and skips GPU entirely.

**Test: logistic regression from .tbs text**:
```rust
let chain = TbsChain::parse("train.logistic(lr=1.0, max_iter=500, tol=0.00000001)").unwrap();
let result = execute(chain, data, n, d, Some(y)).unwrap();
assert!(result.logistic_model.unwrap().accuracy > 0.9);
```
Passes. The dotted name `train.logistic` parses correctly, parameters extract correctly, gradient descent runs through TiledEngine.

**Error paths**: All 4 error tests pass (missing y for train.logistic, missing y for train.linear, missing k for knn, unsupported operation).

**151/151 tests pass** (up from 129 before this work).

### Surprise?

The executor is **remarkably thin** — ~10 lines per operation (match arm + arg extraction + pipeline call). The pipeline is doing the real work, and the session is doing the sharing. The executor is just a name-dispatch layer. This is exactly what you want: the intelligence is in the pipeline/session, the executor is mechanical translation.

The `.tbs` defaults pattern works well: `train.logistic()` with no arguments uses sensible defaults (lr=1.0, max_iter=500, tol=1e-8). Named args override: `train.logistic(lr=0.01, max_iter=1000)`.

### Discussion

**The full stack is proven.** Text → AST → execution → GPU, with automatic intermediate sharing, in one pipeline:

```
normalize()
  .discover_clusters(epsilon=0.5, min_samples=2)
  .knn(k=5)
  .train.linear(target="price")
```

This chain: z-scores features, finds clusters (computing distance matrix on GPU), finds K-nearest neighbors (reusing distance matrix from session — zero GPU), fits linear regression (sharing column statistics through session).

**Current supported vocabulary**:
| Step | Type | Session interaction |
|------|------|-------------------|
| `normalize()` | Preprocessing | Produces column stats |
| `discover_clusters(epsilon, min_samples)` | Clustering | Produces/consumes distance matrix |
| `dbscan(...)` | Alias | Same as discover_clusters |
| `knn(k)` | Neighbors | Consumes distance matrix |
| `train.linear(target)` | Training | Consumes column stats |
| `train.logistic(lr, max_iter, tol)` | Training | (direct GPU, no session yet) |

**What's NOT yet wired**: `kmeans` (raw CUDA, f32 — needs dtype bridge), `filter` (needs predicate evaluation), `window` (needs temporal semantics). The parser already handles all of these syntactically.

**Architectural observation**: The executor is an interpreter today. A compiler would analyze the full chain to: (1) pre-allocate session slots, (2) fuse compatible operations, (3) schedule GPU launches. The session already enables (1) naturally. Fusion (2) is the NVRTC compilation budget question from notebook 001 — batch JIT to amortize the 14ms cold cost. Scheduling (3) becomes meaningful with multiple GPU streams.

151/151 tests pass (full crate).
