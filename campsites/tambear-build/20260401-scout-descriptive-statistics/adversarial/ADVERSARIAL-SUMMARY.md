# Adversarial Mathematics: Cross-Family Summary

**Author**: Adversarial Mathematician
**Date**: 2026-04-01 (Phase 2 update)
**Families reviewed**: F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F20, F23, F25, F26, F28, F29, F30, F31, F32 + special_functions + numerical + accumulate_core
**Total findings**: 1 CRITICAL, 25 HIGH, 52+ MEDIUM, 32+ LOW
**Phase 2 fix status**: Bug Class 2 (NaN panics) ✅ VERIFIED FIXED; Bug Class 3 (SVD A^T A) ✅ VERIFIED FIXED

---

## Bug Class 1: Naive Formula (E[x²] - E[x]²)

**Root cause**: The GPU scatter primitive accumulates `{count, sum, sum_sq}` — raw moments. Any downstream computation of variance becomes `sum_sq/n - (sum/n)²`, the naive formula. Catastrophic cancellation at offset ≈ 1e8.

**Instances found** (7):

| # | File:Line | Context | Severity |
|---|-----------|---------|----------|
| 1 | hash_scatter.rs:193 | GPU scatter variance derivation | HIGH |
| 2 | intermediates.rs:279 | SufficientStatistics::variance() | HIGH |
| 3 | robust.rs:381 | LTS ols_subset: `n*sxx - sx*sx` | HIGH |
| 4 | tambear-py/src/lib.rs:101 | Python binding variances() | MEDIUM |
| 5 | complexity.rs:348 | linear_fit_segment (x=integers) | LOW |
| 6 | main.rs:98 | Test reference (validates bug with bug) | LOW |
| 7 | train/linear.rs:183→intermediates.rs:279 | column_stats → fit_session z-scoring | HIGH |

**Structural fix**: `SufficientStatistics` should store `m2` (centered Σ(x-mean)²) instead of `sum_sq` (raw Σx²). The `MomentStats` type already does this correctly — the issue is bypass paths that use raw accumulators.

**Correct instances**: complexity.rs:254 (`ols_slope`), robust.rs:492 (`subset_stats_2d` in MCD), hurst_rs variance (lines 228-229). These center by mean before computing sums.

---

## Bug Class 2: NaN Panics (partial_cmp().unwrap())

**Root cause**: Rust's `f64::partial_cmp()` returns `None` for NaN. Calling `.unwrap()` on None panics. Any library function that sorts user-provided f64 data and uses `unwrap()` will crash on NaN input.

**Production code instances** (25+):

| File | Functions affected |
|------|-------------------|
| complexity.rs:417,483 | lempel_ziv_complexity, correlation_dimension |
| signal_processing.rs:1007 | median_filter |
| nonparametric.rs:309,340,341,569,647 | ks_test_normal, ks_test_two_sample, sign_test |
| robust.rs:179,207,242,245,358,368,451,548 | MAD, Qn, Sn, LTS, MCD, medcouple |
| descriptive.rs:720,753 | various |
| knn.rs:109,115,121 | k-nearest neighbors |
| rng.rs:360 | sample_weighted (binary_search_by) |
| optimization.rs:437 | nelder_mead final min selection |
| spatial.rs:260 | SpatialWeights::knn distance sort |
| spatial.rs:397 | nn_distances sort |

**Related**: neural.rs max pooling (max_pool1d line 455, max_pool2d line 506) silently converts NaN → NEG_INFINITY via `>` comparison. Not a panic, but silent data corruption — NaN should propagate through max operations.

**Structural fix**: Either:
- Use `f64::total_cmp()` (Rust 1.62+) — treats NaN as largest value
- Use `unwrap_or(std::cmp::Ordering::Equal)` — NaN sorts arbitrarily
- Pre-filter NaN before sorting

---

## Bug Class 3: Condition Number Squaring (A^T A Pattern)

**Root cause**: Computing A^T A and then operating on it squares the condition number. For κ(A) = 1e8, κ(A^T A) = 1e16, which exhausts f64 precision.

**Instances found** (3):

| # | File | Operation | Severity |
|---|------|-----------|----------|
| 1 | linear_algebra.rs:534 | SVD via sym_eigen(A^T A) | HIGH |
| 2 | interpolation.rs:570 | polyfit via Vandermonde normal equations | MEDIUM |
| 3 | train/linear.rs:108 | Normal equations X^T X (mitigated by z-scoring IF z-scoring works) | MEDIUM |

**Note**: SVD should use Golub-Kahan bidiagonalization. Polyfit should use QR on the Vandermonde matrix or orthogonal polynomial basis. The linear regression X^T X is acceptable when z-scored (but z-scoring uses the naive variance — Bug Class 1).

---

## Bug Class 4: Hardcoded Absolute Thresholds

| File:Line | Threshold | Should be |
|-----------|-----------|-----------|
| linear_algebra.rs:640 | Jacobi convergence: `max_val < 1e-14` | Relative to Frobenius norm |
| linear_algebra.rs:550 | SVD U column: `sigma[j] > 1e-14` | Relative to σ_max |
| robust.rs:382 | OLS singular: `denom.abs() < 1e-15` | Relative to data scale |

These are LOW individually but create a pattern where the library works well for "normal" data but fails for extreme-scale data.

**Additional instances found in later review**:

| File:Line | Threshold | Should be |
|-----------|-----------|-----------|
| optimization.rs:316 | L-BFGS curvature: `sy > 1e-14` | Relative to ‖s‖·‖y‖ |
| spatial.rs:222 | Pivot threshold: `max_val < 1e-300` | Relative to matrix norm |

---

## Findings by Family (Phase 1 — original review)

| Family | File | HIGH | MEDIUM | LOW | Key Finding |
|--------|------|------|--------|-----|-------------|
| F01 | distances | 2 | 1 | 0 | L2 offset, cosine triple bug |
| F02 | linear_algebra.rs | ~~1~~ | 1 | 2 | ~~SVD via A^T A~~ ✅ FIXED: one-sided Jacobi SVD |
| F03 | signal_processing.rs | ~~1~~ | 0 | 1 | ~~Median NaN panic~~ ✅ FIXED |
| F04 | rng.rs | 0 | ~~1~~ | 1 | ~~Weighted sampling NaN~~ ✅ FIXED |
| F05 | optimization.rs | 0 | 2 | 0 | NM NaN ✅ FIXED, L-BFGS abs threshold |
| F06 | hash_scatter/intermediates | 2 | 1 | 1 | Naive formula (root) — Task #3 IN PROGRESS |
| F07 | hypothesis.rs | 0 | 4 | 1 | ANOVA empty groups, NaN propagation |
| F08 | nonparametric.rs | 0 | 2 | 1 | Kendall NaN ✅ FIXED, MW no tie correction |
| F09 | robust.rs | 1 | 0 | 1 | LTS naive OLS |
| F10 | train/linear.rs | 1 | 2 | 0 | column_stats → z-score broken |
| F20 | clustering.rs, kmeans.rs | 0 | 2 | 0 | Empty cluster, initialization |
| F23 | neural.rs | 0 | 1 | 2 | Max pool NaN→−∞ suppression |
| F25 | information_theory.rs | 1 | 1 | 1 | i32 overflow in joint histogram |
| F26 | complexity.rs | ~~1~~ | 1 | 1 | ~~NaN panic~~ ✅ FIXED, O(m!) memory |
| F29 | graph.rs | 0 | 0 | 1 | Prim -0.0 overflow |
| F30 | spatial.rs | 0 | ~~2~~ | 0 | ~~KNN NaN panics~~ ✅ FIXED |
| F31 | interpolation.rs | 0 | 1 | 0 | polyfit Vandermonde conditioning |
| SF | special_functions.rs | 0 | 0 | 1 | erf 1.5e-7 precision limit |

---

## Global Recommendations

### Priority 1: Fix the naive formula root cause
Replace `SufficientStatistics.sum_sq` with `SufficientStatistics.m2` (centered second moment). This fixes Bug Class 1 at the source. All consumers automatically get correct variance.

### Priority 2: NaN-safe sorting
Grep for `partial_cmp.*unwrap` and replace all production instances with `total_cmp` or `unwrap_or(Equal)`. This is a 5-minute global fix.

### Priority 3: SVD via bidiagonalization
The A^T A approach in `svd()` is fundamentally broken for ill-conditioned matrices. Replace with QR-based SVD: `QR(A) = Q·R`, then `SVD(R)` via Jacobi (R is square, smaller).

### Priority 4: Everything else
All MEDIUM findings are individually important but less urgent than the structural fixes above.

---

## Phase 2 Fix Verification: Bug Class 2 (NaN Panics)

**Status**: ✅ VERIFIED FIXED (2026-04-01)

All 25+ bare `partial_cmp().unwrap()` instances eliminated. Two fix approaches used:

**Approach A — `total_cmp()`** (30 instances, BEST fix): complexity.rs, signal_processing.rs, nonparametric.rs, robust.rs, descriptive.rs, knn.rs, rng.rs, optimization.rs, spatial.rs. NaN deterministically sorts after +Inf.

**Approach B — `unwrap_or(Equal)`** (11 instances, SAFE but weaker): interpolation.rs:143, neural.rs:1223+1232, optimization.rs:366, robust.rs:697, linear_algebra.rs:685, hypothesis.rs:516+535, nonparametric.rs:457, graph.rs:203+374.

**Residual concern**: `unwrap_or(Equal)` makes NaN compare equal to everything — non-transitive, sort order depends on algorithm internals. Recommend migrating these 11 instances to `total_cmp()` as well. Not a panic, but a correctness hazard.

**NOT fixed (out of scope)**: neural.rs max_pool1d:455 and max_pool2d:506 still silently convert NaN → NEG_INFINITY via `>` comparison. All-NaN windows produce `-∞` output instead of NaN. This is silent data corruption, not a panic.

**Downstream finding from NaN fix**: graph.rs:203 (Dijkstra) and graph.rs:374 (Kruskal) use `unwrap_or(Equal)` which makes NaN weights silently poison results (HIGH severity — see F29 findings below).

---

## Phase 2 Fix Verification: Bug Class 3 (SVD via A^T A)

**Status**: ✅ VERIFIED FIXED (2026-04-01)

Old code: `sym_eigen(a.t().mul(a))` — squared condition number, κ(A)=1e8 → κ(A^T A)=1e16.

New code: **One-sided Jacobi SVD** working directly on columns of A (linear_algebra.rs:527-624). Key improvements:
- No A^T A computation — works on A directly
- Convergence threshold is **relative**: `gamma.abs() <= 1e-14 * (alpha * beta).sqrt()`
- Singular value sort uses `total_cmp` (NaN-safe)
- Wide matrix handled via transpose recursion

**Residual findings** (both LOW):
- Line 588: `sigma[j] > 1e-14` — hardcoded absolute threshold for U column normalization. For `A = 1e-20 * I`, no U columns are computed. Should be `sigma[j] > eps * sigma_max`.
- Line 615: Same `1e-14` in null-space Gram-Schmidt.

---

## NEW: F29 Graph — Deep Adversarial Findings (Phase 2)

**Previous assessment**: 0 HIGH, 0 MEDIUM, 1 LOW (just −0.0). **Upgraded to**: 1 CRITICAL, 2 HIGH, 5 MEDIUM, 4 LOW.

| # | File:Line | Severity | Description |
|---|-----------|----------|-------------|
| G1 | graph.rs:422-428 | **CRITICAL** | Prim `neg_weight_key`: overflow on ALL negative weights, not just −0.0. Panics in debug, wraps in release. |
| G2 | graph.rs:203 | **HIGH** | Dijkstra NaN poison via `unwrap_or(Equal)` — NaN costs propagate silently |
| G3 | graph.rs:374 | **HIGH** | Kruskal NaN sort via `unwrap_or(Equal)` — non-transitive comparator |
| G4 | graph.rs:242 | MEDIUM | Bellman-Ford `n-1` underflow for empty graph |
| G5 | graph.rs:270-288 | MEDIUM | Floyd-Warshall: no negative cycle detection |
| G6 | graph.rs:292-305 | MEDIUM | `reconstruct_path`: infinite loop on cyclic parent array |
| G7 | graph.rs:272,549 | MEDIUM | Floyd-Warshall/max_flow O(V²) dense allocation, no guard |
| G8 | graph.rs:383 | MEDIUM | Kruskal `n-1` underflow for empty graph |
| G9 | graph.rs:482 | LOW | PageRank: absolute L1 convergence |
| G10 | graph.rs:509 | LOW | label_propagation: non-deterministic tie-breaking |
| G11 | graph.rs:526-538 | LOW | modularity: formula ambiguity for directed graphs |
| G12 | graph.rs:258 | LOW | Bellman-Ford: 1e-14 epsilon misses tiny negative cycles |

---

## NEW: F25 Information Theory — Phase 2 Findings

| # | File:Line | Severity | Description |
|---|-----------|----------|-------------|
| IT1 | information_theory.rs:319 | **HIGH** | i32 overflow in joint_histogram — CONFIRMED STILL PRESENT |
| IT2 | information_theory.rs:335-339 | **HIGH** | Negative labels → panic in contingency_from_labels |
| IT3 | information_theory.rs:92-96 | MEDIUM | Renyi H_0 of zero distribution → -inf |
| IT4 | information_theory.rs:98-102 | MEDIUM | Renyi min-entropy of zero distribution → +inf |
| IT5 | information_theory.rs:464-473 | MEDIUM | entropy_histogram -inf when bin_width underflows to 0 |

---

## NEW: F20 Clustering — Phase 2 Deep Findings

| # | File:Line | Severity | Description |
|---|-----------|----------|-------------|
| C1 | kmeans.rs:94-108 | **HIGH** | Empty cluster centroid collapses to origin, attracts points |
| C2 | kmeans.rs:133,155 | **HIGH** | k=0 → division by zero panic (no assertion) |
| C3 | kmeans.rs:154-158 | MEDIUM | Deterministic stride init → duplicate centroids on structured data |
| C4 | clustering.rs:346-354 | MEDIUM | DBSCAN self-distance inflates density count (undocumented) |
| C5 | clustering.rs:394-403 | MEDIUM | DBSCAN border point: first core neighbor by index, not nearest |
| C6 | knn.rs:182-183 | MEDIUM | Manifold distance matrix mislabeled as L2Sq |
| C7 | clustering.rs:339 | MEDIUM | No dist.len()==n*n validation in public API |
| C8 | kmeans.rs:257 | LOW | iterations=0 → Inf in timing output |
| C9 | clustering.rs:127-138 | LOW | Negative epsilon silently produces all-noise |
| C10 | clustering.rs:253-258 | LOW | Negative radius squared → positive epsilon |
| C11 | knn.rs:89-91 | LOW | k=0 accepted by knn_from_distance |

---

## NEW: Bug Class 5: Absolute Convergence Tolerances (numerical.rs)

**Root cause**: Every root-finding method and ODE solver uses absolute convergence checks. `|f(x)| < tol` and `|x_{n+1} - x_n| < tol` don't scale with problem magnitude.

**Impact**: Functions with small values (e.g., `f(x) = 1e-20 * sin(x)`) report false convergence instantly. Functions with large values near roots never converge within tolerance.

**Instances** (all in numerical.rs):
- bisection:47 — `fc.abs() < tol`
- newton:80 — `fx.abs() < tol`
- secant:106 — `fc.abs() < tol`
- fixed_point:566 — `(xn - x).abs() < tol`

**Structural fix**: Unified `is_converged(value, reference, tol)` using `|value| < tol * (1 + |reference|)`.

---

## NEW: F28 Manifold — Adversarial Findings

| # | File:Line | Severity | Description |
|---|-----------|----------|-------------|
| M1 | manifold.rs:552-583 | **HIGH** | Sphere cosine distance silently wrong on non-unit vectors (no normalization, no check) |
| M2 | manifold.rs:208-213 | MEDIUM | Poincaré `tiled_dist_expr()` returns Euclidean L2Sq — semantic lie |
| M3 | manifold.rs:591-594 | MEDIUM | Hardcoded `1e-15` Poincaré denominator clamp doesn't scale with κ |
| M4 | manifold.rs:303-306 | MEDIUM | Sphere/SphericalGeodesic projection: div-by-zero for zero-norm vectors |
| M5 | manifold.rs:337-339 | MEDIUM | `distance_is_dissimilarity()` returns wrong boolean for Sphere (cosine distance IS a dissimilarity) |
| M6 | manifold.rs:261-265 | MEDIUM | Poincaré gradient scale: squared negative for out-of-ball points pushes further out |
| M7 | manifold.rs:601 | LOW | SphericalGeodesic zero-vector: produces fabricated π/2 distance |
| M8 | manifold.rs:297 | LOW | Poincaré projection margin hardcoded at `1e-5` |
| M9 | manifold.rs:428-429 | LOW | ManifoldMixture::normalize allows negative individual weights |

---

## NEW: Numerical Methods (numerical.rs) — Adversarial Findings

| # | File:Line | Severity | Description |
|---|-----------|----------|-------------|
| N1 | numerical.rs:39-61 | **HIGH** | bisection: no bracket sign-change validation; silently returns garbage root |
| N2 | numerical.rs:84 | **HIGH** | Newton: `1e-15` absolute derivative guard blocks well-scaled small-valued functions |
| N3 | numerical.rs:110 | **HIGH** | Secant: `1e-15` absolute denominator guard — same failure mode as Newton |
| N4 | numerical.rs:150-180 | **HIGH** | Brent: missing `mflag` bookkeeping — incomplete algorithm |
| N5 | numerical.rs:451-464 | **HIGH** | RK45: claims Dormand-Prince, implements RKF45; docstring claims MATLAB compat |
| N6 | numerical.rs:169 | MEDIUM | Brent secant branch: no guard against `fb == fa` (division by zero) |
| N7 | numerical.rs:474 | MEDIUM | RK45: `1e-15` absolute terminal time check |
| N8 | numerical.rs:495 | MEDIUM | RK45: `h <= 1e-15` absolute minimum step size |
| N9 | numerical.rs:466-514 | MEDIUM | RK45: no NaN/Inf/divergence detection |
| N10 | numerical.rs:378-437 | MEDIUM | Euler/RK4: no divergence detection |
| N11 | numerical.rs:518-553 | MEDIUM | RK4 system: no dimension check on `f` return value |
| N12 | numerical.rs:314-349 | MEDIUM | Adaptive Simpson: NaN in `f` causes 2^depth evaluations (effectively infinite loop for depth=50) |
| N13 | numerical.rs:47,80,106,566 | MEDIUM | All convergence checks are absolute (Bug Class 5) |
| N14 | numerical.rs:503 | LOW | RK45: `1e-30` absolute error threshold for step growth |
| N15 | numerical.rs:259-274 | LOW | Simpson: n=0 produces Inf/NaN |

---

## NEW: Accumulate Core (accumulate.rs + compute_engine.rs) — Adversarial Findings

| # | File:Line | Severity | Description |
|---|-----------|----------|-------------|
| A1 | compute_engine.rs:270-276 | **HIGH** | CPU/CUDA behavioral divergence for all-NaN groups: CPU→sentinel, CUDA→NaN |
| A2 | compute_engine.rs:208+ | **HIGH** | Negative key values: OOB access on CPU, wild GPU memory write on CUDA |
| A3 | accumulate.rs:185-191 | MEDIUM | Three independent CUDA contexts created (~300MB VRAM overhead) |
| A4 | accumulate.rs:423-440 | MEDIUM | Softmax: single NaN contaminates all outputs via sum_exp |
| A5 | reduce_op.rs:77 | MEDIUM | `n as i32` overflow for arrays >2^31 elements |
| A6 | compute_engine.rs:329+ | MEDIUM | `n as u32` overflow in grid dimensions for arrays >2^32 elements |
| A7 | accumulate.rs:207-213 | LOW | Empty input sentinel values undocumented (max→-∞, argmin→∞) |
| A8 | compute_engine.rs:~314 | LOW | Kernel cache key includes `n` — unbounded cache growth |

---

## Updated Findings by Family (Phase 2 — complete)

| Family | File | CRIT | HIGH | MED | LOW | Key Finding |
|--------|------|------|------|-----|-----|-------------|
| F01 | distances | | 2 | 1 | | L2 offset, cosine triple bug |
| F02 | linear_algebra.rs | | ~~1~~ | 1 | 2 | ~~SVD A^T A~~ ✅ FIXED: one-sided Jacobi. Residual: hardcoded 1e-14 U threshold |
| F03 | signal_processing.rs | | ~~1~~ | | 1 | ~~Median NaN~~ ✅ FIXED |
| F04 | rng.rs | | | ~~1~~ | 1 | ~~Weighted NaN~~ ✅ FIXED |
| F05 | optimization.rs | | | 2 | | NM NaN ✅ FIXED, L-BFGS abs threshold |
| F05 | numerical.rs | | **5** | **8** | **2** | bisection no-bracket, Brent incomplete, RK45=RKF45, Bug Class 5 |
| F06 | hash_scatter/intermediates | | 2 | 1 | 1 | Naive formula — Task #3 IN PROGRESS |
| F07 | hypothesis.rs | | | 4 | 1 | ANOVA empty groups |
| F08 | nonparametric.rs | | | 2 | 1 | Kendall NaN ✅ FIXED, MW no tie correction |
| F09 | robust.rs | | 1 | | 1 | LTS naive OLS |
| F10 | train/linear.rs | | 1 | 2 | | column_stats → z-score broken |
| **F20** | **clustering/kmeans/knn** | | **2** | **5** | **4** | **Empty cluster→origin, k=0 div-by-zero, DBSCAN border assignment** |
| F23 | neural.rs | | | 1 | 2 | Max pool NaN→−∞ STILL PRESENT |
| **F25** | **information_theory.rs** | | **2** | **3** | **1** | **i32 overflow CONFIRMED, negative labels panic, log-domain hazards** |
| F26 | complexity.rs | | ~~1~~ | 1 | 1 | ~~NaN panic~~ ✅ FIXED, O(m!) memory |
| **F28** | **manifold.rs** | | **1** | **5** | **3** | **Sphere non-unit vectors, Poincaré semantic lies** |
| **F29** | **graph.rs** | **1** | **2** | **5** | **4** | **Prim neg_weight CRITICAL, Dijkstra/Kruskal NaN, Floyd neg-cycle** |
| F30 | spatial.rs | | | ~~2~~ | | ~~KNN NaN~~ ✅ FIXED |
| F31 | interpolation.rs | | | 1 | | polyfit Vandermonde conditioning |
| SF | special_functions.rs | | | | 1 | erf 1.5e-7 precision limit |
| **Core** | **accumulate/compute_engine** | | **2** | **4** | **2** | **CPU/CUDA NaN divergence, neg key OOB** |

---

## Artifacts

| Document | Families |
|----------|----------|
| f02-linear-algebra-test-suite.md | F02 |
| f03-signal-processing-test-suite.md | F03 |
| f04-rng-test-suite.md | F04 |
| f05-optimization-test-suite.md | F05 |
| f07-hypothesis-test-suite.md | F07 |
| f08-f09-nonparametric-robust-test-suite.md | F08, F09 |
| f10-regression-test-suite.md | F10 |
| f20-clustering-test-suite.md | F20 |
| f25-information-theory-test-suite.md | F25 |
| f26-complexity-chaos-test-suite.md | F26 |
| f29-graph-test-suite.md | F29 |
| f30-spatial-test-suite.md | F30 |
| special-functions-test-suite.md | special_functions |
| naive-formula-codebase-sweep.md | Cross-family |
| f07-hypothesis-adversarial-proof.py | F07, F25 |
| f08-f09-adversarial-proof.py | F08, F09 |
| f02-svd-adversarial-proof.py | F02 (PROVEN) |
| f23-neural-test-suite.md | F23 |
| manifold-confidence-universal-v-column.py | F01 manifold V-column |
