# F28 Manifold — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01 (Phase 2)
**File**: `src/manifold.rs` (1632 lines)

---

## M1 [HIGH]: Sphere cosine distance on non-unit vectors

**Location**: manifold.rs:552-555 (accumulate), 582-583 (extract)

**Bug**: Sphere variant computes `1 - dot(a,b)` which is only cosine distance when both vectors are unit-normalized. No runtime check. Non-unit vectors produce negative distances or values >1.

**Test vectors**:
```
# Identical non-unit vectors: should be distance 0
a = (2, 0), b = (2, 0) → current: 1 - 4 = -3 (WRONG, negative distance)

# Orthogonal non-unit vectors: should be distance 1
a = (3, 0), b = (0, 3) → current: 1 - 0 = 1 (correct by accident)

# Anti-parallel non-unit vectors: should be distance 2
a = (5, 0), b = (-5, 0) → current: 1 - (-25) = 26 (WRONG, should be 2)
```

**Fix**: Either (a) add `needs_prenormalization()` and call it, or (b) normalize in extract: `1.0 - acc.dot_prod / sqrt(acc.sq_norm_x * acc.sq_norm_y)`.

---

## M2 [MEDIUM]: Poincare tiled_dist_expr returns Euclidean

**Location**: manifold.rs:208-213

**Bug**: `tiled_dist_expr()` for Poincare silently returns `(a[k]-b[k])*(a[k]-b[k])` — Euclidean L2Sq. Any caller using this public method for Poincare gets wrong distances with no warning.

**Test**: Compare `tiled_dist_expr` output vs `ManifoldDistanceOp` output for Poincare points. They will disagree.

**Fix**: Return `Err` or panic for Poincare, or deprecate in favor of `ManifoldDistanceOp`.

---

## M3 [MEDIUM]: Poincare denominator clamp hardcoded at 1e-15

**Location**: manifold.rs:591-594, 781

**Bug**: `fmax(1e-15, ...)` doesn't scale with curvature κ. For extremely large κ (strong curvature, small ball), the clamp may be too loose relative to the actual denominator scale.

**Test**: κ=1e10, points near boundary at radius 1/√(1e10) = 1e-5.

**Fix**: Scale clamp with κ: `fmax(1e-15 / (kappa * kappa), ...)`.

---

## M4 [MEDIUM]: Sphere projection div-by-zero

**Location**: manifold.rs:303-306

**Bug**: `point[k] * (r / sqrt(norm_sq))` — if norm_sq=0 (zero vector), produces NaN/Inf. Unlike Poincare projection which has a threshold check.

**Test**: `project_expr` applied to the zero vector `(0, 0, 0)`.

**Fix**: `norm_sq > eps ? point[k] * (r / sqrt(norm_sq)) : 0.0`.

---

## M5 [MEDIUM]: distance_is_dissimilarity wrong for Sphere

**Location**: manifold.rs:337-339

**Bug**: Returns `false` for Sphere (treating it as similarity), but `1 - dot(a,b)` IS a dissimilarity: identical→0, orthogonal→1, opposite→2. Any algorithm that checks this flag (e.g., k-nearest neighbors) will interpret distances backwards for Sphere.

**Test**: `Manifold::Sphere { .. }.distance_is_dissimilarity()` returns `false`, should be `true`.

**Fix**: Return `true` for Sphere.

---

## M6 [MEDIUM]: Poincare gradient scale for out-of-ball points

**Location**: manifold.rs:261-265

**Bug**: `pow((1.0 - |c| * x_norm_sq) / 2.0, 2.0)` — when point is outside ball (x_norm_sq > 1/|c|), the inner term is negative, squaring makes it positive. Gradient step pushes point further out.

**Test**: Poincare κ=1, point at (1.5, 0) — outside ball of radius 1.

**Fix**: `pow(fmax(0.0, 1.0 - |c|*x_norm_sq) / 2.0, 2.0)` — zero gradient for out-of-ball.

---

## M7 [LOW]: SphericalGeodesic zero-vector → fabricated π/2

**Location**: manifold.rs:601, 795

`fmax(1e-30, sqrt(0*anything))` → `arccos(0/1e-30) = arccos(0) = π/2`. Fabricated answer.

## M8 [LOW]: Poincare projection margin hardcoded 1e-5

**Location**: manifold.rs:297

`(1.0 - 1e-5)` relative margin. For very large κ, the margin in absolute terms approaches f64 precision.

## M9 [LOW]: ManifoldMixture allows negative individual weights

**Location**: manifold.rs:428-429

`normalize()` checks total > 0 but not individual weights. Weights [2.0, -0.5] → normalized [1.33, -0.33] — negative mixture weight produces negative distances.
