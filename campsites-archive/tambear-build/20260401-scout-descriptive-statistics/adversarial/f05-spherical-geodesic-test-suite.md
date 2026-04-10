# Finding 5: SphericalGeodesic Zero-Vector Bug — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN numerically
**Code**: `crates/tam-gpu/src/cpu.rs:426`, `crates/tambear/src/manifold.rs:601`

---

## The Bug

When either input vector has `||x||^2 * ||y||^2 < 1e-60`, the clamp kicks in:
```
denom = max(sq_norm_x * sq_norm_y, 1e-60).sqrt()  // = 1e-30
cos_theta = dot_prod / denom                        // can be anything
cos_theta = clamp(cos_theta, -1, 1)                 // forces into valid range
return acos(cos_theta)                               // but answer is WRONG
```

For zero vector: `dot = 0`, `denom = 1e-30`, `cos_theta = 0`, `acos(0) = pi/2`.
Every direction produces pi/2. The angle becomes meaningless.

---

## Proof Table: Direction Collapse at Small Magnitude

| ||x|| | Direction | True Angle | Computed | Status |
|-------|-----------|------------|----------|--------|
| 1e-6 | parallel | 0.000 | 0.000 | OK |
| 1e-6 | perpendicular | 1.571 | 1.571 | OK |
| 1e-6 | anti-parallel | 3.142 | 3.142 | OK |
| 1e-10 | parallel | 0.000 | 0.000 | OK |
| 1e-10 | perpendicular | 1.571 | 1.571 | OK |
| 1e-20 | parallel | 0.000 | 0.000 | OK |
| 1e-20 | perpendicular | 1.571 | 1.571 | OK |
| **1e-50** | **parallel** | **0.000** | **1.571** | **WRONG** |
| **1e-50** | **anti-parallel** | **3.142** | **1.571** | **WRONG** |
| **1e-50** | **45 degrees** | **0.785** | **1.571** | **WRONG** |
| 1e-100 | parallel | 0.000 | 1.571 | WRONG |
| 1e-150 | parallel | 0.000 | 1.571 | WRONG |

**Breakpoint**: `||x|| = 1e-50` (because `||x||^2 * ||y||^2 = 1e-100 < 1e-60` triggers clamp).
With unit y: `||x||^2 * 1.0 < 1e-60` means `||x|| < 1e-30`.
With `||x|| = ||y||`: `||x||^4 < 1e-60` means `||x|| < 1e-15`.

---

## Proof Table: Two Near-Zero Vectors

Both vectors at same magnitude, 0.5 radians apart:

| ||x|| = ||y|| | True Angle | Computed | Status |
|----------------|------------|----------|--------|
| 1.0 | 0.5000 | 0.5000 | OK |
| 1e-3 | 0.5000 | 0.5000 | OK |
| 1e-6 | 0.5000 | 0.5000 | OK |
| 1e-10 | 0.5000 | 0.5000 | OK |
| **1e-20** | **0.5000** | **1.5708** | **WRONG** |
| 1e-50 | 0.5000 | 1.5708 | WRONG |

**Breakpoint**: `||x||^4 < 1e-60` → `||x|| < 1e-15`. Confirmed: breaks between 1e-10 and 1e-20.

---

## Proof Table: Denormal Vectors

| x | y | True | Computed | Note |
|---|---|------|----------|------|
| [5e-324, 0] | [5e-324, 0] | 0.0 | 1.571 | Parallel → pi/2 WRONG |
| [5e-324, 0] | [0, 5e-324] | pi/2 | 1.571 | Perpendicular → accidentally correct |
| [5e-324, 0] | [-5e-324, 0] | pi | 1.571 | Anti-parallel → pi/2 WRONG |
| [1e-200, 0] | [1e-200, 0] | 0.0 | 1.571 | Parallel → pi/2 WRONG |

---

## Test Vectors

### TV-GEO-01: Normal vectors (must work)
```
x = [1.0, 0.0], y = [0.0, 1.0]
Expected: pi/2 = 1.5707963268
```

### TV-GEO-02: Same direction
```
x = [3.0, 4.0], y = [6.0, 8.0]
Expected: 0.0 (parallel)
```

### TV-GEO-03: Opposite direction
```
x = [1.0, 0.0], y = [-1.0, 0.0]
Expected: pi = 3.1415926536
```

### TV-GEO-04: Zero vector x
```
x = [0.0, 0.0], y = [1.0, 0.0]
Expected: NaN (undefined)
ACTUAL: 1.5708 (pi/2) — WRONG
```

### TV-GEO-05: Both zero
```
x = [0.0, 0.0], y = [0.0, 0.0]
Expected: NaN
ACTUAL: 1.5708 (pi/2) — WRONG
```

### TV-GEO-06: Near-zero parallel (||x|| = 1e-20)
```
x = [1e-20, 0.0], y = [1.0, 0.0]
Expected: 0.0 (parallel)
ACTUAL: 0.0 — OK (mixed magnitudes: ||x||^2 * ||y||^2 = 1e-40 > 1e-60)
```

### TV-GEO-07: Both near-zero (||x|| = ||y|| = 1e-20)
```
x = [1e-20, 0.0], y = [1e-20*cos(0.5), 1e-20*sin(0.5)]
Expected: 0.5
ACTUAL: 1.5708 — WRONG (||x||^2 * ||y||^2 = 1e-80 < 1e-60)
```

### TV-GEO-08: acos instability near 0
```
eps = 1e-15
x = [1.0, 0.0]
y = [1-eps, sqrt(2*eps)]
Expected angle: ~sqrt(2*eps) = 4.47e-8
ACTUAL: drifting (0.04% error)
Fix: use atan2(||cross||, dot) instead of acos(dot/norms)
```

---

## Downstream Impact

- DBSCAN with SphericalGeodesic manifold: if any data point is near-zero, its geodesic distances to ALL other points collapse to pi/2. This point gets mis-clustered.
- KMeans: if a centroid drifts toward zero, all assignments to it become distance-pi/2 regardless of direction.
- Any algorithm that normalizes vectors to unit sphere first is immune (zero becomes undefined before reaching geodesic).

---

## Recommended Fixes

1. **Return NaN for zero vectors**: If `||x||^2 < eps` or `||y||^2 < eps`, return NaN. Don't fake pi/2.
2. **Use atan2**: `atan2(||x cross y||, x dot y)` is better conditioned than `acos(dot/norms)` for near-parallel and near-anti-parallel vectors.
3. **Document the threshold**: The clamp triggers at `||x|| * ||y|| < 1e-30`. Any data with components below 1e-15 is at risk.
