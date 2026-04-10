# Finding 4: Poincare Ball Boundary Instability — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN numerically
**Code**: `crates/tam-gpu/src/cpu.rs:406`, `crates/tambear/src/manifold.rs:592`

---

## The Bug

The Poincare distance formula has denominator `(1 - kappa*||x||^2)(1 - kappa*||y||^2)` which approaches zero as points approach the ball boundary. The current code clamps this to `max(denom, 1e-15)`.

This is wrong in two ways:
1. Before the clamp triggers, noise amplification is already catastrophic
2. When the clamp triggers, it produces a WRONG distance, not an approximate one

---

## Proof Table 1: Conformal Factor Explosion

The Poincare metric amplifies Euclidean rounding error by `4/(1-r^2)^2`:

| ||x||^2 | Conformal Factor | f64 ULP (~1e-16) becomes | Status |
|----------|-----------------|--------------------------|--------|
| 0.0 | 4 | 4e-16 | Safe |
| 0.5 | 16 | 2e-15 | Safe |
| 0.9 | 400 | 4e-14 | OK |
| 0.99 | 40,000 | 4e-12 | OK |
| 0.999 | 4,000,000 | 4e-10 | Visible |
| 0.9999 | 4e8 | 4e-8 | Visible |
| 0.99999 | 4e10 | 4e-6 | VISIBLE |
| 0.999999 | 4e12 | 4e-4 | **DOMINANT** |
| 0.9999999 | 4e14 | **4e-2 (4%)** | **DOMINANT** |

At `||x||^2 = 0.9999999`, f64 rounding alone introduces **4% distance error**.

---

## Proof Table 2: Denominator Collapse

| ||x||^2 | 1-k||x||^2 | denom | noise_amp | Status |
|----------|------------|-------|-----------|--------|
| 0.99 | 1e-2 | 1e-4 | 2.2e-14 | OK |
| 0.999 | 1e-3 | 1e-6 | 2.2e-13 | OK |
| 0.9999 | 1e-4 | 1e-8 | 2.2e-12 | OK |
| 0.99999 | 1e-5 | 1e-10 | 2.2e-11 | OK |
| 0.999999 | 1e-6 | 1e-12 | 2.2e-10 | OK |
| 0.9999999 | 1e-7 | 1e-14 | 2.2e-9 | OK |
| 0.99999999 | 1e-8 | 1e-16 | 2.2e-8 | **CLAMPED** |

Clamp triggers at `||x||^2 > 0.99999999` (8 nines). But noise amplification is already 10^9 at 7 nines.

---

## Proof Table 3: f32 Precision Cliff

| ||x||^2 | f64: 1-||x||^2 | f32: 1-||x||^2 | Status |
|----------|---------------|---------------|--------|
| 0.999 | 1e-3 | 1e-3 | OK |
| 0.9999 | 1e-4 | 1e-4 | MARGINAL |
| 0.99999 | 1e-5 | 1e-5 | MARGINAL |
| 0.999999 | 1e-6 | 1.01e-6 | BAD |
| 0.9999999 | 1e-7 | 1.19e-7 | **BROKEN** |

WGSL fallback to Euclidean (already in code) is the correct decision for f32.

---

## Test Vectors

### TV-POINCARE-01: Interior points (must work)
```
kappa = 1.0
x = [0.3, 0.4]  (||x||^2 = 0.25)
y = [0.1, 0.2]  (||y||^2 = 0.05)
Expected: well-defined, denom = (1-0.25)(1-0.05) = 0.7125
Status: SAFE
```

### TV-POINCARE-02: Moderate radius
```
kappa = 1.0
x = [0.7, 0.0]  (||x||^2 = 0.49)
y = [0.0, 0.7]  (||y||^2 = 0.49)
denom = (0.51)^2 = 0.2601
Status: SAFE
```

### TV-POINCARE-03: Near boundary, same direction
```
kappa = 1.0
r = sqrt(0.9999)
x = [r, 0.0]
y = [r*cos(0.001), r*sin(0.001)]
denom ~ (1e-4)^2 = 1e-8
Expected distance: ~11.99 (large but finite)
Conformal factor: 4e8
Status: MARGINAL — distance is computed but sensitive to perturbation
```

### TV-POINCARE-04: Very near boundary
```
kappa = 1.0
r = sqrt(0.999999)
x = [r, 0.0]
y = [r*cos(0.001), r*sin(0.001)]
denom ~ (1e-6)^2 = 1e-12
Expected distance: ~30.4
Conformal factor: 4e12
Status: f64 rounding error amplified by 4e12 → ~0.04% distance error
```

### TV-POINCARE-05: Boundary death zone
```
kappa = 1.0
r = sqrt(0.9999999)
x = [r, 0.0]
y = [r*cos(0.001), r*sin(0.001)]
denom ~ 1e-14
Expected distance: ~39.6
Conformal factor: 4e14
Status: **4% distance error from f64 rounding alone**
```

### TV-POINCARE-06: Clamp regime
```
kappa = 1.0
r = sqrt(0.99999999)
x = [r, 0.0]
y = [r*cos(0.001), r*sin(0.001)]
True denom: ~1e-16 (below clamp threshold)
Clamped denom: 1e-15
Status: **WRONG — clamped distance != true distance**
```

### TV-POINCARE-07: One point interior, one boundary
```
kappa = 1.0
x = [0.0, 0.0]   (origin)
y = [sqrt(0.9999999), 0.0]  (near boundary)
denom = 1.0 * 1e-7 = 1e-7
Status: Better than both-at-boundary, but still amplified
```

### TV-POINCARE-08: Non-unit curvature
```
kappa = 0.5
r = sqrt(1.9998)  (boundary is at ||x||^2 = 1/kappa = 2.0)
x = [r, 0.0]
y = [r*cos(0.001), r*sin(0.001)]
denom = (1 - 0.5*1.9998)^2 = (0.0001)^2 = 1e-8
Status: Same boundary behavior, just at different radius
```

---

## Downstream Impact

### DBSCAN
- eps-neighborhoods at boundary are unreliable
- Core/border classification can change with rounding
- Cluster assignments for boundary-region points are non-deterministic

### KNN
- k-nearest neighbor ordering near boundary is noise-dependent
- Distance-weighted KNN amplifies the error further

### KMeans
- Centroid updates using boundary-point distances are noisy
- Convergence may oscillate for boundary clusters

---

## Recommended Fixes

1. **V-column confidence**: `confidence = (1-k*||x||^2) * (1-k*||y||^2)`. Downstream algorithms weight by confidence. This is the tambear-native fix — no production gates, just metadata.

2. **Log-space arithmetic**: Compute `log(1-k*||x||^2)` to avoid catastrophic subtraction. Then `log(denom) = log(1-k*||x||^2) + log(1-k*||y||^2)`. This recovers full precision.

3. **Widen clamp with sentinel**: Change clamp from 1e-15 to 1e-10 and mark the distance as boundary-regime.

4. **Klein or half-space model**: For computation, use a model where the boundary isn't singular. Project back for output.
