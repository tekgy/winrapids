# Family 01: Distance & Similarity — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN numerically
**Code**: `crates/tam-gpu/src/cpu.rs:290-510`
**Proof script**: `docs/research/notebooks/f01-distance-adversarial-proof.py`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| dot_product | cpu.rs:318-328 | MEDIUM (cancellation at scale) |
| l2_distance | cpu.rs:331-342 | **HIGH** (breaks at offset 1e12) |
| covariance | cpu.rs:347-358 | MEDIUM (pre-centering not enforced) |
| sphere (cosine) | cpu.rs:431-442 | **HIGH** (cancellation + not a metric) |
| softmax_weighted | cpu.rs:363-383 | LOW (NaN handling wrong, else good) |
| manifold_distance | cpu.rs:388-455 | See F04/F05 separate suites |
| manifold_mixture | cpu.rs:458-498 | Same issues as manifold_distance |

---

## Finding F01-1: L2 Distance BREAKS at Large Offset

**Surprise**: L2 was assumed robust because `a[i]-b[i]` cancels the offset. WRONG.
When the difference is smaller than the ULP of the offset, subtraction rounds to zero.

| Offset | Spread | True L2sq | Computed | Rel Error | Status |
|--------|--------|-----------|----------|-----------|--------|
| 0 | 0.001 | 2.500e-6 | 2.500e-6 | 1.7e-16 | OK |
| 1e4 | 0.001 | 2.500e-6 | 2.500e-6 | 3.2e-10 | MARGINAL |
| 1e8 | 0.001 | 2.500e-6 | 2.500e-6 | 7.9e-6 | MARGINAL |
| **1e12** | **0.001** | **2.500e-6** | **2.384e-6** | **4.6e-2** | **BROKEN** |
| **1e15** | **0.001** | **2.500e-6** | **0.0** | **1.0** | **BROKEN** |

At offset 1e12: `a[i] - b[i]` where `a[i] = 1e12 + i*0.001` and `b[i] = 1e12 + (i+0.5)*0.001`.
The difference is 0.0005, but `1e12 + 0.0005` rounds to `1e12` in f64. The subtraction returns 0.

**Impact**: Financial data with prices in the millions (offset ~1e6) and differences in cents (spread ~0.01) is in the MARGINAL zone. High-frequency tick data with nanosecond timestamps (offset ~1e18) would be BROKEN.

**Fix**: For L2 between points at known large offset, center both sets of coordinates first. Or document the precision requirements: `|offset/spread| < 1e8` for reliable L2 at f64.

---

## Finding F01-2: Cosine Distance — Three Independent Bugs

### Bug 2a: Non-Unit Input Not Checked
```
sphere op: c[i,j] = 1 - dot(a,b)
With a=[3,4], b=[4,3]: 1 - 24 = -23.0
Correct cosine distance: 1 - 24/25 = 0.04
```
**WRONG by 23.04**. No runtime check that inputs are unit-normalized.

### Bug 2b: Triangle Inequality VIOLATION
```
A = [2, 0], B = [0, 0.5], C = [-1, 0]
d(A,B) = 1.0, d(B,C) = 1.0, d(A,C) = 3.0
Triangle: 3.0 > 2.0 — VIOLATED
```
Cosine distance (1-dot) is NOT a metric for non-unit vectors. Any algorithm assuming metric properties (DBSCAN eps-neighborhoods, KNN) will produce wrong results if fed cosine distances from non-unit inputs.

### Bug 2c: Cancellation for Nearly-Identical Vectors
At angle < 1e-8 radians: `1 - cos(angle) < 5e-17`, which is below f64 ULP for values near 1.0. Result rounds to exactly 0.0.

**Fix**: Use `2*sin^2(angle/2)` which equals `1-cos(angle)` but is numerically stable. Or use `sphere_geodesic` (arccos) which avoids the 1-dot subtraction entirely.

---

## Finding F01-3: Softmax NaN/Inf Handling

The online softmax algorithm doesn't correctly handle special values:

| Input | Expected | Actual | Bug |
|-------|----------|--------|-----|
| scores=[1, NaN, 3] | NaN | **0.0** | NaN comparison fails silently |
| scores=[1, +Inf, 3] | values[1] | **0.0** | Inf breaks the max-tracking |
| scores=[1, -Inf, 3] | weighted(1,3) | 27.6 | Correct |

The `score > max_val` comparison returns false when `score` is NaN, so NaN elements are silently ignored with weight `exp(NaN - max_val) = NaN`, which poisons `exp_sum` and `weighted_sum`. The final `weighted_sum / exp_sum` gives `NaN/NaN = NaN`... except in the test it gave 0.0. Investigating: the online algorithm's rescaling may mask the NaN.

**Fix**: Check `score.is_nan()` or `!score.is_finite()` before the comparison.

---

## Finding F01-4: Dot Product Cancellation (Lower Priority)

Dot product cancellation occurs when the true result is much smaller than the individual terms:

| Scenario | True dot | Error | Status |
|----------|----------|-------|--------|
| dim=10, offset=1e12 | 1e25 | 2.15e9 | Relative error 2e-16 = OK |
| Nearly orthogonal, offset=1e8 | ~0 | visible | MEDIUM concern |

For the dot product as implemented, the dominant term is `dim * offset^2` and the cancellation is between that and `sum(i^2)`. When `offset^2` dominates, the relative error is small even though absolute error is large.

**Verdict**: Dot product cancellation is less severe than variance cancellation because the result isn't expected to be small relative to the terms. The problematic case is when the dot product SHOULD be near zero (nearly orthogonal vectors at large offset) — but this is rare in practice.

---

## Test Vectors for Rust Tests

### TV-F01-DOT-01: Basic correctness
```
a = [1, 2, 3], b = [4, 5, 6]
expected: 32.0
```

### TV-F01-DOT-02: Zero vectors
```
a = [0, 0, 0], b = [1, 2, 3]
expected: 0.0
```

### TV-F01-DOT-03: NaN propagation
```
a = [1, NaN, 3], b = [4, 5, 6]
expected: NaN
```

### TV-F01-L2-01: Identical points
```
a = b = [1e8, 1e8+1, 1e8+2]
expected: 0.0 (exactly)
```

### TV-F01-L2-02: Large offset, small spread
```
a = [1e12 + i*0.001 for i in 0..10]
b = [1e12 + (i+0.5)*0.001 for i in 0..10]
expected: 2.5e-6
tolerance: 5% (currently BROKEN at this offset)
```

### TV-F01-COS-01: Unit vectors, 90 degrees
```
a = [1, 0], b = [0, 1]
expected: 1.0
```

### TV-F01-COS-02: Non-unit detection
```
a = [3, 4], b = [4, 3]
MUST either: normalize internally, or error on non-unit input
```

### TV-F01-COS-03: Nearly parallel
```
a = [1, 0], b = [cos(1e-10), sin(1e-10)]
expected: ~5e-21
actual: 0.0 (BROKEN)
```

### TV-F01-SOFT-01: Shift invariance
```
scores_a = [0, 1, 2], scores_b = [1000, 1001, 1002]
values = [1, 2, 3]
result_a must equal result_b
```

### TV-F01-SOFT-02: NaN in scores
```
scores = [1, NaN, 3], values = [10, 20, 30]
expected: NaN (currently returns 0.0)
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F01-1: L2 offset | **HIGH** | Wrong distances for far-apart-offset close-spread data | Center before L2, or document precision boundary |
| F01-2a: Cosine non-unit | **HIGH** | Wrong results, silent | Runtime check or auto-normalize |
| F01-2b: Cosine not metric | **MEDIUM** | Triangle inequality violated | Use angular distance (arccos) for metric algorithms |
| F01-2c: Cosine cancellation | **MEDIUM** | Zero for angle < 1e-8 | Use 2*sin^2(angle/2) |
| F01-3: Softmax NaN | **LOW** | Incorrect handling of special values | Check is_finite() |
| F01-4: Dot cancellation | **LOW** | Rare in practice | Kahan summation for precision-critical paths |
