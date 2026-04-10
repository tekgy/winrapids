# Family 30: Spatial Statistics — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/spatial.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| Euclidean 2D | spatial.rs:39-41 | OK |
| Haversine | spatial.rs:46-55 | OK |
| Empirical variogram | spatial.rs:70-98 | OK |
| Spherical variogram | spatial.rs:112-120 | OK |
| Exponential variogram | spatial.rs:123-126 | OK |
| Gaussian variogram | spatial.rs:129-132 | OK |
| Ordinary kriging | spatial.rs:147-208 | OK (performance note) |
| Moran's I | spatial.rs:301-319 | OK |
| Geary's C | spatial.rs:326-343 | OK |
| Ripley's K | spatial.rs:353-370 | OK |
| Ripley's L | spatial.rs:375-380 | OK |
| Nearest-neighbor distances | spatial.rs:385-399 | **MEDIUM** (NaN panic) |
| Clark-Evans R | spatial.rs:405-413 | OK |
| KNN weights | spatial.rs:252-265 | **MEDIUM** (NaN panic) |
| Distance band weights | spatial.rs:268-281 | OK |
| Row standardize | spatial.rs:284-291 | OK |

---

## Finding F30-1: KNN Spatial Weights NaN Panic (MEDIUM)

**Bug**: `SpatialWeights::knn` at line 260 uses `partial_cmp(&b.1).unwrap()` in the distance sort. If any point has NaN coordinates, the Euclidean distance will be NaN, and the sort panics.

**Impact**: Thread panic when point data contains NaN coordinates.

**Fix**: `partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)`

---

## Finding F30-2: Nearest-Neighbor Distances NaN Panic (MEDIUM)

**Bug**: `nn_distances` at line 397 uses `partial_cmp(b).unwrap()` in the final sort. If any computed distance is NaN, this panics.

**Impact**: Thread panic on NaN coordinate data.

**Fix**: Same as F30-1.

---

## Finding F30-3: Kriging Re-solves System Per Query (Performance)

**Note**: `ordinary_kriging` clones the entire n×n matrix for each query point (line 186), then solves from scratch. The LHS matrix A is the same for all queries — only the RHS b changes. An LU factorization of A could be computed once and reused.

**Impact**: Performance only. O(nq × n³) instead of O(n³ + nq × n²).

---

## Positive Findings

**Variogram models are all correct.** Spherical has proper cubic interpolation with range cutoff. Exponential uses practical range factor of 3. All models return 0 at h=0 (not nugget — correct for semivariogram).

**Moran's I and Geary's C are correct.** Both center by mean. Division by S0 (total weight) handles non-row-standardized weights.

**Kriging system is correct.** Extended kriging matrix with Lagrange multiplier for unbiasedness constraint. Covariance = sill + nugget - variogram conversion is standard.

**Haversine is correct.** Uses proper formula with asin(sqrt(a)) instead of atan2 — both are correct, but asin is simpler.

---

## Test Vectors

### TV-F30-KNN-01: KNN with NaN coordinates (BUG CHECK)
```
points = [(0,0), (1,1), (NaN,NaN), (2,2)], k=2
Expected: graceful handling
Currently: PANIC
```

### TV-F30-KRIG-01: Kriging reproduces known values
```
points = [(0,0,1), (1,0,2), (0,1,3), (1,1,4)]
model = spherical(nugget=0, sill=5, range=2)
query at (0,0): predicted should be ≈ 1.0
```

### TV-F30-MORAN-01: Moran's I for clustered data
```
values = [10, 9, 8, 2, 1, 0] on a line
weights = knn(k=2)
Expected: I > 0 (positive spatial autocorrelation)
```

### TV-F30-VARIO-01: Spherical variogram at boundary
```
model = (nugget=0, sill=1, range=10)
γ(10.0) should equal 1.0 (nugget + sill)
γ(5.0) should be in (0, 1)
γ(0.0) should be 0.0
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F30-1: KNN weights NaN panic | **MEDIUM** | Thread panic | unwrap_or(Equal) |
| F30-2: NN distances NaN panic | **MEDIUM** | Thread panic | unwrap_or(Equal) |
| F30-3: Kriging perf | **Performance** | O(nq·n³) | Factor LU once |
