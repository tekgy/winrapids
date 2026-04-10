# Family 26: Complexity & Chaos — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN by code review
**Code**: `crates/tambear/src/complexity.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| sample_entropy | complexity.rs:46-57 | OK (no self-matches, O(n²m)) |
| approx_entropy | complexity.rs:66-70 | OK (with self-matches, as per Pincus) |
| permutation_entropy | complexity.rs:131-153 | **MEDIUM** (O(m!) memory) |
| normalized_permutation_entropy | complexity.rs:157-163 | OK |
| hurst_rs | complexity.rs:193-251 | OK (correctly centered variance) |
| dfa | complexity.rs:281-332 | OK (uses ols_slope, correctly centered) |
| linear_fit_segment | complexity.rs:335-355 | **LOW** (naive OLS, but x=0..n-1) |
| higuchi_fd | complexity.rs:367-398 | OK |
| lempel_ziv_complexity | complexity.rs:411-449 | **HIGH** (NaN panic in sort) |
| correlation_dimension | complexity.rs:464-512 | **HIGH** (NaN panic in sort) |
| largest_lyapunov | complexity.rs:530-589 | OK |
| ols_slope | complexity.rs:254-267 | OK (correctly centered by mean) |

---

## Finding F26-1: NaN Panics in Sort (HIGH)

**Bug**: Two functions use `a.partial_cmp(b).unwrap()` to sort f64 data. When any value is NaN, `partial_cmp` returns `None`, and `unwrap()` panics.

**Locations**:
- `lempel_ziv_complexity` line 417: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - Sorts user data directly to find median for binarization
- `correlation_dimension` line 483: `distances.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - Sorts pairwise distances (NaN if any input is NaN, since `NaN - x = NaN`)

**Impact**: Any NaN in input data → thread panic. No graceful degradation.

**Fix**: Replace `unwrap()` with `unwrap_or(std::cmp::Ordering::Equal)`, or pre-filter NaN, or use `f64::total_cmp` (Rust 1.62+).

**Note**: This pattern is widespread in the codebase — also affects `nonparametric.rs` (KS tests), `robust.rs` (MAD, Qn, Sn, LTS, MCD, medcouple), `descriptive.rs`, `knn.rs`, and `signal_processing.rs`. All will panic on NaN input.

---

## Finding F26-2: Permutation Entropy O(m!) Memory (MEDIUM)

**Bug**: `permutation_entropy` allocates `vec![0usize; factorial(m)]`. Memory:

| m | factorial(m) | Memory |
|---|-------------|--------|
| 8 | 40,320 | 323 KB |
| 10 | 3,628,800 | 29 MB |
| 11 | 39,916,800 | 320 MB |
| 12 | 479,001,600 | 3.8 GB (OOM) |
| 13 | 6,227,020,800 | 50 GB |

**Impact**: m > 10 risks OOM. m > 20 overflows usize.

**Fix**: Cap m at a reasonable maximum (m ≤ 8 is standard for permutation entropy). Or use a HashMap instead of a fixed-size array for pattern counting.

---

## Finding F26-3: Naive OLS in linear_fit_segment (LOW)

Already documented in naive-formula-codebase-sweep.md.

`denom = n * sxx - sx * sx` with x = 0..n-1 (integer). Denom is exact for integers. Slope numerator `n * sxy - sx * sy` could cancel if y-values are very large relative to the slope, but in DFA context the y-values are profile segments (cumulative deviations from mean), which are typically bounded.

---

## Positive Findings

**ols_slope (line 254-267) is correctly centered.** Computes `mx`, `my` first, then `dx = x[i] - mx`. This is the gold standard for OLS on arbitrary data. Used by `hurst_rs`, `dfa`, `higuchi_fd`, `correlation_dimension`, and `largest_lyapunov`.

**hurst_rs variance (line 228-229) is correctly centered.** Uses `(x - mean).powi(2)`, not the naive formula.

**All log-log regression methods** feed through `ols_slope`, which is correct. The regression quality is sound.

---

## Test Vectors

### TV-F26-NAN-01: NaN in LZ (BUG CHECK)
```
data = [1.0, 2.0, NaN, 4.0, 5.0]
Expected: NaN or graceful error
Currently: PANIC
```

### TV-F26-NAN-02: NaN in correlation_dimension (BUG CHECK)
```
data = [1.0, 2.0, NaN, 4.0, 5.0, ...] (at least 50 points)
Expected: NaN or graceful error
Currently: PANIC
```

### TV-F26-PE-01: Permutation entropy m too large
```
data = [random; 100], m = 12
Expected: NaN or error (3.8 GB allocation)
Currently: OOM panic
```

### TV-F26-SE-01: Sample entropy constant signal
```
data = [1.0; 100], m=2, r=0.2
Expected: SampEn ≈ 0 (all templates match)
```

### TV-F26-PE-02: Permutation entropy monotone
```
data = [0, 1, 2, ..., 99], m=3, tau=1
Expected: PE = 0 (only one pattern: increasing)
```

### TV-F26-H-01: Hurst white noise
```
data = LCG(2000 points)
Expected: H ≈ 0.5 (within [0.3, 0.7])
```

### TV-F26-DFA-01: DFA Brownian motion
```
data = cumsum(white_noise(1000))
Expected: α ≈ 1.5
```

### TV-F26-LZ-01: LZ complexity periodic
```
data = [0,1,0,1,...] (200 points)
Expected: LZ < 0.5
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F26-1: NaN panic | **HIGH** | Thread panic on NaN input | Use total_cmp or unwrap_or |
| F26-2: PE memory | **MEDIUM** | OOM for m > 10 | Cap m or use HashMap |
| F26-3: Naive OLS | **LOW** | Marginal risk (integer x) | Center y by mean |
