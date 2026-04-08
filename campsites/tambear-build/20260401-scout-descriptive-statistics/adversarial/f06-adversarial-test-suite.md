# Family 06: Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Purpose**: Test vectors that MUST pass before any F06 implementation is considered correct. Pathmaker codes AGAINST these from the start. Scientist validates AFTER.

---

## Precision Map: When Naive Moments Break

The naive formula `var = E[x^2] - E[x]^2` loses `2*log10(offset/std)` digits.
Higher moments lose proportionally more: skewness loses 3x, kurtosis loses 4x.

| Offset | Std | Digits Lost (var) | f64 Status | f32 Status |
|--------|-----|-------------------|------------|------------|
| 1e2 | 1.0 | 4.0 | OK | OK |
| 1e3 | 1.0 | 6.0 | OK | MARGINAL |
| 1e4 | 1.0 | 8.0 | OK | BROKEN |
| 1e6 | 1.0 | 12.0 | OK | BROKEN |
| 1e8 | 1.0 | 16.0 | **BROKEN** | BROKEN |
| 1e8 | 0.01 | 20.0 | BROKEN | BROKEN |

Higher moments are worse:

| Offset | Statistic | Digits Lost | f64 Status |
|--------|-----------|-------------|------------|
| 1e4 | Kurtosis | 16.0 | **BROKEN** |
| 1e6 | Skewness | 18.0 | **BROKEN** |
| 1e6 | Kurtosis | 24.0 | BROKEN |
| 1e8 | Variance | 16.0 | BROKEN |
| 1e8 | Skewness | 24.0 | BROKEN |

**Implication**: Any implementation using naive power sums MUST use centering (RefCentered approach) for ALL moments, not just variance. Kurtosis via naive Sx^4 breaks at offset 1e4 in f64 -- that's nearly ALL real-world data.

---

## Test Vectors

### Category A: Happy Path (Must Work)

**TC01: Normal small data**
```
input:    [1.0, 2.0, 3.0, 4.0, 5.0]
n:        5
mean:     3.0
var_pop:  2.0
var_sam:  2.5
std_pop:  1.4142135624
min:      1.0
max:      5.0
skew:     0.0 (exact)
kurt_exc: -1.3
```

**TC05: Three elements (boundary for skewness bias correction)**
```
input:    [1.0, 2.0, 3.0]
n:        3
mean:     2.0
var_pop:  0.6666666667
var_sam:  1.0
skew_pop: 0.0 (exact, symmetric)
Note:     Fisher bias-corrected skewness has (n-2) in denominator = 1. Fine.
```

**TC09: Symmetric distribution**
```
input:    [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
n:        7
mean:     0.0 (exact)
var_pop:  4.0
skew:     0.0 (exact)
kurt_exc: -1.25
```

**TC10: Perfect bimodal**
```
input:    [0.0]*50 + [100.0]*50
n:        100
mean:     50.0
var_pop:  2500.0
skew:     0.0 (exact, symmetric)
kurt_exc: -2.0 (exact, lighter than normal)
```

### Category B: Edge Cases (Boundary Behavior)

**TC02: All identical values**
```
input:    [42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0]
n:        10
mean:     42.0
var_pop:  0.0 (exact)
var_sam:  0.0 (exact)
std:      0.0
skew:     NaN (0/0 -- undefined when std=0)
kurt:     NaN (0/0 -- undefined when std=0)
min:      42.0
max:      42.0
MUST NOT: panic, return Inf, return a finite nonzero value for skew/kurt
```

**TC03: Single element (n=1)**
```
input:    [7.0]
n:        1
mean:     7.0
var_pop:  0.0
var_sam:  NaN (0/0, denominator n-1=0)
std_pop:  0.0
min:      7.0
max:      7.0
skew:     NaN (undefined for n<3)
kurt:     NaN (undefined for n<4)
```

**TC04: Two elements (n=2)**
```
input:    [1.0, 3.0]
n:        2
mean:     2.0
var_pop:  1.0
var_sam:  2.0
skew:     NaN (Fisher bias-corrected has (n-2)=0 in denominator; population skew is 0 but meaningless)
kurt:     NaN (n<4)
```

**TC-EMPTY: Empty input (n=0)**
```
input:    []
n:        0
mean:     NaN
var:      NaN
All stats: NaN or appropriate sentinel
MUST NOT: panic, segfault, divide by zero without NaN result
```

**TC-N3-KURT: n=3, kurtosis denominator**
```
input:    [1.0, 2.0, 100.0]
n:        3
Note:     Bias-corrected kurtosis has (n-2)(n-3) = 1*0 = 0 in denominator.
          Population kurtosis is computable. Bias-corrected is NaN.
          Document which version tambear returns and handle accordingly.
```

### Category C: Catastrophic Cancellation (Variance/Std)

These test the core numerical vulnerability. Naive `E[x^2]-E[x]^2` fails on ALL of these.

**TC06: Large offset, integer spread**
```
input:    [1e8, 1e8+1, 1e8+2, 1e8+3, 1e8+4]
mean:     100000002.0
var_pop:  2.0
var_sam:  2.5
NAIVE:    0.0 or garbage (16 digits lost at f64)
CENTERED: 2.0 (trivial: variance of [0,1,2,3,4])
TOLERANCE: 1e-6 relative error acceptable. Larger = implementation bug.
```

**TC07: Huge offset (f64 total precision loss)**
```
input:    [1e15+1, 1e15+2, 1e15+3]
var_pop:  0.6666666667
NAIVE:    -140737488355328.0 (NEGATIVE! confirmed numerically)
CENTERED: 0.6666666667
TOLERANCE: 1e-6 relative. If negative: FAIL.
```

**TC-CANCEL-FINANCIAL: Realistic financial data**
```
input:    [234567.89 + i*0.01 for i in range(100)]
var_pop:  0.08332500
NAIVE:    0.08332062 (0.005% error -- marginal but demonstrable)
CENTERED: 0.08332500
TOLERANCE: 1e-8 relative (centered should be exact to f64 precision)
```

**TC-CANCEL-GRAD: The destruction gradient (vary offset, fixed spread)**
```
For each offset in [1e4, 1e6, 1e8, 1e10, 1e12, 1e14]:
  input:  [offset + i*0.001 for i in range(100)]
  Expected var_pop: 8.3325e-04 (stable)

  PASS criteria: relative error < 1% for ALL offsets
  This is the DEFINITIVE test for whether centering works.
```

### Category D: Catastrophic Cancellation (Higher Moments)

**TC-SKEW-CANCEL: Skewness at large offset**
```
input:    [1e6 + 0.0, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0, 1e6 + 10.0]
Note:     Asymmetric (skewed right by the 10.0 outlier)
          Naive Sx^3 at offset 1e6: 18 digits lost (f64 has 16). BROKEN.
          Must use centered computation: skew of [0, 1, 2, 3, 10]
TOLERANCE: 1e-6 relative error on skewness
```

**TC-KURT-CANCEL: Kurtosis at moderate offset**
```
input:    [1e4 + 0.0, 1e4 + 1.0, 1e4 + 2.0, 1e4 + 3.0, 1e4 + 100.0]
Note:     Kurtosis via naive Sx^4: 16 digits lost at 1e4. BROKEN at f64.
          Offset of 10,000 is absurdly common in real data.
          Must use centered computation: kurtosis of [0, 1, 2, 3, 100]
TOLERANCE: 1e-4 relative error on excess kurtosis
```

### Category E: Special Values (NaN, Inf, Denormals)

**TC-ALL-NAN: All NaN input**
```
input:    [NaN, NaN, NaN, NaN, NaN]
n:        0 (effective, if NaN-skipping) or 5 (if NaN-propagating)
Decision: Document which convention tambear uses.
          If NaN-skipping: n_valid=0, all stats=NaN
          If NaN-propagating: mean=NaN, var=NaN, etc.
MUST NOT: panic
```

**TC-SOME-NAN: Mixed NaN**
```
input:    [1.0, NaN, 3.0, NaN, 5.0]
If NaN-skipping: mean=3.0 (of [1,3,5]), n_valid=3
If NaN-propagating: mean=NaN
```

**TC-INF: Infinity in input**
```
input:    [1.0, 2.0, Inf, 4.0]
mean:     Inf
var:      NaN (Inf - Inf)
skew:     NaN
MUST NOT: panic
```

**TC-MIXED-INF: +Inf and -Inf**
```
input:    [Inf, -Inf, 1.0]
mean:     NaN (Inf + (-Inf))
All stats: NaN
```

**TC-DENORMAL: Subnormal values**
```
input:    [5e-324, 5e-324, 1e-323]
mean:     ~7e-324
var_pop:  ~0 (values too small to differentiate in squared domain)
Note:     v*v underflows to 0 for subnormals. var_pop = 0.0 is acceptable.
```

**TC-NEAR-OVERFLOW: Values near f64 max**
```
input:    [1e307, 1e307+1e290, 1e307-1e290]
mean:     1e307
Note:     x*x overflows for x > ~1.34e154. Sx^2 will be Inf.
          Centered: values are [-1e290, 0, 1e290]. Centered squares are ~1e580: OVERFLOW.
          Implementation must handle: use relative centering, or detect overflow domain.
TOLERANCE: This is a known hard case. Document behavior, don't pretend it's solved.
```

### Category F: Adversarial Distributions

**TC08: Extreme outlier (Dirac spike)**
```
input:    [0.0]*99 + [1e10]
n:        100
mean:     1e8
var_pop:  9.9e17
skew_pop: ~9.85 (extremely right-skewed)
kurt_exc: ~95.01 (extremely leptokurtic)
Note:     Tests dynamic range. Mean is 1e8, outlier is 1e10.
          Centered computation at mean: values are [-1e8]*99 + [9.9e9].
```

**TC-CAUCHY: Cauchy-like (heavy tails)**
```
input:    [tan((i/(n+1) - 0.5) * pi) for i in 1..n, n=101]
Note:     Cauchy has no finite variance. Empirical variance grows with n.
          The TEST is that the implementation handles this gracefully --
          no overflow, no NaN, just a large finite number.
```

**TC-DIRAC: Dirac delta (single distinct value repeated)**
```
input:    [3.14159]*1000
Same as TC02 but larger n. var=0, skew=NaN, kurt=NaN.
```

**TC-BIMODAL-EXTREME: Extreme bimodal**
```
input:    [-1e10]*50 + [1e10]*50
mean:     0.0
var_pop:  1e20
Note:     Large magnitude but mean is near zero. Naive formula: Sx^2 = 1e22, (Sx)^2/n = 0.
          Actually works for naive formula! The offset cancellation is symmetric.
          This is a case where naive is accidentally fine.
```

### Category G: Order Dependence

**TC-ORDER: Same values, different order**
```
input_a:  [1.0, 2.0, 3.0, 4.0, 5.0]
input_b:  [5.0, 3.0, 1.0, 4.0, 2.0]
input_c:  [5.0, 4.0, 3.0, 2.0, 1.0]
ALL must produce identical mean, var, skew, kurt, min, max.
Tests that the implementation is commutative (no order dependence).
```

**TC-ORDER-LARGE: Order dependence at scale**
```
input_a:  [1e8+i for i in range(1000)]
input_b:  same values, reversed
input_c:  same values, random shuffle
All must produce identical results (within 1 ULP for f64).
Tests GPU atomicAdd determinism on CPU backend.
```

### Category H: Minimum n Requirements Per Statistic

| Statistic | Min n | Behavior at n < min |
|-----------|-------|---------------------|
| count | 0 | Returns 0 |
| sum | 0 | Returns 0.0 |
| mean | 1 | n=0 -> NaN |
| var (pop) | 1 | n=0 -> NaN; n=1 -> 0.0 |
| var (sample) | 2 | n<2 -> NaN |
| std (pop) | 1 | n=0 -> NaN; n=1 -> 0.0 |
| std (sample) | 2 | n<2 -> NaN |
| skew (pop) | 3 | n<3 -> NaN (also NaN when std=0) |
| skew (adj) | 3 | n<3 -> NaN (denominator has n-2) |
| kurt (pop) | 4 | n<4 -> NaN (also NaN when std=0) |
| kurt (adj) | 4 | n<4 -> NaN (denominator has (n-2)(n-3)) |
| min | 1 | n=0 -> +Inf or NaN |
| max | 1 | n=0 -> -Inf or NaN |
| median | 1 | n=0 -> NaN |

---

## Non-Polynomial Statistics (Cannot Share Polynomial MSR)

The following F06 algorithms are NOT polynomial in the data and CANNOT be extracted from power sums {n, Sx, Sx^2, ...}:

1. **Median / Quantiles**: Need T-Digest or sorting. Proof: [1,2,3] and [1,2.5,2.5] have same {n,Sx} but different medians.
2. **Geometric mean**: Needs log transform. MSR: {n, S(log x)}.
3. **Harmonic mean**: Needs reciprocal. MSR: {n, S(1/x)}.
4. **Mode**: Needs frequency counting.
5. **Gini coefficient**: Needs sorted order.
6. **Rank-based** (Spearman, MAD, IQR): Need sorting → gather(sort_indices).
7. **Entropy**: Needs p*log(p) accumulation.
8. **Trimmed/Winsorized**: Two-pass (quantiles first, then accumulate within bounds).

These require separate primitives or gather-based preprocessing. They don't share the polynomial MSR accumulator.

---

## Implementation Requirements Summary

1. **ALL moment computation MUST use centering** (RefCentered approach). Naive power sums fail at offset 1e4 for kurtosis, 1e6 for skewness, 1e8 for variance. These are everyday data magnitudes.

2. **The centering reference should be the per-group mean** (two-pass) or an estimate (running mean from first pass of streaming).

3. **Every statistic must define behavior for insufficient n**. Return NaN, not panic. Document the minimum n.

4. **NaN handling convention must be chosen and documented**: NaN-skip (like R's na.rm=TRUE) or NaN-propagate (like numpy default). Recommendation: NaN-skip with a valid_count, since financial data routinely has missing values.

5. **Near-overflow data requires special handling**. Centering doesn't help when centered^2 still overflows. For |x| > 1e154, x^2 overflows f64. Consider: detecting this regime and using log-space or scaling.
