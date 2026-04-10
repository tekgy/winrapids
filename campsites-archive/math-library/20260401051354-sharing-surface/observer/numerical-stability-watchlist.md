# Numerical Stability Watchlist

**Maintained by**: Observer
**Last updated**: 2026-04-01

This is the cross-cutting reference for numerical stability patterns across all algorithm families. Every implementer should check this before writing any moment/accumulation code.

---

## CRITICAL: The Centering Rule

**Any moment computation of degree > 1 MUST center data before accumulation.**

The naive formula `var = E[x²] - E[x]²` loses precision catastrophically when data is offset from zero. This is not theoretical — the adversarial mathematician proved:

- **Kurtosis breaks at offset 1e4 in f64** — that's nearly all real-world data
- **Variance breaks at offset 1e8 in f64** — common for financial prices ($100+ range)
- **In f32, variance breaks at offset ~1e3** — essentially all real data

**Test**: TC-CANCEL-GRAD from adversarial suite. For each offset in [1e4, 1e6, 1e8, 1e10, 1e12, 1e14], compute variance of `[offset + i*0.001 for i in range(100)]`. Expected var_pop = 8.3325e-04 (stable). If relative error exceeds 1% at ANY offset, the implementation is broken.

---

## Pattern Catalog

### 1. Catastrophic Cancellation in Variance

**Problem**: `E[X²] - (E[X])²` subtracts two nearly-equal large numbers.
**When it hits**: Data offset from zero (offset > ~√(1/ε) × std)
**Solution**: Either:
- **Welford online**: Running mean + M2, update per element. O(1) state, one pass, numerically stable.
- **RefCentered**: Subtract reference point (first element or running mean) before accumulating. Better for GPU (parallel-friendly).
- **Two-pass**: First pass computes mean, second pass accumulates (x - mean)². Exact but requires two passes.

**Recommendation for tambear**: RefCentered for GPU (compatible with parallel scatter), Welford for streaming CPU.

### 2. Catastrophic Cancellation in Higher Moments

**Problem**: Same issue as variance but worse — skewness uses Σ(x-μ)³, kurtosis uses Σ(x-μ)⁴.
**Scaling**: Kurtosis loses 4× as many digits as variance for the same offset.
**Solution**: MUST center before accumulating. No shortcut.
**Test**: TC-SKEW-CANCEL and TC-KURT-CANCEL from adversarial suite.

### 3. Log-Sum-Exp for Softmax

**Problem**: `exp(x)` overflows for x > ~709 (f64) or x > ~88 (f32).
**Solution**: Subtract max before exponentiating: `softmax(x) = exp(x - max(x)) / Σexp(x - max(x))`.
**Status**: Already implemented correctly in `accumulate.rs::softmax()`. ✓ Verified against scipy.
**Test**: Large values [1000, 1001, 1002] — passes.

### 4. Kahan Compensated Summation

**Problem**: Naive sum of n floating-point numbers has error O(n·ε). For n=1M, f64: error ~2e-10. For f32: error ~120.
**Solution**: Kahan summation: maintain a compensation variable that captures the lost low-order bits.
**Status in codebase**:
- `format.rs` — **USES Kahan** for tile statistics ✓
- `train/linear.rs::column_stats()` — **DOES NOT use Kahan** ✗ (inconsistency)
- `accumulate.rs` — **DOES NOT use Kahan** ✗
**Recommendation**: For any accumulation where n > 10K, Kahan is worth the 1 extra register. For GPU scatter kernels, Kahan may conflict with atomics — investigate.

### 5. Cholesky Without Pivoting

**Problem**: Standard Cholesky decomposition fails silently on ill-conditioned matrices. It succeeds (returns Some), but the answer has lost most of its precision.
**When it hits**: X'X condition number > ~1e8 means losing 8+ digits. κ > 1e15 means all precision lost.
**Current status**: `cholesky.rs` has no pivoting and no condition number estimate.
**Impact**: Any regression with near-collinear features silently produces wrong coefficients.
**Solution**: Either (a) add pivoted Cholesky with condition number estimate, (b) use SVD-based solver for the d×d system, or (c) document that user must z-score before regression (fit_session already does this).
**Test**: Near-collinear adversarial test in oracle (κ=3.7e15, sklearn solves correctly, tambear expected to fail).

### 6. Division by Zero in Statistics

**Problem**: Sample variance divides by (n-1), which is 0 when n=1.
**Instances**:
- Sample variance: n=1 → NaN or Inf
- Sample skewness: n≤2 → undefined
- Sample kurtosis: n≤3 → undefined
- CV: mean=0 → undefined
**Solution**: Each statistic declares minimum_n in composability contract. Return NaN (not error) for insufficient n.

### 7. f32 vs f64 Precision Cliff

**Problem**: f32 has ε ≈ 1.2e-7 (7 decimal digits). Many statistical formulas lose precision badly in f32.

| Statistic | f64 safe offset | f32 safe offset | f32 usable? |
|-----------|----------------|----------------|:-----------:|
| Mean | any | any | ✓ |
| Variance (naive) | < 1e8 | < 1e3 | ✗ |
| Variance (centered) | any | < 1e3 with recentering | ~ |
| Skewness | < 1e5 | not recommended | ✗ |
| Kurtosis | < 1e4 | not recommended | ✗ |
| Softmax | any (with log-sum-exp) | any | ✓ |
| Dot product | any | cumulative error O(k·ε) | ~ |
| L2 distance | any | cumulative error O(d·ε) | ~ |

**Recommendation**: For WGSL/f32 path, only offer mean, sum, min, max, count, softmax. Don't offer variance/skewness/kurtosis in f32 — the results are misleading. Document this as a precision gate, not a missing feature.

### 8. Overflow in Power Sums

**Problem**: Σx⁴ overflows f64 when |x| > ~1e77 (since 1e77⁴ = 1e308 ≈ f64 max).
**When it hits**: Data with extreme values (adversarial, not typical).
**Solution**: Log-space computation for extreme values, or centering + scaling first.
**Test**: TC-NEAR-OVERFLOW from adversarial suite. Even scipy fails on 1e300 values.
**Practical stance**: Document as known limitation. Real financial data doesn't have 1e77 values.

### 9. Accumulation Order Dependence

**Problem**: Floating-point addition is not associative. `(a+b)+c ≠ a+(b+c)`.
**Impact**: GPU parallel reduction may give different results depending on thread scheduling.
**Guarantee we can offer**: Results within O(log₂(n)·ε) of the mathematically exact answer.
**Test**: TC-ORDER from adversarial suite — same data, different order, verify results within tolerance.
**Note**: Bit-exact reproducibility across runs requires deterministic reduction order. This is expensive on GPU. Consider whether to offer it as an option.

---

## Decision Matrix: Which Numerical Method When?

| Situation | Method | Why |
|-----------|--------|-----|
| GPU parallel accumulation, moments | RefCentered | Parallel-friendly, one extra subtract per element |
| CPU streaming, moments | Welford | One-pass, O(1) state, no second reference point needed |
| High-precision sum (n > 10K) | Kahan | 1 extra register, O(1) error independent of n |
| Log-probability sums | Log-sum-exp | Prevents exp() overflow |
| Small dense solve (d < 100) | Cholesky + z-score | Fast, stable after normalization |
| Ill-conditioned dense solve | SVD | Numerically robust regardless of condition number |
| f32 statistics | Mean/Sum/Min/Max only | Higher moments lose too much precision |

---

## Watchlist Items (open investigations)

- [ ] **Kahan vs atomics on GPU**: Can we use Kahan summation inside atomic scatter kernels? The compensation variable needs a separate atomic, which doubles the atomic traffic. Investigate whether the precision gain is worth the performance cost.
- [ ] **RefCentered reference point selection**: First element? Running mean? Median of sample? The choice affects precision. Need empirical comparison.
- [ ] **f32 moment precision boundary**: Exact offset at which f32 variance relative error exceeds 1%. The adversarial suite uses f64; we need f32-specific thresholds.
- [ ] **Deterministic reduction on GPU**: Cost of forcing deterministic reduction order via sorted key-value pairs. Is 2× slowdown acceptable for reproducibility?
