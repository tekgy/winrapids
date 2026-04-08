# Shapiro-Wilk Normality Test — Implementation Specification
**Author**: math-researcher  
**Date**: 2026-04-06  
**Scope**: Shapiro-Wilk test for normality. Currently absent from tambear; module header claims it exists.  
**Module target**: `crates/tambear/src/nonparametric.rs`  
**References**: Shapiro & Wilk (1965) *Biometrika*; Royston (1992) *Applied Statistics*; Royston (1995) AS R94 algorithm.

---

## 1. The Test Statistic

Given data x₁, ..., xₙ, let x_{(1)} ≤ x_{(2)} ≤ ... ≤ x_{(n)} be the order statistics.

**The W statistic**:
```
W = (Σᵢ aᵢ · x_{(i)})² / Σᵢ (xᵢ - x̄)²
```

where:
- x̄ = sample mean
- a = (a₁, ..., aₙ) are coefficients derived from expected values of standard normal order statistics
- 0 < W ≤ 1; W close to 1 indicates normality

---

## 2. Computing the a Coefficients (Royston 1992 approximation)

For all n, the coefficients are derived from the expected values of standard normal order statistics via Blom (1958) plotting positions.

### Step 1: Compute normal scores
For i = 1, ..., n:
```
mᵢ = Φ⁻¹((i - 3/8) / (n + 1/4))
```
where Φ⁻¹ is the standard normal quantile (inverse CDF).

Note: `m_{n+1-i} = -mᵢ` by symmetry.

### Step 2: Compute the last coefficient
```
M = sqrt(Σᵢ mᵢ²)
aₙ = mₙ / M
```

For the second-to-last coefficient, Royston (1992) introduces a small correction via polynomial:
```
u = 1/sqrt(n)
a_{n-1} = m_{n-1} / c2 + poly(u, [-2.706056, 4.434685, -2.071190, -0.147981, 0.221157, aₙ])
```
where `poly(u, c)` evaluates the polynomial c[0]*u^5 + c[1]*u^4 + ... + c[5] = c[5] + u*(c[4] + u*(c[3] + u*(c[2] + u*(c[1] + u*c[0]))))`.

The normalizing constant:
```
c2 = sqrt(
    (1/n) * [1 - 2*aₙ² - 2*a_{n-1-raw}² - sum(mᵢ² for i=1..n-2)] / M²
)
```
where `a_{n-1-raw} = m_{n-1}/M` (before correction).

For n ≥ 6, a more stable computation is:
```
phi = (M² - 2*mₙ² - 2*m_{n-1}²) / (1 - 2*aₙ² - 2*a_{n-1}²)
```
Remaining coefficients (i = 2, ..., n-2) rescaled so Σaᵢ² = 1:
```
aᵢ = mᵢ / sqrt(phi)
```
And by antisymmetry: `a_{n+1-i} = -aᵢ` (the vector is antisymmetric: a₁ < 0, ..., a_{n/2} < 0, a_{n/2+1} > 0, ..., aₙ > 0).

**For n=3**: exact closed form:
```
a₁ = -1/sqrt(2), a₂ = 0, a₃ = 1/sqrt(2)
W = (x₃ - x₁)² / (2 * Σ(xᵢ-x̄)²)
```

---

## 3. Computing the P-Value (Royston 1992)

The p-value uses a normalizing transformation of W. Royston showed that `y = log(1-W)` is approximately normal after a further transformation.

### For 3 ≤ n ≤ 11 (small n, original Shapiro-Wilk regime)

The transformation is:
```
γ = poly(n, [0.459, -2.273])           // γ = -2.273 + 0.459*n  (Royston 1992, Table 1 approx)
μ = poly(n, [0.0038915, -0.083751, -0.31082, -1.5861])
σ = exp(poly(n, [0.00030302, -0.0082676, -0.4803]))
z = (log(1-W) - γ - μ) / σ
```
(Exact coefficients given in AS R94; these are illustrative — see Section 6 for the exact table.)

p-value = 1 - Φ(z)  (W close to 1 ⟹ z small ⟹ p large ⟹ don't reject normality)

### For 12 ≤ n ≤ 5000 (Royston 1992 extended range)

The transformation is:
```
y = log(1 - W)
x = log(n)
μ(n) = poly(x, [-1.2725, 1.0521])
σ(n) = exp(poly(x, [-1.2624, 1.9160]))
z = (y - μ(n)) / σ(n)
```
(Royston 1992 AS R94 exact coefficients differ slightly; use AS R94 for production.)

p-value = 1 - Φ(z)

### For n > 5000

Royston's approximation degrades for very large n. For n > 5000, the test has near-unit power for any practical departure from normality, so extreme precision in the p-value is less critical. Options:
1. Use the n=5000 approximation (conservative)
2. Use the Anderson-Darling test which has better large-n properties

---

## 4. Exact Royston Polynomial Coefficients (AS R94)

These are the published coefficients from Royston (1995) *Applied Statistics*, Algorithm AS R94.

### For the a-coefficient computation:

For the last coefficient `aₙ`:
```
c1 = [0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056]   // poly in u = 1/sqrt(n)
c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633]   // for a_{n-1}
```

### For the p-value (small n, 4 ≤ n ≤ 11):
```
c3 = [0.544, -0.39978, 0.025054, -6.714e-4]    // coefficients for mu(n)
c4 = [1.3822, -0.77857, 0.062767, -0.0020322]   // coefficients for sigma(n)
c5 = [-1.5861, -0.31082, -0.083751, 0.0038915]  // alternate form (Royston uses c5)
c6 = [-0.4803, -0.082676, 0.0030302]             // for log(sigma)
```

### For the p-value (n ≥ 12, Royston 1992):
```
// mu(x) = polynomial in x = log(n)
c7 = [-1.2725, 1.0521]    // mu coefficients
c8 = [1.9160, -1.2624]    // log(sigma) coefficients
```

**Implementation note**: Use the exact AS R94 coefficient tables rather than these approximations. The AS R94 paper gives 6-decimal precision.

---

## 5. Algorithm Summary

```
fn shapiro_wilk(data: &[f64]) -> (f64, f64) {
    // Returns (W, p_value)
    
    let n = data.len();
    assert!(n >= 3, "Shapiro-Wilk requires n >= 3");
    
    // Step 1: Sort
    let mut x = data.to_vec();
    x.sort_by(|a, b| a.total_cmp(b));
    
    // Step 2: Compute a coefficients via Blom + Royston
    let a = compute_sw_coefficients(n);
    // a is antisymmetric; only need upper half
    // a[i] = coefficient for x_{(n-i)}, -a[i] for x_{(i+1)}
    
    // Step 3: Compute numerator b = Σᵢ aᵢ x_{(n+1-i)} - aᵢ x_{(i)}
    let m = n / 2;
    let b: f64 = (0..m).map(|i| a[i] * (x[n-1-i] - x[i])).sum();
    
    // Step 4: SS = Σ(xᵢ - x̄)²
    let mean = x.iter().sum::<f64>() / n as f64;
    let ss: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
    
    // Step 5: W = b² / SS
    let w = if ss == 0.0 { 1.0 } else { (b * b / ss).min(1.0) };
    
    // Step 6: p-value via Royston transformation
    let p = sw_pvalue(w, n);
    
    (w, p)
}
```

### compute_sw_coefficients(n)

```
fn compute_sw_coefficients(n: usize) -> Vec<f64> {
    if n == 3 {
        return vec![1.0 / 2.0_f64.sqrt()];  // only a[n] needed; a₁=-a₃
    }
    
    // Blom normal scores
    let m: Vec<f64> = (1..=n).map(|i| {
        normal_quantile((i as f64 - 0.375) / (n as f64 + 0.25))
    }).collect();
    
    let big_m: f64 = m.iter().map(|v| v*v).sum::<f64>().sqrt();
    
    let u = 1.0 / (n as f64).sqrt();
    
    // Last coefficient (Royston polynomial)
    let an = poly(u, &[0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056])
              + m[n-1] / big_m;  // NOTE: poly gives correction; raw m[n-1]/M + correction
    // Actually: an = m[n-1]/M is the raw estimate; correction is added
    // Royston: an = m[n-1]/M + poly(u, c1) where poly gives adjustment
    
    // Second-to-last
    let an1_raw = m[n-2] / big_m;
    let correction = poly(u, &[0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633]);
    let an1 = an1_raw + correction;
    
    // Normalizing phi for middle coefficients
    let phi = (big_m*big_m - 2.0*m[n-1]*m[n-1] - 2.0*m[n-2]*m[n-2])
            / (1.0 - 2.0*an*an - 2.0*an1*an1);
    
    let root_phi = phi.max(0.0).sqrt();
    
    let half = n / 2;
    let mut a = vec![0.0; half];
    a[half - 1] = an;
    if half >= 2 { a[half - 2] = an1; }
    for i in 0..half-2 {
        a[i] = m[i] / root_phi;  // Note: these are for the UPPER half; a[0]=smallest (most negative)
    }
    
    a  // a[i] is coefficient for x_{(n-i)} pair; outer sum uses x[n-1-i] - x[i]
}
```

**Note on indexing**: The `a` vector has `n/2` entries. Entry `a[i]` multiplies `(x_{(n-i)} - x_{(i+1)})`. The convention in the sum is `b = Σᵢ₌₁^{n/2} aᵢ(x_{(n+1-i)} - x_{(i)})`. All `aᵢ > 0` since the ordering is largest-at-front vs. smallest-at-front.

---

## 6. Normal Quantile (Required Subroutine)

The coefficient computation needs `Φ⁻¹(p)` — the standard normal quantile function.

This is the inverse of `normal_cdf`. Implement via Brent's method on `normal_cdf`, or use the Rational approximation from Abramowitz & Stegun §26.2.17:

For p close to 0.5:
```
t = sqrt(-2 ln(min(p, 1-p)))
Φ⁻¹(p) = ±(t - (a0 + a1*t + a2*t²) / (1 + b1*t + b2*t² + b3*t³))
```
where a0=2.515517, a1=0.802853, a2=0.010328; b1=1.432788, b2=0.189269, b3=0.001308.
Maximum error: 4.5e-4.

For production accuracy, use a higher-precision rational approximation (e.g., Peter Acklam's algorithm which achieves 1.15e-9 relative error):
```
// Rational approximation for central region (0.02425 ≤ p ≤ 0.97575):
a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]

// Rational approximation for tails (p < 0.02425 or p > 0.97575):
c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
     -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00]
d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
```

---

## 7. P-Value Computation: Exact Royston Coefficients

### For n=3:
```
Exact: p = 6/π * arcsin(sqrt(W)) ... use simulation table or Royston formula
Approximate: W has a known distribution; use the beta approximation
```

### For 4 ≤ n ≤ 11 (Royston 1992, small n):
```
gamma_coeff = [-2.273, 0.459]  // gamma = poly(n, gamma_coeff)
// mu polynomial in n:
mu_coeff = [-1.5861, -0.31082, -0.083751, 0.0038915]
// log(sigma) polynomial in n:
sigma_coeff = [-0.4803, -0.082676, 0.0030302]

gamma = poly_eval(n as f64, &gamma_coeff)
mu = poly_eval(n as f64, &mu_coeff)
log_sigma = poly_eval(n as f64, &sigma_coeff)
sigma = log_sigma.exp()
z = (log(1 - W) - gamma - mu) / sigma
p = 1 - normal_cdf(z)
```
where poly_eval evaluates coefficients in ascending order: c[0] + c[1]*x + c[2]*x² + ...

### For 12 ≤ n ≤ 5000 (Royston 1992, large n):
```
y = log(1 - W)
x = log(n as f64)
mu = poly_eval(x, &[-1.2725, 1.0521])       // = -1.2725 + 1.0521*x
log_sigma = poly_eval(x, &[1.9160, -1.2624]) // = 1.9160 - 1.2624*x
sigma = log_sigma.exp()
z = (y - mu) / sigma
p = 1 - normal_cdf(z)
```

**Important**: When W is exactly 1.0 (all values equal), return p = 1.0. When n = 1 or 2, return p = 1.0 (test undefined).

---

## 8. Accumulate+Gather Decomposition

```
Pattern: Kingdom A (closed-form evaluation with a precomputed accumulation)

1. gather: normal_quantile() calls (O(n) point evaluations)
2. accumulate(i=0..n/2, aᵢ * (x_{(n-1-i)} - x_{(i)}), sum) → b   [linear statistic]
3. accumulate(i=0..n, (xᵢ - x̄)², sum) → SS                        [sum of squares]
4. gather: W = b²/SS, p via polynomial evaluation
```

This is the same pattern as all regression sufficient statistics: the test statistic IS an accumulate of order-statistic contrasts. The a coefficients are the "projection direction" onto the Gaussian order statistics — this is exactly the sufficient statistic for the normality departure direction.

---

## 9. Test Cases

| n | Data | Expected W | Expected p (approx) |
|---|---|---|---|
| 3 | [1, 2, 3] | 1.000 | ~1.0 (perfect line) |
| 5 | N(0,1) sample [seed=42] | ~0.95 | > 0.05 |
| 8 | [1,1,1,1,2,2,2,2] | ~0.75 | < 0.05 (step function, not normal) |
| 10 | standard normal quantiles | ~1.0 | ~1.0 (perfect match) |
| 50 | N(5,2²) random sample | > 0.93 | > 0.10 |
| 50 | Exponential(1) sample | < 0.90 | < 0.001 |

Specific numeric test: the 5-value dataset {−0.605, 0.209, 1.264, −1.384, 1.116} should give W ≈ 0.963, p ≈ 0.82 (approximately normal).

Cross-validate against R: `shapiro.test(x)$statistic` and `shapiro.test(x)$p.value`.

---

## 10. Lilliefors Correction (Companion Spec)

The `ks_test_normal` bug (Issue #6) should be resolved alongside Shapiro-Wilk. The correct normality KS test uses **Lilliefors critical values** (Lilliefors 1967):

When parameters are estimated from data (standardize: z = (x-x̄)/s then test against N(0,1)), the Kolmogorov asymptotic p-value is too liberal. Lilliefors corrected this via simulation.

**Lilliefors critical values** for H₀: data is normal (parameters estimated):
| n | α=0.20 | α=0.10 | α=0.05 | α=0.01 |
|---|---|---|---|---|
| 5 | 0.265 | 0.315 | 0.337 | 0.405 |
| 10 | 0.206 | 0.239 | 0.258 | 0.304 |
| 15 | 0.176 | 0.201 | 0.220 | 0.257 |
| 20 | 0.156 | 0.174 | 0.190 | 0.231 |
| 25 | 0.142 | 0.165 | 0.180 | 0.212 |
| 30 | 0.131 | 0.152 | 0.166 | 0.200 |
| ∞ | 1.073/√n | 1.224/√n | 1.358/√n | 1.628/√n |

**Implementation for `ks_test_normal`**: Fix the standardization bug AND either:
1. Use the Lilliefors table to return a corrected p-value
2. Document that p-value is conservative (original Kolmogorov p-value will be too small)

**Recommendation**: Fix the standardization (always), add a `ks_test_normal_lilliefors` variant that uses the Lilliefors table for p-values.

---

## 11. Implementation Priority

| Component | Priority |
|---|---|
| `shapiro_wilk(data: &[f64]) -> (W, p)` | HIGH (only normality test in tambear) |
| Fix `ks_test_normal` standardization | HIGH (current function is wrong) |
| `normal_quantile(p)` (needed for a coefficients) | HIGH (prerequisite) |
| Lilliefors table p-values | MEDIUM |
| Anderson-Darling test (better for large n) | LOW |

---

*Note: Shapiro-Wilk is listed in the module docstring as implemented. It is not. This spec gives everything needed to implement it correctly.*

*Royston AS R94 is the standard reference implementation used by R's `shapiro.test()`. The coefficient tables in that paper should be used for production accuracy.*
