# Recipe Numerical Stability Map

*Compiled by scout, 2026-04-11. Verified with Python for concrete failure cases.*

This document maps every recipe in the current catalog to its numerical stability
properties. "Stability" here means: does the formula lose precision when inputs
have a large mean relative to variance? This is distinct from reproducibility
(which RFA solves) — a formula can be stable but non-reproducible, or reproducible
but unstable.

---

## Stability classification

### UNSTABLE — catastrophic cancellation for financial data

These recipes compute differences of large-magnitude sums. When the mean is large
relative to the standard deviation, the difference loses all significant bits.

**Trigger condition:** `mean^2 / variance >> 2^52` (i.e., `mean / std_dev >> 67 million`)

| Recipe | Formula | Where it fails |
|---|---|---|
| `variance` | `(sum_sq - sum^2/n) / (n-1)` | Any large-offset data; returns 0.0 for [1e9..1e9+9] |
| `variance_biased` | `(sum_sq - sum^2/n) / n` | Same |
| `std_dev` | `sqrt(variance)` | Inherits from variance |
| `skewness` | `(m3 - 3*m1*m2 + 2*m1^3) / sigma^3` | Raw moment centering around large mean |
| `kurtosis_excess` | `(m4 - 4*m1*m3 + ...) / sigma^4` | Same; extremely sensitive |
| `covariance` | `(sum_xy - sum_x*sum_y/n) / (n-1)` | Same pattern as variance |
| `pearson_r` | num/sqrt(den_x * den_y) | Den goes to 0.0 for x_base=1e10, n=1000 |

**Verified failure cases:**

```python
# variance: returns 0.0 for data with large mean
data = [1e9 + x for x in range(10)]
# One-pass returns: 0.0  (true: 9.167)

# pearson_r: denominator goes to ZERO (NaN result)
x = [1e10 + i for i in range(1000)]
y = [2e10 + i*0.5 + (i%3)*0.01 for i in range(1000)]
# One-pass den_y_sq = 0.0  (true: ~20,833,314)
# Result: NaN
```

---

### RISKY — precision loss at scale, but not total catastrophic failure

| Recipe | Risk | Mitigation |
|---|---|---|
| `mean_arithmetic` | `sum / n` inherits accumulation error at large n | Compensated summation (Kahan) for large N |
| `mean_quadratic` | `sqrt(sum_sq / n)` — sum_sq can be very large | Less risky than variance (no subtraction) |
| `l2_norm` | `sqrt(sum_sq)` — large elements dominate | Same as mean_quadratic |
| `sum_squared_diff` | `(x-y)^2` — cancellation if x approx y | Use `(x-y)` directly, not `x^2 - 2xy + y^2` |
| `rmse` | Inherits from sum_squared_diff | Same |

Note: `sum_squared_diff` in the current recipe is implemented as `(val - val2)^2` via
`Expr::val().sub(Expr::val2()).sq()` — this is correct! It computes the difference first,
then squares. This avoids the `x^2 - 2xy + y^2` cancellation trap. The current implementation
is actually fine.

---

### STABLE — no catastrophic cancellation risk

These recipes compute single accumulators or use difference-first patterns:

`count`, `sum`, `product`, `mean_geometric` (log space), `mean_harmonic` (reciprocal space),
`l1_norm`, `linf_norm`, `min_all`, `max_all`, `range_all`, `midrange`, `dot_product`, `mae`

---

## The two problems and two fixes

### Problem 1: Numerical instability (the cancellation problem)
**Fix: Algorithm change**
- `variance`, `std_dev`, `variance_biased`: replace with Welford's algorithm
- `skewness`, `kurtosis_excess`: use higher-order Welford (West, 1979)
- `covariance`, `pearson_r`: use the two-pass or running-mean-centered formula

### Problem 2: Non-reproducibility (the atomicAdd ordering problem)
**Fix: RFA (Peak 6)**
- Any recipe with a parallel `Add` reduction
- Even stable formulas (like `sum`) are non-reproducible across GPU architectures

### The compound case: variance needs BOTH fixes
Welford's algorithm (I1 fix) is still order-dependent. A GPU parallel Welford
needs the parallel Welford merge formula (Chan et al., 1979). Even with that, the
merge order affects the result unless RFA is applied to the delta accumulations.

For Phase 1: accept run-to-run determinism (fixed-launch-config) for variance,
and document that gpu_to_gpu bit-exact variance requires Phase 2 (Welford + RFA delta).

---

## Stable numerics references for the Math Researcher

- **Welford (1962)**: "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3).
- **Chan, Golub, LeVeque (1979)**: "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances." Technical Report STAN-CS-79-773, Stanford.
- **Higham (2002)**: "Accuracy and Stability of Numerical Algorithms" — Chapter 4 for summation, Chapter 3 for condition numbers.
- **Pébay (2008)**: "Formulas for Robust, One-Pass Parallel Computation of Covariances and Arbitrary-Order Statistical Moments." Sandia Report SAND2008-6212.

Pébay (2008) is especially relevant — it extends Welford to arbitrary-order moments (skewness, kurtosis) and to the parallel merge case. It covers exactly the `skewness` and `kurtosis_excess` recipes.
