# Finding 2: Cholesky Ill-Conditioning — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN numerically
**Code**: `crates/tambear/src/train/cholesky.rs:21`

---

## The Bug

`cholesky()` checks `if s <= 0.0 { return None; }` to detect non-positive-definite matrices. This catches truly broken matrices but lets ill-conditioned ones through silently. The returned solution can have arbitrarily large error with NO WARNING.

**Used by**: `linear.rs` normal equations solver (the ONLY consumer of Cholesky in this codebase).

---

## Proof Table 1: Hilbert Matrix Error Growth

| d | cond(A) | cholesky | max error | rel error | Status |
|---|---------|----------|-----------|-----------|--------|
| 6 | 1.3e9 | OK | 1.2e-11 | 5.1e-12 | OK |
| 7 | 4.4e10 | OK | 2.7e-9 | 1.0e-9 | MARGINAL |
| 9 | 4.8e13 | OK | 1.4e-6 | 4.6e-7 | MARGINAL |
| 10 | 1.6e15 | OK | **5.7e-4** | 1.8e-4 | **BAD** |
| 12 | 1.7e18 | OK | **7.5e-2** | 2.2e-2 | **BROKEN** |
| 13 | 5.8e19 | OK | **3.28** | 0.91 | **BROKEN** |
| 14 | N/A | FAILED | N/A | N/A | REJECTED |

At d=13, Cholesky returns `Some(solution)` where the solution has **328% error**.

---

## Proof Table 2: Collinear Regression (the actual use case)

X = [[i, i+eps] for i in 1..10], beta_true = [1, 1]:

| epsilon | cond(X'X) | beta_0 | beta_1 | error | Status |
|---------|-----------|--------|--------|-------|--------|
| 1e-2 | 7.2e6 | 1.000 | 1.000 | 1.1e-9 | OK |
| 1e-4 | 7.2e10 | 1.000 | 1.000 | 2.7e-6 | MARGINAL |
| **1e-5** | **7.2e12** | **1.000** | **1.000** | **2.7e-4** | **BAD** |
| **1e-6** | **7.1e14** | **1.056** | **0.944** | **5.6e-2** | **BAD** |
| 1e-7 | inf | FAIL | FAIL | N/A | REJECTED |

Predictor correlation of r=0.999999 (eps=1e-6) gives **5.6% coefficient error** — a silent wrong answer in a regression that appears to succeed.

---

## Test Vectors

### TV-CHOL-01: Well-conditioned 2x2 (must work)
```
A = [[4, 2], [2, 3]], b = [8, 8]
Expected: x = [1, 2]
```

### TV-CHOL-02: Hilbert(5) — moderate ill-conditioning
```
A = hilbert(5), b = A @ [1,1,1,1,1]
Expected: x = [1,1,1,1,1] (within 1e-10)
```

### TV-CHOL-03: Hilbert(10) — severe ill-conditioning
```
A = hilbert(10), b = A @ [1,...,1]
Expected: x deviates by ~5e-4 from [1,...,1]
MUST: return Some (not None) — bug is that it SUCCEEDS with wrong answer
```

### TV-CHOL-04: Hilbert(13) — catastrophic
```
A = hilbert(13), b = A @ [1,...,1]
Expected: x deviates by ~328% from truth
MUST: either return None, or return solution + condition_estimate
```

### TV-CHOL-05: Not positive definite
```
A = [[1, 2], [2, 1]], b = [1, 1]
Expected: None (eigenvalues are 3 and -1)
```

### TV-CHOL-06: Collinear regression
```
X = [[i, i+1e-6] for i in 1..11]
y = X @ [1, 1]
A = X'X, b = X'y
Expected: beta deviates by ~5.6% from [1, 1]
MUST: either detect ill-conditioning or document the risk
```

### TV-CHOL-07: Zero matrix
```
A = [[0, 0], [0, 0]], b = [1, 1]
Expected: None (not positive definite)
```

### TV-CHOL-08: 1x1
```
A = [[4.0]], b = [8.0]
Expected: x = [2.0]
```

---

## Recommended Fixes

1. **Condition number estimate**: After computing L, estimate cond(A) as `(max L[i,i] / min L[i,i])^2`. This is O(d) and nearly free.

2. **Return quality flag**: Change return type from `Option<Vec<f64>>` to `Result<(Vec<f64>, f64), Error>` where the f64 is the condition estimate.

3. **Log warning at threshold**: If estimated condition > 1e12, log a warning. The user (linear.rs) can decide what to do.

4. **Alternative solver**: For regression with potential collinearity, QR decomposition is more numerically stable. SVD is gold standard for rank-deficient problems.
