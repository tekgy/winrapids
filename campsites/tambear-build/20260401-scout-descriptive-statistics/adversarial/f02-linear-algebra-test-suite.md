# Family 02: Linear Algebra — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN by code review
**Code**: `crates/tambear/src/linear_algebra.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| LU (partial pivoting) | linear_algebra.rs:253-291 | OK |
| Cholesky | linear_algebra.rs:365-386 | OK |
| QR (Householder) | linear_algebra.rs:429-477 | OK |
| QR solve (least squares) | linear_algebra.rs:480-502 | OK |
| SVD | linear_algebra.rs:522-575 | **HIGH** (A^T A squares κ) |
| pinv | linear_algebra.rs:578-606 | **HIGH** (uses buggy SVD) |
| sym_eigen (Jacobi) | linear_algebra.rs:614-696 | **MEDIUM** (absolute convergence) |
| power_iteration | linear_algebra.rs:702-719 | **LOW** (all-ones start vector) |
| cond | linear_algebra.rs:725-731 | **HIGH** (uses buggy SVD) |
| rank | linear_algebra.rs:734-737 | **HIGH** (uses buggy SVD) |
| solve | linear_algebra.rs:742-746 | OK (LU) |
| solve_spd | linear_algebra.rs:749-752 | OK (Cholesky) |
| lstsq | linear_algebra.rs:755-757 | OK (QR) |
| det | linear_algebra.rs:334-341 | OK (LU) |
| inv | linear_algebra.rs:344-357 | OK (LU) |

---

## Finding F02-1: SVD via A^T A Squares Condition Number (HIGH)

**Bug**: `svd()` at line 522 claims "one-sided Jacobi rotations" in the docstring, but the actual implementation (lines 534-538) computes:

```rust
let at = a.t();
let ata = mat_mul(&at, a);
let (eigenvalues, v) = sym_eigen(&ata);
```

This is the textbook "never do this" approach to SVD. Computing A^T A:
1. **Squares the condition number**: κ(A^T A) = κ(A)²
2. **Destroys small singular values**: If σ_min(A) = 1e-8, then λ_min(A^T A) = 1e-16, which is below machine epsilon relative to λ_max
3. **Produces wrong rank, wrong condition number, wrong pseudoinverse** for any moderately ill-conditioned matrix

**Proof**:
```
A has singular values [1.0, 1e-8, 1e-12]
A^T A has eigenvalues [1.0, 1e-16, 1e-24]
1e-16 is at the edge of f64 precision (machine eps ≈ 1.1e-16)
1e-24 rounds to 0 or is computed as negative → sqrt fails

Result: SVD reports rank 1 instead of rank 3
        pinv(A) is wrong for the 2nd and 3rd components
        cond(A) reports ∞ instead of 1e12
```

**The correct approach**: Golub-Kahan bidiagonalization (which the comment claims!) operates directly on A, never forming A^T A. This preserves the condition number: κ(bidiagonal) = κ(A), not κ(A)².

**Impact**: ALL SVD-based operations are affected: `pinv`, `cond`, `rank`, and any downstream code using SVD for ill-conditioned matrices.

**Fix**: Replace the A^T A approach with one of:
1. Golub-Kahan bidiagonalization + QR iteration (standard LAPACK DGESVD)
2. Direct one-sided Jacobi rotations on A (what the docstring claims)
3. QR-based SVD: QR factorize A, then SVD of the triangular R

Option 3 is the simplest: `QR = qr(A); SVD(R)` with the Jacobi method on R (which is square n×n even when m >> n). Since QR is already correctly implemented via Householder, this is a small change.

---

## Finding F02-2: Jacobi Absolute Convergence Threshold (MEDIUM)

**Bug**: `sym_eigen` at line 640 uses `max_val < 1e-14` as convergence criterion. This is an absolute threshold.

**Impact**: For matrices with entries ~1e-10, the convergence requires off-diagonal elements < 1e-14, which is only 1e-4 relative accuracy. For matrices with entries ~1e10, the convergence is achieved at off-diagonal = 1e-5, which is 1e-15 relative accuracy (excellent).

**Fix**: Use `max_val < tol * frobenius_norm(A)` where tol = 1e-14 is the relative tolerance.

---

## Finding F02-3: Power Iteration Start Vector (LOW)

**Bug**: `power_iteration` starts with `v = [1, 1, ..., 1]`. If the dominant eigenvector is orthogonal to this (e.g., an alternating eigenvector for certain structured matrices), the method converges to the second eigenvector or fails to converge.

**Fix**: Use a pseudo-random starting vector, or detect stagnation (eigenvalue not changing) and restart with a different vector.

---

## Finding F02-4: Misleading SVD Docstring (LOW)

The function comment says "Golub-Kahan bidiagonalization + QR iteration" (line 505) and the docstring says "one-sided Jacobi rotations" (line 518). Neither is what's implemented. The actual algorithm is A^T A eigendecomposition.

---

## Positive Findings

**QR decomposition (Householder) is correct.** The Householder reflections implementation at lines 429-477 is the gold standard for QR. Sign choice at line 445 prevents cancellation. Applied from the left to R and from the right to Q.

**LU with partial pivoting is correct.** Finds the maximum pivot element, properly tracks row swaps.

**Cholesky is correct.** Detects non-positive-definite matrices.

**lstsq uses QR (not SVD).** This is correct — QR is numerically stable for least squares. The lstsq function is NOT affected by the SVD bug.

---

## Test Vectors

### TV-F02-SVD-01: Ill-conditioned SVD (BUG CHECK)
```
A = diag([1.0, 1e-8, 1e-12]) rotated by random orthogonal matrix
Expected: σ = [1.0, 1e-8, 1e-12]
Currently: σ[2] ≈ 0 or NaN (A^T A loses precision)
```

### TV-F02-SVD-02: cond() for moderate conditioning
```
A = diag([100, 1, 0.01])
Expected: κ = 10000
Test: verify cond(A) within 1% of expected
```

### TV-F02-RANK-01: Rank of nearly singular matrix
```
A = diag([1.0, 1e-8, 1e-16])
rank(A, 1e-10) should be 2 (σ₃ < tol)
Currently: may report rank=1 because A^T A loses σ₂
```

### TV-F02-PINV-01: Pseudoinverse accuracy
```
A = [1 2; 3 4; 5 6] (3×2, rank 2)
A⁺ A should be identity (2×2) within 1e-10
Currently: may have larger error due to squared κ in SVD
```

### TV-F02-QR-01: QR orthogonality
```
A = random 100×10 matrix
Q^T Q should be identity within 1e-12
R should be upper triangular
```

### TV-F02-EIG-01: Eigenvalues of known matrix
```
A = [[2, 1], [1, 3]]
Expected eigenvalues: (5±√5)/2 ≈ 3.618, 1.382
```

### TV-F02-EIG-02: Large-magnitude matrix
```
A = 1e12 * [[2, 1], [1, 3]]
Expected eigenvalues: 1e12 * [3.618, 1.382]
Test: absolute convergence threshold still works (it does, 1e-14 << 1e12)
```

### TV-F02-EIG-03: Small-magnitude matrix
```
A = 1e-12 * [[2, 1], [1, 3]]
Expected eigenvalues: 1e-12 * [3.618, 1.382]
Test: absolute convergence threshold 1e-14 > off-diag ≈ 1e-12 → premature convergence!
Currently: may give poor accuracy
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F02-1: SVD via A^T A | **HIGH** | Squared κ, destroys small σ | Bidiagonalize or QR-based SVD |
| F02-2: Absolute Jacobi tol | **MEDIUM** | Poor for small-magnitude matrices | Relative tolerance |
| F02-3: Power iteration start | **LOW** | May converge to wrong eigenpair | Random start |
| F02-4: Misleading docstring | **LOW** | Confuses readers | Fix comments |
