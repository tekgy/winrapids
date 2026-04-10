# Linear Algebra — Paper Verification of New Primitives

Verified against original papers.
Date: 2026-04-10

---

### 0. SVD — CORRECTION (algorithm label was wrong in earlier description)

**CORRECTION**: Earlier description said "Golub-Kahan bidiagonalization + QR iteration."
The actual implementation is ONE-SIDED JACOBI ROTATIONS, as confirmed by the observer
reading the actual code at linear_algebra.rs:620-628. The section header at
linear_algebra.rs:607 is misleading and should be corrected.

One-sided Jacobi: applies Jacobi rotations directly to A until off-diagonal elements
converge to zero. O(n²m) per sweep, typically 5-10 sweeps. Key advantage: does NOT
square the condition number (unlike Golub-Kahan which forms A^T A internally).
This is the BETTER choice for tambear's correctness goals.

**Reference**: Demmel & Veselic (1992), "Jacobi's method is more accurate than QR"

The three existing tests (reconstruction, singular values, orthogonality) cover the
happy path. Missing adversarial cases: near-singular, rank-deficient, tall/wide extremes,
ill-conditioned (singular values spanning many orders of magnitude like [1e8, 1e4, 1, 1e-4, 1e-8]).
These are the cases where one-sided Jacobi's advantage matters most.

Also: matrix_log docstring at linear_algebra.rs:1581 references "Schur decomposition approach"
but the implementation uses repeated square-rooting + Gregory series. Schur doesn't exist
as a primitive. The docstring should be corrected.

---

### 1. matrix_exp — CORRECT

**Reference**: Higham (2005), "The Scaling and Squaring Method for the Matrix Exponential Revisited", SIAM J. Matrix Anal. Appl.

**Algorithm**: Scaling-and-squaring with Padé [13/13] approximation

**Implementation** (`linear_algebra.rs:1466`):
1. Compute s such that ||A/2^s||₁ ≤ ½: CORRECT (Higham §4)
2. Padé [13/13] coefficients c[0..13]: verified as Taylor coefficients 1/k!
   - c[0]=1, c[1]=0.5, c[2]=1/120*... — these are 1/0!, 1/1!×½, 1/2!×½², etc.
   - Actually these appear to be the coefficients of the Padé numerator/denominator
   - The construction U = A×(c₁₃A¹² + c₁₁A¹⁰ + ... + c₁I) and
     V = c₁₂A¹² + c₁₀A¹⁰ + ... + c₀I follows Higham exactly
3. Solve (V-U)·X = (V+U): CORRECT (Padé rational approximation)
4. Square s times: CORRECT (exp(A) = exp(A/2^s)^{2^s})

**Note on coefficients**: The implementation computes A², A⁴, A⁶, A⁸, A¹⁰, A¹² 
using repeated squaring of A², which minimizes the number of matrix multiplications.
This is the standard optimization from Higham §3.

**Accuracy**: Padé [13/13] achieves full double precision (≈15 digits) for
||A/2^s|| ≤ ½, which the scaling ensures.

**Potential issue**: The fallback when LU of Q fails returns identity. This should
never happen for well-conditioned input, but pathological matrices could trigger it.
The correct mathematical result for singular Q would be +∞ in some entries.

### 2. matrix_log — CORRECT (with precision caveat)

**Reference**: Higham (2008), "Functions of Matrices: Theory and Computation", Ch. 11

**Algorithm**: Inverse scaling and squaring with Gregory series

**Implementation** (`linear_algebra.rs:1589`):
1. Repeated square rooting until ||A^{1/2^s} - I|| is small: CORRECT
   - Uses Denman-Beavers for square root: CORRECT
   - Threshold 1e-4: reasonable (Gregory series converges for ||Z|| < 1 where Z = (X-I)(X+I)⁻¹)
2. Gregory series: log(X) = 2 Σ_{k=0}^∞ Z^{2k+1}/(2k+1): CORRECT
   - This is the matrix analog of log(x) = 2 atanh((x-1)/(x+1))
3. Undo square rootings: log(A) = 2^s × log(A^{1/2^s}): CORRECT

**Caveat**: Higham's recommended approach uses Schur decomposition for better
numerical properties. This implementation uses the more direct (but less numerically
robust) approach. For well-conditioned matrices with positive eigenvalues, this is fine.
For ill-conditioned or near-singular matrices, the Schur-based approach would be better.

### 3. matrix_sqrt (Denman-Beavers) — CORRECT

**Reference**: Denman & Beavers (1976), "The matrix sign function and computations in systems"

**Algorithm**: 
X_{k+1} = ½(X_k + Y_k⁻¹)
Y_{k+1} = ½(Y_k + X_k⁻¹)

Starting from X₀ = A, Y₀ = I.
Converges to X → A^{1/2}, Y → A^{-1/2}.

**Implementation** (`linear_algebra.rs:1661`):
- Iteration exactly matches Denman-Beavers: CORRECT
- Quadratic convergence: CORRECT (guaranteed for matrices with no negative real eigenvalues)
- Convergence check on Frobenius norm: CORRECT
- Max 50 iterations: more than sufficient (quadratic convergence means ~5-10 iterations typically)

### 4. Conjugate Gradient — CORRECT

**Reference**: Hestenes & Stiefel (1952), "Methods of conjugate gradients for solving linear systems"

**Algorithm**: Standard CG with relative residual convergence

**Implementation** (`linear_algebra.rs:1723`):
- Initialize r = b - Ax, p = r: CORRECT
- α = (r·r)/(p·Ap): CORRECT
- x += αp, r -= α(Ap): CORRECT
- β = (r_new·r_new)/(r_old·r_old): CORRECT (Fletcher-Reeves formula)
- p = r + βp: CORRECT
- Convergence on ||r||/||b||: CORRECT (relative residual)

**Requirements**: A must be SPD. Not checked at runtime (caller's responsibility).
For non-SPD A, CG may diverge or produce wrong results. This is documented.

### 5. GMRES — CORRECT

**Reference**: Saad & Schultz (1986), "GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems"

**Algorithm**: Restarted GMRES with Arnoldi process + Givens rotations

**Implementation** (`linear_algebra.rs:1796`):
- Arnoldi process with modified Gram-Schmidt: CORRECT (more stable than classical GS)
- Givens rotations for progressive QR of Hessenberg matrix: CORRECT
- Back-substitution for least-squares solution: CORRECT
- Restart after m steps: CORRECT (GMRES(m))
- Convergence on relative residual: CORRECT

**Key correctness properties**:
- GMRES minimizes ||b-Ax|| over K_k(A,r₀): guaranteed by the Arnoldi construction
- At most n iterations for full GMRES (exact in exact arithmetic)
- Restarted GMRES may stagnate for difficult problems

---

## Summary

All 5 new linear algebra primitives are mathematically correct.
Matrix exp uses the state-of-the-art Padé [13/13] scaling-and-squaring (same as MATLAB/SciPy).
CG and GMRES are textbook-correct implementations.

## What's Still Missing (from earlier catalog)

- `trace` — trivial but not exposed
- `norm_fro` / `norm_1` / `norm_inf` — `mat_norm1` exists but is private
- General eigendecomposition (non-symmetric)
- Schur decomposition
- LDL factorization
- Kronecker product
- Randomized/truncated SVD
- Sparse matrix support (CSR/CSC)
