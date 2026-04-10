# Linear Algebra — Complete Variant Catalog

## What Exists (tambear::linear_algebra)

### Matrix Operations (6)
- `mat_mul`, `mat_add`, `mat_sub`, `mat_scale` — basic arithmetic
- `mat_vec` — matrix-vector multiply
- `dot`, `vec_norm`, `outer` — vector operations

### Factorizations (6)
- `lu(a)` → LuResult {l, u, perm}
- `cholesky(a)` → Option<Mat> (lower triangular L where A = LLᵀ)
- `qr(a)` → QrResult {q, r}
- `svd(a)` → SvdResult {u, s, vt} — Golub-Kahan bidiagonalization
- `sym_eigen(a)` → (eigenvalues, eigenvectors) — QR algorithm with Wilkinson shifts
- `power_iteration(a, max_iter, tol)` — dominant eigenvalue/eigenvector

### Solvers (7)
- `lu_solve`, `cholesky_solve`, `qr_solve` — factorization-based
- `solve(a, b)` — general (delegates to LU)
- `solve_spd(a, b)` — for symmetric positive definite (delegates to Cholesky)
- `lstsq(a, b)` — least squares via QR
- `solve_tridiagonal(a, b, c, d)` — Thomas algorithm + scan variant

### Properties (4)
- `det(a)`, `log_det(a)` — determinant via LU
- `inv(a)` — matrix inverse via LU
- `pinv(a, rcond)` — pseudoinverse via SVD
- `cond(a)` — condition number (ratio of singular values)
- `rank(a, tol)` — numerical rank via SVD

### Orthogonalization (2)
- `gram_schmidt(vectors)` — classical
- `gram_schmidt_modified(vectors)` — numerically stable variant

### Regression (3)
- `simple_linear_regression(x, y)` — y = a + bx
- `ols_slope(x, y)` — slope only
- `ols_normal_equations(x, y, nobs, ncols)` — multivariate OLS
- `ols_residuals(x, y)` — residuals from simple regression

### Other (5)
- `sigmoid(x)` — logistic function (1/(1+e^{-x}))
- `effective_rank_from_sv(sv)` — Shannon entropy of normalized singular values
- `tridiagonal_scan_element/compose` — scan-composable tridiagonal solve
- `forward_solve(l, b)`, `back_solve_transpose(l, b)` — triangular solvers

---

## What's MISSING — Complete Catalog

### A. Missing Factorizations

1. **Eigendecomposition (general, non-symmetric)** — complex eigenvalues
   - Current `sym_eigen` only works for symmetric matrices
   - General case: A = V Λ V⁻¹ where Λ may be complex
   - Need: Hessenberg reduction → QR iteration with implicit shifts
   - Returns: `(Vec<(f64,f64)>, Mat)` — (real, imag) eigenvalues + eigenvectors
   - Failure mode: defective matrices (algebraic > geometric multiplicity)

2. **Schur decomposition** — A = QTQ* where T is upper triangular
   - Parameters: `a: &Mat` → `SchurResult { q, t }`
   - Foundation for: matrix functions (exp, log, sqrt), eigenvalue stability
   - Real Schur form: T is quasi-upper-triangular (2×2 blocks for complex pairs)

3. **Hessenberg decomposition** — A = QHQ* where H is upper Hessenberg
   - Parameters: `a: &Mat` → `HessenbergResult { q, h }`
   - Intermediate step for eigenvalue algorithms
   - O(n³) via Householder reflections

4. **Polar decomposition** — A = UP where U is unitary, P is positive semi-definite
   - Via SVD: U = U_svd V_svd^T, P = V_svd Σ V_svd^T
   - Parameters: `a: &Mat` → `PolarResult { u, p }`
   - Use case: closest rotation matrix, continuum mechanics

5. **LDL decomposition** — A = LDLᵀ where D is diagonal
   - Avoids square roots (cheaper than Cholesky)
   - Works for indefinite symmetric matrices (D can have negative entries)
   - Parameters: `a: &Mat` → `LdlResult { l, d, perm }`
   - Pivoted version: Bunch-Kaufman

6. **Sparse LU / Cholesky** — for banded and sparse matrices
   - Current Mat is dense; need CSR/CSC format + fill-reducing ordering
   - At minimum: banded LU for tridiagonal/pentadiagonal

7. **Generalized eigenvalue** — Ax = λBx
   - Parameters: `a: &Mat, b: &Mat` → `(Vec<f64>, Mat)`
   - Via: Cholesky of B, then standard eigenproblem on L⁻¹AL⁻ᵀ
   - Assumption: B must be positive definite

8. **Jordan normal form** — A = PJP⁻¹ where J has eigenvalues on diagonal
   - Theoretical importance, rarely computed numerically
   - But needed for: matrix exponential of defective matrices

### B. Missing Matrix Functions

1. **Matrix exponential** — exp(A) = Σ Aⁿ/n!
   - Practical: Padé approximation with scaling-and-squaring (Higham 2005)
   - Parameters: `a: &Mat` → `Mat`
   - Use case: ODE solutions x(t) = exp(At)x₀, Lie groups
   - Shares: eigendecomposition or Schur decomposition

2. **Matrix logarithm** — log(A) where exp(log(A)) = A
   - Inverse Padé + scaling-and-squaring
   - Requires: all eigenvalues have positive real parts
   - Parameters: `a: &Mat` → `Option<Mat>`

3. **Matrix square root** — A^{1/2} where (A^{1/2})² = A
   - Denman-Beavers iteration or Schur-based
   - Parameters: `a: &Mat` → `Option<Mat>`
   - Use case: Mahalanobis whitening, covariance square roots

4. **Matrix power** — A^p for real p (fractional power)
   - Via eigendecomposition: A^p = V Λ^p V⁻¹
   - Parameters: `a: &Mat`, `p: f64` → `Mat`

5. **Matrix sign function** — sign(A) where sign²(A) = I
   - Newton iteration: X_{k+1} = (X_k + X_k⁻¹)/2
   - Use case: spectral dichotomy, Riccati equations

### C. Missing Structured Solvers

1. **Toeplitz solve** — O(n²) via Levinson-Durbin (already in time_series)
   - Should be extracted as a linear algebra primitive
   - Toeplitz: constant along diagonals → autocorrelation matrices

2. **Circulant solve** — O(n log n) via FFT
   - Circulant = special Toeplitz where each row is a cyclic shift
   - Diagonalized by DFT: C = F* Λ F, solve via FFT
   - Parameters: `first_row: &[f64]`, `b: &[f64]`

3. **Vandermonde solve** — O(n²)
   - V(x₁,...,xₙ) × c = y
   - Parameters: `nodes: &[f64]`, `values: &[f64]`
   - Use case: polynomial interpolation

4. **Symmetric positive definite banded** — O(n×bw²) Cholesky
   - For: covariance matrices with known bandwidth

5. **Block diagonal solve** — independent blocks solved in parallel
   - Parameters: `blocks: &[Mat]`, `b: &[f64]`
   - Kingdom A: perfectly parallelizable

6. **Kronecker product** — A ⊗ B
   - (A⊗B)(x⊗y) = (Ax)⊗(By)
   - Parameters: `a: &Mat`, `b: &Mat` → `Mat`
   - Kronecker solve: (A⊗B)x = b via vec(X) where AXBᵀ = C

### D. Missing Iterative Solvers

1. **Conjugate gradient** (CG) — for SPD systems
   - O(n√κ) where κ = condition number
   - Parameters: `a_fn: Fn(&[f64]) -> Vec<f64>`, `b`, `tol`, `max_iter`
   - Matrix-free: only needs matrix-vector product
   - Preconditioned CG: `preconditioner: Option<Fn>`

2. **GMRES** — for general non-symmetric systems
   - Generalized Minimum Residual
   - Parameters: `a_fn`, `b`, `restart`, `tol`, `max_iter`
   - Restarted GMRES to bound memory

3. **BiCGSTAB** — for non-symmetric systems
   - Bi-Conjugate Gradient Stabilized
   - Parameters: same as CG
   - Better convergence than CG for non-symmetric

4. **Jacobi / Gauss-Seidel / SOR** — classical iterative
   - Simple, useful as preconditioners
   - Parameters: `a: &Mat`, `b`, `omega` (relaxation), `tol`, `max_iter`

5. **Lanczos algorithm** — for symmetric eigenvalue problems
   - Tridiagonalizes a symmetric matrix
   - Only needs matrix-vector products
   - Parameters: `a_fn`, `n`, `k` (number of eigenvalues), `tol`
   - Foundation for: sparse symmetric eigenvalues

6. **Arnoldi iteration** — for non-symmetric eigenvalue problems
   - Produces upper Hessenberg form
   - Foundation for: GMRES, implicitly restarted Arnoldi

### E. Missing Norms and Properties

1. **Matrix norms** — currently only have `cond` which uses 2-norm implicitly
   - `norm_1(a)` — max column sum of |aᵢⱼ|
   - `norm_inf(a)` — max row sum of |aᵢⱼ|
   - `norm_fro(a)` — Frobenius √(Σ aᵢⱼ²)
   - `norm_2(a)` — spectral norm = largest singular value (already computed in SVD)
   - `norm_nuclear(a)` — nuclear norm = Σ σᵢ

2. **Trace** — `trace(a) = Σ aᵢᵢ`
   - Trivial but should be a first-class primitive
   - trace(AB) = trace(BA), trace(A) = Σ eigenvalues

3. **Permanent** — like determinant but all positive signs
   - #P-hard in general; exact for small matrices
   - Ryser formula: O(2^n × n)

4. **Matrix definiteness test** — positive definite / semidefinite / indefinite
   - Via Cholesky attempt or eigenvalue signs
   - Parameters: `a: &Mat` → `Definiteness` enum

5. **Numerical rank profile** — which columns are linearly independent
   - Via column-pivoted QR or SVD
   - Returns: pivot indices, effective rank

### F. Missing Special Matrices

1. **Hadamard matrix** — entries ±1, H^T H = nI
   - Sylvester construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
   - Parameters: `n: usize` (must be power of 2) → `Mat`

2. **Householder reflector** — H = I - 2vvᵀ/vᵀv
   - Already used implicitly in QR but not exposed as primitive
   - Parameters: `v: &[f64]` → `Mat` (or apply in-place)

3. **Givens rotation** — G(i,j,θ) rotates in the (i,j) plane
   - Used for: QR factorization (element-by-element zeroing), Jacobi SVD
   - Parameters: `a: f64`, `b: f64` → `(c, s)` (cosine, sine)

4. **Permutation matrix** — P applied to rows/columns
   - Already tracked in LU but not exposed as standalone
   - Parameters: `perm: &[usize]` → apply to vector or matrix

5. **Diagonal matrix** operations — scale, invert, multiply
   - Parameters: `diag: &[f64]`
   - Critical for efficiency: never form a full n×n diagonal

### G. Missing Specialized Decompositions

1. **CUR decomposition** — A ≈ CUR where C,R are column/row subsets
   - Parameters: `a: &Mat`, `k: usize` (rank) → `CurResult { c, u, r }`
   - Advantage over SVD: interpretable (actual rows and columns)

2. **Interpolative decomposition** — A ≈ A(:,J) × Z
   - Column skeleton approximation
   - Parameters: `a: &Mat`, `k: usize` → indices J and coefficient Z

3. **Non-negative matrix factorization** (NMF) — A ≈ WH where W,H ≥ 0
   - Parameters: `a: &Mat`, `k: usize`, `max_iter`, `tol`
   - Multiplicative update rules (Lee & Seung 2001)
   - Alternating least squares variant

4. **Randomized SVD** — Halko et al. 2011
   - O(mn log k) for rank-k approximation instead of O(mn²)
   - Parameters: `a: &Mat`, `k: usize`, `oversampling: usize`
   - Critical for large matrices (n > 1000)

5. **Truncated SVD / economy SVD** — only top k singular values/vectors
   - Current SVD computes full; need truncated variant
   - Parameters: `a: &Mat`, `k: usize` → `SvdResult`

6. **Complete orthogonal decomposition** — A = U [T 0; 0 0] Z*
   - Rank-revealing, numerically stable pseudoinverse
   - Via column-pivoted QR + LQ

---

## Decomposition into Primitives

```
householder_reflector ──┬── qr
                        ├── hessenberg_reduction
                        ├── bidiagonalization (for SVD)
                        └── tridiagonalization (for sym_eigen)

givens_rotation ────────┬── qr (row variant)
                        ├── jacobi_svd
                        └── plane_rotation

mat_vec product ────────┬── power_iteration
                        ├── lanczos
                        ├── arnoldi
                        ├── conjugate_gradient
                        ├── gmres
                        └── bicgstab

cholesky ──────────────┬── solve_spd
                       ├── mahalanobis_distance
                       ├── generalized_eigenvalue
                       ├── matrix_sqrt (for SPD)
                       └── whitening

svd ───────────────────┬── pinv
                       ├── lstsq
                       ├── rank
                       ├── cond
                       ├── effective_rank
                       ├── matrix_sqrt
                       ├── polar_decomposition
                       ├── low_rank_approx
                       └── nuclear_norm

eigendecomposition ────┬── matrix_exp
                       ├── matrix_log
                       ├── matrix_power
                       ├── spectral_radius
                       └── definiteness_test

lu ────────────────────┬── det
                       ├── solve
                       ├── inv
                       └── matrix_sign
```

## Intermediate Sharing Map

| Intermediate | Tag | Consumers |
|---|---|---|
| SVD (full) | `svd(A, full)` | pinv, lstsq, rank, cond, eff_rank, polar, low_rank |
| Eigenvalues | `eigen(A)` | exp, log, pow, spectral_radius, definiteness |
| Cholesky factor L | `chol(A)` | solve_spd, whiten, mahalanobis, gen_eigen |
| QR factors | `qr(A)` | lstsq, rank_profile, solve |
| LU factors | `lu(A)` | det, solve, inv |
| Hessenberg form | `hess(A)` | eigenvalues (QR iteration) |
| Frobenius norm | `norm(A, fro)` | scaling for matrix exp, condition estimation |

## Priority

**Tier 1** — Foundation (unlock many downstream methods):
1. `trace` — trivial, used everywhere
2. `norm_fro`, `norm_1`, `norm_inf` — needed for iterative methods, matrix functions
3. `householder` + `givens` as exposed primitives — already used internally
4. `general_eigen` (non-symmetric) — needed for matrix exp, dynamical systems
5. `truncated_svd` / `randomized_svd` — performance for large matrices
6. `conjugate_gradient` — foundation iterative solver
7. `kronecker_product` — needed for vec-operator, multivariate stats

**Tier 2** — Important:
8. `matrix_exp` — ODE solutions, Lie theory
9. `schur_decomposition` — foundation for matrix functions
10. `ldl` — cheaper than Cholesky for symmetric indefinite
11. `circulant_solve` — O(n log n) via FFT
12. `gmres` — non-symmetric iterative solver
13. `polar_decomposition` — geometry, continuum mechanics

**Tier 3** — Specialist:
14-20: NMF, CUR, matrix log/sqrt/sign, Vandermonde, block diagonal
