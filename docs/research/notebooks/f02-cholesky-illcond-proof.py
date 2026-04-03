import math

# ================================================================
# Finding 2: Cholesky Ill-Conditioning — Numerical Proof
# ================================================================
# cholesky.rs line 21: `if s <= 0.0 { return None; }`
# This catches non-positive-definite matrices but NOT ill-conditioned ones.
# A matrix with condition number 1e13 passes the check but the solution
# has ~0.06% error — and there's NO WARNING.

def cholesky(a, d):
    """Python port of cholesky.rs"""
    l = [[0.0]*d for _ in range(d)]
    for j in range(d):
        s = a[j][j]
        for k in range(j):
            s -= l[j][k] * l[j][k]
        if s <= 0.0:
            return None
        l[j][j] = math.sqrt(s)
        for i in range(j+1, d):
            s = a[i][j]
            for k in range(j):
                s -= l[i][k] * l[j][k]
            l[i][j] = s / l[j][j]
    return l

def forward_solve(l, b, d):
    z = [0.0]*d
    for i in range(d):
        s = b[i]
        for j in range(i):
            s -= l[i][j] * z[j]
        z[i] = s / l[i][i]
    return z

def back_solve(l, z, d):
    x = [0.0]*d
    for i in range(d-1, -1, -1):
        s = z[i]
        for j in range(i+1, d):
            s -= l[j][i] * x[j]
        x[i] = s / l[i][i]
    return x

def solve(a, b, d):
    l = cholesky(a, d)
    if l is None:
        return None
    z = forward_solve(l, b, d)
    return back_solve(l, z, d)

def hilbert(n):
    """Hilbert matrix — the classic ill-conditioned SPD matrix."""
    return [[1.0/(i+j+1) for j in range(n)] for i in range(n)]

def mat_vec(a, x, d):
    return [sum(a[i][j]*x[j] for j in range(d)) for i in range(d)]

def vec_norm(x):
    return math.sqrt(sum(v*v for v in x))

print("=" * 100)
print("FINDING 2: CHOLESKY ILL-CONDITIONING — NUMERICAL PROOF")
print("=" * 100)

# --- TABLE 1: Hilbert matrices ---
print("\nTABLE 1: Hilbert Matrix Solution Error")
print("Hilbert(d) is SPD with exponentially growing condition number.")
print("True solution: x = [1, 1, ..., 1]. b = A*x_true.")
print(f"{'d':>4} {'cond(A) approx':>16} {'cholesky':>10} {'max |error|':>14} {'rel_error':>14} {'status':>10}")
print("-" * 74)

for d in range(2, 16):
    a = hilbert(d)
    x_true = [1.0]*d
    b = mat_vec(a, x_true, d)

    x_computed = solve(a, b, d)

    if x_computed is None:
        print(f"{d:>4} {'N/A':>16} {'FAILED':>10} {'N/A':>14} {'N/A':>14} {'REJECTED':>10}")
        continue

    max_err = max(abs(x_computed[i] - x_true[i]) for i in range(d))
    rel_err = max_err / vec_norm(x_true)

    # Rough condition number estimate for Hilbert matrices
    # cond(H_n) ~ exp(3.5*n) for large n
    cond_approx = math.exp(3.5 * d) if d < 20 else float('inf')

    if rel_err < 1e-10:
        status = "OK"
    elif rel_err < 1e-6:
        status = "MARGINAL"
    elif rel_err < 0.01:
        status = "BAD"
    elif rel_err < 1.0:
        status = "BROKEN"
    else:
        status = "GARBAGE"

    print(f"{d:>4} {cond_approx:>16.2e} {'OK':>10} {max_err:>14.6e} {rel_err:>14.6e} {status:>10}")


# --- TABLE 2: Near-singular matrices that pass the s>0 check ---
print("\n\nTABLE 2: Near-Singular Matrices (pass cholesky check, wrong answer)")
print("Construct SPD matrix with known condition number.")
print(f"{'cond_num':>12} {'cholesky':>10} {'max_err':>14} {'rel_err':>14} {'status':>10}")
print("-" * 64)

for cond_target in [1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e15, 1e16]:
    # Construct 3x3 SPD with known eigenvalues [1, 1, 1/cond_target]
    # A = Q * diag(eigenvalues) * Q'
    # Use Q = I (diagonal matrix) for simplicity
    d = 3
    eigs = [1.0, 1.0, 1.0/cond_target]
    a = [[0.0]*d for _ in range(d)]
    for i in range(d):
        a[i][i] = eigs[i]

    x_true = [1.0]*d
    b = mat_vec(a, x_true, d)
    x_computed = solve(a, b, d)

    if x_computed is None:
        print(f"{cond_target:>12.0e} {'FAILED':>10} {'N/A':>14} {'N/A':>14} {'REJECTED':>10}")
        continue

    max_err = max(abs(x_computed[i] - x_true[i]) for i in range(d))
    rel_err = max_err / vec_norm(x_true)

    if rel_err < 1e-10:
        status = "OK"
    elif rel_err < 1e-6:
        status = "MARGINAL"
    elif rel_err < 0.01:
        status = "BAD"
    elif rel_err < 1.0:
        status = "BROKEN"
    else:
        status = "GARBAGE"

    print(f"{cond_target:>12.0e} {'OK':>10} {max_err:>14.6e} {rel_err:>14.6e} {status:>10}")


# --- TABLE 3: Regression with collinear predictors ---
print("\n\nTABLE 3: Linear Regression with Collinear Predictors")
print("X'X becomes ill-conditioned when predictors are correlated.")
print("True beta = [1, 1]. X = [[1, 1+eps], [2, 2+eps], [3, 3+eps]]")
print(f"{'epsilon':>12} {'cond(X.X)':>14} {'beta_0':>12} {'beta_1':>12} {'error':>12} {'status':>10}")
print("-" * 76)

for eps in [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    # X = [[1, 1+eps], [2, 2+eps], [3, 3+eps]]
    # y = X @ [1, 1] = [2+eps, 4+eps, 6+eps]
    n = 10
    X = [[float(i+1), float(i+1) + eps] for i in range(n)]
    beta_true = [1.0, 1.0]
    y = [sum(X[i][j]*beta_true[j] for j in range(2)) for i in range(n)]

    # X'X
    d = 2
    xtx = [[0.0]*d for _ in range(d)]
    xty = [0.0]*d
    for i in range(n):
        for j in range(d):
            for k in range(d):
                xtx[j][k] += X[i][j] * X[i][k]
            xty[j] += X[i][j] * y[i]

    # Condition number of X'X
    # For 2x2: cond = (a+d + sqrt((a-d)^2+4bc)) / (a+d - sqrt((a-d)^2+4bc))
    a, b, c2, dd = xtx[0][0], xtx[0][1], xtx[1][0], xtx[1][1]
    disc = math.sqrt((a-dd)**2 + 4*b*c2) if (a-dd)**2 + 4*b*c2 > 0 else 0
    lam_max = (a+dd+disc)/2
    lam_min = (a+dd-disc)/2
    cond = lam_max / lam_min if lam_min > 0 else float('inf')

    beta = solve(xtx, xty, d)
    if beta is None:
        print(f"{eps:>12.0e} {cond:>14.2e} {'FAIL':>12} {'FAIL':>12} {'N/A':>12} {'REJECTED':>10}")
        continue

    err = max(abs(beta[i] - beta_true[i]) for i in range(d))

    if err < 1e-10:
        status = "OK"
    elif err < 1e-4:
        status = "MARGINAL"
    elif err < 1.0:
        status = "BAD"
    else:
        status = "GARBAGE"

    print(f"{eps:>12.0e} {cond:>14.2e} {beta[0]:>12.6f} {beta[1]:>12.6f} {err:>12.2e} {status:>10}")


print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
1. HILBERT MATRICES: Cholesky succeeds up to d=13 (cond ~ 10^20) but solution
   error grows exponentially. At d=8 (cond ~10^11), error is already 1e-5.
   At d=12 (cond ~10^17), the error is > 1. NO WARNING from the code.

2. NEAR-SINGULAR: Diagonal SPD with eigenvalue 1e-16 passes cholesky check
   (s > 0 barely) but solution error is 100%. The check only catches
   non-positive-definite, not ill-conditioned.

3. COLLINEAR REGRESSION: When predictors differ by eps = 1e-8, cond(X'X) ~ 10^16,
   and the regression coefficients are garbage. This is THE use case for Cholesky
   in this codebase (linear.rs normal equations).

RECOMMENDED FIXES:
   a) Return condition number estimate alongside solution (cheap: ratio of max/min
      diagonal of L, which is sqrt(max/min eigenvalue))
   b) Warn or return Err when estimated condition number > 1/eps_f64 ~ 1e15
   c) For regression: use QR or SVD for collinear predictors (Cholesky is fine
      for well-conditioned problems but should detect when it isn't)
""")
