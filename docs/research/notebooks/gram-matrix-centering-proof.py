import math

# ================================================================
# Does centering help GramMatrix regression the same way it helps variance?
# Naturalist's question: is (X-mu)'(X-mu) more stable than X'X?
# ================================================================
# Short answer: YES, and the proof is elegant.
# Variance IS the (1,1) entry of the centered GramMatrix for 1 column.
# The GramMatrix case is a direct generalization.

def solve_normal_equations(X, y):
    """Solve (X'X)beta = X'y using Cholesky-like approach."""
    n = len(X)
    d = len(X[0])

    # Form X'X
    xtx = [[0.0]*d for _ in range(d)]
    xty = [0.0]*d
    for i in range(n):
        for j in range(d):
            for k in range(d):
                xtx[j][k] += X[i][j] * X[i][k]
            xty[j] += X[i][j] * y[i]

    # Solve via Gaussian elimination (simple, not production)
    # Augmented matrix
    aug = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(d):
        # Pivot
        max_row = col
        for row in range(col+1, d):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-300:
            return None  # singular

        for row in range(col+1, d):
            f = aug[row][col] / aug[col][col]
            for j in range(col, d+1):
                aug[row][j] -= f * aug[col][j]

    # Back substitution
    beta = [0.0]*d
    for i in range(d-1, -1, -1):
        beta[i] = aug[i][d]
        for j in range(i+1, d):
            beta[i] -= aug[i][j] * beta[j]
        beta[i] /= aug[i][i]

    return beta

def solve_centered(X, y):
    """Center X and y, solve centered normal equations, recover intercept."""
    n = len(X)
    d = len(X[0])

    # Compute means
    x_means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    y_mean = sum(y) / n

    # Center (skip intercept column if present)
    Xc = [[X[i][j] - x_means[j] for j in range(d)] for i in range(n)]
    yc = [y[i] - y_mean for i in range(n)]

    # Form centered X'X
    xtx = [[0.0]*d for _ in range(d)]
    xty = [0.0]*d
    for i in range(n):
        for j in range(d):
            for k in range(d):
                xtx[j][k] += Xc[i][j] * Xc[i][k]
            xty[j] += Xc[i][j] * yc[i]

    # Solve
    aug = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(d):
        max_row = col
        for row in range(col+1, d):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-300:
            return None
        for row in range(col+1, d):
            f = aug[row][col] / aug[col][col]
            for j in range(col, d+1):
                aug[row][j] -= f * aug[col][j]

    beta_slopes = [0.0]*d
    for i in range(d-1, -1, -1):
        beta_slopes[i] = aug[i][d]
        for j in range(i+1, d):
            beta_slopes[i] -= aug[i][j] * beta_slopes[j]
        beta_slopes[i] /= aug[i][i]

    # Recover intercept: y_mean - sum(beta_j * x_mean_j)
    intercept = y_mean - sum(beta_slopes[j] * x_means[j] for j in range(d))

    return [intercept] + beta_slopes

print("=" * 100)
print("GRAM MATRIX CENTERING: DOES IT HELP REGRESSION?")
print("=" * 100)

# --- TABLE 1: Simple regression y = 2 + 3*x, x near offset ---
print("\nTABLE 1: Simple Regression y = 2 + 3*x, predictor near large offset")
print("X has intercept column [1] and predictor column [offset + i*0.1]")
print(f"{'offset':>12} {'naive b0':>12} {'naive b1':>12} {'centered b0':>12} {'centered b1':>12} {'naive err':>12} {'cent err':>12}")
print("-" * 92)

n = 100
true_b0 = 2.0
true_b1 = 3.0

for offset in [0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14]:
    # X = [1, offset + i*0.1] for i in range(n)
    X_naive = [[1.0, offset + i * 0.1] for i in range(n)]
    y = [true_b0 + true_b1 * (offset + i * 0.1) for i in range(n)]

    beta_naive = solve_normal_equations(X_naive, y)

    # For centered solve: use only the predictor column (1 col), recover intercept
    X_pred = [[offset + i * 0.1] for i in range(n)]
    beta_centered = solve_centered(X_pred, y)

    if beta_naive is None:
        naive_str = "FAILED"
        naive_err = float('inf')
    else:
        naive_err = max(abs(beta_naive[0] - true_b0), abs(beta_naive[1] - true_b1))
        naive_str = f"{beta_naive[0]:>12.4f} {beta_naive[1]:>12.6f}"

    if beta_centered is None:
        cent_str = "FAILED"
        cent_err = float('inf')
    else:
        cent_err = max(abs(beta_centered[0] - true_b0), abs(beta_centered[1] - true_b1))
        cent_str = f"{beta_centered[0]:>12.4f} {beta_centered[1]:>12.6f}"

    if beta_naive is not None and beta_centered is not None:
        print(f"{offset:>12.0e} {beta_naive[0]:>12.4f} {beta_naive[1]:>12.6f} {beta_centered[0]:>12.4f} {beta_centered[1]:>12.6f} {naive_err:>12.2e} {cent_err:>12.2e}")
    elif beta_naive is None:
        print(f"{offset:>12.0e} {'FAILED':>25} {beta_centered[0]:>12.4f} {beta_centered[1]:>12.6f} {'inf':>12} {cent_err:>12.2e}")
    else:
        print(f"{offset:>12.0e} {beta_naive[0]:>12.4f} {beta_naive[1]:>12.6f} {'FAILED':>25} {naive_err:>12.2e} {'inf':>12}")


# --- TABLE 2: Multiple regression with correlated predictors ---
print("\n\nTABLE 2: Multiple Regression y = 1 + 2*x1 + 3*x2, x1 and x2 near offset")
print("x1 = offset + i*0.1, x2 = offset + i*0.1 + noise")
print(f"{'offset':>12} {'naive err':>12} {'centered err':>12} {'improvement':>14}")
print("-" * 54)

import random
random.seed(42)

for offset in [0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]:
    n = 50
    true_b = [1.0, 2.0, 3.0]

    X_pred = []
    y = []
    for i in range(n):
        x1 = offset + i * 0.1
        x2 = offset + i * 0.1 + random.gauss(0, 0.01)
        X_pred.append([x1, x2])
        y.append(true_b[0] + true_b[1]*x1 + true_b[2]*x2)

    # Naive: with intercept column
    X_naive = [[1.0] + row for row in X_pred]
    beta_naive = solve_normal_equations(X_naive, y)

    # Centered
    beta_centered = solve_centered(X_pred, y)

    if beta_naive is not None:
        naive_err = max(abs(beta_naive[i] - true_b[i]) for i in range(3))
    else:
        naive_err = float('inf')

    if beta_centered is not None:
        cent_err = max(abs(beta_centered[i] - true_b[i]) for i in range(3))
    else:
        cent_err = float('inf')

    if naive_err > 0 and cent_err > 0 and naive_err != float('inf'):
        improvement = naive_err / cent_err
    else:
        improvement = float('inf')

    print(f"{offset:>12.0e} {naive_err:>12.2e} {cent_err:>12.2e} {improvement:>14.1f}x")


# --- TABLE 3: Condition number of X'X vs (X-mu)'(X-mu) ---
print("\n\nTABLE 3: Condition Number: X'X vs (X-mu)'(X-mu)")
print("Single predictor. Condition number of 2x2 normal equations matrix.")
print(f"{'offset':>12} {'cond(X.X)':>14} {'cond(Xc.Xc)':>14} {'ratio':>10}")
print("-" * 54)

for offset in [0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14]:
    n = 100
    x_vals = [offset + i * 0.1 for i in range(n)]
    x_mean = sum(x_vals) / n

    # X'X = [[N, Sx], [Sx, Sx2]]
    sx = sum(x_vals)
    sx2 = sum(v*v for v in x_vals)

    # Condition of 2x2: eigenvalues
    a, b, d = float(n), sx, sx2
    disc = math.sqrt((a-d)**2 + 4*b*b) if (a-d)**2 + 4*b*b >= 0 else 0
    lam1 = (a+d+disc)/2
    lam2 = (a+d-disc)/2
    cond_naive = lam1/lam2 if lam2 > 0 else float('inf')

    # Centered: Xc'Xc = [[N, 0], [0, Sx2c]]  (cross term is 0 when centered!)
    sx2c = sum((v-x_mean)**2 for v in x_vals)
    cond_centered = max(n, sx2c) / min(n, sx2c) if sx2c > 0 else float('inf')

    ratio = cond_naive / cond_centered if cond_centered > 0 else float('inf')

    print(f"{offset:>12.0e} {cond_naive:>14.2e} {cond_centered:>14.2e} {ratio:>10.0f}x")


# --- TABLE 4: Why this is THE SAME BUG as naive variance ---
print("\n\nTABLE 4: GramMatrix Cancellation = Variance Cancellation")
print("The (2,2) entry of X'X is sum(x^2). The intercept solve needs sum(x^2) - sum(x)^2/n.")
print("This IS the naive variance formula. Same bug, same fix.")
print()
print("For offset = 1e8, spread = 0.1:")
print(f"  sum(x^2)        = {sum((1e8 + i*0.1)**2 for i in range(100)):.6e}")
print(f"  sum(x)^2/n      = {sum(1e8 + i*0.1 for i in range(100))**2/100:.6e}")
print(f"  difference       = {sum((1e8 + i*0.1)**2 for i in range(100)) - sum(1e8 + i*0.1 for i in range(100))**2/100:.6e}")
print(f"  true variance*n  = {sum((i*0.1 - 4.95)**2 for i in range(100)):.6e}")
print()

# Check actual computed values
vals = [1e8 + i*0.1 for i in range(100)]
mean_v = sum(vals)/100
naive_var_n = sum(v*v for v in vals) - sum(vals)**2/100
centered_var_n = sum((v-mean_v)**2 for v in vals)
true_var_n = sum((i*0.1 - 4.95)**2 for i in range(100))

print(f"  Naive var*n      = {naive_var_n:.6e}  (error: {abs(naive_var_n - true_var_n)/true_var_n:.2e})")
print(f"  Centered var*n   = {centered_var_n:.6e}  (error: {abs(centered_var_n - true_var_n)/true_var_n:.2e})")
print(f"  True var*n       = {true_var_n:.6e}")


print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
print("""
YES. GramMatrix regression has EXACTLY the same cancellation bug as naive variance.

PROOF: The normal equations matrix X'X for [intercept, x] is:
    [[N,    Sx  ],
     [Sx,   Sx^2]]

The determinant is N*Sx^2 - Sx*Sx = N * (Sx^2/N - (Sx/N)^2) = N^2 * Var(x).
This is literally N^2 times the naive variance formula.

When Var(x) is computed via naive cancellation and goes wrong,
det(X'X) goes wrong, and the ENTIRE regression solution goes wrong.

Centering X before forming X'X:
- Eliminates the intercept column (cross-term with centered X is zero)
- Condition number drops from O(offset^2/spread^2) to O(N/Var(x))
- The improvement factor equals (mean/std)^2 — can be 10^20+ for financial data

CONCLUSION: F10 (Regression) MUST center predictors before forming normal equations.
This is the same fix as F06 (RefCenteredStats). The centering principle is universal
across Kingdom A.

The naturalist's "centered basis is a change of coordinates" observation is
mathematically precise: centering is a change of basis that diagonalizes the
cross-terms, reducing condition number by (mean/std)^2.
""")
