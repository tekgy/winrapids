"""Derive MINIMAX-OPTIMAL polynomial coefficients for tambear's exp_for_lse.

Vanilla Taylor coefficients give ~20 ULP at the edge of the reduction
interval. Tang 1989 uses Remez-minimax coefficients tuned to achieve
~1 ULP across the full r-range.

Strategy:
1. Use mpmath's polynomial fit / chebyshev-equioscillation to find
   minimax coefficients of (exp(r) - 1 - r) on |r| <= ln(2)/64.
2. Verify against test points.
3. Output bit-exact f64 hex constants.

Note: mpmath does not ship Remez directly, but we can do iterative
Chebyshev fitting which converges to minimax for low degree on small
intervals (Chebyshev nodes give near-minimax accuracy at small degree).
"""
import math
import struct
from mpmath import mp, mpf, exp, log, fac, chebyt, chop

mp.dps = 60

LN2 = log(2)
LN2_OVER_32 = LN2 / 32
LN2_OVER_64 = LN2 / 64
INV_LN2_OVER_32 = 32 / LN2

R_MAX = LN2_OVER_64


def hex_of_f64(x):
    if not isinstance(x, float):
        x = float(x)
    return hex(struct.unpack("<Q", struct.pack("<d", x))[0])


def cw_split(x_mp, low_bits_zero=20):
    x_f = float(x_mp)
    bits = struct.unpack("<Q", struct.pack("<d", x_f))[0]
    mask = (~((1 << low_bits_zero) - 1)) & ((1 << 64) - 1)
    hi_bits = bits & mask
    hi_f = struct.unpack("<d", struct.pack("<Q", hi_bits))[0]
    lo_mp = x_mp - mpf(repr(hi_f))
    lo_f = float(lo_mp)
    return hi_f, lo_f


# ============================================================
# Chebyshev approximation of (exp(r) - 1 - r) on [-R_MAX, R_MAX]
# Using polyfit through Chebyshev nodes.
# ============================================================

def chebyshev_nodes(n, a, b):
    """N Chebyshev nodes on [a, b], scaled."""
    nodes = []
    for k in range(n):
        x = mp.cos(mp.pi * (2 * k + 1) / (2 * n))
        nodes.append((a + b) / 2 + (b - a) / 2 * x)
    return nodes


def fit_polynomial_thru_nodes(xs, ys, degree):
    """Solve Vandermonde system to fit polynomial of given degree exactly
    through the given (xs, ys) interpolation nodes (degree+1 nodes)."""
    n = degree + 1
    assert len(xs) == n and len(ys) == n
    # Build Vandermonde matrix
    A = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = xs[i] ** j
    b = mp.matrix(ys)
    coeffs = mp.lu_solve(A, b)
    return [coeffs[i] for i in range(n)]


# Target function: f(r) = exp(r) - 1 - r
# We approximate this with degree-5 poly: a1*r^2 + a2*r^3 + a3*r^4 + a4*r^5
# Equivalently: degree-3 poly in (r) that we multiply by r^2.
# Or directly: 4-term poly in r starting from r^2.

# We have 4 free coefficients (a1, a2, a3, a4) and want minimax error.
# Approach: fit at 4 Chebyshev nodes scaled to [-R_MAX, R_MAX] with the
# constraint that f(0) = 0 and f'(0) = 0 are already enforced by the
# r^2 factor.

# Direct approach: fit g(r) = (exp(r) - 1 - r) / r^2 with degree-3 poly.
# Then a_k = coefficients of that polynomial.

def g(r):
    """g(r) = (exp(r) - 1 - r) / r^2 — well-defined as r→0 (limit = 1/2)."""
    if abs(r) < mpf("1e-30"):
        return mpf("0.5") + r / 6 + r * r / 24
    return (exp(r) - 1 - r) / (r * r)


# Fit g(r) on [-R_MAX, R_MAX] with degree-3 polynomial via Chebyshev nodes.
# 4 nodes give an interpolant; the equioscillation property of Chebyshev
# nodes makes this very close to the minimax (within a small constant).

# Try degree 3 first (matches Tang's expected polynomial size).
deg = 3
nodes = chebyshev_nodes(deg + 1, -R_MAX, R_MAX)
values = [g(r) for r in nodes]
coeffs = fit_polynomial_thru_nodes(nodes, values, deg)
# coeffs[k] is the coefficient of r^k in g(r). Map to a_k:
# g(r) = c0 + c1*r + c2*r^2 + c3*r^3
# (exp(r) - 1 - r) = r^2 * g(r) = c0*r^2 + c1*r^3 + c2*r^4 + c3*r^5
# So a1 = c0, a2 = c1, a3 = c2, a4 = c3
a1, a2, a3, a4 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]


def poly_approx(r, a1, a2, a3, a4):
    """exp(r) ~ 1 + r + (a1 + (a2 + (a3 + a4*r)*r)*r) * r*r"""
    inner = ((a4 * r + a3) * r + a2) * r + a1
    return mpf(1) + r + inner * r * r


# Sample many r values, measure max |error|
def measure_error(a1, a2, a3, a4, n=10000):
    max_abs = mpf(0)
    max_rel = mpf(0)
    for k in range(-n, n + 1):
        r = mpf(k) / n * R_MAX
        truth = exp(r)
        approx = poly_approx(r, a1, a2, a3, a4)
        abs_err = abs(approx - truth)
        rel_err = abs_err / abs(truth) if truth != 0 else mpf(0)
        if abs_err > max_abs:
            max_abs = abs_err
        if rel_err > max_rel:
            max_rel = rel_err
    return max_abs, max_rel


print("=" * 70)
print("CHEBYSHEV-NODE FIT — degree-3 polynomial in r for g(r)")
print("=" * 70)
print(f"  Range: |r| <= {mp.nstr(R_MAX, 15)}")
print(f"  a1 = {mp.nstr(a1, 25)}  -> f64 {float(a1)!r}")
print(f"  a2 = {mp.nstr(a2, 25)}  -> f64 {float(a2)!r}")
print(f"  a3 = {mp.nstr(a3, 25)}  -> f64 {float(a3)!r}")
print(f"  a4 = {mp.nstr(a4, 25)}  -> f64 {float(a4)!r}")
print()

# Compare to vanilla Taylor: 1/2, 1/6, 1/24, 1/120
print("  Vanilla Taylor for reference:")
print(f"  1/2!  = {mp.nstr(mpf(1)/2, 25)}")
print(f"  1/3!  = {mp.nstr(mpf(1)/6, 25)}")
print(f"  1/4!  = {mp.nstr(mpf(1)/24, 25)}")
print(f"  1/5!  = {mp.nstr(mpf(1)/120, 25)}")
print()

abs_err, rel_err = measure_error(a1, a2, a3, a4, n=5000)
print(f"  Chebyshev-fit max abs error: {mp.nstr(abs_err, 5)}")
print(f"  Chebyshev-fit max rel error: {mp.nstr(rel_err, 5)}  (~ {int(rel_err * (1<<52))} ULP near 1)")
print()

# Compare to vanilla Taylor's error
abs_err_tay, rel_err_tay = measure_error(mpf(1)/2, mpf(1)/6, mpf(1)/24, mpf(1)/120, n=5000)
print(f"  Vanilla Taylor max abs error: {mp.nstr(abs_err_tay, 5)}")
print(f"  Vanilla Taylor max rel error: {mp.nstr(rel_err_tay, 5)}  (~ {int(rel_err_tay * (1<<52))} ULP near 1)")
print()

# Convert to f64 (bit-exact embedded constants)
a1_f, a2_f, a3_f, a4_f = float(a1), float(a2), float(a3), float(a4)


# ============================================================
# Final verification: end-to-end Tang algorithm with f64-rounded
# minimax coefficients on test points
# ============================================================
ln2_32_hi, ln2_32_lo = cw_split(LN2_OVER_32, low_bits_zero=20)
inv_f = float(INV_LN2_OVER_32)

table_entries = []
for j in range(32):
    t_mp = mpf(2) ** (mpf(j) / 32)
    table_entries.append(float(t_mp))


def tang_exp_minimax(x):
    n = round(x * inv_f)
    r = (x - n * ln2_32_hi) - n * ln2_32_lo
    inner = ((a4_f * r + a3_f) * r + a2_f) * r + a1_f
    poly = inner * (r * r)
    exp_r = 1.0 + r + poly
    j = n % 32
    if j < 0:
        j += 32
    k = (n - j) // 32
    t_j = table_entries[j]
    return math.ldexp(t_j * exp_r, k)


print("=" * 70)
print("END-TO-END VERIFICATION (Tang minimax-fit + f64-rounded constants)")
print("=" * 70)
print()
test_xs = [
    0.0, -1e-15, -1e-10, -1e-5, -float(LN2_OVER_64) * 0.99,
    -float(LN2_OVER_32) * 0.5, -0.5, -1.0, -2.5, -5.0, -10.0,
    -20.0, -50.0, -100.0, -300.0, -700.0,
]
print("  test x           | tang minimax           | mpmath ref             | ULP")
print("  -----------------+------------------------+------------------------+----")
max_ulp_lse = 0
for x in test_xs:
    got = tang_exp_minimax(x)
    ref_mp = exp(mpf(repr(x)))
    ref_f = float(ref_mp)
    if got == ref_f:
        ulp = 0
    elif ref_f == 0.0:
        ulp = "inf" if got != 0 else 0
    else:
        a_bits = struct.unpack("<q", struct.pack("<d", got))[0]
        b_bits = struct.unpack("<q", struct.pack("<d", ref_f))[0]
        ulp = abs(a_bits - b_bits)
        if isinstance(ulp, int) and ulp > max_ulp_lse:
            max_ulp_lse = ulp
    print(f"  {x!r:16s} | {got!r:22s} | {mp.nstr(ref_mp, 14):22s} | {ulp}")
print()
print(f"  Maximum ULP across test points (Chebyshev minimax): {max_ulp_lse}")
print()
print("=" * 70)
print("BIT-EXACT FINAL CONSTANTS for cranelift IR")
print("=" * 70)
print()
print(f"  inv_ln2_32 = {inv_f!r}  bits {hex_of_f64(inv_f)}")
print(f"  ln2_32_hi  = {ln2_32_hi!r}  bits {hex_of_f64(ln2_32_hi)}")
print(f"  ln2_32_lo  = {ln2_32_lo!r}  bits {hex_of_f64(ln2_32_lo)}")
print(f"  a1         = {a1_f!r}  bits {hex_of_f64(a1_f)}")
print(f"  a2         = {a2_f!r}  bits {hex_of_f64(a2_f)}")
print(f"  a3         = {a3_f!r}  bits {hex_of_f64(a3_f)}")
print(f"  a4         = {a4_f!r}  bits {hex_of_f64(a4_f)}")
