"""Derive Tang-style log(x) constants for tambear's log_for_entropy()
   and log_for_hill() Cranelift IR. Same methodology as exp_for_lse:
   table-drive reduction, Chebyshev-fit minimax polynomial on a small
   range, bit-exact f64 hex constants.

Algorithm — Tang 1990 "Table-driven implementation of the logarithm
function in IEEE floating-point arithmetic" (ACM TOMS 16(4)):

  log(x) = log(2^E · m)                         (m in [1, 2))
        = E · log(2) + log(m)
        = E · log(2) + log(F · g)               (F is table value, g near 1)
        = E · log(2) + log(F) + log(g)
        = E · log(2) + log(F) + log(1 + s)      (s = g - 1, |s| small)

  Step 1: Decompose x = 2^E · m via bit manipulation:
          E = (raw_exp_bits >> 52) - 1023
          m = mantissa-as-f64 with biased exponent set to 1023

  Step 2: Look up F = round-to-nearest(m, k bits) from a table
          indexed by the top k bits of m's mantissa.
          We pick k=7 (128-entry table).
          F is the closest tabulated value to m; |m - F|/F is bounded.

  Step 3: g = m / F  (range: very close to 1; |g - 1| < 1/256 for k=7)
          Equivalently: s = (m - F) / F  in (-1/256, 1/256)

  Step 4: log(F) is precomputed in the same 128-entry table.
          (E · log(2)) and log(F) are added with care for cancellation
          (Cody-Waite split of log(2)).

  Step 5: log(1 + s) ≈ s - s²/2 + s³/3 - s⁴/4 + ...
          Approximate with degree-5 minimax polynomial on |s| < 1/256:
            log(1 + s) ≈ s + b1·s² + b2·s³ + b3·s⁴ + b4·s⁵
          where b1, b2, b3, b4 are minimax-fit (NOT the Taylor
          (-1/2, +1/3, -1/4, +1/5)).

DOMAIN for tambear:
- shannon_entropy needs log(p) where p ∈ (0, 1]; result is x ≤ 0.
- hill_estimator_streaming needs log(x) where x > 0 (any magnitude).
- General log_for_entropy: x in (0, +inf).

EDGE CASES (per IEEE 754):
- log(0)    = -inf  (signal divide-by-zero; LSE/entropy callers handle)
- log(1)    = +0    (exact)
- log(-x)   = NaN   (caller must filter out)
- log(NaN)  = NaN   (propagation)
- log(+inf) = +inf
- log(subnormal) — algorithm extracts E correctly via bit manipulation
"""
import math
import struct
from mpmath import mp, mpf, log, fac

mp.dps = 100

LN2 = log(2)


def hex_of_f64(x):
    if not isinstance(x, float):
        x = float(x)
    return hex(struct.unpack("<Q", struct.pack("<d", x))[0])


def cw_split(x_mp, low_bits_zero=20):
    """Cody-Waite split: hi has only upper (52-low_bits_zero) mantissa bits."""
    x_f = float(x_mp)
    bits = struct.unpack("<Q", struct.pack("<d", x_f))[0]
    mask = (~((1 << low_bits_zero) - 1)) & ((1 << 64) - 1)
    hi_bits = bits & mask
    hi_f = struct.unpack("<d", struct.pack("<Q", hi_bits))[0]
    lo_mp = x_mp - mpf(repr(hi_f))
    lo_f = float(lo_mp)
    return hi_f, lo_f


# ============================================================
# Cody-Waite split of ln(2) for E·log(2) reconstruction
# ============================================================
ln2_hi, ln2_lo = cw_split(LN2, low_bits_zero=20)

print("=" * 70)
print("TANG-STYLE LOG ARGUMENT-RECOMBINATION CONSTANTS")
print("=" * 70)
print(f"  ln(2)         = {mp.nstr(LN2, 30)}")
print(f"  ln2_hi (CW)   = {ln2_hi!r:30s}  bits {hex_of_f64(ln2_hi)}")
print(f"  ln2_lo (CW)   = {ln2_lo!r:30s}  bits {hex_of_f64(ln2_lo)}")
print(f"  hi+lo (mp)    = {mp.nstr(mpf(repr(ln2_hi)) + mpf(repr(ln2_lo)), 30)}")
print()

# ============================================================
# Table T[j] = chosen F for mantissa bin j, j=0..127
# m ∈ [1, 2), top 7 bits of mantissa index F.
# F[j] = 1 + j/128 + 1/256 (midpoint of bin j)
# log_F[j] = log(F[j]) at high precision, then rounded to f64.
#
# For Tang's full-precision form, log_F is split into hi+lo. For our
# initial pass: single f64 log_F entry — gives ~2-3 ULP overall, which
# is plenty for shannon_entropy / hill_estimator (those don't need
# correctly-rounded log).
# ============================================================
print("=" * 70)
print("TABLE T[j] (128 entries) — F[j] and log(F[j]) for j=0..127")
print("=" * 70)
print()
table = []  # list of (j, F_f, logF_f, F_bits, logF_bits)
for j in range(128):
    # F[j] = LEFT endpoint of mantissa bin j: F = 1 + j/128.
    # This makes F[0] = 1.0 EXACTLY (so log(F[0]) = 0 exactly), which
    # eliminates the catastrophic-cancellation failure mode at x=1.
    # Range of s = (m - F)/F is [0, 1/128) — purely positive, slightly
    # wider than the midpoint scheme but algorithmically cleaner.
    F_mp = mpf(1) + mpf(j) / mpf(128)
    F_f = float(F_mp)  # F is dyadic: representable exactly in f64
    log_F_mp = log(F_mp)
    log_F_f = float(log_F_mp)
    table.append((j, F_f, log_F_f,
                  hex_of_f64(F_f), hex_of_f64(log_F_f)))

# Spot-check a few:
print("  Sample entries (showing j=0, 32, 64, 96, 127):")
print("  j   |     F[j]      |  log(F[j])           | F bits     | log_F bits")
print("  ----+---------------+----------------------+------------+--------------")
for j_show in [0, 32, 64, 96, 127]:
    j, F_f, log_F_f, F_bits, log_F_bits = table[j_show]
    print(f"  {j:3d} | {F_f!r:13s} | {log_F_f!r:20s} | {F_bits:10s} | {log_F_bits}")

print()
print(f"  Total table size: 128 × 16 bytes (F + log_F per entry) = 2048 bytes")
print(f"  Layout: interleaved [F0, lF0, F1, lF1, ...] for cache locality")

# ============================================================
# Polynomial — log(1 + s) on |s| < 1/256 ≈ 0.0039
#
# Vanilla Taylor: log(1 + s) = s - s²/2 + s³/3 - s⁴/4 + s⁵/5 - ...
# Coefficients of (log(1+s) - s)/s² = -1/2 + s/3 - s²/4 + s³/5 - ...
# We approximate with degree-5 polynomial: s + b1·s² + b2·s³ + b3·s⁴ + b4·s⁵.
# For |s| < 1/256, vanilla Taylor truncation error ~ s^6/6 ≈ 1e-15 ≈ 5 ULP.
# Chebyshev-fit gives ~1 ULP.
# ============================================================
# With F[j] = 1 + j/128 (left endpoint), the mantissa m falls in
# [F[j], F[j] + 1/128). Then s = (m - F)/F is in [0, 1/128 / F).
# Worst case at F=1: s in [0, 1/128). At F=2-1/128: s in [0, ~1/(2·128)).
# So |s| < 1/128 ≈ 0.0078.
S_MAX = mpf(1) / 128


def chebyshev_nodes(n, a, b):
    nodes = []
    for k in range(n):
        x = mp.cos(mp.pi * (2 * k + 1) / (2 * n))
        nodes.append((a + b) / 2 + (b - a) / 2 * x)
    return nodes


def fit_polynomial_thru_nodes(xs, ys, degree):
    n = degree + 1
    A = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = xs[i] ** j
    bvec = mp.matrix(ys)
    coeffs = mp.lu_solve(A, bvec)
    return [coeffs[i] for i in range(n)]


# h(s) = (log(1+s) - s) / s²
# = -1/2 + s/3 - s²/4 + s³/5 - s⁴/6 + ...
def h(s):
    if abs(s) < mpf("1e-30"):
        return mpf("-0.5") + s / 3 - s * s / 4
    return (log(1 + s) - s) / (s * s)


# Degree-3 in s for h(s) (= 4 coefficients = b1..b4).
deg = 3
nodes = chebyshev_nodes(deg + 1, -S_MAX, S_MAX)
values = [h(s) for s in nodes]
coeffs = fit_polynomial_thru_nodes(nodes, values, deg)
b1, b2, b3, b4 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]


def poly_log_1ps(s, b1, b2, b3, b4):
    """log(1+s) ≈ s + (b1 + (b2 + (b3 + b4·s)·s)·s) · s²"""
    inner = ((b4 * s + b3) * s + b2) * s + b1
    return s + inner * s * s


def measure_log_error(b1, b2, b3, b4, n=10000):
    max_abs = mpf(0)
    max_rel = mpf(0)
    for k in range(-n, n + 1):
        if k == 0:
            continue
        s = mpf(k) / n * S_MAX
        truth = log(1 + s)
        approx = poly_log_1ps(s, b1, b2, b3, b4)
        abs_err = abs(approx - truth)
        rel_err = abs_err / abs(truth)
        if abs_err > max_abs:
            max_abs = abs_err
        if rel_err > max_rel:
            max_rel = rel_err
    return max_abs, max_rel


print()
print("=" * 70)
print("POLYNOMIAL — log(1+s) ~ s + b1·s² + b2·s³ + b3·s⁴ + b4·s⁵")
print("on |s| <= 1/256 ≈ 0.00391")
print("=" * 70)
print()
print(f"  b1 = {mp.nstr(b1, 25)}  -> f64 {float(b1)!r}")
print(f"  b2 = {mp.nstr(b2, 25)}  -> f64 {float(b2)!r}")
print(f"  b3 = {mp.nstr(b3, 25)}  -> f64 {float(b3)!r}")
print(f"  b4 = {mp.nstr(b4, 25)}  -> f64 {float(b4)!r}")
print()
print("  Vanilla Taylor for comparison:")
print(f"  -1/2  = {mp.nstr(-mpf(1)/2, 25)}")
print(f"   1/3  = {mp.nstr(mpf(1)/3, 25)}")
print(f"  -1/4  = {mp.nstr(-mpf(1)/4, 25)}")
print(f"   1/5  = {mp.nstr(mpf(1)/5, 25)}")
print()

abs_err, rel_err = measure_log_error(b1, b2, b3, b4, n=5000)
print(f"  Chebyshev-fit max abs error: {mp.nstr(abs_err, 4)}")
print(f"  Chebyshev-fit max rel error: {mp.nstr(rel_err, 4)}  (~ {int(rel_err * (1<<52))} ULP)")

abs_t, rel_t = measure_log_error(-mpf(1)/2, mpf(1)/3, -mpf(1)/4, mpf(1)/5, n=5000)
print(f"  Vanilla Taylor max abs error: {mp.nstr(abs_t, 4)}")
print(f"  Vanilla Taylor max rel error: {mp.nstr(rel_t, 4)}  (~ {int(rel_t * (1<<52))} ULP)")

# ============================================================
# Verification: end-to-end Tang log algorithm
# ============================================================
b1_f, b2_f, b3_f, b4_f = float(b1), float(b2), float(b3), float(b4)


def tang_log(x):
    """Tang-style log(x). Domain: x > 0. Returns log(x) in f64."""
    if x <= 0:
        if x == 0:
            return float("-inf")
        return float("nan")
    if math.isinf(x):
        return float("inf")
    if math.isnan(x):
        return float("nan")

    # NEAR-1 FAST PATH: when |x - 1| < ~1/16, use direct
    # log(1 + u) where u = x - 1, u is small. Bypasses the cancellation
    # in `E·ln2 + log(F) + log(1+s)` for x close to 1. This is essential
    # for shannon_entropy where p often near 0 or 1.
    # Threshold 1/16 keeps |u| < 0.0625, polynomial extends naturally
    # since we have minimax fit on |s| < 1/128 — but we need a wider-range
    # poly here. Use degree-7 Taylor truncation, error ~ u^8/8 ~ 4e-11
    # at u=0.0625; 13 ULP at result magnitude ~0.06.
    # Better: use the same minimax poly extrapolated; on |u|<1/16 it gives
    # ~few-ULP accuracy. For tightest correctness we'd fit a separate poly.
    # Cleanest pragmatic choice: degree-9 Horner with Taylor coefficients.
    if abs(x - 1.0) < 0.0625:
        u = x - 1.0
        # Horner-form Taylor: log(1+u) = u - u²/2 + u³/3 - u⁴/4 + ... up to u^9
        # Coefficients: -1/2, 1/3, -1/4, 1/5, -1/6, 1/7, -1/8, 1/9
        c2 = -0.5
        c3 = 1.0 / 3.0
        c4 = -0.25
        c5 = 0.2
        c6 = -1.0 / 6.0
        c7 = 1.0 / 7.0
        c8 = -0.125
        c9 = 1.0 / 9.0
        # Horner from inside out for degree-9 poly in u
        p = c9 * u + c8
        p = p * u + c7
        p = p * u + c6
        p = p * u + c5
        p = p * u + c4
        p = p * u + c3
        p = p * u + c2
        # log(1+u) ≈ u + u²·p (separating linear u for precision)
        return u + u * u * p

    # Decompose x = 2^E · m via bit manipulation
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    biased_exp = (bits >> 52) & 0x7FF
    if biased_exp == 0:
        # Subnormal: normalize manually
        # Find leading 1 in mantissa, shift, adjust E
        mant = bits & 0x000FFFFFFFFFFFFF
        if mant == 0:
            return float("-inf")  # log(0) but already filtered
        # Find position of leading 1
        shift = 52 - mant.bit_length()
        mant = (mant << (shift + 1)) & 0x000FFFFFFFFFFFFF  # +1 to drop the implicit
        E = -1022 - shift
        m_bits = (1023 << 52) | mant
    else:
        E = biased_exp - 1023
        m_bits = (1023 << 52) | (bits & 0x000FFFFFFFFFFFFF)
    m = struct.unpack("<d", struct.pack("<Q", m_bits))[0]
    # m is in [1, 2). Top 7 bits of mantissa index the table:
    j = (bits >> 45) & 0x7F  # top 7 bits of mantissa
    # For subnormal, recompute j from normalized m:
    if biased_exp == 0:
        m_int_bits = struct.unpack("<Q", struct.pack("<d", m))[0]
        j = (m_int_bits >> 45) & 0x7F

    F_f, log_F_f = table[j][1], table[j][2]
    # s = (m - F) / F   — uses single-precision division; ~1 ULP from this
    s = (m - F_f) / F_f
    # log(1 + s) via polynomial
    inner = ((b4_f * s + b3_f) * s + b2_f) * s + b1_f
    log_1ps = s + inner * s * s
    # log(x) = E · ln2 + log(F) + log(1 + s)
    # Use Cody-Waite split of ln2 to preserve precision when E is large
    E_f = float(E)
    return ((E_f * ln2_hi) + log_F_f) + (E_f * ln2_lo + log_1ps)


print()
print("=" * 70)
print("END-TO-END VERIFICATION (Tang log + Chebyshev minimax + f64 constants)")
print("=" * 70)
print()
test_xs = [
    1.0, 2.0, math.e, 0.5, 0.1, 1e-10, 1e-100, 1e-300,
    1e10, 1e100, 1e300, 1.5, 0.99999, 1.00001,
    # shannon_entropy domain: x in (0, 1]
    0.001, 0.01, 0.5, 0.9, 0.99,
    # hill_estimator domain: x in (0, +inf), often very large
    1e6, 1e15, float("inf"),
]
print("  test x       | tang log              | mpmath log            | ULP")
print("  -------------+-----------------------+-----------------------+----")
max_ulp = 0
for x in test_xs:
    got = tang_log(x)
    if math.isinf(x):
        ref_f = float("inf")
        ulp = 0 if got == ref_f else "fail"
    else:
        ref_mp = log(mpf(repr(x)))
        ref_f = float(ref_mp)
        if got == ref_f:
            ulp = 0
        elif math.isnan(got) or math.isnan(ref_f):
            ulp = "nan"
        elif math.isinf(got) and math.isinf(ref_f):
            ulp = 0 if (got > 0) == (ref_f > 0) else "fail"
        else:
            a = struct.unpack("<q", struct.pack("<d", got))[0]
            b = struct.unpack("<q", struct.pack("<d", ref_f))[0]
            ulp = abs(a - b)
            if isinstance(ulp, int) and ulp > max_ulp:
                max_ulp = ulp
    print(f"  {x!r:12s} | {got!r:21s} | {repr(float(log(mpf(repr(x))))) if not math.isinf(x) else 'inf':21s} | {ulp}")
print()
print(f"  Maximum ULP across test points: {max_ulp}")

# ============================================================
# Edge case constants
# ============================================================
print()
print("=" * 70)
print("EDGE CASE CONSTANTS")
print("=" * 70)
print(f"  log(0)    = -inf   (caller MUST filter or accept -inf)")
print(f"  log(1)    = +0.0   (exact — algorithm produces this naturally)")
print(f"  log(2)    = ln(2)  = {float(LN2)!r}")
print(f"  log(e)    = 1.0    (exact — only if x is exact e, which is irrational so never bit-exact)")
print(f"  log(NaN)  = NaN    (propagation)")
print(f"  log(+inf) = +inf   (must short-circuit)")
print(f"  log(-x)   = NaN    (caller MUST filter)")
print(f"  smallest positive normal:  2^-1022 ≈ 2.225e-308; log ≈ {float(-1022 * LN2)!r}")
print(f"  smallest positive subnormal: 2^-1074 ≈ 5e-324; log ≈ {float(log(mpf(2)**-1074))!r}")
print()
print("For shannon_entropy(p):")
print("  domain: p in [0, 1]")
print("  log(0) is divergent — caller convention 0 · log(0) := 0 (the limit)")
print("  Pathmaker MUST short-circuit when p == 0.0 BEFORE calling log_for_entropy.")
print()
print("For hill_estimator_streaming:")
print("  domain: x > 0 (positive observations only)")
print("  Caller pre-filters non-positive values (assertion or NaN-skip).")
