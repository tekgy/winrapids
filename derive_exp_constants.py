"""Derive Tang 1989 exp constants for tambear's exp_for_lse() Cranelift IR."""
import math
import struct
from mpmath import mp, mpf, exp, log, fac

mp.dps = 100

LN2 = log(2)
LN2_OVER_32 = LN2 / 32
LN2_OVER_64 = LN2 / 64
INV_LN2_OVER_32 = 32 / LN2


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


print("=" * 70)
print("TANG 1989 ARGUMENT REDUCTION CONSTANTS")
print("=" * 70)
print(f"  ln(2)             = {mp.nstr(LN2, 30)}")
print(f"  ln(2)/32          = {mp.nstr(LN2_OVER_32, 30)}")
print(f"  ln(2)/64 (|r| <=) = {mp.nstr(LN2_OVER_64, 30)}")
print(f"  32/ln(2)          = {mp.nstr(INV_LN2_OVER_32, 30)}")
print()

ln2_32_hi, ln2_32_lo = cw_split(LN2_OVER_32, low_bits_zero=20)
inv_f = float(INV_LN2_OVER_32)
print("Cody-Waite split of ln(2)/32 (low 20 mantissa bits zeroed in hi):")
print(f"  ln2_32_hi = {ln2_32_hi!r:30s}  bits {hex_of_f64(ln2_32_hi)}")
print(f"  ln2_32_lo = {ln2_32_lo!r:30s}  bits {hex_of_f64(ln2_32_lo)}")
print(f"  hi+lo (mp) = {mp.nstr(mpf(repr(ln2_32_hi)) + mpf(repr(ln2_32_lo)), 30)}")
print(f"  ln(2)/32   = {mp.nstr(LN2_OVER_32, 30)}")
print(f"  inv_ln2_32 = {inv_f!r:30s}  bits {hex_of_f64(inv_f)}")
print()

a1 = mpf(1) / 2
a2 = mpf(1) / 6
a3 = mpf(1) / 24
a4 = mpf(1) / 120

print("=" * 70)
print("POLYNOMIAL — exp(r) ~ 1 + r + a1*r^2 + a2*r^3 + a3*r^4 + a4*r^5")
print("=" * 70)
print(f"  a1 = 1/2!  = {float(a1)!r:25s}  bits {hex_of_f64(a1)}")
print(f"  a2 = 1/3!  = {float(a2)!r:25s}  bits {hex_of_f64(a2)}")
print(f"  a3 = 1/4!  = {float(a3)!r:25s}  bits {hex_of_f64(a3)}")
print(f"  a4 = 1/5!  = {float(a4)!r:25s}  bits {hex_of_f64(a4)}")
trunc_err = LN2_OVER_64 ** 6 / fac(6)
print(f"  Worst-case truncation error: {mp.nstr(trunc_err, 4)} (~ {int(trunc_err * (1 << 52))} ULP near 1)")
print()

print("=" * 70)
print("TABLE T[j] = 2^(j/32) for j=0..31  (256-byte data section)")
print("=" * 70)
print()
table_entries = []
for j in range(32):
    t_mp = mpf(2) ** (mpf(j) / 32)
    t_f = float(t_mp)
    bits = hex_of_f64(t_f)
    table_entries.append((j, t_f, bits))
    print(f"  j={j:2d}  T = {t_f!r:30s}  bits {bits}")

print()
print("Rust array literal:")
print()
print("static T_TABLE: [f64; 32] = [")
for j, t_f, _ in table_entries:
    print(f"    {t_f!r},  // 2^({j}/32)")
print("];")
print()

print("=" * 70)
print("EDGE CASE CONSTANTS — for x <= 0 domain (LSE constraint)")
print("=" * 70)
THRESH_DENORMAL = -1022 * LN2
THRESH_ZERO = log(mpf(2) ** (-1074))
print(f"  Underflow-to-denormal: {float(THRESH_DENORMAL)!r}  bits {hex_of_f64(float(THRESH_DENORMAL))}")
print(f"  Underflow-to-zero:     {float(THRESH_ZERO)!r}  bits {hex_of_f64(float(THRESH_ZERO))}")
print()


def tang_exp_for_lse(x):
    n = round(x * inv_f)
    r = (x - n * ln2_32_hi) - n * ln2_32_lo
    poly = ((float(a4) * r + float(a3)) * r + float(a2)) * r + float(a1)
    poly = poly * (r * r)
    exp_r = 1.0 + r + poly
    j = n % 32
    if j < 0:
        j += 32
    k = (n - j) // 32
    t_j = table_entries[j][1]
    return math.ldexp(t_j * exp_r, k)


print("=" * 70)
print("VERIFICATION — Tang impl with derived constants vs mpmath")
print("=" * 70)
print()
test_xs = [
    0.0, -1e-15, -1e-10, -1e-5, -float(LN2_OVER_64) * 0.99,
    -float(LN2_OVER_32) * 0.5, -0.5, -1.0, -2.5, -5.0, -10.0,
    -20.0, -50.0, -100.0, -300.0, -700.0,
    float(THRESH_DENORMAL) - 1.0,
    float(THRESH_ZERO) + 1.0,
    float(THRESH_ZERO) - 1.0,
]
print("  test x                  | tang result            | mpmath exp(x)          | ULP")
print("  ------------------------+------------------------+------------------------+----")
max_ulp = 0
for x in test_xs:
    got = tang_exp_for_lse(x)
    ref_mp = exp(mpf(repr(x)))
    ref_f = float(ref_mp)
    if got == ref_f:
        ulp = 0
    elif ref_f == 0.0 and got == 0.0:
        ulp = 0
    elif ref_f == 0.0:
        ulp = "inf"
    else:
        a_bits = struct.unpack("<q", struct.pack("<d", got))[0]
        b_bits = struct.unpack("<q", struct.pack("<d", ref_f))[0]
        ulp = abs(a_bits - b_bits)
        if isinstance(ulp, int) and ulp > max_ulp:
            max_ulp = ulp
    print(f"  {x!r:23s} | {got!r:22s} | {mp.nstr(ref_mp, 14):22s} | {ulp}")
print()
print(f"  Maximum ULP across test points: {max_ulp}")
print("  Target: <= 1 ULP per Tang 1989 claims for reduced range.")
