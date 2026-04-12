# TMD Corpus — `tam_atan`, `tam_asin`, `tam_acos`, `tam_atan2`

**Owner:** Adversarial Mathematician (I9′ oracle registry)
**Date:** 2026-04-12
**Status:** Initial seed.

---

## Known TMD candidates for inverse trig functions

### Case 1: `atan(1.0)` = `π/4`

True value: `π/4 = 0.785398163397448309615660845819875721049...`

fp64 representation of `π/4`: `0x3FE921FB54442D18` = `0.7853981633974483096...`

The question: is this the correctly-rounded fp64? The true `π/4` ends in `...309615...` and the fp64 value encodes `...3096...`. The 17th significant decimal digit of the fp64 is `6`; the true value is `...09615...`. The rounding point is at the 16th decimal digit.

In binary: `π/4` in binary is `0.11001001000011111101101010100010001000010110100011...`. The first 53 bits determine the fp64 mantissa. This is a known exact value; the fp64 representation `0x3FE921FB54442D18` is the correctly-rounded result.

**Test:** `assert tam_atan(1.0).to_bits() == 0x3FE921FB54442D18u64`

### Case 2: `atan(tan(π/8 + ε))` — shift-boundary input

The range reduction uses a shift for `|x| > tan(π/8) ≈ 0.4142`. For `x` slightly above `tan(π/8)`:
- `x_final = (x - 1)/(x + 1)` where `x ≈ 0.4142 + ε` → `x_final ≈ -0.6 + small`
- Wait, the shift formula is for `x > tan(π/8)`: `x_final = (x - 1)/(x + 1)`. For `x = tan(π/8) + ε ≈ 0.4143`, `x_final = (0.4143 - 1)/(0.4143 + 1) = -0.5857/1.4143 ≈ -0.414`.

This tests the shift boundary precision. Both branches (with and without shift) should agree within 1 ULP for inputs near `x = tan(π/8)`.

**Test:** For `x = tan(π/8)` ≈ `0.4142135623730950`:
- Compute `tam_atan(x - ε)` (no-shift branch) and `tam_atan(x + ε)` (shift branch)
- Both must be within 1 ULP of `mpmath.atan(0.4142135623730950...)` at 200 digits

### Case 3: `atan(1e15)` — near `π/2`

For very large `x`, `atan(x) → π/2`. For `x = 10^{15}`:
- `atan(10^{15}) = π/2 - atan(10^{-15}) ≈ π/2 - 10^{-15}`
- `π/2 - 10^{-15} = 1.5707963267948965705...` (π/2 ≈ 1.5707963267948966...)

The result is very close to `π/2`. The precision of the Cody-Waite reassembly is tested here.

**Test:** `assert |tam_atan(1e15) - mpmath.atan(1e15)| < 2 * ulp(π/2f64)`

### Case 4: `asin(0.5)` = `π/6`

True value: `π/6 = 0.523598775598298873077107230546583814032...`

fp64 representation of `π/6`: `0x3FE0C152382D7365` = `0.5235987755982988...`

Known TMD candidate: the last few bits of the true `π/6` are close to the rounding boundary of fp64.

**Test:** Verify `tam_asin(0.5)` against `mpmath.asin(0.5)` at 200 digits.

### Case 5: `acos(0.5)` = `π/3`

True value: `π/3 = 1.047197551196597746154214461093167628065...`

fp64 representation of `π/3`: `0x3FF0C152382D7366` = `1.0471975511965977...`

**Test:** Verify `tam_acos(0.5)` against `mpmath.acos(0.5)` at 200 digits.

### Case 6: `asin` near `|x| = 1` — precision-stressed inputs

For `x = 1 - 2^{-52}` (the largest fp64 below 1), `asin(x)` is very close to `π/2`. Per adversarial-review-atan.md A1, the composed error from the factored-sqrt formula may exceed 1 ULP here.

True value: `asin(1 - 2^{-52}) ≈ π/2 - sqrt(2^{-51}) ≈ π/2 - 2^{-25.5}`.

**Test:** Measure the actual ULP error for `tam_asin(1.0 - f64::EPSILON)` against mpmath at 200 digits. Document in this file whether it is 1 ULP or 2 ULP. This determines whether a near-unity special case is needed.

### Case 7: `atan2(1.0, 1.0)` = `atan(1.0)` = `π/4`

`atan2(1.0, 1.0) = atan(1.0/1.0) = atan(1.0) = π/4`. The division `1.0 / 1.0 = 1.0` is exact. So `atan2(1.0, 1.0)` should equal `atan(1.0)` bit-exact.

**Test:** `assert tam_atan2(1.0, 1.0).to_bits() == tam_atan(1.0).to_bits()`

### Case 8: Signed-zero atan2 cases (bit-exact)

All four cases where both arguments are signed zeros must be bit-exact per IEEE 754:

| Input | Expected bit pattern |
|-------|---------------------|
| `atan2(+0.0, +0.0)` | `0x0000000000000000` (+0.0) |
| `atan2(-0.0, +0.0)` | `0x8000000000000000` (-0.0) |
| `atan2(+0.0, -0.0)` | `0x400921FB54442D18` (+π) |
| `atan2(-0.0, -0.0)` | `0xC00921FB54442D18` (-π) |

The π values are: `fp64(π) = 0x400921FB54442D18` and `fp64(-π) = 0xC00921FB54442D18`.

---

## Reference computation

```python
import mpmath
mpmath.mp.dps = 100

test_cases = [
    ('atan', 1.0),
    ('atan', 0.4142135623730950),
    ('atan', 1e15),
    ('asin', 0.5),
    ('acos', 0.5),
    ('asin', 1.0 - 2**-52),
]
for fname, x in test_cases:
    f = getattr(mpmath, fname)
    result = f(mpmath.mpf(str(x)))
    print(f"{fname}({x}) = {result}")
```

---

## Expansion protocol

When pathmaker discovers additional hard cases during implementation, log them here with:
1. The input as hex fp64 bit pattern(s)
2. The true value from mpmath at 200+ digits
3. The correct rounded fp64 bit pattern
4. The incorrect value the implementation first produced
