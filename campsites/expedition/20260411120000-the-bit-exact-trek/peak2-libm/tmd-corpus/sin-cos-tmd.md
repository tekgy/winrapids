# TMD Corpus — `tam_sin`, `tam_cos`

**Owner:** Adversarial Mathematician (I9′ oracle registry)
**Date:** 2026-04-12
**Status:** Initial seed.

---

## Known TMD candidates for `sin` and `cos`

### Case 1: `sin(π/4) = cos(π/4) = 1/sqrt(2)`

True value: `1/sqrt(2) = 0.707106781186547524400844362104849039284...`

fp64 neighbors:
- `0x3FE6A09E667F3BCC` = `0.70710678118654746...`
- `0x3FE6A09E667F3BCD` = `0.70710678118654757...`

Midpoint: `0.70710678118654752...`. True value ends in `...524...`, which is slightly below the midpoint at `...752...`. Wait: `...524...` vs midpoint `...752...` — the true value is BELOW the midpoint, so correct rounding is to the lower neighbor `0x3FE6A09E667F3BCC`.

BUT: the input `π/4` is not exactly representable in fp64. The fp64 constant `π/4 ≈ 0.7853981633974483...` differs from the true `π/4` by a fraction of a ULP. So `sin(fp64(π/4))` computes `sin(0.7853981633974483...)`, not exactly `sin(π/4)`. The true value of `sin(fp64(π/4))` differs from `1/sqrt(2)` by a fraction of a ULP.

**The correct test:** verify `tam_sin(fp64_pi_over_4)` against `mpmath.sin(mpmath.mpf('0.7853981633974483096...'))` at 200 digits.

### Case 2: `sin(1.0)`

True value: `sin(1) = 0.841470984807896506652502321630298999621...`

fp64 neighbors:
- `0x3FEAED548F090CEE` = `0.84147098480789650...`
- `0x3FEAED548F090CEF` = `0.84147098480789661...`

The true value `...506...` vs fp64 boundary `...650...`: the true value is very close to the lower fp64 neighbor. This is a known near-TMD case for `sin(1.0)`. The error from the midpoint is about `0.44 ULP` — not a full TMD candidate but a hard rounding case.

**Test:** `assert tam_sin(1.0).to_bits() == 0x3FEAED548F090CEEu64`

### Case 3: `cos(1.0)`

True value: `cos(1) = 0.540302305868139717400936607442976603732...`

fp64 representation: `0x3FE14A280FB5068B` = `0.54030230586813966...`

The true value `...717...` vs fp64 `...966...` — the difference is at the 17th decimal digit. Not a tight TMD case. Test that `tam_cos(1.0)` is correctly rounded.

### Case 4: Small argument — `sin(2^{-27})`

For `x = 2^{-27}`, the polynomial `r + r * r^2 * S(r^2)` gives:
- Leading term: `r = 2^{-27}`
- Cubic correction: `r^3 / 6 = 2^{-81} / 6 ≈ 3.6 × 10^{-25}`

The cubic correction is `2^{-81}/6 ≈ 2^{-84}`, which is far below `ulp(2^{-27}) = 2^{-79}`. So the result is `2^{-27}` exactly (the correction is below the ULP threshold).

**Test:** `assert tam_sin(f64::from_bits(0x3E40000000000000u64)) == f64::from_bits(0x3E40000000000000u64)` (i.e., `sin(2^{-27}) = 2^{-27}` bit-exact)

### Case 5: Cody-Waite boundary inputs for range reduction

For the quadrant dispatch, the range-reduction step uses `k = round(x / (π/2))` and `r = x - k * (π/2)_hi - k * (π/2)_lo`. The hardest inputs for range reduction are those where `x` is very close to `k * π/2` for integer `k`.

Specifically: for `x = k * pi_over_2_hi` (the constant used in range reduction), the residual after the first subtraction step is `r_1 = x - k * pi_over_2_hi = 0` exactly (exact subtraction since `x = k * pi_over_2_hi` exactly). Then `r = r_1 - k * pi_over_2_lo = -k * pi_over_2_lo`. This is a predictable, controllable scenario.

**Test battery for boundary:** For `k ∈ {1, 2, 3, 4, 100, 1000, 2^19, 2^19 + 1}`:
- `x = k * pi_over_2_hi` (exact, representable since `pi_over_2_hi` has trailing zeros)
- Compute `tam_sin(x)` and `tam_cos(x)`
- Verify against `mpmath.sin(k * pi_over_2_hi_as_mpf)` at 200 digits

### Case 6: Quadrant boundary — `sin(π/2)`

`π/2` is not exactly representable. `fp64(π/2) = 0x3FF921FB54442D18 = 1.5707963267948966...`. True `sin(π/2) = 1.0` exactly. But `sin(fp64(π/2))` ≠ `sin(π/2)`.

True value: `sin(1.5707963267948966...) = 1.0 - (fp64(π/2) - π/2)^2/2 + ...`. The error in `fp64(π/2)` from the true `π/2` is about `6.1 × 10^{-17}`, so `sin(fp64(π/2)) ≈ 1 - (6.1e-17)^2/2 ≈ 1 - 1.86 × 10^{-33}`. This is so close to `1.0` that the rounded result IS `1.0`.

**Test:** `assert tam_sin(fp64_pi_over_2) == 1.0` — this should hold bit-exact.

---

## Reference computation

```python
import mpmath
mpmath.mp.dps = 100

test_inputs = [
    1.0,
    0.7853981633974483,  # fp64(π/4)
    1.5707963267948966,  # fp64(π/2)
    2**-27,
]
for x in test_inputs:
    sx = mpmath.sin(mpmath.mpf(str(x)))
    cx = mpmath.cos(mpmath.mpf(str(x)))
    print(f"sin({x}) = {sx}")
    print(f"cos({x}) = {cx}")
```

---

## Expansion protocol

When pathmaker discovers additional hard cases during implementation, log them here with:
1. The input `x` as a hex fp64 bit pattern
2. The true value from mpmath at 200+ digits (for sin and cos both, if relevant)
3. The correct rounded fp64 bit pattern
4. The incorrect value the implementation first produced
