# TMD Corpus — `tam_ln`

**Owner:** Adversarial Mathematician (I9′ oracle registry)
**Date:** 2026-04-12
**Status:** Initial seed.

---

## Known TMD candidates for `log` (natural logarithm)

### Case 1: `log(1.5)`

True value: `ln(3/2) = 0.405465108108164381978013115464349137526...`

This is a well-known TMD candidate for `log`. The true value is very close to the midpoint between two adjacent fp64 values. The correct rounding direction requires more than 53 bits of computation to determine.

fp64 neighbors (approximate):
- `0x3FD9F323ECBF984B`
- `0x3FD9F323ECBF984C`

**Verification:** `import mpmath; mpmath.mp.dps = 100; print(mpmath.log(mpmath.mpf('1.5')))`

### Case 2: `log(e)` where `e` is fp64's approximation

The mathematical identity `log(e) = 1` is exact. But `e` is not exactly representable in fp64. The fp64 representation of `e`:
- `fp64(e) = 0x4005BF0A8B145769` = `2.71828182845904509...`

True `log(fp64(e)) = log(2.71828182845904509...)`. This is NOT equal to 1.0 — it is slightly less than 1.0.

True value: `log(2.71828182845904509...) = 0.9999999999999997779553950749686919152736...`

The nearest fp64 to this value is `0x3FEFFFFFFFFFFFFF` = `0.9999999999999997779553950...` (the largest fp64 below 1.0).

**Test:** `assert tam_ln(fp64(e)).to_bits() == 0x3FEFFFFFFFFFFFFFu64`

Note: `tam_ln(fp64(e))` must NOT return `1.0`. It returns the largest fp64 below 1.0.

### Case 3: `log(1.0 + 2^{-52})`

The input is `1.0 + 2^{-52}` = `0x3FF0000000000001` (the smallest fp64 above 1.0).

True value: `log(1 + 2^{-52}) = 2^{-52} - 2^{-105}/2 + ...`

The leading term is `2^{-52}` exactly. The next term `2^{-105}/2` is far below fp64 precision (which has resolution `2^{-52}` in the neighborhood of `2^{-52}`). So the correctly-rounded result is exactly `2^{-52}` (= `0x3CB0000000000000`).

**Test:** `assert tam_ln(1.0 + f64::EPSILON).to_bits() == (f64::EPSILON).to_bits()`

This tests that: (1) the range reduction extracts `m = 1 + 2^{-52}` with `e = 0`; (2) `f = m - 1 = 2^{-52}` exactly (Sterbenz); (3) the polynomial gives `f + f^2 * Q(f) ≈ f` since `f^2 * Q(f) < ulp(f)`; (4) the reassembly with `e = 0` just returns `f`.

### Case 4: `log(2.0)`

True value: `ln(2) = 0.693147180559945309417232121458176568075...`

fp64 representation of `ln(2)`: `0x3FE62E42FEFA39EF` = `0.6931471805599452862...`

Is this exactly correctly-rounded? The true value ends in `...568075...`, and the fp64 value ends in `...452862...`. The difference is in the 16th decimal digit. Since fp64 has ~15.9 significant decimal digits, we're at the precision boundary. The Cody-Waite constants `ln2_hi + ln2_lo` are designed so that `ln2_hi + ln2_lo` represents `ln(2)` to ~106 bits — but the fp64 constant alone is only accurate to 53 bits.

**Test:** This is a fundamental constant, not a TMD case per se. But: verify `tam_ln(2.0).to_bits() == 0x3FE62E42FEFA39EFu64`.

### Case 5: Sterbenz exactness test for `f = m - 1`

Per adversarial-review-log.md A1: the subtraction `f = m - 1` is exact by Sterbenz when `m ∈ [sqrt(2)/2, sqrt(2)) ≈ [0.707, 1.414)`. Verify this for specific `m` values:

| `m` | `f = m - 1` | Should be exact |
|-----|-------------|-----------------|
| `0.707` (≈ `sqrt(2)/2`) | `≈ -0.293` | Yes (Sterbenz: `0.707/2 = 0.354 > 0.293`) |
| `0.9` | `-0.1` | Yes |
| `1.0` | `0.0` | Yes (trivially) |
| `1.1` | `0.1` | Yes |
| `1.414` (≈ `sqrt(2)`) | `≈ 0.414` | Yes (`0.707 > 0.414`) |

For each of these: `assert (m as f64) - 1.0f64` has zero rounding error (i.e., the true value `m - 1` is exactly representable in fp64 after the subtraction, assuming `m` is computed exactly).

---

## Reference computation

```python
import mpmath
mpmath.mp.dps = 100

test_inputs = [1.5, 2.718281828459045, 1.0 + 2**-52, 2.0, 0.5, 0.1]
for x in test_inputs:
    result = mpmath.log(mpmath.mpf(str(x)))
    print(f"log({x}) = {result}")
```

For bit-exact verification of the Sterbenz tests, mpmath is not needed — these are exact arithmetic claims that can be verified by computing in integer arithmetic at the bit level.

---

## Expansion protocol

When pathmaker discovers additional hard cases during implementation, log them here with:
1. The input `x` as a hex fp64 bit pattern
2. The true value from mpmath at 200+ digits
3. The correct rounded fp64 bit pattern
4. The incorrect value the implementation first produced
