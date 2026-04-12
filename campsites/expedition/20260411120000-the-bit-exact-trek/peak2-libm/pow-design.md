# `tam_pow` — Algorithm Design Document

**Campsite 2.17.** Design for `tam_pow(a, b) = a^b` for `f64` inputs.

**Owner:** math-researcher
**Status:** draft, awaiting navigator + pathmaker review
**Date:** 2026-04-11 (amended 2026-04-12 to relax Phase 1 bound to 2 ULP per adversarial's special-values matrix)

**Upstream dependency:** `exp-design.md`, `log-design.md`, `accuracy-target.md`.
**Downstream:** this is one of the most-consumed libm functions in statistical and ML code. Also one of the most-buggy-edge-case libm functions ever.

---

## What this function does

`pow(a, b)` returns `a` raised to the power `b`, for `a, b : f64`.

**`pow` is not a single mathematical function.** It is a union of:
- Real-valued power `a^b = exp(b * log(a))` when `a > 0`.
- Integer-valued power `a^n = a * a * ... * a` when `b` is a (representable) integer, including `a < 0` (because then `a^n` is well-defined).
- A set of conventional values for edge cases: `pow(0, 0) = 1`, `pow(-1, ±inf) = 1`, `pow(+1, nan) = 1`, `pow(0, negative) = +inf` (divide-by-zero), `pow(negative, non-integer) = nan`.

The IEEE 754 standard and the C standard specify ~20 distinct edge cases for `pow` explicitly. Most of `tam_pow`'s code is case dispatch, not math. **The design prioritizes handling the case table correctly; the real-valued path is comparatively trivial on top of `exp` and `log`.**

## Accuracy target (Phase 1 = 2 ULP, amended 2026-04-12)

**Phase 1 bound: `max_ulp ≤ 2.0`** on 1M random samples from `a ∈ [2^-100, 2^100]` exponent-uniform, `b ∈ [-30, 30]` real-uniform.

This is a **relaxation** from the earlier draft's 1 ULP target. Adversarial's special-values matrix (committed 77f886c) pragmatically assigns `pow = 2 ULP` because the composed exp+log+multiply error is ~2 ULP worst case without double-double intermediates, and building those intermediates inside pow alone is disproportionate Phase 1 complexity.

**The composition math that sets the 2 ULP floor.** `pow(a, b) = exp(b * log(a))`. If `log(a)` is accurate to 1 ULP, `exp(x)` is accurate to 1 ULP, and the multiply `b * log(a)` adds up to 0.5 ULP, the composed error is approximately (1 ULP in log) × exp-sensitivity + 1 ULP in exp + 0.5 ULP in multiply ≈ 2 ULP for `|b| ≤ 30`. For larger `|b|` the bound grows because the multiply amplifies log's error; that's why the domain is capped at `|b| ≤ 30`.

**Phase 2 upgrade path to 1 ULP.** Build Dekker-style double-double (Dekker 1971, §12 below) on the `log(a) * b` intermediate using TwoSum/TwoProd. Both are pure fp64 arithmetic with no FMA (I3-compliant). This is kept in §12 of this document as infrastructure research; Phase 2 activates it. **Do not build this in Phase 1.**

Team-lead signaled that simpler Phase 1 pow + Phase 2 upgrade is the right path — build once, tighten once, don't accidentally over-engineer the first version.

## The algorithm skeleton

```
def tam_pow(a, b):
    # --- Huge case dispatch front-end (most of the code) ---
    handled, value = handle_special_cases(a, b)
    if handled:
        return value

    # --- Negative-base + integer-exponent handling (adversarial B4, 2026-04-12) ---
    # The real-valued path below requires a > 0 (because log(a) is undefined
    # for a < 0). For a < 0, we route to one of two sub-paths depending on
    # whether b is a small or large integer:
    #
    # Case A: a < 0, b is a small integer (|b| <= 32): use integer_power
    #   directly. It handles negative a correctly via repeated signed multiply.
    #
    # Case B: a < 0, b is a large integer (|b| > 32): compute pow(|a|, b) via
    #   the real-valued path, then re-sign via (-1)^b. Requires is_odd_integer(b)
    #   to determine sign.
    #
    # Case C: a < 0, b is non-integer: already caught by handle_special_cases
    #   (returns NaN per IEEE 754 §9.2.1).
    if a < 0:
        if is_small_integer(b) and abs(b_as_int) <= 32:
            return integer_power(a, b_as_int)            # Case A
        # Case B: route through |a| and re-sign
        # Precondition here: b is a large integer; by spec-§9.2.1 non-integer
        # large b was already caught by handle_special_cases.
        magnitude = real_valued_pow(abs(a), b)
        if is_odd_integer(b):
            return -magnitude
        else:
            return magnitude

    # --- Integer-b fast path for a > 0 ---
    if is_small_integer(b):
        return integer_power(a, b_as_int)

    # --- Real-valued path: a^b = exp(b * log(a)) ---
    assert a > 0    # negative-a with non-integer b caught in specials;
                    # negative-a with integer b caught above
    return real_valued_pow(a, b)


def real_valued_pow(a, b):
    """Compute a^b for a > 0 via exp(b * log(a)).
    Phase 1: plain fp64 intermediate. Phase 2: Dekker double-double."""
    assert a > 0
    log_hi, log_lo = log_dd(a)              # double-double log
    prod_hi, prod_lo = dd_mul_f64(log_hi, log_lo, b)   # double-double multiply
    return exp_dd(prod_hi, prod_lo)          # exp with double-double input
```

**`is_odd_integer(b)` helper:** For `|b| ≤ 2^53`, it's `(floor(b) == b) and (int(floor(b)) % 2 != 0)`. For `|b| > 2^53`, every representable fp64 is an even integer (lowest mantissa bits lost to exponent), so `is_odd_integer(b) = false` trivially.

**Testing additions for B4:**
- `pow(-2.0, 3.0) == -8.0` bit-exact
- `pow(-2.0, 4.0) == 16.0` bit-exact
- `pow(-3.0, -1.0)` within 2 ULP of `-1/3`
- `pow(-4.0, 100.0)` via large-integer path, ULP-checked vs mpmath

Three new primitives are needed: `log_dd`, `dd_mul_f64`, and `exp_dd`. All three are thin wrappers over the existing `tam_ln` and `tam_exp` with extended-precision tracking.

## Special cases (the actual work)

The IEEE 754-2019 §9.2.1 specification for `pow(x, y)` is (note the sign-of-zero handling: `pown(-0, n)` for odd `n` preserves the negative sign, which is the easy-to-forget rule):

| Case | Result |
|---|---|
| `pow(+0, y)` for `y > 0`, non-integer or even | `+0` |
| `pow(+0, y)` for `y > 0`, odd integer | `+0` |
| `pow(-0, y)` for `y > 0`, non-integer or even | `+0` |
| `pow(-0, y)` for `y > 0`, odd integer | **`-0`** (sign preserved) |
| `pow(+0, y)` for `y < 0`, non-integer or even | `+inf` (and raises div-by-zero) |
| `pow(+0, y)` for `y < 0`, odd integer | `+inf` |
| `pow(-0, y)` for `y < 0`, non-integer or even | `+inf` |
| `pow(-0, y)` for `y < 0`, odd integer | **`-inf`** (sign preserved) |
| `pow(x, +0)` for any `x` (including nan) | `1` |
| `pow(x, -0)` for any `x` (including nan) | `1` |
| `pow(+1, y)` for any `y` (including nan) | `1` |
| `pow(-1, +inf)` | `1` |
| `pow(-1, -inf)` | `1` |
| `pow(x, nan)` | `nan` (except `pow(1, nan) = 1`, `pow(x, 0) = 1`) |
| `pow(nan, y)` | `nan` (except `pow(nan, 0) = 1`) |
| `pow(+inf, y > 0)` | `+inf` |
| `pow(+inf, y < 0)` | `+0` |
| `pow(-inf, y > 0)`, odd integer | `-inf` |
| `pow(-inf, y > 0)`, non-integer or even | `+inf` |
| `pow(-inf, y < 0)`, odd integer | `-0` |
| `pow(-inf, y < 0)`, non-integer or even | `+0` |
| `pow(x, +inf)` for `|x| > 1` | `+inf` |
| `pow(x, +inf)` for `|x| < 1` | `+0` |
| `pow(x, -inf)` for `|x| > 1` | `+0` |
| `pow(x, -inf)` for `|x| < 1` | `+inf` |
| `pow(x, y)` for `x < 0`, `y` non-integer | `nan` |

**The dispatch is ~30 branches.** Pathmaker writes a long front-end that handles these in order, returning early for each match. Once all the specials are out of the way, the remaining case is `a > 0, a ≠ 1, b ≠ 0, |a|, |b|` finite and non-special — that's where the real-valued path runs.

**Is `b` an integer?** `b : f64` is an integer iff `floor(b) == b` (and `b` is finite). Ranges are split:
- `|b| < 2^53`: we can directly detect integer-ness and compute `floor(b)` exactly.
- `|b| ≥ 2^53`: every fp64 `b` in this range is already an integer (mantissa has no fractional bits). But for even/odd determination, we need `b mod 2`. Since large fp64 integers lose their lowest bits, all such `b` are even in fp64 representation. So for `|b| ≥ 2^53`, treat as "even integer."

**Integer-power fast path.** For `|b| ≤ 30` (or whatever threshold), compute `a^b` via repeated squaring: `O(log b)` multiplies instead of `exp(b * log(a))`. This is both faster and more accurate for small integer exponents. **Threshold at `|b| ≤ 32`**, skip for non-integer or larger `b`.

Repeated squaring in fp64:
```
def integer_power(a, n):
    if n < 0:
        return 1 / integer_power(a, -n)
    result = 1.0
    base = a
    while n > 0:
        if n & 1:
            result = result * base
        base = base * base
        n >>= 1
    return result
```

Each fp64 multiply is ~0.5 ULP error on average; `k` multiplies accumulate to `~sqrt(k) * 0.5 ULP` on average and `k * 0.5 ULP` worst case. For `|n| ≤ 32` that's `≤ 16 ULPs` worst case, which is WORSE than 1 ULP. Hmm.

Actually wait — the integer-power path doesn't need to be more accurate than the real-valued path. For small integer `n`, the accuracy of `a^n` via repeated squaring is already very good: since each multiply is `0.5 ULP` on average with full cancellation of intermediate rounding, the accumulated error is bounded by roughly `(2n - 1) * 0.5 ULP` in the worst case (each multiply contributes `0.5 ULP`). For `n = 10`, that's `9.5 ULPs` worst case, or `~3 ULPs` RMS. Still above our 1 ULP bound.

**The fix for integer path:** do the multiplies in double-double to preserve precision, then round back to fp64 at the end. This costs a few more multiplies per step but preserves 1 ULP easily. Or: skip the fast path for integer `b` and always go through the `exp(b * log(a))` route with double-double intermediate. The `exp(b * log(a))` route *does* give 1 ULP for integer `b` via the double-double machinery.

**Recommendation for Phase 1:** skip the integer-b fast path entirely. Always go through `exp_dd(b * log_dd(a))`. Simpler, no second code path, same accuracy. Phase 2 can add the integer fast path if benchmarks show it matters.

## The double-double infrastructure

**NOTE: this section describes the Phase 2 upgrade path.** Phase 1 `tam_pow` targets 2 ULP using plain fp64 intermediates in the `exp(b · log(a))` path. The double-double machinery below is the route to 1 ULP in Phase 2 and is kept here as infrastructure research, not Phase 1 implementation.

Double-double ("dd") represents a value as an fp64 pair `(hi, lo)` such that `hi + lo` is the full value and `|lo| ≤ ulp(hi)/2`. Standard algorithms:

**TwoSum(a, b):** returns `(s, err)` such that `s + err = a + b` exactly, where `s = a + b` in fp64. Pure fp64 ops, no FMA, safe under I3:
```
s = a + b
bb = s - a
err = (a - (s - bb)) + (b - bb)
```

### MANDATORY implementation constraints for TwoProd and TwoSum (I3 enforcement)

**This is not advisory — violating it silently produces wrong answers that don't show up in simple tests.** Per adversarial's review 2026-04-12 (B2, B3):

Every arithmetic operation inside `TwoSum(a, b)`, `TwoProd(a, b)`, `split(a)`, and all double-double compositions **MUST** emit as a separate `.tam` op. Specifically:

- `split(a)` emits three ops: `c = fmul(a, 134217729.0)`, `tmp = fsub(c, a)`, `aH = fsub(c, tmp)`. **No backend may fuse any subexpression into an FMA.**
- `err = ((aH * bH - p) + aH * bL + aL * bH) + aL * bL` in TwoProd has FIVE multiplies and THREE adds. All emitted separately. Specifically, the subexpression `aH * bH - p` is `tmp = fmul(aH, bH); tmp2 = fsub(tmp, p)` — **NOT** `fma(aH, bH, -p)`. Any contraction here breaks the `p + err = a * b exactly` invariant.
- Pathmaker's `.tam` emitter already upholds I3 at the IR level (per spec §5.3 "all fp ops are non-contracting, FMA is never emitted"), so this constraint is structurally enforced by the IR. **But** if a future op like `fma.f64` is added to the IR, the pow implementation MUST NOT use it inside TwoSum/TwoProd.

This applies to every double-double invariant in tambear-libm, not just pow. Any Kahan/Neumaier/compensated-sum pattern in future libm functions inherits the same constraint.

**TwoProd(a, b):** returns `(p, err)` such that `p + err = a * b` exactly. Without FMA, this uses Dekker's splitting trick:
```
def split(a):
    c = a * 134217729.0      # 2^27 + 1
    aH = c - (c - a)
    aL = a - aH
    return aH, aL
# Then:
p = a * b
aH, aL = split(a)
bH, bL = split(b)
err = ((aH * bH - p) + aH * bL + aL * bH) + aL * bL
```
6 multiplies + 6 adds + 4 subs, no FMA. This is the Dekker 1971 algorithm.

**Double-double multiply by fp64:** given `(hi, lo)` and a scalar `b`, return `(prod_hi, prod_lo)`:
```
p, e = TwoProd(hi, b)
e2 = lo * b
sum = e + e2
prod_hi, prod_lo = TwoSum(p, sum)  # renormalize
```

**log_dd(a):** computed by running `tam_ln` at double precision as the `hi` part, then computing `log(a) - hi` via a Taylor correction in mpmath style — i.e., since `hi ≈ log(a)`, we have `a ≈ exp(hi)`, so `a / exp(hi) ≈ 1 + tiny`, and `log(a / exp(hi)) ≈ a/exp(hi) - 1` to very high precision. The `lo` part is this correction. Practical computation:
```
hi = tam_ln(a)                    # 1 ULP accurate in fp64
scale = exp_fast(hi)              # use lower-precision exp, 1-2 ULP is fine
ratio = a / scale                 # very close to 1
lo = tam_ln_near_1(ratio)          # log(1 + small) via specialized polynomial
# hi + lo = double-double log(a)
```
This is one way. A cleaner way is to fit `log` directly to double-double via a higher-degree polynomial that outputs a pair. That's the Phase 2 path.

**Recommendation for Phase 1:** the first version uses the scale-and-correct approach above. It's ugly but correct, and it sets up a clean seam for Phase 2 to replace.

**exp_dd(hi, lo):** compute `exp(hi + lo)` to 1 ULP in fp64:
```
e_hi = tam_exp(hi)                 # 1 ULP accurate
# correction: exp(hi + lo) = exp(hi) * exp(lo) ≈ exp(hi) * (1 + lo + lo^2/2 + ...)
#   for |lo| << 1 (which holds by construction, |lo| ≤ ulp(hi))
result = e_hi + e_hi * lo          # first-order correction; adequate if |lo| < 2^-30

# Derivation of the 2^-30 bound (per adversarial's B3 review, 2026-04-12):
# The error from truncating exp(lo) ≈ 1 + lo (instead of the full Taylor series)
# is |exp(lo) - (1 + lo)| ≈ lo^2 / 2. For |lo| < 2^-30, lo^2/2 < 2^-61, which is
# below 1 ULP of the result.
#
# Domain check for Phase 2 pow (a ∈ [2^-100, 2^100], b ∈ [-30, 30]):
#   |lo| is the low part of dd_mul_f64(log_dd_lo, b), where log_dd_lo ≤ ulp(log_dd_hi).
#   For |hi| ≤ 709 (exp domain), ulp(hi) ≤ 709 · 2^-52 ≈ 1.6e-13 ≈ 2^-42.8.
#   Multiplying by |b| ≤ 30 (which is < 2^5) gives |lo| ≤ 2^-42.8 · 2^5 ≈ 2^-37.8,
#   comfortably within the 2^-30 bound.
#
# If a future phase extends the b domain beyond ±30, this bound must be re-derived
# and the second-order term e_hi * lo^2 / 2 added to the correction. Do not extend
# the domain without re-deriving the bound.
# (If |lo| is larger, add higher-order terms.)
```

## Pitfalls

1. **Getting the special-case table wrong.** The 30-case table is the most-tested part of this function. Add every entry as an adversarial test with named expected output.
2. **`pow(0, 0) = 1`, not `0` or `nan`.** This is by convention. Some libms got it wrong historically; we do not.
3. **`pow(-1, ±inf) = 1`.** Because `-1` has magnitude `1`, and `1^anything = 1`. Another convention-trap.
4. **`pow(nan, 0) = 1` and `pow(1, nan) = 1`.** Edge cases where nan doesn't propagate. Convention-trap.
5. **Odd/even integer detection for negative bases.** Must be done in fp64 without going through `int` truncation (which would mis-round for very large `b`). Check the parity of the mantissa's lowest set bit relative to the exponent.
6. **Sign preservation for odd-integer exponents of negative bases.** `pow(-2, 3) = -8`, not `+8`. The `exp(log|a| * b)` path loses the sign; must multiply by `(-1)^n` at the end for odd integer `b`.
7. **Subnormal results.** For `pow(tiny, big)`, the result may underflow through the subnormal range. The `exp_dd` reassembly must handle this correctly, same as plain `tam_exp`.

## Testing

- 1M random samples from `a ∈ [2^-100, 2^100]`, `b ∈ [-30, 30]`. `max_ulp ≤ 1.0`.
- All 30 special-value cases, bit-exact.
- Integer exponent test: `pow(2, n)` for `n ∈ [-30, 30]` must return exactly `2^n` (which is representable in fp64).
- `pow(a, 0.5)` must equal `sqrt(a)` to 1 ULP for `a ≥ 0`. (This tests the log/exp/mul pipeline against the `fsqrt` oracle.)
- `pow(a, 2.0)` for integer-flag test: must equal `a * a` bit-exact.
- `pow(a, 1.0) == a` bit-exact.

## Open questions

1. **Double-double inside a single libm function — is that consistent with the project's "pure fp64" aesthetic?** Yes. Double-double is just fp64 pairs manipulated with fp64 ops. The .tam IR sees only fp64. The `pow` function is the one place in Phase 1 where we use pairs of fp64 as a richer number representation, because a single fp64 intermediate cannot carry enough precision for `pow` to hit 1 ULP. Other functions don't need this.
2. **Is Dekker's Split constant (2^27 + 1 = 134217729) an issue for FMA-contracted backends?** The split is `a * const - (a * const - a)`, which looks like it could contract. We must emit it as three explicit ops and no FMA. I3 again.
3. **Phase 1 fallback for `|b| > 30`?** Recommend: return a warning value (just `pow` via exp/log without the double-double cleanup), documented as >1 ULP outside the primary domain. Phase 2 extends the bound.

## References

- T. J. Dekker, "A floating-point technique for extending the available precision," Numerische Mathematik 18:224–242, 1971. (TwoSum, TwoProduct, and the splitting trick.)
- J.-M. Muller et al., "Handbook of Floating-Point Arithmetic," 2nd ed., 2018, Chapter 4 (double-word arithmetic) and Chapter 12 (`pow`).
- IEEE 754-2019, §9.2 (pow special cases) and Annex A.
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic," ACM Computing Surveys 23(1):5–48, 1991. (Overview and rationale for the special-case table.)
