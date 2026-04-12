# `tam_tanh`, `tam_sinh`, `tam_cosh` — Algorithm Design Document

**Campsite 2.18.** Hyperbolic functions — all compositions on `tam_exp`.

**Owner:** math-researcher
**Status:** draft, awaiting navigator + pathmaker review
**Date:** 2026-04-11 (amended 2026-04-12 with stub-vs-composition clarification)

**Upstream dependency:** `exp-design.md`, `accuracy-target.md`.

---

## IR stub layout

**None of these three functions need new IR stubs.** They are pure `.tam` functions in the libm text that call `tam_exp` (which already has a stub). Scout verified 2026-04-12 that the Phase 1 `.tam` spec does not declare `tam_tanh`, `tam_sinh`, or `tam_cosh` stubs — and it shouldn't. They are compositions.

| Function | Mechanism | Stub needed? |
|---|---|---|
| `tam_tanh` | pure `.tam` function calling `tam_exp` | No |
| `tam_sinh` | pure `.tam` function calling `tam_exp` | No |
| `tam_cosh` | pure `.tam` function calling `tam_exp` | No |

**IR ops used:**
- `tam_exp` — the transcendental stub (TamExp in ast.rs); all three functions call it
- `fabs.f64` — used for `|x|` in polynomial-regime dispatch. Confirmed present in spec §5.3 as `FAbs { dst, a }` at ast.rs:198. (Adversarial B3 resolution, 2026-04-12: no IR amendment needed; adding this explicit note.)

**Net ask of pathmaker:** nothing. These functions live in the libm `.tam` text as compositions; no IR amendment required.

---

## Background identities

```
sinh(x) = (e^x - e^(-x)) / 2
cosh(x) = (e^x + e^(-x)) / 2
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)  =  sinh(x) / cosh(x)
```

Using these directly has two problems:
1. **Cancellation near `x = 0`.** For small `x`, `sinh(x) ≈ x` and `tanh(x) ≈ x`, but the formula `(e^x - e^(-x))/2` computes two values that are each near `1`, subtracts them, and loses all significant bits. Catastrophic.
2. **Overflow for large `x`.** `e^x` overflows around `x ≈ 710`. But `tanh(x)` approaches `±1` smoothly and should never overflow. `sinh` and `cosh` do legitimately overflow around `x ≈ 710`.

The fix is a piecewise algorithm that switches formula based on `|x|`.

## `tam_sinh(x)`

Domain: `[-710, 710]` (outside: overflow to `±inf`).

**Piecewise algorithm:**

```
if |x| < 2^-28:
    # Near zero: sinh(x) ≈ x + x^3/6 + x^5/120 + ...
    # Return x directly — the polynomial correction is below 1 ULP.
    return x
elif |x| < 1:
    # Small regime: use a Remez polynomial fit to sinh(x) - x on [-1, 1]
    # to avoid the cancellation in (e^x - e^-x)/2.
    return x + x * x * x * SINH_POLY(x*x)
elif |x| < 22:
    # Medium regime: full formula, cancellation is manageable
    e_x = exp(x)
    e_neg = 1 / e_x             # or: exp(-x), both are 1 ULP
    return (e_x - e_neg) / 2
else:
    # Large regime: e^(-x) is negligible compared to e^x
    # sinh(x) ≈ e^x / 2   (for x >> 0) or -e^(-x) / 2 (for x << 0)
    # But watch overflow near the boundary.
    if x > 0:
        return exp(x - ln(2))   # = e^x / 2, computed without forming the huge value first
    else:
        return -exp(-x - ln(2)) # sign-flipped mirror
```

The threshold at `|x| = 22`: past `|x| = 22`, the term `e^(-x)/2` is below 0.5 ULP of `e^x/2` so dropping the subtraction is safe. **Derivation** (corrected per adversarial review 2026-04-12 B1 — the earlier "e^-22 < 2^-64" claim was mathematically wrong; `e^-22 ≈ 2.8e-10 ≈ 2^-31.8`, not `2^-64`):

For `sinh(x) = (e^x - e^(-x))/2` to round correctly when we drop the `-e^(-x)/2` term, we need
    |e^(-x)/2| < 0.5 · ulp(e^x/2) = 0.5 · (e^x/2) · 2^-52 = e^x · 2^-54.
Dividing both sides by `e^(-x)/2`:
    1 < e^(2x) · 2^-53
    e^(2x) > 2^53
    2x > 53 · ln(2) ≈ 36.7
    x > 18.4

So the strict threshold is `|x| > ~18.4` for the dropped term to be below 0.5 ULP. We use `|x| = 22` with margin to guarantee correctness plus extra slack for accumulated rounding, which gives `e^(-44) ≈ 2^-63.5` in the dropped term vs `~2^-53` for 1 ULP — a ~10-bit safety margin. The same 22 threshold applies to `cosh` and `tanh` for the same reason.

The threshold at `|x| = 1`: below `|x| = 1`, the cancellation loses up to 2 bits, which is already above 1 ULP. The polynomial fit is cheaper and more accurate.

The threshold at `|x| = 2^-28`: the cubic correction is below 1 ULP, return `x` directly.

### The small-regime polynomial

Remez fit to `(sinh(x) - x) / x³` on `[-1, 1]`. This is even in `x`, so a polynomial in `u = x²` suffices. Degree ~6 in `u` reaches 1 ULP.

```
sinh(x) ≈ x + x^3 * SINH_POLY(x^2)
```

### Special values

```
sinh(+0) = +0
sinh(-0) = -0
sinh(+inf) = +inf
sinh(-inf) = -inf
sinh(nan) = nan
|x| > 710: overflow to signed inf
```

## `tam_cosh(x)`

Domain: `[-710, 710]`.

**Piecewise algorithm:**

```
if |x| < 2^-26:
    # cosh(x) ≈ 1 + x^2/2 + x^4/24 + ...
    # Return 1.0 exactly — the quadratic correction is below 1 ULP
    return 1.0
elif |x| < 22:
    # Medium regime: two-call form — exp(x) and exp(-x) are independent
    # 1-ULP calls. Their sum has no cancellation (both positive), so the
    # result is accurate to 1 ULP dominated by the larger term.
    # (Adversarial B2 resolution, 2026-04-12: earlier draft used 1/e_x which
    # compounded error to ~1.5 ULP; replaced with the two-call form.)
    e_pos = tam_exp(x)
    e_neg = tam_exp(-x)
    return (e_pos + e_neg) / 2
else:
    # Large regime: cosh(x) ≈ e^|x| / 2
    if x > 0:
        return exp(x - ln(2))
    else:
        return exp(-x - ln(2))
```

`cosh` is **even**, so `cosh(-x) = cosh(x)`. The algorithm enforces this via the `|x|` reduction.

No polynomial needed for `cosh` since its small-regime formula doesn't cancel — `(e^x + e^-x) / 2` is always a sum of positive values. But near `x = 0`, `e^x ≈ 1 + x + x²/2 + ...` and `e^-x ≈ 1 - x + x²/2 - ...`, so `e^x + e^-x ≈ 2 + x² + x⁴/12 + ...`. No cancellation in the addition, and dividing by 2 gives `1 + x²/2 + x⁴/24 + ...`. For tiny `x`, the leading `1` dominates; the `x²/2` term is below 1 ULP when `x² < 2^-53`, i.e., `|x| < 2^-26.5`. So we return `1.0` for `|x| < 2^-26`.

### Special values

```
cosh(+0) = 1.0
cosh(-0) = 1.0
cosh(+inf) = +inf
cosh(-inf) = +inf       (cosh is even)
cosh(nan) = nan
|x| > 710: overflow to +inf
```

## `tam_tanh(x)`

Domain: `(-inf, +inf)`. Unlike sinh and cosh, `tanh` is bounded — always in `(-1, 1)` — so there is no overflow.

**Piecewise algorithm:**

```
if isnan(x):
    return x
if |x| < 2^-28:
    # tanh(x) ≈ x - x^3/3 + 2x^5/15 - ...
    # Return x directly
    return x
elif |x| < 0.55:
    # Small regime: Remez polynomial fit to tanh(x)/x - 1 on [-0.55, 0.55]
    # to avoid cancellation in (e^2x - 1)/(e^2x + 1)
    return x + x * x * x * TANH_POLY(x*x)
elif |x| < 22:
    # Medium regime: two-call form (per navigator ruling 2026-04-12, adversarial B4).
    # The earlier single-call form `1 - 2/(e^(2x)+1)` cannot achieve 1 ULP near
    # |x| = 0.55 because the final subtraction amplifies relative error by
    # 1/tanh(x). Two independent tam_exp calls avoid this.
    # Sign is implicit: (e_pos - e_neg) is negative for x < 0 automatically.
    # Error budget: each exp call at 1 ULP independently; fdiv at 0.5 ULP;
    # total ≤ 2 ULP worst case (same carve-out as atan2 and pow).
    e_pos = tam_exp(x)
    e_neg = tam_exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)
else:
    # Large regime: tanh(x) → ±1 to full precision
    return copysign(1.0, x)
```

The `|x| > 22` regime: `e^(2x) > 2^63`, so `2/(e^(2x) + 1) < 2^-62`, which is below `1 ULP` of `1.0`. The result is `±1.0` to full precision.

The `|x| > 0.55` regime: use the two-call form `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` per navigator ruling 2026-04-12 (adversarial B4). The earlier draft used `1 - 2/(e^(2x) + 1)` which could not achieve 1 ULP near `|x| = 0.55`; the two-call form bounds error at ≤ 2 ULP.

The `|x| < 0.55` regime: the formula cancels. Use a polynomial.

### The small-regime polynomial

Remez fit to `(tanh(x) - x) / x³` on `[-0.55, 0.55]`. Even in `x` → polynomial in `u = x²`. Degree ~5 in `u` reaches 1 ULP.

### Special values

```
tanh(+0) = +0
tanh(-0) = -0
tanh(+inf) = +1.0
tanh(-inf) = -1.0
tanh(nan) = nan
```

`tanh` never overflows.

## Pitfalls

1. **Cancellation near `x = 0` in `sinh` and `tanh`.** Handled by the piecewise switch to a polynomial. Don't try to use the formula in the small regime.
2. **Overflow near `|x| = 710` in `sinh` and `cosh`.** Handled by the `exp(x - ln(2))` trick — compute `e^|x| / 2` without forming `e^|x|` first.
3. **Symmetry preservation.** `sinh` and `tanh` are odd, `cosh` is even. The piecewise dispatch must not break the symmetry. For `sinh(-0)` we must return `-0` bit-exact, for `cosh(-0)` we must return `+1.0`.
4. **`tanh` overflow illusion.** `tanh` does not overflow, but if you naively compute `(e^(2x) - 1)/(e^(2x) + 1)`, you get `(inf - 1)/(inf + 1) = nan` for large `x`. Use the stable form `1 - 2/(e^(2x) + 1)` which gives the correct `±1` limit.
5. **The polynomial thresholds.** `|x| < 0.55` for `tanh` vs `|x| < 1` for `sinh`: these are chosen so that the polynomial fit reaches 1 ULP at reasonable degree. Don't confuse them.
6. **Composed error for `tanh` via `sinh/cosh`.** Do NOT compute `tanh(x) = sinh(x) / cosh(x)` — the composed error is ~3 ULPs worst case. Compute `tanh` directly.

## Testing

- 1M random samples per function, exponent-uniform on its primary domain.
- Symmetry: `sinh(-x) = -sinh(x)` bit-exact; `cosh(-x) = cosh(x)` bit-exact; `tanh(-x) = -tanh(x)` bit-exact.
- Identity: `cosh²(x) - sinh²(x) = 1` within 3 ULPs (composed).
- Identity: `tanh(x) = sinh(x) / cosh(x)` within 3 ULPs (composed).
- Large-argument overflow: `sinh(710)`, `cosh(710)` → `+inf`; `sinh(-710)` → `-inf`; `tanh(710) = 1.0`, `tanh(-710) = -1.0`.
- Near-zero: `sinh(1e-20) = 1e-20`, `cosh(1e-20) = 1.0`, `tanh(1e-20) = 1e-20`.
- Polynomial boundary: samples at `|x| = 0.55 - ε` and `0.55 + ε` for tanh, `1 - ε` and `1 + ε` for sinh.

## Open questions

1. **Is `1/e_x` or `exp(-x)` the right way to compute the complementary value in the medium regime?** The `1/e_x` form is 1 fdiv (0.5 ULP); `exp(-x)` is another 1 ULP call. For 1 ULP target, `1/e_x` is safer because it stays within the same exp call's rounding neighborhood.
2. **`tanh` small-regime polynomial vs direct formula at `|x| = 0.55`:** the threshold is empirical. Navigator: check by benchmarking both ways at `|x| = 0.54` and `0.56` once pathmaker has the implementation.

## References

- W. J. Cody & W. Waite, "Software Manual for the Elementary Functions," Prentice-Hall, 1980, §§10–11 (sinh, cosh, tanh).
- J.-M. Muller et al., "Handbook of Floating-Point Arithmetic," 2nd ed., 2018, Chapter 11.
- Abramowitz & Stegun, "Handbook of Mathematical Functions," §4.5 (hyperbolic function properties and series).
