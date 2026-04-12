# Adversarial Pre-Code Review — `pow-design.md`

**Reviewer:** Adversarial Mathematician
**Date:** 2026-04-12
**Status:** FOUR BLOCKING ISSUES. TWO near-blocking advisories. Implementation must not start (Campsite 2.17) until resolved.

---

## Summary verdict

The pow design is structurally sound and the choice to skip the integer fast path in Phase 1 is correct. Double-double is the right tool. However, four blocking issues: the special-case table has two errors (not just ambiguities — outright wrong entries), the Dekker splitting constant contraction risk under I3 is acknowledged but not resolved, the error bound analysis for the `exp_dd` approximation has an undocumented failure domain, and the sign preservation problem for negative-base odd-integer exponents is identified but the fix is underspecified.

---

## Blocking issues

### B1 — Special-case table has TWO incorrect entries

The table in the design doc says:

```
pow(+0, y)  for  y > 0, odd integer   → +0
pow(-0, y)  for  y < 0, odd integer   → -inf
```

**Error 1:** `pow(+0, y)` for `y > 0`, odd integer should be `+0`. This is correct per IEEE 754-2019 §9.2.1. BUT: `pow(-0, y)` for `y > 0`, odd integer should be `-0`, NOT `+0`. The table says `+0` for the second row:

```
pow(-0, y)  for  y > 0, non-integer or even   → +0   ✓ correct
pow(-0, y)  for  y > 0, odd integer           → +0   ✗ WRONG: should be -0
```

IEEE 754-2019 §9.2.1 explicitly states: `pown(-0, n)` for odd positive n returns -0. The design doc conflates both `-0` rows as returning `+0`. This is a bit-exact error.

**Error 2:** `pow(-inf, y < 0), non-integer or even → +0`. Per IEEE 754-2019 §9.2.1: `pow(-inf, y)` for `y < 0` with y odd integer is `-0` (the table says this correctly). But the non-integer/even case must be `+0`. The design doc says `+0` — this is CORRECT. So this is not an error.

**Actual error 2:** Look at the row:
```
pow(+0, y)  for  y < 0, odd integer   → +inf
```
IEEE 754 §9.2.1 says `pown(+0, n)` for odd negative n returns `+inf` WITH a divide-by-zero exception. But `pown(-0, n)` for odd negative n returns `-inf`. The design doc does include `pow(-0, y < 0, odd integer) → -inf`. Checking the table again:

```
pow(+0, y)  for  y < 0, non-integer or even   → +inf
pow(+0, y)  for  y < 0, odd integer           → +inf
pow(-0, y)  for  y < 0, non-integer or even   → +inf
pow(-0, y)  for  y < 0, odd integer           → -inf
```

IEEE 754: `pow(+0, y < 0, odd integer) = +inf`. `pow(-0, y < 0, odd integer) = -inf`. These two are correct. BUT: `pow(-0, y > 0, odd integer)` is listed as `+0` in the table. IEEE 754 says it should be `-0`.

**Required fix:** In the special-case table, change:
- `pow(-0, y)` for `y > 0`, odd integer: `+0` → `-0`

This is bit-exact: one entry is incorrect.

---

### B2 — Dekker splitting constant: I3 violation is live risk, not acknowledged

The design doc says:

> "Is Dekker's Split constant (2^27 + 1 = 134217729) an issue for FMA-contracted backends? The split is `a * const - (a * const - a)`, which looks like it could contract. We must emit it as three explicit ops and no FMA. I3 again."

This is recognized as a risk in the Open Questions section but is NOT in the pitfalls section and NOT in the required implementation constraints. A risk acknowledged only in an open-questions footnote is not a specification constraint. The pathmaker will not necessarily see this as a hard requirement.

The specific contraction risk: the expression `c = a * 134217729.0; aH = c - (c - a); aL = a - aH`. If a backend contracts `c - (c - a)` as `fma(-1, (c-a), c)` or similar, the result is different and the double-double invariant breaks. The invariant `aH + aL = a` exactly requires that the splitting multiply is NOT contracted.

More dangerous still: `err = ((aH * bH - p) + aH * bL + aL * bH) + aL * bL`. The sub-expression `aH * bH - p` must be computed as a separate multiply then subtract. If contracted to `fma(aH, bH, -p)`, the result changes.

**Required fix:** Move the I3 constraint on Dekker splitting out of Open Questions and into a MANDATORY implementation constraint section:

> "The TwoProd implementation MUST emit each multiply and add as a separate `.tam` op. Specifically: `split(a)` is 3 ops: `fmul(a, 134217729)`, `fsub(c, a)`, `fsub(c, prev_result)`. Any expression of the form `a * b - c` in TwoProd MUST NOT be emitted as a single FMA. Violating this invalidates the double-double invariant and produces wrong answers that are hard to detect."

---

### B3 — `exp_dd` first-order correction has an undocumented failure domain

The design doc says:

```python
result = e_hi + e_hi * lo      # first-order correction; adequate if |lo| < 2^-30
# (If |lo| is larger, add higher-order terms.)
```

The claim "`adequate if |lo| < 2^-30`" is asserted without derivation, and the failure case is left to pathmaker as an implicit fallback.

The error from truncating `exp(lo) ≈ 1 + lo` (rather than the full Taylor series) is: `|exp(lo) - (1 + lo)| ≈ lo^2/2`. For `|lo| < 2^-30`, `lo^2/2 < 2^-61`, which is below 1 ULP of the result. Good.

But: what is the actual bound on `|lo|`? The `dd_mul_f64` function produces `prod_lo` as the low part of `hi * b + lo * b`. For `|hi| ≤ 709` (exp domain) and `b ∈ [-30, 30]`, `|lo| ≤ ulp(hi) ≤ ulp(709) ≈ 709 * 2^-52 ≈ 1.6 * 10^-13 ≈ 2^-42.8`. So `|lo| < 2^-43` well within the `2^-30` bound. Fine for Phase 1 domain.

BUT: the design doc does NOT derive this bound. A pathmaker who doesn't verify the bound might use exp_dd outside its stated domain (e.g., Phase 2 with larger `b`) and silently get >1 ULP results.

**Required fix:** Add to the `exp_dd` specification: "The first-order correction `e_hi * lo` is adequate when `|lo| < 2^-30`. For the Phase 1 domain (`b ∈ [-30, 30]`), `|lo| ≤ 2^-43`, so the bound holds. If Phase 2 extends to larger `b`, the second-order term `e_hi * lo^2 / 2` must be added, and the bound derivation must be re-verified."

---

### B4 — Sign preservation for negative-base odd-integer exponents: fix is unspecified

Pitfall 6 in the design doc says:

> "`pow(-2, 3) = -8`, not `+8`. The `exp(log|a| * b)` path loses the sign; must multiply by `(-1)^n` at the end for odd integer `b`."

This is correctly identified as a pitfall. But the algorithm skeleton at the top of the doc says:

```python
# --- Real-valued path: a^b = exp(b * log(a)) ---
assert a > 0           # negative-a with non-integer b was caught in specials
log_hi, log_lo = log_dd(a)
```

The comment says "negative-a with non-integer b was caught in specials." But negative-a with INTEGER b is NOT caught in specials (the integer-b case was removed from the fast path). So after removing the integer-b fast path, the algorithm does what for `a = -2, b = 3`?

Tracing through: `handle_special_cases(-2, 3)` — is `a = -2` a special? Looking at the table: `pow(x, y) for x < 0, y non-integer = nan`. But `y = 3` IS an integer. So this does NOT return nan from the specials. The specials table for negative `a` only handles the nan case for non-integer `b`. Integer `b` with negative `a` falls through to the real-valued path.

The real-valued path then has `a = -2`, and the comment says `assert a > 0`. So either the assertion fires (crash) or this path is silently reached with `a < 0`.

**The missing branch:** for `a < 0` and `b` an integer, the algorithm must:
1. Detect that `b` is an integer
2. Compute `integer_power(|a|, b)` (or `exp_dd(b * log_dd(|a|))`)
3. If `b` is an odd integer: negate the result
4. If `b` is an even integer: return the (positive) result

The current design omits this branch entirely. The sign-preservation pitfall is identified but the implementation path for negative `a` with integer `b` is missing from the skeleton.

**Required fix:** Add the following branch to the algorithm skeleton, between the specials dispatch and the real-valued path:

```python
# --- Negative base with integer exponent ---
if a < 0:
    assert is_integer(b), "non-integer exponent of negative base was caught in specials"
    b_int = int(b)
    mag = exp_dd(-b_int, log_dd(-a))   # compute |a|^|b|, careful with sign of b
    if b_int % 2 != 0:
        return -mag   # odd integer: negate
    else:
        return mag    # even integer: positive
```

(The exact code is pathmaker's job; the design must require this branch to exist.)

---

## Advisory notes (non-blocking)

### A1 — `pow(a, 0.5)` vs `sqrt(a)` identity test: the 1 ULP claim

The testing section says: "`pow(a, 0.5)` must equal `sqrt(a)` to 1 ULP for `a ≥ 0`." This is aspirational but may not hold. `pow(a, 0.5)` goes through `exp(0.5 * log(a))`, accumulating: 1 ULP from `log`, 0.5 ULP from the multiply by 0.5, then 1 ULP from `exp`. Composed error: up to ~2.5 ULPs. `fsqrt.f64` is required to be 0.5 ULP by IEEE 754 (correctly-rounded square root).

So `pow(a, 0.5) == sqrt(a)` within 1 ULP is NOT guaranteed by the design. The test as written will FAIL for some inputs where the composed 2.5-ULP pow error happens to differ from the 0.5-ULP sqrt.

**Advisory:** Change the test to: "`pow(a, 0.5) == sqrt(a)` within 3 ULPs." OR: add a special case for `b = 0.5` that routes to `fsqrt`. If the design intent is that `pow(a, 0.5) == sqrt(a)` exactly (bit-exact), that requires routing to `fsqrt` — the general `exp/log` path cannot achieve this.

### A2 — The `log_dd` "ugly but correct" approach: the `tam_ln_near_1` sub-function

The `log_dd` construction uses a function called `tam_ln_near_1(ratio)` where `ratio = a / exp_fast(hi)` is very close to 1. The design says this is a "specialized polynomial." This function must be:
1. Designed before `log_dd` is implemented
2. Accurate to the precision needed for the double-double `lo` part (which requires ~106-bit accuracy)

For `ratio = 1 + ε` with `ε` small, `log(ratio) ≈ ε - ε²/2 + ε³/3 - ...`. The key insight: the full Taylor series converges quickly when `|ε|` is small. But how small is `ε` here? Since `hi = tam_ln(a)` is accurate to 1 ULP, `exp(hi) ≈ a * (1 + error)` where `|error| < 2^-52`. So `ratio = a / exp(hi) ≈ 1 + error`, meaning `|ε| < 2^-52`. Then `log(ratio) ≈ ε - ε²/2 ≈ ε` to 1 ULP (since `ε²/2 < 2^-105`, which is below fp64 resolution). So `lo ≈ a/exp(hi) - 1`.

This simplification should be in the design doc. The "specialized polynomial" reduces to a single fp64 subtract: `lo = ratio - 1.0` (for `|ε| < 2^-52`). No polynomial needed.

**Advisory:** Replace the vague `tam_ln_near_1` reference with the explicit formula: `lo = (a / exp_fast(hi)) - 1.0`. Document why this is exact enough: since `|lo| < 2^-52`, the quadratic and higher terms are below fp64 resolution and can be omitted.

### A3 — TMD corpus for `pow`

Known hard cases:
- `pow(2.0, 0.5)` — should be `sqrt(2) = 1.4142135623730950488...`. The nearest fp64 to the true value is `0x3FF6A09E667F3BCD` (which is just 0.5 ULP below the midpoint). This is a known TMD case for `pow(2, 0.5)`.
- `pow(3.0, 1.0/3.0)` — `∛3`. The true value is `1.4422495703074083...`. Known TMD candidate.
- `pow(e, 1.0)` — should be `e` itself, but `e` is not exactly representable. `pow(fp64(e), 1.0) = fp64(e)` bit-exact (since the exponent is 1.0 exactly, the fp64 multiply should preserve the input).

Log in `peak2-libm/tmd-corpus/pow-tmd.md`.

---

## I11 coverage gap

The current special-case table includes `pow(nan, y) = nan` (except when `y = 0`) and `pow(x, nan) = nan` (except when `x = 1`). The I11 invariant requires NaN propagation through every op. The design correctly handles this. But verify: the front-end dispatch must check NaN BEFORE checking `y == 0` and `x == 1`, because `pow(nan, 0) = 1` and `pow(1, nan) = 1` are the ONLY exceptions to NaN propagation in `pow`. These exceptions are IEEE 754-mandated.

The design doc lists these in the table as `pow(x, ±0) = 1 for any x including nan` and `pow(+1, y) = 1 for any y including nan`. These are noted in pitfall #4. But the dispatch order must be:

```
if b is ±0: return 1   # before NaN checks
if a is +1: return 1   # before NaN checks
if isnan(a) or isnan(b): return nan
```

Any other order breaks one of the two mandatory non-propagation cases.

**Advisory:** Add the required dispatch order explicitly to the front-end as a sequencing constraint, not just as a pitfall note.

---

## Required additions to test battery

1. **Fix Error 1:** Add `assert pow(-0.0, 3.0).to_bits() == (-0.0f64).to_bits()` — the design doc's table says `+0` for this case which is wrong.
2. **Sign preservation for odd-integer negative-base:** `pow(-2.0, 3.0) == -8.0` bit-exact; `pow(-2.0, 4.0) == 16.0` bit-exact; `pow(-3.0, -1.0) == -1.0/3.0` within 1 ULP.
3. **NaN non-propagation exceptions:** `pow(nan, 0.0) == 1.0` and `pow(1.0, nan) == 1.0` — bit-exact.
4. **Dispatch order test:** `pow(nan, nan)` must return `nan` (not `1` due to either exception).
5. **Dekker splitting I3 audit:** For the implementation, add a test that `TwoProd(a, b)` gives exact results matching mpmath at 200-bit precision for a set of carefully chosen inputs.
6. **pow(a, 0.5) vs sqrt(a):** change to 3-ULP bound OR add routing test.
7. **Large-b accuracy:** `pow(1.1, 30)` — compare to mpmath at 200 digits. This is at the boundary of the Phase 1 domain and the composed error should be ≤1 ULP if the double-double infrastructure is correct.

---

## Verdict

**HOLD on Campsite 2.17 implementation until:**

1. B1 resolved: fix the table entry `pow(-0, y > 0, odd integer)` from `+0` to `-0`.
2. B2 resolved: move the I3/Dekker constraint out of Open Questions into a mandatory implementation constraint.
3. B3 resolved: derive the bound on `|lo|` in `exp_dd` and document the failure domain.
4. B4 resolved: add the negative-base integer-exponent branch to the algorithm skeleton with explicit sign restoration logic.
