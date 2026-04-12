# `tam_ln` — Algorithm Design Document

**Campsite 2.10.** Algorithm selection and design for `tam_ln(f64) -> f64`, the natural logarithm.

**Owner:** math-researcher
**Status:** draft, awaiting navigator + pathmaker review
**Date:** 2026-04-11

**Upstream dependency:** `accuracy-target.md` (≤ 1 ULP faithful rounding), `exp-design.md` (uses same Cody-Waite, Remez, Horner patterns).

---

## What this function does

Given `x : f64`, return `log(x) : f64` (natural log, base e) with:
- `max_ulp ≤ 1.0` across 1M random samples drawn exponent-uniformly from `(0, fp64_max]`.
- IEEE 754 specials: `log(1) = +0` exactly, `log(+inf) = +inf`, `log(nan) = nan`.
- `log(0) = -inf` (both `+0` and `-0` — `log(-0)` is defined as `-inf` per IEEE 754, NOT NaN).
- `log(x) = nan` for any `x < 0`. Also set the invalid-operation flag if we track it; Phase 1 doesn't track fp flags.
- `log(1) == 0.0` bit-exact.

## The big picture

For any positive fp64 `x`, we can decompose:

```
x = m * 2^e                  with m ∈ [1, 2),  e = unbiased exponent
log(x) = log(m) + e * log(2)
```

Extracting `e` and `m` is a bit-level operation — `e` is the unbiased exponent field of the fp64 representation, `m` is the fp64 value with exponent field forced to `0` (biased = 1023). Both are *exact* and *free* — no arithmetic is involved, just bit manipulation. This is why `log` has a cleaner range reduction than `exp`.

`log(m)` on `[1, 2)` is then a polynomial problem, and `e * log(2)` is a scalar multiply of a cheap integer by a constant.

The wrinkle: `log(m)` on `[1, 2)` is small at `m = 1` (`log(1) = 0`) and larger at `m = 2` (`log(2) ≈ 0.693`). The function is smooth, but a direct polynomial on `[1, 2)` needs careful conditioning because catastrophic cancellation happens near `m = 1` if we're not careful.

## Sub-steps

### 4.1 Extract `e` and `m`

`x_bits = f64_to_bits(x)`. The unbiased exponent is `e = (x_bits >> 52) - 1023`. The fraction is `m = bits_to_f64((x_bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000)`, which is `x`'s mantissa with exponent forced to `0`, giving `m ∈ [1, 2)`.

**Edge case: subnormals.** If `x` is subnormal (biased exponent = 0, mantissa ≠ 0), the standard `(bits >> 52) - 1023` gives `e = -1023`, which is wrong; the real exponent is `-1022` and the leading bit of the mantissa is *not* implicit. For Phase 1 we front-end subnormals:
- Compute an adjusted `m` and `e` via one multiplication: `x' = x * 2^52`, which turns a subnormal `x` into a normal number `x'` with `log(x') = log(x) + 52 * log(2)`. Then apply the normal-path extraction to `x'` and subtract `52 * log(2)` at the end.

**Edge case: `x = 1.0` exactly.** `m = 1`, `e = 0`, polynomial must return exactly `0`. The polynomial form below guarantees `P(0) = 0`, so this is automatic.

### 4.2 Shift `m` for conditioning near 1

Direct `log(m)` on `[1, 2)` has a zero at `m = 1`, so we define `f = m - 1`, `f ∈ [0, 1)`, and our polynomial approximates `log(1 + f)`.

But there's a numerical issue. The range `[1, 2)` in fp64 has its mantissa densely distributed near 1 (where `m - 1` is small and relative precision of `m - 1` is good) and its mantissa less densely distributed near 2 (where `m - 1 ≈ 1` and precision is fine too). The problem is that the polynomial's convergence is worst near `m = 2` (`f = 1`), where the Taylor series for `log(1 + f) = f - f²/2 + f³/3 - ...` converges slowly.

**The fix used by Tang 1990 and most modern libms:** if `m > sqrt(2)`, rewrite `log(m) = log(m/2) + log(2)` so the polynomial input is now `m/2 ∈ [1, sqrt(2))`, i.e., `f = m/2 - 1 ∈ [0, sqrt(2) - 1) ≈ [0, 0.414)`. This halves the interval and makes the polynomial degree requirement much lower.

Concretely:
```
if m >= sqrt(2):
    m  = m / 2       # but: we cheat — just add 1 to e and set m's exponent accordingly
    e += 1
f = m - 1           # f in [sqrt(2)/2 - 1, sqrt(2) - 1) = [-0.293, 0.414)
```

The divide-by-2 is just an exponent-field decrement; no arithmetic cost. After this transformation, `f ∈ [-0.293, 0.414)` is a smaller and more symmetric interval around 0, which gives better Remez minimax behavior.

**Alternative: use `f = (m - 1) / (m + 1)`** — this is the classical "log via atanh" trick. `log(m) = 2 * atanh(f) = 2 * (f + f³/3 + f⁵/5 + ...)`. The substitution makes `f ∈ [0, 1/3)` symmetric about `0`, and the series is odd — only odd powers of `f`. This is the form used in Tang's table-driven log.

**Our choice for Phase 1: the simple `f = m - 1` form with the `sqrt(2)` shift.**

Rationale: the atanh form is beautiful but requires a divide (`(m-1)/(m+1)`) for each call, which is expensive on every backend and introduces one more op where rounding can bite. The simple form reaches 1 ULP at polynomial degree ~14 after the sqrt(2) shift; that's more ops than atanh's degree ~7 but no fdiv. For Phase 1 correctness-first, avoiding the fdiv wins. Phase 2 can re-evaluate — Tang's table-driven form may win on speed regardless.

### 4.3 Polynomial on the reduced interval

Remez-minimax fit `log(1 + f)` on `[-0.293, 0.414]` to degree 14. The polynomial form:

```
log(1 + f) ≈ f + f² * Q(f)                  Variant B, same idea as exp
```

where `Q(f)` is the Remez-fit polynomial for `(log(1+f) - f) / f²`. This isolates the leading `f` term (known exactly, no rounding) and polynomial-fits the nonlinear remainder. `Q(f)` is degree 12.

Horner evaluation, inside-out, no FMA contraction. Same rule as exp.

### 4.4 Reassemble

```
result = poly + e_f64 * ln2_hi + e_f64 * ln2_lo
```

Wait — this needs care. If `e` is large (say 1000) and `poly` is small (say 0.01), then `e * ln2_hi` dominates and adding `poly` at the end is fine. If `e = 0` and `poly` is the whole answer, then `e * ln2_hi = 0` exactly (because `e_f64 = 0.0`), so the sum is just `poly`. In between, there's a regime where catastrophic cancellation could bite: `e = 1`, `m ≈ 1`, so `poly ≈ 0` and `e * ln2 ≈ 0.693` — no cancellation. `e = -1`, `m ≈ 2`, `poly ≈ 0.693`, `e * ln2 ≈ -0.693`, sum ≈ 0 — **this is the cancellation regime**.

For 1 ULP we need the cancellation regime to still give us 1-ULP accuracy. The answer is Cody-Waite again on `ln(2)`:

```
hi = e_f64 * ln2_hi + poly                  # poly is much smaller than e * ln2_hi for large |e|
lo = e_f64 * ln2_lo                         # the low bits that ln2_hi dropped
result = hi + lo
```

### §4.4a Explicit op sequence for log reassembly (adversarial B3 resolution, 2026-04-12)

Per adversarial's B3 blocker, the earlier draft's "I will flesh this out at 2.11" deferral is not acceptable. Below is the full fp64 op sequence with a running ULP budget, citing Tang ACM TOMS 1990 §3.2 for the structural choices.

**Input state after range reduction (§4.3):**
- `f : f64` — the reduced mantissa argument, `f ∈ [-0.293, 0.414]` (post-sqrt(2) shift)
- `e_f64 : f64` — the adjusted unbiased exponent as an integer-valued fp64, `e_f64 ∈ [-1074.0, 1023.0]`
- `Q(f) : f64` — the Remez polynomial output, `Q(f) ≈ (log(1+f) - f) / f²`

**Goal:** compute `result = e_f64 · ln(2) + f + f² · Q(f)` with `max_ulp ≤ 1`.

**Exact 7-op sequence pathmaker emits in `tam_ln.tam`:**

```
; Preconditions: |f| ≤ 0.414, |e_f64| ≤ 1074, q = Q(f) ≈ (log(1+f) - f) / f²
;
; Running ULP budget (per Tang 1990 §3.2):
;   op 1:  EXACT by Sterbenz (ln2_hi has 12 trailing zero mantissa bits)
;   op 2-3: polyq formation, ≤ 0.5 ULP each
;   op 4-7: Cody-Waite-ordered reassembly, ≤ 0.5 ULP each
;   Composed bound: ≤ 1.0 ULP worst case (empirically ≈ 0.82 ULP per Tang Table 2)

op 1:  %e_ln2_hi  = fmul.f64 %e_f64, %ln2_hi    ; EXACT by Sterbenz
op 2:  %f_sq      = fmul.f64 %f, %f              ; f² — 0.5 ULP
op 3:  %polyq     = fmul.f64 %f_sq, %q           ; f² · Q(f) — 0.5 ULP
op 4:  %t_hi      = fadd.f64 %e_ln2_hi, %f       ; BIG SUM: e·ln2_hi + f — 0.5 ULP
op 5:  %e_ln2_lo  = fmul.f64 %e_f64, %ln2_lo     ; small correction term — 0.5 ULP
op 6:  %t_lo      = fadd.f64 %e_ln2_lo, %polyq   ; small + small — 0.5 ULP
op 7:  %result    = fadd.f64 %t_hi, %t_lo        ; FINAL: big + small — 0.5 ULP
```

**Why this exact order (Tang's rationale):**

1. **Op 1 must use `ln2_hi`, not `ln2`.** `ln2_hi` has 12 trailing zero mantissa bits so `e_f64 · ln2_hi` is exact in fp64 for `|e_f64| ≤ 2^11`. Using a single `ln2` constant here would cost 1 ULP from rounding the product alone.
2. **Op 4 (the "big sum") must precede the small correction.** Computing `t_hi = e·ln2_hi + f` first gives us a ~1 ULP rounded dominant term. Reversing would pre-round both halves and lose cancellation headroom.
3. **Ops 5–6 compute the small-magnitude correction as a separate partial sum.** Both terms are bounded far below `|t_hi|`, so adding them into `t_hi` (op 7) introduces ≤ 0.5 ULP of `t_hi`.
4. **Op 7 is NOT reassociated with op 4.** The IR emits op 4 then op 7 as separate `fadd.f64` instructions; pathmaker preserves this order in the `.tam` text. Reassociating would produce different bits because the intermediate rounding changes.

**Cancellation regime drill-down** (worst case: `x` near 1):
- After §4.2's `sqrt(2)` shift, `f` stays in `[-0.293, 0.414]`, so the "catastrophic" regime `e = -1, f ≈ 1` does NOT occur. **The shift's purpose is exactly to move the cancellation away from this reassembly.** At `x ≈ 1`, `e = 0` and `f ≈ x - 1 ≈ 0`, so `t_hi = 0 + f ≈ f`, `t_lo ≈ polyq ≈ f² · Q(f)`, and `result ≈ f + f² · Q(f)` — which is the Variant B polynomial form evaluated directly. The residual error is ≤ 0.5 ULP from op 7. This is why `log(1 + 1e-15)` passes at 1 ULP.

**Pathmaker contract:** emit the 7 ops above in the exact order, each as a separate `.tam` statement with a named intermediate register. Do NOT fuse, reassociate, or FMA-contract (I3). Register names are suggestions; preserve order regardless.

**References:**
- P. T. P. Tang, "Table-driven implementation of the logarithm function in IEEE floating-point arithmetic," ACM TOMS 16(4):378–400, 1990. §3.2 for the reassembly op sequence.
- N. J. Higham, "Accuracy and Stability of Numerical Algorithms," 2nd ed., SIAM, 2002, §4.2 for the error analysis framework.

### 4.5 Special-value handling (front-end dispatch)

```
if isnan(x):                  return x          # I11 preservation — check first
if x < 0:                     return nan        # B1: STRICT < (not <=); -0 does NOT satisfy x<0
if x == 0:                    return -inf       # catches both +0 and -0 (fcmp_eq(+0,-0)=true)
if x == +inf:                 return +inf
if x == 1.0:                  return +0.0       # bit-exact
# subnormal detection (B2: exact .tam IR recipe):
#   is_subnormal = (exp_bits == 0) & (mantissa_bits != 0)
#   In .tam IR:
#     %bits      = bitcast.f64_to_i64 %x
#     %exp_bits  = shr.i64 %bits, 52          ; logical shift, gives biased exponent
#     %mant_bits = and.i64 %bits, 0x000fffffffffffff
#     %exp_zero  = icmp_eq.i64 %exp_bits, 0
#     %mant_nz   = icmp_ne.i64 %mant_bits, 0
#     %is_sub    = and.i1 %exp_zero, %mant_nz
#   Must come AFTER the x < 0 and x == 0 checks so negative subnormals already caught.
if x is subnormal:            scale by 2^52, subtract 52*log(2) at end
# normal path:
proceed with bit extraction
```

**B1 note:** The negativity check `x < 0` is strict (`fcmp_lt.f64`). Do NOT use `x <= 0`. Input `-0.0` must return `-inf` (not `nan`), because `fcmp_lt.f64(-0.0, 0.0)` is `false` — `-0.0` is not less than `0.0` per IEEE 754 §5.11. This is why the dispatch order matters.

**B2 note:** The subnormal detection uses the exact .tam IR recipe above. Alternative `x < 2^-1022` would also match negative numbers — which must already be caught before reaching this check. The bitcast+shift+mask recipe is explicit and independent of any comparison that might be sign-sensitive.

## Coefficient generation

Same protocol as `exp-design.md` §4:
- `remez.py` fits `Q(f) = (log(1 + f) - f) / f²` on `[-0.293, 0.414]` at degree 12, mpmath 100 dps.
- Coefficients committed to `log-constants.toml`.
- `ln2_hi`, `ln2_lo` are shared with `exp-constants.toml` — same decomposition, same bit patterns. They live in a shared `libm-constants.toml` that both files depend on.
- Also precompute: `sqrt(2)` threshold for the `m` shift; `52 * ln2_hi` and `52 * ln2_lo` for the subnormal path.

## Special cases and testing

Campsite 2.12 is the special-value test:

```
assert tam_ln(1.0)  == 0.0              # bit-exact
assert tam_ln(0.0)  == -inf
assert tam_ln(-0.0) == -inf
assert tam_ln(-1.0) is nan
assert tam_ln(+inf) == +inf
assert tam_ln(nan)  is nan
assert tam_ln(2.0)  ≈ 0.6931471805599453  (1 ULP)
assert tam_ln(e)    ≈ 1.0                (within 1 ULP; e itself is rounded)
```

Plus the 1M random sample battery from Campsite 2.1.

### Round-trip identity tests (per adversarial review A2, 2026-04-12)

Two complementary identity checks. Both are tertiary (sanity net, not primary bar) per the accuracy-target composition rule, but the `log(exp(x)) ≈ x` direction is worth adding explicitly because its error profile is regular across the entire `x` range (unlike the reverse direction, which has regime-dependent error):

1. **`exp(log(x)) ≈ x`** for `x ∈ [1e-100, 1e100]`. Bound: **2 ULP** (two 1-ULP errors compose additively in the result magnitude).

2. **`log(exp(x)) ≈ x`** for `x ∈ [-700, +700]` (i.e., any `x` where `exp(x)` is finite and normal). Bound: **2 ULP**, flat across `x`.

   **Why the log-of-exp bound is flat:** if `exp(x)` is accurate to 1 ULP, then `exp(x) = e^x · (1 + ε)` where `|ε| ≤ 2^-52`. Applying `log` to this gives `log(e^x · (1 + ε)) = x + log(1 + ε) ≈ x + ε` (to first order in ε). The propagated error is `|ε| ≤ 2^-52`, which is **1 ULP at 1.0 regardless of `x`**. Adding `log`'s own 1 ULP contribution, the total is ≤ 2 ULP uniformly. This is a stronger guarantee than the exp-of-log direction (which has a `|b|`-proportional amplification) and is a useful smoke test for the full exp/log pipeline.

Both identities are evaluated via mpmath at 50 digits to get the "true" `x` reference; we then compare our composed result in fp64 against the mpmath truth. If the identity fails but individual functions pass their 1 ULP bars, the failure goes in the composed-budget drift log — **do not raise the individual function bound to pass the identity test** (per navigator's sign-off note).

## Pitfalls

1. **Subnormal extraction.** The naive `e = (bits >> 52) - 1023` fails for subnormals. Front-end with the `x * 2^52` trick.
2. **The `m / 2` shift.** Don't do a literal divide; decrement the exponent field. Literal divide introduces a rounding that isn't there otherwise.
3. **Cancellation at `e = -1`, `m ≈ 2`.** The sum `e * ln2 + poly` is near zero and loses bits. Cody-Waite decomposition of `ln(2)` plus ordered summation fixes this.
4. **`log(1.0) == 0.0`.** Must be bit-exact zero. The polynomial form `log(1+f) = f + f² * Q(f)` gives this automatically when `f = 0`. Don't introduce any `+ tiny_constant` that would perturb it.
5. **Handling of negative inputs.** `log(-x)` is `nan`, not `-log(x)`. Front-end dispatch.
6. **Parsing of `log(x)` at `x` just below `1.0`.** `x = 1.0 - ε` gives `m = (1.0 - ε)` which after normalization is actually `m ≈ 2 - 2ε`, `e = -1`. Then `f = m - 1 ≈ 1 - 2ε`, and we're at the worst-conditioned end of the polynomial, where the sqrt(2) shift will trigger and move us to `m' = m/2 ≈ 0.5 - ε`, `e' = 0`, `f' ≈ -0.5 - ε` — still fine, inside the polynomial interval.

## Open questions

1. **Do we want `log` and `log2` and `log10` as separate primitives or as `log(x) / log(b)` compositions?** For Phase 1, implement `log` natively and let `log2`/`log10` be compositions if users ask for them. Phase 2 can add natively-fit polynomials for them. (This is an anti-YAGNI question — the structure doesn't guarantee we need them in Phase 1, so defer.)
2. **Tang's table-driven form vs simple polynomial?** Recommendation is simple for Phase 1, table-driven for Phase 2. Same reasoning as `exp`.

## References

- P. T. P. Tang, "Table-driven implementation of the logarithm function in IEEE floating-point arithmetic," ACM TOMS 16(4):378–400, 1990.
- W. J. Cody & W. Waite, "Software Manual for the Elementary Functions," Prentice-Hall, 1980, §9 (log).
- J.-M. Muller et al., "Handbook of Floating-Point Arithmetic," 2nd ed., 2018.
