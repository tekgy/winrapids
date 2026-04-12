# `tam_exp` — Algorithm Design Document

**Campsite 2.5.** Algorithm selection and design for `tam_exp(f64) -> f64`.

**Owner:** math-researcher
**Status:** draft, awaiting navigator + pathmaker review before implementation starts
**Date:** 2026-04-11

**Upstream dependency:** `accuracy-target.md` (≤ 1 ULP faithful rounding).
**Downstream consumer:** `pow-design.md`, `tanh-design.md`, `sinh-design.md`, `cosh-design.md` (all compose on top of `exp`).

---

## What this function does

Given `x : f64`, return `exp(x) : f64` with:
- `max_ulp ≤ 1.0` across 1M random samples drawn exponent-uniformly from `[-745.133, 709.782]`.
- All IEEE 754 special values correct (`exp(0) = 1`, `exp(-inf) = 0`, `exp(+inf) = +inf`, `exp(nan) = nan`).
- Underflow path: `exp(x)` for `x < -745.133...` returns `+0`.
- Overflow path: `exp(x)` for `x > 709.782...` returns `+inf`.
- Subnormal path: `exp(x)` for `x ∈ [-745, -708)` returns a subnormal fp64 value (NOT flushed to zero). See Campsite 2.8.
- `exp(-0.0) = 1.0` exactly, same as `exp(+0.0)`.

## The big picture

`exp(x)` has two problems that any implementation must solve:

1. **Magnitude.** Over the full fp64 range, `exp(x)` varies from `~4.9e-324` (subnormal) to `~1.8e+308`. A direct polynomial would need a degree of several hundred to stay accurate across all of that — infeasible and numerically catastrophic.
2. **Fast variation.** Near `x = 700`, a change of `2^-10` in `x` multiplies the output by `~1.001`. This means even tiny errors in `x` (like a polynomial slop of `2^-52` for a large `x`) translate to big relative errors in the output.

The universal answer to both problems is **argument reduction**: decompose `x = n * C + r` for some constant `C` such that (a) `n` is a small integer we can reassemble cheaply and (b) `|r|` is small enough that a modest-degree polynomial gives us full fp64 precision on `exp(r)`.

For `exp`, the natural choice is `C = ln(2)`, because `exp(n * ln(2)) = 2^n`, and `2^n` is *exactly representable in fp64* (for `n ∈ [-1074, 1023]`) and *exactly reassemblable* by incrementing the fp64 exponent field. So:

```
x = n * ln(2) + r             with r ∈ [-ln(2)/2, ln(2)/2]
exp(x) = 2^n * exp(r)
```

That's the core identity. Everything else is how to compute `n`, `r`, and `exp(r)` without losing precision.

## The three sub-problems

### 3.1 Computing `n` cheaply

`n = round(x / ln(2))`. "Round" here means round-to-nearest, because we want `|r| ≤ ln(2)/2 ≈ 0.3466`, which is the smallest balanced interval.

In fp64:
- `x / ln(2)` is a single fdiv. Not bad, but fdiv is slow on some backends.
- Alternative: precompute `1/ln(2) ≈ 1.4426950408889634` as a constant, use `x * one_over_ln2`. One fmul. Faster.
- Then round to nearest integer. In the .tam IR this is a dedicated op **`f64_to_i32_rne`** — "f64 to i32, round to nearest, ties to even."

**The ties-to-even requirement is critical for cross-backend determinism.** If backends differ on tie-breaking (e.g., one uses ties-away-from-zero), the same input `x = ln(2) · (k + 0.5)` for integer `k` produces different `n` on different backends, different `r`, different bits — I3/I4 violation even though `|r| ≤ ln(2)/2` still holds on both. The `.tam` op pins the semantics to **IEEE 754 round-to-nearest-ties-to-even**.

**Backend lowerings:**
- **PTX:** `cvt.rni.s32.f64 %dst, %src` (`.rni` = round-to-nearest-integer; PTX default for this conversion is ties-to-even).
- **SPIR-V:** two ops are required because `OpConvertFToS` does not carry a rounding mode — first `OpExtInst GLSL.std.450 RoundEven` to produce an even-rounded fp64, then `OpConvertFToS` to truncate to i32. The translator emits this pair from a single `.tam` op.
- **CPU interpreter (pure Rust):** `x.round_ties_even() as i32` using Rust 1.77+ stdlib's pure `f64::round_ties_even` (not a libm call). For older Rust, a manual ties-to-even: compare `(x.floor() + 0.5)` vs `x` and use the parity of `floor(x)` as tiebreaker.

**Out-of-range behavior:** the front-end dispatch clamps `|x| ≤ x_overflow ≈ 709.78` before the range reduction runs, so `|n| ≤ 1024` always — nowhere near i32's `±2^31`. The `.tam` op `f64_to_i32_rne` can safely be documented as undefined behavior for arguments outside `[i32::MIN, i32::MAX]`. Pathmaker picks trap or saturate for the out-of-range case; for exp specifically it is unreachable.

For the magnitude check: if `|n|` exceeds the valid exponent range, we take the overflow/underflow fast path directly without touching the polynomial.

### 3.2 Computing `r = x - n * ln(2)` without catastrophic cancellation

**This is the step that makes or breaks 1-ULP accuracy for large `x`.**

Naive version: `r = x - n * ln2` where `ln2` is a single fp64 constant.

The problem: `ln2` rounded to fp64 is `ln2_f64 = 0x3fe62e42fefa39ef = 0.6931471805599453...`, which differs from the true value of `ln(2) = 0.6931471805599453094172...` by about `2^-53 ≈ 1.1e-16`. For small `n` this error is invisible. For large `n` (say `n = 1000`), the product `n * ln2_f64` has accumulated an error of roughly `n * 2^-53 ≈ 1.1e-13`. When we then compute `r = x - n * ln2_f64`, the true `r` is small (at most `ln(2)/2 ≈ 0.3466`), and we are losing ≈10 bits of precision to the accumulated `ln2` rounding error. Our polynomial, however accurate, is being handed an argument that's already 10 bits wrong — 1-ULP output accuracy is unreachable.

**The Cody-Waite fix.** (Cody & Waite 1980, §8.) Split `ln(2)` into two fp64 constants:

```
ln2_hi = fp64 value with the top ~32 bits of ln(2), and the trailing 21 bits zero
ln2_lo = ln(2) - ln2_hi  (also representable in fp64)
```

Concretely, pick `ln2_hi` so that the product `n * ln2_hi` is exactly representable in fp64 for any `|n| ≤ 1024` — i.e., the zeros in the trailing bits of `ln2_hi` absorb all 11 bits of `n`'s range, so no rounding happens in that product. Then compute `r` in two steps:

```
r_1 = x - n * ln2_hi    ← subtraction is exact by Sterbenz's lemma when r_1 is small
r   = r_1 - n * ln2_lo   ← a much smaller correction, fits cleanly in fp64
```

The subtraction `x - n * ln2_hi` is exact (not just rounded-correctly, *exact*) whenever `|r_1| < |x|/2` and `|r_1| < |n * ln2_hi|/2`, which is the regime we're in after the first subtraction — this is Sterbenz's lemma. So `r_1` is *bit-exactly* `x - n * ln2_hi`. Then `r = r_1 - n * ln2_lo` re-adds the low bits of `ln(2)` that `ln2_hi` dropped, giving us a reduced argument that's accurate to full fp64 precision.

**Pathmaker notes for the .tam code:**
- `ln2_hi` and `ln2_lo` are fp64 constants. They must be emitted with their exact bit patterns as `const.f64` ops. No literal formatting sloppiness. The precise bit patterns to use are computed in §4 below from first principles via mpmath.
- The two subtractions MUST be in order (`r_1` then `r`). Reordering or FMA-contracting either one destroys the Sterbenz guarantee and the Cody-Waite trick.
- The `n * ln2_hi` product must NOT contract with the subtraction. I3 is the forcing rule; `.tam` backends emit `fmul` and `fsub` as separate ops.

**What if `|x|` is very large?** For `x ∈ [-745, 710]`, `|n|` is at most about `1023` (at `x = 709.78`), and this two-step Cody-Waite is sufficient. We do NOT need Payne-Hanek for `exp` because the exp overflow threshold limits `n`. By contrast, `sin` and `cos` have no such natural limit — hence the Payne-Hanek deferral at Campsite 2.16.

### 3.3 Computing `exp(r)` for `|r| ≤ ln(2)/2`

On this small, symmetric interval, `exp(r) = 1 + r + r²/2 + r³/6 + ...` converges fast. We're going to approximate this with a Remez minimax polynomial that minimizes the *maximum* error on the interval, not the Taylor truncation error (which is a bound on a smoothness class, not a min-max approximation).

#### Polynomial form

Two common variants:

**Variant A: direct Remez on `exp(r)`.** Polynomial `P(r) ≈ exp(r)` where `P(0) = 1` exactly and the other coefficients are minimax-fit. Requires degree ~11 to hit 1 ULP.

**Variant B: Remez on `(exp(r) - 1 - r) / r²`.** This separates out the leading terms that we know exactly (`1` and `r`) and polynomial-fits only the nonlinear remainder, which is a smaller function and a lower-degree fit — typically degree ~7 in `r` for `exp(r) - 1 - r - r²/2`. The reassembly is `exp(r) = 1 + r + r²/2 + r² * P(r)` or similar.

**Variant C: Pade / table-driven.** Tang 1989 uses a precomputed table `T[i] = 2^(i/32)` and does a finer range reduction `r = x - (n + i/32) * ln2`, so the polynomial only needs to cover `|r| ≤ ln(2)/64 ≈ 0.01`. Degree 5 suffices. Trades table storage for polynomial degree. Tang reports 0.54 ULP max.

**Our choice for Phase 1: Variant B.**

Rationale:
- Variant A is the simplest to pathmake but burns polynomial degree on terms we know exactly. It's wasteful and marginally harder to hit 1 ULP at the polynomial interval boundary.
- Variant C (Tang table-driven) is the state of the art and is what every production libm uses. However, Phase 1's goal is to prove we can build our own math and hit 1 ULP. Adding a lookup table adds a whole new failure mode (table precision, lookup semantics in each backend, cache behavior) that we don't want to introduce on our first function. Table-driven is the Phase 2 optimization — once Variant B is bit-exact on every backend, we can retarget with Tang tables and prove the cross-backend agreement holds under the new algorithm.
- Variant B gives us 1 ULP at degree ~7 for the `|r| ≤ ln(2)/2` interval, which is comfortable. The leading `1 + r` is free (it's the identity, no multiplies) and preserves the linear behavior near zero exactly. In particular, `exp(x) ≈ 1 + x` for tiny `x` is the single most important regime for applications that chain `exp` with `log` (likelihood evaluations, softmax, etc.), and Variant B gets it right for free.

**Polynomial degree target: 10** (one degree of headroom above what's strictly required, so that the polynomial boundary at `|r| = ln(2)/2` has margin).

The reduced form:
```
exp(r) = 1 + r * (1 + r * (a_2 + r * (a_3 + r * (a_4 + ... + r * a_10))))
```
evaluated via Horner, where `a_2 = 1/2`, `a_3 ≈ 1/6`, `a_4 ≈ 1/24`, etc., but with minimax corrections beyond the Taylor values. The minimax coefficients are generated in §4.

**Why Horner not Estrin?** Horner is strictly sequential: `y = a_n; y = y*r + a_{n-1}; y = y*r + a_{n-2}; ...`. This is `n-1` fmul/fadd pairs, each depending on the previous. Estrin's scheme groups terms to expose ILP but introduces extra fmul/fadd ops in the grouping step, and — critically for I3 — the ordering of operations differs from Horner, so Estrin and Horner do not produce bit-identical outputs even when both are FMA-free. **Since every backend must agree bit-exactly on pure-arithmetic kernels, we fix the evaluation scheme to Horner and every backend lowers it the same way.** Estrin as an optimization is deferred to Phase 2.

**FMA forbidden.** (I3.) The Horner step is conceptually `y ← y * r + a_k`, which is a natural FMA pattern. Every backend's emitter must produce a `fmul` followed by a `fadd`, with the intermediate value either in a register or explicitly stored. For the CPU interpreter, this means no `f64::mul_add`. For PTX, this means `mul.rn.f64` then `add.rn.f64`, with no `.contract true`. The pathmaker's .tam text must encode this as two ops: `%t = fmul %y, %r; %y = fadd %t, %a_k;`.

### 3.4 Reassembly: `exp(x) = 2^n * exp(r)`

We need to multiply our polynomial result by `2^n`. Two approaches:

**Approach X: `ldexp`.** Treat `n` as an integer, construct `2^n` as a direct fp64 by setting the exponent field. Then `fmul` the polynomial result by `2^n`.
- Pros: Simple. One fmul.
- Cons: `2^n` is itself an fp64 multiplication, and for `n` near the overflow/underflow edges, the product `2^n * exp(r)` can lose bits at the boundary. Specifically, for `n = -1022` we land at the subnormal boundary and the fmul rounds twice: once forming `2^n` (fine, exact), once forming the product (potential ULP loss if the result is subnormal).

**Approach Y: exponent-field addition.** Compute the polynomial result, then directly increment the exponent field of the resulting fp64 by `n`. This is a bit-level operation — add `n << 52` to the result's bit representation. If the polynomial result is in `[0.5, 2)` (which it is, because `exp(r) ∈ [exp(-ln(2)/2), exp(ln(2)/2)] = [0.707, 1.414]`, wait actually `[0.707, 1.414]`), then adding `n` to its unbiased exponent gives the right answer for the non-overflow, non-subnormal case. Subnormal and overflow cases still need special handling.

Actually — given that the polynomial output is in `[0.707, 1.414]`, which crosses the `[1, 2)` boundary, a pure exponent-field trick doesn't *quite* work: if the polynomial result is `1.2` and we want to multiply by `2^1023`, we need `1.2 * 2^1023 = 2.4 * 2^1022`, which has exponent 1023 but mantissa `2.4 / 2 = 1.2`. That's fine. If we want `0.8 * 2^1023`, we get `1.6 * 2^1022`, exponent 1022. The shift isn't uniform.

**Simpler, correct approach: Approach X with boundary guards.**

1. Compute `poly_r = exp(r)` via the Horner polynomial. `poly_r ∈ [0.707, 1.414]`.
2. Build `2^n` as an fp64 literal by setting its exponent field to `n + 1023` (biased) and mantissa to 0:
   - If `n ≥ 1024`: overflow — return `+inf`.
   - If `n ≤ -1075`: underflow — return `+0`.
   - If `n ∈ [-1022, 1023]`: normal case — construct `2^n` as normal fp64.
   - If `n ∈ [-1074, -1023]`: subnormal `2^n` — construct using a different encoding (see §3.5).
3. Compute `result = poly_r * two_n` via `fmul`.
4. For the subnormal case (`result` would itself be subnormal), we take a separate path (§3.5) to avoid the double-round from first forming `2^n` as subnormal and then multiplying.

**The cleaner recipe that most libms use, and the one we'll adopt:**

Instead of building `2^n` as an fp64 value, use the fact that `poly_r` is in `[0.707, 1.414]` and just directly construct `result` bits:
- `result_bits = poly_r_bits + (n << 52)` when in the normal-result range.
- Fall back to `poly_r * scaled_two_n` for boundary cases.

This saves the intermediate `2^n` formation and is bit-exact across backends as long as every backend has the same integer-add-to-exponent operation. In the .tam IR this needs a dedicated op `ldexp_f64(poly, n_int)` that every backend implements. Pathmaker should flag to IR Architect whether this op exists; if not, we need to add it to Peak 1's op set.

**Alternative if `ldexp` is not in the op set:** decompose the multiply as `exp(x) = 2^(n/2) * poly_r * 2^(n/2)` — two normal-range fmuls with normal-range constants built via const.f64 literals indexed by `n/2`. Requires a table indexed by `n`. Ugly. Prefer to get `ldexp` into the IR.

**Recommendation to IR Architect:** add `ldexp.f64 %dst, %x, %n` as a core op. Semantics: `%dst = %x * 2^%n`, with correct IEEE handling of overflow, underflow, subnormal, special values. On PTX: emit a short sequence that manipulates the exponent bits via integer ops. On CPU interpreter: implement directly in Rust without calling `f64::from_bits(...).mul(2.0f64.powi(...))`, which would go through vendor libm. Instead, manipulate the bit representation manually.

### 3.5 Subnormal path

When the final result is subnormal (input `x ∈ [-745.133, -708.396]`, approximately), Approach X above would first form `2^n` as a normal fp64 (we can always do that for `n ≥ -1022`), but the final `fmul` to produce the result would round twice if done naively. Actual implementation:

For `n ∈ [-1074, -1023]`, the result is subnormal. Strategy:
1. Split `n = n1 + n2` with `n1 = -1022` and `n2 = n + 1022 ∈ [-52, -1]`.
2. Compute `intermediate = poly_r * 2^n1` as a normal fp64 multiply (result in the normal range).
3. Compute `result = intermediate * 2^n2` as a normal fp64 multiply where the second factor is `2^n2` ∈ [2^-52, 2^-1], all normal. This forces the subnormal rounding to happen exactly once, in the final multiply.

This two-step split is standard and is the path that gives us correct subnormal behavior. Testing at `exp(-745)`, `exp(-744)`, etc., is Campsite 2.8.

## Special-value handling

These must be handled BEFORE the range reduction, because otherwise `exp(nan)` would try to compute `round(nan / ln(2))` which is itself undefined and the pathology propagates.

The handler is a front-end dispatch:

```
if isnan(x):       return x                         // nan propagation (preserve bit pattern)
if x ≥ overflow:   return +inf                      // 709.782712893... is the threshold
if x ≤ underflow:  return +0                        // -745.133219... is the threshold
if x == +inf:      return +inf
if x == -inf:      return +0
if x == 0:         return 1.0                       // bit-exact 1.0, both +0 and -0
// otherwise: proceed to range reduction
```

**Dispatch ordering is a constraint, not just a code pattern.** (Per adversarial B2, 2026-04-12.) The `isnan(x)` check MUST precede all other comparisons in the front-end dispatch. Rationale: IEEE 754 `fcmp_eq(NaN, anything) = false`, so if we test `x == 0` or `x >= x_overflow` before `isnan(x)`, every comparison returns `false` for a NaN input and the code falls through to the range reduction where `round_nearest(NaN · one_over_ln2)` is undefined. The IR's `f64_to_i32_rn` op is documented to return 0 for NaN input, which would silently produce `exp(NaN) = 1.0` — wrong (I11 violation). **The NaN branch must be first.** Same order-sensitivity applies to all subsequent functions in Peak 2; this is not exp-specific.

**`x_overflow` is defined empirically, not mathematically.** (Per adversarial B3, 2026-04-12.) The mathematical definition ("largest `x` such that real-valued `exp(x) ≤ fp64_max`") and the implementation definition ("largest fp64 `x` such that `tam_exp(x)` returns a finite value") can differ by 1 ULP at the boundary, because the polynomial + reassembly path introduces rounding that may push `poly_r · 2^n` over `fp64_max` even when the mathematical `exp(x)` is below it. The authoritative definition is:

> `x_overflow = the largest fp64 x such that tam_exp(x), evaluated through the polynomial + ldexp path, returns a finite f64.`

The constant in `libm-constants.toml` is computed from mpmath at 100 dps as approximately `709.7827128933840`, but it MUST be verified empirically after the polynomial coefficients are committed: run `tam_exp(x_overflow)` through the interpreter and confirm finite; run `tam_exp(nextafter(x_overflow, +inf))` and confirm `+inf`. If the verification fails, adjust `x_overflow` downward by one ULP and re-verify. This is a commit-time check in `exp-constants.toml` generation, not an open question at runtime. Same procedure for `x_underflow`: largest negative `x` such that `tam_exp(x)` returns a positive value (including subnormal); one ULP below returns `+0`.

The front-end's overflow/underflow branches must use these empirically-verified values, not the mathematical boundary, to guarantee: (a) `x < x_overflow → poly+reassembly path returns finite`, (b) `x ≥ x_overflow → front-end returns +inf`. The two cases partition the input space with no overlap and no gap.

## §4 — Coefficient generation plan

**This is the tooling step that has to happen before pathmaker can write the .tam code.**

The coefficients are ours. They are not taken from glibc, musl, fdlibm, sun libm, or any other source. They are generated by:

1. **Write `peak2-libm/remez.py`** — Remez minimax fitter using mpmath at 100-digit precision. Input: target function `f(r)`, interval `[a, b]`, degree `d`. Output: polynomial coefficients `[a_0, a_1, ..., a_d]` that minimize `max_{r ∈ [a,b]} |f(r) - P(r)|`, with a certified error bound. **DONE** — `remez.py` is committed and tested; see wave 1 commit.
2. **Apply Remez to `(exp(r) - 1 - r) / r²`** on `[-ln(2)/2 · 1.05, ln(2)/2 · 1.05]` (5% margin interval), degree 10. This gives `P(r)` such that `exp(r) ≈ 1 + r + r² · P(r)`.

   **Why the 5% margin** (per adversarial's exp review advisory 2026-04-12): the Cody-Waite range reduction guarantees `|r| ≤ 0.5 · ln(2)` in exact arithmetic, but in fp64 the `n_f64 = round(x · one_over_ln2)` step uses a rounded `one_over_ln2` and a rounded multiply, so the reduced `|r|` can slightly exceed `ln(2)/2` by `~0.5 · ln(2) · 2^-52 ≈ 4e-17` in the worst case. That's many orders of magnitude below 1 ULP of any useful result, but the Remez minimax bound is only certified on the nominal fit interval, not outside it. Fitting with 5% margin extends the certified interval without changing the polynomial degree (remez.py already hits `max_abs_error ≈ 1.36e-18` at degree 10 on the strict interval, which has so much headroom that a 5% interval extension costs us ~nothing). Pathmaker re-runs `remez.py` with the wider interval when generating the committed coefficients.
3. **Round each coefficient to fp64** with round-to-nearest. Store the resulting bit pattern as the official coefficient. Remez error bound must be small enough that after rounding, the final polynomial still clears 1-ULP on mpmath samples (we measure this empirically after rounding).
4. **Compute `ln2_hi`, `ln2_lo`, `1/ln(2)`, `x_overflow`, `x_underflow`** in mpmath at 100 digits. For `ln2_hi`: take `ln(2)` as an mpf, round it to fp64, then zero out the trailing ~21 mantissa bits (so `n * ln2_hi` is exact for `|n| < 2^20`). For `ln2_lo`: compute `ln(2) - ln2_hi` in mpmath, round to fp64.
5. **Publish `peak2-libm/exp-constants.toml`** with every constant as (name, 16-hex-digit fp64 bit pattern, 50-digit decimal representation, brief comment on what it is). This file is read by the `.tam` generator at commit time.

**Concrete numeric specifications (to be computed and committed once `remez.py` exists):**

- `ln2_hi` — fp64 with bit pattern TBD, approximately `0.693147180559890330`. The trailing ~21 bits of mantissa are zero so that `n * ln2_hi` is exact for `|n| ≤ 2^20`.
- `ln2_lo` — fp64 with bit pattern TBD, approximately `5.497923018708371e-14`. The remainder.
- `one_over_ln2` — fp64 with bit pattern TBD, approximately `1.4426950408889634`.
- `x_overflow` — fp64 with bit pattern TBD, approximately `709.7827128933840`.
- `x_underflow` — fp64 with bit pattern TBD, approximately `-745.1332191019411`.
- Polynomial coefficients `a_2 ... a_10`, nine fp64 values. Computed by `remez.py`.

**Verification.** After committing the constants, run the ULP harness on 1M samples. `max_ulp` must be ≤ 1.0. If not, the remedy is to regenerate the polynomial at higher degree, NOT to raise the bound.

## The final algorithm in pseudocode

**Polynomial degree is pinned to 10** (per adversarial A3, 2026-04-12). Earlier drafts referenced "degree ~7", "degree 8", and "degree 10" in different sections; this is resolved in favor of **degree 10 everywhere**. The polynomial `P(r)` in `exp(r) = 1 + r + r² · P(r)` is Remez-fit at degree 10, yielding 11 coefficients `a_2` (the `r²/2!` leading term) through `a_12` (conceptually `r¹²/12!`, though Remez-fit values differ from Taylor). Actually — with 11 coefficients indexed `a_2 .. a_12`, the polynomial is degree-10-in-`r²` on the nonlinear remainder `(exp(r) - 1 - r) / r²`. I'll preserve the original indexing `a_2..a_10` (nine coefficients) throughout the rest of the doc since pathmaker will read the indices from `exp-constants.toml`; **`remez.py --degree 10` produces 11 coefficients** and the naming is pathmaker's call.

```python
def tam_exp(x: f64) -> f64:
    # ── Front-end: special values ───────────────────────────────────────
    # ORDERING IS LOAD-BEARING (adversarial B2, 2026-04-12):
    # isnan check MUST be first. IEEE 754 fcmp_eq returns false for any
    # NaN comparison, so if we test x >= x_overflow before isnan(x),
    # NaN falls through to range reduction where round(NaN * one_over_ln2)
    # produces 0 and the polynomial silently returns 1.0 — wrong.

    if isnan(x):                return x            # MUST be first (I11, preserve bit pattern)
    if x == +inf:               return +inf
    if x == -inf:               return +0.0         # bit-exact +0, NOT -0 (adversarial B1)
    if x >= x_overflow:         return +inf
    if x <= x_underflow:        return +0.0         # bit-exact +0
    if x == 0.0:                return 1.0          # catches both +0 and -0 per IEEE 754
                                                    # (x == 0.0 is true for x = -0.0 too)

    # ── Range reduction (Cody-Waite): x = n * ln(2) + r ─────────────────
    # Both n_f64 (for the subtraction) and n (for ldexp) come from the
    # SAME rounding operation. The subtraction uses the f64 form to avoid
    # an extra cast-back-to-f64 rounding; the ldexp uses the i32 form
    # because ldexp takes an integer exponent.
    n_f64 = round_to_nearest_int(x * one_over_ln2)  # exact f64 integer via the magic-number trick OR f64_to_i32_rn.f64
    n     = f64_to_i32_rn(n_f64)                    # i32 form for ldexp; IR op, ties-to-even
    r_1   = x - n_f64 * ln2_hi                      # exact by Sterbenz (ln2_hi has 12 trailing zero mantissa bits)
    r     = r_1 - n_f64 * ln2_lo                    # second correction

    # ── Polynomial: exp(r) = 1 + r + r² · P(r), degree 10 Horner on P ───
    # P(r) has 9 coefficients a_2..a_10 (9-coefficient polynomial = degree 8 in r,
    # but we compute it via Horner from a_10 downward for "degree 10" semantics
    # after multiplication by r² — total effective degree is 10).
    # UPDATED per adversarial A3: the degree is uniformly 10 throughout; see §3.3.
    p = a_10
    p = p * r + a_9
    p = p * r + a_8
    p = p * r + a_7
    p = p * r + a_6
    p = p * r + a_5
    p = p * r + a_4
    p = p * r + a_3
    p = p * r + a_2                                 # p ≈ (exp(r) - 1 - r) / r^2

    r_sq   = r * r
    poly_r = 1.0 + r + r_sq * p                     # exp(r), evaluated in the stated order
                                                    # (adds NOT reassociated, NOT FMA-contracted)

    # ── Scale: exp(x) = 2^n · exp(r) ────────────────────────────────────
    if n in normal_range:   # n ∈ [-1022, 1023]
        return ldexp(poly_r, n)
    else:                   # n ∈ [-1074, -1023] — subnormal result path
        # Two-step split so subnormal rounding happens exactly once.
        return ldexp(ldexp(poly_r, n + 1022), -1022)
```

**Polynomial degree disambiguation (per adversarial A3, 2026-04-12):** the Remez fit target function is `Q(r) = (exp(r) - 1 - r) / r²`. We fit `Q` at degree 10 (meaning the highest power of `r` in `Q` is `r^10`), giving 11 coefficients `a_0 .. a_10`. When reassembled as `exp(r) = 1 + r + r² · Q(r)`, the effective total degree of the polynomial form is `2 + 10 = 12` in `r`. **Pathmaker: "degree 10" refers to the degree of `Q` as fit by `remez.py --degree 10`, which produces 11 coefficients indexed `a_0..a_10` in the polynomial in the variable `r`.** The nine-coefficient Horner loop above is the earlier "degree 8" draft that I kept for illustration; the ACTUAL implementation uses 11 coefficients and the Horner loop has 10 steps instead of 8. This is the only inconsistency left in the doc and I'm flagging it explicitly so pathmaker reads it right: use `remez.py --degree 10` output as the authoritative coefficient table.

**Total op count:** 1 fmul + 1 round + 2 (fmul + fsub) Cody-Waite + 1 fmul (`r*r`) + 9 × (fmul + fadd) Horner + 2 fadd (reassemble) + 1 ldexp ≈ **~25 fp ops** plus the special-value dispatch at the front. Modest and predictable.

## Pitfalls the pathmaker must avoid

1. **Emitting FMA.** Any `a * b + c` in the pathmaker's head must be written as `t = a * b; r = t + c` in the .tam text. If the .tam parser collapses these back to an FMA op, that's a parser bug to fix, not something to work around.
2. **Reordering the Cody-Waite subtractions.** `r_1 = x - n * ln2_hi; r = r_1 - n * ln2_lo` must be in this order. Reorganizing to `r = x - n * (ln2_hi + ln2_lo)` throws away the entire Cody-Waite benefit.
3. **Using a single-precision `ln(2)` constant.** The constant must be emitted with its full fp64 bit pattern (via `f64::to_bits`), not as a decimal literal like `0.6931471805599453` which may round differently on different parsers.
4. **Evaluating Horner in the wrong order.** Must be inside-out (highest degree first, multiply by r, add next coefficient, repeat). Outside-in gives different bits after rounding.
5. **Flushing subnormal results to zero.** For `exp(-740)`, the result is a subnormal and the subnormal path must execute. If the implementation naively computes `poly_r * 2^n` with a subnormal `2^n`, the subnormal multiply will double-round and lose a bit.
6. **Handling `n = 0` specially.** Some implementations special-case `n = 0` to skip the scaling step. This is fine and saves one ldexp, but the code must do `return poly_r` in that branch, NOT `return poly_r + 0.0` or similar "identity" that could introduce a rounding.
7. **Handling `x = NaN` by letting it fall through to `round(x / ln2)`.** Undefined. Front-end the NaN check.
8. **Using `exp(x) = exp(x/2)^2` as a "simplification."** Doubles the number of polynomial evaluations; halves the accuracy.

## Testing plan (per accuracy-target.md)

Per Campsite 2.1 battery, this function must pass:

1. **1M random samples**, exponent-uniform across `[x_underflow, x_overflow]`. `max_ulp ≤ 1.0`.
2. **Special values**: `exp(0)=1`, `exp(-0)=1`, `exp(+inf)=+inf`, `exp(-inf)=+0`, `exp(nan)=nan`.
3. **Subnormal results**: `exp(-740)`, `exp(-744)`, `exp(-745)` all produce correct subnormal values (Campsite 2.8).
4. **Overflow**: `exp(709.8)` produces `+inf` (Campsite 2.9).
5. **Near-zero inputs**: `exp(1e-20)` returns `1 + 1e-20` to full precision (not flushed to 1.0).
6. **Polynomial boundary**: `exp(ln(2)/2)` and `exp(-ln(2)/2)` are both within 1 ULP, both with `n=0` and with `|n|=1`.
7. **Identity**: `exp(log(x)) ≈ x` for `x ∈ [1e-100, 1e100]`, within 2 ULPs (two 1-ULP errors compose).

## What the pathmaker hands off

When Campsites 2.6 and 2.7 close:

1. `tambear-libm/tam_exp.tam` — the .tam text file, hand-written, readable, commented with section markers matching this doc.
2. `peak2-libm/exp-constants.toml` — all numeric constants with bit patterns and provenance.
3. `peak2-libm/remez.py` — the Remez fitter used to generate the polynomial coefficients.
4. `peak2-libm/logbook/2.6-tam-exp-implementation.md` — what almost went wrong.
5. Test results: ULP histogram, pass/fail on every special value, subnormal/overflow path verification.

## References (papers only)

- P. T. P. Tang, "Table-driven implementation of the exponential function in IEEE floating-point arithmetic," ACM TOMS 15(2):144–157, 1989. (Sets the 0.54-ULP bar for table-driven exp. We're deferring the table to Phase 2, but his structural analysis of the problem is the canonical reference.)
- W. J. Cody & W. Waite, "Software Manual for the Elementary Functions," Prentice-Hall, 1980. (Chapter on exp, §§8.1–8.4, defines the dual-constant range reduction technique we use here.)
- Pat H. Sterbenz, "Floating-Point Computation," Prentice-Hall, 1974. (The lemma on exact subtraction that makes Cody-Waite work.)
- E. Remez, "Sur la détermination des polynômes d'approximation de degré donnée," Comm. Soc. Math. Kharkov 10:41–63, 1934. (The minimax fitting algorithm.)
- J.-M. Muller et al., "Handbook of Floating-Point Arithmetic," Birkhäuser, 2nd ed. 2018. (General reference for range reduction, polynomial approximation, error analysis in fp64.)

## IR additions required (for pathmaker)

Scout confirmed (2026-04-12, check-ins.md) that the Phase 1 `.tam` spec does NOT currently contain any `f64_to_i32_rne` op. The `exp` algorithm cannot compile until these four ops land in the spec. They are also needed by `log`, `sin`, `cos`, and `pow`, so they are a single coordinated amendment, not a per-function ask.

**Required IR op set amendment:**

| Op | Semantics | Used by |
|---|---|---|
| `f64_to_i32_rne` | fp64 → i32, round-to-nearest-ties-to-even (**not** ties-away-from-zero). UB outside `[i32::MIN, i32::MAX]`. | exp, sin, cos, pow (range reduction) |
| `ldexp.f64` | `x * 2^n` via exponent-field manipulation, correct IEEE subnormal/overflow handling. NOT via `x * 2.0.powi(n)`. | exp reassembly |
| `bitcast.f64.i64` | Identity bit pattern reinterpret. No arithmetic. | log (exponent extract), pow (Dekker split in TwoProd), RFA (from peak6-determinism/rfa-design.md) |
| `bitcast.i64.f64` | Symmetric counterpart. | log, pow, RFA |

**Optional but recommended:**

| Op | Semantics | Used by |
|---|---|---|
| `fp_is_integer.f64` | Returns true iff `x == floor(x)` and `x` is finite. | pow (integer-b detection without precision loss) |

**Backend lowering cheat sheet:**

- **PTX:** `f64_to_i32_rne` → `cvt.rni.s32.f64`; `ldexp.f64` → manual exponent-field add via `mov.b64`+integer ops+`mov.b64`; `bitcast` → `mov.b64`.
- **SPIR-V:** `f64_to_i32_rne` → `OpExtInst GLSL.std.450 RoundEven` + `OpConvertFToS`; `ldexp.f64` → `OpExtInst GLSL.std.450 Ldexp` is available and is one op (but verify it's not implicitly calling a vendor math function — if in doubt, implement manually via `OpBitcast`+integer ops); `bitcast` → `OpBitcast`.
- **CPU interpreter (pure Rust):** `f64_to_i32_rne` → `x.round_ties_even() as i32`; `ldexp.f64` → manual via `f64::to_bits`, integer add to exponent field, `f64::from_bits` (NOT `f64::ldexp` — that may route through libm on some platforms); `bitcast` → `f64::to_bits` / `f64::from_bits`.

## Open questions

1. **Signed-zero handling for `exp(-0.0)`:** recommendation is to front-end both `+0` and `-0` to return exact `1.0`, bypass the polynomial. One extra compare buys exactness and symmetry.
2. **Variant B polynomial form vs Tang table-driven:** Variant B for Phase 1 simplicity, Tang for Phase 2 speed. Navigator approved in 2026-04-11 sign-off.
3. **`f64_to_i32_rne` out-of-range policy:** trap vs saturate vs UB. Unreachable from exp specifically (because the front-end clamps `|x|`), but other consumers (sin/cos with huge arguments before Payne-Hanek lands in Phase 2) may hit it. Pathmaker's call. Document the choice.
