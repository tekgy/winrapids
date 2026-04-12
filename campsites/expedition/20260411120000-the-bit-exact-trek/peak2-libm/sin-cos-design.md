# `tam_sin` and `tam_cos` — Algorithm Design Document

**Campsite 2.13.** Joint design doc for sin and cos because they share the range reduction pipeline and the quadrant dispatch.

**Owner:** math-researcher
**Status:** draft, awaiting navigator + pathmaker review
**Date:** 2026-04-11

**Upstream dependency:** `accuracy-target.md`. Same Cody-Waite and Horner patterns as `exp` and `log`. **Phase 1 scope is `|x| ≤ 2^30`** via three-term Cody-Waite reduction of π/2. Large-argument trig beyond `2^30` is Phase 2 (Payne-Hanek, Campsite 2.16). (Amended 2026-04-12 per navigator's "pick 2^30 and commit" direction — the prior draft said 2^20 in the introduction and 2^30 in the testing section; 2^30 is now the consistent claim.)

---

## What these functions do

Given `x : f64`, return `sin(x)` / `cos(x)` : f64 with:
- `max_ulp ≤ 1.0` across 1M random samples drawn exponent-uniformly from `[-2^30, 2^30]`.
- IEEE specials:
  - `sin(0) = +0` bit-exact. `sin(-0) = -0` bit-exact (odd function preserves sign of zero).
  - `cos(0) = 1.0` bit-exact. `cos(-0) = 1.0` bit-exact (even function).
  - `sin(±inf) = nan` (undefined — `sin` has no limit at infinity), same for `cos`.
  - `sin(nan) = nan`, `cos(nan) = nan`.
- For `|x| > 2^30`: Phase 1 returns `nan` and documents this as out-of-domain. Phase 2 will handle via Payne-Hanek.

## The big picture

`sin` and `cos` are periodic with period `2π`. The universal approach is to reduce the argument `x` modulo `π/2` (quarter-period), track which quadrant we land in as an integer `k`, and then evaluate one of four variants:

```
k mod 4 == 0:  sin(x) =  sin(r),   cos(x) =  cos(r)
k mod 4 == 1:  sin(x) =  cos(r),   cos(x) = -sin(r)
k mod 4 == 2:  sin(x) = -sin(r),   cos(x) = -cos(r)
k mod 4 == 3:  sin(x) = -cos(r),   cos(x) =  sin(r)
```

where `r = x - k * (π/2)` is the reduced argument in `[-π/4, π/4]`.

So we have two polynomials — `sin_poly(r)` on `[-π/4, π/4]` and `cos_poly(r)` on the same interval — and a quadrant dispatch that picks between them with a sign.

**Why reduce to `[-π/4, π/4]` and not `[-π/2, π/2]`?** The polynomial for `sin` on a wider interval needs higher degree to hit 1 ULP. `[-π/4, π/4]` balances both polys and keeps them at degree ~8 (sin) and ~7 (cos).

## The hard part: range reduction

Phase 1 uses **three-term Cody-Waite reduction of π/2**, giving `|x| ≤ 2^30 ≈ 1e9` as the supported domain. (Committed 2026-04-12 per navigator's "pick 2^30 and commit" direction.)

```
k    = round(x / (π/2))              # integer, |k| ≤ 2^31
r_1  = x - k * piover2_hi            # exact by Sterbenz (piover2_hi has 30 trailing zero mantissa bits)
r_2  = r_1 - k * piover2_mid         # second correction (piover2_mid also has 30 trailing zeros)
r    = r_2 - k * piover2_lo          # third correction, captures remaining bits of π/2
```

**Why three parts.** Each `piover2_hi` and `piover2_mid` has 30 trailing zero mantissa bits, so `k * piover2_{hi,mid}` is exact in fp64 for `|k| ≤ 2^30` (which follows from `|x| ≤ 2^30 → |k| = |round(x · 2/π)| ≤ 2^31 · (2/π) < 2^30.34`, comfortably within the 30-zero window with margin). The three-term split gives `piover2_hi + piover2_mid + piover2_lo ≈ π/2` to ~120 bits of total precision (30 preserved bits per split × 3 + some overlap), which is the right amount for `|x| ≤ 2^30`.

The three constants are committed in `libm-constants.toml` as `piover2_hi`, `piover2_mid`, `piover2_lo`, computed by `gen-constants.py` from mpmath at 100 dps. Their exact bit patterns:
- `piover2_hi  = 0x3ff921fb40000000`  (1.5707962512969970703125 — low 30 mantissa bits zero)
- `piover2_mid = 0x3e74442d00000000`  (7.549789415861596e-08  — low 30 mantissa bits zero)
- `piover2_lo  = 0x3cf8469898cc5170`  (5.390302858158119e-15)

**Why not two-term (2^20 ceiling).** Two-term Cody-Waite limits us to `|x| ≤ 2^20 ≈ 1e6`. That's fine for applications that only ever feed small arguments, but it's weaker than the three-term version for effectively zero implementation cost (one more `fmul` + `fsub` per call — two ops). The three-term form is uniformly better and the testing section already assumes it. The earlier draft had 2^20 in the introduction and 2^30 in the testing section; that inconsistency is now resolved in favor of 2^30.

**Why three-term is NOT enough for `|x| > 2^30`.** For larger `x`, either `piover2_{hi,mid}` need more trailing zeros (reducing their precision and `r_1`/`r_2`'s precision) or `piover2_lo` gets big enough that the third subtraction no longer captures the residual cleanly. At `|x| ~ 2^53`, you need `π/2` to ~160 bits to reduce correctly — Payne-Hanek territory. Phase 1 caps at `2^30`.

**The precise Phase 1 cut-off:** `|x| ≤ 2^30 - 1`. For `|x| ≥ 2^30`, return `nan` with the documented "out of Phase 1 domain" rationale. See Campsite 2.16.

The three constants are computed in mpmath at 100 dps:
- `piover2_hi`: the top ~30 bits of `π/2` as an fp64 with trailing mantissa bits zeroed
- `piover2_mid`: the next ~30 bits as a fp64
- `piover2_lo`: the rest, also as fp64

They live in `libm-constants.toml` and are generated once by `gen-constants.py`, not hand-typed.

## The polynomials

### sin polynomial on [-π/4, π/4]

Remez-minimax fit of `sin(r) = r - r³/6 + r⁵/120 - r⁷/5040 + ...` on `[-π/4, π/4]`. Since `sin` is odd, the polynomial has only odd powers:

```
sin(r) = r + r³ * S(r²)             S is a polynomial in r² of degree 4
       = r * (1 + r² * S(r²))        factored form, slightly better numerically
```

Degree 4 in `r²` (so effective degree 9 in `r`) is enough for 1 ULP on `[-π/4, π/4]`. Concretely:

```
sin(r) ≈ r + r * r² * (s1 + r² * (s2 + r² * (s3 + r² * (s4 + r² * s5))))
```

where `s1 ≈ -1/6`, `s2 ≈ 1/120`, ..., but with Remez corrections.

### cos polynomial on [-π/4, π/4]

Remez-minimax fit of `cos(r) = 1 - r²/2 + r⁴/24 - r⁶/720 + ...` on `[-π/4, π/4]`. Even function, only even powers:

```
cos(r) = 1 + r² * C(r²)             C is a polynomial in r² of degree 4
       = 1 - r²/2 + r⁴ * C'(r²)     factored form for numerical stability near r=0
```

We use the second form because near `r = 0`, `cos(r) ≈ 1 - r²/2` and we want that leading subtraction to happen exactly and cheaply — putting the `- r²/2` in the polynomial means it's one of the Horner steps and inherits a small error. Split out:

```
cos(r) ≈ 1.0 - 0.5 * r² + r⁴ * (c2 + r² * (c3 + r² * (c4 + r² * (c5 + r² * c6))))
```

Wait — that's subtle. `1.0 - 0.5 * r²` is a subtraction where cancellation becomes an issue for `|r|` near `π/4` (`r² ≈ 0.617`), where `0.5 * r² ≈ 0.3` and we're computing `1 - 0.3 = 0.7` — no cancellation. At `|r|` near `0`, `0.5 * r² ≈ 0` and we get `1.0` — fine. Cancellation would happen if `0.5 * r² ≈ 1`, which requires `r ≈ sqrt(2)`, outside our interval. So the simple form is safe.

For the higher-order polynomial `C'(r²)`, Remez fit to degree 4 in `r²` (effective degree 10 in `r`).

**Subtle point:** for 1 ULP accuracy near `r = 0`, the `1.0 - 0.5 * r²` must be evaluated as a single step, not combined with the polynomial. If we write `cos(r) = 1.0 - 0.5 * r² + r⁴ * C'`, the evaluation order is:
```
r2 = r * r
r4 = r2 * r2
c_poly = Horner(C', r2)
result = 1.0 - 0.5 * r2 + r4 * c_poly
```
The last line, evaluated in-order, computes `1.0 - 0.5 * r2` first (good), then adds `r4 * c_poly` (fine — smaller magnitude, no cancellation). Bit-exact across backends if we pin the order. Pathmaker writes it as two explicit fadds with a named intermediate register.

## Quadrant dispatch

After range reduction, we have `k : i32` and `r : f64 ∈ [-π/4, π/4]`. The dispatch:

```python
k_low = k & 3      # mod 4

if k_low == 0:
    sin_out = sin_poly(r)
    cos_out = cos_poly(r)
elif k_low == 1:
    sin_out = cos_poly(r)
    cos_out = -sin_poly(r)
elif k_low == 2:
    sin_out = -sin_poly(r)
    cos_out = -cos_poly(r)
else:  # k_low == 3
    sin_out = -cos_poly(r)
    cos_out = sin_poly(r)
```

Implemented via branching in .tam IR. (Predicated select could also work — for Phase 1, branches are fine, the branch mispredict cost doesn't matter at interpreter speed.)

**Important:** For `tam_sin(x)`, we only need `sin_out`; for `tam_cos(x)`, only `cos_out`. But the dispatch calls the *other* polynomial in odd quadrants — so `tam_sin(x)` in quadrant 1 actually evaluates `cos_poly(r)`. This is not a bug; it's the identity `sin(r + π/2) = cos(r)` expressed computationally. Pathmaker must get this right; the test battery includes samples that land in every quadrant.

## Special-value handling

**Front-end dispatch (before range reduction):**

```
if isnan(x):                        return nan  (preserving bit pattern)
if isinf(x):                        return nan  (sin/cos at infinity is undefined)
if x == +0.0:                       sin: return +0.0,  cos: return 1.0
if x == -0.0:                       sin: return -0.0,  cos: return 1.0
if |x| > 2^30:                      return nan  (Phase 1 out-of-domain)
# otherwise: range reduce and dispatch
```

Note `sin(-0.0) = -0.0`. This is the IEEE 754 rule for odd functions and is what every serious libm returns. Our front-end preserves it.

## Coefficient generation

Same protocol as `exp` and `log`:
- `remez.py` fits `S` and `C'` at high precision.
- Coefficients in `sin-constants.toml` and `cos-constants.toml` (or a shared `trig-constants.toml`).
- `piover2_hi`, `piover2_mid`, `piover2_lo` in `libm-constants.toml`.

## Pitfalls

1. **Two-term fallback (which we are NOT using).** A two-term Cody-Waite split would cap the domain at `|x| < 2^20 ≈ 1e6`, which is weaker than Phase 1's `|x| ≤ 2^30`. Phase 1 uses three-term Cody-Waite uniformly; do not regress to two-term under optimization pressure.
2. **Quadrant dispatch off by one.** Easy to mis-map `k mod 4` to the trig identities. Verified by testing: `sin(π/2) = 1`, `cos(π/2) = 0`, `sin(π) = 0`, `cos(π) = -1`, `sin(3π/2) = -1`, `cos(3π/2) = 0`. Each of these hits a different quadrant.
3. **Polynomial boundary at `|r| = π/4`.** The Remez fit is done on `[-π/4, π/4]`, but after rounding `k` to nearest and subtracting `k * π/2`, `|r|` can be very slightly above `π/4` due to rounding in the reduction. Fit the polynomial on a slightly wider interval, say `[-π/4 * 1.01, π/4 * 1.01]`, to give margin.
4. **`sin(0) = 0` bit-exact.** The polynomial form `r + r * r² * S` gives `0` exactly when `r = 0`. Don't introduce any spurious `+ 0.0` that could perturb the sign.
5. **Preserving sign of zero for `sin(-0)`.** `sin(-0) = -0`. The polynomial form preserves this automatically because each term is an odd power of `r` times a positive coefficient (or negated thereof). Don't introduce any `fabs` that would strip the sign.
6. **FMA contraction in the Horner steps.** Same rule as exp and log. No contraction.
7. **Subnormal `r`.** If `x` is near an integer multiple of `π/2`, after reduction `r` can be subnormal (e.g., `x = π/2 * 1e6` gives `r ≈ 0` to the precision of `π/2`, which at 30-term Cody-Waite is ~82 bits, so `r` could be around `2^-82` — subnormal territory doesn't start until `2^-1022`, so we're fine here). The polynomial is linear-leading, so subnormal `r` gives subnormal `sin(r)`, which is correct.
8. **Performance note.** A branching quadrant dispatch is slower in branch-mispredict terms, but Phase 1 is correctness-first. Do not optimize with predicated selects yet.

## Testing

- 1M random samples, exponent-uniform in `[-2^30, 2^30]`.
- Adversarial: samples at `k * π/4` for `k ∈ [-2^30, 2^30]` (sampled sparsely — the full integer sweep is `2^31` values, so adversarial picks ~10^4 across that range). These land at exact quadrant/octant boundaries and stress the dispatch.
- Special values: `sin(0)`, `sin(-0)`, `cos(0)`, `sin(π/2)`, `cos(π/2)`, `sin(π)`, `cos(π)`, etc. Note these aren't bit-exact identities in fp64 because `π/2` itself is rounded — but they're within 1 ULP.
- Identity: `sin²(x) + cos²(x) ≈ 1` within a composed 3-ULP bound.
- Identity: `sin(-x) == -sin(x)` bit-exact (odd symmetry).
- Identity: `cos(-x) == cos(x)` bit-exact (even symmetry).

## Open questions

1. **Two-term or three-term Cody-Waite for Phase 1?** Recommendation: three-term, for `|x| ≤ 2^30`.
2. **Branch-based or select-based quadrant dispatch?** Recommendation: branch for Phase 1 simplicity.
3. **Shared polynomial for sin and cos?** Could factor common subexpressions (`r * r`, `r4 = r²²`). Small win, slight code complexity. Defer to Phase 2.

## References

- W. J. Cody & W. Waite, "Software Manual for the Elementary Functions," Prentice-Hall, 1980, §§3–4 (sin/cos).
- M. H. Payne & R. N. Hanek, "Radian reduction for trigonometric functions," ACM SIGNUM 18(1):19–24, 1983. (For Phase 2 big-arg reduction.)
- S. Boldo, M. Daumas, R.-C. Li, "Formally verified argument reduction with a fused multiply-add," IEEE TC 58(8):1139–1145, 2009. (Relevant if we ever allow FMA — we don't, but it's the state-of-the-art formal reference.)
- J.-M. Muller et al., "Handbook of Floating-Point Arithmetic," 2nd ed., 2018, Chapter 11.
