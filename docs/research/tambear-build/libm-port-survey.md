# libm port survey — winrapids draft audit

**Status**: Scout survey, 2026-05-08/09. Pre-port findings for the 17 libm files in
`R:\winrapids\crates\tambear\src\recipes\libm\`.

The formalization decision (Option A: pull now with BigFloat oracle) is in the
tambear-formalize campsite synthesis. This document records the specific findings
that architectural surgery will need to address.

---

## Coefficient-type audit: complete

All 17 files checked for Taylor vs minimax/rational-Chebyshev polynomial coefficients.

### Problem: `exp.rs` only

- `EXP_TAYLOR` / `EXP_TAYLOR_LONG`: Taylor series, f64-rounded rationals
- Reduction interval: `|r| ≤ ln(2)/2 ≈ 0.347` — too wide for Taylor to achieve <3 ULP
- Worst-case error: ~10 ULP
- `exp_correctly_rounded` claims 0 ULP; admits "true hard cases require arbitrary
  precision and are out of scope" — self-contradicting contract
- **Active remediation**: `derive_exp_minimax.py` (Tang 1989, 32-entry table,
  reduction to `|r| ≤ ln(2)/64`) and `derive_exp_constants.py` are producing
  minimax coefficients. Tang approach targets ≤3 ULP, still not correct rounding.
- **Path to actual correct rounding**: BigFloat::exp() at p=200 as Ziv fallback
  (Sweep 35 implementation list). Near-term sequence: rename → Tang coefficients
  (honest ≤3 ULP) → BigFloat::exp() → hard-case fallback.

### Clean (minimax / rational Chebyshev / fdlibm-lineage)

| File | Coefficients | Source |
|------|-------------|--------|
| `log.rs` | `LOG_COEFFS` (Lg1-Lg7) | fdlibm minimax on s-series |
| `sin.rs` / `cos.rs` | `SIN_COEFFS`, `COS_COEFFS` | Remez refit at 80-digit mpmath |
| `tan.rs` | (none — delegates to sin.rs) | — |
| `atan.rs` | `AT0..AT10` | glibc `s_atan.c` minimax, 5 sub-intervals |
| `asin.rs` | `P_S*`, `Q_S*` rational P/Q | Two digit-transposition bugs already patched |
| `erf.rs` | `PP*`, `QQ*`, `PA*`, `QA*`, `RA*`, `SA*`, `RB*`, `SB*` | fdlibm `s_erf.c` verbatim |

### Acceptable (not Remez but fit for purpose)

| File | Approach | Note |
|------|----------|------|
| `gamma.rs` | Lanczos g=7/n=9 (Pugh 2004) | ~15 digits; standard for gamma; reflection formula makes single-Remez impractical |
| `hyperbolic.rs` | SINH: 6-term fdlibm Taylor on `\|x\| < 1` | Narrow interval, acceptable; but expm1 has precision hole (see below) |

### Clean by delegation (no own polynomial coefficients)

`inv_hyperbolic.rs`, `pi_scaled.rs`, `pi_scaled_inv.rs`, `rare_trig.rs`,
`inv_recip.rs`, `sincos.rs` — pure compositions, cancellation hazards explicitly handled.

---

## Known precision bugs

### `hyperbolic.rs` — `expm1` precision hole

**Location**: `R:\winrapids\crates\tambear\src\recipes\libm\hyperbolic.rs`, lines 23–31

**Bug**: Private `expm1(x)` uses `exp_strict(x) - 1.0` for `|x| ≥ 1e-9`.
For `1e-9 ≤ |x| ≤ 1e-4`, this suffers catastrophic cancellation: `exp(x) ≈ 1 + x`
and subtracting `1.0` loses the leading 13+ bits of the result.

Error magnitude:
- `x = 1e-4`: ~12 ULP in expm1 result
- `x = 1e-6`: ~26 ULP in expm1 result
- `x = 1e-8`: ~30 ULP in expm1 result (Taylor threshold barely helps)

**Impact on `sinh_strict`**: Called with `expm1` for `|x| ≤ 1.0`. The formula
`h*(h+2)/(2*(h+1))` propagates expm1's error with ~1× magnification. The claimed
"≤2 ulps" bound is violated for `1e-9 ≤ |x| ≤ 1e-4`.

**No impact on `cosh_strict`**: Formula squares the expm1 error (`t*t`), making
the contribution sub-ULP in the final result.

**Fix**: fdlibm `s_expm1.c` approach — minimax polynomial valid to `|x| ≈ 0.4`,
then `exp(x) - 1` only for `|x| > 0.4` where cancellation is negligible.
Reference: 12 constants covering regions `[0, 0.5)`, `[0.5, 1.5)`, `[1.5, ∞)`.

### `exp.rs` — `exp_correctly_rounded` naming

**Location**: `R:\winrapids\crates\tambear\src\recipes\libm\exp.rs`, lines ~194–240

The docstring simultaneously claims "Target: 0 ulps" and admits "True hard cases
would require arbitrary precision and are out of scope." Self-contradicting.

**Fix**: Rename to `exp_dd` or `exp_low_error_dd`. The name is the contract;
the current contract is wrong. This should be the first change made before any
other exp work.

---

## Three-tier API audit: hollow tiers are systemic

~70 `_compensated` and `_correctly_rounded` entry points across 17 files.
Most are simple aliases for `_strict`.

**Only `exp` and `log` have genuinely distinct tier implementations.**

| Tier differentiation | Functions |
|---------------------|-----------|
| Real (each tier does distinct work) | `exp`, `log` |
| Hollow (all three alias `_strict`) | `sin`, `cos`, `tan`, `atan`, `asin`, `erf`, `erfc`, `gamma`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, all pi-scaled, all inv-recip, all rare-trig |

**Root cause**: Functions already at their f64 accuracy ceiling with minimax
polynomials have no meaningful "compensated" or "correctly-rounded" tier achievable
with f64 arithmetic. There is no compensated sin that gets more accuracy than
strict sin with the same algorithm family.

**Implication for port**: The three-tier API only makes sense for functions where
there is a real cost/accuracy tradeoff at f64 level. For the port to new tambear,
the honest design is:
1. Functions with real f64-level tiers (exp, log): tiered API, each tier distinct,
   `_correctly_rounded` tier reserved for BigFloat Ziv fallback when implemented
2. Functions already at f64 ceiling: single entry point, honest ULP claim

The current design (three tiers for everything, most hollow) implies differentiation
that doesn't exist and builds false confidence in `_correctly_rounded` variants.

---

## First formalization candidate: `log.rs`

Navigator synthesis identified `log.rs` as the first candidate (339 LoC, cleanest
analytic identities, no coefficient bugs, math-researcher building oracle harness).

### `log_correctly_rounded` overclaim (analogous to `exp_correctly_rounded`)

**Location**: line 67 — `const LOG_COEFFS_LONG: [f64; 7] = LOG_COEFFS;`

The "correctly-rounded" path uses the same 7-coefficient fdlibm polynomial as the
strict and compensated paths. The code's own comment acknowledges this:
> "a future pass with a proper Remez solver will extend to degree 10+ with
> minimax-optimized tails. For now, the correctly-rounded path uses the same
> polynomial as compensated but with DD reconstruction, giving ~2 ulps instead
> of the target 1."

So the function is named `log_correctly_rounded` (implying ≤1 ULP, possibly 0 ULP),
but achieves ~2 ULP because the polynomial degree is unchanged from `log_compensated`.
The DD arithmetic in the reconstruction (`dd_mul_f64`, `dd_add_f64` at lines 129-131)
is correct form — it just can't compensate for a too-coarse polynomial.

**Contrast with `exp_correctly_rounded`**: the `exp` overclaim is stronger (0 ULP
claimed, ~10 ULP delivered). The `log` overclaim is milder (~2 ULP vs claimed ≤1 ULP)
and self-documented, but still a contract violation. Both should be renamed to honest
ULP bounds before the port.

**Fix**: rename `log_correctly_rounded` to `log_dd` (matching the analogous `exp_dd`
rename). The actual correctly-rounded tier is reserved for BigFloat Ziv fallback (Sweep
35), same as `exp`.

### `log_strict` — precision leak for large `k`

**Location**: line 86 — `result + (k as f64) * LN_2_DD.hi`

For `k ≠ 0`, `log_strict` uses only the high part of the double-double `LN_2_DD`,
losing the `LN_2_DD.lo` contribution. For `k = 0` (x already near 1) this adds 0 and
is harmless. For large `k` (e.g. `x = 2^1023`), `k * LN_2_DD.lo ≈ 1e-16 · k`, which
accumulates into the result at the ULP level. This is consistent with the ≤4 ULP claim
for `log_strict` — the claim is loose enough to absorb it — but worth noting.

`log_compensated` (lines 105-107) correctly adds `k_ln2_hi + k_ln2_lo`. The fix for
`log_strict` would be to follow the compensated path for the reconstruction. Low
priority given the generous ≤4 ULP budget.

### s-series protection against the MSVC libm near-1 issue

The MSVC libm `log` degrades to 16 ULP in the [0.93, 1.07] band (see
`R:\tambear\oracle\log\disagreements\20260509-msvc-libm-log-near-one-argreduction.md`).
Root cause: argument reduction `ln(m) + e·ln(2)` at the m-near-1 boundary.

`winrapids/log.rs` uses `s = f / (2 + f)` with `f = m - 1`. Since `|s| ≤ |f| / 2`,
the s-series converges twice as fast as direct Taylor in `f`, and crucially avoids
accumulating the polynomial's error at the near-1 boundary. This is the same structural
fix fdlibm applied to avoid the boundary problem. The winrapids implementation inherits
fdlibm's protection here.

### Architectural surgery needed for port

- Replace inline compensated arithmetic with primitive calls (`two_sum`, `dd_mul`, etc.)
- Add recipe tags and `using()` parameter surface (branch-cut convention for log of
  negative inputs — the current draft handles these silently via `NaN` return)
- Rename `log_correctly_rounded` → `log_dd` before port (honest ≤2 ULP contract)
- Reserve `log_correctly_rounded` for BigFloat Ziv fallback (Sweep 35)
- Fix `log_strict` reconstruction to include `LN_2_DD.lo` (minor; within existing budget)

---

## asin.rs known bugs (already patched in draft)

For reference — these are already fixed in the winrapids draft:
- `P_S2`: digit transposition (was `2.012255…`, correct is `2.012125…`)
- `P_S5`: wrong sign and magnitude (was `-3.25e-6`, correct is `3.479e-5`)

The patches are in the current file. No action needed; note here for audit trail.
