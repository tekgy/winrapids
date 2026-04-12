# Adversarial Review — `tan-design.md`

**Reviewer:** adversarial mathematician
**Date:** 2026-04-12
**Subject:** tan-design.md (2026-04-12 draft by math-researcher)

---

## Summary

Three blocking issues and two advisories.

The design's core choice — `tan = sin/cos` composition — is sound. The blockers are:
(B1) the signed-zero fast-path is structurally broken in the same way as sin/cos B2;
(B2) the 2-ULP accuracy claim is made over the whole domain without excluding pole neighborhoods, making it unmeasurable;
(B3) the special-values-matrix.md is not updated with tan's behavior.

None of these are design errors in the algorithm — they are specification gaps that the pathmaker cannot resolve without guidance.

---

## Blocking Issues

### B1 — Signed-zero front-end check cannot distinguish +0 from -0

**Location:** Algorithm §, special-value handling pseudocode:
```
if x == +0:  return +0        ; sign-preserving
if x == -0:  return -0        ; sign-preserving
```

**Problem:** `fcmp_eq(-0.0, 0.0)` is `true` in IEEE 754 — negative zero compares equal to positive zero. The two-branch structure shown is unreachable: `-0.0` will be caught by the first branch (`x == +0` fires), and the correct return `+0` will be returned for an input of `-0`. The result: `tan(-0)` returns `+0` instead of `-0`. This violates the standard convention that tan is an odd function (`tan(-x) = -tan(x)` exactly for `-0`).

**Same bug:** This is the same structural issue identified in sin/cos adversarial review B2. The fix there was to use `return x` (sign-preserving) instead of `return ±0.0_CONST`.

**Fix:** The zero fast-path must use `return x`, not `return ±0_constant`. Specifically:
```
if isnan(x):       return x
if isinf(x):       return nan
if fcmp_eq(x, 0):  return x    ← sign-preserving: returns -0 when x is -0
; rest of implementation
```

OR: elide the front-end zero check entirely and rely on the composition.

**Verification that elision is safe:** `tam_sin(-0) = -0` (by sin/cos spec, which uses `return x` for the zero case). `tam_cos(-0) = 1.0` (cos is even; returns exact 1.0). `fdiv(-0.0, 1.0) = -0.0` in IEEE 754. So the composition path gives `tan(-0) = -0 / 1.0 = -0` correctly. Both approaches work; either must be stated explicitly and the pseudocode corrected.

**Why this matters beyond signed zero:** If the signed-zero check fires first and returns `+0` for `-0` input, and a backend implements this front-end in a different order (checking `-0` before `+0` via bit-pattern rather than `fcmp_eq`), the backends disagree on the sign of `tan(-0)`. Since the bit-exact tests in `special-values-matrix.md` include signed-zero cases, this fires as a bit-exact failure at campsite 4.8.

---

### B2 — 2-ULP accuracy claim is unmeasurable without pole exclusion

**Location:** Accuracy target §:
> "Phase 1 bound: `max_ulp ≤ 2.0` on 1M random samples, exponent-uniform over `|x| ≤ 2^30`"

**Problem:** Near the poles `x = (k + 1/2)π`, `cos(x) → 0` and `tan(x) → ±∞`. The ULP metric for a result near ±inf is undefined — ULP distance is only meaningful for finite results. More precisely: the "ULP error" of `tan(x)` near a pole is determined by the error in `x`'s representation of the true argument, magnified by the function's derivative (which is `1 + tan²(x)` and diverges at the poles). A 2-ULP bound on the output is not achievable near poles — it requires 1-ULP accuracy in `x` propagated through a function with unbounded derivative.

The accuracy-target.md already hedges this (`tam_tan` row: "pathological near π/2+kπ"), but the design doc states "2 ULP" as a flat bound without specifying the excluded neighborhood.

**This matters for the oracle:** the oracle runner measures `max_ulp` over the injection set and the random sample set. If the random samples include points near poles, the max_ulp result is undefined (comparing a huge finite number to ±inf, or comparing two huge finite numbers that differ enormously). The runner needs to know which inputs to exclude.

**The compositional argument given in the design** ("1 ULP each from sin and cos + 0.5 ULP from fdiv ≈ 2 ULP") applies in the *smooth* region where `|tan(x)|` is bounded. Near poles it doesn't apply.

**Fix required (two parts):**

1. Add a clause to the accuracy target: "The 2-ULP bound applies on `|x| ≤ 2^30` excluding a `δ`-neighborhood of each pole `(k + 1/2)π`, where `δ` is TBD — suggested: `|cos(x)| < 2^-20` (approximately within `2^-20` of the pole). In that neighborhood, the result is defined by IEEE 754 (fdiv gives ±inf or a large finite), but no ULP bound is claimed."

2. Add to the oracle entry for tan (future `oracles/tam_tan.toml`): the injection set must exclude pole neighborhoods, and the random sampling strategy must either reject samples near poles or specifically note that pole-adjacent samples are checked for finiteness/sign only, not ULP distance.

**The exact threshold is math-researcher's call.** A reasonable default: exclude `x` where `|cos(x_f64)| < 2^-26` (approximately within 1 ULP of a pole). This is a tiny fraction of the domain and doesn't meaningfully reduce coverage. The exclusion boundary must appear explicitly in the design doc.

---

### B3 — `special-values-matrix.md` not updated for `tan`

**Location:** The special-values-matrix.md currently has no `tan(x)` column. The tan-design.md says "Per adversarial's special-values matrix `tan` column" in the Testing section, but that column does not yet exist.

**Problem:** The pathmaker cannot write the front-end dispatch without knowing the complete special-value table. The testing section lists the values but they are not in the matrix, which is the authoritative specification that ALL roles (pathmaker, scientist, test oracle) read.

**Specific cases that need matrix entries for `tan`:**

| Input | Expected output | Rationale |
|---|---|---|
| `+0.0` | `+0.0` (bit-exact) | tan is odd; zero of same sign |
| `-0.0` | `-0.0` (bit-exact) | same |
| `+inf` | `nan` | tan(±inf) undefined per IEEE 754 |
| `-inf` | `nan` | same |
| `nan` | `nan` (I11: preserve payload) | NaN propagation |
| `|x| > 2^30` | `nan` | Out of domain (Phase 1 spec cap) |
| `x = π/4` | `≈ 1.0`, within 2 ULP | Primary domain accuracy |
| `x = f64::consts::PI / 2` | `~1.633e16` (not inf) | Pole behavior via exact f64 |
| `x = -f64::consts::PI / 2` | `~-1.633e16` (not -inf) | Pole behavior, negative side |

**Action required:** Math-researcher adds a `tan` column to `special-values-matrix.md` before campsite 2.20 (or wherever tan implementation is scheduled). The matrix is the contract; the testing section in the design doc is a reminder.

Note: `tan(π/4) ≈ 1.0` but not exactly 1.0 — `sin(π/4)` and `cos(π/4)` are both `1/√2` rounded to fp64, and their quotient rounds to something within 1 ULP of 1.0 but is not bit-exact 1.0. Verify via mpmath.

---

## Advisories (non-blocking)

### A1 — The `|x| > 2^30 → nan` boundary is arbitrary and should be documented in accuracy-target.md

The design inherits the `2^30` domain cap from sin/cos by reference, but this cap applies because the Payne-Hanek argument reduction for large trig arguments is deferred to Phase 2. The cap is correct, but it should appear explicitly in `accuracy-target.md`'s tan row, not just "deferred." Proposed addition:

```
| tam_tan | |x| ≤ 2^30 | exponent-uniform excluding pole δ-neighborhoods | 2 ULP | n/a — deferred to Phase 2 for both large arg and pole-neighborhood accuracy |
```

The domain cap is load-bearing for the oracle: a random sample at `x = 1e15` would hit Payne-Hanek territory. Without the domain restriction explicitly in the primary domain column, the oracle runner might sample there.

### A2 — Phase 2 upgrade path for tan is not specified

The design doesn't mention how Phase 2 would improve tan. For context: Phase 2 options include (a) dedicated polynomial with shared sin/cos range reduction to drop to 1 ULP, or (b) keeping the composition but adding double-double arithmetic on the quotient near poles to tighten the bound in the pole neighborhood. Neither approach is required now, but a sentence in the design doc ("Phase 2 direction: dedicated polynomial on the reduced interval, see sin-cos-design.md §Phase 2 notes") would help the next implementer.

---

## I11 coverage

The design correctly handles NaN: `if isnan(x): return x` is the first front-end check. The composition also preserves NaN: `sin(nan)` returns nan (per sin/cos spec), `cos(nan)` returns nan, `nan / nan = nan` in IEEE 754. So NaN propagates even if the front-end check is removed.

Confirming: the I11 NaN propagation check for tan passes by construction from the composition.

---

## Overall verdict

**Hold for B1 and B2.** B3 is a documentation action with no implementation ambiguity, but it should be done before the pathmaker starts so the matrix and the code stay synchronized.

The algorithm (sin/cos composition) is the right choice for Phase 1. The blockers are specification gaps, not algorithm flaws. All three can be resolved quickly by math-researcher:
- B1: change `return +0` / `return -0` to `return x` in the pseudocode (or add "elide the zero check" as an explicit option).
- B2: add a pole-exclusion clause with a threshold to the accuracy target.
- B3: add the tan column to special-values-matrix.md.

After those three fixes, the design can be signed off and the campsite can proceed.

---

## What is NOT a blocker

- The choice to use `sin/cos` composition rather than a dedicated polynomial. Sound for Phase 1. Documented well.
- The "no new IR stub" decision. Correct — tan requires no IR extension beyond what sin/cos already need.
- The out-of-domain `|x| > 2^30 → nan` policy. Correct, matches the sin/cos cap for the same reason (Payne-Hanek deferred).
- The pitfall list. All four pitfalls are accurate and useful for the pathmaker.
