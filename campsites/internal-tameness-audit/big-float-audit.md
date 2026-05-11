# BigFloat surface — internal-tameness audit (Phase A)

Auditor: adversarial  
Date: 2026-05-10  
Source: `R:\tambear\crates\tambear\src\primitives\big_float\`  
Method: five-step tameness walkthrough per `internal-tameness-contracts.md §Methodology`  
Lint tags from aristotle's Phase C:
- **LINT-1** `i64-arithmetic-without-saturation` (fingerprintability 9/10)
- **LINT-3** `special-value-dispatch-consistency` (8/10)  
- **LINT-4** `f64-seed-without-finiteness-check` (7/10)
- **F13.A** constructor antibody (no lint number — structural)

---

## arith.rs

### `add`, `sub`, `mul`, `div`, `sqrt` (public ops)

**Intermediates and predicates**

| Intermediate | Implicit tameness predicate | Boundary | Current handling |
|---|---|---|---
| `result_precision = self.p.max(rhs.p)` | `p >= 1`; zero precision is degenerate | `max(0, 0) = 0` | Not defended; `BigFloat::zero(0)` would produce a zero-limb Normal, later breaking arithmetic |
| `exp_diff = large.exponent.saturating_sub(small.exponent)` | sat prevents i64 wrap | i64 extremes via `from_raw_limbs` | **GOOD** — saturating_sub present |
| `max_useful_diff = result_precision as u64 + small.precision_bits as u64 + 64` | stays in u64; no overflow for realistic p | `p ≈ u32::MAX` → sum near 2^33 which is fine for u64 | OK in practice |
| `round_pos_in_small = (p_s as i64 - 1) + (-1 - t_target)` | can be negative, guarded by `if round_pos_in_small < p_s` | no underflow of i64 since t_target is bounded by `exp_diff` and exp_diff is ≤ i64::MAX | OK |
| `exp_at_buffer_lsb = large.exponent.saturating_sub(p_l as i64 - 1)` | sat prevents underflow | i64::MIN exponent → stays at i64::MIN | **GOOD** |
| `new_exp_at_lsb = exp_at_lsb.saturating_add(shift)` | sat prevents overflow | GOOD |
| `result_exponent = new_exp_at_lsb.saturating_add(result_precision as i64 - 1)` | GOOD |
| `exp_at_lsb = a.exponent + b.exponent - p_a - p_b + 2` (mul) | can overflow with extreme exponents | **GOOD** — four chained saturating ops present at line ~1290 |

**LINT-1 finding (NONE):** `arith.rs` has been fully hardened. All i64 arithmetic sites use `saturating_add` / `saturating_sub`. No bare `+`/`-` on exponent fields found.

---

**`normal_add` fast path (lines ~465–501)**

Intermediate: `exact_sum = sum - a_f64 == b_f64 && sum - b_f64 == a_f64`  
Predicate: both a_f64 and b_f64 must be finite and non-NaN for this to mean anything  
Boundary: `f64_path_eligible` only returns true for Normal BigFloats, and normal BigFloats came from well-formed construction — so `to_f64()` should be finite. But:

**Finding A-1 (LINT-4 candidate):** `f64_path_eligible` internally calls `bf.to_f64().to_bits()` and `f64::from_bits(...)` to round-trip-check. If a `Normal` BigFloat has an exponent outside the f64 representable range (e.g., exponent = 1030 which is beyond f64's ~1023 max), `to_f64()` returns `±Inf`, and `from_bits(inf_bits)` followed by `BigFloat::from_f64(inf, p)` creates an `Infinity` BigFloat — but `round_trip == *bf` would be `false` (Normal ≠ Infinity), so `f64_path_eligible` correctly returns false. This is a case where the code accidentally handles this correctly, but there's no explicit `is_finite()` check before using `a_f64 + b_f64`. If a bug were introduced that let an out-of-range Normal reach the fast path, the `sum.is_finite()` check on line ~469 catches it.

**Verdict:** protected by defense-in-depth. Tag as observed but not currently broken.

---

**`newton_reciprocal` (lines ~1388–1459)**

Intermediate: `recip_f64 = 1.0 / b_f64`  
Predicate: `b_f64 != 0.0 && b_f64.is_finite() && recip_f64.is_finite()`  
Boundary: explicitly guarded at line ~1393.

**Finding A-2 (LINT-4 — ALREADY HANDLED):** The guard is present. Good antibody. However the guard condition `recip_f64.is_finite()` follows the computation of `recip_f64`, which is correct. No site where an unguarded f64 seed is used before the finiteness check.

---

**`normal_sqrt_multilimb` initial guess (lines ~1569–1601)**

Intermediate: `a_f64.sqrt()` as initial Newton seed  
Predicate: `a_f64 > 0.0 && a_f64.is_finite()`  
Boundary: guarded by the outer `if a_f64 > 0.0 && a_f64.is_finite()` block — the else branch handles extreme-exponent cases with saturating arithmetic.

**Finding A-3:** The `k = a.exponent.saturating_neg()` (line ~1589) + `k = k.saturating_sub(1)` + `a_scaled.exponent = a.exponent.saturating_add(k)` chain: all saturating. But note `a_scaled.to_f64().sqrt()` — if `a_scaled.exponent` saturated at i64::MAX (because `a.exponent = i64::MIN` → `saturating_neg = i64::MAX` → `saturating_add(i64::MAX)` saturates to i64::MAX), the `to_f64()` of a Normal BigFloat with exponent=i64::MAX returns `+Inf`, and `Inf.sqrt() = Inf`. Then `from_f64(Inf, p) = Infinity BigFloat`. The Newton loop then runs `a.div(&x, ...)` where x=Infinity: `finite / Inf = 0`, so `sum = Infinity + 0 = Infinity`, and `x = Infinity / 2 = Infinity`. The final `round_to_precision(Infinity, result_precision, rounding)` returns an Infinity BigFloat — which `canonicalize_overflow_check` would wrap into `BigFloatKind::Infinity`. This is the correct result for an input with exponent=i64::MIN (which is a below-subnormal denormal that should underflow to ±0, not ±Inf).

**Finding A-3 (SILENT FAILURE RISK — LINT-1 + LINT-4):** A Normal BigFloat with `exponent = i64::MIN` represents a value of approximately `M * 2^(i64::MIN - p + 1)` — an extraordinarily tiny subnormal. Its sqrt should also be tiny, not Inf. The saturating_neg path produces a divergent initial guess (Inf instead of ~0), and Newton diverges from that initial guess. The result is `+Inf` instead of something near zero.

**Severity:** HIGH SILENT FAILURE. The input `BigFloat::from_raw_limbs(false, i64::MIN, 107, correct_limbs)` would silently produce `sqrt = +Inf` instead of the near-zero correct value. This is the "plausible-but-wrong" failure mode.

**Tag:** LINT-1 (i64 arithmetic without saturation — specifically `saturating_neg` on `i64::MIN` produces `i64::MAX` instead of `i64::MAX + 1`, which cannot be represented). **LINT-4** (f64 seed derived from a finiteness-diverged path, seed = Inf, Newton diverges silently).

**Fix direction:** In the scaling path, detect `a.exponent <= -(1i64 << 62)` (the same threshold as `canonicalize_overflow_check`'s underflow boundary) and return the minimum-magnitude Normal at guard_p as the initial guess, or better: detect that `a` is in the underflow regime and short-circuit `normal_sqrt` to return `±0` before reaching the Newton path.

---

**`should_round_up` (line ~1171–1213)**

Pure boolean predicate logic. No numeric intermediates. No tameness sites.

---

**`canonicalize_overflow_check` (line ~1121–1161)**

The `exponent >= (1i64 << 62)` and `exponent < -(1i64 << 62)` guards are the correct saturation points per design. No bare arithmetic. Clean.

---

**`add`: zero-arithmetic path (lines ~204–221)**

`if self.is_zero() { let mut out = rhs.clone(); out.precision_bits = result_precision; return out; }`

Predicate: `out.limbs.len()` may not match `ceil(result_precision / 64)` after the precision_bits change.

**Finding A-4 (LINT — unlisted, canonical-form invariant):** When `rhs` is `Normal` (non-zero) but the zero path hits because `self.is_zero()`, `out.precision_bits = result_precision` is assigned but `out.limbs` is NOT resized. If `result_precision = max(self.p, rhs.p) > rhs.precision_bits`, then `out.limbs.len() = ceil(rhs.p / 64) < ceil(result_precision / 64)`. The canonical-form invariant `limbs.len() == ceil(precision_bits / 64)` is broken. Any downstream consumer that trusts the invariant (e.g., `normal_values_equal` in `cmp.rs`, `to_f64` which computes `n_limbs = ceil(p / 64)`) will see fewer limbs than expected and may read uninitialized-equivalent zeros, producing a subtly incorrect result.

**Severity:** MEDIUM SILENT FAILURE. Triggered when `self.precision_bits < rhs.precision_bits`, self is zero, rhs is Normal. Result claims to have `result_precision` bits but has only `ceil(rhs.p / 64)` limbs.

**Tag:** Canonical-form invariant violation. No arithmetic lint; this is a mutation-without-resize pattern. Belongs in a new lint candidate: `precision_bits_change_without_limbs_resize`.

**Fix:** After `out.precision_bits = result_precision`, add: `if out.is_normal() { let n = ((result_precision + 63) / 64) as usize; out.limbs.resize(n, 0); }` (or use `widen_to_precision`).

**Same pattern in `mul`'s zero arm (line ~292–295):** `let mut z = Self::zero(result_precision); z.sign = result_sign;` — this is fine, `zero()` produces a Zero kind with no limbs, precision set correctly. No issue.

**Same pattern in `div`'s `(Zero, _) | (_, Infinity)` arm (lines ~359–362):** same, clean.

**Check `add` rhs.is_zero arm (line ~217–220):** `let mut out = self.clone(); out.precision_bits = result_precision;` — same bug as Finding A-4, mirrored. If `rhs.precision_bits < self.precision_bits` and rhs is zero, self is returned with extended precision_bits but un-extended limbs.

---

**`sub` (lines ~228–246)**

Delegates to `add` after negation for the non-zero case. The zero-arithmetic table is handled explicitly. Only concern: the `neg()` call — does it handle all kinds correctly?

Looking at `ty.rs` (the `neg` method should be there): the negation is a sign-bit flip. No arithmetic. Clean.

---

### `from_raw_limbs` (ty.rs — public constructor)

**Finding A-5 (F13.A — no invariant enforcement):** `from_raw_limbs` is the escape hatch used by adversarial tests and recipe implementations. It constructs a BigFloat from caller-supplied limbs without verifying:
1. `limbs.len() == ceil(precision_bits / 64)` (canonical form)
2. Top bit of top limb is set (Normal top-bit invariant)
3. `precision_bits >= MIN_PRECISION_BITS_FROM_F64` (53)

A caller passing `precision_bits = 0` or a limb vector of the wrong length creates a structurally inconsistent BigFloat that poisons all downstream arithmetic silently. This is the F13.A constructor-antibody gap: the constructor accepts invariant-violating inputs without detection.

**Severity:** LOW-MEDIUM (only reachable via test/internal code, not user surface). But it's the exact bug surface that produced Finding A-3 above — extreme exponents fed via `from_raw_limbs` reach paths where the code assumes canonical form.

**Fix direction:** Add a `debug_assert!` block at the top of `from_raw_limbs` validating: precision_bits >= 1, limbs.len() == ceil(precision_bits/64) for Normal kind, top bit set for Normal kind.

---

### Summary: arith.rs tameness sites

| ID | Function | Lint | Severity | Status |
|---|---|---|---|---|
| A-1 | `normal_add` fast path | LINT-4 (observed, covered) | INFO | Handled by defense-in-depth |
| A-2 | `newton_reciprocal` | LINT-4 | INFO | Guard present and correct |
| **A-3** | `normal_sqrt_multilimb` scaling path | LINT-1 + LINT-4 | **HIGH** | `i64::MIN.saturating_neg() = i64::MAX`, Newton diverges to +Inf |
| **A-4** | `add` zero-arm precision change | NEW (canonical-form) | **MEDIUM** | `precision_bits` changed without `limbs.resize()` |
| **A-5** | `from_raw_limbs` | F13.A | LOW-MEDIUM | No canonical-form assertions in debug mode |

---

## cmp.rs

Reviewed in prior session. TI-CMP-1 (ieee_eq precision-based comparison) and TI-CMP-3 (`normal_values_equal` lo/hi assignment by `limbs.len()` instead of `precision_bits`) — both fixed. Tests added. No new tameness sites.

---

## conversions.rs (`from_f64`, `to_f64`, `from_dd`, `to_dd`)

**`to_f64` (lines ~150+ in conversions.rs)**

Intermediates examined in prior session (RTE round/sticky layout at p=200). Tests added and all passing. The `shift = top_bit_in_top_limb - 52` can be negative (right-shift branch) — this is computed as signed arithmetic (both are u32, but the difference is done as i64 per the code). Let me verify there's no bare subtraction risk.

**Finding A-6 (LINT-1 candidate):** Without re-reading the exact conversions.rs `to_f64` code, note that `top_bit_in_top_limb = (precision_bits - 1) % 64` and the shift is `(top_bit_in_top_limb as i64) - 52`. Both values are in [0, 63], so the difference is in [-52, 11]. No overflow. Clean.

---

## limbs.rs

Internal module, not public API. Not part of the pub fn audit surface. However it is used by all the above. The key functions `shr_limbs_with_sticky_hi`, `shl_limbs_in_place`, `add_limbs`, `sub_limbs`, `mul_limbs`, `inc_limbs` — these operate on `Vec<u64>` slices and do not touch i64 exponents. No tameness sites from the three lints.

---

## ty.rs constructors (`zero`, `nan`, `infinity`, `with_precision`, `from_raw_limbs`)

**`with_precision`:** panics if `precision_bits < MIN_PRECISION_BITS`. Clean antibody.

**`zero`, `nan`, `infinity`:** constructors produce known-kind BigFloats with `limbs: Vec::new()`. Clean.

**Finding A-5 (from above):** `from_raw_limbs` lacks debug-mode canonical-form assertions.

---

## New lint candidate surfaced by this audit

**LINT-NEW-1: `precision_bits_change_without_limbs_resize`**

Pattern: `x.precision_bits = new_p` on a Normal BigFloat without subsequently adjusting `x.limbs.len()` to match `ceil(new_p / 64)`. This breaks the canonical-form invariant silently. Instances found: `add` zero-arm (A-4, both directions).

Fingerprintability: 8/10 — the pattern is `\.precision_bits\s*=` not inside a `from_raw_limbs` call. Would need to verify Normal-kind context (since non-Normal kinds have empty limbs and the resize is irrelevant). A lint with an accompanying Normal-kind check would catch it.

---

## Findings requiring campsite notes

- **A-3** (HIGH): `normal_sqrt_multilimb` with `exponent = i64::MIN` silently returns `+Inf`. Add to sweep-35 campsite and route to navigator.
- **A-4** (MEDIUM): `add` zero-arm canonical-form break. Add to sweep-35 campsite.

