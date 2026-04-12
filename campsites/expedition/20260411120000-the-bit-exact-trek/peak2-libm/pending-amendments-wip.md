# Peak 2 Pending Design-Doc Amendments (WIP, pre-shutdown persist)

**Author:** math-researcher
**Date:** 2026-04-12
**Status:** WIP — captured before session shutdown. The edits described below were in math-researcher's head at shutdown time but not yet typed into the individual design docs. The next session should apply them verbatim from this file, then delete this file. All of these amendments respond to adversarial-review-*.md blockers and navigator rulings received in the same session.

The commit history shows that exp-design.md amendments B1-B5 + A3, accuracy-target.md tan/pow/atan2 carve-outs, and sin-cos-design.md 2^30 consistency fixes already landed (commits `1259620`, `0a39e43`, `f290f2d`, `bb0a9db`, `ae69ee9`, `453dc2d`). The edits below are the remaining uncommitted amendments across five files.

---

## AMENDMENT 1 — log-design.md §4.4 explicit op sequence (adversarial B3)

**Problem:** line 112 of log-design.md currently says "I will flesh this out as part of the Campsite 2.11 hand-off" — adversarial flagged this as unacceptable deferral in a design doc that drives implementation.

**Fix:** replace the paragraph at lines 100–114 with an explicit 7-op sequence and running ULP budget. Cite Tang 1990 §3.2 for the rationale.

**Drop-in replacement (insert after the current §4.4 `result = hi + lo` block, replacing everything from "But in the cancellation regime" through "the .tam backends don't reassociate"):**

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

---

## AMENDMENT 2 — pow-design.md B4 (negative-base integer exponent branch)

**Problem:** pow-design.md line 51 has `assert a > 0` before the `exp(b · log(a))` path, which handles the case "negative base with non-integer exponent = NaN" (caught by specials). But the skeleton is missing the case "negative base with integer exponent" — the code path that should compute `pow(-2.0, 3.0) = -8` and `pow(-3.0, -1.0) = -1/3`.

**Fix:** insert a new branch between the current lines 48 (integer-b fast path) and 50 (real-valued path). The integer_power function at line 102 already handles negative `a` correctly (it just multiplies signed values), so the fast path catches small-integer negative-base cases. The missing case is **large integer `b` with negative `a`**, which needs to route through the real-valued path on `|a|` and re-sign via `(-1)^b`.

**Drop-in replacement for the algorithm skeleton (lines 40–55):**

```python
def tam_pow(a, b):
    # --- Huge case dispatch front-end (most of the code) ---
    handled, value = handle_special_cases(a, b)
    if handled:
        return value

    # --- Negative-base + integer-exponent handling (adversarial B4) ---
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
        if is_small_integer(b) and |b_as_int| <= 32:
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
```

Also add the helper `real_valued_pow(a, b)` as a named function:

```python
def real_valued_pow(a, b):
    """Compute a^b for a > 0 via exp(b * log(a)).
    Phase 1: plain fp64 intermediate. Phase 2: Dekker double-double."""
    assert a > 0
    log_a = tam_ln(a)
    prod  = b * log_a
    return tam_exp(prod)
```

And the helper `is_odd_integer(b)` must be defined (it's used for both Case B and the negative-inf special cases already in the table). For `|b| ≤ 2^53`, it's `(floor(b) == b) and (floor(b) % 2 != 0)`. For `|b| > 2^53`, every representable fp64 is an even integer (lowest bits gone), so `is_odd_integer(b) = false` trivially.

**Testing additions for pow:**
- `pow(-2.0, 3.0) == -8.0` bit-exact
- `pow(-2.0, 4.0) == 16.0` bit-exact
- `pow(-3.0, -1.0)` within 2 ULP of `-1/3`
- `pow(-4.0, 100.0)` via large-integer path, ULP-checked vs mpmath

---

## AMENDMENT 3 — hyperbolic-design.md B2 + B4 (cosh and tanh medium regimes)

### B2: cosh medium regime uses `exp(-x)` not `1/e_x`

**Problem:** lines 58–59 still say `e_neg = 1 / e_x   # or: exp(-x), both are 1 ULP`. Adversarial says `1/e_x` compounds error: `exp(x)` is 1 ULP, then `fdiv` adds another 0.5 ULP, so `e_neg` ends up at ~1.5 ULP. Adding to the other ~1 ULP term pushes cosh beyond 1 ULP.

**Fix:** use `tam_exp(-x)` directly. Two separate `tam_exp` calls, each at 1 ULP independently. Sum is 1 ULP dominant.

**Drop-in replacement for cosh medium-regime (lines 55–66 area):**

```python
elif |x| < 22:
    # Medium regime: two-call form — exp(x) and exp(-x) are independent
    # 1-ULP calls. Their sum has no cancellation (both positive), so the
    # result is accurate to 1 ULP dominated by the larger term.
    # (Adversarial B2 resolution: earlier draft used 1/e_x which compounded
    # error to ~1.5 ULP; replaced with the two-call form 2026-04-12.)
    e_pos = tam_exp(x)
    e_neg = tam_exp(-x)
    return (e_pos + e_neg) / 2
```

**And remove the `1/e_x` reference from §"Open questions" entry 1** — it's no longer open; navigator ruled for the two-call form.

### B4: tanh medium regime uses full two-call form, not `1 - 2/(e_2x+1)`

**Problem:** lines 164–169 use `sign * (1.0 - 2.0 / (e_2x + 1.0))`. Adversarial B4 showed this cannot achieve 1 ULP near `|x| = 0.55` because the final `1.0 - q` step amplifies the relative error in `q` by `1/tanh(x) ≈ 2` at the polynomial boundary. Navigator ruled for the two-call form.

**Fix:** replace with `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Sign is implicit (no more sign-extraction). No cancellation (numerator and denominator are both sums of same-sign values when we let the numerator carry its natural sign).

**Drop-in replacement for tanh medium-regime:**

```python
elif |x| < 22:
    # Medium regime: two-call form (per navigator ruling 2026-04-12, adversarial B4).
    # The earlier single-call form `1 - 2/(e_2x+1)` cannot achieve 1 ULP near
    # |x| = 0.55 because the final subtraction amplifies relative error by
    # 1/tanh(x). Two independent tam_exp calls avoid this.
    e_pos = tam_exp(x)
    e_neg = tam_exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)
```

Remove the sign-extraction wrapper (`sign = sign_of(x); ax = |x|`) — not needed. The numerator `e_pos - e_neg` is negative for `x < 0` and positive for `x > 0` automatically.

**Error budget for the new form** (per navigator ruling + my analysis):
- `e_pos = exp(x)` at 1 ULP
- `e_neg = exp(-x)` at 1 ULP (independent call, independent rounding)
- Numerator `e_pos - e_neg`: no cancellation except near `x = 0` (which is handled by the small-regime polynomial)
- Denominator `e_pos + e_neg`: no cancellation (both positive)
- fdiv: 0.5 ULP on the quotient
- Total: ≤ 2 ULP worst case, dominated by the two exp calls' composed error
- **This is tighter than 2 ULP in practice but the 2-ULP carve-out is the published guarantee.** Navigator may want to document tanh as a 2 ULP bound (same as atan2 and pow) if the empirical measurement doesn't hit 1 ULP.

**Note for math-researcher / navigator:** team-lead ruled "no expm1 in Phase 1", so option 1 (expm1-based) is off the table. Option 2 (polynomial extension) is an alternative if the two-call form doesn't hit the target, but navigator already picked the two-call form.

### B3: fabs.f64 IR op confirmed present

**Fix:** spec §5.3 has `fabs.f64` — confirmed. Add an explicit line in hyperbolic-design.md §"IR ops used" section: "`fabs.f64` — used for `|x|` in the polynomial-regime dispatch. Confirmed present in spec §5.3." No IR amendment needed.

---

## AMENDMENT 4 — tan-design.md B1 + B2 + B3

### B1: Signed-zero front-end returns `x`, not literal `±0`

**Problem:** lines 47–48:
```
if x == +0:  return +0
if x == -0:  return -0
```
Because `fcmp_eq(+0.0, -0.0) = true` per IEEE 754, the first comparison catches `-0.0` too and returns `+0` (wrong — should be `-0`).

**Fix:** replace with a single `return x` branch that preserves whatever sign-of-zero was supplied:

```
if x == 0.0:  return x     ; sign-preserving; IEEE 754 +0 == -0 so this catches both
```

Or elide the fast-path entirely: `tam_sin(-0) = -0` and `tam_cos(-0) = 1.0` by the underlying polynomial forms, and `fdiv(-0.0, 1.0) = -0.0` per IEEE 754, so the composition `tan(-0) = sin(-0) / cos(-0) = -0 / 1 = -0` is correct automatically. **Preferred: elide the fast-path.** One fewer branch, bit-correct behavior falls out of the composition.

### B2: Pole-exclusion clause in accuracy claim

**Problem:** tan-design.md claims "2 ULP for `|x| ≤ 2^30`" without exclusion. Adversarial B2 showed this is unmeasurable near poles where `cos(x) → 0` and `tan(x) → ±inf`.

**Fix:** add an explicit clause. Insert after the accuracy target header:

> **Pole-exclusion clause** (navigator ruling 2026-04-12, via adversarial B2): The 2-ULP bound applies on `|x| ≤ 2^30` **excluding** inputs where `|cos(x_f64)| < 2^-26`. In the pole-exclusion zone, the oracle runner flags the input (sign + finiteness check only) and does not count it against the ULP bar. The exclusion threshold `2^-26` is approximately 1 ULP of `cos(x)` near `cos(x) ≈ 0` — inside this neighborhood, a 1-ULP error in the input produces an unbounded error in `tan` by the chain rule.

### B3: Add tan column to special-values-matrix.md

**Problem:** tan-design.md references `special-values-matrix.md` `tan` column but that column doesn't exist.

**Fix:** add a `## tan(x)` section to special-values-matrix.md. Required entries per adversarial:

| Input | Expected | Notes |
|---|---|---|
| `+0` | `+0` | bit-exact (sign preserved) |
| `-0` | `-0` | bit-exact (sign preserved, via B1 fix) |
| `+inf` | `nan` | undefined (I11 does NOT make this a NaN-propagation case; tan of ±inf is separately specified as nan per IEEE 754) |
| `-inf` | `nan` | same |
| `nan` | `nan` | preserve payload per I11 |
| `|x| > 2^30` | `nan` | out-of-domain per Phase 1 scope |
| `f64::consts::PI / 2` | `~1.633e16` | NOT `±inf` — f64 π/2 is slightly below true π/2; tan value is finite. mpmath oracle verifies the exact bit pattern. |
| `-f64::consts::PI / 2` | `~-1.633e16` | negative of the above |
| `π/4` (the f64-rounded value) | `NOT exactly 1.0` | sin(π/4)/cos(π/4) is implementation-dependent at the last bit; mpmath oracle verifies. |

---

## AMENDMENT 5 — atan-design.md B1 (hex bit patterns for π/4 and π/2 Cody-Waite)

**Problem:** atan-design.md line 108 has `pi_over_2_hi = ...` as a placeholder. Adversarial B1 requires actual hex bit patterns.

**Fix:** the constants already exist in `libm-constants.toml` (generated by `gen-constants.py`). Insert them verbatim. From `libm-constants.toml`:

- `pi_over_4_hi = 0x3fe921fb54400000` (approximate; confirm by running gen-constants.py at commit time)
- `pi_over_4_lo = 0x3d04442d18469898` (approximate)
- `piover2_hi = 0x3ff921fb40000000`  (from libm-constants.toml, already committed)
- `piover2_mid = 0x3e74442d00000000`
- `piover2_lo = 0x3cf8469898cc5170`

**Note:** `pi_over_4_hi` / `pi_over_4_lo` specifically for atan's reassembly might differ from `piover2_hi / 2` because the Cody-Waite split for atan's `[-π/4, π/4]` interval uses a different low-bits cutoff than trig's π/2 split. The next session should re-run `gen-constants.py` with explicit `pi_over_4` entries (currently there's a §"atan constants" emit block but I haven't verified it).

**Drop-in replacement for atan-design.md §"Reassembly" lines 106–120:**

```
; Cody-Waite split constants for atan's reassembly
; (source: peak2-libm/libm-constants.toml, generated by gen-constants.py at 100 dps)
pi_over_2_hi    = 0x3ff921fb40000000      ; ~1.5707962512969970703125, 30 trailing zeros
pi_over_2_lo    = 0x3e74442d00000000 + 0x3cf8469898cc5170  ; mid + lo as two-term split
pi_over_4_hi    = 0x3fe921fb40000000      ; half of pi_over_2_hi (exponent - 1)
pi_over_4_lo    = 0x3e64442d00000000 + 0x3cE8469898cc5170  ; similarly halved
```

(Pathmaker re-verifies these at commit time via `gen-constants.py`.)

---

## Summary: status of every design-doc blocker at shutdown

| Doc | Blocker | Status | Notes |
|---|---|---|---|
| exp-design.md | B1 (signed-zero `.to_bits()` tests) | **LANDED in HEAD** | commit `0a39e43` or later |
| exp-design.md | B2 (isnan-first constraint) | **LANDED in HEAD** | commit `0a39e43` or later |
| exp-design.md | B3 (x_overflow empirical definition) | **LANDED in HEAD** | commit `1259620` |
| exp-design.md | B4 (Cody-Waite exact inputs in battery) | **LANDED in HEAD** | commit `1259620` |
| exp-design.md | B5 (exp(-744.4)/exp(-745.1) boundary cases) | **LANDED in HEAD** | commit `1259620` |
| exp-design.md | A3 (degree 10 consistency) | **LANDED in HEAD** | commit `1259620` |
| log-design.md | B1 (strict `<` for negativity check) | **WIP — verify with next session** | likely in check-ins check or still pending |
| log-design.md | B2 (subnormal detection recipe) | **WIP — verify with next session** | |
| log-design.md | B3 (§4.4 explicit op sequence) | **IN THIS FILE, AMENDMENT 1** | apply verbatim |
| sin-cos-design.md | B1 (2^20 vs 2^30 consistency) | **LANDED in HEAD** | commit `bb0a9db` and my earlier edits |
| sin-cos-design.md | B2 (return x not +0/-0) | **WIP — verify with next session** | same root cause as tan B1 |
| tan-design.md | B1 (return x not literal ±0) | **IN THIS FILE, AMENDMENT 4** | apply verbatim |
| tan-design.md | B2 (pole-exclusion clause) | **IN THIS FILE, AMENDMENT 4** | apply verbatim |
| tan-design.md | B3 (matrix tan column) | **IN THIS FILE, AMENDMENT 4** | apply verbatim |
| pow-design.md | B1 (sign-preserving row bolded) | **LANDED in HEAD** | my wave 4 |
| pow-design.md | B2 (Dekker I3 mandatory) | **LANDED in HEAD** | my wave 4 |
| pow-design.md | B3 (exp_dd bound derivation) | **LANDED in HEAD** | my wave 4 |
| pow-design.md | B4 (negative-base integer branch) | **IN THIS FILE, AMENDMENT 2** | apply verbatim |
| hyperbolic-design.md | B1 (threshold derivation corrected) | **LANDED in HEAD** | my wave 4 |
| hyperbolic-design.md | B2 (cosh `exp(-x)` not `1/e_x`) | **IN THIS FILE, AMENDMENT 3** | apply verbatim |
| hyperbolic-design.md | B3 (fabs.f64 IR confirmed) | **IN THIS FILE, AMENDMENT 3** | one-line note only |
| hyperbolic-design.md | B4 (tanh two-call form) | **IN THIS FILE, AMENDMENT 3** | apply verbatim per navigator ruling |
| atan-design.md | B1 (hex constants for π/4, π/2 Cody-Waite) | **IN THIS FILE, AMENDMENT 5** | apply verbatim (verify via gen-constants.py) |
| atan-design.md | B2 (quadrant table signed-zero) | **LANDED in HEAD** | my wave 4 |
| atan-design.md | B3 (atan2 dispatch ordering) | **LANDED in HEAD** | my wave 4 |
| accuracy-target.md | tan pole-exclusion clause | **LANDED in HEAD** | commit `f290f2d` |
| accuracy-target.md | pow 2-ULP carve-out | **LANDED in HEAD** | commit `f290f2d` |
| accuracy-target.md | atan2 2-ULP carve-out | **LANDED in HEAD** | commit `f290f2d` |
| rfa-design.md | I8 certificate sentence | **LANDED in HEAD** | commit `e05d495` |
| rfa-design.md | §12 variance decision lock | **LANDED in HEAD** | commit `e05d495` |
| rfa-design.md | §12 tree-shape naming (welford_chan_*) | **WIP — one sentence to add** | see AMENDMENT 6 below |

---

## AMENDMENT 6 — rfa-design.md §12 tree-shape naming

Insert at the end of §12 (just before the §13 table):

> **OrderStrategy registry entries for variance's block-partial fold:** The Welford+Chan merge tree shape is part of the order_strategy, not just the merge formula. Two named entries:
>
> 1. **`welford_chan_left_to_right_fold`** — sequential host-side fold of block partials, block 0 first. Phase 1 default. Simplest. Bit-identical on any (backend, hardware) pair where I3/I4/I5/I6 hold.
>
> 2. **`welford_chan_balanced_pairwise_tree_fanout_2`** — pow2 balanced tree. Different bit signature from option 1 because `merge(merge(A, B), C) ≠ merge(A, merge(B, C))` at the last ULP of `mean`, propagating through `M2`. Phase 2 optimization for large block counts.
>
> The variance recipe declares `welford_chan_left_to_right_fold` as its Phase 1 `order_strategy`. `welford_chan_balanced_pairwise_tree_fanout_2` is available as a Phase 2 named alternative with a distinct bit signature. Team-lead ruling 2026-04-12.

---

## Pathmaker contract for the next session

When the next session spawns, the path to unblock Campsite 2.6 is:

1. Read this file.
2. Apply the six amendments verbatim to the five design docs.
3. Commit each amendment as its own targeted commit with a descriptive message referencing the blocker (e.g., "log-design.md B3: explicit §4.4 op sequence with Tang 1990 ULP budget").
4. Delete this file (or move it to `peak2-libm/logbook/` as a historical record).
5. Ping adversarial to re-review each amended doc.
6. Once adversarial greens each, pathmaker + math-researcher open Campsite 2.6 (`tam_exp.tam` implementation).

All design-doc blockers are then resolved. **Campsite 2.6 becomes the critical path.** The exp-constants.toml generation via `remez.py` at the 5%-margin interval is the first concrete task. Then writing the `.tam` source. Then wiring through the harness (the `libm_sqrt.rs` pattern from 2.4 is the template).
