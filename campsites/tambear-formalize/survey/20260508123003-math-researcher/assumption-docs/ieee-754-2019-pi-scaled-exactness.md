---
campsite: tambear-formalize/survey/20260508123003-math-researcher
role: math-researcher
date: 2026-05-08
subject: assumption document — IEEE 754-2019 §9.2 pi-scaled trig exactness contract
status: draft for pathmaker review
audience: pathmaker (formalization), aristotle (definitional review), adversarial (regression-test design)
sources:
  - IEEE Std 754-2019 §9.2 "Recommended correctly rounded functions" (sinPi, cosPi, tanPi)
  - R:\winrapids\crates\tambear\src\recipes\libm\pi_scaled.rs (full file)
  - R:\winrapids\crates\tambear\src\recipes\libm\pi_scaled_inv.rs (atan2pi exact diagonals)
  - winrapids commits 5b89ab7 (tanpi quarter-integer fix), 1f9d347 (cospi general-path (-1)^n fix), bffe087 (atan2pi |y|=|x| diagonals), fadc620 (cospi external-oracle accuracy)
---

# Assumption Document: IEEE 754-2019 §9.2 Pi-Scaled Exactness Contract

> **Purpose.** The pi-scaled trig functions (`sinpi`, `cospi`, `tanpi`, plus the inverse forms `asinpi`, `acospi`, `atanpi`, `atan2pi`) carry an *exactness contract* the radian forms do not. At integer, half-integer, and quarter-integer inputs, the output is an exact representable value — not approximated. IEEE 754-2019 §9.2 enumerates these. Three real bugs in the recent winrapids history landed here (commits 5b89ab7, 1f9d347, bffe087) — the contract is correct but the implementations got it wrong in distinct ways. This document specifies the contract exhaustively so the formalization onto the locked-vocabulary substrate doesn't repeat the mistakes.

---

## 1. The contract

### 1.1 sinpi(x) = sin(π·x)

| Input class | Output | Exactness |
|---|---|---|
| `x = ±0` | `±0` (sign preserved) | exact, IEEE 754-2019 §6.3 sign rule |
| `x` integer (n) | `+0` for all n | **exact** |
| `x = n + ½` | `(-1)^n` | **exact** (`+1` or `-1` literal) |
| `x = n + ¼`, `x = n + ¾` | `±√2/2` ≈ `±0.7071...` | NOT exact (irrational) |
| `x` general non-integer | `sin(π·x)` to ≤ ULP budget | per `[parameters.precision]` |
| `x = NaN` | `NaN` | propagation |
| `x = ±∞` | `NaN` | not periodic at infinity |

### 1.2 cospi(x) = cos(π·x)

| Input class | Output | Exactness |
|---|---|---|
| `x = ±0` | `+1` (cos is even) | **exact** |
| `x` integer (n) | `(-1)^n` | **exact** (`+1` for even n, `-1` for odd) |
| `x = n + ½` | `+0` for all n | **exact** |
| `x = n + ¼`, `x = n + ¾` | `±√2/2` ≈ `±0.7071...` | NOT exact (irrational) |
| `x` general non-integer | `cos(π·x)` to ≤ ULP budget | per `[parameters.precision]` |
| `x = NaN` | `NaN` | propagation |
| `x = ±∞` | `NaN` | not periodic at infinity |

### 1.3 tanpi(x) = tan(π·x)

| Input class | Output | Exactness |
|---|---|---|
| `x = ±0` | `±0` (sign preserved) | exact |
| `x` integer (n) | `+0` for all n | **exact** |
| `x = n + ½` | `+∞` if n even, `-∞` if n odd (with sign flip on neg x) | **exact** (signed infinity) |
| `x = n + ¼` | `+1` for ALL n | **exact** — tan has period π, NOT 2π |
| `x = n + ¾` | `-1` for ALL n | **exact** — tan has period π, NOT 2π |
| `x` general non-integer non-half-integer | `tan(π·x)` to ≤ ULP budget | per `[parameters.precision]` |
| `x = NaN` | `NaN` | propagation |
| `x = ±∞` | `NaN` | not periodic at infinity |

**Period mismatch — the central trap**: sin and cos have period 2π → sinpi and cospi have period 2 → n-parity DOES affect their output sign. Tan has period π → tanpi has period 1 → n-parity does NOT affect its output. **The recent winrapids bug 5b89ab7 was exactly this confusion** — the old tanpi code applied `(-1)^n` flip to quarter-integers as if they followed sin/cos parity rules. They do not.

## 2. The asymmetry between sinpi/cospi and tanpi

Stating it explicitly because this is what the bug-history shows is non-obvious:

```
sinpi(n + 0.25) = ?
  Use sin(π·(n + 0.25)) = sin(π·n + π/4) = (-1)^n · sin(π/4) = (-1)^n · √2/2.
  → n-parity FLIPS the sign.

cospi(n + 0.25) = ?
  Use cos(π·(n + 0.25)) = cos(π·n + π/4) = (-1)^n · cos(π/4) = (-1)^n · √2/2.
  → n-parity FLIPS the sign.

tanpi(n + 0.25) = ?
  tan has period π. So tan(π·(n + 0.25)) = tan(π/4) = 1, INDEPENDENT of n.
  → n-parity does NOT flip the sign.
```

The mistake is intuitive: "sinpi uses (-1)^n, cospi uses (-1)^n, so tanpi must also use (-1)^n." It is wrong because tan's period is half the period of sin/cos.

For the formalization: the n-parity logic must be derived per function from the period, not pattern-matched from the sister functions.

## 3. The general-path (-1)^n logic for sinpi/cospi

For non-special inputs, sinpi reduces fractional-part-of-x to `[0, 1)` then to `[0, ½)` (cos symmetry around `½`) then to `[0, ¼)` (sin/cos exchange via the complement identity). The general path:

```
let x_pos = |x|;
let n = floor(x_pos) as i64;
let frac = x_pos - n as f64;     // exact fractional part (since 1.0 is exact)
let integer_sign_neg = (n & 1) != 0;  // (-1)^n flip for sinpi/cospi general path
// ... evaluate kernel ...
let final_sign_neg = integer_sign_neg XOR (sign of x for sinpi only);
```

Bug 1f9d347: cospi general-path was missing the `integer_sign_neg` line. Symptom: cospi(2.7) was computed correctly but cospi(3.7) had wrong sign. **The fix asserts the sign flip post-kernel.**

For tanpi general path: sinpi/cospi ratio handles the n-parity automatically (both numerator and denominator carry the (-1)^n, which cancels). So the general path just calls `sinpi(x) / cospi(x)`. The **EXACT special cases must fire BEFORE the general path** to avoid 0/0 at integers and ∞/0 at half-integers.

## 4. The is_integer / is_half_integer / is_quarter_integer predicates

The exact-special-case detection sits at the head of every pi-scaled function. From `pi_scaled.rs`:

```rust
fn is_integer(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0
}

fn is_half_integer(x: f64) -> bool {
    x.is_finite() && (x * 2.0).fract() == 0.0 && !is_integer(x)
}

fn is_quarter_integer(x: f64) -> bool {
    x.is_finite() && (x * 4.0).fract() == 0.0
        && !is_half_integer(x) && !is_integer(x)
}
```

**Why this works (correctness invariant)**: 0.5, 0.25, 0.75 are all exactly representable in f64. `x * 2.0` and `x * 4.0` are exact for the relevant magnitudes (no rounding in the multiply). So `(x * k).fract() == 0.0` is a sound predicate for "x is a k-multiple of 1/k."

**Why this matters at the boundary**: for very large x where `|x| > 2^52`, every f64 value is an integer (the mantissa cannot hold a fractional part). At that scale, `is_integer(x)` returns true unconditionally and the integer-branch fires. This is correct: at huge magnitudes, the exact-integer answer is the only sensible one.

**Stale comment to fix during formalization** (drift item): in `pi_scaled.rs` line 200, the comment reads "Overall n-parity flips sign" — this is a leftover from the bug 5b89ab7 era. The corrected logic at lines 205-206 says "tan has period π, so ... n-parity does NOT flip." The code is correct; the line-200 comment contradicts the line-205 comment. Delete line 200's comment.

## 5. Subtle edge cases the formalization must preserve

### 5.1 ±0 sign preservation

`sinpi(-0)` must return `-0`, not `+0`. IEEE 754-2019 §6.3 specifies that for sin(±0) (a sign-preserving function). The code at pi_scaled.rs line 60 returns `x` directly when `x == 0.0`, which preserves the sign bit (since `==` ignores sign on zero but returning `x` doesn't). ✓ Correct.

`cospi(-0)` must return `+1` (cos is even, even at zero). pi_scaled.rs's `is_integer` branch fires for x=0, returns `+1`. ✓ Correct.

### 5.2 Sign of infinity for tanpi at half-integers

tanpi(0.5) = +∞. tanpi(-0.5) = -∞. tanpi(1.5) = -∞. tanpi(2.5) = +∞. The pattern: `+∞` for `(n + 0.5)` with n even (counting from 0), `-∞` for n odd, then sign-flip for negative x.

```rust
// From pi_scaled.rs lines 192-196:
let n = (x.abs() - 0.5).floor() as i64;
let pos_sign = n % 2 == 0;
let pos_sign = if x.is_sign_negative() { !pos_sign } else { pos_sign };
```

This is correct. The "n" here is the integer part of `|x| - 0.5`, so for `x = 1.5` we get `n = 1`, `pos_sign = false`, returning `-∞`. ✓

### 5.3 Very large half-integers

For `x = 2^52 + 0.5`: this is NOT representable exactly in f64 (the mantissa runs out at integers near 2^52). The literal value `2^52 + 0.5` rounds to `2^52` in f64. So `is_half_integer(2^52 + 0.5)` is FALSE — the input x already rounded to an integer before we saw it. tanpi returns 0, sinpi returns 0, cospi returns ±1.

This is the IEEE-correct answer because the input the function actually saw IS an integer (the literal was rounded at parse time). It's not a "near-half-integer" question; it's an "exactly-an-integer-after-rounding" question.

**Important for the assumption document of the formalization**: tambear must commit to this behavior consistently. The function does not "know" that the user typed `2^52 + 0.5`; it sees `2^52`. Treating that as an integer is correct. Documenting this in the spec.toml prevents user confusion.

### 5.4 atan2pi exact at quarter-integer diagonals

Bug bffe087 was about atan2pi exactness at the `|y| = |x|` diagonal. The mathematical fact: `atan2pi(y, x) = atan2(y, x) / π`. When `|y| = |x|`:
- `atan2(y, x) = ±π/4 or ±3π/4` exactly (geometric)
- atan2pi(y, x) = `±0.25 or ±0.75` exactly

The implementation must detect `|y| == |x|` BEFORE computing atan2 — because `atan2 / π` involves an irrational division and won't land on `0.25` exactly. The fix at bffe087 added the diagonal-detection.

**Generalization for the formalization**: every inverse-pi-scaled function (asinpi, acospi, atanpi, atan2pi) has an exact-output table at landmark inputs. The implementation must short-circuit those before any transcendental call. List of landmarks per function:

| Function | Exact-output landmarks |
|---|---|
| `asinpi(x)` | `asinpi(0) = 0`, `asinpi(±½) = ±1/6`, `asinpi(±1) = ±½`, `asinpi(±√2/2) = ±¼` (irrational input → output 0.25 NOT exact in f64; only for `x = ±1` and `x = 0` are both input and output exact) |
| `acospi(x)` | `acospi(0) = ½`, `acospi(±1) = 0` or `1`, `acospi(±½) = 1/3` or `2/3` |
| `atanpi(x)` | `atanpi(0) = 0`, `atanpi(±1) = ±¼`, `atanpi(±∞) = ±½` |
| `atan2pi(y, x)` | `atan2pi(0, +x) = 0`, `atan2pi(0, -x) = 1`, `atan2pi(±y, 0) = ±½`, `atan2pi(±x, ±x) = ±¼ or ±¾` (the diagonals) |

The exactness IS the contract. Future implementations must short-circuit each landmark.

## 6. The bugs that landed here

Three bugs in winrapids HEAD (last 4 weeks) all touched the pi-scaled exactness contract:

| Commit | Bug | Class |
|---|---|---|
| `5b89ab7` | tanpi(n+0.25) sign was n-parity-flipped | **period mismatch** (tan period π not 2π) |
| `1f9d347` | cospi general-path missing (-1)^n in non-special-case path | **incomplete general path** |
| `bffe087` | atan2pi(±x, ±x) diagonals not exact (small ULP error from atan2/π) | **missing landmark short-circuit** |
| `fadc620` | cospi external-oracle accuracy regression | **post-fix verification** |

These represent THREE distinct error classes, each non-obvious. The adversarial harness caught them by emitting ULP-neighborhood probes around the n+0.25, n+0.75, n+0.5 landmarks — exactly the regions where the exactness contract requires special handling.

For the formalization: when porting these recipes to `R:\tambear\`, every landmark in §1's tables must have a hand-written test asserting `output.to_bits() == expected.to_bits()` (bit-exact, not ulps-close). Bit-exact is the right test because the contract claims exactness, not approximation. ULP-close tests would have missed all three of these bugs.

## 7. The compensated/correctly_rounded path question — F12 framing

`pi_scaled.rs` aliases `sinpi_compensated`, `sinpi_correctly_rounded`, `cospi_compensated`, etc. all to `_strict`. The exactness contract makes this *defensible* — the exact-special-case branches return `0.0`, `1.0`, `-1.0`, `±∞`, all bit-exact, and the general path calls `sin_strict(π · kernel_arg)` (or `cos_strict`) whose precision is determined by sin/cos's precision contract.

**But under F12** (aristotle's deconstruction at `survey/20260508123003-aristotle/default-is-a-claim.md`), the issue isn't whether the alias is *defensible* — it's whether the alias is *declared*. A spec.toml that lists `precision: {strict, compensated, correctly_rounded}` as parameter values WITHOUT declaring that compensated and correctly_rounded alias to strict is making a silent claim it doesn't keep. That's the contract violation; the aliasing itself is permitted.

**Three F12-compliant fixes** (pathmaker picks):

- **(a) Real recursive dispatch.** `pi_scaled.using(precision="correctly_rounded")` recursively descends into `sin_correctly_rounded(π · kernel_arg)`. Implements the triplet for real. Highest effort; cleanest semantics.
- **(b) Declared aliasing with rationale.** Spec.toml gets:
  ```toml
  [stances.override_transparency.strategy.compensated]
  state = "aliased_to"
  target = "strict"
  rationale = "Pi-scaled exact-special-case branches return bit-exact landmarks (0, ±1, ±∞); compensated arithmetic gains nothing at exact outputs. The general path delegates to sin/cos whose precision triplet carries the actual contract — pi-scaled adds no precision dimension of its own."
  ```
  Same code, different spec. Cheapest. Honest about what's actually implemented.
- **(c) Reduced parameter domain.** Drop `compensated` and `correctly_rounded` from `parameters.precision.domain`, leaving only `strict`. Removes the false-claim surface entirely. Loses the ability to ever extend later without breaking change.

**Recommendation**: **(b) for v1**, with `(a)` as a tracked v2 if a downstream consumer actually needs the recursive dispatch. The rationale at (b) is true and the cost is one spec.toml block; it satisfies F12 without writing new arithmetic. (a) is the right semantic model but pays full implementation cost for a use case (correctly-rounded pi-scaled) that may never have a consumer.

**Filter Test §10 reading**: F12 elevates "publication-grade rigor" from "the strategy works" to "the spec accurately describes what works." A declared alias is rigorous; a silent alias is not — even if both produce the same numerical output. The rigor lives in the contract surface, not just the bits.

## 8. Filter Test recap for the pi-scaled family

For sinpi/cospi/tanpi/asinpi/acospi/atanpi/atan2pi under the locked vocabulary:

- ✅ Custom-implemented (we author the exact-special-case tables, the period reasoning, the general path)
- ✅ Atom decomposition: per-element scalar; `accumulate(All, Expr::SinPi, Op::Identity)` (Expr variants needed — see SURVEY.md)
- ✅ Shareable intermediates: pi-scaled does NOT share `trig_reduce` (different reduction — fractional part of x, not x mod π/2). It MAY share its own intermediate `pi_scaled_reduce` if multiple pi-scaled functions hit the same column.
- ✅ Every parameter tunable: `precision`, `angle_unit` (the user calls `sinpi(degrees_in_pi_units)` for example)
- ⚠️ Every variant: triplet aliases-to-strict for now (drift item §7)
- ✅ Optimized for 2026 hardware: per-element parallel; integer/half-integer/quarter-integer detection vectorizes
- ✅ No vendor lock-in
- ✅ No OS lock-in
- ✅ Lifting to TAM: per-column, per-element. TAM picks ALU surface
- ✅ Publication-grade rigor: IEEE 754-2019 §9.2 cited; bit-exact tests for every landmark; adversarial harness coverage; bug-history documented

## 9. Open questions for pathmaker / aristotle

1. **Recursive precision dispatch**: should `sinpi_correctly_rounded` call `sin_correctly_rounded` for the kernel evaluation? §7 — recommend yes for v1.

2. **Pi-scaled cache key**: should pi-scaled functions register their reduced argument (fractional part of `|x|`, plus quadrant info) in a `pi_scaled_reduce` ComputedTag for sharing across sinpi/cospi/tanpi calls on the same column? Per the trig_reduce sharing assumption doc, sharing-correctness invariant requires the reduction to be the SAME under sharing. Pi-scaled's reduction differs from radian's — different cache namespace. Probably worth implementing for v2.

3. **The atan2pi diagonal short-circuit**: §5.4 lists the exact landmarks for atan2pi. The `|y| == |x|` check is an exact comparison and works for any magnitudes. Confirm the formalization implements it BEFORE the general atan2 / π division.

4. **Integer detection at huge magnitudes** (§5.3): `|x| > 2^52` makes every f64 an integer. tambear must commit to "the function sees what it sees" and not try to detect "user intended a non-integer." Document in spec.toml's long_description.

5. **NaN-policy interaction**: NaN input → NaN output (per §1 tables). Inf input → NaN output (not periodic at infinity). Confirm this is consistent with tambear's NaN/Inf policy (vocabulary.md §"Tier 2 — Op and Expr" / "NaN/Inf policy"). My read: the propagate default applies; pi-scaled's behavior is identical to radian trig at NaN/Inf.

6. **Rounding of inverse pi-scaled at landmarks** (§5.4 table): `asinpi(½) = 1/6`. 1/6 is irrational in binary. The landmark `asinpi(½)` cannot return an exact answer in f64 — it returns the f64-nearest rounding. The exactness contract for asinpi is therefore narrower than for sinpi: only `asinpi(0) = 0` and `asinpi(±1) = ±½` are exact. **Document this asymmetry in the spec.toml.**

## 10. Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-formalize`, after the trig_reduce + CW/PH crossover assumption docs.
- Sources verified: `R:\winrapids\crates\tambear\src\recipes\libm\pi_scaled.rs` (the full file), `pi_scaled_inv.rs` (atan2pi diagonal logic), winrapids commit log for the bug history.
- Cross-references: SURVEY.md "Bug-fix history (genuine adversarial value)" section; `assumption-docs/trig-reduce-sharing.md` §6 (special-case handling parallel); `assumption-docs/cody-waite-payne-hanek-crossover.md` §6 (special-case enumeration parallel).
- This is a draft. Open questions in §9 require pathmaker review before the pi-scaled family ports to `R:\tambear\`. The drift item in §7 (recursive precision dispatch) is the most consequential one for stance-classification.
