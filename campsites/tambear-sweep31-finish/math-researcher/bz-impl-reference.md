---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher
date: 2026-05-08
sweep: 31 (multi-limb arith unstub)
audience: pathmaker (impl lead), aristotle (invariants), adversarial (proptest fuzz), scientist (oracle harness)
purpose: tactical implementation reference for BZ Algorithms 3.1 / 3.3 / 3.5 / 3.10 against the specific BigFloat storage layout shipped at 33d3849.
inputs:
  - Brent & Zimmermann, *Modern Computer Arithmetic*, 2nd ed., §3.1, §3.3 (incl. §1.3.4 schoolbook), §3.5, §3.10
  - DESIGN.md §1 (storage layout), §3 (algorithm dispatch table) — ratification-level
  - arith.rs (33d3849 + c2798f5) — current f64-fast-path + special-value dispatch
  - ty.rs (33d3849) — canonical-form invariant: top bit of `limbs[n-1]` at `(p-1)%64`, little-endian limb order
  - conversions.rs (33d3849) — limb-pack convention (top-bit-set, exponent = binary place value of top bit)
---

# BZ Multi-Limb Arithmetic — Implementation Reference

> **The map**. DESIGN.md is the ratification-level argument: why each algorithm,
> what each invariant. This document is the implementation-level reference: how
> each algorithm operates on the *specific* BigFloat encoding (top-bit-set
> mantissa, little-endian u64 limbs, magnitude+sign, i64 exponent = binary
> place value of top bit).
>
> **The audience**. Pathmaker writes the code. Aristotle pressure-tests
> invariants. Adversarial designs proptests. Scientist runs mpmath oracle. This
> doc is what each of them refers to when checking "does the impl match BZ?"
>
> **What's substrate-verified**: the four `unimplemented!()` sites in arith.rs
> are at lines 441, 467, 492, 512 (per Read at session start). The f64 fast
> path is correct AND covers all f64-representable Normal × Normal cases. The
> multi-limb path triggers when `f64_path_eligible` returns false — i.e., when
> any input or output has > 53 significant mantissa bits.

---

## 0. Encoding refresher — what the algorithms operate on

Per `ty.rs` (lines 8-28) and `conversions.rs::from_f64`:

For `kind = Normal`:
- `limbs: Vec<u64>` little-endian. `limbs.len() == ⌈precision_bits / 64⌉`.
- The bit at position `(precision_bits - 1) % 64` of `limbs[n_limbs - 1]` is set
  (canonical-form / top-bit-set invariant).
- All bits above the top bit in the top limb are zero (implicit by the invariant).
- Bits below the top bit, down through `limbs[0]`, carry the rest of the mantissa.
- `exponent: i64` = binary place value of the top set bit. The numeric value:
  `(-1)^sign · M · 2^(exponent - precision_bits + 1)` where M is the unsigned
  little-endian integer in `limbs`.

Worked example (per ty.rs lines 21-28): `+1.5` at p=53 has `limbs = [0x0018_0000_0000_0000]` (top bit at position 52, second-highest at position 51), `exponent = 0`, `sign = false`. M = 3 · 2^51 = 0x0018_0000_0000_0000. Value = M · 2^(0-53+1) = M · 2^-52 = (3·2^51)·2^-52 = 1.5. ✓

**Critical implication for arithmetic**: arithmetic operates on M (the magnitude integer) and `exponent` separately. After every op, the result must be re-canonicalized: shift M so its top bit is at position `(p_out - 1) % 64`, adjust exponent to compensate.

---

## 1. BZ Algorithm 3.1 — Addition / Subtraction

**BZ reference**: §3.1, "Addition and Subtraction." Algorithm 3.1 box on p.117 of the 2nd edition.

### 1.1 Inputs / outputs

```
inputs:  a, b : BigFloat (Normal)         -- working precision_bits p_a, p_b
         result_precision: u32 (= max(p_a, p_b) per BZ §3.1 convention)
         rounding: RoundingMode
output:  c : BigFloat at result_precision such that c = round(a + b, rounding)
```

### 1.2 The algorithm — alignment + add + canonicalize

**Step 1 — Choose larger-magnitude operand**. Compare `|a|` and `|b|` by exponent first, then by magnitude limb-by-limb (high-to-low). Without loss of generality, let `|a| >= |b|`. If a swap occurred, also swap signs in the bookkeeping.

**Step 2 — Compute the exponent gap**. Let `δ = a.exponent - b.exponent`. (Always ≥ 0 by step 1.)

**Step 3 — Decide same-sign or different-sign path**. Result sign = `a.sign` (the larger-magnitude operand). If `a.sign == b.sign`, perform integer addition of magnitudes. Otherwise, perform integer subtraction (`a_magnitude - b_magnitude`).

**Step 4 — Align b's mantissa**. Construct an "extended" bit-string for b shifted right by δ positions, computing in a working buffer of width `result_precision + g` bits where g is the guard count (see §1.4 below). Bits that fall off the right of the buffer are accumulated into a *sticky* indicator (used by directed rounding).

**Step 5 — Integer add or subtract**. Perform a multi-limb integer add/sub of the aligned mantissas. For add: standard ripple-carry across limbs (intrinsic `u64::overflowing_add` chains). For sub: `a_mag - b_aligned_mag` always non-negative (by step 1).

**Step 6 — Canonicalize the result**. After add: there may be a carry-out into a new top bit (one position above the original top). After sub: there may be leading zeros to skim (`leading_zeros` on the top limb, then on the next, etc.). Adjust `exponent` to compensate. Result mantissa now has top-bit-set at position `(result_precision + g - 1) % 64` of its top limb in the working buffer.

**Step 7 — Round to `result_precision` per `RoundingMode`**. Slice off the bottom `g` bits using the `(round_bit, sticky_bit)` pair to decide the direction. (See §1.4 for the table.)

**Step 8 — Final canonicalize**. Re-apply top-bit-set invariant. Trim or pad limb count to `⌈result_precision / 64⌉`.

### 1.3 Edge cases the unstub MUST handle

| Case | What happens | What the impl must guarantee |
|---|---|---|
| `a = 0`, `b ≠ 0` | already short-circuited at line 185 of arith.rs | (no change, this is the f64 path's responsibility to gate to the multi-limb only when both are Normal) |
| Same magnitude, opposite sign | exact cancellation → result is `+0` or `-0` | `mag(a) == mag(b)` after alignment with `δ = 0` → sub yields all-zero limbs → return `Self::zero` with sign per IEEE 754 (`+0` for default rounding, `-0` under `RoundTowardNegativeInfinity` per Surface 8) |
| `δ > result_precision + 1` | b's contribution is beyond round-bit position; result = a (with sticky update only for directed rounding) | small-magnitude operand contributes ONLY to the sticky bit (when shifted past the round-bit) — the round-bit itself is 0 |
| Carry-out on add | top bit moves up by 1 | after add, check if there's a bit at position `(p+g-1)%64 + 1` of the top limb (or a new high limb); if so, right-shift by 1, exponent += 1 |
| Massive cancellation on sub | result has many leading zeros | shift left until top bit is at canonical position; `exponent -= shift_amount` |
| Subnormal-magnitude result | exponent might underflow i64 (won't happen in practice — i64 is enormous) | no special handling needed for v2; BigFloat exponent has 64-bit range |

### 1.4 Guard bits and rounding (BZ §3.1.6)

For add/sub in v2 we use **g = 2 guard bits + sticky** for round-to-nearest-ties-even, OR **g = 0 + round bit + sticky** depending on implementation taste. BZ §3.1.6 shows that for add/sub specifically, `g = 1` is sufficient when paired with a sticky bit (because cancellation at most produces ONE leading zero bit when both operands are pre-canonicalized; rule of thumb is "no double rounding when guard ≥ 1").

**Recommended for arith.rs unstub**: use 2 guard bits + sticky (slightly conservative; matches the `p+50` framing for div/sqrt). The 2-extra-bit buffer is one extra integer add per limb (no real cost).

**Round-bit, guard-bit, sticky-bit table**:

After computing the unrounded mantissa at `p + 2` bits, examine the lowest 2 bits (`g0`, `g1` with `g1` higher) and the OR of all bits truncated below `g0` (`sticky`).

| RoundingMode | Round up iff |
|---|---|
| `RoundToNearestTiesEven` | `g1 = 1` AND (`g0 = 1` OR `sticky = 1` OR LSB-of-kept-mantissa = 1) |
| `RoundToNearestTiesAwayFromZero` | `g1 = 1` AND (`g0 = 1` OR `sticky = 1`) |
| `RoundTowardZero` | never (truncate) |
| `RoundTowardPositiveInfinity` | (sign positive) AND (`g1 = 1` OR `g0 = 1` OR `sticky = 1`) |
| `RoundTowardNegativeInfinity` | (sign negative) AND (`g1 = 1` OR `g0 = 1` OR `sticky = 1`) |

After the round-bit decision, increment the kept mantissa by 1 ULP (which may cascade carries; if the increment causes a top-bit overflow, right-shift by 1 and increment exponent).

### 1.5 What pathmaker can verify against

For the `add` path:
- `BigFloat::from_f64(1.0, 200).add(&BigFloat::from_f64(2.0, 200), Default)` should equal `BigFloat::from_f64(3.0, 200)` — bit-exact. This is the simplest multi-limb case where the f64 path doesn't apply (the BigFloat at p=200 has 4 limbs, and the f64 path's `f64_path_eligible` triggers because `1.0` and `2.0` are f64-representable). Wait — actually `f64_path_eligible` returns true for these because they ARE f64-representable. To force multi-limb path, pathmaker must construct operands with > 53 significant bits.
- Constructed forcing case: `BigFloat::from_f64(1.0, 200) + BigFloat::from_f64(2.0_f64.powi(-100), 200)` — both f64-representable individually, but the SUM has more than 53 significant bits (it spans `2^0` to `2^-100`, ~100 bits). The result's `f64_path_eligible` returns false because `to_f64` truncates the low bits and the round-trip equality check fails. Multi-limb add is invoked.
- Cancellation forcing case: `(a + b) - a` for `a` = some big number, `b` = some small number, computed at p=200. Tests round-bit handling at the cancellation boundary.
- Identity: `a + 0 = a` (already short-circuited; not a multi-limb test).
- Commutativity: `a + b = b + a` — pathmaker swaps inputs and checks bit-exact equality.

---

## 2. BZ Algorithm 3.3 — Multiplication (schoolbook)

**BZ reference**: §1.3.4 "Schoolbook Multiplication" + §3.3 "Multiplication" (the floating-point wrapper).

### 2.1 Inputs / outputs

```
inputs:  a, b : BigFloat (Normal)
         result_precision: u32 (= max(p_a, p_b))
         rounding: RoundingMode
output:  c : BigFloat at result_precision such that c = round(a × b, rounding)
```

### 2.2 The algorithm — schoolbook product + canonicalize + round

**Step 1 — Compute the full product mantissa**. Schoolbook integer multiplication of `a.limbs` (length `n_a = ⌈p_a / 64⌉`) and `b.limbs` (length `n_b = ⌈p_b / 64⌉`). Output is `n_a + n_b` limbs.

In Rust, the inner loop is:
```rust
for i in 0..n_a {
    let mut carry: u128 = 0;
    for j in 0..n_b {
        let prod = (a.limbs[i] as u128) * (b.limbs[j] as u128) + (out[i + j] as u128) + carry;
        out[i + j] = prod as u64;
        carry = prod >> 64;
    }
    out[i + n_b] = carry as u64;
}
```

`u128` covers `u64 × u64 + u64 + carry` since `(2^64 - 1)^2 + 2·(2^64 - 1) = 2^128 - 1`. No overflow.

**Step 2 — Compute the result exponent**. Both inputs are top-bit-set at position `(p - 1) % 64` of their top limb. The product's leading bit is at one of two positions:

Let `top_a = (p_a - 1)` (absolute bit position in `a.limbs` viewed as a single integer). Similarly `top_b = (p_b - 1)`. The product `M_a · M_b` has its top bit either at position `top_a + top_b` or `top_a + top_b + 1` (depending on whether the product's high bit "carries" — i.e., the top two leading bits of the inputs combined).

The numeric value of the product:
```
(M_a · M_b) · 2^((a.exponent - p_a + 1) + (b.exponent - p_b + 1))
= (product_integer) · 2^(a.exponent + b.exponent - p_a - p_b + 2)
```

If the product's top bit is at position `top_a + top_b + 1` (the "carry" case):
- Result exponent is `a.exponent + b.exponent + 1`.
- The product mantissa as a `(p_a + p_b)`-bit integer has its top bit at position `p_a + p_b - 1`.

If the product's top bit is at position `top_a + top_b` (the "no carry" case):
- Result exponent is `a.exponent + b.exponent`.
- The product mantissa has its top bit at position `p_a + p_b - 2`. **Left-shift by 1 to canonicalize** (top bit moves up to position `p_a + p_b - 1`).

Detect the case: examine bit `p_a + p_b - 1` of the product. If set → carry case (no shift needed). If unset → no-carry case (shift left by 1; exponent unchanged).

**Step 3 — Round to `result_precision` per `RoundingMode`**. The full product has `p_a + p_b` significant bits (top-bit-set at position `p_a + p_b - 1` after step 2). Truncate to `result_precision` significant bits, using the bits below position `(p_a + p_b - 1) - result_precision` as `(round_bit, sticky_bit)` per §1.4 table.

For DESIGN.md §3's "p+50 guard bits" framing for mul: schoolbook naturally produces `p_a + p_b` bits. For the in-scope precisions (p ≤ 1024 → 16 limbs → product has up to 32 limbs), this is far more than 50 guard bits. The "p+50 guard bits" guidance from DESIGN.md (which I had pegged on BZ §3.1.6 for div/sqrt) actually applies to Newton iteration — schoolbook mul has the full product available, so we don't ALLOCATE guard bits separately; we just round the full product back to `p`.

**Step 4 — Final canonicalize**. Pack into `result_precision`-bit BigFloat with top-bit-set invariant. Sign = `a.sign XOR b.sign` (already computed in `mul_sign` at arith.rs:93).

### 2.3 Edge cases

| Case | What happens | What the impl must guarantee |
|---|---|---|
| Carry round-up overflow | round-up produces a top-bit overflow (result mantissa now `2^p`, top bit at position p, not p-1) | right-shift by 1, increment exponent by 1, mantissa = `1 << (p-1)` (a power of 2 result) |
| Subnormal-tier output | exponent underflows | not relevant for BigFloat at v2 (i64 exponent is unbounded for practical use) |
| Both inputs at p=53 (forced via from_f64) | product has 106 significant bits | natural test: does multi-limb mul give the same 106-bit mantissa as `f64 × f64` followed by error-free transformation? Cross-check with DD's `two_product_fma`. |
| Result precision < either input precision | precision_bits is `max(p_a, p_b)`, so this never occurs in v2 | unused; but the `result_precision = max(...)` is enforced in `pub fn mul` at arith.rs:237 |

### 2.4 What pathmaker can verify against

- Identity: `a · 1 = a` (bit-exact). Where `1` is `BigFloat::from_f64(1.0, p)`.
- Commutativity: `a · b = b · a`.
- Distributivity (with rounding): `a · (b + c) ≈ a·b + a·c` to ULP precision (not bit-exact in general; rounding accumulates).
- Cross-check with DD: when both inputs fit in DD (operands exactly DD-representable), `BigFloat::mul(a, b, p=200)` should produce the same value as `DoubleDouble::from_parts(...).mul(...)` → `to_bigfloat(p=200)`. EFT primitives are exact at the (hi, lo) level.
- mpmath gold: `mpf(a) * mpf(b)` at `mp.prec = result_precision` should match bit-exact.

---

## 3. BZ Algorithm 3.5 — Division (Newton-Raphson)

**BZ reference**: §3.5 "Reciprocal and Division." Algorithm 3.5 box on p.130 of the 2nd ed.

### 3.1 Inputs / outputs

```
inputs:  a, b : BigFloat (Normal, b ≠ 0)
         result_precision: u32 (= max(p_a, p_b))
         rounding: RoundingMode
output:  c : BigFloat at result_precision such that c = round(a / b, rounding)
```

### 3.2 The algorithm — Newton on reciprocal, then multiply

**Phase A — Compute `1 / b` to `result_precision + 50` bits** via Newton-Raphson:

```
x_{k+1} = x_k · (2 - b · x_k)
```

Starting from `x_0 = f64::recip(b.to_f64())` (the f64 approximation of `1/b`, accurate to ~53 bits).

Convergence: each iteration *roughly doubles* the number of correct bits. `⌈log₂(result_precision / 53)⌉` iterations suffice to get to `result_precision` correct bits, plus 2 safety iterations to reach `result_precision + 50` correct bits. So:
```
n_iterations = ((result_precision as f64 / 53.0).log2().ceil() as u32) + 2
```

For p=200: `⌈log₂(3.77)⌉ + 2 = 2 + 2 = 4` iterations.
For p=500: `⌈log₂(9.43)⌉ + 2 = 4 + 2 = 6` iterations.
For p=1024: `⌈log₂(19.32)⌉ + 2 = 5 + 2 = 7` iterations.

**Per-iteration computation precision**: each iteration is computed at `result_precision + 50` bits. The multiplications and subtraction inside Newton are themselves multi-limb mul + sub — **the unstub for div depends on the unstub for add/sub/mul being complete first**. (Implementation dependency: do mul before div.)

**Phase B — Multiply by a**:

```
c_unrounded = a · x_n     (at result_precision + 50 bits)
```

This is one more mul at the wider precision.

**Phase C — Final round to `result_precision` per `RoundingMode`**.

### 3.3 The guard-bit argument (BZ §3.1.6 + §3.5)

Why `p + 50` guard bits suffice: BZ §3.1.6 establishes that for correctly-rounded division, the rounding bit + sticky bit are determined by the bits below position `p - 1` of the unrounded result. Because the Newton-iterated reciprocal has bounded error (≤ 1 ULP at the working precision), the multiplication `a · x_n` has error ≤ 2 ULP at the working precision, which means the final round at `p` is correct iff the working precision is `≥ p + ⌈log₂(p)⌉ + O(1)`.

For p ≤ 1024: `⌈log₂(1024)⌉ = 10`. BZ recommends safety margin of ~20 bits; `g = 50` is ~5x safety. **Bit-exact rigor**: the proper invariant for verification-tier is "compute at `g_max` bits where `g_max` is chosen large enough that the round-bit is unambiguously decidable." For DEC-031's bounded-precision regime (p ≤ 1024), `g = 50` is conservatively correct.

### 3.4 Edge cases

| Case | What happens | What the impl must guarantee |
|---|---|---|
| `b = 1.0` | result = a, exact | the Newton reciprocal of 1 IS 1 (`x_0 = 1.0` from f64); iteration is a no-op; multiplication by `a` returns `a`. The full algorithm should produce bit-exact `a`. |
| `b = 2^k` | result = a · 2^-k (exponent shift only, no mantissa change) | the reciprocal is also `2^-k`, which is exactly representable; multiplication amounts to mantissa = a's mantissa, exponent = a.exponent - k. **Detect this special case** at entry to avoid the full Newton (optimization, not correctness). |
| `a = b` | result = 1.0 exactly | the full Newton path produces `b · x_n = 1.0 ± ε` and `a · x_n = 1.0 ± 2ε`; with 50 guard bits + final round, `1.0` is the correctly-rounded answer. **Bit-exact**: verify the Newton's last iteration converges to the exact reciprocal (which is f64-representable) — no rounding error. |
| `a / a` for arbitrary a | result = 1.0 exactly | same argument as above; this is a §6 #13-style invariant for div (the multiplicative identity round-trip). Test: `BigFloat::from_f64(π, 200).div(&BigFloat::from_f64(π, 200), Default) == BigFloat::from_f64(1.0, 200)`. |
| Newton seed flips sign | shouldn't (b > 0 here; if `b.sign = true`, sign-handling is at the public API level, not Newton level) | always operate on `|b|` in Newton; restore sign at end via `mul_sign(a.sign, b.sign)` |
| Very small b (near subnormal-magnitude) | f64 reciprocal might overflow | guard `x_0` against `f64::INFINITY`. If `b.to_f64()` is subnormal or overflow, scale by 2^k first. **Implementation note**: scale b to be in `[0.5, 1.0)` for the f64 seed (multiply b by 2^-b.exponent), then unscale at the end. This is BZ §3.5's "scaled reciprocal" technique. |

### 3.5 What pathmaker can verify against

- Identity: `a / 1 = a` (bit-exact when a's value fits in f64).
- Inverse: `a · (1/a) ≈ 1` to within 1 ULP at result_precision.
- Self-divide: `a / a = 1.0` bit-exact.
- Commutativity-of-args (none — div is non-commutative).
- mpmath gold: `mpf(a) / mpf(b)` at `mp.prec = result_precision` should match bit-exact.

---

## 4. BZ Algorithm 3.10 — Square Root (Newton iteration)

**BZ reference**: §3.10 "Square Root." Algorithm 3.10 box on p.155 of the 2nd ed.

### 4.1 Inputs / outputs

```
inputs:  a : BigFloat (Normal, a ≥ 0)
         result_precision: u32 (= a.precision_bits)
         rounding: RoundingMode
output:  c : BigFloat at result_precision such that c = round(√a, rounding)
```

### 4.2 The algorithm — Newton iteration

Newton on `f(x) = x² - a = 0`:
```
x_{k+1} = (x_k + a / x_k) / 2
```

Starting from `x_0 = f64::sqrt(a.to_f64())` (the f64 approximation of `√a`, accurate to ~53 bits).

**Convergence**: same quadratic-convergence shape as div. `⌈log₂(p / 53)⌉ + 2` iterations to `result_precision + 50` correct bits.

**Per-iteration precision**: each iteration is computed at `result_precision + 50` bits and uses one `div` (a / x_k), one `add` (x_k + (a/x_k)), one `div by 2` (= exponent decrement, exact, no rounding). **Implementation dependency**: sqrt depends on add and div, which depend on add and mul. So the implementation order is: add/sub → mul → div → sqrt.

**Final round to `result_precision` per `RoundingMode`** at the end.

### 4.3 The Karp-Markstein optimization — *NOT for v2*

BZ §3.10 also discusses Karp-Markstein's "Newton without division" trick: iterate `x_{k+1} = x_k · (3 - a · x_k²) / 2` to compute `1/√a`, then multiply by `a` to get `√a`. This avoids the per-iteration division (replacing it with two muls), which is faster for large p. **Defer to v3.** v2 uses BZ Algorithm 3.10 verbatim — it's slower per iteration but simpler to verify.

### 4.4 Edge cases

| Case | What happens | What the impl must guarantee |
|---|---|---|
| `a = 0` | result = 0 | already short-circuited at arith.rs:362 (kind dispatch). Multi-limb path doesn't see Zero. |
| `a < 0` (negative non-zero) | result = NaN | already handled at arith.rs:358. Multi-limb path doesn't see negative. |
| `a = 1.0` | result = 1.0 exactly | x_0 = 1.0; iteration is `(1 + 1/1)/2 = 1.0`. Bit-exact. |
| `a = 4.0` | result = 2.0 exactly | x_0 = 2.0 (f64 sqrt is exact for perfect squares); iteration is `(2 + 4/2)/2 = 2.0`. Bit-exact. |
| `a = perfect square in BigFloat` | result is exact | x_0 from f64 may not be exact (if a is multi-limb), but Newton converges to the exact value; final round at `p + 50` guard bits identifies the exact answer. **This is a verification-tier test case**: `bf_a = (some BigFloat).square(); bf_a.sqrt() == |some BigFloat|`. |
| `a = 2.0` | result = √2 ≈ 1.4142135... at result_precision | classic irrational; Newton converges to the truncation at result_precision. **mpmath gold** is the reference. |
| Subnormal-magnitude a | x_0 from f64 might be inaccurate | scale a by `2^(2k)` to be in `[1, 4)` for the f64 seed (sqrt scales by `2^k`); unscale at the end. |

### 4.5 What pathmaker can verify against

- `√(a²) = a` for any positive Normal a. Bit-exact at `result_precision`.
- `(√a)² ≈ a` to within 1 ULP at result_precision (cumulative rounding from sqrt + mul).
- `√1 = 1` bit-exact.
- `√4 = 2` bit-exact.
- mpmath gold: `mpf(a).sqrt()` at `mp.prec = result_precision`.
- Cross-check: `BigFloat::sqrt(2)` at p=200 should match the first 200 bits of √2 from any reference table.

---

## 5. Implementation order — pathmaker's sequencing

Internal dependencies:

```
add/sub (3.1)          ← independent; do FIRST
   ↓
mul (3.3)              ← uses ONLY integer-limb operations, no add dependency at the integer-mul level. Can be done in parallel with add/sub.
   ↓
div (3.5)              ← uses add (Newton: 2 - b·x), mul (b·x and a·x_n), exponent arithmetic. Depends on add + mul.
   ↓
sqrt (3.10)            ← uses add (x_k + a/x_k), div (a / x_k), exponent halving. Depends on add + div.
```

**Recommended order**:
1. **add / sub** (BZ 3.1) — foundational; aristotle's invariants (commutativity, identity, cancellation) verifiable here.
2. **mul** (BZ 3.3) — schoolbook integer mul + product-exponent + final round. Independent of add/sub at the integer-mul level (the Newton-style iteration ISN'T mul; mul is just schoolbook).
3. **div** (BZ 3.5) — Newton needs add + mul. After 1 and 2 land.
4. **sqrt** (BZ 3.10) — Newton needs add + div. After 1, 2, 3 land.

This is also the order in which the proptests will go green (Surface 3 cross-precision consistency), so pathmaker can land each as a separate commit and have CI validate incrementally.

---

## 6. Cross-precision consistency check (Surface 3-adjacent)

The cross-precision consistency idiom from oracle-validation.md §1.2 + DESIGN.md §5 Q3 (ratified yes): for any op `f` and inputs `(a, b)`:

```
∀ p₁, p₂ with 53 ≤ p₁ ≤ p₂ ≤ 1024:
    f(round_to_p₁(a), round_to_p₁(b), p₁, rounding) ==
        round_to_p₁(f(a, b, p₂, rounding))
```

In words: computing at higher precision then rounding down should agree with computing at lower precision directly. **This catches off-by-one errors in the guard-bit logic.**

For the unstub:
- After implementing add at precision `p`, run the consistency check `add(a, b, p_low) == round(add(a, b, p_high), p_low)` for `p_low ∈ {53, 107, 200}`, `p_high = 1024`.
- If consistency fails: the round-bit / sticky-bit logic is wrong. The disagreement isolates the bit position where the algorithm rounds incorrectly.
- Same harness for mul, div, sqrt.

**Scientist's harness setup** (per the team-briefing assignment): I recommend the harness construct adversarial inputs at the round-bit boundary specifically. Two classes:

1. **Round-bit-on-the-edge**: inputs whose result mantissa, computed at `p_high`, has a `0.5` ULP at position `p_low`. These are the ties-to-even cases. Ties-to-even should round to the even alternative; bit-exactness is testable.

2. **Sticky-bit-on-the-edge**: inputs whose result has all-zero bits below the round bit at `p_low` for some sub-precision but a non-zero bit somewhere below for `p_high`. This is the "looks exact at p_low but isn't actually exact" case. Round-up under directed rounding, round-to-even should round-down.

**Adversarial input generator** (sketch):
```rust
proptest! {
    #[test]
    fn cross_precision_consistency_add(
        a_bits: u64, b_bits: u64,
        p_low in 107u32..=200, p_high_offset in 50u32..=824
    ) {
        let p_high = p_low + p_high_offset;
        let a = f64::from_bits(a_bits);
        let b = f64::from_bits(b_bits);
        prop_assume!(a.is_finite() && b.is_finite() && !a.is_nan() && !b.is_nan());

        let a_low = BigFloat::from_f64(a, p_low);
        let b_low = BigFloat::from_f64(b, p_low);
        let a_high = BigFloat::from_f64(a, p_high);
        let b_high = BigFloat::from_f64(b, p_high);

        let direct = a_low.add(&b_low, RoundingMode::RoundToNearestTiesEven);
        let via_high = {
            let sum_high = a_high.add(&b_high, RoundingMode::RoundToNearestTiesEven);
            // Round sum_high down to p_low precision.
            sum_high.with_precision_rounded(p_low, RoundingMode::RoundToNearestTiesEven)
        };

        prop_assert_eq!(direct, via_high, "cross-precision consistency failed for a={a:e}, b={b:e}, p_low={p_low}, p_high={p_high}");
    }
}
```

Note: `with_precision_rounded` doesn't yet exist in BigFloat — pathmaker may need to add it. It's "round my mantissa to `p_low` significant bits, return a new BigFloat at `p_low`" — equivalent to an identity-op with explicit precision change. Useful for the cross-precision idiom.

---

## 7. NaN payload propagation across multi-limb arithmetic

Per DESIGN.md §5 Q3 (NaN payload preserved): the multi-limb arithmetic NEVER touches `BigFloatKind::NaN`-tagged operands — they're short-circuited at the kind-dispatch level (arith.rs:122-139, 240-256, 304, 354). **The unstub doesn't need to handle NaN**; the special-value dispatch already does.

**However**: the multi-limb arithmetic produces results of `BigFloatKind::Normal`. If the result of the multi-limb add/sub/mul/div is NaN by IEEE 754 semantics (e.g., `0/0` produces NaN — but `0/0` is short-circuited at kind dispatch already), the only NaN-from-Normal-arithmetic case is overflow → `+Inf` (kind = Infinity, not NaN). **No NaN payload is created by multi-limb arithmetic**; payloads only flow through input → output via the special-value dispatch.

This is good. Pathmaker doesn't need to think about NaN payloads in the unstub.

---

## 8. Cross-references to invariant docs

- **Diamond commutativity (§6 #3)**: not affected by the unstub (it's about `from_f64` vs `from_dd`-with-`lo=0`, which is structural short-circuit, not multi-limb arith). The unstub's correctness for multi-limb DD-source case (`from_dd` with `lo ≠ 0` at `p ≥ 106`) DOES depend on multi-limb add — but the diamond invariant only constrains the `lo = 0` case.

- **Round-trip identity (§6 #13)**: not affected by the unstub (it's about `from_f64(_, 53).to_f64()`, which is bit-shuffling, no arithmetic).

- **Aristotle's gauntlet Surface 6** (NaN payload preservation regression): handled by special-value dispatch; multi-limb arith doesn't see NaN.

- **Aristotle's gauntlet Surface 7** (DD↔BigFloat boundary): the `p < 53 reject / p ∈ [53, 106) round / p ≥ 106 strict` boundary is handled in `from_dd_with_rounding` (a separate API surface, not in arith.rs). The unstub doesn't introduce new boundary cases here.

- **Aristotle's gauntlet Surface 8** (zero arithmetic cross-rounding-mode): handled in `add_zero_arithmetic_sign` and `sub_zero_arithmetic_sign` (arith.rs:65-89). Multi-limb path doesn't see zero (already short-circuited at line 178+185+191).

- **F13 antibody pattern** (rules with scope preconditions need antibodies enforcing the precondition at construction): the BZ algorithms have preconditions baked in:
  - BZ 3.1 add/sub: both inputs must be `Normal` and non-zero. **Antibody**: kind dispatch at arith.rs:142-176 + zero-shortcut at 179-194.
  - BZ 3.3 mul: both inputs must be `Normal`. **Antibody**: kind dispatch at arith.rs:261-282.
  - BZ 3.5 div: divisor must be `Normal` non-zero. **Antibody**: kind dispatch at arith.rs:310-336.
  - BZ 3.10 sqrt: input must be `Normal` and non-negative. **Antibody**: kind + sign dispatch at arith.rs:354-376.

All four BZ-algorithm preconditions ARE enforced at the boundary by the existing dispatch. The unstub doesn't need to add new antibodies — the existing F13-shaped checks are correct and load-bearing. **What the unstub DOES need**: assert the canonical-form invariant at the end of every multi-limb computation (top bit set at `(p-1) % 64`, limb count = `⌈p/64⌉`). That's the multi-limb-canonicalization invariant, and it's what BZ §3.1 step 6 + §3.3 step 4 + the final-round step in §3.5 / §3.10 ensure.

---

## 9. Provenance + handoff

- Authored 2026-05-08 by math-researcher in team `tambear-sweep31-finish`.
- Substrate verified at session start: `git -C R:/tambear log --oneline` shows c2798f5 + 33d3849 as the most recent commits; `arith.rs` 519 lines with four `unimplemented!()` sites at the multi-limb path; `ty.rs` storage-layout invariants + `conversions.rs` from_f64/to_f64 confirm the encoding.
- BZ algorithm references: 2nd ed. (2010), Cambridge University Press. Algorithms 3.1 (p.117), 3.3 (p.124), 3.5 (p.130), 3.10 (p.155).
- Cross-checked against DESIGN.md §3 algorithm dispatch table (which made the SAME algorithm choices but at ratification level, not impl level).
- Cross-checked against oracle-validation.md §1.2 (cross-precision consistency check).
- This is a tactical reference for pathmaker. Aristotle is the invariant-pressure-test; adversarial designs the proptests; scientist designs the mpmath oracle harness. The reference IS the BZ text + the DESIGN.md choices, organized for the specific BigFloat encoding shipped at 33d3849.

**Open questions for pathmaker** (to be resolved during impl, NOT before):
- Q1: `with_precision_rounded(p, rounding)` — add to BigFloat as a new method, or fold into a `round_to_precision` internal helper? My recommendation: add as a public method. It's load-bearing for both the cross-precision consistency check AND the final-round step of every BZ algorithm.
- Q2: Sticky-bit accumulator for add/sub — accumulate as `bool` (any bit-below = sticky) or as the full discarded-bits suffix (for mul-exact-detection)? My recommendation: `bool` is sufficient for round-to-nearest; the directed rounding modes also need bool. For Karatsuba (v3) we'd want richer info but not for v2.
- Q3: Newton iteration count — fixed at `⌈log₂(p/53)⌉ + 2` per DESIGN.md §3, OR adaptive (loop until two consecutive iterations agree at `p + 50` bits)? My recommendation: fixed. Adaptive adds termination-correctness as a proof obligation; fixed is BZ-text-aligned and easier to verify. The fixed count is conservative — quadratic convergence overshoots fast.

These are for pathmaker. Math-researcher returns to invariant verification once the impl lands.
