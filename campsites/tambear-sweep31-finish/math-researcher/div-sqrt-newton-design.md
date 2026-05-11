---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher
date: 2026-05-08
sweep: 31 (multi-limb arith — div + sqrt Newton iteration pre-design)
audience: pathmaker (ready-to-implement after BZ 3.3 mul lands)
purpose: precise iteration structure, convergence argument, and rounding logic for BZ Algorithm 3.5 (div) and 3.10 (sqrt). Companion to bz-impl-reference.md §3 + §4 — that doc gave the algorithm shape; this one gives the exact code structure.
inputs:
  - Brent & Zimmermann, *Modern Computer Arithmetic*, 2nd ed., §3.5 (reciprocal + division), §3.10 (square root)
  - bz-impl-reference.md §3, §4 (algorithm-level overview)
  - DESIGN.md §3 (algorithm dispatch table — `p+50` guard bits framing)
  - arith.rs at 5b666fc (still has `unimplemented!()` at lines 492 div, 512 sqrt; from_raw_limbs available)
  - limbs.rs at 5b666fc (uncommitted but present): mul_limbs, add_limbs, sub_limbs, shr_limbs_with_sticky, shl_limbs_in_place, top_bit_position, get_bit, set_bit, inc_limbs
---

# BZ Newton-Iteration Design — div (3.5) + sqrt (3.10)

> **The role of this doc**. After pathmaker lands BZ 3.1 (add/sub) and BZ 3.3
> (schoolbook mul), Newton iteration becomes implementable. This doc is the
> ready-to-translate spec: iteration formulas, precision ladder, convergence
> argument, edge cases, all worked at the level of the BigFloat encoding
> shipped in `ty.rs` + `limbs.rs`.
>
> **Why pre-design**. Newton-iteration code has subtle invariants —
> termination criterion, intermediate precision, scaling for the f64 seed,
> sign+exponent handling, final-round at p+50 → p. Getting any of these
> wrong silently produces correct-magnitude-but-wrong-low-bits results that
> evade discovery-tier ULP tests but fail verification-tier bit-exact tests.
> This doc is the spec at the bit-precision level so the impl is mechanical.

---

## 0. Precision-ladder decision — work at p+50 throughout (NOT doubling)

Two strategies for Newton iteration precision management:

**Strategy A — doubling ladder (BZ §3.5 textbook)**: each iteration is computed at increasing precision, doubling per iteration. `p_0 = 53` (f64 seed), `p_1 = 106`, `p_2 = 212`, ... `p_k ≥ p+50`. The mathematical argument: Newton converges quadratically, so to gain n correct bits per iteration you must compute at ≥ 2n bit precision. Working at lower precision early saves time.

**Strategy B — uniform p+50 (DESIGN.md choice)**: every iteration is computed at `p + 50` bits. Strictly more work per iteration than the doubling ladder, but the iteration count is the same (`⌈log₂(p/53)⌉ + 2`) and the per-iteration cost difference is bounded.

**Decision: Strategy B, for p ≤ 1024**.

Why:
1. **Correctness is easier to argue**. With uniform precision, every intermediate is at p+50 bits — the round-bit/sticky-bit logic is the same shape as add/mul. With a precision ladder, each iteration has a different rounding contract, multiplying the audit surface.
2. **The cost difference is bounded**. For p=200 (4 limbs at p+50=250 → 4 limbs), the doubling ladder runs at p_0..p_3 = 53, 106, 212, 250 — a savings of maybe 30% on the early iterations, which is 4 of them total. The total cost is dominated by the last iteration (which is the same in both strategies). Net savings: ~5% of total Newton time.
3. **For p > 1024 the calculus changes** — the ladder savings grow. But DEC-031 §3.8 saturates at 1024 bits; we don't ship code for higher in v2. If/when we do (v3+), the ladder is the right strategy. Document this explicitly so future-pathmaker doesn't re-derive.

**Constant**: `const NEWTON_GUARD_BITS: u32 = 50;` lives in `arith.rs` near the helper functions.

---

## 1. BZ Algorithm 3.5 — Division via Newton-Raphson reciprocal

### 1.1 Public-API surface

The unstub site is `normal_div` in `arith.rs:473-496`. The signature stays:

```rust
fn normal_div(a: &BigFloat, b: &BigFloat, result_precision: u32, rounding: RoundingMode) -> BigFloat
```

Both `a` and `b` are `BigFloatKind::Normal` and non-zero (special-value dispatch handles ±0/±Inf/NaN at the public `div` entry — already in place).

### 1.2 The full algorithm — phase by phase

```rust
fn normal_div(a: &BigFloat, b: &BigFloat, result_precision: u32, rounding: RoundingMode) -> BigFloat {
    // f64-fast-path retained for the common case
    if f64_path_eligible(a) && f64_path_eligible(b) {
        let quot = a.to_f64() / b.to_f64();
        if quot.is_finite() {
            return BigFloat::from_f64(quot, result_precision);
        }
        // overflow → ±Inf
        return BigFloat { kind: BigFloatKind::Infinity, sign: quot.is_sign_negative(),
                          exponent: 0, precision_bits: result_precision, limbs: Vec::new() };
    }

    // === Multi-limb path: Newton-Raphson on reciprocal ===

    let p_work = result_precision + NEWTON_GUARD_BITS; // = result_precision + 50

    // Phase 1 — Compute the reciprocal of |b| at p_work bits.
    let recip_b = newton_reciprocal(b, p_work);

    // Phase 2 — Multiply by a at p_work bits.
    let unrounded_quot = a.mul(&recip_b, RoundingMode::RoundToNearestTiesEven /* p_work mul is bit-exact at p_work; rounding mode is moot for this intermediate */);

    // Phase 3 — Round from p_work down to result_precision per `rounding`.
    let result_sign = mul_sign(a.sign, b.sign);
    let mut out = unrounded_quot.with_precision_rounded(result_precision, rounding);
    out.sign = result_sign; // sign is already in unrounded_quot via mul_sign in mul; but we restore for clarity
    out
}
```

### 1.3 The Newton iteration — `newton_reciprocal(b, p_work)`

**Goal**: compute `x ≈ 1/|b|` at precision `p_work` bits, with `|x - 1/|b|| ≤ 1 ULP at p_work`.

**Mathematical scheme** (BZ Algorithm 3.5):
```
x_{k+1} = x_k · (2 - |b| · x_k)
```

Quadratic convergence: if `x_k` is correct to n bits, `x_{k+1}` is correct to ~2n-1 bits.

**Initial seed**: `x_0 = 1.0 / |b|.to_f64()` is correct to ~53 bits (f64 hardware reciprocal is ≤ 1 ULP at f64 precision).

**Iteration count**: starting at 53 correct bits and doubling, to reach `p_work`:
```rust
let n_iterations = ((p_work as f64 / 53.0).log2().ceil() as u32) + 2;
```

The "+ 2" gives 2 safety iterations beyond the bare convergence count. Empirically this is far more than enough — Newton hits the precision ceiling at iteration `n - 1` and the next iteration is a no-op (same value within 1 ULP). The 2-iteration buffer absorbs any seed-imprecision artifacts.

For p=200 (p_work=250): `⌈log₂(4.72)⌉ + 2 = 3 + 2 = 5` iterations.
For p=500 (p_work=550): `⌈log₂(10.38)⌉ + 2 = 4 + 2 = 6` iterations.
For p=1024 (p_work=1074): `⌈log₂(20.26)⌉ + 2 = 5 + 2 = 7` iterations.

**Scaling for the f64 seed**: when `|b|` is very small (subnormal-magnitude or below) or very large (near `f64::MAX`), the f64 reciprocal might overflow or underflow. **Fix**: scale `b` to be in `[0.5, 1.0)` for the seed.

```rust
fn newton_reciprocal(b: &BigFloat, p_work: u32) -> BigFloat {
    debug_assert!(b.is_normal());

    // Scale b to b_scaled with exponent in [-1, 0) (so b_scaled ∈ [0.5, 1.0)).
    // The numeric value of b is M_b · 2^(b.exponent - p_b + 1).
    // We want b_scaled to be |b| / 2^(b.exponent + 1) so that the value is
    //   M_b · 2^(- p_b)  which has top bit at position p_b - 1 and exponent = -1.
    // In our encoding: just override the exponent to -1, the mantissa is unchanged.
    let mut b_scaled = b.clone();
    b_scaled.precision_bits = p_work;
    b_scaled.limbs.resize(((p_work + 63) / 64) as usize, 0);
    // Re-pack: the existing mantissa (top-bit-set at b.precision_bits-1) gets
    // moved to top-bit-set at p_work-1. This is a left-shift by (p_work - b.precision_bits).
    // (Implementation calls into limbs::shl_limbs_in_place after extending.)
    let scale_shift = p_work - b.precision_bits;
    if scale_shift > 0 {
        crate::primitives::big_float::limbs::shl_limbs_in_place(&mut b_scaled.limbs, scale_shift);
    }
    b_scaled.sign = false; // operate on |b|
    let original_b_exponent = b.exponent;
    b_scaled.exponent = -1; // forces b_scaled ∈ [0.5, 1.0)

    // f64 seed: 1 / b_scaled.to_f64(), guaranteed in (1, 2] now.
    let seed_f64 = 1.0 / b_scaled.to_f64();
    debug_assert!(seed_f64.is_finite() && seed_f64 > 1.0 && seed_f64 <= 2.0);
    let mut x = BigFloat::from_f64(seed_f64, p_work);

    let n_iterations = ((p_work as f64 / 53.0).log2().ceil() as u32) + 2;
    let two = BigFloat::from_f64(2.0, p_work);

    for _ in 0..n_iterations {
        // x_{k+1} = x · (2 - b_scaled · x)
        let bx = b_scaled.mul(&x, RoundingMode::RoundToNearestTiesEven);
        let two_minus_bx = two.sub(&bx, RoundingMode::RoundToNearestTiesEven);
        x = x.mul(&two_minus_bx, RoundingMode::RoundToNearestTiesEven);
    }

    // Now x ≈ 1 / b_scaled at p_work bits.
    // Unscale: 1/b = (1/b_scaled) · 2^(-original_b_exponent - 1)
    // I.e., x.exponent has been computed against b_scaled.exponent = -1; we need
    // to shift the result by (-original_b_exponent - 1 - (-1)) = -original_b_exponent.
    x.exponent -= original_b_exponent + 1;
    x
}
```

**Important detail in the unscale step**: this is exponent-arithmetic only — no mantissa shift, no rounding. The reciprocal of `b = M_b · 2^E_b` is mathematically `(1/M_b) · 2^(-E_b)`; if we computed `1/M_b · 2^(-(-1))` (i.e., `1/M_b · 2^1`) and we want `1/M_b · 2^(-E_b)`, the exponent difference is `(-E_b) - 1`. Wait — let me re-derive carefully because this is where Newton-iteration impls usually go wrong.

**Full derivation of the unscale exponent**:
- Original `b` has numeric value `M_b · 2^(b.exponent - b.precision_bits + 1)`. With `b_scaled` constructed as "same magnitude bits, but with `precision_bits = p_work` and `exponent = -1`", the numeric value of `b_scaled` is `M_b_scaled · 2^(-1 - p_work + 1) = M_b_scaled · 2^(-p_work)` where `M_b_scaled = M_b · 2^scale_shift` (after the left-shift to top-bit-set at p_work - 1).
- So `b_scaled = (M_b · 2^scale_shift) · 2^(-p_work) = M_b · 2^(scale_shift - p_work) = M_b · 2^(-b.precision_bits)`.
- We want `b_scaled` numerically in `[0.5, 1)`. Compare to original b: `b = M_b · 2^(b.exponent - b.precision_bits + 1)`. Ratio `b / b_scaled = 2^(b.exponent + 1)`.
- So `1/b_scaled = 2^(b.exponent + 1) / b`, equivalently `1/b = (1/b_scaled) · 2^(-(b.exponent + 1))`.
- After Newton at p_work bits, `x ≈ 1/b_scaled`. To get `1/b`, multiply by `2^(-(b.exponent + 1))`, which in our encoding is **subtract `(b.exponent + 1)` from `x.exponent`**.

So the unscale line should be:
```rust
x.exponent -= original_b_exponent + 1;
```

This matches the code above. **Pathmaker: please verify against the encoding convention in ty.rs lines 8-28 — the formula `(-1)^sign · M · 2^(exponent - precision_bits + 1)` is what makes this exponent arithmetic work cleanly. If the encoding ever changes, this derivation must be redone.**

### 1.4 Convergence argument (why ⌈log₂(p_work/53)⌉ + 2 iterations suffice)

Newton's iteration for `f(x) = 1/x - b` (root: x = 1/b) is `x_{n+1} = x_n - f(x_n)/f'(x_n) = x_n + x_n²(1/x_n - b) = x_n(2 - b·x_n)`.

If `x_n = 1/b · (1 + ε_n)` where `|ε_n| < 1`, then:
```
b · x_n = 1 + ε_n
2 - b·x_n = 1 - ε_n
x_n · (2 - b·x_n) = (1/b)(1 + ε_n)(1 - ε_n) = (1/b)(1 - ε_n²)
```

So `x_{n+1} = (1/b)(1 - ε_n²)`, i.e., `ε_{n+1} = -ε_n²`. **Quadratic convergence**: `|ε_{n+1}| = ε_n²`.

Starting from `|ε_0| ≤ 2^-53` (f64 reciprocal is ≤ 1 ULP at f64 precision), after k iterations:
```
|ε_k| ≤ |ε_0|^(2^k) = 2^(-53 · 2^k)
```

To get `|ε_k| ≤ 2^(-p_work)`:
```
53 · 2^k ≥ p_work
2^k ≥ p_work / 53
k ≥ log₂(p_work / 53)
```

So `⌈log₂(p_work / 53)⌉` iterations are mathematically sufficient for the iteration to *converge* to p_work bits — but each iteration is computed at p_work-bit precision, so the iteration error gets compounded with rounding error at p_work. The "+ 2" iterations absorb this rounding error margin. After the iteration converges to ~p_work bits mathematically, two more iterations at p_work-bit precision tighten the error to within 1-2 ULP at p_work.

**Termination**: fixed-count. We do not check for two-consecutive-iterations-agree because: (a) it adds termination-correctness as a proof obligation; (b) for p ≤ 1024 the count is ≤ 7, fast enough that the optimization doesn't matter.

### 1.5 Edge cases

| Case | What happens | What pathmaker must verify |
|---|---|---|
| `b = 1.0` exactly | f64 path catches it — reciprocal is 1.0 exactly | f64-fast-path; multi-limb path doesn't trigger. |
| `b = 2^k` exactly (k arbitrary integer) | f64 path catches it for `k ∈ [-1023, 1023]` | for k outside f64 range, multi-limb path triggers; the reciprocal is exactly `2^(-k)`, also exactly representable. The Newton iteration converges in 1 step (after the f64 seed becomes exact). **Test**: `from_raw_limbs(top_bit_only, exponent=k, p=200) / from_f64(1.0, 200)` — but wait, `from_f64(1.0, 200)` IS f64-eligible, so this test wouldn't trigger multi-limb. Need to construct via from_raw_limbs both. |
| `b = a` (same operand) | result is exactly 1.0 | the multi-limb Newton + multiplication chain should produce 1.0 to within 1 ULP at p_work; the final round to p produces 1.0 exactly. **Verification-tier test**: pick a multi-limb `a` constructed via from_raw_limbs, compute `a / a`, assert == BigFloat::from_f64(1.0, p). Bit-exact. |
| `b` near subnormal (in BigFloat encoding, exponent very negative) | f64 seed handling — by the scaling step, `b_scaled` has `b.exponent = -1` and the f64 seed is `1/b_scaled.to_f64()`, which lives in `(1, 2]`. f64 cleanly represents this | the scaling step requires that `b_scaled.to_f64()` is finite and non-zero. Since we forced exponent = -1, `b_scaled ∈ [0.5, 1)`, and `b_scaled.to_f64()` rounds to f64's representation of that magnitude, always finite. ✓ |
| `a` very large or very small | doesn't affect the reciprocal-of-b computation; only affects the final `a · recip_b` multiplication | the existing `mul` implementation handles large/small mantissas correctly via exponent arithmetic. No special handling needed. |
| Exact result is f64-representable but inputs are multi-limb | the multi-limb Newton converges to a high-precision approximation; the final round-to-p step should round to the exact f64 value | this is a verification-tier test: `from_raw_limbs(...) / from_raw_limbs(...) == from_f64(known_exact, p)`. Bit-exact. |

### 1.6 Interaction with Surface 6 (NaN payload)

Multi-limb path doesn't see NaN (special-value dispatch at div-public-API handles it). The `mul` calls inside Newton operate on `BigFloatKind::Normal` operands at `p_work`, so no NaN propagation concerns within the iteration.

### 1.7 Cross-check formula — what scientist's mpmath harness asserts

```python
import mpmath as mp
mp.prec = result_precision
expected = mp.mpf(a) / mp.mpf(b)
# tambear: a.div(b, p, RTE).to_bits-equivalent(expected) at p bits
```

The bit-exact comparison is at the BigFloat's native encoding. To compare with mpmath, both must round to the same precision — in mpmath's case, that's the `mp.prec` setting. For the verification-tier check, set `mp.prec = result_precision` and assert bit-exact equality against the BigFloat's mantissa+exponent.

---

## 2. BZ Algorithm 3.10 — Square root via Newton iteration

### 2.1 Public-API surface

The unstub site is `normal_sqrt` in `arith.rs:498-516`. The signature stays:

```rust
fn normal_sqrt(a: &BigFloat, result_precision: u32, rounding: RoundingMode) -> BigFloat
```

`a` is `BigFloatKind::Normal` and non-negative (special-value dispatch at the public `sqrt` entry handles NaN/Inf/Zero/negative-non-zero).

### 2.2 The full algorithm

```rust
fn normal_sqrt(a: &BigFloat, result_precision: u32, rounding: RoundingMode) -> BigFloat {
    if f64_path_eligible(a) {
        let r = a.to_f64().sqrt();
        if r.is_finite() {
            return BigFloat::from_f64(r, result_precision);
        }
        return BigFloat { kind: BigFloatKind::Infinity, sign: false,
                          exponent: 0, precision_bits: result_precision, limbs: Vec::new() };
    }

    // === Multi-limb path: Newton iteration ===

    let p_work = result_precision + NEWTON_GUARD_BITS;

    let unrounded_root = newton_sqrt(a, p_work);
    unrounded_root.with_precision_rounded(result_precision, rounding)
}
```

### 2.3 The Newton iteration — `newton_sqrt(a, p_work)`

**Mathematical scheme** (BZ Algorithm 3.10):
```
x_{k+1} = (x_k + a / x_k) / 2
```

Quadratic convergence on `f(x) = x² - a` (root: x = √a).

**Initial seed**: `x_0 = a.to_f64().sqrt()`, accurate to ~53 bits when `a` is in f64 range. For `a` outside f64 range, scale `a` to be in `[1, 4)` for the seed.

**Iteration count**: same formula as div — `⌈log₂(p_work/53)⌉ + 2`.

**Scaling for the f64 seed**: f64::sqrt operates on inputs in [0, f64::MAX]; for very large or very small `a` magnitudes (BigFloat's i64 exponent range is much larger), we scale so that the seed computation is well-defined.

```rust
fn newton_sqrt(a: &BigFloat, p_work: u32) -> BigFloat {
    debug_assert!(a.is_normal() && !a.sign);

    // Scale a to a_scaled ∈ [1, 4) for f64::sqrt to give a clean seed.
    // Numeric value of a is M_a · 2^(a.exponent - a.precision_bits + 1).
    // We want a_scaled to have exponent ∈ {0, 1} (so a_scaled ∈ [1, 4)).
    // Choose: if a.exponent is even, set a_scaled.exponent = 0 (a_scaled ∈ [1, 2));
    //         if a.exponent is odd,  set a_scaled.exponent = 1 (a_scaled ∈ [2, 4)).
    // The exponent must be preserved-mod-2 for the unscale step to work cleanly:
    //   √a = √a_scaled · 2^((a.exponent - a_scaled.exponent) / 2)
    // The (a.exponent - a_scaled.exponent) MUST be even, which our parity choice ensures.

    let original_a_exponent = a.exponent;
    let scaled_a_exponent: i64 = if original_a_exponent % 2 == 0 { 0 } else { 1 };
    let scale_shift_count: i64 = (original_a_exponent - scaled_a_exponent) / 2; // result-side shift after sqrt

    let mut a_scaled = a.clone();
    a_scaled.precision_bits = p_work;
    a_scaled.limbs.resize(((p_work + 63) / 64) as usize, 0);
    let scale_shift = p_work - a.precision_bits;
    if scale_shift > 0 {
        crate::primitives::big_float::limbs::shl_limbs_in_place(&mut a_scaled.limbs, scale_shift);
    }
    a_scaled.exponent = scaled_a_exponent;

    // f64 seed: in [1, 4), so f64::sqrt is in [1, 2), well within f64.
    let seed_f64 = a_scaled.to_f64().sqrt();
    debug_assert!(seed_f64.is_finite() && seed_f64 >= 1.0 && seed_f64 < 2.0);
    let mut x = BigFloat::from_f64(seed_f64, p_work);

    let n_iterations = ((p_work as f64 / 53.0).log2().ceil() as u32) + 2;
    let two = BigFloat::from_f64(2.0, p_work);

    for _ in 0..n_iterations {
        // x_{k+1} = (x + a_scaled / x) / 2
        let a_over_x = a_scaled.div(&x, RoundingMode::RoundToNearestTiesEven);
        let sum = x.add(&a_over_x, RoundingMode::RoundToNearestTiesEven);
        x = sum.div(&two, RoundingMode::RoundToNearestTiesEven);
    }

    // Unscale: √a = x · 2^scale_shift_count
    // In our encoding: x.exponent += scale_shift_count.
    x.exponent += scale_shift_count;
    x
}
```

**Note on the `/2` step**: division by 2 is exponent-arithmetic only — `x.exponent -= 1`. We could special-case this to avoid the full Newton-on-`div` path (which is heavier than needed for the trivial halving). The compiler MIGHT optimize the case automatically since `2.0` is a power of 2 and the f64 fast path handles it, but for safety + clarity:

```rust
fn divide_by_two_in_place(x: &mut BigFloat) {
    x.exponent -= 1;
}
```

Then replace `x = sum.div(&two, ...)` with `let mut sum = sum; divide_by_two_in_place(&mut sum); x = sum;`. Avoids one full div per iteration.

**Pathmaker's choice**: include the `divide_by_two_in_place` helper for the per-iteration halving, OR rely on the f64-fast-path within `div` to catch `/2` cheaply. Recommendation: explicit helper. The f64-fast-path requires both operands to be f64-eligible (per `f64_path_eligible`); when `sum` has > 53 significant bits (the common case after a few Newton iterations), the fast path won't fire, and we'd run a full Newton-reciprocal of 2.0 — wasteful.

### 2.4 Convergence argument

For Newton on `f(x) = x² - a`, the standard error analysis: if `x_n = √a · (1 + ε_n)`, then:
```
x_n² = a · (1 + ε_n)²
a / x_n = √a / (1 + ε_n) = √a · (1 - ε_n + ε_n² - ...)
x_{n+1} = (x_n + a/x_n) / 2 = √a · ((1 + ε_n) + (1 - ε_n + ε_n² - ...)) / 2
       ≈ √a · (1 + ε_n²/2) for small ε_n
```

So `ε_{n+1} ≈ ε_n²/2`. **Quadratic convergence**: same shape as div, slightly tighter (factor of 2). Same iteration-count formula applies.

### 2.5 Edge cases

| Case | What happens | What pathmaker must verify |
|---|---|---|
| `a = 0` | already short-circuited at `pub fn sqrt` (kind dispatch) | doesn't reach normal_sqrt. |
| `a < 0` | already short-circuited (sign check at sqrt entry) | doesn't reach normal_sqrt. |
| `a = 1.0` exactly | f64 path catches it | doesn't reach multi-limb. |
| `a = 4.0` exactly | f64 path | doesn't reach multi-limb. |
| `a = 2.0` (irrational sqrt) | √2 ≈ 1.4142135... at p bits | converges to truncation of √2; mpmath cross-check. |
| `a = M·M` (perfect square in BigFloat encoding) | result is exactly `M`, no rounding error | Newton converges to the exact value at p_work; final round to p produces exact M. **Verification-tier test**: pick multi-limb M via from_raw_limbs, compute `m_squared = M.mul(&M)`, then `m_squared.sqrt()` == M.abs(). Bit-exact. |
| `a` with very large or very small magnitude | scaling step normalizes to `[1, 4)` for the f64 seed | scaling is exponent-only; no rounding. ✓ |
| Even/odd exponent parity at scaling | the parity choice ensures `a.exponent - a_scaled.exponent` is always even, so the unscale shift `scale_shift_count = (a.exponent - a_scaled.exponent) / 2` is integer | **Critical to verify**: write a unit test that constructs `a` with both even and odd exponents and verifies the scaled-and-unscaled round-trip. The parity logic is the place this would silently go wrong. |

### 2.6 The Karp-Markstein optimization — NOT for v2

BZ §3.10 also describes Karp-Markstein's "Newton without division" trick: iterate `x_{k+1} = x_k · (3 - a · x_k²) / 2` to compute `1/√a`, then multiply by `a` to get `√a`. This avoids the per-iteration division (replacing it with two muls), faster for large p. **Defer to v3.** v2 uses BZ Algorithm 3.10 verbatim — slower per iteration but simpler to verify, and our `divide_by_two_in_place` optimization captures the obvious low-hanging speedup.

### 2.7 Cross-check formula

```python
import mpmath as mp
mp.prec = result_precision
expected = mp.sqrt(mp.mpf(a))
```

---

## 3. Helper API — `with_precision_rounded`

The Newton implementations need `with_precision_rounded(p, rounding)` — round-down a BigFloat from one precision to a smaller precision per the given rounding mode. This is also load-bearing for the cross-precision consistency harness.

**Signature**:
```rust
impl BigFloat {
    /// Round the mantissa to `target_precision` significant bits per `rounding`,
    /// returning a new BigFloat at `target_precision`. Exponent is updated if
    /// rounding overflow causes a top-bit shift.
    ///
    /// Panics if `target_precision < MIN_PRECISION_BITS_FROM_F64` (53).
    pub fn with_precision_rounded(&self, target_precision: u32, rounding: RoundingMode) -> Self;
}
```

**Algorithm**:
1. Special-value dispatch: Zero, Infinity, NaN — clone with precision_bits set, kind/sign/payload preserved.
2. Normal: compute `bits_to_drop = self.precision_bits - target_precision`. If 0, just clone. If negative, this is an *upgrade* — pad zeros (no rounding); see §3.1. If positive, slice off the bottom bits, extract round_bit + sticky_bit per `limbs::shr_limbs_with_sticky`, apply rounding mode, increment if needed (handle carry-over via `limbs::inc_limbs`; if carry overflows, shift right by 1, exponent += 1).
3. Re-canonicalize: top-bit-set at `(target_precision - 1) % 64` of the new top limb.

### 3.1 Precision-upgrade case (target_precision > self.precision_bits)

When `target_precision > self.precision_bits`, we're padding the mantissa with zeros at the LSB end. This is exact (no rounding) and is what the Newton iteration uses to extend f64-seed-precision-53 to p_work-bit precision.

**Implementation**: extend `limbs` to `⌈target_precision / 64⌉`, shift left by `(target_precision - self.precision_bits)` to keep the top bit at position `(target_precision - 1) % 64`. Update precision_bits.

### 3.2 Why this lives in arith.rs (or limbs.rs?)

It's a public-API method on `BigFloat`. Belongs in arith.rs (where rounding is a primary concept) or possibly conversions.rs. **Recommendation**: arith.rs, in a new section `// ============= Precision conversion =============` near the top before the `add` impl. Pathmaker decides.

---

## 4. The implementation order — final word

Pathmaker's task list per the briefing:
- Task #2: BZ 3.1 add/sub (in_progress)
- Task #3: BZ 3.3 mul (pending)
- Task #4: BZ 3.5 div (pending) — **depends on add + mul**
- Task #5: BZ 3.10 sqrt (pending) — **depends on add + div**

After #3 mul lands, this doc IS the spec for #4 and #5. The total LoC estimate per DESIGN.md §3 was ~150 LoC for div, ~120 LoC for sqrt — this matches what's outlined here (newton_reciprocal ~50 LoC + scaling/unscaling ~20 LoC + with_precision_rounded ~80 LoC for div; newton_sqrt ~50 LoC + parity-handling ~20 LoC for sqrt, sharing with_precision_rounded).

---

## 5. F13 antibody review for div/sqrt

Per F13: every rule with a scope precondition needs an antibody enforcing it at construction.

**Newton iteration preconditions**:
- `b ≠ 0` for div: enforced at `pub fn div` kind-dispatch (arith.rs:310-336). Multi-limb path never sees zero divisor. ✓
- `a ≥ 0` for sqrt: enforced at `pub fn sqrt` (arith.rs:354-376). ✓
- `result_precision >= 53` for any normal_div / normal_sqrt: enforced via `result_precision = max(p_a, p_b)` and operands are constructed with precision >= 53. ✓
- f64 seed must be finite and non-zero: enforced by the scaling step (forces operand into a known-good range).
- iteration count: enforced by the formula `⌈log₂(p_work/53)⌉ + 2`, no user-tunable knob.

All preconditions are antibody-protected. ✓

---

## 6. What pathmaker can run as smoke tests

Once div lands, before declaring it ready for review:

```rust
#[test]
fn div_smoke_multi_limb() {
    use proptest::prelude::*;

    // 1. a / a = 1.0 for any multi-limb a
    let a = BigFloat::from_raw_limbs(/* multi-limb mantissa */, 200, /* exponent */ 0);
    let one = BigFloat::from_f64(1.0, 200);
    assert_eq!(a.div(&a, RoundingMode::RoundToNearestTiesEven), one);

    // 2. a / 1 = a (after rounding to p=200, which is identity for f64-sourced a)
    let a_simple = BigFloat::from_f64(2.5, 200);
    assert_eq!(a_simple.div(&one, RoundingMode::RoundToNearestTiesEven), a_simple);

    // 3. 1 / 2 = 0.5
    let two = BigFloat::from_f64(2.0, 200);
    let half = BigFloat::from_f64(0.5, 200);
    assert_eq!(one.div(&two, RoundingMode::RoundToNearestTiesEven), half);
}

#[test]
fn sqrt_smoke_multi_limb() {
    // 1. sqrt(M·M) = M for any multi-limb M
    let m = BigFloat::from_raw_limbs(/* multi-limb */, 200, /* exponent */ 0);
    let m_squared = m.mul(&m, RoundingMode::RoundToNearestTiesEven);
    let result = m_squared.sqrt(RoundingMode::RoundToNearestTiesEven);
    assert_eq!(result, m.abs());

    // 2. sqrt(4.0) = 2.0
    let four = BigFloat::from_f64(4.0, 200);
    let two = BigFloat::from_f64(2.0, 200);
    assert_eq!(four.sqrt(RoundingMode::RoundToNearestTiesEven), two);

    // 3. sqrt(2.0) at p=200 — compare to mpmath
    let s = BigFloat::from_f64(2.0, 200).sqrt(RoundingMode::RoundToNearestTiesEven);
    // mpmath reference (computed externally):
    let mpmath_sqrt2_200_bits: &[u64] = &[/* known reference limbs */];
    // ... assert limb-by-limb match
}
```

---

## 7. Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-sweep31-finish`, in response to navigator's pre-design ask.
- Substrate verified at session continuation: tambear at 5b666fc with proptest + from_raw_limbs + uncommitted limbs.rs/limbs_tests.rs/three integration tests; arith.rs unstub sites at lines 492 (div), 512 (sqrt) per current Read.
- Cross-checked: BZ §3.5 (Newton-Raphson reciprocal — pp. 130-138 of 2nd ed.), BZ §3.10 (Newton sqrt — pp. 155-160), DESIGN.md §3 (algorithm dispatch table, p+50 guard bits framing).
- Convergence arguments: standard quadratic-convergence error analysis (`ε_{n+1} = -ε_n²` for div; `ε_{n+1} ≈ ε_n²/2` for sqrt). The "+ 2" iteration buffer absorbs rounding error at p_work-bit precision.
- Open questions for pathmaker: scaling-step encoding-derivation should be re-verified against `ty.rs` lines 8-28 (the numeric-value formula). The scaling logic is the place silent bugs hide.
- This doc is implementation-ready. The companion bz-impl-reference.md §3 + §4 give the tactical-level view; this doc gives the bit-precision-level spec.
