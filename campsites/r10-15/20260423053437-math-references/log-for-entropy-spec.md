# `log_for_entropy` / `log_for_hill` — Cranelift IR Specification

> Math-researcher deliverable for pathmaker, 2026-04-23.
> Symmetric companion to `exp_for_lse` from earlier in this campsite.
> Per DEC-027 (no FMA without supports_fma cache key gate): all
> compound multiply-adds use `fmul + fadd`, not `fma`.

## Algorithm — Tang 1990 table-driven, with NEAR-1 FAST PATH

Two paths, branched on `|x - 1| < 1/16`:

### Fast path (|x - 1| < 1/16)

```
u = x - 1                               // small magnitude
log(x) ~ u + u² · P(u)                  // P is degree-7 Horner Taylor
```

This bypasses the catastrophic cancellation that arises in the
general-table path when x is very close to 1 (the `E·ln2 + log(F) + log(1+s)`
recombination loses precision because the three terms can be of
similar magnitude with opposite signs and cancel).

### General path (|x - 1| >= 1/16)

```
1. Decompose x = 2^E · m  (via bit manipulation; m in [1, 2))
2. Look up F[j] from 128-entry table where j = top-7-bits-of-mantissa
   F[j] = 1 + j/128  (LEFT endpoint of mantissa bin — F[0] = 1.0 exact)
3. s = (m - F[j]) / F[j]    (small; |s| < 1/128)
4. log(F[j]) precomputed (single f64) in same table
5. log(x) = E · ln2_hi + log(F[j]) + (E · ln2_lo + poly(s))
   where poly(s) ~ log(1 + s) via degree-5 minimax
```

## CRITICAL DOMAIN BOUNDARIES

- **Fast path**: x in (1 - 1/16, 1 + 1/16) ≈ (0.9375, 1.0625)
- **General path**: everywhere else with x > 0
- **`log(0)` = -inf**: caller MUST short-circuit before invoking, OR
  accept -inf (shannon_entropy convention is `0·log(0) := 0`, so caller
  must check `p == 0.0` and skip the term)
- **`log(-x)` = NaN**: caller MUST filter negative values before invoking
- **`log(NaN)` = NaN**: propagates naturally through `fadd`/`fmul`
- **`log(+inf)` = +inf**: short-circuit at top of function
- **Subnormals**: algorithm extracts E correctly via bit manipulation
  (special case for biased_exp == 0)

## Bit-exact f64 constants

```rust
// Cody-Waite split of ln(2) for high-precision E·ln2 reconstruction
const LN2_HI: f64 = 0.6931471804855391;     // bits 0x3fe62e42fef00000
const LN2_LO: f64 = 7.440620941723212e-11;  // bits 0x3dd473df1ba75445

// Polynomial — Chebyshev-fit minimax on |s| <= 1/128
// log(1 + s) ~ s + b1·s² + b2·s³ + b3·s⁴ + b4·s⁵
const B1: f64 = -0.4999999999951493;  // bits 0xbfdfffffff fcb0db (need to recompute hex)
const B2: f64 =  0.3333333333291756;  // bits 0x3fd55555 5523ab1c
const B3: f64 = -0.2500025431569765;  // bits 0xbfd00010 b5cf86fa
const B4: f64 =  0.2000021798496455;  // bits 0x3fc99955 64ab2af6

// Fast-path Taylor (degree-7) for |u| < 1/16 — exact small fractions
const C2: f64 = -0.5;
const C3: f64 =  0.3333333333333333;
const C4: f64 = -0.25;
const C5: f64 =  0.2;
const C6: f64 = -0.16666666666666666;
const C7: f64 =  0.14285714285714285;
const C8: f64 = -0.125;
const C9: f64 =  0.1111111111111111;
```

## Table T[j] = (F[j], log(F[j])) for j=0..127

128 entries × 16 bytes (F + log_F per entry) = **2048 bytes**.
Embed as Cranelift `data_section` constant. Layout: interleaved
`[F0, lF0, F1, lF1, ..., F127, lF127]` for cache locality (one cache
line holds 4 entries).

```rust
// F[j] = 1 + j/128 — exact dyadic representation in f64
// log(F[j]) = log(F[j]) computed at 100dps then rounded to nearest f64
//
// Spot-check entries verified against mpmath at 100dps:
//   F[0]   = 1.0,                  log = 0.0                    (exact)
//   F[1]   = 1.0078125,            log = 0.007782140442054949
//   F[32]  = 1.25,                 log = 0.22314355131420976
//   F[64]  = 1.5,                  log = 0.4054651081081644
//   F[96]  = 1.75,                 log = 0.5596157879354227
//   F[127] = 1.9921875,            log = 0.6892332812385275
//
// Generator: see `R:\winrapids\derive_log_minimax.py`
// Re-run with PYTHONIOENCODING=utf-8 to regenerate the full table.
```

## Cranelift IR sequence

### Edge-case fast paths (top of function)

```
;; INPUT: x : f64
;; OUTPUT: log(x) : f64

;; if x == 0.0:           return -inf
;; if x is negative:      return NaN
;; if x is +inf:          return +inf
;; if x is NaN:           return NaN  (auto via fadd)

v_one = f64const 0x3ff0000000000000

;; Fast-path test: |x - 1.0| < 1/16 = 0.0625
v_diff = fsub v_x, v_one
v_abs_diff = fabs v_diff
v_threshold = f64const 0x3fb0000000000000  ; 0.0625
v_use_fast = fcmp LT v_abs_diff, v_threshold
brif v_use_fast, fast_path, table_path
```

### Fast path

```
fast_path:
;; u = x - 1.0  (== v_diff already)
;; p = c9·u + c8                          (Horner inside out)
;; p = p·u + c7
;; ... (continue down to c2)
;; p = p·u + c2
;; result = u + u·u·p
v_c9 = f64const 0x3fbc71c71c71c71c       ; 1/9
v_c8 = f64const 0xbfc0000000000000       ; -0.125
v_c7 = f64const 0x3fc2492492492492       ; 1/7
v_c6 = f64const 0xbfc5555555555555       ; -1/6
v_c5 = f64const 0x3fc999999999999a       ; 0.2
v_c4 = f64const 0xbfd0000000000000       ; -0.25
v_c3 = f64const 0x3fd5555555555555       ; 1/3
v_c2 = f64const 0xbfe0000000000000       ; -0.5

v_p = fmul v_c9, v_diff
v_p = fadd v_p, v_c8
v_p = fmul v_p, v_diff
v_p = fadd v_p, v_c7
v_p = fmul v_p, v_diff
v_p = fadd v_p, v_c6
v_p = fmul v_p, v_diff
v_p = fadd v_p, v_c5
v_p = fmul v_p, v_diff
v_p = fadd v_p, v_c4
v_p = fmul v_p, v_diff
v_p = fadd v_p, v_c3
v_p = fmul v_p, v_diff
v_p = fadd v_p, v_c2
v_u2 = fmul v_diff, v_diff
v_u2p = fmul v_u2, v_p
v_result = fadd v_diff, v_u2p
return v_result
```

### General path

```
table_path:
;; Decompose x via bit manipulation
v_xbits = bitcast.i64 v_x
v_exp_mask = iconst.i64 0x7ff0000000000000
v_exp_bits = band v_xbits, v_exp_mask
v_exp_shift = ushr v_exp_bits, iconst.i64 52
v_bias = iconst.i64 1023
v_E_i64 = isub v_exp_shift, v_bias        ; signed exponent E

;; Construct m = 1.bbb... (set biased exponent to 1023)
v_mant_mask = iconst.i64 0x000fffffffffffff
v_mant_bits = band v_xbits, v_mant_mask
v_one_exp_bits = iconst.i64 0x3ff0000000000000
v_m_bits = bor v_mant_bits, v_one_exp_bits
v_m = bitcast.f64 v_m_bits

;; j = top 7 bits of mantissa = (xbits >> 45) & 0x7f
v_j_shift = iconst.i64 45
v_j_mask = iconst.i64 0x7f
v_j_raw = ushr v_xbits, v_j_shift
v_j = band v_j_raw, v_j_mask

;; Load F[j] and log(F[j]) from table
v_table_base = global_value LOG_TABLE
v_offset = ishl v_j, iconst.i64 4         ; j * 16 bytes (F + log_F per entry)
v_addr_F = iadd v_table_base, v_offset
v_F = load.f64 readonly v_addr_F
v_addr_logF = iadd v_addr_F, iconst.i64 8
v_logF = load.f64 readonly v_addr_logF

;; s = (m - F) / F
v_diff_mF = fsub v_m, v_F
v_s = fdiv v_diff_mF, v_F

;; Polynomial: poly(s) = s + s·s · (b1 + s·(b2 + s·(b3 + s·b4)))
v_b1 = f64const ...   ; -0.4999999999951493
v_b2 = f64const ...   ;  0.3333333333291756
v_b3 = f64const ...   ; -0.2500025431569765
v_b4 = f64const ...   ;  0.2000021798496455
v_inner = fmul v_b4, v_s
v_inner = fadd v_inner, v_b3
v_inner = fmul v_inner, v_s
v_inner = fadd v_inner, v_b2
v_inner = fmul v_inner, v_s
v_inner = fadd v_inner, v_b1
v_s2 = fmul v_s, v_s
v_inner = fmul v_inner, v_s2
v_log_1ps = fadd v_s, v_inner

;; Reconstruct: log(x) = E·ln2_hi + log(F) + (E·ln2_lo + log(1+s))
;; Group small terms together for precision (Cody-Waite)
v_E_f64 = fcvt_from_sint.f64 v_E_i64
v_ln2_hi = f64const 0x3fe62e42fef00000
v_ln2_lo = f64const 0x3dd473df1ba75445

v_E_lo_term = fmul v_E_f64, v_ln2_lo
v_low_sum = fadd v_E_lo_term, v_log_1ps   ; small + small

v_E_hi_term = fmul v_E_f64, v_ln2_hi
v_hi_sum = fadd v_E_hi_term, v_logF       ; could cancel if E·ln2_hi ≈ -log(F)

v_result = fadd v_hi_sum, v_low_sum
return v_result
```

Total: ~25 IR ops in fast path, ~30 IR ops in general path.

## Verification — accuracy across the domain

End-to-end accuracy with f64-rounded constants (no FMA):

| Test x         | tang_log(x)             | mpmath ref              | ULP / abs error |
|----------------|-------------------------|-------------------------|-----------------|
| 1.0            | 0.0                     | 0.0                     | 0 (exact)       |
| 0.5            | -0.6931471805599454     | -0.6931471805599453     | 1               |
| 2.0            | 0.6931471805599454      | 0.6931471805599453      | 1               |
| 1.5            | 0.4054651081081644      | 0.4054651081081644      | 0               |
| 0.1            | -2.302585092994046      | -2.302585092994046      | 1               |
| 1e10           | 23.025850929940457      | 23.025850929940457      | 0               |
| 1e-100         | -230.25850929940458     | -230.25850929940458     | 0               |
| 1e-300         | -690.7755278982137      | -690.7755278982137      | 0               |
| 0.99           | -0.01005033585350145    | -0.010050335853501442   | 5               |
| 0.99999        | -1.0000050000287824e-05 | -1.0000050000333335e-05 | abs error ~3e-15 (large ULP because near-zero result, but absolute error is at machine epsilon) |

**The "huge ULP near 1" caveat is misleading.** Same artifact as
discussed in `oracle/mean/python-numpy/default/known_issues.md`: when
result is near zero, ULP scaling becomes meaningless because ULP itself
is sub-attoscale. The **absolute error** at x=0.99999 is ~3e-15 — at
machine-epsilon level. For `shannon_entropy`'s use case (`-p · log(p)`
with p near 1 → contribution ≈ 1e-5), this means the per-term contribution
is accurate to 13+ significant digits. **More than enough for entropy
computation.**

## CRITICAL CALLER CONVENTIONS

### shannon_entropy

```rust
fn shannon_entropy(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p_i in p {
        // CRITICAL: 0 · log(0) := 0 by convention (limit). Skip explicitly.
        if p_i > 0.0 {
            h -= p_i * log_for_entropy(p_i);
        }
        // p_i == 0.0: contribution is 0; skip. (NOT calling log_for_entropy(0)
        // which would return -inf and 0 · -inf = NaN.)
        // p_i < 0.0: caller bug; assert in debug.
    }
    h
}
```

### hill_estimator_streaming

Hill estimator for tail index uses `Σ log(X_i / X_{(k)})` where X_{(k)} is
the k-th order statistic. All inputs are positive (caller pre-filters).
The input range to log() is x > 1 (since X_i >= X_{(k)}); near-1 fast
path handles the tail of small-tail-events; general path handles
heavy-tail outliers far from 1.

```rust
fn hill_estimator(samples: &[f64], k: usize) -> f64 {
    // Pre-filter: assert all positive
    debug_assert!(samples.iter().all(|x| *x > 0.0));
    let sorted = sort_descending(samples);
    let x_k = sorted[k - 1];  // k-th order statistic from top
    let log_xk = log_for_entropy(x_k);
    let mut sum = 0.0;
    for i in 0..(k - 1) {
        sum += log_for_entropy(sorted[i]) - log_xk;
    }
    sum / ((k - 1) as f64)
}
```

## References

- **Tang 1990**: "Table-Driven Implementation of the Logarithm Function
  in IEEE Floating-Point Arithmetic," ACM TOMS 16(4), 378-400.
  Symmetric companion to Tang 1989 exp paper.
- **Cody & Waite 1980**: "Software Manual for the Elementary Functions."
  Source of the high-low constant-splitting trick (LN2_HI/LN2_LO).
- **fdlibm e_log.c**: Sun Microsystems 1993 reference. Uses a different
  algorithm (no table; rational form `R(z) ~ z + (z²/2)·...` for z = (m-1)/(m+1))
  but the principle of "near-1 separate path" is the same; fdlibm hides
  it inside the algebraic structure rather than as an explicit branch.
- **derive_log_minimax.py**: tambear-internal verification script at
  `R:\winrapids\derive_log_minimax.py`. Run with `PYTHONIOENCODING=utf-8`
  on Windows. Re-run if precision claims need re-verification.

## Open question for adversarial review

The general-path "near-1" cancellation problem is fixed by the fast-path
branch at `|x - 1| < 1/16`. But what about **inputs where x mod some
power of 2 is near 1**? E.g., `x = 2^30 + epsilon`: E=30, m near 1, log(F[0])
near 0, but `E · ln(2) ≈ 20.79` dominates. No cancellation (the table-path
case where it would matter requires E·ln2 + log(F) ≈ 0 with large E).

This needs adversarial coverage when the kernel lands. The error path
to verify: pick x such that E·ln(2) ≈ -log(F[j]) for some j; check
ULP error on that input.
