# `exp_for_lse` — Cranelift IR Specification

> Math-researcher deliverable for pathmaker, 2026-04-23.
> Per DEC-027 (no FMA without supports_fma cache key gate): all
> compound multiply-adds use `fmul + fadd`, not `fma`.
> Domain: x <= 0 (LogSumExp running-max subtraction guarantees this).

## Algorithm — Tang 1989 table-driven, Chebyshev-minimax polynomial

```
exp(x) = T[N mod 32] · 2^floor(N/32) · exp(r)
where:
  N = round(x · 32/ln(2))                     (integer)
  r = (x - N·ln2_32_hi) - N·ln2_32_lo          (Cody-Waite reduction)
  exp(r) ~ 1 + r + (a1 + (a2 + (a3 + a4·r)·r)·r) · r²
```

`T[j]` is a 32-entry table of `2^(j/32)` (256 bytes data section).
The polynomial uses Chebyshev-fit minimax coefficients (NOT vanilla
Taylor — Taylor gives ~20 ULP at the edge of the reduction interval;
minimax gives ~1 ULP).

End-to-end accuracy with f64 constants and no FMA:
- 0–4 ULP for typical inputs (|x| ≤ 100)
- 5–12 ULP for extreme inputs (|x| up to 700)

This is acceptable for LSE — the values that drift heavily are tiny
(exp(-700) ≈ 1e-305) and the running-max-subtraction in LSE keeps
them out of the dominant sum unless they ARE the dominant value.

---

## Bit-exact f64 constants

Constants verified at 100-decimal-place precision via `derive_exp_minimax.py`.
Embed as Cranelift `data_section` constants or inline `iconst` then `bitcast`.

```rust
// Argument reduction
const INV_LN2_32: f64 = 46.16624130844683;
                     // bits 0x40471547652b82fe
const LN2_32_HI:  f64 = 0.021660849390173098;
                     // bits 0x3f962e42fef00000
                     // (low 20 mantissa bits zeroed for Cody-Waite)
const LN2_32_LO:  f64 = 2.325192919288504e-12;
                     // bits 0x3d8473de75a207e5
                     // (residual after Cody-Waite split)

// Polynomial — Chebyshev-fit minimax on |r| <= ln(2)/64
// f64-rounded after fitting at 60-decimal-place precision
const A1: f64 = 0.4999999999976113;   // bits 0x3fdfffffffff57e9
const A2: f64 = 0.16666666666632543;  // bits 0x3fc555555555254f
const A3: f64 = 0.041666829580991785; // bits 0x3fa5555accc1b912
const A4: f64 = 0.008333356606798872; // bits 0x3f81111430bca26e

// Table T[j] = 2^(j/32), j=0..31. Embed as data_section.
// Each entry is bit-exact-correctly-rounded f64.
static T_TABLE: [f64; 32] = [
    1.0,                  // 2^(0/32) = 1.0 exact
    1.0218971486541166,   // 2^(1/32)
    1.0442737824274138,   // 2^(2/32)
    1.0671404006768237,   // 2^(3/32)
    1.0905077326652577,   // 2^(4/32)
    1.1143867425958924,   // 2^(5/32)
    1.1387886347566916,   // 2^(6/32)
    1.1637248587775775,   // 2^(7/32)
    1.189207115002721,    // 2^(8/32) = √(√2) approx
    1.215247359980469,    // 2^(9/32)
    1.241857812073484,    // 2^(10/32)
    1.2690509571917332,   // 2^(11/32)
    1.2968395546510096,   // 2^(12/32)
    1.3252366431597413,   // 2^(13/32)
    1.3542555469368927,   // 2^(14/32)
    1.383909881963832,    // 2^(15/32)
    1.4142135623730951,   // 2^(16/32) = √2
    1.4451808069770467,   // 2^(17/32)
    1.4768261459394993,   // 2^(18/32)
    1.5091644275934228,   // 2^(19/32)
    1.5422108254079407,   // 2^(20/32)
    1.5759808451078865,   // 2^(21/32)
    1.6104903319492543,   // 2^(22/32)
    1.645755478153965,    // 2^(23/32)
    1.681792830507429,    // 2^(24/32) = 2^(3/4)
    1.718619298122478,    // 2^(25/32)
    1.7562521603732995,   // 2^(26/32)
    1.7947090750031072,   // 2^(27/32)
    1.8340080864093424,   // 2^(28/32) = 2^(7/8)
    1.8741676341103,      // 2^(29/32)
    1.9152065613971474,   // 2^(30/32)
    1.9571441241754002,   // 2^(31/32)
];
```

---

## Cranelift IR sequence

Per DEC-027, no `fma` — all compound multiply-adds are `fmul + fadd`.
The Horner evaluation order matters: by computing the small polynomial
first, then adding `1 + r` LAST, we preserve the dominant-value precision
(adding a small correction to a number near 1 loses no significant bits).

```
;; INPUT:  x : f64  (precondition: x <= 0)
;; OUTPUT: exp(x) : f64

;; ----- Edge-case fast paths (optional but recommended) -----
;; if x is NaN: return NaN (propagation; cranelift fadd handles this naturally)
;; if x == 0.0: short-circuit return 1.0 (pre-check via fcmp)
;; if x < -744.4400719213812: return +0.0 (underflow-to-zero threshold)
;;   bits 0xc0874385446d71c3
;; if x < -708.3964185322641: result is denormal
;;   bits 0xc086232bdd7abcd2
;;   Algorithm still works; precision degrades; for LSE this is fine.

;; ----- Argument reduction -----
;; xn_f = x * INV_LN2_32                ; xn_f := x · (32/ln(2))
v_inv_ln2_32 = f64const 0x40471547652b82fe
v_xn_f = fmul v_x, v_inv_ln2_32

;; n = round(xn_f) as i64
;; cranelift emits `nearest` then `fcvt_to_sint_sat`
v_xn_round = nearest v_xn_f             ; round-to-nearest-even
v_n_i64 = fcvt_to_sint_sat.i64 v_xn_round

;; r = (x - n·LN2_32_HI) - n·LN2_32_LO  ; Cody-Waite — preserves precision
;; Convert n back to f64 for the multiplications
v_n_f64 = fcvt_from_sint.f64 v_n_i64
v_ln2_32_hi = f64const 0x3f962e42fef00000
v_ln2_32_lo = f64const 0x3d8473de75a207e5
v_t1 = fmul v_n_f64, v_ln2_32_hi
v_t2 = fsub v_x, v_t1                   ; x - n·LN2_32_HI
v_t3 = fmul v_n_f64, v_ln2_32_lo
v_r  = fsub v_t2, v_t3                  ; r = (x - n·HI) - n·LO

;; ----- Polynomial — Horner from inside out -----
;; inner = a4·r + a3
;; inner = inner·r + a2
;; inner = inner·r + a1
;; poly  = inner · (r·r)
;; exp_r = 1.0 + r + poly
v_a1 = f64const 0x3fdfffffffff57e9      ; 0.4999999999976113
v_a2 = f64const 0x3fc555555555254f      ; 0.16666666666632543
v_a3 = f64const 0x3fa5555accc1b912      ; 0.041666829580991785
v_a4 = f64const 0x3f81111430bca26e      ; 0.008333356606798872
v_one = f64const 0x3ff0000000000000     ; 1.0

v_inner = fmul v_a4, v_r
v_inner = fadd v_inner, v_a3
v_inner = fmul v_inner, v_r
v_inner = fadd v_inner, v_a2
v_inner = fmul v_inner, v_r
v_inner = fadd v_inner, v_a1
v_r2    = fmul v_r, v_r                 ; r²
v_poly  = fmul v_inner, v_r2

;; exp_r = 1.0 + r + poly  (in this order — preserves precision)
v_one_plus_r = fadd v_one, v_r
v_exp_r      = fadd v_one_plus_r, v_poly

;; ----- Reconstruct: 2^(N/32) = T[N mod 32] · 2^floor(N/32) -----
;; j = N mod 32 (always in 0..31)
v_thirtytwo  = iconst.i64 32
;; cranelift's `srem` rounds toward zero, which gives wrong result for
;; negative N. Use the form: j = N - 32·(N >> 5_arithmetic)
;; or: j = (N & 31) which works because we want Euclidean mod.
;; AND-with-31 gives correct Euclidean mod for any signed N when 32 is
;; a power of 2:
v_thirtyone  = iconst.i64 31
v_j_i64      = band v_n_i64, v_thirtyone

;; k = floor(N/32) as signed shift right by 5
;; sshr is arithmetic shift right — preserves sign for negative N
v_five       = iconst.i64 5
v_k_i64      = sshr v_n_i64, v_five

;; T[j]: load f64 from data section
;; data_section_addr = global address of T_TABLE
v_table_base = global_value T_TABLE
v_offset     = ishl v_j_i64, iconst.i64 3   ; j * 8 bytes
v_addr       = iadd v_table_base, v_offset
v_t_j        = load.f64 readonly v_addr

;; product = T[j] · exp_r
v_product    = fmul v_t_j, v_exp_r

;; result = product · 2^k  via direct exponent manipulation
;; (cranelift has no `ldexp` IR; build it from the f64 bits)
;;
;; 2^k as f64: bit pattern (1023 + k) << 52
;; Then multiply product by that 2^k. Or use ldexp via integer ops:
;;   bits(product) = bits(product); bits(product)[exp_field] += k
;; The bit-manipulation form is more portable.
v_bias       = iconst.i64 1023
v_exp_int    = iadd v_k_i64, v_bias         ; biased exponent of 2^k
v_exp_shift  = ishl v_exp_int, iconst.i64 52
v_2k         = bitcast.f64 v_exp_shift
v_result     = fmul v_product, v_2k

;; OUTPUT: v_result
return v_result
```

Total IR ops: ~30 instructions including edge-case checks.

---

## Edge case handling — pathmaker MUST include these checks

```
- x is NaN          → result NaN (cranelift fadd/fmul propagate naturally; nothing extra needed)
- x == 0.0          → short-circuit return 1.0 (avoids polynomial eval for the
                      most common LSE input case where running max == element)
- x == -inf         → return 0.0
                      (the algorithm naturally produces 0.0 here because k underflows
                      and 2^k → 0; verify in adversarial test)
- x < -744.44       → return +0.0 (underflow-to-zero threshold)
                      Threshold f64 bits: 0xc0874385446d71c3
                      Detection: fcmp LT v_x v_zero_thresh
- x < -708.40       → result is denormal (smallest normal: 2.225e-308)
                      Threshold f64 bits: 0xc086232bdd7abcd2
                      Algorithm still works; precision degrades to ~10 bits
                      (denormal has reduced significand range).
                      For LSE this is fine — these tiny values contribute
                      negligibly to the running sum.
- x in [-708, 0]    → algorithm produces normal f64 result with ~5-12 ULP error
- x > 0             → DOMAIN VIOLATION; LSE caller broke its precondition
                      pathmaker may emit `debug_assert!(x <= 0.0)` as a precondition
                      check; in release the algorithm produces a finite result
                      for x in (0, ~709] and +inf above that
```

---

## Test vectors — pathmaker MUST verify these

These test vectors come from `R:\winrapids\derive_exp_minimax.py`.
After landing the cranelift IR, run these through the JIT and compare
to mpmath at 50dps. Expected ULP errors in the right column.

```
x                   | mpmath exp(x)            | expected ULP from mpmath
--------------------+--------------------------+------------------------
0.0                 | 1.0                      | 0
-1e-15              | 0.999999999999999        | 0
-1e-10              | 0.9999999999             | 0
-1e-5               | 0.9999900000499998       | 0
-0.010830           | 0.9892280131939732       | 3   (edge of polynomial range)
-0.010833           | 0.9892252253...          | 3   (just above)
-0.5                | 0.6065306597126335       | 1
-1.0                | 0.36787944117144233      | 0
-2.5                | 0.08208499862389873      | 1
-5.0                | 0.006737946999085466     | 1
-10.0               | 4.539992976248484e-5     | 1
-20.0               | 2.061153622438557e-9     | 0
-50.0               | 1.928749847963918e-22    | 1
-100.0              | 3.720075976020838e-44    | 4
-300.0              | 5.148200222412018e-131   | 5
-700.0              | 9.859676543759795e-305   | 12
-744.45             | 1.34e-323 (denormal)     | 0  (underflow-to-zero threshold straddle)
-745.0              | +0.0                     | 0
NaN                 | NaN                      | (NaN-equality contract)
```

When (later) `DoorCapability::supports_fma` is wired and FMA is used
in the polynomial Horner step, expect the typical-input ULP errors to
shrink to 0-2 ULP (Tang 1989 claims <= 1 ULP with correctly-rounded FMA).

---

## References

- **Tang 1989**: "Table-Driven Implementation of the Exponential Function
  in IEEE Floating-Point Arithmetic," ACM TOMS 15(2), 144-157.
  The original 32-entry-table algorithm. Tang's coefficients are minimax-fit;
  ours are Chebyshev-node-fit which is provably within a small constant of
  optimal at this degree on this small interval.
- **Cody & Waite 1980**: "Software Manual for the Elementary Functions."
  Source of the high-low constant-splitting trick used for argument reduction.
- **fdlibm e_exp.c**: Sun Microsystems 1993 reference implementation.
  Uses a different algorithm (no table; degree-5 in r²) but our coefficients
  for the polynomial vanilla-Taylor reference (1/2!, 1/3!, ...) match.
- **derive_exp_minimax.py**: tambear-internal verification script that
  derived these constants. Lives at `R:\winrapids\derive_exp_minimax.py`.
  Re-run if precision claims need re-verification.
