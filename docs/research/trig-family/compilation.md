# Trig Family — Hardware Mapping and Atom Decomposition

> TRIG-9 + TRIG-11 combined deliverable. Owner: math researcher.
> Per-function: which atoms are consumed, whether DD is required,
> what the compilation differs per precision strategy (`strict` vs.
> `compensated` vs. `correctly_rounded`), and where shared-intermediate
> hooks land.

Related: `catalog.md` (function enumeration), `angle_units.md` (per-unit
reduction), `shared_pass.md` (IntermediateTag recommendations).

---

## The atom palette

From `crates/tambear/src/primitives/`:

**Hardware terminal ops** (each maps to a single hardware instruction):

| Category | Ops |
|---|---|
| Arithmetic | `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt` |
| Fused | `fmadd`, `fmsub`, `fnmadd`, `fnmsub` |
| Unary | `fabs`, `fneg`, `fcopysign` |
| Min/max | `fmin`, `fmax` (NaN-propagating) |
| Compare | `fcmp_eq`, `fcmp_lt`, `fcmp_le`, `fcmp_gt`, `fcmp_ge` |
| Classify | `is_nan`, `is_inf`, `is_finite`, `signbit` |
| Rounding | `frint`, `ffloor`, `fceil`, `ftrunc` |
| Scale | `ldexp`, `frexp` |

**Compensated arithmetic** (error-free transformations):

- `two_sum`, `fast_two_sum`, `two_diff`, `two_product_fma`, `two_square`
- `fma_residual`
- `kahan_sum`, `neumaier_sum`, `pairwise_sum`, `two_sum_accumulation`
- `compensated_horner`, `horner`, `dot_2`

**Double-double (106-bit working precision)**:

- `DoubleDouble` type from `primitives::double_double`
- Operators (add, sub, mul, div, sqrt) from the same module

---

## Atom inventory per function

This table is the TRIG-9 deliverable: per-function, which atoms each
consumes in each precision strategy. Rows with a DD column = "needed"
drive compilation differences.

### Forward trig (radians)

| Function | strict atoms | compensated adds | correctly_rounded adds | DD required? |
|---|---|---|---|---|
| `sin` | fmadd, fabs, fcmp_lt, frint, fcopysign, ffloor | `two_product_fma`, `fast_two_sum` | `DoubleDouble` type + DD ops | CR only |
| `cos` | fmadd, fabs, fcmp_lt, frint, fcopysign | `compensated_horner` | DD poly eval | CR only |
| `tan` | fmadd, fdiv, fabs, frint, fcopysign | `compensated_horner`, `two_product_fma` | DD reciprocal | compensated + CR |
| `cot` | fdiv (inverts tan core) | (inherits from tan) | DD reciprocal | compensated + CR |
| `sec` | fdiv (inverts cos core) | (inherits) | DD reciprocal | compensated + CR |
| `csc` | fdiv (inverts sin core) | (inherits) | DD reciprocal | compensated + CR |
| `sincos` | ⟨sin + cos combined⟩ | ⟨merged⟩ | ⟨merged⟩ | CR only |

### Forward trig (pi-scaled / turns / degrees / gradians)

| Function | strict atoms | compensated adds | correctly_rounded adds | DD required? |
|---|---|---|---|---|
| `sinpi` | fmadd, fabs, frint, ffloor, fcopysign | no reduction error ⇒ no compensation needed | degree-13 poly in DD | CR only |
| `cospi` | same | same | same | CR only |
| `tanpi` | fmadd, fdiv, fabs | fdiv precision via `two_product_fma` | DD div | compensated + CR |
| `sin_turn` / `cos_turn` / `tan_turn` | same palette as pi-scaled | — | — | same |
| `sin_deg` / `cos_deg` | fmadd, frint, fcmp | one DD mul for π/180 | same | — |
| `sin_grad` | fmadd, frint, fcmp | one DD mul for π/200 | — | — |

**Key insight**: non-radian forward trig has no range-reduction error
(the mod is exact), so compensated precision and strict precision are
**the same thing** for these units. Only radians requires the three-tier
ladder.

### Inverse trig

| Function | strict atoms | compensated adds | correctly_rounded adds | DD required? |
|---|---|---|---|---|
| `atan` | fmadd, fabs, fdiv, fcmp | `compensated_horner` | DD poly | compensated + CR |
| `atan2` | fmadd, fabs, fdiv, fcmp, fcopysign, is_inf | `two_product_fma` for y/x | DD div + DD atan_core | compensated + CR |
| `asin` | fmadd, fabs, fsqrt, fsub, fcmp | `two_sum` for (1-x²), `two_square` | DD polynomial on half-angle split | compensated + CR |
| `acos` | (uses asin_core) | (inherits) | (inherits) | compensated + CR |
| `acot` | fdiv, fcmp (inverts atan_core) | (inherits from atan) | DD reciprocal | compensated + CR |
| `asec` | (uses acos_core on 1/x) | (inherits) | (inherits) | compensated + CR |
| `acsc` | (uses asin_core on 1/x) | (inherits) | (inherits) | compensated + CR |
| `asinpi`/`acospi`/`atanpi` | atom set + one DD mul for 1/π | — | — | same as base |
| `atan2pi` | atom set + one DD mul for 1/π | — | — | same as base |

### Hyperbolic forward

| Function | strict atoms | compensated adds | correctly_rounded adds | DD required? |
|---|---|---|---|---|
| `sinh` | fmadd, fabs, fdiv, fcmp, ldexp | `compensated_horner` for expm1 | DD expm1 | compensated + CR |
| `cosh` | same | `compensated_horner` | DD expm1 | compensated + CR |
| `tanh` | fmadd, fdiv, fabs, fcmp, ldexp | compensated expm1 | DD expm1 | compensated + CR |
| `coth` | fdiv on tanh result | (inherits) | DD reciprocal | compensated + CR |
| `sech` | fdiv on cosh | (inherits) | DD reciprocal | compensated + CR |
| `csch` | fdiv on sinh | (inherits) | DD reciprocal | compensated + CR |
| `sinhcosh` | ⟨sinh + cosh combined⟩ | ⟨merged⟩ | ⟨merged⟩ | compensated + CR |

### Hyperbolic inverse

| Function | strict atoms | compensated adds | correctly_rounded adds | DD required? |
|---|---|---|---|---|
| `asinh` | fmadd, fabs, fsqrt, fadd, fdiv, frexp, ldexp | log1p uses `compensated_horner` | DD log1p | compensated + CR |
| `acosh` | fmadd, fabs, fsqrt, fsub, frexp | `two_sum` for (x-1), compensated log1p | DD log1p | compensated + CR |
| `atanh` | fmadd, fabs, fsub, fdiv, fcmp | `two_sum` for (1±x), compensated log1p | DD log1p | compensated + CR |
| `acoth`/`asech`/`acsch` | (inverse-arg dispatch) | (inherits) | (inherits) | compensated + CR |

### Special / non-standard

| Function | strict atoms | compensated adds | correctly_rounded adds | DD required? |
|---|---|---|---|---|
| `haversin(x)` | fmadd, fmul | `two_square` for sin²(x/2) | DD sin²(x/2) | CR only |
| `versin(x)` | (haversin · 2) | (inherits) | (inherits) | CR only |
| `coversin(x)` | sin-core with sign flip | (inherits from sin) | (inherits) | CR only |
| `exsec(x)` | fdiv, fsub (sec − 1) | `two_diff` for the subtract | DD sec | compensated + CR |

---

## DD-required matrix

The "DD required?" column tells the compiler which binary variants to
ship. **CR-only** functions (23 entries) compile to pure-hardware-atom
code in `strict` and `compensated` strategies; DD code only appears in
`correctly_rounded` variants.

**Compensated + CR** functions (19 entries) already need compensated
arithmetic for `compensated` mode; DD further refines in CR.

**Total binary footprint**:

- 32 functions × 5 units × 3 strategies = 480 compilation paths in the
  fully-expanded matrix.
- After deduplication (non-radian units skip `compensated` because
  reduction is exact; inverse reciprocals inherit from their base
  function; hyperbolic reciprocals share expm1-pair code), the real
  number of unique code paths is ~120–150.

---

## Compilation differences per precision strategy

### `strict` lowering

Single-FMA Horner. Hardware atoms only. No compensated arithmetic, no
DD. Cody-Waite reduction with 2-part residual. Worst-case error:

- Forward trig radians: ≤ 4 ULPs (fdlibm-typical).
- Forward trig non-radian: ≤ 1 ULP (no reduction error).
- Inverse trig: ≤ 3 ULPs typical, up to 5 near `|x| = 1` without
  half-angle.
- Hyperbolic forward: ≤ 4 ULPs.
- Hyperbolic inverse: ≤ 3 ULPs.

Atoms consumed: 3–5 per function. FMA-heavy. **Target**: hot-loop
numeric code where < 5 ULPs is acceptable — physics, graphics, heuristics.

### `compensated` lowering

`compensated_horner` for polynomial evaluation — ~3× the flop count of
strict but removes polynomial accumulation error. Cody-Waite reduction
gets `two_product_fma` for the reconstruction step. Worst-case error:

- Forward trig radians: ≤ 2 ULPs.
- Inverse trig: ≤ 2 ULPs.
- Hyperbolic forward: ≤ 2 ULPs.
- Hyperbolic inverse: ≤ 2 ULPs.

Atoms consumed: 10–15 per function. **Target**: quant pipelines where
result enters downstream inference and tail-error compounds.

### `correctly_rounded` lowering

DD working precision throughout. Higher-degree polynomials (typically
degree + 4). Ziv's technique for adaptive precision — evaluate at
working precision, if the result is within `ε` of a half-ULP boundary,
re-evaluate at higher precision. Worst-case error:

- All functions: ≤ 1 ULP (correctly-rounded).

Atoms consumed: 30–50 per function. DD arithmetic has ~5–10× the flop
count of hardware f64. **Target**: publication-grade results,
reference-quality runs, CR-grade scientific workflows.

---

## Per-function compilation skeleton (pathmaker cookbook)

### Template for forward trig (radians)

```
fn sin(x: f64, strategy: Strategy) -> f64 {
    // 1. Classify input
    if is_nan(x) || is_inf(x) { return NaN; }
    if fabs(x) < 2^-27 { return x; }   // tiny

    // 2. Reduce to quadrant
    let (q, r_hi, r_lo) = match strategy {
        Strict       => reduce_pio2_cody_waite(x),          // 2-part r
        Compensated  => reduce_pio2_cw_compensated(x),      // 2-part r, DD residual
        CorrectlyRounded => reduce_pio2_payne_hanek_dd(x),  // DD throughout
    };

    // 3. Evaluate core
    let result = match strategy {
        Strict       => sin_core_fma(r_hi, r_lo),           // fmadd Horner
        Compensated  => sin_core_compensated(r_hi, r_lo),   // compensated_horner
        CorrectlyRounded => sin_core_dd(r_hi, r_lo),        // DD polynomial
    };

    // 4. Quadrant fixup
    apply_quadrant_fixup(q, result)
}
```

### Template for hyperbolic

```
fn sinh(x: f64, strategy: Strategy) -> f64 {
    if is_nan(x) || is_inf(x) { return x; }
    if fabs(x) < 2^-28 { return x; }

    // Look up shared HyperbolicExpm1Pair in TamSession
    let (p, m) = hyperbolic_expm1_pair(x, strategy);

    // Derive: sinh(x) = (p - m) / 2
    match strategy {
        Strict       => fmul(fsub(p, m), 0.5),
        Compensated  => two_diff(p, m).scale(0.5),
        CorrectlyRounded => (DD::from(p) - DD::from(m)).scale(0.5).to_f64(),
    }
}
```

### Template for inverse trig

```
fn asin(x: f64, strategy: Strategy) -> f64 {
    let ax = fabs(x);
    if ax > 1.0 { set_fe_invalid(); return NaN; }

    if ax < 0.975 {
        // Direct branch — look up AsinCoreDirect
        let core = asin_core_direct(ax, strategy);
        fcopysign(core, x)
    } else {
        // Near-unity half-angle — look up AsinCoreHalfAngle
        let t = fsub(1.0, ax) * 0.5;
        let s = fsqrt(t);
        let core = asin_core_half(s, strategy);
        let result = fsub(PI_OVER_2, fmul(2.0, core));
        fcopysign(result, x)
    }
}
```

---

## Atom count summary

For a single-call count, assuming the shared intermediate is NOT in cache:

| Function | strict atoms | compensated atoms | CR atoms |
|---|---|---|---|
| `sin` | ~25 | ~65 | ~180 |
| `cos` | ~25 | ~65 | ~180 |
| `tan` | ~30 | ~75 | ~210 |
| `sincos` | ~30 | ~80 | ~220 |
| `sinpi` | ~18 | — | ~150 |
| `atan` | ~22 | ~50 | ~140 |
| `atan2` | ~28 | ~65 | ~170 |
| `asin` | ~25 (direct) to ~35 (half) | ~60 to ~85 | ~150 to ~200 |
| `sinh` | ~28 | ~70 | ~200 |
| `tanh` | ~30 | ~75 | ~210 |
| `asinh` | ~35 | ~85 | ~220 |
| `haversin` | ~8 (uses sin²) | ~20 | ~60 |

After shared-intermediate caching, the "extra call" cost drops to ~15%
of the listed number, per the TRIG-4 analysis.

---

## Gaps to file (TRIG-10)

Atoms that don't currently exist in tambear but are needed per this
decomposition:

1. **`reduce_pio2_payne_hanek_dd`** — DD-precision Payne-Hanek with the
   1200-bit `2/π` table. Currently sin.rs has it inline; extract to a
   standalone primitive for sharing.
2. **`compensated_horner_dd`** — DD-typed compensated Horner. The
   existing `compensated_horner` is f64-only.
3. **`dd_div`** — double-double division. Exists in `double_double/ops.rs`
   but not yet exposed in the top-level primitive interface.
4. **`expm1_pair`** — fused even/odd split for `(e^x − 1, e^-x − 1)`.
   New primitive specifically for the TRIG-4 shared pair.
5. **`reduce_mod_exact_integer`** — for degrees/gradians. New primitive
   per the TRIG-3 angle-unit design.
6. **`frac_exact`** — fractional-part primitive for turns / pi-scaled.
   `x − round(x, to_even)`. Must not use `ftrunc` (which has sign issues
   at negative inputs).
7. **`complementary_arg_transform`** — named meta-primitive for the
   near-unity cancellation rhyme (asin/acos/acosh/asech/atanh). The
   transform produces the half-angle argument ready for
   `AsinCoreHalfAngle` etc.

These go into TRIG-10's atom-gap ledger.

---

## Bottom line for pathmaker

- **Extract** `reduce_pio2` from `sin.rs` into a standalone primitive
  with 3 variants (Cody-Waite strict, Cody-Waite compensated,
  Payne-Hanek DD).
- **Add** `expm1_pair` as THE top-priority new primitive.
- **Ship** per-unit forward-trig recipes that call the shared core
  primitive — skip Payne-Hanek for non-radian units.
- **Ship** inverse-trig recipes that share `AtanCore` and the two
  `AsinCore` variants.
- **Ship** hyperbolics that consume `HyperbolicExpm1Pair` and derive
  the reciprocals locally.

Compilation differences: 3-tier ladder for radians (strict / compensated /
CR), 2-tier for non-radian forward trig (strict = compensated, CR
separately), 3-tier for inverse trig and hyperbolics.

This unlocks TRIG-11 — the compilation-differences-per-strategy doc is
effectively the per-function column of this matrix, formatted for the
recipe spec generator.
