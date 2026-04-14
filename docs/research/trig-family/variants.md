# Trig Family — Range Variants

> TRIG-2 deliverable. For every forward trig (plus relevant inverses and
> hyperbolics), enumerate the **legitimate range variants** that warrant
> separate algorithmic handling — tiny / small / medium / large — and
> decide which to expose as user-accessible `using()` entries vs.
> auto-dispatch by the generic function.

Related: `catalog.md`, `compilation.md` (per-function atom decomposition).

---

## The motivation

Transcendental functions change character across their domain. A "one
size fits all" implementation misses 10–100× speed wins on the
hot-common path (small argument) and loses precision on the cold
extreme-argument path. Every well-engineered libm splits its domain
into regimes and picks the best algorithm per regime — fdlibm does
this, CORE-MATH does this, Intel SVML does this.

Tambear's question: which regime splits are worth **exposing** through
`using(range_variant=...)` and which should be hidden behind
auto-dispatch? Exposing too much pollutes the API; hiding too much
removes the power-user override that tambear's Layer 2 `using()` design
requires.

---

## The four regime labels

We adopt a shared vocabulary so that the same regime label means the
same thing across functions. Thresholds vary per function but labels
are consistent:

| Label | Meaning |
|---|---|
| **`tiny`** | Argument so small that `f(x) ≈ x` (odd) or `f(x) ≈ 1 + O(x²)` (even). Return input or 1 after short-circuit; no polynomial. |
| **`small`** | Argument in the "core" regime of the minimax polynomial. Single polynomial, no reduction, worst-case ULP is well below 1. |
| **`medium`** | Argument requiring range reduction via Cody-Waite (for radians) or integer mod (for other units). Polynomial + fixup. |
| **`large`** | Argument requiring Payne-Hanek (radians) or expanded-precision reduction. Heavy. |
| **`extreme`** | Past the last regime a nontrivial answer is possible — overflow for `sinh`/`cosh`, zero for `tanh`, saturation for inverse, etc. Short-circuit to the limit value. |

---

## Per-function range breakdown

### `sin`, `cos`, `sincos` (radians)

| Regime | Range | Algorithm | Error |
|---|---|---|---|
| `tiny` | `\|x\| < 2⁻²⁷` | `sin(x) = x`; `cos(x) = 1` | Exact (both are `≤ 1 ULP` off the true value, and we return the input which is within 1 ULP by construction) |
| `small` | `\|x\| < π/4 ≈ 0.785` | Kernel polynomial only, no reduction | ≤ 0.5 ULP |
| `medium` | `π/4 ≤ \|x\| < 2²⁰·π/2 ≈ 1.65e+6` | Cody-Waite reduction + kernel + fixup | ≤ 1 ULP |
| `large` | `\|x\| ≥ 1.65e+6` | Payne-Hanek reduction + kernel + fixup | ≤ 1.5 ULP |
| `extreme` | `\|x\| = ∞` or NaN | Return NaN + `FE_INVALID` | n/a |

**Expose**: `tiny`, `medium`, `large` — these genuinely dispatch to
different code. The `small` regime is a special-case of `medium` that
skips the first reduction step; keep it as an internal optimization
not exposed via `using()`.

**Auto-dispatch default**: YES. Users should never need to know which
regime their input lands in. The `tan.spec.toml` already exposes
`range_reduction = "auto" | "cody_waite" | "payne_hanek"` which is the
correct surface for power users — `auto` lets us pick, the others
force a strategy.

**Power-user override use case**: a user running a benchmark with
arguments known to be `\|x\| < π/4` can set `range_reduction = "none"`
to skip the reduction check for a few extra ns per call. Probably not
worth exposing for a public API; expose inside hot-loop research tools
only.

### `tan`, `cot`, `sec`, `csc` (radians)

Same four regimes as sin/cos. One extra concern: **pole proximity**.

| Regime | Range | Algorithm | Error |
|---|---|---|---|
| `near_pole` | `\|r_hi\| ∈ [π/4 − 2⁻³⁰, π/4]` after reduction | Extra-precise kernel + DD division | ≤ 1 ULP |

**Exposure decision**: hide `near_pole` under `auto`. Only surface
through `using(pole_handling = "strict" | "dd" | "taylor_expansion")`
for power users doing rootfinding near poles.

### `sinpi`, `cospi`, `tanpi`

Only three regimes because reduction is exact.

| Regime | Range | Algorithm | Error |
|---|---|---|---|
| `tiny` | `\|x\| < 2⁻²⁶` | `sinpi(x) = π·x`, `cospi(x) = 1` | ≤ 1 ULP |
| `small` | `\|x\| < 0.25` | Kernel polynomial on `π·x` | ≤ 0.5 ULP |
| `general` | any | Reduce `x mod 2` (exact), then kernel | ≤ 0.5 ULP |

**Exposure**: auto-dispatch only. No user-facing variant parameter.
The pi-scaled family is simple enough that regime-splitting is pure
internal optimization.

### `sin_deg`, `cos_deg`, `tan_deg`, `sin_grad` etc.

Same as pi-scaled — three regimes, auto-dispatch, no exposure.

### `atan`

| Regime | Range | Algorithm | Error |
|---|---|---|---|
| `tiny` | `\|x\| < 2⁻²⁷` | `atan(x) = x` | Exact |
| `small` | `\|x\| < 11/16 ≈ 0.6875` | Direct polynomial, no table | ≤ 0.5 ULP |
| `medium` | `11/16 ≤ \|x\| ≤ 39/16` | fdlibm's table-of-arctangents: 4 table entries | ≤ 1 ULP |
| `large` | `\|x\| > 39/16` | `atan(x) = π/2 − atan(1/x)` reflection | ≤ 1 ULP |
| `saturation` | `\|x\| > 2⁶⁶` | Return `±π/2` | Exact |

**Expose**: none directly. Power users might override table size via
`using(atan_table_size = 128 | 256 | 512)` to trade memory for
accuracy. Default 256.

### `atan2(y, x)`

Multi-dimensional regime — depends on both `y` and `x`. Key splits:

| Regime | Condition | Algorithm |
|---|---|---|
| `axis_edge` | `y == 0` or `x == 0` | Return constant per IEEE 754-2019 Table 9.1 |
| `infinity_mix` | `\|y\| = ∞` or `\|x\| = ∞` | Return constant per Table 9.1 |
| `quadrant_1` | `x > 0` | `atan(y/x)` |
| `quadrant_2/3` | `x < 0` | `atan(y/x) ± π` (sign from `y`) |
| `diagonal` | `\|y\| ≈ \|x\|` | Extra precision on `y/x` to avoid ±1 cancellation |

**Expose**: no user-facing variant. All IEEE edge cases must fire
automatically — deferring to the user is wrong for a standards-defined
function.

### `asin`, `acos`

| Regime | Range | Algorithm | Error |
|---|---|---|---|
| `tiny` | `\|x\| < 2⁻²⁷` | `asin(x) = x`; `acos(x) = π/2 − x` | Exact |
| `small` | `\|x\| < 0.5` | Direct polynomial `x·R(x²)` | ≤ 0.5 ULP |
| `medium` | `0.5 ≤ \|x\| < 0.975` | Same polynomial, different breakpoint | ≤ 1 ULP |
| `near_unity` | `\|x\| ≥ 0.975` | Half-angle transform `asin(x) = π/2 − 2·asin(√((1-x)/2))` | ≤ 1 ULP (strict), ≤ 1 ULP CR with Hermite-Padé |
| `boundary` | `\|x\| = 1` exactly | Return `±π/2` or `0`, `π` | Exact |
| `domain_error` | `\|x\| > 1` | NaN + `FE_INVALID` | n/a |

**Expose**: `using(near_unity_method = "half_angle" | "hermite_pade")`
is worth exposing — it lets users test whether CR-quality matters for
their data. Default: auto (half_angle for strict/compensated,
hermite_pade for CR).

### `asec`, `acsc`, `acot`

Variants inherit from `asin`/`acos`/`atan` through the reciprocal
transform. Users pick via `using(acot_convention)` per TRIG-3 open
question.

### `sinh`, `cosh`, `tanh`

| Regime | Range | Algorithm | Error |
|---|---|---|---|
| `tiny` | `\|x\| < 2⁻²⁸` | `sinh(x) = x`; `cosh(x) = 1`; `tanh(x) = x` | Exact |
| `small` | `\|x\| < 0.3465735903` (= `ln(2)/2`) | Polynomial on `expm1(x)` | ≤ 0.5 ULP |
| `medium` | `0.3465... ≤ \|x\| < 22` | `expm1`-based formula with scale | ≤ 1 ULP |
| `large` | `22 ≤ \|x\| < 710.48` (f64 overflow threshold) | `exp(\|x\|)/2` with sign | ≤ 1 ULP |
| `saturation` | `\|x\| > 710.48` (sinh/cosh) | Return `±∞` + `FE_OVERFLOW` | n/a |
| `saturation` | `\|x\| > 19` (tanh) | Return `±1` exactly | Exact — `tanh` saturates precisely |

**Expose**: none — `tanh` saturation at 19 is already auto-detected.

### `coth`, `sech`, `csch`

Inherit regimes from `sinh`/`cosh`/`tanh` via reciprocal. `coth(0) =
±∞`, `csch(0) = ±∞` — add `singularity` regime at zero.

### `asinh`, `acosh`, `atanh`

| Regime | Range | Algorithm | Error |
|---|---|---|---|
| `tiny` | `\|x\| < 2⁻²⁸` | Short-circuit | Exact |
| `small` | `\|x\| < 0.5` (asinh), `x < 1.5` (acosh), `\|x\| < 0.5` (atanh) | `log1p`-form direct | ≤ 0.5 ULP |
| `medium` | regime before saturation | Asymptotic or scaled form | ≤ 1 ULP |
| `large` | `\|x\| > 2²⁸` (asinh), `x > 2²⁸` (acosh) | `asinh(x) ≈ log(2·\|x\|)` | ≤ 1 ULP |
| `near_boundary` (atanh) | `\|x\| > 0.5` | `(1/2)·log1p(2x/(1-x))`, higher-precision `1-x` | ≤ 1 ULP (strict), ≤ 1 ULP CR with direct table |
| `domain_error` | `acosh(x < 1)` or `atanh(\|x\| > 1)` | NaN | n/a |
| `divergence` | `atanh(\|x\| = 1)` | `±∞ + FE_DIVBYZERO` | n/a |

**Expose**: `using(near_boundary_method = "log1p" | "direct_table")` for
`atanh` / `acosh` — same pattern as `asin`'s half-angle override.

---

## Variant exposure summary

| Function | Auto-dispatch | User-exposed variant | Rationale |
|---|---|---|---|
| `sin`/`cos`/`sincos`/`tan` | yes | `range_reduction = {auto, cody_waite, payne_hanek}` | For researchers benchmarking reduction strategies |
| `cot`/`sec`/`csc` | yes | (inherits from tan) | Reciprocal family shares overrides |
| `sinpi`/`cospi`/`tanpi` | yes | none | Simple enough, no exposure |
| `*_deg` / `*_grad` / `*_turn` | yes | none | Same — simple units |
| `atan` | yes | `table_size = {128, 256, 512}` | Memory/accuracy tradeoff |
| `atan2` | yes | none | Standards-defined edge cases |
| `asin`/`acos` | yes | `near_unity_method = {half_angle, hermite_pade, auto}` | Precision-strategy tuning |
| `asec`/`acsc`/`acot` | yes | `acot_convention = {matlab, mathematica}` | Semantic, not numerical |
| `sinh`/`cosh`/`tanh` | yes | none | Simple regime, no exposure |
| `coth`/`sech`/`csch` | yes | none | Reciprocal |
| `asinh`/`acosh` | yes | none | Simple |
| `atanh` | yes | `near_boundary_method = {log1p, direct_table}` | CR-vs-strict tuning |

**Five functions** have user-exposed variant parameters. Everything else
is auto-dispatch.

---

## Preconditions + violation handling

All regime entries have preconditions (input falls in the range) and
violation behavior (input outside). Per the tambear contract "assumptions
explicit — detected at runtime," every recipe must check preconditions
and either dispatch to the correct regime or produce the defined failure
value. The summary:

| Violation | Behavior |
|---|---|
| NaN input | NaN output, no flag |
| ±∞ input (defined function) | Return per IEEE (e.g., `atan(∞) = π/2`, `sinh(∞) = ∞`) |
| ±∞ input (NaN domain) | NaN + `FE_INVALID` (e.g., `sin(∞)`) |
| Out-of-domain (e.g., `asin(2)`) | NaN + `FE_INVALID` |
| At singularity (e.g., `atanh(1)`) | `±∞` + `FE_DIVBYZERO` |
| Underflow (e.g., `sinh(tiny)`) | Return input or `0` with `FE_UNDERFLOW` if subnormal |
| Overflow (e.g., `sinh(720)`) | `±∞` + `FE_OVERFLOW` |

---

## Shared-intermediate interaction

Range variants interact with TRIG-4's shared intermediates as follows:

- The `tiny` variant short-circuits and does not consume any shared
  intermediate. No entry in TamSession.
- The `small` variant consumes the kernel but not the reduction.
  `TrigSinCosCore` or `AtanCore` compatibility; no `TrigQuadrantReduction`
  dependency.
- The `medium` and `large` variants consume both reduction and kernel.
  All shared-intermediate hooks apply.

**Implication for compatibility tags**: the TamSession
`TrigQuadrantReduction{unit}` tag should include the regime as part of
its metadata, so a `large` consumer cannot reuse a `medium` reduction
(Cody-Waite vs. Payne-Hanek have different precision bounds).

Actually, no — the **output** of both reductions is the same `(q, r_hi,
r_lo)`. Cody-Waite is less precise, but the downstream kernel doesn't
know the difference. **Compatibility rule**: compensated/CR consumers
can reuse a CR-precision reduction, but NOT a strict-precision
reduction. This is the precision-subsumption rule from TRIG-4's
compatibility analysis.

---

## Implementation hook for pathmaker

Per-function regime classification should live in a single dispatch
function:

```
fn classify_regime(x: f64, func: TrigFunction) -> Regime {
    let ax = x.abs();
    match func {
        TrigFunction::Sin => match ax {
            ax if ax < 2.0_f64.powi(-27)       => Regime::Tiny,
            ax if ax < std::f64::consts::FRAC_PI_4 => Regime::Small,
            ax if ax < 2.0_f64.powi(20) * std::f64::consts::FRAC_PI_2 => Regime::Medium,
            ax if ax < f64::INFINITY           => Regime::Large,
            _                                  => Regime::Extreme,
        },
        // ... one arm per function
    }
}
```

Each variant then has its own entry point:

```
fn sin_tiny(x: f64) -> f64 { x }           // exact short-circuit
fn sin_small(x: f64) -> f64 { sin_kernel(x, 0.0) }
fn sin_medium(x: f64) -> f64 { /* Cody-Waite */ }
fn sin_large(x: f64) -> f64 { /* Payne-Hanek */ }
```

The generic `sin(x)` dispatches. Power-user override skips classification
and calls the variant directly:

```
tambear.sin(x).using(regime="small")   // user asserts they know
```

If the assertion is wrong and the true input falls in `large`, we
return a wrong answer. This is intentional — `using()` is a power-user
trust-me override.

---

## Summary

Five classes of variant exposure:

1. **Radian forward trig**: expose `range_reduction` for benchmark control.
2. **Inverse radian trig near-unity**: expose `near_unity_method` for
   precision tuning.
3. **Inverse convention** (`acot` / `asec` / `acsc`): expose
   `acot_convention` for MATLAB vs. Mathematica semantics.
4. **`atan` table size**: expose for memory/accuracy tradeoff.
5. **`atanh` near-boundary**: expose `near_boundary_method`.

Everything else is hidden auto-dispatch. Pathmaker builds the
regime-classifier as a standalone primitive `classify_regime(x, func)`
so all 32 functions share one dispatch implementation.

Next: TRIG-2 extensions per research might include **complex-valued**
variants for each function (`sin(x + i·y)`, `atan(z)`, etc.) — deferred
until the basic real catalog is complete.
