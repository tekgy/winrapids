# TBS Grammar — Trig Family

> TRIG-7 deliverable. Owner: math researcher. Builds on the existing
> minimal `.tbs` grammar from `notebooks/003-tbs-chain-parser.md` and
> aristotle's `notation.md` (which gives the function-name level) to
> specify the full user-facing syntax surface for every trig function —
> invocation forms, `using()` parameter names, fused/tuple output
> handling, and ambiguity resolution.

Related: `catalog.md` (the 32 functions), `variants.md` (exposed
parameters), `notation.md` (function-name-level TBS forms),
`angle_units.md` (unit parameter design), `shared_pass.md`
(IntermediateTag interactions).

---

## Scope

The base `.tbs` grammar already handles the shape of every trig
invocation. This doc specifies what the user can *write*:

1. **Function-name vocabulary** — the complete set of symbols the parser
   accepts as trig function names.
2. **`using()` keys and domains** — the parameters each function accepts,
   their enum values, their defaults.
3. **Fused/tuple outputs** — how `sincos`, `sincospi`, `sinhcosh`, and
   the derivative of every function produce multi-column results.
4. **Ambiguity resolution** — when `sin(0)` could mean several things,
   how we decide.
5. **Shorthand sugar** — any abbreviations that make common cases
   terser without bloating the grammar.

The grammar itself does not change. We're declaring what a `sin(col=0)`
chain means, not altering `notebooks/003-tbs-chain-parser.md`.

---

## Function-name vocabulary

Every entry from `catalog.md`, written as a TBS name. Names use snake_case;
no camelCase, no Unicode, ASCII only (`a–z`, `0–9`, `_`). This is the
flat top-level namespace that `TbsName::Simple` enters.

### Forward core (6)

`sin`, `cos`, `tan`, `cot`, `sec`, `csc`.

### Forward fused (2)

`sincos`, `sincospi`.

### Pi-scaled (7)

`sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`, `atanpi`, `atan2pi`.

### Inverse core (4)

`asin`, `acos`, `atan`, `atan2`.

### Inverse reciprocal (3)

`acot`, `asec`, `acsc`.

### Hyperbolic forward (6)

`sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`.

### Hyperbolic inverse (6)

`asinh`, `acosh`, `atanh`, `acoth`, `asech`, `acsch`.

### Hyperbolic fused (1)

`sinhcosh`.

### Non-standard (4)

`haversin`, `versin`, `coversin`, `exsec`. (Excsc omitted as rarely used.)

**Total**: 39 top-level TBS names. All simple, no dotted-name (`train.linear`)
forms required by the trig family.

### Disallowed name forms

The grammar accepts `TbsName::Dotted(String, String)` for module
namespacing (e.g. `train.linear`), but for the trig family we keep the
namespace flat — users don't write `trig.sin`. This avoids a split in
the catalog and matches libm convention. If we ever need namespacing
for research-stage functions, the chosen prefix is `trig_exp` (e.g.
`trig_exp.mock_sin`) rather than `trig.mock_sin` so that `.using()` on
production names can't accidentally land on research variants.

---

## `using()` keys by function class

Per `variants.md`, only five classes of function expose variant
parameters. The full using-key reference:

### Every forward trig function

```
sin(col=0).using(
    precision     = "strict" | "compensated" | "correctly_rounded",
    unit          = "radians" | "degrees" | "gradians" | "turns" | "pi",
    range_reduction = "auto" | "cody_waite" | "payne_hanek"   // radians only
)
```

Default: `precision="compensated"`, `unit="radians"`, `range_reduction="auto"`.

For non-radian units (`degrees`, `gradians`, `turns`, `pi`),
`range_reduction` is ignored — the parser does not error, the recipe
simply does not consume it.

### `tan`, `cot`, `sec`, `csc`

Same as forward trig, plus:

```
.using(pole_handling = "standard" | "dd_reciprocal" | "taylor_expansion")
```

Default: `"standard"`. `dd_reciprocal` only meaningful under
`precision = "correctly_rounded"`.

### `atan`, `acot`

```
atan(col=0).using(
    precision = ...,
    unit      = ...,   // applies to OUTPUT, not input
    table_size = 128 | 256 | 512
)
```

Default `table_size = 256`. Other arguments unchanged.

### `atan2`

```
atan2(col_y=0, col_x=1).using(
    precision = ...,
    unit      = "radians" | "degrees" | "gradians" | "turns" | "pi"
)
```

No `range_reduction` (atan2 has no angle input — the angle is the
output). No `table_size` override (atan2 reuses `atan`'s table via the
`AtanCore` shared intermediate).

### `asin`, `acos`, `asec`, `acsc`

```
asin(col=0).using(
    precision         = ...,
    unit              = ...,   // applies to output
    near_unity_method = "half_angle" | "hermite_pade" | "auto"
)
```

Default `near_unity_method = "auto"`. `hermite_pade` only meaningful
under `precision = "correctly_rounded"`.

### `acot`, `asec`, `acsc`

Same as above, plus:

```
.using(acot_convention = "matlab" | "mathematica")
```

Default `"matlab"` (continuous at 0). Ignored for asec and acsc.

### Hyperbolic forward and reciprocals

```
sinh(col=0).using(
    precision = ...
)
```

No unit, no range_reduction (argument is a real number, not an angle).

### `asinh`, `acosh`

```
asinh(col=0).using(
    precision = ...
)
```

No unit — output is a real number in ℝ.

### `atanh`, `acoth`, `asech`, `acsch`

```
atanh(col=0).using(
    precision            = ...,
    near_boundary_method = "log1p" | "direct_table"
)
```

Default `near_boundary_method = "log1p"` (works at all precisions).
`direct_table` only meaningful under `precision = "correctly_rounded"`.

### Pi-scaled forward and inverse

```
sinpi(col=0).using(
    precision = ...
)
```

No unit override — the unit is fixed to `pi` by the function name.
Attempting `sinpi(col=0).using(unit="degrees")` is an error (see
§Ambiguity resolution).

### Fused pairs (`sincos`, `sincospi`, `sinhcosh`)

Inherit the parameters of their components. `sincos` takes everything
`sin` or `cos` would take. The output is a **tuple**, handled below.

### Non-standard (haversin, versin, coversin, exsec)

```
haversin(col=0).using(
    precision = ...,
    unit      = ...
)
```

No extras — these are thin wrappers over sin/cos.

---

## Fused and tuple outputs

### `sincos(col=0)` — two-column output

A TBS chain that invokes `sincos` produces a result with two named
columns, `sin` and `cos`. Downstream chaining can address them by name:

```
sincos(col=0).select("sin")    // just sin values
sincos(col=0).select("cos")    // just cos values
sincos(col=0)                  // both columns, default name order (cos, sin) per notation.md
```

The grammar does not change — `select(...)` is already a valid chain
step. The trig-specific convention is:

- Column order for `sincos`: `(cos, sin)` — matches IEEE 754-2019
  recommendation.
- Column order for `sincospi`: `(cospi, sinpi)`.
- Column order for `sinhcosh`: `(sinh, cosh)` — alphabetical, note this
  is reverse of `sincos` but matches the name `sinh`**`cosh`**.

**Ambiguity resolution**: users can override the default naming via
`sincos(col=0).using(column_names = ["first", "second"])`.

### Gradient-augmented output

Under `.using(gradients=true)`, every trig function returns an extra
column `d_result/d_x` (first derivative):

```
sin(col=0).using(gradients=true)   // returns (sin, d_sin_dx) pair
```

This is a general TBS feature but is particularly natural for trig —
the derivative of sin is cos, etc., and can be computed from the **same**
`TrigQuadrantReduction` shared intermediate. The fused-pair recipe
`sincos` naturally subsumes this when `gradients=true` is set on `sin`.

### Multi-output not supported for single-output functions

`sin(col=0).select("cos")` is a TBS-level error (the `sin` recipe
produces a single `result` column). Users who want both should call
`sincos`. This is standard TBS behavior and requires no trig-specific
grammar.

---

## Ambiguity resolution

### `sin` vs. `sinpi` vs. `sinh`

These are three different functions. The name disambiguates. TBS does
not have a "smart sin" that decides based on argument range. Users
always name the specific function they want.

### `unit` on pi-scaled functions

Pi-scaled functions like `sinpi` have `unit="pi"` **implicit in the
name**. Writing `sinpi(col=0).using(unit="radians")` is an error:

```
Error: sinpi does not accept a unit parameter.
       Its unit is fixed to "pi" by the function name.
       Did you mean: sin(col=0)?
```

The TBS parser does not know this — it accepts any `using(...)` key.
The **recipe** performs the validation and raises a TBS-level error
message. This matches the general TBS pattern for invalid keys.

### `atan` vs. `atan2`

`atan(col=0)` takes one column; `atan2(col_y=0, col_x=1)` takes two.
Named parameters resolve ambiguity. `atan(col=0, col=1)` is a TBS
duplicate-key error.

### Positional vs. named arguments

The grammar allows both. For trig functions, the convention is:

- **Positional only for single-argument functions**: `sin(0)` is
  legal and means `sin(col=0)`.
- **Named required for multi-argument functions**: `atan2(0, 1)` is an
  error; must write `atan2(col_y=0, col_x=1)`.

This prevents `atan2(0, 1)` vs. `atan2(1, 0)` ambiguity.

Enforced by the recipe's parameter spec — the TBS parser accepts
either form and the recipe registry validates.

### Acot(0) — the semantic ambiguity from TRIG-3

`acot(col=0)` on an input column containing 0 is ambiguous by
mathematical convention:
- MATLAB: `acot(0) = π/2`.
- Mathematica: `acot(0)` undefined.

Default is MATLAB. Users override via `.using(acot_convention="mathematica")`.
No parser ambiguity; the user surface is explicit.

### Trailing comma

`sin(col=0,)` is accepted (trailing comma after last argument, per
notebook 003's grammar). `sin(,col=0)` is a parse error. These are
ergonomics, not trig-specific.

### Empty `.using()`

`sin(col=0).using()` is equivalent to `sin(col=0)`. No error, no-op.

### Repeat using keys

`sin(col=0).using(precision="strict").using(precision="correctly_rounded")`
is **last-wins**. The second overrides the first. This is standard TBS
behavior for chained `using()` steps.

---

## Shorthand sugar

The catalog is ~39 functions; the `using()` keys are ~8; the full
combinatorial surface is large. Users frequently want common
combinations. To reduce key noise without bloating the grammar:

### Dotted-name method shortcuts

TBS grammar allows `name.subname` (`TbsName::Dotted`). For trig, we
map specific dotted names to common `using()` combinations:

| Short form | Expanded form |
|---|---|
| `sin.strict(col=0)` | `sin(col=0).using(precision="strict")` |
| `sin.cr(col=0)` | `sin(col=0).using(precision="correctly_rounded")` |
| `sin.degrees(col=0)` | `sin(col=0).using(unit="degrees")` |
| `asin.hermite(col=0)` | `asin(col=0).using(near_unity_method="hermite_pade")` |

These are **optional** — both `sin.strict(col=0)` and the
full-explicit form are legal and parse to the same AST. Power users
use the explicit form; casual users read short forms more easily.

**Rule**: every shortcut maps to exactly one `using()` combination.
No shortcut combines multiple keys. Users wanting
`precision="cr", unit="degrees"` write both explicitly.

### Column-name shorthand

When the input table has a single data column, `sin()` is equivalent
to `sin(col=0)`. This is a general TBS convenience and is not
trig-specific.

### Default precision override at pipeline level

Inside a `.tbs` pipeline, users can set a pipeline-wide default:

```
using(precision="correctly_rounded").
  sin(col=0).
  cos(col=1).
  tan(col=2)
```

This sets `precision="correctly_rounded"` for all three calls. Per-call
`using()` overrides (`sin(col=0).using(precision="strict")`) win per
TRIG-2's override semantics.

This is not trig-specific but is especially useful for trig, which has
many calls that should share the same precision within a workflow.

---

## Shared-intermediate interaction (TRIG-4)

TBS itself is unchanged by TamSession caching, but users should be
aware that **the same input column calling multiple trig functions
implicitly shares intermediates**:

```
sin(col=0)     // first call — computes TrigQuadrantReduction, caches
cos(col=0)     // second — reuses cached reduction, only runs kernel
tan(col=0)     // third — reuses cached reduction
```

This is invisible to the grammar. No user-facing keyword marks the
sharing; it happens per the TamSession compatibility rules from
`shared_pass.md`. **The only grammar consequence**: if the user
wants to force a fresh computation (for benchmarking), they use
`.using(cache="bypass")` — a general TBS escape hatch, not trig-specific.

---

## Validation layer

The TBS parser accepts all of the above *syntactically*. Validation is
a separate pass that runs after parsing:

1. Verify function name is in the trig catalog or the wider primitive
   registry.
2. Verify every `using()` key is valid for that function.
3. Verify every enum value is valid for that key.
4. Verify `col` / `col_y` / `col_x` references resolve to actual input columns.

Validation errors produce TBS-level messages with source-location
carets, same infrastructure as `TbsParseError` from notebook 003.

**Catalog ownership**: this doc specifies the TBS surface for the trig
family. When pathmaker adds a new function (e.g. `sincospi` in TRIG-16),
they add it to the catalog and the validator automatically accepts it.
Math-researcher updates this doc with the new function's using-keys.

---

## Summary

The TBS surface for 39 trig functions is fully specified by:

- Flat namespace, snake_case names.
- Five classes of using() parameters, one class per function family.
- Three fused functions (`sincos`, `sincospi`, `sinhcosh`) producing
  named tuples with canonical column order.
- Positional arguments allowed for single-input functions only.
- Dotted-name shortcuts for common `using()` combinations (optional).
- Validation deferred to post-parse pass; grammar itself is unchanged.

**No grammar changes required.** The existing minimal grammar from
notebook 003 handles the entire trig family via the vocabulary +
validation layer above.

This unblocks TRIG-20 (master synthesis) — the user-facing surface is
now a single artifact alongside the catalog, the compilation rules,
and the expert defaults.
