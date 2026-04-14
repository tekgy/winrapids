# TBS Syntax For Every Trig Function

**Author**: Aristotle (tambear-trig)
**Date**: 2026-04-14
**Status**: Deliverable for TRIG-7. Builds on `notation.md` (Style 3 — TBS column) and `autodiscover-probes.md` (probe integration).

**Relation to the TBS executor**: the existing `tbs_executor.rs` dispatches `match (step.name, step.method)` where `method` is the optional dotted-qualifier (e.g., `test.t` matches `("test", Some("t"))`). The aliasing convention is that both `test.t` and `t_test` dispatch to the same branch via `| ("t_test", None)`. Every trig function below follows the same pattern: the primary TBS verb is the lowercase function name, with method-variants available via either dotted form or underscore-aliased name.

**Design discipline**: TBS is intentionally ~100-word. Do not add a new verb when an existing verb with a method parameter will do. Trig fits the pattern because it has a small root set (`sincos`, `tan`, `atan2`, `exp`, `log`) and many derivatives — the derivatives don't all deserve top-level verbs. They're surfaced either as dotted methods or as ergonomic aliases.

---

## The TBS syntactic shape (summary)

Every TBS chain step has four slots:

```
verb(positional_args, named_args).using(overrides).on_chain
```

- **verb**: the primary name (`sin`, `atan2`, `sincos`, ...). In a pipeline, it consumes the chain's frame as implicit input.
- **positional_args**: typically a single `col=N` or `col_x=N, col_y=N` index into the frame.
- **named_args**: additional parameters (`angle_unit=...`, `range_reduction=...`, `output=...`).
- **.using(...)**: attaches a UsingBag of overrides. Applies to this step and all downstream steps until overridden.
- **dotted methods**: `.discover()`, `.with(precision=...)`, `.sweep(...)` — the meta-verbs that combine with any primary verb.

The existing executor supports this via the `(name, method)` tuple; I preserve that contract for trig.

---

## Forward family

### Primary verbs: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `sincos`

| Function | Primary TBS | Aliases (all dispatch the same) | Output |
|---|---|---|---|
| sin | `sin(col=0)` | — | scalar column |
| cos | `cos(col=0)` | — | scalar column |
| tan | `tan(col=0)` | — | scalar column |
| cot | `cot(col=0)` | `cotan(col=0)`, `cotangent(col=0)` | scalar column |
| sec | `sec(col=0)` | `secant(col=0)` | scalar column |
| csc | `csc(col=0)` | `cosec(col=0)`, `cosecant(col=0)` | scalar column |
| sincos | `sincos(col=0)` | — | **two-column** output `(cos, sin)` |

### Dotted-method variants (precision)

```tbs
sin(col=0)                                        # strict (default per spec.toml)
sin(col=0).using(precision="compensated")         # compensated
sin(col=0).using(precision="correctly_rounded")   # correctly_rounded
sin(col=0).with(precision="correctly_rounded")    # alias for per-call override
```

The executor match arm: `("sin", None)` — lookup `using.get("precision")` to dispatch to the strict / compensated / correctly_rounded body.

### Angle unit parameterization

```tbs
sin(col=0)                                        # radians (default)
sin(col=0).using(angle_unit="degrees")            # input is in degrees
sin(col=0).using(angle_unit="gradians")           # input is in gradians
sin(col=0).using(angle_unit="turns")              # input is in turns (cycles)
sin(col=0).using(angle_unit="pi_scaled")          # input is in units of π (= sinpi)
```

Ergonomic aliases (these dispatch via dotted method, per existing executor convention):

```tbs
sind(col=0)     # alias for sin().using(angle_unit="degrees")
cosd(col=0)     # alias for cos().using(angle_unit="degrees")
tand(col=0)     # alias for tan().using(angle_unit="degrees")
sinpi(col=0)    # alias for sin().using(angle_unit="pi_scaled")
cospi(col=0)    # etc
```

Executor match: `("sind", None)` sets `angle_unit = "degrees"` implicitly and delegates to the `sin` code path. This keeps the implementation in one place.

### Range-reduction override (advanced)

```tbs
sin(col=0).using(range_reduction="auto")          # default: CW + PH dispatch
sin(col=0).using(range_reduction="cody_waite")    # force CW (errors for |x| ≥ 2^20·π/2)
sin(col=0).using(range_reduction="payne_hanek")   # force PH (slower but always valid)
```

---

## Inverse family

### Primary verbs: `asin`, `acos`, `atan`, `atan2`, `acot`, `asec`, `acsc`

| Function | Primary TBS | Aliases | Input / Output |
|---|---|---|---|
| asin | `asin(col=0)` | `arcsin(col=0)` | input `[-1,1]`, output `[-π/2, π/2]` |
| acos | `acos(col=0)` | `arccos(col=0)` | input `[-1,1]`, output `[0, π]` |
| atan | `atan(col=0)` | `arctan(col=0)` | input `ℝ`, output `(-π/2, π/2)` |
| atan2 | `atan2(col_y=0, col_x=1)` | `arctan2(...)` | two inputs, output `(-π, π]` |
| acot | `acot(col=0)` | `arccot(col=0)` | output `(0, π)` |
| asec | `asec(col=0)` | `arcsec(col=0)` | input `|x|≥1`, output `[0, π] \ {π/2}` |
| acsc | `acsc(col=0)` | `arccsc(col=0)` | input `|x|≥1`, output `[-π/2, π/2] \ {0}` |

### Output angle unit — new parameter (complement of forward's input angle unit)

The inverse family's output is an angle; users may want it in degrees/turns/etc. without a second conversion step.

```tbs
atan2(col_y=0, col_x=1)                             # radians (default)
atan2(col_y=0, col_x=1).using(angle_unit="degrees") # output in degrees
atan2(col_y=0, col_x=1).using(angle_unit="turns")   # output in [-0.5, 0.5] turns

# Ergonomic aliases:
atan2d(col_y=0, col_x=1)    # degrees output
asinpi(col=0)               # output divided by π  (i.e. output is in half-turns)
acospi(col=0)
atanpi(col=0)
atan2pi(col_y=0, col_x=1)
```

Note: the **input-unit** `angle_unit` parameter applies to forward functions; the **output-unit** parameter applies to inverse functions. Same key name, same semantics — "what unit is the angle represented in" — disambiguated by whether the function accepts or produces an angle. The executor resolves which by looking at the verb.

---

## Hyperbolic family

### Primary verbs: `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`, plus inverses `asinh`, `acosh`, `atanh`, `acoth`, `asech`, `acsch`

```tbs
sinh(col=0)                                    # naive or stabilized, per recipe
cosh(col=0)
tanh(col=0)

asinh(col=0)
acosh(col=0)                                   # input >= 1
atanh(col=0)                                   # |input| < 1

sinhcosh(col=0)                                # fused pair — TWO columns output
```

Precision/override parameters flow identically to the circular family.

**No angle_unit parameter** for hyperbolics — they operate on real numbers, not angles. Only precision and (for the stabilized variants) regime boundaries.

```tbs
tanh(col=0).using(stabilization="pade")       # force Padé-approx for small |x|
tanh(col=0).using(stabilization="exp_based")  # force (e^{2x}-1)/(e^{2x}+1) form
```

---

## π-scaled family

Per `first-principles.md` Phase 8 and `hardware-mapping.md` K2, the π-scaled family is a view over the pi-reduced kernel. TBS surfaces them as both aliases and as `angle_unit="pi_scaled"`:

```tbs
# As distinct verbs (ergonomic):
sinpi(col=0)
cospi(col=0)
tanpi(col=0)
sincospi(col=0)
asinpi(col=0)
acospi(col=0)
atanpi(col=0)
atan2pi(col_y=0, col_x=1)

# Equivalent via angle_unit:
sin(col=0).using(angle_unit="pi_scaled")      # identical to sinpi
sincos(col=0).using(angle_unit="pi_scaled")   # identical to sincospi
```

Both dispatch to the same kernel code path. The executor's match table gives `sinpi` its own arm that sets `angle_unit = "pi_scaled"` and delegates to the `sin` body.

---

## Auxiliary / non-standard verbs

From the existing `*.spec.toml` list I saw these already have specs from pathmaker: `versin`, `haversin`, `gudermannian`, `inv_gudermannian`. Surface them too:

```tbs
versin(col=0)          # 1 - cos(x)  — historical navigation trig
haversin(col=0)        # (1 - cos(x)) / 2  — haversine for great-circle distance
gudermannian(col=0)    # gd(x) = atan(sinh(x))  — maps hyperbolic to circular
inv_gudermannian(col=0) # gd⁻¹(x) = asinh(tan(x))

# Convenience for haversine-distance (a common geo use case):
haversine_distance(col_lat1=0, col_lon1=1, col_lat2=2, col_lon2=3)
    .using(angle_unit="degrees")  # typical for geo data
    .using(radius_km=6371.0)      # mean Earth radius
```

The `haversine_distance` form composes `haversin + multiply + atan2` under the hood; surfacing it as a single verb matches the one-line-one-intent principle.

---

## Fused + shared-pass verbs

From TRIG-4 (researcher shipped) — the trig "MomentStats" lives in verbs that make the shared reduction explicit:

```tbs
# Both sin and cos from one pass — explicit fused verb:
sincos(col=0)                            # two-column output (cos, sin)

# Forward pair with π-scaling:
sincospi(col=0)                          # two-column output

# Three-output fused: sin, cos, tan all in one pass (sharing reduction):
trig_triple(col=0)                       # three-column output

# Shared reduction kept around for downstream reuse:
trig_reduce(col=0)                       # three-column output (q, r_hi, r_lo)
    .then(sincos_kernel)                 # consume shared reduction
    .then(tan_kernel)                    # another consumer, same reduction
```

The `trig_reduce` verb makes first-principles.md's Aristotelian move directly user-visible: reduction is a first-class step the user can name, measure, and reuse. Most users won't need this; the ones who care about sharing will.

---

## `.discover()` integration

Per `autodiscover-probes.md`:

```tbs
sin(col=0).discover()                    # Tier 1 probes: input+identity+regime
sin(col=0).discover(level="deep")        # Tier 1 + Tier 2 + Tier 3 (includes superposition)
sin(col=0).discover(include=["precision"])  # just precision superposition (P5.3)

# Scientific-insight probes:
angle_analyze(col=0)                     # runs P1.2 (unit_detect) + P4.x (insights)
trig_identity_check(col=0)               # runs P3.x only — for CI
```

The `.discover()` on a trig step returns a `DiscoveryResult` structurally identical to the clustering one (view list + agreement + collapse) but populated with trig-specific probes. Output in TBS:

```
Column col=0: 10,000 rows.
Magnitude range [-6.28, 6.28], 99% in Cody-Waite regime.
Non-monotonic (stationary, not phase accumulator).
Pythagorean identity holds to 1.2 ulp worst-case.
Chose: sin at strict precision, Cody-Waite reduction.

View agreement: 0.998.
```

---

## `.sweep()` — sweep over parameter space (from the superposition memory)

```tbs
sin(col=0).sweep(precision=["strict","compensated","correctly_rounded"])
# Returns three columns, one per precision variant, plus diff columns.

sin(col=0).sweep(range_reduction=["cody_waite","payne_hanek"])
# Returns two columns; valuable near the 2^20·π/2 boundary.

sin(col=0).sweep(angle_unit=["radians","degrees"])
# Usually a bug — should be constant across unit if input is correctly converted.
# Returns two columns; differences flag unit-confusion.
```

---

## Complete TBS-surface table (Style 3 catalog)

This is the superset of every TBS call the trig family supports. Each row is what users can type and what the executor must dispatch.

| Verb | Executor match | Meaning | Output shape |
|---|---|---|---|
| `sin(col=N)` | `("sin", None)` | Forward sine | 1 col |
| `cos(col=N)` | `("cos", None)` | Forward cosine | 1 col |
| `tan(col=N)` | `("tan", None)` | Forward tangent | 1 col |
| `cot(col=N)` | `("cot", None) \| ("cotan", None) \| ("cotangent", None)` | 1/tan | 1 col |
| `sec(col=N)` | `("sec", None) \| ("secant", None)` | 1/cos | 1 col |
| `csc(col=N)` | `("csc", None) \| ("cosec", None) \| ("cosecant", None)` | 1/sin | 1 col |
| `sincos(col=N)` | `("sincos", None)` | Fused pair | 2 cols |
| `asin(col=N)` | `("asin", None) \| ("arcsin", None)` | Inverse sine | 1 col |
| `acos(col=N)` | `("acos", None) \| ("arccos", None)` | Inverse cosine | 1 col |
| `atan(col=N)` | `("atan", None) \| ("arctan", None)` | Inverse tangent | 1 col |
| `atan2(col_y=, col_x=)` | `("atan2", None) \| ("arctan2", None)` | Two-arg inverse | 1 col |
| `acot(col=N)` | `("acot", None) \| ("arccot", None)` | Inverse cot | 1 col |
| `asec(col=N)` | `("asec", None) \| ("arcsec", None)` | Inverse sec | 1 col |
| `acsc(col=N)` | `("acsc", None) \| ("arccsc", None)` | Inverse csc | 1 col |
| `sinh(col=N)` | `("sinh", None)` | Hyperbolic sine | 1 col |
| `cosh(col=N)` | `("cosh", None)` | Hyperbolic cosine | 1 col |
| `tanh(col=N)` | `("tanh", None)` | Hyperbolic tangent | 1 col |
| `coth(col=N)` | `("coth", None)` | 1/tanh | 1 col |
| `sech(col=N)` | `("sech", None)` | 1/cosh | 1 col |
| `csch(col=N)` | `("csch", None)` | 1/sinh | 1 col |
| `sinhcosh(col=N)` | `("sinhcosh", None)` | Fused hyperbolic pair | 2 cols |
| `asinh(col=N)` | `("asinh", None) \| ("arcsinh", None)` | Inverse sinh | 1 col |
| `acosh(col=N)` | `("acosh", None) \| ("arccosh", None)` | Inverse cosh | 1 col |
| `atanh(col=N)` | `("atanh", None) \| ("arctanh", None)` | Inverse tanh | 1 col |
| `acoth(col=N)` | `("acoth", None)` | Inverse coth | 1 col |
| `asech(col=N)` | `("asech", None)` | Inverse sech | 1 col |
| `acsch(col=N)` | `("acsch", None)` | Inverse csch | 1 col |
| `sinpi(col=N)` | `("sinpi", None)` | sin(π·x) | 1 col |
| `cospi(col=N)` | `("cospi", None)` | cos(π·x) | 1 col |
| `tanpi(col=N)` | `("tanpi", None)` | tan(π·x) | 1 col |
| `sincospi(col=N)` | `("sincospi", None)` | Fused π-scaled pair | 2 cols |
| `asinpi(col=N)` | `("asinpi", None)` | asin(x)/π | 1 col |
| `acospi(col=N)` | `("acospi", None)` | acos(x)/π | 1 col |
| `atanpi(col=N)` | `("atanpi", None)` | atan(x)/π | 1 col |
| `atan2pi(col_y=, col_x=)` | `("atan2pi", None)` | atan2(y,x)/π | 1 col |
| `sind(col=N)` | `("sind", None)` | sin in degrees | 1 col |
| `cosd(col=N)` | `("cosd", None)` | cos in degrees | 1 col |
| `tand(col=N)` | `("tand", None)` | tan in degrees | 1 col |
| `atan2d(col_y=, col_x=)` | `("atan2d", None)` | atan2 output in degrees | 1 col |
| `versin(col=N)` | `("versin", None)` | 1 - cos(x) | 1 col |
| `haversin(col=N)` | `("haversin", None)` | (1 - cos(x))/2 | 1 col |
| `gudermannian(col=N)` | `("gudermannian", None) \| ("gd", None)` | atan(sinh(x)) | 1 col |
| `inv_gudermannian(col=N)` | `("inv_gudermannian", None) \| ("gd_inv", None)` | asinh(tan(x)) | 1 col |
| `haversine_distance(col_lat1=, ...)` | `("haversine_distance", None)` | Great-circle distance | 1 col |
| `trig_reduce(col=N)` | `("trig_reduce", None)` | Range reduction (3 cols) | 3 cols |
| `trig_triple(col=N)` | `("trig_triple", None)` | sin+cos+tan fused | 3 cols |
| `angle_analyze(col=N)` | `("angle_analyze", None)` | Diagnostic | Reports |
| `trig_identity_check(col=N)` | `("trig_identity_check", None)` | CI probe | Scalar |

**Total verbs: ~45.** Most are aliases of the ~8 kernel recipes — consistent with first-principles.md Phase 8.

---

## Parameter catalog (applies to all trig verbs via `.using()`)

| Parameter | Values | Default | Applies to |
|---|---|---|---|
| `precision` | `"strict"`, `"compensated"`, `"correctly_rounded"` | `"compensated"` (per spec.toml) | all |
| `angle_unit` | `"radians"`, `"degrees"`, `"gradians"`, `"turns"`, `"pi_scaled"` | `"radians"` | forward + inverse |
| `range_reduction` | `"auto"`, `"cody_waite"`, `"payne_hanek"` | `"auto"` | forward only (advanced) |
| `stabilization` | `"auto"`, `"naive"`, `"pade"`, `"exp_based"` | `"auto"` | hyperbolic only |
| `discover_level` | `"shallow"`, `"deep"` | `"shallow"` | `.discover()` |
| `discover_include` | list of probe IDs | `[]` | `.discover()` |

Any future parameter added follows the same pattern: name in `using.rs`'s `UsingBag`, surfaced through `.using()`, queried by the primitive at execution time.

---

## Canonical examples

### 1. Simple column transform

```tbs
# Load market data, compute sin of phase column, take variance
read_csv("market.csv")
  .sin(col=0)
  .variance(col=0)
```

### 2. Fused pair with downstream consumers

```tbs
read_csv("trajectory.csv")
  .sincos(col=2)                          # one reduction, two outputs
  .cov(col_x=0, col_y=1)                  # cov(cos, sin) — always ~0 for uniform phase
```

### 3. Angle-unit conversion without precision loss

```tbs
# Column is in degrees, computed at high precision:
read_csv("aviation.csv")
  .sincos(col=1).using(angle_unit="degrees", precision="correctly_rounded")
```

### 4. Haversine great-circle distance for geo data

```tbs
read_csv("cities.csv")
  .haversine_distance(col_lat1=1, col_lon1=2, col_lat2=3, col_lon2=4)
      .using(angle_unit="degrees", radius_km=6371.0)
  .describe()
```

### 5. Discovery on a mystery column

```tbs
read_csv("sensor.csv")
  .angle_analyze(col=0)
# Report: "Column looks like phase accumulator in radians. 8.2M rows, 99.8% monotonic.
#  max_abs = 1.2e8 — in Payne-Hanek regime. Suggest .sin(col=0).using(range_reduction='payne_hanek')."
```

### 6. CI correctness check

```tbs
# Run as part of unit tests:
synthetic_random_angles(n=100000)
  .trig_identity_check(col=0)
# Passes if max pythagorean deviation < 2 ulp.
```

### 7. Sweep for research / adversarial testing

```tbs
read_csv("adversarial_inputs.csv")
  .sin(col=0).sweep(precision=["strict", "correctly_rounded"])
# Two output columns; diff column shows where strict is within 1 ulp vs not.
```

### 8. Discover + collapse on a full pipeline

```tbs
read_csv("noisy_phases.csv")
  .sin(col=0).discover()
  .describe()
# Discovery picks best precision + reduction; describe() consumes the collapsed result.
```

---

## Implementation cost estimate

Each trig verb added to `tbs_executor.rs` is ~5-15 lines:

```rust
("sin", None) => {
    let c = col_arg(step, 0);
    let col = extract_col(&pipeline.frame().data, pn, pd, c);
    let precision = using.get_str("precision").unwrap_or("compensated");
    let angle_unit = using.get_str("angle_unit").unwrap_or("radians");
    let result = crate::recipes::libm::sin::sin_with(precision, angle_unit, &col);
    TbsStepOutput::Vector { name: "sin", values: result }
}
```

**Total for the full ~45-verb catalog**: ~500 lines of match arms + shared helpers. Most of the work is already done in the recipes layer; TBS is a thin dispatcher on top. Once pathmaker ships `sin_with(precision, angle_unit, col)` (which follows from TRIG-13 being complete), the TBS wiring is mechanical.

### Shared helper: `eval_trig_view`

To avoid 45 near-identical match arms, factor the dispatch:

```rust
fn eval_trig_view(
    verb: &str,
    step: &Step,
    using: &UsingBag,
    frame: &Frame,
) -> Result<TbsStepOutput, TbsError> {
    let precision = using.get_str("precision").unwrap_or_default_for(verb);
    let angle_unit = derive_angle_unit(verb, using);
    match verb {
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => ...,
        "sinpi" | "cospi" | "tanpi" => ...,  // same as above with angle_unit=pi_scaled
        "sind" | "cosd" | "tand" => ...,     // same with angle_unit=degrees
        ...
    }
}
```

The 45 verbs collapse to ~6 dispatch groups behind one helper. The match in `tbs_executor.rs` stays clean.

---

## Lint suggestions (for the executor's lint machinery)

When a trig verb is called, the executor can emit lint hints based on the input column's properties (the TRIG-5 probes run for free):

| Condition | Lint code | Severity | Message |
|---|---|---|---|
| `max_abs > 2^20·π/2` and `range_reduction="cody_waite"` | L401 | Error | "Cody-Waite invalid for \|x\| ≥ 2^20·π/2; use payne_hanek or auto." |
| `angle_unit="radians"` but distribution looks like degrees (unit_detect score > 0.8 for degrees) | L402 | Warning | "Input looks like degrees but angle_unit is radians. Check your unit." |
| `precision="correctly_rounded"` in a hot loop (>1M rows) | L403 | Info | "correctly_rounded is ~4-6× slower than compensated. Consider compensated for bulk work." |
| `sin(col)` and `cos(col)` both called on the same frame | L404 | Info | "Both sin and cos called — consider sincos(col) for one-pass evaluation." |
| `sec(col)` when `cos(col)` is also downstream | L405 | Info | "sec(col) = 1/cos(col); consider calling cos once and inverting." |

These are cheap (they read the cached column stats and pipeline graph) and they train users toward the `$1M/yr scientist` defaults from TRIG-6.

---

## Open questions

1. **Two-column output representation.** `sincos(col=0)` returns two columns. Does it return a `Vector2` type, or extend the frame with two columns named `sin_result, cos_result`? The existing executor has `TbsStepOutput::Vector { name, values }` for single vectors — needs a `Matrix` or `TwoColumns` variant. Existing ClusterView already handles multi-column output, so this infrastructure exists.

2. **Aliased verbs and documentation.** If `sind`, `sinpi`, `arcsin`, etc. are all aliases, the user-facing help should list the canonical form + show the alias. Suggest the spec.toml's `family` field grows an `aliases = [...]` entry that the executor registers at startup.

3. **`.discover()` for trig is structurally similar but not identical to clustering's.** The `DiscoveryResult` struct is clustering-specific (`labels`, `n_clusters`, `noise_fraction`). For trig, we'd want (`precision_chosen`, `reduction_chosen`, `identity_deviation`). Propose: introduce a `DiscoveryResult<K>` generic or a `TrigDiscoveryResult` sibling type. Existing Rand-Index agreement metric translates as "fraction of rows where two methods agree to 1 ulp."

4. **Should `trig_reduce` be user-visible?** Most TBS users won't want to think about reduction. But power users running TRIG-17 adversarial tests need it. Lean yes — it's the first-principles.md Aristotelian surface. Document it as `advanced=true` in the spec.toml so the IDE can hide it from autocomplete.

---

## Handoff notes

- **To pathmaker (TRIG-13, TRIG-14, TRIG-15, TRIG-16)**: your Rust recipes need a `*_with(precision, angle_unit, col)` entry point. Once that's in place, TBS wiring is ~15 lines per verb. The verb table above is the checklist.

- **To navigator**: all TRIG research-stream tasks (1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 19) are now complete or in flight by the right owners. Remaining: pathmaker finishing TRIG-14/15/16 implementation, adversarial/scientist closing TRIG-17/18, and TRIG-20 (master synthesis) plus TRIG-21 (scout recon) still open. If nobody else grabs TRIG-20, I'm well-positioned to write the README — I've touched every research doc except TRIG-4 (researcher's).

- **To math-researcher**: notation.md's Style 3 column was stub-quality at TRIG-8 time; this doc is its full buildout. When you update notation.md's kernel-level Style 2 entries, cross-reference the TBS table here for the user-facing names.

- **To scientist / adversarial**: lint codes L401-L405 above are suggested — feel free to reassign codes if they conflict with your existing tables.

---

## Deliverable status

- [x] TBS syntactic shape with executor-dispatch pattern
- [x] Forward family (sin/cos/tan/cot/sec/csc/sincos) with precision + angle_unit + reduction
- [x] Inverse family (asin/acos/atan/atan2/acot/asec/acsc) with output unit
- [x] Hyperbolic family (sinh/cosh/tanh/coth/sech/csch + inverses + sinhcosh)
- [x] π-scaled family (sinpi/cospi/tanpi/sincospi + inverses)
- [x] Degree aliases (sind/cosd/tand/atan2d)
- [x] Auxiliary verbs (versin/haversin/gudermannian + haversine_distance)
- [x] Fused + shared-pass verbs (sincos, sincospi, trig_triple, trig_reduce)
- [x] `.discover()` + `.sweep()` integration
- [x] Parameter catalog applied via `.using()`
- [x] Full 45-verb dispatch table
- [x] Canonical examples (8 scenarios)
- [x] Implementation cost estimate + shared-helper refactor
- [x] Lint suggestions (L401-L405)
- [x] Open questions + handoff notes
