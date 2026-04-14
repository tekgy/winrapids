# Notation — Three Styles Per Trig Function

**Author**: Aristotle (tambear-trig)
**Date**: 2026-04-13
**Status**: Deliverable for TRIG-8. Shared with math-researcher, who owns the catalog.

**Purpose**: Every trig function needs three formal notations — one for publication (LaTeX/math-paper style), one for the recipe-tree (tambear-composition style), and one for TBS (the user-facing scripting surface). These three notations encode the same mathematical object at three different levels of abstraction; having all three in a uniform table lets the team, the test harness, and the documentation generator cross-reference without ambiguity.

**Relation to first-principles.md**: This doc assumes the reconstruction from Phase 8 — that the library's hand-written surface is ~8 kernel-level recipes with ~30+ one-line views. Each view gets the same three-notation treatment, because users who read the docs shouldn't need to know which are kernels and which are views.

---

## The three styles

### Style 1 — **Publication notation** (what a paper would write)

LaTeX-style math with the conventional symbol, domain, codomain, and defining identity. No implementation detail; expresses the mathematical content.

Example for sin:
```
sin : \mathbb{R} \to [-1, 1]
sin(\theta) = \Im(e^{i\theta}) = \frac{e^{i\theta} - e^{-i\theta}}{2i}
```

### Style 2 — **Recipe-tree notation** (how tambear decomposes it)

A nested-call form showing exactly which primitives and lower-level recipes are called. Parallel to the decomposition trees in the architecture doc's Example 1 (exp). This notation IS the recipe body at a structural level.

Example for sin (post-reconstruction):
```
sin(x) := sincos(x).1
sincos(x) := sincos_kernel(rem_pio2(x))
sincos_kernel((q, r_hi, r_lo)) := quadrant_fixup(q, poly_sin(r_hi, r_lo), poly_cos(r_hi, r_lo))
```

### Style 3 — **TBS notation** (what the user writes)

The scripting-surface expression. This is what the user types into a `.tbs` file or an interactive prompt. Follows the chains-only, ~100-word vocabulary documented in the TBS IDE vision.

Example for sin:
```
sin(col=0)
sin(col=0).using(precision="correctly_rounded")
sin(col=0).using(angle_unit="degrees", range_reduction="payne_hanek")
```

---

## Forward family

### sin — sine

| Style | Notation |
|---|---|
| Publication | `sin : ℝ → [-1, 1];  sin(θ) = Im(e^{iθ})` |
| Recipe-tree | `sin(x) := sincos(x).1` |
| TBS | `sin(col=0)` |

### cos — cosine

| Style | Notation |
|---|---|
| Publication | `cos : ℝ → [-1, 1];  cos(θ) = Re(e^{iθ})` |
| Recipe-tree | `cos(x) := sincos(x).0` |
| TBS | `cos(col=0)` |

### tan — tangent

| Style | Notation |
|---|---|
| Publication | `tan : ℝ \setminus \{(k+1/2)π\} → ℝ;  tan(θ) = sin(θ)/cos(θ)` |
| Recipe-tree | `tan(x) := tan_kernel(rem_pio2(x))` — fused kernel avoiding division |
| TBS | `tan(col=0)` |

### cot — cotangent

| Style | Notation |
|---|---|
| Publication | `cot : ℝ \setminus \{kπ\} → ℝ;  cot(θ) = cos(θ)/sin(θ) = 1/tan(θ)` |
| Recipe-tree | `cot(x) := fdiv(1.0, tan(x))` |
| TBS | `cot(col=0)` |

### sec — secant

| Style | Notation |
|---|---|
| Publication | `sec : ℝ \setminus \{(k+1/2)π\} → (-∞,-1] ∪ [1,+∞);  sec(θ) = 1/cos(θ)` |
| Recipe-tree | `sec(x) := fdiv(1.0, cos(x))` |
| TBS | `sec(col=0)` |

### csc — cosecant

| Style | Notation |
|---|---|
| Publication | `csc : ℝ \setminus \{kπ\} → (-∞,-1] ∪ [1,+∞);  csc(θ) = 1/sin(θ)` |
| Recipe-tree | `csc(x) := fdiv(1.0, sin(x))` |
| TBS | `csc(col=0)` |

### sincos — fused forward pair (PRIMITIVE RECIPE)

| Style | Notation |
|---|---|
| Publication | `sincos : ℝ → [-1,1]²;  sincos(θ) = (cos(θ), sin(θ))` |
| Recipe-tree | `sincos(x) := sincos_kernel(rem_pio2(x))` |
| TBS | `sincos(col=0)` → two-column output |

### sincos_kernel — the primitive kernel (for math-researcher's catalog)

| Style | Notation |
|---|---|
| Publication | `K : [-π/4, π/4] × ℤ/4 → [-1,1]²;  K(r, q) = rotate_by_quadrant(q, (poly_cos(r), poly_sin(r)))` |
| Recipe-tree | `sincos_kernel((q, r_hi, r_lo)) := quadrant_fixup(q, (poly_cos(r_hi, r_lo), poly_sin(r_hi, r_lo)))` |
| TBS | (not user-facing; internal recipe) |

---

## Inverse family

### asin — arcsine

| Style | Notation |
|---|---|
| Publication | `asin : [-1, 1] → [-π/2, π/2];  sin(asin(x)) = x` |
| Recipe-tree | `asin(x) := atan2(x, fsqrt(fsub(1.0, fmul(x, x))))` |
| TBS | `asin(col=0)` |

### acos — arccosine

| Style | Notation |
|---|---|
| Publication | `acos : [-1, 1] → [0, π];  cos(acos(x)) = x` |
| Recipe-tree | `acos(x) := atan2(fsqrt(fsub(1.0, fmul(x, x))), x)` |
| TBS | `acos(col=0)` |

### atan — arctangent (single-arg)

| Style | Notation |
|---|---|
| Publication | `atan : ℝ → (-π/2, π/2);  tan(atan(x)) = x` |
| Recipe-tree | `atan(x) := atan2(x, 1.0)` |
| TBS | `atan(col=0)` |

### atan2 — two-argument arctangent (PRIMITIVE RECIPE)

| Style | Notation |
|---|---|
| Publication | `atan2 : ℝ² \setminus \{(0,0)\} → (-π, π];  atan2(y, x)` gives the angle of `(x, y)` from the positive x-axis, with sign correctly resolved across all quadrants |
| Recipe-tree | `atan2(y, x) := atan2_kernel(y, x)` — hand-written kernel with branch-cut handling |
| TBS | `atan2(col_y=0, col_x=1)` |

### acot, asec, acsc — inverse cot/sec/csc

| Function | Publication | Recipe-tree | TBS |
|---|---|---|---|
| acot | `acot(x) = atan(1/x)` (principal branch) | `acot(x) := atan2(1.0, x)` (cleaner branch) | `acot(col=0)` |
| asec | `asec(x) = acos(1/x)` | `asec(x) := acos(fdiv(1.0, x))` | `asec(col=0)` |
| acsc | `acsc(x) = asin(1/x)` | `acsc(x) := asin(fdiv(1.0, x))` | `acsc(col=0)` |

---

## Hyperbolic family

### sinh — hyperbolic sine

| Style | Notation |
|---|---|
| Publication | `sinh : ℝ → ℝ;  sinh(x) = (e^x - e^{-x})/2` |
| Recipe-tree | `sinh(x) := sinh_kernel(x)` — needs a recipe because the naive formula overflows for `|x| > ~709` and cancels for `|x| < ~0.5` |
| TBS | `sinh(col=0)` |

### cosh — hyperbolic cosine

| Style | Notation |
|---|---|
| Publication | `cosh : ℝ → [1, ∞);  cosh(x) = (e^x + e^{-x})/2` |
| Recipe-tree | `cosh(x) := cosh_kernel(x)` — same overflow concerns as sinh |
| TBS | `cosh(col=0)` |

### tanh — hyperbolic tangent

| Style | Notation |
|---|---|
| Publication | `tanh : ℝ → (-1, 1);  tanh(x) = sinh(x)/cosh(x) = (e^{2x}-1)/(e^{2x}+1)` |
| Recipe-tree | `tanh(x) := tanh_kernel(x)` — uses `1 - 2/(exp(2x)+1)` form for numerical stability at large |x| |
| TBS | `tanh(col=0)` |

### sinh_kernel, cosh_kernel, tanh_kernel — the primitive recipes

Math-researcher owns the specific formulas. Each has a near-zero path (Taylor series) and a large-|x| path (avoiding overflow), bridged at a regime boundary.

| Style | sinh_kernel |
|---|---|
| Publication | `sinh_kernel(x) = { x + x³/6 + x⁵/120 + ...   for |x| < 1; (e^x - e^{-x})/2   for 1 ≤ |x| ≤ ~709; copysign(e^{|x|-ln2}, x)   for |x| > 709 (overflow-safe) }` |
| Recipe-tree | Regime dispatch + mpmath-tuned polynomial for the near-zero branch + exp-based for medium + scaled-exp for large |
| TBS | (internal; not user-facing) |

### coth, sech, csch — hyperbolic cot/sec/csc

| Function | Publication | Recipe-tree | TBS |
|---|---|---|---|
| coth | `coth(x) = cosh(x)/sinh(x)` | `coth(x) := fdiv(1.0, tanh(x))` | `coth(col=0)` |
| sech | `sech(x) = 1/cosh(x)` | `sech(x) := fdiv(1.0, cosh(x))` | `sech(col=0)` |
| csch | `csch(x) = 1/sinh(x)` | `csch(x) := fdiv(1.0, sinh(x))` | `csch(col=0)` |

### asinh, acosh, atanh — inverse hyperbolics

| Function | Publication | Recipe-tree | TBS |
|---|---|---|---|
| asinh | `asinh(x) = ln(x + √(x² + 1))` | `asinh(x) := log(fadd(x, fsqrt(fmadd(x, x, 1.0))))` | `asinh(col=0)` |
| acosh | `acosh(x) = ln(x + √(x² - 1))`, `x ≥ 1` | `acosh(x) := log(fadd(x, fsqrt(fsub(fmul(x, x), 1.0))))` | `acosh(col=0)` |
| atanh | `atanh(x) = ½ ln((1+x)/(1-x))`, `|x| < 1` | `atanh(x) := fmul(0.5, log(fdiv(fadd(1.0, x), fsub(1.0, x))))` | `atanh(col=0)` |

Note: each of these composed forms has numerical-stability gotchas (`asinh` cancels for large negative x; `acosh` cancels near x=1; `atanh` diverges at x=±1). The recipe-tree notation above gives the naive composition; the actual implementations will substitute numerically stable variants. Math-researcher owns the stabilized forms.

### acoth, asech, acsch — inverse hyperbolic cot/sec/csc

| Function | Publication | Recipe-tree | TBS |
|---|---|---|---|
| acoth | `acoth(x) = ½ ln((x+1)/(x-1))`, `|x| > 1` | `acoth(x) := fmul(0.5, log(fdiv(fadd(x, 1.0), fsub(x, 1.0))))` | `acoth(col=0)` |
| asech | `asech(x) = ln((1 + √(1-x²))/x)`, `0 < x ≤ 1` | `asech(x) := log(fdiv(fadd(1.0, fsqrt(fsub(1.0, fmul(x, x)))), x))` | `asech(col=0)` |
| acsch | `acsch(x) = ln(1/x + √(1/x² + 1))` | `acsch(x) := asinh(fdiv(1.0, x))` | `acsch(col=0)` |

---

## Pi-scaled family

### sinpi, cospi, tanpi — circular, input pre-scaled by π

| Function | Publication | Recipe-tree | TBS |
|---|---|---|---|
| sinpi | `sinpi : ℝ → [-1, 1];  sinpi(x) = sin(πx)` | `sinpi(x) := sincospi(x).1` | `sinpi(col=0)` or `sin(col=0).using(angle_unit="pi_scaled")` |
| cospi | `cospi : ℝ → [-1, 1];  cospi(x) = cos(πx)` | `cospi(x) := sincospi(x).0` | `cospi(col=0)` |
| tanpi | `tanpi : ℝ \ {(k+½)} → ℝ;  tanpi(x) = tan(πx)` | `tanpi(x) := tan_kernel(rem_half_turn(x) ∘ scale_by_pi_in_kernel)` | `tanpi(col=0)` |

### sincospi — fused pi-scaled forward pair

| Style | Notation |
|---|---|
| Publication | `sincospi : ℝ → [-1, 1]²;  sincospi(x) = (cos(πx), sin(πx))` |
| Recipe-tree | `sincospi(x) := sincos_kernel(rem_half_turn(x))` — same kernel, different reduction |
| TBS | `sincospi(col=0)` → two-column output |

Note: `rem_half_turn` is the dyadic reduction — `k = frint(2x)`, residual `r = 2x - k`, `r ∈ [-0.5, 0.5]`, and the kernel multiplies r by π only on the final small residual. This is exact until the final kernel step, avoiding precision loss.

### asinpi, acospi, atanpi, atan2pi — inverse pi-scaled

| Function | Publication | Recipe-tree | TBS |
|---|---|---|---|
| asinpi | `asinpi(x) = asin(x)/π` | `asinpi(x) := fdiv(asin(x), PI)` — or better, a dedicated kernel for precision | `asinpi(col=0)` |
| acospi | `acospi(x) = acos(x)/π` | `acospi(x) := fdiv(acos(x), PI)` | `acospi(col=0)` |
| atanpi | `atanpi(x) = atan(x)/π` | `atanpi(x) := fdiv(atan(x), PI)` | `atanpi(col=0)` |
| atan2pi | `atan2pi(y, x) = atan2(y, x)/π` | `atan2pi(y, x) := fdiv(atan2(y, x), PI)` | `atan2pi(col_y=0, col_x=1)` |

Note: the division-by-π form is the standard composition. A dedicated kernel that produces the answer in units of half-turns directly would be more accurate. Math-researcher to evaluate whether that's worth a separate recipe.

---

## Angle-unit parameterized forms

Each of the forward + inverse circular functions accepts an `angle_unit` parameter via `using()`. The TBS style shows this; the recipe-tree style dispatches on the unit to choose the reduction primitive.

Example for sin with all five unit options:

| TBS call | Recipe-tree dispatch |
|---|---|
| `sin(col=0).using(angle_unit="radians")` | `sincos_kernel(rem_pio2(x)).1` |
| `sin(col=0).using(angle_unit="degrees")` | `sincos_kernel(rem_degrees_90(x) ∘ deg_to_rad_residual).1` |
| `sin(col=0).using(angle_unit="gradians")` | `sincos_kernel(rem_gradians_100(x) ∘ grad_to_rad_residual).1` |
| `sin(col=0).using(angle_unit="turns")` | `sincos_kernel(rem_turns_quarter(x) ∘ turn_to_rad_residual).1` |
| `sin(col=0).using(angle_unit="pi_scaled")` | `sincos_kernel(rem_half_turn(x) ∘ pi_scale_residual).1` — equivalent to sinpi |

The `deg_to_rad_residual` etc. are **post-reduction residual conversions** — they multiply the small residual (|r| ≤ 45° or |r| ≤ 50 grads etc.) by the radian conversion constant, not the full input. This is where T-Phase-1-A9 (pre-multiply is wrong) gets fixed.

---

## Style 2 (recipe-tree) as the canonical form

When we write tests, specs, or migration plans, Style 2 is the authoritative form. It:

1. Matches the actual recipe body line-by-line (per the architecture doc's "no inline arithmetic" rule — every operation is an explicit primitive call).
2. Makes the decomposition tree visible, which is what test harnesses and the .tam IR will walk.
3. Highlights shared intermediates — every appearance of `sincos(x)` in the recipe-tree notation is a candidate for TamSession sharing.
4. Makes review easy: reviewers check "does this recipe-tree match what the spec.toml's `primitives_used` and `sharing` fields claim?"

**For math-researcher's catalog**: every kernel-level recipe (sincos_kernel, tan_kernel, atan2_kernel, sinh_kernel, cosh_kernel, tanh_kernel) needs its Style 2 notation written out fully, because those are the hand-written mathematical content.

**For pathmaker**: every view-level function (sin, cos, tan, asin, ...) needs its Style 2 notation written out, because that's the recipe body (1-5 lines each).

**For documentation generator (future)**: iterate over a .toml catalog keyed by function name, emit the three-style table into HTML docs + math paper + TBS reference card.

---

## Open items

- **Complex-input forms.** None of the notations above cover complex arguments (`sin(z)` for `z ∈ ℂ`). The complex family is a future expansion; when we add it, we'll add a fourth column or a separate table.
- **Vector-valued forms.** TBS Style 3 already naturally expresses column input; if we add matrix/tensor overloads they'd slot in under the same TBS notation with broader column-spec keys.
- **Numerically-stabilized variants.** Some functions (atanh, acosh, sinh near zero, tanh at large |x|) require regime-dispatched formulas. The table above shows the canonical mathematical form; the recipe body dispatches internally. Spec.tomls should call out the regime structure under `decomposition.regime_dispatch`.

---

## Handoff notes

**To math-researcher**: this doc is scaffolding. The three-style tables above cover ~30 functions; I've filled in Style 1 (publication) fully and Style 3 (TBS) fully. Style 2 (recipe-tree) is filled at the **view** level everywhere, and the **kernel** level is stubbed (sincos_kernel, tan_kernel, atan2_kernel, sinh_kernel, cosh_kernel, tanh_kernel). You own the kernel-level Style 2 content — the specific polynomial forms, CORDIC iteration counts, regime boundaries. Extend each kernel row when you write its recipe.

**To pathmaker**: if you write spec.tomls while this is being reviewed, please use Style 2 recipe-tree syntax in the `long_description` field so specs agree structurally with what the Rust code will actually do. Use Style 3 in the `examples` field so users see the TBS surface.

**To navigator**: Phase 8 of first-principles.md is the decision point. If the team rejects the ~8-kernel reconstruction, the view-level Style 2 entries here become wrong (every function becomes a full recipe, not a one-liner). I'd prefer to know before math-researcher deeply invests in the kernel-level entries.
