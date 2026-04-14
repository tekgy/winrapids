# Angle Unit Parameterization

> TRIG-3 design doc. Owner: math researcher. Resolves whether angle unit
> is a `using()` parameter on one recipe, a separate recipe per unit, or a
> hybrid. Documents why each unit matters and what exact arithmetic each
> unit enables.

---

## The five units

| Unit | 1 full turn | Typical domains | Range-reduction constant |
|---|---|---|---|
| **Radians** | `2π ≈ 6.2831853...` | Physics, pure math, most APIs | `π/2` (transcendental, Payne-Hanek for large x) |
| **Degrees** | `360` | Navigation, engineering, MATLAB `sind` | `90` (exact) |
| **Gradians** | `400` | Artillery, French surveying | `100` (exact) |
| **Turns** (cycles) | `1` | Audio phase, computer graphics, rotational encoders | `1/4` (exact — power of 2) |
| **Pi-scaled** (half-turns) | `2` | IEEE 754-2019, C23, CUDA `sinpi` | `1/2` (exact — power of 2) |

Exactness refers to **range-reduction precision**. All five units span
`ℝ`, and the output of `sin`/`cos`/`tan` is the same real number — the
question is how much precision is **lost** reducing the argument to the
core interval.

---

## Why range reduction is the deciding factor

For any unit `U`, `sin_U(x) = sin(k·x)` where `k = 2π/turn_U`. To evaluate,
we reduce `x` modulo `turn_U/4` to get `x = q·(turn_U/4) + r` with `r ∈
[-turn_U/8, turn_U/8]`, then evaluate on the radian core.

**The precision loss is entirely in the `mod` step**:

- `mod 1` (turns): exact for any finite f64 (1 is a power of 2).
- `mod 1/2` (pi-scaled): exact.
- `mod 90` (degrees): exact for `|x| < 2⁵³` (90 fits in an f64 mantissa
  with room; `2⁵³ · 90 = 8.1e+17`).
- `mod 100` (gradians): same — exact for `|x| < 2⁵³`.
- `mod π/2` (radians): **inexact** for any `x`. `π/2` has no finite f64
  representation. Naive reduction loses `⌊log₂ |x|⌋` bits. Cody-Waite
  buys back ~50 bits; Payne-Hanek buys back all of them at 1200-bit cost.

**Therefore**: only radians require the Cody-Waite / Payne-Hanek ladder.
Every other unit gets clean bit-perfect reduction for free on sane inputs.

---

## Why turns enables exact arithmetic

The most common angles are rational numbers of turns: `0, 1/12, 1/8, 1/6,
1/4, 1/3, 1/2, ...`. In radians these become irrationals (`π/6`, `π/4`,
...) that cannot be represented exactly in f64. In turns, `1/4` is
**exact** (it's `0.25`, literally `1p-2`), and `sin_turn(1/4) = 1` comes
out **exactly** with no rounding.

**Example**:
```
sin(π/2)  in radians: 6.123233995736766e-17  (should be 0, but π/2 isn't exact in f64)
sinpi(1/2) pi-scaled:  1 exactly
sin_turn(1/4) turns:   1 exactly
```

The `π/6, π/4, π/3` triad are all rational in turns, so for any
"standard" angle from a high-school unit circle, turns give the exact
answer. The same property holds for every multiple of `1/60` turn
(degrees: `6°, 12°, 18°, ...`), `1/400` turn (gradians: `1g, 2g, ...`),
and `1/n` turn for any `n` that is a product of small primes whose
reciprocal is exactly representable.

**This is not a novelty**: it's the reason IEEE 754-2019 recommended
pi-scaled operations, and the reason why computer graphics APIs have
slowly migrated to turns (WebGPU's rotation, Unreal's "Rotator" axis
phases, Bevy's `TAU`-based rotations).

---

## The decision: parameter vs. separate recipe

### Option A: single recipe with `unit` parameter

```
[[parameters]]
key = "unit"
kind = "method"
[parameters.default]
using = "radians"
[parameters.domain]
kind = "enum"
values = ["radians", "degrees", "gradians", "turns", "pi"]
```

**Pros**: one implementation per function; the IDE shows one entry; `sin`
stays a single mathematical concept.

**Cons**: the **compilation** is genuinely different per unit. Radians
need Payne-Hanek; turns need `frac()`. Burying this in one recipe means
the compiler has to dispatch on `unit` at lowering time, making the
generated code non-monomorphic.

### Option B: one recipe per (function, unit) pair

`sin_rad.rs`, `sin_deg.rs`, `sin_grad.rs`, `sin_turn.rs`, `sin_pi.rs` —
five recipes per function, ~150 recipes total for the trig family.

**Pros**: each recipe is monomorphic and compiles to the minimal code
path. `sin_turn` never contains Payne-Hanek code.

**Cons**: explodes the file count. Duplicates the polynomial core and the
special-case handling.

### Option C: hybrid — shared core primitive, per-unit recipes

One **primitive** `sin_core_reduced(r_hi, r_lo)` that takes a
pre-reduced argument in `[-π/4, π/4]` (always radians — the core is
unit-free since it's evaluating the power series). Five per-unit
**recipes** that handle reduction appropriately:

```
sin_rad(x)   → reduce_mod_pio2(x) → sin_core_reduced(r_hi, r_lo) → quadrant_fixup
sin_deg(x)   → reduce_mod_90(x) · (π/180) → sin_core_reduced(...) → fixup
sin_turn(x)  → reduce_mod_qturn(x) · (π/2) → sin_core_reduced(...) → fixup
sin_pi(x)    → reduce_mod_half(x) · (π/2) → sin_core_reduced(...) → fixup
sin_grad(x)  → reduce_mod_100(x) · (π/200) → sin_core_reduced(...) → fixup
```

The multiplication `r_exact · (π/180)` etc. reintroduces a small rounding
error — but it's a **single multiply**, not a 1200-bit Payne-Hanek. The
per-unit reduction stays in the exact domain.

**This is the choice.** It matches tambear's "methods compose primitives"
architecture from CLAUDE.md: `sin_core_reduced` is the primitive; each
per-unit function is a thin composition (reduce + multiply + core + fixup).

---

## How it appears in `using()`

To preserve the user's expectation that "sin" is one function, the TBS
surface and Python binding expose a single `sin` with a `unit` parameter:

```
tambear.sin(x).using(unit="turns")
tambear.sin(x).using(unit="degrees")
tambear.sin(x)                       # defaults to "radians"
```

Under the hood, `using(unit=...)` dispatches to the corresponding recipe:

| User call | Recipe invoked |
|---|---|
| `sin(x).using(unit="radians")` | `sin_rad` |
| `sin(x).using(unit="degrees")` | `sin_deg` |
| `sin(x).using(unit="gradians")` | `sin_grad` |
| `sin(x).using(unit="turns")` | `sin_turn` |
| `sin(x).using(unit="pi")` | `sin_pi` |

No default-unit dispatching: the user always chooses or gets the
radian default. **The IDE shows five recipe cards stacked**, each
describing its own reduction strategy, polynomial evaluation, error
bound, and cost.

---

## The shared reduction primitives

The reduction step is the same pattern in all five cases — `reduce_mod_M`
for various `M`. Factor it out as a primitive `reduce_mod_power_of_two`
that handles any power-of-2 `M` via `x − M·⌊x/M + 0.5⌋`, and a separate
`reduce_mod_exact_integer` for 90 and 100. Only radians get the
`reduce_mod_pio2` specialized path with Cody-Waite + Payne-Hanek.

**Primitive list for TRIG-10**:

- `reduce_mod_pio2(x)` — radians, returns `(q, r_hi, r_lo)`.
- `reduce_mod_half(x)` — pi-scaled, returns `(q, r)` exact.
- `reduce_mod_qturn(x)` — turns, returns `(q, r)` exact. (`qturn = 1/4`.)
- `reduce_mod_n_exact(x, n)` — degrees / gradians, returns `(q, r)`
  exact when `|x| < 2⁵³ · n`.

The `(q, r)` return from turns / pi-scaled / degrees / gradians are
**exact**, so the core can use the `r_lo = 0` path. Only radians needs
the two-part r.

---

## Special constants per unit

Every unit has a set of angles that produce exact output. Cataloguing:

### Turns
```
sin(0)    = 0         sin(1/4) = 1     sin(1/2) = 0      sin(3/4) = -1
cos(0)    = 1         cos(1/4) = 0     cos(1/2) = -1     cos(3/4) = 0
```
The catalog of "n/8 turns" angles yields 8 exact sin values (the
unit-circle quadrant corners).

### Pi-scaled
```
sin_pi(0)    = 0       sin_pi(1/2) = 1    sin_pi(1) = 0   sin_pi(3/2) = -1
cos_pi(0)    = 1       cos_pi(1/2) = 0    cos_pi(1) = -1  cos_pi(3/2) = 0
```
Same story at half the argument range.

### Degrees
```
sin(0°)    = 0         sin(90°) = 1       sin(180°) = 0    sin(270°) = -1
sin(30°)   = 0.5       sin(60°) = √3/2 (NOT exact)
```
The "nice" angles (`0, 30, 45, 60, 90, ...`) give exact outputs only
at the quadrant corners — the 30° / 60° / 45° values require √3/2 / √2/2
which are irrationals.

### Radians — nothing is exact
Because every rational-π radian angle is a transcendental number that
can't be f64-represented, **no radian input gives an exact output**.
This is the fundamental argument for pi-scaled and turn units.

---

## When to default to which unit

Per `using()` precedent: **default is radians** because that's what every
libm on the planet does, and users would be shocked otherwise. But the
IDE and docs **surface turns and pi-scaled heavily** as the "exact-math"
options for users who care.

**Recommendation for tambear**: radian default everywhere, with a strong
hint in the IDE step-card for any user calling `sin` with a rational
argument that "you probably want `unit='pi'` or `unit='turns'` for
exact output." This is Layer 2 (override transparency) in the
layered-architecture doc — the system tells you what it would have
chosen if you'd asked.

---

## Interaction with TRIG-4 shared intermediates

The shared intermediate from the catalog is
`TrigQuadrantReduction{unit}` — **parameterized by unit**. A consumer
requesting `sin(x).using(unit="radians")` cannot reuse a
TamSession-cached reduction for `sin(x).using(unit="turns")` because
the reduction output is mathematically different (`r ∈ [-π/4, π/4]` vs.
`r ∈ [-1/8, 1/8]`).

**Compatibility rule**: the tag carries the unit; the compatibility
predicate is `self.unit == other.unit`. This is the correctness
invariant from CLAUDE.md §3 (conditional sharing).

---

## Atan2 — the axis convention question

For **inverse** trig, the unit parameter lives on the **output**, not
the input. `atan2(y, x).using(unit="radians")` returns an answer in
`(-π, π]`; `.using(unit="degrees")` returns in `(-180, 180]`. The
underlying computation is identical; only the final multiply changes.

**This simplifies inverse trig**: one radian recipe per function, with
a post-scale on output. No need for five separate recipes.

---

## Decision summary

| Question | Answer |
|---|---|
| Unit is parameter or separate recipe? | **Hybrid**: shared core primitive + per-unit recipes for forward trig; single radian recipe + output scale for inverse trig. |
| User-facing surface? | `sin(x).using(unit=...)` — single function with the parameter. |
| Default unit? | Radians (libm convention). Strongly surface turns / pi-scaled in IDE hints. |
| Unit domain values? | `"radians"`, `"degrees"`, `"gradians"`, `"turns"`, `"pi"`. |
| Shared intermediate scope? | Per-unit — `TrigQuadrantReduction{unit}` tag. |
| Which unit skips Payne-Hanek? | All except radians. |
| Which unit gives exact quadrant-corner values? | All except radians. |

---

## Next

- TRIG-4 uses this decision as input (shared intermediate is unit-tagged).
- TRIG-9 atom decomposition covers `reduce_mod_*` as primitives.
- TRIG-13/14/15/16 (pathmaker) ships one recipe per (function, unit) pair
  using the shared core primitive.
- TRIG-11 documents that turns / pi-scaled / degrees / gradians have
  **one** compilation strategy each (strict = correctly_rounded =
  compensated because the reduction is exact), while radians retains
  the three-strategy ladder.
