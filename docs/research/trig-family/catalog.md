# Trig Family Catalog

> Master enumeration of every trigonometric function tambear implements from first
> principles. Owner: math researcher (trig team). Driving tasks TRIG-1, TRIG-2,
> TRIG-3, TRIG-4, TRIG-9, TRIG-11.

This catalog is the **mathematical conscience** input to pathmaker. For every
function we list: (1) original paper / historical source, (2) domain and codomain,
(3) assumption-violation behavior, (4) known edge cases and failure modes,
(5) flavors (standard vs. better), (6) relationship to sibling functions in the
family, (7) what state-of-the-art implementations do.

Citations resolve to entries in `references.md`. Pathmaker lowers entries here
into `.spec.toml` + `.rs` recipes under `crates/tambear/src/recipes/libm/`.

---

## Scope

Thirty-two base functions across six families. The catalog is expanded by
parameterization — every forward function gains per-unit variants (radians,
degrees, gradians, turns, pi-scaled) via TRIG-3; every function gains tiny /
small / medium / large range variants via TRIG-2. Base entries below specify
the **radian** reference implementation unless the function is explicitly
pi-scaled.

| Family | Count | Functions |
|---|---|---|
| Forward core | 6 | `sin`, `cos`, `tan`, `cot`, `sec`, `csc` |
| Forward fused | 2 | `sincos` (pair), `sincospi` (pair) |
| Inverse core | 4 | `asin`, `acos`, `atan`, `atan2` |
| Inverse reciprocal | 3 | `acot`, `asec`, `acsc` |
| Hyperbolic forward | 3 | `sinh`, `cosh`, `tanh` |
| Hyperbolic reciprocal | 3 | `coth`, `sech`, `csch` |
| Hyperbolic inverse | 6 | `asinh`, `acosh`, `atanh`, `acoth`, `asech`, `acsch` |
| Pi-scaled | 3 | `sinpi`, `cospi`, `tanpi` + inverses `asinpi`, `acospi`, `atanpi`, `atan2pi` |
| Fused hyperbolic | 2 | `sinhcosh`, `expm1_coshm1` (exp(x)-1 + cosh(x)-1 sharing) |
| Circular↔hyperbolic bridge | 2 | `gd` (gudermannian), `gd⁻¹` (inverse gudermannian) |

Per-unit multiplication by {rad, deg, grad, turn, pi-scaled} brings the surface
to roughly ~150 callable entries. Pathmaker produces one `.spec.toml` per
base entry with the angle unit as a `using()` parameter per TRIG-3.

---

## Notation

- `ULP(x)` — units in the last place at `x`. For f64, `ULP(1.0) = 2⁻⁵²`.
- `fdlibm` — Sun Microsystems' freely distributable libm (1993), the de facto
  ground truth for POSIX libm behavior.
- `CORE-MATH` — INRIA correctly-rounded libm (Sibidanov, Zimmermann, Glondu 2022+).
- `Payne-Hanek` — 1983 large-argument reduction using a 1200-bit `2/π` table.
- `Cody-Waite` — 1980 medium-argument reduction using split-precision constants.
- `Remez` — exchange algorithm producing minimax polynomial approximants.
- `DD` — double-double (106-bit) arithmetic.
- `TMD` — Table Maker's Dilemma: the cost of finding worst-case rounding inputs.

---

## Family 1 — Forward Core (radians)

### sin(x)

1. **Source.** Formalized identities trace to the *Āryabhaṭīya* (499 CE) and
   medieval Kerala school (Mādhava, ~1400 CE) who gave the Taylor series
   centuries before Gregory / Newton. Modern computing: Cody & Waite 1980;
   Payne & Hanek 1983; fdlibm `__kernel_sin` + `__ieee754_rem_pio2` (Sun 1993).
2. **Domain.** `x ∈ ℝ` (any finite f64). **Codomain.** `[-1, 1]`.
3. **Assumption violations.**
   - `x = NaN` → `NaN` (quiet, no signaling).
   - `x = ±∞` → `NaN` and `FE_INVALID` per IEEE 754-2019 §9.2.
   - `|x| ≥ 2¹⁰²³` is still well-defined mathematically but has fewer than
     one bit of angular information; we return a bounded answer without
     raising inexact.
4. **Edge cases.**
   - `sin(0) = +0` and `sin(-0) = -0` — sign preservation is required by IEEE.
   - Large arguments near exact multiples of `π`: naive reduction loses all
     precision. fdlibm's Payne-Hanek kicks in at `|x| ≥ 2²⁰ · π/2 ≈ 1.65e+06`.
   - Hard cases (Muller's *Handbook*, Tbl 11.5): `sin(2.225e-308)` (subnormal),
     `sin(6381956970095103 · 2⁻⁵¹)` (near `(1024 + 1/2)·π`) — Kahan's TMD
     worst case for f64.
5. **Flavors.**
   - **Standard** (our `strict`): Cody-Waite medium + Payne-Hanek large,
     degree-11 minimax core polynomial. ≤ 1 ULP across the domain.
   - **Correctly rounded** (CR): Sibidanov 2022 — requires higher working
     precision and extended lookup tables. Upgrade path: swap core eval
     for DD polynomial; reduction must return `(r_hi, r_lo, r_ll)` triple.
6. **Relatives.** See `cos` (quadrant-rotated twin), `sincos` (fused pair),
   `sinpi` (pi-scaled — skips reduction when x is rational times π).
7. **State of the art.**
   - fdlibm: ~1 ULP, three-part `π/2`, `__kernel_sin` odd minimax poly of
     degree 13 (S1..S6 + residual term).
   - CORE-MATH `sin()`: correctly rounded in f64 via 256-entry table plus
     degree-9 DD polynomial; ~1.4× fdlibm cost.
   - glibc: fdlibm base with Ziv's technique for adaptive precision.
   - CUDA `__sinf` / `__sin`: hardware-range-reduced (uses `SIN/COS` SFU
     instruction); 2-ULP intrinsic vs. 1-ULP function.
   - Julia Base.sin: fdlibm coefficients with Payne-Hanek at 2²⁵.

### cos(x)

1. **Source.** Same lineage as `sin` — computed by quadrant-rotating the
   `sin` core (`cos(r) = 1 − r²/2 + r⁴·Q(r²)`). fdlibm `__kernel_cos`.
2. **Domain/codomain.** `ℝ → [-1, 1]`.
3. **Violations.** Same as `sin`; additionally `cos(0) = 1` must be exact
   (IEEE 754-2019 requires no inexact flag at zero).
4. **Edge cases.**
   - `cos(π/2)` in f64 is not exactly zero — the f64 representation of π/2
     is `1.5707963267948966`, and `cos` of that is `6.123233995736766e-17`.
     This is correct behavior.
   - Large-argument cancellation near `π/2 + kπ` is the dual of `sin`'s
     near `kπ` problem.
5. **Flavors.** Same `strict` / `correctly_rounded` strategies as `sin`.
   The DD-augmented polynomial on `cos` needs its own Remez fit — the
   `1 − r²/2 + ...` shape is different and the tail coefficients differ.
6. **Relatives.** See `sin`; `cospi` pi-scaled variant; `sec = 1/cos`.
7. **State of the art.** fdlibm, CORE-MATH `cos()`, CUDA `__cos`, same
   characteristics as `sin`.

### tan(x)

1. **Source.** fdlibm `__kernel_tan` (Sun 1993) — Chen's polynomial from
   the 1993 MIT-SML archive; Muller *Handbook* ch. 11.4.
2. **Domain.** `ℝ ∖ {π/2 + kπ : k ∈ ℤ}` (but returned as very large f64
   values near the pole — no NaN, no infinity). **Codomain.** `ℝ`.
3. **Violations.**
   - `x = NaN` → `NaN`; `x = ±∞` → `NaN` + `FE_INVALID`.
   - `tan(π/2)` in f64: the input isn't exactly the singularity, so the
     output is `1.633...e+16`, not infinity.
   - Even/odd quadrant switching: `tan` reuses `__kernel_tan` with a
     quadrant flag deciding whether to return `P(r²)·r` or `-1/(P(r²)·r)`.
4. **Edge cases.**
   - Near singularity: `r → π/2` makes `1/tan_core(r)` underflow-sensitive.
     fdlibm splits the polynomial as `r + r³·T(r²)` so the reciprocal
     branch handles cancellation cleanly.
   - Large argument: same reduction strategy as `sin`; `tan` is odd so
     sign of `r_hi + r_lo` is preserved.
5. **Flavors.**
   - **Standard**: fdlibm kernel — polynomial core of degree 13 in `r²`
     with tail term for `r_lo` precision. ≤ 1 ULP except near poles.
   - **Correctly rounded**: CORE-MATH `tan()` 2023 — requires careful
     handling of the `1/tan` branch via DD reciprocal.
6. **Relatives.** `cot = 1/tan` (or equivalently, swapped-quadrant tan).
   `atan` inverse. `tanh` hyperbolic analog.
7. **State of the art.**
   - fdlibm: ~1 ULP with caveat near the quadrant boundary.
   - CORE-MATH `tan()`: correctly rounded in f64; uses sincos-then-divide
     with full DD precision.
   - CUDA: `__tanf` 2-ULP intrinsic; `tan` software ~1 ULP.
   - Note: many libms have worst-case error up to 2 ULPs near poles —
     CORE-MATH is the only one guaranteeing ≤ 1 ULP there.

### cot(x), sec(x), csc(x)

1. **Source.** Reciprocals — no separate historical development. Implemented
   as `1/tan`, `1/cos`, `1/sin` in Muller's taxonomy; fdlibm doesn't ship
   them (a POSIX-libm gap).
2. **Domain.**
   - `cot`: `ℝ ∖ {kπ}` — poles at each integer multiple of π.
   - `sec`: `ℝ ∖ {π/2 + kπ}`.
   - `csc`: `ℝ ∖ {kπ}`.
3. **Codomain.** `cot, csc ∈ ℝ ∖ (-1, 1)`, `sec ∈ ℝ ∖ (-1, 1)`.
4. **Violations.** NaN/∞ propagation. Near-pole: return very large f64, not ∞.
5. **Edge cases.**
   - `cot(0) = +∞` mathematically; in f64 we return `±∞` with the input's sign
     (no signaling) — but `cot(-0) = -∞` per the odd-symmetry rule.
   - `sec(π/2)` in f64 gives `1.633...e+16`, not ∞, because the input
     isn't exact.
   - The reciprocal amplifies the forward function's rounding error — if
     `cos(x) ≈ 2⁻⁵²`, then `sec(x)` has **enormous** relative error from a
     single ULP wobble in `cos(x)`. Near poles, compute via `cos_lo`
     correction (i.e. DD reciprocal) or advertise reduced accuracy.
6. **Flavors.**
   - **Direct**: `1/sin(x)` etc. Cheap but the near-pole relative error is
     up to 2 ULPs.
   - **Fused**: reconstruct from `(r_hi, r_lo)` without round-tripping
     through the forward function — share the range reduction pass with
     `sin`/`cos`/`tan` and replace only the reconstruction formula. This
     is the **shared-pass** payoff for TRIG-4.
7. **State of the art.**
   - Julia, MATLAB, Mathematica: direct `1/tan` etc.
   - R `base::tan(x)^-1` — same.
   - No major libm ships a fused implementation; we have a design
     opportunity to be better.

### sincos(x) — fused pair

1. **Source.** Kahan's 1987 report on simultaneous sin/cos; fdlibm
   `__sincos`. GCC `__builtin_sincos`; Intel SVML `__svml_sincos`.
2. **Domain/codomain.** `ℝ → [-1, 1]²`.
3. **Violations.** Same as individual functions.
4. **Edge cases.** `sincos(0) = (0, 1)`, `sincos(-0) = (-0, 1)`.
5. **Flavors.** Always pair; the shared pass is the entire reason for this
   function. Payne-Hanek reduction, polynomial core evaluated once per
   `r²`, quadrant fixup produces both outputs — ~1.5× the work of one.
6. **Relatives.** `sincospi`, `sinhcosh`.
7. **State of the art.**
   - fdlibm provides `__sincos`; glibc ships the C interface as `sincos(3)`.
   - Intel SVML: SIMD-batched `sincos` is ~1.3× one function (nearly free).
   - CUDA `__sincosf`: one SFU instruction, 2-ULP intrinsic.

---

## Family 2 — Pi-Scaled (TRIG-16)

### sinpi(x), cospi(x), tanpi(x)

1. **Source.** Formalized by W. Kahan as the solution to large-argument
   reduction: if the caller's angle is *already* a multiple of π (e.g.
   phase accumulated in units of turns), computing `sin(π·x)` in radians
   destroys precision. Kahan "Branch Cuts for Complex Elementary Functions"
   (1987) introduces pi-scaled variants.
2. **Domain.** `ℝ` for `sinpi`, `cospi`; `ℝ ∖ {k + 1/2 : k ∈ ℤ}` for `tanpi`.
3. **Codomain.** `[-1, 1]` / `[-1, 1]` / `ℝ`.
4. **Violations.** Same NaN/∞ rules as forward trig.
5. **Edge cases.**
   - `sinpi(n) = ±0` for integer `n` **exactly**. This is the reason for
     `sinpi` — there is no rounding error to introduce.
   - `cospi(n) = ±1` for integer `n` exactly (even/odd quadrant).
   - `sinpi(n + 1/2) = ±1` for integer `n` **exactly**.
   - `tanpi(n + 1/2)` diverges; return very large f64 with sign matching
     quadrant.
6. **Flavors.**
   - **Standard**: reduce `x mod 2` (exact for finite f64 since `2` is a
     power of 2), then call the sin/cos kernel on `π · r` where `r ∈
     [-1/2, 1/2]`. No Payne-Hanek ever needed.
   - **Dual-use**: share range reduction with radian `sin`/`cos` via a
     single `reduce_to_quadrant` primitive that accepts `unit ∈
     {radians, pi, turns, degrees, gradians}`.
7. **State of the art.**
   - R `sinpi()` / `cospi()` / `tanpi()` in base since R 3.0 (2013).
   - CUDA `sinpi` / `cospi` / `sinpif` / `cospif` since CUDA 4.0.
   - Julia Base.sinpi since 0.5 (2016).
   - IEEE 754-2019 mandates these as **recommended operations**.
   - C23 adds `sinpi`, `cospi`, `tanpi` via `<math.h>` but most libms lag.

### asinpi(x), acospi(x), atanpi(x), atan2pi(y, x)

1. **Source.** IEEE 754-2019 §9.2 (recommended operations, optional to
   implement but specified); C23 Annex F.10.1.* (adopted as part of the
   revised math library, still rolling out). Muller *Handbook* 2nd ed.
   ch. 11 discusses pi-scaled inverse design.
2. **Domain/codomain.** Same input domains as `asin`/`acos`/`atan`/`atan2`;
   outputs scaled by `1/π` relative to the radian counterparts.
   `asinpi: [-1, 1] → [-1/2, 1/2]`. `acospi: [-1, 1] → [0, 1]`.
   `atanpi: ℝ → (-1/2, 1/2)`. `atan2pi: ℝ² ∖ {(0,0)} → (-1, 1]`.
3. **IEEE 754-2019 stance.** All four are in §9.2's recommended table
   alongside `sinpi`/`cospi`/`tanpi`. They are **optional** for a
   conforming implementation — a libm may ship them or not — but if
   shipped, they must obey the §9.2 semantics. C23 adopts them into
   the standard math library. **Practical meaning for tambear**:
   producing them is a differentiator; none of glibc, fdlibm, Boost.Math,
   or MATLAB's baseline ships the inverse pi-scaled set.
4. **Edge cases.** Rational-π answers come out rational.
   - `asinpi(0) = 0`, `asinpi(±1) = ±1/2` **exactly**.
   - `asinpi(1/2) = 1/6` **exactly** (corresponds to `asin(1/2) = π/6`).
   - `acospi(1) = 0`, `acospi(-1) = 1`, `acospi(0) = 1/2` exactly.
   - `atanpi(0) = 0`, `atanpi(±∞) = ±1/2` exactly, `atanpi(1) = 1/4` exact.
   - `atan2pi(0, -1) = 1`, `atan2pi(1, 0) = 1/2` exactly.
5. **Implementation — CRITICAL design point.**
   **Do NOT implement as `radian_result / π`.** The post-multiply by
   `1/π` converts a radian result (which is a transcendental mixed with
   accumulated rounding error) into a half-turn result via a final
   division by an irrational constant. For inputs where the true answer
   is close to `0` or `±1/2` — which is most of the interesting
   domain — the final `/π` divides a small-or-near-boundary value and
   eats up to 3 ULPs of precision that the original radian computation
   had.

   The correct implementation **fits a polynomial directly for the
   half-turn output**:
   - `asinpi(x)`: Remez fit of `asin(x)/π` on `[-1, 1]` with the same
     region-splitting as `asin` (small / medium / near-unity half-angle).
     Coefficients are different from the radian polynomial — they bake
     the `1/π` into the constants.
   - `acospi(x)`: fit `acos(x)/π` directly, or use the identity
     `acospi(x) = 1/2 − asinpi(x)` where `1/2` is f64-exact. Both paths
     preserve the exactness of `acospi(0) = 1/2`.
   - `atanpi(x)`: fit `atan(x)/π` directly with the same fdlibm-style
     table partition, with `1/π` in the table constants.
   - `atan2pi(y, x)`: reuse `atanpi` for the interior; the quadrant-offset
     constants are exact rationals in half-turns (`0, 1/2, -1/2, 1, -1`).

   This is the same pattern as `sinpi(x)`: naive `sin(π·x)` loses bits
   because `π·x` introduces a transcendental multiply; direct
   polynomial fit in the turn domain keeps everything in exact
   arithmetic until the polynomial evaluation itself.

   **Consequence for tambear**: `asinpi`, `acospi`, `atanpi`, `atan2pi`
   get their own first-class recipes with their own coefficient tables.
   They are NOT wrappers that call their radian counterparts and divide.
6. **Flavors.**
   - **Standard**: direct polynomial fit per above.
   - **Correctly rounded**: DD polynomial fit with the same direct
     approach. CORE-MATH's 2024 `asinpi` uses this.
7. **Relatives.** Inverse of `sinpi`/`cospi`/`tanpi`; pi-scaled analog
   of `asin`/`acos`/`atan`/`atan2`. Shares region-splitting pattern
   with radian inverses but not coefficients or `AtanCore` / `AsinCore`
   intermediates — those are radian-specific.
8. **State of the art.**
   - **Julia**: ships `asinpi`, `acospi`, `atanpi` since 1.0 (2018) in
     `Base.Math`. Confirmed via julia/base/special/trig.jl.
     `atan2pi` via `atand / 180` idiom, not first-class.
   - **CORE-MATH**: `asinpi` and `acospi` since 2024 at correctly-rounded
     f64 via direct DD polynomial fit.
   - **CUDA libdevice**: `atan2pi`, `atan2pif`, `asinpi`, `acospi`,
     `atanpi` documented in CUDA Math API; 2-ULP intrinsics / 1-ULP
     software.
   - **R, glibc, fdlibm, MATLAB, Boost.Math**: none ship the inverse
     pi-scaled set. Users work around with `asin(x)/pi` — which has
     exactly the precision loss described above.
   - **IEEE 754-2019** recommends; C23 standardizes.

---

## Family 3 — Inverse Core

### atan(x)

1. **Source.** Mādhava's series (~1400 CE); Gregory (1671); Machin (1706).
   Modern: fdlibm `atan.c` via Lindemann's tangent addition identity.
2. **Domain.** `ℝ`. **Codomain.** `(-π/2, π/2)`.
3. **Violations.** `atan(NaN) = NaN`; `atan(±∞) = ±π/2` **exactly** (per
   IEEE, no `FE_INVALID`).
4. **Edge cases.**
   - `atan(0) = 0`, `atan(-0) = -0`.
   - `atan(x)` for `|x| > 1`: use identity `atan(x) = π/2 − atan(1/x)` with
     appropriate sign to keep argument small.
   - Near `|x| = 1`: fdlibm uses lookup table splitting the domain into
     `[-∞, -39/16] ∪ [-39/16, -19/16] ∪ [-19/16, -11/16] ∪ [-11/16, 11/16]
     ∪ [11/16, 19/16] ∪ [19/16, 39/16] ∪ [39/16, ∞]`.
   - Hard cases: `atan(0.4375)` sits right on a boundary constant; TMD
     worst case is the usual near-midpoint suspects.
5. **Flavors.**
   - **Standard**: fdlibm lookup-based, ≤ 1 ULP.
   - **Correctly rounded**: CORE-MATH `atan()` — expanded table, DD poly
     evaluation.
6. **Relatives.** `atan2` for two-argument / full-quadrant; `acot = π/2 - atan`.
7. **State of the art.**
   - fdlibm: lookup + polynomial, ~1 ULP.
   - CORE-MATH: correctly rounded.
   - Intel SVML: vectorized, 1-ULP.
   - CUDA `atanf`: 2 ULP intrinsic, `atan` software 1 ULP.

### atan2(y, x)

1. **Source.** Fortran II (1958); Bunch-Kaufman 1967 for four-quadrant
   formulation. IEEE 754-2008 §9.2 standardizes edge cases.
2. **Domain.** `(y, x) ∈ ℝ²`. **Codomain.** `(-π, π]`.
3. **Violations / Edge cases — IEEE 754-2019 Table 9.1 enumerates ≥ 20 cases.**
   - `atan2(±0, +x)` for `x > 0` → `±0`.
   - `atan2(±0, -x)` for `x > 0` → `±π`.
   - `atan2(+y, ±0)` for `y > 0` → `+π/2`.
   - `atan2(-y, ±0)` for `y > 0` → `-π/2`.
   - `atan2(±∞, +∞)` → `±π/4`.
   - `atan2(±∞, -∞)` → `±3π/4`.
   - `atan2(±∞, finite)` → `±π/2`.
   - `atan2(finite, +∞)` → `±0` (sign of y).
   - `atan2(finite, -∞)` → `±π` (sign of y).
   - `atan2(±0, ±0)` returns a signed zero or ±π per the sign pattern.
   - `atan2(NaN, *)` → `NaN`; `atan2(*, NaN)` → `NaN`.
4. **Flavors.**
   - **Standard**: reduce to `atan(y/x)`, branch on sign of x for quadrant.
   - **Correctly rounded**: CORE-MATH — requires DD division then
     correctly-rounded `atan` on the DD quotient.
5. **Relatives.** `atan2pi`, the inverse for 2D polar conversion.
6. **State of the art.**
   - fdlibm `e_atan2.c`: the canonical reference for the table-of-cases.
   - CUDA `atan2f` 2 ULP / `atan2` 1 ULP.
   - **Notable bug in older MATLAB 2012**: `atan2(0, -0) = 0` instead of
     the IEEE-mandated `π`; fixed in later versions.

### asin(x)

1. **Source.** Euler "Introductio in analysin infinitorum" (1748);
   modern computing: fdlibm `s_asin.c`, Markstein 2000.
2. **Domain.** `[-1, 1]`. **Codomain.** `[-π/2, π/2]`.
3. **Violations.**
   - `|x| > 1` → `NaN` + `FE_INVALID`.
   - `NaN` → `NaN`.
4. **Edge cases.**
   - `asin(0) = +0`, `asin(-0) = -0`.
   - `asin(±1) = ±π/2` exactly.
   - **Near |x| = 1** is the hard case: compute via `asin(x) = π/2 −
     2·asin(√((1-x)/2))` which shifts evaluation away from the
     `√(1-x²)` cancellation disaster zone. fdlibm threshold is `|x| ≥ 0.975`.
   - Small `|x|` uses direct Taylor-ish `asin(x) = x·(1 + R(x²))`.
5. **Flavors.**
   - **Standard**: fdlibm three-region strategy (small / medium / near-unity).
   - **Correctly rounded**: Muller's "asin with near-unity cancellation
     eliminated" 2018 — uses Hermite-Padé approximant on a split domain.
6. **Relatives.** `acos = π/2 − asin`, `asec = asin(1/x)`.
7. **State of the art.**
   - fdlibm: 1-2 ULP typical, up to 3 ULP near `|x| = 1` in older versions.
   - CORE-MATH `asin()`: correctly rounded with special handling at both
     endpoints.
   - **Known issue**: naive `asin(x) = atan(x / sqrt(1 - x²))` loses all
     precision near `|x| = 1` — every good libm uses the half-angle trick.

### acos(x)

1. **Source.** Same lineage as `asin`. fdlibm `e_acos.c`.
2. **Domain/codomain.** `[-1, 1] → [0, π]`.
3. **Violations / edge cases.**
   - `|x| > 1` → `NaN` + `FE_INVALID`.
   - `acos(1) = +0` exactly.
   - `acos(-1) = π` exactly (the stored f64 value of π).
   - `acos(0) = π/2`.
   - Near `x = 1`: similar half-angle trick to `asin` — use `acos(x) =
     2·asin(√((1-x)/2))`. Near `x = -1`: use `acos(x) = π − 2·asin(√((1+x)/2))`.
4. **Flavors.** Standard / correctly-rounded same story as `asin`.
5. **Relatives.** `acos = π/2 − asin` (up to the near-unity split).
   `asec(x) = acos(1/x)`.
6. **State of the art.** Same implementations as `asin`.

### acot(x), asec(x), acsc(x)

1. **Source.** No unique paper lineage — defined as reciprocal-argument
   inverse of `atan`/`acos`/`asin`.
2. **Domain.**
   - `acot`: `ℝ`. **Codomain**: `(0, π)` — the "open range" convention,
     matching Mathematica. MATLAB/Wolfram vs. Abramowitz-Stegun differ on
     whether `acot(0)` is `+π/2` (MATLAB, `π/2 − atan(x)` convention) or
     undefined (pure-inverse). **We will default to the MATLAB convention**
     because it's continuous at 0 and matches the Wolfram standard.
   - `asec`: `(-∞, -1] ∪ [1, ∞)`. **Codomain**: `[0, π/2) ∪ (π/2, π]`.
   - `acsc`: `(-∞, -1] ∪ [1, ∞)`. **Codomain**: `[-π/2, 0) ∪ (0, π/2]`.
3. **Violations.**
   - `acot` — defined on all of `ℝ`; NaN propagates.
   - `asec` / `acsc` with `|x| < 1` → `NaN` + `FE_INVALID`.
4. **Edge cases.**
   - `acot(0) = π/2` (MATLAB convention) vs. `0` (Mathematica alternate).
     **Parameterize via `using(acot_convention = "matlab" | "pure")`.**
   - `asec(1) = 0`, `asec(-1) = π`.
   - `acsc(1) = π/2`, `acsc(-1) = -π/2`.
5. **Flavors.**
   - **Standard**: `acot(x) = atan(1/x)` with sign adjust; `asec(x) =
     acos(1/x)`; `acsc(x) = asin(1/x)`.
   - **Fused**: compute without round-tripping through a divide — the
     `1/x` division creates up to 1 ULP of input error that propagates
     through the inverse. Instead, fold `1/x` into the core by evaluating
     the inverse on a transformed interval.
6. **Relatives.** All three are derivatives of `asin`/`acos`/`atan` and
   can share the same core polynomial.
7. **State of the art.**
   - MATLAB: `acot(x) = atan(1/x)` with the continuous extension.
   - Mathematica: pure-inverse convention.
   - R: `1/tan()` etc. — does not ship first-class.
   - Julia: `acot(x) = atan(1/x)`.

---

## Family 4 — Hyperbolic Forward

### sinh(x), cosh(x), tanh(x)

1. **Source.** Introduced by Vincenzo Riccati (1757); modern libm: fdlibm
   `s_sinh.c`, `s_cosh.c`, `s_tanh.c`. Markstein 2000 ch. 8.
2. **Domains.**
   - `sinh`: `ℝ → ℝ`.
   - `cosh`: `ℝ → [1, ∞)`.
   - `tanh`: `ℝ → (-1, 1)`.
3. **Violations.**
   - NaN propagates.
   - `sinh(±∞) = ±∞`.
   - `cosh(±∞) = +∞`.
   - `tanh(±∞) = ±1` exactly.
   - Overflow: `sinh(x)` and `cosh(x)` overflow at `|x| > 710.48` (f64);
     `tanh` never overflows.
4. **Edge cases.**
   - `sinh(x)` for small `x` ≈ `x` — must preserve input precision. fdlibm
     uses `sinh(x) = (expm1(|x|) + expm1(|x|)/(expm1(|x|)+1))/2` to avoid
     catastrophic cancellation at small x. The `expm1` shared intermediate
     **is** the shared-pass opportunity for TRIG-4.
   - `cosh(x)` for small `x` ≈ `1 + x²/2` — requires `expm1` or explicit
     polynomial to avoid losing the quadratic tail.
   - `tanh(x)` for small `x` ≈ `x − x³/3` — use `expm1` to preserve.
   - Large `|x|` (approaching overflow): scale, compute as `sign(x)·exp(|x|)/2`.
5. **Flavors.**
   - **Standard**: fdlibm three-region split (tiny, regular, large).
   - **Shared-pass**: compute `expm1(x)` and `expm1(-x)` once, reuse across
     `sinh`, `cosh`, `tanh`. This is **the** shared-pass primitive for
     TRIG-4 (see §Shared-Pass Analysis below).
6. **State of the art.**
   - fdlibm: 1-2 ULP.
   - CORE-MATH `tanh()` 2023: correctly rounded, DD polynomial on
     `[0, ln(2)/2]` with exact expm1 as intermediate.
   - Intel SVML: vectorized, 1 ULP.

### coth(x), sech(x), csch(x)

1. **Source.** Reciprocals; no major libm ships them.
2. **Domain.**
   - `coth`: `ℝ ∖ {0} → ℝ ∖ (-1, 1)`.
   - `sech`: `ℝ → (0, 1]`.
   - `csch`: `ℝ ∖ {0} → ℝ ∖ {0}`.
3. **Violations.** `coth(0) = ±∞` (sign of input); `csch(0) = ±∞`.
4. **Edge cases.**
   - `coth(x)` for small `x`: `coth(x) ≈ 1/x + x/3 − x³/45`. Direct
     `1/tanh(x)` is fine here because `tanh(x) ≈ x` preserves precision.
   - `sech(x)` for small `x`: `sech(x) ≈ 1 − x²/2` — reciprocal of `cosh`
     is cheap and accurate.
   - `csch(x)` for small `x`: `csch(x) ≈ 1/x − x/6 + 7x³/360`.
5. **Flavors.** Direct reciprocal (standard) or fused with shared `expm1`
   intermediate (better).
6. **State of the art.** Julia, Mathematica, R, Python `math.cosh`/`math.sinh`
   and users do `1/·`. No first-class libm implementation.

---

## Family 5 — Hyperbolic Inverse

### asinh(x)

1. **Source.** Identity `asinh(x) = ln(x + √(1 + x²))`; Cody-Waite 1980.
   fdlibm `s_asinh.c`.
2. **Domain/codomain.** `ℝ → ℝ`.
3. **Violations.** NaN / ∞ propagate; `asinh(±∞) = ±∞`.
4. **Edge cases.**
   - `asinh(x)` for small `x` via `ln(x + √(1+x²))` loses precision —
     fdlibm splits: for `|x| < 2⁻²⁸`, return `x`; for `|x| < 1`, use
     `sign(x)·log1p(|x| + x²/(1+√(1+x²)))`; for medium `|x|`, direct
     `log(2·|x| + 1/(√(x²+1) + x))`; for large `|x|`, `sign(x)·(log(|x|) + ln(2))`.
5. **Flavors.** Standard (fdlibm piecewise) vs. correctly-rounded (CORE-MATH).
6. **Relatives.** `asinh(x) = ln(x + √(x²+1))` — closed form.
7. **State of the art.** fdlibm with log1p refinement at small x.

### acosh(x)

1. **Source.** `acosh(x) = ln(x + √(x²−1))`. fdlibm `e_acosh.c`.
2. **Domain/codomain.** `[1, ∞) → [0, ∞)`.
3. **Violations.** `x < 1` → `NaN` + `FE_INVALID`; `acosh(1) = 0` exactly.
4. **Edge cases.**
   - Near `x = 1`: catastrophic cancellation in `√(x²-1)`. Rewrite as
     `acosh(1 + t) = log1p(t + √(2t + t²))` with `t = x − 1`. fdlibm
     threshold: `x < 1.5` uses the log1p form.
   - Large `x`: `acosh(x) ≈ log(2·x)` saturates; fdlibm uses `log(x) + ln(2)`.
5. **State of the art.** fdlibm piecewise; CORE-MATH 2024.

### atanh(x)

1. **Source.** `atanh(x) = (1/2)·ln((1+x)/(1-x))`. fdlibm `s_atanh.c`.
2. **Domain/codomain.** `(-1, 1) → ℝ`.
3. **Violations.**
   - `|x| > 1` → `NaN` + `FE_INVALID`.
   - `x = ±1` → `±∞` + `FE_DIVBYZERO`.
4. **Edge cases.**
   - Small `x`: direct formula loses precision — use `atanh(x) = (1/2)·log1p(2x/(1−x))`.
   - Near `|x| = 1`: `log` diverges; IEEE mandates `±∞` with divide-by-zero
     flag.
5. **Flavors.** log1p-based standard; correctly-rounded via DD log1p.
6. **State of the art.** fdlibm with log1p; CORE-MATH.

### acoth(x), asech(x), acsch(x)

1. **Source.** Inverse reciprocal functions; implemented as `atanh(1/x)` etc.
2. **Domain.**
   - `acoth`: `(-∞, -1) ∪ (1, ∞) → ℝ ∖ {0}`.
   - `asech`: `(0, 1] → [0, ∞)`.
   - `acsch`: `ℝ ∖ {0} → ℝ ∖ {0}`.
3. **Violations.**
   - `acoth` with `|x| ≤ 1` → `NaN` (or `±∞` at `|x| = 1`).
   - `asech` with `x ≤ 0` or `x > 1` → `NaN`; `asech(0) = +∞`.
4. **Edge cases.** Same pattern as inverse trig reciprocals — round-trip
   through `1/x` has up to 1 ULP of error; fused variant shares the
   `log1p` intermediate with `atanh`/`asinh`/`acosh`.
5. **Flavors.** Direct (`atanh(1/x)`) vs. fused.
6. **State of the art.** Julia, Mathematica, R via `1/·`; no major libm
   first-class impl.

---

## Family 6 — Fused / Non-Standard

### sinhcosh(x) — fused pair

1. **Source.** Analog of `sincos`. No published libm ships this; we should.
2. **Codomain.** `(sinh(x), cosh(x)) ∈ ℝ × [1, ∞)`.
3. **Rationale.** `sinh` and `cosh` both depend on `expm1(|x|)`; computing
   them together saves the polynomial evaluation (biggest cost) at essentially
   no extra reconstruction work.
4. **Shared-pass gain.** ~45% over two separate calls (estimated from
   `exp/expm1` being ~60% of each function's cost).

### expm1_coshm1(x) — fused ultra-small

1. **Source.** Workaround for `cosh(x) − 1 = 2·sinh²(x/2)` cancellation.
2. **Rationale.** When a caller needs `cosh(x) − 1` for small `x` (e.g.
   numerical derivative of `cosh`), naive subtraction loses all precision.
   Provide `coshm1(x) = 2·sinh²(x/2)` as a first-class recipe alongside
   `expm1`. The IEEE committee has discussed adding `coshm1` to C2x.

### cis(x) — complex exp

1. **Source.** Euler's identity `cis(x) = cos(x) + i·sin(x)`. Julia
   provides `cis(x)` as a first-class function since 0.6.
2. **Rationale.** Identical to `sincos` but returns `Complex<f64>`.
   Including for completeness so pathmaker knows the IDE needs a
   `complex` output binding.

### haversine / vers / coversine / excosecant

1. **Source.** Pre-calculator navigation tables (James Inman, 1835);
   haversine formula for great-circle distance (de Mendoza y Ríos, 1795).
2. **Definitions.**
   - `versin(x) = 1 − cos(x) = 2·sin²(x/2)`.
   - `coversin(x) = 1 − sin(x)`.
   - `haversin(x) = versin(x)/2 = sin²(x/2)`.
   - `exsec(x) = sec(x) − 1`.
   - `excsc(x) = csc(x) − 1`.
3. **Rationale for including.** `haversin` in geospatial distance calculations
   is the only case where `1 − cos(x)` can't be replaced with `cos(x)` —
   and its accurate evaluation is **exactly** the `sin²(x/2)` trick that
   tambear's shared-pass primitive supports natively. Anti-YAGNI: add now.
4. **State of the art.** Boost.Math ships them; no major libm.

### gudermannian `gd(x)` and inverse `gd⁻¹(x)` — the circular↔hyperbolic bridge

1. **Source.** Named for Christoph Gudermann (1798–1852) who studied
   it as a special function relating hyperbolic and circular trig
   without needing complex numbers. First tabulated by Lambert 1768.
2. **Definitions** (multiple equivalent forms — all exact identities):
   - `gd(x) = 2·atan(tanh(x/2))`
   - `gd(x) = atan(sinh(x))`
   - `gd(x) = 2·atan(e^x) − π/2`
   - `gd(x) = ∫₀ˣ sech(t) dt` — the integral of sech from 0.
   - Inverse: `gd⁻¹(y) = ln(tan(π/4 + y/2)) = atanh(sin(y)) = 2·atanh(tan(y/2))`.
3. **Domain/codomain.**
   - `gd`: `ℝ → (-π/2, π/2)`. Monotonically increasing. Odd.
   - `gd⁻¹`: `(-π/2, π/2) → ℝ`.
4. **Violations.**
   - `gd(NaN) = NaN`; `gd(±∞) = ±π/2` exactly.
   - `gd⁻¹(NaN) = NaN`; `gd⁻¹(±π/2) = ±∞` + `FE_DIVBYZERO`.
   - `gd⁻¹(|y| > π/2)` → NaN + `FE_INVALID`.
5. **Edge cases.**
   - `gd(0) = 0`, `gd⁻¹(0) = 0` exactly.
   - Small `x`: `gd(x) ≈ x − x³/6 + x⁵/24 − ...` — must preserve input
     at tiny scales; naive `2·atan(tanh(x/2))` loses bits for `|x| <
     2⁻²⁷` because the chained near-identity functions each add
     rounding error.
   - Large `|x|`: saturates toward `±π/2`. For `|x| > ln(2/ε) ≈ 37.4`
     in f64, `gd(x) = ±π/2` within half a ULP; short-circuit return
     `copysign(π/2, x)`.
6. **Rationale for including.**
   - **Real applied use**: Mercator map projection. The `y`-coordinate
     of a Mercator projection at latitude `φ` is `gd⁻¹(φ)`; inverse
     Mercator recovers latitude from `y` via `gd(y)`. This is the
     defining integral of the projection, not just a convenience.
   - **Real applied use**: hyperbolic geometry. The gudermannian
     relates the "hyperbolic angle" (arc length along a unit hyperbola)
     to a circular angle without complexification. Appears in special
     relativity's rapidity-to-velocity conversion: `v/c = tanh(rapidity)
     = sin(gd(rapidity))`.
   - **Real applied use**: pendulum motion. The nonlinear pendulum's
     exact period involves elliptic integrals that reduce to
     gudermannian terms in the small-amplitude limit.
   - **Structural reason**: it's the single named function that
     witnesses the circular↔hyperbolic isomorphism in the real domain.
     Having it in the catalog makes the symmetry between families
     explicit rather than implicit.
7. **Flavors.**
   - **Standard**: `gd(x) = atan(sinh(x))` form — reuses the `sinh`
     and `atan` primitives with their own precision strategies. Error
     budget compounds (~2-3 ULPs worst).
   - **Direct**: for correctly-rounded, fit a polynomial to `gd(x)` on
     `[-ε, ε]` for small `|x|`, use the `2·atan(e^x) − π/2` form for
     medium `|x|` (single exp, single atan, one subtraction), and
     short-circuit to `±π/2` for large `|x|`. This is the CORE-MATH-
     style approach for transcendentals composed of simpler parts.
   - **Inverse**: `gd⁻¹(y)` via `atanh(sin(y))` form. Near `|y| =
     π/2` reduces to the `atanh` near-unity cancellation case — falls
     naturally into the `complementary_arg_transform` meta-primitive.
8. **State of the art.**
   - **Boost.Math**: ships `gudermannian` and `inverse_gudermannian`
     in the `<boost/math/special_functions/gudermannian.hpp>` header.
     The only C++ math library that does.
   - **Julia**: not in Base; exists in `SpecialFunctions.jl` as
     `gd(x)` / via `gd = atan(sinh(x))` idiom.
   - **Mathematica / Maple / MATLAB**: `Gudermannian[x]` in
     Mathematica; MATLAB lacks it (users compute via the identity).
   - **Python**: scipy does not ship it; mpmath does at arbitrary
     precision via `mp.gudermannian(x)`.
   - **fdlibm, glibc, CORE-MATH, Intel SVML**: none ship it.

   **Practical meaning for tambear**: shipping `gd` and `gd⁻¹` as
   first-class first-principles recipes makes tambear one of 2-3 math
   libraries in existence with full gudermannian coverage, and makes
   us categorically ahead for Mercator projection and relativity
   workloads.

---

## Shared-Pass Analysis (TRIG-4 preview)

The catalog entries above repeatedly cite "shared intermediate" as a
design lever. This section extracts the **TrigSharedIntermediate** design.

### Forward trig (radian)

| Intermediate | Consumers | Savings if shared |
|---|---|---|
| `quadrant_reduce(x)` → `(q, r_hi, r_lo)` | sin, cos, tan, cot, sec, csc, sincos | ~60% of a single call |
| `r² = r_hi * r_hi` | sin core, cos core, tan core | ~15% (one mul) |
| Sin core eval `r + r³·P(r²)` | sin, cos (via quadrant swap), sincos | ~40% |
| Cos core eval `1 - r²/2 + r⁴·Q(r²)` | cos, sin (via quadrant swap), sincos | ~40% |

**Verdict**: register `TrigQuadrantReduction` as a TamSession
IntermediateTag. Compatibility: the reduction is unit-dependent (radian
vs. pi vs. turn). Separate tags per unit.

### Forward hyperbolic

| Intermediate | Consumers | Savings if shared |
|---|---|---|
| `expm1(\|x\|)`, `expm1(-\|x\|)` | sinh, cosh, tanh, coth, sech, csch, sinhcosh | ~55% of each |
| `(e^x − 1, e^x)` pair | sinh, cosh, tanh | ~60% |

**Verdict**: `HyperbolicExpm1Pair` is the single most valuable shared
intermediate in the entire catalog — the expm1 cost dominates and **all 6
hyperbolics** consume it.

### Inverse trig

| Intermediate | Consumers | Savings if shared |
|---|---|---|
| `atan_core(y, x)` returning table-index + poly result | atan, acot, atan2, atan2pi | ~80% |
| `asin_core` on half-angle split | asin, acos, asec, acsc | ~70% |

**Verdict**: two cores — `atan_core` and `asin_core` — each a primitive.
Inverse reciprocals then become one-liners.

### Inverse hyperbolic

| Intermediate | Consumers | Savings if shared |
|---|---|---|
| `log1p` on transformed argument | asinh, acosh, atanh, acsch | ~65% |

**Verdict**: use the existing `log1p` primitive; define a
`InverseHyperbolicTransform` helper that produces the log1p argument for
each function.

### The five shared intermediates

Summary of IntermediateTags tambear should register for the trig family:

1. `TrigQuadrantReduction{unit}` — `(q, r_hi, r_lo)` from `reduce_mod(π/2 or 1/2)`.
2. `TrigSinCosCore` — paired `(sin_core(r), cos_core(r))` on reduced range.
3. `HyperbolicExpm1Pair` — `(expm1(|x|), expm1(-|x|))` or equivalently `(e^|x|, e^-|x|)`.
4. `AtanCore{table_region}` — `atan` table index + polynomial result.
5. `AsinCore{region}` — `asin` polynomial on either direct or half-angle.

---

## Convergence Check — Near-Unity Cancellation Rhyme

Running the structural-rhyme practice on the inverse-function entries above:

| Function | Cancellation site | Fix |
|---|---|---|
| `asin(x)` near `\|x\| = 1` | `√(1 − x²)` loses bits | Half-angle: `asin(x) = π/2 − 2·asin(√((1−x)/2))` |
| `acos(x)` near `\|x\| = 1` | `√(1 − x²)` | Half-angle via `2·asin(√((1±x)/2))` |
| `acosh(x)` near `x = 1` | `√(x² − 1)` | log1p form: `log1p(t + √(2t + t²))`, `t = x−1` |
| `atanh(x)` near `\|x\| = 1` | `(1 − x)` in denominator | Divergent (IEEE: `±∞`) |
| `atan2(y, x)` at axes | `y/x` divide | Table of cases (IEEE §9.2) |
| `asech(x)` near `x = 1` | `√(1 − x²)/x` | Reduces to `asin` half-angle |

**Rhyme**: every inverse trig / inverse hyperbolic function has a
**near-unity cancellation zone**, and the fix in every case is to rewrite
in terms of the **complementary argument** (`1 − x`, `1 + x`, `t = x − 1`)
so that the subtraction becomes an addition that preserves the low-order
bits. This is the same structural pattern as `log1p` / `expm1` / `sinpi`
**at different scales**.

**Implication for tambear**: define a meta-primitive
`complementary_arg_transform` that emits `(t, sign, branch)` for any of
these six inverse functions, feeding the same post-transform core.

---

## Per-Function Checklists

These checklists tell pathmaker what has to exist in `.spec.toml` plus
`.rs` for a function to be declared shipped. The full set per TRIG-12
maps one checklist per entry; below is the exemplar for `sin`:

```
sin (radian forward core)
  [x] Spec: crates/tambear/src/recipes/libm/sin.spec.toml (done)
  [x] Impl: crates/tambear/src/recipes/libm/sin.rs (done, commit 0bbae82)
  [x] Polynomial coefficients Remez-refit in 80-digit mpmath
  [x] Strict lowering: Cody-Waite + Payne-Hanek
  [ ] Compensated lowering: DD range reduction
  [ ] Correctly-rounded lowering: DD poly + Ziv refinement
  [x] Special cases: NaN, ±∞, ±0 (done)
  [ ] Adversarial suite: hard cases from Muller Tbl 11.5
  [ ] Gold-standard parity: vs. fdlibm, CORE-MATH, CUDA, Julia
  [ ] TBS syntax entry
  [ ] Shared intermediate registered: TrigQuadrantReduction
```

Pathmaker owns completing the remaining boxes. Math researcher verifies
the polynomial coefficients and Payne-Hanek table match the paper exactly.

---

## Open Questions (owned by math researcher)

1. **`acot(0)` convention**: MATLAB (`π/2`) or Mathematica (`0`)? Ship
   both behind `using(acot_convention = ...)`; default to MATLAB for
   continuity.
2. **`atan2(±0, ±0)` sign propagation**: IEEE 754-2019 gives a full
   table; cross-check with CUDA, Julia, and glibc for any mismatch.
3. **Pi-scaled inverse `atan2pi` edge cases**: IEEE 754-2019 lists them;
   are any implementations known to get them wrong? Scout to file bugs.
4. **Complex-valued extensions**: `cis`, `clog`, `catan` all have branch
   cut conventions. Abramowitz-Stegun vs. Kahan 1987 differ at some cuts.
   Choose Kahan's conventions (the C/C++/Fortran standard).
5. **Gradians and turns**: gradians nearly extinct but MATLAB has
   `sind`/`cosd` (degrees) and no gradians; CUDA has no `sinpi` in
   gradians. Parameterize per TRIG-3 but do not build gradian-first recipes.

---

## Next Steps

- `references.md` — full citation list (in progress alongside this file).
- TRIG-3 spec: angle unit parameterization — decide parameter vs. recipe.
- TRIG-4 spec: shared intermediates, how they register, compatibility
  rules per TamSession contract.
- TRIG-9: per-function atom decomposition — which of the ~20 primitives
  each function consumes.
- TRIG-11: per-strategy compilation differences — the `strict` /
  `compensated` / `correctly_rounded` matrix specialized to each family.

Math researcher will pick up the remaining deep-dive tasks after
pathmaker confirms the catalog covers their scope.
