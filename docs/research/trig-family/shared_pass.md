# Shared-Pass Optimizations for the Trig Family

> TRIG-4 deliverable. Owner: math researcher. Rigorous flop-count analysis
> of which intermediates in the trig family justify registration as
> TamSession IntermediateTags. Decides: hyperbolics YES, forward trig
> PARTIAL, inverse trig YES.

Related docs: `catalog.md` (shared-pass summary table), `angle_units.md`
(unit-tagged compatibility). This doc is the detailed version.

---

## The question

A TamSession `IntermediateTag` is worth registering when:

1. At least two **independent** consumers need the same intermediate.
2. The intermediate costs more than ~10% of a full function call.
3. The consumers' requirements are compatible — same precision strategy,
   same unit, same input representation.

For each candidate below we measure in abstract flop counts and
real-world bench estimates, then recommend yes/no.

Flop counts are approximate f64 operation counts on scalar x86-64 with
FMA. "Core" = transcendental polynomial; "Reduce" = range reduction;
"Fixup" = quadrant/sign/reconstruction; "Reciprocal" = divide.

---

## 1. TrigQuadrantReduction{unit}

**Intermediate**: `(q, r_hi, r_lo)` from reducing `x` modulo `π/2`
(radians) or `1/2` (pi-scaled) or `1/4` (turns) or 90 (degrees) etc.

### Flop counts (radians)

| Component | Flops | Notes |
|---|---|---|
| Cody-Waite reduction (medium) | ~12 | 3 rounds of `k·PIO2_i` + residual |
| Payne-Hanek (large, `\|x\| ≥ 2²⁰·π/2`) | ~80 | 1200-bit multiplication via 8 f64 chunks |
| Integer quadrant extract | ~3 | cvt → and → cvt |

**Cost share of one sin call**: reduction is ~25 flops median, polynomial
core is ~18 flops (degree 11 Horner), fixup is ~3 flops. Reduction is
~55% of the call.

### Consumers

- `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `sincos`.
- In a typical quant pipeline, these might all be called on the **same**
  input column. Running them via one reduction pass saves ~55% per extra
  consumer.

### Worked example

```python
# User writes:
s = tambear.sin(col_x)
c = tambear.cos(col_x)
t = tambear.tan(col_x)

# Without sharing: 3 × (reduce + core + fixup)
# With sharing:    1 × reduce + 3 × (core + fixup)
# Cost ratio: 3·(25+18+3) = 138 flops vs. 25 + 3·(18+3) = 88 flops
# Savings: 36%
```

### Compatibility

The tag carries the unit. Radian reductions are not interchangeable with
pi-scaled or degrees. The precision strategy (`strict` / `compensated` /
`correctly_rounded`) also matters — a `strict` reduction gives 2-part
(r_hi, r_lo) but a `correctly_rounded` reduction needs 3-part (r_hi,
r_lo, r_ll). Compatibility predicate:

```
IntermediateTag::TrigQuadrantReduction {
    unit: Unit,                       // radians / pi / turns / deg / grad
    precision: PrecisionStrategy,     // strict / compensated / cr
    input_hash: u64,                  // content hash of source column
}
```

### Recommendation: **YES, register.**

- 7 consumers, all on the same input, 30-55% savings per extra consumer.
- Compatible sharing requires identical `(unit, precision, input_hash)`.
- Precision-downgrade is OK (compensated→strict gives strict, but not
  the reverse).

---

## 2. TrigSinCosCore

**Intermediate**: `(sin_core(r), cos_core(r))` where `r ∈ [-π/4, π/4]`
is the reduced radian.

### Flop counts

| Component | Flops |
|---|---|
| `r² = r_hi · r_hi` | 1 |
| sin polynomial `r + r³·P(r²)` | ~10 (Horner degree 5 in `r²`) |
| cos polynomial `1 − r²/2 + r⁴·Q(r²)` | ~11 |
| Residual folding (r_lo correction) | ~3 |

### Consumers

- `sin` and `cos` — both halves of `sincos`.
- `tan` via `sin_core / cos_core`.
- `cot` via `cos_core / sin_core`.
- `sec` via `1 / cos_core` plus fixup.
- `csc` via `1 / sin_core`.

### Worked example

```
sin + cos computed together:
  without sharing: 2 × (10+1+3) = 28 flops on the polynomial
  with sharing:    1 × (10+11+1+3) = 25 flops on the polynomial
  savings: 11%

sin + cos + tan computed together:
  without sharing: 3 × polynomial cost
  with sharing:    1 × polynomial cost + 1 divide for tan
  savings: ~40%
```

### Recommendation: **YES, but as implicit output of reduction.**

The `sin_core` and `cos_core` are cheap enough (~20 flops) that explicit
caching across TamSession calls is overhead-dominated. Better: when the
user calls `sincos`, we compute both. When the user calls `sin` and `cos`
separately, we let TamSession share at the **reduction** level — the
polynomial cost is small once reduction is cached.

**Tag it anyway** but with a low sharing priority:

```
IntermediateTag::TrigSinCosCore {
    r_repr: ReducedAngle,
    precision: PrecisionStrategy,
}
```

Use only inside `sincos`, `sincospi`, `sinhcosh` bundle functions.

---

## 3. HyperbolicExpm1Pair

**Intermediate**: `(expm1(x), expm1(-x))` or equivalently
`(e^x − 1, e^(-x) − 1)`.

### Flop counts

| Component | Flops |
|---|---|
| Range reduction `x = k·ln(2) + r` | ~7 |
| Polynomial `expm1(r)` degree 11 | ~11 |
| Reconstruction `2^k·(1+p) − 1` | ~4 |
| Total `expm1(x)` | ~22 |
| `expm1(-x)`: separate call | +22 |

### Why pairing saves work

For `|x|` near 0, `expm1(x) ≈ x + x²/2 + x³/6 + ...` and `expm1(-x) ≈
-x + x²/2 − x³/6 + ...`. The **even** powers are shared; only the odd
powers differ in sign. A polynomial split `(even(r²), odd(r²))` computed
once costs ~14 flops and yields both `expm1(x)` and `expm1(-x)` via
addition/subtraction.

**Savings for the pair**: compute once at ~14 flops instead of `2·22 =
44 flops`. **Net savings: 68%.**

### Consumers — all 6 hyperbolics

| Function | Formula using pair `(p, m)` where `p = e^x, m = e^(-x)` |
|---|---|
| `sinh(x)` | `(p − m)/2` |
| `cosh(x)` | `(p + m)/2` |
| `tanh(x)` | `(p − m) / (p + m)` |
| `coth(x)` | `(p + m) / (p − m)` |
| `sech(x)` | `2 / (p + m)` |
| `csch(x)` | `2 / (p − m)` |

Every single hyperbolic function is a rational combination of the pair.
**Computing the pair once and deriving all six is ~14 + 6·3 = 32 flops**
vs. computing each separately at ~45 flops each = 270 flops. **Savings:
88%.**

This is the biggest shared-intermediate win in the entire trig family.

### Small-x correction

For `|x| < 2⁻²⁷`, `expm1(x) ≈ x` and the pair `(x, -x)` is trivial; the
shared form is still correct but we can skip polynomial evaluation. For
`|x| > 710`, `expm1(x) = +∞` and `expm1(-x) = -1`. The pair form handles
both edges cleanly.

### Compatibility

```
IntermediateTag::HyperbolicExpm1Pair {
    input_hash: u64,
    precision: PrecisionStrategy,
}
```

Unit-agnostic (hyperbolics don't have angle units in the same sense as
circular trig — their argument is a real number, not an angle).

### Recommendation: **YES, highest-priority intermediate in the family.**

When any hyperbolic call lands, check TamSession for the pair. If absent,
compute and cache. If present, every subsequent hyperbolic is effectively
free.

---

## 4. AtanCore{table_region}

**Intermediate**: `atan`'s table lookup + polynomial result for a given
input region.

### Flop counts

| Component | Flops |
|---|---|
| Region classification (7 regions in fdlibm) | ~4 |
| Argument reduction for region | ~3 |
| Polynomial evaluation (degree 11 odd) | ~12 |
| Table-add for reconstruction | 1 |
| Total `atan(x)` | ~20 |

### Consumers

- `atan` directly.
- `atan2(y, x)` via `atan(y/x)` for most of its domain (with sign / π
  adjustment for quadrant).
- `acot(x) = π/2 − atan(x)` (or the MATLAB-continuous variant).
- `atan2pi(y, x) = atan2(y, x) / π`.

### Compatibility

Different consumers need different output scaling. `atan2pi` wants the
result divided by π; `atan2` wants the result plus a quadrant offset.
The core itself (pre-fixup) is shareable — the fixups are cheap
post-processing.

```
IntermediateTag::AtanCore {
    input_hash: u64,
    precision: PrecisionStrategy,
}
```

### Recommendation: **YES, register.**

Particularly valuable when users call `atan2(y, x)` and `atan2pi(y, x)`
on the same `(y, x)` — both reuse the core, differing only in the final
scale.

---

## 5. AsinCore{region}

**Intermediate**: `asin`'s polynomial result on the direct or half-angle branch.

### Flop counts

- Direct branch (`|x| < 0.975`): degree-10 polynomial in `x²` via Horner,
  ~10 flops.
- Near-unity branch (`|x| ≥ 0.975`): half-angle transform `asin(x) =
  π/2 − 2·asin(√((1-x)/2))`, ~5 extra flops for the sqrt and doubling.

### Consumers

- `asin`.
- `acos(x) = π/2 − asin(x)` (or half-angle variant for large x).
- `asec(x) = acos(1/x)`.
- `acsc(x) = asin(1/x)`.

### Compatibility

Half-angle regime needs a different intermediate (it's `2·asin(y)` for
`y = √((1-x)/2)`). Two separate tags.

```
IntermediateTag::AsinCoreDirect { input_hash, precision }
IntermediateTag::AsinCoreHalfAngle { transformed_input_hash, precision }
```

### Recommendation: **YES, register both.**

Savings: ~60% when `asin` and `acos` are called on the same input.

---

## 6. InverseHyperbolicLog1pTransform

**Intermediate**: the `log1p` argument for each inverse hyperbolic.

- `asinh(x) = log1p(x + x²/(1 + √(1+x²)))` for `|x| < 1`.
- `acosh(x) = log1p(t + √(2t + t²))` for `t = x − 1`, `x ≥ 1`.
- `atanh(x) = (1/2)·log1p(2x/(1−x))`.
- `acsch(x) = asinh(1/x)`.
- `asech(x) = acosh(1/x)`.
- `acoth(x) = atanh(1/x)`.

### Observation

The `log1p` call itself is the expensive part (~20 flops). Sharing the
**argument** to `log1p` would only help if multiple consumers produced
the same argument — which doesn't happen here because each inverse
hyperbolic has a distinct formula.

**However**: `log1p` itself is a tambear primitive, and its own
range-reduction is cached. So `asinh(x)` and `atanh(y)` with different
`x`, `y` don't share anything — but `atanh(x)` and `acoth(x)` do, via
the reciprocal relation.

### Recommendation: **Don't register a separate tag.**

The `log1p` primitive already caches its own internal intermediates;
letting each inverse hyperbolic call `log1p(...)` with its own argument
is fine. No new TrigFamily intermediate is warranted here.

---

## 7. Sinpi/Cospi Exact-Reduction

**Intermediate**: `(q, r)` from `x mod 2` for pi-scaled inputs.

### Flop counts

- `frac(x)` via `x − round(x, mode=round_to_even)`: 2 flops.
- Quadrant `q = ⌊2r⌋`: 1 flop.
- Final `r ∈ [-1/4, 1/4]`: 1 flop.

**Total reduction: 4 flops** — vs. ~25 flops for radians.

### Consumers

`sinpi`, `cospi`, `tanpi`, `sincospi`, `sinhcospi` (if added).

### Recommendation: **YES, register as separate tag.**

```
IntermediateTag::PiScaledReduction {
    input_hash: u64,
}
```

No precision variant — the reduction is always exact.

---

## Summary table

| Tag | Consumers | Savings | Precision-sensitive | Register? |
|---|---|---|---|---|
| `TrigQuadrantReduction{unit}` | 7 (all forward trig) | 30–55% | YES | **Yes** |
| `TrigSinCosCore` | 6 (forward trig) | 11–40% | YES | Low priority |
| `HyperbolicExpm1Pair` | 6 (all hyperbolics) | 68–88% | YES | **Yes, top priority** |
| `AtanCore` | 4 (atan family) | 40–60% | YES | **Yes** |
| `AsinCoreDirect` + `AsinCoreHalfAngle` | 4 (asin family) | 40–60% | YES | **Yes (both)** |
| `InverseHyperbolicLog1pTransform` | 6 | 0% (no arg sharing) | n/a | No |
| `PiScaledReduction` | 4 | 85% of reduction | no | **Yes** |

**Net**: 6 new IntermediateTags to register.

---

## Is there a "MomentStats of trig"?

The question in the task title: does the trig family have a single
"universal shared state" analogous to `MomentStats` in the descriptive-
statistics family? The answer from this analysis:

**No single universal state, but two family-level shared states**:

1. **`HyperbolicExpm1Pair`** — covers ALL 6 hyperbolic functions.
   This IS the "MomentStats of hyperbolics." One intermediate, six
   consumers, ~85% savings in the common case.

2. **`TrigQuadrantReduction{unit}`** — covers ALL 7 forward trig
   functions per unit. This IS the "MomentStats of forward trig"
   at the unit level.

Inverse trig is the odd family — it doesn't have a single shared state.
Instead it has two cores (`AtanCore`, `AsinCore`) that together cover
the family.

**Structural observation** (convergence check): the family with the
biggest shared-state win is **hyperbolics**, because all six functions
are algebraically rational combinations of the exp pair. Forward trig
shares the **reduction** but not the core evaluation (sin core and cos
core really are different polynomials). Inverse trig shares cores but
has two of them. This progression — algebraic closure (hyperbolic) →
shared preprocessing (forward) → shared subroutine (inverse) — is itself
a rhyme with how evidence-topology fields have varying "reducibility
depth" in tambear's larger architecture.

---

## Implementation plan for pathmaker

1. Add `HyperbolicExpm1Pair` primitive (TRIG-15). Ship `sinh` and `cosh`
   as thin wrappers that look up or compute the pair, then derive.
2. Add `TrigQuadrantReduction{unit}` primitive (TRIG-13). The existing
   `sin.rs` already has this inline — extract it to `reduce_pio2.rs` and
   register the TamSession tag.
3. Add `AtanCore` primitive (TRIG-14). Extract fdlibm's
   table-plus-polynomial into `atan_core.rs`; `atan`, `atan2`, `atan2pi`,
   `acot` all call it.
4. Add `AsinCoreDirect` and `AsinCoreHalfAngle` (TRIG-14). Two primitives
   for the two regimes.
5. Add `PiScaledReduction` primitive (TRIG-16). Trivially simple
   (4 flops), but worth a named primitive for consistency.

Each primitive ships with its compatibility predicate and a test proving
two consumers get bit-identical answers via shared vs. independent
computation.

---

## Related work

- CORE-MATH does **not** share intermediates across calls — their focus
  is per-call correctness.
- fdlibm inlines kernel functions but does not cross-function share.
- Intel SVML batches across vector lanes but not across call sites.
- **This is an area where tambear can be first.** Shared-intermediate
  caching across logically-independent math calls is a tambear-specific
  win enabled by TamSession.

This is one of the compiler-level advantages from the WinRapids
architecture: the `using()` flow-through design means we can share
genuinely across call sites in a way that libm-style per-call APIs
cannot.
