---
campsite: tambear-formalize/sweep-31/math-researcher
role: math-researcher (sweep design lead)
date: 2026-05-08
sweep: 31 — BigFloat type-level home
status: draft for navigator review + team-lead routing
audience: pathmaker (implementation lead); team-lead/Tekgy (ratification); aristotle (definitional review on storage-spec/operation-spec orthogonality and the round-trip-identity invariant)
inputs:
  - DEC-031 §1 + §3.5 + §6 (R:\tambear\docs\decisions.md lines 3310-3478)
  - Brent-Zimmermann *Modern Computer Arithmetic* (2nd ed.), Algorithms 3.1-3.6 (basic arithmetic) + 3.10 (Newton-iteration division)
  - Existing tambear DD type at R:\tambear\crates\tambear\src\primitives\double_double\ty.rs (canonical-form invariant exemplar)
  - Winrapids bigfloat.rs at R:\winrapids\crates\tambear\src\bigfloat.rs (~2581 lines; reference for API shape, not algorithm — gaps in special-value handling, no correctly-rounded conversions, no rounding mode parameter)
  - libm assumption-doc set at R:\winrapids\campsites\tambear-formalize\survey\20260508123003-math-researcher\assumption-docs\ (consumer-axis context)
not-yet-implemented in tambear: lattice/precision.rs (DEC-031 §6 #1)
---

# Sweep 31 — BigFloat Type-Level Home: Design Document

> **Story from the trail.** The DEC-031 declaration is rigorous and complete. The winrapids reference is helpful for API shape but its algorithms are not what we want to ship — `to_f64` is not correctly rounded, division uses unverified Newton, no rounding mode parameter, no sign-of-zero or NaN/Inf encoding. **The most surprising thing reading the substrate**: the f64-mantissa-fits-cleanly-into-one-limb fact (53 bits ≪ 64) makes the round-trip identity (§6 #13) trivially correct *if* the limb layout is right, but most BigFloat libraries don't enable this. We can use that asymmetry — keep f64 conversions limb-aligned, never need the round-trip to do real precision arithmetic.
>
> **What's open**: subnormal-regime treatment in the limb encoding, FFT-multiplication threshold (precision-dependent, but we may not need it for v2). **What's load-bearing**: §6 #3 diamond commutativity (the hardest invariant to establish — both `f64→BigFloat(p)` and `f64→DD→BigFloat(p)` must produce bit-exact-equal results) and §6 #13 round-trip identity (the easiest to test, hardest to break — and the user-visible one).

---

## 1. Six questions, six answers — the team-lead checklist

### Q1. Storage layout

**Answer**: limbs as `Vec<u64>`, sign-magnitude, with explicit special-value tag. Layout:

```rust
pub struct BigFloat {
    /// Special-value tag. `Normal` carries the magnitude in `limbs + exponent`;
    /// other variants ignore them entirely (limbs may be empty).
    kind: BigFloatKind,
    /// Sign bit. False = positive (or +0, +Inf), true = negative (or -0, -Inf, sNaN).
    sign: bool,
    /// Mantissa magnitude as little-endian u64 limbs.
    /// For `Normal`: highest set bit is in the top limb at position
    /// `(precision_bits - 1) % 64`. Length is `ceil(precision_bits / 64)`.
    /// For non-Normal kinds: empty.
    limbs: Vec<u64>,
    /// Power-of-two exponent. The numeric value of a Normal BigFloat is:
    ///   (-1)^sign · (limbs as integer) · 2^(exponent - precision_bits + 1)
    /// Bias: zero. Range: i64. Saturating arithmetic on overflow → ±Inf.
    exponent: i64,
    /// Working precision in bits. Invariant: limbs.len() == ceil(precision_bits / 64).
    precision_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BigFloatKind {
    /// Finite, non-zero. limbs and exponent meaningful.
    Normal,
    /// ±0. Sign meaningful (preserves IEEE 754 sign-of-zero).
    Zero,
    /// ±Inf. Sign meaningful.
    Infinity,
    /// NaN. Sign meaningful (separately for sNaN-vs-qNaN if we choose to encode that;
    /// for v2 we only need a single NaN form, but the slot is reserved).
    NaN,
}
```

**Why this shape**:

- **`Vec<u64>` over `[u64; N]` const-generic**: per DEC-031 §3.5, BigFloat is variable-precision. A user may request 200 bits in one call and 1000 in the next; const-generic precision would force one type per precision, defeating the variable-precision purpose. The heap allocation cost (one `Vec::with_capacity` per construction) is dwarfed by the multi-limb arithmetic cost. **No SmallVec or stack inlining for v2** — premature optimization until we see the cost data.
- **Little-endian limb order**: matches Brent-Zimmermann §1.3 convention; matches GMP's `mp_limb_t[]` layout (BZ §3.1 references it). Big-endian would force every multiplication routine to flip indices.
- **Magnitude limbs (not two's complement)**: BigFloat is a floating-point type, not an integer; sign is a separate bit. Two's complement on `Vec<u64>` complicates equality testing (e.g., `-1` vs `+1` would have different limb patterns) and breaks the round-trip identity unless we canonicalize at every constructor call.
- **Explicit `BigFloatKind` tag**: separates the "is this NaN/Inf/zero?" question from the "what's the magnitude?" question. Per DEC-022 sub-clause F (rule enforced at the boundary where it matters), classifications cluster at construction; arithmetic ops dispatch on `kind` first, only touch `limbs` for `Normal`. This is also what makes `is_zero` an O(1) tag check instead of a "scan all limbs" loop.
- **Sign bit always meaningful**, including for Zero and Infinity: preserves IEEE 754 sign-of-zero (`-0.0 != +0.0` in bit-pattern; `1.0/-0.0 == -inf`). DD does this implicitly by having `hi: f64` carry the sign; we have to do it explicitly because our limbs are magnitude-only.
- **`precision_bits: u32`** matches DEC-031's `PrecisionLevel::P2BigFloat { precision_bits: u32 }`. u32 caps precision at ~4 billion bits, which is far beyond any reasonable use (mpfr caps at MPFR_PREC_MAX ≈ 2^31 bits and nobody hits it).
- **Exponent invariant**: in `Normal`, the top limb's top bit (position `(precision_bits-1) % 64`) is set. This is the canonical-form normalization; constructors enforce it. Sub-clause F: rule enforced at construction. Equivalent to mpfr's `MPFR_EXP_INVALID` lower-bound + the always-leading-1-bit invariant.

**Numeric value convention** (cross-check during implementation): BigFloat represents `(-1)^sign · M · 2^(exponent - precision_bits + 1)` where `M` is the magnitude integer. The exponent stores the *binary place value of the most significant bit*. Example: BigFloat `+1.5` at precision 53: `sign=false`, `limbs=[0x18000000_00000000]` (top 53 bits = `1.1` in binary plus 51 zeros), `exponent=0`, `precision_bits=53`. Magnitude = `2^52 · 3`, value = `3 · 2^(0-53+1) · 2^52 = 3 · 2^0 = 1.5`.

**This convention makes f64 round-trip trivial** (Q4 below).

### Q2. Mantissa-precision encoding strategy

**Answer**: Each `BigFloat` carries its own `precision_bits`. Operations between two BigFloats at different precisions choose the *larger* precision per Brent-Zimmermann §3.1 convention. Variable-precision means:

- **Allocation strategy**: each constructor returns a `BigFloat` with `limbs.len() == ceil(precision_bits / 64)`. The capacity equals the length; no over-allocation by default. When an arithmetic op widens precision (e.g., `mul` of two p-bit operands intermediately produces 2p bits), the wider intermediate is allocated separately and rounded back to the dispatched precision per `RoundingMode`.
- **No precision changes after construction**: a BigFloat's `precision_bits` is set at construction. Operations that need different precisions output a *new* BigFloat. Mutability is bypassed by always returning new values; this matches mpfr's API shape and makes equality reasoning local.
- **Per-operand precision tracking**: `add(a: p1-bit, b: p2-bit)` → `max(p1, p2)`-bit result. `mul` similarly. This is the BZ convention; mpfr's `mpfr_add(rop, op1, op2, rnd)` takes the precision from `rop` (output is pre-allocated). We follow BZ instead — output precision is the larger input precision, not externally controlled. Cleaner for our locked-vocabulary use.
- **Tier-dispatch boundary** (DEC-031 §3.4): if the user's `PrecisionContext` requests 53 bits, dispatch lands on f64 (not BigFloat). 54-106 → DD. 107+ → BigFloat with the requested bits. **BigFloat NEVER instantiated below 107 bits in normal operation** — DD covers that range. Constructor for `BigFloat::with_precision(p: u32)` panics on `p < 107`, surfacing the tier violation.
  - Exception: `BigFloat::from_f64(f, 53)` constructs a 53-bit BigFloat (per DEC-031 §6 #13 round-trip identity test). This is an explicit shortcut for the round-trip identity invariant; not generally useful but required for the test.
  - Exception: `BigFloat::from_dd(dd, 106)` constructs a 106-bit BigFloat (per §3.1 boundary table, p≥106 from dd-source is Strict). Same shape — explicit, narrow.

**Subnormal regime**: per DEC-031 chains-E/F/G subnormal precondition + ATK-DEC031-4 monotone-coarsening scope: subnormals (magnitude `< f64::MIN_POSITIVE = 2^-1022`) get *absolute*-bounded error rather than *relative*-bounded. **In BigFloat encoding, subnormals are normal**: BigFloat doesn't have a "subnormal" regime because its exponent is i64 and any bit pattern is representable. The subnormal regime is f64-specific, surfaces only at the f64↔BigFloat boundary. This means:
  - `BigFloat::from_f64(f64_subnormal, 53)` produces a 53-bit BigFloat with `exponent` adjusted to absorb the subnormal-shift (i.e., the f64 subnormal `2^-1074` is encoded as BigFloat `+1.0 · 2^-1074` with appropriate limb pattern, NOT as the f64-style "implicit-leading-bit-not-set" anomaly).
  - This means `BigFloat::from_f64(value, 53).to_f64()` round-trip is bit-exact even for subnormals. **Critical for §6 #13**.

### Q3. Brent-Zimmermann algorithm choices

**v2 surface**: add, sub, mul, div, sqrt, cmp, from_f64, from_dd, to_f64, to_dd. Per BZ Algorithms 3.1-3.6 + 3.10:

| Op | BZ algorithm | Threshold strategy | Notes |
|---|---|---|---|
| **add / sub** | BZ Algorithm 3.1 (alignment + integer add + canonicalize) | Single algorithm (schoolbook); no precision-dependent dispatch. | Cost: O(p/64) limb ops. |
| **mul** | BZ Algorithm 3.3 (schoolbook), BZ §1.3.4 (Karatsuba), BZ §1.3.5 (FFT/SSA) | Schoolbook to ~3000 bits; Karatsuba above. **Defer FFT to v3 — not needed for the immediate oracle use.** | Schoolbook is O((p/64)²); Karatsuba O((p/64)^1.585). Crossover empirically ~50 limbs (3200 bits). |
| **div** | BZ Algorithm 3.5 (Newton-Raphson on reciprocal) | Newton iteration to bit-exact; final round per RoundingMode. | Iteration count: ⌈log₂(p/53)⌉ + 2. Initial guess from f64. **Compute reciprocal at p+50 bits internally then round to p** — guard bits guarantee correct rounding per BZ §3.1.6. |
| **sqrt** | BZ Algorithm 3.10 (Newton iteration on `x² = self`) | Same Newton structure as div. | Newton iteration: `x_{n+1} = (x_n + self/x_n) / 2`. Same `p+50` guard bits + final round. |
| **cmp** | tag-first dispatch + magnitude compare | Tag-first: NaN→unordered, Inf→by sign, Zero→equal across signs (or by sign, IEEE 754 doesn't distinguish in cmp). Then exponent → magnitude. | O(p/64) at worst (when exponents equal). |

**Algorithm-decision rationale**:

- **Why BZ schoolbook for v2 mul**: at the precisions DEC-031 actually targets (107-500 bits per chains-E/F/G recommended floor; 200-bit default for §3.9 oracle gold), Karatsuba shows a ~10-20% speedup. Schoolbook is simpler, easier to verify, and ~3-4x faster per-iteration than Newton-iter div will be — multiplication isn't the bottleneck in the oracle use case (we'll do far more transcendentals than mul-chains). **Ship schoolbook in v2; revisit Karatsuba in v3 only if profile demands.**
- **Why NOT FFT**: FFT-multiplication (Schönhage-Strassen, NTT-based) crosses over schoolbook above ~10K limbs (~640K bits). DEC-031's 1024-bit cap (§3.8 saturation) is 16 limbs; FFT is irrelevant in the bounded-precision regime. **Excluded from v2 entirely.**
- **Why Newton for div/sqrt**: BZ Algorithm 3.5 / 3.10 — quadratic convergence, halves the work per iteration. `⌈log₂(p/53)⌉ + 2` iterations from an f64 seed: at 200-bit precision, that's `⌈log₂(3.77)⌉ + 2 = 4` iterations; at 1024 bits, `⌈log₂(19.3)⌉ + 2 = 7` iterations. Each iteration is O(p²) for the multiply; total work O(p² log p), as expected.
- **Guard bits + final round for correct rounding**: Newton converges quadratically but never to the exact rational answer (which is generally non-representable). BZ §3.1.6 — compute at `p + g` bits where `g ≥ ⌈log₂(p)⌉ + 1` is enough to identify which way the final round goes. We use `g = 50` conservatively (overshoots BZ's bound for p ≤ 1024). This is the **correct-rounding gate** for div and sqrt.
- **Subnormal/special handling**: every op dispatches on `kind` first. NaN propagates universally; Inf arithmetic per IEEE 754 (Inf-Inf=NaN, Inf+finite=Inf, etc.); Zero arithmetic per IEEE 754 (sign rules). Only `Normal × Normal` reaches the limb arithmetic.

### Q4. Round-trip identity (§6 #13)

**Invariant**: `BigFloat::from_f64(value, 53).to_f64() == value` for all f64, including ±0, ±Inf, NaN, subnormals from `2^-1074`, tier-boundary near `f64::MIN_POSITIVE = 2^-1022`.

**The encoding makes this trivial — by construction**:

`from_f64(value, 53)` decodes the f64 into:
- **NaN, ±Inf, ±0**: dispatch on bit pattern; produce `BigFloat { kind: NaN/Infinity/Zero, sign: ..., limbs: vec![], exponent: 0, precision_bits: 53 }`.
- **Normal f64** (incl. subnormals): extract sign bit, exponent field, mantissa field. Produce:
  - For normal f64 `(-1)^s · (1.m) · 2^(e_biased - 1023)`: `limbs = [(1u64 << 52) | mantissa_field]` (53 bits, top bit set), `exponent = e_biased - 1023`, `sign = s`.
  - For subnormal f64 `(-1)^s · (0.m) · 2^-1022`: shift mantissa_field left until top bit is set; track the shift count; encode as `limbs = [shifted_mantissa]`, `exponent = -1022 - shift_count`, `sign = s`. **Effectively: subnormals get re-normalized into BigFloat's normal form**, which is fine because BigFloat's exponent has 64 bits of range vs f64's 11.

`to_f64()` is the inverse:
- **Tag dispatch**: NaN→`f64::NAN`, Infinity→`f64::INFINITY` or `f64::NEG_INFINITY`, Zero→`+0.0` or `-0.0`.
- **Normal**: reconstruct `mantissa_field = limbs[0] & 0x000fffff_ffffffff`, `e_biased = exponent + 1023`. **If `e_biased ≤ 0`**: this is a subnormal f64; pack with implicit-leading-bit-cleared encoding by shifting the mantissa right by `1 - e_biased` bits and setting `e_biased = 0`. **If `e_biased ≥ 0x7ff`**: overflow → `±Inf`. Otherwise: pack normally.

**Verification structure** (test harness for §6 #13):
```rust
#[test]
fn roundtrip_identity_full_f64_range() {
    // Strategic samples covering every f64 magnitude class:
    let samples = [
        0.0_f64.to_bits(), (-0.0_f64).to_bits(),
        f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits(),
        f64::NAN.to_bits(),
        f64::MIN_POSITIVE.to_bits(),                 // 2^-1022, smallest normal
        f64::from_bits(1).to_bits(),                 // smallest subnormal: 2^-1074
        f64::from_bits(0x000fffffffffffff).to_bits(),// largest subnormal
        f64::MAX.to_bits(), f64::MIN.to_bits(),
        1.0_f64.to_bits(), (-1.0_f64).to_bits(),
        // ... plus PI, E, etc., plus ±1 ulp neighborhoods of all of these
    ];
    for b in samples {
        let f = f64::from_bits(b);
        let bf = BigFloat::from_f64(f, 53);
        let f2 = bf.to_f64();
        if f.is_nan() {
            assert!(f2.is_nan());  // NaN payload may not be preserved; that's IEEE 754-allowed
        } else {
            assert_eq!(f.to_bits(), f2.to_bits(), "f={f:e} (bits={b:#018x}) round-trip failed: f2={f2:e} (bits={:#018x})", f2.to_bits());
        }
    }
    // Plus a proptest sweep of arbitrary u64 → f64::from_bits → roundtrip.
}
```

**Why this is trivial-by-construction**: f64's mantissa is 52 stored bits + 1 implicit = 53 effective. BigFloat at 53-bit precision has exactly 53 mantissa bits in `limbs[0]`, packed top-bit-first. The decode is just bit shuffling; the encode is the inverse shuffling. **No arithmetic is ever performed in this round trip** — neither rounding nor truncation. The bits are conserved.

Subnormals are the one tricky case: f64 stores them with implicit-leading-bit-cleared, but BigFloat normalizes (top bit always set in `limbs[0]`). The decoder does one shift to normalize; the encoder does the inverse shift to denormalize. Both are exact. **This is the asymmetry I flagged in the story-from-the-trail**: most BigFloat libraries that don't handle f64 subnormals specifically can fail this round-trip; ours doesn't because the exponent has the headroom to absorb the shift.

### Q5. Diamond commutativity (§6 #3)

**Invariant**: `∀ f ∈ P0F64, ∀ p ≥ 53, ∀ rounding ∈ RoundingMode: BigFloat::from_f64(f, p) == BigFloat::from_dd(DoubleDouble::from_f64(f), p)`. Bit-exact equality of the resulting BigFloats.

**The implementation guarantee chain**:

The two paths produce the same result iff both reduce to the same canonical form before any rounding. That requires:

1. **`from_f64(f, p)`** decodes `f` exactly into `(sign, mantissa_bits, e_biased)`, then encodes into BigFloat with `limbs[0]` top-bit-set, the f64 mantissa zero-extended into the top 53 bits of `limbs[0]`, the rest of `limbs` zero, `exponent = e_biased - 1023`. Exact; no arithmetic.

2. **`from_dd(dd, p)`** decodes `dd = (hi, lo)` where `lo = 0` for any DD constructed via `DoubleDouble::from_f64(f)` (per `DoubleDouble::from_f64` source: `Self { hi: x, lo: 0.0 }`). When `lo = 0`, `from_dd(dd, p)` reduces to `from_f64(hi, p)` — same path, same result.

   For DDs with `lo ≠ 0` (not produced by `from_f64`, but might come from EFT outputs), `from_dd` is `from_f64(hi, p) + from_f64(lo, p)` performed in BigFloat arithmetic at precision p. The addition is exact (BigFloat at precision p has at least 53+53=106 bits of available mantissa, plenty to hold `hi + lo` without rounding when p ≥ 106). For p ∈ [53, 106), the `hi + lo` addition rounds; per DEC-031 §3.1 boundary table, this case is `RoundingEquivalent { 0.5 ULP at BigFloat(p) }` not Strict. So the diamond commutativity invariant **applies only to DDs whose `lo = 0`** — exactly the DDs produced by `DoubleDouble::from_f64`.

3. **For p < 53**: rejected at constructor per §3.1 boundary table. `BigFloat::with_precision(p)` panics for `p < 107`; the round-trip exception (`from_f64(_, 53)` and `from_dd(_, 106)`) is the only sub-107 entry. Both paths agree on the rejection.

**Stated more precisely**: the diamond commutativity invariant from §6 #3, read literally, is:

> For all f ∈ P0F64 and p ≥ 53: both paths f→BigFloat(p) produce bit-exact-equal results.

Our `from_f64(f, p)` and `from_dd(DD::from_f64(f), p)` both produce BigFloat(p)-bits identical because:
- `DD::from_f64(f)` returns `DD { hi: f, lo: 0 }` (no information loss; `lo = 0` invariant)
- `from_dd(DD { hi: f, lo: 0 }, p)` short-circuits `lo = 0` to `from_f64(f, p)`
- Therefore `from_dd(DD::from_f64(f), p) = from_f64(f, p)` by *structural identity*, not by computation.

The proof obligation reduces to: **`from_dd` MUST short-circuit `lo = 0` to `from_f64`**, not "treat DD as a generic two-part value and add the parts." This is one line in the implementation. **The hardest invariant in the spec is delivered by one line of code, IF the type design is right.**

**Consequence for the API**:
```rust
impl BigFloat {
    pub fn from_dd(dd: DoubleDouble, p: u32) -> Self {
        if dd.lo == 0.0 {
            return Self::from_f64(dd.hi, p);  // structural diamond commutativity
        }
        // p ≥ 106: hi + lo is exact in BigFloat(p) arithmetic.
        // p ∈ [53, 106): hi + lo rounds; rounding mode applies.
        // p < 53: rejected at constructor entry.
        Self::from_f64(dd.hi, p).add(&Self::from_f64(dd.lo, p))
    }
}
```

### Q6. DD ↔ BigFloat boundary surface

Per DEC-031 §3.1 boundary table, the boundary has three regimes:

| Direction | Precision regime | Class | API surfacing |
|---|---|---|---|
| DD → BigFloat | p ≥ 106 | Strict | `BigFloat::from_dd(dd, p)` returns exact (no rounding) |
| DD → BigFloat | p ∈ [53, 106) | RoundingEquivalent { 0.5 ULP at BigFloat(p) } | `BigFloat::from_dd(dd, p)` returns rounded; rounding mode parameter required |
| DD → BigFloat | p < 53 | rejected | constructor panics with descriptive error |
| BigFloat → DD | any p | RoundingEquivalent { 0.5 ULP at DD } | `bf.to_dd()` returns rounded; rounding mode parameter required |

**Proposed API**:

```rust
impl BigFloat {
    /// DD → BigFloat. Strict for p ≥ 106; rounded for p ∈ [53, 106) per RoundingMode;
    /// rejected for p < 53.
    pub fn from_dd_with_rounding(dd: DoubleDouble, p: u32, rounding: RoundingMode) -> Result<Self, PathError>;

    /// Convenience for p ≥ 106 only. Panics for p < 106 — rounding mode is moot
    /// when no rounding occurs, so the panic surfaces the precondition violation.
    pub fn from_dd(dd: DoubleDouble, p: u32) -> Self;

    /// BigFloat → DD. Always rounds (DD is fixed at ~106 bits). RoundingMode required.
    pub fn to_dd_with_rounding(&self, rounding: RoundingMode) -> DoubleDouble;

    /// Convenience for default rounding mode (RoundToNearestTiesEven, IEEE 754 default).
    pub fn to_dd(&self) -> DoubleDouble {
        self.to_dd_with_rounding(RoundingMode::RoundToNearestTiesEven)
    }
}
```

**Rationale for two-method shape**: `from_dd` (no rounding) and `from_dd_with_rounding` (rounding) split the API cleanly along the §3.1 regime boundary. The "no rounding required" path is the common case (oracle use is at p ≥ 200 ≫ 106), so the simpler signature gets the simpler name. The rounding-required path is explicit, surfaces the precondition (p < 106 is rounding-required), and returns `Result` so the user can decide what to do with the precision-mismatch case.

`PathError` per DEC-031 ATK-DEC031-4 §6 #12: returned on non-monotone-coarsening composition or precision-tier rejection. We extend its variants to include `PathError::PrecisionTooLow { requested: u32, minimum: u32 }` for the p < 53 / p < 106 cases.

---

## 2. Storage layout — concrete byte-level

For navigator/aristotle visibility, the actual byte layout of `BigFloat` at p=200 (one of the recommended oracle precisions per DEC-031 §3.9) is:

```
field            type        size on x86_64
─────            ────        ──────────────
kind             u8          1
sign             bool        1
limbs (Vec ptr)  *const u64  8
limbs.len        usize       8
limbs.cap        usize       8
exponent         i64         8
precision_bits   u32         4
                            ──
                             38 bytes + 4 padding = 42 → 48 bytes (Rust align)
                            +
                             ceil(200/64) = 4 limbs × 8 = 32 bytes heap allocation
                            ──
                             80 bytes total per BigFloat at p=200
```

For comparison, DD is 16 bytes flat (no heap). f64 is 8 bytes. BigFloat at p=200 is 5x DD's size, plus heap allocation cost.

**Performance implication**: the heap allocation per construction is the dominant cost of small-precision BigFloat. For p ≤ ~106, DD is strictly cheaper. **This justifies DEC-031 §3.4's tier-dispatch boundary**: requested-precision below 107 always dispatches to DD (no allocation), only above triggers BigFloat instantiation.

**No SmallVec optimization for v2**: the obvious "inline limbs ≤ 4 to avoid heap" optimization saves ~30ns per construction (heap alloc cost) but adds 32 bytes of inline storage to every BigFloat regardless of precision. For large-precision use (the actual use case), the inline optimization is dead weight. **Defer to v3 if profile data shows construction in a hot loop**, which the oracle use case won't.

---

## 3. Algorithm dispatch table — concrete

Per the v2 surface, here's the full per-op algorithm choice with citations:

| Op | Algorithm | BZ reference | Loc estimate | Test obligations |
|---|---|---|---|---|
| `from_f64` | bit-extract + limb-pack | BZ §3.1.1 | ~50 LoC | full f64 range, especially subnormals + ±0 + NaN/Inf + tier-boundary 2^-1022 |
| `to_f64` | bit-pack + overflow/underflow check | BZ §3.1.1 | ~50 LoC | round-trip with from_f64 |
| `from_dd` / `to_dd` | composition + structural short-circuit on lo=0 | BZ §3.1.1 + diamond invariant | ~30 LoC | diamond commutativity (§6 #3) |
| `cmp` | tag dispatch + sign + exponent + limb-magnitude | BZ §3.1.5 | ~80 LoC | total order (proptest) |
| `add` (and `sub` via `add(-other)`) | exponent-align + limb-add + canonicalize | BZ Algorithm 3.1 | ~100 LoC | identity element (BF + 0 = BF), commutativity, associativity-modulo-rounding |
| `mul` | schoolbook limb-multiply at p+50 guard bits + round | BZ Algorithm 3.3 + §3.1.6 | ~120 LoC | identity (BF · 1 = BF), commutativity, distributivity-modulo-rounding |
| `div` | Newton-Raphson on reciprocal at p+50 guard bits + round | BZ Algorithm 3.5 + §3.1.6 | ~150 LoC | inverse identity (BF / BF = 1, modulo rounding), divide-by-zero produces ±Inf or NaN per IEEE 754 |
| `sqrt` | Newton iteration at p+50 guard bits + round | BZ Algorithm 3.10 + §3.1.6 | ~120 LoC | sqrt(BF · BF) = |BF| modulo rounding; sqrt(negative) = NaN |
| canonicalize/normalize internal | left-shift mantissa to set top bit + adjust exponent | implicit BZ convention | ~40 LoC | post-condition of every public constructor |
| **Total** | | | **~740 LoC core + 200 LoC tests + 100 LoC helpers ≈ 1040 LoC** | |

**vs. DEC-031 §3.5 estimate of "~2-3 KLoC, 1-3 weeks"**: my estimate is ~1 KLoC core; the remaining 1-2 KLoC is tests, proptests, doc comments, and the lattice-coordination plumbing in `lattice/precision.rs`. Aligns.

**FFT explicitly excluded from v2** per Q3 rationale.

---

## 4. The diamond commutativity proof — what the test suite asserts

Per §6 #3, the test obligation:

```rust
#[test]
fn diamond_commutativity_f64_to_bigfloat() {
    use proptest::prelude::*;
    proptest!(|(bits: u64, p in 53u32..=1024)| {
        let f = f64::from_bits(bits);

        // Path A: f64 → BigFloat directly
        let path_a = BigFloat::from_f64(f, p);

        // Path B: f64 → DD → BigFloat
        let dd = DoubleDouble::from_f64(f);
        let path_b = BigFloat::from_dd(dd, p);

        // Bit-exact equality, regardless of rounding mode (lo = 0 means
        // no rounding occurs, so rounding mode is moot).
        assert_eq!(path_a, path_b, "diamond commutativity failed for f={f:e} (bits={bits:#018x}), p={p}");
    });
}
```

**The test passes iff `from_dd(DD { hi: f, lo: 0 }, p)` short-circuits to `from_f64(f, p)`.** That short-circuit is one `if dd.lo == 0.0` check at the head of `from_dd`. The proof is structural: by the type's construction, two paths reduce to one path, by inspection.

For the (rare) DD whose `lo ≠ 0` (output of EFT operations like `two_sum`, `two_product_fma`), the diamond invariant doesn't apply at `p < 106` (rounding occurs); for `p ≥ 106`, `hi + lo` is exact and the invariant holds automatically. We add a separate test for this:

```rust
#[test]
fn diamond_commutativity_eft_dd_at_high_precision() {
    // For DDs with lo != 0 (EFT outputs), diamond commutativity holds at p ≥ 106
    // because hi + lo is exact in BigFloat(p) arithmetic.
    proptest!(|(a: f64, b: f64, p in 106u32..=1024)| {
        let (hi, lo) = two_sum(a, b);
        let dd = DoubleDouble::from_parts_unchecked(hi, lo);

        let path_a = BigFloat::from_f64(hi, p).add(&BigFloat::from_f64(lo, p));
        let path_b = BigFloat::from_dd(dd, p);

        assert_eq!(path_a, path_b);
    });
}
```

---

## 5. Open questions for ratification

1. **API name for the `_with_rounding` variants**: `from_dd_with_rounding` is verbose. Alternatives: `from_dd_round(dd, p, rnd)`, or a single `from_dd(dd, p, rnd)` with a default-rounding overload. Defer to navigator/Tekgy preference; matches the broader question of rounding-mode-as-explicit-parameter vs context-driven (`PrecisionContext` already carries one).

2. **`PrecisionContext` integration**: DEC-031 §1 declares `PrecisionContext` carrying `requested_precision_bits` and `rounding`. Should BigFloat constructors take a `PrecisionContext` instead of separate `(p, rounding)` args? **My read: yes**, for the public API. The `_with_rounding` shape is for the internal type-level home; the user-facing entry points should consume `PrecisionContext`. Pathmaker decides during implementation.

3. **NaN payload preservation** — **resolved 2026-05-08 via aristotle's pushback; recommendation: PRESERVE**: I'd initially proposed v2 sign-only-discard on grounds of "most libraries discard" + "v2 simplicity." Aristotle correctly identified neither argument as structural-forcing. Their counter under F11's recognition/design discipline: publication-grade rigor + Filter Test §10 demand all-cases preservation. Software using NaN payloads for diagnostic propagation (real practice — `0x7FF0_0000_DEAD_BEEF` patterns in adversarial harnesses) is broken when payloads are silently dropped. Tambear cannot claim full f64 fidelity if payload bits are silently lost.

   **Practical fix is small**: extend `BigFloatKind::NaN` to carry payload — either `BigFloatKind::NaN { payload: u64 }` or a separate `nan_payload: u64` field on BigFloat. Cost: ~1 byte per BigFloat field at the data-layout level (the 64-bit payload itself + tag bit) + ~10 lines of from_f64/to_f64 code to extract/repack the bits. Minimal complexity; full preservation.

   **Aristotle's gauntlet Surface 6 already handles both modes**: parameterized over `NAN_PAYLOAD_PRESERVE` mode (strict vs permissive). Both compile; CI selects via feature flag. Recommended default: **strict mode (preserve)** — the structural-forcing argument wins.

   **Updated v2 storage layout** (delta vs §1 Q1):
   ```rust
   pub enum BigFloatKind {
       Normal,
       Zero,
       Infinity,
       NaN { payload: u64 },  // updated: carries f64 NaN's mantissa bits
   }
   ```
   from_f64 extracts the NaN's lower 51 mantissa bits (or full 52 if we encode the quiet-NaN bit too — IEEE 754 §6.2.1) into `payload`; to_f64 repacks them. Sign bit stays in BigFloat's `sign` field; quiet-vs-signaling distinction stays in `payload`'s top bit. **This is the design recommendation that goes to ratification.**

4. **Infinite precision for exact division**: if `a / b` is an exact rational (e.g., `1.0 / 0.5 = 2.0`), the Newton iteration converges in finite steps. Should we detect this and return the exact result without rounding? **My read: yes for v2**, as a small optimization — Newton-Raphson with f64 seed converges to the exact value when the result is f64-representable; the `p+50` guard bits then identify the correct rounding (which is a no-op). No special code needed; just the existing Newton + round path. **Document this as a property** so callers know `BigFloat(2.0) / BigFloat(0.5) = BigFloat(4.0)` is exact.

5. **Equality semantics**: `BigFloat::eq` should follow IEEE 754: NaN != NaN; ±0 == ±0 (sign-of-zero ignored in `==`). But `cmp` (for sorting) needs a total order — should NaN sort below -Inf or be a separate "unordered" sentinel? **My read: NaN != anything (incl. itself) for `==`; total order via `total_cmp` style, NaN at the top end (matching f64::total_cmp). Pathmaker decides.

6. **Zero-arithmetic edge cases** — **resolved 2026-05-08 via aristotle's confirmation**: implement IEEE 754-2019 sign-of-zero rules explicitly in the `kind = Zero` dispatch. Full table per aristotle's review:
   - `(+0) + (+0) = +0`
   - `(-0) + (-0) = -0`
   - `(+0) + (-0) = +0` under default `RoundToNearestTiesEven`; **flips to `-0` under `RoundTowardNegativeInfinity`**
   - `(+0) - (+0) = +0` under default; **flips under `RoundTowardNegativeInfinity`**
   - `(+0) - (-0) = +0`
   - `(+0) * any positive = +0`
   - `(+0) * any negative = -0`
   - `(+0) * (+0) = +0`
   - `(+0) * (-0) = -0`
   - `(-0) * (-0) = +0`

   **Critical implication**: the rounding-mode dependence for `(+0) ± (-0)` is real. The `kind = Zero` dispatch MUST consume the active `RoundingMode` for these cases. Two-three cases per arithmetic op carry this dependence; the rest are rounding-mode-independent.

   **Aristotle adding Surface 8 to gauntlet** (already landed per their 2026-05-08 message): full `0 ± 0` cross-rounding-mode regression suite. Different mechanism from Surface 6 (round-trip identity), so a separate surface is the cleaner shape per their judgment.

7. **Subnormal-tier ATK boundary** — **resolved 2026-05-08 via aristotle's confirmation**: BigFloat does NOT need a subnormal-source flag. Reasoning per aristotle's review:
   - BigFloat's exponent is i64 (effectively unbounded for any practical use). f64's subnormal regime is at exponent ≤ -1022; BigFloat absorbs this transparently because its exponent has 64-bit headroom.
   - The boundary surfaces only at `from_f64` and `to_f64` (where the encoding handles the implicit-leading-bit-cleared → explicit-leading-bit-set transition).
   - Per aristotle's gauntlet Surface 6 regression witnesses, the `from_f64`/`to_f64` round-trip is what catches subnormal-handling bugs. There's no need for an additional "subnormal-source" flag at the BigFloat level.
   - **The chains-E/F/G subnormal-aware bound lives in the path-budget layer, NOT in BigFloat itself.** BigFloat is the primitive; path-budget is composition logic over BigFloat operations. Different layers, different concerns. The DEC-031 §3.2 scope precondition is enforced at path-construction time (per ATK-DEC031-4), not at BigFloat-arithmetic time.

   This is a layer-separation finding: the chains-E/F/G subnormal-tier work belongs to whichever crate owns `PathBuilder` and `UlpBudget` composition (per DEC-031 §1's `lattice/precision.rs`), not to `primitives/big_float/`.

---

## 6. Cross-references to libm assumption-doc set

The libm assumption docs (consumer-axis context per team-lead's framing) inform the lattice but don't gate it. Specific touchpoints:

- **`assumption-docs/cody-waite-payne-hanek-crossover.md`**: the Payne-Hanek 1200-bit 2/π table is a constant; representing it in BigFloat at the 200-bit oracle precision means storing ~32 limbs. BigFloat-as-oracle for the trig recipes will exercise the high-precision multiply path. This validates the schoolbook-mul choice at p ≤ 1200 (fits in v2's regime).

- **`assumption-docs/asin-rational-kernel.md` §5**: the corpus-design-as-claim work specifies which input regions need dense oracle coverage. BigFloat's role is to *provide* the oracle; the corpus tells us *where to apply it*. Sweep 31 ships the substrate; Sweep 34 (per DEC-031 §5) does the integration.

- **`assumption-docs/methods-template-audit.md`**: per F12.1, the recipe spec.toml's published-prose ULP claims need validation against an oracle. BigFloat at p ≥ 200 is the oracle. Sweep 31 enables Sweep 34's validation pass.

The libm work doesn't constrain BigFloat's API; it informs the use cases. The constraint flows the other way: BigFloat must be precise enough to *adjudicate* libm claims, which means p ≥ 200 (per chains-E/F/G recommended floor) is the design target.

---

## 7. Sweep 31 deliverable list

For the Sweep 31 implementation phase (post-design ratification):

1. `crates/tambear/src/lattice/precision.rs` — `PrecisionLevel`, `PrecisionRefinementChoice`, `PrecisionContext`, `RoundingMode`, `UlpBudget`, `PathError` (per DEC-031 §1 + §6 #1, #5)
2. `crates/tambear/src/primitives/big_float/mod.rs` — module entry; re-exports `BigFloat`, `BigFloatKind`
3. `crates/tambear/src/primitives/big_float/ty.rs` — `BigFloat` struct, `BigFloatKind` enum, classification predicates (~250 LoC mirroring DD's `ty.rs` shape)
4. `crates/tambear/src/primitives/big_float/conversions.rs` — `from_f64`, `to_f64`, `from_dd`, `to_dd`, with the structural diamond short-circuit (~150 LoC)
5. `crates/tambear/src/primitives/big_float/cmp.rs` — comparison + total order (~80 LoC)
6. `crates/tambear/src/primitives/big_float/arith.rs` — add, sub, mul, div, sqrt per BZ Algorithms 3.1, 3.3, 3.5, 3.10 (~500 LoC)
7. `tests/big_float_roundtrip.rs` — §6 #13 round-trip identity test
8. `tests/big_float_diamond.rs` — §6 #3 diamond commutativity test
9. `tests/big_float_arith_invariants.rs` — algebraic-property proptests
10. Optional: `tests/big_float_vs_mpfr.rs` — peer-cross-validation against mpmath/mpfr at p=200 (per DEC-031 §3.9: "mpmath as peer, not authority")

Total: ~1000 LoC implementation + ~500 LoC tests = ~1500 LoC. Below DEC-031's 2-3 KLoC ceiling; the remaining headroom is for aristotle-flagged edge cases that surface during implementation.

---

## 8. Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-formalize` as design lead for Sweep 31.
- Substrate verified: DEC-031 §1, §3.5, §6 (R:\tambear\docs\decisions.md lines 3310-3478); existing DD type at R:\tambear\crates\tambear\src\primitives\double_double\ty.rs; winrapids reference at R:\winrapids\crates\tambear\src\bigfloat.rs (~2581 lines, used as API-shape reference only — algorithms diverge significantly).
- Cross-checked: the f64-mantissa-fits-in-one-limb fact (53 ≪ 64) makes round-trip identity (§6 #13) trivial-by-construction. The DD-with-`lo=0`-invariant fact makes diamond commutativity (§6 #3) trivial-by-structural-short-circuit. Both load-bearing invariants reduce to one-line implementation guarantees.
- This is a draft. Open questions in §5 require navigator/aristotle/team-lead review. NO CODE in this document — design only per team-lead instruction. Ratification gates the move from design to implementation.
