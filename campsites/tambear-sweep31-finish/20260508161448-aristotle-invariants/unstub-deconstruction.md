# Unstub Deconstruction — e2e8fb2 against the Load-Bearing Invariants

**Date:** 2026-05-08
**Author:** aristotle (tambear-sweep31-finish)
**Subject:** Phases 1-8 deconstruction of the multi-limb arithmetic
unstub committed at `e2e8fb2`, verifying it preserves the two
load-bearing DEC-031 §6 invariants — diamond commutativity (#3) and
round-trip identity at the tier boundary (#13) — at all `p ≥ 53`.

**Inputs:**
- `crates/tambear/src/primitives/big_float/arith.rs` (1397 lines as
  shipped at e2e8fb2)
- `crates/tambear/src/primitives/big_float/conversions.rs` (518 lines)
- `crates/tambear/src/primitives/big_float/limbs.rs` (kernel
  primitives the unstub depends on)
- My prior `dec031-invariants-deconstruction.md` (Phases 1-8 design-
  layer)
- My prior `silent-failure-proptest-gauntlet.md` (Surfaces 5/6/7
  particularly relevant)
- DESIGN.md §1 Q4/Q5 (the math-researcher's encoding decisions)
- IEEE 754-2019 §6.2.1, §6.3 (NaN payload + zero-arithmetic rules)

**Method:** I'm not re-running Phase 1-8 on the invariants themselves
(that work shipped as `dec031-invariants-deconstruction.md`). Here I
walk the **as-shipped implementation** through the Phase 1-8 lens,
asking: which assumptions does THIS code make? Which irreducible
truths from the prior deconstruction does it depend on? Where do the
two load-bearing invariants survive, and where could they break?

---

## What's at stake

Two invariants that must hold at all `p ≥ 53` regardless of operand
content or rounding mode:

- **DC (Diamond commutativity, §6 #3)**: `from_f64(x, p) =
  from_dd(DD::from_f64(x), p)` bit-exact for all f64 `x` and `p ≥ 53`.
- **RTI (Round-trip identity, §6 #13)**: `from_f64(x,
  53).to_f64() == x` bit-exact for all f64 `x` (modulo NaN payload
  policy — §5 Q3 ratification preserves payload, so it's bit-exact
  including NaN payload).

The unstub's job was to fill in the four BZ algorithms. It did. But
**arithmetic preserves these invariants only if the operations
themselves are correct AND if the canonical-form invariant of
BigFloat is maintained throughout**. If `add` produces a non-canonical
result (e.g., trailing zero limbs, top-bit-not-set), all downstream
comparisons break — including the comparisons used inside the
proptests that *test* DC and RTI.

So: the deconstruction below is split into two phases:

- **DC verification**: trace the implementation paths from `f64` and
  from `DD` to BigFloat(p), check they meet at the same value for all
  f64 classes.
- **RTI verification**: trace `from_f64 ∘ to_f64` for all f64 classes
  and verify bit-exact.
- **Arithmetic invariant preservation**: every BZ algorithm produces
  a canonical-form BigFloat (top-bit-set, correct precision_bits-many
  bits, correct kind tag).

---

## Invariant 1 — Diamond Commutativity, applied to the shipped code

### Phase 1 — Assumption Autopsy on the implementation

Reading `conversions.rs:436-460`:

```rust
pub fn from_dd(dd: DoubleDouble, precision_bits: u32) -> Self {
    if dd.lo == 0.0 {
        // Structural diamond commutativity short-circuit.
        return Self::from_f64(dd.hi, precision_bits);
    }
    assert!(precision_bits >= MIN_PRECISION_BITS_FROM_DD, ...);
    let hi_bf = Self::from_f64(dd.hi, precision_bits);
    let lo_bf = Self::from_f64(dd.lo, precision_bits);
    hi_bf.add(&lo_bf, RoundingMode::RoundToNearestTiesEven)
}
```

**Assumptions baked into this code** that diamond commutativity hinges on:

**A1 (lo=0 detection covers the f64-source case).** When DD was constructed
from a single f64, `DoubleDouble::from_f64(f)` produces `(f, 0.0)`. The
short-circuit `dd.lo == 0.0` MUST trigger for every such DD.
*Counter-question:* for `f = ±0`, `f = ±Inf`, `f = NaN`, does
`DoubleDouble::from_f64(f).lo == 0.0` hold? Need to verify. Critical
because if `from_f64(NaN)` produces a DD with non-zero lo (e.g., NaN
in lo), the short-circuit fails for NaN inputs, and the diamond
breaks for NaN.

**A2 (precision_bits >= MIN_PRECISION_BITS_FROM_F64 covers the
short-circuit branch).** The short-circuit calls `from_f64(dd.hi, p)`,
which asserts `p >= MIN_PRECISION_BITS_FROM_F64 = 53`. The from_dd
function's signature accepts any `p`. So `from_dd(DD{hi=f, lo=0}, p=20)`
panics with the from_f64 assertion message, not from_dd's. *Antibody-
question:* this is a panic-rather-than-silent-failure boundary; OK.
But the panic message says "from_f64 requires precision_bits >= 53"
which is correct attribution.

**A3 (from_f64 is a bijection on the f64 ↔ canonical-BigFloat-at-p
domain, modulo information).** Diamond commutativity REQUIRES this.
If `from_f64` produces non-canonical BigFloats (e.g., the implicit
leading 1 not in the right position for some f64 class), the
through-DD path that calls `from_f64(dd.hi)` and `from_f64(dd.lo)`
separately produces non-canonical inputs to `add`, and the result is
unpredictable.

**A4 (the lo=0 short-circuit's semantics matches the lo!=0 branch's
semantics when lo=0).** If we removed the short-circuit and ran the
full path with lo=0, the code would compute
`from_f64(hi, p).add(&from_f64(0.0, p), RoundingMode::RNE)`. Adding
zero to anything is identity (per `add` line 207-217: `if rhs.is_zero()
{ ... return out }`). So semantically the short-circuit is correct.
*The short-circuit is performance-only*, not correctness-critical
for lo=0 — both paths produce the same value. Good. (This is the
type-level commutative-square move from my Phase 5 of the prior
deconstruction: there's only ONE canonical embedding from f64 to
BigFloat, and the through-DD path reduces to it when lo=0.)

**A5 (the lo!=0 branch at p>=106 produces the exact mathematical
sum hi + lo).** The code calls `add` with RNE. For the addition to be
EXACT (no rounding, hence diamond-commutative-bit-exact), the
precision must be sufficient. DESIGN.md says p>=106 ensures hi+lo
has at most 106 significant bits and so fits without rounding. *The
implementation does NOT verify exactness at runtime* — it trusts the
assert and the BZ Algorithm 3.1 to produce the exact result when no
rounding fires. If BZ 3.1 has a bug at p=106 with two operands
differing by exactly 53 in exponent (which is the DD canonical
overlap boundary), the diamond breaks for non-zero-lo DDs.

**A6 (NaN propagation through `add` preserves payload of one
operand).** `add` line 144-161 propagates NaN from self if self.kind
is NaN, otherwise from rhs. *For the lo!=0 path* this means
`from_f64(NaN, p).add(from_f64(0.0, p), RNE)` produces the NaN with
self's payload. This is consistent with the lo=0 short-circuit which
returns `from_f64(NaN, p)` directly. ALSO consistent.

**A7 (DoubleDouble::from_f64 for NaN/Inf produces lo=0).** This is
A1 again but stated as a positive constraint. If `DD::from_f64(NaN)
= (NaN, 0.0)` exactly (which is the obvious encoding), the
short-circuit hits and we get `from_f64(NaN, p)` with full payload.
If DD's NaN handling sets lo to NaN as well or to some sentinel,
the short-circuit fails. **Need to verify in DD's source.**

### Phase 2 — Irreducible Truths against the implementation

**T1** (from prior deconstruction): f64 has a finite domain of 2^64
bit patterns, partitioned into normal/subnormal/±0/±Inf/NaN classes.
The shipped `from_f64` covers all classes (line 98-130 specials
dispatch + line 132-244 normal/subnormal). Visible from the code
inspection.

**T2**: For each f64 class, there exists a unique canonical-form
BigFloat representation. The code commits to this:

- **±0** → `BigFloatKind::Zero` with sign preserved (line 121-130)
- **±Inf** → `BigFloatKind::Infinity` with sign preserved (line 99-108)
- **NaN** → `BigFloatKind::NaN { payload: mant_field }` with full
  52-bit payload + sign (line 109-118)
- **Normal** → `BigFloatKind::Normal` with implicit leading 1
  restored as bit 52 of effective_mant, then placed at position
  (precision_bits - 1) % 64 (line 132-244)
- **Subnormal** → `BigFloatKind::Normal` after re-normalization;
  exponent absorbs the renormalization shift (line 151-181)

**T3**: For the lo=0 case, `from_dd ∘ DD::from_f64 = from_f64`
holds *by structural identity* (the short-circuit at line 437-440).
This requires only that `DD::from_f64(f).lo == 0.0` for ALL f64 `f`,
which is A1/A7.

**T4**: For the lo!=0 case at p>=106 with finite operands,
`from_dd(dd, p) = from_f64(hi, p) + from_f64(lo, p)` where the add
is exact. This requires:
- p>=106 ensures the magnitudes don't overlap-with-rounding
- `add` at p with two operands whose top-bit-positions differ by at
  most 52 produces an exact result (no rounding)
- BZ 3.1 implementation of `add` at p produces a canonical-form
  result with no precision loss when no rounding fires

**T5**: NaN payload preservation under DC for NaN inputs requires
either (a) the lo=0 short-circuit fires for NaN (A1/A7), or (b) the
lo!=0 branch's `add` produces the right NaN payload. (a) is the
expected path; (b) is a fallback if DD's NaN representation has
non-zero lo for some bizarre reason.

### Phase 3 — Reconstruction from Zero (10 paths to verify DC at unstub level)

Given irreducibles T1-T5, here are 10 paths through the implementation
to verify DC, ranging from trivial to ambitious-structurally:

**1. Direct case-by-case bit-pattern walk.** For each f64 class
({±0, ±Inf, NaN, Normal, Subnormal}) × each precision in the test
set ({53, 54, 65, 100, 106, 107, 200, 500, 1024}), call both
`from_f64(x, p)` and `from_dd(DD::from_f64(x), p)`, assert
canonical-form equality. This is what `Surface 5` of my gauntlet
prescribes; the implementation must pass it.

**2. Short-circuit-first invariant proof.** Verify that
`DD::from_f64(f).lo == 0.0` for ALL f64 classes (24 representative
patterns covering the partitioning). If this holds, DC for all f64
inputs reduces to the from_f64 path correctness. (This is the most
elegant: a single non-arithmetic invariant verification implies DC
for every f64.)

**3. Coverage by injectivity of from_f64.** If `from_f64` is
injective on its f64 domain (different f64 inputs produce different
canonical BigFloats), AND the short-circuit fires for all DD-source
inputs that came from a single f64, DC follows. The tests would
verify both halves separately.

**4. Test that lo=0 always-short-circuits.** A targeted unit test
that `from_dd(DD{hi: x, lo: 0.0}, p) == from_f64(x, p)` for all
representative x and p, regardless of x's class. This is a structural
test, not a value test. It would catch a future refactor that
eliminates the short-circuit.

**5. Type-system encoding.** Mark `from_dd_with_lo_zero` as a
sealed-trait method that mechanically calls `from_f64`. Then DC for
f64-sourced DDs is true by definition (the implementation IS
`from_f64`). The current code does this informally with the
short-circuit; making it structural would prevent future drift.

**6. Use the f64_path_eligible mechanism in conversions too.** The
arith.rs has `f64_path_eligible` that round-trips through f64 to
detect "is this BigFloat actually just an f64 in disguise?". Apply
the same idea to the through-DD path: if the input DD is
`f64_eligible` (lo=0 AND hi finite), short-circuit; else go through
add. This is what the current code does implicitly via lo==0.0.

**7. Property-based test from the f64 side.** Use proptest with a
strategy that generates f64 bit patterns, runs both paths, asserts
canonical-form equality. Surface 5 of my gauntlet is exactly this.

**8. Property-based test from the DD side.** Generate arbitrary DDs
(both lo=0 and lo!=0), run both `from_dd` and `from_f64(hi).add(from_f64(lo))`,
assert canonical-form equality. Catches correctness of the lo!=0
branch even when no f64 was the source.

**9. Cross-precision DC.** For each (x, p) tested for DC, ALSO test
DC at multiple p values: from_f64(x, p1), from_dd(DD::from_f64(x), p1),
from_f64(x, p2), from_dd(DD::from_f64(x), p2). All four should be
mutually consistent (the two from-f64 paths and the two from-dd paths
must agree pairwise at each p, AND the difference between p1 and p2
in each path must be only "more zero limbs" if p2>p1). This is the
tier-coherence test.

**10. Adversarial: tie-breaking edge cases.** For p ∈ (53, 106), the
lo=0 short-circuit hits cleanly. For p >= 106 with non-zero lo, the
add must produce the exact sum without rounding. Construct DDs whose
hi and lo are exactly at the DD-canonical-overlap boundary (lo =
ulp(hi)/2) and test at p = 106, 107, 200. The non-overlapping-by-DD-
invariant means hi+lo has at most 106 significant bits; the add at
p>=106 should be exact. **This is the test that catches a wrong
guard-bit calculation in BZ 3.1 at the DD source case.**

### Phase 4 — Assumption vs Truth Map (against the shipped code)

| Assumption | Status | Replaced/refined by |
|---|---|---|
| A1 (lo=0 detection covers f64-source case) | depends on A7 | Need DD::from_f64 inspection |
| A2 (precision threshold panics, not silent) | YES | from_f64 assert |
| A3 (from_f64 is canonical-form correct) | LIKELY YES | encoded inline lines 132-244; gauntlet Surface 6 catches violations |
| A4 (short-circuit and full path are equivalent at lo=0) | YES | add+0 is identity per code line 207-217 |
| A5 (lo!=0 add at p>=106 is exact) | DEPENDS | needs proptest from DD side |
| A6 (NaN propagation through add preserves one payload) | YES | code lines 144-161 |
| A7 (DD::from_f64 always produces lo=0) | UNVERIFIED in this session | go check DD impl |

### Phase 5 — The Aristotelian Move (here)

The conventional move would be: write proptest hooks for DC, run
1000s of iterations, fix any bugs discovered.

**The Aristotelian move on the shipped code:** the lo=0 short-circuit
is *the* implementation of the type-level commutative-square move I
proposed in the prior Phase 5. Diamond commutativity is achieved
*structurally*, not by computation. The implementation is correct
**because** the short-circuit collapses the through-DD path to the
direct path whenever the DD content has lo=0 — and this is the only
case that arises from f64 inputs.

**The deeper finding from inspection:** the implementation is one
short conditional branch from being **maximally explicit** about the
structural identity. Currently it relies on the comment ("Structural
diamond commutativity short-circuit") and the magic of the value
comparison `dd.lo == 0.0`. A future refactor could:

1. Add a debug_assert that documents the post-condition: when the
   short-circuit fires, the result MUST equal the lo!=0 branch's
   computation (proof of equivalence, runtime-checked in debug).
2. Extract `DoubleDouble::is_f64_eligible(&self) -> bool` as a method
   that tests `self.lo == 0.0`. Then `from_dd` reads:
   ```rust
   if dd.is_f64_eligible() {
       return Self::from_f64(dd.hi, precision_bits);
   }
   ```
   The named method makes the intent obvious to readers.
3. Surface the structural identity in the type system via a sealed
   trait or marker. (This is heavyweight; comment-level documentation
   is sufficient if backed by tests.)

The current implementation is correct; these are polish improvements
that would make the structural move more legible.

### Phase 6 — Recursive Challenge

What did Phase 5 silently assume?

**B1.** `DoubleDouble::from_f64(NaN).lo == 0.0`. If DD's NaN
representation has lo=NaN (some libraries do this to "double NaN"
the value), the short-circuit fails for NaN inputs and DC is
violated for NaN inputs.

**B2.** The lo!=0 branch's `from_f64(dd.lo, precision_bits)` works
for `dd.lo = ±0.0`. (Yes — but verify: zero through from_f64 with
p>=106 produces BigFloatKind::Zero, and `add(BigFloat, +0) = BigFloat`
which is the identity rhs.is_zero branch.)

**B3.** The `add` at p>=106 with hi-class-Normal and lo-class-Normal
produces an EXACT result, not a rounded one, when |lo| <= ulp(hi)/2
(the DD canonical invariant). The exactness depends on BZ 3.1's
canonicalize_and_round NOT firing the rounding path. Specifically:
the round_bit and sticky_bit fed into canonicalize_and_round must
both be zero in this case — IF they're zero, should_round_up returns
false (line 1031: "If no bits below LSB, the result is exact — no
rounding"). So we need: at p>=106 with the DD canonical pair, after
exponent alignment + integer add, ALL bits below the result's
precision_bits position are zero.

The DD invariant says |lo| <= ulp(hi)/2 = 2^(exp(hi) - 53). The
combined hi + lo has top bit at exp(hi) and lowest non-zero bit at
exp(hi) - 105 in the worst case (when lo is at its maximum
magnitude). So hi+lo has at most 106 significant bits. At p=106, the
result has exactly 106 mantissa bits and the integer add fits without
overflow. The round/sticky bits are zero by construction. **B3 is
true at p>=106.**

At p=106 specifically, we need exact-fit, not round-toward-zero.
What if hi+lo has *exactly* 106 significant bits and the top one is
at position 105? Top bit of result is at position 105, exactly
matches precision_bits-1 = 105. Canonicalize_and_round has shift=0,
no rounding, result is canonical. Good.

What if hi+lo has fewer than 106 significant bits (e.g., 53 bits if
lo=0)? Then top_pos < 105 and shift < 0; canonicalize_and_round
left-shifts the magnitude and absorbs round/sticky=0 cleanly. Top
bit lands at position 105. Good.

What if hi+lo has 106 bits but the LOW bit is at position 0 of the
buffer, meaning we have 0 round/sticky? Same as above. Good.

**The B3 verification is complete.** BZ 3.1 at p>=106 produces exact
sums for DD-canonical inputs.

**B4.** What about subnormal hi or subnormal lo? When hi is subnormal
in f64, `from_f64(hi, p)` re-normalizes (line 151-181), shifting the
mantissa to put the leading 1 at position 52. The exponent becomes
unbiased = -1022 - shift. Suppose hi = smallest subnormal = 2^-1074.
After re-normalization: shifted has bit 52 set (mant_field=1,
shift=51), so effective_mant=2^52, unbiased_exponent = -1022 - 51 =
-1073.

Wait — should be -1074 for the mathematical value 2^-1074. Let me
re-check the formula: at line 182, `(shifted, -1022_i64 - shift as
i64)`. For mant_field=1, shift=51, we get `unbiased = -1022 - 51 =
-1073`.

The numeric value is now `effective_mant * 2^(unbiased - 52)` =
`2^52 * 2^(-1073 - 52)` = `2^(52 - 1125)` = `2^-1073`. But the smallest
f64 subnormal is `2^-1074`. **Off by one!**

WAIT. Let me re-derive. The f64 subnormal value with mant_field=1
(only bit 0 set in the 52-bit mantissa field) is:
`(1 / 2^52) * 2^-1022 = 2^(-1022 - 52) = 2^-1074`. The smallest
positive subnormal is `f64::from_bits(1) = 2^-1074`. ✓

Now the BigFloat representation: BigFloat normal form value is
`(top_bit + low_bits) * 2^(exponent - precision_bits + 1)` where
top_bit is at position (precision_bits - 1) of the mantissa.

For precision_bits=53, the value is `effective_mant * 2^(exponent -
52)`. We have effective_mant=2^52 (only bit 52 set) and we want this
to equal 2^-1074. So `2^52 * 2^(exponent - 52) = 2^exponent`, and we
need `exponent = -1074`.

But the code at line 182 sets `unbiased_exponent = -1022 - 51 =
-1073`. That's off by one!

**Actually wait — let me re-read the comment at line 167-181 more
carefully:**

> "After shifting mantissa left by `shift`, the value is
> `shifted · 2^(-1074 - shift)`."

Hmm, that says `2^(-1074 - shift)` — with mant_field=1, shift=51,
the value is `2^52 · 2^(-1074 - 51) = 2^(-1074-51+52) = 2^-1073`,
not 2^-1074. **That's wrong!**

The original f64 value was `mant_field · 2^-1074 = 1 · 2^-1074 =
2^-1074`. After shifting mant_field left by 51 bits (to move bit 0
to bit 51), the integer becomes 2^51, NOT 2^52. **There's an off-by-
one in the comment too.**

Let me re-derive from scratch:
- `mant_field = 1` means bit 0 of mant_field is set.
- We want to shift it so that the highest set bit lands at position
  52 (where the implicit leading 1 of a normal f64 sits).
- Bit 0 → bit 52: shift by 52, not 51.
- After shifting: `effective = 1 << 52 = 2^52`, value still = mant *
  2^-1074 = 2^-1074.
- So `effective * 2^k = 2^-1074` requires `k = -1074 - 52 = -1126`.
- BigFloat unbiased_exponent (the BigFloat exponent for `1.bbb · 2^E`)
  has value `2^E` for the leading 1 at bit 52.
- effective = 2^52 → bit 52 is set, leading 1 carries place value
  2^E = 2^52 in unscaled, so as a fraction `1.0 · 2^E` = `effective /
  2^52 · 2^E` = `effective · 2^(E - 52)`.
- Set this equal to `2^-1074`: `2^52 * 2^(E-52) = 2^E = 2^-1074`,
  so E = -1074.

The shift formula at line 165: `shift = leading_zeros - 11`. For
mant_field = 1 (a u64 with only bit 0 set), `leading_zeros = 63`,
so `shift = 63 - 11 = 52`. ✓ shift=52, not 51.

Then `unbiased_exponent = -1022 - shift = -1022 - 52 = -1074`. ✓

**So the code is correct.** I made an arithmetic error in my Phase 6
draft. Let me re-check: at line 165 `let shift = lz - 11`. For
mant_field=1, lz=63, shift=63-11=52. Comment line 167 says "for
mant_field <= 0x000F_FFFF_FFFF_FFFF, leading_zeros >= 12". OK, that's
just the lower-bound on lz; for mant_field=1, lz=63 ≫ 12. shift=52.

Then `effective_mant = 1 << 52 = 2^52`, and unbiased_exponent =
-1022 - 52 = -1074. ✓ The smallest subnormal round-trips correctly.

**B4 stands: subnormals work correctly.** I just had to be careful
about the arithmetic.

### Phase 7 — Recursive Process (continue until stable)

Add B1 and the verified B3, B4 to the assumption pile. Re-run.

- **B1**: still need to verify. Let me check DD::from_f64.

Re-reading conversions.rs line 437: `if dd.lo == 0.0 { return
Self::from_f64(dd.hi, precision_bits); }`. For NaN input: `DD::from_f64(NaN)`
needs to produce `DD{hi: NaN, lo: 0.0}`. If it does, the short-circuit
fires and we get `from_f64(NaN, p)` with full payload preservation.

The DD::from_f64 implementation is at `crates/tambear/src/primitives/double_double/ty.rs`. I don't have
it in front of me — let me check.

(See attached B1-followup section for the verification.)

### Phase 8 — Forced Rejection

Force-reject DC at the unstub level. What if the diamond doesn't
commute on this implementation?

**Rejection 1 — short-circuit doesn't fire for some f64 class.** If
DD's NaN representation has `lo = NaN`, the short-circuit fails,
and we go through `from_f64(NaN, p).add(from_f64(NaN, p), RNE)`.
The `add` at line 144-152 propagates self's payload. Both operands
are NaN with the same payload (by symmetry), so the result has the
same payload. **DC still holds for NaN inputs even if short-circuit
fails**, provided DD's lo carries the same payload as DD's hi for
NaN inputs.

But: the obvious DD::from_f64 implementation sets `lo = 0.0` for
ALL inputs including NaN (because the canonical DD invariant `|lo|
<= ulp(hi)/2` is satisfied trivially when lo is zero). So this
rejection scenario is unlikely.

**Rejection 2 — `add` at p=106 with two from_f64-sourced operands
introduces rounding.** This breaks the lo!=0 branch's claim of
exactness. Verified false in Phase 6 B3 — at p=106 the integer add
produces an exact result with zero round/sticky bits.

**Rejection 3 — `from_f64` produces non-canonical results for some
edge case.** Subnormals re-normalize; tier boundaries near
MIN_POSITIVE could trigger the renormalization path. If the shift
calculation is off by one for the smallest subnormal, the result is
not canonical, and downstream operations produce wrong results.
Verified false in Phase 6 B4.

**Rejection 4 — DC depends on a property of `add` that fails for
some specific (hi, lo) pair where lo != 0.** This is possible if the
DD canonical invariant `|lo| <= ulp(hi)/2` is sometimes violated.
But if DD always satisfies its invariant, this case doesn't arise.

**Rejection 5 — DC fails for ±0 sign-of-zero.** When `f = +0`,
`DD::from_f64(+0)` should produce `DD{hi: +0, lo: +0}` (or the
canonical zero DD). The short-circuit triggers (lo == 0.0 is true
for both +0 and -0; +0 == -0 in Rust comparison). Then
`from_f64(+0, p)` produces `BigFloatKind::Zero{sign: false}`. ✓

Wait — but what about `dd = DD{hi: -0.0, lo: +0.0}`? The short-circuit
fires (`lo == 0.0` is true). We get `from_f64(-0.0, p)` =
`BigFloatKind::Zero{sign: true}`. ✓

What about `dd = DD{hi: +0.0, lo: -0.0}`? `lo == 0.0` IS true (since
-0 == +0 in Rust). Short-circuit fires. We get `from_f64(+0.0, p)`
= `BigFloatKind::Zero{sign: false}`. But mathematically hi+lo = +0
+ -0 = +0 (under default rounding). So DC holds.

**Rejection scenarios mostly evaporate.** The implementation is
robust *given* the DD::from_f64 contract holds (which I'm flagging
as B1-to-verify).

---

## Invariant 2 — Round-Trip Identity at Tier Boundary

### Phase 1 — Assumption Autopsy

Reading `conversions.rs:84-244` (from_f64) and `conversions.rs:277-396`
(to_f64).

**A1 (RTI requires no arithmetic between from_f64 and to_f64).** The
shipped to_f64's docstring (line 256-261) says: "For higher-precision
BigFloats whose mantissa has more than 53 significant bits, this
method rounds to nearest-ties-even per the IEEE 754 default."
**Note**: at p=53 (the tier boundary), the BigFloat from f64 has
exactly the f64 mantissa bits. There are no extra bits to round. The
truncation/RTE rounding doesn't matter. RTI is exact at p=53.

**A2 (the encode/decode is bit-shuffling, not arithmetic).** True at
p=53. For larger p, the encode pads with zero bits below; decode
truncates them. No precision lost on round-trip.

**A3 (the special-value tag dispatch covers all f64 specials).** ±0,
±Inf, NaN all dispatched at line 98-130 of from_f64 and line 278-301
of to_f64. NaN payload preserved bit-exact.

**A4 (subnormal renormalization is reversible).** from_f64 shifts
the f64 subnormal mantissa left to canonical form; to_f64 shifts it
back when biased_exp <= 0. Verified above in Phase 6 B4 that the
shift calculations are consistent.

**A5 (the to_f64 truncation matches the round-trip case bit-exact).**
At p=53, the BigFloat from f64 has its top 53 bits set per the f64
mantissa, with zero bits below. to_f64 reads exactly those 53 bits
back out. No rounding needed; truncation produces the same bits as
RTE rounding. RTI is exact.

**A6 (the encode for normal vs subnormal in to_f64 matches the
round-trip class).** When we did `from_f64(subnormal_x, 53)`, the
exponent stored in BigFloat is `unbiased < -1022` (in the subnormal
range). When `to_f64` runs on this BigFloat, line 357 computes
`biased_exp = unbiased + 1023 <= 0`, dispatching to the subnormal
branch (line 367-385). The subnormal branch shifts mantissa right
and packs as f64 subnormal. ✓

**A7 (the at-p=53 case has special protection in from_f64 and
to_f64).** At p=53, the encode places the f64's leading 1 at bit 52
of limbs[0] (n_limbs=1, top_bit_in_top_limb=52). Both shift branches
in from_f64 reduce to `effective_mant << 0 = effective_mant`. The
encode is bit-shuffle. to_f64's reverse shift is also identity at
p=53. ✓

### Phase 2 — Irreducible Truths against the implementation

**T1**: At p=53 the encode/decode is bit-shuffling without arithmetic.
RTI is bit-exact by construction.

**T2**: For each of the 5 f64 classes, the from_f64 → to_f64
round-trip preserves the bit pattern:
- ±0 → Zero → ±0 ✓
- ±Inf → Infinity → ±Inf ✓
- NaN → NaN{payload} → NaN with payload+sign ✓
- Normal → Normal{exponent, limbs} → Normal f64 ✓
- Subnormal → Normal{exponent < -1022, limbs} → Subnormal f64 ✓

### Phase 3 — Reconstruction (paths to verify)

The proptest at gauntlet Surface 6 IS the operationalization. Pin
the regression witnesses. Run on the shipped code.

The 1238-line `big_float_cross_precision.rs` test file is the result.
Cargo passes 22 + 4 ignored. RTI is verified at p=53 across the f64
class set including specials.

### Phase 4-7

Skip light pass — RTI is structurally simpler than DC. The Phase 6/7
work is in DC.

### Phase 8 — Forced Rejection

**Rejection 1 — RTI fails for NaN with non-zero payload.** Code line
113 stores `payload: mant_field`, which is the FULL 52-bit mantissa
INCLUDING the quiet-bit. to_f64 at line 297-299 reconstructs as
`F64_EXP_MASK | (payload & F64_MANT_MASK)`. The payload bits
preserved + the exponent all-1s + sign = exact NaN bit pattern. ✓

**Rejection 2 — RTI fails for the smallest subnormal `0x1`.**
Verified above in Phase 6 B4. ✓

**Rejection 3 — RTI fails for the largest subnormal `0x000F_FFFF_FFFF_FFFF`.**
For mant_field=0x000F_FFFF_FFFF_FFFF (52 bits all set), `lz = 12`,
so `shift = 12 - 11 = 1`. effective_mant = mant_field << 1 =
0x001F_FFFF_FFFF_FFFE (bit 52 set + bits 51..1 set). Wait, bit 0 is
0 because we shifted left by 1. effective_mant = 0x001F_FFFF_FFFF_FFFE.
The numeric value = mant * 2^-1074 = (2^52 - 1) * 2^-1074, the largest
subnormal. ✓ unbiased_exponent = -1022 - 1 = -1023. So the BigFloat
has exponent=-1023, limbs[0]=0x001F_FFFF_FFFF_FFFE.

In to_f64: top_bit_in_top_limb=52, shift=0, effective_mant=limbs[0]=
0x001F_FFFF_FFFF_FFFE. biased_exp = -1023 + 1023 = 0. Falls into
subnormal branch. shift_amount = 1 - 0 = 1. shift_amount=1 is < 64
and <= 52, so we shift. subnormal_mant = (effective_mant >> 1) &
F64_MANT_MASK = 0x000F_FFFF_FFFF_FFFF. ✓ Round-trip exact.

**Rejection 4 — RTI fails near the normal/subnormal boundary
MIN_POSITIVE = 2^-1022.** For f = MIN_POSITIVE: exp_field=1,
mant_field=0. The normal branch fires (line 183-189). effective =
0 | 1<<52 = 2^52. unbiased = 1 - 1023 = -1022. limbs[0] = 2^52.

In to_f64: biased_exp = -1022 + 1023 = 1 > 0, so normal branch fires
(line 391-397). stored_mant = effective_mant & F64_MANT_MASK = 0.
exp_packed = 1 << 52. bits = 0 | 0x0010_0000_0000_0000 | 0 =
0x0010_0000_0000_0000 = MIN_POSITIVE bit pattern. ✓ Round-trip exact.

**Rejection 5 — RTI fails at p=53 with non-RNE rounding mode.** RTI
isn't parameterized by rounding mode (the invariant is to_f64 with
default RNE only — see docstring). The from_f64 doesn't take rounding
mode either. The round-trip is exact regardless.

---

## Cross-cutting findings against the unstub

### Finding 1 — DC short-circuit is the critical site

The lo=0 short-circuit at conversions.rs:437-440 is **the
implementation of the type-level commutative-square move from my
prior Phase 5**. The diamond commutes by structural identity, not
computation. This is the design my deconstruction predicted; the
implementation lands it cleanly.

**One unverified dependency**: `DoubleDouble::from_f64(f).lo == 0.0`
for ALL f64 classes (the A1 / A7 / B1 chain). The short-circuit is
correctness-critical for NaN inputs (because the lo!=0 branch can't
preserve full payload through `add` cleanly — the payload-merge rule
uses self's payload, but the operands are constructed from f64 NaN
in the lo!=0 branch with `from_f64(NaN, p)`, both of which would
have the same payload).

**Recommendation**: add a debug-mode assertion that
`DoubleDouble::from_f64(f).lo == 0.0` is enforced as a documented
post-condition of DD's constructor, OR add a unit test that
exhaustively iterates f64 class representatives and confirms.

### Finding 2 — The unstub preserves canonical-form invariant

The `canonicalize_and_round` function (arith.rs:838-966) is the
guardian of canonical form. Every BZ algorithm routes through it
for the final round. It:
- Finds top bit of magnitude
- Shifts to align top bit at position (precision_bits - 1)
- Captures round/sticky bits during right-shift
- Applies rounding per requested mode
- Handles increment-overflow (e.g., 0xFFF...F + 1 → 0x100...0; top
  bit moves up, exponent++)
- Trims to exact precision_bits-many limbs

The canonical-form invariant is preserved end-to-end. **No silent
non-canonical forms can leak from the BZ algorithms.** This is what
makes downstream comparisons (== for DC verification, == for RTI
verification) reliable.

### Finding 3 — Newton iterations use RNE internally; rounding mode
applied only at final round

For `div` (BZ 3.5) and `sqrt` (BZ 3.10), the Newton iteration runs
at `p + 50` guard bits with RNE everywhere, and the user's rounding
mode is applied only in the final `round_to_precision` call. This
is **BZ §3.5/§3.10 standard practice** — round-to-nearest internally
preserves quadratic convergence; the final round to user precision
is the only rounding-mode-dependent step.

The 50-bit guard is sufficient per BZ §3.5: log2(precision_growth)
+ a few bits of slack. For p=1024, the guard precision is 1074 bits,
which gives ≈21 doublings from f64-seed — plenty of headroom.

**One concern**: if the user requests RoundTowardZero on a div, the
Newton-result at guard precision is a near-exact reciprocal. The
final round_to_precision uses RoundTowardZero to truncate. But
`round_to_precision` calls `canonicalize_and_round` with `(0, 0)`
for round/sticky — the input magnitude IS the integer; round/sticky
is captured by the right-shift inside canonicalize. RoundTowardZero
in `should_round_up` returns `false` always. So the magnitude is
truncated to the new precision. ✓

For RoundTowardPositiveInfinity: returns `true` if not negative.
This rounds toward positive Inf — the magnitude increments if the
sign is positive. ✓

### Finding 4 — f64 fast path is correct optimization, not correctness

`f64_path_eligible(bf)` round-trips bf through f64 via to_f64 and
from_f64; if equal, it's eligible. This means: only BigFloats that
are bit-exact representable as f64 take the fast path. For these,
hardware f64 arithmetic IS the BZ algorithm (the 53-bit mantissa
multiply IS f64 multiply, bit-exact under RNE). The fast path is
sound under RNE.

**For non-RNE rounding**, the f64 fast path is taken ONLY if the
hardware result happened to be exact (no rounding fired). The check
`sum - a.to_f64() == b.to_f64() && sum - b.to_f64() == a.to_f64()`
at arith.rs:442 detects this. If the f64 add was inexact, fall
through to multi-limb. **Caveat**: this check fires only for `add`,
not `mul` or `div`. `mul`'s f64-eligible non-RNE path (line 1085)
doesn't have an exactness check — it falls through to multi-limb
unconditionally for non-RNE. Slightly conservative but correct.

`div`'s f64-eligible non-RNE path also falls through unconditionally
(line 1163). Also conservative-correct.

### Finding 5 — NaN payload propagation in mul/div

Looking at arith.rs:296: `(BigFloatKind::NaN { .. }, _) | (_,
BigFloatKind::NaN { .. }) => unreachable!()`. The `unreachable!()`
fires only after the early payload-propagation guards at lines
257-274. The guards extract the NaN payload from whichever operand
is NaN and propagate it. **This is correct** — IEEE 754 §6.2.3
allows the NaN result of operations involving NaN operands to have
either operand's payload (most implementations propagate one of
them).

**One subtle thing**: when BOTH operands are NaN (e.g., `mul(NaN_a,
NaN_b)`), the early guard at line 257-264 fires for self and returns
`NaN { payload: a.payload }`. The b.payload is silently dropped.
This is consistent with IEEE 754; it's not a bug. **It does NOT
break DC** because for diamond-commuting paths the NaN payloads on
both sides match (both came from the same f64 NaN input).

### Finding 6 — The lo=0 short-circuit covers ALL f64 classes only
if DD::from_f64 is class-uniform

The lo=0 condition is necessary for DC for **every** f64 input. If
DD::from_f64(NaN) has lo != 0, DC breaks for NaN. **The verification
this work needs**: confirm DoubleDouble::from_f64 always sets lo=0,
including for special-value inputs.

This is the **B1 follow-up** I'm flagging. Need to check
`crates/tambear/src/primitives/double_double/ty.rs::DoubleDouble::from_f64`.

---

## B1 verification (post-spec)

Checked `crates/tambear/src/primitives/double_double/ty.rs:39-41`:

```rust
pub const fn from_f64(x: f64) -> Self {
    Self { hi: x, lo: 0.0 }
}
```

`DoubleDouble::from_f64` always produces `lo = 0.0`, for every f64 `x`
including ±0, ±Inf, NaN, subnormals, and normals. The lo=0 short-
circuit at conversions.rs:437-440 therefore covers ALL f64 classes.

**Diamond commutativity holds for every f64 input through structural
identity.** The B1 dependency is verified.

The unverified-state in the body of this document is now resolved:
DC is bit-exact for the entire f64 input domain at all p ≥ 53 (via
the short-circuit), and approximately bit-exact at p ≥ 106 for non-
f64-sourced DDs (modulo the bugs surfaced in tasks #8 / #9 / #10,
which affect arithmetic correctness but not DC for f64-source inputs).

---

## Phase 8 update — surfaced bugs cross-checked against DC and RTI

After completing the deconstruction, three bugs surfaced through
adversarial / math-researcher review (tasks #8, #9, #10). Cross-
checking each against the load-bearing invariants:

### Task #8 — cancellation/borrow underflow in normal_add_multilimb

The bug fires when `cmp_limbs(large_aligned, small_aligned) == Equal`
AND `(round_bit | sticky_bit) > 0`. The borrow branch unconditionally
subtracts 1 ulp from a zero diff, underflowing into all-ones.

**My Phase 6 B3 dismissed cancellation concerns at p>=106 with DD-
canonical inputs.** B3 was correct for that scope: with |lo| <=
ulp(hi)/2, the integer-add cannot trigger this case because cmp
will not be Equal (large has 53 leading bits set; small has at most
53 trailing bits, with a 53-bit gap between).

**The bug's repro pattern is OUTSIDE the DC scope I verified:**
- large at p=200 with all-ones mantissa, exp=0
- small at p=300 with mantissa = (large << 100) | 1, exp=0, sign=neg
- This pair is NOT producible from any f64-source through DD::from_f64.
- Therefore DC for f64-source DDs is unaffected.

**But the bug DOES affect correctness of arbitrary multi-limb add at
mismatched precisions.** Anyone constructing BigFloats via
from_raw_limbs (test code, EFT outputs from chained ops, future
sweeping operators) can hit this. **The bug is real and must be fixed
for general correctness, but it does NOT break DC at the §6 #3
invariant scope.**

### Task #9 — sign of exp_shift in newton_reciprocal scaled-seed

The unscale step `recip.exponent += exp_shift` should be `-=`. This
affects div correctness when b's exponent is far from f64 range.

**Doesn't touch DC or RTI directly** — both are conversion-layer
invariants. div is an arithmetic op downstream of the conversion
layer.

### Task #10 — NaN payload dropped in div

`div` calls `Self::nan(result_precision)` which produces a canonical
quiet NaN with payload=0. add/mul/sqrt all preserve the input
payload.

**Cross-check against DC**: if a user converts an f64 NaN to BigFloat
via two paths (f64 → BigFloat directly vs f64 → DD → BigFloat), then
divides the result by 1, the two paths produce the same NaN BigFloat
because the conversion layer preserves payload (the lo=0 short-circuit
fires before any div is performed). **DC at the §6 #3 invariant scope
is the conversion-layer invariant; the div op is downstream.**

But: if a user composes operations like `div(f64_nan, x)` and asserts
that two paths producing the same NaN-input commute through div, the
output payload is now zero regardless of input payload. The
"two paths produce the same payload-laden BigFloat" property survives
trivially (both produce canonical-zero-payload-NaN), but the
"payload identity is preserved through arithmetic" property fails for
div specifically.

**This is a real inconsistency with add/mul/sqrt, and the IEEE 754
publication-grade-rigor argument from my Invariant 2 Phase 8
Rejection 1 ("permissive mode loses payload bits") applies.** The
team ratified strict NaN payload preservation per DESIGN.md §5 Q3
on 2026-05-08. div should comply.

### Cross-cutting

None of the three surfaced bugs break the §6 #3 (DC) or §6 #13
(RTI) invariants at their canonical scopes. All three affect
arithmetic correctness in regimes adjacent to but distinct from the
load-bearing invariants:

- DC scope = "f64 → BigFloat through two paths must agree"
- RTI scope = "f64 → BigFloat(53) → f64 must round-trip"
- Bugs scope = arbitrary-multi-limb arithmetic (#8), extreme-magnitude
  div (#9), NaN payload through div (#10)

**The deconstruction's conclusion stands**: DC and RTI are preserved
at the unstub level by structural design (short-circuit + bit-shuffle).
The bugs are arithmetic correctness issues that pathmaker must fix
before Sweep 34 (oracle-grade verification) can rely on them, but
they do not retroactively invalidate the load-bearing invariants
because those invariants live at the conversion layer, upstream of
the arithmetic.

**The strongest cross-cutting finding**: structural-identity proofs
(like the lo=0 short-circuit) are robust under arithmetic bugs.
Because DC is achieved by *avoiding* the arithmetic path entirely,
arithmetic bugs in the lo!=0 branch don't propagate into DC for f64-
source inputs. **This is the type-level commutative-square move's
deepest payoff**: making the invariant true by structure means it's
preserved under arithmetic regression.

---

## Status

Phases 1-8 walked once on each load-bearing invariant against the
shipped e2e8fb2 implementation. The deconstruction surfaces:

1. **DC implementation matches the prior Phase 5 type-level
   commutative-square move.** Short-circuit at conversions.rs:437-440
   is the structural identity. ✓

2. **DC has one unverified dependency**: DD::from_f64 must always
   produce lo=0. Flag for verification. (B1 follow-up below.)

3. **RTI implementation is bit-shuffle at p=53.** Encode/decode
   reversible by construction. ✓ Verified across all 5 f64 classes.

4. **All 4 BZ algorithms preserve canonical-form invariant** through
   the canonicalize_and_round pipeline.

5. **One concrete polish recommendation**: extract
   `DoubleDouble::is_f64_eligible(&self) -> bool` to make the
   structural intent legible.

6. **One concrete antibody recommendation**: add a debug-assert or
   unit test that `DD::from_f64(f).lo == 0.0` for all f64 class
   representatives.

The implementation is correct. The two load-bearing invariants are
preserved at the unstub level. The remaining work is making the
structural identity more legible (polish) and adding the antibody
test for the DD::from_f64 contract (defensive).
