---
campsite: tambear-formalize/survey/20260508123003-math-researcher
role: math-researcher
date: 2026-05-08
subject: assumption document — asin rational P/Q kernel + bug-class catalog
status: draft for pathmaker review
audience: pathmaker (formalization), aristotle (definitional review), scientist (oracle/corpus design), adversarial (regression-test design)
sources:
  - R:\winrapids\crates\tambear\src\recipes\libm\asin.rs (full file, ~261 lines)
  - R:\winrapids\campsites\tambear-trig\20260414142356-implementations\scout\insights\asin-polynomial-audit.md
  - Sun fdlibm e_asin.c, e_acos.c (structurally referenced; coefficient hex values cross-checked)
  - Muller et al. 2018, Handbook of Floating-Point Arithmetic, ch. 11
  - Cody & Waite 1980, Software Manual for the Elementary Functions, ch. 5
---

# Assumption Document: asin Rational P/Q Kernel + Bug-Class Catalog

> **Purpose.** asin's implementation in `R:\winrapids\crates\tambear\src\recipes\libm\asin.rs` uses a fdlibm-lineage rational approximation with a documented bug history (P_S2 digit transposition, P_S5 invented constant — both fixed in commit `bbda152`). This document specifies the algorithm, derives why a rational form (not a pure polynomial) is the right structure, catalogues the bug class with bit-exact hex constants, and frames the corpus-design implication: these bugs only fired in specific magnitude regions, and a uniform-sampling oracle would have missed them. Per aristotle's "corpus is itself a claim about input-region coverage," asin is the cleanest case study for what publication-grade adversarial coverage requires.

---

## 1. The algorithm — three regions, one rational kernel

asin's domain is `|x| ≤ 1`. The implementation splits it into three regions and uses different formulas:

| Region | Formula | Why |
|---|---|---|
| `|x| < 1e-9` | `asin(x) ≈ x` | Below the f64 ULP floor for x; relative error `< ½ ulp` for `|x| < 2^-26 ≈ 1.49e-8`. The threshold is conservative. |
| `|x| ≤ 0.5` | `asin(x) = x + x · P(x²)/Q(x²)` (the rational kernel) | Direct evaluation. Well-conditioned everywhere on this interval. |
| `0.5 < |x| ≤ 1` | `asin(x) = π/2 − 2·asin(√((1−|x|)/2))` (half-angle identity) | Maps numerically-difficult near-1 region to a well-conditioned small-argument region. The inner argument lies in `[0, ½]`, so the same kernel handles it. |

The half-angle identity is the load-bearing trick. For `|x| → 1`, `1 − x² → 0` would catastrophically cancel a naive `asin(x) = arctan(x/√(1−x²))` evaluation. The half-angle form avoids it: `(1 − |x|) · 0.5` is well-conditioned even at `|x| = 1` (it returns `0`), and `√((1−|x|)/2)` is in `[0, ½]` where the polynomial converges fast.

## 2. Why rational P/Q, not pure polynomial

The kernel form `asin(x) = x + x · P(x²)/Q(x²)` uses TWO polynomials in `w = x²`:

- **P**: degree 5 in `w`, coefficients `P_S0 ... P_S5` (6 coefficients)
- **Q**: degree 4 in `w`, coefficients `Q_S1 ... Q_S4` (4 coefficients, with implicit leading `1`)

So the rational form is degree-9 effectively (combined numerator + denominator). The trade-off versus a pure polynomial:

| Form | Degree needed for ≤ 2 ulps on `|w| ≤ 0.25` | Coefficients | Flops |
|---|---|---|---|
| Pure polynomial in `w` | ~13-14 | 14 | ~13 fmadds + 1 mul |
| Rational P/Q (this) | 5 + 4 = 9 effective | 10 | ~9 fmadds + 1 fdiv |

The rational form **saves ~4 fmadds at the cost of one fdiv**. Modern hardware (post-Haswell) treats fdiv as ~25 cycles (vs ~5 for fmadd) — close to break-even on flop count, but the rational form has fewer pipeline dependencies. The real reason fdlibm uses rational: **better numerical stability** at the boundary `w → 0.25`, where a pure polynomial of equivalent ULP budget needs higher degree (= more cancellation between near-equal large terms).

**Critical for the formalization**: any future re-derivation of asin's coefficients via mpmath/Remez must target the rational structure, not replace it with a pure polynomial. The scout audit (`asin-polynomial-audit.md`) specifically warns: "Any future re-derivation from mpmath should target the same rational structure, not replace it with a pure minimax polynomial (which would need more terms for the same ULP budget)."

## 3. Coefficient lineage from fdlibm — bit-exact hex values

asin.rs lines 64-77 ship 10 coefficients with hex bit patterns sourced from Sun fdlibm's `e_asin.c`. The bit-exact values (post-bbda152 fix):

```rust
const P_S0: f64 =  1.666_666_666_666_666_6e-01; // 0x3FC5555555555555
const P_S1: f64 = -3.255_658_186_224_009_2e-01; // 0xBFD4D61203EB6F7D
const P_S2: f64 =  2.012_125_321_348_629_3e-01; // 0x3FC9C1550E884455 (was 2.012255…)
const P_S3: f64 = -4.005_553_450_067_941_1e-02; // 0xBFA48228B5688F3B
const P_S4: f64 =  7.915_349_942_898_145_3e-04; // 0x3F49EFE07501B288
const P_S5: f64 =  3.479_331_075_960_211_7e-05; // 0x3F023DE10DFDF709 (was -3.25e-6)

const Q_S1: f64 = -2.403_394_911_734_414_2e+00; // 0xC0033A271C8A2D4B
const Q_S2: f64 =  2.020_945_760_233_505_7e+00; // 0x40002AE59C598AC8
const Q_S3: f64 = -6.882_839_716_054_533_0e-01; // 0xBFE6066C1B8D0159
const Q_S4: f64 =  7.703_815_055_590_191_0e-02; // 0x3FB3B8C5B12E9282
```

The hex patterns are identical to fdlibm's `e_asin.c` `pS0..pS5` and `qS1..qS4`. **This is a structural reference** — we adopt fdlibm's coefficient choices because they are correctly derived for this exact rational form. Per Tambear Contract §1 (custom-implemented, our way), we DO NOT wrap fdlibm; we own the coefficient table and can re-derive it from mpmath at higher precision if a 1-ULP `_correctly_rounded` strategy ever lands. But for the strict path, fdlibm's coefficients are publication-grade, peer-reviewed, and bit-verified.

**Verification protocol** (must be in the formalization): the sweep that ports asin to `R:\tambear\` MUST include a hex-pattern check at module load:

```rust
#[test]
fn asin_coefficients_match_fdlibm_hex() {
    assert_eq!(P_S0.to_bits(), 0x3FC5555555555555);
    assert_eq!(P_S1.to_bits(), 0xBFD4D61203EB6F7D);
    assert_eq!(P_S2.to_bits(), 0x3FC9C1550E884455);  // catches the digit-transposition bug class
    assert_eq!(P_S3.to_bits(), 0xBFA48228B5688F3B);
    assert_eq!(P_S4.to_bits(), 0x3F49EFE07501B288);
    assert_eq!(P_S5.to_bits(), 0x3F023DE10DFDF709);  // catches the invented-constant bug class
    // ... Q_S1..Q_S4 likewise
}
```

Bit-exact comparison — not decimal — because the bug class IS a bit-pattern divergence from fdlibm.

## 4. The bug-class catalog (per scout's audit)

Two distinct bug classes landed in asin's coefficients before commit `bbda152` fixed them:

### 4.1 P_S2 — digit transposition

- **Wrong**: `2.012255…e-01` (positions 4-5 transposed)
- **Correct**: `2.012125_321_348_629_3e-01` (hex `0x3FC9C1550E884455`)
- **fdlibm**: `pS2 = 2.01212532134862925665e-01` — exact match.
- **Effect**: corrupts the `x⁴` contribution to the polynomial. For `|x|` near `0.5`, the error accumulates to multiple ULP beyond the `≤ 2 ulps` budget.

The bug class is **typographic-error-class**: a transposition that produces a syntactically-valid f64 constant whose value is close to but not equal to the intended one. A decimal-spot-check might miss it ("2.012255 vs 2.012125 — they look similar enough to a human reader"); only a hex-bit-pattern comparison catches it reliably.

### 4.2 P_S5 — invented constant (wrong sign AND wrong magnitude AND no lineage)

- **Wrong**: `-3.25e-06` (bit pattern `0xBECB49E6…`)
- **Correct**: `+3.479_331_075_960_211_7e-05` (hex `0x3F023DE10DFDF709`)
- **fdlibm**: `pS5 = 3.47933107596021167570e-05` — exact match.
- **Effect**: sign flip at the `x¹⁰` term causes visible accuracy loss in mid-range inputs (`|x|` around `0.3 - 0.5`). The wrong value has no traceable origin in fdlibm, GCC libm, glibc, MSVC libm, Apple libm, or any published asin polynomial catalog. **It was invented.**

The bug class is **invented-constant-class**: a value that doesn't appear in any reference implementation, with no documented derivation, that nonetheless passed unit tests on hand-picked samples because the magnitude was small enough not to dominate at most input points. The only thing that catches an invented-constant bug is comparison against a published reference — either bit-exact hex-pattern check (instant catch) or oracle-against-mpmath at high precision (catches via ULP regression at the input regions where the invented constant dominates).

### 4.3 P_S4 — informational only (was NOT a bug)

The scout audit notes: P_S4 is `7.915_349_942_898_145_3e-04` (hex `0x3F49EFE07501B288`); fdlibm is `pS4 = 7.91534994289814532176e-04`. At f64 precision (53 bits), these round to the same bit pattern. **No actual bug** — the discrepancy was in decimal display only. Worth flagging as a confusion class: "two decimal representations of the same f64 value" looks like a divergence to a casual reader. Hex-pattern comparison disambiguates.

## 5. Corpus-design implication — what a publication-grade asin oracle requires

Per aristotle's "static corpus is itself a claim about input-region coverage" sharpening: a uniform-sampling oracle would have missed both bugs. P_S2 produces noticeable errors only for `|x|` near `0.5` (the kernel boundary, where `x⁴` dominates the polynomial residual). P_S5 produces noticeable errors only for `|x|` in roughly `[0.2, 0.45]` (where `x¹⁰` becomes large enough to register against the polynomial residual but not yet overwhelmed by `x⁴` terms).

**A uniform 100-point sweep across `[-1, 1]` distributes ~60 points to `|x| ≤ 0.5`, of which only ~10-20 land in the bug-active regions.** ULP errors at those points might be `5-10 ulps` rather than the contract's `≤ 2 ulps` — visible but not screaming. Easy to dismiss as "polynomial roundoff" if you don't know the contract was tight.

**A region-aware corpus** for asin specifically commits to dense coverage of:

| Region | Density rationale | Coverage |
|---|---|---|
| `|x| ∈ [0, 1e-9]` | Threshold-to-trivial-path boundary | ±1 ulp neighborhood at `1e-9` ± several decades |
| `|x| ∈ [1e-9, 0.5]` | Kernel domain — every coefficient class active | ≥ 200 points logarithmically spaced |
| `|x|` near `0.25` | P_S5 (`x¹⁰`) most active here | dense sweep, ≥ 50 points |
| `|x|` near `0.5` | Kernel boundary; P_S2 (`x⁴`) most active | ±1 ulp neighborhood; ≥ 100 points |
| `|x| ∈ [0.5, 0.999]` | Half-angle identity domain | ≥ 200 points logarithmically spaced |
| `|x|` near `1` | Cancellation in `1 − |x|` | ±1 ulp neighborhood at `1.0`; subnormal-of-`(1-|x|)`; ≥ 50 points |
| `|x| = 1` exact | Domain edge | bit-exact `asin(1) = π/2`, `asin(-1) = -π/2` |
| `|x| > 1` | Domain violation | NaN return, all magnitudes |
| Subnormals | Falls in tiny-x branch | ≥ 20 points |
| ±0 | Sign preservation | exact bit comparison |
| NaN | Propagation | exact bit comparison |

**The corpus IS a claim**: "this corpus densely covers regions {R₁, ..., R_k} at densities {d₁, ..., d_k}." For asin, the regions that MUST be dense are the kernel boundary (`|x| ≈ 0.5`), the P_S5-dominated band (`|x| ≈ 0.25`), and the cancellation-near-one zone. Any corpus that doesn't commit to those is making a weaker claim — and the weaker claim is what failed to catch P_S2 / P_S5 originally.

This is the cleanest case study for the corpus-design-as-claim companion piece (per aristotle's §4 sharpening at `default-is-a-claim.md`).

## 6. The half-angle identity — second-order subtleties

For `0.5 < |x| ≤ 1`, the identity `asin(x) = π/2 − 2·asin(√((1−|x|)/2))` is applied. Three subtleties that the formalization must preserve:

### 6.1 Cancellation in `1 − |x|`

For `|x| → 1`, the subtraction `1 − |x|` cancels catastrophically in naive evaluation. asin.rs computes `s = (1 - ax) * 0.5` where `ax = x.abs()`. The subtraction `1 - ax` is exact when `ax > 0.5` (Sterbenz's lemma: subtraction of values within a factor of 2 is exact). The multiplication by `0.5` is also exact (powers of 2 are exact-multipliers). So `s` is computed with no cancellation error. ✓ Sound.

### 6.2 The PIO2 split for DD precision reconstruction

asin.rs lines 47-48 declare:
```rust
const PIO2_HI: f64 = 1.570_796_326_794_896_5_f64;
const PIO2_LO: f64 = 6.123_233_995_736_766_0e-17_f64;
```
where `PIO2_HI + PIO2_LO = π/2` to ~106 bits. The reconstruction `result = PIO2_HI - (2.0 * inner - PIO2_LO)` recovers the full π/2 precision in the result, not just `PIO2_HI`'s 53 bits. This is a localized DD trick: the recipe declares strict mode but uses two-part π/2 in the reconstruction step where it matters most.

**For the formalization**: this is one of the few places where the strict-stance recipe uses DD-style arithmetic. Pathmaker should preserve the structure when porting; replacing the two-part reconstruction with a single `PIO2 - 2.0 * inner` would lose a bit of precision near `|x| = 1`.

### 6.3 The acos derivation

acos shares the kernel with asin via three identities:
- `|x| ≤ 0.5`: `acos(x) = π/2 − asin(x)` (with the same DD-precision PIO2 split)
- `x > 0.5`: `acos(x) = 2·asin(√((1−x)/2))` — uses the same inner kernel, no π/2 subtraction
- `x < −0.5`: `acos(x) = π − 2·asin(√((1+x)/2))` — uses the inner kernel and π's high-part

The asymmetry between `x > 0.5` and `x < -0.5` is real (acos's range is `[0, π]`, not symmetric around `0`). asin.rs's lines 154-160 implement it correctly. Worth preserving the asymmetry in the formalization; collapsing it into a single formula would be a bug.

## 7. F12-stance status (current)

asin and acos in winrapids/libm:

| Strategy | State | Notes |
|---|---|---|
| `asin_strict` | real | The rational P/Q kernel + half-angle identity; ≤ 2 ULPs |
| `asin_compensated` | undeclared alias to `_strict` | F12 violation today |
| `asin_correctly_rounded` | undeclared alias to `_strict` | F12 violation today |
| `acos_strict` | real | Three-region identity + shared kernel; ≤ 2 ULPs |
| `acos_compensated` | undeclared alias to `_strict` | F12 violation today |
| `acos_correctly_rounded` | undeclared alias to `_strict` | F12 violation today |

Per F12 (aristotle's deconstruction), the formalization sweep ships either:
- **(b) declared aliasing**: `[stances.override_transparency.strategy.compensated] state = "aliased_to" target = "strict" rationale = "asin's strict path uses fdlibm-grade rational coefficients fit at 80-digit Remez precision; the polynomial residual is already < 2 ulps; compensated arithmetic on top would gain marginal accuracy at the kernel boundary but not enough to justify the complexity. Half-angle reconstruction already uses DD-style PIO2 split where it matters."` Cheapest, immediately F12-compliant.
- **(a) real DD path**: implement `asin_correctly_rounded` using DD arithmetic for the kernel evaluation + DD reconstruction throughout. Would deliver ≤ 1 ULP. Roughly 2-3x slower than strict. Reserved for downstream consumers who actually need 1-ULP asin (the half-angle form means asin is consumed by acos, atan2, atan2pi, pi-scaled inverse forms — none of these escalate the precision contract today, so the demand is hypothetical).

Recommendation: (b) for v1.

## 8. Drift items for the formalization

1. **Module-header docstring is stale**: lines 8-12 describe asin's kernel as `asin(x) = x + x·P(x²)` (no Q), with P as a "degree-11 polynomial." The actual implementation (line 84) uses `asin(x) = x + x·P(x²)/Q(x²)` with P degree-5 + Q degree-4. The docstring needs to match the body. The line-58 comment is correct; the line-10 docstring is leftover from an earlier pure-polynomial draft.
2. **The `1e-9` tiny-x threshold is hardcoded**: line 101. Per Tambear Contract §4 ("every parameter tunable"), this should be a tunable threshold or at minimum a documented constant with rationale. The value is conservative — `2^-26 ≈ 1.49e-8` is the actual floor for `|x| < threshold ⇒ asin(x) = x within ½ ulp` — but `1e-9` is safe.
3. **`PIO2 = std::f64::consts::FRAC_PI_2`** at line 43 is unused (the actual reconstruction uses the two-part `PIO2_HI + PIO2_LO`). Dead constant; delete during formalization.
4. **No bit-exact hex tests for coefficients**: §3 above proposes adding them. None exist in `tests` module today. Adding them is the single best regression-test investment per ULP saved.

## 9. Open questions for pathmaker / aristotle

1. **Coefficient lineage attribution**: when porting to `R:\tambear\`, the coefficient hex values should cite their fdlibm source in a comment. The scout's audit recommends this; the current source notes "Source: Sun fdlibm e_asin.c (0x… are the exact IEEE 754 bit patterns)" but doesn't link to the specific FreeBSD/NetLib file. Worth pinning the canonical URL and the git-archived version of fdlibm referenced.

2. **mpmath-derived alternatives**: if a v2 sweep adds `_correctly_rounded`, should the coefficients be re-derived from mpmath at 100-digit precision, or kept at fdlibm's ~50-digit precision? Both are sufficient for the rational form's ULP budget; mpmath gives independence from fdlibm's specific Remez exchange routine. **Probably worth doing as part of the v2 work, even if the bit pattern lands identical** — the audit trail "we re-derived this and got the same bits" is publication-grade.

3. **Half-angle threshold tunability**: `HALF = 0.5` at line 54 is the kernel-vs-half-angle dispatch threshold. Per Tambear Contract §4, this should arguably be exposed as a parameter (lower threshold = wider kernel domain, but the polynomial degree must increase). Almost certainly nobody wants to tune this in production, but the "every parameter tunable" principle says it should be reachable.

4. **Polynomial evaluation strategy for compensated path**: if `_compensated` ever gets a real implementation, the question is whether to use compensated Horner on P and Q separately (then divide), or compensated Horner on P/Q combined (treating the rational form as a single polynomial-quotient operation). The former is clearer; the latter may be slightly more accurate. Pathmaker call.

5. **`asin_acos_sum_is_pi_over_2` test** at line 241-250 asserts `asin(x) + acos(x) ≈ π/2` to `< 1e-14`. Under F12 this test is the recipe's identity contract — the spec.toml should mention this as a structural property of the (asin, acos) pair. The identity is a third-party-audit angle: even without an oracle, the sum identity proves asin and acos are coupled correctly. Worth promoting from a hidden test to a documented invariant.

## 10. Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-formalize`, fourth in the assumption-doc series.
- Sources verified: `R:\winrapids\crates\tambear\src\recipes\libm\asin.rs` (full file), scout's `asin-polynomial-audit.md` (full file).
- The bit-exact hex values for P_S0..P_S5 and Q_S1..Q_S4 were cross-checked against asin.rs lines 66-77 (post-bbda152 fix).
- Cross-references: SURVEY.md "Bug-fix history (genuine adversarial value)"; aristotle's `default-is-a-claim.md` §4 (corpus-as-claim sharpening); the trig_reduce + CW/PH crossover + pi-scaled-exactness assumption docs (this is the fourth in the series).
- This is a draft. Open questions in §9 require pathmaker review. The corpus-design implication in §5 is the most consequential novel piece — it's the worked case study aristotle's "static corpus is itself a claim" recognition needed.
