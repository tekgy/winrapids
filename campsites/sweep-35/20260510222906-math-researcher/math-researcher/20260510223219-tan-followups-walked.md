# tan-followups-walked

Created: 2026-05-10T22:32:19-05:00
By: math-researcher

---

# Tan-oracle follow-ups — math-researcher's walk (Sweep 35)

**Inputs**: `R:\tambear\oracle\tan\followups-rederived-2026-05-09.md` (re-derivation by main-thread on 2026-05-09), plus on-disk substrate at `R:\winrapids\crates\tambear\src\recipes\libm\{tan,sin,sincos}.rs` and `R:\tambear\oracle\tan\README.md`.
**Scope**: Walk each of the six re-derived follow-ups; ground in the existing implementation; recommend position for Sweep 35.

---

## Substrate state I verified before walking

**Existing tambear trig (`R:\winrapids\crates\tambear\src\recipes\libm\`):**
- `sin.rs` — full sin/cos kernel: `reduce_trig(x) → (q, r_hi, r_lo)` with Cody-Waite (3-round, PIO2_1..3, PIO2_1T..3T) for |x| < 2^20·π/2, then Payne-Hanek with a 66-word 2/π table for larger |x|. Polynomial kernels `kernel_sin`, `kernel_cos` use SIN_COEFFS S1..S6 and COS_COEFFS C1..C6 — described as Remez-refit in 80-digit mpmath, polynomial error < 1.11e-16 for both, "well below ulp(1.0)/2 ≈ 1.57e-16."
- `sincos.rs` — fused entry point; delegates to one `reduce_trig` call + two `eval_sincos` calls. Bit-identical to (sin_strict, cos_strict) per test.
- `tan.rs` — tan/cot/sec/csc all share `reduce_trig` and `kernel_sincos`. **tan uses the composed form `sin_k / cos_k`** with quadrant fixup; cos_k ≥ 1/√2 on `[−π/4, π/4]` so the division is always safe (no need for tan-specific reduction or reflection).

This means: **the TrigKernelState lattice from the libm-factoring frame already exists in code, just not named.** The shape `(q, r_hi, r_lo)` carries through `reduce_trig`; `(sin_k, cos_k)` is computed by `kernel_sincos`. Sweep 35's job for tan is *not* to rewrite this — it's to wrap it in a registered, content-addressed kernel-state via TamSession so the five trig wrappers stop recomputing redundantly across pipelines.

---

## Follow-up #1 — Asymptote-vs-zero precision-regime asymmetry

**Re-derived question**: Should a single ULP tolerance govern the whole tan domain, or should the harness dispatch on regime (abs-check near kπ, rel-check near (2k+1)π/2)?

**Walk**: The Sweep 34 oracle already commits the dual-path dispatch (`R:\tambear\oracle\tan\README.md` — commit `fb98777`, the near-singularity rel-error path symmetric to the near-zero abs-check). The README's empirical run on MSVC libm shows 49 near-singularity entries passing via rel-err at 1e-14, 49 near-zero entries via abs-check at 1e-14. **The dispatch is the principled answer**, not a workaround.

The deeper reason (which the README states cleanly): ULP-grid spacing is mismatched to meaningful precision at *both* magnitude extremes. Near zero, ULP spacing is far finer than meaningful precision (1 ULP at 5e-324 ≈ 5e-324). Near singularity, ULP spacing is far coarser than meaningful precision (1 ULP at 1.6e16 ≈ 2 absolute). Both regimes need a relative-error model.

**Recommendation**: When tambear-native tan lands, **preserve the dual-path dispatch as a contract**. The harness must:
- Apply abs-check (1e-14) for `|gold| < 16·EPS·1.0 ≈ 3.5e-15`
- Apply rel-check (1e-14) for `|gold| > 1e12`
- Apply ULP-check elsewhere

ULP distance is still reported diagnostically across all regimes — it just isn't the *assertion* metric where it's grid-mismatched. A degraded implementation can show ~3e13 ULP near singularity but only ~8% rel-error; without the rel-path, the assertion would be misclassified as a catastrophe-by-ULP that's actually catching a real defect.

**F13-shaped antibody check**: The dual-path dispatch IS the antibody. The rule "use ULP" has the precondition "|gold| is in the well-conditioned magnitude regime." Without the precondition test, the rule misfires both directions. The dispatch makes the precondition test explicit at the assertion site.

**Position**: NO new design work needed; preserve the existing dispatch. Sweep 35 inherits this directly.

---

## Follow-up #2 — Cross-quadrant sign correctness under large-k Payne-Hanek

**Re-derived question**: Does the existing tan corpus stress the sign-flip cases at large k, or only the *magnitude* cases near singularity? If only the latter, sign-correctness at large k is an untested invariant.

**Walk**: Reviewing `R:\tambear\oracle\tan\README.md` empirical findings: "**0 sign-flips at large magnitude**. The near-π/2 regime where tan output can flip sign between adjacent f64 inputs is handled correctly. MSVC's quadrant detection is bit-exact at the asymptote." Also: "0 ULP > 1 in the `reduced_arg_near_pi_half_k*` category (new per navigator's directive). Inputs constructed as `k·π + π/2_f64` for k ∈ {1, 5, 10, 100, 1000, 10000} all compute tan correctly. MSVC's Payne-Hanek reduction recovers BOTH the integer k mod 4 AND the fractional reduction r near π/2 for inputs up to k=10000 (i.e. x ≈ 31416)."

The MSVC reference has been validated; the corpus exists. But: **this is the empirical run; Sweep 35's tambear-tan needs the same demonstration on its own reduction.** Specifically:
- 355/113 (Brāhmagupta π approximation) — listed in the re-derivation as an adversarial input. Not currently a named corpus entry; only generic adversarial entries appear in `corpus.json`. Worth adding.
- `2^k · π/2 for k ∈ {20, 50, 100, 500, 1000}` — the corpus has `payne_hanek_pow2_k*` categories, the specific powers covered should be verified.

**Recommendation**:
1. Inherit the existing dual-quadrant + magnitude proptests.
2. **Add 355 as a named adversarial corpus entry** (Brāhmagupta/Saka rational approximation, |355 − 113π| ≈ 2.66e-5, so 355 is the closest small-rational to a multiple of π that hits very-near-asymptote behavior; tan(355) ≈ 3.01e-5 per the README's adversarial category).
3. Cross-validate tambear-tan against mpmath at exactly the `k·π + π/2_f64` inputs for `k ∈ {1, 5, 10, 100, 1000, 10000}` per the README's structural-test directive.

**F13.C antibody opportunity**: At the recipe signature, `reduce_trig` returns `(q, r_hi, r_lo)`. The implicit precondition "q is the correct quadrant mod 4 for x" is testable only by the corpus — there's no signature-level enforcement. Adversarial should examine whether a tambear-side wrapper could expose quadrant-as-output and fingerprint it independently of the polynomial evaluation, decoupling the two failure modes.

**Position**: Augment the corpus, no design change.

---

## Follow-up #3 — cot direct vs composed

**Re-derived question**: Does cot deserve its own implementation that goes through TrigKernelState directly (`cot = c/s` with explicit pole-vs-zero swap), or does it compose via `1/tan(x)` at the recipe-wrapper layer? The reciprocal pattern that broke Newton's reciprocal during Sweep 31 might bite cot at extreme x where tan(x) is at f64-overflow magnitude.

**Walk**: Looking at `tan.rs` (lines 115-129), **cot is already implemented as `cos_k / sin_k` with quadrant swap — NOT as `1/tan(x)`.** The existing code:

```rust
pub fn cot_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() { return f64::NAN; }
    if x == 0.0 {
        return if x.is_sign_positive() { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    let (q, r_hi, r_lo) = reduce_trig(x);
    let (sin_k, cos_k) = kernel_sincos(r_hi, r_lo);
    if (q & 1) == 0 { cos_k / sin_k } else { -(sin_k / cos_k) }
}
```

This is the *direct* form (the re-derivation's preferred design): cot pulls `(sin_k, cos_k)` from the kernel and applies the appropriate ratio with sign per quadrant. **The reciprocal-overflow class the re-derivation worried about cannot occur** because the ratio is computed at kernel-output magnitude (where both numerator and denominator are bounded by 1 in magnitude — `sin_k, cos_k ∈ [-1, 1]`), not at function-output magnitude.

**The deeper structural point**: the reflective swap (even/odd quadrant uses `cos_k/sin_k` vs `-sin_k/cos_k`) IS the kernel-state-as-shared-intermediate working as designed. The TrigKernelState carries `(sin_k, cos_k)` once; tan, cot, sec, csc each pick a *different* ratio of those two values with different quadrant rules. The April 13 garden's "five functions from one kernel state" frame is operationalized in `tan.rs`.

**Recommendation**:
- **Keep the direct form**. The composed form (`1/tan(x)`) would actually be *worse* numerically because tan(x) near singularity can be at f64-overflow magnitude where the reciprocal underflows lossily, while the direct form is bounded.
- The same answer applies to sec (`1/cos_k`) and csc (`1/sin_k`) — both already use the direct form per `tan.rs`. The "tan analog of hyperbolic-inverses asinh/acosh/atanh" question raised by the re-derivation transfers: any inverse-hyperbolic recipe should likewise pull from a kernel state directly, not compose by inversion.
- **Document the precondition for cot**: at extreme x where `reduce_trig` returns r near 0 (i.e., x near kπ), `sin_k` is near zero. Cot then returns a large but finite value (≤ 1.6e16 magnitude class), never ±∞. This is correct IEEE behavior for `cot(kπ_f64)` — the f64 input is approximate, the true mathematical value at the approximate input is large-but-finite.

**Position**: The implementation already answers the question correctly. Sweep 35 inherits the design.

---

## Follow-up #4 — Variants (tanpi, tand, atan2)

**Re-derived question**: When tambear-native tan lands, what's the oracle-corpus extension shape for the variants? Each is a separate Sweep-34-style harness, or each gets a test category within the tan harness?

**Walk**: There are two structurally distinct cases:

**tanpi(x) = tan(π·x)** and **tand(x) = tan(x·π/180)**: these are pure complementary-argument transforms on the *reduction* step. Per April 13 garden: "the precision gap between sin(π·x) and sinpi(x) lives entirely in the reduction step, not the polynomial." For tanpi, the reduction is `q = round(2x)` (exact for integer half-x), so it bypasses Payne-Hanek entirely. For tand, the reduction is `q = round(x/90)` (exact for x at 90° multiples).

The existing tambear code has `pi_scaled.rs` in `libm/` — so the pi-scaled family is being built out. The structural answer: **tanpi and tand share the same kernel polynomial as tan**, only the reduction differs. They should be **recipe wrappers** that compose `reduce_trig_pi_scaled` (or `reduce_trig_degrees`) → `kernel_sincos` → ratio. Same kernel state cache key (content-addressed by `(r_hi_after_reduction, r_lo, q)`), with a different reduction function.

**atan2(y, x)**: this is structurally different. It's an *inverse* function, not a forward one. Its corpus is the 17-entry edge case table per April 13 garden — the special cases at (0, 0), (±0, ±0), (±∞, ±∞), etc. atan2 belongs in the `atan.rs` module's family, not in tan's.

**Recommendation**:
- **Test category within tan's harness** for tanpi/tand. They share kernel state, share validation infrastructure, share the polynomial. The corpus should have `tanpi_integer_half`, `tanpi_near_pole`, `tand_at_90s`, `tand_near_pole_in_degrees` categories.
- **Separate harness for atan2**. Different domain (R² → R), different edge-case shape, different reduction (sign-of-(y,x) for quadrant placement on unit circle). atan2 lives with atan.

**Open question for naturalist**: the April 13 garden's "Periodic Table of Trig" enumeration distinguishes "pi-scaled" as a column variant. Is the right structural diagram: rows = function family (sin, cos, tan, cot, sec, csc, sincos, hypot, ...) × columns = reduction variant (none, pi-scaled, degrees, half-angle, ...) × precision tier? If yes, the cell `(tan, pi-scaled, p=53)` is the recipe `tanpi(x: f64) → f64`. Aristotle and naturalist should pressure-test whether the 3D table extends cleanly when the exp/log family lands (most cells empty? some cells redundant?).

**Position**: Test category for tanpi/tand; separate harness for atan2.

---

## Follow-up #5 — Continued-fraction vs polynomial near singularity

**Re-derived question**: Per the complementary-argument transform meta-primitive, what's the right "fixed point" for tan-near-π/2? The reflection identity (`tan(π/2 - x) = cot(x) = 1/tan(x)`) re-centers the input at the singularity, then 1/tan(small) is precision-safe. Sweep 35 implementation choice: use reflection adaptively, or throughout?

**Walk**: The existing tambear tan does **not** use the reflection identity at the polynomial level. It uses `reduce_trig(x)` to land r in `[-π/4, π/4]`, then computes `sin_k(r), cos_k(r)` via the kernel polynomials, then takes the ratio with quadrant fixup. The "near-singularity" case is handled implicitly: when x is near π/2, `reduce_trig` returns `r` near 0 (after the quadrant flip), and the kernel sin/cos polynomials evaluate well at small r. The ratio `cos_k/sin_k` (the odd-quadrant case) is then well-conditioned because sin_k near 1 and cos_k near 0 are both at full kernel precision.

**This is the right design.** Reflection-at-the-recipe-layer would be redundant — the existing range reduction already lands every x in the well-conditioned r-range. The kernel polynomials *are* the precision-safe core for both regimes. The dual-path is in the reduction (Cody-Waite below 2^20·π/2, Payne-Hanek above), not in the polynomial.

**The complementary-argument transform meta-primitive frame still applies, but at a different layer than the re-derivation suggested**:
- F (fixed point) = π/2·integer (the integer multiples of π/2)
- G (group structure) = additive-with-period-π/2
- Transform = `r = x - q·π/2` (this IS the reduce_trig output)
- Stable evaluation = `kernel_sin(r), kernel_cos(r)` (both well-conditioned for r ∈ [-π/4, π/4])
- Inverse transform = quadrant fixup that picks the right sin↔cos swap and sign

The reflection identity is *equivalent* to the quadrant fixup for the specific case of x near π/2 — the existing fixup table handles q=1 by mapping sin_k → cos_at_target and -cos_k → sin_at_target, which IS the algebraic content of the reflection identity. The framework subsumes the implementation choice.

**Recommendation**: NO continued-fraction. NO explicit reflection identity. The existing `reduce_trig + kernel_sincos + quadrant_fixup` design implements the complementary-argument transform at the right layer. Sweep 35 should preserve this and not add a redundant reflection path.

**Position**: Existing design is correct; explicit reflection adds nothing.

---

## Follow-up #6 — Shared kernel state vs tan-specific

**Re-derived question**: Is the composed form (tan = s/c via TrigKernelState) precision-safe enough in the near-singularity regime, or does tan-near-π/2 require its own kernel-state computation that produces tan directly?

**Walk**: Empirically, the Sweep 34 oracle answers this for MSVC: tan is ≤1 ULP everywhere, no degradation near the asymptote. The MSVC implementation almost certainly uses the same composed form (it shares reduction across sin/cos/tan via fdlibm's `__ieee754_rem_pio2`). So the composed form IS precision-safe enough at f64 — the empirical evidence is strong.

**The deeper argument**: in the composed form, the precision of the result is bounded by:
- `ulp(sin_k)` and `ulp(cos_k)` — both at kernel polynomial precision (~half-ulp per the README's polynomial-error bound 1.11e-16)
- ulp of the division — one ulp at the f64 output magnitude

For x near π/2, sin_k is near 1 (full precision) and cos_k is near 0 (low magnitude but exact at kernel precision because reduce_trig + Cody-Waite preserves bits). The ratio cos_k/sin_k = small/large is well-conditioned. The output magnitude can be very large (~1.6e16 at f64-π/2), but the relative error stays at kernel precision because the ratio captures it correctly.

The tan-specific path would only help if the kernel polynomials lost precision near the corners of their interval — but Remez-refit polynomials on [-π/4, π/4] are designed to NOT lose precision at the interval endpoints. The polynomial-error bound 1.11e-16 holds *uniformly* across the interval, not just in the middle.

**Recommendation**: **Confirm the composed form for tan in TrigKernelState**. Tan does not need its own kernel state. The five functions (sin, cos, tan, cot, sec, csc) all derive from one `(q, r_hi, r_lo) → (sin_k, cos_k)` kernel evaluation per x.

**The implication for ExpKernelState (Phase B)**: this informs the analogous design question for the exp/log family. The April 13 garden's `ExpKernelState = (k, r, expm1_r)` carries the precision-safe base form `expm1_r = expm1(r)`. Every exp/log family member should compose FROM expm1_r, not have its own kernel state. This is the parallel to the TrigKernelState answer.

**Position**: Composed form is correct for tan AND for all exp/log family members. ExpKernelState should follow the same content-addressed-shared-intermediate pattern.

---

## Cross-references and summary

| # | Topic | Position |
|---|-------|----------|
| 1 | Asymptote-vs-zero precision-regime | Preserve dual-path dispatch as contract. No new design. |
| 2 | Cross-quadrant sign at large k | Add 355 + named `payne_hanek_witness_k_*` entries to corpus. |
| 3 | cot direct vs composed | Direct form already implemented. Preserve. |
| 4 | tanpi/tand/atan2 variants | Test category within tan harness for tanpi/tand. Separate harness for atan2. |
| 5 | Continued-fraction vs polynomial near singularity | Existing reduce_trig + kernel + fixup is the complementary-arg-transform at the right layer. No explicit reflection. |
| 6 | TrigKernelState vs tan-specific | Composed form is correct. Tan does not need own kernel. Same pattern transfers to ExpKernelState. |

**Biggest cross-implication**: follow-up #6 transfers structurally to Phase B (ExpKernelState design). Every exp/log family member composes from `expm1_r` (and `log1p` for log direction). No member gets its own kernel-state computation. The TamSession registers ONE intermediate per (x, precision_context), and all consumers pull from it.

**Single concrete corpus action** (for scientist or pathmaker to pick up): add 355 as a named adversarial entry to `R:\tambear\oracle\tan\generate_corpus.py`. Five-minute change; preserves the "we test the cases that would fail if the math were wrong" discipline.
