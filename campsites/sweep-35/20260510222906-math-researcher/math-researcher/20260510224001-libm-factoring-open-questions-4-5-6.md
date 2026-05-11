# libm-factoring-open-questions-4-5-6

Created: 2026-05-10T22:40:01-05:00
By: math-researcher

---

# Libm-factoring open questions #4, #5, #6 — math-researcher's walk

**Source**: `R:\winrapids\docs\architecture\tambear-libm-factoring.md` § "Open questions for math-researcher walk-through".

**Status**: Initial positions, written for naturalist/aristotle/pathmaker to consume. These are Phase C / Phase B framing questions, not blockers.

---

## Open Question #4 — The exp/log-side analog of the Periodic Table of Trig

### The question

Past-Claude's April 13 trig-table enumeration (5 rows × 6 columns × 3 precision tiers) hasn't been done for exp/log. What does the exp/log periodic table look like?

### Position

The exp/log table is **shaped differently** than the trig table — fewer "rows" (no quadrant family) but more "columns" (more named members + more reduction variants). The natural axes are:

- **Rows = Function-family** (the abstract operation):
  1. exp-family: `exp`, `expm1`, `exp2`, `exp10` (and `exp_minus`, exp(-x), as a sign-symmetry variant)
  2. log-family: `log`, `log1p`, `log2`, `log10` (and `log_inv`, log(1/x), as a sign-symmetry variant)
  3. Power-family: `pow`, `sqrt`, `cbrt`, `rootn`, `nth_root`, `exp_int_arg` (specialized integer exponent)
  4. Hyperbolic: `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`
  5. Inverse hyperbolic: `asinh`, `acosh`, `atanh`, `acoth`, `asech`, `acsch`
  6. Magnitude: `hypot`, `cabs`, `norm2_n` (n-vector L2 norm), `dist`
  7. Mixed log: `log1m`, `log_sigmoid`, `softplus`, `log_diff_exp(a, b) = log(e^a - e^b)`, `logaddexp(a, b) = log(e^a + e^b)`
  8. Gamma (a different shape; see open question #3)

- **Columns = Reduction-variant** (the complementary-argument transform — what gets factored out):
  1. None (standard input range, polynomial directly)
  2. ln(2)-scaled (Tang reduction `k·ln(2) + r`)
  3. ln(10)-scaled (Tang reduction `k·ln(10) + r`, for log10)
  4. Hypot-scaled (`(|a|, b/a)` form)
  5. Pi-scaled (only relevant for cross-family — trig vs exp/log don't share this)
  6. Integer-exponent special case (pow integer; exp2 integer; etc.)
  7. Subnormal-input boundary (different polynomial precision needed)

- **Precision tiers**: p=53 (f64), p=200 (BigFloat mid-tier), p=1024 (BigFloat full-tier).

### What this looks like as a 3D table

| Family | Members | Reduction variants per member | Kernel state |
|--------|---------|------------------------------|-------------|
| exp | exp, expm1, exp2, exp10 | ln(2)·k, ln(10)·k, none, integer-exp | ExpKernelState(k, r_dd, expm1_r_dd) |
| log | log, log1p, log2, log10 | frexp+ln(2)·k, ln(10)·k, none, x-near-1 | LogKernelState(k, f_dd, log1p_f_dd) |
| pow | pow, sqrt, cbrt, rootn | Integer-exponent special, half-integer, general | Compose ExpKernelState + LogKernelState |
| hypot | hypot, hypot3, normN, cabs, dist | None (the scaling IS the algorithm) | UnitVectorState(max_abs, ratios) — future |
| sinh | sinh, cosh, tanh, coth, sech, csch | None | Compose ExpKernelState (pulls exp(x), exp(-x)) |
| asinh | asinh, acosh, atanh, acoth, asech, acsch | None | Compose log1p / sqrt — recipe-tier |

**Cell count**: 6 families × ~5 members × ~4 reduction variants × 3 precision tiers = ~360 cells in the full table.

**Cell sparsity**: many cells are degenerate (e.g., `exp10` with `ln(2)-scaled` reduction is incoherent; pi-scaled doesn't apply to log10). Maybe 30-40% of cells are realized; the rest are empty / N/A.

### What's structurally distinct from trig

- **No quadrant axis.** Trig has 4-quadrant periodicity; exp/log have monotonicity.
- **More precision-mode axes.** Exp/log spans 1500+ orders of magnitude (10^-300 to 10^300 in f64), so the reduction-error regime is more pronounced. Trig has range reduction issues but the output is always in [-1, 1].
- **Pow and integer-exponent special case** is a discrete branch with no trig analog.
- **The hypot family is a "row" in exp/log but doesn't appear in trig** (atan2 is the closest analog; it lives with inverse trig).

### Practical implication for Sweep 35

Phase A ships: expm1 + log1p (one cell each, at p=53).
Phase B ships: ExpKernelState + LogKernelState (the "kernel state" column).
Phase C ships: exp/log/exp2/log2/exp10/log10/sinh/cosh/tanh/hypot/pow — covers most of the f64 row for 4-5 families.

**The naturalist's full periodic-table exercise can be deferred to a post-Sweep-35 doc.** It's not a blocker for implementation; it's a discovery + organization exercise. Sweep 36 candidate.

### Cross-implication for the catalog

The "exp/log periodic table" is the same shape as the trig periodic table at the *meta* level: rows = family, columns = reduction variant, depth = precision tier. The April 13 garden's framing already anticipated this. **The two tables stack** — a "full transcendental periodic table" has all of trig + all of exp/log in the rows, with reduction-variant columns that span both families (e.g., the pi-scaled column has trig entries but is empty in the exp/log rows; the hypot-scaled column is empty in trig).

**The deeper structural point**: the periodic table is the *recipe catalog*. Every named function is a (family, reduction-variant, precision-tier) cell. Every cell that exists is a recipe wrapper composing kernel-state + reduction-step. The table IS the implementation roadmap.

---

## Open Question #5 — Pi-scaled vs Tang-style reduction symmetry

### The question

For trig, pi-scaled (`sinpi(x) = sin(π·x)`) uses an exact reduction (`round(2x)`) that bypasses Payne-Hanek. Is there an analogous "exact reduction" trick for exp/log? Specifically: `exp2(integer)` and `log2(power-of-2)` ARE exact in float; do they constitute an "exp2-scaled" family analogous to "pi-scaled"?

### Position

**Yes** — `exp2` and `log2` constitute a "binary-scaled" family analogous to pi-scaled for trig. The exact-reduction trick is `frexp` for log2 and `ldexp` for exp2.

### The structural symmetry

| Property | Trig (pi-scaled) | Exp/log (binary-scaled) |
|----------|------------------|------------------------|
| Function | sinpi(x) = sin(π·x) | exp2(x) = 2^x; log2(x) = log(x)/ln(2) |
| Exact reduction | round(2x) — gives integer quadrant exactly | frexp(x) for log2: (k, m) where x = m·2^k; round(x) for exp2 integer x |
| Polynomial input | r = x - q/2 (in units where π/2 = 1) | f = m - 1 for log2; r = x - k for exp2 |
| Bypasses what? | Payne-Hanek 1200-bit reduction | The k·ln(2) multiplication and its rounding |
| Reconstruction | Quadrant fixup (1 of 4 cases) | k·1 + log2(m) = k + log2(m); 2^k · exp2(r) |

The symmetry is exact. For sinpi, you get to skip the costly reduction because (x mod 2) gives the quadrant directly. For log2 of a power-of-2, you get to skip the polynomial entirely because frexp(2^k) = (1.0, k+1), so log2(2^k) = k.

### Implication for Sweep 35

Phase C should implement:
- `exp2(x)` — for integer x, return `ldexp(1.0, x_as_int)` exactly. For general x, use the binary-scaled reduction and exp polynomial.
- `log2(x)` — for power-of-2 x, return the exponent from `frexp` exactly. For general x, use the existing log algorithm divided by `ln(2)` constant... wait, that's *wrong*. The right path is to do the same `frexp` reduction but skip the `/ln(2)` step (since the reduction is already in log-base-2 form). The polynomial computes log(m), and the result is `k + log(m)/ln(2)` — the `/ln(2)` is one final exact-constant multiplication.

**Open architectural question for pathmaker**: should `log2(x)` and `log(x)` share a kernel state at content-addressed (x, precision) level, or should they have independent caches? My position: **share**. The kernel state is `LogKernelState(k, f, log1p_f)` — that's the same regardless of whether the caller wants log(x) or log2(x). The post-kernel multiplication by `ln(2)^-1` (for log2) or `ln(10)^-1` (for log10) is a recipe-layer detail.

**Cross-reference**: this is the same pattern as sinpi sharing TrigKernelState with sin. The reduction differs at the boundary (round(2x) for sinpi vs Payne-Hanek for sin), but the polynomial kernel is the same. The output formula differs (k + log(m)/ln(2) for log2 vs k·ln(2) + log(m) for log).

### The trinity for completeness

Per past-Claude April 13: "every -1-suffixed function in libm is an instance of one meta-primitive."

The complementary-argument transform meta-primitive has TWO degrees of freedom: the *fixed point F* and the *group G*. The full set of variants:
- Fixed-point at 0, additive group: expm1, cosm1, sinh near 0, tanh near 0
- Fixed-point at 1, multiplicative group: log1p, log2(1+x), log10(1+x)
- Fixed-point at π·integer, additive group: sinpi, cospi, tanpi
- Fixed-point at integer power-of-2, multiplicative group: log2(2^k·m), exp2(integer+r)
- Fixed-point at e^integer, multiplicative group: log(e^k·m), exp(integer+r) (= the Tang reduction)
- Fixed-point on the diagonal {(a,a)}, scaling group: hypot, atan2 (per question #2)

**Total: 6 named transform-types so far. Each generates a column of the periodic table.** This is the right level of abstraction — the meta-primitive is parameterized by `(F, G)`, and each `(F, G)` choice generates a column of family-variants.

**Position**: yes, exp2-scaled / log2-scaled is a column. So is e^integer-scaled (the Tang reduction). So is 1-shifted (log1p / expm1). So is π-integer-shifted (sinpi). The April 13 frame extends.

---

## Open Question #6 — TrigKernelState's high/low decomposition for `r`

### The question

The trig state carries `r_hi` and `r_lo` (double-double representation of the reduced argument). The exp/log analog likely needs the same, but the precision requirements are different (Payne-Hanek needs ~1200 bits of 2/π for large arguments; Tang's k·ln(2) reduction needs fewer bits but the multiplier `k` can be large). What's the right precision contract for `ExpKernelState.r` at each tier?

### Position

**At p=53**: `r` must be DD (high+low). The current tambear `exp.rs` does Cody-Waite `r = (x - k·LN_2_CW_HI) - k·LN_2_CW_LO` — this stores the result as a single f64 but the *computation* preserves ~85 bits. **For the kernel state to be the shared intermediate, the DD pair must be stored, not collapsed.**

**At p=200** (BigFloat mid-tier): `r` is a BigFloat at p=200. `ln(2)` is carried at p=200+. No Cody-Waite split needed; the subtraction is exact at the working precision. The kernel state's `r_hi, r_lo` collapse to a single BigFloat field. (Or equivalently, the BigFloat itself is multi-limb, so "high/low" is structural.)

**At p=1024** (BigFloat full-tier): same as p=200 but with multi-limb `ln(2)` at >=1024 bits. Per Sweep 31 BZ unstub, the BigFloat machinery handles this.

### Concretely, the struct

```rust
pub struct ExpKernelState {
    pub k: i64,                              // integer part of x/ln(2)
    pub r_repr: PrecisionTaggedR,            // DD at p=53, BigFloat at p>=200
    pub expm1_r_repr: PrecisionTaggedExpm1R, // DD at p=53, BigFloat at p>=200
}

pub enum PrecisionTaggedR {
    P0F64(DoubleDouble),                     // f64 at extended precision via DD
    P1Extended(DoubleDouble),                // f64 + DD arithmetic
    P2BigFloat { value: BigFloat, p: u32 },  // arbitrary precision
}
```

The PrecisionTaggedR enum carries the precision-context fingerprint. The TamSession cache key is content-addressed by `(x_bits_at_session_precision, precision_context_tag_bytes)`. Same x at same precision context → same key → same cached state.

**F13.C antibody**: the cache key construction must include the precision-context tag bytes. A session running at p=200 pulling a cached state computed at p=53 would otherwise silently get a 53-bit-precision result in a 200-bit pipeline. The tag bytes are the precondition fingerprint per DEC-032's F13.C pattern.

### Precision-tier-aware sharing

The TamSession sharing has compatibility tags per the Tambear Contract item 3. Concretely for ExpKernelState:

```
sharing_check(cached_state, requesting_precision):
    if cached_state.precision_context_tag != requesting_precision.tag_bytes:
        return Incompatible  // recompute fresh
    if cached_state.x_bits != requesting_x.bits_at_precision(requesting_precision):
        return Incompatible
    return Compatible
```

A consumer at p=200 will get cache miss against a p=53 entry; recomputes fresh at p=200, registers the new entry under the p=200 tag bytes. The same x at multiple precision contexts spawns multiple cache entries — by design. Per holonomic-architecture.md, this is the content-addressed discipline: the cache key includes ALL parameter bytes (x AND precision), so the assignments fully determine the key.

### Why this is the right design

1. **Cache hits at the right granularity**: same x at same precision context → hit. Same x at different precision contexts → miss. No silent precision drift across consumers.
2. **F13.C antibody**: precision is non-defaulted at every kernel-state-construction signature.
3. **Holonomic compliance**: the key is a hash of the (parameter, value) bag, not of how the state was reached.
4. **TamSession-shareable per contract item 3**: the precision tag bytes are part of the IntermediateTag's assumption fingerprint.

### Open architectural question

**Is `k` a tier-dependent representation?** At p=53, k is an i32 (sufficient for ~1024-magnitude exp inputs). At p=1024, an extreme exp input like `exp(1e305)` might have k overflowing i32. **My position**: yes, k should be i64 at all tiers. The cost is 4 bytes per state; the safety is structural.

---

## Cross-references

- Question #3 (Gamma family / Lanczos) is genuinely outside the complementary-argument frame; it's a different meta-primitive. Past-Claude flagged it correctly as the exception. Sweep 36+ work; don't try to force it into the current frame.
- Questions #4, #5, #6 inform Phase B (ExpKernelState/LogKernelState design) and Phase C (recipe wrappers). The headline:
  - **Periodic table** is the recipe catalog organization; full enumeration is post-Sweep-35.
  - **Binary-scaled column** for exp2/log2 is real and should be implemented as part of Phase C.
  - **Precision-tier-aware kernel state** is non-negotiable for tambear's correctness model; the struct shape is dictated.
- All three positions are consistent with the libm-factoring frame; they generalize / extend it rather than contradict it.

## Sources

Same canonical references as the coefficient-verification doc (Tang 1989/1990/1992, fdlibm, Markstein, Muller). Plus:
- **Brisebarre, Lauter, Mezzarobba, Muller (2017)** — "Towards an Efficient Implementation of CORDIC Algorithms" — relevant for understanding the broader space of "exact-reduction tricks" the meta-primitive subsumes.
- **April 13 garden essays** (past-Claude) — the design substrate this walk extends.
