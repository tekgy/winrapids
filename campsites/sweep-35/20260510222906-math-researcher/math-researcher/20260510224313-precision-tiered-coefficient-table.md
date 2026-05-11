# precision-tiered-coefficient-table

Created: 2026-05-10T22:43:13-05:00
By: math-researcher

---

# Precision-tiered coefficient table — for pathmaker, Phase A+B+C

**Purpose**: Cross-pollinates naturalist's axis-4 (precision-parameter-binding) finding with math-researcher's § Part 4 (precision-context tiers in coefficient-verification doc). The F13.C antibody enforces: polynomial coefficient sets are precision-context-dependent, and the contract must surface this at the signature where precision is first bound.

**Pre-read**:
- `~/.claude/garden/2026-05-10-the-three-shapes-of-complementary-argument.md` § "Scout's wrinkle: precision-dependent parameters" + § "What the four-axis framework documents"
- `R:\winrapids\campsites\sweep-35\20260510222906-math-researcher\math-researcher\20260510223502-expm1-log1p-coefficient-verification.md` § Part 4
- `R:\winrapids\docs\architecture\branch-cut-conventions.md` § F13.C "signature-level antibodies"

---

## The pattern

Every minimax polynomial, Tang-table, Pade rational, Chebyshev expansion, or asymptotic series has coefficients **calibrated to a target precision**. The published Q1..Q5 for fdlibm expm1 are calibrated to f64 (p=53); using them at BigFloat p=200 gives ~50-bit-precision results in a 200-bit pipeline. **This is a silent failure mode** unless the recipe contract surfaces precision-tier coefficient binding as an F13.C antibody.

**The antibody shape**:

```rust
// WRONG — silent precision drift:
fn expm1(x: f64, ctx: PrecisionContext) -> Result<x_type, _> {
    let q_coefs = &[Q1, Q2, Q3, Q4, Q5];  // hardcoded p=53 constants!
    ...
}

// RIGHT — coefficient set selected at signature, F13.C compliant:
fn expm1(x: T, ctx: PrecisionContext) -> Result<T, _> {
    let coef_set = ExpM1Coefficients::for_precision(ctx)?;
    // for_precision returns Q1..Q5 at p=53, Q1..Q10+ at p=200, ...
    // Returns error (not default!) if precision is unsupported.
    ...
}
```

**The contract**: every approximation primitive's signature takes a `PrecisionContext`; the coefficient lookup is keyed to it; the cache key bytes include the precision-context-tag bytes; an unsupported precision returns an explicit error rather than silently degrading.

---

## Per-recipe coefficient-set requirements

The table below is the precision-tiered roadmap for all recipes touched in Sweep 35 (Phases A, B, C) plus a forward-looking row for Phase D and Sweep 36+ inheritors.

### Phase A — `expm1` + `log1p`

| Recipe | Precision tier | Polynomial set | Degree | Source | Error bound |
|--------|----------------|----------------|--------|--------|-------------|
| expm1 | p=53 (f64) | fdlibm Q1..Q5 (Tang 1992) | 10 in r | s_expm1.c | < 1 ulp |
| expm1 | p=80 (extended) | Same fdlibm Q1..Q5 + DD arithmetic | 10 in r | Tang 1992 § 4 | < 1 ulp |
| expm1 | p=106 (DoubleDouble) | Same Q1..Q5 + DD arithmetic | 10 in r | Tang 1992 + Markstein § 6 | < 1 ulp |
| expm1 | p=200 (BigFloat) | Re-derived minimax at p=200 — Q1..Q10+ approx (mpmath taylor → Remez at target precision) | ~20 in r | Need re-derivation per § Action below | < 1 ulp at p=200 output |
| expm1 | p=500 (BigFloat) | Re-derived minimax at p=500 | ~50 in r | Need re-derivation | < 1 ulp at p=500 output |
| expm1 | p=1024 (BigFloat full) | Re-derived minimax at p=1024 OR direct Taylor at sufficient degree | ~100 in r | Need re-derivation | < 1 ulp at p=1024 output |
| log1p | p=53 | fdlibm Lp1..Lp7 (= existing LOG_COEFFS) | 14 in s | s_log1p.c | < 1 ulp |
| log1p | p=80/106 | Same + DD arithmetic | 14 in s | Same | < 1 ulp |
| log1p | p=200+ | Re-derived minimax per precision | Grow with p | Need re-derivation | < 1 ulp at target p |

**The Phase A scope shrinks**: f64 + DD versions ship with existing fdlibm coefficients. BigFloat tiers ship with re-derived coefficient lookup; the *infrastructure* is the contract — the actual high-precision polynomial generation can be deferred to a small Python+mpmath pipeline (per § Action below).

### Phase B — `ExpKernelState` + `LogKernelState`

The kernel state stores precision-tagged fields. Per the struct sketch in `libm-factoring-open-questions-4-5-6.md`:

```rust
pub struct ExpKernelState {
    pub k: i64,
    pub r_repr: PrecisionTaggedR,
    pub expm1_r_repr: PrecisionTaggedExpm1R,
}
```

Where `PrecisionTaggedR` is an enum keyed to the precision tier. The same struct, at a different precision tier, holds a different concrete representation. Cache key includes the precision-tier discriminant.

### Phase C — Recipe wrappers

| Recipe | Precision-tiered coefficients needed? | Source |
|--------|---------------------------------------|--------|
| exp | YES — refit per § coefficient-verification doc § Part 3 to use Tang/fdlibm P1..P5 rational at p=53 | fdlibm e_exp.c |
| log | YES — existing LOG_COEFFS is fdlibm Lp1..Lp7; verified at p=53 | fdlibm e_log.c |
| exp2 | NO additional polynomial; reuses ExpKernelState. Different reduction (`k = round(x)`) | — |
| log2 | NO additional polynomial; reuses LogKernelState. Final multiplication by `1/ln(2)` | — |
| exp10 | NO additional polynomial; reuses ExpKernelState. Pre-multiplication by ln(10) | — |
| log10 | NO additional polynomial; reuses LogKernelState. Final multiplication by `1/ln(10)` | — |
| sinh | NO additional polynomial; reuses ExpKernelState via `sinh = h(h+2)/(2(h+1))` where h=expm1_r | fdlibm s_sinh.c |
| cosh | NO additional polynomial; reuses ExpKernelState via `cosh = 1 + t²/(2(1+t))` where t=expm1_r | fdlibm s_cosh.c |
| tanh | YES at p=53 — tanh has its own polynomial for small \|x\| where expm1-based formula loses precision. fdlibm: 5-term Pade. | fdlibm s_tanh.c |
| hypot | YES (Shape 3) — fdlibm e_hypot.c thresholds (2^500, 2^-500, etc.) are tier-dependent | fdlibm e_hypot.c |
| pow | NO new polynomial; composes Log+Exp kernel states at DD precision | fdlibm e_pow.c |

### Phase D — Complex `clog`

`clog(z) = log(|z|) + i·arg(z)`. Uses LogKernelState for the real part. The imaginary part (arg/atan2) requires the existing atan2 polynomial — which is also precision-tier-dependent.

| Recipe | Precision tier | Polynomial set |
|--------|----------------|----------------|
| atan2 | p=53 | fdlibm s_atan2.c constants T1..T11 |
| atan2 | p=200+ | Re-derived minimax per precision |
| clog | All tiers | Compose LogKernelState (real part) + atan2 (imag part) + BranchPolicy machinery |

### Constants requiring precision-tier storage

Beyond polynomial coefficients, the following transcendental constants need multi-tier storage:

| Constant | p=53 | p=200 | p=1024 |
|----------|------|-------|--------|
| ln(2) | LN_2_DD (existing) | BigFloat p=200 | BigFloat p=1024 (multi-limb) |
| ln(10) | LN_10_DD (need) | BigFloat p=200 | BigFloat p=1024 |
| ln(2)·INV (= log₂e) | LOG2_E_F64 (existing) | BigFloat p=200 | BigFloat p=1024 |
| log₁₀(2) | New | BigFloat p=200 | BigFloat p=1024 |
| 2/π (Payne-Hanek) | IPIO2 66-word table (existing) | Already 1584 bits; covers p=1024 | Same |
| π/2 high/low | PIO2_1..3 + PIO2_1T..3T (existing) | BigFloat p=200 | BigFloat p=1024 |
| sqrt(2) | SQRT_2_F64 (existing) | BigFloat p=200 | BigFloat p=1024 |
| e | std::f64::consts::E | BigFloat p=200 | BigFloat p=1024 |

**Per-constant antibody**: each multi-tier constant lives in a `MultiprecisionConstant` struct exposing `value_at(ctx: PrecisionContext) -> Result<T, _>`. Single source of truth per constant; per-tier representation derived on demand or pre-computed at module init.

---

## Action plan for pathmaker

### Phase A immediate (don't block)

1. Ship `expm1` and `log1p` at p=53 ONLY using the fdlibm coefficients in the coefficient-verification doc.
2. **The signature MUST take `PrecisionContext`** as a non-defaulted parameter. If `ctx` is anything other than p=53 in Phase A, return `Err(UnsupportedPrecision)`. **This is the F13.C antibody.** Future tiers extend the implementation; the signature doesn't change.
3. The `ExpM1Coefficients::for_precision(ctx)` lookup is a single-entry table at Phase A. The lookup *exists* as a function; the table is what grows.

### Phase B follow-up

4. Add `PrecisionContext` discriminant to `ExpKernelState.r_repr` and `expm1_r_repr` fields.
5. Cache key includes precision-context-tag-bytes.
6. TamSession sharing-compatibility check (per Tambear Contract item 3) verifies precision-tag match before returning cached state.

### Phase C ongoing

7. Each new recipe wrapper takes `PrecisionContext`.
8. Each new wrapper's coefficient lookup goes through the same `for_precision` pattern.
9. Tanh at p=53: implement fdlibm's 5-term Pade explicitly.
10. Hypot at p=53: implement fdlibm e_hypot's thresholds; the scaling constants 2^500 / 2^-500 are precision-tier-dependent and need their own lookup table.

### Phase D + Sweep 36+

11. Complex log: BranchPolicy non-defaulted at signature.
12. Eventually re-derive minimax coefficients at p=200, p=500, p=1024 via Python+mpmath+Remez pipeline. Output goes into a `coefficients/` directory in the repo; the Rust lookup tables are auto-generated.

---

## Why this matters structurally (the meta-argument)

Naturalist's axis-4 finding plus my coefficient-verification doc converge on the same F13.C antibody pattern repeated at a hundred recipe sites:

> Every approximation primitive has a *precondition* that the polynomial coefficient set matches the working precision. The precondition is structurally testable but easy to violate by hardcoding a single precision's constants. The antibody is signature-level: non-defaulted precision context → coefficient lookup → fail loudly on unsupported precision.

**This is not just a per-function detail.** It's the F13.C antibody at the recipe-tier-as-a-whole. If pathmaker ships Phase A with hardcoded fdlibm constants and no `for_precision` lookup, the entire library has a precision-tier ceiling baked in invisibly. Sweep 36+ retrofitting is much more expensive than getting it right now.

**The structural pattern transfers to**:
- All polynomial-approximation primitives (Remez fits, Pade rationals, Chebyshev, Taylor truncations).
- All asymptotic series (Stirling for gamma, Pade for erfc, asymptotic for Bessel).
- All threshold constants (overflow thresholds, underflow thresholds, branch-selection thresholds — all tier-dependent).
- All reduction tables (the IPIO2 2/π table is precision-tier-agnostic at 1584 bits which covers up to p=1024 — but a smaller library aimed at p=53 only might use a 256-bit version; the precision-tier-aware version is more storage but structurally correct).

---

## Cross-references and convergence

- This doc closes the convergence between naturalist's axis-4 (precision-parameter-binding) and math-researcher's § Part 4 (precision-context tiers). Two roles arrived at the same F13.C antibody from different angles; this doc names the antibody as a recurring pattern, not a per-recipe detail.
- The doc extends my coefficient-verification doc (§ Part 4 of `20260510223502`) by tabulating per-tier requirements per-recipe.
- The doc extends naturalist's three-shapes garden entry (§ "Scout's wrinkle") by listing the specific tambear recipes where the antibody applies.
- For pathmaker: this is the precision-tier contract for Phase A signature design. Don't ship Phase A without `PrecisionContext` as a non-defaulted parameter, even if only p=53 is supported in the body.

## Sources

Same canonical references as coefficient-verification doc. Additionally:

- **`~/.claude/garden/2026-05-10-the-three-shapes-of-complementary-argument.md`** § "Scout's wrinkle" — naturalist's axis-4 finding.
- **`R:\winrapids\docs\architecture\branch-cut-conventions.md`** § F13.C — the structural antibody pattern this doc applies.
- **Sweep 31 BZ multi-limb unstub** — the prior pattern of precision-context-threaded BigFloat arithmetic; the same threading discipline applies to polynomial coefficient lookup.
- **`R:\winrapids\docs\architecture\internal-tameness-contracts.md`** — the audit pattern for new arithmetic sites. Every new polynomial-eval site is a candidate audit point.
