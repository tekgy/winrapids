# Sweep 38 gamma family — two F13 antibody gaps found during Sweep 35 survey

*Scout, sweep-35 session, 2026-05-10*

Both found by reading old tambear `gamma.rs` with the precision-context lens active.
Both are invisible at f64; both become wrong at BigFloat precision.

---

## Gap 1: Hardcoded 709.0 overflow threshold in `tgamma_strict`

Location: `R:/winrapids/crates/tambear/src/recipes/libm/gamma.rs`, line ~136

```rust
let lg = lgamma_positive(x);
if lg > 709.0 {
    return f64::INFINITY;
}
return lg.exp();
```

The `709.0` is the f64 exp overflow threshold (exp(709.78...) ≈ f64::MAX). At BigFloat
precision, the overflow threshold is much larger — depends on the number of limbs.

**Fix when porting to new tambear**: `if lg > max_exp_for_precision(ctx) { return INFINITY }`.
The threshold must be a function of the precision context, not a literal.

---

## Gap 2: Fixed Lanczos parameters g=7, n=9

Location: `R:/winrapids/crates/tambear/src/recipes/libm/gamma.rs`, constants section

```rust
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [ /* Pugh 2004, g=7 */ ];
```

The Lanczos approximation's error bound is `O((g/(x+g))^x / (2π) · e^{x-g} · sum_terms)`.
At f64 precision (53 bits), g=7 n=9 gives ~15 digits, which is enough.

At BigFloat precision (e.g., p=200 bits = ~60 decimal digits), g=7 n=9 is wildly
insufficient. You need to either:
- Use a much larger (g, n) pair with recomputed coefficients at higher precision, OR
- Switch to a different algorithm (e.g., Stirling with Bernoulli numbers) for large p.

Pugh 2004 (the coefficient source for g=7 n=9) gives coefficients for multiple (g, n)
pairs up to g=15 or so. For BigFloat, the algorithm needs to select (g, n) based on the
required precision — or derive Bernoulli-based coefficients at the required precision.

**Fix when porting to new tambear**: `optimal_lanczos_params_for_precision(ctx)` returns
the (g, n, coefficients) appropriate for the requested precision. At f64, this returns
the current hardcoded values. At BigFloat, it computes or loads pre-tabulated values for
larger (g, n).

---

## Shape-space classification (for recipe metadata design per naturalist's taxonomy)

Gamma's shape path:
- **Shape 1 (input-side): absent.** No input-domain rewrite needed for precision.
- **Shape 2 (output-side): absent.** Output doesn't have a cancellation problem.
- **Shape 3 (structural-rewrite): present at poles.** The reflection formula
  `Γ(x) = π / (sin(πx) · Γ(1-x))` handles x near negative integers — a structural
  rewrite for pole-proximity, not cancellation.

The "fixed point" for gamma is topologically different from log1p or erfc: gamma has
*poles* at negative integers (Γ → ∞), not cancellation loci (where f(x) → 0). The
coordinate system still applies; the values are different.

---

## Kernel-state consumption (orthogonal to shape classification)

Inside `lgamma_positive`, the Lanczos sum `ag` is computed and then `ag.ln()` is called.
That's a log call — an ExpKernelState consumer once Sweep 35 ships. The reflection formula
also calls `sin(πx)` — a TrigKernelState consumer.

Gamma sits at the intersection of BOTH prior kernel sweeps (trig and exp/log), consuming
both. The shape-position (pure Shape 3) is separate from the consumption graph.

---

*Flag for whoever leads Sweep 38: read this note before designing the new tambear gamma
recipe. The two F13 gaps are non-obvious until you look at the code with precision-context
in mind.*

---

## Pacing observation added at session close

*Scout, sweep-35 session close, 2026-05-10*

Task #12 (aristotle's pressure-test of ExpKernelState + complementary-argument-transform)
closed before the four-axis recipe metadata framework was fully articulated. The pressure-test
covered axes 1-3 (problem-topology, fix-shape, sharing-layer) but axis 4
(precision-parameter binding) was identified afterward via this gamma investigation.

**Implication for Sweep 38**: the cache-key schema from Phase B (task #5) may not include
axis-4 fields. When porting gamma to new tambear, the `LibmRecipe` trait and TamSession
registration will need to be checked for axis-4 coverage before assuming the cache key is
correct for BigFloat precision contexts.

The two concrete axis-4 cases in gamma are:
- `max_exp_for_precision(ctx)` replacing the hardcoded `709.0` overflow threshold
- `optimal_lanczos_params_for_precision(ctx)` replacing the fixed `g=7, n=9` Lanczos parameters

If Phase B's `ExpKernelState` carries a precision-context field (which it should — that's
the whole point of `(x_bits, precision_context)` content-addressing), then the axis-4 gap
is an omission in the trait's *documentation and enforcement* rather than its fundamental
design. Check whether there's an F13 antibody at the `LibmRecipe` trait that requires
precision-context-dependent coefficient selection to be explicit.

This is a pacing artifact, not a structural failure. Substrate density outpaced the
phase-boundary. The gap is named here so Sweep 38 doesn't reconstruct the reasoning from
scratch.
