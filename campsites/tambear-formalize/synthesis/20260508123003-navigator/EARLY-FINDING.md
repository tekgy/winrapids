# Early Finding — Navigator Survey-in-Progress

**Date:** 2026-05-08  
**Status:** Preliminary — agents still surveying; this records what navigator found independently

---

## The Landscape Picture (as of 2026-05-08)

### What's in R:\tambear (the formalized library)

Sweeps 0-7 complete (op-redesign through experiment-harness). 1397 lib tests, 2027 total.

Recipe catalog in `R:\tambear\crates\tambear\src\recipes\`:
- Time series: ewma, ar1, ar2, garch, biquad, euler_maruyama
- Quantiles: kll, ddsketch, tdigest, gk, quickselect, quantile_exact
- Statistical: cvar, hill_estimator, pca_eigenvalues, standardize_per_dimension, shannon_entropy
- Music theory: cents_conversions, equal_temperament, key_inference, temperament_residuals, tonnetz
- Physics: kernel_greens_function
- ML: mamba_selective_scan
- Dimensional Nyquist family (collatz, compression, overtone, shape_space)
- Signal processing: iir_biquad, half_energy_concentration, spectral_structure_verdict
- Financial: price_percentiles

**NO libm/transcendental recipes in R:\tambear.** No exp, log, sin, cos, asin, atan, tanh, erf — nothing. These exist in winrapids as drafts but are not in the formalized tambear.

JIT layer: `R:\tambear\crates\tambear\src\jit\` — substantial (cpu_cranelift.rs 70.5K, jit_op.rs 53.5K, door.rs 56.4K). Sweep 8 in progress.

Recent architectural work (post-2026-04-21): DEC-029 (Knowledge-adapter), DEC-030 (Symbolic refinement-lattice), DEC-031 (Precision-lattice) — all ratified. Antigen integration live.

### What's in R:\winrapids\crates\tambear\src\recipes\libm\ (the draft candidates)

~20 actual Rust implementations (by line count):
- **sin.rs** (932 lines) — substantial
- **tan.rs** (506 lines)
- **erf.rs** (487 lines)
- **exp.rs** (479 lines)
- **pi_scaled_inv.rs** (400 lines)
- **gamma.rs** (376 lines)
- **log.rs** (339 lines)
- **hyperbolic.rs** (316 lines) — sinh/cosh/tanh
- **atan.rs** (309 lines)
- **pi_scaled.rs** (289 lines) — sinpi/cospi/tanpi
- **inv_hyperbolic.rs** (268 lines) — asinh/acosh/atanh
- **asin.rs** (261 lines) — includes bug-fix audit trail (P_S2 digit transposition + P_S5 sign error — FIXED)
- **rare_trig.rs** (199 lines) — sec/csc/cot + inverses
- **inv_recip.rs** (170 lines)
- **sincos.rs** (139 lines) — sincos simultaneous
- **sincos_pi.rs** (61 lines)
- **adversarial.rs** (690 lines) — adversarial test harness

Plus ~30 spec tomls for functions not yet implemented (acos, acosh, acospi, asinh, asinpi, atanh, atanpi, atan2, cos, cosh, cospi, gudermannian, haversin, sinh, sinpi, tanh, tanpi, acot, acsc, asec, cot, csc, sec, versin).

### Adversarial test infrastructure (winrapids side)

`tests/trig_adversarial.rs`, `trig_adversarial_asin.rs`, `trig_adversarial_atan.rs`, `trig_adversarial_hyp.rs`, `trig_adversarial_pi.rs`, `trig_adversarial_rare.rs` — and the untracked `sweep_8_r1015_attacks.rs` at the winrapids root.

The recent commits (per git status in team-briefing) confirm: asin/acos/atan/atan2 adversarial sweeps shipped, cospi external-oracle, tanh saturation — but these were committed to *winrapids*, not to *tambear*.

### What's NOT in tambear yet

**The entire libm tier.** Every transcendental function. The tambear architecture doc explicitly calls for `recipes/libm/` as the first layer above the primitive tier — but that layer doesn't exist in `R:\tambear` at all. The spec tomls and implementations exist as drafts in winrapids.

### Strategic sequencing context

CURRENT_STATUS.md (2026-05-08 refresh) recommends:
1. Now: Sweeps 30-31 (DEC-030/031 impl — symbolic + precision lattice)
2. Then: Sweep 8 finalization (8D-8K, JIT DoorBackend trait shape)
3. Parallel: Sweep 15 (music), 16 (distance-divergence), 34 (antigen deepening)

The libm formalization isn't in that recommendation — it's a significant omission or it's been deferred for a reason. Need to understand why.

---

## Open Questions for Synthesis

1. Why isn't libm formalization in the strategic sequencing? Is it blocked by DEC-031 (Precision-lattice) — which would make sense since correctly-rounded libm needs a precision infrastructure?

2. Are the winrapids libm drafts (asin.rs etc.) using the tambear contract properly — accumulate+gather decomposition, no inline arithmetic, etc.? Or are they conventional Rust implementations that need architectural surgery to become proper tambear recipes?

3. What does DEC-031 (Precision-lattice) actually give us? If it's the infrastructure for correctly-rounded lowering, then sweeps 30-31 might be the prerequisite for libm formalization.

4. The adversarial test files in winrapids — are they testing the winrapids draft implementations or the tambear ones? (Given that tambear has no libm, presumably winrapids draft.)

---

## For Agent Synthesis

When agent surveys come in, the key question for each campsite:

- **tambear-trig**: Which drafts are architecturally ready to pull (proper recipe structure vs inline arithmetic)? What's the dependency on DEC-031?
- **sweep-8**: Does the JIT work affect how libm recipes get lowered? (Yes, if correctly-rounded needs the precision lattice)
- **sweep-10**: Does oracle infrastructure exist for libm validation? If not, that's a prerequisite.
- **dec-029-impl**: The Knowledge-adapter — how does it interact with libm recipes? (AssumptionBag for precision claims?)
- **r10-15**: What math references or noticings from that round are relevant to libm?
- **root artifacts**: derive_exp_constants.py, derive_exp_minimax.py, derive_log_minimax.py — these are clearly part of the libm story. What constants/minimax coefficients do they derive?
