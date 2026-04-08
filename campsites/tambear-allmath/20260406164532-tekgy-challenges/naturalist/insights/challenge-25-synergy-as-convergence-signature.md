# Challenge 25 — Multi-Adic Synergy ≈ 0 as a Convergence Signature

**Date**: 2026-04-06  
**Type C: Foundation Challenge — structural identity**

---

## The Traditional Assumption

Dynamical system convergence is determined by Lyapunov exponents, fixed-point theorems, or direct computation. These are global properties requiring full trajectory analysis.

## The Potential Alternative

Multi-adic synergy ≈ 0 may be a LOCAL, MEASURABLE signature that distinguishes convergent from divergent (mn+1) maps — without computing the full trajectory.

---

## The Observation

From `multi_adic.rs` and `naturalist_observation.rs`:

- **Synergy** measures: how much variance in `ln(trajectory_value)` is jointly explained by {v₂, v₃, v₅, v₇} beyond the best single prime predictor
- `synergy = R²_joint - max(R²_single)`
- Positive synergy → primes INTERACT (the map preserves cross-prime structure)
- Zero synergy → primes are DECOUPLED (the map destroys cross-prime information)

**Hypothesis**: For Collatz (3n+1), synergy ≈ 0 because the +1 perturbation destroys the mod-3 structure. For diverging maps like (5n+1), synergy > 0 because the primes remain coupled.

---

## Why This Would Be Structurally True

The Collatz map (3n+1)/2^{v₂(n)} acts on odd integers. It maps odd → odd. At each step:
1. Multiply by 3 (preserves mod-3 structure, adds 3-adic coupling)
2. Add 1 (destroys mod-3 structure: 3n+1 ≡ 1 (mod 3) for any n)
3. Divide by 2^{v₂(3n+1)} (purely 2-adic operation)

Step 2 is the key. The +1 maps every residue mod 3 to the SAME residue class (1 mod 3). This is a constant function mod 3. It erases all information about the value's 3-adic structure.

The 5n+1 map: 5n+1 ≡ n+1 (mod 5), which is NOT a constant function mod 5. The 5-adic structure is partially preserved. Hence synergy > 0 for (5n+1).

---

## Formal Statement

For a (mn+1) map with divisor d=2:

**Claim**: The multi-adic synergy between v₂ and v₃ along trajectories starting from any seed is:
- ≈ 0 when m ≡ 2 (mod 3) (i.e., m+1 ≡ 0 mod 3, so mn+1 ≡ 1 mod 3 = constant)
- > 0 when m ≢ 2 (mod 3)

For m=3: 3 ≡ 0 (mod 3), so 3n+1 ≡ 1 (mod 3). Zero synergy.
For m=5: 5 ≡ 2 (mod 3), so 5n+1 ≡ 2n+1 (mod 3). Non-constant, hence synergy > 0.
For m=7: 7 ≡ 1 (mod 3), so 7n+1 ≡ n+1 (mod 3). Non-constant.
For m=2: 2 ≡ 2 (mod 3), so 2n+1 ≡ 2n+1 (mod 3). Non-constant.
For m=8: 8 ≡ 2 (mod 3), so 8n+1 ≡ 2n+1 (mod 3). Non-constant.
For m=11: 11 ≡ 2 (mod 3), so 11n+1 ≡ 2n+1 (mod 3). Non-constant.

Among the subcritical (m+1)/4 < 1 maps: only m ∈ {1, 3} have m ≡ 0 or 2 (mod 3) giving constant behavior mod 3.
- m=1: 1n+1 = n+1. Not interesting (identity near 1).
- m=3: 3n+1 ≡ 1 (mod 3). **Unique convergent + zero-synergy map.**

---

## What This Enables

1. **Computational test**: `naturalist_observation.rs` Experiment 3 already measures this. If the synergy for (3n+1) is consistently near 0 and for (5n+1) is consistently > 0, the hypothesis is confirmed empirically.

2. **Proof strategy via synergy**: If zero synergy can be proven formally (the +1 destroys v₃ structure), AND if zero synergy implies convergence (via the temperature Lyapunov function), then the proof path is:
   - m=3 → zero synergy (provable mod-3 argument)
   - Zero synergy → v₃ steps are independent of trajectory history → temperature process is memoryless in the 3-adic direction
   - Memoryless temperature → (same argument as fold_irreversibility.rs) → convergence

3. **GPU primitive**: batch_profiles in multi_adic.rs is already GPU-layout-ready. Computing synergy for a BATCH of starting values (testing many seeds simultaneously) would directly test the hypothesis at scale.

---

## Connection to Challenge 24 (Prime Thermodynamics)

The synergy measurement and the fold point s*(2,3) are two views of the SAME coupling:

- **Fold point**: thermodynamic — at what s does the {2,3} prime system freeze?
- **Synergy**: information-theoretic — does the trajectory preserve cross-prime information?

If the system is BELOW its fold point (in the hot/deconfined phase, which is where Collatz operates for any finite n), then the primes are thermodynamically independent. Zero synergy is the trajectory-level signature of thermodynamic independence.

The +1 perturbation keeps the system in the deconfined phase by erasing 3-adic memory at each step. This is the same as "decoupling" in thermodynamic language.

---

## Most Actionable Immediate Test

Run `naturalist_observation.rs` Experiment 3 and log the results. The expected outcome:
- (3n+1): synergy ≈ 0 for all seeds tested
- (5n+1): synergy > 0 for seeds tested

If confirmed empirically, this is publishable as a new signature of Collatz uniqueness.
