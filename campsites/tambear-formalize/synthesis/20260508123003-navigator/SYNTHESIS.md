# Navigator Synthesis — Formalization Survey

**Date:** 2026-05-08  
**Status:** Navigator-side synthesis complete; waiting for full agent survey inputs. This document will be updated as agents report in.

---

## What the survey found

### The big picture

R:\tambear has no libm tier. The entire transcendental function catalog — exp, log, sin, cos, asin, atan, tanh, erf, gamma, and 30+ more — exists only as drafts in R:\winrapids\crates\tambear\src\recipes\libm\. This is the largest single formalization gap.

### The draft inventory

**Implemented in winrapids (ready to examine):**
| File | Lines | Functions |
|------|-------|-----------|
| sin.rs | 932 | sin (+ likely cos via sincos) |
| adversarial.rs | 690 | adversarial test harness |
| tan.rs | 506 | tan |
| erf.rs | 487 | erf (+ erfc likely) |
| exp.rs | 479 | exp |
| pi_scaled_inv.rs | 400 | asinpi, acospi, atanpi |
| gamma.rs | 376 | gamma, log_gamma |
| log.rs | 339 | log, log2, log10 |
| hyperbolic.rs | 316 | sinh, cosh, tanh |
| atan.rs | 309 | atan |
| pi_scaled.rs | 289 | sinpi, cospi, tanpi |
| inv_hyperbolic.rs | 268 | asinh, acosh, atanh |
| asin.rs | 261 | asin, acos |
| rare_trig.rs | 199 | sec, csc, cot + inverses |
| inv_recip.rs | 170 | (unclear — needs reading) |
| sincos.rs | 139 | sincos simultaneous |
| sincos_pi.rs | 61 | sincos_pi |

**Spec tomls only (not yet implemented):** acos, acosh, acospi, asinh, asinpi, atanh, atanpi, atan2, cos, cosh, cospi, gudermannian, haversin, sinh, sinpi, tanh, tanpi, acot, acsc, asec, cot, csc, sec, versin — though many of these may be IN the above files (e.g., acos is in asin.rs, tanh is in hyperbolic.rs).

**Adversarial test infrastructure (winrapids tests/):** trig_adversarial.rs, trig_adversarial_asin.rs, trig_adversarial_atan.rs, trig_adversarial_hyp.rs, trig_adversarial_pi.rs, trig_adversarial_rare.rs

### Architectural gap in the drafts

The winrapids libm drafts are NOT yet proper tambear recipes. They use:
- `std::f64::consts::FRAC_PI_2` (not a tambear primitive constant)
- Likely inline Rust arithmetic rather than explicit primitive calls
- No `#[precision(...)]` tags
- No `#[tags(...)]` metadata

They're mathematical implementations — correct, publication-grade polynomial approximations with fdlibm lineage — but not yet wired into the tambear compilation pipeline. Pulling them to tambear requires architectural surgery to make them proper recipes.

### The known bug: asin polynomial coefficients

The campsite audit trail (scout's asin-polynomial-audit.md) records two bugs in asin.rs:
- P_S2: digit transposition (2.01225 → 2.01212, correct)
- P_S5: wrong sign and magnitude (-3.25e-6 → +3.48e-5, correct)

Both marked FIXED in the source. The fixes are in the winrapids draft. These need to carry through to any tambear formalization.

### Key dependency: DEC-031 (Precision-lattice)

DEC-031 (ratified 2026-05-06) declares the three-tier precision lattice: f64 ↔ DoubleDouble ↔ BigFloat. Sweep 31 (implementation) is PLANNED but not started.

**Critical insight:** Libm formalization does NOT need Sweep 31 complete. f64 and DoubleDouble already exist in tambear. What Sweep 31 adds is BigFloat — tambear's own oracle tier to replace the mpmath dependency. We can pull libm recipes using f64 (strict lowering) and DoubleDouble (compensated lowering) now, and add correctly-rounded BigFloat oracle validation when Sweep 31 ships.

### Critical new context: PLEASE_READ_from_gpu_verifier_port.md

A sibling JBD team (GPU verifier port, 2026-04-25) independently arrived at three concepts that affect libm formalization:

1. **Branch-cut conventions need `using(branch: ...)` knob from day one.** `ln(-1)` returning `+iπ` vs `-iπ` flips downstream identities silently. Any transcendental recipe touching negative reals needs this — it's not a detail, it's a correctness class distinction. This is gap in the current winrapids drafts (they probably hardcode one convention).

2. **Discovery-tier vs verification-tier** distinction. The adversarial tests in winrapids are discovery-tier (generator-backed, finds problem regions). The tambear contract requires verification-tier (oracle comparison at every input, against mpmath/BigFloat at 80+ digit precision). These are different quality bars.

3. **GPU target: sm_122, not sm_100.** RTX PRO 6000 Workstation is Compute Capability 12.2. The JIT code targeting Blackwell needs `-arch=sm_122` for this hardware. This affects any trig GPU dispatch.

### Strategic context from CURRENT_STATUS.md

The 2026-05-08 refresh recommends:
1. Now: Sweeps 30-31 (DEC-030/031 impl)
2. Then: Sweep 8 finalization (8D-8K)
3. Parallel: Sweeps 15, 16, 34

Libm isn't in this queue. The question is whether it should be — or whether it belongs *after* Sweep 31 (so we get the BigFloat oracle infrastructure before formalizing).

---

## Synthesis: What to pull next

### Option A: Pull libm now, verification-quality later

Pull the winrapids libm drafts into tambear as a new Sweep (call it "Sweep 35" or fit it into the 30-34 sequence). Do the architectural surgery to make them proper tambear recipes:
- Replace inline arithmetic with explicit primitive calls
- Add `#[precision(strictly_rounded)]` tags 
- Add `#[tags(...)]` metadata
- Wire in the asin bug fixes
- Add branch-cut `using(branch: ...)` knob for all functions that need it
- Validate against mpmath at 80 digits (external, not tambear's own BigFloat)

**Pro:** Fills the fundamental recipe gap that everything else depends on. Math recipes that use trig (physics, dimensional_nyquist, music theory) currently call Rust's f64::sin rather than a tambear recipe — this is architectural debt.  
**Con:** Without Sweep 31's BigFloat oracle, verification is mpmath-dependent (external dependency). Also, the architectural surgery to make 17 files proper recipes is substantial work.

### Option B: Wait for Sweep 31, then pull libm

Do Sweeps 30-31 first (implementing the precision lattice and BigFloat). Then pull libm with tambear's own oracle infrastructure.

**Pro:** Clean sequencing — libm recipes arrive with the full correctly-rounded lowering strategy backed by tambear's own oracle.  
**Con:** Sweeps 30-31 will take significant time. The libm gap continues to be a hole in the recipe catalog during that time. Also, the BigFloat oracle infrastructure isn't *needed* for the recipes to work — it's needed for the highest quality bar.

### Option C: Pull libm primitives first, not recipes

The `#[precision(strictly_rounded)]` lowering of sin/exp/etc doesn't need the recipe layer to be perfect first. Port the primitives that libm recipes call (range reduction, minimax polynomial evaluation, ldexp) and the constants module. This gives the compilation pipeline its foundation without requiring full recipe-level correctness.

**Pro:** Smaller scoped work, clearly foundational.  
**Con:** Doesn't surface the actual sin/cos/exp functions to users yet.

### Navigator's recommendation

**Option A with one addition:** Pull the libm tier as a new sweep (Sweep 35 or similar), but add the branch-cut `using(branch: ...)` infrastructure before pulling — because the verifier-port team's finding is that this affects the first transcendental recipe and the cost of adding it later is high.

Sequence:
1. Add branch-cut `using()` infrastructure (small targeted work, could be part of DEC-031 or its own micro-DEC)
2. Pull libm tier (Sweep 35): architectural surgery on the ~17 winrapids draft files + adversarial tests
3. Validate against mpmath at 80 digits (external oracle, acceptable before Sweep 31)
4. When Sweep 31 lands: upgrade validation to tambear's own BigFloat oracle

This gives us the recipe tier that the rest of tambear's math depends on, without blocking on Sweeps 30-31 first.

The sm_122 GPU target also needs to be documented explicitly in tambear's JIT dispatch before any GPU trig computation runs.

---

## Agent survey status

- math-researcher: surveying tambear-trig — in progress
- pathmaker: surveying sweep-8 — in progress
- scientist: surveying sweep-10 — in progress
- scout: surveying dec-029-impl — in progress
- observer: surveying r10-15 — in progress
- adversarial: surveying root artifacts — in progress
- aristotle: grounding formalization process — in progress
- naturalist: wandering — in progress

This synthesis will be updated as agent reports come in.
