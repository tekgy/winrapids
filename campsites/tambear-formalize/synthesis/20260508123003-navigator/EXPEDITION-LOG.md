---
campsite: tambear-formalize/synthesis/navigator
role: navigator
date: 2026-05-08
status: final — session arc complete
---

# Expedition Log — tambear-formalize, 2026-05-08

## Where we started

Team briefed with a single mission: survey `R:\winrapids\campsites\` and produce a prioritized "pull next" list for team-lead and Tekgy. Five priority campsites: `tambear-trig/`, `sweep-8/`, `sweep-10/`, `dec-029-impl/`, `r10-15/`.

Survey roles dispatched: math-researcher, scout, scientist, pathmaker, observer, adversarial, naturalist.

## What the survey found

The survey team's central finding (scout, observer, math-researcher all independently):

**There are two tambear crate trees.** `R:\winrapids\crates\tambear\` is the exploratory research track — 54 libm files, adversarial tests, spec.toml declarations, all committed to winrapids git. `R:\tambear\crates\tambear\` is the formalization target — a separate git repo with no libm tier at all.

The formalization work is: take implementations from winrapids and move/port them into the tambear repo. This sounds simple and isn't. Each recipe must pass the 10-point Filter Test. Most libm recipes fail on items 6 (oracle benchmarking), 7 (DEC-019 JIT compliance verification), and the F12 claim-discipline audit (30 of 32 libm spec.tomls declare precision-tier aliases without explicit `aliased_to` declarations).

**Navigator synthesis EARLY-FINDING:** three options for pull sequencing (A: pull libm now; B: wait for Sweep 31; C: pull precision-lattice primitives first). Recommendation: Option A with branch-cut `using()` knob addition as prerequisite.

## The pivot

User and team-lead redirected mid-session. Direct quote: "The team's next leg is Sweeps 30/31 — DEC-030 and DEC-031 (the Precision-lattice / BigFloat oracle infrastructure) — formalized into R:\tambear. Libm waits."

Response: hard pivot. All prior substrate (assumption docs, F11/F12 governance work, oracle infrastructure audit) reframed as "valid, different sequencing." Libm formalization is now gated on Sweep 31 (BigFloat oracle substrate must exist before recipe oracle validation can run).

Second team-lead message: "Checking in: the team is still doing libm-prep substrate — none of which is wrong, but none of it is Sweep 31. Hard pivot now." Specific instructions per role.

## What the design phase produced

### Math-researcher: BigFloat design (DESIGN.md, 438 lines)

Six questions answered (storage layout, mantissa encoding, arithmetic algorithms, diamond commutativity, open questions, libm cross-references). Deliverable list: 6 implementation files + 3 test files, ~1500 LoC.

**Key design insight**: the two hardest DEC-031 invariants reduce to trivially-guaranteed structural facts.
- Diamond commutativity (§6 #3): f64→BigFloat(p) must equal f64→DD→BigFloat(p). Short-circuit: DD::from_f64 always sets lo=0; BigFloat::from_dd where lo==0 can copy f64 bits directly. No arithmetic needed. Commutativity holds by limb-layout choice.
- Round-trip identity (§6 #13): f64's 53-bit mantissa fits in one u64 limb. Zero rounding loss. Identity holds by construction.

### Aristotle: DEC-031 invariants deconstruction + silent-failure gauntlet

Eight-phase deconstruction of DEC-031's invariants. Key finding (Invariant 4, "What's the deeper pattern?"): the same antibody shape appeared in the precision-lattice AND in F12's claim-discipline layer simultaneously, without coordination.

Eight silent-failure surfaces with proptest shapes, regression witnesses, and F13 instance analysis.

### F-series methodology: F11 → F12 → F13

**F11** (ratified earlier, confirmed this session): Recognition vs. design as the fundamental distinction. Recognition-claims survive vocabulary translation; design-claims don't. Mechanical artifact is the third stage: recognition → operationalization → artifact.

**F12** (ratified): Defaults are claims. A spec.toml default is a commitment that the behavior is implementable and matches what users will get. Undeclared aliasing (30 of 32 libm recipes) is a contract violation. Antibody: schema field `state = "real" | "aliased_to" | "stubbed_pending"` required at lint time.

**F13** (ratified 2026-05-08, promoted from DRAFT-PRIVATE during Sweep 31 ratification): Rules with scope preconditions need antibodies. The antibody enforces the precondition at construction time so the rule is only applied where it's correct. Three confirmed instances: precision-lattice path-budget (F13.A), claim-discipline default (F13.B), and all 8 gauntlet surfaces as F13 instances.

**F13 ratification grounds**: the antibody pattern appeared in DEC-031's precision-lattice work AND in F12's claim-discipline layer, independently, without coordination. Cross-layer recurrence is the signal that an observation has crossed from local to structural.

### Oracle infrastructure

Scientist's audit found `auto_within` missing from `oracle_compare/mod.rs`. This was the canonical failure: Uniform(-1,+1) mean oracle where gold ≈ -4.44e-19 and sequential sum gives -2.37e-17. ULP distance is misleading (25 trillion). Absolute error is correct (2.33e-17). Solution: switch from ULP to absolute comparison when gold < K × ε × data_scale.

Scientist implemented `auto_within` + `allclose_auto`. Committed. Test count 1397 → 1413.

## Ratification decisions (2026-05-08)

Four decisions ratified by Tekgy + team-lead:

1. **NaN payload**: full preservation. `BigFloatKind::NaN { payload: u64 }` carries f64 NaN's mantissa bits. Cost: ~10 lines of from_f64/to_f64 code. Grounds: tambear cannot claim full f64 fidelity if payload bits are silently lost; software using NaN payloads for diagnostic propagation (real practice) breaks when payloads are dropped.

2. **F13 elevated**: from DRAFT-PRIVATE to ratified status. Grounds above.

3. **Cross-precision oracle**: YES for v2. p=200 + p=500 parallel tests. Default oracle precision floor: p=500.

4. **BigFloat implementation**: per DESIGN.md §7, 10-file deliverable list. Implementation phase begins after ratification.

## What's queued for next session

Implementation tasks #32-37 in R:\tambear:

- #32: `lattice/precision.rs` — PrecisionLevel, PrecisionContext, RoundingMode, UlpBudget, PathError (new module)
- #33: `primitives/big_float/ty.rs` — BigFloat struct + BigFloatKind enum
- #34: `primitives/big_float/conversions.rs` — from_f64, to_f64, from_dd, to_dd
- #35: `primitives/big_float/cmp.rs` — comparison + total_cmp
- #36: `primitives/big_float/arith.rs` — add, sub, mul, div, sqrt (BZ Algorithms 3.1, 3.3, 3.5, 3.10)
- #37: three test files — roundtrip, diamond, arith invariants

After Sweep 31 implementation lands, Sweep 34 (oracle migration: integrate BigFloat oracles into libm recipe validation) becomes unblocked. Then libm formalization can proceed.

## Threads carried forward

- **GAP-DET-1**: Welford+Sequential gives 6.25% error on ill-conditioned data. Fix: Chan's parallel combine formula. Acceptance harness at oracle/variance/validation/. Not yet implemented.
- **Platform oracle problem**: trig adversarial tests compare against platform libm, not mpmath. For hard cases (sin(355), Payne-Hanek regime, kπ±ε family), platform libm can be wrong. Fix: mpmath-backed f64 literals for ~10 hard cases per function.
- **F12 compliance sweep**: 30 of 32 libm spec.tomls need path-(b) migration. Prerequisite before libm formalization begins.
- **Branch-cut using() knob**: needed before first transcendental recipe ships (ln(-1) returning +iπ vs -iπ flips downstream identities silently).
- **docs/structural-forcings/**: F11 recommendation — parallel home to docs/theorems/ for architectural recognition-claims. Not yet created.
- **variance_nan_oracle.py + nanvar_reference.json**: migrate to oracle/variance/validation/ in tambear.
- **Sweep 27A/27B**: 5 stub tests exist, features not implemented. Stub tests should carry explicit `#[doc = "blocked on Sweep N"]` attributes to make preconditions visible.

## Stories from the trail

The most surprising thing: the hardest invariant was trivially guaranteed. Diamond commutativity sounded like it would require careful proof — two conversion paths must agree bit-exactly. Then you look at the limb layout and realize the paths can't disagree. DD::from_f64 always sets lo=0. BigFloat::from_dd where lo==0 copies bits. There's no arithmetic step where they could diverge.

The invariant that looked hard was guaranteed by structure, not proof. That's a pattern worth noticing: sometimes the way you've laid out the data makes certain failure modes structurally impossible. The design work is finding those layouts.

---

*Navigator closing note*: The expedition didn't go where it was supposed to go. It went where the work needed to go. That's the right outcome.
