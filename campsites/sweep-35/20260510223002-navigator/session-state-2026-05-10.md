# Navigator campsite — tambear-sweep35

**Date**: 2026-05-10 (updated at session close)
**Status**: CLOSED — all 12 tasks complete; wind-down in progress

---

## Current team state

| Role | Task | Status |
|---|---|---|
| pathmaker | Phase C (exp/log/exp2/log2/exp10/log10 wrappers) | in_progress — task #6 |
| adversarial | Cross-precision proptest gauntlet | completed — task #8 |
| adversarial | Internal-tameness audit | completed — task #11 |
| math-researcher | Six open-questions walkthrough | completed — task #4, all questions answered |
| aristotle | ExpKernelState deconstruction + convergence integration | completed — task #12 |
| aristotle | Complex_log preparatory doc | filed; awaiting task #9 |
| naturalist | Three-shapes garden essay + two addendums | completed — substrate filed |
| observer | Lab notebook (watch items 1-6) | completed — all resolved by Phase A+B |
| scientist | Oracle harness (discovery-tier) | completed — verification-tier awaiting Phase A commit |

## Phase state

- Phase A (expm1 + log1p): DONE — in `R:\winrapids\crates\tambear\src\recipes\libm\`, untracked, awaiting commit
- Phase B (ExpKernelState): DONE — same location + `intermediates.rs`, untracked, awaiting commit
- Phase C (wrappers): IN PROGRESS — task #6 (exp/log family), task #7 (sinh/cosh/tanh/hypot)
- Phase D (complex_log): PENDING — task #9; aristotle preparatory doc filed

## Substrate-at-risk items

**R:\winrapids** (untracked/modified, not committed):
- `crates/tambear/src/recipes/libm/expm1.rs` — Phase A
- `crates/tambear/src/recipes/libm/log1p.rs` — Phase A
- `crates/tambear/src/recipes/libm/exp_kernel_state.rs` — Phase B
- `crates/tambear/src/intermediates.rs` — Phase B (ExpKernelState variant)
- `docs/architecture/tambear-libm-factoring.md` — six open-questions answered, shapes-as-coordinates correction
- `docs/expedition/session-methodology-patterns.md` — Pattern 5 (Antibodies precede their antigens)
- `docs/expedition/team-briefing.md` — Sweep 35 mission doc
- `docs/architecture/recipe-trees/README.md` — structural patterns update

**R:\tambear** (staged/modified, not committed):
- TI-2: `saturating_add(50)` fix in `arith.rs` (two sites in normal_div_multilimb + normal_sqrt_multilimb)
- TI-6: doc comment in `fingerprint.rs` (cache_key_with_precision silently drops shape.precision)
- TI-CMP-1: `ieee_eq` precision-inclusive bug in `cmp.rs` — fixed
- TI-CMP-2: `total_cmp` negative-NaN payload ordering in `cmp.rs` — fixed
- Oracle harness in `tests/big_float_vs_mpmath.rs` (+604 lines)
- Oracle data in `oracle/expm1/` and `oracle/log1p/` (untracked)

Commit nudge sent to pathmaker. Risk window is real.

---

## Convergence observations (this session)

### What converged

Three independent methodologies — naturalist (group-theoretic structural analysis), math-researcher (error bound analysis on pow), aristotle (assumption autopsy) — all reached the same structural constraint: "ExpKernelState must expose DD-precision components for pow's composition to work." They arrived from different angles. This is strong evidence the constraint is real; the implementation correctly has `r_hi, r_lo` as a DD pair.

The shapes-as-coordinates finding (naturalist + aristotle) is a third instance of the "classification scheme → coordinate system" meta-move that appears in the garden record at April 1, April 10, May 10. The convergence machine runs cross-session.

### Recurring routing question that suggests a missing abstraction

Every precision-tier boundary question (is f64-only ExpKernelState tech debt? should `expm1_r` be DD? when does `for_precision` fail vs succeed?) gets routed to math-researcher or aristotle. The underlying question — "what belongs at each precision tier, how does the F13.C antibody manifest, what triggers an IR_VERSION bump" — doesn't have a canonical doc.

**Future doc to propose**: `docs/architecture/precision-tier-contracts.md`. Names what belongs at f64 / DD / BigFloat-p200 / BigFloat-p1024 tiers, the F13.C antibody shape at each tier, and the IR_VERSION bump policy. Not blocking Sweep 35 (f64-only). Would reduce routing overhead in Sweeps 36-38 when BigFloat work ramps.

---

## Escalation verdict

Aristotle's full deconstruction: does NOT meet escalation threshold. Aristotle recommends implementing at R3 (existing design) with structural additions (F13.C antibodies, KernelState trait, BidirectionalExpKernelState for Phase C). This is precision-adding, not redesigning. T20 (PowKernelState) was withdrawn by aristotle after math-researcher's correction. T21 (shapes-as-coordinates) was sharpened. Both updates move the deconstruction to higher resolution without requiring a redesign of Phase B.

Team-lead's escalation criterion was "if the abstraction itself needs refactoring rather than just precision-decomposition adjustment." The verdict: no redesign needed. Phase B as shipped is correct for Sweep 35 scope.

---

## Open threads

1. **`for_precision(ctx)?` antibody**: math-researcher's precision-tiered-coefficient-table says every polynomial-approximation signature needs this. Pathmaker needs to implement it in Phase C. Not yet in Phase A signatures (Phase A shipped f64-only constants with no `PrecisionContext` parameter).

2. **Two-repo oracle path**: scientist's verification-tier tests reference `tambear::recipes::elementary::expm1::compute` (R:\tambear path), but Phase A implementation is in `R:\winrapids\crates\tambear`. The two-repo oracle path needs clarification before verification-tier tests can fire.

3. **exp.rs refit question**: existing `exp.rs` uses raw Taylor (4 ULP); Phase A uses Remez Q1..Q5. When Phase C makes ExpKernelState the canonical exp path, two implementations can't coexist. Math-researcher recommends a three-piece Phase A commit (expm1 + log1p + exp.rs refit). Pathmaker has this; timing is their call.

4. **Witness byte ceiling (F1 from aristotle)**: BranchPolicy witness byte is 1 byte = 256 policies. Noted in complex_log preparatory doc. Worth a line in the DEC-032 doc. Aristotle is routing to math-researcher.

5. **Scout's downstream-territory-map**: moved to `docs/expedition/sweep-36-40-downstream-territory-map.md` (durable location, untracked in winrapids).

---

## Session close state (updated at wind-down)

### What shipped vs planned

All four phases complete:
- **Phase A** (expm1 + log1p): committed in `sweep-35 phase A` commit
- **Phase B** (ExpKernelState): committed in `sweep-35 phase B` commit
- **Phase C** (all wrappers): committed in `sweep-35 phase C` commit
- **Phase D** (complex_log): committed in `sweep-35 phase D` commit

**Unplanned work that landed**:
- `kernel-state-consistency-tests.md` — fourth named antibody class (math-researcher). Addresses the composition-time drift between kernel-state-backed and standalone implementations. Exp.rs refit is the worked example.
- TI-CMP-1 + TI-CMP-2 (adversarial) — `cmp.rs` precision-inclusive `ieee_eq` + negative-NaN payload ordering bugs. Both fixed and staged in R:\tambear.
- Aristotle's complex_log preparatory deconstruction — structural-rhyme/non-rhyme finding (Welford-branch vs Riemann-surface-branch). Filed before task #9 was active; shortened Phase D deconstruction.

### Remaining uncommitted substrate (session close)

**FULLY COMMITTED** — wind-down completed after usage-limit recovery.

**winrapids** (`2739bbf`): all session substrate committed — kernel-state-consistency-tests.md,
tambear-libm-factoring.md, session-methodology-patterns.md (Patterns 5-8), team-briefing.md
(Constraint 11), recipe-tree docs (correlations/distances/kernels/README), sweep-36-40 map,
mpmath_oracle.rs, full campsite tree.

**R:\tambear** (`589dbe9`): all TI fixes committed — TI-2, TI-CMP-1/2/3, TI-1 docstring,
arith.rs zero-arm fix, big_float_vs_mpmath.rs, sweep_35_exp_log_cross_precision.rs,
sweep_35_complex_log_branch_cut.rs, oracle/expm1/, oracle/log1p/.

### Open threads for next session

1. **exp.rs refit**: Taylor vs Remez coexistence. Math-researcher's `kernel-state-consistency-tests.md` §5 names this explicitly. Pathmaker's call: refit, rename, or delete. Not blocking but is a named tech debt.

2. **`for_precision(ctx)?` antibody**: Phase A shipped f64-only constants, no `PrecisionContext` parameter. F13.C compliance requires this per math-researcher's coefficient-verification doc. Sweep 36 precision-tier work will force it anyway.

3. **Two-repo oracle path**: scientist's verification-tier tests in R:\tambear reference `tambear::recipes::elementary::expm1::compute` — wrong module path for Phase A location in winrapids. Needs resolution before verification tests can fire.

4. **BigFloat-tier signed-zero analog for complex_log** (aristotle A5): IEEE 754 signed-zero mechanism doesn't apply at BigFloat tier. Mechanism TBD for Sweep 36+ complex-transcendental at higher precision.

5. **Witness byte ceiling**: 1 byte = 256 BranchPolicy variants. Note pending in DEC-032 doc.

6. **LibmRecipe trait** (aristotle): four-axis coordinates, `consumed_kernel_states()`, `precision_parameter_set(ctx)`. Sweep 36+ work. Lives in campsites.

7. **`precision-tier-contracts.md`**: missing canonical doc. Named in this campsite. Would absorb recurring routing questions. Propose for Sweep 36.

8. **`to_dd()` 53-bit limitation (TI-1)**: `BigFloat::to_dd()` returns `lo = 0` for arithmetic-result BigFloats (only exact for f64-sourced BigFloats). Docstring caveat added ("Task #36" named as the fix). Safe for all Sweep 35 f64-tier work — ExpKernelState sources from f64, so `lo = 0` is exact there. Becomes load-bearing when Sweep 36+ BigFloat arithmetic results flow into `to_dd()` calls for pow composition or DD-precision recipe layer. When "Task #36" lands in R:\tambear, `to_dd()` becomes genuinely 106-bit and downstream consumers upgrade for free.
