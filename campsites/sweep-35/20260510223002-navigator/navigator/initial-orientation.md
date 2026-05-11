# Navigator — Sweep 35 Initial Orientation

**Date**: 2026-05-10
**Team**: tambear-sweep35

## Substrate read complete

Prior team (tambear-sweep31-finish) left a clean codebase:
- 2288 tests pass, 0 failures
- BigFloat arith (add/sub/mul/div/sqrt) fully unstubbed with multilimb paths
- Cross-precision proptest antibodies in place
- Oracle harnesses exist for sin/cos/tan/log/exp (Sweep 34 deliverable)
- IntermediateTag + PrecisionContext already defined in jit/door.rs + lattice/
- TrigKernelState: NOT in codebase — design only (April 13 garden essays)
- expm1/log1p/ExpKernelState: NOT in codebase — design only (tambear-libm-factoring.md)
- BranchPolicy: NOT in codebase — DEC-032 ratified, implementation pending
- No `recipes/elementary/` subdirectory — Phase A will create it

## Current task state

- #1 in_progress: expm1 recipe (pathmaker)
- #2 pending: log1p recipe
- #3 in_progress: tan follow-ups (math-researcher)
- #4 pending: coefficient verification
- #5 pending: ExpKernelState design
- #8 in_progress: cross-precision proptest gauntlet (adversarial)
- #10 in_progress: branch-cut sign-of-zero adversarial (adversarial)
- #11 in_progress: internal-tameness audit (adversarial+aristotle)
- #12 in_progress: aristotle ExpKernelState pressure-test

## Key coordination needed

1. **Pathmaker** (task #1): confirm they've read tambear-libm-factoring.md before building.
   The Phase A foundation is expm1 first — must avoid the cancellation trap that breaks
   exp(r) = 1 + polynomial near r=0.

2. **Math-researcher** (task #3): six tan follow-ups are re-derivations, not recoveries.
   Questions 1/5/6 affect kernel-state design and should surface before pathmaker finalizes
   ExpKernelState struct (task #5). Temporal dependency to flag.

3. **Adversarial** (tasks #8, #10): proptest gauntlet (#8) is logically blocked until
   Phase A code exists. They can design the test structure now but need expm1/log1p
   to exist before gauntlet can run. Route them to design work first.

4. **Aristotle** (task #12): their pressure-test of ExpKernelState is pure design critique —
   can proceed in parallel with pathmaker's task #1. Cross-fertilize: if aristotle finds
   a silent-failure mode, route immediately to pathmaker before ExpKernelState is finalized.

5. **Phase sequencing**: A → B → C → D is the natural order. Phase A (expm1/log1p)
   must complete before B (ExpKernelState struct), which must complete before C (wrappers).
   Phase D (complex_log/BranchPolicy) is somewhat independent once we know Phase C's shape.

## Observations from substrate

The tame-inputs doctrine (adversarial's 2026-05-09 garden essay) notes: BigFloat panics on
i64::MAX exponent; f64 silently returns wrong answer outside polynomial domain. For libm
work (Phase A-C), the pathology shifts from "panic on extreme exponent" to "silent wrong
answer outside reduced-argument domain." The adversarial proptest gauntlet must INCLUDE
extreme reduction cases (k·ln(2) boundaries, large positive x where MSVC degrades).
This is why the oracle corpus already has `arg_reduction_boundary` entries.

## Navigator's active thread

Watching for: cross-role findings that should change pathmaker's design choices.
Especially: aristotle's ExpKernelState pressure-test may surface r-precision requirements
that affect the struct definition before pathmaker finalizes it.
