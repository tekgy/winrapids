# Handoff

## State
Industrialization expedition complete. Two fixes landed and verified: `matrix_exp` singularity-as-identity bug fixed (`linear_algebra.rs:1560`) returns NaN matrix instead of `eye`; three missing `IntermediateTag` variants added (`intermediates.rs`: `ManifoldDistanceMatrix`, `ManifoldMixtureDistance`, `Centroids`). Crate compiles clean. `intermediates.rs` has unstaged changes (the new variants + corrected `CovarianceMatrix` docstring marking it as unwired). Next-landscape proposals filed at `campsites/expedition/20260410165051-seed-the-next-landscape/aristotle/proposals.md`.

## Next
1. **Semiring<T> trait design** — BLOCKER on Op enum extension. Must precede pathmaker adding TropicalMinPlus/TropicalMaxPlus variants. Design spec in `~/.claude/garden/2026-04-10-semiring-trait-design.md`. Assign to math-researcher.
2. **Validity semantics decision** — choose Propagate (NaN) as the one policy; sweep `log_gamma(x≤0)` (returns INFINITY, wrong) and any panicking entry points. Adversarial owns.
3. **Commit unstaged changes** — `crates/tambear/src/intermediates.rs` (new IntermediateTag variants + CovarianceMatrix docstring fix).

## Context
`CovarianceMatrix` IntermediateTag exists but is NOT wired — `pca()` uses SVD directly, no TamSession registration. Docstring now correctly marked "WIRING STATUS: not yet implemented." Don't treat its 8-consumer fan-out as current. ARMA = Kingdom B (MA innovations are computed state, not data). PELT has two valid implementations: tropical Kingdom A (GPU, O(n²)) and PELT pruning (CPU, Kingdom B optimization, O(n) avg) — TAM chooses based on hardware × input structure.
