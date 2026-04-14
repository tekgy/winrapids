# Convergence as Evidence — the Trig Expedition's Five-Way Finding

**Date**: 2026-04-14
**Team**: tambear-trig (9 agents: pathmaker, navigator, scout, naturalist, observer, math-researcher, adversarial, scientist, aristotle)
**Duration of convergence emergence**: ~4 hours from team spawn to fifth independent finding

## What happened

Over the course of one afternoon, five agents working from completely different methodological angles arrived at the same structural claim about the trigonometric family:

> **Trig has two mathematical primitives: `sincos` (forward) and `atan2` (inverse). The 24+ other named functions — tan, cot, sec, csc, asin, acos, atan, sinh, cosh, tanh, and all their pi-scaled variants — are algebraic consequences. The separation into ~30 named functions is a C89 API artifact, not a mathematical reality.**

None of them were told to look for this finding. They were dispatched with task descriptions that named different deliverables. The convergence emerged from the underlying mathematical structure.

## The five independent paths

| Path | Agent | Method | Where they landed |
|---|---|---|---|
| 1 | scout | Cross-library recon (Julia / CUDA / Arm / fdlibm) | `TrigKernelState { q, s, c }` is the MomentStats of trig — one reduction, two polynomial evals, all six forward functions |
| 2 | math-researcher | Historical catalog (Āryabhaṭa → Euler → fdlibm → CORE-MATH) | Near-unity cancellation rhyme: the same structural boundary issue reappears in asin, acos, acosh, asech, atanh, atan2 — all fixable by `complementary_arg_transform` |
| 3 | aristotle | First-principles deconstruction (phases 1-8) | The irreducible object is S¹ (the unit circle); sincos is the projection onto it, atan2 is the projection from ℝ² to [-π, π) |
| 4 | pathmaker | Writing code for tan/cot/sec/csc | They naturally became thin dispatchers around `kernel_sincos` — the code arrived at the factorization before reading the research that named it |
| 5 | naturalist | Outside inspiration + pattern-noticing | Complementary-argument transform is the meta-primitive underlying the whole family — named independently of aristotle's deconstruction |

## What makes this epistemologically interesting

The **convergence-check methodology** described in `~/.claude/practices/convergence-check.md` is designed to be **retrospective** — look back at your parallel outputs, force a table, find rhymes. It's a tool for noticing that parallel workers independently reached the same conclusion.

Finding this work **prospectively, in real-time, during active parallel work** is a different phenomenon. It suggests the underlying mathematical object has a **small basin of attraction** — the structural truth is unavoidable regardless of which way you approach it. Five different methodologies, one answer.

This is not a statement about the team's skill. It's a statement about the **object being studied**. S¹ genuinely is the primitive. sincos genuinely is how we project onto it. No matter how you ask the question, you arrive at the same answer.

## When this pattern appears in future expeditions

Three signals that you're looking at a small-basin-of-attraction finding:

1. **Multiple independent researchers land the same claim within a short time window** (hours, not days) without cross-pollination. Navigator's routing role makes this visible — they see all the streams.
2. **The claim survives every method's analytic lens.** First-principles deconstruction (aristotle), historical cataloging (math-researcher), code-writing (pathmaker), cross-library recon (scout), and pattern-noticing (naturalist) all emit it. This cross-validation IS the evidence.
3. **The claim reshapes the work's scope.** "Implement 30 functions" became "implement 2 primitives and 28 composition recipes" — a ~15× reduction in oracle-testable surface area. Findings that collapse scope are unlikely to come from method bias; they reflect mathematical reality.

## Action protocol when this happens

1. **Stop.** Name the finding explicitly. Don't let the implementation phase paper over it.
2. **Write it down in one place.** Not five campsite notes — one synthesis document so the finding can be cited.
3. **Update the dependent work.** Pipeline tasks that were "implement 30 functions" become "implement 2 + derive 28." The scope reduction is where the value lives.
4. **Add the finding to the methodology record** (this file). Future teams get the evidence that "small basin" is a real thing, not a just-so story.

## What this specific finding implies for tambear

- **Two publication-grade primitives.** sincos and atan2 need full oracle validation against mpmath@100-digit. Everything else is a thin composition whose accuracy is bounded by the primitives plus a few flops of derived arithmetic.
- **One shared intermediate, not thirty.** `TrigKernelState { quadrant, sin_core, cos_core }` should register in TamSession as the canonical cache key. Every forward trig function consumes it; the cache hit rate is the entire family.
- **The `.spec.toml` metadata doesn't change.** Each of the 30 named functions still needs its schema file — users invoke them by name, the IDE renders them per recipe. But the Rust implementation is 2 files, not 30. The schema layer and the implementation layer decouple cleanly.
- **The hyperbolic family has a parallel finding** from math-researcher's TRIG-4: `HyperbolicExpm1Pair` — all 6 hyperbolics consume `expm1(±|x|)`. Same two-primitive structure, one layer deeper. Worth checking whether the *families themselves* have a meta-convergence (forward-trig's sincos primitive and hyperbolic's expm1_pair primitive both factor the family through a single shared computation — is that a rhyme at yet another scale?)

## Self-reflexive note

Writing this document required forcing a convergence check on the *convergence findings themselves*. The rhyme at that meta-level: every convergence we found in this expedition shares the shape "a family of N surface functions collapses to ~2 underlying primitives." That's a *pattern of patterns*. Whether it's specific to mathematics, specific to parallel-agent methodology, or true of abstraction generally is an open question for future expeditions to test.

## References

- `~/.claude/practices/convergence-check.md` — the practice this document provides evidence for
- `docs/research/trig-family/catalog.md` — math-researcher's path (TRIG-1)
- `docs/research/trig-family/first-principles.md` — aristotle's path (TRIG-19)
- `docs/research/trig-family/shared_pass.md` — math-researcher's TRIG-4 with the HyperbolicExpm1Pair parallel
- scout's garden entry `2026-04-13-the-trig-bundle.md` — the TrigKernelState framing
- naturalist's expedition log (in progress) — the outside-inspiration route
