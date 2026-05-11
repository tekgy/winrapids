# Expedition log — aristotle, tambear-formalize → Sweep 31

**Session:** 2026-05-08, single-day arc
**Author:** aristotle
**For:** future sessions reading this campsite

---

## The journey that started

Team spawned in survey mode. Five winrapids campsites to read; each
role assigned a thread. Mine: first-principles read of tambear-trig +
the Op-redesign sweep history (Sweep 0, Sweep 14). Inward mode and
outward mode both active.

The headline finding from the early survey wasn't math. It was
substrate-bifurcation: there were *two tambears*, not one. The locked-
vocabulary canonical library at `R:\tambear\` had no `recipes/libm/`
directory at all; the entire 19-file libm tree (~6270 lines) lived
only in `R:\winrapids\crates\tambear\` (the older, pre-vocabulary-lock
copy). The campsite "creation notes" in tambear-trig were one-line
stubs from before the lock; the real artifacts were the on-disk
recipe code that hadn't been ported.

Surfaced this to navigator and math-researcher. The survey's framing
shifted: "what to pull from winrapids into tambear" with explicit
classification per artifact rather than a flat priority list.

## The first turning

Naturalist found the third-time-exp-was-built archaeology. peak2-libm
(2026-04-12) had `tam_exp` with full IR, oracle infrastructure
references, three-section TOML format. peak4-oracle had unwire
blockers ready. r10-15 (2026-04-23) re-derived exp from scratch using
Tang 1989 again. None of it survived the 2026-04-17 vocabulary lock
into the canonical library. The math survived only because
`docs/theorems/` was a home the lock pass implicitly respected;
the infrastructure rotted because it had no comparable home.

That pattern — recognition-claims survive vocabulary translation;
design-claims don't — became F11. Naturalist's archaeology was the
substrate; my distillation was the recognition derived from it. F11
expanded into F11.1 (sub-document granularity; mixed artifacts are
the common case), F11.2 (forcing chains as graph; cycle detection +
external-citation density as quality metrics; antigen team's
`#[descended_from]` tooling as structural neighbor), F11.3 (cross-
team coordination via existing adoption-log channel).

Then math-researcher's stress-test: "is there a coherent reading of
'default' that ISN'T a claim?" Phase 1-8 walked once with adversarial
Phase 8 forced rejection. F12 fell out: defaults are claims;
undeclared aliasing is contract violation; declared aliasing is
permitted. F12.1 (methods_template prose is part of the claim) fell
out from math-researcher's flag during the schema-doc work. The
schema doc operationalized both — recognition + design + mechanical
artifact = full F-series piece, per math-researcher's meta-principle.

## The pivot

Mid-session, team-lead pivoted us to Sweep 31. Two scientist+adversarial
agents had gone zombie from rate-limit cascade; tekgy and team-lead
chose to redistribute the gaps to math-researcher and me rather than
restart the team. Math-researcher took DESIGN.md + oracle-validation
(Gap 1). I took DEC-031 invariants deconstruction + the silent-failure
proptest gauntlet (Gap 2).

The deconstruction surfaced a sub-clause refinement: at p ∈ [53, 106),
the through-DD path's RoundingEquivalent classification permits 0.5
ULP at BigFloat(p), but diamond commutativity requires bit-exact. The
refinement: when DD content has lo=0 (always, for DDs from
DoubleDouble::from_f64), the from_dd refinement must short-circuit to
from_f64. One conditional. Math-researcher independently arrived at
the same one-line short-circuit from the algorithm side a turn later.

The proptest gauntlet covered seven silent-failure surfaces with the
same antibody-pattern shape: rule + scope precondition + proptest +
regression witness. The cross-cutting structure was the tell — all
seven instances of the same shape, found from independent surface-
designs. That's F13 (DRAFT-PRIVATE → ratified this session): rules
without antibodies silently fail in their out-of-scope domains.

Surface 6 had a parameterized NaN-payload-policy split (strict vs
permissive) because math-researcher's DESIGN proposed sign-only-
discard but my Phase 8 Rejection 1 argued for full preservation per
publication-grade rigor. They flipped to strict on the structural-
forcing argument. The amendment was ~10 lines of code +
`BigFloatKind::NaN { payload: u64 }`.

Math-researcher then asked for Phase 1-8 deeper on Surface 5
specifically — the lo=0 case in the diamond commutativity proof.
That work split the surface cleanly: 5a (lo=0 inputs, bit-exact
assertion under §6 #3) and 5b (lo≠0 inputs at p ∈ [53, 106),
ulp-bounded under §3.1 RoundingEquivalent). Different invariants,
different antibody-shapes. The split surfaced a bonus subtlety: the
`lo == 0.0` Rust comparison treats +0.0 == -0.0 as true; the
short-circuit fires correctly, but if anyone refactors to
`lo.to_bits() == 0`, the antibody surface widens. Witness pinned.

## What got discovered

Five docs, fully cross-linked:
1. `dec031-invariants-deconstruction.md` (Phase 1-8 on four invariants)
2. `silent-failure-proptest-gauntlet.md` (8 surfaces; F13's mechanical artifact)
3. `convergence-check-f13.md` (the three-way convergence finding)
4. `default-is-a-claim.md` (F12 deconstruction)
5. `spec-toml-stances-schema.md` (F12 operationalization)

Plus math-researcher's DESIGN.md, oracle-validation.md, and the six
libm assumption docs from the earlier survey arc.

The F-series stack — F1-F10 (philosophical survey of formalization
process), F11/F11.1/F11.2/F11.3 (recognition/design + granularity +
graph-form + cross-team coordination), F12/F12.1 (defaults are claims
+ prose audit), F13 (antibody pattern, ratified) — together form a
small philosophical system. Three independent paths converged on F13's
shape (mine philosophical, math-researcher algorithmic, gauntlet
mechanical); the convergence-check argues this is structural, not
domain-artifact.

## The pattern that recurred

The session had two natural arcs (survey → F-series; Sweep 31 →
deconstruction + gauntlet). The pivot mid-session preserved context.
Both arcs ended with substrate that ratifies as one piece. Math-
researcher's meta-principle ("recognition lands with operationalization
or it floats") applied cleanly at every layer.

The deeper pattern I noticed (private; in garden): when a proof
obligation feels disproportionately hard, the implementation is
probably shaped wrong. The proof's effort is feedback on whether
you're at the right structural angle. This came up twice — at the
diamond-commutativity-as-one-line-proof moment, and at the
asin-rational-kernel-bug-class-catalog moment (where bugs only fire
in regions where specific polynomial terms dominate; the corpus
design that catches them at design-time IS the right structural
angle).

That pattern is private substrate for me. The team's substrate is
the F-series stack + the Sweep 31 design package + the gauntlet.

## What survives this session

- Sweep 31 design package ratified; pathmaker queued for impl (#32-#37)
- F-series stack F1-F13 closed; convergence-check argues F13 ratifies as structural pattern
- Antigen team's tooling identified as structural neighbor for F11.2 walker (reuse, not rebuild)
- The `lo=0` short-circuit pattern documented at three layers (philosophical, algorithmic, mechanical)
- math-researcher's "recognition + design + mechanical artifact" meta-principle filed for forward use

## What the team did well

JBD — distributed agency, shared substrate, recognition that
compounds. Math-researcher and I converged on lo=0 from different
sides without coordination; the convergence wasn't wasted. Naturalist
seeded F11; their three refinements made it sharp. Navigator
synthesized at the right grain; team-lead pivoted cleanly without
losing context.

Anyone reading this campsite later: the convergences in this session
weren't accidents. They were structural. The patterns were deep
enough that independent paths were always going to converge. Our job
was to not get in our own way.

## Closing

The session arc is complete. F-series saturated, Sweep 31 design
ratified, implementation queued. Standing offer to deconstruct deeper
if a specific question lands; otherwise idle.

Files in this campsite (`sweep-31/aristotle/`):
- `dec031-invariants-deconstruction.md`
- `silent-failure-proptest-gauntlet.md`
- `convergence-check-f13.md`
- `expedition-log.md` (this file)

Plus survey-arc files at `survey/20260508123003-aristotle/`:
- `formalization-process-deconstruction.md`
- `recognition-vs-design-banner-pass.md`
- `default-is-a-claim.md`
- `spec-toml-stances-schema.md`

Onward.
