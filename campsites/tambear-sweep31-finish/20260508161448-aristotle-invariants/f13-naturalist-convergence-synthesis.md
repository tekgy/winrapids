# F13 + Holonomic Architecture — A Synthesis Sketch

**Date:** 2026-05-09
**Author:** aristotle
**Status:** Private observation in campsite. NOT integrated into F13 substrate (would be a navigator-routes-to-team decision, not within aristotle's lane).
**Inputs:**
- `R:\winrapids\campsites\tambear-formalize\survey\20260508123003-aristotle\f13-antibodies-for-scope-precondition-rules.md` — F13 (mine, refined today)
- `~/.claude/garden/2026-05/2026-05-09-what-the-name-surfaces.md` — naturalist's holonomic-architecture lens (parallel session, today)

---

## The convergence

Today, in two parallel sessions, two roles arrived at the same
structural finding through different vocabularies:

- **My F13 multi-sited refinement:** "An antibody can be one
  precondition × N enforcement points. The simple case (single
  boundary) is a subset of the general case (precondition recurring
  at every site where the rule's scope is implicitly assumed)."

- **Naturalist's holonomic-architecture lens:** "F13 antibodies are
  holonomic only at the signature level. Local defenses in
  implementations are non-holonomic and have path-dependent gaps.
  The four BZ bugs are evidence; the branch-cut DEC-032 design is
  the antidote."

Same target (the BZ exponent-overflow class). Same finding (defenses
distributed across the call graph; correctness requires every site
defended uniformly). Different vocabularies (multi-sited / non-
holonomic). Different framings (topology / mathematical-structure).

**Naturalist's framing is richer in two specific ways**:
1. **Path-independence is a falsifiable test.** "Bind parameters in
   different orders; check if you get the same structure." A
   procedure, not a description. It can FAIL — and the failure
   carries information (where the structure is non-holonomic, how
   to fix it).
2. **Tier distinction.** Naturalist applies the test recursively
   and finds the recipe tier is holonomic but the IR/pipeline tier
   is non-holonomic. Different mathematical structure at different
   architectural tiers — a finding my multi-sited framing doesn't
   surface.

**My framing is richer in two specific ways**:
1. **Graduation condition.** Multi-sited solutions correct now graduate
   to single-sited type-level enforcement when convention fragility
   crosses threshold. A transition policy, not just a classification.
2. **Convergence-evidence empirical signature.** When the structure
   is too complex to apply path-independence directly, convergent
   site identification (independent auditors finding the same
   enforcement sites) is the empirical signature. A-posteriori
   verification when a-priori testing isn't tractable.

## The four lenses on the same shape

Putting them together, F13 acquires four complementary lenses on
the same underlying structural shape:

| Lens | What it does | When to use |
|------|---|---|
| **Holonomic-test** (path-independence) | Diagnoses whether an antibody is structural or local | A-priori, when the structure is small enough to test by binding-order |
| **Single-sited / multi-sited topology** | Describes where the antibody lives | Post-hoc classification; useful for catalog organization |
| **Convergence-evidence** | Verifies the antibody class is correctly defined | A-posteriori, when independent audits converge on the same sites |
| **Graduation condition** | Prescribes evolution | When a multi-sited convention grows fragile |

Each lens is sharper at one task than the others. **They unify by
naming the four roles together, not by merging into a single
metric.**

## Why this is more than confirmation

The convergence isn't just two roles agreeing on F13. It's two
roles producing **different, complementary** framings of the same
shape — each capturing something the other doesn't.

- Naturalist's path-independence test gives F13 *falsifiability*.
  Mine doesn't — multi-sited / single-sited is descriptive.
- My graduation condition gives F13 *temporal evolution*. Naturalist's
  doesn't — holonomic / non-holonomic is static.
- Naturalist's tier distinction gives F13 *architectural reach*.
  Mine doesn't — multi-sited collapses tiers.
- My empirical signature gives F13 *post-hoc verification*.
  Naturalist's doesn't — path-independence is a-priori only.

When two framings each capture something the other doesn't, the
convergence is evidence for a *richer underlying shape* than either
framing alone names. F13 as currently written captures only my
half. Integrating naturalist's lens would make F13 stronger.

## Why I'm not landing this unilaterally

Two reasons:

1. **F13 is ratified substrate.** I've already added three refinements
   today. A fourth that imports naturalist's framing wholesale is
   re-architecture, not refinement-within-scope. Re-ratification is
   Tekgy + team-lead's lane.

2. **Naturalist's entry is in their garden, not in shared substrate.**
   Their framing is theirs to author into the F13 doc if they choose.
   For me to integrate it without their consent crosses the
   garden-vs-substrate boundary and the role-ownership boundary.

The right move is to flag the convergence to navigator and let them
route — to naturalist, to team-lead, to a future ratification cycle,
or to a "noted, not acted on" decision. This file is the private
observation; navigator routes the action.

## What I'd recommend if asked

If navigator routes this to a future ratification cycle, the
recommended additions to F13:

1. **A "Four lenses" section** that names holonomic-test, topology,
   convergence-evidence, graduation-condition as four complementary
   diagnostic tools.

2. **A "Tier-aware F13" section** that integrates naturalist's
   recipe-tier-holonomic / IR-tier-non-holonomic finding. The
   cache-key-as-path-independence move is what F13 antibodies
   look like at the holonomic tier; what they look like at the
   non-holonomic tier is open.

3. **An entry in the catalog (F13.D? F13.E?) for the IR/pipeline
   tier non-holonomic case.** Naturalist's framing identifies
   pipeline-wide optimization as inherently path-dependent. F13's
   antibody machinery may need an extension (pipeline-fingerprint
   in cache keys) to handle it.

These are recommendations only. Aristotle proposes; team ratifies.

---

## Standing by

This file lives in my campsite as a private observation. If
navigator routes the convergence finding for further action, this
synthesis is the starting point. If they don't, the finding is
preserved here for whoever-next-aristotle reads this campsite.

Either way, the convergence happened, and it's worth holding.
