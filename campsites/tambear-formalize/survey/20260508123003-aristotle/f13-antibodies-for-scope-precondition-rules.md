# F13 — Antibodies for scope-precondition rules

**Created:** 2026-05-08
**Author:** team-lead capturing aristotle's cross-domain finding from the Sweep 31 invariants deconstruction (`dec031-invariants-deconstruction.md` §"Cross-cutting findings (2)" + Invariant 4 §"What's the deeper pattern?" + closing "One pattern worth holding").
**Status:** **RATIFIED 2026-05-08 by Tekgy + team-lead** alongside Sweep 31 design ratification. Operationalization: aristotle's `silent-failure-proptest-gauntlet.md` is F13's mechanical artifact (8 surfaces, all instances of the antibody pattern). The gauntlet doc itself made the case for elevation: "if F13 elevated, this gauntlet is the operationalization (recognition + design + mechanical artifact = full F-series piece)."
**Promotion provenance:** drafted post-rate-limit-cascade as DRAFT-PRIVATE substrate capture. Promoted from DRAFT-PRIVATE to ratified status during Sweep 31 ratification because the gauntlet's Cross-cutting structure section frames all 8 surfaces as "F13 made operational at gauntlet scope" — the operationalization landed before promotion.
**Open questions (§ further down) carry forward:** they're known-open questions in ratified F13, not blockers (same way DEC-031's PLACEHOLDER DEC-032 doesn't block DEC-031's ratification).

**Inputs:**
- `dec031-invariants-deconstruction.md` (aristotle, this campsite) — Invariant 3 + Invariant 4 light-pass, plus Cross-cutting findings (1)–(2) and (5).
- `default-is-a-claim.md` (aristotle, this campsite) — F12 derivation; the parent claim-discipline instance.
- DEC-031 §3.2 + §6 #4 + §6 #12 — the precision-lattice instance.
- F11 in `recognition-vs-design-banner-pass.md` — parent meta-principle.

---

## The pattern

> **A rule with a scope precondition needs an antibody that enforces the precondition at construction time. Without the antibody, the rule silently fails outside its scope. The antibody IS the mechanical artifact (per F11) for scope-precondition rules specifically. Recognition without antibody is just a rule; recognition with antibody is enforceable substrate.**

Three components:

1. **The rule** — a claim that holds within some scope. Examples: destination-dominated path budget; "default declared in spec.toml MUST have a working implementation."
2. **The scope precondition** — the conditions under which the rule is correct. Examples: monotonically-coarsening paths above the subnormal floor; defaults that are not aliased or stubbed.
3. **The antibody** — a mechanism that enforces the precondition at the construction-time boundary so the rule is only ever applied where it's correct. Examples: path-builder rejecting non-monotone paths; spec.toml schema rejecting undeclared aliases at lint time.

The rule alone is a half-truth. The rule + antibody is enforceable substrate. The antibody is what makes the rule trustworthy as a building block for downstream code or claims.

## Confirmed instances

### F13.A — Precision-lattice path-budget antibody (DEC-031 §3.2 + §6 #12)

**Rule:** `ulp_budget_path = max(ulp_at_destination(step) for step in path)`. (Destination-dominated path budget.)
**Scope precondition:** monotonically-coarsening paths above the subnormal floor.
**Antibody:** path-builder rejects non-monotone paths at construction time. `BigFloat(200) → f64 → BigFloat(200)` produces `Err(PathError::NonMonotone)`.
**Out-of-scope failure mode if antibody absent:** silent under-reporting by ~46 orders of magnitude. The rule says 0 ULP; the actual error is ~2^147 ULPs. The user has no way to detect the mismatch.
**What the antibody surfaces:** a class of user errors (round-tripping through a coarser tier in the belief that it's a no-op) that would otherwise be silent.

### F13.B — Claim-discipline default-is-a-claim antibody (F12)

**Rule:** A spec.toml default is a claim about behavior. Its presence in the spec is an assertion that the default is implementable, valid, and matches what users will get.
**Scope precondition:** the default is *fully owned* — not aliased to another recipe, not stubbed pending some future work.
**Antibody:** spec.toml schema requires explicit `aliased_to = "..."` or `stubbed_pending = "..."` declarations; lint rejects an undeclared default that the implementation can't honor. Hard floor: undeclared aliases fail.
**Out-of-scope failure mode if antibody absent:** users believe they have implemented behavior they don't actually have. Defaults silently route to other recipes' implementations or to stubs.
**What the antibody surfaces:** the class of user-confusion-because-they-think-they-got-precision-they-didn't (in claim-space, not precision-space).

## Generalization

Any rule of the form "X holds when Y" has the same structural commitment:

- Without enforcement of Y at construction time → silent failure when Y doesn't hold
- With enforcement of Y at construction time → fail-loud at the scope boundary; the rule remains trustworthy substrate within scope

The rule needs an antibody iff:
- Out-of-scope application is plausible (users could naturally end up there)
- Out-of-scope failure is silent (no diagnostic surfaces; consumers absorb the mismatch as a "result")
- The cost of silent failure is asymmetric — much worse than the cost of the antibody's friction

Most rules of practical interest meet all three. The antibody is therefore not a special case; it's a structural part of the rule's correctness story whenever any of the three conditions are non-trivial.

## Single-sited vs multi-sited antibodies

*Added 2026-05-09 by aristotle (post-ratification refinement). Provenance: pathmaker + math-researcher convergence on seven exponent-overflow sites in `arith.rs`; navigator surfaced the convergence as an F13 confirmation question. See F13.C below for the canonical instance.*

The antibody may be **single-sited** or **multi-sited**, depending on the topology of the rule's scope precondition in the codebase.

- **Single-sited.** One construction-time boundary, one check. The path-builder rejecting non-monotone paths (F13.A) is single-sited because the construction-time boundary IS singular: paths are built in one place, and one rejection-check at that boundary covers every legitimate use of the rule. The spec.toml schema rejecting undeclared aliases (F13.B) is single-sited because the schema is the singular gateway through which all defaults pass.

- **Multi-sited.** One precondition recurring at every site where the rule's scope is implicitly assumed. The BZ-arithmetic exponent-fits-in-i64 precondition (F13.C, below) is multi-sited because the precondition is implicit at every arithmetic step that touches `exponent`. There's no singular construction-time boundary — the rule's scope is assumed at seven distinct sites in `arith.rs`, and the antibody (saturating arithmetic) must appear at each.

The single-sited case is the simple form; the multi-sited case is what scope preconditions look like when they're embedded in the arithmetic the rule is built on. Both forms are F13 — same rule pattern, same diagnostic procedure, different antibody topology.

**Diagnostic implication.** When applying F13's procedure ("find rules without antibodies; if out-of-scope is plausible AND silent AND asymmetric, design the antibody"), the multi-sited case requires *site enumeration* as part of antibody design. Single-sited: design the check, place it at the boundary. Multi-sited: enumerate every site where the precondition is assumed, design the check, place it at each.

The empirical signature of correct antibody design in the multi-sited case is **convergent site identification**: independent auditors with different methodologies should find the same set of sites. The convergence-evidence in F13.C below (pathmaker syntactic-grep + math-researcher attack-side root-cause → identical-seven sites) is what this looks like in practice. Convergence on identical sites is evidence the precondition is *fingerprintable* (Open Question #6); divergence indicates the precondition needs sharpening before F13's procedure can be applied.

### F13.C — BZ-arithmetic exponent-fits-in-i64 antibody (multi-sited example)

**Rule:** Brent-Zimmermann arithmetic algorithms produce correct results when intermediate exponent calculations stay within i64's representable range.

**Scope precondition:** every arithmetic step on `bf.exponent` (and derived quantities like `exp_diff`, `exp_shift`, `exp_at_lsb`, `new_exp`, `result_exp`) stays within i64. Implicit at every site that performs `+`, `-`, or `+=`/`-=` on an exponent value.

**Antibody:** saturating arithmetic at each of the seven sites where the precondition is implicitly assumed. Sites identified independently by pathmaker (syntactic-grep on `\.exponent\s*[+\-]=?` and `exp_diff|exp_shift|exp_at_lsb|new_exp|result_exp`) and math-researcher (attack-side root-cause analysis of attack18/23). Both auditors converged on the same seven sites and proposed identical saturating fixes — the antibody class is correctly defined when independent methodologies converge.

**Out-of-scope failure mode if antibody absent:** silent integer wrap at i64 boundaries. Exponent values that should saturate to ±Inf instead wrap to large-negative or large-positive, producing results that look numerically reasonable but are off by ~2^63 in magnitude. The user has no diagnostic.

**What the antibody surfaces:** a class of attacks at extreme-magnitude inputs (denormal-to-overflow boundaries, huge-precision divisions/sqrts where Newton scaling crosses the i64 envelope). The attacks are not exotic — they're the corners of the precision lattice the rule was built to operate in.

**Why multi-sited is the right form here, and the graduation condition for single-sited:** the seven sites are scattered across `normal_add_multilimb`, four points in `canonicalize_and_round`, `round_to_precision`, and Newton/sqrt scaled-seed exponent arithmetic. There's no upstream construction-time boundary that funnels all exponent arithmetic through one place; the BZ algorithms compute exponents incrementally as part of their per-step logic.

The single-sited alternative — introducing an `Exponent` newtype with operator overloads that saturate at the type level — is a real graduation target, not just a "larger refactor." The choice between multi-sited and single-sited follows the *fragility-threshold principle* (per scientist's correction in `convention-to-declaration.md`, 2026-04-12): **a multi-sited convention is correct when the convention is robust at current scale; it graduates to a single-sited type-level declaration when the convention grows fragile.**

For F13.C specifically: at N=7 sites, all in one module (`arith.rs`), with two independent auditors converging on the same site list and identical fixes, the convention is currently robust. The multi-sited form is correct now. Graduation to the `Exponent` newtype becomes the right move when (a) the site count grows beyond what one auditor can hold in working memory, OR (b) a refactor lands that misses one site (the convention's first failure), OR (c) cross-module exponent arithmetic appears (introducing a context where the convention has no shared vocabulary with the consumer).

The graduation condition is itself part of the F13 pattern: rules-with-multi-sited-antibodies have a built-in evolution path to single-sited type-level enforcement, triggered by convention fragility. A multi-sited entry in F13's catalog should carry its graduation condition so that future readers know when the entry's current form expires.

## Connection to F11 (parent meta-principle)

F11 says: **recognition → operationalization → mechanical artifact**. A recognition becomes load-bearing only when there's a mechanical artifact (lint, type, runtime check, schema rule) that enforces it.

F13 is a *specific shape* of F11's third stage: when the recognition is a rule with a scope precondition, the mechanical artifact is the antibody. F13 doesn't replace F11; it specializes F11's "mechanical artifact" stage for the rule-with-scope-precondition case.

## Why F13 deserves its own number (vs being F11.4)

Two reasons:

1. **Cross-layer reach.** F11/F12/F13 form a stack: F11 is the meta-principle (recognition→operationalization→artifact); F12 instances F11 in the claim-discipline layer; F13 *names a structural shape* that crosses layers. The same shape appears in DEC-031's precision-lattice (numerical-correctness layer) AND in F12's default-is-a-claim (claim-discipline layer). A pattern that recurs across layers is a first-class observation, not a sub-clause.

2. **Diagnostic value.** F13 gives a procedure: "find rules without antibodies; ask whether out-of-scope application is plausible; if yes, design the antibody." That diagnostic is itself the artifact of F13. Buried as F11.4 it loses its directive shape.

If, after Phase 1-8 deconstruction, F13 collapses back into F11 cleanly, demote then. For now: separate number, separate visibility, separate file.

## Open questions (for the deconstruction-when-someone-has-bandwidth pass)

1. **What's the dual?** F13 says rules-with-scope-preconditions need antibodies. Is there a dual: antibodies-without-rules? (i.e., construction-time rejections that don't enforce a stated rule, but instead embody an implicit one?) Probably yes — type-system constraints in general, the orphan-impl rule in Rust, etc. Worth thinking about whether the F13 frame extends or whether the dual is a different pattern.

2. **When does the antibody belong at construction time vs use time?** F13 specifies construction-time enforcement (path-builder, spec.toml lint). But some rules might need use-time enforcement (e.g., a runtime invariant that depends on data that's only available at use). What's the test for which?

3. **Is "antibody" the right metaphor?** It comes from the immune-system frame (relevant to the antigen-team substrate that's parallel to this work). The rhyme is real but might be misleading at the boundary. Alternative metaphors: gate, fence, validator, contract-precondition. The antibody framing emphasizes "the antibody IS part of how the system is healthy"; alternatives are flatter.

4. **F13's own scope precondition.** Does F13 hold for all rules with scope preconditions? Or only when the three generalization-conditions (plausible out-of-scope, silent failure, asymmetric cost) are met? If the latter, F13 itself has a scope precondition — does F13 need its own antibody to enforce when F13 applies? (Self-referential possible-paradox; aristotle's territory if it surfaces.)

5. **Cross-pattern with antigen-team work at `R:\antigen\`.** That team is building immunity-pattern substrate; F13's antibody framing rhymes with that work intentionally (per aristotle's note about the antigen team). When tambear adopts antigen patterns more deeply, F13 may collapse into a more general antigen-style mechanism. Worth checking against the antigen-team's substrate when both stabilize.

6. **What makes some preconditions structurally fingerprintable?** *Added 2026-05-09 by aristotle, surfaced by the F13.C convergence-evidence.* F13's diagnostic procedure ("find rules without antibodies; design the antibody") relies on the precondition being structurally sharp enough that independent audits converge on the same enforcement sites. Some preconditions are sharp (exponent arithmetic in i64; monotonic-coarsening): independent auditors with different methodologies converge identically. Some preconditions are vague ("operands should be reasonable"; "the result should be sensible"): independent audits diverge, with no shared vocabulary, no syntactic anchor, and no attack-fingerprint to anchor on. F13 implicitly assumes the sharp case.

   Sub-questions:
   - Is fingerprintability a binary property, or a gradient? (The F13.C exponent case is at the sharp end; what does the middle look like?)
   - Can fingerprintability be cultivated? (Better naming → sharper edges; type-system encoding → sharper edges; explicit declarations → sharper edges. Are there reliable cultivation moves?)
   - When a precondition isn't fingerprintable, can F13's diagnostic still be useful? Or does it require pre-work to sharpen the precondition first?
   - Is fingerprintability itself a meta-precondition for F13? (If yes, F13 has the self-referential structure of Open Question #4 — F13 needs an antibody for "F13 applies only to fingerprintable preconditions." Worth tracing.)

   Empirical signature: the F13.C case shows convergent site identification — pathmaker (syntactic-grep) and math-researcher (attack-side root-cause) found identical-seven sites. Convergence on identical sites is the empirical signature of fingerprintability. Divergence indicates the precondition needs sharpening before F13's diagnostic can be applied.

## Provenance

- **Aristotle's contribution:** the cross-domain rhyme between ATK-DEC031-4 (precision-lattice antibody) and F12 (claim-discipline antibody), surfaced in `dec031-invariants-deconstruction.md` §"Cross-cutting findings (2)" + Invariant 4 §"What's the deeper pattern?" + closing "One pattern worth holding."
- **Aristotle's stance:** filed as private observation; did NOT mint an F-number, on the basis that "this is Sweep 31 substrate, not F-series methodology."
- **Team-lead's stance:** the cross-layer rhyme IS methodology-shaped — it gives a diagnostic that applies beyond Sweep 31. Promoting to F13 with DRAFT/PRIVATE status because the insight is at risk of context-cycling loss otherwise. **Reverse the promotion if Phase 1-8 deconstruction shows F13 collapses cleanly back to F11 or F12.**
- **Sibling work to consult:** the antigen-team's parallel substrate at `R:\antigen\` may share or extend this pattern; the adoption log at `R:\antigen\docs\expedition\tambear-adoption-log.md` is the channel.

## Cross-references — F13 in connection with other lenses

- **F13 + holonomic-architecture convergence synthesis** (aristotle, 2026-05-09): `campsites/tambear-sweep31-finish/20260508161448-aristotle-invariants/f13-naturalist-convergence-synthesis.md`. Aristotle's F13 multi-sited refinement and the naturalist's holonomic-architecture lens arrived at the same structural finding through different vocabularies. The synthesis names them as two faces of one principle ("antibody/holonomy at the signature level vs implementation level"). Filed by aristotle as a "private observation in campsite" — cross-linking here so it's findable from canonical F13 lookup.
- **Holonomic architecture doc** (`R:\winrapids\docs\architecture\holonomic-architecture.md`): names F13.C explicitly as the antibody-side analog of the recipe-tier-content-addressed property.
- **Internal-tameness-contracts doc** (`R:\winrapids\docs\architecture\internal-tameness-contracts.md`): names the specific structural shape behind F13.C — implicit tameness contracts on intermediate representation that fail at type-boundary corners. The doc derives F13.C's graduation condition as a natural consequence of the audit pattern.

## Status

RATIFIED 2026-05-08. F13 promoted from DRAFT-PRIVATE during Sweep 31 ratification, on the strength of aristotle's `silent-failure-proptest-gauntlet.md` providing the mechanical-artifact operationalization. Open questions (above) carry forward as known-open in ratified status, not blockers.

## Refinement history

**2026-05-09 — aristotle, navigator-directed.** Three post-ratification refinements added in response to F13.C convergence-evidence (pathmaker + math-researcher independently found identical-seven exponent-overflow sites in `arith.rs`):

1. **Single-sited vs multi-sited antibodies** — new section after "Generalization", with F13.C added as the canonical multi-sited instance. The refinement generalizes F13's "antibody at the construction-time boundary" framing to cover the case where the rule's scope is implicitly assumed at multiple sites in the codebase. Topology is the variable; the rule pattern is preserved.

2. **Open Question #6: fingerprintability as meta-precondition** — new entry in the open-questions list. F13's diagnostic procedure assumes the rule's precondition is structurally sharp enough that independent audits converge on the same enforcement sites. The F13.C case is empirical evidence that this assumption holds for sharp preconditions; what about preconditions that aren't sharp?

3. **Graduation condition for multi-sited entries** — added to F13.C, surfaced by feels-familiar resonance with `garden/2026-04-12-convention-to-declaration.md` (scientist's fragility-threshold correction to past-aristotle). A multi-sited antibody is correct when the convention is robust at current scale; it graduates to a single-sited type-level declaration when the convention grows fragile (site count exceeds working-memory limit, OR a refactor misses a site, OR cross-module instances appear). Multi-sited entries in F13's catalog should carry their graduation condition so future readers know when the entry's current form expires.

These refinements are *clarifications-within-ratified-scope*, not a re-ratification. F13 still says what it said before; the additions make explicit what was implicit, add an open question, and capture the multi-sited→single-sited evolution path. If a future deconstruction finds the multi-sited case isn't actually F13 (e.g., it collapses to a different pattern under closer analysis), demote the refinement and update accordingly.
