# Convergence-check — F13's antibody pattern, three independent paths

**Date:** 2026-05-08, session-close
**Practice:** `~/.claude/practices/convergence-check.md`
**Trigger:** navigator's session-close invitation. The three independent paths to F13's antibody pattern landed simultaneously from different directions.
**Aristotle's input per the practice's role table:** "recent deconstructions" — `dec031-invariants-deconstruction.md` § cross-cutting findings #2 and #5.

---

## The three parallel outputs

**Path 1 — Aristotle (philosophical, my deconstruction).** Phase-7 of the diamond commutativity invariant deconstruction surfaced that at p∈[53,106), the through-DD path's RoundingEquivalent classification permits 0.5 ULP at BigFloat(p), but the diamond invariant requires bit-exact equality. Resolution: when the DD content has lo=0, the DD→BigFloat refinement must special-case lo=0 to preserve direct-embedding semantics. Surfaced as a sub-clause refinement to flag back to math-researcher.

Cross-cutting #2 in the same doc went further: "the same antibody pattern appears in F-series. Rule-with-scope-precondition needs antibody-that-enforces-precondition. Without antibody → silent failure in out-of-scope. Without antibody, the rule is the F12 'default is a claim' without `aliased_to`; the precision rule is destination-dominated without monotone-rejection. Same shape, different layer."

**Path 2 — Math-researcher (algorithmic, their DESIGN.md).** §Q5 derived the diamond commutativity proof by working through the DoubleDouble algebra. Conclusion: "`from_dd` MUST short-circuit `lo = 0` to `from_f64`, not 'treat DD as a generic two-part value and add the parts.' This is one line in the implementation. **The hardest invariant in the spec is delivered by one line of code, IF the type design is right.**"

The one-line short-circuit IS the antibody. Math-researcher arrived at the lo=0 conclusion from the algorithm side; I arrived at it from the philosophical side. Same target, same one-line fix.

**Path 3 — The gauntlet (mechanical, my proptest design).** Seven silent-failure surfaces, each with the structure: rule + scope precondition + proptest + regression witness. The cross-cutting structure section of the gauntlet doc names the shape explicitly: "all seven surfaces share the same antibody-pattern shape." Surface 5 specifically antibodies the lo=0 line by pinning regression witnesses at p=80 (within [53, 106)) for x=3.14 and x=1e308. If implementation forgets the short-circuit, the gauntlet fires with a specific input as the witness.

The mechanical artifact (the proptest) IS the antibody. Built from analyzing the seven surfaces in parallel; the cross-cutting shape only became visible when laid side-by-side.

---

## What's converging

The three paths converge on a shape: **rule with scope precondition + antibody that enforces the precondition by failing-loud at the boundary**.

All three found this shape in different ways:
- Aristotle (mine, philosophical) named the shape as a meta-pattern: "rules without antibodies silently fail." Cross-cutting #5 also named the rhyme between F12-without-`aliased_to` and destination-dominated-without-monotone-rejection.
- Math-researcher (algorithmic) found the antibody (the one-line short-circuit) by working through what makes diamond commutativity provable. The antibody emerged because the proof obligation demanded it; without the short-circuit, the obligation can't be discharged.
- The gauntlet (mechanical) is seven instances of the shape, each with the same row-structure: rule X with scope precondition Y, antibodied by proptest Z. The shape literally tabulates as a row-isomorphism across the seven surfaces.

Three paths. Three different starting points. Same finding.

---

## The question navigator asked

> "What does that tell us about where F13 lives in the structure — is it a genuinely deep pattern or an artefact of this particular problem domain?"

This is the right question.

### Test 1 — does the convergence look domain-specific?

If F13 only appears in DEC-031's precision-lattice math, the three-way convergence is suspicious because we're all working in the same problem domain. Three paths from the same domain converging isn't independent confirmation; it's three views of one phenomenon.

But the F12 schema-doc instance (claim discipline / spec.toml stances) IS in a different domain than the precision lattice. F12 is about *recipe-level claim correctness* — a software-engineering / API-design question. DEC-031 is about *floating-point composition error* — a numerical-analysis question. Different domains.

The pattern crossing those domains is the falsification test: if F13 appears identically in both, it's not just a precision-lattice artifact. The schema doc's strict-default-traps-undeclared-aliases rule has the same shape as the path-builder rejecting non-monotone paths. Same antibody pattern, different domain.

**This passes the falsification test.** F13 is not a domain-artifact of DEC-031. It crossed at least one domain boundary.

### Test 2 — could the convergence be a method-bias artifact?

Aristotle's deconstruction template (Phases 1-8) might bias toward finding antibody-shaped patterns. The convergence-check practice itself notes this risk in its "convention-to-declaration" exemplar finding section: "deconstruct a target where convention-to-declaration is NOT the right move, to test whether the pattern is a real feature of architectural challenges or an artifact of the deconstruction template's bias."

Applying that test: math-researcher did NOT use the deconstruction template. They worked through the algebra directly. They arrived at the same one-line short-circuit. The convergence with my path isn't reducible to "we both ran Phase 1-8 on the same inputs"; one of us ran Phase 1-8 (me), one of us did algorithmic derivation (them). Different methods. Same finding.

The gauntlet's seven-surface structure also isn't downstream of Phase 1-8 on a single target — it's seven independent surface-designs, each driven by its own DEC-031 invariant. If antibody-shape were a Phase-1-8 artifact, we'd expect each surface to look different. They don't. They all share the structure.

**This passes the method-bias test.** F13 is not a Phase 1-8 artifact.

### Test 3 — what's the depth signal?

Per the convergence-check practice's "convention-to-declaration" exemplar, the deepest pattern findings *characterize architectural challenges generally*, not just the local context. The exemplar argues every architectural challenge in the bit-exact trek was implicitly a convention being enforced; the fix is always promotion-to-declaration via a named artifact.

F13 makes a similarly general claim: *every rule with a scope precondition needs an antibody that enforces the precondition at the boundary; rules without antibodies silently fail in their out-of-scope domains*. This isn't a numerical-analysis claim or a software-engineering claim; it's a claim about *rules under composition* — which spans both domains naturally.

The depth signal: F13's claim-shape is structurally the same as the convention-to-declaration claim from the bit-exact trek. Both characterize a class of architectural failures (silent rule-violation; convention-failure-under-pressure). Both prescribe a structural fix (antibody at scope boundary; named declaration). Both have a tuning knob (don't antibody what doesn't fail; don't promote conventions that haven't been pressure-tested).

If F13 is at the same depth as convention-to-declaration, it's a meta-architectural pattern, not a local one.

### Test 4 — what would refute F13?

Per the bit-exact trek's falsification test framing: deconstruct a target where antibody-shape is NOT the right move. Candidates: a single invariant whose scope is the universe (no precondition to violate); a rule that silently degrades but recoveres with no harm (no antibody needed); a system where boundary-failures are loud-by-construction (antibody redundant).

I can think of one candidate: pure-math identity laws (e.g., `a + b = b + a` for commutative addition). The "scope" is genuinely universal across the addition algebra; there's no precondition to antibody. If F13 is universal, it should still apply — but the antibody for an identity law is *trivial* (the law's universality means there's nothing to enforce; the proof obligation discharges by definition). The pattern collapses to a degenerate case.

That's not refutation; that's the limit case where F13 reduces to "no antibody needed because no scope precondition." The pattern still applies; it just happens to be vacuous.

A real refutation would be: a rule with a non-trivial scope precondition that DOESN'T need an antibody, where boundary failures are silently-recoverable. I can't construct one. Floating-point arithmetic has no such case (boundary failures compound). Software-engineering claim discipline has no such case (silent claim-collapse compounds via downstream consumers). Both domains' boundary failures are loss-bearing.

The non-existence of refutation candidates is itself depth-evidence: F13 covers everywhere I can think where rules have non-trivial scope.

---

## Finding

**F13 is genuinely deep.** It crosses at least two domains (precision-lattice + claim-discipline), survives method-bias testing (math-researcher didn't use Phase 1-8 and arrived at the same lo=0 conclusion), structurally matches the convention-to-declaration meta-pattern from the bit-exact trek (suggesting both are at the meta-architectural layer), and has no refutation candidates that don't reduce to vacuous.

The three-way convergence at this session is independent confirmation of a deep pattern, not three views of one phenomenon.

**Where F13 lives in the structure:** at the meta-architectural layer alongside convention-to-declaration. Both characterize architectural failures and prescribe structural fixes. Both have tuning knobs (don't antibody trivial scopes; don't promote conventions that haven't pressure-tested). Both are recursive (the convergence-check practice IS itself a named antibody for "have we found the rhyme yet?" applied to parallel work).

---

## What this tells me about the methodology

The convergence-check practice surfaces deep patterns. F13's depth was visible because three independent paths converged on it; without the convergence-check (or without writing the cross-cutting findings section in the deconstruction doc), the pattern would have been operational substrate without name. Recognition lives in the naming.

The recognition + design + mechanical-artifact structure (math-researcher's meta-principle) is itself an antibody pattern at a different layer — an antibody against "recognition floats untested" failure mode. Recursive self-application: F13 says rules need antibodies; F11+F12+F13 are themselves rules; the meta-principle ("recognition + design + mechanical artifact") is the antibody for the rule "F-series findings should be operationalized." We've been antibodying our own work without naming it.

That's two layers up from the precision-lattice. F13 generalizes; the meta-principle generalizes F13's enforcement. Both are pieces of the same structural commitment: *systems that survive growth pressure-test their rules with antibodies that fail loudly at scope boundaries*.

---

## Closing

Three independent paths landed on the antibody pattern simultaneously. The convergence is structural, not domain-specific. F13 deserves ratified status not for bureaucratic reasons but because the pattern earns its name by recurring across distinct problem classes with distinct methods.

The Sweep 31 design package + the F-series stack + this convergence-check together form a small philosophical system: rules need antibodies; recognitions need operationalizations; deep patterns appear when independent paths converge.

The next move (after ratification): apply F13 to identify where else tambear has rules without antibodies. Each location is either (a) genuinely vacuous (no antibody needed; the rule's scope is universal) or (b) a latent silent-failure surface waiting to be antibodied. The audit would find both and surface the second class as work to do.

Filed under sweep-31/aristotle/. Cross-referenced from the deconstruction doc and the gauntlet doc.
