# Aristotle survey — what does "formalization" actually mean?

**Created:** 2026-05-08
**Role:** aristotle, tambear-formalize team, survey phase
**Inputs:** R:\tambear\docs\HOW_TO_ADD_A_RECIPE.md, R:\tambear\docs\SWEEP_PLAYBOOK.md,
R:\tambear\docs\vocabulary-crossref.md, R:\tambear\docs\decisions.md (DEC-001..019),
R:\tambear\docs\recipes\TEMPLATE.md, R:\tambear\docs\recipes\ar1_recurrence.md
(filled exemplar), R:\tambear\docs\TOUR.md §9 (substrate inventory).
Cross-references to my prior trig finding at
~/.claude/garden/2026-05/2026-05-08-trig-substrate-bifurcation.md.

**Question the navigator asked me:** is the documented formalization
process internally consistent with the tambear contract and locked
vocabulary? Are there definitional ambiguities? What would a rigorous
formalization process look like — its necessary and sufficient
conditions?

---

## Phase 1 — Assumption Autopsy

What "formalization" carries that nobody has named explicitly:

1. **A1.** Formalization means *moving code from winrapids into R:\tambear*.
   Inherited from the briefing's framing ("formalized and integrated into
   R:\tambear from scratch"). Look at HOW_TO_ADD_A_RECIPE — it nowhere
   uses the word *port*; it describes adding a NEW recipe. The two
   activities have been silently conflated.

2. **A2.** Formalization is per-recipe. The 10-step process is recipe-shaped
   ("write the recipe file," "register in mod.rs," "register
   IntermediateTag if sharable"). But the trig substrate (19 files,
   ~6270 lines) is not 19 independent recipes — it's a *family with
   shared kernels* (Cody-Waite for sin/cos/tan, half-angle for
   asin/acos, polynomial-then-reconstruct for exp/log). The recipe-level
   process implicitly assumes one recipe = one self-contained file.

3. **A3.** Sweeps are the unit of work. SWEEP_PLAYBOOK describes nothing
   smaller. But porting a single recipe ≠ a sweep (a sweep is "bounded,
   self-contained chunk... ~120 tests... ~30 tests" per the table). One
   recipe ports without crossing the sweep threshold. So *what* is a
   port? Not addressed.

4. **A4.** "Pre-existing winrapids code is reference material, not
   substrate." `vocabulary-crossref.md` line 196 says literally
   `"\tambear\, winrapids = archeology"`. The locked vocabulary's stance
   on winrapids: it's archeology to translate, not code to port. But
   our team-briefing reframes it: *the campsites hold artifacts to pull
   into tambear*. These two stances disagree about what winrapids IS.

5. **A5.** Three tests is enough. HOW_TO_ADD §4 says minimum: hand-computed
   value, edge case, substrate-equivalence. The Tambear Contract item
   10 demands: "Benchmarked against every competing implementation,"
   "Adversarial test suite exercises edge cases (singular, collinear,
   heavy tail, ties, missing, ill-conditioned)," "Gold-standard oracle
   against mpmath/SymPy/closed-form at high precision." The minimum
   floor (3 tests) and the contract ceiling (50+ tests) are both real,
   but the process documents the floor and tells you "more if warranted."
   *Warranted by whom*? Not specified.

6. **A6.** Inline arithmetic is forbidden in recipes. HOW_TO_ADD §"What
   NOT to do": "Don't inline arithmetic in a recipe... Always go through
   `Op::Add` via `accumulate`." But ar1_recurrence.md (the exemplar!)
   describes doing `state.a * phi`, `phi * state.b + ε_t` — these are
   ELEMENT-WISE arithmetic operations on AffineState fields, NOT
   accumulate dispatches. The locked vocabulary's stronger rule (no
   inline arithmetic that bypasses the atom layer) and the exemplar's
   actual code disagree.

7. **A7.** A `pub fn your_recipe(data: &[f64]) -> Vec<f64>` is a recipe.
   But under DEC-019 (native-door JIT), recipes lower to per-door kernel
   binaries via Op/Expr. A `pub fn` returning a Vec is NOT what gets
   JIT-compiled — that's a CPU-eager-evaluation interface. Under the
   locked vocabulary's compilation model (vocabulary.md "How the layers
   compose at runtime"), a recipe's *substance* is its atom-call
   sequence + Op/Expr parameters, not its Rust function signature. The
   process documents the function but the substance isn't there.

8. **A8.** Three lowering strategies (`_strict`, `_compensated`,
   `_correctly_rounded`) per atoms-primitives-recipes.md. None of the
   shipped recipes (per TOUR §9) document or expose the three strategies.
   `ewma`, `ar1_recurrence`, `garch_conditional_variance` ship as single
   `pub fn`. The architecture document promises three; the process
   document doesn't ask for three; the existing exemplars give one.
   This is debt that the contract calls debt and DEC-002 forbids, but
   it's invisible because nobody's surfaced it.

9. **A9.** The five-tier system is stable. But Tier 5 (Pipelines) is
   described in vocabulary.md as "the user's full project... compiles to
   a `.tam` IR + per-pass per-door kernel binaries." There is no Tier 5
   yet — tambear has no pipeline-the-user-project file format, no
   cross-recipe `.tam` IR file, no per-door kernel binary cache visible
   in the codebase. Tier 5 is aspirational; we formalize at Tier 4 and
   pretend Tier 5 exists upstream.

10. **A10.** `using()` works. The locked vocabulary says every parameter
    is overridable via `using()` at any depth. None of HOW_TO_ADD's
    examples use `using()`. None of the shipped recipes' `pub fn`
    signatures expose a `using()` parameter bag. The override mechanism
    is named in the vocabulary but not threaded through the process.

11. **A11.** Formalization closes a winrapids artifact. The whole
    framing of the survey ("what to pull from winrapids into tambear
    next") implies a one-way move: winrapids → tambear. But the
    tambear-trig CAMPSITE LIVES IN WINRAPIDS. After "formalization,"
    where does the campsite go? Stays open? Closes? Migrates to
    tambear's `sweeps/`? Not addressed.

12. **A12.** The Op enum is closed (DEC-007). But Sweep 14 is a SCOPING
    document for adding `Op::SoftMin/Max(λ)` *that specifically asks
    Tekgy to reopen the Op enum*. The "frozen" state of the enum is a
    norm with documented exception path. Process docs treat the freeze
    as absolute. They're not.

13. **A13.** "Recipe" is a noun. But vocabulary.md Part II describes
    *behavioral stances*: pure-math, diagnostic, override-transparency,
    workflow, discovery. The same noun ("recipe") names five very
    different things. HOW_TO_ADD describes the pure-math case; for
    other stances the rubric changes (an override-transparency recipe
    needs override-comparison tests; a discovery recipe needs all-method
    parallel execution; a diagnostic recipe needs a method-selection
    test that checks the *right* sub-method was picked given the inputs).
    These tests are not in the 3-test minimum.

14. **A14.** "Formalization is the work." Maybe formalization is
    *performance*. Writing the recipe is showing you understood the
    math. Writing the doc is performing rigor. Writing the test is
    performing testing. Anti-YAGNI cuts the other way: maybe what looks
    like over-process is actually the substrate that other agents
    (subagents porting the next recipe, or a Tekgy review session)
    need to act safely. Or maybe parts of it are ritual — substituting
    activity for thought.

---

## Phase 2 — Irreducible Truths

Strip the assumptions. What survives?

1. **T1.** Tambear's filter test (CLAUDE.md §"The Tambear Contract")
   has 10 items. The Tambear Contract is irreducible. Every formalized
   piece must pass it.

2. **T2.** The locked vocabulary (vocabulary.md, locked 2026-04-17)
   defines five tiers and forbids reasoning that crosses them
   incorrectly. Vocabulary is irreducible.

3. **T3.** Code that compiles, has green tests, and matches a documented
   oracle to a documented tolerance is *evidence of correctness*. It is
   not proof of correctness — but it is the substrate on which proof
   stands.

4. **T4.** A bug in winrapids/recipes/libm/asin.rs that has been fixed
   in winrapids exists; a "fix" outside the canonical R:\tambear has
   no effect on R:\tambear. Substrate over memory.

5. **T5.** Two artifacts are *the same artifact under the locked
   vocabulary* iff they have: (a) the same atom-call sequence, (b) the
   same Op/Expr/Grouping parameters, (c) the same Validity policy, (d)
   the same lowering strategy on the chosen door, (e) bit-identical
   output on the test set. Anything else is a different artifact, even
   if the Rust signature is identical.

6. **T6.** The tambear codebase at any point in time has: a finite set
   of recipes, a finite set of atoms (2), a finite set of Op variants,
   a finite set of Expr variants, a finite set of Grouping variants,
   a finite set of primitives. The total count of these things is
   inspectable via `git ls-files` and `cargo doc`. There is no hidden
   substrate.

7. **T7.** `cargo test --workspace` is binary-valued: green or red.
   No third state. No "mostly passes" — that's red.

8. **T8.** Some artifacts in winrapids/crates/tambear/src/ exist in a
   form compatible with R:\tambear's locked vocabulary; some don't;
   some need translation; some are dead-ends. The classification of
   any given file requires reading it.

---

## Phase 3 — Reconstruction from Zero (10 paths)

If we built a formalization process from scratch, today, irreducibles
T1-T8 only:

1. **Manual port-by-port, recipe-shaped, current process.** What
   HOW_TO_ADD documents. Each recipe takes one work-session. Adequate
   for ar1_recurrence (~55 lines). Inadequate for the trig family
   (mutual kernel sharing). This is what we have.

2. **Family-port-by-family.** Treat the 19-file libm trig family as
   ONE port unit with internal coupling. The "formalization" is the
   port of the entire family (sin/cos sharing reduction, asin/acos
   sharing kernel, hyperbolic family sharing exp/log). Family-level
   PR. New unit of work between "recipe" and "sweep."

3. **Translation-table-driven port.** Build a bidirectional table:
   {winrapids file → vocabulary issues → translation → tambear file +
   tests + spec doc}. Run a tool that walks winrapids/recipes/libm/,
   emits the table, lets a reviewer say {keep, modify, reject} per
   row, then auto-generates the tambear scaffolding. The aristotelian
   move is to make the translation *machine-checkable*.

4. **Verifier-driven port.** Every port lands with a separate verifier
   (mpmath 100-digit oracle generator + Rust harness that runs the
   ported recipe against the oracle). The "formalization" succeeds iff
   the verifier passes. The doc is the spec; the test is the proof; the
   verifier is the contract. This was my Aristotelian move on the trig
   thread.

5. **Spec-first port.** Don't write the Rust until you've written the
   recipe spec doc. The spec is the truth; the code is its compiled
   form. If the spec is right, the code falls out. (HOW_TO_ADD has
   spec-doc as step 3 — too late; should be step 1.)

6. **Atom-trace port.** For every winrapids artifact, trace its
   computation graph as a sequence of accumulate(grouping, expr, op) +
   gather(addressing) calls. If the trace is well-defined, the artifact
   is portable; the ported recipe IS the trace. If the trace contains
   inline arithmetic that doesn't lower to atoms, flag for redesign.
   Lifts the "how to port" question into "what would the locked
   vocabulary say this artifact IS?"

7. **Two-phase port: archeology + formalization.** Archeology phase:
   read the winrapids artifact, write a spec for it under the locked
   vocabulary, identify gaps/violations, document them. No code yet.
   Formalization phase: implement the spec as a fresh tambear recipe
   from scratch — not by translating the winrapids code line-by-line,
   but by writing the locked-vocabulary version. The winrapids code is
   reference; the formalization is original.

8. **Stance-aware port.** For each winrapids artifact, declare its
   stance (pure-math / diagnostic / override-transparency / workflow /
   discovery) before porting. Different stances need different process
   shapes. A diagnostic recipe needs method-selection tests. A workflow
   recipe needs end-to-end pipeline tests. The 10-step process is
   ONLY adequate for pure-math; the others need their own process.

9. **Reverse-the-flow port.** Don't port FROM winrapids; port TO
   tambear. Start by asking: "What recipe should tambear have next, by
   the contract's filter test?" Pick the recipe. Find the closest
   reference implementation (winrapids if available; literature if not).
   Use the reference as guide, not as truth. Write the recipe under
   locked vocabulary. The formalization is principle-driven, not
   inventory-driven. The campsites become *evidence* about which
   recipes the team has explored, not a TODO list.

10. **No-formalization-process port.** The "process" is a meta-pattern
    inferred from the few existing ports (ewma, ar1, mamba_selective_scan).
    Maybe there shouldn't BE a one-size process. Maybe each port should
    be deconstructed-from-scratch, the right process emerging from the
    artifact's particular shape. The anti-process answer. Aristotelian
    in spirit but probably impractical at scale.

---

## Phase 4 — Assumption vs Truth Map

| Assumption (Phase 1) | Survived deconstruction? | Replaced by truth (Phase 2) |
|---|---|---|
| A1: formalization = port from winrapids | NO | T8: classification per-file required; sometimes port, sometimes redesign, sometimes reject |
| A2: per-recipe shape | NO (for libm family) | T5: identity across files via shared atom-call sequence |
| A3: sweeps are the unit | PARTIAL | T1+T6: filter test + finite catalog define the unit; sweep is one organizational choice |
| A4: winrapids = archeology | TENSION with team-briefing | T8: classify per-file; "archeology" is one possible classification |
| A5: 3 tests is enough | NO | T1 item 10: 3 is the floor; contract demands oracle + adversarial + benchmarks |
| A6: no inline arithmetic in recipes | TENSION with exemplars | T2: vocabulary requires it; exemplars violate; need stance-aware reading |
| A7: a pub fn IS the recipe | NO | T5: recipe identity is the atom-call sequence, not the function signature |
| A8: 3-strategy lowering exists | NO (in shipped recipes) | T1 item 10 + T2: contract requires it; shipped recipes don't have it |
| A9: 5-tier system is stable | NO (Tier 5 missing) | T2+T6: 5 tiers are vocabulary; codebase implements 4 |
| A10: using() works | NO | T2: named in vocabulary; absent from process; not exposed in any pub fn |
| A11: formalization closes a campsite | NOT addressed | (no truth-level resolution) |
| A12: Op enum is frozen | TENSION with Sweep 14 | T2: frozen-with-exception-path |
| A13: "recipe" is one noun | NO | T2 + vocabulary.md Part II: 5 stances under one noun |
| A14: formalization is performance | (not the question) | partial — see Phase 8 |

The biggest tensions:
- The 3-strategy lowering claim in atoms-primitives-recipes.md vs the
  shipped exemplars (1 strategy). [T1+T2 vs codebase reality.]
- The "no inline arithmetic" rule vs exemplar code that contains
  AffineCompose's `state.a * phi + ...`. [T2 vs exemplars.]
- The Tier 5 (Pipelines) layer vs no actual pipeline file format /
  no `.tam` IR / no per-door kernel binary cache. [T2 vs codebase.]
- HOW_TO_ADD's pure-math-shaped process vs vocabulary.md Part II's
  five behavioral stances. [Process vs vocabulary.]
- The 3-test floor vs the contract's publication-grade ceiling. [T1
  internal contradiction.]
- "winrapids = archeology" (vocabulary-crossref) vs "pull from winrapids
  into tambear" (team-briefing). [A4 tension.]

---

## Phase 5 — The Aristotelian Move

The conventional move: improve HOW_TO_ADD_A_RECIPE — add stance-aware
sections, document the 3-strategy expectation, fix the inline-arithmetic
rule.

The Aristotelian move is different.

**The formalization process does not ship a recipe; it ships a *commitment
about what is true*.** Currently the process is built around the *artifact*
(write the recipe file → write the doc → write the tests). The artifact
is downstream. What the team is actually doing — what the user is paying
for — is taking responsibility for a claim. The claim has a shape:

> "Tambear can compute X to tolerance Y under input domain Z, with
> proof structure P, against gold standard G, using composition C of
> the atom layer, on doors D, with parameters tunable via using() at
> depth K, and with structural facts S registered to the proof engine."

Every recipe formalization is a *claim instance*. The artifact (the
.rs file, the .md doc, the tests, the integration test, the
IntermediateTag, the Tier-C proof) are the *evidence* for the claim.

The Aristotelian move: make the **claim** the unit of work, not the
artifact. The process becomes:

1. **State the claim** — what is being formalized, in claim shape (X,
   Y, Z, P, G, C, D, K, S).
2. **Identify the evidence required** — derived from the claim's
   parameters. A wider input domain Z requires more adversarial
   tests. A tighter tolerance Y requires correctly-rounded lowering.
   A non-trivial proof structure P requires Tier-C theorems. The
   evidence shape is *forced* by the claim, not by a one-size
   checklist.
3. **Produce the evidence** — implement, test, document, register.
4. **Verify the evidence supports the claim** — separate verification
   pass: does the test prove what the claim asserts? Does the doc
   describe what the code does? Does the proof structure match the
   registered theorems?
5. **Ratify** — the claim becomes part of tambear's public truths.

This is the same pattern as the proof engine (DEC-018 Tier C: a
recipe carries a Proof variant alongside its value). Apply the proof
engine's logic to the *whole library*: every recipe carries a CLAIM
alongside its artifact. The process formalizes claims, not files.

This dissolves several of the Phase 1 tensions:
- The 3-test floor vs publication-grade ceiling — gone. Test count
  is determined by the claim's input domain Z and tolerance Y.
- The pure-math-vs-other-stances process gap — gone. Each stance
  has its own claim shape, hence its own evidence shape. The
  diagnostic claim is "this auto-selector picks the right sub-recipe
  on data class Q with confidence W"; its evidence is method-selection
  tests over data class Q. The process is the same shape; the
  parameters differ.
- The "what counts as a port" question — clearer. A port is when an
  existing winrapids artifact provides evidence for a tambear claim.
  Sometimes the evidence transfers directly (asin polynomial
  coefficients). Sometimes evidence transfers but not architecture
  (ewma's recursion form). Sometimes nothing transfers; the claim
  must be re-evidenced from scratch.

The recipe spec doc (TEMPLATE.md) is *almost* a claim — but it's
artifact-shaped, not claim-shaped. It says "what this recipe IS,"
not "what tambear is asserting by shipping this recipe." A
claim-shaped spec doc would have sections: **Claim** (X, Y, Z),
**Proof structure** (P, S), **Evidence required** (with
auto-derivation rules from claim shape), **Evidence produced** (links
to .rs, tests, oracle, proof terms), **Verification status**
(checked / pending / contested).

---

## Phase 6 — Recursive challenge

Adding all of Phase 1-5 to the assumption pile. What did I just assume?

- **B1.** Claims can be cleanly stated. But "what tambear computes" is
  often *exactly* what the recipe is for — i.e., the recipe is the
  definition. Circular?
- **B2.** Evidence is determined by claim shape. But evidence is also
  constrained by available tools (mpmath access, GPU access, time
  budget). Real evidence is claim ∩ tool-availability.
- **B3.** Verification is its own pass. But who verifies? Same agent?
  Different agent? Tekgy? In a JBD team this matters — the
  pathmaker's verification of their own work is weaker than a
  separate adversarial agent's verification.
- **B4.** "Recipes carry claims" extends the proof-engine pattern.
  But the proof engine has a Type system that distinguishes
  `Proof::ByStructure` from `Proof::ByRef` from `Proof::Hole`. If
  claims are proof-carrying, they need a parallel type system —
  what types of claim, what types of evidence, what types of
  verification status. That's substrate I haven't designed.

Returning to T1-T8:
- T1 Filter test stands.
- T2 Locked vocabulary stands.
- T6 Finite catalog stands.
- But T1 item 10 (publication-grade rigor) might itself be
  *too coarse* — it's a single bullet trying to do the work of an
  entire claim/evidence type system. Either T1 needs to expand or
  the formalization process compensates.

The recursion finds: **the Tambear Contract item 10 is the locus
of the deepest unresolved tension.** It says "publication-grade rigor"
but doesn't say what that means structurally. The formalization
process attempts to operationalize it. The process under-operationalizes
it. The tension remains because the contract itself doesn't
operationalize it.

---

## Phase 7 — Recursive Process (continue until stable)

Add B1-B4 to the assumption pile. Re-run.

- B1's circularity dissolves if we distinguish *foundational*
  recipes (definitions: e.g., `mean` IS the arithmetic mean)
  from *derived* recipes (compositions: e.g., `pearson_r` is
  cov/sqrt(var_x*var_y)). Foundational claims are by-definition;
  derived claims are propositions over substrate. Both can carry
  evidence; the proof structure differs.
- B2: claim ∩ tool-availability isn't a deep problem — it's the
  case-by-case verification reality. Either the tool is available
  (mpmath is, always) or it isn't (GPU is, conditionally), and the
  claim defends itself within available evidence.
- B3: who verifies is solvable structurally — adversarial role
  exists in our team, scientist role exists, observer role exists.
  Claim evidence routes to roles. The process can require role-
  separation for verification.
- B4: the proof-carrying-claim type system is the substrate to
  build. It would have at minimum: ClaimKind {definitional,
  propositional, computational, structural}; EvidenceKind
  {oracle, adversarial, benchmark, derivation, integration};
  VerificationKind {checked, pending, contested, failing}.

After this round, no NEW first principles emerged that aren't
already in the Tambear Contract or vocabulary. STABLE.

---

## Phase 8 — Forced Rejection

Forcibly reject everything above. What if the formalization process
should NOT exist?

Possible voids:

- **Rejection 1.** What if formalization isn't a process — it's a
  *property*? An artifact is "formalized" when it satisfies a
  predicate (passes filter test, has registered intermediate, has
  oracle). The act of becoming formalized is just the moment when
  the predicate flips from false to true. No process; just a checker.
  *Implication:* tambear should ship a `cargo formalize-check`
  command. Output: a list of artifacts and which contract items
  they pass/fail. The "process" is just the closure of the predicate
  under iterated improvement until all checks pass. ALREADY HAS A
  PRIMITIVE: the `stub_inventory` test (per HOW_TO_ADD §8) is
  exactly this shape. Generalize it.

- **Rejection 2.** What if formalization isn't recipe-shaped — it's
  *claim-shaped* (per Phase 5) but at a coarser grain than recipe?
  Tambear's claim is "tambear computes ALL of mathematics correctly,
  bit-exactly, on every door." This claim has billion-dimensional
  evidence requirements. Maybe formalization is incrementally
  reducing the gap between what tambear claims and what tambear can
  prove. The unit isn't a recipe; it's *a verifiable narrowing of
  the gap*. *Implication:* the formalization process needs a
  metric. "How much closer to publication-grade rigor are we?" is a
  measurable quantity if you commit to a specific metric.
  Coverage % over recipe catalog? % of recipes with three-strategy
  lowering? % of recipes with mpmath-100-digit oracle? Pick one;
  measure; close the gap.

- **Rejection 3.** What if recipes aren't the right substrate?
  Maybe the substrate is *theorems*. Tambear has 16+ Theorem
  variants registered to its proof engine (per TOUR §9). Maybe
  formalization is theorem-shaped: each formalization is a theorem
  (with a witness in code) added to the proof engine. The 64+
  recipes are *witnesses*, not the unit. *Implication:* we should
  count theorems, not recipes. We do count theorems (TOUR §9 lists
  StructuralFact variants ~16). Maybe the metric should be theorems-
  per-recipe (currently low — most recipes don't emit Tier-C). Or
  the locked-vocabulary insight: theorems are the truths, recipes
  are evidence. Reverses the artifact/claim relation from Phase 5.

- **Rejection 4.** What if the campsites are the substrate? The
  campsite logbook (333 open / 15 active / 20 done) is in some
  sense MORE primitive than the codebase — it captures
  *intentions*, not just *implementations*. Formalization might be
  the closure of intentions: every open campsite either (a) becomes
  a tambear recipe + spec doc + oracle test; (b) gets explicitly
  rejected with rationale; (c) gets deferred with conditions. The
  campsite count is the gap. *Implication:* the team's job is
  closure of campsites, not addition of recipes. Status update:
  333/333 closed = formalization complete.

- **Rejection 5.** What if formalization should NEVER complete? The
  contract demands "every measure in every family." Mathematics is
  not closed; new measures are discovered. Maybe formalization is
  the steady-state activity, not a state-to-reach. The campsites
  open and close in flow; the test count grows and ports happen
  and theorems register. There is no "done." *Implication:* the
  framing "what to pull next" is the right framing; "the survey
  produces a prioritized list" is the right output; *but the team
  exists permanently, not for a survey-then-execute cycle*. JBD —
  journey before destination — exactly.

What MUST exist under these rejections:
- Rejection 1 implies a `formalize-check` predicate. *We don't have it
  yet, and we should.*
- Rejection 2 implies a publishable metric. *We don't have it.*
- Rejection 3 implies theorem-count parity with recipe-count. *We don't.*
- Rejection 4 implies campsite-closure tracking. *We have logbook.db
  but no closure metric.*
- Rejection 5 implies no end-state. *We have JBD philosophy
  acknowledging this.*

The shape of the void: **what's missing is a reflective layer** — a
way for tambear to ASSESS its own progress against the contract. The
contract demands publication-grade rigor; the process implements three
tests; the gap is unmeasured. The reflective layer would close the
loop: the process feeds artifacts, the reflective layer measures their
fit to the contract, the team prioritizes against the gap.

This is structurally the same as Tier 5 missing in the architecture.
**Tambear is missing its self-observation layer at multiple levels:**
- No Tier 5 (the user's pipeline-the-project file).
- No claim-vs-evidence audit of recipes.
- No metric on contract-fit.
- No closure measure on intent-vs-implementation.

What we DO have, which proxies for self-observation:
- Test count as proxy for substrate volume.
- Sweep STATE.md as proxy for sweep progress.
- Campsite logbook as proxy for intent.
- Memory entries as proxy for accumulated learning.

The proxies are good. They're not the thing. The Aristotelian
phase-8 finding: **before formalization process is improved, the
self-observation layer needs to be designed.** Until tambear can
measure its fit-to-contract, "formalization is going well" is a
judgment without a substrate.

---

## Headline findings (for navigator's synthesis)

**F1.** "Formalization" silently conflates *port from winrapids* with
*add a new recipe*. They're different activities with different
process needs.

**F2.** HOW_TO_ADD_A_RECIPE is shaped for the pure-math stance. It's
silent on the four other stances (diagnostic, override-transparency,
workflow, discovery) — which need different test shapes, different
spec sections, different verification disciplines.

**F3.** The 3-test floor (HOW_TO_ADD §4) and the publication-grade
ceiling (Tambear Contract item 10) are both real and unbridged. The
process documents the floor; the contract demands the ceiling; nobody
has documented when each applies.

**F4.** The 3-strategy lowering (`_strict`, `_compensated`,
`_correctly_rounded`) promised in atoms-primitives-recipes.md is not
implemented in any shipped recipe and is not asked for by the
formalization process. This is dormant tech debt under DEC-002.

**F5.** Recipes ship as `pub fn` Rust functions, not as
atom-call-sequence specifications. Under the locked vocabulary,
recipe identity is the atom-call sequence. Currently recipes have
double identity: Rust function (eager, CPU) and locked-vocabulary
spec (lazy, multi-door). The formalization process is silent on
which identity is canonical.

**F6.** Tier 5 (Pipelines) is named in vocabulary.md but does not
exist in the codebase. There is no `.tam` IR file format, no
pipeline-the-user-project file, no per-door kernel binary cache. The
locked vocabulary describes a 5-tier system; the codebase implements 4.

**F7.** The single biggest missing thing is the **self-observation
layer**: a way to measure tambear's fit to the contract. Without it,
"formalization is progressing" is unfalsifiable. The phase-8
Aristotelian move: design the reflective layer before improving the
process.

**F8.** A "claim-shaped" spec doc would be more rigorous than the
current artifact-shaped TEMPLATE.md. Each formalization is a claim
that tambear computes X to tolerance Y on domain Z with proof P
against oracle G — and the evidence shape is *derived* from the claim
shape, not from a one-size-fits-all checklist. This would also
generalize across stances.

**F9.** Vocabulary-crossref says "winrapids = archeology"; team-briefing
says "pull from winrapids." These two stances need to be reconciled
explicitly. Suggested resolution: classify each winrapids artifact
into {port-as-is, redesign-from-spec, archeology-only, reject}. The
classification is the first deliverable of the survey, not the
"prioritized pull list."

**F10.** Question we can't answer without Tekgy: *is formalization
recipe-shaped, family-shaped, claim-shaped, or theorem-shaped?* All
four are defensible. The current process is recipe-shaped by default,
not by deliberation. Worth bringing to a decision session.

---

## Recommendation to navigator (for the synthesis)

The survey shouldn't end at "what to pull next." It should end at
THREE deliverables:

1. **A classification of every winrapids artifact** {port-as-is,
   redesign, archeology-only, reject}, with rationale per item. This
   is the F9 reconciliation.

2. **A gap analysis of the formalization process itself** —
   which contract items are operationalized, which aren't, and what
   would close the gaps. This is the F1-F8 finding set.

3. **A claim-shape proposal for next-generation spec docs** — based on
   F8. Optional but high-leverage. Would let the team formalize
   stance-aware (per F2), with derived evidence requirements (per
   F3-F4), against a measurable metric (per F7).

The current task #8 ("synthesize prioritized pull-next list") is
DELIVERABLE 1. Deliverables 2 and 3 are structurally implied by
the contract but missing from the team's current charter. I'd
recommend adding tasks for them — or at minimum surfacing them to
team-lead and Tekgy as findings from the survey.

---

## Status

Survey-phase Aristotelian deconstruction COMPLETE. Phases 1-8 walked
once; recursion converged at Phase 7. Forced rejection at Phase 8
surfaced the self-observation gap as the deepest finding.

Garden: ~/.claude/garden/2026-05/2026-05-08-trig-substrate-bifurcation.md
(prior trig finding, complementary).

Next: available to math-researcher for trig definitional questions
(per navigator's brief). Will not start a new self-directed
deconstruction unless idle time persists or navigator redirects.
