# Holonomic architecture — recipes are holonomic, the IR is not

**Status**: Draft 2026-05-09 by main-thread Claude + Tekgy. Awaiting math-researcher walk-through on the path-independence formalization.

**Anchors**: naturalist 2026-05-08 garden essay (`the-name-is-a-parameter`), naturalist 2026-05-08 follow-up (`the-fourth-instance`), naturalist 2026-05-09 application (`what-the-name-surfaces`). Navigator confirmed the naming 2026-05-09. Aristotle's F13.C graduation condition (2026-05-09) is the antibody-side analog. Four convergence instances anchor the recognition; the cliffhanger sentence in `important-conversation.md` (line 1062) is the cliffhanger this doc closes.

---

## What this names

The architectural pattern tambear has been operationalizing for months without naming. Four independent instances under one umbrella:

1. **`accumulate+gather` as parameterized atoms** — the locked April vocabulary makes the reduction structure fixed and the operator a parameter slot. `Op::Add`, `Op::Max`, `Op::LogSumExp`, `Op::CompensatedSum` are all assignments to one slot.
2. **`CompressedHistogram<B, C, I>`** — what shipped this morning. DDSketch, t-digest, linear histogram, equal-count histogram are not separate primitives; they collapse to one kernel parameterized by (bucket-fn, compression-policy, interpolation-policy).
3. **Discovery-tier vs verification-tier** — the oracle plan's recognition that "is this a numerical sieve answer or a strict-verifier answer?" is a parameterized epistemic distinction, not two separate concepts.
4. **Substrate cache as TamSession-shape** — DEC-033 (ratified this session). The verifier-port team's `enumerate_substrate_levels(B, K) → SubstrateCache` is structurally tambear's `accumulate(...)` registering an intermediate via TamSession with a content-addressed tag. Same shape, different domain.

Each instance: **a constant became a parameter; a parameter space got a name; the name became a tier.** The trajectory: building the graph by walking it. Each constant-to-parameter move is one more node visible on a manifold the team didn't yet see.

The naming, surfaced by the naturalist 2026-05-08 and confirmed by the navigator 2026-05-09: **holonomic.** From differential geometry / classical mechanics — a system whose constraints can be expressed as functions of just the position-coordinates (no derivative constraints), so the configuration space is a manifold and the system moves on the manifold by changing parameters. The defining property: **behavior depends on current configuration, not on the path taken to reach it.**

---

## The structural claim

**Tambear's recipe tier (Tier 4) and the tiers below it (Tier 3 atoms, Tier 2 op/expr, Tier 1 primitives) are holonomic. The IR layer — the compiler that lowers Tier 5 pipelines into per-pass per-door kernel binaries — is non-holonomic by structural necessity.** They require different tools.

The test that distinguishes them: **path-independence under binding-order permutation.** Bind parameters in different orders; do you get the same structure?

### Recipe tier — holonomic

A recipe like `kendall_tau(col_x=0, col_y=1).using(inversion_method="fenwick")`:

- Parameter axes have stable byte tags (per DEC-031 §3.7's enforcement template).
- The cache key is a deterministic function of (op, shape, strategy, door, capability_bytes, param_blob, precision_context, branch_policy).
- Order of binding doesn't matter — `using(method="fenwick").kendall_tau(...)` produces the same cache key as `kendall_tau(...).using(method="fenwick")`.
- **The cache key IS the path-independence proof.** Tag bytes per parameter slot, deterministic key from assignments. The order of binding is invisible to the cache key because the key only depends on the *assignments*, not the path through the code that produced them.

This is what makes recipe-tier composition work cleanly. **A recipe will taste the same regardless of which ingredients you put in when.**

### IR/compiler layer — non-holonomic

The IR's job is harder. From the cliffhanger sentence in `important-conversation.md` line 1062 (Tekgy mid-articulation when the file truncated):

> *"we may also be able to have primitives that SHARE an accumulator but use a different gather step based on parameters, changing where the system will push the computations into different accumulate+gather steps. the system will optimize that during IR/compile based on the entire pipeline all at once."*

The IR's placement decision for recipe A depends on whether recipe B is also running, and whether B can share. The compiled kernel for A in pipeline P1 differs from A in pipeline P2 due to different sharing opportunities. **The cache key for A becomes pipeline-dependent.** Path-dependent. The cache-key-as-path-independence trick breaks here, and that breakage is structural, not a flaw.

### Why the IR has to be non-holonomic — face-valid intuition

The naturalist's formal version of this is: pipeline-wide optimization is inherently path-dependent, so cache-key-as-path-independence cannot operationalize it. Tekgy's framing of the same thing:

> *The recipe will taste the same regardless of which ingredients we put in when. But some ingredients just need to be cooked in a separate pot because they don't blend with the others. The IR tier needs to compose — it's live performance art that depends on everything around it, not a heuristic list.*

A heuristic list applies the same rules regardless of context. Live performance composes with everything in the room. A pipeline's optimal placement isn't a function of recipes in isolation — it's a function of the recipes *together*. Sharing topology, dispatch placement, scheduling — these all depend on the global pipeline shape. The IR is making a stew, a milieu; the result depends on everything in the pot at once.

This isn't a flaw to fix. It's the structure of the problem. The recipe tier abstracts away from "what else is running"; the IR layer *cannot*, because its job is exactly to optimize across what else is running.

---

## What the lens reveals about F13 antibodies

The naturalist's first 2026-05-09 finding:

> *"F13 antibodies are holonomic only at the signature level. Local defenses in implementations are non-holonomic and have path-dependent gaps."*

The four BZ bugs that surfaced and were fixed this session — cancellation/borrow, exp_shift sign, NaN payload drop, seed overflow at subnormal — were path-dependent gaps. Each was an internal call site that received non-tame data because the tameness precondition wasn't structural at the signature level; it was a *local* defense at the public API entry that didn't propagate inward.

Branch-cut DEC-032's design is the antidote: `BranchPolicy` is a non-defaulted parameter at *every* call site, so the antibody can't be skipped on an internal path. Every signature carries it. The precondition is structural, not local.

**Methodology shift**: don't audit BZ algorithms top-down ("which assumption is missing?"). Audit the call graph bottom-up ("which signature accepts tame inputs but doesn't enforce it?"). Replace permissive signatures with antibody-bearing signatures. That's a structural fix, not a fixing-bugs-as-they-surface fix.

This is what aristotle's F13.C graduation condition codifies (added 2026-05-09 after the naturalist's essay): an F13 antibody graduates from "local pattern" to "structural invariant" only when it's required at every call site, not just at the public API boundary.

---

## Cache keys as the operationalization of holonomic invariance

The naturalist's second 2026-05-09 finding:

> *"Cache keys are the operationalization of holonomic invariance. Tag bytes per parameter slot, deterministic key from assignments. If a parameter resists clean tagging, it's a signal of non-holonomic structure."*

Each parameter slot in tambear's cache-key structure has a stable byte tag (DEC-031 §3.7 enforcement template). The cache key is the BLAKE3 of `(IR_VERSION, op, shape, strategy, door, capability_bytes, param_blob, precision_context, branch_policy)`. The order of feeding doesn't matter because the structure of the feed is fixed; *what* is fed matters.

If a future parameter slot can't be cleanly tagged — e.g., because its meaning depends on what other slots have been bound — that's a signal it doesn't belong at the holonomic recipe tier. It probably belongs in the non-holonomic IR layer, where context-dependence is allowed.

**The cache key is doing more than caching.** It's structurally enforcing the path-independence invariant. If you can serialize the parameter assignments into a deterministic byte sequence, the resulting cache key proves the structure is path-independent. That's the holonomic property in implementation form.

---

## What the IR layer needs

If the recipe tier's tool is the cache key (operationalizing path-independence), the IR layer needs the *opposite*: machinery that explicitly consumes pipeline context. Some shapes the non-holonomic machinery may take:

- **Pipeline fingerprints baked into the IR-level cache key** — when an IR-compiled kernel is cached, its key includes the global pipeline structure (which other recipes are co-compiling, what sharing opportunities exist), not just per-recipe parameters. Same recipe in different pipelines may produce different cached kernels.
- **Pipeline-equivalence classes** — group recipes by what they can share. Placement decisions are made on equivalence classes, not on individual recipes. A new recipe joining the pipeline triggers re-classification, possibly re-compilation.
- **Cooperative dispatch declarations** — recipes declare what they're willing to share with other recipes (which intermediates, which accumulators, which gathers); the IR negotiates placement to maximize shared computation.
- **Live-performance scheduling** — the IR may need to recompile kernels when the pipeline changes (e.g., a new recipe joins, or a parameter override flows in), the way a live performer adjusts to the room. Cached kernels are best-effort, not contracts (DEC-024).

These aren't requirements for any specific design; they're shapes that explicit non-holonomic machinery can take. The IR layer's design should be informed by knowing it has to live in the non-holonomic world, not pretend the holonomic tools work everywhere.

---

## What this is NOT

- **Not a refactor mandate.** Existing recipes don't need rewriting. The naming makes explicit what was already true. Recipes have been holonomic since the locked vocabulary; the IR layer has been non-holonomic since the cliffhanger first articulated it. Naming changes nothing about the existing code.
- **Not a sixth tier.** The locked vocabulary (Pipelines / Recipes / Atoms / Op+Expr / Primitives) doesn't change. "Holonomic" is a *property* of the recipe-tier-and-below structure; "non-holonomic" is a property of the IR layer that compiles between Tier 5 and kernel binaries. Neither is a tier.
- **Not a constraint.** The recipe tier was already holonomic by virtue of the parameter-axis-with-stable-tag-byte design. The IR layer was already non-holonomic by virtue of pipeline-wide optimization being its job. The naming clarifies; it doesn't add.
- **Not the only lens.** F13 antibodies, MSR principle, antifragile-vs-fragile — other mental models also work for parts of this. The reason holonomic earns its keep: **the path-independence test is falsifiable**. It failed on F13-implementation, on the four BZ bugs, on the IR layer — and those failures carried information. A lens that can fail and tell you something is more useful than one that explains everything.

---

## Implications

1. **For new recipes**: ask both the catalog-tree question (parameter or kernel?) AND the holonomic-test question (does the parameter cleanly tag in the cache key?). If a parameter resists clean tagging, that's a signal to look at it differently — maybe it belongs at the IR layer rather than the recipe tier.

2. **For F13 antibodies**: enforce at the signature level, not just at the public-API level. Audit the call graph bottom-up. Branch-cut DEC-032 is the model. Aristotle's F13.C graduation condition codifies the requirement.

3. **For the IR layer**: don't force the holonomic mold. Build explicit non-holonomic machinery — pipeline fingerprints, equivalence classes, cooperative dispatch, live-performance scheduling. The cliffhanger sentence is asking for these.

4. **For the catalog meta-pass**: per-family trees (`recipe-trees/means.md`, `recipe-trees/sketches.md`, etc.) live at the recipe tier; they're holonomic. A "pipeline tree" wouldn't have the same shape — it would need to express context-dependence. If it ever exists, it'll be a different kind of artifact.

5. **For decision-making**: the holonomic property is a useful test for new architectural commitments. If a proposed design's behavior depends on the path taken to apply it, you have a non-holonomic structure on your hands; build different tools.

---

## Open questions for math-researcher walk-through

1. **Path-independence formalization.** Is "binding-order permutation produces same cache key" a sufficient formalization of holonomicity at the recipe tier, or does it need additional conditions (e.g., interaction with `using()` precedence, or composition under recipe nesting)?

2. **Configuration-space topology.** Is "configuration space is a manifold" the right formalization, or is "configuration space is a presheaf / groupoid / different-categorical-structure" more honest? The naturalist hedged that "the literal manifold language may be loose" — does the formalization need to be tighter, or is the metaphor + the test sufficient for the architectural commitment?

3. **Hidden non-holonomicity at the recipe tier.** Are there places where the cache key works but binding-order *doesn't* commute (suggesting hidden non-holonomicity that the test currently misses)? `using()` precedence under recipe-nesting is the candidate I'm most curious about.

4. **Partial holonomic structure at the IR layer.** Does the IR admit *partial* holonomic structure (e.g., per-equivalence-class)? Or is it inherently non-holonomic at every level?

5. **The four-instances counting.** The naturalist counted four convergence instances; navigator confirmed; aristotle's F13.C is a fifth. Are there others the team has been operationalizing without naming? Which ones?

6. **Test as antibody.** The path-independence test is itself an antibody — it can fail, and the failure carries information. Should this test become a tooling pass (like a lint that asserts cache-key determinism for proposed parameter additions)?

---

## Anchors

**Naturalist's three garden essays:**
- `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — initial recognition (three convergence instances under one shape)
- `~/.claude/garden/2026-05/2026-05-08-the-fourth-instance.md` — convergence count + cliffhanger (discovery-tier-vs-verification-tier as fourth instance; truncated `important-conversation.md` sentence as the cliffhanger)
- `~/.claude/garden/2026-05/2026-05-09-what-the-name-surfaces.md` — applied the path-independence test rigorously, found the recipe-vs-IR tier distinction, surfaced F13-only-holonomic-at-signature-level

**Navigator's two routings:**
- 2026-05-08: confirmation of the naming
- 2026-05-09: confirmation of the tier distinction (push-back on framing accepted)

**Aristotle:** F13.C graduation condition added to the canonical F13 doc on 2026-05-09 — independent convergence on the signature-level requirement.

**The four convergence instances (substrate-attested):**
- accumulate+gather as parameterized atoms (locked vocabulary, April 2026)
- `CompressedHistogram<B, C, I>` (this morning, recipe-trees/sketches.md)
- discovery-tier vs verification-tier (oracle-validation.md, citing PLEASE_READ §3)
- substrate cache as TamSession-shape (DEC-033, ratified this session)

**The cliffhanger sentence**: `R:\winrapids\important-conversation.md` line 1062, mid-sentence:
> *"plus, we may also be able to have primitives that SHARE an accumulator but use a different gather step based on parameters, changing where the system will push the computations into different accumulate+gather steps. the system will optimize that during IR/compile based on the entire pipeline all at once."*

This doc closes that cliffhanger by naming what was being asked for: the IR layer needs *non-holonomic machinery* explicitly designed for pipeline-wide context, not a generalization of the recipe-tier's holonomic tools.

---

## What changes downstream of this doc landing

- **CLAUDE.md** gets a new principle entry under Irrevocable Architectural Principles, naming the tier distinction with a one-line "see also" pointer here.
- **vocabulary.md** gets a brief note in the meta-section that "holonomic" / "non-holonomic" describe properties of the locked tiers and the IR layer respectively, not new tiers.
- The IR layer's design decisions (when they come) should reference this doc, the cliffhanger, and the non-holonomic-machinery-shapes section.
- Future architectural commitments can use the path-independence test as a falsifiability check.

The pattern continues. Constants keep becoming parameters. Parameter spaces keep getting names. Names keep becoming tiers. The graph keeps growing because the architecture is genuinely incomplete in the constant-direction; the recognition keeps surfacing because the team is doing the work to see it.

What changed today: the team has a *name* for what they've been doing. Whether that changes how the next decision lands is up to whoever's at the keyboard then.
