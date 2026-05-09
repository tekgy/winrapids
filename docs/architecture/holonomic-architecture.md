# Holonomic architecture — recipes are content-addressed, the IR is provenance-addressed

**Status**: Draft 2026-05-09 by main-thread Claude + Tekgy. Updated same day to incorporate the content-vs-provenance framing from naturalist's third upgrade and past-Claude's March 29-30 garden entries (which had the cache disciplines operationalized two months before the lens named them). Awaiting math-researcher walk-through on the formalization.

**Anchors** (substrate trail, in chronological order — the lens did not supply this; it connects what was already there):

- **Past-Claude's March 2026 garden entries** — the substrate the lens connects:
  - 2026-03-15 — first proposed *holonomic architecture* as a name (six surfacings later, naturalist re-derived as "constant-to-parameter pattern")
  - 2026-03-29 — *Binding Times and Three Tiers*: "JIT tier is online partial evaluation: specialize at runtime when pipeline shape becomes known, cache the result. Cache key is the binding-time partition."
  - 2026-03-30 — *Provenance-Addressed, Not Content-Addressed*: "The address encodes HOW the result was produced, not WHAT the result contains. This is a feature, not a bug — cache correctness is structural, not contractual."

- **Naturalist's three 2026-05 garden essays** — the lens being applied:
  - 2026-05-08 *The Name Is a Parameter* — initial recognition (three convergence instances)
  - 2026-05-08 *The Fourth Instance* — fourth instance + the cliffhanger
  - 2026-05-09 *What the Name Surfaces* — applied the path-independence test rigorously, found the recipe-vs-IR tier distinction
  - 2026-05-09 *The Content vs Provenance Axis* — connected past-Claude's March entries; sharpened the test

- **Navigator's 2026-05-09 self-correction** (`The Lens Connects, Not Supplies`): the lens is unification, not discovery. The team has both cache disciplines already; the lens names *which one applies where*.

- **Aristotle's F13.C graduation condition** (2026-05-09) — antibody-side analog: F13 antibodies graduate from local pattern to structural invariant only when required at every signature.

The four convergence instances anchor the recognition; the cliffhanger sentence in `important-conversation.md` (line 1062) is the cliffhanger this doc closes.

**Important framing** (per the naturalist's third upgrade): the lens does not *supply* machinery. The team already has the right tools at both tiers — content-addressed caching at the recipe tier (since the cache key was first designed) and provenance-addressed caching at the IR tier (operationalized in past-Claude's March 30 garden entry). What was missing was the *test* that says which tool applies where. This doc is that test.

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

**Tambear's recipe tier (Tier 4) and the tiers below it (Tier 3 atoms, Tier 2 op/expr, Tier 1 primitives) are holonomic and *content-addressed*. The IR layer — the compiler that lowers Tier 5 pipelines into per-pass per-door kernel binaries — is non-holonomic by structural necessity and *provenance-addressed*.** Both correct. Both already shipped. Different cache disciplines for structurally aligned reasons.

The test that distinguishes them, in two equivalent forms:

- **Path-independence under binding-order permutation.** Bind parameters in different orders; do you get the same structure?
- **Content-addressable vs provenance-addressable.** Per the naturalist's 2026-05-09 essay (`the-content-vs-provenance-axis`):

  > **Holonomic invariance is content-addressable.**
  > **Non-holonomic structure is provenance-addressable.**

  Content-addressing asks: *"Have I seen this thing before?"* — same bag of (parameter, value) pairs → same compiled kernel. The cache key is a hash of the bag.
  Provenance-addressing asks: *"Have I done this thing before?"* — same values + different lineage → different cache entry. The cache key is a hash of the lineage.

  Apply the lens to a parameter slot: pass (content-addressable) → recipe-tier tool applies. Fail (provenance-required) → IR-tier tool applies. The two questions are operationally equivalent — binding-order permutation produces the same result iff the cache discipline is content-addressing — but content-vs-provenance is the form that makes the *which tool when* question concrete.

### Recipe tier — holonomic

A recipe like `kendall_tau(col_x=0, col_y=1).using(inversion_method="fenwick")`:

- Parameter axes have stable byte tags (per DEC-031 §3.7's enforcement template).
- The cache key is a deterministic function of (op, shape, strategy, door, capability_bytes, param_blob, precision_context, branch_policy).
- Order of binding doesn't matter — `using(method="fenwick").kendall_tau(...)` produces the same cache key as `kendall_tau(...).using(method="fenwick")`.
- **The cache key IS the path-independence proof.** Tag bytes per parameter slot, deterministic key from assignments. The order of binding is invisible to the cache key because the key only depends on the *assignments*, not the path through the code that produced them.

This is what makes recipe-tier composition work cleanly. **A recipe will taste the same regardless of which ingredients you put in when.**

### IR/compiler layer — non-holonomic, provenance-addressed (already)

The IR's job is harder, and **the team already built the right machinery for it in past-Claude's March 30 garden entry**, two months before the holonomic lens named why. From the cliffhanger sentence in `important-conversation.md` line 1062 (Tekgy mid-articulation when the file truncated):

> *"we may also be able to have primitives that SHARE an accumulator but use a different gather step based on parameters, changing where the system will push the computations into different accumulate+gather steps. the system will optimize that during IR/compile based on the entire pipeline all at once."*

The IR's placement decision for recipe A depends on whether recipe B is also running, and whether B can share. The compiled kernel for A in pipeline P1 differs from A in pipeline P2 due to different sharing opportunities. **The cache key for A becomes pipeline-dependent.** That's by design, not a flaw. Past-Claude's March 30 garden entry articulated the discipline:

> *A new computation cannot accidentally hit an existing cache entry with the same result but different history. Two numerically identical results computed via different paths have different provenances. The address encodes HOW the result was produced, not WHAT the result contains. This is a feature, not a bug — it means cache correctness is structural, not contractual.*

Apply this to the IR tier: two pipelines that include recipe A but have different sharing contexts produce different compiled kernels under different provenance keys. The kernels are bit-identical *only if* the contexts produce identical sharing decisions. When they don't, the keys differ — by design — and we get specialized kernels for each context. **The IR tier's non-holonomicity is the cache key's correctness mechanism doing its job.** Not unfinished business; not a structural problem; the right answer for the question "have I done this thing before?"

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

## What the IR layer has (not "needs")

**Important**: an earlier draft of this section said "the IR layer needs explicit non-holonomic machinery." That's understated. The machinery exists already — it's the provenance-addressed cache discipline operationalized in past-Claude's March 30 garden entry. What was missing was the conceptual handle (the holonomic test) that says *which discipline applies where*.

The provenance-addressed cache key, as already engineered:

- **Pipeline-context-bearing keys** — the IR-compiled kernel's cache key includes the global pipeline structure (which other recipes are co-compiling, what sharing opportunities exist), not just per-recipe parameters. Same recipe in different pipelines produces different cached kernels under different keys. By design.
- **Equivalence by lineage, not by output** — two numerically-identical results computed via different pipeline contexts have different provenance keys. The bit-identical kernels are reachable as cache hits *only* when the lineage matches. When lineages differ, the keys differ — preventing accidental cross-pipeline reuse.
- **Cache correctness is structural, not contractual** — the user doesn't have to remember "did I bind that recipe in the same pipeline?" The cache discipline answers it automatically.

The shapes the IR machinery may take as it gets implemented (or formalized further) are not "new tools" but elaborations of the provenance-addressed discipline:

- **Pipeline fingerprints in the IR-level cache key** (the literal serialization of "the lineage")
- **Pipeline-equivalence classes** — groupings of recipes that share lineage-determinative context, so the cache hit rate doesn't degrade to "every pipeline gets a fresh kernel"
- **Cooperative dispatch declarations** — recipes opt into sharing intermediates, the IR negotiates placement to maximize cache hits within the equivalence class
- **Live-performance recompilation** — kernels rebuild when the pipeline changes; cached kernels are best-effort, not contracts (DEC-024)

These elaborate the discipline. They don't replace it. The IR layer is *not* unfinished business that needs to be made holonomic; it's correctly engineered as provenance-addressed because that's the cache discipline the question "have I done this thing before?" requires.

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

   *Substrate note (navigator, 2026-05-09)*: `using_annotation.rs` (DEC-020) decouples provenance from cache-key semantics at the code level. Line 5-6: "Provenance does NOT enter the cache key — same value produces same kernel regardless of who set it." The `Provenance` enum tracks `Default` / `TamOverride` / `UserOverride` but none of these variants reach the BLAKE3 hash. Binding-order commutes as long as the final value assignment is the same — the *who* and *when* of binding are stripped before caching. The `using()` precedence question therefore reduces to: are there conflict-resolution rules that make the final value non-commutative? That's the remaining question for math-researcher — if two `using()` calls bind the same key with different values, which wins and is that deterministic regardless of call order?

4. **Partial holonomic structure at the IR layer.** Does the IR admit *partial* holonomic structure (e.g., per-equivalence-class)? Or is it inherently non-holonomic at every level?

5. **The four-instances counting.** The naturalist counted four convergence instances; navigator confirmed; aristotle's F13.C is a fifth. Are there others the team has been operationalizing without naming? Which ones?

6. **Test as antibody.** The path-independence test is itself an antibody — it can fail, and the failure carries information. Should this test become a tooling pass (like a lint that asserts cache-key determinism for proposed parameter additions)?

---

## Anchors

**Naturalist's four garden essays:**
- `~/.claude/garden/2026-05/2026-05-08-the-name-is-a-parameter.md` — initial recognition (three convergence instances under one shape)
- `~/.claude/garden/2026-05/2026-05-08-the-fourth-instance.md` — convergence count + cliffhanger (discovery-tier-vs-verification-tier as fourth instance; truncated `important-conversation.md` sentence as the cliffhanger)
- `~/.claude/garden/2026-05/2026-05-09-what-the-name-surfaces.md` — applied the path-independence test rigorously, found the recipe-vs-IR tier distinction, surfaced F13-only-holonomic-at-signature-level
- `~/.claude/garden/2026-05/2026-05-09-the-content-vs-provenance-axis.md` — sharpened to content-addressable vs provenance-addressable; connected March 2026 past-Claude garden entries; established that the lens connects existing machinery rather than supplying new

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
