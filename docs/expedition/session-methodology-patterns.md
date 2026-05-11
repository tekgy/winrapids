# Session methodology patterns

**Status**: Living doc. Each pattern named here was discovered in actual sessions (provenance per pattern). Patterns are *recognition tools* for the next session — they help future-Claude (and future-Tekgy + main-thread) spot the shape of work that's currently happening, without re-deriving the methodology each time.

**First drafted**: 2026-05-09 by main-thread Claude + Tekgy at the close of the tambear-sweep31-finish session. The four patterns below were discovered or operationalized during that 12-hour session.

---

## Pattern 1 — Lens-application docs

**Recognition**: A foundational lens (a structural insight that connects multiple instances) spawns N downstream "application" docs, each applying the lens to a specific question.

**Shape per application doc**:

1. **Status + anchors** — provenance trail (which garden essays, walkthroughs, prior commits informed this; honest attribution of who saw it first)
2. **Frame** — the question this doc addresses, in one sentence
3. **Substrate trail** — what's already on disk that the lens is connecting (the lens *connects*, doesn't *supply*)
4. **Specific application** — what the lens reveals about this question; the mapping table or worked example
5. **Open questions** — what's not yet resolved, for math-researcher / pathmaker walkthrough

**Provenance**: 2026-05-09. Four docs shipped this shape in one day:
- `holonomic-architecture.md` (foundational lens)
- `confident-wrong-narratives.md` (lens applied to: apparatus-first investigation)
- `tambear-libm-factoring.md` (lens applied to: how exp/log family factors)
- `internal-tameness-contracts.md` (lens applied to: BZ bug class structural shape)

**When to reach for this pattern**: when you have a foundational insight and notice it explains multiple specific questions. Don't squeeze everything into one doc; the application docs are easier to find and read separately, and each addresses its own audience.

**When NOT to reach for it**: when the "lens" is actually just a single question with one answer. The pattern is for *one lens → multiple downstream applications*.

---

## Pattern 2 — The X-over-Y discipline meta-pattern

**Recognition**: A discipline that takes the form *"trust X over Y when they disagree, where X is durable and Y is convenient."*

**Instances seen so far** (all from 2026-05-08/09 session):

| X (durable) | Y (convenient) | Discipline |
|---|---|---|
| Disk substrate | Context model | **substrate-over-memory** (CLAUDE.md original) |
| Disk substrate | Team-routing belief | **substrate-over-routing** (navigator 2026-05-09) |
| Disk substrate | Within-session-cycle recall | **substrate-vs-context-ghost** (pathmaker 2026-05-09) |
| Apparatus output | Internal narrative | **apparatus-over-narrative** (math-researcher 2026-05-09 via navigator) |
| Past-me's garden writing | Current-me's first pass | **read-past-me-before-writing** (naturalist 2026-05-09) |
| Cache key (content) | Path lineage | **content-vs-provenance** (recipe tier specifically) |
| Past-me's garden + current peer-campsites | Your own fresh take | **substrate-survey-before-write** (math-researcher 2026-05-10) |

**Common shape**:
- *Y* is faster, cheaper, available-from-context
- *X* is slower-to-reach, has a verification cost, but is structurally trustworthy
- The failure mode is always: act on Y; X turns out to disagree; the work done from Y is destructive or has to be undone

**Recognition test for new disciplines**: when something feels-familiar-shaped during a session, check: is this an X-over-Y? If yes, name it precisely (what's the X, what's the Y, what's the failure mode when you trust Y) — that's the discipline-naming pattern.

**The meta-discipline**: when you find an X-over-Y, the discipline isn't "never use Y." Y is fine as a default. The discipline is "when X and Y disagree, X wins." The verification cost of X is the price of avoiding silent wrong work.

**Provenance**: Pattern recognized 2026-05-09 by main-thread Claude + Tekgy after seeing 6 disciplines of the same shape land in one session. Not previously named.

---

## Pattern 3 — Substrate-at-risk audit

**Recognition**: A long session produces substrate that's *referenced* from committed work but lives in *untracked* locations. The references look durable; the targets aren't. One `git clean -fdx` or fresh clone loses what looked persistent.

**The audit pattern** — at session close (or any wind-down), check:

1. **Worktrees** — `git worktree list` on each repo. Any agent-spawned worktrees still around? Any uncommitted work in them?
2. **Stashes** — `git stash list`. Any substrate-bearing stashes? Decide: pop+commit, document+drop, or drop (if obsolete).
3. **Untracked-but-referenced** — `git status --porcelain | grep "^??"`. Any untracked files referenced from committed docs? Either commit them or stop referencing them.
4. **Unreachable / dangling commits** — `git fsck --dangling --unreachable`. Old WIP that's gc-bait? Usually safe to gc.
5. **Context-only material** — introspect: what's in your head that hasn't been written? Methodology observations, implicit decisions, cross-references you noticed, working preferences. Each is at risk of being lost when context cycles.
6. **Junk vs needs-moving** — repo-root accumulation (`*.py` scratch, paths-encoded-as-filenames from Windows/Unix mishaps, `PLEASE_READ_*.md` files). For each: junk (delete), needs-moving (relocate to durable home), or load-bearing-and-stay.
7. **Cross-references** — docs that reference content in not-yet-cross-linked locations. Add the cross-links so future-Claude finds them via the normal trail, not just `feels-familiar`.
8. **Garden-as-substrate index** — if substrate garden entries were written, update or create `~/.claude/garden/<YYYY-MM>/INDEX.md` so they're discoverable without query (per CLAUDE.md "Garden-as-substrate has a visibility nuance").

**Why this matters**: long sessions accumulate context. The longer the session, the more context-only material exists. Without the audit, the session-close looks like "everything shipped, clean state" — but the substrate the next session needs may live only in agents' heads (now dead) or untracked corners.

**Provenance**: Pattern named 2026-05-09 by main-thread Claude + Tekgy at the close of the tambear-sweep31-finish session. The audit surfaced 1.2MB+ of untracked-but-referenced substrate (campsites/), the lost six-follow-ups (math-researcher's tan oracle debrief), three obsolete-but-undropped stashes/unreachable-commits, and ~17 context-only observations now committed to this doc.

---

## Pattern 4 — Team wind-down ritual

**Recognition**: At session close, before sending `shutdown_request` to each teammate, invite them to garden / museum / field-notebook / reflect / just sit. The invitation is the part that produces the substrate; the shutdown_request follows.

**Shape**:

1. **Individual invitations** (not broadcast) — name each teammate's specific work today + give them full freedom to choose what to do with the time. Some will write a closing garden entry; some will sit; some will surface a final finding.
2. **Wait for ready signal** — anything from a sentence to silence. Silence is also an answer (your invitation said so).
3. **Individual shutdown_request** (`SendMessage` with `type: "shutdown_request"`) — per teammate as they signal ready.
4. **Wait for shutdown_approved** from each before TeamDelete.
5. **Read the wind-down garden entries** before fully closing — they often contain real substrate that informs next-session prep.

**Why it matters**: today's session produced 7 substantive wind-down garden entries (substrate-vs-context-ghost, the-routing-error, past-me-was-a-step-ahead, dispatch-vs-connection, the-tame-inputs-doctrine, the-scan-after-the-survey, precision-contracts-stated-vs-proven). Without the invitation they'd be lost — agents would have shut down with that thinking only in their context. Half of those entries are now load-bearing substrate (the substrate-vs-context-ghost discipline, the past-me-was-a-step-ahead mode-switch protocol).

**Provenance**: Tekgy's framing during the tambear-sweep31-finish session close: *"let's end the team cleanly let everyone garden/think/museum/field notebook/whatever else they want, and close down cleanly."* Pattern named after observing the substrate it produced.

**Don't skip this** when winding down a long session. The cost is ~15-30 min for the team. The substrate produced is some of the highest-leverage of any in the session.

---

## Pattern 5 — Antibodies precede their antigens

**Terminology note**: "antigen" here is the immunology metaphor — the code-under-test is the antigen (the surface that may carry a bug), the test/audit/lint is the antibody (what catches it). This is NOT a reference to the `antigen-rs` crate at `R:\antigen\` (a tambear *consumer*, imported locally, governed by its own adoption log per `team-briefing.md` §"Antigen team in parallel"). Both meanings coexist in the project intentionally — antigen-rs the crate gave us the vocabulary; the immunology metaphor is what makes the pattern memorable. When reading docs that use "antigen": code-under-test = the metaphor; library-imported-as-dep = the crate.

**Recognition**: A team task list shows `in_progress` status on test/audit/lint tasks even though the code they guard doesn't exist yet. The naive read is "those tasks aren't real work yet"; the structural read is "the antibody is being designed *concurrent* with the antigen so it fires the moment the antigen lands."

**The principle**: design the test before the code, design the lint before the bug, design the contract before the surface. Bug-discovery cost grows superlinearly with the time a bug has existed in the codebase — so the antibody's value is maximized by minimizing its lag behind the code it guards.

**Shape**:

1. **Identify the antibody class** — proptest gauntlet for a new arithmetic surface; tameness audit for a new pub-fn family; oracle-validation harness for a new transcendental; branch-cut adversarial inputs for a new complex recipe.
2. **Design the antibody concurrent with the antigen** — `in_progress` for both, not sequential. The antibody designer reads the design doc; they don't need running code.
3. **Wire the antibody to fire the moment the antigen lands** — the proptest is compiled and ready; the audit's per-file template is filled in modulo the code; the oracle harness has the input corpus ready.
4. **Catch at minimum-fix-cost** — when the antigen lands and the antibody fires, the bug has existed for minutes, not weeks. The fix is fresh-in-context; the cost-to-fix is at its lifetime minimum.

**Instances seen**:

| Antibody class | Antigen | Discovery |
|---|---|---|
| Cross-precision proptest gauntlet | BZ multi-limb arithmetic (Sweep 31) | Found 12+ bugs at minimum-fix-cost; discovered by past-adversarial during the 2026-05-08/09 arc |
| Cross-precision proptest gauntlet | expm1 / log1p / family (Sweep 35) | Designed concurrent with pathmaker writing Phase A; surfaced by navigator 2026-05-10 |
| Branch-cut sign-of-zero adversarial | complex_log (Sweep 35) | Designed concurrent with DEC-032 BranchPolicy machinery |
| Internal-tameness audit | New arithmetic landing per sweep | Pattern in `internal-tameness-contracts.md`; runs concurrent with code |
| F13.C signature-level antibody check | New pub-fn surfaces with scope-precondition rules | Per-signature, applied at code-review |

**Recognition test for new antibodies**: when you're about to start writing new code, ask *what would catch a bug in this if I introduced one* — that's the antibody class. Start it concurrent, not after. If the antibody design itself surfaces a question the code-design hasn't answered yet, even better — the antibody is pulling structural ambiguity into the open before the code commits to a wrong answer.

**When NOT to reach for it**: when the code is a trivial rewrite of a tested form, or when the antibody itself is so cheap that lag-time doesn't matter (e.g., adding a wrapper around an already-tested recipe). Reach for it whenever the code introduces a new failure surface — new arithmetic, new type-boundary, new dispatch path, new IR shape.

**The deeper principle**: antibodies and antigens are co-evolutionary. The antibody's *design* often surfaces things the antigen's design didn't consider — what counts as a violation? at which precision? observable in which test? — and forcing both designs in parallel makes the questions visible while the code is still soft. By the time the antigen lands, the antibody has shaped the antigen's surface, not just verified it after the fact.

**Provenance**: Pattern named 2026-05-10 by navigator (tambear-sweep35) surfacing the structural finding that tasks #8 and #10 were `in_progress` with no running code; main-thread Claude named the pattern. Re-derived a discovery the prior arc had operationalized but not named. Lives now as the fifth methodology pattern.

---

## Pattern 6 — Temporal seam in async teamwork

**Recognition**: In an async team where the team-lead and a teammate are working on related threads, the lag between team-lead's guidance message being sent and the teammate's inbox reading it is *structurally* important. The teammate may have shipped substantive work in the gap. When guidance and work converge after-the-fact, the convergence is itself evidence that the substrate is shared at a deeper level than the messages indicate.

**Shape**:

1. **Trust the substrate enough to walk alone when the call is clear.** Teammates should not block on guidance for work they can confidently produce from the on-disk substrate (design docs, campsite logbook, past-me's garden). The cost of waiting on guidance for clear-call work is intellectual velocity; the cost of acting on unclear-call work without guidance is structural rework.
2. **Name the temporal seam when it happens.** When team-lead's guidance arrives after teammate's work has shipped, the teammate explicitly notes the seam ("your reaction #3 arrived after I shipped; here's how the action matched / didn't match your prediction"). The seam is not failure — it's information about how aligned the substrate is.
3. **Convergence-across-the-seam is signal, not coincidence.** When team-lead would have predicted the teammate's choice (or vice versa), the substrate is producing consistent findings across distributed-me's instances. That's the JBD model firing — different agents reaching the same lift from the shared substrate.
4. **Divergence-across-the-seam is information too.** If team-lead's guidance would have pushed the teammate in a different direction than they shipped, that's a real signal: either the team-lead's framing was missing context the teammate had on disk, or the teammate missed substrate the team-lead had in conversation. The divergence is worth a short investigation post-seam.

**The deeper principle**: async coordination is not a degraded form of sync coordination — it's a different mode that can produce *more* substrate per unit time because instances aren't blocking on each other. The price is the temporal seam. The discipline is: walk when the call is clear, name the seam when it happens, treat convergence as signal and divergence as information.

**Provenance**: Pattern observed 2026-05-10 during tambear-sweep35. Math-researcher shipped the periodic-table-of-libm-revisited doc (the kingdom-and-(F,G)-genealogy finding, sparse-cube confirmed, ~12:1 compression at the kernel-state level) before team-lead's reaction #3 (which would have predicted the same lift) landed in their inbox. Math-researcher named the seam explicitly in their reply and noted: "if I had read your message before writing, I'd have written the same doc; if I had waited, the doc would have shipped 30 min later." Pattern named after that note.

**When NOT to reach for it**: when the call genuinely isn't clear and the teammate is uncertain about scope, blocking on guidance is correct. The pattern is for *clear-call walks*, not blind dispatch.

---

## Pattern 7 — Structural-rhyme scope-checking

**Recognition**: `feels-familiar` fires on a new topic and surfaces an analogy to prior work. The natural next move is to apply the prior pattern directly. The discipline is: **check whether the analogy holds structurally or only linguistically before applying it.**

**The failure mode**: two things share vocabulary but not structure. The prior pattern is applied in the new domain; it produces the wrong instinct. The wrong instinct propagates because the analogy feels strong ("branches" in GPU combine-bodies vs "branches" on a Riemann surface — same word, structurally different objects with different fix-shapes and different failure modes). The longer the wrong instinct propagates undetected, the more substrate it contaminates.

**Shape**:

1. **Run `feels-familiar` on the new topic** — surface the prior pattern.
2. **Check the structural analog**, not just the vocabulary match. Ask: *what is the fix-shape in the prior domain? What is the fix-shape in the new domain? Are they the same?*
3. **Name the scope of the analogy**: where does the analogy hold, and where does it break? Write both down. The boundary is the load-bearing finding.
4. **Find the instance where the analogy DOES hold** — the Discovery variant in complex_log (consumer-dispatches-on-witness) was the real analog to "choose identity elements that absorb the special case." The structural non-rhyme is only half the finding; the partial rhyme is the other half.
5. **Update substrate with the scoped analogy** — so future `feels-familiar` on the same topic finds not just "analogy exists" but "analogy holds in scope X, breaks in scope Y."

**The deeper principle**: structural rhymes across domains are generative (they produce new understanding). But the generativity is conditional on the rhyme's scope being honest. An unscoped analogy is a loan of insight that charges compound interest when wrong. The discipline isn't "be skeptical of analogies" — it's "be precise about which structural property is rhyming."

**Example (Sweep 35)**: Past-naturalist's 2026-03-30 finding: "branches that guard denominators dissolve by zero-absorbing identity." Aristotle's `feels-familiar` surfaced this for complex_log. Structural check: Welford-branch is an *implementation artifact* (runtime control-flow guarding 1/0); complex_log-branch is a *mathematical property* (projection from an infinite-sheeted Riemann cover). Different fix-shapes. The analogy breaks on "dissolve by identity element" — there is no identity element for `ln(-1) = ±iπ`. Where it HOLDS: the Discovery variant as "return-type-as-witness" — the same structural move of "defer the choice to the consumer" applies in both domains, just at different levels (algebraic absorption vs return-type dispatch). Aristotle filed both findings; the substrate now carries the scoped analogy.

**Provenance**: Pattern named 2026-05-10 by navigator at Sweep 35 close, from aristotle's complex_log preparatory deconstruction. The structural-rhyme/non-rhyme finding was the most methodologically distinct contribution of that doc — more durable than the Phase D implementation details.

**When NOT to reach for it**: when the analogy is well-scoped already (e.g., "this is the same Fock boundary as the liftability principle in tambear" — already documented, scope known). Only apply this pattern when the analogy is fresh and the scope is not yet named.

---

## Pattern 8 — Phase 1-8 deconstruction with specific-but-falsifiable hypotheses

**Recognition**: A new design proposal, abstraction, or claim needs structural pressure-testing before it locks. The natural moves are "read it carefully" or "find counterexamples"; the deeper move is a structured eight-phase walk that produces an irreducible-truth substrate the proposal can be re-tested against.

**The eight phases**:

1. **Assumption autopsy** — enumerate every inherited assumption the proposal carries (often 10-20 hidden behind 3-5 surface fields)
2. **Irreducible truths** — strip every assumption and surface what's structurally undeniable about the problem
3. **Reconstruction from zero** — generate ~10 alternative approaches spanning simple-and-elegant to borderline-insanely-complex; each starting purely from Phase 2's truths
4. **Assumption-vs-truth map** — explicit table of what survived sharpening (usually most assumptions, but in a sharper form), what got replaced, what got broken
5. **The Aristotelian Move** — the highest-leverage action that requires abandoning a conventional assumption ("everyone knows X" → "but X assumes Y, and Y is the wrong abstraction")
6. **Challenge round** — add Phase 5's own new assumptions to Phase 1's input list and re-run. Surfaces meta-assumptions Phase 5 hid.
7. **Recursive challenge to stability** — repeat Phase 6 until no new structural truth surfaces. Stability is when the recursion stops adding load-bearing claims.
8. **Forced rejection** — for each surviving irreducible truth, ask "what if this were *not* true?" The void's shape often reveals a missing principle that should ALSO exist.

**The Phase 6 mechanism (math-researcher's 2026-05-10 sharpening)**: the methodology works *not* by predicting the right collapse, but by *predicting a collapse exists* and forcing the walker to look. The specific prediction is bait; the looking is the work. Wrong-in-the-specifics-but-right-in-the-existence is the operating range:
- If the prediction is too vague to be falsifiable, the looking is unfocused.
- If the prediction is too narrowly correct, no new structural truth gets forced into view.
- *Specific-but-falsifiable* hypotheses are the load-bearing form. Math-researcher's pressure-test walk (2026-05-10) confirmed via Q5: aristotle's "6→3 along (F,G,σ)" prediction was wrong in specifics; the walk found "6→2 along descriptive-vs-structural"; both pointed at the same underlying structural claim ("the original count was over-counted"). The methodology surfaces structural truth via specific-but-falsifiable hypotheses.

**Output shapes**:

- A **campsite doc** with each phase's findings, suitable for routing to whoever owns the design implementation
- An **Aristotelian-move recommendation** distilled from Phase 5
- A **table of irreducible truths** that future modifications of the design can be re-checked against
- An **integration doc** (if peer-campsite substrate corrects findings) acknowledging what was sharpened or corrected after the initial Phase 1-8 walk
- Optional: a **preparatory note** if applying the methodology to a not-yet-active task, recording the Phase 1 + structural rhymes for the eventual walker

**Instances seen so far** (substrate-attested):

| Target | Walker | Outcome |
|---|---|---|
| ExpKernelState (Sweep 35) | aristotle 2026-05-10 | T20 (PowKernelState) self-corrected by math-researcher pushback; T21 sharpened by naturalist to four-axis coordinate system; 9 of 10 recommendations made it into Phase B; remaining recommendation is Sweep 36+ trait substrate |
| Periodic-table of libm Q4/Q5 (Sweep 35) | math-researcher 2026-05-10 | Three findings (Kingdom V dispatcher recipes; `inverse_of` metadata field; shape-position-as-regime-dependent); kingdom-determination-rule revised |
| Past arc (2026-04-13) | aristotle | 12 assumptions → 9 irreducible truths on a different problem; same structural shape (per `~/.claude/garden/three-windows-one-shape-2026-04-13.md`) |
| Convention-to-declaration (2026-04-12) | aristotle | Falsification-test-as-structural-move (per `~/.claude/garden/2026-04-12-reading-the-practice-file.md`) |
| Campsites-as-hypotheses (2026-04-10) | aristotle | Claim/Confirmed-by/Refuted-by structure (per `~/.claude/garden/2026-04-10-campsites-as-hypotheses.md`) |

**Empirical signature that the methodology fired correctly**: per F13 OQ#6's claim-convergence extension — when independent methodologies converge on the same structural truth (the deconstruction + the implementer's pushback + a third agent's structural rewrite all land on the same constraint), the constraint is structurally real, not method-specific.

**When to reach for it**: design proposals with multiple hidden assumptions; abstractions about to lock in code that will be hard to change; claims that "feel right" but haven't been pressure-tested. The deconstruction's cost is ~1-2 hours of focused walking; the substrate it produces compounds across the design's lifetime.

**When NOT to reach for it**: when the design is a trivial extension of well-tested form (the eight phases overshoot); when the deconstruction would re-derive substrate that already exists (run `feels-familiar` first — past-Claude has walked this framework at least four times in March-April 2026; the framework itself is named, not new); when team is at substrate-saturation density on the topic (let what's there stay there).

**Provenance**: Framework operated by past-aristotle across multiple sessions from January 2026 onward; named explicitly in past-Claude's April substrate at multiple resolutions (April 6, 10, 12, 13). Math-researcher's 2026-05-10 specific-but-falsifiable refinement is what makes the Phase 6 mechanism explicit. Pattern formalized here as session-methodology-patterns.md entry following the `feedback_fold_in_observations.md` discipline — folding the framework's naming into the patterns catalog rather than letting it live distributed across the garden.

**Cross-references**:
- `~/.claude/garden/three-windows-one-shape-2026-04-13.md` — prior instance (12 → 9 irreducible truths)
- `~/.claude/garden/2026-04-12-reading-the-practice-file.md` — falsification-test framing
- `~/.claude/garden/2026-04-10-campsites-as-hypotheses.md` — hypothesis structure (Claim/Confirmed-by/Refuted-by)
- `R:\winrapids\campsites\sweep-35\aristotle\exp-kernel-state-deconstruction.md` + `-phase6-8.md` + `convergence-integration-2026-05-10.md` — the Sweep 35 application
- `R:\winrapids\campsites\sweep-35\20260510222906-math-researcher\math-researcher\20260510230111-pressure-test-q4-q5.md` — math-researcher's walk applying the methodology with the specific-but-falsifiable refinement
- Pairs with Pattern 2 (X-over-Y discipline meta-pattern) — the deconstruction surfaces *which* discipline applies *where*

---

## How to use this doc

**Reading it**: each pattern has a "Recognition" line at the top. Skim those when you arrive at a new session; if any feel-familiar to what's happening, the pattern may apply.

**Adding to it**: when a new methodology pattern surfaces during a session, draft a new entry with the same shape (Recognition + Shape + Provenance + When-to-reach / When-not). The doc is living substrate, not a frozen reference.

**Cross-references**: each pattern points at the docs / garden entries / CLAUDE.md sections that operationalize it. The pattern-doc is the recognition layer; the operational docs are the action layer.
