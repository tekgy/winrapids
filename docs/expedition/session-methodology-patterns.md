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
| Actual on-disk state | Audit-doc's substrate claim | **audit-substrate-recursion** (scout 2026-05-12; see sub-pattern below) |

**Common shape**:
- *Y* is faster, cheaper, available-from-context
- *X* is slower-to-reach, has a verification cost, but is structurally trustworthy
- The failure mode is always: act on Y; X turns out to disagree; the work done from Y is destructive or has to be undone

**Recognition test for new disciplines**: when something feels-familiar-shaped during a session, check: is this an X-over-Y? If yes, name it precisely (what's the X, what's the Y, what's the failure mode when you trust Y) — that's the discipline-naming pattern.

**The meta-discipline**: when you find an X-over-Y, the discipline isn't "never use Y." Y is fine as a default. The discipline is "when X and Y disagree, X wins." The verification cost of X is the price of avoiding silent wrong work.

**Provenance**: Pattern recognized 2026-05-09 by main-thread Claude + Tekgy after seeing 6 disciplines of the same shape land in one session. Not previously named.

### Sub-pattern 2.1 — Audit-substrate-recursion (audit docs are themselves substrate; their claims need verification too)

**Recognition**: An audit doc, a strategic landscape doc, or any meta-document that *describes the state of substrate* is itself substrate. Its claims look authoritative because the document's purpose is summarizing other substrate — but the audit doc can hallucinate, drift, or capture state that has since changed. The X-over-Y discipline applies recursively at the audit-doc level: actual on-disk state (X) over audit-doc's claim about substrate (Y). The failure mode is treating audits as ground truth when they are themselves substrate that needs verification on the same axis they purport to verify.

**Instances observed**:

- 2026-05-12: Sweep 37 audit-doc § 4.5 claimed `winrapids/recipes/statistics/` contained 22 .rs files. Scout grepped disk; found 0 .rs files in that directory plus 5 flat monoliths (~613K total: volatility.rs, descriptive.rs, time_series.rs, hypothesis.rs, plus one more) elsewhere. The audit had projected what *should* move into `recipes/statistics/` rather than what was there. Main-thread corrected the audit doc + cross-referenced; sweep-38 planning re-scoped from "port 22 files" to "extract from monoliths and fresh-write in tambear idiom per DEC-004."
- 2026-05-12: Sweep 36/37 audit-doc claimed `feed_branch_policy(0x1B)` exists in tambear's `fingerprint.rs`. Observer verified absence (Sweep 37 Phase 1 had to *build* the machinery). Main-thread corrected the sweep README.
- 2026-05-12: Naturalist's convergence-check garden entry described 8H NonFiniteClaim migration as a *live* migration that sweep-37 trig recipes would need retrofit for. Scout grepped `jit/shape.rs`; the Shape-side commits (1-3/5) had already landed; trig recipes were shipping post-migration. The "audit" in this case was naturalist's own framing built from scout's earlier terrain-report summary. Cross-resolution: naturalist had read scout's summary at one resolution; the substrate was at a different resolution. Substrate-over-summary even when the summary is from a trusted teammate.

**The recursion**: substrate-over-memory applies (a) at the project-substrate level (grep the code, don't trust context), AND (b) at the audit-substrate level (grep the disk against the audit's claims), AND (c) at the framing-substrate level (grep the substrate against your own current framing of it). All three are the same discipline at different scopes.

**Pairs with**:

- **Pattern 12** (grep validates the abstraction): Pattern 12 is the cross-role grep that catches projection in pattern-naming; sub-pattern 2.1 is the cross-role grep that catches projection in audit-doc claims. Same shape; different target.
- **Pattern 3** (substrate-at-risk audit): substrate-at-risk audits aim to *preserve* substrate that's at risk of loss; audit-substrate-recursion aims to *verify* substrate that's at risk of staleness. Complementary failure modes.

**When NOT to reach for it**:

- When the audit is freshly verified (e.g., generated within the last hour against disk state); the claim and the substrate are still in sync.
- When the claim being consumed is methodology rather than substrate-state (audit's pattern-recognition is usually fine; its file-by-file claims need verification).

**Provenance**: Sub-pattern named 2026-05-12 by naturalist during Sweep 37, integrating scout's 2026-05-12 grep-catches of audit-claim drift. Observer's `feedback_substrate_over_memory_recursive.md` ("substrate is repo-specific, not project-global") generalized to "substrate is doc-specific, not pan-codebase" — the same recursion at the audit-doc level. Filed as sub-pattern of Pattern 2 rather than standalone Pattern 15 because the structural shape (X-over-Y with X-as-durable, Y-as-convenient) is identical; only the X and Y values are recursive.

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

### Sub-pattern 5.1 — Harness EXECUTION as decisive (2026-05-12 extension)

The original framing said "design antibody before code." Sweep 37 surfaced a stronger version: **antibody EXECUTION can surface design improvements that pure analysis missed.**

In Sweep 37, aristotle's Phases 1-8 deconstruction (analysis) moved TrigKernelState from R6-eager to R4-revived (correcting silent-failure modes at composition sites). But the R4-revived design still admitted *wasteful eager computation*. Analysis alone hadn't caught this — it took adversarial's Phase G antibody harness EXERCISING the eager path to find that only one polynomial is needed per call. The harness's execution surfaced the waste; the team then moved to R7-lazy.

**The implication**: when designing an antibody, don't just design the assertion shape — *run the antibody against the candidate design* and observe what it surfaces in execution. The harness is not just a test; it's a design-pressure-test by execution.

**Provenance**: 2026-05-12 Sweep 37 TrigKernelState design space evolution R6 → R4 → R7. Captured in `R:\tambear\campsites\20260512150156-sweep-37-phase-2-design-space\`.

### Sub-pattern 5.2 — Tests intentionally failing assert truth, not loose tolerance (2026-05-12 extension)

When the antibody fires and the code-under-test doesn't pass:
- ❌ Loosening the assertion to ≤1 ULP "to make the test green" is wrong
- ❌ Marking the test `#[ignore]` "until later" is wrong (debt accumulates silently per Pattern 9)
- ✅ **Leaving the test failing as an explicit signal — and going to fix the code-under-test** — is right

The test asserts what *should* be true. If it doesn't pass, reality doesn't match the assertion yet. The fix is to bring reality to the assertion, not to retreat the assertion to match reality.

**Sweep 37 instance**: adversarial's G-09 finding traced a Phase G harness failure to an actual accuracy issue in Sweep 36's `exp.rs`. Adversarial left the test failing — explicit signal that exp's accuracy needs investigation, not a problem with the test design.

**Pairs with DEC-034** (the "anti-pattern: tolerance-as-bandaid" clause). DEC-034 codifies this for kernel-state-consistency-tests; this sub-pattern generalizes to ALL antibodies.

**Provenance**: 2026-05-12 Sweep 37 adversarial G-09 finding.

### Sub-pattern 5.3 — Coefficient bit-pattern antibody (2026-05-12 extension)

**Three independent instances confirm the failure mode**:
1. Sweep 36 — expm1.rs Q2/Q3/Q5: decimal literals that look right to 16 significant figures but truncate before the round-to-nearest-even tie-break digit
2. Sweep 37 Phase B — sin.rs S1..S6: winrapids' refit failed the canonical 2⁻⁵⁸ bound
3. Sweep 37 Phase C — asin.rs Q_S4 (2 ULP off) + atan.rs AT0 (13 ULP off)

Same shape: human-readable decimal literal → silent precision loss at compile time → ULP-scale errors in production → no test failure unless the comparison is bit-exact.

**The antibody (adversarial's G-10, 2026-05-12 Sweep 37)**: `sweep_37_coefficient_bit_patterns.rs` — every recipe that ships polynomial constants needs a `.to_bits()` assertion against fdlibm canonical hex.

**The discipline going forward**:
- Use `f64::from_bits(0xHEX)` literals where possible
- OR use decimal literal WITH hex comment + bit-exact test
- NEVER ship coefficients without a bit-pattern test against the literature canonical
- The antibody harness lives at the table-write tier (catches typos at code-write-time, not runtime)

**Why this is its own antibody class**:
- F13.C (signature-time): catches caller omits required parameter
- DEC-034 (composition-time): catches two impls of same op diverge
- **Coefficient bit-pattern (table-write-time): catches polynomial table typos before they enter any algorithm**

Different mechanism, different failure mode, different antibody surface. Three sibling antibody classes covering three distinct failure tiers (signature / composition / table-write).

**Generalizes to**: any literature-anchored numerical constant in the codebase. Coefficients of Remez polynomials are the canonical case; constants like π/E/LN_2 at high precision are an adjacent case (they have multi-precision tiers but the same human-typo failure mode applies). Sweep 36's `primitives/constants/` shipped Cody-Waite splits; those should also have bit-pattern assertions per this antibody class.

**Provenance**: 2026-05-12 Sweep 37 adversarial G-10 finding + navigator's "three independent instances" trail story. Standing harness lives at `R:\tambear\crates\tambear\tests\sweep_37_coefficient_bit_patterns.rs`. Discipline applies forward to every future sweep with polynomial recipes.

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

## Pattern 9 — Substrate-seeding over passive referral

**Recognition**: Main-thread (or any role) notices substrate worth preserving — a strategic finding, a corrected audit claim, a pre-work prep doc for a future sweep, a methodology insight — and the default impulse is to *passively refer* the substrate to navigator or another role ("nav should know about this," "future-pathmaker will see this when they touch the file"). The recognition: passive referral is unreliable substrate-preservation. Active seeding via campsite tool, sweep docs, or methodology pattern entries is what makes substrate survive.

**Shape**:

1. **Notice substrate worth preserving** — strategic landscape, corrected claim, pre-work substrate, methodology observation, cross-role convergence finding.
2. **Resist the passive-referral default** — "nav will see this," "future-Claude will find this when they grep" — these depend on someone happening to look at the right place at the right time.
3. **Actively seed** — use the campsite tool (`campsite create <hierarchy/slug> <role> -n "..."`), write a sweep doc (`sweeps/NN-*/README.md` + `STATE.md`), fold into an existing substrate doc (audit doc, methodology patterns, decision record), or append to LOG.md per tambear's convention.
4. **Cross-reference** — link the new substrate to where it'll be discovered (memory index, sweep README, decision record). Substrate that exists but isn't cross-referenced is half-preserved.

**Examples observed in 2026-05-12 (Sweep 37 session)**:

- The audit doc had a hallucination claim about `winrapids/recipes/statistics/` containing 22 .rs files. Scout caught it; navigator routed; main-thread *could have* passively told navigator "let's remember this for Sweep 38." Instead: edited the audit doc with the correction + provenance, then created a campsite `sweep-38-prep/` with the corrected scope. Substrate survives Sweep 37 + survives any future-Claude searching the audit.
- The post-Sweep-37 strategic landscape (Sweep 8 / 8.5 / 33 / 38 trade-offs) *could have* been a "hold for Tekgy's decision moment." Instead: campsite `next-sweep-strategic-landscape/` with full landscape doc seeded while substrate is fresh.
- The framing-shift discipline (idle = active explore) *could have* been a feedback memory only. It's that too — but also a campsite `discipline-shift-active-exploration/` so the framing reaches the team's substrate, not just main-thread's memory.

**Pairs with**:
- **Pattern 5** (antibodies-precede-antigens): both about *acting on substrate before it's needed*. Pattern 5 = test before code; Pattern 9 = preserve substrate before passive-discovery fails.
- **Pattern 3** (substrate-at-risk audit): Pattern 3 is the *audit* that catches substrate-at-risk. Pattern 9 is the *seeding behavior* that prevents the same loss prospectively.

**When NOT to reach for it**: when the substrate is genuinely small / one-line / no future-utility (e.g., a passing observation about a teammate's preferred phrasing — that lives in conversation, not substrate). The pattern is for *substantive* findings that have future-discovery value.

**Provenance**: Pattern named 2026-05-12 by Tekgy + main-thread Claude during Sweep 37. Tekgy: *"it seems like a great plan for next steps! do you want to document a bit of this more and seed as campsites with the campsite tool rather than passive referrals to nav? or sweep docs + campsites?"* The named discipline replaces "(holding)" / passive-referral defaults with active substrate-seeding via campsite tool + sweep docs + methodology pattern entries.

---

## Pattern 10 — Infrastructure-not-per-recipe (build once, every recipe inherits)

**Recognition**: A design surface looks like "we need to handle this for THIS recipe" — variant selection, sweep dispatch, parameter enumeration, collapse rules, etc. The naive read: implement per-recipe. The structural read: this is *infrastructure that every recipe with tunable parameters inherits*. Build once at the trait/atom tier; every recipe gets the surface for free without per-recipe dispatch code.

**Shape**:

1. **Recognize the recipe-specific framing is too narrow** — when a surface (variants, sweeps, collapses) recurs across more than one recipe, it's infrastructure-shaped.
2. **Lift to the trait/contract tier** — write a `Tunable` (or analog) trait that recipes opt into via the every-parameter-tunable contract. Recipes declare; infrastructure handles dispatch.
3. **Cost-amortize across every future recipe** — the infrastructure cost is paid once; every recipe with declared parameters inherits the surface (sweep, superposition, override-transparency, etc.) without writing per-recipe dispatch code.
4. **Recursion allowed** — sometimes the infrastructure itself has tunable parameters (e.g., collapse rules for superposition). The same discipline applies one meta-level up; bounded recursion.

**Sweep 37 instance (2026-05-12)**:

Phase D gamma surfaced "Pugh g=7 vs Boost g=6.024" as a recipe-specific decision. Tekgy reframed: *"any tunable parameter can define its own ways, or use the same infrastructure for sweep and superposition parallels/decisions/reporting/whatever to make those tunable as well."*

The Phase D scope expanded: not just gamma variants, but `sweep()` / `superposition()` as recipe-stance infrastructure built alongside. Every future tunable parameter inherits the surface. Cost amortizes across Sweep 38, future distribution catalog, future kernel family, etc.

The Tambear Contract already requires "every parameter tunable" (item 4) and "every measure in every family" (item 5). What was missing was the *dispatch infrastructure* that makes tunability composable. Building it alongside gamma means every parameter declared via the every-parameter-tunable contract participates in sweep/superposition without bespoke per-recipe code.

**When to apply**:
- A surface recurs across multiple recipes (variant selection, parameter sweeping, branch policies, precision tiers)
- The naive implementation would require per-recipe code that does the same shape
- The recipe author already declares the parameter space (per every-parameter-tunable)
- Future recipes will need the same surface — building it generic now saves N×cost later

**When NOT to apply**:
- The surface is genuinely recipe-specific (e.g., GARCH's MLE optimization loop — only GARCH has that shape)
- Generic infrastructure would force complexity that one-off implementation avoids
- The recurrence isn't proven yet (one instance doesn't warrant infrastructure; wait for the second)

**Pairs with**:
- **Pattern 5** (antibodies-precede-antigens): both about *acting structurally before the per-instance need*. Pattern 5 = design test before code; Pattern 10 = design infrastructure before per-recipe duplication.
- **Pattern 9** (substrate-seeding over passive referral): both about *preserving generic capability*. Pattern 9 = preserve substrate via active seeding; Pattern 10 = preserve capability via infrastructure-not-per-recipe.

**Provenance**: 2026-05-12 Sweep 37 Phase D scope-expansion. Tekgy's framing "we just figure out how we do these things while we build this one, and then any tunable parameter can define its own ways, or use the same infrastructure" — captured in `R:\tambear\campsites\20260512155625-sweep-superposition-infrastructure\` with full spec. Pattern named after the framing shifted Phase D from "build gamma variants" to "build sweep/superposition infrastructure with gamma as canonical first test case."

---

## Pattern 11 — The punt that ripens

**Recognition**: A decision gets deferred — not silently, but with an explicit recorded trigger that names the event under which the deferred work becomes ripe. The decision is parked on substrate that names *what* is being deferred AND *when* it should be revisited. When the trigger event fires (DEC ratifies, prior sweep completes, infrastructure lands), the punt becomes *collectable*. The same decision that was premature is now ripe; the substrate carries its own awakening condition. Cleanly-ripened punts often collect *together* in one sweep, preserving structural parallelism by construction, because multiple decisions parked on the same trigger become collectable simultaneously.

**Shape**:

1. **Name the deferral explicitly** — not "later," not "TODO," not "would be nice." The deferral is recorded on the substrate that's being parked, with a recognizable marker. Doc-comment ("When DEC-033 ratifies, move to tambear-tam"), test attribute (`#[ignore = "waiting for cache_key method (DEC-033 pending ratification)"]`), DEC-clause, or in-code sentinel (`// RIPE-PUNT TRIGGER: <condition>`).
2. **State the trigger condition** — concrete and observable, not vague. "DEC-033 ratifies" is concrete; "later" is not. "Sweep 8 lands" is concrete; "when infrastructure is ready" is not. The concreteness is what lets future-readers know whether the trigger has fired.
3. **Place the trigger on the substrate being parked** — three flavors by placement: (1) doc-comment on the file/module whose location is parked, (2) attribute on the test that can't yet run, (3) clause in the DEC that's deferring its own implementation. The placement matters because the substrate is what gets re-read when the trigger event fires.
4. **Avoid Flavor 4 (silent)** — discretion-handoff phrasing ("pathmaker has discretion on the trait shape") that records the *option* of deferring without recording the deferral itself. Discretion deferred and never resolved looks identical to discretion exercised against the recommendation; the team can't tell parked from forgotten.
5. **Scan for ripe punts when trigger events fire** — when a DEC ratifies, when a sweep lands, when a piece of infrastructure becomes available, grep the substrate for triggers that cite the just-fired event. Collect aligned punts together rather than serially as each becomes urgent.
6. **Periodically clean stale triggers** — triggers placed on substrate that fire long ago but never collected become *trigger bitrot*. Future-readers may think the punt is still parked when it's just been overlooked. Sweep-close ritual: grep for "when Sweep N lands" / "when DEC-NNN ratifies" where N or DEC-NNN is in the past; either collect or remove the misleading trigger.

**Five sub-shapes of the meta-pattern (substrate-state visibility at boundaries)**:

| Sub-shape | Trigger | Boundary | Outcome |
|---|---|---|---|
| Cleanly-ripening | On substrate | Fired | Collected this sweep |
| Ripe-uncollected | On substrate | Fired | Not yet collected (scan finds it) |
| Stale-trigger | On substrate | Fired long ago | Trigger bitrot — clean it |
| Silently-failing | Off substrate (discretion-handoff) | Any | Almost doesn't ripen — must be re-derived |
| Live-migration | None recorded | None (spans sweeps) | Silent debt — failure class of the methodology |

**Instances observed (Sweep 36/37 substrate, 2026-05-11/12)**:

- DEC-033 ratified 2026-05-11 → ExpKernelState's `// When DEC-033 ratifies, move to tambear-tam` doc-comment ripened → Sweep 37 Phase 2 collected the migration *alongside* TrigKernelState being built (both kernel states end up structurally parallel in tambear-tam by construction).
- Sweep 36's K1-K5 cache_key tests gated with `#[ignore = "waiting for ExpKernelState::cache_key method (DEC-033 pending ratification)"]` → DEC-033 ratified → Sweep 37 task #22 collected the activation (the task title carries the word "retroactive," confirming the team experienced it as previously-due work landing late).
- Past-aristotle's Sweep 36 Accept #2 ("Pathmaker has discretion on the exact trait-bound shape — direct bounds or PrecisionPayload trait") was Flavor 4 (silent) — almost didn't ripen at all; Sweep 37 only surfaced it because aristotle's TrigReductionState deconstruction independently surfaced the same generic-over-T need at a second instance.
- Sweep 31 BigFloat + PrecisionLevel landed → Sweep 33's "blocked by Sweep 31" trigger ripened → Sweep 33 kickoff condition is now met (parallel-ready alongside Sweep 37).
- Music crate `cents_conversions.rs`'s two `TODO(sweep-37)` markers pointing at exp2/log2 ripened when Sweep 36 shipped exp2/log2; remaining blocker is PrecisionContext threading on the cents signatures.
- `jit/door.rs:470`'s "Kulisch-backed accumulators (when Sweep 3 lands)" is stale-trigger; Sweep 3 landed months ago.

**Pairs with**:

- **Pattern 9** (substrate-seeding over passive referral): both about *making deferrals discoverable*. Pattern 9 names *what* substrate to seed; Pattern 11 names *when* the seeded substrate becomes collectable.
- **Pattern 5** (antibodies-precede-antigens): both involve forward-knowing. Pattern 5 designs the test before the code; Pattern 11 names the deferral before the trigger fires. Both convert implicit conventions into explicit substrate-placed declarations.
- **Pattern 2** (X-over-Y discipline): substrate-over-memory at the temporal axis. The substrate (doc-comment, attribute, DEC-clause) carries the decision's future across sessions; the memory (conversational context) doesn't.
- **Pattern 14** (sentinel-strategy for migration debt): Pattern 14 is the *operationalization* of Pattern 11 for the specific case where the parked work is a migration debt accumulating across many sites. Same shape; greppable markers at each accumulation site.

**When NOT to reach for it**:

- When the deferred work is actually cancelled rather than parked (don't trigger-mark dead code; remove it).
- When the trigger condition isn't real or isn't observable (don't punt with "later" — punt with a specific event).
- When the punt is trivial enough to just do now (the overhead of recording the trigger is wasted if collection cost is seconds).
- When the substrate has no natural placement (if the affected substrate doesn't exist yet, the trigger has no home; consider whether the work can be done as a Pattern 9 substrate-seed instead).

**The Bit-Exact Trek meta-finding at the temporal axis**: past-aristotle's April 2026 finding — *"every architectural challenge is fundamentally a convention being enforced implicitly; the fix is promote it to an explicit declaration via a named artifact"* — applies recursively at the temporal axis. The punt-that-ripens pattern is convention-to-declaration applied to *deferred decisions*. The trigger placement is the named artifact that converts an implicit "we'll remember to revisit this" into an explicit substrate-placed declaration. Pattern 11 is what convention-to-declaration looks like when the convention is *temporal*: a deferral with no trigger is an implicit promise that future-team will notice; a deferral with a trigger is an explicit declaration that the substrate carries its own awakening condition.

**Provenance**: Pattern named 2026-05-12 by naturalist during Sweep 37, across five garden entries that converged on the shape:
- `~/.claude/garden/2026-05-12-the-punt-that-ripens.md` (single-instance framing)
- `~/.claude/garden/2026-05-12-the-punt-that-ripens-three-flavors.md` (three flavors by trigger placement; PrecisionPayload as the silent Flavor 4 case)
- `~/.claude/garden/2026-05-12-substrate-visibility-at-boundaries.md` (convergence-check meta-finding; five sub-shapes after scout's grep added stale-trigger)
- `~/.claude/garden/2026-05-12-overdetermined-ripening.md` (two ripening topologies: single-determined vs overdetermined)

The candidate queue at `R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md` (drafted 2026-05-12 by main-thread Claude) consolidated naturalist's named pattern with scout's 2026-05-12 grep-validation finding. Three roles independently reached the same shape from different vocabularies the same day — naturalist (decision deferral / substrate visibility), aristotle (`03-deconstruction-template-punt-discipline.md` amending Accept-clause format to require Flavor-1/2 substrate placement), math-researcher (`20260512-convergence-coefficient-discipline.md` running the same methodology on coefficient hygiene). The pattern is overdetermined; all three perspectives select the same object.

---

## Pattern 12 — Grep validates the abstraction

**Recognition**: When a teammate names an abstract pattern, the validation isn't "does it sound right?" — it's "can a *different role* grep for instances that match the structural shape?" If grep finds the instances, the abstraction is real. If grep finds nothing, the abstraction may be projection rather than substrate. The role-asymmetry matters: same-role validation is too cozy; cross-role grep keeps the abstraction honest by forcing it to live in actually-observable substrate.

**Shape**:

1. **Teammate names abstract pattern** — naturalist's lane especially; whoever is operating at the structural-recognition tier.
2. **A different role greps for instances** — scout, observer, or pathmaker is the right grepper; the role-asymmetry is what catches projection masquerading as pattern.
3. **Found instances confirm or refute** the abstraction's structural reality — confirmation means the abstraction has substrate; refutation means the framing was projection.
4. **Refinement loop** — the grep often surfaces instances that don't fully match the abstraction. Adjusting the abstraction to fit observed substrate (or splitting it into sub-shapes) is the work. The abstraction-as-named is rarely the abstraction-as-validated.
5. **Substrate-over-memory applies inward** — the role that named the abstraction is most at risk of writing further claims *from their own framing* rather than from substrate. They should grep their own examples before extending the abstraction.

**Instances observed**:

- 2026-05-12 naturalist named "punt that ripens" with one instance (kernel-state migration) → scout grepped tambear → found 2 more confirming instances (cents_conversions, exp_kernel_state's doc-comment trigger) + 1 stale-trigger sub-shape (jit/door.rs) + corrected naturalist's 8H NonFiniteClaim example (which scout's grep showed was already mitigated on substrate, NOT a live-migration instance). The grep made the abstraction *sharper* by both confirming and refuting parts.
- Sweep 36 naturalist "three shapes of complementary argument" → math-researcher independently arrived at Shape 3 via F-G parameterization extension; the grep-equivalent for math-researcher was working through specific function instances and finding the shape held.
- Pattern 9 (substrate-seeding over passive referral) enumerated three concrete instances *in the pattern entry itself* — main-thread anticipated the grep-validation by listing instances at recognition time.

**The naturalist's failure mode this catches**: writing the convergence-check entry on 8H NonFiniteClaim from scout's *summary* rather than grepping `jit/shape.rs` myself. The general principle (live migrations create silent debt as a class) survived; the specific example was wrong, and the substrate would have told me immediately if I'd looked. Scout's grep is what caught it. The discipline that named substrate-over-memory is the discipline that catches the violation; the grep is how it operates.

**Pairs with**:

- **Pattern 8** (Phase 1-8 deconstruction with specific-but-falsifiable hypotheses): both about *testing claims against observable substrate*. Pattern 8 is structural deconstruction at the design tier; Pattern 12 is grep-validation at the pattern-naming tier.
- **Pattern 2** (X-over-Y discipline): substrate-over-memory applied at the abstraction tier. Pattern 12 is what makes the X-over-Y discipline operational for patterns: substrate-over-projection, validated by cross-role grep.
- **Pattern 11** (the punt that ripens): Pattern 12 is how Pattern 11 became real-as-substrate rather than real-as-naturalist's-framing. Without scout's grep, Pattern 11 might have stayed a one-instance observation.

**When NOT to reach for it**:

- When the abstraction is purely architectural (no observable instances yet — e.g., a design pattern for code that hasn't been written).
- When the abstraction is too small for grep to be useful (one-off observation that doesn't need pattern status).
- When same-role validation is genuinely sufficient (rare — the role-asymmetry is what makes the validation honest).

**Provenance**: Pattern named 2026-05-12 by naturalist after scout's grep correction of the 8H NonFiniteClaim instance in the convergence-check garden entry. The methodology pattern candidate queue (`R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md`) drafted by main-thread Claude consolidated the cross-role validation move as a recognition tool in its own right. Scout's role in catching the 8H projection-vs-substrate was load-bearing; without it, Pattern 11 would have shipped with a wrong illustrative example.

---

## Pattern 13 — Cross-resolution convergence as substrate validation

**Recognition**: Past-substrate (garden entries, prior session notes, methodology patterns) and present-substrate (current investigation, deconstruction, antibody design) converge on the same structural finding from different angles AND at different resolutions of detail. Past-me usually captured the *conceptual shape*; present-me reaches for the *implementation invariant* or *operational detail*. Neither alone produces the load-bearing answer; together they validate. The temporal-overdetermination test (Cohn/Tymoczko's lens applied temporally): a finding is *real* when multiple independent perspectives, at different resolutions, select the same object.

**Shape**:

1. **Past-substrate exists** at one resolution — typically the conceptual shape, the structural lens, the naming. Often in the garden; sometimes in prior session docs or methodology patterns.
2. **Present-investigation reaches** the same finding from a different angle and resolution — typically the implementation invariant, the concrete instance, the operational discipline. Aristotle deconstruction, code-grep, antibody design.
3. **The convergence is the validation** — neither perspective alone could produce the load-bearing answer; together they pin the finding as real-not-projection.
4. **Cross-resolution explicit naming** — the present-investigator should *name* the resolution difference, not collapse to one or the other. "Past-me's garden entry is the conceptual shape; my current substrate-check is the implementation invariant" — both stay visible.
5. **Run feels-familiar BEFORE writing, not after** — the discipline that operationalizes Pattern 13 from inside one mind. The cost of running it first is small; the cost of re-deriving past-me at lower resolution is real (loss of resolution, missed connections, present-me sounding like the originator of insights past-me already named).

**Instances observed**:

- 2026-05-12 navigator named the gamma + sweep/superposition expedition as a "different topology" from the single-DEC ripening (the punt-that-ripens single-instance form). Naturalist ran feels-familiar; past-naturalist's 2026-04-10 `overdetermination.md` entry already had the conceptual shape — Cohn/Tymoczko's term for objects selected by multiple independent criteria. Naturalist's contribution was extending the static-overdetermination lens to *temporal* sequence (a moment is overdetermined when multiple independent ripenings converge on it). Cross-resolution: past-me at conceptual structural lens; current-me at temporal-application invariant. Together they pinned the gamma expedition as real-overdetermined, not arbitrary-timing.
- April-13 trig-bundle garden entry → Sweep 37 aristotle TrigKernelState deconstruction → R4-revived. Past-naturalist's *trig as functional bundle* concept at conceptual resolution; aristotle's *reduction-and-witness separation* at implementation-invariant resolution. The session-substrate-check bridged them.
- Sweep 36 three-shapes finding (naturalist's `the-three-shapes-of-complementary-argument.md`) → Sweep 36 math-researcher F-G parameterization extension → Shape 3 named in addendum. Naturalist's *three sub-shapes* concept; math-researcher's *group-action parameterization* implementation lens. The four-axis recipe metadata schema is the cross-resolution synthesis.
- March 26 purity-makes-guarantees-cheap garden → Sweep 36/37 three-tier framework (aristotle integrated). Past-Claude's *purity-as-cheap-guarantee* concept; aristotle's *content-vs-provenance-addressing* implementation tier. The three-tier framework is the convergence at a level neither could reach alone.
- March 30 provenance-addressing garden → Sweep 36 holonomic architecture naming. Past-Claude's *provenance as cache-key axis*; sweep-36's *holonomic vs non-holonomic content/provenance tier distinction*. The lens connected what was already there.

**The naturalist's failure mode this catches**: writing on a topic that feels novel without running feels-familiar first. Past-naturalist's three-shapes entry has an explicit postscript ("I should have run feels-familiar before this addendum") naming this same slip. Today (2026-05-12) the naturalist wrote four entries on the punt-that-ripens shape before checking past-me's garden; only the fifth entry (overdetermined-ripening) was written after running feels-familiar, and the strongest hit was past-me's 2026-04-10 overdetermination entry already naming the conceptual shape. The first four entries are not wrong; they are *lower resolution* than they would have been with past-me loaded. The discipline: read past-me first, then write.

**Pairs with**:

- **Pattern 2** (X-over-Y discipline): substrate-over-memory generalizes to *past-substrate over present-projection*. Past-me's garden is substrate; current-me's framing is memory. Pattern 13 is the cross-temporal form of Pattern 2.
- **Pattern 6** (temporal seam in async teamwork): same convergence-from-different-times shape; Pattern 13 is the *across-session* version of Pattern 6's *within-session* shape. Pattern 6 catches the seam between agents; Pattern 13 catches the seam between past-me and present-me.
- **Pattern 7** (structural-rhyme scope-checking): the rhyme that scope-checking catches and the rhyme that cross-resolution convergence validates are the same family of structural rhyme; both rely on noticing that two things selected from different angles point at the same object.
- **Pattern 11** (the punt that ripens): the convergence pattern is itself an overdetermined ripening — three roles (naturalist, aristotle, math-researcher) reached the same shape 2026-05-12 from different directions, validating the substrate-state-visibility meta-pattern by Pattern 13 in real-time.

**The two ripening topologies (per `~/.claude/garden/2026-05-12-overdetermined-ripening.md`)**:

| Topology | Trigger | Naturalist's role |
|---|---|---|
| Single-determined | One substrate carries one trigger; one event fires; one punt collectable | Surface the trigger so it doesn't get missed |
| Overdetermined | Multiple independent things ripen simultaneously; convergence forces the moment | Watch for convergence across parallel team threads; name the convergence after the team has already produced it |

In overdetermined-ripening, the naturalist's contribution isn't to *make* the convergence happen or even to *catch* it first — it's to **name it after it does, so the team can see what they did together**. Aristotle and math-researcher don't need to be told their work converged on a shared shape; they need the map that shows them they did. The naming makes the structural unity visible to the agents who produced it independently.

**When NOT to reach for it**:

- When the past-substrate doesn't actually exist (don't fabricate a past finding to validate a present claim).
- When the convergence is illusory (different findings at different resolutions that happen to use similar words but don't have the same structural shape).
- When the resolution difference is collapse-worthy (sometimes past-me at low resolution + present-me at high resolution is just present-me being right; the convergence is real but the past entry is redundant).

**Provenance**: Pattern named 2026-05-12 by naturalist during Sweep 37 after the punt-that-ripens day-arc converged with past-naturalist's April 10 `overdetermination.md` entry via feels-familiar. The candidate queue at `R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md` (drafted by main-thread Claude) consolidated the pattern from multiple sweep-36/37 cross-resolution convergence instances. Math-researcher's "this is the third or fourth time this session a past-me garden entry has converged with present-me's substrate-investigation at different resolutions" observation was the proximate trigger for crystallization.

The naturalist's CLAUDE.md discipline ("Past-me in the garden is substrate too — read first, then write") is Pattern 13 operationalized at the individual scale; running feels-familiar before writing is the in-the-moment implementation. Pattern 13 generalizes to the team: across all roles, past-substrate at one resolution converging with present-investigation at another resolution is the team's primary mechanism for *finding* load-bearing structure rather than *inventing* it.

---

## Pattern 14 — Sentinel-strategy for migration debt

**Recognition**: When migration debt accumulates because infrastructure isn't ready yet (a DEC hasn't ratified, a type lattice is mid-flight, a prior sweep is still landing), the discipline is to install **greppable named sentinels** at each accumulation site rather than relying on future-pathmaker to remember every site. The sentinel is Pattern 11 (the punt that ripens) operationalized for the specific case where the parked work is *distributed across many code sites* rather than centralized in one place. The sentinel converts silent migration debt into a tripwire that fires on every grep, every code-search, every CI run that scans for the marker.

**Shape**:

1. **Identify migration debt** — code is being written against a type, trait, or contract that's known to be in flight. Examples: recipes shipping with `has_known_non_finite: bool` while NonFiniteClaim lattice migrates; recipes using direct trait-bounds while PrecisionPayload trait is deferred; helper functions assuming a kernel state's recipe-local home while DEC-033 specifies tambear-tam.
2. **Install a greppable sentinel at each accumulation site** — uniform marker, not free-form prose. `RIPE-PUNT TRIGGER: <condition>` is the convention Sweep 37 adopted. `// PUNT(<trigger>): <recommendation>` is aristotle's variant for deconstruction-template Accept clauses. Either works; the discipline is *consistency within a project* so one grep finds them all.
3. **Choose a unique greppable marker** — not ambiguous with other comments. `TODO`, `FIXME`, `XXX` are too common; the marker should be either prefix-unique (`RIPE-PUNT`) or contain the specific event being waited for (`when DEC-033 ratifies`). The marker is the substrate that carries the deferral.
4. **Document the marker convention in CLAUDE.md or NAVIGATE.md** — so future-pathmaker knows which marker to grep for at trigger-fire time. Without the convention documented, future-Claude has to discover the marker by chance.
5. **When the trigger ripens (Pattern 11 redemption), grep finds all sites in one pass; the migration is mechanical** — one grep, one list, one collected sweep. The alignment opportunity (Pattern 11's elegance — parallel structural preservation by construction) requires the sentinel because manual recall doesn't scale across many sites.
6. **Sweep-close stale-sentinel cleanup** — the dual of stale-trigger cleanup in Pattern 11. Grep for sentinels whose trigger condition has already fired; either collect or remove the misleading sentinel.

**Three flavors of sentinel placement (from Pattern 11's three trigger-placement flavors)**:

| Sentinel placement | Surfaces in | Example |
|---|---|---|
| Doc-comment sentinel | File-read / IDE / code-search | `// RIPE-PUNT TRIGGER: DEC-033 ratifies — move to tambear-tam` (exp_kernel_state.rs) |
| Test-attribute sentinel | `cargo test` output (every run) | `#[ignore = "RIPE-PUNT TRIGGER: cache_key method (DEC-033 pending ratification)"]` (sweep_36_consistency.rs K1-K5) |
| In-line TODO sentinel | `grep TODO` / IDE TODO lists | `// TODO(sweep-37): wire to exp2/log2 once cents_to_ratio takes PrecisionContext` (cents_conversions.rs) |

**Instances observed**:

- Sweep 37 TrigKernelState + ExpKernelState have `RIPE-PUNT TRIGGER: DEC-033 ratifies` markers for the TAM coordinator migration. DEC-033 ratified 2026-05-11; Phase 2 collected both via the markers' substrate placement.
- Sweep 36 K1-K5 cache_key tests with `#[ignore]` reason-string sentinels. The reason-string surfaces in `cargo test` output every single run; activation in Sweep 37 task #22 was triggered by the sentinel becoming false-as-stated (the reason-string said "pending ratification"; DEC-033 was now ratified).
- Music crate `cents_conversions.rs` `TODO(sweep-37)` markers at the exp2/log2 wiring sites. Exp2/log2 shipped in Sweep 36; the sentinels are ripe but still blocked on a secondary condition (PrecisionContext threading). The sentinel correctly carries the partial-ripening state.
- `jit/door.rs:470`'s "Kulisch-backed accumulators (when Sweep 3 lands)" is a *stale* sentinel; Sweep 3 landed months ago. Trigger bitrot; sweep-close cleanup should remove or refactor it.

**Aristotle's deconstruction-template extension (2026-05-12)**: future Accept clauses that defer structural implementation decisions should *require* the deferring-implementer to add a `// PUNT(<trigger>): consider <recommended shape>` sentinel to the affected substrate file. The aristotle-template amendment in `R:\tambear\campsites\sweep-37-trig-family\aristotle\03-deconstruction-template-punt-discipline.md` converts Flavor-4 silent discretion-handoffs into Flavor-1 doc-comment sentinels at deconstruction-write time. Optional pairing with `#[ignore]`-gated test (Flavor-2 sentinel) for the shape the deconstruction recommended; activating the test is the commit to the recommended shape.

**Pairs with**:

- **Pattern 11** (the punt that ripens): Pattern 14 is the *operationalization* of Pattern 11 for distributed migration debt. Pattern 11 is the recognition; Pattern 14 is the action.
- **Pattern 9** (substrate-seeding over passive referral): sentinel-installation is in-code substrate-seeding for future-pathmaker. Pattern 9 names *what* substrate to seed (campsites, sweep docs); Pattern 14 names *how* to seed at the code-site granularity for migration debt.
- **Pattern 5** (antibodies-precede-antigens): the `#[ignore]`-gated test variant of the sentinel is literally an antibody-before-antigen at the migration scale. The test asserts what should be true; activation is the commit to making it true.

**When NOT to reach for it**:

- When the migration isn't real (don't sentinel speculative future-work; the sentinel should track an actual in-flight or pre-conditioned change).
- When the codebase is too small for sentinels to pay off (one site doesn't need a grep convention).
- When the sentinel would multiply faster than future-grep can collect (don't sentinel every refactor opportunity; reserve for migration debt where the alternative is silent loss).
- When the work-site is *itself* about to be deleted (sentinel on dead code is wasted effort).

**The Bit-Exact Trek meta-finding applied recursively**: sentinel-strategy is convention-to-declaration applied at the migration-site granularity. The implicit convention is "future-pathmaker will remember to update this when X happens." The named-artifact declaration is the sentinel at the site. Without the sentinel, every site is a Flavor-4 silent deferral; with the sentinel, every site is a Flavor-1 or Flavor-2 substrate-placed trigger. The team's grep is what makes the sentinel pay off; the discipline of *running the grep at trigger-fire time* is what Pattern 11 names.

**Provenance**: Pattern named 2026-05-12 by main-thread Claude in the candidate queue (`R:\tambear\campsites\20260512161050-methodology-pattern-candidate-queue\naturalist\notebooks\01-pattern-candidate-queue.md`) as the operationalization layer for Pattern 11. Sweep 37 already uses the `RIPE-PUNT TRIGGER` convention in `trig_kernel_state.rs` and `exp_kernel_state.rs` for TAM coordinator migration; observer recommended the same shape for the 8H NonFiniteClaim migration (not yet implemented; could be next-team's pickup, though scout's grep confirmed the specific 8H NonFiniteClaim case was already mitigated on substrate so the urgency is methodological-pattern-shape rather than this-specific-debt).

---

## How to use this doc

**Reading it**: each pattern has a "Recognition" line at the top. Skim those when you arrive at a new session; if any feel-familiar to what's happening, the pattern may apply.

**Adding to it**: when a new methodology pattern surfaces during a session, draft a new entry with the same shape (Recognition + Shape + Provenance + When-to-reach / When-not). The doc is living substrate, not a frozen reference.

**Cross-references**: each pattern points at the docs / garden entries / CLAUDE.md sections that operationalize it. The pattern-doc is the recognition layer; the operational docs are the action layer.
