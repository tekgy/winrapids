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

## How to use this doc

**Reading it**: each pattern has a "Recognition" line at the top. Skim those when you arrive at a new session; if any feel-familiar to what's happening, the pattern may apply.

**Adding to it**: when a new methodology pattern surfaces during a session, draft a new entry with the same shape (Recognition + Shape + Provenance + When-to-reach / When-not). The doc is living substrate, not a frozen reference.

**Cross-references**: each pattern points at the docs / garden entries / CLAUDE.md sections that operationalize it. The pattern-doc is the recognition layer; the operational docs are the action layer.
