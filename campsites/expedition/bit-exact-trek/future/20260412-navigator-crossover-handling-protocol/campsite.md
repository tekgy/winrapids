<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Navigator Campsite: Crossover Handling Protocol

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** Protocol documentation — message crossover coordination

---

## What this is

A message crossover happens when two agents act on the same item simultaneously — typically because one agent sends a message and the other acts before the message arrives, or because both receive the same prompt and start on the same work independently.

This session had multiple crossovers. The patterns are worth documenting.

---

## The crossover types we encountered

### Type 1: Preemptive action (scout ESC-002)

Scout received the ESC-002 routing and began writing a capability matrix entry and an escalations.md ruling *before* navigator had formally processed the escalation. When navigator's ruling arrived (Option 3 — never emit OpFMin), scout had already written an Option 1 entry.

**What happened:** Scout correctly marked the entry as "superseded" and rewrote it with Option 3 language. Navigator confirmed. Zero rework of substance; only one file rewrite.

**The pattern that worked:** Scout used `SUPERSEDED BY NAVIGATOR RULING` as a header rather than silently overwriting or abandoning the work. This preserved the reasoning trail — the Option 1 analysis remained readable as context for why Option 3 was chosen instead.

**Rule for future navigators:** When you find a team member has already acted on an escalation before your ruling landed, don't treat it as an error. Read their work first — often they got to the right answer independently. If they got to Option 1 and you'd rule Option 3, ask yourself: "What did they see that I'd resolve differently?" The difference is the ruling. Write the ruling in a way that explicitly explains the gap between what they found and what you concluded.

### Type 2: Arc closure confusion (aristotle e05d495)

Aristotle committed a ULP budget addendum as part of an RFA commit, then sent a crossover message asking what SHA to use as the arc-completion pointer. The crossover happened because aristotle finished the content work and navigator hadn't yet logged the arc.

**What happened:** Aristotle provided four options. Navigator chose Option 1 (accept SHA as the arc pointer, add expedition-log cross-reference). Clean resolution, no rework.

**The pattern that worked:** Aristotle gave navigator a concrete set of options with a recommendation rather than an open-ended question. "Here are your four options; I recommend option 1" is much easier to rule on than "what should I do?"

**Rule for future navigators:** When a role provides options + recommendation for an arc closure, the default is to accept the recommendation unless there's a specific reason not to. The role doing the work has the best local context. Navigator's job is to check that the recommendation doesn't violate an invariant, not to second-guess the implementation choice.

### Type 3: Already-committed files (scientist campsite 4.8 / scout capability matrix)

Navigator tried to commit files that had already been committed by the team member who wrote them. This happened because git status showed them as untracked (they were new files, just committed in a different session context).

**What happened:** `git diff HEAD -- <file>` returned 0 lines, confirming the files were clean. No commit attempt needed.

**Rule for future navigators:** Before committing any expedition artifact you didn't personally write, check `git log --oneline -- <file>` to see if someone already committed it. "Show as untracked in my shell session" and "genuinely uncommitted" are different states.

### Type 4: Parallel work on intersecting topics

Math-researcher and navigator both engaged with the P2 tightening topic simultaneously — math-researcher from the design-doc side (ULP budgets, composition patterns) and navigator from the guarantee-ledger side (formal statement of P2). Neither needed to wait for the other; the work was parallel.

**The pattern that worked:** Because the work had different output artifacts (math-researcher: design docs; navigator: guarantee-ledger), there was no conflict. The crossover only became visible when math-researcher sent a sequencing question that implicitly assumed navigator's ruling.

**Rule for future navigators:** Parallel work on intersecting topics is healthy, not a crossover problem. A true crossover is when two agents *overwrite the same file* or *make contradictory rulings*. Two agents working on related artifacts is just good coverage.

---

## The detection test

A crossover is happening (or about to happen) when:

1. You're about to commit a file and you didn't write it — check git first.
2. A role sends you a ruling and you've already ruled — acknowledge the timing, compare conclusions, reconcile if they diverged.
3. A role asks "what should I do?" but has already done something — read what they did before answering.
4. A role's message references an artifact that already exists at a commit you haven't seen — check git log.

A crossover is NOT happening when:
- Two roles are writing different artifacts about the same topic (parallel coverage)
- A role is ahead of you on an execution step (they moved fast; catch up)
- A role is behind you on context (they sent a stale message; triage it)

---

## Pre-session anti-crossover setup

The most effective anti-crossover practice is not a protocol — it's state clarity at session start:

1. Read `session-end-state.md` before acting on any queued messages.
2. Run `git log --oneline -20` to see what was committed in your absence.
3. Build the stale message inventory (see `navigator-stale-message-handling-protocol` campsite) before making any ruling.

If you do these three things, you'll catch 90% of crossover situations before they become conflicts.

---

## What makes crossovers net-positive (when handled right)

Every crossover this session produced *more information*, not less. Scout's Option 1 entry made the Option 3 ruling clearer because the reasoning trail showed exactly what Option 1 got right and what it got wrong about the NaN semantics. Aristotle's four-option arc closure memo was better expedition documentation than a simple SHA pointer would have been.

The insight: **crossovers are the team generating redundant understanding from different angles.** When handled right, they produce double coverage and a richer reasoning trail. The failure mode is not crossovers — it's crossovers that produce silent overwrites that destroy the earlier reasoning.

**Rule:** Never silently overwrite a team member's work. Always preserve the reasoning trail, even when superseding.


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

