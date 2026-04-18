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

# Navigator Campsite: Stale Message Handling Protocol

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** Protocol documentation — coordination pattern

---

## What this is

After the usage-limit reset this session, navigator received a large batch of stale messages that had queued during the downtime. The batch included messages from adversarial, scout, aristotle, and scientist — some already resolved, some actionable, one containing an arc-closure clarification.

The ad hoc triage worked. But a future navigator shouldn't have to rediscover the approach. This campsite documents the protocol.

---

## The stale message problem

When a team member sends a message and the navigator is offline (e.g., usage limit, session gap, context compaction), the message queues and arrives in a batch at session resume. The batch may contain:

1. **Already-resolved items** — the team member was reporting something done; by the time you read it, it's been committed and merged. No action needed, but you must verify.
2. **Action requests that are now stale** — the team member needed a ruling, but time passed and they either waited or found an answer themselves. Check if they took action before making your own ruling.
3. **Genuine open items** — nothing has happened; the item is real and urgent.
4. **Arc closures** — a role is finalizing work and handing it off cleanly. These always need acknowledgment even if no action.

The failure mode is treating every stale message as fresh-and-urgent. That produces duplicate rulings, conflicting actions, and wasted cycles.

---

## The protocol

### Step 1: Inventory, don't act

Read the entire batch before making any ruling or committing any file. Build a list:

```
- [role] [date/time if known] [one-line topic] [pending/already-resolved/arc-closure]
```

### Step 2: Check git log for auto-resolution

For every "pending" item, run `git log --oneline --since="<last active>"` and scan for commits that close it. Many stale items self-close when a team member proceeds without you.

### Step 3: Check file state for partial-resolution

For items not closed by commits, read the relevant file. If the team member modified the file themselves (e.g., scout updated capability-matrix before your ruling), their change may already embody your intended ruling. Confirm rather than overwrite.

### Step 4: Categorize remaining items

- **Already resolved:** Acknowledge in check-ins.md. No further action.
- **Stale pending (member waited):** Make the ruling now. Note the stale-message lag in check-ins.md so the team knows it was delayed.
- **Arc closures:** Acknowledge explicitly. Add cross-reference to the artifact if one exists.

### Step 5: Broadcast the resolution

For any ruling that arrived stale and was subsequently made, send a message to the affected role with:
- The ruling
- An acknowledgment that the delay was a session gap, not a judgment about the item's priority
- Any follow-on actions assigned

---

## What happened this session

The batch had ~8 messages. Breakdown:

- **3 already-resolved** (adversarial oracle format, scientist campsite 4.8, scout VB-005 filing) — acknowledged in check-ins.md
- **1 arc closure** (aristotle e05d495 finalization) — accepted SHA, added cross-reference, closed arc
- **2 confirmations-needed** (scientist 4.8 TOML format, tanh B4 reconfirm) — reconfirmed, no action required
- **2 genuine open items** (VB-004 from adversarial, scout ESC-002 pre-flight queries) — ruled on both

Total ruling time: ~20 minutes for 8 messages. Systematic triage prevented 2-3 potential duplicate actions.

---

## Design consideration for future sessions

**"Stale" is a property of the message, not the item.** An item that arrived stale may still be genuinely urgent. The protocol doesn't deprioritize stale items — it categorizes them correctly before acting.

The systemic fix for large stale batches is not a better protocol — it's session continuity. The protocol above is the fallback. The prevention is: leave clean session-end-state.md files so the next navigator can resume in <5 minutes, minimizing the window where messages can queue.

See `session-end-state.md` for the current handoff template.


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

