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
