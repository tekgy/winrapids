# Navigator Campsite: Named Artifacts Convention

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** Future work — formalize the four-registry pattern

---

## What this is

Aristotle's `20260412152403-aristotle-named-artifacts-convention-draft` campsite identified the structural rhyme across the expedition's four named artifacts (OrderStrategy registry, Oracle registry, Guarantee Ledger, Device Capability Matrix). This campsite is the navigator's angle on the same observation — what I need from a formal convention document to route effectively.

This is a companion to aristotle's campsite, not a duplicate. The two angles should be merged when the convention is written.

---

## The four named artifacts

| Artifact | Owner | Function in expedition |
|---|---|---|
| OrderStrategy registry | pathmaker | Named, versioned summation strategies; prevents ad hoc `partial_cmp` choices |
| Oracle registry | scientist | Named oracle contracts; specifies what counts as "correct" for each function |
| Guarantee Ledger | navigator | Named invariants with formal cost-of-relaxation; routes escalations |
| Device Capability Matrix | scout | Named device properties with confirmed values; routes implementation choices |

**The structural rhyme** (aristotle's finding): all four artifacts share the same lifecycle shape:
1. Something in the expedition is non-obvious or contentious
2. A named entry is created with formal specification
3. Future decisions are made by checking the registry, not re-deriving from scratch
4. The registry is the ground truth; the code implements the registry's spec

**The meta-finding**: this lifecycle shape is the three-registry pattern from tambear's broader methodology (named inspection step → formal-spec entries → capability metadata → review-time enforcement). It emerges whenever a system has decisions that need to be made consistently across a long implementation timeline.

---

## What navigator needs from a formal convention

When I receive an escalation or a routing decision, I need to know:

1. **Which artifact is the relevant one?** Is this a summation strategy question (→ OrderStrategy), an oracle correctness question (→ Oracle registry), a hardware capability question (→ Capability Matrix), or a cross-cutting invariant question (→ Guarantee Ledger)?

2. **Does an entry exist?** If not, should I create one (escalation → new ledger entry), or route to the relevant role to create one (new device property → scout creates capability matrix row)?

3. **What's the escalation path if the entry doesn't resolve the question?** Some questions are novel enough that the registry entry doesn't exist yet. What's the protocol for creating a new entry vs. ruling directly?

A formal convention document would answer all three questions in ~2 pages.

---

## What the convention should specify

### Entry lifecycle (for all four artifacts)

```
PROPOSED → DRAFT → ACTIVE → SUPERSEDED
```

- **PROPOSED**: Someone noticed a decision being made ad hoc and proposed formalizing it. Not yet agreed.
- **DRAFT**: The entry is written and under review. Used for routing decisions but explicitly flagged as provisional.
- **ACTIVE**: The entry is agreed and canonical. Used without qualification.
- **SUPERSEDED**: A better version exists. The entry remains for historical context with a "see: <new entry>" pointer.

### Entry fields (minimum required for all artifacts)

| Field | Purpose |
|---|---|
| `name` | Short identifier (used in citations) |
| `status` | PROPOSED / DRAFT / ACTIVE / SUPERSEDED |
| `owner` | Which role created and maintains this entry |
| `spec` | The formal specification — the one thing this entry says |
| `rationale` | Why this entry exists; what decision it prevents from being ad hoc |
| `cost_of_relaxation` | What you give up if you don't follow this entry |
| `related_entries` | Cross-references to other artifacts |

### Registry-specific fields

- **OrderStrategy**: `compat_class` (for `is_fusable_with`), `is_deterministic` flag
- **Oracle registry**: `precision_level` (bit-exact / within-ULP / convergent), `reference_source`
- **Guarantee Ledger**: `precondition` (P1/P2/P3), `user_facing_property`, `cross_reference`
- **Capability Matrix**: `device_id`, `property_key`, `value`, `evidence_source`, `driver_version`

---

## The routing table I want to hand the next navigator

Once the convention exists, the routing table should be:

| Question type | First check | If not found |
|---|---|---|
| "Which summation strategy for this reduction?" | OrderStrategy registry | Escalate to pathmaker |
| "What's the correct output for this input?" | Oracle registry | Escalate to scientist |
| "What does this device support?" | Capability Matrix | Escalate to scout |
| "Does this invariant hold?" | Guarantee Ledger | Escalate to navigator |
| "Should this be in the Guarantee Ledger?" | Navigator judges | Write new ledger entry |

**The decision rule:** If a question is in the registry, route to the registry. If not in the registry, escalate to the owner of the relevant registry.

---

## Relationship to escalation categories

The three escalation categories from `escalations.md` map to the registry lookup:

- **ESC (escalation):** A question that can't be answered from any existing registry entry. Outcome: navigator rules and (if the ruling is structural) creates a new ledger entry.
- **Direct ruling:** A question that IS answerable from a registry entry + a new input. Outcome: navigator applies the registry and documents the application.
- **VB entry:** A question about vendor behavior. Outcome: scout creates a capability matrix entry, possibly escalates to navigator for a ruling if the gap affects the Guarantee Ledger.

This classification is what the `navigator-escalation-pattern-catalog` campsite formalizes.

---

## Note for aristotle

This campsite covers the navigator's perspective on the convention. Aristotle's campsite covers the structural analysis (why all four artifacts rhyme and what makes the pattern general). The actual convention document should integrate both angles.

When writing the convention, start from aristotle's structural claim (these four artifacts are instances of the three-registry pattern) and add the navigator's operational needs (entry lifecycle, minimum fields, routing table). The resulting document should be short (~3 pages) and self-contained.

File it at: `campsites/expedition/20260411120000-the-bit-exact-trek/navigator/named-artifacts-convention.md` — alongside the other navigator standards documents.
