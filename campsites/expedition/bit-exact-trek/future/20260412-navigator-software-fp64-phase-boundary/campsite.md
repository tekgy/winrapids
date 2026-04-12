# Navigator Campsite: Software FP64 Phase Boundary

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** Future work — phase boundary analysis

---

## What this is

Team-lead's Part II.5 update outlined a three-phase software fp64 roadmap:
- Phase 1: Hardware fp64 + composed workarounds (current)
- Phase 1.5: Software fp64 as oracle/validation only (not production path)
- Phase 2/3: Software fp64 as production path, triggered by hybrid classical-quantum workloads

This campsite documents the navigator's analysis of what changes at each phase boundary — specifically, which invariants in the guarantee ledger need new rows, which escalation categories shift, and what the routing looks like when software fp64 becomes a first-class backend.

---

## The Phase 1 → Phase 1.5 boundary

**What changes:** Software fp64 becomes an available oracle but doesn't go on the critical path. The phase boundary is "software fp64 exists but only scientist uses it."

**Guarantee ledger impact:**
- I9 (mpmath oracle) gets a sibling oracle: I9b (software fp64 oracle). They're complementary, not competing — mpmath catches errors in the mathematical definition, software fp64 catches errors in the hardware fp64 execution path.
- I1 (f64 as base precision) remains unchanged. Phase 1.5 software fp64 doesn't affect what precision tambear operations are computed at.

**Escalation category shift:**
- New escalation type emerges: "mpmath oracle agrees, software fp64 oracle disagrees." This would indicate a hardware fp64 numerical stability issue (not a math error and not a code error). Route to: scientist + pathmaker joint investigation.

**Routing change:**
- Scientist gets a new tool: "run this against software fp64 oracle." This doesn't change the routing table for other roles — it adds a new row to scientist's oracle registry.

---

## The Phase 1.5 → Phase 2 boundary (the meaningful one)

**What changes:** Software fp64 moves from oracle-only to production path. Some computations that currently run on hardware fp64 will run on software fp64 instead (because hardware fp64 isn't available on the target, e.g., Majorana quantum co-processors or future NPUs without native f64).

**This is where the guarantee ledger needs significant work.**

### P1 (IR precision) changes

Currently P1 says: "all computation in the .tam IR uses f64 precision — no implicit widening to 128-bit, no narrowing to f32." When software fp64 is a production backend, P1 becomes: "all computation uses at least f64 precision — software fp64 counts as f64 precision because it implements f64 semantics exactly."

This is actually *stronger* than current P1 in one sense: software fp64 gives you bit-exact f64 semantics *regardless* of the hardware, eliminating the I3 (no FMA) and I11 (NaN propagation) concerns. But it introduces a new concern: performance. The ledger would need a new section on "precision vs. performance tradeoffs when software fp64 is available."

### P2 (faithful lowering) simplifies

When software fp64 is the backend, the P2 audit checklist shrinks: there are no Bucket B ops in software fp64 because every fp64 operation is implemented by us. The entire Bucket A/B distinction evaporates — there are no vendor ops with ambiguous specs.

**This is a significant simplification.** Phase 2 software fp64 backend would have trivial P2: everything is faithful by construction.

### P3 (IEEE-754 hardware compliance) is resolved differently

Currently P3 says: "the hardware implements IEEE-754 correctly for the ops used." For software fp64, P3 becomes: "the software fp64 implementation correctly implements IEEE-754 — provable by testing against mpmath at 50-digit precision."

This shifts P3 from "claim about hardware we don't control" to "claim about code we do control." The verification burden shifts accordingly: instead of running a pre-flight device query, you run a software fp64 test suite.

---

## The routing change at Phase 2

When software fp64 becomes a production backend, navigator needs a new escalation category:

**"Hardware fp64 and software fp64 disagree for this input."**

This would be a new class of issue — neither P1, P2, nor P3 in isolation. It would indicate either:
- A bug in the software fp64 implementation (→ pathmaker)
- A hardware fp64 deviation from IEEE-754 for a specific input (→ VB entry + scout)
- A precision loss in the P2 lowering that's acceptable within ULP budget but not bit-exact (→ ledger review)

The routing table would need a new row: "software/hardware fp64 disagreement → create VB entry, classify as hardware precision deviation, check ULP budget."

---

## The Majorana timeline implication

Team-lead flagged Majorana 1 as placing Phase 2 at 2-5 years. From a navigator perspective, this is useful: it means:

1. Phase 2 routing changes are not imminent. Don't design for them now.
2. The current ledger (I1–I11) is sufficient through Phase 1.5 and possibly into early Phase 2.
3. The BackendTrait fp64-availability capability bit (delegated to pathmaker) is the only structural change needed now to be Phase 2 ready.
4. Write the guarantee ledger Kingdom B extension (see `navigator-guarantee-ledger-kingdom-b-c-d-extension`) before writing the Phase 2 extension. Kingdom B is closer in the implementation timeline.

---

## The specific question I didn't answer this session

Team-lead's Part II.5 asked: "What does P1+P2 → bit-exact look like without P3 (Phase 2 software fp64)?"

The honest answer is: **bit-exact under software fp64 is trivially achievable, but it's a different kind of bit-exact than hardware fp64 bit-exact.**

- Hardware fp64 bit-exact: "the same .tam program produces the same bits on any IEEE-754-compliant hardware"
- Software fp64 bit-exact: "the same .tam program produces the same bits on any platform, including platforms where hardware fp64 isn't available"

These are different claims. The first is stronger in one sense (hardware-agnostic), the second is stronger in another (platform-agnostic, including non-fp64 hardware).

The guarantee ledger extension for Phase 2 would need to explicitly separate these two claims and state which one applies to which deployment target. That's a 1-2 page addition to the ledger when Phase 2 lands.

See: `navigator-guarantee-ledger-kingdom-b-c-d-extension` for the related Kingdom B analysis, which also involves expanding the ledger's claim scope.
