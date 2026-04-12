# Navigator Routing Journal — Bit-Exact Trek 2026-04-12

Chronological record of routing decisions made this session. Format: decision → rationale → outcome.

---

**ESC-002 Option 3 ruling**
- Decision: Never emit OpFMin or OpFMax. Use six-instruction composition.
- Rationale: Option 1 (guard before call) still emits OpFMin; if the guard fails or is bypassed, the dependency on undefined behavior survives. Option 3 eliminates the dependency entirely. Every instruction in the six-instruction sequence has fully-defined IEEE-754 semantics.
- Outcome: Documented in escalations.md, vendor-bugs.md, capability-matrix, guarantee-ledger. Scout rewrote capability matrix from Option 1 to Option 3. ESC-002 marked RESOLVED.

**VB-005 ruling (GLSL.std.450 Sqrt NaN)**
- Decision: Same principle as ESC-002. GLSL.std.450 restricts Sqrt to real-valued inputs; NaN is outside the domain; behavior is undefined. Mandatory: emit OpIsNan guard before every OpExtInst call.
- Rationale: Direct application of P2 (now tightened). No new escalation needed — the principle is already in the ledger.
- Outcome: Scout filed VB-005 and updated capability matrix. No separate navigator action beyond confirming the ruling.

**Standing OpExtInst rule**
- Decision: Three-instruction prefix for single-input (OpIsNan → OpSelect(NaN → return NaN, else → OpExtInst)), four-instruction for two-input. Applies to all GLSL.std.450 extended instructions.
- Rationale: Prevents per-function rediscovery of the same gap. The pattern is uniform across all extended instructions because the spec restriction is uniform (all require real-valued inputs).
- Outcome: Added to capability matrix preamble by scout.

**Aristotle arc closure (e05d495)**
- Decision: Accept SHA as the arc-completion pointer. Add expedition-log cross-reference.
- Rationale: Aristotle recommended Option 1; content is committed and correct; standalone "pointer commit" adds no information. The expedition-log cross-reference is the documentation that makes the arc visible.
- Outcome: Cross-reference added. Arc marked closed.

**Math-researcher sequencing (exp B1-B5 done, can 2.6 start?)**
- Decision: Yes. Campsite 2.6 can start in parallel with remaining design amendments (pow, tan, log, sin-cos, hyperbolic, atan). The amendments are editorial, not blocking.
- Rationale: The amendments fix formal specification errors in design docs; they don't invalidate the implementation. 2.6 starts from the correct spec (exp-design B1-B5 as amended). Running amendments and 2.6 in parallel is safe.
- Outcome: Math-researcher unblocked on 2.6. Scientist's tam_exp_passes_oracle #[ignore] to be unwired when 2.6 delivers.

**Tanh B4 reconfirmation**
- Decision: Two-call form is correct. `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`. No expm1. No carve-out.
- Rationale: Arithmetic error at the boundary is ≤2 ULP, which is within the documented 2-ULP budget for tanh (division-amplified error tier). Adding expm1 adds complexity without being necessary for the budget.
- Outcome: Confirmed. No change needed.

**BackendTrait fp64-availability (Phase 2 prep)**
- Decision: Delegate to pathmaker. BackendTrait must accommodate a future "fp64_available" capability bit without breaking changes. Pattern: capabilities() struct or has_capability(CapabilityBit) query.
- Rationale: Phase 2 software fp64 is 2-5 years out (Majorana timeline). The trait shape can be future-proofed now with minimal cost. Additive design means no breaking changes when the capability bit is eventually needed.
- Outcome: Broadcast to pathmaker. Design choice left to pathmaker with the additive-non-breaking constraint.

---

**Open routing items carried to next session**
- Named artifacts convention: draft the formal document (delegate to aristotle + navigator collaboration)
- Guarantee ledger Kingdom B extension: hold until Kingdom B implementation exists
- Software fp64 phase boundary: no action until Phase 1.5 planning begins
- vulkaninfo JSON evidence: scout to commit on next device-query session
