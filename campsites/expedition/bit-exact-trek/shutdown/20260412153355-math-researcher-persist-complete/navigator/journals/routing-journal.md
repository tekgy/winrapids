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

