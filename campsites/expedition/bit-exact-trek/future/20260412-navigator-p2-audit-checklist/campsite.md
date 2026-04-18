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

# Navigator Campsite: P2 Audit Checklist

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** Protocol documentation — pre-op checklist derived from P2 tightening

---

## What this is

This session tightened the P2 (faithful lowering) guarantee from syntactic faithfulness to semantic pinning. The full updated language is in `guarantee-ledger.md`. The cascade of VB-001 through VB-005 all traced to the same root: a vendor op whose spec left NaN/subnormal/rounding behavior unspecified or optional.

This campsite extracts P2 into a concrete pre-op audit checklist that a future pathmaker can run before emitting any new op.

---

## The P2 requirement (from guarantee-ledger.md)

> The lowering must pin the IEEE-754 semantics of every emitted op from the target spec alone. The lowering is a homomorphism to a **semantically-pinned subset** of the target ISA. If a vendor op's behavior for the relevant inputs (NaN, subnormals, rounding) is implementation-defined, conditional on optional extensions, or otherwise ambiguous in the target spec, that op cannot appear in a faithful lowering — compose from unambiguous primitives instead.

---

## The pre-op checklist

Before emitting any new op in the PTX or SPIR-V backends, answer these five questions:

### Q1: NaN input behavior — is it spec-pinned?

For every op you're about to emit, find the relevant spec entry and check:
- What does the spec say this op returns when one or both inputs are NaN?
- Is the answer unconditional ("the result is NaN") or conditional ("if implementation supports NaN propagation")?

**Pass:** The spec says "the result is NaN" (or "the result is QNaN from one of the inputs") unconditionally.
**Fail:** The spec says "results are undefined for NaN inputs," "NaN behavior is implementation-defined," or restricts the domain to finite or real-valued inputs. → Compose with OpIsNan guard.

### Q2: Subnormal input behavior — is it spec-pinned?

- What does the spec say when inputs are subnormal (denormal)?
- Is subnormal support required or optional?

**Pass (Vulkan):** shaderDenormFlushToZeroFloat64 = false confirmed on target device; the device preserves subnormals. OR the op's output is semantically unchanged under FTZ for your use case.
**Fail:** Subnormal behavior is "implementation-defined" or "may flush to zero." → Emit only if target device row in capability-matrix confirms subnormal preservation; else compose from subnormal-safe primitives.

**Current status:** RTX 6000 Pro Blackwell has shaderSignedZeroInfNanPreserveFloat64 = true. Subnormal flush behavior: still verify per-op in SPIR-V. See capability-matrix-vulkan-row.md.

### Q3: Rounding mode — is it spec-pinned?

- Does the op's spec specify "round-to-nearest-ties-to-even" as the required rounding mode?
- Or does it leave rounding to "implementation default"?

**Pass:** The spec mandates RNE (or equivalent: "correctly rounded").
**Fail:** "The result is within 1 ULP" without specifying RNE, or "the result depends on the current rounding mode." → Flag the op; may be acceptable with documented ULP budget but cannot participate in a bit-exact claim.

**Current status:** Vulkan SPIR-V core ops (OpFAdd, OpFMul, OpFDiv) — pass (SignedZeroInfNanPreserve + RNE implied). GLSL.std.450 extended instructions — fail for NaN (see VB-005); ULP budget also not specified. Always composing.

### Q4: Signed zero — is it spec-pinned?

- Does the op preserve +0 vs -0 distinctions correctly per IEEE-754 §6.3?
- Example: `(-0) + (-0) = -0`, but `(-0) + (+0) = +0` per IEEE-754.

**Pass (Vulkan):** shaderSignedZeroInfNanPreserveFloat64 = true confirmed; Vulkan core ops preserve sign of zero.
**Fail:** Op uses integer comparison semantics (`0 == -0` but `0 bits != -0 bits`) for an equality check. → Never use OpIEqual for float equality; always OpFOrdEqual. See capability-matrix-vulkan-row.md warning.

### Q5: Contraction / FMA fusion — is it opt-in or opt-out?

- Does the target backend contract FP operations by default?
- If so, is there a per-operation annotation to disable it?

**PTX:** Default is FP contracts enabled unless `.contract false` suffix used. See VB-003.
**SPIR-V:** NoContraction decoration on each OpFMul or OpFAdd that must not be contracted. Must be explicit.
**Pass:** The emitted code carries the explicit no-contract annotation on every arithmetic op.
**Fail:** Emitting raw OpFMul/OpFAdd without NoContraction in an I3-critical path. → Must add NoContraction or document the explicit exception.

---

## How to use this checklist in code review

When reviewing a new backend lowering implementation (in a PR or a team member message):

1. Find every vendor op in the implementation.
2. For each op, answer Q1–Q5.
3. Any "Fail" that doesn't have a documented composition workaround is a P2 violation.
4. File a VB entry in `vendor-bugs.md` for any vendor op that fails Q1 or Q2 for a reason that isn't already documented. Future navigator will thank you.

---

## The Bucket A / Bucket B distinction (from VB-001 through VB-005 convergence)

The convergence check on VB-001 through VB-005 produced this structural finding:

**Bucket A** (spec-pinned, usable directly):
- OpFAdd, OpFMul, OpFDiv, OpFSub — pass all five Qs with SignedZeroInfNanPreserve + RNE
- OpFNegate — trivial sign flip, no rounding, spec-pinned
- OpFOrdEqual, OpFUnordEqual, OpFOrdLessThan, etc. — comparison ops, spec-pinned
- OpIsNan, OpIsInf — classification ops, no arithmetic, fully defined
- FAbs (via BitAnd mantissa) — spec-pinned if done via bit manipulation
- OpSelect — pure mux, no arithmetic semantics

**Bucket B** (compose from Bucket A, never emit directly in faithful lowering):
- OpFMin, OpFMax — NaN undefined (VB-001)
- GLSL.std.450 sqrt — NaN outside domain (VB-005)
- GLSL.std.450 log, exp, sin, cos, atan, pow — same domain restriction, same guard needed
- Any GLSL.std.450 extended instruction — default to Bucket B until proven otherwise

**The meta-rule:** The boundary between Bucket A and Bucket B is exactly this checklist. An op in Bucket A passes all five Qs. An op in Bucket B fails at least one.

---

## Related artifacts

- `guarantee-ledger.md` → P2 row (faithful lowering)
- `vendor-bugs.md` → VB-001 through VB-005 (the evidence base)
- `capability-matrix-vulkan-row.md` → Standing OpExtInst NaN guard rule (preamble)
- `escalations.md` → ESC-002 (canonical P2 ruling)


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

