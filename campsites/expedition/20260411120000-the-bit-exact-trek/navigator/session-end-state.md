# Navigator Session-End State — 2026-04-12

Written at session shutdown. Authoritative handoff for the next navigator instance.

---

## Where the expedition stands

### Peaks closed
- **Peak 1** (tam IR): closed at 748eb53. 68 tests green. All campsites 1.1–1.18 done.
- **Peak 4** (oracle/replay harness): closed at d4141ec. 67 tests green. Oracle runner has three-section TOML format (bit_exact_checks, constraint_checks, identity_checks). tam_sqrt chain proven bit-exact across 1M samples.

### Peaks in progress
- **Peak 2** (tambear-libm): exp design doc fully amended (B1–B5 resolved). Campsite 2.6 (tam_exp implementation) is the next concrete step — unblocked. All other function design docs (tan, log, sin-cos, pow B1+B4, hyperbolic B1–B4 with tanh using two-call form, atan B1–B3) have clear resolution paths but are not yet amended. Math-researcher has sequencing confirmed: exp first, then 2.6, then other amendments in parallel.

### Peaks not started
- Peak 3 (tam→PTX): lowering-blueprint.md exists, pathmaker has VB-003/VB-004 noted
- Peak 5 (CPU backend formalization): pending
- Peak 6 (deterministic reductions): pending
- Peak 7 (SPIR-V): capability matrix fully drafted and updated, pre-flight queries resolved

---

## Open routing decisions (none blocking, all documented)

### Design-doc amendments (math-researcher)
- pow B1: `pow(-0, y > 0, odd integer)` should be `-0` not `+0` — IEEE 754-2019 §9.2.1
- pow B4: negative-base integer-exponent branch missing from skeleton (pitfall #6 mentions it but algorithm doesn't implement it)
- All other function amendments: have rulings, awaiting math-researcher implementation
- Tanh B4 ruling: two-call form `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` — no expm1, no carve-out

### Pathmaker
- BackendTrait shape: keep "fp64_available" capability bit addable without breaking changes (Part II.5 post-IEEE-754 scope)
- NaN payload / QNAN vs SNAN: check op-reference.md for any ops whose semantics assume specific NaN bit layout beyond IEEE-754 spec

### Scientist
- tam_exp_passes_oracle: unwire the `#[ignore]` when campsite 2.6 delivers tam_exp
- Overflow oracle coverage: verify near_overflow inputs (≥ 709.78) have bit_exact_checks entries for +inf (0x7FF0000000000000) so sign of infinity is confirmed, not just ULP

### Scout / Peak 7
- fsqrt RoundingMode (WithinULP(1)): device test deferred to when Vulkan emit lands. Campsite created.
- vulkaninfo JSON: `peak7-spirv/vulkaninfo-RTX-PRO-6000-Blackwell-driver-595.71.0.0.json` exists and should be committed

---

## Invariant status (all I1–I11)

| Invariant | Status | Notes |
|---|---|---|
| I1 | Green | No vendor math library anywhere |
| I2 | Green | No vendor source compiler |
| I3 | Green | `.contract false` required at PTX emission (VB-003); NoContraction in SPIR-V |
| I4 | Green | No reordering flags in any path |
| I5 | Green | Phase 1 uses single-thread reduce; Peak 6 fixes atomicAdd |
| I6 | Green | Capability matrix gates; I6 trigger path documented |
| I7 | Green | OrderStrategy registry live, `is_fusable_with` implemented |
| I8 | Green | All transcendentals from first principles |
| I9 | Green | Oracle runner live (269a338), mpmath reference files committed |
| I10 | Green | Cross-backend diff infrastructure in place (Peak 4) |
| I11 | Green | NaN guards in IR interpreter; PTX min.NaN.f64 required; Vulkan composed sequences mandated |

---

## Vendor bugs log (vendor-bugs.md) — current entries

- **VB-001**: Vulkan OpFMin/OpFMax NaN undefined → composed from OpIsNan + OpFOrdLessThan + OpSelect (ESC-002 Option 3). MITIGATED.
- **VB-002**: Vulkan shaderDenormPreserveFloat64 optional → bit-exactness claim narrowed to normal fp64 (ESC-001). MITIGATED.
- **VB-003**: PTX .contract default is contract-everywhere → emit .contract false unconditionally (I3). MITIGATED.
- **VB-004**: PTX min.f64 / max.f64 default drops NaN → use min.NaN.f64 / max.NaN.f64 (ISA 7.5+); composed fallback for pre-7.5. MITIGATED.
- **VB-005**: GLSL.std.450 Sqrt NaN underspecified → OpIsNan guard mandatory for all OpExtInst. MITIGATED.

Standing rule in capability matrix preamble: every OpExtInst requires OpIsNan guard. Do not rediscover per function.

---

## Named architectural artifacts (four registries)

1. **OrderStrategy registry** (`crates/tambear-tam-ir/src/order_strategy.rs`): named entries, `is_fusable_with` predicate, verifier-enforced
2. **Oracles registry** (`campsites/expedition/.../oracles/`): TOML per function, three-section format, oracle runner at `crates/tambear-tam-test-harness/src/oracle_runner.rs`
3. **Guarantee Ledger** (`campsites/expedition/.../guarantee-ledger.md`): I1–I11 mapped to P1/P2/P3, cost-of-relaxation, P2 tightened to forbid semantically-ambiguous vendor ops
4. **Device Capability Matrix** (`campsites/expedition/.../peak7-spirv/capability-matrix-vulkan-row.md`): (backend × op × precision) → CapabilityEntry. VulkanBackend row complete with confirmed device values.

---

## P2 tightening — the ruling that matters

The faithful-lowering precondition (P2) was tightened this session. The key addition:

> The lowering must NOT delegate to vendor ops whose behavior is implementation-defined, conditional on optional extensions, or otherwise ambiguous in the target spec. Where a convenient vendor op has ambiguous semantics for the use case (e.g., NaN handling), the backend MUST compose from unambiguous primitives unconditionally.

This is what required ESC-002 Option 3 (compose min/max from scratch, never emit OpFMin) and VB-005 (OpIsNan guard for fsqrt). The Bucket A vs Bucket B distinction: not "vendor op good/bad" but "which part of IEEE-754 is the op touching?" Core arithmetic (OpFAdd, OpFMul) = pinned by spec + SignedZeroInfNanPreserve. NaN corners for non-arithmetic ops = not pinned, must compose.

---

## Commit hygiene note

Policy: `git add <specific-file>`, never `git add .` or `git add -A`. Enforced after e05d495 accidentally swept aristotle's ULP budget addendum into an RFA commit. The policy is documented and all team members have been notified.

---

## What the next navigator needs to do first

1. Read this file and `navigator/escalations.md` (ESC-001 and ESC-002 both ruled and closed)
2. Read `navigator/check-ins.md` — most recent entries capture the VB-005 ruling and sequencing confirmation
3. Check whether math-researcher has started campsite 2.6 — that's the active critical path
4. The campsite log (read via campsite CLI or the campsites/ directories) has the next-session pickup points for each role

No escalations are currently open. No invariants are in tension. The expedition is in execution mode, not architectural mode.
