# Sweep 37 — pointer doc

**The canonical briefing lives in tambear**, per tambear's sweep convention:

- `R:\tambear\sweeps\37-trig-family\README.md` — full scope, framing, 9 work-streams, acceptance criteria
- `R:\tambear\sweeps\37-trig-family\STATE.md` — status, authority boundaries

This doc exists in winrapids only as a pointer for main-thread orchestration.

---

## One-line summary

Build the trig family + complex_log + Complex&lt;T&gt; primitive + feed_branch_policy machinery + TrigKernelState **fresh in `R:\tambear`** using `R:\winrapids\crates\tambear\src\recipes\libm\` as design REFERENCE only. Continuation of Sweep 36's elementary-family pattern.

## Framing

This is NOT a port. Same discipline as Sweep 36: build-fresh-in-tambear using winrapids as REFERENCE. Expect coefficient typos in winrapids' references (Sweep 36 found 3 in expm1.rs); math-researcher verifies polynomial coefficients against literature before implementing.

Sweep 36's idiom is established. Phase 0 is LIGHT GATING — re-read PHASE0_NOTES.md, surface trig-specific deltas, proceed.

## Working directory

Team works in **`R:\tambear`** (commits land there). Main-thread orchestration runs from `R:\winrapids` working directory.

## Why Sweep 37 before Sweep 8

Tekgy's call 2026-05-11: rate-limit pragmatism + momentum-preserving continuation. Sweep 8 (multi-door JIT finalization) is the larger load-bearing critical path that comes after — without Sweep 8, tambear is CPU-eager-only; with it, tambear becomes the multi-door JIT-compiled toolkit DEC-019 promises. But Sweep 37 first finishes Sweep 36's arc cleanly + widens the user-facing math catalog where users actually do most computing.

## Substrate trail

See tambear-side README for full substrate.

## Three RATIFIED DECs unblock this sweep

- DEC-032 (Branch enum + cache-key tag 0x1B) — RATIFIED 2026-05-11
- DEC-033 (TamSession Fork B with R1 TAM-owns-routing + R2 ONE-tambear-algorithm) — RATIFIED 2026-05-11
- DEC-034 (kernel-state-consistency-tests META-class) — RATIFIED 2026-05-11
