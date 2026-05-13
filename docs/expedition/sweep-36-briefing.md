# Sweep 36 — pointer doc

**The canonical briefing lives in tambear**, per tambear's sweep convention:

- `R:\tambear\sweeps\36-elementary-recipes\README.md` — full scope, framing, phases, acceptance criteria
- `R:\tambear\sweeps\36-elementary-recipes\STATE.md` — status, work-streams, authority

This doc exists in winrapids only as a pointer for main-thread orchestration. If you arrived here looking for the Sweep 36 briefing, read the tambear-side files above.

---

## One-line summary

Build the elementary transcendental family (expm1, log1p, exp/log/exp2/log2/exp10/log10, sinh/cosh/tanh, hypot, complex_log) **fresh in `R:\tambear`** using `R:\winrapids\crates\tambear\src\recipes\libm\` as design REFERENCE only — no copy-paste, no lift-and-shift. Tambear's locked vocab, holonomic discipline, F13.C antibodies, and kernel-state-consistency-tests are applied as structural properties from day one.

## Framing (the load-bearing part)

This is NOT a port. This is a **build-fresh-in-tambear** session using winrapids as REFERENCE. The math is settled (Sweep 35 verified Remez coefficients, naturalist named three-shapes structure, aristotle pressure-tested the kernel-state abstraction); the *implementation* is fresh, in tambear's idiom.

## Working directory

Team works in **`R:\tambear`** (commits land there). Main-thread orchestration may continue running from `R:\winrapids` working directory, but the team's edits all hit tambear's tree.

## Substrate trail

See the tambear-side README for full substrate. Key references in winrapids that are READ-ONLY for this sweep:

- `R:\winrapids\docs\architecture\tambear-libm-factoring.md`
- `R:\winrapids\docs\architecture\branch-cut-conventions.md`
- `R:\winrapids\docs\architecture\kernel-state-consistency-tests.md`
- `R:\winrapids\docs\expedition\winrapids-vs-tambear-audit.md` (the scope-finding audit)
- `R:\winrapids\docs\expedition\session-methodology-patterns.md` (8 patterns; especially Pattern 5 antibody-precedes-antigen)
- `~/.claude/garden/2026-05-10-the-three-shapes-of-complementary-argument.md`

These docs WILL migrate to tambear during Sweep 36 Phase F. After that, the team won't need to cross-reference winrapids except for the libm `.rs` reference reads.
