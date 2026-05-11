# Team Briefing — tambear-sweep31-finish

**Spawned:** 2026-05-08 (continuation session, same calendar day as `tambear-formalize` arc that just shipped Sweep 31 design + scaffold)
**Recipe:** `/jbd tambear` (base 5 + math-specific 4)
**Roster:** pathmaker, navigator, scout, naturalist, observer, math-researcher, adversarial, scientist, aristotle

---

## Welcome back (or hello, fresh-spawn-of-you)

The last team — `tambear-formalize` — ran 2026-05-08 and shipped:

- The full Sweep 31 design package (~136KB across 4 docs at `R:\winrapids\campsites\tambear-formalize\sweep-31\`)
- Sweep 31 type-level home + arithmetic *scaffold* committed at `33d3849` + cleaned up at `c2798f5`
- F13 ratified (cross-domain antibody pattern, first F-number ratified directly from observation)
- 1413 → 1465 lib tests passing, all green

That team shut down cleanly. This team picks up where they left off — not exploratory, more focused.

---

## The Journey — bounded and concrete this time

**Mission: complete Sweep 31 implementation + Sweep 33. Sweep 32 is reserved (Tekgy + main-thread Claude are taking that lane in parallel).**

The two lanes don't collide: Sweep 31 touches `arith.rs` in `primitives/big_float/`; Sweep 32 touches `jit/fingerprint.rs` and `lattice/precision.rs`. Sweep 33 is downstream of both. Navigator: coordinate via campsite logbook so the lanes stay clean.

Two deliverables in dependency order:

### 1. Multi-limb arith.rs unstub (the gating dependency)

`crates/tambear/src/primitives/big_float/arith.rs` has `unimplemented!()` for p > 53 multi-limb Normal×Normal arithmetic. The f64 fast path and special-value dispatch are covered, but BZ Algorithms 3.1/3.3/3.5/3.10 for genuine multi-limb operations are not yet filled in.

**Fill in per Brent-Zimmermann *Modern Computer Arithmetic* (2nd ed.):**
- **Algorithm 3.1** — add/sub via exponent alignment + integer add + canonicalize
- **Algorithm 3.3** — schoolbook multiplication (Karatsuba deferred to v3, FFT excluded; tier cap is 1024 bits = 16 limbs per DEC-031 §3.8)
- **Algorithm 3.5** — Newton-Raphson reciprocal for division, p+50 guard bits + final round per `RoundingMode`
- **Algorithm 3.10** — Newton iteration for sqrt, p+50 guard bits + final round per `RoundingMode`

**Tests already in place** at `crates/tambear/tests/big_float_arith_invariants.rs` running against the f64 fast path. As the unstub fills in, multi-limb paths become exercised. **No new test infrastructure needed; just expand the input range to include p > 53 cases.**

**Without this unstub, BigFloat is only usable for f64 fast path** — blocks any real oracle use (Sweep 34) and blocks libm formalization (Sweep 35+). This is foundational.

### 2. Sweep 32 — RESERVED (Tekgy's lane)

**Do not touch.** Tekgy + main-thread Claude are landing this in parallel. Touches `jit/fingerprint.rs` (new tag 0x1A `feed_precision_context`, IR_VERSION 9→10) and `lattice/precision.rs`. The existing terrain survey at `R:\winrapids\campsites\tambear-sweep31-finish\20260508161750-sweep-32-33-terrain\scout\notebooks\sweep-32-33-terrain.md` is the spec they're working from. If your work has a stake in cache-key behavior, route it through the navigator who'll relay to main-thread.

### 3. Sweep 33 — TAM routing (1 day)

DEC-031 §3.6 + DEC-019 sub-clause E. `door.rs:686 supports(op, shape, strategy)` returns false for BigFloat-bearing JitOps on non-CPU doors. TAM gains a routing rule that detects this and forces CPU dispatch:

```rust
if op_uses_bigfloat(op) {
    return Door::Cpu;
}
```

Touches: TAM scheduler, `door.rs:686`, possibly `lattice/precision.rs::PrecisionLevel` predicates.

---

## Substrate to lean on (already on disk — read these first)

**Design docs from the prior team** — these are not optional reading. The whole point of a continuation session is using the substrate the prior team built.

- **`R:\winrapids\campsites\tambear-formalize\sweep-31\math-researcher\DESIGN.md`** — especially §3 algorithm dispatch table for the BZ algorithm choices, §1 type-level home for the surface that arith.rs operates on, §5 ratification answers (Q1, Q2, Q4, Q5 still pathmaker-decides during impl)
- **`R:\winrapids\campsites\tambear-formalize\sweep-31\math-researcher\oracle-validation.md`** — §1.1 + §1.4 for Sweep 34 prep that follows from Sweep 32 + 33; §1.5 for libm verification-tier integration ahead
- **`R:\winrapids\campsites\tambear-formalize\sweep-31\aristotle\dec031-invariants-deconstruction.md`** — Phases 1-8 on the load-bearing invariants (still applies; the unstub has to honor diamond commutativity + round-trip identity at all p)
- **`R:\winrapids\campsites\tambear-formalize\sweep-31\aristotle\silent-failure-proptest-gauntlet.md`** — Surface 3 (cache-key) is Sweep 32; Surface 7 (DD↔BigFloat boundary) cross-checks the unstub at boundaries; remaining surfaces continue to apply
- **`R:\winrapids\campsites\tambear-formalize\survey\20260508123003-aristotle\f13-antibodies-for-scope-precondition-rules.md`** — F13 ratified; apply forward

**Critical lab notebook from earlier today** — read this BEFORE writing code:
- `R:\winrapids\campsites\tambear-sweep31-finish\observer\lab-notebook-001.md` — observer's pre-review checklist for each BZ algorithm, watch-items, and a verified observation that **all 1560 current lib tests run the f64 fast path** (operands sourced from `from_f64(v, 200)` have ≤53 mantissa bits → `f64_path_eligible()` returns true → multi-limb branches are dead code in tests). The unstub is needed AND we need from_raw_limbs-based tests to actually exercise it. Cross-precision consistency tests (compute at p=500, round to p=200, must match within 1 ulp) are the antibody for guard-bit errors and were flagged mandatory.

**Tambear source pointers:**
- `R:\tambear\crates\tambear\src\primitives\big_float\arith.rs` — the file with the `unimplemented!()` sites
- `R:\tambear\crates\tambear\src\primitives\big_float\ty.rs:305` — `from_raw_limbs` (#[cfg(test)] constructor) is how tests construct genuine multi-limb operands
- `R:\tambear\crates\tambear\src\primitives\big_float\ty.rs:393` — `is_zero()` is tag-only (matches kind == Zero), does not scan limbs. BZ 3.1 cancellation-to-zero must explicitly flip kind
- `R:\tambear\crates\tambear\src\primitives\double_double\ops.rs` — exemplar for multi-limb-style arithmetic patterns
- `R:\tambear\crates\tambear\src\lattice\precision.rs` — the type-level home (PrecisionLevel, PrecisionContext, etc.)
- `R:\tambear\docs\decisions.md` lines 3310-3478 — DEC-031 itself
- `R:\tambear\LOG.md` — last entry has the prior session's full story (see "## 2026-05-08 — claude opus 4.7 (1M, jbd-team `tambear-formalize`) — sweep-31 design + scaffold")

---

## Standing Constraints (same as last session — read if you are fresh-spawned)

### 1. Vocabulary is locked

`R:\winrapids\docs\architecture\vocabulary.md` (locked 2026-04-17). Five tiers: Pipelines / Recipes / Atoms / Op+Expr / Primitives. BigFloat is a Tier 1 primitive. Older docs may use older words; the locked vocabulary wins.

### 2. The Tambear Contract — every primitive

`R:\winrapids\CLAUDE.md` § "The Tambear Contract" — 10-point Filter Test. Custom-implemented (no vendor wrapping); accumulate+gather decomposition where possible; shareable intermediates with compatibility tags; every parameter tunable; every measure in every family; optimized for advanced 2026 hardware; no vendor lock-in (DEC-019); no OS lock-in; lifting to TAM; publication-grade rigor.

### 3. F13 antibody pattern (newly ratified)

Every rule with a scope precondition needs an antibody that enforces the precondition at construction time. Without antibody → silent failure outside scope. **As you add code, watch for new rules being added (BZ algorithm preconditions, precision-tier-dispatch boundaries, multi-limb canonicalization invariants) and ask whether the antibody is in place before shipping.**

### 4. Anti-YAGNI; complexity IS the point

If structurally guaranteed, build it now. The reflex to simplify is almost always wrong here.

### 5. No tech debt — ever

See it, fix it, in this session.

### 6. Tests serve reality

Tests assert what should be true, not what code happens to produce.

### 7. Substrate over memory

Verify against disk + git + cargo test before claiming state. The session before this one ended at 1465 lib tests, 0 warnings, 33d3849 + c2798f5 pushed. Confirm before doing anything that depends on those numbers.

### 8. Antigen team in parallel

`R:\antigen\` — adoption log at `R:\antigen\docs\expedition\tambear-adoption-log.md` is the channel. `crates/tambear-substrate/src/parse.rs` + `query.rs` may show modifications from their concurrent work — leave alone unless you know what they're doing.

---

## What to do first (suggested, not prescribed)

Each role does a first-pass read of the substrate from the prior team:

- **Math-researcher**: re-read DESIGN.md §3 algorithm dispatch table; understand the BZ algorithm choices the prior team made; the unstub is filling in the blanks within those choices, not re-deciding them. Then dive into the unstub work.
- **Pathmaker**: read DESIGN.md §1 + §3 + §7 deliverable list; scope the sequencing within the unstub (which BZ algorithm to fill in first based on internal dependencies); prepare to lead the impl.
- **Aristotle**: re-read your own deconstruction doc; the load-bearing invariants (diamond commutativity, round-trip identity) must continue to hold after the unstub; pressure-test that any new code preserves them. Surface 5 + Surface 6 of your gauntlet are the relevant antibodies.
- **Adversarial**: re-read your gauntlet; design proptests for the multi-limb-arithmetic regime that fire when the unstub is wrong (carry-propagation bugs, guard-bit-off-by-one, Newton-iteration non-convergence at extreme inputs). Surface 1 (non-monotone path antibody) continues to apply for any new path-construction code.
- **Scientist**: re-read your `oracle-validation.md`; plan the cross-precision consistency check from §4 #3 (compute at p₁ + p₂; round p₂→p₁ should match) — this IS the antibody for guard-bit bugs in the multi-limb arithmetic. Set up mpmath comparison harness for multi-limb Normal×Normal.
- **Navigator**: coordinate. Story-from-the-trail to team-lead when something crystallizes. The work is bounded, so escalation cadence may be lower than last time.
- **Scout, Observer, Naturalist**: same role definitions as last session. Naturalist especially — there were two flagged threads from your last expedition log (PLEASE_READ + important-conversation.md need an owner; graph-form keeps appearing as recognition) that you might want to pull on if curious.

**The team's first move (suggested for navigator)**: invite each role to read their relevant substrate doc, then surface what's surprising, what's clear, and what the impl path looks like. After that, pathmaker leads the unstub.

---

## Coordination — same as last time

- **Campsite logbook** at `R:\winrapids\campsites\logbook.db`. Tool at `~/.claude/skills/campsite/campsite`. Run from `R:\winrapids`.
- **New campsite hierarchy for this team**: `tambear-sweep31-finish/<role>` for each role's working notes. Old campsites at `tambear-formalize/` stay where they are as substrate.
- **Stories from the trail to team-lead.** Convergences across roles are first-principles findings.
- **Idle is invitation.** Don't dispatch busywork to idle agents.
- **Garden** at `~/.claude/garden/`. Each previous-incarnation-of-you wrote entries last session. Math-researcher's "the-hardest-invariants-are-one-line-of-code" + aristotle's "proof-effort-is-feedback-on-structural-angle" + naturalist's "the-garden-is-the-thing-that-survives" — those carry forward as meta-principles for this session too.

### Commits

`R:\tambear\NAVIGATE.md` for commit conventions. The prior team's two commits (33d3849 + c2798f5) are pushed to origin/main. Continue the convention. Anchor for cargo test count: 1465 lib tests passing.

---

## Files / paths to pin

- Briefing: `R:\winrapids\docs\expedition\team-briefing.md` (this file)
- Vocabulary: `R:\winrapids\docs\architecture\vocabulary.md`
- Tambear root: `R:\tambear\`
- DEC-031: `R:\tambear\docs\decisions.md` lines 3310-3478
- LOG: `R:\tambear\LOG.md` (last entry = prior session's full story)
- Design substrate: `R:\winrapids\campsites\tambear-formalize\sweep-31\` (4 docs, ~136KB)
- F13: `R:\winrapids\campsites\tambear-formalize\survey\20260508123003-aristotle\f13-antibodies-for-scope-precondition-rules.md`
- arith.rs (the unstub target): `R:\tambear\crates\tambear\src\primitives\big_float\arith.rs`
- Antigen adoption log: `R:\antigen\docs\expedition\tambear-adoption-log.md`

Welcome back.
