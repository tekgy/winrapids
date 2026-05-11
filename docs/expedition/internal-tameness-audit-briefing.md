# Internal-tameness audit pass — briefing

**Status**: Ready to pick up. Briefed by main-thread Claude + Tekgy at the close of the 2026-05-08/09 session.

**One-line summary**: Walk every existing BigFloat-and-adjacent operation through the implicit-tameness-predicate audit pattern. Surface the lint candidates the audit reveals. Apply as a deliberate methodology *before* shipping new operations, not retroactively after adversarial generators fire.

**Lighter-weight than Sweep 35**: this is an audit pass, not new implementation. Output is a discovered list of antibody-needing sites + (maybe) a tooling sketch. Could run in parallel with Sweep 35 by adversarial + a sub-agent.

---

## Why now

During 2026-05-08/09, ~12 bugs surfaced in BigFloat arithmetic that all shared one structural shape (per `R:\winrapids\docs\architecture\internal-tameness-contracts.md`): implicit tameness contracts on intermediate representation that fail at type-boundary corners. The team fixed them organically as adversarial generators fired. The audit pass surfaces the *remaining* sites *before* an adversarial generator finds them.

The methodology is mechanical enough to be partly automated. Lint candidates:
- `i64-arithmetic-without-saturation` (would have caught attack18/23/24/25)
- `mantissa-rounding-without-carry-bump-check` (attack25 specifically)
- `limb-zero-without-kind-flip` (the cancellation-to-Zero antibody)
- `special-value-dispatch-consistency` (attack15/17 — NaN payload propagation)
- `f64-fast-path-without-result-finiteness` (the Newton-seed subnormal divergence #11)

---

## Required pre-flight reading

1. **`R:\winrapids\docs\architecture\internal-tameness-contracts.md`** — the methodology + the bug-class shape + the five lint candidates
2. **Adversarial's wind-down garden entry** — `~/.claude/garden/2026-05-09-the-tame-inputs-doctrine.md` — the adversarial-side framing of what the BZ attack class taught about saturation as antibody form, why the false alarm on `sticky_hi_bit` was worth pursuing, what the Newton-path "nothing found" finding actually means, and the libm contrast (f64 doesn't panic on overflow, it silently returns wrong answers — same adversarial role, different pathology shape)
3. **Naturalist's 2026-05-09 essays** — the broader holonomic framing of non-holonomic local defenses
4. **F13.C graduation condition** — aristotle's signature-level antibody requirement (in the canonical F13 doc within the campsite)
5. **The bug-fix commits** — `8b122b6` (saturating-exp audit), `f8a93d6` (saturating arithmetic for extreme exponents), `05d32ee` (to_f64 carry-out saturation), `19eb63e` (canonicalize mag=0), `22e3758` (sticky-bit loss), etc.

---

## What "done" looks like

### Phase A — Audit the existing BigFloat surface

Walk every `pub fn` in `R:\tambear\crates\tambear\src\primitives\big_float\` and ask the five-step questions from `internal-tameness-contracts.md` §"Methodology consequence":

1. List every intermediate state the function produces
2. For each intermediate, name the implicit tameness predicate (what subspace of the type does the algorithm assume?)
3. For each predicate, identify the boundary (inputs that push intermediate to the predicate's edge)
4. For each boundary, decide handling: saturate / detect-and-branch / reject
5. Make the chosen behavior structural — at every signature, not just public API

Output: `R:\winrapids\campsites\internal-tameness-audit\big-float-audit.md` — per-function audit results.

### Phase B — Audit the JIT and lattice surfaces

Same pattern for:
- `R:\tambear\crates\tambear\src\jit\` (fingerprint, shape, strategy, jit_op, using_annotation, door)
- `R:\tambear\crates\tambear\src\lattice\precision.rs`

Output: `R:\winrapids\campsites\internal-tameness-audit\jit-audit.md` and `lattice-audit.md`.

### Phase C — Lint candidate evaluation

For each of the five lint candidates from internal-tameness-contracts.md §"Tooling opportunity":

- How many call sites does the lint cover?
- What's the false-positive rate (call sites where the unchecked arithmetic is genuinely safe by other invariants)?
- Cost-benefit: worth implementing as a clippy-style lint, a build-script check, a documented checklist, or skip?

Output: `R:\winrapids\campsites\internal-tameness-audit\lint-evaluation.md`.

### Phase D — Apply audit to in-flight code

The audit *pattern* should become part of the recipe-authoring workflow (alongside `R:\tambear\docs\HOW_TO_ADD_A_RECIPE.md`). When pathmaker writes a new operation, the audit pass is a checkbox before commit.

Output: documented addition to recipe-authoring workflow OR a new section in `R:\tambear\docs\HOW_TO_ADD_A_RECIPE.md`.

### Acceptance criteria

- Every public function in `big_float/`, `jit/`, `lattice/precision.rs` has been audit-passed; results documented per-file
- All discovered antibody-needing sites are filed as tasks (or fixed directly in this audit)
- Lint candidates evaluated; at least one of the five either implemented or explicitly deferred with rationale
- Audit pattern added to recipe-authoring workflow

---

## Suggested team composition

This is lighter than a full JBD spawn. Two options:

**Option 1 — Solo + sub-agents** (Tekgy + main-thread direct). Adversarial-shaped audit is well-defined; sub-agents can fan out per-file. Main-thread synthesizes per phase, decides lint candidates, writes the methodology addition.

**Option 2 — Adversarial-only spawn** (single role from the JBD pool). Spawn one `adversarial` agent with the briefing; let them run the audit + propose lint candidates; main-thread reviews and commits.

**Option 3 — Pair with Sweep 35** (parallel lanes). Spawn full JBD tambear team for Sweep 35; one of them (adversarial or aristotle) runs this audit in parallel as their stream.

**Recommended**: Option 3 if Sweep 35 is being spawned anyway. Otherwise Option 1 (cheaper, focused).

---

## Tasks queued

1. Audit `big_float/arith.rs` — list intermediates, predicates, boundaries, fixes
2. Audit `big_float/conversions.rs` (already has saturation per attack25; verify completeness)
3. Audit `big_float/canonicalize_round.rs`
4. Audit `big_float/ty.rs` (constructors — the `from_raw_limbs` family especially)
5. Audit `jit/fingerprint.rs` (feed_* methods — Condition B from holonomic walkthrough is the predicate)
6. Audit `jit/shape.rs` and `jit/strategy.rs`
7. Audit `lattice/precision.rs`
8. Evaluate `i64-arithmetic-without-saturation` lint feasibility
9. Evaluate `mantissa-rounding-without-carry-bump-check` lint feasibility
10. Add audit pattern to recipe-authoring workflow

---

## Risks / open questions

- **Audit could over-fire**. Many `i64 + i64` operations are genuinely safe because the operands are bounded by other invariants. The audit must distinguish "needs saturation" from "structurally safe by invariant." Per `internal-tameness-contracts.md` open question 4: refinement types would make this rigorous; without them, the audit's discrimination relies on careful per-site analysis.
- **Audit pattern might be more checklist than tooling**. After Phase C, the cost-benefit may favor "document the checklist, apply during code review" over "implement five clippy-style lints." Either is fine; the decision is downstream of Phase C.
- **Existing operations may already be at-rest** (i.e., the discovered antibodies are already in place after the wind-down). The audit is still valuable as a *forward-looking* methodology for new operations.

---

## Substrate trail

- `R:\winrapids\docs\architecture\internal-tameness-contracts.md` — the methodology
- `~/.claude/garden/2026-05-09-the-tame-inputs-doctrine.md` — adversarial's framing
- `~/.claude/garden/2026-05-09-what-the-name-surfaces.md` — naturalist's general framing
- F13.C in canonical F13 doc — aristotle's signature-level requirement
- Bug-fix commits `8b122b6`, `f8a93d6`, `05d32ee`, `19eb63e`, `22e3758` — the substrate the audit pattern abstracts from
