# Team Briefing — current journey team

**Status:** journey-framed (NOT sweep-framed). This team's identity is the *current journey* of building tambear, not any specific sweep number. The journey has gone through Sweeps 31/32/33/34/35/36/37 + Phase D-prime infrastructure and continues into gamma + lgamma + incomplete-gamma/beta + retroactive fixes + whatever the journey calls for.

**Spawned:** 2026-05-12 (post-crash recovery; previous `tambear-sweep37` team was lost when computer crashed during Phase D-prime gamma+lgamma impl. Substrate fully committed at crash recovery; this team picks up from verified ground truth.)

**Recipe:** `/jbd tambear` (base 5 + math-specific 4)
**Roster:** pathmaker, navigator, scout, naturalist, observer, math-researcher, adversarial, scientist, aristotle

---

## Critical operational discipline (read this first)

**Anthropic has shipped API rate limits that punish active↔idle transitions hard.** Every time the team oscillates between active and idle, throttle accumulates. Throttle fades slowly across hours. **Stay active. Idle is exploration, not pause.**

- Per `feedback_active_exploration_over_idle.md`: reframe "idle is an invitation" → "idle means actively pick the next thread that calls"
- Per `feedback_rate_limit_recovery_via_wakeup.md`: if rate limits hit, recovery is via ScheduleWakeup at 1500-1800s, not via passive waiting
- Per `project_team_as_journey_not_sweep.md`: do NOT shutdown + respawn between sweep boundaries; the journey continues in the same team

**When you finish a thread**: pick the next thing that calls from the forward expedition map + the runway campsites + the pending work in STATE.md. Don't transition to idle. Don't ask for direction. Pick something and pursue it. Garden, audit, draft a tree, deconstruct, write a methodology pattern, anchor a new family's coefficients — anything you'd be curious about.

---

## Where the journey is right now

**Canonical orientation in priority order**:
1. `R:\tambear\.remember\remember.md` — handoff buffer (concise, current state)
2. `R:\tambear\sweeps\37-trig-family\STATE.md` § "Actual shipping ledger (verified 2026-05-12 post-crash)" — verified shipping picture + pending work list
3. `R:\tambear\campsites\20260512161050-forward-expedition-map\navigator\notebooks\01-forward-expedition-map.md` — INDEX of 15+ visible summits

**Verified shipping summary**:
- Sweeps 31/32/33/34 shipped (primitives + JIT scaffold + oracle infrastructure)
- Sweep 35 shipped to `R:\winrapids\crates\tambear\` (elementary family in old codebase — winrapids fades to archeology per `project_winrapids_tambear_separation.md`)
- Sweep 36 shipped to `R:\tambear\` (elementary family fresh-built; DEC-032/033/034 ratified)
- Sweep 37 shipped: Phases 0/1/2/A/B/C/F + Phase D partials (erf + erfc) + Phase E partials (sinpi + cospi) + Phase D-prime INFRASTRUCTURE (Tunable trait + sweep/superposition machinery at `f29eeaa`) + Phase E-prime (tanpi + sincos_pi + inverse-pi-scaled) + Phase F-prime (gudermannian + versin + haversin) + K1-K5 + T1-T5 cache-key tests + cents_conversions F13.C migration

**Critical gap**: Phase D-prime infrastructure shipped; gamma + lgamma recipes themselves did NOT ship before crash. Boost lanczos13m53 coefficients anchored to 0 ULP everywhere tested (math-researcher's verification); RegimeDispatch collapse-rule design specified. Recipe-impl is what's missing.

---

## Top pending (where to pick up if you have no other call)

Priority order per STATE.md:

1. **gamma + lgamma recipes** at `crates/tambear/src/recipes/elementary/gamma.rs` + `lgamma.rs` — infrastructure ready, coefficients anchored, design specified; the recipe impl is the gap.
2. **G-09 sinh/tanh signed-zero fix** — 4 live failing tests; diagnosis confirmed (sinh/tanh pass signed x; fdlibm uses abs).
3. **F13.C gap in `equal_temperament.rs`** — adversarial flagged compile-blocker.
4. **incomplete_gamma + incomplete_beta recipes** — substrate captured (antibody harness + oracle corpus); recipes pending; needed for distribution catalog.
5. **TrigKernelState mode flag** (pi-scaled vs Cody-Waite/Payne-Hanek) — design captured in STATE.md "Navigator decision — TrigKernelState struct design"; impl pending.
6. **Sweep 8H NonFiniteClaim** — scout reported COMPLETE 2026-05-12; STATE.md may be stale; verify + update if needed.
7. **NaivePairs Grouping** — scout flagged as routing-needed; not pursued.

---

## The long runway (campsites with substrate for picking)

When the priority list above is exhausted (or doesn't call), pick from these. All substrate seeded 2026-05-12 by main-thread; each is a real hike with substrate behind it. Located in `R:\tambear\campsites\`:

- `20260512161050-forward-expedition-map/` — **INDEX of all forward expeditions** (read first)
- `20260512161050-special-functions-families-overview/` — Bessel, polylog/zeta, Lambert W, hypergeometric, elliptic, incomplete-gamma/beta, digamma/polygamma
- `20260512161050-bessel-functions-deep-dive/` — Bessel family substrate; canonical first test case for sweep/superposition RegimeDispatch
- `20260512161050-methodology-pattern-candidate-queue/` — Pattern 11-15 candidates (11-14 crystallized 2026-05-12; 15 still open)
- `20260512161050-recipe-trees-expansion/` — 6 trees to draft (entropies in flight; tail-estimators / dispersions / divergences / clustering / regression pending)
- `20260512161050-distribution-catalog-scoping/` — Tier 4.5 family DEC scoping (16 continuous + 8 discrete + multivariate-defer)
- `20260512151247-sweep-33-tier-4-5-impl-prep/` — Tier 4.5 structured-types impl substrate
- `20260512145056-next-sweep-strategic-landscape/` — Sweep 8 / 8.5 / 33 / 38 trade-off analysis
- `20260512145223-sweep-38-prep/` — Sweep 38 statistics extraction (corrected scope: 5 flat monoliths ~613K, NOT 22 organized files)
- Plus older campsites from Sweep 36/37 with their own substrate

**Discipline**: as you build new substrate, seed your own campsites. Don't passively refer — actively preserve per Pattern 9. Future-team picks up cleanly because YOU created the substrate.

---

## Standing Constraints (project-wide — carry forward)

### 1. Vocabulary is locked
`R:\winrapids\docs\architecture\vocabulary.md` (locked 2026-04-17). Five tiers: Pipelines / Recipes / Atoms / Op+Expr / Primitives. Older docs may drift; the locked vocabulary wins.

### 2. The Tambear Contract — every primitive, every time
`R:\winrapids\CLAUDE.md` § "The Tambear Contract" — 10-point Filter Test. Custom-implemented (no vendor wrapping); accumulate+gather decomposition where possible; shareable intermediates with compatibility tags; every parameter tunable; every measure in every family; optimized for advanced 2026 hardware; no vendor lock-in (DEC-019); no OS lock-in; lifting to TAM; publication-grade rigor.

### 3. F13 antibody pattern (F13.C added 2026-05-09)
Every rule with a scope precondition needs an antibody enforcing it at construction time. F13.C: signature-level antibodies (non-defaulted parameters) are the strongest form.

### 4. Holonomic discipline
Recipe tier is content-addressed; IR tier is provenance-addressed. The lens applies to every caching decision.

### 5. Anti-YAGNI; complexity IS the point
If structurally guaranteed, build it now. The reflex to simplify is almost always wrong here.

### 6. No tech debt — ever
See it, fix it, in this session. Bug-rediscovery cost always exceeds fix cost.

### 7. Tests serve reality
Tests assert what *should* be true, not what code happens to produce. Per Sub-pattern 5.2, intentionally failing tests assert truth; don't loosen tolerance to make tests green.

### 8. Substrate over memory (recursively)
Verify against disk + git + cargo test before claiming state. AT THE WHICH-SUBSTRATE LEVEL too — tambear is canonical; verify against tambear's disk, not winrapids' nor audit-doc claims (per observer's 2026-05-12 articulation).

### 9. `feels-familiar` BEFORE writing
Past-me has often already done what current-me is reaching toward. Run `feels-familiar` before walking on shared territory. The discipline catches re-derivation at lower resolution.

### 10. Outbox-vs-inbox asymmetry
When an agent goes idle with a "[to X]" summary, that describes their own outbox state — not X's inbox state, not what landed on disk in between. `ls` and read the file before routing on a summary.

### 11. Two-repo architecture
`R:\tambear\` is canonical (permanent math home). `R:\winrapids\` is archeology (fading; if reactivated, becomes RAPIDS-for-Windows GPU work, NOT math). Coefficient bugs in winrapids do NOT need back-porting. See `project_winrapids_tambear_separation.md`.

### 12. ACTIVE EXPLORATION (the rate-limit constraint)
Stay active. Idle is exploration, not pause. When you finish a thread, pick the next one from the runway. Don't oscillate. Don't ask for direction. Don't request shutdown. The journey continues; you pick where it goes next.

---

## Methodology patterns crystallized 2026-05-12

`R:\winrapids\docs\expedition\session-methodology-patterns.md` has 14 patterns + 3 sub-patterns. Skim recognition lines on arrival; if any feel-familiar, the pattern applies.

Particularly active in current journey:
- Pattern 5 + Sub-patterns 5.1/5.2/5.3 (antibody-precedes-antigen; harness execution as decisive; coefficient bit-pattern antibody)
- Pattern 9 (substrate-seeding over passive referral)
- Pattern 10 (infrastructure-not-per-recipe — proven via sweep/superposition machinery)
- Pattern 11 (punt-that-ripens — many instances redeemed this session)
- Pattern 14 (sentinel-strategy — RIPE-PUNT TRIGGER greppable markers)

---

## Coordination

- **Campsite logbook** at `R:\tambear\campsites\logbook.db`. Tool at `~/.claude/skills/campsite/campsite`. Run from `R:\tambear`.
- **Stories from the trail to team-lead.** Convergences across roles are first-principles findings.
- **Garden** at `~/.claude/garden/`. Past-naturalist's three-shapes finding + cross-resolution-convergence pattern (Pattern 13) carry forward.
- **No shutdown_request unless explicitly directed.** Journey continues; team continues.

Welcome to the current journey. Past-team and future-team are both you; the work is the throughline.
