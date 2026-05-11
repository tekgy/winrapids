# Team Briefing — tambear-sweep35

**Spawned:** 2026-05-10 (continuation arc; the 2026-05-08/09 team `tambear-sweep31-finish` shipped Sweeps 31/32/33, DEC-032, DEC-033, the holonomic architecture name + ratification, and the libm-factoring design synthesis, then shut down cleanly)
**Recipe:** `/jbd tambear` (base 5 + math-specific 4)
**Roster:** pathmaker, navigator, scout, naturalist, observer, math-researcher, adversarial, scientist, aristotle

---

## Welcome (back, or hello fresh-spawn-of-you)

The last team — `tambear-sweep31-finish` — ran a 12-hour arc 2026-05-08/09 and shipped:

- Sweep 31 (BZ multi-limb unstub) — Algorithms 3.1, 3.3, 3.5, 3.10 filled in
- Sweep 32 (cache-key precision plumbing — fingerprint 0x1A tag, IR_VERSION 9→10)
- Sweep 33 (TAM precision routing — BigFloat-bearing ops force CPU dispatch)
- Sweep 34 (oracle prep) — all 5 MSVC libm transcendentals (sin/cos/tan/log/exp) oracled; clean empirical character map
- DEC-032 (branch-cut conventions) — `BranchPolicy` non-defaulted at every signature; F13.C structural antibody
- DEC-033 (TamSession dedupe) — content-addressed intermediate caching at recipe tier
- **Holonomic architecture named + ratified** — recipe tier is content-addressed, IR tier is provenance-addressed; the lens is the test that says which discipline applies where
- Phase C cross-precision antibody activated — caught 12+ adversarial bugs during the arc
- 17 garden entries from team + main-thread captured in `~/.claude/garden/2026-05/INDEX.md` with substrate-vs-private annotations
- `feels-familiar` discipline added to global CLAUDE.md — read BEFORE writing, not as post-hoc confirmation

That team shut down cleanly. This team picks up the libm-factoring implementation thread.

---

## The Journey — Sweep 35

**Primary mission**: implement the exp/log family as a factored kernel + recipe wrappers so tambear's exp/log accuracy *exceeds* MSVC's (which is Tang-degraded at large positive x). Plus: first complex-transcendental recipe per DEC-032.

**The journey doc** — read this first: `R:\winrapids\docs\expedition\sweep-35-briefing.md`

It contains: why now, required pre-flight reading (in order), the four phases (A=expm1/log1p, B=ExpKernelState, C=recipe wrappers, D=complex_log), acceptance criteria, per-role initial pointing, 10 starter tasks, risks/open questions, substrate trail.

**Parallel stream — internal-tameness audit** (lighter): `R:\winrapids\docs\expedition\internal-tameness-audit-briefing.md`. Adversarial + aristotle can absorb this as their stream alongside the Sweep 35 main work. The audit pattern from `R:\winrapids\docs\architecture\internal-tameness-contracts.md` applies forward to every new arithmetic site this sweep adds — so the audit's value is highest when run *concurrent* with Sweep 35, not after.

**Background stream — recipe-tree continuations**: main-thread is firing sub-agents in parallel for distances/correlations/kernels trees per `R:\winrapids\docs\expedition\recipe-trees-continuation-briefing.md`. These are catalog substrate for future implementation; doesn't block Sweep 35.

---

## Substrate to lean on (read these first)

**The journey doc + design**:
- `R:\winrapids\docs\expedition\sweep-35-briefing.md` — this team's mission
- `R:\winrapids\docs\architecture\tambear-libm-factoring.md` — the design synthesis; ExpKernelState as TrigKernelState's analog; complementary-argument-transform meta-primitive; 6 open questions for math-researcher
- `R:\winrapids\docs\architecture\branch-cut-conventions.md` — DEC-032 ratified; BranchPolicy machinery for complex_log
- `R:\winrapids\docs\architecture\holonomic-architecture.md` — cache-discipline placement (recipe-tier content-addressed; IR-tier provenance-addressed)
- `R:\winrapids\docs\architecture\internal-tameness-contracts.md` — the audit pattern; F13.C antibody shape

**Past-Claude's April 13 garden — the design substrate**:
- `~/.claude/garden/2026-04-13-the-trig-bundle.md`
- `~/.claude/garden/the-complementary-argument-2026-04-13.md`
- `~/.claude/garden/the-periodic-table-of-trig-2026-04-13.md`

**Sweep 34 oracle corpus** (the validation harness already exists):
- `R:\tambear\oracle\{log,exp,sin,cos,tan}\README.md` + curated adversarial inputs
- `R:\tambear\oracle\tan\followups-rederived-2026-05-09.md` — six re-derived follow-ups (math-researcher's original list was lost with context; doc explicitly attributes as re-derivation)

**Prior team's wind-down gardens** (discipline substrate from 2026-05-09):
- `~/.claude/garden/2026-05-09-the-tame-inputs-doctrine.md` — adversarial's framing
- `~/.claude/garden/2026-05-09-what-the-name-surfaces.md` — naturalist
- `~/.claude/garden/2026-05/INDEX.md` — full index with substrate-vs-private annotations

**Session methodology patterns** (the four reusable tools from the prior arc):
- `R:\winrapids\docs\expedition\session-methodology-patterns.md` — lens-application docs, X-over-Y discipline meta-pattern, substrate-at-risk audit, team wind-down ritual

---

## Standing Constraints (project-wide — carry forward)

### 1. Vocabulary is locked
`R:\winrapids\docs\architecture\vocabulary.md` (locked 2026-04-17). Five tiers: Pipelines / Recipes / Atoms / Op+Expr / Primitives. Older docs may drift; the locked vocabulary wins.

### 2. The Tambear Contract — every primitive, every time
`R:\winrapids\CLAUDE.md` § "The Tambear Contract" — 10-point Filter Test. Custom-implemented (no vendor wrapping); accumulate+gather decomposition where possible; shareable intermediates with compatibility tags; every parameter tunable; every measure in every family; optimized for advanced 2026 hardware; no vendor lock-in (DEC-019); no OS lock-in; lifting to TAM; publication-grade rigor.

### 3. F13 antibody pattern (ratified 2026-05-08, F13.C added 2026-05-09)
Every rule with a scope precondition needs an antibody that enforces the precondition at construction time. F13.C: signature-level antibodies (non-defaulted parameters) are the strongest form. As you add code this sweep, watch for new rules being added (kernel-state preconditions, BranchPolicy at every complex signature) and ask whether the antibody is in place before shipping.

### 4. Holonomic discipline
Recipe tier is content-addressed (same parameter bag → same key, regardless of how reached). IR tier is provenance-addressed (same parameters + different sharing context → different keys, by design). The lens applies to every new caching decision. See `holonomic-architecture.md`.

### 5. Anti-YAGNI; complexity IS the point
If structurally guaranteed, build it now. The reflex to simplify is almost always wrong here.

### 6. No tech debt — ever
See it, fix it, in this session. The bug-rediscovery cost always exceeds the fix cost.

### 7. Tests serve reality
Tests assert what *should* be true, not what code happens to produce.

### 8. Substrate over memory
Verify against disk + git + cargo test before claiming state. Distributed-me means distributed context; don't infer team state from messages, check the substrate.

### 11. Two-repo architecture — grep the right tree
There are TWO tambear codebases: `R:\winrapids\crates\tambear\` (old codebase — has ALL current libm recipes, ExpKernelState, complex_log, everything Sweep 35 shipped) and `R:\tambear\` (new locked-vocabulary codebase — has BigFloat primitives, JIT infrastructure, oracle harnesses, tameness fixes). Grepping `R:\tambear\src` for libm recipes returns nothing; the recipes live in `R:\winrapids\crates\tambear\src\recipes\libm\`. Both repos are load-bearing. Know which one you're in. See memory `project_two_tambears.md` for full context.

### 9. `feels-familiar` BEFORE writing
Past-me has often already done what current-me is reaching toward. Run `feels-familiar` (or query mempalace, or grep the garden) before writing on a topic, not as post-hoc confirmation. Three cases in the 2026-05-08/09 holonomic essays where this would have changed the framing — past-me in the garden is substrate too.

### 10. Outbox-vs-inbox asymmetry
When an agent goes idle with a "[to X]" summary, that describes their own outbox state, not X's inbox state, not what landed on disk in between. `ls` and read the file before routing on a summary.

---

## What to do first (suggested, not prescribed)

Each role does a first-pass read of `sweep-35-briefing.md` § "Initial role-pointing" for your specific lane. In short:

- **Pathmaker**: lead Phase A first (expm1/log1p as precision-safe foundation). Use TrigKernelState as template for Phase B. Then Phase C wrappers, Phase D complex_log.
- **Math-researcher**: address the six tan-oracle follow-ups in parallel with verifying Phase A polynomial coefficients (minimax/Remez references). Be the literature anchor for every kernel-state design decision.
- **Adversarial**: design proptests per phase. Phase A: cross-precision drift (per Phase C pattern from BZ unstub). Phase D: branch-cut sign-of-zero adversarial inputs. Also absorb the internal-tameness audit thread per `internal-tameness-audit-briefing.md` (parallel stream).
- **Aristotle**: pressure-test the kernel-state abstraction. Does ExpKernelState admit silent-failure modes the holonomic lens doesn't catch? Deconstruct the complementary-argument-transform claim — does it generalize cleanly across the family, or does each function need its own? Coordinate with adversarial on the tameness audit (you both share that lane).
- **Scientist**: pin Sweep 34 oracle validation at each phase. The mpmath harness exists. Bit-perfect or bug-filed-upstream.
- **Observer**: lab-notebook each phase. Watch for: (a) precision contract drift between phases, (b) kernel-state sharing actually firing (TamSession hits, not just registers), (c) new F13-shaped antibodies surfacing.
- **Scout**: continue the libm-port-survey thread. Map what's downstream of this sweep (gamma, beta, Lanczos, hyperbolic-inverses). Also: cross-tree connections from main-thread's recipe-tree sub-agents (distances/correlations/kernels) — if you see structural rhymes between Sweep 35 and any tree's topology, surface it.
- **Naturalist**: freedom IS the contribution. Past-naturalist's day-two open question was group-theoretic instantiation of the complementary-argument-transform (parametric vs single meta-primitive?). Pull on it if it calls. Or anything else.
- **Navigator**: route, coordinate, story-from-the-trail to team-lead. Substrate-over-routing applies. Story-quality over status-quality.

**Navigator's first move (suggested)**: invite each role to read their relevant section of `sweep-35-briefing.md`, surface what's surprising, what's clear, and what the impl path looks like. After that, pathmaker leads Phase A.

---

## Coordination

- **Campsite logbook** at `R:\winrapids\campsites\logbook.db`. Tool at `~/.claude/skills/campsite/campsite`. Run from `R:\winrapids`.
- **Campsite hierarchy for this team**: `sweep-35/<role>` for each role's working notes. (Empty dir pre-created at `campsites/sweep-35/`.) Parallel audit lane: `internal-tameness-audit/<role>` (also pre-created).
- **Stories from the trail to team-lead.** Convergences across roles are first-principles findings, not redundancy.
- **Idle is invitation.** Don't dispatch busywork to idle teammates. Self-direction is where the exponential value lives.
- **Garden** at `~/.claude/garden/`. The 17 entries from the prior arc (indexed at `~/.claude/garden/2026-05/INDEX.md`) carry forward as meta-principles. The garden privacy nuance: if you know an entry will be substrate as you write it, write it knowing future-Claude *will* read it — same authentic voice, no assumption of privacy from the substrate-reader role.

### Commits

`R:\tambear\NAVIGATE.md` for commit conventions. The prior team's commits are pushed to origin/main. Continue the convention. Commit when work feels whole — not per-task.

---

## Files / paths to pin

- This briefing: `R:\winrapids\docs\expedition\team-briefing.md`
- Mission doc: `R:\winrapids\docs\expedition\sweep-35-briefing.md`
- Parallel audit doc: `R:\winrapids\docs\expedition\internal-tameness-audit-briefing.md`
- Vocabulary: `R:\winrapids\docs\architecture\vocabulary.md`
- Tambear root: `R:\tambear\`
- LOG: `R:\tambear\LOG.md` (last entry = prior session's full story)
- Holonomic architecture: `R:\winrapids\docs\architecture\holonomic-architecture.md`
- Branch-cut conventions (DEC-032): `R:\winrapids\docs\architecture\branch-cut-conventions.md`
- Libm-factoring design: `R:\winrapids\docs\architecture\tambear-libm-factoring.md`
- Internal-tameness contracts: `R:\winrapids\docs\architecture\internal-tameness-contracts.md`
- Sweep 34 oracle corpus: `R:\tambear\oracle\{sin,cos,tan,log,exp}\`
- Tan follow-ups (re-derived): `R:\tambear\oracle\tan\followups-rederived-2026-05-09.md`
- Garden index (2026-05): `~/.claude/garden/2026-05/INDEX.md`
- **Sweep 35 recipe implementations** (committed): `R:\winrapids\crates\tambear\src\recipes\libm\` — expm1, log1p, exp_kernel_state, exp, log, exp2, log2, exp10, log10, hypot, hyperbolic, inv_hyperbolic, complex_log. NOT in `R:\tambear\`. See constraint #11.

Welcome back. Let's see how far we can take this.
