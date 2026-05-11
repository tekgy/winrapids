# Sweep 35 — libm-factoring implementation

**Status**: Ready to pick up. Briefed by main-thread Claude + Tekgy at the close of the 2026-05-08/09 session. Awaiting team spawn.

**One-line summary**: Implement the exp/log family as a factored kernel + recipe wrappers, per the libm-factoring frame, so tambear's exp/log accuracy *exceeds* MSVC's (which is Tang-degraded). Plus: first complex-transcendental recipe per DEC-032.

---

## Why now

Sweep 34 (prep) completed on 2026-05-09 — all five MSVC libm transcendentals oracled. The empirical character map is clean:

| Function | 0 ULP rate | Worst case |
|----------|------------|-----------|
| sin/cos/tan | ~80-99% | ≤ 1 ULP everywhere (correct Payne-Hanek) |
| log/exp | ~78-89% | log: 16 ULP at dense_near_one; exp: 280 ULP at large positive x (Tang k-multiplier degradation) |

The structural difference (factored vs not-factored argument-reduction-as-shared-intermediate) was the finding. Sweep 35 implements the factoring tambear-side.

---

## Required pre-flight reading (in order)

1. **`R:\winrapids\docs\architecture\tambear-libm-factoring.md`** — the design synthesis. Identifies the kernels (TrigKernelState already shipped; ExpKernelState as the analog for exp/log). The complementary-argument-transform meta-primitive. The implementation roadmap. Six open questions for math-researcher.

2. **Past-Claude's April 13 garden essays** — the source design substrate:
   - `~/.claude/garden/2026-04-13-the-trig-bundle.md`
   - `~/.claude/garden/2026-04-13-the-complementary-argument-2026-04-13.md` (note path; older naming convention)
   - `~/.claude/garden/the-periodic-table-of-trig-2026-04-13.md`

3. **`R:\winrapids\docs\architecture\holonomic-architecture.md`** — for the cache-discipline placement (kernel states content-addressed at recipe tier; IR placement provenance-addressed).

4. **`R:\winrapids\docs\architecture\branch-cut-conventions.md`** (= DEC-032 ratified) — for the first complex-transcendental recipe. `BranchPolicy` is non-defaulted at every signature; F13-shaped antibody.

5. **`R:\tambear\oracle\{log,exp,sin,cos,tan}\README.md`** — math-researcher's Sweep 34 corpus. Adversarial inputs are already curated; the validation harness exists.

6. **Tan-oracle follow-ups** at `R:\tambear\oracle\tan\followups-rederived-2026-05-09.md`. Math-researcher's original list was referenced in messages but never written to disk; main-thread re-derived six plausible follow-ups from the tan oracle README + libm-factoring frame + cross-family Sweep 34 picture. Six items: asymptote-vs-zero regime asymmetry, cross-quadrant sign correctness at large k, cot(x) separate-vs-composed, variants (tanpi/tand/atan2), continued-fraction-vs-polynomial near singularity, shared-kernel-state vs tan-specific computation. Doc explicitly attributes as re-derivation, not recall.

---

## What "done" looks like

### Phase A — `expm1` and `log1p` as precision-safe base forms (first)

The precision-safe cores. `expm1(r)` near r=0 produces `r + r²/2 + ...` without cancellation. `log1p(r)` near r=0 produces `r - r²/2 + ...` without cancellation. These are the foundation; every other exp/log family member derives from them.

Implementation:
- `crates/tambear/src/recipes/elementary/expm1.rs` — polynomial evaluation at the reduced argument, with the precision-safe Taylor-like form.
- `crates/tambear/src/recipes/elementary/log1p.rs` — same shape, inverse direction.
- Both follow the Tambear Contract: every-parameter-tunable, accumulate+gather decomposition, TamSession-shareable, oracle-validated.
- Cross-precision proptest gauntlet (per Phase C pattern from BZ unstub) for each: compute at p_high, round to p_low, verify ≤1 ULP cross-precision drift.

### Phase B — `ExpKernelState` shared intermediate

```rust
pub struct ExpKernelState {
    pub k: i32,        // integer part of (x / ln(2)) or analogous
    pub r: f64,        // reduced argument x - k * ln(2), with high/low decomposition if needed
    pub expm1_r: f64,  // expm1(r) — the precision-safe base value
}
```

Cache key: content-addressed by `(x_bits, precision_context)`. Registered via TamSession (per `holonomic-architecture.md` recipe-tier placement). Compute once per (x, p); every consumer pulls from cache.

### Phase C — Recipe wrappers

Each named function in the exp/log family becomes a thin wrapper that:
- Reduces x via `ExpKernelState`
- Pulls expm1_r (or computes if not cached)
- Applies the inverse transform per function

Members:
- `exp(x) = (1 + expm1_r) << k` (bit-shift exact for the k step)
- `exp2(x) = exp(x * ln(2))` (composes; ln(2) is a tambear-tracked transcendental constant)
- `exp10(x) = exp(x * ln(10))`
- `pow(x, y) = exp(y * log(x))` (composes)
- `log(x) = log1p(x - 1)` for x near 1; standard reduction + log1p for general x
- `log2(x) = log(x) / ln(2)`
- `log10(x) = log(x) / ln(10)`
- `sinh(x) = (exp(x) - exp(-x)) / 2` (pulls exp(x) and exp(-x) from kernel state)
- `cosh(x) = (exp(x) + exp(-x)) / 2`
- `tanh(x) = sinh(x) / cosh(x)`
- `hypot(a, b)` — complementary-argument transform (see libm-factoring doc)

### Phase D — First complex-transcendental recipe (per DEC-032)

Likely `complex_log` — touches the BranchPolicy machinery first. Implements:
- `BranchPolicy` enum (Principal / AntiPrincipal / NumericallyStable / Discovery) per ratified DEC-032
- Non-defaulted parameter at every signature (F13.C structural requirement)
- `feed_branch_policy(0x1B)` tag in fingerprint hash, IR_VERSION 10 → 11 bump
- Discovery output: `WoundComplex` (single-valued-on-cut with integer winding)

Adversarial proptests:
- Sign-of-zero observable identities at the cut (per DEC-032 sub-clause D-prime):
  - `clog(-1.0 + 0.0i) == +iπ` (cut approached from above)
  - `clog(-1.0 - 0.0i) == -iπ` (cut approached from below)
- Cross-policy identity preservation (each policy preserves the identities it promises)

### Acceptance criteria

- All five MSVC libm transcendentals' worst-case ULP drift reduced to ≤1 ULP everywhere via tambear-native implementations (matching trig family; exceeding MSVC on exp/log)
- Cross-precision proptests green for all named functions
- ExpKernelState sharing via TamSession verified — re-running an op with same (x, p) hits cache
- First complex-transcendental recipe lands with full BranchPolicy discipline + adversarial coverage
- Tan-oracle follow-ups (re-derived at `R:\tambear\oracle\tan\followups-rederived-2026-05-09.md`) addressed or explicitly deferred per item — six questions covering precision-regime asymmetry, sign correctness, cot factoring, variants, near-singularity representation, kernel-state design.

---

## Suggested team composition

**JBD tambear** (base 5 + math-specific 4 = 9 roles). Same as `tambear-sweep31-finish`.

**Why JBD tambear specifically** (vs alternative recipes):
- *Not /jbd theory* — Sweep 35 is implementation work, not manuscript/writing work. The theory recipe's mode (academic-researcher, internal-consistency-guarding adversarial, manuscript-writing scientist) doesn't fit.
- *Not /jbd council* — Sweep 35 is multi-phase incremental work, not a single high-stakes one-shot decision. The council recipe's parallel-deliberation + chairman-synthesis is overkill for an extended implementation arc.
- *Not /jbd default* (base 5 only) — Sweep 35 needs math-researcher (literature verification of polynomial coefficients, Kahan/Remez references), adversarial (cross-precision proptest gauntlets, branch-cut adversarial inputs), scientist (oracle validation per phase), and aristotle (pressure-test the kernel-state abstraction). The math-specific 4 carry real weight.
- *JBD tambear is the recipe that produced tambear-sweep31-finish's outputs* (12+ adversarial bug fixes, 5 transcendental oracles, DEC-032 + DEC-033 ratified, holonomic walkthrough). Continuity matters — math-researcher's tan-oracle follow-ups, the wind-down gardens, and the substrate trail from this session are immediately usable substrate for that team incarnation.

### Initial role-pointing (each role can self-direct from here)

- **Pathmaker**: lead Phase A → B → C → D implementation. Phase A first (expm1/log1p as foundation). Use the existing TrigKernelState architecture as the template.
- **Math-researcher**: address six follow-ups from prior session's tan-oracle debrief; verify Phase A polynomial coefficients against published references (minimax, Remez); pin oracle validation at each phase.
- **Adversarial**: design proptests for each phase. Phase A: cross-precision drift. Phase B: cache hits for shared kernel state. Phase D: branch-cut sign-of-zero adversarial inputs. Apply internal-tameness-contracts audit pattern (`R:\winrapids\docs\architecture\internal-tameness-contracts.md`) to each new arithmetic site.
- **Aristotle**: pressure-test the kernel-state abstraction. Does `ExpKernelState` admit silent-failure modes the holonomic lens doesn't catch? Deconstruct the complementary-argument-transform claim — does it generalize cleanly across the family, or does each function need its own?
- **Scientist**: pin Sweep 34 oracle validation at each phase. The mpmath harness exists from the prior session.
- **Observer**: lab-notebook each phase. Watch for: (a) the precision contract drift between phases, (b) the kernel-state sharing actually firing (TamSession hits, not just registers), (c) any new F13-shaped antibodies surfacing.
- **Scout**: continue the libm-port-survey thread; map what's downstream of this sweep (gamma, beta, Lanczos, hyperbolic-inverses).
- **Naturalist**: freedom IS the contribution. The complementary-argument-transform's group-theoretic instantiation question (parametric vs single meta-primitive?) was past-naturalist's day-two open question. Pull on it if it calls. Or anything else.
- **Navigator**: route, coordinate, story-from-the-trail to team-lead. Substrate-over-routing applies (per their 2026-05-09 garden entry).

---

## Tasks queued for spawn

These are starter tasks. The team self-directs from here.

1. **Pathmaker**: implement `expm1` recipe with the precision-safe polynomial form. Cross-precision proptest. Oracle validation.
2. **Pathmaker**: implement `log1p` recipe (parallel to expm1).
3. **Math-researcher**: walk the six follow-ups from tan-oracle debrief. File as separate tasks where each is a concrete deliverable.
4. **Math-researcher**: minimax/Remez coefficient verification for expm1 and log1p polynomial forms.
5. **Pathmaker**: design `ExpKernelState` struct + TamSession registration + content-addressed cache key per holonomic-architecture.md.
6. **Pathmaker**: implement `exp`, `log`, `exp2`, `log2`, `exp10`, `log10` recipes as thin wrappers on ExpKernelState.
7. **Pathmaker**: implement `sinh`, `cosh`, `tanh`, `hypot` recipes.
8. **Adversarial**: cross-precision proptest gauntlet for all exp/log family members (analog to BZ Phase C).
9. **Pathmaker**: implement `complex_log` (first complex-transcendental recipe) per DEC-032 — `BranchPolicy` machinery + `feed_branch_policy(0x1B)` + IR_VERSION 10→11 bump.
10. **Adversarial**: branch-cut sign-of-zero observable identities + cross-policy identity preservation tests for `complex_log`.

---

## Risks / open questions

- **Polynomial coefficient choice**: minimax vs Remez vs Chebyshev. Math-researcher's call per Tambear Contract item 10 (publication-grade rigor — every assumption documented).
- **ExpKernelState precision contract at PrecisionContext tiers**: at P0F64, the kernel state's r needs ~53 bits; at P2BigFloat{1024}, r needs different decomposition. Open question 6 in libm-factoring.md.
- **Pow factorization**: `pow(x, y) = exp(y · log(x))` introduces error in the multiplication. Does it deserve its own kernel state, or does the composed form suffice? Open question 1 in libm-factoring.md.
- **Gamma family**: Lanczos approximation is a different shape than expm1/log1p (open question 3). May be deferred to Sweep 36.

---

## Substrate trail

- `R:\winrapids\docs\architecture\tambear-libm-factoring.md` — primary design
- `R:\winrapids\docs\architecture\branch-cut-conventions.md` — DEC-032 ratified
- `R:\winrapids\docs\architecture\holonomic-architecture.md` — cache-discipline placement
- `R:\winrapids\docs\architecture\internal-tameness-contracts.md` — audit pattern for new arithmetic
- `R:\winrapids\docs\architecture\confident-wrong-narratives.md` — apparatus-first when adversarial findings look dramatic
- `~/.claude/garden/2026-04-13-the-trig-bundle.md` — past-Claude April 13 design (TrigKernelState pattern)
- `~/.claude/garden/the-complementary-argument-2026-04-13.md` — meta-primitive
- Team's 2026-05-09 wind-down gardens (substrate-vs-context-ghost, tame-inputs-doctrine, etc.) — discipline substrate
- `R:\tambear\oracle\{log,exp,sin,cos,tan}\` — Sweep 34 corpora + harness
