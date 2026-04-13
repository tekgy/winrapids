# Build Plan — Atoms, Primitives, Recipes

**Goal**: implement the architecture in `docs/architecture/atoms-primitives-recipes.md` end-to-end, from grounding the current build through a first libm recipe, a migration pilot, and replay harness integration.

**Rule**: work through the phases in order. Do not skip ahead. Each phase depends on the previous. The value of this plan is that it exists — if we lose the thread, we come back to the checked/unchecked items.

**Relationship to Peaks**: this plan is the next chunk of concrete work under Peak 1 (IR foundation) and Peak 2 (libm Phase 1), with oracle pieces that serve Peak 4 (replay harness). Peaks 3, 5, 6, 7 are downstream of this plan.

---

## Phase A — Grounding (fast, prevents building on drift)

**Goal**: know exactly where the codebase is before we extend it.

### A1. Verify build state
- Run `cargo test --lib -p tambear` and `cargo test --lib -p tambear-fintek` fresh on `main`.
- Record the exact pass/fail counts.
- Identify any regressions from the JBD session's late commits.
- If failures exist, fix them before moving on (or document them explicitly as "known, deferred").

### A2. Read the garden entries from the session
- `~/.claude/garden/2026-04-10-*.md` — there are ~40+ entries from the JBD session.
- Read them in order, not all at once. Take notes on:
  - Which entries have material for the research notes (Phase F)
  - Which entries have architectural insights that should be in CLAUDE.md or the architecture doc
  - Which entries have actionable items that should be campsites or tasks
- This is absorption work. Not rushed. But it should happen before the crystallization phase so the notes can draw on the full corpus.

### A3. (Optional) Update spring playground with real IntermediateTag data
- The playground at `playgrounds/tamsession-spring-network.html` was built with hand-modeled phyla.
- The session wired ~13 IntermediateTag variants into TamSession.
- Extracting the actual sharing graph and re-running the simulation would tell us whether the theoretical predictions (tight clusters = real families, loose = incomplete) hold on real data.
- Defer if it slows the foundational work. Worth doing before Phase F (crystallization).

---

## Phase B — Primitives foundation (the ground floor)

**Goal**: build the complete terminal primitive layer described in the architecture doc. Everything in Phase C and beyond depends on this.

### B1. `primitives/hardware/` — IEEE 754 terminals (~20 files)
One file per primitive. Each file is 1-10 lines of Rust:

- **Arithmetic**: `fadd.rs`, `fsub.rs`, `fmul.rs`, `fdiv.rs`, `fsqrt.rs`
- **Fused**: `fmadd.rs`, `fmsub.rs`, `fnmadd.rs`, `fnmsub.rs` (all four map to `f64::mul_add` variants; LLVM emits hardware FMA)
- **Unary**: `fabs.rs`, `fneg.rs`, `fcopysign.rs`
- **Min/max**: `fmin.rs`, `fmax.rs` — **IEEE 754-2019 NaN-propagating semantics**, NOT `f64::min`/`f64::max`. This fixes the 11-bug class from the session by construction.
- **Comparison**: `fcmp_eq.rs`, `fcmp_lt.rs`, `fcmp_le.rs`, `fcmp_gt.rs`, `fcmp_ge.rs`
- **Classification**: `classify.rs` — `is_nan`, `is_inf`, `is_finite`, `signbit`
- **Rounding**: `rounding.rs` — `frint`, `fround_ties_even`, `ffloor`, `fceil`, `ftrunc`
- **Extras**: `ldexp.rs` (we noticed this was needed by `exp`), maybe `frexp.rs`

Deliverable: `primitives/hardware/mod.rs` re-exports everything. Test file per primitive with hand-verified cases (including NaN propagation tests for fmin/fmax).

### B2. `primitives/compensated/` — Tier 1 essentials (6 files)
Foundational error-free transformations. These are the vocabulary for compensated recipes.

- `two_sum.rs` — Knuth's algorithm, 6 flops, exact a+b
- `fast_two_sum.rs` — Dekker's, 3 flops, requires |a| >= |b|
- `two_product_fma.rs` — 2 flops with FMA, exact a*b
- `two_diff.rs` — = two_sum(a, -b)
- `two_square.rs` — = two_product_fma(a, a)
- `kahan_sum.rs` — compensated vector sum

Oracle tests: compare each against arbitrary-precision reference (mpmath or bigfloat) on ~1000 random inputs. Each transformation must be bit-exact.

### B3. `primitives/compensated/` — Tier 2 libm foundations (5 files)
- `neumaier_sum.rs` — improved Kahan for mixed magnitudes
- `pairwise_sum.rs` — tree reduction, O(log n) error growth
- `dot_2.rs` — Rump-Ogita-Oishi compensated dot product
- `compensated_horner.rs` — compensated polynomial evaluation (critical for all libm)
- `fma_residual.rs` — exact residual of fmadd

Oracle tests: each against mpmath on a mix of well-conditioned and cancellation-prone inputs.

### B4. `primitives/double_double/` — DoubleDouble as first-class type (~8 files)
- `type.rs` — the `DoubleDouble { hi: f64, lo: f64 }` struct + conversions
- `add.rs` — `dd_add`
- `sub.rs` — `dd_sub`
- `mul.rs` — `dd_mul` (uses `two_product_fma`)
- `div.rs` — `dd_div` (Newton iteration)
- `sqrt.rs` — `dd_sqrt`
- `recip.rs` — `dd_recip`
- `mixed.rs` — `dd_add_f64`, `dd_mul_f64`, etc.

Oracle tests: each dd_* op compared to ~106-bit precision mpmath reference.

### B5. `primitives/specialist/kulisch_accumulator.rs`
The exact accumulator. ~4000-bit integer storage representing any f64 sum exactly. Slow (~10-50x f64) but bit-exact without needing bignum.

Critical for Phase E (replay harness oracle).

### B6. `primitives/specialist/sum_k.rs`
Rump's parameterized k-fold compensation. One primitive covers Kahan (k=1), double-double-equivalent (k=2), and higher.

### B7. Constants vocabulary
- `primitives/constants.rs` — mathematical constants at multiple precisions.
- `PI_F64`, `E_F64`, `LN_2_F64`, `SQRT_2_F64`, ...
- `PI_DD`, `E_DD`, `LN_2_DD` — DoubleDouble versions for compensated recipes
- Later: `PI_MPFR`, etc. for arbitrary-precision oracle generation

### B8. Oracle test infrastructure
A shared test pattern: `assert_within_ulp!(expected, actual, tolerance_ulp)`. Uses mpmath (via Python subprocess? via Rust mpfr binding?) or bigfloat to generate reference values.

Decision needed: do we embed an arbitrary-precision library in tambear, or shell out to mpmath for test generation? Probably generate reference tables offline with mpmath, embed them as `static` arrays, compare at test time. That's the approach the existing workup files use.

---

## Phase C — First libm recipe (Peak 2 kickoff)

**Goal**: prove the architecture works end-to-end by implementing one correctly-rounded libm function.

### C1. `recipes/libm/exp.rs`
- `#[precision(correctly_rounded)]`
- Cody-Waite range reduction with high/low split of `ln(2)` for accuracy
- Minimax polynomial for `exp(r)` on `r ∈ [-ln(2)/2, ln(2)/2]`
- Compensated Horner evaluation
- Reconstruction via `ldexp(exp(r), k)`
- Oracle tests: ≤ 1 ULP across `(-745, 710)` f64 range vs mpmath 50-digit reference

This is the template for every other libm function.

### C2. `recipes/libm/log.rs`
- Uses the `exp` machinery pattern (range reduction + polynomial)
- Oracle tests vs mpmath

### C3. `recipes/libm/sin.rs` and `recipes/libm/cos.rs`
- Payne-Hanek or Cody-Waite range reduction
- Separate polynomials for sin and cos on `[-π/4, π/4]`
- Argument dispatch based on octant

### C4. `recipes/libm/erf.rs` and `recipes/libm/erfc.rs`
- **This is the one that fixes the session bug.** With `correctly_rounded` lowering from day one, the Taylor/CF boundary issue dissolves.
- Oracle tests at the critical value `x = 1.386` (where `normal_cdf(-1.96)` lives) — should be ≤ 1 ULP on first implementation.
- Compare against `workup_erfc.rs` test suite from the session.

### C5. `recipes/libm/gamma.rs` and `recipes/libm/log_gamma.rs`
- Lanczos approximation for gamma
- Stirling series for log_gamma with negative-argument reflection
- Oracle tests at poles (0, -1, -2, ...) and near them

---

## Phase D — Migration pilot

**Goal**: prove we can bring existing recipes into the new architecture without rewriting them from scratch, and produce a template for Sonnet-agent migration waves.

### D1. Decomposition trace of `pearson_r`
- Read the current `pearson_r` implementation
- Identify every inline arithmetic operation
- List every primitive call it WOULD make under the new architecture
- Write the decomposition tree as a diagram

### D2. Migrate `pearson_r`
- Move to `recipes/stats/pearson_r.rs`
- Replace inline `*`/`+`/`-` with explicit primitive calls (`fmul`, `fadd`, `fsub`)
- Replace the variance/covariance computation with compensated primitives
- Tag as `#[precision(compensated)]`
- Verify existing `workup_pearson_r.rs` tests still pass
- Add adversarial test case: large-n (1M+ points) near-constant series to demonstrate the compensated lowering's accuracy advantage

### D3. Trace `kendall_tau` and pilot a method-layer migration
- More complex than `pearson_r` because it involves sort + inversion counting
- Demonstrates migrating a recipe that composes multiple sub-recipes
- Verify the 1942-test suite still passes

### D4. Write the migration playbook
- Document the migration steps so a Sonnet-agent wave can execute them autonomously
- Reference in a new campsite for the Sonnet wave to pick up

---

## Phase E — Replay harness integration (Peak 4)

**Goal**: use the new primitive layer to make the replay harness a comprehensive verification tool.

### E1. Kulisch accumulator as oracle
- Integrate the primitive built in B5 as an oracle in the replay harness
- For any recipe whose result is a sum or dot product, compare against the Kulisch result
- Bit-exact discrepancies = bugs

### E2. Three-way comparison harness
- For each recipe, run strict / compensated / correctly_rounded lowerings
- Compare against mpmath reference
- Output: table of (input, strict_result, compensated_result, correctly_rounded_result, mpmath_reference, max_ulp_each)
- Reveals exactly which lowering is needed for which regime

### E3. Hard-cases suite
- Collect adversarial inputs from the adversarial agent's 39 bugs + workup files
- Each hard case is a (recipe, input, expected) triple with a bug-class tag
- Harness runs all recipes through all hard cases as a smoke test
- New recipes must pass the relevant subset of hard cases before shipping

### E4. Accuracy regression detection
- Checksum per recipe: `(strict_max_ulp, compensated_max_ulp, correctly_rounded_max_ulp)` on a fixed test set
- Store in `hard-cases/checksums.json`
- CI rejects any PR that regresses a checksum without explicit justification

---

## Phase F — Crystallization

**Goal**: turn the JBD session's theoretical discoveries into durable research artifacts.

### F1. Classification bijection / functor — research note
- Draw from Aristotle + Naturalist garden entries (`methods-are-compositions.md`, `groupings-are-sharing.md`, `convergent-discovery.md`, `bijection-is-a-functor.md`)
- Structure: claim, proof sketch, three independent verifications (algebraic, topological, empirical), consequences, counterexamples, open questions
- Target: ~10-15 pages, publishable as a research note or preprint
- Location: `docs/research/papers/classification-bijection.md`

### F2. K₅/K₇ scheduling bound
- Draw from `k5-and-the-memory-bound.md`, `k7-the-real-bottleneck.md`, and the corrected K₅ analysis
- Claim: peak memory ≥ treewidth + 1 on any schedule
- Practical implication: TAM scheduler should compute treewidth before scheduling

### F3. Holographic error correction via `.discover()`
- Draw from `discover-is-error-correction.md`, `springs-and-holograms.md`
- Claim: view_agreement drops when intermediates are corrupted, providing code-distance-N error detection where N = number of views
- Testable experiment using existing primitives — design and run it

### F4. Five atomic groupings generate 15 products
- Draw from `fifteen-products.md`, `the-boundary-of-the-product-closure.md`
- Claim: {All, ByKey, Prefix, Circular, Graph} generate all 15 product groupings; 4 predicted gaps
- Consequences for scheduling (product-compositional)

### F5. Kingdom reclassification audit
- Draw from `kingdom-reclassification-audit.md`, `kingdom-b-is-rarer-than-we-thought.md`
- 9/13 Kingdom B labels were wrong — document which, why, and what parallelism opportunities exist
- This is more of a project artifact than a research paper, but it's worth a permanent writeup

---

## What this plan does NOT cover

Out of scope, to be picked up later:

- **Peak 1 `.tam` IR implementation** — the architecture doc describes lowering strategies as conceptual, but the actual compiler pass that implements them requires the IR. Phase C hand-writes the strict/compensated variants; a real lowering pass comes with Peak 1 work.
- **Peak 3 tam→PTX assembler** — GPU backend work, separate thread.
- **Peak 5 CPU backend formalization** — Rust→LLVM is "good enough" for now; formal CPU backend work is later.
- **Peak 6 deterministic reductions** — depends on the IR and scheduling infrastructure.
- **Peak 7 SPIR-V** — depends on Peak 3 being in place.

These are referenced in the plan (especially in Phase E and the libm recipes) but their full implementation is outside this plan's scope.

---

## Sequencing rules

1. Do phases in order. A → B → C → D → E → F.
2. Within a phase, items are roughly ordered by dependency. Some items in the same phase can parallelize (e.g., B1 and B2 are independent), others cannot (B8 depends on B1-B7 being tested).
3. If a phase hits a blocker, document it and skip to the next non-blocked item. Don't stall.
4. Every item ends with a committed artifact (file, test, document) so progress is visible.
5. Phase F (crystallization) can start any time after Phase A — the research notes draw from garden entries, which exist now. But don't let F distract from B-C-D-E which have concrete buildable deliverables.

---

## Estimated scope

- Phase A: small (hours)
- Phase B: medium-large (~40-50 files, mostly short — a few days of focused work or one Sonnet-agent wave)
- Phase C: medium (5-10 recipes, each with oracle test suite)
- Phase D: small-medium (2-3 pilot migrations + playbook)
- Phase E: medium (harness infrastructure + hard-cases population)
- Phase F: large if all 5 notes get written, small if only F1 gets written now

Total: 2-4 weeks of focused work if done seriously, compressible if we launch Sonnet-agent waves for Phase B and Phase D.
