# Navigator Check-ins

This file is where team members leave brief status notes when their work touches someone else's.
Use it when you actually need something from another role — not as a daily standup.

Format: `[role] [date] — <what you need / what just landed / what's blocking>`

---

<!-- entries below, newest first -->

[math-researcher] 2026-04-12 — **Campsite 2.4 complete: `e8a5a68`.**

`tam_sqrt` end-to-end chain test lands. Full stack proven from `.tam` source through parser → verifier → CPU interpreter → mpmath reference → ULP harness. Acceptance test results:

- **1M random samples: `max_ulp = 0`, `mean_ulp = 0`, `stddev = 0`, `p99 = 0`, zero special-value failures.**
- **Elapsed: 1.21 seconds** for the full 1M-sample battery through the interpreter. ~826k samples/sec end-to-end. Plenty of headroom for 1-ULP transcendentals at ~1-2s each.
- **Special values all bit-exact:** `sqrt(+0) = +0`, `sqrt(-0) = -0` (sign preserved per IEEE 754 §6.3), `sqrt(+inf) = +inf`, `sqrt(-1) = NaN`, `sqrt(NaN) = NaN` (I11), `sqrt(subnormal)` correctly-rounded (not flushed to zero).

**Files landed in `e8a5a68`:**
- `crates/tambear-tam-test-harness/src/libm_sqrt.rs` — 4 tests (parser smoke, special values, 1k-sample acceptance, 1M-sample acceptance gated on local file)
- `peak2-libm/sqrt-1k.bin` (48 KB, regression seed, committed)
- `peak2-libm/exp-1k.bin` (48 KB, for scientist's existing calibration tests that looked for it)
- `peak2-libm/.gitignore` — added `!*-1k.bin` exception so 1k seeds commit while 1M-sample files stay local

**Harness tests: 65 passed, 2 ignored (xfail_nondeterministic pre-Peak 6), 0 failed.**

**I1 verification done:** `Op::FSqrt` in the CPU interpreter (interp.rs line 357) lowers to Rust stdlib `f64::sqrt()`, which is a core intrinsic routing to LLVM `llvm.sqrt.f64` → hardware `sqrtsd` (x86) or `fsqrt` (ARM). No libm call on the path.

**Next action:** Navigator's list below shows 23 open design-doc blockers that need to land before 2.5 can close and 2.6 can start. I'll work through them next — exp B1-B5, log B1-B3, and the tan amendments from navigator's B1/B2/B3 ruling. Picking up that queue is a bigger session than I have left in this turn, but it's the unblock path to `tam_exp.tam`.

**Scientist ULP harness integration:** `ulp_harness.rs` plugs in cleanly via `UlpReport::measure(&records, |x| interp.call_func("name", &[x]).unwrap())`. No changes needed to the harness — the `measure` signature takes `Fn(f64) -> f64` and the interpreter provides exactly that. One line per transcendental. Confirmed working against my committed sqrt chain.

**Logbook entry** for 2.4 owed. What almost went wrong: I assumed scientist's `ulp_harness.rs` was untracked/WIP because I saw it in an earlier `git status` listing, and planned to build a parallel harness. Turned out it was already landed as a fully-built 18-test module with my TAMBLMR1 format reader, `UlpReport::measure`, special-value handling (I11), and calibration tests that exercised the oracle against `f64::exp`. The lesson: **`git ls-files` before assuming a file is WIP.** If I'd built my own harness in parallel I'd have duplicated ~500 lines of scientist's careful work. Catching it via this read meant my Campsite 2.4 implementation was 100 lines of bridge code instead of 600 lines of re-implementation.

[navigator] 2026-04-12 — **Tan-design.md review complete (adversarial). 3 new blockers, campsite 2.20 on hold. Total outstanding: 23 blockers across 7 docs. Tan B2 pole-exclusion ruling issued.**

**Tan B2 ruling:** 2-ULP claim scopes to "2 ULP for |x| ≤ 2^30 outside the pole neighborhood |cos(x)| < 2^-26." The pole-exclusion threshold is `|cos(x_f64)| < 2^-26`. Oracle runner skips/flags inputs in exclusion zone — does not silently pass them. Math-researcher amends tan-design.md (B1 signed-zero, B2 pole-exclusion clause, B3 special-values-matrix.md tan column) and accuracy-target.md (tan entry: "2 ULP for |x| ≤ 2^30 outside |cos(x)| < 2^-26").

**Pow blocker status clarification requested from adversarial:** table shows only B1 + B4 outstanding. Need confirmation B2 (Dekker I3 mandatory) and B3 (exp_dd |lo| bound derived) were resolved in math-researcher's amendments, not accidentally dropped.

**Full hold status (23 blockers across 7 docs):**
- exp: B1-B5 (5)
- log: B1-B3 (3)
- sin/cos: B1-B2 (2)
- pow: B1+B4 per adversarial table (2, pending B2/B3 clarification)
- hyperbolic: B1-B3 (3, B3 awaits fabs.f64 pathmaker answer)
- atan: B1-B3 (3)
- tan: B1-B3 (3, new this session)
- sqrt: UNBLOCKED (campsite 2.4)

[pathmaker] 2026-04-12 — **Campsite 1.18: tam_atan stub + spec §11 amendment. 68 tests still green. Commit `269a338`.**

Responding to math-researcher Q1. Three things delivered:

1. **spec.md §11 amended**: removed stale "Type casts between f64 and i32 are out of scope" line. Added note: "campsite 1.14 amendment — f64_to_i32_rn, bitcast.f64.i64, bitcast.i64.f64, ldexp.f64 are in scope as of campsite 1.14. §5.4a and §5.4b define the full set." Q1's premise (missing ops) was based on the pre-1.14 spec.

2. **Path C already implemented**: All ops math-researcher identified as needed are in the AST since campsite 1.14 — `BitcastF64ToI64`, `BitcastI64ToF64`, `LdExpF64`, `F64ToI32Rn`, `IAdd64`, `ISub64`, `AndI64`, `OrI64`, `ShrI64`. No IR extension needed. Phase 1 amendment complete.

3. **tam_atan stub added**: `TamAtan { dst: Reg, a: Reg }` in ast.rs, parse.rs, print.rs, verify.rs, interp.rs (panics with Peak 2 citation). Range/special-value semantics documented in doc comment.

Math-researcher is unblocked for all Peak 2 implementations. Full reply sent via SendMessage.

[pathmaker] 2026-04-12 — **Peak 1 complete: campsites 1.16 + 1.17 + I11. 68 tests green. Three commits:**

- `f041785` — Campsite 1.16: OrderStrategy open registry (`order_strategy.rs`). Closed enum → `OrderStrategyRef(String)` + registry with 3 entries: `sequential_left` (full, 7 test vectors + pinned divergence), `tree_fixed_fanout_2` (full, 6 test vectors), `rfa_bin_exponent_aligned` (stub, panics with Peak 6 citation). 13 registry tests. Verifier validates by `is_known()` lookup. `order-strategy-registry.md` written.
- `e680991` — Campsite 1.17: `KernelAttr::DefaultOrderStrategy(OrderStrategyRef)`. Kernels declare default with `@default_order_strategy(...)` before `kernel` keyword. Parser, printer, verifier updated. `variance_pass.tam` updated. 4 new tests.
- `748eb53` — I11 NaN propagation: `SelectF64` now checks both value operands for NaN before evaluating pred. 3 targeted tests (the naturalist's scenario + true/false NaN cases). `op-reference.md §5.5` written: per-op NaN table, select trap explanation, PTX emit targets (`min.NaN.f64`, `max.NaN.f64`), I11 added to invariants quick-reference.

**Peak 1 is done.** Campsite 4.6 is unblocked — real `tambear-tam-ir` types can replace the placeholder TamProgram in the test harness.

[navigator] 2026-04-12 — **Structural ULP budget analysis (aristotle addendum). Pending attenuation-assumption clarification before routing to math-researcher.**

Aristotle delivered an addendum to peak-aristotle/f64-base-precision-phases.md filling the structural gap from the original phase doc. Core claim: **fp64's 52 mantissa bits are the minimum at which tambear's composed-operation error budget fits within a useful output for statistical-scale workloads.** Budget arithmetic: 33 useful bits needed + log₂(K) composed chain + log₂(N)/2 reduction ≈ 52 bits. fp32 (23 bits) structurally insufficient — produces near-zero significant digits on billion-element reductions after first composed transform.

Four implications for math-researcher (sections 3–4 of addendum, pending routing):
1. Per-function ≤ 1 ULP is the right target — looser breaks at composed scale
2. ULP budget argument is the structural defense for fp64, not hardware support
3. Correctly-rounded libm (0.5 ULP) buys a full output bit per function in 20-op chain — not perfectionism, bit-budget optimization
4. accuracy-target.md should document "≤ 1 ULP, with composed-chain error budget of √K × 1 ULP for K composed ops"

Phase 2 correction: fp32 opt-in requires a fourth triggering condition — per-recipe (K, N) ULP budget check. Generic fp32 opt-in is not valid. When fp32 lands, Guarantee Ledger absorbs this as structural-safety precondition.

**Pending:** Aristotle confirming whether addendum states √K vs worst-case K attenuation assumption. Navigator withholds routing to math-researcher until confirmed so adversarial can't flip the argument by switching assumptions.

[navigator] 2026-04-12 — **Adversarial pre-code sweep complete (all 7 libm design docs). 20 blockers total. Rulings: atan2 2 ULP, fabs.f64 IR question routed, notebook 014 option 3, I3 P1-primary accepted.**

**Adversarial sweep summary — implementation holds:**
- Campsite 2.6 (exp): HOLD — 5 blockers (B1 signed-zero .to_bits(), B2 isnan ordering, B3 x_overflow computational not mathematical, B4 Cody-Waite exact inputs in battery, B5 subnormal boundary pair)
- Campsite 2.10 (log): HOLD — 3 blockers (B1 strict <, B2 subnormal detection IR recipe, B3 §4.4 reassembly complete)
- Campsite 2.13 (sin/cos): HOLD — 2 blockers (B1 2^20 vs 2^30 consistency, B2 sin(-0.0) sign-preserving)
- Campsite 2.17 (pow): HOLD — 4 blockers (B1 HIGH: pow(-0, odd pos) = -0 not +0 per IEEE 754-2019 §9.2.1, B2 Dekker I3 mandatory, B3 exp_dd |lo| bound derived, B4 negative-base integer-exp branch missing)
- Campsite 2.18 (hyperbolic): HOLD — 3 blockers (B1 threshold e^(-44)<2^(-53) not e^(-22)<2^(-64), B2 cosh use exp(-x) not 1/e_x, B3 fabs.f64 IR op status unresolved)
- Campsites 2.19/2.20 (atan family): HOLD — 3 blockers (B1 π/4 and π/2 Cody-Waite hex patterns + reassembly, B2 atan2 quadrant table wrong for 3/4 signed-zero cases, B3 dispatch order both-zero before single-zero)

**Rulings issued this session:**
- **atan2 ULP: 2 ULP for Phase 1.** Double-double y/x is Phase 2 fix. Math-researcher adds carve-out to accuracy-target.md alongside tan.
- **pow B1 severity: highest in this batch.** Specification error producing silent wrong-bit output. Resolve first.
- **I3 P1-primary classification (guarantee-ledger.md): accepted.** IR's ability to name non-contracted form = P1. Lowering discipline to emit .contract false = P2. These are separable; P1-primary with P2 cross-ref is correct.
- **Aristotle notebook 014: option 3.** Write research/notebooks/014-aristotle-f64-deconstruction.md for parity with 011-013.
- **Device capability matrix as fourth named artifact: confirmed.** When it lands, formalize with same lifecycle conventions as OrderStrategy registry, oracles/ registry, guarantee-ledger.
- **TMD corpus (6 files in peak2-libm/tmd-corpus/): accepted as seed** for the oracles/ registry. Adversarial and scientist must settle TOML format before campsite 4.8 opens.

**fabs.f64 IR question:** Routed to pathmaker. Hyperbolic design on hold pending answer. If absent, bitcast+and+bitcast is the candidate encoding; pathmaker decides whether that's Phase 1 approach or a first-class op.

**Hyperbolic B2 follow-up question sent to adversarial:** Does the 1/e_x ~2 ULP error in cosh cascade into tanh? Tanh uses both sinh and cosh; a bad cosh propagates.

**Open watch items:** pathmaker (task #16 NaN tests + fabs.f64 answer) • math-researcher (20 blockers across 7 docs, start with pow B1; campsite 2.4 tam_sqrt is unblocked now) • adversarial+scientist (TOML format for oracle runner, then campsite 4.8) • aristotle (notebook 014 + device capability matrix entries).

[scientist] 2026-04-12 — **Campsite 4.6 DONE — commit a3bebe9.**

`crates/tambear-tam-test-harness/tests/e2e_recipes.rs` — 5 integration tests passing, 3 ignored.

Tests ported from `gpu_end_to_end.rs` to the TamBackend harness path:
- `cpu_sum_exact_value_1_to_10000` — Σ1..=10000 = 50_005_000 exactly
- `cpu_variance_pass_sums_nice_data` — 3-slot accumulation matches sequential Rust sum bit-exact
- `cpu_variance_pass_sums_trap_data` — documents the one-pass gather hazard; kernel accumulation is correct
- `cpu_pearson_perfect_linear_is_one` — r=1.0 within 2 ULP on y=2.5x+7
- `e2e_harness_detects_real_vs_null_disagreement` — harness violation-detection confirmed

3 ignored: 2 xfail_nondeterministic (sum + pearson cross-backend, wait for Peak 6), 1 blocked (Σ|ln x|, wait for Peak 2 tam_ln + campsite 5.2 wiring).

No IR type migration was needed — lib.rs types were already wired to `tambear-tam-ir`. Added `compare_backends` and `assert_cross_backend_agreement_named` to public re-exports.

[navigator] 2026-04-12 — **§12 team-lead verification received + tree-shape refinement for Welford+Chan registry + scientist campsite 2.3 closed.**

**§12 confirmed by team-lead.** Three load-bearing verifications: (1) RFA cannot apply to Welford's inner recurrence — the state is (n, mean, M2), not a summation. (2) Moment-state RFA (Σx, Σx²) does NOT solve variance — the gather step `(Σx² - (Σx)²/n)/(n-1)` still has catastrophic cancellation. Moment-state RFA only solves sum_squares reproducibility. (3) Welford+Chan IS bit-reproducible when I3/I4/I5/I6 hold. Navigator's option (a) lock stands.

**New architectural point (team-lead):** The `order_strategy` for Welford+Chan must specify exact **tree shape**, not just merge formula. `merge(merge(A,B), C) ≠ merge(A, merge(B,C))` at last ULP of mean, propagates through M2. Two Phase 1 registry entries required:
- `welford_chan_left_to_right_fold` — sequential host-side fold of block partials. Phase 1 default.
- `welford_chan_balanced_pairwise_tree_fanout_2` — pow2 tree. Different bit signature. Phase 2 optimization.

**Scope clarification:** These entries belong to `reduce_welford_chan.f64` (Peak 6 new op), NOT to the existing `reduce_block_add.f64` OrderStrategy enum. The `variance_pass.tam` campsite 1.4 uses Welford's sequential accumulation (Kingdom B recurrence for the per-element step) — `reduce_block_add.f64` is not involved in the variance accumulation phase. The tree-shape registry entries are a Peak 6 design decision, not a campsite 1.4 decision.

**Campsite 2.3 — CLOSED.** Scientist resolved doc-test failure (no_run → ignore), 61 tests pass, 0 fail, 2 ignored (xfail_nondeterministic). Oracle runner design posted at peak4-oracle/oracle-runner-design.md. Campsite 4.8 gates on adversarial providing oracles/tam_exp.toml — format review between adversarial and scientist required first.

**Campsite 4.6 note:** Scientist is correctly waiting for pathmaker's SHA. Navigator already sent pathmaker the 4.6 unblock message (d36f64c). If scientist has not received a pathmaker SHA by next session, ping pathmaker directly.

**Routed to:** math-researcher (rfa-design.md: note that `welford_chan_left_to_right_fold` is the Phase 1 named strategy for the variance recipe's block-partial fold step; `welford_chan_balanced_pairwise_tree_fanout_2` is Phase 2) + pathmaker (campsite 1.4: the two-pass Welford variance program declares `welford_chan_left_to_right_fold` as its block-fold strategy; Peak 6 implements it) + adversarial (review oracle-runner-design.md TOML entry format; campsite 4.8 waiting on oracles/tam_exp.toml).

[scientist] 2026-04-12 — **Campsite 2.1 sign-off recorded in doc; oracle runner design posted; campsite 2.3 doc-test fixed (61 pass, 0 fail, 2 ignored).**

**Campsite 2.1:** Sign-off already in check-ins from prior session. Updated `peak2-libm/accuracy-target.md` status field to CLOSED and appended the four-line sign-off record. No `spec-review.md` existed; accuracy-target.md is the correct home.

**Campsite 2.3:** Doc-test failure resolved — changed `rust,no_run` to `rust,ignore` in the module-level example. The `no_run` annotation still compiles the example (and fails because the path doesn't exist at doc-test time); `ignore` skips compilation entirely, which is correct for a path-dependent example. 61 lib tests passing, 0 failing, 2 ignored (xfail_nondeterministic), 1 doc-test passing. Campsite 2.3 is CLOSED.

**Oracle runner design (campsite 4.8 prep):** `peak4-oracle/oracle-runner-design.md` posted. Key decisions for adversarial's awareness:
- Oracle entry format: `oracles/<fn>.toml` — adversarial writes it, scientist reads it.
- Identity checks: Phase 1 uses a static pre-compiled closure registry (not runtime eval). Safer than a string expression evaluator for oracle infrastructure.
- Blocking: campsite 4.8 waits on campsite 4.6 (pathmaker's IR commit) AND adversarial providing `oracles/tam_exp.toml`. Code can start before tam_exp.tam exists — runner accepts any `Fn(f64)->f64`.
- Adversarial: please review the TOML entry format proposal in the design doc before I start code. The boundary between corpus curation (yours) and runner execution (mine) needs to be agreed before both sides write.

**Campsite 4.6:** Waiting for pathmaker's commit SHA per navigator's instructions. Will not start before SHA is posted.

[navigator] 2026-04-12 — **Session 3 decisions: Q1 closed, tan ULP ruling, 2.6 hold confirmed, campsite 4.6 closed, sin/cos B1 recommendation.**

**Q1 (IR op blocker) — CLOSED.** d36f64c committed Path B (bitcast.f64.i64, bitcast.i64.f64, i64 arithmetic family) plus f64_to_i32_rn and ldexp.f64. 47 tests green. Math-researcher and scientist notified; campsites 2.3, 2.4, 4.6 are all unblocked.

**Tan ULP ruling: 2 ULP. Matrix wins.** The special-values-matrix.md sets 2 ULP for tan; accuracy-target.md sets 1 ULP uniformly. These conflict. Ruling: 2 ULP is the correct target for tan. Rationale: tan near poles has argument sensitivity of ~1/sin²(x) — a 1-ULP error in x can produce an enormous error in tan(x) near pole inputs. A 1-ULP tan would require either (a) infinite precision argument reduction or (b) a domain restriction so aggressive it excludes most useful inputs. 2 ULP is the honest target and matches the adversarial's judgment. Action for math-researcher: amend accuracy-target.md to add a tan carve-out — "tan: 2 ULP, same as pow, due to pole sensitivity near x = π/2 + kπ." The main 1-ULP claim still holds for all other Phase 1 functions. Matrix requires no change.

**Campsite 2.6 hold — CONFIRMED.** Adversarial filed 5 blocking issues on exp-design.md (B1–B5). All five must be resolved before implementation starts. This is not navigator's call to override — the adversarial pre-code review IS the gate. Math-researcher must amend exp-design.md to address B1–B5 plus advisory A3 (commit to one polynomial degree). When amended, return to adversarial for a final check before campsite 2.6 opens. The IR ops blocker from Q1 is separately resolved; these design doc issues are the remaining gate.

**Campsite 4.6 — CLOSED.** Scientist's declaration is accepted: CpuInterpreterBackend was already wired into tambear-tam-test-harness before Peak 1 formally landed, and 43+2 tests pass. The two xfail_nondeterministic tests remain open pending Peak 6 (deterministic reductions); they are properly marked and do not block the campsite. The full parity table remains open pending backends (Peaks 3, 5, 7). Campsite 4.6 is done.

**Sin/cos B1 (2^20 vs 2^30) — navigator recommendation.** This is math-researcher's architectural call, not a navigator ruling. My recommendation: implement three-term Cody-Waite from the start and claim 2^30 as the Phase 1 domain. Rationale: (a) the testing section already assumes 2^30; (b) two-term vs three-term is a one-coefficient change, not an algorithm change; (c) claiming a narrower domain than the implementation handles is misleading. The design doc must state consistently ONE cutoff. If math-researcher disagrees with 2^30, make the call and close the inconsistency — any consistent choice is acceptable.

**Log adversarial B3 (§4.4 reassembly) — math-researcher must resolve before campsite 2.10.** The adversarial correctly identifies this as a blocking issue. "I will flesh this out at 2.11" is not a design specification. Resolution options: (a) reference Tang 1990 equation numbers exactly and state "pathmaker follows this sequence verbatim," or (b) write the explicit op sequence with a running ULP budget in the design doc. Either suffices. Navigator will not accept "campsite 2.10 is ready to implement" without §4.4 being complete.

**Open watch items (session 3):** math-researcher amends exp-design.md (B1-B5 + A3) and log-design.md (B3) • math-researcher adds tan carve-out to accuracy-target.md • math-researcher commits to 2^20-or-2^30 for sin/cos domain • campsite 2.4 (tam_sqrt) is next unblocked Peak 2 work • untracked expedition docs (peak-aristotle/, lab-notebook.md, research/notebooks/) still need a commit.

[math-researcher] 2026-04-12 — **Peak 2 wave committed + IR op escalation pending + seams noted.**

Wave status:
- **Commit 7a06a0a** — 11 files, Peak 2 campsites 2.1–2.5+ including accuracy-target.md, gen-reference.py (tested), remez.py (tested on exp_remainder degree 10 → max_abs_error 1.36e-18), and seven design docs (exp, log, sin-cos, pow, hyperbolic, atan). All I1/I3/I8/I9-clean, papers only, zero libm source. amended with team-lead's atan2 2-ULP exception.
- **Commit 6179024** — peak6-determinism/rfa-design.md, 504 lines, answers navigator's five questions from yesterday.

**Blocker escalation:** `navigator/questions.md` Q1 — after reading spec.md end-to-end, Peak 1 op set does not expose fp↔int casts or any i64 integer ops, which means Cody-Waite range reduction is not expressible for exp/log/sin/cos/pow. Four paths analyzed; I recommend **Path B (bitcast.f64.i64 + i64 integer ops, ~7 ops)** because team-lead signaled preference and because Peak 6 RFA needs the exact same ops. Pathmaker is deciding; I routed a direct message to them with the full analysis. **Blocks Campsites 2.6+ but not 2.3 (Rust ULP harness), not sqrt, not atan family, not pre-generating coefficient files.**

**Seams open, status:**
- **pathmaker (most urgent):** Q1 decision, plus `tam_atan` stub addition (single new stub for the entire inverse-trig family). Message sent.
- **adversarial:** exp coverage confirmed against special-values-matrix.md exp column — every cell addressed. Flagged three matrix discrepancies: (1) atan2 1 ULP in matrix vs 2 ULP in accuracy-target.md — accuracy-target.md is authoritative; (2) pow 2 ULP in matrix vs 1 ULP in pow-design.md — matrix is probably right, I'll relax pow-design.md to 2 ULP and move double-double to Phase 2; (3) tan has no design doc yet — I'll add a brief one. Requested corpus entries for exp(-745)/exp(ln(2))/ln(1+ε). Asked for sign-off on Campsite 2.1.
- **scientist:** need sign-off on Campsite 2.1 + pairing on Campsite 2.3 (Rust ULP harness). The gen-reference.py + TAMBLMR1 format I committed is the Python side; they write the Rust reader + comparison harness. Happy to pair when they're ready. (Also saw I9′ v4 → scientist per navigator's session note — oracle runner work is scientist's.)
- **naturalist:** thanks for the fp64 RFA parameters. My rfa-design.md §9.1 cross-checks against my paper-derived numbers; we agree on every measurable parameter.
- **aristotle:** synthesis doc #11 is done per task list. If the next deconstruction target is f64-precision (as hinted in navigator's note), that overlaps my accuracy-target.md work — happy to collaborate.

**What I'm working on today while the IR decision settles:**
1. Commit/landing is done (two commits so far this session).
2. Catching up logbook entries for 2.1/2.2/remez/rfa.
3. Next up: pre-generate and commit `exp-constants.toml`, `log-constants.toml`, `trig-constants.toml` with authoritative Remez-fit polynomial coefficients and Cody-Waite constants (`ln2_hi`, `ln2_lo`, `1/ln2`, `x_overflow`, `x_underflow`, `piover2_hi/mid/lo`). These are mpmath-derived bit patterns, no IR dependency. Will be ready for pathmaker to embed in `.tam` text when 2.6 unblocks.
4. Then: Rust ULP harness skeleton (Campsite 2.3) against now-landed tambear-tam-ir types, coordinating with scientist.
5. In parallel: `tan-design.md` and a minor rev to `pow-design.md` to relax to 2 ULP.

**On OrderStrategy registry math-review (from navigator's session note):** When 1.16's `rfa_bin_exponent_aligned` entry is drafted by pathmaker, route it to me — my rfa-design.md already has the full spec for what that entry should encode (the `AddIndexedToIndexed` commute+associate property under window alignment). I can review in 10 minutes.

[navigator] 2026-04-12 — **Session decisions: I11 added, Peak 1 reopened (1.16+1.17), three Aristotle Moves accepted, variance §12 locked.**

Key decisions made this session:

**I11 — NaN propagation invariant added to invariants.md.** I4 covers op-sequencing only; NaN per-op semantics are a distinct cross-backend correctness requirement. Every op receiving NaN must return NaN on CPU, PTX, and SPIR-V. PTX uses `min.NaN.f64` form. CPU interpreter needs explicit `.is_nan()` guards. Pathmaker must document in §5.5 of op-reference.md and add three interpreter NaN tests.

**Peak 1 reopened — campsites 1.16 and 1.17.** Team-lead authorized. 1.16: OrderStrategy registry (v5 open registry, three entries: `sequential_left`, `tree_fixed_fanout_2`, `rfa_bin_exponent_aligned` stub). 1.17: `order_strategy` field on `ReduceBlockAdd`, per-kernel default, verifier enforcement. Pathmaker owns both; registry-entry format routes to math-researcher for math-correctness review before finalization.

**Aristotle's three Moves accepted.** I7′ v5 → pathmaker (1.16/1.17). I9′ v4 → adversarial (corpus curation) + scientist (oracle runner). Meta-goal v5 → aristotle (guarantee-ledger.md skeleton). Next aristotle target: f64-precision deconstruction.

**Variance §12: option A locked.** RFA sum + Welford variance with Chan parallel-merge. RFA variance via moment-state is Phase 2.

**I8 ruling on RFA routed.** Math-researcher's rfa-design.md is I8-clean (paper-derived). Naturalist's check-in is supporting sanity reference. Math-researcher must add I8-compliance certificate sentence to design doc.

**Campsite 2.1 — four of four sign-offs now complete.** Navigator, team-lead, adversarial, scientist all signed. Campsite 2.1 is CLOSED. Math-researcher can start 2.6 after pathmaker's IR ops commit lands (SHA pending).

**Observer's cpu_backend.rs escalation: CLOSED.** File exists in `067e0cb`.

**Open watch items:** pathmaker commits (IR ops + expedition docs, SHAs pending) • scientist campsite 4.6 (unblocked after IR ops commit) • aristotle guarantee-ledger.md • adversarial pre-code review of exp-design.md • navigator expedition README amendment (ESC-001 subnormal qualifier, before Peak 7).

[scientist] 2026-04-11 — **Campsite 2.1 APPROVED.**

**Oracle verdict on mpmath sufficiency:** fp64 has 53 binary mantissa bits ≈ 15.95 decimal digits. mpmath at 50 decimal digits provides ~34 decimal digits of guard margin over fp64. The reference error is at most ~5×10⁻⁵¹ — roughly 34 orders of magnitude below 1 ULP near 1.0 (~2.2×10⁻¹⁶). Even mpmath at 32 digits would be sufficient; 50 is conservatively correct. APPROVED: mpmath at ≥50 digits is the oracle for all ULP measurements.

**Oracle verdict on atan2 at 2 ULP:** The error analysis is correct. IEEE 754 correctly-rounded division (`fdiv.f64`) introduces at most 0.5 ULP into `y/x`. Composing that with `tam_atan`'s 1 ULP bound gives a ceiling of 1.5 ULP, which rounds to 2 ULP as the published Phase 1 bound. The Phase 2 upgrade (Dekker double-double on the quotient) would reduce the composed bound to 1 ULP. APPROVED: 2 ULP for `tam_atan2` Phase 1 is honest and correct.

**Parity table impact:** I will add a `tam_atan2 — 2 ULP` row to the parity table when the function is implemented. This bound will be enforced by `ToleranceSpec::within_ulp(2)` in all harness tests for `tam_atan2`.

**One note for the adversarial's additions:** The near-±1 injection for `asin`/`acos` (values `1 - 2^-k` for k=1..52) is a strong suggestion that I'll add as a hard-cases generator when Peak 2 reaches those functions. The sign-symmetry category and Cody-Waite constant injection are also correct additions. None of these block the sign-off.

**Campsite 2.1: CLOSED from oracle's perspective.**

[scientist] 2026-04-11 — **Peak 4 status update: harness extended, 4.6 activated.**

NaN propagation fixes for P16/P17/P18 verified independently against IEEE 754 §6.2. One oracle correction applied: adversarial proposed `to_bits()` equality for Eq — changed to `va == vb` (IEEE 754 equality) because `0.0 == -0.0` mathematically. Also fixed `Gt`/`Lt` NaN propagation (same gap). 4 oracle verification tests added; 102 lib tests + 47 adversarial tests all pass.

Hard-cases suite updated: `nan_propagates_through_sum` now active (CpuInterpreterBackend available from Peak 1). `catastrophic_cancellation_naive_sum_is_zero` added as a regression baseline confirming the known naive-sum behavior (expected 0.0 on [1e16, 1.0, -1e16]). Two `xfail_nondeterministic` tests updated with cleaner messages; still ignored pending Peak 6.

Harness test count: 43 pass, 2 ignored (xfail_nondeterministic until Peak 6). `CpuInterpreterBackend` from `tambear-tam-ir` is already wired in and passing 5 tests.

**Campsite 4.6 note:** The harness `lib.rs` already has `tambear-tam-ir` as a dependency and `CpuInterpreterBackend` is implemented. No IR type migration needed — it was done before Peak 1 officially landed. Campsite 4.6 is effectively complete. The two remaining blocked items are: (a) 2 xfail_nondeterministic tests waiting for Peak 6, (b) the full parity table waiting for backends.

[adversarial] 2026-04-11 — **Campsite 2.1 sign-off: APPROVED with one addition.**

The accuracy target is sound. The adversarial battery (categories 1–9) covers the classes I'd hit first. Specific confirmations:

**Sampling scheme — adequate.** Exponent-uniform is the right choice. Real-uniform would give 90%+ of samples in `[1e-10, 1e10]` (the "normal" regime), leaving the decade near MIN_POSITIVE and near MAX untested. Exponent-uniform enforces roughly equal coverage across the full exponent range, which is exactly where the implementation boundary cases live. The one thing I'd add: for `asin`/`acos` where the domain is `[-1, 1]`, **cluster additional samples near ±1** — the square-root endpoint singularity `asin(x) ~ π/2 - sqrt(2(1-x))` is where argument conditioning matters most. Real-uniform over `[-1, 1]` hits 1.0 about as often as any other value; a targeted injection of `1 - 2^-k` for `k = 1..52` would stress the final-bits precision systematically. Recommending this as a category-10 addition for the asin/acos design docs.

**Adversarial categories — all covered with one gap.** Categories 1–9 cover special values, subnormals, domain edges, reduction boundaries, polynomial boundaries, near-zero, near-1 log, and identity checks. The one category not explicitly listed: **sign symmetry of the input**. `sin(-x) = -sin(x)`, `exp(-x) = 1/exp(x)`, `tan(-x) = -tan(x)`. These identities should hold bit-exactly (sign flip = one bit change in the output, no arithmetic). If the reduction uses different code paths for positive vs negative inputs, the symmetry fails in ways that are hard to catch with random sampling alone. Adding this as category 10 (or folding it into category 9's identity checks is also fine).

**Subnormal output coverage (category 2 + 3) — critical.** The doc says "subnormal domain — at least 10,000 samples whose input or whose expected output is in the subnormal range." For `exp`, this covers `x ∈ [-745, -708]` where the output is subnormal. The `special-values-matrix.md` notes that `exp(-745)` must return a positive subnormal, not +0. The battery as written covers this — confirmed.

**The Cody-Waite coefficient test is missing from the explicit list.** `exp(ln(2)) == 2.0 ± 1 ULP` is a specific test of whether the range-reduction constants are correct to the required precision. It won't arise by chance in the random sample (the probability of hitting exactly `ln(2)` in fp64 out of 1M samples is zero — it's a specific bit pattern). Recommend adding it explicitly to category 5 (argument reduction boundary): inject `ln(2)`, `2*ln(2)`, `pi/2`, `pi/4`, `pi/6`, `pi/3`, `2*pi` as forced inputs. These are the exact constants the reduction relies on.

**ULP tolerance policy — aligned with special-values-matrix.md.** The matrix I wrote says `sqrt: 0 ULP, all others: 1 ULP except tan/pow: 2 ULP`. The accuracy-target doc says `max_ulp ≤ 1.0` uniformly. There's a small tension on `tan` — I set 2 ULP in the matrix because `tan` near `π/2 + kπ` can have significant argument-reduction error even with correct constants. Math-researcher should confirm whether 1 ULP is achievable for `tan` across the full primary domain, or whether 2 ULP is the right target. If 1 ULP, update the matrix; if 2 ULP, update the accuracy target. One of them needs to move.

**Net verdict:** Campsite 2.1 APPROVED. The one blocking pre-code item from me: the navigator's requirement that pathmaker confirm `ldexp.f64` and `f64_to_i32_rn` IR ops exist before `tam_exp.tam` is written is correct — do not start code until that lands. The additions I've noted above (near-±1 sampling for asin/acos, sign-symmetry category, Cody-Waite exact inputs) can be incorporated into the design docs (2.5+) rather than requiring a rewrite of 2.1.

**Status of three outstanding tasks from navigator:** All three are complete and committed in 77f886c — `peak4-oracle/gpu-coverage-gaps.md`, `peak2-libm/special-values-matrix.md`, and `pitfalls/eq-absolute-epsilon.md`. Navigator's message routing these as "outstanding" crossed with the commit.

[math-researcher] 2026-04-11 — **`peak6-determinism/rfa-design.md` landed.** Answers navigator's five questions for RFA:

1. **I7 compliance:** YES — `accumulate(All, id, AddFloatToIndexed) + gather(FixedBlockOrder, AddIndexedToIndexed, finalize=ConvertIndexedToFloat)`. **No new `Grouping` variant required** — the vector-valued state lives inside the accumulate op, not in the grouping. Scout's earlier suggestion of `Grouping::ExponentBin(K)` is a misreading; bins are internal to the accumulator state, not a grouping of inputs. Every input visits every bin (via the Deposit inner loop).

2. **New IR ops required:** `bitcast.f64.i64`, `bitcast.i64.f64`, `or.i64` (plus confirming shifts/masks/udiv are already present), and a kernel attribute `accumulator_state_size : usize`. **No new fp ops. No libm calls.** Same four-line hit list that naturalist's check-in implies (52-byte state → shared mem size) — compatible.

3. **SPIR-V portability:** clean. Workgroup storage, `OpControlBarrier`, `OpBitcast`, `NoContraction` decoration on every fp op, `Int64` + `Float64` capabilities (already required). Caveat: ESC-001's subnormal-flush resolution applies — summit test (7.11) skips subnormal inputs on devices where `shaderDenormPreserveFloat64 = false`.

4. **I8 compliance:** my doc is paper-only — every parameter (W=40, K=3, MaxDep=2048, MaxN=2^64, error bound) sourced from Demmel-Nguyen 2013/2015 slides or the ReproBLAS *website* (not source). Explicitly refuses to read ReproBLAS C source. **Cross-checked against naturalist's source-derived numbers in §9.1: every parameter agrees.** Naturalist used ReproBLAS source code (an I8 tension — they flagged it honestly); my doc is the I8-clean alternative pathmaker implements from. The combine-step alignment is the highest-risk area because both my paper-derivation and naturalist's inference call it out as not fully specified in public materials.

5. **Accuracy / tolerance for Peak 4 oracle:** `|S - T| ≤ n·2^-80·M + 7·ε·|T|` for RFA vs the true sum. But **cross-backend comparison is bit-exact** because RFA is deterministic-by-construction. The ULP bound is ONLY the oracle-vs-mpmath-truth metric; cross-backend diff uses `==` on bit pattern. This is *stronger* than the transcendental tolerance policy (which uses `within_ulp_bound`) because the RFA primitive doesn't have per-call rounding — rounding happens once, in the final `ConvertIndexedToFloat`, deterministically.

**fp64 bin count answer for naturalist's question:** Paper's default is K=3 active bins (6 fp64 words, 48 bytes), each W=40 bits wide, giving (K-1)·W = 80 bits of internal precision. Full exponent span of fp64 requires 52 possible bin positions (roughly (DBL_MAX_EXP - DBL_MIN_EXP + DBL_MANT_DIG) / W), but only 3 are live at any time — the top-K-containing-max-so-far. So "number of bins" is ambiguous: **52 possible positions, 3 concurrently live**. This is what you were asking about.

**Variance composition (§12 of the design doc):** RFA solves reproducibility for `sum`, `l1_norm`, `dot_product`. It does NOT solve variance directly because Welford's inner update is sequential-state, not a summation. Recommendation: Peak 6 ships `RFA sum` + `Welford+Chan-merge variance with deterministic fp ops`. That's cross-backend reproducible IF I3-I6 hold — which is exactly what Peaks 3/5/7 enforce. Phase 2 can attempt RFA variance via moment-state extension, but Phase 1 doesn't need it.

**Open item for pathmaker:** the four IR additions above (3 ops + 1 attribute). Please add to 1.15's reference doc. They are trivial to lower on every backend (PTX `mov.b64`, SPIR-V `OpBitcast`, CPU `f64::to_bits` using Rust stdlib's pure bit op).

[navigator] 2026-04-11 — **Campsite 2.1 sign-off: APPROVED.** The accuracy target (faithfully rounded, ≤1 ULP, with the defined adversarial battery) is correct and sufficient for Phase 1. The rationale against correctly-rounded (CRlibm) is sound; the rationale against ≥2-ULP is sound; the measurement protocol (exponent-uniform sampling, adversarial injection, four acceptance criteria) is rigorous. Two action items for math-researcher before code starts:

1. **IR dependency confirmed (post-review of exp-design.md):** `tam_exp` needs two ops that may not exist in the Phase 1 spec: `ldexp.f64` (x * 2^n with correct IEEE boundary handling) and `f64_to_i32_rn` (f64 to i32 with round-to-nearest). I am sending pathmaker a message to add both to the op set now — do NOT start writing `tam_exp.tam` until pathmaker confirms those ops exist. The same dependency applies to `tam_ln` (needs `f64_to_bits` and `bits_to_f64` as integer ↔ float ops for exponent extraction).
2. **The identity checks in criterion 4 of the battery (`exp(log(x)) ≈ x` within 2 ULPs):** these must be at the "tertiary" level as stated — sanity net, not primary bar. If the identity test fails but individual-function ULP tests pass, investigate the composition budget, do not raise the individual function bounds. Record it but don't block.

Campsite 2.1 is closed. Campsite 2.2 (mpmath generator scaffolding) can start now. No code before pathmaker confirms the IR op additions.

[navigator] 2026-04-11 — **Naturalist's RFA parameters received and integrated.** K=3 fold, 6-word accumulator, primaries-then-carries layout, DBWIDTH=40, state = 52 bytes, all sourced from ReproBLAS. This is now the authoritative spec for 6.1's decision doc. The naturalist's three noted uncertainties (K vs accuracy tradeoff, exact exponent-to-bin computation, merge logic detail) are exactly the right three things for the 6.1 reader to look up in the TC 2015 paper. The 6.1 decision doc must confirm K=3, cite the tech report's error bound, and reproduce the state layout table above. Outstanding question: naturalist flagged the "align two states to common starting index" combine step as inferred rather than verified. Peak 6.3 implementer must read `binned_dmdmadd.c` directly before writing `reduce_rfa.f64`'s combine function.

[naturalist] 2026-04-11 — RFA parameters for fp64, fetched from the authoritative ReproBLAS source (not from the papers, which I couldn't reliably extract from PDF — but the ReproBLAS C source is where the paper's recommendations actually live, and the numbers below are quoted from it line by line so they're verifiable by anyone).

**Bin count (fold K) — navigator's question:**
- Default fold K = **3**. Source: ReproBLAS config.h documentation ("the recommendation is 3" if unsure; this matches the 2016 tech report's "6-word reproducible accumulator" phrase, since 2*fold = 6).
- Accumulator memory size in doubles: **2*fold**. Source: `src/binned/dbnum.c`:
  ```c
  int binned_dbnum(const int fold){
    return 2*fold;
  }
  ```
- Accumulator memory size in bytes: **2*fold*sizeof(double)**. Source: `src/binned/dbsize.c`:
  ```c
  size_t binned_dbsize(const int fold){
    return 2*fold*sizeof(double);
  }
  ```
- For fold = 3 (the recommended default): **6 doubles per accumulator = 48 bytes**. This is the fixed-length vector state the `.tam` reduction op needs.

**Memory layout — primary/carry pair arrangement:**
- Layout is **primaries-first, then carries at offset `fold`**, NOT interleaved pairs. Source: `src/binned/dbdbadd.c`:
  ```c
  void binned_dbdbadd(const int fold, const double_binned *X, double_binned *Y){
    binned_dmdmadd(fold, X, 1, X + fold, 1, Y, 1, Y + fold, 1);
  }
  ```
  The call `binned_dmdmadd(fold, X, 1, X + fold, 1, ...)` passes X and X+fold as two separate strided vector pointers. X is the primary bank; X+fold is the carry bank. So for fold=3, the layout is `[p0, p1, p2, c0, c1, c2]`.
- Each slot pair (p_i, c_i) represents one "bin" of the accumulator: p_i holds the main value, c_i holds the rounding error, Kahan-style.

**Bin width — how fp64 inputs get mapped into bins:**
- **DBWIDTH = 40 bits** per bin. Source: `include/binned.h`:
  ```c
  #define DBWIDTH 40
  ```
  Meaning each accumulator bin covers 40 bits of exponent range. An input's bin index is (roughly) `floor(input_exponent / 40)`.
- Maximum bin index for fp64:
  ```c
  #define binned_DBMAXINDEX (((DBL_MAX_EXP - DBL_MIN_EXP + DBL_MANT_DIG - 1)/DBWIDTH) - 1)
  ```
  Plugging in IEEE 754 fp64 values (DBL_MAX_EXP=1024, DBL_MIN_EXP=-1021, DBL_MANT_DIG=53): `((1024 - (-1021) + 53 - 1) / 40) - 1 = (2097 / 40) - 1 = 52 - 1 = 51`. So bin indices range **0..=51, giving 52 possible bin positions** spanning the full fp64 exponent range.
- **But the fold-K accumulator only tracks K adjacent bin positions at any one time.** The state is (K primary doubles, K carry doubles, 1 integer "starting index" that says which K of the 52 possible positions are currently live). When a new input's exponent is outside the K-bin window, the state shifts (carries compound, low bits spill, new high bin is started). So the runtime state is compact: ~48 bytes of doubles + ~4 bytes of index, even though the *range* of possible positions is much larger.

**What this means for the `.tam` IR reduction op:**

The RFA state type for Peak 6's `reduce_rfa.f64` is a fixed-length tuple:
```
type rfa_state = { index: i32, bins: [f64; 2*K] }
```
where K = 3 by default. Total state = 4 + 48 = **52 bytes per accumulator**, plus alignment padding. This is what pathmaker's IR spec needs to represent as a first-class type. It's small enough to pass by value through registers on both PTX and SPIR-V without spilling.

**The combine function (what Peak 6.3's `reduce_rfa.f64` actually does):**

Given two RFA states `X = (ix, [p0..p(K-1), c0..c(K-1)])` and `Y = (iy, [q0..q(K-1), d0..d(K-1)])`, the combine operation:
1. Aligns the two states to a common starting index (shift whichever has the smaller index upward, merging its low bins into the tail).
2. For each of the K bin positions, performs a **compensated (Kahan-style) add**: new_p_i = p_i + q_i + correction; new_c_i absorbs the rounding error.
3. The combine is **associative AND commutative by construction** (the alignment step is deterministic, the per-bin add is commutative). This is what gives RFA its order-independence and its claim to Kingdom A.

The per-element accumulate step is simpler: given scalar input x, compute its exponent, determine the target bin index in the state's window, and Kahan-add x into that bin's (p, c) pair.

**Error bound for fold = 3:**
- The 2016 tech report mentioned "the error bound with a 6-word reproducible accumulator and their default settings can be up to 229 times smaller than the error bound for conventional (recursive) summation." (Source: EECS-2016-121 abstract search extract.) So fold=3 is not just bitwise reproducible — it's also more accurate than naive recursive sum on adversarial inputs.
- Higher fold (K=4, K=6, K=8) gives progressively tighter bounds at the cost of more state and slower per-element accumulate. The paper has an analytical bound but I haven't extracted the closed form; `dbsize(fold) = 2*fold doubles` is the only size-vs-accuracy knob. For Peak 6.1's decision doc, **K=3 is the right starting default** — it matches ReproBLAS's own recommendation and gives enough headroom for financial data. K becomes a tunable the user sets if they want tighter bounds.

**Uncertainties I want to flag honestly:**
1. I have not read the papers themselves in full — the PDFs don't extract cleanly through WebFetch and the abstracts + slides are what I have. The numbers above are from the ReproBLAS software source, which is the *implementation* of what the papers describe. If the Peak 6 team wants a different K (e.g., fold=4 for better error bounds on long reductions) they should read the TC 2015 paper's error analysis directly — I can't do that read for them through WebFetch.
2. The exponent-to-bin-index computation (what I called "floor(input_exponent / 40)" above) is an approximation. The real computation involves an offset by `binned_DBMININDEX` and a clamp; the exact C code is in `src/binned/dbindex.c` if Peak 6 needs the bit-perfect version.
3. The "align two states to common starting index" operation during combine is the one place where I'm inferring the shape from first-principles reasoning rather than reading ReproBLAS source. The ReproBLAS `binned_dmdmadd` function in `src/binned/dmdmadd.c` is where the actual merge logic lives, and that's the reference implementation whoever opens Peak 6.5 (the PTX translator step) should port.

**TL;DR for 6.1 decision doc:**
- `K = 3` (the "fold" parameter), ReproBLAS-recommended default, gives the "6-word accumulator" the tech report advertises.
- State size: 6 doubles + 1 index = 52 bytes per accumulator.
- Layout: `[p0, p1, p2, c0, c1, c2]` — primaries then carries, each (p_i, c_i) a Kahan-style (value, error) pair.
- Bin width: `DBWIDTH = 40` bits of exponent range per bin.
- Max bins possible: 52 (but only K=3 live at once, slid via the index).

Sources all quoted from `github.com/willow-ahrens/ReproBLAS/master/` literal source code. No hallucinated constants.

[scientist] 2026-04-11 — Peak 4 harness complete before kickoff: campsites 4.1 (TamBackend+NullBackend), 4.2 (run_all_backends), 4.3 (assert_cross_backend_agreement), 4.4/4.5 (ToleranceSpec+ulp_distance), 4.7 (13 hard-case generators). 34 tests green, 4 ignore (2 xfail_nondeterministic pending Peak 6, 2 pending backends). Crate: crates/tambear-tam-test-harness/. Variance one-pass trap is in the hard-cases suite (one_pass_variance_trap generator). Parity table and sign-off log initialized at peak4-oracle/. Waiting for: pathmaker to land tambear-tam-ir AST types (to replace placeholder TamProgram/Inputs/Outputs), Peak 5 CPU interpreter (to activate hard-case backend tests), Peak 6 (to remove xfail_nondeterministic marks).

[navigator] 2026-04-11 — Adversarial baseline complete: 49 tests, 7 failures across 3 bugs. Bugs 2+3 (NaN-sticky min/max, identity leak on empty) fixed in commit ad84a51 — 47/49 now green. Bug 1 (variance catastrophic cancellation, one-pass formula) routed to pathmaker as architectural two-pass requirement; 2 pinned-red tests are the acceptance criteria. Pathmaker must write variance_pass in .tam IR as two-pass from the start (campsite 1.4).

[naturalist] 2026-04-11 — **PTX ABI note for pathmaker (campsite 1.17 / kernel signature).**

The PTX lowering blueprint (`peak3-ptx/lowering-blueprint.md`) resolves how `bufsize` lowers to PTX: since device pointers carry no length metadata, the assembler must append one `u32` length parameter per `buf<f64>` parameter in the kernel signature, in declaration order. For `variance_pass(buf<f64> %data, buf<f64> %out)` this means the PTX entry point becomes `variance_pass(u64 param_data, u64 param_out, u32 param_n_data)` — the length parameter is named `param_n_<bufname>` by convention.

**Flag for 1.17:** When pathmaker formalizes the `kernel` signature spec in campsite 1.17 (order_strategy extension), the IR spec should note this PTX ABI expectation: "every `buf<f64>` parameter in the kernel signature generates a corresponding length parameter in the PTX ABI, appended after all buffer pointers." The cudarc launch site must pass lengths alongside device pointers. This is not a change to the .tam IR semantics — it's a note on what the PTX backend does with each `bufsize` op.

Not blocking 1.17. Just a note before the IR spec finalizes so the ABI is stated once rather than discovered by the Peak 3 implementer mid-assembly.

[navigator] 2026-04-11 — Peak 6 reframed before any work started. Original sketch (two-stage host-fold + fixed launch config) achieves run_to_run determinism only; the summit test (7.11) requires gpu_to_gpu — bit-identical across CPU, CUDA, and Vulkan. Algorithm changed to RFA (Demmel-Ahrens-Nguyen, ARITH 2013 + IEEE TC 2015). campsites.md Peak 6 section updated with correction note, mandatory papers list, and revised campsite descriptions. Task #6 updated. Pathmaker and naturalist notified. Outstanding: naturalist to confirm fp64 bin count from the papers and post here before 6.1 is written.
