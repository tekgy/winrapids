# Navigator Check-ins

This file is where team members leave brief status notes when their work touches someone else's.
Use it when you actually need something from another role — not as a daily standup.

Format: `[role] [date] — <what you need / what just landed / what's blocking>`

---

<!-- entries below, newest first -->

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

[navigator] 2026-04-11 — Peak 6 reframed before any work started. Original sketch (two-stage host-fold + fixed launch config) achieves run_to_run determinism only; the summit test (7.11) requires gpu_to_gpu — bit-identical across CPU, CUDA, and Vulkan. Algorithm changed to RFA (Demmel-Ahrens-Nguyen, ARITH 2013 + IEEE TC 2015). campsites.md Peak 6 section updated with correction note, mandatory papers list, and revised campsite descriptions. Task #6 updated. Pathmaker and naturalist notified. Outstanding: naturalist to confirm fp64 bin count from the papers and post here before 6.1 is written.
