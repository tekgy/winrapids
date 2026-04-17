# Plan: Op defaults to cross-platform bit-exact determinism

**Status:** planning, no code touched yet.
**Driver:** TERNYX-SIP (see `R:/ternyx-sip/docs/column-graph.md` and
`R:/ternyx-sip/docs/signal-compute-spec-for-tambear.md`) — bit-exact
reproducibility is a cryptographic requirement, not a nicety.
**Scope:** tambear-wide. Every consumer of `accumulate` benefits.
**Non-goal (for this pass):** GPU deterministic scatter. The CPU path
lands first; GPU is a follow-up once the CPU contract is stable.

---

## Current state (as read, 2026-04-11)

### What exists

- **Kulisch accumulator** — `primitives/specialist/kulisch_accumulator.rs`
  (568 lines). 34 × 128-bit integer words, radix at bit 2100, covers the
  entire f64 finite range with ~1000 bits of headroom. Exact summation:
  `add_f64`, `sub_f64`, `add_slice`, `to_f64` (correctly-rounded RTNE).
  API is single-writer only. **Missing:** `merge(&other)` /
  `add_accumulator(&other)` for parallel associative combine. Trivial
  to add (word-wise signed addition with carry propagation) and
  essential for any tree/split reduction.

- **Op enum** — `accumulate.rs:125`. Variants: `Add`, `Max`, `Min`,
  `ArgMin`, `ArgMax`, `DotProduct`, `Distance`. Contract is informal
  ("additive monoid (ℝ, +). Maps to atomicAdd. The default.") — no
  determinism guarantee anywhere in the docstrings.

- **Semiring trait + SemiringKind enum** — `accumulate.rs:215-398`.
  Already present, with `AdditiveReal`, `TropicalMinPlus`,
  `TropicalMaxPlus`, `LogSumExpSemiring`, `BooleanSemiring`. Good
  foundation: the deterministic work here fits within the semiring's
  `add()` operation, because the semiring is where the non-associativity
  risk lives.

- **AccumulateEngine executor** — `accumulate.rs:442+`. Today it
  dispatches `(Grouping, Op)` combinations to `ComputeEngine::scatter_phi`,
  which emits CUDA kernels that use `atomicAdd` on f64 output buffers
  (source at `compute_engine.rs:675-692`). **This is the non-determinism
  source:** atomic order on GPU is scheduler-dependent; f64 addition is
  non-associative; same input gives different bit patterns run-to-run
  across backends and thread counts.

- **sum_k + two_sum + compensated primitives** — lower-quality
  determinism vehicles that don't match Kulisch's exactness but are
  cheaper per element. Useful as alternative strategies that consumers
  can opt into via `using()` without leaving the deterministic regime.

### What does NOT exist yet

- **Kulisch merge** — two accumulators → one, word-wise signed add.
  Needed for any tree/block reduction.
- **`Op::Add` default implementation that is cross-platform bit-exact.**
  The type exists; the semantic contract and the Kulisch-backed
  implementation do not.
- **`using()` key for reduction strategy.** No `sum_strategy: "..."`
  lookup anywhere in the executor.
- **Cross-platform bit-exactness test harness** — no test asserts
  "1-thread CPU and N-thread CPU give identical bit patterns," let
  alone across GPU backends.
- **`.tam` IR capture of the resolved strategy** — the IR should
  record that a given `accumulate` was compiled with
  `sum_strategy = "kulisch"`, so an audit tool can verify the binary
  actually used the strategy it claimed.

---

## The contract we want

**Every `Op` variant is cross-platform bit-exact deterministic by
default.** Same input data + same call → identical bit pattern
regardless of:

- Thread count (1, 2, 128, 65536)
- Execution order within parallel regions
- Backend (CPU, CUDA, Vulkan, Metal via wgpu)
- CPU architecture (x86_64, aarch64, Apple Silicon)
- Rebuild (fresh compile produces same binary output for same input)

**Non-determinism is opt-in via `using()` with an explicit key.** No
silent non-determinism. If a consumer wants atomicAdd-speed, they
write `using(sum_strategy: "nondet")` and it is visible at the call
site forever.

---

## Op-by-Op determinism strategy

| Op | Default strategy | Notes |
|---|---|---|
| `Op::Add` | `kulisch` | Kulisch register per group; merge tree; final `to_f64()`. Exact. |
| `Op::Max` | `ieee754_2019` | IEEE fmax propagates NaN; associative; order-invariant. Already deterministic at primitive level. Tie-break convention: if two inputs are equal bit-for-bit, result is bit-for-bit the same (trivial). |
| `Op::Min` | `ieee754_2019` | Symmetric to Max. |
| `Op::ArgMax` | `ieee754_2019 + lowest_index_ties` | Tie-break: on equal values, return the lowest input index. Documented convention. |
| `Op::ArgMin` | `ieee754_2019 + lowest_index_ties` | Symmetric. |
| `Op::DotProduct` | `kulisch_over_fma` | For each pair, compute `two_product_fma(a, b) = (hi, lo)` and add both components to the Kulisch register. Exact. |
| `Op::Distance` | `kulisch_over_centered_square` | For each pair, `(a - b)` then `two_product_fma` then add both hi+lo into Kulisch. Exact. |
| `Op::LogSumExp` (via Semiring) | `kulisch_with_max_subtract` | Find global max deterministically; subtract; Kulisch-sum the `exp`s; `log + max_subtract` back. |

The Op enum itself does not grow. Strategy is a `using()` key.

## `using()` strategy keys

Single key: `sum_strategy`. Accepts:

- `"kulisch"` — exact Kulisch accumulation. **Default for every Op whose
  native combine is `+`.** This is what `Op::Add`, `Op::DotProduct`, and
  `Op::Distance` dispatch to unless overridden. Cost: ~10-20× naive
  per element, fully deterministic and exact.
- `"pairwise"` — pairwise tree summation with fixed pairing order. Cheaper
  than Kulisch, O(log n) error growth, cross-platform deterministic if
  the tree shape is fixed. Not exact.
- `"kahan"` — Kahan compensated summation in a fixed scan order. Cheaper
  still. Deterministic within a scan order; merging two kahan states is
  the trickier bit (need two_sum merge, documented in `sum_k.rs`).
- `"neumaier"` — Kahan variant that handles larger-than-sum values.
- `"sum_k_2"`, `"sum_k_4"` — parameterized k-fold compensation from
  `primitives/specialist/sum_k.rs`. Graduated precision / speed tradeoff.
- `"nondet"` — atomicAdd, tree-parallel, whatever is fastest. **Not
  deterministic.** Explicit opt-out, visible at the call site. This is
  the only non-deterministic strategy; anything else on the menu is a
  determinism-preserving alternative.

Strategies other than `"nondet"` are **all cross-platform deterministic**.
They differ in precision and speed, not in reproducibility.

The resolved strategy gets written into the `.tam` IR so an audit
tool can confirm "this accumulate compiled with sum_strategy =
kulisch" against the delivered binary.

---

## Minimum work to land the default-deterministic guarantee (CPU path first)

1. **Kulisch merge.** Add
   `pub fn merge(&mut self, other: &Self)` to `KulischAccumulator` —
   word-wise signed add with carry propagation across 34 words. Maybe
   20 lines of straight integer arithmetic. Tests: merge is associative
   (A⊕B⊕C bit-exact regardless of parenthesization), merge of two
   accumulators = sum of their slice inputs.

2. **Op contract docstrings.** Rewrite the docstring on each Op variant
   in `accumulate.rs` to specify the cross-platform bit-exact
   determinism contract. Name the default strategy for each. Note the
   `using()` override key.

3. **CPU executor path.** Replace `scatter_phi`'s atomicAdd-like `+=`
   with a Kulisch-register-per-group implementation:
   - Allocate `Vec<KulischAccumulator>` with one entry per group.
   - Iterate in fixed index order, call `add_f64(phi(v, r))` into the
     group's Kulisch register.
   - Finalize: call `to_f64()` on each register.
   - This is single-threaded for now; parallelism lands when we wire the
     tree-merge.

   For `All + Add`, the same path with a single Kulisch register and
   all-zero keys.

   For `Prefix + Add`, Kulisch-backed Blelloch scan: store Kulisch states
   at internal tree nodes, combine via `merge`, extract `to_f64()` at
   every output position. Bigger deliverable — queue it after the All
   and ByKey paths work.

4. **`using()` strategy dispatch.** Where the AccumulateEngine resolves
   an Op to an implementation, look up `using_bag.get("sum_strategy")`
   first. Default to `"kulisch"` if absent. Dispatch to named
   implementation.

5. **Parallel CPU tree-merge for ByKey.** Partition the data into P
   chunks (P = rayon threads). Each chunk builds per-group Kulisch
   registers independently (no atomics, no contention). Final merge
   combines chunk registers for each group with `merge`. This is
   embarrassingly parallel AND bit-exact AND order-invariant —
   merging in any order gives the same register, because `merge` is
   associative over exact integer arithmetic.

6. **Bit-exact test harness.** New file `tests/determinism_contract.rs`:
   - For a corpus of inputs (small, medium, adversarial, overflow-prone,
     denormal-heavy), run `accumulate(..., Op::Add)` with 1 thread and
     with rayon's max threads. Assert `a.to_bits() == b.to_bits()`.
   - Same for `Op::Max`, `Op::Min`, `Op::ArgMax`, `Op::ArgMin`,
     `Op::DotProduct`, `Op::Distance`.
   - Cover `Grouping::All`, `Grouping::ByKey`, and (later) `Grouping::Prefix`.
   - Run against random inputs with three seeds to catch any
     thread-count-dependent behavior.

7. **GPU deterministic scatter.** Not in the first pass. The CPU path
   landing first means we can prototype every SIP recipe against the
   deterministic CPU path, get bit-exact results, then port the
   strategy to GPU. For GPU this probably means either:
   - Sort keys, then run deterministic segment-merge on the sorted
     stream (gives bit-exact result regardless of thread count).
   - Or use a fixed-topology reduction tree with Kulisch-in-shared-memory
     per block, merge across blocks in block-order.
   Both are research work on top of the CPU foundation.

---

## Sequencing vs SIP recipes

The "oxygen mask first" rule says we build the tambear primitives we
need before writing a SIP-specific binary. The determinism contract
is exactly that kind of foundational primitive — every SIP recipe
depends on it, and we do not want to rewrite them later.

Concrete ordering:

1. **This plan approved.** (← we are here)
2. Kulisch merge + tests.
3. `Op::Add` default dispatch to Kulisch on CPU path, for All and ByKey
   groupings. Tests: bit-exact across thread counts.
4. `using(sum_strategy: ...)` key wired with at least two strategies
   (`kulisch`, `nondet`) to prove the plumbing.
5. Cross-platform bit-exact harness committed as a permanent CI gate.
6. `Op::DotProduct`, `Op::Distance` on the Kulisch path.
7. **Now write SIP recipes** — they inherit determinism automatically.
   Per the column graph, these are `parkinson_volatility`,
   `roll_spread`, `vpin`, `hawkes_intensity`, `kyle_lambda`,
   `amihud`, `lee_mykland_jump_count`, plus the quantile sketch
   primitive (separate plan needed — see the mergeable quantile
   sketch campsite in `industrialization/missing-primitives/`).
8. GPU deterministic scatter — a separate Peak 6 continuation — is the
   parallel track that does not block SIP recipe work.

---

## Decisions locked (resolved 2026-04-16 session)

All five pre-coding open questions have been resolved. Short-form
decisions here; long-form reasoning in
`R:\ternyx-sip\docs\session-recovery-2026-04-16.md`.

1. **Kulisch per-group allocation — dense, always.**
   34 × i128 = 544 bytes/register. For SIP's realistic load (1 hour,
   1 coin, 36,000 buckets, ~11 fused accumulations) memory is ~220 MB
   — trivial on RTX Pro 6000 (96 GB VRAM) and cheap on system RAM.
   No sparse/hashmap-backed variant is built until a real use case
   demands it. Grouping by `tick_id` (the 54 GB scare) is not a
   realistic scatter key — there is always a coarser grouping.

2. **NaN/Inf policy — two independent knobs, both default to `"propagate"`.**
   Tambear is a general math library; different consumers have different
   conventions for handling non-finite inputs. IEEE 754 propagate is the
   right default (what hypothesis tests, oracles, and general scientific
   code expect). Skip-mode is a consumer opt-in visible at every call
   site.

   ```
   using(nan_policy: "propagate" | "skip" | "error")   // default: propagate
   using(inf_policy: "propagate" | "skip" | "error")   // default: propagate
   ```

   SIP writes `using(nan_policy: "skip", inf_policy: "skip")` at every
   signal call site. Other consumers mix (e.g. physics sim: skip NaN
   for missing data, propagate Inf for real limits).

   **No `count_valid` output from tambear.** This was in the earlier draft
   of this plan; removed after SIP side confirmed they derive it themselves
   as `n_events − n_invalid` from existing header fields (`n_events` in the
   NYXL prefix, `n_invalid` in the NYXD header — where `n_invalid` replaces
   the old `n_nan` field name and counts any tick with `!is_finite(value)`).
   Tambear does the math; SIP handles bookkeeping.

   **Skip-mode semantics** (when a consumer opts in):

   The kernel filters the input via `is_finite(v)` at the phi step (single
   predicate catches NaN, +∞, −∞) and accumulates only finite values. The
   accumulator's internal seed stays the Op's identity. At finalize:

   | Op | All-invalid emit | Reason |
   |---|---|---|
   | `Add` | **0.0** | additive identity is safe; preserves prefix-sum flatline through gaps |
   | `DotProduct` | **0.0** | same reasoning |
   | `Distance` | **0.0** | same reasoning |
   | `Max` | **NaN** | register's `−∞` seed never replaced; consumer must NOT read `−∞` as a real observation |
   | `Min` | **NaN** | same reasoning with `+∞` seed |
   | `ArgMax` | **(NaN, sentinel_idx)** | consumer checks value's NaN flag |
   | `ArgMin` | **(NaN, sentinel_idx)** | same |
   | `LogSumExp` | **NaN** | register's `−∞` seed would poison downstream |

   The asymmetry is intentional. For `Add`-family ops, the identity `0.0`
   is a safe-to-emit element that preserves the **prefix-sum flatline
   property**: an all-invalid bucket contributes +0 to every running sum,
   so `pfx[i] == pfx[i−1]`. Range queries spanning gaps give the right
   answer with no special case. For `Max`/`Min`-family ops, the identity
   is `±∞` which would poison downstream consumers (a risk model reading
   `max_price = −∞` would treat it as a real observation). NaN is the
   honest "no observation" signal.

   Implementation: the skip-mode kernel just doesn't call `add_f64(x)`
   when `!is_finite(x)`. For Max/Min, the kernel initialises the register
   to NaN rather than ±∞; any finite input replaces the NaN via `fmax`/
   `fmin` semantics; all-invalid leaves NaN in place. No sidecar `count_valid`
   needed — the register state itself tells the story.

   **Propagate-mode semantics** (the default):

   Any non-finite input poisons the group. `Op::Add`: NaN input → NaN
   output via a poison flag (Kulisch's existing `!is_finite()` early-return
   means we need explicit poison-tracking, since Kulisch's silent skip
   would otherwise give wrong propagate-mode answers). `Op::Max`/`Min`:
   IEEE 754 fmax/fmin semantics with NaN in IEEE 754-2019 minNum/maxNum
   sense — NaN in → NaN out.

   **Error-mode semantics:** kernel returns an error on first non-finite
   input. For defensive pipelines that refuse to proceed on bad data.

3. **Blelloch over Kulisch — not a problem at SIP scale.**
   SIP's prefix scans are over the 36,000 per-bucket signal arrays,
   not over the 100M-tick input. Tree state for 23 lanes
   (7 new + 16 existing) × 20 MB/lane = ~460 MB. Plain Blelloch is
   fine. Three breakapart patterns documented for when tick-level
   prefix is ever needed: (a) chunked scan with block reduction
   (~557 KB peak per lane for 100M), (b) streaming Kulisch fold
   with checkpoint emission, (c) per-block-store-only with forward
   sweep. None of these need to be built today.

4. **`.tam` IR records resolved execution, not request.** The
   IDE/TBS layer refuses to compile any combination that cannot be
   deterministically lowered on the target backend. The capability
   table per backend lives in the IDE and is consulted at edit time,
   not runtime. By the time anything reaches `.tam`, request ==
   execution — there is no silent runtime downgrade path. If a
   consumer writes `using(sum_strategy: "kulisch")` on a backend
   that doesn't support Kulisch, the IDE flags it at edit time and
   forces a backend change or strategy change before compile.

5. **`Op::Add` name kept, contract silently strengthened.**
   No external consumers yet; no code to break. Future
   `Op::Add` calls automatically inherit the cross-platform
   bit-exact guarantee. Strategy selection is a `using()` key,
   not an enum variant.

---

## Success criteria for this piece of work

- `accumulate(data, All, Value, Op::Add)` on the CPU path produces a
  bit-identical f64 on 1 thread and on 32 threads for 1000+ adversarial
  inputs (denormals, Kahan-traps, overflow-near-sum, heavy cancellation,
  etc.).
- Same for `Grouping::ByKey { keys, n_groups }` across thread counts.
- Consumers writing `accumulate(..., Op::Add)` today continue to work
  with no source changes; they simply get the stronger guarantee.
- `using(sum_strategy: "nondet")` explicit opt-out path compiles and
  runs, and its result may differ from the deterministic path but is
  clearly marked.
- The architecture doc `docs/architecture/atoms-primitives-recipes.md`
  gains a "determinism contract" section codifying all of the above.

---

## What NOT to do in this pass

- Do not extend the `Op` enum with new variants (no
  `Op::AddKulisch`, `Op::AddNondet`). Strategy is a `using()` key.
  Growing the Op enum fragments the semantic surface.
- Do not add new `Semiring` instances beyond what exists. The existing
  `Additive`/`TropicalMin`/`TropicalMax`/`LogSumExp`/`Boolean` set is
  enough for the session's classification-bijection theorem and for
  SIP.
- Do not touch the GPU scatter kernel yet. CPU first; GPU is a
  separate campsite.
- Do not pre-optimize Kulisch for SIMD or vectorized paths. The
  guarantee comes first; the speed work comes after the contract is
  locked.
