# OrderStrategy Registry

**Campsite 1.16 artifact. Authoritative companion to `order_strategy.rs`.**

The OrderStrategy registry is the named-entry list that programs reference when
declaring a reduction order. This document is the human-readable complement to
the Rust module; the Rust module is the machine-executable source of truth.

---

## Why a registry, not a closed enum

The prior implementation (campsite 1.15 → commit 8622ab6) used a closed Rust
enum (`SequentialLeft`, `TreeFixedFanout(u32)`, `BackendDefault`). This
encoding had a structural defect identified by Aristotle's Phase 8 finding:

> A closed enum couples the IR's definition of "which strategies exist" to the
> Rust source file. Every new strategy requires a source change, a version bump,
> and a recompile. Worse, a backend that speaks a strategy the enum doesn't know
> about has nowhere to land.

The v5 design (adopted 2026-04-12, campsite 1.16) replaces the enum with:

- **`OrderStrategyRef(String)`** in `ast.rs` — a named reference. The AST carries
  a name, not a variant. The name is stable; the registry can grow without touching
  the AST type.

- **`order_strategy.rs`** — the registry module. A `HashMap<&'static str, StrategyEntry>`
  built once and reused. Each entry carries a formal spec, a runnable Rust reference
  implementation, bit-exact test vectors, and fusion-compatibility metadata.

- **Verifier validates by registry lookup.** `crate::order_strategy::is_known(name)` —
  unknown names are rejected with a clear error citing all known names.

This follows the engineering shape observed across the trek's three registries:
OrderStrategy registry + oracles registry + guarantee ledger — all named-entry
structures with formal specs and metadata. The shape is a team pattern.

---

## Registry entries

### `sequential_left`

**Status:** Full — spec, implementation, bit-exact test vectors, all green.

**Summary:** Serial left-to-right accumulation — `acc = fold(0.0, +, values)`.

**Formal definition:**
```
sequential_left([]) = 0.0
sequential_left([v]) = v
sequential_left([v0, v1, ..., vN]) = sequential_left([v0, ..., v(N-1)]) + vN
```

**Numerical contracts:**
- Deterministic: same input → same output on every run, every backend.
- IEEE-754 faithful: each addition is a faithful fp64 add (no FMA, no flush-to-zero).
- NaN propagates: any NaN input produces NaN output.
- Inf propagates: Inf + -Inf = NaN.

**CPU interpreter:** The loop body SSA update IS a left fold, so the CPU
interpreter naturally executes `sequential_left` regardless of declared strategy.

**Fusion-compatibility class:** `sequential`

**Bit-exact test vectors (pinned):**

| Label | Inputs | Expected |
|---|---|---|
| empty | `[]` | `0.0` |
| single | `[1.0]` | `1.0` |
| two_elements | `[1.0, 2.0]` | `3.0` |
| three_elements | `[1.0, 2.0, 3.0]` | `6.0` |
| large_then_small | `[1e15, 1.0, -1e15]` | `1.0` |
| nan_propagation | `[1.0, NaN, 2.0]` | `NaN` |
| inf_propagation | `[1.0, +Inf, -Inf]` | `NaN` |

Note on `large_then_small`: the result is `1.0` because IEEE-754 arithmetic
preserves `1e15 + 1.0 = 1000000000000001.0` (the value 1.0 is above the
ULP threshold of 1e15), and then subtracting 1e15 gives 1.0.

---

### `tree_fixed_fanout_2`

**Status:** Full — spec, implementation, bit-exact test vectors, all green.

**Summary:** Recursive binary tree reduction with fanout 2.

**Formal definition:**
```
tree_fixed_fanout_2([]) = 0.0
tree_fixed_fanout_2([v]) = v
tree_fixed_fanout_2(values) =
  let mid = len(values) / 2   -- integer division
  tree_fixed_fanout_2(values[0..mid]) + tree_fixed_fanout_2(values[mid..])
```

The split point `mid = len / 2` is deterministic for every input length.
For a given array length, every backend that implements `tree_fixed_fanout_2`
produces the same tree shape, the same intermediate sums, and the same final result.

**Numerical contracts:**
- Deterministic: same input length → same tree shape → same bits.
- NaN propagates: any NaN input produces NaN output.
- Diverges from `sequential_left` on numerically sensitive inputs (see pinned
  divergence test vector below).

**GPU implementation target (Peak 3 / Peak 7):**
PTX backend uses warp-level tree reduction for the within-warp step (32 threads =
5 tree levels). The block-level fold of warp partial sums uses `sequential_left`
to keep the host-side deterministic. The full kernel declares `tree_fixed_fanout_2`
for the warp step; the inter-warp combination is declared separately.

**Fusion-compatibility class:** `tree_pow2`

**Bit-exact test vectors (pinned):**

| Label | Inputs | Expected |
|---|---|---|
| empty | `[]` | `0.0` |
| single | `[1.0]` | `1.0` |
| two_elements | `[1.0, 2.0]` | `3.0` |
| three_elements | `[1.0, 2.0, 3.0]` | `6.0` |
| large_then_small | `[1e15, 1.0, -1e15]` | `1.0` |
| nan_propagation | `[1.0, NaN, 2.0]` | `NaN` |

**Pinned divergence from `sequential_left`:**

Input `[1.0, 1e-16, -1.0, 1e-16]`:
- `sequential_left` → `1e-16`
  (0→1.0→1.0→0.0→1e-16, where 1e-16 is below ULP(1.0) in step 2 but
  recovered in step 4 from 0.0)
- `tree_fixed_fanout_2` → `2^-53 ≈ 1.1102e-16`
  (tree grouping: left=`1.0+1e-16=1.0`, right=`-1.0+1e-16=-1.0+1e-16`;
  the IEEE-754 result of `-1.0+1e-16` retains the 1e-16 contribution,
  giving a final result different from `1e-16`)

This divergence is documented, not a bug. It is what I7 (total order = bit-exactness)
is protecting: programs that care about numerical precision must declare the strategy
that gives the precision they need.

---

### `rfa_bin_exponent_aligned`

**Status:** Stub — registered, verifier accepts it, interpreter panics at runtime.

**Summary:** Reproducible floating-point accumulation via bin-exponent alignment
(Demmel-Nguyen 2013/2015).

**Why registered now:** Peak 6 (Deterministic Reductions) will implement this
algorithm. IR programs that target deterministic cross-backend reductions may
name this strategy today; the verifier accepts it. The CPU interpreter panics
with a clear message citing Peak 6.

**Algorithm target (to be implemented in Peak 6):**
```
K = 3 folds
DBWIDTH = 40 bits (bin width)
State = 48 bytes (3 × f64 accumulators + 3 × f64 overflow guards)

Pass 1 — Scan: compute max biased exponent E_max of all inputs.
Derive bin boundaries: bin k covers [2^(E_max - k*DBWIDTH), 2^(E_max - (k+1)*DBWIDTH)).
Pass 2 — Deposit: for each input, round to nearest bin boundary and accumulate
  fractional part into bin k.
Combine: sum K bin accumulators from highest to lowest.
```

**Correctness guarantee (once implemented):** Two executions on the same input
values produce bit-identical output regardless of the order inputs arrive,
regardless of backend, and regardless of parallelism level.

**Reference:** Demmel, J., & Nguyen, H. D. (2013). Fast Reproducible Floating-Point
Summation. ARITH 21. And: Demmel & Nguyen (2015). Parallel Reproducible Summation.
IEEE Trans. Computers 64(7).

**Fusion-compatibility class:** `rfa`

**Bit-exact test vectors:** None yet. Will be derived from the Demmel-Nguyen
reference implementation validated against mpmath at 50-digit precision (per I9).

---

## How programs use the registry

In `.tam` source:
```
reduce_block_add.f64 %out, %slot, %acc @order(sequential_left)
```

The verifier calls `order_strategy::is_known("sequential_left")`. If unknown,
the verifier emits:
```
reduce_block_add.f64 @order(foo): 'foo' is not a registered OrderStrategy.
Known strategies: ["rfa_bin_exponent_aligned", "sequential_left", "tree_fixed_fanout_2"]
```

The parser does not validate names — it only checks syntax. Registry validation
is a verifier responsibility, so error messages can cite the full current list.

---

## How to add a new strategy

1. In `order_strategy.rs`:
   - Write the reference implementation function.
   - Write bit-exact test vectors.
   - Insert a `StrategyEntry` into `build_registry()`.
   - Add tests.
2. In this document: add a section for the new entry.
3. The AST, parser, printer, verifier, and interpreter require NO changes.
   They work with names, not variants.

---

## Lifecycle conventions

Following the three-registry convergence pattern (per navigator's 2026-04-12 observation):

- **Names are stable once registered.** A name that appears in a committed `.tam`
  program is a permanent contract. Do not rename or remove entries.
- **Implementations may be refined.** If the reference implementation has a bug,
  fix it and update the test vectors. The name stays the same.
- **Stub entries are acceptable.** `rfa_bin_exponent_aligned` is a valid stub.
  It registers the name, documents the target, and panics at runtime with a
  clear message. This is better than leaving the name unregistered (which would
  cause verifier errors on programs that correctly target Peak 6).
- **`compat_class` is required on all entries** — even stubs. Phase 2 fusion
  uses this field; collecting it now avoids a format version bump later.

---

## Status

- **Campsite:** 1.16 (Peak 1 — IR Architect)
- **Committed:** see SHA in `navigator/check-ins.md`
- **Depends on:** campsites 1.1–1.15 (all prior Peak 1 campsites)
- **Unblocks:** campsite 1.17 (per-kernel default), Peak 3 (PTX strategy dispatch),
  Peak 6 (RFA implementation)
