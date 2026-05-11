# Addendum — Liftability is the Default

**Trigger:** team-lead relayed Tekgy's design note 2026-04-22:

> The dispatcher should ALWAYS lift to parallel-prefix / scan / fused-pass
> form unless one of:
> 1. user explicitly overrode via `using(strategy="sequential")` or equivalent
> 2. the Op's algebraic structure genuinely doesn't admit lifting (no
>    monoid, non-commutative without a conjugation pattern that fits)
> 3. TAM determines it's structurally impossible to elevate (Fock boundary)
>
> In all other cases, lift. Sequential is the last resort, not the default.

This is not a tweak. It's an **inversion of the substrate's default mode**.
Re-running the phases on the trait shape with this constraint baked in.

---

## Phase 1 — Assumption Autopsy (deltas)

The original Phase 1 had implicit assumption **A22** (now-named): "the
backend chooses execution strategy from a menu of options." Tekgy's
note rejects that. New assumption layer:

- **A22 — backend chooses strategy.** REJECTED. The dispatcher (a
  layer above the backend, living in tambear core) chooses; the
  backend implements what the dispatcher requests.
- **A23 — sequential is a peer codepath.** REJECTED. Sequential is
  the *fallback* when the lift-precondition tests fail. Not a peer.
- **A24 — `is_commutative` / `is_associative` on JitOp are advisory.**
  REJECTED. They are **preconditions of the liftability decision**.
  The dispatcher reads them; if both are true, lift is the default
  unless the user overrode.
- **A25 — the Op decides its parallelism.** REJECTED. The Op declares
  its *algebraic structure*; the dispatcher decides whether
  parallelism is *available* (algebra), *appropriate* (data shape,
  override), or *forced sequential* (Fock boundary, override).
- **A26 — the trait surface is "compile + dispatch".** REJECTED. With
  liftability default, the trait surface must include "compile the
  *lifted* form by default; compile the sequential form on fallback."
  Two compile entry points or one entry point with a *strategy* tag.

---

## Phase 2 — Irreducible Truths (additions)

- **T20 — Liftability is a property of (Op × Grouping × Shape × user
  override × Fock-boundary determination), not of the backend.** The
  *decision* lives one tier above the backend. The backend's role is
  to *implement what the dispatcher requested* and *expose what
  strategies it supports*.
- **T21 — There are exactly three exit conditions for liftability:**
  - **(a) lift wins**: parallel-prefix / scan / fused-pass kernel
    emitted.
  - **(b) explicit user override → sequential**: `using(strategy=
    "sequential")` (or per-recipe equivalent) wired through
    `accumulate(...)`.
  - **(c) algebra blocks lift**: Op's `canonical_structure()` lacks
    Associativity OR (Op is non-commutative AND grouping requires
    reordering AND no conjugation pattern fits).
  - **(d) Fock-boundary blocks lift**: TAM determines the
    composition is genuinely sequential (rare; flagged honestly).
  Four conditions, but (a) is the default; (b)/(c)/(d) are the only
  ways out.
- **T22 — Sequential is a *fallback codegen*, not a peer mode.** It
  produces a working kernel; it does NOT participate in the
  dispatcher's strategy selection as an equal option. The
  dispatcher's flowchart is: *"try lift; if any of the three
  fallback conditions fires, emit sequential."*
- **T23 — Conjugation patterns are part of the lift toolkit.** Per
  the conjugation-pattern entry in vocabulary.md (Part II): a
  non-commutative Op may still lift if the data can be permuted into
  scan order (P ∘ T ∘ P⁻¹). The dispatcher must own this toolkit;
  the backend simply implements `compile_lifted(op, shape)` for
  whatever lift form the dispatcher chose.

---

## Phase 3 — Reconstruction with the lift-default

Re-running with the new truths. The trait surface must:

1. **NOT make sequential a peer of lifted.** A trait method
   `compile_sequential` and a trait method `compile_lifted` would
   structurally lie about the priority. Wrong.
2. **NOT bury the strategy choice in `params: &[u8]`.** Then the
   choice is invisible to the cache key, the codegen path, and the
   reviewer. Wrong.
3. **DO surface `Strategy` as a first-class compile-time enum.** The
   dispatcher picks; the backend lowers what was picked. The
   strategy is part of the cache key (lifted Welford ≠ sequential
   Welford in the cache).

Three reconstructions on the strategy axis:

### S1 — Strategy in `compile()` signature (winning shape)

```rust
trait DoorBackend {
    fn door_id(&self) -> DoorId;
    fn compile(
        &self,
        op: &JitOp,
        shape: &Shape,
        strategy: ExecutionStrategy,
        params: &[u8],
    ) -> Result<CompiledKernel, CompileError>;
    fn supports(&self, strategy: ExecutionStrategy) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionStrategy {
    /// **Default for any Op whose algebra admits it.** Parallel
    /// prefix / scan / fused-pass form, lowered to whatever the
    /// door's parallelism primitives are (warp shuffle on NVIDIA,
    /// subgroup ops on Vulkan, threadgroup on Metal, SIMD lanes +
    /// thread pool on CPU). Bit-exact deterministic via fixed
    /// associativity tree.
    Lifted,

    /// **Lifted with a permutation envelope.** For non-commutative
    /// Ops (AffineCompose, MatMulPrefix, ArgMax with strict
    /// tiebreak, stateful Welford-style merges that depend on
    /// observation order) the dispatcher emits a permute-pass +
    /// lifted scan + inverse-permute. Per the conjugation entry
    /// in vocabulary.md.
    LiftedConjugated { perm_kind: PermutationKind },

    /// **Sequential, single-threaded, in-order.** The fallback when
    /// (a) user explicitly forced via `using(strategy="sequential")`,
    /// (b) the Op's algebra blocks lift (no Associativity OR
    /// non-commutative with no fitting conjugation), or (c) TAM
    /// flagged a Fock-boundary genuine-sequential. NEVER chosen by
    /// the backend — only by the dispatcher.
    Sequential { reason: SequentialReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequentialReason {
    /// User explicitly chose via `using(strategy="sequential")`.
    UserOverride,
    /// Op's algebra blocks lift (no Associativity, or non-commutative
    /// with no fitting conjugation).
    AlgebraBlocks,
    /// TAM determined the recipe is past the Fock boundary.
    FockBoundary,
}
```

### S2 — Strategy implicit in the JitOp variant

Reject. Doubles the JitOp enum (`JitOp::AddLifted`, `JitOp::AddSeq`,
`JitOp::WelfordLifted`, ...). Loses the property that an Op IS the
math, not the execution.

### S3 — Strategy lives only in the cache key, never in the trait

Reject. The backend cannot codegen differently if it doesn't see the
choice. Strategy MUST be in the trait method's signature.

**Winner: S1.** `ExecutionStrategy` enters `compile()` as a parameter,
participates in the cache key (different strategy → different cached
kernel), and the `supports()` query lets the dispatcher feature-detect
(some doors won't support `LiftedConjugated` for some perm kinds in
early sweeps; the dispatcher then falls through to `Sequential` with
`reason: AlgebraBlocks` or escalates).

---

## Phase 4 — Mapping table (additions)

| New assumption | Replacing truth |
|---|---|
| A22 backend chooses strategy | T20 dispatcher chooses, backend implements |
| A23 sequential is a peer | T22 sequential is fallback codegen |
| A24 commutativity advisory | T21 commutativity is a precondition |
| A25 Op decides parallelism | T20 Op declares algebra; dispatcher decides |
| A26 compile+dispatch only | S1 compile takes ExecutionStrategy |

---

## Phase 5 — The (revised) Aristotelian Move

The original move was R10′ (trait triad). The liftability default does
**not** invalidate the triad — it adds a parameter to `DoorCodegen::lower`
(or the simplified `DoorBackend::compile` pathmaker scaffolded). The
revised move:

**REVISED MOVE: ship R10′ + ExecutionStrategy.** Specifically:

1. The simplified `DoorBackend::compile(op, shape, strategy, params)`
   shape pathmaker scaffolded gets a `strategy: ExecutionStrategy`
   parameter inserted between `shape` and `params`.
2. `JitOp::is_associative()` and `is_commutative()` (already there
   thanks to pathmaker) are wired to a new `JitOp::default_strategy(
   shape: &Shape) -> ExecutionStrategy` helper that encodes the
   liftability decision tree at the IR layer.
3. The **dispatcher** (a new module `tambear::jit::dispatcher` —
   doesn't exist yet, conceptually a thin layer in `accumulate()`'s
   real body) is the policy seat. It calls
   `default_strategy(shape)`, honours the `using(strategy=...)`
   override, and asks the backend if it `supports(strategy)`. If
   not, falls back per the policy.
4. The cache key includes `ExecutionStrategy` as a typed field
   (BLAKE3-serializable). Lifted Welford and Sequential Welford get
   different cached kernels.
5. The CPU Cranelift backend implements `Lifted` via
   tree-reduction codegen + thread-pool fan-out (with deterministic
   tree shape — fixed associativity for bit-exactness across runs).
   `Sequential` is the literal `for` loop. `LiftedConjugated` is
   permute-pass + lifted-scan + inverse-permute, sharing the
   thread-pool.

---

## Phase 6/7 — Recursive challenge

Re-running with S1 in the assumption set. New questions:

- **Q-rec-1.** Does `ExecutionStrategy` need a 4th variant for "tile-
  based map+reduce" (block-then-cross-block)? My answer: No. That
  IS `Lifted` for `Grouping::All` reductions. Tile size lives in the
  `Shape` cache key, not as a strategy variant.
- **Q-rec-2.** Does `Lifted` need a sub-strategy for "warp-only"
  vs "block-only" vs "device-wide"? My answer: No. That's a
  per-door codegen decision. The dispatcher says `Lifted`; the
  backend picks the right kernel-family entry point. The
  `EntryPoint` mechanism from R10′ Phase 8 covers this.
- **Q-rec-3.** What's the dispatcher's interface to `using(strategy=...)`?
  Per the Tekgy note, this is set at the recipe call site and flows
  down. Concrete answer: a `Strategy` field in the per-recipe
  `using()` bag, read by the dispatcher before it asks
  `default_strategy(shape)`. The dispatcher's pseudocode:

  ```
  let user_choice = using.strategy.unwrap_or(StrategyOverride::Auto);
  let strategy = match user_choice {
      StrategyOverride::Auto => op.default_strategy(shape),
      StrategyOverride::ForceSequential =>
          ExecutionStrategy::Sequential { reason: UserOverride },
      StrategyOverride::ForceLifted =>
          op.lifted_strategy_or_panic(shape),
  };
  if !backend.supports(strategy) {
      strategy = fallback(strategy, backend);
  }
  ```

- **Q-rec-4.** What about Ops where `is_associative() = true` but
  `is_commutative() = false` AND the grouping is `Prefix`? AffineCompose
  is the worked example. Per pathmaker's `JitOp::is_commutative()`,
  this returns `false` for AffineCompose. But AffineCompose IS
  liftable for `Prefix` grouping — that's why it lives in the
  substrate. The decision tree must be: "associative + (commutative
  OR grouping is order-preserving) → Lifted." Prefix preserves order
  by construction; ByKey loses order; All loses order. So the
  decision uses `Grouping::preserves_order()` too.

  Adds a method to `Grouping`: `preserves_order(&self) -> bool`.
  - `All` — no
  - `ByKey` — no
  - `Prefix` — yes
  - `Segmented` — yes (within each segment)
  - `Windowed` — yes
  - `Tiled` — yes
  - `Graph` — depends on adjacency; conservative `false`
  - `Probabilistic` — no (soft membership permutes)

  Then `default_strategy(shape)` becomes:

  ```rust
  pub fn default_strategy(&self, shape: &Shape) -> ExecutionStrategy {
      if !self.is_associative() {
          // Genuine Fock-boundary case (none today; future Ops only).
          return ExecutionStrategy::Sequential {
              reason: SequentialReason::AlgebraBlocks,
          };
      }
      if self.is_commutative() || shape.grouping.preserves_order() {
          return ExecutionStrategy::Lifted;
      }
      // Associative but not commutative AND grouping doesn't preserve
      // order: try conjugation if a permutation kind exists for this
      // (Op, Shape) pair; otherwise sequential.
      if let Some(perm_kind) = conjugation_perm_for(self, shape) {
          return ExecutionStrategy::LiftedConjugated { perm_kind };
      }
      ExecutionStrategy::Sequential {
          reason: SequentialReason::AlgebraBlocks,
      }
  }
  ```

  This is the **liftability default encoded in the IR layer**.
  Sequential is genuinely the last branch in the function. Lifted
  is the early-return.

---

## Phase 8 — Forced Rejection

Forcibly rejecting parts of the new design.

- **What if `ExecutionStrategy` was NOT in the cache key?** Then a
  cached `Lifted` kernel could be returned to a `Sequential` request
  → wrong codegen executes → wrong (or non-deterministic) results.
  **MUST be in cache key.** Confirmed.
- **What if `supports()` did not exist?** The dispatcher would have
  to commit to a strategy without knowing the backend can produce it,
  then fail at compile time. `supports()` lets the dispatcher do
  graceful degradation. **MUST exist.**
- **What if `default_strategy()` lived on the dispatcher, not on
  `JitOp`?** Then the algebraic structure (associativity / commutativity)
  would have to be queried by the dispatcher, but the *decision logic*
  would be split from the data. Cohesion lost. Better: the IR layer
  (`JitOp` + `Grouping`) owns the decision; the dispatcher owns the
  override-and-fallback flowchart that consumes the decision.
  Confirmed: `default_strategy` is a `JitOp` method.
- **What if there were a 4th SequentialReason for "performance
  heuristic said small-N is faster sequential"?** Per Tekgy: lifting
  is *empirically always* an efficiency gain when possible. So no
  fourth reason. The three reasons (UserOverride, AlgebraBlocks,
  FockBoundary) are exhaustive.
- **What if the backend could refuse a strategy without `supports()`
  failing — e.g., return a generic `CompileError::NotSupported`?**
  That's an OK fallback path (CompileError already covers it), but
  the explicit `supports()` query lets the dispatcher *avoid the
  failed compile entirely*, saving the user a confusing error. Keep
  `supports()`.
- **What happens to `LiftedConjugated` if no `perm_kind` fits the
  (Op, Shape)?** The `default_strategy()` falls through to
  `Sequential { reason: AlgebraBlocks }`. Honest.
- **What if the user forces `Lifted` via `using(strategy="lifted")`
  on an Op that doesn't admit it?** Two choices: (a) panic with
  clear message, (b) silently degrade to Sequential. Per the no-tech-
  debt rule and tests-serve-reality: **panic.** Misuse should be
  loud. Adds: `JitOp::lifted_strategy_or_panic(shape)`.

---

## Convergence with R10′

The liftability addendum doesn't redesign R10′ — it **fills in a
parameter** that R10′ left unstated. R10′'s `DoorCodegen::lower(ir,
shape, cap) -> CompiledArtifact` becomes:

```rust
trait DoorCodegen {
    fn lower(&self, ir: &TambearIr, shape: &Shape, strategy: ExecutionStrategy,
             cap: &DoorCapability) -> Result<CompiledArtifact, CompileError>;
    fn supports(&self, ir: &TambearIr, shape: &Shape, strategy: ExecutionStrategy)
        -> bool;
}
```

And `CacheKey` gains a `strategy: ExecutionStrategy` field
(BLAKE3-serialized via its `Hash` impl).

Pathmaker's scaffolded `compile(op, shape, params) -> CompiledKernel`
signature becomes:

```rust
fn compile(&self, op: &JitOp, shape: &Shape, strategy: ExecutionStrategy,
           params: &[u8]) -> Result<CompiledKernel, CompileError>;
fn supports(&self, op: &JitOp, shape: &Shape, strategy: ExecutionStrategy) -> bool;
```

Three lines added; no structural rework.

---

## Concrete deliverables (for pathmaker to land)

If pathmaker accepts this addendum, the implementation deltas are:

1. **`crates/tambear/src/jit/strategy.rs`** (new, ~100 lines) — defines
   `ExecutionStrategy`, `PermutationKind`, `SequentialReason`,
   `StrategyOverride` (the using() variant), and the
   `conjugation_perm_for(jit_op, shape)` helper.
2. **`crates/tambear/src/jit/jit_op.rs`** — add `default_strategy(&self,
   shape: &Shape) -> ExecutionStrategy` and `lifted_strategy_or_panic`.
3. **`crates/tambear/src/accumulate.rs`** — `Grouping::preserves_order()`.
4. **`crates/tambear/src/jit/door.rs`** — `compile()` gains
   `strategy: ExecutionStrategy`; new `supports()` method.
5. **`crates/tambear/src/jit/fingerprint.rs`** — strategy enters the
   BLAKE3 hash.
6. **`crates/tambear/src/jit/dispatcher.rs`** (new, future Sweep 8C
   territory but skeleton ok now) — the `(using_override, op,
   shape) → strategy` flowchart from Phase 6/7 Q-rec-3.

Tests (under `cargo test --workspace`):

- `default_strategy` returns `Lifted` for (Add, All), (Add, Prefix),
  (Welford, All), (Welford, Prefix).
- `default_strategy` returns `Lifted` for (AffineCompose, Prefix)
  (associative-not-commutative + order-preserving grouping).
- `default_strategy` returns `Sequential { AlgebraBlocks }` for
  (AffineCompose, ByKey) (algebra blocks: not commutative AND
  grouping not order-preserving AND no conjugation today).
- `default_strategy` returns `Sequential { UserOverride }` when
  `StrategyOverride::ForceSequential` is set.
- Cache-key smoke test: same (op, shape, params) with different
  strategy → different `CacheKey`.
- `supports()` smoke test: NoOp backend supports all three
  strategies (it produces no real kernel; tests only the surface).

---

## Asks

For pathmaker:
- **Accept this addendum?** It's three lines of trait change plus a
  new file. It locks the liftability default into the IR.
- **Q1/Q2/Q3** from the original Phase 1-8 file are still open;
  please answer when convenient.

For adversarial:
- **New attack**: find a (JitOp, Grouping, Shape) where
  `default_strategy()` returns `Lifted` but lifting genuinely
  produces wrong results. The strongest test of the decision tree
  is finding cases where my logic in Phase 6/7 Q-rec-4 is incomplete.
- Original three attacks still standing.
