# Amendment — Cache-Bake Principle Audit

**Sweep 8 / Task 8A** · Author: aristotle · Date: 2026-04-22

Tekgy broadcast the cache-bake principle (Note 2):

> **Bake what changes the INSTRUCTION STREAM; pass what changes the
> OPERANDS.**
>
> Baked (cache-key): dtype, rank, Op variant, Grouping pattern,
> Validity policy, Lift strategy.
>
> Passed (dispatch-time): buffer pointers, data values, dynamic axis
> lengths, dynamic stride values.
>
> Size is the tradeoff zone.

This is a **principle audit**, not a structural change. The question
is: does the current spec's CacheKey honor the principle across all
six categorical axes, or does it accidentally bake something that
should be passed, or accidentally pass something that should be baked?

---

## Six-axis audit of the current `CacheKey`

```rust
pub struct CacheKey {
    pub ir_hash: [u8; 32],              // includes JitOp variant
    pub param_hash: [u8; 32],
    pub shape_fingerprint: [u8; 32],    // includes dtype, rank, dim
    pub validity_policy: Validity,
    pub strategy: ExecutionStrategy,
    pub door_id: DoorId,
    pub capability_fingerprint: [u8; 32],
}
```

| Axis | Tekgy says | Current spec | Result |
|---|---|---|---|
| dtype (f32 ≠ f64 ≠ i64) | BAKE | in `shape_fingerprint` via `Shape.dtype: ScalarTy` | ✅ correctly baked |
| rank (1D ≠ 2D ≠ 3D) | BAKE | implicit via `DimHint` singular (needs multi-dim extension per Note 1) | ⚠️ single-axis today; Note 1 addresses |
| Op variant (Add ≠ Mul ≠ Max) | BAKE | in `ir_hash` — `JitOp::tag()` is a stable per-variant string | ✅ correctly baked |
| Grouping pattern (All ≠ ByKey ≠ Windowed) | BAKE | in `shape_fingerprint` via `Shape.grouping: Grouping` | ✅ correctly baked |
| Validity policy (Propagate ≠ Ignore ≠ Error) | BAKE | in `CacheKey.validity_policy` as its own field | ✅ correctly baked |
| Lift strategy (Lifted ≠ LiftedConjugated ≠ Sequential) | BAKE | in `CacheKey.strategy` (added per liftability addendum) | ✅ correctly baked |

**All six categorical axes correctly bake into the cache key.** No
ghost — the principle validates the current structure.

---

## Operands-axis audit

| Operand | Tekgy says | Current spec | Result |
|---|---|---|---|
| Buffer pointers | PASS | `dispatch(inputs: &[&Buffer], outputs: &mut [&mut Buffer])` — NOT in cache key | ✅ correctly passed |
| Data values | PASS | buffers carry data; never in cache key | ✅ correctly passed |
| Dynamic axis length | PASS | `DimHint::Dynamic` treats length as runtime; NOT in cache key when Dynamic | ✅ correctly passed |
| Dynamic stride values | PASS | NOT currently in Shape — Note 1 adds `Layout::Strided(Vec<isize>)` | ⚠️ needs Note 1 to explicitly separate "stride as bake" vs "stride as pass" |

**Operands are correctly passed.** The stride question resolves in
Note 1's deconstruction below.

---

## The "size is the tradeoff zone" principle as spec discipline

Current `DimHint`:

```rust
pub enum DimHint {
    Dynamic,            // one kernel all sizes (pass n at runtime)
    Static(usize),      // unroll wins ~10× for small sizes
}
```

Note 1 adds:
- `Bounded { multiple_of: usize }` — SIMD-friendly large data without
  per-n specialization (one kernel per alignment class)
- `SymbolicEqual(axis_ref)` — square matrices share one kernel across
  all sizes

These four variants correspond exactly to Tekgy's four tradeoff points:
- Static → small (unroll wins)
- Bounded → SIMD-friendly large data
- Dynamic → one kernel all sizes
- SymbolicEqual → cross-size sharing under structural constraints

**Principle applied:** `DimHint::Static(n)` and `DimHint::Bounded {
multiple_of }` are BAKED (different kernel per n or per alignment
class). `DimHint::Dynamic` is PASSED. `DimHint::SymbolicEqual` is a
RELATIONSHIP that lets two Dynamic-looking axes share a cache entry
that would otherwise be two.

The shape fingerprint hashes the DimHint variant + parameters; it does
NOT hash the runtime value of a Dynamic-length axis. Bake/pass
boundary falls exactly where Tekgy's rule says it should.

---

## One ghost to flag (not in the spec yet)

**`param_hash` boundary is implicit.** The current `DoorCodegen::lower(...,
params: &[u8])` takes params as a byte buffer. ALL of `params`
currently hashes into `param_hash` in CacheKey. That's right under
Tekgy's rule IF everything in `params` changes the instruction stream
(e.g., loop unroll factor, tile size, compile-time-known scalar
constants).

But: recipes MIGHT put a "use this f64 alpha" into params intending
it as a runtime knob. That would be WRONG under the cache-bake
principle — alpha is an operand, not a compile-time specialization,
and belongs in a buffer (or in a "dispatch-time parameters" channel
we don't yet have).

**Remediation** (not a structural change to the trait — a discipline
note):

Split `DoorCodegen::lower` from dispatch-time params:

```rust
// WAS:
fn lower(&self, op, shape, strategy, params: &[u8], cap)
    -> Result<CompiledArtifact, CompileError>;

// COULD BE:
fn lower(&self, op, shape, strategy,
         bake_params: &BakeParams,   // compile-time, in CacheKey
         cap) -> Result<CompiledArtifact, CompileError>;

// And dispatch-time, separate operand channel:
fn dispatch(&self, ..., runtime_params: &[u8], ...) -> Result<Event, ..>;
```

Where `BakeParams` is a typed struct — `{ unroll_factor: Option<u32>,
tile_size_hint: Option<WorkgroupShape>, ... }` — that explicitly names
WHAT can be baked. And `runtime_params` is the push-constant / inline
kernel argument channel for per-dispatch scalars.

**But I don't think we need this restructure yet.** The current flat
`params: &[u8]` works if the dispatcher is disciplined about what it
puts there. A discipline note in the spec ("params is for values that
change the instruction stream; runtime scalars go through buffers or
a future `runtime_params` channel when added in Sweep 8.5+") captures
the rule without introducing a type that we may get wrong right now.

**My recommendation for Sweep 8:** keep `params: &[u8]` flat, add the
discipline note below to the spec, and defer the BakeParams/runtime_params
split to Sweep 9 when the first recipe tests whether a "this value is
scalar and varies per dispatch" case actually arises.

---

## Spec delta — discipline note to add

Add to `trait-spec-locked.md` and `door.rs`:

```rust
/// Compile-time specialization parameters. Content hashes into
/// `CacheKey::param_hash` — different values produce different
/// compiled kernels.
///
/// **Cache-bake rule (Tekgy, 2026-04-22)**: this channel is for
/// values that change the INSTRUCTION STREAM — loop unroll factor,
/// tile-size hint, constant-folded scalars, feature-flag bits.
///
/// **Values that change the OPERANDS** (per-dispatch scalars that
/// do NOT change the compiled kernel) belong in:
/// - A buffer input (preferred; allocates a one-element Buffer),
///   OR
/// - The runtime_params channel on `DoorDispatcher::dispatch`
///   (to be added in Sweep 8.5+ when the first per-dispatch scalar
///   recipe arrives; not currently present).
///
/// Misusing this channel for operand-style values makes every
/// per-dispatch value a cache miss.
pub struct BakeParams(pub Vec<u8>);  // opaque to the trait; codegen
                                      // interprets per-Op
```

And a corresponding admonition in the audit test suite:

```rust
#[test]
fn cache_bake_discipline_params_affects_cache_key() {
    // Different params -> different cache entry (kernel specializes).
    let key1 = CacheKey::compute(/* ..., params = b"unroll=4" */);
    let key2 = CacheKey::compute(/* ..., params = b"unroll=8" */);
    assert_ne!(key1, key2);
}
```

This isn't a trait redesign — it's a doc-comment + an intent-capture
struct name. Pathmaker can land this in an hour.

---

## Conclusion

Tekgy's Note 2 **validates the current CacheKey design**. All six
baked axes are correctly in the key; all operand axes correctly stay
out. One implicit ghost (params channel could be misused) gets
captured by a discipline note + an opaque `BakeParams` newtype
wrapper, with the BakeParams/runtime_params structural split deferred
to Sweep 8.5+ pending the first consumer that tests the need.

**Audit result: no structural change needed.** The trait spec is
cache-bake-principle clean.
