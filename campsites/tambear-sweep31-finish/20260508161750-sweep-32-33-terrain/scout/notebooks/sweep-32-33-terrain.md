# Sweep 32 + 33 Terrain Survey

**Scout on tambear-sweep31-finish, 2026-05-08.**

All findings verified against substrate — files read directly.

---

## Sweep 32 — Cache-key plumbing for precision coordinates

### Where the cache key is constructed

`R:\tambear\crates\tambear\src\jit\fingerprint.rs`

`FingerprintHasher` with BLAKE3 hasher. `IR_VERSION = 9`. `CRANELIFT_VERSION = "0.131"`.

Entry point: `cache_key_with_strategy(op, shape, strategy, door_id, capability_bytes, param_blob) -> CacheKey`

### What the current key hashes

19 `feed_*` methods with tag-byte disambiguation. Tags 0x01–0x19:

- 0x01: feed_ir_version
- 0x02: feed_door_id
- 0x03: feed_jit_op (via op.tag() string)
- 0x04: feed_scalar_ty (Shape.dtype)
- 0x05: feed_dim_hint (Shape.dim)
- 0x06: feed_grouping (Shape.grouping — complex, with ByKey relabeling)
- 0x07: feed_validity (Shape.validity)
- 0x08: feed_assumption_tag (each tag in Shape.tags)
- 0x09: feed_execution_strategy
- 0x0A: feed_scan_family_policy (u8)
- 0x0B: feed_param_blob (raw bytes)
- 0x0C: feed_capability_bytes (raw bytes)
- 0x0D: feed_non_finite_claim (Shape.non_finite_claim)
- ... (others via the u8/u16/u32/u64/str/bool/bytes primitives)
- 0x18: feed_strategy (via tag byte)
- 0x19: feed_non_finite_claim

**Currently NO precision-context feeding.** `PrecisionContext` is not in the fingerprinter at all today.

### Where PrecisionContext is defined

`R:\tambear\crates\tambear\src\lattice\precision.rs`

```rust
pub struct PrecisionContext {
    pub requested_precision_bits: u32,
    pub rounding: RoundingMode,
}

impl PrecisionContext {
    pub fn dispatch_level(&self) -> PrecisionLevel { /* 0-53→P0F64; 54-106→P1DD; _→P2BigFloat */ }
    pub fn dispatched_precision_bits(&self) -> u32 { /* 53 / 106 / requested */ }
}
```

`PrecisionLevel::tag()` returns stable bytes: P0F64=0, P1DD=1, P2BigFloat=2.
`RoundingMode::tag()` returns 0-4.

### What PrecisionContext's involvement is in the cache path today

None. The fingerprinter doesn't import or use PrecisionContext. Shape doesn't have a precision field.

### Minimal addition needed

**Step 1: Add `feed_precision_context` to `FingerprintHasher`** (fingerprint.rs):

```rust
pub fn feed_precision_context(&mut self, ctx: &PrecisionContext) {
    self.feed_u8(0x1A);                              // new tag, after 0x19
    self.feed_u32(ctx.requested_precision_bits);
    self.feed_u8(ctx.dispatch_level().tag());
    self.feed_u32(ctx.dispatched_precision_bits());
}
```

Tag 0x1A is the next available slot. Tag bytes must never be reused.

**Step 2: Wire into `cache_key_with_strategy`** — this function takes a Shape; when Shape gains a precision field (see Sweep 33), the key construction calls `feed_precision_context` if `shape.precision.is_some()`.

**Step 3: Bump `IR_VERSION` from 9 to 10** — any change to the key invalidates existing caches; version bump makes the invalidation explicit and intentional.

**Critical test gap (DEC-031 §6 #11 — storage-vs-operation cache-key separation):**

The test that MUST exist after Sweep 32:

```rust
// requested=80 → dispatch_level=P1DD, dispatched=106
// requested=106 → dispatch_level=P1DD, dispatched=106
// These have different requested_precision_bits but same dispatch_level+dispatched.
// They MUST produce DISTINCT cache keys (§3.3 orthogonality).
let ctx_80 = PrecisionContext { requested_precision_bits: 80, rounding: RoundingMode::RoundToNearestTiesEven };
let ctx_106 = PrecisionContext { requested_precision_bits: 106, rounding: RoundingMode::RoundToNearestTiesEven };
assert_ne!(feed_precision_context(ctx_80), feed_precision_context(ctx_106));
```

Without this test, the separation between requested and dispatched is untested and could silently collapse.

---

## Sweep 33 — TAM routing for BigFloat → force CPU

### The corrected picture (briefing language vs substrate)

DEC-031 §3.6 says "`supports(op, shape, strategy)` returns false for BigFloat-bearing JitOps on non-CPU doors." The briefing repeats this. **The substrate tells a different story.**

**JitOp has no BigFloat concept.** `R:\tambear\crates\tambear\src\jit\jit_op.rs` is a closed enum: Add, Max, Min, ArgMax, ArgMin, DotProduct, Distance, Welford, LogSumExp, AffineCompose, Scan(SemiringKind), MatMulPrefix { n }. No BigFloat variant. `ScalarTy` is {F64, I64, F32, U8} only. No precision-level concept in any JitOp method.

`op_uses_bigfloat(op)` as a predicate on JitOp cannot be implemented today — there is nothing in JitOp that says "this op operates at BigFloat precision."

**The correct integration point is Shape, not JitOp.**

### Current state of Shape

`R:\tambear\crates\tambear\src\jit\shape.rs`

Shape struct fields today:
- `dtype: ScalarTy`
- `dim: DimHint`
- `grouping: Grouping`
- `validity: Validity`
- `tags: Vec<AssumptionTag>`
- `non_finite_claim: NonFiniteClaim`

No precision field.

DEC-031 §3.3 reserves `ScalarTy::BigFloat(prec)` as a future Tier-1 amendment (storage-precision, not yet shipped). When that arrives, dtype carries storage precision and non-CPU `supports()` rejects it automatically. But Sweep 33 is about operation-precision, which flows through PrecisionContext.

### The current supports() implementations

**CPU backend** (`cpu_cranelift.rs:324`):
```rust
fn supports(&self, op: &JitOp, shape: &Shape, strategy: ExecutionStrategy) -> bool {
    scalar_reduction_kind(op).is_some()
        && matches!(shape.grouping, Grouping::All)
        && matches!(strategy, ExecutionStrategy::Sequential { .. })
        && !matches!(shape.validity, Validity::Error)
}
```
Returns true only for scalar reductions with All grouping + Sequential.

**NoOpBackend** (`door.rs:1104`):
```rust
fn supports(&self, _op: &JitOp, _shape: &Shape, _strategy: ExecutionStrategy) -> bool {
    false
}
```
Returns false unconditionally. This is the placeholder for future GPU/NPU doors.

### What Sweep 33 needs to write

**Step 1: Add `precision: Option<PrecisionContext>` to Shape** (shape.rs):

```rust
pub struct Shape {
    pub dtype: ScalarTy,
    pub dim: DimHint,
    pub grouping: Grouping,
    pub validity: Validity,
    pub tags: Vec<AssumptionTag>,
    pub non_finite_claim: NonFiniteClaim,
    pub precision: Option<PrecisionContext>,   // NEW
}
```

Default: `None` (backwards-compatible — all existing Shape::new() calls still work).

Add `with_precision(ctx: PrecisionContext) -> Self` builder. Add to `canonicalize()` pass-through.

**Step 2: Wire into fingerprinter** — `cache_key_with_strategy` calls `feed_precision_context` when `shape.precision.is_some()`. This is the same method Sweep 32 adds, so the two sweeps coordinate at this method.

**Step 3: Add BigFloat rejection to non-CPU `supports()`** — for any future GPU/NPU backend:

```rust
fn supports(&self, op: &JitOp, shape: &Shape, strategy: ExecutionStrategy) -> bool {
    // BigFloat operations are CPU-only (variable-length Vec<u64> state
    // cannot map to fixed-width SSA registers in GPU IR).
    if let Some(ctx) = &shape.precision {
        if ctx.dispatch_level() == PrecisionLevel::P2BigFloat {
            return false;
        }
    }
    // ... rest of existing checks
    false // NoOpBackend currently always false anyway
}
```

**Step 4: Test** — construct Shape with precision at P2BigFloat, assert NoOpBackend.supports() returns false; assert CPU backend returns true (it falls through to its existing scalar-reduction-kind check, which is unrelated).

### Why BigFloat is structurally CPU-only

BigFloat state is `Vec<u64>` — heap-allocated, variable-length. JitOp's `StateRepr` only knows `ScalarTy` (fixed-width SSA values). No `ScalarTy` variant can represent a BigFloat accumulator. The JIT IR assumes scalar/fixed-width state; BigFloat state doesn't fit. CPU sequential execution calling into Rust BigFloat methods is the only viable door — not as a current capability gap but as an architectural truth. The force-CPU rule is permanent for the foreseeable future, not a temporary workaround.

### Sweep 32 + 33 coordination point

Both sweeps touch:
1. `fingerprint.rs` — `feed_precision_context` method (new)
2. `shape.rs` — `precision` field + `with_precision` builder (Sweep 33)
3. `cache_key_with_strategy` wiring (Sweep 33 calls the Sweep 32 method)

If pathmaker does these in sequence — Sweep 32 first (fingerprinter method + IR_VERSION bump), then Sweep 33 (Shape field + wiring + door routing) — there's no collision risk.

---

## Files to read before implementing

Sweep 32:
- `R:\tambear\crates\tambear\src\jit\fingerprint.rs` — full file (fingerprinter)
- `R:\tambear\crates\tambear\src\lattice\precision.rs` — PrecisionContext + PrecisionLevel

Sweep 33:
- `R:\tambear\crates\tambear\src\jit\shape.rs` — full file (Shape)
- `R:\tambear\crates\tambear\src\jit\door.rs` — supports() signature at line 689, NoOpBackend at line 1104
- `R:\tambear\crates\tambear\src\jit\cpu_cranelift.rs:324` — CPU supports() to understand what stays unchanged
- `R:\tambear\crates\tambear\src\jit\jit_op.rs` — JitOp enum (confirm no BigFloat variant before writing op_uses_bigfloat)
