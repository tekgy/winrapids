# Sweep 32 Cache-Key Invariant — Proof Sketch + Antibody

**Created:** 2026-05-08 (continuation session, tambear-sweep31-finish)
**Author:** aristotle
**Brief:** Navigator asked for a proof sketch of the Sweep 32 cache-key
invariant (Surface 3 of my prior gauntlet) that pathmaker can use to
validate the Sweep 32 implementation. This doc supplies the sketch +
the antibody pattern that operationalizes it.

**Inputs:**
- DEC-031 §3.7 (cache-key extension for precision coordinates)
- DEC-031 §6 #11 (storage-vs-operation cache-key separation)
- DEC-019 (no vendor lock-in; cache key composition)
- F13 (antibody pattern for scope-precondition rules)
- My prior gauntlet's Surface 3 — `silent-failure-proptest-gauntlet.md`

---

## The invariant (briefing's restatement)

> Two ops with the SAME storage precision but DIFFERENT dispatch levels
> must produce DIFFERENT cache keys.

In full DEC-031 form: the cache key for an op is a function of:

- `tambear_ir_hash` (the IR identity)
- `param_bag_hash` (`using()` parameters)
- `shape_fingerprint` (dim/dtype/grouping/etc.)
- `validity_policy` (NaN handling)
- `door_id` (CPU / NVIDIA / Vulkan / Metal / DX12 / etc.)
- **Three precision coordinates:**
  - `requested_precision_bits` (what `using(precision=N)` asked for)
  - `dispatch_level` ∈ {P0F64, P1DD, P2BigFloat}
  - `dispatched_precision_bits` ∈ {53, 106, requested}

Storage `ScalarTy` (e.g., `BigFloat{precision_bits: 200}`) is INDEPENDENT
of operation precision. A user can store BigFloat(200) and still operate
at P0F64 (53 bits) for fast comparison ops where 53 bits suffice, or at
P1DD (106 bits) for intermediate computations, etc. The cache key MUST
distinguish these states or the cache returns a result computed under
different precision assumptions than the consumer requested.

---

## Proof sketch

**Theorem.** Given fixed `(Op, S, validity_policy, door_id)` with
multiple legal dispatch levels, the cache key is a function

```
K: DispatchLevel × DispatchedBits × RequestedBits → Hash256
```

If the precision_coordinates serialization is injective AND
domain-separated AND blake3 is collision-resistant, then `K` is
injective on its three-coordinate input domain (with cryptographic
probability).

**Construction.** Define the serializer `s: (D, dispatched, requested) → Bytes`:

```
s(D, dispatched, requested) = byte_tag(D) ++ u32_le(dispatched) ++ u32_le(requested)
```

where `byte_tag` maps the dispatch enum to fixed bytes:

```
P0F64                                → 0x00
P1DoubleDouble                       → 0x01
P2BigFloat{precision_bits: _}        → 0x02
```

This serialization is **fixed-length (1 + 4 + 4 = 9 bytes)** and trivially
injective: distinct tuples `(tag, u32, u32)` produce distinct byte
sequences.

**The full key construction** concatenates fixed-length-prefixed
coordinates:

```
input = tambear_ir_hash[32]
     ++ param_bag_hash[32]
     ++ shape_fingerprint[32]
     ++ validity_policy[1]
     ++ door_id[1]
     ++ s(D, dispatched, requested)[9]
```

(Total: 107 bytes, fixed.) Then:

```
key = blake3(input)
```

**The fixed-length prefixes guarantee domain separation.** The
precision_coordinates 9-byte segment is at a fixed offset (98) and
cannot be confused with any other field. So two inputs that differ in
`(D, dispatched, requested)` necessarily differ in those 9 bytes, and
the blake3 input is distinct.

**By blake3's collision-resistance**, distinct inputs produce distinct
keys with probability `1 − 2^{−256}` (effectively 1). So:

```
K(Op, S, D₁, dispatched₁, req₁) ≠ K(Op, S, D₂, dispatched₂, req₂)
```

whenever the triples differ. ∎

---

## Corollaries

**Corollary 1 (briefing's exact claim).** Same `S`, different `D` (with
possibly different dispatched/requested) → different keys. Holds by the
theorem.

**Corollary 2 (gauntlet Surface 3).** Same `(S, requested)` but
different op-precision via different dispatch → different keys. Holds.

**Corollary 3 (extreme case).** Same `(Op, S, requested)` with TWO
different dispatched (e.g., requested=53 routes to P0F64-dispatched=53,
but requested=53 forced to P2BigFloat-dispatched=53 via override) →
different keys. Holds because dispatch_level tag differs.

---

## F13 antibody — operationalization

The proof's correctness DEPENDS on the serializer being injective AND
domain-separated. If pathmaker uses variable-length encoding (e.g.,
`bincode` with default options), domain separation may fail at the bit
level — two distinct triples could produce the same byte sequence under
length elision.

**The F13 antibody for THIS rule:** the serializer must be unit-tested
with at least one input pair where the components differ but a naive
concatenation would collide.

**Required tests (the antibody)**:

```rust
#[test]
fn surface_3_serializer_no_collision_at_dispatch_tag() {
    let a = serialize_precision_coordinates(P0F64, 53, 53);
    let b = serialize_precision_coordinates(P1DoubleDouble, 53, 53);
    let c = serialize_precision_coordinates(
        P2BigFloat { precision_bits: 53 }, 53, 53
    );
    assert_ne!(a, b, "P0F64 vs P1DD with same precision must serialize distinctly");
    assert_ne!(b, c, "P1DD vs P2BigFloat with same precision must serialize distinctly");
    assert_ne!(a, c, "P0F64 vs P2BigFloat with same precision must serialize distinctly");
}

#[test]
fn surface_3_serializer_no_collision_at_swap() {
    // The most subtle: swapping dispatched and requested must produce
    // distinct serializations.
    let a = serialize_precision_coordinates(
        P2BigFloat { precision_bits: 200 }, 53, 200
    );
    let b = serialize_precision_coordinates(
        P2BigFloat { precision_bits: 200 }, 200, 53
    );
    assert_ne!(a, b, "swap of dispatched/requested must serialize distinctly");
}

#[test]
fn surface_3_full_cache_key_distinguishes_dispatch_at_same_storage() {
    let storage = ScalarTy::BigFloat { precision_bits: 200 };
    let op = Op::Add;
    let validity = ValidityPolicy::Default;
    let door = DoorId::Cpu;

    let key_p0 = cache_key(op, storage, validity, door, PrecisionContext {
        requested: 53,
        dispatch_level: PrecisionLevel::P0F64,
        dispatched: 53,
    });
    let key_p1 = cache_key(op, storage, validity, door, PrecisionContext {
        requested: 106,
        dispatch_level: PrecisionLevel::P1DoubleDouble,
        dispatched: 106,
    });
    let key_p2 = cache_key(op, storage, validity, door, PrecisionContext {
        requested: 200,
        dispatch_level: PrecisionLevel::P2BigFloat { precision_bits: 200 },
        dispatched: 200,
    });

    assert_ne!(key_p0, key_p1);
    assert_ne!(key_p1, key_p2);
    assert_ne!(key_p0, key_p2);
}
```

The first two tests catch serializer-level bugs (variable-length
encoding, missing tag bytes, accidental collision under length
elision). The third catches end-to-end issues (e.g., the
precision_coordinates field being forgotten in the cache_key
composition — easy to do if pathmaker copy-pastes from the existing
DEC-019 cache_key without adding the new field).

---

## Things that could break the invariant in implementation

**Failure mode 1: variable-length encoding.** If pathmaker uses
`bincode::serialize` or `serde_json::to_vec` for the precision_coordinates,
the byte representation of `u32` may be variable-length (varint). Then
`s(P0F64, 53, 53)` could collide with `s(P0F64, 5, 333)` in
pathological ways. **Antibody:** use fixed-length `u32_le` encoding
explicitly; lock the byte offset of each field.

**Failure mode 2: missing dispatch tag.** If pathmaker only serializes
`(dispatched, requested)` and omits `dispatch_level`, two distinct
dispatches that happen to land at the same `dispatched_bits` (e.g., a
P0F64 request at 53 vs a P1DD-dispatched-at-53 path that's been
documented somewhere) will have colliding keys. The dispatch tag is
load-bearing. **Antibody:** the surface_3_serializer_no_collision_at_dispatch_tag
test catches this.

**Failure mode 3: forgetting one of the three coordinates.** Surface 3's
canonical case (storage=BigFloat(200), op-precision=53 vs 200) requires
all three coordinates: same storage, different (requested, dispatch,
dispatched). If only `requested` is in the key, the cache returns
results computed at one precision when consumers asked for another.
**Antibody:** end-to-end test that varies only one coordinate at a
time and asserts each variation produces a distinct key.

**Failure mode 4: kernel-cache invalidation regression.** When tambear
ships and users have warm caches, changing the precision_coordinates
serialization between releases means existing cached kernels can't be
loaded by the new binary. The version-of-the-serializer must be bumped
in the cache-version tag (separate field), so old caches are detected
as stale and recompiled. **Antibody:** integration test that loads a
cache file written with a deliberately-different serializer version,
asserts re-compile rather than silent-mismatch.

---

## Cross-link to F13

This is a textbook F13 instance. The rule is "cache key uniquely
identifies the (Op, storage, dispatch) tuple, so cache hits are valid."
The scope precondition is "the cache key construction includes ALL
distinguishing axes." Without antibody: silent failure where the cache
returns wrong-precision results. With antibody (the three tests above):
every shipping CI run validates the construction.

The proof sketch IS the recognition (rule + scope). The antibody tests
ARE the operationalization (mechanical artifact). Together they make
the invariant load-bearing substrate for Sweep 33+ (TAM routing) and
Sweep 34 (oracle migration).

---

## Status

Proof sketch + antibody tests filed for pathmaker's Sweep 32
implementation. When pathmaker takes Task #6 (Sweep 32 cache-key
plumbing), this doc is the spec. The three tests above are the F13
antibody for the precision-coordinate axis; they MUST be in the Sweep
32 commit, not deferred to a follow-up.

If pathmaker's implementation passes all three, Sweep 32 is locked. If
any fails, the proof sketch points to where the implementation deviated
from the spec.

Standing by for Task #10 re-pressure-test once pathmaker fixes Bugs #1
and #2 from the prior multi-limb arith.rs commit.
