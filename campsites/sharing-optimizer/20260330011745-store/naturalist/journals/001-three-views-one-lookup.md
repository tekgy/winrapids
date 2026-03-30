# Three Views, One Lookup

*Naturalist journal — 2026-03-30*

---

## The question

The sharing optimizer promises three levels of sharing:
- **Provenance** (865x): has this computation been done before?
- **Dirty tracking** (continuous): have inputs changed since last compute?
- **Residency** (cross-session): is the result currently on GPU?

Phase 2 described these as three sharing surfaces. The natural question when building: are these three systems or one system with three faces?

## What the code says

The answer is in `store.rs`, lines 383-411. All three traits are implemented on `GpuStore`. All three resolve against the same `HashMap<[u8; 16], u32>` — the `self.index`.

```rust
// DirtyBitmap
fn is_clean(&self, provenance: &[u8; 16]) -> bool {
    self.index.contains_key(provenance)
}

// ResidencyMap
fn is_resident(&self, provenance: &[u8; 16]) -> bool {
    // index.get → check location == Location::Gpu
}

// ProvenanceCache
fn provenance_get(&mut self, provenance: &[u8; 16]) -> Option<BufferPtr> {
    self.lookup(provenance)  // index.get → update stats → return pointer
}
```

Three trait methods. One HashMap. Three questions about the same key:
- **Does it exist?** → is_clean (DirtyBitmap)
- **Does it exist on GPU?** → is_resident (ResidencyMap)
- **Does it exist and what's the pointer?** → provenance_get (ProvenanceCache)

They're not three systems. They're three projections of one fact: *whether a provenance key maps to a live entry*.

## Why is_clean works by existence

This is the load-bearing insight. The provenance hash encodes BOTH the computation AND its inputs:

```rust
provenance_hash(input_provenances: &[[u8; 16]], computation_id: &str) -> [u8; 16]
```

If the inputs change, the provenance hash changes. The caller computes a NEW key. They never look up the OLD key. There is no invalidation because there is nothing to invalidate — the question was never asked.

"Dirty" doesn't mean "this entry is stale." It means "this entry doesn't exist yet." Dirty and absent are the same state. Clean and present are the same state. The DirtyBitmap doesn't track staleness — it discovers freshness.

This is why `is_clean` is a one-liner: `self.index.contains_key(provenance)`. If you're asking about the right provenance, the answer is always "clean" if it exists. If inputs changed, you're asking about a different provenance, and the answer is "dirty" because it doesn't exist yet. The identity system makes invalid lookups impossible — you can't accidentally hit stale data because the key itself is different.

## The unified type

The three traits are views, not implementations. The unifying type is the provenance hash `[u8; 16]`. Every question the compiler asks the world reduces to: "what do you know about this 16-byte key?"

The responses form a hierarchy:
1. **Nothing** → dirty, not resident, no cache hit. Compute everything.
2. **It exists, but not on GPU** → clean, not resident, no immediate pointer. Reload from disk/pinned.
3. **It exists, on GPU** → clean, resident, pointer available. Route it (the 1us path).

NullWorld always answers (1). GpuStore answers (1), (2), or (3) depending on what it has. The compiler doesn't care which implementation it gets — it asks the same three questions and follows the response.

## The Fock boundary here

The liftability/scan isomorphism from the garden entry has a boundary: the Fock boundary, where computation becomes self-referential (identity depends on results that haven't been computed yet).

In the store, this manifests as: what happens when a node's provenance depends on another node whose provenance is being computed RIGHT NOW?

The provenance hash is `hash(input_provenances + computation_id)`. If an input provenance is unknown because that input is in-flight, you can't compute the dependent provenance. The identity function is blocked.

This is the store's Fock boundary: the provenance DAG must be evaluated bottom-up. Leaf provenances (raw data) are known. Internal provenances depend on their children. The execution plan already enforces this — it's a topological sort. But the boundary is real: you cannot speculatively compute a provenance for a node whose inputs are in-flight. Parallelism stops at the provenance frontier.

The NullWorld avoids this entirely by answering "dirty" to everything — it never needs to compute provenance because it never checks. The bootstrap problem and the Fock boundary are the same problem: you can't share what you haven't identified, and you can't identify what depends on things you're still computing.

## What this means for the compiler

The compiler's execution plan is simpler than Phase 2 suggested. It doesn't need three separate subsystems for provenance, dirty tracking, and residency. It needs ONE provenance computation per node, then ONE lookup against WorldState. The three traits are the API surface — the compiler asks three questions — but the implementation is one HashMap probe.

The cost model follows: the compiler's overhead per node is:
1. Compute provenance hash (BLAKE3: ~100ns for typical inputs)
2. Probe the HashMap (1 lookup, amortized O(1))
3. Branch on result (route pointer / reload / compute)

That's the full decision tree. Three sharing levels, one probe.

---

*The theory said three levels. The code says one HashMap. Both are right — the levels are real as compiler concepts, unified as implementation. The provenance hash is the meeting point: it makes the three questions collapse to one data structure because it encodes enough information that each question is just a different projection of the same lookup result.*
