//! World state traits for the compiler.
//!
//! The compiler's execution plan probes three aspects of the world:
//!
//! 1. **Provenance**: Has this computation been done before? (865x savings)
//! 2. **Dirty bitmap**: Has the input data changed since last compute? (skip stale checks)
//! 3. **Residency**: Is the result currently on GPU? (pointer handoff vs recompute)
//!
//! GpuStore implements all three. For testing (E04 baseline), use NullWorld
//! which always misses — forcing full computation.
//!
//! The compiler takes `&mut dyn WorldState` and the right implementation
//! is injected at construction time.

use crate::header::BufferPtr;

// ────────────────────────────────────────────────────────────
// Individual capability traits
// ────────────────────────────────────────────────────────────

/// Provenance-based result cache.
/// The 865x optimization: skip computation when inputs + algorithm match.
pub trait ProvenanceCache {
    /// Look up a provenance hash. Returns the buffer pointer if found.
    fn provenance_get(&mut self, provenance: &[u8; 16]) -> Option<BufferPtr>;

    /// Register a computed result.
    fn provenance_put(&mut self, provenance: [u8; 16], ptr: BufferPtr, cost_us: f32);
}

/// Input staleness tracking.
/// When raw data changes, downstream computations become stale.
pub trait DirtyBitmap {
    /// Returns true if the computation's inputs have NOT changed
    /// since it was last computed. False = needs recompute.
    fn is_clean(&self, provenance: &[u8; 16]) -> bool;
}

/// GPU residency tracking.
/// Knows whether a buffer is currently in VRAM (hot) or has been spilled.
pub trait ResidencyMap {
    /// Returns true if the buffer is currently GPU-resident.
    fn is_resident(&self, provenance: &[u8; 16]) -> bool;

    /// Returns the device pointer if GPU-resident.
    fn resident_pointer(&self, provenance: &[u8; 16]) -> Option<BufferPtr>;
}

// ────────────────────────────────────────────────────────────
// Combined world state
// ────────────────────────────────────────────────────────────

/// The compiler's unified view of the execution world.
/// GpuStore implements this. NullWorld is the E04 baseline.
pub trait WorldState: ProvenanceCache + DirtyBitmap + ResidencyMap {}

/// Blanket implementation: anything that implements all three is a WorldState.
impl<T: ProvenanceCache + DirtyBitmap + ResidencyMap> WorldState for T {}

// ────────────────────────────────────────────────────────────
// Null world state (E04 baseline: compute everything)
// ────────────────────────────────────────────────────────────

/// Null world state: provenance always misses, everything dirty,
/// nothing resident. Forces full computation — the E04 baseline.
///
/// Use this when testing the compiler in isolation, or when
/// the persistent store isn't initialized yet.
pub struct NullWorld;

impl ProvenanceCache for NullWorld {
    fn provenance_get(&mut self, _provenance: &[u8; 16]) -> Option<BufferPtr> {
        None
    }

    fn provenance_put(&mut self, _provenance: [u8; 16], _ptr: BufferPtr, _cost_us: f32) {
        // Discard — null world remembers nothing
    }
}

impl DirtyBitmap for NullWorld {
    fn is_clean(&self, _provenance: &[u8; 16]) -> bool {
        false // Everything dirty: always recompute
    }
}

impl ResidencyMap for NullWorld {
    fn is_resident(&self, _provenance: &[u8; 16]) -> bool {
        false // Nothing resident: no pointer handoff shortcuts
    }

    fn resident_pointer(&self, _provenance: &[u8; 16]) -> Option<BufferPtr> {
        None
    }
}
