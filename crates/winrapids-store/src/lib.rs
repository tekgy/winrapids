//! Persistent GPU store with provenance tracking.
//!
//! The sharing optimizer's memory. Every buffer is tracked by its
//! provenance hash — the computational identity derived from its inputs
//! and the algorithm that produced it. Same inputs + same algorithm =
//! same provenance = skip computation.
//!
//! The store implements the zero-translation cache:
//!   result === cache entry === consumer input (pointer handoff, no copy)
//!
//! The compiler's execution plan is a pointer routing graph through this store.
//! Computation is the exceptional case (cache miss). Pointer handoff is the
//! common case (865x from provenance reuse).
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  Compiler asks: "provenance X computed?"          │
//! │                                                  │
//! │  GpuStore.lookup(X)                              │
//! │  ├── Hit: return device pointer  (0 computation) │
//! │  └── Miss: caller computes, then register(X)     │
//! │                                                  │
//! │  On VRAM pressure: cost-aware LRU eviction       │
//! │  └── Cheap-to-recompute, large buffers go first  │
//! └──────────────────────────────────────────────────┘
//! ```

pub mod header;
pub mod provenance;
pub mod store;
pub mod world;

pub use header::{BufferHeader, BufferPtr, DType, EvictedEntry, Location};
pub use provenance::{data_provenance, prov_hex, provenance_hash};
pub use store::{GpuStore, StoreStats};
pub use world::{DirtyBitmap, NullWorld, ProvenanceCache, ResidencyMap, WorldState};
