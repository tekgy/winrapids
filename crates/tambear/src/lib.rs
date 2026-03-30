//! # tambear
//!
//! Sort-free DataFrame engine. Tam doesn't sort. Tam knows.
//!
//! The sort-free principle (manuscript 014): sort is O(n log n) with poor
//! GPU memory access patterns. For every operation that traditionally uses
//! sort — groupby, dedup, join, top-k — there exists an O(n) hash-based
//! alternative. The compiler enforces this: `sort` is emitted only for
//! explicit `sort_values()` and `rank()`.
//!
//! ## Core types
//!
//! - [`Frame`]: a GPU-resident DataFrame. Columns live on GPU until evicted.
//! - [`Column`]: a named, typed, GPU-resident column buffer.
//! - [`GroupIndex`]: pre-built row→group mapping. Built once, reused forever
//!   via provenance hash. Eliminates group-discovery cost from every groupby.
//!
//! ## The sort-free contract
//!
//! The compiler NEVER emits sort for:
//! - GroupBy → `hash_scatter` (this crate)
//! - Dedup → `hash_set` (this crate)
//! - Join → `hash_probe` (this crate)
//! - TopK → `selection` (this crate)
//!
//! Sort emitted only for: `sort_values`, `rank`.

pub mod frame;
pub mod group_index;
pub mod hash_scatter;
pub mod stats;

pub use frame::{Column, DType, Frame};
pub use group_index::GroupIndex;
pub use hash_scatter::HashScatterEngine;
