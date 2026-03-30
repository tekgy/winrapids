//! mktf — Rust implementation of the MKTF v4 binary format.
//!
//! Byte-identical to the Python implementation in fintek/trunk/backends/mktf/.
//! Target: 0.1ms/file writer (vs 0.87ms Python floor).
//!
//! Modules:
//!   format — constants, header layout, pack/unpack
//!   writer — crash-safe atomic writer
//!   reader — header-only, selective, full, status fast path
//!   filter — pre-filters (shuffle, delta) for compression pipeline

pub mod format;
pub mod writer;
pub mod reader;
pub mod filter;
mod bench;
