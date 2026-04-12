//! # tambear-tam-ir
//!
//! The `.tam` intermediate representation.
//!
//! ## What lives here
//!
//! - **AST types** (`ast` module) — `Program`, `KernelDef`, `FuncDef`, `Op`,
//!   `Reg`, `Type`, etc. Zero dependencies except `std`.
//! - **Text printer** (`print` module) — AST → `.tam` text string.
//! - **Text parser** (`parse` module) — `.tam` text → AST.
//! - **Verifier** (`verify` module) — type-check, SSA invariant, use-def check.
//! - **CPU interpreter** (`interp` module) — evaluate a `Program` on concrete
//!   inputs. The reference oracle.
//!
//! ## Invariants enforced here
//!
//! - I3: No FMA. The AST has no `fma` op. Backends that lower `fadd(fmul(a,b),c)`
//!   must not silently fuse it.
//! - I4: No fp reordering. The interpreter executes ops in program order.
//! - I7: Every kernel is loop (accumulate) + reduce_block (gather).
//!
//! ## What does NOT live here
//!
//! - PTX emission → `tambear-tam-ptx`
//! - SPIR-V emission → `tambear-tam-spirv`
//! - tambear-libm `.tam` sources → `tambear-libm`
//! - Cross-backend test harness → `tambear-tam-test-harness`

pub mod ast;
pub mod print;
pub mod parse;
pub mod verify;
pub mod interp;

/// Test fixtures: pre-built Programs for cross-module tests.
#[cfg(test)]
pub mod fixtures;

/// Property tests: 10,000-program round-trip (campsite 1.8).
#[cfg(test)]
pub mod proptest;
