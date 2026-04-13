//! IEEE 754 hardware terminal operations.
//!
//! Every primitive in this module maps to a single hardware instruction on
//! x86, ARM, PTX, and SPIR-V targets. These are the ground floor of the
//! primitive layer — nothing below them is visible at the recipe level.
//!
//! # Conventions
//!
//! - **Every function is `#[inline(always)]`**. The primitive layer must not
//!   introduce function-call overhead in hot numerical loops. LLVM inlines
//!   them and emits the hardware instruction directly.
//! - **IEEE 754-2019 semantics.** Where Rust's standard library diverges from
//!   IEEE 754 (notably `f64::min`/`f64::max` which do NOT propagate NaN), we
//!   provide the standards-compliant version. Recipes must call `primitives::fmin`,
//!   NEVER `f64::min` directly.
//! - **Naming.** We use `f*` prefixes (fadd, fmul, fmadd, ...) to signal that
//!   these are terminal floating-point primitives, not ordinary Rust functions.
//!   This makes violations easy to grep for: any recipe that contains a raw
//!   `a + b` or `a.max(b)` on floats is bypassing the primitive layer.
//! - **Testing.** Each primitive has a test file that verifies the specific
//!   IEEE 754 semantics, including NaN propagation, ±infinity behavior,
//!   subnormal handling, and boundary cases.
//!
//! # The complete list
//!
//! Arithmetic:       `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`
//! Fused:            `fmadd`, `fmsub`, `fnmadd`, `fnmsub`
//! Unary:            `fabs`, `fneg`, `fcopysign`
//! Min/max:          `fmin`, `fmax` — NaN-propagating (IEEE 754-2019)
//! Comparison:       `fcmp_eq`, `fcmp_lt`, `fcmp_le`, `fcmp_gt`, `fcmp_ge`
//! Classification:   `is_nan`, `is_inf`, `is_finite`, `signbit`
//! Rounding:         `frint`, `ffloor`, `fceil`, `ftrunc`
//! Scale:            `ldexp`, `frexp`

mod arithmetic;
mod fused;
mod unary;
mod minmax;
mod compare;
mod classify;
mod rounding;
mod scale;

pub use arithmetic::{fadd, fsub, fmul, fdiv, fsqrt};
pub use fused::{fmadd, fmsub, fnmadd, fnmsub};
pub use unary::{fabs, fneg, fcopysign};
pub use minmax::{fmin, fmax};
pub use compare::{fcmp_eq, fcmp_lt, fcmp_le, fcmp_gt, fcmp_ge};
pub use classify::{is_nan, is_inf, is_finite, signbit};
pub use rounding::{frint, ffloor, fceil, ftrunc};
pub use scale::{ldexp, frexp};
