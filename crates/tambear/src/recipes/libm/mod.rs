//! Elementary transcendental functions — `exp`, `log`, `sin`, etc.
//!
//! Each recipe in this module is a named libm function expressed as a
//! composition of tambear primitives. They do NOT wrap `f64::exp`, `f64::ln`,
//! or any vendor libm; they implement the math from first principles with
//! documented polynomial approximations and documented error bounds.
//!
//! # Contract
//!
//! - Every recipe exposes `_strict`, `_compensated`, and
//!   `_correctly_rounded` entry points.
//! - Every recipe has an oracle test that compares the three strategies
//!   against each other and against `f64::{exp,ln,sin,...}` as a
//!   third-party reference.
//! - Every recipe handles the full f64 input range: finite normal values,
//!   subnormal values, zero, ±infinity, NaN, and the overflow/underflow
//!   thresholds specific to the function.
//!
//! # First pilot: `exp`
//!
//! `exp.rs` is the first recipe we lower end-to-end through the three
//! strategies. It demonstrates:
//! 1. Range reduction via primitives::hardware::frint + LOG2_E
//! 2. Polynomial evaluation via primitives::compensated::horner (strict)
//!    or compensated_horner (compensated) or DD-valued Horner (correctly
//!    rounded)
//! 3. Reconstruction via primitives::hardware::ldexp
//! 4. Validation via primitives::oracle + f64::exp cross-reference
//!
//! Subsequent recipes (`log`, `sin`, `cos`, `erf`, `gamma`) follow the
//! same template.

pub mod erf;
pub mod exp;
pub mod gamma;
pub mod log;
pub mod sin;
