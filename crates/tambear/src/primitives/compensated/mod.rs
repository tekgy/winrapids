//! Compensated-arithmetic primitives.
//!
//! These are *error-free transformations* (EFTs) and their consumers: building
//! blocks that trade constant-factor overhead for dramatic accuracy gains.
//! They are the second layer of precision above raw hardware ops — fast
//! enough to deploy in hot loops, accurate enough to match correctly-rounded
//! references on almost any realistic input distribution.
//!
//! # What makes a compensated primitive
//!
//! A raw hardware add `fadd(a, b) = a + b` loses the rounding error. An EFT
//! like `two_sum(a, b) = (s, e)` returns both the rounded sum `s` AND an
//! exact representation `e` of what was lost, such that `a + b == s + e`
//! holds *exactly* in unrounded arithmetic. Downstream primitives can then
//! re-inject `e` into subsequent operations to recover accuracy.
//!
//! # Organization
//!
//! - `eft`: The error-free transformation kernels themselves (two_sum,
//!   fast_two_sum, two_product_fma, two_diff, two_square). Fixed cost,
//!   constant size output, composable.
//! - `sums`: Compensated reductions built on EFTs (kahan_sum, neumaier_sum,
//!   pairwise_sum). Each targets a different accuracy/speed point.
//! - `dot`: Compensated inner products (dot_2 from Rump-Ogita-Oishi,
//!   compensated_horner). These are the bridge between EFTs and polynomial
//!   evaluation, which is the foundation of every libm recipe.
//!
//! # Relationship to precision strategies
//!
//! Recipes tagged `#[precision(compensated)]` lower to these primitives
//! instead of the raw hardware arithmetic primitives. The recipe source is
//! the same tree; the lowering pass picks which set of terminals to emit.
//!
//! Recipes tagged `#[precision(strict)]` ignore this module and lower
//! directly to `primitives/hardware/`.
//!
//! Recipes tagged `#[precision(correctly_rounded)]` use this module as a
//! stepping stone but may also reach into `primitives/double_double/` and
//! `primitives/specialist/kulisch_accumulator/` for the last few bits.

pub mod eft;
pub mod sums;

pub use eft::{fast_two_sum, two_diff, two_product_fma, two_square, two_sum};
pub use sums::{kahan_sum, neumaier_sum, pairwise_sum};
