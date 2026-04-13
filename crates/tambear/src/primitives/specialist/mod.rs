//! Specialist primitives — slow but maximally accurate.
//!
//! These live outside the hot path. Their purpose is to serve as oracles
//! and reference implementations that the compensated and double-double
//! primitives are validated against. A recipe should not reach for these
//! in a production code path; it reaches for them at test time to prove
//! that the faster strategies give the right answer.
//!
//! # Inventory
//!
//! - `kulisch_accumulator`: an ~4352-bit signed fixed-point accumulator for
//!   exact summation of f64 values. No rounding, ever. Used as the gold
//!   standard when we need to know "what is `sum(xs)` really?"
//!
//! # Why separate from `compensated/` and `double_double/`
//!
//! The compensated primitives trade a constant factor of flops for bounded
//! error. The double-double type trades a larger constant factor for ~106
//! bits of precision. The specialist primitives trade a *much* larger
//! constant factor (often 10-100×) for **exact** results. They are
//! categorically different in cost and use.
//!
//! Recipes tagged `#[precision(correctly_rounded)]` rarely use these at
//! runtime — the double-double arithmetic is usually enough. But the
//! correctness *tests* for those recipes compare against the Kulisch
//! oracle to verify the final answer is correctly rounded.

pub mod kulisch_accumulator;
pub mod sum_k;

pub use kulisch_accumulator::KulischAccumulator;
pub use sum_k::{sum_2, sum_3, sum_4, sum_k};
