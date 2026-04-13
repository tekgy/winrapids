//! Double-double arithmetic: ~106-bit precision via unevaluated pairs of f64.
//!
//! A `DoubleDouble` is an unevaluated sum `hi + lo` where `hi` and `lo` are
//! non-overlapping f64 values (specifically, `|lo| <= ulp(hi) / 2`). The pair
//! carries approximately 106 bits of precision — about 32 decimal digits —
//! compared to the 53 bits (16 digits) of a single f64.
//!
//! This is the intermediate layer between compensated EFTs (which handle
//! specific operations like sum and product) and the ~4000-bit Kulisch
//! accumulator (which is the exact oracle). It's fast enough to run in
//! every element of a hot loop — typical cost is 20-30 flops per op — and
//! accurate enough that a correctly-rounded libm recipe can use it as its
//! working type without needing to reach further.
//!
//! # Algorithm sources
//!
//! - Shewchuk (1997), *Adaptive Precision Floating-Point Arithmetic and Fast
//!   Robust Geometric Predicates* — the foundational treatment. Our `dd_add`
//!   (non-overlap preserving) and `dd_mul` follow his expansion algorithms.
//! - Hida, Li, Bailey (2000), *Quad-Double Arithmetic: Algorithms,
//!   Implementation, and Application* — the QD library. Our `dd_div` and
//!   `dd_sqrt` follow QD's Newton iteration scheme.
//! - Joldes, Muller, Popescu (2017), *Tight and rigorous error bounds for
//!   basic building blocks of double-word arithmetic* — the current
//!   state-of-the-art reference for DD error analysis.
//!
//! # Usage
//!
//! Recipes tagged `#[precision(correctly_rounded)]` use `DoubleDouble` as
//! their working type. Arithmetic is expressed as method calls (`a + b`,
//! `a * b`) via the `Add`/`Mul`/`Sub`/`Div` trait implementations, so the
//! recipe source reads naturally.
//!
//! Recipes tagged `#[precision(compensated)]` use raw `f64` + the
//! `compensated` primitives, avoiding the double-double overhead.
//!
//! # Non-goals
//!
//! - **Not IEEE 754.** `DoubleDouble` does not implement IEEE rounding
//!   modes precisely; the operations are correctly rounded to ~106 bits
//!   but not to a well-defined ulp-size in a double-word number system.
//! - **Not a replacement for arbitrary precision.** For proofs or
//!   certifying correctly-rounded results we use the Kulisch accumulator
//!   or external mpmath references — see `primitives/specialist/`.

pub mod ops;
pub mod ty;

pub use ty::DoubleDouble;
