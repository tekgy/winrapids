//! # TBS — The tambear language
//!
//! One language. Used everywhere:
//! - As the φ (transform) expression before accumulate
//! - As the gather formula after accumulate
//! - As the user's script in the IDE
//! - As the input to the .tam compiler
//!
//! Zero dependencies. The language is self-contained.

pub mod expr;
pub mod parser;
pub mod using;
pub mod advice;
// pub mod lint;  // TODO: depends on proof system, migrate later

// Re-export key types at crate root
pub use expr::Expr;
pub use parser::{TbsChain, TbsStep, TbsArg, TbsName, TbsValue};
pub use using::{UsingBag, UsingValue};
pub use advice::TbsStepAdvice;
