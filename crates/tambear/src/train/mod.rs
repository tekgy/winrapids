//! tb.train — model training via GPU primitives.
//!
//! Each model decomposes into the same primitives that power the DataFrame engine:
//! accumulate (scatter), tiled (GEMM/distance), reduce, gather.
//!
//! ```no_run
//! use tambear::train;
//! # let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
//! # let y: Vec<f64> = vec![1.0, 2.0];
//! # let n: usize = 2; let d: usize = 2;
//!
//! // Linear regression: TiledEngine DotProduct + CPU Cholesky
//! let model = train::linear::fit(&x, &y, n, d).unwrap();
//! println!("R^2 = {:.4}", model.r_squared);
//! ```
//!
//! "Tam doesn't train. Tam accumulates."

pub mod linear;
pub mod logistic;
pub mod naive_bayes;
mod cholesky;
