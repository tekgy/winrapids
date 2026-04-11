// Flat catalog. One module per primitive. Alphabetical.
// Adding a new primitive = adding one folder + one line here.

pub mod log_sum_exp;

// Re-export all primitives at the crate root for flat access:
// `use tambear_primitives::log_sum_exp` not `use tambear_primitives::primitives::log_sum_exp`
pub use log_sum_exp::{log_sum_exp, log_sum_exp_pair};
