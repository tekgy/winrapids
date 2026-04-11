// Flat catalog. One module per primitive. Alphabetical.
// Adding a new primitive = adding one folder + one line here.

pub mod log_sum_exp;
pub mod nan_guard;
pub mod prefix_scan;
pub mod semiring;
pub mod softmax;

// Re-export all primitives flat at crate root.
pub use log_sum_exp::{log_sum_exp, log_sum_exp_pair};
pub use nan_guard::{nan_min, nan_max, has_nan, has_non_finite, finite_only, sorted_total, sorted_finite};
pub use prefix_scan::{prefix_scan_inclusive, prefix_scan_exclusive, reduce, prefix_scan_segmented};
pub use semiring::{Semiring, Additive, TropicalMinPlus, TropicalMaxPlus, LogSumExp, Boolean, MaxTimes};
pub use softmax::{softmax, log_softmax};
