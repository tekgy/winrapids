// Flat catalog. One module per primitive. Alphabetical.
// Adding a new primitive = adding one folder + one line here.

pub mod recipe;
pub mod lehmer_mean;
pub mod log_sum_exp;
pub mod mean_arithmetic;
pub mod mean_exponential_moving;
pub mod mean_geometric;
pub mod mean_harmonic;
pub mod mean_quadratic;
pub mod mean_trimmed;
pub mod mean_weighted;
pub mod mean_winsorized;
pub mod nan_guard;
pub mod prefix_scan;
pub mod semiring;
pub mod softmax;

// Re-export all primitives flat at crate root.
pub use lehmer_mean::{lehmer_mean, mean_contraharmonic};
pub use log_sum_exp::{log_sum_exp, log_sum_exp_pair};
pub use mean_arithmetic::{mean_arithmetic, MeanAccumulator};
pub use mean_exponential_moving::{mean_exponential_moving, mean_exponential_moving_period};
pub use mean_geometric::mean_geometric;
pub use mean_harmonic::mean_harmonic;
pub use mean_quadratic::mean_quadratic;
pub use mean_trimmed::mean_trimmed;
pub use mean_weighted::mean_weighted;
pub use mean_winsorized::mean_winsorized;
pub use nan_guard::{nan_min, nan_max, has_nan, has_non_finite, finite_only, sorted_total, sorted_finite};
pub use prefix_scan::{prefix_scan_inclusive, prefix_scan_exclusive, reduce, prefix_scan_segmented};
pub use semiring::{Semiring, Additive, TropicalMinPlus, TropicalMaxPlus, LogSumExp, Boolean, MaxTimes};
pub use softmax::{softmax, log_softmax};
pub use recipe::{Step, GroupingKind, ExprKind, OpKind, Recipe,
    MEAN_ARITHMETIC, MEAN_GEOMETRIC, MEAN_HARMONIC, MEAN_QUADRATIC,
    VARIANCE, CUMSUM};
