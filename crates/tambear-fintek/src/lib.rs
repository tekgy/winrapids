//! # tambear-fintek — Bridge crate for fintek leaves
//!
//! Drop-in native compute backends for fintek's `trunk-rs` leaves, built
//! on tambear's math primitives. Each function here replaces the CPU-only
//! implementation in `R:/fintek/crates/trunk-rs/src/leaves/<leaf>.rs`.
//!
//! ## Design
//!
//! - **Pure functions**: inputs are `&[f64]` slices (bin-level data),
//!   outputs are `Vec<f64>` (one per output column). No dependency on
//!   fintek's `Leaf` trait or `ExecutionContext` — fintek calls these
//!   functions inside its own `execute()` methods.
//! - **Per-bin processing**: every leaf processes a single variable-length
//!   bin. Fintek iterates over bins; this crate handles the math.
//! - **Bit-perfect where possible**, tolerance documented otherwise.
//! - **NaN handling**: matches fintek's convention (NaN for degenerate bins).
//!
//! ## Module layout
//!
//! - [`family1_distribution`] — distribution, normality, shannon_entropy
//! - [`family2_transforms`] — pointwise maps (returns, log, sqrt, etc.)
//! - [`family3_bin_aggregates`] — ohlcv, counts, validity
//! - [`family4_time_series`] — ar_model, autocorrelation, arma, arima
//! - [`family5_stationarity`] — adf, kpss, ljung_box wrapped
//! - [`family6_spectral`] — fft_spectral, welch, cepstrum
//! - [`family7_wavelets`] — CWT tower (K02P03C01), 12 variants: M×{16,32,64,128} × 3 strategies
//! - [`family9_volatility`] — garch, realized_vol, jump_detection
//! - [`family10_nonlinear`] — sample_entropy, dfa, hurst, lyapunov
//! - [`family13_dim_reduction`] — pca, ssa
//! - [`family14_topological`] — persistent_homology
//!
//! GAP leaves (families with missing primitives like Kalman, ICA,
//! Hawkes, DTW, etc.) are not yet implemented — see tasks #137-#146.

pub mod family1_distribution;
pub mod family2_transforms;
pub mod family3_bin_aggregates;
pub mod family4_time_series;
pub mod family5_stationarity;
pub mod family6_spectral;
pub mod family7_wavelets;
pub mod family8_correlation;
pub mod family9_volatility;
pub mod family10_nonlinear;
pub mod family11_tick_microstructure;
pub mod family12_causality_info;
pub mod family13_dim_reduction;
pub mod family14_topological;
pub mod family15_manifold_topology;
pub mod family16_extremes;

/// Data quality primitives re-exported from tambear::data_quality.
///
/// Fintek leaf executors can call these before running expensive math:
/// ```rust,ignore
/// use tambear_fintek::data_quality;
/// if !data_quality::fft_is_valid(bin_prices, Some(bin_timestamps)) {
///     return LeafOutput::nan();
/// }
/// ```
/// Every function here is O(n) cadence-agnostic: takes a slice, returns a scalar.
pub mod data_quality {
    pub use tambear::data_quality::{
        // Size
        tick_count, nyquist_bins, effective_sample_size,
        // Diversity
        unique_prices, symbolic_diversity, unique_ordinal_3,
        // Variability
        price_cv, split_variance_ratio,
        // Sampling regularity
        sampling_regularity_cv, longest_gap_ratio, coverage_ratio,
        // Structure / trend
        trend_r2, has_trend, acf_decay_exponent,
        // ACF values
        lag1_autocorrelation, lag_k_autocorrelation, acf_lag10, acf_abs_lag1,
        // Stationarity proxy
        is_stationary_adf_05,
        // Jump / vol clustering
        jump_ratio_proxy, has_vol_clustering,
        // Summary struct
        DataQualitySummary,
        // Per-family validity composites (all 18 families from BINNED_METHODS_LIST)
        fft_is_valid, garch_is_valid,
        rank_based_is_valid, permutation_entropy_3_is_valid, sample_entropy_is_valid,
        wavelet_is_valid, time_series_is_valid, fractal_is_valid,
        entropy_complexity_is_valid, chaos_is_valid, manifold_is_valid,
        distance_is_valid, causality_is_valid, regime_detection_is_valid,
        continuous_time_is_valid, decomposition_is_valid,
        statistical_tests_is_valid, correlation_is_valid,
    };
}

/// Shared output type for multi-column leaves.
///
/// Fintek leaves often emit several DO columns per bin. Rather than returning
/// `Vec<Vec<f64>>`, most leaves use named fields via a `...Result` struct.
/// A helper to build a dense matrix-of-outputs per bin list.
pub fn per_bin_to_columns(per_bin: &[Vec<f64>], n_cols: usize) -> Vec<Vec<f64>> {
    let n_bins = per_bin.len();
    let mut cols: Vec<Vec<f64>> = (0..n_cols).map(|_| vec![f64::NAN; n_bins]).collect();
    for (i, row) in per_bin.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if j < n_cols { cols[j][i] = v; }
        }
    }
    cols
}
