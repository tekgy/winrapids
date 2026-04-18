//! Price percentiles — multiple quantile queries from one sketch pass.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! quantile-sketch primitive (Tier 1) plus a single multi-query pass.
//! See `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\column-graph.md` the SIP hour header carries
//! nine price percentiles at the standard breakpoints (F11..F19):
//!
//! ```text
//! price_usd_p01, p05, p10, p25, p50, p75, p90, p95, p99
//! ```
//!
//! All nine come from the same underlying sketch in one pass — far
//! cheaper than nine separate sort-based percentile computations.
//!
//! # Composition
//!
//! - **QuantileSketch primitive** (Tier 1) — KLL by default; t-digest
//!   selectable via `using(sketch: ...)` for better tail accuracy at
//!   q=0.01 and q=0.99.
//! - **Multi-query call** to `sketch.quantiles(&qs)` — one pass over
//!   the sketch state, returning all requested percentiles.
//!
//! # Default parameters
//!
//! - `epsilon` — 0.005 (half a percent rank error). SIP-suitable.
//!   Override per call for tighter / looser accuracy.
//! - `sketch` — KLL by default. t-digest worth choosing when accurate
//!   tail percentiles (p01, p99) matter more than middle-range
//!   accuracy.
//!
//! # NaN/Inf policy
//!
//! Inherited from the sketch: non-finite values are skipped at
//! insertion. If `prices` has no finite values, all returned
//! percentiles are NaN.

use crate::primitives::specialist::quantile_sketch::{
    QuantileSketch, SketchAlgorithm,
};

/// The nine SIP standard percentile breakpoints.
pub const SIP_STANDARD_QUANTILES: [f64; 9] = [
    0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
];

/// Compute the nine SIP standard percentiles from a slice of prices.
///
/// Locked-default: KLL sketch, ε = 0.005. For per-call overrides on
/// sketch algorithm or epsilon, use `price_percentiles_with`.
///
/// Returns a fixed-size array `[p01, p05, p10, p25, p50, p75, p90, p95, p99]`.
/// Empty input → all NaN.
pub fn price_percentiles_sip(prices: &[f64]) -> [f64; 9] {
    let qs = price_percentiles_with(
        prices,
        &SIP_STANDARD_QUANTILES,
        0.005,
        SketchAlgorithm::DEFAULT,
    );
    let mut out = [f64::NAN; 9];
    out.copy_from_slice(&qs);
    out
}

/// Compute arbitrary percentiles with explicit sketch and epsilon.
///
/// `qs` is the list of quantile breakpoints to query, each in `[0, 1]`.
/// Returns the corresponding percentile values in the same order as
/// `qs`. Empty `prices` → vector of NaN with the same length as `qs`.
///
/// # Panics
///
/// Panics if any `q` in `qs` is non-finite or outside `[0, 1]`.
pub fn price_percentiles_with(
    prices: &[f64],
    qs: &[f64],
    epsilon: f64,
    algorithm: SketchAlgorithm,
) -> Vec<f64> {
    if prices.is_empty() {
        return vec![f64::NAN; qs.len()];
    }
    match algorithm {
        SketchAlgorithm::Kll => {
            let mut sk = crate::primitives::specialist::KllSketch::new(epsilon);
            sk.add_slice(prices);
            sk.quantiles(qs)
        }
        SketchAlgorithm::Gk => {
            let mut sk = crate::primitives::specialist::GkSketch::new(epsilon);
            sk.add_slice(prices);
            sk.quantiles(qs)
        }
        SketchAlgorithm::Tdigest => {
            let mut sk = crate::primitives::specialist::TdigestSketch::new(epsilon);
            sk.add_slice(prices);
            sk.quantiles(qs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
        let diff = (got - want).abs();
        assert!(
            diff < tol,
            "{label}: got {got}, want {want}, diff {diff} > tol {tol}"
        );
    }

    #[test]
    fn empty_returns_all_nan() {
        let p = price_percentiles_sip(&[]);
        for v in p {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn uniform_prices_match_breakpoints() {
        // Prices 0..10000 uniform. Percentile p_k% should be ~ k·100.
        let prices: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        let p = price_percentiles_sip(&prices);
        let expected = [100.0, 500.0, 1000.0, 2500.0, 5000.0, 7500.0, 9000.0, 9500.0, 9900.0];
        for (i, (&got, &want)) in p.iter().zip(expected.iter()).enumerate() {
            // 0.5% rank error × 10000 = 50 absolute. Allow generous tol.
            assert_close(got, want, 200.0, &format!("breakpoint #{i}"));
        }
    }

    #[test]
    fn sip_standard_returns_9_values() {
        let prices: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let p = price_percentiles_sip(&prices);
        assert_eq!(p.len(), 9);
        // p01 < p05 < ... < p99 monotonic.
        for i in 1..9 {
            assert!(p[i] >= p[i - 1] - 1.0, "non-monotonic at #{i}: {p:?}");
        }
    }

    #[test]
    fn arbitrary_quantiles() {
        let prices: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let qs = vec![0.001, 0.123, 0.456, 0.789, 0.999];
        let vs = price_percentiles_with(&prices, &qs, 0.01, SketchAlgorithm::Kll);
        assert_eq!(vs.len(), qs.len());
        for i in 1..vs.len() {
            assert!(vs[i] >= vs[i - 1] - 1.0, "non-monotonic at {i}");
        }
    }

    #[test]
    fn three_sketches_in_same_neighborhood() {
        // Three sketches at the same ε on a noisy near-linear series.
        // They will not agree to high precision (each algorithm has
        // its own bias structure — KLL randomized, GK worst-case
        // bound at tails, t-digest tail-accurate but middle-loose).
        // The test asserts they're in the same neighborhood, defined
        // by an absolute slack tied to ε·N on the value scale (~50
        // absolute units for ε=0.01, N=5000 on this series).
        let prices: Vec<f64> = (0..5000)
            .map(|i| (i as f64) + ((i as f64) * 0.31).sin() * 50.0)
            .collect();
        let qs = [0.05, 0.5, 0.95];
        let abs_tol = 250.0; // ~5% of the value range for this series

        let kll = price_percentiles_with(&prices, &qs, 0.01, SketchAlgorithm::Kll);
        let gk = price_percentiles_with(&prices, &qs, 0.01, SketchAlgorithm::Gk);
        let td = price_percentiles_with(&prices, &qs, 0.01, SketchAlgorithm::Tdigest);

        for i in 0..qs.len() {
            let pivot = kll[i];
            assert!(
                (gk[i] - pivot).abs() < abs_tol,
                "GK at q={}: {} vs KLL {} (tol {})",
                qs[i], gk[i], pivot, abs_tol
            );
            assert!(
                (td[i] - pivot).abs() < abs_tol,
                "tdigest at q={}: {} vs KLL {} (tol {})",
                qs[i], td[i], pivot, abs_tol
            );
        }
    }

    #[test]
    fn skips_non_finite_prices() {
        let mut prices: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        prices.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        let p = price_percentiles_sip(&prices);
        // Should match the no-NaN/Inf case.
        let p_clean = price_percentiles_sip(&(0..1000).map(|i| i as f64).collect::<Vec<_>>());
        for i in 0..9 {
            assert_close(p[i], p_clean[i], 1.0, &format!("with NaN match clean #{i}"));
        }
    }
}
