//! Hawkes-process self-exciting intensity at a target time.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! Kulisch-backed exponential-kernel scatter sum + scalar arithmetic.
//! See `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! For a 1-D Hawkes process with exponential kernel, the conditional
//! intensity at time `t_end` given past event times `t[0..n]` (each
//! `t[i] ≤ t_end`) is:
//!
//! ```text
//! λ(t_end)  =  μ  +  α · Σ_{i: t[i] ≤ t_end} exp(−β · (t_end − t[i]))
//! ```
//!
//! where:
//! - `μ` is the background (Poisson) intensity
//! - `α` is the excitation magnitude per past event
//! - `β` is the decay rate (1/half-life)
//!
//! Each past event contributes a decaying excitation that adds to the
//! current intensity. With `α < β`, the process is stationary; with
//! `α ≥ β`, it explodes (caller's responsibility to enforce).
//!
//! # Reference
//!
//! Hawkes, A. G. (1971). Spectra of some self-exciting and mutually
//! exciting point processes. *Biometrika* 58(1): 83–90.
//!
//! # Composition
//!
//! - **Kulisch-backed exponential-kernel scatter** — for each past
//!   event, compute `exp(−β · (t_end − t_i))` and accumulate. Lowers
//!   to `accumulate(timestamps, Grouping::All, Op::Add,
//!   expr=exp(-β·(t_end - v)))` once the parameterized-Custom-expr
//!   atom path is wired; today computed via a local Kulisch
//!   accumulator.
//! - **Scalar arithmetic** — `μ + α · sum`.
//!
//! # NaN/Inf policy
//!
//! - Empty event list → returns `μ` (no excitation contribution).
//! - Events with non-finite timestamps → silently skipped (Kulisch
//!   `is_finite` semantics inherited).
//! - Events at `t[i] > t_end` are silently skipped (future events
//!   cannot influence current intensity).
//! - Negative time gaps from clock skew (events strictly equal to
//!   `t_end`) contribute the maximum kernel weight (`exp(0) = 1`).
//!
//! # Default parameters
//!
//! All three Hawkes parameters are caller-supplied. SIP per
//! signal-compute-spec uses:
//! - `μ` ≈ avg events per bucket (estimated upstream from prefix sums)
//! - `α` = 0.1 (default excitation)
//! - `β` = 1 / 60_000_000 (60-millisecond half-life, expressed as 1/ns)

/// Hawkes intensity at `t_end_ns`, given past event timestamps and
/// process parameters.
///
/// `event_times_ns` is the sequence of past event times in nanoseconds.
/// `t_end_ns` is the time at which intensity is evaluated; events at
/// or before `t_end_ns` contribute.
/// `mu` is the background intensity (≥ 0).
/// `alpha` is the excitation magnitude per past event (≥ 0).
/// `beta_per_ns` is the kernel decay rate in 1/ns (≥ 0).
///
/// # Panics
///
/// Panics if `mu`, `alpha`, or `beta_per_ns` are non-finite or negative.
pub fn hawkes_intensity(
    event_times_ns: &[i64],
    t_end_ns: i64,
    mu: f64,
    alpha: f64,
    beta_per_ns: f64,
) -> f64 {
    assert!(
        mu.is_finite() && mu >= 0.0,
        "hawkes_intensity: mu must be finite and >= 0, got {mu}"
    );
    assert!(
        alpha.is_finite() && alpha >= 0.0,
        "hawkes_intensity: alpha must be finite and >= 0, got {alpha}"
    );
    assert!(
        beta_per_ns.is_finite() && beta_per_ns >= 0.0,
        "hawkes_intensity: beta_per_ns must be finite and >= 0, got {beta_per_ns}"
    );

    if event_times_ns.is_empty() || alpha == 0.0 {
        return mu;
    }

    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;
    let mut sum = KulischAccumulator::new();

    for &t in event_times_ns {
        if t > t_end_ns {
            continue; // future event, doesn't influence intensity at t_end
        }
        let dt_ns = (t_end_ns - t) as f64;
        let weight = (-beta_per_ns * dt_ns).exp();
        if weight.is_finite() {
            sum.add_f64(weight);
        }
    }

    mu + alpha * sum.to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_events_returns_mu() {
        assert_eq!(hawkes_intensity(&[], 1_000_000_000, 0.5, 0.1, 1e-6), 0.5);
    }

    #[test]
    fn alpha_zero_returns_mu() {
        let events = vec![100, 200, 300, 400];
        assert_eq!(hawkes_intensity(&events, 500, 0.7, 0.0, 1e-6), 0.7);
    }

    #[test]
    fn single_event_at_t_end_max_excitation() {
        // Event exactly at t_end → exp(0) = 1 → intensity = μ + α
        let lam = hawkes_intensity(&[1000], 1000, 0.5, 0.3, 1e-6);
        assert!((lam - 0.8).abs() < 1e-12);
    }

    #[test]
    fn distant_event_decayed_to_negligible() {
        // Event 10s in the past with β=1e-9 (1ns half-life ~ 0.7s).
        // exp(-1e-9 · 1e10) = exp(-10) ≈ 4.54e-5. With α=1, intensity ≈ μ + 4.54e-5.
        let mu = 0.5;
        let lam = hawkes_intensity(&[0], 10_000_000_000, mu, 1.0, 1e-9);
        let expected_excitation = (-10.0f64).exp();
        assert!((lam - mu - expected_excitation).abs() < 1e-10);
    }

    #[test]
    fn future_events_ignored() {
        // Events at t > t_end should not contribute.
        let events = vec![100, 500, 800, 1500, 2000];
        let lam_with_future = hawkes_intensity(&events, 1000, 0.5, 0.1, 0.0);
        // β=0 means exp(0)=1 for all included events. 3 events at or before 1000.
        // λ = 0.5 + 0.1 · 3 = 0.8
        assert!((lam_with_future - 0.8).abs() < 1e-12);
    }

    #[test]
    fn beta_zero_uniform_weight() {
        // β=0 → kernel = exp(0) = 1 always; sum = n_events.
        let events = vec![1, 2, 3, 4, 5];
        let lam = hawkes_intensity(&events, 100, 0.0, 1.0, 0.0);
        assert!((lam - 5.0).abs() < 1e-12);
    }

    #[test]
    fn intensity_monotonic_in_alpha() {
        // For fixed events / time / β, larger α gives larger intensity.
        let events = vec![100, 200, 300];
        let l1 = hawkes_intensity(&events, 400, 0.0, 0.1, 0.0);
        let l2 = hawkes_intensity(&events, 400, 0.0, 0.5, 0.0);
        assert!(l2 > l1);
    }

    #[test]
    fn intensity_monotonic_in_mu() {
        let events = vec![100, 200];
        let l1 = hawkes_intensity(&events, 300, 0.1, 0.05, 1e-9);
        let l2 = hawkes_intensity(&events, 300, 0.5, 0.05, 1e-9);
        assert!(l2 > l1);
        // The difference should be exactly μ_2 - μ_1.
        assert!(((l2 - l1) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn intensity_decays_in_beta() {
        // Larger β → faster decay → smaller intensity for past events.
        let events = vec![100];
        let l_slow = hawkes_intensity(&events, 1_000_000, 0.0, 1.0, 1e-9);
        let l_fast = hawkes_intensity(&events, 1_000_000, 0.0, 1.0, 1e-3);
        assert!(l_slow > l_fast);
    }

    #[test]
    #[should_panic(expected = "mu")]
    fn panics_on_negative_mu() {
        let _ = hawkes_intensity(&[100], 200, -0.5, 0.1, 1e-6);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn panics_on_negative_alpha() {
        let _ = hawkes_intensity(&[100], 200, 0.5, -0.1, 1e-6);
    }
}
