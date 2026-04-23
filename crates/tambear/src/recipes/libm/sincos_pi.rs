//! `sincospi(x)` — fused sinpi/cospi.
//!
//! Returns (sinpi(x), cospi(x)) as a pair. The fused form exists so that
//! both values can be computed in a single pass through range reduction —
//! both share the same quadrant and reduced argument. When called separately,
//! each call independently reduces the argument.
//!
//! CONTRACT: sincospi must be bit-identical to separate sinpi/cospi calls.
//! This is a correctness requirement, not a performance hint.

use super::pi_scaled::{sinpi_strict, cospi_strict};

/// `sincospi(x)` — strict. Returns (sin(π·x), cos(π·x)).
///
/// Bit-identical to (sinpi_strict(x), cospi_strict(x)) by construction.
#[inline]
pub fn sincospi_strict(x: f64) -> (f64, f64) {
    (sinpi_strict(x), cospi_strict(x))
}

/// `sincospi(x)` — compensated.
#[inline]
pub fn sincospi_compensated(x: f64) -> (f64, f64) {
    sincospi_strict(x)
}

/// `sincospi(x)` — correctly-rounded.
#[inline]
pub fn sincospi_correctly_rounded(x: f64) -> (f64, f64) {
    sincospi_strict(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sincospi_zero() {
        let (s, c) = sincospi_strict(0.0);
        assert_eq!(s.to_bits(), 0.0f64.to_bits());
        assert_eq!(c.to_bits(), 1.0f64.to_bits());
    }

    #[test]
    fn sincospi_half() {
        let (s, c) = sincospi_strict(0.5);
        assert_eq!(s.to_bits(), 1.0f64.to_bits(), "sincospi(0.5).sin must be EXACTLY 1");
        assert_eq!(c, 0.0, "sincospi(0.5).cos must be EXACTLY 0");
    }

    #[test]
    fn sincospi_matches_separate_calls() {
        for &x in &[0.0_f64, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.3, -0.5] {
            let (s, c) = sincospi_strict(x);
            let s2 = sinpi_strict(x);
            let c2 = cospi_strict(x);
            assert_eq!(s.to_bits(), s2.to_bits(), "sincospi({x}).sin mismatch");
            assert_eq!(c.to_bits(), c2.to_bits(), "sincospi({x}).cos mismatch");
        }
    }
}
