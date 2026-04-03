//! # Beal Conjecture Search — $1,000,000 Prize
//!
//! If A^x + B^y = C^z where A,B,C,x,y,z are positive integers with x,y,z >= 3,
//! then A,B,C must share a common prime factor.
//!
//! Finding a counterexample (coprime A,B,C) = $1,000,000 from the AMS.
//!
//! Current record: A,B,C < 250,000 for low exponents.
//!
//! Strategy: for each exponent triple (x,y,z), sweep A and B,
//! check if A^x + B^y is a perfect z-th power, if so check gcd.

use std::time::Instant;

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn gcd3(a: u128, b: u128, c: u128) -> u128 {
    gcd(gcd(a, b), c)
}

/// Integer z-th root via Newton's method. Returns Some(c) if c^z == n exactly.
fn exact_integer_root(n: u128, z: u32) -> Option<u128> {
    if n == 0 { return Some(0); }
    if n == 1 { return Some(1); }
    if z == 1 { return Some(n); }

    // Initial guess via floating point
    let approx = (n as f64).powf(1.0 / z as f64) as u128;

    // Check approx-1, approx, approx+1 (floating point can be off by 1)
    for c in approx.saturating_sub(2)..=approx + 2 {
        if c == 0 { continue; }
        // Compute c^z carefully to avoid overflow
        if let Some(power) = checked_pow(c, z) {
            if power == n {
                return Some(c);
            }
        }
    }
    None
}

/// Checked integer power that returns None on overflow
fn checked_pow(base: u128, exp: u32) -> Option<u128> {
    let mut result: u128 = 1;
    for _ in 0..exp {
        result = result.checked_mul(base)?;
    }
    Some(result)
}

/// Search for Beal counterexamples with given exponent triple
fn search_exponents(x: u32, y: u32, z: u32, max_base: u64) -> Vec<(u64, u64, u64)> {
    let mut counterexamples = Vec::new();
    let mut checked = 0u64;
    let mut perfect_powers_found = 0u64;

    for a in 2..=max_base {
        let a_pow = match checked_pow(a as u128, x) {
            Some(v) => v,
            None => break, // overflow — a is too large for this exponent
        };

        for b in a..=max_base { // b >= a to avoid duplicates
            let b_pow = match checked_pow(b as u128, y) {
                Some(v) => v,
                None => break, // overflow
            };

            let s = match a_pow.checked_add(b_pow) {
                Some(v) => v,
                None => break, // overflow
            };

            checked += 1;

            // Is s a perfect z-th power?
            if let Some(c) = exact_integer_root(s, z) {
                perfect_powers_found += 1;

                // Check if A, B, C are coprime
                if gcd3(a as u128, b as u128, c) == 1 {
                    eprintln!("  !!! COUNTEREXAMPLE: {}^{} + {}^{} = {}^{}, gcd=1 !!!",
                        a, x, b, y, c, z);
                    counterexamples.push((a, b, c as u64));
                } else {
                    let g = gcd3(a as u128, b as u128, c);
                    // This is expected — most solutions share a common factor
                    if checked < 100 || perfect_powers_found <= 10 {
                        eprintln!("  Found: {}^{} + {}^{} = {}^{} (gcd={}, NOT coprime)",
                            a, x, b, y, c, z, g);
                    }
                }
            }
        }
    }

    eprintln!("  ({},{},{}) checked={}, perfect_powers={}, counterexamples={}",
        x, y, z, checked, perfect_powers_found, counterexamples.len());

    counterexamples
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Beal Conjecture Search — tambear");
    eprintln!("  Prize: $1,000,000 (AMS)");
    eprintln!("  Looking for: A^x + B^y = C^z with gcd(A,B,C) = 1");
    eprintln!("==========================================================\n");

    let t0 = Instant::now();

    // Phase 1: exhaustive search for small exponents and bases
    let max_base = 1000u64;
    let mut total_counterexamples = 0;

    // The exponent triples to check (all with min exponent >= 3)
    let exponent_triples: Vec<(u32, u32, u32)> = vec![
        (3, 3, 3), // Fermat's Last Theorem says no solutions at all for equal exponents
        (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7),
        (3, 4, 4), (3, 4, 5), (3, 4, 6), (3, 4, 7),
        (3, 5, 5), (3, 5, 6), (3, 5, 7),
        (3, 6, 6), (3, 6, 7),
        (3, 7, 7),
        (4, 4, 4), (4, 4, 5), (4, 4, 6), (4, 4, 7),
        (4, 5, 5), (4, 5, 6), (4, 5, 7),
        (4, 6, 6),
        (5, 5, 5), (5, 5, 6), (5, 5, 7),
        (5, 6, 6),
        (6, 6, 6), (6, 6, 7),
        (7, 7, 7),
    ];

    eprintln!("Phase 1: A,B in [2, {}], {} exponent triples\n", max_base, exponent_triples.len());

    for (x, y, z) in &exponent_triples {
        let results = search_exponents(*x, *y, *z, max_base);
        total_counterexamples += results.len();
    }

    let elapsed = t0.elapsed();

    eprintln!("\n==========================================================");
    eprintln!("  RESULTS");
    eprintln!("==========================================================");
    eprintln!("  Exponent triples checked: {}", exponent_triples.len());
    eprintln!("  Max base: {}", max_base);
    eprintln!("  Total time: {:.2}s", elapsed.as_secs_f64());

    if total_counterexamples == 0 {
        eprintln!("\n  NO COUNTEREXAMPLES FOUND.");
        eprintln!("  The Beal conjecture holds for A,B <= {} with tested exponents.", max_base);
        eprintln!("  (This does NOT prove the conjecture — only verifies this range.)");
    } else {
        eprintln!("\n  !!! {} COUNTEREXAMPLES FOUND !!!", total_counterexamples);
        eprintln!("  Contact the AMS to claim the $1,000,000 prize.");
    }

    eprintln!("\n  Note: Solutions with gcd > 1 (like 3^3 + 6^3 = 3^5)");
    eprintln!("  are expected and do NOT disprove the conjecture.");
    eprintln!("  Only COPRIME solutions (gcd=1) would be counterexamples.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_basic() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1); // coprime
        assert_eq!(gcd3(6, 10, 15), 1); // pairwise not coprime but gcd3=1
    }

    #[test]
    fn exact_root_works() {
        assert_eq!(exact_integer_root(27, 3), Some(3));    // 3^3 = 27
        assert_eq!(exact_integer_root(256, 4), Some(4));   // 4^4 = 256
        assert_eq!(exact_integer_root(28, 3), None);       // not a perfect cube
        assert_eq!(exact_integer_root(1, 100), Some(1));   // 1^anything = 1
    }

    #[test]
    fn known_non_coprime_solution() {
        // 3^3 + 6^3 = 3^5 → 27 + 216 = 243 ✓, but gcd(3,6,3) = 3 ≠ 1
        let a_pow = checked_pow(3, 3).unwrap();  // 27
        let b_pow = checked_pow(6, 3).unwrap();  // 216
        assert_eq!(a_pow + b_pow, 243);
        assert_eq!(exact_integer_root(243, 5), Some(3)); // 3^5 = 243
        assert_eq!(gcd3(3, 6, 3), 3); // NOT coprime — not a counterexample
    }

    #[test]
    fn no_coprime_solutions_small() {
        // Verify no counterexamples for (3,3,3) up to base 100
        let results = search_exponents(3, 3, 3, 100);
        assert!(results.is_empty(), "Fermat's Last Theorem: no solutions for x=y=z=3");
    }
}
