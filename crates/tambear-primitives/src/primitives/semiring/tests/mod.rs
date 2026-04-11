use super::*;

/// Test that a semiring satisfies the algebraic laws.
fn verify_semiring_laws<S: Semiring<Elem = f64>>(a: f64, b: f64, c: f64) {
    // add is associative
    let lhs = S::add(S::add(a, b), c);
    let rhs = S::add(a, S::add(b, c));
    assert!((lhs - rhs).abs() < 1e-12, "add assoc: {lhs} != {rhs}");

    // add is commutative
    let lhs = S::add(a, b);
    let rhs = S::add(b, a);
    assert!((lhs - rhs).abs() < 1e-12, "add comm: {lhs} != {rhs}");

    // zero is additive identity
    assert!((S::add(S::zero(), a) - a).abs() < 1e-12, "zero + a != a");
    assert!((S::add(a, S::zero()) - a).abs() < 1e-12, "a + zero != a");

    // one is multiplicative identity
    assert!((S::mul(S::one(), a) - a).abs() < 1e-12, "one * a != a");
    assert!((S::mul(a, S::one()) - a).abs() < 1e-12, "a * one != a");

    // mul distributes over add (left)
    let lhs = S::mul(a, S::add(b, c));
    let rhs = S::add(S::mul(a, b), S::mul(a, c));
    assert!((lhs - rhs).abs() < 1e-10, "left distrib: {lhs} != {rhs}");
}

#[test]
fn additive_laws() {
    verify_semiring_laws::<Additive>(2.0, 3.0, 5.0);
    verify_semiring_laws::<Additive>(-1.0, 0.5, 100.0);
}

#[test]
fn tropical_min_plus_laws() {
    verify_semiring_laws::<TropicalMinPlus>(2.0, 3.0, 5.0);
    verify_semiring_laws::<TropicalMinPlus>(-1.0, 0.5, 100.0);
}

#[test]
fn tropical_max_plus_laws() {
    verify_semiring_laws::<TropicalMaxPlus>(2.0, 3.0, 5.0);
    verify_semiring_laws::<TropicalMaxPlus>(-1.0, 0.5, 100.0);
}

#[test]
fn log_sum_exp_laws() {
    verify_semiring_laws::<LogSumExp>(1.0, 2.0, 3.0);
    verify_semiring_laws::<LogSumExp>(-5.0, 0.0, 5.0);
}

#[test]
fn max_times_laws() {
    // MaxTimes only satisfies distributivity for non-negative values
    verify_semiring_laws::<MaxTimes>(0.5, 0.3, 0.8);
    verify_semiring_laws::<MaxTimes>(1.0, 2.0, 3.0);
}

#[test]
fn boolean_laws() {
    // Exhaustive for 2-element semiring
    for &a in &[false, true] {
        for &b in &[false, true] {
            for &c in &[false, true] {
                // Associativity
                assert_eq!(Boolean::add(Boolean::add(a, b), c),
                           Boolean::add(a, Boolean::add(b, c)));
                // Commutativity
                assert_eq!(Boolean::add(a, b), Boolean::add(b, a));
                // Identity
                assert_eq!(Boolean::add(Boolean::zero(), a), a);
                assert_eq!(Boolean::mul(Boolean::one(), a), a);
                // Distributivity
                assert_eq!(Boolean::mul(a, Boolean::add(b, c)),
                           Boolean::add(Boolean::mul(a, b), Boolean::mul(a, c)));
            }
        }
    }
}
