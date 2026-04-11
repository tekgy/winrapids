//! Generic prefix scan over any semiring.
//!
//! The fundamental Kingdom A primitive. Every parallel prefix computation
//! is this function with a different semiring:
//!
//! - `prefix_scan::<Additive>` = cumsum
//! - `prefix_scan::<TropicalMinPlus>` = running minimum cost
//! - `prefix_scan::<TropicalMaxPlus>` = running maximum
//! - `prefix_scan::<LogSumExp>` = HMM forward (scalar case)
//! - `prefix_scan::<Boolean>` = reachability prefix
//!
//! # Kingdom
//!
//! Kingdom A by definition — this IS the Kingdom A operation.
//! The scan is O(n) sequential, O(log n) parallel (Blelloch).

use crate::semiring::Semiring;

/// Inclusive prefix scan: output[i] = add(input[0], ..., input[i]).
///
/// Uses the semiring's `add` as the associative combine and `zero()`
/// as the identity for empty prefixes.
pub fn prefix_scan_inclusive<S: Semiring>(data: &[S::Elem]) -> Vec<S::Elem> {
    let mut result = Vec::with_capacity(data.len());
    let mut acc = S::zero();
    for &x in data {
        acc = S::add(acc, x);
        result.push(acc);
    }
    result
}

/// Exclusive prefix scan: output[i] = add(input[0], ..., input[i-1]).
///
/// output[0] = zero() (the identity). Equivalent to shifting the
/// inclusive scan right and prepending zero.
pub fn prefix_scan_exclusive<S: Semiring>(data: &[S::Elem]) -> Vec<S::Elem> {
    let mut result = Vec::with_capacity(data.len());
    let mut acc = S::zero();
    for &x in data {
        result.push(acc);
        acc = S::add(acc, x);
    }
    result
}

/// Total reduction: add all elements. Returns zero() for empty input.
pub fn reduce<S: Semiring>(data: &[S::Elem]) -> S::Elem {
    data.iter().fold(S::zero(), |acc, &x| S::add(acc, x))
}

/// Segmented prefix scan: resets accumulator at segment boundaries.
///
/// `starts[i]` is true if element i begins a new segment.
/// Each segment gets its own independent prefix scan.
pub fn prefix_scan_segmented<S: Semiring>(data: &[S::Elem], starts: &[bool]) -> Vec<S::Elem> {
    let mut result = Vec::with_capacity(data.len());
    let mut acc = S::zero();
    for (i, &x) in data.iter().enumerate() {
        if i < starts.len() && starts[i] {
            acc = S::zero();
        }
        acc = S::add(acc, x);
        result.push(acc);
    }
    result
}

#[cfg(test)]
mod tests;
