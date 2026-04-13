//! # Parallel range reduction
//!
//! Embarrassingly-parallel fold pattern: given a range `[start, end)` (or a
//! slice), split it into `n_threads` chunks, run a per-chunk fold in parallel,
//! and merge the partial results via a user-supplied binary merge.
//!
//! The merge must form a **commutative monoid** (or at least an associative
//! operation with an identity) for the parallel result to match a sequential
//! fold. Non-commutative monoids (e.g. matrix multiplication) are fine as long
//! as chunk order is preserved — which this framework guarantees via sorted
//! `JoinHandle` collection.
//!
//! ## When to use
//!
//! Use this when:
//! - The work per element is significant (`> 1µs`), otherwise the thread spawn
//!   dominates.
//! - The per-element computation is CPU-bound (no locks, no I/O).
//! - The merge operation is cheap.
//!
//! Don't use this for:
//! - Tight numerical inner loops (O(1) work per element) — spawn overhead
//!   wipes out any gain.
//! - Sub-millisecond workloads — sequential is faster.
//! - Algorithms where parallel determinism matters and the merge is not
//!   associative (floating-point sum).
//!
//! ## Example
//!
//! ```ignore
//! use tambear::parallel::parallel_range_reduce;
//!
//! // Count primes in [1, N) using parallel fold
//! let n_primes = parallel_range_reduce(
//!     1u64, 1_000_000,
//!     4,                         // n_threads
//!     0u64,                      // identity
//!     |x| if is_prime(x) { 1 } else { 0 },  // map
//!     |a, b| a + b,              // merge
//! );
//! ```

use std::sync::Arc;

/// Parallel range reduction: splits `[start, end)` into `n_threads` chunks,
/// runs `map` over each element, reduces within each chunk using `merge`
/// starting from `identity`, then merges the per-chunk results using the same
/// `merge`.
///
/// # Parameters
/// - `start` / `end`: inclusive / exclusive range bounds (integer).
/// - `n_threads`: number of worker threads. Values < 1 are treated as 1.
/// - `identity`: the monoid identity — `merge(identity, x)` must equal `x`.
/// - `map`: element-to-state function applied to each integer in the range.
/// - `merge`: associative binary operation combining two states.
///
/// # Guarantees
/// - For a commutative monoid, the result is deterministic regardless of
///   thread count.
/// - For a non-commutative monoid, chunks are merged in ascending range order,
///   so the result matches a sequential fold of the range.
///
/// # Thread safety
/// `map` and `merge` must be `Send + Sync + 'static`, and `State` must be
/// `Send + 'static`. The closures are cloned into each worker via `Arc`.
pub fn parallel_range_reduce<State, MapFn, MergeFn>(
    start: u64,
    end: u64,
    n_threads: usize,
    identity: State,
    map: MapFn,
    merge: MergeFn,
) -> State
where
    State: Clone + Send + 'static,
    MapFn: Fn(u64) -> State + Send + Sync + 'static,
    MergeFn: Fn(State, State) -> State + Send + Sync + 'static,
{
    let n_threads = n_threads.max(1);
    if end <= start {
        return identity;
    }

    let map = Arc::new(map);
    let merge = Arc::new(merge);

    let range_size = end - start;
    let chunk_size = range_size / n_threads as u64;
    let remainder = range_size % n_threads as u64;

    let mut handles = Vec::with_capacity(n_threads);

    for t in 0..n_threads {
        // Distribute the remainder: first `remainder` threads get one extra element.
        let chunk_start = start
            + (t as u64) * chunk_size
            + (t as u64).min(remainder);
        let chunk_end = chunk_start + chunk_size + if (t as u64) < remainder { 1 } else { 0 };

        if chunk_start >= chunk_end {
            continue;
        }

        let map = Arc::clone(&map);
        let merge = Arc::clone(&merge);
        let id = identity.clone();

        handles.push(std::thread::spawn(move || {
            let mut acc = id;
            for i in chunk_start..chunk_end {
                acc = merge(acc, map(i));
            }
            acc
        }));
    }

    // Merge in deterministic order (thread spawn order == range order).
    let mut result = identity;
    for h in handles {
        let partial = h.join().expect("parallel_range_reduce: worker panicked");
        result = merge(result, partial);
    }
    result
}

/// Parallel slice reduction: same pattern as `parallel_range_reduce` but over
/// a `&[T]` instead of an integer range.
///
/// The slice is split into `n_threads` contiguous sub-slices. Each worker
/// folds `map` over its sub-slice starting from `identity`. Partial results
/// are then folded together.
pub fn parallel_slice_reduce<T, State, MapFn, MergeFn>(
    data: &[T],
    n_threads: usize,
    identity: State,
    map: MapFn,
    merge: MergeFn,
) -> State
where
    T: Sync,
    State: Clone + Send + 'static,
    MapFn: Fn(&T) -> State + Send + Sync,
    MergeFn: Fn(State, State) -> State + Send + Sync,
{
    let n = data.len();
    let n_threads = n_threads.max(1);
    if n == 0 {
        return identity;
    }

    // If the slice is smaller than n_threads, degenerate to sequential.
    if n <= n_threads || n_threads == 1 {
        let mut acc = identity;
        for x in data {
            acc = merge(acc, map(x));
        }
        return acc;
    }

    // Use std::thread::scope so we can borrow `data` immutably across threads.
    std::thread::scope(|scope| {
        let chunk_size = n / n_threads;
        let remainder = n % n_threads;

        let map_ref = &map;
        let merge_ref = &merge;

        let mut handles = Vec::with_capacity(n_threads);

        let mut cursor = 0usize;
        for t in 0..n_threads {
            let extra = if t < remainder { 1 } else { 0 };
            let chunk_start = cursor;
            let chunk_end = chunk_start + chunk_size + extra;
            cursor = chunk_end;

            if chunk_start >= chunk_end { continue; }

            let chunk: &[T] = &data[chunk_start..chunk_end];
            let id = identity.clone();

            handles.push(scope.spawn(move || {
                let mut acc = id;
                for x in chunk {
                    acc = merge_ref(acc, map_ref(x));
                }
                acc
            }));
        }

        // Merge in deterministic order.
        let mut result = identity;
        for h in handles {
            let partial = h.join().expect("parallel_slice_reduce: worker panicked");
            result = merge_ref(result, partial);
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_reduce_sum() {
        // Σ i for i in [1, 101) = 5050
        let total = parallel_range_reduce(
            1u64, 101,
            4,
            0u64,
            |x| x,
            |a, b| a + b,
        );
        assert_eq!(total, 5050);
    }

    #[test]
    fn range_reduce_empty() {
        let r = parallel_range_reduce(10u64, 10, 4, 0i64, |_| 1, |a, b| a + b);
        assert_eq!(r, 0);
    }

    #[test]
    fn range_reduce_single_thread_matches() {
        let par = parallel_range_reduce(0u64, 1000, 1, 0u64, |x| x, |a, b| a + b);
        let seq: u64 = (0..1000u64).sum();
        assert_eq!(par, seq);
    }

    #[test]
    fn range_reduce_many_threads_matches_sequential() {
        // Use 7 threads on range of 100 to exercise uneven chunk distribution.
        let par = parallel_range_reduce(0u64, 100, 7, 0u64, |x| x * x, |a, b| a + b);
        let seq: u64 = (0..100u64).map(|x| x * x).sum();
        assert_eq!(par, seq);
    }

    #[test]
    fn range_reduce_max() {
        let r = parallel_range_reduce(
            1u64, 1000,
            4,
            0u64,
            |x| x,
            |a, b| a.max(b),
        );
        assert_eq!(r, 999);
    }

    #[test]
    fn range_reduce_tuple_merge() {
        // Compute (count, sum) in parallel — classic monoid fold.
        let (count, sum) = parallel_range_reduce(
            0u64, 100,
            4,
            (0u64, 0u64),
            |x| (1, x),
            |(c1, s1), (c2, s2)| (c1 + c2, s1 + s2),
        );
        assert_eq!(count, 100);
        assert_eq!(sum, (0..100u64).sum::<u64>());
    }

    #[test]
    fn slice_reduce_sum() {
        let data: Vec<u64> = (1..=1000).collect();
        let total = parallel_slice_reduce(
            &data,
            4,
            0u64,
            |&x| x,
            |a, b| a + b,
        );
        let expected: u64 = (1..=1000u64).sum();
        assert_eq!(total, expected);
    }

    #[test]
    fn slice_reduce_matches_iterator() {
        let data: Vec<f64> = (0..10_000).map(|i| (i as f64).sin()).collect();
        let par = parallel_slice_reduce(
            &data,
            4,
            0.0f64,
            |&x| x * x,
            |a, b| a + b,
        );
        let seq: f64 = data.iter().map(|&x| x * x).sum();
        // Parallel float sum may reorder; expect close but not identical.
        assert!((par - seq).abs() < 1e-6, "par={par} seq={seq}");
    }

    #[test]
    fn slice_reduce_small_slice_degenerates() {
        // Slice with fewer elements than threads: should still compute correctly.
        let data = [1u64, 2, 3];
        let r = parallel_slice_reduce(&data, 16, 0u64, |&x| x, |a, b| a + b);
        assert_eq!(r, 6);
    }

    #[test]
    fn slice_reduce_empty() {
        let empty: Vec<u64> = vec![];
        let r = parallel_slice_reduce(&empty, 4, 42u64, |&x| x, |a, b| a + b);
        assert_eq!(r, 42); // identity returned on empty input
    }
}
