# Op Identity Method — Design Specification

## Origin

Aristotle (deconstruction): "the framework says associative but requires monoid"
Adversarial (waves 13-15): found 7 bugs from conflating identity and degenerate
Math-researcher (this doc): formalized as two required trait methods

## The Design Rule

Every scannable Op must provide two methods:

```rust
/// The neutral element of the monoid.
/// Used for padding when n is not a power of 2.
/// MUST satisfy: identity ⊕ x = x ⊕ identity = x for all x.
fn identity(&self) -> State;

/// What to return for invalid/empty/degenerate input.
/// NOT part of the monoid — it signals computation failure.
/// MUST NOT be used for padding.
fn degenerate(&self) -> State;
```

## Why Two Methods, Not One

| Concept | Mathematical role | Used for |
|---|---|---|
| identity | Element OF the monoid | Padding in Blelloch tree |
| degenerate | Signal OUTSIDE the monoid | Empty input, NaN data, singular matrix |

Conflating them produces bugs:
- Padding Max with NaN (degenerate) instead of -inf (identity) → NaN propagates through entire scan
- Padding Welford with NaN-mean (degenerate) instead of zero-count (identity) → corrupted moments
- Returning -inf (identity) for "no data" instead of NaN (degenerate) → false results look valid

## Identity Values for All Ops

| Op | State type | identity() | degenerate() |
|---|---|---|---|
| Add | f64 | 0.0 | NaN |
| Max | f64 | f64::NEG_INFINITY | NaN |
| Min | f64 | f64::INFINITY | NaN |
| ArgMin | (f64, usize) | (f64::INFINITY, usize::MAX) | (NaN, usize::MAX) |
| ArgMax | (f64, usize) | (f64::NEG_INFINITY, usize::MAX) | (NaN, usize::MAX) |
| DotProduct | f64 | 0.0 | NaN |
| Distance | f64 | 0.0 | NaN |
| WelfordMerge | (n,mean,m2) | (0, 0.0, 0.0) | (0, NaN, NaN) |
| AffineMap | (a,b) | (1.0, 0.0) | (NaN, NaN) |
| MatMulPrefix(d) | [f64; d*d] | I_d (identity matrix) | [NaN; d*d] |
| SarkkaMerge | (A,b,C,eta,J) | (I, 0, 0, 0, 0) | (NaN matrices) |
| LogSumExpMerge | (max,sum,val) | (-inf, 0.0, 0.0) | (NaN, NaN, NaN) |

### Verification Property

For every Op, the following must be tested:
```
assert!(combine(identity, x) == x);    // left identity
assert!(combine(x, identity) == x);    // right identity
assert!(combine(identity, identity) == identity);  // idempotent
```

## Why identity() Returns State, Not Option<State>

If an Op doesn't have an identity, it cannot be used in a prefix scan.
Making identity() return Option<State> would:
1. Allow constructing scans over non-monoidal Ops (type-unsafe)
2. Defer the check to runtime (exactly the bug class found in waves 13-15)
3. Require every scan engine callsite to unwrap (boilerplate)

The type system should enforce: **scannable ⟹ monoidal**. A required `identity()` 
method does this at compile time.

## Implementation Path

1. Add `identity()` and `degenerate()` to the Op enum (or trait)
2. Update all scan engines to use `op.identity()` for padding
3. Update all empty-input paths to use `op.degenerate()` for return values
4. Add invariant tests: `combine(identity, x) == x` for random x, for each Op
5. Add adversarial test: all-negative vector with Op::Max, length 2^k-1

## Priority

HIGH. This prevents an entire class of bugs (7 already found empirically).
One-time implementation cost, permanent correctness benefit.
