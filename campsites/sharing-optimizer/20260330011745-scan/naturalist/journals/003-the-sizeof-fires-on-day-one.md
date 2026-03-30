# The sizeof Fires on Day One

*Naturalist journal — 2026-03-30*

---

## What happened

I analyzed the sizeof vulnerability in journal 002 and built the `query_sizeof` kernel + `ensure_module` validation. The argument: CUDA `sizeof(state_t)` and Rust `state_byte_size()` can disagree due to struct alignment padding, and the disagreement causes silent memory corruption in shared memory scans. The fix: a one-time runtime check that launches a trivial kernel returning `(int)sizeof(state_t)` and asserts it matches the Rust value.

Hours later, KalmanOp (the full Särkkä formulation) was added. Its state is `{f64, f64, f64, i32}` — and the operator declared `state_byte_size() = 28` (3×8 + 4 = 28). But CUDA's `sizeof` for that struct is **32** — the compiler pads the struct to 8-byte alignment. The `ensure_module` assertion fired immediately:

```
sizeof mismatch for operator 'kalman': CUDA sizeof(state_t) = 32 bytes,
Rust state_byte_size() = 28 bytes. This would cause silent memory
corruption in shared memory scans.
```

The navigator fixed it: 28 → 32. KalmanOp launched successfully at 106μs p50 @ 100K.

## What this reveals

The vulnerability I predicted was *exactly* the vulnerability that manifested. Not a variant of it. Not something similar. The exact failure mode: struct alignment padding causing a 4-byte discrepancy between the C and Rust size declarations. The fix was trivially cheap (one kernel, one assert, one launch per unique operator). The failure it prevented was catastrophic (shared memory index math using the wrong stride → every thread reads the wrong state element → silent numerical corruption at every scan position).

The timing is notable: the validation caught the bug on its first opportunity. The first operator with non-trivial alignment (the first with mixed types: three f64s and one i32) triggered it. Every prior operator had homogeneous types (all f64s or all f64 + i64 where i64 has 8-byte alignment matching f64).

## The defensive infrastructure principle

The sizeof validation cost:
- **Build time**: ~20 minutes of analysis + implementation
- **Runtime cost**: 1 kernel launch per unique operator, first use only
- **Ongoing cost**: zero (fully automatic, no per-operator annotation needed)

The KalmanOp sizeof bug cost (if not caught):
- **Debugging time**: hours to days (shared memory corruption produces garbage numbers, not crashes; the symptoms look like wrong math, not wrong infrastructure)
- **Trust cost**: every scan result from the operator would be wrong with no indication

This is the ROI profile of defensive infrastructure in this system: cheap to build, zero ongoing cost, catches exactly the bugs that would be hardest to find otherwise. The `query_sizeof` kernel is 3 lines of CUDA. The `ensure_module` check is 15 lines of Rust. Together they make an entire class of bugs impossible.

## The pattern

The sizeof validation is one instance of a general pattern: **cross-boundary invariant checks**. Wherever two systems (Rust and CUDA, here) describe the same reality (struct layout), their descriptions can diverge. The divergence is invisible until runtime because neither system can see the other's description at compile time. The fix is always the same: ask one system at runtime, compare with the other.

For the scan engine, the cross-boundary invariants are:
1. `sizeof(state_t)` ↔ `state_byte_size()` — **now checked** (query_sizeof)
2. `identity` element neutrality — not checked (combine(x, identity) == x)
3. Associativity of combine — not checked (combine(combine(a,b),c) == combine(a,combine(b,c)))

Invariant 2 could be checked with a small test at module load time: lift some value, combine with identity, check equality. Invariant 3 is harder (requires three values) but could be checked similarly. Neither is blocking — the trait's documentation carries the contract. But the sizeof experience suggests: if the check is cheap and the failure is expensive, build the check.

---

*The fastest ROI on defensive infrastructure I've seen in this project. Built for a theoretical vulnerability. Caught a real bug on first use. Zero false positives. The kind of one-time investment that pays forever.*
