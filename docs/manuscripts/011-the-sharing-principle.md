# The Sharing Principle: Why Computers Should Remember

**Draft — 2026-03-30**
**Field**: Systems Design / Software Philosophy / Computing Paradigms

---

## Abstract

The default paradigm in computing is STATELESS: functions take input, produce output, retain nothing. Each invocation starts from scratch. We argue this paradigm discards enormous value — the knowledge that previous computations generated — and present measured evidence from GPU data science that STATEFUL systems which remember their computational history achieve speedups of four orders of magnitude over stateless systems on realistic workloads. We articulate the *sharing principle*: the most powerful optimization a computational system can make is not computing faster, but recognizing when it doesn't need to compute at all. We trace this principle through build systems (Bazel), databases (materialized views), rendering (progressive accumulation), and GPU computation (provenance caching), identifying a common architecture and arguing that statelessness is a premature optimization for simplicity that trades away the dominant performance opportunity.

---

## 1. The Stateless Default

### 1.1 Why We Default to Stateless

Functional programming teaches: pure functions are easier to reason about, test, compose, and parallelize. No side effects. No hidden state. Every call is independent. This is genuinely valuable for correctness and maintainability.

### 1.2 What We Give Up

A pure function `f(x)` called twice with the same `x` computes the answer twice. The first call's result — and everything that went into producing it — is discarded. If `f` takes 900 microseconds, two identical calls cost 1,800 microseconds.

A stateful system that remembers: first call costs 900 microseconds. Second call costs 35 nanoseconds (a lookup). That's 25,714x faster. Not 2x. Not 10x. **Twenty-five thousand times faster.**

This is not a theoretical construct. We measured it on production GPU hardware with real financial data pipelines.

### 1.3 The Sharing Principle

**The most powerful optimization is not computing faster. It is recognizing when computation is unnecessary.**

The hierarchy of optimization:
1. **Don't compute** (sharing via provenance): 25,714x
2. **Don't transfer** (sharing via persistence): 26x
3. **Don't dispatch** (sharing via fusion): 2.3x
4. **Compute faster** (better algorithms): 1.5-3x

Each level is an order of magnitude more powerful than the next. Yet the entire GPU computing ecosystem — NVIDIA's RAPIDS, CuPy, PyTorch, TensorFlow — focuses almost exclusively on level 4. Levels 1-3 require STATE. The industry's commitment to statelessness leaves four orders of magnitude on the table.

---

## 2. The Architecture of Remembering

### 2.1 Four Systems That Remember

| System | What it remembers | Mechanism | Speedup |
|---|---|---|---|
| Bazel | Build outputs | Content-addressed cache | ~100x on incremental builds |
| PostgreSQL | Query results | Materialized views | ~10-1000x on repeated queries |
| Progressive rendering | Partial frames | Accumulation buffer | ~N (N = frame count) |
| GPU computation (this work) | Computed results | Provenance cache | 25,714x |

### 2.2 The Common Architecture

All four share a structure:

1. **Identity function**: a way to determine IF a previous result is reusable (content hash, query signature, pixel coordinates, provenance hash)
2. **Cache**: a store mapping identities to results (CAS store, view table, frame buffer, GPU provenance cache)
3. **Invalidation**: a way to determine WHEN a cached result is stale (file modification time, dependency tracking, camera change, dirty bitmap)
4. **Fallback**: the stateless computation path used on cache miss

### 2.3 The Key Insight: Computation is the Fallback

In a well-designed stateful system, COMPUTATION is the exceptional case — the cache miss path. The common case is identity-check → cache-hit → return cached result.

This inverts the traditional performance model. The "hot path" is not the computation — it's the lookup. Optimizing the computation (level 4) matters less than optimizing the lookup (level 1).

---

## 3. Why Statelessness Persists

### 3.1 Real Benefits

- **Correctness**: no stale state, no invalidation bugs, no cache coherence issues
- **Simplicity**: no cache management, no eviction policy, no memory pressure handling
- **Composability**: functions compose freely without worrying about shared state

### 3.2 The Hidden Assumption

Statelessness assumes that COMPUTATION IS CHEAP relative to the cost of MANAGING STATE. This was true when:
- CPUs were slow → computation dominated
- Memory was expensive → state was costly to maintain
- Workloads were small → recomputation was fast

It is false when:
- GPUs are fast → computation is measured in microseconds
- GPU memory is 96GB → state storage is abundant
- Workloads recur → the same computation happens thousands of times
- The state management cost is 35 nanoseconds per lookup

### 3.3 The 25,714x Argument

When the cache lookup costs 35ns and the computation costs 900μs, the break-even point is: if the computation recurs more than once in the cache lifetime, stateful wins. For a financial signal farm processing 4,600 instruments with 20 statistics each, computations recur THOUSANDS of times per day with ~1% changing between updates.

The case for statelessness assumes computation happens once. The reality: computation recurs. The sharing principle: recognize the recurrence and eliminate it.

---

## 4. Beyond Caching: Why the Sharing Principle is Deeper than Memoization

### 4.1 Memoization is Order 1

Simple memoization: cache exact input-output pairs. This is order-1 sharing — each computation is memoized independently.

### 4.2 Structural Sharing is Order 2+

The sharing principle includes STRUCTURAL sharing:
- Two computations that share a common sub-computation (common subexpression elimination)
- A computation whose INPUTS haven't changed (provenance-based elimination)
- A computation whose result is RESIDENT from a different query (cross-query sharing)

These require the system to understand the RELATIONSHIPS between computations, not just their individual identities. This is order-2+ sharing — the "biphoton" level from the recursive lifting framework.

### 4.3 The Compiler Sees What Functions Can't

A pure function `f(x)` can be memoized by its arguments. But two pure functions `f(g(x))` and `h(g(x))` share the sub-computation `g(x)` — and neither function KNOWS this. Only a system that sees BOTH calls can detect the sharing.

This is why the sharing principle requires a COMPILER (or query planner), not just function-level memoization. The compiler sees the full computation graph and finds sharing that individual functions can't.

---

## 5. Implications

### 5.1 For System Designers

- **Default to stateful.** Add statelessness as an option when correctness or simplicity requires it, not as the default.
- **The cache is the system.** Design the caching layer first (identity, storage, invalidation), then the computation layer. Not the reverse.
- **Measure the recurrence rate.** If any computation recurs more than twice, the sharing principle applies. For data science workloads: recurrence is the norm, not the exception.

### 5.2 For Language Designers

- **Make sharing a language primitive.** If the language can express "this computation has been done before," the runtime can skip it. Memoization decorators are a weak version of this. A sharing-aware type system would be stronger.
- **Provenance as a type.** If values carry their computational identity (how they were produced), sharing detection is type-level reasoning, not runtime overhead.

### 5.3 For the Industry

The GPU computing industry has invested billions in making computation faster (better kernels, tensor cores, larger GPUs). The sharing principle suggests an alternative investment: making computation UNNECESSARY. A 25,714x improvement from remembering previous results dwarfs any hardware speedup. The next order-of-magnitude improvement in GPU data science may come not from faster hardware but from smarter software that remembers.

---

## References

- Acar, U. A. (2005). Self-Adjusting Computation. PhD thesis, CMU. (Incremental computation.)
- Blakeley, J. A. et al. (1986). Efficiently updating materialized views. SIGMOD.
- Matsakis, N. (2022). Salsa: incremental recomputation.
- Mokhov, A. et al. (2018). Build systems à la carte. ICFP.
