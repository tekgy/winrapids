"""
E07 -- Provenance-Based Reuse: Skip Already-Computed Results

The persistent store concept relies on knowing when a GPU buffer is
still valid (same input data, same computation). If we can skip
recomputation, the signal farm only processes what's changed.

This experiment measures:
  1. What does provenance tracking cost? (hash-based identity)
  2. How much time does skipping save for various operations?
  3. What's the optimal granularity for provenance? (per-column, per-leaf, per-section)
  4. How does the dirty/clean ratio affect farm throughput?

Provenance model:
  Each GPU buffer has a provenance tag: hash(input_data_id + computation_id).
  Before computing, check if the output buffer's tag matches the expected tag.
  If match: skip computation (reuse existing result).
  If mismatch: compute, update tag.
"""

import time
import hashlib
import numpy as np
import cupy as cp


class ProvenanceBuffer:
    """GPU buffer with provenance tracking."""

    def __init__(self, data: cp.ndarray, tag: str):
        self.data = data
        self.tag = tag  # Hash identifying input + computation

    @staticmethod
    def compute_tag(input_id: str, computation_id: str) -> str:
        """Compute a provenance tag from input identity + computation identity."""
        return hashlib.md5(f"{input_id}:{computation_id}".encode()).hexdigest()[:16]


class ProvenanceStore:
    """Simulated persistent GPU store with provenance-based reuse."""

    def __init__(self):
        self.buffers: dict[str, ProvenanceBuffer] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: str, input_id: str, computation_id: str,
                       compute_fn) -> cp.ndarray:
        """Return cached result if provenance matches, otherwise compute."""
        expected_tag = ProvenanceBuffer.compute_tag(input_id, computation_id)

        if key in self.buffers and self.buffers[key].tag == expected_tag:
            self.hits += 1
            return self.buffers[key].data

        self.misses += 1
        result = compute_fn()
        self.buffers[key] = ProvenanceBuffer(result, expected_tag)
        return result

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses,
                "total": total, "hit_rate": hit_rate}


def bench(fn, warmup=3, runs=20, label=""):
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    print("=" * 70)
    print("E07 -- Provenance-Based Reuse")
    print("=" * 70)

    # ── Test 1: Provenance tracking overhead ────────────────────
    print("\n--- Test 1: Provenance Tracking Overhead ---\n")

    n = 10_000_000
    data = cp.random.uniform(0, 100, n).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()

    # Measure: computing without provenance
    def raw_compute():
        cs = cp.cumsum(data)
        cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
        return (cs[60:] - cs[:-60]) / 60

    # Measure: computing with provenance check (miss)
    store = ProvenanceStore()

    def provenance_miss():
        # Each call has a unique input_id to force a miss
        tag = str(time.perf_counter())
        return store.get_or_compute(
            "rolling_mean", tag, "rolling_mean_60", raw_compute)

    # Measure: provenance check (hit)
    store2 = ProvenanceStore()
    store2.get_or_compute("rolling_mean", "fixed_input", "rolling_mean_60", raw_compute)
    cp.cuda.Stream.null.synchronize()

    def provenance_hit():
        return store2.get_or_compute(
            "rolling_mean", "fixed_input", "rolling_mean_60", raw_compute)

    raw_p50 = bench(raw_compute)
    miss_p50 = bench(provenance_miss)
    hit_p50 = bench(provenance_hit)

    overhead_miss = miss_p50 - raw_p50
    overhead_hit = hit_p50  # hit path doesn't compute at all

    print(f"  Raw compute (no provenance):     {raw_p50:.3f} ms")
    print(f"  With provenance (miss):          {miss_p50:.3f} ms  "
          f"(overhead: {overhead_miss:.3f} ms)")
    print(f"  With provenance (hit):           {hit_p50:.3f} ms  "
          f"(savings: {raw_p50:.3f} ms = {raw_p50/hit_p50:.0f}x)")

    # ── Test 2: Savings by operation complexity ──────────────────
    print("\n--- Test 2: Reuse Savings by Operation Complexity ---\n")

    keys = cp.random.randint(0, 1000, n).astype(cp.int32)
    cp.cuda.Stream.null.synchronize()

    operations = {
        "sum (simple reduce)": lambda: float(cp.sum(data)),
        "rolling mean (w=60)": raw_compute,
        "sort (argsort)": lambda: cp.argsort(data),
        "groupby sum": lambda: (
            lambda idx: (
                lambda sk, sv: (
                    lambda cs: cs  # just compute cumsum on sorted
                )(cp.cumsum(sv))
            )(keys[idx], data[idx])
        )(cp.argsort(keys)),
        "expression (fused 5-op)": lambda: (data * data + data * 2 - 1) / (data + 1.0001) + cp.sqrt(cp.abs(data)),
        "rolling std (w=60)": lambda: (
            lambda cs, cs2: cp.sqrt(cp.maximum(
                (cs2[60:] - cs2[:-60]) / 60 - ((cs[60:] - cs[:-60]) / 60) ** 2, 0))
        )(cp.concatenate([cp.zeros(1, dtype=data.dtype), cp.cumsum(data)]),
          cp.concatenate([cp.zeros(1, dtype=data.dtype), cp.cumsum(data * data)])),
    }

    print(f"  {'Operation':>30}  {'Compute (ms)':>12}  {'Hit (ms)':>10}  {'Savings':>8}")

    for op_name, op_fn in operations.items():
        store = ProvenanceStore()
        # Prime the cache
        store.get_or_compute(op_name, "input_v1", op_name, op_fn)
        cp.cuda.Stream.null.synchronize()

        compute_time = bench(op_fn)
        hit_time = bench(lambda: store.get_or_compute(op_name, "input_v1", op_name, op_fn))

        print(f"  {op_name:>30}  {compute_time:12.3f}  {hit_time:10.4f}  "
              f"{compute_time/hit_time:7.0f}x")

    # ── Test 3: Farm simulation with dirty ratio ─────────────────
    print("\n--- Test 3: Farm Simulation — Dirty Ratio Impact ---\n")

    n_tickers = 100
    n_cadences = 5
    n_leaves = 10
    total_computations = n_tickers * n_cadences * n_leaves

    # Create resident data for all tickers (small: 100K each to fit memory)
    ticker_data = {}
    for t in range(n_tickers):
        ticker_data[t] = cp.random.uniform(0, 100, 100_000).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()

    def leaf_compute(data):
        """Simulate a leaf computation (rolling mean)."""
        cs = cp.cumsum(data)
        cs = cp.concatenate([cp.zeros(1, dtype=data.dtype), cs])
        return (cs[60:] - cs[:-60]) / 60

    # Simulate different dirty ratios
    for dirty_pct in [100, 50, 20, 10, 5, 1]:
        store = ProvenanceStore()
        n_dirty = max(1, int(n_tickers * dirty_pct / 100))

        # Populate cache (all tickers, all cadences, all leaves)
        for t in range(n_tickers):
            for c in range(n_cadences):
                for l in range(n_leaves):
                    key = f"t{t}_c{c}_l{l}"
                    store.get_or_compute(key, f"v1_{t}", f"leaf_{l}_c{c}",
                                         lambda t=t: leaf_compute(ticker_data[t]))
        cp.cuda.Stream.null.synchronize()

        # Now simulate an update where dirty_pct% of tickers got new data
        dirty_tickers = set(range(n_dirty))

        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        for t in range(n_tickers):
            input_version = f"v2_{t}" if t in dirty_tickers else f"v1_{t}"
            for c in range(n_cadences):
                for l in range(n_leaves):
                    key = f"t{t}_c{c}_l{l}"
                    store.get_or_compute(key, input_version, f"leaf_{l}_c{c}",
                                         lambda t=t: leaf_compute(ticker_data[t]))

        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        farm_ms = (t1 - t0) * 1000
        stats = store.stats()

        print(f"  {dirty_pct:>3}% dirty ({n_dirty:>3} tickers): "
              f"farm={farm_ms:7.1f} ms, "
              f"hits={stats['hits']:>5}, misses={stats['misses']:>5}, "
              f"hit_rate={stats['hit_rate']:.1%}")

    # ── Test 4: Hash computation cost for provenance ─────────────
    print("\n--- Test 4: Hash Computation Cost ---\n")

    # What does MD5 hashing cost for provenance tags?
    short_str = "ticker_AAPL_cadence_60s_leaf_rolling_mean"
    long_str = short_str * 10

    n_hashes = 100_000
    t0 = time.perf_counter()
    for _ in range(n_hashes):
        hashlib.md5(short_str.encode()).hexdigest()[:16]
    t1 = time.perf_counter()
    short_us = (t1 - t0) / n_hashes * 1e6

    t0 = time.perf_counter()
    for _ in range(n_hashes):
        hashlib.md5(long_str.encode()).hexdigest()[:16]
    t1 = time.perf_counter()
    long_us = (t1 - t0) / n_hashes * 1e6

    print(f"  MD5 hash (short tag):  {short_us:.2f} us/hash")
    print(f"  MD5 hash (long tag):   {long_us:.2f} us/hash")
    print(f"  100K hashes:           {n_hashes * short_us / 1e6:.1f} ms")
    print(f"  Cost per farm cycle:   {total_computations * short_us / 1e3:.2f} ms "
          f"({total_computations} computations)")

    # Cleanup
    del ticker_data
    cp.get_default_memory_pool().free_all_blocks()

    print(f"\n{'=' * 70}")
    print("E07 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
