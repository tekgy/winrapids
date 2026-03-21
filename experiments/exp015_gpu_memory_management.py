"""Experiment 015: GPU Memory Management on Windows WDDM

Research questions:
1. What is the overhead of raw cudaMalloc vs CuPy's MemoryPool vs cudaMallocAsync?
2. How does WDDM's memory model affect allocation latency?
3. Can we build a WDDM-aware pool allocator that beats CuPy's default?
4. What is the practical usable VRAM limit under WDDM?
5. How do pinned memory transfers compare with/without pooling?

Device: NVIDIA RTX PRO 6000 Blackwell Max-Q (95.6 GB VRAM)
Platform: Windows 11 WDDM, CUDA 13.0, CuPy 14.0.1
"""

from __future__ import annotations

import gc
import statistics
import time

import cupy as cp
import numpy as np


def _sync():
    """Full GPU synchronize."""
    cp.cuda.Device(0).synchronize()


# ═══════════════════════════════════════════════════════════════════
# Experiment A: Allocation latency — raw cudaMalloc vs MemoryPool
#   vs MemoryAsyncPool across allocation sizes
# ═══════════════════════════════════════════════════════════════════

def exp_a_allocation_latency():
    """Compare allocation strategies across sizes."""
    print("=" * 72)
    print("EXP A: Allocation Latency — cudaMalloc vs MemoryPool vs AsyncPool")
    print("=" * 72)

    sizes = [
        ("1 KB", 1024),
        ("64 KB", 64 * 1024),
        ("1 MB", 1024 * 1024),
        ("16 MB", 16 * 1024 * 1024),
        ("256 MB", 256 * 1024 * 1024),
        ("1 GB", 1024 * 1024 * 1024),
        ("4 GB", 4 * 1024 * 1024 * 1024),
    ]

    n_warmup = 5
    n_trials = 50

    # Strategy 1: Raw cudaMalloc (bypass CuPy pool)
    print("\n--- Strategy 1: Raw cudaMalloc (no pool) ---")
    cp.cuda.set_allocator(None)  # Disable pool, use raw cudaMalloc
    _sync()

    raw_results = {}
    for label, size in sizes:
        # Warmup
        for _ in range(n_warmup):
            ptr = cp.cuda.alloc(size)
            del ptr
            _sync()

        latencies = []
        for _ in range(n_trials):
            _sync()
            t0 = time.perf_counter_ns()
            ptr = cp.cuda.alloc(size)
            _sync()
            t1 = time.perf_counter_ns()
            del ptr
            _sync()
            latencies.append((t1 - t0) / 1000)  # microseconds

        med = statistics.median(latencies)
        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        raw_results[label] = (med, p99)
        print(f"  {label:>8s}: median={med:10.1f} us  p99={p99:10.1f} us")

    # Strategy 2: CuPy MemoryPool (default)
    print("\n--- Strategy 2: CuPy MemoryPool (default) ---")
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    _sync()

    pool_results = {}
    for label, size in sizes:
        # Warmup (first alloc creates the pool bin)
        for _ in range(n_warmup):
            ptr = cp.cuda.alloc(size)
            del ptr
            _sync()

        latencies = []
        for _ in range(n_trials):
            _sync()
            t0 = time.perf_counter_ns()
            ptr = cp.cuda.alloc(size)
            _sync()
            t1 = time.perf_counter_ns()
            del ptr
            _sync()
            latencies.append((t1 - t0) / 1000)

        med = statistics.median(latencies)
        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        pool_results[label] = (med, p99)
        print(f"  {label:>8s}: median={med:10.1f} us  p99={p99:10.1f} us")

    pool.free_all_blocks()
    _sync()

    # Strategy 3: MemoryAsyncPool (cudaMallocAsync)
    print("\n--- Strategy 3: MemoryAsyncPool (cudaMallocAsync) ---")
    try:
        async_pool = cp.cuda.MemoryAsyncPool()
        cp.cuda.set_allocator(async_pool.malloc)
        _sync()

        async_results = {}
        for label, size in sizes:
            for _ in range(n_warmup):
                ptr = cp.cuda.alloc(size)
                del ptr
                _sync()

            latencies = []
            for _ in range(n_trials):
                _sync()
                t0 = time.perf_counter_ns()
                ptr = cp.cuda.alloc(size)
                _sync()
                t1 = time.perf_counter_ns()
                del ptr
                _sync()
                latencies.append((t1 - t0) / 1000)

            med = statistics.median(latencies)
            p99 = sorted(latencies)[int(0.99 * len(latencies))]
            async_results[label] = (med, p99)
            print(f"  {label:>8s}: median={med:10.1f} us  p99={p99:10.1f} us")
    except Exception as e:
        print(f"  AsyncPool failed: {e}")
        async_results = {}

    # Restore default
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
    _sync()

    # Summary
    print("\n--- Speedup: MemoryPool / raw cudaMalloc ---")
    for label in [l for l, _ in sizes]:
        if label in raw_results and label in pool_results:
            raw_med = raw_results[label][0]
            pool_med = pool_results[label][0]
            ratio = raw_med / max(pool_med, 0.001)
            print(f"  {label:>8s}: {ratio:6.1f}x faster with pool")

    if async_results:
        print("\n--- Speedup: AsyncPool / raw cudaMalloc ---")
        for label in [l for l, _ in sizes]:
            if label in raw_results and label in async_results:
                raw_med = raw_results[label][0]
                async_med = async_results[label][0]
                ratio = raw_med / max(async_med, 0.001)
                print(f"  {label:>8s}: {ratio:6.1f}x faster with async pool")

    return raw_results, pool_results, async_results


# ═══════════════════════════════════════════════════════════════════
# Experiment B: Allocation churn — many small alloc/free cycles
#   (simulates leaf pipeline: many temp arrays per tick)
# ═══════════════════════════════════════════════════════════════════

def exp_b_allocation_churn():
    """Simulate leaf pipeline churn: many temp arrays allocated and freed."""
    print("\n" + "=" * 72)
    print("EXP B: Allocation Churn — simulating leaf pipeline temporaries")
    print("=" * 72)

    # Simulate: 20 temp arrays of varying sizes, allocated and freed rapidly
    # This is what a leaf compute stage does: diff_n creates 5 arrays,
    # postprocess creates 5 padded arrays, etc.
    array_sizes = [
        600_000 * 8,   # 600K float64 values (~4.6 MB) — typical ticker day
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        590_000 * 8,   # lag-1 result
        598_000 * 8,   # lag-2 result
        595_000 * 8,   # lag-5 result
        590_000 * 8,   # lag-10 result
        600_000 * 8,   # padded output
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        1024,          # small metadata
        1024,
        4096,          # index arrays
        4096,
        8192,
        8192,
    ]

    n_cycles = 100

    strategies = []

    # Raw cudaMalloc
    cp.cuda.set_allocator(None)
    _sync()
    gc.collect()

    # Warmup
    for _ in range(5):
        ptrs = [cp.cuda.alloc(s) for s in array_sizes]
        del ptrs
        _sync()

    t0 = time.perf_counter()
    for _ in range(n_cycles):
        ptrs = [cp.cuda.alloc(s) for s in array_sizes]
        _sync()
        del ptrs
        _sync()
    elapsed_raw = time.perf_counter() - t0
    strategies.append(("raw cudaMalloc", elapsed_raw))

    # CuPy MemoryPool
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    _sync()

    for _ in range(5):
        ptrs = [cp.cuda.alloc(s) for s in array_sizes]
        del ptrs
        _sync()

    t0 = time.perf_counter()
    for _ in range(n_cycles):
        ptrs = [cp.cuda.alloc(s) for s in array_sizes]
        _sync()
        del ptrs
        _sync()
    elapsed_pool = time.perf_counter() - t0
    strategies.append(("CuPy MemoryPool", elapsed_pool))
    pool.free_all_blocks()
    _sync()

    # MemoryAsyncPool
    try:
        async_pool = cp.cuda.MemoryAsyncPool()
        cp.cuda.set_allocator(async_pool.malloc)
        _sync()

        for _ in range(5):
            ptrs = [cp.cuda.alloc(s) for s in array_sizes]
            del ptrs
            _sync()

        t0 = time.perf_counter()
        for _ in range(n_cycles):
            ptrs = [cp.cuda.alloc(s) for s in array_sizes]
            _sync()
            del ptrs
            _sync()
        elapsed_async = time.perf_counter() - t0
        strategies.append(("MemoryAsyncPool", elapsed_async))
    except Exception as e:
        print(f"  AsyncPool failed: {e}")

    # Restore default
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
    _sync()

    print(f"\n  {n_cycles} cycles x {len(array_sizes)} arrays each:")
    for name, elapsed in strategies:
        per_cycle_us = (elapsed / n_cycles) * 1e6
        per_alloc_us = per_cycle_us / len(array_sizes)
        print(f"  {name:>20s}: {elapsed*1000:8.1f} ms total, "
              f"{per_cycle_us:8.1f} us/cycle, {per_alloc_us:6.1f} us/alloc")

    fastest = min(strategies, key=lambda x: x[1])
    print(f"\n  Winner: {fastest[0]}")

    return strategies


# ═══════════════════════════════════════════════════════════════════
# Experiment C: WDDM VRAM ceiling — how much can we actually use?
# ═══════════════════════════════════════════════════════════════════

def exp_c_vram_ceiling():
    """Find the practical VRAM ceiling under WDDM."""
    print("\n" + "=" * 72)
    print("EXP C: WDDM VRAM Ceiling — practical usable limit")
    print("=" * 72)

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"  Total VRAM: {total / (1024**3):.2f} GB")
    print(f"  Free VRAM:  {free / (1024**3):.2f} GB")
    print(f"  Used VRAM:  {(total - free) / (1024**3):.2f} GB (OS/driver overhead)")

    # Use MemoryPool for this test
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    _sync()

    # Allocate in 1GB chunks until failure
    chunk_size = 1 * 1024 * 1024 * 1024  # 1 GB
    chunks = []
    total_allocated = 0

    print(f"\n  Allocating in 1 GB chunks...")
    while True:
        try:
            _sync()
            t0 = time.perf_counter_ns()
            chunk = cp.zeros(chunk_size // 8, dtype=cp.float64)  # actually touch memory
            _sync()
            t1 = time.perf_counter_ns()
            chunks.append(chunk)
            total_allocated += chunk_size
            alloc_us = (t1 - t0) / 1000
            free_now, _ = cp.cuda.runtime.memGetInfo()
            print(f"    Chunk {len(chunks):2d}: {total_allocated / (1024**3):.0f} GB allocated, "
                  f"{free_now / (1024**3):.2f} GB free, "
                  f"alloc took {alloc_us:,.0f} us")

            # Safety: stop at 88 GB to avoid WDDM paging thrash
            if total_allocated >= 88 * 1024 * 1024 * 1024:
                print(f"    Stopping at 88 GB safety limit")
                break
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"    OOM at {total_allocated / (1024**3):.0f} GB: {e}")
            break
        except Exception as e:
            print(f"    Error at {total_allocated / (1024**3):.0f} GB: {e}")
            break

    # Check if latency degraded as we approached the limit
    print(f"\n  Maximum usable VRAM: {total_allocated / (1024**3):.0f} GB "
          f"out of {total / (1024**3):.0f} GB total "
          f"({100 * total_allocated / total:.1f}%)")

    # Clean up
    del chunks
    pool.free_all_blocks()
    gc.collect()
    _sync()

    # Restore
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
    return total_allocated


# ═══════════════════════════════════════════════════════════════════
# Experiment D: Pinned memory transfer benchmark
# ═══════════════════════════════════════════════════════════════════

def exp_d_pinned_memory():
    """Benchmark pinned vs pageable host memory for H2D/D2H transfers."""
    print("\n" + "=" * 72)
    print("EXP D: Pinned vs Pageable Memory Transfers")
    print("=" * 72)

    sizes = [
        ("1 MB", 1 * 1024 * 1024),
        ("16 MB", 16 * 1024 * 1024),
        ("64 MB", 64 * 1024 * 1024),
        ("256 MB", 256 * 1024 * 1024),
        ("1 GB", 1024 * 1024 * 1024),
    ]

    n_trials = 20

    print("\n  H2D (Host -> Device) transfers:")
    for label, size in sizes:
        n_floats = size // 8

        # Pageable
        host_pageable = np.random.randn(n_floats).astype(np.float64)
        latencies_pageable = []
        for _ in range(n_trials):
            _sync()
            t0 = time.perf_counter_ns()
            d = cp.asarray(host_pageable)
            _sync()
            t1 = time.perf_counter_ns()
            del d
            _sync()
            latencies_pageable.append((t1 - t0) / 1e6)  # ms

        # Pinned
        pinned_pool = cp.get_default_pinned_memory_pool()
        pinned_mem = pinned_pool.malloc(size)
        host_pinned = np.frombuffer(pinned_mem, dtype=np.float64, count=n_floats)
        host_pinned[:] = host_pageable  # Copy data in

        latencies_pinned = []
        for _ in range(n_trials):
            _sync()
            t0 = time.perf_counter_ns()
            d = cp.asarray(host_pinned)
            _sync()
            t1 = time.perf_counter_ns()
            del d
            _sync()
            latencies_pinned.append((t1 - t0) / 1e6)

        med_p = statistics.median(latencies_pageable)
        med_pin = statistics.median(latencies_pinned)
        bw_p = (size / (1024**3)) / (med_p / 1000)
        bw_pin = (size / (1024**3)) / (med_pin / 1000)
        speedup = med_p / max(med_pin, 0.001)

        print(f"    {label:>6s}: pageable={med_p:7.2f} ms ({bw_p:5.1f} GB/s)  "
              f"pinned={med_pin:7.2f} ms ({bw_pin:5.1f} GB/s)  "
              f"speedup={speedup:.2f}x")

        del host_pageable, host_pinned, pinned_mem
        gc.collect()
        _sync()

    # D2H (Device -> Host) transfers
    print("\n  D2H (Device -> Host) transfers:")
    for label, size in sizes:
        n_floats = size // 8
        d_arr = cp.random.randn(n_floats)
        _sync()

        # Pageable D2H
        latencies_pageable = []
        for _ in range(n_trials):
            _sync()
            t0 = time.perf_counter_ns()
            h = cp.asnumpy(d_arr)
            t1 = time.perf_counter_ns()
            del h
            latencies_pageable.append((t1 - t0) / 1e6)

        # Pinned D2H
        latencies_pinned = []
        for _ in range(n_trials):
            pinned_mem = cp.get_default_pinned_memory_pool().malloc(size)
            host_pinned = np.frombuffer(pinned_mem, dtype=np.float64, count=n_floats)
            _sync()
            t0 = time.perf_counter_ns()
            d_arr.get(out=host_pinned)
            _sync()
            t1 = time.perf_counter_ns()
            del host_pinned, pinned_mem
            latencies_pinned.append((t1 - t0) / 1e6)

        med_p = statistics.median(latencies_pageable)
        med_pin = statistics.median(latencies_pinned)
        bw_p = (size / (1024**3)) / (med_p / 1000)
        bw_pin = (size / (1024**3)) / (med_pin / 1000)
        speedup = med_p / max(med_pin, 0.001)

        print(f"    {label:>6s}: pageable={med_p:7.2f} ms ({bw_p:5.1f} GB/s)  "
              f"pinned={med_pin:7.2f} ms ({bw_pin:5.1f} GB/s)  "
              f"speedup={speedup:.2f}x")

        del d_arr
        gc.collect()
        _sync()


# ═══════════════════════════════════════════════════════════════════
# Experiment E: Custom WDDM-aware pool allocator prototype
# ═══════════════════════════════════════════════════════════════════

class WDDMPool:
    """WDDM-aware GPU memory pool.

    Design principles:
    1. Pre-allocate a large arena to avoid repeated WDDM allocation calls
    2. Use power-of-2 bin sizes for fast matching (like RMM)
    3. Track VRAM headroom — never allocate past WDDM's comfort threshold
    4. Reuse freed blocks without returning to OS
    5. Support a high-water mark for monitoring

    Uses cp.cuda.memory.Memory (raw cudaMalloc) directly to avoid recursion
    through CuPy's allocator dispatch.
    """

    MIN_BIN = 12    # 4 KB minimum
    MAX_BIN = 34    # 16 GB maximum
    VRAM_HEADROOM = 4 * 1024 * 1024 * 1024  # Keep 4 GB free for WDDM/OS

    def __init__(self, *, headroom_bytes: int | None = None):
        # bin_power -> list of (Memory, MemoryPointer) tuples for reuse
        self._free_lists: dict[int, list] = {}
        self._bin_map: dict[int, int] = {}  # ptr_value -> bin_power
        self._total_pooled = 0
        self._total_in_use = 0
        self._high_water = 0
        self._alloc_count = 0
        self._reuse_count = 0
        self._headroom = headroom_bytes or self.VRAM_HEADROOM

    def _bin_for_size(self, size: int) -> int:
        """Find the smallest power-of-2 bin that fits the request."""
        if size <= 0:
            return self.MIN_BIN
        power = max(self.MIN_BIN, (size - 1).bit_length())
        return min(power, self.MAX_BIN)

    def malloc(self, size: int) -> cp.cuda.MemoryPointer:
        """Allocate from pool, reusing freed blocks when possible."""
        if size == 0:
            return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(0, 0, None), 0)

        bin_power = self._bin_for_size(size)
        bin_size = 1 << bin_power

        # Try to reuse from free list
        if bin_power in self._free_lists and self._free_lists[bin_power]:
            mem_obj, mem_ptr = self._free_lists[bin_power].pop()
            self._total_in_use += bin_size
            self._high_water = max(self._high_water, self._total_in_use)
            self._reuse_count += 1
            # Return an UnownedMemory-backed pointer so GC won't cudaFree it
            return cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(mem_ptr.ptr, bin_size, self), 0
            )

        # Check VRAM headroom
        free_vram, _ = cp.cuda.runtime.memGetInfo()
        if free_vram - bin_size < self._headroom:
            self._compact()
            free_vram, _ = cp.cuda.runtime.memGetInfo()
            if free_vram - bin_size < self._headroom:
                raise cp.cuda.memory.OutOfMemoryError(
                    bin_size, free_vram - self._headroom
                )

        # Raw cudaMalloc — bypasses CuPy allocator dispatch
        mem_obj = cp.cuda.memory.Memory(bin_size)
        mem_ptr = cp.cuda.MemoryPointer(mem_obj, 0)
        self._bin_map[mem_ptr.ptr] = bin_power
        self._total_pooled += bin_size
        self._total_in_use += bin_size
        self._high_water = max(self._high_water, self._total_in_use)
        self._alloc_count += 1
        # Keep the owning Memory alive; return UnownedMemory to caller
        self._free_lists.setdefault(bin_power, [])
        # Store the owning pair so we can reuse later
        # But we need to give it back to the caller now — save the owning ref
        self._bin_map[mem_ptr.ptr] = bin_power
        # We hold owning ref in a separate dict so it doesn't get freed
        if not hasattr(self, '_owners'):
            self._owners = {}
        self._owners[mem_ptr.ptr] = (mem_obj, mem_ptr)
        return cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(mem_ptr.ptr, bin_size, self), 0
        )

    def free(self, ptr_value: int, size: int = 0):
        """Return a block to the pool for reuse. Called with ptr int value."""
        if ptr_value == 0:
            return
        bin_power = self._bin_map.get(ptr_value)
        if bin_power is None:
            return
        bin_size = 1 << bin_power
        self._total_in_use -= bin_size
        owner = self._owners.get(ptr_value)
        if owner:
            self._free_lists.setdefault(bin_power, []).append(owner)

    def _compact(self):
        """Free the largest unused blocks back to WDDM."""
        for bin_power in sorted(self._free_lists.keys(), reverse=True):
            while self._free_lists[bin_power]:
                mem_obj, mem_ptr = self._free_lists[bin_power].pop()
                bin_size = 1 << bin_power
                self._total_pooled -= bin_size
                ptr_val = mem_ptr.ptr
                if ptr_val in self._bin_map:
                    del self._bin_map[ptr_val]
                if hasattr(self, '_owners') and ptr_val in self._owners:
                    del self._owners[ptr_val]
                del mem_ptr, mem_obj  # Triggers cudaFree via Memory destructor
                _sync()

    def free_all_blocks(self):
        """Free all pooled memory."""
        for bin_power in list(self._free_lists.keys()):
            while self._free_lists[bin_power]:
                mem_obj, mem_ptr = self._free_lists[bin_power].pop()
                bin_size = 1 << bin_power
                self._total_pooled -= bin_size
                ptr_val = mem_ptr.ptr
                if ptr_val in self._bin_map:
                    del self._bin_map[ptr_val]
                if hasattr(self, '_owners') and ptr_val in self._owners:
                    del self._owners[ptr_val]
                del mem_ptr, mem_obj
        self._total_in_use = 0
        _sync()

    def stats(self) -> dict:
        return {
            "total_pooled_mb": self._total_pooled / (1024 * 1024),
            "in_use_mb": self._total_in_use / (1024 * 1024),
            "high_water_mb": self._high_water / (1024 * 1024),
            "alloc_count": self._alloc_count,
            "reuse_count": self._reuse_count,
            "reuse_rate": self._reuse_count / max(1, self._alloc_count + self._reuse_count),
            "free_blocks": sum(len(v) for v in self._free_lists.values()),
        }


def exp_e_wddm_pool():
    """Benchmark custom WDDM pool vs CuPy defaults using direct alloc/free."""
    print("\n" + "=" * 72)
    print("EXP E: Custom WDDM Pool vs CuPy MemoryPool vs AsyncPool")
    print("=" * 72)

    # Same leaf-pipeline churn workload as Exp B
    array_sizes = [
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        590_000 * 8,
        598_000 * 8,
        595_000 * 8,
        590_000 * 8,
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        600_000 * 8,
        1024,
        1024,
        4096,
        4096,
        8192,
        8192,
    ]

    n_cycles = 200

    # CuPy MemoryPool baseline
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    _sync()

    for _ in range(10):
        ptrs = [pool.malloc(s) for s in array_sizes]
        del ptrs
        _sync()

    t0 = time.perf_counter()
    for _ in range(n_cycles):
        ptrs = [pool.malloc(s) for s in array_sizes]
        _sync()
        del ptrs
        _sync()
    elapsed_cupy = time.perf_counter() - t0
    pool.free_all_blocks()
    _sync()

    # WDDM Pool — call malloc/free directly, not through CuPy allocator
    wddm = WDDMPool(headroom_bytes=2 * 1024**3)
    _sync()

    for _ in range(10):
        ptrs = [wddm.malloc(s) for s in array_sizes]
        for ptr in ptrs:
            wddm.free(ptr.ptr)
        del ptrs
        _sync()

    t0 = time.perf_counter()
    for _ in range(n_cycles):
        ptrs = [wddm.malloc(s) for s in array_sizes]
        _sync()
        for ptr in ptrs:
            wddm.free(ptr.ptr)
        del ptrs
        _sync()
    elapsed_wddm = time.perf_counter() - t0

    stats = wddm.stats()
    wddm.free_all_blocks()
    _sync()

    # AsyncPool
    try:
        async_pool = cp.cuda.MemoryAsyncPool()
        cp.cuda.set_allocator(async_pool.malloc)
        _sync()

        for _ in range(10):
            ptrs = [async_pool.malloc(s) for s in array_sizes]
            del ptrs
            _sync()

        t0 = time.perf_counter()
        for _ in range(n_cycles):
            ptrs = [async_pool.malloc(s) for s in array_sizes]
            _sync()
            del ptrs
            _sync()
        elapsed_async = time.perf_counter() - t0
    except Exception as e:
        print(f"  AsyncPool failed: {e}")
        elapsed_async = None

    # Restore default
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
    _sync()

    per_cycle_cupy = (elapsed_cupy / n_cycles) * 1e6
    per_cycle_wddm = (elapsed_wddm / n_cycles) * 1e6
    print(f"\n  {n_cycles} cycles x {len(array_sizes)} arrays:")
    print(f"  {'CuPy MemoryPool':>20s}: {elapsed_cupy*1000:8.1f} ms total, "
          f"{per_cycle_cupy:8.1f} us/cycle")
    print(f"  {'WDDM Pool':>20s}: {elapsed_wddm*1000:8.1f} ms total, "
          f"{per_cycle_wddm:8.1f} us/cycle")
    if elapsed_async is not None:
        per_cycle_async = (elapsed_async / n_cycles) * 1e6
        print(f"  {'AsyncPool':>20s}: {elapsed_async*1000:8.1f} ms total, "
              f"{per_cycle_async:8.1f} us/cycle")

    print(f"\n  WDDM Pool stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")


# ═══════════════════════════════════════════════════════════════════
# Experiment F: First-alloc latency — cold start cost on WDDM
# ═══════════════════════════════════════════════════════════════════

def exp_f_cold_start():
    """Measure first-allocation latency (WDDM context init cost)."""
    print("\n" + "=" * 72)
    print("EXP F: Cold-Start Allocation Latency")
    print("=" * 72)

    # Reset allocator to raw
    cp.cuda.set_allocator(None)

    # Free everything
    gc.collect()
    _sync()

    sizes = [
        ("4 KB", 4096),
        ("1 MB", 1024 * 1024),
        ("64 MB", 64 * 1024 * 1024),
        ("1 GB", 1024 * 1024 * 1024),
    ]

    for label, size in sizes:
        # Try to measure the very first allocation after freeing
        gc.collect()
        _sync()
        t0 = time.perf_counter_ns()
        ptr = cp.cuda.alloc(size)
        _sync()
        t1 = time.perf_counter_ns()
        cold_us = (t1 - t0) / 1000

        # Subsequent allocations
        latencies = []
        for _ in range(50):
            _sync()
            ta = time.perf_counter_ns()
            p2 = cp.cuda.alloc(size)
            _sync()
            tb = time.perf_counter_ns()
            latencies.append((tb - ta) / 1000)
            del p2
            _sync()

        warm_med = statistics.median(latencies)
        del ptr
        _sync()

        print(f"  {label:>6s}: cold={cold_us:10.1f} us  warm_median={warm_med:10.1f} us  "
              f"cold/warm={cold_us/max(warm_med, 0.001):.1f}x")

    # Restore default
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("GPU Memory Management on Windows WDDM")
    print(f"Device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"VRAM: {free/(1024**3):.1f} GB free / {total/(1024**3):.1f} GB total")
    print(f"CuPy: {cp.__version__}, CUDA: {cp.cuda.runtime.runtimeGetVersion()}")
    print()

    exp_a_allocation_latency()
    exp_b_allocation_churn()
    exp_c_vram_ceiling()
    exp_d_pinned_memory()
    exp_e_wddm_pool()
    exp_f_cold_start()

    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 72)
