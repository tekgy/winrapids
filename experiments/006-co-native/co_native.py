"""
WinRapids Experiment 006: Co-Native Data Structures for GPU Computing

The fractal memory pool architecture: four tiers, same interface,
different physics underneath. Every buffer knows where it lives.
Every movement between tiers is explicit and costed.

The co-native split: an AI agent or human can query "where is my data?"
from CPU alone. No GPU round-trip needed. The representation works
natively for both kinds of minds.

Tiers:
  Device   — VRAM (~93 GB, ~1,700 GB/s), cudaMallocAsync pool
  Pinned   — pinned RAM (~128 GB, ~57 GB/s), cudaHostAlloc
  Pageable — standard RAM (larger, ~25 GB/s), malloc
  Storage  — NVMe (TB, ~7 GB/s), file handles

Each tier maps to Arrow device types where possible:
  ARROW_DEVICE_CUDA (2) → Device
  ARROW_DEVICE_CUDA_HOST (3) → Pinned
  ARROW_DEVICE_CPU (1) → Pageable
  Storage → WinRapids extension beyond Arrow
"""

from __future__ import annotations

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cupy as cp


# ============================================================
# Tier System
# ============================================================

class Tier(Enum):
    """Memory tier — where a buffer physically lives."""
    DEVICE = auto()    # GPU VRAM
    PINNED = auto()    # Pinned host memory (DMA-accessible)
    PAGEABLE = auto()  # Regular host memory
    STORAGE = auto()   # NVMe / file-backed

    @property
    def arrow_device_type(self) -> int:
        """Map to Arrow C Device Data Interface device types."""
        return {
            Tier.DEVICE: 2,    # ARROW_DEVICE_CUDA
            Tier.PINNED: 3,    # ARROW_DEVICE_CUDA_HOST
            Tier.PAGEABLE: 1,  # ARROW_DEVICE_CPU
            Tier.STORAGE: -1,  # Not in Arrow spec — WinRapids extension
        }[self]

    @property
    def label(self) -> str:
        return {
            Tier.DEVICE: "GPU",
            Tier.PINNED: "PIN",
            Tier.PAGEABLE: "CPU",
            Tier.STORAGE: "DSK",
        }[self]

    @property
    def bandwidth_gbps(self) -> float:
        """Approximate bandwidth in GB/s (from our experiments)."""
        return {
            Tier.DEVICE: 1677.0,   # GPU memory bandwidth (Exp 001)
            Tier.PINNED: 57.0,     # PCIe pinned transfer (Exp 002)
            Tier.PAGEABLE: 25.0,   # PCIe pageable transfer (Exp 002)
            Tier.STORAGE: 7.0,     # NVMe sequential read (estimated)
        }[self]


# Promotion costs between tiers (from our experiments, in GB/s)
TRANSFER_BANDWIDTH = {
    (Tier.PAGEABLE, Tier.DEVICE): 25.0,   # pageable H2D
    (Tier.PINNED, Tier.DEVICE): 57.0,     # pinned H2D
    (Tier.DEVICE, Tier.PINNED): 57.0,     # pinned D2H
    (Tier.DEVICE, Tier.PAGEABLE): 25.0,   # pageable D2H
    (Tier.PAGEABLE, Tier.PINNED): 30.0,   # memcpy + pin (estimated)
    (Tier.PINNED, Tier.PAGEABLE): 30.0,   # memcpy + unpin (estimated)
    (Tier.STORAGE, Tier.PAGEABLE): 7.0,   # NVMe read
    (Tier.PAGEABLE, Tier.STORAGE): 5.0,   # NVMe write
}


def estimate_transfer_ms(size_bytes: int, src: Tier, dst: Tier) -> float:
    """Estimate transfer cost in milliseconds between tiers."""
    if src == dst:
        return 0.0
    key = (src, dst)
    if key in TRANSFER_BANDWIDTH:
        bw = TRANSFER_BANDWIDTH[key]
    else:
        # Multi-hop: storage -> pageable -> device
        # Find the bottleneck
        if src == Tier.STORAGE and dst == Tier.DEVICE:
            # storage -> pageable -> device
            t1 = estimate_transfer_ms(size_bytes, Tier.STORAGE, Tier.PAGEABLE)
            t2 = estimate_transfer_ms(size_bytes, Tier.PAGEABLE, Tier.DEVICE)
            return t1 + t2
        return float("inf")
    return size_bytes / (bw * 1e9) * 1000  # ms


# ============================================================
# TieredColumn
# ============================================================

@dataclass
class TieredColumn:
    """
    A column with explicit memory tier tracking.

    CPU-resident metadata (always accessible):
      - name, dtype, length, null_count
      - tier: where the data lives right now
      - nbytes: size of the data
      - last_accessed: timestamp of last compute access
      - access_count: number of times accessed for compute

    Data (tier-dependent):
      - Tier.DEVICE: CuPy ndarray
      - Tier.PINNED: numpy ndarray backed by pinned memory
      - Tier.PAGEABLE: numpy ndarray
      - Tier.STORAGE: file path + offset (not implemented in this prototype)
    """
    name: str
    dtype: np.dtype
    length: int
    null_count: int
    tier: Tier
    nbytes: int
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    # The actual data — type depends on tier
    _device_data: Optional[cp.ndarray] = field(default=None, repr=False)
    _host_data: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_numpy(cls, name: str, arr: np.ndarray, tier: Tier = Tier.DEVICE) -> TieredColumn:
        """Create a column, placing it at the specified tier."""
        col = cls(
            name=name,
            dtype=arr.dtype,
            length=len(arr),
            null_count=0,
            tier=tier,
            nbytes=arr.nbytes,
        )
        if tier == Tier.DEVICE:
            col._device_data = cp.asarray(arr)
        elif tier == Tier.PINNED:
            pinned = cp.cuda.alloc_pinned_memory(arr.nbytes)
            pinned_arr = np.frombuffer(pinned, dtype=arr.dtype, count=len(arr))
            pinned_arr[:] = arr
            col._host_data = pinned_arr
        else:
            col._host_data = arr.copy()
        return col

    def promote(self, target: Tier) -> 'TieredColumn':
        """
        Move data to a different tier. Returns self (mutates in place).
        The transfer is explicit — the caller chose this.
        """
        if target == self.tier:
            return self

        t0 = time.perf_counter()

        if self.tier == Tier.DEVICE and target in (Tier.PAGEABLE, Tier.PINNED):
            # D2H
            np_data = cp.asnumpy(self._device_data)
            self._device_data = None
            if target == Tier.PINNED:
                pinned = cp.cuda.alloc_pinned_memory(np_data.nbytes)
                pinned_arr = np.frombuffer(pinned, dtype=np_data.dtype, count=len(np_data))
                pinned_arr[:] = np_data
                self._host_data = pinned_arr
            else:
                self._host_data = np_data
            self.tier = target

        elif self.tier in (Tier.PAGEABLE, Tier.PINNED) and target == Tier.DEVICE:
            # H2D
            self._device_data = cp.asarray(self._host_data)
            self._host_data = None
            self.tier = target

        elif self.tier == Tier.PAGEABLE and target == Tier.PINNED:
            # Page -> Pin
            pinned = cp.cuda.alloc_pinned_memory(self._host_data.nbytes)
            pinned_arr = np.frombuffer(pinned, dtype=self._host_data.dtype,
                                        count=len(self._host_data))
            pinned_arr[:] = self._host_data
            self._host_data = pinned_arr
            self.tier = target

        elif self.tier == Tier.PINNED and target == Tier.PAGEABLE:
            # Pin -> Page
            self._host_data = np.array(self._host_data)
            self.tier = target

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return self

    def ensure_device(self) -> cp.ndarray:
        """Get data on GPU, promoting if necessary."""
        if self.tier != Tier.DEVICE:
            self.promote(Tier.DEVICE)
        self.last_accessed = time.time()
        self.access_count += 1
        return self._device_data

    def ensure_host(self) -> np.ndarray:
        """Get data on CPU, demoting if necessary."""
        if self.tier == Tier.DEVICE:
            self.promote(Tier.PAGEABLE)
        self.last_accessed = time.time()
        return self._host_data

    @property
    def promotion_cost_ms(self) -> float:
        """Estimated cost to promote to GPU (0 if already there)."""
        return estimate_transfer_ms(self.nbytes, self.tier, Tier.DEVICE)

    def __repr__(self) -> str:
        return (f"TieredColumn('{self.name}', {self.dtype}, n={self.length:,}, "
                f"tier={self.tier.label}, {self.nbytes/1e6:.1f}MB, "
                f"cost={self.promotion_cost_ms:.1f}ms)")


# ============================================================
# TieredFrame
# ============================================================

class TieredFrame:
    """
    A DataFrame with explicit memory tier management.

    Every column knows where it lives. Movements are explicit.
    The memory map is the query planner's input.
    """

    def __init__(self, columns: dict[str, TieredColumn]):
        self._columns = columns
        lengths = {col.length for col in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"Column lengths don't match: {lengths}")
        self._nrows = lengths.pop() if lengths else 0

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray], tiers: Optional[dict[str, Tier]] = None) -> TieredFrame:
        """Create from numpy arrays, optionally specifying tiers."""
        if tiers is None:
            tiers = {}
        columns = {}
        for name, arr in data.items():
            tier = tiers.get(name, Tier.DEVICE)
            columns[name] = TieredColumn.from_numpy(name, arr, tier=tier)
        return cls(columns)

    def __getitem__(self, key: str) -> TieredColumn:
        return self._columns[key]

    def __len__(self) -> int:
        return self._nrows

    @property
    def columns(self) -> list[str]:
        return list(self._columns.keys())

    def memory_map(self) -> str:
        """
        Show per-column memory residency, cost to promote, access patterns.
        This IS the query planner's input. Co-native: readable by human AND AI.
        """
        lines = [
            f"TieredFrame: {self._nrows:,} rows x {len(self._columns)} columns",
            "",
            f"  {'Column':<20s} {'Type':<10s} {'Size':>8s} {'Tier':>4s} "
            f"{'GPU Cost':>10s} {'Accesses':>10s}",
            f"  {'-'*20} {'-'*10} {'-'*8} {'-'*4} {'-'*10} {'-'*10}",
        ]

        total_bytes = {tier: 0 for tier in Tier}

        for name, col in self._columns.items():
            cost = col.promotion_cost_ms
            cost_str = "0 (here)" if cost == 0 else f"{cost:.1f} ms"
            lines.append(
                f"  {name:<20s} {str(col.dtype):<10s} {col.nbytes/1e6:>7.1f}M "
                f"{col.tier.label:>4s} {cost_str:>10s} {col.access_count:>10d}"
            )
            total_bytes[col.tier] += col.nbytes

        lines.append("")
        for tier in Tier:
            if total_bytes[tier] > 0:
                lines.append(f"  {tier.label}: {total_bytes[tier]/1e6:.1f} MB")

        return "\n".join(lines)

    def promote(self, col_name: str, target: Tier) -> None:
        """Explicitly move a column to a different tier."""
        self._columns[col_name].promote(target)

    def promote_all(self, target: Tier) -> None:
        """Move all columns to target tier."""
        for col in self._columns.values():
            col.promote(target)

    def query_plan(self, needed_cols: list[str]) -> dict:
        """
        Analyze what needs to happen to run a computation.
        Returns a plan showing which columns need promotion and the cost.

        This is the co-native query planner: an AI agent can read this
        plan and decide whether to proceed, or suggest a different approach.
        """
        plan = {
            "columns": {},
            "total_transfer_mb": 0.0,
            "total_transfer_ms": 0.0,
            "already_on_gpu": [],
            "needs_promotion": [],
        }

        for name in needed_cols:
            col = self._columns[name]
            cost_ms = col.promotion_cost_ms
            plan["columns"][name] = {
                "current_tier": col.tier.label,
                "size_mb": col.nbytes / 1e6,
                "promotion_cost_ms": cost_ms,
            }
            if col.tier == Tier.DEVICE:
                plan["already_on_gpu"].append(name)
            else:
                plan["needs_promotion"].append(name)
                plan["total_transfer_mb"] += col.nbytes / 1e6
                plan["total_transfer_ms"] += cost_ms

        return plan

    def __repr__(self) -> str:
        tier_counts = {}
        for col in self._columns.values():
            tier_counts[col.tier.label] = tier_counts.get(col.tier.label, 0) + 1
        tier_str = ", ".join(f"{k}:{v}" for k, v in tier_counts.items())
        return f"TieredFrame({self._nrows:,} rows, {len(self._columns)} cols, [{tier_str}])"


# ============================================================
# Smart Compute — uses query plan to minimize transfers
# ============================================================

def smart_filtered_sum(frame: TieredFrame, value_col: str, mask_col: str,
                       mask_val, verbose: bool = True) -> float:
    """
    Filtered sum with tier-aware execution.
    Uses the query plan to show what transfers are needed, then executes.
    """
    # Step 1: Query plan
    plan = frame.query_plan([value_col, mask_col])

    if verbose:
        print(f"\n  Query Plan: sum({value_col}) where {mask_col} == {mask_val}")
        for name, info in plan["columns"].items():
            status = "ready" if info["promotion_cost_ms"] == 0 else f"need {info['promotion_cost_ms']:.1f}ms"
            print(f"    {name}: {info['current_tier']} ({info['size_mb']:.1f} MB) — {status}")
        if plan["needs_promotion"]:
            print(f"    Total transfer: {plan['total_transfer_mb']:.1f} MB in {plan['total_transfer_ms']:.1f} ms")
        else:
            print(f"    All data on GPU — no transfer needed")
        print()

    # Step 2: Promote needed columns
    t0 = time.perf_counter()
    for name in plan["needs_promotion"]:
        frame.promote(name, Tier.DEVICE)
    cp.cuda.Device(0).synchronize()
    t_promote = (time.perf_counter() - t0) * 1000

    # Step 3: Compute on GPU
    t0 = time.perf_counter()
    values_gpu = frame[value_col].ensure_device()
    mask_gpu = frame[mask_col].ensure_device()
    result = float(cp.sum(values_gpu[mask_gpu == mask_val]))
    cp.cuda.Device(0).synchronize()
    t_compute = (time.perf_counter() - t0) * 1000

    if verbose:
        print(f"  Promotion: {t_promote:.3f} ms")
        print(f"  Compute:   {t_compute:.3f} ms")
        print(f"  Total:     {t_promote + t_compute:.3f} ms")
        print(f"  Result:    {result:.6f}")

    return result


# ============================================================
# Tests and Demos
# ============================================================

def test_tier_system():
    """Test basic tier operations."""
    print("=== Test 1: Tier System ===\n")

    n = 1_000_000
    data = np.random.randn(n).astype(np.float64)

    # Create at different tiers
    col_gpu = TieredColumn.from_numpy("gpu_col", data, Tier.DEVICE)
    col_pin = TieredColumn.from_numpy("pin_col", data, Tier.PINNED)
    col_cpu = TieredColumn.from_numpy("cpu_col", data, Tier.PAGEABLE)

    print(f"  {col_gpu}")
    print(f"  {col_pin}")
    print(f"  {col_cpu}")
    print()

    # Promotion costs
    print(f"  GPU->GPU cost: {col_gpu.promotion_cost_ms:.1f} ms")
    print(f"  PIN->GPU cost: {col_pin.promotion_cost_ms:.1f} ms")
    print(f"  CPU->GPU cost: {col_cpu.promotion_cost_ms:.1f} ms")
    print()

    # Promote CPU -> GPU
    t0 = time.perf_counter()
    col_cpu.promote(Tier.DEVICE)
    cp.cuda.Device(0).synchronize()
    t_actual = (time.perf_counter() - t0) * 1000

    print(f"  CPU->GPU actual: {t_actual:.3f} ms (estimated: {estimate_transfer_ms(data.nbytes, Tier.PAGEABLE, Tier.DEVICE):.1f} ms)")
    print(f"  After promotion: {col_cpu}")
    print()

    # Verify data integrity
    gpu_sum = float(cp.sum(col_cpu._device_data))
    expected_sum = float(np.sum(data))
    print(f"  Data integrity: {'PASS' if abs(gpu_sum - expected_sum) < 0.01 else 'FAIL'}")
    print()


def test_memory_map():
    """Test the memory map — the co-native interface."""
    print("=== Test 2: Memory Map (Co-Native Interface) ===\n")

    n = 5_000_000
    frame = TieredFrame.from_dict(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "price": np.random.randn(n).astype(np.float64),
            "volume": np.random.randint(0, 10000, n).astype(np.int32),
            "flag": np.random.randint(0, 2, n).astype(np.int8),
        },
        tiers={
            "timestamp": Tier.PAGEABLE,  # Rarely used, keep on CPU
            "price": Tier.DEVICE,        # Hot column, on GPU
            "volume": Tier.DEVICE,       # Hot column, on GPU
            "flag": Tier.PINNED,         # Moderate use, pinned for fast transfer
        }
    )

    print(frame.memory_map())
    print()
    print(f"  repr: {frame}")
    print()


def test_query_plan():
    """Test the query planner."""
    print("=== Test 3: Query Plan ===\n")

    n = 10_000_000
    frame = TieredFrame.from_dict(
        {
            "value": np.random.randn(n).astype(np.float64),
            "category": np.random.randint(0, 10, n).astype(np.int32),
            "flag": np.random.randint(0, 2, n).astype(np.int8),
        },
        tiers={
            "value": Tier.DEVICE,
            "category": Tier.PAGEABLE,
            "flag": Tier.PINNED,
        }
    )

    print("  Memory state:")
    print(frame.memory_map())
    print()

    # Plan 1: filter on flag (pinned) — needs promotion
    plan = frame.query_plan(["value", "flag"])
    print("  Plan: sum(value) where flag == 1")
    print(f"    Already on GPU: {plan['already_on_gpu']}")
    print(f"    Needs promotion: {plan['needs_promotion']}")
    print(f"    Transfer cost: {plan['total_transfer_ms']:.1f} ms ({plan['total_transfer_mb']:.1f} MB)")
    print()

    # Plan 2: filter on category (pageable) — more expensive
    plan2 = frame.query_plan(["value", "category"])
    print("  Plan: sum(value) where category == 5")
    print(f"    Already on GPU: {plan2['already_on_gpu']}")
    print(f"    Needs promotion: {plan2['needs_promotion']}")
    print(f"    Transfer cost: {plan2['total_transfer_ms']:.1f} ms ({plan2['total_transfer_mb']:.1f} MB)")
    print()


def benchmark_tiered_execution():
    """Benchmark the cost of tier-aware execution."""
    print("=== Benchmark: Tiered Filtered Sum ===\n")

    n = 10_000_000
    np.random.seed(42)
    values = np.random.randn(n).astype(np.float64)
    flags = np.random.randint(0, 2, n).astype(np.int8)

    # Scenario 1: Everything on GPU
    frame_gpu = TieredFrame.from_dict(
        {"value": values, "flag": flags},
        tiers={"value": Tier.DEVICE, "flag": Tier.DEVICE}
    )
    print("  Scenario 1: All on GPU")
    result1 = smart_filtered_sum(frame_gpu, "value", "flag", 1)
    print()

    # Scenario 2: Value on GPU, flag on pinned
    frame_mixed = TieredFrame.from_dict(
        {"value": values, "flag": flags},
        tiers={"value": Tier.DEVICE, "flag": Tier.PINNED}
    )
    print("  Scenario 2: Value on GPU, flag on pinned")
    result2 = smart_filtered_sum(frame_mixed, "value", "flag", 1)
    print()

    # Scenario 3: Everything on CPU (worst case)
    frame_cpu = TieredFrame.from_dict(
        {"value": values, "flag": flags},
        tiers={"value": Tier.PAGEABLE, "flag": Tier.PAGEABLE}
    )
    print("  Scenario 3: All on CPU")
    result3 = smart_filtered_sum(frame_cpu, "value", "flag", 1)
    print()

    # Scenario 4: Second run (now promoted — should be instant)
    print("  Scenario 4: Re-run (data already promoted)")
    result4 = smart_filtered_sum(frame_cpu, "value", "flag", 1)
    print()

    # Compare with pandas
    import pandas as pd
    pdf = pd.DataFrame({"value": values, "flag": flags})
    t0 = time.perf_counter()
    for _ in range(5):
        pd_result = pdf.loc[pdf["flag"] == 1, "value"].sum()
    t_pandas = (time.perf_counter() - t0) / 5 * 1000

    print(f"  pandas baseline: {t_pandas:.3f} ms (result: {pd_result:.6f})")
    print()

    # Verify all results match
    assert abs(result1 - pd_result) < 0.01
    assert abs(result2 - pd_result) < 0.01
    assert abs(result3 - pd_result) < 0.01
    assert abs(result4 - pd_result) < 0.01
    print("  All results match. PASS\n")


def demo_co_native_query():
    """
    Demonstrate how an AI agent would interact with the tiered frame.
    The agent reads the memory map, makes decisions, never touches GPU directly.
    """
    print("=== Demo: Co-Native Agent Interaction ===\n")

    n = 10_000_000
    np.random.seed(42)

    frame = TieredFrame.from_dict(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": np.random.randn(n).astype(np.float64) * 10 + 100,
            "high": np.random.randn(n).astype(np.float64) * 10 + 105,
            "low": np.random.randn(n).astype(np.float64) * 10 + 95,
            "close": np.random.randn(n).astype(np.float64) * 10 + 100,
            "volume": np.random.randint(100, 100000, n).astype(np.int64),
            "symbol": np.random.randint(0, 100, n).astype(np.int32),
        },
        tiers={
            "timestamp": Tier.PAGEABLE,
            "open": Tier.PAGEABLE,
            "high": Tier.DEVICE,
            "low": Tier.DEVICE,
            "close": Tier.DEVICE,
            "volume": Tier.PINNED,
            "symbol": Tier.PAGEABLE,
        }
    )

    # Agent reads memory map — no GPU access needed
    print("  [Agent reads memory map — CPU only, no GPU round-trip]\n")
    print(frame.memory_map())
    print()

    # Agent sees: close and high are on GPU, open needs promotion
    # Agent decides: for a simple high-low range check, no promotion needed
    plan = frame.query_plan(["high", "low"])
    print(f"  [Agent queries plan for high-low range]")
    print(f"    Need promotion: {plan['needs_promotion']}")
    print(f"    Transfer cost: {plan['total_transfer_ms']:.1f} ms")
    print()

    # But for a full OHLCV analysis, open needs to come to GPU
    plan2 = frame.query_plan(["open", "high", "low", "close", "volume"])
    print(f"  [Agent queries plan for full OHLCV analysis]")
    print(f"    Need promotion: {plan2['needs_promotion']}")
    print(f"    Transfer cost: {plan2['total_transfer_ms']:.1f} ms ({plan2['total_transfer_mb']:.1f} MB)")
    print()

    # Agent can decide to promote hot columns proactively
    print("  [Agent decision: promote 'open' and 'volume' proactively]")
    t0 = time.perf_counter()
    frame.promote("open", Tier.DEVICE)
    frame.promote("volume", Tier.DEVICE)
    cp.cuda.Device(0).synchronize()
    t_promote = (time.perf_counter() - t0) * 1000
    print(f"  Promotion took: {t_promote:.3f} ms\n")

    print("  [Memory map after promotion]\n")
    print(frame.memory_map())
    print()


if __name__ == "__main__":
    print("WinRapids Experiment 006: Co-Native Data Structures")
    print("=" * 60)
    print()

    test_tier_system()
    test_memory_map()
    test_query_plan()
    benchmark_tiered_execution()
    demo_co_native_query()

    print("=" * 60)
    print("Experiment 006 complete.")
