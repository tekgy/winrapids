"""GPU BinEngine — parallel prefix sums and bin aggregation on GPU.

This is the core K02 bottleneck for Market Atlas. The CPU BinEngine
(fintek's prefix_engine.py) does O(1) range queries via prefix sums.
This GPU version does the same thing but with CuPy, enabling:
- Parallel prefix scan (O(log N) instead of O(N))
- Fused bin statistics (one kernel, all stats)
- Batched multi-cadence computation

The interface mirrors fintek's BinEngine so it can be a drop-in replacement.

Usage:
    from winrapids.bin_engine import GPUBinEngine

    engine = GPUBinEngine(prices_gpu, timestamps_gpu, cadence_boundaries)
    means = engine.bin_means(cadence_id)
    stds = engine.bin_stds(cadence_id)
    all_stats = engine.bin_all_stats(cadence_id)  # fused: one pass
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import cupy as cp

if TYPE_CHECKING:
    pass


class GPUBinEngine:
    """GPU-accelerated bin aggregation via CuPy prefix sums.

    Holds raw tick data on GPU. Computes per-bin statistics using
    prefix sum subtraction (O(1) per bin after O(N) prefix build).
    """

    def __init__(
        self,
        columns: dict[str, cp.ndarray],
        timestamps_ns: cp.ndarray,
        cadence_boundaries: dict[int, cp.ndarray],
    ):
        """Initialize with GPU arrays and precomputed bin boundaries.

        Args:
            columns: Named GPU arrays (e.g., {"price": cp.array(...)})
            timestamps_ns: Tick timestamps in nanoseconds (GPU)
            cadence_boundaries: {cadence_id: cp.array of bin boundary indices}
        """
        self._columns = columns
        self._timestamps = timestamps_ns
        self._boundaries = cadence_boundaries
        self._n = len(timestamps_ns)

        # Build prefix sums for all columns
        self._prefix_sum: dict[str, cp.ndarray] = {}
        self._prefix_sum_sq: dict[str, cp.ndarray] = {}
        self._prefix_count: dict[int, cp.ndarray] = {}

        for name, arr in columns.items():
            # Prefix sum: ps[i] = sum(arr[0:i])
            # bin_sum(a, b) = ps[b] - ps[a]
            self._prefix_sum[name] = cp.concatenate([cp.zeros(1, dtype=arr.dtype), cp.cumsum(arr)])
            self._prefix_sum_sq[name] = cp.concatenate([cp.zeros(1, dtype=arr.dtype), cp.cumsum(arr ** 2)])

    @classmethod
    def from_numpy(
        cls,
        columns: dict[str, np.ndarray],
        timestamps_ns: np.ndarray,
        cadence_boundaries: dict[int, np.ndarray],
    ) -> GPUBinEngine:
        """Create from numpy arrays — handles H2D transfer."""
        from winrapids.transfer import h2d

        gpu_columns = {name: h2d(arr.astype(np.float64)) for name, arr in columns.items()}
        gpu_ts = h2d(timestamps_ns.astype(np.float64))
        gpu_boundaries = {cid: h2d(bounds.astype(np.int64)) for cid, bounds in cadence_boundaries.items()}

        return cls(gpu_columns, gpu_ts, gpu_boundaries)

    def _get_boundaries(self, cadence_id: int) -> cp.ndarray:
        """Get bin boundary indices for a cadence."""
        return self._boundaries[cadence_id]

    def n_bins(self, cadence_id: int) -> int:
        """Number of bins at this cadence."""
        return int(len(self._boundaries[cadence_id]) - 1)

    def bin_counts(self, cadence_id: int) -> cp.ndarray:
        """Tick count per bin. O(1) via boundary subtraction."""
        b = self._get_boundaries(cadence_id)
        return b[1:] - b[:-1]

    def bin_sums(self, col: str, cadence_id: int) -> cp.ndarray:
        """Sum per bin. O(1) via prefix subtraction."""
        ps = self._prefix_sum[col]
        b = self._get_boundaries(cadence_id)
        return ps[b[1:]] - ps[b[:-1]]

    def bin_means(self, col: str, cadence_id: int) -> cp.ndarray:
        """Mean per bin. O(1) via prefix sum / count."""
        sums = self.bin_sums(col, cadence_id)
        counts = self.bin_counts(cadence_id).astype(cp.float64)
        # NaN for empty bins
        result = cp.where(counts > 0, sums / counts, cp.nan)
        return result

    def bin_sum_sq(self, col: str, cadence_id: int) -> cp.ndarray:
        """Sum of squares per bin. O(1) via prefix subtraction."""
        ps2 = self._prefix_sum_sq[col]
        b = self._get_boundaries(cadence_id)
        return ps2[b[1:]] - ps2[b[:-1]]

    def bin_variances(self, col: str, cadence_id: int) -> cp.ndarray:
        """Variance per bin. O(1) via prefix sum of squares."""
        counts = self.bin_counts(cadence_id).astype(cp.float64)
        means = self.bin_means(col, cadence_id)
        sum_sq = self.bin_sum_sq(col, cadence_id)
        # var = E[X^2] - (E[X])^2
        result = cp.where(counts > 1, sum_sq / counts - means ** 2, cp.nan)
        # Clamp small negative values from float arithmetic
        return cp.maximum(result, 0.0)

    def bin_stds(self, col: str, cadence_id: int) -> cp.ndarray:
        """Standard deviation per bin."""
        return cp.sqrt(self.bin_variances(col, cadence_id))

    def bin_mins(self, col: str, cadence_id: int) -> cp.ndarray:
        """Min per bin. Single fused kernel launch via fused_bin_stats."""
        return self._fused_stats(col, cadence_id)["min"]

    def bin_maxs(self, col: str, cadence_id: int) -> cp.ndarray:
        """Max per bin. Single fused kernel launch via fused_bin_stats."""
        return self._fused_stats(col, cadence_id)["max"]

    def bin_firsts(self, col: str, cadence_id: int) -> cp.ndarray:
        """First tick value per bin. O(1) via boundary index."""
        data = self._columns[col]
        b = self._get_boundaries(cadence_id)
        counts = self.bin_counts(cadence_id)
        result = cp.where(counts > 0, data[b[:-1]], cp.nan)
        return result

    def bin_lasts(self, col: str, cadence_id: int) -> cp.ndarray:
        """Last tick value per bin. O(1) via boundary index."""
        data = self._columns[col]
        b = self._get_boundaries(cadence_id)
        counts = self.bin_counts(cadence_id)
        # Last index is b[i+1] - 1, but need to handle empty bins
        last_idx = cp.maximum(b[1:] - 1, b[:-1])
        result = cp.where(counts > 0, data[last_idx], cp.nan)
        return result

    def _fused_stats(self, col: str, cadence_id: int) -> dict[str, cp.ndarray]:
        """Compute all stats via single fused CUDA kernel launch.

        One kernel launch computes count, sum, sum_sq, min, max, first,
        last + derived mean, variance, std for all bins. Replaces the
        N-kernel-launch loops that bin_mins/bin_maxs previously used.
        """
        from winrapids.fused_bin_stats import fused_bin_stats as _fused

        data = self._columns[col]
        boundaries = self._get_boundaries(cadence_id)
        return _fused(data, boundaries)

    def bin_all_stats(self, col: str, cadence_id: int) -> dict[str, cp.ndarray]:
        """All basic statistics in ONE fused kernel launch.

        Returns dict with: count, sum, sum_sq, mean, variance, std,
        min, max, first, last.
        """
        return self._fused_stats(col, cadence_id)

    def slice(self, col: str, cadence_id: int, bin_idx: int) -> cp.ndarray:
        """Raw tick array for one bin. GPU array."""
        b = self._get_boundaries(cadence_id)
        lo, hi = int(b[bin_idx]), int(b[bin_idx + 1])
        return self._columns[col][lo:hi]

    def column(self, col: str) -> cp.ndarray:
        """Raw full column on GPU."""
        return self._columns[col]
