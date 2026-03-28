"""KO05 sufficient statistics — GPU extraction for standalone MKTF files.

Each cadence level gets its own KO05 file with sufficient statistics
as regular MKTF columns. Same GPU kernel produces both KO00 (full data)
and KO05 (stats) — the kernel already computes {sum, sum_sq, min, max,
first, last, count}; KO05 uses 5 of 7.

Column layout: one BUNDLED column per source column. Each column stores
(n_bins × 5) float32 values = [sum, sum_sq, min, max, count] interleaved
per bin. 5 columns instead of 25 → 4x smaller files, 40% faster writes
(alignment: 5×4096 vs 25×4096).

Usage:
    from winrapids.progressive import extract_ko05_columns

    # Extract stats for one cadence level:
    columns = extract_ko05_columns(engine, cadence_id=2)
    write_mktf(path, columns, ko=5, domain=5, ...)

    # Iterate the default cadence grid:
    for cadence_ms, cadence_id in zip(DEFAULT_CADENCES_MS, range(8)):
        columns = extract_ko05_columns(engine, cadence_id)
        write_mktf(ko05_path(cadence_ms), columns, ko=5, domain=5, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from winrapids.bin_engine import GPUBinEngine


STAT_FIELDS = 5  # sum, sum_sq, min, max, count
STAT_NAMES = ("sum", "sum_sq", "min", "max", "count")

# Default cadence grid: session through 30s.
# Each cadence level becomes its own KO05 file.
DEFAULT_CADENCES_MS = [
    86_400_000,   # session (~1 bin)
    1_800_000,    # 30 min (~13 bins)
    900_000,      # 15 min (~26 bins)
    600_000,      # 10 min (~39 bins)
    300_000,      # 5 min (~78 bins) — institutional fingerprint boundary
    120_000,      # 2 min (~195 bins)
    60_000,       # 1 min (~390 bins)
    30_000,       # 30 sec (~780 bins)
]


def extract_ko05_columns(
    engine: GPUBinEngine,
    cadence_id: int,
) -> dict[str, np.ndarray]:
    """Extract sufficient statistics as bundled MKTF columns for one cadence.

    One fused kernel launch per source column. Results transferred D2H as
    float32 and interleaved into one bundled column per source column.

    Args:
        engine: GPUBinEngine with columns and cadence boundaries loaded.
        cadence_id: Key into engine's boundary dict for this cadence.

    Returns:
        Dict of {col_name + "_stats": float32[n_bins * 5]} ready for
        write_mktf(). The 5 values per bin are [sum, sum_sq, min, max,
        count] interleaved. Use unpack_ko05_column() to reshape on read.
    """
    col_names = list(engine._columns.keys())
    n_bins = engine.n_bins(cadence_id)
    columns: dict[str, np.ndarray] = {}

    for name in col_names:
        fused = engine.bin_all_stats(name, cadence_id)

        # Extract 5 stats, transfer D2H, interleave into bundled column
        stats = np.empty((n_bins, STAT_FIELDS), dtype=np.float32)
        stats[:, 0] = fused["sum"].get().astype(np.float32)[:n_bins]
        stats[:, 1] = fused["sum_sq"].get().astype(np.float32)[:n_bins]
        stats[:, 2] = fused["min"].get().astype(np.float32)[:n_bins]
        stats[:, 3] = fused["max"].get().astype(np.float32)[:n_bins]
        stats[:, 4] = fused["count"].get().astype(np.float32)[:n_bins]

        columns[f"{name}_stats"] = stats.ravel()

    return columns


def unpack_ko05_column(flat: np.ndarray) -> np.ndarray:
    """Reshape a bundled KO05 column to (n_bins, 5).

    Args:
        flat: float32[n_bins * 5] from read_columns().

    Returns:
        (n_bins, 5) float32 array: [sum, sum_sq, min, max, count] per bin.
    """
    return flat.reshape(-1, STAT_FIELDS)
