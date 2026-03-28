"""KO05 sufficient statistics — GPU extraction for standalone MKTF files.

Each cadence level gets its own KO05 file with sufficient statistics
as regular MKTF columns. Same GPU kernel produces both KO00 (full data)
and KO05 (stats) — the kernel already computes {sum, sum_sq, min, max,
first, last, count}; KO05 uses 5 of 7.

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
STAT_SUFFIXES = ("_sum", "_sum_sq", "_min", "_max", "_count")

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
    """Extract sufficient statistics as flat MKTF columns for one cadence.

    One fused kernel launch per column. Results transferred D2H as float32.

    Args:
        engine: GPUBinEngine with columns and cadence boundaries loaded.
        cadence_id: Key into engine's boundary dict for this cadence.

    Returns:
        Dict of {col_stat: float32[n_bins]} ready for write_mktf().
        For source columns [price, volume], returns:
          price_sum, price_sum_sq, price_min, price_max, price_count,
          volume_sum, volume_sum_sq, volume_min, volume_max, volume_count
    """
    col_names = list(engine._columns.keys())
    n_bins = engine.n_bins(cadence_id)
    columns: dict[str, np.ndarray] = {}

    for name in col_names:
        fused = engine.bin_all_stats(name, cadence_id)

        # Extract the 5 stats and transfer D2H
        raw = [
            fused["sum"].get().astype(np.float32),
            fused["sum_sq"].get().astype(np.float32),
            fused["min"].get().astype(np.float32),
            fused["max"].get().astype(np.float32),
            fused["count"].get().astype(np.float32),
        ]

        for suffix, arr in zip(STAT_SUFFIXES, raw):
            columns[f"{name}{suffix}"] = arr[:n_bins]

    return columns
