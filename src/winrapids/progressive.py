"""Progressive sufficient statistics — GPU extraction for MKTF writer.

Bridges the GPUBinEngine (fused kernel output on GPU) with the MKTF
writer's progressive_stats parameter (CPU float32 arrays).

Usage:
    from winrapids.progressive import extract_progressive_stats

    # Explicit cadence mapping:
    prog_stats = extract_progressive_stats(
        engine,
        cadences=[(86_400_000, 0), (3_600_000, 1), (60_000, 2), (30_000, 3)],
    )

    # Or use the default 7-level grid (session through 30s):
    prog_stats = extract_progressive_stats(engine)

    write_mktf(path, columns, progressive_stats=prog_stats)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from winrapids.bin_engine import GPUBinEngine


STAT_FIELDS = 5  # sum, sum_sq, min, max, count

# Default cadence grid: session through 30s (~800 bins total).
# Observer Experiment 16 found progressive reads beat full column reads
# up to 1,548 bins (crossover). 30s = 780 bins, safely under.
# Cadences finer than 30s should read raw columns instead.
# Format: (cadence_ms, description) — cadence_id mapping is caller's job.
DEFAULT_CADENCES_MS = [
    86_400_000,   # session (~1 bin)
    1_800_000,    # 30 min (~13 bins)
    900_000,      # 15 min (~26 bins)
    600_000,      # 10 min (~39 bins)
    300_000,      # 5 min (~78 bins) — institutional fingerprint boundary
    120_000,      # 2 min (~195 bins)
    60_000,       # 1 min (~390 bins)
    30_000,       # 30 sec (~780 bins) — crossover floor
]


def extract_progressive_stats(
    engine: GPUBinEngine,
    cadences: list[tuple[int, int]] | None = None,
) -> list[tuple[int, dict[str, np.ndarray]]]:
    """Extract progressive stats from GPUBinEngine via fused kernel.

    One fused kernel launch per (column, cadence) pair. Results are
    transferred D2H and cast to float32 for the MKTF writer.

    Args:
        engine: GPUBinEngine with columns and cadence boundaries loaded.
        cadences: List of (cadence_ms, cadence_id) pairs, ordered
                  coarsest-first. cadence_ms is the label stored in
                  the progressive section; cadence_id is the key into
                  engine's boundary dict.
                  If None, uses DEFAULT_CADENCES_MS with cadence_id =
                  index into engine's boundary dict (assumes boundaries
                  are keyed by ascending cadence_id matching the default
                  grid order).

    Returns:
        List of (cadence_ms, {col_name: (n_bins, 5) float32 ndarray})
        suitable for write_mktf(progressive_stats=...).
        The 5 stats are [sum, sum_sq, min, max, count].
    """
    if cadences is None:
        # Use default grid — cadence_id = index (0, 1, 2, ...)
        cadences = [(ms, i) for i, ms in enumerate(DEFAULT_CADENCES_MS)]

    col_names = list(engine._columns.keys())
    result = []

    for cadence_ms, cadence_id in cadences:
        n_bins = engine.n_bins(cadence_id)
        col_stats: dict[str, np.ndarray] = {}

        for name in col_names:
            fused = engine.bin_all_stats(name, cadence_id)

            # Extract the 5 progressive stats and transfer to CPU
            stats = np.zeros((n_bins, STAT_FIELDS), dtype=np.float32)
            stats[:, 0] = fused["sum"].get().astype(np.float32)
            stats[:, 1] = fused["sum_sq"].get().astype(np.float32)
            stats[:, 2] = fused["min"].get().astype(np.float32)
            stats[:, 3] = fused["max"].get().astype(np.float32)
            stats[:, 4] = fused["count"].get().astype(np.float32)

            col_stats[name] = stats

        result.append((cadence_ms, col_stats))

    return result
