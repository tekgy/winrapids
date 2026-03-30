"""Test KO05 sufficient statistics files — standalone MKTF roundtrip.

Proves:
  1. KO05 files with alignment=64 write/read bit-exact
  2. Column naming convention: {source_col}_{stat_suffix}
  3. alignment=64 produces smaller files than alignment=4096
  4. Sufficient stats compose correctly (monoid: sum/count additive, min/max comparable)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "R:/fintek")

from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_header, read_columns

# Mirror from progressive.py
STAT_FIELDS = 5
STAT_SUFFIXES = ("_sum", "_sum_sq", "_min", "_max", "_count")
KO05_ALIGNMENT = 64


def _make_ko05_columns(
    col_names: list[str],
    n_bins: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate deterministic KO05 stat columns for testing."""
    columns: dict[str, np.ndarray] = {}
    for name in col_names:
        count = rng.integers(100, 1000, n_bins).astype(np.float32)
        mean = rng.normal(100.0, 10.0, n_bins).astype(np.float32)
        std = rng.uniform(0.5, 5.0, n_bins).astype(np.float32)

        columns[f"{name}_sum"] = (mean * count).astype(np.float32)
        columns[f"{name}_sum_sq"] = ((std**2 + mean**2) * count).astype(np.float32)
        columns[f"{name}_min"] = (mean - 3 * std).astype(np.float32)
        columns[f"{name}_max"] = (mean + 3 * std).astype(np.float32)
        columns[f"{name}_count"] = count
    return columns


def test_ko05_roundtrip():
    """Write KO05 file with alignment=64, read back, verify bit-exact."""
    source_cols = ["price", "volume"]
    n_bins = 78  # 5-minute cadence
    rng = np.random.default_rng(42)

    columns = _make_ko05_columns(source_cols, n_bins, rng)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "K02P01C01.TI00TO05.KI00KO05.mktf"
        header = write_mktf(
            path, columns,
            leaf_id="K02P01C01.TI00TO05.KI00KO05",
            ticker="AAPL",
            day="2026-03-27",
            alignment=KO05_ALIGNMENT,
        )

        assert header.alignment == 64

        h2 = read_header(path)
        assert h2.is_complete is True
        assert h2.alignment == 64

        _, cols_read = read_columns(path)

        for col_name, arr in columns.items():
            assert col_name in cols_read, f"Missing column: {col_name}"
            assert np.array_equal(arr, cols_read[col_name]), (
                f"Data mismatch for {col_name}"
            )

        # 2 source cols × 5 stats = 10 columns
        assert len(cols_read) == 10

    print("PASS: KO05 roundtrip bit-exact (alignment=64)")


def test_ko05_alignment_size_reduction():
    """alignment=64 produces smaller files than alignment=4096."""
    source_cols = ["price", "volume", "log_price", "spread", "imbalance"]
    n_bins = 1  # session cadence — maximum alignment waste
    rng = np.random.default_rng(42)

    columns = _make_ko05_columns(source_cols, n_bins, rng)

    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        path_4096 = Path(tmpdir) / "align4096.mktf"
        write_mktf(path_4096, columns, leaf_id="test4096", alignment=4096)
        size_4096 = os.path.getsize(path_4096)

        path_64 = Path(tmpdir) / "align64.mktf"
        write_mktf(path_64, columns, leaf_id="test64", alignment=64)
        size_64 = os.path.getsize(path_64)

        ratio = size_4096 / size_64
        print(f"  alignment=4096: {size_4096:,} bytes")
        print(f"  alignment=64:   {size_64:,} bytes")
        print(f"  Reduction:      {ratio:.1f}x")

        assert size_64 < size_4096, "alignment=64 should produce smaller files"
        # For 25 columns of 4 bytes each:
        # 4096: 25 × 4096 ≈ 100 KB overhead
        # 64:   25 × 64 ≈ 1.6 KB overhead
        assert ratio > 5, f"Expected >5x reduction, got {ratio:.1f}x"

    print("PASS: alignment=64 reduces file size")


def test_ko05_composability():
    """Sufficient stats compose correctly via monoid operations."""
    a_sum, a_sum_sq, a_min, a_max, a_count = 150.0, 25000.0, 90.0, 115.0, 500.0
    b_sum, b_sum_sq, b_min, b_max, b_count = 160.0, 28000.0, 88.0, 120.0, 600.0

    c_sum = a_sum + b_sum
    c_sum_sq = a_sum_sq + b_sum_sq
    c_min = min(a_min, b_min)
    c_max = max(a_max, b_max)
    c_count = a_count + b_count

    mean = c_sum / c_count
    var = c_sum_sq / c_count - mean**2

    assert c_count == 1100.0
    assert c_min == 88.0
    assert c_max == 120.0
    assert var >= 0

    print("PASS: sufficient stats compose correctly")


if __name__ == "__main__":
    test_ko05_roundtrip()
    test_ko05_alignment_size_reduction()
    test_ko05_composability()
    print("\n=== ALL KO05 TESTS PASSED ===")
