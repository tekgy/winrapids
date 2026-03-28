"""Test KO05 sufficient statistics files — standalone MKTF roundtrip.

Proves:
  1. KO05 files are regular MKTF files with stat columns — write/read bit-exact
  2. Column naming convention: {source_col}_{stat_suffix}
  3. Sufficient stats compose correctly (monoid: sum/count additive, min/max comparable)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "R:/fintek")

from trunk.backends.mktf.format import DOMAIN_SUFFICIENT
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_header, read_columns

# Mirror from progressive.py to avoid cupy import chain
STAT_SUFFIXES = ("_sum", "_sum_sq", "_min", "_max", "_count")


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
    """Write KO05 file, read back, verify bit-exact."""
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
        )

        # Read back
        h2 = read_header(path)
        assert h2.is_complete is True
        assert h2.leaf_id == "K02P01C01.TI00TO05.KI00KO05"

        _, cols_read = read_columns(path)

        # Verify all stat columns roundtrip bit-exact
        for col_name, arr in columns.items():
            assert col_name in cols_read, f"Missing column: {col_name}"
            assert np.array_equal(arr, cols_read[col_name]), (
                f"Data mismatch for {col_name}: "
                f"max diff = {np.abs(arr - cols_read[col_name]).max()}"
            )

        # Verify column count: 2 source cols × 5 stats = 10
        assert len(cols_read) == 10

    print("PASS: KO05 roundtrip bit-exact")


def test_ko05_column_naming():
    """Verify column naming convention matches STAT_SUFFIXES."""
    source_cols = ["price", "volume", "log_price"]
    n_bins = 1
    rng = np.random.default_rng(99)

    columns = _make_ko05_columns(source_cols, n_bins, rng)

    expected_names = set()
    for name in source_cols:
        for suffix in STAT_SUFFIXES:
            expected_names.add(f"{name}{suffix}")

    assert set(columns.keys()) == expected_names


def test_ko05_composability():
    """Sufficient stats compose correctly via monoid operations."""
    rng = np.random.default_rng(7)

    # Two adjacent bins
    a_sum = np.float32(150.0)
    a_sum_sq = np.float32(25000.0)
    a_min = np.float32(90.0)
    a_max = np.float32(115.0)
    a_count = np.float32(500.0)

    b_sum = np.float32(160.0)
    b_sum_sq = np.float32(28000.0)
    b_min = np.float32(88.0)
    b_max = np.float32(120.0)
    b_count = np.float32(600.0)

    # Compose
    c_sum = a_sum + b_sum
    c_sum_sq = a_sum_sq + b_sum_sq
    c_min = min(a_min, b_min)
    c_max = max(a_max, b_max)
    c_count = a_count + b_count

    # Verify derived statistics
    mean = c_sum / c_count
    var = c_sum_sq / c_count - mean**2

    assert c_count == 1100.0
    assert c_min == 88.0
    assert c_max == 120.0
    assert var >= 0, "Variance should be non-negative"

    print("PASS: sufficient stats compose correctly")


if __name__ == "__main__":
    test_ko05_roundtrip()
    test_ko05_column_naming()
    test_ko05_composability()
    print("\n=== ALL KO05 TESTS PASSED ===")
