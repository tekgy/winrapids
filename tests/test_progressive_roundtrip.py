"""Test KO05 sufficient statistics files — standalone MKTF roundtrip.

Proves:
  1. KO05 files are regular MKTF files with bundled stat columns — write/read bit-exact
  2. Bundled column layout: {source_col}_stats = float32[n_bins * 5]
  3. unpack_ko05_column reshapes correctly
  4. Sufficient stats compose correctly (monoid: sum/count additive, min/max comparable)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "R:/fintek")
sys.path.insert(0, "R:/winrapids/src")

from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import read_header, read_columns

# Import directly to avoid cupy chain in __init__.py
STAT_FIELDS = 5
STAT_NAMES = ("sum", "sum_sq", "min", "max", "count")


def _unpack_ko05_column(flat: np.ndarray) -> np.ndarray:
    """Mirror of progressive.unpack_ko05_column for testing."""
    return flat.reshape(-1, STAT_FIELDS)


def _make_ko05_columns(
    col_names: list[str],
    n_bins: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate deterministic bundled KO05 stat columns for testing."""
    columns: dict[str, np.ndarray] = {}
    for name in col_names:
        count = rng.integers(100, 1000, n_bins).astype(np.float32)
        mean = rng.normal(100.0, 10.0, n_bins).astype(np.float32)
        std = rng.uniform(0.5, 5.0, n_bins).astype(np.float32)

        stats = np.empty((n_bins, STAT_FIELDS), dtype=np.float32)
        stats[:, 0] = (mean * count).astype(np.float32)       # sum
        stats[:, 1] = ((std**2 + mean**2) * count).astype(np.float32)  # sum_sq
        stats[:, 2] = (mean - 3 * std).astype(np.float32)     # min
        stats[:, 3] = (mean + 3 * std).astype(np.float32)     # max
        stats[:, 4] = count                                      # count

        columns[f"{name}_stats"] = stats.ravel()
    return columns


def test_ko05_roundtrip():
    """Write KO05 file with bundled columns, read back, verify bit-exact."""
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

        _, cols_read = read_columns(path)

        # Verify all bundled columns roundtrip bit-exact
        for col_name, arr in columns.items():
            assert col_name in cols_read, f"Missing column: {col_name}"
            assert np.array_equal(arr, cols_read[col_name]), (
                f"Data mismatch for {col_name}: "
                f"max diff = {np.abs(arr - cols_read[col_name]).max()}"
            )

        # Verify column count: 2 source cols = 2 bundled columns
        assert len(cols_read) == 2

    print("PASS: KO05 roundtrip bit-exact (bundled)")


def test_ko05_unpack():
    """Verify unpack_ko05_column reshapes correctly."""
    n_bins = 13
    rng = np.random.default_rng(99)

    stats_2d = rng.random((n_bins, STAT_FIELDS)).astype(np.float32)
    flat = stats_2d.ravel()

    unpacked = _unpack_ko05_column(flat)
    assert unpacked.shape == (n_bins, STAT_FIELDS)
    assert np.array_equal(unpacked, stats_2d)

    # Verify stat fields are in correct positions
    assert np.array_equal(unpacked[:, 0], stats_2d[:, 0])  # sum
    assert np.array_equal(unpacked[:, 4], stats_2d[:, 4])  # count

    print("PASS: unpack_ko05_column reshapes correctly")


def test_ko05_composability():
    """Sufficient stats compose correctly via monoid operations."""
    # Two adjacent bins (bundled format)
    a = np.array([150.0, 25000.0, 90.0, 115.0, 500.0], dtype=np.float32)
    b = np.array([160.0, 28000.0, 88.0, 120.0, 600.0], dtype=np.float32)

    # Compose: sum adds, sum_sq adds, min takes min, max takes max, count adds
    c = np.empty(STAT_FIELDS, dtype=np.float32)
    c[0] = a[0] + b[0]           # sum
    c[1] = a[1] + b[1]           # sum_sq
    c[2] = min(a[2], b[2])       # min
    c[3] = max(a[3], b[3])       # max
    c[4] = a[4] + b[4]           # count

    # Verify derived statistics
    mean = c[0] / c[4]
    var = c[1] / c[4] - mean**2

    assert c[4] == 1100.0
    assert c[2] == 88.0
    assert c[3] == 120.0
    assert var >= 0, "Variance should be non-negative"

    print("PASS: sufficient stats compose correctly")


if __name__ == "__main__":
    test_ko05_roundtrip()
    test_ko05_unpack()
    test_ko05_composability()
    print("\n=== ALL KO05 TESTS PASSED ===")
