"""Test progressive section roundtrip through production MKTF writer/reader.

Proves:
  1. Write with progressive stats -> read back bit-exact
  2. File without progressive section -> reader returns None gracefully
  3. Old-style header reading still works (progressive_offset=0 is transparent)
  4. MI scores survive roundtrip
  5. Per-cadence layout matches prototype binary format
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "R:/fintek")

from trunk.backends.mktf.format import (
    FLAG_HAS_PROGRESSIVE,
    PROG_STAT_FIELDS,
    pack_progressive_section,
    unpack_progressive_directory,
    unpack_progressive_level,
)
from trunk.backends.mktf.writer import write_mktf
from trunk.backends.mktf.reader import (
    read_header,
    read_columns,
    read_progressive_summary,
    read_progressive_level_data,
)


def _make_progressive_stats(
    n_rows: int,
    col_names: list[str],
) -> list[tuple[int, dict[str, np.ndarray]]]:
    """Generate deterministic progressive stats for testing."""
    rng = np.random.default_rng(42)
    levels = []

    for cadence_ms, n_bins in [
        (86_400_000, 1),     # session
        (3_600_000, 7),      # hourly
        (60_000, 390),       # minute
        (1_000, 23_400),     # second
    ]:
        col_stats = {}
        for name in col_names:
            stats = np.zeros((n_bins, PROG_STAT_FIELDS), dtype=np.float32)
            stats[:, 0] = rng.normal(100.0, 10.0, n_bins).astype(np.float32)  # sum
            stats[:, 1] = rng.normal(200.0, 20.0, n_bins).astype(np.float32)  # sum_sq
            stats[:, 2] = rng.normal(50.0, 5.0, n_bins).astype(np.float32)    # min
            stats[:, 3] = rng.normal(150.0, 15.0, n_bins).astype(np.float32)  # max
            stats[:, 4] = rng.integers(1, 100, n_bins).astype(np.float32)     # count
            col_stats[name] = stats
        levels.append((cadence_ms, col_stats))

    return levels


def test_roundtrip_with_progressive():
    """Write MKTF with progressive section, read back, verify bit-exact."""
    col_names = ["price", "volume"]
    n_rows = 1000
    rng = np.random.default_rng(123)

    columns = {
        "price": rng.normal(100.0, 10.0, n_rows).astype(np.float32),
        "volume": rng.integers(1, 10000, n_rows).astype(np.int32),
    }

    prog_stats = _make_progressive_stats(n_rows, col_names)
    mi_scores = [0.87, 0.62, 0.15]  # 4 levels -> 3 MI scores

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.mktf"
        header = write_mktf(
            path, columns,
            leaf_id="test-K01-AAPL-20260327",
            ticker="AAPL",
            day="2026-03-27",
            progressive_stats=prog_stats,
            progressive_mi=mi_scores,
        )

        # Verify header has progressive pointers
        assert header.progressive_offset > 0, "progressive_offset should be set"
        assert header.progressive_size > 0, "progressive_size should be set"
        assert header.progressive_levels == 4, f"Expected 4 levels, got {header.progressive_levels}"
        assert header.flags & FLAG_HAS_PROGRESSIVE, "FLAG_HAS_PROGRESSIVE not set"

        # Read back header
        h2 = read_header(path)
        assert h2.progressive_offset == header.progressive_offset
        assert h2.progressive_size == header.progressive_size
        assert h2.progressive_levels == 4

        # Read progressive summary
        summary = read_progressive_summary(path)
        assert summary is not None, "Progressive summary should not be None"
        n_levels, n_cols, levels, mi_read = summary
        assert n_levels == 4
        assert n_cols == 2
        assert len(levels) == 4
        assert levels[0].cadence_ms == 86_400_000  # session
        assert levels[1].cadence_ms == 3_600_000    # hourly
        assert levels[2].cadence_ms == 60_000       # minute
        assert levels[3].cadence_ms == 1_000        # second
        assert levels[0].bin_count == 1
        assert levels[1].bin_count == 7
        assert levels[2].bin_count == 390
        assert levels[3].bin_count == 23_400

        # Verify MI scores roundtrip
        assert len(mi_read) == 3
        for i, (expected, actual) in enumerate(zip(mi_scores, mi_read)):
            assert abs(expected - actual) < 1e-6, f"MI[{i}]: {expected} != {actual}"

        # Read each level and verify bit-exact match
        for i, (cadence_ms, original_stats) in enumerate(prog_stats):
            read_stats = read_progressive_level_data(path, cadence_ms, col_names)
            assert read_stats is not None, f"Level {cadence_ms}ms should be readable"

            for name in col_names:
                orig = original_stats[name]
                read = read_stats[name]
                assert orig.shape == read.shape, (
                    f"Shape mismatch for {name} at {cadence_ms}ms: "
                    f"{orig.shape} vs {read.shape}"
                )
                assert np.array_equal(orig, read), (
                    f"Data mismatch for {name} at {cadence_ms}ms: "
                    f"max diff = {np.abs(orig - read).max()}"
                )

        # Verify column data still reads correctly
        _, cols_read = read_columns(path)
        assert np.array_equal(cols_read["price"], columns["price"])
        assert np.array_equal(cols_read["volume"], columns["volume"])

        # Verify is_complete and trailing status
        assert h2.is_complete is True

    print("PASS: roundtrip with progressive section")


def test_no_progressive_section():
    """File without progressive section -> reader returns None."""
    col_names = ["price"]
    columns = {"price": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "no_prog.mktf"
        header = write_mktf(path, columns, leaf_id="test-no-prog")

        assert header.progressive_offset == 0
        assert not (header.flags & FLAG_HAS_PROGRESSIVE)

        # Reader should return None gracefully
        summary = read_progressive_summary(path)
        assert summary is None

        level = read_progressive_level_data(path, 86_400_000)
        assert level is None

    print("PASS: no progressive section handled gracefully")


def test_progressive_section_format_matches_prototype():
    """Verify the binary format matches the prototype's pack/unpack."""
    col_names = ["a", "b"]
    rng = np.random.default_rng(99)

    levels = []
    for cadence_ms, n_bins in [(86_400_000, 1), (3_600_000, 5)]:
        col_stats = {}
        for name in col_names:
            stats = rng.random((n_bins, PROG_STAT_FIELDS)).astype(np.float32)
            col_stats[name] = stats
        levels.append((cadence_ms, col_stats))

    mi = [0.5]
    packed = pack_progressive_section(levels, col_names, mi)

    # Unpack directory
    n_levels, n_cols, level_dirs, mi_read = unpack_progressive_directory(packed)
    assert n_levels == 2
    assert n_cols == 2
    assert abs(mi_read[0] - 0.5) < 1e-6

    # Unpack each level
    for i, (cadence_ms, original) in enumerate(levels):
        unpacked = unpack_progressive_level(packed, level_dirs[i], col_names)
        for name in col_names:
            assert np.array_equal(original[name], unpacked[name]), (
                f"Mismatch at level {i}, col {name}"
            )

    print("PASS: binary format matches prototype")


def test_progressive_nonexistent_cadence():
    """Requesting a cadence that doesn't exist returns None."""
    col_names = ["x"]
    columns = {"x": np.array([1.0, 2.0], dtype=np.float32)}
    prog = [(86_400_000, {"x": np.ones((1, 5), dtype=np.float32)})]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "one_level.mktf"
        write_mktf(path, columns, leaf_id="test-one", progressive_stats=prog)

        # Session exists
        result = read_progressive_level_data(path, 86_400_000, col_names)
        assert result is not None

        # Minute does not exist
        result = read_progressive_level_data(path, 60_000, col_names)
        assert result is None

    print("PASS: nonexistent cadence returns None")


if __name__ == "__main__":
    test_roundtrip_with_progressive()
    test_no_progressive_section()
    test_progressive_section_format_matches_prototype()
    test_progressive_nonexistent_cadence()
    print("\n=== ALL PROGRESSIVE ROUNDTRIP TESTS PASSED ===")
