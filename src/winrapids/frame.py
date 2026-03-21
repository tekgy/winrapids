"""
Frame — a collection of Columns. The GPU DataFrame.

Frame holds named Columns with shared row count. Provides:
  - Column access by name: frame["col_name"]
  - Memory map: CPU-readable summary of what's on GPU
  - Arrow/pandas conversion
  - GroupBy (sort-based and hash-based)
  - Join (direct-index and sort-merge)

Frame is fractal: it's a named collection of Columns, and each Column
is a named buffer with metadata. Both carry their own metadata+data split.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import cupy as cp
import pyarrow as pa

from winrapids.column import Column


class Frame:
    """GPU DataFrame — a named collection of Columns."""

    __slots__ = ("_columns", "_nrows")

    def __init__(self, columns: dict[str, Column]):
        self._columns = columns
        lengths = {col.length for col in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"Column lengths don't match: {lengths}")
        self._nrows = lengths.pop() if lengths else 0

    # ---- Construction ----

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray | list], pinned: bool = False) -> Frame:
        if pinned:
            # Batch transfer: single sync for all columns
            from winrapids.transfer import h2d_batch
            numpy_arrays = {}
            gpu_columns = {}
            for name, arr in data.items():
                if isinstance(arr, cp.ndarray):
                    gpu_columns[name] = Column(name, arr)
                elif isinstance(arr, np.ndarray):
                    numpy_arrays[name] = arr
                else:
                    numpy_arrays[name] = np.asarray(arr)
            gpu_data = h2d_batch(numpy_arrays)
            columns = {**gpu_columns}
            for name, gpu_arr in gpu_data.items():
                columns[name] = Column(name, gpu_arr)
            return cls(columns)

        columns = {}
        for name, arr in data.items():
            if isinstance(arr, np.ndarray):
                columns[name] = Column.from_numpy(name, arr)
            elif isinstance(arr, cp.ndarray):
                columns[name] = Column(name, arr)
            else:
                columns[name] = Column.from_list(name, arr)
        return cls(columns)

    @classmethod
    def from_pandas(cls, df, pinned: bool = False) -> Frame:
        columns = {}
        for name in df.columns:
            columns[name] = Column.from_numpy(name, df[name].values, pinned=pinned)
        return cls(columns)

    @classmethod
    def from_arrow(cls, table: pa.Table, pinned: bool = False) -> Frame:
        columns = {}
        for name in table.column_names:
            columns[name] = Column.from_arrow(name, table.column(name), pinned=pinned)
        return cls(columns)

    @classmethod
    def from_parquet(cls, path: str, columns: list[str] | None = None,
                     pinned: bool = True) -> Frame:
        """Read a Parquet file into a GPU Frame.

        Uses PyArrow for I/O and pinned memory for fast H2D transfer.
        This is the cudf.read_parquet() replacement.
        """
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=columns)
        return cls.from_arrow(table, pinned=pinned)

    # ---- Access ----

    def __getitem__(self, key: str) -> Column:
        return self._columns[key]

    def __setitem__(self, key: str, value):
        """Add or replace a column. If value is an expression, evaluates it first."""
        from winrapids.fusion import Expr, evaluate as _eval_expr

        if isinstance(value, Column):
            if value.length != self._nrows:
                raise ValueError(f"Column length {value.length} != frame rows {self._nrows}")
            value.name = key
            self._columns[key] = value
        elif isinstance(value, Expr):
            # Lazy expression — evaluate it
            result_data = _eval_expr(value)
            self._columns[key] = Column(key, result_data)
        elif isinstance(value, cp.ndarray):
            self._columns[key] = Column(key, value)
        else:
            raise TypeError(f"Cannot assign {type(value)} to Frame column")

    def __len__(self) -> int:
        return self._nrows

    def __contains__(self, key: str) -> bool:
        return key in self._columns

    @property
    def columns(self) -> list[str]:
        return list(self._columns.keys())

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nrows, len(self._columns))

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return {name: col.dtype for name, col in self._columns.items()}

    # ---- Co-native metadata ----

    def memory_map(self) -> str:
        """Show per-column memory residency. Readable by human or AI agent."""
        lines = [f"Frame: {self._nrows:,} rows x {len(self._columns)} columns"]
        total_bytes = 0
        for name, col in self._columns.items():
            lines.append(
                f"  {name:20s}  {str(col.dtype):10s}  "
                f"{col.nbytes / 1e6:8.1f} MB  [{col.location.value}]"
            )
            total_bytes += col.nbytes
        lines.append(f"  Total: {total_bytes / 1e6:.1f} MB")
        return "\n".join(lines)

    # ---- Export ----

    def to_pandas(self, pinned: bool = False):
        import pandas as pd
        return pd.DataFrame({
            name: col.to_numpy(pinned=pinned)
            for name, col in self._columns.items()
        })

    def to_arrow(self, pinned: bool = False) -> pa.Table:
        arrays = {}
        for name, col in self._columns.items():
            arrays[name] = pa.array(col.to_numpy(pinned=pinned))
        return pa.table(arrays)

    def to_parquet(self, path: str, pinned: bool = True):
        """Write GPU Frame to a Parquet file.

        Uses pinned memory for fast D2H transfer + PyArrow for I/O.
        This is the cudf.DataFrame.to_parquet() replacement.
        """
        import pyarrow.parquet as pq
        table = self.to_arrow(pinned=pinned)
        pq.write_table(table, path)

    # ---- Filter ----

    def filter(self, mask) -> Frame:
        """Filter rows by a boolean mask or expression.

        mask can be:
          - A CuPy boolean array
          - A Column (used as boolean)
          - An Expr (evaluated, then > 0 treated as True)
        """
        from winrapids.fusion import Expr, evaluate as _eval_expr

        if isinstance(mask, Expr):
            mask_data = _eval_expr(mask)
            bool_mask = mask_data > 0
        elif isinstance(mask, Column):
            bool_mask = mask._data > 0
        elif isinstance(mask, cp.ndarray):
            bool_mask = mask
        else:
            raise TypeError(f"Cannot filter with {type(mask)}")

        result_cols = {}
        for name, col in self._columns.items():
            result_cols[name] = Column(name, col._data[bool_mask])
        return Frame(result_cols)

    def select(self, columns: list[str]) -> Frame:
        """Select a subset of columns by name."""
        return Frame({name: self._columns[name] for name in columns})

    def head(self, n: int = 5) -> Frame:
        """Return the first n rows."""
        result_cols = {}
        for name, col in self._columns.items():
            result_cols[name] = Column(name, col._data[:n])
        return Frame(result_cols)

    def sort(self, by: str, ascending: bool = True) -> Frame:
        """Sort all columns by values in the given column."""
        sort_idx = cp.argsort(self._columns[by]._data)
        if not ascending:
            sort_idx = sort_idx[::-1]
        result_cols = {}
        for name, col in self._columns.items():
            result_cols[name] = Column(name, col._data[sort_idx])
        return Frame(result_cols)

    def concat(self, other: Frame) -> Frame:
        """Concatenate two frames vertically (same columns required)."""
        if set(self.columns) != set(other.columns):
            raise ValueError(f"Column mismatch: {self.columns} vs {other.columns}")
        result_cols = {}
        for name in self.columns:
            result_cols[name] = Column(name, cp.concatenate([
                self._columns[name]._data,
                other._columns[name]._data,
            ]))
        return Frame(result_cols)

    def describe(self) -> dict[str, dict[str, float]]:
        """Summary statistics for all numeric columns."""
        stats = {}
        for name, col in self._columns.items():
            if col.dtype.kind in ('f', 'i', 'u'):
                data = col._data.astype(cp.float64)
                stats[name] = {
                    "count": len(data),
                    "mean": float(cp.mean(data)),
                    "std": float(cp.std(data)),
                    "min": float(cp.min(data)),
                    "max": float(cp.max(data)),
                }
        return stats

    # ---- GroupBy ----

    def groupby(self, key: str) -> GroupBy:
        """Group by a column. Returns a GroupBy object for aggregation."""
        return GroupBy(self, key)

    # ---- Join ----

    def join(self, other: Frame, on: str, how: str = "inner") -> Frame:
        """Join this frame with another on a key column.

        For dense integer keys [0, max_key), uses direct-index (O(1) lookup).
        Falls back to sort-merge for arbitrary keys.
        """
        if how != "inner":
            raise NotImplementedError(f"Only inner join supported, got '{how}'")

        left_keys = self[on]._data
        right_keys = other[on]._data

        # Try direct-index for dense integer keys
        if left_keys.dtype in (cp.int32, cp.int64) and right_keys.dtype in (cp.int32, cp.int64):
            max_key = max(int(cp.max(left_keys)), int(cp.max(right_keys)))
            if max_key < len(right_keys) * 4:  # reasonably dense
                left_idx, right_idx = _direct_join(left_keys, right_keys, max_key + 1)
            else:
                left_idx, right_idx = _sort_merge_join(left_keys, right_keys)
        else:
            left_idx, right_idx = _sort_merge_join(left_keys, right_keys)

        # Build result frame by gathering from both sides
        result_cols = {}
        for name, col in self._columns.items():
            result_cols[name] = Column(name, col._data[left_idx])
        for name, col in other._columns.items():
            if name != on:  # don't duplicate the join key
                result_cols[name] = Column(name, col._data[right_idx])

        return Frame(result_cols)

    def __repr__(self) -> str:
        col_info = ", ".join(f"{n}:{c.dtype}" for n, c in self._columns.items())
        return f"Frame({self._nrows:,} rows, [{col_info}])"


# ============================================================
# GroupBy
# ============================================================

class GroupBy:
    """Lazy groupby — computes on .sum(), .mean(), etc."""

    __slots__ = ("_frame", "_key")

    def __init__(self, frame: Frame, key: str):
        self._frame = frame
        self._key = key

    def _sort_reduce(self, value_col: str):
        """Sort by key, find boundaries, return (unique_keys, boundary_idx, sorted_vals)."""
        keys = self._frame[self._key]._data
        values = self._frame[value_col]._data.astype(cp.float64)

        sort_idx = cp.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_vals = values[sort_idx]

        boundaries = cp.concatenate([cp.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
        boundary_idx = cp.where(boundaries)[0]
        unique_keys = sorted_keys[boundary_idx]

        return unique_keys, boundary_idx, sorted_vals

    def sum(self, col: str) -> Frame:
        """GroupBy sum. Returns Frame with key + sum columns."""
        unique_keys, boundary_idx, sorted_vals = self._sort_reduce(col)
        n = len(sorted_vals)

        cumsum = cp.cumsum(sorted_vals)
        end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([n - 1])])
        group_sums = cumsum[end_idx].copy()
        group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

        return Frame({
            self._key: Column(self._key, unique_keys),
            col: Column(col, group_sums),
        })

    def mean(self, col: str) -> Frame:
        """GroupBy mean."""
        unique_keys, boundary_idx, sorted_vals = self._sort_reduce(col)
        n = len(sorted_vals)

        cumsum = cp.cumsum(sorted_vals)
        end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([n - 1])])
        group_sums = cumsum[end_idx].copy()
        group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

        group_counts = cp.diff(cp.concatenate([boundary_idx, cp.array([n])]))
        group_means = group_sums / group_counts.astype(cp.float64)

        return Frame({
            self._key: Column(self._key, unique_keys),
            col: Column(col, group_means),
        })

    def count(self) -> Frame:
        """GroupBy count."""
        keys = self._frame[self._key]._data
        sort_idx = cp.argsort(keys)
        sorted_keys = keys[sort_idx]

        boundaries = cp.concatenate([cp.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
        boundary_idx = cp.where(boundaries)[0]
        unique_keys = sorted_keys[boundary_idx]
        n = len(keys)

        group_counts = cp.diff(cp.concatenate([boundary_idx, cp.array([n])]))

        return Frame({
            self._key: Column(self._key, unique_keys),
            "count": Column("count", group_counts),
        })

    def agg(self, col: str, funcs: Sequence[str] = ("sum", "mean", "count")) -> Frame:
        """Multiple aggregations in one pass (single sort)."""
        unique_keys, boundary_idx, sorted_vals = self._sort_reduce(col)
        n = len(sorted_vals)

        cumsum = cp.cumsum(sorted_vals)
        end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([n - 1])])
        group_sums = cumsum[end_idx].copy()
        group_sums[1:] -= cumsum[boundary_idx[1:] - 1]
        group_counts = cp.diff(cp.concatenate([boundary_idx, cp.array([n])]))

        result_cols = {self._key: Column(self._key, unique_keys)}

        for func in funcs:
            if func == "sum":
                result_cols[f"{col}_sum"] = Column(f"{col}_sum", group_sums.copy())
            elif func == "mean":
                result_cols[f"{col}_mean"] = Column(
                    f"{col}_mean", group_sums / group_counts.astype(cp.float64))
            elif func == "count":
                result_cols["count"] = Column("count", group_counts)

        return Frame(result_cols)


# ============================================================
# Join implementations
# ============================================================

def _direct_join(left_keys: cp.ndarray, right_keys: cp.ndarray,
                 table_size: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Direct-index join for dense integer keys [0, table_size)."""
    lookup = cp.full(table_size, -1, dtype=cp.int32)
    lookup[right_keys.astype(cp.int64)] = cp.arange(len(right_keys), dtype=cp.int32)

    left_bounded = cp.clip(left_keys, 0, table_size - 1)
    right_indices = lookup[left_bounded.astype(cp.int64)]

    valid = (left_keys >= 0) & (left_keys < table_size) & (right_indices >= 0)
    left_idx = cp.where(valid)[0]
    right_idx = right_indices[left_idx]

    return left_idx, right_idx


def _sort_merge_join(left_keys: cp.ndarray, right_keys: cp.ndarray
                     ) -> tuple[cp.ndarray, cp.ndarray]:
    """Sort-merge join using searchsorted."""
    right_sort = cp.argsort(right_keys)
    sorted_right = right_keys[right_sort]

    positions = cp.searchsorted(sorted_right, left_keys)
    valid = positions < len(right_keys)
    valid_pos = cp.where(valid, positions, 0)
    matches = valid & (sorted_right[valid_pos] == left_keys)

    left_idx = cp.where(matches)[0]
    right_idx = right_sort[valid_pos[left_idx]]

    return left_idx, right_idx
