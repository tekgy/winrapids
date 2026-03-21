"""
Column — the atomic unit of WinRapids.

A Column is a named GPU-resident buffer with CPU-accessible metadata.
Arithmetic on Columns builds lazy expression trees (no GPU work).
Call .evaluate() to fuse the tree into a single kernel and compute.

This is the co-native split: metadata (name, dtype, length, location)
lives on CPU where any agent can inspect it. Data lives on GPU where
compute happens.

Architecture:
  Column wraps a CuPy array + metadata.
  Arithmetic operators return expression trees (fusion.Expr nodes).
  Column IS-A Expr (it's a leaf node / ColRef in the expression tree).
  evaluate(expr) generates a fused kernel and returns a new Column.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import cupy as cp
import pyarrow as pa

from winrapids.fusion import ColRef, evaluate as _evaluate, fused_sum as _fused_sum


class Location(Enum):
    """Where a buffer physically lives."""
    DEVICE = "gpu"
    HOST_PINNED = "pinned"
    HOST = "cpu"


class Column(ColRef):
    """
    A named GPU-resident column with lazy fusion.

    Metadata (CPU-accessible):
      name, dtype, length, null_count, location

    Data (GPU-resident):
      _data: CuPy ndarray

    Arithmetic builds expression trees:
      c = a * b + 1.5   # no GPU work yet — just a tree
      result = c.compute()  # fuses into one kernel, returns Column

    Direct reductions:
      a.sum()   # CuPy reduction (fast for simple cases)
      (a * b + c).fused_sum()  # fused compute+reduce
    """

    __slots__ = ("dtype", "length", "null_count", "location", "_data")

    def __init__(self, name: str, data: cp.ndarray, null_count: int = 0,
                 location: Location = Location.DEVICE):
        super().__init__(name, data)
        self.dtype = data.dtype
        self.length = len(data)
        self.null_count = null_count
        self.location = location
        self._data = data

    # ---- Construction ----

    @classmethod
    def from_numpy(cls, name: str, arr: np.ndarray, pinned: bool = False) -> Column:
        if pinned:
            from winrapids.transfer import h2d
            return cls(name, h2d(arr))
        return cls(name, cp.asarray(arr))

    @classmethod
    def from_arrow(cls, name: str, arr: pa.Array, pinned: bool = False) -> Column:
        return cls.from_numpy(name, arr.to_numpy(zero_copy_only=False), pinned=pinned)

    @classmethod
    def from_list(cls, name: str, data, dtype=None, pinned: bool = False) -> Column:
        return cls.from_numpy(name, np.array(data, dtype=dtype), pinned=pinned)

    # ---- Materialization ----

    def to_numpy(self, pinned: bool = False) -> np.ndarray:
        if pinned:
            from winrapids.transfer import d2h
            return d2h(self._data)
        return cp.asnumpy(self._data)

    def to_arrow(self) -> pa.Array:
        return pa.array(self.to_numpy())

    # ---- Expression evaluation ----

    def compute(self) -> Column:
        """Evaluate this expression tree and return a new Column.

        If this Column is already a leaf (not an expression), returns self.
        """
        # ColRef.code references self.data directly — evaluate will read it
        result = _evaluate(self)
        return Column(self.name, result)

    def fused_sum(self) -> float:
        """Evaluate expression AND reduce to scalar sum in one kernel."""
        return _fused_sum(self)

    # ---- Direct reductions (for leaf columns, uses CuPy) ----

    def sum(self) -> float:
        return float(cp.sum(self._data))

    def mean(self) -> float:
        return float(cp.mean(self._data))

    def min(self) -> float:
        return float(cp.min(self._data))

    def max(self) -> float:
        return float(cp.max(self._data))

    def std(self) -> float:
        return float(cp.std(self._data))

    # ---- Properties ----

    @property
    def device_ptr(self) -> int:
        return self._data.data.ptr

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return (f"Column('{self.name}', {self.dtype}, "
                f"{self.length:,} rows, {self.location.value})")
