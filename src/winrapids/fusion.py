"""
Kernel fusion engine — generates fused CUDA kernels from expression trees.

Expression trees are built lazily when Column objects are combined with
arithmetic operators. When .evaluate() or fused_sum() is called, the tree
is walked to generate CUDA kernel source, which is compiled via CuPy's
RawKernel JIT and cached for reuse.

This is the core compute differentiator. CuPy launches one kernel per
operation — a 6-operation expression launches 6 kernels with 5 intermediate
buffers. Fusion collapses that to 1 kernel with 0 intermediates. All fused
kernels run at ~0.19 ms (bandwidth-bound at ~1650 GB/s) regardless of
expression depth.

Validated against C++ expression templates in experiment 010 — template-fused
and codegen-fused produce identical performance (within 0.01 ms).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from winrapids.column import Column


# ============================================================
# Expression tree nodes
# ============================================================

class Expr:
    """Base class for expression tree nodes. Supports arithmetic operator chaining."""

    def __add__(self, other): return BinOp("+", self, _wrap(other))
    def __radd__(self, other): return BinOp("+", _wrap(other), self)
    def __sub__(self, other): return BinOp("-", self, _wrap(other))
    def __rsub__(self, other): return BinOp("-", _wrap(other), self)
    def __mul__(self, other): return BinOp("*", self, _wrap(other))
    def __rmul__(self, other): return BinOp("*", _wrap(other), self)
    def __truediv__(self, other): return BinOp("/", self, _wrap(other))
    def __rtruediv__(self, other): return BinOp("/", _wrap(other), self)
    def __pow__(self, other): return PowOp(self, _wrap(other))
    def __rpow__(self, other): return PowOp(_wrap(other), self)
    def __neg__(self): return UnaryOp("neg", self)
    def __abs__(self): return UnaryOp("fabs", self)
    def __gt__(self, other): return Compare(">", self, _wrap(other))
    def __lt__(self, other): return Compare("<", self, _wrap(other))
    def __ge__(self, other): return Compare(">=", self, _wrap(other))
    def __le__(self, other): return Compare("<=", self, _wrap(other))
    def __eq__(self, other): return Compare("==", self, _wrap(other))
    def __ne__(self, other): return Compare("!=", self, _wrap(other))
    def __and__(self, other): return BinOp("*", self, _wrap(other))
    def __or__(self, other): return BoolOr(self, _wrap(other))
    def __invert__(self): return Compare("==", self, Const(0.0))

    def sqrt(self): return UnaryOp("sqrt", self)
    def abs(self): return UnaryOp("fabs", self)
    def log(self): return UnaryOp("log", self)
    def exp(self): return UnaryOp("exp", self)

    def code(self, params: dict) -> str:
        raise NotImplementedError

    def signature_key(self, params: dict) -> str:
        raise NotImplementedError


def _wrap(x):
    """Wrap a scalar into a Const node, pass through Expr unchanged."""
    if isinstance(x, Expr):
        return x
    return Const(float(x))


_DTYPE_TO_CTYPE = {
    cp.float64: "double",
    cp.float32: "float",
    cp.int64: "long long",
    cp.int32: "int",
    cp.int16: "short",
    cp.int8: "char",
    cp.uint64: "unsigned long long",
    cp.uint32: "unsigned int",
    cp.uint16: "unsigned short",
    cp.uint8: "unsigned char",
    cp.bool_: "bool",
}


def _ctype_for(dtype) -> str:
    """Map a numpy/cupy dtype to a CUDA C type string."""
    for cp_dtype, ctype in _DTYPE_TO_CTYPE.items():
        if dtype == cp_dtype:
            return ctype
    return "double"


class ColRef(Expr):
    """Reference to a GPU column buffer (leaf node in the expression tree)."""

    __slots__ = ("name", "data")

    def __init__(self, name: str, data: cp.ndarray):
        self.name = name
        self.data = data

    def code(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            params[self.name] = (f"c{idx}", self.data)
        return f"(double){params[self.name][0]}[idx]"

    def signature_key(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            ctype = _ctype_for(self.data.dtype)
            params[self.name] = f"{ctype}:c{idx}"
        return params[self.name]


class Const(Expr):
    """A scalar constant."""

    __slots__ = ("value",)

    def __init__(self, value: float):
        self.value = value

    def code(self, params: dict) -> str:
        return repr(self.value)

    def signature_key(self, params: dict) -> str:
        return f"K{self.value}"


class BinOp(Expr):
    """Binary arithmetic operation."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def code(self, params: dict) -> str:
        return f"({self.left.code(params)} {self.op} {self.right.code(params)})"

    def signature_key(self, params: dict) -> str:
        return f"({self.left.signature_key(params)}{self.op}{self.right.signature_key(params)})"


class PowOp(Expr):
    """Power: pow(base, exponent)."""

    __slots__ = ("base", "exponent")

    def __init__(self, base: Expr, exponent: Expr):
        self.base = base
        self.exponent = exponent

    def code(self, params: dict) -> str:
        return f"pow({self.base.code(params)}, {self.exponent.code(params)})"

    def signature_key(self, params: dict) -> str:
        return f"pow({self.base.signature_key(params)},{self.exponent.signature_key(params)})"


class Compare(Expr):
    """Comparison — returns 1.0 (true) or 0.0 (false)."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def code(self, params: dict) -> str:
        return f"(({self.left.code(params)} {self.op} {self.right.code(params)}) ? 1.0 : 0.0)"

    def signature_key(self, params: dict) -> str:
        return f"({self.left.signature_key(params)}{self.op}{self.right.signature_key(params)})"


class UnaryOp(Expr):
    """Unary operation (neg, sqrt, abs, log, exp)."""

    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Expr):
        self.op = op
        self.operand = operand

    def code(self, params: dict) -> str:
        inner = self.operand.code(params)
        if self.op == "neg":
            return f"(-{inner})"
        return f"{self.op}({inner})"

    def signature_key(self, params: dict) -> str:
        inner = self.operand.signature_key(params)
        if self.op == "neg":
            return f"(-{inner})"
        return f"{self.op}({inner})"


class BoolOr(Expr):
    """Boolean OR — (a != 0 || b != 0) ? 1.0 : 0.0."""

    __slots__ = ("left", "right")

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def code(self, params: dict) -> str:
        l = self.left.code(params)
        r = self.right.code(params)
        return f"(({l} != 0.0 || {r} != 0.0) ? 1.0 : 0.0)"

    def signature_key(self, params: dict) -> str:
        l = self.left.signature_key(params)
        r = self.right.signature_key(params)
        return f"or({l},{r})"


class WhereExpr(Expr):
    """Ternary: where(cond, then_val, else_val)."""

    __slots__ = ("cond", "then_val", "else_val")

    def __init__(self, cond: Expr, then_val: Expr, else_val: Expr):
        self.cond = cond
        self.then_val = then_val
        self.else_val = else_val

    def code(self, params: dict) -> str:
        c = self.cond.code(params)
        t = self.then_val.code(params)
        e = self.else_val.code(params)
        return f"({c} != 0.0 ? {t} : {e})"

    def signature_key(self, params: dict) -> str:
        c = self.cond.signature_key(params)
        t = self.then_val.signature_key(params)
        e = self.else_val.signature_key(params)
        return f"where({c},{t},{e})"


def where(cond: Expr, then_val, else_val) -> WhereExpr:
    """Fused where(cond, then_val, else_val) expression."""
    return WhereExpr(cond, _wrap(then_val), _wrap(else_val))


# ============================================================
# Tree walking helpers
# ============================================================

def _find_length(expr: Expr) -> int:
    """Walk expression tree to find array length from any ColRef."""
    if isinstance(expr, ColRef):
        return len(expr.data)
    if isinstance(expr, BinOp):
        return _find_length(expr.left) or _find_length(expr.right)
    if isinstance(expr, Compare):
        return _find_length(expr.left) or _find_length(expr.right)
    if isinstance(expr, PowOp):
        return _find_length(expr.base) or _find_length(expr.exponent)
    if isinstance(expr, UnaryOp):
        return _find_length(expr.operand)
    if isinstance(expr, BoolOr):
        return _find_length(expr.left) or _find_length(expr.right)
    if isinstance(expr, WhereExpr):
        return (_find_length(expr.cond)
                or _find_length(expr.then_val)
                or _find_length(expr.else_val))
    return 0


# ============================================================
# Kernel generation and caching
# ============================================================

_kernel_cache: dict[str, cp.RawKernel] = {}


def _cache_key(prefix: str, expr: Expr) -> str:
    sig_params: dict = {}
    sig = prefix + expr.signature_key(sig_params)
    return hashlib.md5(sig.encode()).hexdigest()[:16]


def _gen_eval_kernel(expr: Expr) -> tuple[cp.RawKernel, list[cp.ndarray]]:
    """Generate a fused element-wise evaluation kernel."""
    params: dict = {}
    body = expr.code(params)

    param_decls = []
    data_list = []
    for name, (param_name, data) in params.items():
        ctype = _ctype_for(data.dtype)
        param_decls.append(f"const {ctype}* {param_name}")
        data_list.append(data)

    src = f"""
extern "C" __global__
void fused_eval({', '.join(param_decls)}, double* out, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = {body};
}}
"""
    key = _cache_key("eval:", expr)
    if key not in _kernel_cache:
        _kernel_cache[key] = cp.RawKernel(src, "fused_eval")
    return _kernel_cache[key], data_list


def _gen_reduce_kernel(expr: Expr) -> tuple[cp.RawKernel, list[cp.ndarray]]:
    """Generate a fused compute+reduce kernel (warp-shuffle sum)."""
    params: dict = {}
    body = expr.code(params)

    param_decls = []
    data_list = []
    for name, (param_name, data) in params.items():
        ctype = _ctype_for(data.dtype)
        param_decls.append(f"const {ctype}* {param_name}")
        data_list.append(data)

    src = f"""
extern "C" __global__
void fused_reduce({', '.join(param_decls)}, double* partial, int n) {{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = 0.0;
    if (i < n) {{ int idx = i; val += {body}; }}
    if (i + blockDim.x < n) {{ int idx = i + blockDim.x; val += {body}; }}

    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {{
        val = (lane < num_warps) ? sdata[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) partial[blockIdx.x] = val;
    }}
}}
"""
    key = _cache_key("reduce:", expr)
    if key not in _kernel_cache:
        _kernel_cache[key] = cp.RawKernel(src, "fused_reduce")
    return _kernel_cache[key], data_list


# ============================================================
# Public API
# ============================================================

def evaluate(expr: Expr) -> cp.ndarray:
    """Evaluate an expression tree, returning a CuPy array.

    Generates a single fused CUDA kernel regardless of expression depth.
    The kernel is compiled once and cached for reuse.
    """
    n = _find_length(expr)
    kernel, data_list = _gen_eval_kernel(expr)

    output = cp.empty(n, dtype=cp.float64)
    threads = 256
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), tuple(data_list) + (output, n))
    return output


def fused_sum(expr: Expr) -> float:
    """Evaluate expression AND reduce to scalar sum in a single fused kernel.

    No intermediate buffer is allocated for the expression result.
    The kernel evaluates the expression per-element and reduces via
    warp-shuffle, all in one launch.
    """
    n = _find_length(expr)
    kernel, data_list = _gen_reduce_kernel(expr)

    threads = 256
    blocks = (n + threads * 2 - 1) // (threads * 2)
    smem = (threads // 32) * 8

    partial = cp.empty(blocks, dtype=cp.float64)
    kernel((blocks,), (threads,), tuple(data_list) + (partial, n), shared_mem=smem)
    return float(cp.sum(partial))
