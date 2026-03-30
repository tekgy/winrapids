"""
Pipeline — lazy compiled pipeline via Rust compiler.

Build a pipeline lazily in Python. Nothing executes until .compile() or .collect().
The graph goes to the Rust compiler which optimizes via CSE, provenance reuse,
and fusion — then returns the execution plan.

Usage:
    import winrapids as wr

    pipe = wr.pipeline()
    pipe.rolling_zscore("price", window=20)
    pipe.rolling_std("price", window=20)
    plan = pipe.compile()

    print(plan.cse_stats)
    # {'original_nodes': 6, 'after_cse': 4, 'eliminated': 2, 'elimination_pct': 33}

The compiler automatically finds that rolling_zscore and rolling_std share
scan(price, add) and scan(price_sq, add) — eliminating 33% of the work.
"""

from __future__ import annotations

import _winrapids_core


class CompiledPipeline:
    """A compiled pipeline — the Rust compiler's output.

    Access CSE stats, execution steps, and output mappings.
    """

    __slots__ = ("_plan",)

    def __init__(self, plan: _winrapids_core.Plan):
        self._plan = plan

    @property
    def cse_stats(self) -> dict:
        """CSE statistics: original_nodes, after_cse, eliminated, elimination_pct."""
        return self._plan.cse_stats

    @property
    def steps(self) -> list:
        """Execution steps in topological order."""
        return self._plan.steps

    @property
    def outputs(self) -> list:
        """Output mappings: (call_idx, output_name, identity)."""
        return self._plan.outputs

    def __len__(self) -> int:
        return len(self._plan)

    def __repr__(self) -> str:
        return repr(self._plan)


class Pipeline:
    """Lazy pipeline builder. Add specialist calls, then compile.

    Methods match specialist names in the registry. Each call adds
    a specialist invocation — nothing executes until .compile().
    """

    __slots__ = ("_pipe",)

    def __init__(self):
        self._pipe = _winrapids_core.Pipeline()

    def add(self, specialist: str, data: str, window: int = 20) -> Pipeline:
        """Add a specialist call. Returns self for chaining."""
        self._pipe.add(specialist, data=data, window=window)
        return self

    def rolling_mean(self, data: str, window: int = 20) -> Pipeline:
        """Add rolling_mean specialist."""
        return self.add("rolling_mean", data, window)

    def rolling_std(self, data: str, window: int = 20) -> Pipeline:
        """Add rolling_std specialist."""
        return self.add("rolling_std", data, window)

    def rolling_zscore(self, data: str, window: int = 20) -> Pipeline:
        """Add rolling_zscore specialist."""
        return self.add("rolling_zscore", data, window)

    def compile(self) -> CompiledPipeline:
        """Compile the pipeline via the Rust compiler.

        This is where the sharing optimizer does its work.
        Returns a CompiledPipeline with CSE stats and execution steps.
        """
        plan = self._pipe.compile()
        return CompiledPipeline(plan)

    def execute(self, data: dict, *, use_store: bool = False) -> dict:
        """Execute the pipeline with mock dispatch.

        Args:
            data: mapping of data variable names to (device_ptr, byte_size) tuples.
                  e.g. {"price": (0x100, 8000)}
            use_store: if True, use GpuStore for provenance-based reuse.
                       if False (default), use NullWorld (compute everything).

        Returns a dict with:
            - "plan": the compiled Plan
            - "stats": {"hits": N, "misses": N, "hit_rate": float}
            - "outputs": list of (call_idx, output_name, device_ptr, byte_size, was_hit)
        """
        return self._pipe.execute(data, use_store=use_store)

    def __len__(self) -> int:
        return len(self._pipe)

    def __repr__(self) -> str:
        return repr(self._pipe)


def list_specialists() -> list[str]:
    """List available specialist names."""
    return _winrapids_core.list_specialists()


def specialist_dag(name: str) -> list[tuple[str, str, list[str]]]:
    """Get the primitive decomposition for a specialist."""
    return _winrapids_core.specialist_dag(name)
