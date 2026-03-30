"""
WinRapids — Windows-native GPU-accelerated data science toolkit.

Built for Windows WDDM, informed by constraints, powered by CUDA.
"""

from winrapids.column import Column
from winrapids.frame import Frame
from winrapids.fusion import evaluate, fused_sum
from winrapids.transfer import h2d, d2h

# Compiled pipeline (Rust compiler via PyO3)
try:
    from winrapids.pipeline import Pipeline, CompiledPipeline, list_specialists, specialist_dag

    def pipeline() -> Pipeline:
        """Create a new lazy pipeline builder."""
        return Pipeline()

    __all__ = [
        "Column", "Frame", "evaluate", "fused_sum", "h2d", "d2h",
        "Pipeline", "CompiledPipeline", "pipeline", "list_specialists", "specialist_dag",
    ]
except ImportError:
    # Native module not built yet — pipeline API unavailable
    __all__ = ["Column", "Frame", "evaluate", "fused_sum", "h2d", "d2h"]
