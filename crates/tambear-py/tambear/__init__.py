"""tambear — sort-free GPU DataFrame engine.

Tam doesn't sort. Tam knows where everything is.

Usage:
    import tambear as tb

    df = tb.from_columns({"ticker_id": [0,0,1,1,2], "close": [149.0, 150.1, ...]})
    stats = df.groupby("ticker_id").stats("close")
"""

import os as _os
import sys as _sys

# Auto-discover CUDA DLLs on Windows (cudarc needs nvrtc, cuda driver)
if _sys.platform == "win32":
    _cuda_path = _os.environ.get("CUDA_PATH", "")
    if not _cuda_path:
        # Search standard install locations, prefer latest
        _base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if _os.path.isdir(_base):
            _versions = sorted(_os.listdir(_base), reverse=True)
            if _versions:
                _cuda_path = _os.path.join(_base, _versions[0])
    if _cuda_path:
        for _subdir in ("bin/x64", "bin"):
            _dll_dir = _os.path.join(_cuda_path, _subdir)
            if _os.path.isdir(_dll_dir):
                _os.add_dll_directory(_dll_dir)

from tambear._tambear import (
    Frame,
    GroupByBuilder,
    GroupByResult,
    from_columns,
    read,
)

__all__ = [
    "Frame",
    "GroupByBuilder",
    "GroupByResult",
    "from_columns",
    "read",
]
