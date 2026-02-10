"""
Hardware acceleration backend for Apple Silicon M3 (and compatible fallback).

Priority:
    1. MLX  – Apple-native GPU/Neural-Engine framework (pip install mlx)
    2. NumPy – uses Apple Accelerate (vecLib) on macOS for BLAS/LAPACK

The module exposes a thin wrapper so the rest of the codebase can call
``xp.sqrt``, ``xp.exp``, etc. without caring which backend is active.

Usage::

    from accel_backend import xp, to_numpy, HW_INFO

    a = xp.array([1.0, 2.0, 3.0])
    result = xp.sqrt(a)
    np_result = to_numpy(result)      # always returns a numpy array
"""

from __future__ import annotations

import platform
import numpy as np

# ------------------------------------------------------------------
# Detect hardware & choose backend
# ------------------------------------------------------------------
_backend = "numpy"           # default fallback
_device_name = "CPU (NumPy)"

# Try MLX (Apple Silicon GPU framework)
try:
    import mlx.core as mx          # type: ignore[import-untyped]

    # MLX is only meaningful on Apple Silicon
    if platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin":
        _backend = "mlx"
        _device_name = "Apple Silicon GPU (MLX)"
except ImportError:
    pass

# ------------------------------------------------------------------
# Unified array namespace  (xp)
# ------------------------------------------------------------------
if _backend == "mlx":
    import mlx.core as mx          # type: ignore[import-untyped]
    xp = mx
else:
    xp = np


def to_numpy(a) -> np.ndarray:
    """Convert any backend array to a plain NumPy ndarray."""
    if _backend == "mlx":
        import mlx.core as mx      # type: ignore[import-untyped]
        if isinstance(a, mx.array):
            return np.array(a)
    return np.asarray(a)


# ------------------------------------------------------------------
# Public info dict
# ------------------------------------------------------------------
HW_INFO: dict[str, str] = {
    "backend": _backend,
    "device": _device_name,
    "platform": f"{platform.system()} {platform.machine()}",
}


if __name__ == "__main__":
    print("Hardware acceleration backend info:")
    for k, v in HW_INFO.items():
        print(f"  {k}: {v}")
