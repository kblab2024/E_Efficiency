"""
GPU/CPU backend configuration for multilayer optics simulations.

Auto-detects NVIDIA GPU (CUDA) and selects CuPy for GPU-accelerated array
operations, falling back to NumPy for CPU-only execution.

Hardware targets:
    - GPU:  NVIDIA RTX A6000 Ada (or any CUDA-capable GPU)
    - CPU:  Intel Xeon w5-3423 (or any multi-core CPU)

Usage:
    from gpu_config import xp, using_gpu
    # xp is either cupy or numpy — use it everywhere instead of np
"""

import os

# Allow explicit override: set USE_GPU=0 to force CPU, USE_GPU=1 to force GPU
_force = os.environ.get("USE_GPU", None)

using_gpu = False
xp = None

if _force != "0":
    try:
        import cupy as cp
        # Quick sanity check: allocate a tiny array on the GPU
        _test = cp.zeros(1)
        del _test
        xp = cp
        using_gpu = True
    except Exception:
        pass

if xp is None:
    import numpy as np
    xp = np
    using_gpu = False


def trapz(y, x):
    """Backend-portable trapezoidal integration (numpy.trapz was removed in NumPy 2)."""
    if hasattr(xp, 'trapezoid'):
        return xp.trapezoid(y, x)
    return xp.trapz(y, x)


def to_numpy(arr):
    """Convert array to NumPy (no-op if already NumPy)."""
    if using_gpu:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return arr


def to_device(arr):
    """Move a NumPy array to the active device (GPU or stay on CPU)."""
    if using_gpu:
        import cupy as cp
        return cp.asarray(arr)
    return arr


def print_config():
    """Print detected hardware configuration."""
    if using_gpu:
        import cupy as cp
        dev = cp.cuda.Device()
        print(f"[gpu_config] Using GPU: {dev.id} — {cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()}")
        mem = dev.mem_info
        print(f"[gpu_config] GPU memory: {mem[1] / 1e9:.1f} GB total, {mem[0] / 1e9:.1f} GB free")
    else:
        print("[gpu_config] Using CPU (NumPy). Set USE_GPU=1 or install CuPy for GPU acceleration.")
    print(f"[gpu_config] Backend: {xp.__name__}")
