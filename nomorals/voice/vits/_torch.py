"""Torch gate for the VITS torch half.

Every torch module imports through here so the package imports cleanly
on a box without torch and the failure names exactly what is missing.
"""

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def torch_available():
    return _TORCH_AVAILABLE


def require_torch(what="this VITS operation"):
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{what} needs torch, which is not installed on this machine. "
            "Install torch (CPU build is enough for the prepare path's "
            "parity checks; training wants CUDA) and retry."
        )
    import torch
    return torch
