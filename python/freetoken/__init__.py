"""FreeToken inference runtime."""

import os


def _configure_wsl_rocdxg() -> bool:
    """Enable ROCm's WSL bridge before anything imports and initializes Torch."""
    key = "HSA_ENABLE_DXG_DETECTION"
    if key in os.environ or not os.path.exists("/dev/dxg"):
        return False
    runtime_paths = (
        "/opt/rocm/lib/librocdxg.so",
        "/opt/rocm/lib64/librocdxg.so",
        "/usr/lib/librocdxg.so",
        "/usr/lib/x86_64-linux-gnu/librocdxg.so",
    )
    if not any(os.path.exists(path) for path in runtime_paths):
        return False
    os.environ[key] = "1"
    return True


_configure_wsl_rocdxg()

from freetoken.version import __version__

__all__ = ["__version__"]
