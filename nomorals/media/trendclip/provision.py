"""Self-provisioning for trendclip's heavy dependencies.

Devon installs her own runtimes and dependencies — a missing tool is her
job to fix, never a command handed to the owner. These two functions mirror
the shape of ``nomorals/media/resolver.ensure_ytdlp`` exactly: check for
the import, try one bounded pip install into the current environment, and
never raise — the caller reads the boolean and names the missing piece
honestly. Under pytest they are a no-op (a test run must never install).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys

__all__ = ["ensure_faster_whisper", "ensure_opencv"]

_log = logging.getLogger(__name__)

_PIP_TIMEOUT_S = 240


def _under_pytest() -> bool:
    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _ensure(module: str, package: str, *extra_packages: str) -> bool:
    """True when ``module`` imports (after one bounded pip attempt)."""
    if importlib.util.find_spec(module) is not None:
        return True
    if _under_pytest():
        return False
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             package, *extra_packages],
            timeout=_PIP_TIMEOUT_S, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001 - provision failure is a bool, not a crash
        _log.warning("trendclip provision of %s failed: %s", package, exc)
        return False
    return importlib.util.find_spec(module) is not None


def ensure_faster_whisper() -> bool:
    """Ensure the Faster Whisper STT backend is importable.

    faster-whisper 1.2.1 still calls ``av.open(..., metadata_errors=...)``
    but PyAV 14+ dropped that parameter — an unbounded ``pip install
    faster-whisper`` resolves av 19 and every transcription then dies
    with ``TypeError: open() got an unexpected keyword argument
    'metadata_errors'``. The av bound below is the fix, here and in
    pyproject's media extra; it is pinned in tests.
    """
    return _ensure("faster_whisper", "faster-whisper>=1.1",
                   "av>=11,<14")


def ensure_opencv() -> bool:
    """Ensure OpenCV (the vertical reframe) is importable."""
    return _ensure("cv2", "opencv-python-headless>=4.8")
