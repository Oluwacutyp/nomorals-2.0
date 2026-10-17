"""Cross-platform exclusive file locking.

``fcntl`` is a Unix-only stdlib module — it does not exist on Windows, ever,
so ``import fcntl`` at module top level is a guaranteed crash on Windows
(the ``/play`` blow-up in ``nomorals/browser/daemon.py``). Every file lock in
the repo must go through here; never ``import fcntl`` at module top level.

* POSIX: ``fcntl.flock`` (advisory, per-fd, released on close).
* Windows: ``msvcrt.locking`` (byte-range lock on the first byte — same
  mutual-exclusion semantics between cooperating processes).
* Neither primitive available: best-effort proceed (documented, never a
  crash). No production platform hits this path.
"""

from __future__ import annotations

import contextlib
import os

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows has no msvcrt
    msvcrt = None  # type: ignore[assignment]

__all__ = ["lock_exclusive_nb", "unlock"]


def lock_exclusive_nb(fd: int) -> bool:
    """Take a non-blocking exclusive lock on ``fd``.

    Returns True when the lock was acquired, False when another process
    already holds it. Never raises for contention.
    """
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    if msvcrt is not None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    # No OS locking primitive: single-process fallback. Best-effort only.
    return True


def unlock(fd: int) -> None:
    """Release a lock taken by :func:`lock_exclusive_nb` (best-effort)."""
    if fcntl is not None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
