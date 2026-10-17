"""Cross-platform file locking (nomorals/core/filelock.py) + Windows guards.

Regression: ``/play`` blew up on Windows with
``ModuleNotFoundError: No module named 'fcntl'`` because
``nomorals/browser/daemon.py`` did ``import fcntl`` at top level. ``fcntl``
is Unix-only and never exists on Windows. The lock now goes through
``core.filelock`` (flock on POSIX, msvcrt.locking on Windows), and every
other Unix-only stdlib import / process-group kill in the repo is guarded.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


@contextmanager
def _blocked_imports(*names: str):
    """Block named modules from importing (simulate Windows missing fcntl).

    Restores the exact pre-block ``sys.modules`` state for the affected
    subtree (``nomorals``, ``nomorals.core``, ``nomorals.browser``,
    ``nomorals.core.filelock``, ``nomorals.browser.daemon`` + the blocked
    names). This matters because ``from package import submodule`` resolves
    via the parent package's *attribute*, not ``sys.modules`` — popping the
    submodule alone leaves the parent bound to the blocked module.
    """
    import sys as _sys

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):  # noqa: D102
            if fullname in names:
                raise ImportError(f"blocked for test: {fullname}")
            return None

    watched = (
        "nomorals",
        "nomorals.core",
        "nomorals.browser",
        "nomorals.core.filelock",
        "nomorals.browser.daemon",
        *names,
    )
    _MISSING = object()
    snapshot = {m: _sys.modules.get(m, _MISSING) for m in watched}
    # Drop the watched modules so they re-execute their imports inside
    # the block (with the blocked names raising ImportError).
    for mod in watched:
        _sys.modules.pop(mod, None)
    blocker = _Blocker()
    _sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        _sys.meta_path.remove(blocker)
        for mod, val in snapshot.items():
            if val is _MISSING:
                _sys.modules.pop(mod, None)
            else:
                _sys.modules[mod] = val


class TestFilelock(unittest.TestCase):
    def test_daemon_imports_without_fcntl(self):
        """daemon.py imports cleanly with fcntl AND msvcrt unavailable."""
        with _blocked_imports("fcntl", "msvcrt"):
            filelock = importlib.import_module("nomorals.core.filelock")
            self.assertIsNone(filelock.fcntl)
            self.assertIsNone(filelock.msvcrt)
            daemon = importlib.import_module("nomorals.browser.daemon")
            self.assertTrue(hasattr(daemon, "_exclusive_lock"))

    def test_lock_helper_without_any_primitive(self):
        """No fcntl, no msvcrt: best-effort proceed, never a crash."""
        from nomorals.core import filelock

        with mock.patch.object(filelock, "fcntl", None), \
                mock.patch.object(filelock, "msvcrt", None):
            fd = os.open(__file__, os.O_RDONLY)
            try:
                self.assertTrue(filelock.lock_exclusive_nb(fd))
                filelock.unlock(fd)  # must not raise
            finally:
                os.close(fd)

    def test_lock_helper_msvcrt_branch(self):
        """The Windows branch works: contention fails, unlock frees."""
        from nomorals.core import filelock

        locked: set[tuple[int, int]] = set()

        class _FakeMsvcrt:
            LK_NBLCK = 1
            LK_UNLCK = 2

            @staticmethod
            def locking(fd, mode, _nbytes):
                st = os.fstat(fd)
                key = (st.st_dev, st.st_ino)
                if mode == _FakeMsvcrt.LK_NBLCK:
                    if key in locked:
                        raise OSError("file locked")
                    locked.add(key)
                else:
                    locked.discard(key)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            path.write_bytes(b"")
            fd1 = os.open(path, os.O_RDWR)
            fd2 = os.open(path, os.O_RDWR)
            try:
                with mock.patch.object(filelock, "fcntl", None), \
                        mock.patch.object(filelock, "msvcrt", _FakeMsvcrt):
                    self.assertTrue(filelock.lock_exclusive_nb(fd1))
                    self.assertFalse(filelock.lock_exclusive_nb(fd2))
                    filelock.unlock(fd1)
                    self.assertTrue(filelock.lock_exclusive_nb(fd2))
                    filelock.unlock(fd2)
            finally:
                os.close(fd1)
                os.close(fd2)

    @unittest.skipIf(os.name == "nt", "fcntl branch is POSIX-only")
    def test_lock_helper_fcntl_contention(self):
        """POSIX branch: a second open description cannot take the lock."""
        from nomorals.core import filelock

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            path.write_bytes(b"")
            fd1 = os.open(path, os.O_RDWR)
            fd2 = os.open(path, os.O_RDWR)
            try:
                self.assertTrue(filelock.lock_exclusive_nb(fd1))
                self.assertFalse(filelock.lock_exclusive_nb(fd2))
                filelock.unlock(fd1)
                self.assertTrue(filelock.lock_exclusive_nb(fd2))
                filelock.unlock(fd2)
            finally:
                os.close(fd1)
                os.close(fd2)


class TestDaemonExclusiveLock(unittest.TestCase):
    def test_nested_exclusive_lock_raises(self):
        from nomorals.browser.daemon import DaemonError, _exclusive_lock

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daemon.lock"
            with _exclusive_lock(path):
                with self.assertRaises(DaemonError):
                    with _exclusive_lock(path):
                        pass

    def test_sequential_exclusive_lock_ok(self):
        from nomorals.browser.daemon import _exclusive_lock

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daemon.lock"
            with _exclusive_lock(path):
                pass
            with _exclusive_lock(path):
                pass


class TestSnapshotLock(unittest.TestCase):
    def test_system_state_reflects_held_lock(self):
        from nomorals.os.snapshots import SnapshotManager, system_state

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = home / "db.sqlite"
            db.write_bytes(b"")
            self.assertFalse(system_state(home, db)["running"])
            mgr = SnapshotManager(home, db)
            fd = mgr._acquire_lock(timeout=2.0)
            try:
                self.assertTrue(system_state(home, db)["running"])
            finally:
                mgr._release_lock(fd)
            self.assertFalse(system_state(home, db)["running"])

    def test_acquire_lock_times_out(self):
        from nomorals.os.snapshots import SnapshotManager, SnapshotRefused

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = home / "db.sqlite"
            db.write_bytes(b"")
            mgr = SnapshotManager(home, db)
            fd = mgr._acquire_lock(timeout=2.0)
            try:
                mgr2 = SnapshotManager(home, db)
                with self.assertRaises(SnapshotRefused):
                    mgr2._acquire_lock(timeout=0.3)
            finally:
                mgr._release_lock(fd)


class TestShellUidFallback(unittest.TestCase):
    def test_uid_thread_count_without_getuid(self):
        """Windows has no os.getuid — the sandbox helper returns 0."""
        from nomorals.tools import shell

        with mock.patch.object(
            os, "getuid", side_effect=AttributeError("no getuid")
        ):
            self.assertEqual(shell._uid_thread_count(), 0)


if __name__ == "__main__":
    unittest.main()
