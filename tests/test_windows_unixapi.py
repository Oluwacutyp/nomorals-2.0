"""Windows-compat: Unix-only OS APIs must degrade, never raise AttributeError.

On Windows ``os.killpg``/``os.getpgid``/``os.getuid``/``os.waitpid`` and
``signal.SIGKILL`` do not exist, ``fcntl`` cannot be imported, and
``socket.AF_UNIX`` is absent — so merely *evaluating* a kill expression
raises ``AttributeError``, which every pre-existing ``except (OSError, …)``
missed. These tests simulate the missing APIs (monkeypatch + a subprocess
import blocker) and assert each call site falls back instead of raising.
"""

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


@contextlib.contextmanager
def _no_sigkill():
    """Simulate Windows: ``signal.SIGKILL`` does not exist."""
    orig = signal.SIGKILL
    del signal.SIGKILL
    try:
        yield
    finally:
        signal.SIGKILL = orig


class TestDaemonWindows(unittest.TestCase):
    def test_imports_without_fcntl_or_msvcrt_subprocess(self):
        """Import-blocker subprocess: daemon + snapshots import with
        neither fcntl nor msvcrt available; locks yield, probes stay safe."""
        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name in ('fcntl', 'msvcrt'):\n"
            "            raise ModuleNotFoundError('blocked ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "sys.modules.pop('fcntl', None); sys.modules.pop('msvcrt', None)\n"
            "import tempfile, pathlib\n"
            "import nomorals.core.filelock as fl\n"
            "assert fl.fcntl is None and fl.msvcrt is None\n"
            "import nomorals.browser.daemon as d\n"
            "from nomorals.os import snapshots\n"
            "with tempfile.TemporaryDirectory() as td:\n"
            "    home = pathlib.Path(td)\n"
            "    st = snapshots.system_state(home, home / 'db.sqlite')\n"
            "    assert st == {'running': False, 'dirty': False}, st\n"
            "    with d._exclusive_lock(home / 'daemon.lock'):\n"
            "        pass\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_escalate_sigterm_only_without_sigkill(self):
        import nomorals.browser.daemon as daemon_mod
        control = daemon_mod.DaemonControl.__new__(daemon_mod.DaemonControl)
        calls = []
        stub_signal = types.SimpleNamespace(SIGTERM=signal.SIGTERM)
        with mock.patch.object(daemon_mod, "signal", stub_signal), \
                mock.patch.object(daemon_mod, "_pid_alive", return_value=True), \
                mock.patch.object(control, "_wait_gone", return_value=True), \
                mock.patch.object(control, "_reap", return_value=None), \
                mock.patch("os.kill", side_effect=lambda pid, sig: calls.append(sig)):
            self.assertTrue(control._escalate(1234))
        self.assertEqual(calls, [signal.SIGTERM])

    def test_reap_tolerates_attribute_error(self):
        import nomorals.browser.daemon as daemon_mod
        with mock.patch("os.waitpid", side_effect=AttributeError("no waitpid")):
            daemon_mod.DaemonControl._reap(1234)  # must not raise

    def test_exclusive_lock_without_any_backend(self):
        import nomorals.browser.daemon as daemon_mod
        from nomorals.core import filelock
        with mock.patch.object(filelock, "fcntl", None), \
                mock.patch.object(filelock, "msvcrt", None), \
                tempfile.TemporaryDirectory() as td:
            with daemon_mod._exclusive_lock(Path(td) / "daemon.lock"):
                pass  # no backend: proceeds unlocked, never raises

    def test_client_errors_without_af_unix(self):
        import nomorals.browser.daemon as daemon_mod
        with mock.patch.object(daemon_mod, "_af_unix", return_value=None), \
                tempfile.TemporaryDirectory() as td:
            with self.assertRaises(daemon_mod.DaemonError):
                daemon_mod.DaemonClient(td).call("ping", {})

    def test_serve_returns_1_without_af_unix(self):
        import nomorals.browser.daemon as daemon_mod
        with mock.patch.object(daemon_mod, "_af_unix", return_value=None), \
                tempfile.TemporaryDirectory() as td:
            self.assertEqual(daemon_mod.serve(td), 1)


class TestShellWindows(unittest.TestCase):
    def test_uid_thread_count_without_getuid(self):
        from nomorals.tools.shell import _uid_thread_count
        with mock.patch("os.getuid", side_effect=AttributeError("no getuid")):
            self.assertEqual(_uid_thread_count(), 0)

    def test_kill_tree_falls_back_to_proc_kill(self):
        from nomorals.tools.shell import _kill_tree
        proc = mock.Mock()
        proc.pid = 99999
        with mock.patch("os.killpg", side_effect=AttributeError("no killpg")):
            _kill_tree(proc)
        proc.kill.assert_called_once_with()


class TestSshSocksWindows(unittest.TestCase):
    def test_stop_falls_back_to_terminate(self):
        from nomorals.tools.ssh_socks import SshSocksTunnel
        tunnel = SshSocksTunnel.__new__(SshSocksTunnel)
        tunnel._lock = threading.Lock()
        tunnel._stop = threading.Event()
        proc = mock.Mock()
        proc.pid = 99999
        proc.poll.return_value = None
        tunnel.proc = proc
        with mock.patch("os.killpg", side_effect=AttributeError("no killpg")):
            result = tunnel.stop()
        proc.terminate.assert_called_once_with()
        self.assertTrue(result["stopped"])


class TestMediaWindows(unittest.TestCase):
    def test_videos_kill_tree_falls_back_to_proc_kill(self):
        from nomorals.media_edit.videos import _kill_tree
        proc = mock.Mock()
        proc.pid = 99999
        with mock.patch("os.killpg", side_effect=AttributeError("no killpg")):
            _kill_tree(proc)
        proc.kill.assert_called_once_with()

    def test_playback_kill_simple_falls_back_to_sigterm(self):
        from nomorals.media import playback as playback_mod
        engine = playback_mod.PlaybackEngine.__new__(playback_mod.PlaybackEngine)
        engine._state = {"player_pid": 99999}
        calls = []
        with mock.patch("os.killpg", side_effect=AttributeError("no killpg")), \
                mock.patch("os.kill", side_effect=lambda pid, sig: calls.append((pid, sig))):
            engine._kill_simple()
        self.assertEqual(calls, [(99999, signal.SIGTERM)])
        self.assertNotIn("player_pid", engine._state)

    def test_playback_mpv_connect_none_without_af_unix(self):
        from nomorals.media import playback as playback_mod
        engine = playback_mod.PlaybackEngine.__new__(playback_mod.PlaybackEngine)
        engine._state = {}
        with mock.patch.object(playback_mod, "socket", types.SimpleNamespace()):
            self.assertIsNone(engine._mpv_connect("/nonexistent/mpv.sock",
                                                 timeout=0.1))


class TestLocalServerWindows(unittest.TestCase):
    def test_kill_pid_uses_sigterm_without_sigkill(self):
        from nomorals.llm import local_server
        calls = []
        zero_probes = []

        def fake_kill(pid, sig):
            calls.append(sig)
            if sig == 0:
                zero_probes.append(1)
                if len(zero_probes) >= 3:
                    raise ProcessLookupError("gone")
                return None
            return None

        with _no_sigkill(), mock.patch("os.kill", side_effect=fake_kill):
            self.assertTrue(local_server.kill_pid(99999, timeout=0.05))
        self.assertIn(signal.SIGTERM, calls)


class TestBuildersWindows(unittest.TestCase):
    def test_run_kill_tree_sigkill_falls_back_to_sigterm(self):
        from nomorals.builders import run as run_mod
        proc = mock.Mock()
        proc.pid = 99999
        calls = []
        with mock.patch("os.killpg", side_effect=AttributeError("no killpg")), \
                mock.patch("os.kill", side_effect=lambda pid, sig: calls.append(sig)), \
                _no_sigkill():
            run_mod._kill_tree(proc, "SIGKILL")
        self.assertEqual(calls, [signal.SIGTERM])

    def test_app_builder_signal_tree_sigkill_falls_back_to_sigterm(self):
        from nomorals.builders import app_builder
        calls = []
        with mock.patch("os.killpg", side_effect=AttributeError("no killpg")), \
                mock.patch("os.kill", side_effect=lambda pid, sig: calls.append(sig) or None), \
                _no_sigkill():
            self.assertTrue(app_builder._signal_tree(99999, "SIGKILL"))
        self.assertEqual(calls, [signal.SIGTERM])


class TestSnapshotsWindows(unittest.TestCase):
    def _block_backends(self):
        """Context manager: both fcntl and msvcrt unimportable."""
        return mock.patch.dict(sys.modules, {"fcntl": None, "msvcrt": None})

    def test_fake_msvcrt_lock_contention(self):
        from nomorals.core import filelock
        from nomorals.os import snapshots
        fake = types.ModuleType("msvcrt")
        fake.LK_UNLCK, fake.LK_LOCK, fake.LK_NBLCK, fake.LK_RLCK = 0, 1, 2, 3
        # Locks are per-file (byte range), not per-fd: two fds on the same
        # file contend, like real msvcrt.locking.
        locked: set[tuple[int, int]] = set()

        def file_key(fd):
            st = os.fstat(fd)
            return (st.st_dev, st.st_ino)

        def locking(fd, mode, nbytes):
            key = file_key(fd)
            if mode in (fake.LK_LOCK, fake.LK_NBLCK):
                if key in locked:
                    raise OSError("lock contended")
                locked.add(key)
            elif mode == fake.LK_UNLCK:
                locked.discard(key)

        fake.locking = locking
        with mock.patch.object(filelock, "fcntl", None), \
                mock.patch.object(filelock, "msvcrt", fake), \
                tempfile.TemporaryDirectory() as td:
            home = Path(td)
            lock_path = home / snapshots.LOCK_FILENAME
            fd1 = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            fd2 = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                self.assertTrue(filelock.lock_exclusive_nb(fd1))
                # Second fd in the same process contends: lock held.
                self.assertFalse(filelock.lock_exclusive_nb(fd2))
                self.assertTrue(snapshots.system_state(home, home / "db.sqlite")["running"])
                filelock.unlock(fd1)
                self.assertTrue(filelock.lock_exclusive_nb(fd2))
                filelock.unlock(fd2)
            finally:
                os.close(fd1)
                os.close(fd2)

    def test_acquire_lock_fails_closed_without_backend(self):
        from nomorals.os.snapshots import SnapshotManager, SnapshotRefused
        with self._block_backends(), tempfile.TemporaryDirectory() as td:
            mgr = SnapshotManager(td, str(Path(td) / "db.sqlite"))
            with self.assertRaises(SnapshotRefused):
                mgr._acquire_lock(timeout=0.1)


if __name__ == "__main__":
    unittest.main()
