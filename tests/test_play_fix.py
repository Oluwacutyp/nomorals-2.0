"""Regression tests for the /play mpv IPC fix (Termux bug report).

The bug: on Termux, /play failed with "lost the mpv IPC connection".
Root cause: the phone's OOM killer takes out the detached mpv process,
leaving a stale socket file. _mpv_alive() only checked that the socket
file existed (and that _mpv_get_props returned a non-None dict — but a
dead mpv returns a dict of Nones, which is not None). So the engine
thought mpv was alive, skipped the respawn, and every IPC call failed.

Fixes covered here:
1. _mpv_alive() requires at least one real property value, and cleans
   up the stale socket when mpv is dead.
2. _mpv_play_at() respawns mpv once and retries the playlist sync on
   IPC failure before reporting an error.
3. _mpv_spawn() passes --ao=opensles on Termux (default audio output
   has no server to talk to on Android).
4. Error messages name the fix (pkg install mpv) instead of the raw
   "lost the mpv IPC connection".
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.media.playback import Backend, PlaybackEngine
from nomorals.storage.db import Database


def _context(**over: Any) -> SimpleNamespace:
    root = tempfile.mkdtemp(prefix="play_fix_")
    db = Database(":memory:")
    db.migrate()
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=root),
        db=db, tools=None, router=None, gateway=None,
    )
    for k, v in over.items():
        setattr(ctx, k, v)
    ctx._root = root
    return ctx


def _engine() -> PlaybackEngine:
    eng = PlaybackEngine(_context())
    eng.backend = Backend("mpv", "/bin/fake-mpv", ipc=True)
    return eng


class MpvAliveTests(unittest.TestCase):
    """_mpv_alive must not be fooled by a stale socket."""

    def test_no_socket_file_is_dead(self) -> None:
        eng = _engine()
        eng._state["mpv_sock"] = "/nonexistent/ipc.sock"
        self.assertFalse(eng._mpv_alive())

    def test_no_sock_in_state_is_dead(self) -> None:
        eng = _engine()
        eng._state.pop("mpv_sock", None)
        self.assertFalse(eng._mpv_alive())

    def test_stale_socket_dict_of_nones_is_dead(self) -> None:
        """The exact Termux bug: socket file exists, mpv is gone,
        _mpv_get_props returns {prop: None, ...}."""
        eng = _engine()
        sock = os.path.join(tempfile.mkdtemp(), "ipc.sock")
        open(sock, "w").close()  # stale socket file, no listener
        eng._state["mpv_sock"] = sock
        dead_props = {"playback-status": None, "playlist-index": None,
                      "time-pos": None, "volume": None}
        with mock.patch.object(eng, "_mpv_get_props",
                               return_value=dead_props):
            self.assertFalse(eng._mpv_alive())
        # stale socket cleaned up
        self.assertFalse(os.path.exists(sock))
        self.assertNotIn("mpv_sock", eng._state)

    def test_live_mpv_with_real_props_is_alive(self) -> None:
        eng = _engine()
        sock = os.path.join(tempfile.mkdtemp(), "ipc.sock")
        open(sock, "w").close()
        eng._state["mpv_sock"] = sock
        live_props = {"playback-status": "playing", "playlist-index": 0,
                      "time-pos": 12.5, "volume": 80}
        with mock.patch.object(eng, "_mpv_get_props",
                               return_value=live_props):
            self.assertTrue(eng._mpv_alive())
        # live socket NOT deleted
        self.assertTrue(os.path.exists(sock))

    def test_partially_dead_props_still_alive(self) -> None:
        """One real value is enough — mpv might not report time-pos
        before playback starts."""
        eng = _engine()
        sock = os.path.join(tempfile.mkdtemp(), "ipc.sock")
        open(sock, "w").close()
        eng._state["mpv_sock"] = sock
        props = {"playback-status": None, "playlist-index": 0,
                 "time-pos": None, "volume": None}
        with mock.patch.object(eng, "_mpv_get_props", return_value=props):
            self.assertTrue(eng._mpv_alive())

    def test_none_props_dict_is_dead(self) -> None:
        eng = _engine()
        eng._state["mpv_sock"] = "/tmp/whatever.sock"
        with mock.patch.object(eng, "_mpv_get_props", return_value=None):
            # os.path.exists fails first, but patch exists check too
            with mock.patch("os.path.exists", return_value=True):
                self.assertFalse(eng._mpv_alive())


class MpvSpawnTermuxTests(unittest.TestCase):
    """_mpv_spawn uses opensles audio on Termux."""

    def _spawn_cmd(self, prefix: str) -> list[str]:
        eng = _engine()
        captured: dict[str, Any] = {}

        def fake_popen(cmd: list[str], **kw: Any) -> Any:
            captured["cmd"] = cmd
            class P:  # noqa: D106
                pid = 1234
            return P()

        with mock.patch.dict(os.environ, {"PREFIX": prefix}):
            with mock.patch("subprocess.Popen", fake_popen):
                with mock.patch.object(eng, "_mpv_connect",
                                       return_value=mock.Mock(close=lambda: None)):
                    ok, sock, err = eng._mpv_spawn()
        self.assertTrue(ok, err)
        return captured["cmd"]

    def test_termux_gets_opensles(self) -> None:
        cmd = self._spawn_cmd("/data/data/com.termux/files/usr")
        self.assertIn("--ao=opensles", cmd)

    def test_non_termux_no_opensles(self) -> None:
        cmd = self._spawn_cmd("/usr")
        self.assertNotIn("--ao=opensles", cmd)

    def test_no_prefix_no_opensles(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PREFIX", None)
            cmd = self._spawn_cmd("")
        # empty prefix doesn't start with the termux path
        self.assertNotIn("--ao=opensles", cmd)


class MpvRespawnRetryTests(unittest.TestCase):
    """_mpv_play_at respawns once on IPC failure before giving up."""

    def _eng_with_queue(self) -> PlaybackEngine:
        eng = _engine()
        # add a fake file to the queue (bypass safe_path via direct insert)
        eng.db.execute(
            "INSERT INTO media_queue (id, position, path, title, kind, "
            "artist, album, duration, added_at) VALUES (?,?,?,?,?,?,?,?,?)",
            ("q1", 1.0, "/tmp/song.mp3", "Song", "file", "", "", 0, 1.0))
        return eng

    def test_ipc_failure_triggers_one_respawn(self) -> None:
        eng = self._eng_with_queue()
        eng._state["mpv_sock"] = "/tmp/fake.sock"
        calls = {"ipc": 0, "spawn": 0}

        def fake_ipc(sock: str, cmd: list[Any], timeout: float = 2.0) -> bool:
            calls["ipc"] += 1
            # first sync attempt fails, post-respawn succeeds
            return calls["spawn"] > 0

        def fake_spawn() -> tuple[bool, str, str]:
            calls["spawn"] += 1
            return True, "/tmp/new.sock", ""

        with mock.patch.object(eng, "_mpv_alive", return_value=True):
            with mock.patch.object(eng, "_mpv_ipc", fake_ipc):
                with mock.patch.object(eng, "_mpv_spawn", fake_spawn):
                    ok, err = eng._mpv_play_at(0)
        self.assertTrue(ok, err)
        self.assertEqual(calls["spawn"], 1)
        self.assertEqual(eng._state["mpv_sock"], "/tmp/new.sock")

    def test_double_failure_gives_actionable_error(self) -> None:
        eng = self._eng_with_queue()
        eng._state["mpv_sock"] = "/tmp/fake.sock"
        with mock.patch.object(eng, "_mpv_alive", return_value=True):
            with mock.patch.object(eng, "_mpv_ipc", return_value=False):
                with mock.patch.object(
                        eng, "_mpv_spawn",
                        return_value=(True, "/tmp/new.sock", "")):
                    ok, err = eng._mpv_play_at(0)
        self.assertFalse(ok)
        # error names the fix, not just "lost the mpv IPC connection"
        self.assertIn("pkg install mpv", err)

    def test_spawn_failure_reports_cause(self) -> None:
        eng = self._eng_with_queue()
        eng._state["mpv_sock"] = "/tmp/fake.sock"
        with mock.patch.object(eng, "_mpv_alive", return_value=True):
            with mock.patch.object(eng, "_mpv_ipc", return_value=False):
                with mock.patch.object(
                        eng, "_mpv_spawn",
                        return_value=(False, "", "no mpv binary")):
                    ok, err = eng._mpv_play_at(0)
        self.assertFalse(ok)
        self.assertIn("no mpv binary", err)

    def test_happy_path_no_respawn(self) -> None:
        eng = self._eng_with_queue()
        eng._state["mpv_sock"] = "/tmp/fake.sock"
        spawns = []
        with mock.patch.object(eng, "_mpv_alive", return_value=True):
            with mock.patch.object(eng, "_mpv_ipc", return_value=True):
                with mock.patch.object(
                        eng, "_mpv_spawn",
                        side_effect=lambda: spawns.append(1) or (
                            True, "/tmp/x.sock", "")):
                    ok, err = eng._mpv_play_at(0)
        self.assertTrue(ok, err)
        self.assertEqual(spawns, [])


class NeverRaisesTests(unittest.TestCase):
    def test_mpv_alive_never_raises_on_garbage(self) -> None:
        eng = _engine()
        for bad in (None, "", 123, {"mpv_sock": None}):
            try:
                if isinstance(bad, dict):
                    eng._state.update(bad)
                else:
                    eng._state["mpv_sock"] = bad  # type: ignore[assignment]
                eng._mpv_alive()
            except Exception as exc:  # noqa: BLE001
                self.fail(f"_mpv_alive raised on {bad!r}: {exc}")


if __name__ == "__main__":
    unittest.main()
