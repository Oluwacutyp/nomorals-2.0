"""Tail-32 pins: media/playback.py failure values + NOT-recording.

Verdict: pins-only, 0 converts / 9 exclusions. Every handler below is a
documented never-raises degradation whose fallback value is either the
documented miss value, the input itself, last-known state, or an
explicitly marked unreachable value — never a fabricated claim. Each pin
drives the handler into its failure path and pins the exact fallback
value plus the fact that the agents failures ledger is unchanged
(snapshot before/after, tail-8 method). playback.py calls no record
function anywhere (grep-proven: only _log calls), so the ledger must
not move.

Grounds in brief (full case in SURVEY.md): 286 failed Spotify metadata
lookup falls back to the URI itself, the docstring-documented value, and
the URI is genuine display text; 332 failed state load returns {} —
settings/position defaults only, since the queue itself reads the
media_queue table directly and is unaffected; 1129 failed Spotify live
read falls back to last-commanded engine state for playing/paused while
progress stays honestly None; 1314 failed stale-socket unlink leaves the
verdict unchanged (mpv still dead, state still popped); 1344 missing
stale socket is the expected fresh-spawn case, not a failure; 1378
drain-loop timeout IS the designed loop exit (command already sent);
1415 partial IPC response is parsed only for a success line, else None
— a timeout never fabricates data; 1670 failed Spotify pause still
returns the stop acknowledgement because local audio was stopped and
the state is set stopped (live data self-corrects on the next read);
1698 failed Spotify live read is explicitly marked "unreachable" in the
output — the honest pattern.

The pins file itself contains no exception handlers, so it adds zero
census sites.
"""

import os
import socket
import tempfile

from nomorals.agents import failures as ag_fail
from nomorals.media import playback as pb
from nomorals.media.playback import PlaybackEngine


def _snap():
    return list(ag_fail.recent_failures())


class StubBackend:
    def __init__(self, name="console", binary="mpv"):
        self.name = name
        self.binary = binary

    def describe(self):
        return {"name": self.name}


class StubDb:
    def __init__(self, rows):
        self.rows = rows

    def query(self, *args, **kwargs):
        return self.rows


ROW = {"path": "spotify:track:abc", "title": "Test Song", "kind": "spotify",
       "artist": "A", "album": "", "duration": 200}


def _engine(state=None, spotify=None, db=None):
    e = object.__new__(PlaybackEngine)
    e.context = object()
    e.db = db if db is not None else StubDb([ROW])
    e.backend = StubBackend()
    e._state = dict(state) if state is not None else {"position": 0}
    e._spotify = spotify
    e._soundcloud = None
    e._sock_dir = ""
    return e


class _RaisingSpotify:
    def __init__(self):
        self.calls = []

    def get_track(self, uri):
        self.calls.append(uri)
        raise RuntimeError("adapter down")

    def now_playing(self):
        self.calls.append("now_playing")
        raise RuntimeError("spotify down")

    def pause(self):
        self.calls.append("pause")
        raise RuntimeError("spotify down")


class _FakeMusicLibrary:
    def __init__(self, context):
        pass

    def is_liked(self, path):
        return False


def test_286_spotify_meta_falls_back_to_uri():
    before = _snap()
    e = _engine(spotify=_RaisingSpotify())
    assert e._spotify_meta("spotify:track:abc", "") == ("spotify:track:abc", "")
    assert e._spotify.calls == ["spotify:track:abc"]
    assert _snap() == before


def test_332_state_load_failure_returns_empty_defaults():
    before = _snap()

    class _RaisingKV:
        def __init__(self, db):
            self.calls = []

        def get(self, key, default=None):
            self.calls.append(key)
            raise RuntimeError("db down")

    orig = pb.KVStore
    pb.KVStore = _RaisingKV
    try:
        e = _engine(db=StubDb([]))
        assert e._load_state() == {}
    finally:
        pb.KVStore = orig
    # The queue does not come from state: empty state + queued rows -> rows.
    e = _engine(state={}, db=StubDb([ROW]))
    assert [r["title"] for r in e.queue()] == ["Test Song"]
    assert _snap() == before


def test_1129_now_falls_back_to_engine_state():
    before = _snap()
    orig = pb.MusicLibrary
    pb.MusicLibrary = _FakeMusicLibrary
    try:
        sp = _RaisingSpotify()
        e = _engine(state={"position": 0, "playing": True, "paused": False},
                    spotify=sp)
        out = e.now()
        assert sp.calls == ["now_playing"]
        assert out["playing"] is True and out["time_pos"] is None
        assert out["spotify"] == {}
    finally:
        pb.MusicLibrary = orig
    assert _snap() == before


def test_1314_stale_socket_unlink_failure_keeps_verdict():
    before = _snap()
    with tempfile.TemporaryDirectory() as td:
        sock = os.path.join(td, "mpv.sock")
        open(sock, "w").close()
        e = _engine(state={"mpv_sock": sock, "mpv_map": [0]})
        e._mpv_get_props = lambda props, timeout=1.0, sock_path=None: {
            p: None for p in props}
        orig_unlink = os.unlink
        calls = []

        def _raising_unlink(path):
            calls.append(path)
            raise OSError("unlink denied")

        os.unlink = _raising_unlink
        try:
            assert e._mpv_alive() is False
        finally:
            os.unlink = orig_unlink
        assert calls == [sock] and "mpv_sock" not in e._state
    assert _snap() == before


def test_1344_spawn_tolerates_missing_stale_socket():
    before = _snap()
    import subprocess as _sp

    class _FakeSock:
        def close(self):
            pass

    orig_popen = _sp.Popen
    popen_calls = []
    orig_unlink = os.unlink
    unlink_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return object()

    def _raising_unlink(path):
        unlink_calls.append(path)
        raise FileNotFoundError("no stale socket")

    _sp.Popen = _fake_popen
    os.unlink = _raising_unlink
    try:
        e = _engine(state={"volume": 80})
        e._mpv_sock_path = lambda: "/tmp/t32-nope.sock"
        e._mpv_connect = lambda path, timeout=8.0: _FakeSock()
        assert e._mpv_spawn() == (True, "/tmp/t32-nope.sock", "")
    finally:
        _sp.Popen = orig_popen
        os.unlink = orig_unlink
    assert unlink_calls == ["/tmp/t32-nope.sock"] and popen_calls
    assert _snap() == before


def test_1378_ipc_drain_timeout_is_designed_exit():
    before = _snap()

    class _DrainSock:
        def __init__(self):
            self.sent = b""
            self.timeouts = 0

        def sendall(self, data):
            self.sent += data

        def settimeout(self, t):
            pass

        def recv(self, n):
            self.timeouts += 1
            raise socket.timeout("drain done")

        def close(self):
            pass

    e = _engine()
    ds = _DrainSock()
    e._mpv_connect = lambda path, timeout=2.0: ds
    assert e._mpv_ipc("/tmp/s.sock", ["quit"]) is True
    assert b"quit" in ds.sent and ds.timeouts == 1
    assert _snap() == before


def test_1415_partial_ipc_response_never_fabricates():
    before = _snap()

    class _PropSock:
        def __init__(self, chunks):
            self.chunks = list(chunks)
            self.timeouts = 0

        def sendall(self, data):
            pass

        def settimeout(self, t):
            pass

        def recv(self, n):
            if self.chunks:
                return self.chunks.pop(0)
            self.timeouts += 1
            raise socket.timeout("mid-response")

        def close(self):
            pass

    e = _engine()
    ps = _PropSock([b'{"event":"ping"}'])
    e._mpv_connect = lambda path, timeout=1.5: ps
    assert e._mpv_get_one("/tmp/s.sock", "time-pos") is None
    assert ps.timeouts == 1
    e = _engine()
    ps = _PropSock([b'{"error":"success","data":42}\n'])
    e._mpv_connect = lambda path, timeout=1.5: ps
    assert e._mpv_get_one("/tmp/s.sock", "time-pos") == 42
    assert _snap() == before


def test_1670_stop_survives_spotify_pause_failure():
    before = _snap()
    e = _engine(state={"position": 0, "playing": True, "paused": False})
    stopped = []
    e._stop_local_audio = lambda: stopped.append(1)
    sp = _RaisingSpotify()
    e._spotify = sp
    e._is_spotify_current = lambda: True
    e._save_state = lambda: None
    assert e.stop() == {"status": "stopped"}
    assert e._state["playing"] is False and stopped == [1]
    assert "pause" in sp.calls
    assert _snap() == before


def test_1698_status_marks_spotify_unreachable():
    before = _snap()
    sp = _RaisingSpotify()
    e = _engine(state={"position": 0, "playing": True, "paused": False},
                spotify=sp)
    out = e.status()
    assert "now_playing" in sp.calls
    assert out["live"] == {"via": "spotify", "spotify": "unreachable"}
    assert out["playing"] is True
    assert _snap() == before
