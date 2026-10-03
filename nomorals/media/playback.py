"""PlaybackEngine — real music playback with full transport control.

Design: the player is a **detached mpv process** (own session) whose IPC
socket lives at a stable workspace path, so control commands work across
CLI/chat/tool calls without a resident daemon:

* **mpv** — the queue IS mpv's playlist. ``play`` syncs the durable queue
  (``media_queue`` table, migration 24) into the playlist, then jumps to
  the chosen index.  mpv auto-advances through the playlist when a track
  ends — no supervision process needed.  Transport (pause/resume/seek/
  volume/next/prev) and live status (playback-status, time-pos) go over
  the JSON IPC socket.
* **ffplay / aplay / mpg123 / sox / afplay** — basic play/stop per track
  (pid persisted; no fine transport, no auto-advance — honestly reported).
* **console** — no audio backend: the queue still works and ``play`` says
  exactly what to install (``pkg install mpv`` on Termux).

Volume, position, and the mpv socket path persist in ``kv_store``.

    from nomorals.media.playback import PlaybackEngine
    p = PlaybackEngine(context)
    p.add("workspace/tunes/song.mp3", title="Song")
    p.play()
    p.volume(60); p.seek(30); p.pause(); p.resume(); p.next()
    p.status()          # live: playback-status, time-pos, playlist-index

Registered as the ``player`` tool.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["PlaybackEngine", "Backend", "register"]

_STATE_KEY = "media.player_state"


def _which(names: tuple[str, ...]) -> str:
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return ""


@dataclass
class Backend:
    name: str          # mpv | ffplay | aplay | mpg123 | sox | afplay | console
    binary: str = ""
    ipc: bool = False
    urls: bool = False
    controls: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "binary": self.binary, "ipc": self.ipc,
                "urls": self.urls, "controls": list(self.controls)}


def detect_backend() -> Backend:
    mpv = _which(("mpv",))
    if mpv:
        return Backend("mpv", mpv, ipc=True, urls=True,
                       controls=("play", "pause", "resume", "stop", "seek",
                                 "volume", "next", "prev", "status"))
    ffplay = _which(("ffplay",))
    if ffplay:
        return Backend("ffplay", ffplay, urls=True,
                       controls=("play", "stop", "next", "prev"))
    aplay = _which(("aplay",))
    if aplay:
        return Backend("aplay", aplay, controls=("play", "stop", "next",
                                                 "prev"))
    mpg123 = _which(("mpg123",))
    if mpg123:
        return Backend("mpg123", mpg123, controls=("play", "stop", "next",
                                                   "prev"))
    sox = _which(("play",))
    if sox:
        return Backend("sox", sox, controls=("play", "stop", "next", "prev"))
    afplay = _which(("afplay",))
    if afplay:
        return Backend("afplay", afplay, controls=("play", "stop", "next",
                                                   "prev"))
    return Backend("console")


class PlaybackEngine:
    """Durable queue + detached-player transport."""

    role = "player"

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        if self.db is None:
            raise RuntimeError("PlaybackEngine needs a context with a database")
        self.backend = detect_backend()
        self._state = self._load_state()
        self._state.setdefault("volume", 80)
        self._sock_dir = ""

    # ── persistent state ──────────────────────────────────────────────────
    def _load_state(self) -> dict[str, Any]:
        try:
            row = self.db.query_one("SELECT value FROM kv_store WHERE key=?",
                                    (_STATE_KEY,))
            if row:
                return json.loads(row["value"])
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _save_state(self) -> None:
        try:
            self.db.execute(
                "INSERT INTO kv_store (key, value, kind, updated_at) "
                "VALUES (?,?, 'json', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (_STATE_KEY, json.dumps(self._state), time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.debug("player state save failed: %s", exc)

    def _resolve_position(self) -> int:
        q = self.queue()
        if not q:
            return -1
        pos = int(self._state.get("position", 0))
        return max(0, min(pos, len(q) - 1))

    # ── queue (durable) ───────────────────────────────────────────────────
    def add(self, *targets: str, title: str = "") -> dict[str, Any]:
        """Add file(s)/URL(s) to the queue.  Files may be workspace-relative."""
        added = []
        for t in targets:
            t = (t or "").strip()
            if not t:
                continue
            kind = "url" if t.lower().startswith(("http://", "https://")) \
                else "file"
            if kind == "file":
                from ..tools.filesystem import safe_path

                path = str(safe_path(self.context, t, must_exist=True))
            else:
                path = t
            self.db.execute(
                "INSERT INTO media_queue (id, position, path, title, kind, "
                "added_at) VALUES (?,?,?,?,?,?)",
                (f"mq-{time.time_ns()}-{uuid.uuid4().hex[:8]}",
                 time.time() + len(added), path,
                 title or os.path.basename(t), kind, time.time()))
            added.append({"path": path, "kind": kind,
                          "title": title or os.path.basename(t)})
        return {"added": added, "queue": len(self.queue())}

    def queue(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM media_queue ORDER BY position, added_at")
        return [{"index": i, "path": r["path"], "title": r["title"],
                 "kind": r["kind"]} for i, r in enumerate(rows)]

    def remove(self, index: int) -> dict[str, Any]:
        rows = self.db.query("SELECT * FROM media_queue ORDER BY position, "
                             "added_at")
        if not (0 <= int(index) < len(rows)):
            raise ToolError(f"no queue item {index}")
        self.db.execute("DELETE FROM media_queue WHERE id=?",
                        (rows[int(index)]["id"],))
        return {"removed": rows[int(index)]["path"],
                "queue": len(self.queue())}

    def clear(self) -> dict[str, Any]:
        self.db.execute("DELETE FROM media_queue")
        self._state["position"] = 0
        self._save_state()
        return {"cleared": True, "queue": 0}

    # ── transport ─────────────────────────────────────────────────────────
    def play(self, index: int | None = None) -> dict[str, Any]:
        q = self.queue()
        if not q:
            raise ToolError("queue is empty — add something first")
        if index is not None:
            if not (0 <= int(index) < len(q)):
                raise ToolError(f"no queue item {index}")
            self._state["position"] = int(index)
        else:
            self._state["position"] = self._resolve_position()
        self._save_state()
        return self._start_at(self._state["position"])

    def _start_at(self, pos: int) -> dict[str, Any]:
        q = self.queue()
        item = q[pos]
        if self.backend.name == "console":
            self._state["playing"] = False
            self._save_state()
            return {
                "status": "no-backend",
                "backend": "console",
                "current": item,
                "hint": ("no audio backend found — install mpv "
                         "(Termux: pkg install mpv) or ffmpeg for ffplay; "
                         "the queue is saved"),
            }
        if self.backend.name == "mpv":
            ok, detail = self._mpv_play_at(pos)
        else:
            ok, detail = self._simple_start(item)
        self._state["playing"] = ok
        self._state["current"] = item["path"]
        self._save_state()
        if not ok:
            return {"status": "error", "error": detail,
                    "backend": self.backend.name}
        return {"status": "playing", "backend": self.backend.name,
                "current": item["title"], "path": item["path"]}

    # ── mpv (playlist = queue, IPC = transport) ───────────────────────────
    def _mpv_sock_path(self) -> str:
        from ..tools.filesystem import safe_path

        d = safe_path(self.context, "player")
        d.mkdir(parents=True, exist_ok=True)
        return str(d / "ipc.sock")

    def _mpv_alive(self) -> bool:
        path = str(self._state.get("mpv_sock", ""))
        if not path or not os.path.exists(path):
            return False
        props = self._mpv_get_props([
            "playback-status", "playlist-index", "time-pos", "volume"],
            timeout=1.0, sock_path=path)
        return props is not None

    def _mpv_connect(self, sock_path: str,
                     timeout: float = 8.0) -> socket.socket | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(sock_path)
                s.settimeout(None)
                return s
            except (ConnectionRefusedError, FileNotFoundError,
                    socket.timeout, OSError):
                time.sleep(0.15)
        return None

    def _mpv_spawn(self) -> tuple[bool, str, str]:
        """Spawn a detached idle mpv with IPC; → (ok, sock_path, error)."""
        sock_path = self._mpv_sock_path()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:  # noqa: E103 - stale socket may not exist
            pass
        cmd = [self.backend.binary, "--idle=yes", "--really-quiet",
               "--no-terminal", f"--input-ipc-server={sock_path}",
               f"--volume={int(self._state.get('volume', 80))}"]
        try:
            subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except OSError as exc:
            return False, sock_path, f"mpv failed to start: {exc}"
        sock = self._mpv_connect(sock_path)
        if sock is None:
            return False, sock_path, "mpv started but IPC socket never came up"
        sock.close()
        return True, sock_path, ""

    def _mpv_ipc(self, sock_path: str, command: list[Any],
                 timeout: float = 2.0) -> bool:
        s = self._mpv_connect(sock_path, timeout=timeout)
        if s is None:
            return False
        try:
            s.sendall((json.dumps({"command": command}) + "\n").encode())
            s.settimeout(0.4)
            try:
                while s.recv(4096):
                    pass
            except (socket.timeout, OSError):  # noqa: E103 - timeout ends the drain loop by design
                pass
            return True
        except OSError:
            return False
        finally:
            s.close()

    def _mpv_get_props(self, props: list[str], timeout: float = 2.0,
                       sock_path: str | None = None
                       ) -> dict[str, Any] | None:
        """get_property for each prop; → {prop: value} or None if dead."""
        path = sock_path or str(self._state.get("mpv_sock", ""))
        if not path:
            return None
        # one property per short connection: exact pairing, no ambiguity
        out: dict[str, Any] = {}
        for p in props:
            out[p] = self._mpv_get_one(path, p, timeout=timeout)
        return out

    def _mpv_get_one(self, sock_path: str, prop: str,
                     timeout: float = 1.5) -> Any:
        s = self._mpv_connect(sock_path, timeout=timeout)
        if s is None:
            return None
        try:
            s.sendall((json.dumps({"command": ["get_property", prop]})
                       + "\n").encode())
            s.settimeout(1.0)
            buf = b""
            try:
                while b'"error"' not in buf:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            except socket.timeout:  # noqa: E103 - partial response is still usable
                pass
            for line in buf.split(b"\n"):
                try:
                    ev = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if ev.get("error") == "success" and "data" in ev:
                    return ev["data"]
            return None
        finally:
            s.close()

    def _mpv_play_at(self, pos: int) -> tuple[bool, str]:
        q = self.queue()
        if not self._mpv_alive():
            ok, sock_path, err = self._mpv_spawn()
            if not ok:
                return False, err
            self._state["mpv_sock"] = sock_path
            self._save_state()
        else:
            sock_path = str(self._state.get("mpv_sock", ""))
        # sync the durable queue into the playlist, then jump to pos
        for i, item in enumerate(q):
            how = "replace" if i == 0 else "append"
            if not self._mpv_ipc(sock_path, ["loadfile", item["path"], how]):
                return False, "lost the mpv IPC connection"
        if not self._mpv_ipc(sock_path, ["set_property", "playlist-pos", pos]):
            return False, "lost the mpv IPC connection"
        return True, ""

    def _mpv_transport(self, command: list[Any]) -> bool:
        if self.backend.name != "mpv":
            return False
        sock_path = str(self._state.get("mpv_sock", ""))
        if not sock_path or not self._mpv_alive():
            return False
        return self._mpv_ipc(sock_path, command)

    # ── non-mpv backends ──────────────────────────────────────────────────
    def _kill_simple(self) -> None:
        """Kill the simple-backend player process, if one is tracked.

        Called before starting a new track so the old one never keeps
        playing underneath (and its pid is never orphaned from state).
        """
        pid = int(self._state.get("player_pid", 0) or 0)
        if pid:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):  # noqa: E103 - process already gone
                pass
            self._state.pop("player_pid", None)

    def _simple_start(self, item: dict[str, str]) -> tuple[bool, str]:
        if item["kind"] == "url" and not self.backend.urls:
            return False, (f"{self.backend.name} can't stream URLs — "
                           "install mpv or ffmpeg for that")
        self._kill_simple()  # never stack a new track over a live one
        if self.backend.name == "aplay":
            cmd = [self.backend.binary, "-q", item["path"]]
        elif self.backend.name == "mpg123":
            cmd = [self.backend.binary, item["path"]]
        elif self.backend.name == "sox":
            cmd = ["play", "-q", item["path"]]
        elif self.backend.name == "afplay":
            cmd = [self.backend.binary, item["path"]]
        else:  # ffplay
            cmd = [self.backend.binary, "-nodisp", "-autoexit", "-loglevel",
                   "quiet", item["path"]]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self._state["player_pid"] = proc.pid
            return True, ""
        except OSError as exc:
            return False, f"backend failed to start: {exc}"

    def _simple_alive(self) -> bool:
        pid = int(self._state.get("player_pid", 0) or 0)
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False

    # ── control surface ───────────────────────────────────────────────────
    def pause(self) -> dict[str, Any]:
        if self._mpv_transport(["set_property", "pause", True]):
            self._state["paused"] = True
            self._save_state()
            return {"status": "paused"}
        return self._no_transport("pause")

    def resume(self) -> dict[str, Any]:
        if self._mpv_transport(["set_property", "pause", False]):
            self._state["paused"] = False
            self._save_state()
            return {"status": "resumed"}
        return self._no_transport("resume")

    def seek(self, seconds: float) -> dict[str, Any]:
        if self._mpv_transport(["seek", float(seconds), "absolute"]):
            self._save_state()
            return {"status": "seeked", "seconds": float(seconds)}
        return self._no_transport("seek")

    def volume(self, level: int | float) -> dict[str, Any]:
        level = max(0, min(100, int(level)))
        self._state["volume"] = level
        self._save_state()
        self._mpv_transport(["set_property", "volume", level])
        return {"status": "volume", "level": level}

    def next(self) -> dict[str, Any]:
        q = self.queue()
        if not q:
            raise ToolError("queue is empty")
        self._state["position"] = (self._state.get("position", 0) + 1) \
            % len(q)
        self._save_state()
        if self.backend.name == "mpv" and self._mpv_alive():
            self._mpv_transport(["playlist-next"])
            self._state["playing"] = True
            self._state["paused"] = False
            self._save_state()
            return {"status": "next",
                    "current": q[self._state["position"]]["title"]}
        return self._start_at(self._state["position"])

    def prev(self) -> dict[str, Any]:
        q = self.queue()
        if not q:
            raise ToolError("queue is empty")
        self._state["position"] = (self._state.get("position", 0) - 1) \
            % len(q)
        self._save_state()
        if self.backend.name == "mpv":
            if self._mpv_alive():
                self._mpv_transport(["playlist-prev"])
                self._state["playing"] = True
                self._state["paused"] = False
                self._save_state()
                return {"status": "prev",
                        "current": q[self._state["position"]]["title"]}
        return self._start_at(self._state["position"])

    def stop(self) -> dict[str, Any]:
        if self.backend.name == "mpv":
            sock_path = str(self._state.get("mpv_sock", ""))
            if sock_path and os.path.exists(sock_path):
                # ask mpv politely to quit; it kills itself
                self._mpv_ipc(sock_path, ["quit"])
                for _ in range(20):
                    if not os.path.exists(sock_path):
                        break
                    time.sleep(0.1)
            self._state.pop("mpv_sock", None)
        self._kill_simple()
        self._state["playing"] = False
        self._state["paused"] = False
        self._save_state()
        return {"status": "stopped"}

    def status(self) -> dict[str, Any]:
        q = self.queue()
        pos = self._resolve_position()
        live: dict[str, Any] = {}
        playing = False
        paused = False
        if self.backend.name == "mpv":
            if self._mpv_alive():
                props = self._mpv_get_props(
                    ["playback-status", "playlist-pos", "time-pos",
                     "volume"], timeout=1.0) or {}
                live = {"mpv": True, **props}
                playing = props.get("playback-status") == "playing"
                paused = props.get("playback-status") == "paused"
        elif self.backend.name != "console":
            alive = self._simple_alive()
            live = {"alive": alive}
            playing = bool(self._state.get("playing")) and alive
        return {
            "backend": self.backend.describe(),
            "playing": playing,
            "paused": paused,
            "live": live,
            "current": q[pos]["title"] if q else "",
            "path": q[pos]["path"] if q else "",
            "position": pos,
            "queue": len(q),
            "volume": int(self._state.get("volume", 80)),
        }

    def _no_transport(self, wanted: str) -> dict[str, Any]:
        return {"status": "unsupported",
                "error": (f"{wanted} needs the mpv backend "
                          f"(have: {self.backend.name})")}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "player",
        description=(
            "Music playback: queue files/URLs and control the player. "
            "action=add (target[, target2…] via targets, title) | play "
            "(index, -1 = current position) | pause | resume | stop | "
            "seek (seconds) | volume (0-100) | next | prev | queue | remove "
            "(index) | clear | status. mpv gives full transport + "
            "auto-advance; the queue persists across restarts."
        ),
        capability=Capability.FS_READ,
    )
    def player(action: str = "status", target: str = "", targets: str = "",
               title: str = "", index: int = -1, seconds: float = 0.0,
               level: int = 80) -> dict[str, Any]:
        p = PlaybackEngine(context)
        if action == "add":
            items = [t for t in (target, *filter(None, targets.split("|")))
                     if t.strip()]
            if not items:
                raise ToolError("player add needs target(s)")
            return p.add(*items, title=title)
        if action == "play":
            # index=-1 (the default) means "current position"; an explicit
            # 0 must play queue item 0, so the sentinel is < 0, not falsy
            return p.play(int(index) if int(index) >= 0 else None)
        if action == "pause":
            return p.pause()
        if action == "resume":
            return p.resume()
        if action == "stop":
            return p.stop()
        if action == "seek":
            return p.seek(seconds)
        if action == "volume":
            return p.volume(level)
        if action == "next":
            return p.next()
        if action == "prev":
            return p.prev()
        if action == "queue":
            return {"queue": p.queue()}
        if action == "remove":
            return p.remove(index)
        if action == "clear":
            return p.clear()
        if action in ("status", ""):
            return p.status()
        raise ToolError(f"unknown player action {action!r}")
