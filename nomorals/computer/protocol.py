"""The ONE driver protocol, spoken to both desktop backends.

Devon's desktop bridge (bridge/desktopbridge — her own Go binary:
raw X11 on Linux, raw Win32 on Windows) serves JSON lines over its
stdio:

    request   {"id": <int>, "method": <str>, "params": {...}}
    response  {"id": <int>, "ok": true,  "result": {...}}
              {"id": <int>, "ok": false, "error": "..."}

Methods: probe, ping, screenshot, windows, pointer, mouse_move,
mouse_click, mouse_drag, mouse_scroll, key, type, focus, launch,
kill, close_window, shutdown. This client is the only place the
Python organ speaks that protocol: thread-safe request ids, a
reader thread demultiplexing responses, honest death detection
(a bridge that dies mid-call fails every pending call with
DriverDied — never a hang, never a fake answer).
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

from . import failures

#: Protocol version implemented by the bridge and expected here.
PROTOCOL_VERSION = 1

#: Methods the organ may call. Anything else is a programming error
#: in the organ, refused before it reaches the wire.
METHODS = frozenset({
    "probe", "ping", "screenshot", "windows", "pointer",
    "mouse_move", "mouse_click", "mouse_drag", "mouse_scroll",
    "key", "type", "focus", "launch", "kill", "close_window",
    "shutdown",
})


class DriverError(Exception):
    """The bridge answered with an error, or the call was malformed."""


class DriverDied(DriverError):
    """The bridge process exited or its stdio broke mid-flight."""


class DriverClient:
    """One bridge process, spoken to over its stdio pipes."""

    def __init__(self, command: list[str], env: dict[str, str] | None = None,
                 name: str = "desktop-bridge") -> None:
        self.command = [str(part) for part in command]
        self.env = env
        self.name = name
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._dead: BaseException | None = None
        self._reader: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> DriverClient:
        if self._proc is not None:
            return self
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=self.env)
        except OSError as exc:
            raise DriverDied(
                f"cannot start {self.name} ({self.command[0]}): {exc}") from exc
        self._reader = threading.Thread(
            target=self._read_loop, name=f"{self.name}-reader", daemon=True)
        self._reader.start()
        return self

    @property
    def alive(self) -> bool:
        return (self._proc is not None and self._dead is None
                and self._proc.poll() is None)

    def close(self) -> None:
        """Best-effort shutdown: polite request, then terminate."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(json.dumps(
                        {"id": 0, "method": "shutdown",
                         "params": {}}).encode() + b"\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass  # already gone; terminate path below agrees
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        except Exception as exc:  # noqa: BLE001 — close never raises
            failures.record_failure("protocol close", exc)
        self._fail_pending(DriverDied(f"{self.name} closed"))

    # -- the wire ---------------------------------------------------------

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue  # a stray line is not a response; keep reading
                rid = msg.get("id")
                if not isinstance(rid, int):
                    continue
                with self._lock:
                    slot = self._pending.pop(rid, None)
                if slot is not None:
                    event, box = slot
                    box["msg"] = msg
                    event.set()
        except Exception as exc:  # noqa: BLE001 — EOF lands here too
            self._fail_pending(DriverDied(f"{self.name} output broke: {exc}"))
            return
        self._fail_pending(DriverDied(f"{self.name} exited"))

    def _fail_pending(self, err: DriverError) -> None:
        if self._dead is None:
            self._dead = err
        with self._lock:
            pending, self._pending = self._pending, {}
        for event, box in pending.values():
            box["error"] = err
            event.set()

    def call(self, method: str, params: dict[str, Any] | None = None,
             timeout: float = 30.0) -> dict[str, Any]:
        """One request → its result dict. Raises DriverError/DriverDied."""
        if method not in METHODS:
            raise DriverError(f"unknown driver method {method!r}")
        if self._proc is None:
            self.start()
        if self._dead is not None:
            raise DriverDied(str(self._dead))
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            event = threading.Event()
            box: dict[str, Any] = {}
            self._pending[rid] = (event, box)
        payload = json.dumps(
            {"id": rid, "method": method, "params": params or {}}).encode()
        try:
            proc.stdin.write(payload + b"\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._lock:
                self._pending.pop(rid, None)
            raise DriverDied(f"{self.name} input broke: {exc}") from exc
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise DriverError(
                f"{self.name} did not answer {method} within {timeout:.0f}s")
        if "error" in box:
            raise box["error"]
        msg = box.get("msg") or {}
        if not msg.get("ok"):
            raise DriverError(
                f"{method}: {msg.get('error') or 'driver refused'}")
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    # -- typed conveniences (thin; the organ thinks in these) -----------

    def probe(self) -> dict[str, Any]:
        return self.call("probe", {}, timeout=15.0)

    def ping(self) -> bool:
        return bool(self.call("ping", {}, timeout=10.0).get("pong"))

    def screenshot(self) -> dict[str, Any]:
        return self.call("screenshot", {}, timeout=60.0)

    def windows(self) -> dict[str, Any]:
        return self.call("windows", {}, timeout=15.0)
