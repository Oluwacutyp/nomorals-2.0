"""Persistent browser-service daemon (layer 4).

``nm browse`` in its default mode is restore→act→save per invocation: every
run re-creates each session and *re-navigates every tab's last URL*
(:meth:`BrowserService.restore`), i.e. a full page-load round trip per tab.
The daemon is the opt-in alternative: ``nm browse daemon start`` spawns a
background process holding one live :class:`BrowserService`; later
invocations talk to it over a Unix-domain socket, so sessions, tabs and
cookies stay warm with no restore churn. ``nm browse daemon stop`` shuts it
down and returns to the default flow.

Wire protocol: length-prefixed (4-byte big-endian) JSON frames over
``<data_dir>/daemon.sock``. Request ``{"op": ..., "params": {...}}``; reply
``{"ok": true, "result": ..., "events": [...]}`` or
``{"ok": false, "error": "..."}``. The ``events`` list carries every
``browser.*`` bus event the daemon emitted while handling the request; the
CLI re-publishes them on its own bus (see :func:`republish_events`) so the
timeline records them exactly as in the default flow.

Files (all under the browser data dir, ``~/.nomorals/browser-service``
unless ``NOMORALS_BROWSER_DIR`` is set):

* ``daemon.sock`` — the IPC socket;
* ``daemon.pid``  — the daemon's pid;
* ``daemon.lock`` — start-up exclusion lock;
* ``daemon.log``  — the daemon's stdout/stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .service import BrowserError, BrowserService, _mask_proxy

__all__ = [
    "DaemonClient",
    "DaemonControl",
    "DaemonError",
    "default_data_dir",
    "republish_events",
]

_log = get_logger(__name__)

#: Env var overriding the browser data dir (honored by the CLI, the daemon
#: control, and the daemon itself). Used by tests to stay off the real home.
DATA_DIR_ENV = "NOMORALS_BROWSER_DIR"

_SOCK_NAME = "daemon.sock"
_PID_NAME = "daemon.pid"
_LOCK_NAME = "daemon.lock"
_LOG_NAME = "daemon.log"

_FRAME_HEADER = struct.Struct(">I")
#: Refuse absurd frames early instead of buffering garbage.
_MAX_FRAME = 256 * 1024 * 1024
#: Ops that mutate persisted session state; the daemon saves after each.
_MUTATING_OPS = frozenset({"open", "close", "click", "submit"})
#: Cap on bus events buffered between client drains.
_MAX_BUFFERED_EVENTS = 10_000


class DaemonError(Exception):
    """The daemon is missing, unreachable, unresponsive, or refused an op."""


def default_data_dir() -> Path:
    """Browser data dir: ``NOMORALS_BROWSER_DIR`` or ``~/.nomorals/browser-service``."""
    override = os.environ.get(DATA_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".nomorals" / "browser-service"


# ── wire framing ─────────────────────────────────────────────────────────────


def _send_frame(sock: socket.socket, obj: Any) -> None:
    data = json.dumps(obj, default=str).encode("utf-8")
    sock.sendall(_FRAME_HEADER.pack(len(data)) + data)


def _recvall(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise DaemonError("daemon closed the connection mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(sock: socket.socket, timeout: float) -> Any:
    sock.settimeout(timeout)
    try:
        (size,) = _FRAME_HEADER.unpack(_recvall(sock, _FRAME_HEADER.size))
    except (OSError, struct.error) as exc:
        raise DaemonError(f"daemon sent a broken frame header: {exc}") from exc
    if size > _MAX_FRAME:
        raise DaemonError(f"daemon frame too large ({size} bytes) — refusing")
    try:
        raw = _recvall(sock, size)
    except OSError as exc:
        raise DaemonError(f"daemon connection broke mid-frame: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise DaemonError(f"daemon sent invalid JSON: {exc}") from exc


# ── client ───────────────────────────────────────────────────────────────────


class DaemonClient:
    """One-shot request client for a running browser daemon.

    A fresh socket is opened per call (the daemon handles one request per
    connection), so clients are cheap and never hold stale state.
    """

    def __init__(self, data_dir: str | os.PathLike[str] | None = None,
                 *, timeout: float = 120.0) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.sock_path = self.data_dir / _SOCK_NAME
        self.timeout = timeout

    def call(self, op: str, params: dict[str, Any] | None = None,
             *, timeout: float | None = None) -> tuple[Any, list[dict[str, Any]]]:
        """Send one op; return ``(result, events)``. Raises DaemonError."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout if timeout is not None else self.timeout)
            try:
                sock.connect(str(self.sock_path))
                _send_frame(sock, {"op": op, "params": params or {}})
                resp = _recv_frame(sock, timeout if timeout is not None else self.timeout)
            except OSError as exc:
                raise DaemonError(
                    f"browser daemon connection failed ({self.sock_path}): {exc}") from exc
        finally:
            sock.close()
        if not isinstance(resp, dict) or "ok" not in resp:
            raise DaemonError(f"daemon sent a malformed reply: {resp!r}"[:200])
        if not resp["ok"]:
            raise DaemonError(str(resp.get("error") or "daemon reported an error"))
        events = resp.get("events") or []
        return resp.get("result"), list(events) if isinstance(events, list) else []

    def ping(self, *, timeout: float = 5.0) -> dict[str, Any]:
        """Health check; returns the daemon's status payload."""
        result, _ = self.call("ping", {}, timeout=timeout)
        if not isinstance(result, dict):
            raise DaemonError(f"daemon ping returned garbage: {result!r}"[:200])
        return result


def republish_events(events: list[dict[str, Any]]) -> int:
    """Re-publish daemon-forwarded ``browser.*`` events on this process's bus.

    This is what keeps the timeline firing in daemon mode: the daemon emits
    into its own process bus, forwards the events with each reply, and the
    CLI re-publishes them here, where ``_attach_timeline``'s subscriber
    records them. Best-effort per event — one bad payload never breaks the
    command that produced it. Returns the number re-published.
    """
    count = 0
    for raw in events or []:
        try:
            if not isinstance(raw, dict) or not raw.get("topic"):
                continue
            global_bus.publish(Event(
                topic=str(raw["topic"]),
                data=dict(raw.get("data") or {}),
                source=str(raw.get("source") or ""),
                ts=float(raw.get("ts") or time.time()),
            ))
            count += 1
        except Exception as exc:  # noqa: BLE001 - telemetry is fail-open
            _log.debug("republish of daemon event failed: %r", exc)
    return count


# ── process helpers ──────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """True when pid names a live, non-zombie process."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            state = fh.read().rsplit(b")", 1)[-1].split()[0]
        return state not in (b"Z", b"X", b"x")
    except FileNotFoundError:
        return False
    except (OSError, IndexError) as exc:
        _log.debug("pid %d liveness probe fell back to kill(0): %r", pid, exc)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    """Serialize daemon start/stop races across CLI invocations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DaemonError(
                "another browser-daemon start/stop is in progress") from exc
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ── control ──────────────────────────────────────────────────────────────────


class DaemonControl:
    """Lifecycle management for the browser daemon: start/stop/status."""

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.sock_path = self.data_dir / _SOCK_NAME
        self.pid_path = self.data_dir / _PID_NAME
        self.lock_path = self.data_dir / _LOCK_NAME
        self.log_path = self.data_dir / _LOG_NAME

    # -- introspection -------------------------------------------------------
    def _read_pid(self) -> int | None:
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _try_ping(self, timeout: float) -> dict[str, Any] | None:
        try:
            return DaemonClient(self.data_dir, timeout=timeout).ping(timeout=timeout)
        except DaemonError as exc:
            _log.debug("daemon ping failed: %s", exc)
            return None

    def status(self) -> dict[str, Any]:
        """Honest snapshot: running / stale / unresponsive are distinct."""
        pid = self._read_pid()
        base = {"socket": str(self.sock_path), "data_dir": str(self.data_dir)}
        pong = self._try_ping(timeout=3.0)
        if pid is None:
            if pong is not None:
                # Socket alive but no pid file (e.g. pid file deleted): adopt.
                return {"running": True, "responsive": True, "pid": pong.get("pid"),
                        **base, **pong}
            return {"running": False, **base}
        if not _pid_alive(pid):
            return {"running": False, "stale_pid": pid, **base}
        if pong is None:
            return {"running": True, "responsive": False, "pid": pid, **base}
        return {"running": True, "responsive": True, "pid": pid, **base, **pong}

    def running(self) -> bool:
        """True when a daemon is up *and answering* — the routing predicate."""
        st = self.status()
        return bool(st.get("running") and st.get("responsive", True))

    def client(self, *, timeout: float = 120.0) -> DaemonClient:
        """A client for this daemon. Raises DaemonError when it isn't usable."""
        st = self.status()
        if not st.get("running"):
            raise DaemonError("browser daemon is not running "
                              "(`nm browse daemon start` to start it)")
        if not st.get("responsive", True):
            raise DaemonError(
                f"browser daemon (pid {st.get('pid')}) is not responding — "
                "`nm browse daemon stop` to reset it")
        return DaemonClient(self.data_dir, timeout=timeout)

    # -- lifecycle -----------------------------------------------------------
    def start(self, *, timeout: float = 30.0) -> dict[str, Any]:
        """Start the daemon (idempotent). Returns start/already info."""
        st = self.status()
        if st.get("running") and st.get("responsive", True):
            return {"started": False, "already": True, "pid": st.get("pid"),
                    "socket": str(self.sock_path)}
        with _exclusive_lock(self.lock_path):
            st = self.status()
            if st.get("running") and st.get("responsive", True):
                return {"started": False, "already": True, "pid": st.get("pid"),
                        "socket": str(self.sock_path)}
            replaced_pid = None
            old_pid = st.get("pid")
            if old_pid is not None and _pid_alive(old_pid):
                # Wedged predecessor still holds the socket: clear it so the
                # new daemon binds cleanly (same escalation as stop()).
                _log.info("replacing unresponsive browser daemon (pid %d)", old_pid)
                self._escalate(old_pid)
                replaced_pid = old_pid
            self._drop_dead_socket()
            proc = self._spawn()
            self.pid_path.write_text(str(proc.pid), encoding="utf-8")
        pong = self._await_ping(proc, timeout)
        info: dict[str, Any] = {"started": True, "pid": proc.pid,
                               "socket": str(self.sock_path), **pong}
        if replaced_pid is not None:
            info["replaced_pid"] = replaced_pid
        return info

    def _drop_dead_socket(self) -> None:
        """Remove a socket file nothing listens on (crashed daemon)."""
        if not self.sock_path.exists():
            return
        if self._try_ping(timeout=2.0) is not None:
            return  # someone *is* listening; leave it alone
        with contextlib.suppress(OSError):
            self.sock_path.unlink()
            _log.info("removed stale browser-daemon socket %s", self.sock_path)

    def _spawn(self) -> subprocess.Popen[bytes]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        repo_root = _repo_root()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        log_fh = open(self.log_path, "a", encoding="utf-8")  # noqa: PTH123
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "nomorals.browser.daemon",
                 "serve", "--data-dir", str(self.data_dir)],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(repo_root),
                env=env,
            )
        except OSError as exc:
            raise DaemonError(f"could not spawn browser daemon: {exc}") from exc
        finally:
            # The child dup'd the fd at fork; the parent's copy must close.
            log_fh.close()
        return proc

    def _await_ping(self, proc: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise DaemonError(
                    f"browser daemon exited during startup (code {proc.returncode}); "
                    f"see {self.log_path}")
            pong = self._try_ping(timeout=2.0)
            if pong is not None:
                return pong
            time.sleep(0.25)
        with contextlib.suppress(Exception):
            self.stop()
        raise DaemonError(
            f"browser daemon did not answer within {timeout:.0f}s; see {self.log_path}")

    def stop(self, *, timeout: float = 10.0) -> dict[str, Any]:
        """Stop the daemon; escalate to SIGTERM/SIGKILL; never leave orphans."""
        with _exclusive_lock(self.lock_path):
            st = self.status()
            pid = st.get("pid") or st.get("stale_pid")
            if not st.get("running"):
                self._remove_files()
                return {"stopped": False, "reason": "not running"}
            # Ask nicely first.
            try:
                DaemonClient(self.data_dir, timeout=5.0).call("shutdown", {})
            except DaemonError as exc:
                _log.debug("daemon shutdown op failed, escalating: %s", exc)
            if self._wait_gone(pid, timeout):
                self._remove_files()
                return {"stopped": True, "pid": pid}
            forced = self._escalate(pid)
            self._remove_files()
            return {"stopped": True, "pid": pid, "forced": forced}

    def _wait_gone(self, pid: int | None, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pid is None:
                if self._try_ping(timeout=1.0) is None:
                    return True
            elif not _pid_alive(pid):
                self._reap(pid)
                return True
            time.sleep(0.2)
        if pid is not None and not _pid_alive(pid):
            self._reap(pid)
            return True
        return False

    def _escalate(self, pid: int | None) -> bool:
        """SIGTERM, then SIGKILL. Returns True when force was needed."""
        if pid is None or not _pid_alive(pid):
            return False
        for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 3.0)):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError) as exc:
                _log.debug("kill pid %d with %s failed: %r", pid, sig, exc)
                break
            if self._wait_gone(pid, wait):
                self._reap(pid)
                return True
        if _pid_alive(pid):
            raise DaemonError(
                f"could not stop browser daemon (pid {pid}) — kill it manually")
        self._reap(pid)
        return True

    @staticmethod
    def _reap(pid: int) -> None:
        """Reap a child we may own so it never lingers as a zombie."""
        try:
            while True:
                done, _ = os.waitpid(pid, os.WNOHANG)
                if done == 0:
                    break
        except (ChildProcessError, OSError) as exc:
            # Not our child (or already reaped): nothing to wait for.
            _log.debug("reap of pid %d skipped: %r", pid, exc)

    def _remove_files(self) -> None:
        for path in (self.pid_path, self.sock_path):
            with contextlib.suppress(OSError):
                path.unlink()


# ── daemon side ──────────────────────────────────────────────────────────────


def _active_tab_or_raise(handle: Any):
    """Mirror the CLI's "no tabs open" error so both flows read the same."""
    try:
        return handle.active_tab
    except Exception as exc:  # noqa: BLE001 - BrowserError or exotic failure
        raise BrowserError("no tabs open — `nm browse open <url>` first") from exc


def _rendered_tab_or_raise(svc: BrowserService, tab_id: str):
    tab = svc.find_rendered_tab(tab_id)
    if tab is None:
        raise BrowserError(f"unknown rendered tab {tab_id!r}")
    return tab


def _get_or_open(svc: BrowserService, name: str):
    name = (name or "").strip() or "cli"
    try:
        return svc.get_session(name)
    except BrowserError:
        return svc.open_session(name)


def _handle_op(svc: BrowserService, op: str, params: dict[str, Any]) -> Any:
    """Execute one request against the live service. Mirrors the CLI verbs."""
    if op == "ping":
        sessions = svc.list_sessions()
        tabs = sum(len(svc.get_session(n)._tabs) for n in sessions)
        return {
            "pid": os.getpid(),
            "started_at": _STARTED_AT,
            "uptime_s": round(time.time() - _STARTED_AT, 1),
            "sessions": sessions,
            "tabs": tabs,
        }
    if op == "sessions":
        return {"sessions": svc.list_sessions()}

    name = (params.get("session") or "").strip() or "cli"

    if op == "open":
        url = params.get("url") or ""
        handle = _get_or_open(svc, name)
        tab = handle.open_tab(url)
        return {"tab": tab.to_dict()}
    if op == "tabs":
        return {"tabs": _get_or_open(svc, name).list_tabs()}
    if op == "read":
        kind = params.get("kind") or "text"
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        if kind == "markdown":
            return {"data": tab.markdown()}
        return {"data": tab.text()}
    if op == "links":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"links": tab.links().get("links") or []}
    if op == "shot":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": svc.screenshot(tab).to_dict()}
    if op == "download":
        url = (params.get("url") or "").strip()
        handle = _get_or_open(svc, name)
        try:
            tab = handle.active_tab
        except BrowserError:
            tab = None
        result = svc.download(tab if tab is not None else url,
                              url if tab is not None else "",
                              organize=bool(params.get("organize")))
        return {"result": result.to_dict()}
    if op == "history":
        return {"history": _get_or_open(svc, name).history()}
    if op == "close":
        svc.close_session(name)
        return {"closed": name}
    if op == "shutdown":
        return {"stopped": True}
    # -- forms & interaction --------------------------------------------------
    if op == "fill":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.fill(params.get("name") or "",
                                   params.get("value") or "")}
    if op == "select":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.select(params.get("name") or "",
                                     params.get("value") or "")}
    if op == "check":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.check(
            params.get("name") or "",
            bool(params.get("checked", True)))}
    if op == "captcha_check":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.check_captcha(
            fetch_bytes=bool(params.get("fetch_bytes")))}
    if op == "click":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.click(params.get("target") or "")}
    if op == "submit":
        uploads = params.get("uploads")
        if uploads is not None and not isinstance(uploads, dict):
            raise DaemonError("submit uploads must be a {field: path} object")
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.submit(params.get("target") or "",
                                     uploads=uploads)}
    if op == "extract":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.extract(params.get("target") or "",
                                      kind=params.get("kind") or "")}
    if op == "task":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"result": tab.task(params.get("steps"))}
    # -- cookies --------------------------------------------------------------
    if op == "cookies":
        tab = _active_tab_or_raise(_get_or_open(svc, name))
        return {"cookies": tab.cookies()}
    if op == "cookies_export":
        return {"result": svc.export_cookies(
            name, params.get("path") or "",
            format=params.get("format") or "netscape")}
    if op == "cookies_import":
        return {"result": svc.import_cookies(
            name, params.get("path") or "",
            format=params.get("format") or "netscape")}
    # -- proxies --------------------------------------------------------------
    if op == "proxy":
        action = (params.get("action") or "status").strip().lower()
        if action == "rotate":
            return {"result": svc.rotate_proxy(name)}
        if action == "set":
            return {"result": svc.set_session_proxy(
                name, params.get("proxy_url") or "")}
        if action == "clear":
            return {"result": svc.set_session_proxy(name, "")}
        if action == "status":
            return {"attached": svc.proxy_pool_attached(),
                    "proxy": _mask_proxy(svc.session_proxy(name)) or "direct"}
        raise DaemonError(
            f"unknown proxy action {action!r}: status|rotate|set|clear")
    # -- downloads ------------------------------------------------------------
    if op == "downloads":
        return {"downloads": svc.list_downloads(
            params.get("session") or name,
            category=params.get("category") or "",
            limit=int(params.get("limit") or 100))}
    if op == "wait_download":
        return {"result": svc.wait_for_download(
            params.get("download_id") or "",
            timeout=float(params.get("timeout") or 60.0))}
    # -- rendered tabs ----------------------------------------------------------
    if op == "r_open":
        tab = svc.open_rendered_tab(name, params.get("url") or "",
                                    proxy=params.get("proxy") or "")
        return {"tab": tab.to_dict(), "tab_id": tab.tab_id}
    if op == "r_tabs":
        return {"tabs": svc.list_rendered_tabs()}
    if op == "r_close":
        tab_id = params.get("tab_id") or ""
        svc.close_rendered_tab(tab_id)
        return {"closed": tab_id}
    if op == "r_shot":
        return {"result": svc.screenshot_rendered(
            params.get("tab_id") or "",
            full_page=bool(params.get("full_page"))).to_dict()}
    if op == "r_fill":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.fill(params.get("name") or "",
                                   params.get("value") or "")}
    if op == "r_click":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.click(params.get("target") or "")}
    if op == "r_submit":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.submit(params.get("target") or "")}
    if op == "r_wait":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.wait_for(
            params.get("selector") or "",
            state=params.get("state") or "visible",
            timeout=int(params.get("timeout") or 10_000))}
    if op == "r_wait_url":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.wait_for_url(
            params.get("pattern") or "",
            timeout=int(params.get("timeout") or 10_000))}
    if op == "r_wait_text":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.wait_for_text(
            params.get("text") or "",
            timeout=int(params.get("timeout") or 10_000))}
    if op == "r_select":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.select(
            params.get("name") or "",
            params.get("value") or "",
            by=params.get("by") or "auto")}
    if op == "r_check":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.check(
            params.get("name") or "",
            bool(params.get("checked", True)))}
    if op == "r_captcha":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.check_captcha(
            fetch_bytes=bool(params.get("fetch_bytes")))}
    if op == "r_extract":
        tab = _rendered_tab_or_raise(svc, params.get("tab_id") or "")
        return {"result": tab.extract(params.get("target") or "",
                                      kind=params.get("kind") or "")}
    if op == "r_download":
        return {"result": svc.rendered_tab_download(
            params.get("tab_id") or "", params.get("target") or "")}
    raise DaemonError(f"unknown daemon op {op!r}")


_STARTED_AT = time.time()


def serve(data_dir: str | os.PathLike[str]) -> int:
    """Daemon entry point: hold one live BrowserService and serve the socket."""
    data_path = Path(data_dir).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)
    sock_path = data_path / _SOCK_NAME

    svc = BrowserService(data_dir=data_path)
    try:
        restored = svc.restore()
    except BrowserError as exc:
        print(f"daemon: cannot restore sessions: {exc}", file=sys.stderr)
        return 1
    _log.info("browser daemon up (pid %d), restored %d session(s)", os.getpid(), restored)

    # Collect every browser.* event this process emits so each reply can
    # forward them to the CLI for re-publication (timeline parity).
    buffered: list[dict[str, Any]] = []

    def _collect(event: Event) -> None:
        buffered.append(event.to_dict())
        if len(buffered) > _MAX_BUFFERED_EVENTS:
            del buffered[: len(buffered) - _MAX_BUFFERED_EVENTS]

    sub_id = global_bus.subscribe("browser.*", _collect, sync=True)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            listener.bind(str(sock_path))
        except OSError as exc:
            print(f"daemon: cannot bind {sock_path}: {exc}", file=sys.stderr)
            return 1
        with contextlib.suppress(OSError):
            os.chmod(sock_path, 0o600)
        listener.listen(8)
        listener.settimeout(1.0)

        stopping = False

        def _on_signal(signum: int, _frame: Any) -> None:
            nonlocal stopping
            _log.info("browser daemon got signal %d, shutting down", signum)
            stopping = True

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

        while not stopping:
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                _log.debug("daemon accept failed: %r", exc)
                continue
            try:
                conn.settimeout(120.0)
                try:
                    request = _recv_frame(conn, 120.0)
                except DaemonError as exc:
                    _log.debug("daemon dropped a bad request: %s", exc)
                    continue
                if not isinstance(request, dict):
                    _send_frame(conn, {"ok": False, "error": "request must be an object"})
                    continue
                op = str(request.get("op") or "")
                params = request.get("params")
                if not isinstance(params, dict):
                    params = {}
                try:
                    result = _handle_op(svc, op, params)
                    reply: dict[str, Any] = {"ok": True, "result": result}
                    if op in _MUTATING_OPS:
                        svc.save()
                except BrowserError as exc:
                    reply = {"ok": False, "error": str(exc)}
                except DaemonError as exc:
                    reply = {"ok": False, "error": str(exc)}
                except Exception as exc:  # noqa: BLE001 - one bad op never kills the daemon
                    _log.exception("daemon op %r failed", op)
                    reply = {"ok": False, "error": f"internal error: {exc!r}"}
                events = list(buffered)
                buffered.clear()
                reply["events"] = events
                try:
                    _send_frame(conn, reply)
                except (OSError, DaemonError) as exc:
                    _log.debug("daemon could not send reply: %r", exc)
                if op == "shutdown":
                    stopping = True
            finally:
                conn.close()
    finally:
        with contextlib.suppress(Exception):
            global_bus.unsubscribe(sub_id)
        listener.close()
        # Persist everything: cookies are flushed by tab.close().
        with contextlib.suppress(Exception):
            for session_name in svc.list_sessions():
                try:
                    handle = svc.get_session(session_name)
                    for tab_id in list(handle._tabs):
                        with contextlib.suppress(Exception):
                            handle.close_tab(tab_id)
                except Exception as exc:  # noqa: BLE001 - shutdown must complete
                    _log.debug("daemon shutdown: closing %r failed: %r", session_name, exc)
        with contextlib.suppress(Exception):
            svc.save()
        with contextlib.suppress(OSError):
            sock_path.unlink()
        _log.info("browser daemon (pid %d) stopped", os.getpid())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nomorals.browser.daemon")
    sub = parser.add_subparsers(dest="command", required=True)
    serve_p = sub.add_parser("serve", help="run the daemon (not for interactive use)")
    serve_p.add_argument("--data-dir", default="",
                         help="browser data dir (default: NOMORALS_BROWSER_DIR or "
                              "~/.nomorals/browser-service)")
    args = parser.parse_args(argv)
    if args.command == "serve":
        data_dir = args.data_dir.strip() or default_data_dir()
        return serve(data_dir)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
