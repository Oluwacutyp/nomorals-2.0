"""Run configs and process management for scaffolded projects.

:func:`run_config` detects how a project starts (``run.py`` / ``app.py`` /
``main.py`` / ``package.json`` scripts) and returns the exact command plus
environment.  :func:`serve` spawns an HTTP project as a subprocess,
waits for TCP readiness, then runs a Kubernetes-style **startup probe**
(HTTP GET against the detected health path, with initial delay / period /
failure threshold), and returns a :class:`ServeHandle` with ``stop()``,
``restart()``, PM2-style restart policies, file ``watch`` mode, and log
access.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "RunConfig", "ServeHandle", "ServeError",
    "HttpProbe", "TcpProbe", "ProbeResult",
    "run_config", "serve", "find_free_port", "stop_all",
]

#: Entrypoint filenames probed in order.
_ENTRYPOINTS = ("run.py", "app.py", "main.py", "server.py", "bot.py")

#: Restart policies for serve(): never | on unexpected exit | always.
RESTART_POLICIES = ("no", "on-failure", "always")

#: Health-path markers sniffed from entrypoint source, in priority order.
_HEALTH_PATH_MARKERS = ("/api/health", "/healthz", "/health", "/readyz",
                        "/ready", "/api/ready")


class ServeError(ToolError):
    """Serving failed.  ``stderr`` carries the child's captured output."""

    def __init__(self, message: str, *, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


@dataclass
class HttpProbe:
    """A Kubernetes-style HTTP startup/liveness probe.

    ``statuses`` accepts the k8s convention: any 2xx–3xx is healthy by
    default.  ``body_contains`` asserts on response content (Docker
    HEALTHCHECK ``grep`` semantics).
    """

    path: str = "/"
    statuses: tuple[int, ...] = ()
    body_contains: str = ""
    timeout: float = 2.0
    #: seconds to wait before the first attempt (initialDelaySeconds)
    initial_delay: float = 0.0
    #: seconds between attempts (periodSeconds)
    period: float = 1.0
    #: attempts before the probe is declared failed (failureThreshold)
    failure_threshold: int = 6

    def healthy_status(self, status: int) -> bool:
        if self.statuses:
            return status in self.statuses
        return 200 <= status < 400


@dataclass
class TcpProbe:
    """A Kubernetes-style TCP socket probe (port accepts connections)."""

    timeout: float = 2.0
    initial_delay: float = 0.0
    period: float = 1.0
    failure_threshold: int = 6


@dataclass
class ProbeResult:
    """Outcome of running a probe against a live server."""

    ok: bool
    detail: str = ""
    attempts: int = 0
    status: int | None = None
    elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "detail": self.detail,
                "attempts": self.attempts, "status": self.status,
                "elapsed": round(self.elapsed, 3)}


@dataclass
class RunConfig:
    """How to start a project."""

    project_dir: Path
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    #: "http" (serves TCP), "cli" (one-shot command), "console" (interactive)
    kind: str = "http"
    entrypoint: str = ""
    #: Detected HTTP health path ("/" when nothing was sniffed).
    health_path: str = "/"
    #: Default startup probe for this project (path pre-filled).
    probe: HttpProbe | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": str(self.project_dir),
            "command": list(self.command),
            "env": dict(self.env),
            "kind": self.kind,
            "entrypoint": self.entrypoint,
            "health_path": self.health_path,
            "probe": {"path": self.probe.path,
                      "timeout": self.probe.timeout}
            if self.probe else None,
        }


def _read_manifest(project_dir: Path) -> dict[str, Any]:
    manifest = project_dir / ".builders.json"
    if manifest.is_file():
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            _log.debug("ignoring corrupt builders manifest %s: %s", manifest, e)
            pass
    return {}


def _detect_health_path(entry: Path) -> str:
    """Sniff the entrypoint source for a health-path marker."""
    try:
        text = entry.read_text(encoding="utf-8")
    except OSError:
        return "/"
    for marker in _HEALTH_PATH_MARKERS:
        if marker in text:
            return marker
    return "/"


def _sniff_http_kind(entry: Path) -> bool:
    try:
        text = entry.read_text(encoding="utf-8")
    except OSError:
        return False
    markers = ("ThreadingHTTPServer", "http.server", "Flask(", "FastAPI(",
               "make_server", "serve_forever", "app.run", "uvicorn")
    return any(m in text for m in markers)


def run_config(project_dir: str | Path) -> RunConfig:
    """Detect a project's entrypoint and return its run configuration.

    Prefers the ``.builders.json`` manifest written by :func:`scaffold`
    (which records the template kind), then falls back to filename
    probing and content sniffing.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise ToolError(f"not a directory: {project_dir}")

    manifest = _read_manifest(project_dir)
    kind_hint = manifest.get("kind", "")

    entry = ""
    for candidate in _ENTRYPOINTS:
        if (project_dir / candidate).is_file():
            entry = candidate
            break

    package_json = project_dir / "package.json"
    if not entry and package_json.is_file():
        try:
            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
        except (json.JSONDecodeError, OSError):
            scripts = {}
        start = scripts.get("start", "")
        if start:
            parts = shlex.split(start)
            return RunConfig(project_dir=project_dir, command=parts,
                             kind="http", entrypoint="package.json#scripts.start")

    if not entry:
        raise ToolError(
            f"no entrypoint found in {project_dir} "
            f"(looked for {', '.join(_ENTRYPOINTS)} and package.json scripts.start)"
        )

    command = [sys.executable, entry]
    if kind_hint == "webapp" or _sniff_http_kind(project_dir / entry):
        kind = "http"
    elif kind_hint == "cli_tool":
        kind = "cli"
    elif kind_hint == "bot":
        kind = "console"
    else:
        kind = "console"
    env = {"PORT": os.environ.get("PORT", "8000")} if kind == "http" else {}
    health_path = _detect_health_path(project_dir / entry) if kind == "http" else "/"
    probe = HttpProbe(path=health_path) if kind == "http" else None
    return RunConfig(project_dir=project_dir, command=command, env=env,
                     kind=kind, entrypoint=entry,
                     health_path=health_path, probe=probe)


def find_free_port() -> int:
    """Return a currently-free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tcp_ready(host: str, port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


class _Drainer(threading.Thread):
    """Drain a pipe into a list so a chatty child can never block on it."""

    def __init__(self, pipe: Any) -> None:
        super().__init__(daemon=True)
        self._pipe = pipe
        self.chunks: list[str] = []

    def run(self) -> None:
        try:
            for chunk in iter(self._pipe.readline, ""):
                self.chunks.append(chunk)
        except (OSError, ValueError) as e:
            _log.debug("stderr drain ended: %s", e)
            pass
        finally:
            try:
                self._pipe.close()
            except OSError:  # noqa: E103 - best-effort close in finally; pipe may already be dead
                pass

    def text(self) -> str:
        return "".join(self.chunks)


#: Skip-dirs for the watch-mode mtime scan.
_WATCH_SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules",
               ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
               ".hg", ".svn"}

#: Live handles, for stop_all().  Weak refs: closing a handle drops it.
_HANDLES: weakref.WeakSet = weakref.WeakSet()


def stop_all(timeout: float = 5.0) -> dict[str, Any]:
    """Stop every live :class:`ServeHandle` started by :func:`serve`."""
    stopped, errors = 0, []
    for handle in list(_HANDLES):
        try:
            handle.stop(timeout=timeout)
            stopped += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    return {"stopped": stopped, "errors": errors}


def run_probe(probe: HttpProbe | TcpProbe, host: str, port: int,
              path: str = "") -> ProbeResult:
    """Run a k8s-style probe: initial delay, then attempts until the
    failure threshold.  Returns a :class:`ProbeResult` (never raises)."""
    started = time.monotonic()
    if probe.initial_delay > 0:
        time.sleep(probe.initial_delay)
    attempts = 0
    last = "no attempts made"
    target_path = path or (probe.path if isinstance(probe, HttpProbe) else "")
    while attempts < probe.failure_threshold:
        attempts += 1
        if isinstance(probe, TcpProbe):
            try:
                with socket.create_connection((host, port),
                                              timeout=probe.timeout):
                    return ProbeResult(True, "tcp connect ok",
                                       attempts, None,
                                       time.monotonic() - started)
            except OSError as exc:
                last = f"tcp connect failed: {exc}"
        else:
            url = f"http://{host}:{port}{target_path or '/'}"
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req,
                                            timeout=probe.timeout) as resp:
                    status = resp.status
                    body = resp.read(65536).decode("utf-8", "replace")
            except Exception as exc:  # noqa: BLE001 -- probe detail, not a crash
                last = f"attempt {attempts}: {exc}"
            else:
                if not probe.healthy_status(status):
                    last = (f"attempt {attempts}: status {status} not in "
                            f"healthy range")
                elif probe.body_contains and \
                        probe.body_contains not in body:
                    last = (f"attempt {attempts}: body missing "
                            f"{probe.body_contains!r}")
                else:
                    return ProbeResult(True,
                                       f"GET {target_path or '/'} -> {status}",
                                       attempts, status,
                                       time.monotonic() - started)
        if attempts < probe.failure_threshold:
            time.sleep(probe.period)
    return ProbeResult(False,
                       f"probe failed after {attempts} attempts: {last}",
                       attempts, None, time.monotonic() - started)


def _snapshot_mtimes(root: Path) -> dict[str, float]:
    """mtime snapshot of a project tree (watch mode)."""
    snap: dict[str, float] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in _WATCH_SKIP for part in rel.parts):
            continue
        try:
            snap[rel.as_posix()] = path.stat().st_mtime
        except OSError:
            continue
    return snap


class ServeHandle:
    """A running project server.  Use as a context manager or call :meth:`stop`.

    Extras over a bare subprocess: :meth:`restart` (PM2-style),
    :meth:`logs` (tail the captured stderr), :attr:`uptime`,
    :attr:`restarts`, restart policies (``on-failure`` / ``always``),
    and ``watch`` mode (restart when project files change).
    """

    def __init__(self, proc: subprocess.Popen, url: str,
                 config: RunConfig, stderr_drainer: _Drainer,
                 spawn: Callable[[], tuple[subprocess.Popen, _Drainer]],
                 restart_policy: str = "no", max_restarts: int = 3,
                 watch: bool = False) -> None:
        self._proc = proc
        self.url = url
        self.config = config
        self._stderr = stderr_drainer
        self._spawn = spawn
        self.restart_policy = restart_policy
        self.max_restarts = max_restarts
        self.watch = watch
        self._lock = threading.Lock()
        self._stopped = False
        self._restarts = 0
        self._last_exit: int | None = None
        self._last_restart_reason = ""
        self._started_at = time.monotonic()
        self._watch_snapshot = _snapshot_mtimes(config.project_dir) \
            if watch else {}
        self._supervisor: threading.Thread | None = None
        if restart_policy != "no" or watch:
            self._supervisor = threading.Thread(
                target=self._supervise, daemon=True,
                name=f"serve-supervisor-{config.project_dir.name}")
            self._supervisor.start()
        _HANDLES.add(self)

    # ── introspection ──────────────────────────────────────────────
    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def project_dir(self) -> Path:
        return self.config.project_dir

    @property
    def restarts(self) -> int:
        return self._restarts

    @property
    def last_exit(self) -> int | None:
        return self._last_exit

    @property
    def last_restart_reason(self) -> str:
        return self._last_restart_reason

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._started_at

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def exit_code(self) -> int | None:
        return self._proc.poll()

    def stderr_text(self) -> str:
        return self._stderr.text()

    def logs(self, n: int = 50) -> str:
        """Last ``n`` lines of the child's captured stderr."""
        lines = self._stderr.text().splitlines()
        return "\n".join(lines[-n:]) or "(no output yet)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "pid": self.pid,
            "project_dir": str(self.project_dir),
            "running": self.is_running(), "exit_code": self.exit_code(),
            "uptime": round(self.uptime, 1), "restarts": self._restarts,
            "last_exit": self._last_exit,
            "last_restart_reason": self._last_restart_reason,
            "restart_policy": self.restart_policy, "watch": self.watch,
            "command": list(self.config.command),
        }

    # ── control ────────────────────────────────────────────────────
    def restart(self, reason: str = "manual") -> None:
        """Stop the child (SIGTERM → SIGKILL) and spawn a fresh one."""
        with self._lock:
            self._kill_locked()
            proc, drainer = self._spawn()
            self._proc = proc
            self._stderr = drainer
            self._restarts += 1
            self._last_restart_reason = reason
            self._started_at = time.monotonic()
            self._watch_snapshot = _snapshot_mtimes(self.config.project_dir) \
                if self.watch else {}
        _log.info("serve %s restarted (%s, pid %d)", self.config.project_dir,
                  reason, proc.pid)

    def _kill_locked(self) -> None:
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _log.warning("serve %s would not die",
                                 self.config.project_dir)
        try:
            self._stderr.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass

    def _supervise(self) -> None:
        """Daemon: restart on unexpected exit (policy) or file change (watch)."""
        backoff = 0.5
        while not self._stopped:
            time.sleep(0.5)
            if self._stopped:
                break
            with self._lock:
                code = self._proc.poll()
                changed = False
                if self.watch and code is None:
                    now = _snapshot_mtimes(self.config.project_dir)
                    changed = now != self._watch_snapshot
            if changed:
                _log.info("serve %s: file change detected, restarting",
                          self.config.project_dir)
                try:
                    self.restart(reason="watch: file changed")
                except Exception as exc:  # noqa: BLE001
                    _log.warning("watch restart failed: %s", exc)
                backoff = 0.5
                continue
            if code is not None and not self._stopped:
                self._last_exit = code
                want = (self.restart_policy == "always" or
                        (self.restart_policy == "on-failure" and code != 0))
                if want and self._restarts < self.max_restarts:
                    _log.warning("serve %s exited %d, restarting (%s)",
                                 self.config.project_dir, code,
                                 self.restart_policy)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)
                    try:
                        self.restart(
                            reason=f"policy={self.restart_policy} exit={code}")
                    except Exception as exc:  # noqa: BLE001
                        _log.warning("policy restart failed: %s", exc)
                        break
                elif want:
                    _log.error("serve %s: max_restarts=%d reached, giving up",
                               self.config.project_dir, self.max_restarts)
                    break
                else:
                    break

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the whole process group (child + grandchildren)."""
        if self._stopped:
            return
        self._stopped = True
        with self._lock:
            self._kill_locked()
        if self._supervisor is not None:
            self._supervisor.join(timeout=2.0)

    def __enter__(self) -> "ServeHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def serve(project_dir: str | Path, port: int = 0, *,
          startup_timeout: float = 10.0,
          extra_env: dict[str, str] | None = None,
          host: str = "127.0.0.1",
          probe: HttpProbe | TcpProbe | None = None,
          restart_policy: str = "no",
          max_restarts: int = 3,
          watch: bool = False) -> ServeHandle:
    """Start an HTTP project and wait until it passes its startup probe.

    ``port=0`` picks a free port.  Readiness is two-stage, Kubernetes
    style: first TCP accept (hard deadline ``startup_timeout``), then an
    HTTP probe against the project's health path (``probe`` overrides
    the auto-detected one).  Fails fast: if the child exits during
    startup, or TCP/probe never succeeds, the process group is killed
    and :class:`ServeError` is raised carrying the child's stderr and
    the probe transcript — never a bare timeout.

    ``restart_policy`` is ``"no"`` | ``"on-failure"`` | ``"always"``
    (PM2-style auto-restart, capped by ``max_restarts``);
    ``watch=True`` restarts the server when project files change.
    """
    config = run_config(project_dir)
    if config.kind != "http":
        raise ToolError(
            f"serve() needs an HTTP project, but {config.project_dir.name} "
            f"looks like kind={config.kind!r} (entrypoint {config.entrypoint})"
        )
    if restart_policy not in RESTART_POLICIES:
        raise ToolError(f"restart_policy must be one of {RESTART_POLICIES}, "
                        f"got {restart_policy!r}")
    if port == 0:
        port = find_free_port()

    env = dict(os.environ)
    env.update(config.env)
    env["PORT"] = str(port)
    if extra_env:
        env.update(extra_env)

    _log.info("serving %s on %s:%d: %s", config.project_dir, host, port,
              " ".join(config.command))

    def _spawn() -> tuple[subprocess.Popen, _Drainer]:
        try:
            proc = subprocess.Popen(
                config.command,
                cwd=str(config.project_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # own process group: stop() kills the tree
            )
        except OSError as exc:
            raise ServeError(f"could not spawn {' '.join(config.command)}: {exc}")
        drainer = _Drainer(proc.stderr)
        drainer.start()
        return proc, drainer

    proc, drainer = _spawn()

    def _fail_startup(message: str) -> ServeError:
        drainer.join(timeout=2.0)
        err = drainer.text().strip()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        return ServeError(
            message + (f"\nstderr:\n{err}" if err else " (no stderr output)"),
            stderr=err)

    # stage 1: TCP accept
    deadline = time.monotonic() + startup_timeout
    while True:
        code = proc.poll()
        if code is not None:
            raise _fail_startup(
                f"{config.entrypoint} exited during startup with code {code}")
        if _wait_tcp_ready(host, port, min(deadline, time.monotonic() + 0.2)):
            break
        if time.monotonic() >= deadline:
            raise _fail_startup(
                f"server did not accept connections on {host}:{port} "
                f"within {startup_timeout:.1f}s")

    # stage 2: HTTP startup probe (k8s semantics — TCP up ≠ healthy)
    active_probe = probe or config.probe or HttpProbe(path="/")
    probe_result = run_probe(active_probe, host, port)
    if not probe_result.ok:
        raise _fail_startup(
            f"startup probe failed: {probe_result.detail}")

    url = f"http://{host}:{port}"
    handle = ServeHandle(proc, url, config, drainer, _spawn,
                         restart_policy=restart_policy,
                         max_restarts=max_restarts, watch=watch)
    _log.info("server ready at %s (pid %d, probe: %s)", url, proc.pid,
              probe_result.detail)
    return handle
