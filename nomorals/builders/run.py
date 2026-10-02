"""Run configs and process management for scaffolded projects.

:func:`run_config` detects how a project starts (``run.py`` / ``app.py`` /
``main.py`` / ``package.json`` scripts) and returns the exact command plus
environment.  :func:`serve` spawns an HTTP project as a subprocess,
waits for TCP readiness with a hard deadline (failing fast with the
captured stderr), and returns a :class:`ServeHandle` with ``stop()``.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "RunConfig", "ServeHandle", "ServeError",
    "run_config", "serve", "find_free_port",
]

#: Entrypoint filenames probed in order.
_ENTRYPOINTS = ("run.py", "app.py", "main.py", "server.py", "bot.py")


class ServeError(ToolError):
    """Serving failed.  ``stderr`` carries the child's captured output."""

    def __init__(self, message: str, *, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


@dataclass
class RunConfig:
    """How to start a project."""

    project_dir: Path
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    #: "http" (serves TCP), "cli" (one-shot command), "console" (interactive)
    kind: str = "http"
    entrypoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": str(self.project_dir),
            "command": list(self.command),
            "env": dict(self.env),
            "kind": self.kind,
            "entrypoint": self.entrypoint,
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
    return RunConfig(project_dir=project_dir, command=command, env=env,
                     kind=kind, entrypoint=entry)


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


class ServeHandle:
    """A running project server.  Use as a context manager or call :meth:`stop`."""

    def __init__(self, proc: subprocess.Popen, url: str,
                 config: RunConfig, stderr_drainer: _Drainer) -> None:
        self._proc = proc
        self.url = url
        self.config = config
        self._stderr = stderr_drainer
        self._stopped = False

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def project_dir(self) -> Path:
        return self.config.project_dir

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def exit_code(self) -> int | None:
        return self._proc.poll()

    def stderr_text(self) -> str:
        return self._stderr.text()

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the whole process group (child + grandchildren)."""
        if self._stopped:
            return
        self._stopped = True
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError) as e:
                _log.debug("process already gone on SIGTERM: %s", e)
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError) as e:
                    _log.debug("process already gone on SIGKILL: %s", e)
                    pass
                self._proc.wait(timeout=timeout)
        self._stderr.join(timeout=2.0)

    def __enter__(self) -> "ServeHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def serve(project_dir: str | Path, port: int = 0, *,
          startup_timeout: float = 10.0,
          extra_env: dict[str, str] | None = None,
          host: str = "127.0.0.1") -> ServeHandle:
    """Start an HTTP project and wait until it accepts TCP connections.

    ``port=0`` picks a free port.  Fails fast: if the child exits during
    startup, or the deadline passes without the port opening, the process
    group is killed and :class:`ServeError` is raised carrying the
    child's stderr -- never a bare timeout.
    """
    config = run_config(project_dir)
    if config.kind != "http":
        raise ToolError(
            f"serve() needs an HTTP project, but {config.project_dir.name} "
            f"looks like kind={config.kind!r} (entrypoint {config.entrypoint})"
        )
    if port == 0:
        port = find_free_port()

    env = dict(os.environ)
    env.update(config.env)
    env["PORT"] = str(port)
    if extra_env:
        env.update(extra_env)

    _log.info("serving %s on %s:%d: %s", config.project_dir, host, port,
              " ".join(config.command))
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

    deadline = time.monotonic() + startup_timeout
    while True:
        code = proc.poll()
        if code is not None:
            drainer.join(timeout=2.0)
            err = drainer.text().strip()
            raise ServeError(
                f"{config.entrypoint} exited during startup with code {code}"
                + (f"\nstderr:\n{err}" if err else " (no stderr output)"),
                stderr=err,
            )
        if _wait_tcp_ready(host, port, min(deadline, time.monotonic() + 0.2)):
            break
        if time.monotonic() >= deadline:
            handle = ServeHandle(proc, f"http://{host}:{port}", config, drainer)
            handle.stop()
            err = drainer.text().strip()
            raise ServeError(
                f"server did not accept connections on {host}:{port} "
                f"within {startup_timeout:.1f}s"
                + (f"\nstderr:\n{err}" if err else " (no stderr output)"),
                stderr=err,
            )

    url = f"http://{host}:{port}"
    _log.info("server ready at %s (pid %d)", url, proc.pid)
    return ServeHandle(proc, url, config, drainer)
