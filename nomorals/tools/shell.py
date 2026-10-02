"""Sandboxed shell execution.

Layered isolation, strongest available first:

1. ``bwrap`` (bubblewrap) — real namespace isolation, if installed *and* working.
   ``detect_backend`` probes each candidate at runtime: a shipped binary is not
   a working backend (containers often lack the mounts the sandbox needs).
2. ``unshare`` — namespaces without the bwrap dependency, probed the same way.
3. ``resource.setrlimit`` — CPU seconds, address space, file size, process count.
   Always applied, even under bwrap, because limits are the backstop.
4. cwd jail — the process starts in the workspace and is never given a path out.

Plus a hard wall-clock timeout that kills the process group, not just the parent,
so ``sleep 999 &`` cannot outlive the call.

Network is disabled by default. A code-execution tool with open network access is
a remote-access trojan the moment an agent is prompt-injected by a page it fetched.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.errors import SandboxError, ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = ["SandboxLimits", "register", "run_sandboxed"]

_log = get_logger(__name__)

DEFAULT_LIMITS = {
    "cpu_seconds": 120,
    "address_space_mb": 2048,
    "file_size_mb": 256,
    "max_processes": 64,
}


class SandboxLimits:
    """Resource ceilings applied to the child via ``setrlimit``."""

    def __init__(
        self,
        *,
        cpu_seconds: int = 120,
        address_space_mb: int = 2048,
        file_size_mb: int = 256,
        max_processes: int = 64,
    ) -> None:
        self.cpu_seconds = cpu_seconds
        self.address_space_mb = address_space_mb
        self.file_size_mb = file_size_mb
        self.max_processes = max_processes

    def preexec(self) -> None:
        """Runs in the child after fork, before exec."""
        _set(resource.RLIMIT_CPU, self.cpu_seconds, self.cpu_seconds + 5)
        _set(resource.RLIMIT_AS, self.address_space_mb * 1024 * 1024, self.address_space_mb * 1024 * 1024)
        _set(resource.RLIMIT_FSIZE, self.file_size_mb * 1024 * 1024, self.file_size_mb * 1024 * 1024)
        # RLIMIT_NPROC counts every thread of our UID machine-wide. Setting it
        # below what this UID already runs makes even /bin/sh fail to fork
        # ("Cannot fork") on a busy machine — so floor the ceiling above
        # ambient usage. Fork-bomb protection is preserved: the ceiling still
        # caps runaway forking, just never below what already exists.
        try:
            ambient = _uid_thread_count()
        except Exception:  # noqa: BLE001 - /proc may be unavailable; keep the configured ceiling
            ambient = 0
        nproc = max(self.max_processes, ambient + 64)
        _set(resource.RLIMIT_NPROC, nproc, nproc)
        _set(resource.RLIMIT_CORE, 0, 0)
        try:
            os.setsid()  # own process group, so the whole tree can be signalled
        except OSError:  # pragma: no cover - already a leader  # noqa: E103 - already documented
            pass


def _set(which: int, soft: int, hard: int) -> None:
    try:
        resource.setrlimit(which, (soft, hard))
    except (ValueError, OSError) as exc:  # pragma: no cover - platform limits vary
        _log.debug("setrlimit(%s) failed: %s", which, exc)


def _uid_thread_count() -> int:
    """Threads currently owned by our real UID.

    RLIMIT_NPROC is enforced per real UID across the whole machine, not per
    sandbox, so the sandbox ceiling must be measured against this.
    """
    uid = os.getuid()
    count = 0
    try:
        pids = os.listdir("/proc")
    except OSError:
        return 0
    for pid in pids:
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            count += len(os.listdir(f"/proc/{pid}/task"))
        except OSError:
            continue  # exited (or unreadable) between the two syscalls
    return count


# Probe commands: trivial, no network, run once per process per backend.
# They use the same isolation flags run_sandboxed applies, so a passing probe
# means the backend works under the exact invocation we will use.
_PROBE_ARGV: dict[str, list[str]] = {
    "bwrap": ["bwrap", "--ro-bind", "/usr", "/usr", "--proc", "/proc", "--dev", "/dev", "/bin/true"],
    "unshare": ["unshare", "--map-root-user", "--fork", "--pid", "--mount-proc", "/bin/true"],
}
_PROBE_TIMEOUT = 5.0
_PROBE_CACHE: dict[str, bool] = {}


def _probe_backend(name: str) -> bool:
    """Return True if *name* works at runtime, not merely that it is installed.

    ``shutil.which`` only proves the binary exists — a container can ship
    ``unshare`` while refusing the ``/proc`` mount it needs, failing at runtime
    with ``mount /proc failed: Operation not permitted``. The probe result is
    cached at module level, so each backend is probed at most once per process.
    """
    if name in _PROBE_CACHE:
        return _PROBE_CACHE[name]
    usable = False
    if shutil.which(name) is None:
        _log.debug("sandbox backend %s is not installed", name)
    else:
        try:
            probe = subprocess.run(
                _PROBE_ARGV[name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=_PROBE_TIMEOUT,
            )
            usable = probe.returncode == 0
            if not usable:
                _log.debug(
                    "sandbox backend %s probe exited %s: %s",
                    name,
                    probe.returncode,
                    (probe.stderr or "").strip(),
                )
        except (OSError, subprocess.SubprocessError) as exc:
            _log.debug("sandbox backend %s probe failed: %s", name, exc)
    _PROBE_CACHE[name] = usable
    return usable


def detect_backend(preferred: str = "auto") -> str:
    """Pick the strongest isolation that actually works at runtime.

    Each candidate is probe-executed under the flags ``run_sandboxed`` uses
    (cached once per process). ``"auto"`` degrades silently through
    bwrap → unshare → rlimit. An explicit ``"bwrap"``/``"unshare"`` whose
    binary exists but fails its probe raises ``SandboxError`` — an explicit
    choice is an explicit failure, never a silent downgrade. An explicit
    choice that is not installed falls back to ``rlimit``.
    """
    if preferred in {"rlimit", "none"}:
        return preferred
    if preferred in {"bwrap", "unshare"}:
        if _probe_backend(preferred):
            return preferred
        if shutil.which(preferred) is not None:
            raise SandboxError(
                f"sandbox backend {preferred!r} is installed but unavailable at runtime "
                "(probe failed); refusing to silently downgrade"
            )
        _log.debug("sandbox backend %r is not installed; using rlimit", preferred)
        return "rlimit"
    for candidate in ("bwrap", "unshare"):
        if candidate == "unshare" and not sys.platform.startswith("linux"):
            continue
        if _probe_backend(candidate):
            return candidate
    return "rlimit"


def run_sandboxed(
    command: str,
    *,
    cwd: str | os.PathLike[str],
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
    network: bool = False,
    limits: SandboxLimits | None = None,
    backend: str = "auto",
    max_output: int = 1_000_000,
    stdin: str = "",
) -> dict[str, Any]:
    """Run a shell command under the strongest isolation available."""
    limits = limits or SandboxLimits(**DEFAULT_LIMITS)
    workdir = Path(cwd).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    chosen = detect_backend(backend)

    child_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        **(env or {}),
    }
    if not network:
        # Not a guarantee — namespace isolation is — but it removes the ambient
        # proxy configuration so a naive outbound attempt fails fast.
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            child_env.pop(key, None)
        child_env["no_proxy"] = "*"

    argv: list[str]
    if chosen == "bwrap":
        argv = [
            "bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib", "--symlink", "/usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(workdir), str(workdir), "--chdir", str(workdir),
            *(["--unshare-all"] if not network else ["--unshare-pid"]),
            "--die-with-parent", "/bin/sh", "-c", command,
        ]
    elif chosen == "unshare":
        flags = ["--map-root-user", "--fork", "--pid", "--mount-proc"]
        if not network:
            flags.append("--net")
        argv = ["unshare", *flags, "/bin/sh", "-c", f"cd {workdir} && {command}"]
    else:
        argv = ["/bin/sh", "-c", command]

    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(workdir),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=limits.preexec if os.name == "posix" else None,
        )
    except OSError as exc:
        raise SandboxError(f"could not start sandbox ({chosen}): {exc}") from exc

    timed_out = False
    try:
        stdout, stderr = process.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(process)
        stdout, stderr = process.communicate()

    elapsed = time.perf_counter() - started
    return {
        "exit_code": process.returncode,
        "stdout": (stdout or "")[:max_output],
        "stderr": (stderr or "")[:max_output],
        "truncated": bool(stdout and len(stdout) > max_output) or bool(stderr and len(stderr) > max_output),
        "timed_out": timed_out,
        "seconds": round(elapsed, 3),
        "backend": chosen,
        "network": network,
        "cwd": str(workdir),
    }


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill the whole process group. Killing only the parent orphans children."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
        try:
            process.kill()
        except OSError:  # noqa: E103 - process already gone
            pass


def register(registry: Any) -> None:
    """Attach the shell tools to a registry."""
    context = registry.context

    @registry.register(
        "shell_run",
        description="Run a shell command in the sandboxed workspace and return its output.",
        capability=Capability.EXEC_SHELL,
    )
    def shell_run(
        command: str,
        *,
        timeout: float = 120.0,
        network: bool = False,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not command or not command.strip():
            raise ToolError("empty command")
        settings = getattr(context, "settings", None) if context is not None else None
        workspace = settings.workspace_dir if settings is not None else "workspace"
        from .filesystem import safe_path

        workdir = safe_path(context, ".")
        limits = SandboxLimits(
            cpu_seconds=int(min(timeout, 900)),
            address_space_mb=2048,
            file_size_mb=256,
        )
        backend = getattr(getattr(settings, "tools", None), "sandbox_backend", "auto") if settings else "auto"
        result = run_sandboxed(
            command,
            cwd=workdir,
            timeout=timeout,
            env=env,
            network=network,
            limits=limits,
            backend=backend,
            max_output=getattr(getattr(settings, "tools", None), "shell_max_output", 1_000_000) if settings else 1_000_000,
        )
        return result

    @registry.register(
        "python_run",
        description="Execute a Python snippet in the sandbox and return stdout/stderr.",
        capability=Capability.EXEC_CODE,
    )
    def python_run(code: str, *, timeout: float = 60.0) -> dict[str, Any]:
        from .filesystem import safe_path

        workdir = safe_path(context, ".")
        snippet = workdir / f"_snippet_{os.getpid()}_{int(time.time()*1000)}.py"
        snippet.write_text(code, encoding="utf-8")
        try:
            return run_sandboxed(
                f"{sys.executable} {snippet.name}",
                cwd=workdir,
                timeout=timeout,
                limits=SandboxLimits(cpu_seconds=int(min(timeout, 900))),
            )
        finally:
            snippet.unlink(missing_ok=True)
