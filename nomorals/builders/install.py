"""Dependency installation for scaffolded projects, gated by the capability model.

:func:`install_deps` reads ``requirements.txt`` and installs it **only**
when the supplied (or default) :class:`~nomorals.core.policy.Policy`
allows ``exec.install`` (and ``net.download``, since installers fetch
from the network).  ``exec.install`` is a *confirmable* capability, so
even a fully-granted policy needs an explicit confirmation token --
pass one via ``confirmation=policy.issue_confirmation(Capability.EXEC_INSTALL)``.

The installer is chosen with uv-first semantics (uv is 10–100x faster
than pip and hash-aware); pip is the fallback.  ``venv=`` isolates the
install into a fresh ``.venv`` (PEP 668 compliant, nox-style).
:func:`generate_lock` writes a hash-pinned lockfile
(``uv pip compile --generate-hashes`` when uv is present);
:func:`verify_lock` checks it is newer than ``requirements.txt``.

When no policy is passed the default is deny-all: the result is an
explicit denial naming the missing capability, never a silent skip and
never a bypass.

No ``requirements.txt`` (or an empty one) is an honest
``"nothing-to-install"`` -- not an error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability, CapabilitySet, Policy

_log = get_logger(__name__)

__all__ = ["InstallResult", "install_deps", "parse_requirements",
           "generate_lock", "verify_lock", "resolve_installer",
           "ensure_venv"]

#: Capabilities an install needs: running the installer, and reaching the network.
REQUIRED_CAPABILITIES = (Capability.EXEC_INSTALL, Capability.NET_DOWNLOAD)

#: Name of the hash-pinned lockfile written by generate_lock().
LOCKFILE_NAME = "requirements.lock.txt"


def resolve_installer(prefer: str = "auto") -> str:
    """Return ``"uv"`` or ``"pip"``.

    ``prefer="auto"`` picks ``uv`` when it is on PATH (uv is the modern
    gold standard: hash-verified, 10–100x faster), else ``pip``.
    ``prefer="uv"`` raises :class:`ToolError` when uv is missing.
    """
    if prefer == "uv":
        if not shutil.which("uv"):
            raise ToolError("installer 'uv' requested but uv is not on PATH")
        return "uv"
    if prefer == "pip":
        return "pip"
    if prefer != "auto":
        raise ToolError(f"unknown installer {prefer!r}; want 'auto'|'uv'|'pip'")
    return "uv" if shutil.which("uv") else "pip"


def ensure_venv(project_dir: str | Path, venv_dir: str = ".venv") -> Path:
    """Create (or reuse) a virtualenv inside the project; return its path.

    Never touches the ambient interpreter — PEP 668 compliant, the
    nox-style isolation our old bare ``pip install`` lacked.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    venv = project_dir / venv_dir
    python = venv / ("Scripts" if os.name == "nt" else "bin") / \
        ("python.exe" if os.name == "nt" else "python")
    if python.is_file():
        return venv
    _log.info("creating venv at %s", venv)
    proc = subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not python.is_file():
        raise ToolError(
            f"venv creation failed: {(proc.stderr or proc.stdout).strip()[:300]}")
    return venv


def _venv_python(venv: Path) -> str:
    return str(venv / ("Scripts" if os.name == "nt" else "bin") /
               ("python.exe" if os.name == "nt" else "python"))


@dataclass
class InstallResult:
    """Outcome of :func:`install_deps`."""

    #: "nothing-to-install" | "installed" | "failed" | "denied"
    status: str
    #: installer's combined output tail (or the denial reason)
    detail: str = ""
    returncode: int | None = None
    packages: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    #: "uv" | "pip" — which installer actually ran ("" when none did)
    installer_used: str = ""
    #: venv the install went into ("" for the ambient interpreter)
    venv: str = ""
    #: lockfile consulted/written, if any
    lockfile: str = ""
    #: True when dry_run previewed the install without changing anything
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return self.status in ("nothing-to-install", "installed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "detail": self.detail,
            "returncode": self.returncode,
            "packages": list(self.packages),
            "elapsed": round(self.elapsed, 3),
            "installer_used": self.installer_used,
            "venv": self.venv,
            "lockfile": self.lockfile,
            "dry_run": self.dry_run,
        }


def parse_requirements(project_dir: str | Path) -> list[str]:
    """Parse ``requirements.txt`` into a list of requirement lines.

    Comments, blank lines, and pip option lines (``-r``, ``--...``) are
    skipped; inline comments are stripped.
    """
    req_file = Path(project_dir).expanduser().resolve() / "requirements.txt"
    if not req_file.is_file():
        return []
    reqs: list[str] = []
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        reqs.append(line)
    return reqs


def generate_lock(project_dir: str | Path, *,
                  installer: str = "auto",
                  policy: Policy | None = None,
                  actor: str = "builders",
                  confirmation: str | None = None,
                  timeout: float = 300.0) -> InstallResult:
    """Write a hash-pinned lockfile (``requirements.lock.txt``).

    With uv: ``uv pip compile --generate-hashes -o requirements.lock.txt
    requirements.txt`` — the modern gold standard.  Without uv, falls
    back to resolving with pip and freezing the result
    (``pip install --dry-run --report`` when supported, else a pinned
    copy of ``requirements.txt`` is refused as non-deterministic).

    Policy-gated like :func:`install_deps` (it reaches the network).
    """
    started = time.monotonic()
    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise ToolError(f"not a directory: {project_dir}")
    packages = parse_requirements(project_dir)
    if not packages:
        return InstallResult(status="nothing-to-install",
                             detail="no requirements.txt or it lists no packages",
                             elapsed=time.monotonic() - started)

    pol = policy if policy is not None else Policy(default_grant=CapabilitySet.none())
    for capability in REQUIRED_CAPABILITIES:
        denied = _check_capability(pol, capability, actor, confirmation)
        if denied is not None:
            denied.elapsed = time.monotonic() - started
            denied.packages = packages
            return denied

    which = resolve_installer(installer)
    lock = project_dir / LOCKFILE_NAME
    if which == "uv":
        cmd = ["uv", "pip", "compile", "--generate-hashes",
               "-o", LOCKFILE_NAME, "requirements.txt"]
    else:
        # pip has no hash-pinned compile; --dry-run --report needs pip>=23.1
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run",
               "--ignore-installed", "--report", "-",
               "-r", "requirements.txt"]
    _log.info("generate_lock %s: running %s", project_dir, " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(project_dir), capture_output=True,
                              text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return InstallResult(status="failed",
                             detail=f"lock generation failed: {exc}",
                             packages=packages,
                             elapsed=time.monotonic() - started,
                             installer_used=which)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
        return InstallResult(
            status="failed",
            detail=f"lock generation exited {proc.returncode}:\n" +
                   "\n".join(tail),
            returncode=proc.returncode, packages=packages,
            elapsed=time.monotonic() - started, installer_used=which)
    if which == "pip":
        # pip path: no lockfile artifact — report honestly
        return InstallResult(
            status="failed",
            detail="pip cannot emit a hash-pinned lockfile; install uv "
                   "for deterministic locks (uv pip compile --generate-hashes)",
            packages=packages, elapsed=time.monotonic() - started,
            installer_used=which)
    if not lock.is_file():
        return InstallResult(status="failed",
                             detail="uv reported success but no lockfile appeared",
                             packages=packages,
                             elapsed=time.monotonic() - started,
                             installer_used=which)
    return InstallResult(status="installed",
                         detail=f"wrote {LOCKFILE_NAME} "
                                f"({lock.stat().st_size} bytes, hash-pinned)",
                         returncode=0, packages=packages,
                         elapsed=time.monotonic() - started,
                         installer_used=which, lockfile=str(lock))


def verify_lock(project_dir: str | Path) -> dict[str, Any]:
    """Check the lockfile exists and is newer than ``requirements.txt``.

    A stale lock (requirements edited after the lock) is reported, not
    silently trusted — the uv ``--locked`` staleness philosophy.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    req = project_dir / "requirements.txt"
    lock = project_dir / LOCKFILE_NAME
    if not lock.is_file():
        return {"ok": False, "lockfile": str(lock),
                "reason": "no lockfile — run generate_lock() first"}
    if req.is_file() and req.stat().st_mtime > lock.stat().st_mtime:
        return {"ok": False, "lockfile": str(lock),
                "reason": "requirements.txt is newer than the lockfile — "
                          "regenerate with generate_lock()"}
    return {"ok": True, "lockfile": str(lock),
            "bytes": lock.stat().st_size, "reason": "lockfile is fresh"}


def _check_capability(policy: Policy, capability: str, actor: str,
                      confirmation: str | None) -> InstallResult | None:
    """Check one capability; return an InstallResult on denial, else None."""
    decision = policy.check(capability, actor=actor, confirmation=confirmation)
    if not decision.allowed:
        reason = (f"policy denied capability {decision.capability!r}: "
                  f"{decision.reason}")
        if decision.needs_confirmation:
            reason += (" -- exec.install is confirmable: mint a token with "
                       "policy.issue_confirmation(Capability.EXEC_INSTALL) "
                       "and pass it as confirmation=")
        _log.warning("install_deps denied for %s: %s", actor, reason)
        return InstallResult(status="denied", detail=reason)
    return None


def install_deps(project_dir: str | Path, *,
                 policy: Policy | None = None,
                 actor: str = "builders",
                 confirmation: str | None = None,
                 timeout: float = 300.0,
                 installer: str = "auto",
                 venv: str | None = None,
                 dry_run: bool = False,
                 use_lock: bool = False) -> InstallResult:
    """Install a project's ``requirements.txt`` if policy allows.

    * ``installer`` — ``"auto"`` (uv when on PATH, else pip), ``"uv"``,
      or ``"pip"``.
    * ``venv`` — create/isolate into ``<project>/<venv>`` (default
      ``".venv"`` when ``True`` is passed) instead of the ambient
      interpreter.
    * ``dry_run`` — preview only (``pip --dry-run`` / uv equivalent);
      nothing is installed, status is still ``"installed"``.
    * ``use_lock`` — install from ``requirements.lock.txt`` with hash
      verification (``pip --require-hashes`` / uv's native check).

    Returns an :class:`InstallResult`; never raises on policy denial or
    installer failure (raises :class:`ToolError` only for a bad project
    dir or an unusable installer/venv request).
    """
    started = time.monotonic()
    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise ToolError(f"not a directory: {project_dir}")

    packages = parse_requirements(project_dir)
    if not packages:
        _log.info("install_deps %s: nothing to install", project_dir)
        return InstallResult(status="nothing-to-install",
                             detail="no requirements.txt or it lists no packages",
                             elapsed=time.monotonic() - started)

    pol = policy if policy is not None else Policy(default_grant=CapabilitySet.none())
    for capability in REQUIRED_CAPABILITIES:
        denied = _check_capability(pol, capability, actor, confirmation)
        if denied is not None:
            denied.elapsed = time.monotonic() - started
            denied.packages = packages
            return denied

    which = resolve_installer(installer)
    venv_path = ""
    target_python = sys.executable
    if venv:
        venv_path = str(ensure_venv(project_dir,
                                    venv if isinstance(venv, str) else ".venv"))
        target_python = _venv_python(Path(venv_path))

    reqfile = "requirements.txt"
    lock = project_dir / LOCKFILE_NAME
    if use_lock:
        if not lock.is_file():
            return InstallResult(
                status="failed",
                detail=f"use_lock=True but {LOCKFILE_NAME} is missing — "
                       "run generate_lock() first",
                packages=packages, elapsed=time.monotonic() - started,
                installer_used=which, venv=venv_path)
        reqfile = LOCKFILE_NAME

    if which == "uv":
        cmd = ["uv", "pip", "install", "--python", target_python,
               "-r", reqfile]
        if dry_run:
            cmd.append("--dry-run")
    else:
        cmd = [target_python, "-m", "pip", "install", "-r", reqfile]
        if use_lock:
            cmd.append("--require-hashes")
        if dry_run:
            cmd.append("--dry-run")
    _log.info("install_deps %s: running %s", project_dir, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=str(project_dir), capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        detail = f"{which} install timed out after {timeout:.0f}s"
        _log.warning("install_deps %s: %s", project_dir, detail)
        return InstallResult(status="failed", detail=detail,
                             packages=packages,
                             elapsed=time.monotonic() - started,
                             installer_used=which, venv=venv_path,
                             dry_run=dry_run)
    except OSError as exc:
        return InstallResult(status="failed",
                             detail=f"could not spawn {which}: {exc}",
                             packages=packages,
                             elapsed=time.monotonic() - started,
                             installer_used=which, venv=venv_path,
                             dry_run=dry_run)

    output = (proc.stdout + proc.stderr).strip()
    tail = "\n".join(output.splitlines()[-25:])
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        _log.warning("install_deps %s: %s exited %d", project_dir, which,
                     proc.returncode)
        return InstallResult(status="failed",
                             detail=f"{which} exited {proc.returncode}:\n{tail}",
                             returncode=proc.returncode, packages=packages,
                             elapsed=elapsed, installer_used=which,
                             venv=venv_path, dry_run=dry_run)
    verb = "previewed" if dry_run else "installed"
    _log.info("install_deps %s: %s %d packages via %s", project_dir, verb,
              len(packages), which)
    return InstallResult(status="installed",
                         detail=tail or f"{which} reported success",
                         returncode=0, packages=packages, elapsed=elapsed,
                         installer_used=which, venv=venv_path,
                         lockfile=str(lock) if use_lock else "",
                         dry_run=dry_run)
