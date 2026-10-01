"""Dependency installation for scaffolded projects, gated by the capability model.

:func:`install_deps` reads ``requirements.txt`` and runs
``python -m pip install -r requirements.txt`` **only** when the supplied
(or default) :class:`~nomorals.core.policy.Policy` allows
``exec.install`` (and ``net.download``, since pip fetches from the
network).  ``exec.install`` is a *confirmable* capability, so even a
fully-granted policy needs an explicit confirmation token -- pass one
via ``confirmation=policy.issue_confirmation(Capability.EXEC_INSTALL)``.

When no policy is passed the default is deny-all: the result is an
explicit denial naming the missing capability, never a silent skip and
never a bypass.

No ``requirements.txt`` (or an empty one) is an honest
``"nothing-to-install"`` -- not an error.
"""

from __future__ import annotations

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

__all__ = ["InstallResult", "install_deps", "parse_requirements"]

#: Capabilities an install needs: running pip, and reaching the network.
REQUIRED_CAPABILITIES = (Capability.EXEC_INSTALL, Capability.NET_DOWNLOAD)


@dataclass
class InstallResult:
    """Outcome of :func:`install_deps`."""

    #: "nothing-to-install" | "installed" | "failed" | "denied"
    status: str
    #: pip's combined output tail (or the denial reason)
    detail: str = ""
    returncode: int | None = None
    packages: list[str] = field(default_factory=list)
    elapsed: float = 0.0

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
                 timeout: float = 300.0) -> InstallResult:
    """Install a project's ``requirements.txt`` if policy allows.

    Returns an :class:`InstallResult`; never raises on policy denial or
    pip failure (raises :class:`ToolError` only for a bad project dir).
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

    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    _log.info("install_deps %s: running %s", project_dir, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=str(project_dir), capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        detail = f"pip install timed out after {timeout:.0f}s"
        _log.warning("install_deps %s: %s", project_dir, detail)
        return InstallResult(status="failed", detail=detail,
                             packages=packages, elapsed=time.monotonic() - started)
    except OSError as exc:
        return InstallResult(status="failed", detail=f"could not spawn pip: {exc}",
                             packages=packages, elapsed=time.monotonic() - started)

    output = (proc.stdout + proc.stderr).strip()
    tail = "\n".join(output.splitlines()[-25:])
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        _log.warning("install_deps %s: pip exited %d", project_dir, proc.returncode)
        return InstallResult(status="failed", detail=f"pip exited {proc.returncode}:\n{tail}",
                             returncode=proc.returncode, packages=packages, elapsed=elapsed)
    _log.info("install_deps %s: installed %d packages", project_dir, len(packages))
    return InstallResult(status="installed", detail=tail or "pip reported success",
                         returncode=0, packages=packages, elapsed=elapsed)
