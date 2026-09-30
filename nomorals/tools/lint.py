"""Lint/format gate (audit Phase B).

``lint(paths)`` runs ``ruff check`` and ``ruff format --check`` when ruff
is importable, following the Makefile's conditional pattern.  When ruff is
absent it returns an explicit "not installed" status — never a silent pass —
so the coding loop can warn instead of gating on nothing.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["lint", "ruff_available", "register"]

_RUFF_LINE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<code>[A-Z]+\d*)\s*(?P<msg>.*)$"
)


def ruff_available() -> bool:
    """True when ruff is importable or on PATH (mirrors the Makefile)."""
    if importlib.util.find_spec("ruff") is not None:
        return True
    return shutil.which("ruff") is not None


def _ruff_cmd() -> list[str]:
    if shutil.which("ruff"):
        return ["ruff"]
    return [sys.executable, "-m", "ruff"]


def _parse_ruff_output(text: str) -> list[dict[str, Any]]:
    violations = []
    for line in text.splitlines():
        m = _RUFF_LINE.match(line.strip())
        if m:
            violations.append({
                "file": m.group("file"),
                "line": int(m.group("line")),
                "col": int(m.group("col")),
                "code": m.group("code"),
                "message": m.group("msg").strip(),
            })
    return violations


def lint(paths: list[str] | None = None,
         repo: str | None = None,
         timeout: float = 120.0) -> dict[str, Any]:
    """Lint ``paths`` (repo-relative). Returns::

        {ok, ruff_installed, violations:[{file,line,col,code,message}],
         format_ok, seconds, message?}

    ``ok`` is False when ruff is missing — check ``ruff_installed`` to
    tell "not installed" apart from "real violations".
    """
    t0 = time.perf_counter()
    root = Path(repo).expanduser().resolve() if repo else Path.cwd().resolve()
    targets = paths or ["."]
    if not ruff_available():
        return {"ok": False, "ruff_installed": False, "violations": [],
                "format_ok": None,
                "seconds": round(time.perf_counter() - t0, 2),
                "message": "ruff not installed — pip install ruff (optional); "
                           "lint gate skipped, not passed"}
    cmd = _ruff_cmd()
    violations: list[dict[str, Any]] = []
    try:
        check = subprocess.run([*cmd, "check", *targets],
                               cwd=str(root), capture_output=True,
                               text=True, timeout=timeout)
        fmt = subprocess.run([*cmd, "format", "--check", *targets],
                             cwd=str(root), capture_output=True,
                             text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "ruff_installed": True, "violations": [],
                "format_ok": None,
                "seconds": round(time.perf_counter() - t0, 2),
                "message": f"ruff timed out after {timeout}s"}
    violations = _parse_ruff_output(check.stdout or "")
    format_ok = fmt.returncode == 0
    if not format_ok:
        for line in (fmt.stdout or "").splitlines():
            line = line.strip()
            if line.endswith("would be reformatted"):
                violations.append({"file": line.split(" ")[0], "line": 0,
                                   "col": 0, "code": "FORMAT",
                                   "message": "would be reformatted"})
    ok = check.returncode == 0 and format_ok
    return {"ok": ok, "ruff_installed": True, "violations": violations,
            "format_ok": format_ok,
            "seconds": round(time.perf_counter() - t0, 2)}


def register(registry: Any) -> None:
    """Attach the lint gate to a registry."""
    from ..core.policy import Capability

    @registry.register(
        "lint",
        description=("Lint/format gate: ruff check + ruff format --check on "
                     "paths. Returns {ok, ruff_installed, violations, "
                     "format_ok}. When ruff is absent, ok=False with "
                     "ruff_installed=False (never a silent pass)."),
        capability=Capability.EXEC_SHELL,
    )
    def _lint(paths: list[str] | None = None,
              repo: str | None = None) -> dict[str, Any]:
        return lint(paths=paths, repo=repo)
