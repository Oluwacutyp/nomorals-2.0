"""One case home for the security probe organs.

A case is one folder that holds a whole engagement:

* ``<home>/data/evidence/<case>/scope.txt`` — the scope file. Every
  probe tool reads it (or the default below); no scope file, no probe.
* ``<home>/data/evidence/<case>/hypotheses.jsonl`` — the planner
  ledger (see :mod:`nomorals.security.planner`).
* ``<home>/data/evidence/<case>/attacker-<host>.jsonl`` — the
  attacker's resumable checkpoint, and ``attacker-<host>.txt`` for
  found credentials.
* ``<home>/data/evidence/<case>/chain.jsonl`` — the evidence chain
  (see :mod:`nomorals.security.evidence`).

The default scope file lives at ``<home>/data/security/scope.txt``.
It is a *location*, not a permission: the wiring never creates it
with permissive content, and a tool that finds no scope file
anywhere refuses, naming the path it looked for. Default-deny —
a bare URL never runs unscoped.

Scope resolution order for a probe tool: an explicit ``scope`` path
first, then the case's ``scope.txt``, then the default scope file.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from .scope import Governor, Scope, load_scope

__all__ = [
    "DEFAULT_SCOPE_NAME",
    "case_dir",
    "slugify_case",
    "scope_path",
    "default_scope_path",
    "resolve_scope",
    "governor_for",
    "attacker_paths",
    "init_case",
    "home_of",
    "set_pool_factory",
    "pool_from_factory",
]

#: The proxy-pool factory, plugged in by the L4 layer
#: (tools/security.py). The security organs are L3 and cannot import the
#: proxy pool (L4) themselves; without a factory, replays run direct.
#: One shared slot for every replay-shaped tool (report_amp, replay).
_POOL_FACTORY: Callable[[Any], Any] | None = None


def set_pool_factory(factory: Callable[[Any], Any] | None) -> None:
    """Plug in (or clear) the proxy-pool factory used by the replay
    tools."""
    global _POOL_FACTORY
    _POOL_FACTORY = factory


def pool_from_factory(context: Any) -> Any | None:
    """Build the rotation pool from the L4 factory, or None (direct)."""
    if _POOL_FACTORY is None or context is None:
        return None
    try:
        pool = _POOL_FACTORY(context)
        if pool is not None and len(pool):
            pool.probe_all()
            return pool
    except Exception:  # noqa: BLE001 - direct is valid
        pass
    return None


def home_of(context: Any) -> str:
    """Devon's home from a tool-registry context.

    The production context is an AgentContext: it carries
    ``.settings`` (a Settings with ``home_path`` / ``home``) and no
    bare ``.home`` — reading ``getattr(context, "home", ".")`` falls
    back to the working directory there, so case dirs, ledgers and
    dossiers would land in the caller's cwd instead of the home.
    This is the established idiom (tools/security.py ``_home_of``):
    settings first, a bare ``.home`` (the test harness shape) second,
    ``"."`` only when neither exists.
    """
    settings = getattr(context, "settings", None)
    if settings is not None:
        home = (getattr(settings, "home_path", None)
                or getattr(settings, "home", None))
        if home:
            return str(Path(os.path.expanduser(str(home))).resolve())
    home = getattr(context, "home", None)
    if home:
        return str(Path(os.path.expanduser(str(home))).resolve())
    return "."

#: The scope file's name inside a case dir.
DEFAULT_SCOPE_NAME = "scope.txt"

_SLUG = re.compile(r"[^a-z0-9_\-]+")


def slugify_case(case: str) -> str:
    """The case-name rule — the same one ``nm engage`` uses, so a case
    named in one place resolves in the other."""
    slug = _SLUG.sub("-", (case or "").strip().lower()).strip("-")
    if not slug:
        raise ToolError("a case name is required")
    return slug


def case_dir(home: str | Path, case: str) -> Path:
    """``<home>/data/evidence/<case>/`` — one folder per engagement."""
    return Path(home) / "data" / "evidence" / slugify_case(case)


def scope_path(home: str | Path, case: str) -> Path:
    """The case's scope file."""
    return case_dir(home, case) / DEFAULT_SCOPE_NAME


def default_scope_path(home: str | Path) -> Path:
    """The default scope file — the fallback when a tool is given
    neither a scope path nor a case."""
    return Path(home) / "data" / "security" / DEFAULT_SCOPE_NAME


def resolve_scope(home: str | Path, scope: str = "",
                  case: str = "") -> Scope:
    """The scope a probe runs under. Explicit path first, then the
    case's scope file, then the default scope file. Refuses, naming
    the exact path, when no scope file exists anywhere."""
    if (scope or "").strip():
        return load_scope(scope)
    if (case or "").strip():
        path = scope_path(home, case)
        if path.is_file():
            return load_scope(path)
        raise ToolError(
            f"no scope file for case {case!r}: {path} — create it "
            f"with `nm sec case init {case}` and list the hosts under "
            f"test, or pass --scope <file>")
    path = default_scope_path(home)
    if path.is_file():
        return load_scope(path)
    raise ToolError(
        f"no scope file: {path} — create it and list the hosts under "
        f"test, or pass --scope <file> / --case <name>")


def governor_for(scope: Scope, *, max_total: int = 50_000,
                 max_per_host: float = 5.0,
                 max_seconds: float = 3_600.0) -> Governor:
    """The single Governor construction point. Defaults are the
    Governor's own; a tool passes its own budget when the run's size
    is known (a 20-way race wave, a 400-request inject run)."""
    return Governor(scope, max_total=max_total, max_per_host=max_per_host,
                    max_seconds=max_seconds)


def attacker_paths(home: str | Path, case: str,
                   host: str) -> dict[str, str]:
    """The attacker's checkpoint + found-credentials files for one
    host inside one case dir."""
    slug = _SLUG.sub("-", (host or "").strip().lower()).strip("-") or "host"
    base = case_dir(home, case)
    return {"resume_file": str(base / f"attacker-{slug}.jsonl"),
            "out_file": str(base / f"attacker-{slug}.txt")}


def init_case(home: str | Path, case: str) -> dict[str, Any]:
    """Create a case dir with a scope template. The template
    authorises NOTHING — an empty include list means load_scope
    refuses it until the owner lists the hosts under test."""
    base = case_dir(home, case)
    base.mkdir(parents=True, exist_ok=True)
    path = base / DEFAULT_SCOPE_NAME
    if not path.is_file():
        path.write_text(
            f"# Scope file for case {case} — one host per line after include:\n"
            "# include: example.com, *.example.com, 10.0.0.0/24\n"
            "# exclude: cdn.example.com\n"
            "# allow_all: false\n"
            "include:\n", encoding="utf-8")
    return {"case": slugify_case(case), "dir": str(base),
            "scope": str(path)}
