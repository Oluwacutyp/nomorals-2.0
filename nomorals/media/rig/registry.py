"""Rig sessions — the loaded rigs, shared by the tool, the CLI,
and chat.

One process-level registry, the same shape as the mocap tool's
module-level store: a session is loaded once under a name and
driven many times. Loading the same name twice replaces it;
driving an unknown name is refused by name. The default session
(``default``) is the bundled rig with the BVH remap, created on
first use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import RigError
from .remap import BoneRemap, default_remap
from .rig import RigDefinition, default_rig
from .session import RigSession

__all__ = [
    "get_session",
    "list_sessions",
    "load_session",
    "reset_sessions",
]

_SESSIONS: dict[str, RigSession] = {}


def load_session(name: str, *, rig_path: str | Path | None = None,
                 remap_dialect: str = "bvh",
                 remap_path: str | Path | None = None,
                 width: int = 640, height: int = 360,
                 fps: float = 15.0,
                 smooth: float = 0.35) -> RigSession:
    """Load (or replace) a named session. ``rig_path=None`` loads
    the bundled default rig; ``remap_path=None`` uses the bundled
    table for the dialect."""
    name = (name or "").strip()
    if not name:
        raise RigError("session name must not be empty",
                       kind="bad_session")
    rig = (RigDefinition.from_json(rig_path) if rig_path is not None
           else default_rig())
    if remap_path is not None:
        remap = BoneRemap.from_json(remap_path,
                                    dialect=remap_dialect)
    else:
        remap = default_remap(remap_dialect)
    session = RigSession(rig, remap, width=width, height=height,
                         fps=fps, smooth=smooth)
    _SESSIONS[name] = session
    return session


def get_session(name: str = "default") -> RigSession:
    """The named session — the default is created on first use.
    An unknown name is refused by name."""
    name = (name or "").strip() or "default"
    session = _SESSIONS.get(name)
    if session is None:
        if name == "default":
            session = load_session("default")
            return session
        raise RigError(
            f"no rig session {name!r} — load it first "
            f"(rig_load / nm mocap rig load)",
            kind="unknown_session")
    return session


def list_sessions() -> list[dict[str, Any]]:
    """Every loaded session, in load order."""
    if "default" not in _SESSIONS:
        get_session("default")
    return [dict({"name": name}, **session.to_dict())
            for name, session in _SESSIONS.items()]


def reset_sessions() -> None:
    """Drop every loaded session (tests)."""
    _SESSIONS.clear()
