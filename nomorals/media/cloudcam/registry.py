"""CloudCam sessions — the loaded streams, shared by the tool,
the CLI, and chat.

One process-level registry, the same shape as the rig session
registry: a session is loaded once under a name and driven many
times. Loading the same name twice replaces it (stopping the
old poller first); driving an unknown name is refused by name.
The default session (``default``) is created on first use with
the idle source — it stages the placeholder grid until a real
source is loaded over it.

Sources by name: ``idle`` (nothing ever arrives — the
placeholder), ``zmq`` (ZeroMQ PULL, CloudCam wire),
``store`` (the mocap store's pointcloud channel), ``file``
(a JSON frames file or an ASCII PLY file).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .errors import CloudCamError
from .session import CloudCamSession
from .sources import (
    CallableSource,
    FileSource,
    FrameSource,
    StoreSource,
    ZmqSource,
)

__all__ = [
    "get_session",
    "list_sessions",
    "load_session",
    "reset_sessions",
]

_SESSIONS: dict[str, CloudCamSession] = {}


def _build_source(source: str, *, endpoint: str | None = None,
                  path: str | Path | None = None,
                  store: Any = None) -> FrameSource:
    """Name → a live source. Unknown names are refused by name."""
    kind = (source or "").strip().lower() or "idle"
    if kind == "idle":
        return CallableSource(lambda: None, source="idle")
    if kind == "zmq":
        return ZmqSource(endpoint or "tcp://*:8888")
    if kind == "store":
        return StoreSource(store)
    if kind == "file":
        if not path:
            raise CloudCamError(
                "file source needs path=<frames.json or cloud.ply>",
                kind="bad_source")
        return FileSource(path)
    raise CloudCamError(
        f"unknown cloudcam source {source!r} — use one of "
        f"idle|zmq|store|file",
        kind="bad_source")


def load_session(name: str, *, source: str = "idle",
                 endpoint: str | None = None,
                 path: str | Path | None = None,
                 store: Any = None,
                 width: int = 256, height: int = 256,
                 capacity: int = 10) -> CloudCamSession:
    """Load (or replace) a named session. A replaced session's
    poller is stopped first — a load never leaves two pollers
    on one name."""
    name = (name or "").strip()
    if not name:
        raise CloudCamError("session name must not be empty",
                            kind="bad_session")
    old = _SESSIONS.get(name)
    if old is not None and old.running:
        old.stop()
    built = _build_source(source, endpoint=endpoint, path=path,
                          store=store)
    session = CloudCamSession(name, built, width=width,
                              height=height, capacity=capacity)
    _SESSIONS[name] = session
    return session


def get_session(name: str = "default") -> CloudCamSession:
    """The named session — the default is created on first use
    (idle source, placeholder grid). An unknown name is refused
    by name."""
    name = (name or "").strip() or "default"
    session = _SESSIONS.get(name)
    if session is None:
        if name == "default":
            session = load_session("default")
            return session
        raise CloudCamError(
            f"no cloudcam session {name!r} — load it first "
            f"(cloudcam start / nm cloudcam start)",
            kind="unknown_session")
    return session


def list_sessions() -> list[dict[str, Any]]:
    """Every loaded session, in load order."""
    if "default" not in _SESSIONS:
        get_session("default")
    return [dict({"name": name}, **session.to_dict())
            for name, session in _SESSIONS.items()]


def reset_sessions() -> None:
    """Stop every poller and drop every session (tests)."""
    for session in list(_SESSIONS.values()):
        if session.running:
            with contextlib.suppress(CloudCamError):
                session.stop()
    _SESSIONS.clear()
