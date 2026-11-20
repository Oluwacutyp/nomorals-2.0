"""Show sessions — the loaded shows, shared by the tool, the CLI,
and chat.

One process-level registry, the same shape as the rig and
cloudcam registries: a show is loaded once under a name and
driven many times. Loading the same name twice stops the old
show first, then replaces it; driving an unknown name is
refused by name. A show is never created without a store — a
show with nothing to coordinate is not a show.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import ShowError
from .show import Show

__all__ = [
    "drop_show",
    "get_show",
    "list_shows",
    "load_show",
    "reset_shows",
]

_SHOWS: dict[str, Show] = {}


def load_show(name: str, store: Any, *, rig: Any = None,
              cloudcam: Any = None, fps: float = 15.0,
              render_dir: str | Path | None = None,
              export_path: str | Path | None = None) -> Show:
    """Load (or replace) a named show. Replacing a live show
    stops it first — a replaced show must not leak its channel
    claim or its cloudcam poller."""
    name = (name or "").strip()
    if not name:
        raise ShowError("show name must not be empty",
                        kind="bad_show")
    old = _SHOWS.get(name)
    if old is not None and old.state.value not in (
            "idle", "done", "failed"):
        old.stop()
    show = Show(name, store, rig=rig, cloudcam=cloudcam, fps=fps,
                render_dir=render_dir, export_path=export_path)
    _SHOWS[name] = show
    return show


def get_show(name: str = "default") -> Show:
    """The named show. An unknown name is refused by name."""
    name = (name or "").strip() or "default"
    show = _SHOWS.get(name)
    if show is None:
        raise ShowError(
            f"no show {name!r} — create it first "
            f"(show create / nm show create)",
            kind="unknown_show")
    return show


def list_shows() -> list[dict[str, Any]]:
    """Every loaded show, in load order."""
    return [show.status() for show in _SHOWS.values()]


def drop_show(name: str) -> dict[str, Any]:
    """Stop a show (when it is running) and drop it. A live show
    is stopped first — dropping a running show must not leak its
    channel claim or its cloudcam poller."""
    show = get_show(name)
    if show.state.value not in ("idle", "done", "failed"):
        show.stop()
    del _SHOWS[show.name]
    return {"ok": True, "show": show.name,
            "state": show.state.value}


def reset_shows() -> None:
    """Stop every loaded show and drop them all (tests)."""
    for show in list(_SHOWS.values()):
        try:
            if show.state.value not in ("idle", "done", "failed"):
                show.stop()
        except Exception:
            pass
    _SHOWS.clear()
