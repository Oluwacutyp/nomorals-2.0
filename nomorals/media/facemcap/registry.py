"""Face-take sessions — the loaded takes, shared by the tool,
the CLI, and chat.

One process-level registry, the same shape as the rig and
cloudcam registries: a take is loaded once under a name and
driven many times. Loading the same name twice replaces it;
driving an unknown name is refused by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import FaceMocapError
from .loader import load_csv, load_manifest, load_take_json
from .take import FaceTake

__all__ = [
    "drop_take",
    "get_take",
    "list_takes",
    "load_take",
    "register_take",
    "reset_takes",
]

_TAKES: dict[str, FaceTake] = {}


def load_take(name: str, path: str | Path, *,
              mapping: Any = None, root_name: str | None = None,
              fps: float | None = None) -> FaceTake:
    """Load (or replace) a named face take from a manifest
    (``.morph_anim.json``), a face-take JSON file, or a CSV
    file (``.csv``). A ``.json`` file is a manifest when it
    carries a ``csv`` key and a face take when it carries
    ``frames`` — anything else is refused by name."""
    name = (name or "").strip()
    if not name:
        raise FaceMocapError("take name must not be empty",
                             kind="bad_take")
    target = Path(path).expanduser()
    suffix = target.suffix.lower()
    if suffix == ".json":
        try:
            blob = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise FaceMocapError(
                f"face take at {target} is unreadable: {exc}",
                kind="bad_take") from exc
        if isinstance(blob, dict) and "csv" in blob:
            take = load_manifest(target, name=name)
        elif isinstance(blob, dict) and "frames" in blob:
            take = load_take_json(target, name=name)
        else:
            raise FaceMocapError(
                f"face take {target} is neither a manifest "
                f"(needs a 'csv' key) nor a face take (needs a "
                f"'frames' key)",
                kind="bad_take")
    elif suffix == ".csv":
        take = load_csv(target, name=name, root_name=root_name,
                        fps=fps, mapping=mapping)
    else:
        raise FaceMocapError(
            f"face take {target} must be .morph_anim.json, "
            f".json, or .csv, got {suffix or 'no suffix'}",
            kind="bad_take")
    _TAKES[name] = take
    return take


def register_take(name: str, take: FaceTake) -> FaceTake:
    """Register an in-memory take under a name — the same
    store a file load lands in, for takes built by another
    organ (facemesh captures from a camera, not a CSV).
    Loading the same name twice replaces it, here too."""
    name = (name or "").strip()
    if not name:
        raise FaceMocapError("take name must not be empty",
                             kind="bad_take")
    take.name = name
    _TAKES[name] = take
    return take


def get_take(name: str = "default") -> FaceTake:
    """The named take. An unknown name is refused by name."""
    name = (name or "").strip() or "default"
    take = _TAKES.get(name)
    if take is None:
        raise FaceMocapError(
            f"no face take {name!r} — load it first "
            f"(facemcap load / nm facemcap load)",
            kind="unknown_take")
    return take


def list_takes() -> list[dict[str, Any]]:
    """Every loaded take, in load order."""
    return [take.to_dict() for take in _TAKES.values()]


def drop_take(name: str) -> dict[str, Any]:
    """Drop a loaded take."""
    take = get_take(name)
    del _TAKES[take.name]
    return {"ok": True, "take": take.name,
            "frames": take.total}


def reset_takes() -> None:
    """Drop every loaded take (tests)."""
    _TAKES.clear()
