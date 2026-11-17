"""Annotation lifecycle: scene elements as managed objects.

clickyX's ``Annotation { id, kind, state: Armed -> Completed |
Missed, created_at, timeout_ms }`` with an ``AnnotationManager``
sweep (``overlay/lifecycle.rs``, ``overlay/mod.rs``) is the
reference. An annotation is not a fire-and-forget draw: it is
placed, it lives for a timeout, it completes when its purpose is
served or misses when it expires, and the board can reconcile what
it showed.

Devon's board is durable (atomic JSON under ``<home>/visuals/``)
and owns the cursor position too, so the bridge, the brain tool
and the chat verb all share one cursor and one board. The board's
live annotations (armed + completed) fold straight into an overlay
scene spec for the organ's own renderers.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.failure_store import home_dir as _devon_home_dir
from ..core.ids import new_id
from .spec import validate_scene_spec

__all__ = [
    "Annotation",
    "AnnotationBoard",
    "AnnotationState",
    "get_board",
]

#: clickyX's lifecycle, verbatim: Armed -> Completed | Missed.
ANNOTATION_STATES = ("armed", "completed", "missed")

#: clickyX's cursor default timeout (overlay/mod.rs).
DEFAULT_TIMEOUT_MS = 5000

AnnotationState = str


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Annotation:
    """One managed annotation on the scene surface."""

    id: str
    kind: str
    x: float
    y: float
    w: float = 0.0
    h: float = 0.0
    points: list[list[float]] = field(default_factory=list)
    shape: str = ""
    text: str = ""
    screen: int = 1
    state: str = "armed"
    created_at: float = 0.0
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Annotation:
        data = dict(raw)
        data.setdefault("points", [])
        data.setdefault("shape", "")
        data.setdefault("text", "")
        data.setdefault("screen", 1)
        data.setdefault("state", "armed")
        data.setdefault("created_at", 0.0)
        data.setdefault("timeout_ms", DEFAULT_TIMEOUT_MS)
        return cls(**{k: data[k] for k in (
            "id", "kind", "x", "y", "w", "h", "points", "shape",
            "text", "screen", "state", "created_at", "timeout_ms")})


class AnnotationBoard:
    """The durable board: annotations + the shared cursor position.

    One board per home. The bridge, the brain tool and the chat verb
    all drive the same board through :func:`get_board`, so a tag
    placed by the brain shows on the page the bridge serves.
    """

    def __init__(self, path: Path | str | None = None, *,
                 home: Path | str | None = None) -> None:
        if path is not None:
            self.path = Path(path)
        else:
            base = Path(home) if home is not None else Path(
                _devon_home_dir())
            self.path = base / "visuals" / "annotations.json"
        self._items: dict[str, Annotation] = {}
        self._cursor: tuple[float, float, int] = (0.5, 0.5, 1)
        self._last_flight: list[dict[str, float]] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        for item in raw.get("annotations", []):
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                ann = Annotation.from_dict(item)
            except (TypeError, ValueError):
                continue
            self._items[ann.id] = ann
        cursor = raw.get("cursor") or []
        if (isinstance(cursor, list) and len(cursor) == 3
                and all(isinstance(v, (int, float)) for v in cursor)):
            self._cursor = (float(cursor[0]), float(cursor[1]),
                            int(cursor[2]))
        flight = raw.get("last_flight") or []
        if isinstance(flight, list):
            self._last_flight = [
                dict(f) for f in flight if isinstance(f, dict)]

    def save(self) -> None:
        # A board that cannot persist still works in memory; the
        # failure is visible on the next load, never a crash here.
        with contextlib.suppress(OSError):
            _atomic_write_json(self.path, {
                "annotations": [a.to_dict() for a in self._items.values()],
                "cursor": list(self._cursor),
                "last_flight": self._last_flight,
            })

    # ── cursor ───────────────────────────────────────────────────

    @property
    def cursor(self) -> tuple[float, float, int]:
        """The shared cursor: (x, y, screen), scene-spec coords."""
        return self._cursor

    @property
    def last_flight(self) -> list[dict[str, float]]:
        """The most recent flight frames, for the live page."""
        return [dict(f) for f in self._last_flight]

    def set_last_flight(self, frames: list[dict[str, float]]) -> None:
        self._last_flight = [dict(f) for f in frames]
        self.save()

    # ── lifecycle ────────────────────────────────────────────────

    def create(self, *, kind: str = "cursor", x: float, y: float,
               w: float = 0.0, h: float = 0.0,
               points: list[list[float]] | tuple = (),
               shape: str = "", text: str = "", screen: int = 1,
               timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Annotation:
        ann = Annotation(
            id=new_id(), kind=kind, x=float(x), y=float(y),
            w=float(w), h=float(h),
            points=[[float(px), float(py)] for px, py in points],
            shape=shape, text=text, screen=int(screen),
            state="armed", created_at=time.time(),
            timeout_ms=int(timeout_ms))
        self._items[ann.id] = ann
        self.save()
        return ann

    def move_cursor(self, x: float, y: float,
                    screen: int = 1) -> None:
        """Move the shared cursor without leaving a marker — for
        flights to a point the caller already annotated."""
        self._cursor = (float(x), float(y), int(screen))
        self.save()

    def point(self, x: float, y: float, *, label: str = "",
              screen: int = 1) -> Annotation:
        """Move the cursor here and leave a cursor annotation.

        The cursor move is the shared position; the annotation is
        the visible marker. Neither is an actuation — nothing here
        moves the user's mouse or clicks.
        """
        self._cursor = (float(x), float(y), int(screen))
        ann = self.create(kind="cursor", x=x, y=y, text=label,
                          screen=screen)
        self.save()
        return ann

    def get(self, ann_id: str) -> Annotation | None:
        return self._items.get(ann_id)

    def update(self, ann_id: str, **fields: Any) -> Annotation:
        ann = self._items.get(ann_id)
        if ann is None:
            raise KeyError(f"no annotation with id {ann_id!r}")
        allowed = {"kind", "x", "y", "w", "h", "points", "shape",
                   "text", "screen", "state", "timeout_ms"}
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"cannot update annotation field {key!r}")
            setattr(ann, key, value)
        self.save()
        return ann

    def complete(self, ann_id: str) -> bool:
        ann = self._items.get(ann_id)
        if ann is None:
            return False
        ann.state = "completed"
        self.save()
        return True

    def miss(self, ann_id: str) -> bool:
        ann = self._items.get(ann_id)
        if ann is None:
            return False
        ann.state = "missed"
        self.save()
        return True

    def remove(self, ann_id: str) -> bool:
        if ann_id not in self._items:
            return False
        del self._items[ann_id]
        self.save()
        return True

    def list(self, *, state: str | None = None) -> list[Annotation]:
        items = sorted(self._items.values(),
                       key=lambda a: a.created_at)
        if state is not None:
            items = [a for a in items if a.state == state]
        return items

    def sweep(self, now: float | None = None) -> list[Annotation]:
        """Expire armed annotations past their timeout -> missed.

        Returns the newly missed ones. Completed and already-missed
        annotations are untouched.
        """
        moment = time.time() if now is None else now
        expired: list[Annotation] = []
        for ann in self._items.values():
            if ann.state != "armed":
                continue
            if (moment - ann.created_at) * 1000 >= ann.timeout_ms:
                ann.state = "missed"
                expired.append(ann)
        if expired:
            self.save()
        return expired

    def clear(self) -> int:
        count = len(self._items)
        self._items.clear()
        self.save()
        return count

    # ── spec ─────────────────────────────────────────────────────

    def to_overlay_spec(self, title: str = "") -> dict[str, Any] | None:
        """Live annotations (armed + completed) as an overlay spec.

        ``None`` when the board holds nothing live — an overlay spec
        needs a non-empty annotations list, and an empty page is
        better reported than faked.
        """
        live = [a for a in self.list()
                if a.state in ("armed", "completed")]
        if not live:
            return None
        annotations: list[dict[str, Any]] = []
        for ann in live:
            item: dict[str, Any] = {
                "kind": ann.kind, "x": ann.x, "y": ann.y,
                "text": ann.text, "screen": ann.screen,
            }
            if ann.w > 0:
                item["w"] = ann.w
            if ann.h > 0:
                item["h"] = ann.h
            if ann.points:
                item["points"] = ann.points
            if ann.shape:
                item["shape"] = ann.shape
            annotations.append(item)
        return validate_scene_spec({
            "kind": "overlay", "title": title,
            "annotations": annotations,
        })


_BOARDS: dict[str, AnnotationBoard] = {}


def get_board(home: Path | str | None = None) -> AnnotationBoard:
    """The process-wide board for a home (one board per home)."""
    key = str(Path(home) if home is not None
              else Path(_devon_home_dir()))
    board = _BOARDS.get(key)
    if board is None:
        board = AnnotationBoard(home=home)
        _BOARDS[key] = board
    return board
