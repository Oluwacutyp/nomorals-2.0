"""Scene tools: ``render_scene`` + ``scene_interact`` — the brain's
visual surface.

Devon composes a structured scene spec herself whenever an answer
is spatial — a diagram, a layout, a composition — and renders it
into a deliverable she can use: a PNG image for her own image
outputs, a live HTML page, or terminal text. The validated spec
comes back in the result, so the next stage (the code-video rung)
consumes it without re-authoring it.

``scene_interact`` is the interaction half on top of the same
organ: the brain emits [POINT]/[RECT]/[HIGHLIGHT]/[SCRIBBLE]/
[SHAPE]/[CAPTION] tags inline in her reply text and the tool places
them as managed annotations on the shared board, flies the cursor
to the first point, and hands back the clean text plus the flight
frames — or drives the board and the interactive live page
directly. Annotation is never actuation: a tag places a marker,
nothing here moves a mouse or clicks.

Both tools are thin surfaces over :mod:`nomorals.visuals`; the organ
owns the spec, the renderers, the flight, the board and the bridge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..visuals.annotations import get_board
from ..visuals.flight import flight_frames
from ..visuals.interactive import page_for_board
from ..visuals.render import render_scene as _render_scene
from ..visuals.tags import parse_tags, tags_to_annotations

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``render_scene`` + ``scene_interact`` tools."""
    context = registry.context

    @registry.register(
        "render_scene",
        description=(
            "Render a structured scene spec (diagram, overlay, or 3D "
            "scene) into a deliverable: a PNG image, a live HTML page, "
            "or terminal text. Devon composes the spec herself whenever "
            "an answer is spatial — a diagram, a layout, a composition — "
            "and uses the rendered output in her own images and videos. "
            "target= auto|live|chat|terminal; auto sends diagrams and "
            "overlays to PNG and 3D scenes to the live page. Returns the "
            "file path and the validated spec for the next stage."
        ),
        capability=Capability.MEDIA,
    )
    def render_scene_tool(spec: dict | None = None,
                          target: str = "auto",
                          title: str = "",
                          out_dir: str = "") -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("render_scene needs a spec dict")
        out: str | None = None
        if out_dir:
            out = out_dir
        else:
            home = _home_of(context)
            if home:
                out = str(Path(home) / "visuals")
        return _render_scene(spec, target=target, title=title,
                             out_dir=out)

    @registry.register(
        "scene_interact",
        description=(
            "The Clicky interaction half of the visual surface: place "
            "annotations, fly the cursor, drive the board, render the "
            "interactive live page. action=tag parses [POINT]/[RECT]/"
            "[HIGHLIGHT]/[SCRIBBLE]/[SHAPE]/[CAPTION] tags out of text "
            "(frame pixels when width/height are given, 0..1 without), "
            "places them on the shared board, flies the cursor to the "
            "first point, and returns the clean text plus the flight "
            "frames. action=point moves the cursor to (x, y) with a "
            "label and returns the flight. action=annotate places one "
            "explicit annotation (kind, x, y, w, h, points, shape, "
            "text, screen, timeout_ms). action=list|clear|sweep reads, "
            "clears, or expires the board. action=page writes the "
            "interactive live page from the board and returns its path. "
            "action=idle reports or arms the idle trigger "
            "(text=status|on|off). A tag is a marker only — it never "
            "moves a mouse or clicks."
        ),
        capability=Capability.MEDIA,
    )
    def scene_interact_tool(
            action: str = "tag",
            text: str = "",
            x: float = 0.0,
            y: float = 0.0,
            label: str = "",
            kind: str = "cursor",
            w: float = 0.0,
            h: float = 0.0,
            points: list | None = None,
            shape: str = "",
            screen: int = 1,
            timeout_ms: int = 5000,
            title: str = "",
            width: float | None = None,
            height: float | None = None,
            ann_id: str = "") -> dict[str, Any]:
        home = _home_of(context)
        board = get_board(home)
        verb = (action or "tag").strip().lower()
        if verb == "tag":
            if not isinstance(text, str) or not text:
                raise ValueError("scene_interact tag needs text")
            parsed = parse_tags(text, width=width, height=height)
            placed: list[str] = []
            for ann in tags_to_annotations(parsed.tags):
                created = board.create(
                    kind=ann["kind"], x=ann["x"], y=ann["y"],
                    w=ann.get("w", 0.0), h=ann.get("h", 0.0),
                    points=ann.get("points", ()),
                    shape=ann.get("shape", ""), text=ann.get("text", ""),
                    screen=ann.get("screen", 1))
                placed.append(created.id)
            flight: list[dict[str, float]] = []
            points_tags = [t for t in parsed.tags if t.kind == "point"]
            if points_tags:
                first = points_tags[0]
                start = (board.cursor[0], board.cursor[1])
                flight = flight_frames(start, (first.x, first.y))
                board.move_cursor(first.x, first.y, first.screen)
                board.set_last_flight(flight)
            return {"ok": True, "action": "tag",
                    "text": parsed.text, "placed": placed,
                    "none": parsed.none, "flight": flight,
                    "annotations": len(board.list())}
        if verb == "point":
            start = (board.cursor[0], board.cursor[1])
            frames = flight_frames(start, (float(x), float(y)))
            ann = board.point(float(x), float(y), label=label,
                              screen=int(screen))
            board.set_last_flight(frames)
            return {"ok": True, "action": "point",
                    "annotation": ann.to_dict(), "flight": frames}
        if verb == "annotate":
            ann = board.create(
                kind=kind, x=float(x), y=float(y),
                w=float(w), h=float(h), points=points or (),
                shape=shape, text=text or label,
                screen=int(screen), timeout_ms=int(timeout_ms))
            return {"ok": True, "action": "annotate",
                    "annotation": ann.to_dict()}
        if verb == "list":
            return {"ok": True, "action": "list",
                    "cursor": {"x": board.cursor[0],
                               "y": board.cursor[1],
                               "screen": board.cursor[2]},
                    "annotations": [a.to_dict()
                                    for a in board.list()]}
        if verb == "clear":
            return {"ok": True, "action": "clear",
                    "cleared": board.clear()}
        if verb == "sweep":
            expired = board.sweep()
            return {"ok": True, "action": "sweep",
                    "expired": [a.id for a in expired]}
        if verb == "page":
            out_dir = (str(Path(home) / "visuals")
                       if home else None)
            path = page_for_board(board, title=title or "scene",
                                  out_dir=out_dir)
            if path is None:
                return {"ok": False, "action": "page",
                        "error": "board holds nothing live — place an "
                                 "annotation first"}
            return {"ok": True, "action": "page", "path": path}
        if verb == "idle":
            from ..visuals.idle import get_idle_trigger

            trigger = get_idle_trigger(home)
            sub = (text or "status").strip().lower()
            if sub == "on":
                trigger.arm()
                return {"ok": True, "action": "idle", "armed": True}
            if sub == "off":
                trigger.disarm()
                return {"ok": True, "action": "idle", "armed": False}
            if sub == "status":
                return {"ok": True, "action": "idle",
                        "armed": trigger.armed,
                        "idle_seconds": trigger.idle_seconds(),
                        "would_fire": trigger.check()}
            raise ValueError("scene_interact idle needs "
                             "text=status|on|off")
        raise ValueError(f"unknown scene_interact action {action!r}")
