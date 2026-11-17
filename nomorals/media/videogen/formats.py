"""Video formats as reusable Devon commands.

A format is a parameterized scene composition: the owner invokes
``/map-route``, ``/headline`` or ``/teardown`` (or ``nm video code …``)
and the format builds the scene spec, which the code backend renders —
the Anthropic code-video pattern from the owner's 2026-10-31 link
(prayag_sonar): a format is a reusable slash command over code-defined
video, no video model. Formats live as presets over the style-agnostic
explainer scene engine (the 2026-10-09 architecture principle); the
engine stays style-agnostic, the format carries the composition.

First three (the demo's pattern):

- ``map-route`` — a route draws itself: stops become nodes, legs become
  edges, each revealed in turn.
- ``headline`` — a newspaper push-in: the headline line is boxed and
  highlighted, sub-lines follow.
- ``teardown`` — an exploded parts diagram: numbered parts appear in
  assembly order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .capabilities import VideogenError

__all__ = ["FORMATS", "list_formats", "build_format", "render_format"]


# ---------------------------------------------------------------------------
# builders — params in, explainer design dict out
# ---------------------------------------------------------------------------

def _parse_stops(route: str, stops: Any) -> list[str]:
    """Stops from an explicit list, or by parsing a route string."""
    if stops is not None:
        if isinstance(stops, str):
            stops = [s.strip() for s in stops.split(",")]
        if not isinstance(stops, (list, tuple)) or not stops:
            raise VideogenError(
                "map-route needs 'stops' as a non-empty list of place "
                "names, or a 'route' like 'Lagos to Abuja'")
        names = [str(s).strip() for s in stops if str(s).strip()]
        if len(names) < 2:
            raise VideogenError(
                "map-route needs at least two stops — got "
                f"{names!r}")
        return names
    text = (route or "").strip()
    if not text:
        raise VideogenError(
            "map-route needs a 'route' ('Lagos to Abuja') or 'stops' "
            "(['Lagos', 'Ibadan', 'Abuja'])")
    import re

    via = re.split(r"\s+via\s+", text, flags=re.IGNORECASE)
    if len(via) == 2:
        head = [p.strip() for p in
                re.split(r"\s+to\s+|\s*->\s*", via[0],
                         flags=re.IGNORECASE) if p.strip()]
        mids = [p.strip() for p in via[1].split(",") if p.strip()]
        if len(head) == 2 and mids:
            return [head[0], *mids, head[1]]
    parts = [p.strip() for p in
             re.split(r"\s+to\s+|\s*->\s*|,", text, flags=re.IGNORECASE)
             if p.strip()]
    if len(parts) < 2:
        raise VideogenError(
            f"map-route could not read two stops from {text!r} — try "
            "'Lagos to Abuja' or 'Lagos, Ibadan, Abuja'")
    return parts


def _build_map_route(*, route: str = "", stops: Any = None
                     ) -> dict[str, Any]:
    names = _parse_stops(route, stops)
    n = len(names)
    nodes = [{"id": f"stop{i}", "label": name,
              "x": 0.1 + 0.8 * (i / (n - 1)) if n > 1 else 0.5,
              "y": 0.5, "shape": "circle", "step": i}
             for i, name in enumerate(names)]
    edges = [{"from": f"stop{i}", "to": f"stop{i + 1}",
              "label": f"{names[i]} → {names[i + 1]}", "step": i + 1}
             for i in range(n - 1)]
    return {"engine": "diagram", "kind": "diagram",
            "title": " → ".join(names),
            "nodes": nodes, "edges": edges}


def _build_headline(*, headline: str = "", lines: Any = None,
                    highlight: int = 0) -> dict[str, Any]:
    head = (headline or "").strip()
    if not head:
        raise VideogenError(
            "headline needs a non-empty 'headline' text")
    if lines is None:
        extra: list[str] = []
    elif isinstance(lines, str):
        extra = [s.strip() for s in lines.split("|") if s.strip()]
    else:
        extra = [str(s).strip() for s in lines if str(s).strip()]
    texts = [head, *extra]
    try:
        hl = int(highlight)
    except (TypeError, ValueError):
        hl = 0
    if not 0 <= hl < len(texts):
        hl = 0
    return {"engine": "equation", "kind": "equation", "title": head,
            "lines": [{"text": t, "box": i == hl}
                      for i, t in enumerate(texts)]}


def _build_teardown(*, thing: str = "", parts: Any = None
                    ) -> dict[str, Any]:
    name = (thing or "").strip()
    if not name:
        raise VideogenError("teardown needs a non-empty 'thing' to take "
                            "apart, e.g. 'keyboard'")
    if isinstance(parts, str):
        items = [s.strip() for s in parts.split(",") if s.strip()]
    elif isinstance(parts, (list, tuple)):
        items = [str(p).strip() for p in parts if str(p).strip()]
    else:
        items = []
    if not items:
        raise VideogenError(
            f"teardown needs 'parts' — a non-empty list of part names "
            f"for {name!r}, e.g. ['keycap', 'switch', 'plate', 'case']")
    import math

    cols = max(1, math.ceil(math.sqrt(len(items))))
    rows = math.ceil(len(items) / cols)
    nodes = []
    for i, part in enumerate(items):
        col, row = i % cols, i // cols
        nodes.append({"id": f"part{i}", "label": f"{i + 1}. {part}",
                      "x": (col + 0.5) / cols, "y": (row + 0.5) / rows,
                      "shape": "box", "step": i})
    edges = [{"from": f"part{i}", "to": f"part{i + 1}", "label": "",
              "step": i + 1} for i in range(len(items) - 1)]
    return {"engine": "diagram", "kind": "diagram",
            "title": f"{name} — teardown",
            "nodes": nodes, "edges": edges}


FORMATS: dict[str, dict[str, Any]] = {
    "map-route": {
        "description": ("A route draws itself: stops become nodes, legs "
                        "become edges, each revealed in turn."),
        "params": ["route ('A to B') or stops (['A', 'B', …])"],
        "build": _build_map_route,
    },
    "headline": {
        "description": ("A newspaper push-in: the headline line is boxed "
                        "and highlighted, sub-lines follow."),
        "params": ["headline (required)", "lines (sub-lines)",
                   "highlight (line index, default 0)"],
        "build": _build_headline,
    },
    "teardown": {
        "description": ("An exploded parts diagram: numbered parts appear "
                        "in assembly order."),
        "params": ["thing (required)", "parts (non-empty list)"],
        "build": _build_teardown,
    },
}


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------

def list_formats() -> list[dict[str, Any]]:
    """Every registered format: ``{name, description, params}``."""
    return [{"name": name,
             "description": entry["description"],
             "params": list(entry["params"])}
            for name, entry in FORMATS.items()]


def build_format(name: str, **params: Any) -> dict[str, Any]:
    """Build a format's scene spec from its parameters.

    Raises :class:`VideogenError` naming the known formats for an
    unknown name, or naming the missing parameter for a bad call.
    """
    entry = FORMATS.get((name or "").strip().lower())
    if entry is None:
        known = ", ".join(FORMATS)
        raise VideogenError(
            f"unknown video format {name!r} — known formats: {known}")
    build: Callable[..., dict[str, Any]] = entry["build"]
    try:
        return build(**params)
    except VideogenError:
        raise
    except TypeError as exc:
        raise VideogenError(
            f"format {name!r}: bad parameters: {exc}") from exc


def render_format(name: str, *, duration_s: float = 5.0,
                  fps: float | None = None,
                  size: tuple[int, int] | None = None,
                  out: str | None = None, **params: Any):
    """Build a format's scene and render it on the code backend.

    Returns the :class:`VideoResult` — a real MP4, engine ``"code"``.
    """
    from .pipeline import generate

    design = build_format(name, **params)
    return generate("", backend="code", scene=design,
                    duration_s=duration_s, fps=fps, size=size, out=out)
