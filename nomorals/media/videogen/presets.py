"""Manim-class explainer presets as named, selectable video presets.

The explainer family in :mod:`nomorals.media.motion_studio.explainers`
builds diagram/equation/chart/code scenes, but it shipped as engine
kinds inside procedural segments — never as presets the owner can name
in chat. A preset is a named builder over that same engine: the owner
says ``/video explain code-trace <lines>`` and the preset builds the
scene spec, which the code backend renders — the same path
``/map-route``/``/headline``/``/teardown`` ride (see :mod:`.formats`),
no second renderer.

Four presets (the three existing kinds, plus the code trace this lane
adds to the family):

- ``equation-walkthrough`` — an equation building up line by line,
  the current line boxed and lit.
- ``diagram-build`` — a labelled diagram assembling node by node,
  edges drawn in order.
- ``code-trace`` — code lines lighting up in execution order, the
  state each line leaves behind named beside it.
- ``chart-build`` — a chart growing bar by bar (or drawing its line),
  values labelled.

An unknown preset name is refused with the list of real ones — never
a silent default. A preset that cannot build its scene (empty lines,
more values than code lines, a malformed chart series) raises
:class:`VideogenError` naming the problem before any render work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .capabilities import VideogenError

__all__ = ["PRESETS", "list_presets", "build_preset", "render_preset"]


# ---------------------------------------------------------------------------
# content parsing — chat-shaped strings in, clean lists out
# ---------------------------------------------------------------------------

def _split_lines(value: Any, what: str) -> list[str]:
    """Items from a ``|``-separated string or a list.

    ``|`` is the only separator: code and equations carry commas, so a
    ``,`` fallback would split a line in half. A string with no ``|``
    is one item.
    """
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [s.strip() for s in value.split("|") if s.strip()]
    else:
        items = []
    if not items:
        raise VideogenError(f"{what} needs non-empty content — pass "
                            f"items separated by ' | ', e.g. 'a | b | c'")
    return items


def _split_nodes(value: Any) -> list[str]:
    """Node labels from a string or a list.

    ``|`` first (the chat idiom); a bare comma list reads too, the way
    ``/map-route`` reads ``Lagos, Ibadan, Abuja``.
    """
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        if "|" in value:
            items = [s.strip() for s in value.split("|") if s.strip()]
        else:
            items = [s.strip() for s in value.split(",") if s.strip()]
    else:
        items = []
    if not items:
        raise VideogenError("diagram-build needs non-empty 'nodes' — "
                            "pass labels separated by ' | ', e.g. "
                            "'A | B | C'")
    return items


def _parse_series(value: Any) -> list[dict[str, Any]]:
    """Chart points from ``"label:value, label:value"`` or a list."""
    if isinstance(value, (list, tuple)):
        raw = [dict(v) if isinstance(v, dict) else {"label": str(v)}
               for v in value]
        points = []
        for item in raw:
            try:
                num = float(item["value"])
            except (KeyError, TypeError, ValueError):
                raise VideogenError(
                    "chart-build series points need a numeric 'value' — "
                    f"got {item!r}") from None
            points.append({"label": str(item.get("label", "")),
                           "value": num})
        if not points:
            raise VideogenError("chart-build needs a non-empty 'series'")
        return points
    text = (value or "").strip() if isinstance(value, str) else ""
    if not text:
        raise VideogenError("chart-build needs a non-empty 'series' — "
                            "pass 'label:value, label:value', e.g. "
                            "'a:3, b:5'")
    points = []
    for part in text.split(","):
        label, sep, num = part.strip().partition(":")
        if not sep or not label.strip():
            raise VideogenError(
                f"chart-build could not read a label:value pair from "
                f"{part.strip()!r} — try 'a:3, b:5'")
        try:
            value_f = float(num)
        except ValueError:
            raise VideogenError(
                f"chart-build could not read a number from "
                f"{part.strip()!r} — try 'a:3, b:5'") from None
        points.append({"label": label.strip(), "value": value_f})
    return points


# ---------------------------------------------------------------------------
# builders — params in, explainer design dict out
# ---------------------------------------------------------------------------

def _build_equation_walkthrough(*, lines: Any = None,
                                title: str = "") -> dict[str, Any]:
    items = _split_lines(lines, "equation-walkthrough")
    return {"engine": "equation", "kind": "equation",
            "title": str(title or ""),
            # every line is boxed when it is current — the Manim
            # highlight idiom: the step being explained is lit.
            "lines": [{"text": t, "box": True} for t in items]}


def _build_diagram_build(*, nodes: Any = None,
                         title: str = "") -> dict[str, Any]:
    names = _split_nodes(nodes)
    n = len(names)
    built = [{"id": f"node{i}", "label": name,
              "x": 0.1 + 0.8 * (i / (n - 1)) if n > 1 else 0.5,
              "y": 0.5, "shape": "circle", "step": i}
             for i, name in enumerate(names)]
    edges = [{"from": f"node{i}", "to": f"node{i + 1}", "label": "",
              "step": i + 1} for i in range(n - 1)]
    return {"engine": "diagram", "kind": "diagram",
            "title": str(title or ""), "nodes": built, "edges": edges}


def _build_code_trace(*, code: Any = None, values: Any = None,
                      title: str = "") -> dict[str, Any]:
    lines = _split_lines(code, "code-trace")
    vals: list[str] = []
    if values is not None and str(values).strip():
        vals = _split_lines(values, "code-trace values")
    if len(vals) > len(lines):
        raise VideogenError(
            f"code-trace has {len(vals)} values for {len(lines)} code "
            f"lines — one value per line at most, aligned in order")
    return {"engine": "code", "kind": "code",
            "title": str(title or ""),
            "lines": [{"text": t,
                       "values": vals[i] if i < len(vals) else ""}
                      for i, t in enumerate(lines)]}


def _build_chart_build(*, series: Any = None, title: str = "",
                       chart: str = "bar") -> dict[str, Any]:
    kind = str(chart or "bar").strip().lower()
    if kind not in ("bar", "line"):
        raise VideogenError(
            f"chart-build 'chart' must be 'bar' or 'line', got "
            f"{chart!r}")
    return {"engine": "chart", "kind": "chart",
            "title": str(title or ""), "chart": kind,
            "series": _parse_series(series)}


PRESETS: dict[str, dict[str, Any]] = {
    "equation-walkthrough": {
        "description": ("An equation building up line by line — the "
                        "current line boxed and lit, earlier lines "
                        "settled."),
        "params": ["lines ('a | b | c', required)",
                   "title (optional)"],
        "build": _build_equation_walkthrough,
    },
    "diagram-build": {
        "description": ("A labelled diagram assembling node by node, "
                        "edges drawn in order."),
        "params": ["nodes ('A | B | C', required)",
                   "title (optional)"],
        "build": _build_diagram_build,
    },
    "code-trace": {
        "description": ("Code lines lighting up in execution order, "
                        "the state each line leaves behind named "
                        "beside it."),
        "params": ["code ('x = 1 | x = x + 1', required)",
                   "values ('x = 1 | x = 2', one per line, optional)",
                   "title (optional)"],
        "build": _build_code_trace,
    },
    "chart-build": {
        "description": ("A chart growing bar by bar (or drawing its "
                        "line), values labelled."),
        "params": ["series ('a:3, b:5', required)",
                   "chart ('bar' or 'line', default 'bar')",
                   "title (optional)"],
        "build": _build_chart_build,
    },
}


# ---------------------------------------------------------------------------
# public surface — mirrors formats.py
# ---------------------------------------------------------------------------

def list_presets() -> list[dict[str, Any]]:
    """Every registered preset: ``{name, description, params}``."""
    return [{"name": name,
             "description": entry["description"],
             "params": list(entry["params"])}
            for name, entry in PRESETS.items()]


def build_preset(name: str, **params: Any) -> dict[str, Any]:
    """Build a preset's scene spec from its parameters.

    Raises :class:`VideogenError` naming the known presets for an
    unknown name, or naming the bad parameter for a bad call. The
    design is validated before it is returned, so a bad call fails
    here — never at render time.
    """
    entry = PRESETS.get((name or "").strip().lower())
    if entry is None:
        known = ", ".join(PRESETS)
        raise VideogenError(
            f"unknown explainer preset {name!r} — known presets: {known}")
    build: Callable[..., dict[str, Any]] = entry["build"]
    try:
        design = build(**params)
    except VideogenError:
        raise
    except TypeError as exc:
        raise VideogenError(
            f"preset {name!r}: bad parameters: {exc}") from exc
    from ..motion_studio.explainers import (
        MotionStudioError,
        validate_design,
    )

    try:
        validate_design(design)
    except MotionStudioError as exc:
        raise VideogenError(
            f"preset {name!r}: invalid scene: {exc}") from exc
    return design


def render_preset(name: str, *, duration_s: float = 5.0,
                  fps: float | None = None,
                  size: tuple[int, int] | None = None,
                  out: str | None = None, narrate: bool = True,
                  **params: Any):
    """Build a preset's scene and render it on the code backend.

    Returns the :class:`VideoResult` — a real MP4, engine ``"code"``,
    narrated by default (each build step spoken through the TTS
    ladder); ``narrate=False`` renders it silent on purpose. The
    result names the preset that ran (``meta["preset"]`` and the note).
    """
    from .pipeline import generate

    design = build_preset(name, **params)
    return generate("", backend="code", scene=design,
                    duration_s=duration_s, fps=fps, size=size, out=out,
                    narrate=narrate, preset=name)
