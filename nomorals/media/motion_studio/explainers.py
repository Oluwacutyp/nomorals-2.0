"""Code-driven explainer scenes — the Manim-class family (plan §1.5d).

Wave 1's named gap 1, built. Until this module, a procedural
segment had exactly one engine (``visualizer``) and the explainer
family — diagrams, equations-as-text, chart builds — was a named
compile refusal in :mod:`.project`. The family now ships as three
engines over one discipline:

* **Scenes are data.** A scene is a plain dict in a procedural
  segment's ``design`` (``engine`` = ``diagram`` | ``equation`` |
  ``chart``, or the family name ``explainer`` + ``kind``), validated
  by :func:`validate_design` at compile time — a malformed scene is
  a build error naming the field, never a render-time surprise.
* **Deterministic frame functions.** Every frame is a pure function
  of (spec, size, t): no wall clock, no unseeded randomness, no
  model. The same document renders the same frames on every run —
  the golden-harness contract (frame-sampled stats, not vibes).
* **Builds, not cuts.** Each scene is a list of *steps*; elements
  appear across their step's window with the studio's smooth ease —
  a labelled diagram assembles node by node, an equation accrues
  line by line, a chart grows bar by bar (Manim's build idiom,
  Devon-shaped).
* **Text is rendered, never model-drawn** (the P-brainstorm rule):
  labels, equations and values are composited by the deterministic
  text path — ``_core.find_font``/``text_size``, the same helpers
  the typography tier uses. Narration rides the segment's
  ``text``/``words`` through :func:`typography.to_words`, so caption
  timing keeps its §2.4 provenance label (``measured`` /
  ``estimated`` / ``authored``) and the compile record states it.

Out of scope stays refused, by name: engines outside the three
kinds (an ``explainer`` with no ``kind`` names no buildable scene),
and scripted/freeform scene code — §1.5d's family is declarative
scene data, not an embedded runtime (the §1.3b decision row).
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment without numpy
    np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - environment without Pillow
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

from ._core import MotionStudioError, ease, find_font, render_sequence
from ._core import profile_defaults, text_size
from .typography import Word, timing_sources, to_words
from ..edit_engine.frametime import FrameGrid
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "EXPLAINER_KINDS",
    "EXPLAINER_ENGINES",
    "is_explainer_engine",
    "explainer_kind",
    "list_explainers",
    "validate_design",
    "frame_image",
    "render_explainer",
]

#: the scene kinds the §1.5d family builds.
EXPLAINER_KINDS = ("diagram", "equation", "chart")

#: engine names a procedural segment may carry for this family:
#: the three kinds directly, plus the family alias (``explainer``),
#: which must name its scene in ``design["kind"]``.
EXPLAINER_ENGINES = ("explainer",) + EXPLAINER_KINDS

_KIND_LABELS = {
    "diagram": "labelled diagram build — nodes and edges assemble "
               "step by step",
    "equation": "equation build — lines accrue step by step, the "
                "current line lit, earlier lines settled",
    "chart": "chart build — axes first, then bars grow / the line "
             "draws on, values labelled",
}

# the studio's explainer palette (dark board, one accent family)
BG_TOP = (13, 16, 32)
BG_BOTTOM = (7, 8, 15)
INK = (242, 244, 250)
DIM = (148, 156, 178)
GRID = (52, 60, 88)
ACCENT = (0, 240, 255)
PALETTE = ("#38BDF8", "#F472B6", "#A3E635", "#FBBF24",
           "#C084FC", "#34D399", "#FB7185", "#FACC15")
NODE_SHAPES = ("box", "circle")
CHART_KINDS = ("bar", "line")


def list_explainers() -> list[dict[str, str]]:
    return [{"kind": k, "label": _KIND_LABELS[k]}
            for k in EXPLAINER_KINDS]


def is_explainer_engine(engine: str) -> bool:
    return str(engine or "") in EXPLAINER_ENGINES


def explainer_kind(design: dict[str, Any]) -> str:
    """Resolve a design dict to its scene kind — or refuse, named.

    The refusal is the family's scope boundary: engine ``explainer``
    with no (or an unknown) ``kind`` names no buildable scene, and
    anything outside §1.5d's three kinds stays unbuilt on purpose.
    """
    engine = str((design or {}).get("engine", ""))
    if engine in EXPLAINER_KINDS:
        return engine
    if engine == "explainer":
        kind = str((design or {}).get("kind", ""))
        if kind in EXPLAINER_KINDS:
            return kind
        raise MotionStudioError(
            f"explainer segment names no buildable scene kind "
            f"(kind={kind!r}) — the plan §1.5d family builds "
            f"'diagram', 'equation' and 'chart' scenes; set "
            f"design['kind'] to one of those")
    raise MotionStudioError(
        f"not an explainer engine: {engine!r} — the plan §1.5d "
        f"family builds 'diagram', 'equation' and 'chart'")


# ---------------------------------------------------------------------------
# validation — a scene is data; malformed data is a build error
# ---------------------------------------------------------------------------

def _hex(value: Any, what: str) -> tuple[int, int, int]:
    s = str(value or "").strip()
    if len(s) == 7 and s.startswith("#"):
        try:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        except ValueError as exc:
            _log.debug("explainers._hex failed", exc_info=True)
    raise MotionStudioError(f"{what}: bad color {value!r} — use #RRGGBB")


def _step_of(el: dict[str, Any], default: int, what: str) -> tuple[int, float]:
    """An element's (step, at): explicit step int, or -1 + seconds."""
    at = el.get("at")
    if at is not None:
        at_f = float(at)
        if at_f < 0:
            raise MotionStudioError(f"{what}: 'at' must be ≥ 0, got {at_f}")
        return -1, at_f
    step = el.get("step", default)
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise MotionStudioError(
            f"{what}: 'step' must be an int ≥ 0, got {step!r}")
    return step, -1.0


def _color_of(el: dict[str, Any], idx: int, what: str) -> tuple[int, int, int]:
    if el.get("color"):
        return _hex(el["color"], what)
    return _hex(PALETTE[idx % len(PALETTE)], what)


def _steps_of(elements: Sequence[dict[str, Any]]) -> int:
    steps = [int(el["step"]) for el in elements if int(el["step"]) >= 0]
    return (max(steps) + 1) if steps else 1


def _validate_diagram(design: dict[str, Any]) -> dict[str, Any]:
    nodes_in = design.get("nodes")
    if not isinstance(nodes_in, list) or not nodes_in:
        raise MotionStudioError(
            "diagram scene needs a non-empty 'nodes' list")
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(nodes_in):
        what = f"diagram node #{i}"
        if not isinstance(raw, dict):
            raise MotionStudioError(f"{what}: must be a dict")
        nid = str(raw.get("id", ""))
        if not nid:
            raise MotionStudioError(f"{what}: missing 'id'")
        if nid in seen:
            raise MotionStudioError(f"{what}: duplicate node id {nid!r}")
        seen.add(nid)
        try:
            x, y = float(raw["x"]), float(raw["y"])
        except (KeyError, TypeError, ValueError):
            raise MotionStudioError(
                f"{what} ({nid}): needs numeric 'x' and 'y'") from None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise MotionStudioError(
                f"{what} ({nid}): x/y are normalized 0..1, "
                f"got ({x}, {y})")
        shape = str(raw.get("shape", "box"))
        if shape not in NODE_SHAPES:
            raise MotionStudioError(
                f"{what} ({nid}): shape must be one of {NODE_SHAPES}, "
                f"got {shape!r}")
        step, at = _step_of(raw, 0, f"{what} ({nid})")
        nodes.append({"id": nid, "label": str(raw.get("label", nid)),
                      "x": x, "y": y, "shape": shape,
                      "color": _color_of(raw, i, f"{what} ({nid})"),
                      "step": step, "at": at})
    by_id = {n["id"]: n for n in nodes}
    edges: list[dict[str, Any]] = []
    for i, raw in enumerate(design.get("edges") or []):
        what = f"diagram edge #{i}"
        if not isinstance(raw, dict):
            raise MotionStudioError(f"{what}: must be a dict")
        a, b = str(raw.get("from", "")), str(raw.get("to", ""))
        for end in (a, b):
            if end not in by_id:
                raise MotionStudioError(
                    f"{what}: endpoint {end!r} is not a node id")
        # an edge builds once both its nodes exist, unless authored
        default = max(by_id[a]["step"], by_id[b]["step"])
        step, at = _step_of(raw, default, what)
        edges.append({"a": a, "b": b, "label": str(raw.get("label", "")),
                      "color": _color_of(raw, i, what),
                      "step": step, "at": at})
    return {"nodes": nodes, "edges": edges,
            "steps": _steps_of(nodes + edges)}


def _validate_equation(design: dict[str, Any]) -> dict[str, Any]:
    lines_in = design.get("lines")
    if not isinstance(lines_in, list) or not lines_in:
        raise MotionStudioError(
            "equation scene needs a non-empty 'lines' list")
    lines: list[dict[str, Any]] = []
    for i, raw in enumerate(lines_in):
        what = f"equation line #{i}"
        if isinstance(raw, str):
            raw = {"text": raw}
        if not isinstance(raw, dict):
            raise MotionStudioError(f"{what}: must be a dict (or a string)")
        text = str(raw.get("text", "")).strip()
        if not text:
            raise MotionStudioError(f"{what}: empty 'text'")
        step, at = _step_of(raw, i, what)  # each line its own step
        lines.append({"text": text, "note": str(raw.get("note", "")),
                      "box": bool(raw.get("box", False)),
                      "step": step, "at": at})
    return {"lines": lines, "steps": _steps_of(lines)}


def _validate_chart(design: dict[str, Any]) -> dict[str, Any]:
    chart = str(design.get("chart", "bar"))
    if chart not in CHART_KINDS:
        raise MotionStudioError(
            f"chart scene: 'chart' must be one of {CHART_KINDS}, "
            f"got {chart!r}")
    series_in = design.get("series")
    if not isinstance(series_in, list) or not series_in:
        raise MotionStudioError(
            "chart scene needs a non-empty 'series' list")
    series: list[dict[str, Any]] = []
    for i, raw in enumerate(series_in):
        what = f"chart point #{i}"
        if not isinstance(raw, dict):
            raise MotionStudioError(f"{what}: must be a dict")
        try:
            value = float(raw["value"])
        except (KeyError, TypeError, ValueError):
            raise MotionStudioError(
                f"{what}: needs a numeric 'value'") from None
        if value < 0:
            raise MotionStudioError(
                f"{what}: values must be ≥ 0, got {value}")
        step, at = _step_of(raw, i, what)  # one build step per point
        series.append({"label": str(raw.get("label", f"#{i + 1}")),
                       "value": value,
                       "color": _color_of(raw, i, what),
                       "step": step, "at": at})
    ymax_raw = design.get("ymax")
    if ymax_raw is not None:
        ymax = float(ymax_raw)
        if ymax <= 0:
            raise MotionStudioError(
                f"chart scene: 'ymax' must be > 0, got {ymax}")
    else:
        top = max(p["value"] for p in series)
        ymax = top * 1.15 if top > 0 else 1.0
    return {"chart": chart, "series": series,
            "ylabel": str(design.get("ylabel", "")), "ymax": ymax,
            "steps": _steps_of(series)}


def validate_design(design: dict[str, Any]) -> dict[str, Any]:
    """Validate + normalize an explainer scene design (pure data).

    Returns the normalized spec the frame functions consume. Raises
    :class:`MotionStudioError` naming the offending field — scenes
    are authored by the brain as data, so the error IS the authoring
    feedback.
    """
    if not isinstance(design, dict):
        raise MotionStudioError("explainer design must be a dict")
    kind = explainer_kind(design)
    spec: dict[str, Any] = {"kind": kind,
                            "title": str(design.get("title", ""))}
    if kind == "diagram":
        spec.update(_validate_diagram(design))
    elif kind == "equation":
        spec.update(_validate_equation(design))
    else:
        spec.update(_validate_chart(design))
    return spec


# ---------------------------------------------------------------------------
# step timing — windows over the segment, the studio's smooth ease
# ---------------------------------------------------------------------------

def _window(el: dict[str, Any], n_steps: int, duration: float
            ) -> tuple[float, float]:
    step_len = duration / max(1, n_steps)
    if float(el.get("at", -1.0)) >= 0:
        t0 = float(el["at"])
        return t0, t0 + step_len
    step = int(el.get("step", 0))
    return step * step_len, (step + 1) * step_len


def _progress(el: dict[str, Any], n_steps: int, duration: float,
              t: float) -> float:
    t0, t1 = _window(el, n_steps, duration)
    if t <= t0:
        return 0.0
    if t >= t1 or t1 <= t0:
        return 1.0
    return ease("smooth", (t - t0) / (t1 - t0))


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int],
          p: float) -> tuple[int, int, int]:
    p = max(0.0, min(1.0, p))
    return tuple(int(round(x + (y - x) * p)) for x, y in zip(a, b))  # type: ignore[return-value]


def _fmt_value(v: float) -> str:
    return f"{v:g}" if v == int(v) and abs(v) < 1e15 else f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# the scene — deterministic frame functions per kind
# ---------------------------------------------------------------------------

class _Scene:
    """One validated explainer scene at one canvas size.

    Everything static (background, title, chart axes) is baked once;
    a frame is the baked base + the elements whose build progress at
    ``t`` is > 0. Pure in (spec, size, t) — the determinism the
    golden harness pins.
    """

    def __init__(self, spec: dict[str, Any], size: tuple[int, int],
                 words: Sequence[Word]):
        if Image is None:
            raise MotionStudioError(
                "explainer rendering needs Pillow")
        self.spec = spec
        self.w, self.h = int(size[0]), int(size[1])
        self.words = list(words)
        m = min(self.w, self.h)
        self.f_title = find_font(max(16, int(m * 0.062)),
                                 "DejaVuSans-Bold.ttf")
        self.f_body = find_font(max(12, int(m * 0.046)),
                                "DejaVuSans.ttf")
        self.f_bold = find_font(max(12, int(m * 0.046)),
                                "DejaVuSans-Bold.ttf")
        self.f_eq = find_font(max(14, int(m * 0.058)),
                              "DejaVuSans-Bold.ttf")
        self.f_small = find_font(max(10, int(m * 0.033)),
                                 "DejaVuSans.ttf")
        self._base = self._bake_base()
        self._caption_lines = self._layout_caption()

    # -- static layers ---------------------------------------------------
    def _gradient(self):
        img = Image.new("RGB", (self.w, self.h), BG_TOP)
        px = img.load()
        for y in range(self.h):
            c = _lerp(BG_TOP, BG_BOTTOM, y / max(1, self.h - 1))
            for x in range(self.w):
                px[x, y] = c
        return img

    def _bake_base(self):
        img = self._gradient()
        d = ImageDraw.Draw(img)
        title = self.spec.get("title", "")
        if title:
            tw, _ = text_size(d, title, self.f_title)
            d.text(((self.w - tw) / 2, self.h * 0.045), title,
                   font=self.f_title, fill=INK)
        if self.spec["kind"] == "chart":
            self._draw_axes(d)
        return img

    def _plot(self) -> tuple[float, float, float, float]:
        """Chart plot area (x0, y0, x1, y1) in pixels."""
        return (self.w * 0.13, self.h * 0.20,
                self.w * 0.95, self.h * 0.78)

    def _draw_axes(self, d) -> None:
        x0, y0, x1, y1 = self._plot()
        ymax = float(self.spec["ymax"])
        for i in range(5):
            frac = i / 4
            y = y1 - frac * (y1 - y0)
            d.line([(x0, y), (x1, y)], fill=GRID, width=1)
            label = _fmt_value(ymax * frac)
            tw, th = text_size(d, label, self.f_small)
            d.text((x0 - tw - 8, y - th / 2), label,
                   font=self.f_small, fill=DIM)
        d.line([(x0, y0), (x0, y1)], fill=DIM, width=2)
        d.line([(x0, y1), (x1, y1)], fill=DIM, width=2)
        ylabel = self.spec.get("ylabel", "")
        if ylabel:
            d.text((x0, y0 - self.h * 0.055), ylabel,
                   font=self.f_small, fill=DIM)

    # -- caption (narration on the deterministic text path) ----------------
    def _layout_caption(self) -> list[list[Word]]:
        if not self.words:
            return []
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        max_w = self.w * 0.92
        lines: list[list[Word]] = []
        cur: list[Word] = []
        cur_w = 0.0
        space_w, _ = text_size(probe, " ", self.f_body)
        for word in self.words:
            ww, _ = text_size(probe, word.text, self.f_body)
            add = ww if not cur else ww + space_w
            if cur and cur_w + add > max_w:
                lines.append(cur)
                cur, cur_w = [word], ww
            else:
                cur.append(word)
                cur_w += add
        if cur:
            lines.append(cur)
        return lines

    def _draw_caption(self, d, t: float) -> None:
        if not self._caption_lines:
            return
        _, line_h = text_size(d, "Ag", self.f_body)
        line_h = int(line_h * 1.5)
        total_h = line_h * len(self._caption_lines)
        y = self.h - total_h - int(self.h * 0.035)
        space_w, _ = text_size(d, " ", self.f_body)
        for line in self._caption_lines:
            widths = [text_size(d, wd.text, self.f_body)[0]
                      for wd in line]
            x = (self.w - (sum(widths)
                           + space_w * (len(line) - 1))) / 2
            for wd, ww in zip(line, widths):
                if t >= wd.start:
                    current = t < wd.end
                    d.text((x, y), wd.text, font=self.f_body,
                           fill=ACCENT if current else INK)
                x += ww + space_w
            y += line_h

    # -- diagram -----------------------------------------------------------
    def _node_geom(self, node: dict[str, Any], d) -> dict[str, Any]:
        cx, cy = node["x"] * self.w, node["y"] * self.h
        tw, th = text_size(d, node["label"], self.f_bold)
        m = min(self.w, self.h)
        hw = max(tw / 2 + m * 0.045, m * 0.085)
        hh = max(th / 2 + m * 0.028, m * 0.042)
        return {"cx": cx, "cy": cy, "hw": hw, "hh": hh,
                "r": max(hw, hh)}

    @staticmethod
    def _boundary(geom: dict[str, Any], shape: str,
                  ux: float, uy: float) -> float:
        if shape == "circle":
            return float(geom["r"])
        dists = []
        if abs(ux) > 1e-9:
            dists.append(geom["hw"] / abs(ux))
        if abs(uy) > 1e-9:
            dists.append(geom["hh"] / abs(uy))
        return min(dists) if dists else 0.0

    def _frame_diagram(self, img, t: float, duration: float) -> None:
        d = ImageDraw.Draw(img)
        spec = self.spec
        n_steps = int(spec["steps"])
        geoms = {n["id"]: self._node_geom(n, d) for n in spec["nodes"]}
        width = max(2, int(min(self.w, self.h) * 0.007))
        for edge in spec["edges"]:
            p = _progress(edge, n_steps, duration, t)
            if p <= 0:
                continue
            na = next(n for n in spec["nodes"] if n["id"] == edge["a"])
            nb = next(n for n in spec["nodes"] if n["id"] == edge["b"])
            ga, gb = geoms[edge["a"]], geoms[edge["b"]]
            dx, dy = gb["cx"] - ga["cx"], gb["cy"] - ga["cy"]
            dist = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dist, dy / dist
            da = self._boundary(ga, na["shape"], ux, uy)
            db = self._boundary(gb, nb["shape"], ux, uy)
            x0, y0 = ga["cx"] + ux * da, ga["cy"] + uy * da
            x1, y1 = gb["cx"] - ux * db, gb["cy"] - uy * db
            ex, ey = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
            color = _lerp(BG_TOP, edge["color"], 0.25 + 0.75 * p)
            d.line([(x0, y0), (ex, ey)], fill=color, width=width)
            if p > 0.03:  # arrowhead at the drawn tip
                ang = math.atan2(ey - y0, ex - x0)
                ah = max(7.0, min(self.w, self.h) * 0.03)
                for sgn in (1, -1):
                    a2 = ang + sgn * math.radians(155)
                    d.line([(ex, ey),
                            (ex + ah * math.cos(a2),
                             ey + ah * math.sin(a2))],
                           fill=color, width=width)
            if edge["label"] and p >= 0.55:
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                tw, th = text_size(d, edge["label"], self.f_small)
                d.text((mx - tw / 2, my - th - 4), edge["label"],
                       font=self.f_small,
                       fill=_lerp(BG_TOP, DIM, (p - 0.55) / 0.45))
        for node in spec["nodes"]:
            p = _progress(node, n_steps, duration, t)
            if p <= 0:
                continue
            g = geoms[node["id"]]
            scale = 0.8 + 0.2 * p
            fill = _lerp(BG_TOP, node["color"], 0.30 * p)
            border = _lerp(BG_TOP, node["color"], p)
            box = [g["cx"] - g["hw"] * scale, g["cy"] - g["hh"] * scale,
                   g["cx"] + g["hw"] * scale, g["cy"] + g["hh"] * scale]
            if node["shape"] == "circle":
                d.ellipse(box, fill=fill, outline=border, width=width)
            else:
                d.rounded_rectangle(box, radius=int(g["hh"] * 0.35),
                                    fill=fill, outline=border,
                                    width=width)
            if p > 0.35:
                tw, th = text_size(d, node["label"], self.f_bold)
                d.text((g["cx"] - tw / 2, g["cy"] - th / 2),
                       node["label"], font=self.f_bold,
                       fill=_lerp(BG_TOP, INK, (p - 0.35) / 0.65))

    # -- equation ------------------------------------------------------------
    def _frame_equation(self, img, t: float, duration: float) -> None:
        d = ImageDraw.Draw(img)
        spec = self.spec
        n_steps = int(spec["steps"])
        progresses = [_progress(ln, n_steps, duration, t)
                      for ln in spec["lines"]]
        current = max((i for i, p in enumerate(progresses) if p > 0),
                      default=-1)
        _, glyph_h = text_size(d, "Ag", self.f_eq)
        block_h = glyph_h * 2.05
        y = self.h * 0.21
        x = self.w * 0.09
        for i, line in enumerate(spec["lines"]):
            p = progresses[i]
            if p > 0:
                if line["box"] and i == current:
                    tw, _ = text_size(d, line["text"], self.f_eq)
                    d.rounded_rectangle(
                        [x - 14, y - 8, x + tw + 14, y + glyph_h + 10],
                        radius=8, outline=_lerp(BG_TOP, ACCENT, p),
                        width=2)
                color = INK if i == current else DIM
                d.text((x, y), line["text"], font=self.f_eq,
                       fill=_lerp(BG_TOP, color, p))
                if line["note"]:
                    nw, nh = text_size(d, line["note"], self.f_small)
                    d.text((self.w * 0.91 - nw, y + glyph_h + 4),
                           line["note"], font=self.f_small,
                           fill=_lerp(BG_TOP, ACCENT, p))
            y += block_h

    # -- chart ---------------------------------------------------------------
    def _frame_chart(self, img, t: float, duration: float) -> None:
        d = ImageDraw.Draw(img)
        spec = self.spec
        n_steps = int(spec["steps"])
        series = spec["series"]
        x0, y0, x1, y1 = self._plot()
        plot_w, plot_h = x1 - x0, y1 - y0
        ymax = float(spec["ymax"])
        n = len(series)
        slot = plot_w / n
        progresses = [_progress(pt, n_steps, duration, t)
                      for pt in series]
        if spec["chart"] == "bar":
            for i, pt in enumerate(series):
                p = progresses[i]
                if p <= 0:
                    continue
                cx = x0 + slot * (i + 0.5)
                bw = slot * 0.55
                top = y1 - plot_h * (pt["value"] / ymax) * p
                d.rectangle([cx - bw / 2, top, cx + bw / 2, y1],
                            fill=pt["color"])
                if p > 0.5:
                    label = _fmt_value(pt["value"])
                    tw, th = text_size(d, label, self.f_bold)
                    d.text((cx - tw / 2, top - th - 5), label,
                           font=self.f_bold,
                           fill=_lerp(BG_TOP, INK, (p - 0.5) / 0.5))
                tw, th = text_size(d, pt["label"], self.f_small)
                d.text((cx - tw / 2, y1 + 7), pt["label"],
                       font=self.f_small,
                       fill=_lerp(BG_TOP, DIM, p))
        else:  # line: each segment draws across its point's window
            pts = [(x0 + slot * (i + 0.5),
                    y1 - plot_h * (pt["value"] / ymax))
                   for i, pt in enumerate(series)]
            width = max(2, int(min(self.w, self.h) * 0.008))
            for i, pt in enumerate(series):
                p = progresses[i]
                if p <= 0:
                    continue
                cx, cy = pts[i]
                if i > 0:
                    px, py = pts[i - 1]
                    ex, ey = px + (cx - px) * p, py + (cy - py) * p
                    d.line([(px, py), (ex, ey)], fill=pt["color"],
                           width=width)
                r = width * 1.6
                d.ellipse([cx - r, cy - r, cx + r, cy + r],
                          fill=pt["color"])
                if p > 0.5:
                    label = _fmt_value(pt["value"])
                    tw, th = text_size(d, label, self.f_bold)
                    d.text((cx - tw / 2, cy - th - 6), label,
                           font=self.f_bold,
                           fill=_lerp(BG_TOP, INK, (p - 0.5) / 0.5))
                tw, th = text_size(d, pt["label"], self.f_small)
                d.text((cx - tw / 2, y1 + 7), pt["label"],
                       font=self.f_small,
                       fill=_lerp(BG_TOP, DIM, p))

    # -- frame ---------------------------------------------------------------
    def frame(self, t: float, duration: float):
        img = self._base.copy()
        kind = self.spec["kind"]
        if kind == "diagram":
            self._frame_diagram(img, t, duration)
        elif kind == "equation":
            self._frame_equation(img, t, duration)
        else:
            self._frame_chart(img, t, duration)
        d = ImageDraw.Draw(img)
        self._draw_caption(d, t)
        return img


def frame_image(design: dict[str, Any], size: tuple[int, int],
                t: float, duration: float, *,
                text: str = "",
                words: Sequence | None = None,
                timing_source: str = ""):
    """One explainer frame as a PIL image — the pure frame function.

    Exposed for tests and stills: frame(spec, size, t) is the whole
    contract; :func:`render_explainer` is this, sequenced + encoded.
    """
    spec = validate_design(design)
    wlist = _caption_words(text=text, words=words,
                           timing_source=timing_source)
    return _Scene(spec, size, wlist).frame(float(t), float(duration))


def _caption_words(*, text: str, words: Sequence | None,
                   timing_source: str) -> list[Word]:
    if words:
        return to_words(words, timing_source=timing_source)
    if text:
        return to_words(text, timing_source=timing_source)
    return []


def render_explainer(design: dict[str, Any],
                     out: str | os.PathLike, *,
                     duration: float, size: tuple[int, int],
                     fps: float, seed: int = 21,
                     text: str = "",
                     words: Sequence | None = None,
                     timing_source: str = "") -> dict[str, Any]:
    """Render one explainer scene to ``out``; return the record.

    Deterministic: the record states the kind, the build-step count
    and the caption timing provenance — the compile record in
    :mod:`.project` carries all three into the render's rungs.
    """
    if np is None:
        raise MotionStudioError("explainer rendering needs numpy")
    spec = validate_design(design)
    wlist = _caption_words(text=text, words=words,
                           timing_source=timing_source)
    scene = _Scene(spec, size, wlist)
    grid = FrameGrid(fps)
    n_frames = max(1, grid.frames(duration))
    defaults = profile_defaults()
    render_sequence(
        n_frames, int(size[0]), int(size[1]), float(fps), out,
        lambda i, t: np.asarray(
            scene.frame(t, float(duration)).convert("RGB")),
        crf=int(defaults.get("crf", 20)),
        preset=str(defaults.get("preset", "veryfast")))
    counts = {"diagram": lambda: {"nodes": len(spec["nodes"]),
                                  "edges": len(spec["edges"])},
              "equation": lambda: {"lines": len(spec["lines"])},
              "chart": lambda: {"points": len(spec["series"])}}[spec["kind"]]()
    return {"engine": str(design.get("engine", "")),
            "kind": spec["kind"], "title": spec["title"],
            "steps": int(spec["steps"]), "frames": n_frames,
            "elements": counts,
            "timing_sources": timing_sources(wlist),
            "note": (f"{spec['kind']} build: {int(spec['steps'])} "
                     f"steps, deterministic frame functions")}
