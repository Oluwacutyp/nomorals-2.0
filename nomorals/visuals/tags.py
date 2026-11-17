"""The [POINT] tag protocol + the full annotation tag vocabulary.

The brain emits declarative tags inline in its normal output stream —
no separate tool-call round trip — and this module parses them into
typed tags the render layer consumes, then strips them from the text
so they are never spoken aloud or shown raw.

The vocabulary is the union of the two references, kept as data:

- farzaa/clicky: ``[POINT:x,y:label]``, ``[POINT:x,y:label:screenN]``,
  ``[POINT:none]`` — one pointing tag, coordinates in the frame the
  brain saw, screen suffix for multi-monitor, ``none`` when pointing
  would not help.
- clickyX ``guidance.rs``: ``[RECT:x,y,w,h:label]``,
  ``[HIGHLIGHT:x,y,w,h:label]``, ``[SCRIBBLE:x1,y1;x2,y2;...:label]``,
  ``[SHAPE:arrow|curve:x1,y1:x2,y2:label]``.
- Devon's own addition: ``[CAPTION:x,y:text]`` — a text caption pinned
  to a coordinate (the spec already has a ``caption`` annotation kind;
  the references had no way to emit one).

Better than either reference: tags match anywhere in the text and
several may appear in one reply (farzaa anchors one tag to the very
end); coordinates arrive in the frame's pixels and are normalized to
the scene spec's 0..1 space here, clamped into the frame the way
farzaa clamps (never rejected for being out of bounds); a malformed
tag is left in the text untouched and never crashes the reply.

Annotation is never actuation: a parsed tag places a marker on the
scene. Nothing in this module — or anywhere in the organ — moves a
mouse or clicks. ClickyX's POINT-executes-a-click conflation is
explicitly not inherited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ParsedTags",
    "Tag",
    "parse_tags",
    "strip_tags",
    "tags_to_annotations",
]

#: Tag kinds, in the same words as the scene spec's annotation kinds.
TAG_KINDS = ("point", "rect", "highlight", "scribble", "shape",
             "caption")

_NUM = r"(\d+(?:\.\d+)?)"
_LABEL = r"(?::([^\]:]*?))?"


def _tag_re(name: str, body: str) -> re.Pattern[str]:
    return re.compile(r"\[" + name + ":" + body + r"\]")


_POINT_RE = _tag_re(
    "POINT",
    r"(?:none|" + _NUM + r"\s*,\s*" + _NUM + _LABEL
    + r"(?::screen(\d+))?)")
_RECT_RE = _tag_re(
    "RECT",
    _NUM + r"\s*,\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*,\s*" + _NUM
    + _LABEL)
_HIGHLIGHT_RE = _tag_re(
    "HIGHLIGHT",
    _NUM + r"\s*,\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*,\s*" + _NUM
    + _LABEL)
_SCRIBBLE_RE = _tag_re(
    "SCRIBBLE",
    r"((?:\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*;?\s*)+)" + _LABEL)
_SHAPE_RE = _tag_re(
    "SHAPE",
    r"(arrow|curve):" + _NUM + r"\s*,\s*" + _NUM + r"\s*:\s*"
    + _NUM + r"\s*,\s*" + _NUM + _LABEL)
_CAPTION_RE = _tag_re(
    "CAPTION",
    _NUM + r"\s*,\s*" + _NUM + r"(?::(.+?))?")


@dataclass(frozen=True)
class Tag:
    """One parsed annotation tag, in scene-spec (0..1) coordinates."""

    kind: str
    x: float
    y: float
    w: float = 0.0
    h: float = 0.0
    points: tuple[tuple[float, float], ...] = ()
    shape: str = ""
    label: str = ""
    screen: int = 1


@dataclass
class ParsedTags:
    """The reply with its tags cut out, and the tags themselves."""

    text: str
    tags: list[Tag] = field(default_factory=list)
    none: bool = False


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _norm(value: float, extent: float | None) -> float | None:
    """A tag number into 0..1: pixels against the frame (clamped,
    the farzaa rule), or already-normalized when no frame is given —
    in which case a number outside 0..1 is not a coordinate at all
    and the tag is malformed (``None``)."""
    if extent is None:
        return value if 0.0 <= value <= 1.0 else None
    return _clamp01(value / extent) if extent > 0 else 0.0


def _label_of(raw: str | None) -> str:
    return (raw or "").strip()


def parse_tags(text: str, *,
               width: float | None = None,
               height: float | None = None,
               screen_count: int = 1) -> ParsedTags:
    """Parse every well-formed tag out of ``text``.

    ``width``/``height`` are the frame the brain saw, in pixels —
    tag numbers are pixels in that frame. Without a frame the numbers
    must already be 0..1. ``screen_count`` bounds the ``:screenN``
    suffix; an out-of-range screen falls back to screen 1, the
    cursor screen. Malformed tags are left in the text and ignored.
    """
    if not isinstance(text, str) or not text:
        return ParsedTags(text=text if isinstance(text, str) else "")
    tags: list[Tag] = []
    none = False
    screens = max(1, int(screen_count or 1))
    # Only tags that actually parse are cut out of the text. A tag
    # that matches the shape but carries an out-of-range coordinate
    # (no frame given) stays visible in the reply — honest, and the
    # brain sees it next turn instead of losing it silently.
    spans: list[tuple[int, int]] = []

    def _screen(raw: str | None) -> int:
        if not raw:
            return 1
        try:
            number = int(raw)
        except ValueError:
            return 1
        return number if 1 <= number <= screens else 1

    for match in _POINT_RE.finditer(text):
        x_raw, y_raw, label, screen = match.groups()
        if x_raw is None:
            none = True
            spans.append(match.span())
            continue
        x = _norm(float(x_raw), width)
        y = _norm(float(y_raw), height)
        if x is None or y is None:
            continue
        tags.append(Tag(
            kind="point",
            x=x,
            y=y,
            label=_label_of(label),
            screen=_screen(screen),
        ))
        spans.append(match.span())
    for match in _RECT_RE.finditer(text):
        x_raw, y_raw, w_raw, h_raw, label = match.groups()
        x = _norm(float(x_raw), width)
        y = _norm(float(y_raw), height)
        w = _norm(float(w_raw), width)
        h = _norm(float(h_raw), height)
        if None in (x, y, w, h):
            continue
        tags.append(Tag(
            kind="rect",
            x=x,
            y=y,
            w=w,
            h=h,
            label=_label_of(label),
        ))
        spans.append(match.span())
    for match in _HIGHLIGHT_RE.finditer(text):
        x_raw, y_raw, w_raw, h_raw, label = match.groups()
        x = _norm(float(x_raw), width)
        y = _norm(float(y_raw), height)
        w = _norm(float(w_raw), width)
        h = _norm(float(h_raw), height)
        if None in (x, y, w, h):
            continue
        tags.append(Tag(
            kind="highlight",
            x=x,
            y=y,
            w=w,
            h=h,
            label=_label_of(label),
        ))
        spans.append(match.span())
    for match in _SCRIBBLE_RE.finditer(text):
        raw_points, label = match.groups()
        points: list[tuple[float, float]] = []
        bad = False
        for pair in raw_points.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            parts = pair.split(",")
            if len(parts) != 2:
                continue
            try:
                px, py = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            nx = _norm(px, width)
            ny = _norm(py, height)
            if nx is None or ny is None:
                bad = True
                break
            points.append((nx, ny))
        if bad or not points:
            continue
        tags.append(Tag(
            kind="scribble",
            x=points[0][0], y=points[0][1],
            points=tuple(points),
            label=_label_of(label),
        ))
        spans.append(match.span())
    for match in _SHAPE_RE.finditer(text):
        shape, x1, y1, x2, y2, label = match.groups()
        p1x = _norm(float(x1), width)
        p1y = _norm(float(y1), height)
        p2x = _norm(float(x2), width)
        p2y = _norm(float(y2), height)
        if None in (p1x, p1y, p2x, p2y):
            continue
        tags.append(Tag(
            kind="shape",
            x=p1x,
            y=p1y,
            points=((p1x, p1y),
                    (p2x, p2y)),
            shape=shape,
            label=_label_of(label),
        ))
        spans.append(match.span())
    for match in _CAPTION_RE.finditer(text):
        x_raw, y_raw, label = match.groups()
        x = _norm(float(x_raw), width)
        y = _norm(float(y_raw), height)
        if x is None or y is None:
            continue
        tags.append(Tag(
            kind="caption",
            x=x,
            y=y,
            label=_label_of(label),
        ))
        spans.append(match.span())
    clean = text
    for start, end in sorted(spans, reverse=True):
        clean = clean[:start] + clean[end:]
    return ParsedTags(text=clean, tags=tags, none=none)


def strip_tags(text: str) -> str:
    """The reply with every parsed tag cut out.

    Same rule as :func:`parse_tags`: a tag that does not parse stays
    in the text — stripping never silently drops a malformed tag.
    """
    if not isinstance(text, str):
        return ""
    return parse_tags(text).text


#: Tag kind -> scene-spec overlay annotation kind.
_TAG_TO_ANNOTATION = {
    "point": "cursor",
    "rect": "rect",
    "highlight": "highlight",
    "scribble": "scribble",
    "shape": "shape",
    "caption": "caption",
}


def tags_to_annotations(tags: list[Tag]) -> list[dict[str, Any]]:
    """Tags into overlay annotation dicts for :func:`validate_scene_spec`.

    Sizes, paths and shapes ride along — the spec validator carries
    them (``w``/``h``/``points``/``shape``/``screen``) so a rect drawn
    from a tag keeps the size the brain authored.
    """
    out: list[dict[str, Any]] = []
    for tag in tags:
        ann: dict[str, Any] = {
            "kind": _TAG_TO_ANNOTATION.get(tag.kind, "cursor"),
            "x": tag.x,
            "y": tag.y,
            "text": tag.label,
            "screen": tag.screen,
        }
        if tag.w > 0:
            ann["w"] = tag.w
        if tag.h > 0:
            ann["h"] = tag.h
        if tag.points:
            ann["points"] = [[px, py] for px, py in tag.points]
        if tag.shape:
            ann["shape"] = tag.shape
        out.append(ann)
    return out
