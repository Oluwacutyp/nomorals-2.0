"""The animated rung of emoji creation: her studio designs,
in motion — a deterministic, stdlib-only TGS writer.

Why this exists (owner, 2026-11-11, live): "Telegram emojis
still whack… Telegram has a way where emojis are animated…
Still using still emoji's."  He was right on both counts.
Telegram custom-emoji packs carry an ``animated`` format —
a .tgs file (gzipped Lottie JSON, 100×100 canvas, ≤3 seconds,
looped, ≤64 KB) that animates INLINE in message text — but
her pack registration hardcoded ``format: "static"``, so
every emoji she created was a still.

This module is the missing rung: it re-expresses a studio
design (:mod:`emoji_studio` — the same spec dict, the same
palette, the same proportions) as Lottie vector shapes and
keyframing one of four motions over them (pulse / bounce /
spin / wiggle, chosen by the design, varied by its seed).
The animation is OF her art — the heart she drew beats, the
rocket she drew bounces, the star she drew spins — never a
generic overlay.  Lottie text layers need font files, so the
letter designs are drawn from a built-in 5×7 block glyph
table as rect shapes: shapes only, no external assets, no
fonts, no Pillow, no network — pure stdlib, deterministic:
the same design animates the same way on any host.

The size law is enforced, not hoped for: the gzipped bytes
must fit Telegram's 64 KB animated-emoji cap; the writer
simplifies in steps (sparkles shed, paths resampled, detail
dropped) and raises :class:`EmojiAnimTooLarge` when even the
leanest honest version will not fit — the caller then falls
back to the static rung and records THAT, never a fake
"animated" claim.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from pathlib import Path
from typing import Any

from . import emoji_studio

__all__ = [
    "CANVAS", "FRAME_RATE", "DURATION_FRAMES", "MAX_TGS_BYTES",
    "MOTIONS", "EmojiAnimError", "EmojiAnimTooLarge",
    "design_motion", "lottie_for_design", "tgs_bytes",
    "animated_emoji_tgs",
]

#: Telegram's animated custom-emoji canvas, exactly.
CANVAS = 100
#: TGS plays at 60 fps; her loop is 2 s of it (the cap is 3 s).
FRAME_RATE = 60
DURATION_FRAMES = 120
#: Telegram refuses an animated emoji file over 64 KB.
MAX_TGS_BYTES = 64 * 1024

#: The motion vocabulary — every created emoji moves in one
#: of these four ways, picked by its design (see
#: :func:`design_motion`).
MOTIONS = ("pulse", "bounce", "spin", "wiggle")

_INK = "#26221f"  # the studio's face ink


class EmojiAnimError(RuntimeError):
    """An animated emoji could not be built at all."""


class EmojiAnimTooLarge(EmojiAnimError):
    """Even the leanest honest animation exceeds the 64 KB
    TGS cap — the caller falls back to the static rung and
    records the format truthfully."""


# ── the letter rung: a 5×7 block glyph table ─────────────────
#: Lottie text layers need font files (an external asset —
#: banned on this rung), so letter designs are drawn as rect
#: shapes from this table: glyph → 7 rows of 5 bits, top row
#: first, "1" = a filled cell.  Block letters, readable at
#: 100×100 — the emoji rule.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11100", "10010", "10001", "10001", "10001", "10010", "11100"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "10001", "11001", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "01010", "00100", "00100", "00100", "01010", "10001"),
    "Y": ("10001", "01010", "00100", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "+": ("00000", "00100", "00100", "01110", "00100", "00100", "00000"),
}


# ── Lottie primitives ────────────────────────────────────────

def _rgb(hex_color: str) -> list[float]:
    h = str(hex_color or "#000000").lstrip("#")
    return [round(int(h[i:i + 2], 16) / 255.0, 3) for i in (0, 2, 4)] \
        + [1.0]


def _fill(hex_color: str) -> dict[str, Any]:
    return {"ty": "fl", "c": {"a": 0, "k": _rgb(hex_color)},
            "o": {"a": 0, "k": 100}, "r": 1, "bm": 0, "nm": "fill"}


def _stroke(hex_color: str, width: float) -> dict[str, Any]:
    return {"ty": "st", "c": {"a": 0, "k": _rgb(hex_color)},
            "o": {"a": 0, "k": 100},
            "w": {"a": 0, "k": round(float(width), 2)},
            "lc": 2, "lj": 2, "bm": 0, "nm": "stroke"}


def _group(items: list[dict[str, Any]], name: str) -> dict[str, Any]:
    body = list(items) + [{
        "ty": "tr", "p": {"a": 0, "k": [0, 0]},
        "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]},
        "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100},
        "sk": 0, "sa": 0, "nm": "tr"}]
    return {"ty": "gr", "it": body, "nm": name,
            "np": len(body), "bm": 0}


def _path(points: list[tuple[float, float]], *,
          closed: bool = True) -> dict[str, Any]:
    verts = [[round(float(x), 2), round(float(y), 2)]
             for x, y in points]
    zeros = [[0, 0] for _ in verts]
    return {"ty": "sh",
            "ks": {"a": 0, "k": {"i": zeros, "o": [list(z) for z in zeros],
                                 "v": verts, "c": bool(closed)}},
            "nm": "path"}


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> dict[str, Any]:
    return {"ty": "el", "p": {"a": 0, "k": [round(cx, 2), round(cy, 2)]},
            "s": {"a": 0, "k": [round(rx * 2, 2), round(ry * 2, 2)]},
            "nm": "ellipse"}


def _rect(cx: float, cy: float, w: float, h: float) -> dict[str, Any]:
    return {"ty": "rc", "p": {"a": 0, "k": [round(cx, 2), round(cy, 2)]},
            "s": {"a": 0, "k": [round(w, 2), round(h, 2)]},
            "r": {"a": 0, "k": 0}, "nm": "rect"}


def _poly(points: list[tuple[float, float]], fill: str, *,
          stroke: str = "", stroke_w: float = 2.0,
          name: str = "poly") -> dict[str, Any]:
    items: list[dict[str, Any]] = [_path(points)]
    if fill:
        items.append(_fill(fill))
    if stroke:
        items.append(_stroke(stroke, stroke_w))
    return _group(items, name)


def _ell(cx: float, cy: float, rx: float, ry: float, fill: str, *,
         stroke: str = "", stroke_w: float = 2.0,
         name: str = "ellipse") -> dict[str, Any]:
    items: list[dict[str, Any]] = [_ellipse(cx, cy, rx, ry)]
    if fill:
        items.append(_fill(fill))
    if stroke:
        items.append(_stroke(stroke, stroke_w))
    return _group(items, name)


def _line(points: list[tuple[float, float]], stroke: str,
          width: float, *, name: str = "line") -> dict[str, Any]:
    return _group([_path(points, closed=False),
                   _stroke(stroke, width)], name)


def _arc_points(cx: float, cy: float, rx: float, ry: float,
                start_deg: float, end_deg: float,
                steps: int) -> list[tuple[float, float]]:
    """Points along an ellipse arc, PIL's own convention:
    0° at 3 o'clock, angles growing clockwise (y is down) —
    the studio's arcs, sampled."""
    pts = []
    for i in range(steps + 1):
        a = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return pts


# ── the design, as vector shapes ─────────────────────────────
#: The composition geometry mirrors emoji_studio.render_master
#: at its 100×100 emoji size: pad 6, the subject in the box
#: (6, 6) → (94, 94), centre (50, 50), s = 44.
_CX = _CY = 50.0
_S = 44.0


def _crescent_points(cx: float, cy: float, r: float,
                     bx: float, by: float, br: float,
                     steps: int) -> list[tuple[float, float]]:
    """The studio moon is a CUT (a disc with an offset disc
    removed); Lottie has no boolean cut on this rung, so the
    crescent is drawn as its true outline: the outer circle's
    arc on the side away from the cut, closed by the cut
    circle's arc on the side inside the disc.  Arc directions
    are chosen by midpoint tests, not by hand-derived angles,
    so the outline is right for any offset."""
    dx, dy = bx - cx, by - cy
    d = math.hypot(dx, dy)
    if d <= 0 or d >= r + br or d <= abs(r - br):
        return _arc_points(cx, cy, r, r, 0, 360, steps)
    a = (r * r - br * br + d * d) / (2 * d)
    h = math.sqrt(max(0.0, r * r - a * a))
    mx, my = cx + a * dx / d, cy + a * dy / d
    px, py = -dy / d * h, dx / d * h
    i1 = (mx + px, my + py)
    i2 = (mx - px, my - py)

    def _arc(cx0: float, cy0: float, r0: float,
             p_from: tuple[float, float], p_to: tuple[float, float],
             keep: Any, n: int) -> list[tuple[float, float]]:
        a0 = math.degrees(math.atan2(p_from[1] - cy0, p_from[0] - cx0))
        a1 = math.degrees(math.atan2(p_to[1] - cy0, p_to[0] - cx0))
        for sweep in (1, -1):
            span = (a1 - a0) * sweep % 360.0
            pts = _arc_points(cx0, cy0, r0, r0, a0,
                              a0 + sweep * span, n)
            mid = pts[len(pts) // 2]
            if keep(mid):
                return pts
        return _arc_points(cx0, cy0, r0, r0, a0, a1, n)

    def _outside_cut(p: tuple[float, float]) -> bool:
        return math.hypot(p[0] - bx, p[1] - by) >= br

    def _inside_disc(p: tuple[float, float]) -> bool:
        return math.hypot(p[0] - cx, p[1] - cy) <= r

    outer = _arc(cx, cy, r, i1, i2, _outside_cut, steps)
    inner = _arc(bx, by, br, i2, i1, _inside_disc, max(6, steps // 2))
    return outer + inner[1:]


def _face_feature_groups(spec: dict[str, Any], cx: float, cy: float,
                         s: float, detail: int) -> list[dict[str, Any]]:
    """The studio's face features (_draw_face_features) as
    shapes: eyes, brows of expression, mouths — ink on the
    face, arcs sampled into stroked paths."""
    face = str(spec.get("face") or "happy")
    ex, ey = s * 0.34, s * 0.10
    r = max(3.0, s * 0.085)
    w = max(3.0, s * 0.07)
    steps = 10 if detail < 2 else 6
    groups: list[dict[str, Any]] = []
    if face == "laugh":
        for dx in (-ex, ex):
            pts = _arc_points(cx + dx, cy - ey, r * 1.7, r * 1.4,
                              15, 165, steps)
            groups.append(_line(pts, _INK, w, name="eye"))
        mouth = _arc_points(cx, cy + s * 0.44, s * 0.34, s * 0.28,
                            0, 180, steps)
        groups.append(_poly(mouth, _INK, name="mouth"))
    elif face == "wink":
        groups.append(_ell(cx - ex, cy - ey, r, r, _INK, name="eye"))
        groups.append(_line([(cx + ex - r * 1.6, cy - ey),
                             (cx + ex + r * 1.6, cy - ey)],
                            _INK, w, name="eye"))
        pts = _arc_points(cx, cy + s * 0.27, s * 0.32, s * 0.25,
                          15, 165, steps)
        groups.append(_line(pts, _INK, w, name="mouth"))
    elif face == "sleepy":
        for dx in (-ex, ex):
            pts = _arc_points(cx + dx, cy - ey, r * 1.7, r * 1.2,
                              195, 345, steps)
            groups.append(_line(pts, _INK, w, name="eye"))
        groups.append(_ell(cx, cy + s * 0.34 + r * 0.9, r * 0.9,
                           r * 0.9, "", stroke=_INK,
                           stroke_w=max(3.0, s * 0.06), name="mouth"))
    elif face == "wow":
        for dx in (-ex, ex):
            groups.append(_ell(cx + dx, cy - ey, r * 1.25, r * 1.25,
                               _INK, name="eye"))
        groups.append(_ell(cx, cy + s * 0.30 + r * 1.5, r * 1.5,
                           r * 1.5, _INK, name="mouth"))
    elif face == "cool":
        groups.append(_group([_rect(cx, cy - ey - r * 0.25,
                                    s * 1.56, r * 2.9),
                              _fill(_INK)], "shades"))
        pts = _arc_points(cx, cy + s * 0.30, s * 0.30, s * 0.20,
                          20, 160, steps)
        groups.append(_line(pts, _INK, w, name="mouth"))
    else:  # happy + default
        for dx in (-ex, ex):
            groups.append(_ell(cx + dx, cy - ey, r, r, _INK,
                               name="eye"))
        pts = _arc_points(cx, cy + s * 0.27, s * 0.32, s * 0.25,
                          15, 165, steps)
        groups.append(_line(pts, _INK, w, name="mouth"))
    return groups


def _letter_groups(spec: dict[str, Any],
                   palette: tuple[str, str, str, str],
                   detail: int) -> list[dict[str, Any]]:
    """A letter design without a font: the glyph as filled
    cells of the 5×7 table — a dark pass expanded by a hair
    under the main pass gives the studio letter its outline."""
    main, dark, _light, _accent = palette
    ch = str(spec.get("letter") or "?")[:1].upper() or "?"
    rows = _GLYPHS.get(ch) or _GLYPHS["?"]
    cell = _S * 1.5 / 7.0
    w_total, h_total = cell * 5, cell * 7
    x0, y0 = _CX - w_total / 2, _CY - h_total / 2
    groups: list[dict[str, Any]] = []
    for grow in (1.9, 1.0):
        color = dark if grow > 1 else main
        items: list[dict[str, Any]] = []
        for ri, row in enumerate(rows):
            for ci, bit in enumerate(row):
                if bit == "1":
                    items.append(_rect(
                        x0 + (ci + 0.5) * cell, y0 + (ri + 0.5) * cell,
                        cell * grow, cell * grow))
        merged = _group(items + [_fill(color)], "glyph")
        groups.append(merged)
    return groups


def _design_groups(spec: dict[str, Any],
                   palette: tuple[str, str, str, str],
                   detail: int) -> list[dict[str, Any]]:
    """One studio design → its Lottie shape groups, in the
    studio's own draw order and proportions (emoji_studio.
    _draw_shape + _draw_face at the 100×100 box).  ``detail``
    is the size-cap ladder: ≥2 sheds highlights and resamples
    the curves, ≥3 sheds the secondary sub-shapes too."""
    main, dark, light, accent = palette
    cx, cy, s = _CX, _CY, _S
    shape = str(spec.get("shape") or "face")
    heart_n = 48 if detail < 2 else (28 if detail < 3 else 18)
    flame_n = 40 if detail < 2 else (24 if detail < 3 else 16)
    arc_n = 36 if detail < 2 else 24
    highlights = detail < 2
    groups: list[dict[str, Any]] = []
    if shape == "heart":
        pts = emoji_studio._heart_points(cx, cy, s * 0.95)
        step = max(1, len(pts) // heart_n)
        groups.append(_poly(pts[::step], main, name="heart"))
        if highlights:
            hi = emoji_studio._heart_points(cx - s * 0.18,
                                            cy - s * 0.22, s * 0.34)
            groups.append(_poly(hi[::2], light, name="highlight"))
    elif shape == "star":
        pts = emoji_studio._star_points(cx, cy, s * 0.98)
        groups.append(_poly(pts, main, stroke=dark, name="star"))
    elif shape == "moon":
        pts = _crescent_points(cx, cy, s, cx + s * 0.55,
                               cy - s * 0.25, s * 0.80, arc_n)
        groups.append(_poly(pts, main, name="moon"))
    elif shape == "sun":
        for i in range(8):
            a = i * math.pi / 4
            tip = (cx + s * 1.02 * math.cos(a),
                   cy + s * 1.02 * math.sin(a))
            base_a = (cx + s * 0.62 * math.cos(a - 0.32),
                      cy + s * 0.62 * math.sin(a - 0.32))
            base_b = (cx + s * 0.62 * math.cos(a + 0.32),
                      cy + s * 0.62 * math.sin(a + 0.32))
            groups.append(_poly([tip, base_a, base_b], dark,
                                name="ray"))
        groups.append(_ell(cx, cy, s * 0.66, s * 0.66, main,
                           name="sun"))
    elif shape == "flame":
        pts = []
        for i in range(flame_n):
            t = 2 * math.pi * i / flame_n
            rr = s * (0.72 + 0.28 * math.sin(t + 1.2))
            squash = 1.0 + 0.45 * math.sin(t - math.pi / 2)
            pts.append((cx + rr * math.cos(t) * 0.78,
                        cy + rr * math.sin(t) * squash))
        groups.append(_poly(pts, main, name="flame"))
        if highlights:
            inner = [(cx + (px - cx) * 0.52,
                      cy + (py - cy) * 0.52 + s * 0.22)
                     for px, py in pts]
            groups.append(_poly(inner, light, name="core"))
    elif shape == "drop":
        groups.append(_poly([(cx, cy - s * 1.05),
                             (cx + s * 0.72, cy + s * 0.28),
                             (cx - s * 0.72, cy + s * 0.28)],
                            main, name="drop"))
        groups.append(_ell(cx, cy + s * 0.36, s * 0.74, s * 0.66,
                           main, name="drop"))
        if highlights:
            groups.append(_ell(cx - s * 0.175, cy + s * 0.46,
                               s * 0.125, s * 0.16, light,
                               name="highlight"))
    elif shape == "bolt":
        groups.append(_poly(
            [(cx + s * 0.28, cy - s * 1.0),
             (cx - s * 0.52, cy + s * 0.18),
             (cx - s * 0.02, cy + s * 0.18),
             (cx - s * 0.28, cy + s * 1.0),
             (cx + s * 0.52, cy - s * 0.18),
             (cx + s * 0.02, cy - s * 0.18)],
            main, stroke=dark, name="bolt"))
    elif shape == "disc":
        groups.append(_ell(cx, cy, s * 0.95, s * 0.95, main,
                           name="disc"))
        groups.append(_ell(cx, cy, s * 0.62, s * 0.62, light,
                           name="ring"))
        groups.append(_ell(cx, cy, s * 0.30, s * 0.30, accent,
                           name="core"))
    elif shape == "flower":
        for i in range(6):
            a = i * math.pi / 3
            px, py = cx + s * 0.52 * math.cos(a), \
                cy + s * 0.52 * math.sin(a)
            groups.append(_ell(px, py, s * 0.42, s * 0.42, light,
                               name="petal"))
        groups.append(_ell(cx, cy, s * 0.34, s * 0.34, main,
                           name="heart"))
    elif shape == "crown":
        groups.append(_poly(
            [(cx - s * 0.95, cy + s * 0.62),
             (cx - s * 0.95, cy - s * 0.30),
             (cx - s * 0.45, cy + s * 0.05),
             (cx, cy - s * 0.62),
             (cx + s * 0.45, cy + s * 0.05),
             (cx + s * 0.95, cy - s * 0.30),
             (cx + s * 0.95, cy + s * 0.62)],
            main, stroke=dark, name="crown"))
        if detail < 3:
            for dx in (-0.62, 0.0, 0.62):
                groups.append(_ell(cx + s * dx, cy + s * 0.38,
                                   s * 0.10, s * 0.10, accent,
                                   name="gem"))
    elif shape == "rocket":
        groups.append(_poly(
            [(cx, cy - s * 1.02), (cx + s * 0.46, cy - s * 0.1),
             (cx + s * 0.46, cy + s * 0.52),
             (cx - s * 0.46, cy + s * 0.52),
             (cx - s * 0.46, cy - s * 0.1)],
            light, name="body"))
        groups.append(_poly(
            [(cx - s * 0.46, cy + s * 0.3),
             (cx - s * 0.92, cy + s * 0.78),
             (cx - s * 0.46, cy + s * 0.62)], dark, name="fin"))
        groups.append(_poly(
            [(cx + s * 0.46, cy + s * 0.3),
             (cx + s * 0.92, cy + s * 0.78),
             (cx + s * 0.46, cy + s * 0.62)], dark, name="fin"))
        groups.append(_ell(cx, cy - s * 0.30, s * 0.22, s * 0.22,
                           main, name="window"))
        groups.append(_poly(
            [(cx - s * 0.18, cy + s * 0.98), (cx, cy + s * 0.6),
             (cx + s * 0.18, cy + s * 0.98)], main, name="flame"))
    elif shape == "cat":
        groups.append(_ell(cx, cy + s * 0.15, s * 0.82, s * 0.70,
                           main, name="head"))
        groups.append(_poly(
            [(cx - s * 0.78, cy - s * 0.42),
             (cx - s * 0.62, cy - s * 1.0),
             (cx - s * 0.18, cy - s * 0.62)], main, name="ear"))
        groups.append(_poly(
            [(cx + s * 0.78, cy - s * 0.42),
             (cx + s * 0.62, cy - s * 1.0),
             (cx + s * 0.18, cy - s * 0.62)], main, name="ear"))
        groups.extend(_face_feature_groups(
            spec, cx, cy + s * 0.12, s, detail))
    elif shape == "letter":
        groups.extend(_letter_groups(spec, palette, detail))
    else:  # face — the studio's default, unknown shapes included
        groups.append(_ell(cx, cy, s * 0.98, s * 0.98, main,
                           stroke=dark,
                           stroke_w=max(2.0, s * 0.05), name="face"))
        if highlights:
            groups.append(_ell(cx - s * 0.46, cy + s * 0.48,
                               s * 0.16, s * 0.14, light,
                               name="cheek"))
            groups.append(_ell(cx + s * 0.46, cy + s * 0.48,
                               s * 0.16, s * 0.14, light,
                               name="cheek"))
        groups.extend(_face_feature_groups(spec, cx, cy, s, detail))
    return groups


# ── motion ───────────────────────────────────────────────────

#: design → its motion.  The shape decides first (a heart
#: beats, a rocket launches, a flame flickers); a face's
#: expression decides next; anything unlisted still moves —
#: the seed picks, so two invented designs rarely move alike.
_SHAPE_MOTIONS: dict[str, str] = {
    "heart": "pulse", "disc": "pulse", "crown": "pulse",
    "star": "spin", "sun": "spin", "flower": "spin",
    "rocket": "bounce", "drop": "bounce", "letter": "bounce",
    "flame": "wiggle", "bolt": "wiggle", "cat": "wiggle",
    "moon": "wiggle",
}
_FACE_MOTIONS: dict[str, str] = {
    "laugh": "bounce", "happy": "bounce",
    "wow": "pulse",
    "wink": "wiggle", "cool": "wiggle", "sleepy": "wiggle",
}


def design_motion(spec: dict[str, Any]) -> str:
    """The motion one design moves with — deterministic, and
    always one of :data:`MOTIONS`."""
    spec = dict(spec or {})
    shape = str(spec.get("shape") or "face")
    if shape in _SHAPE_MOTIONS:
        return _SHAPE_MOTIONS[shape]
    face = str(spec.get("face") or "")
    if face in _FACE_MOTIONS:
        return _FACE_MOTIONS[face]
    if shape == "face":
        return "bounce"
    return MOTIONS[emoji_studio._seed_for(spec) % len(MOTIONS)]


_EASE = {"i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}}
#: x == y control points make the segment exactly linear —
#: a spin must not pulse.
_LINEAR = {"i": {"x": [0.8], "y": [0.8]}, "o": {"x": [0.2], "y": [0.2]}}


def _keyframes(table: list[tuple[int, list[float]]],
               ease: dict[str, Any] = _EASE) -> dict[str, Any]:
    frames = []
    for i, (t, values) in enumerate(table):
        frame: dict[str, Any] = {
            "t": t, "s": [round(float(v), 2) for v in values]}
        if i < len(table) - 1:
            frame.update(ease)
        frames.append(frame)
    return {"a": 1, "k": frames}


def _subject_transform(spec: dict[str, Any]) -> dict[str, Any]:
    """The subject layer's transform: anchor at the canvas
    centre, and the design's motion keyframed over the loop.
    Every loop closes on its opening value (rotation's 360
    IS its 0), so Telegram's replay never jumps."""
    seed = emoji_studio._seed_for(dict(spec or {}))
    motion = design_motion(spec)
    ks: dict[str, Any] = {
        "o": {"a": 0, "k": 100},
        "a": {"a": 0, "k": [_CX, _CY, 0]},
        "p": {"a": 0, "k": [_CX, _CY, 0]},
        "r": {"a": 0, "k": 0},
        "s": {"a": 0, "k": [100, 100, 100]},
    }
    end = DURATION_FRAMES
    if motion == "pulse":
        peak = 110 + seed % 8
        ks["s"] = _keyframes([
            (0, [100, 100, 100]), (30, [peak, peak, 100]),
            (60, [100, 100, 100]),
            (90, [100 + (peak - 100) * 0.7] * 2 + [100]),
            (end, [100, 100, 100])])
    elif motion == "bounce":
        lift = 9 + seed % 5
        ks["p"] = _keyframes([
            (0, [_CX, _CY, 0]), (30, [_CX, _CY - lift, 0]),
            (60, [_CX, _CY, 0]), (90, [_CX, _CY - lift * 0.55, 0]),
            (end, [_CX, _CY, 0])])
        ks["s"] = _keyframes([
            (0, [100, 100, 100]), (56, [104, 96, 100]),
            (68, [100, 100, 100]), (end, [100, 100, 100])])
    elif motion == "spin":
        ks["r"] = _keyframes([
            (0, [0]), (30, [90]), (60, [180]), (90, [270]),
            (end, [360])], ease=_LINEAR)
    else:  # wiggle
        amp = 6 + seed % 6
        ks["r"] = _keyframes([
            (0, [0]), (24, [amp]), (48, [-amp]),
            (72, [amp * 0.7]), (96, [-amp * 0.7]), (end, [0])])
    return ks


def _sparkle_layers(spec: dict[str, Any],
                    palette: tuple[str, str, str, str],
                    ) -> list[dict[str, Any]]:
    """Her sparkles, twinkling: the studio scatters five
    4-point stars from the design's seed; here each is its own
    layer breathing in scale + opacity, phase-shifted by its
    index so they never blink in unison."""
    rng = random.Random(emoji_studio._seed_for(dict(spec or {})))
    light = palette[2]
    layers: list[dict[str, Any]] = []
    for i in range(5):
        sx = rng.uniform(0.08, 0.92) * CANVAS
        sy = rng.uniform(0.08, 0.92) * CANVAS
        r = rng.uniform(0.02, 0.045) * CANVAS
        pts = emoji_studio._star_points(0.0, 0.0, r, points=4,
                                        inner=0.22)
        phase = (i * 23) % 60
        scale = _keyframes([
            (0, [70, 70, 100]), ((40 + phase) % 120, [115, 115, 100]),
            ((80 + phase) % 120, [70, 70, 100]),
            (DURATION_FRAMES, [70, 70, 100])])
        # keep the twinkle inside the loop: sort wrapped frames
        scale["k"] = sorted(scale["k"], key=lambda f: f["t"])
        for j, frame in enumerate(scale["k"]):
            if j < len(scale["k"]) - 1:
                frame.update(_EASE)
            else:
                frame.pop("i", None)
                frame.pop("o", None)
        ks = {"o": _keyframes([(0, [60]), (40, [100]), (80, [60]),
                               (DURATION_FRAMES, [60])]),
              "a": {"a": 0, "k": [0, 0, 0]},
              "p": {"a": 0, "k": [round(sx, 1), round(sy, 1), 0]},
              "r": {"a": 0, "k": 0}, "s": scale}
        layers.append({
            "ddd": 0, "ind": 10 + i, "ty": 4, "nm": f"sparkle{i}",
            "sr": 1, "ks": ks, "ao": 0,
            "shapes": [_poly(pts, light, name="sparkle")],
            "ip": 0, "op": DURATION_FRAMES, "st": 0, "bm": 0})
    return layers


# ── composition + the writer ─────────────────────────────────

def lottie_for_design(spec: dict[str, Any], *,
                      detail: int = 0) -> dict[str, Any]:
    """One studio design → its Lottie composition dict:
    100×100, 60 fps, a 2 s loop — Telegram's animated-emoji
    shape.  ``detail`` is the size ladder (see
    :func:`_design_groups`); level 1 sheds the sparkles."""
    spec = dict(spec or {})
    palette = emoji_studio.PALETTES.get(
        str(spec.get("palette") or "gold"), emoji_studio.PALETTES["gold"])
    layers: list[dict[str, Any]] = []
    subject = {
        "ddd": 0, "ind": 1, "ty": 4, "nm": "subject", "sr": 1,
        "ks": _subject_transform(spec), "ao": 0,
        "shapes": _design_groups(spec, palette, detail),
        "ip": 0, "op": DURATION_FRAMES, "st": 0, "bm": 0}
    if spec.get("sparkles") and detail < 1:
        layers.extend(_sparkle_layers(spec, palette))
    layers.append(subject)
    if spec.get("bg") == "disc":
        layers.append({
            "ddd": 0, "ind": 2, "ty": 4, "nm": "badge", "sr": 1,
            "ks": {"o": {"a": 0, "k": 100},
                   "a": {"a": 0, "k": [_CX, _CY, 0]},
                   "p": {"a": 0, "k": [_CX, _CY, 0]},
                   "r": {"a": 0, "k": 0},
                   "s": {"a": 0, "k": [100, 100, 100]}},
            "ao": 0,
            "shapes": [_ell(_CX, _CY, _S, _S, palette[3],
                            name="badge")],
            "ip": 0, "op": DURATION_FRAMES, "st": 0, "bm": 0})
    return {"v": "5.7.4", "fr": FRAME_RATE, "ip": 0,
            "op": DURATION_FRAMES, "w": CANVAS, "h": CANVAS,
            "nm": "devon-emoji", "ddd": 0, "assets": [],
            "layers": layers}


def tgs_bytes(spec: dict[str, Any], *,
              max_bytes: int = MAX_TGS_BYTES) -> bytes:
    """The design's .tgs bytes: gzipped Lottie, guaranteed
    ≤ ``max_bytes`` (Telegram's 64 KB animated cap) or an
    honest :class:`EmojiAnimTooLarge`.  The simplify ladder:
    full detail → sparkles shed → curves resampled, highlights
    dropped → secondary sub-shapes dropped."""
    last_size = 0
    for detail in (0, 1, 2, 3):
        raw = json.dumps(lottie_for_design(spec, detail=detail),
                         separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
        # mtime=0: the bytes are a pure function of the design
        blob = gzip.compress(raw, compresslevel=9, mtime=0)
        last_size = len(blob)
        if last_size <= max_bytes:
            return blob
    raise EmojiAnimTooLarge(
        f"the animated version needs {last_size} bytes even at "
        f"the leanest detail — over the {max_bytes}-byte cap")


def animated_emoji_tgs(spec: dict[str, Any], out_dir: Any, *,
                       name: str = "emoji") -> Path:
    """Write one design's animated emoji file (``<name>.tgs``)
    beside her other expression art and return its path.
    Raises :class:`EmojiAnimTooLarge` when it cannot fit —
    the caller falls back to the static rung, honestly."""
    blob = tgs_bytes(spec)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.tgs"
    path.write_bytes(blob)
    return path
