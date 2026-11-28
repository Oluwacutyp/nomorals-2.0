"""Vector UI icons for the mock-screen renderers (and the
/fakemsg card maker).

Why this module exists: the renderers used to draw UI icons as
font glyphs (``draw.text("☎")``) — but the stocked screen fonts
(Roboto / Inter) do not carry those glyphs, so Pillow substituted
the ``.notdef`` tofu box and the button rendered blank. These
helpers draw the same icons as vectors, in the palette colour the
caller passes, so they render identically on every host and font.

A leaf module on purpose: it imports only Pillow, so the card
maker (``social/chat/makers.py``) can use it too without an import
cycle (the chat renderer imports the maker, never the reverse).

Convention: ``s`` is a scale callable (logical px -> device px) —
pass ``cv.s`` from the mock-screen canvases, an identity scale
from the card maker, which lays out in device pixels already.
``(cx, cy)`` is the icon centre in logical pixels, ``r`` its
half-size. Colours are the caller's palette colour, never
hardcoded: the icon wears the design's own colour.
"""

from __future__ import annotations

import math


def _line(draw, s, pts, color, width: float) -> None:
    draw.line([(s(x), s(y)) for x, y in pts], fill=color,
              width=max(1, s(width)))


def _disc(draw, s, cx: float, cy: float, r: float, color) -> None:
    draw.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)],
                 fill=color)


def _ring(draw, s, cx: float, cy: float, r: float, color,
          width: float) -> None:
    draw.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)],
                 outline=color, width=max(1, s(width)))


def _draw_phone(draw, s, cx: float, cy: float, color, *,
                r: float = 9.0) -> None:
    """A telephone handset: a curved bar from the earpiece
    (top-left) to the mouthpiece (bottom-right), bulging
    toward the bottom-left — the classic handset lean.
    The bar is a polygon sampled along a quadratic curve,
    so it needs no rotation and no font glyph."""
    p0 = (cx - 0.62 * r, cy - 0.62 * r)
    p1 = (cx - 0.85 * r, cy + 0.85 * r)
    p2 = (cx + 0.62 * r, cy + 0.62 * r)
    half = 0.21 * r
    samples = []
    for i in range(13):
        t = i / 12
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] \
            + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] \
            + t ** 2 * p2[1]
        dx = 2 * (1 - t) * (p1[0] - p0[0]) \
            + 2 * t * (p2[0] - p1[0])
        dy = 2 * (1 - t) * (p1[1] - p0[1]) \
            + 2 * t * (p2[1] - p1[1])
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        samples.append((x, y, nx, ny))
    poly = [(s(x + nx * half), s(y + ny * half))
            for x, y, nx, ny in samples]
    poly += [(s(x - nx * half), s(y - ny * half))
             for x, y, nx, ny in reversed(samples)]
    draw.polygon(poly, fill=color)
    for ex, ey in (p0, p2):
        _disc(draw, s, ex, ey, 0.30 * r, color)


def _draw_smiley(draw, s, cx: float, cy: float, color, *,
                 r: float = 9.0) -> None:
    """A smile face: ring + two eyes + smile arc."""
    _ring(draw, s, cx, cy, r, color, 1.6)
    for ex in (-0.35, 0.35):
        _disc(draw, s, cx + ex * r, cy - 0.25 * r, 0.13 * r, color)
    draw.arc([s(cx - 0.55 * r), s(cy - 0.15 * r),
              s(cx + 0.55 * r), s(cy + 0.75 * r)],
             start=20, end=160, fill=color,
             width=max(1, s(1.6)))


def _draw_plus(draw, s, cx: float, cy: float, color, *,
               r: float = 9.0) -> None:
    """A plus: two bars crossing at the centre."""
    _line(draw, s, [(cx - r, cy), (cx + r, cy)], color, 2.2)
    _line(draw, s, [(cx, cy - r), (cx, cy + r)], color, 2.2)


def _draw_close(draw, s, cx: float, cy: float, color, *,
                r: float = 8.0) -> None:
    """A close X: two crossing diagonals."""
    _line(draw, s, [(cx - 0.7 * r, cy - 0.7 * r),
                    (cx + 0.7 * r, cy + 0.7 * r)], color, 2.0)
    _line(draw, s, [(cx - 0.7 * r, cy + 0.7 * r),
                    (cx + 0.7 * r, cy - 0.7 * r)], color, 2.0)


def _draw_check(draw, s, cx: float, cy: float, color, *,
                r: float = 7.0) -> None:
    """A check mark: the short down-stroke then the long
    up-stroke, the same shape `_verified` draws."""
    _line(draw, s, [(cx - 0.55 * r, cy),
                    (cx - 0.15 * r, cy + 0.45 * r),
                    (cx + 0.6 * r, cy - 0.5 * r)], color, 1.8)


def _draw_heart(draw, s, cx: float, cy: float, color, *,
                r: float = 8.0) -> None:
    """A heart: two lobes on top, one point below."""
    for ex in (-0.38, 0.38):
        _disc(draw, s, cx + ex * r, cy - 0.22 * r, 0.42 * r, color)
    draw.polygon([(s(cx - 0.8 * r), s(cy - 0.05 * r)),
                  (s(cx + 0.8 * r), s(cy - 0.05 * r)),
                  (s(cx), s(cy + 0.85 * r))], fill=color)


def _draw_share_arrow(draw, s, cx: float, cy: float, color, *,
                      r: float = 8.0) -> None:
    """A north-east share arrow: a diagonal shaft with its
    head at the top-right."""
    _line(draw, s, [(cx - 0.6 * r, cy + 0.6 * r),
                    (cx + 0.55 * r, cy - 0.55 * r)], color, 1.8)
    _line(draw, s, [(cx + 0.55 * r, cy - 0.55 * r),
                    (cx + 0.0 * r, cy - 0.55 * r)], color, 1.8)
    _line(draw, s, [(cx + 0.55 * r, cy - 0.55 * r),
                    (cx + 0.55 * r, cy + 0.0 * r)], color, 1.8)


def _draw_arrow_in(draw, s, cx: float, cy: float, color, *,
                   r: float = 8.0) -> None:
    """A south-west arrow (incoming call): the mirror of the
    share arrow, head at the bottom-left."""
    _line(draw, s, [(cx + 0.6 * r, cy - 0.6 * r),
                    (cx - 0.55 * r, cy + 0.55 * r)], color, 1.8)
    _line(draw, s, [(cx - 0.55 * r, cy + 0.55 * r),
                    (cx + 0.0 * r, cy + 0.55 * r)], color, 1.8)
    _line(draw, s, [(cx - 0.55 * r, cy + 0.55 * r),
                    (cx - 0.55 * r, cy + 0.0 * r)], color, 1.8)


def _draw_bell_slash(draw, s, cx: float, cy: float, color, *,
                     r: float = 8.0) -> None:
    """A muted bell: dome + skirt + clapper, struck through
    by one diagonal."""
    draw.arc([s(cx - 0.6 * r), s(cy - 0.7 * r),
              s(cx + 0.6 * r), s(cy + 0.4 * r)],
             start=180, end=360, fill=color,
             width=max(1, s(1.6)))
    _line(draw, s, [(cx - 0.7 * r, cy + 0.45 * r),
                    (cx + 0.7 * r, cy + 0.45 * r)], color, 1.6)
    _disc(draw, s, cx, cy + 0.65 * r, 0.12 * r, color)
    _line(draw, s, [(cx - 0.75 * r, cy - 0.75 * r),
                    (cx + 0.75 * r, cy + 0.75 * r)], color, 1.8)


def _draw_pin(draw, s, cx: float, cy: float, color, *,
              r: float = 8.0) -> None:
    """A map pin: a filled head with a point below."""
    _disc(draw, s, cx, cy - 0.25 * r, 0.5 * r, color)
    draw.polygon([(s(cx - 0.42 * r), s(cy + 0.05 * r)),
                  (s(cx + 0.42 * r), s(cy + 0.05 * r)),
                  (s(cx), s(cy + 0.9 * r))], fill=color)


def _draw_star(draw, s, cx: float, cy: float, color, *,
               r: float = 8.0) -> None:
    """A five-point star, one point straight up."""
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else 0.45 * r
        pts.append((s(cx + rr * math.cos(ang)),
                    s(cy + rr * math.sin(ang))))
    draw.polygon(pts, fill=color)


def _draw_person(draw, s, cx: float, cy: float, color, *,
                 r: float = 8.0) -> None:
    """A bust: head disc over a shoulders dome."""
    _disc(draw, s, cx, cy - 0.5 * r, 0.32 * r, color)
    draw.pieslice([s(cx - 0.75 * r), s(cy - 0.1 * r),
                   s(cx + 0.75 * r), s(cy + 1.3 * r)],
                  start=180, end=360, fill=color)


def _draw_voicemail(draw, s, cx: float, cy: float, color, *,
                    r: float = 8.0) -> None:
    """A voicemail tape: a rounded shell with two reels."""
    draw.rounded_rectangle(
        [s(cx - 0.85 * r), s(cy - 0.6 * r),
         s(cx + 0.85 * r), s(cy + 0.6 * r)],
        radius=s(0.2 * r), outline=color,
        width=max(1, s(1.6)))
    for ex in (-0.35, 0.35):
        _disc(draw, s, cx + ex * r, cy, 0.14 * r, color)


def _draw_play(draw, s, cx: float, cy: float, color, *,
               r: float = 8.0) -> None:
    """A bare play triangle, point to the right."""
    draw.polygon([(s(cx - 0.45 * r), s(cy - 0.6 * r)),
                  (s(cx - 0.45 * r), s(cy + 0.6 * r)),
                  (s(cx + 0.6 * r), s(cy))], fill=color)


def _draw_music_note(draw, s, cx: float, cy: float, color, *,
                     r: float = 8.0) -> None:
    """An eighth note: head + stem + flag."""
    draw.ellipse([s(cx - 0.55 * r), s(cy + 0.25 * r),
                  s(cx + 0.05 * r), s(cy + 0.75 * r)], fill=color)
    _line(draw, s, [(cx + 0.02 * r, cy + 0.5 * r),
                    (cx + 0.02 * r, cy - 0.7 * r)], color, 1.6)
    draw.polygon([(s(cx + 0.02 * r), s(cy - 0.7 * r)),
                  (s(cx + 0.65 * r), s(cy - 0.35 * r)),
                  (s(cx + 0.02 * r), s(cy - 0.15 * r))], fill=color)


def _draw_speech(draw, s, cx: float, cy: float, color, *,
                 r: float = 8.0) -> None:
    """A speech bubble: an outlined rounded box with a tail."""
    draw.rounded_rectangle(
        [s(cx - 0.8 * r), s(cy - 0.65 * r),
         s(cx + 0.8 * r), s(cy + 0.45 * r)],
        radius=s(0.3 * r), outline=color,
        width=max(1, s(1.6)))
    draw.polygon([(s(cx - 0.35 * r), s(cy + 0.4 * r)),
                  (s(cx - 0.1 * r), s(cy + 0.85 * r)),
                  (s(cx + 0.15 * r), s(cy + 0.4 * r))], fill=color)


def _draw_video_camera(draw, s, cx: float, cy: float, color, *,
                       w: float = 26.0) -> None:
    """A video camera: body + lens triangle — the shape the
    chat header used to draw inline (same geometry)."""
    draw.rounded_rectangle(
        [s(cx - 0.5 * w), s(cy - 0.27 * w),
         s(cx + 0.15 * w), s(cy + 0.27 * w)],
        radius=s(3), fill=color)
    draw.polygon([(s(cx + 0.23 * w), s(cy - 0.12 * w)),
                  (s(cx + 0.5 * w), s(cy - 0.31 * w)),
                  (s(cx + 0.5 * w), s(cy + 0.31 * w)),
                  (s(cx + 0.23 * w), s(cy + 0.12 * w))], fill=color)


def _draw_headphones(draw, s, cx: float, cy: float, color, *,
                     r: float = 8.0) -> None:
    """Headphones: a band over two ear cups."""
    draw.arc([s(cx - 0.75 * r), s(cy - 0.7 * r),
              s(cx + 0.75 * r), s(cy + 0.7 * r)],
             start=180, end=360, fill=color,
             width=max(1, s(1.8)))
    for ex in (-0.75, 0.75):
        draw.rounded_rectangle(
            [s(cx + ex * r - 0.14 * r), s(cy + 0.05 * r),
             s(cx + ex * r + 0.14 * r), s(cy + 0.6 * r)],
            radius=s(0.1 * r), fill=color)


def _draw_document(draw, s, cx: float, cy: float, color, *,
                   r: float = 8.0) -> None:
    """A document: an outlined page with a folded corner
    and two text lines."""
    draw.rounded_rectangle(
        [s(cx - 0.55 * r), s(cy - 0.75 * r),
         s(cx + 0.55 * r), s(cy + 0.75 * r)],
        radius=s(0.1 * r), outline=color,
        width=max(1, s(1.6)))
    _line(draw, s, [(cx + 0.15 * r, cy - 0.75 * r),
                    (cx + 0.55 * r, cy - 0.35 * r)], color, 1.6)
    _line(draw, s, [(cx - 0.3 * r, cy - 0.1 * r),
                    (cx + 0.3 * r, cy - 0.1 * r)], color, 1.4)
    _line(draw, s, [(cx - 0.3 * r, cy + 0.3 * r),
                    (cx + 0.3 * r, cy + 0.3 * r)], color, 1.4)


def _draw_hourglass(draw, s, cx: float, cy: float, color, *,
                    r: float = 8.0) -> None:
    """An hourglass: two triangles apex to apex, capped
    top and bottom."""
    draw.polygon([(s(cx - 0.6 * r), s(cy - 0.75 * r)),
                  (s(cx + 0.6 * r), s(cy - 0.75 * r)),
                  (s(cx), s(cy))], fill=color)
    draw.polygon([(s(cx - 0.6 * r), s(cy + 0.75 * r)),
                  (s(cx + 0.6 * r), s(cy + 0.75 * r)),
                  (s(cx), s(cy))], fill=color)
    _line(draw, s, [(cx - 0.6 * r, cy - 0.75 * r),
                    (cx + 0.6 * r, cy - 0.75 * r)], color, 1.6)
    _line(draw, s, [(cx - 0.6 * r, cy + 0.75 * r),
                    (cx + 0.6 * r, cy + 0.75 * r)], color, 1.6)


def _draw_no_entry(draw, s, cx: float, cy: float, color, *,
                   r: float = 8.0) -> None:
    """A no-entry sign: a ring struck through."""
    _ring(draw, s, cx, cy, r, color, 1.8)
    _line(draw, s, [(cx - 0.7 * r, cy - 0.7 * r),
                    (cx + 0.7 * r, cy + 0.7 * r)], color, 1.8)


def _draw_paperclip(draw, s, cx: float, cy: float, color, *,
                    r: float = 9.0) -> None:
    """A paperclip: an outer wire loop with the inner end
    curling up inside it — the Android WhatsApp attach
    button, drawn as vectors because no stocked font carries
    the glyph."""
    # outer loop: top arch, two sides, bottom arch
    draw.arc([s(cx - 0.55 * r), s(cy - r), s(cx + 0.55 * r),
              s(cy)], start=180, end=360, fill=color,
             width=max(1, s(1.6)))
    _line(draw, s, [(cx - 0.55 * r, cy - 0.5 * r),
                    (cx - 0.55 * r, cy + 0.5 * r)], color, 1.6)
    _line(draw, s, [(cx + 0.55 * r, cy - 0.5 * r),
                    (cx + 0.55 * r, cy + 0.5 * r)], color, 1.6)
    draw.arc([s(cx - 0.55 * r), s(cy), s(cx + 0.55 * r),
              s(cy + r)], start=0, end=180, fill=color,
             width=max(1, s(1.6)))
    # inner wire: up from the bottom, curling over at the top
    _line(draw, s, [(cx - 0.2 * r, cy + 0.55 * r),
                    (cx - 0.2 * r, cy - 0.35 * r)], color, 1.6)
    draw.arc([s(cx - 0.2 * r), s(cy - 0.7 * r),
              s(cx + 0.2 * r), s(cy - 0.1 * r)],
             start=180, end=360, fill=color, width=max(1, s(1.6)))


def _draw_lock(draw, s, cx: float, cy: float, color, *,
               r: float = 7.0) -> None:
    """A padlock silhouette: shackle arc over a filled body
    — the encryption notice's lock in a WhatsApp chat."""
    draw.arc([s(cx - 0.55 * r), s(cy - r), s(cx + 0.55 * r),
              s(cy + 0.1 * r)], start=180, end=360, fill=color,
             width=max(1, s(1.8)))
    draw.rounded_rectangle(
        [s(cx - 0.75 * r), s(cy - 0.1 * r), s(cx + 0.75 * r),
         s(cy + 0.9 * r)], radius=max(1, s(2)), fill=color)
