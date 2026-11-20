"""Camera shots — a VTuber show cuts cameras on beats.

A shot is a crop of the rendered frame: the rig draws the whole
puppet every tick, and the shot says which part of it the audience
sees. ``wide`` is the full frame; ``medium`` the upper body;
``close`` head and shoulders; ``face`` the face. Crops are row
slices over RGB24 bytes — no image library needed, and the frame
stays honest RGB24 at the cropped size.

The puppet is drawn centred with the head at the top: the head
sits in roughly the top third of the frame, the body fills the
rest. The fractions below are the stage geometry, stated plainly
in the table — a different rig layout wants different fractions,
not a silent re-fit.
"""

from __future__ import annotations

from .errors import VtuberError

__all__ = [
    "SHOTS",
    "apply_shot",
    "shot_names",
]

#: Shot name -> (x0, y0, x1, y1) as fractions of the frame.
SHOTS: dict[str, tuple[float, float, float, float]] = {
    "wide": (0.0, 0.0, 1.0, 1.0),
    "medium": (0.15, 0.05, 0.85, 1.0),
    "close": (0.25, 0.0, 0.75, 0.55),
    "face": (0.35, 0.0, 0.65, 0.35),
}


def shot_names() -> list[str]:
    """The shot names, in cut order (widest first)."""
    return ["wide", "medium", "close", "face"]


def apply_shot(rgb: bytes, width: int, height: int,
               shot: str) -> tuple[bytes, int, int]:
    """Crop an RGB24 frame to a shot. Returns (bytes, w, h)."""
    try:
        x0f, y0f, x1f, y1f = SHOTS[shot]
    except KeyError as exc:
        raise VtuberError(
            f"unknown shot {shot!r} — shots are "
            f"{', '.join(shot_names())}",
            kind="unknown_shot") from exc
    if len(rgb) != width * height * 3:
        raise VtuberError(
            f"frame is {len(rgb)} bytes for a {width}x{height} "
            f"RGB24 frame ({width * height * 3} expected)",
            kind="bad_frame")
    x0 = max(0, int(x0f * width))
    x1 = min(width, max(x0 + 1, int(x1f * width)))
    y0 = max(0, int(y0f * height))
    y1 = min(height, max(y0 + 1, int(y1f * height)))
    new_w = x1 - x0
    new_h = y1 - y0
    out = bytearray()
    row_bytes = width * 3
    for y in range(y0, y1):
        start = y * row_bytes + x0 * 3
        out += rgb[start:start + new_w * 3]
    return bytes(out), new_w, new_h
