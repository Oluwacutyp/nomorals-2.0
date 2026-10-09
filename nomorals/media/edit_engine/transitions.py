"""Style-agnostic transitions → ffmpeg boundary recipes.

Kinds:

* ``cut`` — hard cut (concat, no overlap).
* ``fade`` / ``dissolve`` — true overlap blend via ``xfade``.
* ``flash`` — white blink at the cut (edge fades, no overlap).
* ``dip`` — black blink at the cut (edge fades, no overlap).
* ``whip_pan`` — directional slide via ``xfade`` (slideleft/slideright/
  slideup/slidedown).
* ``glitch_cut`` — hard cut with an rgb-split burst on the outgoing
  tail (composes the ``rgb_split`` effect — no new machinery).

Transitions are parameterized data; :mod:`render` applies them.
"""

from __future__ import annotations

from typing import Any

from ...media_edit.videos import MediaEditError

#: kind → aliases
_TRANSITION_ALIASES = {
    "hard": "cut",
    "hardcut": "cut",
    "hard_cut": "cut",
    "crossfade": "dissolve",
    "xfade": "dissolve",
    "white": "flash",
    "black": "dip",
    "whip": "whip_pan",
    "whippan": "whip_pan",
    "glitch": "glitch_cut",
    "glitchcut": "glitch_cut",
}

#: kinds implemented as true overlaps (xfade). The rest assemble on the
#: concat path (edge fades / effect bursts, no overlap).
_OVERLAP_KINDS = ("fade", "dissolve", "whip_pan")

#: kind → xfade transition name (overlap kinds only)
_XFADE_NAME = {
    "fade": "fade",
    "dissolve": "fade",
    "whip_pan": "slideleft",
}

#: whip_pan direction param → xfade slide transition
_WHIP_DIRS = {
    "left": "slideleft",
    "right": "slideright",
    "up": "slideup",
    "down": "slidedown",
}


def validate_transition(kind: str) -> str:
    """Canonical transition kind; raises on unknown."""
    k = str(kind or "cut").strip().lower()
    k = _TRANSITION_ALIASES.get(k, k)
    if k not in _OVERLAP_KINDS + ("cut", "flash", "dip", "glitch_cut"):
        raise MediaEditError(
            f"unknown transition {kind!r}; use: {list_transitions()}")
    return k


def list_transitions() -> list[str]:
    """All canonical transition kinds."""
    return ["cut", "fade", "dissolve", "flash", "dip", "whip_pan",
            "glitch_cut"]


def is_overlap(kind: str) -> bool:
    """True when the transition overlaps the two clips (xfade)."""
    return validate_transition(kind) in _OVERLAP_KINDS


def xfade_args(kind: str, params: dict[str, Any],
               duration: float) -> tuple[str, float]:
    """(xfade transition name, overlap seconds) for an overlap kind."""
    k = validate_transition(kind)
    if k not in _OVERLAP_KINDS:
        raise MediaEditError(f"transition {k!r} is not an overlap kind")
    td = max(0.05, float(duration))
    if k == "whip_pan":
        direction = str(params.get("direction", "left")).strip().lower()
        name = _WHIP_DIRS.get(direction)
        if name is None:
            raise MediaEditError(
                f"whip_pan direction must be one of "
                f"{sorted(_WHIP_DIRS)}, got {direction!r}")
        return name, td
    return _XFADE_NAME[k], td


def edge_fade_filters(play_d: float, kind: str,
                      params: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """(head_fades, tail_fades) for the blink kinds.

    Returns filter lists to apply to the *incoming* head and *outgoing*
    tail of a boundary: flash/dip blink without any overlap.
    """
    k = validate_transition(kind)
    if k not in ("flash", "dip"):
        return [], []
    color = "white" if k == "flash" else "black"
    d = float((params or {}).get("blink", 0.06))
    d = min(max(d, 0.02), play_d / 2.0 or 0.02)
    head = [f"fade=t=in:st=0:d={d:.3f}:c={color}"]
    tail = [f"fade=t=out:st={max(play_d - d, 0):.3f}:d={d:.3f}:c={color}"]
    return head, tail


def glitch_burst_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """rgb_split params for the glitch_cut tail burst."""
    p = dict(params or {})
    return {"shift": float(p.get("shift", 14.0)), "animate": True,
            "period": float(p.get("period", 0.35)),
            "window": float(p.get("window", 0.18))}
