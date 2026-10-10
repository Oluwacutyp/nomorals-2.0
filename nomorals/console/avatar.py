"""ASCII ninja avatar for the Devon console.

The owner wanted the watch-mode header to resemble their ninja avatar —
a hooded figure with electric-blue accents and one amber eye — instead of
a lone emoji. Terminals can't show the real image, so this is a hand-tuned
multi-line ASCII rendering in the console palette (blues/cyans/amber;
never black backgrounds, never red).

Two sizes:
- :func:`render_ninja`      — full 13-row figure (status view side panel)
- :func:`render_ninja_mini` — compact 7-row figure (watch-mode header)

Both degrade to plain uncolored text when color is off.
"""

from __future__ import annotations

from .palette import BOLD, BRIGHT_CYAN, BRIGHT_YELLOW, CYAN, DIM, paint

_HOOD = BRIGHT_CYAN   # electric-blue hood edge
_FACE = CYAN          # shadowed face opening
_EYE = BRIGHT_YELLOW + BOLD  # the one amber eye
_CLOAK = DIM          # dark cloak folds

# Each line: list of (text, color-code) segments.
_FULL: list[list[tuple[str, str]]] = [
    [(r"         /\         ", _HOOD)],
    [(r"        /  \        ", _HOOD)],
    [(r"       /    \       ", _HOOD)],
    [(r"      /  __  \      ", _HOOD)],
    [(r"     |  /  \  |     ", _FACE)],
    [(r"     | |  ", _FACE), ("◉", _EYE), (r"  | |     ", _FACE)],
    [(r"     |  \__/  |     ", _FACE)],
    [(r"      \      /      ", _FACE)],
    [(r"       |    |       ", _CLOAK)],
    [(r"      /|    |\      ", _CLOAK)],
    [(r"     / |    | \     ", _CLOAK)],
    [(r"    /  |    |  \    ", _CLOAK)],
    [(r"   |   |    |   |   ", _CLOAK)],
]

#: The owner's chosen 7-row mini ninja (exact art, do not redesign).
#: Hood in electric blue, the single eye in amber.
_MINI: list[list[tuple[str, str]]] = [
    [(r"     /\    ", _HOOD)],
    [(r"    /  \   ", _HOOD)],
    [(r"   / || \  ", _HOOD)],
    [(r"  |  ", _FACE), ("◉", _EYE), (r"  |  ", _FACE)],
    [(r"   \ || /  ", _FACE)],
    [(r"    \  /   ", _FACE)],
    [(r"   / \/ \  ", _CLOAK)],
]

NINJA_WIDTH = max(
    sum(len(text) for text, _ in line) for line in _FULL
)
NINJA_HEIGHT = len(_FULL)
NINJA_MINI_WIDTH = max(
    sum(len(text) for text, _ in line) for line in _MINI
)
NINJA_MINI_HEIGHT = len(_MINI)


def _render(lines: list[list[tuple[str, str]]], *, color: bool | None) -> str:
    return "\n".join(
        "".join(paint(text, code, color=color) for text, code in segs)
        for segs in lines
    )


def ninja_mini_lines(*, color: bool | None = None) -> list[str]:
    """The 7-row mini ninja as individual painted lines.

    For side-by-side layouts (e.g. avatar next to the dashboard title).
    Each line is fully painted; use ``visible_width`` from palette to
    measure (the ◉ eye is a wide char).
    """
    return [
        "".join(paint(text, code, color=color) for text, code in segs)
        for segs in _MINI
    ]


def render_ninja(*, color: bool | None = None) -> str:
    """Full 13-row hooded ninja, colored. Plain text when color is off."""
    return _render(_FULL, color=color)


def render_ninja_mini(*, color: bool | None = None) -> str:
    """Compact 7-row hooded ninja for the watch-mode header."""
    return _render(_MINI, color=color)


# ── extra avatar styles (ascii-art density idioms) ─────────────────────────
# The owner's chosen mini/full art above is never redesigned. These are
# additional densities for different surfaces: braille micro-avatar,
# pixel-shaded, and a wide cinematic figure.

_BRAILLE: list[list[tuple[str, str]]] = [
    [(r"  ⣿⣿⣿⣿  ", _HOOD)],
    [(r" ⣿⣿⣿⣿⣿⣿ ", _HOOD)],
    [(r"⣿⣿⣿⣿⣿⣿⣿⣿", _HOOD)],
    [(r"⣿⣿", _FACE), (r"⣿⣿", _EYE), (r"⣿⣿", _FACE)],
    [(r" ⣿⣿⣿⣿⣿⣿ ", _FACE)],
    [(r"  ⣿⣿⣿⣿  ", _CLOAK)],
    [(r" ⣿⣿⣿⣿⣿⣿ ", _CLOAK)],
]

_PIXEL: list[list[tuple[str, str]]] = [
    [(r"     ▓▓     ", _HOOD)],
    [(r"    ▓▓▓▓    ", _HOOD)],
    [(r"   ▓▓▒▒▓▓   ", _HOOD)],
    [(r"  ▓▓ ", _FACE), ("◉", _EYE), (r" ▓▓  ", _FACE)],
    [(r"   ▓▓▒▒▓▓   ", _FACE)],
    [(r"    ▓▓▓▓    ", _FACE)],
    [(r"   ▒▓▓▓▓▒   ", _CLOAK)],
]

_WIDE: list[list[tuple[str, str]]] = [
    [(r"              /    \              ", _HOOD)],
    [(r"             /      \             ", _HOOD)],
    [(r"            /   __   \            ", _HOOD)],
    [(r"           /   /  \   \           ", _HOOD)],
    [(r"          /   |    |   \          ", _HOOD)],
    [(r"         |    |    |    |         ", _FACE)],
    [(r"         |    | ", _FACE), ("◉", _EYE), (r"  |    |         ", _FACE)],
    [(r"         |    |    |    |         ", _FACE)],
    [(r"          \   |    |   /          ", _FACE)],
    [(r"           \  |    |  /           ", _FACE)],
    [(r"            \ |    | /            ", _CLOAK)],
    [(r"             ||    ||             ", _CLOAK)],
    [(r"            /||    ||\            ", _CLOAK)],
    [(r"           / ||    || \           ", _CLOAK)],
    [(r"          |  ||    ||  |          ", _CLOAK)],
]

_AVATARS: dict[str, list[list[tuple[str, str]]]] = {
    "mini": _MINI,
    "full": _FULL,
    "braille": _BRAILLE,
    "pixel": _PIXEL,
    "wide": _WIDE,
}

#: All avatar styles, in display order.
AVATAR_STYLES = ("mini", "full", "braille", "pixel", "wide")


def list_avatar_styles() -> list[str]:
    """Names of all avatar styles."""
    return list(AVATAR_STYLES)


def avatar_size(style: str = "mini") -> tuple[int, int]:
    """(width, height) of an avatar style in display cells."""
    lines = _AVATARS.get((style or "mini").lower(), _MINI)
    w = max(sum(len(text) for text, _ in line) for line in lines)
    return w, len(lines)


def render_avatar(
    style: str = "mini",
    *,
    color: bool | None = None,
    frame: bool = False,
) -> str:
    """Render the ninja avatar in ``style`` (see :func:`list_avatar_styles`).

    ``frame=True`` draws a rounded box around it. The ``mini`` and
    ``full`` styles are the owner's chosen art, rendered exactly.
    """
    lines = _AVATARS.get((style or "mini").lower(), _MINI)
    out = _render(lines, color=color)
    if frame:
        from .palette import BOX_ROUNDED, DIM, paint, visible_width

        b = BOX_ROUNDED
        raw = out.split("\n")
        w = max((visible_width(ln) for ln in raw), default=0)
        top = b["tl"] + b["h"] * (w + 2) + b["tr"]
        bot = b["bl"] + b["h"] * (w + 2) + b["br"]
        framed = [paint(top, DIM, color=color)]
        for ln in raw:
            pad = " " * max(0, w - visible_width(ln))
            framed.append(
                paint(b["v"] + " ", DIM, color=color)
                + ln + pad
                + paint(" " + b["v"], DIM, color=color)
            )
        framed.append(paint(bot, DIM, color=color))
        out = "\n".join(framed)
    return out


__all__ = [
    "NINJA_HEIGHT",
    "NINJA_MINI_HEIGHT",
    "NINJA_MINI_WIDTH",
    "NINJA_WIDTH",
    "AVATAR_STYLES",
    "list_avatar_styles",
    "avatar_size",
    "ninja_mini_lines",
    "render_avatar",
    "render_ninja",
    "render_ninja_mini",
]
