"""ANSI palette for the Devon console.

Hard rule from the owner: NO black backgrounds, NO red text. The palette is
blues, greens, cyans, purples/magentas, whites and yellows — 256-color safe
for Termux — plus the Gemini "Cyber-Stealth" truecolor accents (24-bit SGR,
still stdlib-only, no rich). Everything degrades gracefully when color is
unsupported. The terminal's own background is never overridden.
"""

from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Foreground colors — no red (31), no black background (40) anywhere.
CYAN = "\033[36m"
BRIGHT_CYAN = "\033[96m"
BLUE = "\033[34m"
BRIGHT_BLUE = "\033[94m"
GREEN = "\033[32m"
BRIGHT_GREEN = "\033[92m"
MAGENTA = "\033[35m"          # purple — used where red would normally go
BRIGHT_MAGENTA = "\033[95m"
YELLOW = "\033[33m"
BRIGHT_YELLOW = "\033[93m"
WHITE = "\033[37m"
BRIGHT_WHITE = "\033[97m"
GRAY = "\033[90m"

# Semantic aliases.
INFO = CYAN
OK = GREEN
WARN = YELLOW
ERR = MAGENTA            # errors are magenta, never red
CRIT = BRIGHT_MAGENTA
TITLE = BRIGHT_CYAN
SUBTLE = GRAY
ACCENT = BRIGHT_BLUE

# Text attributes (no background codes — the terminal's own background is
# never overridden, per the owner's rule).
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
STRIKE = "\033[9m"
REVERSE = "\033[7m"

# Box-drawing glyph sets for panels/tables (btop/Rich idiom: box drawing
# for all chrome, never ASCII +-).
BOX_SINGLE = {
    "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
    "h": "─", "v": "│", "lt": "├", "rt": "┤",
    "tt": "┬", "bt": "┴", "cross": "┼",
}
BOX_DOUBLE = {
    "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝",
    "h": "═", "v": "║", "lt": "╠", "rt": "╣",
    "tt": "╦", "bt": "╩", "cross": "╬",
}
BOX_ROUNDED = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "h": "─", "v": "│", "lt": "├", "rt": "┤",
    "tt": "┬", "bt": "┴", "cross": "┼",
}
BOX_HEAVY = {
    "tl": "┏", "tr": "┓", "bl": "┗", "br": "┛",
    "h": "━", "v": "┃", "lt": "┠", "rt": "┨",
    "tt": "┯", "bt": "┷", "cross": "╋",
}

# 1/8-cell sub-precision horizontal fill (btop meter idiom).
_BLOCK_FRACS = " ▏▎▍▌▋▊▉█"


def rgb(r: int, g: int, b: int) -> str:
    """Truecolor (24-bit) foreground SGR sequence. stdlib-only, no rich."""
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"\033[38;2;{r};{g};{b}m"


def hex_color(spec: str) -> str:
    """``#rrggbb`` (or ``rrggbb``) → truecolor SGR sequence."""
    s = spec.strip().lstrip("#")
    if len(s) == 3:  # short form #abc → #aabbcc
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"bad hex color: {spec!r}")
    return rgb(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# Semantic color roles (inspect-rs idiom): colors communicate structural
# roles rather than arbitrary text. Roles map onto the base palette so
# every theme stays consistent; see themes.py for the full token contract.
SEMANTIC: dict[str, str] = {
    "type": BRIGHT_CYAN,
    "field": CYAN,
    "variant": BRIGHT_MAGENTA,
    "key": GREEN,
    "string": BRIGHT_GREEN,
    "number": BRIGHT_YELLOW,
    "bool": MAGENTA,
    "punct": GRAY,
    "sensitive": BRIGHT_MAGENTA + BOLD,
    "truncated": YELLOW + ITALIC,
}


def paint_sem(text: str, role: str, *, color: bool | None = None) -> str:
    """Paint ``text`` with a semantic role (see :data:`SEMANTIC`)."""
    return paint(text, SEMANTIC.get(role, ""), color=color)

# Gemini "Cyber-Stealth" accents — truecolor 24-bit SGR sequences, no
# third-party library. Circuit cyan for primary borders / active
# indicators, amber for warnings (sparingly), icy grey-blue for body
# text instead of pure white. No background codes anywhere.
STEALTH_CYAN = "\033[38;2;0;240;255m"      # #00F0FF
STEALTH_AMBER = "\033[38;2;255;176;0m"     # #FFB000
STEALTH_TEXT = "\033[38;2;163;184;204m"    # #A3B8CC


def supports_color() -> bool:
    """True when ANSI colors are safe to emit."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    stream = sys.stderr
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, *codes: str, color: bool | None = None) -> str:
    """Wrap ``text`` in ANSI codes. No-op when color is disabled."""
    use_color = supports_color() if color is None else color
    if not use_color or not codes:
        return text
    return f"{''.join(codes)}{text}{RESET}"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences — for width math on colored lines."""
    return _csi_re().sub("", text)


_CSI_RE = None


def _csi_re():
    global _CSI_RE
    if _CSI_RE is None:
        import re

        # Any CSI sequence: ESC [ params intermediates final-byte.
        _CSI_RE = re.compile(r"\033\[[0-9;:?]*[ -/]*[@-~]")
    return _CSI_RE


def visible_width(text: str) -> int:
    """Display-cell width of ``text``.

    ANSI escapes cost 0 cells; East-Asian wide / fullwidth chars (most
    emoji) cost 2. Used so full-screen layouts never wrap a line by
    miscounting colored or emoji-heavy text.
    """
    import unicodedata

    clean = _csi_re().sub("", text)
    width = 0
    for ch in clean:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def truncate_visible(text: str, width: int, ellipsis: str = "…") -> str:
    """Truncate ``text`` to ``width`` display cells.

    ANSI escape sequences are passed through untouched (so colors don't
    leak) and wide chars count as 2 cells. A truncated line gets the
    ellipsis plus a reset so no color bleeds into the next line.
    """
    import unicodedata

    width = max(0, int(width))
    if visible_width(text) <= width:
        return text
    csi = _csi_re()
    out: list[str] = []
    cells = 0
    limit = max(0, width - visible_width(ellipsis))
    truncated = False
    pos = 0

    def feed(chunk: str) -> None:
        nonlocal cells, truncated
        for ch in chunk:
            w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if cells + w > limit:
                truncated = True
                return
            out.append(ch)
            cells += w

    for m in csi.finditer(text):
        feed(text[pos : m.start()])
        if truncated:
            break
        out.append(m.group(0))  # keep the escape sequence itself
        pos = m.end()
    if not truncated:
        feed(text[pos:])
    out.append(ellipsis + RESET)
    return "".join(out)


def pad_visible(text: str, width: int, align: str = "left") -> str:
    """Pad ``text`` to ``width`` display cells (ANSI-aware).

    ``align`` is ``"left"``, ``"right"`` or ``"center"``. Text wider than
    ``width`` is truncated with :func:`truncate_visible`.
    """
    width = max(0, int(width))
    vw = visible_width(text)
    if vw > width:
        return truncate_visible(text, width)
    pad = width - vw
    if align == "right":
        return " " * pad + text
    if align == "center":
        left = pad // 2
        return " " * left + text + " " * (pad - left)
    return text + " " * pad


def wrap_visible(text: str, width: int) -> list[str]:
    """Word-wrap ``text`` to ``width`` display cells (ANSI-aware).

    ANSI sequences ride along with the word they precede; wide chars
    count as 2 cells. Returns the wrapped lines (no trailing newline).
    """
    import unicodedata

    width = max(1, int(width))
    csi = _csi_re()

    def word_width(w: str) -> int:
        clean = csi.sub("", w)
        return sum(
            2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            for ch in clean
        )

    lines: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for word in text.split(" "):
        w = word_width(word)
        if w > width and not cur:
            # Single overlong word: hard-truncate, keep color contained.
            lines.append(truncate_visible(word, width))
            continue
        if cur and cur_w + 1 + w > width:
            lines.append("".join(cur))
            cur, cur_w = [], 0
        if cur:
            cur.append(" ")
            cur_w += 1
        cur.append(word)
        cur_w += w
    if cur:
        lines.append("".join(cur))
    return lines or [""]
