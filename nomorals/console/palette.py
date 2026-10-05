"""ANSI palette for the Devon console.

Hard rule from the owner: NO black backgrounds, NO red text. The palette is
blues, greens, cyans, purples/magentas, whites and yellows — 256-color safe
for Termux. Everything degrades gracefully when color is unsupported.
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
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)
