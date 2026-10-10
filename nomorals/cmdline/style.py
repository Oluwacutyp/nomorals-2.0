"""Zero-dependency terminal styling for the ``nm`` CLI.

Best-in-class CLIs (rich, textual, typer) centralize all presentation in one
theme: semantic styles (``success``/``error``/``warning``/``info``) instead of
scattered ANSI codes, icons in one place, TTY detection, and ``NO_COLOR``
support. ``rich`` is not a dependency here (zero-mandatory-deps order, and it
is not installed in this environment), so this module implements the same
pattern with the standard library only:

* :class:`Theme` — nested ``Colors`` / ``Icons`` / ``TextStyles``, the single
  source of truth for how ``nm`` looks.
* Automatic TTY detection: no ANSI when piped, when ``NO_COLOR`` /
  ``NM_NO_COLOR`` is set, or when ``TERM=dumb``. Global ``--no-color`` forces
  it off.
* :class:`Table` — auto-sized column renderer; styled header on a TTY, plain
  aligned text otherwise (byte-identical for captured/test output).
* :func:`kv` — key: value panel for config/status dumps.
* :func:`spinner` — stderr progress indicator for long operations.

Existing ``_emit`` prose paths are deliberately untouched — styling is opt-in
through these helpers, so ``--json`` contracts and captured test output stay
byte-identical. A rich backend can replace :class:`Theme` internals later
without changing any call site.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------

class Theme:
    """Centralized ``nm`` visual identity.

    Colors are semantic (what the text *means*), never decorative — one
    accent plus the semantic set, per terminal-UI best practice. Every icon
    lives in :class:`Icons` so output never mixes emoji styles.
    """

    class Colors:
        ACCENT = "cyan"
        SUCCESS = "green"
        ERROR = "red"
        WARNING = "yellow"
        INFO = "blue"
        DIM = "bright_black"
        TITLE = "bold cyan"
        HEADER = "bold"

    class Icons:
        OK = "✓"
        FAIL = "✗"
        WARN = "⚠"
        INFO = "ℹ"
        ARROW = "→"
        DOT = "•"
        # ASCII fallbacks for terminals without unicode support
        OK_ASCII = "[ok]"
        FAIL_ASCII = "[!!]"
        WARN_ASCII = "[!]"
        INFO_ASCII = "[i]"

    class TextStyles:
        TITLE = "bold cyan"
        HEADER = "bold"
        EMPHASIS = "bold"
        MUTED = "dim"
        CODE = "cyan"


_ANSI: dict[str, str] = {
    "black": "30", "red": "31", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
    "bright_black": "90", "bright_red": "91", "bright_green": "92",
    "bright_yellow": "93", "bright_blue": "94", "bright_magenta": "95",
    "bright_cyan": "96", "bright_white": "97",
    "bold": "1", "dim": "2", "italic": "3", "underline": "4",
}

_NO_COLOR_ENV = ("NO_COLOR", "NM_NO_COLOR")

# Module-level override, set once from ``main()`` via the ``--no-color`` flag.
# Tests flip this directly.
_force_no_color: bool = False


def set_no_color(value: bool = True) -> None:
    """Force colorized output off (or back on) for this process."""
    global _force_no_color
    _force_no_color = value


def color_enabled(stream: object = None) -> bool:
    """Whether styled output is allowed for *stream* (default: stdout).

    Off when: forced off, ``NO_COLOR``/``NM_NO_COLOR`` set, ``TERM=dumb``,
    or the stream is not a TTY.
    """
    if _force_no_color:
        return False
    if any(os.environ.get(var) for var in _NO_COLOR_ENV):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    if stream is None:
        stream = sys.stdout
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or ""
    return enc.lower().replace("-", "") in ("utf8", "utf8sig")


def stylize(text: str, *styles: str, stream: object = None) -> str:
    """Wrap *text* in ANSI codes for *styles* (``"bold cyan"`` etc.).

    Returns *text* unchanged when color is disabled — safe to call
    unconditionally.
    """
    if not styles or not color_enabled(stream):
        return text
    codes: list[str] = []
    for chunk in styles:
        for part in chunk.split():
            code = _ANSI.get(part)
            if code and code not in codes:
                codes.append(code)
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def icon(name: str) -> str:
    """Themed icon by name (``ok``/``fail``/``warn``/``info``/``arrow``/``dot``).

    Falls back to an ASCII token when the stdout encoding cannot do unicode
    or color is disabled — never emits a bare emoji into a pipe.
    """
    plain = not color_enabled()
    table = {
        "ok": (Theme.Icons.OK, Theme.Icons.OK_ASCII),
        "fail": (Theme.Icons.FAIL, Theme.Icons.FAIL_ASCII),
        "warn": (Theme.Icons.WARN, Theme.Icons.WARN_ASCII),
        "info": (Theme.Icons.INFO, Theme.Icons.INFO_ASCII),
        "arrow": (Theme.Icons.ARROW, "->"),
        "dot": (Theme.Icons.DOT, "-"),
    }
    uni, ascii_ = table.get(name, (name, name))
    if plain or not _supports_unicode():
        return ascii_
    return uni


# ---------------------------------------------------------------------------
# semantic one-liners
# ---------------------------------------------------------------------------

def ok(text: str) -> str:
    """Success line: ``✓ <text>`` (``[ok] <text>`` when plain)."""
    return f"{icon('ok')} {stylize(text, Theme.Colors.SUCCESS)}"


def err(text: str) -> str:
    """Error line: ``✗ <text>`` (``[!!] <text>`` when plain)."""
    return f"{icon('fail')} {stylize(text, Theme.Colors.ERROR)}"


def warn(text: str) -> str:
    """Warning line: ``⚠ <text>`` (``[!] <text>`` when plain)."""
    return f"{icon('warn')} {stylize(text, Theme.Colors.WARNING)}"


def info(text: str) -> str:
    """Info line: ``ℹ <text>`` (``[i] <text>`` when plain)."""
    return f"{icon('info')} {stylize(text, Theme.Colors.INFO)}"


def title(text: str) -> str:
    """Section title, themed."""
    return stylize(text, Theme.TextStyles.TITLE)


def dim(text: str) -> str:
    """De-emphasized text."""
    return stylize(text, Theme.Colors.DIM)


def accent(text: str) -> str:
    """Accent-colored text (command names, ids)."""
    return stylize(text, Theme.Colors.ACCENT)


# ---------------------------------------------------------------------------
# table renderer
# ---------------------------------------------------------------------------

def _visible_len(text: str) -> int:
    return len(text)


class Table:
    """Auto-sized column table.

    ``Table(["name", "alias", "help"], rows).render()`` → aligned text.
    On a TTY the header is styled; otherwise output is plain aligned text,
    so captured/test output stays byte-identical.
    """

    def __init__(self, headers: Sequence[str],
                 rows: Iterable[Sequence[object]] = (),
                 max_width: int | None = None) -> None:
        self.headers = [str(h) for h in headers]
        self.rows = [[str(c) for c in r] for r in rows]
        self.max_width = max_width or shutil.get_terminal_size((100, 24)).columns

    def add_row(self, *cells: object) -> None:
        self.rows.append([str(c) for c in cells])

    def _widths(self) -> list[int]:
        widths = [len(h) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], _visible_len(cell))
        # Shrink the last column to fit the terminal rather than wrapping.
        total = sum(widths) + 2 * (len(widths) - 1)
        if total > self.max_width and widths:
            overflow = total - self.max_width
            widths[-1] = max(8, widths[-1] - overflow)
        return widths

    @staticmethod
    def _fit(text: str, width: int) -> str:
        if _visible_len(text) <= width:
            return text.ljust(width)
        return text[: max(0, width - 1)] + "…"

    def render(self) -> str:
        widths = self._widths()
        lines: list[str] = []
        header = "  ".join(self._fit(h, w)
                           for h, w in zip(self.headers, widths)).rstrip()
        lines.append(stylize(header, Theme.TextStyles.HEADER))
        for row in self.rows:
            padded = list(row) + [""] * (len(widths) - len(row))
            line = "  ".join(self._fit(c, w)
                             for c, w in zip(padded, widths)).rstrip()
            lines.append(line)
        return "\n".join(lines)


def kv(mapping: dict[str, object], key_width: int = 0) -> str:
    """Key: value panel — ``key`` dimmed, value plain."""
    items = [(str(k), str(v)) for k, v in mapping.items()]
    width = key_width or (max((len(k) for k, _ in items), default=0))
    lines = []
    for k, v in items:
        lines.append(f"{stylize(k.ljust(width), Theme.Colors.DIM)}  {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# spinner (stderr, so stdout stays machine-clean)
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


@contextmanager
def spinner(label: str) -> Iterator[None]:
    """``with spinner("fetching"):`` — animated progress on stderr.

    Silent when stderr is not a TTY (CI/pipes/agents): the label is simply
    skipped so machine output stays clean.
    """
    if not color_enabled(sys.stderr):
        yield
        return
    stop = threading.Event()

    def _spin() -> None:
        i = 0
        frames = _SPINNER_FRAMES if _supports_unicode() else ["-", "\\", "|", "/"]
        while not stop.wait(0.08):
            frame = frames[i % len(frames)]
            sys.stderr.write(f"\r{stylize(frame, Theme.Colors.ACCENT)} {label}…")
            sys.stderr.flush()
            i += 1
        sys.stderr.write("\r" + " " * (len(label) + 4) + "\r")
        sys.stderr.flush()

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        sys.stderr.write(f"{icon('ok')} {label}\n")
        sys.stderr.flush()
