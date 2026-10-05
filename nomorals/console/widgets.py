"""Console widgets: progress bars, sparklines, live screens, message cards.

All widgets are pure-stdlib ANSI. They emit nothing when color/tty is
unavailable (plain-text fallbacks), and never use black backgrounds or red.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Any, Callable, Iterable

from .palette import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    SUBTLE,
    TITLE,
    paint,
    strip_ansi,
    supports_color,
)

_SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Platform icons for the rich message display (emoji, not colored text).
PLATFORM_ICONS = {
    "telegram": "✈️",
    "telegram-bot": "🤖",
    "local": "💻",
    "discord": "🎮",
    "whatsapp": "💬",
}

_UP_ARROW = "\033[A"
_CLEAR_LINE = "\033[2K\r"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


def sparkline(values: Iterable[float], *, width: int = 24, color: bool | None = None) -> str:
    """Tiny inline bar chart, e.g. ``▁▂▄▇█``. Empty input renders dim dashes."""
    vals = [max(0.0, float(v)) for v in values]
    if not vals:
        return paint("─" * width, DIM, color=color)
    peak = max(vals) or 1.0
    n = len(_SPARK_CHARS) - 1
    chars = [_SPARK_CHARS[min(n, int(v / peak * n))] for v in vals]
    # Resample to width.
    if len(chars) > width:
        step = len(chars) / width
        chars = [chars[int(i * step)] for i in range(width)]
    elif len(chars) < width:
        chars = chars + [_SPARK_CHARS[0]] * (width - len(chars))
    return paint("".join(chars), CYAN, color=color)


class ProgressBar:
    """Thread-safe progress bar with ETA. Pure ANSI, Termux-safe.

    Usage::

        bar = ProgressBar("downloading", total=100)
        for i in ...:
            bar.update(i)
        bar.done()
    """

    def __init__(
        self,
        label: str,
        total: float,
        *,
        width: int | None = None,
        color: bool | None = None,
        out: Any = None,
        show_eta: bool = True,
    ) -> None:
        self.label = label
        self.total = max(1e-9, float(total))
        self.width = width or min(40, max(20, (shutil.get_terminal_size((80, 24)).columns - 40)))
        self.color = supports_color() if color is None else color
        self.out = out or sys.stderr
        self.show_eta = show_eta
        self._lock = threading.Lock()
        self._done = False
        self._start = time.monotonic()
        self._last_len = 0

    def _render(self, done: float) -> str:
        frac = min(1.0, max(0.0, done / self.total))
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        pct = f"{frac * 100:5.1f}%"
        elapsed = time.monotonic() - self._start
        eta = ""
        if self.show_eta and frac > 0.01 and frac < 1.0:
            remain = elapsed / frac * (1 - frac)
            eta = f" eta {remain:4.0f}s"
        line = f"{paint(self.label, CYAN, color=self.color)} [{paint(bar, GREEN, color=self.color)}] {paint(pct, BOLD, color=self.color)}{paint(eta, DIM, color=self.color)}"
        return line

    def update(self, done: float) -> None:
        with self._lock:
            if self._done:
                return
            line = self._render(done)
            # Erase previous line, write new one.
            pad = " " * max(0, self._last_len - len(strip_ansi(line)))
            self.out.write(f"\r{line}{pad}")
            self.out.flush()
            self._last_len = len(strip_ansi(line))

    def done(self, suffix: str = "done") -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
            line = self._render(self.total)
            pad = " " * max(0, self._last_len - len(strip_ansi(line)))
            self.out.write(f"\r{line}{pad} {paint(suffix, GREEN, color=self.color)}\n")
            self.out.flush()


class LiveScreen:
    """In-place redrawing screen for watch-mode dashboards.

    Usage::

        with LiveScreen(interval=2.0) as screen:
            while screen.tick():
                screen.draw(render_dashboard(snapshot()))
    """

    def __init__(self, interval: float = 2.0, *, color: bool | None = None, out: Any = None):
        self.interval = max(0.5, float(interval))
        self.color = supports_color() if color is None else color
        self.out = out or sys.stdout
        self._stop = False
        self._first = True

    def __enter__(self) -> "LiveScreen":
        if self.color:
            self.out.write(_HIDE_CURSOR)
            self.out.flush()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.color:
            self.out.write(_SHOW_CURSOR + "\n")
            self.out.flush()
        self._stop = True

    def tick(self) -> bool:
        """Sleep until the next frame; False when interrupted."""
        if self._stop:
            return False
        try:
            time.sleep(self.interval)
        except KeyboardInterrupt:
            return False
        return not self._stop

    def draw(self, text: str) -> None:
        if not self.color:
            self.out.write(text + "\n")
            self.out.flush()
            return
        if self._first:
            self.out.write("\033[2J\033[H")
            self._first = False
        else:
            self.out.write("\033[H")
        self.out.write(text)
        # Clear any leftover lines below.
        self.out.write("\033[J")
        self.out.flush()

    def stop(self) -> None:
        self._stop = True


def format_message_card(
    *,
    platform: str,
    sender: str,
    text: str,
    chat_title: str = "",
    timestamp: float | None = None,
    incoming: bool = True,
    color: bool | None = None,
) -> str:
    """Rich one-block rendering of a chat message for the console mirror.

    Example::

        ✈️ telegram · 12:04:33
        ┌─ Mary (@chfjdhx) in xauusd_sentinel_signal
        │ /game stats
    """
    icon = PLATFORM_ICONS.get((platform or "").lower(), "💭")
    ts = time.strftime("%H:%M:%S", time.localtime(timestamp or time.time()))
    head = f"{icon} {paint(platform, CYAN, color=color)} · {paint(ts, DIM, color=color)}"
    who = paint(sender or "?", BOLD, color=color)
    where = f" in {paint(chat_title, SUBTLE, color=color)}" if chat_title else ""
    direction = "→" if incoming else "←"
    body_lines = (text or "").splitlines() or [""]
    body = "\n".join(f"{paint('│', DIM, color=color)} {ln}" for ln in body_lines[:6])
    if len(body_lines) > 6:
        body += f"\n{paint('│', DIM, color=color)} {paint('…', DIM, color=color)}"
    return (
        f"{head}\n"
        f"{paint('┌─', DIM, color=color)} {direction} {who}{where}\n"
        f"{body}"
    )


__all__ = [
    "ProgressBar",
    "LiveScreen",
    "sparkline",
    "format_message_card",
    "PLATFORM_ICONS",
]
