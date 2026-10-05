"""Console widgets: progress bars, sparklines, live screens, message cards.

All widgets are pure-stdlib ANSI. They emit nothing when color/tty is
unavailable (plain-text fallbacks), and never use black backgrounds or red.
"""

from __future__ import annotations

import select
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .palette import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    SUBTLE,
    TITLE,
    WARN,
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
    "GodScreen",
    "MessageEvent",
    "MessageFeed",
    "WatchHub",
    "sparkline",
    "barchart",
    "gradient_text",
    "format_message_card",
    "format_feed_line",
    "PLATFORM_ICONS",
]


# ═══════════════════════════════════════════════════════════════════════════
# God-tier watch mode: message feed, watch hub, split-pane live screen.
# ═══════════════════════════════════════════════════════════════════════════

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class MessageEvent:
    """One inbound/outbound chat message, for the watch-mode feed."""

    platform: str = "?"
    sender: str = "?"
    text: str = ""
    chat_title: str = ""
    timestamp: float = 0.0
    incoming: bool = True

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class MessageFeed:
    """Thread-safe ring buffer of recent message events.

    The gateway's console mirror pushes here; the watch screen drains
    from here. Never blocks, never raises.
    """

    def __init__(self, capacity: int = 50) -> None:
        self._capacity = max(10, capacity)
        self._lock = threading.Lock()
        self._events: list[MessageEvent] = []
        self._unread = 0

    def push(self, event: MessageEvent) -> None:
        try:
            with self._lock:
                self._events.append(event)
                if len(self._events) > self._capacity:
                    del self._events[: len(self._events) - self._capacity]
                self._unread += 1
        except Exception:  # noqa: BLE001 - feed must never break the caller
            pass

    def recent(self, n: int) -> list[MessageEvent]:
        with self._lock:
            return list(self._events[-max(0, n):])

    @property
    def unread(self) -> int:
        with self._lock:
            return self._unread

    def mark_read(self) -> int:
        """Reset the unread counter; returns how many were unread."""
        with self._lock:
            n = self._unread
            self._unread = 0
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class WatchHub:
    """Coordinates watch mode with the console message mirror.

    When a ``dashboard --watch`` session is active, the gateway mirror
    must NOT print (it would flash over the dashboard). Instead it
    pushes into the shared feed, and the watch screen renders the feed
    in its own pane. Thread-safe; the mirror runs on gateway threads.
    """

    _lock = threading.Lock()
    _active = False
    _feed = MessageFeed()

    @classmethod
    def set_active(cls, active: bool) -> None:
        with cls._lock:
            cls._active = bool(active)
            if active:
                cls._feed.mark_read()

    @classmethod
    def is_active(cls) -> bool:
        with cls._lock:
            return cls._active

    @classmethod
    def feed(cls) -> MessageFeed:
        return cls._feed


def format_feed_line(event: MessageEvent, *, color: bool | None = None) -> str:
    """Compact one-line feed rendering: ``✈️ 15:32 Mary: /game stats``."""
    icon = PLATFORM_ICONS.get((event.platform or "").lower(), "💭")
    ts = time.strftime("%H:%M", time.localtime(event.timestamp))
    arrow = "→" if event.incoming else "←"
    sender = (event.sender or "?")[:24]
    text = " ".join((event.text or "").split())
    if len(text) > 90:
        text = text[:87] + "…"
    where = f" @{event.chat_title}" if event.chat_title else ""
    line = (
        f"{icon} {paint(ts, DIM, color=color)} "
        f"{paint(arrow, CYAN, color=color)} "
        f"{paint(sender, BOLD, color=color)}"
        f"{paint(where, SUBTLE, color=color)}: {text}"
    )
    return line


def barchart(
    items: list[tuple[str, float]],
    *,
    width: int = 18,
    color: bool | None = None,
    bar_color: str = CYAN,
) -> list[str]:
    """Horizontal bar chart lines: ``[('mafia', 12), ...]``.

    Returns one string per item: ``label  ████████  12``.
    Empty input returns a single dim placeholder line.
    """
    if not items:
        return [paint("(no data)", DIM, color=color)]
    peak = max((v for _, v in items), default=0) or 1.0
    label_w = max(len(str(label)) for label, _ in items)
    lines: list[str] = []
    for label, value in items:
        frac = max(0.0, min(1.0, float(value) / peak))
        filled = int(width * frac)
        bar = "█" * filled + "░" * (width - filled)
        num = f"{value:g}"
        lines.append(
            f"  {paint(str(label).ljust(label_w), SUBTLE, color=color)} "
            f"{paint(bar, bar_color, color=color)} "
            f"{paint(num, BOLD, color=color)}"
        )
    return lines


def gradient_text(
    text: str,
    start: int,
    end: int,
    *,
    color: bool | None = None,
) -> str:
    """Per-character 256-color gradient from ``start`` to ``end``.

    ``start``/``end`` are 256-color palette indexes (e.g. 51 → 201 for
    cyan→magenta). Termux-safe; plain text when color is off.
    """
    use_color = supports_color() if color is None else color
    if not use_color or not text:
        return text
    n = len(text)
    out: list[str] = []
    for i, ch in enumerate(text):
        frac = i / max(1, n - 1)
        code = int(round(start + (end - start) * frac))
        code = max(0, min(255, code))
        out.append(f"\033[38;5;{code}m{ch}")
    out.append("\033[0m")
    return "".join(out)


# View ids for the watch screen's keyboard switching.
WATCH_VIEWS = ("status", "games", "jobs", "brain")
WATCH_VIEW_KEYS = {"1": "status", "2": "games", "3": "jobs", "4": "brain"}


class GodScreen:
    """Split-pane live console: dashboard on top, message feed below.

    Layout (terminal-height aware)::

        ┌─ header (view tabs, spinner) ─────────────────────────────┐
        │ dashboard view content (status/games/jobs/brain)           │
        ├─ messages ────────────────────────────────────────────────┤
        │ feed lines (compact cards)                                 │
        ├─ status bar ──────────────────────────────────────────────┤
        │ uptime · msg rate · games · brain · unread · key hints     │
        └───────────────────────────────────────────────────────────┘

    Keys (canonical mode — type + Enter): 1/2/3/4 switch views, q quits.
    Ctrl-C also quits. Incoming messages never print over the screen —
    the mirror routes them into :class:`WatchHub`'s feed instead.
    """

    HEADER_H = 2
    STATUS_H = 2
    FEED_H = 7

    def __init__(
        self,
        interval: float = 2.0,
        *,
        snapshot: Callable[[], dict[str, Any]] | None = None,
        color: bool | None = None,
        out: Any = None,
    ) -> None:
        self.interval = max(0.5, float(interval))
        self._snapshot = snapshot or (lambda: {})
        self.color = supports_color() if color is None else color
        self.out = out or sys.stdout
        self._stop = False
        self._view = "status"
        self._frame = 0
        self._stdin_ok = bool(
            getattr(sys.stdin, "isatty", lambda: False)()
        )

    # ── public ──

    @property
    def view(self) -> str:
        return self._view

    def stop(self) -> None:
        self._stop = True

    def run(self) -> str:
        """Blocking watch loop. Returns an exit message for the console."""
        if not self.color:
            return paint(
                "watch mode needs a real terminal — "
                "type 'dashboard' for a static snapshot instead.",
                DIM,
            )
        WatchHub.set_active(True)
        try:
            self.out.write("\033[2J\033[H\033[?25l")
            self.out.flush()
            while not self._stop:
                self._render_frame()
                if not self._wait_key():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            WatchHub.set_active(False)
            try:
                self.out.write("\033[?25h\n")
                self.out.flush()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        return paint("exited live dashboard", DIM)

    # ── internals ──

    def _wait_key(self) -> bool:
        """Wait up to ``interval`` for a keypress. False = quit requested."""
        if not self._stdin_ok:
            try:
                time.sleep(self.interval)
            except KeyboardInterrupt:
                return False
            return not self._stop
        try:
            ready, _, _ = select.select([sys.stdin], [], [], self.interval)
        except (OSError, ValueError, KeyboardInterrupt):
            return False
        if not ready:
            return not self._stop
        try:
            line = sys.stdin.readline()
        except (OSError, KeyboardInterrupt):
            return False
        if not line:  # EOF
            return False
        key = line.strip().lower()[:1]
        if key in ("q", "\x03"):  # q or Ctrl-C
            return False
        if key in WATCH_VIEW_KEYS:
            self._view = WATCH_VIEW_KEYS[key]
        return not self._stop

    def _snap(self) -> dict[str, Any]:
        try:
            snap = self._snapshot()
            return snap if isinstance(snap, dict) else {}
        except Exception:  # noqa: BLE001 - dashboard is best-effort
            return {}

    def _render_frame(self) -> None:
        from . import dashboard as _d

        snap = self._snap()
        cols, rows = shutil.get_terminal_size((80, 24))
        feed_h = min(self.FEED_H, max(3, rows // 4))
        dash_h = max(6, rows - self.HEADER_H - feed_h - self.STATUS_H)
        width = max(40, cols)

        lines: list[str] = []
        # Header with spinner + view tabs.
        spin = _SPINNER[self._frame % len(_SPINNER)]
        self._frame += 1
        title = gradient_text("  DEVON · live", 51, 201, color=self.color)
        tabs = "  ".join(
            paint(f"[{i + 1}] {v}", BOLD if v == self._view else DIM, color=self.color)
            for i, v in enumerate(WATCH_VIEWS)
        )
        lines.append(f"{title} {paint(spin, CYAN, color=self.color)}   {tabs}")
        lines.append(paint("─" * min(width, 100), SUBTLE, color=self.color))

        # Dashboard view.
        view_text = _d.render_view(snap, self._view, color=self.color)
        view_lines = view_text.splitlines()[:dash_h]
        lines.extend(view_lines)

        # Feed pane.
        feed = WatchHub.feed()
        unread = feed.unread
        feed.mark_read()
        lines.append(
            paint(f"─ messages (live) ─", SUBTLE, color=self.color)
        )
        events = feed.recent(feed_h - 1)
        if events:
            for ev in events[-(feed_h - 1):]:
                lines.append(format_feed_line(ev, color=self.color)[:width])
        else:
            lines.append(paint("  (quiet — new messages appear here)", DIM, color=self.color))

        # Status bar.
        lines.append(paint("─" * min(width, 100), SUBTLE, color=self.color))
        lines.append(_d.render_statusbar(snap, self._view, unread=unread, color=self.color)[:width])

        # Write: home, all lines, clear leftovers.
        self.out.write("\033[H")
        self.out.write("\n".join(lines))
        self.out.write("\033[J")
        self.out.flush()
