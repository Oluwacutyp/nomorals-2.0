"""Console widgets: progress bars, sparklines, live screens, message cards.

All widgets are pure-stdlib ANSI. They emit nothing when color/tty is
unavailable (plain-text fallbacks), and never use black backgrounds or red.
"""

from __future__ import annotations

import logging
import os
import select
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .palette import (
    BOLD,
    BRIGHT_CYAN,
    BRIGHT_WHITE,
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
    truncate_visible,
    visible_width,
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
# Alternate screen buffer: the watch dashboard gets its own screen,
# fully isolated from log lines on the main screen. On exit the main
# screen (and cursor) is restored exactly as it was — no bleed-through,
# no residual corruption. Termux-safe (standard xterm sequence).
_ALT_SCREEN_ON = "\x1b[?1049h"
_ALT_SCREEN_OFF = "\x1b[?1049l"
_HOME = "\033[H"
_CLEAR_BELOW = "\033[J"


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

    def __enter__(self) -> "LiveScreen":
        if self.color:
            # Alternate screen: isolated buffer, restored on exit.
            self.out.write(_ALT_SCREEN_ON + _HIDE_CURSOR)
            self.out.flush()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.color:
            self.out.write(_SHOW_CURSOR + _ALT_SCREEN_OFF)
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
        # One atomic write: home, full frame, clear below. Any log line
        # that slipped in between frames is wiped by the next redraw,
        # and the alternate screen keeps the main terminal untouched.
        self.out.write(_HOME + text + _CLEAR_BELOW)
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
    "AVATAR",
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


# ═══════════════════════════════════════════════════════════════════════════
# Watch-mode terminal ownership: real single-keypress input + output guard.
# ═══════════════════════════════════════════════════════════════════════════

class _KeyReader:
    """Single-keypress input for watch mode.

    Uses termios raw mode when available (Linux/Termux) so ``1/2/3/4/d/q``
    fire on a bare keypress — no Enter needed. Falls back to canonical
    line input (key + Enter) when raw mode is unavailable. Always
    restores the terminal on exit.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old: Any = None
        try:
            import termios

            fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(fd)
            self._fd = fd
        except Exception:  # noqa: BLE001 - not a tty / no termios
            self._fd = None

    @property
    def raw(self) -> bool:
        """True when single-keypress mode is available."""
        return self._fd is not None

    def __enter__(self) -> "_KeyReader":
        if self._fd is not None:
            try:
                import tty

                tty.setraw(self._fd)
            except Exception:  # noqa: BLE001 - fall back to canonical
                self._fd = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None and self._old is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass

    def get(self, timeout: float) -> str | None:
        """Wait up to ``timeout`` seconds for one keypress.

        Returns the key character (lowercased), ``"esc"`` for escape
        sequences, or None on timeout / EOF / error.
        """
        if self._fd is not None:
            return self._get_raw(timeout)
        return self._get_canonical(timeout)

    def _get_raw(self, timeout: float) -> str | None:
        try:
            ready, _, _ = select.select([self._fd], [], [], max(0.0, timeout))
        except Exception:  # noqa: BLE001
            return None
        if not ready:
            return None
        try:
            data = os.read(self._fd, 16)
        except OSError:
            return None
        if not data:
            return None  # EOF
        ch = data[:1].decode("utf-8", "replace")
        if ch == "\x1b":
            return "esc"  # arrow keys etc. — swallow the whole sequence
        if ch == "\x03":
            return "q"  # Ctrl-C arrives as ETX in raw mode
        return ch.lower() or None

    def _get_canonical(self, timeout: float) -> str | None:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout))
        except Exception:  # noqa: BLE001
            return None
        if not ready:
            return None
        try:
            line = sys.stdin.readline()
        except Exception:  # noqa: BLE001
            return None
        if not line:
            return None  # EOF
        return line.strip().lower()[:1] or None


class _WatchMuteFilter(logging.Filter):
    """Drops log records while the watch screen owns the terminal.

    Attached to every logging handler except DebugHub's capture handler,
    so the debug view keeps receiving telemetry while nothing prints.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not WatchHub.is_active()


class _ScreenGuard:
    """Gives the watch dashboard exclusive ownership of the terminal.

    The alternate-screen buffer alone proved insufficient on the owner's
    phone: gateway threads write to the same fd, so log lines landed
    inside the alt-screen buffer mid-frame. Two stronger layers:

    1. **Logging mute** — every logging handler except DebugHub's capture
       handler gets a mute filter while the guard is active. Logging
       produces zero terminal output, but telemetry still flows to the
       debug view (key ``d``).
    2. **fd redirect** — fds 1 and 2 are redirected to a spill file; the
       dashboard writes through a private duplicate of the original
       stdout fd. Any stray ``print()`` or C-level write lands in the
       spill file, never on screen.

    On exit everything is restored: fds, logging filters, cursor.
    When ``isolate`` is False (tests pass their own stream) only the
    logging mute applies — no fd games.
    """

    def __init__(self, *, isolate: bool = True, out: Any = None) -> None:
        self._isolate = isolate
        self._filter = _WatchMuteFilter()
        self._muted: list[logging.Handler] = []
        self._fd_out: int | None = None
        self._fd_err: int | None = None
        self._spill: Any = None
        self.tty: Any = out if out is not None else sys.stdout

    # ── logging mute ──

    def _mute_logging(self) -> None:
        seen: set[int] = set()
        handlers: list[logging.Handler] = list(logging.root.handlers)
        for lg in logging.Logger.manager.loggerDict.values():
            if isinstance(lg, logging.Logger):
                handlers.extend(lg.handlers)
        for handler in handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            if getattr(handler, "_devon_debug_capture", False):
                continue  # DebugHub keeps capturing for the debug view
            try:
                handler.addFilter(self._filter)
                self._muted.append(handler)
            except Exception:  # noqa: BLE001 - best effort
                pass

    def _unmute_logging(self) -> None:
        for handler in self._muted:
            try:
                handler.removeFilter(self._filter)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        self._muted.clear()

    # ── lifecycle ──

    def __enter__(self) -> "_ScreenGuard":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        self._mute_logging()
        if not self._isolate:
            return self
        try:
            if not os.isatty(1):
                return self
        except Exception:  # noqa: BLE001
            return self
        try:
            self._fd_out = os.dup(1)
            self._fd_err = os.dup(2)
            spill_path = os.path.join(tempfile.gettempdir(),
                                      "devon-watch-spill.log")
            self._spill = open(spill_path, "ab", buffering=0)
            os.dup2(self._spill.fileno(), 1)
            os.dup2(self._spill.fileno(), 2)
            # Private line-buffered handle to the real terminal — the
            # dashboard's only way out while fds 1/2 point at the spill.
            self.tty = os.fdopen(os.dup(self._fd_out), "w", buffering=1)
        except Exception:  # noqa: BLE001 - isolation is best-effort
            self.tty = sys.stdout
            self._fd_out = self._fd_err = None
            self._spill = None
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.tty.flush()
        except Exception:  # noqa: BLE001
            pass
        if self._fd_out is not None:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:  # noqa: BLE001
                    pass
            try:
                os.dup2(self._fd_out, 1)
                os.dup2(self._fd_err, 2)
            except Exception:  # noqa: BLE001
                pass
            for fh in (self.tty, self._spill):
                try:
                    if fh is not None:
                        fh.close()
                except Exception:  # noqa: BLE001
                    pass
            for fd in (self._fd_out, self._fd_err):
                try:
                    if fd is not None:
                        os.close(fd)
                except Exception:  # noqa: BLE001
                    pass
            self.tty = sys.stdout
            self._fd_out = self._fd_err = None
            self._spill = None
        self._unmute_logging()


# View ids for the watch screen's keyboard switching.
WATCH_VIEWS = ("status", "games", "jobs", "brain", "debug")
WATCH_VIEW_KEYS = {
    "1": "status",
    "2": "games",
    "3": "jobs",
    "4": "brain",
    "d": "debug",
}
#: Hotkey shown in the header tabs, per view.
_VIEW_HOTKEY = {"status": "1", "games": "2", "jobs": "3", "brain": "4",
                "debug": "d"}

#: Miniature ninja avatar for the header — the terminal can't show the
#: owner's real ninja avatar, so this stands in next to the DEVON title.
AVATAR = "🥷"

class GodScreen:
    """Full-screen live console: typographic header, view pane, message feed.

    Layout (terminal-size aware, every line width-truncated)::

        DEVON · live ⠋                    ⏱ 2m 30s  📨 49  🧠 groq
        universal agent os · no-morals 2.0
        [1] status  [2] games  [3] jobs  [4] brain  [d] debug
        ─────────────────────────────────────────────────────────
        ┌─ <view> ──────────────────────────────────────────────┐
        │ view content (status / games / jobs / brain / debug)   │
        └───────────────────────────────────────────────────────┘
        ┌─ messages (live) ─────────────────────────────────────┐
        │ compact feed lines                                      │
        └───────────────────────────────────────────────────────┘
        status bar · key hints

    Keys work on a bare keypress (termios raw mode; canonical fallback):
    ``1/2/3/4/d`` switch views, ``q``/Esc/Ctrl-C quits. While active the
    :class:`_ScreenGuard` owns the terminal: logging is muted (except the
    debug telemetry capture) and fds 1/2 are redirected to a spill file,
    so no log line can ever corrupt the frame.
    """

    HEADER_H = 4   # 4-row typographic header + separator
    STATUS_H = 2

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
        # fd isolation only when writing to the real stdout (not tests).
        self._isolate = out is None
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
        from .debug import DebugHub

        DebugHub.install()
        guard = _ScreenGuard(isolate=self._isolate, out=self.out)
        tty_out = self.out
        try:
            with guard:
                tty_out = guard.tty
                # Alternate screen buffer: isolated screen, restored on
                # exit. (The guard's fd redirect + log mute are the real
                # corruption fix; this keeps the main screen pristine.)
                tty_out.write(_ALT_SCREEN_ON + _HIDE_CURSOR)
                tty_out.flush()
                try:
                    with _KeyReader() as keys:
                        while not self._stop:
                            self._render_frame(tty_out)
                            if not self._wait_key(keys):
                                break
                except KeyboardInterrupt:
                    self._stop = True  # Ctrl-C is a documented way out
                finally:
                    # Exit the alt screen BEFORE the guard closes the
                    # tty handle — otherwise this write is swallowed.
                    try:
                        tty_out.write(_SHOW_CURSOR + _ALT_SCREEN_OFF)
                        tty_out.flush()
                    except Exception:  # noqa: BLE001 - best-effort
                        pass
        finally:
            WatchHub.set_active(False)
        return paint("exited live dashboard", DIM)

    # ── internals ──

    def _wait_key(self, keys: "_KeyReader | None" = None) -> bool:
        """Wait up to ``interval`` for a keypress. False = quit requested."""
        if keys is None:  # pragma: no cover - tests patch this method
            try:
                time.sleep(self.interval)
            except KeyboardInterrupt:
                return False
            return not self._stop
        if not self._stdin_ok:
            try:
                time.sleep(self.interval)
            except KeyboardInterrupt:
                return False
            return not self._stop
        try:
            key = keys.get(self.interval)
        except KeyboardInterrupt:
            return False
        if key is None:  # timeout → refresh
            return not self._stop
        if key in ("q", "esc", "\x03"):
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

    # ── frame assembly ──

    def _render_frame(self, out: Any = None) -> None:
        """Render one frame: header, content, feed, status. Simple layout.

        No box-drawing panes — just clean sections separated by rules.
        Every line is width-truncated so nothing wraps or overlaps.
        """
        from . import dashboard as _d

        out = out if out is not None else self.out
        snap = self._snap()
        cols, rows = shutil.get_terminal_size((80, 24))
        width = max(40, cols)
        height = max(16, rows)

        def _t(line: str) -> str:
            return truncate_visible(line, width)

        lines: list[str] = []

        # ── header: ninja avatar beside title + tabs, then a rule ──
        # The owner's ninja (7 rows) sits left of the title block.
        # On narrow terminals (<64 cols) fall back to the plain title line.
        spin = _SPINNER[self._frame % len(_SPINNER)]
        self._frame += 1
        title = (gradient_text("DEVON", 51, 201, color=self.color)
                 + paint(" · live ", TITLE, color=self.color)
                 + paint(spin, CYAN, color=self.color))
        tabs = "  ".join(
            paint(f"[{_VIEW_HOTKEY[v]}] {v}",
                  BRIGHT_CYAN + BOLD if v == self._view else DIM,
                  color=self.color)
            for v in WATCH_VIEWS
        )
        subtitle = paint("universal agent os · no-morals 2.0", DIM,
                         color=self.color)
        if width >= 64:
            from .avatar import NINJA_MINI_HEIGHT, ninja_mini_lines
            from .palette import visible_width

            ninja = ninja_mini_lines(color=self.color)
            avatar_w = max(visible_width(ln) for ln in ninja)
            # Right-side block: title, tabs, subtitle, then blank rows
            # to match the avatar height.
            right = [f"  {title}", f"  {tabs}", f"  {subtitle}"]
            while len(right) < NINJA_MINI_HEIGHT:
                right.append("")
            for i in range(NINJA_MINI_HEIGHT):
                left = ninja[i]
                pad = " " * max(0, avatar_w - visible_width(left))
                gap = "   "
                lines.append(_t(f"{left}{pad}{gap}{right[i]}"))
        else:
            lines.append(_t(f"  {title}   {tabs}"))
        lines.append(_t(paint("─" * width, SUBTLE, color=self.color)))
        header_h = len(lines)

        # ── content: the current view, plain lines ──
        # Fixed lines: header_h + 1 view label + 1 feed label + feed_h +
        # 1 separator + 1 status. Content fills the rest.
        feed_h = 5
        content_h = max(4, height - (header_h + 4 + feed_h))
        view_text = _d.render_view(snap, self._view, color=self.color,
                                   bare=True)
        content_lines = [
            _t(ln) for ln in view_text.splitlines()[:content_h]
        ]
        while len(content_lines) < content_h:
            content_lines.append("")
        view_label = paint(f"── {self._view} ", TITLE + BOLD, color=self.color)
        lines.append(_t(view_label + paint("─" * width, SUBTLE, color=self.color)))
        lines.extend(content_lines)

        # ── feed: title + up to 5 recent messages ──
        feed = WatchHub.feed()
        unread = feed.unread
        feed.mark_read()
        lines.append(_t(paint("── messages ", TITLE + BOLD, color=self.color)
                        + paint("─" * width, SUBTLE, color=self.color)))
        events = feed.recent(feed_h)
        if events:
            for ev in events[-feed_h:]:
                lines.append(_t("  " + format_feed_line(ev, color=self.color)))
            for _ in range(feed_h - len(events)):
                lines.append("")
        else:
            lines.append(_t(paint("  (quiet — new messages appear here)", DIM,
                                  color=self.color)))
            for _ in range(feed_h - 1):
                lines.append("")

        # ── status bar (1 line, pinned to bottom) ──
        lines.append(_t(paint("─" * width, SUBTLE, color=self.color)))
        lines.append(
            _t(_d.render_statusbar(snap, self._view, unread=unread,
                                   color=self.color))
        )

        # Pin to exactly the terminal height.
        while len(lines) < height:
            lines.append("")
        frame = "\n".join(_t(ln) for ln in lines[:height])
        out.write(_HOME + frame + _CLEAR_BELOW)
        out.flush()

    def _header_wide(self, snap: dict[str, Any], width: int) -> list[str]:
        """4-row typographic header: gradient title, subtitle, view tabs.

        Deliberately no ASCII art — clean typography beats a bad figure.
        Stats live in the status bar; the header stays minimal.
        """
        spin = _SPINNER[self._frame % len(_SPINNER)]
        self._frame += 1
        title = (gradient_text("DEVON", 51, 201, color=self.color)
                 + paint(" · live ", TITLE, color=self.color)
                 + paint(spin, CYAN, color=self.color))
        subtitle = paint("  universal agent os · no-morals 2.0", DIM,
                         color=self.color)
        tabs = "   ".join(
            paint(f"[{_VIEW_HOTKEY[v]}] {v}",
                  BRIGHT_CYAN + BOLD if v == self._view else DIM,
                  color=self.color)
            for v in WATCH_VIEWS
        )
        return [
            f"  {title}",
            subtitle,
            f"  {tabs}",
            paint("─" * width, SUBTLE, color=self.color),
        ]

    def _header_narrow(self, width: int) -> list[str]:
        """Fallback header for narrow terminals: title + tabs, one row."""
        spin = _SPINNER[self._frame % len(_SPINNER)]
        self._frame += 1
        title = (paint("🥷 ", CYAN, color=self.color)
                 + gradient_text("DEVON · live", 51, 201, color=self.color)
                 + paint(f" {spin} ", CYAN, color=self.color))
        tabs = " ".join(
            paint(f"[{_VIEW_HOTKEY[v]}]",
                  BRIGHT_CYAN + BOLD if v == self._view else DIM,
                  color=self.color)
            for v in WATCH_VIEWS
        )
        return [
            truncate_visible(f"  {title}  {tabs}", width),
            paint("─" * width, SUBTLE, color=self.color),
        ]


def _fmt_uptime(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
