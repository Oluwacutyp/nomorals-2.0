"""God-tier interactive console: fullscreen, event-isolated, live.

Architecture:
- Alternate screen buffer (like nano): takes over the terminal completely.
- Dedicated input line at the bottom that NEVER gets overwritten.
- Event feed in a separate pane: incoming messages render there, not on your input.
- Background event queue: events are buffered and rendered on the next frame,
  never mid-keystroke.
- Auto-scales to terminal width: every line is width-truncated.
- Cyber-stealth palette: cyan #00F0FF, amber #FFB000, icy #A3B8CC.

Layout:
    ┌─ header ─────────────────────────────────────────────┐
    │ DEVON · live                    ⏱ uptime  📨 msgs  🧠 brain │
    ├─ main ───────────────────────────────────────────────┤
    │                                                      │
    │  dashboard / view content                            │
    │                                                      │
    ├─ events ─────────────────────────────────────────────┤
    │  › [12:34] telegram: new message from ...            │
    │  › [12:35] scheduler: job completed                  │
    ├─ input ──────────────────────────────────────────────┤
    │  ❯ _                                                │
    └─ status ─────────────────────────────────────────────┘
      [1]status [2]games [3]jobs [4]brain [d]debug [q]quit
"""

from __future__ import annotations

import io
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable

from .palette import (
    paint, CYAN, YELLOW, BRIGHT_CYAN, DIM, GREEN, MAGENTA, BOLD, WHITE,
    supports_color,
)
from .widgets import _ScreenGuard, _ALT_SCREEN_ON, _ALT_SCREEN_OFF, _HIDE_CURSOR, _SHOW_CURSOR

# Palette aliases for the cyber-stealth theme
AMBER = YELLOW
ICY = BRIGHT_CYAN


# ── ninja ASCII art ──

NINJA_MARK = "﹝◉﹞"

BANNER = [
    "    ╔═══════════════════════════════════════╗",
    "    ║  ⬢ DEVON · no-morals 2.0             ║",
    "    ║  universal agent os                   ║",
    "    ╚═══════════════════════════════════════╝",
]


class EventBuffer:
    """Thread-safe event queue. Background threads push, UI thread drains."""

    def __init__(self, max_events: int = 100):
        self._queue: queue.Queue[str] = queue.Queue()
        self._events: list[str] = []
        self._max = max_events
        self._lock = threading.Lock()

    def push(self, event: str) -> None:
        """Called from any thread. Never blocks the UI."""
        self._queue.put(event)

    def drain(self) -> list[str]:
        """Called from UI thread. Returns new events since last drain."""
        new = []
        try:
            while True:
                new.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if new:
            with self._lock:
                self._events.extend(new)
                self._events = self._events[-self._max:]
        with self._lock:
            return list(self._events[-20:])  # last 20 for display


class GodConsole:
    """Fullscreen interactive console with event isolation."""

    HEADER_H = 3
    EVENT_H = 6
    INPUT_H = 2
    STATUS_H = 2

    def __init__(
        self,
        *,
        snapshot: Callable[[], dict[str, Any]] | None = None,
        on_command: Callable[[str], str] | None = None,
        event_buffer: EventBuffer | None = None,
    ):
        self._snapshot = snapshot or (lambda: {})
        self._on_command = on_command or (lambda cmd: f"echo: {cmd}")
        self._events = event_buffer or EventBuffer()
        self._input_buf = ""
        self._cursor_pos = 0
        self._view = "status"
        self._running = False
        self._width = 80
        self._height = 24
        self._output_lines: list[str] = []
        self._color = supports_color()

    def push_event(self, event: str) -> None:
        """External API: push a background event. Thread-safe."""
        self._events.push(event)

    def _get_size(self) -> tuple[int, int]:
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24

    def _truncate(self, s: str, width: int) -> str:
        """Truncate to width, preserving ANSI codes."""
        # Simple version: strip ANSI, truncate, re-add
        # For now, just truncate the visible length
        visible = ""
        ansi_buf = ""
        in_ansi = False
        for ch in s:
            if ch == "\x1b":
                in_ansi = True
                ansi_buf = ch
            elif in_ansi:
                ansi_buf += ch
                if ch == "m":
                    in_ansi = False
            else:
                if len(visible) >= width:
                    break
                visible += ch
        return s[:len(s) - (len(s) - len(visible) - len(ansi_buf))] if len(visible) >= width else s

    def _render_header(self, w: int) -> list[str]:
        snap = self._snapshot()
        uptime = snap.get("uptime_s", 0)
        mins = int(uptime // 60)
        brain = snap.get("llm", {}).get("active", "—")
        msgs = snap.get("traffic", {}).get("messages", 0)

        left = f"  {paint(NINJA_MARK, CYAN)} {paint('DEVON', BOLD)} {paint('· live', DIM)}"
        right = f"{paint('⏱', DIM)} {mins}m  {paint('📨', DIM)} {msgs}  {paint('🧠', DIM)} {brain}  "
        # Pad to width
        left_len = 18  # approx visible
        right_len = len(f"⏱ {mins}m  📨 {msgs}  🧠 {brain}  ")
        pad = max(1, w - left_len - right_len)
        lines = [
            left + " " * pad + right,
            paint("  universal agent os · no-morals 2.0", DIM),
            paint("─" * w, DIM),
        ]
        return lines

    def _render_main(self, w: int, h: int) -> list[str]:
        from .dashboard import render_view
        snap = self._snapshot()
        try:
            content = render_view(snap, self._view, bare=True, color=self._color)
        except Exception:
            content = "view unavailable"
        lines = content.split("\n")
        # Truncate each line to width, pad to height
        result = []
        for line in lines[:h]:
            # Simple truncation (ANSI-aware would be better)
            result.append(line[:w])
        while len(result) < h:
            result.append("")
        return result

    def _render_events(self, w: int) -> list[str]:
        events = self._events.drain()
        lines = [paint("─" * w, DIM)]
        for evt in events[-5:]:  # last 5 events
            ts = time.strftime("%H:%M")
            line = f"  {paint('›', AMBER)} {paint(ts, DIM)} {evt[:w-12]}"
            lines.append(line)
        while len(lines) < self.EVENT_H:
            lines.append("")
        return lines[:self.EVENT_H]

    def _render_input(self, w: int) -> list[str]:
        prompt = paint("  ❯ ", CYAN)
        # Show input buffer with cursor
        before = self._input_buf[:self._cursor_pos]
        after = self._input_buf[self._cursor_pos:]
        cursor = paint("█", CYAN)
        line = prompt + before + cursor + after
        return [
            paint("─" * w, DIM),
            line[:w],
        ]

    def _render_status(self, w: int) -> list[str]:
        hints = (
            f"  {paint('[1]', CYAN)}status {paint('[2]', CYAN)}games "
            f"{paint('[3]', CYAN)}jobs {paint('[4]', CYAN)}brain "
            f"{paint('[d]', CYAN)}debug {paint('[q]', CYAN)}quit"
        )
        return [paint("─" * w, DIM), hints[:w]]

    def _render_frame(self) -> str:
        w, h = self._width, self._height
        main_h = h - self.HEADER_H - self.EVENT_H - self.INPUT_H - self.STATUS_H
        main_h = max(5, main_h)

        lines = []
        lines.extend(self._render_header(w))
        lines.extend(self._render_main(w, main_h))
        lines.extend(self._render_events(w))
        lines.extend(self._render_input(w))
        lines.extend(self._render_status(w))

        # Ensure exact height
        while len(lines) < h:
            lines.append("")
        lines = lines[:h]

        # Move cursor to top, render all lines
        out = "\x1b[H"  # cursor home
        for line in lines:
            out += line + "\x1b[K\n"  # clear to end of line
        # Position cursor at input line
        input_row = self.HEADER_H + main_h + self.EVENT_H + 1
        cursor_col = 6 + self._cursor_pos  # "  ❯ " = 4 chars + cursor pos
        out += f"\x1b[{input_row + 1};{cursor_col + 1}H"
        return out

    def _wait_key(self, timeout: float) -> str | None:
        """Return a keypress if one arrives within ``timeout`` seconds, else None.

        ``None`` is a timer tick: the caller should re-render the frame and
        wait again, so background events (pushed via :meth:`push_event`) and
        snapshot changes appear live without any keypress.
        """
        stdin = sys.stdin
        try:
            fd = stdin.fileno()
        except (OSError, io.UnsupportedOperation):
            fd = None
        target = [fd] if fd is not None else [stdin]
        r, _, _ = select.select(target, [], [], timeout)
        if not r:
            return None
        ch = stdin.read(1)
        if ch == "\x1b":
            # Escape sequence (arrows, etc.): read the rest without blocking.
            extra = ""
            while len(extra) < 2:
                r2, _, _ = select.select(target, [], [], 0.05)
                if not r2:
                    break
                c = stdin.read(1)
                extra += c
                if c.isalpha() or c == "~":
                    break
            ch += extra
        return ch

    def _handle_key(self, key: str) -> bool:
        """Handle a keypress. Returns False to quit."""
        if key in ("q", "\x1b", "\x03"):  # q, Esc, Ctrl-C
            return False
        if key in ("1", "2", "3", "4", "d"):
            views = {"1": "status", "2": "games", "3": "jobs", "4": "brain", "d": "debug"}
            self._view = views[key]
            return True
        if key in ("\r", "\n"):  # Enter
            cmd = self._input_buf.strip()
            if cmd:
                try:
                    result = self._on_command(cmd)
                    self._output_lines.append(f"❯ {cmd}")
                    self._output_lines.append(str(result)[:500])
                except Exception as exc:
                    self._output_lines.append(f"error: {exc}")
            self._input_buf = ""
            self._cursor_pos = 0
            return True
        if key == "\x7f":  # Backspace
            if self._cursor_pos > 0:
                self._input_buf = (
                    self._input_buf[:self._cursor_pos - 1]
                    + self._input_buf[self._cursor_pos:]
                )
                self._cursor_pos -= 1
            return True
        if len(key) == 1 and key.isprintable():
            self._input_buf = (
                self._input_buf[:self._cursor_pos]
                + key
                + self._input_buf[self._cursor_pos:]
            )
            self._cursor_pos += 1
            return True
        return True

    def run(self) -> str:
        """Main loop. Takes over the terminal until quit.

        The frame re-renders on every tick (~1s) even with no input, so the
        display stays live: uptime, event feed and view content all refresh
        without a keypress. Raw mode is set once for the session (not per
        keystroke).
        """
        if not self._color:
            return "god console needs a real terminal"

        TICK = 1.0
        guard = _ScreenGuard(isolate=True)
        fd = sys.stdin.fileno()
        try:
            with guard:
                tty_out = guard.tty
                # Alternate screen + hide cursor (we draw our own)
                tty_out.write(_ALT_SCREEN_ON + _HIDE_CURSOR)
                tty_out.flush()

                # Raw mode once for the whole session.
                try:
                    old_attrs = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    raw = True
                except Exception:
                    old_attrs = None
                    raw = False

                try:
                    self._running = True
                    self._width, self._height = self._get_size()

                    while self._running:
                        # Check for resize
                        w, h = self._get_size()
                        if (w, h) != (self._width, self._height):
                            self._width, self._height = w, h
                            # Full clear on resize
                            tty_out.write("\x1b[2J")

                        # Render frame on every iteration (tick or keypress)
                        tty_out.write(self._render_frame())
                        tty_out.flush()

                        # Wait for a key, but wake up on the tick so the
                        # display never goes stale waiting for input.
                        try:
                            key = self._wait_key(TICK)
                        except Exception:
                            break

                        if key is None:
                            continue  # timer tick: re-render

                        if not self._handle_key(key):
                            break
                finally:
                    if raw and old_attrs is not None:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

                # Restore
                tty_out.write(_ALT_SCREEN_OFF + _SHOW_CURSOR)
                tty_out.flush()
                return "console closed"

        except Exception as exc:
            return f"console error: {exc}"
        finally:
            self._running = False
