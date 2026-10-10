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


class InputLine:
    """A real input line (prompt_toolkit idioms, stdlib-only).

    - command history: Up/Down arrows (in-memory; optional file persistence)
    - Tab completion via ``completer`` with a candidate popup
    - emacs bindings: Ctrl-A/E home/end, Ctrl-K kill-to-end, Ctrl-U kill
      line, Ctrl-W delete word, Ctrl-C clears the line (never exits)
    - Left/Right/Home/End/Delete keys, bracketed-paste support
    - syntax highlight via ``highlighter`` (paints the raw buffer)

    Feed it decoded keys: single chars, or escape sequences like
    ``"\\x1b[A"`` (Up). Bracketed paste arrives as one token:
    ``"\\x1b[200~<text>\\x1b[201~"``.
    Returns ``"submit"`` on Enter, ``"cancel"`` on Esc, ``"complete"``
    when Tab opened the popup, else None.
    """

    def __init__(
        self,
        *,
        completer: Callable[[str], list[str]] | None = None,
        highlighter: Callable[[str], str] | None = None,
        history_path: str | None = None,
        history_size: int = 200,
    ) -> None:
        self.completer = completer
        self.highlighter = highlighter
        self.buf = ""
        self.cursor = 0
        self.history: list[str] = []
        self._hist_idx = -1  # -1 = not browsing
        self._history_size = max(10, history_size)
        self._history_path = history_path
        self.popup: list[str] = []
        self._pending_submit = ""
        if history_path:
            self.load_history(history_path)

    # ── history ──

    def load_history(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                self.history = [ln.rstrip("\n") for ln in fh][-self._history_size:]
        except Exception:  # noqa: BLE001 - history is best-effort
            pass

    def save_history(self, path: str | None = None) -> None:
        target = path or self._history_path
        if not target:
            return
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.history[-self._history_size:]))
        except Exception:  # noqa: BLE001 - history is best-effort
            pass

    def _push_history(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if self.history and self.history[-1] == line:
            return
        self.history.append(line)
        del self.history[: -self._history_size]

    def _hist_prev(self) -> None:
        if not self.history:
            return
        if self._hist_idx == -1:
            self._hist_idx = len(self.history) - 1
        elif self._hist_idx > 0:
            self._hist_idx -= 1
        self.buf = self.history[self._hist_idx]
        self.cursor = len(self.buf)

    def _hist_next(self) -> None:
        if self._hist_idx == -1:
            return
        if self._hist_idx < len(self.history) - 1:
            self._hist_idx += 1
            self.buf = self.history[self._hist_idx]
        else:
            self._hist_idx = -1
            self.buf = ""
        self.cursor = len(self.buf)

    # ── editing primitives ──

    def _insert(self, text: str) -> None:
        # Multiline paste: collapse to one line (input is single-line).
        text = " ".join(text.split("\n"))
        self.buf = self.buf[: self.cursor] + text + self.buf[self.cursor:]
        self.cursor += len(text)
        self._hist_idx = -1
        self.popup = []

    def _delete_word_before(self) -> None:
        end = self.cursor
        start = end
        while start > 0 and self.buf[start - 1] == " ":
            start -= 1
        while start > 0 and self.buf[start - 1] != " ":
            start -= 1
        self.buf = self.buf[:start] + self.buf[end:]
        self.cursor = start

    # ── key handling ──

    def key(self, key: str) -> str | None:
        """Feed one decoded key. Returns submit/cancel/complete or None."""
        # Bracketed paste: one token.
        if key.startswith("\x1b[200~"):
            inner = key[len("\x1b[200~"):]
            if inner.endswith("\x1b[201~"):
                inner = inner[: -len("\x1b[201~")]
            self._insert(inner)
            return None
        if key in ("\r", "\n"):
            line = self.buf
            self._push_history(line)
            self.buf = ""
            self.cursor = 0
            self._hist_idx = -1
            self.popup = []
            self._pending_submit = line
            return "submit"
        if key == "\x1b":  # bare Esc cancels the line
            self.buf = ""
            self.cursor = 0
            self.popup = []
            return "cancel"
        if key == "\t":
            return self._complete()
        # Arrows / Home / End / Delete.
        if key in ("\x1b[A", "\x1bOA"):
            self._hist_prev()
            return None
        if key in ("\x1b[B", "\x1bOB"):
            self._hist_next()
            return None
        if key in ("\x1b[C", "\x1bOC"):
            self.cursor = min(len(self.buf), self.cursor + 1)
            return None
        if key in ("\x1b[D", "\x1bOD"):
            self.cursor = max(0, self.cursor - 1)
            return None
        if key in ("\x1b[H", "\x1b[1~", "\x1bOH"):
            self.cursor = 0
            return None
        if key in ("\x1b[F", "\x1b[4~", "\x1bOF"):
            self.cursor = len(self.buf)
            return None
        if key in ("\x1b[3~",):  # Delete
            if self.cursor < len(self.buf):
                self.buf = self.buf[: self.cursor] + self.buf[self.cursor + 1:]
            return None
        # Emacs bindings.
        if key == "\x01":  # Ctrl-A
            self.cursor = 0
            return None
        if key == "\x05":  # Ctrl-E
            self.cursor = len(self.buf)
            return None
        if key == "\x0b":  # Ctrl-K kill to end
            self.buf = self.buf[: self.cursor]
            return None
        if key == "\x15":  # Ctrl-U kill whole line
            self.buf = ""
            self.cursor = 0
            return None
        if key == "\x17":  # Ctrl-W delete word
            self._delete_word_before()
            return None
        if key == "\x03":  # Ctrl-C clears the line, never exits
            self.buf = ""
            self.cursor = 0
            self.popup = []
            return None
        if key in ("\x7f", "\x08"):  # Backspace
            if self.cursor > 0:
                self.buf = self.buf[: self.cursor - 1] + self.buf[self.cursor:]
                self.cursor -= 1
                self.popup = []
            return None
        if len(key) == 1 and key.isprintable():
            self._insert(key)
            return None
        return None

    def _complete(self) -> str | None:
        """Tab completion: popup candidates, Tab cycles, Enter picks."""
        if not self.completer:
            return None
        # Complete the word under/at the cursor.
        head = self.buf[: self.cursor]
        m = len(head)
        while m > 0 and not head[m - 1].isspace():
            m -= 1
        prefix = head[m:]
        cands = [c for c in (self.completer(prefix) or []) if c.startswith(prefix)]
        if not cands:
            return None
        if len(cands) == 1:
            tail = self.buf[self.cursor:]
            self.buf = head[:m] + cands[0] + tail
            self.cursor = m + len(cands[0])
            return None
        # Multiple: popup; Tab cycles through them.
        if not self.popup or self.popup[0] != prefix:
            self.popup = [prefix] + cands
        else:
            self.popup.append(self.popup.pop(1))
        pick = self.popup[1]
        tail = self.buf[self.cursor:]
        self.buf = head[:m] + pick + tail
        self.cursor = m + len(pick)
        return "complete"

    # ── rendering ──

    def take_submit(self) -> str:
        """The line submitted by the last Enter (cleared on read)."""
        line, self._pending_submit = self._pending_submit, ""
        return line

    def render(self, prompt: str = "❯ ", *, color: bool | None = None,
               width: int = 80) -> list[str]:
        """Input row + optional completion popup row."""
        from .palette import truncate_visible

        body = self.highlighter(self.buf) if self.highlighter else self.buf
        # Cursor: split body at the cursor and draw a block.
        before = body[: self.cursor]
        after = body[self.cursor:]
        line = prompt + before + paint("█", CYAN, color=color) + after
        rows = [truncate_visible(line, width)]
        if len(self.popup) > 1:
            cands = "  ".join(self.popup[1:6])
            more = f" (+{len(self.popup) - 6})" if len(self.popup) > 6 else ""
            rows.append(
                truncate_visible(
                    paint("  ↳ ", DIM, color=color)
                    + paint(cands + more, CYAN, color=color),
                    width,
                )
            )
        return rows


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
        completer: Callable[[str], list[str]] | None = None,
        highlighter: Callable[[str], str] | None = None,
        history_path: str | None = None,
    ):
        self._snapshot = snapshot or (lambda: {})
        self._on_command = on_command or (lambda cmd: f"echo: {cmd}")
        self._events = event_buffer or EventBuffer()
        self._input = InputLine(
            completer=completer,
            highlighter=highlighter,
            history_path=history_path,
        )
        self._view = "status"
        self._running = False
        self._width = 80
        self._height = 24
        self._output_lines: list[str] = []
        self._show_output = False  # command output takes the main pane
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
        from .palette import truncate_visible

        if self._show_output and self._output_lines:
            # Command output owns the main pane (latest lines, scrollable
            # region in spirit: newest at the bottom).
            tail = self._output_lines[-h:]
            result = [truncate_visible(ln, w) for ln in tail]
        else:
            snap = self._snapshot()
            try:
                content = render_view(snap, self._view, bare=True, color=self._color)
            except Exception:
                content = "view unavailable"
            lines = content.split("\n")
            result = [truncate_visible(ln, w) for ln in lines[:h]]
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
        rows = [paint("─" * w, DIM)]
        rows.extend(
            self._input.render(paint("  ❯ ", CYAN), color=self._color, width=w)
        )
        return rows

    def _input_height(self, w: int) -> int:
        # Separator row + input row (+ completion popup row when open).
        return 1 + len(self._input.render("", color=False, width=w))

    def _render_status(self, w: int) -> list[str]:
        hints = (
            f"  {paint('[1]', CYAN)}status {paint('[2]', CYAN)}games "
            f"{paint('[3]', CYAN)}jobs {paint('[4]', CYAN)}brain "
            f"{paint('[5]', CYAN)}adapters {paint('[6]', CYAN)}health "
            f"{paint('[d]', CYAN)}debug {paint('[q]', CYAN)}quit "
            f"{paint('↑↓ history · tab complete', DIM)}"
        )
        return [paint("─" * w, DIM), hints[:w]]

    def _render_frame(self) -> str:
        w, h = self._width, self._height
        input_h = self._input_height(w)
        main_h = h - self.HEADER_H - self.EVENT_H - input_h - self.STATUS_H
        main_h = max(5, main_h)

        lines = []
        lines.extend(self._render_header(w))
        lines.extend(self._render_main(w, main_h))
        lines.extend(self._render_events(w))
        input_rows = self._render_input(w)
        lines.extend(input_rows)
        lines.extend(self._render_status(w))

        # Ensure exact height
        while len(lines) < h:
            lines.append("")
        lines = lines[:h]

        # Move cursor to top, render all lines
        out = "\x1b[H"  # cursor home
        for line in lines:
            out += line + "\x1b[K\n"  # clear to end of line
        # Position cursor at the input row, after the prompt + cursor offset.
        input_row = self.HEADER_H + main_h + self.EVENT_H + 1
        cursor_col = 6 + self._input.cursor  # "  ❯ " = 4 cells + block + cursor
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
            while len(extra) < 8:
                r2, _, _ = select.select(target, [], [], 0.05)
                if not r2:
                    break
                c = stdin.read(1)
                extra += c
                if c.isalpha() or c == "~":
                    break
            ch += extra
            if ch == "\x1b[200~":
                # Bracketed paste: swallow everything up to the end marker
                # and deliver it as one token.
                body = ""
                while True:
                    r3, _, _ = select.select(target, [], [], 1.0)
                    if not r3:
                        break
                    body += stdin.read(1)
                    if body.endswith("\x1b[201~"):
                        break
                    if len(body) > 100_000:  # sanity cap
                        break
                ch += body
        return ch

    def _handle_key(self, key: str) -> bool:
        """Handle a keypress. Returns False to quit."""
        if key in ("q", "\x03") and not self._input.buf:
            # q / Ctrl-C on an empty line quits; Ctrl-C on a non-empty
            # line only clears it (InputLine handles that).
            return False
        if key == "\x1b":
            # Esc on empty input quits; on non-empty input cancels the line.
            if not self._input.buf:
                return False
        if key in ("1", "2", "3", "4", "5", "6", "d") and not self._input.buf:
            # Bare view keys only when the input line is empty (typing
            # "1" as a command must not switch views).
            views = {"1": "status", "2": "games", "3": "jobs", "4": "brain",
                     "5": "adapters", "6": "health", "d": "debug"}
            self._view = views[key]
            self._show_output = False
            return True
        action = self._input.key(key)
        if action == "submit":
            cmd = self._input.take_submit().strip()
            if cmd:
                try:
                    result = self._on_command(cmd)
                    self._output_lines.append(f"❯ {cmd}")
                    self._output_lines.extend(str(result).split("\n")[:50])
                    self._output_lines = self._output_lines[-200:]
                    self._show_output = True
                except Exception as exc:
                    self._output_lines.append(f"error: {exc}")
                    self._show_output = True
            return True
        if action == "cancel":
            self._show_output = False
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
                # Alternate screen + hide cursor (we draw our own) +
                # bracketed paste (multi-line paste arrives as one token).
                tty_out.write(_ALT_SCREEN_ON + _HIDE_CURSOR + "\x1b[?2004h")
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

                # Restore: leave paste mode, alt screen, cursor.
                tty_out.write("\x1b[?2004l" + _ALT_SCREEN_OFF + _SHOW_CURSOR)
                tty_out.flush()
                self._input.save_history()
                return "console closed"

        except Exception as exc:
            return f"console error: {exc}"
        finally:
            self._running = False
