"""TUI state and layout — pure, with no curses import.

Everything that can be reasoned about lives here: the scrollback, the input
buffer, cursor motion, which panel is focused, and how a screen of text is
computed from state. The curses driver in ``app.py`` only draws what this module
hands it and translates keypresses into calls here.

That split is the whole reason this file exists. A TUI written directly against
curses cannot be tested without a terminal, so its layout and key handling go
unverified until a human happens to press the right key. Here, they are ordinary
functions returning strings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

__all__ = ["Panel", "Line", "KeyAction", "TuiState", "render"]


class Panel(str, Enum):
    INPUT = "input"
    SCROLLBACK = "scrollback"
    STATUS = "status"


class KeyAction(str, Enum):
    """What a keypress means. The driver maps keys to these; nothing else."""

    NONE = "none"
    SUBMIT = "submit"
    CANCEL = "cancel"
    QUIT = "quit"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    PAGE_UP = "page_up"
    PAGE_DOWN = "page_down"
    SCROLL_TOP = "scroll_top"
    SCROLL_BOTTOM = "scroll_bottom"
    CLEAR = "clear"
    FOCUS_NEXT = "focus_next"
    HISTORY_PREV = "history_prev"
    HISTORY_NEXT = "history_next"
    CURSOR_LEFT = "cursor_left"
    CURSOR_RIGHT = "cursor_right"
    HOME = "home"
    END = "end"
    DELETE_CHAR = "delete_char"
    BACKSPACE = "backspace"
    TAB_COMPLETE = "tab_complete"


@dataclass
class Line:
    """One scrollback entry. ``kind`` drives colour in the driver."""

    text: str
    kind: str = "info"
    at: float = field(default_factory=time.time)

    def render(self, width: int) -> list[str]:
        """Wrap to width. Long lines become several rows, not a truncation."""
        if not self.text:
            return [""]
        rows: list[str] = []
        for paragraph in self.text.split("\n"):
            while len(paragraph) > width:
                cut = paragraph.rfind(" ", 0, width)
                if cut <= 0:
                    cut = width
                rows.append(paragraph[:cut])
                paragraph = paragraph[cut:].lstrip()
            rows.append(paragraph)
        return rows


@dataclass
class TuiState:
    """Everything the screen shows. No I/O, so it can be asserted on directly."""

    lines: list[Line] = field(default_factory=list)
    buffer: str = ""
    cursor: int = 0
    scroll: int = 0
    focus: Panel = Panel.INPUT
    history: list[str] = field(default_factory=list)
    history_index: int = -1
    busy: bool = False
    status: str = "ready"
    prompt: str = "> "
    max_history: int = 200
    max_lines: int = 2000

    # ── output ───────────────────────────────────────────────────────────────

    def say(self, text: str, kind: str = "info") -> None:
        self.lines.append(Line(text=text, kind=kind))
        if len(self.lines) > self.max_lines:
            del self.lines[: len(self.lines) - self.max_lines]
        self.scroll = 0

    def error(self, text: str) -> None:
        self.say(text, kind="error")

    def tool(self, name: str, detail: str = "") -> None:
        # No "[tool]" prefix: the driver already colours this row by kind, so the
        # label was printed twice.
        self.say(f"{name} {detail}".rstrip(), kind="tool")

    def clear(self) -> None:
        self.lines.clear()
        self.scroll = 0

    # ── input ────────────────────────────────────────────────────────────────

    def insert(self, text: str) -> None:
        if not text:
            return
        self.buffer = self.buffer[: self.cursor] + text + self.buffer[self.cursor :]
        self.cursor += len(text)

    def backspace(self) -> None:
        if self.cursor > 0:
            self.buffer = self.buffer[: self.cursor - 1] + self.buffer[self.cursor :]
            self.cursor -= 1

    def delete_char(self) -> None:
        if self.cursor < len(self.buffer):
            self.buffer = self.buffer[: self.cursor] + self.buffer[self.cursor + 1 :]

    def move(self, delta: int) -> None:
        self.cursor = max(0, min(len(self.buffer), self.cursor + delta))

    def home(self) -> None:
        self.cursor = 0

    def end(self) -> None:
        self.cursor = len(self.buffer)

    def submit(self) -> str:
        """Take the buffer, record it in history, and reset. Returns the text."""
        text = self.buffer.strip()
        if text:
            if not self.history or self.history[-1] != text:
                self.history.append(text)
                if len(self.history) > self.max_history:
                    del self.history[0]
        self.buffer = ""
        self.cursor = 0
        self.history_index = -1
        return text

    def history_prev(self) -> None:
        if not self.history:
            return
        if self.history_index < 0:
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        self.buffer = self.history[self.history_index]
        self.cursor = len(self.buffer)

    def history_next(self) -> None:
        if self.history_index < 0:
            return
        self.history_index += 1
        if self.history_index >= len(self.history):
            self.history_index = -1
            self.buffer = ""
        else:
            self.buffer = self.history[self.history_index]
        self.cursor = len(self.buffer)

    # ── scrolling ────────────────────────────────────────────────────────────

    def scroll_by(self, delta: int, *, viewport: int = 20) -> None:
        """``scroll`` is rows up from the bottom, so it never goes negative."""
        maximum = max(0, self._content_height(80) - viewport)
        self.scroll = max(0, min(maximum, self.scroll + delta))

    def scroll_top(self, *, viewport: int = 20) -> None:
        self.scroll = max(0, self._content_height(80) - viewport)

    def scroll_bottom(self) -> None:
        self.scroll = 0

    def _content_height(self, width: int) -> int:
        return sum(len(line.render(width)) for line in self.lines)

    # ── focus ────────────────────────────────────────────────────────────────

    def cycle_focus(self) -> None:
        order = [Panel.INPUT, Panel.SCROLLBACK]
        index = order.index(self.focus) if self.focus in order else 0
        self.focus = order[(index + 1) % len(order)]


DEFAULT_BINDINGS: dict[str, KeyAction] = {
    "\n": KeyAction.SUBMIT,
    "\r": KeyAction.SUBMIT,
    "\x03": KeyAction.CANCEL,
    "\x04": KeyAction.QUIT,
    "\x0c": KeyAction.CLEAR,
    "\t": KeyAction.FOCUS_NEXT,
    "\x1b[A": KeyAction.HISTORY_PREV,
    "\x1b[B": KeyAction.HISTORY_NEXT,
    "\x1b[D": KeyAction.CURSOR_LEFT,
    "\x1b[C": KeyAction.CURSOR_RIGHT,
    "\x01": KeyAction.HOME,
    "\x05": KeyAction.END,
    "\x08": KeyAction.BACKSPACE,
    "\x7f": KeyAction.BACKSPACE,
    "\x1b[3~": KeyAction.DELETE_CHAR,
}


def action_for(key: str, state: TuiState, bindings: dict[str, KeyAction] | None = None) -> KeyAction:
    """Map a raw key to an action, taking focus into account.

    Arrow keys scroll when the scrollback has focus and move the cursor when the
    input does. Same key, different meaning, decided here rather than in the
    driver so it can be tested.
    """
    table = bindings or DEFAULT_BINDINGS
    action = table.get(key)
    if action is not None:
        return action
    if state.focus is Panel.SCROLLBACK:
        return {
            "k": KeyAction.SCROLL_UP, "j": KeyAction.SCROLL_DOWN,
            "u": KeyAction.PAGE_UP, "d": KeyAction.PAGE_DOWN,
            "g": KeyAction.SCROLL_TOP, "G": KeyAction.SCROLL_BOTTOM,
        }.get(key, KeyAction.NONE)
    return KeyAction.NONE


@dataclass
class Rendered:
    """A finished screen: rows of (text, kind) plus where the cursor goes."""

    rows: list[tuple[str, str]] = field(default_factory=list)
    cursor_row: int = 0
    cursor_col: int = 0
    status: str = ""
    input_line: str = ""


def render(state: TuiState, *, width: int, height: int) -> Rendered:
    """Compute the screen. Pure: same state and size always gives same output.

    Layout: a status bar on top, the input line at the bottom, and the
    scrollback filling what is left, anchored to the bottom unless scrolled.
    """
    if width < 20 or height < 5:
        return Rendered(rows=[("terminal too small", "error")], status="resize the window")

    body_height = height - 2
    prompt_line = state.prompt + state.buffer

    rows: list[tuple[str, str]] = []
    wrapped: list[tuple[str, str]] = []
    for line in state.lines:
        for row in line.render(width):
            wrapped.append((row, line.kind))

    # Anchor to the bottom, then honour the scroll offset.
    end = len(wrapped) - state.scroll
    start = max(0, end - body_height)
    visible = wrapped[start:end]
    while len(visible) < body_height:
        visible.insert(0, ("", "blank"))

    rows.extend(visible)
    rows.append((prompt_line[:width], "input"))

    indicator = "…" if state.busy else ""
    status = f" {state.status} {indicator}".strip()
    right = f"[{state.focus.value}] "
    padding = max(1, width - len(status) - len(right))
    rows.insert(0, ((status + " " * padding + right)[:width], "status"))

    return Rendered(
        rows=rows,
        cursor_row=len(rows) - 1,
        cursor_col=min(width - 1, len(state.prompt) + state.cursor),
        status=status,
        input_line=prompt_line,
    )
