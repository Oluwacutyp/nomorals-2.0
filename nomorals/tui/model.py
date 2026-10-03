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

__all__ = [
    "Panel", "Line", "KeyAction", "TuiState", "render",
    "KEY_HELP", "SLASH_HELP", "help_overlay",
]


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
    HELP = "help"


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
    help_visible: bool = False
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

    def toggle_help(self) -> None:
        self.help_visible = not self.help_visible


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
    "?": KeyAction.HELP,
}


# ── help overlay ─────────────────────────────────────────────────────────────
# The single source of truth for what the keys do. The overlay in render() and
# the /help slash command both draw from these tables, so the docs cannot drift
# from the bindings above.

KEY_HELP: list[tuple[str, str]] = [
    ("Enter", "submit the line"),
    ("Ctrl-D", "quit"),
    ("Ctrl-C", "cancel the running command"),
    ("Ctrl-L", "clear the scrollback"),
    ("Tab", "switch panel focus"),
    ("Up / Down", "history (input) · scroll (scrollback)"),
    ("Left / Right", "move the cursor"),
    ("Ctrl-A / Ctrl-E", "line start / end"),
    ("Del", "delete char under cursor"),
    ("PgUp / PgDn", "page the scrollback"),
    ("j / k", "scroll one line (scrollback)"),
    ("u / d", "scroll one page (scrollback)"),
    ("g / G", "top / bottom (scrollback)"),
    ("? or F1", "toggle this help"),
]

SLASH_HELP: list[tuple[str, str]] = [
    ("/help", "list commands"),
    ("/tools", "list available tools"),
    ("/models", "list models"),
    ("/missions", "list missions"),
    ("/mem <text>", "remember something"),
    ("/recall <query>", "search memory"),
    ("/doctor", "environment info"),
    ("/clear", "clear the scrollback"),
    ("/quit", "exit the TUI"),
]


def help_overlay(width: int) -> list[str]:
    """The help panel as plain text rows: boxed and width-clamped.

    Key bindings and slash commands sit side by side on wide terminals; on
    narrow ones the keys get the full width and a pointer to ``/help`` covers
    the commands. Either way the panel is short enough to fit the body.
    """
    left = ["KEY BINDINGS"] + [f"  {keys:<14} {desc}" for keys, desc in KEY_HELP]
    right = ["COMMANDS"] + [f"  {keys:<16} {desc}" for keys, desc in SLASH_HELP]
    left_width = max(len(row) for row in left)
    right_width = max(len(row) for row in right)
    if left_width + right_width + 7 <= width - 4:
        height = max(len(left), len(right))
        left += [""] * (height - len(left))
        right += [""] * (height - len(right))
        content = [
            first.ljust(left_width) + " | " + second
            for first, second in zip(left, right)
        ]
    else:
        content = left + ["", "  type /help for the command list"]

    box_width = min(max(len(row) for row in content) + 4, width - 4)
    inner_width = box_width - 4
    rows = ["+" + "-" * (box_width - 2) + "+"]
    for text in content:
        rows.append("| " + text[:inner_width].ljust(inner_width) + " |")
    rows.append("+" + "-" * (box_width - 2) + "+")
    return rows


def action_for(key: str, state: TuiState, bindings: dict[str, KeyAction] | None = None) -> KeyAction:
    """Map a raw key to an action, taking focus into account.

    Arrow keys scroll when the scrollback has focus and move the cursor when the
    input does. Same key, different meaning, decided here rather than in the
    driver so it can be tested.
    """
    table = bindings or DEFAULT_BINDINGS
    if key == "?" and table.get(key) is KeyAction.HELP:
        # A "?" typed mid-line is punctuation, not a help request: only an
        # empty input line (or the scrollback panel) opens the help overlay,
        # so asking the model a question still works.
        if state.focus is Panel.INPUT and state.buffer:
            return KeyAction.NONE
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

    if state.help_visible:
        overlay = help_overlay(width)
        top = 1 + max(0, (body_height - len(overlay)) // 2)
        for index, text in enumerate(overlay):
            row = top + index
            if 1 <= row < len(rows) - 1:
                rows[row] = (text[:width], "help")

    indicator = "… working" if state.busy else ""
    status = f" {state.status} {indicator}".strip()
    right = f"? help  [{state.focus.value}] "
    padding = max(1, width - len(status) - len(right))
    rows.insert(0, ((status + " " * padding + right)[:width], "status"))

    return Rendered(
        rows=rows,
        cursor_row=len(rows) - 1,
        cursor_col=min(width - 1, len(state.prompt) + state.cursor),
        status=status,
        input_line=prompt_line,
    )
