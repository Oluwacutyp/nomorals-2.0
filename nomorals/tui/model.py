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
    
    # Tab completion
    TAB_COMPLETE = "tab_complete"
    
    # Multi-line input
    TOGGLE_MULTILINE = "toggle_multiline"
    SUBMIT_MULTILINE = "submit_multiline"
    CANCEL_MULTILINE = "cancel_multiline"
    
    # Clipboard
    COPY_LINE = "copy_line"
    PASTE = "paste"
    
    # Command palette
    OPEN_PALETTE = "open_palette"
    CLOSE_PALETTE = "close_palette"
    PALETTE_UP = "palette_up"
    PALETTE_DOWN = "palette_down"
    PALETTE_SELECT = "palette_select"


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
    
    # Tab completion
    completions: list[str] = field(default_factory=list)
    completion_index: int = -1
    completion_prefix: str = ""
    
    # Multi-line input
    multiline: bool = False
    multiline_buffer: list[str] = field(default_factory=list)
    
    # Clipboard
    clipboard: str = ""
    
    # Command palette
    show_palette: bool = False
    palette_items: list[tuple[str, str, Callable[[], None]]] = field(default_factory=list)
    palette_filter: str = ""
    palette_index: int = 0

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
    
    # ── tab completion ───────────────────────────────────────────────────────
    
    def tab_complete(self, available_commands: list[str] | None = None) -> None:
        """Cycle through completions for current word."""
        if not available_commands:
            return
        
        # Extract current word
        before_cursor = self.buffer[:self.cursor]
        words = before_cursor.split()
        current_word = words[-1] if words else ""
        
        if not current_word:
            return
        
        # First tab: find matches
        if self.completion_index < 0:
            self.completion_prefix = current_word
            self.completions = [cmd for cmd in available_commands if cmd.startswith(current_word)]
            if not self.completions:
                return
            self.completion_index = 0
        else:
            # Cycle to next completion
            self.completion_index = (self.completion_index + 1) % len(self.completions)
        
        # Replace current word with completion
        completion = self.completions[self.completion_index]
        word_start = self.cursor - len(current_word)
        self.buffer = self.buffer[:word_start] + completion + self.buffer[self.cursor:]
        self.cursor = word_start + len(completion)
    
    def reset_completion(self) -> None:
        """Reset completion state."""
        self.completions = []
        self.completion_index = -1
        self.completion_prefix = ""
    
    # ── multi-line input ─────────────────────────────────────────────────────
    
    def toggle_multiline(self) -> None:
        """Toggle multi-line input mode."""
        self.multiline = not self.multiline
        if self.multiline:
            self.status = "multi-line mode (Ctrl-J to submit, Ctrl-G to cancel)"
        else:
            self.multiline_buffer.clear()
            self.status = "ready"
    
    def submit_multiline(self) -> str:
        """Submit multi-line input."""
        text = "\n".join(self.multiline_buffer).strip()
        self.multiline_buffer.clear()
        self.multiline = False
        self.status = "ready"
        return text
    
    def add_multiline(self, line: str) -> None:
        """Add a line to multi-line buffer."""
        self.multiline_buffer.append(line)
        self.buffer = ""
        self.cursor = 0
    
    # ── clipboard ────────────────────────────────────────────────────────────
    
    def copy_line(self) -> None:
        """Copy current line to clipboard."""
        if self.lines:
            # Copy the last non-empty line
            for line in reversed(self.lines):
                if line.text.strip():
                    self.clipboard = line.text
                    self.status = f"copied: {line.text[:50]}..."
                    return
    
    def paste(self) -> None:
        """Paste from clipboard."""
        if self.clipboard:
            self.insert(self.clipboard)
    
    # ── command palette ──────────────────────────────────────────────────────
    
    def open_palette(self, items: list[tuple[str, str, Callable[[], None]]]) -> None:
        """Open command palette.
        
        Args:
            items: List of (name, description, callback) tuples
        """
        self.show_palette = True
        self.palette_items = items
        self.palette_filter = ""
        self.palette_index = 0
    
    def close_palette(self) -> None:
        """Close command palette."""
        self.show_palette = False
        self.palette_items = []
        self.palette_filter = ""
        self.palette_index = 0
    
    def filter_palette(self, query: str) -> list[tuple[str, str, Callable[[], None]]]:
        """Filter palette items by query."""
        self.palette_filter = query
        if not query:
            return self.palette_items
        
        query_lower = query.lower()
        filtered = [
            item for item in self.palette_items
            if query_lower in item[0].lower() or query_lower in item[1].lower()
        ]
        return filtered
    
    def select_palette_item(self) -> Callable[[], None] | None:
        """Execute selected palette item."""
        filtered = self.filter_palette(self.palette_filter)
        if filtered and 0 <= self.palette_index < len(filtered):
            _, _, callback = filtered[self.palette_index]
            self.close_palette()
            return callback
        return None
    
    def palette_up(self) -> None:
        """Move palette selection up."""
        if self.palette_index > 0:
            self.palette_index -= 1
    
    def palette_down(self) -> None:
        """Move palette selection down."""
        filtered = self.filter_palette(self.palette_filter)
        if self.palette_index < len(filtered) - 1:
            self.palette_index += 1


DEFAULT_BINDINGS: dict[str, KeyAction] = {
    "\n": KeyAction.SUBMIT,
    "\r": KeyAction.SUBMIT,
    "\x03": KeyAction.CANCEL,
    "\x04": KeyAction.QUIT,
    "\x0c": KeyAction.CLEAR,
    "\t": KeyAction.TAB_COMPLETE,
    "\x1b[A": KeyAction.HISTORY_PREV,
    "\x1b[B": KeyAction.HISTORY_NEXT,
    "\x1b[D": KeyAction.CURSOR_LEFT,
    "\x1b[C": KeyAction.CURSOR_RIGHT,
    "\x01": KeyAction.HOME,
    "\x05": KeyAction.END,
    "\x08": KeyAction.BACKSPACE,
    "\x7f": KeyAction.BACKSPACE,
    "\x1b[3~": KeyAction.DELETE_CHAR,
    
    # Multi-line
    "\x0a": KeyAction.SUBMIT_MULTILINE,  # Ctrl-J
    "\x07": KeyAction.CANCEL_MULTILINE,  # Ctrl-G
    "\x0d": KeyAction.TOGGLE_MULTILINE,  # Ctrl-M
    
    # Clipboard
    "\x19": KeyAction.COPY_LINE,  # Ctrl-Y
    "\x16": KeyAction.PASTE,      # Ctrl-V
    
    # Command palette
    "\x10": KeyAction.OPEN_PALETTE,  # Ctrl-P
    "\x1b": KeyAction.CLOSE_PALETTE,  # Escape
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
    
    # Command palette overlay
    if state.show_palette:
        return _render_palette(state, width=width, height=height)

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
    
    # Multi-line indicator
    if state.multiline:
        rows.append(("─── multi-line mode (Ctrl-J submit, Ctrl-G cancel) ───", "status"))
        for i, line in enumerate(state.multiline_buffer[-3:], 1):
            rows.append((f"  {line[:width-4]}", "info"))
    
    rows.append((prompt_line[:width], "input"))

    indicator = "…" if state.busy else ""
    mode = "[multi]" if state.multiline else ""
    status = f" {state.status} {mode} {indicator}".strip()
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


def _render_palette(state: TuiState, *, width: int, height: int) -> Rendered:
    """Render command palette overlay."""
    rows: list[tuple[str, str]] = []
    
    # Filter items
    filtered = state.filter_palette(state.palette_filter)
    
    # Header
    header = f" Command Palette ({len(filtered)} items) "
    rows.append((header.center(width)[:width], "status"))
    
    # Search box
    search = f" > {state.palette_filter}"
    rows.append((search[:width], "input"))
    rows.append(("─" * width, "status"))
    
    # Items (show up to height-5 items)
    max_items = height - 5
    start = max(0, state.palette_index - max_items + 1)
    visible_items = filtered[start:start + max_items]
    
    for i, (name, desc, _) in enumerate(visible_items):
        actual_index = start + i
        if actual_index == state.palette_index:
            prefix = "▸ "
            kind = "status"
        else:
            prefix = "  "
            kind = "info"
        
        line = f"{prefix}{name:20s} {desc}"
        rows.append((line[:width], kind))
    
    # Pad to fill screen
    while len(rows) < height - 1:
        rows.append(("", "blank"))
    
    # Footer
    footer = " ↑↓ navigate  ⏎ select  esc close "
    rows.append((footer.center(width)[:width], "status"))
    
    # Cursor in search box
    cursor_col = 3 + len(state.palette_filter)
    
    return Rendered(
        rows=rows,
        cursor_row=1,
        cursor_col=min(width - 1, cursor_col),
        status="palette",
        input_line=state.palette_filter,
    )
