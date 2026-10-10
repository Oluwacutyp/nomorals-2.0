"""TUI state and layout — pure, with no curses import.

Everything that can be reasoned about lives here: the scrollback, the input
buffer, cursor motion, which panel is focused, and how a screen of text is
computed from state. The curses driver in ``app.py`` only draws what this module
hands it and translates keypresses into calls here.

That split is the whole reason this file exists. A TUI written directly against
curses cannot be tested without a terminal, so its layout and key handling go
unverified until a human happens to press the right key. Here, they are ordinary
functions returning strings.

Editing model: GNU-readline-style emacs bindings (kill ring, yank-pop, word
motion, transpose, undo) on top of the buffer. Discovery: a fuzzy command
palette (Ctrl-P) and an interactive history search (Ctrl-R). Scrollback search
(Ctrl-S) works like ``less``. Assistant output renders block-level markdown.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

__all__ = [
    "Panel", "Line", "KeyAction", "TuiState", "render",
    "KEY_HELP", "SLASH_HELP", "SLASH_COMMANDS", "help_overlay",
    "PaletteState", "PaletteItem", "SearchState",
    "fuzzy_match", "SPINNER_FRAMES", "SPINNER_FRAMES_ASCII",
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
    ESCAPE = "escape"
    # readline-style editing
    KILL_TO_END = "kill_to_end"
    KILL_TO_START = "kill_to_start"
    KILL_WORD_BACK = "kill_word_back"
    YANK = "yank"
    UNDO = "undo"
    TRANSPOSE = "transpose"
    WORD_LEFT = "word_left"
    WORD_RIGHT = "word_right"
    # discovery overlays
    PALETTE = "palette"
    HIST_SEARCH = "hist_search"
    SEARCH_OPEN = "search_open"
    SEARCH_CLOSE = "search_close"
    SEARCH_NEXT = "search_next"
    SEARCH_PREV = "search_prev"


# Spinner frame sets, after charm's bubbles spinner (Dot + an ASCII-safe Line).
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
SPINNER_FRAMES_ASCII = ("-", "\\", "|", "/")

_UNDO_DEPTH = 100


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
class PaletteItem:
    """One row in the command palette / history picker.

    ``kind`` is ``"slash"`` (insert the text and submit it), ``"insert"``
    (insert the text and leave it for editing), or ``"cmd"`` (an internal
    command the driver executes itself).
    """

    name: str
    hint: str = ""
    kind: str = "slash"
    payload: str = ""


@dataclass
class PaletteState:
    """The fuzzy command palette (or the history picker reusing it)."""

    mode: str = "commands"  # "commands" | "history"
    query: str = ""
    selected: int = 0
    items: list[PaletteItem] = field(default_factory=list)


@dataclass
class SearchState:
    """Incremental scrollback search, less-style. ``matches`` holds line indices."""

    query: str = ""
    matches: list[int] = field(default_factory=list)
    index: int = -1


def fuzzy_match(query: str, target: str) -> tuple[bool, int]:
    """Ordered-subsequence match, case-insensitive.

    Returns (matched, position) where position is the index at which the first
    query character was greedily consumed — the ranking key (lower is better).
    An empty query matches everything at position 0.
    """
    if not query:
        return True, 0
    q = query.lower()
    t = target.lower()
    pos = 0
    first = -1
    for char in q:
        found = t.find(char, pos)
        if found < 0:
            return False, 0
        if first < 0:
            first = found
        pos = found + 1
    return True, first


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
    # sweep additions
    markdown: bool = True
    show_timestamps: bool = False
    palette: PaletteState | None = None
    search: SearchState | None = None
    spin: int = 0
    spin_ascii: bool = False
    busy_since: float | None = None
    history_path: str | None = None
    kill_ring: list[str] = field(default_factory=list)
    undo_stack: list[tuple[str, int]] = field(default_factory=list)

    # internal bookkeeping (not part of the visible state)
    _last_was_kill: bool = field(default=False, repr=False)
    _yank_active: bool = field(default=False, repr=False)
    _last_yank_len: int = field(default=0, repr=False)
    _completion_index: int = field(default=-1, repr=False)
    _completion_stem: str = field(default="", repr=False)

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

    # ── busy / spinner ───────────────────────────────────────────────────────

    def set_busy(self) -> None:
        self.busy = True
        self.busy_since = time.time()

    def set_idle(self) -> None:
        self.busy = False
        self.busy_since = None

    def busy_elapsed(self) -> float:
        if self.busy_since is None:
            return 0.0
        return max(0.0, time.time() - self.busy_since)

    def busy_frame(self) -> str:
        frames = SPINNER_FRAMES_ASCII if self.spin_ascii else SPINNER_FRAMES
        return frames[self.spin % len(frames)]

    def tick(self) -> str:
        """Advance the spinner one frame. The driver calls this while busy."""
        self.spin += 1
        return self.busy_frame()

    # ── input ────────────────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        self.undo_stack.append((self.buffer, self.cursor))
        if len(self.undo_stack) > _UNDO_DEPTH:
            del self.undo_stack[: len(self.undo_stack) - _UNDO_DEPTH]

    def _note_edit(self) -> None:
        """Any buffer mutation that is not yank ends yank-pop and completion."""
        self._yank_active = False
        self._last_was_kill = False
        self._completion_index = -1
        self._completion_stem = ""

    def insert(self, text: str) -> None:
        if not text:
            return
        self._push_undo()
        self.buffer = self.buffer[: self.cursor] + text + self.buffer[self.cursor :]
        self.cursor += len(text)
        self._note_edit()

    def backspace(self) -> None:
        if self.cursor > 0:
            self._push_undo()
            self.buffer = self.buffer[: self.cursor - 1] + self.buffer[self.cursor :]
            self.cursor -= 1
            self._note_edit()

    def delete_char(self) -> None:
        if self.cursor < len(self.buffer):
            self._push_undo()
            self.buffer = self.buffer[: self.cursor] + self.buffer[self.cursor + 1 :]
            self._note_edit()

    def move(self, delta: int) -> None:
        self.cursor = max(0, min(len(self.buffer), self.cursor + delta))

    def home(self) -> None:
        self.cursor = 0

    def end(self) -> None:
        self.cursor = len(self.buffer)

    # ── readline-style editing ───────────────────────────────────────────────

    def _kill(self, text: str) -> None:
        """Record killed text in the ring, appending to consecutive kills."""
        if not text:
            return
        if self._last_was_kill and self.kill_ring:
            self.kill_ring[-1] += text
        else:
            self.kill_ring.append(text)
        self._last_was_kill = True
        self._yank_active = False

    def kill_to_end(self) -> None:
        if self.cursor < len(self.buffer):
            self._push_undo()
            self._kill(self.buffer[self.cursor :])
            self.buffer = self.buffer[: self.cursor]

    def kill_to_start(self) -> None:
        if self.cursor > 0:
            self._push_undo()
            self._kill(self.buffer[: self.cursor])
            self.buffer = self.buffer[self.cursor :]
            self.cursor = 0

    def kill_word_back(self) -> None:
        """Unix word-rubout: kill back to whitespace, like Ctrl-W in shells."""
        if self.cursor == 0:
            return
        end = self.cursor
        start = end
        while start > 0 and self.buffer[start - 1] in " \t":
            start -= 1
        while start > 0 and self.buffer[start - 1] not in " \t":
            start -= 1
        if start < end:
            self._push_undo()
            self._kill(self.buffer[start:end])
            self.buffer = self.buffer[:start] + self.buffer[end:]
            self.cursor = start

    def yank(self) -> None:
        """Paste the kill ring. Repeating it cycles (yank-pop)."""
        if not self.kill_ring:
            return
        self._push_undo()
        if self._yank_active:
            # Replace the previously yanked text with the next ring entry.
            start = self.cursor - self._last_yank_len
            self.kill_ring.append(self.kill_ring.pop(0))
            text = self.kill_ring[0]
            self.buffer = self.buffer[:start] + text + self.buffer[self.cursor :]
            self.cursor = start + len(text)
        else:
            text = self.kill_ring[0]
            self.buffer = self.buffer[: self.cursor] + text + self.buffer[self.cursor :]
            self.cursor += len(text)
        self._last_yank_len = len(text)
        self._yank_active = True
        self._last_was_kill = False

    def undo(self) -> None:
        if self.undo_stack:
            self.buffer, self.cursor = self.undo_stack.pop()
            self._yank_active = False
            self._last_was_kill = False

    def transpose(self) -> None:
        """Swap the char before the cursor with the one under it (Ctrl-T)."""
        if self.cursor <= 0 or len(self.buffer) < 2:
            return
        self._push_undo()
        if self.cursor >= len(self.buffer):
            left, right = self.cursor - 2, self.cursor - 1
        else:
            left, right = self.cursor - 1, self.cursor
        chars = list(self.buffer)
        chars[left], chars[right] = chars[right], chars[left]
        self.buffer = "".join(chars)
        self.cursor = min(len(self.buffer), right + 1)
        self._note_edit()

    def word_left(self) -> None:
        pos = self.cursor
        while pos > 0 and self.buffer[pos - 1] in " \t":
            pos -= 1
        while pos > 0 and self.buffer[pos - 1] not in " \t":
            pos -= 1
        self.cursor = pos

    def word_right(self) -> None:
        pos = self.cursor
        while pos < len(self.buffer) and self.buffer[pos] not in " \t":
            pos += 1
        while pos < len(self.buffer) and self.buffer[pos] in " \t":
            pos += 1
        self.cursor = pos

    def complete_tab(self) -> bool:
        """Cycle through /command completions. Returns True when it completed."""
        if not self.buffer.startswith("/"):
            return False
        if self._completion_index >= 0 and self._completion_stem:
            # Mid-cycle: repeated Tabs reuse the original stem, not the
            # already-completed text (which now ends with a space).
            stem = self._completion_stem
        else:
            stem = self.buffer[1:].rstrip()
        candidates = [cmd for cmd in SLASH_COMMANDS if cmd[1:].startswith(stem)]
        if not candidates:
            return False
        self._push_undo()
        if len(candidates) == 1:
            self._completion_index = -1
            self._completion_stem = ""
            self.buffer = candidates[0] + " "
        else:
            if self._completion_index < 0:
                self._completion_stem = stem
            self._completion_index = (self._completion_index + 1) % len(candidates)
            self.buffer = candidates[self._completion_index] + " "
        self.cursor = len(self.buffer)
        # This path deliberately does not call _note_edit(): resetting the
        # cycle index/stem is exactly what must not happen between Tabs.
        return True

    # ── history ──────────────────────────────────────────────────────────────

    def record_history(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            if len(self.history) > self.max_history:
                del self.history[0]

    def submit(self) -> str:
        """Take the buffer, record it in history, and reset. Returns the text."""
        text = self.buffer.strip()
        self.record_history(text)
        self._push_undo()
        self.buffer = ""
        self.cursor = 0
        self.history_index = -1
        self._note_edit()
        return text

    def history_prev(self) -> None:
        if not self.history:
            return
        if self.history_index < 0:
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        self._push_undo()
        self.buffer = self.history[self.history_index]
        self.cursor = len(self.buffer)
        self._note_edit()

    def history_next(self) -> None:
        if self.history_index < 0:
            return
        self.history_index += 1
        self._push_undo()
        if self.history_index >= len(self.history):
            self.history_index = -1
            self.buffer = ""
        else:
            self.buffer = self.history[self.history_index]
        self.cursor = len(self.buffer)
        self._note_edit()

    def save_history(self, path: str | None = None) -> None:
        target = path or self.history_path
        if not target:
            return
        try:
            with open(target, "w", encoding="utf-8") as handle:
                for entry in self.history[-self.max_history :]:
                    handle.write(entry.replace("\n", " ") + "\n")
        except OSError:
            pass

    def load_history(self, path: str | None = None) -> None:
        target = path or self.history_path
        if not target or not os.path.exists(target):
            return
        try:
            with open(target, encoding="utf-8") as handle:
                entries = [line.rstrip("\n") for line in handle if line.strip()]
        except OSError:
            return
        self.history = entries[-self.max_history :]
        self.history_index = -1

    # ── command palette ──────────────────────────────────────────────────────

    def _palette_commands(self) -> list[PaletteItem]:
        items: list[PaletteItem] = []
        for command in SLASH_COMMANDS:
            hint = next((desc for name, desc in SLASH_HELP if name.split()[0] == command), "")
            if command in ("/mem", "/recall"):
                # These need an argument: insert, don't submit.
                items.append(PaletteItem(command, hint, "insert", command + " "))
            else:
                items.append(PaletteItem(command, hint, "slash", command))
        items.extend(
            [
                PaletteItem("Clear scrollback", "empty the scrollback", "cmd", "clear"),
                PaletteItem("Toggle markdown rendering", "pretty assistant output on/off", "cmd", "toggle-markdown"),
                PaletteItem("Toggle timestamps", "show message times on/off", "cmd", "toggle-timestamps"),
                PaletteItem("Scroll to top", "jump to the oldest output", "cmd", "scroll-top"),
                PaletteItem("Scroll to bottom", "jump to the newest output", "cmd", "scroll-bottom"),
                PaletteItem("Quit the TUI", "exit", "cmd", "quit"),
            ]
        )
        return items

    def open_palette(self) -> None:
        self.palette = PaletteState(mode="commands", items=self._palette_commands())
        self.search = None

    def open_history_search(self) -> None:
        seen: set[str] = set()
        items: list[PaletteItem] = []
        for entry in reversed(self.history):
            if entry not in seen:
                seen.add(entry)
                items.append(PaletteItem(entry, "from history", "insert", entry))
        self.palette = PaletteState(mode="history", items=items)
        self.search = None

    def close_palette(self) -> None:
        self.palette = None

    def palette_matches(self) -> list[PaletteItem]:
        pal = self.palette
        if pal is None:
            return []
        scored: list[tuple[int, str, PaletteItem]] = []
        for item in pal.items:
            matched, position = fuzzy_match(pal.query, item.name)
            if matched:
                scored.append((position, item.name.lower(), item))
        scored.sort(key=lambda triple: (triple[0], triple[1]))
        return [item for _, _, item in scored]

    def palette_move(self, delta: int) -> None:
        pal = self.palette
        if pal is None:
            return
        count = len(self.palette_matches())
        if count:
            pal.selected = (pal.selected + delta) % count

    def palette_type(self, text: str) -> None:
        pal = self.palette
        if pal is None:
            return
        pal.query += text
        pal.selected = 0

    def palette_backspace(self) -> None:
        pal = self.palette
        if pal is not None and pal.query:
            pal.query = pal.query[:-1]
            pal.selected = 0

    def palette_select(self) -> PaletteItem | None:
        pal = self.palette
        if pal is None:
            return None
        matches = self.palette_matches()
        item = matches[pal.selected] if 0 <= pal.selected < len(matches) else None
        self.palette = None
        return item

    # ── scrollback search ────────────────────────────────────────────────────

    def open_search(self) -> None:
        self.search = SearchState()
        self.palette = None
        self.focus = Panel.SCROLLBACK

    def close_search(self) -> None:
        self.search = None

    def search_recalc(self) -> None:
        state = self.search
        if state is None:
            return
        query = state.query.lower()
        if not query:
            state.matches = []
            state.index = -1
            return
        state.matches = [
            index for index, line in enumerate(self.lines) if query in line.text.lower()
        ]
        state.index = 0 if state.matches else -1
        if state.index >= 0:
            self.search_goto()

    def search_type(self, text: str) -> None:
        if self.search is not None:
            self.search.query += text
            self.search_recalc()

    def search_backspace(self) -> None:
        if self.search is not None and self.search.query:
            self.search.query = self.search.query[:-1]
            self.search_recalc()

    def search_step(self, delta: int) -> None:
        state = self.search
        if state is None or not state.matches:
            return
        state.index = (state.index + delta) % len(state.matches)
        self.search_goto()

    def search_goto(self, *, viewport: int = 20, width: int = 80) -> None:
        """Scroll so the current match sits at the bottom of the viewport."""
        state = self.search
        if state is None or not state.matches:
            return
        target = state.matches[state.index % len(state.matches)]
        rows_before = sum(len(line.render(width)) for line in self.lines[:target])
        total = self._content_height(width)
        maximum = max(0, total - viewport)
        self.scroll = max(0, min(maximum, total - rows_before - viewport))

    # ── scrolling ────────────────────────────────────────────────────────────

    def scroll_by(self, delta: int, *, viewport: int = 20) -> None:
        """``scroll`` is rows up from the bottom, so it never goes negative."""
        maximum = max(0, self._content_height(80) - viewport)
        self.scroll = max(0, min(maximum, self.scroll + delta))

    def scroll_top(self, *, viewport: int = 20) -> None:
        self.scroll = max(0, self._content_height(80) - viewport)

    def scroll_bottom(self) -> None:
        self.scroll = 0

    def scroll_percent(self, *, viewport: int = 20, width: int = 80) -> int:
        maximum = max(0, self._content_height(width) - viewport)
        if maximum == 0:
            return 0
        return round(100 * self.scroll / maximum)

    def _content_height(self, width: int) -> int:
        return sum(len(line.render(width)) for line in self.lines)

    # ── focus ────────────────────────────────────────────────────────────────

    def cycle_focus(self) -> None:
        order = [Panel.INPUT, Panel.SCROLLBACK]
        index = order.index(self.focus) if self.focus in order else 0
        self.focus = order[(index + 1) % len(order)]

    def toggle_help(self) -> None:
        self.help_visible = not self.help_visible

    def toggle_markdown(self) -> None:
        self.markdown = not self.markdown

    def toggle_timestamps(self) -> None:
        self.show_timestamps = not self.show_timestamps


DEFAULT_BINDINGS: dict[str, KeyAction] = {
    "\n": KeyAction.SUBMIT,
    "\r": KeyAction.SUBMIT,
    "\x03": KeyAction.CANCEL,
    "\x04": KeyAction.QUIT,
    "\x0c": KeyAction.CLEAR,
    "\t": KeyAction.FOCUS_NEXT,
    "\x1b": KeyAction.ESCAPE,
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
    # readline-style editing
    "\x02": KeyAction.WORD_LEFT,
    "\x06": KeyAction.WORD_RIGHT,
    "\x0b": KeyAction.KILL_TO_END,
    "\x14": KeyAction.TRANSPOSE,
    "\x15": KeyAction.KILL_TO_START,
    "\x17": KeyAction.KILL_WORD_BACK,
    "\x19": KeyAction.YANK,
    "\x1a": KeyAction.UNDO,
    # discovery
    "\x10": KeyAction.PALETTE,
    "\x12": KeyAction.HIST_SEARCH,
    "\x13": KeyAction.SEARCH_OPEN,
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
    ("Ctrl-P", "command palette"),
    ("Ctrl-R", "search history"),
    ("Ctrl-S", "search the scrollback"),
    ("Tab", "complete /command · switch panel focus"),
    ("Up / Down", "history (input) · scroll (scrollback)"),
    ("Left / Right", "move the cursor"),
    ("Ctrl-A / Ctrl-E", "line start / end"),
    ("Ctrl-B / Ctrl-F", "word left / right"),
    ("Ctrl-K / Ctrl-U", "kill to end / start of line"),
    ("Ctrl-W", "kill word back"),
    ("Ctrl-Y", "yank (repeat to cycle the kill ring)"),
    ("Ctrl-Z", "undo"),
    ("Ctrl-T", "transpose characters"),
    ("Del", "delete char under cursor"),
    ("PgUp / PgDn", "page the scrollback"),
    ("j / k", "scroll one line (scrollback)"),
    ("u / d", "scroll one page (scrollback)"),
    ("g / G", "top / bottom (scrollback)"),
    ("Esc", "close palette / search / help"),
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

SLASH_COMMANDS: list[str] = [name.split()[0] for name, _ in SLASH_HELP]


def _box(rows: list[str], width: int) -> list[str]:
    """Draw an ASCII box around rows, clamped to the terminal width."""
    box_width = min(max(len(row) for row in rows) + 4, width - 4) if rows else 8
    inner_width = box_width - 4
    boxed = ["+" + "-" * (box_width - 2) + "+"]
    for text in rows:
        boxed.append("| " + text[:inner_width].ljust(inner_width) + " |")
    boxed.append("+" + "-" * (box_width - 2) + "+")
    return boxed


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
    return _box(content, width)


def palette_overlay(state: TuiState, width: int, max_items: int = 8) -> list[tuple[str, str]]:
    """The command palette / history picker as boxed (text, kind) rows."""
    pal = state.palette
    if pal is None:
        return []
    title = "COMMAND PALETTE" if pal.mode == "commands" else "HISTORY SEARCH"
    matches = state.palette_matches()
    content: list[tuple[str, str]] = [(title, "palette"), ("> " + pal.query, "palette")]
    for index, item in enumerate(matches[:max_items]):
        marker = "> " if index == pal.selected else "  "
        row = f"{marker}{item.name}"
        if item.hint:
            row += f"  — {item.hint}"
        content.append((row, "palette_sel" if index == pal.selected else "palette"))
    if not matches:
        content.append((f'  No results for "{pal.query}"', "palette"))
    boxed = _box([text for text, _ in content], width)
    kinds = [kind for _, kind in content]
    # The border rows reuse the title kind; pad the kind list to the box size.
    kinds = ["palette"] + kinds + ["palette"]
    return list(zip(boxed, kinds))


def _markdown_rows(text: str, width: int) -> list[tuple[str, str]]:
    """Block-level markdown for assistant output: headings, fences, quotes,
    lists, rules. Inline markup is left alone — never reinterpreted."""
    rows: list[tuple[str, str]] = []
    in_fence = False
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            for wrapped in Line("  " + raw).render(width):
                rows.append((wrapped, "code"))
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped[level:].strip()
            for wrapped in Line(heading).render(width):
                rows.append((wrapped, "md_head"))
        elif stripped.startswith(">"):
            quote = stripped[1:].lstrip()
            for wrapped in Line("| " + quote).render(width):
                rows.append((wrapped, "md_quote"))
        elif stripped in ("---", "***", "___") and len(stripped) >= 3:
            rows.append(("─" * min(width, 40), "md_rule"))
        elif stripped.startswith(("- ", "* ")):
            for wrapped in Line("  • " + stripped[2:]).render(width):
                rows.append((wrapped, "info"))
        else:
            for wrapped in Line(raw).render(width):
                rows.append((wrapped, "info"))
    return rows or [("", "info")]


def action_for(key: str, state: TuiState, bindings: dict[str, KeyAction] | None = None) -> KeyAction:
    """Map a raw key to an action, taking focus into account.

    Arrow keys scroll when the scrollback has focus and move the cursor when the
    input does. Same key, different meaning, decided here rather than in the
    driver so it can be tested.
    """
    table = bindings or DEFAULT_BINDINGS
    if key == "?":
        if table.get(key) is KeyAction.HELP:
            # A "?" typed mid-line is punctuation, not a help request: only an
            # empty input line (or the scrollback panel) opens the help overlay,
            # so asking the model a question still works.
            if state.focus is Panel.INPUT and state.buffer:
                return KeyAction.NONE
            return KeyAction.HELP
    if key == "\t" and state.focus is Panel.INPUT and state.buffer.startswith("/"):
        # Tab completes /commands; everywhere else it switches panel focus.
        return KeyAction.TAB_COMPLETE
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


def _render_line(line: Line, width: int, state: TuiState) -> list[tuple[str, str]]:
    """One Line -> wrapped (text, kind) rows, with markdown/timestamps applied."""
    if state.markdown and line.kind == "assistant":
        rows = _markdown_rows(line.text, width)
    else:
        rows = [(row, line.kind) for row in line.render(width)]
    if state.show_timestamps:
        stamp = time.strftime("[%H:%M] ", time.localtime(line.at))
        rows = [
            (stamp + text if index == 0 else " " * len(stamp) + text, kind)
            for index, (text, kind) in enumerate(rows)
        ]
    return rows


def render(state: TuiState, *, width: int, height: int) -> Rendered:
    """Compute the screen. Pure: same state and size always gives same output.

    Layout: a status bar on top, the input line at the bottom, and the
    scrollback filling what is left, anchored to the bottom unless scrolled.
    """
    if width < 20 or height < 5:
        return Rendered(rows=[("terminal too small", "error")], status="resize the window")

    search_open = state.search is not None
    body_height = height - 2 - (1 if search_open else 0)
    prompt_line = state.prompt + state.buffer

    wrapped: list[tuple[str, str, int]] = []
    for index, line in enumerate(state.lines):
        for text, kind in _render_line(line, width, state):
            wrapped.append((text, kind, index))

    current_match: int | None = None
    if state.search is not None and state.search.matches:
        current_match = state.search.matches[state.search.index % len(state.search.matches)]
        # Pin the current match into the viewport. The model cannot know the
        # real viewport when the search runs, so render enforces visibility
        # here — deterministic for a given state, like everything else.
        rows_before = sum(len(line.render(width)) for line in state.lines[:current_match])
        maximum = max(0, len(wrapped) - body_height)
        state.scroll = max(0, min(maximum, len(wrapped) - rows_before - body_height))

    # Anchor to the bottom, then honour the scroll offset.
    end = len(wrapped) - state.scroll
    start = max(0, end - body_height)
    visible = wrapped[start:end]
    rows: list[tuple[str, str]] = []
    while len(visible) < body_height:
        visible.insert(0, ("", "blank", -1))
    for text, kind, line_index in visible:
        if current_match is not None and line_index == current_match:
            kind = "search_match"
        rows.append((text[:width], kind))

    overlay = state.palette is not None
    if overlay:
        boxed = palette_overlay(state, width, max_items=max(1, body_height - 5))
        top = 1 + max(0, (body_height - len(boxed)) // 2)
        for index, (text, kind) in enumerate(boxed):
            row = top + index
            if 1 <= row < len(rows) - 1:
                rows[row] = (text[:width], kind)
    elif state.help_visible:
        boxed = [(text, "help") for text in help_overlay(width)]
        top = 1 + max(0, (body_height - len(boxed)) // 2)
        for index, (text, kind) in enumerate(boxed):
            row = top + index
            if 1 <= row < len(rows) - 1:
                rows[row] = (text[:width], kind)

    if search_open:
        query = state.search.query if state.search else ""
        matches = state.search.matches if state.search else []
        if not query:
            bar = "/ type to search the scrollback (Enter: next, Up: prev, Esc: close)"
        elif matches:
            bar = f"/{query}  match {state.search.index + 1}/{len(matches)}"
        else:
            bar = f"/{query}  no matches"
        rows.append((bar[:width], "search"))

    rows.append((prompt_line[:width], "input"))

    # Status bar: left segments (status, busy spinner + elapsed), right
    # segments (focus, scroll position, hints) — powerline-style.
    left = state.status
    if state.busy:
        left += f"  {state.busy_frame()} …working {state.busy_elapsed():.1f}s"
    right_segments = [f"[{state.focus.value}]"]
    if state.scroll > 0:
        right_segments.append(f"scroll {state.scroll_percent(viewport=body_height, width=width)}%")
    right_segments.append("? help")
    right = "  ".join(right_segments)
    status = f" {left}".strip()
    padding = max(1, width - len(status) - len(right))
    rows.insert(0, ((status + " " * padding + right)[:width], "status"))

    if search_open:
        cursor_row = len(rows) - 2
        cursor_col = min(width - 1, 1 + len(state.search.query))
    else:
        cursor_row = len(rows) - 1
        cursor_col = min(width - 1, len(state.prompt) + state.cursor)

    if state.palette is not None:
        # Cursor sits at the end of the palette's query line: box top border
        # (row 0) + title + query row, offset by the overlay's top.
        boxed_len = len(palette_overlay(state, width, max_items=max(1, body_height - 5)))
        top = 1 + max(0, (body_height - boxed_len) // 2)
        cursor_row = 1 + top + 2
        cursor_col = min(width - 1, 4 + len(state.palette.query))

    return Rendered(
        rows=rows,
        cursor_row=cursor_row,
        cursor_col=cursor_col,
        status=status,
        input_line=prompt_line,
    )
