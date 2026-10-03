"""The curses driver. Deliberately thin.

Every decision — what to show, where the cursor goes, what a key means — is made
in ``model.py``. This file reads keys, calls the model, and blits the result.
There is almost nothing here to get wrong, and what is here cannot be unit
tested, so keeping it small is the point.
"""

from __future__ import annotations

import curses
import threading
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .model import DEFAULT_BINDINGS, KeyAction, Panel, TuiState, action_for, render

__all__ = ["TuiApp", "run"]

_log = get_logger(__name__)

_KINDS = {"status": 1, "input": 2, "error": 3, "tool": 4, "assistant": 5, "user": 6, "help": 7}


class TuiApp:
    """Owns the curses window and the :class:`TuiState`."""

    def __init__(
        self,
        context: Any = None,
        *,
        state: TuiState | None = None,
        on_submit: Callable[[str], None] | None = None,
        bindings: dict[str, KeyAction] | None = None,
    ) -> None:
        self.context = context
        self.state = state or TuiState()
        self.on_submit = on_submit or (lambda text: None)
        self.bindings = bindings or DEFAULT_BINDINGS
        self._screen: Any = None
        self._running = False

    # ── key handling ─────────────────────────────────────────────────────────

    def handle(self, key: str) -> bool:
        """Apply one key. Returns False when the app should exit."""
        action = action_for(key, self.state, self.bindings)
        state = self.state

        if action is KeyAction.QUIT:
            return False
        if state.help_visible and action is not KeyAction.HELP:
            # Any key dismisses the help overlay; the key itself is swallowed
            # so a stray keypress cannot type into the buffer underneath.
            state.help_visible = False
            return True
        if action is KeyAction.HELP:
            state.toggle_help()
            return True
        if action is KeyAction.SUBMIT:
            text = state.submit()
            if text:
                state.say(f"{state.prompt}{text}", kind="user")
                state.busy = True
                if self._screen is None:
                    # Embedded/test mode: run inline so the caller sees the
                    # result before handle() returns.
                    self._run_submit(text)
                else:
                    # Live mode: run in the background so the loop keeps
                    # redrawing and the busy indicator is actually visible.
                    worker = threading.Thread(
                        target=self._run_submit, args=(text,), daemon=True
                    )
                    worker.start()
        elif action is KeyAction.CANCEL:
            state.busy = False
            state.status = "cancelled"
        elif action is KeyAction.CLEAR:
            state.clear()
        elif action is KeyAction.FOCUS_NEXT:
            state.cycle_focus()
        elif action is KeyAction.HISTORY_PREV:
            state.history_prev()
        elif action is KeyAction.HISTORY_NEXT:
            state.history_next()
        elif action is KeyAction.CURSOR_LEFT:
            state.move(-1)
        elif action is KeyAction.CURSOR_RIGHT:
            state.move(1)
        elif action is KeyAction.HOME:
            state.home()
        elif action is KeyAction.END:
            state.end()
        elif action is KeyAction.BACKSPACE:
            state.backspace()
        elif action is KeyAction.DELETE_CHAR:
            state.delete_char()
        elif action is KeyAction.SCROLL_UP:
            state.scroll_by(1, viewport=self._viewport_height())
        elif action is KeyAction.SCROLL_DOWN:
            state.scroll_by(-1, viewport=self._viewport_height())
        elif action is KeyAction.PAGE_UP:
            state.scroll_by(self._viewport_height(), viewport=self._viewport_height())
        elif action is KeyAction.PAGE_DOWN:
            state.scroll_by(-self._viewport_height(), viewport=self._viewport_height())
        elif action is KeyAction.SCROLL_TOP:
            state.scroll_top(viewport=self._viewport_height())
        elif action is KeyAction.SCROLL_BOTTOM:
            state.scroll_bottom()
        elif len(key) == 1 and key.isprintable() and self.state.focus is Panel.INPUT:
            state.insert(key)
        return True

    def _run_submit(self, text: str) -> None:
        """Run the submit handler, reporting failures without killing the UI."""
        try:
            self.on_submit(text)
        except KeyboardInterrupt:  # noqa: E106 - /quit arrives as KeyboardInterrupt; stop the loop
            self.stop()
        except Exception as exc:  # noqa: BLE001 - one bad command must not kill the UI
            self.state.error(f"{type(exc).__name__}: {exc}")
        finally:
            self.state.busy = False

    def _viewport_height(self) -> int:
        if self._screen is None:
            return 20
        return max(3, curses.LINES - 2)

    # ── drawing ──────────────────────────────────────────────────────────────

    def draw(self) -> None:
        if self._screen is None:
            return
        height, width = self._screen.getmaxyx()
        frame = render(self.state, width=width, height=height)
        try:
            curses.curs_set(0 if self.state.help_visible else 1)
        except curses.error:  # noqa: E103 - some terminals reject cursor visibility changes
            pass
        self._screen.erase()
        for index, (text, kind) in enumerate(frame.rows[:height]):
            attribute = curses.A_NORMAL
            if curses.has_colors():
                attribute = curses.color_pair(_KINDS.get(kind, 0))
            if kind == "status":
                attribute |= curses.A_REVERSE
            if kind == "help":
                attribute |= curses.A_BOLD
            try:
                self._screen.addnstr(index, 0, text, width - 1, attribute)
            except curses.error:  # noqa: E103 - writing the bottom-right cell always raises; ignore it
                pass
        try:
            self._screen.move(frame.cursor_row, frame.cursor_col)
        except curses.error:  # noqa: E103 - cursor off-screen; refresh still paints the frame
            pass
        self._screen.refresh()

    # ── loop ─────────────────────────────────────────────────────────────────

    def run(self, screen: Any) -> None:
        self._screen = screen
        self._running = True
        curses.curs_set(1)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            for kind, index in _KINDS.items():
                if index and index < curses.COLORS:
                    curses.init_pair(index, _color_for(kind), -1)
        screen.keypad(True)
        # Poll instead of blocking forever: the loop redraws ~8x a second so
        # the busy indicator animates while a submit handler runs in a worker.
        screen.timeout(120)
        self.state.say(
            "NoMorals Core — type a message and press Enter, ? for keys, "
            "/help for commands, Ctrl-D to quit, Tab to switch panels."
        )
        while self._running:
            self.draw()
            try:
                code = screen.get_wch()
            except curses.error:
                continue
            except KeyboardInterrupt:  # noqa: E106 - deliberate: break the input loop
                break
            key = code if isinstance(code, str) else _key_name(code)
            if not self.handle(key):
                break

    def stop(self) -> None:
        self._running = False


def _color_for(kind: str) -> int:
    return {
        "status": curses.COLOR_WHITE, "input": curses.COLOR_GREEN,
        "error": curses.COLOR_RED, "tool": curses.COLOR_YELLOW,
        "assistant": curses.COLOR_CYAN, "user": curses.COLOR_MAGENTA,
        "help": curses.COLOR_WHITE,
    }.get(kind, curses.COLOR_WHITE)


_KEY_NAMES = {
    curses.KEY_UP: "\x1b[A", curses.KEY_DOWN: "\x1b[B", curses.KEY_LEFT: "\x1b[D",
    curses.KEY_RIGHT: "\x1b[C", curses.KEY_HOME: "\x01", curses.KEY_END: "\x05",
    curses.KEY_BACKSPACE: "\x08", curses.KEY_DC: "\x1b[3~",
    curses.KEY_PPAGE: "u", curses.KEY_NPAGE: "d",
    curses.KEY_F1: "?",
}


def _key_name(code: int) -> str:
    """Translate a curses key code into the escape sequence the model expects."""
    return _KEY_NAMES.get(code, "")


def run(context: Any = None, *, on_submit: Callable[[str], None] | None = None) -> int:
    """Start the TUI. Returns 0 on clean exit."""
    app = TuiApp(context, on_submit=on_submit)
    try:
        curses.wrapper(app.run)
    except curses.error as exc:  # pragma: no cover - not a terminal
        print(f"could not start the TUI: {exc}", flush=True)
        return 1
    return 0
