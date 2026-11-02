"""Tail-95 pins: nomorals/tui/app.py census sites. All exclusions.

251 — _run_submit: ``except KeyboardInterrupt: self.stop()``.
/quit is DELIVERED as KeyboardInterrupt by design (the file's
own noqa comment says so); stopping the loop is the handler's
contract, not a swallowed failure. The sibling branch proves
the distinction: a real exception from on_submit is surfaced
via state.error, pinned as the control.

274 / 289 / 293 — draw(): ``except curses.error: pass`` on
curs_set, addnstr and move. Canonical curses idiom, each with
its reason in the file's own comments: some terminals reject
cursor-visibility changes, writing the bottom-right cell
always raises, and an off-screen cursor must not stop the
frame painting. Nothing is recorded or decided on these
values; refresh() still runs, which is the observable
contract each pin asserts.

327 — run(): ``except KeyboardInterrupt: break`` on the input
poll — the deliberate exit path of the loop (finally still
saves history, pinned).

NOT-recording is pinned with the snapshot method: app.py
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import curses as real_curses
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.tui.app as app_mod
from nomorals.tui.app import TuiApp
from nomorals.tui.model import TuiState


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _curses_error(*args, **kwargs):
    raise real_curses.error("terminal refused")


def _fake_curses(**over):
    base = dict(
        curs_set=lambda *a: None,
        has_colors=lambda: False,
        color_pair=lambda n: 0,
        start_color=lambda: None,
        use_default_colors=lambda: None,
        init_pair=lambda *a: None,
        error=real_curses.error,
        A_NORMAL=0, A_REVERSE=0, A_BOLD=0,
        LINES=24, COLORS=256,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Screen:
    def __init__(self, *, fail_addnstr=False, fail_move=False):
        self.refreshed = 0
        self.addnstr_calls = 0
        self._fail_addnstr = fail_addnstr
        self._fail_move = fail_move

    def getmaxyx(self):
        return (10, 40)

    def erase(self):
        pass

    def addnstr(self, *args):
        self.addnstr_calls += 1
        if self._fail_addnstr:
            raise real_curses.error("bottom-right cell")

    def move(self, *args):
        if self._fail_move:
            raise real_curses.error("cursor off-screen")

    def refresh(self):
        self.refreshed += 1

    def keypad(self, *args):
        pass

    def timeout(self, *args):
        pass


def _app_with_screen(screen):
    app = TuiApp(state=TuiState())
    app.state.say("hello frame")
    app._screen = screen
    return app


# ── 251: KeyboardInterrupt is /quit -> stop(), not an error ─────

def test_251_run_submit_keyboardinterrupt_stops_quietly():
    before = _snap()

    def _quit(text):
        raise KeyboardInterrupt

    app = TuiApp(state=TuiState(), on_submit=_quit)
    app._running = True
    app._run_submit("anything")
    assert app._running is False
    assert not any(line.kind == "error" for line in app.state.lines)
    assert app.state.busy is False

    # control: a real handler failure IS surfaced, loop keeps running
    def _bad(text):
        raise ValueError("handler boom")

    app2 = TuiApp(state=TuiState(), on_submit=_bad)
    app2._running = True
    app2._run_submit("anything")
    assert app2._running is True
    assert any(line.kind == "error" for line in app2.state.lines)
    assert _snap() == before


# ── 274: curs_set refusal must not break the frame ──────────────

def test_274_draw_survives_curs_set_refusal():
    before = _snap()
    screen = _Screen()
    app = _app_with_screen(screen)
    with mock.patch.object(app_mod, "curses",
                           _fake_curses(curs_set=_curses_error)):
        app.draw()
    assert screen.refreshed == 1
    assert _snap() == before


# ── 289: the bottom-right cell raise must not break the frame ───

def test_289_draw_survives_addnstr_bottom_right():
    before = _snap()
    screen = _Screen(fail_addnstr=True)
    app = _app_with_screen(screen)
    with mock.patch.object(app_mod, "curses", _fake_curses()):
        app.draw()
    assert screen.addnstr_calls > 0
    assert screen.refreshed == 1
    assert _snap() == before


# ── 293: an off-screen cursor must not break the frame ──────────

def test_293_draw_survives_move_off_screen():
    before = _snap()
    screen = _Screen(fail_move=True)
    app = _app_with_screen(screen)
    with mock.patch.object(app_mod, "curses", _fake_curses()):
        app.draw()
    assert screen.refreshed == 1
    assert _snap() == before


# ── 327: KeyboardInterrupt on the poll breaks the run loop ──────

def test_327_run_breaks_on_keyboardinterrupt_and_saves_history():
    before = _snap()
    saved = []

    class _QuitScreen(_Screen):
        def get_wch(self):
            raise KeyboardInterrupt

    app = TuiApp(state=TuiState())
    with mock.patch.object(app.state, "save_history",
                           lambda: saved.append(True)), \
         mock.patch.object(app_mod, "curses", _fake_curses()):
        app.run(_QuitScreen())
    # the loop was entered (banner line) and exited via the
    # KeyboardInterrupt break, with history saved by the finally
    assert any("NoMorals Core" in line.text for line in app.state.lines)
    assert saved == [True]
    assert _snap() == before
