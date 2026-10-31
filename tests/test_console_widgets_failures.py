"""Tail-71 pins: nomorals/console/widgets.py silent-swallow sites.

Adjudications (see ~/workspace/nm-tail71-verify/SURVEY.md) — all six
sites are E106 KeyboardInterrupt handlers in the interactive watch
UI, and all six are EXCLUSIONS: Ctrl-C is a documented way out of
these loops (GodScreen's class docstring: "``q``/Esc/Ctrl-C quits";
in raw mode _KeyReader translates the Ctrl-C byte into "q", so these
handlers are the canonical-mode / sleep-path translation of the
same keypress).  The E106 rule itself exempts a deliberate
shutdown hook.  Each pin proves the guarded operation was ATTEMPTED
(spy) and asserts the documented fallback, plus — where the console
failure ledger is in scope — that a deliberate Ctrl-C is NOT
recorded as a failure.
"""

from __future__ import annotations

import io
import time

import pytest

import nomorals.console.widgets as w
from nomorals.console import failures as cf


@pytest.fixture(autouse=True)
def _clean_ledger():
    cf.clear_failures()
    yield
    cf.clear_failures()


class _FakeKeys:
    """Stand-in for _KeyReader: records get() calls, replays a script."""

    def __init__(self, exc=None, val=None):
        self.exc, self.val, self.calls = exc, val, []

    def get(self, timeout):
        self.calls.append(timeout)
        if self.exc is not None:
            raise self.exc
        return self.val


def _god(color=False):
    return w.GodScreen(interval=0.5, snapshot=lambda: {},
                       color=color, out=io.StringIO())


def _ki_sleep(calls):
    def boom(dt):
        calls.append(dt)
        raise KeyboardInterrupt
    return boom


# ── 751: LiveScreen.tick — Ctrl-C ends the frame loop ──────────────────────

def test_751_tick_keyboard_interrupt_returns_false(monkeypatch):
    scr = w.LiveScreen(interval=0.5, color=False, out=io.StringIO())
    calls = []
    monkeypatch.setattr(time, "sleep", _ki_sleep(calls))
    assert scr.tick() is False  # "False when interrupted" (docstring)
    assert calls == [0.5]  # the sleep was attempted with the interval
    # Control: an uninterrupted sleep reports "keep going".
    monkeypatch.setattr(time, "sleep", lambda dt: None)
    assert scr.tick() is True
    assert not any(f.exc_type == "KeyboardInterrupt"
                   for f in cf.recent_failures(limit=200))


# ── 1410: GodScreen.run — the top-level Ctrl-C shutdown hook ───────────────

def test_1410_run_keyboard_interrupt_exits_cleanly(monkeypatch):
    gs = _god(color=True)

    def boom(self, out):
        raise KeyboardInterrupt

    monkeypatch.setattr(w.GodScreen, "_render_frame", boom)
    out = gs.run()
    assert gs._stop is True
    assert "exited live dashboard" in out
    # A deliberate Ctrl-C is not a failure: nothing is recorded under
    # the run site and no KeyboardInterrupt lands in the ledger.  (An
    # unrelated _KeyReader termios probe entry may legitimately
    # appear on a non-tty stdin; it is not this site.)
    assert not any(f.site == "widgets.GodScreen.run"
                   for f in cf.recent_failures(limit=200))
    assert not any(f.exc_type == "KeyboardInterrupt"
                   for f in cf.recent_failures(limit=200))


# ── 1431 / 1437: _wait_key sleep branches — Ctrl-C means quit ──────────────

def test_1431_wait_key_no_keys_sleep_interrupt_quits(monkeypatch):
    gs = _god()
    calls = []
    monkeypatch.setattr(time, "sleep", _ki_sleep(calls))
    assert gs._wait_key(None) is False  # False = quit requested
    assert calls == [0.5]


def test_1437_wait_key_stdin_unusable_sleep_interrupt_quits(monkeypatch):
    gs = _god()
    gs._stdin_ok = False
    calls = []
    monkeypatch.setattr(time, "sleep", _ki_sleep(calls))
    assert gs._wait_key(_FakeKeys(val="x")) is False
    assert calls == [0.5]


# ── 1454: _wait_key key read — canonical-mode Ctrl-C means quit ────────────

def test_1454_wait_key_get_interrupt_quits():
    gs = _god()
    gs._stdin_ok = True
    keys = _FakeKeys(exc=KeyboardInterrupt())
    assert gs._wait_key(keys) is False
    assert keys.calls == [0.5]  # get() was attempted with the interval


# ── 1504: _read_line_mode — Ctrl-C aborts the edit, loop lives ─────────────

def test_1504_line_mode_interrupt_aborts_edit():
    gs = _god()
    gs._cmd_buf = "qui"
    keys = _FakeKeys(exc=KeyboardInterrupt())
    # not self._stop -> True: the edit is aborted (buffers cleared,
    # Esc-parity), the watch loop keeps rendering.
    assert gs._read_line_mode(keys) is True
    assert gs._cmd_buf is None and gs._filter_buf is None
    assert keys.calls == [0.2]
