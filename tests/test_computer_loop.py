"""The control loop, exercised for real: perceive → decide → act
→ verify against a live X application on a real virtual display.

The fixture is a Tk app whose Apply button retitles the window
with the entry's text — so a full click/type/click round trip is
verified through TWO independent channels (the X window title and
the screenshot pixels). The failure semantics are pinned too: an
unmet expectation ends the run as a failure with the reason, no
silent retry; the budget ends it; the gate closing mid-run ends
it.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import pytest

from nomorals.computer import loop, provision
from nomorals.computer.loop import (
    Action,
    DeciderError,
    LLMDecider,
    PointerAt,
    ScriptedDecider,
    TitleIs,
    WindowPresent,
    parse_action_reply,
)
from nomorals.computer.sessions import SessionManager

pytestmark = pytest.mark.skipif(
    shutil.which("Xvfb") is None, reason="Xvfb not installed on this host")

FIXTURE_APP = '''
import tkinter as tk

root = tk.Tk()
root.title("FixturePad")
root.geometry("420x160+120+90")
entry = tk.Entry(root, width=40)
entry.pack(padx=10, pady=10)


def apply():
    text = entry.get()
    if text:
        root.title(text)


tk.Button(root, text="Apply", command=apply).pack(pady=6)
entry.focus_force()
root.mainloop()
'''


@pytest.fixture()
def session(tmp_path, monkeypatch):
    number = provision.find_free_display()
    auth = tmp_path / "xauth"
    provision.write_xauthority(auth, number)
    server = provision.start_xvfb(number, 1280, 1024, auth)
    monkeypatch.setenv("NM_COMPUTER_DISPLAY", f":{number}")
    monkeypatch.setenv("XAUTHORITY", str(auth))
    manager = SessionManager(home=tmp_path / "nmhome")
    rec = manager.open_session()
    yield manager, rec
    manager.close_session(rec.id)
    provision.stop_process(server.pid)


@pytest.fixture()
def app_path(tmp_path):
    path = tmp_path / "fixture_app.py"
    path.write_text(FIXTURE_APP)
    return path


def _fixture_window(perception):
    for win in perception.windows:
        if win.get("title") in ("FixturePad",):
            return win
    return None


class PadDriver:
    """A decider that reads each perception and drives the fixture:
    launch → click the entry → type → click Apply → done, with an
    expectation riding on the steps that must change the world."""

    def __init__(self, app_path, text):
        self.app_path = app_path
        self.text = text
        self.stage = 0

    def decide(self, perception, goal, history):
        win = _fixture_window(perception)
        if self.stage == 0:
            self.stage = 1
            return Action(
                "launch", {"argv": [sys.executable, str(self.app_path)]},
                expect=WindowPresent("FixturePad"))
        if self.stage == 1 and win is not None:
            self.stage = 2
            return Action(
                "click", {"x": win["x"] + 200, "y": win["y"] + 22},
                expect=PointerAt(win["x"] + 200, win["y"] + 22))
        if self.stage == 2:
            self.stage = 3
            return Action("type", {"text": self.text})
        if self.stage == 3 and win is not None:
            self.stage = 4
            return Action(
                "click", {"x": win["x"] + 210, "y": win["y"] + 62},
                expect=TitleIs(self.text))
        return None


class TestLoopEndToEnd:
    def test_full_round_trip_verified(self, session, app_path):
        manager, rec = session
        report = loop.run_task(
            manager, rec, "type into the pad and apply",
            PadDriver(app_path, "hello desktop"),
            max_steps=12, settle_seconds=0.45)
        assert report.status == "completed", report.describe()
        assert len(report.steps) == 4
        assert all(step.ok for step in report.steps)
        verified = [s for s in report.steps if s.verified is not None]
        assert verified and all(s.verified for s in verified)
        # The world really changed: the title now IS the text, seen
        # in a fresh perception after the run.
        seen = loop.capture(manager, rec)
        assert any(w.get("title") == "hello desktop"
                   for w in seen.windows)
        for step in report.steps:
            assert step.before_frame and step.after_frame
            assert os.path.exists(step.before_frame)

    def test_llm_decider_drives_a_step(self, session, app_path):
        manager, rec = session
        import json as _json

        replies = iter([
            _json.dumps({"action": {
                "kind": "launch",
                "args": {"argv": [sys.executable, str(app_path)]}},
                "note": "open it"}),
            _json.dumps({"done": True, "note": "launched"}),
        ])
        decider = LLMDecider(complete_fn=lambda prompt: next(replies))
        report = loop.run_task(manager, rec, "open the pad", decider,
                               max_steps=4, settle_seconds=0.4)
        assert report.status == "completed", report.describe()
        seen = loop.capture(manager, rec)
        assert any(w.get("title") == "FixturePad" for w in seen.windows)


class TestLoopHonesty:
    def test_unmet_expectation_fails_without_retry(self, session,
                                                   app_path):
        manager, rec = session
        actions = [
            Action("launch", {"argv": [sys.executable, str(app_path)]},
                   expect=TitleIs("a title that will never exist")),
            Action("type", {"text": "must never be typed"}),
        ]
        report = loop.run_task(manager, rec, "impossible",
                               ScriptedDecider(actions),
                               max_steps=6, settle_seconds=0.4)
        assert report.status == "failed"
        assert "did not do what was expected" in report.reason
        # Exactly one step ran: the failure stopped the run; the
        # second action was never attempted (no silent retry loop).
        assert len(report.steps) == 1
        assert report.steps[0].verified is False

    def test_driver_error_is_reported_not_retried(self, session):
        manager, rec = session
        actions = [Action("launch", {"argv": ["/nonexistent/binary"]}),
                   Action("wait", {"seconds": 0.1})]
        report = loop.run_task(manager, rec, "launch nothing",
                               ScriptedDecider(actions), max_steps=4)
        assert report.status == "failed"
        assert "launch" in report.reason
        assert len(report.steps) == 1

    def test_budget_exhaustion(self, session):
        manager, rec = session

        class ForeverWait:
            def decide(self, perception, goal, history):
                return Action("wait", {"seconds": 0.05})

        report = loop.run_task(manager, rec, "wait forever",
                               ForeverWait(), max_steps=3,
                               settle_seconds=0.0)
        assert report.status == "budget_exhausted"
        assert len(report.steps) == 3

    def test_gate_closing_mid_run_stops_the_run(self, session):
        manager, rec = session

        class StopAfterFirst:
            def __init__(self):
                self.done = False

            def decide(self, perception, goal, history):
                if not self.done:
                    self.done = True
                    return Action("move", {"x": 40, "y": 40},
                                  expect=PointerAt(40, 40))
                manager.set_owner_stop(rec.id, True, by="test owner")
                return Action("move", {"x": 80, "y": 80})

        report = loop.run_task(manager, rec, "two moves",
                               StopAfterFirst(), max_steps=5,
                               settle_seconds=0.1)
        assert report.status == "gate_refused"
        assert "sticky" in report.reason
        # The pointer never reached the second target.
        seen = loop.capture(manager, rec)
        assert (seen.pointer["x"], seen.pointer["y"]) == (40, 40)
        manager.set_owner_stop(rec.id, False)


class TestReplyParsing:
    def test_done_reply(self):
        assert parse_action_reply('{"done": true}') is None

    def test_action_reply_with_prose_around_it(self):
        action = parse_action_reply(
            'Here is my move: {"action": {"kind": "click", '
            '"args": {"x": 3, "y": 4}}, "note": "go"} — done')
        assert action is not None
        assert action.kind == "click"
        assert action.args == {"x": 3, "y": 4}

    def test_garbage_reply_raises(self):
        with pytest.raises(DeciderError):
            parse_action_reply("I would click the thing, probably")

    def test_unknown_kind_raises(self):
        with pytest.raises(DeciderError):
            parse_action_reply(
                '{"action": {"kind": "teleport", "args": {}}}')

    def test_pixel_diff_measured_when_pillow_present(self, session):
        manager, rec = session
        before = loop.capture(manager, rec)
        manager.act(rec, "launch",
                    {"argv": [sys.executable, "-c",
                              "import tkinter as tk; r = tk.Tk(); "
                              "r.title('PixelPop'); r.geometry('300x200'); "
                              "r.after(1500, r.destroy); r.mainloop()"]})
        deadline = time.time() + 10
        after = before
        while time.time() < deadline:
            after = loop.capture(manager, rec)
            if any(w.get("title") == "PixelPop" for w in after.windows):
                break
            time.sleep(0.2)
        fraction = loop.pixel_diff_fraction(before.png, after.png)
        assert fraction is not None and fraction > 0.001
