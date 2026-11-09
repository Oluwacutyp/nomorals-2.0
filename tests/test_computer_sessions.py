"""Desktop sessions against real displays: persistence, recovery,
and the personal-surface control gate.

Every session here runs against a real X server (Xvfb): the
lifecycle claims — reattach after a restart, honest loss when the
display dies, the sticky owner stop surviving a fresh manager —
are proven against the real store file and the real server, not
against a model of them.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

import pytest

from nomorals.computer import loop, provision
from nomorals.computer.sessions import GateRefused, SessionError, SessionManager

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
def host_display(tmp_path, monkeypatch):
    """A test-owned Xvfb the manager attaches to in host mode."""
    number = provision.find_free_display()
    auth = tmp_path / "xauth"
    provision.write_xauthority(auth, number)
    server = provision.start_xvfb(number, 1280, 1024, auth)
    monkeypatch.setenv("NM_COMPUTER_DISPLAY", f":{number}")
    monkeypatch.setenv("XAUTHORITY", str(auth))
    yield f":{number}"
    provision.stop_process(server.pid)


@pytest.fixture()
def home(tmp_path):
    return tmp_path / "nmhome"


def _wait_for(predicate, timeout=12.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.15)
    return False


class TestLifecycle:
    def test_open_persists_and_reattaches(self, host_display, home):
        first = SessionManager(home=home)
        rec = first.open_session(mission="recon")
        assert rec.state == "open" and rec.mode == "host"
        assert rec.width == 1280 and rec.height == 1024
        # The store on disk carries the session (Devon-restart truth).
        doc = json.loads(
            (home / "computer" / "sessions.json").read_text())
        assert rec.id in doc["sessions"]
        # A fresh manager (a restarted Devon) sees it and reattaches.
        second = SessionManager(home=home)
        seen = second.get(rec.id)
        assert seen is not None and seen.state == "open"
        client = second.client_for(seen)
        assert client.ping() is True
        events = [row["event"] for row in second.journal_tail(20)]
        assert "session_reattached" in events
        second.close_session(rec.id)
        assert second.get(rec.id).state == "closed"

    def test_virtual_session_owns_its_display(self, home, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("NM_COMPUTER_DISPLAY", raising=False)
        manager = SessionManager(home=home)
        rec = manager.open_session()
        try:
            assert rec.mode == "virtual"
            assert provision.process_alive(rec.xvfb_pid)
            seen = loop.capture(manager, rec)
            assert (seen.width, seen.height) == (1280, 1024)
        finally:
            manager.close_session(rec.id)
        assert not provision.process_alive(rec.xvfb_pid)

    def test_lost_when_the_display_dies(self, home, tmp_path, monkeypatch):
        number = provision.find_free_display()
        auth = tmp_path / "xauth"
        provision.write_xauthority(auth, number)
        server = provision.start_xvfb(number, 1280, 1024, auth)
        monkeypatch.setenv("NM_COMPUTER_DISPLAY", f":{number}")
        monkeypatch.setenv("XAUTHORITY", str(auth))
        manager = SessionManager(home=home)
        rec = manager.open_session()
        assert manager.client_for(rec).ping() is True
        # The display dies underneath the session (host reboot, X
        # crash): the record must turn *lost*, honestly.
        provision.stop_process(server.pid)
        reconciled = manager.reconcile(rec)
        assert reconciled.state == "lost"
        assert "no longer answers" in reconciled.note or \
            "gone" in reconciled.note
        with pytest.raises(SessionError):
            manager.client_for(reconciled)
        events = [row["event"] for row in manager.journal_tail(20)]
        assert "session_lost" in events


class TestControlGate:
    def test_owner_stop_is_sticky_across_managers(self, host_display,
                                                  home):
        owner = SessionManager(home=home)
        rec = owner.open_session()
        owner.set_owner_stop(rec.id, True)
        # A brand-new manager — a Devon restart, a reconcile, any
        # supervisor — still sees the stop. Nothing clears it but
        # the owner's resume.
        restarted = SessionManager(home=home)
        fresh = restarted.get(rec.id)
        assert fresh is not None and fresh.owner_stop is True
        reasons = restarted.gate_reasons(fresh)
        assert any("sticky" in r for r in reasons)
        with pytest.raises(GateRefused):
            restarted.act(fresh, "mouse_move", {"x": 1, "y": 1})
        # Watching still works under the stop (eyes, not hands).
        seen = loop.capture(restarted, fresh)
        assert seen.width == 1280
        # A takeover + handback cycle must NOT clear the stop.
        restarted.set_takeover(rec.id, True)
        restarted.set_takeover(rec.id, False)
        assert restarted.get(rec.id).owner_stop is True
        # Only the explicit resume clears it.
        restarted.set_owner_stop(rec.id, False)
        assert restarted.gate_reasons(restarted.get(rec.id)) == []
        restarted.act(restarted.get(rec.id), "mouse_move",
                      {"x": 5, "y": 5})
        restarted.close_session(rec.id)

    def test_takeover_pauses_and_handback_resumes(self, host_display,
                                                  home):
        manager = SessionManager(home=home)
        rec = manager.open_session()
        manager.act(rec, "mouse_move", {"x": 10, "y": 10})
        manager.set_takeover(rec.id, True, by="owner phone")
        with pytest.raises(GateRefused) as refused:
            manager.act(rec, "mouse_move", {"x": 20, "y": 20})
        assert "taken over" in str(refused.value)
        manager.set_takeover(rec.id, False)
        result = manager.act(rec, "mouse_move", {"x": 30, "y": 30})
        assert (result["x"], result["y"]) == (30, 30)
        manager.close_session(rec.id)

    def test_degraded_brain_is_observe_only(self, host_display, home):
        owner = SessionManager(home=home)
        rec = owner.open_session()
        # A second manager over the same home whose brain probe
        # reports degraded: the session may be watched, never
        # driven — the gate binds the hands, not the eyes.
        degraded_view = SessionManager(home=home,
                                       brain_probe=lambda: False)
        target = degraded_view.get(rec.id)
        assert target is not None
        with pytest.raises(GateRefused) as refused:
            degraded_view.act(target, "mouse_move", {"x": 1, "y": 1})
        assert "observe-only" in str(refused.value)
        seen = loop.capture(degraded_view, target)
        assert seen.height == 1024
        owner.close_session(rec.id)


class TestRealWindows:
    def test_launched_app_appears_in_perception(self, host_display, home,
                                                tmp_path):
        app = tmp_path / "fixture_app.py"
        app.write_text(FIXTURE_APP)
        manager = SessionManager(home=home)
        rec = manager.open_session()
        try:
            manager.act(rec, "launch",
                        {"argv": [sys.executable, str(app)]})
            found = _wait_for(lambda: any(
                w.get("title") == "FixturePad"
                for w in loop.capture(manager, rec).windows))
            assert found, "the fixture window never appeared"
            seen = loop.capture(manager, rec)
            win = next(w for w in seen.windows
                       if w["title"] == "FixturePad")
            assert (win["width"], win["height"]) == (420, 160)
            assert seen.frame_path and \
                os.path.getsize(seen.frame_path) > 0
            assert manager.latest_frame(rec.id) is not None
            manager.act(rec, "close_window", {"window": win["id"]})
            gone = _wait_for(lambda: all(
                w.get("title") != "FixturePad"
                for w in loop.capture(manager, rec).windows))
            assert gone, "the fixture window never closed"
        finally:
            manager.close_session(rec.id)
