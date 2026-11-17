"""Gapfix R3-55: the focus interruption policy gates Notifier.publish.

Failing-first: on the unmodified tree, a publish during an active
focus session is delivered exactly as if no session existed — the
policy in ``focus/interruption.py`` is never consulted. These tests
prove the wiring at the one delivery choke point (``publish``):

- no focus session → delivery byte-identical to before;
- active session + ordinary page → held for the digest
  (``held-focus``), nothing pushed, row stored, reason on record;
- active session + escalated page (failure/actionable) → still
  delivered, and the interruption is logged on the session;
- ``critical`` bypasses the gate (same precedent as quiet hours);
- session over → delivery resumes; no home path → gate is a no-op.

Only interrupt/defer arise at this seam (neutral urgency/importance
— no signal in the publish args, stated not invented): the policy's
``drop`` branch is pinned at the module level in
``tests/test_focus_interruption.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.notifier import Notifier
from nomorals.focus.sessions import FocusTimer


class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.message_id = "m1"


class FakeGateway:
    """Records every send; the telegram owner channel is live."""

    def __init__(self):
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        return {"telegram": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, buttons=None):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


def make_ctx(tmp_path, gateway=None):
    from nomorals.storage.db import Database

    db = Database(os.path.join(str(tmp_path), "test.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=0,
        quiet_end=0,  # empty quiet window: time-of-day never gates
        timezone="UTC",
    )
    settings = SimpleNamespace(
        workspace_dir=str(tmp_path), partner=partner,
        home_path=Path(tmp_path),
        resolve=lambda rel: os.path.join(str(tmp_path), rel))
    ctx = SimpleNamespace(
        db=db, settings=settings,
        extras={"gateway": gateway} if gateway else {})
    return ctx


def _focus_path(tmp_path):
    return str(Path(tmp_path) / "data" / "focus.json")


def _start_focus(tmp_path, label="deep work"):
    timer = FocusTimer(_focus_path(tmp_path))
    session = timer.start(label)
    return session


def _rows(ctx, state=""):
    sql = "SELECT kind, title, delivered, delivery_state FROM notifications"
    params: tuple = ()
    if state:
        sql += " WHERE delivery_state = ?"
        params = (state,)
    return [dict(r) for r in ctx.db.query(sql + " ORDER BY rowid", params)]


@pytest.fixture()
def gw():
    return FakeGateway()


class TestFocusGate:
    def test_no_focus_session_delivers_unchanged(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        result = Notifier(ctx, gateway=gw).publish(
            "watcher", "routine ping", "body")
        assert len(gw.sent) == 1
        assert result["delivery_state"] == "sent"
        assert result["delivered"] == 1

    def test_ordinary_page_held_during_focus(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        _start_focus(tmp_path)
        result = Notifier(ctx, gateway=gw).publish(
            "watcher", "routine ping", "body")
        assert gw.sent == []
        assert result["delivery_state"] == "held-focus"
        assert result["delivered"] == 0
        rows = _rows(ctx, "held-focus")
        assert len(rows) == 1
        assert rows[0]["title"] == "routine ping"

    def test_escalated_page_gets_through_during_focus(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        _start_focus(tmp_path)
        result = Notifier(ctx, gateway=gw).publish(
            "watcher", "disk failure", "body",
            alert_hints={"failed": True, "subject": "disk"})
        assert len(gw.sent) == 1
        assert result["delivery_state"] == "sent"
        # The interruption is logged on the session (the counter was
        # dead before this lane — no production caller ever moved it).
        timer = FocusTimer(_focus_path(tmp_path))
        active = timer.active()
        assert active is not None
        assert active.interruptions == 1

    def test_critical_bypasses_focus_gate(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        _start_focus(tmp_path)
        result = Notifier(ctx, gateway=gw).publish(
            "watcher", "routine ping", "body", critical=True)
        assert len(gw.sent) == 1
        assert result["delivery_state"] == "sent"

    def test_delivery_resumes_when_session_ends(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        session = _start_focus(tmp_path)
        first = Notifier(ctx, gateway=gw).publish(
            "watcher", "routine ping", "body")
        assert first["delivery_state"] == "held-focus"
        FocusTimer(_focus_path(tmp_path)).stop(session.id)
        second = Notifier(ctx, gateway=gw).publish(
            "watcher", "routine ping 2", "body")
        assert second["delivery_state"] == "sent"
        assert len(gw.sent) == 1

    def test_no_home_path_is_a_noop(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        del ctx.settings.home_path
        _start_focus(tmp_path)
        result = Notifier(ctx, gateway=gw).publish(
            "watcher", "routine ping", "body")
        assert len(gw.sent) == 1
        assert result["delivery_state"] == "sent"
