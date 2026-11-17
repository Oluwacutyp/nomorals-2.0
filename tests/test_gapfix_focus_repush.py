"""Gapfix: held-focus notifications arrive when the session ends (R3-55 remainder).

The R3-55 focus gate holds an ordinary page as ``held-focus`` while a
session runs — but nothing ever released it: ``RETRYABLE_STATES`` is
(pending, failed), so the redelivery queue could never pick it up, and
no replay path existed. These tests prove the release at the two seams
that now exist:

- ``Notifier.flush_held_focus`` — called by the ``/focus stop`` verb
  right after the session stops: still-fresh rows go out through the
  normal publish path at once;
- ``Notifier.redeliver`` — the periodic tick also releases held-focus
  rows, but ONLY once no session is active (the hold still applies
  while one runs).

Release is never a blind replay (the notifier's "never redelivered
stale" rule): every row passes the Pass-3 freshness recheck first —
superseded facts retire unsent, moot-deadline actions demote to the
digest tier — and the quiet-hours + brain gates are re-checked at
release time. A row the gates still hold stays held (it is not lost:
it reads on /notify and releases at the next stop/tick once the gate
clears). A failed send leaves the row held-focus for the next stop.

Failing-first: on the unmodified tree ``flush_held_focus`` does not
exist and ``redeliver`` ignores held-focus rows entirely, so every
test here fails there.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nomorals.agents.alert_policy import AlertPolicy
from nomorals.agents.notifier import (
    STATE_DIGEST,
    STATE_HELD_FOCUS,
    STATE_SENT,
    STATE_SUPERSEDED,
    Notifier,
)
from nomorals.focus.sessions import FocusTimer


class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.message_id = "m1"


class FakeGateway:
    """Records every send; ``live`` says which platforms run."""

    def __init__(self, live=()):
        self.live = set(live)
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        return {p: {"running_in_session": p in self.live}
                for p in ("telegram", "whatsapp")}

    def send(self, platform, chat_ref, text, buttons=None):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)


class FakeBrainGate:
    """A competence gate the test can degrade on demand."""

    def __init__(self, holding=False):
        self.holding = holding

    def hold_unsolicited(self, *, platform, chat_key, note="",
                         personal=False):
        return bool(self.holding)


def make_ctx(tmp_path, gateway=None, brain_gate=None):
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
    extras = {}
    if gateway is not None:
        extras["gateway"] = gateway
    if brain_gate is not None:
        extras["brain_gate"] = brain_gate
    return SimpleNamespace(db=db, settings=settings, extras=extras)


def _focus_path(tmp_path):
    return str(Path(tmp_path) / "data" / "focus.json")


def _start_focus(tmp_path, label="deep work"):
    return FocusTimer(_focus_path(tmp_path)).start(label)


def _stop_focus(tmp_path):
    return FocusTimer(_focus_path(tmp_path)).stop()


def _rows(ctx, state=""):
    sql = "SELECT kind, title, delivered, delivery_state FROM notifications"
    params: tuple = ()
    if state:
        sql += " WHERE delivery_state = ?"
        params = (state,)
    return [dict(r) for r in ctx.db.query(sql + " ORDER BY rowid", params)]


def _row(ctx, nid):
    rows = ctx.db.query(
        "SELECT * FROM notifications WHERE id = ?", (nid,))
    return rows[0] if rows else None


def _cand(ctx, nid):
    return AlertPolicy.candidate_for_notification(ctx.db, nid)


def _held_focus(ctx, gateway, kind="watcher", title="routine ping",
                body="body", **publish_kw):
    """Start a focus session, publish an ordinary page, assert the hold."""
    _start_focus(ctx.settings.home_path)
    notifier = Notifier(ctx, gateway)
    res = notifier.publish(kind, title, body, **publish_kw)
    assert res["delivery_state"] == STATE_HELD_FOCUS
    assert res["delivered"] == 0
    assert gateway.sent == []
    return notifier, res


def _quiet_now(ctx):
    """Point the partner quiet window at the current UTC hour.

    The same deterministic trick the Pass-3 suite uses: the window
    always covers "now", whatever hour the test runs in — no wall-clock
    flake, no clock mock.
    """
    hour = datetime.now(ZoneInfo("UTC")).hour
    partner = ctx.settings.partner
    old = (partner.quiet_start, partner.quiet_end)
    partner.quiet_start = hour
    partner.quiet_end = (hour + 1) % 24
    return old


@pytest.fixture()
def gw():
    return FakeGateway(live={"telegram"})


# ── the stop flush ───────────────────────────────────────────────────


class TestStopFlush:
    def test_held_row_flushed_on_stop(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, held = _held_focus(ctx, gw)
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out == {"sent": 1, "held": 0, "retired": 0}
        assert len(gw.sent) == 1
        assert "routine ping" in gw.sent[0][2]
        row = _row(ctx, held["id"])
        assert row["delivery_state"] == STATE_SENT
        assert row["delivered"] == 1

    def test_flush_clears_notify_held_list(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, held = _held_focus(ctx, gw)
        assert [r["id"] for r in notifier.pending()] == [held["id"]]
        assert notifier.queue_depth()["held"] == 1
        _stop_focus(tmp_path)
        notifier.flush_held_focus()
        assert notifier.pending() == []
        assert notifier.queue_depth() == {
            "retryable": 0, "held": 0, "dead": 0}

    def test_flush_with_nothing_held_is_silent(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        assert Notifier(ctx, gw).flush_held_focus() == {
            "sent": 0, "held": 0, "retired": 0}
        assert gw.sent == []

    def test_stop_verb_flushes_and_says_so(self, tmp_path, gw):
        from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

        ctx = make_ctx(tmp_path, gateway=gw)
        _held_focus(ctx, gw)
        obj = object.__new__(RuntimeReachMixin)
        obj.context = ctx
        obj.gateway = gw
        reply = obj._control_focus("stop")
        assert "stopped" in reply
        assert "1 held alert(s) delivered" in reply
        assert len(gw.sent) == 1
        assert _rows(ctx, STATE_HELD_FOCUS) == []


# ── freshness: superseded + stale never page ─────────────────────────


class TestFlushFreshness:
    # A held-focus row's candidate is a legacy notice: any asserted
    # fact (revision transition, outcome, action need) escalates the
    # decision and the focus gate interrupts instead of holding.  So
    # the held row carries its subject via alert_subject only — no
    # fact asserted, no escalation, still subject-bearing for the
    # recheck — and the superseding record is the later "done"
    # publish, which pages straight through focus as an
    # interruption (the realistic shape: the completion arrives,
    # the stale wait never does).
    _DONE = {"subject": "job:render-7", "prev_revision": "r1",
             "current_revision": "r2", "outcome": "succeeded",
             "result_refs": ["artifact:frame-9"]}
    _STALE = {"subject": "ops:deploy",
              "change_evidence": "blocked on credentials",
              "action_needed": True,
              "next_actions": ["provide credential"],
              "deadline": "2020-01-01T00:00:00+00:00"}

    def test_superseded_row_never_flushed(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, waiting = _held_focus(
            ctx, gw, kind="job", title="render-7 waiting", body="at r1",
            alert_subject="job:render-7")
        done = notifier.publish(
            "job", "render-7 done", "at r2",
            alert_subject="job:render-7",
            output_meta=dict(self._DONE))
        assert done["delivered"] is True  # paged through focus at once
        assert len(gw.sent) == 1
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out == {"sent": 0, "held": 0, "retired": 1}
        assert len(gw.sent) == 1  # the stale text never went out
        row = _row(ctx, waiting["id"])
        assert row["delivery_state"] == STATE_SUPERSEDED
        assert row["delivered"] == 0
        cand = _cand(ctx, waiting["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "superseded"
        assert release["ack"] == "superseded"
        assert "stale" in release["detail"]

    def test_stale_row_demotes_not_pages(self, tmp_path, gw):
        # An action-need candidate escalates and interrupts focus,
        # so no real publish can park one as held-focus.  Park it
        # the way the Pass-3 suite parks held rows: publish with
        # the gateway down (the page fails), then put the decided
        # row in exactly the state the flush keys on.
        ctx = make_ctx(tmp_path, gateway=gw)
        _start_focus(tmp_path)
        gw.live = set()
        notifier = Notifier(ctx, gw)
        held = notifier.publish(
            "ops", "deploy blocked", "needs you",
            output_meta=dict(self._STALE))
        assert held["delivery_state"] == "failed"
        with ctx.db.transaction():
            ctx.db.execute(
                "UPDATE notifications SET delivery_state = ? "
                "WHERE id = ?", (STATE_HELD_FOCUS, held["id"]))
        gw.live = {"telegram"}
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out == {"sent": 0, "held": 0, "retired": 1}
        assert gw.sent == []  # never paged as a live action
        row = _row(ctx, held["id"])
        assert row["delivery_state"] == STATE_DIGEST
        assert row["delivered"] == 0
        cand = _cand(ctx, held["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "moot-deadline"
        assert release["ack"] == "demoted-to-digest"


# ── gates re-checked at release ──────────────────────────────────────


class TestFlushGates:
    def test_quiet_hours_at_stop_keeps_hold(self, tmp_path, gw):
        # kind="job": "watcher"/"briefing" are quiet-exempt by
        # design (their own quiet logic), so only a non-exempt
        # kind proves the flush re-checks the clock.
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, held = _held_focus(ctx, gw, kind="job",
                                     title="render done")
        old = _quiet_now(ctx)
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out == {"sent": 0, "held": 1, "retired": 0}
        assert gw.sent == []
        assert _row(ctx, held["id"])["delivery_state"] == STATE_HELD_FOCUS
        # the tick honours the same gate — and releases once it clears
        assert notifier.redeliver() == 0
        assert _row(ctx, held["id"])["delivery_state"] == STATE_HELD_FOCUS
        ctx.settings.partner.quiet_start, ctx.settings.partner.quiet_end = old
        assert notifier.redeliver() == 1
        assert _row(ctx, held["id"])["delivery_state"] == STATE_SENT

    def test_quiet_exempt_kind_flushes_during_quiet(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, held = _held_focus(ctx, gw, kind="briefing",
                                     title="morning brief")
        _quiet_now(ctx)
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out["sent"] == 1
        assert _row(ctx, held["id"])["delivery_state"] == STATE_SENT

    def test_brain_degraded_at_stop_keeps_hold(self, tmp_path, gw):
        brain = FakeBrainGate(holding=False)
        ctx = make_ctx(tmp_path, gateway=gw, brain_gate=brain)
        notifier, held = _held_focus(ctx, gw)
        brain.holding = True  # degraded between the hold and the release
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out == {"sent": 0, "held": 1, "retired": 0}
        assert gw.sent == []
        assert _row(ctx, held["id"])["delivery_state"] == STATE_HELD_FOCUS
        brain.holding = False
        out = notifier.flush_held_focus()
        assert out["sent"] == 1
        assert _row(ctx, held["id"])["delivery_state"] == STATE_SENT

    def test_brain_gate_does_not_hold_non_proactive_kinds(self, tmp_path, gw):
        brain = FakeBrainGate(holding=True)
        ctx = make_ctx(tmp_path, gateway=gw, brain_gate=brain)
        notifier, held = _held_focus(ctx, gw, kind="job",
                                     title="render done")
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out["sent"] == 1
        assert _row(ctx, held["id"])["delivery_state"] == STATE_SENT


# ── the tick safety net ──────────────────────────────────────────────


class TestTickRelease:
    def test_tick_flushes_after_session_ends(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, held = _held_focus(ctx, gw)
        _stop_focus(tmp_path)
        assert notifier.redeliver() == 1
        assert len(gw.sent) == 1
        assert _row(ctx, held["id"])["delivery_state"] == STATE_SENT

    def test_tick_never_flushes_while_session_active(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, held = _held_focus(ctx, gw)
        assert notifier.redeliver() == 0
        assert gw.sent == []
        assert _row(ctx, held["id"])["delivery_state"] == STATE_HELD_FOCUS

    def test_failed_flush_stays_held_for_next_stop(self, tmp_path, gw):
        ctx = make_ctx(tmp_path, gateway=gw)
        notifier, held = _held_focus(ctx, gw)
        gw.live = set()  # gateway down at release time
        _stop_focus(tmp_path)
        out = notifier.flush_held_focus()
        assert out == {"sent": 0, "held": 1, "retired": 0}
        assert _row(ctx, held["id"])["delivery_state"] == STATE_HELD_FOCUS
        gw.live = {"telegram"}
        out = notifier.flush_held_focus()
        assert out["sent"] == 1
        assert _row(ctx, held["id"])["delivery_state"] == STATE_SENT
