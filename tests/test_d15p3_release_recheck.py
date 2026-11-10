"""Doc 15 Pass 3 — freshness recheck before release + routing.

Pinned here:

- a quiet-hours-held item whose fact is superseded while held is
  retired at release (state ``superseded``, journaled, never sent);
  the same item not superseded releases and sends, with the
  release verdict + ack on its candidate record;
- a failed-send retry whose fact moved on before the retry fires
  does not deliver the stale text; a persisting failure (same
  state re-asserted) is NOT superseded and still delivers;
- a held item past a parseable action deadline demotes to the
  digest tier instead of paging as a live action; a future or
  unparseable deadline changes nothing;
- routing is a decision output: the decision record carries the
  channels the decision chose, redelivery follows the RECORD
  (a publish restricted to whatsapp redelivers to whatsapp only),
  and release-time annotations never rewrite the routing;
- delivery acks land on the candidate record for sent / digest /
  suppressed / deduped / disabled / pending publishes, and the
  terminal redelivery acks (sent / dead) complete the chain;
- the digest fold is a release seam: a superseded digest item is
  retired at unfold and never folds (an all-superseded fold is
  empty — nothing is sent); a folded item's ack lands as
  ``digest-delivered``;
- the named bound: rows stored without the decision machinery
  (missions progress's direct quiet-hours stores) carry no
  recheck source and release exactly as before, with no verdict
  record fabricated for them;
- P2's decision-time semantics are untouched (the adjoining suite
  re-pins them).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nomorals.agents.alert_policy import (
    CANDIDATE_JOURNAL_KEY,
    DEFAULT_RECURRENCE_WINDOW_S,
    AlertPolicy,
)
from nomorals.agents.notifier import (
    STATE_DEAD,
    STATE_DIGEST,
    STATE_FAILED,
    STATE_HELD_QUIET,
    STATE_SENT,
    STATE_SUPERSEDED,
    Notifier,
)
from nomorals.agents.report_candidate import parse_deadline


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


def make_ctx(gateway=None, **partner_over):
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="d15p3-")
    db = Database(os.path.join(tmp, "test.db"))
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
    for key, value in partner_over.items():
        setattr(partner, key, value)
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner,
        resolve=lambda rel: os.path.join(tmp, rel))
    ctx = SimpleNamespace(
        db=db, settings=settings,
        extras={"gateway": gateway} if gateway else {})
    return ctx, tmp


@pytest.fixture()
def ctx():
    context, tmp = make_ctx()
    yield context
    shutil.rmtree(tmp, ignore_errors=True)


def _row(context, nid):
    rows = context.db.query(
        "SELECT * FROM notifications WHERE id = ?", (nid,))
    return rows[0] if rows else None


def _cand(context, nid):
    """The candidate record a publish's notification row links to."""
    return AlertPolicy.candidate_for_notification(context.db, nid)


def _set_state(context, nid, state):
    with context.db.transaction():
        context.db.execute(
            "UPDATE notifications SET delivery_state = ? WHERE id = ?",
            (state, nid))


_WAITING = {"subject": "job:render-7", "prev_revision": "r0",
            "current_revision": "r1",
            "change_evidence": "render started"}
_DONE = {"subject": "job:render-7", "prev_revision": "r1",
         "current_revision": "r2", "outcome": "succeeded",
         "result_refs": ["artifact:frame-9"]}


def _held_waiting(context, gateway):
    """A decided 'waiting' fact, stored held for quiet hours.

    Production held-quiet rows for governed kinds are stored by
    the proactive gate; the row state is what redeliver() keys
    on, so the test parks a decided row in exactly that state
    (the unlinked direct-store population is pinned separately).
    """
    notifier = Notifier(context, gateway)
    res = notifier.publish("job", "render-7 waiting", "at r1",
                           output_meta=dict(_WAITING))
    assert res["delivery_state"] == STATE_FAILED  # gateway down
    _set_state(context, res["id"], STATE_HELD_QUIET)
    return notifier, res


# ── (a) quiet-hours release: superseded retires, fresh sends ─────────


class TestQuietHoursRelease:
    def test_superseded_held_item_never_sends(self, ctx):
        gateway = FakeGateway(live=())
        notifier, held = _held_waiting(ctx, gateway)
        gateway.live = {"telegram"}
        done = notifier.publish("job", "render-7 done", "at r2",
                                output_meta=dict(_DONE))
        assert done["delivered"] is True
        assert len(gateway.sent) == 1

        assert notifier.redeliver() == 0
        assert len(gateway.sent) == 1  # the stale text never went out
        row = _row(ctx, held["id"])
        assert row["delivery_state"] == STATE_SUPERSEDED
        assert row["delivered"] == 0
        cand = _cand(ctx, held["id"])
        assert cand["delivery"]["state"] == "superseded"
        (release,) = cand["releases"]
        assert release["path"] == "redelivery"
        assert release["verdict"] == "superseded"
        assert release["ack"] == "superseded"
        assert release["superseded_by"] == done_cand_id(ctx, done)
        assert "stale" in release["detail"]
        assert release["age_s"] is not None
        # the row leaves every work queue
        assert notifier.pending() == []
        assert notifier.queue_depth() == {
            "retryable": 0, "held": 0, "dead": 0}

    def test_fresh_held_item_releases_with_ack(self, ctx):
        gateway = FakeGateway(live=())
        notifier, held = _held_waiting(ctx, gateway)
        gateway.live = {"telegram"}
        assert notifier.redeliver() == 1
        assert len(gateway.sent) == 1
        assert "render-7 waiting" in gateway.sent[0][2]
        row = _row(ctx, held["id"])
        assert row["delivery_state"] == STATE_SENT
        cand = _cand(ctx, held["id"])
        assert cand["delivery"]["state"] == "sent"
        (release,) = cand["releases"]
        assert release["verdict"] == "fresh"
        assert release["ack"] == "sent"
        assert release["age_s"] is not None

    def test_still_quiet_means_no_release_no_recheck(self, ctx):
        gateway = FakeGateway(live=())
        notifier, held = _held_waiting(ctx, gateway)
        hour = datetime.now(ZoneInfo("UTC")).hour
        partner = ctx.settings.partner
        partner.quiet_start = hour
        partner.quiet_end = (hour + 1) % 24
        gateway.live = {"telegram"}
        notifier.publish("job", "render-7 done", "at r2",
                         output_meta=dict(_DONE))
        assert notifier.redeliver() == 0
        row = _row(ctx, held["id"])
        assert row["delivery_state"] == STATE_HELD_QUIET  # still held
        cand = _cand(ctx, held["id"])
        assert cand.get("releases") in (None, [])  # recheck is at release


def done_cand_id(context, res):
    cand = _cand(context, res["id"])
    return cand["event_id"]


# ── (c) failed-send retry: moved-on facts are not delivered ──────────


class TestRetryRecheck:
    _FAIL = {"subject": "svc:api", "prev_revision": "r1",
             "current_revision": "r1", "outcome": "failed"}
    _FIXED = {"subject": "svc:api", "prev_revision": "r1",
              "current_revision": "r1", "outcome": "succeeded",
              "result_refs": ["artifact:health-1"]}

    def test_retry_does_not_deliver_stale_failure(self, ctx):
        gateway = FakeGateway(live=())
        notifier = Notifier(ctx, gateway)
        failed = notifier.publish("ops", "api is down", "boom",
                                  output_meta=dict(self._FAIL))
        assert failed["delivery_state"] == STATE_FAILED
        gateway.live = {"telegram"}
        fixed = notifier.publish("ops", "api recovered", "ok",
                                 output_meta=dict(self._FIXED))
        assert fixed["delivered"] is True
        assert notifier.redeliver() == 0
        assert len(gateway.sent) == 1  # only the recovery paged
        row = _row(ctx, failed["id"])
        assert row["delivery_state"] == STATE_SUPERSEDED
        cand = _cand(ctx, failed["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "superseded"
        assert release["superseded_by"] == done_cand_id(ctx, fixed)

    def test_persisting_failure_is_not_superseded(self, ctx):
        gateway = FakeGateway(live=())
        notifier = Notifier(ctx, gateway)
        failed = notifier.publish("ops", "api is down", "boom",
                                  output_meta=dict(self._FAIL))
        # the same failure re-asserted (deduped at publish) must
        # not retire the original: same subject, same state
        again = notifier.publish("ops", "api still down", "boom",
                                 output_meta=dict(self._FAIL))
        assert again["deduped"] is True
        gateway.live = {"telegram"}
        assert notifier.redeliver() == 1
        assert "api is down" in gateway.sent[0][2]
        cand = _cand(ctx, failed["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "fresh"
        assert release["ack"] == "sent"


# ── (b) moot deadline: demote, never page as a live action ───────────


class TestMootDeadline:
    _BASE = {"subject": "ops:deploy",
             "change_evidence": "blocked on credentials",
             "action_needed": True,
             "next_actions": ["provide credential"]}

    def _parked(self, context, gateway, **meta_over):
        notifier = Notifier(context, gateway)
        meta = dict(self._BASE, **meta_over)
        res = notifier.publish("ops", "deploy blocked", "needs you",
                               output_meta=meta)
        assert res["delivery_state"] == STATE_FAILED
        return notifier, res

    def test_past_deadline_demotes_to_digest(self, ctx):
        gateway = FakeGateway(live=())
        notifier, res = self._parked(
            ctx, gateway, deadline="2020-01-01T00:00:00+00:00")
        gateway.live = {"telegram"}
        assert notifier.redeliver() == 0
        assert gateway.sent == []  # never paged as a live action
        row = _row(ctx, res["id"])
        assert row["delivery_state"] == STATE_DIGEST
        assert row["delivered"] == 0
        cand = _cand(ctx, res["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "moot-deadline"
        assert release["ack"] == "demoted-to-digest"
        assert cand["delivery"]["state"] == "demoted-to-digest"
        # demoted means digest-level: the fold can still surface it
        unfolded = notifier.digest_unfolded()
        assert [r["id"] for r in unfolded] == [res["id"]]

    def test_epoch_deadline_also_demotes(self, ctx):
        gateway = FakeGateway(live=())
        notifier, res = self._parked(
            ctx, gateway, deadline=str(time.time() - 60.0))
        gateway.live = {"telegram"}
        assert notifier.redeliver() == 0
        assert _row(ctx, res["id"])["delivery_state"] == STATE_DIGEST

    def test_future_deadline_releases_normally(self, ctx):
        gateway = FakeGateway(live=())
        notifier, res = self._parked(
            ctx, gateway, deadline="2999-01-01T00:00:00+00:00")
        gateway.live = {"telegram"}
        assert notifier.redeliver() == 1
        assert len(gateway.sent) == 1
        cand = _cand(ctx, res["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "fresh"
        assert release["ack"] == "sent"

    def test_unparseable_deadline_never_demotes(self, ctx):
        gateway = FakeGateway(live=())
        notifier, res = self._parked(ctx, gateway,
                                     deadline="next Friday")
        gateway.live = {"telegram"}
        assert notifier.redeliver() == 1
        assert len(gateway.sent) == 1
        cand = _cand(ctx, res["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "fresh"
        assert "unparseable" in release["detail"]


# ── (d) routing: the decision record is what delivery follows ────────


class TestRoutingRecord:
    def test_redelivery_follows_recorded_channels(self):
        gateway = FakeGateway(live=())
        context, tmp = make_ctx(
            gateway, )
        context.settings.partner.owner_chats = \
            "telegram:111,whatsapp:222"
        try:
            notifier = Notifier(context, gateway)
            res = notifier.publish("ops", "wa only", "for whatsapp",
                                   channels=["whatsapp"])
            assert res["delivery_state"] == STATE_FAILED
            cand = _cand(context, res["id"])
            assert cand["decision"]["routing"] == {
                "audience": "owner",
                "channel_policy": "channels:whatsapp",
                "channels": ["whatsapp"]}
            gateway.live = {"telegram", "whatsapp"}
            assert notifier.redeliver() == 1
            platforms = {p for p, _, _ in gateway.sent}
            assert platforms == {"whatsapp"}  # the record ruled
            row = _row(context, res["id"])
            assert row["channel"] == "whatsapp"
            # release annotations never rewrite the routing
            after = _cand(context, res["id"])
            assert after["decision"]["routing"] == \
                cand["decision"]["routing"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_all_channels_record_and_release(self):
        gateway = FakeGateway(live=())
        context, tmp = make_ctx(gateway)
        context.settings.partner.owner_chats = \
            "telegram:111,whatsapp:222"
        try:
            notifier = Notifier(context, gateway)
            res = notifier.publish("ops", "everywhere", "all")
            assert res["delivery_state"] == STATE_FAILED
            cand = _cand(context, res["id"])
            assert cand["decision"]["routing"]["channel_policy"] == \
                "all-owner-channels"
            assert cand["decision"]["routing"]["channels"] == []
            gateway.live = {"telegram", "whatsapp"}
            assert notifier.redeliver() == 1
            platforms = {p for p, _, _ in gateway.sent}
            assert platforms == {"telegram", "whatsapp"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── (e) delivery acks complete the record ────────────────────────────


class TestDeliveryAcks:
    def test_sent_and_digest_and_suppressed_and_deduped(self, ctx):
        gateway = FakeGateway(live={"telegram"})
        notifier = Notifier(ctx, gateway)
        sent = notifier.publish("ops", "plain notice", "body")
        cand = _cand(ctx, sent["id"])
        assert cand["delivery"]["state"] == "sent"
        assert cand["delivery"]["notification_id"] == sent["id"]
        assert cand["delivery"]["channel"] == "telegram"

        digest = notifier.publish(
            "immune", "gate note", "batched",
            alert_hints={"notice": True, "subject": "gate"})
        cand = _cand(ctx, digest["id"])
        assert cand["delivery"]["state"] == "digest"

        tick_meta = {"subject": "watch:x", "prev_revision": "r1",
                     "current_revision": "r1"}
        tick = notifier.publish("watch", "tick", "same",
                                output_meta=dict(tick_meta))
        assert tick["suppressed"] is True
        rows = notifier.candidate_journal(limit=10)
        tick_row = next(r for r in rows
                        if r["candidate"]["subject"] == "watch:x")
        assert tick_row["delivery"]["state"] == "suppressed"
        assert tick_row["delivery"]["notification_id"] == ""

        notifier.publish("ops", "dup notice", "body")
        dup = notifier.publish("ops", "dup notice", "body")
        assert dup["deduped"] is True
        rows = notifier.candidate_journal(limit=10)
        dup_rows = [r for r in rows
                    if r["candidate"]["title"] == "dup notice"]
        assert dup_rows[0]["delivery"]["state"] == "deduped"

    def test_disabled_gate_ack(self):
        gateway = FakeGateway(live={"telegram"})
        context, tmp = make_ctx(gateway, proactive_enabled=False)
        try:
            notifier = Notifier(context, gateway)
            res = notifier.publish("briefing", "morning", "news")
            assert res["delivery_state"] == "disabled"
            cand = _cand(context, res["id"])
            assert cand["delivery"]["state"] == "disabled"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_pending_then_sent_release_ack(self, ctx):
        notifier = Notifier(ctx)  # no gateway at all: parked pending
        res = notifier.publish("ops", "parked", "body")
        assert res["delivery_state"] == "pending"
        cand = _cand(ctx, res["id"])
        assert cand["delivery"]["state"] == "pending"
        gateway = FakeGateway(live={"telegram"})
        live = Notifier(ctx, gateway)
        assert live.redeliver() == 1
        cand = _cand(ctx, res["id"])
        assert cand["delivery"]["state"] == "sent"
        (release,) = cand["releases"]
        assert release["ack"] == "sent"

    def test_dead_letter_terminal_ack(self, ctx):
        gateway = FakeGateway(live=())
        notifier = Notifier(ctx, gateway)
        res = notifier.publish("ops", "doomed", "body")
        assert res["delivery_state"] == STATE_FAILED
        with ctx.db.transaction():
            ctx.db.execute(
                "UPDATE notifications SET retry_count = ?, "
                "next_retry_at = 0 WHERE id = ?",
                (7, res["id"]))
        assert notifier.redeliver() == 0
        row = _row(ctx, res["id"])
        assert row["delivery_state"] == STATE_DEAD
        cand = _cand(ctx, res["id"])
        assert cand["delivery"]["state"] == "dead"
        (release,) = cand["releases"]
        assert release["ack"] == "dead"
        # a subject-less legacy candidate: no freshness verdict
        # could honestly be given — the record says so, with age
        assert release["verdict"] == "unverifiable"
        assert release["age_s"] is not None

    def test_release_age_is_recorded(self, ctx):
        gateway = FakeGateway(live=())
        notifier = Notifier(ctx, gateway)
        res = notifier.publish("job", "render-7 waiting", "at r1",
                               output_meta=dict(_WAITING))
        assert res["delivery_state"] == STATE_FAILED
        # age the fact two hours in the journal (white-box: the
        # candidate's own event_time is the age anchor)
        rows = AlertPolicy._load_journal(ctx.db, CANDIDATE_JOURNAL_KEY)
        rows[-1]["candidate"]["event_time"] -= 7200.0
        AlertPolicy._save_journal(
            ctx.db, rows, time.time(), DEFAULT_RECURRENCE_WINDOW_S,
            key=CANDIDATE_JOURNAL_KEY, max_rows=500)
        gateway.live = {"telegram"}
        assert notifier.redeliver() == 1
        cand = _cand(ctx, res["id"])
        (release,) = cand["releases"]
        assert release["verdict"] == "fresh"
        assert release["age_s"] >= 7000.0


# ── the digest fold is a release seam ────────────────────────────────


class TestDigestFoldRecheck:
    def _digest_item(self, notifier):
        res = notifier.publish(
            "job", "patrol waiting", "still waiting",
            alert_hints={"notice": True, "subject": "job:patrol"})
        assert res["tier"] == "digest"
        return res

    def test_superseded_digest_item_never_folds(self, ctx):
        gateway = FakeGateway(live={"telegram"})
        notifier = Notifier(ctx, gateway)
        item = self._digest_item(notifier)
        done = notifier.publish(
            "job", "patrol done", "finished",
            alert_hints={"resolved": True, "subject": "job:patrol",
                         "incident_key": "p1"})
        # the fold carries the closure, never the stale waiting
        # text it superseded (doc §5: replace stale "waiting" with
        # the actual completion/closure)
        unfolded = notifier.digest_unfolded()
        assert [r["id"] for r in unfolded] == [done["id"]]
        row = _row(ctx, item["id"])
        assert row["delivery_state"] == STATE_SUPERSEDED
        assert [r["id"] for r in notifier.digest_pending()] == \
            [done["id"]]
        cand = _cand(ctx, item["id"])
        (release,) = cand["releases"]
        assert release["path"] == "digest-fold"
        assert release["verdict"] == "superseded"
        assert release["ack"] == "superseded"

    def test_all_superseded_fold_is_empty(self, ctx):
        gateway = FakeGateway(live={"telegram"})
        notifier = Notifier(ctx, gateway)
        item = self._digest_item(notifier)
        # the completion arrives as a paged fact, so the digest
        # queue holds ONLY the stale waiting item
        done = notifier.publish(
            "job", "patrol done", "finished",
            output_meta={"subject": "job:patrol",
                         "prev_revision": "r0",
                         "current_revision": "r1",
                         "outcome": "succeeded",
                         "result_refs": ["artifact:p1"]})
        assert done["delivered"] is True
        # nothing folds → no digest section → nothing is sent
        assert notifier.digest_unfolded() == []
        assert _row(ctx, item["id"])["delivery_state"] == \
            STATE_SUPERSEDED

    def test_fresh_digest_item_folds_with_ack(self, ctx):
        gateway = FakeGateway(live={"telegram"})
        notifier = Notifier(ctx, gateway)
        item = self._digest_item(notifier)
        unfolded = notifier.digest_unfolded()
        assert [r["id"] for r in unfolded] == [item["id"]]
        assert notifier.mark_digest_folded([item["id"]]) == 1
        cand = _cand(ctx, item["id"])
        assert cand["delivery"]["state"] == "digest-delivered"
        (release,) = cand["releases"]
        assert release["path"] == "digest-fold"
        assert release["verdict"] == "fresh"
        assert release["ack"] == "digest-delivered"


# ── the named bound: unlinked rows release as before ─────────────────


class TestUnlinkedBound:
    def test_direct_store_held_row_releases_unverdicted(self, ctx):
        gateway = FakeGateway(live={"telegram"})
        notifier = Notifier(ctx, gateway)
        # missions progress's shape exactly: a direct store, no
        # publish, no candidate — no recheck source exists
        res = notifier._store(
            "mission", "mission update: X — 40%", "half way", 0,
            delivery_state=STATE_HELD_QUIET)
        assert notifier.redeliver() == 1
        assert len(gateway.sent) == 1
        assert _row(ctx, res["id"])["delivery_state"] == STATE_SENT
        # and no verdict was fabricated for it
        assert notifier.candidate_journal() == []


# ── units: deadline parsing + supersession rules ─────────────────────


class TestDeadlineParsing:
    def test_shapes(self):
        assert parse_deadline(1700000000) == 1700000000.0
        assert parse_deadline("1700000000") == 1700000000.0
        assert parse_deadline("2020-01-01T00:00:00+00:00") == \
            1577836800.0
        assert parse_deadline("2020-01-01T00:00:00Z") == 1577836800.0
        # naive ISO names no clock: refused, not guessed
        assert parse_deadline("2020-01-01T00:00:00") is None
        assert parse_deadline("next Friday") is None
        assert parse_deadline("") is None
        assert parse_deadline(None) is None


def _journal_row(db, subject, rev="", outcome="", ts=0.0, event="e"):
    cand = {"subject": subject, "current_revision": rev,
            "outcome": outcome, "event_time": ts}
    AlertPolicy.record_candidate(
        db, dict(cand, kind="job", event_id=event), now=ts)
    rows = AlertPolicy._load_journal(db, CANDIDATE_JOURNAL_KEY)
    return rows[-1]


class TestSupersessionRules:
    def test_outcome_change_supersedes(self, ctx):
        a = _journal_row(ctx.db, "s", rev="r1", ts=100.0, event="a")
        _journal_row(ctx.db, "s", rev="r1", outcome="succeeded",
                     ts=200.0, event="b")
        sup = AlertPolicy.superseding_record(ctx.db, a)
        assert sup is not None and sup["event_id"] == "b"

    def test_progress_replaces_progress(self, ctx):
        a = _journal_row(ctx.db, "s", rev="r1", ts=100.0, event="a")
        _journal_row(ctx.db, "s", rev="r2", ts=200.0, event="b")
        sup = AlertPolicy.superseding_record(ctx.db, a)
        assert sup is not None and sup["current_revision"] == "r2"

    def test_no_downgrade_of_a_recorded_result(self, ctx):
        a = _journal_row(ctx.db, "s", rev="r2", outcome="succeeded",
                         ts=100.0, event="a")
        _journal_row(ctx.db, "s", rev="r1", ts=200.0, event="b")
        assert AlertPolicy.superseding_record(ctx.db, a) is None

    def test_same_state_never_supersedes(self, ctx):
        a = _journal_row(ctx.db, "s", rev="r1", outcome="failed",
                         ts=100.0, event="a")
        _journal_row(ctx.db, "s", rev="r1", outcome="failed",
                     ts=200.0, event="b")
        assert AlertPolicy.superseding_record(ctx.db, a) is None

    def test_subjectless_has_no_recheck_source(self, ctx):
        a = _journal_row(ctx.db, "", rev="r1", ts=100.0, event="a")
        _journal_row(ctx.db, "", rev="r2", ts=200.0, event="b")
        assert AlertPolicy.superseding_record(ctx.db, a) is None
