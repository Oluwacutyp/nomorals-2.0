"""Doc 15 Pass 2 — the mandatory decision path + semantic dedup.

Pinned here:

- acceptance scenario 1 through the REAL publish path: one hundred
  unchanged ticks (a subject re-asserting its standing revision)
  produce ZERO unsolicited pushes and one hundred journaled
  suppressions — and an urgency flag cannot make a tick news;
- a real semantic transition pages exactly once; re-publishing the
  same transition under a reworded title is deduped on its semantic
  identity, not its words; the next transition pages again;
- two different operations sharing the title "Render completed"
  both page — the legacy (kind, title) choke point no longer
  collapses independent results that carry a semantic identity;
- a candidate asserting a fact with material evidence missing is
  HELD to the digest tier with the gaps named (never paged as if
  complete, never silently dropped); the same candidate completed
  pages;
- owner-visible semantics preserved: critical bypasses the
  proactive gates but NOT the decision or dedupe; force remains
  the only dedupe escape; DIGEST rows are read on /notify and
  never pushed or redelivered;
- the hinted golden battery delivers identically under the gate;
- decision-journal coverage: in a mixed battery, decision rows ==
  publishes, each carrying the candidate identity + verdict +
  reason; identity-less publishes are journaled as legacy-fallback.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from nomorals.agents.alert_policy import (
    TIER_DEDUPED,
    TIER_DIGEST,
    TIER_LOG,
    TIER_PAGE,
    TIER_SUPPRESS,
    AlertPolicy,
)
from nomorals.agents.notifier import Notifier
from nomorals.agents.report_candidate import (
    build_legacy_candidate,
    is_unchanged_tick,
    semantic_identity,
)


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


def make_ctx(gateway=None, **partner_over):
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="d15p2-")
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


def _rows(context, state=""):
    sql = "SELECT kind, title, delivered, delivery_state FROM notifications"
    params: tuple = ()
    if state:
        sql += " WHERE delivery_state = ?"
        params = (state,)
    return [dict(r) for r in context.db.query(sql + " ORDER BY rowid", params)]


# ── (a) scenario 1: unchanged ticks → zero pushes, all journaled ─────


class TestUnchangedTicks:
    def test_100_declared_ticks_zero_pushes(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        meta = {"subject": "watch:example.com",
                "prev_revision": "hash-a", "current_revision": "hash-a"}
        for _ in range(100):
            res = notifier.publish(
                "watch", "example.com — no change", "still hash-a",
                output_meta=dict(meta))
            assert res["suppressed"] is True
            assert res["tier"] == TIER_SUPPRESS
            assert res["delivered"] is False
        assert gateway.sent == []
        assert _rows(ctx) == []  # suppressed ticks store no row
        decisions = notifier.decision_journal(limit=200,
                                              verdict=TIER_SUPPRESS)
        assert len(decisions) == 100
        assert all("unchanged tick" in d["reason"] for d in decisions)
        assert all(d["event_id"] for d in decisions)
        coverage = notifier.candidate_coverage()
        assert coverage["candidates"] == 100
        assert coverage["decisions"] == {TIER_SUPPRESS: 100}
        assert coverage["suppressed"] == 100

    def test_standing_state_assertion_is_a_tick(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        meta = {"subject": "job:heartbeat", "current_revision": "rev-7"}
        for _ in range(3):
            res = notifier.publish(
                "job", "job is at rev-7", "standing state",
                output_meta=dict(meta))
            assert res["suppressed"] is True
        assert gateway.sent == []

    def test_critical_cannot_make_a_tick_news(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        res = notifier.publish(
            "watch", "unchanged but shouted", "same revision",
            critical=True,
            output_meta={"subject": "watch:x",
                         "prev_revision": "r1",
                         "current_revision": "r1"})
        assert res["suppressed"] is True
        assert gateway.sent == []

    def test_outcome_or_action_lifts_the_tick_shape(self):
        cand = build_legacy_candidate(
            kind="watch", title="t",
            output_meta={"subject": "s", "prev_revision": "r1",
                         "current_revision": "r1",
                         "outcome": "failed"},
            producer="nomorals.agents.watchers", now=1.0)
        assert is_unchanged_tick(cand) is False
        cand2 = build_legacy_candidate(
            kind="watch", title="t",
            output_meta={"subject": "s", "prev_revision": "r1",
                         "current_revision": "r1"},
            producer="nomorals.agents.watchers", now=1.0)
        assert is_unchanged_tick(cand2) is True


# ── (b)+(c) semantic transitions + semantic dedup ────────────────────


_DEPLOY_META = {"subject": "deploy:app", "prev_revision": "rev-1",
                "current_revision": "rev-2", "outcome": "succeeded",
                "result_refs": ["artifact:build-9"]}


class TestSemanticDedup:
    def test_transition_pages_once_reworded_replay_deduped(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        first = notifier.publish(
            "deploy", "Deploy finished", "v2 is live",
            output_meta=dict(_DEPLOY_META))
        assert first["delivered"] is True
        assert len(gateway.sent) == 1
        # the fact page names its subject (the decision line rides)
        assert "deploy:app" in gateway.sent[0][2]
        assert "alert policy" in gateway.sent[0][2]

        replay = notifier.publish(
            "deploy", "The application deployment has completed",
            "completely different words, same event",
            output_meta=dict(_DEPLOY_META))
        assert replay["deduped"] is True
        assert replay["delivered"] is False
        assert len(gateway.sent) == 1  # still exactly one page
        decisions = notifier.decision_journal(limit=10,
                                              verdict=TIER_DEDUPED)
        assert len(decisions) == 1
        assert decisions[0]["dedupe_mode"] == "semantic"
        assert "deploy:app" in decisions[0]["identity"]

        nxt = dict(_DEPLOY_META, prev_revision="rev-2",
                   current_revision="rev-3")
        third = notifier.publish(
            "deploy", "Deploy finished", "v3 is live",
            output_meta=nxt)
        assert third["delivered"] is True
        assert len(gateway.sent) == 2

    def test_same_title_different_subjects_both_page(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        for who in ("mission:render-a", "mission:render-b"):
            res = notifier.publish(
                "mission", "Render completed", "done",
                alert_subject=who,
                output_meta={"attempt": "attempt-1",
                             "prev_revision": "r0",
                             "current_revision": "r1",
                             "outcome": "succeeded",
                             "result_refs": [f"artifact:{who}"]})
            assert res["delivered"] is True, who
        # the legacy (kind, title) test would have collapsed these
        assert len(gateway.sent) == 2
        assert len(_rows(ctx, "sent")) == 2

    def test_evidence_change_pages_once_per_asserted_change(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        meta = {"subject": "watch:gold",
                "change_evidence": "price crossed 4100"}
        first = notifier.publish("watch", "gold moved", "4100",
                                  output_meta=dict(meta))
        assert first["delivered"] is True
        replay = notifier.publish("watch", "gold update", "4100!",
                                   output_meta=dict(meta))
        assert replay["deduped"] is True
        moved = notifier.publish(
            "watch", "gold moved again", "4200",
            output_meta={"subject": "watch:gold",
                         "change_evidence": "price crossed 4200"})
        assert moved["delivered"] is True
        assert len(gateway.sent) == 2

    def test_hinted_replay_deduped_by_identity_despite_rewording(
            self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        meta = {"subject": "deploy:app", "prev_revision": "rev-1",
                "current_revision": "rev-2"}
        hints = {"subject": "deploy:app", "failed": True}
        first = notifier.publish(
            "deploy", "Deploy failed", "boom",
            alert_hints=dict(hints), output_meta=dict(meta))
        assert first["delivered"] is True
        replay = notifier.publish(
            "deploy", "The deployment has failed", "boom, reworded",
            alert_hints=dict(hints), output_meta=dict(meta))
        assert replay["deduped"] is True
        assert len(gateway.sent) == 1

    def test_identity_shape(self):
        a = build_legacy_candidate(
            kind="mission", title="Render completed",
            output_meta=dict(_DEPLOY_META, attempt="attempt-1"),
            producer="nomorals.missions.progress", now=1.0)
        b = build_legacy_candidate(
            kind="mission", title="A totally different title",
            output_meta=dict(_DEPLOY_META, attempt="attempt-1"),
            producer="nomorals.missions.progress", now=2.0)
        # title is not identity; the semantic tuple is
        assert semantic_identity(a) == semantic_identity(b)
        assert semantic_identity(a).startswith("deploy:app|attempt-1|")
        c = build_legacy_candidate(
            kind="mission", title="Render completed",
            output_meta=dict(_DEPLOY_META, attempt="attempt-2"),
            producer="nomorals.missions.progress", now=3.0)
        assert semantic_identity(c) != semantic_identity(a)
        bare = build_legacy_candidate(
            kind="ops", title="plain", producer="", now=1.0)
        assert semantic_identity(bare) == ""


# ── (d) material gaps: held to digest, completed pages ───────────────


class TestMaterialGapHold:
    def test_result_without_refs_held_then_completed_pages(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        gapped = dict(_DEPLOY_META)
        del gapped["result_refs"]
        held = notifier.publish(
            "deploy", "Deploy finished", "v2 is live",
            output_meta=gapped)
        assert held["tier"] == TIER_DIGEST
        assert not held["delivered"]
        assert held["material_gaps"] == ["result_refs"]
        assert "result_refs" in held["tier_reason"]
        assert gateway.sent == []
        digest = notifier.digest_pending()
        assert len(digest) == 1
        assert notifier.pending() == []  # digest is never redelivered

        done = notifier.publish(
            "deploy", "Deploy finished", "v2 is live",
            output_meta=dict(_DEPLOY_META))
        assert done["delivered"] is True
        assert len(gateway.sent) == 1

    def test_subject_is_material_to_a_fact(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        held = notifier.publish(
            "deploy", "Something failed", "no subject anywhere",
            output_meta={"prev_revision": "r1",
                         "current_revision": "r2",
                         "outcome": "failed"})
        assert held["tier"] == TIER_DIGEST
        assert held["material_gaps"] == ["subject"]
        assert gateway.sent == []

    def test_action_need_without_actions_held(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        held = notifier.publish(
            "ops", "Deploy blocked", "needs a credential",
            output_meta={"subject": "ops:deploy",
                         "change_evidence": "blocked on credentials",
                         "action_needed": True})
        assert held["tier"] == TIER_DIGEST
        assert held["material_gaps"] == ["next_actions"]
        done = notifier.publish(
            "ops", "Deploy blocked", "needs a credential",
            output_meta={"subject": "ops:deploy",
                         "change_evidence": "blocked on credentials",
                         "action_needed": True,
                         "next_actions": ["provide credential"]})
        assert done["delivered"] is True

    def test_attempt_is_material_for_mission_and_job_classes(self):
        policy = AlertPolicy()
        cand = build_legacy_candidate(
            kind="mission", title="step done",
            output_meta={"subject": "mission:m1",
                         "prev_revision": "r1",
                         "current_revision": "r2",
                         "outcome": "succeeded",
                         "result_refs": ["artifact:x"]},
            producer="nomorals.missions.progress", now=1.0)
        assert cand.producer_class == "mission"
        decision = policy.decide_candidate(None, cand)
        assert decision.tier == TIER_DIGEST
        assert decision.material_gaps == ("attempt",)
        cand2 = build_legacy_candidate(
            kind="job", title="job done",
            output_meta={"subject": "job:j1", "attempt": "attempt-2",
                         "prev_revision": "r1",
                         "current_revision": "r2",
                         "outcome": "succeeded",
                         "result_refs": ["artifact:x"]},
            producer="nomorals.agents.scheduler", now=1.0)
        decision2 = policy.decide_candidate(None, cand2)
        assert decision2.tier == TIER_PAGE

    def test_impact_is_material_for_trading_class(self):
        policy = AlertPolicy()
        cand = build_legacy_candidate(
            kind="signal", title="signal fired",
            output_meta={"subject": "desk:xau", "attempt": "a1",
                         "current_revision": "rev-2",
                         "outcome": "succeeded",
                         "result_refs": ["artifact:s"]},
            producer="nomorals.agents.pumpdesk.calls", now=1.0)
        assert cand.producer_class == "trading"
        decision = policy.decide_candidate(None, cand)
        assert decision.tier == TIER_DIGEST
        assert decision.material_gaps == ("impact",)


# ── (e) preserved owner-visible semantics ────────────────────────────


class TestPreservedSemantics:
    def test_critical_bypasses_proactive_gate_but_not_dedupe(self):
        gateway = FakeGateway()
        context, tmp = make_ctx(gateway, proactive_enabled=False)
        try:
            notifier = Notifier(context, gateway)
            held = notifier.publish(
                "briefing", "morning briefing", "the news")
            assert held["delivery_state"] == "disabled"
            sent = notifier.publish(
                "briefing", "build failed", "critical body",
                critical=True)
            assert sent["delivered"] is True
            assert len(gateway.sent) == 1
            again = notifier.publish(
                "briefing", "build failed", "critical body",
                critical=True)
            assert again["deduped"] is True
            assert len(gateway.sent) == 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_force_is_the_only_dedupe_escape(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        notifier.publish("ops", "Deploy finished", "v2 live")
        dup = notifier.publish("ops", "Deploy finished", "v2 live")
        assert dup["deduped"] is True
        assert len(gateway.sent) == 1
        forced = notifier.publish(
            "ops", "Deploy finished", "v2 live", force=True)
        assert forced["delivered"] is True
        assert len(gateway.sent) == 2

    def test_digest_rows_are_read_never_pushed(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        res = notifier.publish(
            "immune", "baseline ready", "30 samples learned",
            alert_hints={"subject": "immune baseline", "notice": True})
        assert res["tier"] == TIER_DIGEST
        assert gateway.sent == []
        assert len(notifier.digest_pending()) == 1
        assert notifier.pending() == []
        assert notifier.redeliver() == 0
        assert gateway.sent == []


# ── (f) the hinted golden battery under the gate ─────────────────────


class TestHintedBatteryUnchanged:
    def test_hinted_trio_outcomes_identical(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        notice = notifier.publish(
            "immune", "gate note", "worth knowing",
            alert_hints={"notice": True, "subject": "gate"})
        assert notice["tier"] == TIER_DIGEST
        logged = notifier.publish(
            "immune", "gate self-healed", "handled internally",
            alert_hints={"self_resolving": True, "subject": "gate",
                         "incident_key": "g1"})
        assert logged["tier"] == TIER_LOG
        failed = notifier.publish(
            "immune", "gate failed", "containment failed",
            alert_hints={"failed": True, "subject": "gate"})
        assert failed["delivered"] is True
        assert len(gateway.sent) == 1
        assert "alert policy" in gateway.sent[0][2]
        states = [r["delivery_state"] for r in _rows(ctx)]
        assert states == ["digest", "logged", "sent"]


# ── (g) decision-journal coverage ────────────────────────────────────


class TestDecisionCoverage:
    def test_mixed_battery_rows_equal_publishes(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        publishes = 0

        def pub(*a, **k):
            nonlocal publishes
            publishes += 1
            return notifier.publish(*a, **k)

        pub("ops", "bare legacy notice", "one-shot")                 # page (legacy)
        pub("immune", "note", "b",
            alert_hints={"notice": True, "subject": "gate"})         # digest
        pub("watch", "tick", "b",
            output_meta={"subject": "watch:x", "prev_revision": "r1",
                         "current_revision": "r1"})                  # suppress
        pub("deploy", "Deploy finished", "v2",
            output_meta=dict(_DEPLOY_META))                         # page (fact)
        pub("deploy", "Deployment done (reworded)", "v2",
            output_meta=dict(_DEPLOY_META))                         # deduped
        gapped = dict(_DEPLOY_META, subject="deploy:other")
        del gapped["result_refs"]
        pub("deploy", "Deploy finished", "v2", output_meta=gapped)   # digest (hold)

        candidates = notifier.candidate_journal(limit=50)
        assert len(candidates) == publishes
        assert all("decision" in row for row in candidates)
        decisions = notifier.decision_journal(limit=50)
        assert len(decisions) == publishes
        verdicts = [d["verdict"] for d in reversed(decisions)]
        assert verdicts == [TIER_PAGE, TIER_DIGEST, TIER_SUPPRESS,
                            TIER_PAGE, TIER_DEDUPED, TIER_DIGEST]
        assert all(d["event_id"] and d["reason"] for d in decisions)
        modes = [d["dedupe_mode"] for d in reversed(decisions)]
        assert modes[0] == "legacy-fallback"   # the bare notice
        assert modes[3] == "semantic"          # the fact page
        coverage = notifier.candidate_coverage()
        assert coverage["legacy_fallback"] >= 1
        assert coverage["decisions"][TIER_SUPPRESS] == 1
        assert len(gateway.sent) == 2  # bare notice + fact page only
