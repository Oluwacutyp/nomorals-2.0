"""Doc 15 Pass 1 — producer inventory + ReportCandidate (observe mode).

Pinned here:

- the producer census is EXACT: a mechanical AST scan of every
  publish-shaped call site in the package matches the inventory in
  :mod:`nomorals.agents.notify_producers` — a planted unregistered
  producer fails the check, stale rows fail it, posture drift fails
  it;
- the legacy adapter resolves what a publish call genuinely carries
  (subject / incident / outcome / urgency / output_meta evidence)
  and MARKS the rest missing — a fully-evidenced call produces a
  complete candidate; a bare call produces a gapped candidate, not
  an exception and not a silent pass;
- OBSERVE mode changes nothing the owner can see: a battery of
  representative publishes (plain, deduped, forced, critical,
  DIGEST-tier, LOG-tier, PAGE-tier, Doc) delivers byte-identically
  with the adapter on vs off — same gateway sends, same stored
  rows, same publish() results;
- the candidate journal records hinted AND unhinted publishes
  (before Pass 1 only hinted alerts left any record at all), and
  coverage is measurable from it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents import notify_producers as census
from nomorals.agents.notifier import Notifier, notify
from nomorals.agents.report_candidate import build_legacy_candidate

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── harness (the test_pass7 shape) ───────────────────────────────────


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


def make_ctx(gateway=None):
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="d15p1-")
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=22,
        quiet_end=8,
        timezone="UTC",
    )
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


def _notification_rows(context):
    rows = context.db.query(
        "SELECT kind, title, body, delivered, delivery_state, channel, "
        "doc_json FROM notifications ORDER BY rowid")
    return [dict(r) for r in rows]


# ── (a) the inventory is complete and mechanically pinned ────────────


class TestProducerInventory:
    def test_inventory_is_exact_over_the_repo(self):
        assert census.validate_inventory(REPO_ROOT) == []

    def test_census_counts(self):
        summary = census.summarize()
        # the census reproduced from code on current main: 135
        # publish-shaped call sites = 65 producers + 70 excluded.
        assert summary["producers"] == 65
        assert summary["excluded"] == 70
        assert summary["by_posture"] == {
            "hinted": 30, "unhinted": 28, "forwarding": 7}
        # the bypass is the unhinted + hints-dependent-forwarding set
        assert summary["by_class"]["mission"] == 6
        assert summary["by_class"]["internal"] == 3

    def test_scanner_finds_a_planted_producer(self, tmp_path):
        package = tmp_path / "nomorals"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "planted.py").write_text(
            "from nomorals.agents.notifier import Notifier, notify\n"
            "\n"
            "def go(ctx):\n"
            "    n = Notifier(ctx)\n"
            "    n.publish('planted', 't', 'b')\n"
            "    notify(ctx, 'planted-helper', 't', 'b')\n"
            "    bus.publish('not-the-notifier')\n"
            "\n"
            "def callback_shaped(notify):\n"
            "    notify('a parameter, not the helper')\n",
            encoding="utf-8")
        sites = census.scan_publish_sites(tmp_path)
        # two producer-shaped calls + the bus-shaped attribute call;
        # the notify *parameter* is not import-bound and never scans.
        assert [(s.qualname, s.method) for s in sites] == [
            ("go", "publish"), ("go", "notify"), ("go", "publish")]
        unregistered = census.unregistered_sites(tmp_path)
        assert {s.key for s in unregistered} == {s.key for s in sites}
        assert census.validate_inventory(tmp_path) != []


# ── (b) the legacy adapter: resolution + missing-evidence marks ──────


class TestLegacyAdapter:
    def test_fully_evidenced_call_produces_a_complete_candidate(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        notifier.publish(
            "mission", "Render failed", "the render died at frame 9",
            critical=True, channels=["telegram"],
            alert_subject="mission render-7",
            alert_hints={"failed": True, "subject": "mission render-7",
                         "incident_key": "render-7"},
            output_meta={
                "attempt": "attempt-3",
                "current_revision": "rev-12",
                "impact": "the owner's video is not delivered",
                "uncertainty": "remote effect not yet read back",
                "deadline": "tonight",
                "next_actions": ["open control"],
                "evidence_refs": ["run:42"],
            })
        rows = notifier.candidate_journal()
        assert len(rows) == 1
        cand = rows[0]["candidate"]
        assert rows[0]["complete"] is True
        assert cand["evidence_gaps"] == []
        assert cand["subject"] == "mission render-7"
        assert cand["source_id"] == "mission render-7"
        assert cand["attempt"] == "attempt-3"
        assert cand["current_revision"] == "rev-12"
        assert cand["outcome"] == "failed"
        assert cand["action_needed"] is True
        assert cand["urgency"] == "critical"
        assert cand["channel_policy"] == "channels:telegram"
        assert cand["change_evidence"] == "incident render-7 opened"
        assert "incident:render-7" in cand["evidence_refs"]
        assert cand["report_class"] == "proactive"
        assert cand["sensitivity"] == "owner"
        assert cand["legacy"] is True
        # the producer resolves through the stack to this test module
        assert cand["producer"].endswith("test_d15p1_report_candidates")

    def test_bare_call_marks_missing_evidence(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        result = notifier.publish("housekeeping", "a title", "a body")
        assert result["delivered"] is True  # publish proceeds as before
        rows = notifier.candidate_journal()
        assert len(rows) == 1
        cand = rows[0]["candidate"]
        assert rows[0]["complete"] is False
        for gap in ("subject", "source_id", "attempt", "revision",
                    "change_evidence", "outcome", "impact",
                    "uncertainty", "action_needed", "urgency",
                    "evidence_refs"):
            assert gap in cand["evidence_gaps"], gap
        # what the call DID carry is resolved, not marked missing
        assert cand["kind"] == "housekeeping"
        assert cand["body_present"] is True
        assert cand["channel_policy"] == "all-owner-channels"
        assert cand["observed_at"] == cand["event_time"]

    def test_adapter_directly_with_schedule_meta(self):
        cand = build_legacy_candidate(
            kind="schedule", title="✅ scheduled: memory-consolidation",
            body="done", output_kind="scheduled",
            output_meta={"job": "memory-consolidation"},
            producer="nomorals.agents.scheduler", now=1000.0)
        assert cand.producer_class == "job"
        assert cand.source_kind == "job"
        assert cand.source_id == "job:memory-consolidation"
        assert "source_id" not in cand.evidence_gaps

    def test_helper_hop_resolves_the_real_producer(self, ctx):
        gateway = FakeGateway()
        # notify() builds its own Notifier inside notifier.py — the
        # candidate's producer must still be THIS module, not the
        # notifier's plumbing.
        notify(ctx, "ops", "via the helper", "body")
        rows = Notifier(ctx, gateway).candidate_journal()
        assert len(rows) == 1
        assert rows[0]["candidate"]["producer"].endswith(
            "test_d15p1_report_candidates")


# ── (c) observe mode: delivery is byte-identical on vs off ───────────


def _battery(observe: bool):
    gateway = FakeGateway()
    context, tmp = make_ctx(gateway)
    notifier = Notifier(context, gateway)
    notifier.observe_candidates = observe
    from nomorals.social.chat.present import Doc

    results = []
    results.append(notifier.publish("ops", "Deploy finished", "v2 live"))
    results.append(notifier.publish("ops", "Deploy finished", "v2 live"))
    results.append(notifier.publish(
        "ops", "Deploy finished", "v2 live", force=True))
    results.append(notifier.publish(
        "immune", "gate tripped", "throttle engaged", critical=True))
    results.append(notifier.publish(
        "immune", "gate note", "worth knowing",
        alert_hints={"notice": True, "subject": "gate"}))
    results.append(notifier.publish(
        "immune", "gate self-healed", "handled internally",
        alert_hints={"self_resolving": True, "subject": "gate",
                     "incident_key": "g1"}))
    results.append(notifier.publish(
        "immune", "gate failed", "containment failed",
        alert_hints={"failed": True, "subject": "gate"}))
    results.append(notifier.publish_doc(
        "ops", Doc(title="DocTitle", summary="a doc body")))
    def _strip(value):
        # delivery outcomes must match exactly; row ids and attempt
        # wall-clock stamps are run-local by definition.
        if isinstance(value, dict):
            return {k: _strip(v) for k, v in value.items()
                    if k not in ("id", "attempted_at")}
        if isinstance(value, list):
            return [_strip(v) for v in value]
        return value

    stripped = [_strip(r) for r in results]
    outcome = (list(gateway.sent), _notification_rows(context), stripped)
    candidates = notifier.candidate_journal(limit=100)
    shutil.rmtree(tmp, ignore_errors=True)
    return outcome, candidates


class TestObserveModeIsInvisible:
    def test_delivery_identical_with_adapter_on_vs_off(self):
        off_outcome, off_candidates = _battery(observe=False)
        on_outcome, on_candidates = _battery(observe=True)
        assert on_outcome == off_outcome
        assert off_candidates == []
        # one candidate per publish call — including the deduped
        # one and the DIGEST/LOG-tier ones that never deliver.
        assert len(on_candidates) == 8


# ── (d) the journal covers hinted AND unhinted; coverage measures ────


class TestCandidateJournal:
    def test_hinted_and_unhinted_both_recorded(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        notifier.publish("ops", "plain unhinted", "body")
        notifier.publish(
            "ops", "classified", "body",
            alert_hints={"notice": True, "subject": "thing"})
        candidates = notifier.candidate_journal()
        assert len(candidates) == 2
        assert {c["kind"] for c in candidates} == {"ops"}
        # the decision journal stays what it was: hinted only
        decisions = notifier.alert_journal()
        assert len(decisions) == 1
        assert decisions[0]["subject"] == "thing"

    def test_coverage_measures_the_population(self, ctx):
        gateway = FakeGateway()
        notifier = Notifier(ctx, gateway)
        notifier.publish("ops", "one", "body")
        notifier.publish("ops", "two", "body")
        notifier.publish(
            "ops", "three", "body",
            alert_hints={"notice": True, "subject": "thing"})
        coverage = notifier.candidate_coverage()
        assert coverage["candidates"] == 3
        assert coverage["complete"] == 0
        assert coverage["with_gaps"] == 3
        # every bare/legacy candidate in this harness misses outcome
        assert coverage["gap_fields"]["outcome"] == 3
        assert coverage["by_class"] == {"(unregistered)": 3}

    def test_candidates_survive_a_fresh_notifier(self, ctx):
        gateway = FakeGateway()
        Notifier(ctx, gateway).publish("ops", "persisted", "body")
        rows = Notifier(ctx, gateway).candidate_journal()
        assert len(rows) == 1
        assert rows[0]["candidate"]["title"] == "persisted"
