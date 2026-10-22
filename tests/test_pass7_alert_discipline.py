"""Pass 7 — alert discipline: ONE paging policy for unsolicited alerts.

The owner's rule (2026-10-22): systems keep working exactly as they do
internally, but the owner is paged only for alerts worth his attention.
Pinned here:

- a first-instance self-resolving event (the immune gate enforcing
  and throttling, exactly the live spam case) is LOG — silent;
- recurrence of the same subject beyond the threshold is PAGE, and
  the page names the subject and states the recurrence;
- a closure inherits its opening's tier (quiet opens, quiet closes;
  paged opens, paged closes);
- failure / owner-actionable events PAGE on first instance;
- noteworthy non-actionable events are DIGEST — stored, batched,
  never pushed, never redelivered;
- alerts published without classification keep the historical
  behaviour exactly (the genuinely actionable paths still page);
- every decision is journaled (kind, subject, tier, reason);
- immune enforcement behaviour itself is unchanged (the gate still
  enforces and quarantines — only the paging is tiered).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from types import SimpleNamespace

import pytest

from nomorals.agents.alert_policy import (
    TIER_DIGEST,
    TIER_LOG,
    TIER_PAGE,
    AlertHints,
    AlertPolicy,
)
from nomorals.agents.notifier import Notifier
from nomorals.immune.wiring import (
    IMMUNE_SAMPLER_TOOL,
    ImmuneMonitor,
    default_store_path,
    reset_shared_monitors,
    shared_monitor,
    summarize_tick_event,
)

NORMAL = {"processes_spawned": 100.0, "process_growth": 1.0}
RUNAWAY = {"processes_spawned": 5000.0, "process_growth": 50.0}
GATE_SUBJECT = "host process population (adaptive cancer gate)"


@pytest.fixture(autouse=True)
def _clean_monitors():
    reset_shared_monitors()
    yield
    reset_shared_monitors()


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


def make_partner(**kw):
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=22,
        quiet_end=8,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_ctx(test=None, partner=None, gateway=None):
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="pass7-alerts-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner or make_partner(),
        resolve=lambda rel: os.path.join(tmp, rel))
    ctx = SimpleNamespace(
        db=db, settings=settings,
        extras={"gateway": gateway} if gateway else {})
    return ctx, tmp


def _rows(ctx, state=None):
    if state:
        return ctx.db.query(
            "SELECT * FROM notifications WHERE delivery_state = ? "
            "ORDER BY created_at", (state,))
    return ctx.db.query("SELECT * FROM notifications ORDER BY created_at")


# ── the policy itself ─────────────────────────────────────────────

class TestPolicyDecisions:
    def setup_method(self):
        self.ctx, self.tmp = make_ctx()
        self.addCleanup = None  # tmp cleanup via fixture-less mkdtemp
        self.db = self.ctx.db
        self.policy = AlertPolicy()

    def _housekeeping(self, n=1, **kw):
        hints = AlertHints(subject=GATE_SUBJECT, self_resolving=True,
                           incident_key="adaptive-gate", **kw)
        out = None
        for _ in range(n):
            out = self.policy.decide(self.db, kind="immune", hints=hints)
        return out

    def test_first_self_resolving_event_logs_silently(self):
        d = self._housekeeping()
        assert d.tier == TIER_LOG
        assert d.occurrences == 1
        assert not d.escalated

    def test_recurrence_pages_and_names_subject(self):
        self._housekeeping(2)
        d = self._housekeeping()
        assert d.tier == TIER_PAGE
        assert d.escalated and d.force
        assert d.occurrences == 3
        assert "recurrence" in d.reason
        assert GATE_SUBJECT in d.body_line()

    def test_recurrence_pages_once_per_window(self):
        self._housekeeping(3)          # third pages
        d = self._housekeeping()       # fourth: already paged
        assert d.tier == TIER_LOG
        assert "already paged" in d.reason

    def test_closure_inherits_quiet_opening(self):
        self._housekeeping()
        d = self.policy.decide(
            self.db, kind="immune",
            hints=AlertHints(subject=GATE_SUBJECT, resolved=True,
                             incident_key="adaptive-gate"))
        assert d.tier == TIER_LOG
        assert "inherits" in d.reason

    def test_closure_inherits_paged_opening(self):
        self._housekeeping(3)          # opening escalates to PAGE
        d = self.policy.decide(
            self.db, kind="immune",
            hints=AlertHints(subject=GATE_SUBJECT, resolved=True,
                             incident_key="adaptive-gate"))
        assert d.tier == TIER_PAGE

    def test_closure_without_opening_digests(self):
        d = self.policy.decide(
            self.db, kind="immune",
            hints=AlertHints(subject="x", resolved=True,
                             incident_key="never-opened"))
        assert d.tier == TIER_DIGEST

    def test_failure_pages_on_first_instance(self):
        d = self.policy.decide(
            self.db, kind="immune",
            hints=AlertHints(subject="immune sampler", failed=True))
        assert d.tier == TIER_PAGE and d.escalated and not d.force

    def test_actionable_pages_on_first_instance(self):
        d = self.policy.decide(
            self.db, kind="trial",
            hints=AlertHints(subject="trial signup", actionable=True))
        assert d.tier == TIER_PAGE

    def test_notice_digests(self):
        d = self.policy.decide(
            self.db, kind="immune",
            hints=AlertHints(subject="immune baseline", notice=True))
        assert d.tier == TIER_DIGEST

    def test_threshold_is_config_driven(self):
        ctx, _ = make_ctx(partner=make_partner(
            alert_recurrence_threshold=2))
        policy = AlertPolicy.from_context(ctx)
        assert policy.recurrence_threshold == 2
        hints = AlertHints(subject="s", self_resolving=True)
        assert policy.decide(ctx.db, kind="k", hints=hints).tier == TIER_LOG
        assert policy.decide(ctx.db, kind="k", hints=hints).tier == TIER_PAGE

    def test_decisions_are_journaled(self):
        self._housekeeping(2)
        self.policy.decide(
            self.db, kind="immune",
            hints=AlertHints(subject="immune sampler", failed=True))
        journal = AlertPolicy.journal(self.db, kind="immune")
        assert len(journal) == 3
        assert journal[0]["tier"] == TIER_PAGE  # newest first
        assert journal[0]["subject"] == "immune sampler"
        assert all(j["reason"] for j in journal)


# ── the notifier seam ─────────────────────────────────────────────

class TestPublishTiers:
    def setup_method(self):
        self.gateway = FakeGateway()
        self.ctx, self.tmp = make_ctx(gateway=self.gateway)
        self.notifier = Notifier(self.ctx)

    def test_log_tier_stores_silently(self):
        res = self.notifier.publish(
            "immune", "🛡️ immune gate ENFORCED", "gate enforced: x",
            alert_subject=GATE_SUBJECT,
            alert_hints={"subject": GATE_SUBJECT, "self_resolving": True,
                         "incident_key": "adaptive-gate"})
        assert res["tier"] == TIER_LOG
        assert self.gateway.sent == []
        rows = _rows(self.ctx, "logged")
        assert len(rows) == 1 and rows[0]["delivered"] == 0
        assert self.notifier.pending() == []  # never redelivered

    def test_digest_tier_batches_without_paging(self):
        res = self.notifier.publish(
            "immune", "🛡️ immune baseline ready", "30 samples learned",
            alert_hints={"subject": "immune baseline", "notice": True})
        assert res["tier"] == TIER_DIGEST
        assert self.gateway.sent == []
        digest = self.notifier.digest_pending()
        assert len(digest) == 1
        assert "baseline ready" in digest[0]["title"]
        assert self.notifier.pending() == []

    def test_failure_hint_pages_and_names_subject(self):
        res = self.notifier.publish(
            "immune", "❌ immune sampler error", "tick failed: boom",
            alert_hints={"subject": "immune sampler", "failed": True})
        assert res.get("delivered") is True
        assert len(self.gateway.sent) == 1
        text = self.gateway.sent[0][2]
        assert "immune sampler" in text
        assert "alert policy" in text  # the decision line rides along

    def test_recurrence_pages_through_publish(self):
        hints = {"subject": GATE_SUBJECT, "self_resolving": True,
                 "incident_key": "adaptive-gate"}
        for _ in range(2):
            self.notifier.publish(
                "immune", "🛡️ immune gate ENFORCED", "gate enforced",
                alert_subject=GATE_SUBJECT, alert_hints=hints)
        assert self.gateway.sent == []  # occurrences 1-2 silent
        self.notifier.publish(
            "immune", "🛡️ immune gate ENFORCED", "gate enforced",
            alert_subject=GATE_SUBJECT, alert_hints=hints)
        assert len(self.gateway.sent) == 1  # occurrence 3 pages
        text = self.gateway.sent[0][2]
        assert "recurrence" in text and GATE_SUBJECT in text

    def test_unclassified_alert_keeps_historical_behaviour(self):
        """No hints → no tiering: genuinely actionable paths (ops
        findings, trial waits, research results) page as before."""
        res = self.notifier.publish(
            "ops", "stuck mission — self-heal failed", "needs the owner")
        assert res.get("delivered") is True
        assert len(self.gateway.sent) == 1
        assert self.notifier.alert_journal() == []

    def test_dict_and_dataclass_hints_agree(self):
        a = self.notifier.publish(
            "immune", "t1", "b",
            alert_hints=AlertHints(subject="s", notice=True))
        b = self.notifier.publish(
            "immune", "t2", "b",
            alert_hints={"subject": "s", "notice": True})
        assert a["tier"] == b["tier"] == TIER_DIGEST


# ── the immune events carry their classification ──────────────────

class TestSummarizeClassification:
    def test_enforcement_event_is_self_resolving_housekeeping(self):
        event = summarize_tick_event(
            {"ok": True, "enforced": True, "quarantined": 1, "samples": 41,
             "sample": dict(RUNAWAY), "reason": "gate enforced"},
            ["fever started"])
        assert event["hints"]["self_resolving"] is True
        assert event["hints"]["incident_key"] == "adaptive-gate"
        assert "process population" in event["subject"]

    def test_closure_event_is_a_closure(self):
        event = summarize_tick_event(
            {"ok": True, "sampled": True, "learned": True,
             "quarantined": 4, "samples": 41},
            ["immune incident closed — Fever throttled 0 action(s)"])
        assert event["hints"]["resolved"] is True
        assert event["hints"]["incident_key"] == "adaptive-gate"

    def test_error_event_is_a_failure(self):
        event = summarize_tick_event({"ok": False, "error": "tick failed"})
        assert event["hints"]["failed"] is True

    def test_baseline_ready_event_is_a_notice(self):
        event = summarize_tick_event(
            {"ok": True, "sampled": True, "learned": True,
             "quarantined": 0, "samples": 30},
            ["immune baseline ready — 30 samples learned"])
        assert event["hints"]["notice"] is True


# ── end to end: the live spam scenario, replayed ─────────────────

def _capture_tools(ctx):
    from nomorals.tools import autonomy as autonomy_tools

    captured: dict = {}

    class _Registry:
        def __init__(self):
            self.context = ctx

        def register(self, name, fn=None, **kw):
            if fn is None:
                def _deco(f):
                    captured[name] = f
                    return f
                return _deco
            captured[name] = fn
            return fn

    autonomy_tools.register(_Registry())
    return captured


def _immune_world(tmp_path):
    gateway = FakeGateway()
    ctx, _ = make_ctx(gateway=gateway)
    # settings.resolve must point at this test's tmp tree
    ctx.settings.resolve = lambda rel: os.path.join(tmp_path, rel)
    ctx.settings.workspace_dir = tmp_path
    captured = _capture_tools(ctx)
    box = {"fn": lambda: dict(NORMAL)}
    monitor = shared_monitor(
        default_store_path(ctx), collect=lambda baseline: box["fn"]())
    return ctx, gateway, captured, monitor, box


def test_immune_spam_scenario_replayed(tmp_path):
    """Two marginal enforcement+closure pairs in one morning paged
    four times.  Replayed through the real sampler tool: the pairs
    are silent; the third enforcement — recurrence — pages, naming
    the subject; its closure pages too (it inherits the paged tier).
    Enforcement itself is untouched: every runaway still enforces,
    every sample still quarantines."""
    ctx, gateway, captured, monitor, box = _immune_world(str(tmp_path))
    tool = captured[IMMUNE_SAMPLER_TOOL]

    # warm the baseline through the tool — the ready milestone is a
    # digest-tier notice: recorded, never pushed.
    for _ in range(40):
        tool()
    assert gateway.sent == []
    assert len(Notifier(ctx).digest_pending()) == 1

    def cycle():
        box["fn"] = lambda: dict(RUNAWAY)
        out = tool()
        assert out["enforced"] is True  # enforcement unchanged
        box["fn"] = lambda: dict(NORMAL)
        tool()  # the runaway stops — the incident closes

    cycle()  # incident 1: open LOG, close LOG
    cycle()  # incident 2: open LOG, close LOG
    assert gateway.sent == []
    logged = _rows(ctx, "logged")
    assert len(logged) == 4  # both pairs journaled as stored rows

    cycle()  # incident 3: recurrence pages; closure inherits PAGE
    assert len(gateway.sent) == 2
    page_text = gateway.sent[0][2]
    assert "ENFORCED" in page_text
    assert GATE_SUBJECT in page_text
    assert "recurrence" in page_text
    assert "incident closed" in gateway.sent[1][2]

    # the policy journal tells the whole story, in order
    journal = AlertPolicy.journal(ctx.db, kind="immune")
    tiers = [j["tier"] for j in reversed(journal)]
    assert tiers == [TIER_DIGEST, TIER_LOG, TIER_LOG, TIER_LOG,
                     TIER_LOG, TIER_PAGE, TIER_PAGE]


def test_immune_sampler_error_still_pages_first_time(tmp_path):
    ctx, gateway, captured, monitor, box = _immune_world(str(tmp_path))

    def _boom():
        raise RuntimeError("collect boom")

    box["fn"] = _boom
    tool = captured[IMMUNE_SAMPLER_TOOL]
    tool()
    assert len(gateway.sent) == 1
    assert "error" in gateway.sent[0][2]
    assert "collect boom" in gateway.sent[0][2]
