"""Formatting F5 — alert templates on the F2 presentation layer.

Pass 7 (``agents/alert_policy.py``) decides WHETHER an unsolicited
alert pages; F5 decides HOW it looks.  Every producer family that used
to hand-compose a publish body now builds a presentation ``Doc``
(``agents/alert_templates.py``) and publishes it under the EXACT
historical kind + title with the exact historical kwargs, so the
policy sees identical inputs.

Pinned here:

- immune events carry structured facts (spawned / growth /
  quarantined / baseline samples) as labeled rows — values from the
  tick result, never parsed back out of prose — and the prose
  context lines are not duplicated once rows exist;
- the watchers' hard ``body[:3500]`` mid-sentence cut is gone: long
  hit bodies and digests ride the Doc path complete;
- the upgrade-proposal ``rationale[:400]`` and the research-loop
  ``[:3000]``/``[:3200]`` body cuts are gone the same way;
- Pass-7 decision equivalence end-to-end: the same immune events
  through the real Notifier produce the same tiers (LOG, LOG, PAGE,
  closure inherits PAGE) with the decision line in the delivered text
  AND the stored body;
- kind/title/dedupe stability: publish-arg titles byte-identical to
  the pre-F5 strings, and a repeat publish dedupes exactly as before;
- the templates never raise on junk input — they degrade to a
  minimal truthful Doc.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from types import SimpleNamespace

import pytest

from nomorals.agents import alert_templates as T
from nomorals.agents.notifier import Notifier
from nomorals.immune.wiring import summarize_tick_event
from nomorals.social.chat.present import body_text, render

NORMAL = {"processes_spawned": 100.0, "process_growth": 1.0}
RUNAWAY = {"processes_spawned": 5000.0, "process_growth": 50.0}
GATE_SUBJECT = "host process population (adaptive cancer gate)"


# ── harness (mirrors tests/test_pass7_alert_discipline.py) ────────

class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.message_id = "m1"


class FakeGateway:
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

    tmp = tempfile.mkdtemp(prefix="f5-alerts-")
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


def _enforced_event():
    result = {"ok": True, "enforced": True,
              "reason": "process population beyond the adaptive "
                        "baseline (spawned 5000, growth 50)",
              "sample": dict(RUNAWAY), "quarantined": 1, "samples": 41}
    return summarize_tick_event(result, ["fever started"])


# ── immune: wiring facts/tone + template shape ────────────────────

class TestImmuneWiringFacts:
    def test_enforced_event_carries_facts_and_tone(self):
        ev = _enforced_event()
        # legacy keys byte-identical in shape
        assert ev["title"] == "🛡️ immune gate ENFORCED"
        assert ev["critical"] is True
        assert ev["subject"] == GATE_SUBJECT
        assert ev["hints"] == {"subject": GATE_SUBJECT,
                                "self_resolving": True,
                                "incident_key": "adaptive-gate"}
        # F5 additions: plain data only
        assert ev["tone"] == "error"
        assert ("Spawned", 5000.0) in ev["facts"]
        assert ("Growth", 50.0) in ev["facts"]
        assert ("Quarantined", 1) in ev["facts"]
        assert ("Baseline samples", 41) in ev["facts"]

    def test_unavailable_and_error_events(self):
        ev = summarize_tick_event(
            {"ok": False, "unavailable": True,
             "error": "POSIX sampler unavailable on this host"},
            ["immune sampler unavailable on this host"])
        assert ev["tone"] == "warn"
        assert ev["title"] == "🛡️ immune sampler unavailable on this host"
        ev2 = summarize_tick_event({"ok": False, "error": "boom"}, [])
        assert ev2["tone"] == "error" and ev2["facts"] == []

    def test_quiet_tick_stays_none(self):
        assert summarize_tick_event(
            {"ok": True, "sample": dict(NORMAL), "samples": 5}, []) is None


class TestImmuneTemplate:
    def test_enforced_doc_rows_no_prose_duplication(self):
        ev = _enforced_event()
        doc = T.immune_event_doc(ev)
        assert doc.tone == "error"
        assert doc.title == "immune gate ENFORCED"
        assert doc.icon == "🛡️"
        assert "adaptive baseline" in doc.summary
        rows = {kv.label: kv.value for s in doc.sections for kv in s.rows}
        assert rows["Spawned"] == 5000.0
        assert rows["Growth"] == 50.0
        assert rows["Quarantined"] == 1
        assert rows["Baseline samples"] == 41
        body = body_text(doc)
        # the structured facts ride as rows; the prose context lines
        # that composed them are NOT duplicated underneath
        assert "sample: spawned 5000" not in body
        assert "quarantined: 1 · baseline samples: 41" not in body
        assert "fever started" in body  # announcement prose kept
        text = render(doc, "telegram").text
        assert "Spawned" in text and "5000" in text

    def test_closure_doc_is_ok_tone(self):
        result = {"ok": True, "sample": dict(NORMAL),
                  "quarantined": 0, "samples": 44}
        ev = summarize_tick_event(
            result, ["immune incident closed — the runaway stopped"])
        doc = T.immune_event_doc(ev)
        assert doc.tone == "ok" and "incident closed" in doc.title

    def test_error_doc(self):
        ev = summarize_tick_event({"ok": False, "error": "tick failed: boom"}, [])
        doc = T.immune_event_doc(ev)
        assert doc.tone == "error"
        assert "tick failed: boom" in body_text(doc)

    def test_junk_event_degrades(self):
        doc = T.immune_event_doc({})
        assert doc is not None
        assert isinstance(body_text(doc), str)


# ── Pass-7 decision equivalence through the REAL Notifier ────────

class TestPass7Equivalence:
    def setup_method(self):
        self.gateway = FakeGateway()
        self.ctx, self.tmp = make_ctx(gateway=self.gateway)
        self.notifier = Notifier(self.ctx)

    def _publish_event(self, event):
        doc = T.immune_event_doc(event)
        return self.notifier.publish(
            "immune", event["title"], body_text(doc),
            critical=bool(event.get("critical")),
            alert_subject=str(event.get("subject") or ""),
            alert_hints=event.get("hints"), doc=doc)

    def test_tier_sequence_matches_string_path(self):
        # occurrences 1-2 LOG (silent), 3 PAGEs with the decision line
        ev = _enforced_event()
        self._publish_event(ev)
        self._publish_event(ev)
        assert self.gateway.sent == []
        res = self._publish_event(ev)
        assert res.get("delivered") is True
        assert len(self.gateway.sent) == 1
        text = self.gateway.sent[0][2]
        assert "ENFORCED" in text
        assert "recurrence" in text and GATE_SUBJECT in text
        # the stored body is the Doc's canonical body + decision line
        rows = self.ctx.db.query(
            "SELECT * FROM notifications WHERE kind = 'immune'")
        bodies = "\n".join(r["body"] for r in rows)
        assert "Spawned" in bodies and "alert policy" in bodies
        # closure inherits the paged opening
        closed = summarize_tick_event(
            {"ok": True, "sample": dict(NORMAL), "quarantined": 0,
             "samples": 44},
            ["immune incident closed — the runaway stopped"])
        res = self._publish_event(closed)
        assert res.get("delivered") is True
        assert "incident closed" in self.gateway.sent[-1][2]

    def test_journal_tiers_match_pass7_replay(self):
        ev = _enforced_event()
        for _ in range(3):
            self._publish_event(ev)
        closed = summarize_tick_event(
            {"ok": True, "sample": dict(NORMAL), "quarantined": 0,
             "samples": 44},
            ["immune incident closed — the runaway stopped"])
        self._publish_event(closed)
        tiers = [j["tier"] for j in self.notifier.alert_journal()]
        # the journal lists newest first; the decision MULTISET is the
        # invariant: two silent logs, the recurrence page, and the
        # closure page that inherits it — the Pass 7 replay exactly.
        assert sorted(tiers) == ["log", "log", "page", "page"]

    def test_dedupe_keys_on_exact_title(self):
        ev = summarize_tick_event({"ok": False, "error": "boom"}, [])
        self._publish_event(ev)
        self._publish_event(ev)
        rows = self.ctx.db.query(
            "SELECT * FROM notifications WHERE kind = 'immune'")
        assert len(rows) == 1  # second publish deduped on kind+title


# ── watchers ──────────────────────────────────────────────────────

class _RecordingNotifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body,
                           **kw})
        return {"delivered": True}


class TestWatcherTemplates:
    def _engine(self, tmp):
        from nomorals.agents.watchers import AlertEngine, WatcherStore

        ctx, _t = make_ctx()
        store = WatcherStore(ctx.db)
        rec = _RecordingNotifier()
        engine = AlertEngine(store, ctx, notifier=rec)
        return engine, rec

    def test_hit_doc_shape_and_no_length_cut(self, tmp_path):
        from nomorals.agents.watchers import Watcher, WatchResult

        engine, rec = self._engine(str(tmp_path))
        watcher = Watcher(id="w1", name="gold price", kind="price",
                          severity="urgent")
        long_summary = ("price moved " + "x" * 5000 + " TAILMARKER")
        engine.handle(watcher, WatchResult(
            changed=True, summary=long_summary), now=time.time())
        call = rec.calls[-1]
        assert call["kind"] == "watcher"
        assert call["title"] == f"👁 gold price: {long_summary[:120]}"
        assert call["critical"] is True
        assert call["doc"] is not None
        # the [:3500] cut is retired: the full summary rides the body
        assert "TAILMARKER" in call["body"]
        rows = {kv.label: kv.value for s in call["doc"].sections
                for kv in s.rows}
        assert rows["Watcher"] == "gold price"
        assert rows["Watcher id"] == "w1"
        assert rows["Severity"] == "urgent"

    def test_digest_carries_every_held_line(self, tmp_path):
        from nomorals.agents.watchers import Watcher

        engine, rec = self._engine(str(tmp_path))
        watcher = Watcher(id="w9", name="slow feed", kind="rss",
                          severity="info")
        now = time.time()
        # 20 held items ~210 chars each ≈ 4400 chars of digest —
        # beyond the retired [:3500] cut, within the audit store's
        # own per-title cap (record_alert stores title[:300], a
        # pre-existing storage shape F5 does not change).
        for i in range(20):
            title = f"held item {i} " + "y" * 180 + f" END{i}"
            engine._hold(watcher, title, "body", "digest", now)
        out = engine.flush_digest(now=now)
        assert out["action"] == "digested" and out["count"] == 20
        call = rec.calls[-1]
        assert call["title"] == "👁 watcher digest (20)"
        assert len(call["body"]) > 3500
        for i in range(20):
            assert f"END{i}" in call["body"]

    def test_digest_doc_builder_direct(self):
        doc = T.watcher_digest_doc(2, [("alpha", ["• [info] one"]),
                                       ("beta", ["• [info] two"])])
        assert doc.title == "watcher digest (2)"
        body = body_text(doc)
        assert "alpha" in body and "beta" in body

    def test_notice_doc(self):
        doc = T.watcher_notice_doc("watcher paused: flappy",
                                   "⏸ flappy: flapping (7 flips)")
        assert doc.tone == "warn"
        assert "flapping" in body_text(doc)


# ── ops alerts ────────────────────────────────────────────────────

class TestOpsTemplate:
    def test_stuck_finding_rows(self):
        # the producer's real shape: kind "stuck_mission", data = the
        # mission-health entry (completed is the finished-step list)
        finding = {"kind": "stuck_mission", "key": "m7",
                   "title": "⚠️ stuck mission — self-heal failed",
                   "body": "mission m7 looks stuck",
                   "data": {"id": "m7", "goal": "ship it",
                            "iterations": 42,
                            "completed": ["a", "b", "c"],
                            "last_error": "timeout"}}
        doc = T.ops_finding_doc(finding)
        rows = {kv.label: kv.value for s in doc.sections for kv in s.rows}
        assert rows["Goal"] == "ship it"
        assert rows["Iterations"] == 42
        assert rows["Steps done"] == 3
        assert rows["Last error"] == "timeout"
        assert doc.tone == "warn"

    def test_pivot_and_chronic_rows(self):
        pivot = {"kind": "unrecovered_pivot", "key": "p", "title": "t",
                 "body": "b",
                 "data": {"task": "code", "pivot": "try the repl",
                          "recovered": False}}
        doc = T.ops_finding_doc(pivot)
        rows = {kv.label: kv.value for s in doc.sections for kv in s.rows}
        assert rows["Task"] == "code"
        assert "Approach change" in [s.title for s in doc.sections]
        # the producer's real shape: kind "chronic_role", data =
        # Skill.to_dict() (name "route:<role>")
        chronic = {"kind": "chronic_role", "key": "c", "title": "t",
                   "body": "b",
                   "data": {"name": "route:brain", "success_count": 1,
                            "failure_count": 9, "success_rate": 0.1,
                            "last_used": 123.0}}
        rows = {kv.label: kv.value
                for s in T.ops_finding_doc(chronic).sections
                for kv in s.rows}
        assert rows["Failures"] == 9 and rows["Role"] == "brain"
        assert rows["Success rate"] == "10%"


# ── backup reports ────────────────────────────────────────────────

class _StrictNotifier:
    """github_backup-style fake: publish(kind, title, body, *, critical)."""

    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", *, critical=False, **kwargs):
        self.calls.append({"kind": kind, "title": title, "body": body,
                           "critical": critical, **kwargs})
        return {"delivered": True}


class TestBackupTemplates:
    def test_result_doc(self):
        from nomorals.backup.report import report_backup_result

        fake = _StrictNotifier()
        result = {"backup_id": "bk-7", "assets": ["a", "b"],
                  "total_encrypted_bytes": 2048, "tag": "backup-bk-7",
                  "manifest_pushed": True, "release_url": "https://x/r"}
        report_backup_result(fake, result)
        call = fake.calls[-1]
        assert call["kind"] == "backup"
        assert call["title"] == "backup shipped: bk-7"
        assert call["doc"] is not None
        assert "Assets" in call["body"] and "Encrypted bytes" in call["body"]

    def test_failure_doc_is_critical_error(self):
        from nomorals.backup.report import report_backup_failure

        fake = _StrictNotifier()
        report_backup_failure(fake, "bk-8", RuntimeError("disk full"))
        call = fake.calls[-1]
        assert call["title"] == "BACKUP FAILED: bk-8"
        assert call["critical"] is True
        assert call["doc"].tone == "error"
        assert "disk full" in call["body"]

    def test_queued_doc_warns(self):
        from nomorals.backup.report import report_queued

        fake = _StrictNotifier()
        report_queued(fake, {"backup_id": "bk-9", "reason": "offline"})
        call = fake.calls[-1]
        assert call["title"] == "backup queued (offline): bk-9"
        assert call["doc"].tone == "warn"


# ── research ──────────────────────────────────────────────────────

class TestResearchTemplates:
    def test_loop_doc_briefs_complete(self):
        brief = "finding: " + "z" * 4000 + " BRIEFEND"
        doc = T.research_loop_doc(
            title="research loop: 1 finding(s), 1 proposal(s)",
            briefs=[brief], findings=1, proposals=1, signals="quiet")
        body = body_text(doc)
        assert "BRIEFEND" in body  # no [:3000] cut
        assert "/proposals" in doc.footer

    def test_job_doc_all_sources(self):
        job = {"query": "gold outlook", "state": "done",
               "report": {"summary": "bullish",
                          "sources": [{"n": i, "title": f"s{i}",
                                       "url": f"https://x/{i}"}
                                      for i in range(1, 13)]}}
        doc = T.research_job_doc(job)
        assert doc.title == "Research ready: gold outlook"
        body = body_text(doc)
        assert "https://x/12" in body  # beyond the old [:8] cap

    def test_job_doc_failed(self):
        doc = T.research_job_doc({"query": "q", "state": "failed",
                                  "error": "timeout"})
        assert doc.tone == "error"
        assert doc.title == "Research failed: q"


# ── trial + task notices ──────────────────────────────────────────

class TestNoticeTemplates:
    def test_trial_tones(self):
        assert T.trial_notice_doc(
            "trial assist — acme", "waiting for your reply").tone == "info"
        assert T.trial_notice_doc(
            "trial assist — acme", "⏸ Paused at captcha").tone == "warn"
        assert T.trial_notice_doc(
            "trial assist — acme", "📩 code arrived: 123").tone == "ok"

    def test_trial_body_shape(self):
        doc = T.trial_notice_doc(
            "trial assist — acme", "first line summary\nsecond detail")
        assert doc.summary == "first line summary"
        assert "second detail" in body_text(doc)

    def test_task_tones(self):
        assert T.task_notice_doc("task done: x", "ok").tone == "ok"
        assert T.task_notice_doc("task failed: x", "boom").tone == "error"


# ── upgrade queue (end-to-end) ────────────────────────────────────

class TestUpgradeTemplates:
    def setup_method(self):
        self.gateway = FakeGateway()
        self.ctx, self.tmp = make_ctx(gateway=self.gateway)

    def _queue(self):
        from nomorals.agents.upgrade_queue import UpgradeQueue

        return UpgradeQueue(self.ctx)

    def _stored(self, kind="upgrade_proposal"):
        return self.ctx.db.query(
            "SELECT * FROM notifications WHERE kind = ? "
            "ORDER BY created_at", (kind,))

    def test_proposal_full_rationale_no_cut(self):
        q = self._queue()
        rationale = "because " + "r" * 900 + " RATIONALE_END"
        pid = q.propose(title="Better caching", rationale=rationale,
                        patch_plan={"edits": ["add an LRU cache"]},
                        files=["nomorals/agents/cache.py"],
                        tests=["tests/test_cache.py"],
                        claim_ids=["c1"], source="research")
        assert pid.startswith("upg_")
        rows = self._stored()
        assert rows and rows[-1]["title"] == "new upgrade proposal: Better caching"
        assert "RATIONALE_END" in rows[-1]["body"]  # no [:400] cut
        assert "nomorals/agents/cache.py" in rows[-1]["body"]

    def test_decisions_and_implementation_titles(self):
        q = self._queue()
        pid = q.propose(
            title="Tidy logs", patch_plan={"edits": ["trim log noise"]},
            files=["nomorals/agents/notifier.py"],
            tests=["quieter logs asserted"],
            rationale="Log output has grown noisy enough to bury real "
                      "failures; trimming the repetitive lines makes "
                      "incidents visible again during reviews.")
        q.approve(pid, by="owner")
        q.record_implemented(pid, {"ok": True,
                                   "evolution_proposal_id": "evo_1",
                                   "files_changed": 2})
        titles = [r["title"] for r in self._stored()]
        assert f"upgrade approved: Tidy logs" in titles
        assert f"upgrade implemented: {pid}" in titles
        body = [r["body"] for r in self._stored()
                if r["title"].startswith("upgrade implemented")][0]
        assert "evo_1" in body

    def test_deny(self):
        q = self._queue()
        pid = q.propose(
            title="Risky change", patch_plan={"edits": ["swap the core"]},
            files=["nomorals/core/engine.py"],
            tests=["core swap smoke test"],
            rationale="A risky structural change that the owner may "
                      "well refuse; filed so the deny path is covered "
                      "by the template tests end to end here.")
        q.deny(pid, by="owner", reason="not now")
        titles = [r["title"] for r in self._stored()]
        assert "upgrade denied: Risky change" in titles


# ── directives (end-to-end with a recording notifier) ─────────────

class TestDirectiveTemplates:
    def test_full_result_rides_the_doc(self, tmp_path):
        from nomorals.agents.directives import DirectivesAgent

        rec = _RecordingNotifier()
        ctx, _tmp = make_ctx()
        d = DirectivesAgent(ctx, notifier=rec)
        long_result = "done: " + "w" * 3000 + " RESULT_END"
        row = d.add("long task")
        d._execute = lambda text: long_result  # noqa: SLF001
        out = d.run(row["id"])
        assert out["ok"] is True
        call = rec.calls[-1]
        assert call["kind"] == "task"
        assert call["title"].startswith("task done: long task")
        # the old publish cut the result at 1500 chars
        assert "RESULT_END" in call["body"]
        assert call["doc"].tone == "ok"


# ── templates never raise ─────────────────────────────────────────

class TestNeverRaise:
    def test_builders_tolerate_junk(self):
        for doc in (
            T.immune_event_doc(None),
            T.watcher_hit_doc(title="", watcher_name="", watcher_id="",
                              severity="", kind="", summary=""),
            T.watcher_notice_doc("", ""),
            T.watcher_digest_doc(0, []),
            T.ops_finding_doc({}),
            T.backup_result_doc({}),
            T.backup_failure_doc("", None),
            T.backup_drill_doc(SimpleNamespace(ok=False, backup_id="b",
                                               steps=[])),
            T.backup_queued_doc({}),
            T.research_loop_doc(title="", briefs=[], findings=0,
                                proposals=0, signals=""),
            T.research_job_doc({}),
            T.trial_notice_doc("", ""),
            T.task_notice_doc("", ""),
            T.upgrade_proposed_doc(pid="", title="", rationale=""),
            T.upgrade_merged_doc(pid="", title="", target_id=""),
            T.upgrade_decided_doc(pid="", title="", by="",
                                  approved=False),
            T.upgrade_implemented_doc(pid="", status="", evo_id="",
                                      result={}),
        ):
            assert doc is not None
            assert isinstance(body_text(doc), str)
