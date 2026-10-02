"""Wave D Stream E: proactive organs — useful, measurable, non-spammy.

Covers the mission's guarantees:
- the Notifier dedupe choke point: double-send -> one delivery, critical
  sends dedupe too, ``force`` is the only escape hatch
- metrics visibility: delivery_counts + proactive_status counts/health
- quiet-hours hold mechanism (proactive gate)
- owner-only delivery through every choke (incl. the arena loop wrap)
- scheduler: a stuck failing job pages once per failure episode, not
  every run
- evolution + skill evolution go through the upgrade queue approve/deny
  path (no parallel approval surfaces)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import notifier as notmod
from nomorals.agents.notifier import Notifier, proactive_gate
from nomorals.agents.upgrade_queue import UpgradePipeline, UpgradeQueue
from nomorals.storage.db import Database


class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok


class FakeGateway:
    def __init__(self, live_platforms=("telegram",)):
        self.live = set(live_platforms)
        self.sent = []

    def status(self):
        return {p: {"running_in_session": True} for p in self.live}

    def send(self, platform, chat_ref, text):
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


def make_ctx(partner=None, gateway=None, **kw):
    tmp = tempfile.mkdtemp(prefix="wave-d-e-")
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner or make_partner(), **kw)
    ctx = SimpleNamespace(db=db, settings=settings,
                          extras={"gateway": gateway} if gateway else {})
    return ctx, tmp


# ── 1. the dedupe choke point ───────────────────────────────────────────────

class DedupeChokeTests(unittest.TestCase):
    def test_double_send_delivers_once(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        r1 = n.publish("watcher", "price hit: BTC", "body")
        r2 = n.publish("watcher", "price hit: BTC", "body")
        self.assertEqual(r1["delivery_state"], "sent")
        self.assertEqual(r2["delivery_state"], "deduped")
        self.assertTrue(r2["deduped"])
        self.assertEqual(len(gw.sent), 1)

    def test_critical_still_dedupes(self):
        # critical bypasses the feature flag, NOT the dedupe window
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        r1 = n.publish("watcher", "urgent: disk full", "b", critical=True)
        r2 = n.publish("watcher", "urgent: disk full", "b", critical=True)
        self.assertEqual(r1["delivery_state"], "sent")
        self.assertEqual(r2["delivery_state"], "deduped")
        self.assertEqual(len(gw.sent), 1)

    def test_force_is_the_escape_hatch(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        n.publish("briefing", "manual resend", "b", force=True)
        r = n.publish("briefing", "manual resend", "b", force=True)
        self.assertEqual(r["delivery_state"], "sent")
        self.assertEqual(len(gw.sent), 2)

    def test_critical_bypasses_muted_flag_but_not_dedupe(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(partner=make_partner(), gateway=gw)
        n = Notifier(ctx)
        with patch("nomorals.agents.features.feature_enabled", return_value=False):
            r = n.publish("watcher", "urgent: disk full 2", "b",
                          critical=True)
            self.assertEqual(r["delivery_state"], "sent")
            # ...but a repeat inside the window is still spam
            r2 = n.publish("watcher", "urgent: disk full 2", "b",
                           critical=True)
            self.assertEqual(r2["delivery_state"], "deduped")

    def test_deduped_row_is_not_stored_twice(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        n.publish("news", "digest", "b")
        n.publish("news", "digest", "b")
        rows = ctx.db.query(
            "SELECT COUNT(*) AS c FROM notifications WHERE kind='news'")
        self.assertEqual(rows[0]["c"], 1)


# ── 2. metrics visibility ───────────────────────────────────────────────────

class MetricsVisibilityTests(unittest.TestCase):
    def test_delivery_counts_aggregates_states(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        n = Notifier(ctx)
        n.publish("watcher", "a1", "b")          # sent
        n.publish("watcher", "a1", "b")          # deduped (not stored)
        n.publish("watcher", "a2", "b")          # sent
        counts = n.delivery_counts(24.0, kinds=("watcher",))
        self.assertEqual(counts.get("sent"), 2)
        # a muted send is recorded too
        with patch("nomorals.agents.features.feature_enabled", return_value=False):
            n.publish("news", "quiet one", "b")
        counts_all = n.delivery_counts(24.0)
        self.assertEqual(counts_all.get("muted"), 1)

    def test_proactive_status_exposes_counts_and_health(self):
        from nomorals.agents.morning_briefing import proactive_status

        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        Notifier(ctx).publish("briefing", "morning", "b")
        st = proactive_status(ctx)
        for key in ("settings", "recent", "counts", "health"):
            self.assertIn(key, st)
        self.assertEqual(st["counts"].get("sent"), 1)
        self.assertTrue(st["health"]["ok"])
        self.assertEqual(st["health"]["live_channels"], ["telegram"])

    def test_proactive_status_names_degradation(self):
        from nomorals.agents.morning_briefing import proactive_status

        # no gateway at all: persist-only — must be said out loud
        ctx, _ = make_ctx()
        st = proactive_status(ctx)
        self.assertFalse(st["health"]["ok"])
        self.assertTrue(any("no live gateway" in d
                            for d in st["health"]["degraded"]))

    def test_proactive_status_names_missing_owner_chats(self):
        from nomorals.agents.morning_briefing import proactive_status

        ctx, _ = make_ctx(partner=make_partner(owner_chats=""),
                          gateway=FakeGateway())
        st = proactive_status(ctx)
        self.assertFalse(st["health"]["ok"])
        self.assertTrue(any("owner_chats" in d
                            for d in st["health"]["degraded"]))


# ── 3. quiet-hours hold ─────────────────────────────────────────────────────

class QuietHoursTests(unittest.TestCase):
    def _ctx(self):
        return make_ctx(partner=make_partner(timezone="UTC"),
                        gateway=FakeGateway())

    def test_gate_holds_non_exempt_kind(self):
        ctx, _ = self._ctx()
        with patch.dict(notmod.PROACTIVE_KIND_SETTINGS,
                        {"testkind": "proactive_testkind"}), \
             patch.object(notmod, "_in_quiet_hours_now", return_value=True):
            ctx.settings.partner.proactive_testkind = True
            gate = proactive_gate(ctx, "testkind")
        self.assertEqual(gate, "held-quiet-hours")

    def test_gate_disabled_when_master_off(self):
        ctx, _ = make_ctx(partner=make_partner(proactive_enabled=False),
                          gateway=FakeGateway())
        self.assertEqual(proactive_gate(ctx, "briefing"), "disabled")

    def test_held_send_is_stored_with_state(self):
        ctx, _ = self._ctx()
        with patch.dict(notmod.PROACTIVE_KIND_SETTINGS,
                        {"testkind": "proactive_testkind"}), \
             patch.object(notmod, "_in_quiet_hours_now", return_value=True):
            ctx.settings.partner.proactive_testkind = True
            res = Notifier(ctx).publish("testkind", "t", "b")
        self.assertEqual(res["delivery_state"], "held-quiet-hours")
        self.assertFalse(res["delivered"])

    def test_briefing_exempt_like_an_alarm(self):
        ctx, _ = self._ctx()
        with patch.object(notmod, "_in_quiet_hours_now", return_value=True):
            # the scheduled briefing is exempt from the global quiet window
            self.assertIsNone(proactive_gate(ctx, "briefing"))


# ── 4. owner-only through every choke ───────────────────────────────────────

class OwnerOnlyChokeTests(unittest.TestCase):
    def test_arena_loop_push_is_owner_only(self):
        from nomorals.agents.arena.core import Arena

        gw = FakeGateway()
        ctx, _ = make_ctx(
            partner=make_partner(owner_chats="telegram:111"), gateway=gw)
        arena = Arena(ctx)
        wrapped = arena._choked_review_push(lambda p: None)
        self.assertIsNotNone(wrapped)
        wrapped("🧪 arena build awaiting your review\nid: abc\n"
                "name: test build\npurpose: x\n")
        self.assertEqual(len(gw.sent), 1)
        plat, chat_id, text = gw.sent[0]
        self.assertEqual((plat, chat_id), ("telegram", "111"))
        # and the push went through the dedupe choke: same packet twice
        wrapped("🧪 arena build awaiting your review\nid: abc\n"
                "name: test build\npurpose: x\n")
        self.assertEqual(len(gw.sent), 1)
        rows = ctx.db.query(
            "SELECT delivery_state FROM notifications "
            "WHERE kind='arena_build' ORDER BY created_at DESC")
        self.assertEqual(rows[0]["delivery_state"], "sent")

    def test_arena_loop_wrap_none_stays_none(self):
        from nomorals.agents.arena.core import Arena

        ctx, _ = make_ctx()
        arena = Arena(ctx)
        self.assertIsNone(arena._choked_review_push(None))


# ── 5. scheduler: one page per failure episode ──────────────────────────────

class SchedulerSpamTests(unittest.TestCase):
    def _scheduler(self, gw):
        from nomorals.agents.scheduler import Scheduler

        ctx, _ = make_ctx(gateway=gw)
        sched = Scheduler(ctx)
        sched.notifier = Notifier(ctx, gateway=gw)
        return sched, ctx

    def _add_failing_job(self, ctx, name="flaky"):
        jid = "job-flaky-1"
        ctx.db.execute(
            "INSERT INTO schedule_jobs (id, name, kind, spec, payload_kind, "
            "payload, enabled, next_run, last_run, last_result, created_at, "
            "updated_at) VALUES (?, ?, 'every', '60', 'command', ?, 1, 0, "
            "NULL, '', ?, ?)",
            (jid, name, json.dumps({"command": "exit 1"}),
             time.time(), time.time()))
        return jid

    def test_stuck_job_pages_once(self):
        gw = FakeGateway()
        sched, ctx = self._scheduler(gw)
        jid = self._add_failing_job(ctx)
        with patch.object(sched, "_run_command",
                          side_effect=RuntimeError("boom")):
            sched._execute(dict(ctx.db.query_one(
                "SELECT * FROM schedule_jobs WHERE id=?", (jid,))))
            sched._execute(dict(ctx.db.query_one(
                "SELECT * FROM schedule_jobs WHERE id=?", (jid,))))
            sched._execute(dict(ctx.db.query_one(
                "SELECT * FROM schedule_jobs WHERE id=?", (jid,))))
        fails = [r for r in ctx.db.query(
            "SELECT * FROM notifications WHERE kind='schedule'")]
        # one page for the episode — not three
        self.assertEqual(len(fails), 1)
        self.assertIn("❌", fails[0]["title"])

    def test_recovered_then_failing_pages_again(self):
        gw = FakeGateway()
        sched, ctx = self._scheduler(gw)
        jid = self._add_failing_job(ctx)
        with patch.object(sched, "_run_command",
                          side_effect=RuntimeError("boom")):
            sched._execute(dict(ctx.db.query_one(
                "SELECT * FROM schedule_jobs WHERE id=?", (jid,))))
        with patch.object(sched, "_run_command", return_value="ok"):
            sched._execute(dict(ctx.db.query_one(
                "SELECT * FROM schedule_jobs WHERE id=?", (jid,))))
        # force the success alert out of the dedupe window
        ctx.db.execute(
            "UPDATE notifications SET created_at=? WHERE kind='schedule'",
            (time.time() - 700,))
        with patch.object(sched, "_run_command",
                          side_effect=RuntimeError("boom again")):
            sched._execute(dict(ctx.db.query_one(
                "SELECT * FROM schedule_jobs WHERE id=?", (jid,))))
        fails = [r for r in ctx.db.query(
            "SELECT * FROM notifications WHERE kind='schedule' "
            "AND title LIKE '❌%'")]
        self.assertEqual(len(fails), 2)


# ── 6. evolution + skill evolution use the upgrade queue ────────────────────

class UpgradeQueueApprovalTests(unittest.TestCase):
    def _ctx(self):
        return make_ctx()

    def test_evolve_plan_files_into_queue(self):
        from nomorals.agents.evolution import EvolutionAgent, EvolutionProposal

        ctx, _ = self._ctx()
        agent = EvolutionAgent(ctx)
        prop = EvolutionProposal(id="evo-test-1", instruction="make x faster",
                                 edits=[{"path": "a.py", "old": "x",
                                         "new": "y"}],
                                 rationale="benchmark says so, clearly")
        agent._save(prop)
        qid = agent.submit_to_queue("evo-test-1")
        self.assertTrue(qid)
        row = UpgradeQueue(ctx).get(qid)
        self.assertEqual(row["status"], "proposed")
        self.assertEqual(row["patch_plan"]["evolution_proposal_id"],
                         "evo-test-1")
        self.assertEqual(row["source"], "evolution")

    def test_queue_approve_applies_existing_evolution_proposal(self):
        from nomorals.agents.evolution import EvolutionAgent, EvolutionProposal

        ctx, _ = self._ctx()
        agent = EvolutionAgent(ctx)
        prop = EvolutionProposal(id="evo-test-2", instruction="make y faster",
                                 edits=[], rationale="r" * 30)
        agent._save(prop)
        qid = UpgradeQueue(ctx).propose(
            title="evolution: make y faster",
            rationale="benchmark says so, really" + "x" * 10,
            patch_plan={"source": "evolution",
                        "evolution_proposal_id": "evo-test-2"},
            files=[], tests=[], source="evolution")
        with patch.object(EvolutionAgent, "apply",
                          return_value={"ok": True, "applied": True,
                                        "proposal": "evo-test-2"}) as m:
            out = UpgradePipeline(ctx).approve_and_implement(qid, by="owner")
        m.assert_called_once()
        args, kwargs = m.call_args
        self.assertEqual(args[0], "evo-test-2")
        self.assertTrue(kwargs.get("verify"))
        self.assertEqual(out["status"], "implemented")

    def test_queue_deny_rejects_evolution_proposal(self):
        from nomorals.agents.evolution import EvolutionAgent, EvolutionProposal

        ctx, _ = self._ctx()
        agent = EvolutionAgent(ctx)
        prop = EvolutionProposal(id="evo-test-3", instruction="drop tables",
                                 edits=[], rationale="r" * 30)
        agent._save(prop)
        qid = UpgradeQueue(ctx).propose(
            title="evolution: drop tables",
            rationale="seemed fun at the time, honestly" + "x" * 10,
            patch_plan={"source": "evolution",
                        "evolution_proposal_id": "evo-test-3"},
            files=[], tests=[], source="evolution")
        UpgradePipeline(ctx).deny_with_reason(qid, "absolutely not",
                                              by="owner")
        self.assertEqual(agent._load("evo-test-3").status, "rejected")
        self.assertEqual(UpgradeQueue(ctx).get(qid)["status"], "denied")

    def test_skill_approval_mode_files_into_queue(self):
        from nomorals.agents.skill_evolution import SkillEvolutionLoop

        ctx, tmp = self._ctx()
        ctx.settings.improvement = SimpleNamespace(mode="approval")
        loop = SkillEvolutionLoop(ctx)
        proposal = {
            "skill_name": "testskill",
            "target_kind": "db",
            "target_ref": "skill:testskill",
            "diff": ("--- a\n+++ b\n@@ -1 +1 @@\n"
                     "-old line\n+new line\n"),
            "changed_lines": 2,
            "before_hash": "a" * 16,
            "after_hash": "b" * 16,
            "new_text": "new line\n",
            "failures": [{"summary": "kept failing"}],
            "fingerprint": "fp1",
        }
        with patch.object(SkillEvolutionLoop, "gate",
                          return_value=(True, {"static": {"ok": True}})):
            rec = loop.apply_proposal(dict(proposal), mode="approval")
        self.assertEqual(rec.status, "staged")
        qid = rec.gate_results.get("upgrade_proposal_id")
        self.assertTrue(qid)
        row = UpgradeQueue(ctx).get(qid)
        self.assertEqual(row["patch_plan"]["skill_edit_id"], rec.id)
        self.assertEqual(row["source"], "skill_evolution")

    def test_queue_approve_applies_staged_skill_edit(self):
        from nomorals.agents.skill_evolution import SkillEvolutionLoop

        ctx, tmp = self._ctx()
        ctx.settings.improvement = SimpleNamespace(mode="approval")
        loop = SkillEvolutionLoop(ctx)
        proposal = {
            "skill_name": "testskill2",
            "target_kind": "db",
            "target_ref": "skill:testskill2",
            "diff": ("--- a\n+++ b\n@@ -1 +1 @@\n"
                     "-old line\n+new line\n"),
            "changed_lines": 2,
            "before_hash": "a" * 16,
            "after_hash": "b" * 16,
            "new_text": "new line\n",
            "failures": [],
            "fingerprint": "fp2",
        }
        with patch.object(SkillEvolutionLoop, "gate",
                          return_value=(True, {"static": {"ok": True}})):
            rec = loop.apply_proposal(dict(proposal), mode="approval")
        qid = rec.gate_results.get("upgrade_proposal_id")
        with patch.object(SkillEvolutionLoop, "apply_staged_edit",
                          return_value={"ok": True, "edit": rec.id,
                                        "status": "applied"}) as m:
            out = UpgradePipeline(ctx).approve_and_implement(qid, by="owner")
        m.assert_called_once_with(rec.id)
        self.assertEqual(out["status"], "implemented")

    def test_queue_deny_denies_staged_skill_edit(self):
        from nomorals.agents.skill_evolution import SkillEvolutionLoop

        ctx, tmp = self._ctx()
        ctx.settings.improvement = SimpleNamespace(mode="approval")
        loop = SkillEvolutionLoop(ctx)
        proposal = {
            "skill_name": "testskill3",
            "target_kind": "db",
            "target_ref": "skill:testskill3",
            "diff": ("--- a\n+++ b\n@@ -1 +1 @@\n"
                     "-old line\n+new line\n"),
            "changed_lines": 2,
            "before_hash": "a" * 16,
            "after_hash": "b" * 16,
            "new_text": "new line\n",
            "failures": [],
            "fingerprint": "fp3",
        }
        with patch.object(SkillEvolutionLoop, "gate",
                          return_value=(True, {"static": {"ok": True}})):
            rec = loop.apply_proposal(dict(proposal), mode="approval")
        qid = rec.gate_results.get("upgrade_proposal_id")
        UpgradePipeline(ctx).deny_with_reason(qid, "nope", by="owner")
        row = ctx.db.query_one("SELECT status FROM skill_edits WHERE id=?",
                               (rec.id,))
        self.assertEqual(row["status"], "denied")


# ── 7. research digest: no empty/filler notifies, confidence filtering ──────

class ResearchDigestTightenTests(unittest.TestCase):
    def test_brief_filters_low_confidence_findings(self):
        from nomorals.agents.research_digest import (
            ResearchDigest, ResearchClaim)

        claims = [
            ResearchClaim(id="c1", claim="strong claim here", confidence=0.9),
            ResearchClaim(id="c2", claim="weak claim here", confidence=0.1),
        ]
        brief = ResearchDigest.operator_brief(claims, min_confidence=0.5)
        self.assertIn("strong claim here", brief)
        self.assertNotIn("weak claim here", brief)

    def test_brief_empty_when_nothing_survives(self):
        from nomorals.agents.research_digest import (
            ResearchDigest, ResearchClaim)

        claims = [ResearchClaim(id="c1", claim="weak", confidence=0.1)]
        self.assertEqual(
            ResearchDigest.operator_brief(claims, min_confidence=0.5), "")

    def test_pipeline_notify_reports_real_delivery(self):
        from nomorals.agents import research_digest as rdmod
        from nomorals.agents.research_digest import ResearchPipeline

        gw = FakeGateway()
        ctx, _ = make_ctx(gateway=gw)
        report = SimpleNamespace(
            query="q", synthesis="synth", findings=[], conflicts=[],
            failed_angles=[], critique="",
            to_dict=lambda: {})
        with patch.object(rdmod.ResearchSwarm, "run", return_value=report):
            out = ResearchPipeline.run("q", ctx, notify=True,
                                       promote_claims=False)
        # empty brief -> nothing sent, notified must be honest
        self.assertEqual(out["brief"], "")
        self.assertFalse(out["notified"])
        self.assertEqual(len(gw.sent), 0)


if __name__ == "__main__":
    unittest.main()
