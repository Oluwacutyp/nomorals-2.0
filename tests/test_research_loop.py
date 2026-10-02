"""Wave E acceptance tests: the always-on research loop.

Covers research_loop.py:
 1. research_hours reads NM_RESEARCH_LOOP_HOURS (default 6), clamps, and
    falls back on garbage.
 2. ensure_research_job is idempotent by name and replaces a stale
    interval; the payload fires the research_loop tool's tick action.
 3. loop_gates defers when the research feature flag is off, when the
    proactive master switch is off, and during quiet hours — and lets a
    fully-enabled context through. Reuses the existing settings only.
 4. owner_topics falls back to the built-in seeds; set_owner_topics
    persists and validates.
 5. pick_topics skips topics covered in recent runs (anti-repeat) and
    surfaces KG contradicted-claim follow-ups.
 6. ResearchLoop.tick defers cleanly (recorded, never raises) when gated;
    with a fake pipeline it runs the full cycle — findings counted,
    tickets filed into the upgrade queue, owner notified, run recorded.
 7. status exposes job health, gates, last run, and pending proposals.
 8. migration 64 creates research_loop_runs (latest_version >= 64).
"""
from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import research_loop as rl
from nomorals.storage import migrations
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None, proactive=True):
    from nomorals.agents.features import FeatureRegistry

    db = db or _db()
    partner = SimpleNamespace(proactive_enabled=proactive, quiet_start=22,
                              quiet_end=8)
    settings = SimpleNamespace(partner=partner)
    return SimpleNamespace(db=db, settings=settings)


def _enable_research(ctx):
    from nomorals.agents.features import FeatureRegistry

    FeatureRegistry(ctx.db).set("research", True)


class HoursTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("NM_RESEARCH_LOOP_HOURS")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("NM_RESEARCH_LOOP_HOURS", None)
        else:
            os.environ["NM_RESEARCH_LOOP_HOURS"] = self._old

    def test_default_six(self):
        os.environ.pop("NM_RESEARCH_LOOP_HOURS", None)
        self.assertEqual(rl.research_hours(), 6.0)

    def test_env_parses(self):
        os.environ["NM_RESEARCH_LOOP_HOURS"] = "12"
        self.assertEqual(rl.research_hours(), 12.0)

    def test_garbage_falls_back(self):
        os.environ["NM_RESEARCH_LOOP_HOURS"] = "soon"
        self.assertEqual(rl.research_hours(), 6.0)

    def test_clamped(self):
        os.environ["NM_RESEARCH_LOOP_HOURS"] = "500"
        self.assertEqual(rl.research_hours(), 48.0)
        os.environ["NM_RESEARCH_LOOP_HOURS"] = "0.1"
        self.assertEqual(rl.research_hours(), 1.0)


class GatesTests(unittest.TestCase):
    def test_feature_off_defers(self):
        ctx = _ctx()
        ok, reason = rl.loop_gates(ctx)
        self.assertFalse(ok)
        self.assertIn("research", reason)

    def test_proactive_master_off_defers(self):
        ctx = _ctx(proactive=False)
        _enable_research(ctx)
        ok, reason = rl.loop_gates(ctx)
        self.assertFalse(ok)
        self.assertIn("proactive", reason)

    def test_quiet_hours_defers(self):
        ctx = _ctx()
        _enable_research(ctx)
        with mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                        return_value=True):
            ok, reason = rl.loop_gates(ctx)
        self.assertFalse(ok)
        self.assertIn("quiet hours", reason)

    def test_fully_enabled_passes(self):
        ctx = _ctx()
        _enable_research(ctx)
        with mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                        return_value=False):
            ok, reason = rl.loop_gates(ctx)
        self.assertTrue(ok)
        self.assertEqual(reason, "")


class JobTests(unittest.TestCase):
    def test_ensure_idempotent(self):
        ctx = _ctx()
        first = rl.ensure_research_job(ctx)
        second = rl.ensure_research_job(ctx)
        self.assertTrue(second["already_scheduled"])
        self.assertEqual(first["job_id"], second["job_id"])
        from nomorals.agents.scheduler import Scheduler

        jobs = [j for j in Scheduler(ctx).list_jobs()
                if j["name"] == rl.RESEARCH_LOOP_JOB]
        self.assertEqual(len(jobs), 1)
        self.assertTrue(jobs[0]["enabled"])
        # payload fires the tool's tick action
        import json

        row = ctx.db.query_one(
            "SELECT payload FROM schedule_jobs WHERE name=?",
            (rl.RESEARCH_LOOP_JOB,))
        payload = json.loads(row["payload"])
        self.assertEqual(payload["tool"], "research_loop")
        self.assertEqual(payload["args"], {"action": "tick"})

    def test_stale_interval_replaced(self):
        ctx = _ctx()
        from nomorals.agents.scheduler import Scheduler

        sched = Scheduler(ctx)
        sched.add(rl.RESEARCH_LOOP_JOB, "every 1h", "tool",
                  {"tool": "research_loop", "args": {"action": "tick"}})
        out = rl.ensure_research_job(ctx)
        self.assertNotIn("already_scheduled", out)
        jobs = [j for j in sched.list_jobs()
                if j["name"] == rl.RESEARCH_LOOP_JOB]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["spec"], "every 21600s")


class TopicsTests(unittest.TestCase):
    def test_default_seeds(self):
        ctx = _ctx()
        topics = rl.owner_topics(ctx)
        self.assertTrue(len(topics) >= 4)
        self.assertTrue(all(topics))

    def test_set_and_get(self):
        ctx = _ctx()
        stored = rl.set_owner_topics(ctx, ["alpha", "beta"])
        self.assertEqual(stored, ["alpha", "beta"])
        self.assertEqual(rl.owner_topics(ctx), ["alpha", "beta"])

    def test_set_rejects_empty(self):
        ctx = _ctx()
        with self.assertRaises(ValueError):
            rl.set_owner_topics(ctx, ["  ", ""])

    def test_anti_repeat(self):
        ctx = _ctx()
        rl.set_owner_topics(ctx, ["alpha", "beta", "gamma"])
        rl.record_run(ctx, {"started_at": time.time() - 10,
                            "finished_at": time.time(),
                            "topics": ["alpha"]})
        picked = rl.pick_topics(ctx, limit=3)
        self.assertNotIn("alpha", picked)
        self.assertIn("beta", picked)
        self.assertIn("gamma", picked)

    def test_contradicted_kg_followups(self):
        import json

        from nomorals.agents.kg import KnowledgeGraph

        ctx = _ctx()
        rl.set_owner_topics(ctx, ["alpha"])
        kg = KnowledgeGraph(ctx.db)
        kg.upsert_node("claim:old1", type="claim",
                       properties={"claim": "X improves throughput",
                                   "query": "stale distributed caching wisdom",
                                   "status": "contradicted"})
        picked = rl.pick_topics(ctx, limit=3)
        self.assertIn("stale distributed caching wisdom", picked)


class TickTests(unittest.TestCase):
    def _enabled_ctx(self):
        ctx = _ctx()
        _enable_research(ctx)
        return ctx

    def test_tick_deferred_records_run(self):
        ctx = _ctx()  # feature flag off by default
        out = rl.ResearchLoop(ctx).tick()
        self.assertFalse(out["ok"])
        self.assertTrue(out["skipped_reason"])
        row = ctx.db.query_one(
            "SELECT skipped_reason FROM research_loop_runs")
        self.assertEqual(row["skipped_reason"], out["skipped_reason"])

    def test_tick_never_raises(self):
        ctx = self._enabled_ctx()
        with (mock.patch("nomorals.agents.research_loop.pick_topics",
                         side_effect=RuntimeError("boom")),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False)):
            out = rl.ResearchLoop(ctx).tick()
        self.assertFalse(out["ok"])
        self.assertIn("RuntimeError", out["error"])

    def test_full_cycle_with_fake_pipeline(self):
        ctx = self._enabled_ctx()
        rl.set_owner_topics(ctx, ["alpha"])
        fake_result = {
            "report": {"findings": [{"claim": "a"}, {"claim": "b"}]},
            "claims": [{"id": "c1"}],
            "brief": "brief text",
            "tickets": [{"title": "[systems] foo bar baz", "confidence": 0.9}],
        }
        fake_pipeline = mock.Mock()
        fake_pipeline.run.return_value = fake_result
        fake_upgrade = mock.Mock()
        fake_upgrade.propose_from_ticket.return_value = "upg_123"
        fake_upgrade_cls = mock.Mock(return_value=fake_upgrade)
        fake_notifier = mock.Mock()
        fake_notifier.publish.return_value = {"delivered": True}
        with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                         fake_pipeline),
              mock.patch("nomorals.agents.upgrade_queue.UpgradePipeline",
                         fake_upgrade_cls),
              mock.patch("nomorals.agents.research_loop.Notifier",
                         return_value=fake_notifier),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False)):
            out = rl.ResearchLoop(ctx).tick()
        self.assertTrue(out["ok"])
        self.assertEqual(out["topics"], ["alpha"])
        self.assertEqual(out["findings_count"], 2)
        self.assertEqual(out["claims_count"], 1)
        self.assertEqual(out["proposals_created"], 1)
        self.assertEqual(out["proposal_ids"], ["upg_123"])
        self.assertTrue(out["notified"])
        fake_upgrade.propose_from_ticket.assert_called_once()
        args, kwargs = fake_upgrade.propose_from_ticket.call_args
        self.assertEqual(kwargs.get("source"), "research_loop")
        fake_notifier.publish.assert_called_once()
        kind = fake_notifier.publish.call_args.args[0]
        self.assertEqual(kind, "research")
        # run history persisted
        row = ctx.db.query_one(
            "SELECT findings_count, proposals_created, proposal_ids, notified "
            "FROM research_loop_runs")
        self.assertEqual(row["findings_count"], 2)
        self.assertEqual(row["proposals_created"], 1)
        self.assertIn("upg_123", row["proposal_ids"])
        self.assertEqual(row["notified"], 1)


class StatusTests(unittest.TestCase):
    def test_status_shape(self):
        ctx = _ctx()
        rl.ensure_research_job(ctx)
        rl.record_run(ctx, {"started_at": time.time() - 60,
                            "finished_at": time.time() - 50,
                            "ok": True, "topics": ["alpha"],
                            "findings_count": 3, "proposals_created": 1,
                            "proposal_ids": ["upg_1"]})
        st = rl.status(ctx)
        self.assertTrue(st["job"]["scheduled"])
        self.assertTrue(st["job"]["enabled"])
        self.assertIsNotNone(st["job"]["next_run"])
        self.assertFalse(st["gates"]["feature_research"])
        self.assertIsNotNone(st["last_run"])
        self.assertEqual(st["last_run"]["topics"], ["alpha"])
        self.assertEqual(st["pending_proposals"], 0)
        self.assertTrue(st["topics"])


class RegisterTests(unittest.TestCase):
    def test_tool_registers_and_serves_status(self):
        ctx = _ctx()
        calls = {}

        class _Registry:
            context = ctx

            def register(self, name, **kw):
                def deco(fn):
                    calls[name] = fn
                    return fn
                return deco

        rl.register(_Registry())
        self.assertIn("research_loop", calls)
        out = calls["research_loop"](action="status")
        self.assertTrue(out["ok"])
        self.assertIn("job", out)
        out = calls["research_loop"](action="topics")
        self.assertTrue(out["ok"])
        with self.assertRaises(ValueError):
            calls["research_loop"](action="bogus")


class MigrationTests(unittest.TestCase):
    def test_migration_64_applied(self):
        self.assertGreaterEqual(migrations.latest_version(), 64)
        db = _db()
        row = db.query_one("SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name='research_loop_runs'")
        self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
