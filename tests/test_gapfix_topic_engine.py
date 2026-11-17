"""Topic-engine acceptance tests (owner phase 2026-10-21, item 1).

The research loop was starving: owner seeds capped at 12, 4 evergreen
fallbacks, at most 4 KG contradicted follow-ups, and a hard
last-3-runs anti-repeat wall — a finite pool that empties. The topic
engine feeds ``pick_topics`` a third source (open questions from
findings, KG gaps/staleness, world signals, work-generated questions)
and the staleness curve replaces the wall: a topic covered recently
ranks last, never excluded, so the loop can never starve.

Covers ``nomorals/agents/topic_engine.py`` + the ``pick_topics``
integration in ``research_loop.py``.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from nomorals.agents import topic_engine as te
except ImportError:  # failing-first: module not built yet
    te = None

from nomorals.agents import research_loop as rl
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None, proactive=True):
    db = db or _db()
    partner = SimpleNamespace(proactive_enabled=proactive, quiet_start=22,
                              quiet_end=8)
    settings = SimpleNamespace(partner=partner)
    return SimpleNamespace(db=db, settings=settings)


def _enable_research(ctx):
    from nomorals.agents.features import FeatureRegistry

    FeatureRegistry(ctx.db).set("research", True)


def _ensure_gap_tables(db):
    """knowledge_gaps + research_watches, mirroring
    ``research/autonomy.py::ensure_schema`` DDL (those tables are NOT
    created by migrations — the research organ creates them at runtime)."""
    db.execute(
        "CREATE TABLE IF NOT EXISTS knowledge_gaps ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "question TEXT NOT NULL,"
        "context TEXT NOT NULL DEFAULT '',"
        "ts REAL NOT NULL,"
        "status TEXT NOT NULL DEFAULT 'open',"
        "resolved_ts REAL NOT NULL DEFAULT 0,"
        "priority REAL NOT NULL DEFAULT 0,"
        "attempts INTEGER NOT NULL DEFAULT 0)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS research_watches ("
        "id TEXT PRIMARY KEY,"
        "topic TEXT NOT NULL,"
        "queries TEXT NOT NULL DEFAULT '[]',"
        "cadence_hours REAL NOT NULL DEFAULT 24.0,"
        "goals TEXT NOT NULL DEFAULT '[]',"
        "max_results INTEGER NOT NULL DEFAULT 6,"
        "fetch_top INTEGER NOT NULL DEFAULT 2,"
        "enabled INTEGER NOT NULL DEFAULT 1,"
        "created_at REAL NOT NULL,"
        "created_by TEXT NOT NULL DEFAULT 'system')")


class _BoomDB:
    def query(self, *args, **kwargs):
        raise RuntimeError("boom")

    def query_one(self, *args, **kwargs):
        raise RuntimeError("boom")

    def execute(self, *args, **kwargs):
        raise RuntimeError("boom")


class EngineRequired(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(te, "topic_engine not built yet")


class NormalizeTests(EngineRequired):
    def test_normalize(self):
        self.assertEqual(te.normalize_topic("  Hello   World?  "),
                         "hello world")
        self.assertEqual(te.normalize_topic("A  B\tC\n"), "a b c")

    def test_duplicate_exact_and_near(self):
        self.assertTrue(te.topics_duplicate(
            "how do agents learn from failures",
            "How do agents learn from failures?"))
        self.assertTrue(te.topics_duplicate(
            "autonomous coding agent self-improvement loop techniques",
            "autonomous coding agent self improvement loop technique"))
        self.assertFalse(te.topics_duplicate(
            "how do agents learn from failures",
            "best protein targets for strength training"))

    def test_duplicate_needs_real_overlap(self):
        self.assertFalse(te.topics_duplicate("alpha beta gamma", "x y z"))
        self.assertFalse(te.topics_duplicate("", "something"))


class CurveTests(EngineRequired):
    def test_never_covered_is_fully_eligible(self):
        self.assertEqual(te.staleness_eligibility(None), 1.0)

    def test_just_covered_is_floored_not_zero(self):
        self.assertEqual(te.staleness_eligibility(0.0),
                         te.STALENESS_FLOOR)
        self.assertGreater(te.staleness_eligibility(0.0), 0.0)

    def test_halfway_is_halfway(self):
        half = te.STALENESS_FULL_DAYS / 2.0
        got = te.staleness_eligibility(half)
        expect = te.STALENESS_FLOOR + (1.0 - te.STALENESS_FLOOR) * 0.5
        self.assertAlmostEqual(got, expect, places=6)

    def test_full_after_horizon_and_never_above_one(self):
        self.assertEqual(te.staleness_eligibility(te.STALENESS_FULL_DAYS),
                         1.0)
        self.assertEqual(te.staleness_eligibility(1000.0), 1.0)


class AgesTests(EngineRequired):
    def test_ages_from_run_history(self):
        ctx = _ctx()
        now = time.time()
        rl.record_run(ctx, {"started_at": now - 2 * 86400,
                            "finished_at": now, "topics": ["alpha", "beta"]})
        rl.record_run(ctx, {"started_at": now - 10 * 86400,
                            "finished_at": now, "topics": ["alpha"]})
        ages = te.topic_ages(ctx, now=now)
        self.assertAlmostEqual(ages["alpha"], 2.0, places=3)
        self.assertAlmostEqual(ages["beta"], 2.0, places=3)
        self.assertNotIn("gamma", ages)

    def test_ages_empty_without_runs(self):
        self.assertEqual(te.topic_ages(_ctx()), {})


class HarvestGapTests(EngineRequired):
    def test_open_gap_harvested_with_provenance(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, context, ts, status,"
            " priority) VALUES (?, ?, ?, 'open', ?)",
            ("why do agents forget long conversations", "memory", time.time(),
             0.8))
        got = te.harvest_gaps(ctx)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["text"],
                         "why do agents forget long conversations")
        self.assertEqual(got[0]["source"], "gap")
        self.assertIn("knowledge_gap#", got[0]["provenance"])
        self.assertAlmostEqual(got[0]["priority"], 0.8)

    def test_resolved_gap_ignored(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, ts, status, priority)"
            " VALUES (?, ?, 'resolved', ?)",
            ("done question", time.time(), 0.9))
        self.assertEqual(te.harvest_gaps(ctx), [])

    def test_missing_table_is_empty_not_a_crash(self):
        self.assertEqual(te.harvest_gaps(_ctx()), [])


class HarvestKgStaleTests(EngineRequired):
    def _claim(self, ctx, cid, query, status, age_days):
        ctx.db.execute(
            "INSERT INTO kg_nodes (id, label, type, properties, created_at)"
            " VALUES (?, ?, 'claim', ?, ?)",
            (cid, "c", json.dumps({"claim": "some claim text",
                                   "query": query, "status": status}),
             time.time() - age_days * 86400.0))

    def test_old_active_claim_becomes_recheck(self):
        ctx = _ctx()
        self._claim(ctx, "claim:old", "stale distributed caching wisdom",
                    "active", 40)
        got = te.harvest_kg_stale(ctx)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["text"], "stale distributed caching wisdom")
        self.assertEqual(got[0]["source"], "kg_stale")

    def test_fresh_claim_not_stale(self):
        ctx = _ctx()
        self._claim(ctx, "claim:new", "fresh caching wisdom", "active", 1)
        self.assertEqual(te.harvest_kg_stale(ctx), [])

    def test_contradicted_claim_left_to_the_loop(self):
        ctx = _ctx()
        self._claim(ctx, "claim:bad", "contradicted wisdom", "contradicted",
                    40)
        self.assertEqual(te.harvest_kg_stale(ctx), [])


class HarvestWatchTests(EngineRequired):
    def test_enabled_watch_queries_harvested(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        ctx.db.execute(
            "INSERT INTO research_watches (id, topic, queries, enabled,"
            " created_at) VALUES (?, ?, ?, 1, ?)",
            ("w_1", "money", json.dumps(["bitcoin etf flows",
                                         "nigeria fintech funding"]),
             time.time()))
        got = te.harvest_watches(ctx)
        self.assertEqual([g["text"] for g in got],
                         ["bitcoin etf flows", "nigeria fintech funding"])
        self.assertTrue(all(g["source"] == "watch" for g in got))

    def test_disabled_watch_ignored(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        ctx.db.execute(
            "INSERT INTO research_watches (id, topic, queries, enabled,"
            " created_at) VALUES (?, ?, ?, 0, ?)",
            ("w_1", "money", json.dumps(["bitcoin etf flows"]), time.time()))
        self.assertEqual(te.harvest_watches(ctx), [])


class HarvestFailureTests(EngineRequired):
    def setUp(self):
        super().setUp()
        from nomorals.agents import failures

        self._failures = failures
        self._old_home = os.environ.get("NM_HOME")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["NM_HOME"] = self._tmp.name
        failures.clear_failures()

    def tearDown(self):
        self._failures.clear_failures()
        self._tmp.cleanup()
        if self._old_home is None:
            os.environ.pop("NM_HOME", None)
        else:
            os.environ["NM_HOME"] = self._old_home
        super().tearDown()

    def test_repeated_site_becomes_a_question(self):
        self._failures.record_failure("research_loop.tick",
                                      ValueError("boom"))
        self._failures.record_failure("research_loop.tick",
                                      ValueError("boom again"))
        got = te.harvest_failures(_ctx())
        self.assertEqual(len(got), 1)
        self.assertIn("research_loop.tick", got[0]["text"])
        self.assertEqual(got[0]["source"], "failure")

    def test_single_failure_is_noise(self):
        self._failures.record_failure("research_loop.tick",
                                      ValueError("boom"))
        self.assertEqual(te.harvest_failures(_ctx()), [])


class HarvestProposalTests(EngineRequired):
    def _proposal(self, ctx, pid, title, status, age_days):
        ctx.db.execute(
            "INSERT INTO upgrade_proposals (id, title, status, created_at)"
            " VALUES (?, ?, ?, ?)",
            (pid, title, status, time.time() - age_days * 86400.0))

    def test_old_denied_proposal_becomes_followup(self):
        ctx = _ctx()
        self._proposal(ctx, "upg_1", "cache the briefing ranker", "denied",
                       10)
        got = te.harvest_proposals(ctx)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["text"], "cache the briefing ranker")
        self.assertEqual(got[0]["source"], "proposal")

    def test_fresh_proposal_not_yet_a_followup(self):
        ctx = _ctx()
        self._proposal(ctx, "upg_1", "cache the briefing ranker", "proposed",
                       1)
        self.assertEqual(te.harvest_proposals(ctx), [])

    def test_approved_proposal_never_a_followup(self):
        ctx = _ctx()
        self._proposal(ctx, "upg_1", "cache the briefing ranker", "approved",
                       30)
        self.assertEqual(te.harvest_proposals(ctx), [])


class SuggestTests(EngineRequired):
    def test_suggest_dedupes_near_identical_gaps(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        for q in ("why do agents forget long conversations",
                  "why do agents forget long conversations?"):
            ctx.db.execute(
                "INSERT INTO knowledge_gaps (question, ts, status, priority)"
                " VALUES (?, ?, 'open', ?)", (q, time.time(), 0.8))
        engine = te.TopicEngine(ctx)
        got = engine.suggest(limit=10)
        self.assertEqual(len(got), 1)
        self.assertEqual(len(engine.load_ledger()), 1)

    def test_suggest_persists_provenance_and_ranks_by_priority(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, ts, status, priority)"
            " VALUES (?, ?, 'open', ?)",
            ("low priority question here", time.time(), 0.2))
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, ts, status, priority)"
            " VALUES (?, ?, 'open', ?)",
            ("high priority question here", time.time(), 0.9))
        engine = te.TopicEngine(ctx)
        got = engine.suggest(limit=10)
        self.assertEqual(got, ["high priority question here",
                               "low priority question here"])
        ledger = engine.load_ledger()
        self.assertEqual({e["source"] for e in ledger}, {"gap"})
        self.assertTrue(all(e["provenance"].startswith("knowledge_gap#")
                            for e in ledger))

    def test_note_picked_bumps_only_picked(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        for q in ("first question here", "second question here"):
            ctx.db.execute(
                "INSERT INTO knowledge_gaps (question, ts, status, priority)"
                " VALUES (?, ?, 'open', ?)", (q, time.time(), 0.8))
        engine = te.TopicEngine(ctx)
        engine.suggest(limit=10)
        engine.note_picked(["first question here"])
        ledger = {e["text"]: e for e in engine.load_ledger()}
        self.assertEqual(ledger["first question here"]["times_suggested"], 1)
        self.assertEqual(ledger["second question here"]["times_suggested"], 0)

    def test_ledger_is_bounded(self):
        ctx = _ctx()
        engine = te.TopicEngine(ctx)
        now = time.time()
        seed = [{"text": f"topic number {i} here", "source": "gap",
                 "provenance": f"knowledge_gap#{i}", "priority": 0.5,
                 "first_seen_at": now, "last_suggested_at": 0.0,
                 "times_suggested": 0} for i in range(te.LEDGER_CAP + 5)]
        engine.save_ledger(seed)
        engine.suggest(limit=10)
        self.assertLessEqual(len(engine.load_ledger()), te.LEDGER_CAP)

    def test_stats_counts_by_source(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, ts, status, priority)"
            " VALUES (?, ?, 'open', ?)",
            ("a gap question here", time.time(), 0.8))
        engine = te.TopicEngine(ctx)
        engine.suggest(limit=10)
        stats = engine.stats()
        self.assertEqual(stats["topics"], 1)
        self.assertEqual(stats["by_source"], {"gap": 1})


class PickTopicsTests(unittest.TestCase):
    def test_fresh_beats_just_covered_stale_goes_last(self):
        ctx = _ctx()
        rl.set_owner_topics(ctx, ["alpha", "beta", "gamma"])
        rl.record_run(ctx, {"started_at": time.time() - 10,
                            "finished_at": time.time(),
                            "topics": ["alpha"]})
        picked = rl.pick_topics(ctx, limit=3)
        # Curve, not wall (owner order 2026-11-17): alpha is last,
        # not excluded.
        self.assertEqual(picked, ["beta", "gamma", "alpha"])

    def test_only_candidate_never_starves(self):
        ctx = _ctx()
        rl.set_owner_topics(ctx, ["alpha"])
        rl.record_run(ctx, {"started_at": time.time() - 10,
                            "finished_at": time.time(),
                            "topics": ["alpha"]})
        self.assertEqual(rl.pick_topics(ctx, limit=2), ["alpha"])

    def test_engine_topic_beats_stale_owner_topic(self):
        self.assertIsNotNone(te, "topic_engine not built yet")
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        rl.set_owner_topics(ctx, ["alpha"])
        rl.record_run(ctx, {"started_at": time.time() - 10,
                            "finished_at": time.time(),
                            "topics": ["alpha"]})
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, ts, status, priority)"
            " VALUES (?, ?, 'open', ?)",
            ("why do agents forget long conversations", time.time(), 0.8))
        picked = rl.pick_topics(ctx, limit=2)
        self.assertEqual(picked[0], "why do agents forget long conversations")
        self.assertEqual(picked[1], "alpha")

    def test_engine_failure_never_breaks_pick(self):
        ctx = _ctx(_BoomDB())
        picked = rl.pick_topics(ctx, limit=2)
        self.assertTrue(picked)  # owner defaults survive a dead engine


class TickTests(unittest.TestCase):
    def test_tick_draws_engine_topic_when_owner_pool_is_stale(self):
        self.assertIsNotNone(te, "topic_engine not built yet")
        ctx = _ctx()
        _enable_research(ctx)
        _ensure_gap_tables(ctx.db)
        rl.set_owner_topics(ctx, ["alpha"])
        rl.record_run(ctx, {"started_at": time.time() - 10,
                            "finished_at": time.time(),
                            "topics": ["alpha"]})
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, ts, status, priority)"
            " VALUES (?, ?, 'open', ?)",
            ("why do agents forget long conversations", time.time(), 0.8))
        fake_result = {
            "report": {"findings": [{"claim": "a"}]},
            "claims": [{"id": "c1"}],
            "brief": "brief text",
            "tickets": [],
        }
        fake_pipeline = mock.Mock()
        fake_pipeline.run.return_value = fake_result
        fake_notifier = mock.Mock()
        fake_notifier.publish.return_value = {"delivered": True}
        with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                         fake_pipeline),
              mock.patch("nomorals.agents.research_loop.Notifier",
                         return_value=fake_notifier),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False)):
            out = rl.ResearchLoop(ctx).tick()
        self.assertTrue(out["ok"])
        self.assertIn("why do agents forget long conversations",
                      out["topics"])


class StatusTests(EngineRequired):
    def test_status_carries_engine_counts(self):
        ctx = _ctx()
        _ensure_gap_tables(ctx.db)
        ctx.db.execute(
            "INSERT INTO knowledge_gaps (question, ts, status, priority)"
            " VALUES (?, ?, 'open', ?)",
            ("a gap question here", time.time(), 0.8))
        # Status reports the ledger, not the raw harvest pool — the
        # gap lands in the ledger when the engine suggests it.
        te.TopicEngine(ctx).suggest(limit=10)
        out = rl.status(ctx)
        self.assertIn("topic_engine", out)
        self.assertEqual(out["topic_engine"]["topics"], 1)
        self.assertEqual(out["topic_engine"]["by_source"], {"gap": 1})


if __name__ == "__main__":
    unittest.main()
