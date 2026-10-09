"""Memory health, explainability, additive consolidation, drift repair.

Layers on top of the trusted recall path — every test here asserts the new
introspection/repair surface WITHOUT changing existing recall behaviour.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import time
import unittest

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.memory.base import (
    TRUSTED,
    UNTRUSTED,
    MemoryKind,
    MemoryRecord,
    score_memory,
)
from nomorals.memory.embeddings import Embedder
from nomorals.memory.manager import _explain_score


def temp_dir() -> str:
    return tempfile.mkdtemp(prefix="nm-memhealth-")


class ExplainParityTests(unittest.TestCase):
    """_explain_score must be bit-identical to the trusted score_memory."""

    def test_parity_with_score_memory_across_random_inputs(self):
        rng = random.Random(20261009)
        kinds = ["episode", "fact", "preference", "skill", "lesson"]
        for _ in range(300):
            record = MemoryRecord(
                id="x",
                kind=rng.choice(kinds),
                content="c",
                importance=rng.random(),
                access_count=rng.randint(0, 100),
                created_at=0.0,
                trust=rng.choice([TRUSTED, UNTRUSTED]),
            )
            sem = rng.uniform(-1.5, 1.5)
            lex = rng.uniform(-0.5, 1.5)
            weights = {
                "recency": rng.random(),
                "importance": rng.random(),
                "semantic": rng.random(),
                "lexical": rng.random(),
            }
            expected = score_memory(
                record, semantic=sem, lexical=lex, weights=weights, now=1000.0
            )
            got, contributions = _explain_score(
                record, semantic=sem, lexical=lex, weights=weights, now=1000.0
            )
            self.assertAlmostEqual(got, expected, places=9)
            self.assertAlmostEqual(sum(contributions.values()), got, places=6)


class ExplainRecallTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.memory = self.context.memory
        self.memory.embedder = Embedder(provider="hashing", dimensions=512)
        self.memory.consolidate_every = 10 ** 9

    def test_explain_attaches_traceable_breakdown(self):
        self.memory.remember(
            "The user prefers dark mode interfaces.", kind="fact",
            importance=0.9, source="user:chat",
        )
        self.memory.remember(
            "The user once tried light mode.", kind="episode",
            importance=0.2, source="user:chat",
        )
        result = self.memory.recall("dark mode", limit=2, explain=True)
        self.assertTrue(result.records)
        top = result.records[0]
        expl = top.explanation
        self.assertTrue(expl, "explain=True must attach an explanation")
        self.assertIn("lane", expl)
        self.assertIn("signals", expl)
        self.assertSetEqual(
            set(expl["signals"]), {"recency", "importance", "semantic", "lexical"}
        )
        # Contributions sum to the pre-adjustment score: the trace is exact.
        self.assertAlmostEqual(
            sum(expl["signals"].values()), expl["score_before_adjustments"], places=4
        )
        self.assertIn(expl["trust"], (TRUSTED, UNTRUSTED))

    def test_explain_records_adjustments(self):
        self.memory.remember(
            "The owner drinks espresso.", kind="fact", importance=0.9,
            source="user:chat", origin="chat:tg:1",
        )
        self.memory.remember(
            "The owner drinks tea.", kind="fact", importance=0.9,
            source="web:scrape", origin="chat:tg:2",  # untrusted, other session
        )
        result = self.memory.recall(
            "what does the owner drink", limit=5, origin="chat:tg:1", explain=True
        )
        by_content = {r.content: r for r in result.records}
        espresso = by_content.get("The owner drinks espresso.")
        self.assertIsNotNone(espresso)
        self.assertIn("session_boost:+0.15", espresso.explanation["adjustments"])
        tea = by_content.get("The owner drinks tea.")
        if tea is not None:  # untrusted + cross-session: both downranks traced
            self.assertIn("untrusted_downrank:x0.5", tea.explanation["adjustments"])
            self.assertIn("cross_session_downrank:x0.5", tea.explanation["adjustments"])

    def test_explain_false_is_unchanged_behaviour(self):
        self.memory.remember("alpha beta gamma", source="test")
        result = self.memory.recall("alpha", limit=3)
        for record in result.records:
            self.assertEqual(record.explanation, {})


class DegradedRecallTests(unittest.TestCase):
    """Recall never raises: a dead semantic lane degrades to lexical."""

    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.memory = self.context.memory
        self.memory.embedder = Embedder(provider="hashing", dimensions=512)
        self.memory.consolidate_every = 10 ** 9

    def test_broken_vector_index_falls_back_to_lexical(self):
        self.memory.remember(
            "The launch code is 8473.", kind="fact", importance=0.9, source="test"
        )

        def boom(*_a, **_k):
            raise RuntimeError("index exploded")

        self.memory.semantic.search = boom
        result = self.memory.recall("launch code", limit=3)
        self.assertTrue(
            any("8473" in r.content for r in result.records),
            "lexical lane must still find the record",
        )
        health = self.memory.health()
        self.assertGreaterEqual(health["degraded_recalls"], 1)
        self.assertTrue(
            any(e["lane"] == "semantic" for e in health["recent_events"])
        )

    def test_dimension_drift_degrades_then_repairs(self):
        self.memory.remember(
            "The user prefers dark mode.", kind="fact", source="user:chat"
        )
        # Simulate an embedding provider switch (hashing 512 -> qwen3 1024).
        self.memory.embedder = Embedder(provider="hashing", dimensions=1024)
        self.memory.remember(
            "The user dislikes light mode.", kind="fact", source="user:chat"
        )
        # Recall must NOT raise despite the mixed-dimension index.
        result = self.memory.recall("dark mode", limit=3)
        self.assertTrue(
            any("dark mode" in r.content for r in result.records),
            "lexical fallback must keep recall working",
        )
        health = self.memory.health()
        self.assertTrue(health["dimension_drift"])
        self.assertTrue(
            any("dimension drift" in p for p in health["problems"])
        )
        # Repair is additive: both records survive, vectors unified.
        repair = self.memory.repair_embeddings(dry_run=True)
        self.assertEqual(repair["drifted"], 1)
        repair = self.memory.repair_embeddings()
        self.assertEqual(repair["repaired"], 1)
        self.assertEqual(repair["failed"], 0)
        health = self.memory.health()
        self.assertFalse(health["dimension_drift"])
        self.assertEqual(health["records"], 2)
        result = self.memory.recall("dark mode", limit=3)
        self.assertTrue(any("dark mode" in r.content for r in result.records))


class AdditiveConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings, with_router=False)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.memory = self.context.memory
        self.memory.embedder = Embedder(provider="hashing", dimensions=512)
        self.memory.consolidate_every = 10 ** 9

    def test_additive_consolidation_preserves_raw_episodes(self):
        episodes = [
            "Deployed the build on Monday morning.",
            "Deployed the build again on Tuesday afternoon.",
            "Rolled back the deploy on Wednesday night.",
            "Fixed the deploy pipeline on Thursday.",
            "Deploy pipeline is green on Friday.",
            "Weekend deploy freeze is in effect.",
        ]
        ids = self.memory.remember_many(
            [(e, MemoryKind.EPISODE) for e in episodes], source="test"
        )
        before = self.memory.repo.count()
        report = self.memory.consolidate_additive(min_episodes=5, batch=50)
        self.assertEqual(report["mode"], "additive")
        self.assertEqual(report["forgotten"], 0, "additive must never forget")
        self.assertGreaterEqual(report["summaries"], 1)
        # Raw episodes are untouched and still recallable.
        self.assertEqual(
            self.memory.repo.count(), before + report["summaries"]
        )
        for record_id in ids:
            self.assertIsNotNone(self.memory.get(record_id))
        result = self.memory.recall("deploy pipeline", limit=10)
        self.assertTrue(
            any("Deployed the build" in r.content for r in result.records)
        )
        # The distilled fact links back to its sources.
        facts = [
            r for r in result.records
            if (r.metadata or {}).get("additive")
        ]
        self.assertTrue(facts)
        self.assertEqual(
            len(facts[0].metadata["distilled_ids"]), len(episodes)
        )
        # Re-run is idempotent: nothing new distilled, nothing deleted.
        report2 = self.memory.consolidate_additive(min_episodes=5, batch=50)
        self.assertEqual(report2["summaries"], 0)
        self.assertGreaterEqual(report2["skipped_distilled"], len(episodes))
        self.assertEqual(self.memory.repo.count(), before + report["summaries"])
        # health() surfaces the consolidation.
        health = self.memory.health()
        self.assertEqual(
            health["last_consolidation_report"]["mode"], "additive"
        )


class HealthParityTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.memory = self.context.memory
        self.memory.embedder = Embedder(provider="hashing", dimensions=512)
        self.memory.consolidate_every = 10 ** 9

    def test_health_is_clean_on_a_healthy_store(self):
        self.memory.remember("healthy record one", source="test")
        self.memory.remember("healthy record two", source="test")
        health = self.memory.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["problems"], [])
        self.assertEqual(health["records"], 2)
        self.assertEqual(health["vectors"], 2)
        self.assertEqual(health["parity"]["records_missing_vectors"], [])
        self.assertFalse(health["dimension_drift"])

    def test_health_detects_missing_vector(self):
        record_id = self.memory.remember("orphaned record", source="test")
        self.memory.db.execute(
            "DELETE FROM embeddings WHERE owner_id = ? AND owner_type = 'memory'",
            (record_id,),
        )
        health = self.memory.health()
        self.assertFalse(health["ok"])
        self.assertIn(record_id, health["parity"]["records_missing_vectors"])
        # Repair restores the vector without touching the record.
        repair = self.memory.repair_embeddings()
        self.assertEqual(repair["repaired"], 1)
        health = self.memory.health()
        self.assertEqual(health["parity"]["records_missing_vectors"], [])

    def test_health_never_raises(self):
        # Even with the DB in a weird state, health() reports, not raises.
        health = self.memory.health()
        self.assertIn("ok", health)
        self.assertIn("problems", health)


class RecallPerformanceTests(unittest.TestCase):
    def test_recall_stays_fast_at_thousands_of_records(self):
        home = temp_dir()
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        settings = Settings(home=home)
        context = build_context(settings, with_router=False)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        memory = context.memory
        memory.embedder = Embedder(provider="hashing", dimensions=512)
        memory.consolidate_every = 10 ** 9
        memory.remember(
            "The user prefers dark mode interfaces.", kind="fact",
            importance=0.9, source="user:chat",
        )
        memory.remember_many(
            [
                (f"Routine logistics note {i}: scheduling and follow-ups.", "episode")
                for i in range(2000)
            ],
            source="test",
        )
        memory.recall("dark mode", limit=5)  # warm caches
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            result = memory.recall("dark mode interface", limit=12)
            timings.append((time.perf_counter() - started) * 1000)
        mean_ms = sum(timings) / len(timings)
        self.assertTrue(
            any("dark mode" in r.content for r in result.records),
            "recall must still find the record",
        )
        self.assertLess(
            mean_ms, 2000.0, f"recall too slow at 2k records: {mean_ms:.1f}ms mean"
        )


if __name__ == "__main__":
    unittest.main()
