"""L3 — memory: storage, four-signal recall, consolidation, forgetting.

Offline by design: embeddings come from the deterministic feature-hashing
fallback, so every assertion here is reproducible.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.memory.base import MemoryRecord, decay_factor, normalize_scores, score_memory
from nomorals.memory.embeddings import Embedder


def temp_dir() -> str:
    return tempfile.mkdtemp(prefix="nm-mem-")


class PureFunctionTests(unittest.TestCase):
    def test_decay_is_one_at_zero_and_shrinks_with_age(self):
        self.assertAlmostEqual(decay_factor(0.0, half_life_seconds=100.0), 1.0, places=6)
        self.assertAlmostEqual(decay_factor(100.0, half_life_seconds=100.0), 0.5, places=6)
        self.assertLess(decay_factor(300.0, half_life_seconds=100.0), decay_factor(100.0, half_life_seconds=100.0))

    def test_normalize_rescales_in_place_so_the_top_hit_is_one(self):
        records = [
            MemoryRecord(id="a", content="a", kind="episode", score=2.0),
            MemoryRecord(id="b", content="b", kind="episode", score=4.0),
        ]
        normalize_scores(records)
        self.assertAlmostEqual(records[0].score, 0.5, places=6)
        self.assertAlmostEqual(records[1].score, 1.0, places=6)

    def test_normalize_handles_flat_and_empty_input_without_dividing_by_zero(self):
        flat = [
            MemoryRecord(id="x", content="x", kind="episode", score=3.0),
            MemoryRecord(id="y", content="y", kind="episode", score=3.0),
        ]
        normalize_scores(flat)
        self.assertTrue(all(0.0 <= r.score <= 1.0 for r in flat))
        normalize_scores([])  # must not raise

    def test_score_memory_rewards_recent_important_records(self):
        old = MemoryRecord(id="1", content="old", kind="episode", importance=0.1, created_at=0.0)
        new = MemoryRecord(id="2", content="new", kind="episode", importance=0.9, created_at=1000.0)
        self.assertGreater(
            score_memory(new, now=1000.0),
            score_memory(old, now=1000.0),
        )

    def test_facts_decay_far_slower_than_episodes(self):
        fact = MemoryRecord(id="f", content="f", kind="fact", importance=0.5, created_at=0.0)
        episode = MemoryRecord(id="e", content="e", kind="episode", importance=0.5, created_at=0.0)
        age = 100_000.0
        self.assertGreater(score_memory(fact, now=age), score_memory(episode, now=age))


class EmbedderTests(unittest.TestCase):
    def test_fallback_embedder_is_deterministic_and_normalized(self):
        embedder = Embedder()
        first = embedder.embed_many(["hello world", "hello world"])
        self.assertEqual(first[0], first[1])
        magnitude = sum(x * x for x in first[0]) ** 0.5
        self.assertAlmostEqual(magnitude, 1.0, places=5)

    def test_similar_texts_score_higher_than_unrelated(self):
        embedder = Embedder()
        a, b, c = embedder.embed_many(
            ["the quick brown fox", "the quick brown fox jumps", "quantum chromodynamics"]
        )
        similar = sum(x * y for x, y in zip(a, b))
        unrelated = sum(x * y for x, y in zip(a, c))
        self.assertGreater(similar, unrelated)

    def test_repeated_text_is_served_from_cache(self):
        embedder = Embedder()
        embedder.embed_many(["cached text"])
        self.assertEqual(embedder.stats["cache_hits"], 0)
        embedder.embed_many(["cached text"])
        self.assertGreaterEqual(embedder.stats["cache_hits"], 1)

    def test_stemmer_collapses_morphological_variants(self):
        embedder = Embedder()
        a, b, c = embedder.embed_many(
            ["the user prefers python", "the user preferred python", "the capital of peru is lima"]
        )
        cosine = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
        self.assertGreater(cosine(a, b), cosine(a, c))

    def test_empty_input(self):
        self.assertEqual(Embedder().embed_many([]), [])


class MemoryManagerTests(unittest.TestCase):
    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.context.__enter__()
        self.memory = self.context.memory

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_remember_returns_an_id_and_persists(self):
        record_id = self.memory.remember("the sky is blue", source="test")
        self.assertTrue(record_id)
        result = self.memory.recall("sky", limit=5)
        self.assertTrue(any(r.id == record_id for r in result.records))

    def test_recall_returns_the_relevant_record_first(self):
        self.memory.remember_many(
            [
                ("The user prefers Python for backend services.", "preference"),
                ("A recipe for tomato soup with basil.", "fact"),
                ("The capital of Peru is Lima.", "fact"),
            ],
            source="test",
        )
        result = self.memory.recall("which programming language is preferred?", limit=3)
        self.assertTrue(result.records)
        self.assertIn("Python", result.records[0].content)

    def test_recall_ranks_within_a_bounded_score_range(self):
        self.memory.remember("alpha beta gamma", source="test")
        result = self.memory.recall("alpha", limit=5)
        for record in result.records:
            self.assertGreaterEqual(record.score, 0.0)
            self.assertLessEqual(record.score, 1.0)

    def test_recall_with_no_matches_returns_empty_not_an_error(self):
        result = self.memory.recall("nothing was ever stored about this", limit=3)
        self.assertEqual(result.records, [])

    def test_kind_filter_restricts_recall(self):
        self.memory.remember("a stated preference", kind="preference", source="test")
        self.memory.remember("an episode happened", kind="episode", source="test")
        result = self.memory.recall("stated", limit=5, kind="preference")
        self.assertTrue(all(r.kind == "preference" for r in result.records))

    def test_consolidation_distils_episodes_into_a_fact(self):
        self.memory.remember_many(
            [
                ("Deployed the build on Monday.", "episode"),
                ("Deployed the build again on Tuesday.", "episode"),
                ("Deployed the build on Wednesday.", "episode"),
            ],
            source="test",
        )
        report = self.memory.consolidate()
        self.assertGreaterEqual(report["episodes"], 3)

    def test_stated_facts_survive_forgetting(self):
        self.memory.remember("user API key preference", kind="fact", source="test")
        self.memory.remember("user prefers dark mode", kind="preference", source="test")
        self.memory.forget_below(0.999)
        stats = self.memory.stats_snapshot()
        self.assertEqual(stats["by_kind"].get("fact", 0), 1)
        self.assertEqual(stats["by_kind"].get("preference", 0), 1)

    def test_access_count_increases_on_recall(self):
        record_id = self.memory.remember("remember this specific detail", source="test")
        before = self.memory.stats_snapshot()["remembered"]
        self.memory.recall("specific detail", limit=5)
        self.assertGreaterEqual(self.memory.stats_snapshot()["recalls"], 1)
        self.assertGreaterEqual(before, 1)
        self.assertTrue(record_id)

    def test_tune_weights_renormalizes(self):
        self.memory.tune_weights({"semantic": 0.6, "recency": 0.2})
        weights = self.memory.stats_snapshot()["weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

    def test_ingest_document_chunks_and_indexes(self):
        text = "Sentence one about apples. " * 200
        chunks = self.memory.ingest_document(text, source="test")
        self.assertGreater(chunks, 1)
        found = self.memory.recall("apples", limit=5)
        self.assertTrue(found.records)

    def test_stats_reflects_embedder_backend(self):
        self.memory.remember("probe", source="test")
        stats = self.memory.stats_snapshot()
        self.assertEqual(stats["embedder"]["provider"], "auto")
        self.assertGreater(stats["embedder"]["dimensions"], 0)


class SupersessionTests(unittest.TestCase):
    """Facts are never deleted, only superseded — audit trail preserved."""

    def setUp(self):
        self.home = temp_dir()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        settings = Settings(home=self.home)
        self.context = build_context(settings)
        self.context.__enter__()
        self.memory = self.context.memory

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_supersede_links_old_and_new(self):
        old_id = self.memory.remember("Lagos rent is 2m", kind="fact")
        new_id = self.memory.supersede(old_id, "Lagos rent is 2.5m", kind="fact")
        self.assertTrue(new_id)
        self.assertNotEqual(old_id, new_id)
        old = self.memory.get(old_id)
        new = self.memory.get(new_id)
        # Old record kept, marked superseded.
        self.assertIsNotNone(old)
        self.assertEqual(old.metadata.get("superseded_by"), new_id)
        self.assertIn("superseded_at", old.metadata)
        # New record links back.
        self.assertEqual(new.metadata.get("supersedes"), old_id)
        self.assertEqual(new.content, "Lagos rent is 2.5m")

    def test_supersede_unknown_id_returns_empty(self):
        self.assertEqual(self.memory.supersede("nope", "x"), "")
        self.assertEqual(self.memory.supersede("", "x"), "")

    def test_superseded_excluded_from_default_recall(self):
        old_id = self.memory.remember("the sky is green", kind="fact", importance=0.9)
        self.memory.supersede(old_id, "the sky is blue", kind="fact", importance=0.9)
        hits = self.memory.recall("sky", limit=10)
        ids = [r.id for r in hits.records]
        self.assertNotIn(old_id, ids)

    def test_include_superseded_brings_back_history(self):
        old_id = self.memory.remember("the sky is green", kind="fact", importance=0.9)
        self.memory.supersede(old_id, "the sky is blue", kind="fact", importance=0.9)
        hits = self.memory.recall("sky", limit=10, include_superseded=True)
        ids = [r.id for r in hits.records]
        self.assertIn(old_id, ids)

    def test_supersession_chain_oldest_to_newest(self):
        id1 = self.memory.remember("v1", kind="fact")
        id2 = self.memory.supersede(id1, "v2", kind="fact")
        id3 = self.memory.supersede(id2, "v3", kind="fact")
        chain = self.memory.supersession_chain(id3)
        self.assertEqual([r.id for r in chain], [id1, id2, id3])
        # Chain from the middle also resolves fully.
        chain2 = self.memory.supersession_chain(id2)
        self.assertEqual([r.id for r in chain2], [id1, id2, id3])

    def test_chain_unknown_id_is_empty(self):
        self.assertEqual(self.memory.supersession_chain("nope"), [])

    def test_supersede_never_raises_on_garbage(self):
        self.assertEqual(self.memory.supersede(None, None), "")
        self.assertEqual(self.memory.supersession_chain(None), [])


if __name__ == "__main__":
    unittest.main()
