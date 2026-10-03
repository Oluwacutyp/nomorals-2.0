"""Universal wave — MemoryManager over the pluggable vector backends.

Pins the manager contract across backends: remember/recall/update/forget,
hybrid semantic+lexical recall, and stats reporting. The default ("auto")
path is exercised wherever the environment allows; explicit backend pins are
exercised wherever the backend is installed.
"""

from __future__ import annotations

import unittest

from nomorals.memory.manager import MemoryManager
from nomorals.memory.vector_backends import SqliteVecBackend, USearchBackend
from nomorals.storage.db import Database

SQLITE_VEC_OK, _ = SqliteVecBackend.available()
USEARCH_OK, _ = USearchBackend.available()


class _Context:
    """Minimal stand-in for AgentContext: the manager only needs db,
    settings, and router, all optional."""

    def __init__(self, db: Database):
        self.db = db
        self.settings = None
        self.router = None


def _manager(vector_backend: str | None = None) -> tuple[MemoryManager, Database]:
    db = Database(":memory:")
    db.migrate()
    context = _Context(db)
    return MemoryManager(context, vector_backend=vector_backend), db


class ManagerBackendSelectionTests(unittest.TestCase):
    def test_default_auto_selects_best_available(self):
        memory, db = _manager()
        try:
            expected = "sqlite-vec" if SQLITE_VEC_OK else "usearch" if USEARCH_OK else "legacy"
            self.assertEqual(memory.semantic.name, expected)
        finally:
            db.close()

    def test_explicit_legacy_pin(self):
        memory, db = _manager(vector_backend="legacy")
        try:
            self.assertEqual(memory.semantic.name, "legacy")
            record_id = memory.remember("legacy pinned memory", source="test")
            result = memory.recall("legacy pinned", limit=5)
            self.assertTrue(any(r.id == record_id for r in result.records))
        finally:
            db.close()

    def test_unknown_backend_name_rejected(self):
        db = Database(":memory:")
        db.migrate()
        try:
            with self.assertRaises(ValueError):
                MemoryManager(_Context(db), vector_backend="nope")
        finally:
            db.close()

    def test_stats_snapshot_names_active_backend(self):
        memory, db = _manager(vector_backend="legacy")
        try:
            memory.remember("stats probe", source="test")
            stats = memory.stats_snapshot()
            self.assertEqual(stats["vector_backend"], "legacy")
            self.assertEqual(stats["vectors"], 1)
        finally:
            db.close()


class ManagerLifecycleTests(unittest.TestCase):
    """remember → recall → update → forget, on the default backend."""

    def setUp(self):
        self.memory, self.db = _manager()

    def tearDown(self):
        self.db.close()

    def test_remember_recall_update_forget(self):
        record_id = self.memory.remember(
            "the staging server password is hunter2 staging", kind="fact", source="test"
        )
        result = self.memory.recall("staging server password", limit=5)
        self.assertTrue(any(r.id == record_id for r in result.records))

        self.memory.update(record_id, content="the staging server password is hunter3 staging")
        result = self.memory.recall("staging server password", limit=5)
        hit = next(r for r in result.records if r.id == record_id)
        self.assertIn("hunter3", hit.content)

        self.memory.forget(record_id)
        self.assertIsNone(self.memory.get(record_id))
        result = self.memory.recall("staging server password", limit=5)
        self.assertFalse(any(r.id == record_id for r in result.records))

    def test_hybrid_recall_lexical_lane_catches_exact_identifiers(self):
        """A rare identifier FTS can match even when the semantic pass is weak."""
        record_id = self.memory.remember(
            "deployment runbook mentions token ZK-9917-QWERTY", kind="fact", source="test"
        )
        result = self.memory.recall("ZK-9917-QWERTY", limit=5)
        self.assertTrue(any(r.id == record_id for r in result.records))

    def test_semantic_lane_ranks_shared_topic_first(self):
        self.memory.remember("the user prefers python for backend services", kind="preference")
        self.memory.remember("a recipe for tomato soup with basil", kind="fact")
        result = self.memory.recall("which programming language is preferred", limit=2)
        self.assertTrue(result.records)
        self.assertIn("python", result.records[0].content.lower())

    def test_remember_many_bulk_path(self):
        ids = self.memory.remember_many(
            [(f"bulk memory number {i} about starships", "fact") for i in range(10)],
            source="test",
        )
        self.assertEqual(len(ids), 10)
        result = self.memory.recall("starships", limit=20)
        found = {r.id for r in result.records}
        self.assertTrue(set(ids) <= found)


@unittest.skipUnless(SQLITE_VEC_OK, "sqlite-vec not installed")
class ManagerSqliteVecTests(unittest.TestCase):
    def test_full_lifecycle_on_sqlite_vec(self):
        memory, db = _manager(vector_backend="sqlite-vec")
        try:
            self.assertEqual(memory.semantic.name, "sqlite-vec")
            record_id = memory.remember("sqlite-vec backed memory about vector search", source="test")
            result = memory.recall("vector search", limit=5)
            self.assertTrue(any(r.id == record_id for r in result.records))
            self.assertEqual(memory.stats_snapshot()["vector_backend"], "sqlite-vec")

            memory.update(record_id, content="updated content about databases")
            result = memory.recall("databases", limit=5)
            self.assertTrue(any(r.id == record_id for r in result.records))

            memory.forget(record_id)
            result = memory.recall("vector search", limit=5)
            self.assertFalse(any(r.id == record_id for r in result.records))
        finally:
            db.close()

    def test_recall_parity_with_legacy_backend(self):
        """Same corpus through legacy and sqlite-vec: same top hit."""
        corpus = [
            ("the capital of france is paris", "fact"),
            ("python list comprehensions are concise", "fact"),
            ("the user prefers dark mode interfaces", "preference"),
            ("deploy checklist: tests, changelog, tag", "fact"),
        ]
        legacy, legacy_db = _manager(vector_backend="legacy")
        vec, vec_db = _manager(vector_backend="sqlite-vec")
        try:
            legacy.remember_many(corpus, source="test")
            vec.remember_many(corpus, source="test")
            for query in ["french capital city", "pythonic iteration", "dark ui theme"]:
                want = legacy.recall(query, limit=3).records
                got = vec.recall(query, limit=3).records
                self.assertTrue(want and got)
                self.assertEqual(got[0].content, want[0].content)
        finally:
            legacy_db.close()
            vec_db.close()


@unittest.skipUnless(USEARCH_OK, "usearch not installed")
class ManagerUSearchTests(unittest.TestCase):
    def test_full_lifecycle_on_usearch(self):
        memory, db = _manager(vector_backend="usearch")
        try:
            self.assertEqual(memory.semantic.name, "usearch")
            record_id = memory.remember("usearch backed memory about ann search", source="test")
            result = memory.recall("ann search", limit=5)
            self.assertTrue(any(r.id == record_id for r in result.records))
            memory.forget(record_id)
            result = memory.recall("ann search", limit=5)
            self.assertFalse(any(r.id == record_id for r in result.records))
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
