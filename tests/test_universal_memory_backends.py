"""Universal wave — vector-search backends.

Every new vector backend gets real tests here: availability reporting,
fail-fast selection, CRUD, search ordering/scores, deletion, and parity with
the legacy store (the referee). Backends that are not installed are skipped,
never faked — the selection tests pin the skip logic itself.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest import mock

from nomorals.core.errors import ValidationError
from nomorals.memory import vector_backends as vb
from nomorals.memory.vector_backends import (
    LegacyStoreBackend,
    SqliteVecBackend,
    USearchBackend,
    available_backends,
    select_vector_backend,
)
from nomorals.storage.db import Database

SQLITE_VEC_OK, _ = SqliteVecBackend.available()
USEARCH_OK, _ = USearchBackend.available()


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _unit(i: int, dim: int = 8) -> list[float]:
    """Deterministic pseudo-random unit vector from an integer seed."""
    import random

    rng = random.Random(10_000 + i)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class AvailabilityTests(unittest.TestCase):
    def test_report_lists_every_backend_with_reason_and_hint(self):
        report = available_backends()
        self.assertEqual([r["name"] for r in report], ["sqlite-vec", "usearch", "legacy"])
        for entry in report:
            self.assertIn("available", entry)
            self.assertTrue(entry["reason"])
            self.assertIn("install", entry)

    def test_legacy_is_always_available(self):
        ok, reason = LegacyStoreBackend.available()
        self.assertTrue(ok)
        self.assertTrue(reason)

    def test_auto_selects_first_available_in_quality_order(self):
        db = _db()
        try:
            backend = select_vector_backend(db)
            expected = "sqlite-vec" if SQLITE_VEC_OK else "usearch" if USEARCH_OK else "legacy"
            self.assertEqual(backend.name, expected)
        finally:
            db.close()

    def test_named_legacy_backend_selects(self):
        db = _db()
        try:
            backend = select_vector_backend(db, preference="legacy")
            self.assertIsInstance(backend, LegacyStoreBackend)
        finally:
            db.close()

    def test_unknown_backend_name_fails_fast(self):
        db = _db()
        try:
            with self.assertRaises(ValueError):
                select_vector_backend(db, preference="nope")
        finally:
            db.close()

    def test_named_but_missing_backend_fails_fast_with_install_hint(self):
        db = _db()
        try:
            with (
                mock.patch.object(vb, "_sqlite_vec", None),
                self.assertRaisesRegex(RuntimeError, "pip install sqlite-vec"),
            ):
                select_vector_backend(db, preference="sqlite-vec")
            with (
                mock.patch.object(vb, "_usearch_index", None),
                self.assertRaisesRegex(RuntimeError, "pip install usearch"),
            ):
                select_vector_backend(db, preference="usearch")
        finally:
            db.close()

    def test_missing_backend_never_silently_downgrades(self):
        """A named request must raise, not quietly return the legacy store."""
        db = _db()
        try:
            with mock.patch.object(vb, "_sqlite_vec", None):
                try:
                    select_vector_backend(db, preference="sqlite-vec")
                except RuntimeError:
                    pass
                else:
                    self.fail("expected RuntimeError, got a backend")
        finally:
            db.close()


class LegacyBackendTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()

    def tearDown(self):
        self.db.close()

    def test_put_search_delete_owner_roundtrip(self):
        backend = LegacyStoreBackend(self.db)
        vectors = [_unit(i) for i in range(5)]
        ids = backend.put_many([(v, f"owner-{i}") for i, v in enumerate(vectors)])
        self.assertEqual(len(ids), 5)
        self.assertEqual(backend.count(), 5)

        hits = backend.search(_unit(2), limit=3)
        self.assertEqual(hits[0].owner_id, "owner-2")
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        self.assertEqual(hits[0].owner_type, "memory")

        removed = backend.delete_owner("owner-2")
        self.assertEqual(removed, 1)
        self.assertEqual(backend.count(), 4)
        hits = backend.search(_unit(2), limit=5)
        self.assertNotIn("owner-2", [h.owner_id for h in hits])

    def test_empty_vector_fails_fast(self):
        backend = LegacyStoreBackend(self.db)
        with self.assertRaises(ValidationError):
            backend.put([], "owner-x")

    def test_search_empty_store_returns_empty(self):
        backend = LegacyStoreBackend(self.db)
        self.assertEqual(backend.search(_unit(0), limit=5), [])


@unittest.skipUnless(SQLITE_VEC_OK, "sqlite-vec not installed")
class SqliteVecBackendTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()

    def tearDown(self):
        self.db.close()

    def test_put_search_scores_are_cosine(self):
        backend = SqliteVecBackend(self.db)
        vectors = [_unit(i) for i in range(6)]
        backend.put_many([(v, f"rec-{i}") for i, v in enumerate(vectors)])
        self.assertEqual(backend.count(), 6)

        hits = backend.search(_unit(4), limit=3)
        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].owner_id, "rec-4")
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        # every score equals the true cosine against the stored vector
        for hit in hits:
            idx = int(hit.owner_id.split("-")[1])
            expected = sum(a * b for a, b in zip(_unit(4), vectors[idx], strict=True))
            self.assertAlmostEqual(hit.score, expected, places=5)

    def test_search_order_matches_legacy_exactly(self):
        """Parity with the legacy store: same corpus, same top-k order."""
        legacy = LegacyStoreBackend(self.db)
        vectors = [_unit(i, dim=16) for i in range(20)]
        legacy.put_many([(v, f"p-{i}") for i, v in enumerate(vectors)])

        vec_backend = SqliteVecBackend(self.db)
        vec_backend.put_many([(v, f"p-{i}") for i, v in enumerate(vectors)])

        query = _unit(999, dim=16)
        want = [h.owner_id for h in legacy.search(query, limit=8)]
        got = [h.owner_id for h in vec_backend.search(query, limit=8)]
        self.assertEqual(got, want)

    def test_delete_owner_and_delete_by_id(self):
        backend = SqliteVecBackend(self.db)
        vectors = [_unit(i) for i in range(4)]
        ids = backend.put_many([(v, f"o-{i}") for i, v in enumerate(vectors)])
        self.assertEqual(backend.delete_owner("o-1"), 1)
        self.assertEqual(backend.count(), 3)
        self.assertEqual(backend.delete(ids[0]), 1)
        self.assertEqual(backend.count(), 2)
        self.assertEqual(backend.delete("not-a-rowid"), 0)

    def test_dimension_drift_gets_a_fresh_table(self):
        """A provider change that alters dims degrades into a new table."""
        backend = SqliteVecBackend(self.db)
        backend.put(_unit(0, dim=8), "eight")
        backend.put(_unit(1, dim=12), "twelve")
        self.assertEqual(backend.count(), 2)
        hits8 = backend.search(_unit(0, dim=8), limit=5)
        hits12 = backend.search(_unit(1, dim=12), limit=5)
        self.assertEqual([h.owner_id for h in hits8], ["eight"])
        self.assertEqual([h.owner_id for h in hits12], ["twelve"])
        tables = [name for name, _ in backend._tables()]
        self.assertIn("memory_vec_8", tables)
        self.assertIn("memory_vec_12", tables)

    def test_search_unknown_dimension_returns_empty(self):
        backend = SqliteVecBackend(self.db)
        backend.put(_unit(0, dim=8), "eight")
        self.assertEqual(backend.search(_unit(0, dim=32), limit=5), [])

    def test_tables_live_inside_the_same_database_file(self):
        path = os.path.join(tempfile.mkdtemp(prefix="nm-vec-"), "m.db")
        db = Database(path)
        db.migrate()
        try:
            backend = SqliteVecBackend(db)
            backend.put(_unit(0), "x")
            names = db.tables()
            self.assertTrue(any(n.startswith("memory_vec_") for n in names))
        finally:
            db.close()


@unittest.skipUnless(USEARCH_OK, "usearch not installed")
class USearchBackendTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()

    def tearDown(self):
        self.db.close()

    def test_put_search_delete_owner_roundtrip(self):
        backend = USearchBackend(self.db)
        vectors = [_unit(i) for i in range(10)]
        ids = backend.put_many([(v, f"u-{i}") for i, v in enumerate(vectors)])
        self.assertEqual(len(ids), 10)
        self.assertEqual(backend.count(), 10)

        hits = backend.search(_unit(7), limit=3)
        self.assertEqual(hits[0].owner_id, "u-7")
        self.assertAlmostEqual(hits[0].score, 1.0, places=4)

        self.assertEqual(backend.delete_owner("u-7"), 1)
        self.assertEqual(backend.count(), 9)
        self.assertNotIn("u-7", [h.owner_id for h in backend.search(_unit(7), limit=10)])

    def test_delete_by_record_id(self):
        backend = USearchBackend(self.db)
        ids = backend.put_many([(_unit(i), f"d-{i}") for i in range(3)])
        self.assertEqual(backend.delete(ids[1]), 1)
        self.assertEqual(backend.count(), 2)
        self.assertEqual(backend.delete("bogus"), 0)

    def test_top1_matches_legacy_on_small_corpus(self):
        """On a small corpus HNSW should agree with the exact referee."""
        legacy = LegacyStoreBackend(self.db)
        vectors = [_unit(i, dim=16) for i in range(30)]
        legacy.put_many([(v, f"q-{i}") for i, v in enumerate(vectors)])
        ann = USearchBackend(self.db)
        ann.put_many([(v, f"q-{i}") for i, v in enumerate(vectors)])
        query = _unit(4242, dim=16)
        want = legacy.search(query, limit=1)[0].owner_id
        got = ann.search(query, limit=1)[0].owner_id
        self.assertEqual(got, want)

    def test_index_persists_across_instances(self):
        home = tempfile.mkdtemp(prefix="nm-usearch-")
        path = os.path.join(home, "m.db")
        db = Database(path)
        db.migrate()
        try:
            first = USearchBackend(db)
            first.put_many([(_unit(i), f"s-{i}") for i in range(5)])
            index_file = first._index_file(8)
            self.assertTrue(index_file and os.path.exists(index_file))
        finally:
            db.close()
        # Fresh process-equivalent: new Database, new backend, same files.
        db2 = Database(path)
        db2.migrate()
        try:
            second = USearchBackend(db2)
            self.assertEqual(second.count(), 5)
            hits = second.search(_unit(3), limit=2)
            self.assertEqual(hits[0].owner_id, "s-3")
        finally:
            db2.close()


if __name__ == "__main__":
    unittest.main()
