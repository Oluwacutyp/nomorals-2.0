"""Universal wave: vector index — build, KNN, persistence, fail-fast."""
from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path

from nomorals.wisdom import HashEmbedBackend, VectorIndex, open_index
from nomorals.wisdom.errors import VectorStoreError
from nomorals.wisdom.vectorstore import (
    _cosine,
    _deserialize_f32,
    _serialize_f32,
)

DOCS = [
    ("k1", "the kingdom of heaven is within you and all around you"),
    ("k2", "blessed are the seekers of the inner light"),
    ("k3", "quantum field theory describes particle interactions"),
]


class SerializationTests(unittest.TestCase):
    def test_f32_round_trip(self):
        vec = [0.1, -2.5, 3.14159, 0.0, 1e-10]
        self.assertEqual(len(_serialize_f32(vec)), 5 * 4)
        back = _deserialize_f32(_serialize_f32(vec))
        for a, b in zip(vec, back):
            self.assertAlmostEqual(a, b, places=6)

    def test_cosine_unit_vectors(self):
        self.assertAlmostEqual(_cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(_cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(_cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_cosine_dim_mismatch_raises(self):
        with self.assertRaises(VectorStoreError):
            _cosine([1.0], [1.0, 2.0])


class VectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wisdom-vec-")
        self.path = os.path.join(self.tmp, "vectors.db")
        self.backend = HashEmbedBackend()
        self.index = open_index(self.path, self.backend)
        keys = [k for k, _ in DOCS]
        texts = [t for _, t in DOCS]
        self.index.build(keys, texts, self.backend)

    def tearDown(self):
        self.index.close()

    def test_engine_is_known(self):
        self.assertIn(self.index.engine, ("python", "vec0"))

    def test_count(self):
        self.assertEqual(self.index.count(), 3)

    def test_identical_query_scores_one(self):
        q = self.backend.embed_one(DOCS[0][1])
        hits = self.index.search(q, top=3)
        self.assertEqual(hits[0][0], "k1")
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)

    def test_ranking_prefers_lexical_overlap(self):
        q = self.backend.embed_one("the inner light of the kingdom")
        hits = self.index.search(q, top=3)
        keys = [k for k, _ in hits]
        # physics doc must rank last with the hashing backend
        self.assertEqual(keys[-1], "k3")
        self.assertNotEqual(keys[0], "k3")

    def test_top_limits_results(self):
        q = self.backend.embed_one("kingdom")
        self.assertEqual(len(self.index.search(q, top=2)), 2)
        self.assertEqual(len(self.index.search(q, top=0)), 0)

    def test_upsert_replaces(self):
        vec = self.backend.embed_one("completely new text about sailing")
        self.index.upsert("k1", vec)
        self.assertEqual(self.index.count(), 3)
        hits = self.index.search(vec, top=1)
        self.assertEqual(hits[0][0], "k1")

    def test_upsert_dim_mismatch_raises(self):
        with self.assertRaises(VectorStoreError):
            self.index.upsert("k1", [0.5] * 7)

    def test_query_dim_mismatch_raises(self):
        with self.assertRaises(VectorStoreError):
            self.index.search([0.5] * 7, top=3)

    def test_build_key_text_mismatch_raises(self):
        with self.assertRaises(VectorStoreError):
            self.index.build(["a", "b"], ["only one"], self.backend)

    def test_persists_across_reopen(self):
        self.index.close()
        idx2 = open_index(self.path, HashEmbedBackend())
        try:
            self.assertEqual(idx2.count(), 3)
            q = self.backend.embed_one(DOCS[1][1])
            hits = idx2.search(q, top=1)
            self.assertEqual(hits[0][0], "k2")
        finally:
            idx2.close()

    def test_backend_switch_refuses_to_serve(self):
        # A store built by hashing must not silently serve vectors
        # from a different model space.
        self.index.close()
        other = HashEmbedBackend()
        other.key = "different-model"
        with self.assertRaises(VectorStoreError):
            open_index(self.path, other)

    def test_drop_empties(self):
        self.index.drop()
        self.assertEqual(self.index.count(), 0)

    def test_no_bare_except_in_module(self):
        import nomorals.wisdom.vectorstore as mod
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.fail("bare except: found in vectorstore.py")


if __name__ == "__main__":
    unittest.main()
