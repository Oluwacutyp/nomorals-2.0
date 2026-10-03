"""Universal wave: embedding backends — contract, selection, fallbacks.

All tests run without any third-party package installed: the hashing
backend is the always-available baseline, and the neural backends are
exercised through their availability/selection logic (forced via
monkeypatching so the tests are deterministic on any machine).
"""
from __future__ import annotations

import math
import unittest
from unittest import mock

from nomorals.wisdom import (
    BACKENDS,
    EmbeddingBackend,
    FastEmbedBackend,
    HashEmbedBackend,
    OllamaBackend,
    SentenceTransformersBackend,
    auto_backend,
    available_backends,
    embed_texts,
)
from nomorals.wisdom.errors import EmbeddingError


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


class BackendRegistryTests(unittest.TestCase):
    def test_four_providers_registered(self):
        self.assertEqual(set(BACKENDS), {
            "fastembed", "sentence-transformers", "ollama", "hashing"})

    def test_available_never_raises(self):
        for key, cls in BACKENDS.items():
            try:
                result = cls.available()
            except Exception as exc:  # pragma: no cover
                self.fail(f"{key}.available() raised: {exc}")
            self.assertIsInstance(result, bool, key)

    def test_hashing_always_available(self):
        self.assertTrue(HashEmbedBackend.available())

    def test_unknown_backend_name_rejected(self):
        with self.assertRaises(EmbeddingError):
            auto_backend("definitely-not-a-backend")

    def test_forced_backend_unavailable_raises(self):
        with mock.patch.object(
                FastEmbedBackend, "available",
                classmethod(lambda cls: False)):
            with self.assertRaises(EmbeddingError):
                auto_backend("fastembed")
        with mock.patch.object(
                SentenceTransformersBackend, "available",
                classmethod(lambda cls: False)):
            with self.assertRaises(EmbeddingError):
                auto_backend("sentence-transformers")

    def test_auto_prefers_hashing_last_resort(self):
        # With everything else unavailable, auto() must land on hashing —
        # never raise, never return None.
        with mock.patch.object(FastEmbedBackend, "available",
                               classmethod(lambda cls: False)), \
             mock.patch.object(SentenceTransformersBackend, "available",
                               classmethod(lambda cls: False)), \
             mock.patch.object(OllamaBackend, "available",
                               classmethod(lambda cls: False)):
            backend = auto_backend()
        self.assertIsInstance(backend, HashEmbedBackend)


class HashBackendTests(unittest.TestCase):
    def setUp(self):
        self.b = HashEmbedBackend()

    def test_dim(self):
        self.assertEqual(self.b.dim, 512)
        self.assertEqual(len(self.b.embed_one("hello")), 512)

    def test_deterministic(self):
        a = self.b.embed_one("the kingdom of heaven")
        b = self.b.embed_one("the kingdom of heaven")
        self.assertEqual(a, b)

    def test_unit_normalized(self):
        for text in ["hello world", "x", "a much longer sentence " * 20]:
            self.assertAlmostEqual(_norm(self.b.embed_one(text)), 1.0,
                                   places=6, msg=text[:20])

    def test_empty_text_gives_zero_vector(self):
        vec = self.b.embed_one("")
        self.assertEqual(vec, [0.0] * 512)

    def test_related_texts_score_higher_than_unrelated(self):
        # Hashing is lexical, not neural: near-identical wording must
        # still outscore unrelated wording — the vector path is usable.
        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))

        q = self.b.embed_one("kingdom of heaven within you")
        near = self.b.embed_one("the kingdom of heaven is within you")
        far = self.b.embed_one("quantum chromodynamics lattice gauge")
        self.assertGreater(cos(q, near), cos(q, far))

    def test_batch_matches_single(self):
        texts = ["alpha", "beta gamma", "delta"]
        batch = self.b.embed(texts)
        for text, vec in zip(texts, batch):
            self.assertEqual(vec, self.b.embed_one(text))

    def test_embed_texts_normalizes(self):
        vecs = embed_texts(self.b, ["hello", "world"])
        self.assertEqual(len(vecs), 2)
        for v in vecs:
            self.assertAlmostEqual(_norm(v), 1.0, places=6)


class NeuralBackendShapeTests(unittest.TestCase):
    """The neural backends' construction/availability contract without
    requiring the packages: they must fail fast with EmbeddingError,
    never with ImportError leaking out."""

    def test_fastembed_init_fails_fast_when_missing(self):
        with mock.patch.object(FastEmbedBackend, "available",
                               classmethod(lambda cls: False)):
            with self.assertRaises(EmbeddingError):
                FastEmbedBackend()

    def test_sentence_transformers_init_fails_fast_when_missing(self):
        with mock.patch.object(SentenceTransformersBackend, "available",
                               classmethod(lambda cls: False)):
            with self.assertRaises(EmbeddingError):
                SentenceTransformersBackend()

    def test_default_models_documented(self):
        self.assertEqual(FastEmbedBackend.default_model(),
                         "BAAI/bge-small-en-v1.5")
        self.assertEqual(SentenceTransformersBackend.default_model(),
                         "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(OllamaBackend.default_model(), "nomic-embed-text")


class OllamaBackendTests(unittest.TestCase):
    def test_unreachable_host_fails_fast(self):
        b = OllamaBackend(host="http://127.0.0.1:1", timeout=0.5)
        with self.assertRaises(EmbeddingError) as ctx:
            b.embed_one("hello")
        self.assertIn("127.0.0.1:1", str(ctx.exception))

    def test_no_bare_except_in_module(self):
        # The wave forbids bare `except:` in new code; enforce it here.
        import ast
        from pathlib import Path
        import nomorals.wisdom.embeddings as mod
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.fail("bare except: found in embeddings.py")


if __name__ == "__main__":
    unittest.main()
