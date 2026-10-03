"""Universal wave: hybrid search (keyword + semantic RRF fusion) and the
end-to-end semantic path on CanonCorpus: build index → semantic ask →
hybrid ask, with provenance preserved and fail-fast behavior intact."""
from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.wisdom import (
    CanonCorpus,
    ManifestEntry,
    WisdomKeeper,
    fuse_hits,
    reciprocal_rank_fusion,
)
from nomorals.wisdom.errors import CorpusError


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-hybrid-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings)


def _entry(slug="gospel-of-thomas", **kw):
    d = {
        "slug": slug,
        "title": "Gospel of Thomas",
        "tradition": "christian-gnostic",
        "canon_status": "gnostic",
        "translator": "Patterson & Robinson",
        "source_url": "http://gnosis.org/naghamm/gth_pat_rob.htm",
        "license": "public-domain",
    }
    d.update(kw)
    return ManifestEntry.from_dict(d)


TEXT = (
    "# The Kingdom\n\n"
    + "The kingdom of heaven is within you and all around you. " * 40
    + "\n\n# Sayings\n\n"
    + "Blessed are the seekers of the inner light and the quiet mind. " * 40
)


def _corpus_with_text():
    ctx = _ctx()
    corpus = CanonCorpus(ctx)
    corpus.register(_entry())
    corpus.ingest_text("gospel-of-thomas", TEXT)
    return corpus


class RRFFusionTests(unittest.TestCase):
    def test_exact_scores(self):
        fused = reciprocal_rank_fusion([["a"], ["a"]])
        self.assertAlmostEqual(fused["a"], 2 / 61)

    def test_both_lists_boost_shared_doc(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        self.assertGreater(fused["b"], fused["a"])
        self.assertGreater(fused["b"], fused["c"])

    def test_single_ranking_preserves_order(self):
        fused = reciprocal_rank_fusion([["x", "y", "z"]])
        self.assertEqual(
            sorted(fused, key=fused.get, reverse=True), ["x", "y", "z"])

    def test_empty_rankings(self):
        self.assertEqual(reciprocal_rank_fusion([]), {})
        self.assertEqual(reciprocal_rank_fusion([[], []]), {})

    def test_k_parameter(self):
        shallow = reciprocal_rank_fusion([["a", "b"]], k=1)
        deep = reciprocal_rank_fusion([["a", "b"]], k=1000)
        # larger k compresses the gap between rank 1 and rank 2
        self.assertGreater(shallow["a"] - shallow["b"],
                           deep["a"] - deep["b"])


class FuseHitsTests(unittest.TestCase):
    def _hit(self, key, score=0.0):
        return SimpleNamespace(key=key, score=score)

    def test_keyword_hit_wins_tie(self):
        kw = self._hit("a")
        sem = self._hit("a")
        fused = fuse_hits([kw], [sem], key_fn=lambda h: h.key)
        self.assertEqual(len(fused), 1)
        self.assertIs(fused[0], kw)

    def test_fused_ordering(self):
        kw = [self._hit("a"), self._hit("b")]
        sem = [self._hit("b"), self._hit("c")]
        fused = fuse_hits(kw, sem, key_fn=lambda h: h.key, top=3)
        self.assertEqual([h.key for h in fused], ["b", "a", "c"])

    def test_top_limits(self):
        kw = [self._hit("a"), self._hit("b")]
        fused = fuse_hits(kw, [], key_fn=lambda h: h.key, top=1)
        self.assertEqual([h.key for h in fused], ["a"])


class SemanticCorpusTests(unittest.TestCase):
    def test_status_unbuilt_initially(self):
        corpus = CanonCorpus(_ctx())
        status = corpus.semantic_index_status()
        self.assertFalse(status["built"])
        self.assertEqual(status["vectors"], 0)
        self.assertIn("hashing", status["available_backends"])

    def test_semantic_without_index_fails_fast(self):
        corpus = _corpus_with_text()
        with self.assertRaises(CorpusError) as ctx:
            corpus.ask("kingdom of heaven", mode="semantic")
        self.assertIn("build_semantic_index", str(ctx.exception))

    def test_build_index(self):
        corpus = _corpus_with_text()
        status = corpus.build_semantic_index()
        self.assertTrue(status["built"])
        self.assertEqual(status["backend"], "hashing")
        self.assertIn(status["engine"], ("python", "vec0"))
        self.assertGreater(status["vectors"], 0)

    def test_semantic_ask_returns_provenance(self):
        corpus = _corpus_with_text()
        corpus.build_semantic_index()
        answer = corpus.ask("kingdom of heaven", mode="semantic", top=3)
        self.assertTrue(answer.passages, "expected semantic hits")
        hit = answer.passages[0]
        # provenance must trace back to the manifest entry
        self.assertIn("Gospel of Thomas", hit.work)
        self.assertEqual(hit.translator, "Patterson & Robinson")
        self.assertEqual(hit.canon_status, "gnostic")
        self.assertTrue(hit.url.startswith("http"))
        self.assertTrue(hit.snippet)

    def test_hybrid_ask(self):
        corpus = _corpus_with_text()
        corpus.build_semantic_index()
        answer = corpus.ask("kingdom of heaven", mode="hybrid", top=3)
        self.assertTrue(answer.passages)
        self.assertTrue(all(p.work for p in answer.passages))

    def test_hybrid_with_tradition_filter(self):
        corpus = _corpus_with_text()
        corpus.build_semantic_index()
        answer = corpus.ask("kingdom", mode="hybrid", top=5,
                            tradition="christian-gnostic")
        self.assertTrue(answer.passages)
        # every surviving hit must carry the filtered tradition's entry
        for p in answer.passages:
            self.assertEqual(p.translator, "Patterson & Robinson")

    def test_unknown_mode_rejected(self):
        corpus = _corpus_with_text()
        with self.assertRaises(CorpusError):
            corpus.ask("kingdom", mode="lexical")

    def test_keyword_mode_unchanged(self):
        # default path is byte-for-byte the old behavior
        corpus = _corpus_with_text()
        answer = corpus.ask("kingdom of heaven", top=3)
        self.assertTrue(answer.passages)
        self.assertIn("Gospel of Thomas", answer.passages[0].work)

    def test_keeper_passthrough(self):
        keeper = WisdomKeeper(_ctx())
        keeper.corpus.register(_entry())
        keeper.corpus.ingest_text("gospel-of-thomas", TEXT)
        keeper.build_semantic_index()
        answer = keeper.ask("inner light", mode="hybrid", top=2)
        self.assertTrue(answer.passages)
        status = keeper.semantic_index_status()
        self.assertTrue(status["built"])

    def test_rebuild_is_idempotent(self):
        corpus = _corpus_with_text()
        s1 = corpus.build_semantic_index()
        s2 = corpus.build_semantic_index()
        self.assertEqual(s1["vectors"], s2["vectors"])
        s3 = corpus.build_semantic_index(rebuild=True)
        self.assertEqual(s3["vectors"], s1["vectors"])

    def test_no_bare_except_in_hybrid(self):
        import nomorals.wisdom.hybrid as mod
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.fail("bare except: found in hybrid.py")


if __name__ == "__main__":
    unittest.main()
