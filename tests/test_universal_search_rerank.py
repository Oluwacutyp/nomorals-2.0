"""BM25 re-ranking: ordering, determinism, edge cases (pure stdlib)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.search.model import SearchResult
from nomorals.search.rerank import bm25_rerank, bm25_scores, tokenize


def _hit(title, snippet, source="web_searxng"):
    return SearchResult(query="python", title=title, snippet=snippet,
                        source=source, type="web", raw_score=0.5,
                        source_id=f"{source}:{title}")


class TokenizeTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(tokenize("Hello, WORLD! x"), ["hello", "world"])

    def test_drops_single_chars(self):
        self.assertEqual(tokenize("a b cd"), ["cd"])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class BM25ScoresTests(unittest.TestCase):
    def test_term_frequency_matters(self):
        scores = bm25_scores(["python"], [["python"], ["python", "python", "python"]])
        self.assertGreater(scores[1], scores[0])

    def test_rare_term_matters_more(self):
        # "xylophone" appears in 1 of 3 docs, "the"-like common term in all
        docs = [["common", "xylophone"], ["common"], ["common"]]
        scores = bm25_scores(["common", "xylophone"], docs)
        self.assertGreater(scores[0], scores[1])

    def test_empty_inputs(self):
        self.assertEqual(bm25_scores([], [["a"]]), [0.0])
        self.assertEqual(bm25_scores(["a"], []), [])

    def test_no_match_is_zero(self):
        self.assertEqual(bm25_scores(["zzz"], [["aaa", "bbb"]]), [0.0])


class BM25RerankTests(unittest.TestCase):
    def test_relevant_first(self):
        hits = [
            _hit("unrelated post", "nothing about snakes here at all"),
            _hit("python guide", "python python python tutorial"),
            _hit("mildly relevant", "mentions python once in passing"),
        ]
        ranked = bm25_rerank(hits, "python tutorial")
        self.assertEqual(ranked[0].title, "python guide")
        self.assertEqual(ranked[-1].title, "unrelated post")

    def test_raw_score_overwritten_with_bm25(self):
        hits = [_hit("python guide", "python stuff")]
        ranked = bm25_rerank(hits, "python")
        self.assertGreater(ranked[0].raw_score, 0.0)
        self.assertNotEqual(ranked[0].raw_score, 0.5)  # native score replaced

    def test_deterministic_tie_break(self):
        hits = [_hit("bbb", "python"), _hit("aaa", "python")]
        first = [h.title for h in bm25_rerank(hits, "python")]
        second = [h.title for h in bm25_rerank(list(hits), "python")]
        self.assertEqual(first, second)
        self.assertEqual(first, ["aaa", "bbb"])  # tie -> title order

    def test_empty_results(self):
        self.assertEqual(bm25_rerank([], "python"), [])

    def test_empty_query_keeps_stable_order(self):
        hits = [_hit("b", "x"), _hit("a", "y")]
        ranked = bm25_rerank(hits, "   ")
        # all scores 0.0 -> deterministic title order
        self.assertEqual([h.title for h in ranked], ["a", "b"])

    def test_does_not_mutate_input_list_order(self):
        hits = [_hit("zzz", "nothing"), _hit("python", "python")]
        original = list(hits)
        bm25_rerank(hits, "python")
        self.assertEqual(hits, original)


if __name__ == "__main__":
    unittest.main()
