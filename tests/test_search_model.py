"""SearchResult/SearchResponse model: hashing, normalization, dedupe, ranking."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.search.model import (
    SearchResponse,
    SearchResult,
    content_hash,
    dedupe_results,
    normalize_scores,
    rank_results,
)


def _hit(**kw):
    base = dict(
        query="q", title="t", snippet="s", source="books", type="book",
        raw_score=1.0,
    )
    base.update(kw)
    return SearchResult(**base)


class ContentHashTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(content_hash("A", "b"), content_hash("A", "b"))

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(content_hash("Hello  World", "x"),
                         content_hash("hello world", "x"))

    def test_differs_on_content(self):
        self.assertNotEqual(content_hash("a", "b"), content_hash("a", "c"))


class NormalizeTests(unittest.TestCase):
    def test_min_max(self):
        hits = [_hit(raw_score=10.0), _hit(raw_score=20.0), _hit(raw_score=30.0)]
        normalize_scores(hits)
        self.assertAlmostEqual(hits[0].score, 0.0)
        self.assertAlmostEqual(hits[1].score, 0.5)
        self.assertAlmostEqual(hits[2].score, 1.0)

    def test_single_hit_is_one(self):
        hits = [_hit(raw_score=-7.5)]
        normalize_scores(hits)
        self.assertEqual(hits[0].score, 1.0)

    def test_identical_scores_are_one(self):
        hits = [_hit(raw_score=3.0), _hit(raw_score=3.0)]
        normalize_scores(hits)
        self.assertEqual([h.score for h in hits], [1.0, 1.0])

    def test_empty_is_noop(self):
        normalize_scores([])


class DedupeTests(unittest.TestCase):
    def test_identical_content_deduped(self):
        a = _hit(title="Same", snippet="text", score=0.9)
        b = _hit(title="same", snippet="text", score=0.4)
        kept, dropped = dedupe_results([a, b])
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].score, 0.9)  # winner kept

    def test_distinct_kept(self):
        kept, dropped = dedupe_results(
            [_hit(title="a", snippet="x"), _hit(title="b", snippet="y")])
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 2)


class RankTests(unittest.TestCase):
    ORDER = ["memory", "wisdom", "books", "docs", "code", "timeline"]

    def test_score_descending(self):
        lo = _hit(score=0.2, source="memory")
        hi = _hit(score=0.9, source="memory")
        self.assertEqual(rank_results([lo, hi], self.ORDER)[0].score, 0.9)

    def test_tie_breaks_on_source_order(self):
        b = _hit(score=1.0, source="books", title="b")
        m = _hit(score=1.0, source="memory", title="m")
        ranked = rank_results([b, m], self.ORDER)
        self.assertEqual([h.source for h in ranked], ["memory", "books"])

    def test_tie_breaks_on_recency_then_title(self):
        old = _hit(score=1.0, source="memory", title="b", timestamp=100.0)
        new = _hit(score=1.0, source="memory", title="a", timestamp=200.0)
        undated = _hit(score=1.0, source="memory", title="c", timestamp=None)
        ranked = rank_results([old, undated, new], self.ORDER)
        self.assertEqual([h.title for h in ranked], ["a", "b", "c"])

    def test_deterministic(self):
        hits = [_hit(score=1.0, source="memory", title=t) for t in ("x", "y")]
        self.assertEqual(
            [h.title for h in rank_results(hits, self.ORDER)],
            [h.title for h in rank_results(hits, self.ORDER)],
        )


class ShapeTests(unittest.TestCase):
    def test_result_to_dict(self):
        d = _hit(provenance={"work": "w"}, timestamp=1.5,
                 source_id="x:1").to_dict()
        self.assertEqual(d["source"], "books")
        self.assertEqual(d["provenance"], {"work": "w"})
        self.assertEqual(d["timestamp"], 1.5)
        self.assertEqual(d["source_id"], "x:1")
        self.assertIn("score", d)

    def test_response_to_dict(self):
        r = SearchResponse(query="q", hits=[_hit()],
                           sources_searched=["memory"],
                           sources_skipped={"code": "empty"}, deduped=2)
        d = r.to_dict()
        self.assertEqual(d["total"], 1)
        self.assertEqual(d["deduped"], 2)
        self.assertEqual(d["sources_skipped"], {"code": "empty"})


if __name__ == "__main__":
    unittest.main()
