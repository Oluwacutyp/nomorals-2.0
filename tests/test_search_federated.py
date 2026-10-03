"""federated_search: fan-out, merge, dedupe, ranking, filters, error paths."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.search.errors import (
    InvalidDateError,
    SearchError,
    UnknownSourceError,
    UnknownTypeError,
)
from nomorals.search.federated import federated_search, parse_date
from nomorals.search.model import SearchResult
from nomorals.search.sources import SourceAdapter, valid_source_names


class FakeAdapter(SourceAdapter):
    name = "memory"
    result_type = "memory"
    description = "fake"

    def __init__(self, hits=None, note=None, boom=None):
        self._hits = hits or []
        self._note = note
        self._boom = boom
        self.calls = []

    def probe(self):
        return self._note

    def search(self, query, *, limit, since=None, before=None):
        self.calls.append((query, limit, since, before))
        if self._boom:
            raise self._boom
        return list(self._hits[:limit])


def _hit(source, rtype, title, raw, ts=None):
    return SearchResult(query="q", title=title, snippet=f"snippet {title}",
                        source=source, type=rtype, raw_score=raw,
                        timestamp=ts, source_id=f"{source}:{title}")


def _adapters(**over):
    ads = {}
    for name in valid_source_names():
        ads[name] = FakeAdapter()
    ads.update(over)
    return ads


class FederatedTests(unittest.TestCase):
    def test_merges_and_normalizes_per_source(self):
        mem = FakeAdapter(hits=[_hit("memory", "memory", "m1", 0.5),
                                _hit("memory", "memory", "m2", 1.0)])
        books = FakeAdapter(hits=[_hit("books", "book", "b1", -99.0)])
        books.name = "books"
        resp = federated_search("q", adapters=_adapters(memory=mem, books=books))
        by_title = {h.title: h for h in resp.hits}
        self.assertEqual(by_title["m2"].score, 1.0)
        self.assertEqual(by_title["m1"].score, 0.0)
        self.assertEqual(by_title["b1"].score, 1.0)  # single hit normalizes to 1
        self.assertEqual(resp.total, 3)

    def test_dedupes_across_sources(self):
        mem = FakeAdapter(hits=[_hit("memory", "memory", "same", 1.0)])
        books = FakeAdapter(hits=[_hit("books", "book", "same", 5.0)])
        books.name = "books"
        resp = federated_search("q", adapters=_adapters(memory=mem, books=books))
        self.assertEqual(resp.total, 1)
        self.assertEqual(resp.deduped, 1)

    def test_skipped_source_noted_not_crashed(self):
        code = FakeAdapter(note="no code indexed yet")
        code.name = "code"
        resp = federated_search("q", adapters=_adapters(code=code))
        self.assertEqual(resp.sources_skipped, {"code": "no code indexed yet"})
        self.assertNotIn("code", resp.sources_searched)
        self.assertEqual(code.calls, [])

    def test_zero_hits_names_searched_sources(self):
        resp = federated_search("q", adapters=_adapters())
        self.assertEqual(resp.hits, [])
        self.assertEqual(resp.sources_searched, valid_source_names())
        self.assertEqual(resp.sources_skipped, {})

    def test_source_selection(self):
        mem = FakeAdapter(hits=[_hit("memory", "memory", "m1", 1.0)])
        resp = federated_search("q", sources=["memory"],
                               adapters=_adapters(memory=mem))
        self.assertEqual(resp.sources_searched, ["memory"])
        self.assertEqual(resp.total, 1)

    def test_type_filter(self):
        mem = FakeAdapter(hits=[_hit("memory", "memory", "m1", 1.0)])
        books = FakeAdapter(hits=[_hit("books", "book", "b1", 1.0)])
        books.name = "books"
        resp = federated_search("q", types=["book"],
                               adapters=_adapters(memory=mem, books=books))
        self.assertEqual([h.type for h in resp.hits], ["book"])

    def test_date_filter_keeps_undated(self):
        old = _hit("memory", "memory", "old", 1.0, ts=100.0)
        new = _hit("memory", "memory", "new", 0.9, ts=300.0)
        undated = _hit("books", "book", "u", 0.8, ts=None)
        books = FakeAdapter(hits=[undated])
        books.name = "books"
        mem = FakeAdapter(hits=[old, new])
        resp = federated_search("q", since=200.0,
                               adapters=_adapters(memory=mem, books=books))
        titles = [h.title for h in resp.hits]
        self.assertIn("new", titles)
        self.assertIn("u", titles)  # undated hits are kept, not dropped
        self.assertNotIn("old", titles)

    def test_limit_cut(self):
        mem = FakeAdapter(hits=[_hit("memory", "memory", f"m{i}", float(i))
                                for i in range(20)])
        resp = federated_search("q", limit=5, adapters=_adapters(memory=mem))
        self.assertEqual(resp.total, 5)

    def test_source_error_fails_fast_with_name(self):
        bad = FakeAdapter(boom=RuntimeError("db gone"))
        bad.name = "memory"
        with self.assertRaises(SearchError) as cm:
            federated_search("q", adapters=_adapters(memory=bad))
        self.assertIn("'memory'", str(cm.exception))

    # ── fail-fast inputs ──
    def test_empty_query(self):
        with self.assertRaises(SearchError):
            federated_search("   ", adapters=_adapters())

    def test_unknown_source_lists_valid(self):
        with self.assertRaises(UnknownSourceError) as cm:
            federated_search("q", sources=["nope"], adapters=_adapters())
        for name in valid_source_names():
            self.assertIn(name, str(cm.exception))

    def test_unknown_type_lists_valid(self):
        with self.assertRaises(UnknownTypeError) as cm:
            federated_search("q", types=["nope"], adapters=_adapters())
        self.assertIn("memory", str(cm.exception))

    def test_bad_date(self):
        with self.assertRaises(InvalidDateError):
            federated_search("q", since="not a date", adapters=_adapters())

    def test_bad_limit(self):
        with self.assertRaises(SearchError):
            federated_search("q", limit=0, adapters=_adapters())

    def test_needs_context_or_adapters(self):
        with self.assertRaises(SearchError):
            federated_search("q", context=None)


class ParseDateTests(unittest.TestCase):
    def test_none_and_blank(self):
        self.assertIsNone(parse_date(None))
        self.assertIsNone(parse_date("  "))

    def test_epoch(self):
        self.assertEqual(parse_date(1700000000), 1700000000.0)
        self.assertEqual(parse_date("1700000000"), 1700000000.0)

    def test_iso(self):
        ts = parse_date("2026-01-15T12:00:00Z")
        self.assertAlmostEqual(ts, 1768478400.0, delta=2)

    def test_iso_naive_is_utc(self):
        self.assertAlmostEqual(parse_date("2026-01-15T12:00:00"),
                               parse_date("2026-01-15T12:00:00Z"), delta=1)

    def test_garbage(self):
        with self.assertRaises(InvalidDateError):
            parse_date("next tuesday-ish")


if __name__ == "__main__":
    unittest.main()
