"""Reciprocal rank fusion + federated_search(fusion=..., parallel=...).

RRF math is pinned against hand-computed values; the federated wiring
is tested with fake adapters (no network, no local indexes).
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.search.errors import SearchError
from nomorals.search.federated import federated_search
from nomorals.search.model import SearchResult, reciprocal_rank_fusion
from nomorals.search.sources import SourceAdapter, build_adapters, valid_source_names


def _hit(source, title, raw=0.0, ts=None):
    return SearchResult(query="q", title=title, snippet=f"snippet {title}",
                        source=source, type="web", raw_score=raw,
                        timestamp=ts, source_id=f"{source}:{title}")


class FakeAdapter(SourceAdapter):
    name = "web_searxng"
    result_type = "web"
    description = "fake"

    def __init__(self, source_name, hits=None, note=None, boom=None):
        self.name = source_name
        self._hits = hits or []
        self._note = note
        self._boom = boom

    def probe(self):
        return self._note

    def search(self, query, *, limit, since=None, before=None):
        if self._boom:
            raise self._boom
        return list(self._hits[:limit])


def _adapters(**over):
    ads = {name: FakeAdapter(name) for name in valid_source_names()}
    ads.update(over)
    return ads


class RRFFusionTests(unittest.TestCase):
    def test_shared_hit_outranks_single_source_hits(self):
        # shared rank-1 in both sources: 2/61. solo rank-2 in one: 1/62.
        a1 = _hit("s1", "shared")
        a2 = _hit("s2", "shared")  # same title+snippet => duplicate
        b = _hit("s1", "solo")
        fused, dropped = reciprocal_rank_fusion([[a1, b], [a2]])
        self.assertEqual(dropped, 1)
        self.assertEqual([h.title for h in fused], ["shared", "solo"])
        self.assertAlmostEqual(fused[0].score, 2 / 61, places=9)
        self.assertAlmostEqual(fused[1].score, 1 / 62, places=9)

    def test_rank_decay(self):
        # rank 1 vs rank 10 in one source
        first = _hit("s1", "first")
        tenth = _hit("s1", "tenth")
        rest = [_hit("s1", f"pad{i}") for i in range(8)]
        fused, _ = reciprocal_rank_fusion([[first] + rest + [tenth]])
        by_title = {h.title: h for h in fused}
        self.assertAlmostEqual(by_title["first"].score, 1 / 61, places=9)
        self.assertAlmostEqual(by_title["tenth"].score, 1 / 70, places=9)
        self.assertEqual(fused[0].title, "first")
        self.assertEqual(fused[-1].title, "tenth")

    def test_keeps_strongest_representative(self):
        a_weak = _hit("s1", "same")
        a_strong = _hit("s2", "same")
        fused, dropped = reciprocal_rank_fusion(
            [[_hit("s1", "pad0"), _hit("s1", "pad1"), a_weak], [a_strong]]
        )
        self.assertEqual(dropped, 1)
        # s2's copy ranked 1 (contrib 1/61) beats s1's rank-3 copy (1/63)
        self.assertEqual(fused[0].source, "s2")

    def test_empty_input(self):
        fused, dropped = reciprocal_rank_fusion([[], []])
        self.assertEqual(fused, [])
        self.assertEqual(dropped, 0)

    def test_bad_k_rejected(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[ _hit("s1", "x") ]], k=0)

    def test_deterministic_tie_break(self):
        a = _hit("s1", "bbb")
        b = _hit("s2", "aaa")
        fused, _ = reciprocal_rank_fusion([[a], [b]])
        self.assertEqual([h.title for h in fused], ["aaa", "bbb"])


class FederatedRRFTests(unittest.TestCase):
    def test_rrf_mode_fuses_across_sources(self):
        a = FakeAdapter("web_searxng", hits=[_hit("web_searxng", "shared"), _hit("web_searxng", "a-only")])
        b = FakeAdapter("web_tavily", hits=[_hit("web_tavily", "shared"), _hit("web_tavily", "b-only")])
        resp = federated_search(
            "q", sources=["web_searxng", "web_tavily"], fusion="rrf",
            adapters=_adapters(web_searxng=a, web_tavily=b),
        )
        titles = [h.title for h in resp.hits]
        self.assertEqual(titles[0], "shared")  # fused 2/61 beats either 1/61
        self.assertEqual(resp.deduped, 1)
        self.assertEqual(resp.sources_searched, ["web_searxng", "web_tavily"])

    def test_rrf_respects_type_and_date_filters(self):
        old = _hit("web_searxng", "oldie", ts=100.0)
        new = _hit("web_tavily", "newie", ts=300.0)
        a = FakeAdapter("web_searxng", hits=[old])
        b = FakeAdapter("web_tavily", hits=[new])
        resp = federated_search(
            "q", sources=["web_searxng", "web_tavily"], fusion="rrf", since=200.0,
            adapters=_adapters(web_searxng=a, web_tavily=b),
        )
        self.assertEqual([h.title for h in resp.hits], ["newie"])

    def test_unknown_fusion_rejected(self):
        with self.assertRaises(SearchError):
            federated_search("q", fusion="pagerank",
                             adapters=_adapters())

    def test_legacy_default_unchanged(self):
        # legacy: per-source min-max normalize — the historic contract
        a = FakeAdapter("web_searxng", hits=[_hit("web_searxng", "a1", raw=5.0),
                                       _hit("web_searxng", "a2", raw=1.0)])
        resp = federated_search("q", sources=["web_searxng"],
                                adapters=_adapters(web_searxng=a))
        by_title = {h.title: h for h in resp.hits}
        self.assertEqual(by_title["a1"].score, 1.0)
        self.assertEqual(by_title["a2"].score, 0.0)


class FederatedParallelTests(unittest.TestCase):
    def test_parallel_matches_sequential(self):
        def make():
            return _adapters(
                web_searxng=FakeAdapter("web_searxng", hits=[_hit("web_searxng", "a1", raw=2.0)]),
                web_tavily=FakeAdapter("web_tavily", hits=[_hit("web_tavily", "b1", raw=1.0)]),
                web_serper=FakeAdapter("web_serper", note="unconfigured backend"),
            )
        seq = federated_search("q", sources=["web_searxng", "web_tavily", "web_serper"],
                               adapters=make())
        par = federated_search("q", sources=["web_searxng", "web_tavily", "web_serper"],
                               parallel=True, adapters=make())
        self.assertEqual([h.title for h in par.hits],
                         [h.title for h in seq.hits])
        self.assertEqual(par.sources_searched, seq.sources_searched)
        self.assertEqual(par.sources_skipped, seq.sources_skipped)

    def test_parallel_error_names_source(self):
        bad = FakeAdapter("web_tavily", boom=RuntimeError("backend exploded"))
        with self.assertRaises(SearchError) as cm:
            federated_search("q", sources=["web_searxng", "web_tavily"], parallel=True,
                             adapters=_adapters(web_tavily=bad))
        self.assertIn("'web_tavily'", str(cm.exception))
        self.assertIn("backend exploded", str(cm.exception))

    def test_parallel_rrf(self):
        a = FakeAdapter("web_searxng", hits=[_hit("web_searxng", "shared")])
        b = FakeAdapter("web_tavily", hits=[_hit("web_tavily", "shared")])
        resp = federated_search("q", sources=["web_searxng", "web_tavily"],
                                fusion="rrf", parallel=True,
                                adapters=_adapters(web_searxng=a, web_tavily=b))
        self.assertEqual(resp.total, 1)
        self.assertEqual(resp.deduped, 1)


class BuildAdaptersWebTests(unittest.TestCase):
    def test_web_backends_filter(self):
        import tempfile
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            db=None, settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()))
        adapters = build_adapters(ctx, web_backends=["web_tavily"])
        self.assertIn("web_tavily", adapters)
        self.assertNotIn("web_searxng", adapters)
        self.assertIn("memory", adapters)  # local sources always built

    def test_unknown_web_tavilyackend_rejected(self):
        import tempfile
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            db=None, settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()))
        with self.assertRaises(SearchError):
            build_adapters(ctx, web_backends=["web_nonexistent"])


if __name__ == "__main__":
    unittest.main()
