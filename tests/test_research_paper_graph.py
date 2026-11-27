"""Tests for the Connected Papers organ (52-websites #36).

The proof the report asks for: a synthetic citation corpus with
planted clusters → the recovered neighbourhood matches ground
truth; prior/derivative views return the planted foundations and
follow-ups; scores are deterministic (pinned by tests).

Fixture corpus (OpenAlex-shaped work dicts; ids are full OpenAlex
URLs, references are id lists):

    seed S (2020) refs {a1, a2, b1, f1, f2}
    a1 (2018) refs {f1, f2, x1, x2}          a2 (2019) refs {f1, f2, x1, x3}
    a3 (2017) refs {f1, x1, x2}              b1 (2019) refs {g1, g2, y1}
    b2 (2021) refs {g1, y1, y2}
    f1 (2005) refs {}  f2 (2008) refs {}     g1 (2010) refs {}
    c1 (2023) refs {a1, a2, a3, S}           c2 (2024) refs {b1, b2, S}
    c3 (2022) refs {a1, S}

Ground truth (hand-computed, then verified against the organ):
  coupling(S, a1) = {f1, f2} = 2; coupling(S, a2) = 2; coupling(S, b1) = 0
  coupling(a1, a2) = {f1, f2, x1} = 3; coupling(b1, b2) = {g1, y1} = 2
  co-citation(S, a1) = {c1, c3} = 2; co-citation(S, f1) = 0 (nobody
  cites both the seed and f1); co-citation(a1, a2) = {seed, c1} = 2
  (the seed itself counts as a citer)
  prior works (older than S, by in-graph citations): f1 (3), f2 (3)
  — a3 is not a pool paper (only in c1's reference list), so f1 is
  cited by {seed, a1, a2}, not 4
  derivative works (newer than S, by graph out-degree): c1 (3),
  c2 (2), c3 (2) — c2's b2 is not a pool paper (only in c2's
  reference list), so c2 cites {b1, seed}; the 2-way tie breaks
  newest first, so c2 leads c3
  clusters at raw >= 2 over the full fixture: {S, a1, a2, a3, c1, c3},
  {f1, f2} (co-cited by S, a1, a2 but sharing no reference with the
  seed), {b1, b2}; c2 stands alone.
  In the built graph (1-hop pool, a3 absent) f1/f2 join the big
  school through their raw-2 edges to the seed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.research.paper_graph import (  # noqa: E402
    PaperGraph,
    PaperNode,
    build_graph,
    clusters_of,
    co_citation,
    coupling,
    graph_diagram,
    open_graph_store,
    reading_order,
    resolve_seed,
    similarity_edges,
)

SEED_ID = "https://openalex.org/W100"


def _work(wid: str, title: str, year: int, cites: int,
          refs: list[str], doi: str = "") -> dict[str, Any]:
    return {"id": wid, "title": title, "publication_year": year,
            "doi": doi or f"https://doi.org/10.0000/{wid[-3:]}",
            "cited_by_count": cites,
            "abstract_inverted_index": {},
            "referenced_works": refs,
            "cited_by_api_url":
                f"https://api.openalex.org/works?filter=cites:{wid[-4:]}"}


def _fixture_corpus() -> dict[str, dict[str, Any]]:
    a1, a2, a3 = "https://openalex.org/W201", "https://openalex.org/W202", \
        "https://openalex.org/W203"
    b1, b2 = "https://openalex.org/W204", "https://openalex.org/W205"
    f1, f2 = "https://openalex.org/W301", "https://openalex.org/W302"
    x1, x2, x3 = "https://openalex.org/W401", "https://openalex.org/W402", \
        "https://openalex.org/W403"
    g1, g2 = "https://openalex.org/W404", "https://openalex.org/W405"
    y1, y2 = "https://openalex.org/W406", "https://openalex.org/W407"
    c1, c2, c3 = "https://openalex.org/W501", "https://openalex.org/W502", \
        "https://openalex.org/W503"
    works = {
        SEED_ID: _work(SEED_ID, "Vitamin D and respiratory infections: "
                       "a meta-analysis", 2020, 12,
                       [a1, a2, b1, f1, f2],
                       doi="https://doi.org/10.0000/vitd"),
        a1: _work(a1, "Vitamin D supplementation and acute respiratory "
                  "infection", 2018, 40, [f1, f2, x1, x2]),
        a2: _work(a2, "Daily vitamin D and respiratory illness in adults",
                  2019, 25, [f1, f2, x1, x3]),
        a3: _work(a3, "Vitamin D status and winter respiratory illness",
                  2017, 8, [f1, x1, x2]),
        b1: _work(b1, "Sunlight exposure and immune response",
                  2019, 30, [g1, g2, y1]),
        b2: _work(b2, "Seasonal sunlight and T-cell activation",
                  2021, 5, [g1, y1, y2]),
        f1: _work(f1, "Foundations of vitamin D immunology",
                  2005, 300, []),
        f2: _work(f2, "Vitamin D receptor signalling in immunity",
                  2008, 120, []),
        c1: _work(c1, "A 2023 trial of vitamin D in respiratory care",
                  2023, 2, [a1, a2, a3, SEED_ID]),
        c2: _work(c2, "Sunlight, vitamin D and immunity: 2024 review",
                  2024, 1, [b1, b2, SEED_ID]),
        c3: _work(c3, "Vitamin D dosing after the 2020 meta-analysis",
                  2022, 0, [a1, SEED_ID]),
    }
    return works


class _FakeFetcher:
    """Serves the fixture corpus as OpenAlex-shaped JSON."""

    def __init__(self, works: dict[str, dict[str, Any]]) -> None:
        self.works = works
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: int = 15) -> Any:
        import urllib.parse

        self.calls.append(url)
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(url).query)
        filt = (query.get("filter") or [""])[0]
        if filt.startswith("cites:"):
            citing = [w for w in self.works.values()
                      if SEED_ID in (w.get("referenced_works") or [])]
            return {"results": citing}
        if filt.startswith("openalex:"):
            ids = [f"https://openalex.org/{piece}"
                   for piece in filt[len("openalex:"):].split("|")
                   if piece]
            return {"results": [self.works[i] for i in ids
                                if i in self.works]}
        if "/works/doi:" in url or "works/https://doi.org" in url:
            for work in self.works.values():
                if work.get("doi", "").endswith("vitd"):
                    return work
            return {}
        if "search=" in url:
            if "vitamin" in url.lower():
                return {"results": [self.works[SEED_ID]]}
            return {"results": []}
        if "semanticscholar" in url:
            return {"data": []}
        return None


def _refsets() -> dict[str, set[str]]:
    works = _fixture_corpus()
    return {wid: set(w.get("referenced_works") or [])
            for wid, w in works.items()}


class SimilarityMathTests(unittest.TestCase):
    def test_coupling_counts_shared_references(self) -> None:
        refs = _refsets()
        self.assertEqual(coupling(refs[SEED_ID], refs["https://openalex.org/W201"]), 2)
        self.assertEqual(coupling(refs[SEED_ID], refs["https://openalex.org/W202"]), 2)
        self.assertEqual(coupling(refs[SEED_ID], refs["https://openalex.org/W204"]), 0)
        self.assertEqual(
            coupling(refs["https://openalex.org/W201"],
                     refs["https://openalex.org/W202"]), 3)
        self.assertEqual(
            coupling(refs["https://openalex.org/W204"],
                     refs["https://openalex.org/W205"]), 2)

    def test_co_citation_counts_shared_citers(self) -> None:
        refs = _refsets()
        self.assertEqual(co_citation(SEED_ID, "https://openalex.org/W201", refs), 2)
        # Nobody cites both the seed and f1 — f1 is a foundation the
        # seed cites, not a paper co-cited with it. (The seed itself
        # counts as a citer: it cites both a1 and a2.)
        self.assertEqual(co_citation(SEED_ID, "https://openalex.org/W301", refs), 0)
        self.assertEqual(
            co_citation("https://openalex.org/W201",
                        "https://openalex.org/W202", refs), 2)
        self.assertEqual(
            co_citation("https://openalex.org/W201",
                        "https://openalex.org/W301", refs), 1)
        self.assertEqual(
            co_citation("https://openalex.org/W301",
                        "https://openalex.org/W302", refs), 3)
        self.assertEqual(co_citation(SEED_ID, "https://openalex.org/W204", refs), 1)

    def test_similarity_edges_rank_planted_cluster_first(self) -> None:
        refs = _refsets()
        ids = sorted(refs)
        edges = similarity_edges(ids, refs)
        by_pair = {(e.a, e.b): e for e in edges}
        seed_a1 = by_pair[("https://openalex.org/W100",
                           "https://openalex.org/W201")]
        self.assertEqual((seed_a1.coupling, seed_a1.co_citation), (2, 2))
        self.assertEqual(seed_a1.raw, 4)
        # Max raw in the corpus is 5 (a1–a2: coupling 3 + co-citation
        # 2 — the seed itself cites both), so the seed's best edge
        # scores 4/5, not 1.0.
        self.assertAlmostEqual(seed_a1.score, 0.8)
        top = by_pair[("https://openalex.org/W201",
                       "https://openalex.org/W202")]
        self.assertEqual((top.coupling, top.co_citation), (3, 2))
        self.assertEqual(top.raw, 5)
        self.assertAlmostEqual(top.score, 1.0)
        # The planted cluster A outranks cluster B against the seed.
        seed_b1 = by_pair[("https://openalex.org/W100",
                           "https://openalex.org/W204")]
        self.assertLess(seed_b1.raw, seed_a1.raw)
        # Scores are deterministic and normalised to [0, 1].
        for edge in edges:
            self.assertGreaterEqual(edge.score, 0.0)
            self.assertLessEqual(edge.score, 1.0)
        self.assertEqual(similarity_edges(ids, refs), edges)

    def test_clusters_recover_planted_schools(self) -> None:
        refs = _refsets()
        ids = sorted(refs)
        edges = similarity_edges(ids, refs)
        nodes = [PaperNode(paper_id=i, title=i, year=None,
                           citation_count=None) for i in ids]
        clusters = clusters_of(nodes, edges, min_raw=2)
        member_sets = [set(c.member_ids) for c in clusters]
        big = {SEED_ID, "https://openalex.org/W201",
               "https://openalex.org/W202", "https://openalex.org/W203",
               "https://openalex.org/W501", "https://openalex.org/W503"}
        small = {"https://openalex.org/W204", "https://openalex.org/W205"}
        foundations = {"https://openalex.org/W301",
                       "https://openalex.org/W302"}
        self.assertIn(big, member_sets)
        self.assertIn(small, member_sets)
        # The two foundations cluster together on their own: they
        # are co-cited by S, a1 and a2, but neither shares a
        # reference with the seed, so no strong edge binds them to
        # the big school.
        self.assertIn(foundations, member_sets)
        # c2 stands alone — named as such, never forced into a school.
        # (c3 joins the big cluster: it shares {a1, seed} with c1.)
        singletons = [c for c in clusters if len(c.member_ids) == 1]
        self.assertEqual(
            {c.member_ids[0] for c in singletons},
            {"https://openalex.org/W502"})
        for cluster in clusters:
            self.assertTrue(cluster.name)
            self.assertTrue(cluster.top_paper_id)

    def test_cluster_names_come_from_the_top_paper(self) -> None:
        refs = _refsets()
        ids = sorted(refs)
        edges = similarity_edges(ids, refs)
        nodes = [PaperNode(paper_id=i, title=f"Title {i[-3:]}", year=2020,
                           citation_count=5) for i in ids]
        clusters = clusters_of(nodes, edges, min_raw=2)
        biggest = max(clusters, key=lambda c: len(c.member_ids))
        self.assertIn("Title", biggest.name)


class SeedResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = _FakeFetcher(_fixture_corpus())

    def test_resolve_seed_by_title(self) -> None:
        paper, openalex_id, notes = resolve_seed(
            "vitamin D respiratory infections", fetch_json=self.fetcher)
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(openalex_id, SEED_ID)
        self.assertIn("meta-analysis", paper.title)
        self.assertTrue(notes)

    def test_resolve_seed_by_doi(self) -> None:
        paper, openalex_id, _notes = resolve_seed(
            "10.0000/vitd", fetch_json=self.fetcher)
        self.assertIsNotNone(paper)
        self.assertEqual(openalex_id, SEED_ID)

    def test_unknown_seed_resolves_to_nothing(self) -> None:
        paper, openalex_id, notes = resolve_seed(
            "quasars made of cheese", fetch_json=self.fetcher)
        self.assertIsNone(paper)
        self.assertEqual(openalex_id, "")
        self.assertTrue(notes)


class BuildGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.fetcher = _FakeFetcher(_fixture_corpus())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _build(self) -> dict[str, Any]:
        return build_graph(self.home, "vitamin D respiratory infections",
                           fetch_json=self.fetcher)

    def test_neighbourhood_matches_ground_truth(self) -> None:
        ans = self._build()
        self.assertTrue(ans["ok"], ans.get("error"))
        graph = PaperGraph.from_dict(ans["graph"])
        self.assertEqual(graph.seed_id, SEED_ID)
        self.assertEqual(graph.pool_size, 8)
        self.assertEqual(len(graph.nodes), 9)  # seed + 8 pool papers
        # The planted cluster A tops the similarity ranking to the seed.
        ranked = sorted(graph.edges, key=lambda e: (-e.raw, e.a, e.b))
        seed_edges = [e for e in ranked
                      if SEED_ID in (e.a, e.b)]
        top = max(seed_edges, key=lambda e: e.raw)
        self.assertIn("https://openalex.org/W201", (top.a, top.b))
        # b1 (cluster B) is related to the seed only weakly.
        b1_edge = next(e for e in seed_edges
                       if "https://openalex.org/W204" in (e.a, e.b))
        self.assertEqual(b1_edge.raw, 1)

    def test_prior_works_are_the_planted_foundations(self) -> None:
        ans = self._build()
        graph = PaperGraph.from_dict(ans["graph"])
        self.assertEqual(
            [p["paper_id"] for p in graph.prior_works[:2]],
            ["https://openalex.org/W301", "https://openalex.org/W302"])
        # a3 is not a pool paper — it is only in c1's reference list,
        # never a seed reference or a citer of the seed — so f1 is
        # cited by 3 graph papers (seed, a1, a2), not 4.
        self.assertEqual(graph.prior_works[0]["in_graph_citations"], 3)

    def test_derivative_works_are_the_planted_follow_ups(self) -> None:
        ans = self._build()
        graph = PaperGraph.from_dict(ans["graph"])
        # c1 cites 3 graph papers ({a1, a2, seed} — a3 is not a pool
        # paper), c2 cites 2 ({b1, seed} — b2 is not a pool paper
        # either); the 2-way tie breaks newest first, so c2 leads c3.
        self.assertEqual(
            [p["paper_id"] for p in graph.derivative_works],
            ["https://openalex.org/W501", "https://openalex.org/W502",
             "https://openalex.org/W503"])
        self.assertEqual(graph.derivative_works[0]["graph_out_degree"], 3)
        self.assertEqual(graph.derivative_works[1]["graph_out_degree"], 2)
        self.assertEqual(graph.derivative_works[2]["graph_out_degree"], 2)

    def test_reading_order_seed_then_foundations_then_similar(self) -> None:
        ans = self._build()
        graph = PaperGraph.from_dict(ans["graph"])
        order = reading_order(graph)
        self.assertEqual(order[0].paper_id, SEED_ID)
        self.assertIn("seed", order[0].reason.lower())
        ids = [step.paper_id for step in order]
        self.assertLess(ids.index("https://openalex.org/W301"),
                        ids.index("https://openalex.org/W201"))
        self.assertLess(ids.index("https://openalex.org/W201"),
                        ids.index("https://openalex.org/W501"))
        for step in order:
            self.assertTrue(step.reason)

    def test_graph_states_its_pool_honestly(self) -> None:
        ans = self._build()
        reply = ans["reply"]
        self.assertIn("8", reply)
        self.assertIn("OpenAlex", reply)
        self.assertIn("pool", reply.lower())

    def test_unknown_seed_is_refused_honestly(self) -> None:
        ans = build_graph(self.home, "quasars made of cheese",
                          fetch_json=self.fetcher)
        self.assertFalse(ans["ok"])
        self.assertIn("no paper", ans["reply"].lower())
        self.assertIn("review_id", ans)
        # The miss is still recorded, so it is visible.
        store = open_graph_store(self.home)
        self.assertIsNotNone(store.get(ans["review_id"]))

    def test_store_round_trip(self) -> None:
        ans = self._build()
        store = open_graph_store(self.home)
        saved = store.get(ans["review_id"])
        self.assertIsNotNone(saved)
        self.assertEqual(saved["graph"]["seed_id"], SEED_ID)
        listed = store.list()
        self.assertEqual(listed[0]["id"], ans["review_id"])
        self.assertEqual(listed[0]["nodes"], 9)

    def test_deterministic_rebuild(self) -> None:
        first = self._build()
        second = self._build()
        self.assertEqual(first["graph"]["edges"],
                         second["graph"]["edges"])
        self.assertEqual(first["graph"]["clusters"],
                         second["graph"]["clusters"])

    def test_diagram_spec_validates(self) -> None:
        from nomorals.visuals.spec import diagram_from_parts

        ans = self._build()
        graph = PaperGraph.from_dict(ans["graph"])
        spec = graph_diagram(graph)
        validated = diagram_from_parts(
            spec["title"], spec["nodes"], spec["edges"])
        self.assertEqual(validated["kind"], "diagram")
        self.assertEqual(len(validated["nodes"]), len(graph.nodes))
        self.assertTrue(all(e["from"] in {n["id"] for n in spec["nodes"]}
                            for e in spec["edges"]))

    def test_reply_names_clusters_and_reading_order(self) -> None:
        ans = self._build()
        reply = ans["reply"]
        self.assertIn("Cluster", reply)
        self.assertIn("Prior works", reply)
        self.assertIn("Derivative works", reply)
        self.assertIn("Read next", reply)
        self.assertIn("stands alone", reply)


class ToolAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.fetcher = _FakeFetcher(_fixture_corpus())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _context(self) -> Any:
        home = str(self.home)

        class Settings:
            home_path = home

        class Ctx:
            settings = Settings()

        return Ctx()

    def test_tool_registers(self) -> None:
        from nomorals.tools.paper_graph import register
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register(registry)
        self.assertIn("paper_graph", registry.names())

    def test_cli_map_answers_from_fixture(self) -> None:
        import argparse
        from unittest import mock

        from nomorals.cmdline.commands.paper_graph import _cmd_paper_graph

        answer = build_graph(self.home, "vitamin D respiratory infections",
                             fetch_json=self.fetcher)
        args = argparse.Namespace(
            paper_graph_query=["vitamin", "D", "respiratory", "infections"],
            json=False, deep=False)
        with mock.patch(
                "nomorals.cmdline.commands.paper_graph.build_graph",
                return_value=answer):
            self.assertEqual(_cmd_paper_graph(args, self._context()), 0)

    def test_cli_map_refusal_exits_1(self) -> None:
        import argparse
        from unittest import mock

        from nomorals.cmdline.commands.paper_graph import _cmd_paper_graph

        args = argparse.Namespace(
            paper_graph_query=["quasars", "made", "of", "cheese"],
            json=False, deep=False)
        refusal = {"ok": False, "review_id": "pg0",
                   "error": "no paper found",
                   "reply": "no paper found"}
        with mock.patch(
                "nomorals.cmdline.commands.paper_graph.build_graph",
                return_value=refusal):
            self.assertEqual(_cmd_paper_graph(args, self._context()), 1)

    def test_cli_list_empty(self) -> None:
        import argparse

        from nomorals.cmdline.commands.paper_graph import _cmd_paper_graph

        args = argparse.Namespace(paper_graph_query=["list"],
                                  json=False, deep=False)
        self.assertEqual(_cmd_paper_graph(args, self._context()), 0)

    def test_chat_registration(self) -> None:
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            PUBLIC_COMMANDS,
        )

        self.assertIn("papergraph", PUBLIC_COMMANDS)
        self.assertEqual(291, len(PUBLIC_COMMANDS))
        self.assertIn("papergraph", CONTROL_COMMANDS)
        self.assertIn("papergraph", COMMAND_DETAILS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("papergraph", grouped)
        self.assertEqual(grouped.count("papergraph"), 1)
        self.assertIn("/papergraph <title or DOI>",
                      COMMAND_DETAILS["papergraph"]["usage"])


if __name__ == "__main__":
    unittest.main()
