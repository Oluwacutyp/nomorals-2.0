"""Deep research: expansion, parallel fan-out, ranking, sections, cited synthesis.

Offline by design — the engine's search/read are stubbed; the power-mode gate
is exercised for real (owner key in settings).
"""

from __future__ import annotations

import tempfile
import time
import unittest
from typing import Any

from tests.test_search_and_model import _FakeRouter

from nomorals.agents.context import build_context
from nomorals.agents.power import power_mode_for
from nomorals.agents.search.deep import (
    DeepResearcher,
    freshness_signal,
    n_of,
    score_section,
    split_sections,
)
from nomorals.agents.search.engine import SearchEngine
from nomorals.core.config import load_settings
from nomorals.core.errors import ToolError
from nomorals.core.policy import Capability, CapabilitySet

YEAR = time.localtime().tm_year

RESULTS = [
    {"url": f"https://a.com/guide", "title": f"Topic guide {YEAR}",
     "snippet": f"comprehensive topic guide from {YEAR}"},
    {"url": f"https://a.com/blog", "title": "Topic blog",
     "snippet": "an older topic post about the topic"},
    {"url": f"https://b.com/report", "title": f"Independent report {YEAR}",
     "snippet": f"report on the topic, updated {YEAR}"},
    {"url": "https://c.com/notes", "title": "Field notes on topic",
     "snippet": "notes collected about the topic"},
    {"url": f"https://d.com/latest", "title": f"Latest topic developments {YEAR}",
     "snippet": f"the latest developments in the topic, announced this year"},
]

PAGES = {
    f"https://a.com/guide": (
        f"The topic guide explains everything. First, the basics of the topic. "
        f"The topic works because of three core principles. {YEAR} saw the biggest "
        f"update to the topic in a decade. Practitioners now agree on the new standard. "
        f"Further reading covers the topic's history and its open problems. "
        f"Common mistakes when working with the topic include ignoring the second principle. "
        f"Advanced techniques in the topic require care. The topic community meets annually."
    ),
    f"https://b.com/report": (
        f"Independent report on the topic. The report measured the topic across twelve "
        f"organizations in {YEAR}. Results show the topic improved by a wide margin. "
        f"The report also flags where the topic still falls short. Funding for the topic "
        f"grew faster than any other area this year."
    ),
    "https://c.com/notes": (
        "Field notes: the topic is harder than the guides claim. People who work with "
        "the topic daily say the edge cases matter most. Notes collected over a year of "
        "working with the topic. The quiet failures are the interesting part."
    ),
    f"https://d.com/latest": (
        f"Latest developments in the topic, announced in {YEAR}. A new standard for the "
        f"topic was ratified. Vendors shipped topic tooling within a month. The topic "
        f"ecosystem responded quickly and the update is already widely adopted."
    ),
}


def _settings(tmp: str) -> Any:
    return load_settings(overrides={"home": tmp, "partner.platforms": "local",
                                    "chat.local_enabled": "true"})


def _make_context(tmp: str) -> Any:
    context = build_context(_settings(tmp.name), with_executor=False, with_tools=False,
                            with_router=False, with_memory=False)
    context.router = _FakeRouter()
    return context


class TextUtilTests(unittest.TestCase):
    def test_split_sections_chunks_on_paragraphs(self) -> None:
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(8))
        sections = split_sections(text)
        self.assertGreaterEqual(len(sections), 2)
        self.assertLessEqual(len(sections), 12)
        for s in sections:
            self.assertGreaterEqual(len(s), 40)

    def test_split_sections_empty(self) -> None:
        self.assertEqual(split_sections(""), [])
        self.assertEqual(split_sections("short"), [])

    def test_score_section_rewards_query_overlap(self) -> None:
        good = "the topic works because of three core principles of the topic"
        bad = "a recipe for stew needs low heat and time"
        self.assertGreater(score_section(good, "how does the topic work"), 0.2)
        self.assertLess(score_section(bad, "how does the topic work"), 0.1)

    def test_score_section_zero_without_overlap(self) -> None:
        self.assertEqual(score_section("quantum foam and jazz", "the topic"), 0.0)

    def test_freshness_signal(self) -> None:
        self.assertGreaterEqual(freshness_signal(f"announced in {YEAR}"), 0.5)
        self.assertEqual(freshness_signal("back in 1999 nothing since"), 0.0)
        self.assertGreater(freshness_signal(f"updated {YEAR - 2}"), 0.0)
        self.assertEqual(freshness_signal(""), 0.0)

    def test_n_of_maps_urls_to_numbers(self) -> None:
        pages = [{"url": "https://a.com/x/"}, {"url": "https://b.com"}]
        self.assertEqual(n_of(pages, "https://a.com/x"), "1")
        self.assertEqual(n_of(pages, "https://b.com/"), "2")
        self.assertEqual(n_of(pages, "https://nope.com"), "0")


class DeepResearcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-deep-")
        self.context = _make_context(self.tmp)
        self.engine = SearchEngine(self.context)
        self.engine.search = lambda sub, max_results=8: [
            dict(r) for r in RESULTS if r["url"] in self._per_sub(sub)
        ]
        self.engine.read = lambda url, max_chars=40000: (
            {"url": url, "title": url.split("/")[2], "domain": url.split("/")[2],
             "text": PAGES.get(url, ""), "chars": len(PAGES.get(url, ""))}
            if url in PAGES else None
        )
        self.researcher = DeepResearcher(self.context, engine=self.engine, max_pages=4)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _per_sub(self, sub: str) -> list[str]:
        # two sub-queries share the a.com/guide + b.com/report URLs → corroboration
        if "guide" in sub.lower():
            return [f"https://a.com/guide", f"https://b.com/report"]
        return [r["url"] for r in RESULTS]

    def _unlock_power(self) -> None:
        self.context.settings.partner.owner_key = "test-key"
        result = power_mode_for(self.context).unlock("test-key", actor="test")
        self.assertTrue(result["ok"], result)

    def test_expand_queries_adds_freshness_variants(self) -> None:
        subs = self.researcher.expand_queries("the topic")
        self.assertGreaterEqual(len(subs), 3)
        self.assertIn("the topic", subs)
        self.assertIn(f"the topic {YEAR}", subs)
        self.assertIn("the topic latest", subs)

    def test_expand_queries_caps(self) -> None:
        self.researcher.max_subqueries = 4
        subs = self.researcher.expand_queries("the topic")
        self.assertLessEqual(len(subs), 4)

    def test_fan_out_merges_and_corroborates(self) -> None:
        subs = self.researcher.expand_queries("the topic guide")
        merged = self.researcher.fan_out(subs, time.time())
        by_url = {m["url"]: m for m in merged}
        shared = by_url.get(f"https://a.com/guide")
        self.assertIsNotNone(shared)
        self.assertGreaterEqual(len(shared["_subs"]), 2)

    def test_rank_caps_two_per_domain(self) -> None:
        subs = self.researcher.expand_queries("the topic")
        merged = self.researcher.fan_out(subs, time.time())
        ranked = self.researcher.rank(merged, "the topic")
        domains = [m.get("domain") or m["url"].split("/")[2] for m in ranked]
        for d in set(domains):
            self.assertLessEqual(domains.count(d), 2, f"domain {d} over-represented")

    def test_deep_run_end_to_end(self) -> None:
        self._unlock_power()
        report = self.researcher.run("the topic")
        self.assertEqual(report["mode"], "deep")
        self.assertGreaterEqual(len(report["sources"]), 2)
        # numbered, in order, mapped in citations
        self.assertEqual([s["n"] for s in report["sources"]],
                         list(range(1, len(report["sources"]) + 1)))
        for s in report["sources"]:
            self.assertEqual(report["citations"][str(s["n"])], s["url"])
        # extractive (no real model) but cited
        self.assertFalse(report["model_summary"])
        self.assertIn("[1]", report["summary"])
        # sections are scored and capped
        self.assertGreaterEqual(len(report["sections"]), 1)
        self.assertLessEqual(len(report["sections"]), 10)
        self.assertLessEqual(len(report["sections"]), 2 * len(report["pages_read"]))
        # journal landed
        rows = self.context.db.query("SELECT * FROM search_log WHERE mode='deep'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "the topic")

    def test_engine_deep_gate_and_delegation(self) -> None:
        with self.assertRaises(ToolError) as ctx:
            self.engine.run("the topic", mode="deep")
        self.assertIn("power", str(ctx.exception).lower())
        self._unlock_power()
        report = self.engine.run("the topic", mode="deep")
        self.assertEqual(report["mode"], "deep")
        self.assertIn("sources", report)
        self.assertIn("citations", report)

    def test_quick_run_unchanged(self) -> None:
        report = self.engine.run("the topic", mode="quick")
        self.assertEqual(report["mode"], "quick")
        self.assertNotIn("citations", report)


class DeepSearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-deep-tool-")
        self.context = build_context(
            _settings(self.tmp.name), with_executor=False, with_router=False,
            with_memory=False, with_tools=True,
        )

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_registered_and_power_gated(self) -> None:
        reg = self.context.tools
        self.assertIn("deep_search", reg.names())
        schema = next(s for s in reg.schemas() if s["name"] == "deep_search")
        self.assertEqual(schema["capability"], Capability.NET_OUT)

        denied = reg.call("deep_search", query="the topic",
                          capabilities=CapabilitySet.of("fs.read"))
        self.assertFalse(denied.ok)

        # power locked: the tool refuses with the power-mode message
        self.context.settings.partner.owner_key = ""
        outcome = reg.call("deep_search", query="the topic",
                           capabilities=CapabilitySet.all())
        self.assertFalse(outcome.ok)
        self.assertIn("power", str(outcome.error).lower())


if __name__ == "__main__":
    unittest.main()
