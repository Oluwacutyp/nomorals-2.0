"""Bounded multi-hop dig: max 3 hops, reflection-gated early stop,
circle-proof via the visited-URL set.

The fake engine routes follow-up queries to canned pages; each hop's
mined phrases are real 2-word terms sitting next to the query in the
page text, so mine_followups finds them deterministically.
"""

from __future__ import annotations

import tempfile
import unittest
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.search.deep import DeepResearcher
from nomorals.agents.search.engine import SearchEngine
from nomorals.core.config import load_settings

START = "https://hop.com/start"
MID = "https://hop.com/mid"
END = "https://hop.com/end"

TEXTS = {
    START: ("The quantum topic advances through alpha synthesis techniques "
            "in modern labs. Researchers keep refining the quantum topic "
            "with each new alpha synthesis run."),
    MID: ("Researchers applied quantum topic methods with beta calibration "
          "steps for precision. The quantum topic benefits greatly from "
          "beta calibration across repeated trials."),
    END: ("Final quantum topic notes conclude the study. The quantum topic "
          "field keeps growing year after year."),
}


def _settings(tmp: str) -> Any:
    return load_settings(overrides={"home": tmp, "partner.platforms": "local",
                                    "chat.local_enabled": "true",
                                    "partner.power_default_on": False})


class _FakeEngine(SearchEngine):
    """Routes searches to canned pages. mode='endless' always returns a
    fresh page for new follow-ups; mode='dead-end' makes the second hop
    rediscover only seen URLs (reflection must stop the loop)."""

    def __init__(self, context: Any, mode: str = "endless") -> None:
        # bypass SearchEngine.__init__ (no tool registry needed)
        self.context = context
        self.mode = mode

    def search(self, query: str, max_results: int | None = None,
               freshness: str = "") -> list[dict[str, str]]:
        q = (query or "").lower()
        if "beta calibration" in q:
            if self.mode == "dead-end":
                return [{"url": START, "title": "start",
                         "snippet": "quantum topic start page"}]
            return [{"url": END, "title": "end",
                     "snippet": "quantum topic final notes"}]
        if "alpha synthesis" in q:
            return [{"url": MID, "title": "mid",
                     "snippet": "quantum topic mid page with beta calibration"}]
        return [{"url": START, "title": "start",
                 "snippet": "quantum topic start page"}]

    def read(self, url: str, max_chars: int = 40000) -> dict[str, Any] | None:
        if url not in TEXTS:
            return None
        text = TEXTS[url]
        return {"url": url, "title": url.rsplit("/", 1)[-1],
                "domain": "hop.com", "text": text, "chars": len(text),
                "links": []}

    def _decompose(self, query: str) -> list[str]:
        return [query]


def _make_context(tmp: str) -> Any:
    return build_context(_settings(tmp.name), with_executor=False,
                         with_tools=False, with_router=False, with_memory=False)


class HopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-hops-")
        self.context = _make_context(self.tmp)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _researcher(self, mode: str = "endless", **kw: Any) -> DeepResearcher:
        engine = _FakeEngine(self.context, mode=mode)
        kw.setdefault("max_hops", 3)
        return DeepResearcher(self.context, engine=engine, max_pages=4, **kw)

    def test_dig_follows_leads_across_hops(self) -> None:
        report = self._researcher().run("quantum topic")
        self.assertGreaterEqual(report["hops"], 2, report["hop_detail"])
        self.assertIn(MID, report["pages_read"])
        self.assertIn(END, report["pages_read"])
        self.assertTrue(report["followups"], "hop queries must be recorded")

    def test_hops_bounded_at_max_hops(self) -> None:
        report = self._researcher(max_hops=1).run("quantum topic")
        self.assertEqual(report["hops"], 1)
        self.assertEqual(len(report["hop_detail"]), 1)
        # hop 2 never ran: the end page (reachable only via hop 2) is absent
        self.assertNotIn(END, report["pages_read"])

    def test_reflection_stops_dead_hop_early(self) -> None:
        report = self._researcher(mode="dead-end").run("quantum topic")
        # hop 1 found the mid page; hop 2 rediscovered only seen URLs →
        # reflection stops the loop at 2 hops, never a third
        self.assertEqual(report["hops"], 2)
        detail = report["hop_detail"]
        self.assertEqual(detail[0]["new_pages"], 1)
        self.assertEqual(detail[1]["new_pages"], 0)

    def test_no_circles_no_double_reads(self) -> None:
        report = self._researcher(mode="dead-end").run("quantum topic")
        self.assertEqual(report["pages_read"].count(START), 1)
        self.assertEqual(report["pages_read"].count(MID), 1)

    def test_hop_detail_shape(self) -> None:
        report = self._researcher().run("quantum topic")
        for entry in report["hop_detail"]:
            self.assertIn("hop", entry)
            self.assertIn("followups", entry)
            self.assertIn("new_pages", entry)
            self.assertIsInstance(entry["followups"], list)

    def test_dig_off_means_no_hops(self) -> None:
        researcher = self._researcher()
        researcher.dig = False
        report = researcher.run("quantum topic")
        self.assertEqual(report["hops"], 0)
        self.assertEqual(report["followups"], [])

    def test_expand_queries_accepts_scope(self) -> None:
        researcher = self._researcher()
        subs = researcher.expand_queries("quantum topic", scope="auto")
        self.assertIn("quantum topic", subs)
        # a scope-sensitive question gains regional sub-queries
        ng_subs = researcher.expand_queries("best bank accounts", scope="auto")
        self.assertTrue(any("nigeria" in s.lower() or "lagos" in s.lower()
                            for s in ng_subs), ng_subs)


class QuickScopeFanoutTests(unittest.TestCase):
    """Quick mode fans out scope-sensitive queries and spreads both
    regions across the top set."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-qfan-")
        self.context = _make_context(self.tmp)
        self.engine = SearchEngine(self.context)
        self.searched: list[str] = []

        def _search(query: str, max_results: int | None = None,
                    freshness: str = "") -> list[dict[str, str]]:
            self.searched.append(query)
            q = query.lower()
            if "nigeria" in q or "lagos" in q:
                dom = "ng-bank.com"
            elif "united states" in q or q.endswith(" us"):
                dom = "us-bank.com"
            else:
                dom = "global-bank.com"
            return [{
                "url": f"https://{dom}/accounts",
                "title": f"Best bank accounts ({dom})",
                "snippet": "Bank accounts compared: fees, interest rates, and minimum balances for savers.",
            }]

        def _read(url: str, max_chars: int = 40000) -> dict[str, Any] | None:
            text = ("Bank accounts differ by fees and interest rates. "
                    "Savers should compare minimum balances before opening accounts. "
                    "The best bank accounts combine low fees with fair rates.")
            return {"url": url, "title": "accounts", "domain": url.split("/")[2],
                    "text": text, "chars": len(text), "links": []}

        self.engine.search = _search  # type: ignore[method-assign]
        self.engine.read = _read  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_scope_sensitive_query_fans_out(self) -> None:
        report = self.engine.run("best bank accounts", mode="quick", pages=4)
        self.assertGreater(len(self.searched), 1, self.searched)
        self.assertTrue(any("nigeria" in s.lower() or "lagos" in s.lower()
                            for s in self.searched), self.searched)
        self.assertTrue(any("united states" in s.lower() for s in self.searched),
                        self.searched)

    def test_report_carries_scopes_and_citations(self) -> None:
        report = self.engine.run("best bank accounts", mode="quick", pages=4)
        self.assertIn("ng", report["scopes"])
        self.assertIn("us", report["scopes"])
        self.assertGreaterEqual(report["scope_relevance"], 0.40)
        domains = {s["domain"] for s in report["sources"]}
        self.assertIn("ng-bank.com", domains, domains)
        self.assertIn("us-bank.com", domains, domains)
        for s in report["sources"]:
            self.assertEqual(report["citations"][str(s["n"])], s["url"])
        # every claim clickable: full URLs in the summary's Sources block
        self.assertIn("https://ng-bank.com/accounts", report["summary"])
        self.assertIn("https://us-bank.com/accounts", report["summary"])

    def test_factoid_stays_single_shot(self) -> None:
        self.engine.run("what is photosynthesis", mode="quick", pages=3)
        self.assertEqual(len(self.searched), 1, self.searched)


if __name__ == "__main__":
    unittest.main()
