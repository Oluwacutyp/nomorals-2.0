"""Wave 70 — search deep-dive: multi-engine parsing, snippets, PDF sources,
and the dig loop (mined follow-up queries + external link following).

Offline: canned HTML fixtures for the parsers; a stubbed engine for the dig
loop (same contract as tests.test_deep_research).
"""

from __future__ import annotations

import tempfile
import time
import unittest
from typing import Any
from urllib.parse import quote_plus

from tests.test_deep_research import _make_context  # settings+context helper
from tests.test_search_and_model import _FakeRouter

from nomorals.agents.search import curate
from nomorals.agents.search.deep import DeepResearcher, mine_followups
from nomorals.agents.search.engine import SearchEngine, _extract_external_links
from nomorals.core.pdf import render_pdf
from nomorals.core.policy import Capability
from nomorals.tools import web


# ═══════════════════════════════════════════════════════════════════════════
# Parsers
# ═══════════════════════════════════════════════════════════════════════════

DDG_FIXTURE = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide&rut=x">Example Guide</a>
  <a class="result__snippet" href="#">This is a guide about <b>example topics</b> with real detail.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="https://second.org/page">Second Result</a>
  <a class="result__snippet" href="#">Another snippet that explains the second page contents.</a>
</div>
</body></html>
"""

LITE_FIXTURE = """
<table>
<tr><td><a rel="nofollow" href="//lite.ddg/y/?rut=x">Lite Title One</a></td></tr>
<tr><td class="result-snippet">Lite snippet one with substance here.</td></tr>
<tr><td><a rel="nofollow" href="https://b.org/two">Lite Title Two</a></td></tr>
<tr><td class="result-snippet">Lite snippet two.</td></tr>
</table>
"""

BING_FIXTURE = """
<html><ol id="b_results">
<li class="b_algo"><h2><a href="https://binged.com/story">Binged Story</a></h2>
<p class="b_lineclamp2">Bing caption about the story with useful detail.</p></li>
<li class="b_algo"><h2><a href="https://other.net/next">Other Next</a></h2>
<p>Another caption here.</p></li>
</ol></html>
"""

MOJEEK_FIXTURE = """
<html><div class="results">
<div class="results-row"><a class="title" href="https://moj.example/one">Moj One</a>
<p class="description">Mojeek description for the first result entry.</p></div>
<div class="results-row"><a class="title" href="https://moj.example/two">Moj Two</a>
<p class="description">Second description text here.</p></div>
</div></html>
"""


class ParserTests(unittest.TestCase):
    def test_ddg_extracts_snippets(self) -> None:
        out = web._parse_ddg(DDG_FIXTURE)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["url"], "https://example.com/guide")
        self.assertEqual(out[0]["title"], "Example Guide")
        self.assertIn("guide about", out[0]["snippet"])
        self.assertIn("explains the second page", out[1]["snippet"])

    def test_ddg_fallback_nofollow(self) -> None:
        markup = '<a rel="nofollow" href="https://x.io/doc">X Doc</a>'
        out = web._parse_ddg(markup)
        self.assertEqual(out[0]["url"], "https://x.io/doc")
        self.assertEqual(out[0]["snippet"], "")

    def test_lite_extracts_rows(self) -> None:
        out = web._parse_lite(LITE_FIXTURE)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["title"], "Lite Title One")
        self.assertIn("Lite snippet one", out[0]["snippet"])

    def test_bing_extracts_caption(self) -> None:
        out = web._parse_bing(BING_FIXTURE)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["url"], "https://binged.com/story")
        self.assertIn("Bing caption", out[0]["snippet"])

    def test_mojeek_extracts_description(self) -> None:
        out = web._parse_mojeek(MOJEEK_FIXTURE)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["title"], "Moj One")
        self.assertIn("Mojeek description", out[0]["snippet"])

    def test_unwrap_ddg_redirect(self) -> None:
        href = f"https://duckduckgo.com/l/?uddg={quote_plus('https://real.com/p')}&rut=1"
        self.assertEqual(web._unwrap_ddg(href), "https://real.com/p")
        self.assertEqual(web._unwrap_ddg("https://plain.com/x"), "https://plain.com/x")
        self.assertEqual(web._unwrap_ddg("javascript:void(0)"), "")


class SearchChainTests(unittest.TestCase):
    """The web_search tool's engine fallback chain, with a canned HTTP client."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-chain-")
        self.context = _make_context(self.tmp)
        self.context.router = _FakeRouter()
        self.context.tools = self._StubTools()
        self._canned: dict[str, Any] = {}
        self._calls: list[str] = []
        self._orig_client = web.HttpClient

        calls: list[str] = self._calls
        canned: dict[str, Any] = self._canned

        def _factory(**kw: Any) -> Any:
            class _C:
                def get(self, url: str, **kw2: Any) -> Any:
                    calls.append(url)
                    return canned[url]
            return _C()

        web.HttpClient = _factory
        # register the web tools against a fresh registry bound to our context
        from nomorals.tools.registry import ToolRegistry

        self.registry = ToolRegistry(self.context).register_builtins()

    def tearDown(self) -> None:
        web.HttpClient = self._orig_client
        self.context.close()
        self.tmp.cleanup()

    def _StubTools(self) -> Any:
        # SearchEngine needs context.tools.call for its own search(); the
        # chain test only drives the registered web_search tool directly.
        class _S:
            def call(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("not used in this test")
        return _S()

    def _ok(self, name: str, body: str) -> Any:
        class _Resp:
            status = 200
            headers = {}
            content_type = "text/html"
            url = f"https://{name}/resp"

            def __init__(self, text: str) -> None:
                self._text = text

            @property
            def ok(self) -> bool:
                return True

            @property
            def text(self) -> str:
                return self._text

            @property
            def body(self) -> bytes:
                return self._text.encode()

        return _Resp(body)

    def test_ddg_first(self) -> None:
        self._canned[f"https://html.duckduckgo.com/html/?q={quote_plus('example topics')}"] = \
            self._ok("ddg", DDG_FIXTURE)
        out = self.registry.call("web_search", query="example topics",
                                 capabilities=CapabilitySet_all())
        self.assertTrue(out.ok, out.error)
        data = out.unwrap()
        self.assertEqual(data["results"][0]["engine"], "ddg")
        self.assertIn("guide about", data["results"][0]["snippet"])

    def test_falls_to_lite_when_ddg_dead(self) -> None:
        q = quote_plus("example topics")
        self._canned[f"https://lite.duckduckgo.com/lite/?q={q}"] = self._ok("lite", LITE_FIXTURE)
        out = self.registry.call("web_search", query="example topics",
                                 capabilities=CapabilitySet_all())
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.unwrap()["results"][0]["engine"], "lite")
        # ddg was tried first (and failed), then lite
        self.assertIn("html.duckduckgo.com", self._calls[0])
        self.assertIn("lite.duckduckgo.com", self._calls[1])

    def test_falls_to_bing(self) -> None:
        q = quote_plus("example topics")
        self._canned[f"https://www.bing.com/search?q={q}"] = self._ok("bing", BING_FIXTURE)
        out = self.registry.call("web_search", query="example topics",
                                 capabilities=CapabilitySet_all())
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.unwrap()["results"][0]["engine"], "bing")

    def test_all_dead_returns_empty(self) -> None:
        out = self.registry.call("web_search", query="example topics",
                                 capabilities=CapabilitySet_all())
        self.assertTrue(out.ok)
        self.assertEqual(out.unwrap()["count"], 0)

    def test_freshness_param_applied_to_ddg(self) -> None:
        q = quote_plus("example topics")
        self._canned[f"https://html.duckduckgo.com/html/?q={q}&df=w"] = \
            self._ok("ddg", DDG_FIXTURE)
        out = self.registry.call("web_search", query="example topics", freshness="w",
                                 capabilities=CapabilitySet_all())
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.unwrap()["results"][0]["engine"], "ddg")


def CapabilitySet_all() -> Any:
    from nomorals.core.policy import CapabilitySet

    return CapabilitySet.all()


# ═══════════════════════════════════════════════════════════════════════════
# Curation: PDFs as sources
# ═══════════════════════════════════════════════════════════════════════════


class CuratePdfTests(unittest.TestCase):
    def test_binary_dropped_always(self) -> None:
        r = {"url": "https://a.com/file.zip", "title": "Zip thing", "snippet": "s"}
        self.assertEqual(curate.curate([r], "file", allow_pdf=True), [])

    def test_pdf_dropped_by_default(self) -> None:
        r = {"url": "https://a.com/report.pdf", "title": "Annual Report 2026",
             "snippet": "a full annual report on the topic"}
        self.assertEqual(curate.curate([r], "report"), [])

    def test_pdf_kept_when_allowed(self) -> None:
        r = {"url": "https://a.com/report.pdf", "title": "Annual Report 2026",
             "snippet": "a full annual report on the topic"}
        out = curate.curate([r], "annual report", allow_pdf=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["domain"], "a.com")


# ═══════════════════════════════════════════════════════════════════════════
# mine_followups
# ═══════════════════════════════════════════════════════════════════════════


class MineFollowupsTests(unittest.TestCase):
    def test_mines_cross_page_phrase(self) -> None:
        pages = [
            {"domain": "a.com", "url": "https://a.com/1",
             "text": "The corbomite lattice fails under thermal load. "
                     "Most practitioners miss the corbomite lattice entirely. "
                     "Other topics are less interesting."},
            {"domain": "b.com", "url": "https://b.com/2",
             "text": "Our measurements of the corbomite lattice showed drift. "
                     "Unrelated filler sentence goes here for balance."},
        ]
        out = mine_followups("thermal drift in systems", pages, max_followups=2)
        self.assertTrue(any("corbomite lattice" in p for p in out), out)

    def test_excludes_query_phrases(self) -> None:
        pages = [{"domain": "a.com", "url": "u",
                  "text": "thermal drift in systems is common. " * 5}]
        out = mine_followups("thermal drift in systems", pages, max_followups=3)
        for p in out:
            self.assertNotIn(p, "thermal drift in systems")

    def test_stopword_phrases_skipped(self) -> None:
        pages = [{"domain": "a.com", "url": "u",
                  "text": "this is that of the in on for. " * 10}]
        out = mine_followups("something", pages, max_followups=3)
        self.assertEqual(out, [])

    def test_respects_cap(self) -> None:
        text = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 8)
        pages = [{"domain": "a.com", "url": "u", "text": text}]
        out = mine_followups("alpha systems", pages, max_followups=2)
        self.assertLessEqual(len(out), 2)


# ═══════════════════════════════════════════════════════════════════════════
# The dig loop (stubbed engine, power unlocked)
# ═══════════════════════════════════════════════════════════════════════════

YEAR = time.localtime().tm_year

DIG_PAGES = {
    "https://a.com/one": (
        "The corbomite lattice is the heart of the system. "
        "Every measurement of the corbomite lattice showed drift. "
        "The thermal envelope matters. "
        "Further work on the corbomite lattice is ongoing."
    ),
    "https://b.com/two": (
        "Independent analysis of the corbomite lattice in " + str(YEAR) + ". "
        "The corbomite lattice degrades under sustained load. "
        "Results are consistent with earlier findings."
    ),
    "https://e.com/corbomite": (
        "A dedicated page on the corbomite lattice failure mode. "
        "This source only appears for follow-up queries about the lattice."
    ),
    "https://f.com/internals": (
        "The internals of the lattice, linked out from a top source. "
        "Deep technical detail lives here."
    ),
    "https://d.com/report.pdf": (
        "The full report on the corbomite lattice, as a PDF document. "
        "Page one of the report describes the lattice in detail."
    ),
}

DIG_RESULTS = [
    {"url": "https://a.com/one", "title": "Lattice guide",
     "snippet": "the corbomite lattice guide"},
    {"url": "https://b.com/two", "title": "Independent analysis",
     "snippet": "independent analysis of the lattice"},
    {"url": "https://d.com/report.pdf", "title": "Full lattice report 2026",
     "snippet": "the full report on the corbomite lattice"},
]

DIG_FOLLOWUP_RESULTS = [
    {"url": "https://e.com/corbomite", "title": "Corbomite failure mode",
     "snippet": "the corbomite lattice failure mode"},
]


class DigLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-dig-")
        self.context = _make_context(self.tmp)
        self.context.router = _FakeRouter()
        self.engine = SearchEngine(self.context)
        self.engine.search = self._stub_search
        self.engine.read = self._stub_read
        self._unlock_power()
        self.researcher = DeepResearcher(self.context, engine=self.engine,
                                         max_pages=4, follow_links=3)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _unlock_power(self) -> None:
        from nomorals.agents.power import power_mode_for

        self.context.settings.partner.owner_key = "test-key"
        result = power_mode_for(self.context).unlock("test-key", actor="test")
        self.assertTrue(result["ok"], result)

    def _stub_search(self, sub: str, max_results: int = 8, **kw: Any) -> list:
        low = sub.lower()
        # the mined follow-up carries BOTH the query's words and the new term;
        # the base sub-queries (incl. freshness variants) carry only the query
        if "corbomite lattice" in low and "thermal" in low:
            return [dict(r) for r in DIG_FOLLOWUP_RESULTS]
        return [dict(r) for r in DIG_RESULTS]

    def _stub_read(self, url: str, max_chars: int = 40000, **kw: Any) -> dict | None:
        if url not in DIG_PAGES:
            return None
        page = {"url": url, "title": url.split("/")[2], "domain": url.split("/")[2],
                "text": DIG_PAGES[url], "chars": len(DIG_PAGES[url]), "pdf": False,
                "links": []}
        if url == "https://a.com/one":
            page["links"] = [("https://f.com/internals", "corbomite lattice internals"),
                             ("https://a.com/self", "same domain link"),
                             ("https://junk.com/file.pdf", "pdf anchor")]
        if url.endswith(".pdf"):
            page["pdf"] = True
        return page

    def test_dig_mines_followup_and_reads_new_page(self) -> None:
        report = self.researcher.run("thermal drift in systems")
        self.assertTrue(report["dig"])
        # a follow-up query was mined from the read pages and issued
        self.assertTrue(any("corbomite lattice" in f for f in report["followups"]),
                        report["followups"])
        # the follow-up-only page ended up in the corpus
        self.assertIn("https://e.com/corbomite", report["pages_read"])
        self.assertGreaterEqual(len(report["sources"]), 3)

    def test_dig_follows_external_links(self) -> None:
        report = self.researcher.run("thermal drift in systems")
        self.assertIn("https://f.com/internals", report["external_followed"])
        self.assertIn("https://f.com/internals", report["pages_read"])
        # same-domain and junk links are not followed
        self.assertNotIn("https://a.com/self", report["external_followed"])
        self.assertNotIn("https://junk.com/file.pdf", report["external_followed"])

    def test_dig_reads_pdf_sources(self) -> None:
        report = self.researcher.run("thermal drift in systems")
        self.assertIn("https://d.com/report.pdf", report["pdfs_read"])
        self.assertIn("https://d.com/report.pdf", report["pages_read"])

    def test_no_dig_skips_everything(self) -> None:
        self.researcher.dig = False
        report = self.researcher.run("thermal drift in systems")
        self.assertFalse(report["dig"])
        self.assertEqual(report["followups"], [])
        self.assertEqual(report["external_followed"], [])
        # pdf still readable via the normal top results (curate allow_pdf)
        self.assertIn("https://d.com/report.pdf", report["pdfs_read"])

    def test_report_shape_backcompat(self) -> None:
        report = self.researcher.run("thermal drift in systems")
        for key in ("id", "query", "mode", "sub_queries", "results", "sources",
                    "citations", "sections", "pages_read", "pages", "summary",
                    "model_summary", "error", "seconds", "followups",
                    "external_followed", "pdfs_read", "dig"):
            self.assertIn(key, report)
        self.assertEqual([s["n"] for s in report["sources"]],
                         list(range(1, len(report["sources"]) + 1)))
        for s in report["sources"]:
            self.assertEqual(report["citations"][str(s["n"])], s["url"])


# ═══════════════════════════════════════════════════════════════════════════
# Engine: PDF reading + external link extraction
# ═══════════════════════════════════════════════════════════════════════════


class EngineReadPdfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-pdfsrc-")
        self.context = _make_context(self.tmp)
        self.engine = SearchEngine(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_read_extracts_pdf_text(self) -> None:
        data = render_pdf("The lattice degrades under sustained thermal load. " * 10,
                          title="Lattice Report")

        class _Resp:
            status = 200
            url = "https://a.com/report.pdf"
            headers = {"content-type": "application/pdf"}

            @property
            def ok(self) -> bool:
                return True

            @property
            def body(self) -> bytes:
                return data

            @property
            def text(self) -> str:
                raise AssertionError("pdf must not be decoded as text")

            @property
            def content_type(self) -> str:
                return "application/pdf"

        self.engine._client = type("_C", (), {"get": staticmethod(lambda url, **k: _Resp())})
        self.engine._robots.allowed = lambda url, ua="*", client=None: True
        page = self.engine.read("https://a.com/report.pdf")
        self.assertIsNotNone(page)
        self.assertTrue(page["pdf"])
        self.assertIn("lattice degrades", page["text"])

    def test_extract_external_links(self) -> None:
        markup = (
            '<a href="https://other.com/deep">deep link</a>'
            '<a href="https://other.com/deep">dup</a>'
            '<a href="/relative">same host</a>'
            '<a href="https://self.com/x">self</a>'
            '<a href="https://junk.com/a.pdf">pdf</a>'
            '<a href="mailto:x@y.z">mail</a>'
            '<a href="//proto.com/p">protocol-relative</a>'
        )
        links = _extract_external_links(markup, "self.com")
        urls = [u for u, _ in links]
        self.assertEqual(urls, ["https://other.com/deep", "https://proto.com/p"])
        self.assertEqual(links[0][1], "deep link")


if __name__ == "__main__":
    unittest.main()
