"""Web tools god-tier pass tests.

Covers (no live network — HTTP is mocked):
- DuckDuckGo result parsing against canned HTML (incl. /l/?uddg= redirect unwrap)
- Bing result parsing against canned HTML (li.b_algo blocks)
- Bing fallback triggers when DDG returns nothing or errors
- URL dedupe across engines
- per-engine short timeouts (8s search client)
- charset detection from Content-Type header and <meta charset>
- readability extraction prefers the article body over nav/footer
- web_fetch extract=True path and the web_extract tool
- web_fetch/web_search backward compatibility (old call shapes still work)
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.tools import web
from nomorals.tools.registry import ToolRegistry

DDG_HTML = """
<html><body>
<a class="result__a" rel="noopener" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha&amp;rut=x">Alpha title</a>
<a class="result__a" href="https://example.com/beta">Beta title</a>
<a class="result__a" href="javascript:void(0)">not-a-result</a>
</body></html>
"""

BING_HTML = """
<html><body><ol id="b_results">
<li class="b_algo"><h2><a href="https://example.com/beta">Beta via Bing</a></h2>
<div class="b_caption"><p>Bing's snippet for beta.</p></div></li>
<li class="b_algo"><h2><a href="https://example.org/gamma">Gamma title</a></h2></li>
<li class="b_algo"><h2><a href="javascript:void(0)">junk</a></h2></li>
</ol></body></html>
"""

ARTICLE_HTML = b"""<html><head><title>Test Article</title></head><body>
<nav><a href="/a">Home</a> <a href="/b">About</a> <a href="/c">Contact</a>
<a href="/d">Privacy</a> <a href="/e">Terms</a> <a href="/f">Careers</a></nav>
<aside><p>Sponsored sidebar widget with promotional copy and links everywhere.</p></aside>
<article>
<h1>The Real Article</h1>
<p>This is the first paragraph of the genuine article content. It is deliberately
long so the block comfortably exceeds the minimum length the extractor demands
before it will treat a region as the main body of the page.</p>
<p>The second paragraph continues the article with further meaningful prose about
the topic, adding more substance and still more length to the genuine content.</p>
</article>
<footer><p>Copyright 2026 Acme Corp. All rights reserved. Footer link farm follows.</p></footer>
</body></html>"""


def _resp(body=b"", *, content_type="text/html", url="https://example.com/", text=None):
    ns = SimpleNamespace(
        status=200,
        body=body,
        headers={"content-type": content_type} if content_type else {},
        url=url,
        ok=True,
        content_type=content_type,
    )
    ns.text = text if text is not None else body.decode("utf-8", errors="replace")
    return ns


class WebGodtierTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("nomorals.tools.web.HttpClient")
        self.addCleanup(patcher.stop)
        self.MockClient = patcher.start()
        self.client = self.MockClient.return_value
        self.robots = mock.patch.object(web, "_robots")
        self.mock_robots = self.robots.start()
        self.addCleanup(self.robots.stop)
        self.mock_robots.allowed.return_value = True
        self.registry = ToolRegistry(None)
        web.register(self.registry)

    def _call(self, name, **kwargs):
        outcome = self.registry.call(name, **kwargs)
        self.assertTrue(outcome.ok, f"{name} failed: {outcome.error}")
        return outcome.value

    # ── DDG parsing ──────────────────────────────────────────────────────
    def test_parse_ddg_unwraps_redirect(self):
        results = web._parse_results(DDG_HTML)
        urls = [r["url"] for r in results]
        self.assertIn("https://example.com/alpha", urls)
        self.assertIn("https://example.com/beta", urls)
        self.assertNotIn("javascript:void(0)", urls)  # junk href filtered
        titles = dict((r["url"], r["title"]) for r in results)
        self.assertEqual(titles["https://example.com/alpha"], "Alpha title")

    # ── Bing parsing ─────────────────────────────────────────────────────
    def test_parse_bing_blocks(self):
        results = web._parse_bing(BING_HTML)
        urls = [r["url"] for r in results]
        self.assertEqual(urls, ["https://example.com/beta", "https://example.org/gamma"])
        self.assertIn("Bing's snippet", results[0]["snippet"])
        self.assertEqual(results[1]["title"], "Gamma title")

    def test_per_engine_parsers_exist_for_video_finder(self):
        # nomorals/media/video.py imports these names; they must exist.
        ddg = web._parse_ddg(DDG_HTML)
        self.assertTrue(ddg)
        lite = web._parse_lite('<a rel="nofollow" href="https://example.com/z">Zed</a>')
        self.assertEqual(lite[0]["url"], "https://example.com/z")
        mojeek = web._parse_mojeek(
            '<a class="title" href="https://example.com/m">Mojeek hit</a>')
        self.assertEqual(mojeek[0]["url"], "https://example.com/m")

    # ── fallback behaviour ───────────────────────────────────────────────
    def _search_dispatch(self, ddg_mode):
        def get(url, **kwargs):
            if "bing.com" in url:
                return _resp(text=BING_HTML)
            if ddg_mode == "error":
                raise Exception("ddg is down")
            return _resp(text="<html><body>no results</body></html>")
        return get

    def test_fallback_when_ddg_empty(self):
        self.client.get.side_effect = self._search_dispatch("empty")
        result = self._call("web_search", query="god tier agents")
        self.assertEqual(result["engines"], ["bing"])
        urls = [r["url"] for r in result["results"]]
        self.assertIn("https://example.org/gamma", urls)

    def test_fallback_when_ddg_errors(self):
        self.client.get.side_effect = self._search_dispatch("error")
        result = self._call("web_search", query="god tier agents")
        self.assertEqual(result["engines"], ["bing"])
        self.assertEqual(result["count"], 2)

    def test_no_fallback_when_ddg_hits(self):
        def get(url, **kwargs):
            if "bing.com" in url:
                return _resp(text=BING_HTML)
            return _resp(text=DDG_HTML)
        self.client.get.side_effect = get
        result = self._call("web_search", query="x")
        self.assertEqual(result["engines"], ["duckduckgo"])
        urls = [r["url"] for r in result["results"]]
        self.assertIn("https://example.com/alpha", urls)

    def test_search_timeout_is_short(self):
        timeouts = [c.kwargs.get("timeout") for c in self.MockClient.call_args_list]
        self.assertIn(web.SEARCH_TIMEOUT, timeouts)
        self.assertLessEqual(web.SEARCH_TIMEOUT, 8.0)

    # ── dedupe ───────────────────────────────────────────────────────────
    def test_dedupe_normalizes_urls(self):
        items = [
            {"url": "https://Example.com/beta", "title": "a", "snippet": ""},
            {"url": "https://example.com/beta/", "title": "b", "snippet": ""},
            {"url": "https://example.com/beta?utm=x", "title": "c", "snippet": ""},
            {"url": "https://example.org/gamma", "title": "d", "snippet": ""},
        ]
        deduped = web._dedupe_results(items)
        self.assertEqual([d["url"] for d in deduped],
                         ["https://Example.com/beta", "https://example.org/gamma"])

    # ── charset ──────────────────────────────────────────────────────────
    def test_charset_from_header(self):
        body = "Caf\xe9 au lait".encode("latin-1")
        text = web._decode_body(body, "text/html; charset=iso-8859-1")
        self.assertIn("Caf\xe9", text)

    def test_charset_from_meta_tag(self):
        body = b'<html><head><meta charset="windows-1252"></head><body><p>Caf\xe9</p></body></html>'
        text = web._decode_body(body, "text/html")
        self.assertIn("Caf\xe9", text)

    def test_charset_falls_back_to_utf8_replace(self):
        body = b"<p>\xff\xfe invalid</p>"
        text = web._decode_body(body, "text/html")
        self.assertIn("invalid", text)

    # ── readability ──────────────────────────────────────────────────────
    def test_readability_picks_article_over_chrome(self):
        article = web.readability_extract(ARTICLE_HTML.decode("utf-8"))
        self.assertEqual(article["title"], "Test Article")
        self.assertIn("genuine article content", article["text"])
        self.assertNotIn("Copyright 2026", article["text"])
        self.assertNotIn("Sponsored sidebar", article["text"])
        self.assertEqual(article["words"], len(article["text"].split()))

    def test_readability_without_article_tag_scores_blocks(self):
        markup = (
            "<html><body>"
            "<div class='menu'>" + "<a href='/x'>link</a> " * 30 + "</div>"
            "<div class='post'>" + "<p>Real body prose. " * 40 + "</p></div>"
            "</body></html>"
        )
        article = web.readability_extract(markup)
        self.assertIn("Real body prose", article["text"])
        self.assertNotIn("class='menu'", article["text"])

    # ── tools ────────────────────────────────────────────────────────────
    def test_web_extract_tool(self):
        self.client.get.return_value = _resp(body=ARTICLE_HTML, url="https://example.com/a")
        result = self._call("web_extract", url="https://example.com/a")
        self.assertEqual(result["title"], "Test Article")
        self.assertIn("genuine article content", result["text"])
        self.assertNotIn("Copyright 2026", result["text"])
        self.assertGreater(result["words"], 30)

    def test_web_fetch_extract_param(self):
        self.client.get.return_value = _resp(body=ARTICLE_HTML, url="https://example.com/a")
        result = self._call("web_fetch", url="https://example.com/a", extract=True)
        self.assertIn("genuine article content", result["text"])
        self.assertNotIn("Copyright 2026", result["text"])

    def test_web_fetch_default_is_full_page_text(self):
        self.client.get.return_value = _resp(body=ARTICLE_HTML, url="https://example.com/a")
        result = self._call("web_fetch", url="https://example.com/a")
        # default (extract=False) converts the whole page, chrome included
        self.assertIn("Copyright 2026", result["text"])

    def test_web_fetch_old_signature_still_works(self):
        self.client.get.return_value = _resp(body=b"<title>T</title><p>hi</p>")
        result = self._call("web_fetch", url="https://example.com/a", max_chars=10, raw=True)
        self.assertEqual(result["title"], "T")
        self.assertTrue(result["truncated"])

    def test_web_fetch_decodes_charset(self):
        body = "Caf\xe9".encode("latin-1")
        self.client.get.return_value = _resp(
            body=body, content_type="text/html; charset=iso-8859-1")
        result = self._call("web_fetch", url="https://example.com/a")
        self.assertIn("Caf\xe9", result["text"])

    def test_web_search_backward_compat(self):
        def get(url, **kwargs):
            return _resp(text=DDG_HTML)
        self.client.get.side_effect = get
        result = self._call("web_search", query="x", max_results=1, site="example.com")
        self.assertEqual(result["count"], 1)
        self.assertIn("site:example.com", result["query"])

    def test_robots_still_enforced(self):
        self.mock_robots.allowed.return_value = False
        for tool in ("web_fetch", "web_extract"):
            outcome = self.registry.call(tool, url="https://example.com/a")
            self.assertFalse(outcome.ok)
            self.assertIn("robots.txt", str(outcome.error))


if __name__ == "__main__":
    unittest.main()
