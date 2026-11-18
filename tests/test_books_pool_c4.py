"""Census C4 — books pool wiring: the NOT-wired verdict, pinned.

The five descriptor adapters (gutenberg, scribblehub, ao3, wattpad,
webnovel) were probed live on 2026-11-12 and NONE could be honestly
wired into the search pool:

* gutenberg — Gutendex API down (HTTP 503 + 30s read timeouts;
  gutenberg.org itself HTTP 200).  The adapter fails honestly, but
  wiring it now would make every search pay a 30s timeout against a
  down API.
* scribblehub — plain fetch Cloudflare "Just a moment" 403; a
  browser-like fetch returns a near-empty shell; the Fetcher's
  browser-context fallback is untestable on this box (BrowserSession
  SSL failure even on example.com).
* ao3 — HTTP 418 "Request temporarily denied" from plain fetch AND a
  browser-like fetch; the wall is not client-specific.
* wattpad — requires_key (WATTPAD_API_KEY), no search template;
  search raises the honest key error with no key set.
* webnovel — plain fetch gets an empty reply, but a browser-like
  fetch passes the wall: the search page is live and server-rendered
  with real book results.  The adapter's link-extraction shape
  against the real markup is still unverified (the browser fetch
  exposes rendered text, not raw hrefs), and with no
  ``search_link_must_contain`` set the adapter would return nav
  links as stories.

The evidence and the named next probe for each live in the descriptor
table's notes (``books/descriptors.py``).

52-websites #10 (2026-11-17) SUPERSEDES the C4 verdict for
gutenberg ONLY: the re-probe still could not show Gutendex
answering from this box (DNS resolves to CGNAT 198.18.76.133 and
reads time out even at 30s — box-egress-shaped, not proven down;
gutenberg.org itself served a genuine 'Temporarily Down for
Maintenance' 503 page), but the adapter joins the pool with the
failure cost engineered away — 8s search timeout, 15-minute
cooldown after a failed search, last in pool order — so a down
Gutendex costs at most one short bounded wait per cooldown window
instead of 30s on every search.  The pins below now pin THAT
contract for gutenberg (in the pool, bounded) while the other four
stay pinned absent until their probes land — never by editing the
pool to pad the count.  A live search succeeding is still owed on
a host with real egress.  No network in this suite: every fetch is
stubbed.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import sources as book_sources
from nomorals.books.descriptors import (
    DESCRIPTORS,
    DescriptorAdapter,
    GutenbergAdapter,
    descriptor_by_id,
)
from nomorals.books.sources import (
    ADAPTERS,
    BOOK_SEARCH_SOURCES,
    FreeWebNovelAdapter,
    NovelFullAdapter,
    RoyalRoadAdapter,
    SearchHit,
    SourceError,
    search_all,
)
from nomorals.tools.browser import parse_html

#: the four descriptor ids the C4 probe still keeps out of the pool
DESCRIPTOR_IDS = ("scribblehub", "ao3", "wattpad", "webnovel")

FWN_SEARCH = """<html><body>
<a href="/novel/alpha-tale">Alpha Tale</a>
</body></html>"""

NF_SEARCH = """<html><body>
<a href="https://novelfull.net/beta-tale.html">Beta Tale</a>
</body></html>"""

RR_SEARCH = """<html><body>
<a href="https://www.royalroad.com/fiction/111/gamma-tale">Gamma Tale</a>
</body></html>"""

#: a real-shape Gutendex payload (Middlemarch, id 145) for the
#: bounded-search pins — catalogue fields as the API returns them.
GUTENDEX_MIDDLEMARCH = {
    "count": 1,
    "results": [{
        "id": 145,
        "title": "Middlemarch",
        "authors": [{"name": "Eliot, George",
                     "birth_year": 1819, "death_year": 1880}],
        "subjects": ["Didactic fiction", "England -- Fiction"],
        "bookshelves": ["Best Books Ever Listings"],
        "languages": ["en"],
        "copyright": False,
        "media_type": "Text",
        "formats": {
            "text/html": "https://www.gutenberg.org/files/145/145-h/145-h.htm",
            "application/epub+zip": "https://www.gutenberg.org/files/145/145.epub",
            "application/x-mobipocket-ebook": "https://www.gutenberg.org/files/145/145.kindle",
            "text/plain; charset=utf-8": "https://www.gutenberg.org/files/145/145.txt",
            "application/rdf+xml": "https://www.gutenberg.org/ebooks/145.rdf",
        },
        "download_count": 12345,
    }],
}


class _FakeResp:
    """A urlopen response: context manager over canned bytes."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._payload


def _gutendex_urlopen(payload: dict) -> mock.Mock:
    body = json.dumps(payload).encode("utf-8")
    return mock.Mock(return_value=_FakeResp(body))


class _FakeFetcher:
    """Serves canned pages instead of the network."""

    def __init__(self, pages):
        self.pages = pages

    def get(self, url, **_):
        for key in sorted(self.pages, key=len, reverse=True):
            if url.startswith(key):
                return url, parse_html(self.pages[key]), self.pages[key]
        raise SourceError(f"no fixture for {url}")


def _search_adapters():
    return [
        FreeWebNovelAdapter(fetcher=_FakeFetcher({
            "https://freewebnovel.com/search": FWN_SEARCH})),
        NovelFullAdapter(fetcher=_FakeFetcher({
            "https://novelfull.net/search": NF_SEARCH})),
        RoyalRoadAdapter(fetcher=_FakeFetcher({
            "https://www.royalroad.com/fictions/search": RR_SEARCH})),
    ]


class _FakeGutenberg:
    """A stubbed pool Gutenberg adapter (verified, last in order)."""

    name = "gutenberg"
    verified = True

    def search(self, query, *, limit=10):
        return [SearchHit(title="Middlemarch",
                          url="https://www.gutenberg.org/ebooks/145",
                          source="gutenberg", author="Eliot, George")]


class TestBooksPoolC4(unittest.TestCase):
    def test_pool_is_the_proven_four_plus_gutenberg_last(self) -> None:
        self.assertEqual(BOOK_SEARCH_SOURCES, (
            "freewebnovel", "novelfull", "royalroad", "pandanovel",
            "gutenberg"))

    def test_other_descriptor_adapters_still_not_in_the_pool(self) -> None:
        for source_id in DESCRIPTOR_IDS:
            self.assertNotIn(source_id, BOOK_SEARCH_SOURCES)
            self.assertNotIn(
                source_id, [a.name for a in ADAPTERS],
                f"{source_id} must not be in the working pool until "
                "its probe lands")

    def test_search_all_tries_gutenberg_last(self) -> None:
        with mock.patch.object(book_sources, "ADAPTERS",
                               _search_adapters()), \
                mock.patch.object(book_sources, "_gutenberg_adapter",
                                  return_value=_FakeGutenberg()):
            hits = search_all("tale", limit=10)
        self.assertEqual(
            [h.source for h in hits],
            ["freewebnovel", "novelfull", "royalroad", "gutenberg"])
        self.assertTrue(all(
            h.source not in DESCRIPTOR_IDS for h in hits))

    def test_search_all_sources_filter_can_isolate_gutenberg(self) -> None:
        with mock.patch.object(book_sources, "ADAPTERS",
                               _search_adapters()), \
                mock.patch.object(book_sources, "_gutenberg_adapter",
                                  return_value=_FakeGutenberg()):
            hits = search_all("tale", limit=10, sources=["gutenberg"])
        self.assertEqual([h.source for h in hits], ["gutenberg"])

    def test_descriptor_notes_record_the_c4_verdict(self) -> None:
        by_id = {d.id: d for d in DESCRIPTORS}
        for source_id in DESCRIPTOR_IDS:
            desc = by_id[source_id]
            self.assertIn("C4 probe 2026-11-12", desc.notes, source_id)
            self.assertIn(
                f"Do NOT add {source_id} to BOOK_SEARCH_SOURCES",
                desc.notes, source_id)

    def test_gutenberg_notes_record_reprobe_and_supersession(self) -> None:
        desc = descriptor_by_id("gutenberg")
        self.assertIsNotNone(desc)
        assert desc is not None
        self.assertIn("Re-probe 2026-11-17", desc.notes)
        self.assertIn("SUPERSEDED for gutenberg only", desc.notes)
        self.assertIn("8s search timeout", desc.notes)

    def test_wattpad_search_is_an_honest_key_error(self) -> None:
        desc = descriptor_by_id("wattpad")
        self.assertIsNotNone(desc)
        adapter = DescriptorAdapter(desc)
        with self.assertRaises(SourceError) as ctx:
            adapter.search("tale", limit=5)
        self.assertIn("WATTPAD_API_KEY", str(ctx.exception))

    def test_gutenberg_search_fails_honestly_when_the_api_is_down(self) -> None:
        adapter = GutenbergAdapter()
        with mock.patch("urllib.request.urlopen",
                        side_effect=TimeoutError("timed out")), \
                self.assertRaises(SourceError) as ctx:
            adapter.search("pride", limit=3)
        self.assertIn("gutenberg search failed", str(ctx.exception))

    def test_gutenberg_search_uses_the_short_timeout(self) -> None:
        adapter = GutenbergAdapter()
        opener = _gutendex_urlopen(GUTENDEX_MIDDLEMARCH)
        with mock.patch("urllib.request.urlopen", opener):
            hits = adapter.search("middlemarch", limit=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "Middlemarch")
        self.assertEqual(hits[0].author, "Eliot, George")
        self.assertEqual(hits[0].url,
                         "https://www.gutenberg.org/ebooks/145")
        self.assertEqual(hits[0].snippet, "12345 downloads")
        _, kwargs = opener.call_args
        self.assertEqual(kwargs.get("timeout"),
                         GutenbergAdapter._SEARCH_TIMEOUT_S)
        self.assertEqual(GutenbergAdapter._SEARCH_TIMEOUT_S, 8.0)

    def test_gutenberg_search_cools_down_after_failure(self) -> None:
        adapter = GutenbergAdapter()
        opener = mock.Mock(side_effect=TimeoutError("timed out"))
        with mock.patch("urllib.request.urlopen", opener):
            with self.assertRaises(SourceError) as ctx:
                adapter.search("pride", limit=3)
            self.assertIn("gutenberg search failed", str(ctx.exception))
            # second search fails FAST, naming the cooldown — the
            # network is not touched again
            with self.assertRaises(SourceError) as ctx2:
                adapter.search("pride", limit=3)
            self.assertIn("cooling down", str(ctx2.exception))
        self.assertEqual(opener.call_count, 1)
        # cooldown expiry retries the network
        adapter._search_down_until = 0.0
        opener2 = _gutendex_urlopen(GUTENDEX_MIDDLEMARCH)
        with mock.patch("urllib.request.urlopen", opener2):
            hits = adapter.search("middlemarch", limit=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(opener2.call_count, 1)

    def test_gutenberg_cooldown_is_not_registry_evidence(self) -> None:
        from nomorals.books.sources import _book_error_evidence
        kind, _ = _book_error_evidence(SourceError(
            "gutenberg: Gutendex unreachable — search cooling down "
            "for 14m 59s more"))
        self.assertEqual(kind, "none")

    def test_descriptor_adapters_still_serve_direct_urls(self) -> None:
        # NOT-wired means "absent from the search pool", not "dead":
        # a pasted URL still resolves to its descriptor adapter.
        from nomorals.books.sources import adapter_for_url
        self.assertIsInstance(
            adapter_for_url("https://www.gutenberg.org/ebooks/1342"),
            GutenbergAdapter)
        self.assertIsInstance(
            adapter_for_url("https://www.scribblehub.com/series/1/x"),
            DescriptorAdapter)
        self.assertIsInstance(
            adapter_for_url("https://archiveofourown.org/works/123"),
            DescriptorAdapter)
        self.assertIsInstance(
            adapter_for_url("https://www.webnovel.com/book/123"),
            DescriptorAdapter)


if __name__ == "__main__":
    unittest.main()
