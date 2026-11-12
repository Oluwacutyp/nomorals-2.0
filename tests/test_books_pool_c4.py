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
table's notes (``books/descriptors.py``).  These pins are removable
only when a probe lands for that source — never by editing the pool
to pad the count.  No network in this suite: every fetch is stubbed.
"""

from __future__ import annotations

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
    SourceError,
    search_all,
)
from nomorals.tools.browser import parse_html

#: the five descriptor ids the C4 probe covered
DESCRIPTOR_IDS = ("gutenberg", "scribblehub", "ao3", "wattpad", "webnovel")

FWN_SEARCH = """<html><body>
<a href="/novel/alpha-tale">Alpha Tale</a>
</body></html>"""

NF_SEARCH = """<html><body>
<a href="https://novelfull.net/beta-tale.html">Beta Tale</a>
</body></html>"""

RR_SEARCH = """<html><body>
<a href="https://www.royalroad.com/fiction/111/gamma-tale">Gamma Tale</a>
</body></html>"""


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


class TestBooksPoolC4(unittest.TestCase):
    def test_pool_is_exactly_the_proven_four_in_order(self) -> None:
        self.assertEqual(BOOK_SEARCH_SOURCES, (
            "freewebnovel", "novelfull", "royalroad", "pandanovel"))

    def test_no_descriptor_adapter_is_in_the_pool(self) -> None:
        for source_id in DESCRIPTOR_IDS:
            self.assertNotIn(source_id, BOOK_SEARCH_SOURCES)
            self.assertNotIn(
                source_id, [a.name for a in ADAPTERS],
                f"{source_id} must not be in the working pool until "
                "its probe lands")

    def test_search_all_returns_the_proven_four_first(self) -> None:
        with mock.patch.object(book_sources, "ADAPTERS",
                               _search_adapters()):
            hits = search_all("tale", limit=10)
        self.assertEqual(
            [h.source for h in hits],
            ["freewebnovel", "novelfull", "royalroad"])
        self.assertTrue(all(
            h.source not in DESCRIPTOR_IDS for h in hits))

    def test_descriptor_notes_record_the_c4_verdict(self) -> None:
        by_id = {d.id: d for d in DESCRIPTORS}
        for source_id in DESCRIPTOR_IDS:
            desc = by_id[source_id]
            self.assertIn("C4 probe 2026-11-12", desc.notes, source_id)
            self.assertIn(
                f"Do NOT add {source_id} to BOOK_SEARCH_SOURCES",
                desc.notes, source_id)

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
