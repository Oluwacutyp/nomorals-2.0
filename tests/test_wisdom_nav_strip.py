"""Nav-boilerplate strip for WisdomKeeper ingest (2026-11-14).

The gospels.net Philip/Mary pages are Squarespace sites: a <header>
wrapping five <nav> menus ("Welcome", "Sayings of Jesus: P.Oxy 5575
Gospel of Thomas ...") sits at the top of every page, and the document
engine faithfully extracted it as text — so ask("Philip said") returned
the site menu as its top passage. ArchiveIngestor.parse now strips
nav/header/footer/aside subtrees (plus common nav class/id patterns)
from the DOM before parsing, so the stored text is the text's own
words. The strip runs at ingest time only; ask() is untouched.

The nav fixture is built from the live page: the <header> block is
verbatim from gospels.net/philip and the content <p>s are verbatim
from the same page. The <footer>, <aside>, <div class="main-nav">
and <div id="topNav"> are synthetic (the live page has none) and pin
those branches. All HTTP is mocked; no test touches the network.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.errors import ProviderError
from nomorals.core.http import HttpResponse
from nomorals.wisdom import ArchiveIngestor

_FIXTURES = Path(__file__).parent / "fixtures"
NAV_URL = "https://www.gospels.net/philip"

#: Strings that appear ONLY in the site menu (never in the gospel text).
NAV_MARKERS = (
    "Sayings of Jesus:",
    "P.Oxy 5575",
    "Gospel of Thomas",
    "Infancy Gospel of James",
    "Odes of Solomon",
)

#: Marker strings of the synthetic footer/aside/class-nav/id-nav blocks.
SYNTHETIC_MARKERS = (
    "FOOTER-BOILERPLATE-MARKER",
    "ASIDE-BOILERPLATE-MARKER",
    "CLASSNAV-BOILERPLATE-MARKER",
    "IDNAV-BOILERPLATE-MARKER",
)


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-nav-strip-test-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


class _FakeHttp:
    """Stand-in for HttpClient: serves fixture bytes for the mirror URL
    only; every other URL raises like a refused connection."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if url in self.mapping:
            return HttpResponse(status=200, body=self.mapping[url],
                                headers={}, url=url)
        raise ProviderError(f"connection refused by {url}",
                            retryable=False)


def _make_ingestor(mapping):
    ctx, tmp = _ctx()
    ing = ArchiveIngestor(ctx)
    ing._http = _FakeHttp(mapping)
    ing._robots = SimpleNamespace(allowed=lambda *a, **k: True)
    return ing, tmp


def _nav_fixture() -> bytes:
    return (_FIXTURES / "wisdom-gospels-nav.html").read_bytes()


class ParseStripTests(unittest.TestCase):
    def test_parse_strips_squarespace_nav_and_header(self):
        ing, _tmp = _make_ingestor({})
        text = ing.parse(_nav_fixture(), "philip")
        for marker in NAV_MARKERS:
            self.assertNotIn(marker, text)
        self.assertNotIn("Welcome", text)
        # Real content survives: the translator's notice, a chapter
        # heading, and a real "said" sentence.
        self.assertIn("committed to the public domain", text)
        self.assertIn("Overcoming the World", text)
        self.assertIn("The Lord said to the", text)

    def test_parse_strips_footer_aside_and_class_id_nav(self):
        ing, _tmp = _make_ingestor({})
        text = ing.parse(_nav_fixture(), "philip")
        for marker in SYNTHETIC_MARKERS:
            self.assertNotIn(marker, text)
        self.assertIn("The Lord said to the", text)

    def test_strip_returns_input_unchanged_for_clean_and_non_html(self):
        ing, _tmp = _make_ingestor({})
        plain = b"Just some plain text, no HTML at all.\n"
        self.assertIs(ing.strip_site_boilerplate(plain), plain)
        excerpt = (_FIXTURES / "wisdom-gospel-philip-excerpt.html"
                   ).read_bytes()
        self.assertIs(ing.strip_site_boilerplate(excerpt), excerpt)
        clean_html = (b"<!doctype html><html><head><title>T</title></head>"
                      b"<body><h1>Heading</h1><p>Real content here.</p>"
                      b"</body></html>")
        self.assertIs(ing.strip_site_boilerplate(clean_html), clean_html)
        # The no-op path parses exactly as before.
        self.assertIn("Real content here.",
                      ing.parse(clean_html, "clean.html"))


class IngestStripTests(unittest.TestCase):
    def test_ingest_philip_no_nav_passages(self):
        ing, _tmp = _make_ingestor({NAV_URL: _nav_fixture()})
        ing.corpus.seed()
        entry = ing.ingest_entry(ing.corpus.get("gospel-of-philip"))
        state = ing.corpus.work_state(entry)
        self.assertTrue(state["ingested"])
        self.assertGreater(state["chars"], 200)
        ans = ing.corpus.ask("Philip said", top=5)
        slugs = [h.slug for h in ans.passages]
        self.assertIn("gospel-of-philip", slugs)
        for hit in ans.passages:
            for marker in NAV_MARKERS + SYNTHETIC_MARKERS:
                self.assertNotIn(marker, hit.snippet)
        # The real "said" sentence survives ingest (pinned at parse
        # level above); the passage returned here is the text's own
        # opening, not the site menu.
        ans2 = ing.corpus.ask("The Lord said", top=5)
        self.assertIn("gospel-of-philip",
                      [h.slug for h in ans2.passages])
        self.assertIn("public domain", ans2.passages[0].snippet)


if __name__ == "__main__":
    unittest.main()
