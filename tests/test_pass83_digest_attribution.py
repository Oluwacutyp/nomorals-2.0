"""PASS #83: digests credit the publisher — stored attribution, truthful
grouping.

Pass #80 taught ``parse_feed`` to read the RSS ``<source>`` element so a
live aggregator-carried item knows its publisher (Reuters, not Google
News).  Two residues are closed here:

(1) GROUPING: ``NewsAgent.run()``'s digest still headered blocks by FEED
    name, so a Reuters story via the generic Google News feed rendered
    under "■ Google News" — the publisher invisible in the briefing.
(2) STORAGE: ``news_items`` kept only the feed name in ``source``, so
    every stored-path consumer (morning briefing, /news status, later
    pulses) lost the publisher forever.

Pins:
(a) a digest from a fixture run headers a Reuters story by "Reuters"
    whether it arrived via the generic Google News feed or the Reuters
    site-scoped feed; an unattributed outlet's items render under their
    feed name byte-identically to before.
(b) stored rows persist publisher name + URL; the stored path (morning
    briefing NewsProvider) labels + groups by publisher identically to
    the live path.
(c) legacy rows (pre-pass shape: no attribution columns, inserted with
    the old column list) render byte-identical to today everywhere.
(d) a many-publisher aggregator fixture groups per publisher with the
    digest's existing item cap still authoritative — headers are
    bounded by the item count, unattributed items still coalesce under
    their feed header, and no "■ Google News" header appears for
    attributed items.
(e) the briefing's per-source anti-monopoly cap keys on the publisher:
    Reuters via two feeds counts ONCE (max 3 picked), while two
    different publishers carried by one aggregator feed no longer share
    a single cap slot.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.news import NewsAgent, publisher_of
from nomorals.storage.db import Database

GN_FEED = ("Google News", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en")
REUTERS_FEED = (
    "Reuters",
    "https://news.google.com/rss/search?q=site:reuters.com&hl=en-US&gl=US&ceid=US:en")
PLAIN_FEED = ("Plain Outlet", "https://plain.example.com/feed")

# Generic aggregator shape: two publishers in one feed.
GN_MIXED_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
    "<title>Top stories - Google News</title>"
    "<item><title>Fed holds rates steady - Reuters</title>"
    "<link>https://news.google.com/rss/articles/CBMiAAA111?oc=5</link>"
    "<description>Central bank keeps rates unchanged.</description>"
    "<pubDate>Wed, 21 Oct 2026 10:00:00 GMT</pubDate>"
    '<source url="https://www.reuters.com">Reuters</source></item>'
    "<item><title>Storm lashes coast - BBC News</title>"
    "<link>https://news.google.com/rss/articles/CBMiCCC333?oc=5</link>"
    "<description>Weather body.</description>"
    "<pubDate>Wed, 21 Oct 2026 09:30:00 GMT</pubDate>"
    '<source url="https://www.bbc.co.uk">BBC News</source></item>'
    "<item><title>Unattributed wire brief</title>"
    "<link>https://news.google.com/rss/articles/CBMiDDD444?oc=5</link>"
    "<description>No source element on this one.</description>"
    "<pubDate>Wed, 21 Oct 2026 09:00:00 GMT</pubDate></item>"
    "</channel></rss>")

REUTERS_FEED_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
    "<title>Reuters - Google News</title>"
    "<item><title>Markets rally on open - Reuters</title>"
    "<link>https://news.google.com/rss/articles/CBMiBBB222?oc=5</link>"
    "<description>Stocks climb.</description>"
    "<pubDate>Wed, 21 Oct 2026 10:05:00 GMT</pubDate>"
    '<source url="https://www.reuters.com">Reuters</source></item>'
    "</channel></rss>")

PLAIN_XML = (
    '<?xml version="1.0"?><rss version="2.0"><channel>'
    "<item><title>Plain story</title>"
    "<link>https://plain.example.com/plain</link>"
    "<description>Body.</description>"
    "<pubDate>Wed, 21 Oct 2026 09:00:00 GMT</pubDate></item>"
    "</channel></rss>")


def _many_publisher_xml(count: int) -> str:
    items = []
    for i in range(count):
        items.append(
            f"<item><title>Story {i} - Pub{i}</title>"
            f"<link>https://news.google.com/rss/articles/ART{i}?oc=5</link>"
            f"<description>Body {i}.</description>"
            f"<pubDate>Wed, 21 Oct 2026 10:{i:02d}:00 GMT</pubDate>"
            f'<source url="https://pub{i}.example.com">Pub{i}</source></item>')
    return ('<?xml version="1.0"?><rss version="2.0"><channel>'
            + "".join(items) + "</channel></rss>")


def _make_ctx(test, feeds_setting="", cap=10):
    tmp = tempfile.mkdtemp(prefix="pass83-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        news=SimpleNamespace(feeds=feeds_setting, cap=cap))
    return SimpleNamespace(db=db, settings=settings, extras={})


def _run_with_xml(agent, feeds_with_xml):
    """feeds_with_xml: list of ((name, url), xml) pairs."""
    feeds = [f for f, _ in feeds_with_xml]
    by_url = {url: xml for (_, url), xml in feeds_with_xml}

    def _get(url, *args, **kwargs):
        return SimpleNamespace(text=by_url[url])

    with mock.patch.object(agent, "_feeds", return_value=feeds), \
            mock.patch.object(agent, "_client") as client:
        client.get.side_effect = _get
        return agent.run()


def _legacy_insert(db, source, title, url, summary=""):
    """Insert a row with the PRE-PASS column list only."""
    db.execute(
        "INSERT INTO news_items (id, source, title, url, summary, "
        "published, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (url, source, title, url, summary, 0.0, 1761091200.0))


# ── (a) live digest grouping credits the publisher ─────────────────────

class LiveDigestAttributionTests(unittest.TestCase):

    def test_reuters_via_generic_feed_headers_as_reuters(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        report = _run_with_xml(agent, [(GN_FEED, GN_MIXED_XML)])
        digest = report["digest"]
        self.assertIn("■ Reuters", digest)
        self.assertIn("■ BBC News", digest)
        # the Reuters story must NOT sit under the aggregator's header
        reuters_block = digest.split("■ Reuters", 1)[1].split("■", 1)[0]
        self.assertIn("Fed holds rates steady - Reuters", reuters_block)
        # the unattributed item still coalesces under its feed header
        self.assertIn("■ Google News", digest)
        gn_block = digest.split("■ Google News", 1)[1]
        self.assertIn("Unattributed wire brief", gn_block)
        self.assertNotIn("Fed holds rates steady", gn_block)

    def test_reuters_via_reuters_feed_headers_as_reuters(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        report = _run_with_xml(agent, [(REUTERS_FEED, REUTERS_FEED_XML)])
        self.assertIn("■ Reuters", report["digest"])
        self.assertIn("Markets rally on open - Reuters", report["digest"])

    def test_unattributed_feed_renders_byte_identical(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        report = _run_with_xml(agent, [(PLAIN_FEED, PLAIN_XML)])
        # exact legacy rendering: feed header + title — summary line
        self.assertEqual(report["digest"],
                         "■ Plain Outlet\n  • Plain story — Body.")


# ── (b) stored attribution + stored-path parity ────────────────────────

class StoredAttributionTests(unittest.TestCase):

    def test_stored_row_persists_publisher(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        _run_with_xml(agent, [(GN_FEED, GN_MIXED_XML)])
        rows = ctx.db.query(
            "SELECT source, publisher, publisher_url FROM news_items")
        self.assertEqual(len(rows), 3)
        reuters = [r for r in rows if r["publisher"] == "Reuters"]
        self.assertEqual(len(reuters), 1)
        self.assertEqual(reuters[0]["publisher_url"],
                         "https://www.reuters.com")
        # provenance: the carrying feed is still the stored `source`
        self.assertEqual(reuters[0]["source"], "Google News")
        # the unattributed row records NO publisher (NULL), not a guess
        plain = [r for r in rows if not r["publisher"]]
        self.assertEqual(len(plain), 1)
        self.assertEqual(plain[0]["source"], "Google News")
        bbc = [r for r in rows if r["publisher"] == "BBC News"]
        self.assertEqual(len(bbc), 1)
        self.assertEqual(bbc[0]["publisher_url"], "https://www.bbc.co.uk")

    def test_recent_normalizes_stored_rows_to_live_shape(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        _run_with_xml(agent, [(GN_FEED, GN_MIXED_XML)])
        rows = agent.recent(limit=10)
        reuters = [r for r in rows if r.get("publisher") == "Reuters"]
        self.assertEqual(len(reuters), 1)
        self.assertEqual(reuters[0]["source_name"], "Reuters")
        self.assertEqual(reuters[0]["source_url"], "https://www.reuters.com")
        self.assertEqual(publisher_of(reuters[0]), "Reuters")

    def test_stored_path_briefing_labels_publisher(self):
        from nomorals.agents.morning_briefing import NewsProvider
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        _run_with_xml(agent, [(GN_FEED, GN_MIXED_XML)])
        section = NewsProvider().collect(ctx, since=0.0)
        self.assertIsNotNone(section)
        lines = "\n".join(section.lines)
        self.assertIn("(Reuters)", lines)
        self.assertIn("(BBC News)", lines)
        # the feed name survives ONLY as the unattributed item's label —
        # exactly one occurrence, on that item's line
        self.assertIn("• Unattributed wire brief (Google News)", lines)
        self.assertEqual(lines.count("(Google News)"), 1)


# ── (c) legacy rows render byte-identical ──────────────────────────────

class LegacyRowTests(unittest.TestCase):

    def test_legacy_row_briefing_byte_identical(self):
        from nomorals.agents.morning_briefing import NewsProvider
        ctx = _make_ctx(self)
        _legacy_insert(ctx.db, "Google News", "Legacy story",
                       "https://news.google.com/legacy1", "Old body.")
        section = NewsProvider().collect(ctx, since=0.0)
        self.assertIsNotNone(section)
        self.assertEqual(section.lines,
                         ["• Legacy story (Google News)"])
        self.assertEqual(section.items[0]["source"], "Google News")

    def test_legacy_row_publisher_falls_back_to_feed(self):
        ctx = _make_ctx(self)
        _legacy_insert(ctx.db, "BBC News", "Old BBC story",
                       "https://bbc.example.com/old")
        agent = NewsAgent(ctx)
        rows = agent.recent(limit=5)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].get("publisher"))
        self.assertEqual(publisher_of(rows[0]), "BBC News")


# ── (d) many-publisher aggregator: caps hold, no explosion ─────────────

class ManyPublisherTests(unittest.TestCase):

    def test_many_publishers_group_per_publisher_within_cap(self):
        ctx = _make_ctx(self, cap=40)
        agent = NewsAgent(ctx)
        report = _run_with_xml(agent, [(GN_FEED, _many_publisher_xml(15))])
        digest = report["digest"]
        headers = [ln for ln in digest.splitlines() if ln.startswith("■ ")]
        bullets = [ln for ln in digest.splitlines() if ln.startswith("  • ")]
        self.assertEqual(report["items"], 15)
        self.assertEqual(len(bullets), 15)
        # one header per distinct publisher — bounded by the item cap,
        # never more blocks than items, and no aggregator header
        self.assertEqual(len(headers), 15)
        self.assertLessEqual(len(headers), len(bullets))
        self.assertNotIn("■ Google News", headers)
        for i in range(15):
            self.assertIn(f"■ Pub{i}", headers)


# ── (e) briefing anti-monopoly cap keys on the publisher ───────────────

class BriefingCapTests(unittest.TestCase):

    def _seed_two_routes(self, ctx):
        # Reuters via its own feed: 4 distinct stories
        items = "".join(
            f"<item><title>Reuters story {i}</title>"
            f"<link>https://news.google.com/rss/articles/R{i}?oc=5</link>"
            f"<description>Body {i}.</description>"
            f"<pubDate>Wed, 21 Oct 2026 1{i}:00:00 GMT</pubDate>"
            f'<source url="https://www.reuters.com">Reuters</source></item>'
            for i in range(4))
        reuters_xml = ('<?xml version="1.0"?><rss version="2.0"><channel>'
                       + items + "</channel></rss>")
        agent = NewsAgent(ctx)
        _run_with_xml(agent, [(REUTERS_FEED, reuters_xml)])
        # …and one more Reuters story via the generic feed (fresh run:
        # distinct title so title-dedupe does not eat it)
        gn_xml = (
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            "<item><title>Reuters extra wire story</title>"
            "<link>https://news.google.com/rss/articles/GN1?oc=5</link>"
            "<description>Extra.</description>"
            "<pubDate>Wed, 21 Oct 2026 12:00:00 GMT</pubDate>"
            '<source url="https://www.reuters.com">Reuters</source></item>'
            "<item><title>BBC story one</title>"
            "<link>https://news.google.com/rss/articles/GN2?oc=5</link>"
            "<description>B1.</description>"
            "<pubDate>Wed, 21 Oct 2026 12:01:00 GMT</pubDate>"
            '<source url="https://www.bbc.co.uk">BBC News</source></item>'
            "<item><title>BBC story two</title>"
            "<link>https://news.google.com/rss/articles/GN3?oc=5</link>"
            "<description>B2.</description>"
            "<pubDate>Wed, 21 Oct 2026 12:02:00 GMT</pubDate>"
            '<source url="https://www.bbc.co.uk">BBC News</source></item>'
            "<item><title>BBC story three</title>"
            "<link>https://news.google.com/rss/articles/GN4?oc=5</link>"
            "<description>B3.</description>"
            "<pubDate>Wed, 21 Oct 2026 12:03:00 GMT</pubDate>"
            '<source url="https://www.bbc.co.uk">BBC News</source></item>'
            "</channel></rss>")
        _run_with_xml(agent, [(GN_FEED, gn_xml)])

    def test_cap_counts_publisher_once_across_routes(self):
        from nomorals.agents.morning_briefing import NewsProvider
        ctx = _make_ctx(self, cap=40)
        self._seed_two_routes(ctx)
        section = NewsProvider().collect(ctx, since=0.0)
        self.assertIsNotNone(section)
        sources = [i["source"] for i in section.items]
        # 5 Reuters stories stored; the publisher cap (3) applies ONCE
        # across both carrying feeds — not 3 per feed (which would be 4)
        self.assertEqual(sources.count("Reuters"), 3)
        # BBC carried by the same aggregator feed gets its OWN cap slot:
        # legacy feed-name grouping would have capped ALL Google-News-
        # carried items (1 Reuters + 3 BBC) at 3 combined.
        self.assertEqual(sources.count("BBC News"), 3)


if __name__ == "__main__":
    unittest.main()
