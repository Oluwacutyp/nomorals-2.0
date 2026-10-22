"""PASS #80: Reuters is a source — live wire feed, truthful attribution.

Owner directive 2026-10-22: "add the reuter to sources too".

Pins:
(a) ("Reuters", site-scoped Google News query) is in feeds_for("wires")
    and in the DEFAULT flattening (DEFAULT_CATEGORIES includes wires).
(b) parse_feed reads the RSS <source> element: an aggregator-carried
    item is attributed to its publisher (reuters.com — trust tier 0.80),
    NOT to the redirect host (news.google.com).  Feeds without <source>
    parse exactly as before.
(c) The same story arriving via the generic wires feed and via the
    Reuters feed — different redirect URLs, same title + publisher —
    is delivered ONCE, including across runs (the news_items store is
    the organ's memory; its documented contract is dedupe by URL +
    normalized title).
(d) settings.news.feeds override semantics unchanged: custom feeds
    replace the defaults wholesale.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlparse

from nomorals.agents.news import (
    DEFAULT_FEEDS, NewsAgent, feeds_for, parse_feed,
)
from nomorals.core.trust import domain_tier
from nomorals.storage.db import Database

REUTERS_FEED_URL = ("https://news.google.com/rss/search?"
                    "q=site:reuters.com&hl=en-US&gl=US&ceid=US:en")

# Google-News-shaped RSS: item links are news.google.com redirects; the
# publisher lives ONLY in the item's <source> element.
GENERIC_WIRES_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
    "<title>Top stories - Google News</title>"
    "<item><title>Fed holds rates steady - Reuters</title>"
    "<link>https://news.google.com/rss/articles/CBMiAAA111?oc=5</link>"
    "<description>Central bank keeps rates unchanged.</description>"
    "<pubDate>Wed, 21 Oct 2026 10:00:00 GMT</pubDate>"
    '<source url="https://www.reuters.com">Reuters</source></item>'
    "</channel></rss>")

REUTERS_FEED_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
    "<title>Reuters - Google News</title>"
    "<item><title>Fed holds rates steady - Reuters</title>"
    "<link>https://news.google.com/rss/articles/CBMiBBB222?oc=5</link>"
    "<description>Central bank keeps rates unchanged.</description>"
    "<pubDate>Wed, 21 Oct 2026 10:05:00 GMT</pubDate>"
    '<source url="https://www.reuters.com">Reuters</source></item>'
    "</channel></rss>")

NO_SOURCE_XML = (
    '<?xml version="1.0"?><rss version="2.0"><channel>'
    "<item><title>Plain story</title>"
    "<link>https://example.com/plain</link>"
    "<description>Body.</description>"
    "<pubDate>Wed, 21 Oct 2026 09:00:00 GMT</pubDate></item>"
    "</channel></rss>")


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _make_ctx(test, feeds_setting=""):
    tmp = tempfile.mkdtemp(prefix="pass80-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        news=SimpleNamespace(feeds=feeds_setting, cap=10))
    return SimpleNamespace(db=db, settings=settings, extras={})


def _run_with_xml(agent, feed_name, feed_url, xml):
    with mock.patch.object(agent, "_feeds",
                           return_value=[(feed_name, feed_url)]), \
         mock.patch.object(agent, "_client") as client:
        client.get.return_value = SimpleNamespace(text=xml)
        return agent.run()


# ── (a) Reuters is a wires source ─────────────────────────────────────

class ReutersListedTests(unittest.TestCase):
    def test_reuters_in_wires_category(self):
        wires = dict(feeds_for("wires"))
        self.assertIn("Reuters", wires)
        self.assertEqual(wires["Reuters"], REUTERS_FEED_URL)

    def test_reuters_in_default_flattening(self):
        self.assertIn(("Reuters", REUTERS_FEED_URL), list(DEFAULT_FEEDS))


# ── (b) truthful attribution via <source> ──────────────────────────────

class AttributionTests(unittest.TestCase):
    def test_parser_reads_source_element(self):
        items = parse_feed(REUTERS_FEED_XML)
        self.assertEqual(len(items), 1)
        item = items[0]
        # The article link stays the aggregator redirect (Pass 2's
        # unwrap resolves it at open time)...
        self.assertEqual(_domain(item["url"]), "news.google.com")
        # ...but attribution is the PUBLISHER, from <source>.
        self.assertEqual(item["source_url"], "https://www.reuters.com")
        self.assertEqual(item["source_name"], "Reuters")
        self.assertEqual(_domain(item["source_url"]), "reuters.com")

    def test_publisher_scores_as_reuters_not_google(self):
        item = parse_feed(REUTERS_FEED_XML)[0]
        tier, note = domain_tier(item["source_url"])
        self.assertEqual((tier, note), (0.80, "established press"))
        # The redirect link alone would NOT earn the Reuters tier.
        self.assertNotEqual(domain_tier(item["url"])[0], 0.80)

    def test_feed_without_source_parses_as_before(self):
        items = parse_feed(NO_SOURCE_XML)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["title"], "Plain story")
        self.assertEqual(item["url"], "https://example.com/plain")
        self.assertEqual(item["summary"], "Body.")
        self.assertEqual(item["source_url"], "")
        self.assertEqual(item["source_name"], "")


# ── (c) one story, two routes, one delivery ───────────────────────────

class CrossRouteDedupeTests(unittest.TestCase):
    def test_same_run_both_routes_delivers_once(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        with mock.patch.object(agent, "_feeds", return_value=[
                ("Google News", "https://example.test/gn.xml"),
                ("Reuters", "https://example.test/reuters.xml")]), \
             mock.patch.object(agent, "_client") as client:
            def _get(url, *args, **kwargs):
                return SimpleNamespace(
                    text=GENERIC_WIRES_XML if "gn.xml" in url
                    else REUTERS_FEED_XML)
            client.get.side_effect = _get
            report = agent.run()
        self.assertEqual(report["items"], 1, report)

    def test_second_run_other_route_delivers_nothing_new(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        first = _run_with_xml(agent, "Google News",
                              "https://example.test/gn.xml",
                              GENERIC_WIRES_XML)
        self.assertEqual(first["items"], 1, first)
        # Same story, hours later, via the Reuters feed — a DIFFERENT
        # redirect URL.  URL-only memory would deliver it AGAIN.
        second = _run_with_xml(agent, "Reuters",
                               "https://example.test/reuters.xml",
                               REUTERS_FEED_XML)
        self.assertEqual(second["fresh"], 0, second)
        self.assertEqual(second["items"], 0, second)
        rows = ctx.db.query("SELECT title, url FROM news_items")
        self.assertEqual(len(rows), 1, rows)


# ── (d) settings override unchanged ───────────────────────────────────

class SettingsOverrideTests(unittest.TestCase):
    def test_custom_feeds_replace_defaults(self):
        ctx = _make_ctx(self, feeds_setting="https://example.com/custom/feed.xml")
        agent = NewsAgent(ctx)
        feeds = agent._feeds()
        self.assertEqual(feeds, [("custom",
                                  "https://example.com/custom/feed.xml")])
        self.assertNotIn("Reuters", dict(feeds))

    def test_empty_setting_uses_defaults_including_reuters(self):
        ctx = _make_ctx(self, feeds_setting="")
        agent = NewsAgent(ctx)
        self.assertIn(("Reuters", REUTERS_FEED_URL), agent._feeds())


if __name__ == "__main__":
    unittest.main()
