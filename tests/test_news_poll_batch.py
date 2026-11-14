"""News poll batching (2026-11-14 incident): the `news poll` job timed
out at its 300s wall on EC2. Two measured sinks:

(a) The dedupe asked the DB once per fetched item
    (``query_one("SELECT id FROM news_items WHERE url = ?", ...)`` per
    item — 102 feeds x ~10-20 items each = ~1000-2000 round trips
    through ``execute`` -> ``_with_lock_retry``). The batched form is
    ONE ``SELECT url ... WHERE url IN (...)`` per 500 URLs (SQLite)
    or ONE ``find({"url": {"$in": [...]}})`` (rowstore).
(b) Each feed fetch carried the shared ``http_timeout`` (30-60s), so
    a few slow feeds ate half the wall in SSL time. The poll now caps
    each feed at NEWS_FEED_TIMEOUT_S (10s); RSS answers in seconds
    or not at all, and the existing SourceHealth stand-down records
    the slow feed and re-activates it on recovery.

Pins: the batched check issues one query for many URLs and returns
the same known/unknown split as the old per-item loop; the news
client carries the news-specific timeout, not the shared one.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import nomorals.agents.news as nw
from nomorals.agents.news import NewsAgent
from nomorals.storage.db import Database

#: The news-specific per-feed cap; None until the fix defines it.
NEWS_FEED_TIMEOUT_S = getattr(nw, "NEWS_FEED_TIMEOUT_S", None)

FEED = ("Test Feed", "https://feed.example.com/rss")


def _xml(items):
    """items: list of (title, url, summary)."""
    body = "".join(
        f"<item><title>{t}</title><link>{u}</link>"
        f"<description>{s}</description>"
        f"<pubDate>Wed, 21 Oct 2026 10:00:00 GMT</pubDate></item>"
        for t, u, s in items)
    return ('<?xml version="1.0"?><rss version="2.0"><channel>'
            + body + "</channel></rss>")


def _items(n, prefix="Story"):
    return [(f"{prefix} {i}", f"https://feed.example.com/{prefix}{i}",
             f"Body of {prefix} {i}.") for i in range(n)]


def _make_ctx(test, *, http_timeout=None):
    tmp = tempfile.mkdtemp(prefix="newspollbatch-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    news_ns = SimpleNamespace(feeds="")
    tools_ns = SimpleNamespace()
    if http_timeout is not None:
        tools_ns.http_timeout = http_timeout
    settings = SimpleNamespace(news=news_ns, tools=tools_ns)
    return SimpleNamespace(db=db, settings=settings, extras={})


def _poll(ctx, xml):
    agent = NewsAgent(ctx)
    with mock.patch.object(agent, "_feeds", return_value=[FEED]), \
            mock.patch.object(agent, "_client") as client:
        client.get.return_value = SimpleNamespace(text=xml)
        return agent, agent.poll()


class _CountingDb:
    """Wraps a Database, counting the URL-lookup statements."""

    def __init__(self, db):
        self.db = db
        self.url_query_one = 0
        self.url_in_queries = 0

    def __getattr__(self, name):
        return getattr(self.db, name)

    def query_one(self, sql, params=()):
        if "WHERE url = ?" in sql:
            self.url_query_one += 1
        return self.db.query_one(sql, params)

    def query(self, sql, params=()):
        if "WHERE url IN (" in sql:
            self.url_in_queries += 1
        return self.db.query(sql, params)


class BatchedUrlCheckTests(unittest.TestCase):

    def test_url_known_check_is_one_batched_query(self):
        ctx = _make_ctx(self)
        # First poll stores 40 stories.
        _poll(ctx, _xml(_items(40)))
        # Second poll over the same 40 + 10 new ones: the dedupe must
        # ask the DB ONCE (batched IN), never once per item.
        counting = _CountingDb(ctx.db)
        ctx.db = counting
        agent = NewsAgent(ctx)
        with mock.patch.object(agent, "_feeds", return_value=[FEED]), \
                mock.patch.object(agent, "_client") as client:
            client.get.return_value = SimpleNamespace(
                text=_xml(_items(40) + _items(10, prefix="Fresh")))
            report = agent.poll()
        self.assertEqual(report["polled"], 10)
        self.assertEqual(counting.url_query_one, 0,
                         "per-item URL checks must be gone")
        self.assertEqual(counting.url_in_queries, 1,
                         "one batched IN query for 50 URLs")

    def test_batched_check_matches_per_item_split(self):
        ctx = _make_ctx(self)
        _, report1 = _poll(ctx, _xml(_items(30)))
        self.assertEqual(report1["polled"], 30)
        # A second poll over the identical feed stores nothing.
        _, report2 = _poll(ctx, _xml(_items(30)))
        self.assertEqual(report2["polled"], 0)
        # A mixed feed (20 known + 5 new) stores exactly the new ones.
        _, report3 = _poll(
            ctx, _xml(_items(20) + _items(5, prefix="Brand")))
        self.assertEqual(report3["polled"], 5)
        rows = ctx.db.query("SELECT url FROM news_items ORDER BY url")
        self.assertEqual(len(rows), 35)

    def test_sqlite_path_chunks_large_url_sets(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        urls = [f"https://feed.example.com/item{i}" for i in range(600)]
        counting = _CountingDb(ctx.db)
        ctx.db = counting
        agent.db = counting
        known = agent._known_urls(urls)
        self.assertEqual(known, set())
        # 600 URLs at 500 per chunk = exactly 2 batched queries.
        self.assertEqual(counting.url_in_queries, 2)
        self.assertEqual(counting.url_query_one, 0)

    def test_rowstore_path_batches_into_one_find(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        seen_filters = []

        class _FakeItems:
            def find(self, filt, **kw):
                seen_filters.append(filt)
                return [{"url": "https://feed.example.com/known1"},
                        {"url": "https://feed.example.com/known2"}]

        with mock.patch.object(agent, "_items",
                               return_value=_FakeItems()):
            known = agent._known_urls(
                ["https://feed.example.com/known1",
                 "https://feed.example.com/known2",
                 "https://feed.example.com/new1"])
        self.assertEqual(known, {"https://feed.example.com/known1",
                                 "https://feed.example.com/known2"})
        self.assertEqual(len(seen_filters), 1)
        self.assertIn("$in", seen_filters[0]["url"])
        self.assertEqual(
            sorted(seen_filters[0]["url"]["$in"]),
            ["https://feed.example.com/known1",
             "https://feed.example.com/known2",
             "https://feed.example.com/new1"])

    def test_rowstore_path_chunks_large_url_sets(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        seen_filters = []

        class _FakeItems:
            def find(self, filt, **kw):
                seen_filters.append(filt)
                return []

        urls = [f"https://feed.example.com/item{i}" for i in range(600)]
        with mock.patch.object(agent, "_items",
                               return_value=_FakeItems()):
            agent._known_urls(urls)
        self.assertEqual(len(seen_filters), 2)
        self.assertTrue(all(len(f["url"]["$in"]) <= 500
                            for f in seen_filters))


class NewsFeedTimeoutTests(unittest.TestCase):

    def test_news_feed_timeout_constant_is_ten_seconds(self):
        self.assertEqual(NEWS_FEED_TIMEOUT_S, 10.0)

    def test_client_carries_the_news_timeout_not_the_shared_one(self):
        # Even when the shared http_timeout is 60s, the news client
        # caps each feed at the news-specific 10s.
        ctx = _make_ctx(self, http_timeout=60.0)
        agent = NewsAgent(ctx)
        self.assertEqual(agent._client.timeout, NEWS_FEED_TIMEOUT_S)
        self.assertEqual(agent._client.timeout, 10.0)

    def test_client_timeout_is_ten_when_no_shared_timeout_is_set(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        self.assertEqual(agent._client.timeout, 10.0)


if __name__ == "__main__":
    unittest.main()
