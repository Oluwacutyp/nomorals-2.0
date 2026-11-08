"""Formatting F6: full briefings — the 12-cap stack is gone.

Owner complaint (2026-10-22): the briefings deliver "not just 12, but
full new ones".  The survey found the 12 was a STACK: the morning
radar re-read recent(limit=12); the composer word budget squeezed the
news section; afternoon/evening read recent(limit=15) through a
parallel seen-marker that drifted from the news delivery ledger.

These tests pin the F6 contract:
- every slot sources from NewsAgent's delivery ledger (undelivered),
  uncapped — 40 stories in, 40 stories out;
- a story is marked delivered ONLY when its publish confirms — a
  failed publish's stories stay undelivered and arrive whole next
  pulse;
- morning-delivered stories never reappear in the afternoon;
- the retired seen-marker is folded into the ledger ONCE at cutover
  (no re-flood) and never consulted again;
- a story with a summary renders it in an expandable blockquote;
- the ledger is the single truth: no pulse_seen.json is written.

PAGER UPDATE (novel pass, 2026-11-07 — the owner's live-display
law: nothing splits into (1/2), (2/2) dumps anymore): the story
list now arrives as ONE paged live message — page 1 with Next/Prev
buttons, no "(i/n)" labels anywhere, and every story reachable
exactly once across the pages (morning_pulse.story_page_render).
The confirmed-marking contract stands in the paged shape: the one
publish confirms → all of the list's URLs mark; it fails → none do.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import morning_pulse as mp
from nomorals.agents.news import NewsAgent
from nomorals.core.ids import new_id
from nomorals.storage.db import Database


class _SendResult:
    def __init__(self, ok=True, message_id="m1", error=""):
        self.ok = ok
        self.message_id = message_id
        self.error = error


class FakeGateway:
    """Records every send; can be told to fail specific send calls."""

    def __init__(self, fail_on=()):
        self.sent = []  # (platform, chat_id, text)
        self.fail_on = set(fail_on)  # 1-based send-call numbers
        self.calls = 0

    def status(self):
        return {"telegram-bot": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, **kwargs):
        self.calls += 1
        if self.calls in self.fail_on:
            return _SendResult(ok=False, error="boom (fake)")
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)

    def platforms_for_chat_id(self, chat_id):
        return []


def make_ctx(test, gateway=None):
    tmp = tempfile.mkdtemp(prefix="f6-briefings-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats="telegram:111", proactive_enabled=True,
        proactive_briefing=True, proactive_watchers=True,
        quiet_start=22, quiet_end=8, timezone="UTC")
    settings = SimpleNamespace(workspace_dir=tmp, partner=partner,
                               pulse=None)
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    if gateway is not None:
        ctx.gateway = gateway
    return ctx, tmp


def seed_stories(ctx, n, *, summary_words=0, start=0):
    """Insert n undelivered stories; returns their urls in seed order."""
    NewsAgent(ctx)  # ensures the attribution/delivery columns exist
    urls = []
    for i in range(start, start + n):
        # trailing path segment: no URL is a substring of another
        url = f"https://news.test/s{i}/full"
        summary = ""
        if summary_words:
            summary = " ".join(
                f"Detail{i} sentence number {j} carries real substance."
                for j in range(summary_words))
        ctx.db.execute(
            "INSERT INTO news_items (id, source, publisher, title, url, "
            "summary, published, created_at, delivery_state) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), "testfeed", "TestWire", f"Story {i} headline",
             url, summary, 1000.0 + i, time.time(), "undelivered"))
        urls.append(url)
    return urls


def sent_text(gateway):
    return "\n".join(t for _, _, t in gateway.sent)


def pager_texts(ctx):
    """Every page of the latest delivered story list (the pager)."""
    out, page = [], 1
    while True:
        render = mp.story_page_render(ctx, "latest", page)
        if render is None:
            break
        out.append(render["text"])
        if page >= render["pages"]:
            break
        page += 1
    return out


def undelivered_urls(ctx):
    return {str(r.get("url")) for r in NewsAgent(ctx).undelivered()}


class FullDeliveryTests(unittest.TestCase):
    """40 undelivered stories ALL deliver in one pulse, uncapped."""

    def test_afternoon_delivers_all_40(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(self, gw)
        urls = seed_stories(ctx, 40, summary_words=2)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r = mp.run_slot(ctx, "afternoon")
        self.assertTrue(r["ok"])
        self.assertEqual(r["stories_delivered"], 40)
        self.assertEqual(r["stories_pending"], 0)
        # ONE paged story-list message went out — not a (i/n) dump
        text = sent_text(gw)
        self.assertIn(mp.STORY_LIST_TITLE, text)
        self.assertNotIn("(1/", text)
        self.assertIn("— page 1 of", text)
        story_sends = [t for _, _, t in gw.sent
                       if mp.STORY_LIST_TITLE in t]
        self.assertEqual(len(story_sends), 1)
        # every story's FULL block (headline + URL) is reachable
        # exactly once across the pager's pages
        pages = pager_texts(ctx)
        self.assertGreater(len(pages), 1)
        for i in range(40):
            hits = [p for p in pages if f"Story {i} headline" in p]
            self.assertEqual(len(hits), 1,
                             f"story {i} not paged exactly once")
            self.assertIn(f"https://news.test/s{i}/full", hits[0])
        # everything marked in the ledger; nothing left undelivered
        self.assertEqual(undelivered_urls(ctx), set())

    def test_morning_pulse_delivers_all_40(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(self, gw)
        seed_stories(ctx, 40)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_compose_briefing",
                          return_value="composed briefing"), \
             patch.object(mp, "_synthesize", return_value=""):
            r = mp.run_pulse(ctx)
        self.assertEqual(r["stories_delivered"], 40)
        # no summaries → the whole list packs into ONE page, so it
        # publishes as a single message exactly as it always did
        # (no stash, no buttons) — every headline in the sent text,
        # and still no (i/n) labelling anywhere
        text = sent_text(gw)
        self.assertNotIn("(1/", text)
        for i in range(40):
            self.assertIn(f"Story {i} headline", text)
        self.assertEqual(undelivered_urls(ctx), set())

    def test_story_block_format(self):
        ctx, _ = make_ctx(self)
        seed_stories(ctx, 1, summary_words=3)
        item = NewsAgent(ctx).undelivered()[0]
        block = mp._story_block(item)
        self.assertIn("Story 0 headline", block)
        self.assertIn("TestWire", block)          # publisher named
        self.assertIn("https://news.test/s0/full", block)  # link present
        self.assertIn("<blockquote expandable>", block)
        self.assertIn("Detail0 sentence number 0", block)
        self.assertIn("</blockquote>", block)

    def test_summary_escaped_in_blockquote(self):
        ctx, _ = make_ctx(self)
        NewsAgent(ctx)  # ensures columns
        ctx.db.execute(
            "INSERT INTO news_items (id, source, publisher, title, url, "
            "summary, published, created_at, delivery_state) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), "testfeed", "TestWire", "Angle story",
             "https://news.test/angle", "a <b>bold</b> & raw claim",
             1000.0, time.time(), "undelivered"))
        item = NewsAgent(ctx).undelivered()[0]
        block = mp._story_block(item)
        self.assertIn("a &lt;b&gt;bold&lt;/b&gt; &amp; raw claim", block)


class CrossSlotDedupeTests(unittest.TestCase):
    """The ledger dedupes across the day's slots — no repeats, ever."""

    def test_morning_stories_absent_from_afternoon(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(self, gw)
        seed_stories(ctx, 6)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_compose_briefing",
                          return_value="composed briefing"), \
             patch.object(mp, "_synthesize", return_value=""):
            mp.run_pulse(ctx)
        morning_text = sent_text(gw)
        for i in range(6):
            self.assertIn(f"Story {i} headline", morning_text)
        # afternoon: the morning's stories must NOT reappear
        gw.sent.clear()
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r = mp.run_slot(ctx, "afternoon")
        self.assertEqual(r["stories_delivered"], 0)
        afternoon_text = sent_text(gw)
        for i in range(6):
            self.assertNotIn(f"Story {i} headline", afternoon_text)
        # …but stories arriving after the morning DO deliver
        seed_stories(ctx, 2, start=100)
        gw.sent.clear()
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r = mp.run_slot(ctx, "afternoon")
        self.assertEqual(r["stories_delivered"], 2)
        self.assertIn("Story 100 headline", sent_text(gw))

    def test_evening_and_midnight_share_the_ledger(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(self, gw)
        seed_stories(ctx, 3)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}):
            r = mp.run_slot(ctx, "evening")
        self.assertEqual(r["stories_delivered"], 3)
        gw.sent.clear()
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}):
            r = mp.run_slot(ctx, "midnight")
        self.assertEqual(r["stories_delivered"], 0)
        self.assertNotIn("Story 0 headline", sent_text(gw))


class FailedPublishTests(unittest.TestCase):
    """Confirmed marking in the paged shape: the list is ONE publish,
    so marking is per-list — a failed publish loses nothing (the
    whole list stays pending and arrives next pulse) and a confirmed
    one is never repeated."""

    def test_failed_publish_stays_undelivered_and_resumes_whole(self):
        # every gateway call fails on the first run — head text AND
        # the story-list publish
        gw = FakeGateway(fail_on=set(range(1, 13)))
        ctx, _ = make_ctx(self, gw)
        urls = seed_stories(ctx, 24, summary_words=6)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r = mp.run_slot(ctx, "afternoon")
        self.assertEqual(r["stories_delivered"], 0)
        self.assertEqual(r["stories_pending"], 24)
        self.assertEqual(undelivered_urls(ctx), set(urls))
        # next pulse with a healthy gateway: the whole list arrives
        # through the pager, every story exactly once
        gw2 = FakeGateway()
        ctx.gateway = gw2
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r2 = mp.run_slot(ctx, "afternoon")
        self.assertEqual(r2["stories_delivered"], 24)
        pages = pager_texts(ctx)
        for u in urls:
            hits = [p for p in pages if u in p]
            self.assertEqual(len(hits), 1,
                             "story missing or repeated after resume")
        self.assertEqual(undelivered_urls(ctx), set())

    def test_total_delivery_failure_marks_nothing(self):
        gw = FakeGateway(fail_on={1, 2, 3, 4, 5, 6, 7, 8})
        ctx, _ = make_ctx(self, gw)
        seed_stories(ctx, 5)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r = mp.run_slot(ctx, "afternoon")
        self.assertEqual(r["stories_delivered"], 0)
        self.assertEqual(len(undelivered_urls(ctx)), 5)


class CutoverMigrationTests(unittest.TestCase):
    """The retired seen-marker folds into the ledger once — no
    re-flood — and is never consulted again."""

    def test_cutover_marks_seen_and_retires_marker(self):
        gw = FakeGateway()
        ctx, tmp = make_ctx(self, gw)
        urls = seed_stories(ctx, 3)
        # a title-keyed story (the old marker also keyed by title)
        ctx.db.execute(
            "INSERT INTO news_items (id, source, publisher, title, url, "
            "summary, published, created_at, delivery_state) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), "testfeed", "TestWire", "Title keyed tale",
             "https://news.test/title-keyed", "", 999.0, time.time(),
             "undelivered"))
        marker = {"2026-10-21": {urls[0]: "morning",
                                 "Title keyed tale": "evening"},
                  "2026-10-22": {}}
        p = mp._seen_path(ctx)
        p.write_text(json.dumps(marker), encoding="utf-8")
        marked = mp._migrate_seen_store(ctx)
        self.assertEqual(marked, 2)
        # marker retired, not deleted-blind and never re-read
        self.assertFalse(p.is_file())
        self.assertTrue(p.with_name(p.name + ".retired").is_file())
        left = undelivered_urls(ctx)
        self.assertNotIn(urls[0], left)
        self.assertNotIn("https://news.test/title-keyed", left)
        self.assertEqual(len(left), 2)

    def test_marker_not_consulted_after_cutover(self):
        gw = FakeGateway()
        ctx, tmp = make_ctx(self, gw)
        urls = seed_stories(ctx, 1)
        p = mp._seen_path(ctx)
        p.write_text(json.dumps({"2026-10-22": {urls[0]: "morning"}}),
                     encoding="utf-8")
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}):
            items = mp._fresh_news_items(ctx)
        self.assertEqual(items, [])  # cutover folded it into the ledger
        # a NEW marker file appears (stale writer, restored backup…):
        # the pipeline must ignore it — the ledger is the only truth.
        new_urls = seed_stories(ctx, 1, start=50)
        p.write_text(json.dumps(
            {"2026-10-22": {new_urls[0]: "morning"}}),
            encoding="utf-8")
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}):
            items = mp._fresh_news_items(ctx)
        self.assertEqual([str(i.get("url")) for i in items], new_urls)

    def test_no_marker_file_written_by_slots(self):
        gw = FakeGateway()
        ctx, tmp = make_ctx(self, gw)
        seed_stories(ctx, 2)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            mp.run_slot(ctx, "afternoon")
        self.assertFalse(os.path.exists(os.path.join(tmp, "pulse_seen.json")))


class PackingTests(unittest.TestCase):
    """Segment packing: order kept, nothing dropped, overflow merges."""

    def test_packing_keeps_every_story_once(self):
        ctx, _ = make_ctx(self)
        urls = seed_stories(ctx, 50, summary_words=6)
        items = NewsAgent(ctx).undelivered()
        segments = mp._pack_story_segments(items)
        all_keys = [k for _, ks in segments for k in ks]
        self.assertEqual(len(all_keys), 50)
        self.assertEqual(len(set(all_keys)), 50)
        for text, keys in segments:
            for k in keys:
                i = urls.index(k)
                self.assertIn(f"Story {i} headline", text)

    def test_overflow_merges_tail_into_one_segment(self):
        ctx, _ = make_ctx(self)
        seed_stories(ctx, 8)
        items = NewsAgent(ctx).undelivered()
        # tiny budget → many raw segments → merged at max_inline
        segments = mp._pack_story_segments(items, budget=60, max_inline=3)
        self.assertEqual(len(segments), 3)
        all_keys = [k for _, ks in segments for k in ks]
        self.assertEqual(len(all_keys), 8)


class NewsProviderLedgerTests(unittest.TestCase):
    """The composer's highlights come from the ledger too."""

    def test_worth_knowing_skips_delivered(self):
        from nomorals.agents.morning_briefing import NewsProvider
        ctx, _ = make_ctx(self)
        urls = seed_stories(ctx, 4)
        NewsAgent(ctx).mark_delivered(urls[:2], channel="pulse-morning")
        sec = NewsProvider().collect(ctx, 0.0)
        self.assertIsNotNone(sec)
        titles = [i["title"] for i in sec.items]
        self.assertIn("Story 2 headline", titles)
        self.assertIn("Story 3 headline", titles)
        self.assertNotIn("Story 0 headline", titles)
        self.assertNotIn("Story 1 headline", titles)


if __name__ == "__main__":
    unittest.main()
