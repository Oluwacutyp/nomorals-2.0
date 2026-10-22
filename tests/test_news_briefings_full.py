"""News briefings: full delivery + four-a-day cadence.

Owner complaints (screenshots + chat 2026-10-19):
1. Briefings ALWAYS get cut off — even a briefing must deliver the FULL
   gist, not a truncated fragment. A long briefing paginates (the gateway
   already chunks losslessly); it must never shrink to fit by cutting
   content at delivery time.
2. News must not be morning-only — briefings run morning, afternoon,
   evening, AND midnight. Each covers what's new SINCE the last one
   (no repeating the morning stories at night).

Failing-first: these tests fail on the pre-fix tree (hard [:3500]/[:3000]
cuts, truncate(3900), 3 slots, no midnight, morning never marks seen).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import morning_pulse as mp
from nomorals.agents.notifier import Notifier
from nomorals.storage.db import Database


# ── shared fakes (same shape as test_delivery_fallback) ──────────────

class _SendResult:
    def __init__(self, ok=True, message_id="", error=""):
        self.ok = ok
        self.message_id = message_id
        self.error = error


class FakeGateway:
    def __init__(self, running=("telegram-bot",)):
        self.running = set(running)
        self.sent = []  # (platform, chat_id, text)

    def status(self):
        return {p: {"running_in_session": True} for p in self.running}

    def send(self, platform, chat_ref, text, **kwargs):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True, message_id="mid-1")

    def platforms_for_chat_id(self, chat_id):
        return []


def make_partner(**kw):
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=22,
        quiet_end=8,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_ctx(test=None, partner=None):
    tmp = tempfile.mkdtemp(prefix="news-briefings-test-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner or make_partner())
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    return ctx, tmp


def _long_briefing(n=30):
    """A briefing with n complete-sentence items (>3500 chars total)."""
    lines = ["📰 Full briefing — every story, nothing cut.\n"]
    for i in range(n):
        lines.append(
            f"• Story {i}: the council approved the new market road "
            f"after a long debate about funding and timelines.")
    return "\n".join(lines)


class DeliveryCompletenessTests(unittest.TestCase):
    """(1) No truncation: the full briefing reaches the owner."""

    def test_notifier_delivers_long_body_whole(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(self)
        body = _long_briefing(60)
        self.assertGreater(len(body), 3900, "test body must exceed old caps")
        res = Notifier(ctx, gateway=gw).publish(
            "pulse", "🌙 Morning pulse", body, force=True)
        self.assertTrue(res.get("delivered"))
        delivered_text = "".join(t for _, _, t in gw.sent)
        for i in range(60):
            self.assertIn(f"Story {i}:", delivered_text,
                          f"item {i} dropped in delivery")
        # no mid-sentence cut: the last story arrives complete
        self.assertIn("about funding and timelines.", delivered_text)
        # no truncation ellipsis from the old truncate(3900) path
        self.assertNotIn("…", delivered_text)

    def test_news_digest_not_cut(self):
        from nomorals.agents.news import NewsAgent

        ctx, _ = make_ctx(self)
        agent = NewsAgent(ctx, notifier=None)
        # fake feed: 30 items, each with a long summary
        items_xml = "".join(
            f"<item><title>Headline {i} about markets</title>"
            f"<link>https://example.test/n{i}</link>"
            f"<description>Full summary {i} with several sentences. "
            f"Second sentence carries the actual detail here. "
            f"Third sentence wraps up the story completely.</description>"
            f"</item>"
            for i in range(30))

        class FakeResp:
            text = ("<?xml version='1.0'?><rss><channel>" + items_xml +
                    "</channel></rss>")

        class FakeClient:
            def get(self, url, headers=None):
                return FakeResp()

        agent._client = FakeClient()
        agent._feeds = lambda: [("testfeed", "https://example.test/rss")]
        report = agent.run(cap=40)  # the documented max: the full version
        self.assertTrue(report["ok"])
        digest = report["digest"]
        for i in range(30):
            self.assertIn(f"Headline {i} about markets", digest,
                          f"digest dropped item {i}")
        # every item's gist arrives as complete sentences, not a cut
        self.assertIn("Third sentence wraps up the story completely.",
                      digest)


class FourSlotsTests(unittest.TestCase):
    """(2) Four briefings a day: morning / afternoon / evening / midnight."""

    def test_four_slots_defined(self):
        self.assertEqual(set(mp.BRIEFING_SLOTS),
                         {"morning", "afternoon", "evening", "midnight"})
        self.assertEqual(mp.BRIEFING_SLOTS["midnight"]["default_time"], "00:00")
        self.assertEqual(mp.BRIEFING_SLOTS["midnight"]["job_name"],
                         "midnight-pulse")
        for slot, spec in mp.BRIEFING_SLOTS.items():
            self.assertTrue(spec["job_name"])
            self.assertTrue(spec["default_time"])
            self.assertTrue(spec["title"])

    def test_midnight_slot_runs(self):
        ctx, _ = make_ctx(self)
        # workspace-dir settings so markers land in tmp, not cwd
        ctx.settings = SimpleNamespace(workspace_dir=_tmpdir(ctx),
                                        partner=make_partner(), pulse=None)
        r = mp.run_slot(ctx, "midnight")
        self.assertTrue(r["ok"])
        self.assertEqual(r["slot"], "midnight")
        self.assertIn("Midnight", r["text"])

    def test_midnight_dedupes_across_day_boundary(self):
        """Midnight (00:00, a NEW day) must not repeat evening's stories.

        F6: dedupe is the news delivery ledger, which is not day-scoped
        at all — a story marked delivered by the evening slot (even
        "yesterday") simply isn't in the undelivered set anymore.
        (The retired seen-store pruned to today+yesterday and needed
        this test's boundary dance; the ledger makes it structural.)
        """
        from nomorals.agents.news import NewsAgent
        from nomorals.core.ids import new_id
        ctx, tmp = make_ctx(self)
        ctx.settings = SimpleNamespace(workspace_dir=tmp,
                                        partner=make_partner(), pulse=None)
        agent = NewsAgent(ctx)  # ensures delivery columns
        for url, title in (("https://x/evening-story", "Evening story"),
                           ("https://x/fresh", "Fresh story")):
            ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at, delivery_state) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (new_id(), "src", title, url, "", 0, time.time(),
                 "undelivered"))
        agent.mark_delivered(["https://x/evening-story"],
                             channel="pulse-evening")
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}):
            text, items = mp._compose_midnight(ctx)
        self.assertEqual([i["url"] for i in items], ["https://x/fresh"])
        self.assertIn("Fresh story", text)
        self.assertNotIn("Evening story", text)

    def test_second_briefing_excludes_first_stories(self):
        """Afternoon #2 must not repeat afternoon #1's headlines."""
        from nomorals.agents.news import NewsAgent
        from nomorals.core.ids import new_id
        ctx, tmp = make_ctx(self)
        ctx.settings = SimpleNamespace(workspace_dir=tmp,
                                        partner=make_partner(), pulse=None)
        gw = FakeGateway()
        ctx.gateway = gw
        agent = NewsAgent(ctx)  # ensures delivery columns
        for i in range(4):
            ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at, delivery_state) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (new_id(), "src", f"Story {i}", f"https://x/s{i}", "", 0,
                 time.time(), "undelivered"))
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r1 = mp.run_slot(ctx, "afternoon")
            text2, items2 = mp._compose_afternoon(ctx)
        self.assertEqual(r1["stories_delivered"], 4)
        self.assertEqual(items2, [])
        for i in range(4):
            self.assertNotIn(f"Story {i}", text2,
                              "second briefing repeated a delivered story")


def _tmpdir(ctx):
    return getattr(ctx.settings, "workspace_dir", tempfile.mkdtemp())


class ComposerNoDropTests(unittest.TestCase):
    """The morning composer must not drop whole sections to fit a cap."""

    def test_composer_drops_no_sections(self):
        from nomorals.agents import morning_briefing as mb
        from nomorals.agents.news import NewsAgent
        from nomorals.core.ids import new_id
        ctx, _ = make_ctx(self)
        NewsAgent(ctx)  # ensures the delivery columns (F6 seed realism:
        # every row poll() stores carries delivery_state='undelivered')
        for i in range(200):
            long_title = f"headline number {i} " + "word " * 80
            ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at, delivery_state) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (new_id(), f"src-{i % 5}", long_title, f"https://x/{i}",
                 "summary " * 40, 0, time.time(), "undelivered"))
        b = mb.BriefingComposer().compose(ctx, "2026-10-01")
        # no sections silently dropped, no false "full digest" promise
        self.assertEqual(b.truncated_note, "")
        names = {s.name for s in b.sections}
        self.assertIn("news", names)


class PulseCommandTests(unittest.TestCase):
    """Owner can check the schedule + trigger any briefing on demand."""

    def test_pulse_status_lists_all_four_slots(self):
        from nomorals.agents.partner.runtime_live import RuntimeLiveMixin
        ctx, tmp = make_ctx(self)
        ctx.settings = SimpleNamespace(workspace_dir=tmp,
                                        partner=make_partner(), pulse=None)
        mixin = RuntimeLiveMixin()
        mixin.context = ctx
        mixin.gateway = FakeGateway()
        out = mixin._control_pulse("status")
        for slot in ("morning", "afternoon", "evening", "midnight"):
            self.assertIn(slot, out)
        self.assertIn("00:00", out)

    def test_pulse_bare_means_status(self):
        from nomorals.agents.partner.runtime_live import RuntimeLiveMixin
        ctx, tmp = make_ctx(self)
        ctx.settings = SimpleNamespace(workspace_dir=tmp,
                                        partner=make_partner(), pulse=None)
        mixin = RuntimeLiveMixin()
        mixin.context = ctx
        mixin.gateway = FakeGateway()
        out = mixin._control_pulse("")
        self.assertIn("morning", out)
        self.assertIn("midnight", out)

    def test_pulse_run_midnight_delivers_full_text(self):
        from nomorals.agents.partner.runtime_live import RuntimeLiveMixin
        ctx, tmp = make_ctx(self)
        ctx.settings = SimpleNamespace(workspace_dir=tmp,
                                        partner=make_partner(), pulse=None)
        gw = FakeGateway()
        mixin = RuntimeLiveMixin()
        mixin.context = ctx
        mixin.gateway = gw
        # quiet news: no fresh items, so the briefing is the quiet text —
        # the point is the run path works and reports delivery honestly.
        with patch("nomorals.agents.morning_pulse._fresh_news_items",
                   return_value=[]):
            out = mixin._control_pulse("run midnight")
        self.assertIn("midnight", out.lower())
        self.assertIn("deliver", out.lower())


if __name__ == "__main__":
    unittest.main()
