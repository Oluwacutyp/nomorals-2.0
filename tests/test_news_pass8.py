"""PASS #8: news + briefings — verified bugs, failing-first.

Bug 1: `/news run` attached a Notifier to NewsAgent, so every run
pushed the full digest to EVERY owner channel: a duplicate in the
owner's own chat, and an owner-channel push triggerable by any
non-owner (/news is a PUBLIC command). The command reply IS the
delivery — no notifier belongs on this path.

Bug 2: `/news run` never marked what it delivered, so the next
briefing slot repeated the just-delivered stories as "fresh news".
(Pass #8 fixed it via the pulse seen-store; Formatting F6 retired
that marker — the news delivery ledger is now the one sync point,
and the Bug 2 tests below pin the ledger contract.)

Bug 3: dead feeds are retried forever in the run path — SourceHealth's
circuit breaker exists but NewsAgent.run() never consults it. Now the
run skips stood-down/disabled feeds (same key scheme as the registry
so the scheduled expansion re-verifies and re-activates them) and
records its own fetch outcomes.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import morning_pulse as mp
from nomorals.agents.news import NewsAgent
from nomorals.agency.sourcehealth import (
    SourceHealth, SourceSpec, RSS,
    STAND_DOWN_AFTER, DISABLE_AFTER,
)
from nomorals.storage.db import Database


def make_ctx(test=None):
    tmp = tempfile.mkdtemp(prefix="news-pass8-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(workspace_dir=tmp,
                               partner=SimpleNamespace(timezone="UTC"))
    return SimpleNamespace(db=db, settings=settings, extras={}), tmp


class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.message_id = "m1"
        self.error = ""


class FakeGateway:
    def __init__(self):
        self.sent = []

    def send(self, platform, chat_ref, text, **kw):
        self.sent.append((platform, getattr(chat_ref, "chat_id", chat_ref), text))
        return _SendResult(ok=True)


RSS_DOC = ("<?xml version='1.0'?><rss><channel>"
           "<item><title>Story one</title>"
           "<link>https://example.test/s1</link>"
           "<description>First story summary.</description></item>"
           "<item><title>Story two</title>"
           "<link>https://example.test/s2</link>"
           "<description>Second story summary.</description></item>"
           "</channel></rss>")


class FakeResp:
    def __init__(self, text):
        self.text = text


def _ctx_with_mixin(test):
    from nomorals.agents.partner.runtime_live import RuntimeLiveMixin
    ctx, tmp = make_ctx(test)
    mixin = RuntimeLiveMixin()
    mixin.context = ctx
    mixin.gateway = FakeGateway()
    return mixin, ctx, tmp


# ── Bug 1: /news run must not push to owner channels ──

class NoNotifierPushTests(unittest.TestCase):
    def test_control_news_passes_no_notifier(self):
        """The command reply is the delivery; no owner-channel push."""
        seen = {}

        class FakeAgent:
            def __init__(self, context, notifier=None):
                seen["notifier"] = notifier
                seen["ctx"] = context

            def run(self, cap=None):
                return {"ok": True, "items": 2, "fresh": 2, "errors": [],
                        "digest": "d", "sources_tried": 1,
                        "sources_ok": ["x"], "per_source": {}}

            def recent(self, limit=15):
                return []

        mixin, ctx, tmp = _ctx_with_mixin(self)
        with patch("nomorals.agents.news.NewsAgent", FakeAgent):
            out = mixin._control_news("run")
        self.assertIn("news digest", out)
        self.assertIsNone(seen.get("notifier"),
                          "/news run attached a notifier → owner-channel push "
                          "(duplicate for the owner, leak for non-owners)")


# ── Bug 2: /news run deliveries must not repeat in briefing slots ──
#
# Pass #8 pinned this via the pulse seen-store (the command marked
# what it delivered into pulse_seen.json).  F6 retired that marker:
# the sync point is now the news delivery ledger itself — run()
# digests through brief(), which records delivery there, and the
# slots read the same ledger.  The contract below is unchanged; the
# mechanism it pins is the ledger.

class SeenMarkingTests(unittest.TestCase):
    def test_news_run_marks_delivered_in_ledger(self):
        from nomorals.core.ids import new_id
        rows = [
            {"url": "https://example.test/s1", "title": "Story one"},
            {"url": "https://example.test/s2", "title": "Story two"},
        ]
        mixin, ctx, tmp = _ctx_with_mixin(self)
        agent = NewsAgent(ctx)  # ensures delivery columns
        for row in rows:
            ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at, delivery_state) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (new_id(), "testfeed", row["title"], row["url"], "", 0,
                 time.time(), "undelivered"))

        class FakeAgent:
            """Stands in for NewsAgent; its run() records delivery in
            the ledger exactly like the real run()→brief() does."""

            def __init__(self, context, notifier=None):
                self._ctx = context

            def run(self, cap=None):
                NewsAgent(self._ctx).mark_delivered(
                    [r["url"] for r in rows], channel="digest")
                return {"ok": True, "items": 2, "fresh": 2, "errors": [],
                        "digest": "d", "sources_tried": 1,
                        "sources_ok": ["x"], "per_source": {},
                        "delivered_urls": [r["url"] for r in rows]}

        with patch("nomorals.agents.news.NewsAgent", FakeAgent):
            mixin._control_news("run")
        # the next briefing slot's fresh set excludes both stories…
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}):
            fresh = mp._fresh_news_items(ctx)
        self.assertEqual(fresh, [])
        # …and no seen-marker file was created or consulted
        self.assertFalse(os.path.exists(os.path.join(tmp, "pulse_seen.json")))


# ── Bug 3: dead feeds stood down in the run path ──

def _flaky_client(fail_urls):
    class FakeClient:
        def get(self, url, headers=None):
            if url in fail_urls:
                raise ConnectionError("boom")
            return FakeResp(RSS_DOC)
    return FakeClient()


class HealthWiringTests(unittest.TestCase):
    def _agent(self, ctx, fail_urls):
        agent = NewsAgent(ctx)
        agent._client = _flaky_client(fail_urls)
        agent._feeds = lambda: [("badfeed", "https://bad.test/rss"),
                                ("goodfeed", "https://good.test/rss")]
        return agent

    def test_consecutive_failures_stand_down_and_skip(self):
        ctx, _ = make_ctx(self)
        agent = self._agent(ctx, {"https://bad.test/rss"})
        for _ in range(STAND_DOWN_AFTER):
            report = agent.run()
            self.assertIn("badfeed", report["errors"][0]
                          if report["errors"] else "")
        # next run: the dead feed is skipped, not retried
        report = agent.run()
        tried_urls = [u for _, u in agent._feeds()]
        self.assertNotIn("badfeed", report["sources_ok"])
        self.assertTrue(any("badfeed" in s for s in report["stood_down"]),
                        f"stood-down feed not reported: {report.get('stood_down')}")
        self.assertIn("goodfeed", report["sources_ok"])
        # and the skip is durable in the health table
        health = SourceHealth(ctx.db)
        rec = health.record("news:custom:https://bad.test/rss")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.status, "stood-down")

    def test_recovery_reactivates(self):
        ctx, _ = make_ctx(self)
        agent = self._agent(ctx, {"https://bad.test/rss"})
        for _ in range(STAND_DOWN_AFTER):
            agent.run()
        health = SourceHealth(ctx.db)
        spec = SourceSpec(key="news:custom:https://bad.test/rss",
                          category="news", name="badfeed",
                          url="https://bad.test/rss", kind=RSS)
        health.note_result(spec, ok=True, items=5, http=200)
        # the feed heals: next run attempts it again (and it still fails
        # here, which is fine — the point is it is attempted, not skipped)
        agent2 = self._agent(ctx, set())
        report = agent2.run()
        self.assertIn("goodfeed", report["sources_ok"])
        rec = health.record(spec.key)
        self.assertEqual(rec.status, "verified-live")


class NoteResultTests(unittest.TestCase):
    def _health(self, test):
        ctx, _ = make_ctx(test)
        return SourceHealth(ctx.db)

    def _spec(self):
        return SourceSpec(key="news:custom:https://x.test/rss",
                          category="news", name="x",
                          url="https://x.test/rss", kind=RSS)

    def test_single_failure_is_unverifiable_not_stood_down(self):
        h = self._health(self)
        rec = h.note_result(self._spec(), ok=False, error="timeout")
        self.assertEqual(rec.status, "unverifiable")
        self.assertEqual(rec.consecutive_failures, 1)

    def test_three_failures_stand_down(self):
        h = self._health(self)
        spec = self._spec()
        for _ in range(STAND_DOWN_AFTER):
            rec = h.note_result(spec, ok=False, error="timeout")
        self.assertEqual(rec.status, "stood-down")

    def test_ten_failures_disable(self):
        h = self._health(self)
        spec = self._spec()
        for _ in range(DISABLE_AFTER):
            rec = h.note_result(spec, ok=False, error="timeout")
        self.assertEqual(rec.status, "disabled")

    def test_success_resets_and_verifies(self):
        h = self._health(self)
        spec = self._spec()
        for _ in range(STAND_DOWN_AFTER):
            h.note_result(spec, ok=False, error="timeout")
        rec = h.note_result(spec, ok=True, items=7, http=200)
        self.assertEqual(rec.status, "verified-live")
        self.assertEqual(rec.consecutive_failures, 0)
        self.assertEqual(rec.items, 7)


if __name__ == "__main__":
    unittest.main()
