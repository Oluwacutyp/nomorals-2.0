"""PASS #85: news never drops a story — continuous poll, interest-gated
pushes, complete digests.

The defect (owner spec 2026-10-21): ``NewsAgent.run()`` coupled fetch
and digest — it stored and delivered only ``fresh[:cap]`` (default 12,
max 40).  Stories beyond the cap were silently dropped, there was no
interest model, no anytime delivery, and no delivery state: storage
itself was the seen-marker, so "stored" and "delivered" were the same
fact and neither could be audited.

Pins:
(a) poll() stores EVERY fresh story (no cap), each row carrying an
    honest delivery_state starting at "undelivered".
(b) run()'s digest carries ALL undelivered stories — a 15-story poll
    under the legacy cap of 5 still digests 15 — and marks exactly
    those rows delivered; a second run redelivers nothing.
(c) with NO brain (no router — the module's own convention), scoring
    degrades honestly: nothing is pushed, scores stay NULL, and
    status() reports interest "unavailable" instead of inventing one.
(d) with a brain, stories tier by score: high → anytime push through
    the notifier (marked "pushed"), mid → held for the digest,
    low → "archived" (stored, queryable, never delivered).
(e) pushes obey delivery discipline: quiet hours hold them (the story
    waits, undelivered, and a later poll outside quiet hours pushes
    it), and a per-day cap stops further pushes (overflow waits for
    the digest).
(f) the interest profile persists: owner seed beats ship as data,
    pin()/unpin() round-trip across agent instances, and briefing
    topics the owner already configured count as stated interests.
(g) mark_delivered() lets the pulse surfaces record their deliveries
    in the same state machine — a story the pulse delivered is not
    re-digested.
(h) legacy rows (pre-pass shape: delivery columns NULL) are treated
    as already delivered — never re-digested, still listed by recent().
(i) cross-run dedupe stays single-sourced in news_items: the same
    story arriving via a second route in a later poll stores nothing.
(j) the "news" agent tool exists, binds the scheduler's exact call
    shape, and a poll through the registry really stores stories.
(k) the "news poll" scheduler job registers via ensure_news_jobs
    (finance/jobs.py pattern), executes end-to-end through the real
    scheduler → registry → tool → poll path, and is idempotent.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.news import NewsAgent
from nomorals.storage.db import Database

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


class _StubBrain:
    """Scores by title substring; answers summaries with silence."""

    def __init__(self, scores):
        self.scores = scores  # substring -> float

    def chat(self, messages, task_kind=None):
        prompt = messages[-1].content if messages else ""
        if "interest" not in prompt.lower():
            return SimpleNamespace(text="")
        lines = []
        for line in prompt.splitlines():
            m = re.match(r"(\d+)\.\s+(.*)", line.strip())
            if not m:
                continue
            score = 0.5
            for key, val in self.scores.items():
                if key.lower() in m.group(2).lower():
                    score = val
            lines.append(f"{m.group(1)}. {score}")
        return SimpleNamespace(text="\n".join(lines))


class _StubNotifier:
    def __init__(self):
        self.published = []

    def publish(self, kind, title, body="", **kw):
        self.published.append(
            {"kind": kind, "title": title, "body": body})
        return {"delivered": True, "delivery_state": "sent"}


def _make_ctx(test, *, cap=5, router=None, push_cap=None, feeds=""):
    tmp = tempfile.mkdtemp(prefix="pass85-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    news_ns = SimpleNamespace(feeds=feeds, cap=cap)
    if push_cap is not None:
        news_ns.push_cap_per_day = push_cap
    settings = SimpleNamespace(news=news_ns)
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    if router is not None:
        ctx.router = router
    return ctx


def _poll(ctx, xml, notifier=None):
    agent = NewsAgent(ctx, notifier=notifier)
    with mock.patch.object(agent, "_feeds", return_value=[FEED]), \
            mock.patch.object(agent, "_client") as client:
        client.get.return_value = SimpleNamespace(text=xml)
        return agent, agent.poll()


def _run(ctx, xml, notifier=None):
    agent = NewsAgent(ctx, notifier=notifier)
    with mock.patch.object(agent, "_feeds", return_value=[FEED]), \
            mock.patch.object(agent, "_client") as client:
        client.get.return_value = SimpleNamespace(text=xml)
        return agent, agent.run()


def _rows(ctx):
    return ctx.db.query("SELECT * FROM news_items ORDER BY created_at, url")


# ── (a)+(b) poll stores everything; the digest carries everything ────

class PollAndDigestTests(unittest.TestCase):

    def test_poll_stores_beyond_the_legacy_cap(self):
        ctx = _make_ctx(self, cap=5)
        _, report = _poll(ctx, _xml(_items(15)))
        self.assertEqual(report["polled"], 15)
        rows = _rows(ctx)
        self.assertEqual(len(rows), 15)
        self.assertTrue(
            all(r["delivery_state"] == "undelivered" for r in rows),
            [r["delivery_state"] for r in rows])

    def test_digest_carries_every_undelivered_story(self):
        ctx = _make_ctx(self, cap=5)
        _, report = _run(ctx, _xml(_items(15)))
        self.assertTrue(report["ok"])
        self.assertEqual(report["items"], 15)
        for i in range(15):
            self.assertIn(f"Story {i}", report["digest"])
        rows = _rows(ctx)
        self.assertTrue(
            all(r["delivery_state"] == "digested" for r in rows))
        # a second run over the same feed redelivers nothing
        _, report2 = _run(ctx, _xml(_items(15)))
        self.assertEqual(report2["items"], 0)
        self.assertEqual(report2["digest"], "")

    def test_brief_alone_delivers_the_backlog(self):
        ctx = _make_ctx(self)
        agent, _ = _poll(ctx, _xml(_items(4)))
        brief = agent.brief()
        self.assertEqual(brief["items"], 4)
        for i in range(4):
            self.assertIn(f"Story {i}", brief["digest"])
        self.assertEqual(len(brief["delivered_urls"]), 4)
        self.assertTrue(all(
            r["delivery_state"] == "digested" for r in _rows(ctx)))


# ── (c) no brain: honest degradation ───────────────────────────────────

class NoBrainTests(unittest.TestCase):

    def test_no_brain_pushes_nothing_and_says_so(self):
        ctx = _make_ctx(self)  # no router attribute at all
        notifier = _StubNotifier()
        _, report = _poll(ctx, _xml(_items(3)), notifier=notifier)
        self.assertEqual(report["pushed"], 0)
        self.assertEqual(notifier.published, [])
        self.assertEqual(report["interest"], "unavailable")
        rows = _rows(ctx)
        self.assertTrue(all(r["interest"] is None for r in rows))
        status = NewsAgent(ctx).status()
        self.assertEqual(status["interest"], "unavailable")
        self.assertEqual(status["undelivered"], 3)


# ── (d)+(e) brain tiers + delivery discipline ─────────────────────────

class BrainTierTests(unittest.TestCase):

    def _brain_ctx(self, scores, **kw):
        ctx = _make_ctx(self, router=object(), **kw)
        brain = _StubBrain(scores)
        return ctx, mock.patch(
            "nomorals.agents.news.brain_for", return_value=brain)

    def test_tiers_push_hold_archive(self):
        ctx, patcher = self._brain_ctx(
            {"Gold surges": 0.95, "Council meets": 0.5,
             "Celebrity gossip": 0.1})
        items = [("Gold surges to record", "https://feed.example.com/g",
                  "Gold body."),
                 ("Council meets on roads", "https://feed.example.com/c",
                  "Council body."),
                 ("Celebrity gossip roundup", "https://feed.example.com/x",
                  "Gossip body.")]
        notifier = _StubNotifier()
        with patcher:
            _, report = _poll(ctx, _xml(items), notifier=notifier)
        self.assertEqual(report["interest"], "ok")
        self.assertEqual(report["pushed"], 1)
        self.assertEqual(report["archived"], 1)
        self.assertEqual(len(notifier.published), 1)
        self.assertIn("Gold surges", notifier.published[0]["title"])
        by_title = {r["title"]: r for r in _rows(ctx)}
        self.assertEqual(
            by_title["Gold surges to record"]["delivery_state"], "pushed")
        self.assertEqual(
            by_title["Gold surges to record"]["interest"], 0.95)
        self.assertEqual(
            by_title["Council meets on roads"]["delivery_state"],
            "undelivered")
        self.assertEqual(
            by_title["Celebrity gossip roundup"]["delivery_state"],
            "archived")
        # the digest then carries ONLY the held story
        with patcher:
            brief = NewsAgent(ctx).brief()
        self.assertIn("Council meets", brief["digest"])
        self.assertNotIn("Gold surges", brief["digest"])
        self.assertNotIn("Celebrity gossip", brief["digest"])

    def test_quiet_hours_hold_then_later_poll_pushes(self):
        ctx, patcher = self._brain_ctx({"Gold surges": 0.95})
        items = [("Gold surges again", "https://feed.example.com/g2",
                  "Body.")]
        notifier = _StubNotifier()
        with patcher, mock.patch(
                "nomorals.agents.notifier._in_quiet_hours_now",
                return_value=True):
            _, report = _poll(ctx, _xml(items), notifier=notifier)
        self.assertEqual(report["pushed"], 0)
        self.assertTrue(report["quiet_hold"])
        self.assertEqual(notifier.published, [])
        self.assertEqual(_rows(ctx)[0]["delivery_state"], "undelivered")
        # a later poll, outside quiet hours, pushes the held story —
        # no new items needed
        with patcher, mock.patch(
                "nomorals.agents.notifier._in_quiet_hours_now",
                return_value=False):
            _, report2 = _poll(ctx, _xml([]), notifier=notifier)
        self.assertEqual(report2["pushed"], 1)
        self.assertEqual(_rows(ctx)[0]["delivery_state"], "pushed")

    def test_daily_push_cap_holds_the_overflow(self):
        ctx, patcher = self._brain_ctx({"Alert": 0.95}, push_cap=2)
        items = [(f"Alert story {i}", f"https://feed.example.com/a{i}",
                  "Body.") for i in range(4)]
        notifier = _StubNotifier()
        with patcher:
            _, report = _poll(ctx, _xml(items), notifier=notifier)
        self.assertEqual(report["pushed"], 2)
        self.assertTrue(report["push_cap_reached"])
        states = sorted(r["delivery_state"] for r in _rows(ctx))
        self.assertEqual(states, ["pushed", "pushed",
                                  "undelivered", "undelivered"])


# ── (f) the interest profile ────────────────────────────────────────────

class InterestProfileTests(unittest.TestCase):

    def test_seeds_ship_as_data(self):
        ctx = _make_ctx(self)
        profile = NewsAgent(ctx).interest_profile()
        beats = " ".join(profile["beats_text"].lower().split())
        for want in ("xauusd", "ai", "nigeria", "devon", "loans"):
            self.assertIn(want, beats)

    def test_pin_unpin_persist_across_instances(self):
        ctx = _make_ctx(self)
        agent = NewsAgent(ctx)
        agent.pin("pension funds")
        again = NewsAgent(ctx)
        self.assertIn("pension funds",
                      [p.lower() for p in again.interest_profile()["pins"]])
        again.unpin("pension funds")
        self.assertNotIn(
            "pension funds",
            [p.lower() for p in NewsAgent(ctx).interest_profile()["pins"]])

    def test_briefing_topics_count_as_stated_interests(self):
        ctx = _make_ctx(self)
        ctx.settings.briefing = SimpleNamespace(topics=["crude oil"])
        profile = NewsAgent(ctx).interest_profile()
        self.assertIn("crude oil", profile["beats_text"].lower())


# ── (g)+(h) shared state machine + legacy rows ─────────────────────────

class SharedStateTests(unittest.TestCase):

    def test_mark_delivered_excludes_from_digest(self):
        ctx = _make_ctx(self)
        agent, _ = _poll(ctx, _xml(_items(3)))
        agent.mark_delivered(["https://feed.example.com/Story0"],
                             channel="pulse")
        brief = agent.brief()
        self.assertEqual(brief["items"], 2)
        self.assertNotIn("Story 0", brief["digest"])
        self.assertIn("Story 1", brief["digest"])
        row = [r for r in _rows(ctx)
               if r["url"].endswith("Story0")][0]
        self.assertEqual(row["delivery_state"], "digested")
        self.assertEqual(row["delivery_channel"], "pulse")

    def test_legacy_rows_never_redigested_but_still_recent(self):
        ctx = _make_ctx(self)
        ctx.db.execute(
            "INSERT INTO news_items (id, source, title, url, summary, "
            "published, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy1", "Old Feed", "Legacy headline",
             "https://old.example.com/1", "Old body.", 0.0, 1761091200.0))
        agent = NewsAgent(ctx)
        brief = agent.brief()
        self.assertNotIn("Legacy headline", brief["digest"])
        rows = agent.recent(limit=5)
        self.assertTrue(any(r["title"] == "Legacy headline"
                            for r in rows))


# ── (i) cross-run dedupe stays single-sourced ──────────────────────────

class CrossRunDedupeTests(unittest.TestCase):

    def test_same_story_second_route_stores_nothing(self):
        ctx = _make_ctx(self)
        story = [("Markets rally on open - Reuters",
                  "https://news.google.com/rss/articles/AAA?oc=5",
                  "Stocks climb.")]
        _poll(ctx, _xml(story))
        other_route = [("Markets rally on open - Reuters",
                        "https://news.google.com/rss/articles/BBB?oc=5",
                        "Stocks climb.")]
        _, report = _poll(ctx, _xml(other_route))
        self.assertEqual(report["polled"], 0)
        self.assertEqual(len(_rows(ctx)), 1)


# ── (j)+(k) the tool + the job ──────────────────────────────────────────

class ToolAndJobTests(unittest.TestCase):

    def _registry_ctx(self):
        from nomorals.tools.registry import ToolRegistry
        tmp = tempfile.mkdtemp(prefix="pass85-reg-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = Database(os.path.join(tmp, "test.db"))
        db.migrate()
        ctx = SimpleNamespace(
            db=db,
            settings=SimpleNamespace(
                news=SimpleNamespace(feeds="", cap=5),
                home_path=tmp, workspace_dir=tmp),
            extras={}, tools=None)
        registry = ToolRegistry(ctx)
        registry.register_builtins()
        ctx.tools = registry
        return ctx, registry

    def test_news_tool_registered_and_binds(self):
        _, registry = self._registry_ctx()
        spec = registry._tools.get("news")
        self.assertIsNotNone(spec, "no 'news' tool in the registry")
        inspect.signature(spec.fn).bind(action="poll")
        inspect.signature(spec.fn).bind(action="status")
        inspect.signature(spec.fn).bind(action="brief")

    def test_news_tool_poll_really_stores(self):
        from nomorals.core.http import HttpClient
        ctx, registry = self._registry_ctx()
        xml = _xml(_items(2, prefix="ToolStory"))
        with mock.patch.object(HttpClient, "get",
                               return_value=SimpleNamespace(text=xml)):
            outcome = registry.call("news", action="poll")
        self.assertTrue(outcome.ok,
                        getattr(outcome, "error", outcome))
        rows = ctx.db.query("SELECT COUNT(*) AS n FROM news_items")
        self.assertEqual(rows[0]["n"], 2)

    def test_poll_job_registers_executes_and_is_idempotent(self):
        from nomorals.agents.scheduler import Scheduler
        from nomorals.agents.news_jobs import ensure_news_jobs
        from nomorals.core.http import HttpClient

        class _SchedNotifier:
            def publish(self, kind, title, body="", **kw):
                return {"delivered": True, "channel": "test"}

        ctx, _ = self._registry_ctx()
        sched = Scheduler(ctx)
        sched.notifier = _SchedNotifier()
        out = ensure_news_jobs(sched)
        self.assertEqual([o["name"] for o in out], ["news poll"])
        rows = [j for j in sched.list_jobs() if j["name"] == "news poll"]
        self.assertEqual(len(rows), 1)
        payload = json.loads(ctx.db.query_one(
            "SELECT payload FROM schedule_jobs WHERE name = ?",
            ("news poll",))["payload"] or "{}")
        self.assertEqual(payload, {"tool": "news",
                                   "args": {"action": "poll"}})
        xml = _xml(_items(2, prefix="JobStory"))
        with mock.patch.object(HttpClient, "get",
                               return_value=SimpleNamespace(text=xml)):
            result = sched.run_now(rows[0]["id"])
        self.assertTrue(result["ok"], result)
        stored = ctx.db.query("SELECT COUNT(*) AS n FROM news_items")
        self.assertEqual(stored[0]["n"], 2)
        ensure_news_jobs(sched)
        rows = [j for j in sched.list_jobs() if j["name"] == "news poll"]
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
