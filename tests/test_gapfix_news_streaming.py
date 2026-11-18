"""Streaming news delivery (owner order 2026-11-18): the scheduled
digest is gone — stories arrive as they clear the interest bar.

The owner's live report: his morning briefing carried ~65 headlines
per page across 82 pages — thousands of stories in one digest.  He
rejected the cap ("That's not a fix delivering stories as they come
in sounds better to me"): the poll now STREAMS every story at or
above the interest bar through the anytime push in the same poll run,
and what quiet hours / the per-day cap hold surfaces in the next
slot's SHORT catch-up (top by interest, bounded), whose scored rest
archives once the catch-up confirms — the pile drains every slot.
``/news brief`` is the bounded on-demand peek at the latest stories.

Pins (this file):
(a) a mid-tier story (0.5, above the 0.35 bar) pushes in the SAME
    poll run as a high story — no more "held for the digest".
(b) the bar itself is the boundary: exactly 0.35 pushes, 0.34
    archives.
(c) quiet-hours / cap-held stories appear in the next slot's
    catch-up (top by interest, bounded) and then drain: the scored
    rest archives, the shown ones deliver, the next pick is empty.
(d) unscored rows are NEVER archived by the catch-up — no brain
    means no guess; they stay held.
(e) brief() is the bounded latest view: N rows in, at most the
    catch-up bound out, an honest "N more held" line, shown rows
    marked digested, the rest still undelivered for the stream —
    and a second brief() never re-shows the first.
(f) no scheduled digest exists on a seeded pile: with 60 held
    stories, no single call (poll / brief / pick) ever returns more
    than the bound.
(g) /news brief (the command path) returns the bounded latest view.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
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
        f"<description>{s}</description></item>"
        for t, u, s in items)
    return ('<?xml version="1.0"?><rss version="2.0"><channel>'
            + body + "</channel></rss>")


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


def _make_ctx(test, *, router=None, push_cap=None, catchup_max=None):
    tmp = tempfile.mkdtemp(prefix="stream-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    news_ns = SimpleNamespace(feeds="")
    if push_cap is not None:
        news_ns.push_cap_per_day = push_cap
    if catchup_max is not None:
        news_ns.catchup_max = catchup_max
    settings = SimpleNamespace(news=news_ns)
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    if router is not None:
        ctx.router = router
    return ctx


def _poll(ctx, xml, notifier=None, brain=None):
    import contextlib

    agent = NewsAgent(ctx, notifier=notifier)
    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch.object(agent, "_feeds", return_value=[FEED]))
    client = stack.enter_context(mock.patch.object(agent, "_client"))
    if brain is not None:
        stack.enter_context(mock.patch(
            "nomorals.agents.news.brain_for", return_value=brain))
    with stack:
        client.get.return_value = SimpleNamespace(text=xml)
        return agent, agent.poll()


def _poll_empty(ctx, notifier=None, brain=None):
    """A poll with no feeds — re-tiers the stored backlog only."""
    import contextlib

    agent = NewsAgent(ctx, notifier=notifier)
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(agent, "_feeds", return_value=[]))
    if brain is not None:
        stack.enter_context(mock.patch(
            "nomorals.agents.news.brain_for", return_value=brain))
    with stack:
        return agent, agent.poll()


def _rows(ctx):
    return ctx.db.query("SELECT * FROM news_items ORDER BY created_at, url")


def _seed(ctx, n, *, interest=None, prefix="Story", base=None):
    """Insert n undelivered rows with distinct published times."""
    agent = NewsAgent(ctx)  # ensures delivery columns
    base = base if base is not None else time.time()
    for i in range(n):
        ctx.db.execute(
            "INSERT INTO news_items (id, source, title, url, summary, "
            "published, created_at, delivery_state, interest) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (f"seed-{prefix}-{i}", "src", f"{prefix} {i}",
             f"https://x/{prefix}{i}", "", base + i, base + i,
             "undelivered", interest))
    return agent


class StreamTierTests(unittest.TestCase):
    """(a)+(b) the bar streams; below it archives."""

    def test_mid_tier_streams_in_the_same_poll(self):
        ctx = _make_ctx(self, router=object(), push_cap=10)
        brain = _StubBrain({"Gold surges": 0.95, "Council meets": 0.5,
                            "Celebrity gossip": 0.1})
        items = [("Gold surges to record", "https://feed.example.com/g",
                  "Gold body."),
                 ("Council meets on roads", "https://feed.example.com/c",
                  "Council body."),
                 ("Celebrity gossip roundup", "https://feed.example.com/x",
                  "Gossip body.")]
        notifier = _StubNotifier()
        _, report = _poll(ctx, _xml(items), notifier=notifier, brain=brain)
        self.assertEqual(report["interest"], "ok")
        self.assertEqual(report["send_bar"], 0.35)
        # mid (0.5) streams WITH high (0.95) — nothing waits for a digest
        self.assertEqual(report["pushed"], 2)
        self.assertEqual(report["archived"], 1)
        self.assertEqual(len(notifier.published), 2)
        titles = [p["title"] for p in notifier.published]
        self.assertTrue(any("Gold surges" in t for t in titles))
        self.assertTrue(any("Council meets" in t for t in titles))
        by_title = {r["title"]: r for r in _rows(ctx)}
        self.assertEqual(
            by_title["Council meets on roads"]["delivery_state"], "pushed")
        self.assertEqual(
            by_title["Celebrity gossip roundup"]["delivery_state"],
            "archived")
        self.assertEqual(
            by_title["Celebrity gossip roundup"]["interest"], 0.1)
        # nothing left undelivered for a digest to carry
        self.assertEqual(
            NewsAgent(ctx).status()["undelivered"], 0)

    def test_the_bar_itself_is_the_boundary(self):
        ctx = _make_ctx(self, router=object(), push_cap=10)
        brain = _StubBrain({"Edge high": 0.35, "Edge low": 0.34})
        items = [("Edge high story", "https://feed.example.com/hi",
                  "Body."),
                 ("Edge low story", "https://feed.example.com/lo",
                  "Body.")]
        notifier = _StubNotifier()
        _, report = _poll(ctx, _xml(items), notifier=notifier, brain=brain)
        self.assertEqual(report["pushed"], 1)
        self.assertEqual(report["archived"], 1)
        by_title = {r["title"]: r for r in _rows(ctx)}
        self.assertEqual(
            by_title["Edge high story"]["delivery_state"], "pushed")
        self.assertEqual(
            by_title["Edge low story"]["delivery_state"], "archived")


class CatchupTests(unittest.TestCase):
    """(c)+(d) held stories surface in the catch-up, then drain."""

    def test_quiet_held_story_reaches_the_catchup_then_streams(self):
        ctx = _make_ctx(self, router=object(), push_cap=10)
        brain = _StubBrain({"Gold surges": 0.95})
        items = [("Gold surges again", "https://feed.example.com/g2",
                  "Body.")]
        notifier = _StubNotifier()
        with mock.patch(
                "nomorals.agents.notifier._in_quiet_hours_now",
                return_value=True):
            _, report = _poll(ctx, _xml(items), notifier=notifier,
                              brain=brain)
        self.assertEqual(report["pushed"], 0)
        self.assertTrue(report["quiet_hold"])
        # the held story is the catch-up's whole list
        pick = NewsAgent(ctx).pick_catchup()
        self.assertEqual(len(pick["shown"]), 1)
        self.assertEqual(pick["archivable"], [])
        self.assertEqual(pick["not_shown"], 0)
        self.assertEqual(pick["line"], "")
        # a later poll outside quiet hours still streams it live —
        # the catch-up never consumed it
        with mock.patch(
                "nomorals.agents.notifier._in_quiet_hours_now",
                return_value=False):
            _, report2 = _poll_empty(ctx, notifier=notifier, brain=brain)
        self.assertEqual(report2["pushed"], 1)
        self.assertEqual(_rows(ctx)[0]["delivery_state"], "pushed")

    def test_cap_overflow_drains_through_the_catchup(self):
        ctx = _make_ctx(self, router=object(), push_cap=1)
        brain = _StubBrain({"Alert": 0.95})
        items = [(f"Alert story {i}", f"https://feed.example.com/a{i}",
                  "Body.") for i in range(3)]
        notifier = _StubNotifier()
        _, report = _poll(ctx, _xml(items), notifier=notifier, brain=brain)
        self.assertEqual(report["pushed"], 1)
        self.assertTrue(report["push_cap_reached"])
        agent = NewsAgent(ctx)
        pick = agent.pick_catchup()
        self.assertEqual(len(pick["shown"]), 2)
        self.assertEqual(pick["archivable"], [])
        self.assertEqual(pick["not_shown"], 0)
        # the slot delivers the catch-up, then the drain runs
        agent.mark_delivered(
            [r["url"] for r in pick["shown"]], channel="pulse-afternoon")
        self.assertEqual(
            agent.archive_rows(pick["archivable"],
                               channel="pulse-afternoon-catchup"), 0)
        self.assertEqual(agent.pick_catchup()["shown"], [])

    def test_big_held_pile_drains_top_first_then_empties(self):
        ctx = _make_ctx(self, catchup_max=10)
        agent = _seed(ctx, 12, interest=0.5, prefix="Held")
        pick = agent.pick_catchup()
        self.assertEqual(len(pick["shown"]), 10)
        self.assertEqual(len(pick["archivable"]), 2)
        self.assertEqual(pick["not_shown"], 2)
        self.assertIn("2 more stories were held", pick["line"])
        self.assertIn("2 archived after this catch-up", pick["line"])
        # the slot confirms the catch-up, then archives the rest
        agent.mark_delivered(
            [r["url"] for r in pick["shown"]], channel="pulse-morning")
        self.assertEqual(
            agent.archive_rows(pick["archivable"],
                               channel="pulse-catchup"), 2)
        states = sorted(r["delivery_state"] for r in _rows(ctx))
        self.assertEqual(states, ["archived", "archived"]
                         + ["digested"] * 10)
        self.assertEqual(agent.pick_catchup()["shown"], [])
        self.assertEqual(agent.status()["undelivered"], 0)

    def test_unscored_rows_are_never_archived(self):
        ctx = _make_ctx(self, catchup_max=10)
        agent = _seed(ctx, 12, interest=None, prefix="Unscored")
        pick = agent.pick_catchup()
        self.assertEqual(len(pick["shown"]), 10)
        self.assertEqual(pick["archivable"], [])
        self.assertEqual(pick["held_rest"], 2)
        self.assertEqual(pick["not_shown"], 2)
        self.assertIn("2 more stories were held", pick["line"])
        self.assertNotIn("archived", pick["line"])
        # the drain door has nothing to archive — all 12 stay held
        self.assertEqual(
            agent.archive_rows(pick["archivable"],
                               channel="pulse-catchup"), 0)
        self.assertEqual(agent.status()["undelivered"], 12)

    def test_catchup_ranks_by_interest_then_newest(self):
        ctx = _make_ctx(self, catchup_max=10)
        agent = NewsAgent(ctx)
        base = time.time()
        for title, interest, off in (
                ("Low story", 0.4, 1), ("Top story", 0.9, 2),
                ("Mid story", 0.6, 3)):
            ctx.db.execute(
                "INSERT INTO news_items (id, source, title, url, summary, "
                "published, created_at, delivery_state, interest) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (f"rank-{title}", "src", title,
                 f"https://x/{title.replace(' ', '')}", "", base + off,
                 base + off, "undelivered", interest))
        pick = agent.pick_catchup()
        self.assertEqual([r["title"] for r in pick["shown"]],
                         ["Top story", "Mid story", "Low story"])


class BriefTests(unittest.TestCase):
    """(e) brief() is the bounded latest view, never the pile."""

    def test_brief_shows_the_newest_bound_and_names_the_rest(self):
        ctx = _make_ctx(self, catchup_max=10)
        agent = _seed(ctx, 15, interest=None, prefix="Story")
        brief = agent.brief()
        self.assertTrue(brief["ok"])
        self.assertEqual(brief["items"], 10)
        self.assertEqual(brief["left_out"], 5)
        self.assertIn("5 more stories are held", brief["digest"])
        # the newest ten, by URL — the oldest five are not in it
        shown_urls = set(brief["delivered_urls"])
        self.assertEqual(
            shown_urls,
            {f"https://x/Story{i}" for i in range(5, 15)})
        rows = {r["url"]: r for r in _rows(ctx)}
        self.assertTrue(all(rows[u]["delivery_state"] == "digested"
                            for u in shown_urls))
        self.assertTrue(all(
            rows[f"https://x/Story{i}"]["delivery_state"] == "undelivered"
            for i in range(5)))
        # a second brief shows the next five — never a re-show
        brief2 = agent.brief()
        self.assertEqual(brief2["items"], 5)
        self.assertEqual(brief2["left_out"], 0)
        self.assertNotIn("more stories are held", brief2["digest"])
        self.assertEqual(
            set(brief2["delivered_urls"]),
            {f"https://x/Story{i}" for i in range(5)})
        self.assertFalse(agent.brief()["ok"])

    def test_brief_under_the_bound_is_unchanged(self):
        ctx = _make_ctx(self, catchup_max=10)
        agent = _seed(ctx, 4, interest=None, prefix="Small")
        brief = agent.brief()
        self.assertEqual(brief["items"], 4)
        self.assertEqual(brief["left_out"], 0)
        self.assertNotIn("more stories are held", brief["digest"])
        self.assertTrue(all(
            r["delivery_state"] == "digested" for r in _rows(ctx)))

    def test_brief_limit_is_an_explicit_override(self):
        ctx = _make_ctx(self, catchup_max=10)
        agent = _seed(ctx, 8, interest=None, prefix="Lim")
        brief = agent.brief(limit=3)
        self.assertEqual(brief["items"], 3)
        self.assertEqual(brief["left_out"], 5)


class NoDigestTests(unittest.TestCase):
    """(f) no scheduled digest exists on a seeded pile."""

    def test_seeded_pile_never_produces_a_full_digest(self):
        ctx = _make_ctx(self, push_cap=0, catchup_max=10)
        agent = _seed(ctx, 60, interest=0.5, prefix="Pile")
        _, report = _poll_empty(ctx, notifier=_StubNotifier())
        self.assertEqual(report["pushed"], 0)
        self.assertEqual(report["held"], 60)
        # the slot's catch-up is bounded (the pick changes no state)
        pick = agent.pick_catchup()
        self.assertEqual(len(pick["shown"]), 10)
        self.assertEqual(len(pick["archivable"]), 50)
        self.assertEqual(pick["not_shown"], 50)
        # and so is the on-demand view
        brief = agent.brief()
        self.assertEqual(brief["items"], 10)
        self.assertEqual(brief["left_out"], 50)
        self.assertEqual(brief["digest"].count("Pile "), 10)


class CommandPathTests(unittest.TestCase):
    """(g) the /news command path returns the bounded latest view."""

    def test_news_brief_command_returns_the_bounded_view(self):
        from nomorals.agents.partner.runtime_live import RuntimeLiveMixin

        ctx = _make_ctx(self, catchup_max=10)
        _seed(ctx, 15, interest=None, prefix="Cmd")
        mixin = RuntimeLiveMixin()
        mixin.context = ctx
        # the bare mixin lacks the runtime's offline gate; the real
        # method returns None with no offline host — same answer here
        mixin._offline_feature_refusal = lambda feature: None
        out = mixin._control_news("brief")
        self.assertIn("news — latest 10 stories", out)
        self.assertIn("5 more stories are held", out)


if __name__ == "__main__":
    unittest.main()
