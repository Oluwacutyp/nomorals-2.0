"""L4 — research pipeline: the worth gate must be conservative, delivery must
be exactly-once, and the scheduler must not hot-loop failures.

The security property that matters: a background pipeline that can message
the owner must default to silence. Every accept-path test has a matching
reject-path test.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nomorals.core.result import Err, Ok
from nomorals.research.pipeline import (
    Assessment,
    ResearchContext,
    ResearchFinding,
    ResearchJob,
    assess_worth,
    deliver,
    ensure_schema,
    execute_job,
    run_job,
)
from nomorals.research.scheduler import ResearchScheduler, default_jobs
from nomorals.storage.db import Database


def _finding(**kw):
    base = dict(
        job_id="money-watch",
        title="Outlier AI is hiring coding experts in Nigeria today",
        url="https://example.com/outlier-nigeria",
        snippet=(
            "Outlier AI just launched a new coding-expert project open to "
            "Nigeria. Apply this week: $25/hr, remote, weekly payout."
        ),
    )
    base.update(kw)
    return ResearchFinding(**base)


class _FakeRegistry:
    """Canned web_search / web_fetch."""

    def __init__(self, results=None, fetch_text="full article text here"):
        self._results = results if results is not None else []
        self._fetch_text = fetch_text
        self.calls = []

    def call(self, name, *, actor="system", **kwargs):
        self.calls.append((name, kwargs))
        if name == "web_search":
            return Ok({"query": kwargs.get("query"), "results": self._results})
        if name == "web_fetch":
            return Ok({"url": kwargs.get("url"), "text": self._fetch_text})
        return Err(ValueError(f"unknown tool {name}"))


class _FailRegistry:
    def call(self, name, *, actor="system", **kwargs):
        return Err(ValueError("network down"))


class _FakeGateway:
    def __init__(self, owner_chats=None, ok=True):
        self.owner_chats = set(owner_chats or ())
        self.sent = []
        self._ok = ok

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, str(chat), text))
        return SimpleNamespace(
            ok=self._ok, platform=platform, message_id="m1", error="" if self._ok else "boom"
        )


class _FakeMemory:
    def __init__(self, records=()):
        self._records = list(records)

    def recall(self, query, limit=8):
        return SimpleNamespace(records=self._records[:limit])


def _ctx(db, registry=None, gateway=None, memory=None, **kw):
    args = dict(db=db, registry=registry or _FakeRegistry(), memory=memory, gateway=gateway)
    args.update(kw)
    return ResearchContext(**args)


class AssessWorthTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        ensure_schema(self.db)

    def tearDown(self):
        self.db.close()

    def test_accepts_fresh_relevant_actionable(self):
        a = assess_worth(_finding(), _ctx(self.db))
        self.assertTrue(a.worth, a.reasons)
        self.assertGreaterEqual(a.score, 0.65)

    def test_rejects_already_delivered(self):
        import hashlib

        url = "https://example.com/outlier-nigeria"
        h = hashlib.sha256(url.strip().lower().encode()).hexdigest()[:32]
        self.db.execute(
            "INSERT INTO research_deliveries (url_hash, job_id, title, delivered_at, score, channel)"
            " VALUES (?, 'j', 't', ?, 0.9, 'telegram:1')",
            (h, time.time()),
        )
        a = assess_worth(_finding(url=url), _ctx(self.db))
        self.assertFalse(a.worth)
        self.assertIn("already delivered", a.reasons)

    def test_rejects_irrelevant(self):
        f = _finding(
            title="Best gardening tips for spring planting season",
            url="https://example.com/garden",
            snippet=(
                "A complete guide to planting tomatoes and roses this spring "
                "with soil preparation advice for beginners."
            ),
        )
        a = assess_worth(f, _ctx(self.db))
        self.assertFalse(a.worth)
        self.assertIn("no relevance to owner goals", a.reasons)

    def test_rejects_thin_content(self):
        f = _finding(title="Hi", url="https://example.com/x", snippet="short")
        a = assess_worth(f, _ctx(self.db))
        self.assertFalse(a.worth)
        self.assertIn("too thin to judge", a.reasons)

    def test_rejects_already_known(self):
        mem = _FakeMemory(
            [SimpleNamespace(text="Outlier AI is hiring coding experts in Nigeria today")]
        )
        a = assess_worth(_finding(), _ctx(self.db, memory=mem))
        self.assertFalse(a.worth)
        self.assertTrue(any("already known" in r for r in a.reasons))

    def test_rejects_seen_recently_not_delivered(self):
        import hashlib

        url = "https://example.com/outlier-nigeria"
        h = hashlib.sha256(url.strip().lower().encode()).hexdigest()[:32]
        self.db.execute(
            "INSERT INTO research_seen (url_hash, job_id, title, first_seen, best_score)"
            " VALUES (?, 'money-watch', 't', ?, 0.4)",
            (h, time.time() - 3600),
        )
        a = assess_worth(_finding(url=url), _ctx(self.db))
        self.assertFalse(a.worth)
        self.assertIn("seen recently, not delivered", a.reasons)

    def test_needs_threshold_despite_relevance(self):
        # Relevant but stale and not actionable: relevance 0.30 + novelty 0.25
        # = 0.55 < 0.65 -> silence. This is the conservative core.
        f = _finding(
            title="Outlier AI platform overview and company history",
            url="https://example.com/outlier-about",
            snippet=(
                "Outlier AI is a Scale AI company that hires experts for "
                "AI training work across many countries and domains."
            ),
        )
        a = assess_worth(f, _ctx(self.db))
        self.assertFalse(a.worth, a.reasons)


class RunJobTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        ensure_schema(self.db)

    def tearDown(self):
        self.db.close()

    def test_dedupes_urls_across_queries(self):
        reg = _FakeRegistry(
            results=[
                {"title": "A", "url": "https://example.com/a", "snippet": "s"},
                {"title": "A again", "url": "https://example.com/a", "snippet": "s"},
                {"title": "B", "url": "https://example.com/b", "snippet": "s"},
            ]
        )
        job = ResearchJob(id="j", topic="t", queries=["q1", "q2"], fetch_top=0)
        findings = run_job(job, _ctx(self.db, registry=reg))
        self.assertEqual(len(findings), 2)
        self.assertEqual(reg.calls[0][0], "web_search")

    def test_fetches_top_results(self):
        reg = _FakeRegistry(
            results=[{"title": "A", "url": "https://example.com/a", "snippet": "s"}],
            fetch_text="DEEP DIVE",
        )
        job = ResearchJob(id="j", topic="t", queries=["q1"], fetch_top=1)
        findings = run_job(job, _ctx(self.db, registry=reg))
        self.assertEqual(findings[0].detail, "DEEP DIVE")

    def test_all_endpoints_failing_raises(self):
        job = ResearchJob(id="j", topic="t", queries=["q1"])
        with self.assertRaises(RuntimeError):
            run_job(job, _ctx(self.db, registry=_FailRegistry()))

    def test_job_without_queries_raises(self):
        job = ResearchJob(id="j", topic="t", queries=[])
        with self.assertRaises(ValueError):
            run_job(job, _ctx(self.db))


class DeliverTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        ensure_schema(self.db)

    def tearDown(self):
        self.db.close()

    def test_delivers_and_records(self):
        gw = _FakeGateway(owner_chats={"telegram:123"})
        rctx = _ctx(self.db, gateway=gw)
        a = assess_worth(_finding(), rctx)
        self.assertTrue(a.worth)
        sent = deliver(_finding(), a, rctx)
        self.assertEqual(sent, ["telegram:123"])
        self.assertEqual(len(gw.sent), 1)
        self.assertIn("Outlier AI", gw.sent[0][2])
        # second assess of the same finding now rejects: exactly-once
        a2 = assess_worth(_finding(), rctx)
        self.assertFalse(a2.worth)

    def test_no_gateway_raises(self):
        rctx = _ctx(self.db, gateway=None)
        a = Assessment(True, 0.9, ["test"])
        with self.assertRaises(RuntimeError):
            deliver(_finding(), a, rctx)

    def test_no_owner_chats_raises(self):
        rctx = _ctx(self.db, gateway=_FakeGateway(owner_chats=set()))
        a = Assessment(True, 0.9, ["test"])
        with self.assertRaises(RuntimeError):
            deliver(_finding(), a, rctx)

    def test_daily_cap_blocks(self):
        gw = _FakeGateway(owner_chats={"telegram:123"})
        rctx = _ctx(self.db, gateway=gw, daily_delivery_cap=0)
        a = Assessment(True, 0.9, ["test"])
        with self.assertRaises(RuntimeError):
            deliver(_finding(), a, rctx)
        self.assertEqual(gw.sent, [])

    def test_unworthy_finding_rejected(self):
        gw = _FakeGateway(owner_chats={"telegram:123"})
        with self.assertRaises(ValueError):
            deliver(_finding(), Assessment(False, 0.1, ["nope"]), _ctx(self.db, gateway=gw))


class ExecuteJobTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        ensure_schema(self.db)

    def tearDown(self):
        self.db.close()

    def test_end_to_end_delivers_worthy_skips_rest(self):
        reg = _FakeRegistry(
            results=[
                {
                    "title": "Outlier AI is hiring coding experts in Nigeria today",
                    "url": "https://example.com/outlier-nigeria",
                    "snippet": (
                        "Outlier AI just launched a new coding-expert project open "
                        "to Nigeria. Apply this week: $25/hr, remote."
                    ),
                },
                {
                    "title": "Best gardening tips for spring planting season",
                    "url": "https://example.com/garden",
                    "snippet": (
                        "A complete guide to planting tomatoes and roses this "
                        "spring with soil advice for beginners everywhere."
                    ),
                },
            ],
            fetch_text="",
        )
        gw = _FakeGateway(owner_chats={"telegram:123"})
        job = ResearchJob(id="money-watch", topic="t", queries=["q1"], fetch_top=0)
        report = execute_job(job, _ctx(self.db, registry=reg, gateway=gw))
        self.assertEqual(report.findings, 2)
        self.assertEqual(report.delivered, 1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(report.errors, [])


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        ensure_schema(self.db)
        self.now = time.time()

    def tearDown(self):
        self.db.close()

    def _sched(self, **kw):
        rctx = _ctx(self.db, registry=_FakeRegistry(results=[]), gateway=_FakeGateway())
        return ResearchScheduler(self.db, rctx, tick_seconds=60, clock=lambda: self.now, **kw)

    def test_due_when_never_run(self):
        s = self._sched()
        s.add_job(ResearchJob(id="j", topic="t", queries=["q"], cadence_hours=24))
        self.assertEqual([j.id for j in s.due_jobs()], ["j"])

    def test_not_due_after_run(self):
        s = self._sched()
        s.add_job(ResearchJob(id="j", topic="t", queries=["q"], cadence_hours=24))
        reports = s.run_due()
        self.assertEqual(len(reports), 1)
        self.assertEqual(s.due_jobs(), [])

    def test_due_again_after_cadence(self):
        s = self._sched()
        s.add_job(ResearchJob(id="j", topic="t", queries=["q"], cadence_hours=1))
        s.run_due()
        self.now += 3700
        self.assertEqual([j.id for j in s.due_jobs()], ["j"])

    def test_disabled_job_never_due(self):
        s = self._sched()
        s.add_job(
            ResearchJob(id="j", topic="t", queries=["q"], cadence_hours=1, enabled=False)
        )
        self.assertEqual(s.due_jobs(), [])

    def test_failed_job_does_not_hot_loop(self):
        s = self._sched()
        s.db  # ensure schema'd
        rctx = _ctx(self.db, registry=_FailRegistry(), gateway=_FakeGateway())
        s2 = ResearchScheduler(self.db, rctx, tick_seconds=60, clock=lambda: self.now)
        s2.add_job(ResearchJob(id="j", topic="t", queries=["q"], cadence_hours=24))
        reports = s2.run_due()
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].errors)
        # run recorded despite failure -> not due again immediately
        self.assertEqual(s2.due_jobs(), [])

    def test_remove_job(self):
        s = self._sched()
        s.add_job(ResearchJob(id="j", topic="t", queries=["q"]))
        self.assertTrue(s.remove_job("j"))
        self.assertFalse(s.remove_job("j"))
        self.assertEqual(s.list_jobs(), [])

    def test_default_jobs_sane(self):
        jobs = default_jobs()
        self.assertGreaterEqual(len(jobs), 1)
        for job in jobs:
            self.assertTrue(job.queries)
            self.assertGreaterEqual(job.cadence_hours, 12.0)


if __name__ == "__main__":
    unittest.main()
