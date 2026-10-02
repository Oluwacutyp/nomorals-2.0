"""Wave G2 acceptance tests: research-loop rate limiting + usefulness signals.

1. Per-cycle proposal cap — the loop files at most
   ``NM_RESEARCH_LOOP_MAX_PROPOSALS`` (default 5) new queue proposals per
   tick; overflow tickets are counted as ``proposals_capped`` and left in
   the digest, never silently dropped from the record.
2. Cross-cycle dedupe — a finding already proposed by the loop inside the
   dedupe window (title identity, near-copy titles, or same target files
   with related wording) files no second ticket; ``proposals_deduped``
   counts the drops, in-cycle and across cycles.
3. Notification cap — one digest publish per cycle no matter how many
   topic briefs came back, and the digest carries a one-line
   usefulness-signals summary.
4. Usefulness signals — raw counts only (no synthesized score): proposed,
   approved, denied, ignored (undecided past the ignore window), pending,
   applied, tests_passed/tests_failed from ``record_implemented``
   outcomes. Exposed via ``status()`` / ``nm research-loop status`` with
   honest zeros when there is no data.
5. Opt-in untouched — a disabled loop still never runs the pipeline
   (extends the F2 loop-gated-off tests).
"""
from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import research_loop as rl
from nomorals.agents.features import FeatureRegistry
from nomorals.agents.research_digest import ResearchClaim, ResearchDigest
from nomorals.agents.upgrade_queue import UpgradePipeline, UpgradeQueue
from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None, proactive=True):
    db = db or _db()
    partner = SimpleNamespace(proactive_enabled=proactive, quiet_start=22,
                              quiet_end=8)
    settings = SimpleNamespace(partner=partner)
    return SimpleNamespace(db=db, settings=settings)


def _enable(ctx):
    FeatureRegistry(ctx.db).set("research", True)


def _claim(cid, text, confidence=0.85, domain="systems", query="q"):
    return ResearchClaim(
        id=cid, claim=text, domain=domain, angle="core",
        sources=[{"url": "https://example.com/x", "title": "X",
                  "trust": 0.8}],
        confidence=confidence, fetched_at=1.0, query=query)


def _ticket(title, files=(), confidence=0.9):
    return {"title": title, "rationale": "x" * 40,
            "suggested_files": list(files), "confidence": confidence,
            "expected_tests": ["test_a", "test_b"],
            "acceptance_criteria": ["a", "b", "c"]}


def _cycle_result(tickets, brief="cycle brief"):
    return {"report": {"findings": [{"f": 1}]}, "claims": [],
            "tickets": tickets, "brief": brief}


def _run_loop(ctx, tickets, *, max_topics=1, max_proposals=None,
              brief="cycle brief"):
    """Run one gated-on tick with a mocked pipeline/notifier."""
    fake_pipeline = mock.Mock()
    fake_pipeline.run.return_value = _cycle_result(tickets, brief=brief)
    fake_notifier = mock.Mock()
    fake_notifier.publish.return_value = {"delivered": True,
                                          "deduped": False, "id": "n1"}
    loop_kwargs = {"max_topics": max_topics}
    if max_proposals is not None:
        loop_kwargs["max_proposals"] = max_proposals
    with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                     fake_pipeline),
          mock.patch("nomorals.agents.research_loop.Notifier",
                     return_value=fake_notifier),
          mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                     return_value=False),
          mock.patch.dict(os.environ, {}, clear=False)):
        out = rl.ResearchLoop(ctx, **loop_kwargs).tick()
    return out, fake_pipeline, fake_notifier


# ── per-cycle proposal cap ─────────────────────────────────────────────────

class ProposalCapTests(unittest.TestCase):
    def test_cap_enforced_per_cycle(self):
        ctx = _ctx()
        _enable(ctx)
        titles = [
            "adopt vector-clock conflict resolution for the sync engine",
            "cache compiled regexes in the router hot path",
            "add dark mode toggle to the settings page",
            "shard the session store by tenant id",
            "backfill missing created_at indexes on the events table",
        ]
        tickets = [_ticket(t, files=[f"nomorals/agents/mod{i}.py"])
                   for i, t in enumerate(titles)]
        with mock.patch.object(UpgradePipeline, "propose_from_ticket",
                               side_effect=[f"upg-{i}" for i in range(5)]
                               ) as propose:
            out, _, _ = _run_loop(ctx, tickets, max_proposals=2)
        self.assertEqual(out["proposals_created"], 2)
        self.assertEqual(out["proposals_capped"], 3)
        self.assertEqual(propose.call_count, 2)
        row = ctx.db.query_one(
            "SELECT proposals_created, proposals_capped, proposals_deduped "
            "FROM research_loop_runs")
        self.assertEqual(row["proposals_created"], 2)
        self.assertEqual(row["proposals_capped"], 3)
        self.assertEqual(row["proposals_deduped"], 0)

    def test_env_overrides_cap(self):
        self.assertEqual(rl.research_max_proposals(), 5)
        with mock.patch.dict(os.environ,
                             {"NM_RESEARCH_LOOP_MAX_PROPOSALS": "2"}):
            self.assertEqual(rl.research_max_proposals(), 2)
        with mock.patch.dict(os.environ,
                             {"NM_RESEARCH_LOOP_MAX_PROPOSALS": "junk"}):
            self.assertEqual(rl.research_max_proposals(), 5)
        with mock.patch.dict(os.environ,
                             {"NM_RESEARCH_LOOP_MAX_PROPOSALS": "0"}):
            self.assertEqual(rl.research_max_proposals(), 1)
        with mock.patch.dict(os.environ,
                             {"NM_RESEARCH_LOOP_MAX_PROPOSALS": "999"}):
            self.assertEqual(rl.research_max_proposals(), 50)

    def test_no_cap_pressure_no_capped_count(self):
        ctx = _ctx()
        _enable(ctx)
        tickets = [_ticket("adopt vector-clock conflict resolution"),
                   _ticket("cache compiled regexes in the router hot path")]
        with mock.patch.object(UpgradePipeline, "propose_from_ticket",
                               side_effect=["upg-a", "upg-b"]):
            out, _, _ = _run_loop(ctx, tickets, max_proposals=5)
        self.assertEqual(out["proposals_created"], 2)
        self.assertEqual(out["proposals_capped"], 0)


# ── cross-cycle dedupe ─────────────────────────────────────────────────────

class DedupeTests(unittest.TestCase):
    def _file_proposal(self, ctx, title, files):
        return UpgradeQueue(ctx).propose(
            title=title, rationale="rationale " * 10, patch_plan={},
            files=list(files), tests=[], claim_ids=[],
            source="research_loop")

    def test_same_finding_twice_one_proposal(self):
        ctx = _ctx()
        _enable(ctx)
        title = "adopt vector-clock conflict resolution for the sync engine"
        files = ["nomorals/agents/sync.py"]
        self._file_proposal(ctx, title, files)
        with mock.patch.object(UpgradePipeline, "propose_from_ticket",
                               return_value="upg-new") as propose:
            out, _, _ = _run_loop(ctx, [_ticket(title, files=files)],
                                  max_proposals=5)
        self.assertEqual(propose.call_count, 0)
        self.assertEqual(out["proposals_created"], 0)
        self.assertEqual(out["proposals_deduped"], 1)

    def test_in_cycle_dedupe(self):
        ctx = _ctx()
        _enable(ctx)
        title = "cache compiled regexes in the router hot path"
        tickets = [_ticket(title, files=["nomorals/agents/router.py"]),
                   _ticket(title, files=["nomorals/agents/router.py"])]
        with mock.patch.object(UpgradePipeline, "propose_from_ticket",
                               return_value="upg-1") as propose:
            out, _, _ = _run_loop(ctx, tickets, max_proposals=5)
        self.assertEqual(propose.call_count, 1)
        self.assertEqual(out["proposals_created"], 1)
        self.assertEqual(out["proposals_deduped"], 1)

    def test_similar_title_same_files_is_duplicate(self):
        ctx = _ctx()
        _enable(ctx)
        self._file_proposal(
            ctx, "use vector clocks to resolve sync engine conflicts",
            ["nomorals/agents/sync.py"])
        dup = rl.is_duplicate_proposal(
            ctx, "resolve sync engine conflicts with vector clocks",
            ["nomorals/agents/sync.py"])
        self.assertTrue(dup)

    def test_different_finding_not_duplicate(self):
        ctx = _ctx()
        _enable(ctx)
        self._file_proposal(ctx, "adopt vector-clock conflict resolution",
                           ["nomorals/agents/sync.py"])
        self.assertFalse(rl.is_duplicate_proposal(
            ctx, "add dark mode toggle to the settings page",
            ["nomorals/ui/settings.py"]))

    def test_other_sources_do_not_trigger_dedupe(self):
        ctx = _ctx()
        _enable(ctx)
        UpgradeQueue(ctx).propose(
            title="adopt vector-clock conflict resolution for the sync engine",
            rationale="rationale " * 10, patch_plan={},
            files=["nomorals/agents/sync.py"], tests=[], claim_ids=[],
            source="manual")
        self.assertFalse(rl.is_duplicate_proposal(
            ctx, "adopt vector-clock conflict resolution for the sync engine",
            ["nomorals/agents/sync.py"]))

    def test_dedupe_looks_across_cycles(self):
        # cycle 1 files a real proposal; cycle 2 re-surfaces the same
        # finding (different topic, same ticket) and must dedupe it.
        ctx = _ctx()
        _enable(ctx)
        ticket = ResearchDigest.upgrade_ticket(
            _claim("c1", "Redis is the fastest in-memory cache for session "
                         "storage in this stack", confidence=0.9))
        out1, _, _ = _run_loop(ctx, [ticket], max_proposals=5)
        self.assertEqual(out1["proposals_created"], 1)
        self.assertEqual(out1["proposals_deduped"], 0)
        out2, _, _ = _run_loop(ctx, [ticket], max_proposals=5)
        self.assertEqual(out2["proposals_created"], 0)
        self.assertEqual(out2["proposals_deduped"], 1)


# ── notification cap ───────────────────────────────────────────────────────

class NotifyCapTests(unittest.TestCase):
    def test_one_digest_per_cycle_with_many_briefs(self):
        ctx = _ctx()
        _enable(ctx)
        fake_pipeline = mock.Mock()
        fake_pipeline.run.side_effect = [
            _cycle_result([_ticket(title)], brief=f"brief {i}")
            for i, title in enumerate([
                "adopt vector-clock conflict resolution for the sync engine",
                "cache compiled regexes in the router hot path",
                "add dark mode toggle to the settings page",
            ])]
        fake_notifier = mock.Mock()
        fake_notifier.publish.return_value = {"delivered": True,
                                              "deduped": False, "id": "n1"}
        with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                         fake_pipeline),
              mock.patch("nomorals.agents.research_loop.Notifier",
                         return_value=fake_notifier),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False),
              mock.patch.object(UpgradePipeline, "propose_from_ticket",
                                side_effect=["upg-1", "upg-2", "upg-3"])):
            out = rl.ResearchLoop(ctx, max_topics=3,
                                  max_proposals=5).tick()
        self.assertTrue(out["ok"])
        # exactly one digest publish, not one per finding/brief
        self.assertEqual(fake_notifier.publish.call_count, 1)
        self.assertTrue(out["notified"])
        _kind, title, body = fake_notifier.publish.call_args[0][:3]
        self.assertIn("brief 1", body)
        self.assertIn("brief 2", body)
        # one-line signals summary rides along, no extra noise
        self.assertIn("signals:", body)
        self.assertIn("proposed", body)


# ── usefulness signals ─────────────────────────────────────────────────────

class UsefulnessSignalsTests(unittest.TestCase):
    def _propose(self, ctx, title, files):
        return UpgradeQueue(ctx).propose(
            title=title, rationale="rationale " * 10, patch_plan={},
            files=list(files), tests=[], claim_ids=[],
            source="research_loop")

    def test_honest_zeros_when_no_data(self):
        ctx = _ctx()
        sig = rl.usefulness_signals(ctx)
        self.assertEqual(sig, {"proposed": 0, "approved": 0, "denied": 0,
                               "ignored": 0, "pending": 0, "applied": 0,
                               "tests_passed": 0, "tests_failed": 0})
        self.assertIn("0 proposed", rl.signals_summary(sig))

    def test_counts_and_pass_rate(self):
        ctx = _ctx()
        q = UpgradeQueue(ctx)
        approved = self._propose(ctx, "add connection pooling to the "
                                 "postgres client", ["nomorals/storage/db.py"])
        denied = self._propose(ctx, "rewrite the command line interface "
                               "in rust", ["nomorals/cli.py"])
        ignored = self._propose(ctx, "migrate the session cache from "
                                "memory to redis",
                                ["nomorals/agents/session.py"])
        pending = self._propose(ctx, "add prometheus metrics to the "
                                "scheduler",
                                ["nomorals/agents/scheduler.py"])
        passed = self._propose(ctx, "backfill created_at indexes on the "
                               "events table",
                               ["nomorals/storage/migrations.py"])
        failed = self._propose(ctx, "lazy-load the embedding model on "
                               "first use",
                               ["nomorals/memory/embeddings.py"])
        other = q.propose(title="not from the loop at all",
                          rationale="rationale " * 10, patch_plan={},
                          files=[], tests=[], claim_ids=[], source="manual")
        q.approve(approved, by="owner")
        q.deny(denied, "not worth it", by="owner")
        q.approve(passed, by="owner")
        q.record_implemented(passed, {"ok": True, "tests": "3 passed"})
        q.approve(failed, by="owner")
        q.record_implemented(failed, {"ok": False, "error": "boom"})
        # backdate the "ignored" one past the ignore window
        ctx.db.execute(
            "UPDATE upgrade_proposals SET created_at = ? WHERE id = ?",
            (time.time() - 8 * 86400.0, ignored))
        ctx.db.execute(
            "UPDATE upgrade_proposals SET created_at = ? WHERE id = ?",
            (time.time(), pending))
        self.assertTrue(other)  # other-source proposal exists, must be excluded

        sig = rl.usefulness_signals(ctx)
        self.assertEqual(sig["proposed"], 6)
        self.assertEqual(sig["approved"], 3)   # approved + 2 implemented
        self.assertEqual(sig["denied"], 1)
        self.assertEqual(sig["ignored"], 1)
        self.assertEqual(sig["pending"], 1)
        self.assertEqual(sig["applied"], 2)
        self.assertEqual(sig["tests_passed"], 1)
        self.assertEqual(sig["tests_failed"], 1)

        summary = rl.signals_summary(sig)
        self.assertEqual(summary, "6 proposed · 3 approved · 1 denied · "
                                 "1 ignored · applied 1/2 tests passed")

    def test_status_exposes_signals(self):
        ctx = _ctx()
        _enable(ctx)
        pid = self._propose(ctx, "a loop proposal for status checks",
                            ["nomorals/agents/research_loop.py"])
        UpgradeQueue(ctx).approve(pid, by="owner")
        with mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False):
            data = rl.status(ctx)
        self.assertIn("signals", data)
        self.assertEqual(data["signals"]["proposed"], 1)
        self.assertEqual(data["signals"]["approved"], 1)

    def test_run_record_snapshots_signals(self):
        ctx = _ctx()
        _enable(ctx)
        pid = self._propose(ctx, "a loop proposal filed before the run",
                            ["nomorals/agents/research_loop.py"])
        UpgradeQueue(ctx).deny(pid, "nope", by="owner")
        out, _, _ = _run_loop(ctx, [_ticket("fresh cycle ticket about "
                                           "scheduler jitter")],
                              max_proposals=5)
        self.assertIn("signals", out)
        self.assertEqual(out["signals"]["proposed"], 1)
        self.assertEqual(out["signals"]["denied"], 1)
        row = rl.last_run(ctx)
        self.assertEqual(row["signals"]["proposed"], 1)
        self.assertEqual(row["signals"]["denied"], 1)


# ── opt-in stays opt-in (extends the F2 loop-gated-off tests) ───────────────

class DisabledLoopTests(unittest.TestCase):
    def test_disabled_loop_never_runs_pipeline(self):
        # feature flag "research" defaults to off — the opt-in loop stays
        # off unless the owner explicitly enables it.
        ctx = _ctx()
        fake_pipeline = mock.Mock()
        fake_notifier = mock.Mock()
        with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                         fake_pipeline),
              mock.patch("nomorals.agents.upgrade_queue.UpgradePipeline",
                         mock.Mock()),
              mock.patch("nomorals.agents.research_loop.Notifier",
                         return_value=fake_notifier),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False)):
            out = rl.ResearchLoop(ctx).tick()
        self.assertFalse(out["ok"])
        self.assertIn("research", out["skipped_reason"])
        fake_pipeline.run.assert_not_called()
        fake_notifier.publish.assert_not_called()
        self.assertEqual(out["proposals_created"], 0)
        self.assertEqual(out["proposals_deduped"], 0)
        self.assertEqual(out["proposals_capped"], 0)

    def test_run_topic_disabled_defers(self):
        ctx = _ctx()
        fake_pipeline = mock.Mock()
        with (mock.patch("nomorals.agents.research_digest.ResearchPipeline",
                         fake_pipeline),
              mock.patch("nomorals.agents.research_loop._in_quiet_hours_now",
                         return_value=False)):
            out = rl.run_topic(ctx, "anything at all")
        self.assertTrue(out.get("deferred"))
        fake_pipeline.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
