"""PASS #1 — self-improvement loop: real bugs, failing-first.

Each test below failed on the pre-fix tree (verified) and pins the fix:

1. approval-mode cycles must actually file an UpgradeQueue proposal
   (improvement.py omitted the required patch_plan/tests kwargs, so
   propose() raised TypeError on every cycle — swallowed by bare except).
2. an apply failure (dirty tree, model down, ...) must NOT be recorded
   as "regressed" — nothing regressed, and the 24h regression cooldown
   must not trip on an infrastructure failure.
3. CycleRecord.from_row must map the 0.0 no-score sentinel back to None
   exactly for the (action, status) pairs that can only have been written
   with None — history must not show a measured-looking 0.0 for cycles
   that never measured anything. Genuine measured 0.0 scores are kept.
4. forcing an unmeasurable dimension must skip honestly instead of
   proposing an edit scored with another dimension's number.
5. end-to-end approval flow: propose -> approve -> re-benchmark ->
   recorded improvement.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import improvement as improv_mod
from nomorals.agents.benchmark import DimensionScore
from nomorals.agents.improvement import ImprovementLoop
from nomorals.agents.upgrade_queue import UpgradeQueue
from nomorals.core.config import ImprovementSettings
from nomorals.storage.db import Database


def _migrated_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(*, mode: str = "approval", target: float = 0.8) -> SimpleNamespace:
    settings = SimpleNamespace(
        improvement=ImprovementSettings(mode=mode, target=target))
    return SimpleNamespace(settings=settings, db=_migrated_db(), router=None)


def _report(scores: dict[str, DimensionScore | None]) -> SimpleNamespace:
    return SimpleNamespace(scores=scores, provider="test")


def _scores(**kwargs) -> dict[str, DimensionScore]:
    out: dict[str, DimensionScore] = {}
    for name, score in kwargs.items():
        out[name] = DimensionScore(
            name, score, 0, 1,
            details=[{"pass": False, "task": "t1", "detail": "d1"}])
    return out


class _FakeProposal:
    id = "prop-1"
    edits = [{"path": "nomorals/agents/orchestrator.py"}]


class _FakeEvolver:
    def __init__(self, context):
        self.context = context

    def plan(self, instruction, focus=None):
        return _FakeProposal()


def _patch_cycle(report):
    p1 = mock.patch.object(improv_mod, "measurable", return_value=True)
    p2 = mock.patch.object(improv_mod, "run_benchmark", return_value=report)
    return p1, p2


class TestApprovalFilesUpgradeQueue(unittest.TestCase):
    def test_proposal_reaches_upgrade_queue(self):
        """Pre-fix: 0 queue proposals — propose() raised TypeError
        (missing patch_plan/tests), swallowed by bare except."""
        ctx = _ctx(mode="approval", target=0.8)
        p1, p2 = _patch_cycle(_report(_scores(reasoning=0.9, planning=0.3)))
        with p1, p2, mock.patch("nomorals.agents.evolution.EvolutionAgent",
                                _FakeEvolver):
            rec = ImprovementLoop(ctx).run_cycle()
        self.assertEqual(rec.action, "proposed")
        mine = [p for p in UpgradeQueue(ctx).list(status="proposed", limit=50)
                if p.get("source") == "improvement"]
        self.assertEqual(len(mine), 1)
        self.assertIn("planning", mine[0]["title"])
        self.assertTrue(mine[0].get("files"))


class TestApplyFailureIsNotRegression(unittest.TestCase):
    def test_apply_error_records_failed_and_no_cooldown(self):
        """Pre-fix: status came back "regressed" and _recently_regressed
        tripped the 24h cooldown — for an edit that never landed."""
        class BoomEvolver(_FakeEvolver):
            def apply(self, proposal_id):
                raise RuntimeError("working tree is dirty")

        ctx = _ctx(mode="autonomous", target=0.8)
        p1, p2 = _patch_cycle(_report(_scores(reasoning=0.9, planning=0.3)))
        loop = ImprovementLoop(ctx)
        with p1, p2, mock.patch("nomorals.agents.evolution.EvolutionAgent",
                                BoomEvolver):
            rec = loop.run_cycle()
        self.assertEqual(rec.action, "apply_failed")
        self.assertEqual(rec.status, "failed")
        self.assertFalse(loop._recently_regressed("planning"))


class TestFromRowSentinelScores(unittest.TestCase):
    def test_unmeasurable_skip_reads_back_none(self):
        """Pre-fix: history showed before_score 0.0 for a cycle that
        never measured anything."""
        ctx = _ctx(mode="off")
        rec = ImprovementLoop(ctx).run_cycle()
        self.assertEqual(rec.action, "off")
        back = ImprovementLoop(ctx).history(limit=1)[0]
        self.assertIsNone(back.before_score)
        self.assertIsNone(back.after_score)
        self.assertIsNone(back.delta)

    def test_pending_proposal_keeps_real_before_none_after(self):
        ctx = _ctx(mode="approval", target=0.8)
        p1, p2 = _patch_cycle(_report(_scores(reasoning=0.9, planning=0.3)))
        with p1, p2, mock.patch("nomorals.agents.evolution.EvolutionAgent",
                                _FakeEvolver), \
                mock.patch("nomorals.agents.upgrade_queue.UpgradeQueue"):
            ImprovementLoop(ctx).run_cycle()
        back = ImprovementLoop(ctx).pending_approvals()[0]
        self.assertEqual(back.before_score, 0.3)
        self.assertIsNone(back.after_score)
        self.assertIsNone(back.delta)

    def test_genuine_zero_score_is_preserved(self):
        """A dimension genuinely measured at 0.0 must stay 0.0, not None."""
        ctx = _ctx(mode="approval", target=0.0)
        p1, p2 = _patch_cycle(_report(_scores(reasoning=0.0, planning=0.0)))
        with p1, p2:
            rec = ImprovementLoop(ctx).run_cycle()
        self.assertEqual(rec.action, "meets_target")
        back = ImprovementLoop(ctx).history(limit=1)[0]
        self.assertEqual(back.before_score, 0.0)
        self.assertIsNotNone(back.before_score)


class TestForcedDimensionHonesty(unittest.TestCase):
    def test_forced_unmeasurable_dimension_skips(self):
        """Pre-fix: forced an edit for 'planning' scored with reasoning's
        0.5 — planning itself was unmeasurable."""
        ctx = _ctx(mode="approval", target=0.8)
        p1, p2 = _patch_cycle(
            _report(_scores(reasoning=0.5, planning=None)))
        with p1, p2, mock.patch("nomorals.agents.evolution.EvolutionAgent",
                                _FakeEvolver), \
                mock.patch("nomorals.agents.upgrade_queue.UpgradeQueue"):
            rec = ImprovementLoop(ctx).run_cycle(dimension="planning")
        self.assertEqual(rec.action, "unmeasurable")
        self.assertEqual(rec.dimension, "planning")
        self.assertEqual(rec.status, "skipped")


class TestEndToEndApprovalFlow(unittest.TestCase):
    def test_propose_then_approve_records_improvement(self):
        before = _report(_scores(reasoning=0.9, planning=0.3))
        after = _report(_scores(reasoning=0.9, planning=0.9))

        class ApplyEvolver(_FakeEvolver):
            def apply(self, proposal_id):
                return {"applied": True, "status": "applied"}

        ctx = _ctx(mode="approval", target=0.8)
        loop = ImprovementLoop(ctx)
        with mock.patch.object(improv_mod, "measurable",
                               return_value=True), \
             mock.patch.object(improv_mod, "run_benchmark",
                               side_effect=[before, after]), \
             mock.patch("nomorals.agents.evolution.EvolutionAgent",
                        ApplyEvolver), \
             mock.patch("nomorals.agents.upgrade_queue.UpgradeQueue"):
            rec = loop.run_cycle()
            self.assertEqual(rec.action, "proposed")
            out = loop.approve(rec.id)
        self.assertTrue(out["ok"])
        self.assertEqual(out["status"], "improved")
        self.assertAlmostEqual(out["delta"], 0.6)
        back = loop.history(limit=1)[0]
        self.assertEqual(back.before_score, 0.3)
        self.assertEqual(back.after_score, 0.9)


class TestEndToEndAutonomousFlow(unittest.TestCase):
    def test_autonomous_apply_records_improvement(self):
        """The autonomous path — benchmark -> propose -> apply (gate) ->
        re-benchmark -> record — completing with a real improvement."""
        before = _report(_scores(reasoning=0.9, planning=0.3))
        after = _report(_scores(reasoning=0.9, planning=0.9))

        class ApplyEvolver(_FakeEvolver):
            def apply(self, proposal_id):
                return {"applied": True, "status": "applied"}

        ctx = _ctx(mode="autonomous", target=0.8)
        with mock.patch.object(improv_mod, "measurable",
                               return_value=True), \
             mock.patch.object(improv_mod, "run_benchmark",
                               side_effect=[before, after]), \
             mock.patch("nomorals.agents.evolution.EvolutionAgent",
                        ApplyEvolver):
            rec = ImprovementLoop(ctx).run_cycle()
        self.assertEqual(rec.action, "applied")
        self.assertEqual(rec.status, "improved")
        self.assertAlmostEqual(rec.delta, 0.6)
        self.assertFalse(
            ImprovementLoop(ctx)._recently_regressed("planning"))
        back = ImprovementLoop(ctx).history(limit=1)[0]
        self.assertEqual((back.before_score, back.after_score),
                         (0.3, 0.9))


if __name__ == "__main__":
    unittest.main()
