"""Regression: ImprovementLoop.run_cycle crashed comparing DimensionScore objects.

The cognitive loop's improvement stage reported::

    improvement:
      error: '<' not supported between instances of 'DimensionScore' and 'DimensionScore'

Root cause: run_cycle built ``measured`` from ``report.scores`` values
(``DimensionScore`` dataclass instances, which define no ordering) and then
ran ``min(measured, key=lambda k: measured[k])`` — ``min`` compares the
KEYS with ``<``, i.e. ``DimensionScore < DimensionScore`` -> TypeError.
Downstream code also treated ``before`` as a float (``before >= target``,
``f"{before:.2f}"``, ``delta = after - before``), so the dict must hold
numeric scores, not DimensionScore objects.

These tests drive the real run_cycle with a stubbed benchmark report made
of real DimensionScore objects.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import improvement as improv_mod
from nomorals.agents.benchmark import DimensionScore
from nomorals.agents.improvement import ImprovementLoop
from nomorals.core.config import ImprovementSettings
from nomorals.storage.db import Database


def _migrated_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(*, mode: str = "approval", target: float = 0.8,
         db: Database | None = None) -> SimpleNamespace:
    settings = SimpleNamespace(
        improvement=ImprovementSettings(mode=mode, target=target))
    return SimpleNamespace(settings=settings, db=db or _migrated_db())


def _report(scores: dict[str, DimensionScore]) -> SimpleNamespace:
    return SimpleNamespace(scores=scores, provider="test")


def _scores(**kwargs: float | None) -> dict[str, DimensionScore]:
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


class _FakeQueue:
    def __init__(self, context):
        self.context = context
        self.proposed = []

    def propose(self, **kwargs):
        self.proposed.append(kwargs)


class TestWeakestDimensionSelection(unittest.TestCase):
    def _patch_benchmark(self, report):
        p1 = mock.patch.object(improv_mod, "measurable", return_value=True)
        p2 = mock.patch.object(improv_mod, "run_benchmark",
                               return_value=report)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_meets_target_does_not_compare_dimension_scores(self):
        """Pre-fix: TypeError '<' not supported between DimensionScore."""
        self._patch_benchmark(_report(_scores(reasoning=0.9, planning=0.5)))
        ctx = _ctx(mode="approval", target=0.4)
        rec = ImprovementLoop(ctx).run_cycle()
        self.assertEqual(rec.action, "meets_target")
        self.assertEqual(rec.dimension, "planning")
        self.assertIsInstance(rec.before_score, float)
        self.assertEqual(rec.before_score, 0.5)

    def test_approval_path_before_is_float_not_dimension_score(self):
        self._patch_benchmark(_report(_scores(reasoning=0.9, planning=0.3)))
        ctx = _ctx(mode="approval", target=0.8)
        with mock.patch("nomorals.agents.evolution.EvolutionAgent",
                        _FakeEvolver), \
                mock.patch("nomorals.agents.upgrade_queue.UpgradeQueue",
                           _FakeQueue):
            rec = ImprovementLoop(ctx).run_cycle()
        self.assertEqual(rec.action, "proposed")
        self.assertEqual(rec.dimension, "planning")
        self.assertIsInstance(rec.before_score, float)
        self.assertEqual(rec.before_score, 0.3)
        self.assertIn("0.30", rec.rationale)

    def test_unmeasurable_none_dimensions_are_filtered(self):
        self._patch_benchmark(
            _report(_scores(reasoning=None, planning=0.5)))
        ctx = _ctx(mode="approval", target=0.8)
        with mock.patch("nomorals.agents.evolution.EvolutionAgent",
                        _FakeEvolver), \
                mock.patch("nomorals.agents.upgrade_queue.UpgradeQueue",
                           _FakeQueue):
            rec = ImprovementLoop(ctx).run_cycle()
        # None (unmeasurable) must not be picked as "weakest".
        self.assertEqual(rec.dimension, "planning")
        self.assertEqual(rec.before_score, 0.5)

    def test_all_unmeasurable_skips_honestly(self):
        self._patch_benchmark(
            _report(_scores(reasoning=None, planning=None)))
        ctx = _ctx(mode="approval", target=0.8)
        rec = ImprovementLoop(ctx).run_cycle()
        self.assertEqual(rec.action, "unmeasurable")
        self.assertEqual(rec.status, "skipped")


class TestImprovementSettingsDefaults(unittest.TestCase):
    def test_tolerance_and_max_cycles_exist(self):
        # run_cycle (autonomous path) and tick() read these; a missing
        # attribute would raise AttributeError mid-cycle.
        settings = ImprovementSettings()
        self.assertIsInstance(settings.tolerance, float)
        self.assertIsInstance(settings.max_cycles, int)
        self.assertGreater(settings.max_cycles, 0)


if __name__ == "__main__":
    unittest.main()
