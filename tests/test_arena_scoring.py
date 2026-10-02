"""Scoring for arena runs: record, aggregate, difficulty, coverage.

Covers nomorals/agents/arena/scoring.py:
- record_score id format, row contents, kind coercion, db=None safety
- overall_score weighting math, latency mapping, clamp, None cases
- category_scores aggregates incl. empty-db defaults
- difficulty_target thresholds incl. the runs>=3 guard
- coverage_weights starvation boost and empty-db defaults
"""
from __future__ import annotations

import re
import unittest

from nomorals.agents.arena import scoring
from nomorals.storage.db import Database


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _perfect(**over) -> dict:
    axes = {"research_usefulness": 1.0, "build_compiled": True,
            "tests_passed": True, "edit_precision": 1.0,
            "latency_s": 0.0}
    axes.update(over)
    return axes


def _zero() -> dict:
    return {"research_usefulness": 0.0, "build_compiled": False,
            "tests_passed": False, "edit_precision": 0.0,
            "latency_s": 600.0}


# ── record_score ──────────────────────────────────────────────────────────

class RecordScoreTests(unittest.TestCase):
    def test_returns_12_hex_id_and_writes_row(self):
        db = make_db()
        sid = scoring.record_score(db, topic="build a ring buffer",
                                   category="concurrency",
                                   kind="code", tests_passed=True,
                                   latency_s=12.5, notes="fast")
        self.assertRegex(sid, r"^[0-9a-f]{12}$")
        rows = db.query("SELECT * FROM arena_scores WHERE id = ?", (sid,))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["topic"], "build a ring buffer")
        self.assertEqual(row["category"], "concurrency")
        self.assertEqual(row["kind"], "code")
        self.assertEqual(row["tests_passed"], 1)
        self.assertAlmostEqual(row["latency_s"], 12.5)
        self.assertEqual(row["notes"], "fast")

    def test_optional_axes_default_to_null(self):
        db = make_db()
        sid = scoring.record_score(db, topic="t", category="c")
        row = db.query_one("SELECT * FROM arena_scores WHERE id = ?", (sid,))
        self.assertIsNone(row["research_usefulness"])
        self.assertIsNone(row["build_compiled"])
        self.assertIsNone(row["latency_s"])
        self.assertEqual(row["notes"], "")

    def test_kind_coercion(self):
        db = make_db()
        sid = scoring.record_score(db, topic="t", category="c", kind="bogus")
        row = db.query_one("SELECT kind FROM arena_scores WHERE id = ?", (sid,))
        self.assertEqual(row["kind"], "code")
        sid2 = scoring.record_score(db, topic="t", category="c",
                                    kind="research")
        row2 = db.query_one("SELECT kind FROM arena_scores WHERE id = ?",
                            (sid2,))
        self.assertEqual(row2["kind"], "research")

    def test_never_raises_on_none_db(self):
        self.assertEqual(scoring.record_score(None, topic="t",
                                              category="c"), "")


# ── overall_score ─────────────────────────────────────────────────────────

class OverallScoreTests(unittest.TestCase):
    def test_all_perfect_is_one(self):
        self.assertAlmostEqual(scoring.overall_score(_perfect()), 1.0)

    def test_all_zero_is_zero(self):
        self.assertAlmostEqual(scoring.overall_score(_zero()), 0.0)

    def test_weighting_math(self):
        # only research_usefulness present at 0.5 → renormalized to 0.5
        self.assertAlmostEqual(
            scoring.overall_score({"research_usefulness": 0.5}), 0.5)
        # 0.25*1 + 0.20*1 + 0.25*1 + 0.20*0 + 0.10*1, weights sum to 1.0
        axes = _perfect(edit_precision=0.0)
        self.assertAlmostEqual(scoring.overall_score(axes), 0.8)

    def test_bool_axes_coerced(self):
        self.assertAlmostEqual(
            scoring.overall_score({"build_compiled": True}), 1.0)
        self.assertAlmostEqual(
            scoring.overall_score({"tests_passed": 0}), 0.0)

    def test_latency_mapping(self):
        self.assertAlmostEqual(
            scoring.overall_score({"latency_s": 0}), 1.0)
        self.assertAlmostEqual(
            scoring.overall_score({"latency_s": 300}), 0.5)
        self.assertAlmostEqual(
            scoring.overall_score({"latency_s": 600}), 0.0)
        self.assertAlmostEqual(
            scoring.overall_score({"latency_s": 10_000}), 0.0)
        # "latency" alias works too
        self.assertAlmostEqual(
            scoring.overall_score({"latency": 300}), 0.5)

    def test_clamps_inputs(self):
        self.assertAlmostEqual(
            scoring.overall_score({"research_usefulness": 99.0}), 1.0)
        self.assertAlmostEqual(
            scoring.overall_score({"edit_precision": -3.0}), 0.0)

    def test_none_when_no_axes(self):
        self.assertIsNone(scoring.overall_score({}))
        self.assertIsNone(scoring.overall_score(None))
        self.assertIsNone(scoring.overall_score(
            {"research_usefulness": None, "latency_s": None}))


# ── category_scores ───────────────────────────────────────────────────────

class CategoryScoreTests(unittest.TestCase):
    def test_aggregates(self):
        db = make_db()
        scoring.record_score(db, topic="a", category="web",
                             research_usefulness=1.0, latency_s=60.0)
        scoring.record_score(db, topic="b", category="web",
                             research_usefulness=0.0, latency_s=120.0)
        scoring.record_score(db, topic="c", category="ai",
                             build_compiled=True)
        out = scoring.category_scores(db)
        self.assertEqual(out["web"]["runs"], 2)
        expected = (scoring.overall_score({"research_usefulness": 1.0,
                                           "latency_s": 60.0})
                    + scoring.overall_score({"research_usefulness": 0.0,
                                             "latency_s": 120.0})) / 2
        self.assertAlmostEqual(out["web"]["avg"], expected)
        self.assertAlmostEqual(out["web"]["avg_latency"], 90.0)
        self.assertEqual(out["ai"]["runs"], 1)
        self.assertAlmostEqual(out["ai"]["avg"], 1.0)
        self.assertIsNone(out["ai"]["avg_latency"])

    def test_empty_db_gives_empty_dict(self):
        self.assertEqual(scoring.category_scores(make_db()), {})

    def test_never_raises_on_none_db(self):
        self.assertEqual(scoring.category_scores(None), {})


# ── difficulty_target ─────────────────────────────────────────────────────

class DifficultyTargetTests(unittest.TestCase):
    def test_aces_push_harder(self):
        db = make_db()
        for i in range(3):
            scoring.record_score(db, topic=f"t{i}", category="web",
                                 **_perfect())
        self.assertEqual(scoring.difficulty_target(db, "web"), 3)

    def test_struggles_ease_off(self):
        db = make_db()
        for i in range(4):
            scoring.record_score(db, topic=f"t{i}", category="ai",
                                 **_zero())
        self.assertEqual(scoring.difficulty_target(db, "ai"), 1)

    def test_runs_below_three_guard(self):
        db = make_db()
        for i in range(2):
            scoring.record_score(db, topic=f"t{i}", category="web",
                                 **_perfect())
        self.assertEqual(scoring.difficulty_target(db, "web"), 2)

    def test_mid_band_is_two(self):
        db = make_db()
        for i in range(3):
            scoring.record_score(db, topic=f"t{i}", category="web",
                                 research_usefulness=0.5)
        self.assertEqual(scoring.difficulty_target(db, "web"), 2)

    def test_unknown_category_and_none_db(self):
        db = make_db()
        scoring.record_score(db, topic="t", category="web", **_perfect())
        self.assertEqual(scoring.difficulty_target(db, "nope"), 2)
        self.assertEqual(scoring.difficulty_target(None, "web"), 2)


# ── coverage_weights ──────────────────────────────────────────────────────

class CoverageWeightTests(unittest.TestCase):
    def test_starved_categories_boosted(self):
        from nomorals.agents.arena.topics import all_categories

        db = make_db()
        for i in range(4):
            scoring.record_score(db, topic=f"t{i}", category="web",
                                 tests_passed=True)
        weights = scoring.coverage_weights(db)
        self.assertAlmostEqual(weights["web"], 1.0)
        self.assertAlmostEqual(weights["ai"], 3.0)
        for cat in all_categories():
            self.assertIn(cat, weights)
            self.assertGreaterEqual(weights[cat], 1.0)
            self.assertLessEqual(weights[cat], 3.0)

    def test_partial_coverage_scales(self):
        db = make_db()
        for i in range(4):
            scoring.record_score(db, topic=f"w{i}", category="web",
                                 tests_passed=True)
        for i in range(2):
            scoring.record_score(db, topic=f"d{i}", category="data",
                                 tests_passed=True)
        weights = scoring.coverage_weights(db)
        self.assertAlmostEqual(weights["web"], 1.0)
        self.assertAlmostEqual(weights["data"], 2.0)
        self.assertAlmostEqual(weights["ai"], 3.0)

    def test_empty_db_all_one(self):
        from nomorals.agents.arena.topics import all_categories

        weights = scoring.coverage_weights(make_db())
        self.assertTrue(weights)
        for cat in all_categories():
            self.assertAlmostEqual(weights[cat], 1.0)

    def test_none_db_all_one(self):
        from nomorals.agents.arena.topics import all_categories

        weights = scoring.coverage_weights(None)
        for cat in all_categories():
            self.assertAlmostEqual(weights[cat], 1.0)


if __name__ == "__main__":
    unittest.main()
