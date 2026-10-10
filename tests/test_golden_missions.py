"""Tests for the golden end-to-end mission drills.

Short versions run here (seconds, tiny fixtures).  The ``--long`` drills
are real multi-minute runs, exercised manually via ``nm golden run <key>
--long`` — the repair paths they exercise are covered deterministically
below by driving the repair functions directly.
"""

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nomorals.llm.benchmarks import BenchmarkDB
from nomorals.missions.golden import (
    GOLDEN_MISSIONS,
    GoldenContext,
    GoldenRunner,
    _audit_configs,
    _remediate,
    _repair_calc,
    _scaffold,
    _verify_clean,
    _verify_tests,
    list_golden_missions,
)
from nomorals.missions.mission import MissionStatus, MissionStore
from nomorals.storage.db import Database


def make_db(test):
    tmp = tempfile.mkdtemp(prefix="golden-test-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(str(Path(tmp) / "golden.db"))
    db.migrate()
    test.addCleanup(db.close)
    return db, tmp


def benchmark_rows(db, model_id):
    return db.query(
        "SELECT * FROM model_benchmarks WHERE model_id = ?", (model_id,))


class TestGoldenMissions(unittest.TestCase):
    def test_registry_has_five_missions(self):
        missions = list_golden_missions()
        self.assertEqual(
            {m["key"] for m in missions},
            {"research_write_verify", "build_test_fix", "audit_remediate_rescan",
             "crash_no_dup", "saga_undo"},
        )
        for m in missions:
            self.assertTrue(m["steps"], m["key"])

    def test_research_write_verify_short(self):
        db, tmp = make_db(self)
        runner = GoldenRunner(db, workdir_root=tmp)
        result = runner.run("research_write_verify")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertEqual([s["step"] for s in result.steps],
                         ["collect", "draft"])
        self.assertTrue(all(s["ok"] for s in result.steps))
        rows = benchmark_rows(db, "golden:research_write_verify")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["success"], 1)
        self.assertEqual(rows[0]["source"], "golden")

    def test_build_test_fix_short(self):
        db, tmp = make_db(self)
        runner = GoldenRunner(db, workdir_root=tmp)
        result = runner.run("build_test_fix")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.status, MissionStatus.DONE)
        self.assertTrue(all(s["ok"] for s in result.steps))
        rows = benchmark_rows(db, "golden:build_test_fix")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_kind"], "build_test_fix")

    def test_audit_remediate_rescan_short(self):
        db, tmp = make_db(self)
        runner = GoldenRunner(db, workdir_root=tmp)
        result = runner.run("audit_remediate_rescan")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.status, MissionStatus.DONE)
        by_name = {s["step"]: s for s in result.steps}
        self.assertEqual(
            [s for s in by_name], ["generate", "audit", "remediate", "rescan"])
        self.assertIn("0 issues", by_name["rescan"]["detail"])
        rows = benchmark_rows(db, "golden:audit_remediate_rescan")
        self.assertEqual(len(rows), 1)

    def test_unknown_key_raises(self):
        db, tmp = make_db(self)
        with self.assertRaises(KeyError):
            GoldenRunner(db, workdir_root=tmp).run("nope")

    def test_kill_then_resume_completes(self):
        db, tmp = make_db(self)
        runner = GoldenRunner(db, workdir_root=tmp)

        # Slow the first step so the kill lands mid-mission.
        mission = GOLDEN_MISSIONS["build_test_fix"]
        step0 = mission.steps[0]
        original_run = step0.run

        def slow_run(ctx):
            time.sleep(5)
            return original_run(ctx)

        step0.run = slow_run
        box: dict = {}
        try:
            thread = threading.Thread(
                target=lambda: box.setdefault("result", runner.run("build_test_fix")),
                daemon=True,
            )
            thread.start()
            time.sleep(1.0)
            runner.kill("test kill")
            thread.join(timeout=30)
        finally:
            step0.run = original_run

        self.assertFalse(thread.is_alive(), "golden run did not stop after kill")
        killed = box["result"]
        self.assertEqual(killed.status, MissionStatus.PAUSED)
        completed = MissionStore(db).get(killed.mission_id).state["golden"]["completed"]
        self.assertEqual(completed, ["scaffold"])
        # The killed run still recorded its (failed) benchmark row.
        self.assertEqual(len(benchmark_rows(db, "golden:build_test_fix")), 1)

        # Resume on a fresh runner: skips the completed step, finishes clean.
        resumed = GoldenRunner(db, workdir_root=tmp).resume(killed.mission_id)
        self.assertTrue(resumed.ok, resumed.error)
        self.assertEqual(resumed.status, MissionStatus.DONE)
        self.assertEqual(
            MissionStore(db).get(killed.mission_id).state["golden"]["completed"],
            ["scaffold", "test"],
        )
        rows = benchmark_rows(db, "golden:build_test_fix")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["success"], 1)

    def test_resume_terminal_mission_returns_status(self):
        db, tmp = make_db(self)
        runner = GoldenRunner(db, workdir_root=tmp)
        done = runner.run("research_write_verify")
        again = runner.resume(done.mission_id)
        self.assertEqual(again.status, MissionStatus.DONE)
        self.assertTrue(again.ok)


class TestGoldenRepairPaths(unittest.TestCase):
    """The repair/verify/guard cycle, driven deterministically without sleeps."""

    def test_build_repair_fixes_deterministic_bug(self):
        tmp = tempfile.mkdtemp(prefix="golden-repair-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ctx = GoldenContext(workdir=Path(tmp), long=False)

        # Scaffold the *buggy* variant by hand, like the long drill does.
        from nomorals.missions.golden import _BUGGY_LINE, _CALC_SRC, _CALC_TEST
        src = _CALC_SRC.replace("    return a + b\n", _BUGGY_LINE, 1)
        (ctx.workdir / "calc.py").write_text(src)
        (ctx.workdir / "test_calc.py").write_text(_CALC_TEST)

        from nomorals.missions.golden import _run_tests
        failing = _run_tests(ctx)
        ok, _detail = _verify_tests(failing)
        self.assertFalse(ok, "buggy scaffold must fail its tests")

        repaired = _repair_calc(failing, ctx)
        self.assertTrue(repaired.get("repaired"))
        ok, detail = _verify_tests(repaired)
        self.assertTrue(ok, f"repair must make tests green: {detail}")

    def test_rescan_guard_fails_until_remediated(self):
        tmp = tempfile.mkdtemp(prefix="golden-audit-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ctx = GoldenContext(workdir=Path(tmp), long=False)

        from nomorals.missions.golden import _generate_configs
        _generate_configs(ctx)
        before = _audit_configs(ctx)
        self.assertGreater(before["count"], 0)

        # Guard fires while issues remain.
        out = {**before, "confdir": str(ctx.workdir / "configs")}
        ok, _detail = _verify_clean(out)
        self.assertFalse(ok)

        fixed = _remediate(ctx)
        self.assertGreater(fixed["fixed"], 0)
        after = _audit_configs(ctx)
        self.assertEqual(after["count"], 0)
        out = {**after, "confdir": str(ctx.workdir / "configs")}
        ok, detail = _verify_clean(out)
        self.assertTrue(ok, detail)

    def test_research_repair_recovers_lost_fact(self):
        tmp = tempfile.mkdtemp(prefix="golden-research-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ctx = GoldenContext(workdir=Path(tmp), long=False)

        from nomorals.missions.golden import (
            _collect, _draft, _repair_draft, _verify_draft,
        )
        collected = _collect(ctx)
        ctx.params["facts"] = collected["facts"][:-1]  # simulate the lost fact
        drafted = _draft(ctx)
        drafted["facts"] = collected["facts"]
        ok, _detail = _verify_draft(drafted)
        self.assertFalse(ok, "draft missing a fact must fail verification")

        repaired = _repair_draft(drafted, ctx)
        self.assertTrue(repaired.get("repaired"))
        ok, detail = _verify_draft(repaired)
        self.assertTrue(ok, f"repair must restore the draft: {detail}")


if __name__ == "__main__":
    unittest.main()
