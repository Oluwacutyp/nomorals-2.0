"""The K3 scoreboard: an honest, end-to-end benchmark harness.

Covers nomorals/agents/benchmark.py (the K3 scoreboard half):
- task banks: built-ins present, registration works, suites listed
- swe_coding: reference solution passes hidden tests, the starter fails
  (negative control), timeouts count as failures, empty solution fails
- research: hermetic claim checks (all kinds), claim-labeling accuracy,
  the flipped-label control detects a broken scorer, unverifiable claims
  excluded from the denominator
- edits: hunks applied cleanly through the Wave-A edit_loop, bogus hunks
  rejected, zero collateral lines
- builds: injected backend scored stage-by-stage; no backend = honestly
  unmeasurable; backend exceptions count as failures
- latency: fast-path p50/p99 vs heavy-path p50/p99, percentiles, the fast
  tier beats the heavy tier
- run_scoreboard: end-to-end in harness-self-test mode (no live LLM),
  honest mode label, legacy suites unmeasurable without a model
- persistence: save/list/get/compare/export round-trip on migration 63,
  mode-mismatch comparisons refused
- CLI: `nm benchmark run|list|compare` parses, `bm` alias registered
"""
from __future__ import annotations

import json
import time
import unittest

from nomorals.agents import benchmark as bench
from nomorals.storage.db import Database


# ── fakes ──────────────────────────────────────────────────────────────────


class _FakeSettings:
    def resolve(self, key):
        raise RuntimeError("no settings in tests")


class _FakeContext:
    def __init__(self, db=None):
        self.settings = _FakeSettings()
        self.router = None
        self.extras = {}
        self.memory = None
        self.db = db


def _migrated_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _OkBackend:
    def __init__(self, fail_stage: str = ""):
        self.fail_stage = fail_stage
        self.calls: list[str] = []

    def run(self, kind: str, workdir: str) -> dict:
        self.calls.append(kind)
        stages = {"scaffold": True, "serve": True, "smoke": True}
        if self.fail_stage:
            stages[self.fail_stage] = False
        stages.update(seconds=0.1, detail="fake", timed_out=False)
        return stages


class _ExplodingBackend:
    def run(self, kind: str, workdir: str) -> dict:
        raise RuntimeError("backend blew up")


# ── task banks ─────────────────────────────────────────────────────────────


class TestTaskBanks(unittest.TestCase):
    def test_builtin_swe_tasks(self):
        tasks = bench.list_swe_tasks()
        self.assertGreaterEqual(len(tasks), 3)
        ids = {t.id for t in tasks}
        self.assertTrue({"fizzbuzz", "flatten", "dedupe"} <= ids)
        for t in tasks:
            self.assertTrue(t.spec.strip())
            self.assertTrue(t.hidden_tests.strip())
            self.assertTrue(t.reference_solution.strip())

    def test_register_swe_task(self):
        before = len(bench.list_swe_tasks())
        bench.register_swe_task(bench.SweTask(
            id="tmp-task", title="tmp", spec="s", starter="s",
            hidden_tests="t", reference_solution="r"))
        try:
            self.assertEqual(len(bench.list_swe_tasks()), before + 1)
        finally:
            bench._SWE_TASKS.pop()

    def test_register_research_and_edit_tasks(self):
        bench.register_research_task(
            bench.ResearchTask(id="tmp", question="q", claims=[]))
        bench.register_edit_task(
            bench.EditTask(id="tmp", filename="f", original="o",
                           edits=[], bogus_edits=[]))
        try:
            self.assertTrue(any(t.id == "tmp" for t in bench._RESEARCH_TASKS))
            self.assertTrue(any(t.id == "tmp" for t in bench._EDIT_TASKS))
        finally:
            bench._RESEARCH_TASKS.pop()
            bench._EDIT_TASKS.pop()

    def test_suites_registered(self):
        suites = bench.list_suites()
        for name in ("swe_coding", "research", "edits", "builds", "latency",
                     "reasoning", "planning", "tool_use", "self_correction"):
            self.assertIn(name, suites)

    def test_register_suite_custom(self):
        bench.register_suite(
            "tmp_suite",
            lambda ctx, limit, **kw: bench.DimensionScore("tmp_suite", 1.0,
                                                          1, 1, []))
        try:
            self.assertIn("tmp_suite", bench.list_suites())
        finally:
            del bench._SUITE_FNS["tmp_suite"]


# ── swe_coding ─────────────────────────────────────────────────────────────


class TestSweCoding(unittest.TestCase):
    def test_reference_solutions_pass(self):
        for task in bench.list_swe_tasks():
            if task.id == "tmp-task":
                continue
            outcome = bench._run_swe_task(task, task.reference_solution)
            self.assertTrue(outcome["pass"],
                            f"{task.id} reference failed: {outcome['detail']}")
            self.assertFalse(outcome["timed_out"])

    def test_starters_fail_negative_control(self):
        # the negative control: if a starter ever passes, the fixture is
        # broken and the harness must not rubber-stamp it
        for task in bench.list_swe_tasks():
            if task.id == "tmp-task":
                continue
            outcome = bench._run_swe_task(task, task.starter)
            self.assertFalse(outcome["pass"],
                             f"{task.id} starter passes — broken fixture")

    def test_empty_solution_fails(self):
        task = bench.list_swe_tasks()[0]
        outcome = bench._run_swe_task(task, "   \n")
        self.assertFalse(outcome["pass"])

    def test_broken_solution_fails(self):
        task = bench.list_swe_tasks()[0]
        outcome = bench._run_swe_task(task, "def fizzbuzz(n):\n    return 42\n")
        self.assertFalse(outcome["pass"])
        self.assertIn("task", outcome)

    def test_timeout_counts_as_failure(self):
        task = bench.SweTask(
            id="slow", title="slow", spec="s", starter="s",
            hidden_tests=("import unittest\nfrom solution import f\n\n\n"
                          "class T(unittest.TestCase):\n"
                          "    def test_x(self):\n"
                          "        self.assertEqual(f(), 1)\n"),
            reference_solution="import time\ndef f():\n    time.sleep(30)\n",
            timeout_s=3.0)
        outcome = bench._run_swe_task(task, task.reference_solution)
        self.assertFalse(outcome["pass"])
        self.assertTrue(outcome["timed_out"])

    def test_dim_swe_selftest_scores(self):
        ctx = _FakeContext()
        dim = bench._dim_swe_coding(ctx, 0, mode=bench.MODE_SELF_TEST)
        self.assertEqual(dim.name, "swe_coding")
        self.assertEqual(dim.score, 1.0)
        self.assertEqual(dim.passed, dim.total)
        self.assertGreater(dim.total, 0)
        for d in dim.details:
            self.assertIn("timed_out", d)


# ── research ───────────────────────────────────────────────────────────────


class TestResearch(unittest.TestCase):
    def test_claim_check_kinds(self):
        self.assertTrue(
            bench._run_claim_check(
                ("symbol", "nomorals.agents.benchmark", "DimensionScore")))
        self.assertFalse(
            bench._run_claim_check(
                ("symbol", "nomorals.agents.benchmark", "NoSuchThing")))
        self.assertTrue(bench._run_claim_check(("file_exists", "nomorals/cli.py")))
        self.assertFalse(
            bench._run_claim_check(("file_exists", "nomorals/nope.py")))
        self.assertTrue(
            bench._run_claim_check(
                ("line_count_ge", "nomorals/agents/benchmark.py", 100)))
        self.assertFalse(
            bench._run_claim_check(
                ("line_count_ge", "nomorals/agents/benchmark.py", 10**9)))
        self.assertTrue(
            bench._run_claim_check(
                ("regex_in_file", "nomorals/cli.py", r"benchmark")))
        # the CLI change in this wave registered the command
        self.assertTrue(bench._run_claim_check(("cli_command", "benchmark")))
        self.assertFalse(bench._run_claim_check(("cli_command", "nope-cmd")))

    def test_unverifiable_check_kinds(self):
        self.assertIsNone(bench._run_claim_check(None))
        self.assertIsNone(bench._run_claim_check(("mystery-kind",)))

    def test_selftest_labeler_is_perfect(self):
        task = bench._RESEARCH_TASKS[0]
        labels = bench._label_claims_selftest(task)
        scored = bench._score_claim_labels(task, labels)
        verifiable = [c for c in task.claims if c.get("check")]
        self.assertEqual(scored["verifiable"], len(verifiable))
        self.assertEqual(scored["correct"], scored["verifiable"])
        # the unverifiable claim is excluded from the denominator
        self.assertEqual(scored["unverifiable"], 1)

    def test_flipped_label_detected(self):
        task = bench._RESEARCH_TASKS[0]
        labels = bench._label_claims_selftest(task)
        first = next(c["id"] for c in task.claims if c.get("check"))
        labels[first] = "false" if labels[first] == "true" else "true"
        scored = bench._score_claim_labels(task, labels)
        self.assertLess(scored["correct"], scored["verifiable"])

    def test_dim_research_selftest(self):
        ctx = _FakeContext()
        dim = bench._dim_research(ctx, 0, mode=bench.MODE_SELF_TEST)
        self.assertEqual(dim.name, "research")
        self.assertEqual(dim.score, 1.0)
        self.assertEqual(dim.passed, dim.total)
        self.assertGreater(dim.total, 0)


# ── edit precision ─────────────────────────────────────────────────────────


class TestEditPrecision(unittest.TestCase):
    def test_reference_plan_all_clean_no_collateral(self):
        task = bench._EDIT_TASKS[0]
        m = bench._apply_edit_plan(task, bench._edit_plan_selftest(task))
        self.assertEqual(m["clean"], m["hunks"])
        self.assertEqual(m["hunks"], 3)
        self.assertEqual(m["rejected_bogus"], m["bogus"])
        self.assertEqual(m["bogus"], 2)
        self.assertEqual(m["collateral_lines"], 0)
        self.assertEqual(m["semantic_hits"], m["semantic_total"])

    def test_ambiguous_hunk_rejected(self):
        # "MAX_RETRIES" alone occurs twice -> the edit_loop must refuse it
        task = bench._EDIT_TASKS[0]
        m = bench._apply_edit_plan(task, [("MAX_RETRIES", "MAX_RETRIES_X")])
        self.assertEqual(m["clean"], 0)

    def test_collateral_detection(self):
        task = bench._EDIT_TASKS[0]
        plan = bench._edit_plan_selftest(task) + [
            ("def connect(host):", "def connect(host, port=80):")]
        m = bench._apply_edit_plan(task, plan)
        self.assertGreater(m["collateral_lines"], 0)

    def test_dim_edits_selftest(self):
        ctx = _FakeContext()
        dim = bench._dim_edits(ctx, 0, mode=bench.MODE_SELF_TEST)
        self.assertEqual(dim.name, "edits")
        self.assertEqual(dim.score, 1.0)
        self.assertEqual(dim.passed, dim.total)


# ── builds ─────────────────────────────────────────────────────────────────


class TestBuilds(unittest.TestCase):
    def test_backend_scored_stage_by_stage(self):
        ctx = _FakeContext()
        dim = bench._dim_builds(ctx, 0, mode=bench.MODE_SELF_TEST,
                                build_backend=_OkBackend())
        self.assertEqual(dim.name, "builds")
        self.assertEqual(dim.score, 1.0)
        # cli_tool: scaffold+smoke (2), webapp: scaffold+serve+smoke (3)
        self.assertEqual((dim.passed, dim.total), (5, 5))

    def test_failed_stage_lowers_score(self):
        ctx = _FakeContext()
        dim = bench._dim_builds(ctx, 0, mode=bench.MODE_SELF_TEST,
                                build_backend=_OkBackend(fail_stage="smoke"))
        self.assertEqual(dim.score, 3 / 5)
        self.assertEqual((dim.passed, dim.total), (3, 5))

    def test_no_backend_is_honestly_unmeasurable(self):
        ctx = _FakeContext()
        dim = bench._dim_builds(ctx, 0, mode=bench.MODE_SELF_TEST,
                                build_backend=None)
        self.assertIsNone(dim.score)
        self.assertEqual((dim.passed, dim.total), (0, 0))

    def test_exploding_backend_counts_as_failure(self):
        ctx = _FakeContext()
        dim = bench._dim_builds(ctx, 0, mode=bench.MODE_SELF_TEST,
                                build_backend=_ExplodingBackend())
        self.assertEqual(dim.score, 0.0)
        self.assertEqual(dim.passed, 0)
        self.assertGreater(dim.total, 0)


# ── latency ────────────────────────────────────────────────────────────────


class TestLatency(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(bench._percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(bench._percentile([5], 99), 5)
        self.assertEqual(bench._percentile([], 50), 0.0)

    def test_fast_beats_heavy_selftest(self):
        ctx = _FakeContext()
        dim = bench._dim_latency(ctx, 10, mode=bench.MODE_SELF_TEST)
        self.assertEqual(dim.name, "latency")
        self.assertEqual(dim.score, 1.0)
        self.assertEqual(dim.passed, dim.total)
        fast = next(d for d in dim.details if "fast-path" in d["task"])
        heavy = next(d for d in dim.details if "heavy-path" in d["task"])
        self.assertIn("p50=", fast["detail"])
        self.assertIn("p99=", heavy["detail"])
        self.assertIn("simulated router", heavy["detail"])


# ── run_scoreboard end to end ──────────────────────────────────────────────


class TestRunScoreboard(unittest.TestCase):
    def test_selftest_mode_labels_honestly(self):
        ctx = _FakeContext()
        report = bench.run_scoreboard(
            ctx, suites=["swe_coding", "research", "edits", "latency"],
            build_backend=_OkBackend())
        self.assertEqual(report.mode, bench.MODE_SELF_TEST)
        self.assertFalse(report.measurable)
        self.assertEqual(report.scoreboard, "k3")
        self.assertTrue(report.run_id)
        self.assertIsNotNone(report.overall)
        for name in ("swe_coding", "research", "edits", "latency"):
            self.assertIn(name, report.scores)

    def test_legacy_suites_unmeasurable_without_model(self):
        ctx = _FakeContext()
        report = bench.run_scoreboard(ctx, suites=["reasoning"])
        dim = report.scores["reasoning"]
        self.assertIsNone(dim.score)
        self.assertIn("live model", dim.details[0]["detail"])

    def test_unknown_suites_ignored(self):
        ctx = _FakeContext()
        report = bench.run_scoreboard(ctx, suites=["nope"])
        self.assertEqual(report.scores, {})

    def test_report_serializes(self):
        ctx = _FakeContext()
        report = bench.run_scoreboard(ctx, suites=["research"])
        payload = report.as_dict()
        self.assertEqual(payload["mode"], bench.MODE_SELF_TEST)
        self.assertEqual(payload["scoreboard"], "k3")
        json.dumps(payload)  # must be JSON-clean


# ── persistence ────────────────────────────────────────────────────────────


class TestPersistence(unittest.TestCase):
    def _report(self, **over):
        scores = {
            "swe_coding": bench.DimensionScore("swe_coding", 1.0, 3, 3, []),
            "research": bench.DimensionScore("research", 0.5, 3, 6, []),
        }
        params = dict(scores=scores, overall=0.75, measurable=True,
                      provider="mock", seconds=1.5,
                      run_id="abc123", suite="all",
                      mode=bench.MODE_SELF_TEST)
        params.update(over)
        return bench.ScoreboardReport(**params)

    def test_save_list_get_roundtrip(self):
        db = _migrated_db()
        rid = bench.save_run(db, self._report())
        self.assertEqual(rid, "abc123")
        rows = bench.list_runs(db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "abc123")
        self.assertEqual(rows[0]["passed"], 6)
        self.assertEqual(rows[0]["total"], 9)
        full = bench.get_run(db, "abc123")
        self.assertIsNotNone(full)
        self.assertEqual(full["dimensions"]["swe_coding"]["score"], 1.0)
        self.assertEqual(full["dimensions"]["research"]["passed"], 3)

    def test_list_newest_first_and_suite_filter(self):
        db = _migrated_db()
        bench.save_run(db, self._report(run_id="r1", suite="swe_coding"))
        time.sleep(0.01)
        bench.save_run(db, self._report(run_id="r2", suite="research"))
        rows = bench.list_runs(db)
        self.assertEqual([r["id"] for r in rows], ["r2", "r1"])
        only = bench.list_runs(db, suite="research")
        self.assertEqual([r["id"] for r in only], ["r2"])

    def test_get_missing_returns_none(self):
        db = _migrated_db()
        self.assertIsNone(bench.get_run(db, "nope"))

    def test_save_without_db_never_raises(self):
        self.assertEqual(bench.save_run(None, self._report()), "")
        self.assertEqual(bench.list_runs(None), [])
        self.assertIsNone(bench.get_run(None, "x"))

    def test_compare_deltas(self):
        db = _migrated_db()
        bench.save_run(db, self._report(run_id="a"))
        bench.save_run(db, self._report(run_id="b", overall=1.0,
                                        scores={"swe_coding":
                                                bench.DimensionScore(
                                                    "swe_coding", 1.0, 3, 3,
                                                    [])}))
        cmp = bench.compare_runs(db, "a", "b")
        self.assertTrue(cmp["ok"])
        self.assertAlmostEqual(cmp["overall_delta"], 0.25)
        self.assertIn("swe_coding", cmp["dimensions"])
        self.assertIn("research", cmp["dimensions"])

    def test_compare_mode_mismatch_refused(self):
        db = _migrated_db()
        bench.save_run(db, self._report(run_id="a",
                                        mode=bench.MODE_SELF_TEST))
        bench.save_run(db, self._report(run_id="b", mode=bench.MODE_MODEL))
        cmp = bench.compare_runs(db, "a", "b")
        self.assertFalse(cmp["ok"])
        self.assertIn("mode mismatch", cmp["error"])

    def test_compare_missing_run(self):
        db = _migrated_db()
        bench.save_run(db, self._report(run_id="a"))
        cmp = bench.compare_runs(db, "a", "ghost")
        self.assertFalse(cmp["ok"])

    def test_export_json(self):
        db = _migrated_db()
        bench.save_run(db, self._report())
        run = bench.get_run(db, "abc123")
        text = bench.export_run_json(run)
        back = json.loads(text)
        self.assertEqual(back["id"], "abc123")


# ── CLI surface ────────────────────────────────────────────────────────────


class TestCliSurface(unittest.TestCase):
    def test_parser_accepts_benchmark_commands(self):
        from nomorals.cli import _parser, _canonical_command, CLI_ALIASES

        self.assertIn("bm", CLI_ALIASES["benchmark"])
        for argv in (["benchmark", "run"],
                     ["benchmark", "run", "swe_coding"],
                     ["benchmark", "run", "all", "--limit", "2"],
                     ["bm", "run", "latency"],
                     ["benchmark", "list"],
                     ["benchmark", "list", "--limit", "5"],
                     ["benchmark", "compare", "aaa", "bbb"]):
            args = _parser().parse_args(argv)
            # argparse keeps the typed alias; _dispatch canonicalizes it
            self.assertEqual(_canonical_command(args.command), "benchmark")

    def test_benchmark_list_action_with_stub_context(self):
        from nomorals import cli as cli_mod

        db = _migrated_db()
        bench.save_run(db, self._report_for_cli())
        ctx = _FakeContext(db=db)
        args = cli_mod._parser().parse_args(["benchmark", "list"])
        self.assertEqual(cli_mod._cmd_benchmark(args, ctx), 0)

    def test_benchmark_compare_action_with_stub_context(self):
        from nomorals import cli as cli_mod

        db = _migrated_db()
        bench.save_run(db, self._report_for_cli(run_id="a1"))
        bench.save_run(db, self._report_for_cli(run_id="b1"))
        ctx = _FakeContext(db=db)
        args = cli_mod._parser().parse_args(["benchmark", "compare",
                                             "a1", "b1"])
        self.assertEqual(cli_mod._cmd_benchmark(args, ctx), 0)

    def test_benchmark_run_unknown_suite_rejected(self):
        from nomorals import cli as cli_mod

        ctx = _FakeContext(db=_migrated_db())
        args = cli_mod._parser().parse_args(["benchmark", "run", "nope"])
        self.assertEqual(cli_mod._cmd_benchmark(args, ctx), 2)

    def _report_for_cli(self, run_id="cli1"):
        return bench.ScoreboardReport(
            scores={"research": bench.DimensionScore("research", 1.0, 6, 6,
                                                     [])},
            overall=1.0, measurable=False, provider="", seconds=0.5,
            run_id=run_id, suite="research", mode=bench.MODE_SELF_TEST)


if __name__ == "__main__":
    unittest.main()
