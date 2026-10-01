"""Acceptance tests for audit Phase D (prompt 19): scale.

- relevance-ranked repo context: "fix the coding loop" ranks coding.py,
  edit_loop.py, registry.py at the top
- context metadata shows included_files / dropped_files / token
  accounting; a tiny budget drops low-rank files first
- the explore phase of a 5-file fixture task issues one parallel
  call_many block, not 5 sequential calls
- a slow suite runs in the background while the agent completes a lint
  pass concurrently
- a run exceeding the cap is killed and returns partial results (no hang)
- the coding loop's verify step uses the background runner with the
  lint gate running concurrently
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.bg_tests import BackgroundTestRun, background_run_tests
from nomorals.agents.coding import CodingAgent, _DIFF_REVIEW_FOCUS
from nomorals.tools import repo_context as repo_context_mod
from nomorals.tools.repo_context import (
    legacy_context,
    rank_context,
)

REPO = Path(__file__).resolve().parent.parent


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class ScriptedRouter:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, params=None):
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")


def _json_block(payload):
    return "```json\n" + json.dumps(payload) + "\n```"


class StubDB:
    """coding_log + skills tables; dict rows like the real Database."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE coding_log (id TEXT, task TEXT, filename TEXT,"
            " attempt INT, exit_code INT, timed_out INT, stdout TEXT,"
            " stderr TEXT, code TEXT, created_at REAL)")
        self.conn.execute(
            "CREATE TABLE skills (id TEXT PRIMARY KEY, name TEXT NOT NULL"
            " DEFAULT '', description TEXT NOT NULL DEFAULT '',"
            " kind TEXT NOT NULL DEFAULT 'strategy',"
            " body TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',"
            " source TEXT NOT NULL DEFAULT '', uses INTEGER NOT NULL DEFAULT 0,"
            " success_count INTEGER NOT NULL DEFAULT 0,"
            " failure_count INTEGER NOT NULL DEFAULT 0,"
            " last_used REAL NOT NULL DEFAULT 0,"
            " version INTEGER NOT NULL DEFAULT 1,"
            " pruned INTEGER NOT NULL DEFAULT 0,"
            " created_at REAL NOT NULL DEFAULT 0,"
            " updated_at REAL NOT NULL DEFAULT 0)")

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def query_one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None


class StubContext:
    def __init__(self, router):
        self.router = router
        self.db = StubDB()
        self.settings = SimpleNamespace(reasoning_mode="off")
        self.executor = None
        self.blackboard = None


class FixtureRepo:
    def __init__(self, with_git=False):
        self.root = Path(tempfile.mkdtemp(prefix="phase_d_"))
        if with_git:
            _git("init", "-q", cwd=self.root)
            _git("config", "user.email", "t@example.com", cwd=self.root)
            _git("config", "user.name", "T", cwd=self.root)

    def write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _devon_like_repo(fx):
    """A miniature of the Devon layout: coding.py imports the registry
    and the edit loop; a test covers coding.py; unrelated.py is noise."""
    fx.write("coding.py",
             "from registry import ToolRegistry\n"
             "from edit_loop import EditLoop\n"
             "\n"
             "class CodingAgent:\n"
             "    def run(self, task):\n"
             "        loop = EditLoop()\n"
             "        return loop.run(task)\n")
    fx.write("edit_loop.py",
             "class EditLoop:\n"
             "    def run(self, task):\n"
             "        return 'done'\n")
    fx.write("registry.py",
             "class ToolRegistry:\n"
             "    def call(self, name):\n"
             "        return name\n")
    fx.write("test_coding.py",
             "from coding import CodingAgent\n"
             "\n"
             "def test_run():\n"
             "    assert CodingAgent() is not None\n")
    fx.write("unrelated.py", "CONSTANT = 1\n" * 300)


class RankedContextTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        _devon_like_repo(self.fx)

    def test_fix_coding_loop_ranks_core_files_top(self):
        ctx = rank_context(
            self.fx.root, "fix the coding loop", targets=["coding.py"],
            symbol_search=lambda q, lim: ["unrelated.py"])
        top3 = ctx.included_files[:3]
        self.assertIn("coding.py", top3)
        # import neighbors (1 hop) outrank symbol-search hits
        self.assertIn("edit_loop.py", ctx.included_files[:4])
        self.assertIn("registry.py", ctx.included_files[:4])
        self.assertLess(ctx.included_files.index("edit_loop.py"),
                        ctx.included_files.index("unrelated.py"))
        self.assertLess(ctx.included_files.index("registry.py"),
                        ctx.included_files.index("unrelated.py"))

    def test_test_files_outrank_symbol_hits(self):
        ctx = rank_context(
            self.fx.root, "fix the coding loop", targets=["coding.py"],
            symbol_search=lambda q, lim: ["unrelated.py"])
        self.assertLess(ctx.included_files.index("test_coding.py"),
                        ctx.included_files.index("unrelated.py"))

    def test_budget_metadata_drops_low_rank_first(self):
        ctx = rank_context(
            self.fx.root, "fix the coding loop", targets=["coding.py"],
            budget_tokens=100,
            symbol_search=lambda q, lim: ["unrelated.py"])
        meta = ctx.metadata()
        self.assertIn("included_files", meta)
        self.assertIn("dropped_files", meta)
        self.assertIn("budget_tokens", meta)
        # the direct target survives; the lowest-rank file is cut first
        self.assertIn("coding.py", meta["included_files"])
        self.assertIn("unrelated.py", meta["dropped_files"])
        self.assertNotIn("coding.py", meta["dropped_files"])

    def test_token_accounting_never_silently_exceeds(self):
        for budget in (100, 500, 2000):
            ctx = rank_context(
                self.fx.root, "fix the coding loop",
                targets=["coding.py"], budget_tokens=budget,
                symbol_search=lambda q, lim: ["unrelated.py"])
            self.assertEqual(ctx.budget_tokens, budget)
            meta = ctx.metadata()
            # hard invariant: the budget is never silently exceeded
            self.assertLessEqual(ctx.used_tokens, budget)
            self.assertTrue(meta["within_budget"])

    def test_tiny_budget_truncates_top_file_instead_of_empty(self):
        self.fx.write("big.py", "X = 1\n" * 5000)
        ctx = rank_context(self.fx.root, "big", targets=["big.py"],
                           budget_tokens=100)
        meta = ctx.metadata()
        self.assertLessEqual(ctx.used_tokens, 100)
        self.assertTrue(meta["within_budget"])
        # top-ranked file present but explicitly cut, not silently whole
        self.assertIn("big.py", ctx.included_files)
        self.assertIn("big.py", meta["truncated_files"])
        self.assertIn("truncated to fit", ctx.files["big.py"])

    def test_search_many_parallel_fast_path_used(self):
        seen = {}

        class ParallelSearch:
            def __call__(self, query, limit=10):
                seen["serial"] = True
                return []

            def search_many(self, queries, limit=10):
                seen["queries"] = list(queries)
                return {q: ["coding.py"] for q in queries}

        ctx = rank_context(self.fx.root, "fix the coding loop",
                           symbol_search=ParallelSearch())
        # more than one keyword -> the parallel entry point, not serial
        self.assertNotIn("serial", seen)
        self.assertGreaterEqual(len(seen["queries"]), 2)
        self.assertIn("coding.py", ctx.included_files)

    def test_plain_callable_search_still_serial(self):
        calls = []

        def search(query, limit=10):
            calls.append(query)
            return ["coding.py"]

        ctx = rank_context(self.fx.root, "fix the coding loop",
                           symbol_search=search)
        self.assertGreaterEqual(len(calls), 1)
        self.assertIn("coding.py", ctx.included_files)

    def test_legacy_fallback_flag(self):
        ctx = legacy_context(self.fx.root, ["coding.py"])
        self.assertIn("structure", ctx)
        self.assertIn("coding.py", ctx["files"])
        meta = json.loads(ctx["metadata"])
        self.assertFalse(meta["ranked"])


class RealRepoRankingTests(unittest.TestCase):
    """Phase D acceptance on the actual Devon tree, not just the
    miniature: a coding-loop task must surface the real coding agent,
    its edit loop, and the registry near the top."""

    def test_real_repo_coding_task_ranks_core_modules(self):
        ctx = rank_context(
            REPO, "fix the coding agent edit loop",
            targets=["nomorals/agents/coding.py"],
            budget_tokens=150000,
            symbol_search=lambda q, lim: ["nomorals/tools/registry.py"])
        self.assertEqual(ctx.included_files[0],
                         "nomorals/agents/coding.py")
        # 1-hop import neighbors (relative imports resolved) ...
        self.assertIn("nomorals/tools/edit_loop.py", ctx.included_files)
        # ... outrank symbol-search hits
        self.assertIn("nomorals/tools/registry.py", ctx.included_files)
        self.assertLess(
            ctx.included_files.index("nomorals/tools/edit_loop.py"),
            ctx.included_files.index("nomorals/tools/registry.py"))
        meta = ctx.metadata()
        self.assertLessEqual(ctx.used_tokens, ctx.budget_tokens)
        self.assertTrue(meta["within_budget"])


class ParallelExploreTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        _devon_like_repo(self.fx)
        from nomorals.tools.registry import ToolRegistry
        from nomorals.tools import filesystem

        self.registry = ToolRegistry(
            context=SimpleNamespace(
                settings=SimpleNamespace(
                    workspace_dir=str(self.fx.root))))
        filesystem.register(self.registry)
        self.ctx = StubContext(ScriptedRouter([]))
        self.ctx.tools = self.registry
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))

    def test_explore_issues_one_call_many_block(self):
        plan = [{"path": p, "why": "x", "new_file": False}
                for p in ("coding.py", "edit_loop.py", "registry.py",
                          "test_coding.py", "unrelated.py")]
        with mock.patch.object(self.registry, "call_many",
                               wraps=self.registry.call_many) as m_many:
            texts = self.agent._explore_reads(plan)
        # one parallel block, not 5 sequential calls
        m_many.assert_called_once()
        (calls,), kwargs = m_many.call_args
        self.assertEqual(len(calls), 5)
        self.assertEqual(kwargs.get("max_workers"), 8)
        self.assertEqual(set(texts), {s["path"] for s in plan})
        for rel, content in texts.items():
            self.assertTrue(content, f"{rel} came back empty")
            self.assertIn(content,
                          (self.fx.root / rel).read_text())

    def test_explore_falls_back_without_registry(self):
        ctx = StubContext(ScriptedRouter([]))  # no .tools
        agent = CodingAgent(ctx, root=str(self.fx.root))
        plan = [{"path": "coding.py", "why": "x", "new_file": False}]
        texts = agent._explore_reads(plan)
        self.assertEqual(texts["coding.py"],
                         (self.fx.root / "coding.py").read_text())


class BackgroundTestRunTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("tests/__init__.py", "")

    def _slow_suite(self, sleep_s=5):
        self.fx.write(
            "tests/test_slow.py",
            "import time\n"
            "import unittest\n"
            "\n"
            "class SlowTests(unittest.TestCase):\n"
            "    def test_slow_one(self):\n"
            f"        time.sleep({sleep_s})\n"
            "        self.assertTrue(True)\n")

    def test_slow_suite_runs_while_lint_pass_completes(self):
        from nomorals.tools import lint as lint_mod

        self._slow_suite(sleep_s=5)
        run = background_run_tests(str(self.fx.root), cap_seconds=120)
        self.assertIsInstance(run, BackgroundTestRun)
        # the run is in flight; the agent does its lint pass now
        st = run.poll()
        self.assertTrue(st["running"], "suite should still be running")
        lint_t0 = time.perf_counter()
        lint_res = lint_mod.lint(["tests/test_slow.py"],
                                 repo=str(self.fx.root))
        lint_s = time.perf_counter() - lint_t0
        self.assertIsInstance(lint_res, dict)  # lint pass completed
        # ...while the suite was still going: real concurrency
        st = run.poll()
        self.assertTrue(st["running"],
                        "lint finished but the suite was already done — "
                        "no overlap happened")
        final = run.wait()
        self.assertTrue(final["ok"], final)
        self.assertGreaterEqual(final["passed"], 1)
        # overlapped: wall time < suite + lint run back-to-back
        self.assertLess(final["seconds"], 5 + lint_s + 5)

    def test_cap_kill_returns_partial_results_no_hang(self):
        self.fx.write(
            "tests/test_hang.py",
            "import time\n"
            "import unittest\n"
            "\n"
            "class HangTests(unittest.TestCase):\n"
            "    def test_hang(self):\n"
            "        time.sleep(60)\n"
            "        self.assertTrue(True)\n")
        t0 = time.perf_counter()
        run = background_run_tests(str(self.fx.root), cap_seconds=3)
        final = run.wait()
        elapsed = time.perf_counter() - t0
        self.assertTrue(final["timed_out"])
        self.assertTrue(final["partial"])
        self.assertFalse(final["ok"])
        self.assertLess(elapsed, 25, "the cap kill hung")
        # the process tree is actually dead
        try:
            os.kill(run._proc.pid, 0)
            self.fail("test process still alive after cap kill")
        except ProcessLookupError:
            pass

    def test_live_progress_surfaces_counts(self):
        self.fx.write(
            "tests/test_two.py",
            "import time\n"
            "import unittest\n"
            "\n"
            "class TwoTests(unittest.TestCase):\n"
            "    def test_one(self):\n"
            "        time.sleep(1)\n"
            "        self.assertTrue(True)\n"
            "    def test_two(self):\n"
            "        time.sleep(1)\n"
            "        self.assertTrue(True)\n")
        run = background_run_tests(str(self.fx.root), cap_seconds=60)
        seen = []
        final = run.wait(
            on_progress=lambda st: seen.append(st["passed"]))
        self.assertTrue(final["ok"], final)
        self.assertTrue(any(p >= 1 for p in seen),
                        f"no live progress surfaced: {seen}")


class BackgroundLoopIntegrationTests(unittest.TestCase):
    """The coding loop verifies in the background while the lint gate
    runs concurrently."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("calc.py", "def add(a, b):\n    return a - b\n")
        self.fx.write("tests/__init__.py", "")
        self.fx.write(
            "tests/test_calc.py",
            "import time\n"
            "from calc import add\n"
            "\n"
            "def test_add():\n"
            "    time.sleep(3)\n"
            "    assert add(2, 3) == 5\n")
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "calc.py", "why": "fix add", "new_file": False}]}),
            _json_block({"edits": [
                {"old_text": "return a - b",
                 "new_text": "return a + b"}]}),
        ])
        self.ctx = StubContext(router)
        self.ctx.settings = SimpleNamespace(reasoning_mode="always")
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))
        self.agent._recall_path = (
            Path(tempfile.mkdtemp(prefix="phase_d_bg_")) / "index.json")
        self.addCleanup(shutil.rmtree,
                        self.agent._recall_path.parent, True)

    def test_loop_verifies_in_background_with_concurrent_lint(self):
        import nomorals.agents.bg_tests as bg_mod
        import nomorals.tools.lint as lint_mod

        real_bg = bg_mod.background_run_tests
        bg_used = {}
        lint_finished_at = {}
        bg_done_at = {}

        def bg_spy(repo, **kwargs):
            run = real_bg(repo, **kwargs)
            bg_used["run"] = run
            real_wait = run.wait

            def wait_spy(**wkwargs):
                out = real_wait(**wkwargs)
                bg_done_at["t"] = time.perf_counter()
                return out

            run.wait = wait_spy
            return run

        real_lint = lint_mod.lint

        def lint_spy(paths, repo=None):
            out = real_lint(paths, repo=repo)
            lint_finished_at["t"] = time.perf_counter()
            return out

        def review(context, text, *, focus="", max_calls=None):
            return []  # per-diff review and the gate both pass

        with mock.patch.object(bg_mod, "background_run_tests",
                               side_effect=bg_spy), \
             mock.patch.object(lint_mod, "lint", side_effect=lint_spy), \
             mock.patch("nomorals.agents.reasoning.review_text",
                        side_effect=review):
            result = self.agent.run(
                "Fix add() to add", max_iterations=5, timeout=30,
                background_tests=True, test_cap_seconds=120)

        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertIn("run", bg_used, "verify did not use the background runner")
        body = (self.fx.root / "calc.py").read_text()
        self.assertIn("return a + b", body)
        # the lint gate finished before the background suite did:
        # they overlapped instead of running back-to-back
        self.assertIn("t", lint_finished_at)
        self.assertIn("t", bg_done_at)
        self.assertLess(lint_finished_at["t"], bg_done_at["t"],
                        "lint did not run concurrently with the test suite")


if __name__ == "__main__":
    unittest.main()
