"""Phase A tests: dead coding tools wired up, git tool, `nm code` CLI.

Covers:
- edit_file surgical replacement (exact-once match enforcement)
- git_commit refuses when unrelated dirty changes are present
- execute_plan validates the JSON plan schema
- execute_plan runs rollback steps when a step fails (fixture repo)
- the registry exposes all Phase A tools
"""

import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.errors import ToolError, ValidationError
from nomorals.tools import code_executor, edit_loop, git as git_tool
from nomorals.tools.registry import ToolRegistry


def _git(*args, cwd):
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr[:300]}")
    return proc.stdout


class GitFixture(unittest.TestCase):
    """A throwaway git repo for the git/executor tests."""

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git not available")
        self.repo = Path(tempfile.mkdtemp(prefix="phase_a_"))
        _git("init", "-q", cwd=self.repo)
        _git("config", "user.email", "test@example.com", cwd=self.repo)
        _git("config", "user.name", "Test", cwd=self.repo)
        (self.repo / "app.py").write_text('VALUE = 1\nprint("hello")\n')
        _git("add", ".", cwd=self.repo)
        _git("commit", "-qm", "initial", cwd=self.repo)

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)


class EditFileTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="phase_a_edit_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.loop = edit_loop.EditLoop(agent=None, project_root=str(self.root))

    def _write(self, name, content):
        p = self.root / name
        p.write_text(content)
        return name

    def test_surgical_replace_happy_path(self):
        self._write("a.py", "x = 1\ny = 2\n")
        diff = self.loop.surgical_replace("a.py", "x = 1", "x = 42")
        self.assertIn("-x = 1", diff)
        self.assertIn("+x = 42", diff)
        self.assertEqual((self.root / "a.py").read_text(), "x = 42\ny = 2\n")

    def test_surgical_replace_no_match_raises(self):
        self._write("a.py", "x = 1\n")
        with self.assertRaises(ValueError):
            self.loop.surgical_replace("a.py", "zzz", "q")

    def test_surgical_replace_ambiguous_match_raises(self):
        self._write("a.py", "x = 1\nx = 1\n")
        with self.assertRaises(ValueError):
            self.loop.surgical_replace("a.py", "x = 1", "x = 9")
        # File untouched.
        self.assertEqual((self.root / "a.py").read_text(), "x = 1\nx = 1\n")


class GitToolTests(GitFixture):
    def test_status(self):
        status = git_tool.git_status(str(self.repo))
        expected = _git("branch", "--show-current", cwd=self.repo).strip()
        self.assertEqual(status["branch"], expected)
        self.assertFalse(status["dirty"])

    def test_commit_refuses_with_unrelated_dirty_files(self):
        (self.repo / "app.py").write_text('VALUE = 2\n')
        (self.repo / "other.py").write_text('OTHER = 1\n')
        with self.assertRaises(ToolError) as ctx:
            git_tool.git_commit("scoped change", paths=["app.py"], repo=str(self.repo))
        self.assertIn("unrelated dirty", str(ctx.exception))

    def test_commit_scoped_paths_succeeds(self):
        (self.repo / "app.py").write_text('VALUE = 2\n')
        result = git_tool.git_commit("bump value", paths=["app.py"], repo=str(self.repo))
        self.assertTrue(result["committed"])
        log = git_tool.git_log(1, str(self.repo))
        self.assertEqual(log["commits"][0]["subject"], "bump value")

    def test_commit_without_paths_refuses_unstaged(self):
        (self.repo / "app.py").write_text('VALUE = 2\n')
        with self.assertRaises(ToolError):
            git_tool.git_commit("nope", repo=str(self.repo))

    def test_restore(self):
        (self.repo / "app.py").write_text('VALUE = 99\n')
        git_tool.restore("app.py", str(self.repo))
        self.assertEqual((self.repo / "app.py").read_text(), 'VALUE = 1\nprint("hello")\n')


class ExecutePlanValidationTests(unittest.TestCase):
    def test_bad_json_rejected(self):
        with self.assertRaises(ValidationError):
            code_executor._parse_plan("{not json")

    def test_unknown_action_rejected(self):
        plan = json.dumps({"goal": "x", "steps": [{"action": "rm_rf", "target": "/"}]})
        with self.assertRaises(ValidationError):
            code_executor._parse_plan(plan)

    def test_missing_target_rejected(self):
        plan = json.dumps({"goal": "x", "steps": [{"action": "edit_file", "target": "  "}]})
        with self.assertRaises(ValidationError):
            code_executor._parse_plan(plan)

    def test_empty_steps_rejected(self):
        with self.assertRaises(ValidationError):
            code_executor._parse_plan(json.dumps({"goal": "x", "steps": []}))

    def test_valid_plan_parses(self):
        plan = json.dumps({
            "goal": "bump",
            "steps": [{"action": "edit_file", "target": "app.py",
                       "description": "bump", "estimated_risk": "low"}],
        })
        parsed = code_executor._parse_plan(plan)
        self.assertEqual(len(parsed.steps), 1)
        self.assertEqual(len(parsed.rollback_steps), 1)
        self.assertEqual(parsed.rollback_steps[0].action, "restore_file")


class StubAgent:
    """Deterministic stand-in for the LLM: always returns canned code."""

    def __init__(self, code):
        self._code = code

    def chat(self, prompt):
        return SimpleNamespace(content=f"```python\n{self._code}\n```")


class ExecutePlanRollbackTests(GitFixture):
    def test_failing_step_triggers_rollback(self):
        original = (self.repo / "app.py").read_text()
        agent = StubAgent('VALUE = 2\nprint("hello")\n')
        executor = code_executor.CodeExecutor(agent, project_root=str(self.repo))
        plan = code_executor.ExecutionPlan(
            plan_id="p1", goal="bump then fail", summary="",
            steps=[
                code_executor.PlanStep(step_id="s1", action="edit_file",
                                       target="app.py", description="bump value"),
                code_executor.PlanStep(step_id="s2", action="run_tests",
                                       target="false", description="always fails"),
            ],
        )
        plan.rollback_steps = code_executor.CodeExecutor._generate_rollback_steps(plan.steps)

        result = asyncio.run(executor.execute_plan(plan))

        self.assertFalse(result.success)
        # Rollback restored the committed file.
        self.assertEqual((self.repo / "app.py").read_text(), original)


class RegistryExposureTests(unittest.TestCase):
    def test_phase_a_tools_registered(self):
        root = tempfile.mkdtemp(prefix="phase_a_reg_")
        self.addCleanup(shutil.rmtree, root, True)
        context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=root), db=None, router=None)
        registry = ToolRegistry(context)
        registry.register_builtins()
        names = set(registry.names())
        for tool in ("edit_file", "apply_patch", "show_diff", "execute_plan",
                     "index_repo", "search_code", "get_symbol",
                     "git_status", "git_diff", "git_log", "git_branch",
                     "git_commit", "git_push", "git_stash"):
            self.assertIn(tool, names, f"tool {tool!r} not registered")

    def test_write_tools_require_confirmation(self):
        root = tempfile.mkdtemp(prefix="phase_a_reg2_")
        self.addCleanup(shutil.rmtree, root, True)
        context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=root), db=None, router=None)
        registry = ToolRegistry(context)
        registry.register_builtins()
        self.assertTrue(registry._tools["git_commit"].confirm)
        self.assertTrue(registry._tools["git_push"].confirm)
        self.assertTrue(registry._tools["git_stash"].confirm)
        self.assertFalse(registry._tools["git_status"].confirm)


class CodeIndexerSmokeTests(unittest.TestCase):
    def test_index_and_search_roundtrip(self):
        from nomorals.memory.embeddings import Embedder
        from nomorals.storage.db import Database
        from nomorals.storage.vectors import VectorStore
        from nomorals.tools.code_indexer import CodeIndexer, _EmbedAdapter

        src = Path(tempfile.mkdtemp(prefix="phase_a_idx_"))
        self.addCleanup(shutil.rmtree, src, True)
        (src / "mod.py").write_text('def frobnicate(x):\n    """Frobnicate the x."""\n    return x * 2\n')

        db = Database(":memory:")
        db.migrate()  # creates the embeddings table, like build_context does
        indexer = CodeIndexer(db, _EmbedAdapter(Embedder(provider="hashing")),
                              VectorStore(db))
        stats = asyncio.run(indexer.index_repo(str(src)))
        self.assertGreaterEqual(stats.get("units_indexed", 0), 1)

        unit = asyncio.run(indexer.get_function(str(src / "mod.py"), "frobnicate"))
        self.assertIsNotNone(unit)


if __name__ == "__main__":
    unittest.main()
