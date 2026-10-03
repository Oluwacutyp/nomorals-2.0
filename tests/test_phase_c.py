"""Acceptance tests for audit Phase C (prompt 18): trust.

- coding-role task in an orchestrator plan executes with exactly the
  allowlisted tools; a non-allowlisted tool (browser) is refused with a
  clear capability-naming error
- renamed-variable variant of a distilled error still recalls the fix
- the critic catches an obviously-wrong fix and forces a rework iteration
- an always-objecting critic terminates after 2 rounds with objections
  attached (no hang)
- `nm code review` shows the diff + critic verdict + commit prompt
"""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coding import CodingAgent, _DIFF_REVIEW_FOCUS
from nomorals.core.tasks import Task
from nomorals.tools import agents as agents_mod
from nomorals.tools.agents import CODING_TOOLS

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
        self.root = Path(tempfile.mkdtemp(prefix="phase_c_"))
        if with_git:
            _git("init", "-q", cwd=self.root)
            _git("config", "user.email", "t@example.com", cwd=self.root)
            _git("config", "user.name", "T", cwd=self.root)

    def write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def commit_all(self, msg="init"):
        _git("add", ".", cwd=self.root)
        _git("commit", "-qm", msg, cwd=self.root)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _gate_only_review(text_ctx):
    """review_text side effect: flaws only for the diff-review gate."""
    def _review(context, text, *, focus="", max_calls=None):
        if focus == _DIFF_REVIEW_FOCUS:
            return list(text_ctx["flaws"])
        return []

    return _review


class RoleAllowlistTests(unittest.TestCase):
    def setUp(self):
        from nomorals.tools.registry import ToolRegistry

        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        router = ScriptedRouter([
            _json_block({"files": []}),  # plan -> default single file
            # new-file draft: substantive enough to pass honest acceptance
            # (a trivial one-liner with no output is correctly rejected as
            # fake success — see the CodingAgent acceptance guard).
            "```python\n\"\"\"Main module.\"\"\"\n\nX = 1\n\n\n"
            "def get_x():\n    \"\"\"Return the module constant.\"\"\"\n"
            "    return X\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    print(f\"X = {get_x()}\")\n```\n",
        ])
        self.ctx = StubContext(router)
        self.registry = ToolRegistry(context=self.ctx).register_builtins()
        self.ctx.tools = self.registry

    def test_tool_surface_is_exactly_the_allowlist(self):
        agent = self.registry.agent_for("coding")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.tool_names(), sorted(CODING_TOOLS))

    def test_browser_refused_with_capability_named(self):
        agent = self.registry.agent_for("coding")
        out = agent.tools.call("browser", url="https://example.com")
        self.assertFalse(out.ok)
        msg = str(out.error)
        self.assertIn("not granted to this role", msg)
        self.assertIn("net.browser", msg)

    def test_unknown_role_unchanged(self):
        self.assertIsNone(self.registry.agent_for("vision"))
        self.assertIsNone(self.registry.agent_for("definitely-not-a-role"))

    def test_coding_task_executes_through_default_handler(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        orch = MasterOrchestrator(self.ctx)
        task = Task(name="fix", role="coding",
                    payload={"task": "Create the main module",
                             "root": str(self.fx.root),
                             "max_iterations": 2, "timeout": 20})
        result = orch._default_handler(task)
        self.assertNotEqual(result.get("status"), "no handler")
        self.assertTrue(result["ok"])
        self.assertIn("X = 1", (self.fx.root / "main.py").read_text())

    def test_non_coding_task_still_no_handler(self):
        from nomorals.agents.orchestrator import MasterOrchestrator

        orch = MasterOrchestrator(self.ctx)
        task = Task(name="browse", role="research",
                    payload={"goal": "look something up"})
        result = orch._default_handler(task)
        self.assertEqual(result["status"], "no handler")


class EmbeddingRecallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="phase_c_recall_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ctx = StubContext(ScriptedRouter([]))
        self.agent = CodingAgent(self.ctx)
        self.agent._recall_path = self.tmp / "index.json"

    def test_renamed_variable_variant_recalls_fix(self):
        from nomorals.agents.skills import SkillLibrary

        now = 1.0
        self.ctx.db.execute(
            "INSERT INTO skills (id, name, kind, body, source, created_at,"
            " updated_at) VALUES (?,?,?,?,?,?,?)",
            ("skill-1", "fix-nameerror", "code",
             "pattern: define the variable before use", "coding_session",
             now, now))
        self.agent._error_recall().index(
            "skill-1", "NameError: name 'result' is not defined")
        block = self.agent._recall_error_fixes(
            "NameError: name 'output' is not defined")
        self.assertIn("fix-nameerror", block)
        self.assertIn("similarity", block)

    def test_distill_then_mutated_query(self):
        # journal: one failed attempt, then recovery
        self.ctx.db.execute(
            "INSERT INTO coding_log (id, task, filename, attempt, exit_code,"
            " timed_out, stdout, stderr, code, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("t1", "Fix it", "fix.py", 1, 1, 0, "",
             "Traceback (most recent call last):\n"
             "NameError: name 'result' is not defined", "", 1.0))
        self.ctx.db.execute(
            "INSERT INTO coding_log (id, task, filename, attempt, exit_code,"
            " timed_out, stdout, stderr, code, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("t2", "Fix it", "fix.py", 2, 0, 0, "ok", "", "", 2.0))
        self.agent._distill_session("Fix it", "fix.py", 2)
        block = self.agent._recall_error_fixes(
            "NameError: name 'output' is not defined")
        self.assertIn("fix-nameerror-name-result-is-not", block)
        self.assertIn("KNOWN FIX", block)

    def test_unrelated_error_recalls_nothing(self):
        block = self.agent._recall_error_fixes(
            "completely unrelated http timeout xyzzy plugh")
        self.assertEqual(block, "")


class CriticGateTests(unittest.TestCase):
    """The critic catches a wrong-but-green fix and forces rework."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("calc.py", "def add(a, b):\n    return a - b\n")
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "calc.py", "why": "fix add", "new_file": False}]}),
            # round 1: wrong fix (still not addition) — tests are absent
            # so the loop goes green; only the critic can catch it
            _json_block({"edits": [
                {"old_text": "return a - b", "new_text": "return a * b"}]}),
            # round 2 (critic rework): the right fix
            _json_block({"edits": [
                {"old_text": "return a * b", "new_text": "return a + b"}]}),
        ])
        self.ctx = StubContext(router)
        self.ctx.settings = SimpleNamespace(reasoning_mode="always")
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))
        self.agent._recall_path = (
            Path(tempfile.mkdtemp(prefix="phase_c_gate_")) / "index.json")
        self.addCleanup(shutil.rmtree,
                        self.agent._recall_path.parent, True)

    def test_critic_forces_rework_iteration(self):
        gate = {"flaws": ["the diff multiplies instead of adding"]}

        def review(context, text, *, focus="", max_calls=None):
            if focus == _DIFF_REVIEW_FOCUS:
                flaws = list(gate["flaws"])
                gate["flaws"] = []  # pass on the second gate round
                return flaws
            return []

        with mock.patch("nomorals.agents.reasoning.review_text",
                        side_effect=review):
            result = self.agent.run("Fix add() to add", max_iterations=5,
                                    timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        body = (self.fx.root / "calc.py").read_text()
        self.assertIn("return a + b", body)
        self.assertEqual(result.review["rounds"], 1)
        rows = self.ctx.db.query(
            "SELECT stderr FROM coding_log WHERE filename=?", ("calc.py",))
        self.assertTrue(
            any("code-review round 1 objections" in (r["stderr"] or "")
                for r in rows),
            "no critic rework round found in the journal")


class CriticBoundTests(unittest.TestCase):
    """An always-objecting critic terminates after 2 rounds — no hang."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("f.py", "V = 1\n")
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "f.py", "why": "bump", "new_file": False}]}),
            _json_block({"edits": [
                {"old_text": "V = 1", "new_text": "V = 2"}]}),
            _json_block({"edits": [
                {"old_text": "V = 2", "new_text": "V = 3"}]}),
        ])
        self.ctx = StubContext(router)
        self.ctx.settings = SimpleNamespace(reasoning_mode="always")
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))
        self.agent._recall_path = (
            Path(tempfile.mkdtemp(prefix="phase_c_bound_")) / "index.json")
        self.addCleanup(shutil.rmtree,
                        self.agent._recall_path.parent, True)
        self.router = router

    def test_two_rounds_then_stop_with_objections(self):
        with mock.patch("nomorals.agents.reasoning.review_text",
                        side_effect=_gate_only_review({"flaws": ["nope"]})):
            result = self.agent.run("Bump V", max_iterations=6, timeout=30)
        self.assertTrue(result.ok)
        self.assertFalse(result.review["passed"])
        self.assertEqual(result.review["rounds"], 2)
        self.assertGreaterEqual(len(result.review["objections"]), 2)
        # plan + round-1 edits + round-2 (rework) edits: no third rework
        self.assertEqual(self.router.calls, 3)


class CodeReviewCLITests(unittest.TestCase):
    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git not available")
        self.fx = FixtureRepo(with_git=True)
        self.addCleanup(self.fx.cleanup)
        self.fx.write("app.py", "X = 1\n")
        self.fx.commit_all()
        (self.fx.root / "app.py").write_text("X = 2\n")

    def test_review_shows_diff_verdict_and_commit_prompt(self):
        from nomorals import cli

        ctx = SimpleNamespace(
            db=None, router=None,
            settings=SimpleNamespace(reasoning_mode="always"))
        args = SimpleNamespace(root=str(self.fx.root), json=False)
        buf = io.StringIO()
        with mock.patch("nomorals.agents.reasoning.review_text",
                        return_value=["suspicious change"]), \
             redirect_stdout(buf):
            rc = cli._cmd_code_review(args, ctx, None)
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("X = 2", out)            # the diff
        self.assertIn("VERDICT: FAIL", out)    # the critic verdict
        self.assertIn("suspicious change", out)
        self.assertIn("Commit these changes?", out)  # explicit commit prompt


if __name__ == "__main__":
    unittest.main()
