"""Acceptance tests for audit Phase B (prompt 17): the rebuilt coding loop.

- 3-file fixture bug fixed; untouched files' hashes identical before/after
- run_tests(changed_only=True) returns the failing test id + snippet
- a lint violation produces a lint-fix iteration visible in the journal
- the harsh reviewer runs on every applied diff (counted for a 2-file edit)
- `nm code test --changed` routes to the runner end to end
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coding import CodingAgent, _parse_edits_block
from nomorals.tools import pytest_runner
from nomorals.tools import lint as lint_mod

REPO = Path(__file__).resolve().parent.parent


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class ScriptedRouter:
    """Deterministic stand-in for the LLM: canned responses in call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, params=None):
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")


class StubDB:
    """In-memory coding_log; dict rows like the real Database."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE coding_log (id TEXT, task TEXT, filename TEXT,"
            " attempt INT, exit_code INT, timed_out INT, stdout TEXT,"
            " stderr TEXT, code TEXT, created_at REAL)")

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


class StubContext:
    def __init__(self, router):
        self.router = router
        self.db = StubDB()
        self.settings = SimpleNamespace(reasoning_mode="off")


def _json_block(payload):
    return "```json\n" + json.dumps(payload) + "\n```"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixtureRepo:
    """A throwaway project dir with an optional git repo + tests."""

    def __init__(self, with_git=False):
        self.root = Path(tempfile.mkdtemp(prefix="phase_b_"))
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


class ThreeFileFixTests(unittest.TestCase):
    """A real 3-file bug is fixed; untouched files are byte-identical."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("a.py", "OFFSET = 1\n")
        self.fx.write("b.py", "from a import OFFSET\n\n\ndef total():\n    return OFFSET + 5\n")
        self.fx.write("c.py", "UNTOUCHED = True\n")
        self.fx.write("d.py", "ALSO_UNTOUCHED = 42\n")
        self.fx.write("tests/__init__.py", "")
        self.fx.write("tests/test_ab.py",
                      "import a\nimport b\n\n\n"
                      "def test_offset():\n    assert a.OFFSET == 10\n\n\n"
                      "def test_total():\n    assert b.total() == 25\n")
        self.hash_before = {f: _hash(self.fx.root / f) for f in ("c.py", "d.py")}

        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "a.py", "why": "OFFSET is wrong", "new_file": False},
                {"path": "b.py", "why": "total formula is wrong", "new_file": False},
            ]}),
            _json_block({"edits": [
                {"old_text": "OFFSET = 1", "new_text": "OFFSET = 10"}]}),
            _json_block({"edits": [
                {"old_text": "return OFFSET + 5",
                 "new_text": "return OFFSET * 2 + 5"}]}),
        ])
        self.agent = CodingAgent(StubContext(router), root=str(self.fx.root))

    def test_three_file_bug_fixed_and_untouched_hashes_identical(self):
        result = self.agent.run("Fix the offset and total bugs",
                                max_iterations=4, timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertIn("OFFSET = 10", (self.fx.root / "a.py").read_text())
        self.assertIn("return OFFSET * 2 + 5", (self.fx.root / "b.py").read_text())
        for f, h in self.hash_before.items():
            self.assertEqual(_hash(self.fx.root / f), h,
                             f"{f} was rewritten but never in the plan")
        self.assertEqual(sorted(result.files), ["a.py", "b.py"])


class ChangedOnlyTests(unittest.TestCase):
    """changed_only returns the failing test id + snippet, not the suite."""

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git not available")
        self.fx = FixtureRepo(with_git=True)
        self.addCleanup(self.fx.cleanup)
        self.fx.write("mod.py", "VALUE = 1\n")
        self.fx.write("other.py", "VALUE = 2\n")
        self.fx.write("tests/__init__.py", "")
        self.fx.write("tests/test_mod.py",
                      "import mod\n\n\ndef test_value():\n    assert mod.VALUE == 99\n")
        # this one must NOT run under changed_only
        self.fx.write("tests/test_other.py",
                      "import other\n\n\ndef test_other():\n    assert other.VALUE == 2\n")
        self.fx.commit_all()

    def test_changed_only_selects_and_reports(self):
        (self.fx.root / "mod.py").write_text("VALUE = 2\n")
        selected = pytest_runner.select_changed_tests(str(self.fx.root))
        self.assertEqual(selected, ["tests/test_mod.py"])
        result = pytest_runner.run_tests(changed_only=True,
                                         repo=str(self.fx.root), timeout=60)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["failed"]), 1)
        fail = result["failed"][0]
        self.assertIn("test_mod", fail["test_id"])
        self.assertTrue(fail["error_snippet"].strip())
        # the passing unrelated test file was never selected
        self.assertNotIn("test_other", result["selected"])


class LintGateTests(unittest.TestCase):
    """A lint violation becomes a lint-fix iteration in the journal."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("fix.py", "import os\n\n\nVALUE = 1\n")
        self.fx.write("tests/__init__.py", "")
        self.fx.write("tests/test_fix.py",
                      "import fix\n\n\ndef test_value():\n    assert fix.VALUE == 2\n")
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "fix.py", "why": "wrong value", "new_file": False}]}),
            # round 1: fixes the value but leaves the unused import
            _json_block({"edits": [
                {"old_text": "VALUE = 1", "new_text": "VALUE = 2"}]}),
            # round 2 (lint feedback): removes the unused import
            _json_block({"edits": [
                {"old_text": "import os\n\n\n", "new_text": ""}]}),
        ])
        self.ctx = StubContext(router)
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))

    def test_lint_fix_iteration_is_journaled(self):
        calls = {"n": 0}

        def fake_lint(paths, repo=None, timeout=120.0):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"ok": False, "ruff_installed": True,
                        "violations": [{"file": "fix.py", "line": 1, "col": 1,
                                        "code": "F401",
                                        "message": "`os` imported but unused"}],
                        "format_ok": True, "seconds": 0.01}
            return {"ok": True, "ruff_installed": True, "violations": [],
                    "format_ok": True, "seconds": 0.01}

        with mock.patch.object(lint_mod, "lint", side_effect=fake_lint):
            result = self.agent.run("Fix the value", max_iterations=5,
                                    timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertNotIn("import os", (self.fx.root / "fix.py").read_text())
        rows = self.ctx.db.query(
            "SELECT stderr FROM coding_log WHERE filename=? ORDER BY attempt",
            ("fix.py",))
        lint_rows = [r for r in rows if "ruff lint failed" in (r["stderr"] or "")]
        self.assertTrue(lint_rows,
                        "no lint-fix iteration found in the session journal")


class ReviewEveryDiffTests(unittest.TestCase):
    """The harsh reviewer runs once per applied diff (2-file edit = 2)."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("one.py", "A = 1\n")
        self.fx.write("two.py", "B = 1\n")
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "one.py", "why": "bump A", "new_file": False},
                {"path": "two.py", "why": "bump B", "new_file": False},
            ]}),
            _json_block({"edits": [
                {"old_text": "A = 1", "new_text": "A = 2"}]}),
            _json_block({"edits": [
                {"old_text": "B = 1", "new_text": "B = 2"}]}),
        ])
        self.ctx = StubContext(router)
        self.ctx.settings = SimpleNamespace(reasoning_mode="always")
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))

    def test_reviewer_called_per_diff(self):
        with mock.patch("nomorals.agents.reasoning.review_text",
                        return_value=[]) as rt, \
             mock.patch.object(CodingAgent, "_review_flaws",
                               wraps=self.agent._review_flaws) as counted:
            result = self.agent.run("Bump both constants", max_iterations=3,
                                    timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertTrue(rt.called)
        self.assertEqual(counted.call_count, 2,
                         f"expected one review per applied diff, got {counted.call_count}")


class ParseEditsTests(unittest.TestCase):
    def test_valid_block(self):
        edits = _parse_edits_block(_json_block(
            {"edits": [{"old_text": "a", "new_text": "b"}]}))
        self.assertEqual(edits, [{"old_text": "a", "new_text": "b"}])

    def test_empty_edits_ok(self):
        self.assertEqual(_parse_edits_block(_json_block({"edits": []})), [])

    def test_garbage_is_none(self):
        self.assertIsNone(_parse_edits_block("no block here"))
        self.assertIsNone(_parse_edits_block(_json_block({"nope": 1})))
        self.assertIsNone(_parse_edits_block(
            _json_block({"edits": [{"old_text": "a"}]})))


class LintToolTests(unittest.TestCase):
    def test_ruff_absent_is_honest(self):
        with mock.patch.object(lint_mod, "ruff_available", return_value=False):
            result = lint_mod.lint(["x.py"])
        self.assertFalse(result["ok"])
        self.assertFalse(result["ruff_installed"])
        self.assertIn("not installed", result["message"])


class CodeTestCLITests(unittest.TestCase):
    """`nm code test --changed` routes to the runner end to end."""

    def test_changed_flag_reaches_runner(self):
        from nomorals import cli

        tmp = Path(tempfile.mkdtemp(prefix="phase_b_cli_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        args = SimpleNamespace(changed=True, root=str(tmp), json=False)
        fake = {"ok": True, "passed": 3, "failed": [], "errors": 0,
                "seconds": 0.2, "runner": "pytest", "selected": ["tests/test_x.py"]}
        with mock.patch("nomorals.tools.pytest_runner.run_tests",
                        return_value=fake) as rt:
            rc = cli._cmd_code_test(args, object())
        self.assertEqual(rc, 0)
        _, kwargs = rt.call_args
        self.assertTrue(kwargs["changed_only"])
        self.assertEqual(kwargs["repo"], str(tmp))

    def test_changed_only_schema_on_real_repo(self):
        # acceptance: works end to end on the Devon repo itself
        result = pytest_runner.run_tests(changed_only=True, repo=str(REPO),
                                         timeout=120)
        for key in ("ok", "passed", "failed", "errors", "seconds"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
