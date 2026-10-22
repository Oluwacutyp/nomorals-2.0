"""Pass #58: zero executed tests must never certify a coding run green.

The defect: ``pytest_runner.run_tests`` reports an honest skip
(``ok=True, passed=0, "nothing ran"``) when its targeted selection
matches no tests, and ``CodingAgent.run`` read that as green — a change
could ship "verified" with zero tests executed.  These tests pin the
new contract:

- the runner result carries an explicit ``ran`` / ``nothing_ran`` signal;
- the loop makes ONE widened attempt (the repo's test dirs wholesale)
  when targeted selection matches nothing;
- still nothing → the run fails honestly as UNVERIFIED, never green;
- the background-tests path has the same rule;
- an explicit accept command still decides green/red by its own rules;
- a workdir with no tests at all is disclosed as a smoke run
  (``verification == "smoke"``), not dressed up as tested.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.coding import CodingAgent
from nomorals.tools import pytest_runner


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


class FixtureRepo:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass58_"))
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


def _json_block(payload):
    return "```json\n" + json.dumps(payload) + "\n```"


def _plan(*paths):
    return _json_block({"files": [
        {"path": p, "why": "test", "new_file": False} for p in paths]})


_EDIT = _json_block({"edits": [
    {"old_text": "VALUE = 1", "new_text": "VALUE = 2"}]})

FAIL_ACCEPT = f'"{sys.executable}" -c "import sys; sys.exit(1)"'
OK_ACCEPT = f'"{sys.executable}" -c "print(\'accept ok\')"'


@unittest.skipUnless(shutil.which("git"), "git not available")
class RunnerSignalTests(unittest.TestCase):
    """The runner result must explicitly say when nothing ran."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("foo.py", "X = 1\n")
        self.fx.write("tests/__init__.py", "")
        self.fx.write("tests/test_foo.py",
                      "import foo\n\n\ndef test_x():\n    assert foo.X == 1\n")
        self.fx.commit_all()

    def test_no_changes_is_nothing_ran(self):
        res = pytest_runner.run_tests(repo=str(self.fx.root))
        self.assertTrue(res["ok"])  # skip semantics unchanged for callers
        self.assertEqual(res["ran"], 0)
        self.assertTrue(res["nothing_ran"])

    def test_targeted_run_reports_ran_count(self):
        (self.fx.root / "foo.py").write_text("X = 1  # touched\n")
        res = pytest_runner.run_tests(repo=str(self.fx.root), timeout=60)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["ran"], 1)
        self.assertFalse(res["nothing_ran"])

    def test_empty_collection_is_nothing_ran(self):
        self.fx.write("tests/test_foo.py",
                      '"""No tests in here."""\n\n\ndef helper():\n'
                      "    return 1\n")
        self.fx.commit_all("empty the suite")
        res = pytest_runner.run_tests(paths=["tests"],
                                      repo=str(self.fx.root), timeout=60)
        self.assertEqual(res["ran"], 0)
        self.assertTrue(res["nothing_ran"])


@unittest.skipUnless(shutil.which("git"), "git not available")
class ZeroTestGreenLoopTests(unittest.TestCase):
    """Loop-level contract: nothing ran ⇒ not green, unless the widened
    attempt or an explicit accept command really verifies the change."""

    def _agent(self, fx, responses):
        # pass #67 fixture note: the run-start brief now fires for
        # EVERY task (it used to be length-gated off for these short
        # ones), so each script gains a leading "{}" — a model reply
        # with no objective, which sends the brief to its heuristic
        # floor and leaves the task text (and this script) unchanged.
        router = ScriptedRouter(["{}"] + list(responses))
        return CodingAgent(StubContext(router), root=str(fx.root))

    def _repo_with_uncollectable_tests(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("widget.py", "VALUE = 1\n")
        fx.write("tests/__init__.py", "")
        # a test-named file with zero collectable tests: targeted
        # selection cannot match it to widget.py AND the widened
        # directory run collects nothing.
        fx.write("tests/test_empty.py",
                 '"""Helpers only."""\n\n\ndef helper():\n    return 1\n')
        fx.commit_all()
        return fx

    def test_nothing_ran_fails_honestly_unverified(self):
        fx = self._repo_with_uncollectable_tests()
        agent = self._agent(fx, [_plan("widget.py"), _EDIT, _EDIT])
        result = agent.run("Fix widget value", max_iterations=2,
                           timeout=30)
        self.assertFalse(result.ok)
        self.assertEqual(result.verification, "unverified")
        self.assertIn("UNVERIFIED", result.error)
        self.assertIn("no tests ran", result.error)
        self.assertEqual(result.to_dict()["verification"], "unverified")
        # the failed run rolled the unverified change back
        self.assertEqual((fx.root / "widget.py").read_text(), "VALUE = 1\n")

    def test_background_nothing_ran_is_not_green(self):
        fx = self._repo_with_uncollectable_tests()
        agent = self._agent(fx, [_plan("widget.py"), _EDIT])
        result = agent.run("Fix widget value", max_iterations=1,
                           timeout=30, background_tests=True,
                           test_cap_seconds=60)
        self.assertFalse(result.ok)
        self.assertEqual(result.verification, "unverified")
        self.assertIn("UNVERIFIED", result.error)

    def test_widened_attempt_runs_suite_and_goes_green(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("widget.py", "VALUE = 1\n")
        fx.write("tests/__init__.py", "")
        # stem "test_other" does NOT contain "widget": targeted selection
        # matches nothing, but the widened directory run executes it.
        fx.write("tests/test_other.py",
                 "import widget\n\n\ndef test_value():\n"
                 "    assert widget.VALUE == 2\n")
        fx.commit_all()
        agent = self._agent(fx, [_plan("widget.py"), _EDIT])
        result = agent.run("Fix widget value", max_iterations=3,
                           timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertEqual(result.verification, "tests")
        self.assertEqual((fx.root / "widget.py").read_text(), "VALUE = 2\n")

    def test_targeted_match_still_goes_green(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("widget.py", "VALUE = 1\n")
        fx.write("tests/__init__.py", "")
        fx.write("tests/test_widget.py",
                 "import widget\n\n\ndef test_value():\n"
                 "    assert widget.VALUE == 2\n")
        fx.commit_all()
        agent = self._agent(fx, [_plan("widget.py"), _EDIT])
        result = agent.run("Fix widget value", max_iterations=3,
                           timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertEqual(result.verification, "tests")

    def test_accept_command_decides_when_present(self):
        fx = self._repo_with_uncollectable_tests()
        agent = self._agent(fx, [_plan("widget.py"), _EDIT])
        ok_result = agent.run("Fix widget value", accept=OK_ACCEPT,
                              max_iterations=1, timeout=30)
        self.assertTrue(ok_result.ok, f"loop failed: {ok_result.error}")
        self.assertEqual(ok_result.verification, "accept")

        fx2 = self._repo_with_uncollectable_tests()
        agent2 = self._agent(fx2, [_plan("widget.py"), _EDIT])
        bad_result = agent2.run("Fix widget value", accept=FAIL_ACCEPT,
                                max_iterations=1, timeout=30)
        self.assertFalse(bad_result.ok)
        self.assertEqual(bad_result.verification, "accept")


class SmokeDisclosureTests(unittest.TestCase):
    """No tests anywhere + no accept command: the run may still succeed
    by executing the file, but it must disclose itself as a smoke run."""

    def test_testless_generation_discloses_smoke(self):
        root = Path(tempfile.mkdtemp(prefix="pass58_smoke_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        router = ScriptedRouter([
            "{}",  # pass #67: the run-start brief's slot (see above)
            _json_block({"files": [
                {"path": "main.py", "why": "new", "new_file": True}]}),
            "```python\nprint('hello from main')\n```\n",
        ])
        agent = CodingAgent(StubContext(router), root=str(root))
        result = agent.run("create main", filename="main.py",
                           max_iterations=2, timeout=30)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertEqual(result.verification, "smoke")
        self.assertIn("smoke run only", result.output)
        self.assertTrue((root / "main.py").is_file())


if __name__ == "__main__":
    unittest.main()
