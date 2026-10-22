"""Ceiling fix: verify/accept execution budgets derive from the run budget.

The defect: ``CodingAgent.run`` takes a per-operation ``timeout``
budget from its caller (mission bridge default 300s, chat /code 120s,
projects 60s, loop default 60s — all payload/caller overridable), but
two fixed clamps silently overrode any budget above them:

- every foreground verify call derived its pytest timeout as
  ``min(max(timeout * 5.0, 60.0), 600.0)`` — a run budgeted at 300s
  got 600s of verify time, never the 1500s its budget implies;
- the accept path built ``SandboxLimits(cpu_seconds=int(min(
  max(timeout, 5.0), 600.0)))`` while the sandbox wall clock was the
  raw ``timeout`` — a CPU-bound accept command was killed at 600
  CPU-seconds even when the caller had budgeted more.

The fix removes the 600 caps and keeps the derivations: verify is
``max(timeout * 5.0, 60.0)`` (x5 derivation, 60s floor) and accept
``cpu_seconds`` is ``int(max(timeout, 5.0))`` (5s floor), so the CPU
allowance always covers the caller's wall budget.  No caller default
moved.  Boundary: for ``timeout <= 120`` every computed value is
identical to the pre-fix code — only budgets whose derived verify
time exceeded 600 change.

These tests pin that contract:

(a) a run with timeout=300 drives verify calls at 1500, not 600;
(b) timeout=60 and timeout=120 produce exactly the pre-fix values
    (300 / 600), and the 60s verify floor still holds below them;
(c) the accept path passes cpu_seconds == 900 for timeout=900;
(d) the accept path passes cpu_seconds == 300 for timeout=300.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coding import CodingAgent
from nomorals.tools import pytest_runner, shell


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


def _json_block(payload):
    return "```json\n" + json.dumps(payload) + "\n```"


_PLAN = _json_block({"files": [
    {"path": "widget.py", "why": "test", "new_file": False}]})
_EDIT = _json_block({"edits": [
    {"old_text": "VALUE = 1", "new_text": "VALUE = 2"}]})

_GREEN = {
    "ok": True, "runner": "pytest", "passed": 1, "failed": [],
    "errors": 0, "ran": 1, "nothing_ran": False, "seconds": 0.1,
    "note": None,
}


class ScriptedRouter:
    """Deterministic stand-in for the LLM: canned responses in order."""

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


@unittest.skipUnless(shutil.which("git"), "git not available")
class VerifyTimeoutTests(unittest.TestCase):
    """(a) + (b): the verify derivation follows the run's budget."""

    def _repo(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="pass_ceiling_"))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        _git("init", "-q", cwd=root)
        _git("config", "user.email", "t@example.com", cwd=root)
        _git("config", "user.name", "T", cwd=root)
        _write(root, "widget.py", "VALUE = 1\n")
        _write(root, "tests/__init__.py", "")
        _write(root, "tests/test_behavior.py",
               "import widget\n\n\ndef test_value():\n"
               "    assert widget.VALUE == 2\n")
        _git("add", ".", cwd=root)
        _git("commit", "-qm", "init", cwd=root)
        return root

    def _verify_timeouts(self, timeout: float) -> list[float]:
        """Run the loop once with a spied run_tests; return the
        timeout of every verify call the loop made.

        Pass #68 fixture note: the loop now makes TWO verify calls
        in this single-step green run — the step's isolated check
        and the round verify.  Both derive from the run budget by
        the same rule (that derivation is what this class pins), so
        the assertions below pin every observed timeout to the
        derived value instead of the old single-call list."""
        root = self._repo()
        # pass #67 fixture note: the leading "{}" is the run-start
        # brief's slot — the brief now fires for every task (it used
        # to be length-gated off for this short one); "{}" carries no
        # objective, so the brief takes its heuristic floor and the
        # task text (and this script) flow exactly as before.
        agent = CodingAgent(
            StubContext(ScriptedRouter(["{}", _PLAN, _EDIT])),
            root=str(root))
        seen: list[float] = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("timeout"))
            return dict(_GREEN)

        with mock.patch.object(pytest_runner, "run_tests", spy):
            result = agent.run("Fix widget value", max_iterations=3,
                               timeout=timeout)
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertEqual(result.verification, "tests")
        self.assertTrue(seen, "the loop never called the test runner")
        return seen

    def test_a_timeout_300_drives_verify_at_1500_not_600(self):
        seen = self._verify_timeouts(300.0)
        self.assertEqual(seen, [1500.0] * len(seen))

    def test_b_timeout_60_and_120_match_pre_fix_values(self):
        # Pre-fix: min(max(60*5, 60), 600) == 300 and
        # min(max(120*5, 60), 600) == 600 — defaults unmoved.
        # (Pass #68: [derived] * len(seen) — the per-step check is
        # a second call, held to the same derivation; see the
        # _verify_timeouts docstring.)
        seen60 = self._verify_timeouts(60.0)
        self.assertEqual(seen60, [300.0] * len(seen60))
        seen120 = self._verify_timeouts(120.0)
        self.assertEqual(seen120, [600.0] * len(seen120))

    def test_b_verify_floor_60s_still_holds(self):
        # max(10*5, 60) == 60 before and after the fix.
        seen = self._verify_timeouts(10.0)
        self.assertEqual(seen, [60.0] * len(seen))


class AcceptCpuTests(unittest.TestCase):
    """(c) + (d): the accept path's CPU allowance covers the budget."""

    def _accept_call(self, timeout: float) -> dict:
        agent = CodingAgent(StubContext(ScriptedRouter([])))
        captured: dict = {}

        def spy(command, **kwargs):
            captured.update(kwargs)
            captured["command"] = command
            return {"exit_code": 0, "stdout": "ok", "stderr": "",
                    "timed_out": False, "seconds": 0.0}

        with mock.patch.object(shell, "run_sandboxed", spy):
            agent._run("echo ok", Path(tempfile.mkdtemp()), timeout)
        self.assertIn("limits", captured)
        return captured

    def test_c_accept_cpu_seconds_900_not_clamped_to_600(self):
        call = self._accept_call(900.0)
        self.assertEqual(call["timeout"], 900.0)
        self.assertEqual(call["limits"].cpu_seconds, 900)

    def test_d_accept_cpu_seconds_300_matches_budget(self):
        call = self._accept_call(300.0)
        self.assertEqual(call["timeout"], 300.0)
        self.assertEqual(call["limits"].cpu_seconds, 300)

    def test_accept_cpu_midrange_and_floor_unchanged(self):
        # Pre-fix values, still the values: min(max(t, 5), 600) and
        # max(t, 5) agree for every t <= 600, floor included.
        self.assertEqual(self._accept_call(120.0)["limits"].cpu_seconds,
                         120)
        self.assertEqual(self._accept_call(2.0)["limits"].cpu_seconds, 5)


if __name__ == "__main__":
    unittest.main()
