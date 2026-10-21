"""Pass #63 (W5): the diff reviewer is always on for applied diffs.

Pinned here:
- review fires for a SHORT task with reasoning mode OFF — the exact
  pre-#63 hole ("fix the parser bug" disabled review + the gate);
- depth scales with the DIFF (size / risk signals), never the prompt:
  the same diff gets the same depth under a 9-char task and a 900-char
  one, and reasoning mode can only raise the floor, never disable;
- a review that could not run is reported review-unavailable in the
  result (available=False, passed=False, notes) — never a silent
  clean pass;
- gate behavior on findings is unchanged (objections force rework).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coding import (
    CodingAgent, DiffReview, _DIFF_REVIEW_FOCUS, _diff_metrics,
    _review_tier, _unified_diff)
from nomorals.cmdline.commands.code import _critic_verdict


class ScriptedRouter:
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
    def __init__(self, router, mode="off"):
        self.router = router
        self.db = StubDB()
        self.settings = SimpleNamespace(reasoning_mode=mode)


def _json_block(payload):
    return "```json\n" + json.dumps(payload) + "\n```"


class FixtureRepo:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass63_"))

    def write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _diff(rel, before, after):
    return _unified_diff(before, after, rel)


class RecordingReviewer:
    """review_text double on the post-#63 signature: records every
    call's kwargs and reports itself available."""

    def __init__(self, flaws_by_focus=None):
        self.calls = []
        self._flaws_by_focus = flaws_by_focus or {}

    def __call__(self, context, text, *, focus="", max_calls=None,
                 status=None):
        self.calls.append({"focus": focus, "max_calls": max_calls,
                           "chars": len(text)})
        if status is not None:
            status.update(available=True, error="")
        return list(self._flaws_by_focus.get(focus, []))


class DepthScalingTests(unittest.TestCase):
    """Depth is a function of the diff, not the prompt (outcomes 1-2)."""

    def setUp(self):
        self.ctx = StubContext(ScriptedRouter([]), mode="off")
        self.agent = CodingAgent(self.ctx, root="/tmp")

    def review(self, text, task="fix it"):
        rec = RecordingReviewer()
        with mock.patch("nomorals.agents.reasoning.review_text", rec):
            outcome = self.agent._review_diff(task, text)
        return outcome, rec

    def test_small_diff_short_task_reviewed_focused(self):
        outcome, rec = self.review(_diff("x.py", "A = 1\n", "A = 2\n"))
        self.assertTrue(outcome.available)
        self.assertEqual(outcome.depth, "focused")
        self.assertEqual(outcome.calls, 1)
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0]["max_calls"], 1)  # cheap tier

    def test_large_diff_short_task_reviewed_deep(self):
        big = "\n".join(
            _diff(f"f{i}.py", "v = 0\n", "v = 1\n") for i in range(6))
        outcome, rec = self.review(big, task="fix it")
        self.assertEqual(outcome.depth, "deep")
        # 6 per-file calls (at the cap) + 1 combined call
        self.assertEqual(outcome.calls, 7)
        self.assertEqual(len(rec.calls), 7)

    def test_two_file_diff_is_standard(self):
        two = "\n".join(_diff(f"f{i}.py", "v = 0\n", "v = 1\n")
                        for i in range(2))
        outcome, _ = self.review(two)
        self.assertEqual(outcome.depth, "standard")
        self.assertEqual(outcome.calls, 1)

    def test_prompt_wording_does_not_change_depth(self):
        diff = _diff("x.py", "A = 1\n", "A = 2\n")
        short, _ = self.review(diff, task="fix it")
        long, _ = self.review(
            diff, task=("please carefully architect a comprehensive "
                        "redesign of the whole subsystem and explain "
                        "the trade-offs in depth — " * 12))
        self.assertEqual(short.depth, long.depth)
        self.assertEqual(short.calls, long.calls)

    def test_risky_small_diff_goes_deep_plain_small_diff_does_not(self):
        risky, _ = self.review(_diff("core/config.py", "A=1\n", "A=2\n"))
        self.assertEqual(risky.depth, "deep")
        self.assertIn("sensitive-path", risky.risks)
        plain, _ = self.review(_diff("notes.py", "A=1\n", "A=2\n"))
        self.assertEqual(plain.depth, "focused")
        api, _ = self.review(_diff(
            "lib.py", "x = 1\n", "x = 1\n\ndef new_api():\n    pass\n"))
        self.assertEqual(api.depth, "deep")
        self.assertIn("public-api-change", api.risks)

    def test_deep_per_file_flaws_carry_their_path(self):
        big = "\n".join(
            _diff(f"f{i}.py", "v = 0\n", "v = 1\n") for i in range(5))
        rec = RecordingReviewer()

        def per_file(context, text, *, focus="", max_calls=None,
                     status=None):
            rec.calls.append({"focus": focus})
            if status is not None:
                status.update(available=True, error="")
            if "ONE file" in focus:
                return ["something off in this file"]
            return []

        with mock.patch("nomorals.agents.reasoning.review_text",
                        per_file):
            outcome = self.agent._review_diff("fix it", big)
        self.assertEqual(outcome.depth, "deep")
        self.assertTrue(outcome.flaws)
        self.assertTrue(all(f.startswith("[f") for f in outcome.flaws),
                        outcome.flaws)


class ModeFloorTests(unittest.TestCase):
    """Reasoning mode deepens; it never disables (outcome 2)."""

    def depth(self, mode, text):
        ctx = StubContext(ScriptedRouter([]), mode=mode)
        agent = CodingAgent(ctx, root="/tmp")
        rec = RecordingReviewer()
        with mock.patch("nomorals.agents.reasoning.review_text", rec):
            outcome = agent._review_diff("fix it", text)
        return outcome, rec

    def test_off_still_reviews(self):
        outcome, rec = self.depth("off", _diff("x.py", "A=1\n", "A=2\n"))
        self.assertTrue(outcome.available)
        self.assertEqual(len(rec.calls), 1)

    def test_always_floors_small_diff_at_standard(self):
        outcome, _ = self.depth("always", _diff("x.py", "A=1\n", "A=2\n"))
        self.assertEqual(outcome.depth, "standard")

    def test_always_never_lowers_an_earned_deep(self):
        big = "\n".join(
            _diff(f"f{i}.py", "v = 0\n", "v = 1\n") for i in range(6))
        outcome, _ = self.depth("always", big)
        self.assertEqual(outcome.depth, "deep")


class MetricsTests(unittest.TestCase):
    """The scaling table's inputs are diff-structure facts."""

    def test_file_deletion_is_a_hard_risk(self):
        text = "--- a/old.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
        m = _diff_metrics(text)
        self.assertIn("file-deletion", m["risks"])
        self.assertEqual(m["files"], ["old.py"])
        self.assertEqual(_review_tier(m), "deep")

    def test_heavy_deletion_is_a_hard_risk(self):
        body = "".join(f"-line {i}\n" for i in range(60))
        text = f"--- a/m.py\n+++ b/m.py\n@@ -60,0 +0,0 @@\n{body}"
        m = _diff_metrics(text)
        self.assertIn("heavy-deletion", m["risks"])
        self.assertEqual(_review_tier(m), "deep")

    def test_wide_change_by_lines(self):
        body = "".join(f"+line {i}\n" for i in range(401))
        text = f"--- a/m.py\n+++ b/m.py\n@@ -0,0 +1,401 @@\n{body}"
        self.assertEqual(_review_tier(_diff_metrics(text)), "deep")
        body = "".join(f"+line {i}\n" for i in range(100))
        text = f"--- a/m.py\n+++ b/m.py\n@@ -0,0 +1,100 @@\n{body}"
        self.assertEqual(_review_tier(_diff_metrics(text)), "standard")

    def test_all_edit_is_public_api_change(self):
        m = _diff_metrics(_diff("m.py", "__all__ = ['a']\n",
                                "__all__ = ['a', 'b']\n"))
        self.assertIn("public-api-change", m["risks"])

    def test_lookalike_paths_are_not_sensitive(self):
        for path in ("monkey.py", "keyboard.py", "authors.py",
                     "tokenizer.py"):
            m = _diff_metrics(_diff(path, "A=1\n", "A=2\n"))
            self.assertNotIn("sensitive-path", m["risks"], path)
        m = _diff_metrics(_diff("app/.env", "A=1\n", "A=2\n"))
        self.assertIn("sensitive-path", m["risks"])


class UnavailableReviewerTests(unittest.TestCase):
    """A dead reviewer is reported, never read as clean (outcome 3)."""

    def test_review_diff_reports_unavailable(self):
        ctx = StubContext(ScriptedRouter([]), mode="off")
        agent = CodingAgent(ctx, root="/tmp")

        def dead(context, text, *, focus="", max_calls=None, status=None):
            if status is not None:
                status.update(
                    available=False,
                    error="ToolError: reasoning: empty model response")
            return []

        with mock.patch("nomorals.agents.reasoning.review_text", dead):
            outcome = agent._review_diff("fix it",
                                         _diff("x.py", "A=1\n", "A=2\n"))
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.flaws, [])
        self.assertIn("empty model response", outcome.error)

    def test_pre63_signature_double_still_works(self):
        # Fixtures written before the status channel (phase B/C shape)
        # reject the extra kwarg; their verdict is taken at face value.
        ctx = StubContext(ScriptedRouter([]), mode="off")
        agent = CodingAgent(ctx, root="/tmp")

        def old_style(context, text, *, focus="", max_calls=None):
            return ["a real flaw"]

        with mock.patch("nomorals.agents.reasoning.review_text",
                        old_style):
            outcome = agent._review_diff("fix it",
                                         _diff("x.py", "A=1\n", "A=2\n"))
        self.assertTrue(outcome.available)
        self.assertEqual(outcome.flaws, ["a real flaw"])


class ShortTaskGateTests(unittest.TestCase):
    """End-to-end W5 scenario: 9-char task, reasoning OFF, the gate
    still catches a wrong-but-green fix and forces the rework round."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("calc.py", "def add(a, b):\n    return a - b\n")
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "calc.py", "why": "fix add", "new_file": False}]}),
            _json_block({"edits": [
                {"old_text": "return a - b", "new_text": "return a * b"}]}),
            _json_block({"edits": [
                {"old_text": "return a * b", "new_text": "return a + b"}]}),
        ])
        self.ctx = StubContext(router, mode="off")
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))

    def test_gate_fires_on_short_task_with_reasoning_off(self):
        gate = {"flaws": ["the diff multiplies instead of adding"]}
        calls = {"n": 0}

        def review(context, text, *, focus="", max_calls=None,
                   status=None):
            calls["n"] += 1
            if status is not None:
                status.update(available=True, error="")
            if focus == _DIFF_REVIEW_FOCUS:
                flaws = list(gate["flaws"])
                gate["flaws"] = []
                return flaws
            return []

        with mock.patch("nomorals.agents.reasoning.review_text",
                        side_effect=review):
            result = self.agent.run(
                "Fix add()", max_iterations=5, timeout=30,
                accept='python3 -c "import calc; print(calc.add(2,3))"')
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertIn("return a + b", (self.fx.root / "calc.py").read_text())
        self.assertGreaterEqual(calls["n"], 3)  # per-diff + gate reviews ran
        self.assertEqual(result.review["rounds"], 1)
        self.assertTrue(result.review["passed"])
        self.assertTrue(result.review["available"])
        self.assertIn("depth", result.review)


class RunLevelUnavailableTests(unittest.TestCase):
    """A green run with a dead reviewer ships WITHOUT a clean-review
    claim: available=False, passed=False, and the notes say why."""

    def setUp(self):
        self.fx = FixtureRepo()
        self.addCleanup(self.fx.cleanup)
        self.fx.write("v.py", "V = 1\n")
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "v.py", "why": "bump", "new_file": False}]}),
            _json_block({"edits": [
                {"old_text": "V = 1", "new_text": "V = 2"}]}),
        ])
        self.ctx = StubContext(router, mode="off")
        self.agent = CodingAgent(self.ctx, root=str(self.fx.root))

    def test_unavailable_gate_is_reported_not_clean(self):
        def dead(context, text, *, focus="", max_calls=None, status=None):
            if status is not None:
                status.update(available=False,
                              error="ToolError: model unavailable")
            return []

        with mock.patch("nomorals.agents.reasoning.review_text",
                        side_effect=dead):
            result = self.agent.run(
                "Bump V", max_iterations=3, timeout=30,
                accept='python3 -c "import v; print(v.V)"')
        self.assertTrue(result.ok, f"loop failed: {result.error}")
        self.assertIn("V = 2", (self.fx.root / "v.py").read_text())
        self.assertFalse(result.review["available"])
        self.assertFalse(result.review["passed"])
        self.assertTrue(
            any("unavailable" in n for n in result.review["notes"]),
            result.review)


class CriticVerdictTests(unittest.TestCase):
    """`nm code review`'s critic: honest about unavailability, and no
    longer silenced by mode/length (its canned task string is 28
    chars — under the old gate it never ran in auto mode)."""

    def ctx(self):
        return SimpleNamespace(
            db=StubDB(), router=ScriptedRouter([]),
            settings=SimpleNamespace(reasoning_mode="off"))

    def test_unavailable_critic_is_not_a_pass(self):
        def dead(context, text, *, focus="", max_calls=None, status=None):
            if status is not None:
                status.update(available=False, error="no model")
            return []

        with mock.patch("nomorals.agents.reasoning.review_text",
                        side_effect=dead):
            verdict, flaws = _critic_verdict(
                self.ctx(), "--- a/x.py\n+++ b/x.py\n@@\n-A=1\n+A=2\n")
        self.assertTrue(verdict.startswith("critic unavailable"), verdict)
        self.assertEqual(flaws, [])

    def test_critic_fires_with_reasoning_off(self):
        with mock.patch("nomorals.agents.reasoning.review_text",
                        return_value=["suspicious change"]):
            verdict, flaws = _critic_verdict(
                self.ctx(), "--- a/x.py\n+++ b/x.py\n@@\n-A=1\n+A=2\n")
        self.assertIn("VERDICT: FAIL", verdict)
        self.assertEqual(flaws, ["suspicious change"])


if __name__ == "__main__":
    unittest.main()
