"""The coding agent: draft -> run in the real sandbox -> fix, iteratively.

The model is scripted (fake router), but the filesystem tool and the sandbox
are the real ones: files actually land in the workspace and code actually
executes under ``run_sandboxed``.
"""

from __future__ import annotations

import tempfile
import unittest

from nomorals.agents.context import build_context
from nomorals.agents.coding import CodingAgent, CodingResult, extract_code_block
from nomorals.core.config import Settings
from nomorals.llm.base import LLMResponse


class ExtractCodeBlockTest(unittest.TestCase):
    def test_fenced_python_block(self) -> None:
        text = "Here you go:\n```python\nprint('hi')\n```\nDone."
        self.assertEqual(extract_code_block(text), "print('hi')\n")

    def test_inline_fence(self) -> None:
        # Lazy model shape: code starts on the fence line, separated by a space.
        self.assertEqual(extract_code_block("result: ``` x=1 ```"), "x=1\n")

    def test_no_block_is_empty(self) -> None:
        self.assertEqual(extract_code_block("I refuse to write code."), "")
        self.assertEqual(extract_code_block(""), "")


class _ScriptedRouter:
    """Pops one canned reply per chat() call — the 'model' for the loop."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, params=None, **kw) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.replies.pop(0), model="scripted-7b")


class CodingAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-code-")
        self.context = build_context(
            Settings(home=self.tmp.name),
            with_executor=False,
            with_router=False,
            with_memory=False,
            with_tools=False,
        )
        # The scripted router counts exact chat() calls; these tests
        # exercise the draft→sandbox→fix loop itself, so the always-on
        # reasoning draft review (its own extra model call) is pinned off.
        # The review path is covered by the reasoning test suites.
        self.context.settings.reasoning_mode = "off"
        self.context.__enter__()

    def tearDown(self) -> None:
        try:
            self.context.__exit__(None, None, None)
        finally:
            self.tmp.cleanup()

    def test_fixes_itself_after_a_real_sandbox_failure(self) -> None:
        router = _ScriptedRouter([
            # Attempt 1: broken code — the sandbox will run it and fail.
            "```python\nprint('attempt one')\nraise ValueError('boom')\n```",
            # Attempt 2: the model reads the real stderr and fixes it.
            "```python\nprint('fixed, value =', 6 * 7)\n```",
        ])
        self.context.router = router
        agent = CodingAgent(self.context)

        result = agent.run("print the answer to life", filename="main.py", max_iterations=3)

        self.assertIsInstance(result, CodingResult)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(router.calls, 2)
        self.assertIn("fixed, value = 42", result.output)

        # The file is the FINAL working version, inside the workspace.
        from nomorals.tools.filesystem import safe_path

        path = safe_path(self.context, "main.py")
        self.assertTrue(path.exists())
        self.assertIn("6 * 7", path.read_text())  # the source, not the output

        # Both iterations journaled, with the real exit codes.
        rows = self.context.db.query(
            "SELECT attempt, exit_code FROM coding_log ORDER BY attempt"
        )
        self.assertEqual([r["attempt"] for r in rows], [1, 2])
        self.assertEqual([r["exit_code"] for r in rows], [1, 0])

    def test_custom_acceptance_command(self) -> None:
        router = _ScriptedRouter([
            "```python\nassert 1 + 1 == 2, 'math broke'\nprint('tests green')\n```",
        ])
        self.context.router = router
        result = CodingAgent(self.context).run(
            "a file that passes its own assert",
            filename="check.py",
            accept="python3 check.py",
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("tests green", result.output)

    def test_gives_up_after_max_iterations(self) -> None:
        broken = "```python\nraise RuntimeError('nope')\n```"
        router = _ScriptedRouter([broken] * 3)
        self.context.router = router
        result = CodingAgent(self.context).run(
            "impossible task", filename="main.py", max_iterations=2
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.iterations, 2)
        self.assertIn("still failing", result.error)
        self.assertIn("nope", result.error)

    def test_model_refusal_fails_fast(self) -> None:
        self.context.router = _ScriptedRouter(["I cannot write code for this task."])
        result = CodingAgent(self.context).run("anything", max_iterations=5)
        self.assertFalse(result.ok)
        self.assertEqual(result.iterations, 0)
        self.assertIn("no code block", result.error)

    def test_sessions_lists_journal_newest_first(self) -> None:
        self.context.router = _ScriptedRouter(["```python\nprint('ok')\n```"])
        CodingAgent(self.context).run("task A", filename="a.py")
        sessions = CodingAgent(self.context).sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["filename"], "a.py")
        self.assertEqual(sessions[0]["attempt"], 1)


if __name__ == "__main__":
    unittest.main()
