"""Failing-first tests: DevonAgent._tool_run_tests must be honest + chunked.

Devon's self-report ran the full suite with a 120s cap and concluded the
tests "hang". The real shape: full suite ~815 files / ~85 min, so the
120s cap (hard-capped at 300s) guarantees a timeout that looks like a hang,
with no partial results and no way to continue. These tests pin the fix:

* no args → targeted run (tests touching changed files), NOT the full suite;
* full-suite runs split into chunks, with per-chunk progress;
* a timeout says the suite was STILL RUNNING and how to resume — never a
  bare "timed out" that reads as a hang;
* long timeouts are accepted (cap raised for full runs).
"""

from __future__ import annotations

import subprocess
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import devon as devon_mod
from nomorals.agents.devon import DevonAgent


class FakeContext:
    def __init__(self) -> None:
        self.settings = SimpleNamespace()
        self.extras = {}
        self.memory = None
        self.router = None


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                   check=True, timeout=30)


def _make_repo(tmp: Path) -> Path:
    root = tmp / "repo"
    (root / "tests").mkdir(parents=True)
    _git("init", cwd=root)
    _git("config", "user.email", "t@t", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    (root / "foo.py").write_text("X = 1\n", encoding="utf-8")
    (root / "tests" / "test_foo.py").write_text(
        textwrap.dedent("""
            import unittest

            class FooTests(unittest.TestCase):
                def test_passes(self) -> None:
                    self.assertTrue(True)
        """), encoding="utf-8")
    (root / "tests" / "test_slow.py").write_text(
        textwrap.dedent("""
            import time
            import unittest

            class SlowTests(unittest.TestCase):
                def test_sleeps(self) -> None:
                    time.sleep(30)
        """), encoding="utf-8")
    _git("add", "-A", cwd=root)
    _git("commit", "-m", "init", cwd=root)
    (root / "foo.py").write_text("X = 2\n", encoding="utf-8")
    return root


class DevonRunTestsTests(unittest.TestCase):
    def _agent(self, root: Path) -> DevonAgent:
        return DevonAgent(FakeContext(), repo_root=root)

    def _run_with_fake(self, agent, args, exc=None):
        calls: list = []
        real_run = subprocess.run

        def fake(cmd, **kwargs):
            # Only intercept the unittest invocation; git probes and
            # everything else go through the real subprocess.run.
            if "-m" in cmd and "unittest" in cmd:
                calls.append({"cmd": list(cmd),
                              "timeout": kwargs.get("timeout")})
                if exc is not None:
                    raise exc
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="Ran 1 test in 0.001s\n\nOK\n", stderr="")
            return real_run(cmd, **kwargs)

        with mock.patch.object(devon_mod.subprocess, "run", fake):
            out = agent._tool_run_tests(args)
        return out, calls

    def test_default_is_targeted_not_full_suite(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = _make_repo(Path(td))
            out, calls = self._run_with_fake(self._agent(root), {})
            joined = " ".join(" ".join(c["cmd"]) for c in calls)
            self.assertIn("test_foo", joined)
            self.assertNotIn("discover", joined)
            self.assertIn("targeted", out.lower())

    def test_default_no_changes_is_honest_skip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = _make_repo(Path(td))
            # Revert the change: nothing changed → honest skip, no run.
            _git("checkout", "--", "foo.py", cwd=root)
            out, calls = self._run_with_fake(self._agent(root), {})
            self.assertEqual(calls, [])
            self.assertIn("nothing ran", out.lower())

    def test_timeout_message_says_still_running_and_how_to_resume(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = _make_repo(Path(td))
            out, _ = self._run_with_fake(
                self._agent(root),
                {"pattern": "tests/test_foo.py", "timeout": 5},
                exc=subprocess.TimeoutExpired(cmd=["x"], timeout=5))
            self.assertIn("still running", out.lower())
            self.assertIn("chunk", out.lower())

    def test_long_timeout_accepted_not_clamped_to_300(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = _make_repo(Path(td))
            _, calls = self._run_with_fake(
                self._agent(root),
                {"pattern": "tests/test_foo.py", "timeout": 3600})
            self.assertEqual(calls[0]["timeout"], 3600)

    def test_full_suite_runs_in_chunks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = _make_repo(Path(td))
            out, calls = self._run_with_fake(
                self._agent(root),
                {"pattern": "tests", "chunk": "1/2", "timeout": 600})
            self.assertEqual(len(calls), 1)
            joined = " ".join(calls[0]["cmd"])
            # Chunk 1/2 of [test_foo, test_slow] runs only test_foo.
            self.assertIn("test_foo", joined)
            self.assertNotIn("test_slow", joined)
            self.assertIn("chunk 1/2", out.lower())

    def test_full_suite_without_chunk_runs_all_chunks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = _make_repo(Path(td))
            out, calls = self._run_with_fake(
                self._agent(root),
                {"pattern": "tests", "timeout": 600})
            # Two chunks (test_foo, test_slow), both attempted.
            self.assertEqual(len(calls), 2)
            self.assertIn("chunk", out.lower())
            self.assertIn("85 min", out)


if __name__ == "__main__":
    unittest.main()
