"""Tests for the $PROJECT_NAME CLI tool.  Run from the project root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_PY), *argv],
        capture_output=True, text=True, timeout=30,
    )


class CliToolTest(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        proc = run_cli("--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("greet", proc.stdout)

    def test_greet(self) -> None:
        proc = run_cli("greet", "--name", "Ada")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Hello, Ada!", proc.stdout)

    def test_greet_default_name(self) -> None:
        proc = run_cli("greet")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Hello, world!", proc.stdout)

    def test_greet_shout(self) -> None:
        proc = run_cli("greet", "--name", "Ada", "--shout")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("HELLO, ADA!", proc.stdout)

    def test_no_command_fails(self) -> None:
        proc = run_cli()
        self.assertNotEqual(proc.returncode, 0)

    def test_version(self) -> None:
        proc = run_cli("--version")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("$PROJECT_NAME", proc.stdout)


if __name__ == "__main__":
    unittest.main()
