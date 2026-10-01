"""Tests for the $PROJECT_NAME bot.  Run from the project root:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from bot import Bot

RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


class BotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bot = Bot()

    def test_start_greets(self) -> None:
        reply = self.bot.on_message("/start")
        self.assertIn("$PROJECT_NAME", reply)

    def test_help_lists_commands(self) -> None:
        reply = self.bot.on_message("/help")
        for cmd in ("/start", "/ping", "/echo", "/reverse", "/upper"):
            self.assertIn(cmd, reply)

    def test_ping(self) -> None:
        self.assertEqual(self.bot.on_message("/ping"), "pong")

    def test_echo(self) -> None:
        self.assertEqual(self.bot.on_message("/echo hello there"), "hello there")

    def test_echo_without_arg_shows_usage(self) -> None:
        self.assertIn("Usage", self.bot.on_message("/echo"))

    def test_reverse(self) -> None:
        self.assertEqual(self.bot.on_message("/reverse abc"), "cba")

    def test_upper(self) -> None:
        self.assertEqual(self.bot.on_message("/upper hello"), "HELLO")

    def test_plain_text_is_echoed(self) -> None:
        self.assertEqual(self.bot.on_message("just talking"), "You said: just talking")

    def test_unknown_command(self) -> None:
        reply = self.bot.on_message("/frobnicate")
        self.assertIn("Unknown command", reply)
        self.assertIn("/help", reply)

    def test_empty_message(self) -> None:
        self.assertIn("/help", self.bot.on_message("   "))

    def test_commands_are_case_insensitive(self) -> None:
        self.assertEqual(self.bot.on_message("/PING"), "pong")

    def test_console_entrypoint_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(RUN_PY), "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("$PROJECT_NAME", proc.stdout)


if __name__ == "__main__":
    unittest.main()
