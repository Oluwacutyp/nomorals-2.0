"""Tests for the $PROJECT_NAME Telegram bot.  Run from the project root:

    python -m unittest discover -s tests -t .

No network: the client runs against a recording fake transport.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import run
from run import Bot, TelegramClient, TelegramError

RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def _make_bot() -> tuple[Bot, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []

    def post(url: str, payload: dict) -> dict:
        calls.append((url, payload))
        if url.endswith("/getUpdates"):
            return {"ok": True, "result": []}
        return {"ok": True, "result": {"message_id": len(calls)}}

    client = TelegramClient("fake-token", post=post)
    return Bot(client), calls


def _update(update_id: int, text: str, chat_id: int = 42) -> dict:
    return {"update_id": update_id,
            "message": {"message_id": update_id,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": text}}


class TelegramBotTest(unittest.TestCase):
    def test_start_greets_with_name(self) -> None:
        bot, _ = _make_bot()
        self.assertIn("$PROJECT_NAME", bot.dispatch("/start"))

    def test_help_lists_commands(self) -> None:
        bot, _ = _make_bot()
        reply = bot.dispatch("/help")
        for cmd in ("/start", "/ping", "/echo", "/reverse", "/upper"):
            self.assertIn(cmd, reply)

    def test_ping(self) -> None:
        bot, _ = _make_bot()
        self.assertEqual(bot.dispatch("/ping"), "pong")

    def test_echo_reverse_upper(self) -> None:
        bot, _ = _make_bot()
        self.assertEqual(bot.dispatch("/echo hello there"), "hello there")
        self.assertEqual(bot.dispatch("/reverse abc"), "cba")
        self.assertEqual(bot.dispatch("/upper hello"), "HELLO")

    def test_echo_without_arg_shows_usage(self) -> None:
        bot, _ = _make_bot()
        self.assertIn("Usage", bot.dispatch("/echo"))

    def test_unknown_command(self) -> None:
        bot, _ = _make_bot()
        reply = bot.dispatch("/frobnicate")
        self.assertIn("Unknown command", reply)
        self.assertIn("/help", reply)

    def test_plain_text_is_echoed(self) -> None:
        bot, _ = _make_bot()
        self.assertEqual(bot.dispatch("just talking"), "You said: just talking")

    def test_command_with_bot_suffix(self) -> None:
        bot, _ = _make_bot()
        self.assertEqual(bot.dispatch("/ping@$PROJECT_NAME".lower()), "pong")

    def test_handle_update_sends_reply(self) -> None:
        bot, calls = _make_bot()
        reply = bot.handle_update(_update(1, "/ping"))
        self.assertEqual(reply, "pong")
        sent = [p for url, p in calls if url.endswith("/sendMessage")]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["chat_id"], 42)
        self.assertEqual(sent[0]["text"], "pong")

    def test_handle_update_ignores_non_messages(self) -> None:
        bot, calls = _make_bot()
        self.assertIsNone(bot.handle_update({"update_id": 9}))
        self.assertIsNone(bot.handle_update(
            {"update_id": 10, "message": {"chat": {"id": 1}}}))
        self.assertFalse(any(url.endswith("/sendMessage")
                             for url, _ in calls))

    def test_api_error_raises_telegram_error(self) -> None:
        client = TelegramClient(
            "fake", post=lambda url, payload: {"ok": False,
                                               "description": "bad"})
        with self.assertRaises(TelegramError):
            client.send_message(1, "hi")

    def test_empty_token_raises(self) -> None:
        with self.assertRaises(TelegramError):
            TelegramClient("")

    def test_poll_once_advances(self) -> None:
        seen: list[dict] = []

        def post(url: str, payload: dict) -> dict:
            if url.endswith("/getUpdates"):
                seen.append(payload)
                return {"ok": True,
                        "result": [_update(5, "/ping"), _update(6, "/echo x")]}
            return {"ok": True, "result": {}}

        client = TelegramClient("fake", post=post)
        bot = Bot(client)
        self.assertEqual(run.poll(client, bot, once=True), 0)
        self.assertEqual(seen[0]["offset"], 0)

    def test_missing_token_fails_fast(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "BOT_TOKEN"}
        proc = subprocess.run(
            [sys.executable, str(RUN_PY), "--once"],
            env={**env, "PYTHONPATH": str(RUN_PY.parent)},
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30,
            cwd=str(RUN_PY.parent),
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("BOT_TOKEN", proc.stderr)

    def test_self_test_passes_offline(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "BOT_TOKEN"}
        proc = subprocess.run(
            [sys.executable, str(RUN_PY), "--self-test"],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30,
            cwd=str(RUN_PY.parent),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)

    def test_entrypoint_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(RUN_PY), "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30,
            cwd=str(RUN_PY.parent),
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("$PROJECT_NAME", proc.stdout)


if __name__ == "__main__":
    unittest.main()
