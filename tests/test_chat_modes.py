"""Per-chat games-only mode: store, /chatmode command, reply gate.

A games-only chat stays silent for normal chat, but game commands
(and game moves while a game is live) still work, and owner control
commands still work so the owner can unset it. Mode persists across
restarts (JSON store, same pattern as voice conversation).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.partner import chat_modes
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import (
    COMMAND_DETAILS, CONTROL_COMMANDS, LIST_GROUPS, LIST_ONELINERS,
    parse_control)


def _msg(text: str, *, chat_key: str = "telegram:999",
         owner: bool = False) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "999", kind=ChatKind.DM),
        incoming=True, text=text,
        sender="owner" if owner else "stranger",
        sender_id="5478650254" if owner else "999",
        meta={"is_owner": True} if owner else {},
    )


class _RecordingGateway:
    adapters: dict = {}
    sent: list[tuple[str, str]]

    def __init__(self) -> None:
        self.sent = []

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, error="")


class ChatModesStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-chatmodes-")
        self.path = os.path.join(self.tmp.name, "chat_modes.json")
        self._old = os.environ.get("NOMORALS_CHAT_MODES_PATH")
        os.environ["NOMORALS_CHAT_MODES_PATH"] = self.path

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("NOMORALS_CHAT_MODES_PATH", None)
        else:
            os.environ["NOMORALS_CHAT_MODES_PATH"] = self._old
        self.tmp.cleanup()

    def test_default_is_normal(self) -> None:
        self.assertEqual(chat_modes.get_chat_mode("telegram:1"), "normal")
        self.assertFalse(chat_modes.is_games_only("telegram:1"))

    def test_set_and_get_games_only(self) -> None:
        self.assertEqual(
            chat_modes.set_chat_mode("telegram:1", "games_only"), "games_only")
        self.assertTrue(chat_modes.is_games_only("telegram:1"))
        self.assertEqual(chat_modes.get_chat_mode("telegram:1"), "games_only")

    def test_set_normal_removes_entry(self) -> None:
        chat_modes.set_chat_mode("telegram:1", "games_only")
        chat_modes.set_chat_mode("telegram:1", "normal")
        self.assertFalse(chat_modes.is_games_only("telegram:1"))
        self.assertEqual(chat_modes.list_chat_modes(), {})

    def test_persists_across_reload(self) -> None:
        chat_modes.set_chat_mode("telegram:1", "games_only")
        # a fresh load from the same file still sees it (no in-memory cache)
        self.assertTrue(chat_modes.is_games_only("telegram:1"))
        self.assertEqual(chat_modes.list_chat_modes(),
                         {"telegram:1": "games_only"})

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chat_modes.set_chat_mode("telegram:1", "loud")
        self.assertFalse(chat_modes.is_games_only("telegram:1"))

    def test_empty_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chat_modes.set_chat_mode("", "games_only")

    def test_corrupt_file_degrades_to_normal(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertFalse(chat_modes.is_games_only("telegram:1"))


class ChatmodeRegistrationTest(unittest.TestCase):
    def test_registered_everywhere(self) -> None:
        self.assertIn("chatmode", CONTROL_COMMANDS)
        self.assertIn("chatmode", COMMAND_DETAILS)
        self.assertIn("chatmode", LIST_ONELINERS)
        grouped = {c for _, cs in LIST_GROUPS for c in cs}
        self.assertIn("chatmode", grouped)
        self.assertIn("/chatmode", COMMAND_DETAILS["chatmode"]["usage"])

    def test_parse_control(self) -> None:
        cmd = parse_control("/chatmode games-only telegram:123")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "chatmode")
        self.assertEqual(cmd.arg, "games-only")
        self.assertEqual(cmd.tail, "games-only telegram:123")
        self.assertIsNotNone(parse_control("/chatmode"))
        self.assertIsNotNone(parse_control("/chatmode status"))

    def test_too_many_args_is_error(self) -> None:
        cmd = parse_control("/chatmode games-only telegram:1 extra")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "error")


class ChatmodeCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-chatmode-")
        self.modes_path = os.path.join(self.tmp.name, "chat_modes.json")
        self._old = os.environ.get("NOMORALS_CHAT_MODES_PATH")
        os.environ["NOMORALS_CHAT_MODES_PATH"] = self.modes_path
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.runtime = PartnerRuntime(self.context, gateway=_RecordingGateway())

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        if self._old is None:
            os.environ.pop("NOMORALS_CHAT_MODES_PATH", None)
        else:
            os.environ["NOMORALS_CHAT_MODES_PATH"] = self._old
        self.tmp.cleanup()

    def test_owner_sets_games_only_for_current_chat(self) -> None:
        out = self.runtime.handle_control(
            "/chatmode games-only", "telegram:999",
            message=_msg("/chatmode games-only", owner=True))
        self.assertIn("games-only", out)
        self.assertTrue(chat_modes.is_games_only("telegram:999"))

    def test_status_shows_mode(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        out = self.runtime.handle_control(
            "/chatmode status", "telegram:999",
            message=_msg("/chatmode status", owner=True))
        self.assertIn("games-only", out)
        out = self.runtime.handle_control(
            "/chatmode", "telegram:999",
            message=_msg("/chatmode", owner=True))
        self.assertIn("games-only", out)

    def test_owner_restores_normal(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        out = self.runtime.handle_control(
            "/chatmode normal", "telegram:999",
            message=_msg("/chatmode normal", owner=True))
        self.assertIn("normal", out)
        self.assertFalse(chat_modes.is_games_only("telegram:999"))

    def test_owner_sets_remote_chat_by_key(self) -> None:
        out = self.runtime.handle_control(
            "/chatmode games-only telegram:555", "telegram:999",
            message=_msg("/chatmode games-only telegram:555", owner=True))
        self.assertIn("telegram:555", out)
        self.assertTrue(chat_modes.is_games_only("telegram:555"))
        self.assertFalse(chat_modes.is_games_only("telegram:999"))

    def test_bad_mode_is_usage_error(self) -> None:
        out = self.runtime.handle_control(
            "/chatmode loud", "telegram:999",
            message=_msg("/chatmode loud", owner=True))
        self.assertIn("usage", out)
        self.assertFalse(chat_modes.is_games_only("telegram:999"))

    def test_bad_target_key_is_usage_error(self) -> None:
        out = self.runtime.handle_control(
            "/chatmode games-only 555", "telegram:999",
            message=_msg("/chatmode games-only 555", owner=True))
        self.assertIn("telegram:123456", out)
        self.assertFalse(chat_modes.is_games_only("555"))

    def test_non_owner_cannot_set(self) -> None:
        out = self.runtime.handle_control(
            "/chatmode games-only", "telegram:999",
            message=_msg("/chatmode games-only", owner=False))
        self.assertEqual(out, "that one's just for the owner.")
        self.assertFalse(chat_modes.is_games_only("telegram:999"))


class GamesOnlyGateTest(ChatmodeCommandTest):
    """The _process gate: silence for chat, pass for games + owner control."""

    def test_plain_chat_dropped(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        self.assertFalse(
            self.runtime._passes_games_only(_msg("hey how are you")))

    def test_game_command_passes(self) -> None:
        self.assertTrue(
            self.runtime._passes_games_only(_msg("/game hangman")))
        self.assertTrue(
            self.runtime._passes_games_only(_msg("/hangman")))
        self.assertTrue(
            self.runtime._passes_games_only(_msg("/game list")))

    def test_play_passes_for_everyone(self) -> None:
        self.assertTrue(
            self.runtime._passes_games_only(_msg("/play some song")))

    def test_arena_blocked_for_non_owner(self) -> None:
        self.assertFalse(
            self.runtime._passes_games_only(_msg("/arena status")))
        self.assertTrue(
            self.runtime._passes_games_only(
                _msg("/arena status", owner=True)))

    def test_owner_control_passes(self) -> None:
        self.assertTrue(
            self.runtime._passes_games_only(
                _msg("/chatmode normal", owner=True)))
        self.assertTrue(
            self.runtime._passes_games_only(_msg("/status", owner=True)))

    def test_non_owner_owner_command_dropped(self) -> None:
        self.assertFalse(
            self.runtime._passes_games_only(_msg("/status")))
        self.assertFalse(
            self.runtime._passes_games_only(
                _msg("/chatmode normal", owner=False)))

    def test_unknown_slash_dropped(self) -> None:
        self.assertFalse(
            self.runtime._passes_games_only(_msg("/notarealcommand")))

    def test_process_drops_plain_chat_silently(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        gateway = self.runtime.gateway
        assert isinstance(gateway, _RecordingGateway)
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None):
            self.runtime._process(_msg("hey how are you"))
        self.assertEqual(gateway.sent, [])

    def test_process_passes_game_command(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        gateway = self.runtime.gateway
        assert isinstance(gateway, _RecordingGateway)
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None), \
                mock.patch.object(
                    self.runtime, "_control_game",
                    return_value="game started"):
            self.runtime._process(_msg("/game hangman"))
        self.assertTrue(any("game started" in text for _, text in gateway.sent))

    def test_process_passes_owner_chatmode_normal(self) -> None:
        chat_modes.set_chat_mode("telegram:999", "games_only")
        gateway = self.runtime.gateway
        assert isinstance(gateway, _RecordingGateway)
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None):
            self.runtime._process(
                _msg("/chatmode normal", owner=True))
        self.assertFalse(chat_modes.is_games_only("telegram:999"))
        self.assertTrue(any("normal" in text for _, text in gateway.sent))

    def test_process_normal_chat_still_replies(self) -> None:
        # Control: the gate only fires for games-only chats.
        gateway = self.runtime.gateway
        assert isinstance(gateway, _RecordingGateway)
        with mock.patch.object(
                self.runtime, "_route_game_move", return_value=None), \
                mock.patch.object(
                    self.runtime.brain, "handle_message",
                    return_value=SimpleNamespace(parts=["hello back"])):
            self.runtime._process(_msg("hey how are you"))
        self.assertTrue(any("hello back" in text for _, text in gateway.sent))


if __name__ == "__main__":
    unittest.main()
