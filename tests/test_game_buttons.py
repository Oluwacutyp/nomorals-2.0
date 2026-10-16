"""Part 2e: game-results keyboard + rematch difficulty.

* Game-results replies get Rematch / Next / Arena / Expert inline
  buttons (game menu + live-game patterns unchanged).
* ``engine.rematch(chat_key, difficulty=...)`` passes difficulty to
  ``start`` (default "normal" — unchanged behavior).
* ``/game rematch [easy|normal|hard|expert]`` parses the difficulty
  word the same way the start path does.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.social.chat.tgbot_buttons import buttons_for_text
from nomorals.storage.db import Database


class _Ctx:
    def __init__(self, db):
        self.db = db


def _make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(_Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db


class GameResultsButtonsTests(unittest.TestCase):
    def test_results_keyboard_has_four_actions(self):
        kb = buttons_for_text(
            "The House wins! 50 points and coins credited.")
        self.assertEqual(kb, [[("🔁 Rematch", "game rematch"),
                               ("⏭️ Next", "game list")],
                              [("🏟️ Arena", "arena"),
                               ("💀 Expert", "game rematch expert")]])

    def test_menu_keyboard_unchanged(self):
        kb = buttons_for_text("modes & mastery\n/game <name> — start")
        self.assertEqual(kb, [
            [("🎮 Join Game", "game join"), ("📊 Stats", "game stats")],
            [("🏆 Leaderboard", "game leaderboard")],
        ])

    def test_plain_text_has_no_buttons(self):
        self.assertIsNone(buttons_for_text("just chatting"))


class RematchDifficultyTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db = _make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def _finished_game(self, chat="local:r1"):
        p = Player(key="local:rm", platform="local", name="Rm")
        room, _ = self.engine.start(chat, "ttt", p)
        room.state["done"] = True
        self.engine._finish(room)
        return p

    def test_rematch_passes_difficulty_to_start(self):
        self._finished_game()
        with mock.patch.object(
                self.engine, "start",
                wraps=self.engine.start) as start:
            room, msgs = self.engine.rematch("local:r1",
                                              difficulty="expert")
            self.assertIsNotNone(room)
            _, kwargs = start.call_args
            self.assertEqual(kwargs.get("difficulty"), "expert")

    def test_rematch_default_is_normal(self):
        self._finished_game()
        with mock.patch.object(
                self.engine, "start",
                wraps=self.engine.start) as start:
            room, _ = self.engine.rematch("local:r1")
            self.assertIsNotNone(room)
            _, kwargs = start.call_args
            self.assertEqual(kwargs.get("difficulty"), "normal")

    def test_control_game_rematch_parses_difficulty(self):
        # _control_game("rematch expert") -> engine.rematch(difficulty=expert).
        stub = SimpleNamespace(context=SimpleNamespace())
        seen: dict = {}

        class _FakeEngine:
            games = {}

            def live(self, chat_key):
                return None

            def rematch(self, chat_key, difficulty="normal"):
                seen["difficulty"] = difficulty
                return None, ["ok"]

        stub._game_engine = lambda: _FakeEngine()
        out = RuntimeGamesMixin._control_game(
            stub, "rematch expert", "local:x", player=None, kind="dm")
        self.assertEqual(seen["difficulty"], "expert")
        self.assertEqual(out, "ok")

    def test_control_game_rematch_default_normal(self):
        stub = SimpleNamespace(context=SimpleNamespace())
        seen: dict = {}

        class _FakeEngine:
            games = {}

            def live(self, chat_key):
                return None

            def rematch(self, chat_key, difficulty="normal"):
                seen["difficulty"] = difficulty
                return None, ["ok"]

        stub._game_engine = lambda: _FakeEngine()
        RuntimeGamesMixin._control_game(
            stub, "rematch", "local:x", player=None, kind="dm")
        self.assertEqual(seen["difficulty"], "normal")
