"""Group-as-player arena defect (owner-reported 2026-11-26): in the
Devon Telegram group he ran ``/game arena expert`` and the arena
seated the GROUP chat id as the human — "Novice -1004381594376
enters" — then doubled the join line and stalled on a seat that
could never move.

Covers:
- ``_game_player_for_command``: a group message with no sender
  yields None (refused, never minted as the chat); a group message
  with a real sender yields that human; a console call (no message)
  keeps the chat-keyed player — the DM/console legitimate use.
- ``_control_game``: a group start with player None is refused
  with the guard's own words, never seated as the chat.
- The engine arena start in a group seats the real human + the
  house, the turn clock never waits on the chat id, and the intro
  names the human.
- ``engine.join`` / ``engine.leave`` are return-only: the send
  callback is never called by them, and the join line appears
  exactly once in the dispatch's joined reply (the emit doubled
  it: once from the emit, once inside the joined reply).
- ``_control_math``: a group round with no player is refused
  honestly, never billed to a group-keyed player.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.storage.db import Database

GROUP_KEY = "telegram:-1004381594376"
GROUP_CHAT_ID = "-1004381594376"
OWNER = Player.from_sender("telegram", "5478650254", "Peacethefirst")
SECOND = Player.from_sender("telegram", "999", "Second")


def group_message(*, sender: str = "Peacethefirst",
                  sender_id: str = "5478650254") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform="telegram", chat_id=GROUP_CHAT_ID,
                     kind=ChatKind.GROUP),
        incoming=True, text="/game arena expert",
        sender=sender, sender_id=sender_id)


class Ctx:
    def __init__(self, db: Database) -> None:
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, sent


class GamePlayerForCommandTests(unittest.TestCase):
    def test_group_message_without_sender_yields_none(self):
        mixin = RuntimeGamesMixin()
        player = mixin._game_player_for_command(
            group_message(sender="", sender_id=""), GROUP_KEY)
        self.assertIsNone(player)

    def test_group_message_with_sender_yields_that_human(self):
        mixin = RuntimeGamesMixin()
        player = mixin._game_player_for_command(
            group_message(), GROUP_KEY)
        self.assertEqual(player.key, "telegram:5478650254")
        self.assertEqual(player.name, "Peacethefirst")

    def test_console_call_keeps_chat_keyed_player(self):
        mixin = RuntimeGamesMixin()
        player = mixin._game_player_for_command(None, "telegram:12345")
        self.assertEqual(player.key, "telegram:12345")


class ControlGameGuardTests(unittest.TestCase):
    def _stub(self):
        stub = SimpleNamespace(context=SimpleNamespace())
        # §33 claimed-room gate: this chat is not blocked.
        stub._group_system_blocked = lambda chat_key: False

        class _FakeEngine:
            games = {"arena": SimpleNamespace(variants={})}

            def live(self, chat_key):
                return None

        stub._game_engine = lambda: _FakeEngine()
        return stub

    def test_group_start_without_player_is_refused_not_seated(self):
        stub = self._stub()
        out = RuntimeGamesMixin._control_game(
            stub, "arena expert", GROUP_KEY, player=None, kind="group")
        self.assertEqual(
            out, "start a game from a chat — I need to know who's at "
                 "the table.")


class ArenaGroupSeatTests(unittest.TestCase):
    def test_group_arena_seats_the_human_and_the_house(self):
        engine, _sent = make_engine()
        room, msgs = engine.start(GROUP_KEY, "arena", OWNER,
                                  kind="group", difficulty="expert")
        self.assertEqual(room.humans[0].key, OWNER.key)
        self.assertNotIn(GROUP_KEY, [p.key for p in room.players])
        self.assertNotIn(GROUP_CHAT_ID, [p.name for p in room.players])
        # One human + one AI house seat.
        self.assertEqual(len(room.players), 2)
        self.assertEqual(len(room.humans), 1)
        # The clock waits on the human, never on the chat id.
        self.assertEqual(room.current.key, OWNER.key)
        intro = "\n".join(msgs)
        self.assertIn("Peacethefirst", intro)
        self.assertNotIn(GROUP_CHAT_ID, intro)


class JoinPostsOnceTests(unittest.TestCase):
    def test_join_is_return_only_and_posts_exactly_once(self):
        engine, sent = make_engine()
        engine.start(GROUP_KEY, "arena", OWNER, kind="group",
                     difficulty="expert")
        join_msgs = engine.join(GROUP_KEY, SECOND)
        # The join never sends by itself — the dispatch's joined
        # reply is the only send of the line.
        self.assertEqual(sent, [])
        joined = "\n".join(join_msgs)
        self.assertEqual(joined.count("sits down"), 1)
        self.assertIn("replacing", joined)
        # The move reply rides the same joined reply as the join
        # line (the owner's screenshot shape) — still exactly once.
        move_msgs = engine.move(GROUP_KEY, "attack", SECOND)
        final = "\n".join(join_msgs + move_msgs)
        self.assertEqual(final.count("sits down"), 1)
        self.assertIn("waiting on", final)

    def test_leave_is_return_only(self):
        # The arena's on_leave is None, so leave() returns nothing
        # there — the raid carries the real leave notice, and the
        # same return-only shape applies.
        engine, sent = make_engine()
        engine.start(GROUP_KEY, "raid", OWNER, kind="group")
        engine.join(GROUP_KEY, SECOND)
        self.assertEqual(sent, [])
        leave_msgs = engine.leave(GROUP_KEY, SECOND)
        self.assertEqual(sent, [])
        self.assertEqual(leave_msgs,
                         ["Second retreats from the raid."])
        self.assertEqual("\n".join(leave_msgs).count("retreats"), 1)


class ControlMathGroupTests(unittest.TestCase):
    def test_group_round_without_player_is_refused(self):
        stub = SimpleNamespace(context=SimpleNamespace())
        stub._math_handler = lambda: object()
        out = RuntimeGamesMixin._control_math(
            stub, "roll", GROUP_KEY, player=None, kind="group")
        self.assertIn("couldn't tell who you are", out)
        self.assertNotIn(GROUP_CHAT_ID, out)


if __name__ == "__main__":
    unittest.main()
