"""Game move routing: a live room OWNS the message.

Regression test for the leak where ``_route_game_move`` returned None when
``_game_player()`` couldn't resolve a sender (missing sender_id, unrecognized
chat kind) — even with a game LIVE in that chat — and the move silently fell
through to the general brain ("Attack" answered as a chat question).

The rule: when ``engine.live(chat_key)`` finds an active room, the message
must NEVER silently become a brain question.
- DM + player None -> best-effort chat-key player, move proceeds.
- group + player None -> honest "couldn't identify you" error, not None.
"""
from __future__ import annotations

import types
import unittest

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.social.chat.base import ChatKind


class _FakeRelay:
    def get_relay_for_chat(self, chat_key: str):  # type: ignore[no-untyped-def]
        return None


class _FakeRoom:
    kind = "dm"

    def player(self, key: str):  # type: ignore[no-untyped-def]
        return None


class _FakeEngine:
    def __init__(self) -> None:
        self.relay = _FakeRelay()
        self.moved: list = []

    def live(self, chat_key: str):  # type: ignore[no-untyped-def]
        return _FakeRoom()

    def move(self, chat_key: str, text: str, player, kind: str = "dm"):  # type: ignore[no-untyped-def]
        self.moved.append((chat_key, text, player, kind))
        return [f"move accepted: {text}"]


class _RouteHarness(RuntimeGamesMixin):
    def __init__(self) -> None:
        self._engine = _FakeEngine()
        self.context = types.SimpleNamespace(db=None)

    def _game_engine(self):  # type: ignore[no-untyped-def]
        return self._engine


class RouteGameMovePlayerNoneTests(unittest.TestCase):
    def test_dm_player_none_uses_chat_key_fallback(self):
        """DM + unresolvable player: chat-key identity, move proceeds."""
        h = _RouteHarness()
        out = h._route_game_move("telegram:12345", "Attack",
                                 player=None, kind=ChatKind.DM)
        # Must NOT be None (None = falls through to the brain).
        self.assertIsNotNone(out)
        self.assertIn("move accepted", out)
        # The engine got a real player derived from the chat key.
        self.assertEqual(len(h._engine.moved), 1)
        _, _, player, _ = h._engine.moved[0]
        self.assertIsNotNone(player)
        self.assertIn("12345", player.key)

    def test_group_player_none_returns_honest_error(self):
        """Group + unresolvable player: honest error, never None."""
        h = _RouteHarness()
        out = h._route_game_move("telegram:-999", "Attack",
                                 player=None, kind=ChatKind.GROUP)
        # Must NOT be None (None = falls through to the brain).
        self.assertIsNotNone(out)
        self.assertIn("couldn't tell who you are", out)
        # The engine must NOT have run a move for a phantom player.
        self.assertEqual(h._engine.moved, [])

    def test_no_live_room_still_returns_none(self):
        """No live game: plain messages still fall through normally."""
        h = _RouteHarness()
        h._engine.live = lambda chat_key: None  # type: ignore[method-assign]
        out = h._route_game_move("telegram:12345", "hello there",
                                 player=None, kind=ChatKind.DM)
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
