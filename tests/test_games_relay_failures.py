"""Pins for nomorals/games/relay.py silent-swallow census (tail-12).

Every site below is an EXCLUSION: each failure degrades to a supported
state (memory-only operation = the db=None configuration, empty state =
fresh start, in-memory-correct with only restart-survival lost) or to an
omission that claims nothing. Nothing here fabricates success, moves
money, or strands a player: the relay object itself is always correct in
memory, and the one residue (a stale DB row when a DELETE fails, site
179) is bounded by the 1h invite expiry and documented in SURVEY.md.

NOT-recording is pinned per site with the tail-8 snapshot method:
``recent_failures()`` before the trigger must equal ``recent_failures()``
after it — relay.py records nothing anywhere, and the games ledger is
the package surface that would carry it.
"""
from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace

from nomorals.games.failures import recent_failures
from nomorals.games.players import Player
from nomorals.games.relay import GameRelay, RelayRoom

ADA = Player.from_sender("telegram", "1", "Ada")
BOB = Player.from_sender("telegram", "2", "Bob")


class FailingDb:
    """Fake DB: execute/query raise only on the configured statements."""

    def __init__(self, fail_execute=False, fail_query=False, fail_on=""):
        self.fail_execute = fail_execute
        self.fail_query = fail_query
        self.fail_on = fail_on
        self.ok_writes = []

    def execute(self, sql, *args):
        if self.fail_execute or (self.fail_on and self.fail_on in sql):
            raise RuntimeError("boom")
        self.ok_writes.append(sql)

    def query(self, sql, *args):
        if self.fail_query:
            raise RuntimeError("boom")
        return []


def stub_engine(db=None, games=None, **kw):
    eng = SimpleNamespace(db=db, games=games or {})
    for k, v in kw.items():
        setattr(eng, k, v)
    return eng


def snapshot():
    return recent_failures(200)


class RelayFailuresTest(unittest.TestCase):
    def assert_not_recorded(self, before):
        self.assertEqual(snapshot(), before)

    def test_116_ensure_tables_failure_stays_memory_only(self):
        before = snapshot()
        relay = GameRelay(stub_engine(db=FailingDb(fail_execute=True,
                                                   fail_query=True)))
        self.assertEqual(relay.invites, {})
        self.assertEqual(relay.relays, {})
        self.assert_not_recorded(before)

    def test_154_load_failure_starts_empty(self):
        before = snapshot()
        relay = GameRelay(stub_engine(db=FailingDb(fail_query=True)))
        self.assertEqual(relay.invites, {})
        self.assertEqual(relay.relays, {})
        self.assert_not_recorded(before)

    def test_170_save_invite_failure_keeps_memory_invite(self):
        before = snapshot()
        db = FailingDb(fail_on="INSERT INTO game_invites")
        relay = GameRelay(stub_engine(
            db=db, games={"duel": SimpleNamespace(max_players=2)}))
        inv = relay.create_invite("chat:a", ADA, "duel")
        self.assertIs(relay.invites[inv.code], inv)
        self.assertFalse(any("INSERT INTO game_invites" in s
                             for s in db.ok_writes))
        self.assert_not_recorded(before)

    def test_179_delete_invite_failure_keeps_memory_correct(self):
        # In-memory state is correct (invite gone, honest count); the
        # DB row stays stale — the documented corner, bounded by the
        # 1h invite expiry. The pin records the observed behavior, not
        # a judgment that the residue is good.
        before = snapshot()
        db = FailingDb(fail_on="DELETE FROM game_invites")
        relay = GameRelay(stub_engine(
            db=db, games={"duel": SimpleNamespace(max_players=2)}))
        inv = relay.create_invite("chat:a", ADA, "duel")
        self.assertEqual(relay.cancel_invites_from_chat("chat:a"), 1)
        self.assertNotIn(inv.code, relay.invites)
        self.assertFalse(any("DELETE FROM game_invites" in s
                             for s in db.ok_writes))
        self.assert_not_recorded(before)

    def test_197_save_relay_failure_does_not_raise(self):
        before = snapshot()
        relay = GameRelay(stub_engine(
            db=FailingDb(fail_on="INSERT INTO game_relays")))
        room = RelayRoom(room_id="c", game_name="duel", chat_a="a",
                         chat_b="b", player_a=ADA, player_b=BOB)
        relay._save_relay(room)
        self.assert_not_recorded(before)

    def test_206_delete_relay_failure_does_not_raise(self):
        before = snapshot()
        relay = GameRelay(stub_engine(
            db=FailingDb(fail_on="DELETE FROM game_relays")))
        relay._delete_relay("c")
        self.assert_not_recorded(before)

    def test_292_on_join_failure_does_not_strand_relay(self):
        before = snapshot()

        class FakeRoom:
            guard = contextlib.nullcontext()

            def __init__(self):
                self.players = []

            def player(self, key):
                return None

        game = SimpleNamespace(max_players=2)

        def bad_join(room, player, mind):
            raise RuntimeError("boom")

        game.on_join = bad_join
        eng = stub_engine(
            games={"duel": game},
            start=lambda *a, **k: (FakeRoom(), []),
            _mirror_player=lambda *a: None,
            _persist=lambda *a: None,
            _mind=None)
        relay = GameRelay(eng)
        inv = relay.create_invite("chat:a", ADA, "duel")
        room = relay.accept_invite(inv.code, "chat:b", BOB)
        self.assertIs(relay.relays[room.room_id], room)
        self.assert_not_recorded(before)

    def test_379_turn_line_failure_still_renders_status(self):
        before = snapshot()

        class BadRoom:
            @property
            def state(self):
                raise RuntimeError("boom")

        eng = stub_engine(live=lambda vc: BadRoom())
        relay = GameRelay(eng)
        room = RelayRoom(room_id="c", game_name="duel", chat_a="a",
                         chat_b="b", player_a=ADA, player_b=BOB)
        relay.relays["c"] = room
        relay.chat_to_relay["a"] = "c"
        relay.chat_to_relay["b"] = "c"
        text = relay.status_text("a")
        self.assertIsNotNone(text)
        self.assertIn("relay c", text)
        self.assertIn("idle", text)
        self.assert_not_recorded(before)

    def test_460_quit_failure_still_closes_relay(self):
        before = snapshot()

        def bad_quit(vc):
            raise RuntimeError("boom")

        eng = stub_engine(quit=bad_quit)
        relay = GameRelay(eng)
        room = RelayRoom(room_id="c", game_name="duel", chat_a="a",
                         chat_b="b", player_a=ADA, player_b=BOB)
        relay.relays["c"] = room
        relay.chat_to_relay["a"] = "c"
        relay.chat_to_relay["b"] = "c"
        relay.close_relay("c")
        self.assertNotIn("c", relay.relays)
        self.assertNotIn("a", relay.chat_to_relay)
        self.assert_not_recorded(before)

    def test_476_cleanup_failure_retries_next_tick(self):
        before = snapshot()
        relay = GameRelay(stub_engine())
        relay._last_cleanup = 0.0

        def bad_cleanup():
            raise RuntimeError("boom")

        relay.cleanup_expired = bad_cleanup
        relay.maybe_cleanup()
        self.assertGreater(relay._last_cleanup, 0.0)
        self.assert_not_recorded(before)


if __name__ == "__main__":
    unittest.main()
