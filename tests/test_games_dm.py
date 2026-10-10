"""The DM narrates: feed → mood-voiced narration → finale scenes."""
from __future__ import annotations

import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.gamemaster import GameMaster, feed
from nomorals.games.games.base import Room
from nomorals.games.npc import NPCStore
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db):
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


ADA = Player.from_sender("telegram", "111", "Ada")


def make_room(game="arena"):
    return Room(id="r1", game=game, chat_key="test:x", platform="test",
                kind="dm", players=[ADA], state={})


class FeedTests(unittest.TestCase):
    def test_feed_appends_to_state(self):
        room = make_room()
        feed(room, "Ada lands a killing crit", big=True)
        self.assertEqual(len(room.state["_dm_feed"]), 1)
        self.assertTrue(room.state["_dm_feed"][0]["big"])

    def test_feed_caps_and_ignores_junk(self):
        room = make_room()
        for i in range(10):
            feed(room, f"event {i}")
        self.assertEqual(len(room.state["_dm_feed"]), 6)
        feed(room, "")
        feed(room, "   ")
        self.assertEqual(len(room.state["_dm_feed"]), 6)

    def test_feed_never_raises(self):
        feed(None, "x")
        feed(make_room(), None)
        r = make_room()
        r.state = None
        feed(r, "x")  # no state dict — must not raise


class AnnounceTests(unittest.TestCase):
    def setUp(self):
        self.gm = GameMaster(NPCStore(), suggest=None, seed=7)

    def test_announce_narrates_headliner(self):
        line = self.gm.announce("arena", [
            {"text": "Ada strikes down the foe", "big": False},
            {"text": "Ada lands a killing CRIT", "big": True},
        ])
        self.assertIsNotNone(line)
        self.assertIn("🎲", line)
        # the big event is the headliner
        self.assertIn("killing CRIT", line)

    def test_announce_prefers_last_when_nothing_big(self):
        line = self.gm.announce("arena", ["first", "second"])
        self.assertIn("second", line)

    def test_announce_empty_is_silent(self):
        self.assertIsNone(self.gm.announce("arena", []))
        self.assertIsNone(self.gm.announce("arena", None))

    def test_announce_honors_mood(self):
        self.gm.set_dm_mood("arena", "grim")
        line = (self.gm.announce("arena", ["Ada wins the duel"]) or "").lower()
        # grim templates have a signature flavor
        self.assertTrue(
            any(w in line for w in ("shadows", "heavy", "cold", "dust",
                                    "quietly")),
            line)

    def test_announce_with_cast_reacts(self):
        cast = self.gm.ensure_cast("arena")
        self.assertTrue(len(cast) >= 1)
        line = self.gm.announce(
            "arena", [{"text": "Ada wins the duel", "big": True}],
            cast=cast)
        # narration + an NPC reaction line
        self.assertIn("\n", line)


class EngineDrainTests(unittest.TestCase):
    def test_move_drains_feed_into_output(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("test:dm", "arena", ADA, kind="dm")
        # push a feed event, then move: the DM narrates after the move
        from nomorals.games.gamemaster import feed as push
        push(room, "Ada lands a killing CRIT on the foe", big=True)
        # force past the cooldown
        room.state["_dm_last_narrate"] = 0
        msgs = engine.move("test:dm", "attack", ADA, kind="dm")
        self.assertTrue(any("🎲" in m for m in msgs),
                        f"no narration in: {msgs}")

    def test_cooldown_suppresses_spam(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("test:dm2", "arena", ADA, kind="dm")
        from nomorals.games.gamemaster import feed as push
        import time
        push(room, "event one", big=True)
        room.state["_dm_last_narrate"] = time.time()  # just narrated
        out: list[str] = []
        engine._drain_dm_feed(room, engine.games["arena"], out)
        self.assertEqual(out, [])
        # the feed was still consumed
        self.assertNotIn("_dm_feed", room.state)

    def test_finale_narration_for_arena(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("test:dm3", "arena", ADA, kind="dm")
        # rig the kill: house at 0 HP
        room.state["house"]["hp"] = 0
        room.state["done"] = True
        engine.quit("test:dm3")
        joined = "\n".join(sent)
        self.assertIn("🎲", joined)

    def test_no_finale_for_opt_out_games(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("test:dm4", "wordle", ADA, kind="dm")
        self.assertFalse(
            getattr(engine.games["wordle"], "dm_finale", False))
        engine.quit("test:dm4")

    def test_dm_finale_event_shapes(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("test:dm5", "arena", ADA, kind="dm")
        game = engine.games["arena"]
        room.state["house"]["hp"] = 0
        evt = game.dm_finale_event(room)
        self.assertIsNotNone(evt)
        self.assertIn("victorious", evt)
        room.state["house"]["hp"] = 50
        room.state["you"]["hp"] = 0
        evt = game.dm_finale_event(room)
        self.assertIn("falls", evt)
        engine.quit("test:dm5")


class DmCommandsTests(unittest.TestCase):
    def test_narrate_say_path(self):
        gm = GameMaster(NPCStore(), suggest=None, seed=3)
        gm.set_dm_mood("arena", "epic")
        line = gm.narrate("arena", "the gates open")
        self.assertTrue(line)
        self.assertNotEqual(line.strip(), "the gates open")


if __name__ == "__main__":
    unittest.main()
