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
        # the cooldown defers, it does not discard: the event is
        # still on the feed, waiting for the cooldown to lapse
        self.assertEqual(
            [i["text"] for i in room.state["_dm_feed"]], ["event one"])

    def test_finale_narration_for_arena(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("test:dm3", "arena", ADA, kind="dm")
        # rig the kill: house at 0 HP
        room.state["house"]["hp"] = 0
        room.state["done"] = True
        # the settle (finale narration included) is RETURNED as one
        # message since the P0 settle fix — the finish no longer
        # emits line by line into `sent`.
        out = engine.quit("test:dm3")
        joined = "\n".join(out)
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


class RecordingGM:
    """A DM stand-in that records every batch it is asked to narrate."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def announce(self, game_id, items, cast=None):
        texts = [i.get("text") if isinstance(i, dict) else str(i)
                 for i in items]
        self.batches.append(texts)
        return "🎲 " + " / ".join(texts)

    def ensure_cast(self, game_id):
        return []


class CooldownDeferralTests(unittest.TestCase):
    """The drain cooldown defers narration — it must never discard it.

    Regression class for the pop-before-cooldown drain: events used
    to be popped off the feed BEFORE the cooldown check, so anything
    a game fed during the 25s window vanished unnarrated.
    """

    import time as _time

    def _room(self, engine, key="test:defer"):
        room, _ = engine.start(key, "arena", ADA, kind="dm")
        return room

    def _cool(self, room):
        room.state["_dm_last_narrate"] = self._time.time()

    def _lapse(self, room):
        room.state["_dm_last_narrate"] = self._time.time() - 26

    def test_events_during_cooldown_narrate_after_lapse_in_order(self):
        engine, db, sent = make_engine()
        gm = RecordingGM()
        engine._dm_obj = gm
        room = self._room(engine)
        game = engine.games["arena"]
        self._cool(room)
        feed(room, "first blood", big=True)
        feed(room, "a parry")
        feed(room, "the killing blow", big=True)
        out: list[str] = []
        engine._drain_dm_feed(room, game, out)
        engine._drain_dm_feed(room, game, out)  # still cooling: nothing
        self.assertEqual(out, [])
        self.assertEqual(gm.batches, [])
        # the whole feed survived both drains, in push order
        self.assertEqual(
            [i["text"] for i in room.state["_dm_feed"]],
            ["first blood", "a parry", "the killing blow"])
        self._lapse(room)
        engine._drain_dm_feed(room, game, out)
        # one batch, exact original order, feed consumed, moments kept
        self.assertEqual(
            gm.batches, [["first blood", "a parry", "the killing blow"]])
        self.assertEqual(len(out), 1)
        self.assertNotIn("_dm_feed", room.state)
        for text in ("first blood", "a parry", "the killing blow"):
            self.assertIn(text, room.state["_dm_narrated"])

    def test_cooldown_still_rate_limits(self):
        engine, db, sent = make_engine()
        gm = RecordingGM()
        engine._dm_obj = gm
        room = self._room(engine, "test:rate")
        game = engine.games["arena"]
        room.state["_dm_last_narrate"] = 0
        feed(room, "opener", big=True)
        out: list[str] = []
        engine._drain_dm_feed(room, game, out)
        self.assertEqual(gm.batches, [["opener"]])
        # inside the window, repeated drains post nothing at all
        feed(room, "follow-up")
        feed(room, "and another")
        for _ in range(3):
            engine._drain_dm_feed(room, game, out)
        self.assertEqual(gm.batches, [["opener"]])
        self.assertEqual(len(out), 1)
        # after the lapse the held events narrate as one batch
        self._lapse(room)
        engine._drain_dm_feed(room, game, out)
        self.assertEqual(
            gm.batches, [["opener"], ["follow-up", "and another"]])

    def test_no_event_narrated_twice_across_the_boundary(self):
        engine, db, sent = make_engine()
        gm = RecordingGM()
        engine._dm_obj = gm
        room = self._room(engine, "test:twice")
        game = engine.games["arena"]
        room.state["_dm_last_narrate"] = 0
        feed(room, "the warden falls", big=True)
        out: list[str] = []
        engine._drain_dm_feed(room, game, out)
        self.assertEqual(gm.batches, [["the warden falls"]])
        # the game re-feeds the same moment mid-cooldown, plus a new one
        feed(room, "the warden falls", big=True)
        feed(room, "dust settles")
        engine._drain_dm_feed(room, game, out)  # cooling: deferred whole
        self.assertEqual(len(gm.batches), 1)
        self._lapse(room)
        engine._drain_dm_feed(room, game, out)
        # only the genuinely new moment is narrated; the kill is not
        # told twice, and the leftover duplicate leaves with the feed
        self.assertEqual(
            gm.batches, [["the warden falls"], ["dust settles"]])
        self.assertEqual(
            room.state["_dm_narrated"].count("the warden falls"), 1)
        self.assertNotIn("_dm_feed", room.state)
        engine._drain_dm_feed(room, game, out)  # nothing left to tell
        self.assertEqual(len(gm.batches), 2)

    def test_deferred_batch_still_passes_the_sanitizer(self):
        scaffold = (
            "1. Analyze the Request:\n"
            "    *   Role: Dungeon master of a chat game.\n"
            "    *   Event: Ada obliterates the Hollow Warden.\n"
            "    *   Constraint Checklist & Confidence Score:\n"
            "        1. Neutral tone? Yes.")
        db = Database(":memory:")
        db.migrate()
        engine = GameEngine(Ctx(db), send=lambda chat, text: None,
                            suggest=lambda prompt: scaffold)
        room = self._room(engine, "test:sanitized")
        game = engine.games["arena"]
        self._cool(room)
        feed(room, "Ada obliterates the Hollow Warden", big=True)
        out: list[str] = []
        engine._drain_dm_feed(room, game, out)
        self.assertEqual(out, [])  # deferred, not dropped
        self._lapse(room)
        engine._drain_dm_feed(room, game, out)
        self.assertEqual(len(out), 1)
        self.assertIn("🎲", out[0])
        # the model's scaffold never reaches the table, deferred or not
        self.assertNotIn("Analyze the Request", out[0])
        self.assertNotIn("Constraint Checklist", out[0])
        self.assertNotIn("Confidence Score", out[0])

    def test_rooms_do_not_cross_contaminate(self):
        engine, db, sent = make_engine()
        gm = RecordingGM()
        engine._dm_obj = gm
        game = engine.games["arena"]
        held = self._room(engine, "test:held")
        free = self._room(engine, "test:free")
        self._cool(held)
        free.state["_dm_last_narrate"] = 0
        feed(held, "held room moment", big=True)
        feed(free, "free room moment", big=True)
        out: list[str] = []
        engine._drain_dm_feed(free, game, out)
        self.assertEqual(gm.batches, [["free room moment"]])
        # the held room's feed is untouched by the other room's drain
        self.assertEqual(
            [i["text"] for i in held.state["_dm_feed"]],
            ["held room moment"])
        engine._drain_dm_feed(held, game, out)  # still cooling
        self.assertEqual(len(gm.batches), 1)
        self._lapse(held)
        engine._drain_dm_feed(held, game, out)
        self.assertEqual(
            gm.batches, [["free room moment"], ["held room moment"]])

    def test_finish_force_drain_narrates_deferred_events(self):
        engine, db, sent = make_engine()
        gm = RecordingGM()
        engine._dm_obj = gm
        room = self._room(engine, "test:force")
        game = engine.games["arena"]
        self._cool(room)
        feed(room, "the killing blow", big=True)
        out: list[str] = []
        engine._drain_dm_feed(room, game, out, force=True)
        self.assertEqual(gm.batches, [["the killing blow"]])
        self.assertEqual(len(out), 1)
        self.assertNotIn("_dm_feed", room.state)

    def test_deferred_feed_keeps_the_pushers_cap(self):
        engine, db, sent = make_engine()
        gm = RecordingGM()
        engine._dm_obj = gm
        room = self._room(engine, "test:cap")
        game = engine.games["arena"]
        self._cool(room)
        for n in range(8):
            feed(room, f"moment {n}")
        # the pusher's rolling cap (6) still bounds a deferred feed
        self.assertEqual(len(room.state["_dm_feed"]), 6)
        engine._drain_dm_feed(room, game, [])  # cooling: cap holds
        self.assertEqual(len(room.state["_dm_feed"]), 6)
        self._lapse(room)
        engine._drain_dm_feed(room, game, [])
        self.assertEqual(
            gm.batches,
            [[f"moment {n}" for n in range(2, 8)]])


class DmCommandsTests(unittest.TestCase):
    def test_narrate_say_path(self):
        gm = GameMaster(NPCStore(), suggest=None, seed=3)
        gm.set_dm_mood("arena", "epic")
        line = gm.narrate("arena", "the gates open")
        self.assertTrue(line)
        self.assertNotEqual(line.strip(), "the gates open")


if __name__ == "__main__":
    unittest.main()
