"""P0 settle contract: a finished game reaches the chat exactly once.

Owner design (2026-11-01): the settle joins into ONE message at the
send seam — no per-line fan-out, no duplicate combined render, the
fairness reveal on demand instead of blasted after every game, the
final card showing the player's real level, and season lines that
never double an event's name.
"""
from __future__ import annotations

import unittest

from nomorals.games import fairness, seasons
from nomorals.games.engine import GameEngine, render_game_over_card
from nomorals.games.players import Player, Profile
from nomorals.games.progression import xp_for_level
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


class SettleIsOneMessageTests(unittest.TestCase):
    def test_slots_spin_settle_is_one_message_and_not_emitted(self):
        engine, db, sent = make_engine()
        # wager integrity: a spin stakes real coins — fund the table
        engine.store.add_coins(ADA, 100, reason="test funding")
        room, _ = engine.start("test:sl", "slots", ADA, kind="dm")
        seed = room.state["fair"]["seed"]
        sent.clear()
        out = engine.move("test:sl", "spin", ADA, kind="dm")
        # the finish emitted NOTHING itself — the caller owns delivery,
        # so the settle can only ever reach the chat once.
        self.assertEqual(sent, [])
        # the whole settle (outcome, coins, commitment line, card) is
        # a single element of the returned messages.
        settle = [m for m in out if "provably fair" in m]
        self.assertEqual(len(settle), 1)
        self.assertIn("🪙", settle[0])
        self.assertIn("🏁", settle[0])
        self.assertIn("+", settle[0])
        # the seed is never in the settle — reveal is on demand.
        self.assertNotIn(seed, settle[0])
        reveal = engine.last_fair_reveal("test:sl")
        self.assertIn(seed, reveal)
        self.assertTrue(
            fairness.verify(room.state["fair"]["commit"], seed))

    def test_idle_finish_emits_exactly_once(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("test:idle", "slots", ADA, kind="dm")
        sent.clear()
        engine._finish_idle(room)
        self.assertEqual(len(sent), 1)
        self.assertIn("idle too long", sent[0])
        self.assertIn("🏁", sent[0])

    def test_quit_returns_single_settle_and_emits_nothing(self):
        engine, db, sent = make_engine()
        # wager integrity: blackjack stakes its bet at the deal
        engine.store.add_coins(ADA, 100, reason="test funding")
        engine.start("test:q", "blackjack", ADA, kind="dm")
        sent.clear()
        out = engine.quit("test:q")
        self.assertEqual(sent, [])
        self.assertEqual(len(out), 1)
        self.assertIn("table closed", out[0])


class CardLevelTests(unittest.TestCase):
    def test_card_shows_real_level_not_always_one(self):
        class FakeStore:
            def get_for(self, p):
                return Profile(key=p.key, name=p.name, coins=42,
                               xp=xp_for_level(5))

        card = render_game_over_card(
            "slots", [ADA], {ADA.key: 10}, store=FakeStore())
        self.assertIn("lvl 5", card)
        self.assertNotIn("lvl 1", card)


class SeasonNameTests(unittest.TestCase):
    def test_no_event_line_doubles_its_name(self):
        for ev in seasons.SEASON_ROSTER:
            line = (f"{ev['emoji']} **{ev['name']}** — {ev['blurb']}")
            self.assertNotIn(f"**{ev['name']}** — {ev['name']}", line,
                             ev["id"])
            self.assertFalse(ev["blurb"].startswith(ev["name"]),
                             ev["id"])

    def test_describe_seasons_has_no_doubling(self):
        text = seasons.describe_seasons(None)
        for ev in seasons.SEASON_ROSTER:
            self.assertNotIn(f"{ev['name']} — {ev['name']}", text)


if __name__ == "__main__":
    unittest.main()
