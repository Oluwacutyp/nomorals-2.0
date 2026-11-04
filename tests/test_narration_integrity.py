"""Narration integrity + arena score.

The model behind the DM is a small one: asked to narrate, it may
answer with its reasoning scaffold ("1. Analyze the Request: …")
instead of narration. These tests pin the three guarantees:

* the sanitizer — scaffold never reaches a table; what survives is
  narration, or the caller falls back to the template forge;
* one narration per moment — a kill narrated by the move drain is
  not narrated again by the finish finale, and assembled output
  never repeats a line back-to-back;
* the arena scores — a fight's score is the damage the player dealt
  (plus a finish bonus), not a flat inherited 0.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from nomorals.games import engine_failures
from nomorals.games.engine import GameEngine, _collapse_dup_lines
from nomorals.games.gamemaster import GameMaster, _sanitize_model_line
from nomorals.games.games.ambitious import BattleArenaGame
from nomorals.games.games.base import Room
from nomorals.games.npc import NPCStore
from nomorals.games.players import Player
from nomorals.storage.db import Database


SCAFFOLD = (
    "1. Analyze the Request:\n"
    "    *   Role: Dungeon master of a chat game, narrating in a "
    "neutral tone.\n"
    "    *   Event: Peacethefirst obliterates Gore the Death of "
    "Hope — a BRUTAL FINISH.\n"
    "    *   Format: 1-3 vivid sentences. No stage directions, no "
    "fourth-wall breaks.\n"
    "    *   Constraint Checklist & Confidence Score:\n"
    "        1. Neutral tone? Yes.\n"
    "        2. 1-3 vivid sentences? Yes."
)

CLEAN_NARRATION = (
    "The gate groans open and dust pours from the dark. "
    "Ada steps through, blade low and ready."
)

EVENT = "Ada obliterates the Hollow Warden — a BRUTAL FINISH"


def make_gm(suggest, seed=7):
    store = NPCStore(Path(tempfile.mkdtemp()))
    return GameMaster(store, suggest=suggest, seed=seed)


def _npc(gm):
    return gm.ensure_cast("arena")[0]


class SanitizerUnitTests(unittest.TestCase):
    def test_pure_scaffold_rejected(self):
        clean, reason = _sanitize_model_line(SCAFFOLD, kind="narrate")
        self.assertEqual(clean, "")
        self.assertTrue(reason)

    def test_scaffold_plus_narration_extracts(self):
        raw = (SCAFFOLD + "\n\nHope dies under a grey sky, and the "
               "arena holds its breath. The silence after is total.")
        clean, reason = _sanitize_model_line(raw, kind="narrate")
        self.assertEqual(reason, "")
        self.assertIn("Hope dies under a grey sky", clean)
        self.assertNotIn("Analyze", clean)
        self.assertNotIn("Confidence", clean)

    def test_clean_narration_passes_unchanged(self):
        clean, reason = _sanitize_model_line(CLEAN_NARRATION,
                                             kind="narrate")
        self.assertEqual(reason, "")
        self.assertEqual(clean, CLEAN_NARRATION)

    def test_stage_direction_span_rejected(self):
        raw = "_pauses, hands folded, watching the smoke rise_"
        clean, _ = _sanitize_model_line(raw, kind="speak")
        self.assertEqual(clean, "")

    def test_truncated_fragment_rejected(self):
        raw = ("The bards will sing of this night in every hall from "
               "the Ashen Docks to the Violet Spire, of the gate that "
               "groaned and the dust that poured and the blade that "
               "sang once before the hall fell into a deep and "
               "sudden si")
        self.assertGreater(len(raw), 200)
        clean, reason = _sanitize_model_line(raw, kind="narrate")
        self.assertEqual(clean, "")
        self.assertIn("truncated", reason)

    def test_short_fragment_passes(self):
        # a lone short line without terminal punctuation is a
        # complete-feeling narration line, not a truncation
        clean, reason = _sanitize_model_line("Steel rang on stone",
                                             kind="narrate")
        self.assertEqual(reason, "")
        self.assertEqual(clean, "Steel rang on stone")

    def test_too_many_sentences_rejected(self):
        raw = " ".join(f"Sentence number {i} lands." for i in range(5))
        clean, _ = _sanitize_model_line(raw, kind="narrate")
        self.assertEqual(clean, "")

    def test_bold_unwrapped_not_rejected(self):
        raw = "The **blade** sang once, and the hall went quiet."
        clean, reason = _sanitize_model_line(raw, kind="narrate")
        self.assertEqual(reason, "")
        self.assertEqual(clean, "The blade sang once, and the hall "
                                  "went quiet.")


class SanitizerFlowTests(unittest.TestCase):
    def setUp(self):
        engine_failures.clear_failures()

    def test_narrate_pure_scaffold_falls_back_to_template(self):
        gm = make_gm(lambda prompt: SCAFFOLD)
        line = gm.narrate("arena", EVENT)
        # the template forge speaks (it always carries the event);
        # the scaffold never does
        self.assertIn(EVENT, line)
        for junk in ("Analyze", "Constraint", "Confidence", "Role:"):
            self.assertNotIn(junk, line)
        sites = [f.site for f in engine_failures.recent_failures()]
        self.assertIn("gamemaster.narrate_sanitized", sites)

    def test_narrate_extracts_buried_narration(self):
        tail = ("Hope dies under a grey sky, and the arena holds "
                "its breath. The silence after is total.")
        gm = make_gm(lambda prompt: SCAFFOLD + "\n\n" + tail)
        line = gm.narrate("arena", EVENT)
        self.assertEqual(line, tail)

    def test_narrate_clean_reply_passes_through(self):
        gm = make_gm(lambda prompt: CLEAN_NARRATION)
        self.assertEqual(gm.narrate("arena", EVENT), CLEAN_NARRATION)

    def test_npc_speak_stage_direction_falls_back(self):
        sermon = (
            "_pauses, hands folded, watching the smoke rise_\n\n"
            "\"Hope dies, yet from its ashes grows the understanding "
            "of why it mattered. Such endings are not cruel — they "
            "are necessary pruning. The old stories endure because "
            "they know this: destruction serves the garden.\"")
        gm = make_gm(lambda prompt: sermon)
        npc = _npc(gm)
        line = gm.npc_speak(npc, "the foe falls")
        self.assertNotIn("pauses, hands folded", line)
        self.assertIn(npc.name, line)  # template lines name the NPC
        sites = [f.site for f in engine_failures.recent_failures()]
        self.assertIn("gamemaster.npc_speak_sanitized", sites)

    def test_announce_with_scaffold_model_uses_forge(self):
        gm = make_gm(lambda prompt: SCAFFOLD)
        line = gm.announce("arena", [{"text": EVENT, "big": True}],
                           cast=[])
        self.assertIsNotNone(line)
        self.assertTrue(line.startswith("🎲 "))
        for junk in ("Analyze", "Constraint", "Confidence"):
            self.assertNotIn(junk, line)


class CollapseTests(unittest.TestCase):
    def test_consecutive_duplicates_collapse_across_items(self):
        items = ["skill lands.\n🏁 the house goes down.",
                 "🏁 the house goes down.",
                 "🏁 Peacethefirst wins."]
        self.assertEqual(
            _collapse_dup_lines(items),
            ["skill lands.\n🏁 the house goes down.",
             "🏁 Peacethefirst wins."])

    def test_trailing_segment_repeat_collapses(self):
        # the live shape: the win line ends the skill message AND is
        # appended again as its own item by the caller
        items = ["🥋 you unleashes X! you lands 1184 — house at 0 HP. "
                 "💀 BRUTAL FINISH! 🏁 the house goes down.",
                 "🏁 the house goes down.",
                 "🏁 Peacethefirst wins."]
        self.assertEqual(
            _collapse_dup_lines(items),
            [items[0], "🏁 Peacethefirst wins."])

    def test_short_echoes_survive(self):
        items = ["he said yes.", "yes."]
        self.assertEqual(_collapse_dup_lines(items), items)

    def test_non_consecutive_repeats_survive(self):
        items = ["again.", "something else.", "again."]
        self.assertEqual(_collapse_dup_lines(items), items)

    def test_blank_line_breaks_the_run(self):
        items = ["same.\n\nsame."]
        self.assertEqual(_collapse_dup_lines(items), items)


class Ctx:
    def __init__(self, db):
        self.db = db


ADA = Player.from_sender("telegram", "111", "Ada")


def make_engine(suggest=None):
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text),
                        suggest=suggest)
    return engine, db, sent


def rig_kill(room, house_hp=5):
    room.state["house"]["hp"] = house_hp
    room.state["house"]["max_hp"] = house_hp
    room.state["house"]["shield"] = False
    room.state["house"].pop("dodge_next", None)
    room.state["you"]["atk"] = 500
    room.state["you"]["def"] = 0


class FinishFlowTests(unittest.TestCase):
    def test_kill_narrated_once_across_move_and_finale(self):
        engine, db, sent = make_engine(
            suggest=lambda prompt: CLEAN_NARRATION)
        room, _ = engine.start("test:narr1", "arena", ADA, kind="dm")
        rig_kill(room)
        msgs = engine.move("test:narr1", "attack", ADA, kind="dm")
        joined = "\n".join(msgs)
        self.assertEqual(joined.count("🎲"), 1, joined)
        # the game's finish line also appears exactly once
        self.assertEqual(
            joined.count("🏁 the house goes down — the arena is "
                         "yours."), 1, joined)
        # and the arena's per-game best now records
        from nomorals.games.achievements import get_leaderboard
        board = get_leaderboard(db, "arena")
        self.assertTrue(board and board[0]["score"] > 0, board)

    def test_finale_still_fires_when_move_never_narrated(self):
        engine, db, sent = make_engine(
            suggest=lambda prompt: CLEAN_NARRATION)
        room, _ = engine.start("test:narr2", "arena", ADA, kind="dm")
        room.state["house"]["hp"] = 0
        room.state["done"] = True
        out = engine.quit("test:narr2")
        joined = "\n".join(out)
        self.assertEqual(joined.count("🎲"), 1, joined)

    def test_scaffold_model_never_posts_through_engine(self):
        engine, db, sent = make_engine(suggest=lambda prompt: SCAFFOLD)
        room, _ = engine.start("test:narr3", "arena", ADA, kind="dm")
        rig_kill(room)
        msgs = engine.move("test:narr3", "attack", ADA, kind="dm")
        joined = "\n".join(msgs)
        for junk in ("Analyze the Request", "Constraint Checklist",
                     "Confidence Score"):
            self.assertNotIn(junk, joined)
        # the kill still got its one (template) narration
        self.assertEqual(joined.count("🎲"), 1, joined)


class ArenaScoreTests(unittest.TestCase):
    def make_room(self):
        game = BattleArenaGame()
        room = Room(id="r1", game="arena", chat_key="test:s",
                    platform="test", kind="dm",
                    players=[ADA], state={})
        import random
        room.state = game.new_state(random.Random(3))
        return game, room

    def test_no_damage_scores_zero(self):
        game, room = self.make_room()
        self.assertEqual(game.score(room, ADA), 0)

    def test_loss_still_scores_damage(self):
        game, room = self.make_room()
        room.state["dmg_dealt"] = 45
        room.state["you"]["hp"] = 0
        room.state["house"]["hp"] = 30
        self.assertEqual(game.score(room, ADA), 45)

    def test_kill_adds_finish_bonus(self):
        game, room = self.make_room()
        room.state["dmg_dealt"] = 120
        room.state["house"]["hp"] = 0
        room.state["you"]["hp"] = 50  # untouched: full style bonus
        self.assertEqual(game.score(room, ADA), 120 + 25 + 25)

    def test_brutal_finish_scales_with_damage(self):
        game, room = self.make_room()
        room.state["dmg_dealt"] = 120
        room.state["house"]["hp"] = 0
        room.state["you"]["hp"] = 50
        room.state["brutal_finish"] = "you"
        brutal = game.score(room, ADA)
        room.state["dmg_dealt"] = 240
        bigger = game.score(room, ADA)
        self.assertEqual(brutal, 120 + 25 + 25 + 60)
        self.assertGreater(bigger, brutal)

    def test_house_seat_scores_zero(self):
        game, room = self.make_room()
        room.state["dmg_dealt"] = 90
        house = Player(key="ai:arena", platform="ai", name="The House",
                       is_ai=True)
        self.assertEqual(game.score(room, house), 0)

    def test_hit_accumulates_damage_dealt(self):
        game, room = self.make_room()
        from nomorals.games.ai import GameMind
        room.state["house"]["def"] = 0
        room.state["you"]["atk"] = 30
        mind = GameMind()
        game._hit(room, "you", "house", mind)
        dealt = room.state["dmg_dealt"]
        self.assertGreater(dealt, 0)
        self.assertEqual(dealt, 50 - max(0, room.state["house"]["hp"]))
        # the house hitting back is not the player's damage
        game._hit(room, "house", "you", mind)
        self.assertEqual(room.state["dmg_dealt"], dealt)


if __name__ == "__main__":
    unittest.main()
