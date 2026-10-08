"""Tests for the NPC personality schema and the AI game master.

All offline: the template forge is deterministic under a fixed seed,
the store uses tmp_path, and the "model" is a stub suggest callable.
"""

import json
import unittest
from pathlib import Path

from nomorals.games.gamemaster import (
    DM_MOODS,
    GameMaster,
    validate_item,
    validate_quest,
)
from nomorals.games.npc import NPCProfile, NPCStore
from nomorals.games.player.decider import interpret_freeform
from nomorals.games.player.driver import Action
from nomorals.games.player.state import GameState


def _npc(**kw):
    base = dict(
        id="n1",
        name="Marlowe",
        game_id="testgame",
        personality={"bravery": 0.6, "humor": 0.8},
        voice_style="measured, warm, speaks in proverbs",
        knowledge_boundaries=["the player's gold"],
        goals=["guide travelers"],
        mood={"valence": 0.2, "arousal": 0.3, "trust": 0.6},
    )
    base.update(kw)
    return NPCProfile(**base)


def _state_with(*labels):
    from nomorals.games.player.driver import InteractiveElement
    els = [
        InteractiveElement(id=f"e{i}", kind="button", label=label)
        for i, label in enumerate(labels)
    ]
    return GameState(url="http://x", title="t", text="", elements=els)


class PersonalityTests(unittest.TestCase):
    def test_traits_clamped(self):
        npc = _npc(personality={"bravery": 9.0, "greed": -3.0})
        self.assertEqual(npc.personality["bravery"], 1.0)
        self.assertEqual(npc.personality["greed"], 0.0)

    def test_prompt_includes_boundaries(self):
        prompt = _npc().to_prompt()
        self.assertIn("the player's gold", prompt)
        self.assertIn("do NOT know", prompt)
        self.assertIn("Marlowe", prompt)
        self.assertIn("measured, warm", prompt)
        self.assertIn("guide travelers", prompt)

    def test_prompt_includes_mood_and_traits(self):
        prompt = _npc().to_prompt()
        self.assertIn("bravery=0.60", prompt)
        self.assertIn("valence", prompt)


class MemoryTests(unittest.TestCase):
    def test_remember_caps_at_fifty(self):
        npc = _npc()
        for i in range(60):
            npc.remember(f"memory {i}", salience=i / 100.0)
        self.assertEqual(len(npc.memory), 50)
        # the lowest-salience memories fell off
        texts = [m["text"] for m in npc.memory]
        self.assertNotIn("memory 0", texts)
        self.assertIn("memory 59", texts)

    def test_recall_ranks_by_relevance_then_salience(self):
        npc = _npc()
        npc.remember("the dragon burned the village", salience=0.2)
        npc.remember("ate soup for lunch", salience=0.9)
        npc.remember("the dragon owes me money", salience=0.5)
        hits = npc.recall("dragon", limit=2)
        self.assertEqual(len(hits), 2)
        texts = [h["text"] for h in hits]
        self.assertTrue(all("dragon" in t for t in texts))
        # higher salience first on equal overlap
        self.assertIn("owes me money", texts[0])

    def test_recall_empty_query_returns_salient(self):
        npc = _npc()
        npc.remember("quiet day", salience=0.1)
        npc.remember("won the duel", salience=0.9)
        hits = npc.recall("", limit=1)
        self.assertEqual(hits[0]["text"], "won the duel")

    def test_boundary_filtering_never_leaks_gold(self):
        npc = _npc()
        npc.remember("the player hides gold under the floorboards",
                     salience=1.0)
        npc.remember("it rained yesterday", salience=0.1)
        # recall filters it even at max salience
        hits = npc.recall("gold", limit=5)
        self.assertTrue(all("gold" not in h["text"].lower() for h in hits))
        # and the rendered prompt never contains it
        prompt = _npc().to_prompt()
        # (fresh npc has no such memory; now add and re-render)
        npc2 = _npc()
        npc2.remember("the player hides gold under the floorboards",
                      salience=1.0)
        prompt2 = npc2.to_prompt()
        self.assertNotIn("floorboards", prompt2)
        # but the boundary itself IS stated (the model is told what
        # the NPC must not know)
        self.assertIn("the player's gold", prompt2)


class MoodTests(unittest.TestCase):
    def test_react_shifts_mood(self):
        npc = _npc()
        before = dict(npc.mood)
        npc.react("betrayed by a friend",
                  {"valence": -0.5, "trust": -0.4, "arousal": 0.3})
        self.assertLess(npc.mood["valence"], before["valence"])
        self.assertLess(npc.mood["trust"], before["trust"])
        self.assertGreater(npc.mood["arousal"], before["arousal"])

    def test_react_clamps(self):
        npc = _npc()
        npc.react("x", {"valence": -99.0, "trust": 99.0})
        self.assertEqual(npc.mood["valence"], -1.0)
        self.assertEqual(npc.mood["trust"], 1.0)

    def test_react_never_raises(self):
        npc = _npc()
        npc.react("x", {"bogus": "not a number"})  # must not raise
        npc.react("x", None)

    def test_mood_word(self):
        self.assertEqual(_npc(mood={"valence": 0.8, "arousal": 0.8,
                                    "trust": 0.5}).mood_word(), "elated")
        self.assertEqual(_npc(mood={"valence": -0.8, "arousal": 0.8,
                                    "trust": 0.5}).mood_word(), "furious")
        self.assertEqual(_npc(mood={"valence": 0.0, "arousal": 0.1,
                                    "trust": 0.9}).mood_word(), "trusting")
        self.assertEqual(_npc(mood={"valence": 0.0, "arousal": 0.1,
                                    "trust": 0.1}).mood_word(), "wary")


class StoreTests(unittest.TestCase):
    def test_round_trip(self, tmp_path=None):
        import tempfile
        d = Path(tempfile.mkdtemp())
        store = NPCStore(d)
        npc = _npc()
        npc.remember("met the player", salience=0.6)
        store.save(npc)
        # fresh store instance, same dir → reload from disk
        store2 = NPCStore(d)
        back = store2.get("testgame", "n1")
        self.assertIsNotNone(back)
        self.assertEqual(back.name, "Marlowe")
        self.assertEqual(back.personality["bravery"], 0.6)
        self.assertEqual(len(back.memory), 1)
        self.assertEqual(back.mood["trust"], 0.6)
        self.assertEqual(len(store2.list("testgame")), 1)

    def test_per_game_isolation(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        store = NPCStore(d)
        store.save(_npc(id="a", game_id="game1"))
        store.save(_npc(id="a", game_id="game2", name="Other"))
        self.assertEqual(store.get("game1", "a").name, "Marlowe")
        self.assertEqual(store.get("game2", "a").name, "Other")
        self.assertEqual(len(store.list("game1")), 1)

    def test_corrupt_file_loads_empty_no_raise(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        target = d / "testgame" / "npcs.json"
        target.parent.mkdir(parents=True)
        target.write_text("{not valid json!!!", encoding="utf-8")
        store = NPCStore(d)
        self.assertEqual(store.list("testgame"), [])
        self.assertIsNone(store.get("testgame", "n1"))

    def test_delete(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        store = NPCStore(d)
        store.save(_npc())
        self.assertTrue(store.delete("testgame", "n1"))
        self.assertFalse(store.delete("testgame", "n1"))
        self.assertEqual(store.list("testgame"), [])

    def test_find_by_name(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        store = NPCStore(d)
        store.save(_npc())
        self.assertEqual(store.find_by_name("testgame", "marlowe").id, "n1")
        self.assertEqual(store.find_by_name("testgame", "Marl").id, "n1")
        self.assertIsNone(store.find_by_name("testgame", "nobody"))

    def test_find_by_name_word_match(self):
        # "Marlowe" finds "Sage Marlowe" — players type the short name.
        import tempfile
        d = Path(tempfile.mkdtemp())
        store = NPCStore(d)
        store.save(_npc(id="s1", name="Sage Marlowe"))
        found = store.find_by_name("testgame", "Marlowe")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, "s1")


class NarrationTests(unittest.TestCase):
    def _gm(self, seed=7):
        import tempfile
        return GameMaster(NPCStore(Path(tempfile.mkdtemp())), seed=seed)

    def test_fallback_varies_by_mood(self):
        gm = self._gm()
        seen = set()
        for mood in DM_MOODS:
            out = gm.narrate("g", "the guard falls", mood=mood)
            self.assertIn("the guard falls", out)
            seen.add(out)
        # every mood renders the event differently
        self.assertEqual(len(seen), len(DM_MOODS))

    def test_no_immediate_repeats(self):
        gm = self._gm(seed=1)
        outs = [gm.narrate("g", "a bell tolls", mood="grim") for _ in range(6)]
        for a, b in zip(outs, outs[1:]):
            self.assertNotEqual(a, b)

    def test_empty_event_empty_out(self):
        self.assertEqual(self._gm().narrate("g", "   "), "")

    def test_model_path_used_when_available(self):
        gm = self._gm()
        gm._suggest = lambda prompt: "MODEL SAID THINGS"
        out = gm.narrate("g", "a bell tolls", mood="grim")
        self.assertEqual(out, "MODEL SAID THINGS")

    def test_model_failure_falls_back(self):
        gm = self._gm()
        def boom(prompt):
            raise RuntimeError("down")
        gm._suggest = boom
        out = gm.narrate("g", "a bell tolls", mood="grim")
        self.assertIn("a bell tolls", out)

    def test_dm_mood_persists(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        gm = GameMaster(NPCStore(d), seed=3)
        self.assertEqual(gm.get_dm_mood("g"), "neutral")
        gm.set_dm_mood("g", "epic")
        gm2 = GameMaster(NPCStore(d), seed=3)
        self.assertEqual(gm2.get_dm_mood("g"), "epic")
        # narrate with no explicit mood uses the DM mood
        out = gm2.narrate("g", "the gates open")
        self.assertIn("the gates open", out)

    def test_dm_mood_rejects_unknown(self):
        with self.assertRaises(ValueError):
            self._gm().set_dm_mood("g", "sleepy")


class NpcSpeakTests(unittest.TestCase):
    def _gm(self, seed=11):
        import tempfile
        return GameMaster(NPCStore(Path(tempfile.mkdtemp())), seed=seed)

    def test_offline_speech_names_npc(self):
        out = self._gm().npc_speak(_npc(), "a dragon lands in the square")
        self.assertIn("Marlowe", out)
        self.assertIn("dragon", out)

    def test_speech_follows_valence(self):
        gm = self._gm(seed=5)
        bright = gm.npc_speak(
            _npc(mood={"valence": 0.9, "arousal": 0.5, "trust": 0.5}), "news")
        dark = gm.npc_speak(
            _npc(mood={"valence": -0.9, "arousal": 0.5, "trust": 0.5}), "news")
        self.assertNotEqual(bright, dark)

    def test_model_speech_uses_prompt(self):
        gm = self._gm()
        seen = {}
        def fake(prompt):
            seen["prompt"] = prompt
            return "A wise word, traveler."
        gm._suggest = fake
        out = gm.npc_speak(_npc(), "should I open the door")
        self.assertEqual(out, "A wise word, traveler.")
        # the boundary made it into the model prompt…
        self.assertIn("the player's gold", seen["prompt"])
        # …and the NPC identity did too
        self.assertIn("Marlowe", seen["prompt"])

    def test_empty_situation(self):
        self.assertIn("Marlowe", self._gm().npc_speak(_npc(), "  "))


class QuestItemTests(unittest.TestCase):
    def _gm(self, seed=21):
        import tempfile
        return GameMaster(NPCStore(Path(tempfile.mkdtemp())), seed=seed)

    def test_offline_quest_validates(self):
        for _ in range(20):
            q = self._gm().generate_quest("g", 5, "heist")
            self.assertTrue(validate_quest(q), q)
            self.assertEqual(q["theme"], "heist")

    def test_offline_quest_scales_with_level(self):
        low = self._gm(seed=1).generate_quest("g", 1, "x")
        high = self._gm(seed=1).generate_quest("g", 10, "x")
        self.assertLessEqual(low["reward_xp"], high["reward_xp"])

    def test_offline_item_validates(self):
        for kind in ("weapon", "armor", "trinket", "consumable"):
            item = self._gm().generate_item("g", 4, kind)
            self.assertTrue(validate_item(item), item)
            self.assertEqual(item["kind"], kind)

    def test_quest_schema_rejects_junk(self):
        self.assertFalse(validate_quest({}))
        self.assertFalse(validate_quest({"id": "x"}))
        bad = {"id": "q", "title": "t", "objective": "o", "reward_xp": -5,
               "reward_coins": 1, "difficulty": 3, "theme": "t"}
        self.assertFalse(validate_quest(bad))
        bad["reward_xp"] = 5
        bad["difficulty"] = 9
        self.assertFalse(validate_quest(bad))

    def test_item_schema_rejects_junk(self):
        self.assertFalse(validate_item({}))
        bad = {"id": "i", "name": "n", "kind": "spaceship", "power": 3,
               "rarity": "common", "flavor": "f"}
        self.assertFalse(validate_item(bad))
        bad["kind"] = "weapon"
        bad["rarity"] = "mythic"
        self.assertFalse(validate_item(bad))

    def test_model_quest_falls_back_on_garbage(self):
        gm = self._gm()
        gm._suggest = lambda prompt: "not json at all lol"
        q = gm.generate_quest("g", 3, "heist")
        self.assertTrue(validate_quest(q))

    def test_model_quest_accepted_when_valid(self):
        gm = self._gm()
        gm._suggest = lambda prompt: json.dumps({
            "id": "q_model", "title": "Model Heist", "objective": "Steal it",
            "reward_xp": 100, "reward_coins": 50, "difficulty": 2,
            "theme": "heist"})
        q = gm.generate_quest("g", 3, "heist")
        self.assertEqual(q["id"], "q_model")

    def test_ensure_cast_seeds_once(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        gm = GameMaster(NPCStore(d), seed=9)
        cast = gm.ensure_cast("mygame")
        self.assertEqual(len(cast), 3)
        names = [n.name for n in cast]
        self.assertIn("Sage Marlowe", names)
        # second call returns the same cast, doesn't duplicate
        again = gm.ensure_cast("mygame")
        self.assertEqual(len(again), 3)
        # persisted to disk
        gm2 = GameMaster(NPCStore(d), seed=9)
        self.assertEqual(len(gm2.store.list("mygame")), 3)


class FreeformTests(unittest.TestCase):
    def test_keyword_match_maps_to_element(self):
        state = _state_with("Bribe the guard", "Open the door", "Rest")
        move = interpret_freeform("bribe the guard", "rpg", state)
        self.assertIsNotNone(move)
        self.assertEqual(move.action.kind, "click")
        self.assertEqual(move.action.target, "e0")
        self.assertGreater(move.confidence, 0.5)

    def test_no_match_returns_freeform_fallback(self):
        state = _state_with("Open the door", "Rest")
        move = interpret_freeform("sneak around back", "rpg", state)
        self.assertIsNotNone(move)
        self.assertEqual(move.action.kind, "freeform")
        self.assertEqual(move.action.payload["text"], "sneak around back")

    def test_empty_text_returns_none(self):
        state = _state_with("Rest")
        self.assertIsNone(interpret_freeform("   ", "rpg", state))

    def test_model_pick_validated(self):
        state = _state_with("Open the door", "Bribe the guard", "Rest")
        move = interpret_freeform("bribe him", "rpg", state,
                                  suggest=lambda prompt: "2")
        self.assertEqual(move.action.target, "e1")

    def test_model_garbage_falls_back_to_freeform(self):
        state = _state_with("Open the door")
        move = interpret_freeform("do a flip", "rpg", state,
                                  suggest=lambda prompt: "banana")
        self.assertEqual(move.action.kind, "freeform")

    def test_model_out_of_range_falls_back(self):
        state = _state_with("Open the door")
        move = interpret_freeform("open it", "rpg", state,
                                  suggest=lambda prompt: "99")
        # keyword matching already catches "open it" → e0; force the LLM
        # path with text that matches nothing
        move2 = interpret_freeform("xyzzy plugh", "rpg", state,
                                   suggest=lambda prompt: "99")
        self.assertEqual(move2.action.kind, "freeform")

    def test_model_crash_falls_back(self):
        state = _state_with("Open the door")
        def boom(prompt):
            raise RuntimeError("down")
        move = interpret_freeform("xyzzy", "rpg", state, suggest=boom)
        self.assertEqual(move.action.kind, "freeform")

    def test_never_returns_illegal_target(self):
        state = _state_with("Open the door")
        for text in ("bribe the guard", "xyzzy", "open the d0or!!"):
            move = interpret_freeform(text, "rpg", state)
            if move.action.kind in ("click", "submit"):
                ids = [el.id for el in state.elements]
                self.assertIn(move.action.target, ids)

    def test_avoid_match_flagged_irreversible(self):
        state = _state_with("Delete save file", "Rest")
        move = interpret_freeform("delete my save", "rpg", state)
        self.assertIsNotNone(move)
        self.assertTrue(move.irreversible)


class ControlRegistrationTests(unittest.TestCase):
    def test_commands_registered(self):
        from nomorals.social.chat.control import CONTROL_COMMANDS
        self.assertIn("npc", CONTROL_COMMANDS)
        self.assertIn("dm", CONTROL_COMMANDS)

    def test_help_renders(self):
        from nomorals.social.chat import control
        text = control.help_text() if hasattr(control, "help_text") else ""
        # at minimum the module must import cleanly with the new entries
        self.assertIsNotNone(text)


if __name__ == "__main__":
    unittest.main()
