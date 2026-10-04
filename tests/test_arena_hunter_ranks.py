"""Hunter-rank scaling for the battle arena house AI (Solo Leveling style):
rank thresholds, stat mirroring, and skill-gated combat smarts."""
import random
import unittest
from types import SimpleNamespace

from nomorals.games.ai import GameMind
from nomorals.games.games.ambitious import BattleArenaGame


class FakeRoom:
    def __init__(self, prog):
        self.state = {"progression": {"p1": prog}}
        self.humans = [SimpleNamespace(key="p1")]


def make_game(prog):
    game = BattleArenaGame()
    room = FakeRoom(prog)
    state = game.new_state(random.Random(1))
    room.state.update(state)
    room.state["progression"] = {"p1": prog}
    game._apply_progression(room)
    return room


class HunterRankTest(unittest.TestCase):
    def test_rank_thresholds(self) -> None:
        self.assertEqual(BattleArenaGame.house_rank_for(1), ("E", 0))
        self.assertEqual(BattleArenaGame.house_rank_for(2), ("E", 0))
        self.assertEqual(BattleArenaGame.house_rank_for(3), ("D", 1))
        self.assertEqual(BattleArenaGame.house_rank_for(5), ("C", 2))
        self.assertEqual(BattleArenaGame.house_rank_for(8), ("B", 3))
        self.assertEqual(BattleArenaGame.house_rank_for(12), ("A", 4))
        self.assertEqual(BattleArenaGame.house_rank_for(16), ("S", 5))
        self.assertEqual(BattleArenaGame.house_rank_for(99), ("S", 5))

    def test_e_rank_mirrors_small_share(self) -> None:
        room = make_game({"level": 1, "max_hp": 20, "atk": 4, "def": 2})
        h = room.state["house"]
        # E: 40% share, no flat bonus
        self.assertEqual(h["max_hp"], 50 + 8)
        self.assertEqual(h["atk"], 10 + 1)   # int(4 * 0.4)
        self.assertEqual(h["def"], 5 + 0)    # int(2 * 0.4)
        self.assertEqual(room.state["house_rank"], "E")
        self.assertEqual(room.state["house_skill"], 0)

    def test_b_rank_scales_up(self) -> None:
        room = make_game({"level": 8, "max_hp": 40, "atk": 8, "def": 4})
        h = room.state["house"]
        # B: 70% share + flat (12 hp, 2 atk, 1 def)
        self.assertEqual(h["max_hp"], 50 + 28 + 12)
        self.assertEqual(h["atk"], 10 + 5 + 2)
        self.assertEqual(h["def"], 5 + 2 + 1)
        self.assertEqual(room.state["house_rank"], "B")
        self.assertEqual(room.state["house_skill"], 3)

    def test_s_rank_is_near_mirror(self) -> None:
        room = make_game({"level": 20, "max_hp": 100, "atk": 20, "def": 10})
        h = room.state["house"]
        self.assertEqual(h["max_hp"], 50 + 90 + 24)
        self.assertEqual(room.state["house_rank"], "S")
        self.assertEqual(room.state["house_skill"], 5)

    def test_house_never_fully_mirrors_bonus(self) -> None:
        # the mirrored *share* stays below 1.0 at every rank — your own
        # progression is always the bigger number
        for lvl in (1, 5, 10, 16, 30):
            room = make_game({"level": lvl, "max_hp": 60, "atk": 12,
                              "def": 6})
            _n, _m, share, _fh, _fa, _fd = BattleArenaGame.HUNTER_RANKS[
                room.state["house_skill"]]
            self.assertLess(share, 1.0)
            self.assertLess(int(60 * share), 60)

    def test_s_rank_is_a_wall(self) -> None:
        # Solo Leveling rule: at S-rank the hunter out-stats you raw.
        # You win with gear, potions, and better play — not bigger numbers.
        room = make_game({"level": 20, "max_hp": 100, "atk": 20, "def": 10})
        h, y = room.state["house"], room.state["you"]
        self.assertGreater(h["max_hp"], y["max_hp"])


class RankSkillCombatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mind = GameMind()

    def test_veteran_potions_earlier(self) -> None:
        me = {"hp": 20, "max_hp": 50, "potions": 1, "atk": 10, "def": 5}
        foe = {"hp": 50, "atk": 5, "def": 5}
        self.assertEqual(
            self.mind.combat_move(me, foe, skill=5)["action"], "potion")
        self.assertNotEqual(
            self.mind.combat_move(me, foe, skill=0)["action"], "potion")

    def test_veteran_furies_more_freely(self) -> None:
        me = {"hp": 34, "max_hp": 50, "potions": 0, "atk": 10, "def": 5,
              "fury_cd": 0}
        foe = {"hp": 50, "atk": 5, "def": 5}
        self.assertEqual(
            self.mind.combat_move(me, foe, skill=5)["action"], "fury")
        self.assertEqual(
            self.mind.combat_move(me, foe, skill=0)["action"], "attack")

    def test_skill_defaults_to_rookie(self) -> None:
        # old callers (no skill kwarg) behave like E-rank
        me = {"hp": 10, "max_hp": 50, "potions": 1}
        self.assertEqual(
            self.mind.combat_move(me, {"attack": 5})["action"], "potion")

    def test_skill_clamped(self) -> None:
        me = {"hp": 20, "max_hp": 50, "potions": 1, "atk": 10, "def": 5}
        foe = {"hp": 50, "atk": 5, "def": 5}
        self.assertEqual(
            self.mind.combat_move(me, foe, skill=99)["action"], "potion")


if __name__ == "__main__":
    unittest.main()
