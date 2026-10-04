"""Dynamic enemy scaling: power ratings, rolled enemies, house skill
casting, rank-scaled XP, and the 20% power cap."""
import random
import unittest
from types import SimpleNamespace

from nomorals.games.ai import GameMind
from nomorals.games.enemies import (
    GRADE_POOLS,
    SKILL_COUNTS,
    roll_enemy,
    roll_enemy_gear,
    roll_enemy_name,
    roll_enemy_skills,
)
from nomorals.games.games.ambitious import BattleArenaGame
from nomorals.games.power import (
    fighter_power,
    power_bar,
    skill_power_score,
    skills_power,
)
from nomorals.games.skills import SKILL_CATALOG


class FakeRoom:
    def __init__(self, prog):
        self.state = {"progression": {"p1": prog}}
        self.humans = [SimpleNamespace(key="p1")]


def make_room(prog, seed=7):
    game = BattleArenaGame()
    room = FakeRoom(prog)
    room.state.update(game.new_state(random.Random(seed)))
    room.state["progression"] = {"p1": prog}
    mind = GameMind(seed=seed)
    game.setup(room, mind)
    return game, room, mind


class PowerTest(unittest.TestCase):
    def test_formula(self) -> None:
        f = {"max_hp": 50, "atk": 10, "def": 5}
        self.assertEqual(fighter_power(f), 50 + 100 + 50)

    def test_gear_counts_through_stats(self) -> None:
        base = {"max_hp": 50, "atk": 10, "def": 5}
        geared = {"max_hp": 50, "atk": 30, "def": 15}
        self.assertEqual(fighter_power(geared) - fighter_power(base),
                         20 * 10 + 10 * 10)

    def test_strike_skill_adds_power(self) -> None:
        f = {"max_hp": 50, "atk": 10, "def": 5}
        self.assertGreater(fighter_power(f, ["dragon_punch"]),
                           fighter_power(f))

    def test_higher_tier_more_power(self) -> None:
        f = {"max_hp": 50, "atk": 10, "def": 5}
        self.assertGreater(fighter_power(f, ["dragon_punch"],
                                         {"dragon_punch": 3}),
                           fighter_power(f, ["dragon_punch"],
                                         {"dragon_punch": 1}))

    def test_passive_skills_add_nothing_directly(self) -> None:
        # passives are already folded into stats — no double count
        f = {"max_hp": 50, "atk": 10, "def": 5}
        self.assertEqual(skills_power(["iron_skin"]), 0)

    def test_skill_power_score_positive(self) -> None:
        for slug, defn in SKILL_CATALOG.items():
            if defn.kind == "active":
                self.assertGreater(skill_power_score(defn), 0, slug)

    def test_power_bar_shape(self) -> None:
        bar = power_bar(100, 100)
        self.assertEqual(len(bar), 10)
        self.assertIn("▰", bar)


class EnemyRollTest(unittest.TestCase):
    def test_name_has_rank_title(self) -> None:
        name = roll_enemy_name(random.Random(1), 5)
        self.assertIn(" ", name)
        self.assertTrue(len(name.split()) >= 2)

    def test_grade_pools_by_rank(self) -> None:
        self.assertNotIn("legendary", GRADE_POOLS[0])
        self.assertNotIn("legendary", GRADE_POOLS[2])
        self.assertEqual(GRADE_POOLS[5], ("legendary",))
        self.assertNotIn("myth", GRADE_POOLS[5])

    def test_e_rank_never_legendary_gear(self) -> None:
        rng = random.Random(11)
        for _ in range(30):
            gear = roll_enemy_gear(rng, 0)
            for piece in gear.values():
                self.assertNotIn(piece["grade"], ("legendary", "myth"))

    def test_s_rank_only_legendary_or_set(self) -> None:
        # S-rank rolls legendary pieces, or a matched epic set
        # (storm/shadow carry the set bonus instead of raw grade)
        rng = random.Random(13)
        for _ in range(10):
            gear = roll_enemy_gear(rng, 5)
            for piece in gear.values():
                self.assertIn(piece["grade"], ("legendary", "epic"))
                if piece["grade"] == "epic":
                    self.assertTrue(piece["set"])

    def test_skill_counts_grow_with_rank(self) -> None:
        self.assertEqual(SKILL_COUNTS[0], 0)
        self.assertGreater(SKILL_COUNTS[5], SKILL_COUNTS[0])

    def test_e_rank_has_no_skills(self) -> None:
        rng = random.Random(17)
        for _ in range(10):
            self.assertEqual(roll_enemy_skills(rng, 0), {})

    def test_slaying_force_never_rolled(self) -> None:
        rng = random.Random(19)
        for rank in range(6):
            for _ in range(20):
                skills = roll_enemy_skills(rng, rank)
                self.assertNotIn("slaying_force", skills)

    def test_rolls_are_deterministic(self) -> None:
        e1 = roll_enemy(random.Random(23), 3, 500)
        e2 = roll_enemy(random.Random(23), 3, 500)
        self.assertEqual(e1, e2)


class SpawnIntegrationTest(unittest.TestCase):
    def test_setup_spawns_named_enemy(self) -> None:
        _game, room, _mind = make_room(
            {"level": 5, "max_hp": 30, "atk": 6, "def": 3})
        self.assertIn("house_name", room.state)
        self.assertTrue(room.state["house_name"])
        self.assertIn("house_skills", room.state)
        self.assertIn("house_power", room.state)
        self.assertIn("player_power", room.state)

    def test_power_cap_holds(self) -> None:
        # across ranks and seeds the house never out-powers the
        # player by more than 20% — 30% when it fights with forbidden
        # techniques (they're scary, but cooldown-gated)
        from nomorals.games.skills import is_enemy_skill
        for level, prog in (
                (1, {"level": 1, "max_hp": 0, "atk": 0, "def": 0}),
                (8, {"level": 8, "max_hp": 40, "atk": 8, "def": 4}),
                (20, {"level": 20, "max_hp": 100, "atk": 20,
                      "def": 10})):
            for seed in range(5):
                _game, room, _mind = make_room(prog, seed=seed)
                you = room.state["player_power"]
                foe = room.state["house_power"]
                skills = room.state.get("house_skills", {})
                cap = 1.3 if any(is_enemy_skill(s) for s in skills) \
                    else 1.2
                self.assertLessEqual(
                    foe, you * cap + 1,
                    f"level {level} seed {seed}: {foe} vs {you}")

    def test_enemy_gear_is_battle_worn(self) -> None:
        # gear contributes at 50%: house atk gain <= half the piece atk
        _game, room, _mind = make_room(
            {"level": 8, "max_hp": 40, "atk": 8, "def": 4}, seed=3)
        gear = room.state["house_gear"]
        weapon_atk = int((gear.get("weapon") or {}).get("atk", 0))
        # B-rank base atk is 17; gear adds at most half the piece
        self.assertLessEqual(room.state["house"]["atk"],
                             17 + weapon_atk // 2 + 1)

    def test_setup_shows_power_line(self) -> None:
        game = BattleArenaGame()
        room = FakeRoom({"level": 3, "max_hp": 20, "atk": 4, "def": 2})
        room.state.update(game.new_state(random.Random(5)))
        room.state["progression"] = {"p1": {"level": 3, "max_hp": 20,
                                            "atk": 4, "def": 2}}
        text = game.setup(room, GameMind(seed=5))
        self.assertIn("power", text)
        self.assertIn(room.state["house_name"], text)

    def test_higher_rank_rolls_better_gear_on_average(self) -> None:
        # average weapon atk rises with rank (grade-gated pools)
        def avg_weapon(rank, seeds=12):
            total = 0
            rng = random.Random(1000 + rank)
            for _ in range(seeds):
                gear = roll_enemy_gear(rng, rank)
                total += int((gear.get("weapon") or {}).get("atk", 0))
            return total / seeds
        self.assertGreater(avg_weapon(5), avg_weapon(0))


class HouseSkillCastTest(unittest.TestCase):
    def _room_with_house_skills(self):
        game, room, mind = make_room(
            {"level": 8, "max_hp": 40, "atk": 8, "def": 4}, seed=7)
        room.state["house_skills"] = {"dragon_punch": 1,
                                      "second_wind": 1}
        room.state["house_skill_cd"] = {}
        room.state["house_skill_used"] = []
        return game, room, mind

    def test_house_casts_strike(self) -> None:
        game, room, mind = self._room_with_house_skills()
        # force the pick: nothing else ready matters, dragon punch is
        msg = game._cast_skill(room, None, "dragon_punch", mind,
                               src="house", dst="you")
        self.assertIsNotNone(msg)
        self.assertIn("Dragon Punch", msg)
        self.assertIn("the house", msg)
        self.assertLess(room.state["you"]["hp"],
                        room.state["you"]["max_hp"])

    def test_house_heals_when_bleeding(self) -> None:
        game, room, mind = self._room_with_house_skills()
        room.state["house"]["hp"] = 10
        pick = game._house_skill_pick(room, mind)
        self.assertEqual(pick, "second_wind")

    def test_house_cooldown_blocks_recast(self) -> None:
        game, room, mind = self._room_with_house_skills()
        room.state["house_skill_cd"] = {"dragon_punch": 2}
        msg = game._cast_skill(room, None, "dragon_punch", mind,
                               src="house", dst="you")
        self.assertIsNone(msg)  # silently picks another move

    def test_e_rank_never_casts(self) -> None:
        game, room, mind = make_room(
            {"level": 1, "max_hp": 0, "atk": 0, "def": 0}, seed=7)
        room.state["house_skills"] = {"dragon_punch": 1}
        self.assertIsNone(game._house_skill_pick(room, mind))

    def test_house_cooldowns_tick(self) -> None:
        game, room, mind = self._room_with_house_skills()
        room.state["house_skill_cd"] = {"dragon_punch": 2}
        game._house_act(room, mind, [])
        self.assertEqual(room.state["house_skill_cd"]["dragon_punch"], 1)

    def test_house_combo_fires(self) -> None:
        game, room, mind = make_room(
            {"level": 5, "max_hp": 30, "atk": 6, "def": 3}, seed=7)
        # give the house a set combo directly
        room.state["house"]["combo_every"] = 2
        room.state["house"]["combo_name"] = "test combo"
        room.state["house"]["combo_count"] = 1
        room.state["you"]["hp"] = 200
        room.state["you"]["max_hp"] = 200
        msg = game._hit(room, "house", "you", mind)
        self.assertIn("test combo", msg)


class XpScalingTest(unittest.TestCase):
    def _win_room(self, rank_idx, player_power=500, house_power=500):
        game = BattleArenaGame()
        room = FakeRoom({"level": 1, "max_hp": 0, "atk": 0, "def": 0})
        room.state.update(game.new_state(random.Random(1)))
        room.state["house_skill"] = rank_idx
        room.state["player_power"] = player_power
        room.state["house_power"] = house_power
        room.state["you"]["hp"] = room.state["you"]["max_hp"]
        return game, room

    def test_xp_grows_with_rank(self) -> None:
        xps = []
        for rank_idx in range(6):
            game, room = self._win_room(rank_idx)
            xps.append(game.xp_reward(True, room, room.humans[0]))
        for earlier, later in zip(xps, xps[1:]):
            self.assertGreaterEqual(later, earlier)
        self.assertGreater(xps[5], xps[0])

    def test_upset_bonus(self) -> None:
        game, room = self._win_room(2, player_power=400,
                                    house_power=500)
        upset = game.xp_reward(True, room, room.humans[0])
        _game2, room2 = self._win_room(2, player_power=500,
                                       house_power=400)
        routine = game.xp_reward(True, room2, room2.humans[0])
        self.assertGreater(upset, routine)

    def test_loss_xp_flat(self) -> None:
        from nomorals.games.progression import ARENA_LOSS_XP
        game, room = self._win_room(5)
        self.assertEqual(game.xp_reward(False, room, room.humans[0]),
                         ARENA_LOSS_XP)


if __name__ == "__main__":
    unittest.main()
