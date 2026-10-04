"""The ambitious table: long arcs, persistent state, and stakes that
outlive a single game.

* **world** — a continuing town that never ends: the room stays live
  between visits (the bridge resumes it instead of restarting), the
  simulation ticks once per visit, and the town you leave is the town
  you come back to.
* **battle_arena** — turn-to-death combat where shop items are real
  gear (the engine consumes them from the player's inventory on close).
* **escape_room** — cooperative puzzles, three strikes per lock, and a
  keycard that opens exactly one of them.
* **political** — three elections, campaign pledges, vetoes, and a
  mayor at the end.
"""
from __future__ import annotations

import random
import re
from typing import Any

from ..ai import GameMind
from ..players import Player
from .base import MultiGame, Room

__all__ = ["AMBITIOUS_GAMES"]

# ── world: the town's building catalog ──────────────────────────────────────

BUILDINGS: dict[str, dict[str, Any]] = {
    "house": {"cost_g": 10, "cost_f": 0, "cost_t": 0,
              "effect": "+2 max population"},
    "shed": {"cost_g": 8, "cost_f": 4, "cost_t": 1,
             "effect": "+2 food/day"},
    "farm": {"cost_g": 12, "cost_f": 6, "cost_t": 1,
             "effect": "+3 food/day"},
    "market": {"cost_g": 15, "cost_f": 0, "cost_t": 0,
               "effect": "trade price +1"},
    "workshop": {"cost_g": 12, "cost_f": 0, "cost_t": 2,
                 "effect": "+1 tool per craft"},
    "well": {"cost_g": 14, "cost_f": 2, "cost_t": 1,
             "effect": "famine losses halved"},
    "wall": {"cost_g": 18, "cost_f": 0, "cost_t": 2,
             "effect": "raids bounce off"},
    "temple": {"cost_g": 20, "cost_f": 6, "cost_t": 1,
               "effect": "plague keeps its distance"},
    "granary": {"cost_g": 10, "cost_f": 4, "cost_t": 0,
                "effect": "trade price +1"},
}

#: four-season wheel — 10 days each. Farm yield by season.
SEASONS: tuple[str, ...] = ("spring", "summer", "autumn", "winter")
SEASON_DAYS = 10
FARM_YIELD: dict[str, int] = {"spring": 5, "summer": 4,
                              "autumn": 6, "winter": 1}

#: weighted town events — the town has a life of its own. Each tick the
#: table is rolled once; nothing happens most days.
#: Rewards scale with day count so long-term play stays rewarding.
TOWN_EVENTS: tuple[tuple[str, int, str], ...] = (
    # (event, weight, flavor)
    ("rain", 3, "rain soaks the fields — +5 food"),
    ("caravan", 2, "a caravan crosses the gate — +6 gold"),
    ("festival", 2, "the festival brings a stranger — +1 person (−2 food)"),
    ("drought", 2, "the well sputters — −4 food"),
    ("merchant", 1, "a merchant lingers — trade pays +2 for 3 days"),
    ("plague", 1, "a cough moves through the houses"),
    ("raid", 1, "raiders hit the night market"),
    ("goldrush", 1, "prospectors strike a rich vein — +15 gold"),
    ("taxday", 1, "tax collectors make their rounds — +2 gold per 10 people"),
)

#: population milestones — the town earns titles as it grows
MILESTONES: tuple[tuple[int, str], ...] = (
    (20, "the huts are a village now"),
    (40, "the village is a town — the market ring is full"),
    (60, "the town is a city — the walls ring with hammers"),
)


def season_of(day: int) -> str:
    return SEASONS[(day - 1) // SEASON_DAYS % len(SEASONS)]


class WorldGame(MultiGame):
    name = "world"
    description = "a continuing town — seasons, events, and it just grows"
    min_players = 1
    max_players = 3
    ai_seats = 0
    move_timeout = 0
    rules = ("Your town, one action at a time: farm (+food, more in "
             "autumn, less in winter), mine (+gold), craft (+tools), "
             "trade (sell food), rest (+people, −food), or build "
             "house|shed|farm|market|workshop|well|wall|temple|granary. "
             "Four seasons turn every 10 days, and the town has a life "
             "of its own — rain, caravans, festivals, droughts, plague, "
             "raids. Leave and come back: the town keeps living.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"day": 1, "pop": 8, "food": 24, "gold": 12, "tools": 3,
                "buildings": {}, "max_pop": 10, "log": [],
                "merchant_days": 0, "milestone": 0}

    def setup(self, room, mind):
        s = room.state
        if s.get("day"):
            return self._report(room)
        return ("your town wakes on day one, in spring — 8 people, 24 "
                "food, 12 gold, 3 tools. farm · mine · craft · trade · "
                "rest · build <name>. seasons turn, events roll, and "
                "the town ticks after you act.")

    def _max_pop(self, room: Room) -> int:
        return 10 + 2 * room.state["buildings"].get("house", 0)

    def _food_per_day(self, room: Room) -> int:
        return 2 * room.state["buildings"].get("shed", 0) + \
            3 * room.state["buildings"].get("farm", 0)

    def _gold_per_day(self, room: Room) -> int:
        """Buildings generate gold income — the town earns, not just events."""
        b = room.state["buildings"]
        # market: trade hub, granary: surplus sales, workshop: tool sales,
        # temple: donations scale with population
        income = (2 * b.get("market", 0) +
                  1 * b.get("granary", 0) +
                  1 * b.get("workshop", 0))
        # temple donations scale with town size
        if b.get("temple"):
            income += max(1, room.state["pop"] // 10)
        return income

    def _mine_yield(self, room: Room) -> int:
        """Mine scales with progress — deeper shafts, better tools."""
        day = room.state["day"]
        # +3 base, +1 per 75 days, +1 per workshop level (better picks)
        return 3 + day // 75 + room.state["buildings"].get("workshop", 0)

    def _report(self, room: Room) -> str:
        s = room.state
        b = ", ".join(f"{k}×{v}" for k, v in s["buildings"].items()) \
            or "none yet"
        season = season_of(s["day"])
        days_left = SEASON_DAYS - ((s["day"] - 1) % SEASON_DAYS)
        lines = [f"day {s['day']} ({season}, {days_left} days in) — "
                 f"population {s['pop']}/{self._max_pop(room)} · "
                 f"food {s['food']} · gold {s['gold']} · tools "
                 f"{s['tools']}",
                 f"buildings: {b} · yield: +{self._food_per_day(room)} "
                 f"food/day, +{self._gold_per_day(room)} gold/day · eat "
                 f"1/person/day · farm pays "
                 f"{FARM_YIELD[season]} + farm levels"]
        if s["merchant_days"] > 0:
            lines.append(f"merchant in town — trade pays +2 for "
                         f"{s['merchant_days']} more day(s)")
        for line in s["log"][-4:]:
            lines.append(f"  {line}")
        lines.append("your move: farm · mine · craft · trade · rest · "
                     "build <name>")
        return "\n".join(lines)

    def _tick(self, room: Room) -> list[str]:
        s = room.state
        events: list[str] = []
        old_season = season_of(s["day"])
        s["day"] += 1
        s["food"] += self._food_per_day(room)
        # buildings generate gold income every day
        gold_income = self._gold_per_day(room)
        if gold_income > 0:
            s["gold"] += gold_income
            events.append(f"the town's enterprises bring +{gold_income} gold")
        # a season turning is a town-wide event
        if season_of(s["day"]) != old_season:
            events.append(f"the season turns — {season_of(s['day'])} "
                          f"arrives.")
        # the town's own life: one weighted roll per tick
        self._roll_event(room, events)
        eat = s["pop"]
        s["food"] -= eat
        if s["food"] < 0:
            lost = min(s["pop"] - 1, -s["food"] // 3 + 1)
            if s["buildings"].get("well"):
                lost = max(1, lost // 2)  # the well stretches the food
            s["pop"] = max(1, s["pop"] - lost)
            s["food"] = 0
            events.append(f"famine — {lost} left the town")
        elif (s["food"] >= s["pop"] * 1.3 and
              s["pop"] < self._max_pop(room) and
              season_of(s["day"]) != "winter"):
            if s["day"] % 2 == 0:
                s["pop"] += 1
                events.append("a family moved in (+1)")
        if s["merchant_days"] > 0:
            s["merchant_days"] -= 1
        # milestones: the town earns titles
        while (s["milestone"] < len(MILESTONES) and
               s["pop"] >= MILESTONES[s["milestone"]][0]):
            events.append("🏆 " + MILESTONES[s["milestone"]][1])
            s["milestone"] += 1
        s["log"].extend(events)
        s["log"] = s["log"][-12:]
        return events

    def _roll_event(self, room: Room, events: list[str]) -> None:
        s = room.state
        rng = room.rng()
        table = list(TOWN_EVENTS)
        if season_of(s["day"]) == "summer":
            # droughts bite harder in summer: double the weight
            table = [e for e in table if e[0] != "drought"] + \
                    [("drought", 4, "the summer heat cracks the fields — −4 food")]
        total_w = sum(w for _, w, _ in table)
        roll = rng.randrange(total_w)
        pick = None
        for name, w, flavor in table:
            roll -= w
            if roll < 0:
                pick = (name, flavor)
                break
        if pick is None:
            return
        name, flavor = pick
        b = s["buildings"]
        # event rewards scale with progress (day // 50 bonus)
        progress_bonus = s["day"] // 50
        if name == "rain":
            s["food"] += 5
        elif name == "caravan":
            gain = 6 + progress_bonus * 2
            s["gold"] += gain
            flavor = f"a caravan crosses the gate — +{gain} gold"
        elif name == "festival":
            if s["pop"] < self._max_pop(room) and s["food"] >= 2:
                s["pop"] += 1
                s["food"] -= 2
            else:
                gain = 2 + progress_bonus
                s["gold"] += gain
                flavor = f"the festival passes — +{gain} gold"
        elif name == "goldrush":
            gain = 15 + progress_bonus * 3
            s["gold"] += gain
            flavor = f"prospectors strike a rich vein — +{gain} gold"
        elif name == "taxday":
            gain = max(2, (s["pop"] // 10) * 2 + progress_bonus)
            s["gold"] += gain
            flavor = f"tax collectors make their rounds — +{gain} gold"
        elif name == "drought":
            s["food"] = max(0, s["food"] - 4)
        elif name == "merchant":
            s["merchant_days"] = 3
        elif name == "plague":
            if b.get("temple"):
                events.append("the plague comes — and the temple keeps "
                               "it at the gate.")
                return
            lost = min(2, s["pop"] - 1)
            if lost > 0:
                s["pop"] -= lost
                events.append(f"the plague takes {lost}.")
                return
        elif name == "raid":
            if b.get("wall"):
                events.append("raiders try the wall — and bounce off.")
                return
            lost = min(6, s["gold"])
            s["gold"] -= lost
            flavor = f"raiders take {lost} gold"
        events.append(flavor)

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        out: list[str] = []
        if t == "farm":
            gain = (FARM_YIELD[season_of(s["day"])] +
                    s["buildings"].get("farm", 0))
            s["food"] += gain
            out.append(f"the {season_of(s['day'])} fields give +{gain} "
                       "food.")
        elif t == "mine":
            gain = self._mine_yield(room)
            s["gold"] += gain
            out.append(f"the mine gives +{gain} gold.")
        elif t == "craft":
            bonus = s["buildings"].get("workshop", 0)
            if s["gold"] >= 2 and s["tools"] < 12:
                s["gold"] -= 2
                s["tools"] += 1 + bonus
                out.append(f"the forge makes +{1 + bonus} tool(s) (−2g).")
            else:
                out.append("crafting needs 2 gold and room in the "
                           "tool shed (12 max).")
        elif t == "trade":
            if s["food"] >= 5:
                price = (1 + s["buildings"].get("market", 0) +
                         s["buildings"].get("granary", 0) +
                         (2 if s["merchant_days"] > 0 else 0))
                sold = min(10, s["food"] - 2)
                s["food"] -= sold
                s["gold"] += sold * price
                note = " (merchant paying well)" if s["merchant_days"] > 0 else ""
                out.append(f"the {price}c/food market takes {sold} food "
                           f"for +{sold * price}g{note}.")
            else:
                out.append("the market wants 5+ food on the table.")
        elif t == "rest":
            if s["food"] >= 2 and s["pop"] < self._max_pop(room):
                s["food"] -= 2
                s["pop"] += 1
                out.append("the doors are open — +1 person (−2 food).")
            elif s["pop"] >= self._max_pop(room):
                out.append("the town is full — build a house first.")
            else:
                out.append("resting costs 2 food you don't have.")
        elif t.startswith("build "):
            name = t[6:].strip()
            spec = BUILDINGS.get(name)
            if spec is None:
                out.append("buildable: " + ", ".join(BUILDINGS) + ".")
            elif s["gold"] >= spec["cost_g"] and s["food"] >= spec["cost_f"] \
                    and s["tools"] >= spec["cost_t"]:
                s["gold"] -= spec["cost_g"]
                s["food"] -= spec["cost_f"]
                s["tools"] -= spec["cost_t"]
                s["buildings"][name] = \
                    int(s["buildings"].get(name, 0)) + 1
                out.append(f"a {name} rises — {spec['effect']}.")
            else:
                out.append(f"a {name} needs {spec['cost_g']}g, "
                           f"{spec['cost_f']}f, {spec['cost_t']}t.")
        else:
            out.append("farm · mine · craft · trade · rest · "
                       "build <name>.")
        out.extend(self._tick(room))
        out.append(self._report(room))
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return False  # the town doesn't end

    def score(self, room, player):
        s = room.state
        return (s["pop"] + sum(s["buildings"].values()) * 2 +
                s.get("milestone", 0) * 10)


# ── battle arena ─────────────────────────────────────────────────────────────

class BattleArenaGame(MultiGame):
    name = "arena"
    description = "turn-to-death combat — shop gear is real equipment"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 60
    rules = ("You vs the house: 50 HP each (your level adds more). attack (deal atk − their "
             "defense, 10% crits double it), focus (next hit +50%, "
             "costs your turn), fury (two 80% attacks, 2-turn cooldown), "
             "defend (halve the next hit), potion (+30 HP), skill <name> "
             "(martial arts you learned with /skill — each has its own "
             "cooldown), or equip "
             "shop gear — /game shop buys real swords & armor with "
             "durability that wears down and can be repaired. "
             "Matching gear sets unlock combo attacks. "
             "Winning earns XP — level up for +max HP, +atk, +def. "
             "The house ranks up with you, E-rank to S-rank: higher ranks "
             "hit harder and fight smarter. "
             "Every fight spawns a fresh named hunter with their own "
             "rolled weapons, armor (better grades at higher ranks), "
             "martial-arts skills, and potions — D-rank and up cast "
             "techniques back at you. "
             "⚡ power rates both fighters (HP + attack×10 + defense×10 + "
             "skills): if the foe out-powers you, the win pays an upset "
             "bonus on top of the rank-scaled XP. "
             "First to 0 HP loses.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        def fighter():
            return {"hp": 50, "max_hp": 50, "atk": 10, "def": 5,
                    "potions": 1, "defending": False, "shield": False,
                    "focused": False, "fury_cd": 0,
                    "combo_every": 0, "combo_count": 0, "combo_name": ""}
        return {"you": fighter(), "house": fighter(),
                "done": False, "consumed": {}, "gear_wear": {},
                "combo_triggers": 0,
                "gear_base": {"atk": 10, "def": 5}, "set_bonus": None}

    def setup(self, room, mind):
        s = room.state
        self._apply_progression(room)
        self._apply_loadout(room)
        self._apply_skills(room)
        self._spawn_enemy(room, getattr(mind, "rng", None))
        # power is finalized once both fighters are fully kitted
        try:
            from ..power import power_bar
            power_line = (f"\n⚡ power {s['player_power']} "
                          f"{power_bar(s['player_power'], s.get('house_power', 1))} "
                          f"vs {s.get('house_power', '?')}")
        except Exception:  # noqa: BLE001
            power_line = ""
        gear_note = self._gear_note(room)
        skill_note = self._skill_note(room)
        prog = self._prog_bonus(room)
        lvl_note = (f"level {prog['level']} — +{prog['max_hp']} HP, "
                    f"+{prog['atk']} atk, +{prog['def']} def. "
                    if prog["level"] > 1 else "")
        rank = s.get("house_rank", "E")
        foe = s.get("house_name", "a hunter")
        foe_gear = self._house_gear_note(room)
        foe_skills = s.get("house_skills", {})
        myth_foe = bool(s.get("myth_foe"))
        foe_line = (f"a {rank}-rank hunter, {foe}, blocks your path."
                    + (" 👹 a MYTH-FOE — they rose to kill a legend."
                       if myth_foe else "")
                    + (f" {foe_gear}" if foe_gear else "")
                    + (f" they know {len(foe_skills)} "
                       f"{'skill' if len(foe_skills) == 1 else 'skills'}."
                       if foe_skills else ""))
        # the champion's entrance: wear your earned title
        who = ""
        for p in room.humans:
            title = s.get("titles", {}).get(p.key, "")
            pname = getattr(p, "name", "") or "you"
            who = f"{title} {pname}".strip() if title else pname
            break
        intro_name = f"{who} enters" if who else "you enter"
        return ("⚔️ battle arena — "
                f"{intro_name}: "
                f"{s['you']['max_hp']} HP, {s['you']['atk']} atk, "
                f"{s['you']['def']} def, 1 potion.\n"
                + (lvl_note if lvl_note else "")
                + (skill_note + "\n" if skill_note else "")
                + foe_line + "\n"
                + "attack · focus · fury · defend · potion · "
                "skill <name> · item <gear|potion|shield>\n"
                + (gear_note + "\n" if gear_note else "")
                + power_line
                + ("\n" if power_line else "")
                + "the house is already warming up.")

    def _house_gear_note(self, room) -> str:
        """One-liner describing the enemy's rolled gear."""
        gear = room.state.get("house_gear", {})
        bits = []
        weapon = gear.get("weapon") or {}
        armor = gear.get("armor") or {}
        if weapon.get("name"):
            bits.append(weapon["name"])
        if armor.get("name"):
            bits.append(armor["name"])
        note = ""
        if bits:
            note = "wielding " + " and ".join(bits) + "."
        if room.state.get("house_set_bonus"):
            note += f" ✨ {room.state['house_set_bonus']} set bonus!"
        return note

    # ── progression ──────────────────────────────────────────────────────────
    #: Solo-Leveling-style hunter ranks for the house AI. Higher player
    #: level → higher rank → the house mirrors a bigger share of the
    #: player's progression bonus, gains flat rank stats, and fights with
    #: a higher combat skill (see GameMind.combat_move).
    #: (rank, min_player_level, bonus_share, flat_hp, flat_atk, flat_def)
    HUNTER_RANKS: tuple = (
        ("E", 1, 0.40, 0, 0, 0),
        ("D", 3, 0.50, 4, 1, 0),
        ("C", 5, 0.60, 8, 1, 1),
        ("B", 8, 0.70, 12, 2, 1),
        ("A", 12, 0.80, 16, 2, 2),
        ("S", 16, 0.90, 24, 3, 3),
        ("SS", 20, 0.95, 32, 4, 4),
        ("X", 25, 1.00, 40, 5, 5),
    )

    @classmethod
    def house_rank_for(cls, level: int) -> tuple[str, int]:
        """(rank_name, rank_index) for a player level."""
        idx = 0
        for i, (_name, min_level, _sh, _hp, _atk, _df) in enumerate(
                cls.HUNTER_RANKS):
            if level >= min_level:
                idx = i
        return cls.HUNTER_RANKS[idx][0], idx

    def _prog_bonus(self, room) -> dict[str, int]:
        """The human's mirrored level bonus (engine fills state)."""
        for p in room.humans:
            prog = room.state.get("progression", {}).get(p.key)
            if prog:
                return {"level": int(prog.get("level", 1)),
                        "max_hp": int(prog.get("max_hp", 0)),
                        "atk": int(prog.get("atk", 0)),
                        "def": int(prog.get("def", 0))}
        return {"level": 1, "max_hp": 0, "atk": 0, "def": 0}

    def _apply_progression(self, room) -> None:
        """Fold the player's persistent level into base stats.

        The house ranks up Solo-Leveling style: each hunter rank mirrors a
        bigger share of the player's bonus, adds flat rank stats, and
        fights smarter.  The dynamic opponent (gear, skills, name) spawns
        separately in ``_spawn_enemy``.  Progression always feels
        powerful — but the house never falls too far behind.
        """
        s = room.state
        prog = self._prog_bonus(room)
        rank, rank_idx = self.house_rank_for(prog["level"])
        _name, _min, share, fhp, fatk, fdef = self.HUNTER_RANKS[rank_idx]
        s["gear_base"] = {"atk": 10 + prog["atk"], "def": 5 + prog["def"]}
        y, h = s["you"], s["house"]
        y["max_hp"] = 50 + prog["max_hp"]
        y["hp"] = y["max_hp"]
        h["max_hp"] = 50 + int(prog["max_hp"] * share) + fhp
        h["hp"] = h["max_hp"]
        h["atk"] = 10 + int(prog["atk"] * share) + fatk
        h["def"] = 5 + int(prog["def"] * share) + fdef
        s["player_level"] = prog["level"]
        s["house_rank"] = rank
        s["house_skill"] = rank_idx

    def _spawn_enemy(self, room, rng=None) -> None:
        """Roll the dynamic opponent: name, gear, skills, potions.

        Called after the player's loadout and skills are applied so the
        power backstop measures the real, fully-kitted player.  Higher
        ranks roll better gear grades and more skills; a power backstop
        re-rolls the gear up a grade when the enemy would otherwise be
        trivial next to the player.
        """
        import random as _random
        s = room.state
        rank_idx = int(s.get("house_skill", 0))
        h = s["house"]
        try:
            from ..enemies import roll_enemy
            from ..power import fighter_power
        except Exception:  # noqa: BLE001
            return
        rng = rng or _random.Random()
        pkey = next((p.key for p in room.humans), "")
        slugs = s.get("skills", {}).get(pkey, [])
        tiers = s.get("skill_tiers", {}).get(pkey, {})
        player_power = fighter_power(s["you"], slugs, tiers)
        s["player_power"] = player_power
        # myth check: if the player brought myth-tier gear, the house
        # answers with a myth-foe hunter — same Cutyp legacy stays
        # player-exclusive, but the fight scales up to meet it.
        loadout = self._loadout(room)
        myth_foe = any(isinstance(piece, dict)
                       and piece.get("grade") == "myth"
                       for piece in loadout.values())
        s["myth_foe"] = myth_foe
        enemy = roll_enemy(
            rng, rank_idx, player_power,
            foe_base={"max_hp": h["max_hp"], "atk": h["atk"],
                      "def": h["def"]},
            myth_foe=myth_foe)
        # house_skills must land before _apply_house_gear: the 20%
        # power cap counts skill power when it measures.
        s["house_skills"] = enemy["skills"]
        self._apply_house_gear(room, enemy)
        s["house_name"] = enemy["name"]
        s["house_skill_cd"] = {}
        s["house_skill_used"] = []
        h["potions"] = enemy["potions"]
        s["house_power"] = fighter_power(h, tuple(enemy["skills"]),
                                        enemy["skills"])

    def _apply_house_gear(self, room, enemy) -> None:
        """Fold the rolled enemy gear into the house fighter's stats.

        Enemy gear fights at 50% effectiveness — the hunter's kit is
        battle-worn and notched, not shop-fresh — except for myth-foe
        hunters, whose myth-forged kit is maintained at full power.
        After gear and any set bonus land, a power cap trims the house
        back when it would otherwise wall the player: the house may
        out-power the player by at most 20%.  The fight stays
        competitive; never a foregone conclusion either way.
        """
        s = room.state
        h = s["house"]
        gear = enemy.get("gear", {})
        myth_foe = bool(enemy.get("myth_foe"))
        weapon = gear.get("weapon") or {}
        armor = gear.get("armor") or {}
        # myth-foe hunters maintain their kit at full effectiveness
        effectiveness = 1.0 if myth_foe else 0.5
        h["atk"] += int(int(weapon.get("atk", 0)) * effectiveness)
        h["def"] += int(int(armor.get("def", 0)) * effectiveness)
        h["combo_every"], h["combo_count"], h["combo_name"] = 0, 0, ""
        s["house_set_bonus"] = None
        # set bonus: matching weapon + armor of one set
        sets: dict[str, set] = {}
        for slot, piece in (("weapon", weapon), ("armor", armor)):
            if piece.get("set"):
                sets.setdefault(piece["set"], set()).add(slot)
        try:
            from ..gear import SET_BONUSES
            from ..power import fighter_power
        except Exception:  # noqa: BLE001
            SET_BONUSES = {}
        for set_name, slots in sets.items():
            bonus = SET_BONUSES.get(set_name)
            if bonus and all(x in slots for x in bonus.needs):
                h["atk"] = int(round(h["atk"] * (1 + bonus.atk_pct)))
                h["def"] = int(round(h["def"] * (1 + bonus.def_pct)))
                h["combo_every"] = bonus.combo_every
                h["combo_name"] = bonus.combo_name
                s["house_set_bonus"] = set_name
                break
        s["house_gear"] = gear
        # ── mythic ascension ────────────────────────────────────────
        # A myth-foe hunter rose to kill a legend: after gear lands,
        # their raw stats swell until the fight is genuinely
        # competitive — targeting ~85% of the player's power.  The
        # power cap below still holds as the ceiling, so the house
        # can challenge but never wall.
        try:
            player_power = int(s.get("player_power") or 0)
        except Exception:  # noqa: BLE001
            player_power = 0
        if myth_foe and player_power > 0:
            skills = s.get("house_skills", {})
            try:
                from ..power import fighter_power as _fp
            except Exception:  # noqa: BLE001
                _fp = None
            if _fp is not None:
                target = player_power * 0.85
                for _ in range(25):
                    if _fp(h, tuple(skills), skills) >= target:
                        break
                    h["atk"] = int(h["atk"] * 1.08) + 2
                    h["def"] = int(h["def"] * 1.08) + 2
                    h["max_hp"] = int(h["max_hp"] * 1.04) + 5
                    h["hp"] = h["max_hp"]
        # ── the power cap ─────────────────────────────────────────────
        # The house may out-power the player by at most 20% — 30% when
        # it fights with forbidden techniques (they're meant to be
        # scary, but they have cooldowns and the player has their own
        # arts).  The fight stays competitive; never a foregone
        # conclusion either way.
        try:
            player_power = int(s.get("player_power") or 0)
        except Exception:  # noqa: BLE001
            player_power = 0
        if player_power > 0:
            skills = s.get("house_skills", {})
            try:
                from ..skills import is_enemy_skill
                has_forbidden = any(is_enemy_skill(sl) for sl in skills)
            except Exception:  # noqa: BLE001
                has_forbidden = False
            cap = 1.3 if has_forbidden else 1.2
            for _ in range(10):
                if fighter_power(h, tuple(skills), skills) \
                        <= player_power * cap:
                    break
                h["atk"] = max(1, int(h["atk"] * 0.9))
                h["def"] = max(0, int(h["def"] * 0.9))

    #: win-XP multiplier per hunter rank — harder opponents pay more.
    #: E×1.0 → X×2.25, so climbing is always worth it.
    RANK_XP_MULT = (1.0, 1.15, 1.3, 1.45, 1.6, 1.75, 2.0, 2.25)

    def xp_reward(self, won: bool | None, room, player) -> int:
        """Rich arena XP: wins pay, losses still move the bar, and clean
        fighting earns bonuses (capped so farming one trick stalls).

        The payout scales with the opponent: higher hunter ranks multiply
        the win, and beating a stronger foe (higher power than you) adds
        an upset bonus.  A tough win is worth far more than a routine one.
        """
        from ..progression import ARENA_LOSS_XP, ARENA_WIN_XP
        s = room.state
        if won is not True:
            return ARENA_LOSS_XP
        xp = ARENA_WIN_XP
        bonus = 0
        y = s.get("you", {})
        # flawless-ish: finished above half HP
        if y.get("hp", 0) >= y.get("max_hp", 50) // 2:
            bonus += 15
        # killing crit
        if s.get("crit_kill_by") == "you":
            bonus += 10
        # set combos actually fired
        bonus += 5 * int(s.get("combo_triggers", 0))
        total = xp + min(30, bonus)
        # a brutal finish — overkill pays outside the bonus cap
        if s.get("brutal_finish") == "you":
            total += 10
        # the opponent's strength sets the stakes
        rank_idx = max(0, min(7, int(s.get("house_skill", 0))))
        total = int(round(total * self.RANK_XP_MULT[rank_idx]))
        you_pow = int(s.get("player_power", 0))
        foe_pow = int(s.get("house_power", 0))
        if you_pow > 0 and foe_pow >= you_pow * 1.1:
            total = int(round(total * 1.25))  # the upset bonus
        return total

    # ── gear ───────────────────────────────────────────────────────────────
    def _loadout(self, room) -> dict[str, dict[str, Any]]:
        """Equipped gear snapshot for the human seat."""
        s = room.state
        for p in room.humans:
            return s.get("loadout", {}).get(p.key, {})
        return {}

    def _gear_wear(self, room, player_key: str,
                   instance_id: str) -> int:
        """Wear recorded so far this battle for one piece."""
        return int(room.state.get("gear_wear", {})
                   .get(player_key, {}).get(instance_id, 0))

    def _record_wear(self, room, player_key: str,
                     instance_id: str, amount: int = 1) -> None:
        s = room.state
        wear = s.setdefault("gear_wear", {}).setdefault(player_key, {})
        wear[instance_id] = int(wear.get(instance_id, 0)) + amount

    def _apply_loadout(self, room) -> None:
        """Fold equipped gear into the fighter's stats + set bonus."""
        s = room.state
        y = s["you"]
        base_atk, base_def = s["gear_base"]["atk"], s["gear_base"]["def"]
        y["atk"], y["def"] = base_atk, base_def
        y["combo_every"], y["combo_count"], y["combo_name"] = 0, 0, ""
        s["set_bonus"] = None
        loadout = self._loadout(room)
        for slot, piece in loadout.items():
            if slot == "weapon":
                y["atk"] += int(piece.get("atk", 0))
            elif slot == "armor":
                y["def"] += int(piece.get("def", 0))
        # set bonus: matching weapon + armor of one set
        sets: dict[str, set] = {}
        for slot, piece in loadout.items():
            if piece.get("set"):
                sets.setdefault(piece["set"], set()).add(slot)
        try:
            from ..gear import SET_BONUSES
        except Exception:  # noqa: BLE001
            SET_BONUSES = {}
        for set_name, slots in sets.items():
            bonus = SET_BONUSES.get(set_name)
            if bonus and all(x in slots for x in bonus.needs):
                y["atk"] = int(round(y["atk"] * (1 + bonus.atk_pct)))
                y["def"] = int(round(y["def"] * (1 + bonus.def_pct)))
                y["combo_every"] = bonus.combo_every
                y["combo_name"] = bonus.combo_name
                s["set_bonus"] = set_name
                break

    def _apply_skills(self, room) -> None:
        """Fold learned *passive* skills into the fighter's stats.

        Reads the engine-mirrored ``state["skills"][player_key]`` list so
        game code stays store-free.  Active skills are cast with
        ``skill <name>`` during the fight (see ``on_move``).
        """
        s = room.state
        y = s["you"]
        try:
            from ..skills import passive_bonuses
        except Exception:  # noqa: BLE001
            return
        slugs: list[str] = []
        for p in room.humans:
            slugs = s.get("skills", {}).get(p.key, [])
            break
        bonus = passive_bonuses(slugs)
        if bonus["max_hp"]:
            y["max_hp"] += bonus["max_hp"]
            y["hp"] = y["max_hp"]
        y["atk"] += bonus["atk"]
        y["def"] += bonus["def"]
        s["crit_bonus"] = bonus["crit"]
        s["skill_cd"] = {}
        s["skill_used"] = []

    def _skill_note(self, room) -> str:
        """One-liner listing learned skills at battle start."""
        try:
            from ..skills import SKILL_CATALOG, effective_def
        except Exception:  # noqa: BLE001
            return ""
        slugs: list[str] = []
        pkey = ""
        for p in room.humans:
            pkey = p.key
            slugs = room.state.get("skills", {}).get(p.key, [])
            break
        if not slugs:
            return ""
        tiers = room.state.get("skill_tiers", {}).get(pkey, {})
        names = []
        for s_ in slugs:
            if s_ not in SKILL_CATALOG:
                continue
            names.append(effective_def(s_, tiers.get(s_, 1)).name)
        return "🥋 skills: " + ", ".join(names) + " — cast with skill <name>."

    def _gear_note(self, room) -> str:
        loadout = self._loadout(room)
        if not loadout:
            return ""
        bits = []
        for slot in ("weapon", "armor", "trinket"):
            piece = loadout.get(slot)
            if piece:
                dur = ("∞ unbreakable" if piece.get("unbreakable")
                       else f"{piece['durability']} dur")
                bits.append(f"{piece['name']} ({dur})")
        note = "wearing: " + ", ".join(bits) + "."
        if room.state.get("set_bonus"):
            note += f" ✨ {room.state['set_bonus']} set bonus active!"
        return note

    def _break_check(self, room, player_key: str) -> list[str]:
        """Shatter anything worn down to 0 durability this turn."""
        out: list[str] = []
        s = room.state
        loadout = s.get("loadout", {}).get(player_key, {})
        for slot in list(loadout):
            piece = loadout[slot]
            if piece.get("unbreakable"):
                continue  # the Cutyp legacy never shatters
            left = int(piece.get("durability", 0)) - self._gear_wear(
                room, player_key, piece["id"])
            if left <= 0:
                del loadout[slot]
                out.append(f"💥 your {piece['name']} SHATTERS!")
        if out:
            self._apply_loadout(room)
            out.append("repair it with /repair — or fight on bare-handed.")
        return out

    def _hit(self, room: Room, src: str, dst: str, mind: GameMind,
             mult: float = 1.0, ignore_def: float = 0.0) -> str:
        from ..combat import strike
        s = room.state
        a, d = s[src], s[dst]
        crit_bonus = (float(s.get("crit_bonus") or 0.0)
                      if src == "you" else 0.0)
        rep = strike(a, d, mind.rng, mult=mult, ignore_def=ignore_def,
                     crit_bonus=crit_bonus)
        if rep["dodged"]:
            return (f"{src} swings — and hits only air. "
                    f"shadow step dodged it clean.")
        raw, crit, focused = rep["dmg"], rep["crit"], rep["focused"]
        # track the beating the human takes (flawless-win achievements)
        # and the truly excessive kills (brutal finishes)
        if dst == "you":
            s["dmg_taken"] = int(s.get("dmg_taken", 0)) + raw
            # track the lowest HP for comeback achievements
            cur_hp = d["hp"]
            if "lowest_hp" not in s or cur_hp < s["lowest_hp"]:
                s["lowest_hp"] = cur_hp
        hp_before = d["hp"] + raw  # strike already applied
        if d["hp"] <= 0 and raw >= 2 * max(1, hp_before):
            s["brutal_finish"] = src
            msg_brutal = " 💀 BRUTAL FINISH!"
        else:
            msg_brutal = ""
        # gear wear: the attacker's weapon and the defender's armor
        wear_notes: list[str] = []
        if src == "you" or dst == "you":
            wear_notes = self._apply_wear(room, src, dst)
        if rep["shielded"]:
            return (f"the shield SHATTERS — {dst} is burned to 1 HP. "
                    f"one more hit and it's over.")
        if crit and d["hp"] <= 0:
            # a killing crit — "you" is the human seat, "house" the AI
            s["crit_kill_by"] = src
        kind = "CRIT — " if crit else ""
        tag = " (focused)" if focused else ""
        msg = (f"{kind}{src} lands {raw}{tag} — "
               f"{dst} at {max(0, d['hp'])} HP.{msg_brutal}")
        # set combo: every Nth attack strikes twice — both seats can combo
        if d["hp"] > 0 and a.get("combo_every"):
            a["combo_count"] = int(a.get("combo_count", 0)) + 1
            if a["combo_count"] % int(a["combo_every"]) == 0:
                if src == "you":
                    s["combo_triggers"] = int(s.get("combo_triggers", 0)) + 1
                second = max(1, int((a["atk"] - d["def"] // 2) * 0.7))
                d["hp"] -= second
                msg += (f" ⚡ {a.get('combo_name', 'combo')}! "
                        f"a second strike for {second} — "
                        f"{dst} at {max(0, d['hp'])} HP.")
                if src == "you":
                    wear_extra = self._apply_wear(room, src, dst)
                    wear_notes.extend(w for w in wear_extra
                                      if w not in wear_notes)
        if wear_notes:
            msg += " " + " ".join(wear_notes)
        return msg

    def _apply_wear(self, room, src: str, dst: str) -> list[str]:
        """Wear the human's gear for one exchange. Returns break notes."""
        s = room.state
        player_key = None
        for p in room.humans:
            player_key = p.key
            break
        if player_key is None:
            return []
        loadout = s.get("loadout", {}).get(player_key, {})
        if src == "you":
            piece = loadout.get("weapon")
            if piece and not piece.get("unbreakable"):
                self._record_wear(room, player_key, piece["id"], 1)
        if dst == "you":
            piece = loadout.get("armor")
            if piece and not piece.get("unbreakable"):
                self._record_wear(room, player_key, piece["id"], 1)
        return self._break_check(room, player_key)

    def _check(self, room: Room) -> str | None:
        s = room.state
        if s["house"]["hp"] <= 0:
            s["done"] = True
            return "🏁 the house goes down — the arena is yours."
        if s["you"]["hp"] <= 0:
            s["done"] = True
            return "🏁 the house takes you. the arena keeps its score."
        return None

    def _equip(self, room: Room, player: Player, item: str) -> str | None:
        """Wear gear or spend a consumable.

        Gear (swords, armor) is *equipped* — it stays yours, wears with
        use, and can be repaired.  Only true consumables (shield, potion)
        are spent.  Reads the engine-mirrored snapshots in state so game
        code stays store-free.
        """
        s = room.state
        item = (item or "").strip().lower()
        if not item:
            return None
        # ── consumables keep the old spend path ──
        if item in ("shield", "potion"):
            inv = s.get("inventory", {}).get(player.key, {})
            if int(inv.get(item, 0)) <= 0:
                return None
            inv[item] = int(inv[item]) - 1
            s["consumed"].setdefault(player.key, {})
            s["consumed"][player.key][item] = \
                int(s["consumed"][player.key].get(item, 0)) + 1
            y = s["you"]
            if item == "shield":
                y["shield"] = True
                return "the shield is up — it will take one fatal hit."
            y["hp"] = min(y["max_hp"], y["hp"] + 30)
            s["potions_used"] = int(s.get("potions_used", 0)) + 1
            return "the potion drinks down — +30 HP."
        # ── gear: resolve against the mirrored closet ──
        closet = s.get("gear_closet", {}).get(player.key, [])
        piece = None
        for cand in closet:
            if item == cand["slug"].lower() \
                    or item in cand["slug"].lower() \
                    or item in cand["name"].lower():
                piece = cand
                break
        if piece is None:
            return None
        if int(piece.get("durability", 0)) <= 0:
            return (f"your {piece['name']} is broken — "
                    f"/repair {piece['slug']} first.")
        loadout = s.setdefault("loadout", {}).setdefault(player.key, {})
        slot = piece["slot"]
        old = loadout.get(slot)
        loadout[slot] = dict(piece)
        self._apply_loadout(room)
        # record the equip so the engine persists it on close
        s.setdefault("gear_equipped", {})[player.key] = {
            sl: p["id"] for sl, p in loadout.items()
        }
        swapped = f" (replacing {old['name']})" if old else ""
        msg = f"equipped {piece['name']}{swapped}."
        if s.get("set_bonus"):
            msg += f" ✨ {s['set_bonus']} set bonus active!"
        return msg

    def _cast_skill(self, room, player, ref: str, mind,
                    src: str = "you", dst: str = "house") -> str | None:
        """Cast a learned active skill. Returns the result message, or
        None if the ref isn't a usable skill (so the caller can fall
        through to the move list).  ``src``/``dst`` let the house cast
        its own rolled skills — including forbidden techniques — with
        the same code path."""
        from ..skills import (SKILL_CATALOG, effective_def, resolve_skill,
                              lookup_skill, is_enemy_skill)
        s = room.state
        me = s[src]
        foe = s[dst]
        foe_name = "the house" if dst == "house" else "you"
        my_name = "the house" if src == "house" else "you"
        if src == "you":
            # players can only cast what they learned — the forbidden
            # catalog is invisible to them
            defn = resolve_skill(ref)
        else:
            defn = lookup_skill(ref)
        if defn is None or defn.kind != "active":
            return None
        if src == "you":
            learned: list[str] = []
            pkey = ""
            for p in room.humans:
                pkey = p.key
                learned = s.get("skills", {}).get(p.key, [])
                break
            tiers = s.get("skill_tiers", {}).get(pkey, {})
            cd = s.setdefault("skill_cd", {})
            used = s.setdefault("skill_used", [])
            if defn.slug not in learned:
                known = [SKILL_CATALOG[x].name for x in learned
                         if x in SKILL_CATALOG]
                hint = (f"you know: {', '.join(known)}."
                        if known else "you haven't learned any skills yet — "
                        "/skill to see the school.")
                return f"you don't know {defn.name}. {hint}"
        else:
            learned = list(s.get("house_skills", {}).keys())
            tiers = s.get("house_skills", {})
            cd = s.setdefault("house_skill_cd", {})
            used = s.setdefault("house_skill_used", [])
            if defn.slug not in learned:
                return None
            pkey = "house"
            if is_enemy_skill(defn.slug):
                # the player just witnessed a forbidden technique
                s["enemy_skill_cast"] = True
        # upgrades fight: fold the caster's tier into the blueprint
        # (forbidden arts have no tiers — they fight as written)
        tier = int(tiers.get(defn.slug, 1))
        defn = effective_def(defn, tier)
        if int(cd.get(defn.slug, 0)) > 0:
            if src == "you":
                return (f"{defn.name} is recovering — "
                        f"{cd[defn.slug]} turn(s) left.")
            return None  # the house silently picks another move
        if defn.once_per_battle and defn.slug in used:
            if src == "you":
                return f"{defn.name} is spent for this battle."
            return None
        # ── effects ──
        msg = ""
        if defn.slug == "shadow_step":
            me["dodge_next"] = True
            msg = (f"{my_name} {'melt' if src == 'you' else 'melts'} into "
                   f"shadow — {foe_name}'s next attack will miss.")
            if defn.counter_mult:
                parts = [self._hit(room, src, dst, mind,
                                   mult=defn.counter_mult)]
                win = self._check(room)
                if win:
                    parts.append(win)
                msg += (" You strike from the dark! " if src == "you"
                        else " It strikes from the dark! ") + " ".join(parts)
        elif defn.slug == "war_cry":
            me["atk"] += defn.atk_buff
            me["warcry_turns"] = defn.buff_turns
            me["warcry_amt"] = defn.atk_buff
            verb = "ROAR" if src == "you" else "roars"
            msg = (f"{my_name} {verb} — +{defn.atk_buff} attack for "
                   f"{defn.buff_turns} turns!")
        elif defn.slug == "second_wind":
            heal = int(me["max_hp"] * defn.heal_pct)
            me["hp"] = min(me["max_hp"], me["hp"] + heal)
            msg = (f"{my_name} {'breathe' if src == 'you' else 'breathes'} "
                   f"deep — +{heal} HP ({me['hp']}/{me['max_hp']}).")
        else:
            # striking skills: dragon_punch / whirlwind / thousand_fists /
            # pressure_point — N hits at mult×, optionally ignoring
            # defense.  Forbidden techniques add their own horrors:
            # lifesteal, venom, debuffs, frenzy, executions.
            parts = []
            mult = float(defn.mult or 0.0)
            # frenzy: the bloodied hit harder
            if defn.frenzy and me["hp"] < me["max_hp"] * 0.5:
                mult = (mult * 1.5) if mult > 0 else 1.5
                parts.append("🩸 blood frenzy — the wounds only "
                             "feed it!")
            # execution: the mercy stroke lands heaviest on the dying
            if defn.execute_mult:
                foe_frac = foe["hp"] / max(1, foe["max_hp"])
                if foe_frac < (defn.execute_below or 0.3):
                    mult = defn.execute_mult
                    parts.append("⚰️ it smells the end — EXECUTION!")
            target_before = foe["hp"]
            if mult > 0:
                for _ in range(max(1, defn.hits)):
                    parts.append(self._hit(room, src, dst, mind,
                                           mult=mult,
                                           ignore_def=defn.ignore_def_pct))
                    win = self._check(room)
                    if win:
                        parts.append(win)
                        # a skill landed the killing blow
                        s["killing_skill"] = defn.slug
                        break
            # lifesteal: drink what you dealt
            if defn.lifesteal_pct:
                dealt = max(0, target_before - foe["hp"])
                if dealt > 0:
                    heal = int(dealt * defn.lifesteal_pct)
                    me["hp"] = min(me["max_hp"], me["hp"] + heal)
                    parts.append(f"🩸 {my_name} drinks {heal} HP "
                                 f"from the wound.")
            # venom: the wound keeps bleeding
            if defn.poison_turns and defn.poison_dmg and foe["hp"] > 0:
                foe["poison"] = {"turns": defn.poison_turns,
                                 "dmg": defn.poison_dmg}
                parts.append(f"☠️ venom seeps in — {defn.poison_dmg} "
                             f"dmg for {defn.poison_turns} turns.")
            # debuffs: take their strength while it lasts
            if defn.debuff_turns:
                if defn.atk_debuff and not foe.get("atk_debuff"):
                    amt = defn.atk_debuff
                    foe["atk"] = max(1, foe["atk"] - amt)
                    foe["atk_debuff"] = {"turns": defn.debuff_turns,
                                         "amt": amt}
                    parts.append(f"🌑 dread settles — {foe_name}'s "
                                 f"attack withers (−{amt}).")
                if defn.def_debuff and not foe.get("def_debuff"):
                    amt = defn.def_debuff
                    foe["def"] = max(0, foe["def"] - amt)
                    foe["def_debuff"] = {"turns": defn.debuff_turns,
                                         "amt": amt}
                    parts.append(f"🦴 armor cracks — {foe_name}'s "
                                 f"defense crumbles (−{amt}).")
            head = ("🥋" if not is_enemy_skill(defn.slug) else "😈")
            msg = f"{head} {my_name} unleashes {defn.name}! " + " ".join(parts)
        cd[defn.slug] = defn.cooldown
        if defn.once_per_battle:
            used.append(defn.slug)
        return msg
    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        out: list[str] = []
        # start-of-turn decay: fury/skill cooldowns, war-cry expiry,
        # poison burns, debuff expiry
        from ..combat import tick_fighter
        out.extend(tick_fighter(s["you"], s.get("skill_cd")))
        # venom can finish the job before you even move
        win = self._check(room)
        if win:
            out.append(win)
            return out
        if t == "attack":
            out.append(self._hit(room, "you", "house", mind))
            win = self._check(room)
            if win:
                out.append(win)
                return out
        elif t == "focus":
            if s["you"]["focused"]:
                return ["you're already focused — attack to spend it."]
            s["you"]["focused"] = True
            out.append("you steady your breathing — next hit +50%.")
        elif t == "fury":
            if s["you"]["fury_cd"] > 0:
                return [f"fury is warming up — {s['you']['fury_cd']} "
                        "turn(s) left."]
            out.append(self._hit(room, "you", "house", mind, mult=0.8))
            win = self._check(room)
            if not win:
                out.append(self._hit(room, "you", "house", mind,
                                     mult=0.8))
                win = self._check(room)
            s["you"]["fury_cd"] = 2
            if win:
                out.append(win)
                return out
            out.append("you're spent — fury warms up 2 turns.")
        elif t == "defend":
            s["you"]["defending"] = True
            out.append("you raise your guard.")
        elif t == "potion":
            if s["you"]["potions"] > 0 and s["you"]["hp"] < 50:
                s["you"]["potions"] -= 1
                s["you"]["hp"] = min(50, s["you"]["hp"] + 30)
                out.append(f"potion — {s['you']['hp']} HP "
                           f"({s['you']['potions']} left).")
            else:
                out.append("no potion — or you're already full.")
                return out
        elif t.startswith("item "):
            item = t[5:].strip()
            msg = self._equip(room, player, item)
            if msg is None:
                out.append(f"you don't hold {item!r} — /game shop to "
                           "buy it.")
                return out
            out.append(msg)
        elif t.startswith("skill "):
            msg = self._cast_skill(room, player, t[6:].strip(), mind)
            if msg is None:
                out.append("no such skill — your moves: attack · focus · "
                           "fury · defend · potion · skill <name> · "
                           "item <gear>.")
                return out
            out.append(msg)
            win = self._check(room)
            if win:
                out.append(win)
                return out
        else:
            out.append("attack · focus · fury · defend · potion · "
                       "skill <name> · item <gear>.")
            return out
        # the house answers
        self._house_act(room, mind, out)
        win = self._check(room)
        if win:
            out.append(win)
        else:
            extra = []
            if s["you"]["focused"]:
                extra.append("focused")
            if s["you"]["fury_cd"]:
                extra.append(f"fury in {s['you']['fury_cd']}")
            tail = f" ({', '.join(extra)})" if extra else ""
            out.append(f"your turn — {s['you']['hp']} HP "
                       f"({s['you']['potions']} potions){tail}.")
        return out

    def _house_skill_pick(self, room, mind) -> str | None:
        """Pick a skill for the house to cast this turn, or None.

        E-rank hunters are bare-knuckle brawlers; D+ cast.  The pick is
        tactical, not random: heal when bleeding, dodge when the player
        is focused (or a small feint chance), buff early, otherwise the
        hardest-hitting ready strike.  Forbidden techniques get their
        own instincts: siphon when hurt, dread early, execute the dying.
        """
        from ..skills import lookup_skill, effective_def
        s = room.state
        if int(s.get("house_skill", 0)) < 1:
            return None
        skills: dict[str, int] = s.get("house_skills", {})
        if not skills:
            return None
        cd = s.setdefault("house_skill_cd", {})
        used = s.setdefault("house_skill_used", [])
        house, you = s["house"], s["you"]
        ready = []
        for slug, tier in skills.items():
            # forbidden techniques live outside the learnable catalog
            defn = lookup_skill(slug)
            if defn is None:
                continue
            if int(cd.get(slug, 0)) > 0:
                continue
            if defn.once_per_battle and slug in used:
                continue
            ready.append((slug, effective_def(defn, int(tier))))
        if not ready:
            return None
        by_slug = {slug: d for slug, d in ready}
        # bleeding: the breath that saves lives — or the drink that
        # steals them back
        if "second_wind" in by_slug and \
                house["hp"] < house["max_hp"] * 0.45:
            return "second_wind"
        if "soul_siphon" in by_slug and \
                house["hp"] < house["max_hp"] * 0.6:
            return "soul_siphon"
        # the kill is close: no mercy
        if "executioner" in by_slug and \
                you["hp"] < you["max_hp"] * 0.35:
            return "executioner"
        # the player is winding up something big: vanish
        if "shadow_step" in by_slug and \
                (you.get("focused") or mind.rng.random() < 0.12):
            return "shadow_step"
        # roar — or dread — early while there's still fight left in it
        if house["hp"] > house["max_hp"] * 0.6 and not house.get("warcry_turns"):
            if "war_cry" in by_slug:
                return "war_cry"
            if "dread_aura" in by_slug and not you.get("atk_debuff"):
                return "dread_aura"
        # soften them up before the big swings
        if "bone_crusher" in by_slug and not you.get("def_debuff") \
                and mind.rng.random() < 0.5:
            return "bone_crusher"
        if "venom_fang" in by_slug and not you.get("poison") \
                and mind.rng.random() < 0.4:
            return "venom_fang"
        # otherwise the hardest ready strike
        strikes = [(slug, d) for slug, d in ready
                   if (d.mult or 0) > 0 or d.execute_mult > 0]
        if strikes:
            def _threat(sd):
                slug, d = sd
                base = (d.mult or 0) * (d.hits or 1)
                if slug == "executioner" and \
                        you["hp"] < you["max_hp"] * 0.5:
                    base = d.execute_mult
                if d.frenzy and house["hp"] < house["max_hp"] * 0.5:
                    base *= 1.5
                return base
            strikes.sort(key=_threat, reverse=True)
            return strikes[0][0]
        # pure debuff with no strike left to cast
        if "dread_aura" in by_slug:
            return "dread_aura"
        return None

    def _house_act(self, room: Room, mind: GameMind,
                   out: list[str]) -> None:
        """The house's turn, per the combat brain — focus, fury, guard,
        drink, swing, or (D+ hunters) cast a rolled skill."""
        from ..combat import tick_fighter
        s = room.state
        house = s["house"]
        out.extend(tick_fighter(house, s.get("house_skill_cd")))
        # venom can finish the house before it acts
        if self._check(room):
            return
        if house["fury_cd"] > 0:
            house["fury_cd"] -= 1
        # a skilled hunter opens with technique, not fists
        pick = self._house_skill_pick(room, mind)
        if pick:
            msg = self._cast_skill(room, None, pick, mind,
                                   src="house", dst="you")
            if msg:
                out.append(msg)
                return
        move = mind.combat_move(house, s["you"],
                                skill=int(s.get("house_skill", 0)))
        if move["action"] == "potion" and house["potions"] > 0:
            house["potions"] -= 1
            house["hp"] = min(house["max_hp"], house["hp"] + 30)
            out.append("the house drinks — it'll be back at 50.")
            return
        if move["action"] == "defend":
            house["defending"] = True
            out.append("the house guards.")
            return
        if move["action"] == "focus" and not house["focused"]:
            house["focused"] = True
            out.append("the house settles in — its next hit will land "
                       "harder.")
            return
        if move["action"] == "fury" and house["fury_cd"] <= 0:
            out.append(self._hit(room, "house", "you", mind, mult=0.8))
            if not self._check(room):
                out.append(self._hit(room, "house", "you", mind,
                                     mult=0.8))
            house["fury_cd"] = 2
            return
        out.append(self._hit(room, "house", "you", mind))

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        if room.state["house"]["hp"] <= 0:
            return room.humans[0] if room.humans else None
        if room.state["you"]["hp"] <= 0:
            return Player(key="ai:arena", platform="ai",
                          name="The House", is_ai=True)
        return None

    def describe_state(self, room):
        s = room.state
        base = (f"you {s['you']['hp']} HP · "
                f"{s.get('house_name', 'the house')} "
                f"{s['house']['hp']} HP")
        if s.get("player_power") and s.get("house_power"):
            base += (f" · ⚡{s['player_power']} vs "
                     f"{s['house_power']}")
        return base


# ── escape room ──────────────────────────────────────────────────────────────

PUZZLES: tuple[dict[str, Any], ...] = (
    {"kind": "number", "prompt": "Lock 1 — a dial: 'twice nine, plus the "
     "number of letters in the word door' — what number?", "answer": "26",
     "hint": "18 + 4"},
    {"kind": "word", "prompt": "Lock 2 — a word wheel: 'I speak without "
     "a mouth and hear without ears. What am I?'", "answer": "echo",
     "hint": "a sound returning"},
    {"kind": "number", "prompt": "Lock 3 — a code: 'the square of 7 minus "
     "the square of 5' — what number?", "answer": "24", "hint": "49 − 25"},
    {"kind": "word", "prompt": "Lock 4 — a cipher plate: 'the opposite of "
     "north, in one word' — what word?", "answer": "south", "hint": "go down the map"},
    {"kind": "number", "prompt": "Lock 5 — a sequence plate: 2, 6, 18, "
     "54, … — what number comes next?", "answer": "162", "hint": "×3 each step"},
    {"kind": "word", "prompt": "Lock 6 — a mirror plate: 'what small boat "
     "reads the same forwards and backwards, in one word?'",
     "answer": "kayak", "hint": "you sit in it, it bobs"},
)


class EscapeRoomGame(MultiGame):
    name = "escape"
    description = "6 locks, 3 strikes each — the table breaks out together"
    min_players = 1
    max_players = 6
    ai_seats = 1
    move_timeout = 180
    rules = ("Six locks, in order. On your turn: 'answer <your guess>' "
             "(or 'hint' for a nudge, or 'use keycard' if you own one). "
             "Three wrong attempts across the table on one lock and it "
             "seals forever — the room wins. Break all six and the door "
             "is yours. The house tries too — it's good, but it misses.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"puzzles": list(PUZZLES), "lock": 0, "fails": 0,
                "max_fails": 3, "hints_used": 0, "done": False,
                "consumed": {}}

    def setup(self, room, mind):
        p = room.state["puzzles"][0]
        n = len(room.state["puzzles"])
        return (f"escape room — the door is {n} locks deep.\n"
                f"{p['prompt']}\n"
                f"'answer <guess>', 'hint', or 'use keycard'. "
                f"the house is looking too.")

    def _cur(self, room: Room) -> dict[str, Any]:
        return room.state["puzzles"][room.state["lock"]]

    def _advance(self, room: Room) -> list[str]:
        s = room.state
        s["lock"] += 1
        s["fails"] = 0
        if s["lock"] >= len(s["puzzles"]):
            s["done"] = True
            return ["🏁 the last lock turns. the door opens — the table "
                    "walks out together."]
        p = s["puzzles"][s["lock"]]
        return [f"lock {s['lock']+1}/{len(s['puzzles'])} — {p['prompt']}"]

    def _attempt(self, room: Room, who: str, text: str) -> list[str]:
        s = room.state
        p = self._cur(room)
        t = re.sub(r"answer\s*", "", text.strip().lower())
        if not t:
            return ["'answer <your guess>' — or 'hint'."]
        a = p["answer"].lower()
        if a in t or t in a or (a.isdigit() and a in t):
            out = [f"{who} cracks it — {p['answer']}."]
            out.extend(self._advance(room))
            return out
        s["fails"] += 1
        left = s["max_fails"] - s["fails"]
        if left <= 0:
            s["done"] = True
            return [f"{who} is wrong — and the lock seals itself. "
                    f"the room keeps everyone. 🏁"]
        out = [f"{who} is wrong. {left} strikes left on this lock."]
        if s["fails"] == 2:
            out.append(f"the room offers a hint: {p['hint']}")
        return out

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if t.startswith("hint"):
            s["hints_used"] += 1
            return [f"hint: {self._cur(room)['hint']}"]
        if t.startswith("use keycard"):
            inv = s.get("inventory", {}).get(player.key, {})
            if int(inv.get("keycard", 0)) > 0:
                inv["keycard"] = int(inv["keycard"]) - 1
                s["consumed"].setdefault(player.key, {})
                s["consumed"][player.key]["keycard"] = \
                    int(s["consumed"][player.key].get("keycard", 0)) + 1
                out = ["the keycard hums — the lock takes it and opens."]
                out.extend(self._advance(room))
                return out
            return ["you don't own a keycard — /game shop (it's 500c)."]
        if t.startswith("answer"):
            return self._attempt(room, player.name, t)
        return self._attempt(room, player.name, "answer " + t)

    def ai_turn(self, room, mind):
        s = room.state
        # the house solves known locks with 40% — a real teammate
        if mind.rng.random() < 0.4:
            p = self._cur(room)
            out = [f"{room.current.name} cracks it — {p['answer']}."]
            out.extend(self._advance(room))
            return out
        if mind.rng.random() < 0.5:
            out = [f"{room.current.name} tries a guess — the lock "
                   "shakes its head."]
            s["fails"] += 1
            left = s["max_fails"] - s["fails"]
            if left <= 0:
                s["done"] = True
                out.append("the lock seals. the room wins. 🏁")
            else:
                out.append(f"{left} strikes left.")
            return out
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        if room.state["lock"] >= len(room.state["puzzles"]):
            return "draw"  # the table escaped together — everyone scores
        return Player(key="ai:escape", platform="ai", name="The Room",
                      is_ai=True)

    def score(self, room, player):
        return room.state["lock"] * 10 - room.state["hints_used"] * 2

    def describe_state(self, room):
        s = room.state
        return (f"lock {min(s['lock'] + 1, len(s['puzzles']))}/"
                f"{len(s['puzzles'])} · strikes: {s['fails']}/"
                f"{s['max_fails']}")


# ── political simulator ──────────────────────────────────────────────────────

PLEDGES: tuple[str, ...] = (
    "I will plant a tree on every street corner.",
    "I will turn the town clock back to 3:33 — the good hours.",
    "I will make the market pay for its own stalls.",
    "I will build a bridge, and name it after nobody.",
    "I will give the library a second door.",
    "I will tax the fog 10% on market days.",
    "I will appoint a committee for the fountain.",
    "I will make the town band play exactly one song a year.",
)


class PoliticalGame(MultiGame):
    name = "political"
    description = "3 elections — pledge, campaign, vote, be mayor"
    min_players = 2
    max_players = 6
    ai_seats = 2
    needs_group = True
    move_timeout = 120
    rules = ("Everyone here is a candidate. Three elections, each with "
             "two phases: campaign (your turn: a pledge — your words or "
             "'pledge N' from the board) and vote (your turn: a "
             "candidate's name — you can vote yourself). Most votes wins "
             "the election; ties void it. After 3, the most election "
             "wins makes mayor. One 'veto <name>' per game cancels one "
             "vote (if you own a veto token).")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"election": 1, "elections": 3, "phase": "campaign",
                "pledged": {}, "votes": {}, "wins": {}, "done": False,
                "veto_used": {}}

    def setup(self, room, mind):
        s = room.state
        for p in room.players:
            s["wins"][p.key] = 0
        names = ", ".join(p.name for p in room.players)
        return (f"political — candidates: {names}.\n"
                f"election 1/3, campaign phase — "
                f"{room.players[0].name}, your pledge "
                f"(your words, or 'pledge 1–8').")

    def _candidates(self, room: Room) -> list[Player]:
        return list(room.players)

    def _election_over(self, room: Room) -> bool:
        s = room.state
        return all(p.key in s["pledged"] for p in self._candidates(room))

    def _vote_over(self, room: Room) -> bool:
        s = room.state
        return all(p.key in s["votes"]
                   for p in self._candidates(room))

    def _tally(self, room: Room) -> list[str]:
        s = room.state
        counts: dict[str, int] = {}
        for target in s["votes"].values():
            if target:
                counts[target] = counts.get(target, 0) + 1
        lines = [f"the ballot counts: " +
                 ", ".join(f"{room.player(k).name} {v}"
                           for k, v in counts.items()) or "empty"]
        if counts:
            top = max(counts.values())
            leaders = [k for k, v in counts.items() if v == top]
            if len(leaders) == 1:
                w = room.player(leaders[0])
                s["wins"][leaders[0]] = int(s["wins"].get(leaders[0], 0)) + 1
                lines.append(f"{w.name if w else '?'} wins election "
                             f"{s['election']}.")
            else:
                lines.append("a tie — the election is void.")
        else:
            lines.append("nobody voted — a void election.")
        s["election"] += 1
        if s["election"] > s["elections"]:
            s["done"] = True
            best = max(s["wins"].items(), key=lambda kv: kv[1],
                       default=("", 0))
            w = room.player(best[0]) if best[0] else None
            lines.append("🏁 the final count — " +
                         (f"**{w.name} is mayor** with {best[1]} "
                          f"elections." if (w and best[1] > 0)
                          else "no clear mayor — the town keeps "
                          "campaigning."))
        else:
            s["phase"] = "campaign"
            s["pledged"], s["votes"] = {}, {}
            first = room.players[0]
            lines.append(f"election {s['election']}/{s['elections']} — "
                         f"campaign phase. {first.name}, your pledge?")
        return lines

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip()
        tl = t.lower()
        if s["phase"] == "campaign":
            if tl.startswith("pledge ") and tl[7:].strip().isdigit():
                n = int(tl[7:].strip())
                if 1 <= n <= len(PLEDGES):
                    s["pledged"][player.key] = PLEDGES[n - 1]
                    out = [f"{player.name} pledges: “{PLEDGES[n-1]}”"]
                else:
                    return ["pledge 1 through 8 — or your own words."]
            elif len(tl) >= 3:
                s["pledged"][player.key] = t[:200]
                out = [f"{player.name} pledges: “{t[:200]}”"]
            else:
                return ["a real pledge — your words, or 'pledge N'."]
        else:  # voting
            if tl.startswith("veto ") and player.key not in \
                    s["veto_used"]:
                target = tl[5:].strip().lower()
                for v, cand in list(s["votes"].items()):
                    c = room.player(cand)
                    if c and target in c.name.lower():
                        s["votes"][v] = None
                        s["veto_used"][player.key] = True
                        out = [f"{player.name} vetoes {c.name}'s vote. "
                               "the ballot shrinks."]
                        if self._vote_over(room):
                            out.extend(self._tally(room))
                        return out
                return ["veto a candidate who already voted — "
                        f"'veto <name>'."]
            for p in self._candidates(room):
                if tl and (p.name.lower() in tl or tl in p.name.lower()):
                    s["votes"][player.key] = p.key
                    out = [f"{player.name} votes {p.name}."]
                    if self._vote_over(room):
                        out.extend(self._tally(room))
                    return out
            names = ", ".join(p.name for p in self._candidates(room))
            return [f"vote a candidate: {names} (or 'veto <name>')."]
        if self._election_over(room) and s["phase"] == "campaign":
            s["phase"] = "vote"
            s["votes"] = {}
            out.append("the pledges are on the board. voting — name a "
                       "candidate (you can vote yourself).")
        return out

    def ai_turn(self, room, mind):
        s = room.state
        me = room.current
        if s["phase"] == "campaign" and me.key not in s["pledged"]:
            pledge = mind.choice(list(PLEDGES),
                                 "political campaign pledge")
            s["pledged"][me.key] = pledge
            out = [f"{me.name} pledges: “{pledge}”"]
            if self._election_over(room):
                s["phase"] = "vote"
                s["votes"] = {}
                out.append("the pledges are on the board. voting — "
                           "name a candidate.")
            return out
        if s["phase"] == "vote" and me.key not in s["votes"]:
            cands = self._candidates(room)
            # the house votes strategically: favors whoever pledged
            # something the table liked most (pledge length as a
            # deterministic stand-in for "conviction")
            weights = {p.key: len(s["pledged"].get(p.key, "")) +
                       mind.rng.random() * 6 for p in cands}
            pick = max(weights.items(), key=lambda kv: kv[1])
            s["votes"][me.key] = pick[0]
            out = [f"{me.name} votes {room.player(pick[0]).name}."]
            if self._vote_over(room):
                out.extend(self._tally(room))
            return out
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        best = max(room.state.get("wins", {}).items(),
                   key=lambda kv: kv[1], default=("", 0))
        w = room.player(best[0]) if best[0] else None
        if w is None or best[1] == 0:
            return "draw"
        if w.is_ai:
            return w
        return w

    def score(self, room, player):
        return int(room.state.get("wins", {}).get(player.key, 0)) * 10

    def describe_state(self, room):
        s = room.state
        return (f"election {min(s['election'], s['elections'])}/"
                f"{s['elections']} · phase: {s['phase']} · standings: "
                + " · ".join(f"{room.player(k).name if room.player(k) else k} "
                             f"{v}" for k, v in s["wins"].items()))


AMBITIOUS_GAMES: tuple[MultiGame, ...] = (
    WorldGame(), BattleArenaGame(), EscapeRoomGame(), PoliticalGame(),
)
