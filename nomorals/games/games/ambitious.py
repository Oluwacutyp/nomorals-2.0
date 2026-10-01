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
TOWN_EVENTS: tuple[tuple[str, int, str], ...] = (
    # (event, weight, flavor)
    ("rain", 3, "rain soaks the fields — +5 food"),
    ("caravan", 2, "a caravan crosses the gate — +6 gold"),
    ("festival", 2, "the festival brings a stranger — +1 person (−2 food)"),
    ("drought", 2, "the well sputters — −4 food"),
    ("merchant", 1, "a merchant lingers — trade pays +2 for 3 days"),
    ("plague", 1, "a cough moves through the houses"),
    ("raid", 1, "raiders hit the night market"),
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
                 f"food/day, eat 1/person/day · farm pays "
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
        if name == "rain":
            s["food"] += 5
        elif name == "caravan":
            s["gold"] += 6
        elif name == "festival":
            if s["pop"] < self._max_pop(room) and s["food"] >= 2:
                s["pop"] += 1
                s["food"] -= 2
            else:
                s["gold"] += 2  # the stranger leaves a coin and goes
                flavor = "the festival passes — +2 gold"
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
            s["gold"] += 3
            out.append("the mine gives +3 gold.")
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
    description = "turn-to-death combat — shop items are real gear"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 60
    rules = ("You vs the house: 50 HP each. attack (deal atk − their "
             "defense, 10% crits double it), focus (next hit +50%, "
             "costs your turn), fury (two 80% attacks, 2-turn cooldown), "
             "defend (halve the next hit), potion (+30 HP), or equip "
             "shop gear — sword +10 atk, armor +15 def, shield "
             "absorbs one death, potion item +30. Gear is consumed "
             "from your real inventory. First to 0 HP loses — the "
             "shield buys exactly one second life.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        def fighter():
            return {"hp": 50, "max_hp": 50, "atk": 10, "def": 5,
                    "potions": 1, "defending": False, "shield": False,
                    "focused": False, "fury_cd": 0}
        return {"you": fighter(), "house": fighter(),
                "done": False, "consumed": {}}

    def setup(self, room, mind):
        prof = None
        for p in room.humans:
            prof = p
            break
        return ("⚔️ battle arena — 50 HP, 10 atk, 5 def, 1 potion.\n"
                "attack · defend · potion · item <sword|armor|shield|potion>\n"
                "the house is already warming up.")

    def _hit(self, room: Room, src: str, dst: str, mind: GameMind,
             mult: float = 1.0) -> str:
        s = room.state
        a, d = s[src], s[dst]
        # a focused fighter spends its focus on this hit
        focused = a.get("focused", False)
        a["focused"] = False
        raw = max(1, a["atk"] - d["def"] // 2 + mind.rng.randint(-2, 3))
        if mult != 1.0:
            raw = max(1, int(raw * mult))
        if focused:
            raw = max(1, int(raw * 1.5))
        crit = mind.rng.random() < 0.10
        if crit:
            raw *= 2
        if d["defending"]:
            raw = max(1, raw // 2)
            d["defending"] = False
        d["hp"] -= raw
        if d["hp"] <= 0 and d["shield"]:
            d["shield"] = False
            d["hp"] = 1
            return (f"the shield SHATTERS — {dst} is burned to 1 HP. "
                    f"one more hit and it's over.")
        if crit and d["hp"] <= 0:
            # a killing crit — "you" is the human seat, "house" the AI
            s["crit_kill_by"] = src
        kind = "CRIT — " if crit else ""
        tag = " (focused)" if focused else ""
        return (f"{kind}{src} lands {raw}{tag} — "
                f"{dst} at {max(0, d['hp'])} HP.")

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
        """Check inventory via the state bridge: the engine mirrors
        owned items into state['inventory'][player.key] at setup time."""
        s = room.state
        inv = s.get("inventory", {}).get(player.key, {})
        if int(inv.get(item, 0)) <= 0:
            return None
        inv[item] = int(inv[item]) - 1
        s["consumed"].setdefault(player.key, {})
        s["consumed"][player.key][item] = \
            int(s["consumed"][player.key].get(item, 0)) + 1
        y = s["you"]
        if item == "sword":
            y["atk"] += 10
            return "the steel sword goes on — +10 atk."
        if item == "armor":
            y["def"] += 15
            return "iron armor, +15 def."
        if item == "shield":
            y["shield"] = True
            return "the shield is up — it will take one fatal hit."
        if item == "potion":
            y["hp"] = min(50, y["hp"] + 30)
            return "the potion drinks down — +30 HP."
        return None

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        out: list[str] = []
        # the player's fury cooldown ticks down once per turn
        if s["you"]["fury_cd"] > 0:
            s["you"]["fury_cd"] -= 1
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
        else:
            out.append("attack · focus · fury · defend · potion · "
                       "item <gear>.")
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

    def _house_act(self, room: Room, mind: GameMind,
                   out: list[str]) -> None:
        """The house's turn, per the combat brain — focus, fury, guard,
        drink, or swing."""
        s = room.state
        house = s["house"]
        if house["fury_cd"] > 0:
            house["fury_cd"] -= 1
        move = mind.combat_move(house, s["you"])
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
        return (f"you {s['you']['hp']} HP · house {s['house']['hp']} HP")


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
