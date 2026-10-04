"""PvP duel + raid boss — the arena's multiplayer organs.

* **duel** — 1v1 turn-based combat, human vs human. Your level, gear,
  and learned skills all fight with you. Start a lobby in a group
  (``/pvp``) or challenge someone DM-to-DM (``/pvp @user``,
  ``/arena challenge @user`` — the relay connects both chats).
  120 seconds per move: stall twice and you auto-guard, stall a third
  time and you forfeit.
* **raid** — 1–6 players team up against a raid boss with massive HP,
  cleave attacks, and an enrage phase. The boss acts after every full
  round of player moves; loot splits by damage dealt.

Both games run on the standard engine path, so they work on every
platform the gateway serves (Telegram bot, WhatsApp, …) — moves,
timeouts, and finishes all flow through the same send/relay plumbing.
"""
from __future__ import annotations

import random
from typing import Any

from ..ai import GameMind
from ..combat import new_fighter, strike, tick_fighter
from ..players import Player
from .base import MultiGame, Room

__all__ = ["DuelGame", "RaidGame", "PVP_GAMES"]

MOVE_HELP = ("attack · focus · fury · defend · potion · "
             "skill <name> · item <gear|potion|shield>")

#: raid boss names — one is drawn per raid
BOSS_NAMES: tuple[str, ...] = (
    "Gorefang the Render",
    "Mama Brute",
    "The Hollow King",
    "Rustmaw",
    "Vex the Unbroken",
)


class _ArenaCombat:
    """Shared combat plumbing for duel + raid.

    State layout (under ``room.state``)::

        fighters:   {seat_key: fighter}      # seat = player.key, or "boss"
        base:       {seat_key: {atk, def}}   # pre-gear stats (for recalc)
        skill_cd:   {seat_key: {slug: n}}
        skill_used: {seat_key: [slug]}
        crit_bonus: {seat_key: float}
        dmg:        {seat_key: int}           # damage DEALT by this seat
        misses:     {player_key: int}         # consecutive timeouts
        set_bonus:  {seat_key: set_name}
        gear_wear / consumed / gear_equipped: engine shapes (per player key)
    """

    # ── fighter construction ──────────────────────────────────────────
    def _build_fighter(self, room: Room, player: Player) -> dict[str, Any]:
        """Build one human's fighter from the engine mirrors
        (progression / loadout / skills). Idempotent — safe to call on
        join and on accept."""
        from ..skills import passive_bonuses
        s = room.state
        key = player.key
        prog = s.get("progression", {}).get(key, {})
        slugs = s.get("skills", {}).get(key, [])
        pb = passive_bonuses(slugs)
        f = new_fighter(
            hp=50 + int(prog.get("max_hp", 0)) + int(pb["max_hp"]),
            atk=10 + int(prog.get("atk", 0)) + int(pb["atk"]),
            dfn=5 + int(prog.get("def", 0)) + int(pb["def"]),
        )
        s.setdefault("base", {})[key] = {
            "atk": f["atk"], "def": f["def"]}
        s.setdefault("fighters", {})[key] = f
        s.setdefault("skill_cd", {})[key] = {}
        s.setdefault("skill_used", {})[key] = []
        s.setdefault("crit_bonus", {})[key] = float(pb["crit"])
        s.setdefault("dmg", {}).setdefault(key, 0)
        s.setdefault("misses", {}).setdefault(key, 0)
        self._recache_fighter(room, key, key)
        return f

    def _recache_fighter(self, room: Room, player_key: str,
                         seat: str) -> None:
        """Re-fold gear + set bonus into a fighter (equip/break)."""
        from ..gear import SET_BONUSES
        s = room.state
        f = s["fighters"][seat]
        base = s.get("base", {}).get(seat, {"atk": 10, "def": 5})
        f["atk"], f["def"] = int(base["atk"]), int(base["def"])
        f["combo_every"], f["combo_count"], f["combo_name"] = 0, 0, ""
        loadout = s.get("loadout", {}).get(player_key, {})
        for slot, piece in loadout.items():
            if slot == "weapon":
                f["atk"] += int(piece.get("atk", 0))
            elif slot == "armor":
                f["def"] += int(piece.get("def", 0))
        sets: dict[str, set] = {}
        for slot, piece in loadout.items():
            if piece.get("set"):
                sets.setdefault(piece["set"], set()).add(slot)
        s.setdefault("set_bonus", {}).pop(seat, None)
        for sname, slots in sets.items():
            bonus = SET_BONUSES.get(sname)
            if bonus and all(x in slots for x in bonus.needs):
                f["atk"] = int(round(f["atk"] * (1 + bonus.atk_pct)))
                f["def"] = int(round(f["def"] * (1 + bonus.def_pct)))
                f["combo_every"] = bonus.combo_every
                f["combo_name"] = bonus.combo_name
                s["set_bonus"][seat] = sname
                break

    # ── strikes ───────────────────────────────────────────────────────
    def _strike_msg(self, room: Room, mind: GameMind,
                    atk_seat: str, dfn_seat: str,
                    atk_name: str, dfn_name: str,
                    atk_key: str | None = None,
                    dfn_key: str | None = None,
                    mult: float = 1.0,
                    ignore_def: float = 0.0) -> tuple[str, dict]:
        """One strike between seats; handles wear + combo + messages.
        ``atk_key``/``dfn_key`` are player keys (None for the boss —
        the boss wears no gear). Returns (message, strike_report)."""
        s = room.state
        a, d = s["fighters"][atk_seat], s["fighters"][dfn_seat]
        rep = strike(
            a, d, mind.rng, mult=mult, ignore_def=ignore_def,
            crit_bonus=float(s.get("crit_bonus", {}).get(atk_seat, 0.0)))
        if rep["dodged"]:
            return (f"{atk_name} swings — and hits only air. "
                    f"{dfn_name} isn't there.", rep)
        raw, crit = rep["dmg"], rep["crit"]
        s.setdefault("dmg", {})[atk_seat] = \
            int(s.get("dmg", {}).get(atk_seat, 0)) + raw
        wear_notes = self._wear(room, atk_key, dfn_key)
        if rep["shielded"]:
            msg = (f"🛡️ {dfn_name}'s shield SHATTERS — burned to 1 HP!")
            if wear_notes:
                msg += " " + " ".join(wear_notes)
            return msg, rep
        kind = "CRIT — " if crit else ""
        tag = " (focused)" if rep["focused"] else ""
        msg = (f"{kind}{atk_name} lands {raw}{tag} — "
               f"{dfn_name} at {max(0, d['hp'])} HP.")
        # set combo: every Nth attack strikes twice
        if d["hp"] > 0 and a.get("combo_every"):
            a["combo_count"] = int(a.get("combo_count", 0)) + 1
            if a["combo_count"] % int(a["combo_every"]) == 0:
                second = max(1, int((a["atk"] - d["def"] // 2) * 0.7))
                d["hp"] -= second
                s["dmg"][atk_seat] = int(s["dmg"].get(atk_seat, 0)) + second
                msg += (f" ⚡ {a.get('combo_name', 'combo')}! "
                        f"a second strike for {second} — "
                        f"{dfn_name} at {max(0, d['hp'])} HP.")
        if wear_notes:
            msg += " " + " ".join(wear_notes)
        return msg, rep

    def _wear(self, room: Room, atk_key: str | None,
              dfn_key: str | None) -> list[str]:
        """Record gear wear for one exchange; break at 0."""
        s = room.state
        for key, slot in ((atk_key, "weapon"), (dfn_key, "armor")):
            if not key:
                continue
            piece = s.get("loadout", {}).get(key, {}).get(slot)
            if piece and not piece.get("unbreakable"):
                wear = s.setdefault("gear_wear", {}).setdefault(key, {})
                wear[piece["id"]] = int(wear.get(piece["id"], 0)) + 1
        notes: list[str] = []
        for key in (k for k in (atk_key, dfn_key) if k):
            notes.extend(self._break_check(room, key))
        return notes

    def _break_check(self, room: Room, player_key: str) -> list[str]:
        out: list[str] = []
        s = room.state
        loadout = s.get("loadout", {}).get(player_key, {})
        for slot in list(loadout):
            piece = loadout[slot]
            if piece.get("unbreakable"):
                continue  # the Cutyp legacy never shatters
            left = int(piece.get("durability", 0)) - int(
                s.get("gear_wear", {}).get(player_key, {})
                .get(piece["id"], 0))
            if left <= 0:
                del loadout[slot]
                out.append(f"💥 {piece['name']} SHATTERS!")
        if out:
            if player_key in s.get("fighters", {}):
                self._recache_fighter(room, player_key, player_key)
            out.append("repair it with /repair — or fight on bare-handed.")
        return out

    # ── skills ────────────────────────────────────────────────────────
    def _cast_skill(self, room: Room, player: Player, seat: str,
                    foe_seat: str, ref: str, mind: GameMind,
                    me_name: str, foe_name: str,
                    foe_key: str | None = None) -> str | None:
        """Cast a learned active skill. None = not a usable skill ref."""
        from ..skills import SKILL_CATALOG, effective_def, resolve_skill
        s = room.state
        me = s["fighters"][seat]
        defn = resolve_skill(ref)
        if defn is None or defn.kind != "active":
            return None
        learned = s.get("skills", {}).get(player.key, [])
        if defn.slug not in learned:
            known = [SKILL_CATALOG[x].name for x in learned
                     if x in SKILL_CATALOG]
            hint = (f"you know: {', '.join(known)}."
                    if known else "you haven't learned any skills yet — "
                    "/skill to see the school.")
            return f"you don't know {defn.name}. {hint}"
        # upgrades fight: fold the player's tier into the blueprint
        tier = s.get("skill_tiers", {}).get(player.key, {}).get(defn.slug, 1)
        defn = effective_def(defn, tier)
        cd = s.setdefault("skill_cd", {}).setdefault(seat, {})
        if int(cd.get(defn.slug, 0)) > 0:
            return (f"{defn.name} is recovering — "
                    f"{cd[defn.slug]} turn(s) left.")
        used = s.setdefault("skill_used", {}).setdefault(seat, [])
        if defn.once_per_battle and defn.slug in used:
            return f"{defn.name} is spent for this battle."
        msg = ""
        if defn.slug == "shadow_step":
            me["dodge_next"] = True
            msg = f"{me_name} melts into shadow — the next attack misses."
            if defn.counter_mult:
                pm, rep = self._strike_msg(
                    room, mind, seat, foe_seat, me_name, foe_name,
                    atk_key=player.key, dfn_key=foe_key,
                    mult=defn.counter_mult)
                msg += f" A strike from the dark! {pm}"
        elif defn.slug == "war_cry":
            me["atk"] += defn.atk_buff
            me["warcry_turns"] = defn.buff_turns
            me["warcry_amt"] = defn.atk_buff
            msg = (f"{me_name} ROARS — +{defn.atk_buff} attack for "
                   f"{defn.buff_turns} turns!")
        elif defn.slug == "second_wind":
            heal = int(me["max_hp"] * defn.heal_pct)
            me["hp"] = min(me["max_hp"], me["hp"] + heal)
            msg = (f"{me_name} breathes deep — +{heal} HP "
                   f"({me['hp']}/{me['max_hp']}).")
        else:
            parts = []
            for _ in range(max(1, defn.hits)):
                pm, rep = self._strike_msg(
                    room, mind, seat, foe_seat, me_name, foe_name,
                    atk_key=player.key, dfn_key=foe_key,
                    mult=defn.mult or 1.0,
                    ignore_def=defn.ignore_def_pct)
                parts.append(pm)
                if rep.get("crit") and \
                        s["fighters"][foe_seat]["hp"] <= 0:
                    s["crit_kill_by"] = seat
                if s["fighters"][foe_seat]["hp"] <= 0:
                    break
            msg = f"🥋 {defn.name}! " + " ".join(parts)
        cd[defn.slug] = defn.cooldown
        if defn.once_per_battle:
            used.append(defn.slug)
        return msg

    # ── items ─────────────────────────────────────────────────────────
    def _use_item(self, room: Room, player: Player, seat: str,
                  item: str, me_name: str) -> str | None:
        """Consumables + mid-battle gear equips. None = not usable."""
        s = room.state
        me = s["fighters"][seat]
        item = (item or "").strip().lower()
        if not item:
            return None
        if item in ("shield", "potion"):
            inv = s.get("inventory", {}).get(player.key, {})
            if int(inv.get(item, 0)) <= 0:
                return None
            inv[item] = int(inv[item]) - 1
            s.setdefault("consumed", {}).setdefault(player.key, {})
            s["consumed"][player.key][item] = \
                int(s["consumed"][player.key].get(item, 0)) + 1
            if item == "shield":
                me["shield"] = True
                return (f"{me_name} raises a shield — "
                        f"it will take one fatal hit.")
            me["hp"] = min(me["max_hp"], me["hp"] + 30)
            return f"{me_name} drinks a potion — +30 HP."
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
            return (f"{piece['name']} is broken — "
                    f"/repair {piece['slug']} first.")
        loadout = s.setdefault("loadout", {}).setdefault(player.key, {})
        slot = piece["slot"]
        old = loadout.get(slot)
        loadout[slot] = dict(piece)
        self._recache_fighter(room, player.key, seat)
        s.setdefault("gear_equipped", {})[player.key] = {
            sl: p["id"] for sl, p in loadout.items()
        }
        swapped = f" (replacing {old['name']})" if old else ""
        msg = f"{me_name} equips {piece['name']}{swapped}."
        if s.get("set_bonus", {}).get(seat):
            msg += f" ✨ {s['set_bonus'][seat]} set bonus active!"
        return msg

    # ── one full combat move ──────────────────────────────────────────
    def _do_move(self, room: Room, player: Player, seat: str,
                 text: str, mind: GameMind, foe_seat: str,
                 foe_name: str, me_name: str,
                 foe_key: str | None = None) -> list[str]:
        """Parse and apply one combat move against a single foe seat."""
        s = room.state
        me = s["fighters"][seat]
        t = text.strip().lower()
        out: list[str] = []
        out.extend(tick_fighter(me, s.get("skill_cd", {}).get(seat)))
        if t == "attack":
            msg, rep = self._strike_msg(
                room, mind, seat, foe_seat, me_name, foe_name,
                atk_key=player.key, dfn_key=foe_key)
            out.append(msg)
            if rep.get("crit") and s["fighters"][foe_seat]["hp"] <= 0:
                s["crit_kill_by"] = seat
        elif t == "focus":
            if me["focused"]:
                return ["already focused — attack to spend it."]
            me["focused"] = True
            out.append(f"{me_name} steadies — next hit +50%.")
        elif t == "fury":
            if me["fury_cd"] > 0:
                return [f"fury is warming up — {me['fury_cd']} turn(s) left."]
            for _ in range(2):
                msg, _rep = self._strike_msg(
                    room, mind, seat, foe_seat, me_name, foe_name,
                    atk_key=player.key, dfn_key=foe_key, mult=0.8)
                out.append(msg)
                if s["fighters"][foe_seat]["hp"] <= 0:
                    break
            me["fury_cd"] = 2
            out.append(f"{me_name} is spent — fury warms up 2 turns.")
        elif t == "defend":
            me["defending"] = True
            out.append(f"{me_name} raises their guard.")
        elif t == "potion":
            if me["potions"] > 0 and me["hp"] < me["max_hp"]:
                me["potions"] -= 1
                me["hp"] = min(me["max_hp"], me["hp"] + 30)
                out.append(f"potion — {me['hp']} HP ({me['potions']} left).")
            else:
                out.append("no potion — or already full.")
        elif t.startswith("item "):
            msg = self._use_item(room, player, seat, t[5:].strip(),
                                 me_name)
            if msg is None:
                return [f"you don't hold {t[5:].strip()!r} — "
                        "/game shop to buy it."]
            out.append(msg)
        elif t.startswith("skill "):
            msg = self._cast_skill(room, player, seat, foe_seat,
                                   t[6:].strip(), mind, me_name, foe_name,
                                   foe_key=foe_key)
            if msg is None:
                return [f"no such skill — your moves: {MOVE_HELP}."]
            out.append(msg)
        else:
            return [f"your moves: {MOVE_HELP}."]
        return out


# ── duel ──────────────────────────────────────────────────────────────────

class DuelGame(_ArenaCombat, MultiGame):
    """1v1 PvP — human vs human, gear + skills + levels all count."""

    name = "pvp"
    description = "1v1 PvP combat — challenge another human"
    min_players = 1      # the lobby fills via join / invite; the fight
    max_players = 2      # needs 2 humans and the game enforces it
    ai_seats = 0
    move_timeout = 120.0
    rules = ("A duel to the death against another human — no house AI. "
             "attack · focus · fury · defend · potion · skill <name> · "
             "item <gear>. Your level, equipped gear, and learned skills "
             "all fight with you. 120 seconds per move: stall twice and "
             "you auto-guard, stall a third time and you forfeit. "
             "Start one in a group (/pvp) or challenge DM-to-DM "
             "(/pvp @user, /arena challenge @user).")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"fighters": {}, "base": {}, "skill_cd": {},
                "skill_used": {}, "crit_bonus": {}, "dmg": {},
                "misses": {}, "set_bonus": {}, "gear_wear": {},
                "consumed": {}, "gear_equipped": {},
                "done": False, "winner_key": None, "lobby": True}

    def _ready(self, room: Room) -> bool:
        return len(room.humans) >= 2

    def setup(self, room: Room, mind: GameMind) -> str:
        for p in room.humans:
            self._build_fighter(room, p)
        if not self._ready(room):
            room.state["lobby"] = True
            return ("⚔️ duel lobby — waiting for a challenger.\n"
                    "in a group: /game join (or just speak). "
                    "DM-to-DM: /pvp @user or /arena challenge @user.")
        room.state["lobby"] = False
        a, b = room.humans[0], room.humans[1]
        fa, fb = room.state["fighters"][a.key], room.state["fighters"][b.key]
        return self._intro(a, b, fa, fb)

    def _intro(self, a: Player, b: Player,
               fa: dict, fb: dict) -> str:
        return (
            f"⚔️ DUEL — {a.name} vs {b.name}!\n"
            f"{a.name}: {fa['hp']} HP · {fa['atk']} atk · {fa['def']} def\n"
            f"{b.name}: {fb['hp']} HP · {fb['atk']} atk · {fb['def']} def\n"
            f"{MOVE_HELP}\n{a.name} moves first — 120s on the clock.")

    def on_join(self, room: Room, player: Player,
                mind: GameMind) -> str | None:
        if player.key in room.state.get("fighters", {}):
            return None
        self._build_fighter(room, player)
        if self._ready(room) and room.state.get("lobby"):
            room.state["lobby"] = False
            a, b = room.humans[0], room.humans[1]
            fa = room.state["fighters"][a.key]
            fb = room.state["fighters"][b.key]
            return self._intro(a, b, fa, fb)
        return None

    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        if not self._ready(room):
            return ["still waiting for a challenger — /game join, "
                    "or challenge someone with /pvp @user."]
        foe = next(p for p in room.humans if p.key != player.key)
        s["misses"][player.key] = 0  # they showed up
        out = self._do_move(room, player, player.key, text, mind,
                            foe.key, foe.name, player.name,
                            foe_key=foe.key)
        win = self._check(room, player, foe)
        if win:
            out.append(win)
        else:
            me = s["fighters"][player.key]
            out.append(f"{foe.name}'s turn — "
                       f"{s['fighters'][foe.key]['hp']} HP vs your "
                       f"{me['hp']} HP.")
        return out

    def _check(self, room: Room, player: Player,
               foe: Player) -> str | None:
        s = room.state
        if s["fighters"][foe.key]["hp"] <= 0:
            s["done"] = True
            s["winner_key"] = player.key
            return f"🏁 {foe.name} goes down — {player.name} takes the duel!"
        if s["fighters"][player.key]["hp"] <= 0:
            s["done"] = True
            s["winner_key"] = foe.key
            return f"🏁 {player.name} goes down — {foe.name} takes the duel!"
        return None

    def on_timeout(self, room: Room, player: Player,
                   mind: GameMind) -> list[str]:
        s = room.state
        if not self._ready(room) or s.get("done"):
            room.advance_turn()
            return ["the duel lobby is still waiting."]
        foe = next(p for p in room.humans if p.key != player.key)
        misses = int(s["misses"].get(player.key, 0)) + 1
        s["misses"][player.key] = misses
        if misses >= 3:
            s["done"] = True
            s["winner_key"] = foe.key
            return [f"⏰ {player.name} never showed — three misses, "
                    f"that's a forfeit. {foe.name} takes the duel!"]
        s["fighters"][player.key]["defending"] = True
        room.advance_turn()
        return [f"⏰ {player.name} stalled — auto-guard "
                f"(miss {misses}/3). {foe.name}, your move."]

    def on_leave(self, room: Room, player: Player,
                 mind: GameMind) -> str | None:
        s = room.state
        if s.get("done") or not self._ready(room):
            return None
        # walkover: the one who stays takes it
        foe = next((p for p in room.humans if p.key != player.key), None)
        if foe is not None:
            s["done"] = True
            s["winner_key"] = foe.key
            return (f"🏳️ {player.name} walked out — "
                    f"{foe.name} takes the duel by walkover.")
        return None

    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("done", False))

    def winner(self, room: Room):
        key = room.state.get("winner_key")
        if key:
            p = room.player(key)
            if p is not None:
                return p
        return None

    def score(self, room: Room, player: Player) -> int:
        return int(room.state.get("dmg", {}).get(player.key, 0))

    def xp_reward(self, won: bool | None, room: Room,
                  player: Player) -> int:
        s = room.state
        if won is True:
            xp = 60
            me = s.get("fighters", {}).get(player.key, {})
            if me and me.get("hp", 0) >= me.get("max_hp", 50) // 2:
                xp += 15  # clean win
            if s.get("crit_kill_by") == player.key:
                xp += 10
            return xp
        return 20

    def final_message(self, room: Room, mind: GameMind) -> str:
        w = self.winner(room)
        if w is not None:
            dmg = int(room.state.get("dmg", {}).get(w.key, 0))
            return f"🏁 {w.name} wins the duel ({dmg} damage dealt)."
        return "the duel fizzles out."

    def describe_state(self, room: Room) -> str:
        s = room.state
        bits = []
        for p in room.humans:
            f = s.get("fighters", {}).get(p.key)
            if f:
                bits.append(f"{p.name}: {max(0, f['hp'])}/{f['max_hp']} HP")
        return " · ".join(bits) if bits else "waiting for a challenger…"


# ── raid ──────────────────────────────────────────────────────────────────

class RaidGame(_ArenaCombat, MultiGame):
    """Co-op raid boss — 1–6 players vs one very big problem."""

    name = "raid"
    description = "raid boss — team up, bring the big one down, split loot"
    min_players = 1
    max_players = 6
    ai_seats = 0
    move_timeout = 120.0
    rules = ("The party vs one raid boss. attack · focus · fury · "
             "defend · potion · skill <name> · item <gear> — everything "
             "hits the boss. It slams back after every full round, "
             "cleaves the whole party every 3rd round, and ENRAGES at "
             "30% HP. Bring friends: the boss scales with party size, "
             "and loot splits by damage dealt. 120s per move.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"fighters": {}, "base": {}, "skill_cd": {},
                "skill_used": {}, "crit_bonus": {}, "dmg": {},
                "misses": {}, "set_bonus": {}, "gear_wear": {},
                "consumed": {}, "gear_equipped": {},
                "round": 0, "round_moves": 0, "done": False, "won": None}

    def _boss_for(self, n: int, rng: random.Random) -> dict[str, Any]:
        boss = new_fighter(hp=300 + 200 * n, atk=14 + 2 * n,
                           dfn=6 + n)
        boss["potions"] = 0
        boss["name"] = rng.choice(BOSS_NAMES)
        boss["enraged"] = False
        return boss

    def setup(self, room: Room, mind: GameMind) -> str:
        s = room.state
        for p in room.humans:
            self._build_fighter(room, p)
        s["fighters"]["boss"] = self._boss_for(
            len(room.humans), self.rng(room))
        b = s["fighters"]["boss"]
        party = ", ".join(p.name for p in room.humans)
        return (
            f"🐲 RAID — {b['name']} rises! "
            f"({b['hp']} HP · {b['atk']} atk · {b['def']} def)\n"
            f"party: {party} — more hunters can /game join, "
            f"but the boss grows with them.\n"
            f"{MOVE_HELP} — everything hits the boss.\n"
            f"{room.humans[0].name} strikes first!")

    def on_join(self, room: Room, player: Player,
                mind: GameMind) -> str | None:
        s = room.state
        if player.key in s.get("fighters", {}):
            return None
        self._build_fighter(room, player)
        b = s.get("fighters", {}).get("boss")
        if b is not None and not s.get("done"):
            # reinforcements rally the boss too — the fight stays fair
            b["max_hp"] += 200
            b["hp"] += 200
            b["atk"] += 2
            return (f"⚔️ {player.name} joins the raid! "
                    f"{b['name']} feeds on the numbers "
                    f"(+200 HP, +2 atk) — now {b['hp']} HP.")
        return f"⚔️ {player.name} joins the raid!"

    def _alive(self, room: Room) -> list[Player]:
        s = room.state
        return [p for p in room.humans
                if s.get("fighters", {}).get(p.key, {}).get("hp", 0) > 0]

    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        b = s.get("fighters", {}).get("boss")
        if b is None or s.get("done"):
            return []
        me = s["fighters"].get(player.key)
        if me is None:
            return ["you're not in this raid — /game join."]
        if me["hp"] <= 0:
            return [f"{player.name} is down — the raid fights on "
                    "without you."]
        s["misses"][player.key] = 0
        out = self._do_move(room, player, player.key, text, mind,
                            "boss", b["name"], player.name,
                            foe_key=None)
        return self._after_player(room, player, mind, out)

    def _after_player(self, room: Room, player: Player, mind: GameMind,
                      out: list[str]) -> list[str]:
        """Victory check, then the boss turn once every living hunter
        has acted this round."""
        s = room.state
        b = s["fighters"]["boss"]
        if b["hp"] <= 0:
            s["done"] = True
            s["won"] = True
            out.append(f"🏁 {b['name']} falls! the raid is victorious!")
            return out
        alive = self._alive(room)
        if not alive:
            # nobody left standing (the boss ended it outside the
            # normal flow) — record the wipe
            s["done"] = True
            s["won"] = False
            out.append(f"💀 the party wipes — {b['name']} stands "
                       "triumphant.")
            return out
        s["round_moves"] = int(s.get("round_moves", 0)) + 1
        if s["round_moves"] >= len(alive):
            s["round_moves"] = 0
            self._boss_turn(room, mind, out)
            if not self._alive(room):
                s["done"] = True
                s["won"] = False
                out.append(f"💀 the party wipes — {b['name']} stands "
                           "triumphant.")
        return out

    def _boss_turn(self, room: Room, mind: GameMind,
                   out: list[str]) -> None:
        s = room.state
        b = s["fighters"]["boss"]
        s["round"] = int(s.get("round", 0)) + 1
        rnd = s["round"]
        # enrage: once, at 30% HP
        if not b.get("enraged") and b["hp"] <= b["max_hp"] * 0.3:
            b["enraged"] = True
            b["atk"] = int(round(b["atk"] * 1.5))
            out.append(f"😡 {b['name']} ENRAGES — its attacks surge!")
        alive = self._alive(room)
        if not alive:
            return
        if rnd % 3 == 0:
            # cleave: everyone eats it
            out.append(f"🌪️ {b['name']} CLEAVES — the whole party "
                       "is in the arc!")
            for p in alive:
                msg, _rep = self._strike_msg(
                    room, mind, "boss", p.key, b["name"], p.name,
                    atk_key=None, dfn_key=p.key, mult=0.9)
                out.append(msg)
        else:
            target = mind.rng.choice(alive)
            msg, _rep = self._strike_msg(
                room, mind, "boss", target.key, b["name"], target.name,
                atk_key=None, dfn_key=target.key, mult=1.6)
            out.append(f"💥 {b['name']} SLAMS {target.name}! {msg}")

    def on_timeout(self, room: Room, player: Player,
                   mind: GameMind) -> list[str]:
        s = room.state
        b = s.get("fighters", {}).get("boss")
        if b is None or s.get("done"):
            return []
        me = s["fighters"].get(player.key)
        if me is None or me["hp"] <= 0:
            room.advance_turn()
            return []
        misses = int(s["misses"].get(player.key, 0)) + 1
        s["misses"][player.key] = misses
        # raids keep pace: a stalled hunter auto-attacks
        out = [f"⏰ {player.name} hesitated — auto-attack!"]
        msg, _rep = self._strike_msg(
            room, mind, player.key, "boss", player.name, b["name"],
            atk_key=player.key, dfn_key=None)
        out.append(msg)
        return self._after_player(room, player, mind, out)

    def on_leave(self, room: Room, player: Player,
                 mind: GameMind) -> str | None:
        # the raid fights on — the boss doesn't care who left
        return f"{player.name} retreats from the raid."

    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("done", False))

    def winner(self, room: Room):
        s = room.state
        if s.get("won") is True:
            return "all"  # the whole party won — engine credits everyone
        if s.get("won") is False:
            return Player(key="ai:raid", platform="ai",
                          name=room.state["fighters"]["boss"]["name"], is_ai=True)
        return None

    def _shares(self, room: Room) -> dict[str, int]:
        """Damage share per player, 0–100."""
        s = room.state
        total = sum(int(s.get("dmg", {}).get(p.key, 0))
                    for p in room.humans) or 1
        return {p.key: int(round(
            100 * int(s.get("dmg", {}).get(p.key, 0)) / total))
            for p in room.humans}

    def score(self, room: Room, player: Player) -> int:
        # the economy turns score into a proportional coin bonus —
        # loot splits by damage dealt
        return self._shares(room).get(player.key, 0)

    def xp_reward(self, won: bool | None, room: Room,
                  player: Player) -> int:
        if won is True:
            share = self._shares(room).get(player.key, 0)
            return 70 + min(50, share // 2)
        return 20

    def final_message(self, room: Room, mind: GameMind) -> str:
        s = room.state
        b = s.get("fighters", {}).get("boss") or {}
        if s.get("won") is True:
            shares = self._shares(room)
            order = sorted(room.humans,
                           key=lambda p: shares.get(p.key, 0),
                           reverse=True)
            lines = [f"🏁 {b.get('name', 'the boss')} falls — "
                     "the raid splits the loot by damage:"]
            for p in order:
                dmg = int(s.get("dmg", {}).get(p.key, 0))
                lines.append(f"  {p.name}: {dmg} dmg "
                             f"({shares.get(p.key, 0)}%)")
            if order:
                lines.append(f"👑 MVP: {order[0].name}!")
            return "\n".join(lines)
        if s.get("won") is False:
            return (f"💀 {b.get('name', 'the boss')} stands over the "
                    "fallen party.")
        return "the raid disperses."

    def describe_state(self, room: Room) -> str:
        s = room.state
        b = s.get("fighters", {}).get("boss")
        if b is None:
            return "gathering the party…"
        bits = [f"🐲 {b['name']}: {max(0, b['hp'])}/{b['max_hp']} HP"]
        for p in room.humans:
            f = s.get("fighters", {}).get(p.key, {})
            hp = max(0, f.get("hp", 0))
            bits.append(f"{p.name}: {hp} HP")
        return " · ".join(bits)


PVP_GAMES: tuple[MultiGame, ...] = (DuelGame(), RaidGame())
