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
from ..combat import new_fighter, strike, tick_fighter, skill_power_mult
from ..players import Player
from .base import MultiGame, Room

__all__ = ["DuelGame", "RaidGame", "PVP_GAMES"]

MOVE_HELP = ("attack · focus · fury · defend · potion · "
             "skill <name> · combo <a> + <b> · item <gear|potion|shield>")

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

    #: first words that count as a combat move attempt. Used by
    #: ``is_move_text`` so casual chat during a duel ("say that again?")
    #: falls through to normal conversation instead of triggering
    #: "waiting on X" spam.
    MOVE_VERBS = frozenset({
        "attack", "focus", "fury", "defend", "potion",
        "item", "skill", "combo",
    })

    def is_move_text(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        first = t.split(None, 1)[0]
        return first in self.MOVE_VERBS

    # ── fighter construction ──────────────────────────────────────────
    def _build_fighter(self, room: Room, player: Player) -> dict[str, Any]:
        """Build one human's fighter from the engine mirrors
        (progression / loadout / skills / RPG attributes / titles).
        Idempotent — safe to call on join and on accept."""
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
            "atk": f["atk"], "def": f["def"], "max_hp": f["max_hp"]}
        s.setdefault("fighters", {})[key] = f
        s.setdefault("skill_cd", {})[key] = {}
        s.setdefault("skill_used", {})[key] = []
        s.setdefault("crit_bonus", {})[key] = float(pb["crit"])
        s.setdefault("dmg", {}).setdefault(key, 0)
        s.setdefault("misses", {}).setdefault(key, 0)
        self._recache_fighter(room, key, key)
        # a fresh fighter starts at full HP (recaches preserve damage)
        s["fighters"][key]["hp"] = s["fighters"][key]["max_hp"]
        return f

    def _apply_rpg_and_titles(self, room: Room, player_key: str,
                              seat: str) -> None:
        """Fold RPG attributes + active-title battle effects into a
        fighter — the same kit the solo arena applies (see
        ``BattleArenaGame._apply_stats``).

        Runs at the end of ``_recache_fighter``, which resets atk/def/
        max_hp to base first, so re-application is idempotent.  The
        stamina HP only *widens* the pool: current HP is restored to
        its pre-apply value (clamped to the new max) so a mid-duel
        recache — gear shattering or being equipped — never heals.
        """
        s = room.state
        f = s["fighters"][seat]
        hp_before = int(f.get("hp", 0))
        try:
            from ..stats import (StatBlock, apply_stats_to_fighter,
                                 gear_stat_bonuses)
            raw = s.get("rpg_stats", {}).get(player_key)
            stats = StatBlock.from_dict(raw)
            loadout = s.get("loadout", {}).get(player_key, {})
            apply_stats_to_fighter(
                f, stats, gear_stat_bonuses(loadout))
        except Exception:  # noqa: BLE001
            pass
        try:
            from ..titles import title_battle_effects
            title = s.get("titles", {}).get(player_key, "")
            effects = title_battle_effects(title)
            for ek, val in effects.items():
                if ek in ("atk", "def", "max_hp"):
                    f[ek] = int(f.get(ek, 0)) + int(val)
        except Exception:  # noqa: BLE001
            pass
        # apply_stats_to_fighter bumps current HP by the stamina gain —
        # undo it so recaches never heal; only the pool widens.
        f["hp"] = min(hp_before, int(f.get("max_hp", hp_before)))

    def _recache_fighter(self, room: Room, player_key: str,
                         seat: str) -> None:
        """Re-fold gear + set bonus into a fighter (equip/break).

        Resets to base first (now including max_hp) so RPG attributes
        and title effects re-apply cleanly without stacking."""
        from ..gear import SET_BONUSES
        s = room.state
        f = s["fighters"][seat]
        base = s.get("base", {}).get(seat, {"atk": 10, "def": 5,
                                            "max_hp": 50})
        f["atk"], f["def"] = int(base["atk"]), int(base["def"])
        f["max_hp"] = int(base.get("max_hp", 50))
        f["hp"] = min(int(f.get("hp", f["max_hp"])), int(f["max_hp"]))
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
        # RPG attributes + title effects ride on every recache
        self._apply_rpg_and_titles(room, player_key, seat)
        f["hp"] = min(int(f.get("hp", 0)), int(f["max_hp"]))

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
        # mana: techniques burn fuel — same rule as the solo arena.
        # Fighters without a mana pool (legacy, unit tests) cast freely.
        if "max_mana" in me:
            need = int(defn.mana_cost)
            have = int(me.get("mana", 0))
            if have < need:
                return (f"not enough mana — {defn.name} needs {need}, "
                        f"you have {have}. It regenerates each turn.")
            me["mana"] = have - need
        msg = ""
        if defn.slug == "shadow_step":
            me["dodge_next"] = True
            msg = f"{me_name} melts into shadow — the next attack misses."
            if defn.counter_mult:
                pm, rep = self._strike_msg(
                    room, mind, seat, foe_seat, me_name, foe_name,
                    atk_key=player.key, dfn_key=foe_key,
                    mult=defn.counter_mult * skill_power_mult(me))
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
            # intelligence sharpens every striking technique
            tech_mult = (defn.mult or 1.0) * skill_power_mult(me)
            for _ in range(max(1, defn.hits)):
                pm, rep = self._strike_msg(
                    room, mind, seat, foe_seat, me_name, foe_name,
                    atk_key=player.key, dfn_key=foe_key,
                    mult=tech_mult,
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

    def _dual_cast(self, room: Room, player: Player, seat: str,
                   foe_seat: str, ref: str, mind: GameMind,
                   me_name: str, foe_name: str,
                   foe_key: str | None = None) -> str | None:
        """Weave two complementary skills as one dual-cast combo.

        ``ref`` is ``"<skill1> + <skill2>"`` or ``"<skill1> <skill2>"`` —
        the same ordered-pair catalog the solo arena uses (including
        ``war_cry → slaying_force``, "Cutyp's Judgment").  None = not a
        usable pairing ref, so the caller falls through to the move list.
        """
        from ..skills import (SKILL_CATALOG, effective_def, resolve_skill,
                              find_combo, combo_success_chance)
        s = room.state
        me = s["fighters"][seat]
        foe = s["fighters"][foe_seat]
        ref = (ref or "").strip()
        if "+" in ref:
            first_ref, second_ref = [p.strip() for p in ref.split("+", 1)]
        else:
            bits = ref.split()
            if len(bits) < 2:
                return None
            first_ref, second_ref = bits[0], " ".join(bits[1:])
        d1 = resolve_skill(first_ref)
        d2 = resolve_skill(second_ref)
        if d1 is None or d2 is None or d1.kind != "active" \
                or d2.kind != "active":
            return None
        learned = s.get("skills", {}).get(player.key, [])
        tiers = s.get("skill_tiers", {}).get(player.key, {})
        if d1.slug not in learned or d2.slug not in learned:
            known = [SKILL_CATALOG[x].name for x in learned
                     if x in SKILL_CATALOG]
            hint = (f"you know: {', '.join(known)}."
                    if known else "you haven't learned any skills yet.")
            return f"dual-cast needs both techniques learned. {hint}"
        combo = find_combo(d1.slug, d2.slug)
        if combo is None:
            if find_combo(d2.slug, d1.slug) is not None:
                return (f"{d2.name} → {d1.name} chains, not the reverse — "
                        f"the setup must come first.")
            return (f"{d1.name} and {d2.name} don't chain — only "
                    f"complementary techniques combo. /skill combos "
                    f"lists every pairing.")
        cd = s.setdefault("skill_cd", {}).setdefault(seat, {})
        used = s.setdefault("skill_used", {}).setdefault(seat, [])
        t1 = int(tiers.get(d1.slug, 1))
        t2 = int(tiers.get(d2.slug, 1))
        e1 = effective_def(d1, t1)
        e2 = effective_def(d2, t2)
        for e in (e1, e2):
            if int(cd.get(e.slug, 0)) > 0:
                return (f"{e.name} is recovering — "
                        f"{cd[e.slug]} turn(s) left.")
            if e.once_per_battle and e.slug in used:
                return f"{e.name} is spent for this battle."
        # mana: both skills' costs plus the combo's weave cost.
        # Fighters without a mana pool (legacy, tests) weave freely.
        mana_need = int(e1.mana_cost) + int(e2.mana_cost) + combo.mana_cost
        if "max_mana" in me:
            mana_have = int(me.get("mana", 0))
            if mana_have < mana_need:
                return (f"not enough mana — the weave needs {mana_need}, "
                        f"you have {mana_have}. It regenerates each turn.")
            me["mana"] = mana_have - mana_need
        hp_cost = max(1, int(me["max_hp"] * combo.hp_cost_pct))
        me["hp"] = max(1, int(me["hp"]) - hp_cost)
        cd[e1.slug] = int(e1.cooldown) + combo.extra_cd
        cd[e2.slug] = int(e2.cooldown) + combo.extra_cd
        if e1.once_per_battle:
            used.append(e1.slug)
        if e2.once_per_battle:
            used.append(e2.slug)
        # ── the weave: can you hold both techniques at once? ──
        intel = int(me.get("intelligence", 0))
        chance = combo_success_chance(combo, t1, t2, intel)
        if chance < float(s.get("dual_lowest_odds", 1.0)):
            s["dual_lowest_odds"] = chance
        if mind.rng.random() >= chance:
            return (f"💔 the dual-cast unravels! You wove {e1.name} into "
                    f"{e2.name} and it slipped — −{hp_cost} HP, both "
                    f"techniques recovering. ({int(chance * 100)}% chance)")
        s["dual_casts"] = int(s.get("dual_casts", 0)) + 1
        parts = [f"⚡ DUAL-CAST — {combo.name}! {combo.desc}"]
        parts.append(f"🩸 −{hp_cost} HP sacrificed.")
        # setup: apply the first skill's non-strike effect
        if e1.slug == "war_cry":
            me["atk"] += int(e1.atk_buff)
            me["warcry_turns"] = int(e1.buff_turns)
            me["warcry_amt"] = int(e1.atk_buff)
            parts.append(f"🗣️ +{e1.atk_buff} attack for "
                         f"{e1.buff_turns} turns!")
        elif e1.slug in ("shadow_step", "smoke_bomb", "crane_dance"):
            me["dodge_next"] = True
            parts.append("🌫️ you vanish — their next attack will miss.")
            if e1.slug == "smoke_bomb" and e1.atk_debuff:
                amt = int(e1.atk_debuff)
                foe["atk"] = max(1, int(foe["atk"]) - amt)
                foe["atk_debuff"] = {"turns": int(e1.debuff_turns),
                                    "amt": amt}
                parts.append(f"🌑 −{amt} enemy attack for "
                             f"{e1.debuff_turns} turns.")
            if e1.slug == "crane_dance" and e1.heal_pct:
                heal = int(me["max_hp"] * float(e1.heal_pct))
                me["hp"] = min(me["max_hp"], int(me["hp"]) + heal)
                parts.append(f"💚 +{heal} HP as you flow.")
        elif e1.slug == "second_wind":
            heal = int(me["max_hp"] * float(e1.heal_pct))
            me["hp"] = min(me["max_hp"], int(me["hp"]) + heal)
            parts.append(f"💚 +{heal} HP — breath returns.")
            if combo.second == "war_cry":
                # Phoenix Rising: the combo's own payoff is the buff
                me["atk"] += 5
                me["warcry_turns"] = 4
                me["warcry_amt"] = 5
                parts.append("🔥 +5 attack for 4 turns — risen!")
                return " ".join(parts)
        # payoff: the combined strike — intelligence sharpens the weave
        if combo.mult > 0:
            tech_mult = combo.mult * skill_power_mult(me)
            for _ in range(max(1, combo.hits)):
                pm, rep = self._strike_msg(
                    room, mind, seat, foe_seat, me_name, foe_name,
                    atk_key=player.key, dfn_key=foe_key,
                    mult=tech_mult, ignore_def=combo.ignore_def_pct)
                parts.append(pm)
                if rep.get("crit") and s["fighters"][foe_seat]["hp"] <= 0:
                    s["crit_kill_by"] = seat
                if s["fighters"][foe_seat]["hp"] <= 0:
                    break
        if intel >= 10:
            parts.append(f"(woven at {int(chance * 100)}% — "
                         f"intelligence steadied your hands)")
        return " ".join(parts)

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
        # mana regenerates a little every turn — the well refills
        try:
            from ..stats import MANA_REGEN_PER_TURN
            if "max_mana" in me:
                me["mana"] = min(int(me["max_mana"]),
                                 int(me.get("mana", 0))
                                 + MANA_REGEN_PER_TURN)
        except Exception:  # noqa: BLE001
            pass
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
        elif t.startswith("combo ") or t.startswith("dual "):
            ref = t[6:].strip() if t.startswith("combo ") else t[5:].strip()
            msg = self._dual_cast(room, player, seat, foe_seat, ref,
                                  mind, me_name, foe_name, foe_key=foe_key)
            if msg is None:
                return [f"no such pairing — your moves: {MOVE_HELP}."]
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
             "combo <a> + <b> · item <gear>. Your level, equipped gear, "
             "learned skills, RPG attributes and title all fight with you. "
             "120 seconds per move: stall twice and "
             "you auto-guard, stall a third time and you forfeit. "
             "Start one in a group (/pvp) or challenge DM-to-DM "
             "(/pvp @user, /arena challenge @user).")
    howto = ("\u2694\ufe0f PvP duel lobby open — /game join to step in (2 fighters).\n"
             "Your level, gear, skills and titles all fight with you.\n"
             "Moves: attack \u00b7 focus \u00b7 fury \u00b7 defend \u00b7 potion "
             "\u00b7 skill <name> \u00b7 combo <a> + <b> \u00b7 item <gear>.\n"
             "120s per move — stall twice: auto-guard. Third stall: forfeit.")

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
        return self._intro(room, a, b)

    def _intro(self, room: Room, a: Player, b: Player) -> str:
        return (
            f"⚔️ DUEL — {a.name} vs {b.name}!\n"
            f"{self._fighter_line(room, a)}\n"
            f"{self._fighter_line(room, b)}\n"
            f"{MOVE_HELP}\n{a.name} moves first — 120s on the clock.")

    def _fighter_line(self, room: Room, p: Player) -> str:
        """One-liner for a duelist: stats + level, gear, skills."""
        from ..skills import SKILL_CATALOG, effective_def
        s = room.state
        f = s["fighters"][p.key]
        bits = [f"{p.name}: {f['hp']} HP · {f['atk']} atk · {f['def']} def"]
        lvl = s.get("progression", {}).get(p.key, {}).get("level", 1)
        if int(lvl) > 1:
            bits.append(f"level {lvl}")
        loadout = s.get("loadout", {}).get(p.key, {})
        worn = [pc["name"] for sl, pc in
                (("weapon", loadout.get("weapon")),
                 ("armor", loadout.get("armor")))
                if pc and pc.get("name")]
        if worn:
            bits.append("wielding " + " + ".join(worn))
        slugs = s.get("skills", {}).get(p.key, [])
        tiers = s.get("skill_tiers", {}).get(p.key, {})
        names = [effective_def(sl, tiers.get(sl, 1)).name
                 for sl in slugs if sl in SKILL_CATALOG]
        if names:
            bits.append("🥋 " + ", ".join(names))
        title = s.get("titles", {}).get(p.key, "")
        if title:
            bits[0] = f"{title} " + bits[0]
        return " — ".join(bits)

    def on_join(self, room: Room, player: Player,
                mind: GameMind) -> str | None:
        if player.key in room.state.get("fighters", {}):
            return None
        self._build_fighter(room, player)
        if self._ready(room) and room.state.get("lobby"):
            room.state["lobby"] = False
            a, b = room.humans[0], room.humans[1]
            return self._intro(room, a, b)
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
            # don't ping an empty lobby forever — 3 nudges, then quiet.
            # the lobby stays open; anyone can still /game join.
            nudges = int(s.get("lobby_nudges", 0)) + 1
            s["lobby_nudges"] = nudges
            if nudges <= 3:
                return ["the duel lobby is still waiting."]
            return []
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
                vitals = f"{p.name}: {max(0, f['hp'])}/{f['max_hp']} HP"
                if "max_mana" in f:
                    vitals += (f" · ⚡{int(f.get('mana', 0))}/"
                               f"{int(f['max_mana'])} mana")
                bits.append(vitals)
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
             "defend · potion · skill <name> · combo <a> + <b> · item <gear> — everything "
             "hits the boss. It slams back after every full round, "
             "cleaves the whole party every 3rd round, and ENRAGES at "
             "30% HP. Bring friends: the boss scales with party size, "
             "and loot splits by damage dealt. 120s per move.")
    howto = ("\U0001f432 Raid boss is live — /game join to fight (up to 6).\n"
             "Everyone hits the boss: attack \u00b7 focus \u00b7 fury \u00b7 defend "
             "\u00b7 potion \u00b7 skill <name> \u00b7 combo <a> + <b>.\n"
             "It slams back every round and ENRAGES at 30% HP. "
             "Loot splits by damage dealt.\n"
             "120s per move.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"fighters": {}, "base": {}, "skill_cd": {},
                "skill_used": {}, "crit_bonus": {}, "dmg": {},
                "misses": {}, "set_bonus": {}, "gear_wear": {},
                "consumed": {}, "gear_equipped": {},
                "round": 0, "round_moves": 0, "done": False, "won": None}

    def _boss_for(self, n: int, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        """The raid boss, scaled by party size and difficulty.

        Harder modes field a meaner boss — difficulty is a real dial
        here too, not a label.
        """
        mods = {"easy": (250, 150, 12, 2, 5),
                "normal": (300, 200, 14, 2, 6),
                "hard": (350, 250, 16, 2, 7),
                "expert": (400, 300, 18, 3, 8),
                }.get(difficulty, (300, 200, 14, 2, 6))
        base_hp, per_hp, base_atk, per_atk, base_def = mods
        boss = new_fighter(hp=base_hp + per_hp * n, atk=base_atk + per_atk * n,
                           dfn=base_def + n)
        boss["potions"] = 0
        boss["name"] = rng.choice(BOSS_NAMES)
        boss["enraged"] = False
        boss["difficulty"] = difficulty
        return boss

    def setup(self, room: Room, mind: GameMind) -> str:
        s = room.state
        for p in room.humans:
            self._build_fighter(room, p)
        difficulty = str(s.get("_difficulty") or "normal").lower()
        s["difficulty"] = difficulty
        s["fighters"]["boss"] = self._boss_for(
            len(room.humans), self.rng(room), difficulty=difficulty)
        b = s["fighters"]["boss"]
        party = ", ".join(p.name for p in room.humans)
        diff_note = (f" 🎯 {difficulty} — a meaner beast."
                     if difficulty in ("hard", "expert") else "")
        return (
            f"🐲 RAID — {b['name']} rises! "
            f"({b['hp']} HP · {b['atk']} atk · {b['def']} def){diff_note}\n"
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

    #: raid-exclusive drop odds — base chance plus a damage-share kicker.
    #: The MVP (100% share) rolls at 15%; a bystander (0%) at 5%.  Low
    #: enough to feel lucky, high enough to keep raiders coming back.
    DROP_BASE = 0.05
    DROP_SHARE_BONUS = 0.10

    def victory_loot(self, room: Room) -> dict[str, list[str]]:
        """Raid-exclusive gear drops, rolled on a boss kill.

        Returns {player_key: [gear_slug, ...]}.  Only raid bosses drop
        these pieces — they're never sold in the shop.  The engine
        grants them via GearStore (see ``_grant_victory_loot``).
        """
        from ..gear import RAID_EXCLUSIVE_GEAR
        s = room.state
        if s.get("won") is not True or not RAID_EXCLUSIVE_GEAR:
            return {}
        rng = self.rng(room)
        shares = self._shares(room)
        out: dict[str, list[str]] = {}
        for p in room.humans:
            share = shares.get(p.key, 0)
            chance = self.DROP_BASE + self.DROP_SHARE_BONUS * (share / 100.0)
            if rng.random() < chance:
                out.setdefault(p.key, []).append(
                    rng.choice(RAID_EXCLUSIVE_GEAR))
        return out

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
