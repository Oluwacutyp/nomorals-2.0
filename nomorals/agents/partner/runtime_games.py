"""RuntimeGamesMixin: PartnerRuntime command group (games)."""

from __future__ import annotations

import time
from typing import Any, Callable
from ...core.logging_setup import get_logger
from ...social.chat.base import ChatKind, ChatMessage, ChatRef
_log = get_logger(__name__)


def _pop_opt(toks: list, *names: str) -> tuple:
    """Pop ``--name value`` / ``--name=value`` from a token list.

    Returns (value, remaining_tokens).  Last occurrence wins.
    """
    out: list = []
    val = None
    i = 0
    while i < len(toks):
        t = toks[i]
        hit = False
        for nm in names:
            if t == nm and i + 1 < len(toks):
                val = toks[i + 1]
                i += 2
                hit = True
                break
            if t.startswith(nm + "="):
                val = t.split("=", 1)[1]
                i += 1
                hit = True
                break
        if not hit:
            out.append(t)
            i += 1
    return val, out


class RuntimeGamesMixin:
    """RuntimeGamesMixin for :class:`PartnerRuntime`."""


    def _control_game(self, tail: str, chat_key: str, *,
                       player: Any = None, kind: str = "dm") -> str:
        from ..features import feature_enabled

        _log.debug("_control_game: tail=%r, chat_key=%s, player=%s, kind=%s",
                  tail, chat_key, player, kind)
        if not feature_enabled(self.context, "games"):
            _log.debug("games feature is disabled")
            return "games are off. /features games on"
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else ""
        _log.debug("_control_game: verb=%r", verb)
        engine = self._game_engine()
        live = engine.live(chat_key)
        _log.debug("_control_game: live=%s", live)

        if verb in {"", "list", "help"}:
            lines = []
            if live is not None:
                lines.append(
                    f"live game: {live.game} — send your move (or /game quit)")
            lines.append(engine.list_games())
            lines.append("  /game <name> — start · /game rematch — run it back")
            lines.append("  /game <name> [easy|normal|hard|expert] — AI/puzzle difficulty")
            lines.append("  /game <name> [daily|timed|<variant>] — modes & mastery unlocks")
            lines.append("  /mastery — your per-game mastery tiers")
            lines.append("  /game invite <game> [who] · /game accept <code> — DM duels")
            lines.append("  /game relay — live DM-to-DM connection status")
            lines.append("  /pvp · /pvp @user — 1v1 PvP lobby or DM-to-DM challenge")
            lines.append("  /arena challenge @user · /arena raid — same, arena-branded")
            lines.append("  /raid — team up against the raid boss")
            lines.append("  duel moves: attack · focus · fury · defend · potion · "
                         "skill <name> · combo <a> + <b> · item <gear|potion|shield>")
            lines.append("  /gift @user 100 — gift coins · /gift @user <gear|item>")
            lines.append("  /gift confirm · /gift cancel · /gift history")
            lines.append("  /game stats [name] · /game balance · /game shop · "
                         "/game leaderboard")
            return "\n".join(lines)
        if verb == "quit":
            # capture the relay (if any) BEFORE quitting — the engine
            # tears it down, and the opponent deserves to hear about it
            relay = self._game_relay()
            doomed = relay.get_relay_for_chat(chat_key)
            # mint the rematch invite BEFORE the relay closes (it needs
            # the room to exist)
            rematch = (relay.rematch_invite(chat_key)
                       if doomed is not None else None)
            out = engine.quit(chat_key)
            if doomed is not None:
                who = player.name if player is not None else "your opponent"
                msg = (f"🏁 {who} closed the {doomed.game_name} duel.")
                if rematch is not None:
                    msg += (f"\n🔁 rematch? /game accept {rematch.code} "
                            f"(1 hour)")
                self._relay_send(doomed.other_chat(chat_key), msg)
            return "\n".join(out)
        if verb == "rematch":
            room, msgs = engine.rematch(chat_key)
            return "\n".join(msgs)
        if verb in {"leaderboard", "board", "ranks"}:
            game = parts[1].lower() if len(parts) > 1 else ""
            return engine.board.render(10, game=game)
        if verb == "stats":
            want = " ".join(parts[1:]).strip().lower()
            if want:
                for prof in engine.store.all():
                    if want in prof.key.lower() or want in prof.name.lower():
                        return self._profile_line(prof)
                return f"no player found matching {want!r}."
            if player is not None:
                return self._profile_line(
                    engine.store.get(player.key),
                    display_name=getattr(player, "name", ""),
                )
            return "usage: /game stats [name]"
        if verb == "balance":
            if player is None:
                return "balance needs a chat sender — run it where you play."
            prof = engine.store.get(player.key)
            items = ", ".join(f"{k}×{v}" for k, v in (prof.items or {}).items()) or "none"
            return f"🪙 {prof.coins} coins · {prof.points} points · items: {items}"
        if verb == "shop":
            rest = " ".join(parts[1:]).strip()
            if rest.startswith("buy ") and player is not None:
                ok, msg = engine.economy.purchase(player, rest[4:].strip())
                return msg
            return engine.economy.catalog_text("", player)
        if verb == "join":
            if player is None:
                return "join needs a chat sender — say it where the game is live."
            return "\n".join(engine.join(chat_key, player)) or "joined."
        if verb == "invite":
            if player is None:
                return "invite needs a chat sender — say it where the game is live."
            if len(parts) < 2:
                return ("usage: /game invite <game_name> [who]\n"
                        "the code is the invite — your friend accepts with "
                        "/game accept <code> from any chat.")
            game_name = parts[1].lower()
            to_label = parts[2] if len(parts) > 2 else ""
            return self._challenge_player(game_name, to_label, chat_key,
                                          player)
        if verb == "pvp" and len(parts) > 1 and parts[1].lower() not in (
                "easy", "normal", "hard", "expert"):
            # /pvp @user — challenge a specific person DM-to-DM
            if player is None:
                return "pvp needs a chat sender — say it where you play."
            return self._challenge_player("pvp", parts[1], chat_key,
                                          player)
        if verb == "accept":
            if player is None:
                return ("accept needs a chat sender — paste the /game accept "
                        "<code> in the chat where you play.")
            if len(parts) < 2:
                return "usage: /game accept <invite_code>"
            code = parts[1]
            try:
                relay = self._game_relay()
                relay_room = relay.accept_invite(code, chat_key, player)
                # tell the inviter their friend joined (the accepter is
                # already looking at this chat)
                self._relay_send(
                    relay_room.chat_a,
                    f"🎮 {player.name} accepted your "
                    f"{relay_room.game_name} invite — game on! play in "
                    f"your DM, moves are relayed.")
                return (f"🎮 connected! {relay_room.game_name} vs "
                        f"{relay_room.player_a.name} — play here in this "
                        f"chat, your moves are relayed to them.\n"
                        f"/game relay shows the connection status anytime.")
            except ValueError as exc:
                return str(exc)
        if verb == "relay":
            # /game relay — connection status for DM-to-DM multiplayer
            relay = self._game_relay()
            status = relay.status_text(chat_key)
            if status is None:
                return ("no active relay here.\n"
                        "/game invite <game> [who] — challenge someone\n"
                        "/game accept <code> — join their game")
            # heartbeat: a status check proves this side is alive
            code = relay.chat_to_relay.get(chat_key)
            if code:
                relay.touch(code)
            return status
        if verb in engine.games:
            if player is None:
                return "start a game from a chat — I need to know who's at the table."
            # "/hangman daily" — same word for everyone, all day
            # "/game case timed" — countdown mode with a speed bonus
            # "/game connect4 hard" — AI/puzzle difficulty
            # "/game gomoku big" — mastery-gated variant (see game.variants)
            words = [p.lower() for p in parts[1:]]
            daily = "daily" in words
            timed = "timed" in words
            difficulty = next(
                (w for w in words
                 if w in ("easy", "normal", "hard", "expert")),
                "normal")
            game_def = engine.games[verb]
            variant = next(
                (w for w in words
                 if w in getattr(game_def, "variants", {})),
                "")
            try:
                room, msgs = engine.start(chat_key, verb, player, kind=kind,
                                          daily=daily, timed=timed,
                                          difficulty=difficulty,
                                          variant=variant)
            except ValueError as exc:
                return str(exc)
            if msgs and not msgs[0].startswith("🎮"):
                msgs[0] = f"🎮 {msgs[0]}"  # the table-opening banner
            # show the difficulty mode so it's clear what's active
            diff_line = f"🎯 difficulty: {difficulty}"
            if daily:
                diff_line += " · 📅 daily"
            if timed:
                diff_line += " · ⏱️ timed"
            if variant:
                diff_line += f" · 🃏 {variant}"
            # mastery tier: shown at every game start
            try:
                from ...games.mastery import (get_one_game_stats,
                                              mastery_line)
                diff_line += " · " + mastery_line(
                    verb, get_one_game_stats(engine.db, player.key, verb))
            except Exception:  # noqa: BLE001
                pass
            body = "\n".join(msgs) or engine.describe(room)
            return f"{diff_line}\n{body}"
        return (f"unknown game {verb!r} — /game list to see the table.")

    def _challenge_player(self, game_name: str, to_label: str,
                          chat_key: str, player: Any) -> str:
        """DM-to-DM challenge: create a relay invite for ``game_name``
        addressed to ``to_label``. Shared by ``/game invite``,
        ``/pvp @user`` and ``/arena challenge``."""
        if player is None:
            return "challenging needs a chat sender — say it where you play."
        try:
            relay = self._game_relay()
            invite = relay.create_invite(
                chat_key, player, game_name, to_label=to_label)
        except ValueError as exc:
            return str(exc)
        share = (f"invite ready for {invite.game_name} — share this:\n"
                 f"/game accept {invite.code}\n"
                 f"(expires in 1 hour, works from any chat)")
        target = self._resolve_invite_target(chat_key, to_label)
        if target is not None:
            sent = self._relay_send(
                target,
                f"🎮 {player.name} invited you to play "
                f"{invite.game_name}!\n"
                f"to accept: /game accept {invite.code}\n"
                f"(expires in 1 hour)")
            if sent:
                return (f"invite sent to {to_label} for "
                        f"{invite.game_name}.\n{share}")
        return share

    # ── arena gear commands ──────────────────────────────────────────────────
    def _control_gear(self, cmd: str, ref: str, *,
                      player: Any = None) -> str:
        """Persistent equipment: /inventory /equip /unequip /repair.

        Gear lives in the game_gear table (never in-memory only): buying
        from /game shop forges a piece with its own durability, wearing
        it down in arena battles, breaking at 0, repairable for coins.
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        engine = self._game_engine()
        store, gear = engine.store, engine.gear
        prof = store.get(player.key)
        ref = (ref or "").strip().lower()

        def _piece_line(inst: Any) -> str:
            from ...games.gear import (GEAR_CATALOG, durability_display,
                                       effective_stats)
            defn = GEAR_CATALOG.get(inst.slug)
            stats = ""
            if defn is not None:
                atk, df = effective_stats(defn)
                stats = (f"+{atk} atk" if defn.slot == "weapon"
                         else f"+{df} def")
                if defn.set_name:
                    stats += f" · {defn.set_name} set"
            broken = " 💥 BROKEN" if inst.broken else ""
            return (f"{inst.display_name()} [{inst.slug}] — {stats} · "
                    f"{durability_display(defn, inst.durability, inst.max_durability)}"
                    f"{broken}")

        if cmd == "inventory":
            lines = [f"🎒 {player.name}'s kit — {prof.coins} coins"]
            worn = gear.equipped(player.key)
            if worn:
                lines.append("equipped:")
                for slot in ("weapon", "armor", "trinket"):
                    inst = worn.get(slot)
                    if inst:
                        lines.append(f"  [{slot}] {_piece_line(inst)}")
            pieces = [i for i in gear.list(player.key) if not i.equipped]
            if pieces:
                lines.append("closet:")
                for inst in pieces:
                    lines.append(f"  {_piece_line(inst)}")
            if not worn and not pieces:
                lines.append("no gear yet — /game shop has swords & armor.")
            from ...games.gear import detect_set_bonus
            bonus = detect_set_bonus(worn)
            if bonus:
                lines.append(
                    f"✨ {bonus.set_name} set complete — {bonus.combo_name}! "
                    f"(+{int(bonus.atk_pct * 100)}% atk, "
                    f"+{int(bonus.def_pct * 100)}% def, combo every "
                    f"{bonus.combo_every} attacks)")
            consumables = ", ".join(
                f"{k}×{v}" for k, v in prof.items.items()) or "none"
            lines.append(f"consumables: {consumables}")
            return "\n".join(lines)

        if cmd == "equip":
            if not ref:
                return "equip what? /inventory lists your gear."
            ok, msg = gear.equip(player.key, ref)
            return msg

        if cmd == "unequip":
            ok, msg = gear.unequip(player.key, ref)
            return msg

        if cmd == "repair":
            if not ref:
                return "repair what? /inventory lists your gear."
            ok, msg = gear.repair(player.key, ref)
            if not ok:
                return msg
            # msg is "DisplayName|cost" — deduct coins atomically, then apply.
            # spend_coins re-checks the balance inside its lock, so a stale
            # prof.coins read can't let two concurrent repairs overspend.
            try:
                name, cost_s = msg.rsplit("|", 1)
                cost = int(cost_s)
            except ValueError:
                return msg
            inst = gear.find(player.key, ref)
            if inst is None:
                return "lost track of that piece — try again."
            # re-verify the piece still needs repair (durability may have
            # changed since the cost was computed)
            if inst.durability >= inst.max_durability:
                return f"{inst.display_name()} is already at full."
            new_balance = store.spend_coins(player, cost,
                                            f"repair:{inst.slug}")
            if new_balance is None:
                prof_now = store.get(player.key)
                return (f"repairing {name} costs {cost}c — "
                        f"you have {prof_now.coins}c.")
            if gear.apply_repair(inst.id):
                return f"🔧 {name} — good as new, {cost}c."
            # repair failed after taking coins — refund, never lose coins.
            # (False also means a concurrent repair already fixed it.)
            store.add_coins(player, cost, f"repair-refund:{inst.slug}")
            return f"repair didn't go through — {cost}c refunded."

        return f"unknown gear command {cmd!r}."

    # ── progression ──────────────────────────────────────────────────────────
    def _control_level(self, *, player: Any = None) -> str:
        """Show persistent progression: level, XP bar, stat growth."""
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        from ...games.progression import (
            ARENA_LOSS_XP, ARENA_WIN_XP, GAME_LOSS_XP, GAME_WIN_XP,
            level_for_xp, level_stat_bonus, xp_bar, xp_for_level,
            xp_progress,
        )
        engine = self._game_engine()
        prof = engine.store.get(player.key)
        level, into, span = xp_progress(prof.xp)
        bonus = level_stat_bonus(level)
        nxt = xp_for_level(level + 1)
        lines = [
            f"⭐ {player.name} — level {level} ({prof.xp} XP)",
            f"   {xp_bar(prof.xp)}",
            f"   {nxt - prof.xp} XP to level {level + 1}",
        ]
        if level > 1:
            bits = [f"+{bonus['max_hp']} max HP", f"+{bonus['atk']} atk"]
            if bonus["def"]:
                bits.append(f"+{bonus['def']} def")
            lines.append(f"   arena stats: {', '.join(bits)}")
        else:
            lines.append("   arena stats: base (level up to grow)")
        lines.append(
            f"   earn: arena win +{ARENA_WIN_XP} / loss +{ARENA_LOSS_XP}, "
            f"other games win +{GAME_WIN_XP} / loss +{GAME_LOSS_XP}")
        if prof.games_played:
            lines.append(
                f"   {prof.games_played} games played · "
                f"{prof.wins} wins · best streak {prof.best_streak}")
        return "\n".join(lines)

    def _control_skills(self, tail: str, *, player: Any = None) -> str:
        """Learnable battle skills: /skill | /skill learn <name> |
        /skill upgrade <name> | /skill combos.

        Skills persist in ``game_skills`` — learned once, yours forever.
        Active skills are cast in the arena with ``skill <name>``;
        passives apply automatically every battle.  Learning costs
        coins and may require a player level.  The exciting actives
        can be upgraded through tiers (II, III) — stronger hits,
        shorter cooldowns, new effects.
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        from ...games.skills import (SKILL_CATALOG, SkillStore, tier_name,
                                     effective_def, max_tier, resolve_skill)
        from ...games.progression import level_for_xp
        engine = self._game_engine()
        store = engine.store
        skills = SkillStore(engine.db)
        prof = store.get(player.key)
        level = level_for_xp(prof.xp)
        learned = skills.learned(player.key)
        tiers = skills.tiers(player.key)
        tail = (tail or "").strip()

        def _skill_line(defn: Any) -> str:
            kind = "⚔️" if defn.kind == "active" else "🛡️"
            cd = f", {defn.cooldown}-turn cd" if defn.cooldown else ""
            extra = ""
            if defn.mult:
                extra = f" · {defn.hits}×{defn.mult:g}× dmg"
            if defn.heal_pct:
                extra = f" · heals {int(defn.heal_pct * 100)}% HP"
            if defn.dodge:
                extra = " · dodges next attack"
            if defn.counter_mult:
                extra += f" · {defn.counter_mult:g}× counter"
            if defn.atk_buff:
                extra = f" · +{defn.atk_buff} atk {defn.buff_turns}t"
            if defn.def_bonus:
                extra = f" · +{defn.def_bonus} def"
            if defn.atk_bonus:
                extra = f" · +{defn.atk_bonus} atk"
            if defn.hp_bonus:
                extra = f" · +{defn.hp_bonus} max HP"
            if defn.crit_bonus:
                extra = f" · +{int(defn.crit_bonus * 100)}% crit"
            if defn.ignore_def_pct:
                extra = (f" · ignores {int(defn.ignore_def_pct * 100)}% "
                         f"def")
            up = " ⬆️" if max_tier(defn) > 1 else ""
            lock = "" if level >= defn.level_req else " 🔒"
            return (f"{kind} {defn.name} [{defn.slug}]{up}{lock}\n"
                    f"   {defn.school} · {defn.desc}\n"
                    f"   lvl {defn.level_req}+ · {defn.cost}c{cd}{extra}")

        def _learned_line(slug: str) -> str:
            defn = SKILL_CATALOG.get(slug)
            if not defn:
                return ""
            tier = tiers.get(slug, 1)
            eff = effective_def(defn, tier)
            tag = "active" if defn.kind == "active" else "passive"
            nxt = ""
            if tier < max_tier(defn):
                up = defn.tiers[tier - 1]
                nxt_name = tier_name(defn, tier + 1)
                nxt = (f" → ⬆️ {nxt_name} ({up.cost}c, lvl {up.level_req}+)")
            return f"  ✅ {eff.name} ({tag}){nxt}"

        if not tail or tail.lower() in ("list", "show"):
            lines = [f"🥋 {player.name}'s skills — level {level}, "
                     f"{prof.coins}c"]
            if learned:
                lines.append("learned:")
                for slug in learned:
                    line = _learned_line(slug)
                    if line:
                        lines.append(line)
                lines.append("cast actives in the arena with "
                             "skill <name>; upgrade with "
                             "/skill upgrade <name>.")
            else:
                lines.append("no skills learned yet.")
            avail = [d for s, d in SKILL_CATALOG.items() if s not in learned]
            if avail:
                lines.append("the school teaches:")
                for defn in avail:
                    lines.append("  " + _skill_line(defn).replace("\n", "\n  "))
                lines.append("learn one: /skill learn <name>")
            return "\n".join(lines)

        parts = tail.split(None, 1)
        verb = parts[0].lower()
        if verb == "combos":
            from ...games.skills import COMBO_CATALOG
            if not COMBO_CATALOG:
                return "no dual-cast combos exist yet."
            lines = ["⚡ dual-cast combos — setup first, payoff second:"]
            for (first, second), combo in sorted(
                    COMBO_CATALOG.items(), key=lambda kv: kv[1].name):
                costs = (f"{int(combo.hp_cost_pct * 100)}% HP, "
                         f"{combo.mana_cost} mana, +{combo.extra_cd} cd")
                lines.append(
                    f"  🔥 {combo.name}\n"
                    f"     {first} → {second}\n"
                    f"     {combo.desc}\n"
                    f"     costs: {costs}")
            lines.append("cast with: combo <first> + <second>")
            return "\n".join(lines)
        if verb not in ("learn", "upgrade") or len(parts) < 2:
            return ("usage: /skill  ·  /skill learn <name>  ·  "
                    "/skill upgrade <name>  ·  /skill combos")
        defn = resolve_skill(parts[1])
        if defn is None:
            return (f"no such skill {parts[1]!r} — /skill lists the "
                    f"school.")
        if verb == "learn":
            if defn.slug in learned:
                return f"you already know {defn.name}."
            if level < defn.level_req:
                return (f"{defn.name} needs level {defn.level_req} — "
                        f"you're level {level}.")
            if prof.coins < defn.cost:
                return (f"{defn.name} costs {defn.cost}c — "
                        f"you have {prof.coins}c.")
            new_balance = store.spend_coins(player, defn.cost,
                                            f"skill:{defn.slug}")
            if new_balance is None:
                return "couldn't take the coins — try again."
            if not skills.learn(player.key, defn.slug):
                # charged but not recorded — refund, never lose coins
                store.add_coins(player, defn.cost,
                                f"skill-refund:{defn.slug}")
                return "learning failed — coins refunded."
            # the legacy recognizes its heir
            title_note = ""
            if defn.slug == "slaying_force":
                try:
                    from ...games.titles import TitleStore
                    if TitleStore(engine.db).unlock(player.key,
                                                    "cutyps_heir"):
                        title_note = " 👑 title unlocked: Cutyp's Heir!"
                except Exception:  # noqa: BLE001
                    pass
            up = (" ⬆️ upgradeable — /skill upgrade later"
                  if max_tier(defn) > 1 else "")
            use = ("cast it in the arena with "
                   f"skill {defn.slug}." if defn.kind == "active"
                   else "it's passive — always on in the arena.")
            return (f"🥋 learned {defn.name}! ({defn.school} school){up} — "
                    f"{new_balance}c left. {use}{title_note}")
        # ── upgrade ──
        if defn.slug not in learned:
            return (f"you haven't learned {defn.name} yet — "
                    f"/skill learn {defn.slug} first.")
        cur = tiers.get(defn.slug, 1)
        top = max_tier(defn)
        if cur >= top:
            return f"{defn.name} is already at max tier."
        up = defn.tiers[cur - 1]
        nxt_name = tier_name(defn, cur + 1)
        if level < up.level_req:
            return (f"{nxt_name} needs level {up.level_req} — "
                    f"you're level {level}.")
        if prof.coins < up.cost:
            return (f"{nxt_name} costs {up.cost}c to forge — "
                    f"you have {prof.coins}c.")
        new_balance = store.spend_coins(player, up.cost,
                                        f"skill-up:{defn.slug}:{cur + 1}")
        if new_balance is None:
            return "couldn't take the coins — try again."
        if not skills.upgrade(player.key, defn.slug):
            store.add_coins(player, up.cost,
                            f"skill-up-refund:{defn.slug}")
            return "the upgrade failed — coins refunded."
        more = (" — maxed out! 🏆" if cur + 1 >= top
                else " — more power waits at the next tier.")
        return (f"⬆️ {defn.name} → {nxt_name}! {up.desc} "
                f"({new_balance}c left){more}")

    def _control_titles(self, tail: str, *, player: Any = None) -> str:
        """Earnable titles: /title | /title set <name>.

        Titles unlock through arena achievements and milestones.  The
        active title is worn next to your name in the arena.
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        from ...games.titles import TITLE_CATALOG, TitleStore
        engine = self._game_engine()
        store = TitleStore(engine.db)
        # refresh unlocks — achievements earned since last check
        new = store.check_unlocks(player.key)
        tail = (tail or "").strip()
        lines: list[str] = []
        for name in new:
            lines.append(f"👑 new title unlocked: {name}!")
        if not tail or tail.lower() in ("list", "show"):
            cmap = {t.id: t for t in TITLE_CATALOG}
            unlocked = store.unlocked(player.key)
            active = store.active(player.key)
            lines.append(f"👑 {player.name}'s titles")
            for tid in unlocked:
                t = cmap.get(tid)
                if not t:
                    continue
                mark = " ⭐ active" if t.name == active else ""
                lines.append(f"  ✅ {t.name}{mark} — {t.desc}")
            locked = [t for t in TITLE_CATALOG if t.id not in unlocked]
            if locked:
                lines.append("still to earn:")
                for t in locked:
                    lines.append(f"  🔒 {t.name} — {t.desc}")
            lines.append("wear one: /title set <name>")
            return "\n".join(lines)
        parts = tail.split(None, 1)
        if parts[0].lower() not in ("set", "wear", "use") or len(parts) < 2:
            return "usage: /title  ·  /title set <name>"
        want = parts[1].strip().lower()
        cmap = {t.id: t for t in TITLE_CATALOG}
        match = None
        for t in TITLE_CATALOG:
            if want == t.id or want == t.name.lower() \
                    or want in t.name.lower():
                match = t
                break
        if match is None:
            return f"no such title {parts[1]!r} — /title lists them."
        if not store.set_active(player.key, match.id):
            return (f"{match.name} isn't unlocked yet — {match.desc}")
        return f"👑 you now fight as {match.name} {player.name}."

    def _control_stats(self, tail: str, *, player: Any = None) -> str:
        """RPG attributes: /stats | /stats <attr> [points].

        Each level grants attribute points — spend them on strength
        (attack), stamina (HP/defense), mana (skill fuel), or
        intelligence (skill power + combo luck).
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        from ...games.stats import (STAT_NAMES, StatStore, describe_stats,
                                    gear_stat_bonuses)
        engine = self._game_engine()
        store = StatStore(engine.db)
        # Display-only: points are granted at the actual level-up event
        # in the game engine, never here. This just shows the current
        # state.
        stats = store.get(player.key)
        tail = (tail or "").strip().lower()
        lines: list[str] = []
        if not tail or tail in ("show", "list"):
            gear_bonus = {n: 0 for n in STAT_NAMES}
            try:
                from ...games.gear import GearStore
                gs = GearStore(engine.db)
                loadout = gs.loadout(player.key)
                gear_bonus = gear_stat_bonuses(loadout)
            except Exception:  # noqa: BLE001
                pass
            lines.append(f"📊 {player.name}'s attributes")
            lines.append(describe_stats(stats, gear_bonus))
            lines.append("spend: /stats <strength|stamina|mana|intelligence> "
                         "[points]")
            return "\n".join(lines)
        parts = tail.split()
        attr = parts[0]
        pts = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        ok, msg = store.spend(player.key, attr, pts, stats=stats)
        return ("📊 " if ok else "") + msg

    def _control_gift(self, tail: str, *, player: Any = None) -> str:
        """Player-to-player gifting: /gift @name <coins|gear|item>.

        Two-step with confirmation so nobody fat-fingers their myth
        katana away:

            /gift @ada 100            → preview a 100-coin gift
            /gift @ada katana_rare    → preview a gear transfer
            /gift @ada potion         → preview a shop-item gift
            /gift confirm             → execute the pending gift
            /gift cancel              → drop it
            /gift history             → recent gifts sent/received
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        from ...games.gifting import GiftStore, resolve_recipient
        from ...games.players import Player
        engine = self._game_engine()
        store = GiftStore(engine.db)

        tail = (tail or "").strip()
        low = tail.lower()

        if low in ("", "help"):
            pending = store.pending_for(player.key)
            lines = [
                "🎁 gifting — send coins, gear, or items to another player:",
                "  /gift @name 100          — gift 100 coins",
                "  /gift @name katana_rare  — gift a gear piece you own",
                "  /gift @name potion       — gift a shop item you own",
                "  /gift confirm | /gift cancel",
                "  /gift history            — recent gifts",
            ]
            if pending is not None:
                lines.append(
                    f"\n⏳ pending: {pending.describe()} — "
                    f"/gift confirm or /gift cancel")
            return "\n".join(lines)

        if low == "history":
            hist = store.history(player.key)
            if not hist:
                return "no gifts yet — be the first to give. /gift @name 100"
            lines = ["🎁 recent gifts:"]
            for g in hist[:10]:
                arrow = "→" if g.giver_key == player.key else "←"
                who = (g.recipient_name if g.giver_key == player.key
                       else g.giver_name)
                lines.append(f"  {arrow} {g.describe()} ({who})")
            return "\n".join(lines)

        if low == "cancel":
            dropped = store.cancel(player.key)
            if dropped is None:
                return "nothing pending — /gift @name 100 to start one."
            return f"cancelled: {dropped.describe()}. nothing moved."

        if low == "confirm":
            gift = store.pending_for(player.key)
            if gift is None:
                return ("nothing pending (gifts expire after 5 minutes) — "
                        "/gift @name 100 to start one.")
            return self._execute_gift(engine, store, gift, player)

        # /gift @name <what>
        parts = tail.split(None, 2)
        if len(parts) < 2 or not parts[0].startswith("@"):
            return ("usage: /gift @name <coins|gear|item>\n"
                    "  /gift @ada 100 · /gift @ada katana_rare · "
                    "/gift @ada potion")
        target_label, what = parts[0], parts[1].strip()
        if target_label.lstrip("@").lower() == (player.name or "").lower():
            return "gifting yourself? bold. (no — pick someone else.)"

        prof, err = resolve_recipient(engine.store, target_label)
        if prof is None:
            return err
        if prof.key == player.key:
            return "that's you — pick someone else."
        recipient = Player(key=prof.key, platform=prof.platform or "local",
                           name=prof.name or target_label.lstrip("@"))

        # coins: a bare number
        if what.replace(",", "").isdigit():
            amount = int(what.replace(",", ""))
            if amount <= 0:
                return "gift a positive amount of coins."
            if amount > 1_000_000:
                return "that's a lot — keep gifts under 1,000,000 coins."
            bal = engine.store.get(player.key).coins
            if bal < amount:
                return (f"you have {bal} coins — not enough for {amount}. "
                        f"win games to earn more.")
            gift = store.create(player.key, player.name, recipient.key,
                                recipient.name, "coins", str(amount),
                                amount=amount)
            return (f"🎁 preview: send {amount} coins to {recipient.name}?\n"
                    f"you'll have {bal - amount} left.\n"
                    f"/gift confirm to send · /gift cancel to drop "
                    f"(expires in 5 min)")

        slug = what.lower()
        # gear?
        from ...games.gear import GEAR_CATALOG
        if slug in GEAR_CATALOG:
            from ...games.gear import GearStore
            gs = GearStore(engine.db)
            inst = gs.find(player.key, slug)
            if inst is None:
                return (f"you don't own {slug!r} — /inventory to see "
                        f"your gear.")
            gift = store.create(player.key, player.name, recipient.key,
                                recipient.name, "gear", inst.slug)
            return (f"🎁 preview: send {inst.display_name()} "
                    f"({inst.slug}) to {recipient.name}?\n"
                    f"it will be unequipped and leave your inventory.\n"
                    f"/gift confirm to send · /gift cancel to drop "
                    f"(expires in 5 min)")

        # shop item?
        count = engine.economy.count(player, slug)
        if count > 0:
            gift = store.create(player.key, player.name, recipient.key,
                                recipient.name, "item", slug)
            item = engine.economy._items.get(slug)
            label = item.name if item else slug
            return (f"🎁 preview: send {label} ×1 to {recipient.name}?\n"
                    f"you'll have {count - 1} left.\n"
                    f"/gift confirm to send · /gift cancel to drop "
                    f"(expires in 5 min)")

        return (f"can't gift {what!r} — not coins, not gear you own, not "
                f"an item you own. /inventory and /game balance to check.")

    def _execute_gift(self, engine: Any, store: Any, gift: Any,
                      player: Any) -> str:
        """Run a confirmed gift. Atomic where the stores allow it; the
        giver is refunded on any mid-transfer failure."""
        from ...games.players import Player
        recipient = Player(key=gift.recipient_key,
                           platform=(gift.recipient_key.partition(":")[0]
                                     or "local"),
                           name=gift.recipient_name)
        giver = Player(key=player.key, platform=player.platform,
                       name=player.name)

        if gift.kind == "coins":
            new_bal = engine.store.spend_coins(
                giver, gift.amount, f"gift:out:{gift.id}")
            if new_bal is None:
                store.cancel(player.key)
                return ("couldn't send — your balance changed. gift "
                        "cancelled.")
            try:
                engine.store.add_coins(
                    recipient, gift.amount,
                    f"gift:in:{gift.id}:{giver.name}")
            except Exception as exc:  # noqa: BLE001
                engine.store.add_coins(
                    giver, gift.amount, f"gift:refund:{gift.id}")
                store.cancel(player.key)
                return f"transfer failed ({exc}) — refunded."
            store.mark_done(gift.id)
            self._notify_gift(recipient, gift, player)
            return (f"🎁 sent {gift.amount} coins to {gift.recipient_name}! "
                    f"({new_bal} left)")

        if gift.kind == "gear":
            from ...games.gear import GearStore
            gs = GearStore(engine.db)
            inst = gs.find(player.key, gift.ref)
            if inst is None:
                store.cancel(player.key)
                return ("that gear is gone (sold? broken?) — gift "
                        "cancelled.")
            ok, name = gs.transfer(inst.id, recipient.key)
            if not ok:
                store.cancel(player.key)
                return f"transfer failed: {name} — gift cancelled."
            store.mark_done(gift.id)
            self._notify_gift(recipient, gift, player)
            return (f"🎁 sent {name} to {gift.recipient_name}! "
                    f"it's in their inventory now.")

        # item
        if not engine.store.consume_item(giver, gift.ref):
            store.cancel(player.key)
            return "that item is gone — gift cancelled."
        try:
            engine.store.grant_item(recipient, gift.ref)
        except Exception as exc:  # noqa: BLE001
            engine.store.grant_item(giver, gift.ref)  # refund
            store.cancel(player.key)
            return f"transfer failed ({exc}) — refunded."
        store.mark_done(gift.id)
        self._notify_gift(recipient, gift, player)
        item = engine.economy._items.get(gift.ref)
        label = item.name if item else gift.ref
        return f"🎁 sent {label} to {gift.recipient_name}!"

    def _notify_gift(self, recipient: Any, gift: Any, giver: Any) -> None:
        """Best-effort DM to the recipient: 'you got a gift!'.

        Recipient chat key == player key for DMs (platform:sender).
        Never raises — a missed ping must not break the gift.
        """
        try:
            chat_key = gift.recipient_key
            if gift.kind == "coins":
                what = f"{gift.amount} coins"
            elif gift.kind == "gear":
                what = gift.ref
            else:
                what = gift.ref
            self._relay_send(
                chat_key,
                f"🎁 {giver.name} sent you a gift: {what}!\n"
                f"check /game balance or /inventory.")
        except Exception:  # noqa: BLE001
            pass

    def _control_daily(self, *, player: Any = None) -> str:
        """The daily hunt: /daily shows today's double-XP status."""
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        from ...games.daily import daily_hunt_status
        engine = self._game_engine()
        return daily_hunt_status(engine.db, player.key)

    def _control_mastery(self, tail: str, *, player: Any = None) -> str:
        """Per-game mastery tiers: /mastery | /mastery <game>.

        Every non-arena game tracks a mastery rank (Novice → Legend,
        with game-specific names like sudoku's Solver → Grandmaster)
        from cumulative wins, games played, and best score.  Tiers gate
        unlockables — harder sudoku boards, bigger gomoku boards, the
        2048 marathon, long hangman words, trivia sudden death.
        """
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here — run this from the chat where you play."
        from ...games.mastery import (describe_game_mastery,
                                      describe_mastery)
        engine = self._game_engine()
        want = (tail or "").strip().lower()
        if want:
            game = engine.games.get(want)
            if game is None:
                return (f"no game {want!r} — /mastery alone lists your "
                        f"tiers.")
            return describe_game_mastery(engine.db, player.key, game.name)
        return describe_mastery(engine.db, player.key, player.name)

    def _is_relay_move(self, relay_room: Any, text: str) -> bool:
        """Does this message belong to the relayed game?

        Slash engine commands (/pass /status /help …) always go
        through. Plain text only goes through when the relay's game
        recognizes it as a move attempt — casual chat falls through
        to the normal conversation flow instead of triggering
        "waiting on X" spam.
        """
        t = (text or "").strip()
        if not t:
            return False
        if t.startswith("/"):
            from ...games.games.base import parse_command
            cmd, _rest = parse_command(t)
            # /game is the control plane; anything else slash-prefixed
            # is an engine command worth forwarding (the engine itself
            # ignores what it doesn't understand)
            return cmd not in ("", "game")
        try:
            engine = self._game_engine()
            game = engine.games.get(relay_room.game_name)
            if game is not None:
                return bool(game.is_move_text(t))
        except Exception:  # noqa: BLE001 - never break chat on a lookup
            pass
        return True

    def _route_game_move(self, chat_key: str, text: str, *,
                         player: Any = None, kind: str = "dm") -> str | None:
        """While a game is live in this chat, plain messages are game moves.

        The multi-player engine owns every game (42 and counting, every
        platform). Relay rooms (DM-to-DM multiplayer) are checked first.
        """
        try:
            from ..features import feature_enabled

            if not text:
                return None
            if not feature_enabled(self.context, "games"):
                return None
            
            # Check for relay room first (DM-to-DM multiplayer)
            relay = self._game_relay()
            relay_room = relay.get_relay_for_chat(chat_key)
            if relay_room is not None and player is not None:
                # Only game moves and engine commands go through the
                # relay. Casual chat ("say that again?") falls through
                # to normal conversation — otherwise every message
                # while it's not your turn spams "waiting on X".
                if not self._is_relay_move(relay_room, text):
                    return None
                # route through the virtual room; the reply goes back to
                # this chat via the normal path, the opponent gets theirs
                # on their own platform
                msgs = relay.relay_move(chat_key, text, player)
                if not msgs:
                    return None
                reply = "\n".join(msgs)
                self._relay_send(relay_room.other_chat(chat_key), reply)
                return reply
            
            engine = self._game_engine()
            room = engine.live(chat_key)
            if room is not None:
                # A live room owns plain messages (moves) and the engine's
                # own commands (/status /pass /shop /leave …).  /game is the
                # control plane, and any OTHER slash command (/mood, /status
                # the bot, …) falls through to the normal control/brain flow.
                if text.startswith("/"):
                    from ...games.games.base import parse_command

                    cmd, _rest = parse_command(text)
                    if cmd in ("", "game"):
                        return None  # not an engine command — let it pass
                    is_engine_command = True
                else:
                    is_engine_command = False
                if player is None:
                    return None
                msgs: list[str] = []
                # a group auto-seats a new human the moment they SPEAK —
                # engine commands don't seat, they just work
                if (room.kind == "group" and room.player(player.key) is None
                        and not is_engine_command):
                    msgs.extend(engine.join(chat_key, player))
                msgs.extend(engine.move(chat_key, text, player, kind=kind))
                return "\n".join(msgs) or None
            return None
        except Exception:  # noqa: BLE001 - a game bug must never eat the chat
            _log.exception("game move failed")
            return None

    # ── games ────────────────────────────────────────────────────────────────
    def _game_engine(self) -> Any:
        """The multi-player GameEngine — ONE instance serves every platform
        at once. It never touches a platform: it sees chat keys, sender
        keys and a send callback, and the gateway feeds it inbound text."""
        engine = getattr(self, "_game_engine_obj", None)
        if engine is None:
            from ...games.engine import GameEngine

            engine = GameEngine(
                self.context,
                send=self._game_send,
                suggest=self._game_suggest(),
            )
            self._game_engine_obj = engine
        return engine

    def _game_relay(self) -> Any:
        """The game relay system — one per engine, so the ticker reaps
        expired invites and every platform shares the same relay table."""
        return self._game_engine().relay

    def _relay_send(self, chat_key: str, text: str) -> bool:
        """Send to a relay chat on its own platform. Chat keys are
        ``platform:chat_id`` — never assume telegram."""
        platform = (chat_key or "").partition(":")[0].strip() or "telegram"
        try:
            result = self.gateway.send(platform, chat_key, text)
            return bool(getattr(result, "ok", False))
        except Exception:  # noqa: BLE001
            _log.warning("relay send failed for %s", chat_key,
                         exc_info=True)
            return False

    @staticmethod
    def _resolve_invite_target(inviter_chat_key: str,
                               to_label: str) -> str | None:
        """Best-effort chat key for a direct invite DM on the inviter's
        own platform. Returns None when the label can't be resolved
        (then the invite code is the delivery mechanism)."""
        label = (to_label or "").strip()
        if not label:
            return None
        platform = (inviter_chat_key or "").partition(":")[0].strip()
        if platform == "telegram":
            return f"telegram:{label.lstrip('@')}"
        if platform == "whatsapp":
            digits = "".join(c for c in label if c.isdigit())
            # a phone number → DM chat; the adapter normalizes the JID
            if len(digits) >= 7:
                return f"whatsapp:{digits}"
            return None
        if platform == "discord":
            digits = "".join(c for c in label if c.isdigit())
            if len(digits) >= 5:
                return f"discord:{digits}"
            return None
        return None

    def _game_suggest(self) -> Callable[[str], str] | None:
        """Optional LLM brain for the game AI — the same router the legacy
        bridge used. Without one the AI plays seeded deterministic moves,
        which is fully functional on its own."""
        router = getattr(self.context, "router", None)
        if router is None:
            return None

        def _call(prompt: str) -> str:
            from ...llm.base import Message

            response = router.chat([Message(role="user", content=prompt)])
            return getattr(response, "text", "") or ""

        return _call

    def _game_send(self, chat_key: str, text: str) -> None:
        """Engine output → the same chat it came from, on any platform.

        Virtual relay rooms (``relay:<code>``) fan out to BOTH players'
        real chats — otherwise scheduler emissions (turn timeouts, idle
        expiry, finishes) would vanish into a chat key no platform
        owns."""
        if (chat_key or "").startswith("relay:"):
            try:
                relay = self._game_engine().relay
                room = relay.get_relay_for_virtual(chat_key)
            except Exception:  # noqa: BLE001
                room = None
            if room is not None:
                for other in (room.chat_a, room.chat_b):
                    self._relay_send(other, text)
                return
        ref = ChatRef.parse(chat_key)
        for chunk in self._game_chunks(text):
            try:
                result = self.gateway.send(ref.platform, ref, chunk)
                if not result.ok:
                    _log.warning("game send failed in %s: %s",
                                 chat_key, result.error)
            except Exception:  # noqa: BLE001
                _log.exception("game send failed in %s", chat_key)

    @staticmethod
    def _game_chunks(text: str, limit: int = 3800) -> list[str]:
        """Split long game reports at newlines for platform limits."""
        text = text or ""
        if not text:
            return []
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        while len(text) > limit:
            cut = text.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        if text:
            chunks.append(text)
        return chunks

    @staticmethod
    def _game_player(message: Any) -> Any:
        from ...games.players import Player

        sender = (message.sender or "").strip() or "unknown"
        sender_id = (getattr(message, "sender_id", "") or "").strip()
        if sender_id:
            # Stable platform user id: one human, one game identity —
            # display-name changes and the telegram/telegram-bot endpoint
            # split can no longer fork profiles.  The display name rides
            # along as the label (name-authority rules in PlayerStore).
            return Player.from_sender(message.chat.platform, sender_id, sender)
        return Player.from_sender(message.chat.platform, sender, sender)

    @staticmethod
    def _game_player_for_key(chat_key: str) -> Any:
        """A stable player for console-style calls that only carry a key."""
        from ...games.players import Player

        try:
            ref = ChatRef.parse(chat_key)
        except Exception:  # noqa: BLE001
            ref = ChatRef(platform="local", chat_id="console")
        return Player.from_sender(ref.platform, ref.chat_id, ref.chat_id[:40])

    @staticmethod
    @staticmethod
    def _profile_line(prof: Any, display_name: str = "") -> str:
        items = ", ".join(f"{k}×{v}" for k, v in (prof.items or {}).items()) or "none"
        games = prof.games_played or 0
        pct = round(100 * prof.wins / games) if games else 0
        if prof.streak > 0:
            streak = f"win streak {prof.streak}"
        elif prof.streak < 0:
            streak = f"loss streak {-prof.streak}"
        else:
            streak = "no streak"
        # Prefer the live sender name (from the current message) over the
        # stored DB name, which can go stale after merges — this keeps
        # /game stats consistent with /inventory and /skill.
        name = (display_name or "").strip() or prof.name or prof.key
        return (f"📊 {name} — {prof.wins}W {prof.losses}L "
                f"{prof.draws}D · {pct}% · {streak}\n"
                f"   {games} games · {prof.coins} coins · {prof.points} points "
                f"· items: {items}")

    # ── arena ────────────────────────────────────────────────────────────────
    def _control_arena(self, tail: str, chat_key: str, *,
                       player: Any = None, kind: str = "dm") -> str:
        from ..arena import Arena
        from ..features import feature_enabled

        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        # ── PvP verbs: duel challenges + raid boss ────────────────────
        # (the rest of this handler is the self-improvement arena)
        if verb == "challenge":
            if not feature_enabled(self.context, "games"):
                return "games are off. /features games on"
            who = parts[1] if len(parts) > 1 else ""
            if not who:
                return ("usage: /arena challenge @user — challenge "
                        "someone to a 1v1 duel. they accept with "
                        "/game accept <code> (or /arena accept <code>).")
            return self._challenge_player("pvp", who, chat_key, player)
        if verb == "raid":
            if not feature_enabled(self.context, "games"):
                return "games are off. /features games on"
            if player is None:
                return "raid needs a chat sender — run it where you play."
            engine = self._game_engine()
            try:
                room, msgs = engine.start(chat_key, "raid", player,
                                          kind=kind)
            except ValueError as exc:
                return str(exc)
            body = "\n".join(msgs) or engine.describe(room)
            return f"🎮 {body}"
        if verb == "accept":
            if not feature_enabled(self.context, "games"):
                return "games are off. /features games on"
            if player is None:
                return ("accept needs a chat sender — paste the /arena "
                        "accept <code> in the chat where you play.")
            if len(parts) < 2:
                return "usage: /arena accept <invite_code>"
            code = parts[1]
            try:
                relay = self._game_relay()
                relay_room = relay.accept_invite(code, chat_key, player)
                self._relay_send(
                    relay_room.chat_a,
                    f"🎮 {player.name} accepted your "
                    f"{relay_room.game_name} invite — game on! play in "
                    f"your DM, moves are relayed.")
                return ("game started! play here in this chat — your moves "
                        "are relayed to your opponent.")
            except ValueError as exc:
                return str(exc)
        # ── self-improvement arena below ──────────────────────────────
        # Reuse the runtime's arena (loop state lives per instance).
        arena = getattr(self, "_arena", None) or Arena(self.context)
        if verb == "stats":
            s = arena.stats()
            builds = ", ".join(f"{k}={v}" for k, v in s["builds"].items())
            last = time.strftime("%m-%d %H:%M", time.localtime(s["last_cycle"])) if s["last_cycle"] else "never"
            interval = arena.interval_hours()
            interval_txt = f"{interval:g}h" if interval else "config default"
            return (f"arena stats — cycles: {s['cycles']} · knowledge rows: {s['knowledge']} · "
                    f"builds: {builds or '0'}\n"
                    f"last cycle: {last} · loop interval: {interval_txt}")
        if verb == "digest":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            rows = arena.digests(n)
            if not rows:
                return "no digests yet — run a cycle first (/arena run)"
            lines = []
            for r in rows:
                when = time.strftime("%m-%d %H:%M", time.localtime(r.get("created_at", 0)))
                lines.append(f"── {when} · {r['topic']} ({r.get('category', '?')})\n{str(r.get('digest', ''))[:900]}")
            text = "\n\n".join(lines)
            if len(text) > 1800:
                chat = self._ref_from_key(chat_key)
                return self._send_long_checked(chat.platform, chat, text)
            return text
        if verb == "schedule":
            if len(parts) > 1 and parts[1].replace(".", "", 1).isdigit():
                hours = float(parts[1])
                if not arena.set_interval(hours):
                    return ("couldn't save the interval (the store refused the "
                            "write) — try again, or /arena status to check "
                            "the loop is alive")
                return f"loop interval set to {hours:g}h (live — the running loop picks it up next cycle)"
            live = arena.interval_hours()
            live_txt = f"{live:g}h" if live else "config default"
            return f"loop interval: {live_txt} — /arena schedule <hours> to change"
        if verb == "loop":
            mode = parts[1].lower() if len(parts) > 1 else "status"
            if mode == "on":
                if not feature_enabled(self.context, "arena"):
                    return "loop blocked: feature off — /features arena on first"
                from ..power import power_mode_for
                if not power_mode_for(self.context).active:
                    return "loop blocked: power mode is locked (the loop is a power-mode capability)"
                if arena.start_loop():
                    persistent = "" if getattr(self, "_arena", None) is arena else (
                        "\n(note: this session only — add NM_ARENA_ENABLED=1 to ~/.nomorals/.env "
                        "to auto-start it on every boot)"
                    )
                    live = arena.interval_hours()
                    live_txt = f"{live:g}h" if live else "config default"
                    return (f"background loop running — interval {live_txt}, "
                            f"it researches + digests on its own{persistent}")
                return "loop is already running"
            if mode == "off":
                arena.stop_loop()
                return "background loop stopped"
            state = "running" if arena.loop_running() else "idle"
            live = arena.interval_hours()
            live_txt = f"{live:g}h" if live else "config default"
            return f"loop: {state} (interval {live_txt})"
        if verb == "topic":
            sub = parts[1].lower() if len(parts) > 1 else "list"
            if sub == "add":
                text = " ".join(parts[2:]).strip()
                if not text:
                    return "usage: /arena topic add <topic>"
                return arena.add_topic(text)
            custom = arena.custom_topics()
            if not custom:
                return "topic bank is empty — /arena topic add <topic> to add one"
            return f"topic bank ({len(custom)}): " + " · ".join(custom[:15])
        if verb == "builds":
            rows = arena.all_builds(20)
            if not rows:
                return "no arena builds yet (power mode + /features arena on + arena.build config)"
            lines = ["arena builds:"]
            for r in rows:
                when = time.strftime("%m-%d %H:%M", time.localtime(r.get("created_at", 0)))
                lines.append(f"  {r['id']} [{r['status']}] {r.get('name', '?')} — {str(r.get('purpose', ''))[:60]} ({when})")
            return "\n".join(lines)
        if verb == "status":
            pending = arena.builds()
            stream = arena.stream(1)
            loop = "loop: running" if arena.loop_running() else "loop: idle"
            lines = [f"arena — {loop}"]
            lines.append(f"feature: {'on' if feature_enabled(self.context, 'arena') else 'off'} "
                         f"(/features arena on|off)")
            lines.append(f"knowledge rows: {self._arena_knowledge_count()}")
            if pending:
                lines.append("pending builds:")
                for row in pending[:5]:
                    lines.append(f"  {row['id']} — {row.get('name')} ({row.get('purpose', '')[:50]})")
            else:
                lines.append("pending builds: none")
            if stream:
                last = stream[0]
                lines.append(f"last event: {last['kind']} {str(last.get('payload', {}).get('topic', ''))[:50]}")
            return "\n".join(lines)
        if verb == "run":
            if not feature_enabled(self.context, "arena"):
                return "arena is off. /features arena on"
            topic = " ".join(parts[1:]).strip() or None
            chat = self._ref_from_key(chat_key)

            def _notify(text: str) -> None:
                try:
                    self.gateway.send(chat.platform, chat, text)
                except Exception:  # noqa: BLE001
                    pass

            try:
                self.gateway.send(chat.platform, chat,
                                  f"⏳ arena cycle starting{': ' + topic[:60] if topic else ''}…")
            except Exception:  # noqa: BLE001
                pass
            result = arena.run_cycle(topic=topic, notify=_notify)
            if not result.get("ok"):
                return f"arena cycle failed: {result.get('error')}"
            build = result.get("build")
            build_line = (f"build {build['id']} sent for review." if build
                          else "no build this cycle (power mode + arena.build required).")
            return (f"arena cycle done: “{result['topic']}” — digested into knowledge "
                    f"in {result.get('seconds', 0)}s. {build_line}")
        if verb == "topics":
            from ..arena.topics import category_table, topics_table

            if tail and tail.strip().lower() not in {"", "all"}:
                return category_table(tail.strip().lower()[:40])
            try:
                profile = arena.interest_profile()
            except Exception:  # noqa: BLE001
                profile = None
            return topics_table(profile=profile, db=self.context.db)
        if verb == "surprise":
            if not feature_enabled(self.context, "arena"):
                return "arena is off. /features arena on"
            seed = None
            if tail and tail.strip().lstrip("-").isdigit():
                seed = int(tail.strip())
            chat = self._ref_from_key(chat_key)

            def _notify2(text: str) -> None:
                try:
                    self.gateway.send(chat.platform, chat, text)
                except Exception:  # noqa: BLE001
                    pass

            result = arena.run_cycle(notify=_notify2, surprise=True, seed=seed)
            if not result.get("ok"):
                return f"surprise cycle failed: {result.get('error')}"
            return (f"🎲 surprise cycle done: “{result['topic']}” "
                    f"[{result.get('category')}] — digested into knowledge "
                    f"in {result.get('seconds', 0)}s.")
        if verb == "stream":
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            kind_filter = parts[1].lower() if len(parts) > 1 and not parts[1].isdigit() else None
            events = arena.stream(limit if not kind_filter else limit * 10)
            if kind_filter:
                events = [e for e in events if e.get("kind") == kind_filter][:limit]
            if not events:
                return "arena stream is empty — run a cycle first (/arena run)."
            lines = [f"arena stream (last {len(events)}):"]
            for ev in events:
                payload = ev.get("payload") or {}
                detail = (payload.get("topic") or payload.get("name")
                          or payload.get("error") or ev.get("kind", ""))
                when = time.strftime("%m-%d %H:%M", time.localtime(ev.get("ts", 0)))
                lines.append(f"  {when}  {ev.get('kind', '?'):<8} {str(detail)[:70]}")
            return "\n".join(lines)
        if verb == "export":
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 200
            text = arena.export(limit)
            if not text.strip():
                return "arena knowledge is empty — run some cycles first."
            try:
                path = self.context.settings.home_path / f"arena_export_{int(time.time())}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n", "utf-8")
            except Exception as exc:  # noqa: BLE001
                return f"export failed: {exc}"
            return (f"exported {len(text.splitlines())} training rows → {path}\n"
                    "that's the live stream, in the export format — ready for the Colab training run.")
        if verb == "ship":
            from ..arena.ship import ship_queue

            try:
                queue = ship_queue(self.context.db, self.context)
            except Exception as exc:  # noqa: BLE001
                return f"ship queue failed: {exc}"
            if not queue:
                return ("ship queue is empty — promote a build first "
                        "(/arena promote <build-id>)")
            lines = ["arena ship queue:"]
            for q in queue:
                files = q["edits"]
                shown = ", ".join(files[:4])
                if len(files) > 4:
                    shown += f" (+{len(files) - 4} more)"
                lines.append(f"  {q['id']} [{q['status']}] "
                             f"{q['instruction'][:60]} :: {shown}")
            return "\n".join(lines)
        if verb == "promote":
            if len(parts) < 2:
                return "usage: /arena promote <build-id>"
            from ..arena.ship import promote_build
            from ..evolution import EvolutionAgent, _REPO_ROOT

            try:
                pid = promote_build(self.context.db, self.context, parts[1],
                                    _REPO_ROOT)
                prop = EvolutionAgent(self.context)._load(pid)
                files = [e["path"] for e in (prop.edits if prop else [])
                         if isinstance(e, dict)]
            except Exception as exc:  # noqa: BLE001
                return f"promote failed: {exc}"
            shown = ", ".join(files[:6])
            if len(files) > 6:
                shown += f" (+{len(files) - 6} more)"
            return (f"promoted build {parts[1]} → proposal {pid} "
                    f"({len(files)} files: {shown})\n"
                    f"verify with /arena apply {pid}")
        if verb == "apply":
            if len(parts) < 2:
                return "usage: /arena apply <proposal-id>"
            from ..arena.ship import approve_ship
            from ..evolution import _REPO_ROOT

            try:
                result = approve_ship(self.context.db, self.context, parts[1],
                                      _REPO_ROOT, commit=True, full_suite=False)
            except Exception as exc:  # noqa: BLE001
                return f"apply failed: {exc}"
            if result.get("applied"):
                files = ", ".join(result.get("edits", []))
                return (f"shipped {result['proposal']} → {files} "
                        f"(commit {result.get('commit', 'n/a')})")
            return (f"not applied: {result.get('reason', 'unknown')} — "
                    f"{str(result.get('report', ''))[:400]}")
        if verb == "reject":
            if len(parts) < 3:
                return "usage: /arena reject <proposal-id> <reason>"
            from ..arena.ship import deny_ship

            if deny_ship(self.context.db, self.context, parts[1],
                         " ".join(parts[2:])):
                return f"rejected {parts[1]}"
            return f"no proposal {parts[1]!r}"
        if verb == "scores":
            try:
                from ..arena import scoring
            except ImportError:
                return ("arena scoring isn't available — the arena_scores "
                        "table exists (migration 61) but no scoring module "
                        "could be imported; nothing to show")
            try:
                table = scoring.category_scores(self.context.db)
            except Exception as exc:  # noqa: BLE001
                return f"scores failed: {exc}"
            if not table:
                return ("no arena scores recorded yet — the scoring worker "
                        "logs to arena_scores after each build")
            lines = ["arena category scores:"]
            for cat in sorted(table):
                s = table[cat]
                avg = (f"{s['avg']:.2f}" if s.get("avg") is not None
                       else "n/a")
                lat = (f"{s['avg_latency']:.1f}s"
                       if s.get("avg_latency") is not None else "n/a")
                lines.append(f"  {cat:<16} runs={s['runs']:<4} avg={avg:<6} "
                             f"latency={lat}")
            return "\n".join(lines)
        if verb == "sample":
            from ..arena.sampling import sample_challenge

            try:
                profile = arena.interest_profile()
            except Exception:  # noqa: BLE001
                profile = None
            # db=None: a dry run — sample_challenge never touches the
            # database then, so no anti-repeat state is written and the
            # topic bank stays untouched.
            cat, topic, entry = sample_challenge(db=None, profile=profile,
                                                 anti_repeat=0)
            return (f"arena sample (preview — nothing recorded):\n"
                    f"  category:   {cat}\n"
                    f"  difficulty: {entry.get('d', '?')}\n"
                    f"  kind:       {entry.get('kind', '?')}\n"
                    f"  topic:      {topic}\n"
                    f"  verify:     {str(entry.get('verify', ''))[:120]}")
        if verb in {"approve", "deny"}:
            if len(parts) < 2:
                return f"usage: /arena {verb} <build-id>"
            return arena.approve(parts[1]) if verb == "approve" else arena.deny(parts[1])
        # An unknown verb is a topic: /arena hacking == /arena run hacking.
        if (tail or "").strip():
            return self._control_arena(f"run {tail.strip()}", chat_key=chat_key)
        return ("usage: /arena [status|run [topic]|surprise [seed]|stats|digest [n]|schedule [h]|loop on|off|"
                "topic add <t>|builds|topics [category]|stream [kind|n]|export [n]|approve <id>|deny <id>|"
                "promote <build-id>|ship|apply <proposal-id>|reject <proposal-id> <reason>|scores|sample]")

    def _arena_knowledge_count(self) -> int:
        try:
            row = self.context.db.query_one("SELECT COUNT(*) AS n FROM arena_knowledge")
            return int(row.get("n", 0)) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    # ── trial accounts ───────────────────────────────────────────────────────
    def _control_trial(self, tail: str, chat_key: str) -> str:
        from ..trial import TrialFlow

        flow = TrialFlow(self.context, gateway=getattr(self, "gateway", None))
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "list"
        if verb == "list":
            return flow.list()
        if verb == "start":
            if len(parts) < 2:
                return "usage: /trial start <platform>"
            return flow.start(" ".join(parts[1:]))
        if verb == "assist":
            if len(parts) < 2:
                return "usage: /trial assist <platform>"
            return flow.assist(" ".join(parts[1:]), chat_key=chat_key)
        if verb == "status":
            return flow.assist_status()
        if verb == "resume":
            if len(parts) < 2:
                return "usage: /trial resume <checkpoint-id>"
            return flow.resume(parts[1], chat_key=chat_key)
        if verb == "sms":
            rest = parts[1:]
            if rest and rest[0].lower() == "code":
                return flow.temp_sms_code_async(chat_key)
            country = rest[0] if rest else "us"
            return flow.temp_number(country)
        if verb == "inbox":
            if len(parts) < 2:
                return ("usage: /trial inbox <service>  "
                        "(e.g. /trial inbox email_mailtm)")
            return flow.disposable_inbox(parts[1])
        if verb == "save":
            if len(parts) < 4:
                return "usage: /trial save <platform> <login> <password>"
            platform, login = parts[1], parts[2]
            secret = " ".join(parts[3:])
            flow.save(platform, login, secret)
            try:
                sent = flow.deliver(platform, gateway=self.gateway, via=chat_key)
            except Exception as exc:  # noqa: BLE001
                sent = f"stored, but delivery failed: {exc}"
            return f"stored the {platform.lower()} trial account (encrypted, local only). {sent}"
        if verb == "send":
            if len(parts) < 2:
                return "usage: /trial send <platform>"
            return flow.deliver(parts[1], gateway=self.gateway, via=chat_key)
        if verb == "rm":
            if len(parts) < 2:
                return "usage: /trial rm <platform>"
            return flow.remove(parts[1])
        return ("usage: /trial [list|start <p>|assist <p>|status|resume <id>|sms [country]|"
                "sms code|inbox <service>|save <p> <login> <pass>|send <p>|rm <p>]")

    # ── sports bet analyst: /bet (analysis only — never places bets) ─────────
    def _control_bet(self, tail: str, chat_key: str) -> str:
        """The ensemble-ML sports bet analyst.

        /bet analyze <home> vs <away> [h d a] [--league L]
        /bet analyze [league] [--top N]  — auto-fetch upcoming fixtures
        /bet bankroll [set <amount>]
        /bet backtest [n] [--seed S]
        /bet record <home> <away> <hg>-<ag> [--league L]
        """
        from ..sports_bet import (BetStore, Fixture, OddsSnapshot, backtest,
                                 render_analysis, render_backtest,
                                 synthetic_history)

        store = BetStore()
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        def _usage() -> str:
            return (
                "/bet analyze <home> vs <away> [home_odds draw_odds away_odds] "
                "[--league L]\n"
                "/bet analyze [league] [--top N] — fetch upcoming fixtures "
                "from the majors & analyze the best ones\n"
                "/bet bankroll [set <amount>]  — the paper bankroll\n"
                "/bet backtest [n] [--seed S]  — walk-forward backtest on "
                "synthetic history\n"
                "/bet record <home> <away> <hg>-<ag> [--league L]  — feed a "
                "result back in\n"
                f"bankroll: {store.bankroll:.2f}")

        if verb in ("", "help"):
            return _usage()

        if verb == "bankroll":
            if rest.lower().startswith("set"):
                try:
                    amt = float(rest.split()[1])
                except (IndexError, ValueError):
                    return "usage: /bet bankroll set <amount>"
                store.bankroll = amt
                store.save()
                return f"bankroll set to {amt:.2f}"
            return f"paper bankroll: {store.bankroll:.2f}"

        if verb == "backtest":
            n = 400
            seed = 7
            toks = rest.split()
            i = 0
            while i < len(toks):
                t = toks[i]
                if t == "--seed" and i + 1 < len(toks):
                    try:
                        seed = int(toks[i + 1])
                    except ValueError:  # noqa: E103 - unparseable --seed keeps default 7
                        pass
                    i += 2
                    continue
                if t.startswith("--seed="):
                    try:
                        seed = int(t.split("=", 1)[1])
                    except ValueError:  # noqa: E103 - unparseable --seed keeps default 7
                        pass
                elif t.isdigit():
                    n = int(t)
                i += 1
            entries = synthetic_history(n=n, seed=seed)
            r = backtest(entries, bankroll=store.bankroll, seed=seed)
            return render_backtest(r)

        if verb == "record":
            toks = rest.split()
            if len(toks) < 3:
                return "usage: /bet record <home> <away> <hg>-<ag> [--league L]"
            home, away = toks[0], toks[1]
            try:
                hg_s, ag_s = toks[2].split("-")
                hg, ag = int(hg_s), int(ag_s)
            except ValueError:
                return "usage: /bet record <home> <away> <hg>-<ag> [--league L]"
            league = "GEN"
            if "--league" in toks:
                try:
                    league = toks[toks.index("--league") + 1]
                except IndexError:  # noqa: E103 - missing --league value keeps default GEN
                    pass
            store.record(Fixture(home=home, away=away, league=league,
                                 home_goals=hg, away_goals=ag))
            elo_h = store.analyst.elo.rating(home)
            elo_a = store.analyst.elo.rating(away)
            return (f"recorded: {home} {hg}-{ag} {away}  "
                    f"(elo {elo_h:.0f} / {elo_a:.0f})")

        if verb == "analyze":
            from ..sports_bet import fixture_digest
            toks = rest.split()
            league_raw, toks = _pop_opt(toks, "--league")
            top_raw, toks = _pop_opt(toks, "--top", "--n")
            league = league_raw or "GEN"
            try:
                top_n = max(1, min(10, int(top_raw))) if top_raw else 3
            except (TypeError, ValueError):
                top_n = 3
            rest = " ".join(toks)
            import re as _re
            m = _re.split(r"\s+vs\.?\s+", rest, maxsplit=1, flags=_re.I)
            if len(m) < 2:
                # fixture mode: no "<home> vs <away>" given
                text = rest.strip()
                if text:
                    # maybe a bare league alias, e.g. "/bet analyze epl"
                    from ..sports_bet import EspnFetcher
                    if not EspnFetcher.resolve_league(text):
                        return ("usage: /bet analyze <home> vs <away> "
                                "[home_odds draw_odds away_odds]\n"
                                "   or: /bet analyze [league] [--top N] "
                                "(fetches upcoming fixtures)")
                    league = text
                return fixture_digest(store, league_text=league, top_n=top_n)
            home = m[0].strip()
            tail2 = m[1].strip().split()
            away_parts: list = []
            odds: list = []
            for t in tail2:
                try:
                    odds.append(float(t))
                except ValueError:
                    if odds:
                        break  # odds started, team name done
                    away_parts.append(t)
                if len(odds) == 3:
                    break
            away = " ".join(away_parts).strip()
            if not home or not away:
                return ("usage: /bet analyze <home> vs <away> "
                        "[home_odds draw_odds away_odds]")
            snaps = []
            if len(odds) == 3:
                snaps = [OddsSnapshot(bookmaker="chat", home=odds[0],
                                      draw=odds[1], away=odds[2])]
            elif odds:
                return "give all three odds (home draw away) or none"
            a = store.analyst.analyze(
                home, away, league=league, odds=snaps,
                fixtures=store.fixtures())
            return render_analysis(a)

        return f"unknown /bet verb {verb!r}\n{_usage()}"
