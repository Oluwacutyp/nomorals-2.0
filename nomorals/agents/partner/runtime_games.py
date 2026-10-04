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
            lines.append("  /game invite <game> [who] · /game accept <code> — DM duels")
            return "\n".join(lines)
        if verb == "quit":
            # capture the relay (if any) BEFORE quitting — the engine
            # tears it down, and the opponent deserves to hear about it
            relay = self._game_relay()
            doomed = relay.get_relay_for_chat(chat_key)
            out = engine.quit(chat_key)
            if doomed is not None:
                who = player.name if player is not None else "your opponent"
                self._relay_send(
                    doomed.other_chat(chat_key),
                    f"🏁 {who} closed the {doomed.game_name} duel.")
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
                return self._profile_line(engine.store.get(player.key))
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
                return "invite needs a chat sender."
            if len(parts) < 2:
                return ("usage: /game invite <game_name> [who]\n"
                        "the code is the invite — your friend accepts with "
                        "/game accept <code> from any chat.")
            game_name = parts[1].lower()
            to_label = parts[2] if len(parts) > 2 else ""
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
        if verb == "accept":
            if player is None:
                return "accept needs a chat sender."
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
                return ("game started! play here in this chat — your moves "
                        "are relayed to your opponent.")
            except ValueError as exc:
                return str(exc)
        if verb in engine.games:
            if player is None:
                return "start a game from a chat — I need to know who's at the table."
            # "/hangman daily" — same word for everyone, all day
            # "/game case timed" — countdown mode with a speed bonus
            # "/game connect4 hard" — AI/puzzle difficulty
            words = [p.lower() for p in parts[1:]]
            daily = "daily" in words
            timed = "timed" in words
            difficulty = next(
                (w for w in words
                 if w in ("easy", "normal", "hard", "expert")),
                "normal")
            try:
                room, msgs = engine.start(chat_key, verb, player, kind=kind,
                                          daily=daily, timed=timed,
                                          difficulty=difficulty)
            except ValueError as exc:
                return str(exc)
            if msgs and not msgs[0].startswith("🎮"):
                msgs[0] = f"🎮 {msgs[0]}"  # the table-opening banner
            return "\n".join(msgs) or engine.describe(room)
        return (f"unknown game {verb!r} — /game list to see the table.")

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
            return "no player here."
        engine = self._game_engine()
        store, gear = engine.store, engine.gear
        prof = store.get(player.key)
        ref = (ref or "").strip().lower()

        def _piece_line(inst: Any) -> str:
            from ...games.gear import (GEAR_CATALOG, durability_bar,
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
                    f"{durability_bar(inst.durability, inst.max_durability)}"
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
            # msg is "DisplayName|cost" — deduct coins, then apply
            try:
                name, cost_s = msg.rsplit("|", 1)
                cost = int(cost_s)
            except ValueError:
                return msg
            if prof.coins < cost:
                return (f"repairing {name} costs {cost}c — "
                        f"you have {prof.coins}c.")
            inst = gear.find(player.key, ref)
            if inst is None:
                return "lost track of that piece — try again."
            store.add_coins(player, -cost, f"repair:{inst.slug}")
            if gear.apply_repair(inst.id):
                return f"🔧 {name} — good as new, {cost}c."
            # repair failed after taking coins — refund, never lose coins
            store.add_coins(player, cost, f"repair-refund:{inst.slug}")
            return f"repair failed — {cost}c refunded."

        return f"unknown gear command {cmd!r}."

    # ── progression ──────────────────────────────────────────────────────────
    def _control_level(self, *, player: Any = None) -> str:
        """Show persistent progression: level, XP bar, stat growth."""
        from ..features import feature_enabled
        if not feature_enabled(self.context, "games"):
            return "games are off. /features games on"
        if player is None:
            return "no player here."
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
        """Engine output → the same chat it came from, on any platform."""
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
    def _profile_line(prof: Any) -> str:
        items = ", ".join(f"{k}×{v}" for k, v in (prof.items or {}).items()) or "none"
        games = prof.games_played or 0
        pct = round(100 * prof.wins / games) if games else 0
        if prof.streak > 0:
            streak = f"win streak {prof.streak}"
        elif prof.streak < 0:
            streak = f"loss streak {-prof.streak}"
        else:
            streak = "no streak"
        return (f"📊 {prof.name or prof.key} — {prof.wins}W {prof.losses}L "
                f"{prof.draws}D · {pct}% · {streak}\n"
                f"   {games} games · {prof.coins} coins · {prof.points} points "
                f"· items: {items}")

    # ── arena ────────────────────────────────────────────────────────────────
    def _control_arena(self, tail: str, chat_key: str) -> str:
        from ..arena import Arena
        from ..features import feature_enabled

        # Reuse the runtime's arena (loop state lives per instance).
        arena = getattr(self, "_arena", None) or Arena(self.context)
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
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
                    return "could not save the interval"
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

        flow = TrialFlow(self.context)
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
            return flow.assist(" ".join(parts[1:]))
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
        return "usage: /trial [list|start <p>|assist <p>|save <p> <login> <pass>|send <p>|rm <p>]"

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
