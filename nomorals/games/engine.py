"""The game engine: rooms, turns, timers, persistence, rewards.

One :class:`GameEngine` serves every platform at once — Telegram,
WhatsApp, Discord, the local console — because it never touches a
platform: it takes a normalized chat key, a sender key, and a send
callback, and produces messages.  The chat gateway feeds it inbound
text; the engine's output goes back out the same chat it came from.

Concurrency model: rooms are mutable state guarded by one engine lock —
a move is a handful of dict ops, the lock is held for microseconds, and
the scheduler thread never does rule work under the lock (it only
*selects* rooms whose clocks ran out, then moves them like any other
move).  The scheduler thread is weakref-bound to its engine so a
disposed engine can never leak a ticker (the same discipline as the
virtual CPU farm).
"""
from __future__ import annotations

import inspect
import threading
import time
import weakref
from typing import Any, Callable

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .ai import GameMind
from .economy import GameEconomy, ShopItem
from .games.base import GAME_COMMANDS, MultiGame, Room, parse_command
from .players import AI_PLAYER, Leaderboard, Player, PlayerStore

__all__ = ["GameEngine", "SendFn"]

_log = get_logger(__name__)

SendFn = Callable[[str, str], None]  # (chat_key, text) -> None

#: A live room with no human activity (move/join/leave) for this long is
#: closed by the scheduler and its table freed. Covers games with
#: ``move_timeout = 0`` (no per-turn clock at all) and abandoned tables
#: whose per-turn timeouts would otherwise ping-pong forever.
IDLE_ROOM_TTL = 3600.0


def _new_state_accepts_kwargs(game: MultiGame) -> bool:
    """True when the game's new_state takes **kw (hangman: the daily flag)."""
    try:
        params = inspect.signature(game.new_state).parameters
    except (TypeError, ValueError):
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _new_state_kwargs(game: MultiGame, **kw: Any) -> dict[str, Any]:
    """Keep only the kwargs a game's ``new_state`` will accept.

    Extra flags (e.g. ``timed``, ``history``) are opt-in: games whose
    ``new_state`` takes ``**kw`` get everything; the rest only get the
    named parameters they declare. Everyone else plays as always.
    """
    try:
        params = inspect.signature(game.new_state).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in params.values()):
        return dict(kw)
    named = {name for name, p in params.items()
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.KEYWORD_ONLY)}
    return {k: v for k, v in kw.items() if k in named}


def _scheduler_worker(engine_ref: "weakref.ref") -> None:
    """Ticker thread target. Holds only a weakref + the wake event
    (neither references the engine), so a dead engine's thread dies at
    the next wake instead of outliving its owner."""
    while True:
        engine = engine_ref()
        if engine is None:
            return
        event = engine._wake
        interval = 1.0
        engine = None  # release before the wait
        if event.wait(timeout=interval):
            event.clear()
        engine = engine_ref()
        if engine is None or engine._stopping:
            return
        try:
            engine._sweep_timeouts()
        except Exception:  # noqa: BLE001 - the ticker must never die
            _log.debug("game sweep failed", exc_info=True)


class GameEngine:
    """The table host for every game in every chat, on every platform."""

    def __init__(self, context: Any, *, send: SendFn | None = None,
                 suggest: Callable[[str], str] | None = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self._send = send
        self._suggest = suggest
        self._lock = threading.RLock()
        self._rooms: dict[str, Room] = {}        # chat_key -> live room
        self._by_id: dict[str, Room] = {}
        self._mind = GameMind(suggest=suggest)
        self.store = PlayerStore(self.db)
        from .gear import GearStore
        self.gear = GearStore(self.db)
        self.economy = GameEconomy(self.store, gear_store=self.gear)
        self.board = Leaderboard(self.store)
        self.games: dict[str, MultiGame] = {}
        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._relay_obj: Any = None  # lazy GameRelay (see relay property)
        self._last_game: dict[str, tuple[str, list[Player], str]] = {}
        self._register_builtins()
        self._start_scheduler()

    @property
    def relay(self) -> Any:
        """The DM-to-DM game relay, created on first use. Lives on the
        engine so the ticker can reap expired invites/relays and so every
        platform shares one relay table."""
        relay = self._relay_obj
        if relay is None:
            from .relay import GameRelay  # lazy: relay imports engine
            relay = GameRelay(self)
            self._relay_obj = relay
        return relay

    # ── game registry ──────────────────────────────────────────────────────
    def _register_builtins(self) -> None:
        from .games.easy import EASY_GAMES
        from .games.medium import MEDIUM_GAMES
        from .games.ambitious import AMBITIOUS_GAMES
        from .games.wild import WILD_GAMES
        from .games.arcade import ARCADE_GAMES
        from .games.casino import CASINO_GAMES
        from .games.inbox import INBOX_GAMES
        from .games.puzzles import PUZZLE_GAMES
        from .games.pvp import PVP_GAMES
        for game in (*EASY_GAMES, *MEDIUM_GAMES, *AMBITIOUS_GAMES,
                     *WILD_GAMES, *ARCADE_GAMES, *CASINO_GAMES,
                     *INBOX_GAMES, *PUZZLE_GAMES, *PVP_GAMES):
            self.games[game.name] = game

    def register(self, game: MultiGame) -> None:
        self.games[game.name] = game

    # ── scheduler ──────────────────────────────────────────────────────────
    def _start_scheduler(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=_scheduler_worker, args=(weakref.ref(self),),
            name="nm-game-turns", daemon=True)
        self._stopping = False
        self._thread.start()

    def shutdown(self) -> None:
        self._stopping = True
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def _sweep_timeouts(self) -> None:
        now = time.time()
        due: list[Room] = []
        idle: list[Room] = []
        with self._lock:
            for room in self._rooms.values():
                if room.status != "active":
                    continue
                # idle expiry runs regardless of the per-turn clock —
                # games with move_timeout=0 never time out otherwise.
                # inbox games (one message per turn, days between moves)
                # bring their own longer TTL.
                game = self.games.get(room.game)
                ttl = IDLE_ROOM_TTL
                if game is not None and game.idle_ttl:
                    ttl = game.idle_ttl
                if now - room.last_activity > ttl:
                    idle.append(room)
                    continue
                if (game is None or game.move_timeout <= 0):
                    continue
                cur = room.current
                if (cur is not None and not cur.is_ai
                        and now - room.turn_started > game.move_timeout):
                    due.append(room)
        for room in due:
            try:
                self._handle_timeout(room)
            except Exception:  # noqa: BLE001
                _log.exception("game timeout failed: %s", room.game)
        for room in idle:
            try:
                self._finish_idle(room)
            except Exception:  # noqa: BLE001
                _log.exception("game idle-expiry failed: %s", room.game)
        # reap expired invites / long-idle relay rooms (throttled inside)
        try:
            relay = self._relay_obj
            if relay is not None:
                relay.maybe_cleanup()
        except Exception:  # noqa: BLE001
            _log.debug("relay cleanup failed", exc_info=True)
        self._maybe_prune_history(now)

    def _maybe_prune_history(self, now: float) -> None:
        """Throttled: drop finished room rows older than 30 days.

        Finished rows are write-only history — nothing reads them back
        (``_load_live`` only ever wants 'active'), so letting them pile
        up forever is just leaked DB weight.
        """
        if self.db is None:
            return
        last = getattr(self, "_last_prune", 0.0)
        if now - last < 3600:
            return
        self._last_prune = now
        try:
            self.db.execute(
                "DELETE FROM game_rooms WHERE status = 'finished' "
                "AND updated_at < ?", (now - 30 * 86400,))
        except Exception:  # noqa: BLE001
            _log.debug("game room history prune failed", exc_info=True)

    # ── sending ────────────────────────────────────────────────────────────
    def _emit(self, room: Room, *texts: str) -> None:
        for text in texts:
            if not text:
                continue
            room.messages.append(text[:400])
            room.messages = room.messages[-40:]
            if self._send is not None:
                try:
                    self._send(room.chat_key, text)
                except Exception:  # noqa: BLE001 - a send failure never
                    _log.debug("game send failed", exc_info=True)
            else:
                print(text)  # CLI/standalone: visible output

    # ── persistence ───────────────────────────────────────────────────────
    def _persist(self, room: Room) -> None:
        if self.db is None:
            return
        # serialize under the room guard: a concurrent move mutating
        # state mid-dumps would otherwise raise or persist a torn room
        with room.guard:
            self._persist_inner(room)

    def _persist_inner(self, room: Room) -> None:
        try:
            import json as _json
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO game_rooms (id, game, chat_key, platform, kind, "
                    "players, turn, state, status, started_at, seed, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET turn = excluded.turn, "
                    "state = excluded.state, status = excluded.status, "
                    "players = excluded.players, updated_at = excluded.updated_at",
                    (room.id, room.game, room.chat_key, room.platform,
                     room.kind, _json.dumps([p.key for p in room.players]),
                     room.turn, _json.dumps(room.state), room.status,
                     room.started_at, room.seed, time.time()),
                )
        except Exception:  # noqa: BLE001
            _log.debug("game room persist failed", exc_info=True)

    def _load_live(self, chat_key: str) -> Room | None:
        """Restore a room that survived a restart (mid-game reboot)."""
        if self.db is None:
            return None
        try:
            row = self.db.query_one(
                "SELECT * FROM game_rooms WHERE chat_key = ? AND status = 'active' "
                "ORDER BY updated_at DESC LIMIT 1", (chat_key,)
            )
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        room = Room.from_row(row, {})
        self._rooms[room.chat_key] = room
        self._by_id[room.id] = room
        return room

    def _delete_room_row(self, room_id: str) -> None:
        """Hard-delete one room's DB row.

        Used when a start aborts mid-setup: the room was already
        persisted as 'active' but never made it into the live tables, so
        without this the next ``live()`` would resurrect it as a zombie.
        """
        if self.db is None:
            return
        try:
            self.db.execute("DELETE FROM game_rooms WHERE id = ?",
                            (room_id,))
        except Exception:  # noqa: BLE001
            _log.debug("game room row delete failed", exc_info=True)

    # ── room lifecycle ─────────────────────────────────────────────────────
    def live(self, chat_key: str) -> Room | None:
        with self._lock:
            room = self._rooms.get(chat_key)
            if room is not None and room.status == "active":
                return room
            # DB restore happens under the same lock so two concurrent
            # lookups can't resurrect the same room twice
            return self._load_live(chat_key)

    def start(self, chat_key: str, game_name: str,
              host: Player, *, kind: str = "dm",
              platform: str = "", daily: bool = False,
              timed: bool = False,
              difficulty: str = "normal",
              variant: str = "") -> tuple[Room, list[str]]:
        """Open a room and seat the host (plus AI seats).

        Returns (room, messages_to_send). Raises ValueError when the
        game is unknown or the chat kind doesn't fit (group-only game in
        a DM, …). ``difficulty`` is only handed to games that declare a
        named ``difficulty`` parameter on ``new_state``. ``variant`` is a
        game-specific mode word (gomoku "big"/"huge", hangman "long",
        trivia "sudden", 2048 "big") — only games that declare it get
        it. ``mastery`` (the host's current tier index for this game)
        lets games gate their unlockables.
        """
        name = (game_name or "").strip().lower()
        game = self.games.get(name)
        if game is None:
            raise ValueError(
                f"unknown game {name!r}. try: {', '.join(sorted(self.games))}")
        platform = platform or chat_key.split(":", 1)[0]
        if game.needs_group and kind == "dm":
            raise ValueError(
                f"{game.name} needs a group — start it in a group chat "
                f"(/game {game.name} there).")
        with self._lock:
            existing = self._rooms.get(chat_key)
            if existing is not None and existing.status == "active":
                raise ValueError(
                    f"a {existing.game} is already live here — /game quit "
                    "first.")
            room = Room(
                id=new_id(), game=game.name, chat_key=chat_key,
                platform=platform, kind=kind,
            )
            room.players = [host]
            self._fill_ai(room, game)
            # daily/timed/difficulty/variant/history/mastery are opt-in:
            # only handed to games whose new_state accepts kwargs
            # (hangman: daily; case: history + timed; connect4/reversi/…:
            # difficulty; gomoku: variant + mastery). The rest play as
            # always.
            history = None
            load = getattr(game, "load_history", None)
            if load is not None and _new_state_accepts_kwargs(game):
                try:
                    history = load(self.store, host)
                except Exception:  # noqa: BLE001
                    _log.debug("history load failed", exc_info=True)
            # the host's mastery tier gates game-specific unlockables
            # (harder sudoku boards, bigger gomoku boards, …)
            try:
                from .mastery import get_one_game_stats, mastery_tier
                mastery_idx = mastery_tier(
                    name, get_one_game_stats(self.db, host.key, name))[1]
            except Exception:  # noqa: BLE001
                mastery_idx = 0
            room.state = game.new_state(
                game.rng(room),
                **_new_state_kwargs(
                    game, daily=daily, timed=timed,
                    difficulty=difficulty, variant=variant,
                    mastery=mastery_idx, history=history))
            # stash difficulty on the room so start/status messages can show it
            room.state["_difficulty"] = difficulty
            self._mirror_inventory(room)
            self._rooms[chat_key] = room
            self._by_id[room.id] = room
        msgs: list[str] = []
        try:
            intro = game.setup(room, self._mind)
            if intro:
                msgs.append(intro)
            room.status = "active"
            room.turn_started = time.time()
            room.last_activity = time.time()
            self._persist(room)
            self._pump_ai(room, msgs)
            self._persist(room)
        except Exception:  # noqa: BLE001 - a bad setup must not orphan a room
            with self._lock:
                self._rooms.pop(chat_key, None)
                self._by_id.pop(room.id, None)
            # the first _persist already wrote an 'active' row — kill it
            # too, or the next live() resurrects this stillborn table
            self._delete_room_row(room.id)
            raise
        # a fresh table supersedes the rematch memory — the old finished
        # game is no longer the one to run back
        with self._lock:
            self._last_game.pop(chat_key, None)
        return room, msgs

    def _mirror_inventory(self, room: Room) -> None:
        """Games that spend real gear (arena, escape) read each human's
        items from ``state['inventory'][player.key]`` — a snapshot taken
        at setup so game code stays pure (no store access inside a
        move). The engine reconciles the actual ledger when the room
        closes, consuming whatever the state says was used.

        Durable gear is mirrored separately into ``state['loadout']``:
        slot → {id, slug, atk, def, durability, set} for what's equipped.
        """
        for p in room.humans:
            self._mirror_player(room, p)

    def _mirror_player(self, room: Room, player: Player) -> None:
        """Mirror ONE player's inventory/gear/progression/skills into
        ``room.state``. Additive: only this player's keys are written,
        so a mid-game join (raid reinforcements, duel challengers) can
        never clobber another player's snapshot."""
        key = player.key
        try:
            room.state.setdefault("inventory", {})[key] = dict(
                self.store.get(key).items)
        except Exception:  # noqa: BLE001
            room.state.setdefault("inventory", {}).setdefault(key, {})
        try:
            from .gear import GEAR_CATALOG, effective_stats
            worn: dict[str, dict[str, Any]] = {}
            owned: list[dict[str, Any]] = []
            for inst in self.gear.list(key):
                defn = GEAR_CATALOG.get(inst.slug)
                if defn is None:
                    continue
                atk, df = effective_stats(defn)
                entry = {"id": inst.id, "slug": inst.slug,
                         "name": defn.name, "slot": defn.slot,
                         "grade": defn.grade,
                         "atk": atk, "def": df,
                         "durability": inst.durability,
                         "max_durability": inst.max_durability,
                         "set": defn.set_name,
                         "unbreakable": bool(defn.unbreakable),
                         "equipped": inst.equipped}
                owned.append(entry)
                if inst.equipped and not inst.broken:
                    worn.setdefault(defn.slot, entry)
            room.state.setdefault("loadout", {})[key] = worn
            room.state.setdefault("gear_closet", {})[key] = owned
        except Exception:  # noqa: BLE001
            room.state.setdefault("loadout", {}).setdefault(key, {})
            room.state.setdefault("gear_closet", {}).setdefault(key, [])
        # persistent progression: level + arena stat bonus per human,
        # so games apply it store-free (see progression.level_stat_bonus)
        try:
            from .progression import level_for_xp, level_stat_bonus
            prof = self.store.get(key)
            level = level_for_xp(prof.xp)
            room.state.setdefault("progression", {})[key] = {
                "level": level, **level_stat_bonus(level)}
        except Exception:  # noqa: BLE001
            _log.debug("progression mirror failed", exc_info=True)
        # learned battle skills, so combat games apply them store-free
        try:
            from .skills import SkillStore
            store = SkillStore(self.db)
            room.state.setdefault("skills", {})[key] = store.learned(key)
            # skill upgrade tiers, so battles use the fighting stats
            room.state.setdefault("skill_tiers", {})[key] = \
                store.tiers(key)
        except Exception:  # noqa: BLE001
            _log.debug("skills mirror failed", exc_info=True)
        # earned title, so the arena can wear it next to your name
        try:
            from .titles import active_title
            room.state.setdefault("titles", {})[key] = active_title(
                self.db, key)
        except Exception:  # noqa: BLE001
            _log.debug("title mirror failed", exc_info=True)
        # RPG attributes, so combat games apply them store-free
        try:
            from .stats import StatStore
            store = StatStore(self.db)
            stats = store.get(key)
            # Safety net: top up any level-up points missed due to
            # transient failures at the level-up event. Idempotent —
            # never double-grants.
            try:
                from .progression import level_for_xp
                prof = self.store.get(key)
                _, stats = store.grant_level_points(
                    key, level_for_xp(prof.xp))
            except Exception:  # noqa: BLE001
                pass
            room.state.setdefault("rpg_stats", {})[key] = stats.to_dict()
        except Exception:  # noqa: BLE001
            _log.debug("rpg stats mirror failed", exc_info=True)

    def _reconcile_items(self, room: Room) -> None:
        consumed = room.state.get("consumed", {})
        for p in room.players:
            if p.is_ai:
                continue
            used = consumed.get(p.key) or consumed.get("you") or {}
            for item, n in used.items():
                for _ in range(max(0, int(n))):
                    self.economy.consume(p, item)
        # durable gear: apply the wear the battle recorded, in the same
        # pass — breakage only ever zeroes durability, the row survives
        # for repair.
        wear = room.state.get("gear_wear", {})
        for p in room.players:
            if p.is_ai:
                continue
            used = wear.get(p.key) or wear.get("you") or {}
            for instance_id, n in used.items():
                for _ in range(max(0, int(n))):
                    self.gear.wear(instance_id, 1)
        # gear equipped mid-battle persists as the new loadout
        equipped_final = room.state.get("gear_equipped", {})
        for p in room.players:
            if p.is_ai:
                continue
            final = equipped_final.get(p.key) or {}
            if final:
                self.gear.sync_equipped(p.key, final)

    def _fill_ai(self, room: Room, game: MultiGame) -> None:
        """Seat enough AI players for the game to work. Channel rooms
        are house-played: all seats are AI, humans spectate."""
        humans_wanted = 0
        if room.kind == "channel" and game.channel_mode == "house":
            pass  # all AI
        elif room.kind == "dm":
            humans_wanted = 1
        # Calculate total slots needed, accounting for humans already seated
        humans_present = len(room.humans)
        ai_needed = max(0, game.min_players - humans_present)
        # But don't exceed max_players total
        ai_needed = min(ai_needed, game.max_players - humans_present)
        # Also ensure we have at least game.ai_seats AI players
        ai_needed = max(ai_needed, game.ai_seats)
        
        have = len(room.ai_seats)
        for i in range(ai_needed - have):
            room.players.append(Player(
                key=f"{AI_PLAYER}:{room.game}:{i}", platform="ai",
                name=f"House {i + 1}", is_ai=True))

    def join(self, chat_key: str, player: Player) -> list[str]:
        """A human joins a live room (groups). Returns messages."""
        with self._lock:
            room = self._rooms.get(chat_key)
            if room is None or room.status != "active":
                return ["no game is live here — start one with /game <name>."]
            game = self.games.get(room.game)
            if room.player(player.key) is not None:
                return [f"{player.name} is already at the table."]
        with room.guard:
            if room.status != "active":
                return ["no game is live here — start one with /game <name>."]
            return self._join_inner(room, game, chat_key, player)

    def _join_inner(self, room: Room, game: MultiGame, chat_key: str,
                    player: Player) -> list[str]:
        """The body of :meth:`join`. Caller holds ``room.guard``."""
        replaced_ai = None
        if len(room.players) >= game.max_players:
            # Try to replace an AI player
            ai_players = [p for p in room.players if p.is_ai]
            if ai_players:
                # Remove the first AI player to make room
                replaced_ai = ai_players[0]
                room.players = [p for p in room.players if p.key != replaced_ai.key]
                # Adjust turn if needed
                if room.turn >= len(room.players):
                    room.turn = 0
                room.players.append(player)
            else:
                return ["the table is full."]
        else:
            room.players.append(player)
        # the joiner needs their own snapshots (inventory/gear/skills/
        # progression) before the game's on_join builds their seat —
        # additive, so nobody else's mid-game state is touched
        self._mirror_player(room, player)
        msgs: list[str] = []
        note = game.on_join(room, player, self._mind)
        if note:
            msgs.append(note)
        else:
            if replaced_ai:
                msgs.append(f"{player.name} sits down (replacing {replaced_ai.name}).")
            else:
                msgs.append(f"{player.name} sits down.")
        room.last_activity = time.time()
        self._persist(room)
        self._emit(room, *msgs)
        return msgs

    def leave(self, chat_key: str, player: Player) -> list[str]:
        with self._lock:
            room = self._rooms.get(chat_key)
            if room is None or room.status != "active":
                return []
            if room.player(player.key) is None:
                return [f"{player.name} isn't in this game."]
            if room.player(player.key).is_ai:
                return []
            game = self.games.get(room.game)
        with room.guard:
            if room.status != "active":
                return []
            return self._leave_inner(room, game, chat_key, player)

    def _leave_inner(self, room: Room, game: MultiGame | None, chat_key: str,
                     player: Player) -> list[str]:
        """The body of :meth:`leave`. Caller holds ``room.guard``."""
        notice = game.on_leave(room, player, self._mind) if game else None
        room.players = [p for p in room.players if p.key != player.key]
        if room.turn >= len(room.players):
            room.turn = 0 % max(1, len(room.players))
        # is_over: a game may end itself in on_leave (duel walkover) —
        # honor that instead of leaving a decided table open
        if (not room.humans or len(room.players) <= len(room.ai_seats)
                or self.is_over(room)):
            return self._finish(room, notice)
        room.turn_started = time.time()
        room.last_activity = time.time()
        self._persist(room)
        msgs = []
        if notice:
            msgs.append(notice)
        self._emit(room, *msgs)
        self._pump_ai(room, [])
        self._persist(room)
        return msgs

    def move(self, chat_key: str, text: str, sender: Player,
             *, kind: str = "dm") -> list[str]:
        """Route one inbound message at a live room.

        Handles engine commands (/pass /status /shop …), turn
        enforcement (only the current seat moves; in a DM the single
        human always may), game moves, and AI pumping afterwards.
        """
        with self._lock:
            room = self._rooms.get(chat_key)
            if room is None or room.status != "active":
                return []
            game = self.games.get(room.game)
            if game is None:
                return []
            room.last_activity = time.time()
        # Serialize per room: two inbound messages for the same chat must
        # not interleave a move with a quit/timeout, nor double-apply a
        # turn. Lock order is always engine → room; here we hold only the
        # room guard (the game callbacks may call the model — the global
        # lock must never be held across that).
        with room.guard:
            if room.status != "active":
                return []  # quit/timeout closed it while we waited
            return self._move_inner(room, game, chat_key, text, sender)

    def _move_inner(self, room: Room, game: MultiGame, chat_key: str,
                    text: str, sender: Player) -> list[str]:
        """The body of :meth:`move`. Caller holds ``room.guard``."""
        # channel spectator mode: humans don't move, they watch
        if room.kind == "channel" and game.channel_mode == "house" \
                and not sender.is_ai:
            return []
        cmd, rest = parse_command(text)
        if cmd == "pass" or cmd == "skip":
            if not self._is_turn(room, sender):
                return [f"it's not your turn — waiting on "
                        f"{room.current.name if room.current else '…'}."]
            room.state["passes"] = int(room.state.get("passes") or 0) + 1
            self._emit(room, f"{sender.name} passes.")
            room.advance_turn()
            out: list[str] = []
            self._pump_ai(room, out)
            if self.is_over(room):
                out.extend(self._finish(room))
            else:
                self._persist(room)
            return out or []
        if cmd == "status":
            return [self.describe(room)]
        if cmd == "help":
            return [f"📜 {game.name} rules:\n{game.rules or game.description}"]
        if cmd == "shop":
            if rest.startswith("buy "):
                ok, msg = self.economy.purchase(sender, rest[4:].strip())
                if ok:
                    # refresh this player's mirrored inventory: games that
                    # spend items mid-match (sudoku/cryptogram hints) read the
                    # snapshot, so a purchase must land there immediately.
                    # consumption is still reconciled from
                    # state["consumed"] when the room closes.
                    try:
                        inv = room.state.setdefault("inventory", {})
                        inv[sender.key] = dict(
                            self.store.get(sender.key).items)
                        # same for the gear closet (arena equips mid-fight)
                        from .gear import GEAR_CATALOG, effective_stats
                        owned: list[dict[str, Any]] = []
                        for inst in self.gear.list(sender.key):
                            defn = GEAR_CATALOG.get(inst.slug)
                            if defn is None:
                                continue
                            atk, df = effective_stats(defn)
                            owned.append(
                                {"id": inst.id, "slug": inst.slug,
                                 "name": defn.name, "slot": defn.slot,
                                 "atk": atk, "def": df,
                                 "durability": inst.durability,
                                 "max_durability": inst.max_durability,
                                 "set": defn.set_name,
                                 "unbreakable": bool(defn.unbreakable),
                                 "equipped": inst.equipped})
                        room.state.setdefault("gear_closet", {})[sender.key] \
                            = owned
                    except Exception:  # noqa: BLE001
                        _log.debug("inventory mirror refresh failed",
                                   exc_info=True)
                return [msg]
            return [self.economy.catalog_text(game.name, sender)]
        if cmd == "balance":
            prof = self.store.get(sender.key)
            items = ", ".join(f"{k}×{v}" for k, v in prof.items.items()) or "none"
            return [f"🪙 {prof.coins} coins · {prof.points} points · items: {items}"]
        if cmd == "leave":
            return self.leave(chat_key, sender)
        if cmd == "game":
            return []  # control commands are the runtime's, not the game's

        # an actual move — only the current seat (or the solo human in a DM)
        if not self._is_turn(room, sender):
            cur = room.current
            if room.kind == "group" and cur is not None and not cur.is_ai:
                return [f"waiting on {cur.name} — your move goes in after "
                        "theirs. (/pass /status /help)"]
            return []
        out: list[str] = []
        try:
            out.extend(game.on_move(room, sender, text, self._mind))
        except Exception as exc:  # noqa: BLE001
            # surface the REAL error — the player deserves to know the
            # game broke, not a vague hiccup and not silent nothing
            _log.exception("game move failed: %s", room.game)
            detail = f"{type(exc).__name__}: {exc}".strip()
            if len(detail) > 200:
                detail = detail[:200].rstrip() + "…"
            out.append(
                f"⚠️ {game.name} errored on that move "
                f"({detail or 'unknown error'}) — the table is still open.")
        if self.is_over(room):
            out.extend(self._finish(room))
        else:
            room.advance_turn()
            self._pump_ai(room, out)
            if self.is_over(room):
                out.extend(self._finish(room))
            else:
                self._persist(room)
        return out

    def _is_turn(self, room: Room, sender: Player) -> bool:
        if room.kind == "dm":
            return any(not p.is_ai for p in room.players) and \
                sender in room.humans
        cur = room.current
        return cur is not None and not cur.is_ai and cur.key == sender.key

    def quit(self, chat_key: str) -> list[str]:
        """Close everything this chat has open: the live room, a
        DB-restored room (mid-game restart), the relay duel, and any
        pending invites this chat sent. Nothing sticks around."""
        with self._lock:
            room = self._rooms.get(chat_key)
            live = room is not None and room.status == "active"
        if live:
            return self._finish(room, "everyone up? table closed.")
        # the room may only exist in the DB (a restart stranded it):
        # restore-then-finish so quit actually kills it instead of
        # reporting "no game" while the active row keeps resurrecting
        with self._lock:
            restored = self._load_live(chat_key)
        if restored is not None:
            return self._finish(restored, "everyone up? table closed.")
        # relay players quit from their own DM, but the room lives at the
        # virtual relay key — tear the relay down too, or the duel sticks
        # around forever with no way to close it from either chat
        relay = self.relay.get_relay_for_chat(chat_key)
        if relay is not None:
            self.relay.close_relay(relay.room_id, reason="quit")
            return [f"duel closed — the {relay.game_name} relay table "
                    "is shut."]
        # pending invites die with the quitter — nobody left to play with
        cancelled = self.relay.cancel_invites_from_chat(chat_key)
        if cancelled:
            s = "s" if cancelled > 1 else ""
            return [f"pending game invite{s} cancelled — table closed."]
        return ["no game is live here."]

    def _pump_ai(self, room: Room, out: list[str]) -> None:
        """Play every consecutive AI seat until a human's turn (or the
        game ends). Bounded so a stuck AI loop can't hang the chat."""
        game = self.games.get(room.game)
        if game is None:
            return
        for _ in range(64):
            if self.is_over(room):
                return
            cur = room.current
            if cur is None or not cur.is_ai:
                return
            try:
                out.extend(game.ai_turn(room, self._mind))
            except Exception as exc:  # noqa: BLE001
                _log.exception("ai turn failed: %s", room.game)
                out.append(
                    f"⚠️ the house ({game.name} AI) errored "
                    f"({type(exc).__name__}) — seat skipped.")
            if self.is_over(room):
                return
            # did the game make progress? (removed the AI seat, changed
            # state, or advanced). If nothing moved at all, bail out.
            before = (room.turn, room.status)
            room.advance_turn()
            after = (room.turn, room.status)
            if before == after and not room.state:
                room.status = "finished"
                out.append("the house got stuck — closing the table.")
                return

    def _handle_timeout(self, room: Room) -> None:
        # Narrow the engine lock to the liveness check: the sweep selected
        # this room before the tick ran, so it may have been quit/finished
        # since — never tick a dead room. Everything below runs under the
        # room guard (like move()): on_timeout and AI turns may call the
        # model, and the global lock must never be held across that or one
        # chat's timeout stalls every other chat.
        with self._lock:
            if (self._rooms.get(room.chat_key) is not room
                    or room.status != "active"):
                return
        with room.guard:
            if room.status != "active":
                return  # a move/quit closed it while we waited for the guard
            game = self.games.get(room.game)
            cur = room.current
            if game is None or cur is None or cur.is_ai:
                return
            msgs = game.on_timeout(room, cur, self._mind)
            if self.is_over(room):
                msgs.extend(self._finish(room))
            else:
                self._pump_ai(room, msgs)
                if self.is_over(room):
                    msgs.extend(self._finish(room))
                else:
                    self._persist(room)
            for m in msgs:
                self._emit(room, m)

    def _finish_idle(self, room: Room) -> None:
        """Close a table the players abandoned: re-check liveness under
        the lock (it may have been quit after the sweep selected it),
        then finish it with an idle notice."""
        with self._lock:
            if (self._rooms.get(room.chat_key) is not room
                    or room.status != "active"):
                return
        self._finish(room, "the table closed — idle too long.")

    def is_over(self, room: Room) -> bool:
        """Pure check — never mutates. ``_finish`` owns the close:
        persisting the finished status and popping the room."""
        if room.status == "finished":
            return True
        game = self.games.get(room.game)
        if game is None:
            return False
        try:
            return bool(game.is_over(room))
        except Exception:  # noqa: BLE001
            _log.exception("game is_over failed: %s", room.game)
            return False

    def _finish(self, room: Room, extra: str | None = None) -> list[str]:
        """Close a room: final message + ledger credits + persistence.

        Serialized on the room guard so a timeout/quit racing an
        in-flight move can't double-award or tear the room down mid-move.
        The room-table pops happen after the guard is released (under the
        engine lock) to keep lock order engine → room everywhere.
        """
        with room.guard:
            msgs, rematch_mem = self._finish_inner(room, extra)
        with self._lock:
            self._rooms.pop(room.chat_key, None)
            self._by_id.pop(room.id, None)
            # remember the table for /game rematch (relay virtual rooms
            # are excluded — a rematch there needs a fresh invite)
            if rematch_mem is not None:
                self._last_game[room.chat_key] = rematch_mem
        return msgs

    def _finish_inner(self, room: Room,
                      extra: str | None = None
                      ) -> tuple[list[str], tuple | None]:
        """The body of :meth:`_finish`. Caller holds ``room.guard``.

        Returns (messages, rematch_memory) — the wrapper stores the
        rematch memory under the engine lock after releasing the guard,
        keeping lock order engine → room everywhere.
        """
        game = self.games.get(room.game)
        if room.status == "finished":
            return ([extra] if extra else [], None)
        room.status = "finished"
        msgs: list[str] = []
        if extra:
            msgs.append(extra)
        if game is not None:
            try:
                msgs.append(game.final_message(room, self._mind))
            except Exception:  # noqa: BLE001
                msgs.append("game over.")
            # victory loot: games with exclusive drops (raid bosses)
            # roll them here; grants go through GearStore so pieces
            # persist like any other gear.
            try:
                for line in self._grant_victory_loot(room, game):
                    msgs.append(line)
            except Exception:  # noqa: BLE001
                _log.debug("victory loot failed", exc_info=True)
            try:
                winner = game.winner(room)
                for p in list(room.players):
                    if p.is_ai:
                        continue
                    won = self._settle_won(game, room, p, winner)
                    points = GameEconomy.reward_points(won, game.score(room, p))
                    score = game.score(room, p)
                    try:
                        difficulty = game.difficulty(room)
                    except Exception:  # noqa: BLE001
                        difficulty = "normal"
                    try:
                        pre = self.store.get(p.key, name=p.name,
                                             platform=p.platform)
                        streak_after = pre.streak + 1 if won is True else pre.streak
                    except Exception:  # noqa: BLE001
                        streak_after = 1 if won is True else 0
                    coins, coin_why = self._settle_coins(
                        game, room, p, won, score, difficulty,
                        streak_after)
                    # boosters: coin charm is consumed on use
                    try:
                        prof_items = self.store.get(p.key).items
                        if int(prof_items.get("coin_charm", 0)) > 0:
                            coins = int(coins * 1.5)
                            coin_why += " + coin charm"
                            msgs.append("✨ Coin Charm consumed!")
                            self.store.consume_item(p, "coin_charm")
                    except Exception:  # noqa: BLE001
                        _log.debug("booster apply failed", exc_info=True)
                    prof_after = self.store.record_outcome(
                        p, won=won, game=room.game, points=points,
                        coins=coins, score=score)
                    msgs.append(f"🪙 {p.name}: +{coins} coins ({coin_why})")
                    # streak milestones: use the streak from the profile
                    # record_outcome just wrote (not the pre-read estimate
                    # above) so the fanfare matches the stored streak.
                    stored_streak = prof_after.streak if prof_after else streak_after
                    if won is True and stored_streak >= 5 and stored_streak % 5 == 0:
                        msgs.append(
                            f"🔥 {stored_streak}-win streak! the bonus "
                            f"tiers keep paying richer.")
                    # record to leaderboard if score > 0
                    if score > 0:
                        try:
                            from .achievements import record_score
                            record_score(self.db, room.game, p.key, p.name, score)
                        except Exception:  # noqa: BLE001
                            _log.debug("leaderboard record failed", exc_info=True)
                    # update game stats
                    try:
                        from .achievements import update_game_stats
                        from .mastery import (get_one_game_stats,
                                              mastery_tier, new_unlocks)
                        before = mastery_tier(
                            room.game,
                            get_one_game_stats(self.db, p.key, room.game))
                        update_game_stats(self.db, p.key, room.game,
                                          won=bool(won), score=score)
                        after = mastery_tier(
                            room.game,
                            get_one_game_stats(self.db, p.key, room.game))
                        if after[1] > before[1]:
                            msgs.append(
                                f"🏅 {p.name} mastery up: {before[0]} → "
                                f"{after[0]} ({room.game})!")
                            for unlock in new_unlocks(room.game, before[1],
                                                      after[1]):
                                msgs.append(f"🔓 {p.name}: {unlock}")
                    except Exception:  # noqa: BLE001
                        _log.debug("game stats update failed", exc_info=True)
                    # persist per-player history (case game: anti-repeat,
                    # skill adaptation, streaks)
                    try:
                        save = getattr(game, "save_history", None)
                        if save is not None:
                            hist = save(self.store, p, room.state)
                            room.state.setdefault(
                                "player_histories", {})[p.key] = hist
                    except Exception:  # noqa: BLE001
                        _log.debug("history save failed", exc_info=True)
                    # award achievements
                    try:
                        new_ach = self._award_achievements(
                            room, game, p, won, score)
                        for ach_name in new_ach:
                            msgs.append(
                                f"🏆 {p.name} unlocked “{ach_name}”!")
                    except Exception:  # noqa: BLE001
                        _log.debug("achievement award failed", exc_info=True)
                    # persistent progression: XP for every finished game,
                    # levels with real stat growth
                    try:
                        from .progression import (
                            award_xp, describe_level_up, xp_bar)
                        amount = game.xp_reward(won, room, p)
                        # daily hunt: the first arena win of the day
                        # pays double XP
                        if room.game == "arena" and won is True:
                            try:
                                from .daily import complete_daily_hunt
                                if complete_daily_hunt(self.db, p.key):
                                    amount *= 2
                                    msgs.append(
                                        "🎯 daily hunt complete! "
                                        "double XP today.")
                            except Exception:  # noqa: BLE001
                                _log.debug("daily hunt failed",
                                           exc_info=True)
                        # Double XP Charm: consumed for the next finished game
                        try:
                            _prof_items = self.store.get(p.key).items
                            if int(_prof_items.get("double_xp", 0)) > 0:
                                amount *= 2
                                msgs.append("✨ Double XP Charm consumed!")
                                self.store.consume_item(p, "double_xp")
                        except Exception:  # noqa: BLE001
                            _log.debug("double xp charm failed", exc_info=True)
                        new_level, gained = award_xp(
                            self.store, p, amount,
                            reason=f"{room.game}:{'win' if won else 'loss' if won is False else 'draw'}")
                        if amount > 0:
                            prof = self.store.get(p.key)
                            msgs.append(
                                f"+{amount} XP {xp_bar(prof.xp)}")
                        for lvl in gained:
                            msgs.append(describe_level_up(lvl))
                            # RPG attributes: each level grants points
                            try:
                                from .stats import StatStore
                                pts, _ = StatStore(self.db).grant_level_points(
                                    p.key, lvl)
                                if pts:
                                    msgs.append(
                                        f"📊 +{pts} attribute points! "
                                        f"Spend with /stats "
                                        f"(strength · stamina · mana · "
                                        f"intelligence).")
                            except Exception:  # noqa: BLE001
                                _log.debug("stat points grant failed",
                                           exc_info=True)
                    except Exception:  # noqa: BLE001
                        _log.debug("xp award failed", exc_info=True)
                msgs.append(
                    f"📈 points and coins credited — /game leaderboard"
                )
            except Exception:  # noqa: BLE001
                _log.debug("game ledger credit failed", exc_info=True)
        try:
            self._reconcile_items(room)
        except Exception:  # noqa: BLE001
            _log.debug("game item reconcile failed", exc_info=True)
        self._persist(room)
        self._emit(room, *msgs)
        # remember the table for /game rematch (relay virtual rooms are
        # excluded — a rematch there needs a fresh invite). The wrapper
        # stores this under the engine lock; we only compute it here.
        rematch_mem = None
        if not room.chat_key.startswith("relay:"):
            humans = [p for p in room.players if not p.is_ai]
            if humans:
                rematch_mem = (room.game, humans, room.kind)
        return msgs, rematch_mem

    @staticmethod
    def _settle_won(game: MultiGame, room: Room, player: Player,
                    winner: Player | str | None) -> bool | None:
        """Per-player win/loss/draw outcome for finish payout.

        Games may settle the outcome themselves via ``finish_won()`` —
        never-ending games like ``world`` treat a prosperous run as a
        win instead of a draw.  Everything else derives from
        ``winner()`` exactly as before.
        """
        try:
            hook = game.finish_won(room, player)
        except Exception:  # noqa: BLE001
            _log.debug("finish_won hook failed", exc_info=True)
            hook = "winner"
        if hook is True or hook is False or hook is None:
            return hook
        # default: the winner()-based derivation
        if winner == "draw":
            return None
        if winner == "all":
            # co-op victory (raid): every human at the table won —
            # full win credit, streaks move
            return True
        if isinstance(winner, Player):
            return winner.key == player.key
        return None

    @staticmethod
    def _settle_coins(game: MultiGame, room: Room, player: Player,
                      won: bool | None, score: int, difficulty: str,
                      streak_after: int) -> tuple[int, str]:
        """Coin payout for a finish.

        Games may take over via ``coin_payout()`` — e.g. ``world``
        settles prosperity with an uncapped score share.  Everything
        else uses the standard ``GameEconomy.coin_breakdown``.
        """
        try:
            custom = game.coin_payout(room, player, won=won, score=score,
                                      difficulty=difficulty,
                                      streak_after=streak_after)
        except Exception:  # noqa: BLE001
            _log.debug("coin_payout hook failed", exc_info=True)
            custom = None
        if custom is not None:
            coins, why = custom
            return max(0, int(coins)), str(why)
        return GameEconomy.coin_breakdown(
            won, score=score, difficulty=difficulty,
            streak_after=streak_after)

    def rematch(self, chat_key: str) -> tuple[Room | None, list[str]]:
        """Start the last finished game again with the same humans.

        Returns (room, messages) or (None, [why-not])."""
        with self._lock:
            last = self._last_game.get(chat_key)
            live = self._rooms.get(chat_key)
        # a live table blocks everything — even a rematchable memory
        if live is not None and live.status == "active":
            return None, [f"a {live.game} is already live — /game quit first."]
        if last is None:
            return None, ["no finished game here yet — /game list to start one."]
        game_name, humans, kind = last
        host, rest = humans[0], humans[1:]
        try:
            room, msgs = self.start(chat_key, game_name, host, kind=kind)
        except ValueError as exc:
            return None, [str(exc)]
        joined: list[str] = []
        for p in rest:
            try:
                joined.extend(self.join(chat_key, p))
            except Exception:  # noqa: BLE001
                _log.debug("rematch reseat failed for %s", p.key,
                           exc_info=True)
        return room, ["🔁 rematch — same game, same table."] + joined + msgs

    def _grant_victory_loot(self, room: Room, game: Any) -> list[str]:
        """Grant exclusive gear drops from a game's ``victory_loot``.

        Games opt in by defining ``victory_loot(room)`` returning
        ``{player_key: [gear_slug, ...]}``.  Pieces are granted through
        ``GearStore`` so they persist, wear, and repair like shop gear.
        """
        loot = getattr(game, "victory_loot", None)
        if loot is None:
            return []
        try:
            drops = loot(room) or {}
        except Exception:  # noqa: BLE001
            _log.debug("victory_loot roll failed", exc_info=True)
            return []
        lines: list[str] = []
        names = {p.key: p.name for p in room.humans}
        for key, slugs in drops.items():
            for slug in slugs or ():
                try:
                    inst = self.gear.grant(key, slug)
                except Exception:  # noqa: BLE001
                    _log.debug("victory loot grant failed for %s", slug,
                               exc_info=True)
                    continue
                who = names.get(key, "you")
                lines.append(f"🎁 {who} loots {inst.display_name()} "
                             f"from the fallen boss! /equip {slug}")
        return lines

    def _award_achievements(self, room: Room, game: Any, player: Player,
                            won: bool | None, score: int) -> list[str]:
        """Award achievements based on game outcome and state.

        Returns the display names of newly unlocked achievements so the
        caller can announce them.
        """
        from .achievements import _achievement_map, unlock_achievement
        db = self.db
        game_name = room.game
        names = _achievement_map()
        newly: list[str] = []

        def grant(achievement_id: str) -> None:
            try:
                if unlock_achievement(db, player.key, achievement_id):
                    ach = names.get(achievement_id)
                    if ach is not None:
                        newly.append(ach.name)
            except Exception:  # noqa: BLE001
                _log.debug("achievement grant failed", exc_info=True)

        # milestones come from the ledger (already includes this game —
        # record_outcome runs before this)
        try:
            prof = self.store.get(player.key)
        except Exception:  # noqa: BLE001
            prof = None

        if prof is not None:
            if prof.games_played >= 10:
                grant("games_10")
            if prof.games_played >= 100:
                grant("games_100")
            if prof.wins >= 25:
                grant("wins_25")
            if prof.wins >= 100:
                grant("wins_100")

        # Poker achievements
        if game_name == "poker":
            if won:
                grant("poker_win")
                if room.state.get("human_allin"):
                    grant("poker_allin_win")
            # straight or better at any showdown this match (rank>=4)
            if room.state.get("human_best_rank", 0) >= 4:
                grant("poker_straight")

        # Hangman achievements
        elif game_name == "hangman":
            if won:
                if room.state.get("wrong", 0) == 0:
                    grant("hangman_perfect")
                if prof is not None:
                    hang_wins = (prof.per_game.get("hangman", {})
                                 .get("wins", 0))
                    if hang_wins >= 5:
                        grant("hangman_5_wins")

        # Arena achievements
        elif game_name == "arena":
            if won:
                grant("arena_win")
                # myth-forged full set: check equipped gear
                try:
                    from .gear import GEAR_CATALOG
                    worn = self.gear.equipped(player.key)
                    if len(worn) >= 2 and all(
                            GEAR_CATALOG.get(i.slug) is not None
                            and GEAR_CATALOG[i.slug].grade == "myth"
                            for i in worn.values()):
                        grant("arena_myth_set")
                except Exception:  # noqa: BLE001
                    pass
                if room.state.get("crit_kill_by") == "you":
                    grant("arena_crit_kill")
                rank = room.state.get("house_rank")
                if rank == "S":
                    grant("arena_s_rank")
                elif rank == "SS":
                    grant("arena_ss_rank")
                elif rank == "X":
                    grant("arena_x_rank")
                if room.state.get("myth_foe"):
                    grant("arena_myth_foe")
                if int(room.state.get("dmg_taken", 0)) <= 0:
                    grant("arena_flawless")
                you_pow = int(room.state.get("player_power", 0))
                foe_pow = int(room.state.get("house_power", 0))
                if you_pow > 0 and foe_pow >= you_pow * 1.1:
                    grant("arena_upset")
                if room.state.get("killing_skill"):
                    grant("arena_skill_kill")
                if room.state.get("brutal_finish") == "you":
                    grant("arena_brutal")
                # comeback: won from below 20% HP
                you = room.state.get("you", {})
                max_hp = int(you.get("max_hp", 1))
                lowest = int(room.state.get("lowest_hp", max_hp))
                if lowest < max_hp * 0.2:
                    grant("arena_comeback")
                # purist: no potions used
                if int(room.state.get("potions_used", 0)) <= 0:
                    grant("arena_no_potion")
                # ── unlikely scenarios: the strange glories ──
                if int(you.get("hp", 0)) == 1:
                    grant("arena_1hp_win")
                if int(room.state.get("turns", 99)) <= 3:
                    grant("arena_fast_win")
                if int(room.state.get("turns", 0)) >= 20:
                    grant("arena_marathon")
                if not bool(room.state.get("used_basic_attack", True)):
                    grant("arena_skills_only")
                if bool(room.state.get("mana_hit_zero", False)):
                    grant("arena_mana_starved")
                duals = int(room.state.get("dual_casts", 0))
                if duals >= 10:
                    grant("arena_dual_10")
                if float(room.state.get("dual_lowest_odds", 1.0)) < 0.30 \
                        and duals > 0:
                    grant("arena_lucky_dual")
                if float(room.state.get("dual_lowest_odds", 1.0)) >= 0.95 \
                        and duals > 0:
                    grant("arena_perfect_dual")
                ks = str(room.state.get("killing_skill", ""))
                if ks.startswith("combo:"):
                    grant("arena_dual_kill")
                # barehanded: no gear equipped at battle start
                try:
                    loadout = room.state.get("loadout", {}).get(
                        player.key, {})
                    if not loadout:
                        grant("arena_no_gear")
                except Exception:  # noqa: BLE001
                    pass
                # underdog: track wins as the weaker fighter (persistent)
                if you_pow > 0 and foe_pow >= you_pow * 1.1:
                    try:
                        db.execute(
                            "CREATE TABLE IF NOT EXISTS game_counters ("
                            "player_key TEXT NOT NULL, "
                            "counter TEXT NOT NULL, "
                            "value INTEGER NOT NULL DEFAULT 0, "
                            "PRIMARY KEY (player_key, counter))")
                        rows = db.query(
                            "SELECT value FROM game_counters "
                            "WHERE player_key = ? AND counter = ?",
                            (player.key, "underdog_wins"))
                        cur = int(rows[0]["value"]) + 1 if rows else 1
                        db.execute(
                            "INSERT INTO game_counters "
                            "(player_key, counter, value) VALUES (?, ?, ?) "
                            "ON CONFLICT(player_key, counter) DO UPDATE SET "
                            "value = excluded.value",
                            (player.key, "underdog_wins", cur))
                        if cur >= 5:
                            grant("arena_underdog_5")
                    except Exception:  # noqa: BLE001
                        _log.debug("underdog counter failed", exc_info=True)
            # losing streak: 10 in a row earns a strange glory
        if game_name == "arena" and not won:
            try:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS game_counters ("
                    "player_key TEXT NOT NULL, "
                    "counter TEXT NOT NULL, "
                    "value INTEGER NOT NULL DEFAULT 0, "
                    "PRIMARY KEY (player_key, counter))")
                rows = db.query(
                    "SELECT value FROM game_counters "
                    "WHERE player_key = ? AND counter = ?",
                    (player.key, "loss_streak"))
                cur = int(rows[0]["value"]) + 1 if rows else 1
                db.execute(
                    "INSERT INTO game_counters "
                    "(player_key, counter, value) VALUES (?, ?, ?) "
                    "ON CONFLICT(player_key, counter) DO UPDATE SET "
                    "value = excluded.value",
                    (player.key, "loss_streak", cur))
                if cur >= 10:
                    grant("arena_lose_10")
            except Exception:  # noqa: BLE001
                _log.debug("loss streak counter failed", exc_info=True)
        # a win resets the losing streak
        if game_name == "arena" and won:
            try:
                db.execute(
                    "INSERT INTO game_counters (player_key, counter, value) "
                    "VALUES (?, ?, 0) "
                    "ON CONFLICT(player_key, counter) DO UPDATE SET "
                    "value = 0",
                    (player.key, "loss_streak"))
            except Exception:  # noqa: BLE001
                pass
                if prof is not None:
                    if prof.streak >= 5:
                        grant("arena_streak_5")
                    if prof.streak >= 10:
                        grant("arena_streak_10")
                    if prof.streak >= 15:
                        grant("arena_streak_15")
                    if prof.streak >= 20:
                        grant("arena_streak_20")
                    if prof.streak >= 25:
                        grant("arena_streak_25")
                    try:
                        if int(prof.wins) >= 100:
                            grant("arena_100_wins")
                    except Exception:  # noqa: BLE001
                        pass
            # witnessing a forbidden technique counts win or lose
            if room.state.get("enemy_skill_cast"):
                grant("arena_forbidden")

        # PvP / raid achievements
        elif game_name == "pvp":
            if won:
                grant("arena_pvp_win")
        elif game_name == "raid":
            if won:
                grant("arena_raid_win")

        # 2048 achievements
        elif game_name == "2048":
            max_tile = max(max(r) for r in room.state.get("grid", [[0]]))
            if max_tile >= 2048:
                grant("2048_win")
            if max_tile >= 4096:
                grant("2048_4096")
            if score >= 5000:
                grant("2048_score_5k")

        # Snake achievements
        elif game_name == "snake":
            if score >= 50:
                grant("snake_50")
            if score >= 200:
                grant("snake_200")
            moves = room.state.get("moves", 0)
            if moves >= 20 and not room.state.get("alive", True):
                grant("snake_no_crash_20")

        # Connect Four achievements
        elif game_name == "connect4" and won:
            grant("connect4_win")
            moves = room.state.get("moves", 0)
            if moves < 10:
                grant("connect4_quick")

        # Battleship achievements
        elif game_name == "battleship" and won:
            grant("battleship_win")
            shots = room.state.get("ai_shots", [])
            hits = sum(shots[r][c] == 1 for r in range(10) for c in range(10))
            total = sum(shots[r][c] > 0 for r in range(10) for c in range(10))
            if total > 0 and hits / total >= 0.8:
                grant("battleship_perfect")

        # World achievements
        elif game_name == "world":
            pop = room.state.get("pop", 0)
            if pop >= 50:
                grant("world_50_pop")
            buildings = room.state.get("buildings", {})
            if len(buildings) >= 5:  # has all building types
                grant("world_all_buildings")

        # RPG achievements
        elif game_name == "rpg":
            sheets = room.state.get("sheets", {})
            player_sheet = sheets.get(player.key, {})
            if room.state.get("done"):
                grant("rpg_finish")
            if player_sheet.get("level", 1) >= 5:
                grant("rpg_level_5")

        # Casino achievements
        elif game_name == "blackjack" and won:
            grant("blackjack_win")
            player_hand = room.state.get("player", [])
            if len(player_hand) >= 2:
                # Check for exact 21 (not blackjack which is 2 cards)
                hand_val = sum(min(c, 10) for c in player_hand)
                aces = sum(1 for c in player_hand if c == 14)
                while hand_val > 21 and aces > 0:
                    hand_val -= 10
                    aces -= 1
                if hand_val == 21 and len(player_hand) > 2:
                    grant("blackjack_21")

        elif game_name == "roulette":
            bet_type = room.state.get("bet_type", "")
            payout = room.state.get("payout", 0)
            if bet_type == "number" and payout > 0:
                grant("roulette_number")

        elif game_name == "slots":
            reels = room.state.get("reels", [])
            if len(reels) == 3 and reels[0] == reels[1] == reels[2] == "💎":
                grant("slots_jackpot")

        # Case achievements
        elif game_name == "case":
            hist = ((room.state.get("player_histories") or {})
                    .get(player.key) or {})
            if won and room.state.get("solved"):
                grant("case_first")
                streak = int(hist.get("streak") or 0)
                if streak >= 3:
                    grant("case_streak_3")
                if streak >= 5:
                    grant("case_streak_5")
                tier = (room.state.get("tier")
                        or (room.state.get("case") or {}).get("tier"))
                if tier == "expert":
                    grant("case_expert")
                if (room.state.get("strikes", 0) == 0
                        and room.state.get("hints_used", 0) == 0):
                    grant("case_clean")
                if (room.state.get("timed")
                        and int(room.state.get("time_bonus", 0)) > 0):
                    grant("case_timed")

        # Sudoku achievements
        elif game_name == "sudoku":
            if won:
                grant("sudoku_win")
                if room.state.get("difficulty") in ("hard", "expert"):
                    grant("sudoku_hard")
                if (room.state.get("mistakes", 0) == 0
                        and room.state.get("hints_used", 0) == 0):
                    grant("sudoku_clean")

        # Anagram achievements
        elif game_name == "anagram":
            if won:
                grant("anagram_win")
                rounds = int(room.state.get("rounds", 6))
                rw = int((room.state.get("round_wins") or {})
                         .get(player.key, 0))
                if rw >= rounds:
                    grant("anagram_ace")

        # Cryptogram achievements
        elif game_name == "cryptogram":
            if won:
                grant("cryptogram_win")
            rounds = room.state.get("rounds", [])
            wrong_total = sum(int((r.get("wrong") or {}).get(player.key, 0))
                              for r in rounds)
            solved_any = any(r.get("solved_by") == player.key
                             for r in rounds)
            if solved_any and wrong_total == 0:
                grant("cryptogram_perfect")

        # Wordle achievements
        elif game_name == "wordle":
            if won:
                grant("wordle_win")
                if len(room.state.get("guesses", [])) <= 3:
                    grant("wordle_ace")

        # Minesweeper
        elif game_name == "mines" and won:
            grant("mines_win")

        # Concentration
        elif game_name == "memory":
            if won:
                grant("memory_win")
                if int(room.state.get("moves", 999)) <= 24:
                    grant("memory_sharp")

        # Craps
        elif game_name == "craps":
            if won:
                grant("craps_win")
                bank = int((room.state.get("bank") or {})
                           .get(player.key, 0))
                if bank >= 200:
                    grant("craps_high_roller")

        # Inbox classics
        elif game_name == "reversi" and won:
            grant("reversi_win")

        elif game_name == "checkers" and won:
            grant("checkers_win")

        elif game_name == "gomoku" and won:
            grant("gomoku_win")

        # Quiz duel
        elif game_name == "duel":
            if won:
                grant("duel_win")
                opp = [int(v) for k, v in
                       (room.state.get("points") or {}).items()
                       if k != player.key]
                if opp and max(opp) == 0:
                    grant("duel_flawless")

        # Trivia royale
        elif game_name == "trivia" and won:
            grant("trivia_win")

        # Tic-tac-toe: a draw vs the perfect house is the achievement
        elif game_name == "ttt":
            if won is None and room.state.get("winner") == "draw":
                grant("ttt_draw")

        # Mafia: surviving the five nights
        elif game_name == "mafia":
            if player.key in (room.state.get("alive") or []):
                grant("mafia_win")

        # Escape room: the table got out together
        elif game_name == "escape":
            puzzles = room.state.get("puzzles") or []
            if puzzles and int(room.state.get("lock", 0)) >= len(puzzles):
                grant("escape_win")

        # Political / spy / auction / 20q / bulls / numberguess / king
        elif game_name == "political" and won:
            grant("political_win")

        elif game_name == "spy" and won:
            grant("spy_win")

        elif game_name == "auction" and won:
            grant("auction_win")

        elif game_name == "20q" and won:
            grant("twentyq_win")

        elif game_name == "bulls" and won:
            grant("bulls_win")

        elif game_name == "numberguess" and won:
            grant("numberguess_win")

        elif game_name == "king" and won:
            grant("king_win")

        return newly

    # ── inspection ─────────────────────────────────────────────────────────
    def describe(self, room: Room) -> str:
        game = self.games.get(room.game)
        diff = (room.state or {}).get("_difficulty", "normal")
        lines = [
            f"🎮 {room.game} — {room.status} ({room.kind}) · 🎯 {diff}",
            "players:",
        ]
        for i, p in enumerate(room.players):
            marker = "←" if i == room.turn else "·"
            tag = " 🤖" if p.is_ai else ""
            lines.append(f"  {marker} {p.name}{tag}")
        if game is not None:
            try:
                extra = game.describe_state(room)
                if extra:
                    lines.append(extra)
            except Exception:  # noqa: BLE001
                pass
        cur = room.current
        if cur is not None and not cur.is_ai and room.status == "active":
            lines.append(f"turn: {cur.name}")
        return "\n".join(lines)

    def list_games(self, *, with_groups: bool = False) -> str:
        from .games.easy import EASY_GAMES
        from .games.medium import MEDIUM_GAMES
        from .games.ambitious import AMBITIOUS_GAMES
        from .games.wild import WILD_GAMES
        from .games.arcade import ARCADE_GAMES
        from .games.casino import CASINO_GAMES
        from .games.inbox import INBOX_GAMES
        from .games.puzzles import PUZZLE_GAMES
        lines = ["games — start one with /game <name>:"]
        for label, group in (("easy", EASY_GAMES),
                             ("medium", MEDIUM_GAMES),
                             ("ambitious", AMBITIOUS_GAMES),
                             ("wild", WILD_GAMES),
                             ("arcade", ARCADE_GAMES),
                             ("casino", CASINO_GAMES),
                             ("puzzles", PUZZLE_GAMES),
                             ("inbox — async, one message per turn, "
                              "no clock", INBOX_GAMES)):
            lines.append(f"  — {label} —")
            for g in group:
                extra = " (group)" if g.needs_group else ""
                diff = (" [easy|normal|hard|expert]"
                        if g.difficulties else "")
                lines.append(f"  /game {g.name:<18} {g.description}"
                             f"{diff}{extra}")
        lines.append("  /game leaderboard [game]   the rankings")
        lines.append("  /game stats [name]         a player's record")
        lines.append("  /game shop                 spend your coins")
        lines.append("  /game quit                 leave the table")
        return "\n".join(lines)

    def rooms(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"id": r.id, "game": r.game, "chat": r.chat_key,
                 "kind": r.kind, "status": r.status,
                 "players": len(r.players),
                 "humans": len(r.humans)}
                for r in self._rooms.values()
            ]
