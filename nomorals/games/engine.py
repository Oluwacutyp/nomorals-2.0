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
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .ai import GameMind
from .economy import GameEconomy, ShopItem
from .games.base import GAME_COMMANDS, MultiGame, Room, parse_command
from .players import AI_PLAYER, Leaderboard, Player, PlayerStore
from .purse import BoundPurse, StorePurseBackend
from .engine_failures import record_failure


def _collapse_dup_lines(items: list[str]) -> list[str]:
    """Collapse identical consecutive lines across an assembled output.

    Chat output is assembled from many hands — a game's own lines, the
    settle lines, the cards. When two of them independently append the
    same line back-to-back, the table sees it once. Two shapes count:
    a line repeated verbatim (across or within items), and an item
    that repeats the trailing line of the item before it (an arena
    skill message already ends with the win line; its caller then
    appends that same win line as its own item). Only exact,
    non-empty, immediately repeated lines collapse — short echoes and
    deliberate non-consecutive repeats pass through untouched.
    """
    out: list[str] = []
    prev: str | None = None
    for item in items:
        raw = str(item)
        lines = raw.split("\n")
        if len(lines) == 1 and raw.strip() and len(raw.strip()) >= 12 \
                and prev is not None and prev.strip() \
                and (prev == raw or prev.endswith(" " + raw)):
            continue  # the previous item already ends with this line
        kept: list[str] = []
        for line in lines:
            if line.strip() and line == prev:
                continue
            kept.append(line)
            prev = line
        text = "\n".join(kept)
        if text or not raw:
            out.append(text)
    return out


def render_game_over_card(game_name: str, players: list[Player],
                          scores: dict[str, int],
                          winner_label: str = "",
                          store: Any | None = None,
                          db: Any | None = None,
                          balances: dict[str, int] | None = None) -> str:
    """The final table card: standings, scores, balances, tier moves.

    ``players`` in seat order; ``scores`` maps player key → final score.
    Sorted by score desc (then name). Ranked games get Glicko tier lines
    when ``db`` is provided. Pure render — never raises, never writes.

    The score renders AS a score (never dressed as "points" — points
    are the separate progression currency).  ``balances`` (player key
    → post-settlement balance) overrides the store read per player:
    wager games pass the bound purse's balance, which is the purse
    that actually moved.
    """
    try:
        humans = [p for p in players if not getattr(p, "is_ai", False)]
        rows = sorted(humans,
                      key=lambda p: (-scores.get(p.key, 0), p.name))
        medals = ("🥇", "🥈", "🥉")
        title = f"🏁 **{game_name}** — final"
        if winner_label:
            title += f" · 👑 {winner_label}"
        lines = [title]
        for i, p in enumerate(rows):
            mark = medals[i] if i < 3 else f"{i + 1}."
            bits = [f"{mark} {p.name}"]
            sc = scores.get(p.key)
            if sc is not None:
                bits.append(f"{sc} score")
            bal = balances.get(p.key) if balances is not None else None
            if bal is not None:
                bits.append(f"{bal}c")
                if store is not None:
                    try:
                        from .progression import level_for_xp

                        prof = store.get_for(p)
                        bits.append(f"lvl {level_for_xp(prof.xp)}")
                    except Exception as e:  # noqa: BLE001
                        record_failure("render_game_over_card:bal_lvl", e)
            elif store is not None:
                try:
                    from .progression import level_for_xp

                    prof = store.get_for(p)
                    bits.append(f"{prof.coins}c")
                    # level is DERIVED from xp (Profile has no level
                    # field) — getattr(..., 'level', 1) pinned every
                    # card to "lvl 1" forever.
                    bits.append(f"lvl {level_for_xp(prof.xp)}")
                except Exception as e:  # noqa: BLE001
                    record_failure("render_game_over_card:store_coins", e)
            if db is not None:
                try:
                    from .matchmaking import get_glicko
                    g = get_glicko(db, p.key, game_name)
                    if int(g.get("games", 0)) > 0:
                        from .matchmaking import rank_tier_emoji
                        bits.append(f"{rank_tier_emoji(g['ordinal'])}"
                                    f"{g['tier']} {int(g['ordinal'])}")
                except Exception as e:  # noqa: BLE001
                    record_failure("render_game_over_card:glicko", e)
            lines.append(" · ".join(bits))
        # No "↩️ /game rematch — run it back" TEXT line (2026-11-06):
        # rematch is a declared action on the funnel renders that
        # carry a finish (the ActionableMessage buttons), never an
        # affordance printed into the card every surface shares.
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        record_failure("render_game_over_card:outer", e)
        return ""

__all__ = ["GameEngine", "SendFn", "FencedMove"]

_log = get_logger(__name__)

SendFn = Callable[[str, str], None]  # (chat_key, text) -> None


@dataclass
class FencedMove:
    """The outcome of :meth:`GameEngine.move_fenced`.

    ``status``: ``ok`` (the move ran), ``stale`` (the named table
    revision has been superseded — nothing applied, the current
    state rides in ``messages``), ``missing`` (no live room),
    ``finished`` (the room is closed), ``not_player`` (the sender
    holds no seat), ``not_turn`` (a seat, but not the current one).
    """

    status: str
    messages: list[str] = field(default_factory=list)
    room: Room | None = None


class _SettleAbort(Exception):
    """Internal: a db-backed settle failed mid-apply and rolled
    back. Caught at the top of ``_finish_inner`` — never escapes
    the engine."""

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
        #: Optional group-economy bridge: callable(chat_key, player,
        #: coins, game) invoked when a human wins coins in a group
        #: room.  Installed by the runtime; None = no bridge.
        self.coin_sink = None
        #: Optional wager-purse source: callable(room) -> purse
        #: backend | None, installed by the runtime (group rooms are
        #: bound to the group economy estate).  None = every room
        #: plays against the games wallet (StorePurseBackend).
        self.purse_source = None
        #: Optional live-table sink: callable(room, event) invoked
        #: on table events (started/moved/joined/left/timeout/
        #: finished) so a surface can keep ONE living card revised
        #: in place. Installed by the runtime; None = no card.
        self.table_sink = None
        #: Persistence health: save outcomes for the room store.
        #: A persist failure used to vanish into a debug log; now it
        #: is counted here, recorded in the failure ledger, and
        #: named on the table surface while it is the latest word.
        self._persist_stats: dict[str, Any] = {
            "ok": 0, "failed": 0, "last_error": "",
            "last_ok_at": 0.0, "last_fail_at": 0.0}
        self._settlements: Any = None  # lazy SettlementStore
        self._sweep_count = 0
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
        #: Agent seat deciders: key-prefix -> callable(room, player, game,
        #: mind) -> list[str] | None. Registered by the characters module
        #: (or others) so character/brain seats play through the real
        #: engine instead of the built-in house brain. Returning None
        #: falls back to the game's ai_turn.
        self.seat_deciders: dict[str, Any] = {}
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._relay_obj: Any = None  # lazy GameRelay (see relay property)
        self._matchmaker_obj: Any = None  # lazy Matchmaker (see below)
        self._dm_obj: Any = None  # lazy GameMaster (see dm property)
        self._forge_obj: Any = None  # lazy TriviaForge (see forge property)
        self._last_game: dict[str, tuple[str, list[Player], str]] = {}
        # last finished fair table per chat: (game, fair fragment) —
        # the fairness reveal is served ON DEMAND from here (/game
        # fair), never blasted into the chat at settle time.
        self._last_fair: dict[str, tuple[str, dict[str, Any]]] = {}
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

    @property
    def matchmaker(self) -> Any:
        """The matchmaking queue, created on first use. The ticker
        sweeps it (see _sweep_timeouts) so queued players get paired
        without anyone tripping a command."""
        mm = self._matchmaker_obj
        if mm is None:
            from .matchmaking import Matchmaker
            mm = Matchmaker(self)
            self._matchmaker_obj = mm
        return mm

    @property
    def dm(self) -> Any:
        """The table's dungeon master — narrates notable game events in
        each game's DM mood (``/dm mood``). Games push bare events with
        :func:`nomorals.games.gamemaster.feed`; the engine drains the
        feed after moves and at finish. Created on first use."""
        gm = self._dm_obj
        if gm is None:
            from .gamemaster import GameMaster
            from .npc import NPCStore
            gm = GameMaster(NPCStore(), suggest=self._suggest)
            self._dm_obj = gm
        return gm

    @property
    def forge(self) -> Any:
        """The trivia forge — dynamic question generation with
        interest-profile topics and persisted anti-repeat. Created on
        first use; shared by every trivia/quiz game."""
        forge = self._forge_obj
        if forge is None:
            from .trivia_forge import TriviaForge
            forge = TriviaForge(suggest=self._suggest, db=self.db,
                                context=self.context)
            self._forge_obj = forge
        return forge

    def _drain_dm_feed(self, room: Room, game: MultiGame, out: list[str],
                       *, force: bool = False) -> None:
        """Narrate the game's DM feed (see
        :func:`nomorals.games.gamemaster.feed`).

        Cooldown-guarded (25s) so a chatty game never spams the table;
        the finish drain passes ``force=True`` so the last moments
        still get their scene. Never raises — a silent DM beats a
        broken game.

        The cooldown DEFERS, it never discards: the feed is peeked,
        and events are consumed only at the moment they are actually
        narrated. A drain that lands inside the cooldown leaves the
        feed exactly as it stands, so the moments narrate — in their
        original order — on the first drain after the cooldown lapses
        (or on the finish drain, which forces).

        One narration per moment: every event text this room narrates
        is remembered in ``_dm_narrated`` and never narrated again,
        and when a drain narrates while the game is already over, the
        room is marked ``_dm_end_narrated`` — the finish path checks
        both before letting a ``dm_finale`` game tell its ending a
        second time.
        """
        feed = room.state.get("_dm_feed")
        if not feed:
            return
        narrated: list[str] = room.state.setdefault("_dm_narrated", [])
        fresh: list[Any] = []
        for item in feed:
            text = item.get("text") if isinstance(item, dict) else str(item)
            if text and str(text).strip() in narrated:
                continue  # this moment already had its narration
            fresh.append(item)
        if not fresh:
            # Nothing new to tell — the feed held only moments that
            # already had their narration. Clear it so they are not
            # re-scanned on every later drain.
            room.state.pop("_dm_feed", None)
            return
        now = time.time()
        if not force:
            last = float(room.state.get("_dm_last_narrate") or 0)
            if now - last < 25:
                return  # deferred: the feed stays for the next drain
        # Narrating now — consume the feed, then tell it.
        room.state.pop("_dm_feed", None)
        room.state["_dm_last_narrate"] = now
        try:
            gm = self.dm
            cast = None
            if room.kind == "group":
                try:
                    cast = gm.ensure_cast(game.name)
                except Exception as e:  # noqa: BLE001
                    record_failure("_drain_dm_feed:ensure_cast", e)
                    cast = None
            line = gm.announce(game.name, fresh, cast=cast)
            if line:
                out.append(line)
                for item in fresh:
                    text = item.get("text") if isinstance(item, dict) \
                        else str(item)
                    if text:
                        narrated.append(str(text).strip())
                del narrated[:-24]  # bounded room memory
                try:
                    over = bool(game.is_over(room))
                except Exception:  # noqa: BLE001
                    over = bool(room.state.get("done"))
                if over:
                    room.state["_dm_end_narrated"] = True
        except Exception:  # noqa: BLE001
            _log.debug("dm narration failed", exc_info=True)

    def _drain_dm_feed_safe(self, room: Room, game: MultiGame,
                            out: list[str], *, force: bool = False) -> None:
        """Version-skew-safe DM drain.

        The DM narration is a nice-to-have layer — it must never break
        the core game flow. This guards against both narration bugs and
        the method itself missing (stale code during a rolling update):
        either way the game completes and the player gets their result.
        """
        drain = getattr(self, "_drain_dm_feed", None)
        if drain is None:
            _log.debug("dm drain unavailable (version skew) — skipping")
            return
        try:
            drain(room, game, out, force=force)
        except Exception:  # noqa: BLE001
            _log.debug("dm drain failed", exc_info=True)

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
        # pair queued players into duels (throttled inside)
        try:
            mm = self._matchmaker_obj
            if mm is not None:
                mm.maybe_sweep()
        except Exception:  # noqa: BLE001
            _log.debug("match sweep failed", exc_info=True)
        self._maybe_recover_settlements()
        self._maybe_prune_history(now)

    def _maybe_recover_settlements(self) -> None:
        """Throttled: re-finish rooms whose game is decided but
        whose settle never landed — a crash mid-settle (the plan
        row stayed ``planned``, the credits rolled back), a crash
        after the settle commit (receipts exist; the replay path
        re-issues them and moves nothing), or a finish aborted
        in-process. The re-finish is the ordinary one: plan-first,
        atomic, receipt-resolving, so recovery can never double-pay.
        """
        self._sweep_count += 1
        if self._sweep_count % 5 != 0:
            return
        store = self._settlement_store() if self.db is not None else None
        due: list[Room] = []
        with self._lock:
            rooms = list(self._rooms.values())
        for room in rooms:
            if room.status != "active":
                continue
            try:
                if (store is not None and store.available
                        and store.get(room.id) is not None) \
                        or self.is_over(room):
                    due.append(room)
            except Exception:  # noqa: BLE001 - one bad room never
                _log.debug("settlement recovery scan failed",
                           exc_info=True)
        for room in due:
            try:
                self._finish(room)
            except Exception:  # noqa: BLE001
                _log.exception("settlement recovery finish failed: %s",
                               room.game)

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
    def _bind_purse(self, room: Room) -> None:
        """Bind the room's wager purse (transient — rebound on every
        open and restore, never persisted).  The runtime's
        ``purse_source`` may claim the room (group rooms play in the
        group economy estate); anything unclaimed plays against the
        games wallet."""
        backend = None
        source = getattr(self, "purse_source", None)
        if source is not None:
            try:
                backend = source(room)
            except Exception:  # noqa: BLE001 - a bad source never blocks a table
                _log.debug("purse source failed", exc_info=True)
                backend = None
        if backend is None:
            backend = StorePurseBackend(self.store, self.db)
        room.purse = BoundPurse(backend, room)

    # ── persistence health ──────────────────────────────────────────
    def persist_health(self) -> dict[str, Any]:
        """Save-health for the room store: counts plus the latest
        error. ``degraded`` is True when the most recent save
        outcome is a failure — the state the table surface names,
        because a game whose saves fail will not survive a restart
        and must not look identical to a healthy one."""
        stats = dict(self._persist_stats)
        stats["degraded"] = bool(
            stats["failed"] and stats["last_fail_at"] >= stats["last_ok_at"])
        return stats

    def _settlement_store(self) -> Any:
        """The settlement ledger over this engine's db (lazy)."""
        if self._settlements is None:
            from .settlement import SettlementStore

            self._settlements = SettlementStore(self.db)
        return self._settlements

    def _table_event(self, room: Room, event: str) -> None:
        """Notify the live-table sink (when installed). A sink
        failure never breaks a game — the card is a surface, the
        room is the truth."""
        sink = getattr(self, "table_sink", None)
        if sink is None:
            return
        try:
            sink(room, event)
        except Exception:  # noqa: BLE001
            _log.debug("game table sink failed", exc_info=True)

    @staticmethod
    def _journal_event(room: Room, kind: str, player: Player | None,
                       text: str = "") -> None:
        """Append one real event to the room's replay journal
        (``state['_events']``, persisted with the room, capped).
        Replays are built from these records — never invented."""
        try:
            journal = room.state.setdefault("_events", [])
            journal.append({
                "ts": time.time(), "kind": kind,
                "actor": player.key if player is not None else "",
                "name": player.name if player is not None else "",
                "text": str(text or "")[:120]})
            del journal[:-80]
        except Exception:  # noqa: BLE001 - journaling never breaks play
            _log.debug("event journal failed", exc_info=True)

    def _persist(self, room: Room) -> bool:
        """Persist one room; bump its table revision.

        Returns whether the row landed. Every call is a new table
        revision — the persisted snapshot IS the state actions are
        fenced against. Failures are counted and recorded (see
        :meth:`persist_health`), never just debug-logged.
        """
        room.revision += 1
        if self.db is None:
            return False
        # serialize under the room guard: a concurrent move mutating
        # state mid-dumps would otherwise raise or persist a torn room
        with room.guard:
            try:
                self._persist_inner(room)
            except Exception as exc:  # noqa: BLE001
                self._persist_stats["failed"] += 1
                self._persist_stats["last_error"] = \
                    f"{type(exc).__name__}: {exc}"[:200]
                self._persist_stats["last_fail_at"] = time.time()
                record_failure("engine:_persist", exc)
                return False
            self._persist_stats["ok"] += 1
            self._persist_stats["last_ok_at"] = time.time()
            return True

    def _persist_inner(self, room: Room) -> None:
        """Write the room row. Raises on failure — the caller
        (:meth:`_persist`) owns the health bookkeeping."""
        import json as _json
        # engine bookkeeping rides a COPY of the state blob: the
        # revision and the turn/idle clocks, so a restart restores
        # the same revision and the same deadline (Room.from_row
        # pops them back out; games never see them in state).
        blob = dict(room.state)
        blob["_rev"] = room.revision
        blob["_turn_started"] = room.turn_started
        blob["_last_activity"] = room.last_activity
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO game_rooms (id, game, chat_key, platform, kind, "
                "players, turn, state, status, started_at, seed, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET turn = excluded.turn, "
                "state = excluded.state, status = excluded.status, "
                "players = excluded.players, updated_at = excluded.updated_at",
                (room.id, room.game, room.chat_key, room.platform,
                 room.kind, _json.dumps([
                     # full identity, not just the key: from_row used
                     # to rebuild the display name from the key, so a
                     # restart turned every player into their raw
                     # sender id ("Ada" -> "456"). Dict shape; the
                     # legacy plain-key list still loads (see
                     # Room.from_row).
                     {"key": p.key, "name": p.name,
                      "platform": p.platform, "is_ai": bool(p.is_ai)}
                     for p in room.players]),
                 room.turn, _json.dumps(blob), room.status,
                 room.started_at, room.seed, time.time()),
            )

    def _load_live(self, chat_key: str) -> Room | None:
        """Restore a room that survived a restart (mid-game reboot)."""
        if self.db is None:
            return None
        try:
            row = self.db.query_one(
                "SELECT * FROM game_rooms WHERE chat_key = ? AND status = 'active' "
                "ORDER BY updated_at DESC LIMIT 1", (chat_key,)
            )
        except Exception as e:  # noqa: BLE001
            record_failure("_load_live:db_query", e)
            return None
        if row is None:
            return None
        room = Room.from_row(row, {})
        self._bind_purse(room)
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
        except Exception as exc:  # noqa: BLE001
            record_failure("engine:_delete_room_row", exc)

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
              variant: str = "",
              bet: int = 0) -> tuple[Room, list[str]]:
        """Open a room and seat the host (plus AI seats).

        Returns (room, messages_to_send). Raises ValueError when the
        game is unknown or the chat kind doesn't fit (group-only game in
        a DM, …). ``difficulty`` is only handed to games that declare a
        named ``difficulty`` parameter on ``new_state``. ``variant`` is a
        game-specific mode word (gomoku "big"/"huge", hangman "long",
        trivia "sudden", 2048 "big") — only games that declare it get
        it. ``mastery`` (the host's current tier index for this game)
        lets games gate their unlockables. ``bet`` is the one-command
        wager for casino starts (``/blackjack 500``): handed through
        the same opt-in kwargs, so only games whose ``new_state``
        declares it (blackjack) ever see it.
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
            # the wager purse binds BEFORE setup: buy-in games
            # (blackjack, craps, poker) stake their escrow there.
            self._bind_purse(room)
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
            except Exception as e:  # noqa: BLE001
                record_failure("start:mastery", e)
                mastery_idx = 0
            room.state = game.new_state(
                game.rng(room),
                **_new_state_kwargs(
                    game, daily=daily, timed=timed,
                    difficulty=difficulty, variant=variant,
                    mastery=mastery_idx, history=history,
                    bet=bet,
                    trivia_forge=self.forge))
            # stash difficulty on the room so start/status messages can show it
            room.state["_difficulty"] = difficulty
            # provably-fair tables: open the commit-reveal table BEFORE
            # setup() draws anything, so the commitment covers the whole
            # game (deck order, spins, dice, reels).
            if getattr(game, "fair", False):
                try:
                    from .fairness import get_client_seed, init_table
                    client = (get_client_seed(self.db, host.key)
                              if self.db is not None else "house")
                    room.state["fair"] = init_table(client)
                except Exception:  # noqa: BLE001
                    _log.debug("fair table init failed", exc_info=True)
            self._mirror_inventory(room)
            self._rooms[chat_key] = room
            self._by_id[room.id] = room
        msgs: list[str] = []
        try:
            intro = game.setup(room, self._mind)
            if intro:
                msgs.append(intro)
            # the fairness commitment is MADE before any draw —
            # the player sees the note before play starts. The hash
            # itself stays in the engine's records (no ids in chat,
            # 2026-11-06); the note is the in-chat face of it.
            if room.state.get("fair") is not None:
                try:
                    from .fairness import commit_note
                    msgs.append(commit_note(room.state["fair"]))
                except Exception:  # noqa: BLE001
                    _log.debug("fair commit line failed", exc_info=True)
            # the season touches this table — say so up front.
            try:
                from .seasons import event_blurb
                blurb = event_blurb(self.db, game.name)
                if blurb:
                    msgs.append(blurb)
            except Exception:  # noqa: BLE001
                _log.debug("season blurb failed", exc_info=True)
            # Group starts get a brief how-to-play blurb (join + core
            # loop) so nobody stares at the table wondering what to do.
            # Solo/DM games don't need it — the intro is enough there.
            howto = (getattr(game, "howto", "") or "").strip()
            if howto and kind != "dm":
                msgs.append(howto)
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
        self._table_event(room, "started")
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
                self.store.get_for(player).items)
        except Exception as e:  # noqa: BLE001
            record_failure("_mirror_player:inventory", e)
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
        except Exception as e:  # noqa: BLE001
            record_failure("_mirror_player:gear", e)
            room.state.setdefault("loadout", {}).setdefault(key, {})
            room.state.setdefault("gear_closet", {}).setdefault(key, [])
        # persistent progression: level + arena stat bonus per human,
        # so games apply it store-free (see progression.level_stat_bonus)
        try:
            from .progression import level_for_xp, level_stat_bonus
            prof = self.store.get_for(player)
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
                prof = self.store.get_for(player)
                _, stats = store.grant_level_points(
                    key, level_for_xp(prof.xp))
            except Exception as e:  # noqa: BLE001
                record_failure("_mirror_player:level_points", e)
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
        self._table_event(room, "joined")
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
        # wager integrity: a departing player's escrowed money is
        # cashed out BEFORE the seat drops — the finish below only
        # settles players still seated, so an uncashed escrow would
        # be silently kept.
        wager_note = None
        if game is not None and getattr(game, "takes_wagers", False):
            try:
                wager_note = game.wager_leave(room, player)
            except Exception as e:  # noqa: BLE001
                record_failure("_leave_inner:wager_leave", e)
        room.players = [p for p in room.players if p.key != player.key]
        if room.turn >= len(room.players):
            room.turn = 0 % max(1, len(room.players))
        # is_over: a game may end itself in on_leave (duel walkover) —
        # honor that instead of leaving a decided table open
        if (not room.humans or len(room.players) <= len(room.ai_seats)
                or self.is_over(room)):
            extra = "\n".join(x for x in (notice, wager_note) if x) \
                or None
            return self._finish(room, extra)
        room.turn_started = time.time()
        room.last_activity = time.time()
        self._persist(room)
        msgs = []
        if notice:
            msgs.append(notice)
        if wager_note:
            msgs.append(wager_note)
        self._emit(room, *msgs)
        self._pump_ai(room, [])
        self._persist(room)
        self._table_event(room, "left")
        return msgs

    def _resign_inner(self, room: Room, game: MultiGame, chat_key: str,
                      player: Player) -> list[str]:
        """An explicit resignation: a recorded loss, distinct from
        walking away (``leave`` drops the seat; a resign DECIDES
        the game against the resigner). Caller holds ``room.guard``
        (via ``_move_inner``).

        With 2+ other humans at the table the resigner is
        eliminated and play continues; otherwise the game ends and
        the remaining human takes it. The decision is recorded in
        ``state['decided']`` and honored by the settle over the
        game's own winner() read — the same discipline as the §34
        forfeit record.
        """
        if room.player(player.key) is None or player.is_ai:
            return [f"{player.name} isn't seated at this table."]
        # wager integrity mirrors leave: escrow is cashed out
        # before the seat drops, never silently kept.
        wager_note = None
        if getattr(game, "takes_wagers", False):
            try:
                wager_note = game.wager_leave(room, player)
            except Exception as e:  # noqa: BLE001
                record_failure("_resign_inner:wager_leave", e)
        self._journal_clock(room, "resign", player)
        self._journal_event(room, "resign", player)
        resigned = room.state.setdefault("_resigned", [])
        if player.key not in resigned:
            resigned.append(player.key)
        line = f"🏳 {player.name} resigns."
        others = [p for p in room.players if p.key != player.key]
        other_humans = [p for p in others if not p.is_ai]
        if len(other_humans) >= 2:
            # elimination: the decision binds the resigner alone;
            # the winner half stays empty so a later natural
            # finish is not hijacked by this record.
            room.state["decided"] = {
                "how": "resign", "loser": player.key, "winner": ""}
            idx = room.players.index(player)
            room.players = others
            if room.turn > idx:
                room.turn -= 1
            if room.turn >= len(room.players):
                room.turn = 0
            room.turn_started = time.time()
            room.last_activity = time.time()
            out = [line + f" {player.name} is out of the game."]
            if wager_note:
                out.append(wager_note)
            self._pump_ai(room, out)
            if self.is_over(room):
                out.extend(self._finish(room))
            else:
                self._persist(room)
            return out
        # the resign decides the game: the remaining human takes
        # it, or the house seat when no human remains.
        winner_key = ""
        if len(other_humans) == 1:
            winner_key = other_humans[0].key
        elif others:
            winner_key = others[0].key
        room.state["decided"] = {
            "how": "resign", "loser": player.key, "winner": winner_key}
        if wager_note:
            line = line + "\n" + wager_note
        return self._finish(room, line)

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
            out = self._move_inner(room, game, chat_key, text, sender)
            if room.status == "active":
                self._table_event(room, "moved")
            return out

    def move_fenced(self, chat_key: str, text: str, sender: Player,
                    *, revision: int, kind: str = "dm") -> FencedMove:
        """A move fenced against the table revision it was cut at.

        Card buttons and other revision-carrying actions route
        here: a tap cut at a superseded revision is refused —
        nothing applies — and the answer carries the CURRENT
        table state. Identity is enforced the same way: a sender
        with no seat is ``not_player``; a seated player moving out
        of turn is ``not_turn``. Engine commands (status/help/…)
        are not turn moves and pass straight through.
        """
        room = self.live(chat_key)
        if room is None:
            return FencedMove("missing",
                              ["no game is live here."], None)
        if room.status != "active":
            return FencedMove("finished",
                              ["that game is over."], room)
        cmd, _ = parse_command(text)
        if not cmd:
            if int(revision) != room.revision:
                return FencedMove(
                    "stale",
                    ["that action was cut at an older table state — "
                     "nothing was played. Current table:",
                     self.describe(room)], room)
            if room.player(sender.key) is None or sender.is_ai:
                return FencedMove(
                    "not_player",
                    [f"{sender.name} isn't seated at this table."],
                    room)
            if not self._is_turn(room, sender):
                cur = room.current
                return FencedMove(
                    "not_turn",
                    [f"it's not {sender.name}'s turn — waiting on "
                     f"{cur.name if cur else '…'}.",
                     self.describe(room)], room)
        out = self.move(chat_key, text, sender, kind=kind)
        return FencedMove("ok", out, room)

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
            # a deliberate pass is presence, not a stall (§34)
            (room.state.get("_stalls") or {}).pop(sender.key, None)
            room.state["passes"] = int(room.state.get("passes") or 0) + 1
            self._journal_event(room, "pass", sender)
            self._emit(room, f"{sender.name} passes.")
            room.advance_turn()
            out: list[str] = []
            self._pump_ai(room, out)
            if self.is_over(room):
                out.extend(self._finish(room))
            else:
                self._persist(room)
            return _collapse_dup_lines(out) or []
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
                            self.store.get_for(sender).items)
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
            prof = self.store.get_for(sender)
            items = ", ".join(f"{k}×{v}" for k, v in prof.items.items()) or "none"
            return [f"🪙 {prof.coins} coins · {prof.points} points · items: {items}"]
        if cmd == "leave":
            return self.leave(chat_key, sender)
        if cmd == "resign":
            return self._resign_inner(room, game, chat_key, sender)
        if cmd == "game":
            return []  # control commands are the runtime's, not the game's

        # an actual move — only the current seat (or the solo human in a DM)
        if not self._is_turn(room, sender):
            cur = room.current
            if room.kind == "group" and cur is not None and not cur.is_ai:
                # only nudge when the text looks like a move attempt —
                # casual chat ("say that again?") stays silent instead
                # of spamming "waiting on X"
                try:
                    if game.is_move_text(text):
                        return [f"waiting on {cur.name} — your move goes in after "
                                "theirs. (/pass /status /help)"]
                except Exception as e:  # noqa: BLE001 - a bad hook never breaks chat
                    record_failure("_move_inner:is_move_text", e)
                return []
            return []
        out: list[str] = []
        # the current player showed up and played — their §34
        # consecutive-stall count resets (presence, not absence).
        (room.state.get("_stalls") or {}).pop(sender.key, None)
        self._journal_event(room, "move", sender, text)
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
        # the DM narrates what just happened (cooldown-guarded).
        # Safe wrapper: narration must never break the move itself.
        self._drain_dm_feed_safe(room, game, out)
        if self.is_over(room):
            out.extend(self._finish(room))
        else:
            room.advance_turn()
            self._pump_ai(room, out)
            if self.is_over(room):
                out.extend(self._finish(room))
            else:
                self._persist(room)
        return _collapse_dup_lines(out)

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
            # agent seats (characters, brain) play via their registered
            # decider; house seats use the game's built-in ai_turn.
            decider = None
            for prefix, fn in self.seat_deciders.items():
                if cur.key.startswith(prefix):
                    decider = fn
                    break
            try:
                if decider is not None:
                    msgs = decider(room, cur, game, self._mind)
                    if msgs is None:
                        msgs = game.ai_turn(room, self._mind)
                else:
                    msgs = game.ai_turn(room, self._mind)
                out.extend(msgs or [])
            except Exception as exc:  # noqa: BLE001
                _log.exception("ai turn failed: %s", room.game)
                out.append(
                    f"⚠️ the house ({game.name} AI) errored "
                    f"({type(exc).__name__}) — seat skipped.")
            # did the game make progress? (removed the AI seat, changed
            # state, or advanced). If nothing moved at all, bail out.
            before = (room.turn, room.status)
            room.advance_turn()
            after = (room.turn, room.status)
            if before == after and not room.state:
                room.status = "finished"
                out.append("the house got stuck — closing the table.")
                return

    @staticmethod
    def _journal_clock(room: Room, event: str, player: Player,
                       **fields: Any) -> None:
        """Append one turn-clock event to the room's held journal.

        The journal rides ``room.state`` (persisted with the room,
        capped) — stall / pass / forfeit facts live in state, never
        in member-visible text beyond the board lines themselves.
        """
        try:
            journal = room.state.setdefault("_clock_events", [])
            entry = {"event": event, "player": player.key,
                     "ts": time.time()}
            entry.update(fields)
            journal.append(entry)
            del journal[:-50]
        except Exception:  # noqa: BLE001 - journaling never breaks a tick
            _log.debug("clock journal failed", exc_info=True)

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
            self._journal_clock(room, "timeout", cur)
            # §34 turn-clock law: games riding the DEFAULT on_timeout
            # get engine-counted consecutive stalls and a forfeit at
            # the game's stall_limit. Games with their own on_timeout
            # own their timeout semantics (pvp counts misses to a
            # forfeit itself; the raid auto-plays) — untouched here.
            law = (type(game).on_timeout is MultiGame.on_timeout
                   and int(getattr(game, "stall_limit", 0) or 0) > 0)
            stalls: dict[str, Any] = {}
            count = 0
            if law:
                stalls = room.state.setdefault("_stalls", {})
                count = int(stalls.get(cur.key, 0)) + 1
                stalls[cur.key] = count
                self._journal_clock(room, "stall", cur, n=count)
                if count >= int(game.stall_limit):
                    for m in self._forfeit_staller(room, game, cur,
                                                   count):
                        self._emit(room, m)
                    return
            msgs = game.on_timeout(room, cur, self._mind)
            if law:
                self._journal_clock(room, "pass", cur, n=count)
                msgs.append(
                    f"⏳ stall {count}/{int(game.stall_limit)} for "
                    f"{cur.name} — reach the limit and the game is "
                    f"forfeited.")
            if self.is_over(room):
                msgs.extend(self._finish(room))
            else:
                self._pump_ai(room, msgs)
                if self.is_over(room):
                    msgs.extend(self._finish(room))
                else:
                    # the board re-renders on a pass: whose turn it
                    # is now, in the game's own surface words.
                    nxt = room.current
                    if nxt is not None and not nxt.is_ai:
                        try:
                            board = game.describe_state(room)
                        except Exception:  # noqa: BLE001
                            _log.debug("timeout board render failed",
                                       exc_info=True)
                            board = ""
                        msgs.append(board or game.status_line(room))
                    self._persist(room)
            for m in msgs:
                self._emit(room, m)
            if room.status == "active":
                self._table_event(room, "timeout")

    def _forfeit_staller(self, room: Room, game: MultiGame,
                         staller: Player, count: int) -> list[str]:
        """§34 forfeit: a player who stalled ``stall_limit`` turns in
        a row loses. Caller holds ``room.guard``; returns the lines
        to emit (a finish composes its own single settle message).

        With 2+ other humans at the table the staller is eliminated
        and play continues; otherwise the game ends — the remaining
        human (or the house seat) takes it, recorded in room state
        so the settle honors the forfeit over the game's own
        winner() read.
        """
        self._journal_clock(room, "forfeit", staller, n=count)
        line = (f"⏰ {staller.name} stalled {count} turns in a row — "
                f"forfeit.")
        others = [p for p in room.players if p.key != staller.key]
        other_humans = [p for p in others if not p.is_ai]
        if len(other_humans) >= 2:
            idx = room.players.index(staller)
            room.players = others
            if room.turn > idx:
                room.turn -= 1
            if room.turn >= len(room.players):
                room.turn = 0
            room.turn_started = time.time()
            out = [line + f" {staller.name} is out of the game."]
            self._pump_ai(room, out)
            if self.is_over(room):
                out.extend(self._finish(room))
            else:
                self._persist(room)
            return out
        room.state["forfeit_loser"] = staller.key
        if len(other_humans) == 1:
            room.state["forfeit_winner"] = other_humans[0].key
        elif others:
            room.state["forfeit_winner"] = others[0].key
        # the unified decision record (resign writes the same
        # shape); the legacy forfeit_* keys stay for compat.
        room.state["decided"] = {
            "how": "forfeit", "loser": staller.key,
            "winner": str(room.state.get("forfeit_winner") or "")}
        return self._finish(room, line)

    def _finish_idle(self, room: Room) -> None:
        """Close a table the players abandoned: re-check liveness under
        the lock (it may have been quit after the sweep selected it),
        then finish it with an idle notice."""
        with self._lock:
            if (self._rooms.get(room.chat_key) is not room
                    or room.status != "active"):
                return
        # no caller waits on this finish — the engine delivers the
        # settle itself (one joined message, as composed).
        for text in self._finish(room, "the table closed — idle too long."):
            self._emit(room, text)

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
            msgs, rematch_mem, fair_mem = self._finish_inner(room, extra)
        if getattr(room, "_settle_aborted", False):
            # the settle rolled back and the room was restored to
            # active (see _finish_inner): it stays live so the
            # recovery sweep can re-finish it. Do NOT pop it, do
            # NOT remember it as a finished table.
            return msgs
        with self._lock:
            self._rooms.pop(room.chat_key, None)
            self._by_id.pop(room.id, None)
            # remember the table for /game rematch (relay virtual rooms
            # are excluded — a rematch there needs a fresh invite)
            if rematch_mem is not None:
                self._last_game[room.chat_key] = rematch_mem
            if fair_mem is not None:
                self._last_fair[room.chat_key] = fair_mem
        self._table_event(room, "finished")
        return msgs

    def _finish_inner(self, room: Room,
                      extra: str | None = None
                      ) -> tuple[list[str], tuple | None, tuple | None]:
        """The body of :meth:`_finish`. Caller holds ``room.guard``.

        Returns (messages, rematch_memory, fair_memory) — the wrapper
        stores the memories under the engine lock after releasing the
        guard, keeping lock order engine → room everywhere.

        The settle is ONE message: every finish line is composed here
        and joined into a single text, returned — never emitted line
        by line. Callers own delivery (the chat reply, the timeout
        pump's emit, the idle close), so a finish reaches the chat
        exactly once on every path.
        """
        game = self.games.get(room.game)
        if room.status == "finished":
            return ([extra] if extra else [], None, None)
        room.status = "finished"
        room._settle_aborted = False
        msgs: list[str] = []
        card_scores: dict[str, int] = {}
        season_results: list[tuple[str, str, int, bool]] = []
        winner_label = ""
        fair_mem: tuple[str, dict[str, Any]] | None = None
        # ── settlement ledger (games/settlement.py) ──
        # plan first: the row exists BEFORE any credit moves, so a
        # crash leaves a durable retry signal. An 'applied' row
        # means this finish is a REPLAY of a committed settle: the
        # credit loop resolves to the stored receipts and writes
        # nothing. A db-less rig runs the legacy settle (no ledger,
        # no transaction) exactly as before.
        settle_store = self._settlement_store() \
            if self.db is not None else None
        settle_rec = settle_store.get(room.id) \
            if settle_store is not None and settle_store.available \
            else None
        settle_replay = bool(
            settle_rec and settle_rec["status"] == "applied")
        settle_active = False
        if settle_store is not None and settle_store.available \
                and not settle_replay:
            settle_active = settle_store.begin(
                room.id, room.game, room.chat_key)
        settle_receipts: dict[str, Any] = {"players": {}}
        sink_due: list[tuple[Player, int]] = []
        settle_cm: Any = None
        settle_aborted = False
        if extra:
            msgs.append(extra)
        if game is not None:
            try:
                msgs.append(game.final_message(room, self._mind))
            except Exception as e:  # noqa: BLE001
                record_failure("_finish_inner:final_message", e)
                msgs.append("game over.")
            # provably fair: the settle carries the hash-free note
            # (2026-11-06: the commitment hash never prints — it is
            # an internal id). The full reveal (server seed +
            # recipe) is served ON DEMAND from the retained
            # fragment — /game fair via last_fair_reveal() — never
            # blasted into the chat after every game.
            if room.state.get("fair") is not None:
                try:
                    from .fairness import commit_note
                    fair_mem = (room.game, dict(room.state["fair"]))
                    msgs.append(commit_note(room.state["fair"]))
                except Exception:  # noqa: BLE001
                    _log.debug("fair commit line failed", exc_info=True)
            # the DM gets the last word: drain any remaining feed, then
            # narrate the finale for games that opted into one.
            # Safe wrapper: narration must never break the quit path.
            self._drain_dm_feed_safe(room, game, msgs, force=True)
            if getattr(game, "dm_finale", False):
                try:
                    evt = game.dm_finale_event(room)
                    if evt:
                        # one narration per moment: if the move drain
                        # (or the drain just above) already narrated
                        # this ending — same event text, or any
                        # narration while the game was over — the
                        # finale does not tell it again.
                        narrated = room.state.get("_dm_narrated") or []
                        already = bool(room.state.get("_dm_end_narrated")) \
                            or evt.strip() in narrated
                        if not already:
                            line = self.dm.announce(
                                game.name, [{"text": evt, "big": True}])
                            if line:
                                msgs.append(line)
                                narrated = list(narrated)
                                narrated.append(evt.strip())
                                room.state["_dm_narrated"] = narrated
                                room.state["_dm_end_narrated"] = True
                except Exception:  # noqa: BLE001
                    _log.debug("dm finale failed", exc_info=True)
            # victory loot: games with exclusive drops (raid bosses)
            # roll them here; grants go through GearStore so pieces
            # persist like any other gear. A settlement REPLAY grants
            # nothing again — the original settle already did.
            if not settle_replay:
                try:
                    for line in self._grant_victory_loot(room, game):
                        msgs.append(line)
                except Exception:  # noqa: BLE001
                    _log.debug("victory loot failed", exc_info=True)
            # the settle transaction: every games-wallet write
            # below (credits, XP, achievements, ranked) joins ONE
            # transaction with the settlement flip, so the settle
            # lands whole or not at all. Entered by hand because
            # the block it wraps predates it; exited explicitly on
            # both paths below.
            if settle_active:
                settle_cm = self.db.transaction()
                settle_cm.__enter__()
            try:
                winner = game.winner(room)
                # §34 forfeit: the turn-clock law already decided
                # this table (see _forfeit_staller) — the settle
                # honors the forfeit over the game's winner() read.
                fw = (room.state or {}).get("forfeit_winner")
                if fw:
                    fp = room.player(str(fw))
                    if fp is not None:
                        winner = fp
                # the unified decision record (forfeit AND resign
                # write it): a recorded decision outranks winner().
                dec = (room.state or {}).get("decided") or {}
                if dec.get("winner"):
                    dp = room.player(str(dec["winner"]))
                    if dp is not None:
                        winner = dp
                winner_label = getattr(winner, "name", None) or (
                    str(winner) if winner not in (None, "draw") else "")
                # ranked 1v1: move the ELO board (queue matches and
                # hand-arranged duels alike) — never twice for one
                # game, so a settlement replay skips it.
                if not settle_replay:
                    try:
                        from .matchmaking import maybe_record_ranked
                        for line in maybe_record_ranked(
                                self.db, room, game, winner):
                            msgs.append(line)
                    except Exception:  # noqa: BLE001
                        _log.debug("ranked record failed", exc_info=True)
                for p in list(room.players):
                    if p.is_ai:
                        continue
                    if settle_replay:
                        # resolve to the ORIGINAL receipt: the coin
                        # line is re-issued verbatim; nothing writes.
                        rec = (settle_rec["receipts"].get("players")
                               or {}).get(p.key) or {}
                        r_coins = int(rec.get("coins") or 0)
                        card_scores[p.key] = int(rec.get("score") or 0)
                        msgs.append(
                            f"🪙 {p.name}: {r_coins:+} coins "
                            f"({rec.get('why') or 'settled'})")
                        continue
                    won = self._settle_won(game, room, p, winner)
                    if (room.state or {}).get("forfeit_loser") == p.key \
                            or dec.get("loser") == p.key \
                            or p.key in ((room.state or {})
                                         .get("_resigned") or []):
                        won = False  # forfeit/resign is a loss, full stop
                    points = GameEconomy.reward_points(won, game.score(room, p))
                    # quarterly season (AA-16): this result folds
                    # into the group's season board after the settle
                    # (points only — prizes are the season tick's
                    # job, paid through the group economy).
                    season_results.append(
                        (p.name,
                         p.key.split(":", 1)[1] if ":" in p.key else "",
                         int(points), bool(won)))
                    score = game.score(room, p)
                    card_scores[p.key] = score
                    try:
                        difficulty = game.difficulty(room)
                    except Exception as e:  # noqa: BLE001
                        record_failure("_finish_inner:difficulty", e)
                        difficulty = "normal"
                    try:
                        pre = self.store.get_for(p)
                        streak_after = pre.streak + 1 if won is True else pre.streak
                    except Exception as e:  # noqa: BLE001
                        record_failure("_finish_inner:streak", e)
                        streak_after = 1 if won is True else 0
                    # wager games settle REAL money through the
                    # room's bound purse — the wager settlement IS
                    # the coin delta (negative on a loss).  No
                    # performance formula, no season multiplier, no
                    # charm, no sink: the purse already journaled
                    # every leg, and record_outcome below takes
                    # coins=0 so progression (points/XP/streaks)
                    # moves without touching money twice.
                    wager_net: tuple[int, str] | None = None
                    if getattr(game, "takes_wagers", False):
                        try:
                            wager_net = game.wager_settle(room, p)
                        except Exception as e:  # noqa: BLE001
                            record_failure(
                                "_finish_inner:wager_settle", e)
                            wager_net = (
                                0, "wager held — settle errored, the "
                                   "escrow stays on the table")
                    if wager_net is not None:
                        coins, coin_why = wager_net
                    else:
                        coins, coin_why = self._settle_coins(
                            game, room, p, won, score, difficulty,
                            streak_after)
                        # seasonal event: the week's multiplier on coins
                        try:
                            from .seasons import apply_event
                            coins, _season_note = apply_event(
                                self.db, room.game, "coins", coins)
                            coin_why += _season_note
                        except Exception:  # noqa: BLE001
                            _log.debug("season coin mult failed",
                                       exc_info=True)
                        # boosters: coin charm is consumed on use
                        try:
                            prof_items = self.store.get_for(p).items
                            if int(prof_items.get("coin_charm", 0)) > 0:
                                coins = int(coins * 1.5)
                                coin_why += " + coin charm"
                                msgs.append("✨ Coin Charm consumed!")
                                self.store.consume_item(p, "coin_charm")
                        except Exception:  # noqa: BLE001
                            _log.debug("booster apply failed",
                                       exc_info=True)
                    prof_after = self.store.record_outcome(
                        p, won=won, game=room.game, points=points,
                        coins=0 if wager_net is not None else coins,
                        score=score)
                    msgs.append(f"🪙 {p.name}: {coins:+} coins "
                                f"({coin_why})")
                    # the receipt this settle will be remembered by:
                    # a replayed finish re-issues these exact numbers
                    # and moves nothing again.
                    settle_receipts["players"][p.key] = {
                        "name": p.name, "coins": int(coins),
                        "why": coin_why, "points": int(points),
                        "score": int(score),
                        "won": (True if won is True else
                                False if won is False else None),
                        "sink_done": False}
                    if wager_net is not None:
                        # the health ledger records the NET flow
                        # (the purse journaled the legs themselves)
                        try:
                            from .economy import record_ledger
                            if coins > 0:
                                record_ledger(self.db, p.key, "source",
                                              coins, f"game:{room.game}")
                            elif coins < 0:
                                record_ledger(self.db, p.key, "sink",
                                              -coins, f"game:{room.game}")
                        except Exception:  # noqa: BLE001
                            _log.debug("wager ledger record failed",
                                       exc_info=True)
                    else:
                        # one purse (group build P2): coins won in a
                        # GROUP game also land in that group's economy
                        # purse for the member — the runtime installs
                        # the sink; the engine only reports the win.
                        # Wager games skip this: their settlement
                        # already landed in the bound purse, and a
                        # sink pass would move the money twice.
                        # the group-purse mirror is a SECOND store:
                        # it cannot join the settle transaction, so
                        # the obligation is collected here and fired
                        # after the commit (see the sink block after
                        # the credit loop), marked per player in the
                        # receipts so recovery fires only what's owed.
                        if coins > 0 and room.kind == "group" \
                                and not p.is_ai:
                            sink_due.append((p, int(coins)))
                        # economy ledger: every faucet tagged for the
                        # health report (sources/sinks, inflation watch)
                        try:
                            from .economy import record_ledger
                            if coins > 0:
                                record_ledger(self.db, p.key, "source",
                                              coins, f"game:{room.game}")
                        except Exception:  # noqa: BLE001
                            _log.debug("ledger record failed",
                                       exc_info=True)
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
                        # seasonal event: the week's multiplier on XP
                        season_xp_note = ""
                        try:
                            from .seasons import apply_event
                            amount, season_xp_note = apply_event(
                                self.db, room.game, "xp", amount)
                        except Exception:  # noqa: BLE001
                            _log.debug("season xp mult failed",
                                       exc_info=True)
                        # daily hunt: the first arena win of the day
                        # pays double XP
                        if room.game == "arena" and won is True:
                            try:
                                from .daily import (bump_streak,
                                                    complete_daily_hunt)
                                if complete_daily_hunt(self.db, p.key):
                                    amount *= 2
                                    msgs.append(
                                        "🎯 daily hunt complete! "
                                        "double XP today.")
                                    st = bump_streak(self.db, p.key)
                                    if st.get("milestone"):
                                        msgs.append(f"🏆 {st['milestone']}")
                                    elif not st.get("continued", True):
                                        msgs.append(
                                            "🔥 new streak started — "
                                            "come back tomorrow.")
                            except Exception:  # noqa: BLE001
                                _log.debug("daily hunt failed",
                                           exc_info=True)
                        # Double XP Charm: consumed for the next finished game
                        try:
                            _prof_items = self.store.get_for(p).items
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
                            prof = self.store.get_for(p)
                            msgs.append(
                                f"+{amount} XP {xp_bar(prof.xp)}"
                                f"{season_xp_note}")
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
                if getattr(game, "takes_wagers", False):
                    # No "— /game leaderboard" text tail (2026-11-06):
                    # a result's next actions are declared buttons
                    # at the funnel, not hints printed in the wall.
                    msgs.append(
                        "📈 points credited · wagers settled in the "
                        "purse"
                    )
                else:
                    msgs.append(
                        f"📈 points and coins credited — /game leaderboard"
                    )
            except Exception as exc:  # noqa: BLE001
                if settle_cm is not None:
                    # a db-backed settle failed mid-apply: roll the
                    # WHOLE settle back (nobody is paid), keep the
                    # plan row as the retry signal, and surface the
                    # failure instead of the old silent half-pay.
                    settle_cm.__exit__(type(exc), exc,
                                       exc.__traceback__)
                    settle_cm = None
                    settle_aborted = True
                    record_failure("_finish_inner:settle", exc)
                else:
                    _log.debug("game ledger credit failed",
                               exc_info=True)
        if settle_aborted:
            # restore the room to active and persist THAT (the
            # finished write never happens): the table stays open,
            # the plan row stays 'planned', and the recovery sweep
            # re-runs this finish once the fault clears. The
            # wrapper sees _settle_aborted and keeps the room live.
            room.status = "active"
            room._settle_aborted = True
            self._persist(room)
            return (["⚠ the settle failed — nothing was paid. "
                     "the table stays open; the settle will retry "
                     "on its own."], None, None)
        if settle_cm is not None:
            # commit the settle: the receipts flip lands in the
            # SAME transaction as every credit above.
            settle_store.mark_applied(room.id, settle_receipts)
            settle_cm.__exit__(None, None, None)
            settle_cm = None
        # post-commit group-purse mirrors: fire each due sink once
        # and mark it in the receipts. On a replay, fire only what
        # the receipts say is still owed. A db-less settle fires
        # its dues directly (legacy behavior, nothing to mark).
        if sink_due or settle_replay:
            self._fire_sink_dues(room, sink_due, settle_rec,
                                 settle_receipts, settle_store)
        # quarterly season board (AA-16): group-room results only —
        # a season is a group's standing table, DM tables don't feed
        # it. Fail-soft like every other settle bridge: a season
        # write must never sink a finished game.
        if season_results and room.kind == "group" \
                and not settle_replay:
            try:
                from .tournaments_seasons import record_season_results
                record_season_results(room.chat_key, room.game,
                                      season_results, context=self.context)
            except Exception:  # noqa: BLE001
                _log.debug("season record failed", exc_info=True)
        if not settle_replay:
            try:
                self._reconcile_items(room)
            except Exception:  # noqa: BLE001
                _log.debug("game item reconcile failed", exc_info=True)
        self._persist(room)
        # the final table card: standings, scores, balances, tier moves
        try:
            if game is not None and card_scores:
                # wager games: the card's balance is the bound
                # purse's POST-SETTLEMENT balance (the purse that
                # actually moved), not the games wallet's.
                card_balances = None
                if getattr(game, "takes_wagers", False) \
                        and room.purse is not None:
                    card_balances = {
                        p.key: room.purse.balance(p)
                        for p in room.players if not p.is_ai}
                card = render_game_over_card(
                    room.game, list(room.players), card_scores,
                    winner_label=winner_label, store=self.store,
                    db=self.db, balances=card_balances)
                if card:
                    msgs.append(card)
        except Exception:  # noqa: BLE001
            _log.debug("game over card failed", exc_info=True)
        # the settle leaves as ONE joined message (see docstring):
        # record it in the room history exactly as _emit would have,
        # then hand it to the caller as a single-element list.
        settle = "\n".join(_collapse_dup_lines([m for m in msgs if m]))
        if settle:
            room.messages.append(settle[:4000])
            room.messages = room.messages[-40:]
        # remember the table for /game rematch (relay virtual rooms are
        # excluded — a rematch there needs a fresh invite). The wrapper
        # stores this under the engine lock; we only compute it here.
        rematch_mem = None
        if not room.chat_key.startswith("relay:"):
            humans = [p for p in room.players if not p.is_ai]
            if humans:
                rematch_mem = (room.game, humans, room.kind)
        return ([settle] if settle else []), rematch_mem, fair_mem

    def _fire_sink_dues(self, room: Room,
                        sink_due: list[tuple[Player, int]],
                        settle_rec: dict[str, Any] | None,
                        settle_receipts: dict[str, Any],
                        settle_store: Any) -> None:
        """Fire the group-purse mirror credits a settle owes.

        The sink is a second store (the group economy) and cannot
        join the settle transaction, so it runs post-commit: each
        due credit fires once and is marked ``sink_done`` in the
        receipts. A replayed finish fires only the dues the
        receipts still show as owed. With no sink installed this
        is a no-op, exactly as the inline call was.
        """
        sink = getattr(self, "coin_sink", None)
        if sink is None:
            return
        receipts: dict[str, Any] | None = None
        store = None
        if settle_rec is not None \
                and settle_rec.get("status") == "applied":
            # replay: what's owed is read back from the receipts
            receipts = settle_rec["receipts"]
            store = settle_store
            due: list[tuple[Player, int]] = []
            for key, rec in (receipts.get("players") or {}).items():
                if int(rec.get("coins") or 0) > 0 \
                        and not rec.get("sink_done"):
                    p = room.player(key)
                    if p is not None and not p.is_ai:
                        due.append((p, int(rec["coins"])))
        else:
            due = list(sink_due)
            if settle_store is not None and settle_store.available:
                receipts = settle_receipts
                store = settle_store
        changed = False
        for p, amount in due:
            try:
                sink(room.chat_key, p, int(amount), room.game)
            except Exception:  # noqa: BLE001 - bridge is best-effort
                _log.debug("coin sink failed", exc_info=True)
                continue
            if receipts is not None:
                rec = (receipts.get("players") or {}).get(p.key)
                if rec is not None and not rec.get("sink_done"):
                    rec["sink_done"] = True
                    changed = True
        if changed and store is not None:
            store.save_receipts(room.id, receipts)

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
        if isinstance(winner, str):
            # A string outside the winner() contract ("draw"/"all"
            # returned above) — the census pin should have caught the
            # game at build time. Journal it so the violation is
            # visible in the logs instead of silently paying a draw;
            # a draw stays the least-wrong runtime settle.
            _log.debug("unknown winner value %r from game %s — "
                       "settling as draw", winner,
                       getattr(game, "name", "?"))
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

    def last_fair(self, chat_key: str) -> tuple[str, dict[str, Any]] | None:
        """The last finished fair table in this chat: (game, fragment).

        The fragment is a copy — callers may render or mutate it
        without touching the retained record.
        """
        mem = self._last_fair.get(chat_key)
        if mem is None:
            return None
        game, fair = mem
        return game, dict(fair)

    def last_fair_reveal(self, chat_key: str) -> str | None:
        """The on-demand fairness reveal for the last finished fair
        table in this chat (server seed + verification recipe), or
        None when no fair table has finished here."""
        mem = self.last_fair(chat_key)
        if mem is None:
            return None
        game, fair = mem
        try:
            from .fairness import reveal_block

            return f"🎲 {game} — last table\n" + reveal_block(fair)
        except Exception:  # noqa: BLE001
            _log.debug("last fair reveal failed", exc_info=True)
            return None

    def rematch(self, chat_key: str, difficulty: str = "normal",
                ) -> tuple[Room | None, list[str]]:
        """Start the last finished game again with the same humans.

        Returns (room, messages) or (None, [why-not]). ``difficulty``
        is handed to :meth:`start` (only games that declare a named
        ``difficulty`` parameter on ``new_state`` see it).
        """
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
            room, msgs = self.start(chat_key, game_name, host, kind=kind,
                                    difficulty=difficulty)
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
            prof = self.store.get_for(player)
        except Exception as e:  # noqa: BLE001
            record_failure("_award_achievements:profile", e)
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
                except Exception as e:  # noqa: BLE001
                    record_failure("_award_achievements:myth_set", e)
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
                except Exception as e:  # noqa: BLE001
                    record_failure("_award_achievements:no_gear", e)
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
            except Exception as e:  # noqa: BLE001
                record_failure("_award_achievements:loss_streak_reset", e)
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
                    except Exception as e:  # noqa: BLE001
                        record_failure("_award_achievements:wins_100", e)
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
    def table_view(self, room: Room,
                   viewer_key: str | None = None) -> dict[str, Any]:
        """One player's entitled view of the table (JSON-able).

        The state inside is the game's REDACTED view for this
        viewer (see ``MultiGame.view_state``): a per-player secret
        appears only under its owner's own key in their own view,
        and in nobody else's — the shared view (``viewer_key``
        None) carries no secrets at all.
        """
        from .table import game_family, stakes_label, timeout_policy

        game = self.games.get(room.game)
        view_state = game.view_state(room, viewer_key) \
            if game is not None else dict(room.state or {})
        cur = room.current
        deadline = None
        if game is not None and game.move_timeout > 0 \
                and room.status == "active":
            deadline = room.turn_started + game.move_timeout
        return {
            "room_id": room.id, "game": room.game,
            "family": game_family(game) if game is not None else "table",
            "status": room.status, "kind": room.kind,
            "revision": room.revision,
            "players": [{"key": p.key, "name": p.name,
                         "is_ai": bool(p.is_ai)} for p in room.players],
            "turn": ({"key": cur.key, "name": cur.name}
                     if cur is not None else None),
            "deadline_at": deadline,
            "stakes": stakes_label(game, room) if game is not None
            else "",
            "policy": timeout_policy(game) if game is not None else "",
            "state": view_state,
            "persist": self.persist_health(),
        }

    def replay_text(self, chat_key: str) -> str | None:
        """The replay of this chat's most recent finished game,
        built from its real event record — or None when no
        finished game is on record here."""
        if self.db is None:
            return None
        try:
            row = self.db.query_one(
                "SELECT * FROM game_rooms WHERE chat_key = ? "
                "AND status = 'finished' ORDER BY updated_at DESC "
                "LIMIT 1", (chat_key,))
        except Exception as e:  # noqa: BLE001
            record_failure("replay_text:db_query", e)
            return None
        if row is None:
            return None
        from .replay import build_replay

        rec = None
        store = self._settlement_store()
        if store is not None and store.available:
            rec = store.get(row["id"])
        return build_replay(row, rec)

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
            except Exception as e:  # noqa: BLE001
                record_failure("describe:state", e)
        cur = room.current
        if cur is not None and not cur.is_ai and room.status == "active":
            lines.append(f"turn: {cur.name}")
        health = self.persist_health()
        if health["degraded"]:
            # OFF vs BROKEN: a table whose saves are failing must
            # not look like a healthy one — the next restart would
            # lose it, and the players deserve to know now.
            lines.append(
                f"⚠ table saves failing (last: {health['last_error']}) "
                "— this game may not survive a restart")
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
        lines.append("  /game profile [name]       your full character sheet")
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
