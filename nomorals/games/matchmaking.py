"""Native matchmaking: queue up, get paired, duel.

``/game queue <game>`` puts you in line. Every ~20s the matcher pairs
waiting players by ELO — the tolerance widens the longer you wait, so a
quiet week still finds you a game — then auto-creates a relay duel
between the two chats (both queued = both consented; no invites to
strangers, no external service).

Ratings are per-game ELO (start 1000, K=32), updated on every ranked
1v1 finish — queue matches and hand-arranged duels alike. ``/game
ratings [game]`` shows the board.

Ranked games are the strict 1v1 tables (see ``RANKED_GAMES``); the queue
only serves games built for exactly two humans (``QUEUE_GAMES``).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("nomorals.games.matchmaking")

__all__ = [
    "QUEUE_GAMES",
    "RANKED_GAMES",
    "Matchmaker",
    "get_rating",
    "match_status",
    "maybe_record_ranked",
    "record_elo",
    "render_ratings",
]

#: games the queue will match: verified true 1v1 human tables (each
#: human plays their own side — most board games are human-side vs
#: house-side, so they can't host a real duel).
QUEUE_GAMES = ("pvp", "connect4")

#: games whose 1v1 human finishes move ELO (a superset of the queue).
RANKED_GAMES = QUEUE_GAMES

START_ELO = 1000
K_FACTOR = 32
SWEEP_INTERVAL = 20.0


def _ensure(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS match_queue ("
            "player_key TEXT PRIMARY KEY, "
            "chat_key TEXT NOT NULL, "
            "player_name TEXT NOT NULL DEFAULT '', "
            "platform TEXT NOT NULL DEFAULT '', "
            "game TEXT NOT NULL, "
            "elo INTEGER NOT NULL DEFAULT 1000, "
            "enqueued_at REAL NOT NULL DEFAULT 0)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS game_elo ("
            "player_key TEXT NOT NULL, "
            "game TEXT NOT NULL, "
            "rating INTEGER NOT NULL DEFAULT 1000, "
            "games INTEGER NOT NULL DEFAULT 0, "
            "player_name TEXT NOT NULL DEFAULT '', "
            "PRIMARY KEY (player_key, game))")
    except Exception:  # noqa: BLE001
        _log.debug("matchmaking ensure failed", exc_info=True)


# ── ELO ────────────────────────────────────────────────────────────────────

def get_rating(db: Any, player_key: str, game: str) -> int:
    """Current ELO for this player+game (1000 when unrated)."""
    _ensure(db)
    try:
        row = db.query_one(
            "SELECT rating FROM game_elo WHERE player_key = ? AND game = ?",
            (player_key, game))
        if row:
            return int(row.get("rating") or START_ELO)
    except Exception:  # noqa: BLE001
        _log.debug("elo read failed", exc_info=True)
    return START_ELO


def record_elo(db: Any, game: str, a_key: str, b_key: str,
               outcome: float, a_name: str = "",
               b_name: str = "") -> tuple[int, int, int, int]:
    """Record a 1v1 result. ``outcome`` is A's score: 1 win, 0.5 draw, 0 loss.

    Returns ``(a_new, a_delta, b_new, b_delta)``.
    """
    _ensure(db)
    ra, rb = get_rating(db, a_key, game), get_rating(db, b_key, game)
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    eb = 1.0 - ea
    ra_new = int(round(ra + K_FACTOR * (outcome - ea)))
    rb_new = int(round(rb + K_FACTOR * ((1.0 - outcome) - eb)))
    try:
        for key, new, name in ((a_key, ra_new, a_name),
                               (b_key, rb_new, b_name)):
            db.execute(
                "INSERT INTO game_elo "
                "(player_key, game, rating, games, player_name) "
                "VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(player_key, game) DO UPDATE SET "
                "rating = excluded.rating, "
                "games = game_elo.games + 1, "
                "player_name = excluded.player_name",
                (key, game, new, name or ""))
    except Exception:  # noqa: BLE001
        _log.debug("elo write failed", exc_info=True)
    return ra_new, ra_new - ra, rb_new, rb_new - rb


def render_ratings(db: Any, game: str = "", limit: int = 10) -> str:
    """``/game ratings`` text: top ELO boards."""
    _ensure(db)
    try:
        if game:
            rows = db.query(
                "SELECT player_name, player_key, rating, games FROM game_elo "
                "WHERE game = ? ORDER BY rating DESC LIMIT ?",
                (game, limit)) or []
            if not rows:
                return (f"no rated {game} players yet — queue up with "
                        f"/game queue {game}.")
            lines = [f"📊 {game} ratings:"]
        else:
            rows = db.query(
                "SELECT player_name, player_key, game, rating, games "
                "FROM game_elo ORDER BY rating DESC LIMIT ?",
                (limit,)) or []
            if not rows:
                return ("no rated players yet — ranked 1v1 games "
                        "(/game queue pvp) build the board.")
            lines = ["📊 top ratings:"]
        medals = ("🥇", "🥈", "🥉")
        for i, r in enumerate(rows):
            mark = medals[i] if i < 3 else f"{i + 1}."
            name = r.get("player_name") or r.get("player_key", "?").split(
                ":", 1)[-1]
            g = "" if game else f" · {r.get('game')}"
            lines.append(f" {mark} {name}{g} — {r.get('rating')} "
                         f"({r.get('games', 0)} games)")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        _log.debug("ratings render failed", exc_info=True)
        return "ratings are unavailable right now."


def maybe_record_ranked(db: Any, room: Any, game: Any,
                        winner: Any) -> list[str]:
    """ELO update for a ranked 1v1 finish. Returns announcement lines."""
    try:
        if getattr(game, "name", "") not in RANKED_GAMES:
            return []
        humans = [p for p in (room.players or []) if not p.is_ai]
        if len(humans) != 2:
            return []
        a, b = humans
        if winner == "draw" or winner is None:
            outcome = 0.5
        elif hasattr(winner, "key"):
            if winner.key == a.key:
                outcome = 1.0
            elif winner.key == b.key:
                outcome = 0.0
            else:
                return []
        else:
            return []
        ra_new, da, rb_new, db_ = record_elo(
            db, game.name, a.key, b.key, outcome,
            a_name=a.name, b_name=b.name)
        tag = "draw" if outcome == 0.5 else f"{(a.name if outcome == 1.0 else b.name)} takes it"
        return [f"📊 rated {game.name} ({tag}): {a.name} {ra_new} "
                f"({da:+d}) · {b.name} {rb_new} ({db_:+d})"]
    except Exception:  # noqa: BLE001
        _log.debug("ranked record failed", exc_info=True)
        return []


# ── the queue ──────────────────────────────────────────────────────────────

@dataclass
class QueuedPlayer:
    player_key: str
    chat_key: str
    player_name: str
    platform: str
    game: str
    elo: int
    enqueued_at: float


def _tolerance(enqueued_at: float, now: float) -> float:
    """ELO pairing tolerance widens the longer you wait: ±100 fresh,
    +120 per minute, uncapped — a quiet week still finds you a game,
    nobody waits forever."""
    waited_min = max(0.0, (now - enqueued_at) / 60.0)
    return 100.0 + 120.0 * waited_min


def match_status(db: Any, player_key: str) -> str | None:
    """Where you are in the queue, or None when not queued."""
    _ensure(db)
    try:
        row = db.query_one(
            "SELECT game, enqueued_at FROM match_queue WHERE player_key = ?",
            (player_key,))
        if not row:
            return None
        ahead = db.query_one(
            "SELECT COUNT(*) AS n FROM match_queue WHERE game = ? "
            "AND enqueued_at < ?",
            (row.get("game"), row.get("enqueued_at") or 0))
        n = int((ahead or {}).get("n") or 0)
        waited = int(time.time() - float(row.get("enqueued_at") or 0))
        return (f"⏳ queued for {row.get('game')} — {n} ahead of you, "
                f"waiting {waited}s. /game dequeue to leave.")
    except Exception:  # noqa: BLE001
        _log.debug("match status failed", exc_info=True)
        return None


class Matchmaker:
    """Pairs queued players into relay duels. Lives on the engine."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.db = getattr(engine, "db", None)
        self._lock = threading.RLock()
        self._last_sweep = 0.0

    # ── queue ops ──────────────────────────────────────────────────────────
    def enqueue(self, player: Any, chat_key: str,
                game_name: str) -> str:
        """Join the matchmaking queue. Returns the reply text."""
        _ensure(self.db)
        game_name = (game_name or "").strip().lower()
        if self.db is None:
            return "matchmaking needs the database — it's not available."
        try:
            if match_status(self.db, player.key) is not None:
                return ("you're already queued — /game dequeue to leave "
                        "first.")
            if game_name not in QUEUE_GAMES:
                return ("queueable games: " + ", ".join(QUEUE_GAMES) + "\n"
                        "usage: /game queue <game>")
            # don't queue mid-game in the same chat
            live = self.engine.live(chat_key)
            if live is not None:
                return ("finish your live game first — /game quit, "
                        "then queue.")
            elo = get_rating(self.db, player.key, game_name)
            self.db.execute(
                "INSERT OR REPLACE INTO match_queue "
                "(player_key, chat_key, player_name, platform, game, elo, "
                "enqueued_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (player.key, chat_key, player.name or "",
                 player.platform or "", game_name, elo, time.time()))
        except Exception:  # noqa: BLE001
            _log.debug("enqueue failed", exc_info=True)
            return "couldn't join the queue — try again."
        return (f"⏳ queued for {game_name} (ELO {elo}) — I'll pair you "
                f"with the closest rated player waiting. /game dequeue "
                f"to leave.")

    def dequeue(self, player_key: str) -> str:
        _ensure(self.db)
        try:
            cur = self.db.execute(
                "DELETE FROM match_queue WHERE player_key = ?",
                (player_key,))
            if cur.rowcount:
                return "left the queue."
        except Exception:  # noqa: BLE001
            _log.debug("dequeue failed", exc_info=True)
        return "you weren't queued."

    # ── the sweep ──────────────────────────────────────────────────────────
    def maybe_sweep(self) -> None:
        now = time.time()
        if now - self._last_sweep < SWEEP_INTERVAL:
            return
        self._last_sweep = now
        try:
            self.sweep()
        except Exception:  # noqa: BLE001
            _log.debug("match sweep failed", exc_info=True)

    def sweep(self) -> int:
        """Pair everyone pairable. Returns matches made."""
        if self.db is None:
            return 0
        _ensure(self.db)
        try:
            rows = self.db.query(
                "SELECT * FROM match_queue ORDER BY enqueued_at") or []
        except Exception:  # noqa: BLE001
            return 0
        now = time.time()
        by_game: dict[str, list[QueuedPlayer]] = {}
        for r in rows:
            try:
                by_game.setdefault(r["game"], []).append(QueuedPlayer(
                    player_key=r["player_key"], chat_key=r["chat_key"],
                    player_name=r.get("player_name") or "",
                    platform=r.get("platform") or "",
                    game=r["game"], elo=int(r.get("elo") or START_ELO),
                    enqueued_at=float(r.get("enqueued_at") or now)))
            except Exception:  # noqa: BLE001
                continue
        made = 0
        for game, entries in by_game.items():
            made += self._pair_game(game, entries, now)
        return made

    def _pair_game(self, game: str, entries: list[QueuedPlayer],
                   now: float) -> int:
        unpaired = sorted(entries, key=lambda e: e.enqueued_at)
        waiting: list[QueuedPlayer] = []
        made = 0
        while unpaired:
            a = unpaired.pop(0)
            tol_a = _tolerance(a.enqueued_at, now)
            best: QueuedPlayer | None = None
            best_diff = float("inf")
            for b in unpaired:
                diff = abs(a.elo - b.elo)
                if diff <= max(tol_a, _tolerance(b.enqueued_at, now)) \
                        and diff < best_diff:
                    best, best_diff = b, diff
            if best is None:
                waiting.append(a)
                continue
            unpaired.remove(best)
            try:
                self._make_match(game, a, best)
                made += 1
            except Exception:  # noqa: BLE001
                _log.debug("match creation failed", exc_info=True)
                waiting.extend([a, best])
        return made

    def _make_match(self, game: str, a: QueuedPlayer,
                    b: QueuedPlayer) -> None:
        """Create the relay duel and tell both chats."""
        from .players import Player
        relay = self.engine.relay
        pa = Player(key=a.player_key, platform=a.platform or "local",
                    name=a.player_name or a.player_key)
        pb = Player(key=b.player_key, platform=b.platform or "local",
                    name=b.player_name or b.player_key)
        invite = relay.create_invite(a.chat_key, pa, game)
        relay.accept_invite(invite.code, b.chat_key, pb)
        # both leave the queue — they're at a table now
        try:
            self.db.execute(
                "DELETE FROM match_queue WHERE player_key IN (?, ?)",
                (a.player_key, b.player_key))
        except Exception:  # noqa: BLE001
            pass
        msg_a = (f"⚔️ match found! **{game}** vs {pb.name} "
                 f"(ELO {b.elo}) — play here, moves are relayed.")
        msg_b = (f"⚔️ match found! **{game}** vs {pa.name} "
                 f"(ELO {a.elo}) — play here, moves are relayed.")
        self._notify(a.chat_key, msg_a)
        self._notify(b.chat_key, msg_b)

    def _notify(self, chat_key: str, text: str) -> None:
        send = getattr(self.engine, "_send", None)
        if send is None:
            return
        try:
            send(chat_key, text)
        except Exception:  # noqa: BLE001
            _log.debug("match notify failed", exc_info=True)
