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
import math as _math
import threading
import time
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("nomorals.games.matchmaking")

__all__ = [
    "QUEUE_GAMES",
    "RANKED_GAMES",
    "Matchmaker",
    "RANK_TIERS",
    "get_rating",
    "match_status",
    "maybe_record_ranked",
    "record_elo",
    "render_ratings",
    # Glicko-2 rating (uncertainty-aware) additions
    "get_glicko",
    "record_glicko2",
    "glicko_ordinal",
    "rank_tier",
    "predict_win_probability",
    "render_tiers",
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


# ── Glicko-2: uncertainty-aware ratings ─────────────────────────────────
#
# Elo treats every player as equally well-known. Glicko-2 tracks a rating
# deviation (RD / sigma) that grows while a player is idle and shrinks as
# they play — a rusty veteran and a hot newcomer are matched by their
# *conservative ordinal* (mu − 3·sigma), so uncertain players can't farm
# certain ones. This is the OpenSkill/TrueSkill family approach.
#
# Scale: we keep the familiar 1000-center but run Glicko-2 internally on
# its native ~1500 scale, converting at the boundary. New players start
# (mu=1500, phi=350, sigma=0.06); phi=350 means "we know nothing yet".

# Scale: we keep the familiar 1000-center but run Glicko-2 internally on
# its native ~1500 scale, converting at the boundary. New players start
# (mu=1500, phi=350, sigma=0.06); phi=350 means "we know nothing yet".

_GLICKO_SCALE_MU = 1500.0
_GLICKO_SCALE_PHI = 350.0
_GLICKO_TAU = 0.5            # volatility constraint (Glickman's default)
_GLICKO_EPS = 1e-6
_PLACEMENT_GAMES = 5         # provisional: gains x2, never drop below start
_RD_GROWTH_PER_DAY = 12.0    # RD inflation per idle day (rust), capped
_RD_CAP = 350.0

#: Visible rank tiers: (name, emoji, ordinal floor). Ordinal = mu − 3·sigma
#: mapped back onto the 1000-scale. Tiers are display + matchmaking
#: bands; the raw ordinal stays the pairing input.
RANK_TIERS: tuple[tuple[str, str, float], ...] = (
    ("Bronze", "🟤", 0),
    ("Silver", "⚪", 900),
    ("Gold", "🟡", 1100),
    ("Platinum", "🔷", 1300),
    ("Diamond", "💎", 1500),
    ("Mythic", "🟣", 1700),
)


def _ensure_glicko(db: Any) -> None:
    _ensure(db)
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS game_glicko ("
            "player_key TEXT NOT NULL, "
            "game TEXT NOT NULL, "
            "mu REAL NOT NULL DEFAULT 1500, "
            "phi REAL NOT NULL DEFAULT 350, "
            "sigma REAL NOT NULL DEFAULT 0.06, "
            "games INTEGER NOT NULL DEFAULT 0, "
            "last_played REAL NOT NULL DEFAULT 0, "
            "player_name TEXT NOT NULL DEFAULT '', "
            "PRIMARY KEY (player_key, game))")
    except Exception:  # noqa: BLE001
        _log.debug("glicko ensure failed", exc_info=True)


def _to_glicko_scale(rating_1000: float) -> float:
    return (rating_1000 - 1000.0) + _GLICKO_SCALE_MU


def _from_glicko_scale(mu: float) -> float:
    return (mu - _GLICKO_SCALE_MU) + 1000.0


def _g(rd: float) -> float:
    return 1.0 / _math.sqrt(1.0 + 3.0 * rd * rd / (_math.pi * _math.pi))


def _E(mu: float, muj: float, rdj: float) -> float:
    return 1.0 / (1.0 + _math.exp(-_g(rdj) * (mu - muj)))


def get_glicko(db: Any, player_key: str,
               game: str) -> dict[str, float]:
    """``{'mu','phi','sigma','games','ordinal','tier'}``.

    ``phi`` is inflated by idle days since ``last_played`` (rust) —
    the ordinal decays toward the start rating the longer you're away.
    ``ordinal`` is the conservative mu − 3·phi on the 1000-scale.
    """
    _ensure_glicko(db)
    mu, phi, sigma, games = (_GLICKO_SCALE_MU, _GLICKO_SCALE_PHI,
                            0.06, 0)
    name = ""
    try:
        row = db.query_one(
            "SELECT mu, phi, sigma, games, last_played, player_name "
            "FROM game_glicko WHERE player_key = ? AND game = ?",
            (player_key, game))
        if row:
            mu = float(row.get("mu") or mu)
            phi = float(row.get("phi") or phi)
            sigma = float(row.get("sigma") or 0.06)
            games = int(row.get("games") or 0)
            name = str(row.get("player_name") or "")
            last = float(row.get("last_played") or 0)
            if last:
                idle_days = max(0.0, (time.time() - last) / 86400.0)
                phi = min(_RD_CAP, _math.sqrt(
                    phi * phi + (_RD_GROWTH_PER_DAY ** 2) * idle_days))
    except Exception:  # noqa: BLE001
        _log.debug("glicko read failed", exc_info=True)
    ordinal = _from_glicko_scale(mu - 3.0 * phi)
    return {"mu": _from_glicko_scale(mu), "phi": phi, "sigma": sigma,
            "games": games, "ordinal": ordinal,
            "tier": rank_tier(ordinal), "name": name}


def glicko_ordinal(db: Any, player_key: str, game: str) -> float:
    """Conservative skill estimate for pairing (mu − 3·sigma)."""
    return get_glicko(db, player_key, game)["ordinal"]


def rank_tier(ordinal: float) -> str:
    """Visible tier name for a 1000-scale ordinal."""
    tier = RANK_TIERS[0][0]
    for name, _emoji, floor in RANK_TIERS:
        if ordinal >= floor:
            tier = name
    return tier


def rank_tier_emoji(ordinal: float) -> str:
    emoji = RANK_TIERS[0][1]
    for _name, e, floor in RANK_TIERS:
        if ordinal >= floor:
            emoji = e
    return emoji


def predict_win_probability(db: Any, game: str, a_key: str,
                            b_key: str) -> float:
    """P(A beats B) from Glicko mus — for fair pairings and upset calls."""
    try:
        ga = get_glicko(db, a_key, game)
        gb = get_glicko(db, b_key, game)
        mu_a = _to_glicko_scale(ga["mu"])
        mu_b = _to_glicko_scale(gb["mu"])
        # uncertainty-aware: combine RDs like Glickman's expected score
        rd = _math.sqrt(ga["phi"] ** 2 + gb["phi"] ** 2)
        return 1.0 / (1.0 + _math.exp(-_g(rd / 173.7178) * (mu_a - mu_b)
                                      / 173.7178))
    except Exception:  # noqa: BLE001
        return 0.5


def _glicko2_update(mu: float, phi: float, sigma: float,
                    opponents: list[tuple[float, float, float]]
                    ) -> tuple[float, float, float]:
    """One rating period. ``opponents`` = [(mu_j, phi_j, outcome)]."""
    # step 2: convert to Glicko-2 scale
    mu2 = (mu - 1500.0) / 173.7178
    phi2 = phi / 173.7178
    # step 3-4: estimated variance and improvement
    v_inv = 0.0
    delta_sum = 0.0
    for muj, phij, score in opponents:
        muj2 = (muj - 1500.0) / 173.7178
        phij2 = phij / 173.7178
        e = _E(mu2, muj2, phij2)
        g = _g(phij2)
        v_inv += g * g * e * (1.0 - e)
        delta_sum += g * (score - e)
    v = 1.0 / max(v_inv, _GLICKO_EPS)
    delta = v * delta_sum
    # step 5: new volatility via the illustrated algorithm
    a = _math.log(sigma * sigma)
    tau2 = _GLICKO_TAU * _GLICKO_TAU

    def f(x: float) -> float:
        ex = _math.exp(x)
        num = ex * (delta * delta - phi2 * phi2 - v - ex)
        den = 2.0 * (phi2 * phi2 + v + ex) ** 2
        return num / den - (x - a) / tau2

    A = a
    if delta * delta > phi2 * phi2 + v:
        B = _math.log(delta * delta - phi2 * phi2 - v)
    else:
        k = 1
        while f(a - k * _GLICKO_TAU) < 0:
            k += 1
        B = a - k * _GLICKO_TAU
    fa, fb = f(A), f(B)
    while abs(B - A) > _GLICKO_EPS:
        C = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if fc * fb <= 0:
            A, fa = B, fb
        else:
            fa /= 2.0
        B, fb = C, fc
    sigma_p = _math.exp(A / 2.0)
    # step 6-8: new RD and rating
    phi_star = _math.sqrt(phi2 * phi2 + sigma_p * sigma_p)
    phi_p = 1.0 / _math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_p = mu2 + phi_p * phi_p * delta_sum
    return mu_p * 173.7178 + 1500.0, phi_p * 173.7178, sigma_p


def record_glicko2(db: Any, game: str, a_key: str, b_key: str,
                   outcome: float, a_name: str = "",
                   b_name: str = "") -> dict[str, Any]:
    """Record a 1v1 result under Glicko-2. ``outcome`` is A's score.

    Placement: a player's first ``_PLACEMENT_GAMES`` games move double
    and can never drop them below the start rating — newcomers find
    their level fast without being punished for trying.

    Returns a report dict with new ordinals, deltas, tiers and upset flag.
    """
    _ensure_glicko(db)
    ga = get_glicko(db, a_key, game)
    gb = get_glicko(db, b_key, game)
    mu_a = _to_glicko_scale(ga["mu"])
    mu_b = _to_glicko_scale(gb["mu"])
    prob_a = predict_win_probability(db, game, a_key, b_key)
    new_a = _glicko2_update(mu_a, ga["phi"], ga["sigma"],
                            [(mu_b, gb["phi"], outcome)])
    new_b = _glicko2_update(mu_b, gb["phi"], gb["sigma"],
                            [(mu_a, ga["phi"], 1.0 - outcome)])
    now = time.time()
    results: dict[str, Any] = {"prob_a": prob_a,
                               "upset": (outcome == 1.0 and prob_a < 0.35)
                               or (outcome == 0.0 and prob_a > 0.65)}
    for key, old, new, name, is_a in (
            (a_key, ga, new_a, a_name, True),
            (b_key, gb, new_b, b_name, False)):
        old_ord = old["ordinal"]
        games = int(old["games"]) + 1
        placed = games > _PLACEMENT_GAMES
        mu_n, phi_n, sigma_n = new
        if not placed:
            # provisional: double THIS game's movement only (relative to
            # the pre-game rating, so it can't compound), and never drop
            # a newcomer below the start rating for trying.
            pre_mu = mu_a if is_a else mu_b
            mu_n = pre_mu + 2.0 * (mu_n - pre_mu)
            mu_n = max(mu_n, _GLICKO_SCALE_MU)
        new_ord = _from_glicko_scale(mu_n - 3.0 * phi_n)
        tier = rank_tier(new_ord)
        try:
            db.execute(
                "INSERT INTO game_glicko "
                "(player_key, game, mu, phi, sigma, games, last_played,"
                " player_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(player_key, game) DO UPDATE SET "
                "mu = excluded.mu, phi = excluded.phi, "
                "sigma = excluded.sigma, games = excluded.games, "
                "last_played = excluded.last_played, "
                "player_name = excluded.player_name",
                (key, game, mu_n, phi_n, sigma_n, games, now,
                 name or old.get("name") or ""))
        except Exception:  # noqa: BLE001
            _log.debug("glicko write failed", exc_info=True)
        results["a" if is_a else "b"] = {
            "ordinal": round(new_ord), "delta": round(new_ord - old_ord),
            "tier": tier, "emoji": rank_tier_emoji(new_ord),
            "provisional": not placed, "games": games}
    return results


def render_tiers(db: Any, game: str = "", limit: int = 10) -> str:
    """``/game tiers``: Glicko ordinal board with rank tiers."""
    _ensure_glicko(db)
    try:
        q = ("SELECT player_key, player_name, mu, phi, games FROM "
             "game_glicko")
        args: tuple = ()
        if game:
            q += " WHERE game = ?"
            args = (game,)
        rows = db.query(q, args) or []
        if not rows:
            return ("no ranked players yet — ranked 1v1 games build "
                    "the board.")
        scored = []
        for r in rows:
            mu = float(r.get("mu") or _GLICKO_SCALE_MU)
            phi = float(r.get("phi") or _GLICKO_SCALE_PHI)
            ordinal = _from_glicko_scale(mu - 3.0 * phi)
            scored.append((ordinal, r))
        scored.sort(key=lambda t: -t[0])
        head = f"🏅 {game or 'all games'} tiers (skill ± uncertainty):"
        lines = [head]
        for i, (ordinal, r) in enumerate(scored[:limit], 1):
            name = r.get("player_name") or str(r.get("player_key", "?")).split(
                ":", 1)[-1]
            g = "" if game else f" · {game}"
            prov = " 🌱" if int(r.get("games") or 0) <= _PLACEMENT_GAMES else ""
            lines.append(f" {i}. {rank_tier_emoji(ordinal)} {name}{g} — "
                         f"{rank_tier(ordinal)} {int(ordinal)}{prov}")
        lines.append("🌱 = provisional (first 5 games)")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        _log.debug("tiers render failed", exc_info=True)
        return "tiers are unavailable right now."

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
        lines = [f"📊 rated {game.name} ({tag}): {a.name} {ra_new} "
                 f"({da:+d}) · {b.name} {rb_new} ({db_:+d})"]
        # Glicko-2 runs alongside Elo: uncertainty-aware rating, visible
        # tiers, placement protection, upset detection.
        try:
            rep = record_glicko2(db, game.name, a.key, b.key, outcome,
                                 a_name=a.name, b_name=b.name)
            ra, rb = rep["a"], rep["b"]
            tier_line = (f"{ra['emoji']} {a.name}: {ra['tier']} "
                         f"{ra['ordinal']} ({ra['delta']:+d}) · "
                         f"{rb['emoji']} {b.name}: {rb['tier']} "
                         f"{rb['ordinal']} ({rb['delta']:+d})")
            if ra["provisional"] or rb["provisional"]:
                tier_line += " 🌱 provisional"
            if rep.get("upset"):
                underdog = b.name if outcome == 1.0 else a.name
                tier_line += f" — 😱 UPSET! {underdog} defied the odds"
            lines.append(tier_line)
        except Exception:  # noqa: BLE001
            _log.debug("glicko record failed", exc_info=True)
        return lines
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
        # Pair on the conservative Glicko ordinal (mu − 3·sigma), not raw
        # Elo: uncertain newcomers can't be fed to certain veterans, and
        # rusty returners land fair fights. Players with no rated games
        # yet fall back to Elo — identical default ordinals would
        # otherwise pair a 1600 with a 1000.
        def ordinal(e: QueuedPlayer) -> float | None:
            try:
                if self.db is not None:
                    g = get_glicko(self.db, e.player_key, e.game)
                    if int(g.get("games", 0)) > 0:
                        return float(g["ordinal"])
            except Exception:  # noqa: BLE001
                pass
            return None

        def skill(e: QueuedPlayer) -> float:
            return ordinal(e) if ordinal(e) is not None else float(e.elo)

        waiting: list[QueuedPlayer] = []
        made = 0
        while unpaired:
            a = unpaired.pop(0)
            tol_a = _tolerance(a.enqueued_at, now)
            best: QueuedPlayer | None = None
            best_diff = float("inf")
            sa = skill(a)
            for b in unpaired:
                diff = abs(sa - skill(b))
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
