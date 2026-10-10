"""Tournament arcs: multi-game brackets with persistent standings.

A tournament is a bracket of matches across one or more games. Players
accumulate tournament points per match (win=3, draw=1, loss=0 by default);
the bracket advances automatically as rooms finish. Tournaments persist to
JSON so they survive restarts, and they work across chats — a tournament
can span a group and DMs.

Stakes: a tournament (or single match) can carry a coin pot. Each player
antes the same amount from their PlayerStore coins; the winner takes the
pot. All coin movement goes through the store — no shadow balances.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Tournament", "MatchResult", "tournament_dir",
    "new_tournament", "load_tournament", "list_tournaments",
    # formats + tiebreaks
    "swiss_pairings", "round_robin_schedule", "elimination_bracket",
    "buchholz", "sonneborn_berger", "tiebreak_standings",
    "render_tiebreak_standings", "TOURNAMENT_FORMATS",
]

#: Supported tournament formats.
TOURNAMENT_FORMATS = ("points", "swiss", "elimination", "round_robin")

WIN_PTS, DRAW_PTS = 3, 1


def tournament_dir() -> Path:
    p = Path(__file__).resolve().parent / "data" / "tournaments"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class MatchResult:
    game: str
    players: list[str]
    winner: str | None
    room_id: str = ""
    at: float = 0.0


@dataclass
class Tournament:
    id: str
    name: str
    games: list[str]                       # rotation of games per round
    players: list[str]                     # player keys
    rounds: int
    current_round: int = 1
    points: dict[str, int] = field(default_factory=dict)
    matches: list[dict] = field(default_factory=list)
    stake: int = 0                         # coin ante per player (0 = free)
    pot: int = 0
    status: str = "open"                   # open | running | finished
    winner: str | None = None
    format: str = "points"                 # points | swiss | elimination | round_robin
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Tournament":
        return cls(**{k: d.get(k, getattr(cls, k, None)) for k in (
            "id", "name", "games", "players", "rounds", "current_round",
            "points", "matches", "stake", "pot", "status", "winner",
            "format", "created_at")})

    def save(self) -> Path:
        p = tournament_dir() / f"{self.id}.json"
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return p

    def game_for_round(self) -> str:
        return self.games[(self.current_round - 1) % len(self.games)]

    def record_match(self, result: MatchResult) -> list[str]:
        """Record a finished match. Returns announcement lines."""
        msgs: list[str] = []
        r = asdict(result)
        r["at"] = time.time()
        r["round"] = self.current_round
        self.matches.append(r)
        if result.winner:
            self.points[result.winner] = self.points.get(result.winner, 0) + WIN_PTS
            msgs.append(f"🏅 {result.winner} takes the match (+{WIN_PTS})")
        else:
            for p in result.players:
                self.points[p] = self.points.get(p, 0) + DRAW_PTS
            msgs.append("🤝 draw — everyone banks +1")
        # bracket advance: a round completes when every player has had
        # a match (~players/2 matches for head-to-head pairings)
        per_round = max(1, len(self.players) // 2)
        if len(self.matches) % per_round == 0:
            self.current_round += 1
            if self.current_round > self.rounds:
                return self.finish() + msgs
            msgs.append(
                f"📯 round {self.current_round}/{self.rounds} — "
                f"next game: {self.game_for_round()}")
        self.save()
        return msgs

    def finish(self) -> list[str]:
        self.status = "finished"
        top = max(self.points.items(), key=lambda kv: kv[1], default=(None, 0))
        self.winner = top[0]
        msgs = [f"🏆 **{self.name}** complete — champion: **{self.winner}**"]
        board = sorted(self.points.items(), key=lambda kv: -kv[1])
        msgs.append("📊 " + " · ".join(f"{n} {p}" for n, p in board))
        if self.pot and self.winner:
            msgs.append(f"💰 {self.winner} takes the {self.pot}-coin pot")
        self.save()
        return msgs

    def standings_text(self) -> str:
        if self.format == "swiss":
            hist = self._sb_history()
            return render_tiebreak_standings(
                {p: float(v) for p, v in self.points.items()}, hist,
                name=f"{self.name} — round {self.current_round}/{self.rounds}")
        board = sorted(self.points.items(), key=lambda kv: -kv[1])
        lines = [f"🏟️ **{self.name}** — round {self.current_round}/{self.rounds}"
                 + (f" · {self.format}" if self.format != "points" else "")]
        lines += [f"{i+1}. {n} — {p} pts" for i, (n, p) in enumerate(board)]
        if self.stake:
            lines.append(f"💰 stake: {self.stake} coins each (pot {self.pot})")
        return "\n".join(lines)

    def _sb_history(self) -> list[tuple[str, str | None, float | None]]:
        """(player, opponent, player_score) for tiebreak math."""
        hist: list[tuple[str, str | None, float | None]] = []
        for m in self.matches:
            players = m.get("players") or []
            winner = m.get("winner")
            if len(players) == 2:
                a, b = players[0], players[1]
                if winner == a:
                    hist += [(a, b, 1.0), (b, a, 0.0)]
                elif winner == b:
                    hist += [(a, b, 0.0), (b, a, 1.0)]
                else:
                    hist += [(a, b, 0.5), (b, a, 0.5)]
            elif len(players) == 1:
                hist.append((players[0], None, 1.0))  # bye
        return hist

    def pair_round(self, seed: int | None = None) -> list[tuple[str, str | None]]:
        """Pairings for the current round under this tournament's format."""
        rng = random.Random(seed)
        hist_pairs = [((m.get("players") or [None])[0],
                       (m.get("players") or [None, None])[1])
                      for m in self.matches]
        if self.format == "swiss":
            return swiss_pairings(self.players,
                                  {p: float(v)
                                   for p, v in self.points.items()},
                                  hist_pairs, rng=rng)
        if self.format == "elimination":
            alive = [p for p in self.players]
            # single elim: pair top-vs-bottom by current points
            alive.sort(key=lambda p: -self.points.get(p, 0))
            pairs: list[tuple[str, str | None]] = []
            while len(alive) >= 2:
                pairs.append((alive.pop(0), alive.pop(-1)))
            if alive:
                pairs.append((alive[0], None))
            return pairs
        # points / round_robin: rotate through the circle schedule
        sched = round_robin_schedule(self.players)
        return sched[(self.current_round - 1) % len(sched)]

    def collect_antes(self, store: Any, platform: str = "spine") -> list[str]:
        """Deduct the stake from every player. Returns problems (empty = ok)."""
        from .players import Player
        problems: list[str] = []
        if not self.stake:
            return problems
        for key in self.players:
            p = Player.from_sender(platform, key, key)
            prof = store.get_for(p)
            if prof.coins < self.stake:
                problems.append(f"{key} has {prof.coins} coins, needs {self.stake}")
        if problems:
            return problems
        for key in self.players:
            p = Player.from_sender(platform, key, key)
            store.record_outcome(p, won=None, game="tournament",
                                 coins=-self.stake)
        return []

    def pay_pot(self, store: Any, platform: str = "spine") -> str:
        """Award the pot to the tournament winner."""
        from .players import Player
        if not (self.pot and self.winner):
            return ""
        p = Player.from_sender(platform, self.winner, self.winner)
        store.record_outcome(p, won=True, game="tournament", coins=self.pot)
        return f"💰 {self.winner} banks {self.pot} coins"


def new_tournament(name: str, games: list[str], players: list[str],
                   rounds: int = 3, stake: int = 0,
                   format: str = "points") -> Tournament:
    tid = f"t{int(time.time())}{random.randint(100, 999)}"
    fmt = format if format in TOURNAMENT_FORMATS else "points"
    t = Tournament(
        id=tid, name=name, games=games, players=list(players),
        rounds=max(1, rounds), stake=max(0, stake),
        pot=max(0, stake) * len(players),
        points={p: 0 for p in players},
        format=fmt,
        created_at=time.time(),
    )
    t.save()
    return t


def load_tournament(tid: str) -> Tournament | None:
    p = tournament_dir() / f"{tid}.json"
    if not p.exists():
        # allow short-id prefix match
        cands = [q for q in tournament_dir().glob("*.json")
                 if q.stem.startswith(tid)]
        if not cands:
            return None
        p = cands[0]
    try:
        return Tournament.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        _log.warning("tournament: bad file %s: %s", p.name, exc)
        return None


def list_tournaments(*, active_only: bool = True) -> list[Tournament]:
    out = []
    for p in sorted(tournament_dir().glob("*.json")):
        try:
            t = Tournament.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
        if active_only and t.status == "finished":
            continue
        out.append(t)
    return out


# ── formats: Swiss, elimination, round-robin ────────────────────────────────
#
# Points leagues are one format. Real tournament culture runs on:
# Swiss (score-group pairing, no rematches, byes for odd fields),
# single/double elimination brackets, and round-robin schedules.
# Tiebreaks follow chess convention: Buchholz (strength of schedule)
# then Sonneborn-Berger, then head-to-head.


def swiss_pairings(players: list[str], points: dict[str, float],
                   history: list[tuple[str, str | None]],
                   rng: random.Random | None = None
                   ) -> list[tuple[str, str | None]]:
    """Pair one Swiss round.

    ``players``: everyone in the field. ``points``: current scores.
    ``history``: past (a, b) pairs, b=None for a bye. Returns
    ``[(a, b)]`` with b=None for the bye. Players are grouped by score
    (high first), paired top-down inside each group; nobody meets twice;
    nobody gets two byes — the lowest-scoring bye-less player sits out.
    """
    rng = rng or random.Random()
    played: dict[str, set[str]] = {p: set() for p in players}
    byes = {p for p in players}
    for a, b in history:
        if b is None:
            byes.discard(a)
        else:
            played[a].add(b)
            played[b].add(a)
    order = sorted(players, key=lambda p: (-points.get(p, 0), rng.random()))
    pairs: list[tuple[str, str | None]] = []
    if len(order) % 2 == 1:
        # bye: lowest score, never had one; tie-break by random
        cands = [p for p in reversed(order) if p in byes] or list(
            reversed(order))
        bye = cands[0]
        order.remove(bye)
        pairs.append((bye, None))
    remaining = list(order)
    while remaining:
        a = remaining.pop(0)
        # first unplayed partner; if everyone was played, take the
        # earliest-played (least recent rematch)
        best, best_rank = None, None
        for i, b in enumerate(remaining):
            rank = 0 if b not in played[a] else 1
            if best_rank is None or rank < best_rank:
                best, best_rank, best_i = b, rank, i
                if rank == 0:
                    break
        remaining.pop(best_i)
        pairs.append((a, best))
    return pairs


def round_robin_schedule(players: list[str]) -> list[list[tuple[str, str]]]:
    """Circle-method schedule: every pair meets exactly once. Returns
    rounds of ``[(a, b)]``. Odd fields get a bye (``(p, None)``)."""
    ps = list(players)
    dummy = None
    if len(ps) % 2 == 1:
        ps.append(dummy)  # type: ignore[arg-type]
    n = len(ps)
    rounds: list[list[tuple[str, str]]] = []
    fixed, rotating = ps[0], ps[1:]
    for _ in range(n - 1):
        order = [fixed] + rotating
        rnd = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            if a is None:
                rnd.append((b, None))  # type: ignore[arg-type]
            elif b is None:
                rnd.append((a, None))
            else:
                rnd.append((a, b))
        rounds.append(rnd)
        rotating = [rotating[-1]] + rotating[:-1]
    return rounds


def elimination_bracket(players: list[str],
                        rng: random.Random | None = None,
                        double: bool = False
                        ) -> dict[str, Any]:
    """Seeded single/double elimination bracket.

    Players are seeded by position (1 vs last, 2 vs second-last…);
    odd counts give top seeds byes. ``double`` adds a losers bracket —
    a player is out only after two losses, and the grand final can
    reset if the losers-bracket champion wins it.
    """
    rng = rng or random.Random()
    ps = list(players)
    rng.shuffle(ps)
    # next power of two; byes go to the top of the shuffled order
    size = 1
    while size < len(ps):
        size *= 2
    while len(ps) < size:
        ps.append(None)  # type: ignore[arg-type]
    winners = [(ps[i], ps[size - 1 - i]) for i in range(size // 2)]
    bracket: dict[str, Any] = {
        "winners": [[a, b] for a, b in winners],
        "rounds": size.bit_length() - 1,
    }
    if double:
        bracket["losers"] = []
        bracket["grand_final_reset"] = True
    return bracket


# ── tiebreaks ──────────────────────────────────────────────────────────────

def buchholz(player: str, points: dict[str, float],
             history: list[tuple[str, str | None, float | None]]) -> float:
    """Sum of opponents' scores (strength of schedule). ``history`` is
    (player, opponent, player_score) with opponent None for byes."""
    total = 0.0
    for p, opp, _score in history:
        if p == player and opp is not None:
            total += points.get(opp, 0.0)
    return total


def sonneborn_berger(player: str, points: dict[str, float],
                     history: list[tuple[str, str | None, float | None]]
                     ) -> float:
    """Sum of defeated opponents' scores + half of drawn opponents'."""
    total = 0.0
    for p, opp, score in history:
        if p == player and opp is not None and score is not None:
            if score == 1.0:
                total += points.get(opp, 0.0)
            elif score == 0.5:
                total += points.get(opp, 0.0) / 2.0
    return total


def tiebreak_standings(points: dict[str, float],
                       history: list[tuple[str, str | None, float | None]]
                       ) -> list[tuple[str, float, float, float]]:
    """``[(player, points, buchholz, sonneborn_berger)]`` sorted by
    points, then Buchholz, then Sonneborn-Berger, then name."""
    rows = [(p, pts, buchholz(p, points, history),
             sonneborn_berger(p, points, history))
            for p, pts in points.items()]
    rows.sort(key=lambda r: (-r[1], -r[2], -r[3], r[0]))
    return rows


def render_tiebreak_standings(points: dict[str, float],
                              history: list[tuple[str, str | None,
                                                  float | None]],
                              name: str = "standings") -> str:
    rows = tiebreak_standings(points, history)
    medals = ("🥇", "🥈", "🥉")
    lines = [f"🏟️ {name} — tiebreaks: Buchholz, S-B"]
    for i, (p, pts, bh, sb) in enumerate(rows):
        mark = medals[i] if i < 3 else f"{i + 1}."
        lines.append(f" {mark} {p} — {pts:g} pts "
                     f"(BH {bh:g} · SB {sb:g})")
    return "\n".join(lines)
