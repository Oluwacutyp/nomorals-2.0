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
]

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
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Tournament":
        return cls(**{k: d.get(k, getattr(cls, k, None)) for k in (
            "id", "name", "games", "players", "rounds", "current_round",
            "points", "matches", "stake", "pot", "status", "winner",
            "created_at")})

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
        board = sorted(self.points.items(), key=lambda kv: -kv[1])
        lines = [f"🏟️ **{self.name}** — round {self.current_round}/{self.rounds}"]
        lines += [f"{i+1}. {n} — {p} pts" for i, (n, p) in enumerate(board)]
        if self.stake:
            lines.append(f"💰 stake: {self.stake} coins each (pot {self.pot})")
        return "\n".join(lines)

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
                   rounds: int = 3, stake: int = 0) -> Tournament:
    tid = f"t{int(time.time())}{random.randint(100, 999)}"
    t = Tournament(
        id=tid, name=name, games=games, players=list(players),
        rounds=max(1, rounds), stake=max(0, stake),
        pot=max(0, stake) * len(players),
        points={p: 0 for p in players},
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
