"""Group tournaments + seasons + async multiplayer + group trivia.

* **Seasons** — quarterly seasons (Q1–Q4 by calendar quarter) with a
  persistent leaderboard. Points accumulate per game result; the
  leaderboard sums across games and breaks ties by wins, then name.
  Prizes come from the economy through an *injected* ledger callable —
  this module never touches a wallet itself, so a season can pay coins,
  gems, roles, or nothing, depending on what the caller wires in.
* **Async multiplayer** — games that run over days inside a group, not
  minutes: murder mystery, scavenger hunt, and election. Each is an
  explicit state machine (phases, legal actions, deadlines); illegal
  actions return ``(False, reason)`` instead of corrupting state.
  State persists to JSON per group under ``~/.devon/community/``.
* **Trivia from the group's own material** — questions are built from
  *injected* group messages (``guess who said this`` / wisdom / book),
  never from owner data. Deterministic given ``(messages, seed)``.

Only stdlib + ``nomorals.core``. No owner-data imports.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Season", "SeasonLeaderboard", "SeasonPrize", "quarter_of",
    "season_id_for", "AsyncGameStore", "MurderMysteryGame",
    "ScavengerHuntGame", "ElectionGame", "TriviaQuestion",
    "build_trivia", "ASYNC_GAMES",
]

ASYNC_GAMES = ("murder_mystery", "scavenger_hunt", "election")


# ───────────────────────── seasons ──────────────────────────────────────────

def quarter_of(ts: float | None = None) -> tuple[int, int]:
    """(year, quarter) for a unix timestamp (UTC)."""
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(),
                                tz=timezone.utc)
    return dt.year, (dt.month - 1) // 3 + 1


def season_id_for(ts: float | None = None) -> str:
    year, quarter = quarter_of(ts)
    return f"{year}-Q{quarter}"


@dataclass
class Season:
    """One quarterly season for one group."""

    id: str                       # e.g. "2026-Q4"
    group_id: str
    name: str = ""
    starts_at: float = 0.0
    ends_at: float = 0.0
    status: str = "active"        # active | closed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SeasonLeaderboard:
    """Season points. ``add_result`` is the only mutator; ``standings``
    is the single source of truth for ranking."""

    season_id: str
    group_id: str
    points: dict[str, int] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, player: str, points: int, *, game: str = "",
                   won: bool = False) -> None:
        self.points[player] = self.points.get(player, 0) + int(points)
        self.games[player] = self.games.get(player, 0) + 1
        if won:
            self.wins[player] = self.wins.get(player, 0) + 1
        self.history.append({"player": player, "points": int(points),
                             "game": game, "won": won, "at": time.time()})

    def total_points(self) -> int:
        return sum(self.points.values())

    def standings(self) -> list[dict[str, Any]]:
        """Ranked: points desc, wins desc, name asc (fully deterministic)."""
        rows = [{"player": p, "points": pts, "wins": self.wins.get(p, 0),
                 "games": self.games.get(p, 0)}
                for p, pts in self.points.items()]
        rows.sort(key=lambda r: (-r["points"], -r["wins"], r["player"]))
        for i, row in enumerate(rows, 1):
            row["rank"] = i
        return rows

    def render(self) -> str:
        lines = [f"🏆 season {self.season_id} leaderboard "
                 f"({self.total_points()} pts total)"]
        for row in self.standings():
            lines.append(f"  {row['rank']}. {row['player']} — "
                         f"{row['points']} pts · {row['wins']}W / {row['games']}G")
        return "\n".join(lines) if len(lines) > 1 else lines[0] + "\n  (no results yet)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SeasonLeaderboard":
        return cls(season_id=str(d["season_id"]), group_id=str(d["group_id"]),
                   points={str(k): int(v) for k, v in d.get("points", {}).items()},
                   wins={str(k): int(v) for k, v in d.get("wins", {}).items()},
                   games={str(k): int(v) for k, v in d.get("games", {}).items()},
                   history=list(d.get("history", [])))


@dataclass
class SeasonPrize:
    """Prize table + payout through an injected ledger.

    ``prize_table`` maps rank → amount. ``ledger`` is a callable
    ``(player, amount, reason) -> bool`` supplied by the caller (e.g. a
    wrapper around the game economy). No ledger → dry run that reports
    what *would* be paid, paying nothing.
    """

    season_id: str
    prize_table: dict[int, int] = field(default_factory=lambda: {1: 500, 2: 250, 3: 100})

    def distribute(self, board: SeasonLeaderboard,
                   ledger: Callable[[str, int, str], bool] | None = None) -> list[dict[str, Any]]:
        payouts: list[dict[str, Any]] = []
        for row in board.standings():
            amount = self.prize_table.get(row["rank"], 0)
            if amount <= 0:
                continue
            paid = False
            if ledger is not None:
                try:
                    paid = bool(ledger(row["player"], amount,
                                       f"season {self.season_id} rank {row['rank']}"))
                except Exception as exc:  # noqa: BLE001 — ledger is caller code
                    _log.warning("season prize ledger failed: %s", exc)
            payouts.append({"player": row["player"], "rank": row["rank"],
                            "amount": amount, "paid": paid,
                            "dry_run": ledger is None})
        return payouts


# ───────────────────────── async multiplayer ────────────────────────────────

def _async_root() -> Path:
    return Path.home() / ".devon" / "community" / "async_games"


@dataclass
class AsyncGameState:
    """Serialized state for one async game. ``phase`` is the state machine."""

    id: str
    game: str
    group_id: str
    players: list[str]
    phase: str
    state: dict[str, Any] = field(default_factory=dict)
    created_ts: float = 0.0
    updated_ts: float = 0.0
    deadline_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AsyncGameState":
        return cls(id=str(d["id"]), game=str(d["game"]),
                   group_id=str(d["group_id"]), players=list(d.get("players", [])),
                   phase=str(d["phase"]), state=dict(d.get("state", {})),
                   created_ts=float(d.get("created_ts", 0)),
                   updated_ts=float(d.get("updated_ts", 0)),
                   deadline_ts=d.get("deadline_ts"))


class AsyncGameStore:
    """JSON persistence for async games, one file per group."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else _async_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, group_id: str) -> Path:
        safe = group_id.replace("/", "_").replace("\\", "_").strip() or "group"
        return self.root / f"{safe}.json"

    def save(self, game: AsyncGameState) -> None:
        path = self._path(game.group_id)
        items = self.load_all(game.group_id)
        items = [g for g in items if g.id != game.id] + [game]
        path.write_text(json.dumps([g.to_dict() for g in items], indent=1),
                        encoding="utf-8")

    def load_all(self, group_id: str) -> list[AsyncGameState]:
        path = self._path(group_id)
        if not path.exists():
            return []
        try:
            return [AsyncGameState.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            return []

    def load(self, group_id: str, game_id: str) -> AsyncGameState | None:
        for game in self.load_all(group_id):
            if game.id == game_id:
                return game
        return None


class MurderMysteryGame:
    """Multi-day murder mystery. Phases: setup → investigating → accusation → solved."""

    PHASES = ("setup", "investigating", "accusation", "solved")

    def __init__(self, group_id: str, players: list[str], seed: int = 0,
                 clues: list[str] | None = None) -> None:
        rng = random.Random(seed or hash(group_id) & 0xFFFFFFFF)
        culprit = rng.choice(players) if players else ""
        self.state = AsyncGameState(
            id=new_short_id(), game="murder_mystery", group_id=group_id,
            players=list(players), phase="investigating",
            state={"culprit": culprit,
                   "clues": list(clues or ["a muddy footprint", "a torn invitation",
                                           "a missing key", "a whispered name"]),
                   "revealed": [], "accusations": {}, "solved_by": None},
            created_ts=time.time(), updated_ts=time.time())

    def reveal_clue(self) -> tuple[bool, str]:
        st = self.state.state
        if self.state.phase != "investigating":
            return False, f"no clues to reveal in phase {self.state.phase}"
        remaining = [c for c in st["clues"] if c not in st["revealed"]]
        if not remaining:
            self.state.phase = "accusation"
            return False, "all clues revealed — time to accuse"
        clue = remaining[0]
        st["revealed"].append(clue)
        if len(st["revealed"]) >= len(st["clues"]):
            self.state.phase = "accusation"
        self.state.updated_ts = time.time()
        return True, f"🔍 clue: {clue}"

    def accuse(self, player: str, suspect: str) -> tuple[bool, str]:
        if self.state.phase not in ("investigating", "accusation"):
            return False, f"cannot accuse in phase {self.state.phase}"
        if player not in self.state.players:
            return False, f"{player} is not playing"
        self.state.state["accusations"][player] = suspect
        if suspect == self.state.state["culprit"]:
            self.state.phase = "solved"
            self.state.state["solved_by"] = player
            self.state.updated_ts = time.time()
            return True, f"🎉 {player} solved it! The culprit was {suspect}."
        return False, f"❌ {player} accused {suspect} — wrong. The mystery continues."

    def status(self) -> str:
        st = self.state.state
        return (f"🕵️ murder mystery — {self.state.phase} · "
                f"{len(st['revealed'])}/{len(st['clues'])} clues revealed · "
                f"{len(st['accusations'])} accusations")


class ScavengerHuntGame:
    """Multi-day scavenger hunt. Phases: hunting → finished."""

    def __init__(self, group_id: str, players: list[str],
                 items: list[str] | None = None) -> None:
        self.state = AsyncGameState(
            id=new_short_id(), game="scavenger_hunt", group_id=group_id,
            players=list(players), phase="hunting",
            state={"items": list(items or ["a photo of a cat", "a local proverb",
                                           "a screenshot of today's weather",
                                           "a voice note saying hello"]),
                   "found": {}, "first_finish": None},
            created_ts=time.time(), updated_ts=time.time())

    def submit(self, player: str, item: str, proof: str = "") -> tuple[bool, str]:
        if self.state.phase != "hunting":
            return False, f"hunt is {self.state.phase}"
        if player not in self.state.players:
            return False, f"{player} is not playing"
        if item not in self.state.state["items"]:
            return False, f"{item!r} is not on the hunt list"
        found = self.state.state["found"].setdefault(player, [])
        if item in found:
            return False, f"{player} already found {item!r}"
        found.append(item)
        self.state.updated_ts = time.time()
        if len(found) >= len(self.state.state["items"]):
            self.state.phase = "finished"
            self.state.state["first_finish"] = player
            return True, f"🏁 {player} found everything — hunt complete!"
        return True, f"✅ {player} found {item!r} ({len(found)}/{len(self.state.state['items'])})"

    def standings(self) -> list[tuple[str, int]]:
        found = self.state.state["found"]
        return sorted(((p, len(found.get(p, []))) for p in self.state.players),
                      key=lambda r: (-r[1], r[0]))

    def status(self) -> str:
        top = self.standings()
        lead = f"leading: {top[0][0]} ({top[0][1]})" if top else "no finds yet"
        return f"🧭 scavenger hunt — {self.state.phase} · {lead}"


class ElectionGame:
    """Group election over days. Phases: nominations → voting → closed."""

    def __init__(self, group_id: str, players: list[str], office: str = "mayor") -> None:
        self.state = AsyncGameState(
            id=new_short_id(), game="election", group_id=group_id,
            players=list(players), phase="nominations",
            state={"office": office, "candidates": [], "votes": {},
                   "winner": None},
            created_ts=time.time(), updated_ts=time.time())

    def nominate(self, nominator: str, candidate: str) -> tuple[bool, str]:
        if self.state.phase != "nominations":
            return False, "nominations are closed"
        if candidate not in self.state.state["candidates"]:
            self.state.state["candidates"].append(candidate)
        return True, f"🗳️ {nominator} nominated {candidate} for {self.state.state['office']}"

    def open_voting(self) -> tuple[bool, str]:
        if self.state.phase != "nominations":
            return False, f"cannot open voting from {self.state.phase}"
        if not self.state.state["candidates"]:
            return False, "no candidates — nominate someone first"
        self.state.phase = "voting"
        return True, f"🗳️ voting open for {self.state.state['office']}: " + \
            ", ".join(self.state.state["candidates"])

    def vote(self, voter: str, candidate: str) -> tuple[bool, str]:
        if self.state.phase != "voting":
            return False, f"voting is not open (phase: {self.state.phase})"
        if candidate not in self.state.state["candidates"]:
            return False, f"{candidate} is not a candidate"
        if voter in self.state.state["votes"]:
            return False, f"{voter} already voted"
        self.state.state["votes"][voter] = candidate
        return True, f"✅ {voter} voted for {candidate}"

    def close(self) -> tuple[bool, str]:
        if self.state.phase != "voting":
            return False, f"cannot close from {self.state.phase}"
        tally: dict[str, int] = {}
        for cand in self.state.state["votes"].values():
            tally[cand] = tally.get(cand, 0) + 1
        if not tally:
            self.state.phase = "closed"
            return True, "election closed — no votes cast"
        winner = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        self.state.state["winner"] = winner
        self.state.phase = "closed"
        return True, f"🏆 {winner} wins {self.state.state['office']} with {tally[winner]} vote(s)"


# ───────────────────────── trivia from group material ───────────────────────

@dataclass
class TriviaQuestion:
    kind: str                     # guess_who | wisdom | book
    question: str
    options: list[str]
    answer: str
    source: str = ""              # the original message id / ref, if any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def check(self, choice: str) -> bool:
        return choice.strip() == self.answer


def build_trivia(messages: list[dict[str, Any]], *, seed: int = 0,
                 max_questions: int = 5) -> list[TriviaQuestion]:
    """Build trivia from injected group messages.

    ``messages`` are dicts with ``author`` / ``text`` and optional
    ``kind`` (``wisdom`` / ``book`` / anything) and ``id``. Deterministic
    given ``(messages, seed)``. Messages with < 3 distinct authors yield
    no guess-who questions (not enough options to be fair).
    """
    rng = random.Random(seed)
    usable = [m for m in messages
              if str(m.get("text", "")).strip() and str(m.get("author", "")).strip()]
    if not usable:
        return []
    questions: list[TriviaQuestion] = []
    authors = sorted({str(m["author"]) for m in usable})
    # Guess-who: quote a message, guess the author.
    if len(authors) >= 3:
        pool = list(usable)
        rng.shuffle(pool)
        for msg in pool:
            if len(questions) >= max_questions:
                break
            others = [a for a in authors if a != msg["author"]]
            rng.shuffle(others)
            options = sorted([msg["author"], *others[:3]])
            questions.append(TriviaQuestion(
                kind="guess_who",
                question=f"Who said this? “{msg['text'][:140]}”",
                options=options, answer=str(msg["author"]),
                source=str(msg.get("id", ""))))
    # Wisdom / book: quote a tagged message, guess its theme.
    tagged = [m for m in usable if str(m.get("kind", "")) in ("wisdom", "book")]
    rng.shuffle(tagged)
    for msg in tagged:
        if len(questions) >= max_questions + 2:
            break
        kind = str(msg["kind"])
        questions.append(TriviaQuestion(
            kind=kind,
            question=(f"From the group's {kind} collection — who shared this? "
                      f"“{msg['text'][:140]}”"),
            options=sorted(authors), answer=str(msg["author"]),
            source=str(msg.get("id", ""))))
    return questions[: max_questions + 2]
