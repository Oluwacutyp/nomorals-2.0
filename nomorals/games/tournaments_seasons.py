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

Production wiring (AA-16): the engine folds every finished group
game's results into the group's season board
(:func:`record_season_results`, persisted by :class:`SeasonStore`),
a durable hourly scheduler job (:func:`ensure_season_tick_job` →
``__season_tick__`` tool → :func:`run_season_tick_once`) rolls each
group into the current quarter and pays closed seasons' prizes
through the group economy — the one purse — and ``/game season``
renders the live board. Before this wiring the season system had
no production caller at all: no game ever scored, no season ever
closed, no prize ever moved.

Only stdlib + ``nomorals.core``. No owner-data imports.
"""

from __future__ import annotations

import calendar
import contextlib
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
    "SeasonStore", "seasons_root", "record_season_results",
    "current_season", "run_season_tick_once", "ensure_season_tick_job",
    "SEASON_JOB", "SEASON_SPEC", "SEASON_TOOL",
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


# ─────────────────── production wiring (AA-16) ─────────────────────────────
# The classes above are the organ; everything below is its production
# life: persistence, the engine's result feed, the quarterly rollover
# + prize tick, and the durable scheduler job.

#: Durable season job: name, tool payload, cadence. Hourly is plenty
#: for a quarterly boundary — the tick is a cheap directory scan, and
#: the rollover also happens lazily on the next recorded result or
#: board read, so a boundary is never missed by more than the tick.
SEASON_JOB = "season tick"
SEASON_TOOL = "__season_tick__"
SEASON_SPEC = "every 1h"


def _seasons_default_root() -> Path:
    return Path.home() / ".devon" / "community" / "seasons"


def seasons_root(context: Any = None) -> Path:
    """The season store root: the community dir's ``seasons`` folder
    when the runtime context carries one (the same resolution the
    community orchestration uses), else the default community root
    the async-game store also lives under."""
    settings = getattr(context, "settings", None)
    root = getattr(settings, "community_dir", None)
    if root:
        return Path(root) / "seasons"
    return _seasons_default_root()


def _quarter_bounds(ts: float) -> tuple[float, float]:
    """(starts_at, ends_at) unix bounds of the quarter holding ``ts``
    (UTC, computed with calendar arithmetic — no tz objects)."""
    tm = time.gmtime(ts)
    start_month = 3 * ((tm.tm_mon - 1) // 3) + 1  # 1, 4, 7, 10
    start = calendar.timegm((tm.tm_year, start_month, 1, 0, 0, 0, 0, 0, 0))
    if start_month == 10:
        end = calendar.timegm((tm.tm_year + 1, 1, 1, 0, 0, 0, 0, 0, 0))
    else:
        end = calendar.timegm(
            (tm.tm_year, start_month + 3, 1, 0, 0, 0, 0, 0, 0))
    return float(start), float(end)


class SeasonStore:
    """JSON persistence for group seasons, one file per group.

    The file holds the active ``season`` + ``board``, a
    display-name → member-id map (prize payouts credit through it —
    member ids are routing data and never reach member-facing text),
    and the ``archive`` of closed seasons with their payout records
    (``payouts`` is None until the tick has attempted them;
    ``settled`` flips only when every prize actually landed).
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else _seasons_default_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, group_id: str) -> Path:
        safe = group_id.replace("/", "_").replace("\\", "_").strip() or "group"
        return self.root / f"{safe}.json"

    def load(self, group_id: str) -> dict[str, Any]:
        path = self._path(group_id)
        if not path.exists():
            return {"group_id": group_id, "season": None, "board": None,
                    "members": {}, "archive": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"group_id": group_id, "season": None, "board": None,
                    "members": {}, "archive": []}
        data.setdefault("group_id", group_id)
        data.setdefault("season", None)
        data.setdefault("board", None)
        data.setdefault("members", {})
        data.setdefault("archive", [])
        return data

    def save(self, group_id: str, data: dict[str, Any]) -> None:
        path = self._path(group_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)

    def group_ids(self) -> list[str]:
        """Groups with a season file, by the id stored inside each
        file (the filename is only a sanitized routing key)."""
        out: list[str] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                gid = json.loads(path.read_text(encoding="utf-8")).get("group_id")
            except (json.JSONDecodeError, OSError):
                continue
            if gid:
                out.append(str(gid))
        return out


def _rollover(data: dict[str, Any], group_id: str, now: float) -> bool:
    """Close a stale active season into the archive and open the
    current quarter's. A stale season with points is archived for
    prize payout; an empty one simply lapses. Returns True when a
    season with results was closed."""
    cur_id = season_id_for(now)
    season = data.get("season")
    if season and season.get("id") == cur_id \
            and season.get("status") == "active":
        return False
    closed = False
    if season:
        board = data.get("board") or {}
        if board.get("points"):
            archived = dict(season)
            archived["status"] = "closed"
            data.setdefault("archive", []).append({
                "season": archived, "board": board,
                "members": dict(data.get("members") or {}),
                "payouts": None, "settled": False})
            closed = True
    starts, ends = _quarter_bounds(now)
    data["season"] = Season(id=cur_id, group_id=group_id,
                            name=f"Season {cur_id}", starts_at=starts,
                            ends_at=ends, status="active").to_dict()
    data["board"] = SeasonLeaderboard(
        season_id=cur_id, group_id=group_id).to_dict()
    data["members"] = {}
    return closed


def record_season_results(group_id: str, game: str,
                          results: list[tuple[str, str, int, bool]], *,
                          context: Any = None, root: Path | str | None = None,
                          now: float | None = None) -> dict[str, Any]:
    """Fold one finished group game's results into the season board.

    ``results`` are ``(player_name, member_id, points, won)`` tuples —
    the engine's settle produces them per human at the table. Points
    only: no money moves here. Prizes are the tick's job, paid
    through the group economy when the season closes.
    """
    now = now if now is not None else time.time()
    store = SeasonStore(root if root is not None else seasons_root(context))
    data = store.load(group_id)
    _rollover(data, group_id, now)
    board = SeasonLeaderboard.from_dict(data["board"])
    members = dict(data.get("members") or {})
    recorded = 0
    for name, member_id, points, won in results:
        if not name:
            continue
        board.add_result(str(name), int(points), game=game, won=bool(won))
        if member_id and str(name) not in members:
            members[str(name)] = str(member_id)
        recorded += 1
    data["board"] = board.to_dict()
    data["members"] = members
    store.save(group_id, data)
    return {"season": board.season_id, "recorded": recorded,
            "players": len(board.standings())}


def current_season(group_id: str, *, context: Any = None,
                   root: Path | str | None = None,
                   now: float | None = None) -> tuple[dict[str, Any], SeasonLeaderboard]:
    """The group's (season, board) for the current quarter — the
    store rolls over first, so a surface never renders a stale
    season as the live one."""
    now = now if now is not None else time.time()
    store = SeasonStore(root if root is not None else seasons_root(context))
    data = store.load(group_id)
    before = (data.get("season") or {}).get("id")
    _rollover(data, group_id, now)
    if (data.get("season") or {}).get("id") != before:
        store.save(group_id, data)
    return data["season"], SeasonLeaderboard.from_dict(data["board"])


def _prize_economy(context: Any = None) -> Any:
    """The group economy at the surface's root (workspace
    ``data/social/group_economy`` — the root GroupSurface uses), or
    the economy's own default root when no workspace is known."""
    from ..community.group_economy import GroupEconomy

    settings = getattr(context, "settings", None)
    ws = getattr(settings, "workspace_dir", None)
    if ws:
        return GroupEconomy(Path(ws) / "data" / "social" / "group_economy")
    return GroupEconomy()


def _ledger_for(econ: Any, group_id: str, members: dict[str, str],
                ) -> Callable[[str, int, str], bool]:
    """A SeasonPrize ledger over the group economy — the one purse.

    A season prize is a competition *payout*, not an earn, so it
    settles the way the token-economy facade (AA-21) settles its
    non-earn movements: appended straight to the group's ledger as
    kind ``prize``. Routing it through ``award()`` would make the
    top prizes unpayable forever — award()'s daily earn cap (200)
    sits below the rank-1 prize (500), and a blocked earn never
    lands no matter how often it is retried. The winner is verified
    on first credit, exactly as ``GroupSurface.game_award`` verifies
    a member before crediting game winnings."""

    def ledger(player: str, amount: int, reason: str) -> bool:
        member = members.get(player)
        if not member or int(amount) <= 0:
            return False
        try:
            if not econ.is_verified(group_id, member):
                econ.verify_member(group_id, member)
            econ._append(group_id, econ._new_entry(
                group_id, member, "prize", int(amount), reason,
                ref="season-prize"))
            return True
        except Exception as exc:  # noqa: BLE001 — one prize never kills the tick
            _log.warning("season prize payout failed for %s in %s: %s",
                         player, group_id, exc)
            return False

    return ledger


def run_season_tick_once(context: Any = None, *,
                         root: Path | str | None = None,
                         ledger_for: Callable[[str, dict[str, str]],
                                                Callable[[str, int, str], bool]] | None = None,
                         now: float | None = None) -> dict[str, Any]:
    """One production season tick: roll every known group into the
    current quarter, then pay every closed season's outstanding
    prizes through the group economy.

    Returns a summary for the scheduler journal — the tick posts
    nothing anywhere (a routine job with nothing to report stays
    silent). ``ledger_for`` is injectable for tests; production
    builds the group-economy ledger per closed season.
    """
    now = now if now is not None else time.time()
    store = SeasonStore(root if root is not None else seasons_root(context))
    summary: dict[str, Any] = {"ok": True, "groups": 0, "closed": [],
                               "paid": [], "unpaid": []}
    econ: Any = None
    for group_id in store.group_ids():
        data = store.load(group_id)
        if _rollover(data, group_id, now):
            summary["closed"].append(
                {"group": group_id, "season": data["archive"][-1]["season"]["id"]})
        for entry in data.get("archive", []):
            if entry.get("settled"):
                continue
            board = SeasonLeaderboard.from_dict(entry["board"])
            members = dict(entry.get("members") or {})
            if ledger_for is not None:
                ledger = ledger_for(group_id, members)
            else:
                if econ is None:
                    econ = _prize_economy(context)
                ledger = _ledger_for(econ, group_id, members)
            payouts = SeasonPrize(
                season_id=entry["season"]["id"]).distribute(board, ledger)
            entry["payouts"] = payouts
            entry["settled"] = all(p.get("paid") for p in payouts)
            for p in payouts:
                row = {"group": group_id, "season": entry["season"]["id"],
                       "player": p["player"], "rank": p["rank"],
                       "amount": p["amount"]}
                summary["paid" if p.get("paid") else "unpaid"].append(row)
        store.save(group_id, data)
        summary["groups"] += 1
    return summary


def _job_field(job: Any, name: str) -> Any:
    if isinstance(job, dict):
        return job.get(name, "")
    return getattr(job, name, "")


def _spec_matches(stored: Any, spec: str) -> bool:
    return str(stored or "") == spec


def ensure_season_tick_job(scheduler: Any, *,
                           spec: str = SEASON_SPEC) -> Any:
    """Register the durable season tick job. Idempotent — mirrors
    ``ensure_failure_loop_job``: an existing job at the requested
    cadence is returned as-is, a stale one is replaced. Best-effort:
    never raises (a missing scheduler just means no scheduled tick
    runs; rollover still happens lazily on recorded results). The
    job is ``routine=True`` — runs journal with AA-67 honesty
    (skipped/missed runs record ok=False) and never page."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            if _job_field(job, "name") == SEASON_JOB:
                if _spec_matches(_job_field(job, "spec"), spec):
                    return job
                with contextlib.suppress(Exception):
                    scheduler.remove(_job_field(job, "id"))
                break
        return scheduler.add(SEASON_JOB, spec, "tool",
                             {"tool": SEASON_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


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
