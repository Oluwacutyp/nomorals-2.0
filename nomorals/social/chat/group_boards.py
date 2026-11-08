"""Leaderboards (DESIGN §14) — one board organ, many boards.

A board is ``<scope>:<metric>`` over a living data source:

- ``group:<metric>`` — one group's own standings (sparks worth,
  gems, messages, REVIEWED contributions, gang strength);
- ``platform:<platform>:<metric>`` — every group Devon runs on
  one platform, members aggregated across those groups;
- ``global:<metric>`` — cross-platform: the same aggregates over
  every group, with people stitched across platforms by the §32
  identity bridge (a shared normalized username, or the same
  platform account seen in several groups) so one person ranks
  ONCE, their platforms noted;
- ``community:<id>:<metric>`` — the groups carrying that
  community tag in their settings (the scope exists where
  communities exist in the store; untagged communities render
  the honest empty board);
- ``game:<game>:<metric>`` / ``games:<metric>`` — the games
  spine (ELO, wins, points per game; XP, coins, wins, mastery
  across games), fed by a provider the runtime injects — the
  surface owns no games database.
- ``gangs:<metric>`` — the §36 gang scope: global gang records
  ranked across every group (prestige, chapters held, roster
  size), read off the gang registry.

Reputation is NEVER a board metric: flags are a safety record,
not a standing, and nobody gets ranked by them (§14/§32).

Every board is a LIVING surface: publishing stores a bounded
snapshot (the last :data:`SNAPSHOTS_KEPT` per board) in a book
beside the group store, and renders show MOVEMENT against the
newest snapshot — ▲n climbed n places, ▼n fell, — held, "new"
had no published place before.  Peeking never advances the
snapshot; only a publish does, so a curious look can't erase
the movement the next scheduled post should show.

This module is data-only (id grammar, the snapshot book,
movement math).  Row assembly lives on the surface, which owns
the store and the organs; rendering goes through typography
like every other presented surface.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .failures import record_failure

__all__ = [
    "BOARD_LIMIT",
    "SNAPSHOTS_KEPT",
    "BoardBook",
    "GROUP_METRICS",
    "PEOPLE_METRICS",
    "GAME_METRICS",
    "GAMES_METRICS",
    "GANG_METRICS",
    "board_title",
    "is_game_flavored",
    "metric_format",
    "movement_marks",
    "parse_board_id",
]

#: Rows kept per snapshot / shown per render.
BOARD_LIMIT = 25
#: Published snapshots retained per board (bounded by design).
SNAPSHOTS_KEPT = 4

#: Metrics over one group's own ledgers.  ``fmt`` picks the
#: value rendering: sparks/gems carry their currency glyphs,
#: counts are plain tabular numbers.
GROUP_METRICS: dict[str, dict[str, str]] = {
    "sparks": {"label": "sparks held", "fmt": "sparks"},
    "gems": {"label": "gems held", "fmt": "gems"},
    "messages": {"label": "messages sent", "fmt": "count"},
    "contributions": {"label": "reviewed contributions",
                      "fmt": "count"},
    "gangs": {"label": "gang strength", "fmt": "count"},
}

#: Metrics that aggregate over people across groups (platform /
#: global / community scopes) — the per-group currencies read as
#: standings: a person's sparks across groups are their score,
#: never a transferable balance (money never moves here).
PEOPLE_METRICS: dict[str, dict[str, str]] = {
    "sparks": {"label": "sparks held", "fmt": "sparks"},
    "messages": {"label": "messages sent", "fmt": "count"},
    "contributions": {"label": "reviewed contributions",
                      "fmt": "count"},
}

#: Per-game metrics, read off the games spine's profiles +
#: Glicko table through the injected provider.
GAME_METRICS: dict[str, dict[str, str]] = {
    "elo": {"label": "ELO rating", "fmt": "count"},
    "wins": {"label": "wins", "fmt": "count"},
    "points": {"label": "points", "fmt": "count"},
}

#: Whole-spine metrics over the games profiles.
GAMES_METRICS: dict[str, dict[str, str]] = {
    "xp": {"label": "XP", "fmt": "count"},
    "coins": {"label": "game coins", "fmt": "count"},
    "wins": {"label": "wins", "fmt": "count"},
    "mastery": {"label": "mastery points", "fmt": "count"},
}

#: Metrics over the §36 global gang registry — standings of
#: gangs themselves, across every group they hold chapters in.
GANG_METRICS: dict[str, dict[str, str]] = {
    "prestige": {"label": "gang prestige", "fmt": "count"},
    "chapters": {"label": "chapters held", "fmt": "count"},
    "members": {"label": "members worldwide", "fmt": "count"},
}

_SCOPES = ("group", "platform", "global", "community", "game",
           "games", "gangs")


def parse_board_id(board_id: Any) -> dict[str, Any] | None:
    """Parse + validate a board id.  Returns the parts dict or
    None for anything malformed or unknown — callers render the
    honest "no such board" answer, never a guessed board."""
    raw = str(board_id or "").strip().lower()
    if not raw:
        return None
    parts = raw.split(":")
    scope = parts[0]
    if scope not in _SCOPES:
        return None
    out: dict[str, Any] = {"scope": scope, "id": raw}
    if scope == "group":
        if len(parts) != 2 or parts[1] not in GROUP_METRICS:
            return None
        out["metric"] = parts[1]
    elif scope == "platform":
        if len(parts) != 3 or not parts[1] \
                or parts[2] not in PEOPLE_METRICS:
            return None
        out["platform"] = parts[1]
        out["metric"] = parts[2]
    elif scope == "global":
        if len(parts) != 2 or parts[1] not in PEOPLE_METRICS:
            return None
        out["metric"] = parts[1]
    elif scope == "community":
        if len(parts) != 3 or not parts[1] \
                or parts[2] not in PEOPLE_METRICS:
            return None
        out["community"] = parts[1]
        out["metric"] = parts[2]
    elif scope == "game":
        if len(parts) != 3 or not parts[1] \
                or parts[2] not in GAME_METRICS:
            return None
        out["game"] = parts[1]
        out["metric"] = parts[2]
    elif scope == "games":
        if len(parts) != 2 or parts[1] not in GAMES_METRICS:
            return None
        out["metric"] = parts[1]
    elif scope == "gangs":
        if len(parts) != 2 or parts[1] not in GANG_METRICS:
            return None
        out["metric"] = parts[1]
    return out


def _metric_meta(parsed: dict[str, Any]) -> dict[str, str]:
    scope = parsed["scope"]
    if scope == "group":
        return GROUP_METRICS[parsed["metric"]]
    if scope in ("platform", "global", "community"):
        return PEOPLE_METRICS[parsed["metric"]]
    if scope == "game":
        return GAME_METRICS[parsed["metric"]]
    if scope == "gangs":
        return GANG_METRICS[parsed["metric"]]
    return GAMES_METRICS[parsed["metric"]]


def metric_format(parsed: dict[str, Any]) -> str:
    """The value format of a parsed board's metric."""
    return _metric_meta(parsed)["fmt"]


def board_title(parsed: dict[str, Any]) -> str:
    """The human title of a board — what the card head says.
    Titles name standings, never internal ids."""
    label = _metric_meta(parsed)["label"]
    scope = parsed["scope"]
    if scope == "group":
        return f"{label} — this group"
    if scope == "platform":
        return f"{label} — {parsed['platform']}"
    if scope == "global":
        return f"{label} — everywhere"
    if scope == "community":
        return f"{label} — community {parsed['community']}"
    if scope == "game":
        return f"{parsed['game']} — {label}"
    if scope == "gangs":
        return f"{label} — all gangs, everywhere"
    return f"games — {label}"


def is_game_flavored(parsed: dict[str, Any]) -> bool:
    """Game-flavored boards render on the board preset; the
    rest render clean (§14 via §27)."""
    return parsed["scope"] in ("game", "games")


def movement_marks(previous: list[dict[str, Any]] | None,
                   rows: list[dict[str, Any]]) -> dict[str, str]:
    """Movement of the CURRENT rows against a published
    snapshot's rows: ``{row_key: mark}`` where the mark is
    ``▲n`` / ``▼n`` / ``—`` / ``new`` by RANK movement (the
    board is a ranking; places are what move).  No snapshot →
    no marks at all (nothing honest to compare against)."""
    if not previous:
        return {}
    before = {str(r.get("key") or ""): i
              for i, r in enumerate(previous)}
    marks: dict[str, str] = {}
    for i, row in enumerate(rows):
        key = str(row.get("key") or "")
        if key not in before:
            marks[key] = "new"
        elif before[key] == i:
            marks[key] = "—"
        elif before[key] > i:
            marks[key] = f"▲{before[key] - i}"
        else:
            marks[key] = f"▼{i - before[key]}"
    return marks


class BoardBook:
    """The published-snapshot book: boards.json beside the
    group store.  Snapshots are presentation state, not
    evidence — a corrupt book is quarantined and restarted
    (movement simply re-seeds), never raised."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: dict[str, Any] = {"boards": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) \
                    and isinstance(raw.get("boards"), dict):
                self.data = raw
        except (ValueError, OSError):
            try:
                bad = self.path.with_suffix(
                    self.path.suffix + ".corrupt")
                self.path.replace(bad)
            except OSError as exc:
                record_failure("group_boards._load", exc)
            self.data = {"boards": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.path)

    def latest(self, board_id: str) -> list[dict[str, Any]] | None:
        """The newest published snapshot's rows, or None when
        the board was never published."""
        entry = (self.data.get("boards") or {}).get(board_id) \
            or {}
        snaps = entry.get("snapshots") or []
        if not snaps:
            return None
        return list(snaps[-1].get("rows") or [])

    def publish(self, board_id: str, rows: list[dict[str, Any]],
                *, now: float | None = None) -> None:
        """Record a publish: the rows AS RANKED become the
        baseline the next render's movement reads.  Bounded —
        the last SNAPSHOTS_KEPT snapshots, BOARD_LIMIT rows."""
        now = now if now is not None else time.time()
        boards = self.data.setdefault("boards", {})
        entry = boards.setdefault(
            board_id, {"snapshots": [], "published": 0})
        entry["snapshots"] = (entry.get("snapshots") or []) + [{
            "ts": now,
            "rows": [{"key": str(r.get("key") or ""),
                      "value": int(r.get("value") or 0)}
                     for r in rows[:BOARD_LIMIT]],
        }]
        entry["snapshots"] = entry["snapshots"][-SNAPSHOTS_KEPT:]
        entry["published"] = int(entry.get("published") or 0) + 1
        self.save()
