"""Challenge memory: what worked against which wall, remembered.

Every challenge encounter is recorded (site, challenge kind, strategies
tried with outcomes, what cleared it, whether owner takeover was
needed, time-to-clear). The next encounter on the same site/kind ranks
strategies by past success — Devon tries what worked last time first
instead of re-learning from scratch every run.

Scoring is Laplace-smoothed ``(wins + 1) / (tries + 2)`` so a strategy
with one lucky win does not permanently outrank an untried one, and an
untried strategy (0.5) still beats a strategy that has failed repeatedly.
Site-specific history wins over global kind history; with no history at
all the caller's default order is preserved exactly.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["ChallengeMemory"]


def _site_of(url_or_site: str) -> str:
    """Registrable-ish host for a URL or a bare site name.

    No public-suffix list here — the last two labels are good enough for
    grouping (``sub.example.com`` and ``example.com`` share a record).
    """
    raw = (url_or_site or "").strip().lower()
    for prefix in ("https://", "http://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    raw = raw.split("/")[0].split(":")[0].split("?")[0]
    raw = raw.removeprefix("www.")
    parts = [p for p in raw.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return raw or "unknown"


class ChallengeMemory:
    """SQLite-backed record of challenge encounters + strategy outcomes."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS encounters ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "site TEXT NOT NULL, kind TEXT NOT NULL, vendor TEXT NOT NULL,"
            "started_at REAL NOT NULL, cleared INTEGER NOT NULL,"
            "via TEXT NOT NULL, takeover_needed INTEGER NOT NULL,"
            "ms INTEGER NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS attempts ("
            "encounter_id INTEGER NOT NULL, strategy TEXT NOT NULL,"
            "ok INTEGER NOT NULL, ms INTEGER NOT NULL,"
            "detail TEXT NOT NULL,"
            "FOREIGN KEY(encounter_id) REFERENCES encounters(id))"
        )
        self._conn.commit()

    # -- recording -----------------------------------------------------------
    def record(self, *, site: str, kind: str, vendor: str = "",
               strategies: Iterable[tuple[str, bool, int]] = (),
               cleared: bool = False, via: str = "",
               takeover_needed: bool = False,
               ms: int = 0, started_at: float | None = None) -> int:
        """Append one encounter; returns its row id."""
        cur = self._conn.execute(
            "INSERT INTO encounters (site, kind, vendor, started_at, "
            "cleared, via, takeover_needed, ms) VALUES (?,?,?,?,?,?,?,?)",
            (_site_of(site), (kind or "").lower(), vendor or "",
             started_at if started_at is not None else time.time(),
             1 if cleared else 0, via or "", 1 if takeover_needed else 0,
             max(0, int(ms))),
        )
        encounter_id = int(cur.lastrowid or 0)
        for name, ok, attempt_ms in strategies:
            self._conn.execute(
                "INSERT INTO attempts (encounter_id, strategy, ok, ms, "
                "detail) VALUES (?,?,?,?,?)",
                (encounter_id, name, 1 if ok else 0,
                 max(0, int(attempt_ms)), ""),
            )
        self._conn.commit()
        return encounter_id

    # -- ranking ---------------------------------------------------------------
    def _rates(self, *, site: str | None, kind: str) -> dict[str, list[int]]:
        """{strategy: [wins, tries]} for a (site, kind) or global kind."""
        if site:
            rows = self._conn.execute(
                "SELECT a.strategy, a.ok FROM attempts a "
                "JOIN encounters e ON e.id = a.encounter_id "
                "WHERE e.site = ? AND e.kind = ?",
                (_site_of(site), (kind or "").lower()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT a.strategy, a.ok FROM attempts a "
                "JOIN encounters e ON e.id = a.encounter_id "
                "WHERE e.kind = ?",
                ((kind or "").lower(),),
            ).fetchall()
        rates: dict[str, list[int]] = {}
        for name, ok in rows:
            entry = rates.setdefault(name, [0, 0])
            entry[0] += 1 if ok else 0
            entry[1] += 1
        return rates

    def rank_strategies(self, site: str, kind: str,
                        candidates: list[str]) -> list[str]:
        """Order ``candidates`` best-first from past outcomes.

        Site-specific rates win; a strategy with no site history falls
        back to its global kind rate; with no history anywhere the
        caller's order is preserved (stable sort).
        """
        site_rates = self._rates(site=site, kind=kind)
        kind_rates = self._rates(site=None, kind=kind)

        def score(name: str) -> tuple[float, int]:
            rates = site_rates.get(name) or kind_rates.get(name)
            if rates is None:
                return (0.5, 0)  # untried: neutral, keeps default order
            wins, tries = rates
            return ((wins + 1) / (tries + 2), tries)

        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda pair: (-score(pair[1])[0], pair[0]))
        return [name for _, name in indexed]

    # -- stats -----------------------------------------------------------------
    def stats(self, *, site: str | None = None,
              kind: str | None = None) -> dict[str, Any]:
        """Aggregate counts: encounters, cleared rate, takeover rate."""
        clauses, params = [], []
        if site:
            clauses.append("site = ?")
            params.append(_site_of(site))
        if kind:
            clauses.append("kind = ?")
            params.append(kind.lower())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(cleared),0), "
            f"COALESCE(SUM(takeover_needed),0) FROM encounters {where}",
            params,
        ).fetchone() or (0, 0, 0)
        total, cleared, takeovers = (int(row[0]), int(row[1]), int(row[2]))
        return {
            "encounters": total,
            "cleared": cleared,
            "cleared_rate": (cleared / total) if total else 0.0,
            "takeovers": takeovers,
            "takeover_rate": (takeovers / total) if total else 0.0,
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - close is best effort
            pass
