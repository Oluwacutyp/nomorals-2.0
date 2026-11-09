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


_DDL = (
    "CREATE TABLE IF NOT EXISTS encounters ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "site TEXT NOT NULL, kind TEXT NOT NULL, vendor TEXT NOT NULL,"
    "started_at REAL NOT NULL, cleared INTEGER NOT NULL,"
    "via TEXT NOT NULL, takeover_needed INTEGER NOT NULL,"
    "ms INTEGER NOT NULL);"
    "CREATE TABLE IF NOT EXISTS attempts ("
    "encounter_id INTEGER NOT NULL, strategy TEXT NOT NULL,"
    "ok INTEGER NOT NULL, ms INTEGER NOT NULL,"
    "detail TEXT NOT NULL,"
    "FOREIGN KEY(encounter_id) REFERENCES encounters(id))"
)


class ChallengeMemory:
    """Record of challenge encounters + strategy outcomes.

    Native path: a SQLite file at ``path``. When a document store is
    resolved (Mongo program P5 — an explicit ``store`` or the boot
    ambient), the same API runs over the seam in the
    ``challenge-memory__*`` collections instead (registered in
    storage.sidecars), and the file's existing rows are imported
    once, through the same mapping, at first construction.
    """

    def __init__(self, path: str | Path, *, store: Any = None) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._port: Any = None
        from ..storage.sidecars import sidecar_port

        port = sidecar_port("challenge-memory", _DDL, ("encounters",),
                            self._path, store=store)
        if port is not None:
            self._imported_max_encounter = 0

            def write_encounters(rows: list[dict[str, Any]]) -> int:
                table = port.tables["encounters"]
                for row in rows:
                    table.put(row)
                    self._imported_max_encounter = max(
                        self._imported_max_encounter, int(row["id"]))
                return len(rows)

            def write_attempts(rows: list[dict[str, Any]]) -> int:
                ordinals: dict[int, int] = {}
                for row in rows:
                    encounter_id = int(row["encounter_id"])
                    ordinal = ordinals.get(encounter_id, 0)
                    ordinals[encounter_id] = ordinal + 1
                    port.put_raw("attempts", f"{encounter_id}:{ordinal}", {
                        "encounter_id": encounter_id,
                        "strategy": row["strategy"], "ok": row["ok"],
                        "ms": row["ms"], "detail": row["detail"],
                    })
                return len(rows)

            def seed_counter(_counts: dict[str, int]) -> None:
                doc = port.store.get(port.meta_collection, "counters") or {}
                if int(doc.get("encounters") or 0) < self._imported_max_encounter:
                    port.store.put(port.meta_collection, "counters", {
                        **doc, "encounters": self._imported_max_encounter})

            port.ensure_imported(
                {"encounters": write_encounters, "attempts": write_attempts},
                after_import=seed_counter)
            self._port = port
            return
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
        if self._port is not None:
            encounter_id = self._port.next_id("encounters")
            self._port.tables["encounters"].put({
                "id": encounter_id,
                "site": _site_of(site), "kind": (kind or "").lower(),
                "vendor": vendor or "",
                "started_at": started_at if started_at is not None else time.time(),
                "cleared": 1 if cleared else 0, "via": via or "",
                "takeover_needed": 1 if takeover_needed else 0,
                "ms": max(0, int(ms)),
            })
            for ordinal, (name, ok, attempt_ms) in enumerate(strategies):
                self._port.put_raw("attempts", f"{encounter_id}:{ordinal}", {
                    "encounter_id": encounter_id, "strategy": name,
                    "ok": 1 if ok else 0, "ms": max(0, int(attempt_ms)),
                    "detail": "",
                })
            return encounter_id
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
        if self._port is not None:
            table = self._port.tables["encounters"]
            if site:
                encounters = table.find(
                    {"site": _site_of(site), "kind": (kind or "").lower()})
            else:
                encounters = table.find({"kind": (kind or "").lower()})
            ids = [int(r["id"]) for r in encounters]
            rates: dict[str, list[int]] = {}
            if ids:
                for doc in self._port.find_raw(
                        "attempts", {"encounter_id": {"$in": ids}}):
                    entry = rates.setdefault(doc["strategy"], [0, 0])
                    entry[0] += 1 if doc["ok"] else 0
                    entry[1] += 1
            return rates
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
        if self._port is not None:
            filter_: dict[str, Any] = {}
            if site:
                filter_["site"] = _site_of(site)
            if kind:
                filter_["kind"] = kind.lower()
            rows = self._port.tables["encounters"].find(filter_ or None)
            total = len(rows)
            cleared = sum(int(r["cleared"]) for r in rows)
            takeovers = sum(int(r["takeover_needed"]) for r in rows)
            return {
                "encounters": total,
                "cleared": cleared,
                "cleared_rate": (cleared / total) if total else 0.0,
                "takeovers": takeovers,
                "takeover_rate": (takeovers / total) if total else 0.0,
            }
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
        if self._conn is None:
            return  # seam path: the store is shared, not organ-owned
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - close is best effort
            pass
