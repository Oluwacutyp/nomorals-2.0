"""Per-skill benchmarks: record runs, score reliability and latency.

Every skill run (via :class:`nomorals.skills.runner.SkillRunner` or by
hand) records ``(success, latency_ms)`` here.  :meth:`SkillBench.score`
turns the run history into a success-rate/latency summary — the number
the recall/ranking paths and the ``nm skill benchmark`` CLI read.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ..storage.db import Database

_log = get_logger(__name__)

__all__ = ["SkillBench", "BENCH_RUNS_DDL"]

BENCH_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS skill_bench_runs (
    id          TEXT PRIMARY KEY,
    skill_name  TEXT NOT NULL,
    version     TEXT NOT NULL DEFAULT '',
    success     INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bench_runs_skill
    ON skill_bench_runs(skill_name, created_at);
"""


class SkillBench:
    """Record skill runs and score them."""

    def __init__(self, db: "Database") -> None:
        self.db = db
        db.executescript(BENCH_RUNS_DDL)

    # ── write ───────────────────────────────────────────────────────────
    def record(self, skill_name: str, version: str, *, success: bool,
               latency_ms: float) -> str:
        """Record one run.  Returns the run id.  Never raises — benchmark
        bookkeeping must not sink the run it measures."""
        run_id = new_short_id("br")
        try:
            self.db.execute(
                "INSERT INTO skill_bench_runs (id, skill_name, version, "
                "success, latency_ms, created_at) VALUES (?,?,?,?,?,?)",
                (run_id, skill_name, version, 1 if success else 0,
                 max(0.0, float(latency_ms)), time.time()),
            )
        except Exception as exc:  # noqa: BLE001 - bench must never sink a run
            _log.debug("skill bench record failed for %s: %s", skill_name,
                       exc)
        return run_id

    # ── read ────────────────────────────────────────────────────────────
    def score(self, skill_name: str, *, version: str = "",
              limit: int = 500) -> dict[str, Any]:
        """Success-rate/latency summary over the recorded runs.

        ``version=""`` aggregates every version; pass one to scope it.
        Empty history yields a zeroed summary (``runs: 0``), never an
        exception.
        """
        params: list[Any] = [skill_name]
        sql = ("SELECT version, success, latency_ms, created_at FROM "
               "skill_bench_runs WHERE skill_name=?")
        if version:
            sql += " AND version=?"
            params.append(version)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.db.query(sql, tuple(params))

        per_version: dict[str, dict[str, Any]] = {}
        for row in rows:
            bucket = per_version.setdefault(row["version"] or "", {
                "runs": 0, "successes": 0, "latencies": []})
            bucket["runs"] += 1
            bucket["successes"] += int(row["success"])
            bucket["latencies"].append(float(row["latency_ms"]))

        def _summarize(bucket: dict[str, Any]) -> dict[str, Any]:
            runs = bucket["runs"]
            lats = sorted(bucket["latencies"])
            return {
                "runs": runs,
                "successes": bucket["successes"],
                "success_rate": round(bucket["successes"] / runs, 4)
                if runs else 0.0,
                "avg_latency_ms": round(sum(lats) / len(lats), 3)
                if lats else 0.0,
                "min_latency_ms": round(lats[0], 3) if lats else 0.0,
                "max_latency_ms": round(lats[-1], 3) if lats else 0.0,
            }

        total = {"runs": 0, "successes": 0, "latencies": []}
        for bucket in per_version.values():
            total["runs"] += bucket["runs"]
            total["successes"] += bucket["successes"]
            total["latencies"].extend(bucket["latencies"])
        summary = _summarize(total)
        summary.update({
            "skill": skill_name,
            "version": version,
            "versions": {v or "(unversioned)": _summarize(b)
                         for v, b in sorted(per_version.items())},
            "last_run_at": max((r["created_at"] for r in rows),
                               default=None),
        })
        return summary

    def recent(self, skill_name: str = "",
               limit: int = 20) -> list[dict[str, Any]]:
        """Newest runs first; pass a name to scope to one skill."""
        limit = max(1, int(limit))
        if skill_name:
            rows = self.db.query(
                "SELECT * FROM skill_bench_runs WHERE skill_name=? "
                "ORDER BY created_at DESC LIMIT ?", (skill_name, limit))
        else:
            rows = self.db.query(
                "SELECT * FROM skill_bench_runs ORDER BY created_at DESC "
                "LIMIT ?", (limit,))
        return [{
            "id": r["id"],
            "skill": r["skill_name"],
            "version": r.get("version", ""),
            "success": bool(r["success"]),
            "latency_ms": float(r["latency_ms"]),
            "created_at": float(r["created_at"]),
        } for r in rows]

    def overview(self, limit_per_skill: int = 200) -> list[dict[str, Any]]:
        """One score summary per skill that has any recorded runs."""
        rows = self.db.query(
            "SELECT DISTINCT skill_name FROM skill_bench_runs "
            "ORDER BY skill_name")
        return [self.score(r["skill_name"], limit=limit_per_skill)
                for r in rows]
