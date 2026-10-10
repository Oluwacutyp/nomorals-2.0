"""Per-skill benchmarks: record runs, score reliability and latency.

Every skill run (via :class:`nomorals.skills.runner.SkillRunner` or by
hand) records ``(success, latency_ms)`` here.  :meth:`SkillBench.score`
turns the run history into a success-rate/latency summary — the number
the recall/ranking paths and the ``nm skill benchmark`` CLI read.

Summaries report percentiles (p50/p95/p99), not just averages —
averages hide the tails where bad versions live.  :meth:`canary_gate`
is the automated promotion gate (Argo Rollouts AnalysisTemplate shape):
it answers "is this skill version safe to pin?" from the recorded
evidence, and :meth:`regression` diffs a candidate version against its
baseline.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ..storage.db import Database

    from .runner import StepResult

_log = get_logger(__name__)

__all__ = ["SkillBench", "BENCH_RUNS_DDL", "BENCH_STEPS_DDL",
           "format_score"]

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

BENCH_STEPS_DDL = """
CREATE TABLE IF NOT EXISTS skill_bench_steps (
    id          TEXT PRIMARY KEY,
    skill_name  TEXT NOT NULL,
    version     TEXT NOT NULL DEFAULT '',
    step_index  INTEGER NOT NULL,
    tool        TEXT NOT NULL DEFAULT '',
    success     INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bench_steps_skill
    ON skill_bench_steps(skill_name, version, step_index, created_at);
"""


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile over an ascending-sorted list."""
    if not sorted_vals:
        return 0.0
    rank = max(1, int(-(-len(sorted_vals) * pct // 1)))  # ceil(p*n)
    rank = min(rank, len(sorted_vals))
    return sorted_vals[rank - 1]


def format_score(summary: dict[str, Any], *, spark: str = "") -> str:
    """Render a :meth:`SkillBench.score` summary as readable text.

    Pass ``spark=`` with a sparkline (e.g. from recent runs) to append
    it; the CLI builds one via ``recent()``.
    """
    name = summary.get("skill", "?")
    version = summary.get("version") or "all versions"
    lines = [f"benchmark {name} ({version}): "
             f"{summary.get('runs', 0)} runs"]
    rate = summary.get("success_rate", 0.0)
    bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
    lines.append(f"  success {rate * 100:5.1f}%  {bar}")
    lines.append(
        f"  latency ms  avg {summary.get('avg_latency_ms', 0.0):8.1f}  "
        f"p50 {summary.get('p50_latency_ms', 0.0):8.1f}  "
        f"p95 {summary.get('p95_latency_ms', 0.0):8.1f}  "
        f"p99 {summary.get('p99_latency_ms', 0.0):8.1f}")
    versions = summary.get("versions") or {}
    if len(versions) > 1:
        lines.append("  per version:")
        for ver, sub in sorted(versions.items()):
            lines.append(
                f"    {ver or '(unversioned)':<14} "
                f"{sub['runs']:>4} runs  "
                f"{sub['success_rate'] * 100:5.1f}% ok  "
                f"p95 {sub['p95_latency_ms']:8.1f}ms")
    if spark:
        lines.append(f"  recent: {spark}")
    return "\n".join(lines)


def sparkline(runs: list[dict[str, Any]], width: int = 40) -> str:
    """ASCII success/failure sparkline, newest on the right.

    ``█`` = success, ``·`` = failure — the shape of recent reliability
    at a glance.
    """
    cells = runs[:max(1, width)]
    cells = list(reversed(cells))  # oldest first, newest last
    return "".join("█" if r.get("success") else "·" for r in cells)


class SkillBench:
    """Record skill runs and score them."""

    def __init__(self, db: "Database") -> None:
        self.db = db
        db.executescript(BENCH_RUNS_DDL)
        db.executescript(BENCH_STEPS_DDL)

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

    def record_steps(self, skill_name: str, version: str,
                     steps: "list[StepResult]") -> None:
        """Record per-step latencies for one run.  Never raises.

        Per-step data is what makes a slow version diagnosable: the
        run-level average says "slower", the step table says *which*
        tool got slower.
        """
        now = time.time()
        try:
            for step in steps:
                self.db.execute(
                    "INSERT INTO skill_bench_steps (id, skill_name, "
                    "version, step_index, tool, success, latency_ms, "
                    "created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (new_short_id("bs"), skill_name, version,
                     step.index, step.tool, 1 if step.ok else 0,
                     max(0.0, float(step.latency_ms)), now),
                )
        except Exception as exc:  # noqa: BLE001 - bench never sinks a run
            _log.debug("skill bench step record failed for %s: %s",
                       skill_name, exc)

    def prune(self, older_than_days: float = 90.0) -> dict[str, int]:
        """Delete bench rows older than the retention window.

        Benchmark tables grow forever otherwise.  Returns counts removed.
        Never raises.
        """
        cutoff = time.time() - max(0.0, older_than_days) * 86400.0
        removed = {"runs": 0, "steps": 0}
        try:
            for table, key in (("skill_bench_runs", "runs"),
                               ("skill_bench_steps", "steps")):
                cur = self.db.execute(
                    f"DELETE FROM {table} WHERE created_at < ?", (cutoff,))
                removed[key] = cur.rowcount or 0
        except Exception as exc:  # noqa: BLE001
            _log.debug("skill bench prune failed: %s", exc)
        return removed

    # ── read ────────────────────────────────────────────────────────────
    def score(self, skill_name: str, *, version: str = "",
              limit: int = 500) -> dict[str, Any]:
        """Success-rate/latency summary over the recorded runs.

        ``version=""`` aggregates every version; pass one to scope it.
        Empty history yields a zeroed summary (``runs: 0``), never an
        exception.  Latency reports avg/min/max **and** p50/p95/p99 —
        tails matter more than averages when judging a version.
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
                "p50_latency_ms": round(_percentile(lats, 0.50), 3),
                "p95_latency_ms": round(_percentile(lats, 0.95), 3),
                "p99_latency_ms": round(_percentile(lats, 0.99), 3),
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

    def step_summary(self, skill_name: str, version: str = "",
                     limit: int = 500) -> list[dict[str, Any]]:
        """Per-step latency/success summary — which tool in the chain is
        slow or flaky.  Empty list when no step data was recorded."""
        params: list[Any] = [skill_name]
        sql = ("SELECT step_index, tool, success, latency_ms FROM "
               "skill_bench_steps WHERE skill_name=?")
        if version:
            sql += " AND version=?"
            params.append(version)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.db.query(sql, tuple(params))
        buckets: dict[tuple[int, str], dict[str, Any]] = {}
        for row in rows:
            key = (int(row["step_index"]), row["tool"] or "")
            bucket = buckets.setdefault(key, {"runs": 0, "successes": 0,
                                              "latencies": []})
            bucket["runs"] += 1
            bucket["successes"] += int(row["success"])
            bucket["latencies"].append(float(row["latency_ms"]))
        out = []
        for (index, tool), bucket in sorted(buckets.items()):
            lats = sorted(bucket["latencies"])
            out.append({
                "step_index": index,
                "tool": tool,
                "runs": bucket["runs"],
                "success_rate": round(bucket["successes"] / bucket["runs"],
                                      4),
                "avg_latency_ms": round(sum(lats) / len(lats), 3),
                "p95_latency_ms": round(_percentile(lats, 0.95), 3),
            })
        return out

    def compare_versions(self, skill_name: str, v_a: str, v_b: str,
                         limit: int = 500) -> dict[str, Any]:
        """Side-by-side scores for two versions, with deltas.

        ``success_rate_delta`` is b − a; ``p95_latency_delta_ms`` is
        b − a (negative means b is faster).
        """
        a = self.score(skill_name, version=v_a, limit=limit)
        b = self.score(skill_name, version=v_b, limit=limit)
        return {
            "skill": skill_name,
            "a": {"version": v_a, **{k: a[k] for k in (
                "runs", "success_rate", "avg_latency_ms",
                "p50_latency_ms", "p95_latency_ms", "p99_latency_ms")}},
            "b": {"version": v_b, **{k: b[k] for k in (
                "runs", "success_rate", "avg_latency_ms",
                "p50_latency_ms", "p95_latency_ms", "p99_latency_ms")}},
            "success_rate_delta": round(b["success_rate"]
                                        - a["success_rate"], 4),
            "p95_latency_delta_ms": round(b["p95_latency_ms"]
                                          - a["p95_latency_ms"], 3),
        }

    def regression(self, skill_name: str, candidate: str, baseline: str,
                   *, max_success_drop: float = 0.05,
                   max_p95_increase_ms: float | None = None,
                   limit: int = 500) -> dict[str, Any]:
        """Did ``candidate`` regress vs ``baseline``?

        Returns the comparison plus a boolean ``regressed`` verdict and
        the reasons.  The evidence behind a pin/rollback decision.
        """
        cmp = self.compare_versions(skill_name, baseline, candidate,
                                    limit=limit)
        reasons: list[str] = []
        drop = -cmp["success_rate_delta"]  # positive when candidate is worse
        if drop > max_success_drop:
            reasons.append(
                f"success rate dropped {drop * 100:.1f}pp "
                f"({cmp['a']['success_rate'] * 100:.1f}% → "
                f"{cmp['b']['success_rate'] * 100:.1f}%)")
        if max_p95_increase_ms is not None:
            inc = cmp["p95_latency_delta_ms"]
            if inc > max_p95_increase_ms:
                reasons.append(
                    f"p95 latency rose {inc:.1f}ms "
                    f"({cmp['a']['p95_latency_ms']:.1f} → "
                    f"{cmp['b']['p95_latency_ms']:.1f})")
        cmp.update({"baseline": baseline, "candidate": candidate,
                    "regressed": bool(reasons), "reasons": reasons})
        return cmp

    def canary_gate(self, skill_name: str, candidate_version: str, *,
                    min_runs: int = 20,
                    min_success_rate: float = 0.95,
                    max_p95_latency_ms: float | None = None,
                    limit: int = 500) -> dict[str, Any]:
        """Automated promotion gate for a candidate skill version.

        The Argo Rollouts AnalysisTemplate shape applied to skill pins:
        enough evidence (``min_runs``), success rate at or above the
        threshold, p95 latency within budget when given.  Returns
        ``{"pass": bool, "reasons": [...], "score": {...}}`` — reasons
        are empty on a pass.  A failing gate means "do not pin this
        version"; a passing one is evidence *for* promotion, never an
        automatic pin (pins stay explicit and human).
        """
        score = self.score(skill_name, version=candidate_version,
                           limit=limit)
        reasons: list[str] = []
        if score["runs"] < min_runs:
            reasons.append(
                f"only {score['runs']} run(s) recorded for "
                f"v{candidate_version} — need {min_runs} for a verdict")
        elif score["success_rate"] < min_success_rate:
            reasons.append(
                f"success rate {score['success_rate'] * 100:.1f}% below "
                f"threshold {min_success_rate * 100:.1f}%")
        if max_p95_latency_ms is not None and score["runs"] >= min_runs \
                and score["p95_latency_ms"] > max_p95_latency_ms:
            reasons.append(
                f"p95 latency {score['p95_latency_ms']:.1f}ms above budget "
                f"{max_p95_latency_ms:.1f}ms")
        return {
            "skill": skill_name,
            "candidate_version": candidate_version,
            "pass": not reasons,
            "reasons": reasons,
            "score": score,
        }

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
