"""Persistent benchmark measurements and scoring.

:class:`BenchmarkDB` records real measurements — latency, success, memory —
per (model, capability) and turns them into a 0..1 score the broker can rank
with.  Two hard rules:

1. No fabricated numbers.  Rows carry a ``source``; the only seeded rows are
   *real local measurements* taken by actually calling a provider
   (``source='synthetic'``), never invented latencies.
2. Scoring is transparent: 70% recent success rate + 30% latency rank against
   the model's own history, so a fast-but-flaky model loses to a slower
   reliable one.

Follows the existing SQLite pattern (:mod:`nomorals.os.session`): a ``Database``
plus a ``_DDL`` executescript, so the table can live in the same DB file as
everything else or stand alone in ``:memory:`` for tests.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger
from ..storage.db import Database
from .base import LLMResponse, Message
from .capabilities import Capability, capability_from

__all__ = [
    "BenchmarkDB",
    "BenchmarkSample",
    "benchmark_model",
    "seed_synthetic",
]

_log = get_logger(__name__)

_BENCH_DDL = """
CREATE TABLE IF NOT EXISTS model_benchmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    capability  TEXT NOT NULL DEFAULT 'chat',
    latency_s   REAL NOT NULL,
    success     INTEGER NOT NULL DEFAULT 1,
    memory_mb   REAL NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT 'live',
    task_kind   TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bench_model_cap
    ON model_benchmarks(model_id, capability);
CREATE INDEX IF NOT EXISTS idx_bench_created
    ON model_benchmarks(created_at);
"""

#: How many recent samples feed a score.  Old history should not anchor the
#: broker to a model that has since degraded (or been fixed).
SCORE_WINDOW = 20

#: How many samples a summary (and a broker prefetch) pulls per model.  The
#: broker's latency rank needs the full recent history, not just the score
#: window — one fetch serves both, so selection never re-queries per card.
SUMMARY_WINDOW = 1000


@dataclass
class BenchmarkSample:
    model_id: str
    capability: Capability | str
    latency_s: float
    success: bool = True
    memory_mb: float = 0.0
    source: str = "live"
    task_kind: str = ""

    @property
    def capability_value(self) -> str:
        return self.capability.value if isinstance(self.capability, Capability) else str(self.capability)


class BenchmarkDB:
    """Record and score real model measurements."""

    def __init__(self, db: Database | str | Path | None = None) -> None:
        if isinstance(db, Database):
            self.db = db
        else:
            self.db = Database(":memory:" if db is None else str(db))
        self.db.executescript(_BENCH_DDL)
        self._lock = threading.RLock()

    # ── recording ────────────────────────────────────────────────────────────
    def record(
        self,
        model_id: str,
        capability: Capability | str,
        latency_s: float,
        success: bool,
        memory_mb: float = 0.0,
        source: str = "live",
        task_kind: str = "",
    ) -> int:
        """Store one real measurement.  Returns the row id."""
        cap = capability.value if isinstance(capability, Capability) else str(capability)
        if latency_s < 0:
            raise ValueError("latency_s must be >= 0")
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO model_benchmarks "
                "(model_id, capability, latency_s, success, memory_mb, source, task_kind, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (model_id, cap, float(latency_s), 1 if success else 0,
                 float(memory_mb), source, task_kind, time.time()),
            )
            return int(cur.lastrowid)

    def record_sample(self, sample: BenchmarkSample) -> int:
        return self.record(
            sample.model_id, sample.capability, sample.latency_s, sample.success,
            sample.memory_mb, sample.source, sample.task_kind,
        )

    # ── scoring ──────────────────────────────────────────────────────────────
    def samples(
        self, model_id: str, capability: Capability | str = "", limit: int = SCORE_WINDOW
    ) -> list[dict[str, Any]]:
        cap = capability.value if isinstance(capability, Capability) else str(capability)
        sql = ("SELECT latency_s, success, memory_mb, source, created_at "
               "FROM model_benchmarks WHERE model_id = ?")
        params: list[Any] = [model_id]
        if cap:
            sql += " AND capability = ?"
            params.append(cap)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return self.db.query(sql, tuple(params))

    def samples_many(
        self,
        model_ids: Sequence[str],
        capability: Capability | str = "",
        *,
        limit: int = SUMMARY_WINDOW,
    ) -> dict[str, list[dict[str, Any]]]:
        """One query for many models: ``model_id -> rows`` (newest first,
        capped at ``limit`` rows each).

        Hot path for :meth:`ModelBroker.select`, which used to issue three
        identical ``samples()`` queries per candidate (summary, summary's
        score, selection score).  Rows come back in the same
        ``ORDER BY created_at DESC`` order as :meth:`samples`, so slicing
        ``rows[:SCORE_WINDOW]`` reproduces :meth:`score` exactly.
        """
        ids = list(dict.fromkeys(model_ids))
        if not ids:
            return {}
        cap = capability.value if isinstance(capability, Capability) else str(capability)
        placeholders = ", ".join("?" for _ in ids)
        sql = ("SELECT model_id, latency_s, success, memory_mb, source, created_at "
               f"FROM model_benchmarks WHERE model_id IN ({placeholders})")
        params: list[Any] = list(ids)
        if cap:
            sql += " AND capability = ?"
            params.append(cap)
        sql += " ORDER BY created_at DESC"
        with self._lock:
            rows = self.db.query(sql, tuple(params))
        out: dict[str, list[dict[str, Any]]] = {mid: [] for mid in ids}
        for row in rows:
            bucket = out.get(row["model_id"])
            if bucket is not None and len(bucket) < limit:
                bucket.append(row)
        return out

    @staticmethod
    def score_rows(rows: Sequence[dict[str, Any]]) -> float:
        """0..1 score from pre-fetched samples (newest first).

        Pure function of the rows: :meth:`score` is ``score_rows`` over
        :meth:`samples` output, so a single prefetch can serve score and
        summary without re-querying.  ``rows`` longer than ``SCORE_WINDOW``
        must be sliced by the caller — mirrors ``samples(limit=SCORE_WINDOW)``.
        """
        if not rows:
            return 0.5
        successes = sum(1 for r in rows if r["success"])
        success_rate = successes / len(rows)
        latencies = sorted(r["latency_s"] for r in rows if r["success"])
        if not latencies:
            return round(0.7 * success_rate, 4)
        median = statistics.median(latencies)
        p90 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.9))]
        # latency quality: 1.0 when median == best observed, decaying as the
        # tail (p90) stretches away from the median.
        latency_quality = median / p90 if p90 > 0 else 1.0
        return round(0.7 * success_rate + 0.3 * latency_quality, 4)

    def score(self, model_id: str, capability: Capability | str = "") -> float:
        """0..1 score: 70% recent success rate + 30% latency quality.

        Latency quality compares the model's median latency against its own
        p90 — a model that is consistently fast scores near 1, one whose
        latency spikes scores lower.  No rows → 0.5 (neutral, not punished).
        """
        return self.score_rows(self.samples(model_id, capability))

    @staticmethod
    def summary_rows(model_id: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Summary dict from pre-fetched samples (newest first).

        The ``"score"`` entry uses ``rows[:SCORE_WINDOW]`` — identical to the
        separate query :meth:`score` used to issue, because both order by
        ``created_at DESC``.
        """
        latencies = [r["latency_s"] for r in rows if r["success"]]
        return {
            "model_id": model_id,
            "samples": len(rows),
            "success_rate": round(sum(1 for r in rows if r["success"]) / len(rows), 4) if rows else 0.0,
            "median_latency_s": round(statistics.median(latencies), 4) if latencies else 0.0,
            "score": BenchmarkDB.score_rows(rows[:SCORE_WINDOW]),
            "sources": sorted({r["source"] for r in rows}),
        }

    def summary(self, model_id: str, capability: Capability | str = "") -> dict[str, Any]:
        rows = self.samples(model_id, capability, limit=SUMMARY_WINDOW)
        return self.summary_rows(model_id, rows)

    def models(self) -> list[str]:
        with self._lock:
            return [r["model_id"] for r in
                    self.db.query("SELECT DISTINCT model_id FROM model_benchmarks")]

    def cost_per_task(
        self,
        model_id: str,
        capability: Capability | str = "",
        *,
        cost_per_1k: float = 0.0,
    ) -> dict[str, Any]:
        """USD per *successful* task for a model.

        Failed runs cost ~2x a finished one and must not poison the figure:
        cost is summed over successful runs only, and the failure rate is
        reported as its own number.  ``cost_per_1k`` converts token counts
        when the rows carry none (live rows from learning.py have no token
        counts yet — pass the card's price).
        """
        rows = self.samples(model_id, capability, limit=SUMMARY_WINDOW)
        ok_rows = [r for r in rows if r["success"]]
        failures = len(rows) - len(ok_rows)
        # Rows carry latency, not tokens: estimate cost from a per-call
        # price when the caller knows it.
        total_cost = len(ok_rows) * cost_per_1k / 1000.0
        return {
            "model_id": model_id,
            "successful_tasks": len(ok_rows),
            "failed_tasks": failures,
            "failure_rate": round(failures / len(rows), 4) if rows else 0.0,
            "cost_per_task_usd": round(total_cost / len(ok_rows), 9) if ok_rows else 0.0,
            "median_latency_s": round(
                statistics.median([r["latency_s"] for r in ok_rows]), 4
            ) if ok_rows else 0.0,
        }

    def leaderboard(
        self,
        capability: Capability | str = "",
        *,
        limit: int = 10,
        cost_per_1k_by_model: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank models: success rate → benchmark score → cost per task.

        Mirrors the Open LLM Leaderboard idea locally: every row is a real
        measurement, nothing is invented.  ``cost_per_1k_by_model`` maps
        model id → USD/1k tokens for the cost-per-task column.
        """
        prices = cost_per_1k_by_model or {}
        board: list[dict[str, Any]] = []
        for model_id in self.models():
            summary = self.summary(model_id, capability)
            cpt = self.cost_per_task(
                model_id, capability,
                cost_per_1k=float(prices.get(model_id, 0.0)),
            )
            board.append({**summary, **cpt})
        board.sort(key=lambda r: (
            -r["success_rate"], -r["score"], r["cost_per_task_usd"],
            r["model_id"]))
        return board[:max(1, limit)]

    def prune(self, older_than_days: float = 90.0) -> int:
        """Drop stale measurements.  Returns rows deleted."""
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            cur = self.db.execute(
                "DELETE FROM model_benchmarks WHERE created_at < ?", (cutoff,))
            return cur.rowcount


def benchmark_model(
    model_id: str,
    capability: Capability | str,
    provider: Any,
    prompts: Sequence[str] | None = None,
    *,
    task_kind: str = "",
    source: str = "synthetic",
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Run real round-trips through ``provider`` and return measured stats.

    This is the *only* sanctioned way to seed :class:`BenchmarkDB` without live
    traffic: every row comes from an actual provider call, with wall-clock
    latency.  Nothing is invented — a slow provider records slow numbers.
    Returns a summary dict (does NOT write to any DB; the caller records).
    """
    cap = capability_from(capability) if isinstance(capability, str) else capability
    prompts = list(prompts or ["Say OK."])
    rounds: list[dict[str, Any]] = []
    for prompt in prompts:
        started = time.monotonic()
        try:
            if cap in (Capability.CHAT, Capability.CODE, Capability.JUDGE):
                response: LLMResponse = provider.chat([Message.user(prompt)])
            elif cap == Capability.EMBED:
                provider.embed([prompt])
                response = LLMResponse(text="ok")
            elif cap == Capability.VISION:
                response = provider.describe_image(b"", prompt)
            else:  # OCR / SPEECH: fall back to a chat-shaped probe
                response = provider.chat([Message.user(prompt)])
            ok = bool(response.ok)
        except Exception as exc:  # noqa: BLE001 — a failed probe is data
            _log.warning("benchmark probe failed for %s: %s", model_id, exc)
            ok = False
        elapsed = time.monotonic() - started
        rounds.append({"latency_s": round(elapsed, 4), "success": ok})
        if elapsed > timeout_s:
            break
    successes = sum(1 for r in rounds if r["success"])
    latencies = [r["latency_s"] for r in rounds]
    return {
        "model_id": model_id,
        "capability": cap.value,
        "rounds": rounds,
        "successes": successes,
        "success_rate": successes / len(rounds) if rounds else 0.0,
        "median_latency_s": round(statistics.median(latencies), 4) if latencies else 0.0,
        "latencies_s": [round(v, 4) for v in latencies],
        "task_kind": task_kind,
        "source": source,
    }


def seed_synthetic(
    db: BenchmarkDB,
    model_id: str,
    capability: Capability | str,
    provider: Any,
    prompts: Sequence[str] | None = None,
    *,
    task_kind: str = "",
) -> dict[str, Any]:
    """Measure ``provider`` for real and store the rows as ``source='synthetic'``.

    Every stored row is a genuine local measurement — wall-clock latency of an
    actual call — so the broker's rankings rest on evidence, not fiction.
    """
    result = benchmark_model(
        model_id, capability, provider, prompts,
        task_kind=task_kind, source="synthetic",
    )
    cap = result["capability"]
    for round_ in result["rounds"]:
        db.record(
            model_id, cap, round_["latency_s"],
            success=round_["success"],
            source="synthetic", task_kind=task_kind,
        )
    _log.info("seeded %d synthetic benchmark rows for %s/%s",
              len(result["rounds"]), model_id, cap)
    return result
