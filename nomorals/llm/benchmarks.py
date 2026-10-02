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

    def score(self, model_id: str, capability: Capability | str = "") -> float:
        """0..1 score: 70% recent success rate + 30% latency quality.

        Latency quality compares the model's median latency against its own
        p90 — a model that is consistently fast scores near 1, one whose
        latency spikes scores lower.  No rows → 0.5 (neutral, not punished).
        """
        rows = self.samples(model_id, capability)
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

    def summary(self, model_id: str, capability: Capability | str = "") -> dict[str, Any]:
        rows = self.samples(model_id, capability, limit=1000)
        latencies = [r["latency_s"] for r in rows if r["success"]]
        return {
            "model_id": model_id,
            "samples": len(rows),
            "success_rate": round(sum(1 for r in rows if r["success"]) / len(rows), 4) if rows else 0.0,
            "median_latency_s": round(statistics.median(latencies), 4) if latencies else 0.0,
            "score": self.score(model_id, capability),
            "sources": sorted({r["source"] for r in rows}),
        }

    def models(self) -> list[str]:
        with self._lock:
            return [r["model_id"] for r in
                    self.db.query("SELECT DISTINCT model_id FROM model_benchmarks")]

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
        "task_kind": task_kind,
        "source": source,
    }
    return {
        "model_id": model_id,
        "capability": cap.value,
        "rounds": len(latencies),
        "successes": successes,
        "success_rate": successes / len(latencies) if latencies else 0.0,
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
