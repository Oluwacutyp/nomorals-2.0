"""Research organ — autonomous knowledge acquisition.

The ResearchScheduler already knew how to run jobs on a tick loop, but it
was never started anywhere and its jobs lived only in memory. This organ
makes research genuinely autonomous:

- **Persistent watches** (``research_watches`` table): the brain (or the
  owner) schedules a watch once via the ``research_schedule`` spine tool;
  the organ runs it on cadence across restarts. No commands needed.
- **Knowledge-gap loop**: when the brain hits something it can't answer,
  it logs a gap via ``note_knowledge_gap``; the organ turns open gaps
  into one-shot research runs and marks them resolved.
- **Source quality scoring**: every finding's worth assessment feeds a
  rolling per-domain quality score (``source_quality`` table). Domains
  that consistently produce noise get deprioritized — learned from
  evidence, never a hardcoded allowlist.
- **Cross-organ events**: esoteric/religious/philosophical findings are
  forwarded to the wisdom organ for corpus ingestion; high-worth
  findings matching the owner's goals are delivered as briefings
  through the existing ``pipeline.deliver`` path.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ..core.logging_setup import get_logger
from .. import organs as _bus
from .pipeline import ResearchContext, ResearchJob

_log = get_logger(__name__)

# Traditions/topics that belong in the wisdom corpus rather than a briefing.
_WISDOM_HINTS = (
    "apocrypha", "gnostic", "hermetic", "sufi", "vedanta", "upanishad",
    "tao", "buddhist", "kabbalah", "esoteric", "metaphysical", "scripture",
    "sacred text", "ancient wisdom", "mysticism", "kundalini",
    "astral", "akashic",
)


def ensure_schema(db: Any) -> None:
    from .pipeline import ensure_schema as _pens
    _pens(db)
    _bus.ensure_schema(db)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS research_watches (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            queries TEXT NOT NULL DEFAULT '[]',
            cadence_hours REAL NOT NULL DEFAULT 24.0,
            goals TEXT NOT NULL DEFAULT '[]',
            max_results INTEGER NOT NULL DEFAULT 6,
            fetch_top INTEGER NOT NULL DEFAULT 2,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            created_by TEXT NOT NULL DEFAULT 'system'
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '',
            ts REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            resolved_ts REAL NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS source_quality (
            domain TEXT PRIMARY KEY,
            runs INTEGER NOT NULL DEFAULT 0,
            worth_runs INTEGER NOT NULL DEFAULT 0,
            last_ts REAL NOT NULL DEFAULT 0
        )
        """
    )


@dataclass
class OrganTickReport:
    watches_run: int = 0
    findings: int = 0
    delivered: int = 0
    gaps_opened: int = 0
    gaps_resolved: int = 0
    events_emitted: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class ResearchOrgan:
    """The autonomous research loop. One ``tick()`` = one full cycle."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        if self.db is None:
            raise ValueError("research organ needs context.db")
        ensure_schema(self.db)

    # ── watches (persistent jobs) ────────────────────────────────────

    def add_watch(self, topic: str, queries: list[str],
                  cadence_hours: float = 24.0,
                  goals: list[str] | None = None,
                  created_by: str = "system") -> str:
        topic = (topic or "").strip()
        queries = [q.strip() for q in (queries or []) if q.strip()]
        if not topic or not queries:
            raise ValueError("watch needs a topic and at least one query")
        import hashlib
        wid = "w_" + hashlib.sha256(
            (topic + "|" + "|".join(queries)).encode()).hexdigest()[:12]
        self.db.execute(
            "INSERT INTO research_watches"
            " (id, topic, queries, cadence_hours, goals, created_at, created_by)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            " topic=excluded.topic, queries=excluded.queries,"
            " cadence_hours=excluded.cadence_hours, goals=excluded.goals,"
            " enabled=1",
            (wid, topic, json.dumps(queries), float(cadence_hours),
             json.dumps(goals or []), time.time(), created_by),
        )
        _log.info("research watch added: %s (%s)", wid, topic)
        return wid

    def remove_watch(self, watch_id: str) -> bool:
        cur = self.db.execute(
            "UPDATE research_watches SET enabled = 0 WHERE id = ?", (watch_id,))
        return (cur.rowcount or 0) > 0

    def list_watches(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, topic, queries, cadence_hours, goals, enabled,"
            " created_by FROM research_watches ORDER BY created_at DESC")
        out = []
        for r in rows or []:
            out.append({
                "id": r["id"], "topic": r["topic"],
                "queries": json.loads(r["queries"] or "[]"),
                "cadence_hours": r["cadence_hours"],
                "goals": json.loads(r["goals"] or "[]"),
                "enabled": bool(r["enabled"]),
                "created_by": r["created_by"],
                "last_run": self._last_run(r["id"]),
            })
        return out

    def _last_run(self, watch_id: str) -> float:
        row = self.db.query_one(
            "SELECT last_run FROM research_job_state WHERE job_id = ?",
            (watch_id,))
        return float(row["last_run"]) if row else 0.0

    # ── knowledge gaps ───────────────────────────────────────────────

    def note_gap(self, question: str, context: str = "") -> int:
        question = (question or "").strip()
        if not question:
            raise ValueError("gap needs a question")
        # De-dupe: don't re-open an identical open gap.
        row = self.db.query_one(
            "SELECT id FROM knowledge_gaps WHERE question = ? AND status = 'open'",
            (question,))
        if row:
            return int(row["id"])
        cur = self.db.execute(
            "INSERT INTO knowledge_gaps (question, context, ts, status)"
            " VALUES (?, ?, ?, 'open')",
            (question, context or "", time.time()))
        try:
            return int(cur.lastrowid or 0)
        except Exception:  # noqa: BLE001
            return 0

    def open_gaps(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, question, context, ts FROM knowledge_gaps"
            " WHERE status = 'open' ORDER BY ts ASC LIMIT ?", (limit,))
        return [dict(r) for r in (rows or [])]

    # ── source quality (learned, not hardcoded) ──────────────────────

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower().lstrip("www.")
        except Exception:  # noqa: BLE001
            return ""

    def record_source_quality(self, url: str, worth: bool) -> None:
        domain = self._domain(url)
        if not domain:
            return
        self.db.execute(
            "INSERT INTO source_quality (domain, runs, worth_runs, last_ts)"
            " VALUES (?, 1, ?, ?)"
            " ON CONFLICT(domain) DO UPDATE SET"
            " runs = runs + 1,"
            " worth_runs = worth_runs + excluded.worth_runs,"
            " last_ts = excluded.last_ts",
            (domain, 1 if worth else 0, time.time()))

    def domain_score(self, domain: str) -> float | None:
        """Rolling worth-rate for a domain, or None when unseen."""
        row = self.db.query_one(
            "SELECT runs, worth_runs FROM source_quality WHERE domain = ?",
            (domain,))
        if not row or not row["runs"]:
            return None
        return float(row["worth_runs"]) / float(row["runs"])

    def weak_sources(self, min_runs: int = 3,
                     threshold: float = 0.25) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT domain, runs, worth_runs FROM source_quality"
            " WHERE runs >= ? ORDER BY runs DESC", (min_runs,))
        out = []
        for r in rows or []:
            score = float(r["worth_runs"]) / float(r["runs"])
            if score < threshold:
                out.append({"domain": r["domain"], "runs": r["runs"],
                            "worth_rate": round(score, 3)})
        return out

    # ── the autonomous tick ──────────────────────────────────────────

    def _rctx(self) -> ResearchContext:
        return ResearchContext(
            db=self.db,
            registry=getattr(self.context, "tools", None),
            memory=getattr(self.context, "memory", None),
            gateway=getattr(self.context, "gateway", None),
        )

    def _job_for_watch(self, watch: dict[str, Any]) -> ResearchJob:
        return ResearchJob(
            id=watch["id"], topic=watch["topic"], queries=watch["queries"],
            cadence_hours=float(watch["cadence_hours"]),
            goals=watch["goals"])

    def tick(self) -> OrganTickReport:
        """One autonomous cycle. Safe to call on a schedule; idempotent."""
        report = OrganTickReport()
        now = time.time()
        try:
            self._drain_directives(report)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"directives: {exc}")
        try:
            self._run_due_watches(report, now)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"watches: {exc}")
        try:
            self._work_gaps(report)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"gaps: {exc}")
        return report

    def _drain_directives(self, report: OrganTickReport) -> None:
        for event in _bus.drain(self.db, "research"):
            kind = event["kind"]
            payload = event["payload"] or {}
            if kind == "directive.watch":
                try:
                    self.add_watch(
                        payload.get("topic", ""), payload.get("queries", []),
                        float(payload.get("cadence_hours", 24.0)),
                        payload.get("goals", []),
                        created_by=str(event.get("src", "brain")))
                except Exception as exc:  # noqa: BLE001
                    _log.warning("bad research directive: %s", exc)
            elif kind == "directive.gap":
                try:
                    self.note_gap(payload.get("question", ""),
                                  payload.get("context", ""))
                    report.gaps_opened += 1
                except Exception as exc:  # noqa: BLE001
                    _log.warning("bad gap directive: %s", exc)

    def _run_due_watches(self, report: OrganTickReport, now: float) -> None:
        from .pipeline import run_job, assess_worth, deliver
        rctx = self._rctx()
        for watch in self.list_watches():
            if not watch["enabled"]:
                continue
            if now - self._last_run(watch["id"]) < \
                    float(watch["cadence_hours"]) * 3600.0:
                continue
            job = self._job_for_watch(watch)
            errors: list[str] = []
            delivered = skipped = 0
            try:
                findings = run_job(job, rctx)
                report.watches_run += 1
                report.findings += len(findings)
                for finding in findings:
                    try:
                        assessment = assess_worth(finding, rctx)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"assess: {exc}"[:120])
                        skipped += 1
                        continue
                    # Learned source quality: every assessment teaches the
                    # organ which domains are worth its time.
                    self.record_source_quality(finding.url, assessment.worth)
                    self._maybe_forward_to_wisdom(finding)
                    if not assessment.worth:
                        skipped += 1
                        continue
                    try:
                        deliver(finding, assessment, rctx)
                        delivered += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"deliver: {exc}"[:120])
                        skipped += 1
                report.delivered += delivered
                status = (f"ok d={delivered} s={skipped}"
                          if not errors else f"errors: {errors[0][:100]}")
            except Exception as exc:  # noqa: BLE001
                status = f"failed: {exc}"[:200]
                _log.warning("research watch %s failed: %s", watch["id"], exc)
            # Record the run even on failure — a broken watch must not hot-loop.
            self.db.execute(
                "INSERT INTO research_job_state (job_id, last_run, last_status)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(job_id) DO UPDATE SET"
                " last_run = excluded.last_run,"
                " last_status = excluded.last_status",
                (watch["id"], now, status))

    def _maybe_forward_to_wisdom(self, finding: Any) -> None:
        """Esoteric findings belong in the corpus, not just a briefing."""
        text = ((getattr(finding, "title", "") or "") + " " +
                (getattr(finding, "snippet", "") or "")).lower()
        if any(h in text for h in _WISDOM_HINTS):
            _bus.emit(self.db, "research", "wisdom", "finding.esoteric", {
                "title": getattr(finding, "title", ""),
                "url": getattr(finding, "url", ""),
                "snippet": getattr(finding, "snippet", ""),
            })

    def _work_gaps(self, report: OrganTickReport) -> None:
        from .pipeline import research_deep, _router_llm_fn
        gaps = self.open_gaps(limit=3)  # conservative: 3 per tick max
        if not gaps:
            return
        rctx = self._rctx()
        llm_fn = _router_llm_fn(getattr(self.context, "router", None))
        for gap in gaps:
            try:
                deep = research_deep(
                    gap["question"], rctx, llm_fn=llm_fn, max_queries=4)
                answered = bool((deep.synthesis or "").strip())
                self.db.execute(
                    "UPDATE knowledge_gaps SET status = ?, resolved_ts = ?"
                    " WHERE id = ?",
                    ("resolved" if answered else "stalled",
                     time.time(), gap["id"]))
                if answered:
                    report.gaps_resolved += 1
                    # The answer goes back to the brain as an event so it
                    # can actually use it instead of the gap rotting in a table.
                    _bus.emit(self.db, "research", "brain", "gap.answered", {
                        "question": gap["question"],
                        "synthesis": deep.synthesis[:2000],
                        "citations": [
                            {"title": f.title, "url": f.url}
                            for f in (deep.findings or [])[:5]],
                    })
                    report.events_emitted += 1
            except Exception as exc:  # noqa: BLE001
                _log.warning("gap research failed for %s: %s",
                             gap["question"][:60], exc)
