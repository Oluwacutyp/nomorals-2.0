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
            resolved_ts REAL NOT NULL DEFAULT 0,
            priority REAL NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Migrations for DBs created before priority/attempts existed.
    for ddl in (
        "ALTER TABLE knowledge_gaps ADD COLUMN priority REAL NOT NULL DEFAULT 0",
        "ALTER TABLE knowledge_gaps ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            db.execute(ddl)
        except Exception:  # noqa: BLE001 - column already there
            pass
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

    @staticmethod
    def _gap_priority(question: str, context: str = "") -> float:
        """Priority score for a knowledge gap (qwen_ai_scientist pattern).

        Gaps with more substance, time sensitivity, and context get worked
        first. Heuristic and documented: content-word count, time-anchor
        bonus, context bonus. Capped at 1.0.
        """
        import re as _re
        stop = frozenset(
            "a an the and or but if then else when at by for with about into "
            "of as it its this that these those is are was were be been have "
            "has had do does did what why how which who whom whose can could "
            "should would will shall may might must tell me my i you your we "
            "us s t".split())
        words = [w for w in _re.findall(r"[a-z0-9]+", question.lower())
                 if w not in stop and len(w) > 1]
        score = min(0.5, 0.05 * len(words))
        time_anchors = frozenset(
            "today yesterday week month year recent latest current now "
            "upcoming soon deadline new just 2024 2025 2026 2027 2028".split())
        if set(words) & time_anchors:
            score += 0.3
        if (context or "").strip():
            score += 0.2
        return round(min(1.0, score), 3)

    @staticmethod
    def _gap_overlap(a: str, b: str) -> float:
        import re as _re
        wa = set(_re.findall(r"[a-z0-9]{3,}", a.lower()))
        wb = set(_re.findall(r"[a-z0-9]{3,}", b.lower()))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / max(len(wa), len(wb))

    def note_gap(self, question: str, context: str = "") -> int:
        question = (question or "").strip()
        if not question:
            raise ValueError("gap needs a question")
        # De-dupe: exact match first, then fuzzy (same question asked in
        # different words must not open a second gap).
        row = self.db.query_one(
            "SELECT id FROM knowledge_gaps WHERE question = ? AND status = 'open'",
            (question,))
        if row:
            return int(row["id"])
        for existing in self.open_gaps(limit=50):
            if self._gap_overlap(question, existing["question"]) >= 0.6:
                return int(existing["id"])
        cur = self.db.execute(
            "INSERT INTO knowledge_gaps (question, context, ts, status, priority)"
            " VALUES (?, ?, ?, 'open', ?)",
            (question, context or "", time.time(),
             self._gap_priority(question, context)))
        try:
            return int(cur.lastrowid or 0)
        except Exception:  # noqa: BLE001
            return 0

    def open_gaps(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, question, context, ts, priority, attempts"
            " FROM knowledge_gaps"
            " WHERE status = 'open' ORDER BY priority DESC, ts ASC LIMIT ?",
            (limit,))
        return [dict(r) for r in (rows or [])]

    def gap_stats(self) -> dict[str, Any]:
        """Observability for the gap loop: counts by status + top gaps."""
        out: dict[str, Any] = {"open": 0, "resolved": 0, "stalled": 0,
                               "top_open": []}
        try:
            rows = self.db.query(
                "SELECT status, COUNT(*) AS n FROM knowledge_gaps GROUP BY status")
            for r in rows or []:
                if r["status"] in out:
                    out[r["status"]] = int(r["n"])
            out["top_open"] = [
                {"id": g["id"], "question": g["question"][:100],
                 "priority": g.get("priority", 0),
                 "attempts": g.get("attempts", 0)}
                for g in self.open_gaps(limit=5)]
        except Exception:  # noqa: BLE001 - stats never break the organ
            pass
        return out

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

    #: a gap that fails this many research attempts stops being retried.
    _MAX_GAP_ATTEMPTS = 3

    def _work_gaps(self, report: OrganTickReport) -> None:
        from .pipeline import research_deep, _router_llm_fn
        gaps = self.open_gaps(limit=3)  # conservative: 3 per tick max
        if not gaps:
            return
        rctx = self._rctx()
        llm_fn = _router_llm_fn(getattr(self.context, "router", None))
        for gap in gaps:
            attempts = int(gap.get("attempts", 0) or 0) + 1
            try:
                deep = research_deep(
                    gap["question"], rctx, llm_fn=llm_fn, max_queries=4)
                answered = bool((deep.synthesis or "").strip())
                status = ("resolved" if answered
                          else "stalled" if attempts >= self._MAX_GAP_ATTEMPTS
                          else "open")
                self.db.execute(
                    "UPDATE knowledge_gaps SET status = ?, resolved_ts = ?,"
                    " attempts = ? WHERE id = ?",
                    (status, time.time() if answered else 0,
                     attempts, gap["id"]))
                if answered:
                    report.gaps_resolved += 1
                    # INTEGRATE (MAIL loop): the answer is stored back as
                    # learned knowledge, not just emitted — the gap
                    # genuinely closes instead of rotting in a table.
                    self._integrate_gap_answer(gap, deep)
                    # The answer also goes back to the brain as an event so
                    # it can act on it immediately.
                    _bus.emit(self.db, "research", "brain", "gap.answered", {
                        "question": gap["question"],
                        "synthesis": deep.synthesis[:2000],
                        "citations": [
                            {"title": f.title, "url": f.url}
                            for f in (deep.findings or [])[:5]],
                    })
                    report.events_emitted += 1
            except Exception as exc:  # noqa: BLE001
                self.db.execute(
                    "UPDATE knowledge_gaps SET attempts = ?,"
                    " status = CASE WHEN ? >= ? THEN 'stalled' ELSE status END"
                    " WHERE id = ?",
                    (attempts, attempts, self._MAX_GAP_ATTEMPTS, gap["id"]))
                _log.warning("gap research failed for %s: %s",
                             gap["question"][:60], exc)

    def _integrate_gap_answer(self, gap: dict[str, Any], deep: Any) -> None:
        """Store a resolved gap's answer in memory as a learned fact."""
        memory = getattr(self.context, "memory", None)
        if memory is None:
            return
        remember = getattr(memory, "remember", None)
        if not callable(remember):
            return
        try:
            from ..memory.base import MemoryKind
            kind = MemoryKind.FACT
        except Exception:  # noqa: BLE001
            kind = "fact"
        try:
            remember(
                f"Research answer — {gap['question']}\n"
                f"{(deep.synthesis or '')[:1500]}",
                kind=kind, importance=0.7, source="research-organ",
                tags="research,gap",
                metadata={"gap_id": gap["id"],
                          "learnings": list(
                              getattr(deep, "learnings", []) or [])[:5]})
        except Exception as exc:  # noqa: BLE001 - integrate is best-effort
            _log.debug("gap memory integrate failed (%s)", exc)
