"""Async research jobs: deep research that doesn't block chat.

``web_research(deep=True)`` used to hold the tool call (and the chat
turn) for up to three minutes. The job model instead:

* :func:`start` — validates, registers the job, returns a job id
  *immediately*. A daemon worker thread runs the research.
* :func:`status` — one cheap row read: queued / running / done / failed,
  elapsed seconds, and a human-readable progress note.
* :func:`result` — the full report once done.
* On completion the worker delivers through :mod:`nomorals.agents.notifier`
  (durable first, then every live channel) — the owner gets the answer
  when it's ready instead of watching a spinner.

Durability: jobs live in the ``research_jobs`` table (see
``_apply_search_engine_schema`` in ``nomorals.storage.migrations``) with
an in-memory fallback when the context has no database (tests, CLI
snippets).  The worker never raises: every failure is recorded on the
job row *and* delivered as a failure notice, never swallowed.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...core.ids import new_short_id
from ...core.logging_setup import get_logger

__all__ = ["start", "status", "result", "cancel", "list_jobs", "STATES"]

_log = get_logger(__name__)

STATES = ("queued", "running", "done", "failed", "cancelled")

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research-job")
#: RLock so the check-and-set helpers can nest _load/_store safely.
_LOCK = threading.RLock()

#: In-memory fallback when the context carries no database. Keys are job
#: ids; values mirror the table's columns.
_FALLBACK: dict[str, dict[str, Any]] = {}


# ── storage ──────────────────────────────────────────────────────────────────


def _db(context: Any) -> Any | None:
    return getattr(context, "db", None)


def _now() -> float:
    return time.time()


def _store(context: Any, job: dict[str, Any]) -> None:
    db = _db(context)
    if db is None:
        with _LOCK:
            _FALLBACK[job["id"]] = dict(job)
        return
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO research_jobs "
                "(id, query, mode, params, state, progress, result, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "state=excluded.state, progress=excluded.progress, result=excluded.result, "
                "error=excluded.error, updated_at=excluded.updated_at",
                (
                    job["id"], job["query"], job["mode"],
                    json.dumps(job.get("params") or {}),
                    job["state"], job.get("progress", ""),
                    json.dumps(job.get("report") or {}),
                    job.get("error", ""),
                    job.get("created_at", _now()), _now(),
                ),
            )
    except Exception as exc:  # noqa: BLE001 - db write is best-effort; keep the row in memory too
        _log.debug("research job persist failed (%s); using memory fallback", exc)
        with _LOCK:
            _FALLBACK[job["id"]] = dict(job)


def _load(context: Any, job_id: str) -> dict[str, Any] | None:
    db = _db(context)
    if db is not None:
        try:
            row = db.query_one("SELECT * FROM research_jobs WHERE id = ?", (job_id,))
        except Exception:  # noqa: BLE001
            row = None
        if row:
            job = {
                "id": row["id"],
                "query": row.get("query", ""),
                "mode": row.get("mode", "quick"),
                "params": json.loads(row.get("params") or "{}"),
                "state": row.get("state", "queued"),
                "progress": row.get("progress", ""),
                "report": json.loads(row.get("result") or "{}"),
                "error": row.get("error", ""),
                "created_at": float(row.get("created_at") or 0),
                "updated_at": float(row.get("updated_at") or 0),
            }
            with _LOCK:
                _FALLBACK[job["id"]] = dict(job)
            return job
    with _LOCK:
        job = _FALLBACK.get(job_id)
        return dict(job) if job else None


def _list(context: Any, limit: int = 20) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    db = _db(context)
    if db is not None:
        try:
            rows = db.query(
                "SELECT * FROM research_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            for row in rows or []:
                job = _load(context, row["id"])
                if job:
                    out[job["id"]] = job
        except Exception:  # noqa: BLE001
            pass
    with _LOCK:
        for job_id, job in _FALLBACK.items():
            out.setdefault(job_id, dict(job))
    jobs = sorted(out.values(), key=lambda j: -float(j.get("created_at") or 0))
    return jobs[:limit]


# ── the API ──────────────────────────────────────────────────────────────────


def start(
    context: Any,
    query: str,
    *,
    mode: str = "quick",
    pages: int = 3,
    dig: bool = True,
    freshness: str = "",
    scope: str = "auto",
    crawl: bool = False,
) -> str:
    """Register a research job and return its id immediately.  The work
    runs on a daemon thread; use :func:`status` / :func:`result` to
    follow it.  Never raises — a bad call produces a failed job with a
    clear error instead."""
    query = (query or "").strip()
    mode = (mode or "quick").strip().lower()
    job = {
        "id": new_short_id("rj"),
        "query": query,
        "mode": mode if mode in {"quick", "deep"} else "quick",
        "params": {
            "pages": max(1, min(int(pages), 12)),
            "dig": bool(dig),
            "freshness": freshness or "",
            "scope": scope or "auto",
            "crawl": bool(crawl),
        },
        "state": "queued",
        "progress": "queued",
        "report": {},
        "error": "",
        "created_at": _now(),
        "updated_at": _now(),
    }
    if not query:
        job["state"] = "failed"
        job["error"] = "research needs a query"
        _store(context, job)
        return job["id"]
    if job["mode"] == "deep" and not _power_active(context):
        # fail fast with a clear message instead of a job that "runs"
        # and then dies in the worker
        job["state"] = "failed"
        job["error"] = "deep research is a power-mode capability — enable power mode first"
        _store(context, job)
        return job["id"]
    _store(context, job)
    _POOL.submit(_work, context, job["id"])
    return job["id"]


def _power_active(context: Any) -> bool:
    """Best-effort power-mode check.  Unknown (no settings, test stubs) →
    True: the worker performs the authoritative check anyway."""
    try:
        from ..power import power_mode_for

        return bool(power_mode_for(context).active)
    except Exception:  # noqa: BLE001 - the worker re-checks for real
        return True


def status(context: Any, job_id: str) -> dict[str, Any]:
    """One cheap read: state, progress note, elapsed seconds.  Never raises."""
    job = _load(context, (job_id or "").strip())
    if job is None:
        return {"id": job_id, "state": "unknown",
                "error": f"no research job {job_id!r}"}
    elapsed = round(_now() - float(job.get("created_at") or _now()), 1)
    return {
        "id": job["id"],
        "query": job.get("query", ""),
        "mode": job.get("mode", "quick"),
        "state": job.get("state", "queued"),
        "progress": job.get("progress", ""),
        "error": job.get("error", ""),
        "elapsed_seconds": elapsed,
        "report_id": (job.get("report") or {}).get("id", ""),
    }


def result(context: Any, job_id: str) -> dict[str, Any] | None:
    """The full report once the job is done, else ``None``.  Never raises."""
    job = _load(context, (job_id or "").strip())
    if job is None or job.get("state") != "done":
        return None
    report = job.get("report") or {}
    return dict(report) if report else None


def cancel(context: Any, job_id: str) -> bool:
    """Mark a queued/running job cancelled.  A worker already mid-fetch
    finishes its current step and then records the cancellation instead of
    delivering.  Never raises."""
    return _cas_state(context, (job_id or "").strip(), {"queued", "running"},
                      "cancelled", progress="cancelled by the owner") is not None


def list_jobs(context: Any, limit: int = 20) -> list[dict[str, Any]]:
    """Newest-first job envelopes (state, progress, timings — not the full
    reports).  Never raises."""
    return [
        {k: j.get(k) for k in ("id", "query", "mode", "state", "progress",
                               "error", "created_at", "updated_at")}
        for j in _list(context, limit)
    ]


# ── the worker ───────────────────────────────────────────────────────────────


def _set(context: Any, job_id: str, **fields: Any) -> None:
    job = _load(context, job_id) or {"id": job_id}
    job.update(fields)
    job["updated_at"] = _now()
    _store(context, job)


def _cas_state(context: Any, job_id: str, expected: set[str], new: str,
               **fields: Any) -> dict[str, Any] | None:
    """Atomic check-and-set on the job state: only transitions when the
    current state is in ``expected``.  This is what makes cancel-vs-worker
    race-free (a cancel landing mid-run wins; a finished job can't be
    un-cancelled)."""
    with _LOCK:
        job = _load(context, job_id)
        if job is None or job.get("state") not in expected:
            return None
        job["state"] = new
        job.update(fields)
        job["updated_at"] = _now()
        _store(context, job)
        return job


def _deliver(context: Any, job: dict[str, Any]) -> None:
    """Completion delivery through the Notifier (durable first, then live
    channels).  Best-effort — a dead notifier must not lose the row."""
    try:
        from ..notifier import notify

        query = job.get("query", "")
        if job.get("state") == "done":
            report = job.get("report") or {}
            lines = [f"Research done: {query}", ""]
            summary = str(report.get("summary") or "").strip()
            if summary:
                lines.append(summary[:1500])
            sources = report.get("sources") or []
            if sources:
                lines.append("")
                lines.append("Sources:")
                for s in sources[:8]:
                    lines.append(f"  [{s.get('n')}] {s.get('title') or s.get('url')} — {s.get('url')}")
            notify(context, "research", f"Research ready: {query[:80]}",
                   "\n".join(lines), force=True)
        else:
            notify(context, "research", f"Research failed: {query[:80]}",
                   f"The background research run failed: {job.get('error') or 'unknown error'}",
                   force=True)
    except Exception as exc:  # noqa: BLE001
        _log.debug("research job delivery failed: %s", exc)


def _work(context: Any, job_id: str) -> None:
    """The daemon worker.  Never raises — every outcome lands on the row
    and (unless cancelled) is delivered."""
    job = _load(context, job_id)
    if job is None:
        return
    if _cas_state(context, job_id, {"queued"}, "running",
                  progress="searching") is None:
        return  # cancelled (or raced) before the worker started
    try:
        from .engine import SearchEngine

        engine = SearchEngine(context)
        params = job.get("params") or {}
        if job.get("mode") == "deep":
            from ..power import power_mode_for

            if not power_mode_for(context).active:
                raise RuntimeError("deep research is a power-mode capability — enable power mode first")
            from .deep import DeepResearcher

            _set(context, job_id, progress="decomposing + fan-out")
            researcher = DeepResearcher(
                context, engine=engine,
                max_pages=params.get("pages", 8),
                pages_per_query=max(2, params.get("pages", 8) // 3),
                dig=params.get("dig", True),
            )
            report = researcher.run(job["query"], scope=params.get("scope", "auto"))
        else:
            _set(context, job_id, progress="searching + reading")
            report = engine.run(
                job["query"], mode="quick", pages=params.get("pages", 3),
                crawl=params.get("crawl", False),
                freshness=params.get("freshness", ""),
                scope=params.get("scope", "auto"),
            )
        done = _cas_state(
            context, job_id, {"running"}, "done", report=report,
            progress=f"done in {report.get('seconds', 0)}s, "
                     f"{len(report.get('pages_read', []))} pages read",
        )
        if done is None:
            # cancelled while the worker ran: record it, don't deliver
            _set(context, job_id, progress="cancelled before delivery")
            return
        _deliver(context, done)
    except Exception as exc:  # noqa: BLE001 - the whole point: a failed run is a failed *job*, not a crash
        _log.warning("research job %s failed: %s", job_id, exc)
        failed = _cas_state(context, job_id, {"queued", "running"}, "failed",
                            error=f"{type(exc).__name__}: {exc}",
                            progress="failed")
        if failed is None:
            return  # cancelled won the race; the cancellation stands
        try:
            _deliver(context, failed)
        except Exception:  # noqa: BLE001
            _log.debug("failure delivery failed too", exc_info=True)
        _log.debug("research job traceback: %s", traceback.format_exc())
