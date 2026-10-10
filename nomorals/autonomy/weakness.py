"""Weakness detection → research → proposal → sandbox build/test.

The self-healing loop. The system watches itself for:

* **Tool failures** — tools that error repeatedly (tracked from the tool
  loop's audit trail).
* **Knowledge gaps** — questions the brain couldn't answer (the research
  organ's ``knowledge_gaps`` table).
* **Error patterns** — exceptions clustering in logs.
* **Capability gaps** — the owner asks for something no tool covers.

For each weakness:
1. **Detect** — score it (frequency × impact). Above threshold → open.
2. **Research** — the weakness is translated into a ``directive.gap``
   organ event for the research organ (the one directive kind the
   research organ's tick actually drains). The research organ
   investigates on its tick.
3. **Propose** — when research lands, draft a fix proposal (what's wrong,
   what the research found, what to change). Proposals go to the owner
   for approval — the system never rewrites itself silently.
4. **Sandbox test** — approved proposals emit a ``weakness.approved`` bus
   event and a scheduler organ event so a sandbox build mission can be
   queued without anyone tripping a command. Building and merging still
   need the owner's explicit go-ahead. That's the safety line.
5. **Recover** — success streaks auto-resolve ``tool_failure`` cases, so
   fixed tools stop haunting the list.

Detection, research, and proposals are automatic; recovery is automatic;
building and merging need the owner. Nothing here modifies production
code autonomously.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .. import organs as _organs

_log = get_logger(__name__)

#: Failure count in the window that opens a weakness case.
FAILURE_THRESHOLD = 3
#: Seconds of history considered for failure clustering.
FAILURE_WINDOW = 86400  # 24h
#: Consecutive successes that auto-resolve a tool_failure case.
RECOVERY_STREAK = 5


@dataclass
class Weakness:
    """A detected weakness case."""
    id: str = ""
    kind: str = ""  # tool_failure | knowledge_gap | error_pattern | capability_gap
    subject: str = ""  # tool name, question, error signature, requested capability
    evidence: dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    occurrences: int = 1
    status: str = "open"  # open | researching | proposed | approved | resolved | dismissed
    proposal: str = ""


def ensure_schema(db: Any) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS weaknesses (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            subject TEXT NOT NULL,
            evidence TEXT NOT NULL DEFAULT '{}',
            first_seen REAL NOT NULL DEFAULT 0,
            last_seen REAL NOT NULL DEFAULT 0,
            occurrences INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'open',
            proposal TEXT NOT NULL DEFAULT ''
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_weaknesses_status "
        "ON weaknesses(status, last_seen)"
    )
    # Tool health streaks for auto-recovery.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_health (
            tool TEXT PRIMARY KEY,
            ok_streak INTEGER NOT NULL DEFAULT 0,
            fail_streak INTEGER NOT NULL DEFAULT 0,
            last_ts REAL NOT NULL DEFAULT 0
        )
        """
    )


def _new_id() -> str:
    from ..core.ids import new_id
    return new_id("weak")


def _ledger(db: Any, kind: str, ref_id: str, summary: str,
            *, ok: bool = True,
            metadata: dict[str, Any] | None = None) -> None:
    try:
        from ..agents.autonomy_ledger import record_ledger
        record_ledger(db, "weakness", kind, ref_id, summary, ok=ok,
                      metadata=metadata)
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("weakness ledger write failed", exc_info=True)


def report_weakness(db: Any, kind: str, subject: str,
                    evidence: dict[str, Any] | None = None) -> str:
    """Report a weakness sighting. Clusters by (kind, subject).

    Returns the weakness id. When occurrences cross the threshold,
    emits ``weakness.detected`` and kicks off research.
    """
    ensure_schema(db)
    now = time.time()
    ev = json.dumps(evidence or {})
    row = db.execute(
        "SELECT id, occurrences, status FROM weaknesses "
        "WHERE kind = ? AND subject = ? AND status NOT IN "
        "('resolved', 'dismissed')",
        (kind, subject),
    ).fetchone()
    if row:
        wid, occ, status = row[0], int(row[1]), row[2]
        db.execute(
            "UPDATE weaknesses SET occurrences = occurrences + 1, "
            "last_seen = ?, evidence = ? WHERE id = ?",
            (now, ev, wid),
        )
        new_occ = occ + 1
    else:
        wid = _new_id()
        db.execute(
            "INSERT INTO weaknesses (id, kind, subject, evidence, "
            "first_seen, last_seen, occurrences) "
            "VALUES (?, ?, ?, ?, ?, ?, 1)",
            (wid, kind, subject, ev, now, now),
        )
        new_occ = 1
        status = "open"
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass

    if new_occ >= FAILURE_THRESHOLD and status == "open":
        db.execute(
            "UPDATE weaknesses SET status = 'researching' WHERE id = ?",
            (wid,),
        )
        try:
            db.commit()
        except Exception:  # noqa: BLE001
            pass
        _log.info("weakness detected: %s %s (%d occurrences)",
                  kind, subject, new_occ)
        _ledger(db, "detected", wid,
                f"weakness detected: {kind} {subject} "
                f"({new_occ} occurrences)",
                metadata={"kind": kind, "subject": subject,
                          "occurrences": new_occ})
        investigate_weakness(db, wid)
    return wid


def investigate_weakness(db: Any, weakness_id: str) -> bool:
    """Hand a weakness to the research organ for investigation.

    The research organ's tick only drains ``directive.watch`` and
    ``directive.gap`` organ events — so the weakness is translated into
    a ``directive.gap`` (a question it genuinely investigates) rather
    than a custom kind that would be consumed and ignored. Returns True
    when the handoff was queued.
    """
    ensure_schema(db)
    row = db.execute(
        "SELECT kind, subject, evidence, occurrences FROM weaknesses "
        "WHERE id = ?", (weakness_id,),
    ).fetchone()
    if not row:
        return False
    kind, subject, evidence_raw, occurrences = row
    try:
        evidence = json.loads(evidence_raw or "{}")
    except (ValueError, TypeError):
        evidence = {}
    question = _investigation_question(kind, subject, evidence,
                                       int(occurrences or 0))
    try:
        _organs.emit(
            db, src="autonomy.weakness", dst="research",
            kind="directive.gap",
            payload={
                "question": question,
                "context": json.dumps({
                    "weakness_id": weakness_id, "kind": kind,
                    "subject": subject, "evidence": evidence,
                }),
                "origin": "weakness.investigation",
            })
        _ledger(db, "investigating", weakness_id,
                f"investigating {kind} {subject} via research organ",
                metadata={"question": question[:300]})
        _log.info("weakness %s → research (directive.gap)", weakness_id)
        try:
            global_bus.publish(Event(
                topic="weakness.detected",
                data={"id": weakness_id, "kind": kind, "subject": subject},
                source="nomorals.autonomy.weakness",
            ))
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception:  # noqa: BLE001
        _log.warning("weakness→research handoff failed", exc_info=True)
        return False


def _investigation_question(kind: str, subject: str,
                            evidence: dict[str, Any],
                            occurrences: int) -> str:
    """Frame the weakness as a researchable question."""
    if kind == "tool_failure":
        return (
            f"Why is the tool '{subject}' failing? It failed {occurrences} "
            f"times in 24h. Known error evidence: "
            f"{json.dumps(evidence)[:400]}. Find the likely root cause and "
            "the minimal fix (prompt change, input validation, retry/backoff "
            "strategy, or replacement approach)."
        )
    if kind == "knowledge_gap":
        return (
            f"The system could not answer: '{subject}'. Research this topic "
            "and produce a concise, sourced brief the knowledge base can "
            "ingest."
        )
    if kind == "error_pattern":
        return (
            f"Exceptions are clustering around: {subject} "
            f"({occurrences} occurrences). Evidence: "
            f"{json.dumps(evidence)[:400]}. Identify the root cause pattern "
            "and the minimal hardening fix."
        )
    # capability_gap
    return (
        f"The owner asked for '{subject}' and no tool covers it. Research "
        "how this capability could be built: existing APIs, libraries, and "
        "the smallest viable implementation approach."
    )


def idle_tick(db: Any) -> dict[str, Any]:
    """The idle-cycle weakness step. Called by the IdleCoordinator.

    Ensures every open/thresholded case is routed to research, and
    returns a status report (never raises).
    """
    report: dict[str, Any] = {"routed": 0, "open": 0}
    try:
        ensure_schema(db)
        rows = db.execute(
            "SELECT id, status FROM weaknesses "
            "WHERE status NOT IN ('resolved', 'dismissed')"
        ).fetchall()
        report["open"] = len(rows)
        for wid, status in rows:
            if status in ("open",):
                # Thresholded but never routed (e.g. crashed before).
                occurrences = db.execute(
                    "SELECT occurrences FROM weaknesses WHERE id = ?",
                    (wid,)).fetchone()
                if occurrences and int(occurrences[0]) >= FAILURE_THRESHOLD:
                    db.execute(
                        "UPDATE weaknesses SET status = 'researching' "
                        "WHERE id = ?", (wid,))
                    try:
                        db.commit()
                    except Exception:  # noqa: BLE001
                        pass
                    if investigate_weakness(db, wid):
                        report["routed"] += 1
            elif status == "researching":
                # Still researching — the research organ owns it now.
                pass
    except Exception as exc:  # noqa: BLE001
        _log.warning("weakness idle_tick failed: %s", exc, exc_info=True)
        report["error"] = str(exc)[:200]
    return report


def record_proposal(db: Any, weakness_id: str, proposal: str) -> None:
    """Record a fix proposal for a weakness. Owner approves via chat."""
    ensure_schema(db)
    db.execute(
        "UPDATE weaknesses SET status = 'proposed', proposal = ? "
        "WHERE id = ?",
        (proposal, weakness_id),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    _log.info("weakness proposal recorded: %s", weakness_id)
    _ledger(db, "proposed", weakness_id,
            f"fix proposal recorded for {weakness_id}",
            metadata={"proposal": proposal[:500]})
    try:
        global_bus.publish(Event(
            topic="weakness.proposed",
            data={"id": weakness_id, "proposal": proposal[:500]},
            source="nomorals.autonomy.weakness",
        ))
    except Exception:  # noqa: BLE001
        pass


def approve_weakness(db: Any, weakness_id: str) -> bool:
    """Owner approves a proposal → status approved, sandbox build queued.

    Approval doesn't build anything itself — it queues a sandbox-build
    request through the scheduler organ event and a ``weakness.approved``
    bus event. The improvement/scheduler systems pick those up without
    manual commands. The sandbox build itself still runs outside the
    live tree and the merge needs the owner.
    """
    ensure_schema(db)
    row = db.execute(
        "SELECT kind, subject, proposal FROM weaknesses "
        "WHERE id = ? AND status = 'proposed'",
        (weakness_id,),
    ).fetchone()
    if not row:
        return False
    kind, subject, proposal = row
    db.execute(
        "UPDATE weaknesses SET status = 'approved' WHERE id = ?",
        (weakness_id,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    _log.info("weakness approved: %s", weakness_id)
    _ledger(db, "approved", weakness_id,
            f"proposal approved for {kind} {subject} — sandbox build queued",
            metadata={"kind": kind, "subject": subject})
    payload = {"weakness_id": weakness_id, "kind": kind,
               "subject": subject, "proposal": proposal,
               "sandbox": True}
    try:
        _organs.emit(db, src="autonomy.weakness", dst="scheduler",
                     kind="weakness.sandbox_build", payload=payload)
    except Exception:  # noqa: BLE001
        _log.warning("weakness→scheduler handoff failed", exc_info=True)
    try:
        global_bus.publish(Event(
            topic="weakness.approved",
            data=payload,
            source="nomorals.autonomy.weakness",
        ))
    except Exception:  # noqa: BLE001
        pass
    return True


def resolve_weakness(db: Any, weakness_id: str, how: str = "") -> bool:
    """Mark a weakness resolved (fixed, recovered, or obsolete)."""
    ensure_schema(db)
    cur = db.execute(
        "UPDATE weaknesses SET status = 'resolved' "
        "WHERE id = ? AND status NOT IN ('resolved', 'dismissed')",
        (weakness_id,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    if cur.rowcount:
        _ledger(db, "resolved", weakness_id,
                f"weakness resolved: {how or 'no reason given'}")
        try:
            global_bus.publish(Event(
                topic="weakness.resolved",
                data={"id": weakness_id, "how": how[:200]},
                source="nomorals.autonomy.weakness",
            ))
        except Exception:  # noqa: BLE001
            pass
    return bool(cur.rowcount)


def dismiss_weakness(db: Any, weakness_id: str) -> bool:
    """Owner dismisses a weakness without a fix."""
    ensure_schema(db)
    cur = db.execute(
        "UPDATE weaknesses SET status = 'dismissed' WHERE id = ?",
        (weakness_id,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    if cur.rowcount:
        _ledger(db, "dismissed", weakness_id,
                f"weakness dismissed by owner: {weakness_id}")
    return bool(cur.rowcount)


def record_tool_result(db: Any, tool: str, ok: bool,
                       error: str | None = None) -> dict[str, Any]:
    """Feed one tool-loop outcome into detection AND recovery.

    Failures cluster into weakness cases (via :func:`report_weakness`);
    successes build an ok-streak that auto-resolves a ``tool_failure``
    case once the tool is healthy again. This closes the loop: fixed
    tools stop haunting the weakness list without manual dismissal.
    """
    ensure_schema(db)
    now = time.time()
    tool = str(tool or "?")
    db.execute(
        "INSERT INTO tool_health (tool, ok_streak, fail_streak, last_ts) "
        "VALUES (?, 0, 0, ?) "
        "ON CONFLICT (tool) DO UPDATE SET last_ts = excluded.last_ts",
        (tool, now),
    )
    out: dict[str, Any] = {"tool": tool}
    if ok:
        db.execute(
            "UPDATE tool_health SET ok_streak = ok_streak + 1, "
            "fail_streak = 0 WHERE tool = ?",
            (tool,),
        )
        streak = db.execute(
            "SELECT ok_streak FROM tool_health WHERE tool = ?",
            (tool,)).fetchone()
        ok_streak = int(streak[0]) if streak else 0
        out["ok_streak"] = ok_streak
        if ok_streak >= RECOVERY_STREAK:
            row = db.execute(
                "SELECT id FROM weaknesses WHERE kind = 'tool_failure' "
                "AND subject = ? AND status NOT IN "
                "('resolved', 'dismissed')",
                (tool,)).fetchone()
            if row:
                resolve_weakness(db, row[0],
                                 f"auto-resolved: {ok_streak} consecutive "
                                 "successes")
                out["auto_resolved"] = row[0]
    else:
        db.execute(
            "UPDATE tool_health SET fail_streak = fail_streak + 1, "
            "ok_streak = 0 WHERE tool = ?",
            (tool,),
        )
        wid = report_weakness(db, "tool_failure", tool,
                              {"error": (error or "")[:300],
                               "ts": now})
        out["weakness_id"] = wid
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return out


def tool_health(db: Any, tool: str) -> dict[str, Any]:
    """Current health streaks for one tool."""
    ensure_schema(db)
    row = db.execute(
        "SELECT ok_streak, fail_streak, last_ts FROM tool_health "
        "WHERE tool = ?", (str(tool or "?",),)).fetchone()
    if not row:
        return {"tool": tool, "ok_streak": 0, "fail_streak": 0,
                "last_ts": 0.0}
    return {"tool": tool, "ok_streak": int(row[0]),
            "fail_streak": int(row[1]), "last_ts": float(row[2] or 0.0)}


def open_weaknesses(db: Any) -> list[dict[str, Any]]:
    """All weaknesses needing attention."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT id, kind, subject, evidence, first_seen, last_seen, "
        "occurrences, status, proposal FROM weaknesses "
        "WHERE status NOT IN ('resolved', 'dismissed') "
        "ORDER BY occurrences DESC, last_seen DESC"
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0], "kind": r[1], "subject": r[2],
            "evidence": json.loads(r[3] or "{}"),
            "first_seen": r[4], "last_seen": r[5],
            "occurrences": r[6], "status": r[7], "proposal": r[8],
        })
    return out


def scan_tool_failures(db: Any, tool_audit: list[dict[str, Any]]) -> int:
    """Feed tool-loop audit records into weakness detection.

    ``tool_audit``: list of {tool, ok, error, ts}. Returns new cases opened.
    Success records also feed recovery streaks.
    """
    opened = 0
    cutoff = time.time() - FAILURE_WINDOW
    failures: dict[str, int] = {}
    for rec in tool_audit:
        if rec.get("ts", 0) < cutoff:
            continue
        tool = str(rec.get("tool", "?"))
        if rec.get("ok"):
            record_tool_result(db, tool, True)
        else:
            failures[tool] = failures.get(tool, 0) + 1
    for tool, count in failures.items():
        if count >= FAILURE_THRESHOLD:
            report_weakness(db, "tool_failure", tool,
                            {"failures_in_24h": count})
            opened += 1
    return opened


def scan_ledger_failures(db: Any, window_hours: float = 24.0,
                         min_runs: int = 10,
                         rate_threshold: float = 0.5) -> list[str]:
    """Cross-system trigger: ledger failure rates → weakness cases.

    Any autonomous system whose recent failure rate crosses the threshold
    gets a weakness case automatically. This is systems watching each
    other — the scheduler's repeated failures, for example, become a
    self-investigation without anyone tripping a command.
    """
    from ..agents.autonomy_ledger import ledger_failure_rate
    opened: list[str] = []
    systems = ("scheduler", "mission", "trigger", "cognition", "pulse",
               "autonomy", "watcher", "idle", "presence", "weakness",
               "improvement")
    for system in systems:
        stats = ledger_failure_rate(db, system, window_hours=window_hours)
        if stats["runs"] >= min_runs and \
                stats["failure_rate"] >= rate_threshold:
            wid = report_weakness(
                db, "error_pattern", f"ledger:{system}",
                {"system": system,
                 "failure_rate": stats["failure_rate"],
                 "runs": stats["runs"], "failures": stats["failures"],
                 "window_hours": window_hours})
            opened.append(wid)
    return opened
