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
2. **Research** — emit an organ event to research; the research organ
   investigates on its tick.
3. **Propose** — when research lands, draft a fix proposal (what's wrong,
   what the research found, what to change). Proposals go to the owner
   for approval — the system never rewrites itself silently.
4. **Sandbox test** — approved proposals get built and tested in a
   sandbox (never on the live tree) before the owner merges.

Nothing here modifies production code autonomously. Detection,
research, and proposals are automatic; building and merging need the
owner's explicit go-ahead. That's the safety line.
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
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
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


def _new_id() -> str:
    from ..core.ids import new_id
    return new_id("weak")


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
        # Hand off to research via the organ bus.
        try:
            _organs.emit(db, src="autonomy.weakness", dst="research",
                         kind="weakness.investigate",
                         payload={"weakness_id": wid, "kind": kind,
                                  "subject": subject,
                                  "evidence": evidence or {}})
        except Exception:  # noqa: BLE001
            _log.warning("weakness→research handoff failed", exc_info=True)
        try:
            global_bus.publish(Event(
                topic="weakness.detected",
                data={"id": wid, "kind": kind, "subject": subject},
                source="nomorals.autonomy.weakness",
            ))
        except Exception:  # noqa: BLE001
            pass
    return wid


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
    try:
        global_bus.publish(Event(
            topic="weakness.proposed",
            data={"id": weakness_id, "proposal": proposal[:500]},
            source="nomorals.autonomy.weakness",
        ))
    except Exception:  # noqa: BLE001
        pass


def approve_weakness(db: Any, weakness_id: str) -> bool:
    """Owner approves a proposal → status approved, ready for sandbox."""
    ensure_schema(db)
    cur = db.execute(
        "UPDATE weaknesses SET status = 'approved' "
        "WHERE id = ? AND status = 'proposed'",
        (weakness_id,),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return cur.rowcount > 0


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
    """
    opened = 0
    cutoff = time.time() - FAILURE_WINDOW
    failures: dict[str, int] = {}
    for rec in tool_audit:
        if rec.get("ts", 0) < cutoff:
            continue
        if not rec.get("ok"):
            tool = str(rec.get("tool", "?"))
            failures[tool] = failures.get(tool, 0) + 1
    for tool, count in failures.items():
        if count >= FAILURE_THRESHOLD:
            report_weakness(db, "tool_failure", tool,
                            {"failures_in_24h": count})
            opened += 1
    return opened
