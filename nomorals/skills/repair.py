"""Structured repair tickets for failed skill runs.

When a skill run fails, the runner builds a :class:`RepairTicket`: the
skill name + version, the failing step, the exact inputs that step got,
the error, and — the part that makes it actionable — a concrete
``suggested_fix`` naming what to change (a wiring expression, a schema,
a missing tool, a disabled skill).  Tickets persist in the
``skill_repair_tickets`` table so repeated failures are visible as a
pattern, not just a log line.

Tickets have a lifecycle — ``open`` → ``resolved`` (→ ``open`` on
reopen) — because a ticket you can't close is a write-only log.
:meth:`RepairTicketStore.patterns` clusters open tickets by normalized
error signature, which finally delivers the "repeated failures are
visible as a pattern" promise: same signature, many tickets, one root
cause.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ..storage.db import Database

_log = get_logger(__name__)

__all__ = ["RepairTicket", "RepairTicketStore", "build_ticket", "suggest_fix",
           "format_ticket", "REPAIR_TICKETS_DDL"]

REPAIR_TICKETS_DDL = """
CREATE TABLE IF NOT EXISTS skill_repair_tickets (
    id          TEXT PRIMARY KEY,
    skill_name  TEXT NOT NULL,
    version     TEXT NOT NULL,
    step_index  INTEGER NOT NULL,
    tool        TEXT NOT NULL DEFAULT '',
    inputs      TEXT NOT NULL DEFAULT '{}',
    error       TEXT NOT NULL DEFAULT '',
    suggested_fix TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'open',
    resolved_at REAL,
    resolution_note TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_repair_tickets_skill
    ON skill_repair_tickets(skill_name, created_at);
CREATE INDEX IF NOT EXISTS idx_repair_tickets_status
    ON skill_repair_tickets(status, created_at);
"""


def _ensure_columns(db: "Database") -> None:
    """Guarded migration for installs predating the ticket lifecycle."""
    try:
        cols = {r["name"]
                for r in db.query("PRAGMA table_info(skill_repair_tickets)")}
    except Exception:  # noqa: BLE001 — table may not exist yet
        return
    wanted = {
        "status": "ALTER TABLE skill_repair_tickets ADD COLUMN "
                  "status TEXT NOT NULL DEFAULT 'open'",
        "resolved_at": "ALTER TABLE skill_repair_tickets ADD COLUMN "
                       "resolved_at REAL",
        "resolution_note": "ALTER TABLE skill_repair_tickets ADD COLUMN "
                           "resolution_note TEXT NOT NULL DEFAULT ''",
    }
    for col, ddl in wanted.items():
        if col not in cols:
            try:
                db.execute(ddl)
            except Exception as exc:  # noqa: BLE001 — never break startup
                _log.debug("repair tickets migration skipped: %s", exc)


def _normalize_signature(error: str) -> str:
    """Normalize an error into a clusterable signature.

    Lowercases, strips hex ids / numbers / quoted specifics, collapses
    whitespace, and truncates — so "connection refused on attempt 3" and
    "connection refused on attempt 7" land in the same bucket.
    """
    text = (error or "").lower()
    text = re.sub(r"0x[0-9a-f]+", "#", text)
    text = re.sub(r"\b[a-f0-9]{8,}\b", "#", text)
    text = re.sub(r"\d+(\.\d+)+", "#", text)   # versions, ips, decimals
    text = re.sub(r"(?<!\w)\d+(?!\w)", "#", text)  # bare numbers
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "#", text)  # quoted specifics
    text = re.sub(r"\s+", " ", text).strip()
    return text[:160]


@dataclass
class RepairTicket:
    """One failed skill run, diagnosed."""

    id: str
    skill_name: str
    version: str
    step_index: int  # -1 = input validation, -2 = skill-level (unknown/disabled)
    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    suggested_fix: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = "open"
    resolved_at: float | None = None
    resolution_note: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_name": self.skill_name,
            "version": self.version,
            "step_index": self.step_index,
            "tool": self.tool,
            "inputs": dict(self.inputs),
            "error": self.error,
            "suggested_fix": self.suggested_fix,
            "created_at": self.created_at,
            "status": self.status,
            "resolved_at": self.resolved_at,
            "resolution_note": self.resolution_note,
        }


def suggest_fix(*, step_index: int, tool: str, error: str,
                step_input: dict[str, Any] | None = None,
                missing_keys: list[str] | None = None,
                available_tools: list[str] | None = None,
                disabled: bool = False,
                unknown_skill: bool = False,
                skill_name: str = "") -> str:
    """Build a concrete suggested fix from the failure evidence.

    Every branch names the thing to change — a wiring expression, a schema
    key, a tool name, a timeout, a retry policy, or a CLI command — never
    a placeholder.
    """
    where = (f"step {step_index} ({tool!r})" if step_index >= 0
             else "skill input validation" if step_index == -1
             else f"skill {skill_name!r}")
    err = (error or "").strip()

    if unknown_skill:
        names = available_tools or []
        hint = ""
        close = difflib.get_close_matches(skill_name, names, n=3, cutoff=0.6)
        if close:
            hint = f" Did you mean: {', '.join(close)}?"
        known = (f" Installed skills: {', '.join(sorted(names)[:8])}."
                 if names else "")
        return (f"{where}: no installed skill named {skill_name!r}.{hint}"
                f"{known} Install it first: "
                f"`nm skill install --manifest <file.json>`.")

    if disabled:
        return (f"{where}: the skill is installed but disabled, so the "
                f"runner refused to execute it. Enable it: "
                f"`nm skill enable '{skill_name}'`.")

    if missing_keys:
        keys = ", ".join(f"{k!r}" for k in missing_keys)
        return (f"{where} output is missing required key(s): {keys}. Either "
                f"tighten the producing step so it returns them, add a "
                f"transform step that derives them from an earlier step's "
                f"output (wire it as e.g. "
                f"{{\"result\": \"${step_index - 1 if step_index > 0 else 0}.<key>\"}}), "
                f"or relax the manifest's output_schema.")

    low = err.lower()
    if "timed out" in low or "timeout" in low:
        return (f"{where}: the step exceeded its time budget ({err[:160]}). "
                f"If the tool is just slow, raise the manifest's "
                f"`timeout_s`; if it hangs, fix the tool. Consider a "
                f"`retries` policy too — timeouts are transient-shaped.")

    if "attempt" in low and ("retries" in low or "retry" in low
                             or "exhausted" in low):
        return (f"{where}: retries exhausted ({err[:160]}). The downstream "
                f"is sick, not flaky — check the tool/service health "
                f"before raising `retries.max_attempts`; a circuit breaker "
                f"or a fallback step (`on_error` with `continue: true`) "
                f"fits better than more attempts.")

    if "unknown tool" in low or "not registered" in low:
        names = available_tools or []
        close = difflib.get_close_matches(tool, names, n=3, cutoff=0.6)
        hint = (f" Did you mean: {', '.join(close)}?"
                if close else "")
        return (f"{where}: tool {tool!r} is not registered.{hint} Fix the "
                f"manifest's tools list, or install the module that provides "
                f"the tool (custom tools drop into nomorals/tools/custom/).")

    if "lacks" in low and "capability" in low or "capabilitydenied" in low.replace(" ", ""):
        return (f"{where}: the runner's capability grant denied tool {tool!r} "
                f"({err[:160]}). Grant the tool's capability to the skill "
                f"runner, or run the skill under an actor that holds it.")

    if "$env" in err and "disabled" in low:
        return (f"{where}: the manifest wires `$env.*` but the runner was "
                f"created with allow_env=False ({err[:160]}). Either pass "
                f"the value as run input instead, or construct the "
                f"SkillRunner with allow_env=True.")

    if "could not resolve" in low or "not present in the referenced output" in low:
        return (f"{where}: a wiring expression could not be resolved "
                f"({err[:200]}). Wire the parameter from a key that exists: "
                f"`$input.<key>` for run inputs, `$<n>.<key>` or "
                f"`$<step_id>.<key>` for an earlier step's output "
                f"(n < {step_index}), or pass the value literally.")

    if step_input is not None and ("missing" in low and "required" in low):
        return (f"{where}: the tool rejected its inputs ({err[:200]}). "
                f"Add the missing parameter to the step's wiring or to the "
                f"run input, e.g. "
                f"`nm skill run '{skill_name}' --input '{{\"<key>\": ...}}'`. "
                f"If the key has a sensible fallback, declare it as a "
                f"`default` in the manifest's input_schema.")

    detail = f": {err[:220]}" if err else ""
    return (f"{where} failed{detail}. Inspect the recorded step inputs, fix "
            f"the data or the wiring that produced them, and re-run. "
            f"Repeated identical failures suggest the manifest's schemas no "
            f"longer match what the tools return — update them.")


def build_ticket(skill_name: str, version: str, *, step_index: int,
                 tool: str = "", step_input: dict[str, Any] | None = None,
                 error: str = "", missing_keys: list[str] | None = None,
                 available_tools: list[str] | None = None,
                 disabled: bool = False,
                 unknown_skill: bool = False) -> RepairTicket:
    """Assemble a ticket and its suggested fix from failure evidence."""
    return RepairTicket(
        id=new_short_id("rt"),
        skill_name=skill_name,
        version=version,
        step_index=step_index,
        tool=tool,
        inputs=dict(step_input or {}),
        error=error,
        suggested_fix=suggest_fix(
            step_index=step_index, tool=tool, error=error,
            step_input=step_input, missing_keys=missing_keys,
            available_tools=available_tools, disabled=disabled,
            unknown_skill=unknown_skill, skill_name=skill_name),
    )


def format_ticket(ticket: RepairTicket) -> str:
    """Render a ticket as a readable incident card."""
    where = (f"step {ticket.step_index} ({ticket.tool!r})"
             if ticket.step_index >= 0
             else "skill input validation" if ticket.step_index == -1
             else "skill-level")
    status = "● OPEN" if ticket.is_open else "✓ RESOLVED"
    lines = [
        f"ticket {ticket.id}  {status}",
        f"  skill   {ticket.skill_name} v{ticket.version} — {where}",
        f"  error   {(ticket.error or '(no error text)')[:200]}",
    ]
    if ticket.inputs:
        peek = json.dumps(ticket.inputs, ensure_ascii=False, default=str)
        lines.append(f"  inputs  {peek[:160]}")
    if ticket.suggested_fix:
        lines.append(f"  fix     {ticket.suggested_fix[:300]}")
    if not ticket.is_open:
        note = ticket.resolution_note or "(no note)"
        lines.append(f"  closed  {note[:160]}")
    return "\n".join(lines)


class RepairTicketStore:
    """Persisted repair tickets."""

    def __init__(self, db: "Database") -> None:
        self.db = db
        db.executescript(REPAIR_TICKETS_DDL)
        _ensure_columns(db)

    def save(self, ticket: RepairTicket) -> RepairTicket:
        self.db.execute(
            "INSERT OR REPLACE INTO skill_repair_tickets "
            "(id, skill_name, version, step_index, tool, inputs, error, "
            "suggested_fix, status, resolved_at, resolution_note, "
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ticket.id, ticket.skill_name, ticket.version,
             ticket.step_index, ticket.tool,
             json.dumps(ticket.inputs, ensure_ascii=False, default=str),
             ticket.error, ticket.suggested_fix, ticket.status,
             ticket.resolved_at, ticket.resolution_note,
             ticket.created_at),
        )
        _log.info("repair ticket %s saved for skill %s (%s)", ticket.id,
                  ticket.skill_name, ticket.error[:80])
        return ticket

    def get(self, ticket_id: str) -> RepairTicket | None:
        row = self.db.query_one(
            "SELECT * FROM skill_repair_tickets WHERE id=?", (ticket_id,))
        return self._from_row(row) if row else None

    def list(self, skill_name: str = "", limit: int = 50,
             status: str | None = None) -> list[RepairTicket]:
        clauses: list[str] = []
        params: list[Any] = []
        if skill_name:
            clauses.append("skill_name=?")
            params.append(skill_name)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        rows = self.db.query(
            f"SELECT * FROM skill_repair_tickets {where} "
            f"ORDER BY created_at DESC LIMIT ?", tuple(params))
        return [self._from_row(r) for r in rows]

    def count(self, skill_name: str = "",
              status: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if skill_name:
            clauses.append("skill_name=?")
            params.append(skill_name)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.db.query_one(
            f"SELECT COUNT(*) AS n FROM skill_repair_tickets {where}",
            tuple(params))
        return int(row["n"]) if row else 0

    # ── lifecycle ───────────────────────────────────────────────────────
    def resolve(self, ticket_id: str, note: str = "") -> bool:
        """Close a ticket with a resolution note.  Returns False when the
        ticket does not exist."""
        cur = self.db.execute(
            "UPDATE skill_repair_tickets SET status='resolved', "
            "resolved_at=?, resolution_note=? "
            "WHERE id=? AND status='open'",
            (time.time(), note or "", ticket_id))
        return (cur.rowcount or 0) > 0

    def reopen(self, ticket_id: str) -> bool:
        """Re-open a resolved ticket.  Returns False when the ticket does
        not exist or is already open."""
        cur = self.db.execute(
            "UPDATE skill_repair_tickets SET status='open', "
            "resolved_at=NULL, resolution_note='' "
            "WHERE id=? AND status='resolved'", (ticket_id,))
        return (cur.rowcount or 0) > 0

    # ── patterns: repeated failures as a pattern ────────────────────────
    def patterns(self, skill_name: str = "", limit: int = 50,
                 stale_days: float = 7.0) -> list[dict[str, Any]]:
        """Cluster open tickets by normalized error signature.

        Returns one entry per signature: count, first/last seen, an
        example ticket id, and whether the pattern went ``stale``
        (no occurrence in ``stale_days``).  A high count on one
        signature is one root cause wearing many ticket ids.
        """
        tickets = self.list(skill_name, limit=max(1, limit) * 4,
                            status="open")
        now = time.time()
        buckets: dict[str, dict[str, Any]] = {}
        for ticket in tickets:
            sig = _normalize_signature(ticket.error)
            bucket = buckets.setdefault(sig, {
                "signature": sig or "(empty error)",
                "count": 0,
                "first_at": ticket.created_at,
                "last_at": ticket.created_at,
                "example_id": ticket.id,
                "skill_name": ticket.skill_name,
                "tool": ticket.tool,
            })
            bucket["count"] += 1
            bucket["first_at"] = min(bucket["first_at"], ticket.created_at)
            bucket["last_at"] = max(bucket["last_at"], ticket.created_at)
        out = []
        for bucket in buckets.values():
            bucket["stale"] = (now - bucket["last_at"]) > stale_days * 86400
            out.append(bucket)
        out.sort(key=lambda b: b["count"], reverse=True)
        return out[:max(1, limit)]

    @staticmethod
    def _from_row(row: dict[str, Any]) -> RepairTicket:
        try:
            inputs = json.loads(row.get("inputs") or "{}")
            if not isinstance(inputs, dict):
                inputs = {}
        except (TypeError, ValueError):
            inputs = {}
        return RepairTicket(
            id=row["id"],
            skill_name=row.get("skill_name", ""),
            version=row.get("version", ""),
            step_index=int(row.get("step_index", -2)),
            tool=row.get("tool", "") or "",
            inputs=inputs,
            error=row.get("error", "") or "",
            suggested_fix=row.get("suggested_fix", "") or "",
            created_at=float(row.get("created_at", 0.0)),
            status=row.get("status", "open") or "open",
            resolved_at=(float(row["resolved_at"])
                         if row.get("resolved_at") is not None else None),
            resolution_note=row.get("resolution_note", "") or "",
        )
