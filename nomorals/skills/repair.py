"""Structured repair tickets for failed skill runs.

When a skill run fails, the runner builds a :class:`RepairTicket`: the
skill name + version, the failing step, the exact inputs that step got,
the error, and — the part that makes it actionable — a concrete
``suggested_fix`` naming what to change (a wiring expression, a schema,
a missing tool, a disabled skill).  Tickets persist in the
``skill_repair_tickets`` table so repeated failures are visible as a
pattern, not just a log line.
"""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ..storage.db import Database

_log = get_logger(__name__)

__all__ = ["RepairTicket", "RepairTicketStore", "build_ticket", "suggest_fix",
           "REPAIR_TICKETS_DDL"]

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
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_repair_tickets_skill
    ON skill_repair_tickets(skill_name, created_at);
"""


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
    key, a tool name, or a CLI command — never a placeholder.
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

    if "could not resolve" in low or "not present in the referenced output" in low:
        return (f"{where}: a wiring expression could not be resolved "
                f"({err[:200]}). Wire the parameter from a key that exists: "
                f"`$input.<key>` for run inputs, `$<n>.<key>` for an "
                f"earlier step's output (n < {step_index}), or pass the "
                f"value literally.")

    if step_input is not None and ("missing" in low and "required" in low):
        return (f"{where}: the tool rejected its inputs ({err[:200]}). "
                f"Add the missing parameter to the step's wiring or to the "
                f"run input, e.g. "
                f"`nm skill run '{skill_name}' --input '{{\"<key>\": ...}}'`.")

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


class RepairTicketStore:
    """Persisted repair tickets."""

    def __init__(self, db: "Database") -> None:
        self.db = db
        db.executescript(REPAIR_TICKETS_DDL)

    def save(self, ticket: RepairTicket) -> RepairTicket:
        self.db.execute(
            "INSERT OR REPLACE INTO skill_repair_tickets "
            "(id, skill_name, version, step_index, tool, inputs, error, "
            "suggested_fix, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (ticket.id, ticket.skill_name, ticket.version,
             ticket.step_index, ticket.tool,
             json.dumps(ticket.inputs, ensure_ascii=False, default=str),
             ticket.error, ticket.suggested_fix, ticket.created_at),
        )
        _log.info("repair ticket %s saved for skill %s (%s)", ticket.id,
                  ticket.skill_name, ticket.error[:80])
        return ticket

    def get(self, ticket_id: str) -> RepairTicket | None:
        row = self.db.query_one(
            "SELECT * FROM skill_repair_tickets WHERE id=?", (ticket_id,))
        return self._from_row(row) if row else None

    def list(self, skill_name: str = "",
             limit: int = 50) -> list[RepairTicket]:
        if skill_name:
            rows = self.db.query(
                "SELECT * FROM skill_repair_tickets WHERE skill_name=? "
                "ORDER BY created_at DESC LIMIT ?", (skill_name, limit))
        else:
            rows = self.db.query(
                "SELECT * FROM skill_repair_tickets ORDER BY created_at DESC "
                "LIMIT ?", (limit,))
        return [self._from_row(r) for r in rows]

    def count(self, skill_name: str = "") -> int:
        if skill_name:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM skill_repair_tickets "
                "WHERE skill_name=?", (skill_name,))
        else:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM skill_repair_tickets")
        return int(row["n"]) if row else 0

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
        )
