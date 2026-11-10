"""The owner ticket board (public-surface fix 5) — ``/runs``' home.

Doc 18's ExecutionTicket (agents/run_ticket.py) and doc 20's healing
ticket (agents/healing_ticket.py) shipped as read-computed projections
with no production reader: the census (public-surface Item 6) named
them shelfware. This module is their consumer — the one place the two
projections are composed into the owner's live-work board, following
the doc-15 P6 inspection pattern (agents/inspection.py + the /cases
board): one render home, board ordinals, NO internal ids printed.

The board is a pure composition of the projections: enumeration comes
from their own coverage functions (operations_detail), labels/status
from their own assemblers. Nothing is persisted here and no organ
state is read around the projections — the ticket doctrine (the
ticket is a projection, never a parallel store) holds end to end.

Liveness ordering: work that is still live sorts above settled work
(stable within each band, in the coverage enumeration order —
contracts, missions, jobs, browser runs, media jobs, mind queue, then
the healing kinds). Every row carries the projection's absent-field
count, and the detail view renders each absent field WITH the
projection's own reason — the honesty design, never papered over.
"""

from __future__ import annotations

import time
from typing import Any

from .healing_ticket import (
    HEALING_FIELDS,
    healing_ticket_coverage,
    healing_ticket_for,
)
from .run_ticket import TICKET_FIELDS, ticket_coverage, ticket_for

__all__ = [
    "board_rows",
    "render_board",
    "render_ticket",
    "render_coverage",
]

#: Status words that mean the work is settled — everything else on a
#: board row is treated as live and sorts into the attention band.
_TERMINAL = {
    "completed", "complete", "done", "cancelled", "canceled",
    "failed", "error", "succeeded", "success", "closed", "verified",
    "rejected", "expired", "resolved", "retired", "healed",
    "stopped", "finished", "dead", "exhausted", "unhonored",
    "not_found", "refused",
}

#: Keys inside projected detail structures that carry internal ids —
#: never rendered (the doc-15 law: board ordinals only). Values under
#: these keys identify records, not work.
_ID_KEYS = {
    "id", "ref", "lease_id", "receipt_id", "request_id", "job_id",
    "contract_id", "mission_id", "claim_id", "case_id",
    "incident_id", "run_id",
}


def board_rows(db: Any, *, registry: Any = None,
               tool_registry: Any = None, coremind: Any = None,
               journal: Any = None, immune_states: Any = None,
               antibody_stores: Any = None, baseline: Any = None,
               limit: int = 50) -> dict[str, Any]:
    """Enumerate the board: every execution + healing ticket, live
    first. Each row carries its full projected ticket, so detail and
    stop act on exactly what the board showed."""
    run_cov = ticket_coverage(
        db, registry=registry, tool_registry=tool_registry,
        coremind=coremind, limit=limit)
    heal_cov = healing_ticket_coverage(
        journal=journal, immune_states=immune_states,
        antibody_stores=antibody_stores, baseline=baseline,
        limit=limit)
    rows: list[dict[str, Any]] = []
    for entry in run_cov.get("operations_detail") or []:
        ref = entry.get("ref")
        if not ref:
            continue
        ticket = ticket_for(ref, db, registry=registry,
                            tool_registry=tool_registry,
                            coremind=coremind)
        if not ticket.get("found", True):
            continue
        rows.append(_row(ref, "run", ticket))
    for entry in heal_cov.get("operations_detail") or []:
        ref = entry.get("ref")
        if not ref:
            continue
        ticket = healing_ticket_for(
            ref, journal=journal, immune_states=immune_states,
            antibody_stores=antibody_stores, baseline=baseline)
        if not ticket.get("found", True):
            continue
        rows.append(_row(ref, "heal", ticket))
    rows.sort(key=lambda r: 0 if r["live"] else 1)
    for index, row in enumerate(rows, 1):
        row["ordinal"] = index
    return {"rows": rows, "run": run_cov, "healing": heal_cov}


def _row(ref: str, family: str, ticket: dict[str, Any]) -> dict[str, Any]:
    status = str(ticket.get("status") or "")
    return {
        "ref": ref,
        "family": family,
        "kind": str(ticket.get("kind") or "?"),
        "label": str(ticket.get("label") or ref.split(":", 1)[0]),
        "status": status,
        "absent": len(ticket.get("absent") or {}),
        "live": status.lower() not in _TERMINAL,
        "ticket": ticket,
    }


def render_board(board: dict[str, Any]) -> str:
    """The board: header counts, then one numbered line per ticket."""
    rows = board["rows"]
    run_cov = board.get("run") or {}
    heal_cov = board.get("healing") or {}
    live = sum(1 for r in rows if r["live"])
    lines = [
        f"🎫 runs — {len(rows)} tickets ({live} live, "
        f"{len(rows) - live} settled): "
        f"{run_cov.get('operations', 0)} executions · "
        f"{heal_cov.get('operations', 0)} healing",
    ]
    leases_held = int(run_cov.get("leases_held") or 0)
    if leases_held:
        lines.append(
            f"leases held right now: {leases_held} across "
            f"{int(run_cov.get('lease_holders') or 0)} ticket(s) — "
            "each ticket's detail names its resources.")
    if not rows:
        lines.append(
            "nothing on the board — no delegations, missions, jobs, "
            "browser runs, media jobs or open incidents.")
    for row in rows:
        tag = "run" if row["family"] == "run" else "heal"
        line = (f"{row['ordinal']}. [{tag} · {row['kind']}] "
                f"{row['label']} — {row['status'] or 'status unknown'}")
        if row["absent"]:
            line += f" · {row['absent']} field(s) not recorded"
        lines.append(line)
    for note in heal_cov.get("unread") or []:
        lines.append(f"not read: {note}")
    if rows:
        lines.append("/runs <n> — one ticket in full · "
                     "/runs stop <n> [reason] — stop it (receipted) · "
                     "/runs coverage — the field-coverage rollup")
    return "\n".join(lines)


def render_coverage(board: dict[str, Any]) -> str:
    """The projections' own coverage rollups, verbatim."""
    lines = ["🎫 ticket coverage — what the records can and cannot say:"]
    for title, cov in (("executions", board.get("run") or {}),
                       ("healing", board.get("healing") or {})):
        lines.append(f"— {title} —")
        lines.extend(str(s) for s in cov.get("summary") or [])
    return "\n".join(lines)


def render_ticket(row: dict[str, Any]) -> str:
    """One ticket in full: every projected field, present or
    absent-with-its-reason. No internal ids (see _ID_KEYS)."""
    ticket = row["ticket"]
    fields = TICKET_FIELDS if row["family"] == "run" else HEALING_FIELDS
    lines = [
        f"🎫 {row['ordinal']}. {row['label']}",
        f"{row['kind']} ticket · status: {row['status'] or 'unknown'}",
    ]
    absent = ticket.get("absent") or {}
    for name in fields:
        value = ticket.get(name)
        if value is None:
            reason = absent.get(name) or "not recorded"
            lines.append(f"  {name} — not recorded: {reason}")
        else:
            lines.append(f"  {name} — {_fmt(value)}")
    detail = ticket.get("detail") or {}
    if detail:
        lines.append(f"  detail — {_fmt(detail)}")
    sources = ticket.get("sources") or {}
    if sources:
        lines.append("  read from: " + ", ".join(
            f"{k} ← {v}" for k, v in sorted(sources.items())))
    if row["family"] == "run" and row["live"]:
        cancel = ticket.get("cancel") or {}
        if cancel.get("available"):
            lines.append(
                f"stop it here: /runs stop {row['ordinal']} — the "
                "cancel lands natively and a receipt is minted.")
    return "\n".join(lines)


def _fmt(value: Any, *, depth: int = 0) -> str:
    """Compact one projected value for chat. Dict keys carrying
    internal ids are dropped (see _ID_KEYS); epoch ``*_at`` values
    render as clock times; depth is capped so a nested projection
    can never flood the chat."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:297] + "…"
    if depth >= 3:
        return "…"
    if isinstance(value, (list, tuple)):
        parts = [_fmt(v, depth=depth + 1) for v in value]
        text = "; ".join(p for p in parts if p and p != "—")
        return text if len(text) <= 400 else text[:397] + "…"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if key in _ID_KEYS or item is None:
                continue
            if (key.endswith("_at") or key == "at") \
                    and isinstance(item, (int, float)) \
                    and item > 100_000_000:
                item = time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(float(item)))
            parts.append(f"{key}: {_fmt(item, depth=depth + 1)}")
        text = "; ".join(parts)
        return text if len(text) <= 400 else text[:397] + "…"
    return str(value)
