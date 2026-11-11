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
    "health_state",
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
    "not_found", "refused", "promoted",
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
               grant_registry: Any = None,
               limit: int = 50) -> dict[str, Any]:
    """Enumerate the board: every execution + healing ticket, live
    first. Each row carries its full projected ticket, so detail and
    stop act on exactly what the board showed."""
    if grant_registry is None:
        # Doc 20 P7: the board is the production reader of the
        # grant registry (a disarm ticket's evidence cites the
        # registry's record). Fail-open: an unreadable registry
        # degrades to the pre-P7 projection, never a broken board.
        try:
            from ..core.grant_registry import default_grant_registry

            grant_registry = default_grant_registry()
        except Exception:  # noqa: BLE001 - reader degrades, board lives
            grant_registry = None
    run_cov = ticket_coverage(
        db, registry=registry, tool_registry=tool_registry,
        coremind=coremind, limit=limit)
    heal_cov = healing_ticket_coverage(
        journal=journal, immune_states=immune_states,
        antibody_stores=antibody_stores, baseline=baseline,
        grant_registry=grant_registry,
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
            antibody_stores=antibody_stores, baseline=baseline,
            grant_registry=grant_registry)
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


def health_state(board: dict[str, Any]) -> dict[str, Any]:
    """The five-state health of the whole board (doc 20 §7 + §9).

    One word — ``healthy`` / ``degraded`` / ``contained`` /
    ``recovering`` / ``unknown`` — derived ONLY from evidence the
    projections themselves read: fever and drift on the immune-state
    tickets, incident standings and their recovery records, open
    case files, and the coverage ``unread`` lists. The derivation
    never upgrades past its evidence: trouble that was read outranks
    sources that could not be read, but a clean bill requires every
    incident-bearing source to have been read — an unread journal
    reads ``unknown``, never ``healthy`` (never fake green).

    Precedence: contained (a fever is active or a case file is
    open — the organism is actively holding something) → degraded
    (an open incident with no recovery attempted, or drift at
    watch/act) → recovering (open incidents remain, but every one
    has a recovery record, or a recovery is on the board
    unverified) → unknown (no trouble read, but a source was not)
    → healthy (everything read, nothing open).
    """
    rows = board.get("rows") or []
    heal_cov = board.get("healing") or {}
    run_cov = board.get("run") or {}
    unread = [str(u) for u in
              (heal_cov.get("unread") or []) + (run_cov.get("unread")
                                                or [])]
    fever = False
    drift_words: list[str] = []
    open_cases = 0
    open_incidents = 0
    open_without_recovery = 0
    live_recoveries = 0
    undetermined = 0
    for row in rows:
        ticket = row.get("ticket") or {}
        lifecycle = ticket.get("lifecycle") or {}
        kind = row.get("kind") or ""
        if kind == "immune_state":
            if lifecycle.get("fever_active"):
                fever = True
            verdict = str(lifecycle.get("drift_verdict") or "")
            if verdict in ("watch", "act"):
                drift_words.append(verdict)
        elif kind == "immune_case":
            if str(ticket.get("status") or "") == "open":
                open_cases += 1
        elif kind == "incident":
            standing = str(lifecycle.get("standing") or "")
            if standing.startswith("undetermined"):
                undetermined += 1
            elif standing == "open":
                open_incidents += 1
                detail = ticket.get("detail") or {}
                recoveries = detail.get("recoveries")
                if recoveries is None:
                    has_recovery = any(
                        str(r).startswith("heal:")
                        for r in ticket.get("related_refs") or [])
                else:
                    has_recovery = bool(recoveries)
                if not has_recovery:
                    open_without_recovery += 1
        elif kind == "healing_run":
            if row.get("live"):
                live_recoveries += 1
    reasons: list[str] = []
    if fever or open_cases:
        if fever:
            reasons.append("a fever is active — the immune system "
                           "is throttling its own responses while "
                           "it contains this")
        if open_cases:
            reasons.append(
                f"{open_cases} case file(s) open — evidence is "
                "being held for review")
        if open_incidents:
            reasons.append(
                f"{open_incidents} incident(s) still open")
        return {"state": "contained", "reasons": reasons}
    if open_without_recovery or drift_words:
        if open_without_recovery:
            reasons.append(
                f"{open_without_recovery} open incident(s) with "
                "no recovery attempted yet")
        if drift_words:
            reasons.append(
                "behaviour drift at " +
                "/".join(sorted(set(drift_words))) +
                " — recent behaviour is moving away from baseline")
        return {"state": "degraded", "reasons": reasons}
    if open_incidents or live_recoveries:
        if open_incidents:
            reasons.append(
                f"{open_incidents} incident(s) still open, each "
                "with a recovery on record — work is underway")
        if live_recoveries:
            reasons.append(
                f"{live_recoveries} recovery record(s) not yet "
                "verified")
        return {"state": "recovering", "reasons": reasons}
    if unread or undetermined:
        for name in unread:
            reasons.append(f"not read: {name}")
        if undetermined:
            reasons.append(
                f"{undetermined} incident(s) whose recovery "
                "standing could not be determined from the records "
                "read")
        reasons.append("no trouble was read, but the all-clear "
                       "cannot be asserted over unread sources")
        return {"state": "unknown", "reasons": reasons}
    return {"state": "healthy",
            "reasons": ["every source read — no open incidents, "
                        "no fever, no drift, no case files open"]}


def render_board(board: dict[str, Any]) -> str:
    """The board: header counts, then one numbered line per ticket."""
    rows = board["rows"]
    run_cov = board.get("run") or {}
    heal_cov = board.get("healing") or {}
    live = sum(1 for r in rows if r["live"])
    health = health_state(board)
    lines = [
        f"🎫 runs — {len(rows)} tickets ({live} live, "
        f"{len(rows) - live} settled): "
        f"{run_cov.get('operations', 0)} executions · "
        f"{heal_cov.get('operations', 0)} healing",
        f"health: {health['state']} — "
        + "; ".join(health["reasons"]),
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
