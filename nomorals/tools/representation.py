"""Representation-lookup tool: ``representation_lookup`` — the brain's
answer to "what did you tell X for me?"

One tool, action-dispatched, over
:mod:`nomorals.cognition.representation`; the ledger owns the records,
the scoring and the audit trail — this module is only the surface.

``action=lookup`` answers the question directly
(``counterparty=<who>``, ``action_type=<type>``, ``days=<window>``,
``limit=<n>``): every recorded action whose counterparty or
description names them, newest first, with its outcome and the grant
it ran under. ``action=show`` renders one record by id, with its
counterfactuals and audit trail. ``action=list`` lists recent records.
``action=audit`` reads the append-only audit trail (one record, or
the whole trail).  Reads by the owner are themselves audited
(event "read", actor "owner") — reading the audit trail is the one
exception, so the trail never audits itself.

An unknown counterparty is answered honestly — "nothing on record"
— never invented.  The phrasing parser
:func:`parse_representation_lookup` is shared with the chat NL hook,
so the question and the tool agree on what counts as the question.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..core.policy import Capability

__all__ = [
    "register",
    "parse_representation_lookup",
    "format_lookup",
    "format_record",
    "format_audit_trail",
]

_LOOKUP_RES: tuple[re.Pattern[str], ...] = (
    # "what did you tell Chidi for me?" / "... the group for me?"
    re.compile(r"^what did you tell (.+?) for me[?.!]*$", re.IGNORECASE),
    # "what did you say to Ada?"
    re.compile(r"^what did you say to (.+?)[?.!]*$", re.IGNORECASE),
    # "what have you told Chidi" / "... Chidi for me?"
    re.compile(r"^what have you told (.+?)(?: for me)?[?.!]*$",
               re.IGNORECASE),
    # "what did you tell Chidi?"
    re.compile(r"^what did you tell (.+?)[?.!]*$", re.IGNORECASE),
)


def parse_representation_lookup(text: str) -> str | None:
    """Extract the counterparty from "what did you tell X for me?"
    phrasings.  None when the text is not that question — the hook
    must never steal an unrelated question."""
    raw = (text or "").strip()
    if not raw:
        return None
    for pattern in _LOOKUP_RES:
        match = pattern.match(raw)
        if match is not None:
            name = match.group(1).strip().rstrip("?.!").strip()
            if name.lower().startswith("the "):
                name = name[4:].strip()
            return name or None
    return None


def _when(created_at: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(created_at))


def _feedback_mark(owner_feedback: int | None) -> str:
    return {1: " 👍", -1: " 👎"}.get(owner_feedback, "")


def _authority_text(authority: str | None) -> str:
    return authority or "no grant recorded"


def format_record(rec: Any, *, with_audit: bool = False,
                  audit_rows: list[dict[str, Any]] | None = None) -> str:
    """One record, plain words: what, when, outcome, grant."""
    lines = [
        f"[{rec.action_type}] {rec.description}",
        f"  outcome: {rec.outcome}{_feedback_mark(rec.owner_feedback)}",
        f"  when: {_when(rec.created_at)} · id: {rec.id}",
        f"  grant: {_authority_text(getattr(rec, 'authority', None))}",
    ]
    counterparty = (rec.metadata or {}).get("counterparty")
    if counterparty:
        lines.append(f"  with: {counterparty}")
    if rec.feedback_text:
        lines.append(f"  owner said: {rec.feedback_text[:120]}")
    for cf in (rec.counterfactuals or []):
        lines.append(f"  better: {cf.get('better_action', '')}"
                     + (f" — {cf.get('why', '')}" if cf.get("why") else ""))
    if with_audit and audit_rows is not None:
        lines.append("  audit trail:")
        if not audit_rows:
            lines.append("    (no audit rows)")
        for row in audit_rows:
            lines.append(
                f"    {_when(float(row.get('created_at', 0.0)))}"
                f" {row.get('actor', '')} {row.get('event', '')}"
                + (f" — {row.get('detail', '')}" if row.get("detail") else ""))
    return "\n".join(lines)


def format_lookup(records: list[Any], counterparty: str) -> str:
    """The "what did you tell X for me?" answer, plain words."""
    if not records:
        return (f"Nothing on record for {counterparty} — I haven't"
                " logged an action with them yet.")
    lines = [f"What I told {counterparty} for you"
             f" ({len(records)} action(s)):"]
    for rec in records:
        desc = (rec.description[:100] + "…") \
            if len(rec.description) > 100 else rec.description
        lines.append(
            f"• [{rec.action_type}] {desc} — {rec.outcome}"
            f"{_feedback_mark(rec.owner_feedback)}"
            f" ({_when(rec.created_at)}, via"
            f" {_authority_text(getattr(rec, 'authority', None))})")
    return "\n".join(lines)


def format_audit_trail(rows: list[dict[str, Any]],
                       action_id: str | None = None) -> str:
    """The append-only audit trail, newest first, plain words."""
    head = ("Audit trail"
            + (f" for {action_id}" if action_id else "")
            + (f" ({len(rows)} event(s)):" if rows else " — no events yet."))
    if not rows:
        return head
    lines = [head]
    for row in rows:
        lines.append(
            f"• {_when(float(row.get('created_at', 0.0)))}"
            f" {row.get('actor', '')} {row.get('event', '')}"
            f" {row.get('action_id') or ''}"
            + (f" — {row.get('detail', '')}" if row.get("detail") else ""))
    return "\n".join(lines)


def _ledger(context: Any) -> Any:
    from ..cognition.representation import _ledger_for

    settings = getattr(context, "settings", None) \
        if context is not None else None
    return _ledger_for(settings)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def register(registry: Any) -> None:
    """Attach the ``representation_lookup`` tool."""
    context = registry.context

    @registry.register(
        "representation_lookup",
        description=(
            "What Devon told someone on the owner's behalf, from the "
            "representation ledger — the durable record of her "
            "autonomous actions, with outcomes and the grant each ran "
            "under. action=lookup answers 'what did you tell X for me?' "
            "(counterparty=<who>, action_type=<negotiate|purchase|apply|"
            "message|post>, days=<window, default 30>, limit=<n, default "
            "10>). action=show renders one record by id "
            "(record_id=<id>) with its counterfactuals and audit trail. "
            "action=list lists recent records (limit=<n>, days=<window>). "
            "action=audit reads the append-only audit trail "
            "(record_id=<id> for one record, or the whole trail). An "
            "unknown counterparty or id is answered honestly as nothing "
            "on record, never invented."
        ),
        capability=Capability.MEM_READ,
    )
    def representation_lookup_tool(
        action: str = "lookup",
        counterparty: str = "",
        action_type: str = "",
        days: str = "",
        limit: str = "",
        record_id: str = "",
    ) -> dict[str, Any]:
        verb = (action or "lookup").strip().lower()
        ledger = _ledger(context)
        if verb == "lookup":
            who = (counterparty or "").strip()
            if not who:
                return {"ok": False,
                        "error": "lookup needs counterparty=<who>"}
            recs = ledger.lookup(
                counterparty=who,
                action_type=(action_type or "").strip() or None,
                since_days=_as_float(days, 30.0) if days else 30.0,
                limit=_as_int(limit, 10) if limit else 10,
            )
            ledger.audit(None, "owner", "read",
                         f"lookup counterparty={who}: {len(recs)} records")
            return {
                "ok": True,
                "text": format_lookup(recs, who),
                "records": [{
                    "id": r.id,
                    "action_type": r.action_type,
                    "description": r.description,
                    "outcome": r.outcome,
                    "owner_feedback": r.owner_feedback,
                    "authority": getattr(r, "authority", None),
                    "counterparty": (r.metadata or {}).get("counterparty"),
                    "created_at": r.created_at,
                } for r in recs],
            }
        if verb == "show":
            rid = (record_id or "").strip()
            if not rid:
                return {"ok": False, "error": "show needs record_id=<id>"}
            rec = ledger.get(rid)
            if rec is None:
                return {"ok": True,
                        "text": f"No record with id {rid}.",
                        "record": None}
            trail = ledger.audit_trail(action_id=rid)
            ledger.audit(rid, "owner", "read", "show record")
            return {
                "ok": True,
                "text": format_record(rec, with_audit=True,
                                      audit_rows=trail),
                "record": {
                    "id": rec.id,
                    "action_type": rec.action_type,
                    "description": rec.description,
                    "outcome": rec.outcome,
                    "owner_feedback": rec.owner_feedback,
                    "authority": getattr(rec, "authority", None),
                    "counterparty": (rec.metadata or {}).get("counterparty"),
                    "created_at": rec.created_at,
                    "counterfactuals": rec.counterfactuals,
                },
                "trail": trail,
            }
        if verb == "list":
            recs = ledger.recent(
                limit=_as_int(limit, 10) if limit else 10,
                since_days=_as_float(days, 30.0) if days else 30.0,
            )
            ledger.audit(None, "owner", "read",
                         f"list: {len(recs)} records")
            lines = ["Recent actions on your behalf:"]
            if not recs:
                lines.append("No autonomous actions on record yet.")
            for r in recs:
                lines.append(
                    f"• [{r.action_type}] {r.description[:90]} —"
                    f" {r.outcome}{_feedback_mark(r.owner_feedback)}"
                    f" ({_when(r.created_at)}, via"
                    f" {_authority_text(getattr(r, 'authority', None))},"
                    f" id {r.id})")
            return {"ok": True, "text": "\n".join(lines),
                    "records": [{"id": r.id, "action_type": r.action_type,
                                 "description": r.description,
                                 "outcome": r.outcome,
                                 "authority": getattr(r, "authority", None),
                                 "created_at": r.created_at}
                                for r in recs]}
        if verb == "audit":
            rid = (record_id or "").strip() or None
            trail = ledger.audit_trail(
                action_id=rid, limit=_as_int(limit, 50) if limit else 50)
            return {"ok": True,
                    "text": format_audit_trail(trail, rid),
                    "trail": trail}
        return {"ok": False,
                "error": (f"unknown action {verb!r}; use lookup | show"
                          " | list | audit")}
