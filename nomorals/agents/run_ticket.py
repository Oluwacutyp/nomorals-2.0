"""Execution tickets — the doc 18 envelope, as a projection (P2).

ONE ticket shape over the shipped execution organs, computed on
read from the organs' own records. This module stores NOTHING: no
table, no file, no cache of record. A ticket minted here is a
*view* — the doc-18 evaluation's doctrine is explicit that a ticket
kept as a parallel store would create a second fencing authority
beside delegation's generation counter and a second outcome
authority beside the outcome steward. So the ticket REFERENCES:
fencing is the contract's ``generation`` copied verbatim, outcomes
stay the steward's, lifecycle stays each organ's own record.

THE HONESTY RULE (binds every field): a field is either copied
from a named source record — cited in the ticket's ``sources`` map
(field → organ) — or it is ABSENT, named in the ``absent`` map with
the reason it cannot be filled from any shipped record. No field
ever receives a plausible default. Leases resolve from the doc
18 P3 lease authority (:mod:`nomorals.core.leases`) whenever a
database handle is supplied — a ticket whose holder holds no
lease carries the field as a cited EMPTY list (present-but-none
is a fact about the store, not an absence); without a db handle
the field stays absent, with the reason naming the unread store.
Contract cancellation resolves since doc 18 P5: delegation has
a ``cancelled`` terminal and the ticket reads it from the
contract record. Cancels that went through the P5 receipt facade
(:mod:`nomorals.core.cancellation`) carry their receipt in the
cancel field's ``receipt`` entry — the latest receipt for the
ref, read with that module's no-DDL readers; the other organs'
native records stay the primary source and the receipt rides
alongside, never instead. Help requests recorded under the P5
help protocol (:mod:`.help_protocol`) ride in ``detail`` when any
stand for the ref. An unknown reference returns a typed
not-found, never an empty ticket.

Reference shapes are the steward's subject keys, adopted not
replaced: ``contract:<id>`` · ``mission:<id>`` · ``job:<id>`` ·
``browser:<run_id>`` · ``render:<job_id>`` — plus ``task:<id>``
resolved against a caller-supplied TaskGraph (task graphs are
in-memory; there is no durable task record to read without one),
and ``mind:<job_id>`` resolved against a caller-supplied CoreMind
instance (doc 18 P4: coremind's job records — including its
admission queue, which queues with a record instead of shedding —
live on the instance and its durable state file; the projection
reads them through the mind's record accessors and constructs
nothing).

The evidence field speaks doc 24's grade vocabulary
(:mod:`nomorals.core.proof_grade`): a contract's
``required_evidence`` kinds ride verbatim, each kind's grade comes
from PG1's mapping table (``None`` where PG1 records no mapping,
with PG1's reason), and ``required_grade`` is the strongest mapped
grade — completion requires every kind satisfied, so the claim's
evidence must reach the strongest kind's grade. This module
defines no grade of its own; the adoption rule recorded in
proof_grade.py is that there is exactly one scale, in core.

SHADOW (P2): no producer is required to mint tickets, no gate
consults them, and no existing surface changes its output. The
deliverable is the projection itself plus :func:`ticket_coverage`
— the evaluation §3 matrix made queryable: over the live stores
it counts operations per source organ and names exactly which
ticket fields are absent where.

Layer: agents (L5). Reads: delegation + missions (L5 peers, via
their stores), core.jobs snapshots, core.proof_grade (L1),
core.leases live-lease reads (L1, via its module-level readers —
no store is constructed, no DDL runs), core.cancellation receipt
reads (L1, same reader discipline), the help protocol's request
reads (L5 peer, same discipline), and the browser/media
durable record tables directly by SQL (the outcome steward's
precedent — observation constructs no manager and writes
nothing).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.cancellation import latest_receipt_for_ref
from ..core.jobs import JobRegistry, shared_registry
from ..core.leases import live_leases_for_holder
from ..core.proof_grade import (
    DELEGATION_KIND_MAPPINGS,
    DELEGATION_UNMAPPED_REASON,
    ProofGrade,
    grade_for_delegation_kind,
)
from ..missions.mission import MissionStore, mission_liveness
from .delegation import DelegationStore
from .help_protocol import help_requests_for_ref

__all__ = [
    "TICKET_FIELDS",
    "ExecutionTicket",
    "ticket_for",
    "ticket_coverage",
]

#: The envelope's fields, in ticket order. Every field on every
#: ticket is either in ``sources`` (populated, cited) or in
#: ``absent`` (with the reason) — never neither, never both.
TICKET_FIELDS = (
    "purpose",
    "actors",
    "owner_chat",
    "capabilities",
    "fencing",
    "evidence",
    "acceptance",
    "budget",
    "liveness",
    "cancel",
    "leases",
)

#: Reference prefixes this projection reads, mapped to ticket kinds.
_REF_KINDS = {
    "contract": "contract",
    "mission": "mission",
    "job": "job",
    "browser": "browser_run",
    "render": "media_job",
    "media": "media_job",
    "task": "task",
    "mind": "coremind_job",
}

_LEASES_NO_DB = ("leases live in the doc 18 P3 lease authority's "
                 "store (core.leases); no database handle was "
                 "supplied, so the store cannot be read for this "
                 "ticket")


@dataclass
class ExecutionTicket:
    """One operation, projected. ``ref``/``kind``/``label``/
    ``status`` are the envelope's own frame (the label is id-free);
    every TICKET_FIELDS member is populated-and-cited or absent-
    with-reason. Build via :func:`ticket_for`, never by hand."""

    ref: str
    kind: str
    label: str
    status: str
    purpose: Any = None
    actors: Any = None
    owner_chat: Any = None
    capabilities: Any = None
    fencing: Any = None
    evidence: Any = None
    acceptance: Any = None
    budget: Any = None
    liveness: Any = None
    cancel: Any = None
    leases: Any = None
    sources: dict[str, str] = field(default_factory=dict)
    absent: dict[str, str] = field(default_factory=dict)
    related_refs: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    read_at: float = 0.0

    def fill(self, name: str, value: Any, source: str) -> None:
        setattr(self, name, value)
        self.sources[name] = source

    def mark_absent(self, name: str, reason: str) -> None:
        self.absent[name] = reason

    def absent_fields(self) -> list[str]:
        return [name for name in TICKET_FIELDS if name in self.absent]

    def complete(self) -> bool:
        """A ticket is complete only when EVERY field resolved to
        a source record. (Leases resolve since P3 wherever a db
        handle is supplied; other fields — owner_chat on every
        organ, acceptance/cancel on several — still keep tickets
        incomplete. The count is reported, not rounded up.)"""
        return not self.absent

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "found": True,
            "ref": self.ref,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "sources": dict(self.sources),
            "absent": dict(self.absent),
            "related_refs": list(self.related_refs),
            "detail": dict(self.detail),
            "read_at": self.read_at,
        }
        for name in TICKET_FIELDS:
            out[name] = getattr(self, name)
        return out


def _apply_leases(ticket: ExecutionTicket, db: Any,
                  now: float) -> None:
    """Resolve the leases field against the doc 18 P3 lease
    authority, keyed by the ticket's own ref (leases name their
    holder in P2's ref shapes). With a db handle the field is
    ALWAYS cited — the live leases the holder holds, or an empty
    list when it holds none (present-but-none is a fact about the
    store). Without one it stays absent, with the reason."""
    if db is None:
        ticket.mark_absent("leases", _LEASES_NO_DB)
        return
    held = live_leases_for_holder(db, ticket.ref, now=now)
    ticket.fill("leases", [
        {"lease_id": lease["lease_id"],
         "resource": lease["resource"],
         "generation": lease["generation"],
         "acquired_at": lease["acquired_at"],
         "expires_at": lease["expires_at"]}
        for lease in held], "core.leases")


def _apply_cancel_receipt(ticket: ExecutionTicket, db: Any) -> None:
    """Ride the latest doc 18 P5 cancellation receipt for this
    ticket's ref inside its cancel field, when one exists. The
    organ's own record stays the field's primary source (the
    receipt was minted from its answer); the receipt adds the
    uniform state, who asked, and when. Per-ref detail — this
    runs in :func:`ticket_for` only, never in coverage. Receipt
    ids are structure keys here, never labels."""
    if db is None:
        return
    receipt = latest_receipt_for_ref(db, ticket.ref)
    if receipt is None:
        return
    cancel = ticket.cancel
    if not isinstance(cancel, dict):
        return
    cancel["receipt"] = {
        "receipt_id": receipt["receipt_id"],
        "state": receipt["receipt_state"],
        "organ": receipt["organ"],
        "at": receipt["at"],
        "requested_by": receipt["requested_by"],
        "native_state": receipt["native_state"],
        "detail": receipt["detail"],
    }


def _apply_help(ticket: ExecutionTicket, db: Any) -> None:
    """Surface the ticket's help requests (doc 18 P5's help
    protocol) in ``detail`` when any stand for the ref — open and
    answered/withdrawn alike, so a reader sees both the ask and
    what landed on it. Per-ref detail — :func:`ticket_for` only."""
    if db is None:
        return
    rows = help_requests_for_ref(db, ticket.ref)
    if rows:
        ticket.detail["help_requests"] = [
            {"request_id": row["id"], "kind": row["kind"],
             "state": row["state"], "requester": row["requester"],
             "helper": row["helper"], "detail": row["detail"],
             "created_at": row["created_at"],
             "answer": row["answer"],
             "answered_by": row["answered_by"],
             "answered_at": row["answered_at"]}
            for row in rows]


def _not_found(ref: str, reason: str) -> dict[str, Any]:
    """The typed answer for a reference no source record backs —
    distinct from a ticket, and honest about why."""
    return {"found": False, "ref": ref, "reason": reason}


def _age(now: float, ts: Any) -> float | None:
    try:
        stamp = float(ts or 0)
    except (TypeError, ValueError):
        return None
    if stamp <= 0:
        return None
    return round(max(0.0, now - stamp), 1)


# ── shared field builders ─────────────────────────────────────────


def _no_owner_chat(ticket: ExecutionTicket, organ: str) -> None:
    ticket.mark_absent(
        "owner_chat",
        f"the {organ} record carries no owner/chat attribution "
        "field — attribution is never guessed from neighbouring "
        "records")


def _evidence_from_kinds(kinds: list[str]) -> dict[str, Any]:
    """The evidence requirement for a set of delegation
    ``required_evidence`` kinds, verbatim, with PG1's grade reading
    per kind. ``required_grade`` is the strongest MAPPED grade (a
    claim's completion evidence must reach the strongest kind it
    satisfies); kinds PG1 does not map carry ``None`` + the reason.
    """
    grade_by_kind: dict[str, str | None] = {}
    unmapped: dict[str, str] = {}
    strongest: ProofGrade | None = None
    for kind in kinds:
        grade = grade_for_delegation_kind(kind)
        if grade is not None:
            grade_by_kind[kind] = grade.name
            if strongest is None or grade > strongest:
                strongest = grade
        else:
            grade_by_kind[kind] = None
            row = DELEGATION_KIND_MAPPINGS.get(kind)
            unmapped[kind] = (row.reason if row is not None
                              else DELEGATION_UNMAPPED_REASON)
    return {
        "kinds": list(kinds),
        "grade_by_kind": grade_by_kind,
        "unmapped": unmapped,
        "required_grade": strongest.name if strongest else None,
    }


def _cancel_from_state(state: str, requested_at: Any,
                       detail: str) -> dict[str, Any]:
    """The uniform cancel shape: the source's own state word
    (``none`` when its record carries the empty/no-request marker),
    when the request landed, and the bound on what the source can
    prove. A requested flag is not a receipt — receipts are minted
    by the doc 18 P5 facade (:mod:`nomorals.core.cancellation`),
    and when one exists for the ref it rides in the dict's
    ``receipt`` entry via :func:`_apply_cancel_receipt`."""
    try:
        asked = float(requested_at or 0)
    except (TypeError, ValueError):
        asked = 0.0
    return {
        "state": state if state else "none",
        "requested_at": asked if asked > 0 else None,
        "detail": detail,
    }


# ── per-organ assemblers ──────────────────────────────────────────


def _contract_ticket(contract: Any, now: float) -> ExecutionTicket:
    ticket = ExecutionTicket(
        ref=f"contract:{contract.id}", kind="contract",
        label=contract.title or contract.goal[:60],
        status=contract.status, read_at=now)
    ticket.fill("purpose", contract.goal, "delegation")
    actors = {role: contract.role_holder(role)
              for role in ("worker", "witness", "skeptic", "verifier",
                           "integration_owner", "recovery_supervisor",
                           "arbiter")
              if contract.role_holder(role)}
    ticket.fill("actors", actors, "delegation")
    _no_owner_chat(ticket, "contract")
    ticket.fill("capabilities",
                {"specialist_class": contract.specialist_class,
                 "scope": list(contract.scope)},
                "delegation")
    ticket.fill("fencing",
                {"generation": contract.generation,
                 "base_revision": contract.base_revision},
                "delegation")
    ticket.fill("evidence",
                _evidence_from_kinds(list(contract.required_evidence)),
                "delegation")
    ticket.mark_absent(
        "acceptance",
        "a contract's completion bar is delegation's own sign-off "
        "and skeptic gate over the claim ledger — no TaskGraph "
        "acceptance predicate is recorded on the contract")
    ticket.fill("budget",
                {"step_budget": contract.step_budget,
                 "steps_used": contract.steps_used,
                 "deadline": contract.deadline},
                "delegation")
    ticket.fill("liveness",
                {"status": contract.status,
                 "created_at": contract.created_at,
                 "updated_at": contract.updated_at,
                 "age_s": _age(now, contract.updated_at)},
                "delegation")
    if contract.status == "cancelled":
        ticket.fill(
            "cancel",
            _cancel_from_state(
                "cancelled", contract.completed_at,
                "the contract stands at delegation's cancelled "
                "terminal (doc 18 P5); the cancellation claim on "
                "its ledger names the claims the cancel left "
                "unresolved — the ledger is append-only and "
                "terminal transitions do not rewrite claims"),
            "delegation")
    else:
        ticket.fill(
            "cancel",
            _cancel_from_state(
                "", None,
                "no cancellation is recorded on this contract; "
                "a cancel lands in delegation as the cancelled "
                "terminal (doc 18 P5)"),
            "delegation")
    if contract.mission_id:
        ticket.related_refs.append(f"mission:{contract.mission_id}")
    ticket.detail = {
        "deliverable": contract.deliverable,
        "consequential": bool(contract.consequential),
        "completed_at": contract.completed_at,
        "refusal": contract.refusal,
    }
    return ticket


def _mission_ticket(store: MissionStore, mission: Any,
                    now: float) -> ExecutionTicket:
    ticket = ExecutionTicket(
        ref=f"mission:{mission.id}", kind="mission",
        label=mission.name or mission.goal[:60],
        status=mission.status, read_at=now)
    ticket.fill("purpose", mission.goal, "missions")
    ticket.mark_absent(
        "actors",
        "the mission record names no actors — its steps bind "
        "specialist classes at the runner, not on the record")
    _no_owner_chat(ticket, "mission")
    ticket.mark_absent(
        "capabilities",
        "the mission record carries no capability set — capability "
        "scopes live on its delegation contracts and the tool "
        "registry, not on the mission row")
    contracts = store_contracts_for(store, mission.id)
    for contract in contracts:
        ticket.related_refs.append(f"contract:{contract.id}")
    if contracts:
        kinds: list[str] = []
        for contract in contracts:
            for kind in contract.required_evidence:
                if kind not in kinds:
                    kinds.append(kind)
        ticket.fill("evidence", _evidence_from_kinds(kinds),
                    "delegation")
        ticket.mark_absent(
            "fencing",
            "fencing generations live on the mission's delegation "
            f"contracts ({len(contracts)} attached) — each contract "
            "carries its own generation, so no single generation "
            "can be stated for the mission; see related_refs")
    else:
        ticket.mark_absent(
            "evidence",
            "no delegation contract is attached to this mission "
            "and the mission record names no evidence requirement "
            "of its own")
        ticket.mark_absent(
            "fencing",
            "the mission record carries no fencing generation — "
            "missions hold durable idempotency claims, and fencing "
            "generations live on delegation contracts (none is "
            "attached to this mission)")
    acceptance = mission.acceptance
    if acceptance:
        ticket.fill("acceptance", acceptance, "missions")
    else:
        ticket.mark_absent(
            "acceptance",
            "the mission carries no persisted acceptance record — "
            "it completes without verification (the record's own "
            "acceptance property reads None)")
    ticket.fill("budget",
                {"budget_wall": mission.budget_wall,
                 "budget_tokens": mission.budget_tokens,
                 "spent_wall": mission.spent_wall,
                 "spent_tokens": mission.spent_tokens},
                "missions")
    live = mission_liveness(mission)
    ticket.fill("liveness",
                {"status": mission.status,
                 "created_at": mission.created_at,
                 "updated_at": mission.updated_at,
                 "age_s": _age(now, mission.updated_at),
                 "alive": live.get("alive"),
                 "heartbeat_age_s": live.get("heartbeat_age_s"),
                 "process_alive": live.get("process_alive")},
                "missions")
    if mission.status == "cancelled":
        ticket.fill(
            "cancel",
            _cancel_from_state("cancelled", mission.finished_at,
                               "the mission record's durable "
                               "CANCELLED status + optional note "
                               "is the organ's record; when the "
                               "cancel went through the doc 18 P5 "
                               "facade its receipt rides in this "
                               "field's receipt entry"),
            "missions")
    else:
        ticket.fill(
            "cancel",
            _cancel_from_state("", None,
                               "no cancellation is recorded on "
                               "this mission; missions cancel as a "
                               "durable status flip when they do"),
            "missions")
    if mission.parent_id:
        ticket.related_refs.append(f"mission:{mission.parent_id}")
    ticket.detail = {
        "iterations": mission.iterations,
        "success": mission.success,
        "priority": mission.priority,
        "tags": list(mission.tags),
        "finished_at": mission.finished_at,
    }
    return ticket


def store_contracts_for(store: MissionStore, mission_id: str) -> list:
    """The mission's delegation contracts, read through the
    DelegationStore the steward's contracts adapter uses. A store
    that cannot be read yields no contracts — the caller's ticket
    then states the no-contract absence (coverage over a partial
    read is still honest about the mission itself)."""
    try:
        return DelegationStore(store.db).contracts_for_mission(
            mission_id)
    except Exception:  # noqa: BLE001 - unreadable = none attached
        return []


def _job_ticket(snap: dict[str, Any], tool_registry: Any,
                now: float) -> ExecutionTicket:
    ticket = ExecutionTicket(
        ref=f"job:{snap['id']}", kind="job",
        label=f"{snap.get('tool') or 'job'} job",
        status=snap.get("status", ""), read_at=now)
    ticket.mark_absent(
        "purpose",
        "a registry job names its tool, not a goal — its args are "
        "inputs, and inputs are not a purpose statement")
    ticket.mark_absent(
        "actors",
        "the registry snapshot names no actor for the job")
    _no_owner_chat(ticket, "job")
    tool = str(snap.get("tool") or "")
    spec = None
    if tool_registry is not None and tool:
        try:
            spec = tool_registry.get(tool)
        except Exception:  # noqa: BLE001 - unreadable registry
            spec = None
    if spec is not None:
        annotations = spec.resolved_annotations() \
            if callable(getattr(spec, "resolved_annotations", None)) \
            else dict(getattr(spec, "annotations", {}) or {})
        ticket.fill("capabilities",
                    {"tool": tool,
                     "capability": getattr(spec, "capability", ""),
                     "kind": getattr(spec, "kind", ""),
                     "annotations": annotations,
                     "has_output_schema": bool(
                         getattr(spec, "output_schema", None))},
                    "tools.registry")
    elif tool:
        ticket.fill("capabilities", {"tool": tool}, "core.jobs")
    else:
        ticket.mark_absent(
            "capabilities",
            "the job names no tool, so no capability can be read")
    ticket.mark_absent(
        "fencing",
        "the JobRegistry keeps no fencing generation — a late "
        "result is recorded as a late result, never fenced out")
    ticket.mark_absent(
        "evidence",
        "a registry job names no evidence requirement for its "
        "completion")
    ticket.mark_absent(
        "acceptance",
        "no acceptance predicate is attached to a registry job — "
        "its result is collected, not verified against criteria")
    ticket.mark_absent(
        "budget",
        "the registry bounds a job by the heartbeat dead-man's "
        "switch, not by a budget — no budget is recorded")
    ticket.fill("liveness",
                {"status": snap.get("status"),
                 "last_heartbeat_at": snap.get("last_heartbeat_at"),
                 "heartbeat_age_s": _age(
                     now, snap.get("last_heartbeat_at")),
                 "heartbeat_count": snap.get("heartbeat_count"),
                 "last_progress": snap.get("last_progress"),
                 "late_result": snap.get("late_result"),
                 "outcome_unknown": snap.get("outcome_unknown"),
                 "elapsed_s": snap.get("elapsed_s")},
                "core.jobs")
    ticket.fill(
        "cancel",
        _cancel_from_state(
            "requested" if snap.get("cancel_requested") else "",
            None,
            "the registry's cooperative flag is the organ's whole "
            "record: it cannot say whether the worker honored the "
            "request or what state the effect was in; a cancel "
            "through the doc 18 P5 facade mints the durable "
            "receipt, which rides in this field's receipt entry "
            "when one exists"),
        "core.jobs")
    ticket.detail = {
        "error": snap.get("error"),
        "collected": snap.get("collected"),
        "labels": dict(snap.get("labels") or {}),
        "started_at": snap.get("started_at"),
        "ended_at": snap.get("ended_at"),
    }
    return ticket


def _rows(db: Any, table: str, where: str, params: tuple,
          limit: int = 1) -> list[dict[str, Any]]:
    """Direct table reads, the outcome steward's precedent: a
    missing table reads as no records, and observation never
    constructs a manager (construction starts threads / runs DDL).
    """
    try:
        return [dict(r) for r in db.query(
            f"SELECT * FROM {table} WHERE {where} "
            "ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)))]
    except Exception:  # noqa: BLE001 - no table = no records
        return []


def _browser_ticket(row: dict[str, Any], now: float) -> ExecutionTicket:
    from urllib.parse import urlsplit
    host = ""
    try:
        host = urlsplit(str(row.get("url") or "")).hostname or ""
    except ValueError:
        host = ""
    goal = str(row.get("goal") or "").strip()
    where = host or "browser run"
    label = f"browser run: {where} — {goal[:40]}" if goal \
        else f"browser run: {where}"
    ticket = ExecutionTicket(
        ref=f"browser:{row.get('run_id')}", kind="browser_run",
        label=label, status=str(row.get("status") or ""), read_at=now)
    ticket.fill("purpose", str(row.get("goal") or ""), "browser")
    ticket.mark_absent(
        "actors", "the browser run record names no actor")
    _no_owner_chat(ticket, "browser run")
    ticket.mark_absent(
        "capabilities",
        "a browser run record names no capability set — the "
        "browser's per-domain concurrency slots are counting "
        "limits, not a capability scope")
    ticket.mark_absent(
        "fencing", "browser runs keep no fencing generation")
    ticket.mark_absent(
        "evidence",
        "the run record names no evidence requirement for its "
        "completion")
    ticket.mark_absent(
        "acceptance",
        "no acceptance predicate is recorded for a browser run — "
        "its wall outcome is an observation, not a verification")
    ticket.mark_absent(
        "budget",
        "a browser run is bounded by wall-clock timeouts only — "
        "no budget is recorded")
    ticket.fill("liveness",
                {"status": str(row.get("status") or ""),
                 "created_at": row.get("created_at"),
                 "started_at": row.get("started_at"),
                 "finished_at": row.get("finished_at"),
                 "updated_at": row.get("updated_at"),
                 "age_s": _age(now, row.get("updated_at")),
                 "steps": row.get("steps")},
                "browser")
    ticket.fill(
        "cancel",
        _cancel_from_state(
            str(row.get("cancel_state") or ""),
            row.get("cancel_requested_at"),
            "the run record's own typed cancel state (doc 18 P1): "
            "requested is not confirmed, and a run that ended "
            "without honoring a request reads unhonored"),
        "browser")
    ticket.detail = {
        "url": str(row.get("url") or ""),
        "final_url": str(row.get("final_url") or ""),
        "stopped_reason": str(row.get("stopped_reason") or ""),
        "wall_outcome": str(row.get("wall_outcome") or ""),
        "error": str(row.get("error") or ""),
    }
    return ticket


def _media_ticket(row: dict[str, Any], now: float) -> ExecutionTicket:
    status = str(row.get("status") or "")
    label = str(row.get("label") or "media job").strip() or "media job"
    ticket = ExecutionTicket(
        ref=f"render:{row.get('job_id')}", kind="media_job",
        label=f"media job: {label[:40]}", status=status, read_at=now)
    ticket.mark_absent(
        "purpose",
        "the job record's label names the job; no goal statement "
        "is recorded for a media job")
    ticket.mark_absent(
        "actors", "the media job record names no actor")
    _no_owner_chat(ticket, "media job")
    ticket.mark_absent(
        "capabilities",
        "a media job record names its kind and backend, not a "
        "capability scope — kind/backend ride in detail verbatim")
    ticket.mark_absent(
        "fencing", "media jobs keep no fencing generation")
    ticket.mark_absent(
        "evidence",
        "the job record names no evidence requirement for its "
        "completion")
    ticket.mark_absent(
        "acceptance",
        "no acceptance predicate is recorded for a media job")
    ticket.mark_absent(
        "budget",
        "a media job is bounded by a wait timeout only — no "
        "budget is recorded")
    ticket.fill("liveness",
                {"status": status,
                 "created_at": row.get("created_at"),
                 "started_at": row.get("started_at"),
                 "finished_at": row.get("finished_at"),
                 "updated_at": row.get("updated_at"),
                 "age_s": _age(now, row.get("updated_at"))},
                "render")
    # The record's cancel evidence is its terminal status plus the
    # request timestamp (doc 18 P1): a job standing at cancelled
    # confirms its cancel — queued cancels land there directly and
    # mid-run cancels land there via the progress callback — while
    # a request timestamp on a non-cancelled job is a request the
    # record cannot yet confirm.
    if status == "cancelled":
        cancel_state = "confirmed"
        cancel_detail = ("the job stands at cancelled: the record "
                         "confirms the cancel landed (queued cancels "
                         "land here directly; mid-run cancels land "
                         "here via the job's progress callback)")
    elif float(row.get("cancel_requested_at") or 0) > 0:
        cancel_state = "requested"
        cancel_detail = ("a cancel request is timestamped on the "
                         "record but the job has not landed at "
                         "cancelled — requested, not confirmed")
    else:
        cancel_state = ""
        cancel_detail = "no cancellation is recorded on this job"
    ticket.fill(
        "cancel",
        _cancel_from_state(cancel_state,
                           row.get("cancel_requested_at"),
                           cancel_detail),
        "render")
    ticket.detail = {
        "kind": str(row.get("kind") or ""),
        "backend": str(row.get("backend") or ""),
        "input_ref": str(row.get("input_ref") or ""),
        "output_ref": str(row.get("output_ref") or ""),
        "error": str(row.get("error") or ""),
    }
    return ticket


def _task_ticket(task: Any, graph: Any, now: float) -> ExecutionTicket:
    state = getattr(task.state, "value", str(task.state))
    ticket = ExecutionTicket(
        ref=f"task:{task.id}", kind="task",
        label=str(task.name or "task"), status=str(state), read_at=now)
    ticket.mark_absent(
        "purpose",
        "a graph task's name labels it; the task records no goal "
        "statement of its own (its graph/mission carries the goal)")
    ticket.mark_absent(
        "actors",
        "the task names a role, not an actor — no holder is "
        "recorded on the task")
    _no_owner_chat(ticket, "task")
    if task.role:
        ticket.fill("capabilities", {"role": task.role}, "core.tasks")
    else:
        ticket.mark_absent(
            "capabilities",
            "the task declares no role and tasks carry no "
            "capability set — registry capabilities gate the tools "
            "a task calls, not the task record")
    ticket.mark_absent(
        "fencing",
        "the task's idempotency key dedupes re-runs; it is not a "
        "fencing generation and fences no writer")
    ticket.mark_absent(
        "evidence",
        "the task names no evidence requirement — its acceptance "
        "criteria (where defined) are predicates over its result, "
        "not an evidence-kind requirement")
    criteria = [c.to_dict() for c in (task.acceptance or [])]
    if criteria:
        results = None
        result = task.result
        if result is not None and getattr(result, "acceptance_results",
                                          None):
            results = [r.to_dict() for r in result.acceptance_results]
        ticket.fill("acceptance",
                    {"criteria": criteria, "results": results},
                    "core.tasks")
    else:
        ticket.mark_absent(
            "acceptance",
            "the task defines no acceptance criteria — under the "
            "graph's own semantics, running to completion is the "
            "only bar, and the empty criteria list says so")
    ticket.mark_absent(
        "budget",
        "the retry budget belongs to the graph, not the task; the "
        "task's own timeout/retries ride in detail, and no budget "
        "record names this task")
    ticket.fill("liveness",
                {"state": str(state),
                 "created_at": task.created_at,
                 "started_at": task.started_at,
                 "finished_at": task.finished_at,
                 "attempts": task.attempts},
                "core.tasks")
    if str(state) == "cancelled":
        ticket.fill(
            "cancel",
            _cancel_from_state(
                "cancelled", task.finished_at,
                "graph cancellation propagated to this task and "
                "its reason (where the graph recorded one) is the "
                "task's error text — in-memory only, no durable "
                "receipt (doc 18 P5)"),
            "core.tasks")
    else:
        ticket.fill(
            "cancel",
            _cancel_from_state(
                "", None,
                "no cancellation is recorded on this task; graph "
                "cancellation is in-memory and leaves no receipt"),
            "core.tasks")
    if task.mission_id:
        ticket.related_refs.append(f"mission:{task.mission_id}")
    ticket.detail = {
        "graph": getattr(graph, "name", ""),
        "role": task.role,
        "deps": list(task.deps),
        "timeout": task.timeout,
        "retries": task.retries,
        "idempotency_key": task.idempotency_key,
        "error": task.error,
    }
    return ticket


def _coremind_ticket(row: dict[str, Any], mind: Any,
                     now: float) -> ExecutionTicket:
    status = str(row.get("status") or "")
    kind_name = str(row.get("kind") or "job")
    target = str(row.get("target") or "")
    label = f"coremind job: {kind_name} — {target[:40]}" if target \
        else f"coremind job: {kind_name}"
    ticket = ExecutionTicket(
        ref=f"mind:{row.get('id')}", kind="coremind_job",
        label=label, status=status, read_at=now)
    if target:
        ticket.fill("purpose", target, "coremind")
    else:
        ticket.mark_absent(
            "purpose",
            "the coremind job record's target is empty — its kind "
            "names the route taken, not a goal statement")
    ticket.mark_absent(
        "actors", "the coremind job record names no actor")
    _no_owner_chat(ticket, "coremind job")
    ticket.mark_absent(
        "capabilities",
        "a coremind job record names its kind and route, not a "
        "capability set — kind/route ride in detail verbatim")
    ticket.mark_absent(
        "fencing", "coremind jobs keep no fencing generation")
    ticket.mark_absent(
        "evidence",
        "the coremind job record names no evidence requirement "
        "for its completion")
    ticket.mark_absent(
        "acceptance",
        "no acceptance predicate is recorded for a coremind job — "
        "its outcome is the note the organ returns")
    ticket.mark_absent(
        "budget",
        "a coremind job is bounded by the admission inflight cap "
        "and the queue's overdue bound, not by a per-job budget — "
        "both bounds ride in liveness")
    state = mind.queue_state()
    order = [str(job_id) for job_id in (state.get("order") or [])]
    job_key = str(row.get("id"))
    position = order.index(job_key) + 1 if job_key in order else None
    queued_at = row.get("queued_at")
    wait_s: float | None = None
    overdue = False
    if status == "queued" and queued_at:
        wait_s = round(max(0.0, now - float(queued_at)), 1)
        bound = float(state.get("overdue_after_s") or 0)
        overdue = bool(bound) and wait_s > bound
    ticket.fill("liveness",
                {"status": status,
                 "created_at": row.get("created"),
                 "queued_at": queued_at,
                 "started_at": row.get("started_at"),
                 "finished_at": row.get("finished"),
                 "age_s": _age(now, row.get("created")),
                 "queue_position": position,
                 "queue_depth": state.get("depth"),
                 "wait_s": wait_s,
                 "overdue": overdue,
                 "overdue_after_s": state.get("overdue_after_s")},
                "coremind")
    if status == "cancelled":
        ticket.fill(
            "cancel",
            _cancel_from_state(
                "cancelled", row.get("finished"),
                "cancelled while queued via CoreMind.cancel_job "
                "(doc 18 P4): the job never ran; when the cancel "
                "went through the doc 18 P5 facade its receipt "
                "rides in this field's receipt entry"),
            "coremind")
    elif status == "queued":
        ticket.fill(
            "cancel",
            _cancel_from_state(
                "", None,
                "the job is queued and cancellable via "
                "CoreMind.cancel_job — no cancellation is "
                "recorded yet"),
            "coremind")
    else:
        ticket.fill(
            "cancel",
            _cancel_from_state(
                "", None,
                "no cancellation is recorded on this job; only "
                "queued coremind jobs are cancellable (doc 18 P4)"),
            "coremind")
    ticket.detail = {
        "kind": kind_name,
        "route": str(row.get("route") or ""),
        "target": target,
        "note": str(row.get("note") or ""),
        "queue_reason": str(row.get("queue_reason") or ""),
        "queue_position_when_queued": row.get("queue_position"),
        "waited_s": row.get("waited_s"),
    }
    return ticket


# ── the read API ──────────────────────────────────────────────────


def ticket_for(ref: str, db: Any = None, *,
               registry: JobRegistry | None = None,
               tool_registry: Any = None, graph: Any = None,
               coremind: Any = None,
               now: float | None = None) -> dict[str, Any]:
    """One operation's ticket, by steward-shaped reference, or a
    typed not-found dict (``{"found": False, "ref", "reason"}``).

    ``db`` backs the durable organs (contracts, missions, browser
    runs, media jobs). ``registry`` defaults to the shared
    JobRegistry for ``job:`` refs; ``tool_registry`` (optional)
    resolves a job's tool name to its ToolSpec; ``graph`` (a
    TaskGraph) resolves ``task:`` refs; ``coremind`` (a CoreMind)
    resolves ``mind:`` refs.
    """
    now = time.time() if now is None else now
    prefix, _, ident = str(ref or "").partition(":")
    kind = _REF_KINDS.get(prefix)
    if kind is None or not ident:
        return _not_found(
            str(ref or ""),
            "not a ticket reference — expected one of "
            + ", ".join(f"{p}:<id>" for p in sorted(_REF_KINDS)))
    if kind == "contract":
        if db is None:
            return _not_found(ref, "reading a contract ticket "
                                   "needs a database handle")
        contract = DelegationStore(db).get_contract(ident)
        if contract is None:
            return _not_found(ref, "no delegation contract record "
                                   f"with id {ident!r} exists")
        ticket = _contract_ticket(contract, now)
        _apply_leases(ticket, db, now)
        _apply_cancel_receipt(ticket, db)
        _apply_help(ticket, db)
        return ticket.to_dict()
    if kind == "mission":
        if db is None:
            return _not_found(ref, "reading a mission ticket needs "
                                   "a database handle")
        store = MissionStore(db)
        try:
            mission = store.get(ident)
        except Exception:  # noqa: BLE001 - NotFound or unreadable
            return _not_found(ref, "no mission record with id "
                                   f"{ident!r} exists")
        ticket = _mission_ticket(store, mission, now)
        _apply_leases(ticket, db, now)
        _apply_cancel_receipt(ticket, db)
        _apply_help(ticket, db)
        return ticket.to_dict()
    if kind == "job":
        reg = registry if registry is not None else shared_registry()
        job = reg.poll(ident)
        if job is None:
            return _not_found(ref, "no job with that id is in the "
                                   "registry (terminal jobs are "
                                   "swept after their TTL)")
        ticket = _job_ticket(job.snapshot(), tool_registry, now)
        _apply_leases(ticket, db, now)
        _apply_cancel_receipt(ticket, db)
        _apply_help(ticket, db)
        return ticket.to_dict()
    if kind == "browser_run":
        if db is None:
            return _not_found(ref, "reading a browser-run ticket "
                                   "needs a database handle")
        rows = _rows(db, "browser_run_records", "run_id = ?",
                     (ident,))
        if not rows:
            return _not_found(ref, "no browser_run_records row "
                                   f"with run_id {ident!r} exists")
        ticket = _browser_ticket(rows[0], now)
        _apply_leases(ticket, db, now)
        _apply_cancel_receipt(ticket, db)
        _apply_help(ticket, db)
        return ticket.to_dict()
    if kind == "media_job":
        if db is None:
            return _not_found(ref, "reading a media-job ticket "
                                   "needs a database handle")
        rows = _rows(db, "media_job_records", "job_id = ?",
                     (ident,))
        if not rows:
            return _not_found(ref, "no media_job_records row with "
                                   f"job_id {ident!r} exists")
        ticket = _media_ticket(rows[0], now)
        _apply_leases(ticket, db, now)
        _apply_cancel_receipt(ticket, db)
        _apply_help(ticket, db)
        return ticket.to_dict()
    if kind == "coremind_job":
        if coremind is None:
            return _not_found(ref, "coremind job records live on a "
                                   "CoreMind instance — one must be "
                                   "supplied to read a mind ticket")
        row = coremind.job_record(ident)
        if row is None:
            return _not_found(ref, "no coremind job record with id "
                                   f"{ident!r} exists")
        ticket = _coremind_ticket(row, coremind, now)
        _apply_leases(ticket, db, now)
        _apply_cancel_receipt(ticket, db)
        _apply_help(ticket, db)
        return ticket.to_dict()
    # kind == "task"
    if graph is None:
        return _not_found(ref, "task graphs are in-memory — a "
                               "TaskGraph instance must be supplied "
                               "to read a task ticket")
    task = graph.tasks.get(ident)
    if task is None:
        return _not_found(ref, "no task with that id is in the "
                               "supplied graph")
    ticket = _task_ticket(task, graph, now)
    _apply_leases(ticket, db, now)
    _apply_cancel_receipt(ticket, db)
    _apply_help(ticket, db)
    return ticket.to_dict()


def ticket_coverage(db: Any, *, registry: JobRegistry | None = None,
                    tool_registry: Any = None, limit: int = 50,
                    coremind: Any = None,
                    now: float | None = None) -> dict[str, Any]:
    """The §3 matrix over the live stores: every operation the
    durable organs + the job registry currently hold (up to
    ``limit`` per organ), plus a supplied CoreMind's job records
    when ``coremind`` is passed. Each ticket's absent fields are
    counted per field and per source. ``complete_tickets`` counts tickets with
    NO absent field; the number is reported, never rounded up.
    Lease coverage rides along: with a db handle, every ticket's
    leases resolve against the P3 authority and the result
    reports how many leases are held, by how many operations."""
    now = time.time() if now is None else now
    tickets: list[ExecutionTicket] = []
    if db is not None:
        delegation = DelegationStore(db)
        for contract in delegation.recent_contracts(limit=limit):
            tickets.append(_contract_ticket(contract, now))
        store = MissionStore(db)
        for mission in store.list(limit=limit):
            tickets.append(_mission_ticket(store, mission, now))
        for row in _rows(db, "browser_run_records", "1 = 1", (),
                         limit=limit):
            tickets.append(_browser_ticket(row, now))
        for row in _rows(db, "media_job_records", "1 = 1", (),
                         limit=limit):
            tickets.append(_media_ticket(row, now))
    reg = registry if registry is not None else shared_registry()
    for snap in reg.list():
        tickets.append(_job_ticket(snap, tool_registry, now))
    if coremind is not None:
        for row in coremind.job_records():
            tickets.append(_coremind_ticket(row, coremind, now))
    for ticket in tickets:
        _apply_leases(ticket, db, now)
    leases_held = sum(len(ticket.leases or []) for ticket in tickets)
    lease_holders = sum(1 for ticket in tickets if ticket.leases)
    by_source: dict[str, int] = {}
    absent_totals: dict[str, int] = {}
    absent_by_source: dict[str, dict[str, int]] = {}
    detail_rows: list[dict[str, Any]] = []
    complete = 0
    for ticket in tickets:
        by_source[ticket.kind] = by_source.get(ticket.kind, 0) + 1
        per_source = absent_by_source.setdefault(ticket.kind, {})
        for name in ticket.absent_fields():
            absent_totals[name] = absent_totals.get(name, 0) + 1
            per_source[name] = per_source.get(name, 0) + 1
        if ticket.complete():
            complete += 1
        detail_rows.append({
            "ref": ticket.ref, "kind": ticket.kind,
            "absent_fields": ticket.absent_fields()})
    total = len(tickets)
    first_line = (
        f"{total} operations: contracts-backed "
        f"{by_source.get('contract', 0)} · missions "
        f"{by_source.get('mission', 0)} · registry jobs "
        f"{by_source.get('job', 0)} · browser runs "
        f"{by_source.get('browser_run', 0)} · media jobs "
        f"{by_source.get('media_job', 0)} · tasks "
        f"{by_source.get('task', 0)}")
    if coremind is not None:
        first_line += (f" · coremind jobs "
                       f"{by_source.get('coremind_job', 0)}")
    summary = [first_line]
    for name in TICKET_FIELDS:
        count = absent_totals.get(name, 0)
        if not count:
            continue
        if count == total and total:
            summary.append(f"{name}: absent on all {total}")
        else:
            where = ", ".join(
                f"{kind} {per[name]}"
                for kind, per in sorted(absent_by_source.items())
                if per.get(name))
            summary.append(f"{name}: absent on {count} of {total} "
                           f"({where})")
    if db is not None:
        summary.append(f"leases: {leases_held} held by "
                       f"{lease_holders} of {total} operations")
    summary.append(f"complete tickets (no absent field): {complete}")
    return {
        "operations": total,
        "by_source": by_source,
        "complete_tickets": complete,
        "leases_held": leases_held,
        "lease_holders": lease_holders,
        "fields_absent": absent_totals,
        "absent_by_source": absent_by_source,
        "operations_detail": detail_rows,
        "summary": summary,
    }
