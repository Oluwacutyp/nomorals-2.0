"""Outcome-reconciliation cases (L5) — the steward's durable record.

Doc 15 (Notifications + Outcome Stewardship, adopted 2026-11-09), P4:
every overdue, stalled, orphaned or abandoned operation gets an
accountable outcome-reconciliation CASE — not an invented success or
failure. This module is the case store; :mod:`outcome_steward` is the
worker that opens and investigates cases. Domain state stays
domain-owned: a case is a projection ABOUT an operation, never a
second state machine for it, and opening or reading a case never
mutates the operation it concerns.

The discipline the store enforces in code, not in convention:

* **Unknown stays unknown.** A case opens with the verdict
  ``outcome_unknown`` (the shared result-contract vocabulary,
  ``core/result_contract.py``). The ONLY way a case reaches
  ``resolved`` is :meth:`OutcomeCaseStore.resolve`, which demands a
  verdict from the resolvable set AND a cited evidence receipt that
  exists on this very case. There is no other code path to a resolved
  state, no default verdict, and no timeout that "probably" closes a
  case as succeeded or failed. A run that cannot establish the outcome
  ends as ``closed-unknown`` with what was checked recorded — a
  legitimate, successful stewardship outcome, not a failure.
* **Append-only evidence.** Receipts (observations, checks, verdicts,
  transitions) are only ever inserted, never updated or deleted, so
  the trail a verdict rests on can always be re-read.
* **Generation fencing.** A case carries a generation; a steward run
  claims the case by CAS-bumping it. Receipts and transitions that
  name a superseded generation are refused
  (:class:`StaleCaseGeneration`), so a stale worker can never move a
  case a newer run owns.

Persistence is runtime-created tables on the shared ``Database``
(the Batch B pattern — ``storage/migrations.py`` is frozen).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "UNKNOWN_VERDICT",
    "RESOLVABLE_VERDICTS",
    "CASE_STATES",
    "TERMINAL_CASE_STATES",
    "OPEN_REASONS",
    "OutcomeCase",
    "CaseReceipt",
    "OutcomeCaseStore",
    "CaseError",
    "StaleCaseGeneration",
    "ack_snapshot",
    "ack_changed",
]

#: The standing verdict of every case until evidence says otherwise —
#: spelled exactly as the shared result contract spells it.
UNKNOWN_VERDICT = "outcome_unknown"

#: Verdicts a case may be resolved TO, with cited evidence. Mirrors
#: ``core/result_contract.ExecutionStatus`` minus ``outcome_unknown``
#: (a standing state, never a resolution) and ``waiting`` (a live
#: state, not an outcome).
RESOLVABLE_VERDICTS = ("succeeded", "verified", "failed", "partial",
                       "stopped", "delivered")

#: Case lifecycle: open → investigating → resolved | closed-unknown.
CASE_STATES = ("open", "investigating", "resolved", "closed-unknown")
TERMINAL_CASE_STATES = ("resolved", "closed-unknown")

#: Why a case may open (doc 15 §7 discovery triggers, P4 scope).
OPEN_REASONS = ("overdue", "stalled", "orphaned", "abandoned",
                "unknown-outcome")

_DDL = """
CREATE TABLE IF NOT EXISTS outcome_cases (
  id TEXT PRIMARY KEY,
  domain TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT '',
  subject TEXT NOT NULL DEFAULT '',
  subject_key TEXT NOT NULL DEFAULT '',
  reason TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'open',
  verdict TEXT NOT NULL DEFAULT 'outcome_unknown',
  evidence_receipt_id TEXT NOT NULL DEFAULT '',
  contract_id TEXT NOT NULL DEFAULT '',
  generation INTEGER NOT NULL DEFAULT 0,
  detail TEXT NOT NULL DEFAULT '{}',
  opened_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0,
  closed_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_outcome_cases_identity
  ON outcome_cases(domain, subject_key, reason);
CREATE INDEX IF NOT EXISTS idx_outcome_cases_state
  ON outcome_cases(state);
CREATE TABLE IF NOT EXISTS outcome_case_receipts (
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT '',
  actor TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  evidence TEXT NOT NULL DEFAULT '{}',
  generation INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_outcome_receipts_case
  ON outcome_case_receipts(case_id, created_at);
"""


class CaseError(Exception):
    """Base for case-store refusals."""


class StaleCaseGeneration(CaseError):
    """Work arrived from a superseded case generation."""


@dataclass
class OutcomeCase:
    id: str
    domain: str = ""
    kind: str = ""
    subject: str = ""
    subject_key: str = ""
    reason: str = ""
    state: str = "open"
    verdict: str = UNKNOWN_VERDICT
    evidence_receipt_id: str = ""
    contract_id: str = ""
    generation: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    opened_at: float = 0.0
    updated_at: float = 0.0
    closed_at: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_CASE_STATES


@dataclass
class CaseReceipt:
    id: str
    case_id: str = ""
    kind: str = ""
    source: str = ""
    actor: str = ""
    note: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    created_at: float = 0.0


def ack_snapshot(case: OutcomeCase) -> dict[str, Any] | None:
    """The owner's acknowledgement snapshot on a case, if any.

    P6: acknowledgement is a recorded act on the case's detail —
    ``{"at", "by", "state", "verdict"}`` captured at ack time.  It is
    NOT a state: the case's own state/verdict only ever move through
    :meth:`OutcomeCaseStore.resolve` / ``close_unknown``.
    """
    ack = (case.detail or {}).get("acknowledged")
    if not isinstance(ack, dict):
        return None
    if "state" not in ack or "verdict" not in ack:
        return None
    return ack


def ack_changed(case: OutcomeCase) -> bool:
    """True when a case was acknowledged and has since materially
    changed (state or verdict differ from the ack snapshot) — the
    condition under which an acknowledged case resurfaces."""
    ack = ack_snapshot(case)
    if ack is None:
        return False
    return (case.state != ack.get("state")
            or case.verdict != ack.get("verdict"))


def _loads(raw: Any) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _row_case(row: Any) -> OutcomeCase:
    return OutcomeCase(
        id=row["id"], domain=row["domain"], kind=row["kind"],
        subject=row["subject"], subject_key=row["subject_key"],
        reason=row["reason"], state=row["state"], verdict=row["verdict"],
        evidence_receipt_id=row["evidence_receipt_id"],
        contract_id=row["contract_id"], generation=int(row["generation"]),
        detail=_loads(row["detail"]), opened_at=float(row["opened_at"]),
        updated_at=float(row["updated_at"]),
        closed_at=float(row["closed_at"]))


def _row_receipt(row: Any) -> CaseReceipt:
    return CaseReceipt(
        id=row["id"], case_id=row["case_id"], kind=row["kind"],
        source=row["source"], actor=row["actor"], note=row["note"],
        evidence=_loads(row["evidence"]),
        generation=int(row["generation"]),
        created_at=float(row["created_at"]))


class OutcomeCaseStore:
    """Cases + append-only receipts over a shared ``Database``."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_DDL)

    # -- opening / dedupe -------------------------------------------------

    def open_case(self, *, domain: str, kind: str, subject: str,
                  subject_key: str, reason: str,
                  detail: dict[str, Any] | None = None,
                  actor: str = "") -> tuple[OutcomeCase, bool]:
        """Open the case for one operation + opening reason.

        Deduplicated by identity ``(domain, subject_key, reason)``:
        one operation gets ONE case per reason, ever. A live case is
        returned as-is; a ``closed-unknown`` case is REOPENED (its
        verdict is still unknown — the question is still open) with a
        receipt; a resolved case is returned untouched, because its
        outcome was established with evidence and re-discovery of the
        same operation does not un-establish it. Returns
        ``(case, created)`` — ``created`` is True only for a genuinely
        new case.
        """
        if reason not in OPEN_REASONS:
            raise CaseError(f"unknown opening reason {reason!r}")
        if not domain or not subject_key:
            raise CaseError("a case names its domain and subject_key")
        existing = self.case_for(domain=domain, subject_key=subject_key,
                                 reason=reason)
        if existing is not None:
            if existing.state == "closed-unknown":
                now = time.time()
                self.db.execute(
                    "UPDATE outcome_cases SET state = 'open',"
                    " closed_at = 0, updated_at = ? WHERE id = ?",
                    (now, existing.id))
                existing.state = "open"
                existing.closed_at = 0.0
                existing.updated_at = now
                self._append(existing, kind="reopened",
                             source="outcome-cases", actor=actor,
                             note="re-discovered while still unresolved; "
                                  "the case reopens rather than "
                                  "duplicating")
            return existing, False
        now = time.time()
        case = OutcomeCase(
            id=f"oc-{uuid.uuid4().hex[:12]}", domain=domain, kind=kind,
            subject=subject, subject_key=subject_key, reason=reason,
            detail=dict(detail or {}), opened_at=now, updated_at=now)
        self.db.execute(
            "INSERT INTO outcome_cases (id, domain, kind, subject,"
            " subject_key, reason, state, verdict, evidence_receipt_id,"
            " contract_id, generation, detail, opened_at, updated_at,"
            " closed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (case.id, case.domain, case.kind, case.subject,
             case.subject_key, case.reason, case.state, case.verdict,
             "", "", 0, json.dumps(case.detail), case.opened_at,
             case.updated_at, 0.0))
        self._append(case, kind="opened", source="outcome-cases",
                     actor=actor,
                     note=f"case opened: {reason}",
                     evidence={"detail": case.detail})
        return case, True

    def case_for(self, *, domain: str, subject_key: str,
                 reason: str) -> OutcomeCase | None:
        row = self.db.query_one(
            "SELECT * FROM outcome_cases WHERE domain = ?"
            " AND subject_key = ? AND reason = ?"
            " ORDER BY opened_at LIMIT 1",
            (domain, subject_key, reason))
        return _row_case(row) if row else None

    def get(self, case_id: str) -> OutcomeCase | None:
        row = self.db.query_one(
            "SELECT * FROM outcome_cases WHERE id = ?", (case_id,))
        return _row_case(row) if row else None

    def _case_or_raise(self, case_id: str) -> OutcomeCase:
        case = self.get(case_id)
        if case is None:
            raise CaseError(f"no case {case_id!r}")
        return case

    # -- receipts (append-only) --------------------------------------------

    def _append(self, case: OutcomeCase, *, kind: str, source: str,
                actor: str = "", note: str = "",
                evidence: dict[str, Any] | None = None,
                generation: int | None = None) -> CaseReceipt:
        receipt = CaseReceipt(
            id=f"ocr-{uuid.uuid4().hex[:12]}", case_id=case.id,
            kind=kind, source=source, actor=actor, note=note,
            evidence=dict(evidence or {}),
            generation=case.generation if generation is None
            else int(generation),
            created_at=time.time())
        self.db.execute(
            "INSERT INTO outcome_case_receipts (id, case_id, kind,"
            " source, actor, note, evidence, generation, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (receipt.id, receipt.case_id, receipt.kind, receipt.source,
             receipt.actor, receipt.note, json.dumps(receipt.evidence),
             receipt.generation, receipt.created_at))
        return receipt

    def add_receipt(self, case_id: str, *, kind: str, source: str,
                    note: str = "", actor: str = "",
                    evidence: dict[str, Any] | None = None,
                    generation: int | None = None) -> CaseReceipt:
        """Append one receipt. A receipt naming a superseded
        generation is refused — stale workers leave no trail on a case
        a newer run owns."""
        case = self._case_or_raise(case_id)
        if generation is not None and int(generation) != case.generation:
            raise StaleCaseGeneration(
                f"case {case_id} is at generation {case.generation}; "
                f"a receipt arrived from generation {generation}")
        return self._append(case, kind=kind, source=source, actor=actor,
                            note=note, evidence=evidence,
                            generation=generation)

    def get_receipt(self, receipt_id: str) -> CaseReceipt | None:
        row = self.db.query_one(
            "SELECT * FROM outcome_case_receipts WHERE id = ?",
            (receipt_id,))
        return _row_receipt(row) if row else None

    def receipts(self, case_id: str, *, limit: int = 500
                 ) -> list[CaseReceipt]:
        rows = self.db.query(
            "SELECT * FROM outcome_case_receipts WHERE case_id = ?"
            " ORDER BY created_at, rowid LIMIT ?",
            (case_id, int(limit)))
        return [_row_receipt(r) for r in rows]

    # -- run ownership (generation fencing) ---------------------------------

    def claim(self, case_id: str, *, actor: str = ""
              ) -> tuple[OutcomeCase, int]:
        """Claim a case for one steward run: the generation bumps by
        CAS and the case moves to ``investigating``. Every receipt or
        transition the run then attempts must name the returned
        generation; anything carrying an older one is fenced.
        Returns ``(case, generation)``."""
        case = self._case_or_raise(case_id)
        if case.terminal:
            raise CaseError(
                f"case {case_id} is {case.state} — terminal cases "
                "cannot be claimed")
        new_generation = case.generation + 1
        cursor = self.db.execute(
            "UPDATE outcome_cases SET generation = ?,"
            " state = 'investigating', updated_at = ?"
            " WHERE id = ? AND generation = ?",
            (new_generation, time.time(), case_id, case.generation))
        if not cursor.rowcount:
            raise StaleCaseGeneration(
                f"case {case_id} moved under the claim — re-read and "
                "retry")
        case.generation = new_generation
        case.state = "investigating"
        self._append(case, kind="run-claimed", source="outcome-cases",
                     actor=actor,
                     note=f"steward run claimed the case at generation "
                          f"{new_generation}",
                     generation=new_generation)
        return case, new_generation

    def note_run(self, case_id: str, *, contract_id: str,
                 generation: int) -> OutcomeCase:
        """Record which delegation contract is investigating."""
        case = self._case_or_raise(case_id)
        if int(generation) != case.generation:
            raise StaleCaseGeneration(
                f"case {case_id} is at generation {case.generation}; "
                f"run note arrived from generation {generation}")
        self.db.execute(
            "UPDATE outcome_cases SET contract_id = ?, updated_at = ?"
            " WHERE id = ?", (contract_id, time.time(), case_id))
        case.contract_id = contract_id
        return case

    # -- transitions (the ONLY two ways a case closes) -----------------------

    def _check_transition(self, case_id: str, generation: int
                          ) -> OutcomeCase:
        case = self._case_or_raise(case_id)
        if case.terminal:
            raise CaseError(
                f"case {case_id} is already {case.state} — terminal "
                "cases do not transition again")
        if int(generation) != case.generation:
            raise StaleCaseGeneration(
                f"case {case_id} is at generation {case.generation}; "
                f"a transition arrived from generation {generation}")
        return case

    def resolve(self, case_id: str, *, verdict: str,
                evidence_receipt_id: str, generation: int,
                actor: str = "", note: str = "") -> OutcomeCase:
        """Resolve a case to an evidence-backed verdict.

        Refuses — by construction, not by policy text — when: the
        verdict is ``outcome_unknown`` or outside the resolvable set
        (unknown is closed via :meth:`close_unknown`, never resolved);
        the generation is stale; or the cited receipt does not exist
        on THIS case as an ``evidence`` receipt. A verdict without
        cited evidence cannot be recorded through any API on this
        store.
        """
        if verdict not in RESOLVABLE_VERDICTS:
            raise CaseError(
                f"verdict {verdict!r} is not resolvable — a case "
                "resolves only to one of "
                f"{', '.join(RESOLVABLE_VERDICTS)} with cited "
                "evidence; outcome_unknown closes via close_unknown")
        case = self._check_transition(case_id, generation)
        receipt = self.get_receipt(evidence_receipt_id)
        if receipt is None or receipt.case_id != case_id \
                or receipt.kind != "evidence":
            raise CaseError(
                f"evidence receipt {evidence_receipt_id!r} is not an "
                f"evidence receipt on case {case_id} — a verdict "
                "without cited evidence is refused")
        now = time.time()
        cursor = self.db.execute(
            "UPDATE outcome_cases SET state = 'resolved', verdict = ?,"
            " evidence_receipt_id = ?, closed_at = ?, updated_at = ?"
            " WHERE id = ? AND generation = ?",
            (verdict, evidence_receipt_id, now, now, case_id,
             case.generation))
        if not cursor.rowcount:
            raise StaleCaseGeneration(
                f"case {case_id} moved under the resolution")
        case.state = "resolved"
        case.verdict = verdict
        case.evidence_receipt_id = evidence_receipt_id
        case.closed_at = now
        self._append(case, kind="resolution", source="outcome-cases",
                     actor=actor,
                     note=note or f"resolved: {verdict}",
                     evidence={"verdict": verdict,
                               "evidence_receipt_id":
                                   evidence_receipt_id},
                     generation=generation)
        return case

    def close_unknown(self, case_id: str, *, note: str,
                      generation: int, actor: str = "") -> OutcomeCase:
        """Close a case with its outcome still unknown.

        The verdict stays ``outcome_unknown`` — this records that the
        checks were made and what they found, never a guessed result.
        The case may be reopened by a later re-discovery (the evidence
        situation may change); its receipts stay as the trail.
        """
        case = self._check_transition(case_id, generation)
        now = time.time()
        cursor = self.db.execute(
            "UPDATE outcome_cases SET state = 'closed-unknown',"
            " closed_at = ?, updated_at = ?"
            " WHERE id = ? AND generation = ?",
            (now, now, case_id, case.generation))
        if not cursor.rowcount:
            raise StaleCaseGeneration(
                f"case {case_id} moved under the closure")
        case.state = "closed-unknown"
        case.closed_at = now
        self._append(case, kind="closed-unknown",
                     source="outcome-cases", actor=actor, note=note,
                     generation=generation)
        return case

    # -- reads (the inspection API P6 puts clothes on) ------------------------

    def acknowledge(self, case_id: str, *, actor: str = "",
                    note: str = "") -> OutcomeCase:
        """Record the owner's acknowledgement of a case (P6).

        Acknowledgement is a recorded act, NEVER a resolution: the
        case's state and verdict are untouched, its generation is not
        bumped (an in-flight investigation is not fenced by an ack),
        and the case may still only reach a verdict through
        :meth:`resolve`'s evidence path or :meth:`close_unknown`.
        What the ack records: an ``acknowledged`` receipt on the
        trail, plus a snapshot in the case detail — the state and
        verdict the owner saw — which is what lets the steward's
        sweep quiet an acknowledged case until it materially changes
        (:func:`ack_changed`).  Any state may be acknowledged,
        including a terminal one (marking it seen).
        """
        case = self._case_or_raise(case_id)
        now = time.time()
        detail = dict(case.detail or {})
        detail["acknowledged"] = {
            "at": now, "by": actor or "owner",
            "state": case.state, "verdict": case.verdict,
        }
        detail["ack_count"] = int(detail.get("ack_count") or 0) + 1
        self.db.execute(
            "UPDATE outcome_cases SET detail = ?, updated_at = ? "
            "WHERE id = ?", (json.dumps(detail), now, case.id))
        case.detail = detail
        case.updated_at = now
        self._append(
            case, kind="acknowledged", source="outcome-cases",
            actor=actor or "owner",
            note=note or (f"owner acknowledged the case as seen — "
                           f"state {case.state}, verdict "
                           f"{case.verdict}; acknowledgement does not "
                           f"resolve it"))
        return case

    def list_cases(self, *, state: str = "", domain: str = "",
                   verdict: str = "", limit: int = 100
                   ) -> list[OutcomeCase]:
        clauses, params = [], []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.db.query(
            "SELECT * FROM outcome_cases" + where
            + " ORDER BY opened_at DESC LIMIT ?",
            (*params, int(limit)))
        return [_row_case(r) for r in rows]

    def counts(self) -> dict[str, int]:
        rows = self.db.query(
            "SELECT state, COUNT(*) AS n FROM outcome_cases"
            " GROUP BY state")
        out = {state: 0 for state in CASE_STATES}
        for row in rows:
            out[row["state"]] = int(row["n"])
        return out
