"""Healing tickets — the doc 20 envelope, as a projection (P1).

ONE ticket shape over the shipped immune + healing organs,
computed on read from the organs' own records. This module stores
NOTHING: no table, no file, no cache of record. It is the sibling
of doc 18's execution ticket (:mod:`.run_ticket`, whose
architecture this follows) over a different set of organs — doc
20's evaluation is explicit that a ticket *store* here would be a
fifth incident store forking incident truth, so the ticket
REFERENCES: incidents stay the incident journal's, antibodies stay
the antibody store's, case files and police receipts stay the
immune state's, outcomes stay the doc-15 steward's.

THE HONESTY RULE (binds every field, as in run_ticket): a field is
either copied from a named source record — cited in the ticket's
``sources`` map (field → organ) — or it is ABSENT, named in the
``absent`` map with the reason it cannot be filled from any
shipped record. No field ever receives a plausible default.

WHAT THE SURVEY FOUND (the field partitions below are this table):

* incident journal (``core/incidents.py``) — DURABLE (SQLite).
  Incidents and recovery records (the self-healing executor's
  journal: ``core/selfheal.py`` records a ``RecoveryRecord`` per
  attempt, verified or explicitly unverified) are real records
  with real read paths.
* antibody store (``immune/antibodies.py``) — durable: since
  doc 20 P3 the production chat state shares ONE path-backed
  store (``immune/wiring.shared_antibody_store``), and every
  record carries its P3 lifecycle (probation / active / retired,
  match + false-positive evidence, transition reasons) — the
  lifecycle this projection reports. The projection reads
  whatever store instance it is handed.
* immune case files + police receipts (``immune/casefile.py``,
  ``immune/police.py``) — records exist ONLY on the per-chat
  in-memory immune state (``immune/chat.py``'s ``_ImmuneState``);
  no durable store backs them today, so — exactly as run_ticket
  resolves ``task:`` refs against a caller-supplied TaskGraph —
  ``case:`` / ``receipt:`` / ``state:`` refs resolve against
  caller-supplied immune states.
* fever / drift (``immune/fever.py``, ``immune/drift.py``) —
  in-memory state on that same object; the adaptive baseline
  (``immune/baseline.py``) lives on wiring's ``ImmuneMonitor``
  and is supplied separately when the caller has it.
* command repair (``core/self_heal.py``) keeps NO run records —
  outcomes are reported to an injected error-intelligence KB by
  fingerprint — so there is no healing-run record to project for
  it; the executor's journal above is the healing-run source.
* the doc-15 outcome steward — NO shipped immune or healing
  record carries an outcome-case reference, and no steward
  adapter opens cases for immune/healing subjects. The
  ``steward`` field is therefore absent-with-reason on every
  ticket in this pass: a citation is never fabricated.

CLAIM GRADES speak doc 24's vocabulary
(:mod:`nomorals.core.proof_grade`) and are DERIVED, never copied:
where a record asserts an effect (a police receipt's reason text,
an antibody's match claim, a recovery's success), the ticket
attaches the evidence the record itself cites as
:class:`EvidenceRef` records and the grade is whatever
:func:`grade_of` derives from them. A receipt that *words* an
effect it does not evidence — the evaluation's sharpest finding:
the police disarm receipt reads "capability grant revoked" even
when no revoke function was wired — grades at what its cited
evidence earns (nothing attached → G0), with the reason named.

Doc 20 P2 has since fixed the police receipts' semantics: they
now record the effect outcome on the record itself
(``details.effect`` / ``details.effect_outcomes`` — whether the
performing function was wired, and what it returned or raised).
The projection reads that record first: a receipt that records
its effect as not performed (or failed) carries NO claim — the
absence is the honest projection; a receipt that records a
performed effect grades its claim on that outcome record (an
operation pass, cited as this receipt). Receipts written before
P2 carry no effect record; they are graded exactly as P1 graded
them — on the artifacts they cite, never on their wording — so a
bare legacy success claim still earns G0 here.

Doc 20 P7 has since given the police real teeth: issued grants
are recorded in the core grant registry and revocation acts on
them by id, so a disarm performed through the wired surface
records its effect as performed and grades on that record exactly
as P2 specified. When a caller supplies the registry
(``grant_registry=``), this projection also READS it: a disarm
receipt's evidence field cites the registry's own record for the
target grant. The reader changes no grading rule — a legacy
receipt's claim still grades on what the receipt itself cites —
it adds the durable record the claim is ultimately about.

Reference shapes are the organs' own keys: ``incident:<id>`` ·
``heal:<recovery row id>`` (both incident-journal keys) ·
``case:<casefile id>`` · ``antibody:<id>`` · ``receipt:<id>``
(the immune state's records) · ``state:<chat key>`` (the chat
state's own keying in ``immune/chat.py``).

SHADOW (P1): no producer is required to mint tickets, no gate
consults them, no existing surface changes its output. The
deliverable is the projection itself plus
:func:`healing_ticket_coverage`.

Layer: agents (L5). Reads: core.incidents' journal via its public
API and its SQLite file directly (the outcome steward's direct-
table precedent — observation constructs no journal and writes
nothing), core.proof_grade (L1), and caller-supplied immune
objects duck-typed (immune/ is an unmapped leaf package).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.proof_grade import ClaimGrade, EvidenceRef

__all__ = [
    "HEALING_FIELDS",
    "HealingTicket",
    "healing_ticket_for",
    "healing_ticket_coverage",
]

#: The envelope's fields, in ticket order. Every field on every
#: ticket is either in ``sources`` (populated, cited) or in
#: ``absent`` (with the reason) — never neither, never both.
HEALING_FIELDS = (
    "purpose",
    "actors",
    "claim",
    "evidence",
    "lifecycle",
    "liveness",
    "steward",
)

#: Reference prefixes this projection reads, mapped to ticket kinds.
_REF_KINDS = {
    "incident": "incident",
    "heal": "healing_run",
    "case": "immune_case",
    "antibody": "antibody",
    "receipt": "police_receipt",
    "state": "immune_state",
}

#: Police rung names, mirrored from immune/police.py RUNGS (the
#: receipts this projection reads carry the rung number; the name
#: table is display data in the owning module).
_RUNG_NAMES = {
    1: "Observe", 2: "Mark", 3: "Quarantine", 4: "Disarm",
    5: "Arrest", 6: "Case file", 7: "Notify",
}

#: Incident-journal chronicity thresholds, mirrored from
#: core/incidents.py IncidentJournal (CHRONIC_1H / CHRONIC_24H);
#: a supplied journal instance's own attributes win when present.
_CHRONIC_1H = 3
_CHRONIC_24H = 10

_STEWARD_ABSENT = (
    "no shipped immune or healing record carries an outcome-case "
    "reference, and no doc-15 steward adapter opens cases for "
    "immune/healing subjects — there is no recorded link to cite, "
    "and a citation is never fabricated")


@dataclass
class HealingTicket:
    """One immune/healing operation, projected. ``ref``/``kind``/
    ``label``/``status`` are the envelope's own frame; every
    HEALING_FIELDS member is populated-and-cited or absent-with-
    reason. Build via :func:`healing_ticket_for`, never by hand."""

    ref: str
    kind: str
    label: str
    status: str
    purpose: Any = None
    actors: Any = None
    claim: Any = None
    evidence: Any = None
    lifecycle: Any = None
    liveness: Any = None
    steward: Any = None
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
        return [name for name in HEALING_FIELDS if name in self.absent]

    def complete(self) -> bool:
        """Complete only when EVERY field resolved to a source
        record. (Most kinds keep actors/steward absent — the
        records genuinely do not carry them — so complete tickets
        are rare; the count is reported, never rounded up.)"""
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
        for name in HEALING_FIELDS:
            out[name] = getattr(self, name)
        return out


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


def _get(record: Any, name: str, default: Any = None) -> Any:
    """One field from a record that may be a dataclass or a dict
    (the immune organs accept both shapes in their own stores)."""
    if isinstance(record, Mapping):
        value = record.get(name, default)
    else:
        value = getattr(record, name, default)
    return default if value is None else value


def _claim(text: str, evidence: list[EvidenceRef],
           grade_reason: str) -> dict[str, Any]:
    """A claim field whose grade is DERIVED from the attached
    evidence via PG1's machinery — the grade is read off the
    ClaimGrade, never stated by this module."""
    graded = ClaimGrade(evidence=tuple(evidence))
    return {
        "text": text,
        "grade": graded.grade.name,
        "evidence": [item.to_dict() for item in graded.evidence],
        "grade_reason": grade_reason,
    }


def _no_steward(ticket: HealingTicket) -> None:
    ticket.mark_absent("steward", _STEWARD_ABSENT)


# ── supplied immune states ────────────────────────────────────────


def _state_items(immune_states: Any) -> list[tuple[str, Any]]:
    """Normalise the caller-supplied immune states to (key, state)
    pairs. Accepts the chat module's own keying (a mapping of
    chat key → state) or a bare iterable of states."""
    if immune_states is None:
        return []
    if isinstance(immune_states, Mapping):
        return [(str(key), state)
                for key, state in immune_states.items()]
    return [(str(getattr(state, "chat_key", "") or ""), state)
            for state in immune_states]


def _find_case(items: list[tuple[str, Any]], ident: str) -> Any:
    for _key, state in items:
        for case in getattr(state, "casefiles", None) or []:
            if str(_get(case, "id", "")) == ident:
                return case
    return None


def _find_receipt(items: list[tuple[str, Any]], ident: str) -> Any:
    for _key, state in items:
        for receipt in getattr(state, "receipts", None) or []:
            if str(_get(receipt, "id", "")) == ident:
                return receipt
    return None


def _find_antibody(items: list[tuple[str, Any]],
                   stores: Any, ident: str) -> tuple[Any, Any]:
    """(record, resolving store) — the store rides along because
    it is the only reader of the record's version history."""
    candidates = list(stores or [])
    for _key, state in items:
        store = getattr(state, "store", None)
        if store is not None:
            candidates.append(store)
    for store in candidates:
        try:
            record = store.get(ident)
        except Exception:  # noqa: BLE001 - an unreadable store
            record = None
        if record is not None:
            return record, store
    return None, None


# ── incident-journal reads ────────────────────────────────────────


def _journal_conn(journal: Any, journal_db: Any) -> sqlite3.Connection | None:
    """A read connection over the incident journal's SQLite file,
    or None when no readable file backs the journal. A path that
    does not exist reads as no records — and is never created:
    observation writes nothing. A supplied connection is used
    as-is; a supplied journal instance contributes its file path
    when it keeps one."""
    if isinstance(journal_db, sqlite3.Connection):
        return journal_db
    path = journal_db
    if path is None and journal is not None:
        path = getattr(journal, "_path", None)
    if path is None or str(path) == ":memory:":
        return None
    target = Path(str(path))
    if not target.exists():
        return None
    try:
        return sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _sql_one(conn: sqlite3.Connection | None, query: str,
             params: tuple) -> dict[str, Any] | None:
    if conn is None:
        return None
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(query, params).fetchone()
    except sqlite3.Error:
        return None
    return dict(row) if row is not None else None


def _sql_all(conn: sqlite3.Connection | None, query: str,
             params: tuple) -> list[dict[str, Any]]:
    if conn is None:
        return []
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query, params)]
    except sqlite3.Error:
        return []


def _incident_row(journal: Any, conn: sqlite3.Connection | None,
                  ident: str) -> dict[str, Any] | None:
    """One incident as a plain dict — via the journal's own public
    API when an instance is supplied (its hydration computes the
    repeat counts), else by direct SQL with the counts computed
    the same way."""
    if journal is not None:
        try:
            incident = journal.get_incident(int(ident))
        except Exception:  # noqa: BLE001 - KeyError or unreadable
            incident = None
        if incident is not None:
            return {
                "id": incident.id, "signature": incident.signature,
                "exc_type": incident.exc_type,
                "subsystem": incident.subsystem,
                "location": incident.location,
                "message": incident.message,
                "category": incident.category,
                "severity": incident.severity,
                "ts": incident.ts,
                "repeat_count_1h": incident.repeat_count_1h,
                "repeat_count_24h": incident.repeat_count_24h,
                "verified_fix": incident.verified_fix,
            }
    row = _sql_one(conn, "SELECT * FROM incidents WHERE id = ?",
                   (ident,))
    if row is None:
        return None
    now = time.time()
    counts = _sql_one(
        conn,
        "SELECT"
        " (SELECT COUNT(*) FROM incidents WHERE signature = ?"
        "  AND ts >= ?) AS c1h,"
        " (SELECT COUNT(*) FROM incidents WHERE signature = ?"
        "  AND ts >= ?) AS c24h",
        (row["signature"], now - 3600, row["signature"], now - 86400))
    fix = _sql_one(
        conn,
        "SELECT strategy, detail, successes, failures"
        " FROM verified_fixes WHERE signature = ?"
        " ORDER BY successes DESC LIMIT 1",
        (row["signature"],))
    row["repeat_count_1h"] = counts["c1h"] if counts else 0
    row["repeat_count_24h"] = counts["c24h"] if counts else 0
    row["verified_fix"] = fix
    return row


# ── per-organ assemblers ──────────────────────────────────────────


def _incident_ticket(row: dict[str, Any], conn: sqlite3.Connection | None,
                     journal: Any, now: float) -> HealingTicket:
    ident = row["id"]
    ticket = HealingTicket(
        ref=f"incident:{ident}", kind="incident",
        label=f"incident: {row.get('exc_type') or 'failure'} in "
              f"{row.get('subsystem') or '?'}",
        status="recorded", read_at=now)
    ticket.fill("purpose",
                {"exc_type": row.get("exc_type"),
                 "subsystem": row.get("subsystem"),
                 "location": row.get("location"),
                 "category": row.get("category"),
                 "severity": row.get("severity")},
                "core.incidents")
    ticket.mark_absent(
        "actors",
        "the incident journal names the failing subsystem (in "
        "purpose) but no actor — the catcher that recorded it and "
        "the organ that later recovers it are not named on the "
        "incident row")
    ticket.mark_absent(
        "claim",
        "an incident asserts that a failure occurred; it claims "
        "no effect — effect claims live on its recovery records "
        "(heal:<recovery id>)")
    ticket.fill("evidence",
                {"location": row.get("location"),
                 "repeat_count_1h": row.get("repeat_count_1h"),
                 "repeat_count_24h": row.get("repeat_count_24h"),
                 "verified_fix": row.get("verified_fix")},
                "core.incidents")
    chronic_1h = getattr(journal, "CHRONIC_1H", _CHRONIC_1H)
    chronic_24h = getattr(journal, "CHRONIC_24H", _CHRONIC_24H)
    lifecycle: dict[str, Any] = {
        "severity": row.get("severity"),
        "category": row.get("category"),
        "chronic_1h": (row.get("repeat_count_1h") or 0) >= chronic_1h,
        "chronic_24h": (row.get("repeat_count_24h") or 0) >= chronic_24h,
    }
    recoveries = _sql_all(
        conn,
        "SELECT id, verified FROM recoveries WHERE incident_id = ?"
        " ORDER BY id", (ident,))
    if conn is not None:
        lifecycle["standing"] = (
            "recovered" if any(r["verified"] for r in recoveries)
            else "open")
        for rec in recoveries:
            ticket.related_refs.append(f"heal:{rec['id']}")
    else:
        lifecycle["standing"] = (
            "undetermined — the incident's recovery rows are "
            "unread without the journal's database file")
    ticket.fill("lifecycle", lifecycle, "core.incidents")
    ticket.fill("liveness",
                {"recorded_at": row.get("ts"),
                 "age_s": _age(now, row.get("ts"))},
                "core.incidents")
    _no_steward(ticket)
    ticket.detail = {
        "signature": row.get("signature"),
        "message": str(row.get("message") or "")[:300],
        "recoveries": len(recoveries) if conn is not None else None,
    }
    return ticket


def _heal_ticket(row: dict[str, Any], conn: sqlite3.Connection | None,
                 now: float) -> HealingTicket:
    rid = row["id"]
    strategy = str(row.get("strategy") or "")
    verified = bool(row.get("verified"))
    ticket = HealingTicket(
        ref=f"heal:{rid}", kind="healing_run",
        label=f"healing run: {strategy or 'recovery'}",
        status="verified" if verified else "unverified",
        read_at=now)
    ticket.fill("purpose",
                {"strategy": strategy,
                 "signature": row.get("signature"),
                 "incident_id": row.get("incident_id"),
                 "detail": str(row.get("detail") or "")[:300]},
                "core.incidents")
    ticket.mark_absent(
        "actors",
        "the recovery row names its strategy (in purpose) but "
        "not the organ that ran it — the self-healing executor "
        "is the only writer today, and 'today' is not a record")
    text = f"recovery via {strategy}" if strategy else "recovery"
    detail_text = str(row.get("detail") or "").strip()
    if detail_text:
        text = f"{text}: {detail_text[:160]}"
    if verified:
        claim = _claim(
            text,
            [EvidenceRef("artifact_pass", f"heal:{rid}")],
            "the journal records this recovery VERIFIED: the "
            "executor's verify step re-ran the recovered call and "
            "it passed, and this journal row is the record of that "
            "check (its verify_note rides in evidence) — an "
            "operation pass, which is G5's evidence type. Mined "
            "rule 3: only verified recoveries promote to the "
            "verified-fix table")
    else:
        claim = _claim(
            text, [],
            "the journal itself records this recovery UNVERIFIED "
            "— no verification result is cited, so the claim "
            "grades on its cited evidence: none attached. "
            "Unverified recoveries are never promoted to the "
            "verified-fix table (mined rule 3)")
    ticket.fill("claim", claim, "core.incidents")
    promoted = None
    if verified and conn is not None:
        fix = _sql_one(
            conn,
            "SELECT signature FROM verified_fixes"
            " WHERE signature = ? AND strategy = ?",
            (row.get("signature"), strategy))
        promoted = fix is not None
    ticket.fill("evidence",
                {"verified": verified,
                 "verify_note": str(row.get("verify_note") or ""),
                 "promoted_fix": promoted},
                "core.incidents")
    standing = ("promoted" if promoted else
                "verified" if verified else "unverified")
    ticket.fill("lifecycle",
                {"verified": verified, "standing": standing},
                "core.incidents")
    ticket.fill("liveness",
                {"recorded_at": row.get("ts"),
                 "age_s": _age(now, row.get("ts"))},
                "core.incidents")
    _no_steward(ticket)
    if row.get("incident_id") is not None:
        ticket.related_refs.append(f"incident:{row['incident_id']}")
    ticket.detail = {"recovery_id": rid}
    return ticket


def _case_ticket(case: Any, items: list[tuple[str, Any]],
                 now: float) -> HealingTicket:
    ident = str(_get(case, "id", ""))
    event = _get(case, "event")
    sealed = bool(_get(case, "sealed", False))
    evidence = _get(case, "evidence", []) or []
    timeline = _get(case, "timeline", []) or []
    ticket = HealingTicket(
        ref=f"case:{ident}", kind="immune_case",
        label=f"immune case: {str(event)[:40]}" if event
        else "immune case",
        status="sealed" if sealed else "open", read_at=now)
    if event is not None:
        ticket.fill("purpose", str(event)[:300], "immune.casefile")
    else:
        ticket.mark_absent(
            "purpose",
            "the case file records no triggering event")
    ticket.mark_absent(
        "actors",
        "a case file names no actor — it bundles an event, a "
        "timeline and evidence; who built it is not recorded on "
        "the file")
    ticket.mark_absent(
        "claim",
        "a case file asserts no effect — it holds evidence about "
        "an event; effect claims live on the police receipts and "
        "recovery records that cite it")
    ticket.fill("evidence",
                {"evidence_items": len(evidence),
                 "timeline_items": len(timeline),
                 "sealed": sealed},
                "immune.casefile")
    ticket.fill("lifecycle",
                {"sealed": sealed,
                 "state": "sealed" if sealed else "open"},
                "immune.casefile")
    created = _get(case, "created_ts")
    ticket.fill("liveness",
                {"created_at": created, "age_s": _age(now, created)},
                "immune.casefile")
    _no_steward(ticket)
    for _key, state in items:
        for receipt in getattr(state, "receipts", None) or []:
            details = _get(receipt, "details", {}) or {}
            if str(details.get("case_id") or "") == ident:
                ticket.related_refs.append(
                    f"receipt:{_get(receipt, 'id', '')}")
    ticket.detail = {"case_id": ident}
    return ticket


def _antibody_ticket(record: Any, store: Any,
                     now: float) -> HealingTicket:
    ident = str(_get(record, "id", ""))
    status = str(_get(record, "status", "") or "")
    pattern = str(_get(record, "pattern", "") or "")
    pathogen_class = str(_get(record, "pathogen_class", "") or "")
    ticket = HealingTicket(
        ref=f"antibody:{ident}", kind="antibody",
        label=f"antibody: {pathogen_class or 'unclassified'} pattern",
        status=status, read_at=now)
    ticket.fill("purpose",
                {"pattern": pattern, "pathogen_class": pathogen_class},
                "immune.antibodies")
    ticket.mark_absent(
        "actors",
        "the antibody record names no author — proposals pass "
        "through the thymus, but who proposed a record is not "
        "carried on it")
    claim_text = (f"pattern matches {pathogen_class} content"
                  if pathogen_class else "pattern matches content")
    ticket.fill(
        "claim",
        _claim(
            claim_text, [],
            "the record cites no evidence artifact — its reason "
            "is a rationale for existing, and the negative-"
            "selection result that 'active' implies in the "
            "shipped flow is not recorded on the record, so it "
            "cannot be attached as evidence; the claim grades on "
            "its cited evidence: none attached"),
        "immune.antibodies")
    history = None
    if store is not None and callable(getattr(store, "history", None)):
        try:
            history = store.history(ident)
        except Exception:  # noqa: BLE001 - unreadable history
            history = None
    if history is not None:
        ticket.fill("evidence",
                    {"versions_in_history": len(history),
                     "current_version": _get(record, "version")},
                    "immune.antibodies")
    else:
        ticket.mark_absent(
            "evidence",
            "the resolving store exposes no version history for "
            "this record, and the record itself cites no evidence")
    lifecycle = {"status": status,
                 "version": _get(record, "version"),
                 "retire_reason": _get(record, "retire_reason")}
    # Doc 20 P3 lifecycle detail, copied when the record carries
    # it (P3 records always do; a pre-P3 record simply lacks the
    # keys and the ticket says no more than the record knows).
    for key in ("origin", "status_reason", "status_ts",
                "match_count", "false_positive_count",
                "probation_since", "probation_matches",
                "probation_false_positives", "status_evidence"):
        value = _get(record, key)
        if value is not None:
            lifecycle[key] = value
    ticket.fill("lifecycle", lifecycle, "immune.antibodies")
    created = _get(record, "created_ts")
    ticket.fill("liveness",
                {"created_at": created, "age_s": _age(now, created)},
                "immune.antibodies")
    _no_steward(ticket)
    ticket.detail = {
        "antibody_id": ident,
        "reason": str(_get(record, "reason", "") or ""),
    }
    return ticket


def _settle_claim_from_effect(ticket: HealingTicket, receipt: Any,
                              action: str, reason_text: str,
                              details: dict) -> bool:
    """Settle a police receipt's claim from its doc 20 P2 effect
    record (``details.effect`` / ``details.effect_outcomes``).

    Returns True when the effect record settled the claim field:

    * ``not_performed`` / ``failed`` — the record itself states no
      effect was taken, so the claim is ABSENT, with the recorded
      cause named. There is nothing to grade.
    * ``performed`` / ``partial`` (non-casefile) — the claim grades
      on the outcome record: the performing function ran during
      this action and its return (or the partial truth) is carried
      on this very receipt, cited as the claim's evidence — an
      operation pass, G5's evidence type. The claim text is the
      receipt's own reason, which under P2 states exactly what
      was and was not performed.

    Returns False when the receipt carries no effect record (a
    pre-P2 record) or is a casefile (whose claim grades on the
    case artifact it cites) — the caller then applies the legacy
    cited-evidence grading.
    """
    effect = details.get("effect")
    if effect not in ("performed", "partial", "not_performed", "failed"):
        return False
    if action == "casefile" and effect in ("performed", "partial"):
        return False  # grades on the cited case artifact, below
    outcomes = details.get("effect_outcomes") or []
    causes: list[str] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        fn_name = str(outcome.get("fn") or "the effect function")
        if not outcome.get("wired"):
            causes.append(f"{fn_name} was not wired")
        elif not outcome.get("ok"):
            causes.append(
                f"{fn_name} raised {outcome.get('error') or 'an error'}")
    cause_text = "; ".join(causes)
    if effect == "not_performed":
        ticket.mark_absent(
            "claim",
            "the receipt itself records that NO effect was "
            "performed"
            + (f" — {cause_text}" if cause_text else "")
            + "; there is no effect claim to grade")
        return True
    if effect == "failed":
        ticket.mark_absent(
            "claim",
            "the receipt itself records that the attempted effect "
            "FAILED"
            + (f" — {cause_text}" if cause_text else "")
            + "; no effect was taken, so there is no effect claim "
              "to grade")
        return True
    ident = str(_get(receipt, "id", ""))
    if effect == "partial":
        grade_reason = (
            "the receipt records the effect as PARTIAL in its "
            "doc 20 P2 effect record (details.effect_outcomes): "
            "the claim text states which part was not performed, "
            "and the performed part grades on the recorded outcome "
            "— this receipt is the operation's outcome record, "
            "cited as the claim's evidence (an operation pass)")
    else:
        grade_reason = (
            "the receipt records the effect as PERFORMED in its "
            "doc 20 P2 effect record (details.effect_outcomes): "
            "the performing function ran during this action and "
            "its outcome is carried on the record — this receipt "
            "is the operation's outcome record, cited as the "
            "claim's evidence (an operation pass)")
    ticket.fill(
        "claim",
        _claim(reason_text,
               [EvidenceRef("artifact_pass", f"receipt:{ident}")],
               grade_reason),
        "immune.police")
    return True


def _grant_record_for(action: str, target: str, details: dict,
                      grant_registry: Any) -> dict[str, Any] | None:
    """The grant registry's record for a disarm receipt's target
    grant (doc 20 P7) — read at projection time from the registry,
    the revocation's durable record of authority. None when no
    registry is supplied, the receipt is not a disarm, the receipt
    names no grant, or the registry holds no such grant."""
    if grant_registry is None or action != "disarm":
        return None
    grant_id = str(details.get("grant_id") or target or "")
    if not grant_id:
        return None
    try:
        row = grant_registry.get(grant_id)
    except Exception:  # noqa: BLE001 - an unreadable registry
        return None
    if row is None:
        return None
    standing = ""
    standing_fn = getattr(grant_registry, "standing", None)
    if callable(standing_fn):
        try:
            standing = str(standing_fn(row))
        except Exception:  # noqa: BLE001 - derive below instead
            standing = ""
    if not standing:
        standing = ("revoked" if _get(row, "state") == "revoked"
                    else "active")
    return {
        "grant_id": grant_id,
        "patterns": _get(row, "patterns"),
        "actor": _get(row, "actor"),
        "subject": _get(row, "subject"),
        "source": _get(row, "source"),
        "issued_at": _get(row, "issued_at"),
        "expires_at": _get(row, "expires_at"),
        "standing": standing,
        "revoked_at": _get(row, "revoked_at"),
        "revoked_by": _get(row, "revoked_by"),
    }


def _receipt_ticket(receipt: Any, now: float,
                    grant_registry: Any = None) -> HealingTicket:
    ident = str(_get(receipt, "id", ""))
    rung = _get(receipt, "rung", 0)
    action = str(_get(receipt, "action", "") or "")
    target = str(_get(receipt, "target", "") or "")
    details = _get(receipt, "details", {}) or {}
    ticket = HealingTicket(
        ref=f"receipt:{ident}", kind="police_receipt",
        label=f"police {action or 'action'} on {target[:30]}"
        if target else f"police {action or 'action'}",
        status=action, read_at=now)
    try:
        rung_int = int(rung)
    except (TypeError, ValueError):
        rung_int = 0
    ticket.fill("purpose",
                {"rung": rung_int,
                 "rung_name": _RUNG_NAMES.get(rung_int, ""),
                 "action": action, "target": target},
                "immune.police")
    ticket.fill("actors",
                {"actor": str(_get(receipt, "actor", "") or "")},
                "immune.police")
    reason_text = str(_get(receipt, "reason", "") or "")
    if action == "report_only":
        ticket.mark_absent(
            "claim",
            "the receipt is report-only: it asserts that NO "
            "effect was taken, so there is no effect claim to "
            "grade")
    elif action == "observe":
        ticket.mark_absent(
            "claim",
            "an observation asserts no effect — it records that "
            "an event was seen")
    elif _settle_claim_from_effect(ticket, receipt, action,
                                   reason_text, details):
        pass  # the doc 20 P2 effect record settled the claim field
    else:
        evidence_refs: list[EvidenceRef] = []
        case_id = details.get("case_id")
        if action == "casefile" and case_id:
            evidence_refs.append(
                EvidenceRef("artifact_pass", f"case:{case_id}"))
            grade_reason = (
                "the receipt cites the case file it built "
                "(details.case_id) — a built artifact, attached "
                "as the claim's evidence")
        elif action == "casefile":
            grade_reason = (
                "the receipt asserts a case file was built but "
                "records no case id (details.case_id is empty — "
                "no casefile function produced one) — no artifact "
                "is cited, so the claim grades on its cited "
                "evidence: none attached")
        elif action == "disarm":
            grade_reason = (
                "the receipt asserts the grant was revoked, but "
                "it records neither whether a revoke function "
                "was wired nor what it returned — its details "
                "name the target grant, which is a target, not "
                "evidence; the claim grades on its cited "
                "evidence: none attached")
        elif action == "arrest":
            grade_reason = (
                "the receipt asserts the process was killed and "
                "its mission cancelled, but records neither "
                "whether the kill/cancel functions were wired "
                "nor what they returned — no evidence artifact "
                "is cited")
        elif action == "quarantine":
            grade_reason = (
                "the receipt asserts the artifact was vaulted, "
                "but records neither whether a vault function "
                "was wired nor its result — its details name "
                "the artifact, which is a target, not evidence")
        elif action == "enforce":
            grade_reason = (
                "details.enforced is the receipt's own assertion "
                "that enforcement ran, not independent evidence "
                "of the effect — no evidence artifact is cited")
        elif action in ("notify", "notify_batched"):
            grade_reason = (
                "the receipt asserts delivery (mode in details) "
                "but records no delivery receipt — no evidence "
                "artifact is cited")
        else:
            grade_reason = (
                "the receipt cites no evidence artifact for the "
                "effect its reason text asserts")
        ticket.fill("claim",
                    _claim(reason_text, evidence_refs, grade_reason),
                    "immune.police")
    cited: list[str] = []
    if details.get("case_id"):
        cited.append(f"case:{details['case_id']}")
        ticket.related_refs.append(f"case:{details['case_id']}")
    evidence: dict[str, Any] = {}
    if cited:
        evidence["cited_refs"] = cited
    grant_record = _grant_record_for(action, target, details,
                                     grant_registry)
    if grant_record is not None:
        # Doc 20 P7: the registry reader — a disarm receipt's
        # evidence cites the registry's own durable record for
        # the grant it names. The claim's grading (above) is
        # untouched: it reads the receipt, exactly as P1/P2
        # specified; this field is the record the claim is about.
        evidence["grant_registry"] = grant_record
        ticket.related_refs.append(
            f"grant:{grant_record['grant_id']}")
    if evidence:
        ticket.fill("evidence", evidence,
                    "immune.police" if cited
                    else "core.grant_registry")
    else:
        ticket.mark_absent(
            "evidence",
            "the receipt's details cite no evidence artifact — "
            "they name targets and delivery modes, not evidence")
    ticket.fill("lifecycle",
                {"rung": rung_int,
                 "rung_name": _RUNG_NAMES.get(rung_int, ""),
                 "reversible": bool(_get(receipt, "reversible", False))},
                "immune.police")
    ts = _get(receipt, "ts")
    ticket.fill("liveness",
                {"recorded_at": ts, "age_s": _age(now, ts)},
                "immune.police")
    _no_steward(ticket)
    ticket.detail = {"receipt_id": ident, "details": dict(details)}
    return ticket


def _state_ticket(key: str, state: Any, baseline: Any,
                  now: float) -> HealingTicket:
    shown_key = key or "default"
    ticket = HealingTicket(
        ref=f"state:{shown_key}", kind="immune_state",
        label=f"immune state: {shown_key}", status="", read_at=now)
    ticket.fill("purpose",
                {"chat_key": shown_key,
                 "surface": "per-chat immune state (immune/chat.py)"},
                "immune.chat")
    ticket.mark_absent(
        "actors",
        "the state is a container — it names no actor; the "
        "actions taken through it are on its receipts")
    ticket.mark_absent(
        "claim",
        "the state asserts no effect — claims live on the "
        "receipts and antibodies it holds (receipt:<id>, "
        "antibody:<id>)")
    corpus = getattr(getattr(state, "thymus", None), "corpus", None)
    evidence: dict[str, Any] = {
        "self_corpus_exemplars": len(corpus) if corpus is not None
        else None,
    }
    if baseline is not None:
        thresholds = baseline.thresholds()
        evidence["baseline"] = {
            "samples": len(baseline),
            "status": thresholds.status,
            "spawned_p95": thresholds.spawned_p95,
            "growth_p95": thresholds.growth_p95,
            "reason": thresholds.reason,
        }
    else:
        ticket.detail["baseline_unread"] = (
            "the chat immune state carries no adaptive baseline — "
            "it lives on wiring's ImmuneMonitor (persisted to "
            "data/immune-baseline.json); supply it to read it")
    observations = getattr(state, "observations", None)
    if observations is not None:
        # Doc 20 P5: the observation journal (drift days, fever
        # windows, false-positive verdicts) is durable evidence
        # the state carries — its size and location ride the
        # already-sourced evidence field, so the field partition
        # (sources ∪ absent) is untouched.
        try:
            obs_records: Any = observations.count()
        except Exception:  # noqa: BLE001 — a read must not sink
            obs_records = None  # the ticket it rides on
        obs_path = getattr(observations, "path", None)
        evidence["observations"] = {
            "records": obs_records,
            "log": str(obs_path) if obs_path else
                   "in-memory only (not persisted)",
        }
    ticket.fill("evidence", evidence, "immune.chat")
    fever = getattr(state, "fever", None)
    drift = getattr(state, "drift", None)
    lifecycle: dict[str, Any] = {}
    if fever is not None:
        lifecycle["fever_active"] = bool(fever.is_active(now))
        lifecycle["fever_remaining_s"] = round(fever.remaining(now), 1)
        lifecycle["fever_reason"] = fever.reason
        lifecycle["actions_throttled"] = fever.actions_throttled
    if drift is not None:
        report = drift.current_report()
        lifecycle["drift_verdict"] = report.verdict
        lifecycle["drift_signals"] = list(report.signals)
        lifecycle["drift_days"] = report.days_considered
    ticket.fill("lifecycle", lifecycle, "immune.chat")
    store = getattr(state, "store", None)
    antibodies = store.all() if store is not None else []
    casefiles = getattr(state, "casefiles", None) or []
    receipts = getattr(state, "receipts", None) or []
    ticket.fill("liveness",
                {"antibodies": len(antibodies),
                 "active_antibodies": sum(
                     1 for ab in antibodies
                     if _get(ab, "status") == "active"),
                 "casefiles": len(casefiles),
                 "receipts": len(receipts),
                 "vault_items": len(getattr(state, "vault", None)
                                    or [])},
                "immune.chat")
    ticket.status = ("fever" if lifecycle.get("fever_active")
                     else "calm")
    _no_steward(ticket)
    for case in casefiles[:50]:
        ticket.related_refs.append(f"case:{_get(case, 'id', '')}")
    for ab in antibodies[:50]:
        ticket.related_refs.append(f"antibody:{_get(ab, 'id', '')}")
    for receipt in receipts[:50]:
        ticket.related_refs.append(f"receipt:{_get(receipt, 'id', '')}")
    ticket.detail["related_capped_at_50_per_kind"] = (
        len(casefiles) > 50 or len(antibodies) > 50
        or len(receipts) > 50)
    return ticket


# ── the read API ──────────────────────────────────────────────────


def healing_ticket_for(ref: str, *, journal: Any = None,
                       journal_db: Any = None,
                       immune_states: Any = None,
                       antibody_stores: Any = None,
                       baseline: Any = None,
                       grant_registry: Any = None,
                       now: float | None = None) -> dict[str, Any]:
    """One immune/healing operation's ticket, by the owning
    organ's own reference, or a typed not-found dict
    (``{"found": False, "ref", "reason"}``).

    ``journal`` is an IncidentJournal instance (incidents read via
    its public API); ``journal_db`` is the journal's SQLite file
    path or connection (recovery rows and SQL-side reads). Either
    alone is enough for the refs it can serve. ``immune_states``
    is the chat module's keying (chat key → state) or an iterable
    of immune states; ``antibody_stores`` adds standalone
    AntibodyStore instances; ``baseline`` is an AdaptiveBaseline
    (wiring's ImmuneMonitor.baseline) for ``state:`` tickets;
    ``grant_registry`` is the core grant registry (doc 20 P7)
    whose record a disarm receipt's evidence field cites.
    """
    now = time.time() if now is None else now
    prefix, _, ident = str(ref or "").partition(":")
    kind = _REF_KINDS.get(prefix)
    if kind is None or not ident:
        return _not_found(
            str(ref or ""),
            "not a healing-ticket reference — expected one of "
            + ", ".join(f"{p}:<id>" for p in sorted(_REF_KINDS)))
    items = _state_items(immune_states)
    if kind == "incident":
        if journal is None and journal_db is None:
            return _not_found(ref, "reading an incident ticket "
                                   "needs an incident journal or "
                                   "its database file")
        conn = _journal_conn(journal, journal_db)
        row = _incident_row(journal, conn, ident)
        if row is None:
            return _not_found(ref, "no incident record with id "
                                   f"{ident!r} exists")
        return _incident_ticket(row, conn, journal, now).to_dict()
    if kind == "healing_run":
        conn = _journal_conn(journal, journal_db)
        if conn is None:
            return _not_found(ref, "reading a healing-run ticket "
                                   "needs the incident journal's "
                                   "database file (recovery rows "
                                   "have no other read path)")
        row = _sql_one(conn, "SELECT * FROM recoveries WHERE id = ?",
                       (ident,))
        if row is None:
            return _not_found(ref, "no recovery record with id "
                                   f"{ident!r} exists")
        return _heal_ticket(row, conn, now).to_dict()
    if kind == "immune_case":
        if not items:
            return _not_found(ref, "immune case files live on a "
                                   "per-chat immune state — one "
                                   "must be supplied to read a "
                                   "case ticket")
        case = _find_case(items, ident)
        if case is None:
            return _not_found(ref, "no case file with id "
                                   f"{ident!r} exists in the "
                                   "supplied immune states")
        return _case_ticket(case, items, now).to_dict()
    if kind == "antibody":
        if not items and not antibody_stores:
            return _not_found(ref, "reading an antibody ticket "
                                   "needs an antibody store or an "
                                   "immune state")
        record, store = _find_antibody(items, antibody_stores, ident)
        if record is None:
            return _not_found(ref, "no antibody record with id "
                                   f"{ident!r} exists in the "
                                   "supplied stores")
        return _antibody_ticket(record, store, now).to_dict()
    if kind == "police_receipt":
        if not items:
            return _not_found(ref, "police receipts live on a "
                                   "per-chat immune state — one "
                                   "must be supplied to read a "
                                   "receipt ticket")
        receipt = _find_receipt(items, ident)
        if receipt is None:
            return _not_found(ref, "no police receipt with id "
                                   f"{ident!r} exists in the "
                                   "supplied immune states")
        return _receipt_ticket(receipt, now, grant_registry).to_dict()
    # kind == "immune_state"
    if not items:
        return _not_found(ref, "reading an immune-state ticket "
                               "needs a supplied immune state")
    for key, state in items:
        if key == ident or (ident == "default" and key in ("", "default")):
            return _state_ticket(key, state, baseline, now).to_dict()
    return _not_found(ref, "no supplied immune state is keyed "
                           f"{ident!r}")


def healing_ticket_coverage(*, journal: Any = None,
                            journal_db: Any = None,
                            immune_states: Any = None,
                            antibody_stores: Any = None,
                            baseline: Any = None, limit: int = 50,
                            grant_registry: Any = None,
                            now: float | None = None) -> dict[str, Any]:
    """The envelope over the live stores: every operation the
    supplied organs currently hold (up to ``limit`` per source),
    each ticket's absent fields counted per field and per source,
    and its effect claims' derived grades tallied. ``complete``
    counts tickets with NO absent field; the number is reported,
    never rounded up. Sources that cannot be read (no journal
    database for recovery rows, no supplied states) are named in
    ``unread`` rather than silently skipped."""
    now = time.time() if now is None else now
    tickets: list[HealingTicket] = []
    unread: list[str] = []
    items = _state_items(immune_states)
    conn = _journal_conn(journal, journal_db)
    if journal is not None or conn is not None:
        if journal is not None:
            try:
                recent = journal.recent_incidents(limit=limit)
            except Exception:  # noqa: BLE001 - unreadable journal
                recent = []
            seen: set[Any] = set()
            for entry in recent:
                row = _incident_row(journal, conn, str(entry["id"]))
                if row is not None and row["id"] not in seen:
                    seen.add(row["id"])
                    tickets.append(
                        _incident_ticket(row, conn, journal, now))
        else:
            for row in _sql_all(
                    conn,
                    "SELECT id FROM incidents"
                    " WHERE signature NOT LIKE 'heartbeat:%'"
                    " ORDER BY ts DESC LIMIT ?",
                    (int(limit),)):
                full = _incident_row(None, conn, str(row["id"]))
                if full is not None:
                    tickets.append(
                        _incident_ticket(full, conn, None, now))
        if conn is not None:
            for row in _sql_all(
                    conn,
                    "SELECT * FROM recoveries ORDER BY id DESC"
                    " LIMIT ?", (int(limit),)):
                tickets.append(_heal_ticket(row, conn, now))
        else:
            unread.append(
                "healing runs (recoveries have no read path "
                "without the journal's database file)")
    else:
        unread.append("incident journal (no journal or journal "
                      "database supplied)")
    seen_stores: set[int] = set()
    stores: list[Any] = list(antibody_stores or [])
    for key, state in items:
        tickets.append(_state_ticket(key, state, baseline, now))
        for case in getattr(state, "casefiles", None) or []:
            tickets.append(_case_ticket(case, items, now))
        for receipt in getattr(state, "receipts", None) or []:
            tickets.append(_receipt_ticket(receipt, now,
                                           grant_registry))
        store = getattr(state, "store", None)
        if store is not None:
            stores.append(store)
    for store in stores:
        if id(store) in seen_stores:
            continue
        seen_stores.add(id(store))
        try:
            records = store.all()
        except Exception:  # noqa: BLE001 - unreadable store
            records = []
        for record in records[:limit]:
            tickets.append(_antibody_ticket(record, store, now))
    if not items and not antibody_stores:
        unread.append("immune states (none supplied — case files, "
                      "police receipts and state-held antibodies "
                      "are unread)")
    by_source: dict[str, int] = {}
    absent_totals: dict[str, int] = {}
    absent_by_source: dict[str, dict[str, int]] = {}
    detail_rows: list[dict[str, Any]] = []
    claim_grades: dict[str, int] = {}
    complete = 0
    for ticket in tickets:
        by_source[ticket.kind] = by_source.get(ticket.kind, 0) + 1
        per_source = absent_by_source.setdefault(ticket.kind, {})
        for name in ticket.absent_fields():
            absent_totals[name] = absent_totals.get(name, 0) + 1
            per_source[name] = per_source.get(name, 0) + 1
        if ticket.complete():
            complete += 1
        claim = ticket.claim
        if isinstance(claim, dict):
            grade = str(claim.get("grade") or "?")
            claim_grades[grade] = claim_grades.get(grade, 0) + 1
        detail_rows.append({
            "ref": ticket.ref, "kind": ticket.kind,
            "absent_fields": ticket.absent_fields()})
    total = len(tickets)
    summary = [
        f"{total} operations: incidents "
        f"{by_source.get('incident', 0)} · healing runs "
        f"{by_source.get('healing_run', 0)} · immune cases "
        f"{by_source.get('immune_case', 0)} · antibodies "
        f"{by_source.get('antibody', 0)} · police receipts "
        f"{by_source.get('police_receipt', 0)} · immune states "
        f"{by_source.get('immune_state', 0)}",
    ]
    for name in HEALING_FIELDS:
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
    if claim_grades:
        graded = ", ".join(f"{grade} {n}" for grade, n
                           in sorted(claim_grades.items()))
        summary.append(f"effect claims by derived grade: {graded}")
    for note in unread:
        summary.append(f"unread: {note}")
    summary.append(f"complete tickets (no absent field): {complete}")
    return {
        "operations": total,
        "by_source": by_source,
        "complete_tickets": complete,
        "claim_grades": claim_grades,
        "fields_absent": absent_totals,
        "absent_by_source": absent_by_source,
        "unread": unread,
        "operations_detail": detail_rows,
        "summary": summary,
    }
