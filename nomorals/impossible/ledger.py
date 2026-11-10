"""Stage 10 — HONESTY LEDGER. Every claim gets a record; conflicts surface,
never averaged. A claim is never promoted past its evidence.

Backed by a JSON file (durable, human-readable, no new DB). Mirrors the
ResearchClaim conflict discipline in agents/research_digest.py: contradictions
are detected and recorded, not smoothed away.

PG2 (Doc 24) — PROMOTION ENFORCEMENT. Until this pass, ``record`` and
``record_attempt(new_status=...)`` stored ANY status string a caller
passed: a claim could be relabelled ``measured`` by fiat, with no
evidence, and every reader believed the label. Now the write boundary
enforces the charter's own lifecycle (``charter.STATUSES``) against
PG1's proof grades (``core/proof_grade.py``):

* the status vocabulary is CLOSED — a status outside ``STATUSES`` is
  refused at write with :class:`LedgerStatusError`, never stored;
* a status is recorded only as high as its evidence earns. Each status
  has an evidence floor (PG1's default grade reading of the status);
  the evidence attached to the write — typed evidence refs, graded by
  kind — determines the EARNED status. Asking above the evidence
  stores the earned status and preserves the request as a CLAIM on
  the record: every status event appends to ``status_audit`` with
  ``requested`` / ``earned`` / ``applied`` / the evidence refs / the
  derived grade, so an auditor can see every unearned claim and what
  it earned;
* a claim recorded at ``hypothesis`` earns that rung from its own
  formalization — a hypothesis naming its falsifier IS the logical
  check (G1) PG1's mapping row describes — recorded as an intrinsic
  evidence ref on the audit, not as an invisible exception;
* readers read the EARNED status: :func:`earned_status_of` derives a
  record's standing from its audit trail (PG2 records) or, for legacy
  records written before enforcement, from the record's intrinsic
  evidence alone — a legacy label with no evidence behind it is not
  inherited as fact.

The grade tables below MIRROR ``core/proof_grade.py``
(``EVIDENCE_KIND_GRADES`` and ``IMPOSSIBLE_STATUS_MAPPINGS``): this
package is self-contained (layer 0) and may not import core (L1) —
the same mirror discipline PG1 itself uses for the vocabularies it
cannot import. ``tests/test_d24pg2_ledger_enforcement.py`` pins the
mirrors equal to the real tables, so the two can never drift apart
silently.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .charter import STATUSES

DEFAULT_PATH = Path("data/impossible_ledger.json")


def _default_path() -> Path:
    return DEFAULT_PATH


class LedgerStatusError(ValueError):
    """A status or evidence ref was refused at the ledger's write
    boundary: a status outside the charter lifecycle, or evidence
    whose kind PG1's vocabulary does not know. Raised, never silently
    stored or coerced — the closed vocabulary fails closed."""


# ── PG1 mirrors (pinned equal to core/proof_grade.py by the PG2
#    suite — see the module docstring for why these are mirrors) ────

#: Mirror of proof_grade.EVIDENCE_KIND_GRADES: evidence kind -> the
#: grade (0-6) that kind of evidence earns.
_KIND_GRADES: dict[str, int] = {
    "generative_thought": 0,
    "logical_check": 1,
    "derivation": 2,
    "simulation_check": 3,
    "empirical_source": 4,
    "artifact_pass": 5,
    "monitoring_receipt": 6,
}

#: Mirror of proof_grade's per-status grades
#: (grade_for_impossible_status over charter.STATUSES): the evidence
#: floor a status claims. idea G0, hypothesis G1, tested G3,
#: prototype G5, built G5, measured G6.
_STATUS_FLOORS: dict[str, int] = {
    "idea": 0,
    "hypothesis": 1,
    "tested": 3,
    "prototype": 5,
    "built": 5,
    "measured": 6,
}


def _grade_name(grade: int) -> str:
    return f"G{grade}"


def _normalize_refs(evidence_refs: Iterable[Any] | None,
                    ) -> list[dict[str, str]]:
    """Validate + normalize caller-supplied evidence refs to plain
    ``{"kind", "ref"}`` dicts for the JSON record. Accepts EvidenceRef
    objects (duck-typed: anything with ``kind``/``ref`` attributes —
    this module cannot import core's class), mappings, or (kind, ref)
    pairs. An unknown kind or an empty ref is refused: ungradeable
    material is not evidence."""
    refs: list[dict[str, str]] = []
    for item in evidence_refs or ():
        if isinstance(item, Mapping):
            kind, ref = item.get("kind"), item.get("ref")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            kind, ref = item
        else:
            kind = getattr(item, "kind", None)
            ref = getattr(item, "ref", None)
        if not isinstance(kind, str) or kind not in _KIND_GRADES:
            raise LedgerStatusError(
                f"unknown evidence kind: {kind!r} (known: "
                f"{', '.join(sorted(_KIND_GRADES))})")
        if not isinstance(ref, str) or not ref.strip():
            raise LedgerStatusError(
                "an evidence ref must name where the evidence lives")
        refs.append({"kind": kind, "ref": ref})
    return refs


def _grade_of(refs: Iterable[Mapping[str, Any]]) -> int:
    """The grade of the strongest evidence attached — 0 for none.
    Mirrors proof_grade.grade_of over the normalized refs."""
    strongest = 0
    for ref in refs:
        grade = _KIND_GRADES.get(str(ref.get("kind")), 0)
        if grade > strongest:
            strongest = grade
    return strongest


def _highest_earnable(grade: int) -> str:
    """The highest lifecycle status whose evidence floor ``grade``
    meets. Floors are monotonic along STATUSES, so the ladder is
    climbed rung by rung — there is no rung an unearned grade can
    skip to."""
    earned = STATUSES[0]
    for status in STATUSES:
        if _STATUS_FLOORS[status] <= grade:
            earned = status
    return earned


def _intrinsic_refs(claim_id: str, falsifier: str) -> list[dict[str, str]]:
    """The evidence a record carries by its own construction: a
    formalized hypothesis naming its falsifier is the logical check
    (G1) PG1's hypothesis row describes."""
    if falsifier and falsifier.strip():
        return [{"kind": "logical_check",
                 "ref": f"claim:{claim_id}:hypothesis"}]
    return []


def earned_status_of(entry: Mapping[str, Any]) -> str:
    """The status a stored record has EARNED — the only status a
    reader should surface as the claim's standing.

    PG2 records carry a ``status_audit`` trail and store only earned
    values in ``status``, so the stored status is the earned one.
    Legacy records (written before enforcement, no audit trail) are
    derived from the record's intrinsic evidence alone: the label a
    legacy write asserted by fiat is not inherited — a legacy claim
    whose hypothesis names its falsifier stands at ``hypothesis``,
    anything else at ``idea``."""
    raw = str(entry.get("status") or "")
    if entry.get("status_audit"):
        return raw if raw in STATUSES else STATUSES[0]
    falsifier = str(entry.get("falsifier") or "")
    if not falsifier:
        hypothesis = entry.get("hypothesis")
        if isinstance(hypothesis, Mapping):
            falsifier = str(hypothesis.get("falsifier") or "")
    return _highest_earnable(_grade_of(_intrinsic_refs(
        str(entry.get("id") or ""), falsifier)))


def _check_status(status: Any) -> str:
    if status not in STATUSES:
        raise LedgerStatusError(
            f"{status!r} is not a claim status (closed lifecycle: "
            f"{', '.join(STATUSES)})")
    return str(status)


class HonestyLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"claims": []}, indent=2), encoding="utf-8")

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {"claims": []}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, claim: str, formulation: dict | None = None, classification: dict | None = None,
               prior_art: dict | None = None, transfer: dict | None = None,
               gate: dict | None = None, hypothesis: dict | None = None,
               reframed: dict | None = None, status: str = "hypothesis",
               evidence_refs: Iterable[Any] | None = None) -> dict:
        """Record a claim. ``status`` is a CLAIM, gated like every
        status write (see the module docstring): refused outright if
        outside the lifecycle; recorded at the highest status the
        attached evidence — plus the record's own formalization —
        earns, with the request preserved on the record's
        ``requested_status`` + ``status_audit``."""
        _check_status(status)
        claim_id = hashlib.sha256(
            f"{claim}|{time.time()}".encode()).hexdigest()[:12]
        refs = (_intrinsic_refs(claim_id, (hypothesis or {}).get(
            "falsifier", "")) + _normalize_refs(evidence_refs))
        grade = _grade_of(refs)
        earned = _highest_earnable(grade)
        applied = STATUSES[min(STATUSES.index(status),
                               STATUSES.index(earned))]
        entry = {
            "id": claim_id,
            "claim": claim,
            "formulation": formulation or {},
            "classification": classification or {},
            "prior_art_summary": (prior_art or {}).get("summary", ""),
            "transfer_methods": list((transfer or {}).keys()),
            "gate": {"verdict": (gate or {}).get("verdict"), "why": (gate or {}).get("why"),
                     "failed": (gate or {}).get("failed", [])},
            "hypothesis": hypothesis or {},
            "reframed": reframed or {},
            "status": applied,
            "requested_status": status,
            "status_audit": [{
                "action": "record", "requested": status,
                "earned": earned, "applied": applied,
                "grade": _grade_name(grade), "evidence": refs,
                "at": time.time()}],
            "falsifier": (hypothesis or {}).get("falsifier", ""),
            "attempts": [],
            "conflicts": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "checked_by": "impossible-gate",
            "what_would_change_my_mind": "A test result contradicting the gate, or prior art showing the exact thing already built/failed.",
        }
        data = self._load()
        data["claims"].append(entry)
        self._save(data)
        return entry

    def get(self, claim_id: str) -> dict | None:
        for c in self._load().get("claims", []):
            if c["id"] == claim_id or c["id"].startswith(claim_id):
                return c
        return None

    def list(self, status: str | None = None) -> list[dict]:
        claims = self._load().get("claims", [])
        return [c for c in claims
                if status is None or earned_status_of(c) == status]

    def record_attempt(self, claim_id: str, outcome: str, evidence: str,
                       lesson: str, new_status: str | None = None,
                       evidence_refs: Iterable[Any] | None = None) -> dict | None:
        """Log an attempt + its lesson. Failed attempts change the next attempt:
        the lesson is stored on the claim and surfaced by next_attempt_hint().

        ``new_status`` is a promotion CLAIM, gated by PG2 enforcement:
        refused outright if outside the lifecycle (nothing is stored,
        the attempt included); otherwise the claim moves only as high
        as this attempt's ``evidence_refs`` earn — never below its
        current standing on a failed promotion, and down only when a
        lower status is explicitly requested. The event (requested /
        earned / applied / grade / evidence) is appended to the
        claim's ``status_audit`` and to the attempt record itself."""
        if new_status is not None:
            _check_status(new_status)
        refs = _normalize_refs(evidence_refs)
        data = self._load()
        for c in data.get("claims", []):
            if c["id"] == claim_id or c["id"].startswith(claim_id):
                attempt: dict[str, Any] = {
                    "outcome": outcome, "evidence": evidence,
                    "lesson": lesson, "at": time.time(),
                    "evidence_refs": refs}
                if new_status is not None:
                    grade = _grade_of(refs)
                    earned = _highest_earnable(grade)
                    current = earned_status_of(c)
                    req_i = STATUSES.index(new_status)
                    cur_i = STATUSES.index(current)
                    if req_i <= cur_i:
                        # An explicit step down (or a re-assertion of
                        # the current standing) is always honest.
                        applied = new_status
                    else:
                        applied = STATUSES[min(
                            req_i, max(cur_i, STATUSES.index(earned)))]
                    attempt["requested_status"] = new_status
                    attempt["applied_status"] = applied
                    c["status"] = applied
                    c["requested_status"] = new_status
                    c.setdefault("status_audit", []).append({
                        "action": "attempt", "requested": new_status,
                        "earned": earned, "applied": applied,
                        "grade": _grade_name(grade), "evidence": refs,
                        "at": attempt["at"]})
                c["attempts"].append(attempt)
                c["updated_at"] = time.time()
                self._save(data)
                return c
        return None

    def add_conflict(self, claim_id: str, contradicting: str, evidence: str) -> dict | None:
        """Surface a contradiction — never average it away."""
        data = self._load()
        for c in data.get("claims", []):
            if c["id"] == claim_id or c["id"].startswith(claim_id):
                c["conflicts"].append({"contradicting": contradicting, "evidence": evidence, "at": time.time()})
                self._save(data)
                return c
        return None

    def record_digestion(self, claim_id: str, sources_read: list, hits: int,
                         summary: str) -> dict | None:
        """Record which digestion sources were read for a claim this tick.
        The digest is question-led; this is the audit trail of what it read."""
        data = self._load()
        for c in data.get("claims", []):
            if c["id"] == claim_id or c["id"].startswith(claim_id):
                c.setdefault("digestion_log", []).append(
                    {"sources_read": list(sources_read), "hits": hits,
                     "summary": summary, "at": time.time()})
                c["updated_at"] = time.time()
                self._save(data)
                return c
        return None

    def next_attempt_hint(self, claim_id: str) -> str:
        c = self.get(claim_id)
        if not c:
            return ""
        lessons = [a.get("lesson", "") for a in c.get("attempts", []) if a.get("lesson")]
        if not lessons:
            return "No attempts yet — run the cheapest killing experiment first."
        return "Learned from past attempts (do not repeat these): " + " | ".join(lessons)
