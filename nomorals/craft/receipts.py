"""The universal receipt schema (L5) — docs 22+23, Stage A P4.

Devon already keeps receipts everywhere — and in at least five
incompatible shapes (the content cell's receipt rows, the outcome
steward's case receipts, the immune police's receipts, the media
mixer's delivery receipts, the community bill-split receipts), so
no cross-cell chain could render against them.  This module is the
**one typed schema** over those stores: :class:`QualityReceipt`
(and :class:`HandoffReceipt` for the accountable-human handoff),
with a read-side adapter per shipped store.  **No store is
migrated and no sixth store exists** — an adapter reads a record
as its own store wrote it and projects it into the schema, the
doc-15/18 projection doctrine applied to receipts.

Receipt law (doc 20's doctrine, enforced here):

* A receipt claims **only recorded outcomes** — every field comes
  from the source record's own fields; an adapter never upgrades,
  rounds or invents.  What the record does not say, the receipt
  does not say.
* Its grade **derives from attached evidence** through
  :mod:`nomorals.core.proof_grade` — the schema stores evidence
  refs, never a grade opinion.  A receipt *asserted* at a grade
  above its evidence is refused at construction
  (:meth:`QualityReceipt.asserting`), and a stored grade above its
  evidence is refused at read (:meth:`QualityReceipt.from_dict`) —
  the no-silent-upgrade law, inherited from PG1 rather than
  re-invented.

The evidence each adapter attaches is deliberately conservative —
the mapping is documented per adapter below, in PG1's style
(honesty over completeness): every adapted record earns the G1
floor (it exists, schema-shaped, in its shipped store — a
logical/schema check), and stronger kinds attach only where the
record itself carries the stronger fact (a post reference, a
performed effect, a sent timestamp).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..core.proof_grade import (
    ClaimGrade,
    EvidenceRef,
    ProofGrade,
    ProofGradeError,
    grade_of,
)

__all__ = [
    "QualityReceipt",
    "HandoffReceipt",
    "handoff_for",
    "adapt_content_receipt",
    "adapt_case_receipt",
    "adapt_police_receipt",
    "adapt_delivery_receipt",
    "adapt_bill_receipt",
    "render_receipt_chain",
    "SOURCE_NAMES",
]

#: Source keys → the organ name the chain render shows.
SOURCE_NAMES: dict[str, str] = {
    "content_cell": "content cell",
    "outcome_case": "outcome case",
    "immune_police": "immune police",
    "media_delivery": "media delivery",
    "community_bill": "community bill",
    # Stage B pilots: the record each pilot's receipt derives from.
    "delegation_contract": "delegation contract",
    "media_job": "media job",
    "cbt_intervention": "CBT intervention",
}


@dataclass
class HandoffReceipt:
    """Recorded proof that work changed hands to an accountable
    human (or back).  A handoff claims only that the underlying
    recorded work was handed over — it rides on a quality receipt's
    identity (``basis``) and cannot exist without one."""

    from_actor: str
    to_actor: str
    subject: str
    basis: str            # "<source>:<source_id>" of the receipt
    recorded_at: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.basis or "").strip():
            raise ProofGradeError(
                "a handoff receipt needs the recorded work it "
                "rides on (basis) — a handoff of nothing recorded "
                "is not a receipt")
        if not str(self.to_actor or "").strip():
            raise ProofGradeError(
                "a handoff receipt names who the work went to")

    def to_dict(self) -> dict[str, Any]:
        return {"from_actor": self.from_actor,
                "to_actor": self.to_actor,
                "subject": self.subject,
                "basis": self.basis,
                "recorded_at": self.recorded_at,
                "note": self.note}

    @classmethod
    def from_dict(cls, data: Any) -> HandoffReceipt:
        if not isinstance(data, dict):
            raise ProofGradeError(
                f"handoff entry is not a record: {data!r}")
        return cls(from_actor=str(data.get("from_actor") or ""),
                   to_actor=str(data.get("to_actor") or ""),
                   subject=str(data.get("subject") or ""),
                   basis=str(data.get("basis") or ""),
                   recorded_at=float(data.get("recorded_at") or 0.0),
                   note=str(data.get("note") or ""))


@dataclass
class QualityReceipt:
    """One recorded outcome, in the universal shape.

    ``source``/``source_id`` identify the record in its own store;
    ``kind``/``outcome`` are the source record's own words;
    ``evidence`` is the attached proof the grade derives from.
    There is deliberately **no grade field** — :attr:`grade` is
    derived on every read, exactly like PG1's ``ClaimGrade``.
    """

    source: str
    source_id: str
    kind: str = ""
    subject: str = ""
    actor: str = ""
    outcome: str = ""
    recorded_at: float = 0.0
    evidence: tuple[EvidenceRef, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)
    handoff: HandoffReceipt | None = None

    @property
    def grade(self) -> ProofGrade:
        """Derived, every read: the grade of the strongest
        evidence attached."""
        return grade_of(self.evidence)

    @property
    def claim(self) -> ClaimGrade:
        return ClaimGrade(evidence=tuple(self.evidence))

    @property
    def basis(self) -> str:
        """The identity a handoff rides on: ``<source>:<id>``."""
        return f"{self.source}:{self.source_id}"

    @classmethod
    def asserting(cls, grade: Any, **fields: Any) -> QualityReceipt:
        """State a receipt AT a grade, with its evidence.  Refused
        when the demanded grade exceeds what the evidence earns —
        the receipt then does not exist, by design."""
        evidence = tuple(fields.get("evidence") or ())
        ClaimGrade.asserting(grade, evidence)
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "kind": self.kind,
            "subject": self.subject,
            "actor": self.actor,
            "outcome": self.outcome,
            "recorded_at": self.recorded_at,
            "grade": self.grade.name,
            "evidence": [item.to_dict() for item in self.evidence],
            "detail": dict(self.detail),
            "handoff": self.handoff.to_dict() if self.handoff else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> QualityReceipt:
        """Read a stored receipt.  The stored grade token is a
        claim about the derivation and is checked, never trusted
        (via :meth:`ClaimGrade.from_dict`): above the evidence it
        travels with → refused; at or below → the record stands on
        its evidence."""
        if not isinstance(data, dict):
            raise ProofGradeError(
                f"quality receipt payload is not a record: {data!r}")
        claim = ClaimGrade.from_dict({
            "grade": data.get("grade"),
            "evidence": data.get("evidence") or [],
        })
        handoff = data.get("handoff")
        return cls(
            source=str(data.get("source") or ""),
            source_id=str(data.get("source_id") or ""),
            kind=str(data.get("kind") or ""),
            subject=str(data.get("subject") or ""),
            actor=str(data.get("actor") or ""),
            outcome=str(data.get("outcome") or ""),
            recorded_at=float(data.get("recorded_at") or 0.0),
            evidence=claim.evidence,
            detail=dict(data.get("detail") or {}),
            handoff=HandoffReceipt.from_dict(handoff)
            if handoff else None,
        )


def handoff_for(receipt: QualityReceipt, *, to_actor: str,
                from_actor: str = "", note: str = "",
                recorded_at: float | None = None) -> HandoffReceipt:
    """Mint the handoff receipt for recorded work changing hands.

    The handoff rides on the quality receipt's basis — unrecorded
    work cannot be handed off, because there is nothing a receiver
    could later check the handoff against.
    """
    return HandoffReceipt(
        from_actor=from_actor or receipt.actor,
        to_actor=to_actor,
        subject=receipt.subject,
        basis=receipt.basis,
        recorded_at=(time.time() if recorded_at is None
                     else recorded_at),
        note=note)


# ── adapters: one per shipped receipt store, read-side only ──────


def _floor(source: str, source_id: str) -> EvidenceRef:
    """The G1 every recorded receipt earns: the record exists,
    schema-shaped, in its shipped store (a logical/schema check)."""
    return EvidenceRef("logical_check", f"{source}:{source_id}")


def adapt_content_receipt(row: dict[str, Any]) -> QualityReceipt:
    """A content-cell receipt row (``ReceiptStore`` dict shape).

    Evidence: the floor, plus — when the row carries a post
    reference — an empirical ref to the post itself: a primary
    record of a real-world delivery (G4).  A row without a post
    reference records a drafted piece and earns only the floor;
    its outcome says "drafted", never "delivered".
    """
    source_id = str(row.get("id") or "")
    evidence = [_floor("content_cell", source_id)]
    post_ref = str(row.get("post_ref") or "")
    if post_ref:
        evidence.append(EvidenceRef("empirical_source", post_ref))
    return QualityReceipt(
        source="content_cell",
        source_id=source_id,
        kind=str(row.get("kind") or ""),
        subject=str(row.get("label") or row.get("kind") or ""),
        outcome="delivered" if post_ref else "drafted",
        recorded_at=float(row.get("created_at") or 0.0),
        evidence=tuple(evidence),
        detail={"platform": str(row.get("platform") or ""),
                "contract_id": str(row.get("contract_id") or ""),
                "draft_id": str(row.get("draft_id") or ""),
                "content_sha": str(row.get("content_sha") or "")},
    )


def adapt_case_receipt(receipt: Any, case: Any = None) -> QualityReceipt:
    """An outcome steward's case receipt (``CaseReceipt``).

    Evidence: the floor, plus — when the receipt's own evidence
    record is non-empty — an empirical ref to the receipt: the
    steward's receipts are investigation records over real
    operations (G4).  This grades the *receipt's evidence*, never
    the case's verdict: PG1 already ruled steward verdicts are an
    outcome vocabulary and do not map onto the proof ladder.
    """
    source_id = str(getattr(receipt, "id", "") or "")
    evidence = [_floor("outcome_case", source_id)]
    recorded = getattr(receipt, "evidence", None) or {}
    if recorded:
        evidence.append(EvidenceRef(
            "empirical_source", f"outcome_case_receipt:{source_id}"))
    subject = ""
    if case is not None:
        subject = str(getattr(case, "subject", "") or "")
    return QualityReceipt(
        source="outcome_case",
        source_id=source_id,
        kind=str(getattr(receipt, "kind", "") or ""),
        subject=subject or str(getattr(receipt, "note", "") or "")[:80],
        actor=str(getattr(receipt, "actor", "") or ""),
        outcome=str(getattr(receipt, "kind", "") or ""),
        recorded_at=float(getattr(receipt, "created_at", 0.0) or 0.0),
        evidence=tuple(evidence),
        detail={"case_id": str(getattr(receipt, "case_id", "") or ""),
                "source": str(getattr(receipt, "source", "") or ""),
                "generation": int(getattr(receipt, "generation", 0)
                                  or 0)},
    )


def adapt_police_receipt(receipt: Any) -> QualityReceipt:
    """An immune police receipt (``immune.police.Receipt``).

    Evidence: the floor, plus — when the receipt's details record a
    *performed* effect — an artifact-pass ref (G5): an operation
    that ran and was recorded as performed.  A *partial* effect
    earns the empirical ref (G4).  A report-only receipt (the
    default rung) earns the floor alone; its outcome stays the
    record's own action word.
    """
    source_id = str(getattr(receipt, "id", "") or "")
    evidence = [_floor("immune_police", source_id)]
    details = getattr(receipt, "details", None) or {}
    effect = str(details.get("effect") or "")
    if effect == "performed":
        evidence.append(EvidenceRef(
            "artifact_pass", f"police_receipt:{source_id}"))
    elif effect == "partial":
        evidence.append(EvidenceRef(
            "empirical_source", f"police_receipt:{source_id}"))
    return QualityReceipt(
        source="immune_police",
        source_id=source_id,
        kind=str(getattr(receipt, "action", "") or ""),
        subject=str(getattr(receipt, "target", "") or ""),
        actor=str(getattr(receipt, "actor", "") or ""),
        outcome=str(getattr(receipt, "action", "") or ""),
        recorded_at=float(getattr(receipt, "ts", 0.0) or 0.0),
        evidence=tuple(evidence),
        detail={"rung": int(getattr(receipt, "rung", 0) or 0),
                "reason": str(getattr(receipt, "reason", "") or ""),
                "reversible": bool(getattr(receipt, "reversible",
                                           True))},
    )


def adapt_delivery_receipt(receipt: Any) -> QualityReceipt:
    """A media delivery receipt (``media.mix_surface`` shape).

    Evidence: the floor, plus — when the receipt records a send
    (``sent_at`` set) — an empirical ref to the rendered mix's
    fingerprint (G4): the primary record of a real delivery.  An
    unsent receipt earns the floor alone and says "unsent".
    """
    mix_id = str(getattr(receipt, "mix_id", "") or "")
    fingerprint = str(getattr(receipt, "fingerprint", "") or "")
    sent_at = float(getattr(receipt, "sent_at", 0.0) or 0.0)
    evidence = [_floor("media_delivery", mix_id or fingerprint)]
    if sent_at > 0:
        evidence.append(EvidenceRef(
            "empirical_source",
            f"media_delivery:{fingerprint or mix_id}"))
    return QualityReceipt(
        source="media_delivery",
        source_id=mix_id or fingerprint,
        kind=str(getattr(receipt, "kind", "") or ""),
        subject=fingerprint or mix_id,
        outcome="sent" if sent_at > 0 else "unsent",
        recorded_at=sent_at,
        evidence=tuple(evidence),
        detail={"chat": str(getattr(receipt, "chat", "") or ""),
                "audio_path": str(getattr(receipt, "audio_path", "")
                                  or "")},
    )


def adapt_bill_receipt(receipt: Any) -> QualityReceipt:
    """A community bill-split receipt (``community.miniapps2``).

    Evidence: the floor alone.  A bill split is bookkeeping — the
    record proves the split was recorded, and nothing stronger is
    claimed for it.
    """
    source_id = str(getattr(receipt, "id", "") or "")
    shares = getattr(receipt, "shares", None) or {}
    return QualityReceipt(
        source="community_bill",
        source_id=source_id,
        kind="bill_split",
        subject=str(getattr(receipt, "description", "") or "")
        or "bill split",
        actor=str(getattr(receipt, "payer_name", "") or ""),
        outcome="recorded",
        recorded_at=float(getattr(receipt, "created_ts", 0.0)
                          or 0.0),
        evidence=(_floor("community_bill", source_id),),
        detail={"total_cents": int(getattr(receipt, "total_cents", 0)
                                   or 0),
                "shares": len(shares)},
    )


# ── the chain render: one shape, every source ────────────────────


def render_receipt_chain(receipts: Iterable[QualityReceipt], *,
                         title: str = "receipt chain") -> str:
    """Render receipts from any mix of stores in ONE shape, oldest
    first — the cross-cell chain the fragmented stores could not
    produce.  Subjects, kinds and grades only: no internal ids (the
    doc-15 surface law), and every line carries the grade its
    evidence derived, with the evidence kinds named.
    """
    rows = sorted(receipts, key=lambda r: (r.recorded_at, r.source))
    lines = [f"{title} — one shape, every source"]
    if not rows:
        lines.append("no receipts recorded yet")
        return "\n".join(lines)
    for idx, receipt in enumerate(rows, 1):
        when = time.strftime(
            "%H:%M", time.localtime(receipt.recorded_at)) \
            if receipt.recorded_at else "--:--"
        organ = SOURCE_NAMES.get(receipt.source, receipt.source)
        subject = receipt.subject or receipt.kind or "record"
        kinds = ", ".join(item.kind for item in receipt.evidence)
        lines.append(
            f"{idx}. {when} · {organ} · {subject} — "
            f"{receipt.outcome or receipt.kind} · grade "
            f"{receipt.grade.name} ({kinds})")
        if receipt.handoff is not None:
            handoff = receipt.handoff
            line = f"   ↳ handed to {handoff.to_actor}"
            if handoff.from_actor:
                line += f" by {handoff.from_actor}"
            if handoff.note:
                line += f" — {handoff.note}"
            lines.append(line)
    return "\n".join(lines)
