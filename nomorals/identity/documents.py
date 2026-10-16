"""Identity documents — expiry tracking over vault references.

The owner keeps identity documents (passport, visa, licence, id card,
permit, other) and wants to know *when* each one expires and what
renewal involves — without this module ever holding the document
number itself.

Security contract (enforced in code, not in prompts):

* ``number_ref`` is a *vault reference string* (``"vault:..."``) —
  NEVER the raw document number. The tracker stores only the
  reference; it has no field for a raw number anywhere.
* A ``number_ref`` that does not look like a reference (does not start
  with ``"vault:"``, or is empty/blank after the prefix) is rejected
  with :class:`ValueError`. A raw-number-looking value (digits, spaces
  and dashes, no ``vault:`` prefix) therefore cannot be stored.
* Owner-only: every public method requires ``owner=True`` (keyword),
  otherwise :class:`PermissionError`. The default is ``False`` so a
  caller that forgets the flag is denied, not silently admitted.

Storage: SQLite, ``:memory:`` by default. Pass a file path for a
durable tracker. Universal: ``issuing_country`` is free text — any
country works through the same code paths; no country is a boundary.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DOC_TYPES",
    "RENEWAL_CHECKLISTS",
    "IdentityDocument",
    "IdentityTracker",
    "validate_number_ref",
]

#: Allowed document types.
DOC_TYPES: tuple[str, ...] = (
    "passport", "visa", "licence", "id_card", "permit", "other",
)

#: Renewal steps as data, per document type. Each step is a dict with
#: ``order``, ``title`` and ``detail`` — callers render them; this
#: module never formats user-facing prose from them.
RENEWAL_CHECKLISTS: dict[str, list[dict[str, Any]]] = {
    "passport": [
        {"order": 1, "title": "Confirm expiry and eligibility",
         "detail": "Check the expiry date and whether renewal (vs a new application) applies."},
        {"order": 2, "title": "Gather supporting documents",
         "detail": "Collect the current document, proof of identity and any required photos/forms."},
        {"order": 3, "title": "Submit the renewal application",
         "detail": "File with the issuing authority and keep the receipt / tracking reference."},
        {"order": 4, "title": "Track issuance and collect",
         "detail": "Follow the application status and collect the new document when ready."},
    ],
    "visa": [
        {"order": 1, "title": "Confirm expiry and stay conditions",
         "detail": "Check the expiry date and the conditions attached to the current visa."},
        {"order": 2, "title": "Gather supporting documents",
         "detail": "Collect passport, proof of purpose, funds and any sponsor documents."},
        {"order": 3, "title": "Submit the renewal / extension application",
         "detail": "File before expiry where the rules require it; keep the receipt."},
        {"order": 4, "title": "Track the decision",
         "detail": "Follow the application status and record the new expiry when granted."},
    ],
    "licence": [
        {"order": 1, "title": "Confirm expiry",
         "detail": "Check the expiry date printed on the licence."},
        {"order": 2, "title": "Gather supporting documents",
         "detail": "Collect the current licence, proof of identity and any medical/test evidence required."},
        {"order": 3, "title": "Submit the renewal application",
         "detail": "File with the issuing authority and keep the receipt."},
        {"order": 4, "title": "Collect the renewed licence",
         "detail": "Collect the new licence and update the stored expiry."},
    ],
    "id_card": [
        {"order": 1, "title": "Confirm expiry",
         "detail": "Check the expiry date on the card."},
        {"order": 2, "title": "Gather supporting documents",
         "detail": "Collect proof of identity and address as required by the issuer."},
        {"order": 3, "title": "Submit the renewal application",
         "detail": "File with the issuing authority and keep the receipt."},
        {"order": 4, "title": "Collect the renewed card",
         "detail": "Collect the new card and update the stored expiry."},
    ],
    "permit": [
        {"order": 1, "title": "Confirm expiry and conditions",
         "detail": "Check the expiry date and the conditions attached to the permit."},
        {"order": 2, "title": "Gather supporting documents",
         "detail": "Collect the current permit and any supporting evidence the issuer requires."},
        {"order": 3, "title": "Submit the renewal application",
         "detail": "File before expiry where the rules require it; keep the receipt."},
        {"order": 4, "title": "Track the decision",
         "detail": "Follow the application status and record the new expiry when granted."},
    ],
    "other": [
        {"order": 1, "title": "Confirm expiry",
         "detail": "Check the expiry date on the document."},
        {"order": 2, "title": "Gather supporting documents",
         "detail": "Collect whatever the issuing authority requires for renewal."},
        {"order": 3, "title": "Submit the renewal application",
         "detail": "File with the issuing authority and keep the receipt."},
        {"order": 4, "title": "Record the new expiry",
         "detail": "Update the stored expiry when the renewed document arrives."},
    ],
}


def validate_number_ref(number_ref: str) -> str:
    """Validate a vault reference; return it stripped.

    Raises :class:`ValueError` unless the value looks like a
    reference (``"vault:"`` prefix + non-empty remainder, no
    whitespace). Raw numbers never satisfy this shape.
    """
    ref = (number_ref or "").strip()
    if not ref.startswith("vault:"):
        raise ValueError(
            "number_ref must be a vault reference starting with 'vault:' — "
            "raw document numbers are never stored")
    remainder = ref[len("vault:"):].strip()
    if not remainder:
        raise ValueError("number_ref must name a vault entry after 'vault:'")
    if any(ch.isspace() for ch in ref):
        raise ValueError("number_ref must not contain whitespace")
    return ref


def _require_owner(owner: bool) -> None:
    if owner is not True:
        raise PermissionError("identity documents are owner-only")


@dataclass
class IdentityDocument:
    """One identity document. ``number_ref`` is a vault reference only."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    owner_id: str = ""
    doc_type: str = "other"
    issuing_country: str = ""
    number_ref: str = ""
    expiry_ts: float = 0.0
    issued_ts: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("owner_id is required")
        doc_type = (self.doc_type or "").strip().lower()
        if doc_type not in DOC_TYPES:
            raise ValueError(f"unknown doc_type {self.doc_type!r}; expected one of {DOC_TYPES}")
        self.doc_type = doc_type
        # Validates the reference shape; raw numbers raise here.
        self.number_ref = validate_number_ref(self.number_ref)
        if self.expiry_ts <= 0:
            raise ValueError("expiry_ts is required (epoch seconds)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "IdentityDocument":
        return cls(
            id=str(row.get("id", "")),
            owner_id=str(row.get("owner_id", "")),
            doc_type=str(row.get("doc_type", "other")),
            issuing_country=str(row.get("issuing_country", "")),
            number_ref=str(row.get("number_ref", "")),
            expiry_ts=float(row.get("expiry_ts", 0.0)),
            issued_ts=float(row.get("issued_ts", 0.0)),
            notes=str(row.get("notes", "")),
        )


class IdentityTracker:
    """SQLite tracker for identity documents (vault references only)."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._db = sqlite3.connect(db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS identity_documents (
                   id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                   doc_type TEXT NOT NULL, issuing_country TEXT NOT NULL,
                   number_ref TEXT NOT NULL, expiry_ts REAL NOT NULL,
                   issued_ts REAL NOT NULL, notes TEXT NOT NULL)""")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_identity_expiry "
            "ON identity_documents(expiry_ts)")
        self._db.commit()

    # ── CRUD ─────────────────────────────────────────────────────
    def add(self, doc: IdentityDocument, *, owner: bool = False) -> IdentityDocument:
        """Store a document. Re-validates the reference on the way in."""
        _require_owner(owner)
        if not isinstance(doc, IdentityDocument):
            raise TypeError("add() expects an IdentityDocument")
        # Re-validate: a document built elsewhere must still pass.
        doc.number_ref = validate_number_ref(doc.number_ref)
        existing = self._db.execute(
            "SELECT id FROM identity_documents WHERE id = ?", (doc.id,)).fetchone()
        if existing is not None:
            raise ValueError(f"document {doc.id!r} already exists")
        self._db.execute(
            "INSERT INTO identity_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc.id, doc.owner_id, doc.doc_type, doc.issuing_country,
             doc.number_ref, doc.expiry_ts, doc.issued_ts, doc.notes))
        self._db.commit()
        return doc

    def get(self, doc_id: str, *, owner: bool = False) -> IdentityDocument | None:
        """Fetch one document by id, or ``None`` when absent."""
        _require_owner(owner)
        row = self._db.execute(
            "SELECT * FROM identity_documents WHERE id = ?", (doc_id,)).fetchone()
        return IdentityDocument.from_dict(dict(row)) if row is not None else None

    def list(self, owner_id: str | None = None, *, owner: bool = False) -> list[IdentityDocument]:
        """All documents (optionally filtered by ``owner_id``), by expiry."""
        _require_owner(owner)
        if owner_id is None:
            rows = self._db.execute(
                "SELECT * FROM identity_documents ORDER BY expiry_ts ASC").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM identity_documents WHERE owner_id = ? "
                "ORDER BY expiry_ts ASC", (owner_id,)).fetchall()
        return [IdentityDocument.from_dict(dict(r)) for r in rows]

    # ── expiry queries ───────────────────────────────────────────
    def expiring_within(self, days: float, now: float | None = None, *,
                        owner: bool = False) -> list[IdentityDocument]:
        """Documents expiring within ``days`` from ``now``, by expiry.

        Includes already-expired documents (they are within any
        forward window in the sense that they need action now); use
        :meth:`overdue` for the expired-only subset.
        """
        _require_owner(owner)
        now = float(now if now is not None else time.time())
        horizon = now + float(days) * 86400.0
        rows = self._db.execute(
            "SELECT * FROM identity_documents WHERE expiry_ts <= ? "
            "ORDER BY expiry_ts ASC", (horizon,)).fetchall()
        return [IdentityDocument.from_dict(dict(r)) for r in rows]

    def overdue(self, now: float | None = None, *,
                owner: bool = False) -> list[IdentityDocument]:
        """Documents whose expiry is before ``now``, by expiry."""
        _require_owner(owner)
        now = float(now if now is not None else time.time())
        rows = self._db.execute(
            "SELECT * FROM identity_documents WHERE expiry_ts < ? "
            "ORDER BY expiry_ts ASC", (now,)).fetchall()
        return [IdentityDocument.from_dict(dict(r)) for r in rows]

    # ── renewal ──────────────────────────────────────────────────
    def renewal_checklist(self, doc: IdentityDocument | str, *,
                          owner: bool = False) -> list[dict[str, Any]]:
        """Renewal steps (as data) for a document's type.

        Accepts a document or a document id (looked up first).
        Returns a fresh list of step dicts, ordered by ``order``.
        """
        _require_owner(owner)
        if isinstance(doc, str):
            found = self.get(doc, owner=True)
            if found is None:
                raise KeyError(f"no identity document {doc!r}")
            doc = found
        steps = RENEWAL_CHECKLISTS.get(doc.doc_type, RENEWAL_CHECKLISTS["other"])
        return [dict(s) for s in sorted(steps, key=lambda s: s["order"])]

    def close(self) -> None:
        self._db.close()
