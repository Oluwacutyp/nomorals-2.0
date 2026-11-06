"""The Evidence core — Devon's reverse-engineering evidence store.

REA's real product was never its decompilers; it was the discipline
every result is forced through. This module adopts that record
wholesale and puts it in Devon's own storage:

* **Evidence** — one immutable, content-addressed record per claim.
  The id is ``ev:<sha256>`` over a canonical digest of subject +
  provider + predicate + parameters, so the same observation made
  twice is the same record, and every later claim can link back to
  the exact evidence it rests on (``links`` — a provenance graph,
  not a pile of text). Three honesty axes ride on every record:
  ``confidence`` (observed | derived | inferred), ``authority``
  (shipped-artifact | controlled-replay | historical-reference |
  external-service | analyst-inference), and ``limitations`` — the
  limits of the observation, carried *on the record*, never in a
  footnote somebody can lose.

* **Residual unknowns** — what the evidence does *not* establish is
  a first-class record too: a question, the evidence for and against
  it, a status lifecycle (open → resolved / contradicted), and
  ``required_authority`` + ``required_confidence`` — what it would
  *take* to close it. The lifecycle rules are enforced here, in the
  store, not by convention: "resolved" demands a resolution text,
  "contradicted" demands contradicting evidence, and no close lands
  whose stated authority/confidence is weaker than what the unknown
  requires.

Storage follows the house discipline for organ-local tables (the
autonomy-ledger pattern): the store creates its own two tables on
the shared :class:`~nomorals.storage.db.Database` with
``CREATE TABLE IF NOT EXISTS``. The global migration list is not
touched — these tables belong to this organ.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .failures import record_failure

__all__ = [
    "CONFIDENCES", "AUTHORITIES", "Evidence", "Unknown",
    "EvidenceStore", "UnknownCloseError", "evidence_id_for",
]

#: The confidence axis, weakest last. Rank order is the close-rule
#: order: a close must state confidence at least this strong.
CONFIDENCES: tuple[str, ...] = ("observed", "derived", "inferred")

#: The authority axis, strongest first: evidence taken from the
#: shipped artifact itself outranks a controlled replay of it, which
#: outranks a historical reference, an external service's word, and
#: finally an analyst's inference.
AUTHORITIES: tuple[str, ...] = (
    "shipped-artifact", "controlled-replay", "historical-reference",
    "external-service", "analyst-inference",
)

_CONF_RANK = {name: len(CONFIDENCES) - i for i, name in enumerate(CONFIDENCES)}
_AUTH_RANK = {name: len(AUTHORITIES) - i for i, name in enumerate(AUTHORITIES)}

#: Raw inspector output kept on the record is capped — the record is
#: an index into the truth, not a second copy of the artifact.
_RAW_CAP = 20000


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def evidence_id_for(*, subject_sha256: str, provider_id: str,
                    predicate_type: str, operation: str,
                    parameters: dict[str, Any]) -> str:
    """The content address of one observation.

    Digest inputs are exactly REA's: subject + provider + predicate
    + parameters. The result body is NOT an input — re-observing the
    same thing and getting the same answer is the same evidence;
    getting a *different* answer under the same address is a conflict
    the store surfaces instead of overwriting (see
    :meth:`EvidenceStore.record`).
    """
    digest = hashlib.sha256(_canonical({
        "subject_sha256": subject_sha256,
        "provider_id": provider_id,
        "predicate_type": predicate_type,
        "operation": operation,
        "parameters": parameters,
    }).encode("utf-8")).hexdigest()
    return f"ev:{digest}"


def _unknown_id_for(*, question: str, subject_sha256: str,
                    supporting: tuple[str, ...]) -> str:
    digest = hashlib.sha256(_canonical({
        "question": question,
        "subject_sha256": subject_sha256,
        "supporting": sorted(supporting),
    }).encode("utf-8")).hexdigest()
    return f"un:{digest[:24]}"


@dataclass(frozen=True)
class Evidence:
    """One immutable observation, content-addressed."""

    evidence_id: str
    subject_sha256: str
    subject_name: str = ""
    subject_size: int = 0
    provider_id: str = ""
    provider_name: str = ""
    provider_version: str = ""
    predicate_type: str = ""
    operation: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    raw_result: str = ""
    confidence: str = "observed"
    authority: str = "shipped-artifact"
    limitations: tuple[str, ...] = ()
    locations: tuple[dict[str, Any], ...] = ()
    links: tuple[str, ...] = ()
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "subject_sha256": self.subject_sha256,
            "subject_name": self.subject_name,
            "subject_size": self.subject_size,
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "predicate_type": self.predicate_type,
            "operation": self.operation,
            "parameters": self.parameters,
            "result": self.result,
            "raw_result": self.raw_result,
            "confidence": self.confidence,
            "authority": self.authority,
            "limitations": list(self.limitations),
            "locations": [dict(loc) for loc in self.locations],
            "links": list(self.links),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Evidence:
        return cls(
            evidence_id=str(row["evidence_id"]),
            subject_sha256=str(row.get("subject_sha256") or ""),
            subject_name=str(row.get("subject_name") or ""),
            subject_size=int(row.get("subject_size") or 0),
            provider_id=str(row.get("provider_id") or ""),
            provider_name=str(row.get("provider_name") or ""),
            provider_version=str(row.get("provider_version") or ""),
            predicate_type=str(row.get("predicate_type") or ""),
            operation=str(row.get("operation") or ""),
            parameters=dict(row.get("parameters") or {}),
            result=dict(row.get("result") or {}),
            raw_result=str(row.get("raw_result") or ""),
            confidence=str(row.get("confidence") or "observed"),
            authority=str(row.get("authority") or "shipped-artifact"),
            limitations=tuple(row.get("limitations") or ()),
            locations=tuple(dict(loc) for loc in (row.get("locations") or ())),
            links=tuple(row.get("links") or ()),
            created_at=float(row.get("created_at") or 0.0),
        )


@dataclass(frozen=True)
class Unknown:
    """One residual unknown: a question the evidence has not closed."""

    unknown_id: str
    question: str
    subject_sha256: str = ""
    raised_by: str = ""
    supporting: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()
    required_authority: str = "analyst-inference"
    required_confidence: str = "inferred"
    status: str = "open"
    resolution: str = ""
    resolution_authority: str = ""
    resolution_confidence: str = ""
    created_at: float = 0.0
    closed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "unknown_id": self.unknown_id,
            "question": self.question,
            "subject_sha256": self.subject_sha256,
            "raised_by": self.raised_by,
            "supporting": list(self.supporting),
            "contradicting": list(self.contradicting),
            "required_authority": self.required_authority,
            "required_confidence": self.required_confidence,
            "status": self.status,
            "resolution": self.resolution,
            "resolution_authority": self.resolution_authority,
            "resolution_confidence": self.resolution_confidence,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Unknown:
        return cls(
            unknown_id=str(row["unknown_id"]),
            question=str(row.get("question") or ""),
            subject_sha256=str(row.get("subject_sha256") or ""),
            raised_by=str(row.get("raised_by") or ""),
            supporting=tuple(row.get("supporting") or ()),
            contradicting=tuple(row.get("contradicting") or ()),
            required_authority=str(row.get("required_authority")
                                   or "analyst-inference"),
            required_confidence=str(row.get("required_confidence")
                                    or "inferred"),
            status=str(row.get("status") or "open"),
            resolution=str(row.get("resolution") or ""),
            resolution_authority=str(row.get("resolution_authority") or ""),
            resolution_confidence=str(row.get("resolution_confidence") or ""),
            created_at=float(row.get("created_at") or 0.0),
            closed_at=float(row.get("closed_at") or 0.0),
        )


class UnknownCloseError(ValueError):
    """A close attempt the lifecycle rules refuse. The message says
    exactly which rule and what would satisfy it."""


def _row_evidence(row: dict[str, Any]) -> Evidence:
    row = dict(row)
    for key in ("parameters", "result", "limitations", "locations", "links"):
        raw = row.get(key)
        if isinstance(raw, str):
            try:
                row[key] = json.loads(raw) if raw else (
                    {} if key in ("parameters", "result") else [])
            except ValueError:
                row[key] = {} if key in ("parameters", "result") else []
    return Evidence.from_dict(row)


def _row_unknown(row: dict[str, Any]) -> Unknown:
    row = dict(row)
    for key in ("supporting", "contradicting"):
        raw = row.get(key)
        if isinstance(raw, str):
            try:
                row[key] = json.loads(raw) if raw else []
            except ValueError:
                row[key] = []
    return Unknown.from_dict(row)


class EvidenceStore:
    """Evidence + residual unknowns on the shared house database.

    Organ-local tables (``re_evidence``, ``re_unknowns``), created
    idempotently at construction — the autonomy-ledger discipline.
    """

    def __init__(self, db: Any) -> None:
        if db is None:
            raise ValueError("EvidenceStore needs a database handle")
        self.db = db
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        try:
            with self.db.transaction():
                self.db.execute(
                    """CREATE TABLE IF NOT EXISTS re_evidence (
                        evidence_id TEXT PRIMARY KEY,
                        subject_sha256 TEXT NOT NULL DEFAULT '',
                        subject_name TEXT NOT NULL DEFAULT '',
                        subject_size INTEGER NOT NULL DEFAULT 0,
                        provider_id TEXT NOT NULL DEFAULT '',
                        provider_name TEXT NOT NULL DEFAULT '',
                        provider_version TEXT NOT NULL DEFAULT '',
                        predicate_type TEXT NOT NULL DEFAULT '',
                        operation TEXT NOT NULL DEFAULT '',
                        parameters TEXT NOT NULL DEFAULT '{}',
                        result TEXT NOT NULL DEFAULT '{}',
                        raw_result TEXT NOT NULL DEFAULT '',
                        confidence TEXT NOT NULL DEFAULT 'observed',
                        authority TEXT NOT NULL DEFAULT 'shipped-artifact',
                        limitations TEXT NOT NULL DEFAULT '[]',
                        locations TEXT NOT NULL DEFAULT '[]',
                        links TEXT NOT NULL DEFAULT '[]',
                        created_at REAL NOT NULL DEFAULT 0
                    )""")
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_re_evidence_subject "
                    "ON re_evidence(subject_sha256, created_at)")
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_re_evidence_created "
                    "ON re_evidence(created_at)")
                self.db.execute(
                    """CREATE TABLE IF NOT EXISTS re_unknowns (
                        unknown_id TEXT PRIMARY KEY,
                        question TEXT NOT NULL DEFAULT '',
                        subject_sha256 TEXT NOT NULL DEFAULT '',
                        raised_by TEXT NOT NULL DEFAULT '',
                        supporting TEXT NOT NULL DEFAULT '[]',
                        contradicting TEXT NOT NULL DEFAULT '[]',
                        required_authority TEXT NOT NULL
                            DEFAULT 'analyst-inference',
                        required_confidence TEXT NOT NULL DEFAULT 'inferred',
                        status TEXT NOT NULL DEFAULT 'open',
                        resolution TEXT NOT NULL DEFAULT '',
                        resolution_authority TEXT NOT NULL DEFAULT '',
                        resolution_confidence TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL DEFAULT 0,
                        closed_at REAL NOT NULL DEFAULT 0
                    )""")
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_re_unknowns_status "
                    "ON re_unknowns(status, created_at)")
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_re_unknowns_subject "
                    "ON re_unknowns(subject_sha256)")
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced
            record_failure("reverse.evidence.ensure_tables", exc)
            raise

    # ── evidence ─────────────────────────────────────────────────────
    def record(self, *, subject_sha256: str, predicate_type: str,
               operation: str, provider_id: str,
               subject_name: str = "", subject_size: int = 0,
               provider_name: str = "", provider_version: str = "",
               parameters: dict[str, Any] | None = None,
               result: dict[str, Any] | None = None,
               raw_result: str = "",
               confidence: str = "observed",
               authority: str = "shipped-artifact",
               limitations: tuple[str, ...] | list[str] = (),
               locations: tuple[dict[str, Any], ...]
               | list[dict[str, Any]] = (),
               links: tuple[str, ...] | list[str] = (),
               ts: float | None = None) -> Evidence:
        """Record one observation; returns the stored record.

        Content-addressed: recording the same observation twice
        returns the existing record unchanged. If the same address
        arrives with a *different* result body, the first record
        stands (evidence is immutable) and the conflict is recorded
        in the failure ledger rather than silently overwritten.
        """
        if confidence not in _CONF_RANK:
            raise ValueError(
                f"confidence {confidence!r} not in {list(CONFIDENCES)}")
        if authority not in _AUTH_RANK:
            raise ValueError(
                f"authority {authority!r} not in {list(AUTHORITIES)}")
        params = dict(parameters or {})
        ev_id = evidence_id_for(
            subject_sha256=subject_sha256, provider_id=provider_id,
            predicate_type=predicate_type, operation=operation,
            parameters=params)
        existing = self.get(ev_id)
        if existing is not None:
            if existing.result != dict(result or {}):
                record_failure(
                    "reverse.evidence.record_conflict",
                    ValueError(
                        f"evidence {ev_id} re-observed with a different "
                        "result; the original record stands"))
            return existing
        record = Evidence(
            evidence_id=ev_id,
            subject_sha256=subject_sha256,
            subject_name=subject_name,
            subject_size=int(subject_size or 0),
            provider_id=provider_id,
            provider_name=provider_name,
            provider_version=provider_version,
            predicate_type=predicate_type,
            operation=operation,
            parameters=params,
            result=dict(result or {}),
            raw_result=str(raw_result or "")[:_RAW_CAP],
            confidence=confidence,
            authority=authority,
            limitations=tuple(str(x) for x in limitations),
            locations=tuple(dict(x) for x in locations),
            links=tuple(str(x) for x in links),
            created_at=float(ts if ts is not None else time.time()),
        )
        with self.db.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO re_evidence (evidence_id, "
                "subject_sha256, subject_name, subject_size, provider_id, "
                "provider_name, provider_version, predicate_type, "
                "operation, parameters, result, raw_result, confidence, "
                "authority, limitations, locations, links, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?)",
                (record.evidence_id, record.subject_sha256,
                 record.subject_name, record.subject_size,
                 record.provider_id, record.provider_name,
                 record.provider_version, record.predicate_type,
                 record.operation, _canonical(record.parameters),
                 _canonical(record.result), record.raw_result,
                 record.confidence, record.authority,
                 _canonical(list(record.limitations)),
                 _canonical([dict(x) for x in record.locations]),
                 _canonical(list(record.links)), record.created_at))
        return record

    def get(self, evidence_id: str) -> Evidence | None:
        """Exact id, else a unique id-prefix, else None."""
        ref = (evidence_id or "").strip()
        if not ref:
            return None
        row = self.db.query_one(
            "SELECT * FROM re_evidence WHERE evidence_id = ?", (ref,))
        if row is not None:
            return _row_evidence(row)
        rows = self.db.query(
            "SELECT * FROM re_evidence WHERE evidence_id LIKE ? "
            "ORDER BY created_at LIMIT 2", (f"ev:{ref.removeprefix('ev:')}%",))
        if len(rows) == 1:
            return _row_evidence(rows[0])
        return None

    def list_recent(self, limit: int = 20) -> list[Evidence]:
        rows = self.db.query(
            "SELECT * FROM re_evidence ORDER BY created_at DESC, "
            "evidence_id LIMIT ?", (max(0, int(limit)),))
        return [_row_evidence(r) for r in rows]

    def list_for_subject(self, subject_sha256: str) -> list[Evidence]:
        rows = self.db.query(
            "SELECT * FROM re_evidence WHERE subject_sha256 = ? "
            "ORDER BY created_at, evidence_id", (subject_sha256,))
        return [_row_evidence(r) for r in rows]

    def count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM re_evidence",
                                  default=0))

    def provenance_chain(self, evidence_id: str) -> list[Evidence]:
        """The record plus everything it links to, transitively.

        Breadth-first from the record itself, links in stored order,
        cycle-safe. A link whose target is not in the store is
        skipped (the link itself stays on the record that carries
        it — a dangling link is visible, never hidden)."""
        root = self.get(evidence_id)
        if root is None:
            return []
        out: list[Evidence] = [root]
        seen = {root.evidence_id}
        queue = list(root.links)
        while queue:
            nxt = queue.pop(0)
            if nxt in seen:
                continue
            seen.add(nxt)
            rec = self.get(nxt)
            if rec is None:
                continue
            out.append(rec)
            queue.extend(rec.links)
        return out

    # ── residual unknowns ────────────────────────────────────────────
    def open_unknown(self, question: str, *,
                     subject_sha256: str = "", raised_by: str = "",
                     supporting: tuple[str, ...] | list[str] = (),
                     contradicting: tuple[str, ...] | list[str] = (),
                     required_authority: str = "analyst-inference",
                     required_confidence: str = "inferred",
                     ts: float | None = None) -> Unknown:
        """Register one residual unknown (idempotent on its address).

        Re-opening the same question against the same subject with
        the same supporting evidence returns the existing record —
        including a closed one: a closed unknown is history, and a
        genuinely new doubt is a new question, not a resurrection.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("an unknown needs a question")
        if required_authority not in _AUTH_RANK:
            raise ValueError(
                f"required_authority {required_authority!r} "
                f"not in {list(AUTHORITIES)}")
        if required_confidence not in _CONF_RANK:
            raise ValueError(
                f"required_confidence {required_confidence!r} "
                f"not in {list(CONFIDENCES)}")
        sup = tuple(str(x) for x in supporting)
        con = tuple(str(x) for x in contradicting)
        un_id = _unknown_id_for(question=question,
                                subject_sha256=subject_sha256,
                                supporting=sup)
        existing = self.get_unknown(un_id)
        if existing is not None:
            return existing
        rec = Unknown(
            unknown_id=un_id, question=question,
            subject_sha256=subject_sha256, raised_by=raised_by,
            supporting=sup, contradicting=con,
            required_authority=required_authority,
            required_confidence=required_confidence,
            created_at=float(ts if ts is not None else time.time()))
        with self.db.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO re_unknowns (unknown_id, "
                "question, subject_sha256, raised_by, supporting, "
                "contradicting, required_authority, "
                "required_confidence, status, resolution, "
                "resolution_authority, resolution_confidence, "
                "created_at, closed_at) VALUES (?, ?, ?, ?, ?, ?, ?, "
                "?, 'open', '', '', '', ?, 0)",
                (rec.unknown_id, rec.question, rec.subject_sha256,
                 rec.raised_by, _canonical(list(sup)),
                 _canonical(list(con)), rec.required_authority,
                 rec.required_confidence, rec.created_at))
        return rec

    def get_unknown(self, unknown_id: str) -> Unknown | None:
        """Exact id, else a unique id-prefix, else None."""
        ref = (unknown_id or "").strip()
        if not ref:
            return None
        row = self.db.query_one(
            "SELECT * FROM re_unknowns WHERE unknown_id = ?", (ref,))
        if row is not None:
            return _row_unknown(row)
        rows = self.db.query(
            "SELECT * FROM re_unknowns WHERE unknown_id LIKE ? "
            "ORDER BY created_at LIMIT 2",
            (f"un:{ref.removeprefix('un:')}%",))
        if len(rows) == 1:
            return _row_unknown(rows[0])
        return None

    def list_unknowns(self, status: str | None = None,
                      limit: int = 50) -> list[Unknown]:
        if status:
            rows = self.db.query(
                "SELECT * FROM re_unknowns WHERE status = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (status, max(0, int(limit))))
        else:
            rows = self.db.query(
                "SELECT * FROM re_unknowns ORDER BY created_at DESC "
                "LIMIT ?", (max(0, int(limit)),))
        return [_row_unknown(r) for r in rows]

    def close_unknown(self, unknown_id: str, *, status: str,
                      resolution: str = "",
                      resolution_authority: str = "analyst-inference",
                      resolution_confidence: str = "inferred",
                      evidence_ids: tuple[str, ...] | list[str] = (),
                      ts: float | None = None) -> Unknown:
        """Close an unknown under the lifecycle rules.

        * ``resolved`` requires a resolution text; the closing
          evidence ids join the supporting set.
        * ``contradicted`` requires at least one contradicting
          evidence id (existing or newly supplied).
        * The stated resolution authority/confidence must meet the
          unknown's required authority/confidence — a guess cannot
          close a question that demands artifact evidence.
        * Closing an already-closed unknown is refused: the record
          is history.
        """
        rec = self.get_unknown(unknown_id)
        if rec is None:
            raise UnknownCloseError(f"no unknown matches {unknown_id!r}")
        if rec.status != "open":
            raise UnknownCloseError(
                f"unknown {rec.unknown_id} is already {rec.status} — "
                "a closed unknown is history, not a draft")
        if status not in ("resolved", "contradicted"):
            raise UnknownCloseError(
                f"status {status!r} is not a closing status "
                "(resolved|contradicted)")
        if resolution_authority not in _AUTH_RANK:
            raise UnknownCloseError(
                f"resolution_authority {resolution_authority!r} "
                f"not in {list(AUTHORITIES)}")
        if resolution_confidence not in _CONF_RANK:
            raise UnknownCloseError(
                f"resolution_confidence {resolution_confidence!r} "
                f"not in {list(CONFIDENCES)}")
        if _AUTH_RANK[resolution_authority] < _AUTH_RANK[rec.required_authority]:
            raise UnknownCloseError(
                f"closing {rec.unknown_id} requires authority "
                f"{rec.required_authority!r}; the close states "
                f"{resolution_authority!r}, which is weaker")
        if (_CONF_RANK[resolution_confidence]
                < _CONF_RANK[rec.required_confidence]):
            raise UnknownCloseError(
                f"closing {rec.unknown_id} requires confidence "
                f"{rec.required_confidence!r}; the close states "
                f"{resolution_confidence!r}, which is weaker")
        new_ids = tuple(str(x) for x in evidence_ids)
        supporting = rec.supporting
        contradicting = rec.contradicting
        if status == "resolved":
            if not (resolution or "").strip():
                raise UnknownCloseError(
                    "a resolved close requires the resolution — what "
                    "the answer turned out to be, in words")
            supporting = tuple(dict.fromkeys([*supporting, *new_ids]))
        else:
            contradicting = tuple(
                dict.fromkeys([*contradicting, *new_ids]))
            if not contradicting:
                raise UnknownCloseError(
                    "a contradicted close requires contradicting "
                    "evidence — at least one evidence id that cuts "
                    "against the question's premise")
        closed_at = float(ts if ts is not None else time.time())
        with self.db.transaction():
            self.db.execute(
                "UPDATE re_unknowns SET status = ?, resolution = ?, "
                "resolution_authority = ?, resolution_confidence = ?, "
                "supporting = ?, contradicting = ?, closed_at = ? "
                "WHERE unknown_id = ?",
                (status, (resolution or "").strip(),
                 resolution_authority, resolution_confidence,
                 _canonical(list(supporting)),
                 _canonical(list(contradicting)), closed_at,
                 rec.unknown_id))
        closed = self.get_unknown(rec.unknown_id)
        assert closed is not None  # the row was just written
        return closed

    def audit(self) -> dict[str, Any]:
        """The unknowns audit: what is open, what closing it takes,
        and which open questions nothing on record could close yet.

        An open unknown is *unclosable on current evidence* when no
        recorded evidence meets its required authority and confidence
        together — those are the questions that name the next rung
        (a deeper engine, a replay) instead of pretending to be
        answerable from what is already in the store.
        """
        rows = self.db.query(
            "SELECT status, COUNT(*) AS n FROM re_unknowns "
            "GROUP BY status")
        counts = {str(r["status"]): int(r["n"]) for r in rows}
        best_auth = 0
        best_conf = 0
        for rec in self.list_recent(limit=100000):
            best_auth = max(best_auth, _AUTH_RANK[rec.authority])
            best_conf = max(best_conf, _CONF_RANK[rec.confidence])
        unclosable: list[str] = []
        unsupported: list[str] = []
        for unk in self.list_unknowns(status="open", limit=100000):
            if (_AUTH_RANK[unk.required_authority] > best_auth
                    or _CONF_RANK[unk.required_confidence] > best_conf):
                unclosable.append(unk.unknown_id)
            if not unk.supporting:
                unsupported.append(unk.unknown_id)
        return {
            "evidence_records": self.count(),
            "unknowns_open": counts.get("open", 0),
            "unknowns_resolved": counts.get("resolved", 0),
            "unknowns_contradicted": counts.get("contradicted", 0),
            "open_unclosable_on_current_evidence": unclosable,
            "open_without_supporting_evidence": unsupported,
            "strongest_evidence_authority": next(
                (a for a in AUTHORITIES if _AUTH_RANK[a] == best_auth),
                None),
            "strongest_evidence_confidence": next(
                (c for c in CONFIDENCES if _CONF_RANK[c] == best_conf),
                None),
        }

    # ── import / export ──────────────────────────────────────────────
    def export(self) -> dict[str, Any]:
        """The whole store as one JSON-able dict (evidence + unknowns)."""
        return {
            "evidence": [r.to_dict() for r in self.list_recent(
                limit=100000)],
            "unknowns": [u.to_dict() for u in self.list_unknowns(
                limit=100000)],
        }

    def import_(self, payload: dict[str, Any]) -> dict[str, int]:
        """Merge an exported store. Evidence re-records under its own
        content address (duplicates are the same record); unknowns
        import open, or closed with their recorded resolution when
        their lifecycle fields are complete. Returns counts of rows
        newly landed."""
        landed = {"evidence": 0, "unknowns": 0}
        unknowns_before = len(self.list_unknowns(limit=100000))
        for row in payload.get("evidence") or []:
            before = self.count()
            self.record(
                subject_sha256=str(row.get("subject_sha256") or ""),
                subject_name=str(row.get("subject_name") or ""),
                subject_size=int(row.get("subject_size") or 0),
                provider_id=str(row.get("provider_id") or ""),
                provider_name=str(row.get("provider_name") or ""),
                provider_version=str(row.get("provider_version") or ""),
                predicate_type=str(row.get("predicate_type") or ""),
                operation=str(row.get("operation") or ""),
                parameters=dict(row.get("parameters") or {}),
                result=dict(row.get("result") or {}),
                raw_result=str(row.get("raw_result") or ""),
                confidence=str(row.get("confidence") or "observed"),
                authority=str(row.get("authority") or "shipped-artifact"),
                limitations=tuple(row.get("limitations") or ()),
                locations=tuple(row.get("locations") or ()),
                links=tuple(row.get("links") or ()),
                ts=float(row.get("created_at") or 0) or None)
            if self.count() > before:
                landed["evidence"] += 1
        for row in payload.get("unknowns") or []:
            unk = self.open_unknown(
                str(row.get("question") or ""),
                subject_sha256=str(row.get("subject_sha256") or ""),
                raised_by=str(row.get("raised_by") or ""),
                supporting=tuple(row.get("supporting") or ()),
                contradicting=tuple(row.get("contradicting") or ()),
                required_authority=str(
                    row.get("required_authority") or "analyst-inference"),
                required_confidence=str(
                    row.get("required_confidence") or "inferred"),
                ts=float(row.get("created_at") or 0) or None)
            if unk.status == "open" and row.get("status") in (
                    "resolved", "contradicted"):
                try:
                    self.close_unknown(
                        unk.unknown_id, status=str(row["status"]),
                        resolution=str(row.get("resolution") or ""),
                        resolution_authority=str(
                            row.get("resolution_authority")
                            or "analyst-inference"),
                        resolution_confidence=str(
                            row.get("resolution_confidence") or "inferred"),
                        evidence_ids=())
                except UnknownCloseError as exc:
                    record_failure("reverse.evidence.import_close", exc)
                    continue
        landed["unknowns"] = max(
            0, len(self.list_unknowns(limit=100000)) - unknowns_before)
        return landed
