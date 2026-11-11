"""Delivery record registry (L2) — transfers, delivery receipts,
publication receipts.

Doc 15 (Notifications + Outcome Stewardship) P5 named two record
gaps on the record, in ``agents/outcome_steward.adapter_health()``:

* *transfers* — "no transfer registry exists … begun transfers
  leave no queryable record, and uploads keep no records at all";
* *artifacts* — "artifacts carry no delivery or publication
  record — nothing records whether an artifact was ever delivered".

This module is those missing records. It is written at the seams
where an artifact actually changes hands — ``/artifact send`` /
``/artifact export`` (the artifact store → a chat) and the social
pipeline's ``execute_post`` (a draft's artifact media → a platform)
— and read by the steward's transfers/artifacts adapters, whose
evidence is what lets an outcome case resolve *citing a record*
instead of staying unknown.

The discipline mirrors :mod:`nomorals.agents.outcome_cases`:

* **Records claim only recorded outcomes.** A transfer entry is
  ``begun`` when the send is attempted and moves to ``delivered``
  or ``failed`` only when the send's own outcome is known. A
  delivery receipt exists only for a ``delivered`` transfer; a
  publication receipt records the platform result the publish
  itself returned (its post reference, its media accounting) —
  cited, never re-measured.
* **Append-only.** Rows are inserted and, for a transfer's single
  state transition, updated exactly once (``begun`` → terminal).
  Nothing is ever deleted, so the trail a verdict rests on can
  always be re-read.
* **Fail-open at the funnels.** This store raises honestly on a bad
  write; the funnel call sites contain the failure (the codebase's
  record-failure idiom) so a registry problem can never block a
  delivery or a post.

Persistence is runtime-created tables on the shared ``Database``
(the Batch B pattern — ``storage/migrations.py`` is frozen). Chat
keys are recorded because a transfer record names its destination
in the system's own terms (the outbox rows in the same database
carry chat topics the same way); owner-facing renders never print
them — the steward's subjects name the platform and the artifact,
never the chat key, exactly as the outbox adapter disciplines its
subjects.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TRANSFER_STATES",
    "TransferEntry",
    "DeliveryReceipt",
    "PublicationReceipt",
    "DeliveryRecordStore",
    "DeliveryRecordError",
    "artifact_label",
]

#: A transfer's lifecycle: begun → delivered | failed. Terminal
#: states are written exactly once, by :meth:`complete_transfer`.
TRANSFER_STATES = ("begun", "delivered", "failed")

_DDL = """
CREATE TABLE IF NOT EXISTS transfer_entries (
  id TEXT PRIMARY KEY,
  artifact_id TEXT NOT NULL DEFAULT '',
  artifact_label TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT 'send',
  from_surface TEXT NOT NULL DEFAULT '',
  platform TEXT NOT NULL DEFAULT '',
  chat_key TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'begun',
  send_ref TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '{}',
  begun_at REAL NOT NULL DEFAULT 0,
  finished_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_transfer_entries_artifact
  ON transfer_entries(artifact_id, begun_at);
CREATE INDEX IF NOT EXISTS idx_transfer_entries_state
  ON transfer_entries(state, begun_at);
CREATE TABLE IF NOT EXISTS delivery_receipts (
  id TEXT PRIMARY KEY,
  transfer_id TEXT NOT NULL DEFAULT '',
  artifact_id TEXT NOT NULL DEFAULT '',
  platform TEXT NOT NULL DEFAULT '',
  send_ref TEXT NOT NULL DEFAULT '',
  outcome TEXT NOT NULL DEFAULT 'delivered',
  detail TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_delivery_receipts_artifact
  ON delivery_receipts(artifact_id, created_at);
CREATE TABLE IF NOT EXISTS publication_receipts (
  id TEXT PRIMARY KEY,
  draft_id TEXT NOT NULL DEFAULT '',
  artifact_id TEXT NOT NULL DEFAULT '',
  platform TEXT NOT NULL DEFAULT '',
  outcome TEXT NOT NULL DEFAULT 'posted',
  external_id TEXT NOT NULL DEFAULT '',
  url TEXT NOT NULL DEFAULT '',
  media_offered INTEGER NOT NULL DEFAULT 0,
  media_carried INTEGER NOT NULL DEFAULT 0,
  error TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_publication_receipts_artifact
  ON publication_receipts(artifact_id, created_at);
CREATE INDEX IF NOT EXISTS idx_publication_receipts_draft
  ON publication_receipts(draft_id, created_at);
"""

#: Metadata keys that name an artifact for humans, in priority
#: order — the same keys the artifact store's own title logic reads.
_LABEL_KEYS = ("title", "name", "filename")


def artifact_label(art: Any) -> str:
    """The human label records name an artifact by: its metadata
    title/name/filename when present, else ``<type> artifact``.
    Never the artifact id — records are cited in owner language."""
    metadata = getattr(art, "metadata", None) or {}
    for key in _LABEL_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    art_type = str(getattr(art, "type", "") or "").strip()
    return f"{art_type} artifact" if art_type else "artifact"


class DeliveryRecordError(Exception):
    """Base for delivery-record store refusals."""


@dataclass
class TransferEntry:
    """One artifact hand-off, from attempt to recorded outcome."""

    id: str
    artifact_id: str = ""
    artifact_label: str = ""
    kind: str = "send"            # send | export
    from_surface: str = ""
    platform: str = ""
    chat_key: str = ""
    state: str = "begun"
    send_ref: str = ""            # the send's own reference (message id)
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    begun_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class DeliveryReceipt:
    """Proof a transfer was delivered — minted only on delivery."""

    id: str
    transfer_id: str = ""
    artifact_id: str = ""
    platform: str = ""
    send_ref: str = ""
    outcome: str = "delivered"
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class PublicationReceipt:
    """Proof of one artifact's publication outcome on one platform,
    tied to the draft whose publish produced it."""

    id: str
    draft_id: str = ""
    artifact_id: str = ""
    platform: str = ""
    outcome: str = "posted"       # posted | failed
    external_id: str = ""         # the post's own reference
    url: str = ""
    media_offered: int = 0
    media_carried: int = 0
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


def _loads(raw: Any) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _row_transfer(row: Any) -> TransferEntry:
    return TransferEntry(
        id=row["id"], artifact_id=row["artifact_id"],
        artifact_label=row["artifact_label"], kind=row["kind"],
        from_surface=row["from_surface"], platform=row["platform"],
        chat_key=row["chat_key"], state=row["state"],
        send_ref=row["send_ref"], error=row["error"],
        detail=_loads(row["detail"]), begun_at=float(row["begun_at"]),
        finished_at=float(row["finished_at"]))


def _row_delivery(row: Any) -> DeliveryReceipt:
    return DeliveryReceipt(
        id=row["id"], transfer_id=row["transfer_id"],
        artifact_id=row["artifact_id"], platform=row["platform"],
        send_ref=row["send_ref"], outcome=row["outcome"],
        detail=_loads(row["detail"]),
        created_at=float(row["created_at"]))


def _row_publication(row: Any) -> PublicationReceipt:
    return PublicationReceipt(
        id=row["id"], draft_id=row["draft_id"],
        artifact_id=row["artifact_id"], platform=row["platform"],
        outcome=row["outcome"], external_id=row["external_id"],
        url=row["url"], media_offered=int(row["media_offered"]),
        media_carried=int(row["media_carried"]), error=row["error"],
        detail=_loads(row["detail"]),
        created_at=float(row["created_at"]))


class DeliveryRecordStore:
    """Transfers + receipts over a shared ``Database``."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_DDL)

    # -- transfers ------------------------------------------------------

    def begin_transfer(self, *, artifact_id: str, artifact_label: str = "",
                       kind: str = "send", from_surface: str = "",
                       platform: str = "", chat_key: str = "",
                       detail: dict[str, Any] | None = None
                       ) -> TransferEntry:
        """Record that an artifact hand-off has begun.

        Written BEFORE the send is attempted, so a process that dies
        mid-send still leaves the queryable record the steward's
        transfers discovery reads — the gap P5 named.
        """
        if not artifact_id:
            raise DeliveryRecordError("a transfer names its artifact")
        now = time.time()
        entry = TransferEntry(
            id=f"tr-{uuid.uuid4().hex[:12]}", artifact_id=artifact_id,
            artifact_label=artifact_label, kind=kind,
            from_surface=from_surface, platform=platform,
            chat_key=chat_key, detail=dict(detail or {}), begun_at=now)
        self.db.execute(
            "INSERT INTO transfer_entries (id, artifact_id,"
            " artifact_label, kind, from_surface, platform, chat_key,"
            " state, send_ref, error, detail, begun_at, finished_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (entry.id, entry.artifact_id, entry.artifact_label,
             entry.kind, entry.from_surface, entry.platform,
             entry.chat_key, entry.state, "", "",
             json.dumps(entry.detail), entry.begun_at, 0.0))
        return entry

    def complete_transfer(self, transfer_id: str, *, ok: bool,
                          send_ref: str = "", error: str = ""
                          ) -> tuple[TransferEntry,
                                     DeliveryReceipt | None]:
        """Land a begun transfer's outcome — the entry's ONLY state
        transition. On success a delivery receipt is minted in the
        same call (the receipt the steward cites); on failure the
        entry records the failure and NO receipt exists. Completing
        an already-terminal transfer changes nothing and mints
        nothing — it returns the recorded state as it stands."""
        entry = self.get_transfer(transfer_id)
        if entry is None:
            raise DeliveryRecordError(f"no transfer {transfer_id!r}")
        if entry.state != "begun":
            receipt = self.delivery_receipt_for_transfer(transfer_id) \
                if entry.state == "delivered" else None
            return entry, receipt
        now = time.time()
        entry.state = "delivered" if ok else "failed"
        entry.send_ref = str(send_ref or "")
        entry.error = "" if ok else str(error or "")[:500]
        entry.finished_at = now
        self.db.execute(
            "UPDATE transfer_entries SET state = ?, send_ref = ?,"
            " error = ?, finished_at = ? WHERE id = ?",
            (entry.state, entry.send_ref, entry.error, now, entry.id))
        receipt = None
        if ok:
            receipt = DeliveryReceipt(
                id=f"dr-{uuid.uuid4().hex[:12]}",
                transfer_id=entry.id, artifact_id=entry.artifact_id,
                platform=entry.platform, send_ref=entry.send_ref,
                detail={"kind": entry.kind,
                        "artifact_label": entry.artifact_label},
                created_at=now)
            self.db.execute(
                "INSERT INTO delivery_receipts (id, transfer_id,"
                " artifact_id, platform, send_ref, outcome, detail,"
                " created_at) VALUES (?,?,?,?,?,?,?,?)",
                (receipt.id, receipt.transfer_id, receipt.artifact_id,
                 receipt.platform, receipt.send_ref, receipt.outcome,
                 json.dumps(receipt.detail), receipt.created_at))
        return entry, receipt

    def get_transfer(self, transfer_id: str) -> TransferEntry | None:
        row = self.db.query_one(
            "SELECT * FROM transfer_entries WHERE id = ?",
            (transfer_id,))
        return _row_transfer(row) if row else None

    def transfers_for_artifact(self, artifact_id: str, *, limit: int = 100
                               ) -> list[TransferEntry]:
        rows = self.db.query(
            "SELECT * FROM transfer_entries WHERE artifact_id = ?"
            " ORDER BY begun_at, rowid LIMIT ?",
            (artifact_id, int(limit)))
        return [_row_transfer(r) for r in rows]

    def stuck_transfers(self, now: float, *, older_than_s: float
                        ) -> list[TransferEntry]:
        """Transfers still ``begun`` long past their attempt — the
        send's outcome was never recorded (the process died, or the
        completion write was lost), so whether the artifact landed
        is genuinely unknown."""
        rows = self.db.query(
            "SELECT * FROM transfer_entries WHERE state = 'begun'"
            " AND begun_at <= ? ORDER BY begun_at",
            (now - float(older_than_s),))
        return [_row_transfer(r) for r in rows]

    def abandoned_deliveries(self, now: float, *, older_than_s: float
                             ) -> list[TransferEntry]:
        """Artifacts whose hand-off story ends in an old failure:
        the artifact's LATEST transfer failed longer ago than
        ``older_than_s`` and no delivery receipt exists after that
        failure. Returns the latest failed entry per such artifact.
        An artifact later delivered is not abandoned — its receipt
        is the answer, and it is excluded here."""
        cutoff = now - float(older_than_s)
        rows = self.db.query(
            "SELECT * FROM transfer_entries WHERE state = 'failed'"
            " AND finished_at <= ? ORDER BY finished_at",
            (cutoff,))
        latest_failed: dict[str, TransferEntry] = {}
        for row in rows:
            entry = _row_transfer(row)
            current = latest_failed.get(entry.artifact_id)
            if current is None or entry.finished_at >= current.finished_at:
                latest_failed[entry.artifact_id] = entry
        out: list[TransferEntry] = []
        for artifact_id, entry in latest_failed.items():
            transfers = self.transfers_for_artifact(artifact_id)
            if any(t.state == "delivered"
                   and t.finished_at > entry.finished_at
                   for t in transfers):
                continue
            if any(t.state == "begun"
                   and t.begun_at > entry.finished_at
                   for t in transfers):
                continue  # a newer attempt is in flight, not abandoned
            receipts = self.delivery_receipts_for_artifact(artifact_id)
            if any(r.created_at > entry.finished_at for r in receipts):
                continue
            out.append(entry)
        out.sort(key=lambda e: e.finished_at)
        return out

    # -- receipts ---------------------------------------------------------

    def delivery_receipt_for_transfer(self, transfer_id: str
                                      ) -> DeliveryReceipt | None:
        row = self.db.query_one(
            "SELECT * FROM delivery_receipts WHERE transfer_id = ?"
            " ORDER BY created_at LIMIT 1", (transfer_id,))
        return _row_delivery(row) if row else None

    def delivery_receipts_for_artifact(self, artifact_id: str, *,
                                       limit: int = 100
                                       ) -> list[DeliveryReceipt]:
        rows = self.db.query(
            "SELECT * FROM delivery_receipts WHERE artifact_id = ?"
            " ORDER BY created_at, rowid LIMIT ?",
            (artifact_id, int(limit)))
        return [_row_delivery(r) for r in rows]

    def record_publication(self, *, draft_id: str, artifact_id: str,
                           platform: str, ok: bool, external_id: str = "",
                           url: str = "", media_offered: int = 0,
                           media_carried: int = 0, error: str = "",
                           detail: dict[str, Any] | None = None
                           ) -> PublicationReceipt:
        """Record one artifact's publication outcome on one
        platform, tied to the draft that carried it. The outcome,
        the post's own reference and the publish's media accounting
        are the publish's own recorded facts — cited here, never
        re-measured."""
        if not artifact_id:
            raise DeliveryRecordError(
                "a publication receipt names its artifact")
        receipt = PublicationReceipt(
            id=f"pr-{uuid.uuid4().hex[:12]}", draft_id=draft_id,
            artifact_id=artifact_id, platform=platform,
            outcome="posted" if ok else "failed",
            external_id=str(external_id or ""), url=str(url or ""),
            media_offered=max(0, int(media_offered or 0)),
            media_carried=max(0, int(media_carried or 0)),
            error="" if ok else str(error or "")[:500],
            detail=dict(detail or {}), created_at=time.time())
        self.db.execute(
            "INSERT INTO publication_receipts (id, draft_id,"
            " artifact_id, platform, outcome, external_id, url,"
            " media_offered, media_carried, error, detail, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (receipt.id, receipt.draft_id, receipt.artifact_id,
             receipt.platform, receipt.outcome, receipt.external_id,
             receipt.url, receipt.media_offered, receipt.media_carried,
             receipt.error, json.dumps(receipt.detail),
             receipt.created_at))
        return receipt

    def publication_receipts_for_artifact(self, artifact_id: str, *,
                                          limit: int = 100
                                          ) -> list[PublicationReceipt]:
        rows = self.db.query(
            "SELECT * FROM publication_receipts WHERE artifact_id = ?"
            " ORDER BY created_at, rowid LIMIT ?",
            (artifact_id, int(limit)))
        return [_row_publication(r) for r in rows]

    def publication_receipts_for_draft(self, draft_id: str, *,
                                       limit: int = 100
                                       ) -> list[PublicationReceipt]:
        rows = self.db.query(
            "SELECT * FROM publication_receipts WHERE draft_id = ?"
            " ORDER BY created_at, rowid LIMIT ?",
            (draft_id, int(limit)))
        return [_row_publication(r) for r in rows]

    def counts(self) -> dict[str, int]:
        out = {"transfers": 0, "begun": 0, "delivered": 0, "failed": 0,
               "delivery_receipts": 0, "publication_receipts": 0}
        for row in self.db.query(
                "SELECT state, COUNT(*) AS n FROM transfer_entries"
                " GROUP BY state"):
            out[row["state"]] = int(row["n"])
            out["transfers"] += int(row["n"])
        for table, key in (("delivery_receipts", "delivery_receipts"),
                           ("publication_receipts",
                            "publication_receipts")):
            rows = self.db.query(
                f"SELECT COUNT(*) AS n FROM {table}")
            out[key] = int(rows[0]["n"]) if rows else 0
        return out
