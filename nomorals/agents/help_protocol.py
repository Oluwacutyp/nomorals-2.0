"""The help protocol — doc 18 P5's typed request-for-help record.

An operation that cannot proceed used to have exactly two
options: fail, or guess. The evaluation names the third —
*honestly a delegation contract*: a typed record saying who is
stuck, on what reference, why, and what kind of help would unblock
it, with an answer path that lands on the record. This module is
that record: :class:`HelpStore` over the shared ``Database``,
beside delegation (the accountability substrate it rides), with
the same Batch B discipline — DDL in code, runtime-created table,
migrations untouched.

Survey for an existing ask surface to compose through (the P5
brief's condition): Devon ships no shipped owner-ask record this
record could ride — partner-runtime asks are ephemeral chat
turns, not durable typed records — so the record stands on its
own table here, and surfaces (chat or otherwise) compose over it
when they are built; the ticket projection already reads it (see
``run_ticket``'s ``detail["help_requests"]``).

Scope is the record + the answer path + ticket visibility. No
chat surface ships in this pass.

Kinds are closed on purpose: ``blocked`` (something else must
move first), ``needs_input`` (a decision or value only the helper
can give), ``needs_resource`` (a credential, file, budget or
capability), ``needs_review`` (a second pair of eyes before
proceeding). States: ``open`` → ``answered`` (the answer and its
author land on the record) or ``withdrawn`` (the requester takes
it back — only the requester may).

Layer: agents (L5) — imported by the ticket projection. Imports
stdlib only; the store is duck-typed over the shared Database.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

__all__ = [
    "HELP_KINDS",
    "HELP_STATES",
    "HelpError",
    "HelpStore",
    "request_help",
    "help_requests_for_ref",
]

HELP_KINDS = ("blocked", "needs_input", "needs_resource",
              "needs_review")
HELP_STATES = ("open", "answered", "withdrawn")

_HELP_DDL = """
CREATE TABLE IF NOT EXISTS help_requests (
  id TEXT PRIMARY KEY,
  ref TEXT NOT NULL DEFAULT '',
  requester TEXT NOT NULL DEFAULT '',
  helper TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'open',
  created_at REAL NOT NULL DEFAULT 0,
  answered_by TEXT NOT NULL DEFAULT '',
  answer TEXT NOT NULL DEFAULT '',
  answered_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_help_requests_ref_state
  ON help_requests(ref, state);
"""


class HelpError(Exception):
    """A help-protocol refusal: answering or withdrawing a
    request that is not open, withdrawing someone else's request,
    or naming a request that does not exist."""


def _help_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "ref": row["ref"],
        "requester": row["requester"],
        "helper": row["helper"],
        "kind": row["kind"],
        "detail": row["detail"],
        "state": row["state"],
        "created_at": float(row["created_at"] or 0),
        "answered_by": row["answered_by"],
        "answer": row["answer"],
        "answered_at": float(row["answered_at"] or 0),
    }


def _table_missing(exc: Exception) -> bool:
    return "no such table" in str(exc)


def help_requests_for_ref(db: Any, ref: str, limit: int = 20
                          ) -> list[dict[str, Any]]:
    """Every help request recorded for one reference, oldest
    first, in every state — the ticket projection's read.
    Read-only: runs no DDL, and a database where the table was
    never created reads as "no help requests"."""
    try:
        rows = db.query(
            "SELECT * FROM help_requests WHERE ref = ?"
            " ORDER BY rowid ASC LIMIT ?",
            (str(ref), int(limit)))
    except Exception as exc:  # noqa: BLE001 - no table = no requests
        if _table_missing(exc):
            return []
        raise
    return [_help_dict(row) for row in rows]


class HelpStore:
    """Help requests over the shared ``Database``."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_HELP_DDL)

    def request_help(self, ref: str, *, requester: str, kind: str,
                     detail: str, helper: str = "",
                     now: float | None = None) -> dict[str, Any]:
        """Open a help request on one reference. ``kind`` must be
        one of :data:`HELP_KINDS` and ``ref`` a run-ticket-shaped
        reference (``<prefix>:<id>``) — both are caller bugs when
        wrong, so both raise rather than recording garbage."""
        if kind not in HELP_KINDS:
            raise ValueError(
                f"unknown help kind {kind!r} — expected one of "
                + ", ".join(HELP_KINDS))
        if not str(ref or "").strip() or ":" not in str(ref):
            raise ValueError(
                f"help requests name a run-ticket reference, "
                f"got {ref!r}")
        now = time.time() if now is None else now
        record = {
            "id": f"help-{uuid.uuid4().hex[:12]}",
            "ref": str(ref),
            "requester": str(requester or ""),
            "helper": str(helper or ""),
            "kind": kind,
            "detail": str(detail or ""),
            "state": "open",
            "created_at": float(now),
            "answered_by": "",
            "answer": "",
            "answered_at": 0.0,
        }
        self.db.execute(
            "INSERT INTO help_requests (id, ref, requester, helper,"
            " kind, detail, state, created_at, answered_by, answer,"
            " answered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record["id"], record["ref"], record["requester"],
             record["helper"], record["kind"], record["detail"],
             record["state"], record["created_at"],
             record["answered_by"], record["answer"],
             record["answered_at"]))
        return record

    def get(self, request_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM help_requests WHERE id = ?",
            (str(request_id),))
        return _help_dict(row) if row is not None else None

    def _open_or_raise(self, request_id: str,
                       action: str) -> dict[str, Any]:
        record = self.get(request_id)
        if record is None:
            raise HelpError(
                f"no help request {request_id!r} to {action}")
        if record["state"] != "open":
            raise HelpError(
                f"help request {request_id} is {record['state']} — "
                f"only an open request can be {action}ed")
        return record

    def answer_help(self, request_id: str, *, by: str, answer: str,
                    now: float | None = None) -> dict[str, Any]:
        """Land an answer on an open request: the answer, its
        author and its time become part of the record. The state
        flip is compare-and-set on ``state = 'open'`` so two
        answers racing cannot both land."""
        record = self._open_or_raise(request_id, "answer")
        now = time.time() if now is None else now
        cur = self.db.execute(
            "UPDATE help_requests SET state = 'answered',"
            " answered_by = ?, answer = ?, answered_at = ?"
            " WHERE id = ? AND state = 'open'",
            (str(by or ""), str(answer or ""), float(now),
             record["id"]))
        if not cur.rowcount:
            raise HelpError(
                f"help request {request_id} was answered by a "
                "racing write first")
        return self.get(request_id) or record

    def withdraw_help(self, request_id: str, *, by: str
                      ) -> dict[str, Any]:
        """The requester takes an open request back. Only the
        requester may — a helper answers, they do not withdraw
        someone else's ask."""
        record = self._open_or_raise(request_id, "withdraw")
        if str(by or "") != record["requester"]:
            raise HelpError(
                f"only the requester {record['requester']!r} may "
                f"withdraw help request {request_id}")
        cur = self.db.execute(
            "UPDATE help_requests SET state = 'withdrawn'"
            " WHERE id = ? AND state = 'open'",
            (record["id"],))
        if not cur.rowcount:
            raise HelpError(
                f"help request {request_id} changed under the "
                "withdrawal")
        return self.get(request_id) or record

    def requests_for_ref(self, ref: str, *, state: str = "",
                         limit: int = 50) -> list[dict[str, Any]]:
        if state:
            rows = self.db.query(
                "SELECT * FROM help_requests WHERE ref = ?"
                " AND state = ? ORDER BY rowid ASC LIMIT ?",
                (str(ref), state, int(limit)))
        else:
            rows = self.db.query(
                "SELECT * FROM help_requests WHERE ref = ?"
                " ORDER BY rowid ASC LIMIT ?",
                (str(ref), int(limit)))
        return [_help_dict(row) for row in rows]

    def open_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM help_requests WHERE state = 'open'"
            " ORDER BY rowid ASC LIMIT ?", (int(limit),))
        return [_help_dict(row) for row in rows]


def request_help(db: Any, ref: str, *, requester: str, kind: str,
                 detail: str, helper: str = "",
                 now: float | None = None) -> dict[str, Any]:
    """Module-level convenience over :class:`HelpStore` — the
    one-call shape an organ mid-operation uses."""
    return HelpStore(db).request_help(
        ref, requester=requester, kind=kind, detail=detail,
        helper=helper, now=now)
