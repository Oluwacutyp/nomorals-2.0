"""Cancellation receipts — the doc 18 P5 facade over the cancel shapes.

Cancellation existed in five incompatible shapes and a receipt in
none: delegation (the accountability organ) had no cancel path at
all, the job registry answered a bare bool over a cooperative flag,
browser and media runs carried P1's typed states on their durable
records, coremind answered a typed dict for queued jobs only, and
missions cancelled as a durable status flip. This module is the
ONE place a cancel becomes a *receipt*: :func:`cancel_with_receipt`
calls the organ's own mechanism — it never reimplements one —
captures the organ's typed answer, maps the answer to the uniform
receipt vocabulary through the explicit per-organ table
(:data:`CANCEL_MAPPINGS`), and writes a durable receipt row naming
who asked, what the organ answered, and what state the effect was
in at cancel time.

The mapping discipline is fail-closed on semantics and fail-open
on the cancel itself:

* A native answer whose word is not in its organ's mapping table
  is a MAPPING ERROR: no receipt is written (the table is data,
  pinned per organ by tests — a new native state must be added to
  it deliberately, never defaulted), the failure is recorded, and
  the caller still gets the native answer.
* A receipt write that fails (or no database handle at all) never
  blocks or undoes the cancel: the native mechanism has already
  executed, the failure is recorded, and the caller still gets the
  native answer with ``receipt`` None and ``receipt_error`` named.
* Where the organ's own mechanism raises (delegation's typed
  refusals, the media manager's unknown-job error, the mission
  store's NotFound), the facade receipts the classified answer and
  then re-raises the organ's exception unchanged — caller-visible
  behavior is exactly the organ's.

The organs are caller-supplied handles (the run_ticket precedent):
live managers and registries are process objects, not database
stores, so the facade wraps instances it is handed and imports no
organ. ``db`` is used only for the receipt table itself. The one
exception is the job registry: ``core.jobs`` is a core peer, so a
``job:`` cancel with no registry supplied falls back to the shared
registry, exactly as the ticket projection does.

Receipts are evidence about a cancel REQUEST and its answer at
that moment; they are never rewritten afterwards. A browser run's
later confirmation or dishonor lands on P1's run record (the
ticket reads it there); the receipt keeps the answer it was
minted with, and a second cancel mints a second receipt.

Layer: core (L1) — imported downward by agents (the ticket
projection reads receipts through this module's readers). Imports
stdlib + core peers only; organ handles are duck-typed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .failures import record_failure
from .jobs import shared_registry

__all__ = [
    "RECEIPT_STATES",
    "CANCEL_MAPPINGS",
    "UnmappedCancelState",
    "CancelReceiptStore",
    "cancel_with_receipt",
    "map_cancel_state",
    "latest_receipt_for_ref",
    "receipts_for_ref",
]

#: The uniform receipt vocabulary. ``requested`` — the organ took
#: the request; the effect's fate is not yet proven. ``confirmed``
#: — the cancel landed: the work never ran, or stands cancelled on
#: the organ's own record. ``unhonored`` — a request was made and
#: the work ended on its own terms anyway. ``refused`` — the organ
#: declined: the work is live and not interruptible there, already
#: terminal, or fenced. ``not_found`` — no such operation.
RECEIPT_STATES = ("requested", "confirmed", "unhonored",
                  "refused", "not_found")

#: Per-organ mapping tables: the organ's native answer word → the
#: receipt state. These tables are the contract — a native word
#: that is not here does not get a receipt (see the module
#: docstring). The word vocabularies:
#:
#: * delegation — the contract's status word after the attempt
#:   (``cancelled`` also covers "already stands cancelled"), or
#:   ``stale_generation`` / ``not_found`` from its typed refusals.
#: * mission — ``<prior>->cancelled`` for a landed flip, the
#:   terminal status word for the facade's refusal (the store's
#:   own bulk_cancel skips terminal missions; the facade follows
#:   the organ's rule for the single case rather than rewriting a
#:   finished mission to cancelled), ``not_found``.
#: * jobs — the registry's bool, disambiguated by its own read:
#:   ``requested`` (flag set), ``already_terminal``, ``not_found``.
#: * browser — the manager's typed dict: its ``state`` word, with
#:   ``already_terminal`` qualified by the record's cancel_state
#:   (``own_terms`` = ended with no cancel ever recorded).
#: * media — the manager's returned status dict, qualified:
#:   ``cancelled_queued`` (removed before it started) /
#:   ``cancelled_running`` (cooperative cancel honored), plus
#:   ``requested`` and the ``already_<status>`` no-op answers.
#: * coremind — cancel_job's dict: ``cancelled`` when its
#:   ``cancelled`` flag is true, else its ``state`` word verbatim
#:   (``active`` = started, not interrupted there; ``queued`` = a
#:   record with no live queue entry in this process).
#: * computer_watch — the watch manager's typed dict: its
#:   ``state`` word, with ``already_terminal`` qualified by the
#:   record's cancel_state — the browser scheme, shared because
#:   the watch record carries the same cancel lifecycle.
CANCEL_MAPPINGS: dict[str, dict[str, str]] = {
    "delegation": {
        "cancelled": "confirmed",
        "completed": "refused",
        "waived": "refused",
        "refused": "refused",
        "stale_generation": "refused",
        "not_found": "not_found",
    },
    "mission": {
        "pending->cancelled": "confirmed",
        "running->cancelled": "confirmed",
        "paused->cancelled": "confirmed",
        "cancelled": "confirmed",
        "done": "refused",
        "failed": "refused",
        "not_found": "not_found",
    },
    "jobs": {
        "requested": "requested",
        "already_terminal": "refused",
        "not_found": "not_found",
    },
    "browser": {
        "requested": "requested",
        "not_found": "not_found",
        "already_terminal:confirmed": "confirmed",
        "already_terminal:unhonored": "unhonored",
        "already_terminal:requested": "requested",
        "already_terminal:own_terms": "refused",
    },
    "media": {
        "cancelled_queued": "confirmed",
        "cancelled_running": "confirmed",
        "requested": "requested",
        "already_done": "refused",
        "already_failed": "refused",
        "not_found": "not_found",
    },
    "coremind": {
        "cancelled": "confirmed",
        "not_found": "not_found",
        "active": "refused",
        "queued": "refused",
        "done": "refused",
        "failed": "refused",
        "lost": "refused",
    },
    # computer_watch — the watch manager's typed dict: its
    # ``state`` word, with ``already_terminal`` qualified by the
    # record's cancel_state — the browser table's exact scheme
    # (``own_terms`` = ended with no cancel ever recorded), because
    # the watch record carries the same cancel lifecycle.
    "computer_watch": {
        "requested": "requested",
        "not_found": "not_found",
        "already_terminal:confirmed": "confirmed",
        "already_terminal:unhonored": "unhonored",
        "already_terminal:requested": "requested",
        "already_terminal:own_terms": "refused",
    },
}

#: Reference prefixes (the ticket projection's shapes) → the organ
#: name its receipts and mapping table are filed under.
_REF_ORGANS = {
    "contract": "delegation",
    "mission": "mission",
    "job": "jobs",
    "browser": "browser",
    "render": "media",
    "media": "media",
    "mind": "coremind",
    "watch": "computer_watch",
}

_RECEIPT_DDL = """
CREATE TABLE IF NOT EXISTS cancel_receipts (
  receipt_id TEXT PRIMARY KEY,
  at REAL NOT NULL DEFAULT 0,
  ref TEXT NOT NULL DEFAULT '',
  organ TEXT NOT NULL DEFAULT '',
  requested_by TEXT NOT NULL DEFAULT '',
  reason TEXT NOT NULL DEFAULT '',
  native_state TEXT NOT NULL DEFAULT '',
  receipt_state TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_cancel_receipts_ref
  ON cancel_receipts(ref);
"""


class UnmappedCancelState(Exception):
    """A native cancel answer its organ's mapping table does not
    name. Raised inside the facade only — it never reaches the
    caller as an exception; it becomes the result's
    ``receipt_error`` (and a failure-ledger entry), because an
    unmapped answer means the table is behind the organ, and the
    honest response is a loud refusal to receipt, not a guess."""


def map_cancel_state(organ: str, native_state: str) -> str:
    """The receipt state for one native answer word, per the
    organ's mapping table. Raises :class:`UnmappedCancelState`
    for any word the table does not name."""
    table = CANCEL_MAPPINGS.get(organ)
    if table is None:
        raise UnmappedCancelState(f"no mapping table for {organ!r}")
    state = table.get(native_state)
    if state is None:
        raise UnmappedCancelState(
            f"organ {organ!r} answered {native_state!r}, which its "
            "cancellation mapping table does not name — add the "
            "word deliberately; the facade does not guess")
    return state


def _receipt_dict(row: Any) -> dict[str, Any]:
    return {
        "receipt_id": row["receipt_id"],
        "at": float(row["at"] or 0),
        "ref": row["ref"],
        "organ": row["organ"],
        "requested_by": row["requested_by"],
        "reason": row["reason"],
        "native_state": row["native_state"],
        "receipt_state": row["receipt_state"],
        "detail": row["detail"],
    }


def _table_missing(exc: Exception) -> bool:
    return "no such table" in str(exc)


def receipts_for_ref(db: Any, ref: str, limit: int = 50
                     ) -> list[dict[str, Any]]:
    """Every receipt minted for one reference, oldest first — the
    ticket projection's history read. Read-only: runs no DDL, and
    a database where the receipts table was never created reads
    as "no receipts" (the leases-reader precedent)."""
    try:
        rows = db.query(
            "SELECT * FROM cancel_receipts WHERE ref = ?"
            " ORDER BY rowid ASC LIMIT ?",
            (str(ref), int(limit)))
    except Exception as exc:  # noqa: BLE001 - no table = no receipts
        if _table_missing(exc):
            return []
        raise
    return [_receipt_dict(row) for row in rows]


def latest_receipt_for_ref(db: Any, ref: str) -> dict[str, Any] | None:
    """The most recent receipt for one reference, or None. Same
    read-only discipline as :func:`receipts_for_ref`."""
    try:
        row = db.query_one(
            "SELECT * FROM cancel_receipts WHERE ref = ?"
            " ORDER BY rowid DESC LIMIT 1",
            (str(ref),))
    except Exception as exc:  # noqa: BLE001 - no table = no receipts
        if _table_missing(exc):
            return None
        raise
    return _receipt_dict(row) if row is not None else None


class CancelReceiptStore:
    """The receipt journal over the shared ``Database`` (Batch B
    pattern — DDL in code, migrations untouched). One row per
    receipt ever minted; rows are never updated or deleted — the
    journal is append-only, like the claim ledger it sits beside
    in spirit."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_RECEIPT_DDL)

    def record(self, *, ref: str, organ: str, requested_by: str,
               reason: str, native_state: str, receipt_state: str,
               detail: str, now: float | None = None
               ) -> dict[str, Any]:
        now = time.time() if now is None else now
        receipt = {
            "receipt_id": f"cr-{uuid.uuid4().hex[:12]}",
            "at": float(now),
            "ref": str(ref),
            "organ": str(organ),
            "requested_by": str(requested_by or ""),
            "reason": str(reason or ""),
            "native_state": str(native_state),
            "receipt_state": str(receipt_state),
            "detail": str(detail or ""),
        }
        self.db.execute(
            "INSERT INTO cancel_receipts (receipt_id, at, ref,"
            " organ, requested_by, reason, native_state,"
            " receipt_state, detail) VALUES (?, ?, ?, ?, ?, ?,"
            " ?, ?, ?)",
            (receipt["receipt_id"], receipt["at"], receipt["ref"],
             receipt["organ"], receipt["requested_by"],
             receipt["reason"], receipt["native_state"],
             receipt["receipt_state"], receipt["detail"]))
        return receipt

    def for_ref(self, ref: str, limit: int = 50
                ) -> list[dict[str, Any]]:
        return receipts_for_ref(self.db, ref, limit=limit)

    def latest_for_ref(self, ref: str) -> dict[str, Any] | None:
        return latest_receipt_for_ref(self.db, ref)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM cancel_receipts ORDER BY rowid DESC"
            " LIMIT ?", (int(limit),))
        return [_receipt_dict(row) for row in rows]


# ── the facade ────────────────────────────────────────────────────


@dataclass
class _Outcome:
    """One organ's classified answer: the native word the mapping
    table reads, the receipt detail, the organ's raw answer for
    the caller, and — where the organ's mechanism itself raises —
    the exception to re-raise after the receipt is written."""

    word: str
    detail: str
    raw: Any = None
    raise_after: BaseException | None = None


def _cancel_delegation(ident: str, store: Any, *,
                       requested_by: str, reason: str,
                       generation: int | None) -> _Outcome:
    try:
        contract = store.cancel(ident, by=requested_by,
                                reason=reason, generation=generation)
    except Exception as exc:
        if type(exc).__name__ == "StaleGenerationError":
            return _Outcome("stale_generation", str(exc), None, exc)
        contract = None
        try:
            contract = store.get_contract(ident)
        except Exception:  # noqa: BLE001 - classification read only
            contract = None
        if contract is None:
            return _Outcome("not_found", str(exc), None, exc)
        status = str(contract.status or "")
        raw = {"contract_id": ident, "status": status,
               "generation": contract.generation}
        if status == "cancelled":
            return _Outcome(
                "cancelled",
                f"{exc} — the contract already stands cancelled, "
                "so the refusal changes nothing: the cancel of "
                "record stands", raw, exc)
        if status in ("completed", "waived", "refused"):
            return _Outcome(status, str(exc), raw, exc)
        return _Outcome(f"unclassified:{status}", str(exc), None, exc)
    detail = ""
    try:
        for claim in store.claims(ident):
            if claim.kind == "cancellation":
                detail = claim.claim
    except Exception:  # noqa: BLE001 - detail is best-effort color
        detail = ""
    if not detail:
        detail = (f"contract cancelled in delegation at generation "
                  f"{contract.generation}")
    return _Outcome(
        "cancelled", detail,
        {"contract_id": ident, "status": "cancelled",
         "generation": contract.generation})


def _cancel_mission(ident: str, store: Any, *, requested_by: str,
                    reason: str) -> _Outcome:
    try:
        mission = store.get(ident)
    except Exception as exc:
        return _Outcome("not_found", str(exc), None, exc)
    status = str(mission.status or "")
    if mission.terminal:
        return _Outcome(
            status,
            f"the mission already stands {status} — terminal "
            "missions are not cancellable (the store's own "
            "bulk_cancel skips them for the same reason: a finished "
            "mission's record is history, not a state to rewrite)",
            {"mission_id": ident, "status": status})
    note = reason or "cancelled"
    if requested_by:
        note = f"{note} (requested by {requested_by})"
    store.set_status(ident, "cancelled", note=note)
    details = {
        "pending": "the mission had not started — no step ran "
                   "under this cancel",
        "running": "the durable status flip is the cancel of "
                   "record; a live runner's in-flight step ends "
                   "under its own checkpoint discipline",
        "paused": "the mission was paused; the flip is the cancel "
                  "of record",
    }
    detail = details.get(
        status,
        f"the mission stood at {status}; the flip landed")
    return _Outcome(
        f"{status}->cancelled", detail,
        {"mission_id": ident, "status": "cancelled",
         "prior_status": status})


def _cancel_job(ident: str, registry: Any) -> _Outcome:
    reg = registry if registry is not None else shared_registry()
    if reg.cancel(ident):
        return _Outcome(
            "requested",
            "the registry set its cooperative cancel_requested "
            "flag; workers that poll the flag stop themselves — a "
            "worker that never polls is reported as still running, "
            "never as cancelled",
            {"job_id": ident, "cancel_requested": True})
    job = reg.poll(ident)
    if job is None:
        return _Outcome(
            "not_found",
            "no job with that id is in the registry (terminal "
            "jobs are swept after their TTL)",
            {"job_id": ident, "cancel_requested": False})
    snap = job.snapshot()
    detail = (f"the job already stands {snap['status']} — a "
              "finished job cannot be cancelled")
    if snap["status"] == "presumed-dead":
        detail += ("; the dead-man's switch fired, so its outcome "
                   "is UNKNOWN — the effect of any work it did is "
                   "unproven either way")
    return _Outcome("already_terminal", detail, snap)


def _cancel_browser(ident: str, manager: Any) -> _Outcome:
    answer = manager.cancel(ident)
    state = str(answer.get("state") or "")
    if state == "not_found":
        return _Outcome("not_found",
                        "no browser run with that id (or unique "
                        "prefix) exists in the manager", answer)
    if state == "requested":
        return _Outcome(
            "requested",
            f"cancel requested while the run stood at "
            f"{answer.get('status')}; it takes effect "
            f"{answer.get('takes_effect')} — the run record "
            "carries the final word (confirmed when honored, "
            "unhonored when the run ends on its own terms)",
            answer)
    if state == "already_terminal":
        cancel_state = str(answer.get("cancel_state") or "")
        if cancel_state in ("confirmed", "unhonored", "requested"):
            word = f"already_terminal:{cancel_state}"
        else:
            word = "already_terminal:own_terms"
        shown = cancel_state or "empty (no cancel was ever recorded)"
        return _Outcome(
            word,
            f"the run already finished at {answer.get('status')}; "
            f"its record's cancel_state is {shown}", answer)
    return _Outcome(f"unclassified:{state}",
                    "the browser manager answered a state word "
                    "the facade does not know", answer)


def _cancel_watch(ident: str, manager: Any) -> _Outcome:
    answer = manager.cancel(ident)
    state = str(answer.get("state") or "")
    if state == "not_found":
        return _Outcome("not_found",
                        "no computer watch with that id (or unique "
                        "prefix) exists in the manager", answer)
    if state == "requested":
        return _Outcome(
            "requested",
            f"cancel requested while the watch stood at "
            f"{answer.get('status')}; it takes effect "
            f"{answer.get('takes_effect')} — the watch record "
            "carries the final word (confirmed when honored, "
            "unhonored when the watch ends on its own terms)",
            answer)
    if state == "already_terminal":
        cancel_state = str(answer.get("cancel_state") or "")
        if cancel_state in ("confirmed", "unhonored", "requested"):
            word = f"already_terminal:{cancel_state}"
        else:
            word = "already_terminal:own_terms"
        shown = cancel_state or "empty (no cancel was ever recorded)"
        return _Outcome(
            word,
            f"the watch already finished at {answer.get('status')}; "
            f"its record's cancel_state is {shown}", answer)
    return _Outcome(f"unclassified:{state}",
                    "the watch manager answered a state word "
                    "the facade does not know", answer)


def _cancel_media(ident: str, manager: Any) -> _Outcome:
    try:
        answer = manager.cancel(ident)
    except Exception as exc:
        if "unknown media job" in str(exc):
            return _Outcome("not_found", str(exc), None, exc)
        raise
    status = str(answer.get("status") or "")
    if status == "cancelled":
        if not answer.get("started_at"):
            return _Outcome(
                "cancelled_queued",
                "the job was queued and was dropped before it "
                "started — it never ran", answer)
        return _Outcome(
            "cancelled_running",
            "the job stands at cancelled: a cancel requested "
            "while it ran was honored at a progress report (or "
            "the request landed before its worker picked it up)",
            answer)
    if status == "running":
        return _Outcome(
            "requested",
            "the job is running; the manager set its cancel "
            "request and the worker stops at its next progress "
            "report — a job that never reports progress cannot "
            "be interrupted mid-flight", answer)
    if status in ("done", "failed"):
        return _Outcome(
            f"already_{status}",
            f"the job already finished at {status}; the cancel "
            "was a no-op", answer)
    return _Outcome(f"unclassified:{status}",
                    "the media manager answered a status word "
                    "the facade does not know", answer)


def _cancel_coremind(ident: str, mind: Any) -> _Outcome:
    answer = mind.cancel_job(ident)
    word = ("cancelled" if answer.get("cancelled")
            else str(answer.get("state") or ""))
    return _Outcome(word, str(answer.get("detail") or ""), answer)


def cancel_with_receipt(db: Any, ref: str, *,
                        requested_by: str = "", reason: str = "",
                        generation: int | None = None,
                        delegation: Any = None, missions: Any = None,
                        registry: Any = None,
                        browser_manager: Any = None,
                        media_manager: Any = None,
                        coremind: Any = None,
                        watch_manager: Any = None,
                        now: float | None = None
                        ) -> dict[str, Any]:
    """Cancel one operation through its organ's own mechanism and
    mint the uniform receipt. ``ref`` uses the ticket
    projection's shapes (``contract:`` / ``mission:`` / ``job:`` /
    ``browser:`` / ``render:`` / ``media:`` / ``mind:`` /
    ``watch:``); each organ is the caller-supplied live handle
    its wrapper needs (``delegation`` a DelegationStore,
    ``missions`` a MissionStore, ``registry`` a JobRegistry — the
    shared registry when omitted, ``browser_manager`` /
    ``media_manager`` the P1 managers, ``coremind`` a CoreMind,
    ``watch_manager`` the computer watch manager).
    ``generation`` is honoured only by delegation, whose cancel
    is generation-fenced.

    Returns ``{"ref", "organ", "native", "native_state",
    "receipt", "receipt_error"}``: the organ's raw answer, the
    durable receipt row (or None), and the named reason whenever
    no receipt could be minted (unmapped answer / receipt write
    failed / missing handle). When the organ's mechanism itself
    raises, the classified receipt is written first and the
    organ's exception is then re-raised unchanged — so a raising
    organ's caller sees exactly the organ's behavior, with the
    receipt already durable in the store."""
    now = time.time() if now is None else now
    prefix, _, ident = str(ref or "").partition(":")
    organ = _REF_ORGANS.get(prefix)
    if organ is None or not ident:
        raise ValueError(
            f"not a cancellable reference {ref!r} — expected one "
            "of " + ", ".join(f"{p}:<id>" for p in sorted(_REF_ORGANS)))
    handles = {
        "delegation": delegation,
        "mission": missions,
        "browser": browser_manager,
        "media": media_manager,
        "coremind": coremind,
        "computer_watch": watch_manager,
    }
    if organ in handles and handles[organ] is None:
        return {
            "ref": str(ref), "organ": organ, "native": None,
            "native_state": "", "receipt": None,
            "receipt_error":
                f"no {organ} handle was supplied — the facade "
                "wraps live organ instances (handles are "
                "caller-supplied, the run_ticket precedent); "
                "nothing was cancelled",
        }
    if organ == "delegation":
        outcome = _cancel_delegation(
            ident, delegation, requested_by=requested_by,
            reason=reason, generation=generation)
    elif organ == "mission":
        outcome = _cancel_mission(
            ident, missions, requested_by=requested_by,
            reason=reason)
    elif organ == "jobs":
        outcome = _cancel_job(ident, registry)
    elif organ == "browser":
        outcome = _cancel_browser(ident, browser_manager)
    elif organ == "media":
        outcome = _cancel_media(ident, media_manager)
    elif organ == "computer_watch":
        outcome = _cancel_watch(ident, watch_manager)
    else:
        outcome = _cancel_coremind(ident, coremind)
    receipt = None
    receipt_error = None
    try:
        receipt_state = map_cancel_state(organ, outcome.word)
    except UnmappedCancelState as exc:
        receipt_error = str(exc)
        record_failure("cancellation.unmapped_native_state", exc)
    if receipt_error is None:
        if db is None:
            receipt_error = (
                "no database handle was supplied — receipts are "
                "durable records on the shared Database; the "
                "cancel executed but no receipt could be written")
            record_failure("cancellation.receipt_write",
                           RuntimeError(receipt_error))
        else:
            try:
                receipt = CancelReceiptStore(db).record(
                    ref=str(ref), organ=organ,
                    requested_by=requested_by, reason=reason,
                    native_state=outcome.word,
                    receipt_state=receipt_state,
                    detail=outcome.detail, now=now)
            except Exception as exc:  # noqa: BLE001 - fail-open
                receipt_error = f"receipt write failed: {exc}"
                record_failure("cancellation.receipt_write", exc)
    result = {
        "ref": str(ref), "organ": organ, "native": outcome.raw,
        "native_state": outcome.word, "receipt": receipt,
        "receipt_error": receipt_error,
    }
    if outcome.raise_after is not None:
        raise outcome.raise_after
    return result
