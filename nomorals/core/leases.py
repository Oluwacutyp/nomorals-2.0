"""Named-resource leases — the doc 18 P3 lease authority.

Exclusive, generation-fenced leases over NAMED resources (a
browser profile, an account grant, a shared file, a GPU, a paid
quota): two operations never hold the same exclusive resource at
once, a second acquirer gets a typed contention answer naming the
current holder and its expiry — never a silent wait, never a
steal — and every lease carries a fencing generation so a holder
whose lease lapsed and was re-acquired by someone else is refused
on every later mutation (doc §6: "rush cannot steal it; lease
expiry/fencing protects interrupted process").

This module is the AUTHORITY: the only place lease state is
minted or mutated. It stores lease records and nothing else — it
enforces nothing about what holders DO with a resource (doc 18
P4's admission is the first consumer; doc 20's Plane-F
enforcement builds on :meth:`LeaseStore.check`). The doc 18 P2
ticket projection reads live leases from here (via the
module-level readers, which never run DDL — observation writes
nothing).

Fencing: ``generation`` is per-resource and monotonic — it
increments on EVERY acquisition of the resource and is never
reused, even after release or expiry. Every mutating call
(``renew``/``release``) names the generation it believes is
current; when the resource's current generation has moved past
it, the call is refused ``stale_generation`` — a fenced-out
ex-holder cannot renew, release, or otherwise touch the lease
its successor holds.

Expiry is COMPUTED, never swept: a lease whose ``expires_at`` has
passed is expired at read/acquire time whether or not any sweep
ran (there is no background thread; correctness does not depend
on one). When a new acquisition supersedes a lapsed lease, the
lapsed row is marked ``expired`` inside the same transaction so
the record shows what happened.

Atomicity: acquire is check-and-set inside ONE write transaction
(``db.transaction()`` — BEGIN IMMEDIATE, the triggers store's
claim pattern), and a partial unique index guarantees at most
one row per resource is ever in the recorded ``held`` state, so
a double grant is structurally impossible rather than merely
unlikely.

Resource names are OPEN strings. :data:`KNOWN_RESOURCES` is the
doc §2A seed vocabulary — surveyable in code, useful for
surfaces — never a closed enum: a caller naming a resource the
vocabulary doesn't know yet gets a lease on it.

Layer: core (L1) — imported downward by agents (the ticket
projection, doc 20's enforcement), browser and media (L4)
consumers. Imports stdlib only; the database handle is the
shared ``Database`` duck type (``execute``/``query``/
``query_one``/``executescript``/``transaction``).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

__all__ = [
    "KNOWN_RESOURCES",
    "RESOURCE_ACCOUNT_GRANT",
    "RESOURCE_BROWSER_PROFILE",
    "RESOURCE_FILE",
    "RESOURCE_GPU",
    "RESOURCE_PAID_QUOTA",
    "RESOURCE_TOOL",
    "LeaseStore",
    "live_lease_for_resource",
    "live_leases_for_holder",
]

#: The doc §2A named resources — the seed vocabulary, NOT a
#: closed enum (see the module docstring).
RESOURCE_BROWSER_PROFILE = "browser-profile"
RESOURCE_FILE = "file"
RESOURCE_ACCOUNT_GRANT = "account-grant"
RESOURCE_TOOL = "tool"
RESOURCE_GPU = "gpu"
RESOURCE_PAID_QUOTA = "paid-quota"

KNOWN_RESOURCES = (
    RESOURCE_BROWSER_PROFILE,
    RESOURCE_FILE,
    RESOURCE_ACCOUNT_GRANT,
    RESOURCE_TOOL,
    RESOURCE_GPU,
    RESOURCE_PAID_QUOTA,
)

_LEASE_DDL = """
CREATE TABLE IF NOT EXISTS resource_leases (
  lease_id TEXT PRIMARY KEY,
  resource TEXT NOT NULL,
  holder_ref TEXT NOT NULL,
  generation INTEGER NOT NULL,
  state TEXT NOT NULL DEFAULT 'held',
  ttl_s REAL NOT NULL DEFAULT 0,
  acquired_at REAL NOT NULL DEFAULT 0,
  expires_at REAL NOT NULL DEFAULT 0,
  released_at REAL NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_resource_leases_one_held
  ON resource_leases(resource) WHERE state = 'held';
CREATE INDEX IF NOT EXISTS idx_resource_leases_holder
  ON resource_leases(holder_ref, state);
"""


def _effective_state(row: dict[str, Any], now: float) -> str:
    """The state a row is IN at ``now``: a recorded ``held`` past
    its expiry is expired whether or not anything marked it."""
    state = str(row.get("state") or "held")
    if state == "held" and float(row.get("expires_at") or 0) <= now:
        return "expired"
    return state


def _lease_dict(row: dict[str, Any], now: float) -> dict[str, Any]:
    """The public lease shape — the recorded row with its
    EFFECTIVE state at ``now`` (never the raw stored state)."""
    return {
        "lease_id": row["lease_id"],
        "resource": row["resource"],
        "holder_ref": row["holder_ref"],
        "generation": int(row["generation"]),
        "state": _effective_state(row, now),
        "ttl_s": float(row.get("ttl_s") or 0),
        "acquired_at": float(row.get("acquired_at") or 0),
        "expires_at": float(row.get("expires_at") or 0),
        "released_at": float(row.get("released_at") or 0),
    }


def _table_missing(exc: Exception) -> bool:
    return "no such table" in str(exc)


def live_leases_for_holder(db: Any, holder_ref: str,
                           now: float | None = None) -> list[dict]:
    """The LIVE leases one holder ref currently holds — the
    ticket projection's read. Read-only: runs no DDL, and a
    database where the lease table was never created reads as
    "holds nothing" (the run_ticket ``_rows`` precedent)."""
    now = time.time() if now is None else now
    try:
        rows = db.query(
            "SELECT * FROM resource_leases WHERE holder_ref = ?"
            " AND state = 'held' AND expires_at > ?"
            " ORDER BY acquired_at ASC",
            (str(holder_ref), now))
    except Exception as exc:  # noqa: BLE001 - no table = no leases
        if _table_missing(exc):
            return []
        raise
    return [_lease_dict(row, now) for row in rows]


def live_lease_for_resource(db: Any, resource: str,
                            now: float | None = None) -> dict | None:
    """The LIVE lease on one resource, or None — the operator
    read. Same read-only discipline as
    :func:`live_leases_for_holder`."""
    now = time.time() if now is None else now
    try:
        row = db.query_one(
            "SELECT * FROM resource_leases WHERE resource = ?"
            " AND state = 'held' AND expires_at > ?"
            " ORDER BY generation DESC LIMIT 1",
            (str(resource), now))
    except Exception as exc:  # noqa: BLE001 - no table = no lease
        if _table_missing(exc):
            return None
        raise
    return _lease_dict(row, now) if row is not None else None


class LeaseStore:
    """The lease authority over the shared ``Database`` (Batch B
    pattern — DDL in code, migrations untouched). One row per
    lease ever granted; rows are never deleted, so generations
    and history survive for the ticket/coverage reads."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_LEASE_DDL)

    # -- reads ----------------------------------------------------

    def get(self, lease_id: str,
            now: float | None = None) -> dict | None:
        """One lease by id, with its effective state at ``now``."""
        now = time.time() if now is None else now
        row = self.db.query_one(
            "SELECT * FROM resource_leases WHERE lease_id = ?",
            (str(lease_id),))
        return _lease_dict(row, now) if row is not None else None

    def current(self, resource: str,
                now: float | None = None) -> dict | None:
        """The live lease on ``resource``, or None."""
        return live_lease_for_resource(self.db, resource, now=now)

    def leases_for_resource(self, resource: str, *,
                            include_history: bool = False,
                            now: float | None = None) -> list[dict]:
        """Leases on one resource: the live one by default, or
        every lease ever granted (newest generation first) with
        ``include_history`` — the operator read."""
        now = time.time() if now is None else now
        if not include_history:
            live = self.current(resource, now=now)
            return [live] if live is not None else []
        rows = self.db.query(
            "SELECT * FROM resource_leases WHERE resource = ?"
            " ORDER BY generation DESC",
            (str(resource),))
        return [_lease_dict(row, now) for row in rows]

    def leases_held_by(self, holder_ref: str,
                       now: float | None = None) -> list[dict]:
        """The live leases one holder ref holds."""
        return live_leases_for_holder(self.db, holder_ref, now=now)

    def check(self, resource: str, holder_ref: str, generation: int,
              now: float | None = None) -> bool:
        """The fencing question, answered definitively: does
        ``holder_ref`` CURRENTLY hold ``resource`` at exactly
        ``generation`` (live, unexpired)? This is the read an
        enforcer gates a late write on — a holder whose lease
        lapsed, was released, or was superseded gets False."""
        live = self.current(resource, now=now)
        return bool(live is not None
                    and live["holder_ref"] == str(holder_ref)
                    and live["generation"] == int(generation))

    # -- mutations --------------------------------------------------

    def acquire(self, resource: str, holder_ref: str, ttl_s: float,
                now: float | None = None) -> dict[str, Any]:
        """Grant ``resource`` to ``holder_ref`` for ``ttl_s``, or
        refuse with the typed contention record. Granted:
        ``{"granted": True, "lease": {...}}`` (plus
        ``"already_held": True`` when the SAME holder already
        holds the live lease — an idempotent re-acquire returns
        the existing lease unchanged; extending is renew's job).
        Refused: ``{"granted": False, "reason": "held",
        "resource", "holder_ref", "lease_id", "generation",
        "expires_at"}`` — the current holder and its expiry,
        named. Never waits, never steals an unexpired lease."""
        now = time.time() if now is None else now
        resource = str(resource or "").strip()
        holder_ref = str(holder_ref or "").strip()
        if not resource or not holder_ref:
            raise ValueError("a lease needs a resource and a holder_ref")
        ttl = float(ttl_s)
        if ttl <= 0:
            raise ValueError("a lease needs a positive ttl_s")
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT * FROM resource_leases WHERE resource = ?"
                " AND state = 'held' ORDER BY generation DESC"
                " LIMIT 1",
                (resource,))
            if row is not None:
                live = _lease_dict(row, now)
                if live["state"] == "held":
                    if live["holder_ref"] == holder_ref:
                        return {"granted": True,
                                "already_held": True, "lease": live}
                    return {"granted": False, "reason": "held",
                            "resource": resource,
                            "holder_ref": live["holder_ref"],
                            "lease_id": live["lease_id"],
                            "generation": live["generation"],
                            "expires_at": live["expires_at"]}
                # The recorded-held row has lapsed: mark it in the
                # same transaction, then grant fresh.
                self.db.execute(
                    "UPDATE resource_leases SET state = 'expired'"
                    " WHERE lease_id = ?",
                    (row["lease_id"],))
            top = self.db.query_one(
                "SELECT MAX(generation) AS g FROM resource_leases"
                " WHERE resource = ?",
                (resource,))
            generation = int(top["g"] or 0) + 1 if top else 1
            lease = {
                "lease_id": f"lease-{uuid.uuid4().hex[:12]}",
                "resource": resource,
                "holder_ref": holder_ref,
                "generation": generation,
                "state": "held",
                "ttl_s": ttl,
                "acquired_at": now,
                "expires_at": now + ttl,
                "released_at": 0.0,
            }
            self.db.execute(
                "INSERT INTO resource_leases (lease_id, resource,"
                " holder_ref, generation, state, ttl_s, acquired_at,"
                " expires_at, released_at) VALUES (?, ?, ?, ?, ?,"
                " ?, ?, ?, ?)",
                (lease["lease_id"], resource, holder_ref, generation,
                 "held", ttl, now, now + ttl, 0.0))
            return {"granted": True, "lease": lease}

    def _fencing_refusal(self, lease_id: str, generation: int,
                         now: float) -> dict[str, Any] | None:
        """Classify a renew/release attempt: None when the caller
        is the lease's live holder at the current generation,
        else the typed refusal. Runs inside the caller's write
        transaction. The stale check compares the PRESENTED
        generation against the resource's CURRENT generation —
        once a successor holds the resource, the ex-holder's
        generation can never match again."""
        row = self.db.query_one(
            "SELECT * FROM resource_leases WHERE lease_id = ?",
            (str(lease_id),))
        if row is None:
            return {"reason": "not_found", "lease_id": str(lease_id)}
        top = self.db.query_one(
            "SELECT MAX(generation) AS g FROM resource_leases"
            " WHERE resource = ?",
            (row["resource"],))
        current_generation = int(top["g"] or 0) if top else 0
        if int(generation) != current_generation:
            return {"reason": "stale_generation",
                    "lease_id": str(lease_id),
                    "resource": row["resource"],
                    "current_generation": current_generation}
        state = _effective_state(row, now)
        if state != "held":
            return {"reason": state, "lease_id": str(lease_id),
                    "resource": row["resource"],
                    "current_generation": current_generation}
        return None

    def renew(self, lease_id: str, generation: int,
              ttl_s: float | None = None,
              now: float | None = None) -> dict[str, Any]:
        """Extend a live lease's expiry to ``now + ttl_s`` (the
        lease's own TTL when ``ttl_s`` is omitted). Refused —
        ``{"renewed": False, "reason": ...}`` — when the lease is
        unknown (``not_found``), the presented generation is no
        longer the resource's current one
        (``stale_generation``), or the lease has already ended
        (``expired`` / ``released``). The extension lands through
        a generation-CAS update (delegation's rowcount idiom), so
        a renew that loses a race writes nothing."""
        now = time.time() if now is None else now
        with self.db.transaction():
            refusal = self._fencing_refusal(lease_id, generation, now)
            if refusal is not None:
                return {"renewed": False, **refusal}
            row = self.db.query_one(
                "SELECT * FROM resource_leases WHERE lease_id = ?",
                (str(lease_id),))
            ttl = float(ttl_s) if ttl_s is not None \
                else float(row["ttl_s"])
            if ttl <= 0:
                raise ValueError("renew needs a positive ttl_s")
            new_expiry = now + ttl
            cursor = self.db.execute(
                "UPDATE resource_leases SET expires_at = ?"
                " WHERE lease_id = ? AND generation = ?"
                " AND state = 'held'",
                (new_expiry, str(lease_id), int(generation)))
            if not cursor.rowcount:
                return {"renewed": False, "reason": "stale_generation",
                        "lease_id": str(lease_id),
                        "resource": row["resource"]}
            return {"renewed": True,
                    "lease": self.get(lease_id, now=now)}

    def release(self, lease_id: str, generation: int,
                now: float | None = None) -> dict[str, Any]:
        """End a live lease now, freeing the resource for an
        immediate re-acquire (at the NEXT generation — generations
        are never reused). Refusals are the renew vocabulary:
        ``not_found`` / ``stale_generation`` / ``expired`` /
        ``released``. A release by a fenced-out ex-holder is
        refused and the successor's lease is untouched."""
        now = time.time() if now is None else now
        with self.db.transaction():
            refusal = self._fencing_refusal(lease_id, generation, now)
            if refusal is not None:
                return {"released": False, **refusal}
            cursor = self.db.execute(
                "UPDATE resource_leases SET state = 'released',"
                " released_at = ? WHERE lease_id = ?"
                " AND generation = ? AND state = 'held'",
                (now, str(lease_id), int(generation)))
            if not cursor.rowcount:
                return {"released": False,
                        "reason": "stale_generation",
                        "lease_id": str(lease_id)}
            return {"released": True,
                    "lease": self.get(lease_id, now=now)}
