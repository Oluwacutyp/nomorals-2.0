"""Consent ledger for owner-data egress (Echelon Wave 4).

Which destinations may receive which kinds of the owner's data is a
*recorded decision*, not a vibe.  This ledger is Devon's version of
craft-ai's ConsentLedger, homed with the capability policy
(:mod:`nomorals.core.policy`): every check returns a
:class:`~nomorals.core.policy.PolicyDecision`, and a denied check
raises :class:`~nomorals.core.errors.CapabilityDenied` — code
``policy.denied`` — when required.

A grant is keyed, as in the donor design, by

    app × connection × destination × capability × data-kind

and carries who granted it, when, and an optional expiry.  Grants are
*data*: they list, serialize, persist (an optional JSON file, written
atomically), expire, and revoke.  Matching is exact on every key
field — no globs, no prefix magic — so the read surface never
overstates what was consented to.  The one wildcard is a grant whose
``data_kinds`` contains ``"*"``, which covers every data kind *for
that exact key* and reads as such in the record.

The personal-surface lesson of 2026-11-04 is why this exists: data
leaving the machine for a third-party provider must trace to a
recorded owner grant, and "no grant on record" must be a denial with
a reason, not a silent pass.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import CapabilityDenied
from .policy import PolicyDecision

__all__ = [
    "DATA_KINDS",
    "ConsentLedger",
    "ConsentRecord",
]

#: The data kinds the ledger knows by name.  The set is open — a grant
#: may name any kind string — but these are the kinds surfaces use, so
#: a typo'd kind in a *check* is visible against this list.
DATA_KINDS: tuple[str, ...] = (
    "chat_history",     # prior conversation turns
    "prompts",          # the current instruction / prompt text
    "documents",        # owner documents and files
    "images",           # image bytes / vision payloads
    "audio",            # voice recordings and transcripts-to-be
    "code",             # source code from the owner's machines
    "personal_data",    # names, contacts, calendar, anything identifying
    "embeddings_text",  # raw text sent to an embeddings endpoint
)


@dataclass
class ConsentRecord:
    """One recorded grant (or its revoked/expired remnant)."""

    id: str
    app: str
    connection: str
    destination: str
    capability: str
    data_kinds: tuple[str, ...]
    granted_by: str
    granted_at: float
    expires_at: float | None = None
    revoked_at: float | None = None
    revoked_by: str = ""
    note: str = ""

    def covers(self, data_kind: str) -> bool:
        return "*" in self.data_kinds or data_kind in self.data_kinds

    def active(self, now: float) -> bool:
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or now < self.expires_at

    def matches(self, *, app: str, connection: str, destination: str,
                capability: str, data_kind: str) -> bool:
        return (
            self.app == app
            and self.connection == connection
            and self.destination == destination
            and self.capability == capability
            and self.covers(data_kind)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "app": self.app,
            "connection": self.connection,
            "destination": self.destination,
            "capability": self.capability,
            "data_kinds": list(self.data_kinds),
            "granted_by": self.granted_by,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
            "revoked_by": self.revoked_by,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ConsentRecord":
        return cls(
            id=str(row["id"]),
            app=str(row.get("app", "")),
            connection=str(row.get("connection", "")),
            destination=str(row.get("destination", "")),
            capability=str(row.get("capability", "")),
            data_kinds=tuple(str(k) for k in row.get("data_kinds") or ()),
            granted_by=str(row.get("granted_by", "")),
            granted_at=float(row.get("granted_at") or 0.0),
            expires_at=(float(row["expires_at"])
                        if row.get("expires_at") is not None else None),
            revoked_at=(float(row["revoked_at"])
                        if row.get("revoked_at") is not None else None),
            revoked_by=str(row.get("revoked_by", "")),
            note=str(row.get("note", "")),
        )


class ConsentLedger:
    """Recorded owner consent for data egress, keyed as data.

    In-memory by default; pass ``path`` for a JSON-backed ledger that
    survives restarts.  A corrupt backing file raises — a ledger that
    silently "forgets" every grant (or invents them) is worse than one
    that refuses to load.
    """

    def __init__(self, path: str | Path | None = None, *,
                 clock=time.time) -> None:
        self._path = Path(path) if path is not None else None
        self._clock = clock
        self._records: list[ConsentRecord] = []
        if self._path is not None and self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"consent ledger at {self._path} is unreadable: {exc}"
                ) from exc
            rows = payload.get("records") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError(
                    f"consent ledger at {self._path} has no records list")
            self._records = [ConsentRecord.from_dict(r) for r in rows]

    # ── writes ───────────────────────────────────────────────────────────────
    def grant(
        self,
        *,
        destination: str,
        data_kinds: Iterable[str],
        granted_by: str,
        app: str = "devon",
        connection: str = "default",
        capability: str = "llm.chat",
        ttl_s: float | None = None,
        note: str = "",
    ) -> ConsentRecord:
        """Record a grant.  Returns the stored record (with its id)."""
        kinds = tuple(str(k).strip() for k in data_kinds if str(k).strip())
        if not destination or not str(destination).strip():
            raise ValueError("consent grant needs a destination")
        if not kinds:
            raise ValueError("consent grant needs at least one data kind")
        if not granted_by or not str(granted_by).strip():
            raise ValueError("consent grant must name who granted it")
        now = float(self._clock())
        record = ConsentRecord(
            id=uuid.uuid4().hex[:16],
            app=app, connection=connection,
            destination=str(destination).strip(),
            capability=capability, data_kinds=kinds,
            granted_by=str(granted_by), granted_at=now,
            expires_at=(now + float(ttl_s)) if ttl_s is not None else None,
            note=note,
        )
        self._records.append(record)
        self._save()
        return record

    def revoke(self, record_id: str, *, revoked_by: str = "") -> bool:
        """Revoke one grant by id.  True when a live record was revoked."""
        now = float(self._clock())
        for record in self._records:
            if record.id == record_id and record.revoked_at is None:
                record.revoked_at = now
                record.revoked_by = revoked_by
                self._save()
                return True
        return False

    # ── reads ────────────────────────────────────────────────────────────────
    def records(self, *, active_only: bool = False) -> list[ConsentRecord]:
        """Every recorded grant (the honest read surface)."""
        now = float(self._clock())
        if active_only:
            return [r for r in self._records if r.active(now)]
        return list(self._records)

    def find(self, *, app: str = "devon", connection: str = "default",
             destination: str, capability: str = "llm.chat",
             data_kind: str, now: float | None = None) -> ConsentRecord | None:
        """The active record covering this exact key + kind, if any."""
        at = float(self._clock()) if now is None else float(now)
        for record in self._records:
            if record.active(at) and record.matches(
                app=app, connection=connection, destination=destination,
                capability=capability, data_kind=data_kind,
            ):
                return record
        return None

    def check(self, *, destination: str, data_kind: str,
              app: str = "devon", connection: str = "default",
              capability: str = "llm.chat",
              now: float | None = None) -> PolicyDecision:
        """May ``destination`` receive ``data_kind`` under this key?

        Allowed only under a recorded, unexpired, unrevoked grant.
        The decision is a policy decision: deny carries the reason,
        and :meth:`PolicyDecision.raise_if_denied` raises
        ``CapabilityDenied`` (code ``policy.denied``).
        """
        record = self.find(app=app, connection=connection,
                           destination=destination, capability=capability,
                           data_kind=data_kind, now=now)
        if record is not None:
            return PolicyDecision(
                allowed=True,
                reason=(f"consent recorded ({record.id}) by "
                        f"{record.granted_by}"),
                capability=capability, actor=app,
            )
        return PolicyDecision(
            allowed=False,
            reason=(f"no consent on record for {destination!r} to "
                    f"receive {data_kind!r} (app={app!r}, "
                    f"connection={connection!r}, "
                    f"capability={capability!r})"),
            capability=capability, actor=app,
        )

    def require(self, *, destination: str, data_kind: str,
                app: str = "devon", connection: str = "default",
                capability: str = "llm.chat",
                now: float | None = None) -> ConsentRecord:
        """The covering grant, or raise ``CapabilityDenied``.

        The denial carries code ``policy.denied`` and structured
        details (destination, data kind, key fields) so the failure
        ledger and the operator see exactly what was refused.
        """
        record = self.find(app=app, connection=connection,
                           destination=destination, capability=capability,
                           data_kind=data_kind, now=now)
        if record is not None:
            return record
        decision = self.check(destination=destination, data_kind=data_kind,
                              app=app, connection=connection,
                              capability=capability, now=now)
        raise CapabilityDenied(
            decision.reason,
            capability=capability, actor=app,
            details={
                "destination": destination,
                "data_kind": data_kind,
                "app": app,
                "connection": connection,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {"records": [r.to_dict() for r in self._records]}

    # ── persistence ──────────────────────────────────────────────────────────
    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8")
        os.replace(tmp, self._path)
