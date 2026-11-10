"""The grant registry — the durable record of timed grants (doc 20 P7).

Until this pass, :class:`~nomorals.core.policy.TimedGrant` existed
only as an in-hand object: ``Policy.issue_grant`` minted it, the
holder checked it, and ``Policy.revoke_grant`` could revoke it —
but only if the caller still HELD the object. Nothing recorded
that a grant had ever been issued, so revocation by identity (the
immune police's disarm rung is handed a grant *id*) had nothing to
act on: the police's ``revoke_fn`` was wired nowhere, and a
"grant revoked" claim on a healing ticket graded G0/unwired.

This module is the missing record. It is a path-backed store in
the shipped immune-store idiom (atomic tmp + fsync +
``os.replace``; ``path=None`` is in-memory), holding one row per
issued grant — who/what it was issued to (``actor`` / ``subject``),
its scope (``patterns``), its issuance seam (``source``), its
issue/expiry stamps, and its revocation state — plus an append-
only event journal (issued / revoked) inside the same file. The
registry also keeps a bounded table of the live grant objects
issued in this process, so a revocation by id flips the actual
object (``grant.grants()`` starts answering False immediately),
not just the row.

Expiry is DERIVED, never stored as a state: a row's standing is
``revoked`` when a revocation is recorded, else ``expired`` once
``expires_at`` passes, else ``active`` — so the registry's checker
(:meth:`GrantRegistry.grants`) and the live object's own
``grants()`` can never disagree about time.

Revocation semantics: :meth:`GrantRegistry.revoke` raises
:class:`GrantNotFound` for an unknown id (a revocation that names
nothing must not be recordable as performed — the police receipt
machinery turns the raise into a failed effect, in words), and is
idempotent for a known id (re-revoking returns the row marked
``already_revoked`` and records no second event).

Layer: core (L1). The police factories at the bottom are the
wiring the immune surfaces consume: immune code never imports
``core.policy`` for revocation — it calls the registry by id.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .failures import record_failure
from .logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "GrantNotFound",
    "GrantRegistry",
    "default_grant_registry",
    "default_grants_path",
    "police_revoke_fn",
    "reset_shared_grant_registries",
    "shared_grant_registry",
]

_FORMAT = "core.grant_registry.v1"
_MAX_EVENTS = 1000
_MAX_LIVE = 512


class GrantNotFound(KeyError):
    """No grant with that id exists in the registry.

    A KeyError (an unknown id IS a missing key), with a plain
    ``str()`` — KeyError's repr-quoting would garble the message
    where it is quoted as an outcome (police receipts)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def default_grants_path() -> Path:
    """``<home>/data/grant-registry.json`` — home is ``$NM_HOME``
    when set, else ``~/.nomorals`` (the braineval-store idiom)."""
    home = os.environ.get("NM_HOME", "").strip()
    base = Path(home) if home else Path.home() / ".nomorals"
    return base / "data" / "grant-registry.json"


class GrantRegistry:
    """Durable record of issued timed grants, keyed by grant id."""

    def __init__(self, path: Any = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._grants: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._live: dict[str, Any] = {}
        if self.path is not None:
            self._load()

    # ── persistence (the antibody-store idiom) ─────────────────

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            payload = json.load(fh)
        self._grants = {
            str(gid): dict(row)
            for gid, row in (payload.get("grants") or {}).items()
        }
        self._events = [dict(e) for e in payload.get("events") or []]

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = {
                "format": _FORMAT,
                "saved_ts": time.time(),
                "grants": self._grants,
                "events": self._events[-_MAX_EVENTS:],
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)

    # ── issuance ───────────────────────────────────────────────

    def record_issuance(self, grant: Any, *, source: str = "",
                        subject: str = "") -> dict[str, Any]:
        """Record one issued grant (the object is duck-typed: a
        :class:`~nomorals.core.policy.TimedGrant`). Returns the row.

        Re-recording the same grant (a holder re-presenting it)
        refreshes the live reference but never resurrects a
        recorded revocation and never writes a second issued
        event."""
        grant_id = str(grant.grant_id)
        ttl_s = float(grant.ttl_s)
        expires_at = float(grant.expires_at)
        with self._lock:
            existing = self._grants.get(grant_id)
            if existing is not None and existing.get("state") == "revoked":
                self._remember_live(grant_id, grant)
                return dict(existing)
            row = {
                "grant_id": grant_id,
                "patterns": sorted(grant.patterns),
                "actor": str(grant.actor or ""),
                "subject": str(subject or ""),
                "note": str(grant.note or ""),
                "source": str(source or ""),
                "issued_at": round(expires_at - ttl_s, 3),
                "expires_at": round(expires_at, 3),
                "ttl_s": ttl_s,
                "state": "active",
                "revoked_at": None,
                "revoked_by": "",
                "revoke_note": "",
            }
            is_new = existing is None
            self._grants[grant_id] = row
            self._remember_live(grant_id, grant)
            if is_new:
                self._events.append({
                    "ts": time.time(), "kind": "issued",
                    "grant_id": grant_id,
                    "actor": row["actor"],
                    "detail": (f"issued via {row['source'] or 'policy'} "
                               f"to {row['subject'] or row['actor'] or '?'}: "
                               f"{', '.join(row['patterns'])} "
                               f"for {ttl_s:g}s"),
                })
            self.save()
            return dict(row)

    def _remember_live(self, grant_id: str, grant: Any) -> None:
        """Keep the live object for in-process revocation. Bounded:
        past _MAX_LIVE the oldest reference is dropped — its row
        still flips on revoke; only the object flip is lost (such
        a grant is, at that volume, long past its TTL anyway)."""
        self._live[grant_id] = grant
        while len(self._live) > _MAX_LIVE:
            self._live.pop(next(iter(self._live)))

    # ── reads ──────────────────────────────────────────────────

    def get(self, grant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._grants.get(str(grant_id))
            return dict(row) if row is not None else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._grants.values()]

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events]

    @staticmethod
    def standing(row: dict[str, Any], *, now: float | None = None) -> str:
        """A row's standing at ``now``: revoked / expired / active.
        Expiry is derived from the recorded expiry stamp — never a
        stored state — so time alone moves a grant out of force."""
        if row.get("state") == "revoked":
            return "revoked"
        now = time.time() if now is None else now
        if now >= float(row.get("expires_at") or 0):
            return "expired"
        return "active"

    def grants(self, grant_id: str, capability: str, *,
               now: float | None = None) -> bool:
        """The registry-side checker: does this recorded grant
        currently authorize ``capability``? Mirrors the live
        object's ``grants()`` — same standing rules, same pattern
        semantics (CapabilitySet's) — for readers that hold only
        the id (the healing ticket, a restarted process)."""
        row = self.get(grant_id)
        if row is None or self.standing(row, now=now) != "active":
            return False
        from .policy import CapabilitySet

        return CapabilitySet(
            frozenset(row.get("patterns") or [])).grants(capability)

    def live_grant(self, grant_id: str) -> Any:
        """The live grant object issued in this process, or None."""
        with self._lock:
            return self._live.get(str(grant_id))

    # ── revocation ─────────────────────────────────────────────

    def revoke(self, grant_id: str, *, actor: str = "",
               note: str = "") -> dict[str, Any]:
        """Revoke a grant BY ID. Flips the row, flips the live
        object when this process holds it, and journals the event.

        Raises :class:`GrantNotFound` for an unknown id. Re-revoking
        a known id is idempotent: the row comes back marked
        ``already_revoked`` and no second event is recorded — the
        first revocation is the record."""
        grant_id = str(grant_id)
        with self._lock:
            row = self._grants.get(grant_id)
            if row is None:
                raise GrantNotFound(
                    f"no grant with id {grant_id!r} exists in the "
                    "grant registry — nothing was revoked")
            if row.get("state") == "revoked":
                out = dict(row)
                out["already_revoked"] = True
                return out
            row["state"] = "revoked"
            row["revoked_at"] = round(time.time(), 3)
            row["revoked_by"] = str(actor or "")
            row["revoke_note"] = str(note or "")
            live = self._live.get(grant_id)
            if live is not None:
                try:
                    live.revoke()
                except Exception as exc:  # noqa: BLE001 - the row is
                    record_failure("grant_registry.revoke_live", exc)
            self._events.append({
                "ts": time.time(), "kind": "revoked",
                "grant_id": grant_id, "actor": str(actor or ""),
                "detail": (note or "revoked")
                          + f" (grant to {row.get('subject') or row.get('actor') or '?'}: "
                            f"{', '.join(row.get('patterns') or [])})",
            })
            self.save()
            out = dict(row)
            out["already_revoked"] = False
            return out


# ── shared instance (the wiring pattern, one per path) ───────────

_SHARED: dict[str, GrantRegistry] = {}
_SHARED_LOCK = threading.Lock()


def shared_grant_registry(path: Any = None) -> GrantRegistry:
    """The process-wide grant registry for a path (one per path).

    Chat states, the CLI and the egress minter must all see ONE
    registry over one file — two instances over one file would
    lose each other's revocations on save. A registry whose file
    cannot load degrades to a private in-memory instance (and says
    so via record_failure): issuance must never break because the
    record of it failed."""
    resolved = Path(path) if path is not None else default_grants_path()
    key = str(resolved)
    with _SHARED_LOCK:
        registry = _SHARED.get(key)
        if registry is None:
            try:
                registry = GrantRegistry(resolved)
            except Exception as exc:  # noqa: BLE001 - degrade, loudly
                record_failure("grant_registry.shared_load", exc)
                registry = GrantRegistry()
            _SHARED[key] = registry
        return registry


def default_grant_registry() -> GrantRegistry:
    return shared_grant_registry()


def reset_shared_grant_registries() -> None:
    """Drop all shared registries (tests). The next call reloads
    from disk — also the restart simulation."""
    with _SHARED_LOCK:
        _SHARED.clear()


def police_revoke_fn(registry: GrantRegistry | None = None
                     ) -> Callable[[str], Any]:
    """The police's ``revoke_fn``: revocation by grant id, through
    :meth:`~nomorals.core.policy.Policy.revoke_issued_grant` so the
    policy audit records it too. The registry resolves at call
    time — building the closure cannot fail, so a surface can wire
    it unconditionally and still never break."""

    def _revoke(grant_id: str) -> dict[str, Any]:
        from .policy import Policy

        return Policy().revoke_issued_grant(
            str(grant_id), actor="immune.police", registry=registry)

    return _revoke
