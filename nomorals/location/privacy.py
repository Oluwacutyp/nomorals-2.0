"""Consent: opt-in, per-person, per-scope, revocable. Default: NO.

Every location evaluation requires consent. There is no ambient
"the user probably agreed" — the caller passes ``consent=True`` with
the update, and a ``ConsentStore`` can back that decision with a
persisted grant. Revoking consent makes future evaluations refuse
until consent is granted again.

Scopes are plain strings so new uses get their own consent bucket,
e.g. ``"geofence"``, ``"reminders"``, ``"history"``. A grant for one
scope never implies another.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["ConsentError", "ConsentStore", "require_consent", "default_location_root"]


class ConsentError(PermissionError):
    """Raised when location is used without opt-in consent."""


def default_location_root() -> Path:
    """Location state root (consent records, fence states)."""
    return Path.home() / ".devon" / "location"


def require_consent(consent: bool, *, scope: str = "geofence") -> None:
    """Refuse unless ``consent`` is explicitly True. Fail-closed."""
    if consent is not True:
        raise ConsentError(
            f"location use refused: no opt-in consent for scope {scope!r}")


class ConsentStore:
    """Persisted per-person, per-scope consent grants (opt-in only)."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_location_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.root / "consent.json"

    def _load(self) -> dict[str, Any]:
        p = self._path()
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, Any]) -> None:
        self._path().write_text(json.dumps(data, indent=2, sort_keys=True),
                                encoding="utf-8")

    def grant(self, person_id: str, scope: str) -> dict[str, Any]:
        """Record an explicit opt-in grant. Returns the grant record."""
        data = self._load()
        key = f"{person_id}:{scope}"
        rec = {"person_id": person_id, "scope": scope,
               "granted_ts": time.time(), "revoked_ts": None}
        data[key] = rec
        self._save(data)
        return dict(rec)

    def revoke(self, person_id: str, scope: str) -> dict[str, Any] | None:
        """Revoke a grant. Later evaluations for this scope refuse."""
        data = self._load()
        key = f"{person_id}:{scope}"
        rec = data.get(key)
        if rec is None:
            return None
        rec["revoked_ts"] = time.time()
        self._save(data)
        return dict(rec)

    def has_consent(self, person_id: str, scope: str) -> bool:
        """True iff a grant exists and has not been revoked."""
        rec = self._load().get(f"{person_id}:{scope}")
        return bool(rec and rec.get("granted_ts") and not rec.get("revoked_ts"))

    def check(self, person_id: str, scope: str) -> None:
        """Raise ConsentError unless a live grant exists."""
        if not self.has_consent(person_id, scope):
            raise ConsentError(
                f"no opt-in consent for {person_id!r} scope {scope!r}")
