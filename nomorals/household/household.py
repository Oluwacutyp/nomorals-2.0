"""Household — the SHARED layer on top of per-person family state.

``nomorals.family`` owns per-person state: roster, roles, per-person
items, and the closed allowlist dispatcher (``FamilyRegistry``). This
module does NOT duplicate any of that. It adds the *shared* layer a
household needs on top:

* ``shared_commitments`` — what / owner_person / due / status
* ``shared_calendar`` — title / ts / person_id
* ``shared_assets`` — name / owner_person

Shared records live in SQLite under the household root
(``household_shared.db``) and are visible to every *active* household
member. Per-person items stay in family: :meth:`Household.get_visible`
delegates to ``family_item_list`` so private items are visible only
to their owner and the household owner, and shared items to members —
the family store enforces that, not this module.

Roles: enforcement is delegated to the family registry where
possible (membership, revoke, audit reads, item visibility). Where
this module checks a role itself it uses ``nomorals.family.roles``
directly — hierarchy ``owner (4) > adult (3) > teen (2) > child (1)``
— and documents it here. Revoke semantics are delegated, not
duplicated: :meth:`Household.revoke_member` calls
``family_person_revoke`` on the registry.

The audit log is append-only: entries are inserted, never updated
or deleted, and reading it requires the owner role.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from ..family import roles as roles_mod

__all__ = ["Household", "ROLE_RANK"]

#: Role hierarchy, mirroring ``nomorals.family.roles``:
#: owner (4) > adult (3) > teen (2) > child (1).
ROLE_RANK: dict[str, int] = {"child": 1, "teen": 2, "adult": 3, "owner": 4}


def _actor_id(actor: Any) -> str:
    """Accept an actor id string or a dict with ``id``/``actor_id``."""
    if isinstance(actor, dict):
        for key in ("actor_id", "id", "person_id"):
            if actor.get(key):
                return str(actor[key])
        raise ValueError("actor dict must carry an id")
    return str(actor)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Household:
    """Shared household layer wrapping a :class:`FamilyRegistry`.

    Accepts a ``FamilyRegistry`` instance (``registry=``) or a root
    path (``root=``) from which a registry is built. When neither is
    given, the default family registry is built.
    """

    def __init__(self, registry: Any = None, root: Any = None) -> None:
        if registry is not None:
            self.registry = registry
        else:
            from ..family.registry import family_registry
            if root is not None:
                self.registry = family_registry(root)
            else:
                self.registry = family_registry()
        # Household root = the family household's root directory.
        family_household = getattr(self.registry, "household", None)
        raw_root = getattr(family_household, "root", None) if family_household is not None else None
        if root is not None and raw_root is None:
            raw_root = root
        self.root = Path(raw_root) if raw_root is not None else Path(".")
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.root / "household_shared.db"))
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS shared_commitments (
                   id TEXT PRIMARY KEY, what TEXT NOT NULL,
                   owner_person TEXT NOT NULL, due REAL,
                   status TEXT NOT NULL, created_by TEXT NOT NULL,
                   created_ts REAL NOT NULL)""")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS shared_calendar (
                   id TEXT PRIMARY KEY, title TEXT NOT NULL,
                   ts REAL NOT NULL, person_id TEXT NOT NULL,
                   created_by TEXT NOT NULL, created_ts REAL NOT NULL)""")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS shared_assets (
                   id TEXT PRIMARY KEY, name TEXT NOT NULL,
                   owner_person TEXT NOT NULL,
                   created_by TEXT NOT NULL, created_ts REAL NOT NULL)""")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS household_audit (
                   id TEXT PRIMARY KEY, ts REAL NOT NULL,
                   actor_id TEXT NOT NULL, action TEXT NOT NULL,
                   target TEXT NOT NULL, detail TEXT NOT NULL)""")
        self._db.commit()

    # ── membership helpers (delegated to family) ─────────────────
    def _member(self, person_id: str) -> dict[str, Any] | None:
        try:
            return self.registry.household.get_member(person_id)
        except Exception:  # noqa: BLE001 — unknown member → None
            return None

    def _require_active(self, actor: Any) -> dict[str, Any]:
        actor_id = _actor_id(actor)
        member = self._member(actor_id)
        if member is None or member.get("status") != "active":
            raise PermissionError(f"actor {actor_id!r} is not an active household member")
        return member

    def _require_owner(self, actor: Any) -> dict[str, Any]:
        member = self._require_active(actor)
        role = roles_mod.validate_role(member.get("role", ""))
        # Local role check, using the family hierarchy (owner > adult
        # > teen > child). Owner-only reads go through this.
        if ROLE_RANK.get(role, 0) < ROLE_RANK["owner"]:
            raise PermissionError(f"role {role!r} cannot perform an owner-only action")
        return member

    def _role_of(self, actor_id: str) -> str:
        member = self._member(actor_id)
        if member is None:
            raise PermissionError(f"unknown member {actor_id!r}")
        return roles_mod.validate_role(member.get("role", ""))

    # ── audit (append-only) ──────────────────────────────────────
    def _audit(self, actor_id: str, action: str, target: str = "",
               detail: str = "") -> dict[str, Any]:
        entry = {"id": _new_id(), "ts": time.time(), "actor_id": actor_id,
                 "action": action, "target": target, "detail": detail}
        self._db.execute(
            "INSERT INTO household_audit VALUES (?, ?, ?, ?, ?, ?)",
            (entry["id"], entry["ts"], entry["actor_id"], entry["action"],
             entry["target"], entry["detail"]))
        self._db.commit()
        return dict(entry)

    def audit_log(self, actor: Any) -> list[dict[str, Any]]:
        """Read the household audit log. Owner-only. Append-only log."""
        self._require_owner(actor)
        rows = self._db.execute(
            "SELECT * FROM household_audit ORDER BY ts ASC").fetchall()
        return [dict(r) for r in rows]

    # ── roster / revoke (delegated to family, not duplicated) ────
    def add_member(self, *, actor_id: str, actor_role: str, person_id: str,
                   display_name: str, role: str, locale: str = "ng") -> dict[str, Any]:
        """Add a household member — delegated to ``family_person_add``."""
        out = self.registry.call(
            "family_person_add", actor_id=actor_id, actor_role=actor_role,
            person_id=person_id, display_name=display_name, role=role, locale=locale)
        self._audit(actor_id, "person_add", person_id, f"role={role}")
        return out

    def list_members(self, *, actor_role: str, include_revoked: bool = False) -> list[dict[str, Any]]:
        """List the roster — delegated to ``family_person_list``."""
        return self.registry.call(
            "family_person_list", actor_role=actor_role,
            include_revoked=include_revoked)

    def revoke_member(self, *, actor_id: str, actor_role: str,
                      person_id: str, reason: str = "") -> dict[str, Any]:
        """Revoke a member — delegated to ``family_person_revoke``.

        Revoke semantics (immediate access cut, data kept owner-only,
        last owner protected) live in family; they are NOT duplicated
        here. This method only forwards and audit-logs the call.
        """
        out = self.registry.call(
            "family_person_revoke", actor_id=actor_id, actor_role=actor_role,
            person_id=person_id, reason=reason)
        self._audit(actor_id, "person_revoke", person_id, reason)
        return out

    # ── shared commitments ───────────────────────────────────────
    def add_shared_commitment(self, actor: Any, what: str, *,
                              owner_person: str | None = None,
                              due: float | None = None,
                              status: str = "open") -> dict[str, Any]:
        """Add a shared commitment. Any active member may add one."""
        member = self._require_active(actor)
        actor_id = member["person_id"]
        if not (what or "").strip():
            raise ValueError("what is required")
        owner_person = owner_person or actor_id
        target = self._member(owner_person)
        if target is None or target.get("status") != "active":
            raise PermissionError(f"owner_person {owner_person!r} is not an active member")
        entry = {"id": _new_id(), "what": what, "owner_person": owner_person,
                 "due": due, "status": status,
                 "created_by": actor_id, "created_ts": time.time()}
        self._db.execute(
            "INSERT INTO shared_commitments VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry["id"], entry["what"], entry["owner_person"], entry["due"],
             entry["status"], entry["created_by"], entry["created_ts"]))
        self._db.commit()
        self._audit(actor_id, "shared_commitment_add", entry["id"], what)
        return dict(entry)

    def list_shared_commitments(self, actor: Any) -> list[dict[str, Any]]:
        """Shared commitments — visible to every active member."""
        self._require_active(actor)
        rows = self._db.execute(
            "SELECT * FROM shared_commitments ORDER BY created_ts ASC").fetchall()
        return [dict(r) for r in rows]

    # ── shared calendar ──────────────────────────────────────────
    def add_shared_calendar(self, actor: Any, title: str, ts: float, *,
                            person_id: str | None = None) -> dict[str, Any]:
        """Add a shared calendar entry. Any active member may add one."""
        member = self._require_active(actor)
        actor_id = member["person_id"]
        if not (title or "").strip():
            raise ValueError("title is required")
        person_id = person_id or actor_id
        target = self._member(person_id)
        if target is None or target.get("status") != "active":
            raise PermissionError(f"person_id {person_id!r} is not an active member")
        entry = {"id": _new_id(), "title": title, "ts": float(ts),
                 "person_id": person_id,
                 "created_by": actor_id, "created_ts": time.time()}
        self._db.execute(
            "INSERT INTO shared_calendar VALUES (?, ?, ?, ?, ?, ?)",
            (entry["id"], entry["title"], entry["ts"], entry["person_id"],
             entry["created_by"], entry["created_ts"]))
        self._db.commit()
        self._audit(actor_id, "shared_calendar_add", entry["id"], title)
        return dict(entry)

    def list_shared_calendar(self, actor: Any) -> list[dict[str, Any]]:
        """Shared calendar — visible to every active member."""
        self._require_active(actor)
        rows = self._db.execute(
            "SELECT * FROM shared_calendar ORDER BY ts ASC, created_ts ASC").fetchall()
        return [dict(r) for r in rows]

    # ── shared assets ────────────────────────────────────────────
    def add_shared_asset(self, actor: Any, name: str, *,
                         owner_person: str | None = None) -> dict[str, Any]:
        """Add a shared asset. Any active member may add one."""
        member = self._require_active(actor)
        actor_id = member["person_id"]
        if not (name or "").strip():
            raise ValueError("name is required")
        owner_person = owner_person or actor_id
        target = self._member(owner_person)
        if target is None or target.get("status") != "active":
            raise PermissionError(f"owner_person {owner_person!r} is not an active member")
        entry = {"id": _new_id(), "name": name, "owner_person": owner_person,
                 "created_by": actor_id, "created_ts": time.time()}
        self._db.execute(
            "INSERT INTO shared_assets VALUES (?, ?, ?, ?, ?)",
            (entry["id"], entry["name"], entry["owner_person"],
             entry["created_by"], entry["created_ts"]))
        self._db.commit()
        self._audit(actor_id, "shared_asset_add", entry["id"], name)
        return dict(entry)

    def list_shared_assets(self, actor: Any) -> list[dict[str, Any]]:
        """Shared assets — visible to every active member."""
        self._require_active(actor)
        rows = self._db.execute(
            "SELECT * FROM shared_assets ORDER BY created_ts ASC").fetchall()
        return [dict(r) for r in rows]

    # ── visibility of per-person items (delegated to family) ─────
    def get_visible(self, actor_id: str, person_id: str) -> list[dict[str, Any]]:
        """Per-person items of ``person_id`` visible to ``actor_id``.

        Delegates to ``family_item_list``: a member sees their own
        items (private + shared); the household owner additionally
        sees another member's private items (audit); any other member
        sees only that person's *shared* items. Revoked members see
        nothing — the family store enforces all of it.
        """
        self._require_active(actor_id)
        actor_role = self._role_of(actor_id)
        return self.registry.call(
            "family_item_list", actor_id=actor_id, actor_role=actor_role,
            person_id=person_id)

    def close(self) -> None:
        self._db.close()
