"""Household roster + owner controls + audit log.

The household is the family boundary: members live in ``members.json``
at the family root, the audit log in ``audit.jsonl`` at the same
root, and each member's items in their own per-person directory
(``store.py``).

Owner controls, enforced here in code:

* Only an OWNER may add members, revoke members, or change roles.
* Bootstrap: the first member MUST be an owner (a household with no
  owner cannot exist).
* Revoking a member sets status ``revoked`` and cuts their access
  immediately (the store refuses revoked requesters). Their data is
  NOT deleted — it becomes owner-only for audit — and the revocation
  itself is audit-logged.
* The last owner cannot be revoked or demoted: a household always
  has exactly at least one owner.
* Every membership change is appended to the audit log. The audit
  log is append-only and readable only by an OWNER.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import roles as roles_mod
from .locales import DEFAULT_LOCALE
from .store import PerPersonStore, default_family_root, sanitize_person_id

__all__ = ["Household", "Member"]


class Member(dict):
    """A household member record (plain dict subclass for JSON ease)."""


class Household:
    """Owns the roster + audit log. Item data lives in PerPersonStore."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_family_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = PerPersonStore(self.root)

    # ── persistence ──────────────────────────────────────────────
    def _members_path(self) -> Path:
        return self.root / "members.json"

    def _audit_path(self) -> Path:
        return self.root / "audit.jsonl"

    def _load_members(self) -> dict[str, dict[str, Any]]:
        p = self._members_path()
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_members(self, members: dict[str, dict[str, Any]]) -> None:
        self._members_path().write_text(
            json.dumps(members, indent=2, sort_keys=True), encoding="utf-8")

    def _audit(self, actor_id: str, action: str, target: str = "",
               detail: str = "") -> dict[str, Any]:
        entry = {
            "ts": time.time(), "actor_id": actor_id,
            "action": action, "target": target, "detail": detail,
        }
        with self._audit_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    # ── roster ───────────────────────────────────────────────────
    def add_person(self, *, actor_id: str, actor_role: str,
                   person_id: str, display_name: str,
                   role: str, locale: str = DEFAULT_LOCALE) -> dict[str, Any]:
        """Add a member. OWNER-only, except the bootstrap first owner."""
        role = roles_mod.validate_role(role)
        members = self._load_members()
        if members:
            roles_mod.require(actor_role, "manage_members")
            if actor_id not in members or members[actor_id].get("status") != "active":
                raise PermissionError("only an active owner may add members")
        elif role != roles_mod.OWNER:
            raise ValueError("the first household member must be an owner")
        pid = sanitize_person_id(person_id)
        if pid in members and members[pid].get("status") == "active":
            raise ValueError(f"person {pid!r} is already a member")
        now = time.time()
        members[pid] = {
            "person_id": pid, "display_name": display_name, "role": role,
            "locale": locale or DEFAULT_LOCALE, "status": "active",
            "created_ts": now, "updated_ts": now,
        }
        self._save_members(members)
        self.store.put_profile(pid, {"display_name": display_name,
                                     "role": role, "locale": locale})
        self._audit(actor_id, "person_add", pid, f"role={role}")
        return dict(members[pid])

    def revoke_person(self, *, actor_id: str, actor_role: str,
                      person_id: str, reason: str = "") -> dict[str, Any]:
        """Revoke a member: immediate access cut, data kept owner-only."""
        roles_mod.require(actor_role, "manage_members")
        members = self._load_members()
        pid = sanitize_person_id(person_id)
        target = members.get(pid)
        if target is None or target.get("status") != "active":
            raise KeyError(f"no active member {pid!r}")
        if target.get("role") == roles_mod.OWNER:
            owners = [m for m in members.values()
                      if m.get("role") == roles_mod.OWNER
                      and m.get("status") == "active"]
            if len(owners) <= 1:
                raise ValueError("cannot revoke the last owner")
        target["status"] = "revoked"
        target["revoked_ts"] = time.time()
        target["revoked_by"] = actor_id
        target["revoke_reason"] = reason
        self._save_members(members)
        self._audit(actor_id, "person_revoke", pid, reason)
        return dict(target)

    def set_role(self, *, actor_id: str, actor_role: str,
                 person_id: str, role: str) -> dict[str, Any]:
        """Change a member's role. OWNER-only; cannot demote the last owner."""
        roles_mod.require(actor_role, "manage_members")
        role = roles_mod.validate_role(role)
        members = self._load_members()
        pid = sanitize_person_id(person_id)
        target = members.get(pid)
        if target is None or target.get("status") != "active":
            raise KeyError(f"no active member {pid!r}")
        if (target.get("role") == roles_mod.OWNER and role != roles_mod.OWNER):
            owners = [m for m in members.values()
                      if m.get("role") == roles_mod.OWNER
                      and m.get("status") == "active"]
            if len(owners) <= 1:
                raise ValueError("cannot demote the last owner")
        old = target["role"]
        target["role"] = role
        target["updated_ts"] = time.time()
        self._save_members(members)
        self._audit(actor_id, "role_change", pid, f"{old} -> {role}")
        return dict(target)

    def get_member(self, person_id: str) -> dict[str, Any] | None:
        return self._load_members().get(sanitize_person_id(person_id))

    def list_members(self, *, actor_role: str,
                     include_revoked: bool = False) -> list[dict[str, Any]]:
        """Roster. Revoked members are visible only with the flag."""
        members = self._load_members()
        out = []
        for m in sorted(members.values(), key=lambda x: x["person_id"]):
            if m.get("status") == "revoked" and not include_revoked:
                continue
            out.append(dict(m))
        return out

    def is_revoked(self, person_id: str) -> bool:
        m = self.get_member(person_id)
        return bool(m and m.get("status") == "revoked")

    def is_active(self, person_id: str) -> bool:
        m = self.get_member(person_id)
        return bool(m and m.get("status") == "active")

    # ── audit ────────────────────────────────────────────────────
    def audit_log(self, *, actor_role: str) -> list[dict[str, Any]]:
        """Read the audit log. OWNER-only."""
        roles_mod.require(actor_role, "view_audit")
        p = self._audit_path()
        if not p.exists():
            return []
        return [json.loads(line) for line in
                p.read_text(encoding="utf-8").splitlines() if line.strip()]
