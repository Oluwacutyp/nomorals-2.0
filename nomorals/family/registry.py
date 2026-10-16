"""Family tool registry: the closed allowlist of tools a family-facing
agent may call. Everything else is denied — there is no fallback tool,
no shell, no generic "run anything".

ALLOWLIST (the whole thing — nothing else is registered here):

* ``family_person_add``     (family.profile) — add a household member
* ``family_person_revoke``  (family.profile) — revoke a member (owner)
* ``family_person_list``    (family.profile) — list the roster
* ``family_item_put``       (family.item)    — store a per-person item
* ``family_item_get``       (family.item)    — read one item (visibility enforced)
* ``family_item_list``      (family.item)    — list visible items
* ``family_item_share``     (family.share)   — flip own item to shared
* ``family_audit_list``     (family.audit)   — read the audit log (owner)
* ``family_kid_check``      (family.safety)  — kid-safe scan of text

Every tool takes explicit actor ids / roles: no ambient owner context
leaks in. The registry is a tiny self-contained dispatcher (stdlib
only) — it deliberately does NOT import the global tool registry, so
a family agent can never reach tools outside this allowlist through
a shared registry object.

EXPLICITLY EXCLUDED: memory tools, vault/account tools, private
connectors, exec/shell, filesystem tools, network tools, model calls,
social post/DM tools, database tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import roles as roles_mod
from .members import Household
from .policy import FAMILY_CAPABILITIES, is_family_capability
from .safety import filter_for_role
from .store import FamilyItem, default_family_root

__all__ = ["FAMILY_TOOL_NAMES", "family_registry", "FamilyRegistry"]


@dataclass
class _ToolSpec:
    name: str
    fn: Callable[..., Any]
    capability: str
    description: str = ""


class FamilyRegistry:
    """Closed-allowlist dispatcher. Unknown tool → LookupError, always."""

    def __init__(self, household: Household) -> None:
        self.household = household
        self._tools: dict[str, _ToolSpec] = {}
        self._register_all()

    # ── registration ─────────────────────────────────────────────
    def _reg(self, name: str, fn: Callable[..., Any], capability: str,
             description: str) -> None:
        if capability not in FAMILY_CAPABILITIES:
            raise ValueError(f"tool {name} capability {capability!r} outside allowlist")
        self._tools[name] = _ToolSpec(name, fn, capability, description)

    def _register_all(self) -> None:
        h = self.household
        self._reg("family_person_add", h.add_person, "family.profile",
                  "Add a household member (owner-only).")
        self._reg("family_person_revoke", h.revoke_person, "family.profile",
                  "Revoke a household member (owner-only).")
        self._reg("family_person_list", h.list_members, "family.profile",
                  "List household members.")
        self._reg("family_item_put", self._item_put, "family.item",
                  "Store a per-person item (kid-safe enforced for teen/child).")
        self._reg("family_item_get", self._item_get, "family.item",
                  "Read one item with visibility enforced.")
        self._reg("family_item_list", self._item_list, "family.item",
                  "List the items of one person visible to the requester.")
        self._reg("family_item_share", self._item_share, "family.share",
                  "Flip your own item to shared (owner/adult).")
        self._reg("family_audit_list", h.audit_log, "family.audit",
                  "Read the household audit log (owner-only).")
        self._reg("family_kid_check", self._kid_check, "family.safety",
                  "Kid-safe scan of text for a role + locale.")

    # ── tool implementations (explicit actor context, no ambient state) ──
    def _item_put(self, *, actor_id: str, actor_role: str, person_id: str,
                  item_id: str, kind: str, title: str, body: str = "",
                  visibility: str = "private",
                  locale: str = "ng") -> dict[str, Any]:
        if actor_id != person_id and roles_mod.validate_role(actor_role) != roles_mod.OWNER:
            raise PermissionError("cannot store items for another member")
        if not self.household.is_active(person_id):
            raise PermissionError(f"person {person_id!r} is not an active member")
        # Teen content is masked before storage; child content refused.
        title_checked = filter_for_role(title, actor_role, locale)
        body_checked = filter_for_role(body, actor_role, locale)
        if not title_checked["allowed"] or not body_checked["allowed"]:
            raise ValueError("content refused: not kid-safe for this role")
        item = FamilyItem(
            id=item_id, owner_id=person_id, kind=kind,
            title=str(title_checked["text"]), body=str(body_checked["text"]),
            visibility=visibility, locale=locale,
            created_ts=time.time())
        stored = self.household.store.put_item(item, requester_role=actor_role)
        return stored.to_dict()

    def _item_get(self, *, actor_id: str, actor_role: str,
                  person_id: str, item_id: str) -> dict[str, Any]:
        if not self.household.is_active(actor_id):
            raise PermissionError("revoked/inactive members read nothing")
        item = self.household.store.get_item(
            person_id, item_id, requester_id=actor_id,
            requester_role=actor_role,
            requester_revoked=self.household.is_revoked(actor_id),
            owner_revoked=self.household.is_revoked(person_id))
        # Kid-safe on READ too: a child never receives flagged content,
        # even from a shared item an adult stored.
        if roles_mod.validate_role(actor_role) == roles_mod.CHILD:
            checked = filter_for_role(f"{item.title}\n{item.body}",
                                      actor_role, item.locale)
            if not checked["allowed"]:
                raise PermissionError("content withheld: not kid-safe for a child")
        return item.to_dict()

    def _item_list(self, *, actor_id: str, actor_role: str,
                   person_id: str) -> list[dict[str, Any]]:
        if not self.household.is_active(actor_id):
            raise PermissionError("revoked/inactive members read nothing")
        items = self.household.store.list_items(
            person_id, requester_id=actor_id, requester_role=actor_role,
            requester_revoked=self.household.is_revoked(actor_id),
            owner_revoked=self.household.is_revoked(person_id))
        if roles_mod.validate_role(actor_role) == roles_mod.CHILD:
            items = [it for it in items
                     if filter_for_role(f"{it.title}\n{it.body}",
                                        actor_role, it.locale)["allowed"]]
        return [it.to_dict() for it in items]

    def _item_share(self, *, actor_id: str, actor_role: str,
                    person_id: str, item_id: str) -> dict[str, Any]:
        roles_mod.require(actor_role, "share_item")
        if actor_id != person_id and roles_mod.validate_role(actor_role) != roles_mod.OWNER:
            raise PermissionError("cannot share another member's item")
        item = self.household.store.get_item(
            person_id, item_id, requester_id=actor_id,
            requester_role=actor_role,
            owner_revoked=self.household.is_revoked(person_id))
        item.visibility = "shared"
        item.updated_ts = time.time()
        stored = self.household.store.put_item(item, requester_role=actor_role)
        self.household._audit(actor_id, "item_share", item_id,
                              f"owner={person_id}")
        return stored.to_dict()

    @staticmethod
    def _kid_check(*, text: str, role: str, locale: str = "ng") -> dict[str, Any]:
        result = filter_for_role(text, role, locale)
        return dict(result)

    # ── dispatch ─────────────────────────────────────────────────
    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        """Dispatch one tool. Unknown name → LookupError. Denied → raises."""
        spec = self._tools.get(name)
        if spec is None:
            raise LookupError(f"no such family tool {name!r}: denied")
        if not is_family_capability(spec.capability):
            raise PermissionError(f"tool {name} capability denied")
        return spec.fn(**kwargs)


def family_registry(root: Path | str | None = None) -> FamilyRegistry:
    """Build the family registry over a household at ``root``."""
    household = Household(root if root is not None else default_family_root())
    return FamilyRegistry(household)


#: The allowlist, as a constant the isolation test can compare against.
FAMILY_TOOL_NAMES: tuple[str, ...] = (
    "family_audit_list",
    "family_item_get",
    "family_item_list",
    "family_item_put",
    "family_item_share",
    "family_kid_check",
    "family_person_add",
    "family_person_list",
    "family_person_revoke",
)
