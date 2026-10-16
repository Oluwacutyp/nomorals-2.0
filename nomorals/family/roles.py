"""Family roles + permissions. Roles are enforced in code, not in prompts.

Hierarchy: owner (4) > adult (3) > teen (2) > child (1). A permission
check is ``can(role, permission)`` — there is no other path to
privilege, and roles never widen: a member's effective permission set
is exactly their role's set.

Permissions:

* ``manage_members`` — add / revoke members, change roles. OWNER only.
* ``view_audit`` — read the household audit log. OWNER only.
* ``read_shared`` — read shared items. Everyone.
* ``write_shared`` — write shared items. Everyone (content still
  passes the kid-safe filter for teen/child — see safety.py).
* ``read_private_own`` — read your own private items. Everyone.
* ``read_private_other`` — read another member's private items.
  OWNER only (audit), and a revoked member's items are owner-only.
* ``share_item`` — flip your own item to shared. OWNER + ADULT.
  (Teen/child items become shared only via an owner/adult.)
* ``delete_own`` — delete your own items. Everyone.
"""

from __future__ import annotations

OWNER = "owner"
ADULT = "adult"
TEEN = "teen"
CHILD = "child"

ROLES: tuple[str, ...] = (OWNER, ADULT, TEEN, CHILD)

_RANK: dict[str, int] = {CHILD: 1, TEEN: 2, ADULT: 3, OWNER: 4}

PERMISSIONS: dict[str, frozenset[str]] = {
    OWNER: frozenset({
        "manage_members", "view_audit", "read_shared", "write_shared",
        "read_private_own", "read_private_other", "share_item", "delete_own",
    }),
    ADULT: frozenset({
        "read_shared", "write_shared", "read_private_own",
        "share_item", "delete_own",
    }),
    TEEN: frozenset({
        "read_shared", "write_shared", "read_private_own", "delete_own",
    }),
    CHILD: frozenset({
        "read_shared", "write_shared", "read_private_own", "delete_own",
    }),
}


def validate_role(role: str) -> str:
    """Normalise + validate a role string; raises ValueError if unknown."""
    role = (role or "").strip().lower()
    if role not in _RANK:
        raise ValueError(f"unknown family role {role!r}; expected one of {ROLES}")
    return role


def role_rank(role: str) -> int:
    """Higher number = more privilege. Owner outranks everyone."""
    return _RANK[validate_role(role)]


def can(role: str, permission: str) -> bool:
    """True iff ``role`` holds ``permission``. Unknown permission: False."""
    return permission in PERMISSIONS.get(validate_role(role), frozenset())


def require(role: str, permission: str) -> None:
    """Raise PermissionError unless ``role`` holds ``permission``."""
    if not can(role, permission):
        raise PermissionError(f"role {role!r} lacks permission {permission!r}")
