"""Multi-person / Family mode — MORE_TO_ADD gap 4.

Second person, one level down from community: the household is a
*private* multi-person space, not a public group. The architectural
axis this adds over community/ is per-person state + roles:

ISOLATION CONTRACT (enforced by ``tests/test_more_gaps456.py``):

* Nothing in this package may import ``nomorals.memory``,
  ``nomorals.accounts``, any vault module, or any private connector
  (``nomorals.connectors.*``). Family state is its own world.
* Family state lives under ``~/.devon/family/`` — one directory per
  person (``~/.devon/family/<person_id>/``) plus a household-level
  ``members.json`` / ``audit.jsonl``. It NEVER lives in the owner's
  single-owner memory tables, and a lookup for person B physically
  cannot read person A's directory.
* Per-person state never crosses: a private item is readable only by
  its owner and by the household owner (audit). Shared items are
  visible to every household member. Nothing else crosses.
* Closed tool allowlist (``registry.py`` + ``policy.py``): everything
  not listed is denied. When in doubt, exclude.
* Roles (``roles.py``): owner > adult > teen > child. Only the owner
  can add / revoke members, change roles, or read the audit log.
  Revoking a member cuts their access immediately; the last owner
  cannot be revoked.
* Kid-safe (``safety.py``): content for child/teen members passes a
  locale-aware filter; unsafe content is refused (child) or masked
  (teen), and every item carries a ``kid_safe`` flag.

UNIVERSAL — N PACKS, NIGERIA DEFAULT, NEVER A BOUNDARY: locale packs
(``locales.py``) are data, not gates. The default pack is ``ng``
(Nigeria: Lagos, Africa/Lagos). Any pack — or a caller-supplied one —
works through exactly the same code paths; no pack is ever rejected
for not being Nigeria, and no feature is Nigeria-only.
"""

from __future__ import annotations

from . import locales, members, policy, registry, roles, safety, store
from .locales import DEFAULT_LOCALE, LOCALE_PACKS, get_locale_pack, list_locale_packs
from .members import Household
from .policy import FAMILY_CAPABILITIES, check_capability, family_policy
from .registry import FAMILY_TOOL_NAMES, family_registry
from .roles import CHILD, ADULT, OWNER, TEEN, can, role_rank
from .safety import filter_for_role, is_kid_safe, scan_text
from .store import FamilyItem, PerPersonStore, default_family_root

__all__ = [
    "locales", "members", "policy", "registry", "roles", "safety", "store",
    "DEFAULT_LOCALE", "LOCALE_PACKS", "get_locale_pack", "list_locale_packs",
    "Household", "FAMILY_CAPABILITIES", "check_capability", "family_policy",
    "FAMILY_TOOL_NAMES", "family_registry",
    "CHILD", "ADULT", "OWNER", "TEEN", "can", "role_rank",
    "filter_for_role", "is_kid_safe", "scan_text",
    "FamilyItem", "PerPersonStore", "default_family_root",
]
