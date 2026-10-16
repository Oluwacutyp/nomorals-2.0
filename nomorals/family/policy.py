"""Family capability policy: a closed allowlist, fail-closed.

The family surface may only exercise these capabilities:

* ``family.profile`` — per-person profiles
* ``family.item``    — per-person items (shared / private)
* ``family.share``   — flip an item's visibility to shared
* ``family.audit``   — household audit log (owner-only)
* ``family.safety``  — kid-safe content checks

Everything else — memory, vaults, accounts, connectors, exec/shell,
network, model calls, social posting, the owner's single-owner
database — is denied by default-deny. This module imports stdlib
only: the policy is self-contained so the isolation boundary cannot
be bypassed through a shared policy import.
"""

from __future__ import annotations

from typing import Any

#: The only capabilities a family agent may ever hold.
FAMILY_CAPABILITIES: frozenset[str] = frozenset({
    "family.profile",
    "family.item",
    "family.share",
    "family.audit",
    "family.safety",
})


def is_family_capability(capability: str) -> bool:
    """True iff ``capability`` is inside the family allowlist."""
    return capability in FAMILY_CAPABILITIES


def check_capability(capability: str) -> bool:
    """Fail-closed capability check: unknown capabilities are denied."""
    return is_family_capability(capability)


def family_policy() -> dict[str, Any]:
    """The family policy as plain data: allowlist + deny-everything-else."""
    return {
        "allowlist": sorted(FAMILY_CAPABILITIES),
        "default": "deny",
    }
