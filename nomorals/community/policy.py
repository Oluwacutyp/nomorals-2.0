"""Community capability policy: a closed allowlist, fail-closed.

The community surface may only exercise these capabilities:

* ``community.miniapp`` — create / mutate / read group mini-apps
* ``community.fun``      — dice rolls, coin flips, other pure utilities

Everything else — filesystem, shell, network, memory, database, model
calls, social posting, owner vaults — is denied by default-deny. The
mini-app store writes JSON under the community data dir through plain
Python file I/O inside this package, not through a granted capability,
so even ``fs.write`` stays out of the allowlist.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import CapabilitySet, Policy

#: The only capabilities a community agent may ever hold.
COMMUNITY_CAPABILITIES: frozenset[str] = frozenset(
    {"community.miniapp", "community.fun"}
)


def community_policy(**kwargs: Any) -> Policy:
    """A Policy granting exactly the community allowlist. Fail-closed."""
    return Policy(default_grant=CapabilitySet(COMMUNITY_CAPABILITIES), **kwargs)


def is_community_capability(capability: str) -> bool:
    """True iff ``capability`` is inside the community allowlist."""
    grant = CapabilitySet(COMMUNITY_CAPABILITIES)
    return grant.grants(capability)
