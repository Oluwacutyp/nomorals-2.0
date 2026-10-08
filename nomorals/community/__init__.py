"""Community surfaces: group mini-apps and other community-facing features.

ISOLATION CONTRACT (architectural, enforced by
``tests/test_community_isolation.py``):

* Nothing in this package may import ``nomorals.memory``,
  ``nomorals.accounts``, any vault module, or any private connector
  (``nomorals.connectors.*``).
* Community state lives under ``~/.devon/community/`` and NEVER in the
  owner's database tables.
* No owner data flows into community state: no names from people pages,
  no memory lookups, no account or credential access. Participants are
  identified only by the platform user id / display name that arrives
  with the chat message itself.
* The community tool registry (``registry.py``) and policy
  (``policy.py``) are a closed allowlist. When in doubt, exclude.

Rationale: community chats are semi-public. A bug here must be
incapable of leaking the owner's private world, by construction rather
than by care.
"""

__all__ = ["miniapps", "policy", "registry"]
