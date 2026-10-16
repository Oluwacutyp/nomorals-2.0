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

__all__ = ["events", "groups", "meetups", "miniapps", "policy", "registry",
           "audio_rooms"]

# Second-wave community modules (ajo, miniapps2, audio, bridges,
# group_memory, persona, insights) — same isolation contract as above.
#
# CONTRACT CLARIFICATION: the isolation contract contains OWNER DATA,
# it does not withhold capability inside the boundary. A group agent
# built on group_memory / persona may be richly capable — remembering
# group history, reasoning over it, running ajo circles, bridges, and
# mini-apps — as long as it cannot reach owner data. Containment of
# data, not poverty of features.
__all__ += ["ajo", "miniapps2", "audio", "bridges", "group_memory",
            "persona", "insights"]

# GROUPS Tier 1 — welcome/moderation + digest/rituals (same isolation contract).
from .digest import GroupDigest, Ritual, RitualScheduler, build_digest
from .welcome_moderation import BanList, CaptchaGate, Moderator, Roster, WelcomeConfig

__all__ += ["welcome_moderation", "digest", "BanList", "CaptchaGate",
            "Moderator", "Roster", "WelcomeConfig", "GroupDigest",
            "Ritual", "RitualScheduler", "build_digest"]

# GROUPS Tier 3 (orchestration, mirroring, group_extras) — same isolation
# contract: injected senders / callables only, no owner data.
__all__ += ["orchestration", "mirroring", "group_extras"]

# Groups Tier 2 (parties, group economy, media tools) — same isolation
# contract as above: stdlib + nomorals.core + sibling community modules.
__all__ += ["parties", "group_economy", "media_tools"]
