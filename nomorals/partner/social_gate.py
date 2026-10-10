"""Code-enforced social gating: the execution layer, not prompts.

The problem: gating lived in prompt text (gate_block). A clever outsider
message — or a model hiccup — could talk its way past instructions. The
user's standing rule: gating must be CODE, never prompt-only.

The fix: enforce at the tool-call boundary. The spine's tool loop already
filters the tool list by capability AND denies at call time. This module
adds the social-specific layer:

* **Actor model** — every social tool call carries an actor: "owner" or
  "outsider". Derived from the message's is_owner flag, never from text.
* **Capability matrix** — what each actor may do, enforced in code:
    - owner: everything (send, read, admin, bulk, characters, memory).
    - outsider in DM: may converse (the brain replies), may NOT invoke
      tools beyond a tiny public set (games in groups, nothing private).
    - outsider in group: public features only (games, public info).
      No memory access, no owner tools, no DMs to others.
* **Deny, don't deflect** — a denied call returns a machine-readable
  denial. The brain turns it into a natural in-character deflection;
  the ENFORCEMENT is the denial, not the wording.

This module never sees message text. It sees actors, capabilities, and
chat kinds. Prompt injection can't reach it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.policy import Capability
from ..social.chat.base import ChatKind

__all__ = [
    "ACTOR_OWNER",
    "ACTOR_OUTSIDER",
    "CAPABILITY_MATRIX",
    "PUBLIC_CATEGORIES",
    "PRIVATE_CATEGORIES",
    "SocialGrant",
    "grant_for",
    "check_tool_call",
    "ACTOR_GROUP_ADMIN",
    "ADMIN_GROUP_TOOLS",
    "audit_matrix",
]

ACTOR_OWNER = "owner"
ACTOR_OUTSIDER = "outsider"
ACTOR_GROUP_ADMIN = "group_admin"

#: Admin tools a group admin (non-owner) may invoke in groups where they
#: hold admin status. These are the group-management spine tools — pin,
#: ban, restrict, promote, group settings, invite links.
ADMIN_GROUP_TOOLS: frozenset[str] = frozenset({
    # Telegram bot admin tools
    "tgbot_pin", "tgbot_unpin",
    "tgbot_ban", "tgbot_unban",
    "tgbot_restrict",
    "tgbot_promote", "tgbot_demote",
    "tgbot_delete",
    "tgbot_invite_link",
    # Telegram MTProto admin tools
    "telegram_pin", "telegram_unpin",
    "telegram_ban", "telegram_unban",
    "telegram_promote", "telegram_demote",
    "telegram_restrict",
    "telegram_delete",
    "telegram_invite_link",
    # WhatsApp group admin tools
    "whatsapp_group_create",
    "whatsapp_group_members",
    "whatsapp_group_rename",
    "whatsapp_group_describe",
    "whatsapp_group_settings",
    "whatsapp_group_leave",
    "whatsapp_group_invite",
    "whatsapp_community_create",
    "whatsapp_community_broadcast",
    "whatsapp_community_link",
    "whatsapp_channel_create",
    "whatsapp_channel_post",
})

#: Tool categories and their visibility. PUBLIC categories are available to
#: outsiders in groups (no gating). PRIVATE categories are owner-only,
#: enforced at the call boundary in code.
PUBLIC_CATEGORIES: frozenset[str] = frozenset({
    "games",        # game_move, game_join, game_list, game_status, …
    "dj_requests",  # music_request, dj_request — request songs from the DJ
    "public_info",  # public, non-sensitive lookups (help, public tracklists)
})

PRIVATE_CATEGORIES: frozenset[str] = frozenset({
    "admin",        # ban, pin, promote, restrict, delete, invite links, …
    "memory",       # recall, remember, forget — the owner's memories
    "owner_tools",  # research, wisdom, trading, connectors, system tools
    "messaging",    # sending DMs, reading private chats
    "characters",   # character management (creation, editing)
    "media_create", # generating media on demand (beyond public requests)
})

#: The explicit matrix: category -> {public_in_group, public_in_dm}.
#: Owner sees everything regardless. Auditable via audit_matrix().
CAPABILITY_MATRIX: dict[str, dict[str, bool]] = {
    cat: {"public_in_group": True, "public_in_dm": False}
    for cat in PUBLIC_CATEGORIES
}
CAPABILITY_MATRIX.update({
    cat: {"public_in_group": False, "public_in_dm": False}
    for cat in PRIVATE_CATEGORIES
})

#: Tools outsiders may invoke, by chat kind. Everything else is denied
#: at the call boundary — no prompt needed, no prompt can override.
PUBLIC_DM_TOOLS: frozenset[str] = frozenset({
    # An outsider DMing gets conversation only. No tools at all — the
    # brain replies in character, guarded, with no system access.
})
PUBLIC_GROUP_TOOLS: frozenset[str] = frozenset({
    # Games and public fun, per the user's spec: "in groups, games and
    # other public features are NOT gated."
    "game_move",
    "game_join",
    "game_list",
    "game_status",
    "music_request",   # request a song from the DJ
    "dj_request",
})


@dataclass(frozen=True)
class SocialGrant:
    """What an actor may do on social surfaces."""

    actor: str
    may_send: bool = False
    may_read_chats: bool = False
    may_use_tools: frozenset[str] = frozenset()
    may_admin: bool = False  # group admin actions (ban, pin, …)
    may_dm_others: bool = False  # send DMs to people who aren't the owner
    may_access_memory: bool = False
    reason: str = ""


def grant_for(*, is_owner: bool, chat_kind: str = ChatKind.DM,
              group_role: str = "member") -> SocialGrant:
    """One grant per actor per surface. Pure — no I/O, fully testable.

    ``group_role`` is "admin" when the sender holds admin status in THIS
    group (resolved via group_roles). Group admins get public tools plus
    admin tools — but only in groups, and never owner-private categories
    (memory, owner tools, DMs to others).
    """
    if is_owner:
        return SocialGrant(
            actor=ACTOR_OWNER,
            may_send=True,
            may_read_chats=True,
            may_use_tools=frozenset({"*"}),
            may_admin=True,
            may_dm_others=True,
            may_access_memory=True,
            reason="owner: full capability",
        )
    if chat_kind == ChatKind.GROUP and group_role == "admin":
        return SocialGrant(
            actor=ACTOR_GROUP_ADMIN,
            may_send=False,  # group admins don't send THROUGH her account
            may_read_chats=False,
            may_use_tools=PUBLIC_GROUP_TOOLS | ADMIN_GROUP_TOOLS,
            may_admin=True,
            may_dm_others=False,
            may_access_memory=False,
            reason="group admin: public + admin tools in this group",
        )
    if chat_kind == ChatKind.GROUP:
        return SocialGrant(
            actor=ACTOR_OUTSIDER,
            may_send=False,  # outsiders never send THROUGH her account
            may_read_chats=False,
            may_use_tools=PUBLIC_GROUP_TOOLS,
            may_admin=False,
            may_dm_others=False,
            may_access_memory=False,
            reason="outsider in group: public features only",
        )
    return SocialGrant(
        actor=ACTOR_OUTSIDER,
        may_send=False,
        may_read_chats=False,
        may_use_tools=PUBLIC_DM_TOOLS,
        may_admin=False,
        may_dm_others=False,
        may_access_memory=False,
        reason="outsider in DM: conversation only, no tools",
    )


def check_tool_call(
    tool_name: str, *, grant: SocialGrant
) -> tuple[bool, str]:
    """Enforce the grant at the call boundary. Returns (allowed, reason).

    Called by the tool loop before executing any social-adjacent tool for
    a non-owner actor. The owner path never reaches here (full grant).
    """
    if grant.actor == ACTOR_OWNER:
        return True, "owner"
    allowed = grant.may_use_tools
    if "*" in allowed or tool_name in allowed:
        return True, f"public tool: {tool_name}"
    return False, (
        f"denied: {tool_name} is not available here "
        f"({grant.reason})"
    )


def capabilities_for_grant(grant: SocialGrant) -> set[str]:
    """Map a social grant onto registry capabilities for the tool loop's
    filtered listing. An outsider's model prompt literally cannot see
    tools it may not call."""
    if grant.actor == ACTOR_OWNER:
        return {"*"}
    caps: set[str] = set()
    # Outsiders keep read-only public capabilities at most; sends are
    # never in their capability set — the registry denies at call time
    # even if a tool name leaks through.
    if grant.may_use_tools:
        caps.add(Capability.SOCIAL_READ)
    return caps


def audit_matrix() -> dict[str, Any]:
    """Dump the full public/private matrix for inspection.

    Returns every category, its visibility, and the concrete public
    tool lists. Used by tests and by the owner to verify gating.
    Pure — no I/O.
    """
    return {
        "public_categories": sorted(PUBLIC_CATEGORIES),
        "private_categories": sorted(PRIVATE_CATEGORIES),
        "matrix": {
            cat: dict(vis) for cat, vis in CAPABILITY_MATRIX.items()
        },
        "public_group_tools": sorted(PUBLIC_GROUP_TOOLS),
        "public_dm_tools": sorted(PUBLIC_DM_TOOLS),
        "owner": "everything (may_use_tools={'*'})",
        "outsider_dm": "conversation only — no tools",
        "outsider_group": "public categories only — no admin, no memory, "
                          "no owner tools, no DMs",
    }
