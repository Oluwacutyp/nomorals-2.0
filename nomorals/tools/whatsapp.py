"""WhatsApp group/community/channel tools for the spine.

The brain's hands on WhatsApp groups: create groups, manage members,
rename, settings, communities, announcements, channels. All mutations go
through the owner's personal WhatsApp account via the Baileys bridge —
WhatsApp may flag aggressive automation, so every tool here is
owner-confirmed: capability SOCIAL_BULK (owner-only in the public/private
matrix; outsiders are denied at call time).

Gating is CODE, not prompts. The gateway reference is set once at runtime
boot; tools fail closed when it's absent.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

_gateway: Any = None


def set_gateway(gateway: Any) -> None:
    """Bind the chat gateway. Called once at runtime boot."""
    global _gateway
    _gateway = gateway


def _require_gateway() -> Any:
    gw = _gateway
    if gw is None:
        raise RuntimeError("social not connected — no chat gateway is bound")
    return gw


def _wa_admin(method: str, **kwargs: Any) -> dict[str, Any]:
    gw = _require_gateway()
    adapter = (gw._snapshot_adapters() or {}).get("whatsapp")
    if adapter is None:
        return {"ok": False, "error": "whatsapp adapter not connected"}
    fn = getattr(adapter, method, None)
    if fn is None:
        return {"ok": False, "error": f"no {method} on whatsapp adapter"}
    try:
        return fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - honest error, never a crash
        return {"ok": False, "error": str(exc)[:200]}


def _str_list(name: str, props: dict[str, Any]) -> dict[str, Any]:
    props[name] = {"type": "array", "items": {"type": "string"}}
    return props


def register(registry: Any) -> None:
    """Register all WhatsApp group/community/channel tools."""
    from ..social.render import (
        render_channel, render_community, render_group_card,
        render_group_list, render_members,
    )

    def _wa() -> Any:
        gw = _require_gateway()
        return (gw._snapshot_adapters() or {}).get("whatsapp")

    # ── reads (rendered, god-tier output) ─────────────────────────────

    @registry.register(
        "whatsapp_groups",
        description="List WhatsApp groups. Returns a rendered group list.",
        capability=Capability.SOCIAL_READ,
        parameters={"type": "object", "properties": {}},
    )
    def whatsapp_groups() -> dict[str, Any]:
        adapter = _wa()
        if adapter is None:
            return {"ok": False, "error": "whatsapp not connected"}
        groups = adapter.groups()
        return {"ok": True, "rendered": render_group_list(groups, "whatsapp"),
                "count": len(groups)}

    @registry.register(
        "whatsapp_group_info",
        description="Show a rich info card for a WhatsApp group.",
        capability=Capability.SOCIAL_READ,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "group JID or name"},
            },
            "required": ["chat"],
        },
    )
    def whatsapp_group_info(chat: str) -> dict[str, Any]:
        adapter = _wa()
        if adapter is None:
            return {"ok": False, "error": "whatsapp not connected"}
        info = adapter.group_info(chat)
        if not info:
            return {"ok": False, "error": f"no group info for {chat!r}"}
        members = adapter.group_participants(chat)
        return {"ok": True,
                "rendered": render_group_card(info, members, "whatsapp"),
                "info": info}

    @registry.register(
        "whatsapp_group_roster",
        description="Show the member roster for a WhatsApp group, with roles.",
        capability=Capability.SOCIAL_READ,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "group JID or name"},
            },
            "required": ["chat"],
        },
    )
    def whatsapp_group_roster(chat: str) -> dict[str, Any]:
        adapter = _wa()
        if adapter is None:
            return {"ok": False, "error": "whatsapp not connected"}
        members = adapter.group_participants(chat)
        return {"ok": True,
                "rendered": render_members(members, "whatsapp"),
                "count": len(members)}

    # ── groups ────────────────────────────────────────────────────────

    @registry.register(
        "whatsapp_group_create",
        description=(
            "Create a WhatsApp group. ('make a group called Lagos foodies', "
            "'create a group for the project team'). Optional initial members "
            "as user JIDs. Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "group name"},
                "participants": {"type": "array", "items": {"type": "string"},
                                 "description": "optional user JIDs"},
            },
            "required": ["subject"],
        },
    )
    def whatsapp_group_create(subject: str,
                              participants: list[str] | None = None) -> dict[str, Any]:
        return _wa_admin("group_create", subject=subject,
                         participants=participants)

    @registry.register(
        "whatsapp_group_members",
        description=(
            "Add/remove/promote/demote members in a WhatsApp group. "
            "('add Ada to Lagos foodies', 'make him admin in the group', "
            "'remove that number from the group'). Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string",
                         "description": "group JID (…@g.us) or name"},
                "action": {"type": "string",
                           "description": "add|remove|promote|demote"},
                "participants": {"type": "array", "items": {"type": "string"},
                                 "description": "user JIDs"},
            },
            "required": ["chat", "action", "participants"],
        },
    )
    def whatsapp_group_members(chat: str, action: str,
                               participants: list[str]) -> dict[str, Any]:
        return _wa_admin("group_members_update", chat=chat, action=action,
                         participants=participants)

    @registry.register(
        "whatsapp_group_rename",
        description="Rename a WhatsApp group. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "group JID or name"},
                "subject": {"type": "string", "description": "new name"},
            },
            "required": ["chat", "subject"],
        },
    )
    def whatsapp_group_rename(chat: str, subject: str) -> dict[str, Any]:
        return _wa_admin("group_set_subject", chat=chat, subject=subject)

    @registry.register(
        "whatsapp_group_describe",
        description="Set a WhatsApp group's description. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "group JID or name"},
                "description": {"type": "string"},
            },
            "required": ["chat", "description"],
        },
    )
    def whatsapp_group_describe(chat: str, description: str) -> dict[str, Any]:
        return _wa_admin("group_set_description", chat=chat,
                         description=description)

    @registry.register(
        "whatsapp_group_settings",
        description=(
            "Group settings: announce (only admins can send) and restrict "
            "(only admins can change settings). Pass only what should change. "
            "Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "group JID or name"},
                "announce": {"type": "boolean"},
                "restrict": {"type": "boolean"},
            },
            "required": ["chat"],
        },
    )
    def whatsapp_group_settings(chat: str, announce: bool | None = None,
                                restrict: bool | None = None) -> dict[str, Any]:
        return _wa_admin("group_set_settings", chat=chat, announce=announce,
                         restrict=restrict)

    @registry.register(
        "whatsapp_group_leave",
        description="Leave a WhatsApp group. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "group JID or name"},
            },
            "required": ["chat"],
        },
    )
    def whatsapp_group_leave(chat: str) -> dict[str, Any]:
        return _wa_admin("group_leave", chat=chat)

    @registry.register(
        "whatsapp_group_invite",
        description=(
            "Revoke a WhatsApp group's invite link and get a fresh one. "
            "Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "group JID or name"},
            },
            "required": ["chat"],
        },
    )
    def whatsapp_group_invite(chat: str) -> dict[str, Any]:
        return _wa_admin("group_revoke_invite", chat=chat)

    # ── communities ───────────────────────────────────────────────────

    @registry.register(
        "whatsapp_community_create",
        description="Create a WhatsApp community. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "community name"},
                "description": {"type": "string"},
            },
            "required": ["subject"],
        },
    )
    def whatsapp_community_create(subject: str,
                                  description: str = "") -> dict[str, Any]:
        return _wa_admin("community_create", subject=subject,
                         description=description)

    @registry.register(
        "whatsapp_community_broadcast",
        description=(
            "Post an announcement to a WhatsApp community — goes to the "
            "announcement group so every member sees it. Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "community": {"type": "string",
                              "description": "community JID or name"},
                "text": {"type": "string",
                         "description": "the announcement"},
            },
            "required": ["community", "text"],
        },
    )
    def whatsapp_community_broadcast(community: str,
                                     text: str) -> dict[str, Any]:
        return _wa_admin("community_broadcast", community=community, text=text)

    @registry.register(
        "whatsapp_community_link",
        description=(
            "Link a subgroup into a WhatsApp community (or unlink it). "
            "Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "community": {"type": "string"},
                "group": {"type": "string"},
                "link": {"type": "boolean",
                         "description": "true=link, false=unlink"},
            },
            "required": ["community", "group", "link"],
        },
    )
    def whatsapp_community_link(community: str, group: str,
                                link: bool = True) -> dict[str, Any]:
        method = "community_link_group" if link else "community_unlink_group"
        return _wa_admin(method, community=community, group=group)

    # ── channels ────────────────────────────────────────────────────

    @registry.register(
        "whatsapp_channel_create",
        description="Create a WhatsApp channel (newsletter). Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["name"],
        },
    )
    def whatsapp_channel_create(name: str,
                                description: str = "") -> dict[str, Any]:
        return _wa_admin("channel_create", name=name, description=description)

    @registry.register(
        "whatsapp_channel_post",
        description="Post to a WhatsApp channel. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "channel": {"type": "string",
                            "description": "channel JID (…@newsletter)"},
                "text": {"type": "string"},
            },
            "required": ["channel", "text"],
        },
    )
    def whatsapp_channel_post(channel: str, text: str) -> dict[str, Any]:
        return _wa_admin("channel_post", chat=channel, text=text)
