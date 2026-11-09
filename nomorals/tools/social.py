"""Social tools for the spine — the brain's hands on every chat platform.

The problem: 282 tools existed but NONE could send a message, read chat
history, or list chats. The spine (tool_loop) could think about social
but never act on it. Social was command-driven, not capability-driven.

The fix: real tools, registered with capabilities, code-enforced gating.
The brain picks them from plain language — "text mom back", "check what
I missed in the group", "send the mix to the channel".

Gating is CODE, not prompts:
- Every send tool checks the actor at execution time via the registry's
  capability enforcement (SOCIAL_POST / SOCIAL_DM / SOCIAL_BULK).
- Owner actors get full send. Outsider actors see a filtered tool list
  AND get denied at call time if they reach for more.
- The gateway reference is set once at runtime boot; tools fail closed
  (honest "social not connected") when it's absent.

Never break character: these tools send exactly what the brain composes.
No AI-behavior leaks — the persona lives in the brain, the tools are dumb
pipes.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

#: Set once at runtime boot by the partner runtime. Tools fail closed
#: without it — never silently pretend to send.
_gateway: Any = None


def set_gateway(gateway: Any) -> None:
    """Bind the chat gateway. Called once at runtime boot."""
    global _gateway
    _gateway = gateway


def get_gateway() -> Any:
    return _gateway


def _require_gateway() -> Any:
    gw = get_gateway()
    if gw is None:
        raise RuntimeError("social not connected — no chat gateway is bound")
    return gw


def register(registry: Any) -> None:
    """Register all social tools on the given ToolRegistry."""
    from ..social.chat.base import ChatRef

    @registry.register(
        "social_send",
        description=(
            "Send a text message to a chat. chat_key looks like "
            "'telegram:123456' or 'whatsapp:2348012345678@s.whatsapp.net'. "
            "Use for proactive outreach, replies the brain composes itself, "
            "delivering results to chats. The message sends exactly as given."
        ),
        capability=Capability.SOCIAL_POST,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {
                    "type": "string",
                    "description": "Platform chat key, e.g. 'telegram:123456'",
                },
                "text": {
                    "type": "string",
                    "description": "Message text to send",
                },
                "reply_to": {
                    "type": "string",
                    "description": "Message ID to reply to (optional)",
                },
            },
            "required": ["chat_key", "text"],
        },
    )
    def social_send(chat_key: str, text: str, reply_to: str = "") -> dict[str, Any]:
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        chat = ChatRef(platform=platform, chat_id=rest)
        result = gw.send(platform, chat, text, reply_to=reply_to or None)
        return {
            "ok": result.ok,
            "message_id": result.message_id,
            "error": result.error,
            "seconds": round(result.seconds, 2),
        }

    @registry.register(
        "social_history",
        description=(
            "Read recent message history from a chat. Returns the last N "
            "messages (text, sender, timestamp) so the brain has context "
            "beyond its own memory window. Use before replying in chats "
            "you haven't seen recently."
        ),
        capability=Capability.SOCIAL_READ,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {
                    "type": "string",
                    "description": "Platform chat key, e.g. 'telegram:123456'",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many recent messages (1-50, default 20)",
                },
            },
            "required": ["chat_key"],
        },
    )
    def social_history(chat_key: str, limit: int = 20) -> dict[str, Any]:
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        chat = ChatRef(platform=platform, chat_id=rest)
        limit = max(1, min(50, int(limit or 20)))
        messages = gw.history(platform, chat, limit=limit)
        return {
            "ok": True,
            "count": len(messages),
            "messages": [
                {
                    "text": m.text,
                    "sender": m.sender,
                    "sender_id": m.sender_id,
                    "ts": m.ts,
                    "incoming": m.incoming,
                    "message_id": m.message_id,
                }
                for m in messages
            ],
        }

    @registry.register(
        "social_chats",
        description=(
            "List known chats across platforms — DMs, groups, channels the "
            "system has seen. Returns chat keys, titles, kinds, and last "
            "activity. Use to find where someone is, or to pick a target "
            "for proactive outreach."
        ),
        capability=Capability.SOCIAL_READ,
        parameters={
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": "Filter by platform (optional, e.g. 'telegram')",
                },
                "kind": {
                    "type": "string",
                    "description": "Filter by kind: dm, group, channel (optional)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max chats to return (default 30)",
                },
            },
        },
    )
    def social_chats(
        platform: str = "", kind: str = "", limit: int = 30
    ) -> dict[str, Any]:
        gw = _require_gateway()
        limit = max(1, min(100, int(limit or 30)))
        # The gateway's chat registry lives in its DB; ask each adapter.
        chats: list[dict[str, Any]] = []
        for name, adapter in (gw._snapshot_adapters() or {}).items():
            if platform and name != platform:
                continue
            try:
                if hasattr(adapter, "chats"):
                    for c in adapter.chats(limit=limit) or []:
                        if kind and c.get("kind") != kind:
                            continue
                        c = dict(c)
                        c.setdefault("platform", name)
                        chats.append(c)
            except Exception as exc:  # noqa: BLE001 — one bad adapter
                _log.debug("social_chats: %s failed: %s", name, exc)
        return {"ok": True, "count": len(chats), "chats": chats[:limit]}

    @registry.register(
        "social_typing",
        description=(
            "Show the typing indicator in a chat for a few seconds. Use "
            "before sending a longer composed message so it feels human — "
            "a real person types before the message lands."
        ),
        capability=Capability.SOCIAL_POST,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {
                    "type": "string",
                    "description": "Platform chat key",
                },
                "seconds": {
                    "type": "number",
                    "description": "How long to show typing (1-30, default 3)",
                },
            },
            "required": ["chat_key"],
        },
    )
    def social_typing(chat_key: str, seconds: float = 3.0) -> dict[str, Any]:
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        chat = ChatRef(platform=platform, chat_id=rest)
        seconds = max(1.0, min(30.0, float(seconds or 3.0)))
        ok = gw.typing(platform, chat, seconds=seconds)
        return {"ok": bool(ok)}

    @registry.register(
        "social_mark_read",
        description=(
            "Mark a chat as read (sends read receipts where the platform "
            "supports it). Use after catching up on a chat so the other "
            "side sees their messages were seen."
        ),
        capability=Capability.SOCIAL_READ,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string", "description": "Platform chat key"},
            },
            "required": ["chat_key"],
        },
    )
    def social_mark_read(chat_key: str) -> dict[str, Any]:
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        chat = ChatRef(platform=platform, chat_id=rest)
        adapter = (gw._snapshot_adapters() or {}).get(platform)
        if adapter is None or not hasattr(adapter, "mark_read"):
            return {"ok": False, "error": f"no mark_read on {platform}"}
        return {"ok": bool(adapter.mark_read(chat))}

    @registry.register(
        "social_send_file",
        description=(
            "Send a file (audio, image, video, document) to a chat. "
            "path must be a local file. Use for delivering mixes, voice "
            "notes, images, documents the brain produced."
        ),
        capability=Capability.SOCIAL_POST,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string", "description": "Platform chat key"},
                "path": {"type": "string", "description": "Local file path"},
                "caption": {"type": "string", "description": "Caption (optional)"},
            },
            "required": ["chat_key", "path"],
        },
    )
    def social_send_file(
        chat_key: str, path: str, caption: str = ""
    ) -> dict[str, Any]:
        import os

        if not os.path.exists(path):
            return {"ok": False, "error": f"file not found: {path}"}
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        chat = ChatRef(platform=platform, chat_id=rest)
        result = gw.send_file(platform, chat, path, caption=caption or "")
        return {
            "ok": result.ok,
            "message_id": result.message_id,
            "error": result.error,
        }

    # ── Telegram group admin power tools ─────────────────────────
    # Owner-only, enforced by capability (SOCIAL_BULK is confirmable).
    # The brain checks admin_my_rights first — honest capability, never
    # assume.

    def _tg_admin(chat_key: str, method: str, **kwargs: Any) -> dict[str, Any]:
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        if platform != "telegram":
            return {"ok": False, "error": "admin tools are telegram-only"}
        adapter = (gw._snapshot_adapters() or {}).get("telegram")
        if adapter is None:
            return {"ok": False, "error": "telegram adapter not connected"}
        fn = getattr(adapter, method, None)
        if fn is None:
            return {"ok": False, "error": f"no {method} on telegram adapter"}
        chat = ChatRef(platform=platform, chat_id=rest)
        return fn(chat, **kwargs)

    @registry.register(
        "telegram_admin_rights",
        description=(
            "Check OUR OWN admin rights in a Telegram group/channel. Call "
            "this BEFORE any admin action — honest capability check, never "
            "assume we can act."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string", "description": "telegram:<chat_id>"},
            },
            "required": ["chat_key"],
        },
    )
    def telegram_admin_rights(chat_key: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_my_rights")

    @registry.register(
        "telegram_pin",
        description="Pin a message in a Telegram group/channel. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "message_id": {"type": "string", "description": "Message to pin"},
                "silent": {"type": "boolean", "description": "No notification (default true)"},
            },
            "required": ["chat_key", "message_id"],
        },
    )
    def telegram_pin(chat_key: str, message_id: str, silent: bool = True) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_pin", message_id=message_id, silent=silent)

    @registry.register(
        "telegram_unpin",
        description="Unpin the pinned message. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {"chat_key": {"type": "string"}},
            "required": ["chat_key"],
        },
    )
    def telegram_unpin(chat_key: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_unpin")

    @registry.register(
        "telegram_ban",
        description="Ban a user from a Telegram group/channel. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string", "description": "Numeric user ID"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def telegram_ban(chat_key: str, user_id: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_ban", user_id=user_id)

    @registry.register(
        "telegram_unban",
        description="Unban a user. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def telegram_unban(chat_key: str, user_id: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_unban", user_id=user_id)

    @registry.register(
        "telegram_promote",
        description="Promote a user to admin. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def telegram_promote(chat_key: str, user_id: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_promote", user_id=user_id)

    @registry.register(
        "telegram_invite_link",
        description="Get the group's invite link. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {"chat_key": {"type": "string"}},
            "required": ["chat_key"],
        },
    )
    def telegram_invite_link(chat_key: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_invite_link")

    @registry.register(
        "telegram_demote",
        description="Demote an admin back to member. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string", "description": "Numeric user ID"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def telegram_demote(chat_key: str, user_id: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_demote", user_id=user_id)

    @registry.register(
        "telegram_restrict",
        description=(
            "Mute a user in a Telegram group (block sending messages) for "
            "N minutes. 0 = indefinite. Owner only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string", "description": "Numeric user ID"},
                "minutes": {"type": "integer", "description": "Mute duration (default 60, 0=indefinite)"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def telegram_restrict(chat_key: str, user_id: str,
                          minutes: int = 60) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_restrict", user_id=user_id,
                         minutes=minutes)

    @registry.register(
        "telegram_delete",
        description="Delete a message in a Telegram group/channel (admin). Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "message_id": {"type": "string", "description": "Message to delete"},
            },
            "required": ["chat_key", "message_id"],
        },
    )
    def telegram_delete(chat_key: str, message_id: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_delete", message_id=message_id)

    @registry.register(
        "telegram_members",
        description="List members of a Telegram group/channel. Owner only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "limit": {"type": "integer", "description": "Max members (default 100)"},
            },
            "required": ["chat_key"],
        },
    )
    def telegram_members(chat_key: str, limit: int = 100) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_members", limit=limit)

    @registry.register(
        "telegram_forum_topic",
        description=(
            "Create a forum topic in a Telegram supergroup (topics must be "
            "enabled). MTProto path. Owner only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "title": {"type": "string", "description": "Topic title"},
            },
            "required": ["chat_key", "title"],
        },
    )
    def telegram_forum_topic(chat_key: str, title: str) -> dict[str, Any]:
        return _tg_admin(chat_key, "admin_forum_topic", title=title)

    # ── telegram-bot (Bot API) group/channel admin ────────────────────
    # Same operations as the MTProto tools above, but through Devon's own
    # bot identity (@HerBot) instead of the owner's account. The bot can
    # only act where it is an admin — admin_rights() is the honest check.

    def _tgbot_admin(chat_key: str, method: str, **kwargs: Any) -> dict[str, Any]:
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        if platform != "telegram-bot":
            return {"ok": False,
                    "error": "bot admin tools need a telegram-bot: chat key"}
        adapter = (gw._snapshot_adapters() or {}).get("telegram-bot")
        if adapter is None:
            return {"ok": False, "error": "telegram bot not connected"}
        fn = getattr(adapter, method, None)
        if fn is None:
            return {"ok": False, "error": f"no {method} on telegram bot"}
        chat = ChatRef(platform=platform, chat_id=rest)
        try:
            return fn(chat, **kwargs)
        except Exception as exc:  # noqa: BLE001 - honest error
            return {"ok": False, "error": str(exc)[:200]}

    @registry.register(
        "tgbot_admin_rights",
        description=(
            "Check the BOT's OWN admin rights in a group/channel. Call this "
            "BEFORE any bot admin action — honest capability check."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string",
                             "description": "telegram-bot:<chat_id>"},
            },
            "required": ["chat_key"],
        },
    )
    def tgbot_admin_rights(chat_key: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_rights")

    @registry.register(
        "tgbot_pin",
        description="Pin a message in a group/channel via the bot. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "message_id": {"type": "string"},
                "silent": {"type": "boolean", "default": True},
            },
            "required": ["chat_key", "message_id"],
        },
    )
    def tgbot_pin(chat_key: str, message_id: str,
                  silent: bool = True) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_pin",
                                        message_id=message_id, silent=silent)

    @registry.register(
        "tgbot_unpin",
        description="Unpin the pinned message via the bot. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {"chat_key": {"type": "string"}},
            "required": ["chat_key"],
        },
    )
    def tgbot_unpin(chat_key: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_unpin")

    @registry.register(
        "tgbot_ban",
        description="Ban a user from a group/channel via the bot. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def tgbot_ban(chat_key: str, user_id: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_ban", user_id=user_id)

    @registry.register(
        "tgbot_unban",
        description="Unban a user via the bot. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def tgbot_unban(chat_key: str, user_id: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_unban",
                                        user_id=user_id)

    @registry.register(
        "tgbot_restrict",
        description=(
            "Restrict a member via the bot (mute). Pass permission keys to "
            "ALLOW (can_send_messages etc.); everything else is denied. "
            "Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string"},
                "until": {"type": "integer",
                          "description": "unix timestamp, 0 = forever"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def tgbot_restrict(chat_key: str, user_id: str,
                       until: int = 0) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_restrict",
                                        user_id=user_id, until=until)

    @registry.register(
        "tgbot_promote",
        description=(
            "Promote a user to admin via the bot, with standard rights and "
            "an optional custom title. Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def tgbot_promote(chat_key: str, user_id: str,
                      title: str = "") -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_promote",
                                        user_id=user_id, title=title)

    @registry.register(
        "tgbot_demote",
        description="Strip a user's admin rights via the bot. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "user_id": {"type": "string"},
            },
            "required": ["chat_key", "user_id"],
        },
    )
    def tgbot_demote(chat_key: str, user_id: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_demote",
                                        user_id=user_id)

    @registry.register(
        "tgbot_delete",
        description="Delete a message via the bot. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "message_id": {"type": "string"},
            },
            "required": ["chat_key", "message_id"],
        },
    )
    def tgbot_delete(chat_key: str, message_id: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_delete",
                                        message_id=message_id)

    @registry.register(
        "tgbot_admins",
        description="List a group/channel's admins via the bot.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {"chat_key": {"type": "string"}},
            "required": ["chat_key"],
        },
    )
    def tgbot_admins(chat_key: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_administrators")

    @registry.register(
        "tgbot_invite_link",
        description="Get a group's invite link via the bot. Owner-only.",
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {"chat_key": {"type": "string"}},
            "required": ["chat_key"],
        },
    )
    def tgbot_invite_link(chat_key: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_invite_link")

    @registry.register(
        "tgbot_set_chat_photo",
        description=(
            "Set a Telegram group/channel's photo via the bot, from a "
            "local image file (setChatPhoto). The bot needs the "
            "can_change_info admin right. Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "photo_path": {"type": "string",
                               "description": "Local image file to upload"},
            },
            "required": ["chat_key", "photo_path"],
        },
    )
    def tgbot_set_chat_photo(chat_key: str,
                             photo_path: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_set_photo",
                            photo_path=photo_path)

    @registry.register(
        "tgbot_create_invite_link",
        description=(
            "Mint an ADDITIONAL invite link for a group/channel via "
            "the bot (createChatInviteLink) — beyond the primary link. "
            "Optional: name (≤32 chars), expire_date (unix timestamp, "
            "0 = never), member_limit (0 = uncapped), "
            "creates_join_request (admins approve each join; cannot "
            "combine with member_limit). Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "name": {"type": "string"},
                "expire_date": {"type": "integer"},
                "member_limit": {"type": "integer"},
                "creates_join_request": {"type": "boolean"},
            },
            "required": ["chat_key"],
        },
    )
    def tgbot_create_invite_link(chat_key: str, name: str = "",
                                 expire_date: int = 0,
                                 member_limit: int = 0,
                                 creates_join_request: bool = False
                                 ) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_create_invite_link",
                            name=name, expire_date=expire_date,
                            member_limit=member_limit,
                            creates_join_request=creates_join_request)

    @registry.register(
        "tgbot_edit_invite_link",
        description=(
            "Retune an additional invite link the bot created "
            "(editChatInviteLink): name, expiry, member limit or "
            "join-request mode — only the fields passed change. "
            "Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "invite_link": {"type": "string",
                                "description": "The link to edit"},
                "name": {"type": "string"},
                "expire_date": {"type": "integer"},
                "member_limit": {"type": "integer"},
                "creates_join_request": {"type": "boolean"},
            },
            "required": ["chat_key", "invite_link"],
        },
    )
    def tgbot_edit_invite_link(chat_key: str, invite_link: str,
                               name: str = "", expire_date: int = 0,
                               member_limit: int = 0,
                               creates_join_request: bool = False
                               ) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_edit_invite_link",
                            invite_link=invite_link, name=name,
                            expire_date=expire_date,
                            member_limit=member_limit,
                            creates_join_request=creates_join_request)

    @registry.register(
        "tgbot_revoke_invite_link",
        description=(
            "Revoke an invite link via the bot (revokeChatInviteLink) "
            "— the link dies for good. Revoking the PRIMARY link makes "
            "Telegram mint a replacement, which is what comes back. "
            "Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "invite_link": {"type": "string",
                                "description": "The link to revoke"},
            },
            "required": ["chat_key", "invite_link"],
        },
    )
    def tgbot_revoke_invite_link(chat_key: str,
                                 invite_link: str) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_revoke_invite_link",
                            invite_link=invite_link)

    @registry.register(
        "tgbot_copy_message",
        description=(
            "Copy a message into a group/channel via the bot "
            "(copyMessage): the duplicate stands as the bot's own "
            "post — no 'forwarded from' header, no original author. "
            "An optional caption REPLACES the original caption. Use "
            "the forward tool instead when attribution should ride. "
            "Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string",
                             "description": "Destination telegram-bot:<chat_id>"},
                "from_chat_key": {"type": "string",
                                  "description": "Source telegram-bot:<chat_id>"},
                "message_id": {"type": "string"},
                "caption": {"type": "string"},
            },
            "required": ["chat_key", "from_chat_key", "message_id"],
        },
    )
    def tgbot_copy_message(chat_key: str, from_chat_key: str,
                           message_id: str,
                           caption: str = "") -> dict[str, Any]:
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        fplatform, _, frest = from_chat_key.partition(":")
        if platform != "telegram-bot" or fplatform != "telegram-bot":
            return {"ok": False,
                    "error": "copy needs telegram-bot: chat keys"}
        adapter = (gw._snapshot_adapters() or {}).get("telegram-bot")
        if adapter is None:
            return {"ok": False, "error": "telegram bot not connected"}
        fn = getattr(adapter, "copy_message", None)
        if fn is None:
            return {"ok": False,
                    "error": "no copy_message on telegram bot"}
        chat = ChatRef(platform=platform, chat_id=rest)
        from_chat = ChatRef(platform=fplatform, chat_id=frest)
        try:
            res = fn(chat, from_chat, message_id, caption=caption)
        except Exception as exc:  # noqa: BLE001 - honest error
            return {"ok": False, "error": str(exc)[:200]}
        return {"ok": res.ok, "message_id": res.message_id,
                "error": res.error}

    @registry.register(
        "tgbot_forum_topic_create",
        description=(
            "Create a forum topic in a Telegram forum supergroup via "
            "the bot (createForumTopic). Topics must be enabled and "
            "the bot needs can_manage_topics. Returns the new topic's "
            "thread_id. Owner-only."
        ),
        capability=Capability.SOCIAL_BULK,
        parameters={
            "type": "object",
            "properties": {
                "chat_key": {"type": "string"},
                "name": {"type": "string",
                         "description": "Topic name"},
                "icon_color": {"type": "integer",
                               "description": "Bot API topic color (optional)"},
                "icon_custom_emoji_id": {"type": "string"},
            },
            "required": ["chat_key", "name"],
        },
    )
    def tgbot_forum_topic_create(chat_key: str, name: str,
                                 icon_color: int = 0,
                                 icon_custom_emoji_id: str = ""
                                 ) -> dict[str, Any]:
        return _tgbot_admin(chat_key, "admin_forum_topic_create",
                            name=name, icon_color=icon_color,
                            icon_custom_emoji_id=icon_custom_emoji_id)

    @registry.register(
        "tgbot_group_info",
        description="Show a rich info card for a Telegram group/channel. Returns rendered HTML.",
        capability=Capability.SOCIAL_READ,
        parameters={
            "type": "object",
            "properties": {"chat_key": {"type": "string"}},
            "required": ["chat_key"],
        },
    )
    def tgbot_group_info(chat_key: str) -> dict[str, Any]:
        from ..social.render import render_group_card
        from ..social.chat.base import ChatRef
        gw = _require_gateway()
        platform, _, rest = chat_key.partition(":")
        if platform != "telegram-bot":
            return {"ok": False,
                    "error": "needs a telegram-bot:<chat_id> key"}
        adapter = (gw._snapshot_adapters() or {}).get("telegram-bot")
        if adapter is None:
            return {"ok": False, "error": "telegram bot not connected"}
        try:
            chat = adapter._api("getChat", chat_id=rest)
            count = adapter._api("getChatMemberCount", chat_id=rest)
            admins_raw = adapter._api("getChatAdministrators", chat_id=rest)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:200]}
        info = {
            "subject": str(chat.get("title") or ""),
            "desc": str(chat.get("description") or ""),
            "size": int(count or 0),
            "admins": [str(a.get("user", {}).get("id") or "")
                       for a in (admins_raw or [])],
            "invite": str(chat.get("invite_link") or ""),
        }
        members = [
            {"jid": str(a.get("user", {}).get("id") or ""),
             "name": str((a.get("user") or {}).get("first_name") or ""),
             "role": "admin"
             if str(a.get("status") or "") in ("creator", "administrator")
             else ""}
            for a in (admins_raw or [])
        ]
        return {"ok": True,
                "rendered": render_group_card(info, members, "telegram"),
                "info": info}

    @registry.register(
        "weekly_content",
        description=(
            "Run the weekly social content pipeline NOW: bounded research "
            "on the niche -> drafts in the owner's voice -> virality "
            "scoring -> queued for REVIEW (never auto-posted) at the "
            "audience's best windows. Returns the review batch. This is "
            "what the durable 'weekly content' job runs every Monday "
            "08:00; call directly only when the owner asks for this "
            "week's batch early. Args: niche (optional, '' = general)."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "type": "object",
            "properties": {
                "niche": {"type": "string",
                          "description": "Content niche; '' = general"},
            },
        },
    )
    def weekly_content_tool(niche: str = "") -> dict[str, Any]:
        from ..llm.brain import brain_for
        from ..research.pipeline import ResearchContext
        from ..social.content_pipeline import weekly_content as _run
        from ..social.drafts import DraftQueue

        context = registry.context

        def _llm(prompt: str) -> str:
            # Brain.complete never raises; on a dead brain it returns an
            # empty text and the pipeline falls back to its honest
            # template drafts. Drafts are queued for review either way —
            # this path never posts or speaks.
            resp = brain_for(context).complete(prompt, task_kind="creative")
            return resp.text if hasattr(resp, "text") else str(resp or "")

        rctx = ResearchContext(
            db=getattr(context, "db", None),
            registry=getattr(context, "tools", None))
        result = _run(
            niche or "",
            db=getattr(context, "db", None),
            queue=DraftQueue(),
            rctx=rctx,
            llm_fn=_llm,
        )
        return {"ok": True, "niche": result.niche,
                "drafts": len(result.drafts),
                "windows_from_data": result.windows_from_data,
                "seconds": round(result.seconds, 1),
                "text": result.message,
                "message": result.message}
