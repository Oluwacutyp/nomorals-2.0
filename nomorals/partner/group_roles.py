"""Group role resolution — who is admin in THIS group.

Three roles:
- "owner"   — Devon's owner (resolved elsewhere, always full access)
- "admin"   — group/channel admin in this specific chat
- "member"  — everyone else (regular member, restricted, unknown)

Resolution is platform-specific:
- Telegram Bot API: getChatMember → status field
- Telegram MTProto: GetParticipantRequest → participant type
- WhatsApp: group_info admins list / group_participants roster

Results are cached for 60s per (platform, chat_key, user_id).
Fail closed: lookup errors → "member", never "admin".
"""

from __future__ import annotations

import time
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "ROLE_ADMIN",
    "ROLE_MEMBER",
    "ROLE_UNKNOWN",
    "resolve_group_role",
    "clear_role_cache",
]

_log = get_logger(__name__)

ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
ROLE_UNKNOWN = "unknown"

_CACHE_TTL_S = 60.0
_cache: dict[tuple[str, str, str], tuple[str, float]] = {}


def clear_role_cache() -> None:
    """Drop all cached role lookups (tests, role changes)."""
    _cache.clear()


def _cached(platform: str, chat_key: str, user_id: str) -> str | None:
    key = (platform, chat_key, user_id)
    hit = _cache.get(key)
    if hit is None:
        return None
    role, ts = hit
    if time.time() - ts > _CACHE_TTL_S:
        _cache.pop(key, None)
        return None
    return role


def _store(platform: str, chat_key: str, user_id: str, role: str) -> None:
    _cache[(platform, chat_key, user_id)] = (role, time.time())


def _telegram_bot_role(adapter: Any, chat_key: str, user_id: str) -> str:
    """Resolve via Bot API getChatMember."""
    try:
        out = adapter.admin_member(chat_key, user_id)
    except Exception as exc:  # noqa: BLE001
        _log.debug("bot role lookup failed: %s", exc)
        return ROLE_UNKNOWN
    if not isinstance(out, dict) or not out.get("ok"):
        return ROLE_UNKNOWN
    status = str(out.get("status") or "").lower()
    if status in ("creator", "administrator"):
        return ROLE_ADMIN
    return ROLE_MEMBER


def _telegram_mtproto_role(adapter: Any, chat_key: str, user_id: str) -> str:
    """Resolve via MTProto GetParticipantRequest."""
    try:
        out = adapter.admin_member_mtproto(chat_key, user_id)
    except Exception as exc:  # noqa: BLE001
        _log.debug("mtproto role lookup failed: %s", exc)
        return ROLE_UNKNOWN
    if not isinstance(out, dict) or not out.get("ok"):
        return ROLE_UNKNOWN
    ptype = str(out.get("participant_type") or "").lower()
    if "creator" in ptype or "admin" in ptype:
        return ROLE_ADMIN
    return ROLE_MEMBER


#: Discord permission bits that count as "runs this server" for gating.
#: ADMINISTRATOR 0x8, MANAGE_GUILD 0x20, MANAGE_MESSAGES 0x2000,
#: MODERATE_MEMBERS 0x1000000. Fail closed: unknown shapes -> not admin.
_DISCORD_ADMIN_PERMS = 0x8 | 0x20 | 0x2000 | 0x1000000
_DISCORD_ADMIN_NAMES = frozenset({"admin", "administrator", "moderator", "mod"})


def _discord_role(adapter: Any, chat_key: str, user_id: str) -> str:
    """Resolve via the Discord adapter's member-roles hook.

    The adapter exposes ``discord_member_roles(chat_key, user_id)`` ->
    ``{"ok": True, "roles": [{"name": str, "permissions": int}]}``
    (a plain list of role names is also accepted). Any role carrying an
    admin-ish permission bit — or named like a mod — resolves admin.
    Everything else (errors, unknown shapes, no roles) fails closed.
    """
    try:
        out = adapter.discord_member_roles(chat_key, user_id)
    except Exception as exc:  # noqa: BLE001
        _log.debug("discord role lookup failed: %s", exc)
        return ROLE_UNKNOWN
    roles: list[Any] = []
    if isinstance(out, dict) and out.get("ok"):
        roles = list(out.get("roles") or [])
    elif isinstance(out, (list, tuple)):
        roles = list(out)
    else:
        return ROLE_UNKNOWN
    for role in roles:
        if isinstance(role, str):
            if role.strip().lower() in _DISCORD_ADMIN_NAMES:
                return ROLE_ADMIN
            continue
        if not isinstance(role, dict):
            continue
        name = str(role.get("name") or "").strip().lower()
        if name in _DISCORD_ADMIN_NAMES:
            return ROLE_ADMIN
        try:
            perms = int(role.get("permissions") or 0)
        except (TypeError, ValueError):
            perms = 0
        if perms & _DISCORD_ADMIN_PERMS:
            return ROLE_ADMIN
    return ROLE_MEMBER if roles else ROLE_UNKNOWN


def _whatsapp_role(adapter: Any, chat_key: str, user_id: str) -> str:
    """Resolve via group admins list."""
    try:
        info = adapter.group_info(chat_key)
    except Exception as exc:  # noqa: BLE001
        _log.debug("whatsapp role lookup failed: %s", exc)
        return ROLE_UNKNOWN
    if not isinstance(info, dict):
        return ROLE_UNKNOWN
    admins = info.get("admins") or []
    # Normalize JIDs for comparison (strip resource, lowercase)
    norm = lambda j: str(j).split("/")[0].split("@")[0].lower()  # noqa: E731
    target = norm(user_id)
    for admin_jid in admins:
        if norm(admin_jid) == target:
            return ROLE_ADMIN
    # Also check the owner field
    owner = str(info.get("owner") or "")
    if owner and norm(owner) == target:
        return ROLE_ADMIN
    return ROLE_MEMBER


def resolve_group_role(
    platform: str,
    adapter: Any,
    chat_key: str,
    user_id: str,
) -> str:
    """Resolve a user's role in a specific group. Cached 60s. Fail closed.

    Returns "admin", "member", or "unknown". "unknown" is treated as "member"
    by the permission layer — never grants admin.
    """
    platform = (platform or "").lower()
    chat_key = str(chat_key or "")
    user_id = str(user_id or "")
    if not chat_key or not user_id:
        return ROLE_UNKNOWN

    hit = _cached(platform, chat_key, user_id)
    if hit is not None:
        return hit

    if platform in ("telegram", "telegram-bot"):
        # Bot API path — the adapter tells us which it is
        role = _telegram_bot_role(adapter, chat_key, user_id)
    elif platform == "telegram-mtproto":
        role = _telegram_mtproto_role(adapter, chat_key, user_id)
    elif platform in ("whatsapp", "wa"):
        role = _whatsapp_role(adapter, chat_key, user_id)
    elif platform in ("discord",):
        role = _discord_role(adapter, chat_key, user_id)
    else:
        _log.debug("no role resolver for platform %r", platform)
        role = ROLE_UNKNOWN

    _store(platform, chat_key, user_id, role)
    return role
