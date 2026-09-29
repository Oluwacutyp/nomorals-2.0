"""L4 — social/chat: bidirectional conversation across platforms.

The companion's three (or four) faces:

* ``telegram``  — Telethon *userbot*: full account control (DMs, groups, channels)
* ``discord``   — the owner's own Discord *account* (user client): DMs,
  servers, threads; opens first DMs, reports new server members
* ``whatsapp``  — Node/Baileys bridge (``bridge/whatsapp-bridge.mjs``)
* ``local``     — console, for development and tests

``build_adapters`` turns settings into whichever adapters are configured and
importable, skipping (not failing on) platforms whose optional dependency is
missing. The gateway then runs them all at once.

    gateway = ChatGateway(build_adapters(settings), db=context.db, dry_run=False)
    gateway.start(runtime.on_message)
"""

from __future__ import annotations

from typing import Any

from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, MediaRef, SendResult
from .gateway import ChatGateway, parse_chat_keys
from .local import LocalAdapter

__all__ = [
    "ChatAdapter",
    "ChatGateway",
    "ChatKind",
    "ChatMessage",
    "ChatRef",
    "LocalAdapter",
    "MediaRef",
    "SendResult",
    "build_adapters",
    "build_adapter",
]

_log = get_logger(__name__)


def build_adapter(settings: Any, name: str, *, on_new_member: Any = None) -> ChatAdapter | None:
    """Build one adapter by name; None when unavailable (missing dependency,
    disabled, or bad credentials). Used both at boot and for hot starts."""
    name = name.strip().lower()
    chat = settings.chat
    if name == "local":
        return LocalAdapter()
    if name == "telegram":
        if not chat.telegram_enabled:
            return None
        try:
            from .telegram import TelegramAdapter

            return TelegramAdapter(
                api_id=chat.telegram_api_id,
                api_hash=chat.telegram_api_hash,
                session_path=settings.resolve(chat.telegram_session),
                chat_allow=chat.telegram_chats,
                media_dir=str(settings.resolve("data/media/telegram")),
                media_in_groups=chat.media_in_groups,
                media_max_mb=chat.media_max_mb,
                threads_enabled=chat.threads_enabled,
            )
        except Exception as exc:  # noqa: BLE001 - optional dependency or bad creds
            raise ValueError(f"telegram unavailable: {exc}") from exc
    if name == "discord":
        if not chat.discord_enabled:
            return None
        try:
            from .discord import DiscordAdapter

            return DiscordAdapter(
                token=chat.discord_token,
                channel_allow=chat.discord_channels,
                media_dir=str(settings.resolve("data/media/discord")),
                greet_new=chat.discord_greet_new,
                on_new_member=on_new_member,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"discord unavailable: {exc}") from exc
    if name == "whatsapp":
        if not chat.whatsapp_enabled:
            return None
        try:
            from .whatsapp import WhatsAppAdapter

            return WhatsAppAdapter(
                host=chat.whatsapp_host,
                port=chat.whatsapp_port,
                media_dir=str(settings.resolve("data/media/whatsapp")),
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"whatsapp unavailable: {exc}") from exc
    return None


def build_adapters(
    settings: Any, *, on_new_member: Any = None
) -> tuple[dict[str, ChatAdapter], list[str]]:
    """Instantiate the chat adapters that are enabled in settings.

    Returns ``(adapters, skipped)`` where ``skipped`` names platforms that are
    enabled but unusable (missing dependency or missing credentials). Boot
    continues without them; the partner talks on what's left.

    ``on_new_member`` (guild_name, member dict) is handed to the Discord
    adapter so the runtime can greet people who join the owner's servers.
    """
    partner = settings.partner
    wanted = {p.strip().lower() for p in (partner.platforms or "").split(",") if p.strip()}
    adapters: dict[str, ChatAdapter] = {}
    skipped: list[str] = []
    for name in ("local", "telegram", "discord", "whatsapp"):
        if name not in wanted and not (name == "local" and settings.chat.local_enabled):
            continue
        try:
            adapter = build_adapter(settings, name, on_new_member=on_new_member)
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        if adapter is None:
            skipped.append(f"{name} (disabled in settings)")
        else:
            adapters[name] = adapter
    if skipped:
        _log.warning("chat adapters skipped: %s", "; ".join(skipped))
    return adapters, skipped
