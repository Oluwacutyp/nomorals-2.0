"""L4 — social/chat: bidirectional conversation across platforms.

The companion's three (or four) faces:

* ``telegram``      — Telethon *userbot*: full account control (DMs, groups, channels)
* ``telegram-bot``  — the Bot API: her *own* bot identity, long-polling, no
  webhook server needed; runs alongside the userbot
* ``discord``   — the owner's own Discord *account* (user client): DMs,
  servers, threads; opens first DMs, reports new server members
* ``whatsapp``  — Node/Baileys bridge (``bridge/whatsapp-bridge.mjs``)
* ``local``     — console, for development and tests
* ``sms``         — Twilio SMS fallback (opt-in, costs money): text a number,
  get Devon; restricted command set, DM-only

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


def build_adapter(settings: Any, name: str, *, on_new_member: Any = None,
                  db: Any = None) -> ChatAdapter | None:
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

            # Companion bot ID for the self-reply loop guard: the BotFather
            # token is "<bot_id>:<secret>", so the ID is the part before ':'.
            _bot_token = str(chat.telegram_bot_token or "")
            _companion_bot_id = _bot_token.split(":")[0] if ":" in _bot_token else None
            return TelegramAdapter(
                api_id=chat.telegram_api_id,
                api_hash=chat.telegram_api_hash,
                session_path=settings.resolve(chat.telegram_session),
                chat_allow=chat.telegram_chats,
                media_dir=str(settings.resolve("data/media/telegram")),
                media_in_groups=chat.media_in_groups,
                media_max_mb=chat.media_max_mb,
                threads_enabled=chat.threads_enabled,
                companion_bot_id=_companion_bot_id,
            )
        except Exception as exc:  # noqa: BLE001 - optional dependency or bad creds
            raise ValueError(f"telegram unavailable: {exc}") from exc
    if name == "telegram-bot":
        if not chat.telegram_bot_enabled:
            return None
        try:
            from .telegram import TelegramBotAdapter

            return TelegramBotAdapter(
                token=chat.telegram_bot_token,
                chat_allow=chat.telegram_bot_chats,
                media_dir=str(settings.resolve("data/media/telegram-bot")),
                media_in_groups=chat.media_in_groups,
                media_max_mb=chat.media_max_mb,
            )
        except Exception as exc:  # noqa: BLE001 - bad token or no requests
            raise ValueError(f"telegram-bot unavailable: {exc}") from exc
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
    if name == "webhook":
        if not chat.webhook_enabled:
            return None
        try:
            from .webhook import WebhookAdapter

            return WebhookAdapter(
                host=chat.webhook_host,
                port=chat.webhook_port,
                token=chat.webhook_token,
                reply_url=chat.webhook_reply_url,
                media_dir=str(settings.resolve("data/media/webhook")),
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"webhook unavailable: {exc}") from exc
    if name == "sms":
        from .sms import sms_enabled

        if not sms_enabled(settings):
            return None
        try:
            import os

            from .sms import SMSAdapter

            twilio = None
            if db is not None:
                try:
                    from ...accounts.vault import CredentialVault
                    from ...connectors import create_connector

                    vault = CredentialVault(
                        db, master_passphrase=os.environ.get("NM_VAULT_PASSPHRASE", ""))
                    twilio = create_connector("twilio", vault)
                except Exception as exc:  # noqa: BLE001 - sends fail closed
                    _log.warning("sms: twilio connector unavailable: %s", exc)
            return SMSAdapter(
                twilio,
                from_number=chat.sms_from_number,
                host=chat.sms_host,
                port=chat.sms_port,
                media_dir=str(settings.resolve("data/media/sms")),
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"sms unavailable: {exc}") from exc
    return None


def build_adapters(
    settings: Any, *, on_new_member: Any = None, db: Any = None
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
    for name in ("local", "telegram", "telegram-bot", "discord", "whatsapp",
                 "webhook", "sms"):
        if name not in wanted and not (name == "local" and settings.chat.local_enabled):
            continue
        try:
            adapter = build_adapter(settings, name, on_new_member=on_new_member, db=db)
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
