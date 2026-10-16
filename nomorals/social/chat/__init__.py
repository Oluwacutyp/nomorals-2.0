"""L4 — social/chat: bidirectional conversation across platforms.

The companion's three (or four) faces:

* ``telegram``      — Telethon *userbot*: full account control (DMs, groups, channels)
* ``telegram-bot``  — the Bot API: her *own* gated bot identity,
  long-polling, no webhook server needed; owner DM + allowlist, groups
  get no partner context
* ``telegram-personal`` — a SECOND Bot API bot: her *personal* bot,
  owner-only (NOT public), every chat on it is an owner chat, always
  carries partner context; non-owners get one honest denial, no brain
  call, no credit spend
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
from .chunking import (
    chunk_for_platform,
    chunk_message,
    label_chunks,
    media_kind_for,
    media_plan,
    split_caption,
    strip_label,
)
from .conformance import (
    all_adapter_classes,
    base_send_signature,
    check_adapter_conformance,
)
from .gateway import ChatGateway, parse_chat_keys
from .local import LocalAdapter
from .outbox import Outbox, parse_since
from .platform_limits import PlatformLimits, all_limits, for_platform, register_limits
from .send_retry import (
    classify_send_error,
    get_retry_budget,
    reset_retry_budgets,
    send_with_retry,
)

__all__ = [
    "ChatAdapter",
    "ChatGateway",
    "ChatKind",
    "ChatMessage",
    "ChatRef",
    "LocalAdapter",
    "MediaRef",
    "SendResult",
    "Outbox",
    "PlatformLimits",
    "all_adapter_classes",
    "all_limits",
    "base_send_signature",
    "build_adapters",
    "build_adapter",
    "check_adapter_conformance",
    "chunk_for_platform",
    "chunk_message",
    "classify_send_error",
    "for_platform",
    "get_retry_budget",
    "label_chunks",
    "media_kind_for",
    "media_plan",
    "parse_chat_keys",
    "parse_since",
    "register_limits",
    "reset_retry_budgets",
    "send_with_retry",
    "split_caption",
    "strip_label",
]

_log = get_logger(__name__)


def _split_ids(raw: Any) -> set[str]:
    """Split a comma-separated settings string into a clean ID set."""
    return {p.strip() for p in str(raw or "").split(",") if p.strip()}


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

            # Companion bot IDs for the self-reply loop guard: each
            # BotFather token is "<bot_id>:<secret>", so the ID is the
            # part before ':'. BOTH bots (gated + personal) go in the
            # set — the userbot must not reply to either bot's messages.
            _bot_tokens = [chat.telegram_bot_token,
                           getattr(chat, "telegram_personal_bot_token", "")]
            _companion_ids = {
                _t.split(":")[0] for _t in _bot_tokens
                if _t and ":" in str(_t)
            }
            return TelegramAdapter(
                api_id=chat.telegram_api_id,
                api_hash=chat.telegram_api_hash,
                session_path=settings.resolve(chat.telegram_session),
                chat_allow=chat.telegram_chats,
                media_dir=str(settings.resolve("data/media/telegram")),
                media_in_groups=chat.media_in_groups,
                media_max_mb=chat.media_max_mb,
                threads_enabled=chat.threads_enabled,
                companion_bot_ids=_companion_ids,
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
                adapter_name="telegram-bot",
                partner_context=bool(getattr(
                    chat, "telegram_bot_partner_context", False)),
                owner_only=False,
                lock_path="data/telegram-bot.lock",
            )
        except Exception as exc:  # noqa: BLE001 - bad token or no requests
            raise ValueError(f"telegram-bot unavailable: {exc}") from exc
    if name == "telegram-personal":
        if not getattr(chat, "telegram_personal_bot_enabled", False):
            return None
        try:
            from .telegram import TelegramBotAdapter

            partner = getattr(settings, "partner", None)
            return TelegramBotAdapter(
                token=getattr(chat, "telegram_personal_bot_token", ""),
                chat_allow=getattr(chat, "telegram_personal_bot_chats", ""),
                media_dir=str(settings.resolve("data/media/telegram-personal")),
                media_in_groups=chat.media_in_groups,
                media_max_mb=chat.media_max_mb,
                adapter_name="telegram-personal",
                partner_context=bool(getattr(
                    chat, "telegram_personal_bot_partner_context", True)),
                owner_only=True,
                owner_sender_ids=_split_ids(
                    getattr(partner, "owner_sender_ids", "") or ""),
                owner_chat_keys=_split_ids(
                    getattr(partner, "owner_chats", "") or ""),
                lock_path="data/telegram-personal.lock",
            )
        except Exception as exc:  # noqa: BLE001 - bad token or no requests
            raise ValueError(f"telegram-personal unavailable: {exc}") from exc
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
                outbox_dir=str(settings.resolve(
                    getattr(chat, "whatsapp_outbox_dir", "data/chat/outbox"))),
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
                auth_token=str(getattr(chat, "sms_auth_token", "") or ""),
                public_url=str(getattr(chat, "sms_public_url", "") or ""),
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
    for name in ("local", "telegram", "telegram-bot", "telegram-personal",
                 "discord", "whatsapp", "webhook", "sms"):
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
