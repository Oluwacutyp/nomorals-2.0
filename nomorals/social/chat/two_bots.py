"""Two BotFather bots: one gated, one personal.

Devon talks on Telegram through TWO BotFather bots plus the userbot:

* **Bot 1 — gated** (``telegram-bot``): the existing bot. Owner DM +
  allowlist; groups get conversation + games only, no partner
  (romantic) context, restricted commands answer
  ``"that one's just for the owner."``
* **Bot 2 — personal** (``telegram-personal``): owner-only. Every chat
  on it is an owner chat, it always carries partner context, and it is
  NOT public — anyone who finds it and isn't the owner gets one honest
  denial at the adapter (no brain call, no credit spend) and the
  message never reaches the pipeline.

The split is per-adapter *data*, not code: one
:class:`~nomorals.social.chat.telegram.TelegramBotAdapter` class,
two configs. :class:`TwoBotConfig` holds both; :func:`build_both`
builds them; :func:`route` picks the adapter for a chat or a bot id;
:func:`gate_message` is the pure per-bot gating decision (personal:
owner or deny; gated: owner or restricted-command deny).

``is_owner`` itself stays per-chat data — ``partner.owner_chats``
config or the ``chats.is_owner`` registry column — exactly as the
single owner test (:func:`is_owner_chat`) already computes it. This
module never decides who the owner is; it only decides what each bot
does with the verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...core.logging_setup import get_logger

__all__ = [
    "DENY_NOT_OWNER",
    "DENY_RESTRICTED",
    "TwoBotConfig",
    "bot_id_from_token",
    "build_both",
    "companion_bot_ids",
    "gate_message",
    "route",
]

_log = get_logger(__name__)

#: Honest denial for a stranger on the personal bot. Sent directly by
#: the adapter — the message never reaches the brain, so it spends no
#: credits and no LLM call.
DENY_NOT_OWNER = "This is a private bot — only the owner can chat here."

#: Restricted-command denial on the gated bot (same string the runtime
#: control gate already returns, so both surfaces agree).
DENY_RESTRICTED = "that one's just for the owner."


def bot_id_from_token(token: str | None) -> int | None:
    """Bot id from a BotFather token (``"<bot_id>:<secret>"``).

    None when the token is missing or malformed. The id is the part
    before ':' — no network call.
    """
    try:
        raw = str(token or "").split(":", 1)[0].strip()
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None


@dataclass
class TwoBotConfig:
    """Per-bot config for both BotFather bots. Data, not code."""

    # Bot 1 — gated (existing).
    gated_enabled: bool = False
    gated_token: str = ""
    gated_chats: str = ""          # allowlist, comma-separated chat ids
    gated_partner_context: bool = False
    gated_media_dir: str = "data/media/telegram-bot"
    gated_lock_path: str = "data/telegram-bot.lock"
    # Bot 2 — personal (owner-only).
    personal_enabled: bool = False
    personal_token: str = ""
    personal_chats: str = ""       # extra allowlist filter (owner gate governs)
    personal_partner_context: bool = True
    personal_media_dir: str = "data/media/telegram-personal"
    personal_lock_path: str = "data/telegram-personal.lock"
    personal_deny_text: str = DENY_NOT_OWNER
    # Shared owner data (from partner settings).
    owner_sender_ids: set[str] = field(default_factory=set)
    owner_chat_keys: set[str] = field(default_factory=set)
    # Shared Bot API knobs.
    media_in_groups: bool = False
    media_max_mb: float = 25.0

    @classmethod
    def from_settings(cls, settings: Any) -> "TwoBotConfig":
        """Build from the live Settings tree (chat + partner sections)."""
        chat = settings.chat
        partner = getattr(settings, "partner", None)

        def _split(raw: Any) -> set[str]:
            return {p.strip() for p in str(raw or "").split(",") if p.strip()}

        return cls(
            gated_enabled=bool(chat.telegram_bot_enabled),
            gated_token=str(chat.telegram_bot_token or ""),
            gated_chats=str(chat.telegram_bot_chats or ""),
            gated_partner_context=bool(
                getattr(chat, "telegram_bot_partner_context", False)),
            personal_enabled=bool(
                getattr(chat, "telegram_personal_bot_enabled", False)),
            personal_token=str(
                getattr(chat, "telegram_personal_bot_token", "") or ""),
            personal_chats=str(
                getattr(chat, "telegram_personal_bot_chats", "") or ""),
            personal_partner_context=bool(
                getattr(chat, "telegram_personal_bot_partner_context", True)),
            owner_sender_ids=_split(
                getattr(partner, "owner_sender_ids", "") or ""),
            owner_chat_keys=_split(
                getattr(partner, "owner_chats", "") or ""),
            media_in_groups=bool(chat.media_in_groups),
            media_max_mb=float(chat.media_max_mb),
        )

    def bot_ids(self) -> set[int]:
        """Both bot ids — the userbot loop-guard set."""
        return companion_bot_ids(self)


def companion_bot_ids(config: "TwoBotConfig") -> set[int]:
    """Both companion bot ids from the two tokens.

    Feeds the userbot's ``companion_bot_ids`` set so it never replies
    to either bot's messages.
    """
    out: set[int] = set()
    for token in (config.gated_token, config.personal_token):
        bot_id = bot_id_from_token(token)
        if bot_id is not None:
            out.add(bot_id)
    return out


def build_both(config: TwoBotConfig) -> dict[str, Any]:
    """Build both Bot API adapters (or None per bot when disabled /
    token-less). Never raises for a missing bot — a bot that isn't
    configured simply doesn't run, like every other adapter.

    Keys are ``"telegram-bot"`` (gated) and ``"telegram-personal"``
    (personal), matching the gateway's adapter map.
    """
    from .telegram import TelegramBotAdapter

    out: dict[str, Any] = {}
    if config.gated_enabled and config.gated_token.strip():
        out["telegram-bot"] = TelegramBotAdapter(
            token=config.gated_token,
            chat_allow=config.gated_chats,
            media_dir=config.gated_media_dir,
            media_in_groups=config.media_in_groups,
            media_max_mb=config.media_max_mb,
            adapter_name="telegram-bot",
            partner_context=config.gated_partner_context,
            owner_only=False,
            lock_path=config.gated_lock_path,
        )
    else:
        out["telegram-bot"] = None
    if config.personal_enabled and config.personal_token.strip():
        out["telegram-personal"] = TelegramBotAdapter(
            token=config.personal_token,
            chat_allow=config.personal_chats,
            media_dir=config.personal_media_dir,
            media_in_groups=config.media_in_groups,
            media_max_mb=config.media_max_mb,
            adapter_name="telegram-personal",
            partner_context=config.personal_partner_context,
            owner_only=True,
            owner_sender_ids=set(config.owner_sender_ids),
            owner_chat_keys=set(config.owner_chat_keys),
            deny_text=config.personal_deny_text,
            lock_path=config.personal_lock_path,
        )
    else:
        out["telegram-personal"] = None
    _log.info("two_bots: gated=%s personal=%s",
              out["telegram-bot"] is not None,
              out["telegram-personal"] is not None)
    return out


def route(adapters: Mapping[str, Any], chat: Any,
          bot_id: int | str | None = None) -> Any | None:
    """Pick the adapter for a chat (or a bot id).

    * ``bot_id`` given → the adapter whose token-derived bot id matches
      (used when routing a raw Bot API update / delivery by bot).
    * otherwise → the adapter whose name matches ``chat.platform``
      (``"telegram-personal"`` → personal, ``"telegram-bot"`` → gated).
    * anything else (userbot ``"telegram"``, unknown) → None: this
      router only owns the two Bot API bots.

    Never raises; None means "not one of the two bots".
    """
    if bot_id is not None:
        try:
            wanted = int(bot_id)
        except (TypeError, ValueError):
            return None
        for adapter in adapters.values():
            if adapter is None:
                continue
            if getattr(adapter, "_bot_id", None) == wanted:
                return adapter
        return None
    platform = str(getattr(chat, "platform", "") or "").strip().lower()
    if platform in ("telegram-bot", "telegram-personal"):
        return adapters.get(platform)
    return None


def gate_message(adapter: Any, *, is_owner: bool,
                 is_restricted_command: bool = False) -> tuple[str, str]:
    """Pure per-bot gating decision for one inbound message.

    Returns ``(verdict, reply)`` where verdict is ``"allow"`` or
    ``"deny"`` and reply is the denial text ("" when allowed).
    ``is_owner`` is the caller's per-chat verdict (owner_chats config
    or ``chats.is_owner``) — this function never recomputes it.

    * Personal bot: non-owner → deny with the honest private-bot
      text. No brain call, no credit spend.
    * Gated bot: non-owner + restricted command → deny with
      ``"that one's just for the owner."`` Owner or public command →
      allow (partner context is stamped by the adapter and applied
      by the brain).
    """
    if adapter is None:
        return "deny", DENY_NOT_OWNER
    if bool(getattr(adapter, "owner_only", False)):
        if is_owner:
            return "allow", ""
        return "deny", str(getattr(adapter, "deny_text", "") or DENY_NOT_OWNER)
    if not is_owner and is_restricted_command:
        return "deny", DENY_RESTRICTED
    return "allow", ""
