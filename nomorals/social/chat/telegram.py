"""Telegram adapter — userbot style, full control.

This drives a real user account over MTProto (Telethon), which is what gives
the "full control" the companion needs: read and send in DMs, groups, and
channels, reply to specific messages, download media, and (when the account
has permission) post into channels.

Operational notes that matter:

* **First run is interactive.** ``client.start()`` asks for the phone number,
  login code, and 2FA password on the console. After that the session file
  (``*.session``) is the credential — treat it like a password and keep it
  out of any repo that gets backed up to git.
* **Userbot accounts can be limited by Telegram** if they behave bot-like at
  scale. The gateway's per-platform rate limit and the autonomy agent's caps
  exist for this reason.
* **Talking to yourself works.** The owner can message the companion by
  texting their own account (Saved Messages): outgoing messages in the
  self-chat are treated as inbound from the owner. Every other outgoing
  message is ignored, and messages this adapter sent itself are filtered by
  id so replies never re-trigger replies (loop guard).
* Everything async lives inside ``run()`` on this adapter's own thread.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import time
from pathlib import Path
from typing import Any

from ...core.errors import ValidationError
from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult

__all__ = ["TelegramAdapter", "TelegramBotAdapter"]

_log = get_logger(__name__)

def media_download_allowed(kind: str, doc_size: int, media_in_groups: bool, media_max_mb: float) -> bool:
    """Pure gate for inbound media downloads (unit-testable without Telethon).

    Groups flood media (spam groups post a photo every few seconds), so group
    downloads are opt-in; oversized files are always skipped (0 = no cap).
    """
    if kind == ChatKind.GROUP and not media_in_groups:
        return False
    if media_max_mb > 0 and doc_size > media_max_mb * 1024 * 1024:
        return False
    return True


_MIME_HINTS = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".mp4": "video/mp4",
    ".mkv": "video/x-matroska", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
    ".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _mime_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _MIME_HINTS.get(ext, "application/octet-stream")


def _kind_for(entity: Any) -> str:
    if getattr(entity, "megagroup", False) or getattr(entity, "gigagroup", False):
        return ChatKind.GROUP
    if getattr(entity, "channel", False) and not getattr(entity, "megagroup", False):
        return ChatKind.CHANNEL
    return ChatKind.DM


class TelegramAdapter(ChatAdapter):
    """A Telethon userbot: full account control over MTProto."""

    name = "telegram"

    def __init__(
        self,
        *,
        api_id: int | str,
        api_hash: str,
        session_path: str = "data/telegram.session",
        chat_allow: str = "",
        media_dir: str = "data/media/telegram",
        media_in_groups: bool = False,
        media_max_mb: float = 20.0,
        threads_enabled: bool = True,
        companion_bot_id: int | str | None = None,
    ) -> None:
        super().__init__(media_dir=media_dir)
        self.media_in_groups = bool(media_in_groups)
        self.media_max_mb = float(media_max_mb)
        # ID of the companion BotFather bot (if any). The bot's messages
        # arrive at this userbot as incoming — without this, the
        # _sender_is_bot guard relies solely on Telethon's .bot flag,
        # which isn't always populated (unresolved senders, certain
        # chat types). An explicit ID match is bulletproof.
        try:
            self._companion_bot_id = int(companion_bot_id) if companion_bot_id else None
        except (TypeError, ValueError):
            self._companion_bot_id = None
        try:
            self.api_id = int(api_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("telegram api_id must be an integer", field="api_id") from exc
        self.threads_enabled = bool(threads_enabled)
        self.api_hash = api_hash
        self.session_path = session_path
        self.chat_allow = {c.strip() for c in chat_allow.split(",") if c.strip()}
        self._client: Any = None
        self._me_id: int | None = None
        # the account's own handle/name — used to detect tags/mentions
        self._me_username: str = ""
        self._me_first_name: str = ""
        # ids of messages this adapter sent — the self-chat loop guard
        self._sent_ids: set[int] = set()
        # the connection's event loop (set in run()); all outbound coroutines
        # must run on it — Telethon binds the client to that loop
        self._loop: asyncio.AbstractEventLoop | None = None
        # Cache of input entities from inbound messages — keyed by chat_id.
        # Telethon's get_entity() fails for users not in the session cache,
        # but get_input_entity() works with input peers stored from received
        # messages. We cache those so outbound sends can reply to DM chats
        # from users we've heard from but can't fully resolve.
        self._input_entity_cache: dict[str, Any] = {}
        #: Hard cap on the input-entity cache: it gains an entry per new
        #: contact/dialog, and an uncapped dict on a years-running userbot is
        #: a slow memory leak. Eviction is oldest-first (dict insertion order).
        self._input_entity_cache_max = 2000

    def _cache_input_entity(self, chat_id: str, entity: Any) -> None:
        """Store an input entity, evicting the oldest entries past the cap."""
        self._input_entity_cache[chat_id] = entity
        overflow = len(self._input_entity_cache) - self._input_entity_cache_max
        if overflow > 0:
            for old in list(self._input_entity_cache)[:overflow]:
                self._input_entity_cache.pop(old, None)

    def _run_on_loop(self, coro: Any, timeout: float = 60.0) -> Any:
        """Run a coroutine on the connection's loop from another thread.

        A fresh ``asyncio.run()`` here would fail with "the asyncio event
        loop must not change after connection" — the client is bound to the
        loop that holds the live connection.
        """
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("telegram adapter is not connected")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── self-chat (Saved Messages) support ─────────────────────────────────
    @staticmethod
    def _me_id_from(me: Any) -> int | None:
        try:
            return int(getattr(me, "id", 0) or 0) or None
        except (TypeError, ValueError):
            return None

    def _remember_sent(self, message_id: Any) -> None:
        try:
            mid = int(message_id)
        except (TypeError, ValueError):
            return
        if mid:
            self._sent_ids.add(mid)
            if len(self._sent_ids) > 2000:
                # the guard only needs recent ids; keep the set bounded
                for old in list(self._sent_ids)[:1000]:
                    self._sent_ids.discard(old)

    def _mentions_me(self, text: str) -> bool:
        """True if this text tags the account itself — ``@username`` or its
        first name.  The companion's persona name is a DIFFERENT string, so
        without this check every Telegram tag/mention in a group would be
        silently ignored."""
        lowered = (text or "").lower()
        if not lowered.strip():
            return False
        if self._me_username and re.search(
            rf"(?<![a-z0-9_])@{re.escape(self._me_username.lower())}(?![a-z0-9_])", lowered
        ):
            return True
        name = self._me_first_name.lower()
        if len(name) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", lowered):
            return True
        return False

    def _outgoing_is_self_chat(self, event: Any) -> bool:
        """True iff this outgoing event is the owner texting themselves
        (Saved Messages) and not one of our own replies (loop guard)."""
        if not self._me_id:
            return False
        if getattr(event, "chat_id", None) != self._me_id:
            return False
        if getattr(getattr(event, "message", None), "action", None):
            return False  # service message (pin, edit notice, ...)
        try:
            mid = int(getattr(event.message, "id", 0) or 0)
        except (TypeError, ValueError):
            mid = 0
        return mid not in self._sent_ids

    def _sender_is_bot(self, event: Any, message: Any) -> bool:
        """True iff the sender of this inbound message is a bot account.

        The companion BotFather bot runs as a separate Telegram account. When
        both adapters are live, the bot's messages arrive at this userbot as
        incoming — without this guard the bot would reply to its own output
        in a loop.
        """
        # Primary: explicit companion bot ID match — bulletproof even when
        # Telethon hasn't resolved the sender entity.
        if self._companion_bot_id:
            for candidate in (
                getattr(event, "sender_id", None),
                getattr(getattr(event, "message", None), "sender_id", None),
                getattr(message, "sender_id", None),
            ):
                try:
                    if candidate is not None and int(candidate) == self._companion_bot_id:
                        return True
                except (TypeError, ValueError):
                    pass
        # Secondary: Telethon resolves event.sender to a User with .bot flag.
        sender = getattr(event, "sender", None)
        if sender is not None and bool(getattr(sender, "bot", False)):
            return True
        # Fallback: the raw message's from_id may carry a bot flag on some
        # Telethon versions (PeerUser with bot attribute on the sender object).
        from_id = getattr(message, "from_id", None)
        if from_id is not None and bool(getattr(from_id, "bot", False)):
            return True
        return False

    # ── entity resolution ────────────────────────────────────────────────────
    async def _resolve(self, chat: ChatRef) -> Any:
        """Resolve a chat entity, using cached input entities from inbound messages.
        
        Telethon's get_entity() fails for users not in the session cache (fresh
        sessions, privacy settings), causing send failures. Resolution order:
        1. Cached input entity (from received messages — most reliable for DMs)
        2. client.get_input_entity() (uses Telethon's internal peer cache)
        3. client.get_entity() (full resolution — fails for unknown users)
        4. Raw integer ID (Telethon can sometimes send to just the ID)
        """
        chat_id = chat.chat_id
        
        # 1. Check the input entity cache (from inbound messages)
        if chat_id in self._input_entity_cache:
            _log.debug("telegram: using cached input entity for chat_id=%s", chat_id)
            return self._input_entity_cache[chat_id]
        
        # 2. Try get_input_entity (uses Telethon's internal peer cache)
        if self._client is not None:
            try:
                if chat_id.startswith("@"):
                    return await self._client.get_input_entity(chat_id)
                return await self._client.get_input_entity(int(chat_id))
            except Exception:  # noqa: BLE001
                pass
        
        # 3. Try get_entity (full resolution)
        if self._client is not None:
            try:
                if chat_id.startswith("@"):
                    return await self._client.get_entity(chat_id)
                return await self._client.get_entity(int(chat_id))
            except Exception:  # noqa: BLE001
                pass
        
        # 4. Last resort: return the raw integer ID.
        # Telethon's send_message can sometimes work with just a user ID.
        if chat_id.lstrip("-").isdigit():
            _log.warning("telegram: could not resolve chat_id=%s, using raw ID", chat_id)
            return int(chat_id)
        
        raise ValueError(f"cannot resolve chat_id={chat_id!r}")

    # ── inbound ──────────────────────────────────────────────────────────────
    def preflight(self) -> None:
        """First-run login (once): Telethon's interactive start (phone number
        → code → optional 2FA) must own the keyboard before the console does.

        A session file existing is NOT proof of authorization — an
        interrupted first run leaves an unauthenticated stub behind — so we
        ask Telethon directly. After a successful login the session file is
        the credential and every later boot is silent."""
        from telethon import TelegramClient

        # Each client binds to the event loop of its first connection, so the
        # auth check and the interactive login must use SEPARATE clients —
        # sharing one across two asyncio.run() loops fails with "the asyncio
        # event loop must not change after connection".
        probe = TelegramClient(self.session_path, self.api_id, self.api_hash)
        try:
            authorized = asyncio.run(self._is_authorized(probe))
        except Exception as exc:  # noqa: BLE001 - network hiccup: let run() retry
            _log.warning("telegram preflight check failed (%s); adapter start will retry", exc)
            return
        if authorized:
            return
        print("\n" + "=" * 62, flush=True)
        print("TELEGRAM FIRST-TIME LOGIN (one-time)", flush=True)
        print("Enter your number in international format (e.g. 234803...),", flush=True)
        print("then the code Telegram sends you. The console starts after.", flush=True)
        print("=" * 62, flush=True)
        login = TelegramClient(self.session_path, self.api_id, self.api_hash)

        async def _login() -> None:
            try:
                await login.start()
            finally:
                await login.disconnect()

        asyncio.run(_login())
        _log.info("telegram first-run login complete; session saved to %s", self.session_path)

    async def _is_authorized(self, client: Any) -> bool:
        try:
            await client.connect()
            return bool(await client.is_user_authorized())
        finally:
            await client.disconnect()

    def run(self, handler: IncomingHandler) -> None:
        from telethon import TelegramClient, events

        os.makedirs(self.media_dir, exist_ok=True)
        client = TelegramClient(self.session_path, self.api_id, self.api_hash)
        self._client = client

        async def _loop() -> None:
            try:
                self._loop = asyncio.get_running_loop()
                await client.start()
                # Pre-fetch all dialogs to populate Telethon's internal entity
                # cache. Without this, get_entity() fails for users not yet
                # cached in the session (fresh login, privacy settings).
                # This makes DM replies work for everyone the account has
                # ever chatted with.
                try:
                    dialog_count = 0
                    async for dialog in client.iter_dialogs(limit=200):
                        dialog_count += 1
                        entity = getattr(dialog, "entity", None)
                        input_entity = getattr(dialog, "input_entity", None)
                        if entity is not None and input_entity is not None:
                            eid = str(getattr(entity, "id", ""))
                            if eid:
                                self._cache_input_entity(eid, input_entity)
                    _log.info("telegram: pre-cached %d dialogs (%d input entities)",
                              dialog_count, len(self._input_entity_cache))
                except Exception as exc:  # noqa: BLE001
                    _log.warning("telegram: dialog pre-fetch failed: %s", exc)
                # explicit get_me: client.me can still be None right after
                # start() — the self-chat gate depends on this id
                me = await client.get_me()
                self._me_id = self._me_id_from(me)
                self._me_username = str(getattr(me, "username", None) or "")
                self._me_first_name = str(getattr(me, "first_name", None) or "")
                if self._me_id:
                    _log.info(
                        "telegram userbot connected as %s (id=%s)",
                        getattr(me, "username", None) or getattr(me, "first_name", None),
                        self._me_id,
                    )
                else:
                    _log.warning("telegram: own user id unknown — self-chat (Saved Messages) is off")

                @client.on(events.NewMessage(incoming=True))
                async def on_incoming(event: Any) -> None:
                    # Loud on purpose: this line is the proof that a DM actually
                    # reached the userbot's update stream. If it never prints,
                    # the message is being stopped by Telegram itself (privacy
                    # settings / wrong account), not by this code.
                    _log.info("telegram: incoming message event (chat_id=%s)",
                              getattr(event, "chat_id", "?"))
                    await self._handle_inbound(event, client, handler, allow_check=True)

                # Self-chat (Saved Messages): when the owner texts their own
                # account, Telethon sees it as an OUTGOING message. Treat those
                # as inbound from the owner; everything else stays ignored.
                @client.on(events.NewMessage(outgoing=True))
                async def on_outgoing(event: Any) -> None:
                    if not self._outgoing_is_self_chat(event):
                        return
                    await self._handle_inbound(event, client, handler, allow_check=False)

                await client.run_until_disconnected()
            finally:
                try:
                    await client.disconnect()
                except Exception:  # noqa: BLE001
                    pass

        try:
            asyncio.run(_loop())
        except Exception as exc:  # noqa: BLE001 - surfaced to the guarded runner
            if self.stopped:
                return
            raise exc

    async def _handle_inbound(self, event: Any, client: Any, handler: IncomingHandler, *,
                              allow_check: bool) -> None:
        """Process one NewMessage event and deliver it to the handler.

        Telethon hands the handler an *Event* object in most builds, but some
        versions pass the raw *Message* — support both shapes.  The chat
        entity is resolved through the CLIENT (``client.get_entity(chat_id)``)
        rather than ``event.get_entity()``, which does not exist on every
        Telethon version; a missing method there used to drop every inbound
        message silently.
        
        Fallback chain (DM chats from unknown users break get_entity):
        1. event.chat / event.sender (already resolved by Telethon)
        2. input_chat / input_sender (Telethon internal refs)
        3. client.get_entity(chat_id) (the original method)
        4. Synthesize a minimal entity from event IDs (last resort — 
           keeps the message alive even when Telethon can't resolve the user)"""
        message = getattr(event, "message", None) or event
        chat_id = str(getattr(event, "chat_id", "") or "")
        # ── bot self-reply loop guard ──────────────────────────────────
        # When the companion BotFather bot runs alongside this userbot, the
        # bot's outgoing messages arrive here as incoming (different account).
        # Processing them would make the bot reply to itself in a loop.
        # Drop any message sent by a bot account.
        if self._sender_is_bot(event, message):
            _log.debug(
                "telegram: DROPPED inbound chat_id=%s — sender is a bot (self-reply loop guard)",
                chat_id,
            )
            return
        entity = None
        
        # Try to get entity from the event first (already resolved by Telethon)
        event_chat = getattr(event, "chat", None)
        event_sender = getattr(event, "sender", None)
        
        # For DMs, try sender first (the person who messaged us).
        # For groups/channels, prefer the chat entity (the group itself).
        # Check chat kind to decide: if it looks like a group/channel, use chat;
        # otherwise (DM), use sender.
        if event_chat is not None:
            is_group_or_channel = (
                getattr(event_chat, "megagroup", False)
                or getattr(event_chat, "gigagroup", False)
                or getattr(event_chat, "channel", False)
            )
            if is_group_or_channel:
                entity = event_chat
            elif event_sender is not None:
                entity = event_sender
            else:
                entity = event_chat
        elif event_sender is not None:
            entity = event_sender
            
        # Last resort: try input_chat or input_sender.
        # CRITICAL ORDER: input_chat FIRST. For a group message, input_sender
        # is the SENDER's peer (their DM), not the group. If we resolve
        # input_sender first and it succeeds, the message gets misclassified
        # as a DM and replies go to the sender's inbox instead of the group.
        # (This was the persistent group→DM redirect bug.)
        if entity is None:
            input_chat = getattr(event, "input_chat", None)
            input_sender = getattr(event, "input_sender", None)

            _log.debug(
                "telegram: entity resolve via input peers chat_id=%s "
                "input_chat=%r input_sender=%r is_group=%r is_channel=%r",
                chat_id, input_chat, input_sender,
                getattr(event, "is_group", None),
                getattr(event, "is_channel", None),
            )

            if input_chat is not None:
                try:
                    entity = await client.get_entity(input_chat)
                    _log.debug(
                        "telegram: entity resolved via input_chat chat_id=%s -> %r",
                        chat_id, getattr(entity, "id", None),
                    )
                except Exception:  # noqa: BLE001
                    pass

            if entity is None and input_sender is not None:
                try:
                    entity = await client.get_entity(input_sender)
                    _log.debug(
                        "telegram: entity resolved via input_sender chat_id=%s -> %r",
                        chat_id, getattr(entity, "id", None),
                    )
                except Exception:  # noqa: BLE001
                    pass
        
        # Fallback: resolve through client using chat_id
        if entity is None:
            try:
                if chat_id.lstrip("-").isdigit():
                    entity = await client.get_entity(int(chat_id))
                else:
                    entity = await client.get_entity(chat_id)
            except Exception:  # noqa: BLE001
                pass
        
        # FINAL FALLBACK: synthesize a minimal entity from event data.
        # This keeps DM messages alive even when Telethon can't resolve
        # the user (fresh session, user not cached, privacy settings).
        if entity is None:
            # CRITICAL: use the CHAT id (event.chat_id), not the sender id.
            # The old code preferred event.sender_id here, which for a group
            # message is the SENDER's user ID — synthesizing a DM entity for
            # a group chat. The runtime then treated the message as a DM and
            # every reply went to the sender's inbox instead of the group.
            # (This was the group→DM redirect bug.)
            target_id = chat_id or str(getattr(event, "sender_id", "") or "")
            if target_id:
                # Build a minimal synthetic entity
                from types import SimpleNamespace
                # Detect group vs DM from the event so _kind_for() classifies
                # the synthetic entity correctly.
                is_group = bool(
                    getattr(event, "is_group", False)
                    or getattr(event, "is_channel", False)
                )
                sender_username = ""
                sender_first_name = f"user_{target_id}"

                # Try to extract username from the message's sender info
                msg_from = getattr(message, "from_id", None) or getattr(message, "peer_id", None)
                if msg_from is not None:
                    from_user_id = getattr(msg_from, "user_id", None) or target_id
                    sender_first_name = f"user_{from_user_id}"

                entity = SimpleNamespace(
                    id=int(str(target_id).lstrip("-")),
                    first_name=sender_first_name,
                    last_name=None,
                    username=sender_username,
                    title=None,
                    megagroup=is_group,
                    gigagroup=False,
                    channel=is_group,
                    is_forum=False,
                )
                _log.info(
                    "telegram: synthesized minimal entity for chat_id=%s (user not cached, group=%s)",
                    chat_id,
                    is_group,
                )
            else:
                _log.warning(
                    "telegram: DROPPED inbound chat_id=%s — all entity resolve methods failed",
                    chat_id,
                )
                return
        
        # Cache the input entity for outbound sends. Telethon's get_entity()
        # fails for users not in the session cache, but we can use the input
        # peer from received messages to send replies.
        #
        # CRITICAL: never cache input_sender under a group chat's ID. If
        # input_chat is missing on a group message and we fall back to
        # input_sender, the group's cache entry points at the sender's DM
        # peer — and every later reply to that group lands in the DM instead.
        # (This was the group→DM redirect bug.)
        input_chat = getattr(event, "input_chat", None)
        is_group_entity = (
            getattr(entity, "megagroup", False)
            or getattr(entity, "gigagroup", False)
            or getattr(entity, "channel", False)
        )
        if is_group_entity:
            input_entity = input_chat
        else:
            input_entity = input_chat or getattr(event, "input_sender", None)
        if input_entity is not None:
            entity_id = str(getattr(entity, "id", ""))
            if entity_id:
                self._cache_input_entity(entity_id, input_entity)
                _log.debug("telegram: cached input entity for chat_id=%s", entity_id)
        
        chat_id = str(getattr(entity, "id", ""))
        if allow_check and self.chat_allow and chat_id not in self.chat_allow and not any(
            c == chat_id for c in self.chat_allow
        ):
            _log.info(
                "telegram: ignoring message in %s (id=%s) — not in NM_CHAT_TELEGRAM_CHATS",
                getattr(entity, "title", None) or chat_id,
                chat_id,
            )
            return
        kind = _kind_for(entity)
        # ── diagnostic: log full routing decision for group messages ──
        # The user reported group commands going to DM. This logs every
        # data point in the routing chain so we can see exactly where a
        # group message gets misclassified.
        _log.debug(
            "telegram: inbound routing decision "
            "event.chat_id=%r event.sender_id=%r event.is_group=%r event.is_channel=%r "
            "entity.id=%r entity.megagroup=%r entity.channel=%r entity.title=%r "
            "kind=%r final_chat_id=%r",
            getattr(event, "chat_id", None),
            getattr(event, "sender_id", None),
            getattr(event, "is_group", None),
            getattr(event, "is_channel", None),
            getattr(entity, "id", None),
            getattr(entity, "megagroup", None),
            getattr(entity, "channel", None),
            getattr(entity, "title", None),
            kind,
            chat_id,
        )
        text = getattr(message, "raw_text", None) or ""
        media: list[MediaRef] = []
        if getattr(message, "media", None):
            # Download gates: groups flood media (spam groups post a photo
            # every few seconds) — by default we only pull media in DMs, and
            # never pull files above the cap.
            doc_size = 0
            doc = getattr(getattr(message, "media", None), "document", None)
            if doc is not None:
                doc_size = int(getattr(doc, "size", 0) or 0)
            if not media_download_allowed(kind, doc_size, self.media_in_groups, self.media_max_mb):
                _log.debug("telegram: skipping media download (group/size gate)")
            else:
                try:
                    fname = f"tg-{int(time.time() * 1000)}-{len(media)}"
                    path = await client.download_media(message, file=f"{self.media_dir}/{fname}")
                    if path:
                        if not path.endswith(tuple(_MIME_HINTS)):
                            ext = os.path.splitext(str(path))[1]
                            if not ext:
                                path = f"{path}.bin"
                        media.append(
                            MediaRef(
                                path=str(path),
                                mime=_mime_for(str(path)),
                                kind="image" if _mime_for(str(path)).startswith("image") else "file",
                            )
                        )
                except Exception as exc:  # noqa: BLE001 - media is best-effort
                    _log.debug("telegram media download failed: %s", exc)
        if not text and not media:
            _log.info("telegram: DROPPED inbound chat_id=%s — message has no text or media", chat_id)
            return
        title = getattr(entity, "title", None) or getattr(entity, "first_name", None) or ""
        
        # For DMs, the sender is the same as the chat entity. For groups, we need
        # to get the actual sender from the message.
        sender_entity = getattr(event, "sender", None)
        if sender_entity is not None:
            first = getattr(sender_entity, "first_name", None) or ""
            last = getattr(sender_entity, "last_name", None) or ""
            sender_name = (first + " " + last).strip()
            if not sender_name:
                sender_name = getattr(sender_entity, "username", None) or title
            # Stable numeric Telegram user id — game identity keys off
            # this so display-name changes don't split profiles.
            sender_id = str(getattr(sender_entity, "id", "") or "")
            # Platform handle (no "@") — stored on the game profile for
            # @mention lookup. Same on every endpoint.
            sender_username = str(
                getattr(sender_entity, "username", None) or "")
        else:
            sender_name = title
            sender_id = ""
            sender_username = ""
        # Forum-topic detection (best effort): in a forum supergroup, replies
        # inside a topic carry reply_to pointing at the topic's anchor.
        thread_id = ""
        if self.threads_enabled and kind == ChatKind.GROUP:
            is_forum = bool(getattr(entity, "is_forum", False)) or (
                bool(getattr(entity, "is_channel", False))
                and bool(getattr(entity, "megagroup", False))
            )
            if is_forum:
                reply_to = getattr(message, "reply_to", None)
                if reply_to is not None and getattr(reply_to, "id", 0):
                    thread_id = str(reply_to.id)
        handler_msg = ChatMessage(
            chat=ChatRef(
                platform=self.name,
                chat_id=chat_id,
                kind=kind,
                title=str(title),
                peer=str(getattr(entity, "username", None) or chat_id) if kind == ChatKind.DM else "",
                thread_id=thread_id,
            ),
            incoming=True,
            text=text,
            sender=str(sender_name),
            sender_id=sender_id,
            sender_username=sender_username,
            mentioned=self._mentions_me(text),
            media=media,
            reply_to=str(getattr(getattr(message, "reply_to", None), "id", "") or ""),
            message_id=str(getattr(message, "id", "")),
            ts=(time.time() if not getattr(message, "date", None)
                else message.date.timestamp()),
        )
        _log.info("telegram: DELIVERING inbound from %s in %s: %.60s", handler_msg.sender, chat_id, text)
        self._deliver(handler, handler_msg)

    # ── outbound ─────────────────────────────────────────────────────────────
    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        client = self._client
        if client is None:
            return SendResult(ok=False, platform=self.name, error="not connected")
        started = time.perf_counter()
        try:
            result = self._run_on_loop(self._send_async(client, chat, text, reply_to))
            return SendResult(
                ok=True, platform=self.name, message_id=str(getattr(result, "id", "")),
                seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - ordinary failure, report as result
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def _thread_kwargs(self, chat: ChatRef) -> dict[str, Any]:
        """Forum-topic routing: replies land inside the topic, not the group."""
        if not getattr(chat, "thread_id", ""):
            return {}
        try:
            return {"message_thread_id": int(chat.thread_id)}
        except (TypeError, ValueError):
            return {}

    async def _send_async(self, client: Any, chat: ChatRef, text: str, reply_to: str) -> Any:
        # Try to use cached input entity first (most reliable for DMs)
        entity = self._input_entity_cache.get(chat.chat_id)
        if entity is None:
            # Fall back to resolution
            entity = await self._resolve(chat)
        kwargs: dict[str, Any] = {}
        if reply_to:
            kwargs["reply_to"] = int(reply_to)
        kwargs.update(self._thread_kwargs(chat))
        result = None
        for chunk in _chunk_text(text, 4096):
            try:
                result = await client.send_message(entity, chunk, **kwargs)
            except TypeError:  # older Telethon without message_thread_id
                thread = kwargs.pop("message_thread_id", None)
                if thread is not None and not kwargs.get("reply_to"):
                    kwargs["reply_to"] = thread
                result = await client.send_message(entity, chunk, **kwargs)
            # Only the first chunk carries the reply/thread reference.
            kwargs.pop("reply_to", None)
            kwargs.pop("message_thread_id", None)
        self._remember_sent(getattr(result, "id", None))  # self-chat loop guard
        return result

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "") -> SendResult:
        client = self._client
        if client is None:
            return SendResult(ok=False, platform=self.name, error="not connected")
        started = time.perf_counter()

        async def _do() -> Any:
            # Try to use cached input entity first (most reliable for DMs)
            entity = self._input_entity_cache.get(chat.chat_id)
            if entity is None:
                entity = await self._resolve(chat)
            kwargs = self._thread_kwargs(chat)
            try:
                result = await client.send_file(
                    entity, media.path, caption=caption or None, **kwargs
                )
            except TypeError:  # older Telethon without message_thread_id
                kwargs.pop("message_thread_id", None)
                result = await client.send_file(entity, media.path, caption=caption or None)
            self._remember_sent(getattr(result, "id", None))  # self-chat loop guard
            return result

        try:
            result = self._run_on_loop(_do())
            return SendResult(ok=True, platform=self.name,
                              message_id=str(getattr(result, "id", "")),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        client = self._client
        if client is None or seconds <= 0:
            _log.debug("telegram typing: skipped (client=%s, seconds=%s)", client is not None, seconds)
            return False
        if self._loop is None or self._loop.is_closed():
            # Not connected (startup/shutdown/restart) — a typing indicator is
            # cosmetic; skip quietly instead of spamming warnings per message.
            _log.debug("telegram typing: skipped, adapter loop not running")
            return False
        started = time.time()

        async def _do() -> bool:
            from telethon.tl.functions.messages import SetTypingRequest
            from telethon.tl.types import SendMessageTypingAction

            # Try to use cached input entity first (most reliable for DMs)
            entity = self._input_entity_cache.get(chat.chat_id)
            if entity is None:
                _log.debug("telegram typing: no cached entity for chat_id=%s, resolving", chat.chat_id)
                try:
                    entity = await self._resolve(chat)
                except Exception as exc:
                    _log.warning("telegram typing: resolve raised for chat_id=%s: %s", chat.chat_id, exc)
                    entity = None
            else:
                _log.debug("telegram typing: using cached entity for chat_id=%s", chat.chat_id)
            
            if entity is None:
                _log.warning("telegram typing: entity resolution failed for chat_id=%s", chat.chat_id)
                return False
            
            _log.debug("telegram typing: starting for chat_id=%s, seconds=%s, entity_type=%s", 
                      chat.chat_id, seconds, type(entity).__name__)
            
            sent_count = 0
            while time.time() - started < seconds and not self.stopped:
                try:
                    # Use the low-level SetTypingRequest directly — more
                    # reliable than send_action("typing") across Telethon
                    # versions and userbot/bot differences.
                    await client(SetTypingRequest(entity, SendMessageTypingAction()))
                    sent_count += 1
                    _log.debug("telegram typing: SetTypingRequest succeeded for chat_id=%s (count=%d)", 
                              chat.chat_id, sent_count)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("telegram typing: SetTypingRequest failed for chat_id=%s: %s, trying send_action", 
                                chat.chat_id, exc)
                    # Fallback: try send_action with string
                    try:
                        await client.send_action(entity, "typing")
                        sent_count += 1
                        _log.debug("telegram typing: send_action fallback succeeded for chat_id=%s", chat.chat_id)
                    except Exception as exc2:
                        _log.warning("telegram typing: both methods failed for chat_id=%s: %s", chat.chat_id, exc2)
                        return sent_count > 0
                sleep_for = min(5.0, max(0.1, seconds - (time.time() - started)))
                await asyncio.sleep(sleep_for)
            
            _log.info("telegram typing: finished for chat_id=%s, sent %d action(s)", chat.chat_id, sent_count)
            return sent_count > 0

        try:
            result = self._run_on_loop(_do(), timeout=seconds + 10)
            return bool(result)
        except Exception as exc:  # noqa: BLE001 - typing is cosmetic, never fatal
            _log.debug("telegram typing: _run_on_loop failed for chat_id=%s: %s", chat.chat_id, exc)
            return False

    def history(self, chat: ChatRef, limit: int = 20) -> list[ChatMessage]:
        client = self._client
        if client is None:
            return []

        async def _do() -> list[ChatMessage]:
            # Try to use cached input entity first (most reliable for DMs)
            entity = self._input_entity_cache.get(chat.chat_id)
            if entity is None:
                entity = await self._resolve(chat)
            out: list[ChatMessage] = []
            async for msg in client.iter_messages(entity, limit=limit):
                out.append(
                    ChatMessage(
                        chat=chat,
                        incoming=not bool(getattr(msg, "out", False)),
                        text=getattr(msg, "raw_text", None) or "",
                        message_id=str(getattr(msg, "id", "")),
                        ts=(msg.date.timestamp() if getattr(msg, "date", None) else time.time()),
                    )
                )
            out.reverse()
            return out

        try:
            return self._run_on_loop(_do())
        except Exception as exc:  # noqa: BLE001
            _log.debug("telegram history failed: %s", exc)
            return []


# ── Bot API adapter ────────────────────────────────────────────────────────────

_BOT_API = "https://api.telegram.org/bot{token}/{method}"
_BOT_FILE_API = "https://api.telegram.org/file/bot{token}/{path}"

_BOT_CHAT_TYPES = {
    "private": ChatKind.DM,
    "group": ChatKind.GROUP,
    "supergroup": ChatKind.GROUP,
    "channel": ChatKind.CHANNEL,
}


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split text into chunks of at most ``limit`` chars, preferring newlines."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    if rest:
        chunks.append(rest)
    return chunks


class TelegramBotAdapter(ChatAdapter):
    """Devon's own Telegram bot, over the Bot API.

    This is the bot's *own identity* (``@HerBot`` talking to people), not the
    owner's account — the companion to :class:`TelegramAdapter`, which drives
    the owner's user account over MTProto. Both can run at once: the userbot
    sees everything the owner sees, the bot talks as herself.

    Long-polling ``getUpdates`` on this adapter's own thread; no webhook
    server needed, works behind NAT. Media downloads go through
    ``getFile``. Sending splits at Telegram's 4096-char limit.
    """

    name = "telegram-bot"
    supported_kinds = (ChatKind.DM, ChatKind.GROUP)

    #: Bot API text limit per message.
    MAX_TEXT = 4096

    def __init__(
        self,
        *,
        token: str,
        chat_allow: str = "",
        media_dir: str = "data/media/telegram-bot",
        media_in_groups: bool = False,
        media_max_mb: float = 25.0,
        poll_timeout: int = 30,
        session: Any = None,
    ) -> None:
        super().__init__(media_dir=media_dir)
        if not token:
            raise ValidationError("telegram bot token is required")
        self.token = token
        self.chat_allow = {c.strip() for c in chat_allow.split(",") if c.strip()}
        self.media_in_groups = media_in_groups
        self.media_max_mb = media_max_mb
        self.poll_timeout = poll_timeout
        self._session = session
        self._offset = 0
        self._bot_id: int | None = None
        self._bot_username = ""

    # ── HTTP ────────────────────────────────────────────────────────────────
    def _sess(self) -> Any:
        if self._session is None:
            from ...core.http import HttpClient

            self._session = HttpClient()
        return self._session

    def _api(self, method: str, **params: Any) -> Any:
        url = _BOT_API.format(token=self.token, method=method)
        resp = self._sess().post_json(url, params, timeout=self.poll_timeout + 10)
        try:
            data = resp.json()
        except ValueError:
            raise ValidationError(f"telegram bot api: non-JSON reply from {method}")
        if not data.get("ok"):
            raise ValidationError(
                f"telegram bot api: {method} failed: {data.get('description', 'unknown')}"
            )
        return data["result"]

    # ── lifecycle ───────────────────────────────────────────────────────────
    def preflight(self) -> None:
        me = self._api("getMe")
        self._bot_id = me.get("id")
        self._bot_username = me.get("username", "")
        _log.info("telegram bot connected as @%s (id=%s)",
                  self._bot_username, self._bot_id)

    def run(self, handler: IncomingHandler) -> None:
        try:
            self.preflight()
        except Exception as exc:  # noqa: BLE001 - fail fast, gateway logs the skip
            raise ValueError(f"telegram-bot unavailable: {exc}") from exc
        # Drop whatever piled up while we were away; only fresh mail.
        try:
            pending = self._api("getUpdates", offset=-1, timeout=1)
            if pending:
                self._offset = max(u["update_id"] for u in pending) + 1
                _log.info("telegram-bot: skipped %d stale updates", len(pending))
        except Exception as exc:  # noqa: BLE001 - non-fatal, polling continues
            _log.debug("telegram-bot: stale-skip failed: %s", exc)
        poll_fail_delay = 5.0
        while not self._stopped.is_set():
            try:
                updates = self._api("getUpdates", offset=self._offset,
                                    timeout=self.poll_timeout,
                                    allowed_updates=["message", "edited_message",
                                                     "channel_post", "callback_query"])
            except Exception as exc:  # noqa: BLE001 - transient, back off a little
                # A persistent failure (revoked token, dead DNS) must not
                # hot-spin the poll loop every 5s — back off exponentially,
                # reset on the next success.
                _log.warning("telegram-bot poll failed (%s); retry in %.0fs",
                             exc, poll_fail_delay)
                self._stopped.wait(poll_fail_delay)
                poll_fail_delay = min(poll_fail_delay * 2.0, 300.0)
                continue
            poll_fail_delay = 5.0
            for update in updates:
                self._offset = max(self._offset, update.get("update_id", 0) + 1)
                # Callback queries (inline button taps) get their own path.
                if "callback_query" in update:
                    self._handle_callback(update["callback_query"], handler)
                    continue
                message = self._convert(update)
                if message is not None:
                    self._deliver(handler, message)

    # ── inbound ───────────────────────────────────────────────────────────
    def _convert(self, update: dict[str, Any]) -> ChatMessage | None:
        msg = update.get("message") or update.get("edited_message") \
            or update.get("channel_post")
        if not msg:
            return None
        sender = msg.get("from") or {}
        if sender.get("is_bot"):
            return None  # loop guard: never answer other bots (or our echoes)
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if not chat_id:
            return None
        if self.chat_allow and chat_id not in self.chat_allow:
            _log.info("telegram-bot: ignoring chat %s — not in allowlist", chat_id)
            return None
        kind = _BOT_CHAT_TYPES.get(chat.get("type", ""), ChatKind.DM)
        text = msg.get("text") or msg.get("caption") or ""
        mentioned = False
        if kind != ChatKind.DM and self._bot_username:
            for ent in msg.get("entities") or msg.get("caption_entities") or []:
                if ent.get("type") == "mention":
                    piece = text[ent["offset"]:ent["offset"] + ent["length"]]
                    if piece.lower() == f"@{self._bot_username.lower()}":
                        mentioned = True
                        break
        media = self._inbound_media(msg, kind)
        reply_to = ""
        replied = msg.get("reply_to_message") or {}
        if replied.get("message_id"):
            reply_to = str(replied["message_id"])
        who = sender.get("username") or sender.get("first_name") or str(sender.get("id", ""))
        # Stable numeric Telegram user id — game identity keys off this
        # so username/display-name changes don't split profiles.
        who_id = str(sender.get("id", "") or "")
        who_username = str(sender.get("username") or "")
        return ChatMessage(
            chat=ChatRef(platform=self.name, chat_id=chat_id, kind=kind,
                         title=chat.get("title", "") or who, peer=who),
            incoming=True,
            text=text,
            sender=who,
            sender_id=who_id,
            sender_username=who_username,
            media=media,
            reply_to=reply_to,
            mentioned=mentioned,
            ts=float(msg.get("date", time.time())),
            message_id=str(msg.get("message_id", "")),
            meta={"update_id": update.get("update_id")},
        )

    # ── inline buttons ──────────────────────────────────────────────────
    def _sign_callback(self, data: str) -> str:
        """HMAC-sign callback data to prevent spoofing."""
        sig = hmac.new(self.token.encode(), data.encode(),
                       hashlib.sha256).hexdigest()[:16]
        return f"{data}|{sig}"

    def _verify_callback(self, signed: str) -> str | None:
        """Verify signature, return the original data or None."""
        if "|" not in signed:
            return None
        data, sig = signed.rsplit("|", 1)
        expected = hmac.new(self.token.encode(), data.encode(),
                            hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(sig, expected):
            return None
        return data

    def _handle_callback(self, query: dict[str, Any],
                         handler: IncomingHandler) -> None:
        """Process an inline button tap as a synthetic command message."""
        query_id = query.get("id", "")
        sender = query.get("from") or {}
        if sender.get("is_bot"):
            self._answer_callback(query_id)
            return
        # Owner gate: only the owner can trigger button actions.
        msg = query.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if self.chat_allow and chat_id not in self.chat_allow:
            _log.info("telegram-bot: ignoring callback from %s — not in allowlist",
                      chat_id)
            self._answer_callback(query_id, text="Not authorized")
            return
        signed = query.get("data", "")
        data = self._verify_callback(signed)
        if data is None:
            _log.warning("telegram-bot: ignoring callback with bad signature")
            self._answer_callback(query_id, text="Invalid button")
            return
        self._answer_callback(query_id)
        # Deliver as a synthetic "/" command so the normal command
        # pipeline (owner gating, parsing, dispatch) handles it.
        kind = _BOT_CHAT_TYPES.get(chat.get("type", ""), ChatKind.DM)
        who = sender.get("username") or sender.get("first_name") or str(sender.get("id", ""))
        message = ChatMessage(
            chat=ChatRef(platform=self.name, chat_id=chat_id, kind=kind,
                         title=chat.get("title", "") or who, peer=who),
            incoming=True,
            text=data if data.startswith("/") else f"/{data}",
            sender=who,
            sender_id=str(sender.get("id", "") or ""),
            sender_username=str(sender.get("username") or ""),
            media=[],
            reply_to="",
            mentioned=False,
            ts=time.time(),
            message_id=f"cb_{query_id}",
            meta={"callback_query": True},
        )
        self._deliver(handler, message)

    def _answer_callback(self, query_id: str, text: str = "") -> None:
        """Dismiss the button's loading spinner, optionally with a toast."""
        try:
            params: dict[str, Any] = {"callback_query_id": query_id}
            if text:
                params["text"] = text[:200]
            self._api("answerCallbackQuery", **params)
        except Exception as exc:  # noqa: BLE001 - non-fatal
            _log.debug("telegram-bot: answerCallbackQuery failed: %s", exc)

    def _inbound_media(self, msg: dict[str, Any], kind: str) -> list[MediaRef]:
        """Download the largest attached file, when the gates allow it."""
        file_id, fkind, fname = "", "", ""
        if msg.get("photo"):
            biggest = max(msg["photo"], key=lambda p: p.get("file_size", 0))
            file_id, fkind, fname = biggest["file_id"], "image", "photo.jpg"
        elif msg.get("document"):
            d = msg["document"]
            file_id, fkind = d["file_id"], "document"
            fname = d.get("file_name", "file")
        elif msg.get("video"):
            file_id, fkind, fname = msg["video"]["file_id"], "video", "video.mp4"
        elif msg.get("voice"):
            file_id, fkind, fname = msg["voice"]["file_id"], "audio", "voice.ogg"
        elif msg.get("audio"):
            file_id, fkind, fname = msg["audio"]["file_id"], "audio", "audio.mp3"
        elif msg.get("video_note"):
            file_id, fkind, fname = msg["video_note"]["file_id"], "video", "note.mp4"
        if not file_id:
            return []
        size = 0
        try:
            info = self._api("getFile", file_id=file_id)
            size = int(info.get("file_size", 0))
            path = info.get("file_path", "")
        except Exception as exc:  # noqa: BLE001 - media is best-effort
            _log.debug("telegram-bot getFile failed: %s", exc)
            return []
        if not media_download_allowed(kind, size, self.media_in_groups, self.media_max_mb):
            return []
        if not path:
            return []
        dest = Path(self.media_dir) / f"{file_id[:16]}_{fname}"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = _BOT_FILE_API.format(token=self.token, path=path)
            resp = self._sess().get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001
            _log.debug("telegram-bot media download failed: %s", exc)
            return []
        return [MediaRef(path=str(dest), kind=fkind, name=fname)]

    # ── outbound ──────────────────────────────────────────────────────────
    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None) -> SendResult:
        started = time.perf_counter()
        last_id = ""
        try:
            params_base: dict[str, Any] = {"chat_id": int(chat.chat_id)}
            if chat.thread_id:
                try:
                    params_base["message_thread_id"] = int(chat.thread_id)
                except (TypeError, ValueError) as e:
                    _log.debug("dropping bad thread_id %r: %s", chat.thread_id, e)
            if reply_to:
                try:
                    params_base["reply_parameters"] = {"message_id": int(reply_to)}
                except (TypeError, ValueError) as e:
                    _log.debug("dropping bad reply_to %r: %s", reply_to, e)
            # Inline keyboard: [[(label, callback_data), ...], ...]
            # Callback data is HMAC-signed to prevent spoofing.
            if buttons:
                keyboard = []
                for row in buttons:
                    krow = []
                    for label, data in row:
                        krow.append({"text": label,
                                     "callback_data": self._sign_callback(data)})
                    keyboard.append(krow)
                params_base["reply_markup"] = {"inline_keyboard": keyboard}
            for chunk in _chunk_text(text, self.MAX_TEXT):
                result = self._api("sendMessage", text=chunk, **params_base)
                last_id = str(result.get("message_id", ""))
                # Only the first chunk carries the reply reference.
                params_base.pop("reply_parameters", None)
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name, message_id=last_id,
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - ordinary failure, report as result
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "") -> SendResult:
        started = time.perf_counter()
        path = Path(media.path)
        if not path.is_file():
            return SendResult(ok=False, platform=self.name,
                              error=f"media file not found: {media.path}")
        method = {
            "image": "sendPhoto",
            "video": "sendVideo",
            "audio": "sendVoice" if path.suffix.lower() == ".ogg" else "sendAudio",
        }.get(media.kind, "sendDocument")
        field = {"sendPhoto": "photo", "sendVideo": "video",
                 "sendVoice": "voice", "sendAudio": "audio"}.get(method, "document")
        try:
            url = _BOT_API.format(token=self.token, method=method)
            data: dict[str, Any] = {"chat_id": chat.chat_id}
            if caption:
                data["caption"] = caption[:1024]
            with open(path, "rb") as fh:
                files = {field: (path.name, fh, _mime_for(str(path)))}
                resp = self._sess().post(url, data=data, files=files, timeout=120)
            result = resp.json()
            if not result.get("ok"):
                raise ValidationError(result.get("description", "send failed"))
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name,
                              message_id=str(result["result"].get("message_id", "")),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        # The Bot API clears a typing indicator after ~5s server-side, so a
        # requested duration longer than that is held by re-firing
        # sendChatAction — the indicator stays up for the whole requested
        # window, proportional to the outgoing message length.
        if seconds <= 0:
            return False
        deadline = time.time() + float(seconds)
        sent = False
        try:
            while time.time() < deadline:
                self._api("sendChatAction", chat_id=int(chat.chat_id), action="typing")
                sent = True
                time.sleep(min(4.0, max(0.1, deadline - time.time())))
            return sent
        except Exception:  # noqa: BLE001 - best-effort
            return sent

    def health(self) -> dict[str, Any]:
        info = super().health()
        info["bot_username"] = self._bot_username
        return info
