"""The chat adapter protocol: bidirectional messaging across platforms.

This is the *companion* face of the social layer. ``social/`` (its parent
package) is about *publishing* — one post, many platforms, official APIs.
``social/chat/`` is about *conversation* — one person, many surfaces,
simultaneously: a Telegram userbot, a Discord bot, and a WhatsApp bridge all
delivering inbound messages to one brain and receiving its replies.

Conventions every adapter obeys:

* ``run(handler)`` is blocking and runs on the adapter's own daemon thread.
  Adapters that are inherently asyncio (Telethon, discord.py) own their event
  loop inside ``run`` — the rest of the system stays thread-based.
* ``send`` is thread-safe: the runtime and the autonomy agent may call it from
  different threads; the gateway adds a per-chat lock for ordering.
* Adapters never store credentials. They take them in the constructor from
  settings/env at boot and hold nothing that a backup would leak.
* Everything returns :class:`SendResult`, never raises, for ordinary
  failures (platform down, chat gone, rate limited).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.logging_setup import get_logger

__all__ = [
    "ChatKind",
    "ChatRef",
    "ChatMessage",
    "MediaRef",
    "SendResult",
    "ChatAdapter",
    "IncomingHandler",
    "is_owner_chat",
]

_log = get_logger(__name__)

IncomingHandler = Callable[["ChatMessage"], None]


def is_owner_chat(chat: Any, *, owner_chats: Any = (), db_is_owner: bool = False) -> bool:
    """One owner test for every surface — the single source of truth.

    Lives at L4 (``social.chat``) so both the chat gateway (inbound
    rate-limit exemption) and the partner runtime (gating, romantic
    context, command authorization) can use it without a layering
    violation. ``nomorals/partner/gating.py`` re-exports it next to
    :func:`classify_chat`.

    ``owner_chats`` is the configured ``partner.owner_chats`` key set;
    ``db_is_owner`` is the chat registry row's stored flag. The
    local/web console (``*:console``) is always the owner's own terminal.
    """
    key = str(getattr(chat, "key", "") or "")
    if key and key in set(owner_chats or ()):
        return True
    if db_is_owner:
        return True
    return key.endswith(":console")


class ChatKind:
    DM = "dm"
    GROUP = "group"
    CHANNEL = "channel"

    ALL = (DM, GROUP, CHANNEL)


@dataclass(frozen=True)
class ChatRef:
    """One conversation surface: a DM, a group, or a channel on a platform.

    ``thread_id`` refines a surface to a conversation *inside* it (Telegram
    forum topics, Discord thread channels). Threads get their own rate
    windows, memory context and reply routing — a flood in one topic can't
    starve the rest of the group, and a reply lands back in the thread.
    """

    platform: str
    chat_id: str
    kind: str = ChatKind.DM
    title: str = ""
    peer: str = ""  # the other human, when the chat is a DM
    thread_id: str = ""

    def __post_init__(self) -> None:
        # Same None-coercion as ChatMessage: adapters build refs from raw
        # payloads where fields can be missing. Frozen dataclass, so the
        # coercion goes through object.__setattr__.
        if self.platform is None:
            object.__setattr__(self, "platform", "")
        if self.chat_id is None:
            object.__setattr__(self, "chat_id", "")
        if self.kind is None:
            object.__setattr__(self, "kind", ChatKind.DM)
        if self.title is None:
            object.__setattr__(self, "title", "")
        if self.peer is None:
            object.__setattr__(self, "peer", "")
        if self.thread_id is None:
            object.__setattr__(self, "thread_id", "")

    @property
    def key(self) -> str:
        base = f"{self.platform}:{self.chat_id}"
        return f"{base}:{self.thread_id}" if self.thread_id else base

    @classmethod
    def parse(cls, key: str, *, kind: str = ChatKind.DM) -> "ChatRef":
        platform, _, chat_id = key.partition(":")
        if not platform or not chat_id:
            raise ValueError(f"bad chat key: {key!r} (want 'platform:chat_id')")
        return cls(platform=platform, chat_id=chat_id, kind=kind)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.key


@dataclass
class MediaRef:
    """A media attachment, already on local disk."""

    path: str
    mime: str = ""
    kind: str = "image"  # image | video | audio | document
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "mime": self.mime, "kind": self.kind, "name": self.name}


@dataclass
class ChatMessage:
    """One normalized message, inbound or outbound."""

    chat: ChatRef
    incoming: bool
    text: str = ""
    sender: str = ""  # display name or id of the other side (inbound)
    media: list[MediaRef] = field(default_factory=list)
    reply_to: str = ""
    mentioned: bool = False  # the platform tagged/mentioned our account in this message
    ts: float = field(default_factory=time.time)
    message_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Adapters build these from raw platform payloads where any field
        # can arrive as None (missing text, anonymous sender, …). Coerce
        # once here so the dozen downstream ``.strip()`` call sites never
        # see a None.
        if self.text is None:
            self.text = ""
        if self.sender is None:
            self.sender = ""
        if self.reply_to is None:
            self.reply_to = ""
        if self.message_id is None:
            self.message_id = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat": self.chat.key,
            "incoming": self.incoming,
            "text": self.text,
            "sender": self.sender,
            "media": [m.to_dict() for m in self.media],
            "reply_to": self.reply_to,
            "mentioned": self.mentioned,
            "ts": self.ts,
            "message_id": self.message_id,
            "meta": self.meta,
        }

    @property
    def has_media(self) -> bool:
        return bool(self.media)

    @property
    def first_media(self) -> MediaRef | None:
        return self.media[0] if self.media else None


@dataclass
class SendResult:
    ok: bool
    platform: str
    message_id: str = ""
    error: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "platform": self.platform, "message_id": self.message_id,
                "error": self.error, "seconds": round(self.seconds, 3)}


class ChatAdapter(ABC):
    """A connected platform. Subclasses own their network details entirely."""

    name: str = "unknown"
    #: Kinds of chats this adapter can deliver/receive.
    supported_kinds: tuple[str, ...] = (ChatKind.DM, ChatKind.GROUP, ChatKind.CHANNEL)

    def __init__(self, *, media_dir: str = "data/media") -> None:
        self.media_dir = media_dir
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self.stats = {"received": 0, "sent": 0, "send_errors": 0, "started_at": 0.0}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self, handler: IncomingHandler) -> None:  # pragma: no cover - abstract
        """Block until stopped, delivering inbound messages via ``handler``."""
        raise NotImplementedError

    def preflight(self) -> None:
        """One-time interactive setup, run on the main thread BEFORE any
        adapter thread starts (e.g. Telethon's first-run login).

        The local console reads stdin continuously; an interactive login
        prompt on an adapter thread would race it for the keyboard and be
        unreachable. Default: nothing to do.
        """

    def start(self, handler: IncomingHandler) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._run_guarded, args=(handler,), name=f"chat-{self.name}", daemon=True
        )
        self._thread.start()
        return True

    def _run_guarded(self, handler: IncomingHandler) -> None:
        self.stats["started_at"] = time.time()
        try:
            self.run(handler)
        except Exception as exc:  # noqa: BLE001 - an adapter dying must not kill the process
            _log.exception("adapter %s crashed: %s", self.name, exc)
        finally:
            self._stopped.set()

    def stop(self) -> None:
        self._stopped.set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def _deliver(self, handler: IncomingHandler, message: ChatMessage) -> None:
        """Deliver one inbound message, counting it and isolating handler errors."""
        self.stats["received"] += 1
        try:
            handler(message)
        except Exception as exc:  # noqa: BLE001 - one bad message must not kill the feed
            _log.exception("inbound handler failed for %s: %s", message.chat.key, exc)

    # ── sending ──────────────────────────────────────────────────────────────
    @abstractmethod
    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        """Send one text message. Returns a result, never raises for ordinary failure."""

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "") -> SendResult:
        return SendResult(ok=False, platform=self.name, error="media not supported")

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        """Show a typing indicator. Best-effort; adapters without one return False."""
        return False

    def history(self, chat: ChatRef, limit: int = 20) -> list[ChatMessage]:
        """Recent inbound history, oldest first. Empty when unavailable."""
        return []

    # ── health ───────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "running": self._thread is not None and self._thread.is_alive(),
            **self.stats,
        }
