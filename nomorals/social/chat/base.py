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
    "location_shape_error",
    "contact_shape_error",
    "search_shape_error",
    "STATUS_AUDIENCE_MODES",
    "status_audience_shape_error",
    "label_edit_shape_error",
    "label_assoc_shape_error",
    "catalog_product_shape_error",
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
    A bare id in ``owner_chats`` (owner wrote ``5478650254`` without a
    platform prefix) matches any platform whose chat id equals it —
    the owner's Telegram user id is the same number on
    ``telegram``/``telegram-bot``/``telegram-personal`` in DMs.
    """
    key = str(getattr(chat, "key", "") or "")
    if key and key in set(owner_chats or ()):
        return True
    chat_id = str(getattr(chat, "chat_id", "") or "")
    if chat_id and chat_id in set(owner_chats or ()):
        return True
    if db_is_owner:
        return True
    return key.endswith(":console")


def location_shape_error(latitude: Any, longitude: Any) -> str:
    """Why these coordinates cannot be a location pin, or ""
    when they can (parity bundle #9 — the one shape rule every
    surface checks BEFORE traffic, mirrored in the bridges).

    Coordinates are degrees: latitude in [-90, 90], longitude
    in [-180, 180].  NaN and infinities are not places."""
    import math

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return ("a location needs numeric latitude and "
                "longitude (degrees)")
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return "a location's coordinates must be finite numbers"
    if not -90.0 <= lat <= 90.0:
        return "latitude must be between -90 and 90"
    if not -180.0 <= lon <= 180.0:
        return "longitude must be between -180 and 180"
    return ""


def contact_shape_error(phone: Any, name: Any) -> str:
    """Why this (phone, name) pair cannot be a contact card,
    or "" when it can (parity bundle #9).  A card needs a
    display name and a dialable number: 7–15 digits once the
    punctuation a human writes (+, spaces, dashes) is stripped
    — a shorter or longer digit string is not a phone number
    anyone can be reached at, and inventing digits to fill the
    shape is never an option."""
    if not str(name or "").strip():
        return "a contact card needs the person's name"
    digits = "".join(
        ch for ch in str(phone or "") if ch.isdigit())
    if not digits:
        return "a contact card needs a phone number"
    if not 7 <= len(digits) <= 15:
        return ("a phone number is 7–15 digits — that one has "
                + str(len(digits)))
    return ""


def search_shape_error(query: Any) -> str:
    """Why this text cannot be a message search, or "" when
    it can (parity bundle #11 — the one shape rule every
    surface checks BEFORE traffic, mirrored in the bridges).

    A search needs actual words: an empty query would
    "match" everything, which is not a search result but a
    history dump wearing search's clothes."""
    if not str(query or "").strip():
        return "a search needs the words to look for"
    return ""


#: The status-audience modes every surface speaks (parity
#: bundle #13): the account's status posts go to all its
#: contacts, to its contacts except a named list, or only
#: to a named list.  The wire words (WhatsApp's contacts /
#: blacklist / whitelist) are translated inside the bridges
#: and never leak above them.
STATUS_AUDIENCE_MODES = ("contacts", "except", "only")


def status_audience_shape_error(mode: Any,
                                audience: Any) -> str:
    """Why this (mode, list) pair cannot be a status
    audience, or "" when it can (parity bundle #13 — the
    one shape rule every surface checks BEFORE traffic,
    mirrored word for word in both bridges).

    ``contacts`` is everyone, so it takes no list; the
    other two modes ARE their list — an ``except`` with
    nobody named excludes nobody and an ``only`` with
    nobody named shares with nobody, and neither is a
    set anyone meant to make.  Entries must be person
    JIDs (``…@s.whatsapp.net``): a group or a channel in
    an audience list is a category error, not a wider
    audience."""
    mode = str(mode or "").strip().lower()
    if mode not in STATUS_AUDIENCE_MODES:
        return "a status audience mode is contacts, " \
               "except or only"
    entries = [str(j).strip() for j in (audience or [])
               if str(j).strip()]
    if mode == "contacts" and entries:
        return "the contacts audience takes no list — " \
               "it is everyone"
    if mode != "contacts" and not entries:
        return ("an " + mode + " audience needs at least "
                "one person")
    for jid in entries:
        if not jid.endswith("@s.whatsapp.net"):
            return ("a status audience list holds people, "
                    "not " + jid)
#: Bot API ceilings for a bot's own profile fields (parity
#: bundle #18): the display name is capped at 64 characters,
#: the long description at 512, the short description at 120.
#: An EMPTY value is legal for all three — it clears the
#: custom value and the field falls back to its default.
BOT_PROFILE_LIMITS = {"name": 64, "description": 512,
                      "short_description": 120}


def bot_profile_shape_error(field: Any, value: Any) -> str:
    """Why this value cannot be written to a bot's own
    profile field, or "" when it can (parity bundle #18 —
    the shape rule checked BEFORE traffic).  ``field`` is
    one of name / description / short_description; each has
    the Bot API's own length ceiling, and a value over it is
    refused in words instead of being truncated silently —
    a profile is identity, never something to clip."""
    field = str(field or "")
    limit = BOT_PROFILE_LIMITS.get(field)
    if limit is None:
        return (f"“{field}” is not a bot profile field — "
                f"use name, description or short")
    text = str(value or "")
    if len(text) > limit:
        return (f"a bot's {field} is at most {limit} "
                f"characters — that one has {len(text)}")
    return ""


def invoice_shape_error(title: Any, description: Any,
                        amount: Any) -> str:
    """Why this (title, description, amount) cannot be a
    Telegram Stars invoice, or "" when it can (parity
    bundle #18 — checked BEFORE traffic).  The Bot API's
    own bounds: a title of 1–32 characters, a description
    of 1–255, and for currency XTR the price is a positive
    WHOLE number of Stars — Stars have no fractional unit,
    so a float or a zero/negative amount is not a price."""
    title = str(title or "").strip()
    if not title:
        return "an invoice needs a title — what is being sold"
    if len(title) > 32:
        return (f"an invoice title is at most 32 characters "
                f"— that one has {len(title)}")
    description = str(description or "").strip()
    if not description:
        return ("an invoice needs a description — what the "
                "buyer gets")
    if len(description) > 255:
        return (f"an invoice description is at most 255 "
                f"characters — that one has {len(description)}")
    if isinstance(amount, bool) or not isinstance(amount, int):
        return ("a Stars price is a whole number of Stars "
                "(⭐ XTR has no fractional unit)")
    if amount <= 0:
        return "a Stars price must be at least 1 ⭐"
    return ""


def label_edit_shape_error(name: Any = "", color: Any = None,
                           *, deleted: bool = False,
                           label_id: Any = "") -> str:
    """Why this label-definition write cannot go out, or ""
    when it can (parity bundle #17 — the one shape rule
    every surface checks BEFORE traffic, mirrored word for
    word in both bridges).

    A label definition is a whole record — name, color and
    the deleted flag travel together in one app-state
    mutation — so a create or rename needs the name it will
    carry, a delete needs the id of the label it tombstones,
    and a color, when one is carried, is an index into
    WhatsApp's own palette of twenty."""
    if deleted:
        if not str(label_id or "").strip():
            return "deleting a label needs the label's id"
        return ""
    if not str(name or "").strip():
        return "a label needs a name"
    if color is not None:
        try:
            value = int(color)
        except (TypeError, ValueError):
            return ("label colors run 0 to 19 — WhatsApp's "
                    "twenty label colors")
        if value != color or not 0 <= value <= 19:
            return ("label colors run 0 to 19 — WhatsApp's "
                    "twenty label colors")
    return ""


def label_assoc_shape_error(chat: Any, label_id: Any,
                            message_id: Any = "", *,
                            need_message: bool = False
                            ) -> str:
    """Why this label association cannot go out, or "" when
    it can (parity bundle #17, same before-traffic law):
    labeling acts on a chat and a label, and the message
    form names the message too."""
    if not str(chat or "").strip():
        return "labeling needs the chat"
    if not str(label_id or "").strip():
        return "labeling needs the label's id"
    if need_message and not str(message_id or "").strip():
        return "labeling a message needs the message's id"
    return ""


def catalog_product_shape_error(name: Any = "",
                                price: Any = None,
                                currency: Any = "") -> str:
    """Why this catalog product write cannot go out, or ""
    when it can (parity bundle #17, same before-traffic
    law, mirrored in the Baileys bridge — the one engine
    whose library carries catalog writes).

    A product needs its name, a price that is a number at
    zero or above (a free product is a real shape; a
    negative or non-numeric price is not a price), and its
    currency as the three-letter code the catalog itself
    speaks."""
    if not str(name or "").strip():
        return "a product needs a name"
    try:
        value = float(price)
    except (TypeError, ValueError):
        return "a product price must be a number, 0 or more"
    if value != value or value < 0:  # NaN or negative
        return "a product price must be a number, 0 or more"
    code = str(currency or "").strip().upper()
    if len(code) != 3 or not code.isalpha():
        return ("a product needs its currency as a "
                "three-letter code, e.g. NGN")
    return ""


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
    kind: str = "image"  # image | video | audio | voice | document |
    #                      video_note (a round/circular video note)
    name: str = ""
    voice_note: bool = False  # True → send as a Telegram voice bubble
    # True → send as VIEW-ONCE media (parity bundle #10): the
    # recipient gets one look, then it's gone. A WhatsApp form
    # (photos/videos); adapters whose platform has no view-once
    # shape refuse honestly instead of sending a permanent copy.
    view_once: bool = False
    # True → this attachment IS a Telegram ``voice`` message: a spoken
    # voice note, never a song clip. Voice notes must route to STT;
    # music ID only ever runs on an explicit /sham request. Distinct
    # from ``voice_note`` (outbound send-as-voice-bubble flag) — this
    # one records what the attachment WAS on arrival.
    is_voice: bool = False
    # Platform file handle (a Telegram file_id): set on inbound media
    # the platform already hosts, and on outbound refs to RE-SEND by
    # handle instead of uploading bytes again.
    file_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "mime": self.mime, "kind": self.kind,
                "name": self.name, "voice_note": self.voice_note,
                "view_once": self.view_once,
                "is_voice": self.is_voice, "file_id": self.file_id}


@dataclass
class ChatMessage:
    """One normalized message, inbound or outbound."""

    chat: ChatRef
    incoming: bool
    text: str = ""
    sender: str = ""  # display name or id of the other side (inbound)
    sender_id: str = ""  # stable platform user id (numeric on Telegram);
    # when present, game identity keys off this instead of the display
    # name, so renames don't split profiles.
    sender_username: str = ""  # platform handle without "@" (Telegram
    # username); stored on the game profile and used for @mention lookup
    # (gifting, pvp challenges). Same value on every endpoint.
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
        if self.sender_id is None:
            self.sender_id = ""
        else:
            self.sender_id = str(self.sender_id)
        if self.sender_username is None:
            self.sender_username = ""
        else:
            self.sender_username = str(self.sender_username).lstrip("@")
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
            "sender_id": self.sender_id,
            "sender_username": self.sender_username,
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
    #: True when durability was requested but no outbox was available, so
    #: the send went direct without a receipt. Degradation is reported
    #: ON the result — never only in a log line the caller never sees.
    delivery_degraded: bool = False
    #: The platform's file handle for media just sent (a Telegram
    #: file_id). The expression library caches it per item so the
    #: next send of the same media goes by handle, no re-upload.
    media_file_id: str = ""
    #: The view count the platform reported for the message just
    #: sent, when it reported one at all (a Telegram channel post
    #: carries views; most sends on most platforms report none).
    #: None means "not returned" — never a fabricated 0.  The
    #: durable series is the channel drive's views re-read; this
    #: is only the send-time seed.
    views: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "platform": self.platform, "message_id": self.message_id,
                "error": self.error, "seconds": round(self.seconds, 3),
                "delivery_degraded": self.delivery_degraded,
                "media_file_id": self.media_file_id,
                "views": self.views}


class ChatAdapter(ABC):
    """A connected platform. Subclasses own their network details entirely."""

    name: str = "unknown"
    #: Kinds of chats this adapter can deliver/receive.
    supported_kinds: tuple[str, ...] = (ChatKind.DM, ChatKind.GROUP, ChatKind.CHANNEL)

    #: Restart policy for a crashed adapter thread (see ``_run_guarded``).
    #: A platform that dies must come back on its own — on a phone nobody is
    #: watching the logs. Backoff keeps a persistently broken config (revoked
    #: token, dead network) from hot-spinning the CPU.
    RESTART_INITIAL_DELAY_S = 5.0
    RESTART_MAX_DELAY_S = 300.0
    #: Consecutive crashes before the adapter gives up and stays down. A run
    #: that survives ``RESTART_RESET_AFTER_S`` resets the counter, so only a
    #: genuinely broken adapter exhausts it.
    RESTART_MAX_CRASHES = 10
    RESTART_RESET_AFTER_S = 60.0

    #: Media cache bounds (see ``_prune_media_dir``). Inbound photos/voice
    #: notes land on disk and were never cleaned up — on a phone that is a
    #: slow disk-exhaustion bug. Age is the primary bound; the size cap is
    #: the backstop for a flood of fresh files.
    MEDIA_MAX_AGE_DAYS = 7.0
    MEDIA_MAX_TOTAL_MB = 512.0

    def __init__(self, *, media_dir: str = "data/media") -> None:
        self.media_dir = media_dir
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self.stats = {"received": 0, "sent": 0, "send_errors": 0, "started_at": 0.0}
        #: Consecutive crashes of the adapter thread; reset by a healthy run
        #: or a manual ``start()``.
        self._crash_count = 0
        self._last_crash = ""
        self._last_crash_at = 0.0
        self._gave_up = False

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
        # A manual (re)start is a fresh lease: the operator may have fixed
        # whatever was crashing it.
        self._crash_count = 0
        self._gave_up = False
        self._thread = threading.Thread(
            target=self._run_guarded, args=(handler,), name=f"chat-{self.name}", daemon=True
        )
        self._thread.start()
        return True

    def _note_crash(self, exc: BaseException) -> None:
        self._crash_count += 1
        self._last_crash = f"{type(exc).__name__}: {exc}"
        self._last_crash_at = time.time()

    def _run_guarded(self, handler: IncomingHandler) -> None:
        self.stats["started_at"] = time.time()
        # Inbound media accumulates on disk forever without this; prune once
        # per thread start (not per restart — a crash loop must not re-walk
        # a huge directory on every attempt).
        self._prune_media_dir()
        delay = self.RESTART_INITIAL_DELAY_S
        while True:
            run_started = time.time()
            try:
                self.run(handler)
            except Exception as exc:  # noqa: BLE001 - a crashed adapter restarts, it never kills the process
                if self.stopped:
                    return
                # A run that lived long enough was healthy; only consecutive
                # quick deaths count toward giving up.
                if time.time() - run_started >= self.RESTART_RESET_AFTER_S:
                    self._crash_count = 0
                self._note_crash(exc)
                if self._crash_count >= self.RESTART_MAX_CRASHES:
                    _log.error(
                        "adapter %s crashed %d times in a row (%s) — giving up; "
                        "fix the config then restart it",
                        self.name, self._crash_count, self._last_crash,
                    )
                    self._gave_up = True
                    self._stopped.set()
                    return
                _log.warning(
                    "adapter %s crashed (%s) — restarting in %.0fs (crash %d/%d)",
                    self.name, self._last_crash, delay,
                    self._crash_count, self.RESTART_MAX_CRASHES,
                )
                if self._stopped.wait(delay):
                    return
                delay = min(delay * 2.0, self.RESTART_MAX_DELAY_S)
                self._stopped.clear()
                continue
            # run() returned cleanly (not via stop): mark stopped and exit.
            self._stopped.set()
            return

    def _prune_media_dir(
        self,
        *,
        max_age_days: float | None = None,
        max_total_mb: float | None = None,
    ) -> int:
        """Delete stale inbound media so the disk can't fill up silently.

        Removes files older than ``max_age_days`` (default
        :attr:`MEDIA_MAX_AGE_DAYS`), then — if the directory is still over
        ``max_total_mb`` — evicts oldest-first until under the cap. Hidden
        files (``.greeted.json`` and friends) are adapter state, not media,
        and are never touched. Returns the number of files removed; never
        raises.
        """
        import os

        if max_age_days is None:
            max_age_days = self.MEDIA_MAX_AGE_DAYS
        if max_total_mb is None:
            max_total_mb = self.MEDIA_MAX_TOTAL_MB
        removed = 0
        try:
            root = self.media_dir
            if not root or not os.path.isdir(root):
                return 0
            now = time.time()
            cutoff = now - max_age_days * 86400.0
            entries: list[tuple[float, int, str]] = []  # (mtime, size, path)
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    if fname.startswith("."):
                        continue
                    path = os.path.join(dirpath, fname)
                    try:
                        st = os.stat(path)
                    except OSError as exc:
                        _log.debug("media prune: stat failed for %s: %s", path, exc)
                        continue
                    if st.st_mtime < cutoff:
                        try:
                            os.unlink(path)
                            removed += 1
                        except OSError as exc:
                            _log.debug("media prune: unlink failed for %s: %s", path, exc)
                    else:
                        entries.append((st.st_mtime, st.st_size, path))
            cap_bytes = max_total_mb * 1024 * 1024
            if cap_bytes > 0:
                total = sum(size for _, size, _ in entries)
                if total > cap_bytes:
                    entries.sort(key=lambda e: e[0])  # oldest first
                    for _mtime, _size, path in entries:
                        if total <= cap_bytes:
                            break
                        try:
                            os.unlink(path)
                            removed += 1
                            total -= _size
                        except OSError as exc:
                            _log.debug("media prune: unlink failed for %s: %s", path, exc)
            if removed:
                _log.info("adapter %s: pruned %d stale media file(s) from %s",
                          self.name, removed, root)
        except Exception as exc:  # noqa: BLE001 - pruning must never break the adapter
            _log.debug("adapter %s: media prune failed: %s", self.name, exc)
        return removed

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
    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             reply_keyboard: list[list[str]] | None = None,
             parse_mode: str = "",
             fresh: bool = False) -> SendResult:
        """Send one text message. Returns a result, never raises for ordinary failure.

        ``buttons`` is ``[[(label, callback_data), ...], ...]`` — an inline
        keyboard. Adapters whose platform has no button concept ignore it;
        TelegramBotAdapter renders it as an inline keyboard.

        ``reply_keyboard`` is ``[[text, ...], ...]`` — a persistent
        bottom reply keyboard (tapping a button sends its text as a
        message). Only the Telegram adapters render it; every other
        adapter ignores it. When both ``buttons`` and ``reply_keyboard``
        are passed, ``reply_keyboard`` wins (a message carries one
        keyboard). An EMPTY ``reply_keyboard`` (``[]``) is the remove
        verb: the Telegram adapters hide the persistent keyboard
        (Bot API ``remove_keyboard`` / MTProto ``ReplyKeyboardHide``);
        every other adapter ignores it, like a keyboard it cannot
        render.

        ``parse_mode`` (e.g. "HTML") enables rich formatting — opt-in per
        message so plain-text messages are never accidentally parsed.

        ``fresh`` marks the first dispatch attempt of a reply composed
        for a live inbound turn (the gateway sets it on that attempt
        only; retry layers re-enter without it). Adapters with a
        send-once guard (WhatsApp) release the guard for a fresh
        solicited reply — a repeated question must always be
        answerable — while unmarked repeats stay guarded. Adapters
        without such a guard ignore it.
        """

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "",
                   parse_mode: str = "") -> SendResult:
        # ``parse_mode`` (Formatting F7) is the caption's parse mode,
        # mirroring send(): adapters whose platform formats captions
        # (the Telegram family) apply it — with the plain-text retry
        # on an entity rejection; every other adapter ignores it.
        return SendResult(ok=False, platform=self.name, error="media not supported")

    def edit_media(self, chat: ChatRef, message_id: str, media: MediaRef,
                   *, caption: str = "") -> SendResult:
        """Replace the media (and caption) of a previously sent message.

        Used by the live agent window to update one message in place instead
        of spamming new ones. Adapters without in-place media editing return
        ``ok=False`` and callers fall back to throttled new messages.
        """
        return SendResult(ok=False, platform=self.name,
                          error="edit_media not supported")

    # ── reactions (DESIGN §22) ──────────────────────────────────────────
    def react(self, chat: ChatRef, message_id: str, emoji: str) -> SendResult:
        """Place a reaction on a message. The base adapter reports the
        truth — this platform path carries no reaction verb — and each
        adapter that CAN place one overrides. Never a fake success."""
        return SendResult(ok=False, platform=self.name,
                          error="reactions not supported on this adapter")

    def read_reactions(self, chat: ChatRef,
                       message_id: str) -> dict[str, int]:
        """``{emoji: count}`` currently on a message, where the platform
        exposes them (Discord); ``{}`` where it does not."""
        return {}

    # ── locations & contact cards (parity bundle #9) ─────────
    def send_location(self, chat: ChatRef, latitude: float,
                      longitude: float) -> SendResult:
        """Drop a static location pin in a chat.  The base
        adapter reports the truth — this platform path carries
        no location verb — and each adapter that CAN send one
        overrides.  Never a fake success; shape rules are
        :func:`location_shape_error`, checked before traffic
        by every override."""
        return SendResult(ok=False, platform=self.name,
                          error="location sending not supported "
                                "on this adapter")

    def send_venue(self, chat: ChatRef, latitude: float,
                   longitude: float, title: str,
                   address: str = "") -> SendResult:
        """Send a NAMED place (a venue): a pin that carries its
        name and street address, the form a person shares for
        a business or a meeting point.  Each platform renders
        it in its own named-pin form (WhatsApp's location
        message name/address fields, Telegram's venue).  The
        base adapter reports the truth like send_location."""
        return SendResult(ok=False, platform=self.name,
                          error="venue sending not supported "
                                "on this adapter")

    def send_contact(self, chat: ChatRef, phone: str, name: str,
                     *, vcard: str = "") -> SendResult:
        """Share a contact card (name + phone number) in a
        chat — the card a recipient saves into their own
        address book with one tap.  ``vcard`` lets a caller
        pass a richer card; adapters build the standard
        vCard 3.0 (vcf.build_vcard) when it is empty.  The
        base adapter reports the truth like send_location;
        shape rules are :func:`contact_shape_error`."""
        return SendResult(ok=False, platform=self.name,
                          error="contact cards not supported "
                                "on this adapter")

    # ── scheduled sends (parity bundle #14) ──────────────
    def send_scheduled(self, chat: ChatRef, text: str,
                       when: float, *,
                       reply_to: str = "") -> SendResult:
        """Queue one text message to land at ``when``
        (epoch seconds) instead of now — the platform's
        native scheduled send where the platform HAS one
        (Telegram's MTProto schedule date).  The base
        adapter reports the truth — this platform path
        carries no schedule verb — and each adapter that
        CAN schedule overrides.  Never a fake success, and
        never a silent downgrade to sending now."""
        return SendResult(ok=False, platform=self.name,
                          error="scheduled messages not "
                                "supported on this adapter")

    def send_scheduled_media(self, chat: ChatRef,
                             media: MediaRef, when: float, *,
                             caption: str = "") -> SendResult:
        """The media form of :meth:`send_scheduled`: one
        attachment queued to land at ``when`` (epoch
        seconds), ``caption`` riding as its caption.  Same
        truth rule as the text form — the base reports the
        absence, capable adapters override."""
        return SendResult(ok=False, platform=self.name,
                          error="scheduled media not "
                                "supported on this adapter")

    def typing(self, chat: ChatRef, seconds: float = 3.0,
             action: str = "typing") -> bool:
        """Show a typing indicator. Best-effort; adapters without one return False."""
        return False

    def history(self, chat: ChatRef, limit: int = 20) -> list[ChatMessage]:
        """Recent inbound history, oldest first. Empty when unavailable."""
        return []

    # ── message search (parity bundle #11) ───────────────
    def search_messages(self, chat: ChatRef, query: str,
                        limit: int = 20) -> dict[str, Any]:
        """Search ONE chat's messages for ``query`` — the
        in-chat half of the search-depth bundle.  Returns a
        verdict dict, the read-verb shape the poll-results
        family established:

        ``{"ok": True, "query": …, "messages": […],
        "total": n, "coverage": "…"}`` — messages are
        :class:`ChatMessage` newest-first, ``total`` counts
        every match found (it exceeds ``len(messages)``
        when the cap trimmed the list), and ``coverage``
        names in words what was actually searched, because
        each surface's reach differs and a result that hides
        its reach overstates itself.  Failure is
        ``{"ok": False, "error": "…"}`` with the specific
        reason.  The base adapter reports the truth — this
        platform path carries no search verb — and each
        adapter that CAN search overrides.  Shape rules are
        :func:`search_shape_error`, checked before traffic
        by every override."""
        return {"ok": False,
                "error": "message search not supported "
                         "on this adapter"}

    def search_global(self, query: str,
                      limit: int = 20) -> dict[str, Any]:
        """Search across EVERY chat on this surface — the
        cross-chat half of the search-depth bundle.  Same
        verdict shape as :meth:`search_messages`; each
        message's ``chat`` carries its own ChatRef, so a
        hit always names the conversation it lives in.
        The base adapter reports the truth like
        search_messages."""
        return {"ok": False,
                "error": "message search not supported "
                         "on this adapter"}

    # ── status depth (parity bundle #13) ─────────────────
    def status_audience(self) -> dict[str, Any]:
        """Read the account's status audience — who its
        status posts go to.  Returns a verdict dict:

        ``{"ok": True, "mode": "contacts" | "except" |
        "only", "list": [person jids], "list_known": bool}``
        — ``list_known`` False means the surface could not
        hand back the list itself, so the mode stands alone
        and nobody invents the names.  Failure is
        ``{"ok": False, "error": "…"}`` with the specific
        reason.  The base adapter reports the truth — this
        platform path carries no status-audience list —
        and adapters that can read one override."""
        return {"ok": False,
                "error": "a status audience list is not "
                         "supported on this adapter"}

    def set_status_audience(self, mode: str,
                            audience: list[str] | None = None
                            ) -> dict[str, Any]:
        """Set the account's status audience.  Same verdict
        shape as :meth:`status_audience`, carrying the
        values read back after the write (a set that cannot
        be read back is reported, never assumed).  Shape
        rules are :func:`status_audience_shape_error`,
        checked before traffic by every override."""
        return {"ok": False,
                "error": "a status audience list is not "
                         "supported on this adapter"}

    def status_viewers(self, message_id: str
                       ) -> dict[str, Any]:
        """Who has viewed one of the account's status
        posts.  Returns a verdict dict:

        ``{"ok": True, "message_id": …, "viewers":
        [{"jid": …, "ts": ms}], "count": n, "posted_here":
        bool, "coverage": "…"}`` — ``coverage`` names in
        words what the count actually covers, because no
        surface hands a program the server-side viewer
        list and a count that hides its reach overstates
        itself.  The base adapter reports the truth."""
        return {"ok": False,
                "error": "status viewers are not supported "
                         "on this adapter"}

    # ── bot self-profile + Stars invoices (parity bundle #18) ──
    def get_bot_profile(self) -> dict[str, Any]:
        """Read a BOT's own public identity: its display
        name, long description and short description (the
        BotFather fields).  Only a bot surface HAS a bot
        profile — a personal account's identity lives in
        the /myprofile family instead — so the base adapter
        reports the truth and the one bot adapter overrides.
        Verdict shape like the read verbs: ``{"ok": True,
        "name": …, "description": …, "short_description": …,
        "username": …}`` or ``{"ok": False, "error": …}``.
        Shape rules are :func:`bot_profile_shape_error`."""
        return {"ok": False,
                "error": "bot self-profile not supported "
                         "on this adapter"}

    def set_bot_name(self, name: str) -> dict[str, Any]:
        """Set a bot's display name ("" reverts to the
        default).  Bot surfaces only, like get_bot_profile."""
        return {"ok": False,
                "error": "bot self-profile not supported "
                         "on this adapter"}

    def set_bot_description(self, description: str
                            ) -> dict[str, Any]:
        """Set a bot's long description — the text on its
        profile page ("" clears it).  Bot surfaces only."""
        return {"ok": False,
                "error": "bot self-profile not supported "
                         "on this adapter"}

    def set_bot_short_description(self, short: str
                                  ) -> dict[str, Any]:
        """Set a bot's short description — the line shown
        when its profile is shared ("" clears it).  Bot
        surfaces only."""
        return {"ok": False,
                "error": "bot self-profile not supported "
                         "on this adapter"}

    def send_invoice(self, chat: ChatRef, title: str,
                     description: str, amount_stars: int, *,
                     payload: str = "") -> dict[str, Any]:
        """Send a Telegram Stars (XTR) invoice INTO a chat:
        the buyer pays in Stars, Telegram runs the checkout,
        and a paid invoice lands back as a
        successful_payment service message.  Only the
        Telegram bot surface can charge — Stars are a bot
        payment rail, no personal account or other platform
        has them — so the base adapter reports the truth
        and the bot adapter overrides.  ``{"ok": True,
        "message_id": …}`` or ``{"ok": False, "error": …}``;
        shape rules are :func:`invoice_shape_error`,
        checked before traffic by every override."""
        return {"ok": False,
                "error": "invoices not supported "
                         "on this adapter"}

    def create_invoice_link(self, title: str,
                            description: str,
                            amount_stars: int, *,
                            payload: str = "") -> dict[str, Any]:
        """Mint a shareable Stars invoice LINK instead of
        posting one into a chat — the same invoice, as a
        t.me URL anyone can open and pay.  ``{"ok": True,
        "link": …}`` or the honest refusal, like
        send_invoice."""
        return {"ok": False,
                "error": "invoices not supported "
                         "on this adapter"}

    # ── commerce: labels + catalog (parity bundle #17) ──
    def get_labels(self) -> dict[str, Any]:
        """The account's WhatsApp labels — definitions and
        chat/message associations — as a verdict dict:
        ``{"ok": True, "labels": […], "chat_labels": {…},
        "message_labels": […], "coverage": "…"}``, where
        ``coverage`` names in words what the read actually
        reached (each engine's label reach differs, and a
        result that hides its reach overstates itself).
        Labels are a WhatsApp Business surface; the base
        adapter reports the truth — this platform path
        carries no label state — and the WhatsApp adapter
        overrides.  Shape rules are
        :func:`label_edit_shape_error` /
        :func:`label_assoc_shape_error`, checked before
        traffic by every override."""
        return {"ok": False,
                "error": "labels not supported on this adapter"}

    def edit_label(self, name: str = "", *,
                   label_id: str = "", color: Any = None,
                   deleted: bool = False) -> dict[str, Any]:
        """Create, rename/recolor, or delete one label
        definition.  The definition is a whole record —
        name, color and the deleted flag travel together —
        so a rename carries the label's current color, which
        the caller resolves from a fresh :meth:`get_labels`
        read.  The base adapter reports the truth like
        get_labels."""
        return {"ok": False,
                "error": "labels not supported on this adapter"}

    def label_chat(self, chat: ChatRef, label_id: str, *,
                   labeled: bool = True) -> dict[str, Any]:
        """Apply a label to a whole chat (or lift it, with
        ``labeled=False``).  The base adapter reports the
        truth like get_labels."""
        return {"ok": False,
                "error": "labels not supported on this adapter"}

    def label_message(self, chat: ChatRef, message_id: str,
                      label_id: str, *,
                      labeled: bool = True) -> dict[str, Any]:
        """Apply a label to one message (or lift it).  The
        base adapter reports the truth like get_labels."""
        return {"ok": False,
                "error": "labels not supported on this adapter"}

    def get_catalog(self, jid: str = "", *, limit: int = 10,
                    cursor: str = "") -> dict[str, Any]:
        """Read a product catalog — the account's own when
        ``jid`` is empty, another business's when it names
        one — as a verdict dict: ``{"ok": True, "products":
        […], "collections": […], "next_cursor": "…"}``.
        A catalog is a WhatsApp Business surface and only
        one bridge engine's library carries it; the base
        adapter reports the truth and the WhatsApp adapter
        overrides, surfacing the engine's own refusal when
        the engine cannot.  Shape rules for the writes are
        :func:`catalog_product_shape_error`."""
        return {"ok": False,
                "error": "product catalog not supported "
                         "on this adapter"}

    def create_product(self, name: str, price: Any,
                       currency: str, *,
                       description: str = "",
                       retailer_id: str = "",
                       url: str = "") -> dict[str, Any]:
        """Add one product to the account's own catalog.
        The base adapter reports the truth like
        get_catalog."""
        return {"ok": False,
                "error": "product catalog not supported "
                         "on this adapter"}

    def update_product(self, product_id: str, *,
                       name: Any = None, price: Any = None,
                       currency: Any = None,
                       description: Any = None,
                       retailer_id: Any = None,
                       url: Any = None) -> dict[str, Any]:
        """Change one product in the account's own catalog.
        The write is a whole record, so the WhatsApp
        adapter read-modify-writes: it resolves the current
        product from a fresh catalog read and sends the
        merged record — a field the caller leaves as None
        keeps the value the catalog already carries.  The
        base adapter reports the truth like get_catalog."""
        return {"ok": False,
                "error": "product catalog not supported "
                         "on this adapter"}

    def delete_product(self, product_id: str
                       ) -> dict[str, Any]:
        """Remove one product from the account's own
        catalog.  The base adapter reports the truth like
        get_catalog."""
        return {"ok": False,
                "error": "product catalog not supported "
                         "on this adapter"}

    # ── health ───────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "running": self._thread is not None and self._thread.is_alive(),
            "restart_crashes": self._crash_count,
            "last_crash": self._last_crash,
            "gave_up": self._gave_up,
            **self.stats,
        }
