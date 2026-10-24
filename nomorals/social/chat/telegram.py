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
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ...core.errors import ValidationError
from ...core.logging_setup import get_logger
from ...core.text import safe_filename
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult
from .platform_limits import for_platform
from .failures import record_failure


def _is_entity_parse_error(exc: BaseException) -> bool:
    """True when a send failure is the API rejecting the *formatting*.

    The Bot API answers malformed entities with ``Bad Request: can't
    parse entities: …``. The adapters retry such a send once as PLAIN
    text rather than losing the message — a formatting failure must
    never eat an alert (F1). Deliberately narrow: peer-resolution
    errors ("Cannot find any entity corresponding to…") are about the
    *chat*, not the markup, and must not trigger the fallback.
    """
    s = str(exc or "").lower()
    if "corresponding to" in s:
        return False
    return ("can't parse entities" in s
            or "can not parse entities" in s
            or "cannot parse entities" in s
            or "parse entities" in s
            or "entity offset" in s)

__all__ = ["TelegramAdapter", "TelegramBotAdapter"]

_log = get_logger(__name__)

def _flood_wait_seconds(exc: BaseException) -> int:
    """Telegram's native rate-limit signal, duck-typed.

    Telethon raises ``errors.FloodWaitError`` with a ``.seconds`` attr;
    the Bot API answers HTTP 429 with ``parameters.retry_after``. This
    helper reads either, so the retry logic doesn't import Telethon at
    module level (optional dependency). 0 = not a rate-limit signal.
    """
    try:
        seconds = int(getattr(exc, "seconds", 0) or 0)
        if seconds > 0 and "flood" in type(exc).__name__.lower():
            return seconds
    except (TypeError, ValueError):
        pass
    try:
        status = int(getattr(exc, "status", 0) or 0)
        if status == 429:
            return int((getattr(exc, "parameters", None) or {}).get("retry_after", 0) or 0)
    except (TypeError, ValueError) as exc:
        record_failure('telegram._flood_wait_seconds', exc)
    return 0


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


def _safe_ts(value: Any) -> float:
    """Coerce an external ``date`` field to a float timestamp.

    Telegram sends a Unix int, but a malformed / forged update can carry
    a string, list, or null — ``float(...)`` on those raises. Fall back
    to now so one bad timestamp never drops the message.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return time.time()


_MIME_HINTS = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".mp4": "video/mp4",
    ".mkv": "video/x-matroska", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
    # Voice-note / audio formats (Telegram voice notes are .ogg Opus;
    # WhatsApp/iOS/Android recorders send .opus/.m4a/.wav/.aac/.amr):
    ".m4a": "audio/mp4", ".opus": "audio/opus", ".oga": "audio/ogg",
    ".wav": "audio/wav", ".aac": "audio/aac", ".flac": "audio/flac",
    ".amr": "audio/amr", ".3gp": "audio/3gpp",
    ".aiff": "audio/aiff", ".aif": "audio/aiff",
    ".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _mime_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _MIME_HINTS.get(ext, "application/octet-stream")


def _mime_for_download(path: str, doc_mime: str = "") -> str:
    """MIME for a downloaded file: the extension map first, then the
    Telegram document's own ``mime_type`` for extension-less downloads
    (the common MTProto case — Telethon returns a bare name, which
    otherwise lands as ``application/octet-stream`` and fails every
    STT check even though Telegram told us it's audio)."""
    mime = _mime_for(path)
    if mime == "application/octet-stream" and doc_mime:
        mime = doc_mime.lower()
    return mime


def _kind_for_mime(mime: str) -> str:
    """MediaRef kind from a MIME type: audio stays audio so the STT
    checks (``kind in ("audio", "voice")`` or ``mime.startswith("audio")``)
    match. Previously every non-image download became ``"file"`` and
    voice notes only survived via their mime — which the narrow MIME
    map usually got wrong."""
    mime = (mime or "").lower()
    if mime.startswith("audio"):
        return "audio"
    if mime.startswith("image"):
        return "image"
    if mime.startswith("video"):
        return "video"
    return "file"


def _kind_for(entity: Any) -> str:
    if getattr(entity, "megagroup", False) or getattr(entity, "gigagroup", False):
        return ChatKind.GROUP
    if getattr(entity, "channel", False) and not getattr(entity, "megagroup", False):
        return ChatKind.CHANNEL
    return ChatKind.DM


def _event_kind_override(kind: str, event: Any) -> str:
    """Force group/channel classification when the event says so.

    Basic groups (Telethon ``Chat``, not ``Channel``) resolved via a
    fallback carry no ``megagroup``/``channel`` flags, so :func:`_kind_for`
    defaults them to DM. Trust the event: if Telegram says it's a group,
    it's a group. Without this, a basic group like ``-5223197263`` was
    delivered as DM ``5223197263`` and every reply left the group.
    """
    if kind != ChatKind.DM:
        return kind
    if bool(getattr(event, "is_group", False)):
        _log.debug(
            "telegram: event.is_group=True but entity has no group flags — "
            "forcing kind=group"
        )
        return ChatKind.GROUP
    if bool(getattr(event, "is_channel", False)):
        _log.debug(
            "telegram: event.is_channel=True but entity has no channel flags — "
            "forcing kind=channel"
        )
        return ChatKind.CHANNEL
    return kind


def _canonical_chat_id(orig_chat_id: str, entity_id: str, kind: str) -> str:
    """Canonical chat id after entity resolution.

    Basic groups: Telethon ``Chat`` entities carry the positive id, but the
    real Telegram id is negative (``-5223197263``). Preserve the minus so
    routing, replies, and the allowlist target the group — not a phantom DM.

    Supergroups are unaffected and keep the existing positive-id convention:
    ``-100…`` lstrips to ``100…``, which never equals the entity id.
    """
    orig = (orig_chat_id or "").strip()
    if orig.startswith("-") and entity_id == orig.lstrip("-") and kind != ChatKind.DM:
        return orig
    return entity_id


def _sender_entity_matches(sender_entity: Any, event_sender_id: Any) -> bool:
    """True when ``event.sender`` really is the message's author.

    Telethon can populate ``event.sender`` with the GROUP/CHANNEL entity
    instead of the human sender (channels, channel-like supergroups where
    the post's author isn't a resolvable user). The numeric
    ``event.sender_id`` is authoritative for the human author; if the
    entity's id disagrees with it, the entity is the group itself and
    must not be used — it would set ``sender_id``/``sender_name`` to the
    group's id/title (e.g. entity.id=5223197263 vs
    event.sender_id=7541672134 → "xauusd_sentinel_signal" as the player
    name) and corrupt downstream identity lookups.

    Returns True (trust the entity) when there is no positive numeric
    ``event.sender_id`` to cross-check against — e.g. channel posts,
    whose sender id is the channel itself (negative), carry no human
    author to fall back to.
    """
    if sender_entity is None:
        return False
    evt_sender_id = str(event_sender_id or "")
    if not evt_sender_id.isdigit():
        return True
    ent_id = str(getattr(sender_entity, "id", "") or "")
    return not ent_id or ent_id == evt_sender_id


_MISS = object()  # negative-cache sentinel: "looked up, not found"


class EntityCache:
    """TTL entity cache with multi-key indexing, negative caching, and stats.

    One Telegram entity is reachable under several keys — numeric id
    (``7541672134``), ``@username``, phone digits — and every one of them
    is indexed, so a later lookup under any form is a cache hit instead
    of another MTProto round-trip.

    Failures are cached too (short TTL): a privacy-locked user that
    ``get_entity`` can't resolve won't be re-probed on every inbound
    message — the negative entry expires after ``negative_ttl`` so a
    later retry can still succeed (e.g. after the dialog pre-fetch).

    Thread-safe; oldest-first eviction past ``max_entries``.
    """

    def __init__(self, max_entries: int = 2000, ttl: float = 3600.0,
                 negative_ttl: float = 300.0) -> None:
        import threading
        self._lock = threading.Lock()
        self._max = max(1, int(max_entries))
        self._ttl = float(ttl)
        self._negative_ttl = float(negative_ttl)
        # primary key -> [value, expires_at, [alias keys...]]
        self._data: dict[str, list] = {}
        # alias key -> primary key
        self._aliases: dict[str, str] = {}
        # insertion order for oldest-first eviction
        self._order: list[str] = []
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.neg_hits = 0
        self.lookups = 0
        self.total_lookup_s = 0.0
        self.slow_lookups = 0  # > 0.5s

    # -- dict-compatible surface (existing call sites / tests) --------------
    def __len__(self) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for v in self._data.values() if v[0] is not _MISS and v[1] > now)

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None

    # -- core ----------------------------------------------------------------
    @staticmethod
    def _norm(key: object) -> str:
        return str(key or "").strip()

    def _primary(self, key: str) -> str:
        return self._aliases.get(key, key)

    def get(self, key: object) -> Any:
        """Return the cached entity, or None. Negative entries -> None."""
        k = self._norm(key)
        if not k:
            return None
        now = time.monotonic()
        with self._lock:
            pk = self._primary(k)
            row = self._data.get(pk)
            if row is None:
                return None
            value, expires, _aliases = row
            if expires <= now:
                self._drop_locked(pk)
                return None
            if value is _MISS:
                return None
            return value

    def is_negative(self, key: object) -> bool:
        """True if this key recently failed to resolve (don't re-probe yet)."""
        k = self._norm(key)
        if not k:
            return False
        now = time.monotonic()
        with self._lock:
            pk = self._primary(k)
            row = self._data.get(pk)
            if row is None:
                return False
            value, expires, _aliases = row
            if expires <= now:
                self._drop_locked(pk)
                return False
            return value is _MISS

    def set(self, key: object, value: Any, aliases: list[str] | None = None,
            ttl: float | None = None) -> None:
        """Store an entity under ``key`` plus optional alias keys."""
        k = self._norm(key)
        if not k or value is None:
            return
        now = time.monotonic()
        expires = now + (self._ttl if ttl is None else float(ttl))
        alias_keys = [self._norm(a) for a in (aliases or [])]
        alias_keys = [a for a in alias_keys if a and a != k]
        with self._lock:
            # don't let an alias hijack an existing primary entry
            if k in self._aliases and self._aliases[k] != k:
                old_pk = self._aliases[k]
                self._drop_locked(old_pk)
            self._data[k] = [value, expires, alias_keys]
            for a in alias_keys:
                # never overwrite a primary entry with an alias pointer
                if a not in self._data:
                    self._aliases[a] = k
                elif self._aliases.get(a, a) == a and a != k:
                    # alias collides with another primary: keep the primary
                    pass
                else:
                    self._aliases[a] = k
            if k not in self._order:
                self._order.append(k)
            self.sets += 1
            self._evict_locked()

    def set_negative(self, key: object) -> None:
        """Remember that ``key`` failed to resolve (short TTL)."""
        k = self._norm(key)
        if not k:
            return
        now = time.monotonic()
        with self._lock:
            pk = self._primary(k)
            # don't overwrite a good entry with a failure
            row = self._data.get(pk)
            if row is not None and row[0] is not _MISS and row[1] > now:
                return
            self._data[pk] = [_MISS, now + self._negative_ttl, []]
            if pk not in self._order:
                self._order.append(pk)

    def record_lookup(self, duration_s: float, hit: bool) -> None:
        with self._lock:
            self.lookups += 1
            self.total_lookup_s += duration_s
            if hit:
                self.hits += 1
            else:
                self.misses += 1
            if duration_s > 0.5:
                self.slow_lookups += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            live = sum(1 for v in self._data.values()
                       if v[0] is not _MISS and v[1] > time.monotonic())
            avg = self.total_lookup_s / self.lookups if self.lookups else 0.0
            return {
                "entries": live,
                "max_entries": self._max,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) else 0.0,
                "sets": self.sets,
                "evictions": self.evictions,
                "neg_hits": self.neg_hits,
                "lookups": self.lookups,
                "avg_lookup_s": round(avg, 4),
                "slow_lookups": self.slow_lookups,
            }

    def _drop_locked(self, pk: str) -> None:
        row = self._data.pop(pk, None)
        if row is not None:
            for a in row[2]:
                if self._aliases.get(a) == pk:
                    self._aliases.pop(a, None)
        # also drop alias->pk pointers pointing at pk
        for a in [a for a, p in self._aliases.items() if p == pk]:
            self._aliases.pop(a, None)
        if pk in self._order:
            self._order.remove(pk)

    def _evict_locked(self) -> None:
        while len(self._data) > self._max and self._order:
            oldest = self._order[0]
            self._drop_locked(oldest)
            self.evictions += 1


def _entity_aliases(entity: Any) -> list[str]:
    """All cache keys one entity answers to: id, @username, phone digits."""
    out: list[str] = []
    try:
        eid = getattr(entity, "id", None)
        if eid:
            out.append(str(eid))
        for attr in ("user_id", "chat_id", "channel_id"):
            v = getattr(entity, attr, None)
            if v:
                out.append(str(v))
                break
        uname = getattr(entity, "username", None)
        if uname:
            out.append("@" + str(uname).strip().lstrip("@").lower())
        phone = getattr(entity, "phone", None)
        if phone:
            digits = re.sub(r"\D", "", str(phone))
            if digits:
                out.append("tel:" + digits)
    except Exception as exc:  # noqa: BLE001 - alias extraction is best-effort
        _log.debug("telegram: entity alias extraction failed: %s", exc)
    # de-dupe, preserve order
    seen: set[str] = set()
    return [k for k in out if k and not (k in seen or seen.add(k))]


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
        companion_bot_ids: Any = None,
        flood_max_wait_s: float = 300.0,
    ) -> None:
        super().__init__(media_dir=media_dir)
        self.media_in_groups = bool(media_in_groups)
        self.media_max_mb = float(media_max_mb)
        # Telegram flood waits: how long a send retry will honor (per
        # retry). Larger waits fail fast instead of parking the send
        # thread — the gateway/outbound caller can retry later.
        self.flood_max_wait_s = max(1.0, float(flood_max_wait_s))
        # IDs of the companion BotFather bots (if any). The bots' messages
        # arrive at this userbot as incoming — without this, the
        # _sender_is_bot guard relies solely on Telethon's .bot flag,
        # which isn't always populated (unresolved senders, certain
        # chat types). An explicit ID match is bulletproof.
        # ``companion_bot_id`` (singular) is kept for back-compat: it is
        # folded into the set below. New code passes ``companion_bot_ids``
        # with BOTH bot IDs — the userbot must never reply to either bot.
        _ids: set[int] = set()
        _raw_ids: Any = companion_bot_ids
        if _raw_ids is None:
            _raw_ids = ()
        elif isinstance(_raw_ids, (str, int)):
            _raw_ids = (_raw_ids,)
        for _cand in _raw_ids:
            try:
                if _cand:
                    _ids.add(int(_cand))
            except (TypeError, ValueError) as exc:
                record_failure('telegram.TelegramAdapter.__init__', exc)
        try:
            if companion_bot_id:
                _ids.add(int(companion_bot_id))
        except (TypeError, ValueError) as exc:
            record_failure('telegram.TelegramAdapter.__init__#2', exc)
        self._companion_bot_ids: frozenset[int] = frozenset(_ids)
        # Singular back-compat view: the first (lowest) id, or None.
        self._companion_bot_id: int | None = min(_ids) if _ids else None
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
        # (chat_id, message_id) of inbound messages already delivered.
        # Telethon can replay the same Telegram update (channel difference
        # on reconnect, update re-delivery); the adapter guarantees
        # at-most-once processing per Telegram message here, at the source.
        self._seen_inbound: set[tuple[str, int]] = set()
        # the connection's event loop (set in run()); all outbound coroutines
        # must run on it — Telethon binds the client to that loop
        self._loop: asyncio.AbstractEventLoop | None = None
        # Entity cache: input entities from inbound messages + fully
        # resolved entities, keyed by id / @username / phone, with TTL.
        # Telethon's get_entity() fails for users not in the session cache
        # (fresh sessions, privacy settings), but get_input_entity() works
        # with input peers stored from received messages. We cache those so
        # outbound sends can reply to DM chats from users we've heard from
        # but can't fully resolve. Negative entries (failed lookups) are
        # cached briefly so one privacy-locked user doesn't cost an MTProto
        # round-trip on every inbound message.
        self._input_entity_cache: EntityCache = EntityCache(max_entries=2000)
        #: Hard cap on the entity cache (mirrors EntityCache max; kept as
        #: a plain attribute for introspection).
        self._input_entity_cache_max = 2000

    @property
    def companion_bot_ids(self) -> frozenset[int]:
        """Both companion BotFather bot IDs (gated + personal)."""
        return self._companion_bot_ids

    def _cache_input_entity(self, chat_id: str, entity: Any) -> None:
        """Store an input entity, indexed under every key it answers to."""
        if entity is None:
            return
        aliases = _entity_aliases(entity)
        # always index under the chat_id we were given, too
        cid = str(chat_id or "").strip()
        if cid and cid not in aliases:
            aliases.append(cid)
        stripped = cid.lstrip("-")
        if stripped and stripped != cid and stripped not in aliases:
            aliases.append(stripped)
        self._input_entity_cache.set(cid or (aliases[0] if aliases else ""),
                                     entity, aliases=aliases)

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

    def _note_inbound_seen(self, chat_id: str, message_id: int) -> bool:
        """Record an inbound Telegram message id; True if already seen.

        At-most-once delivery guard: Telethon can replay the same update
        (channel difference on reconnect, update re-delivery). The bounded
        set keeps the last ~5000 (chat_id, message_id) pairs.
        """
        key = (str(chat_id), int(message_id))
        if key in self._seen_inbound:
            return True
        self._seen_inbound.add(key)
        if len(self._seen_inbound) > 5000:
            for _ in range(2500):
                self._seen_inbound.pop()
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

    def saved_messages_ref(self) -> ChatRef | None:
        """ChatRef for the owner's Saved Messages (their self-chat).

        Generated artifacts on this personal surface default here (see
        ``social.chat.voice_transport``). The self-chat is addressed by
        the account's own user id — the same id the inbound self-chat
        gate keys on. None until connect() has learned the id: a send
        must never guess the destination.
        """
        if not self._me_id:
            return None
        return ChatRef(platform=self.name, chat_id=str(self._me_id),
                       kind=ChatKind.DM, title="Saved Messages")

    def _sender_is_bot(self, event: Any, message: Any) -> bool:
        """True iff the sender of this inbound message is a bot account.

        The companion BotFather bots run as separate Telegram accounts. When
        the adapters are live, the bots' messages arrive at this userbot as
        incoming — without this guard the userbot would reply to its own
        bots' output in a loop.
        """
        # Primary: explicit companion bot ID match — bulletproof even when
        # Telethon hasn't resolved the sender entity. Membership test
        # against the full set: the userbot must not reply to EITHER bot.
        if self._companion_bot_ids:
            for candidate in (
                getattr(event, "sender_id", None),
                getattr(getattr(event, "message", None), "sender_id", None),
                getattr(message, "sender_id", None),
            ):
                try:
                    if candidate is not None and int(candidate) in self._companion_bot_ids:
                        return True
                except (TypeError, ValueError) as exc:
                    _log.debug("bot detection: candidate %r not numeric (%s)", candidate, exc)
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
    async def _timed_get_entity(self, peer: Any) -> Any | None:
        """Single get_entity call with timing + result caching.

        Successes are indexed under every key the entity answers to
        (id, @username, phone); failures are negative-cached briefly.
        Returns None instead of raising.
        """
        if self._client is None:
            return None
        t0 = time.perf_counter()
        try:
            entity = await self._client.get_entity(peer)
        except Exception as exc:  # noqa: BLE001
            record_failure('telegram.TelegramAdapter._timed_get_entity', exc)
            entity = None
        dt = time.perf_counter() - t0
        self._input_entity_cache.record_lookup(dt, entity is not None)
        if dt > 0.5:
            _log.debug("telegram: slow get_entity(%r) took %.2fs", peer, dt)
        if entity is None:
            self._input_entity_cache.set_negative(str(peer))
            return None
        aliases = _entity_aliases(entity)
        primary = aliases[0] if aliases else str(peer)
        self._input_entity_cache.set(primary, entity, aliases=aliases)
        return entity

    async def _resolve(self, chat: ChatRef) -> Any:
        """Resolve a chat entity, using cached input entities from inbound messages.

        Telethon's get_entity() fails for users not in the session cache (fresh
        sessions, privacy settings), causing send failures. Resolution order:
        1. Entity cache (id / @username / phone — TTL'd, negative-cached)
        2. client.get_input_entity() (uses Telethon's internal peer cache)
        3. client.get_entity() (full resolution — timed, result cached)
        4. Raw integer ID (Telethon can sometimes send to just the ID)
        """
        chat_id = chat.chat_id
        t0 = time.perf_counter()

        # 1. Entity cache (covers id, @username, stripped basic-group forms)
        entity = self._input_entity_cache.get(chat_id)
        if entity is None:
            stripped = chat_id.lstrip("-")
            if stripped != chat_id:
                entity = self._input_entity_cache.get(stripped)
        if entity is not None:
            self._input_entity_cache.record_lookup(time.perf_counter() - t0, True)
            _log.debug("telegram: entity cache hit for chat_id=%s", chat_id)
            return entity
        if self._input_entity_cache.is_negative(chat_id):
            self._input_entity_cache.neg_hits += 1
            _log.debug("telegram: negative cache hit for chat_id=%s — skipping probe", chat_id)
            if chat_id.lstrip("-").isdigit():
                return int(chat_id)
            return None

        # 2. Try get_input_entity (uses Telethon's internal peer cache)
        if self._client is not None:
            try:
                if chat_id.startswith("@"):
                    return await self._client.get_input_entity(chat_id)
                return await self._client.get_input_entity(int(chat_id))
            except Exception as exc:  # noqa: BLE001
                # Fall through to full resolution — but log it: a
                # silently-failing peer cache used to be invisible.
                _log.debug(
                    "telegram: get_input_entity failed for chat_id=%s: %s",
                    chat_id, exc,
                )

        # 3. Timed full resolution (caches success + failure)
        if self._client is not None:
            try:
                peer = chat_id if chat_id.startswith("@") else int(chat_id)
            except (TypeError, ValueError):
                peer = chat_id
            entity = await self._timed_get_entity(peer)
            if entity is not None:
                return entity

        # 4. Last resort: return the raw integer ID.
        # Telethon's send_message can sometimes work with just a user ID.
        if chat_id.lstrip("-").isdigit():
            _log.warning("telegram: could not resolve chat_id=%s, using raw ID", chat_id)
            return int(chat_id)
        return None

    async def resolve_username(self, username: str) -> Any | None:
        """Fast ``@username`` → entity resolution with caching.

        Returns the cached entity when seen before (no network), else a
        single timed ``get_entity`` whose result is indexed under the
        username, numeric id, and phone. None on failure.
        """
        uname = (username or "").strip().lstrip("@").lower()
        if not uname:
            return None
        key = "@" + uname
        t0 = time.perf_counter()
        entity = self._input_entity_cache.get(key)
        if entity is not None:
            self._input_entity_cache.record_lookup(time.perf_counter() - t0, True)
            return entity
        if self._input_entity_cache.is_negative(key):
            self._input_entity_cache.neg_hits += 1
            return None
        entity = await self._timed_get_entity(uname)
        # _timed_get_entity already cached under discovered aliases; make
        # sure the queried username form hits too
        if entity is not None and self._input_entity_cache.get(key) is None:
            self._input_entity_cache.set(key, entity,
                                         aliases=_entity_aliases(entity))
        return entity

    async def resolve_many(self, keys: list[str], limit: int = 5) -> dict[str, Any]:
        """Resolve many keys at once → ``{key: entity}`` for successes.

        Cache hits return instantly; uncached keys resolve concurrently
        (bounded by ``limit``) through the timed path, so a 50-member
        group roster doesn't cost 50 sequential round-trips.
        """
        out: dict[str, Any] = {}
        pending: list[str] = []
        for k in keys:
            k = str(k or "").strip()
            if not k:
                continue
            ent = self._input_entity_cache.get(k)
            if ent is not None:
                out[k] = ent
            elif not self._input_entity_cache.is_negative(k):
                pending.append(k)
        if pending and self._client is not None:
            sem = asyncio.Semaphore(max(1, int(limit)))

            async def _one(k: str) -> tuple[str, Any | None]:
                async with sem:
                    try:
                        peer: Any = k if k.startswith("@") else int(k)
                    except (TypeError, ValueError):
                        peer = k
                    return k, await self._timed_get_entity(peer)

            for k, ent in await asyncio.gather(*(_one(k) for k in pending)):
                if ent is not None:
                    out[k] = ent
        return out

    def entity_cache_stats(self) -> dict[str, Any]:
        """Cache health for the dashboard: hit rate, avg lookup time, etc."""
        return self._input_entity_cache.stats()
        
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
        # ── at-most-once inbound ─────────────────────────────────────
        # Telethon can deliver the same Telegram message twice (channel
        # difference fetch on reconnect, update replay). Drop the replay
        # here so nothing downstream ever sees the message twice — the
        # dashboard, the brain, and the game engine all get exactly one
        # copy. This is delivery guarantee, not UI deduplication.
        try:
            _mid = int(getattr(message, "id", 0) or 0)
        except (TypeError, ValueError):
            _mid = 0
        if _mid and self._note_inbound_seen(chat_id, _mid):
            _log.debug(
                "telegram: skipping replayed message id=%s in chat_id=%s",
                _mid, chat_id,
            )
            return
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
            # CRITICAL: only fall back to the sender for DMs. If the original
            # chat_id looks like a group (negative), event_chat being None
            # means Telethon couldn't resolve the GROUP — using event_sender
            # here would make entity.id the sender's user ID, and the later
            # `chat_id = str(entity.id)` overwrite would redirect the message
            # to the sender's DM instead of the group.
            # (This was the persistent group→DM redirect bug.)
            _cid = (chat_id or "").strip()
            _looks_like_group = _cid.startswith("-")
            if not _looks_like_group:
                entity = event_sender
            else:
                _log.debug(
                    "telegram: event_chat is None for group chat_id=%s — "
                    "skipping event_sender fallback (would redirect to DM)",
                    chat_id,
                )

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
                entity = await self._timed_get_entity(input_chat)
                if entity is not None:
                    _log.debug(
                        "telegram: entity resolved via input_chat chat_id=%s -> %r",
                        chat_id, getattr(entity, "id", None),
                    )

            if entity is None and input_sender is not None:
                # Same guard as the event_sender fallback above: for a group
                # chat_id, resolving input_sender yields the sender's user —
                # misclassifying the message as a DM. Only use it for DMs.
                _cid2 = (chat_id or "").strip()
                if not _cid2.startswith("-"):
                    entity = await self._timed_get_entity(input_sender)
                    if entity is not None:
                        _log.debug(
                            "telegram: entity resolved via input_sender chat_id=%s -> %r",
                            chat_id, getattr(entity, "id", None),
                        )
                else:
                    _log.debug(
                        "telegram: skipping input_sender resolve for group "
                        "chat_id=%s (would redirect to DM)",
                        chat_id,
                    )
        
        # Fallback: resolve through client using chat_id (timed + cached,
        # so a later message from the same chat is instant)
        if entity is None:
            try:
                peer: Any = chat_id
                if chat_id.lstrip("-").isdigit() and not chat_id.startswith("@"):
                    peer = int(chat_id)
            except (TypeError, ValueError):
                peer = chat_id
            entity = await self._timed_get_entity(peer)
        
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
        
        # ── Finalize chat identity: kind + canonical chat_id ──
        # NOTE on ordering: kind must be resolved BEFORE the input-entity
        # cache below, because the cache decision depends on whether this
        # is a group (never cache the sender's DM peer under a group id).
        orig_chat_id = (chat_id or "").strip()
        entity_id = str(getattr(entity, "id", ""))
        kind = _event_kind_override(_kind_for(entity), event)
        event_is_group = bool(getattr(event, "is_group", False))
        chat_id = _canonical_chat_id(orig_chat_id, entity_id, kind)
        if allow_check and self.chat_allow and not any(
            c == chat_id or c == chat_id.lstrip("-") for c in self.chat_allow
        ):
            _log.info(
                "telegram: ignoring message in %s (id=%s) — not in NM_CHAT_TELEGRAM_CHATS",
                getattr(entity, "title", None) or chat_id,
                chat_id,
            )
            return
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

        # Cache the input entity for outbound sends. Telethon's get_entity()
        # fails for users not in the session cache, but we can use the input
        # peer from received messages to send replies.
        #
        # CRITICAL: never cache input_sender under a group chat's ID. If
        # input_chat is missing on a group message and we fall back to
        # input_sender, the group's cache entry points at the sender's DM
        # peer — and every later reply to that group lands in the DM instead.
        # (This was the group→DM redirect bug.) The check below uses the
        # FINAL kind (which accounts for basic groups resolved without
        # entity flags), not the raw entity flags.
        input_chat = getattr(event, "input_chat", None)
        if kind in (ChatKind.GROUP, ChatKind.CHANNEL):
            input_entity = input_chat
        else:
            input_entity = input_chat or getattr(event, "input_sender", None)
        if input_entity is not None:
            self._cache_input_entity(chat_id, input_entity)
            _log.debug("telegram: cached input entity for chat_id=%s", chat_id)
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
                    path = await client.download_media(
                        message,
                        file=os.path.join(self.media_dir, fname))
                    if path:
                        if not path.endswith(tuple(_MIME_HINTS)):
                            ext = os.path.splitext(str(path))[1]
                            if not ext:
                                path = f"{path}.bin"
                        mime = _mime_for_download(
                            str(path),
                            str(getattr(doc, "mime_type", "") or ""))
                        # Telethon voice-note detection: a Telegram
                        # ``voice`` message is a spoken note, never a song
                        # clip. Telethon exposes it as message.voice, or as
                        # a DocumentAttributeAudio with voice=True. Without
                        # this, voice notes collapse to kind="audio" and
                        # the brain sees a bare "audio: ..." note and can
                        # hallucinate a music-ID reply. Defensive: any
                        # unexpected shape falls back to the MIME kind.
                        _is_voice = False
                        try:
                            _voice_attr = getattr(message, "voice", None)
                            # Real Telethon: message.voice is a
                            # DocumentAttributeAudio with voice=True, or
                            # None. ``is True`` (not truthy) so a Mock's
                            # auto-created attribute can't false-positive.
                            if _voice_attr is not None and getattr(
                                    _voice_attr, "voice", False) is True:
                                _is_voice = True
                            else:
                                for _attr in (getattr(doc, "attributes", None)
                                              or []):
                                    if getattr(_attr, "voice", False) is True:
                                        _is_voice = True
                                        break
                        except Exception as exc:  # noqa: BLE001 - best-effort
                            record_failure('telegram.TelegramAdapter._handle_inbound', exc)
                            _is_voice = False
                        _kind = "voice" if _is_voice else _kind_for_mime(mime)
                        media.append(
                            MediaRef(
                                path=str(path),
                                mime=mime,
                                kind=_kind,
                                is_voice=_is_voice,
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
        # Cross-check: Telethon can hand us the GROUP/CHANNEL entity as
        # `event.sender` instead of the human sender (channels and
        # channel-like supergroups). A mismatch against the event's
        # numeric sender id means the entity is NOT the message's author —
        # treat it as unresolvable so the event.sender_id fallback below
        # attributes the message to the real human sender.
        if not _sender_entity_matches(
            sender_entity, getattr(event, "sender_id", None)
        ):
            if sender_entity is not None:
                _log.debug(
                    "telegram: sender entity mismatch — entity.id=%s != "
                    "event.sender_id=%s; treating sender as unresolvable",
                    getattr(sender_entity, "id", "?"),
                    getattr(event, "sender_id", "?"),
                )
                sender_entity = None
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
            sender_id = ""
            sender_username = ""
            # Sender entity unresolvable — fall back to numeric IDs on the
            # event so game identity (and anything else keyed on sender_id)
            # doesn't fork into name-keyed phantom profiles like
            # `telegram:Mary`.
            #
            # Two legs, different scope:
            # 1. event.sender_id — the human who sent the message. Valid for
            #    DMs AND groups (a group /game must still attribute to the
            #    human sender). Only positive ids: negative sender ids are
            #    channel/group entities, not humans — never use those.
            # 2. chat_id as sender_id — DM-ONLY. In a DM the chat IS the user;
            #    in a group the chat is the group, and using it here would
            #    misattribute the message. Explicitly excluded for groups.
            evt_sender_id = str(getattr(event, "sender_id", "") or "")
            if evt_sender_id.isdigit():
                sender_id = evt_sender_id
            elif kind == ChatKind.DM and not event_is_group:
                if chat_id and chat_id.lstrip("-").isdigit():
                    sender_id = chat_id
            # Display name for an unresolvable sender: in a DM the chat
            # title IS the peer (= the sender), so it's the right label.
            # In a group the title is the GROUP's name — using it as the
            # sender's name misattributes the message ("xauusd_sentinel_
            # signal is already at the table") and, under the userbot's
            # name authority, clobbers the sender's canonical game profile
            # name with the group title.  Use a neutral `user_<id>`
            # placeholder instead; downstream layers resolve the real
            # display name from the stored profile.
            if kind == ChatKind.DM and not event_is_group:
                sender_name = title
            elif sender_id:
                sender_name = f"user_{sender_id}"
            else:
                sender_name = title
            if sender_id:
                _log.debug(
                    "telegram: sender_id fallback — using %s "
                    "(sender entity unresolvable, kind=%s)",
                    sender_id,
                    kind,
                )
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
    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             reply_keyboard: list[list[str]] | None = None,
             parse_mode: str = "") -> SendResult:
        client = self._client
        if client is None:
            return SendResult(ok=False, platform=self.name, error="not connected")
        started = time.perf_counter()
        try:
            result = self._run_on_loop(self._send_async(
                client, chat, text, reply_to, parse_mode,
                reply_keyboard=reply_keyboard))
            try:
                from ...core.error_system import heartbeat
                heartbeat("telegram", True)
            except Exception as exc:  # noqa: BLE001 - telemetry never breaks sends
                record_failure('telegram.TelegramAdapter.send', exc)
            return SendResult(
                ok=True, platform=self.name, message_id=str(getattr(result, "id", "")),
                seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - ordinary failure, report as result
            self.stats["send_errors"] += 1
            try:
                from ...core.error_system import heartbeat
                heartbeat("telegram", False)
            except Exception as exc:  # noqa: BLE001 - telemetry never breaks sends
                record_failure('telegram.TelegramAdapter.send#2', exc)
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

    async def _send_with_flood_retry(self, fn: Any, what: str) -> Any:
        """Run one MTProto send, honoring ONE flood wait.

        Telegram answers bursts with FloodWaitError(seconds) — the native
        rate-limit signal. We sleep exactly that long (capped at
        ``flood_max_wait_s``) and retry once; a second flood means the
        account is genuinely throttled and the send fails honestly instead
        of parking the thread forever.
        """
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - check for the native signal
            wait = _flood_wait_seconds(exc)
            if wait <= 0:
                raise
            if wait > self.flood_max_wait_s:
                _log.warning(
                    "telegram %s: flood wait %ds exceeds cap %.0fs — failing fast",
                    what, wait, self.flood_max_wait_s)
                raise
            _log.info("telegram %s: flood wait %ds — sleeping then retrying once",
                      what, wait)
            await asyncio.sleep(min(wait, self.flood_max_wait_s))
            return await fn()

    async def _send_async(self, client: Any, chat: ChatRef, text: str, reply_to: str,
                          parse_mode: str = "",
                          reply_keyboard: list[list[str]] | None = None) -> Any:
        # Try to use cached input entity first (most reliable for DMs)
        entity = self._input_entity_cache.get(chat.chat_id)
        if entity is None:
            # Fall back to resolution
            entity = await self._resolve(chat)
        kwargs: dict[str, Any] = {}
        if reply_to:
            kwargs["reply_to"] = int(reply_to)
        if parse_mode in ("html", "HTML", "md", "markdown"):
            kwargs["parse_mode"] = parse_mode
        kwargs.update(self._thread_kwargs(chat))
        # Persistent bottom reply keyboard (Telethon ReplyKeyboardMarkup).
        # Guarded import + best-effort: if Telethon lacks the type, the
        # message still goes out, just without the keyboard. resize +
        # single_use=False are the universally supported fields; the
        # Bot API side adds is_persistent explicitly.
        markup = None
        if reply_keyboard:
            try:
                from telethon.tl.types import (
                    KeyboardButton, ReplyKeyboardMarkup)
                markup = ReplyKeyboardMarkup(
                    rows=[[KeyboardButton(text=t) for t in row]
                          for row in reply_keyboard if row],
                    resize=True, single_use=False)
            except Exception:  # noqa: BLE001 - keyboard is cosmetic
                _log.debug("telegram: could not build reply keyboard",
                           exc_info=True)
        chunks = _chunk_text(text, for_platform("telegram").max_text)
        result = None
        plain_fallback = False
        for i, chunk in enumerate(chunks):
            # The keyboard goes on the last chunk only — it replaces the
            # chat's current keyboard, so an earlier chunk carrying it
            # would be immediately superseded anyway.
            if markup is not None and i == len(chunks) - 1:
                kwargs["buttons"] = markup
            if plain_fallback:
                from .platforms import strip_html

                chunk = strip_html(chunk)

            async def _one(body: str = chunk) -> Any:
                try:
                    return await client.send_message(entity, body, **kwargs)
                except TypeError:  # older Telethon without message_thread_id
                    thread = kwargs.pop("message_thread_id", None)
                    if thread is not None and not kwargs.get("reply_to"):
                        kwargs["reply_to"] = thread
                    return await client.send_message(entity, body, **kwargs)

            try:
                result = await self._send_with_flood_retry(_one, "send_message")
            except Exception as exc:  # noqa: BLE001 - classified below
                if not ("parse_mode" in kwargs and not plain_fallback
                        and _is_entity_parse_error(exc)):
                    raise
                # Telethon rejected the markup: retry this chunk as
                # PLAIN text (tags stripped) and keep the rest of the
                # message plain — a formatting failure must never eat
                # the message (mirrors the bot adapter's fallback).
                _log.warning(
                    "telegram: entity parse rejected (%s) — retrying "
                    "chunk %d/%d as plain text", exc, i + 1, len(chunks))
                plain_fallback = True
                kwargs.pop("parse_mode", None)
                from .platforms import strip_html

                result = await self._send_with_flood_retry(
                    lambda: _one(strip_html(chunk)), "send_message")
            # Only the first chunk carries the reply/thread reference.
            kwargs.pop("reply_to", None)
            kwargs.pop("message_thread_id", None)
            kwargs.pop("buttons", None)
        self._remember_sent(getattr(result, "id", None))  # self-chat loop guard
        return result

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "",
                   parse_mode: str = "") -> SendResult:
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
            if getattr(media, "voice_note", False):
                kwargs["voice_note"] = True
            # F7: the caption's parse mode rides Telethon's own
            # parser (same accepted spellings as send()).
            if parse_mode in ("html", "HTML", "md", "markdown"):
                kwargs["parse_mode"] = parse_mode

            async def _one(cap: str = caption, **overrides: Any) -> Any:
                call_kwargs = {**kwargs, **overrides}
                try:
                    result = await client.send_file(
                        entity, media.path, caption=cap or None, **call_kwargs
                    )
                except TypeError:  # older Telethon without message_thread_id
                    call_kwargs.pop("message_thread_id", None)
                    result = await client.send_file(entity, media.path, caption=cap or None)
                self._remember_sent(getattr(result, "id", None))  # self-chat loop guard
                return result

            try:
                return await self._send_with_flood_retry(_one, "send_file")
            except Exception as exc:  # noqa: BLE001 - classified below
                if not ("parse_mode" in kwargs
                        and _is_entity_parse_error(exc)):
                    raise
                # The caption's markup was rejected: retry once as
                # PLAIN text (tags stripped, no parse mode) — mirrors
                # send()'s fallback; the file itself must still land.
                _log.warning(
                    "telegram: caption entity parse rejected (%s) — "
                    "retrying send_file as plain text", exc)
                from .platforms import strip_html

                return await self._send_with_flood_retry(
                    lambda: _one(strip_html(caption), parse_mode=None),
                    "send_file")

        try:
            result = self._run_on_loop(_do())
            return SendResult(ok=True, platform=self.name,
                              message_id=str(getattr(result, "id", "")),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def edit_media(self, chat: ChatRef, message_id: str, media: MediaRef,
                   *, caption: str = "") -> SendResult:
        """In-place media swap via Telethon ``edit_message`` (the live-view
        path: one message updated instead of a photo per action)."""
        client = self._client
        if client is None:
            return SendResult(ok=False, platform=self.name, error="not connected")
        started = time.perf_counter()

        async def _do() -> Any:
            entity = self._input_entity_cache.get(chat.chat_id)
            if entity is None:
                entity = await self._resolve(chat)
            # Editing keeps the message in place; no thread routing needed.
            result = await client.edit_message(
                entity, int(message_id), file=media.path,
                text=caption or None)
            self._remember_sent(getattr(result, "id", None))
            return result

        try:
            result = self._run_on_loop(_do())
            return SendResult(ok=True, platform=self.name,
                              message_id=str(getattr(result, "id", message_id)),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def typing(self, chat: ChatRef, seconds: float = 3.0,
             action: str = "typing") -> bool:
        """Chat action indicator: "typing" or "recording" (voice)."""
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
            from telethon.tl.types import (SendMessageRecordAudioAction,
                                           SendMessageTypingAction)
            _action = (SendMessageRecordAudioAction()
                       if action == "recording" else SendMessageTypingAction())

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
                    await client(SetTypingRequest(entity, _action))
                    sent_count += 1
                    _log.debug("telegram typing: SetTypingRequest succeeded for chat_id=%s (count=%d)", 
                              chat.chat_id, sent_count)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("telegram typing: SetTypingRequest failed for chat_id=%s: %s, trying send_action", 
                                chat.chat_id, exc)
                    # Fallback: try send_action with string
                    try:
                        await client.send_action(entity, "record-audio" if action == "recording" else "typing")
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

    # ── group admin power mode ─────────────────────────────────────
    # Full admin surface for when the account is an admin. Every method
    # is owner-gated at the tool layer (social_gate) — the adapter is a
    # dumb pipe, the enforcement lives in code above it.

    def _admin(self, coro: Any, what: str, timeout: float = 30.0) -> dict[str, Any]:
        """Run an admin coroutine, returning an honest result dict."""
        try:
            self._run_on_loop(coro, timeout=timeout)
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{what} failed: {exc}"}

    async def _pin_coro(self, chat: ChatRef, message_id: str, silent: bool) -> None:
        from telethon.tl import functions
        entity = await self._resolve(chat)
        await self._client(functions.messages.UpdatePinnedMessageRequest(
            peer=entity, id=int(message_id), silent=silent))

    def admin_pin(self, chat: ChatRef, message_id: str, silent: bool = True) -> dict[str, Any]:
        """Pin a message in a group/channel."""
        return self._admin(self._pin_coro(chat, message_id, silent), "pin")

    async def _unpin_coro(self, chat: ChatRef) -> None:
        from telethon.tl import functions
        entity = await self._resolve(chat)
        await self._client(functions.messages.UpdatePinnedMessageRequest(
            peer=entity, id=0, silent=True))

    def admin_unpin(self, chat: ChatRef) -> dict[str, Any]:
        """Unpin the pinned message."""
        return self._admin(self._unpin_coro(chat), "unpin")

    async def _ban_coro(self, chat: ChatRef, user_id: str) -> None:
        from telethon.tl import functions
        from telethon.tl.types import ChatBannedRights
        entity = await self._resolve(chat)
        user = await self._client.get_entity(int(user_id))
        await self._client(functions.channels.EditBannedRequest(
            channel=entity, participant=user,
            banned_rights=ChatBannedRights(
                until_date=None, view_messages=True)))

    def admin_ban(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Ban a user from a group/channel."""
        return self._admin(self._ban_coro(chat, user_id), "ban")

    async def _unban_coro(self, chat: ChatRef, user_id: str) -> None:
        from telethon.tl import functions
        from telethon.tl.types import ChatBannedRights
        entity = await self._resolve(chat)
        user = await self._client.get_entity(int(user_id))
        await self._client(functions.channels.EditBannedRequest(
            channel=entity, participant=user,
            banned_rights=ChatBannedRights(until_date=None)))

    def admin_unban(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Unban a user."""
        return self._admin(self._unban_coro(chat, user_id), "unban")

    async def _promote_coro(self, chat: ChatRef, user_id: str) -> None:
        from telethon.tl import functions
        from telethon.tl.types import ChatAdminRights
        entity = await self._resolve(chat)
        user = await self._client.get_entity(int(user_id))
        await self._client(functions.channels.EditAdminRequest(
            channel=entity, user_id=user,
            admin_rights=ChatAdminRights(
                change_info=True, post_messages=True,
                edit_messages=True, delete_messages=True,
                ban_users=True, invite_users=True,
                pin_messages=True, add_admins=False,
                manage_call=True),
            rank="admin"))

    def admin_promote(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Promote a user to admin."""
        return self._admin(self._promote_coro(chat, user_id), "promote")

    async def _demote_coro(self, chat: ChatRef, user_id: str) -> None:
        from telethon.tl import functions
        from telethon.tl.types import ChatAdminRights
        entity = await self._resolve(chat)
        user = await self._client.get_entity(int(user_id))
        await self._client(functions.channels.EditAdminRequest(
            channel=entity, user_id=user,
            admin_rights=ChatAdminRights(), rank=""))

    def admin_demote(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Demote an admin."""
        return self._admin(self._demote_coro(chat, user_id), "demote")

    async def _invite_link_coro(self, chat: ChatRef) -> str:
        from telethon.tl import functions
        entity = await self._resolve(chat)
        result = await self._client(
            functions.messages.ExportChatInviteRequest(peer=entity))
        return getattr(result, "link", "")

    def admin_invite_link(self, chat: ChatRef) -> dict[str, Any]:
        """Get (or create) the group's invite link."""
        try:
            link = self._run_on_loop(self._invite_link_coro(chat), timeout=30.0)
            return {"ok": True, "link": link}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"invite link failed: {exc}"}

    async def _my_rights_coro(self, chat: ChatRef) -> dict[str, Any]:
        """What admin rights WE have in this chat. Honest capability check."""
        from telethon.tl import functions
        entity = await self._resolve(chat)
        me = await self._client.get_me()
        try:
            result = await self._client(functions.channels.GetParticipantRequest(
                channel=entity, participant=me))
            participant = getattr(result, "participant", None)
            rights = getattr(participant, "admin_rights", None)
            if rights is None:
                return {"is_admin": False, "rights": {}}
            return {
                "is_admin": True,
                "rights": {
                    k: bool(v) for k, v in vars(rights).items()
                    if not k.startswith("_")
                },
            }
        except Exception as exc:
            record_failure('telegram.TelegramAdapter._my_rights_coro', exc)
            return {"is_admin": False, "rights": {}}

    def admin_my_rights(self, chat: ChatRef) -> dict[str, Any]:
        """Check our own admin rights in a chat before acting."""
        try:
            return self._run_on_loop(self._my_rights_coro(chat), timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"rights check failed: {exc}"}

    async def _member_role_coro(self, chat: ChatRef,
                                user_id: str) -> dict[str, Any]:
        """One member's participant type in a group/channel (MTProto)."""
        from telethon.tl import functions
        entity = await self._resolve(chat)
        try:
            user = await self._client.get_entity(int(user_id))
        except Exception as exc:
            record_failure('telegram.TelegramAdapter._member_role_coro', exc)
            return {"ok": False, "error": "user not found"}
        result = await self._client(functions.channels.GetParticipantRequest(
            channel=entity, participant=user))
        participant = getattr(result, "participant", None)
        ptype = type(participant).__name__ if participant else ""
        return {"ok": True, "participant_type": ptype}

    def admin_member_mtproto(self, chat: ChatRef,
                             user_id: str) -> dict[str, Any]:
        """One member's role via MTProto (for group_roles resolution)."""
        try:
            return self._run_on_loop(
                self._member_role_coro(chat, user_id), timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"member lookup failed: {exc}"}

    async def _restrict_coro(self, chat: ChatRef, user_id: str,
                             minutes: int = 60) -> None:
        from telethon.tl import functions
        from telethon.tl.types import ChatBannedRights
        import time as _t
        entity = await self._resolve(chat)
        user = await self._client.get_entity(int(user_id))
        until = int(_t.time()) + max(60, minutes * 60) if minutes > 0 else None
        await self._client(functions.channels.EditBannedRequest(
            channel=entity, participant=user,
            banned_rights=ChatBannedRights(
                until_date=until, send_messages=True)))

    def admin_restrict(self, chat: ChatRef, user_id: str,
                       minutes: int = 60) -> dict[str, Any]:
        """Mute a user (restrict sending) for N minutes. 0 = indefinite."""
        return self._admin(
            self._restrict_coro(chat, user_id, minutes), "restrict")

    async def _delete_coro(self, chat: ChatRef, message_id: str) -> None:
        from telethon.tl import functions
        entity = await self._resolve(chat)
        await self._client(functions.channels.DeleteMessagesRequest(
            channel=entity, id=[int(message_id)]))

    def admin_delete(self, chat: ChatRef, message_id: str) -> dict[str, Any]:
        """Delete a message (admin)."""
        return self._admin(self._delete_coro(chat, message_id), "delete")

    async def _members_coro(self, chat: ChatRef,
                            limit: int = 100) -> list[dict[str, Any]]:
        entity = await self._resolve(chat)
        out: list[dict[str, Any]] = []
        async for u in self._client.iter_participants(
                entity, limit=max(1, min(500, limit))):
            out.append({
                "id": str(getattr(u, "id", "")),
                "name": " ".join(filter(None, [
                    getattr(u, "first_name", ""),
                    getattr(u, "last_name", "")])).strip(),
                "username": getattr(u, "username", "") or "",
                "bot": bool(getattr(u, "bot", False)),
            })
        return out

    def admin_members(self, chat: ChatRef,
                      limit: int = 100) -> dict[str, Any]:
        """List group/channel members."""
        try:
            members = self._run_on_loop(
                self._members_coro(chat, limit), timeout=60.0)
            return {"ok": True, "members": members,
                    "count": len(members)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"member list failed: {exc}"}

    async def _forum_topic_coro(self, chat: ChatRef, title: str) -> int:
        from telethon.tl import functions
        entity = await self._resolve(chat)
        result = await self._client(functions.channels.CreateForumTopicRequest(
            channel=entity, title=title[:128], random_id=__import__(
                "random").randrange(1, 2**63)))
        topic = result.updates[0] if getattr(result, "updates", None) else None
        return int(getattr(topic, "id", 0) or 0)

    def admin_forum_topic(self, chat: ChatRef,
                          title: str) -> dict[str, Any]:
        """Create a forum topic (supergroups with topics enabled). MTProto."""
        try:
            tid = self._run_on_loop(
                self._forum_topic_coro(chat, title), timeout=30.0)
            return {"ok": True, "topic_id": tid}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"forum topic failed: {exc}"}


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
    ``getFile``. Sending splits at the PlatformLimits text cap.
    """

    name = "telegram-bot"
    supported_kinds = (ChatKind.DM, ChatKind.GROUP, ChatKind.CHANNEL)

    #: Bot API text limit per message.
    MAX_TEXT = for_platform("telegram-bot").max_text

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
        lock_path: str = "data/telegram-bot.lock",
        # ── two-bot support ──────────────────────────────────────────
        # ``adapter_name`` is the adapter's identity (``"telegram-bot"``
        # for the gated bot, ``"telegram-personal"`` for the personal
        # bot). It keys ChatRef.platform, chat keys, media routing and
        # the gateway's adapter map — same class, per-adapter config
        # as data, not code.
        adapter_name: str | None = None,
        # Whether this adapter's chats may carry partner (romantic)
        # context in the brain: True on the personal bot (every chat
        # on it is the owner), False on the gated bot (groups never
        # carry it; an owner DM on the gated bot still does — see
        # brain._generate_and_persist). Stamped on every inbound
        # message's meta so the brain never has to guess the adapter.
        partner_context: bool = False,
        # Personal-bot gate: when True, ONLY the owner may use this
        # bot. Non-owners get one honest denial sent directly (no
        # brain call, no credit spend) and the message never reaches
        # the pipeline. The personal bot is NOT public.
        owner_only: bool = False,
        # Owner recognition for the owner_only gate: sender IDs
        # (``partner.owner_sender_ids``) and full chat keys
        # (``partner.owner_chats``, e.g. ``"telegram-personal:123"``).
        owner_sender_ids: Any = (),
        owner_chat_keys: Any = (),
        # Honest denial for non-owners on the personal bot.
        deny_text: str = (
            "This is a private bot — only the owner can chat here."
        ),
    ) -> None:
        super().__init__(media_dir=media_dir)
        if not token:
            raise ValidationError("telegram bot token is required")
        self.token = token
        # Instance identity: shadows the class-level ``name`` so one
        # class serves both BotFather bots with distinct chat keys.
        self.name = (adapter_name or "telegram-bot").strip().lower() \
            or "telegram-bot"
        self.partner_context = bool(partner_context)
        self.owner_only = bool(owner_only)
        self.owner_sender_ids = {str(v).strip() for v in (owner_sender_ids or ())
                                 if str(v).strip()}
        self.owner_chat_keys = {str(v).strip() for v in (owner_chat_keys or ())
                                if str(v).strip()}
        self.deny_text = deny_text or (
            "This is a private bot — only the owner can chat here.")
        self.chat_allow = {c.strip() for c in chat_allow.split(",") if c.strip()}
        self.media_in_groups = media_in_groups
        self.media_max_mb = media_max_mb
        self.poll_timeout = poll_timeout
        self._session = session
        self._offset = 0
        # Bot ID straight from the token ("<bot_id>:<secret>") — available
        # without a getMe round-trip, so the two-bot router and the
        # userbot loop guard can identify this bot before it ever polls.
        # preflight() overwrites it with the authoritative getMe id.
        try:
            self._bot_id: int | None = int(str(token).split(":", 1)[0])
        except (TypeError, ValueError):
            self._bot_id = None
        self._bot_username = ""
        # (chat_id, message_id) already delivered — getUpdates offset
        # tracking normally prevents replays, but a restarted poll loop
        # can re-fetch recent updates; guarantee at-most-once here too.
        self._seen_inbound: set[tuple[str, str]] = set()
        # Single getUpdates consumer: lockfile path (see single_consumer).
        self._lock_path = lock_path
        # Last inbound /command per chat — the origin command for
        # origin-derived buttons on a command result (Retry re-runs
        # it). Only trusted when the last inbound message in the chat
        # WAS that command (see ``_last_inbound_was_command``): a
        # scheduled push must never inherit buttons from a stale
        # command the owner typed earlier.
        self._last_command: dict[str, str] = {}
        # Whether the last inbound message in each chat was a
        # ``/command`` (True) or plain chat (False). Origin-derived
        # buttons only attach when this is True — a chat reply never
        # inherits Retry for an old command.
        self._last_inbound_was_command: dict[str, bool] = {}
        # Progress-edit throttle (edit-in-place ~1/3s, only on change).
        from .progress_throttle import ProgressThrottler
        self._progress = ProgressThrottler()

    def _note_inbound_seen(self, chat_id: str, message_id: str) -> bool:
        """Record an inbound Bot API message id; True if already seen.

        getUpdates offset tracking normally prevents replays, but a
        restarted poll loop can re-fetch recent updates. Same at-most-once
        guarantee as the userbot adapter.
        """
        key = (str(chat_id), str(message_id))
        if key in self._seen_inbound:
            return True
        self._seen_inbound.add(key)
        if len(self._seen_inbound) > 5000:
            for _ in range(2500):
                self._seen_inbound.pop()
        return False

    # ── HTTP ────────────────────────────────────────────────────────────────
    def _sess(self) -> Any:
        """The adapter's ONE session contract: the core ``HttpClient``
        surface (``post_json`` / ``get`` / ``post_multipart``).

        Every Bot API call — JSON posts AND media uploads — goes through
        that contract and nothing else. Media sends used to call a
        requests-style ``session.post(url, data=, files=)`` here; no
        production factory injects a session, so the session was always
        the core HttpClient, which has no ``.post`` — every media send
        died with ``'HttpClient' object has no attribute 'post'`` while
        text sends (``post_json``) worked. An injected ``session`` must
        implement the same HttpClient surface.
        """
        if self._session is None:
            from ...core.http import HttpClient

            self._session = HttpClient()
        return self._session

    def _api_upload(self, method: str, fields: dict[str, Any],
                    file_field: str, path: Path, *,
                    file_name: str = "") -> dict[str, Any]:
        """One Bot API multipart upload, via ``post_multipart`` only.

        ``fields`` are the form fields (chat_id, caption, …); the file
        rides as ``file_field`` with an explicit multipart filename
        (Bot API decides media type from the filename extension).
        Returns the API ``result`` object; raises ValidationError with
        Telegram's description when the API rejects the upload.
        """
        url = _BOT_API.format(token=self.token, method=method)
        resp = self._sess().post_multipart(
            url, fields,
            files=[(file_field, path, _mime_for(str(path)),
                    file_name or path.name)],
            timeout=120)
        result = resp.json()
        if not result.get("ok"):
            raise ValidationError(result.get("description", "send failed"))
        return result.get("result") or {}

    #: how many times _api replays a request after a 429 before giving up
    _API_429_MAX_RETRIES = 2

    def _api(self, method: str, **params: Any) -> Any:
        url = _BOT_API.format(token=self.token, method=method)
        for attempt_429 in range(self._API_429_MAX_RETRIES + 1):
            resp = self._sess().post_json(url, params, timeout=self.poll_timeout + 10)
            try:
                data = resp.json()
            except ValueError:
                raise ValidationError(f"telegram bot api: non-JSON reply from {method}")
            if data.get("ok"):
                return data["result"]
            # Production behavior (Telegram Bot API): on 429 the response
            # carries parameters.retry_after — hold for exactly that long
            # (plus a small buffer) then replay, bounded attempts.  Racing
            # ahead would only earn more 429s.  Every other error (400/403
            # like "bot was blocked", chat-not-found, …) fails fast — a
            # retry can never fix those.
            if data.get("error_code") == 429 and attempt_429 < self._API_429_MAX_RETRIES:
                retry_after = (data.get("parameters") or {}).get("retry_after", 1)
                try:
                    wait = min(max(int(retry_after), 1), 60) + 1
                except (TypeError, ValueError):
                    wait = 2
                _log.warning("telegram bot api: %s rate-limited (429), "
                             "retrying in %ds (attempt %d/%d)",
                             method, wait, attempt_429 + 1,
                             self._API_429_MAX_RETRIES)
                time.sleep(wait)
                continue
            raise ValidationError(
                f"telegram bot api: {method} failed: {data.get('description', 'unknown')}"
            )
        # unreachable — the loop either returns or raises
        raise ValidationError(f"telegram bot api: {method} failed: rate limited")

    # ── lifecycle ───────────────────────────────────────────────────────────
    def preflight(self) -> None:
        me = self._api("getMe")
        self._bot_id = me.get("id")
        self._bot_username = me.get("username", "")
        _log.info("telegram bot connected as @%s (id=%s)",
                  self._bot_username, self._bot_id)

    def run(self, handler: IncomingHandler) -> None:
        from .single_consumer import ConsumerAlreadyRunning, SingleConsumerLock

        lock = SingleConsumerLock(self._lock_path)
        if not lock.acquire():
            info = lock.holder() or {}
            raise ConsumerAlreadyRunning(
                "another getUpdates consumer is already running "
                f"(pid={info.get('pid')}); refusing to start — a second "
                "poll loop would silently steal updates / hit 409 Conflict")
        try:
            self._run_locked(handler)
        finally:
            lock.release()

    def _run_locked(self, handler: IncomingHandler) -> None:
        try:
            self.preflight()
        except Exception as exc:  # noqa: BLE001 - fail fast, gateway logs the skip
            raise ValueError(f"telegram-bot unavailable: {exc}") from exc
        # Drop whatever piled up while we were away; only fresh mail.
        try:
            pending = self._api("getUpdates", offset=-1, timeout=1)
            if pending:
                # Int-only ids: a malformed pending update (missing /
                # non-numeric update_id) used to raise here and kill
                # startup. Ignore those entries; valid ids still advance.
                ids = [
                    u["update_id"]
                    for u in pending
                    if isinstance(u, dict)
                    and isinstance(u.get("update_id"), int)
                    and not isinstance(u.get("update_id"), bool)
                ]
                if ids:
                    self._offset = max(ids) + 1
                _log.info("telegram-bot: skipped %d stale updates", len(pending))
        except Exception as exc:  # noqa: BLE001 - non-fatal, polling continues
            _log.debug("telegram-bot: stale-skip failed: %s", exc)
        poll_fail_delay = 5.0
        while not self._stopped.is_set():
            try:
                updates = self._api("getUpdates", offset=self._offset,
                                    timeout=self.poll_timeout,
                                    allowed_updates=["message", "edited_message",
                                                     "channel_post", "callback_query",
                                                     "inline_query"])
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
            if not isinstance(updates, list):
                _log.warning("telegram-bot: getUpdates returned %s, not a list — skipping batch",
                             type(updates).__name__)
                continue
            for update in updates:
                # Per-update protection: one malformed update used to raise
                # out of this loop and kill the poll thread — the bot went
                # deaf with no further log. Now it is logged and skipped.
                try:
                    self._dispatch_update(update, handler)
                except Exception as exc:  # noqa: BLE001 - per-update isolation
                    _log.warning("telegram-bot: dropping malformed update %r: %s",
                                 type(update).__name__, exc)

    def _dispatch_update(self, update: Any, handler: IncomingHandler) -> None:
        """Route one polled update to its handler.

        Validates the update's shape first: external update payloads are
        untrusted, and every sub-handler assumes dicts. Malformed updates
        are logged and dropped, never raised.
        """
        try:
            if not isinstance(update, dict):
                _log.warning("telegram-bot: dropping non-dict update %r",
                             type(update).__name__)
                return
            uid = update.get("update_id")
            if isinstance(uid, int) and not isinstance(uid, bool):
                self._offset = max(self._offset, uid + 1)
            # Callback queries (inline button taps) get their own path.
            if "callback_query" in update:
                query = update["callback_query"]
                if not isinstance(query, dict):
                    _log.warning("telegram-bot: dropping non-dict callback_query %r",
                                 type(query).__name__)
                    return
                self._handle_callback(query, handler)
                return
            # Inline queries (@BotName <query> in any chat).
            if "inline_query" in update:
                query = update["inline_query"]
                if not isinstance(query, dict):
                    _log.warning("telegram-bot: dropping non-dict inline_query %r",
                                 type(query).__name__)
                    return
                self._handle_inline_query(query)
                return
            message = self._convert(update)
            if message is not None:
                # At-most-once: a restarted poll loop can re-fetch
                # updates the offset already passed. Skip replays by
                # (chat_id, message_id) so the brain/feed see each
                # message exactly once.
                if message.message_id and self._note_inbound_seen(
                    message.chat.chat_id, message.message_id
                ):
                    _log.debug(
                        "telegram-bot: skipping replayed message %s in %s",
                        message.message_id, message.chat.chat_id,
                    )
                    return
                self._deliver(handler, message)
        except Exception as exc:  # noqa: BLE001 - per-update isolation
            _log.warning("telegram-bot: dropping malformed update: %s", exc)

    # ── inbound ───────────────────────────────────────────────────────────
    # ── two-bot: owner gate + partner-context stamping ──────────────
    def _is_owner_sender(self, sender_id: Any, chat_id: str) -> bool:
        """True iff this sender/chat is the owner (personal-bot gate).

        Sender-ID match covers DMs (chat_id == sender_id there) and
        groups the owner writes in; chat-key match covers owner chats
        registered as ``"<adapter>:<chat_id>"`` in
        ``partner.owner_chats``. Either is enough — both are data from
        settings, never code.
        """
        if str(sender_id or "").strip() in self.owner_sender_ids:
            return True
        if f"{self.name}:{chat_id}" in self.owner_chat_keys:
            return True
        return False

    def _deny_non_owner(self, chat_id: str) -> None:
        """Send the honest personal-bot denial directly (no brain, no
        credit spend). Best-effort: a failed denial must never raise
        into the poll loop. Bot API getUpdates never delivers the bot's
        own messages back, so this cannot loop."""
        try:
            self._api("sendMessage", chat_id=int(chat_id),
                      text=self.deny_text)
        except Exception as exc:  # noqa: BLE001 - denial is best-effort
            _log.debug("telegram-personal: denial send failed: %s", exc)

    def _convert(self, update: dict[str, Any]) -> ChatMessage | None:
        if not isinstance(update, dict):
            return None
        msg = update.get("message") or update.get("edited_message") \
            or update.get("channel_post")
        if not isinstance(msg, dict) or not msg:
            return None
        sender = msg.get("from") or {}
        if not isinstance(sender, dict):
            sender = {}
        if sender.get("is_bot"):
            return None  # loop guard: never answer other bots (or our echoes)
        chat = msg.get("chat") or {}
        if not isinstance(chat, dict):
            chat = {}
        chat_id = str(chat.get("id", ""))
        if not chat_id:
            return None
        # Personal bot (owner_only) FIRST, before the chat_allow filter:
        # its allowlist IS the owner, and a stranger must get the honest
        # denial — not a silent drop. Only fresh messages get a denial
        # (an edited_message re-denial would spam on every edit).
        if self.owner_only and not self._is_owner_sender(
                sender.get("id", ""), chat_id):
            _log.info("%s: denying non-owner sender %s in chat %s",
                      self.name, sender.get("id", "?"), chat_id)
            if "message" in update:
                self._deny_non_owner(chat_id)
            return None
        if self.chat_allow and chat_id not in self.chat_allow:
            _log.info("telegram-bot: ignoring chat %s — not in allowlist", chat_id)
            return None
        chat_type = chat.get("type", "")
        if not isinstance(chat_type, str):
            chat_type = ""
        kind = _BOT_CHAT_TYPES.get(chat_type, ChatKind.DM)
        text = msg.get("text") or msg.get("caption") or ""
        if not isinstance(text, str):
            text = str(text)
        # Track the origin per chat: whether the last inbound message
        # WAS a /command, and if so which one. Origin-derived buttons
        # (Retry) only attach to a reply when the last inbound message
        # was that command — a chat reply never inherits buttons from
        # an old command, and a scheduled push never inherits any.
        _is_cmd = text.strip().startswith("/")
        self._last_inbound_was_command[chat_id] = _is_cmd
        if _is_cmd:
            self._last_command[chat_id] = text.strip()
        mentioned = False
        if kind != ChatKind.DM and self._bot_username:
            entities = msg.get("entities") or msg.get("caption_entities") or []
            if not isinstance(entities, list):
                entities = []
            for ent in entities:
                # External entity entries are untrusted: non-dict,
                # non-int, or out-of-bounds offset/length used to raise
                # (TypeError / IndexError via slicing semantics) here.
                # Skip bad entries; a bad entity never drops the message.
                if not isinstance(ent, dict):
                    continue
                if ent.get("type") != "mention":
                    continue
                offset, length = ent.get("offset"), ent.get("length")
                if (not isinstance(offset, int) or isinstance(offset, bool)
                        or not isinstance(length, int) or isinstance(length, bool)):
                    continue
                if offset < 0 or length < 0 or offset + length > len(text):
                    continue
                piece = text[offset:offset + length]
                if piece.lower() == f"@{self._bot_username.lower()}":
                    mentioned = True
                    break
        media = self._inbound_media(msg, kind)
        reply_to = ""
        replied = msg.get("reply_to_message") or {}
        if not isinstance(replied, dict):
            replied = {}
        if replied.get("message_id"):
            reply_to = str(replied["message_id"])
        # /sham support: stash the replied-to audio/voice file_id so the
        # command can download it on demand. The Bot API includes the full
        # replied message object (with file ids), but there's no
        # get-by-message-id — this is the only chance to grab it.
        replied_audio_file_id = ""
        for _key in ("voice", "audio"):
            _rm = replied.get(_key) or {}
            if not isinstance(_rm, dict):
                continue
            if _rm.get("file_id"):
                replied_audio_file_id = str(_rm["file_id"])
                break
        meta = {"update_id": update.get("update_id"),
                "partner_context": self.partner_context,
                "bot_role": "personal" if self.owner_only else "gated"}
        if replied_audio_file_id:
            meta["replied_audio_file_id"] = replied_audio_file_id
        who = sender.get("username") or sender.get("first_name") or str(sender.get("id", ""))
        if not isinstance(who, str):
            who = str(who)
        # Stable numeric Telegram user id — game identity keys off this
        # so username/display-name changes don't split profiles.
        who_id = str(sender.get("id", "") or "")
        who_username = str(sender.get("username") or "")
        _title = chat.get("title", "") or who
        if not isinstance(_title, str):
            _title = str(_title)
        return ChatMessage(
            chat=ChatRef(platform=self.name, chat_id=chat_id, kind=kind,
                         title=_title, peer=who),
            incoming=True,
            text=text,
            sender=who,
            sender_id=who_id,
            sender_username=who_username,
            media=media,
            reply_to=reply_to,
            mentioned=mentioned,
            ts=_safe_ts(msg.get("date", time.time())),
            message_id=str(msg.get("message_id", "")),
            meta=meta,
        )

    # ── inline buttons ──────────────────────────────────────────────────
    def _sign_callback(self, data: str) -> str:
        """HMAC-sign callback data to prevent spoofing."""
        from .tgbot_buttons import check_callback_data
        check_callback_data(data)
        sig = hmac.new(self.token.encode(), data.encode(),
                       hashlib.sha256).hexdigest()[:16]
        return f"{data}|{sig}"

    def _verify_callback(self, signed: str) -> str | None:
        """Verify signature, return the original data or None."""
        if not isinstance(signed, str):
            # Non-string callback data (external input) used to raise
            # AttributeError on ``"|" not in signed`` — reject it.
            return None
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
        if not isinstance(query, dict):
            _log.warning("telegram-bot: dropping non-dict callback_query %r",
                         type(query).__name__)
            return
        query_id = query.get("id", "")
        if not isinstance(query_id, str):
            query_id = str(query_id)
        sender = query.get("from") or {}
        if not isinstance(sender, dict):
            sender = {}
        if sender.get("is_bot"):
            self._answer_callback(query_id)
            return
        # Owner gate: only the owner can trigger button actions — EXCEPT
        # the commands a non-owner could already trigger by TYPING them.
        # A button tap must never grant more than typing the same
        # command: game + public commands flow to the normal pipeline
        # gates (game dispatch runs games; the public-bot gate redirects
        # owner-only kinds); owner-only buttons stay owner-gated here.
        # "arena" is the one deliberate exception: its button sits on
        # public game messages, so it flows to the public-bot gate's
        # friendly redirect instead of dying here as "Not authorized".
        msg = query.get("message") or {}
        if not isinstance(msg, dict):
            msg = {}
        chat = msg.get("chat") or {}
        if not isinstance(chat, dict):
            chat = {}
        chat_id = str(chat.get("id", ""))
        signed = query.get("data", "")
        data = self._verify_callback(signed)
        if data is None:
            _log.warning("telegram-bot: ignoring callback with bad signature")
            self._answer_callback(query_id, text="Invalid button")
            return
        # Personal bot: every callback must come from the owner — a
        # stranger's button tap dies here, before any pipeline spend.
        if self.owner_only and not self._is_owner_sender(
                sender.get("id", ""), chat_id):
            _log.info("%s: denying non-owner callback from %s in chat %s",
                      self.name, sender.get("id", "?"), chat_id)
            self._answer_callback(query_id, text="Not authorized")
            return
        if self.chat_allow and chat_id not in self.chat_allow:
            from .control import GAME_COMMANDS, PUBLIC_COMMANDS
            kind = (data or "").lstrip("/").split(None, 1)
            kind = kind[0].lower() if kind and kind[0] else ""
            if (kind != "game" and kind not in GAME_COMMANDS
                    and kind not in PUBLIC_COMMANDS and kind != "arena"):
                _log.info("telegram-bot: ignoring callback from %s — not in allowlist",
                          chat_id)
                self._answer_callback(query_id, text="Not authorized")
                return
        self._answer_callback(query_id)
        # Deliver as a synthetic "/" command so the normal command
        # pipeline (owner gating, parsing, dispatch) handles it.
        chat_type = chat.get("type", "")
        if not isinstance(chat_type, str):
            chat_type = ""
        kind = _BOT_CHAT_TYPES.get(chat_type, ChatKind.DM)
        who = sender.get("username") or sender.get("first_name") or str(sender.get("id", ""))
        if not isinstance(who, str):
            who = str(who)
        _title = chat.get("title", "") or who
        if not isinstance(_title, str):
            _title = str(_title)
        _cb_text = data if data.startswith("/") else f"/{data}"
        # A button tap is a synthetic /command: the reply to it earns
        # origin-derived buttons for THAT command, not for whatever
        # the owner last typed.
        self._last_inbound_was_command[chat_id] = True
        self._last_command[chat_id] = _cb_text
        message = ChatMessage(
            chat=ChatRef(platform=self.name, chat_id=chat_id, kind=kind,
                         title=_title, peer=who),
            incoming=True,
            text=_cb_text,
            sender=who,
            sender_id=str(sender.get("id", "") or ""),
            sender_username=str(sender.get("username") or ""),
            media=[],
            reply_to="",
            mentioned=False,
            ts=time.time(),
            message_id=f"cb_{query_id}",
            meta={"callback_query": True,
                  "partner_context": self.partner_context,
                  "bot_role": "personal" if self.owner_only else "gated"},
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

    # ── inline mode (@BotName <query> in any chat) ─────────────────────
    # NOTE: inline mode must be enabled for the bot via BotFather
    # (/setinline). Until then Telegram simply never sends inline_query
    # updates — everything else keeps working.
    def _handle_inline_query(self, query: dict[str, Any]) -> None:
        """Answer an inline query with tappable command articles."""
        from .tgbot_buttons import inline_results_for

        if not isinstance(query, dict):
            _log.warning("telegram-bot: dropping non-dict inline_query %r",
                         type(query).__name__)
            return
        query_id = query.get("id", "")
        if not isinstance(query_id, str):
            query_id = str(query_id)
        if not query_id:
            return
        sender = query.get("from") or {}
        if not isinstance(sender, dict):
            sender = {}
        if sender.get("is_bot"):
            return
        qtext = query.get("query", "") or ""
        if not isinstance(qtext, str):
            qtext = str(qtext)
        try:
            results = inline_results_for(qtext)
            self._api("answerInlineQuery", inline_query_id=query_id,
                      results=results, cache_time=300, is_personal=True)
            _log.info("telegram-bot: answered inline query %r from %s",
                      qtext[:40], sender.get("username") or sender.get("id"))
        except Exception as exc:  # noqa: BLE001 - non-fatal
            _log.debug("telegram-bot: answerInlineQuery failed: %s", exc)

    def _inbound_media(self, msg: dict[str, Any], kind: str) -> list[MediaRef]:
        """Download the largest attached file, when the gates allow it."""
        if not isinstance(msg, dict):
            return []
        file_id, fkind, fname, fmime = "", "", "", ""

        def _fid(entry: Any) -> str:
            # Direct ``entry["file_id"]`` indexing used to raise
            # KeyError / TypeError on malformed media entries.
            if not isinstance(entry, dict):
                return ""
            fid = entry.get("file_id")
            return fid if isinstance(fid, str) and fid else ""

        def _size(entry: Any) -> float:
            if not isinstance(entry, dict):
                return 0
            size = entry.get("file_size", 0)
            return size if isinstance(size, (int, float)) and not isinstance(size, bool) else 0

        photo = msg.get("photo")
        if isinstance(photo, list) and photo:
            candidates = [p for p in photo if _fid(p)]
            if candidates:
                biggest = max(candidates, key=_size)
                file_id, fkind, fname = _fid(biggest), "image", "photo.jpg"
        elif _fid(msg.get("document")):
            d = msg["document"]
            file_id, fkind = _fid(d), "document"
            _fn = d.get("file_name", "file")
            fname = _fn if isinstance(_fn, str) and _fn else "file"
            # Bot API: audio sent as a file arrives as "document" with
            # mime_type audio/* — previously kind="document", mime=""
            # matched NO STT check, so the note fell through entirely.
            _mt = d.get("mime_type", "")
            fmime = _mt if isinstance(_mt, str) else ""
            if fmime.lower().startswith("audio"):
                fkind = "audio"
        elif _fid(msg.get("video")):
            file_id, fkind, fname = _fid(msg.get("video")), "video", "video.mp4"
        elif _fid(msg.get("voice")):
            # Telegram ``voice`` = a spoken voice note, NEVER a song clip.
            # It keeps its own kind + is_voice flag so downstream routing
            # sends it to STT and never lets the brain mistake it for a
            # music clip to identify. (Previously it collapsed to
            # kind="audio", indistinguishable from a music file — the
            # brain saw a bare "audio: voice.ogg" and hallucinated a
            # Shazam-style "what's the clip? upload the audio file" reply.)
            file_id, fkind, fname = _fid(msg.get("voice")), "voice", "voice.ogg"
            fmime = "audio/ogg"
        elif _fid(msg.get("audio")):
            # Telegram ``audio`` = a music/audio file (title/artist).
            # Possible music — but music ID only ever runs on an explicit
            # /sham request, never automatically.
            file_id, fkind, fname = _fid(msg.get("audio")), "audio", "audio.mp3"
            _mt = msg["audio"].get("mime_type", "")
            fmime = _mt if isinstance(_mt, str) and _mt else "audio/mpeg"
        elif _fid(msg.get("video_note")):
            file_id, fkind, fname = _fid(msg.get("video_note")), "video", "note.mp4"
        if not file_id:
            return []
        size = 0
        try:
            info = self._api("getFile", file_id=file_id)
            size = int(info.get("file_size", 0))
            path = info.get("file_path", "")
        except Exception as exc:  # noqa: BLE001 - media is best-effort
            _log.debug("telegram-bot getFile failed: %s", exc)
            # Placeholder, NOT []: the message DID carry media. Returning
            # [] would make downstream think there was no attachment at
            # all, and the brain would see an empty message. With the
            # placeholder (empty path) the STT path reports "no audio
            # file" honestly instead of hallucinating about a clip.
            return [MediaRef(path="", kind=fkind, name=fname, mime=fmime,
                             is_voice=(fkind == "voice"))]
        if not media_download_allowed(kind, size, self.media_in_groups, self.media_max_mb):
            return []
        if not path:
            return [MediaRef(path="", kind=fkind, name=fname, mime=fmime,
                             is_voice=(fkind == "voice"))]
        # fname is sender-controlled (Telegram document file_name): it
        # can contain "../", ":", or other Windows-illegal characters.
        # Sanitise before it becomes a path — the MediaRef keeps the
        # original name for display.
        orig_fname = fname
        fname = safe_filename(fname, fallback="file")
        dest = Path(self.media_dir) / f"{file_id[:16]}_{fname}"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = _BOT_FILE_API.format(token=self.token, path=path)
            resp = self._sess().get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.body)
        except Exception as exc:  # noqa: BLE001
            _log.debug("telegram-bot media download failed: %s", exc)
            return [MediaRef(path="", kind=fkind, name=orig_fname, mime=fmime,
                             is_voice=(fkind == "voice"))]
        return [MediaRef(path=str(dest), kind=fkind, name=orig_fname,
                         mime=fmime, is_voice=(fkind == "voice"))]

    def download_file(self, file_id: str, fname: str = "audio.ogg") -> str:
        """Download a Telegram file_id to the media dir. Returns the local
        path, or "" on failure. Never raises. Used by /sham for
        replied-to audio the adapter didn't download inline."""
        if not file_id:
            return ""
        try:
            info = self._api("getFile", file_id=file_id)
            path = info.get("file_path", "")
            if not path:
                return ""
            # fname is caller/sender-controlled: sanitise before it
            # becomes a path (see the download branch above).
            fname = safe_filename(fname, fallback="audio.ogg")
            dest = Path(self.media_dir) / f"{file_id[:16]}_{fname}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = _BOT_FILE_API.format(token=self.token, path=path)
            resp = self._sess().get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.body)
            return str(dest)
        except Exception as exc:  # noqa: BLE001 - best-effort
            _log.debug("telegram-bot download_file failed: %s", exc)
            return ""

    # ── outbound ──────────────────────────────────────────────────────────
    def send_long(self, chat: ChatRef, text: str, *, reply_to: str = "",
                  stem: str = "output") -> SendResult:
        """Send long output as ONE document, not N message chunks.

        The document holds the full text (nothing truncated); the
        caption carries the char count. The temp file is removed after
        the send attempt, success or failure.
        """
        from .progress_throttle import LongOutput
        from .base import MediaRef as _MediaRef

        router = LongOutput()
        tmp = Path(self.media_dir) / f".long-{time.time_ns()}-{router.document_name(stem)}"
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes((text or "").encode("utf-8"))
            return self.send_media(chat, _MediaRef(path=str(tmp), mime="text/plain",
                                                   kind="document", name=tmp.name),
                                   caption=router.caption(text))
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def edit_progress(self, chat: ChatRef, message_id: str, new_text: str) -> dict[str, Any] | None:
        """Edit a progress message in place — throttled (~1/3s, on change).

        Returns the API result, or None when the edit was suppressed
        (unchanged text / too soon) or failed. Never raises.
        """
        if not self._progress.should_edit(new_text):
            return None
        try:
            return self._api("editMessageText", chat_id=int(chat.chat_id),
                             message_id=int(message_id),
                             text=(new_text or "")[:for_platform("telegram-bot").max_text])
        except Exception as exc:  # noqa: BLE001 - progress edits are best-effort
            _log.debug("telegram-bot: edit_progress failed: %s", exc)
            return None

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             reply_keyboard: list[list[str]] | None = None,
             parse_mode: str = "") -> SendResult:
        from .adaptive_buttons import adaptive_buttons_for
        from .progress_throttle import LongOutput
        from .tgbot_buttons import personal_buttons_for

        # Long output goes out as ONE document, not 40 chunks.
        if LongOutput().route(text)[0] == "document":
            if parse_mode:
                # A .txt document renders no markup — file the plain
                # text (tags stripped), not the HTML wire form.
                from .platforms import strip_html

                text = strip_html(text)
            return self.send_long(chat, text, reply_to=reply_to)
        started = time.perf_counter()
        last_id = ""
        try:
            params_base: dict[str, Any] = {"chat_id": int(chat.chat_id)}
            if parse_mode in ("HTML", "Markdown", "MarkdownV2"):
                params_base["parse_mode"] = parse_mode
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
            # Reply keyboard wins over the inline keyboard: a message
            # carries one keyboard. The reply keyboard is the persistent
            # bottom menu (tapping a button sends its text as a message).
            keyboard = None
            if reply_keyboard:
                keyboard = {
                    "keyboard": [[{"text": t} for t in row]
                                 for row in reply_keyboard if row],
                    "resize_keyboard": True,
                    "one_time_keyboard": False,
                    "is_persistent": True,
                }
            else:
                # Inline keyboard: [[(label, callback_data), ...], ...]
                # Callback data is HMAC-signed to prevent spoofing.
                # When the caller passes no explicit buttons, derive
                # adaptive ones from the output (kind + content) — see
                # adaptive_buttons. The origin command only counts
                # when the last inbound message in this chat WAS that
                # command: a chat reply never inherits Retry for an
                # old command, and a scheduled push (which passes
                # explicit buttons via the Notifier) never reaches
                # this branch at all.
                if buttons is not None:
                    resolved = buttons
                else:
                    fresh_origin = (
                        self._last_command.get(chat.chat_id, "")
                        if self._last_inbound_was_command.get(chat.chat_id)
                        else ""
                    )
                    if self.owner_only:
                        # Personal bot only: richer keyboards (help
                        # categories, power controls, next-actions)
                        # first, then adaptive content/origin buttons.
                        # The gated bot never takes this branch.
                        resolved = (
                            personal_buttons_for(text, fresh_origin)
                            or adaptive_buttons_for(
                                origin_command=fresh_origin, text=text)
                        )
                    else:
                        resolved = adaptive_buttons_for(
                            origin_command=fresh_origin, text=text)
                if resolved:
                    keyboard = []
                    for row in resolved:
                        krow = []
                        for label, data in row:
                            krow.append({"text": label,
                                         "callback_data": self._sign_callback(data)})
                        keyboard.append(krow)
            chunks = _chunk_text(text, self.MAX_TEXT)
            plain_fallback = False
            for i, chunk in enumerate(chunks):
                params = dict(params_base)
                if plain_fallback:
                    params.pop("parse_mode", None)
                # The keyboard goes on the last chunk only — every chunk
                # carrying it would repeat the buttons N times.
                if keyboard and i == len(chunks) - 1:
                    if reply_keyboard:
                        params["reply_markup"] = keyboard
                    else:
                        params["reply_markup"] = {"inline_keyboard": keyboard}
                try:
                    result = self._api("sendMessage", text=chunk, **params)
                except Exception as exc:  # noqa: BLE001 - classified below
                    if not (parse_mode and not plain_fallback
                            and _is_entity_parse_error(exc)):
                        raise
                    # The API rejected the markup itself: retry this
                    # chunk as PLAIN text (and keep the rest of the
                    # message plain — half-formatted reads worse than
                    # plain). A formatting failure must never eat the
                    # message.
                    _log.warning(
                        "telegram-bot: entity parse rejected (%s) — "
                        "retrying chunk %d/%d as plain text",
                        exc, i + 1, len(chunks))
                    plain_fallback = True
                    params.pop("parse_mode", None)
                    result = self._api("sendMessage", text=chunk, **params)
                last_id = str(result.get("message_id", ""))
                # Only the first chunk carries the reply reference.
                params_base.pop("reply_parameters", None)
            self.stats["sent"] += 1
            try:
                from ...core.error_system import heartbeat
                heartbeat("telegram", True)
            except Exception as exc:  # noqa: BLE001 - telemetry never breaks sends
                record_failure('telegram.TelegramBotAdapter.send', exc)
            return SendResult(ok=True, platform=self.name, message_id=last_id,
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - ordinary failure, report as result
            self.stats["send_errors"] += 1
            try:
                from ...core.error_system import heartbeat
                heartbeat("telegram", False)
            except Exception as exc:  # noqa: BLE001 - telemetry never breaks sends
                record_failure('telegram.TelegramBotAdapter.send#2', exc)
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "",
                   parse_mode: str = "") -> SendResult:
        started = time.perf_counter()
        path = Path(media.path)
        if not path.is_file():
            return SendResult(ok=False, platform=self.name,
                              error=f"media file not found: {media.path}")
        if media.kind == "voice" or getattr(media, "voice_note", False):
            # A voice note must arrive as a voice bubble, not a file
            # attachment: kind="voice" used to fall through the map to
            # sendDocument, so TTS replies landed as documents on both
            # bots. (Producers convert to ogg first; see voice paths.)
            method = "sendVoice"
        else:
            method = {
                "image": "sendPhoto",
                "video": "sendVideo",
                "audio": "sendVoice" if path.suffix.lower() == ".ogg" else "sendAudio",
            }.get(media.kind, "sendDocument")
        field = {"sendPhoto": "photo", "sendVideo": "video",
                 "sendVoice": "voice", "sendAudio": "audio"}.get(method, "document")
        cap = for_platform("telegram-bot").max_caption
        try:
            data: dict[str, Any] = {"chat_id": chat.chat_id}
            if caption:
                # A caption carrying a parse mode arrives pre-fitted
                # by the gateway (F7): it is sent whole — truncating
                # HTML mid-tag would break the entity parse.  Plain
                # captions keep the historical truncation.
                data["caption"] = caption if parse_mode else caption[:cap]
            if parse_mode in ("HTML", "Markdown", "MarkdownV2"):
                data["parse_mode"] = parse_mode
            try:
                result = self._api_upload(method, data, field, path)
            except Exception as exc:  # noqa: BLE001 - classified below
                if not (parse_mode and _is_entity_parse_error(exc)):
                    raise
                # The API rejected the caption's markup itself: retry
                # as PLAIN text (tags stripped, caption cap applied),
                # mirroring send()'s fallback — a formatting failure
                # must never eat the file.
                _log.warning(
                    "telegram-bot: caption entity parse rejected (%s) "
                    "— retrying %s as plain text", exc, method)
                from .platforms import strip_html

                data.pop("parse_mode", None)
                data["caption"] = strip_html(caption)[:cap]
                result = self._api_upload(method, data, field, path)
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name,
                              message_id=str(result.get("message_id", "")),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def edit_media(self, chat: ChatRef, message_id: str, media: MediaRef,
                   *, caption: str = "") -> SendResult:
        """In-place media swap via Bot API ``editMessageMedia``."""
        started = time.perf_counter()
        path = Path(media.path)
        if not path.is_file():
            return SendResult(ok=False, platform=self.name,
                              error=f"media file not found: {media.path}")
        try:
            media_obj = {"type": "photo",
                         "media": f"attach://{path.name}"}
            if caption:
                media_obj["caption"] = caption[:for_platform("telegram-bot").max_caption]
            data = {"chat_id": chat.chat_id,
                    "message_id": message_id,
                    "media": json.dumps(media_obj)}
            result = self._api_upload("editMessageMedia", data, path.name,
                                      path, file_name=path.name)
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name,
                              message_id=str(result.get("message_id", message_id)),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def typing(self, chat: ChatRef, seconds: float = 3.0,
               action: str = "typing") -> bool:
        # The Bot API clears a typing indicator after ~5s server-side, so a
        # requested duration longer than that is held by re-firing
        # sendChatAction — the indicator stays up for the whole requested
        # window, proportional to the outgoing message length.
        # ``action="recording"`` maps to the Bot API's record_voice action.
        if seconds <= 0:
            return False
        api_action = "record_voice" if action == "recording" else "typing"
        deadline = time.time() + float(seconds)
        sent = False
        try:
            while time.time() < deadline:
                self._api("sendChatAction", chat_id=int(chat.chat_id), action=api_action)
                sent = True
                time.sleep(min(4.0, max(0.1, deadline - time.time())))
            return sent
        except Exception as exc:  # noqa: BLE001 - best-effort
            record_failure('telegram.TelegramBotAdapter.typing', exc)
            return sent

    # ── group/channel administration (Bot API) ──────────────────────────
    # The bot can only act where it is an admin with the right privilege.
    # Every method returns {"ok": True, ...} or {"ok": False, "error": ...}
    # with Telegram's honest reason. Call admin_rights() first — never
    # assume the bot can act.

    def _admin(self, method: str, chat: ChatRef, **params: Any) -> dict[str, Any]:
        try:
            result = self._api(method, chat_id=int(chat.chat_id), **params)
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - honest error, never a crash
            return {"ok": False, "error": str(exc)[:200]}

    def admin_rights(self, chat: ChatRef) -> dict[str, Any]:
        """Our OWN admin rights in a group/channel. Honest capability check."""
        if self._bot_id is None:
            try:
                self._bot_id = self._api("getMe").get("id")
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)[:200]}
        out = self._admin("getChatMember", chat, user_id=self._bot_id)
        if not out["ok"]:
            return out
        member = out["result"] or {}
        return {"ok": True, "status": member.get("status", ""),
                "can_post_messages": bool(member.get("can_post_messages")),
                "can_edit_messages": bool(member.get("can_edit_messages")),
                "can_delete_messages": bool(member.get("can_delete_messages")),
                "can_restrict_members": bool(member.get("can_restrict_members")),
                "can_promote_members": bool(member.get("can_promote_members")),
                "can_pin_messages": bool(member.get("can_pin_messages")),
                "can_invite_users": bool(member.get("can_invite_users"))}

    def admin_pin(self, chat: ChatRef, message_id: str,
                 silent: bool = True) -> dict[str, Any]:
        return self._admin("pinChatMessage", chat,
                           message_id=int(message_id),
                           disable_notification=bool(silent))

    def admin_unpin(self, chat: ChatRef) -> dict[str, Any]:
        return self._admin("unpinChatMessage", chat)

    def admin_unpin_all(self, chat: ChatRef) -> dict[str, Any]:
        return self._admin("unpinAllChatMessages", chat)

    def admin_ban(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        return self._admin("banChatMember", chat, user_id=int(user_id))

    def admin_unban(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        return self._admin("unbanChatMember", chat, user_id=int(user_id),
                           only_if_banned=True)

    def admin_restrict(self, chat: ChatRef, user_id: str, *,
                       until: int = 0, **permissions: bool) -> dict[str, Any]:
        """Restrict a member. Permissions are Bot API ChatPermissions keys
        (can_send_messages, can_send_media_messages, …); omitted keys are
        denied. ``until`` is a unix timestamp, 0 = forever."""
        perms = {"can_send_messages": False,
                 "can_send_audios": False,
                 "can_send_documents": False,
                 "can_send_photos": False,
                 "can_send_videos": False,
                 "can_send_video_notes": False,
                 "can_send_voice_notes": False,
                 "can_send_polls": False,
                 "can_send_other_messages": False,
                 "can_add_web_page_previews": False,
                 "can_change_info": False,
                 "can_invite_users": False,
                 "can_pin_messages": False,
                 "can_manage_topics": False}
        perms.update({k: bool(v) for k, v in permissions.items()
                      if k in perms})
        params: dict[str, Any] = {"user_id": int(user_id),
                                  "permissions": perms}
        if until:
            params["until_date"] = int(until)
        return self._admin("restrictChatMember", chat, **params)

    def admin_promote(self, chat: ChatRef, user_id: str,
                      title: str = "") -> dict[str, Any]:
        """Promote to admin with standard rights. ``title`` = custom title."""
        params: dict[str, Any] = {
            "user_id": int(user_id),
            "can_change_info": True,
            "can_delete_messages": True,
            "can_invite_users": True,
            "can_restrict_members": True,
            "can_pin_messages": True,
            "can_promote_members": False,
            "can_manage_chat": True,
            "can_manage_video_chats": False,
            "can_manage_topics": True,
        }
        out = self._admin("promoteChatMember", chat, **params)
        if out["ok"] and title.strip():
            self._api("setChatAdministratorCustomTitle", chat_id=int(chat.chat_id),
                      user_id=int(user_id), custom_title=title.strip()[:16])
        return out

    def admin_demote(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """Strip all admin rights (promoteChatMember with everything false)."""
        return self._admin("promoteChatMember", chat, user_id=int(user_id),
                           can_change_info=False, can_post_messages=False,
                           can_edit_messages=False, can_delete_messages=False,
                           can_invite_users=False, can_restrict_members=False,
                           can_pin_messages=False, can_promote_members=False,
                           can_manage_chat=False, can_manage_video_chats=False,
                           can_manage_topics=False)

    def admin_delete(self, chat: ChatRef, message_id: str) -> dict[str, Any]:
        return self._admin("deleteMessage", chat,
                           message_id=int(message_id))

    def admin_member(self, chat: ChatRef, user_id: str) -> dict[str, Any]:
        """One member's status/role in a group/channel."""
        out = self._admin("getChatMember", chat, user_id=int(user_id))
        if not out["ok"]:
            return out
        m = out["result"] or {}
        return {"ok": True, "status": m.get("status", ""),
                "user": (m.get("user") or {}).get("id", "")}

    def admin_administrators(self, chat: ChatRef) -> dict[str, Any]:
        """Full admin list with roles."""
        out = self._admin("getChatAdministrators", chat)
        if not out["ok"]:
            return out
        admins = []
        for m in out["result"] or []:
            u = m.get("user") or {}
            admins.append({"id": u.get("id", ""),
                           "name": u.get("first_name", ""),
                           "status": m.get("status", ""),
                           "title": m.get("custom_title", "")})
        return {"ok": True, "administrators": admins}

    def admin_member_count(self, chat: ChatRef) -> dict[str, Any]:
        out = self._admin("getChatMemberCount", chat)
        return {"ok": out["ok"], "count": out["result"] if out["ok"] else 0,
                "error": out.get("error", "")}

    def admin_set_title(self, chat: ChatRef, title: str) -> dict[str, Any]:
        return self._admin("setChatTitle", chat, title=(title or "").strip())

    def admin_set_description(self, chat: ChatRef,
                              description: str) -> dict[str, Any]:
        return self._admin("setChatDescription", chat,
                           description=description or "")

    def admin_invite_link(self, chat: ChatRef) -> dict[str, Any]:
        out = self._admin("exportChatInviteLink", chat)
        return {"ok": out["ok"], "invite": out["result"] if out["ok"] else "",
                "error": out.get("error", "")}

    def admin_leave(self, chat: ChatRef) -> dict[str, Any]:
        return self._admin("leaveChat", chat)

    def health(self) -> dict[str, Any]:
        info = super().health()
        info["bot_username"] = self._bot_username
        return info
