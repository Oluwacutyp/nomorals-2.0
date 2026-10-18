"""The chat gateway: every platform, one brain, at the same time.

The gateway owns:

* **Simultaneity.** One adapter thread per platform (Telegram userbot,
  Discord bot, WhatsApp bridge, local console). A message from any of them
  lands on one inbound callback; a reply from the brain goes out on the
  platform it came from.
* **Ordering.** Sends to the same chat are serialized with a per-chat lock,
  so the runtime and the autonomy agent (which may send at the same moment)
  never interleave badly.
* **Rate limiting.** A per-platform rolling one-hour window, enforced here
  and never trusted to the platform — this is what keeps a userbot account
  from getting flagged in the first week.
* **Registry.** Every chat the companion ever sees is upserted into the
  ``chats`` table, tagged as owner (primary partner) or US-based per config.
* **Dry run.** In dry-run mode sends are logged and acknowledged but never
  leave the machine — how you test the whole loop without touching an account.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ...core.ids import new_id, new_short_id
from ...core.logging_setup import get_logger, scrub_secrets
from ...core.retry import BackoffPolicy
from ...storage.db import Database
from .base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    IncomingHandler,
    MediaRef,
    SendResult,
    is_owner_chat,
)
from .chunking import (
    chunk_for_platform,
    label_chunks,
    media_kind_for,
    split_caption,
)
from .outbox import Outbox
from .platform_limits import for_platform
from .send_retry import send_with_retry

__all__ = ["ChatGateway", "OwnerTarget", "parse_chat_keys",
           "resolve_owner_targets"]

_log = get_logger(__name__)


class _HourWindow:
    """A rolling per-hour counter for one platform.

    A limit of ``0`` (or less) means *unlimited* — no counting at all.
    """

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self._events: list[float] = []
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        now = now if now is not None else time.time()
        with self._lock:
            cutoff = now - 3600.0
            while self._events and self._events[0] < cutoff:
                self._events.pop(0)
            if len(self._events) >= self.limit:
                return False
            self._events.append(now)
            return True

    def pending(self, now: float | None = None) -> int:
        if self.limit <= 0:
            return 0
        now = now if now is not None else time.time()
        cutoff = now - 3600.0
        with self._lock:
            return sum(1 for t in self._events if t >= cutoff)

    def set_limit(self, limit: int) -> None:
        """Change the cap live, under the window's own lock (the same lock
        ``allow()``/``pending()`` read it under)."""
        with self._lock:
            self.limit = int(limit)


class ChatGateway:
    """Runs N chat adapters at once and funnels everything through one callback."""

    def __init__(
        self,
        adapters: dict[str, ChatAdapter],
        *,
        db: Database | None = None,
        dry_run: bool = False,
        max_per_hour: int = 60,
        owner_chats: set[str] | None = None,
        us_chats: set[str] | None = None,
        clock: Callable[[], float] = time.time,
        adapter_builder: Callable[[str], ChatAdapter | None] | None = None,
        session_bridge: Any = None,
        durable_outbox: bool = False,
        outbox: Outbox | None = None,
        durable_drain_sync: bool = True,
        retry_policy: BackoffPolicy | None = None,
    ) -> None:
        """Create the gateway.

        ``session_bridge`` is an optional ``os.SessionBridge`` (injected —
        this module is L4 and must not import ``os``/L6). When present,
        every inbound message is attached to its OS Session before the
        brain's handler runs (``message.meta["os_session_id"]``).

        ``durable_outbox`` (default OFF) routes sends through the durable
        outbox (:mod:`nomorals.social.chat.outbox`): every send becomes a
        row before it becomes a network call, and receipts are queryable
        via :meth:`receipts` / ``nm sent``. Per-send ``durable=True``
        overrides the flag. Durable sends are for anything the owner
        would care about losing (reports, media, digests); typing
        indicators never use the outbox. ``durable_drain_sync`` drains
        the outbox synchronously inside :meth:`send` (so callers still
        get a delivery verdict); False returns a receipt handle
        immediately and leaves delivery to a later drain. ``retry_policy``
        overrides the default classified-retry backoff (tests pass a
        zero-delay policy).
        """
        if not adapters:
            raise ValueError("ChatGateway needs at least one adapter")
        self.adapters = dict(adapters)
        #: Preflight failures by adapter name (redacted). Populated in
        #: :meth:`start` when an adapter's preflight raises — the caller
        #: (e.g. ``run_chat_bot``) reads this to report the ACTUAL failure
        #: reason instead of a generic "did NOT start — set the token"
        #: message that sends the owner chasing a token that's already set.
        self.preflight_errors: dict[str, str] = {}
        self._known = dict(adapters)
        #: Lazy factory for hot-starting a platform that was skipped at boot
        #: (dependency installed since, credentials added, …).
        self._adapter_builder = adapter_builder
        #: Optional os.SessionBridge (L6), injected to keep layering clean.
        self.session_bridge = session_bridge
        self.db = db
        self.dry_run = dry_run
        self.clock = clock
        self.owner_chats = set(owner_chats or ())
        self.us_chats = set(us_chats or ())
        self._inbound: IncomingHandler | None = None
        #: Optional console mirror: called with each inbound ChatMessage so
        #: the local terminal can show a rich card. Best-effort — never
        #: raises, never blocks the feed. Set by scripts/run_chat_bot.py.
        self.console_mirror: Callable[[Any], None] | None = None
        self._chat_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        #: Durable outbox (G2). Built lazily from ``db`` when the flag is
        #: on and no outbox was injected. Without a db there is nowhere to
        #: persist rows, so durability degrades to direct sends (logged,
        #: never a silent drop).
        self.durable_outbox = bool(durable_outbox)
        self.outbox = outbox
        if self.durable_outbox and self.outbox is None:
            if db is not None:
                try:
                    self.outbox = Outbox(db)
                except Exception as exc:  # noqa: BLE001 - durability is opt-in, never fatal
                    _log.warning("chat gateway: could not build outbox: %s", exc)
                    self.outbox = None
            else:
                _log.warning("chat gateway: durable_outbox needs a db — "
                             "sends will go direct (no receipts)")
        self.durable_drain_sync = bool(durable_drain_sync)
        self.retry_policy = retry_policy
        #: Guards ``adapters``/``_known``: start_one/stop_one mutate them from
        #: the control thread while send()/status()/stop() read them from
        #: adapter, pool, and autonomy threads. Unlocked, a stop_one() pop
        #: racing a stop()/status() iteration raised "dictionary changed size
        #: during iteration".
        self._adapters_guard = threading.Lock()
        #: Guards ``stats``: inbound/dropped/dry_run counters are bumped from
        #: every adapter thread plus the pool; ``+=`` on a dict value is a
        #: read-modify-write and lost increments otherwise.
        self._stats_lock = threading.Lock()
        #: Inbound windows are PER-CHAT, not per-platform. A single spam group
        #: flooding the platform must not starve the owner's DMs (that bug
        #: dropped the owner's own /help while a group posted 30 times an hour).
        #: Keyed by chat.key, created lazily on first message.
        self._windows: dict[str, _HourWindow] = {}
        self._max_per_hour = int(max_per_hour)
        self.stats = {"dropped_rate_limited": 0, "dry_run_sends": 0, "inbound": 0}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _bump(self, key: str, amount: int = 1) -> None:
        """Thread-safe stats increment (see ``_stats_lock``)."""
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    def _adapter_for(self, platform: str) -> ChatAdapter | None:
        """Thread-safe adapter lookup (see ``_adapters_guard``)."""
        with self._adapters_guard:
            return self.adapters.get(platform)

    def _snapshot_adapters(self) -> dict[str, ChatAdapter]:
        with self._adapters_guard:
            return dict(self.adapters)

    def start(self, handler: IncomingHandler) -> list[str]:
        # Adapters always deliver into the gateway's funnel (registry, rate
        # limit, error isolation); the brain's handler sits at the far end of it.
        self._inbound = handler
        # One-time interactive setup (e.g. Telegram first-run login) must run
        # on the main thread while the keyboard is still free — the console
        # starts below and reads stdin from then on.
        for name in list(self._snapshot_adapters()):
            try:
                adapter = self._adapter_for(name)
                if adapter is None:
                    continue
                adapter.preflight()
            except KeyboardInterrupt:  # noqa: E106 - re-raised; only adapter errors are swallowed
                raise
            except Exception as exc:  # noqa: BLE001 - login failed: keep the rest
                # scrub_secrets: the exception may carry the full request URL
                # including a Telegram bot token (HttpClient embeds it).
                reason = scrub_secrets(str(exc))
                self.preflight_errors[name] = reason
                _log.warning("chat preflight failed for %s: %s", name, reason)
                with self._adapters_guard:
                    self.adapters.pop(name, None)
        started: list[str] = []
        for name, adapter in self._snapshot_adapters().items():
            if adapter.start(self._on_inbound):
                started.append(name)
        _log.info("chat gateway up: %s (dry_run=%s)", ",".join(started) or "none", self.dry_run)
        return started

    def stop(self) -> None:
        for adapter in self._snapshot_adapters().values():
            try:
                adapter.stop()
            except Exception as exc:  # noqa: BLE001
                # One adapter's stop must never prevent the rest from
                # stopping — but the failure is now logged, not swallowed.
                _log.warning(
                    "chat gateway: stop failed for %s: %s",
                    getattr(adapter, "name", "?"),
                    scrub_secrets(str(exc)),
                )

    # ── hot start/stop (control commands) ───────────────────────────────────
    def known_platforms(self) -> list[str]:
        return sorted(self._known)

    def set_rate_limit(self, limit: int) -> None:
        """Change the per-chat hourly inbound cap live.

        Power mode calls this with 0 (= unlimited); locking power restores the
        configured base. Existing windows pick the new limit up immediately.
        """
        self._max_per_hour = int(limit)
        # Snapshot first, then update each window under its own lock (see
        # _HourWindow.set_limit): no lock is ever held across a window lock
        # here, so the _locks_guard -> window-lock ordering used by
        # _on_inbound/_pending_total can never invert.
        with self._locks_guard:
            windows = list(self._windows.values())
        for window in windows:
            window.set_limit(self._max_per_hour)

    def start_one(self, name: str, handler: IncomingHandler) -> dict[str, Any]:
        name = name.strip().lower()
        with self._adapters_guard:
            adapter = self._known.get(name)
        if adapter is None and self._adapter_builder is not None:
            try:
                adapter = self._adapter_builder(name)
            except Exception as exc:  # noqa: BLE001 - report, don't crash the session
                return {"ok": False, "error": scrub_secrets(str(exc))}
        if adapter is None:
            return {"ok": False, "error": f"platform {name!r} is unknown or disabled in settings"}
        # Check-and-register is atomic under _adapters_guard: adapter.start
        # only spawns the receive thread (non-blocking, no callbacks), so
        # holding the guard across it is safe and closes the
        # double-start race between two control threads.
        with self._adapters_guard:
            current = self.adapters.get(name)
            if (current is not None and current._thread is not None
                    and current._thread.is_alive()):
                return {"ok": False, "error": f"{name} is already running"}
            self._known[name] = adapter
            self.adapters[name] = adapter
            try:
                started = adapter.start(handler)
            except Exception as exc:  # noqa: BLE001 - report, don't crash the session
                # adapter.start runs preflight internally; its exception may
                # carry a token-bearing URL — redact before it leaves here.
                reason = scrub_secrets(str(exc))
                self.preflight_errors[name] = reason
                self.adapters.pop(name, None)
                return {"ok": False, "error": reason}
        if started:
            _log.info("chat platform started on demand: %s", name)
            return {"ok": True, "platform": name}
        return {"ok": False, "error": f"{name} was already running"}

    def stop_one(self, name: str) -> dict[str, Any]:
        name = name.strip().lower()
        # Pop-then-stop is atomic under _adapters_guard: adapter.stop only
        # sets the stop event (non-blocking), so no deadlock with an adapter
        # thread calling back into the gateway.
        with self._adapters_guard:
            adapter = self.adapters.pop(name, None)
        if adapter is None:
            return {"ok": False, "error": f"{name} is not running"}
        try:
            adapter.stop()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": scrub_secrets(str(exc))}
        _log.info("chat platform stopped on demand: %s", name)
        return {"ok": True, "platform": name}

    @property
    def delivery_mode(self) -> str:
        """Effective delivery mode for substantive sends.

        ``durable`` — sends go through the outbox (row first, receipt
        after); ``direct`` — durability off by construction;
        ``durable-requested-unavailable`` — durability was requested
        but no outbox exists (no db), so sends go direct and carry
        ``delivery_degraded=True`` on their result.
        """
        if self.outbox is not None:
            return "durable"
        if self.durable_outbox:
            return "durable-requested-unavailable"
        return "direct"

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        with self._adapters_guard:
            names = sorted(set(list(self.adapters) + list(self._known)))
            known = dict(self._known)
            running = set(self.adapters)
        for name in names:
            adapter = known.get(name)
            if adapter is None:
                continue
            out[name] = {
                **adapter.health(),
                "running_in_session": name in running,
                "hour_pending": self._pending_total(),
            }
        with self._stats_lock:
            out["_stats"] = dict(self.stats)
        out["_delivery_mode"] = self.delivery_mode
        return out

    def _pending_total(self) -> int:
        """Sum of in-window inbound events across all chats (diagnostics)."""
        total = 0
        with self._locks_guard:
            for window in self._windows.values():
                total += window.pending()
        return total

    def _prune_windows(self, now: float) -> None:
        """Drop rate windows idle for a day. One entry per chat key seen is
        otherwise a slow unbounded leak on a years-running bot. Acquires
        ``_locks_guard`` itself — never call while holding it."""
        with self._locks_guard:
            if len(self._windows) <= 1000:
                return
            stale = [
                key for key, window in self._windows.items()
                if not window._events or window._events[-1] < now - 86400.0
            ]
            for key in stale:
                self._windows.pop(key, None)
        if stale:
            _log.info("chat gateway: pruned %d idle rate windows", len(stale))

    # ── registry ─────────────────────────────────────────────────────────────
    def platforms_for_chat_id(self, chat_id: str) -> list[str]:
        """Platforms the chat registry has seen this chat id on.

        Owner rows first, then most-recently-active — the data-driven way
        to resolve a bare ``owner_chats`` entry (``"123456789"``) to a
        platform without guessing.  Never raises.
        """
        if self.db is None or not chat_id:
            return []
        try:
            rows = self.db.query(
                "SELECT platform, MAX(is_owner) AS own, MAX(last_active) AS la "
                "FROM chats WHERE chat_id = ? "
                "GROUP BY platform ORDER BY own DESC, la DESC",
                (str(chat_id),),
            )
            return [str(r.get("platform") or "")
                    for r in rows if r.get("platform")]
        except Exception:  # noqa: BLE001 - registry is best-effort
            return []

    def register_chat(self, chat: ChatRef) -> dict[str, Any]:
        """Upsert a chat into the registry; returns the row (or a local one)."""
        key = chat.key
        is_owner = 1 if key in self.owner_chats else 0
        in_us = 1 if key in self.us_chats else 0
        if self.db is None:
            return {"key": key, "is_owner": is_owner, "in_us": in_us}
        try:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO chats (id, platform, chat_id, kind, title, peer, is_owner, in_us, last_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                         kind=excluded.kind,
                         title=CASE WHEN excluded.title != '' THEN excluded.title ELSE chats.title END,
                         peer=CASE WHEN excluded.peer != '' THEN excluded.peer ELSE chats.peer END,
                         last_active=excluded.last_active""",
                    (key, chat.platform, chat.chat_id, chat.kind, chat.title, chat.peer,
                     is_owner, in_us, self.clock()),
                )
            row = self.db.query_one("SELECT * FROM chats WHERE id = ?", (key,))
            return row or {"key": key, "is_owner": is_owner, "in_us": in_us}
        except Exception as exc:  # noqa: BLE001 - registry must never block a message
            _log.debug("chat registry upsert failed: %s", exc)
            return {"key": key, "is_owner": is_owner, "in_us": in_us}

    def touch(self, chat: ChatRef) -> None:
        if self.db is None:
            return
        try:
            self.db.execute(
                "UPDATE chats SET last_active = ? WHERE id = ?", (self.clock(), chat.key)
            )
        except Exception as exc:  # noqa: BLE001 - touch is best-effort, but log it
            _log.warning("chat touch failed for %s: %s", chat.key, exc)

    # ── inbound ──────────────────────────────────────────────────────────────
    def _on_inbound(self, message: ChatMessage) -> None:
        self._bump("inbound")
        row = self.register_chat(message.chat)
        self.touch(message.chat)
        # Owner messages are NEVER rate-limited: the cap exists to protect
        # the bot from group floods, not to drop the owner's own commands.
        # Same single owner test the partner runtime uses for gating, so
        # the two can never disagree.
        is_owner = is_owner_chat(
            message.chat,
            owner_chats=self.owner_chats,
            db_is_owner=bool(row.get("is_owner")),
        )
        # Stash the owner verdict on the message so downstream (runtime
        # command gating) uses the same answer instead of recomputing it
        # with less information.
        message.meta["is_owner"] = is_owner
        if not is_owner:
            prune_windows = False
            with self._locks_guard:
                window = self._windows.get(message.chat.key)
                if window is None:
                    window = _HourWindow(self._max_per_hour)
                    self._windows[message.chat.key] = window
                    prune_windows = len(self._windows) > 1000
            if prune_windows:
                self._prune_windows(self.clock())
            if not window.allow(self.clock()):
                self._bump("dropped_rate_limited")
                _log.warning("rate limit: dropping inbound from %s (>%s/h in this chat)",
                             message.chat.key, window.limit)
                return
        # Attach the OS Session (when a bridge is injected): every surface
        # shares one Session per chat, so memory/persona/missions are keyed
        # off the session, not the platform. Never drops the message.
        if self.session_bridge is not None:
            try:
                session = self.session_bridge.session_for_message(
                    message, is_owner=is_owner)
                message.meta["os_session_id"] = session.id
                message.meta["os_gating_mode"] = session.state.get(
                    "gating_mode", "")
            except Exception:  # noqa: BLE001 — session attach never drops
                _log.debug("session attach failed for %s", message.chat.key,
                           exc_info=True)
        handler = self._inbound
        if handler is not None:
            _log.info("gateway: dispatching inbound %s to the brain", message.chat.key)
            try:
                handler(message)
            except Exception as exc:  # noqa: BLE001 - brain errors must not kill the feed
                _log.exception("inbound brain failed for %s: %s", message.chat.key, exc)
        # Console mirror: rich inbound card on the local terminal.
        # Best-effort, never raises, never blocks.
        mirror = self.console_mirror
        if mirror is not None and message.chat.platform != "local":
            try:
                mirror(message)
            except Exception:  # noqa: BLE001 - mirror must never break the feed
                pass

    # ── outbound ────────────────────────────────────────────────────────────
    def _lock_for(self, chat: ChatRef) -> threading.Lock:
        with self._locks_guard:
            lock = self._chat_locks.get(chat.key)
            if lock is None:
                lock = threading.Lock()
                self._chat_locks[chat.key] = lock
            return lock

    # ── outbound: one chunk through one adapter, with classified retry ─
    def _send_one_chunk(
        self,
        adapter: ChatAdapter,
        chat: ChatRef,
        chunk: str,
        *,
        reply_to: str = "",
        buttons: list[list[tuple[str, str]]] | None = None,
        reply_keyboard: list[list[str]] | None = None,
        parse_mode: str = "",
    ) -> tuple[SendResult, int]:
        """Send one already-sized chunk. Returns ``(result, attempts)``.

        Keyboards and parse mode are gated on the platform's descriptor:
        platforms that cannot render them never receive them (previously
        they were forwarded and silently ignored). Retries are classified
        (transient → backoff+jitter, terminal → 1 attempt).
        """
        limits = for_platform(chat.platform)
        kwargs: dict[str, Any] = {"reply_to": reply_to}
        if buttons is not None and limits.supports_buttons:
            kwargs["buttons"] = buttons
        if parse_mode and limits.supports_html:
            kwargs["parse_mode"] = parse_mode
        if reply_keyboard is not None and limits.supports_buttons:
            kwargs["reply_keyboard"] = reply_keyboard

        def _call() -> SendResult:
            result = adapter.send(chat, chunk, **kwargs)
            if result.ok:
                self.touch(chat)
            return result

        return send_with_retry(
            _call, platform=chat.platform, policy=self.retry_policy,
        )

    def _send_one_media(
        self,
        adapter: ChatAdapter,
        chat: ChatRef,
        media: MediaRef,
        *,
        caption: str = "",
    ) -> tuple[SendResult, int]:
        """Send one media item with classified retry."""
        def _call() -> SendResult:
            result = adapter.send_media(chat, media, caption=caption)
            if result.ok:
                self.touch(chat)
            return result

        return send_with_retry(
            _call, platform=chat.platform, policy=self.retry_policy,
        )

    def _deliver_outbox_payload(
        self, payload: dict[str, Any],
    ) -> tuple[SendResult, int]:
        """The outbox drain's send function: one row → one adapter call."""
        platform = str(payload.get("platform") or "")
        chat_key = str(payload.get("chat_key") or "")
        try:
            chat = ChatRef(
                platform=platform,
                chat_id=str(payload.get("chat_id") or ""),
                kind=str(payload.get("chat_kind") or ChatKind.DM),
                thread_id=str(payload.get("thread_id") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - bad row fails loudly, not silently
            return SendResult(ok=False, platform=platform,
                              error=f"bad outbox row for {chat_key}: {exc}"), 1
        adapter = self._adapter_for(platform)
        if adapter is None:
            return SendResult(ok=False, platform=platform,
                              error=f"no adapter for {platform!r}"), 1
        if str(payload.get("kind") or "") == "chat_send_file":
            media = payload.get("media") or {}
            ref = MediaRef(
                path=str(media.get("path") or ""),
                mime=str(media.get("mime") or ""),
                kind=str(media.get("kind") or "file"),
                name=str(media.get("name") or ""),
            )
            result, attempts = self._send_one_media(
                adapter, chat, ref,
                caption=str(media.get("caption") or ""),
            )
            # A caption that was chunked at submit time: the remainder
            # goes out as text, in order, as part of the same row.
            rest = payload.get("caption_rest") or []
            if result.ok and rest:
                for piece in rest:
                    piece_result, _ = self._send_one_chunk(
                        adapter, chat, str(piece))
                    if not piece_result.ok:
                        return piece_result, attempts
            return result, attempts
        buttons_raw = payload.get("buttons")
        buttons: list[list[tuple[str, str]]] | None = None
        if buttons_raw is not None:
            try:
                buttons = [[(str(a), str(b)) for a, b in row]
                           for row in buttons_raw]
            except (TypeError, ValueError):
                buttons = None
        return self._send_one_chunk(
            adapter, chat, str(payload.get("text") or ""),
            reply_to=str(payload.get("reply_to") or ""),
            buttons=buttons,
            reply_keyboard=payload.get("reply_keyboard"),
            parse_mode=str(payload.get("parse_mode") or ""),
        )

    def _chunked(self, chat: ChatRef, text: str) -> list[str]:
        """Labelled chunks for ``text`` on this chat's platform. SMS is
        returned as a single chunk — its adapter owns segmentation."""
        chunks = chunk_for_platform(text, chat.platform)
        return label_chunks(chunks)

    def _send_direct(
        self,
        platform: str,
        chat: ChatRef,
        text: str,
        *,
        ordered: bool = True,
        reply_to: str = "",
        buttons: list[list[tuple[str, str]]] | None = None,
        reply_keyboard: list[list[str]] | None = None,
        parse_mode: str = "",
    ) -> SendResult:
        """Synchronous path: chunk, then send chunk-by-chunk in order.
        Stops at the first failed chunk — later chunks must never overtake
        a failed earlier one (ordering survives retry)."""
        adapter = self._adapter_for(chat.platform)
        if adapter is None:
            return SendResult(ok=False, platform=platform,
                              error=f"no adapter for {platform!r}")
        chunks = self._chunked(chat, text)
        total = len(chunks)

        def _do() -> SendResult:
            last = SendResult(ok=True, platform=platform,
                              message_id=new_short_id("skip"))
            for i, chunk in enumerate(chunks):
                is_last = i == total - 1
                result, _attempts = self._send_one_chunk(
                    adapter, chat, chunk,
                    reply_to=reply_to if i == 0 else "",
                    buttons=buttons if is_last else None,
                    reply_keyboard=reply_keyboard if is_last else None,
                    parse_mode=parse_mode if is_last else "",
                )
                if not result.ok:
                    return SendResult(
                        ok=False, platform=platform,
                        error=f"chunk {i + 1}/{total}: {result.error}")
                last = result
            return last

        if ordered:
            with self._lock_for(chat):
                return _do()
        return _do()

    def _send_durable(
        self,
        platform: str,
        chat: ChatRef,
        text: str,
        *,
        ordered: bool = True,
        reply_to: str = "",
        buttons: list[list[tuple[str, str]]] | None = None,
        reply_keyboard: list[list[str]] | None = None,
        parse_mode: str = "",
        durable_kind: str = "chat",
    ) -> SendResult:
        """Durable path: every chunk becomes a row, then the chat's queue
        drains in order. Returns the delivery verdict when
        ``durable_drain_sync`` is on, else a receipt handle (the first
        job id) for ``nm sent``."""
        assert self.outbox is not None  # noqa: S101 - caller checked
        chunks = self._chunked(chat, text)
        total = len(chunks)
        group_id = new_id("send")

        def _submit() -> list[str]:
            job_ids: list[str] = []
            for i, chunk in enumerate(chunks):
                is_last = i == total - 1
                job_ids.append(self.outbox.submit(
                    platform=platform, chat=chat, text=chunk,
                    reply_to=reply_to if i == 0 else "",
                    buttons=buttons if is_last else None,
                    reply_keyboard=reply_keyboard if is_last else None,
                    parse_mode=parse_mode if is_last else "",
                    durable_kind=durable_kind, group_id=group_id,
                ))
            return job_ids

        if ordered:
            with self._lock_for(chat):
                job_ids = _submit()
                if not self.durable_drain_sync:
                    return SendResult(ok=True, platform=platform,
                                      message_id=job_ids[0] if job_ids else "")
                return self._collect_durable(platform, chat, job_ids)
        job_ids = _submit()
        if not self.durable_drain_sync:
            return SendResult(ok=True, platform=platform,
                              message_id=job_ids[0] if job_ids else "")
        return self._collect_durable(platform, chat, job_ids)

    def _collect_durable(
        self, platform: str, chat: ChatRef, job_ids: list[str],
    ) -> SendResult:
        """Drain this chat's queue and reduce it to one verdict for the
        submitted jobs. A job still queued after the drain (transient
        head in backoff) is a queued-for-retry failure, not a silent
        drop — the row stays queryable via ``nm sent``."""
        assert self.outbox is not None  # noqa: S101 - caller checked
        # Idempotent replay: a submit that hit an already-done row
        # returns its stored verdict without re-sending.
        if len(job_ids) == 1:
            stored = self.outbox.job_result(job_ids[0])
            if stored is not None:
                return SendResult(
                    ok=bool(stored["ok"]), platform=platform,
                    message_id=str(stored.get("message_id") or ""),
                    error=str(stored.get("error") or ""))
        outcomes = self.outbox.deliver_chat(
            chat.key, platform, self._deliver_outbox_payload)
        by_job = {o["job_id"]: o for o in outcomes}
        last_id = ""
        for job_id in job_ids:
            outcome = by_job.get(job_id)
            if outcome is None:
                stored = self.outbox.job_result(job_id)
                if stored is not None and stored["ok"]:
                    last_id = str(stored.get("message_id") or last_id)
                    continue
                return SendResult(
                    ok=False, platform=platform,
                    error="queued for retry (see: nm sent --to "
                          f"{chat.key})")
            if not outcome["ok"]:
                return SendResult(
                    ok=False, platform=platform, error=str(outcome["error"]))
            last_id = str(outcome.get("message_id") or last_id)
        return SendResult(ok=True, platform=platform, message_id=last_id)

    def receipts(
        self,
        *,
        chat_key: str = "",
        platform: str = "",
        since: float = 0.0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Delivery receipts from the durable outbox (G4). Empty when
        durability was never enabled — receipts only exist for durable
        sends."""
        if self.outbox is None:
            return []
        return self.outbox.receipts(
            chat_key=chat_key, platform=platform, since=since, limit=limit)

    def send(
        self,
        platform: str,
        chat: ChatRef | str,
        text: str,
        *,
        reply_to: str = "",
        ordered: bool = True,
        buttons: list[list[tuple[str, str]]] | None = None,
        reply_keyboard: list[list[str]] | None = None,
        parse_mode: str = "",
        durable: bool | None = None,
        durable_kind: str = "chat",
    ) -> SendResult:
        """Send one message on one platform. Ordered per chat by default.

        ``buttons`` is ``[[(label, callback_data), ...], ...]`` — rendered
        as an inline keyboard by adapters whose platform supports it
        (TelegramBotAdapter); stripped elsewhere. ``reply_keyboard`` is
        ``[[text, ...], ...]`` — a persistent bottom keyboard, rendered
        by the Telegram adapters, stripped elsewhere. When both are
        passed, ``reply_keyboard`` wins.

        ``durable`` routes the send through the durable outbox (row
        first, then network call; receipts via :meth:`receipts` /
        ``nm sent``). None means the gateway's ``durable_outbox`` flag
        (default OFF). ``durable_kind`` labels the row (``chat``,
        ``report``, ``media``, ``digest``).
        """
        chat = chat if isinstance(chat, ChatRef) else ChatRef.parse(str(chat))
        text = text if text is not None else ""
        # Outbound secret guard: agent text leaving the system must never
        # carry secret-shaped values (defense against exfiltration via a
        # compromised model response). Tool inputs and stored memory are
        # NOT scrubbed — they legitimately carry credentials for API use.
        scrubbed = scrub_secrets(text)
        if scrubbed != text:
            _log.warning(
                "gateway.send: scrubbed secret-shaped content for %s",
                chat.key)
            text = scrubbed
        adapter = self._adapter_for(chat.platform)
        if adapter is None:
            return SendResult(ok=False, platform=platform, error=f"no adapter for {platform!r}")
        if not text.strip():
            return SendResult(ok=True, platform=platform, message_id=new_short_id("skip"))
        if self.dry_run:
            self._bump("dry_run_sends")
            _log.info("DRY-RUN send %s: %s", chat.key, text[:120])
            return SendResult(ok=True, platform=platform, message_id=new_short_id("dry"))

        use_durable = durable if durable is not None else self.durable_outbox
        if use_durable and self.outbox is not None:
            return self._send_durable(
                platform, chat, text, ordered=ordered, reply_to=reply_to,
                buttons=buttons, reply_keyboard=reply_keyboard,
                parse_mode=parse_mode, durable_kind=durable_kind)
        if use_durable and self.outbox is None:
            _log.warning(
                "gateway.send: durable requested for %s but no outbox "
                "(no db) — sending direct", chat.key)
            degraded = self._send_direct(
                platform, chat, text, ordered=ordered, reply_to=reply_to,
                buttons=buttons, reply_keyboard=reply_keyboard,
                parse_mode=parse_mode)
            degraded.delivery_degraded = True
            return degraded
        return self._send_direct(
            platform, chat, text, ordered=ordered, reply_to=reply_to,
            buttons=buttons, reply_keyboard=reply_keyboard,
            parse_mode=parse_mode)

    def send_file(
        self,
        platform: str,
        chat: ChatRef | str,
        path: str,
        *,
        caption: str = "",
        max_send_mb: float = 0.0,
        durable: bool | None = None,
        durable_kind: str = "media",
    ) -> SendResult:
        """Send a file, auto-compressing when it would exceed ``max_send_mb``.

        Compression is best-effort: if it doesn't help (already-compressed
        formats, no ffmpeg for video), the original goes out — the caller
        can pre-check with the compress tool.

        Media over the platform's inline cap goes out as a document
        (``PlatformLimits.max_file_mb``), never as a failed inline
        attempt. A caption over the caption cap is chunked: first chunk
        as the caption, the rest as follow-up text. ``durable`` routes
        through the durable outbox like :meth:`send` (default: the
        gateway's ``durable_outbox`` flag).
        """
        chat = chat if isinstance(chat, ChatRef) else ChatRef.parse(str(chat))
        adapter = self._adapter_for(chat.platform)
        if adapter is None:
            return SendResult(ok=False, platform=platform, error=f"no adapter for {platform!r}")
        target = path
        if max_send_mb > 0:
            try:
                from ...tools.compress import compress_file

                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > max_send_mb:
                    report = compress_file(path)
                    if report.get("ok") and report.get("new_bytes", 0) < report.get("original_bytes", 0):
                        target = str(report["path"])
                        _log.info("send_file: compressed %s -> %s (ratio %s)",
                                  path, target, report.get("ratio"))
            except Exception as exc:  # noqa: BLE001 - never block the send on compression
                _log.debug("send_file compression failed: %s", exc)
        limits = for_platform(chat.platform)
        try:
            size_bytes = os.path.getsize(target)
        except OSError:
            size_bytes = 0
        kind = media_kind_for(size_bytes, "file", limits)
        caption_chunks = split_caption(caption, max_caption=limits.max_caption) \
            if caption and limits.max_caption > 0 and len(caption) > limits.max_caption \
            else [caption]
        caption_first = caption_chunks[0]
        caption_rest = caption_chunks[1:]
        media = MediaRef(path=target, mime="", kind=kind,
                         name=os.path.basename(target))
        use_durable = durable if durable is not None else self.durable_outbox
        if use_durable and self.outbox is not None:
            group_id = new_id("send")
            with self._lock_for(chat):
                job_id = self.outbox.submit(
                    platform=platform, chat=chat, kind="chat_send_file",
                    durable_kind=durable_kind, group_id=group_id,
                    media={"path": target, "mime": "", "kind": kind,
                           "name": os.path.basename(target),
                           "caption": caption_first},
                    caption_rest=caption_rest,
                )
                if not self.durable_drain_sync:
                    return SendResult(ok=True, platform=platform,
                                      message_id=job_id)
                return self._collect_durable(platform, chat, [job_id])

        def _do() -> SendResult:
            result, _attempts = self._send_one_media(
                adapter, chat, media, caption=caption_first)
            if not result.ok:
                return result
            for piece in caption_rest:
                piece_result = self._send_direct(
                    platform, chat, piece, ordered=False)
                if not piece_result.ok:
                    return SendResult(
                        ok=False, platform=platform,
                        error=f"caption remainder: {piece_result.error}")
            return result

        with self._lock_for(chat):
            return _do()

    def broadcast(
        self,
        platform: str,
        chats: list[ChatRef | str],
        text: str,
        *,
        buttons: list[list[tuple[str, str]]] | None = None,
        parse_mode: str = "",
        pace_seconds: float = 1.0,
        stop_on: int = 3,
    ) -> dict[str, object]:
        """One message to many chats, with per-chat failure isolation.

        Broadcasts are where accounts get rate-limited into oblivion:
        ``pace_seconds`` spaces the sends (Telegram groups: 20 msg/min —
        keep it ≥3.0 there), and after ``stop_on`` consecutive failures the
        broadcast aborts rather than hammering a dead adapter. Each chat
        is its own result — one dead chat never kills the rest.

        Returns ``{"sent", "failed", "aborted", "results"}``.
        """
        results: list[dict[str, object]] = []
        sent = failed = 0
        aborted = False
        consecutive_failures = 0
        for chat in chats or []:
            if consecutive_failures >= stop_on:
                aborted = True
                _log.warning("broadcast aborted after %d consecutive failures",
                             consecutive_failures)
                break
            try:
                result = self.send(platform, chat, text, buttons=buttons,
                                   parse_mode=parse_mode)
            except Exception as exc:  # noqa: BLE001 - one chat, not all
                result = SendResult(ok=False, platform=platform,
                                    error=f"{type(exc).__name__}: {exc}")
            ref = chat if isinstance(chat, ChatRef) else ChatRef.parse(str(chat))
            results.append({"chat": ref.key, "ok": result.ok,
                            "error": result.error or ""})
            if result.ok:
                sent += 1
                consecutive_failures = 0
            else:
                failed += 1
                consecutive_failures += 1
            if pace_seconds > 0:
                time.sleep(pace_seconds)
        self._bump("broadcast_sends", sent)
        return {"sent": sent, "failed": failed, "aborted": aborted,
                "results": results}

    def typing(self, platform: str, chat: ChatRef | str, seconds: float = 3.0,
             action: str = "typing") -> bool:
        chat = chat if isinstance(chat, ChatRef) else ChatRef.parse(str(chat))
        adapter = self._adapter_for(chat.platform)
        if adapter is None or self.dry_run:
            return False
        try:
            return adapter.typing(chat, seconds, action=action)
        except Exception as exc:  # noqa: BLE001 - typing is cosmetic
            _log.debug("typing failed on %s: %s", platform, exc)
            return False

    def history(self, platform: str, chat: ChatRef | str, limit: int = 20) -> list[ChatMessage]:
        chat = chat if isinstance(chat, ChatRef) else ChatRef.parse(str(chat))
        adapter = self._adapter_for(chat.platform)
        if adapter is None:
            return []
        try:
            return adapter.history(chat, limit=limit)
        except Exception as exc:  # noqa: BLE001
            _log.debug("history failed on %s: %s", platform, exc)
            return []

    def media_for(self, message: ChatMessage) -> list[str]:
        """Local paths of this message's media, ready for the tool layer."""
        return [m.path for m in message.media if m.path]


def parse_chat_keys(raw: str, *, platform: str = "") -> set[str]:
    """Parse a config string like 'telegram:123, discord:456' or '123,456'.

    A bare id (no colon) is kept as-is when no ``platform`` is given —
    callers that compare against a bare chat id (``is_owner_chat``) use
    it, and it is never silently dropped: a configured owner id must
    not vanish because the owner wrote it without a platform prefix.
    """
    keys: set[str] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            p, _, cid = part.partition(":")
            keys.add(f"{p.strip()}:{cid.strip()}")
        elif platform:
            keys.add(f"{platform}:{part}")
        else:
            keys.add(part)
    return keys


@dataclass
class OwnerTarget:
    """One resolved owner destination: platform + chat id to attempt.

    ``source`` is the raw config entry this came from; ``note`` says how
    the platform was chosen (``explicit``, ``alias``,
    ``chat-registry``, ``id-shape`` or ``console-fallback``) so the
    delivery log can show *why* a target was tried.
    """
    platform: str
    chat_id: str
    source: str = ""
    note: str = ""


def _running_platforms(status: dict[str, Any] | None) -> set[str]:
    out: set[str] = set()
    for name, info in (status or {}).items():
        if not name or str(name).startswith("_"):
            continue
        if isinstance(info, dict) and info.get("running_in_session"):
            out.add(str(name))
    return out


def _looks_like_phone(value: str) -> bool:
    v = value.strip().replace(" ", "")
    return v.startswith("+") and v[1:].isdigit() and len(v) >= 8


def _looks_numeric_id(value: str) -> bool:
    v = value.strip()
    return v.lstrip("-").isdigit() and len(v.lstrip("-")) >= 3


def _family(name: str) -> str:
    return str(name).strip().lower().split("-")[0]


def resolve_owner_targets(
    raw: str,
    *,
    status: dict[str, Any] | None,
    registry_lookup: Callable[[str], list[str]] | None = None,
) -> list[OwnerTarget]:
    """Turn ``partner.owner_chats`` into an ordered delivery-attempt list.

    The audit root cause: ``NM_PARTNER_OWNER_CHATS`` is documented (see
    ``docs/TERMUX_ENV_TEMPLATE.txt``) as *"your numeric Telegram ID"* — a
    bare ID with no ``platform:`` prefix.  The old notifier only parsed
    ``platform:id`` entries and silently skipped bare ones, so ``_deliver``
    returned 0 and every proactive send was stored ``"failed"`` even with
    a live gateway.  This resolver closes that gap, systematically:

    * ``platform:id`` entries are kept as-is; when the named platform is
      unknown to the gateway but a same-family one is known (``telegram``
      vs ``telegram-bot``), it is aliased — and the alias is noted on the
      target so the delivery log stays honest.
    * bare IDs are resolved against the chat registry first (data, not
      guesses: the platform the registry has seen that chat id on), then
      by ID shape against *running* platforms — numeric IDs go to the
      telegram family / discord, ``+``-prefixed numbers to
      whatsapp/sms.  Numeric IDs are never offered to phone-number
      platforms (a Telegram user id must not become an SMS recipient).
    * when nothing resolves and the local console is running, one
      last-resort ``local:console`` target keeps the alert visible on the
      owner's terminal instead of silently lost.

    ``status`` is ``gateway.status()``; ``registry_lookup`` maps a chat id
    to the platforms the chat registry knows it on (owner rows first).
    The returned order is the attempt order — the fallback chain.
    Never raises.
    """
    targets: list[OwnerTarget] = []
    try:
        running = _running_platforms(status)
        known = {str(n) for n in (status or {}) if n and not str(n).startswith("_")}
        for part in (raw or "").split(","):
            entry = part.strip()
            if not entry:
                continue
            if ":" in entry:
                plat_raw, _, cid = entry.partition(":")
                plat = plat_raw.strip().lower()
                cid = cid.strip()
                if not (plat and cid):
                    continue
                if plat in known:
                    targets.append(OwnerTarget(plat, cid, source=entry,
                                              note="explicit"))
                else:
                    # Same-family alias: prefer a RUNNING platform
                    # (telegram -> telegram-bot), else any known one.
                    chosen = next(
                        (n for n in sorted(known)
                         if _family(n) == _family(plat) and n in running),
                        "",
                    ) or next(
                        (n for n in sorted(known)
                         if _family(n) == _family(plat) and n != plat),
                        "",
                    )
                    if chosen:
                        targets.append(OwnerTarget(
                            chosen, cid, source=entry,
                            note=f"alias {plat}->{chosen}"))
                    else:
                        targets.append(OwnerTarget(
                            plat, cid, source=entry,
                            note="platform not known to gateway"))
                continue
            # ── bare entry (no platform prefix) ──
            resolved: list[str] = []
            if registry_lookup is not None:
                try:
                    resolved = [p for p in (registry_lookup(entry) or [])
                                if p]
                except Exception:  # noqa: BLE001 - registry is best-effort
                    resolved = []
            if resolved:
                for plat in resolved:
                    targets.append(OwnerTarget(
                        plat.strip().lower(), entry, source=entry,
                        note="chat-registry"))
            elif _looks_like_phone(entry):
                for plat in ("whatsapp", "sms"):
                    if plat in running:
                        targets.append(OwnerTarget(
                            plat, entry, source=entry,
                            note="id-shape: phone number"))
            elif _looks_numeric_id(entry):
                for plat in ("telegram-bot", "telegram", "discord"):
                    if plat in running:
                        targets.append(OwnerTarget(
                            plat, entry, source=entry,
                            note="id-shape: numeric chat id"))
            else:
                targets.append(OwnerTarget(
                    "", entry, source=entry,
                    note="could not resolve to a platform"))
        # dedupe, order-preserving
        seen: set[tuple[str, str]] = set()
        unique: list[OwnerTarget] = []
        for t in targets:
            key = (t.platform, t.chat_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)
        targets = unique
        if not targets and "local" in running:
            targets.append(OwnerTarget(
                "local", "console", source="",
                note="console-fallback: no owner channel resolved"))
        return targets
    except Exception:  # noqa: BLE001 - resolution must never break delivery
        return targets
