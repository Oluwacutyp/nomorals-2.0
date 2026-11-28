"""The bot-token MTProto session — the /lookup spine that is
not the owner's personal account.

Devon is built to become a public bot: a stranger's lookup can
never depend on the owner's userbot account. This session is a
Telethon client authorized with the MAIN BOT's own BotFather
token (no new login, no new account, no personal credentials
anywhere near it). It resolves a public @username
(``contacts.ResolveUsernameRequest`` — Telethon's
``get_entity`` does exactly that and keeps the access hash)
and reads the same full-user dossier the userbot used to
return (``users.GetFullUserRequest``).

What it canNOT do is honest, not a failure: a bot session
cannot call user-only methods (``contacts.search``,
``contacts.resolvePhone``, ``contacts.importContacts`` all
answer ``BOT_METHOD_INVALID``), and it can only read a user
it has encountered (a bare ID with no cached entity has no
access hash to ask with). Those gaps are what the getChat /
t.me / userbot-bonus rungs are for — never a reason to fall
back to the owner's account as a dependency.

Threading: the client lives on its own thread + event loop
(the userbot adapter's shape), so a lookup never blocks the
chat loop and a dead session fails fast instead of hanging
one. A per-minute call cap + a cooldown after a failed start
keep a sick session from hammering Telegram. No telethon
import at module level: the client factory builds the real
client lazily, and the session talks to a small duck-type
interface (``start`` / ``get_entity`` / ``get_full_user`` /
``get_photo_count`` / ``is_connected`` / ``disconnect``) so
it is testable with no telethon installed.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

_START_COOLDOWN_S = 60.0


class _TelethonClientAdapter:
    """The small interface BotTokenSession needs, over a real
    Telethon client. Built lazily so importing this module
    never imports telethon."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def start(self, bot_token: str | None = None) -> None:
        await self._client.start(bot_token=bot_token)

    def is_connected(self) -> bool:
        try:
            return bool(self._client.is_connected())
        except Exception:  # noqa: BLE001 - a dead client reads down
            return False

    async def get_entity(self, ref: Any) -> Any:
        return await self._client.get_entity(ref)

    async def get_full_user(self, user_id: Any) -> Any | None:
        from telethon.tl.functions.users import GetFullUserRequest

        try:
            peer: Any = int(str(user_id))
        except (TypeError, ValueError):
            peer = str(user_id)
        try:
            return await self._client(GetFullUserRequest(peer))
        except Exception:  # noqa: BLE001 - unreadable reads None
            return None

    async def get_photo_count(self, user_id: Any) -> int | None:
        from telethon.tl.functions.photos import GetUserPhotosRequest

        try:
            peer: Any = int(str(user_id))
        except (TypeError, ValueError):
            peer = str(user_id)
        try:
            photos = await self._client(
                GetUserPhotosRequest(peer, offset=0, max_id=0,
                                     limit=1))
            return int(getattr(photos, "count", 0) or 0)
        except Exception:  # noqa: BLE001 - count stays absent
            return None

    async def disconnect(self) -> None:
        # Best-effort: a dead client must not break the close.
        with contextlib.suppress(Exception):
            await self._client.disconnect()


def _default_client_factory(api_id: int, api_hash: str,
                            session_path: str) -> Any:
    from telethon import TelegramClient

    return _TelethonClientAdapter(
        TelegramClient(session_path, api_id, api_hash))


class BotTokenSession:
    """A Telethon client signed in as the main bot itself.

    Fail-closed by construction: no bot token, no api creds,
    or a failed start all read as \"no session\" — the caller
    falls through to the next rung, never to the owner's
    account. Never raises out of a lookup method.
    """

    def __init__(self, *, api_id: Any, api_hash: str,
                 bot_token: str, session_path: str,
                 client_factory: Callable[..., Any] | None = None,
                 calls_per_minute: int = 30,
                 call_timeout: float = 30.0,
                 clock: Callable[[], float] | None = None) -> None:
        self._api_id = api_id
        self._api_hash = api_hash or ""
        self._bot_token = (bot_token or "").strip()
        self._session_path = session_path or ""
        self._factory = client_factory or _default_client_factory
        self._calls_per_minute = max(1, int(calls_per_minute))
        self._call_timeout = float(call_timeout)
        self._now = clock or time.time
        self._lock = threading.Lock()
        self._loop: Any = None
        self._thread: Any = None
        self._client: Any = None
        self._ready = False
        self._next_retry_at = 0.0
        self._calls: deque[float] = deque()

    # ── lifecycle ───────────────────────────────────────────────

    def _configured(self) -> bool:
        try:
            return bool(int(self._api_id)) and bool(self._api_hash) \
                and bool(self._bot_token) and bool(self._session_path)
        except (TypeError, ValueError):
            return False

    def healthy(self) -> bool:
        return bool(self._ready and self._client is not None
                    and self._client.is_connected())

    def _ensure(self) -> bool:
        """Start the session's thread + client once; False when
        it cannot start (cooldown applies — a sick session is
        not re-tried on every lookup)."""
        if self._ready and self.healthy():
            return True
        if not self._configured():
            return False
        now = self._now()
        if now < self._next_retry_at:
            return False
        with self._lock:
            if self._ready and self.healthy():
                return True
            if self._now() < self._next_retry_at:
                return False
            try:
                client = self._factory(self._api_id, self._api_hash,
                                       self._session_path)
            except Exception:  # noqa: BLE001 - no client, no session
                self._next_retry_at = self._now() + _START_COOLDOWN_S
                return False
            loop = asyncio.new_event_loop()
            started = threading.Event()
            failure: list[BaseException] = []

            def _run() -> None:
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        client.start(bot_token=self._bot_token))
                    started.set()
                    loop.run_forever()
                except BaseException as exc:  # noqa: BLE001
                    failure.append(exc)
                    started.set()

            thread = threading.Thread(target=_run, daemon=True,
                                      name="bot-token-session")
            thread.start()
            started.wait(timeout=self._call_timeout)
            if failure or not client.is_connected():
                self._next_retry_at = self._now() + _START_COOLDOWN_S
                # Already dying; stopping the loop is best-effort.
                with contextlib.suppress(Exception):
                    loop.call_soon_threadsafe(loop.stop)
                return False
            self._loop = loop
            self._thread = thread
            self._client = client
            self._ready = True
            return True

    def _take_slot(self) -> bool:
        """The per-minute cap: False refuses the call instead
        of queueing it — a lookup never waits on a flood."""
        now = self._now()
        while self._calls and self._calls[0] <= now - 60.0:
            self._calls.popleft()
        if len(self._calls) >= self._calls_per_minute:
            return False
        self._calls.append(now)
        return True

    def _call(self, make_coro: Any) -> Any:
        """Run one call on the session's loop, capped and
        timed. ``make_coro`` is a factory — the coroutine is
        only built once the call is accepted, so a refused
        call never leaves an un-awaited coroutine behind."""
        if not self._ensure():
            return None
        if not self._take_slot():
            return None
        try:
            future = asyncio.run_coroutine_threadsafe(
                make_coro(), self._loop)
            return future.result(timeout=self._call_timeout)
        except Exception:  # noqa: BLE001 - a failed call reads None
            return None

    def close(self) -> None:
        loop, client = self._loop, self._client
        self._loop = None
        self._client = None
        self._ready = False
        if loop is not None and client is not None:
            # Close is best-effort: a dead loop must not break it.
            with contextlib.suppress(Exception):
                future = asyncio.run_coroutine_threadsafe(
                    client.disconnect(), loop)
                future.result(timeout=5.0)
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(loop.stop)

    # ── the spine reads ─────────────────────────────────────────

    def resolve_username(self, username: str) -> dict[str, Any] | None:
        """@username → the entity (id, handle, name) via the
        bot's own session. None when the name is unoccupied,
        the bot has never met it, or the session is down."""
        handle = str(username or "").strip().lstrip("@")
        if not handle or not self._ensure():
            return None
        entity = self._call(
            lambda: self._client.get_entity(handle))
        if entity is None:
            return None
        first = str(getattr(entity, "first_name", "") or "")
        last = str(getattr(entity, "last_name", "") or "")
        return {
            "id": str(getattr(entity, "id", "") or ""),
            "username": str(getattr(entity, "username", "") or ""),
            "name": " ".join(p for p in (first, last) if p),
            "access_hash": getattr(entity, "access_hash", None),
        }

    def user_full(self, user_id: str) -> dict[str, Any] | None:
        """The full-user dossier in exactly the shape
        TelegramAdapter.user_full returns, read through the
        bot's own session. None when the bot has never
        encountered the account (a bare ID carries no access
        hash to ask with) or the session is down."""
        if not self._ensure():
            return None
        full = self._call(
            lambda: self._client.get_full_user(user_id))
        if full is None:
            return None
        photo_count = self._call(
            lambda: self._client.get_photo_count(user_id))
        try:
            from .telegram import _shape_full_user

            return _shape_full_user(full, photo_count=photo_count)
        except Exception:  # noqa: BLE001 - shaping never raises
            return None
