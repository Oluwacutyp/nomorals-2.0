"""Discord adapter — **your own account** (discord.py user client).

Like the Telegram side (Telethon userbot), this runs as *you*: the token is
your personal account token (copied once from the browser — see ``nm
discord --guide``), so the companion sees and texts exactly what you can:

* every DM on your account — reply, follow up, and start new ones
  (``start_dm`` opens a DM with anyone by id, or by username if you've
  ever seen them: a DM, or a server you're both in)
* every server you're a member of — text channels and threads map to group
  chats, so games, long conversations and media all work there
* new members: when someone joins one of your servers the adapter reports
  it (``on_new_member``) so the brain can decide to say hi *first* — each
  person is offered once, deduplicated across restarts

The one honest caveat, kept in plain sight: Discord's ToS reserves
automation for bot accounts, so a user client is done at your own risk —
the account running it is the one on the line. Enable with
``chat.discord_enabled = true`` + ``NM_CHAT_DISCORD_TOKEN``.

The client owns its asyncio loop inside ``run()``; ``send``/``typing``/
``history``/``start_dm`` are called from other threads and are marshaled
onto that loop with ``run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from typing import Any, Callable

from ...core.errors import ValidationError
from ...core.logging_setup import get_logger
from .base import ChatAdapter, ChatKind, ChatMessage, ChatRef, IncomingHandler, MediaRef, SendResult

__all__ = ["DiscordAdapter"]

_log = get_logger(__name__)


class DiscordAdapter(ChatAdapter):
    """A discord.py bot: DMs + server text channels, official API only."""

    name = "discord"
    supported_kinds = (ChatKind.DM, ChatKind.GROUP)

    #: callback(guild_name: str, member: dict) -> None, fired (at most once
    #: per member, across restarts) when someone joins one of your servers
    NewMemberCallback = Callable[[str, dict[str, Any]], None]

    def __init__(
        self,
        *,
        token: str,
        channel_allow: str = "",
        media_dir: str = "data/media/discord",
        greet_new: bool = True,
        on_new_member: "DiscordAdapter.NewMemberCallback | None" = None,
    ) -> None:
        super().__init__(media_dir=media_dir)
        if not token:
            raise ValidationError("discord token is empty", field="token")
        self.token = token
        self.channel_allow = {c.strip() for c in channel_allow.split(",") if c.strip()}
        self.greet_new = bool(greet_new)
        self.on_new_member = on_new_member
        self._bot: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._me: Any = None
        self._greeted_members: set[str] = self._load_greeted()

    # ── newcomer bookkeeping ────────────────────────────────────────────────
    def _greeted_path(self) -> str:
        return os.path.join(self.media_dir, ".greeted.json")

    def _load_greeted(self) -> set[str]:
        try:
            with open(self._greeted_path(), encoding="utf-8") as fh:
                data = json.load(fh)
            return {str(x) for x in data.get("members", [])}
        except Exception:  # noqa: BLE001 - no file yet, corrupt file: start fresh
            return set()

    def _save_greeted(self) -> None:
        try:
            os.makedirs(self.media_dir, exist_ok=True)
            with open(self._greeted_path(), "w", encoding="utf-8") as fh:
                json.dump({"members": sorted(self._greeted_members)}, fh)
        except Exception as exc:  # noqa: BLE001 - dedupe persistence is best-effort
            _log.debug("discord greeted-state save failed: %s", exc)

    def _fire_new_member(self, guild_name: str, member: Any) -> None:
        """Report a new server member at most once, on a side thread.

        Called from the event loop — the callback may talk to the brain,
        which must never happen inside the gateway loop.
        """
        if not self.on_new_member or not self.greet_new:
            return
        user_id = str(getattr(member, "id", "") or "")
        if not user_id or user_id in self._greeted_members:
            return
        self._greeted_members.add(user_id)
        self._save_greeted()

        def _job() -> None:
            try:
                self.on_new_member(
                    str(guild_name),
                    {
                        "id": user_id,
                        "name": str(getattr(member, "global_name", "") or getattr(member, "name", "") or user_id),
                        "tag": str(member),
                        "joined": str(getattr(member, "joined_at", "") or ""),
                    },
                )
            except Exception:  # noqa: BLE001 - one bad greeting must not kill the loop
                _log.exception("discord new-member callback failed for %s", user_id)

        threading.Thread(target=_job, name=f"discord-new-member-{user_id}", daemon=True).start()

    def preflight(self) -> None:
        """Reject a BOT token before it's used as an account token.

        A bot token (from the developer portal) would fail obscurely at
        connect time; the user flow needs your *personal* account token.
        """
        try:
            payload = self.token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            data = json.loads(base64.urlsafe_b64decode(payload.encode()))
        except Exception:  # noqa: BLE001 - not a JWT; let connect() surface it
            return
        if data.get("bot"):
            raise ValidationError(
                "that is a BOT token (developer portal) — the Discord companion "
                "runs YOUR account and needs your personal account token; "
                "run `nm discord --guide` for the one-time copy",
                field="discord_token",
            )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _allowed(self, channel_id: str, kind: str) -> bool:
        if kind == ChatKind.CHANNEL:
            return False  # the bot API does not post to news channels as a conversation
        if not self.channel_allow:
            return True
        return channel_id in self.channel_allow

    def _run_on_loop(self, coro: Any, timeout: float = 20.0) -> Any:
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("discord adapter is not connected")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── inbound ──────────────────────────────────────────────────────────────
    def run(self, handler: IncomingHandler) -> None:
        try:
            import discord
        except ImportError:
            raise RuntimeError(
                "discord.py is not installed: pip install nomorals[chat]"
            ) from None

        intents = discord.Intents.default()
        intents.message_content = True

        bot = discord.Client(intents=intents)
        self._bot = bot

        @bot.event
        async def on_ready() -> None:  # pragma: no cover - needs a live token
            self._loop = asyncio.get_running_loop()
            self._me = bot.user
            _log.info("discord bot connected as %s", bot.user)

        @bot.event
        async def on_message(message: Any) -> None:
            try:
                if getattr(message.author, "bot", False):
                    return
                channel = message.channel
                if isinstance(channel, discord.DMChannel):
                    chat_id, kind, peer = str(channel.id), ChatKind.DM, str(message.author)
                elif isinstance(channel, (discord.TextChannel, discord.Thread)):
                    # Thread channels are first-class here: the thread's own
                    # channel id is the chat_id, so each thread already gets
                    # its own key, rate window, and memory context.
                    chat_id, kind, peer = str(channel.id), ChatKind.GROUP, str(message.author)
                else:
                    return
                if not self._allowed(chat_id, kind):
                    return
                text = getattr(message, "content", "") or ""
                media: list[MediaRef] = []
                os.makedirs(self.media_dir, exist_ok=True)
                for i, attachment in enumerate(getattr(message, "attachments", ()) or ()):
                    try:
                        fname = f"dc-{int(time.time() * 1000)}-{i}{os.path.splitext(str(attachment.filename or ''))[1] or '.bin'}"
                        path = os.path.join(self.media_dir, fname)
                        await attachment.save(path)
                        media.append(
                            MediaRef(path=path, mime=str(attachment.content_type or ""),
                                     kind="image" if str(attachment.content_type or "").startswith("image") else "file",
                                     name=str(attachment.filename or ""))
                        )
                    except Exception as exc:  # noqa: BLE001 - media is best-effort
                        _log.debug("discord media save failed: %s", exc)
                if not text and not media:
                    return
                handler(
                    ChatMessage(
                    chat=ChatRef(
                        platform=self.name,
                        chat_id=chat_id,
                        kind=kind,
                        title=str(getattr(channel, "name", "") or chat_id),
                        peer=peer,
                    ),
                        incoming=True,
                        text=text,
                        sender=peer,
                        media=media,
                        message_id=str(message.id),
                        ts=(message.created_at.timestamp() if hasattr(message, "created_at") else time.time()),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad message must not kill the bot
                _log.exception("discord on_message failed: %s", exc)

        @bot.event
        async def on_member_join(member: Any) -> None:
            try:
                guild = getattr(member, "guild", None)
                self._fire_new_member(str(getattr(guild, "name", "") or ""), member)
            except Exception as exc:  # noqa: BLE001
                _log.debug("discord on_member_join failed: %s", exc)

        try:
            bot.run(self.token, log_handler=None)
        except Exception as exc:  # noqa: BLE001 - bad token, network down, ...
            if self.stopped:
                return
            raise exc

    # ── outbound ─────────────────────────────────────────────────────────────
    async def _channel(self, chat: ChatRef) -> Any:
        if chat.kind == ChatKind.DM:
            user = await self._bot.fetch_user(int(chat.chat_id))
            return await user.create_dm()
        return await self._bot.fetch_channel(int(chat.chat_id))

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        started = time.perf_counter()
        try:

            async def _do() -> Any:
                channel = await self._channel(chat)
                kwargs: dict[str, Any] = {}
                if reply_to:
                    try:
                        kwargs["reference"] = discord.MessageReference(message_id=int(reply_to),
                                                                       channel_id=int(chat.chat_id))
                    except (TypeError, ValueError):  # noqa: E103 - invalid reply_to id, send without reference
                        pass
                return await channel.send(text, **kwargs)

            message = self._run_on_loop(_do())
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name, message_id=str(getattr(message, "id", "")),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    def send_media(self, chat: ChatRef, media: MediaRef, *, caption: str = "") -> SendResult:
        started = time.perf_counter()
        try:

            async def _do() -> Any:
                import io

                channel = await self._channel(chat)
                with open(media.path, "rb") as fh:
                    file = discord.File(fh, filename=os.path.basename(media.path))
                return await channel.send(caption or None, file=file)

            message = self._run_on_loop(_do())
            self.stats["sent"] += 1
            return SendResult(ok=True, platform=self.name, message_id=str(getattr(message, "id", "")),
                              seconds=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            self.stats["send_errors"] += 1
            return SendResult(ok=False, platform=self.name, error=str(exc),
                              seconds=time.perf_counter() - started)

    #: Discord clears a typing indicator after ~10 s; refresh cadence.
    TYPING_REFRESH = 10.0

    def typing(self, chat: ChatRef, seconds: float = 3.0) -> bool:
        if seconds <= 0:
            return False
        seconds = min(float(seconds), 120.0)

        async def _do() -> bool:
            channel = await self._channel(chat)
            deadline = time.time() + seconds
            while time.time() < deadline and not self.stopped:
                try:
                    await channel.send_typing()
                except Exception as exc:  # noqa: BLE001 - cosmetic
                    _log.debug("discord typing event failed: %s", exc)
                    return False
                await asyncio.sleep(
                    min(self.TYPING_REFRESH, max(0.2, deadline - time.time()))
                )
            return True

        try:
            return bool(self._run_on_loop(_do(), timeout=seconds + 15))
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord typing failed: %s", exc)
            return False

    def history(self, chat: ChatRef, limit: int = 20) -> list[ChatMessage]:
        try:

            async def _do() -> list[ChatMessage]:
                channel = await self._channel(chat)
                out: list[ChatMessage] = []
                async for message in channel.history(limit=limit, oldest_first=True):
                    if getattr(message.author, "bot", False) and message.author.id == getattr(self._me, "id", None):
                        continue
                    out.append(
                        ChatMessage(
                            chat=chat,
                            incoming=True,
                            text=message.content or "",
                            sender=str(message.author),
                            message_id=str(message.id),
                            ts=message.created_at.timestamp() if hasattr(message, "created_at") else time.time(),
                        )
                    )
                return out

            return self._run_on_loop(_do())
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord history failed: %s", exc)
            return []

    # ── people: who's on my account, and DM them first ──────────────────────
    def servers(self) -> list[dict[str, Any]]:
        """Every server this account is a member of."""
        try:

            async def _do() -> list[dict[str, Any]]:
                return [
                    {
                        "id": str(g.id),
                        "name": str(getattr(g, "name", "") or g.id),
                        "members": int(getattr(g, "member_count", 0) or 0),
                    }
                    for g in getattr(self._bot, "guilds", []) or []
                ]

            return self._run_on_loop(_do())
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord servers failed: %s", exc)
            return []

    def members(self, guild: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """People in one of your servers (id or name; default: first server)."""
        try:

            async def _do() -> list[dict[str, Any]]:
                guilds = list(getattr(self._bot, "guilds", []) or [])
                target = None
                for g in guilds:
                    if guild and (str(g.id) == guild or str(getattr(g, "name", "")).lower() == guild.lower()):
                        target = g
                        break
                if target is None and guild and guild.isdigit():
                    for g in guilds:
                        if str(g.id) == guild:
                            target = g
                            break
                if target is None and not guild:  # no name given: first server
                    target = guilds[0] if guilds else None
                if target is None:
                    return []
                people = list(getattr(target, "members", []) or [])
                if not people:  # big server: cache is empty, ask the API
                    people = [m async for m in target.fetch_members(limit=limit)]
                out: list[dict[str, Any]] = []
                for m in people[: max(1, int(limit))]:
                    joined = getattr(m, "joined_at", None)
                    out.append(
                        {
                            "id": str(m.id),
                            "name": str(getattr(m, "global_name", "") or getattr(m, "name", "") or m.id),
                            "tag": str(m),
                            "joined": str(joined) if joined else "",
                        }
                    )
                return out

            return self._run_on_loop(_do(), timeout=30)
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord members failed: %s", exc)
            return []

    def contacts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent DM people, most recent conversation first."""
        try:

            async def _do() -> list[dict[str, Any]]:
                channels = list(getattr(self._bot, "get_private_channels", lambda: [])() or [])
                rows: list[tuple[int, dict[str, Any]]] = []
                for ch in channels:
                    user = getattr(ch, "recipient", None)
                    if user is None:
                        continue
                    last = int(getattr(ch, "last_message_id", 0) or 0)
                    rows.append(
                        (
                            last,
                            {
                                "id": str(user.id),
                                "name": str(getattr(user, "global_name", "") or getattr(user, "name", "") or user.id),
                                "tag": str(user),
                                "dm_id": str(ch.id),
                                "last": last,
                            },
                        )
                    )
                rows.sort(key=lambda r: r[0], reverse=True)
                return [r[1] for r in rows[: max(1, int(limit))]]

            return self._run_on_loop(_do())
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord contacts failed: %s", exc)
            return []

    def start_dm(self, target: str) -> ChatRef | None:
        """Open (or find) the DM with a person and hand back its chat ref.

        ``target`` is a user id, or a username you've already seen — in a DM
        or a shared server (Discord has no global username search). The
        ref works with ``send``/``typing``/``history`` exactly like any
        other chat; this is the "say hi first" primitive.
        """
        target = str(target).strip()
        if not target:
            return None
        try:

            async def _do() -> ChatRef | None:
                user: Any = None
                if target.isdigit():
                    user = getattr(self._bot, "get_user", lambda _id: None)(int(target))
                    if user is None:
                        user = await self._bot.fetch_user(int(target))
                else:
                    needle = target.lower()
                    for u in getattr(self._bot, "get_users", lambda: iter(()))():
                        if str(u).lower() == needle or str(getattr(u, "name", "")).lower() == needle:
                            user = u
                            break
                    if user is None:
                        for ch in getattr(self._bot, "get_private_channels", lambda: iter(()))():
                            r = getattr(ch, "recipient", None)
                            if r is not None and (str(r).lower() == needle or str(getattr(r, "name", "")).lower() == needle):
                                user = r
                                break
                    if user is None:
                        for g in getattr(self._bot, "guilds", []) or []:
                            for m in getattr(g, "members", []) or []:
                                if str(m).lower() == needle or str(getattr(m, "name", "")).lower() == needle:
                                    user = m
                                    break
                            if user is not None:
                                break
                if user is None:
                    _log.info("discord start_dm: could not resolve %r (seen DMs/servers only)", target)
                    return None
                dm = await user.create_dm()
                return ChatRef(
                    platform=self.name,
                    chat_id=str(dm.id),
                    kind=ChatKind.DM,
                    title=str(getattr(user, "global_name", "") or getattr(user, "name", "") or dm.id),
                    peer=str(user),
                )

            return self._run_on_loop(_do(), timeout=30)
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord start_dm failed for %r: %s", target, exc)
            return None

    def health(self) -> dict[str, Any]:
        base = super().health()
        base["platform"] = self.name
        base["me"] = str(self._me) if self._me is not None else ""
        base["connected"] = self._loop is not None and not self._loop.is_closed()
        try:
            base["servers"] = len(getattr(self._bot, "guilds", []) or [])
            base["dms"] = len(getattr(self._bot, "get_private_channels", lambda: [])() or [])
        except Exception:  # noqa: BLE001
            pass
        return base


