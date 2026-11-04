"""Mass actions (§14 life-hacks): MASS REACT, MASS VOTE, CHFOLLOW.

The reference bots (FEI XMD / BENFEI TECH) fan one command out
across every paired account: ``massreact <post link>/100``,
``pollvote <poll link> | 2``, ``chfollow <channel link>``.  Devon
adopts the grammar and the run-report-as-deliverable shape — with
one honest difference that is the whole design: her pool is the
owner's OWN accounts from the §15 pairing registry (plus the live
gateway sessions), never a strangers' pairing farm, and never a
bot token.  Telegram acts through MTProto sessions (the Bot API
cannot vote in polls at all); WhatsApp acts through each
account's own Baileys bridge.

Engineering controls, per the design (power with a dashboard,
not a firehose):

- **Arm gate + warn-first.**  Each tool carries an armed flag in
  a durable store.  Arming is a separate explicit owner act whose
  reply IS the warning (platform systems flag coordinated
  multi-account actions; accounts can be restricted).  An unarmed
  tool refuses live runs — and the refusal is journaled.
- **Dry-run first.**  A tool's first-ever run defaults to a
  dry-run: the full plan (accounts, order, pacing offsets, what
  each account would do) with ZERO platform calls — actors are
  not even constructed.  ``live`` must be said out loud to act.
- **Pacing floor.**  Actions are spaced by a per-tool pacing the
  owner can raise but never lower below ``PACING_FLOOR_S``,
  plus jitter.  Stop-on-error is the default.
- **Journal.**  Every run — live, dry, or refused — lands in the
  mass journal with its full account list and per-account
  results.  The run report the owner reads is rendered from the
  same record.

Platform truth (never faked):

- Telegram (MTProto): react = real (SendReactionRequest);
  vote = real (SendVoteRequest; option text resolved against the
  poll's own answers, fetched read-only from the first account);
  follow = real (JoinChannelRequest; already-a-participant is a
  successful no-op, which is what makes chfollow idempotent).
- WhatsApp (Baileys bridge): follow = real (``channel_follow``);
  channel react = real where the bridge's Baileys build carries
  ``newsletterReactMessage`` (the bridge says so honestly when
  it does not); poll VOTE = NO Baileys verb exists — protocol
  level only, unverified (DESIGN §16) — so pollvote on WhatsApp
  refuses with that reason instead of pretending.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .pairing import AccountRecord, PairingRegistry

#: The three mass tools, by their chat command names.
TOOLS = ("massreact", "pollvote", "chfollow")

#: Minimum seconds between two accounts' actions in one run.
#: The owner can raise a tool's pacing; the floor cannot be
#: lowered — it is the control that makes the tool survivable.
PACING_FLOOR_S = 20.0

#: Default jitter ceiling: each gap is pacing + U(0, jitter).
DEFAULT_JITTER_S = 10.0

#: Hard cap on accounts in one run, whatever the pool holds.
MAX_ACCOUNTS_PER_RUN = 200

#: The warn-first text (owner's standing rule): shown when a
#: tool is armed, in its status, and in every refusal.  Plain
#: words, once, as state — not a lecture stapled to every run.
WARN_TEXT = (
    "⚠️ Before this runs live, the risk in plain words: "
    "coordinated actions from many of your own accounts at once "
    "are exactly what platform anti-abuse systems look for. "
    "Telegram or WhatsApp may flag, restrict, or ban accounts "
    "that take part — pacing and jitter lower the temperature, "
    "they do not make it safe. These tools only ever act through "
    "your own paired accounts, never the bot, and every run is "
    "journaled with the accounts it used.")


class MassStoreError(RuntimeError):
    """A mass-action store exists but cannot be read — never a
    silent reset (an arms store that forgot it was disarmed
    would be a safety failure, not a convenience)."""


class MassRefusal(RuntimeError):
    """A run cannot proceed (unarmed, no capable accounts, bad
    target).  The message is the owner-facing truth."""


class AlreadyFollowing(Exception):
    """The account already follows the target — chfollow treats
    this as a successful no-op (idempotence), not an error."""


# ── targets ─────────────────────────────────────────────────────────

@dataclass
class MassTarget:
    """One parsed mass-action target (a post, a poll, a channel)."""
    platform: str                 # "telegram" | "whatsapp"
    raw: str                      # the link as the owner gave it
    peer: str = ""                # TG username (no @) / WA invite code or JID
    peer_id: int = 0              # TG marked id (-100…) for c/ links
    message_id: int = 0           # TG message/post id
    server_id: str = ""           # WA channel post server id

    @property
    def is_message(self) -> bool:
        return bool(self.message_id or self.server_id)

    def chat_ref_id(self) -> str:
        """The chat_id form the gateway adapters resolve."""
        if self.peer_id:
            return str(self.peer_id)
        if self.peer:
            return self.peer if self.peer.startswith("@") \
                or self.peer.endswith("@newsletter") \
                else f"@{self.peer}"
        return ""

    def describe(self) -> str:
        return self.raw


_TG_HOSTS = ("t.me", "telegram.me", "www.t.me", "www.telegram.me")
_WA_HOSTS = ("whatsapp.com", "www.whatsapp.com")


def _tg_target(raw: str, parts: list[str]) -> MassTarget:
    """parts = path segments of a t.me link (no empties)."""
    segs = [p for p in parts if p]
    if segs and segs[0] == "s":            # t.me/s/<user>/<id> preview
        segs = segs[1:]
    if not segs:
        raise MassRefusal(f"that Telegram link names nothing: {raw}")
    if segs[0] == "c":                     # private channel: c/<internal>/<msg>
        if len(segs) < 2 or not segs[1].isdigit():
            raise MassRefusal(
                f"that private-channel link is missing its id: {raw}")
        target = MassTarget(platform="telegram", raw=raw,
                            peer_id=int(f"-100{segs[1]}"))
        if len(segs) >= 3 and segs[-1].isdigit():
            target.message_id = int(segs[-1])
        return target
    target = MassTarget(platform="telegram", raw=raw,
                        peer=segs[0].lstrip("@"))
    if len(segs) >= 2 and segs[1].isdigit():
        target.message_id = int(segs[1])
    return target


def _wa_target(raw: str, parts: list[str]) -> tuple[MassTarget, int]:
    """whatsapp.com/channel/<code>[/<serverId>[/<count>]].

    Returns (target, riding_count) — the reference grammar lets
    the count ride the link; on WhatsApp links the segments are
    unambiguous (code, server id, count)."""
    segs = [p for p in parts if p]
    if len(segs) < 2 or segs[0] != "channel":
        raise MassRefusal(
            f"that WhatsApp link is not a channel link: {raw}")
    target = MassTarget(platform="whatsapp", raw=raw, peer=segs[1])
    count = 0
    if len(segs) >= 3:
        target.server_id = segs[2]
    if len(segs) >= 4 and segs[3].isdigit():
        count = int(segs[3])
    return target, count


def parse_target(text: str) -> tuple[MassTarget, int]:
    """Parse a mass-action link.  Returns (target, riding_count).

    Accepts t.me / telegram.me message + channel links and
    whatsapp.com/channel links; a bare ``…@newsletter`` JID is a
    WhatsApp channel target as-is.  Anything else refuses —
    guessing at a target is how accounts get banned for typos.
    """
    raw = (text or "").strip()
    if not raw:
        raise MassRefusal("no link given — paste the post/channel link")
    if raw.endswith("@newsletter"):
        return MassTarget(platform="whatsapp", raw=raw, peer=raw), 0
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host in _TG_HOSTS:
        return _tg_target(raw, parts), 0
    if host in _WA_HOSTS:
        return _wa_target(raw, parts)
    raise MassRefusal(
        f"I don't recognize that link ({raw}) — mass tools take "
        "t.me / telegram.me links and whatsapp.com/channel links")


# ── poll option resolution ──────────────────────────────────────────

def resolve_option(options: list[str], spec: str) -> int:
    """Resolve an owner's option spec to a 0-based poll index.

    A bare number is the 1-based option number (the reference
    grammar's ``| 2``); anything else matches the option TEXT —
    exact (case-insensitive) first, then a unique substring.
    Ambiguity and absence refuse: voting for the wrong option at
    scale is the one error this tool must never make quietly."""
    spec = (spec or "").strip()
    if not spec:
        raise MassRefusal("no poll option given — name its text or number")
    if spec.isdigit():
        idx = int(spec) - 1
        if 0 <= idx < len(options):
            return idx
        raise MassRefusal(
            f"option {spec} doesn't exist — the poll has "
            f"{len(options)} options")
    low = spec.lower()
    exact = [i for i, o in enumerate(options) if o.strip().lower() == low]
    if len(exact) == 1:
        return exact[0]
    partial = [i for i, o in enumerate(options)
               if low in o.strip().lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise MassRefusal(
            f"“{spec}” matches {len(partial)} poll options — "
            "be exact, or use the option number")
    raise MassRefusal(
        f"no poll option matches “{spec}” — options are: "
        + " · ".join(f"{i + 1}. {o}" for i, o in enumerate(options)))


# ── arms store ──────────────────────────────────────────────────────

class MassArms:
    """Per-tool armed flags + pacing, durably stored.

    The armed flag is the owner's explicit act (warn-first rule);
    the store also counts completed LIVE runs per tool, which is
    what "dry-run is the default for a tool's first run" reads.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._tools: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            tools = data.get("tools") if isinstance(data, dict) else None
            if not isinstance(tools, dict):
                raise ValueError("arms store has no tools object")
            self._tools = {k: dict(v) for k, v in tools.items()
                           if isinstance(v, dict)}
        except Exception as exc:  # noqa: BLE001 - quarantine, no reset
            aside = self.path.with_suffix(
                f".corrupt-{int(time.time())}.json")
            try:
                os.replace(self.path, aside)
            except OSError:
                aside = self.path  # type: ignore[assignment]
            raise MassStoreError(
                f"mass-action arms store unreadable ({exc}); "
                f"original moved to {aside} — tools stay DISARMED "
                "until you arm them again") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"tools": self._tools}, indent=1),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def _row(self, tool: str) -> dict[str, Any]:
        return self._tools.setdefault(tool, {
            "armed": False, "armed_at": 0.0,
            "pacing_s": PACING_FLOOR_S, "live_runs": 0})

    def state(self, tool: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._row(tool))

    def armed(self, tool: str) -> bool:
        return bool(self.state(tool).get("armed"))

    def live_runs(self, tool: str) -> int:
        return int(self.state(tool).get("live_runs") or 0)

    def pacing(self, tool: str) -> float:
        return max(PACING_FLOOR_S,
                   float(self.state(tool).get("pacing_s")
                         or PACING_FLOOR_S))

    def arm(self, tool: str, *, at: float | None = None) -> dict[str, Any]:
        with self._lock:
            row = self._row(tool)
            row["armed"] = True
            row["armed_at"] = float(at or time.time())
            self._save()
            return dict(row)

    def disarm(self, tool: str) -> dict[str, Any]:
        with self._lock:
            row = self._row(tool)
            row["armed"] = False
            self._save()
            return dict(row)

    def set_pacing(self, tool: str, seconds: float) -> float:
        seconds = float(seconds)
        if seconds < PACING_FLOOR_S:
            raise MassRefusal(
                f"pacing floor is {PACING_FLOOR_S:.0f}s between "
                f"accounts — {seconds:g}s is below it. The floor "
                "is the safety control, not a suggestion; raise "
                "it freely, it cannot go under.")
        with self._lock:
            row = self._row(tool)
            row["pacing_s"] = seconds
            self._save()
            return seconds

    def note_live_run(self, tool: str) -> int:
        with self._lock:
            row = self._row(tool)
            row["live_runs"] = int(row.get("live_runs") or 0) + 1
            self._save()
            return int(row["live_runs"])


# ── journal ─────────────────────────────────────────────────────────

class MassJournal:
    """Append-only JSONL run journal (the pairing/personal
    stores' discipline): every run — live, dry, refused — with
    its full account list."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def tail(self, limit: int = 20, *, tool: str = "") -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and (not tool
                                          or row.get("tool") == tool):
                out.append(row)
        return out[-limit:]


def default_mass_root(workspace_dir: str | Path) -> Path:
    return Path(workspace_dir) / "data" / "social" / "mass"


# ── account handles + actors ────────────────────────────────────────

#: What an actor can do, per tool name.
TG_CAPABILITIES = frozenset(TOOLS)
WA_CAPABILITIES = frozenset(("massreact", "chfollow"))

#: Why WhatsApp cannot vote (DESIGN §16) — named, never hidden.
WA_VOTE_GAP = ("whatsapp poll voting has no Baileys verb — poll "
               "votes are protocol-level only and unverified "
               "(DESIGN §16); pollvote runs on Telegram")


@dataclass
class MassAccount:
    """One pool account as a mass run sees it: the registry
    record, the tools it can perform, and a LAZY actor factory —
    dry-runs never call the factory, so planning touches no
    platform at all."""
    record: AccountRecord
    capabilities: frozenset
    factory: Callable[[], Any] | None = None

    @property
    def id(self) -> str:
        return self.record.id


def _run_coro(coro: Any) -> Any:
    """Drive one standalone coroutine on a private loop (the
    pairer-probe pattern: a per-action Telethon client is
    created, used and disconnected inside one call, so a private
    loop per action is correct)."""
    return asyncio.run(coro)


class _TelethonMassClient:
    """Production MTProto client for one paired Telegram account:
    a thin async wrapper over Telethon with exactly the verbs
    the mass actor drives, mirroring telegram_pairing's wrapper.
    Already-following surfaces as :class:`AlreadyFollowing`."""

    def __init__(self, session_path: str, api_id: int,
                 api_hash: str) -> None:
        from telethon import TelegramClient

        self._client = TelegramClient(session_path, api_id, api_hash)

    async def connect(self) -> None:
        await self._client.connect()

    async def disconnect(self) -> None:
        await self._client.disconnect()

    async def _entity(self, target: MassTarget) -> Any:
        if target.peer_id:
            return await self._client.get_entity(target.peer_id)
        return await self._client.get_entity(target.peer)

    async def react(self, target: MassTarget, emoji: str) -> None:
        from telethon.tl.functions.messages import SendReactionRequest
        from telethon.tl.types import ReactionEmoji

        entity = await self._entity(target)
        await self._client(SendReactionRequest(
            peer=entity, msg_id=int(target.message_id),
            reaction=[ReactionEmoji(emoticon=str(emoji))]))

    async def get_poll(self, target: MassTarget) -> list[dict[str, Any]]:
        entity = await self._entity(target)
        msg = await self._client.get_messages(
            entity, ids=int(target.message_id))
        media = getattr(msg, "media", None)
        poll = getattr(media, "poll", None)
        if poll is None:
            raise MassRefusal(
                f"message {target.message_id} in {target.raw} is "
                "not a poll — pollvote needs a poll post")
        out = []
        for ans in poll.answers or []:
            text = getattr(ans, "text", "")
            if not isinstance(text, str):
                text = str(getattr(text, "text", "") or "")
            out.append({"text": text, "option": bytes(ans.option)})
        return out

    async def vote(self, target: MassTarget, option: bytes) -> None:
        from telethon.tl.functions.messages import SendVoteRequest

        entity = await self._entity(target)
        await self._client(SendVoteRequest(
            peer=entity, msg_id=int(target.message_id),
            options=[option]))

    async def follow(self, target: MassTarget) -> None:
        from telethon.errors import UserAlreadyParticipantError
        from telethon.tl.functions.channels import JoinChannelRequest

        entity = await self._entity(target)
        try:
            await self._client(JoinChannelRequest(channel=entity))
        except UserAlreadyParticipantError as exc:
            raise AlreadyFollowing(
                "already a participant of this channel") from exc


class TelegramSessionActor:
    """Mass verbs for one paired Telegram account, over its OWN
    session file — never the live userbot's, never a bot token.
    The client is injectable (``client_factory``) so tests drive
    the whole actor without Telethon or a network."""

    def __init__(self, record: AccountRecord, *,
                 api_id: int | str = "", api_hash: str = "",
                 client_factory: Callable[..., Any] | None = None
                 ) -> None:
        self.record = record
        self.api_id = api_id
        self.api_hash = api_hash
        self._factory = client_factory

    def _client(self) -> Any:
        if self._factory is not None:
            return self._factory(self.record.session_path)
        try:
            api_id = int(self.api_id)
        except (TypeError, ValueError):
            api_id = 0
        if not api_id or not self.api_hash:
            raise MassRefusal(
                "Telegram mass actions need the MTProto "
                "api_id/api_hash (chat.telegram_api_id / "
                "chat.telegram_api_hash) — not configured")
        if not self.record.session_path:
            raise MassRefusal(
                f"{self.record.id} has no session file of its own")
        return _TelethonMassClient(
            self.record.session_path, api_id, str(self.api_hash))

    def _drive(self, verb: str, fn: Callable[[Any], Any]) -> dict[str, Any]:
        client = self._client()

        async def _main() -> Any:
            await client.connect()
            try:
                return await fn(client)
            finally:
                await client.disconnect()

        try:
            out = _run_coro(_main())
        except AlreadyFollowing as exc:
            return {"ok": True, "already": True, "detail": str(exc)}
        except MassRefusal:
            raise
        except Exception as exc:  # noqa: BLE001 - the platform's
            return {"ok": False, "error": str(exc)[:300]}  # own words
        return {"ok": True, "detail": out if isinstance(out, str) else ""}

    def react(self, target: MassTarget, emoji: str) -> dict[str, Any]:
        return self._drive(
            "react", lambda c: c.react(target, emoji))

    def poll_options(self, target: MassTarget) -> list[str]:
        client = self._client()

        async def _main() -> list[dict[str, Any]]:
            await client.connect()
            try:
                return await client.get_poll(target)
            finally:
                await client.disconnect()

        return [str(o.get("text") or "")
                for o in _run_coro(_main())]

    def vote(self, target: MassTarget,
             option_index: int) -> dict[str, Any]:
        # The option BYTES are per-poll and identical for every
        # account; resolve them through this account's own read
        # of the poll, then cast with SendVoteRequest.
        client = self._client()

        async def _main() -> None:
            await client.connect()
            try:
                answers = await client.get_poll(target)
                if not (0 <= option_index < len(answers)):
                    raise MassRefusal(
                        f"option {option_index + 1} doesn't exist "
                        f"— the poll has {len(answers)} options")
                await client.vote(
                    target, bytes(answers[option_index]["option"]))
            finally:
                await client.disconnect()

        try:
            _run_coro(_main())
        except MassRefusal:
            raise
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:300]}
        return {"ok": True}

    def follow(self, target: MassTarget) -> dict[str, Any]:
        return self._drive("follow", lambda c: c.follow(target))


def bridge_request(port: int, payload: dict[str, Any], *,
                   host: str = "127.0.0.1",
                   timeout: float = 30.0) -> dict[str, Any]:
    """One command over a WhatsApp bridge's JSON-lines socket
    (the protocol BridgeSupervisor.probe already speaks):
    connect, send one line, read until our response id lands."""
    req_id = f"mass-{int(time.time() * 1000)}-{payload.get('cmd', '')}"
    wire = dict(payload)
    wire["id"] = req_id
    try:
        with socket.create_connection((host, int(port)),
                                      timeout=timeout) as sock:
            sock.sendall((json.dumps(wire) + "\n").encode("utf-8"))
            sock.settimeout(timeout)
            buf = b""
            while len(buf) < 1_000_000:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                for line in buf.decode("utf-8", "replace").splitlines():
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if obj.get("id") == req_id:
                        return obj
    except OSError as exc:
        return {"ok": False,
                "error": f"bridge on port {port} unreachable: {exc}"}
    return {"ok": False, "error": "bridge gave no answer"}


class WhatsAppBridgeActor:
    """Mass verbs for one paired WhatsApp account, over its own
    bridge process (its own port, its own creds — slice 1's
    supervision shape).  ``request`` is injectable for tests."""

    def __init__(self, record: AccountRecord, *,
                 request: Callable[[dict[str, Any]], dict[str, Any]]
                 | None = None) -> None:
        self.record = record
        self._request = request or (
            lambda payload: bridge_request(int(record.port), payload))
        self._jid_cache: dict[str, str] = {}

    def _resolve_jid(self, target: MassTarget) -> str:
        if target.peer.endswith("@newsletter"):
            return target.peer
        cached = self._jid_cache.get(target.peer)
        if cached:
            return cached
        resp = self._request({"cmd": "channel_resolve",
                              "invite": target.peer})
        if not resp.get("ok"):
            raise MassRefusal(
                f"cannot resolve channel link {target.raw}: "
                f"{resp.get('error') or 'bridge refused'} — the "
                "…@newsletter JID works too")
        jid = str(resp.get("chat") or "")
        if not jid.endswith("@newsletter"):
            raise MassRefusal(
                f"channel link {target.raw} resolved to {jid!r}, "
                "which is not a channel")
        self._jid_cache[target.peer] = jid
        return jid

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._request(payload)
        if resp.get("ok"):
            return {"ok": True}
        return {"ok": False,
                "error": str(resp.get("error") or "bridge refused")[:300]}

    def react(self, target: MassTarget, emoji: str) -> dict[str, Any]:
        if not target.server_id:
            raise MassRefusal(
                "that link names the channel, not a post — "
                "massreact needs the post link (…/channel/<code>/"
                "<post id>)")
        jid = self._resolve_jid(target)
        return self._call({"cmd": "channel_react", "chat": jid,
                           "server_id": target.server_id,
                           "emoji": str(emoji)})

    def follow(self, target: MassTarget) -> dict[str, Any]:
        jid = self._resolve_jid(target)
        return self._call({"cmd": "channel_follow", "chat": jid})

    def vote(self, target: MassTarget,
             option_index: int) -> dict[str, Any]:
        return {"ok": False, "error": WA_VOTE_GAP}


class GatewayAdapterActor:
    """Mass verbs for a LIVE gateway session (the primary
    account): wraps the gateway's own adapter — the userbot for
    Telegram (its ``react`` is MTProto; vote/join ride the
    adapter's new verbs), the WhatsApp adapter for WhatsApp
    (``channel_follow`` / ``channel_react`` / invite resolution
    are its bridge admin verbs).  Never a bot adapter: the
    runtime only ever builds this over personal-account
    adapters, and Telegram votes are impossible for bots anyway.
    """

    def __init__(self, adapter: Any, platform: str) -> None:
        self.adapter = adapter
        self.platform = platform

    def _chat(self, target: MassTarget) -> Any:
        from .chat.base import ChatKind, ChatRef

        return ChatRef(platform=self.platform,
                       chat_id=target.chat_ref_id(),
                       kind=ChatKind.CHANNEL)

    def react(self, target: MassTarget, emoji: str) -> dict[str, Any]:
        if self.platform == "whatsapp":
            jid = self._wa_jid(target)
            if not target.server_id:
                raise MassRefusal(
                    "that link names the channel, not a post — "
                    "massreact needs the post link")
            resp = self.adapter.channel_react(
                jid, target.server_id, str(emoji))
            return {"ok": True} if resp.get("ok") else {
                "ok": False,
                "error": str(resp.get("error") or "refused")[:300]}
        result = self.adapter.react(
            self._chat(target), str(target.message_id), str(emoji))
        return {"ok": True} if result.ok else {
            "ok": False, "error": str(result.error or "refused")[:300]}

    def _wa_jid(self, target: MassTarget) -> str:
        if target.peer.endswith("@newsletter"):
            return target.peer
        resp = self.adapter.channel_resolve_invite(target.peer)
        if not resp.get("ok"):
            raise MassRefusal(
                f"cannot resolve channel link {target.raw}: "
                f"{resp.get('error') or 'bridge refused'}")
        return str(resp.get("chat") or "")

    def follow(self, target: MassTarget) -> dict[str, Any]:
        if self.platform == "whatsapp":
            resp = self.adapter.channel_follow(self._wa_jid(target))
            return {"ok": True} if resp.get("ok") else {
                "ok": False,
                "error": str(resp.get("error") or "refused")[:300]}
        # The adapter's join treats already-a-participant as the
        # success it is (the account follows — the goal state).
        result = self.adapter.join_channel(self._chat(target))
        return {"ok": True} if result.ok else {
            "ok": False, "error": str(result.error or "refused")[:300]}

    def poll_options(self, target: MassTarget) -> list[str]:
        return list(self.adapter.poll_options(
            self._chat(target), str(target.message_id)))

    def vote(self, target: MassTarget,
             option_index: int) -> dict[str, Any]:
        if self.platform == "whatsapp":
            return {"ok": False, "error": WA_VOTE_GAP}
        result = self.adapter.vote_poll(
            self._chat(target), str(target.message_id),
            int(option_index))
        return {"ok": True} if result.ok else {
            "ok": False, "error": str(result.error or "refused")[:300]}


# ── the runner ──────────────────────────────────────────────────────

@dataclass
class MassRun:
    """The outcome of one run — rendered for the owner and
    journaled from the same record."""
    run_id: str
    tool: str
    dry_run: bool
    target: str
    platform: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def ok_count(self) -> int:
        return sum(1 for e in self.entries if e.get("ok"))

    def to_journal(self) -> dict[str, Any]:
        return {"kind": "run", "run_id": self.run_id,
                "tool": self.tool, "dry_run": self.dry_run,
                "target": self.target, "platform": self.platform,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "note": self.note, "accounts": self.entries}


class MassRunner:
    """Plans and executes mass runs under the arms + journal.

    Clock, sleep and RNG are injectable; production sleeps for
    real between accounts (that IS the pacing control)."""

    def __init__(self, arms: MassArms, journal: MassJournal, *,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: random.Random | None = None,
                 jitter_s: float = DEFAULT_JITTER_S,
                 pacing_floor: float = PACING_FLOOR_S) -> None:
        self.arms = arms
        self.journal = journal
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.jitter_s = max(0.0, float(jitter_s))
        self.pacing_floor = float(pacing_floor)

    # ── selection ───────────────────────────────────────────────────
    def _select(self, accounts: list[MassAccount], tool: str,
                count: int | None) -> list[MassAccount]:
        linked = [a for a in accounts if a.record.status == "linked"]
        capable = [a for a in linked if tool in a.capabilities]
        if count:
            capable = capable[:max(0, int(count))]
        return capable[:MAX_ACCOUNTS_PER_RUN]

    def _gaps(self, n: int, pacing: float) -> list[float]:
        return [0.0] + [pacing + self._rng.uniform(0.0, self.jitter_s)
                        for _ in range(max(0, n - 1))]

    # ── run ─────────────────────────────────────────────────────────
    def run(self, tool: str, target: MassTarget, *,
            accounts: list[MassAccount],
            emoji: str = "", option_spec: str = "",
            count: int | None = None,
            dry_run: bool | None = None,
            stop_on_error: bool = True) -> MassRun:
        if tool not in TOOLS:
            raise MassRefusal(f"unknown mass tool {tool!r}")
        started = self._clock()
        run_id = (f"{tool}-{int(started)}-"
                  f"{hashlib.sha256(target.raw.encode()).hexdigest()[:6]}")
        pacing = max(self.pacing_floor, self.arms.pacing(tool))
        if dry_run is None:
            # The first-run rule: until a tool has completed a
            # live run, planning is all it does unless the owner
            # says "live" out loud.
            dry_run = self.arms.live_runs(tool) == 0
        chosen = self._select(accounts, tool, count)
        run = MassRun(run_id=run_id, tool=tool, dry_run=dry_run,
                      target=target.raw, platform=target.platform,
                      started_at=started)

        def _refuse(reason: str) -> MassRun:
            # A refused run is journaled WITH the accounts it
            # would have used — the audit trail names the pool,
            # not just the verdict.
            if chosen and not run.entries:
                run.entries = [
                    {"account": h.id, "offset_s": 0.0, "at": 0.0,
                     "ok": False, "error": "", "detail": reason}
                    for h in chosen]
            run.note = reason
            run.finished_at = self._clock()
            self.journal.append({**run.to_journal(),
                                 "kind": "run_refused"})
            raise MassRefusal(reason)

        if not dry_run and not self.arms.armed(tool):
            _refuse(
                f"{tool} is not armed. Arming is the owner's "
                f"explicit act, and its text is the warning:\n"
                f"{WARN_TEXT}\nArm it with: /{tool} arm")
        linked_any = [a for a in accounts
                      if a.record.status == "linked"]
        if not linked_any:
            _refuse(
                f"no connected {target.platform} accounts in the "
                "pool — pair one first (/pair); mass tools act "
                "only through your own live sessions")
        if not chosen:
            incapable = [a.id for a in linked_any
                         if tool not in a.capabilities]
            reason = (f"no connected account can do {tool} on "
                      f"{target.platform}")
            if tool == "pollvote" and target.platform == "whatsapp":
                reason += f" — {WA_VOTE_GAP}"
            elif incapable:
                reason += (f" ({', '.join(incapable)} lack the "
                           "verb)")
            _refuse(reason)

        gaps = self._gaps(len(chosen), pacing)
        offsets = []
        acc = 0.0
        for gap in gaps:
            acc += gap
            offsets.append(acc)

        # Poll option resolution (Telegram): a NUMBER needs no
        # platform read; TEXT resolves against the poll's own
        # answers — fetched ONCE, read-only, from the first
        # chosen account, before any vote is cast.  In a dry-run
        # nothing is fetched (zero platform calls is the point),
        # so a text option stays named, unresolved, in the plan.
        option_index: int | None = None
        option_note = ""
        if tool == "pollvote":
            spec = (option_spec or "").strip()
            if not spec:
                _refuse("pollvote needs an option — its text or "
                        "its number (| 2)")
            if spec.isdigit():
                option_index = int(spec) - 1
                option_note = f"option {spec} (by number)"
            elif dry_run:
                option_note = (f"option “{spec}” — resolves "
                               "against the poll's answers at "
                               "live time")
            else:
                first = chosen[0]
                actor = first.factory() if first.factory else None
                if actor is None or not hasattr(actor, "poll_options"):
                    _refuse(f"{first.id} cannot read the poll to "
                            f"resolve “{spec}”")
                try:
                    options = list(actor.poll_options(target))
                    option_index = resolve_option(options, spec)
                except MassRefusal as exc:
                    _refuse(str(exc))
                except Exception as exc:  # noqa: BLE001 - a poll
                    _refuse(f"couldn't read the poll to resolve "  # that
                            f"“{spec}”: {exc}")                    # won't open
                option_note = (f"option {option_index + 1} "
                               f"“{options[option_index]}”")

        verb = {"massreact": "react", "pollvote": "vote",
                "chfollow": "follow"}[tool]
        stopped = False
        for i, handle in enumerate(chosen):
            entry: dict[str, Any] = {
                "account": handle.id, "offset_s": round(offsets[i], 1),
                "at": 0.0, "ok": False, "error": "", "detail": ""}
            if dry_run:
                entry["ok"] = True
                entry["detail"] = f"would {verb}"
                if tool == "massreact":
                    entry["detail"] += f" {emoji}"
                if option_note:
                    entry["detail"] += f" · {option_note}"
                run.entries.append(entry)
                continue
            if stopped:
                entry["detail"] = "skipped — run stopped on error"
                run.entries.append(entry)
                continue
            if i > 0:
                self._sleep(gaps[i])
            entry["at"] = self._clock()
            actor = handle.factory() if handle.factory else None
            if actor is None:
                result = {"ok": False,
                          "error": "no live session for this "
                                   "account right now"}
            else:
                try:
                    if tool == "massreact":
                        result = actor.react(target, str(emoji))
                    elif tool == "pollvote":
                        result = actor.vote(target, int(option_index
                                                        or 0))
                    else:
                        result = actor.follow(target)
                except MassRefusal as exc:
                    result = {"ok": False, "error": str(exc)}
                except AlreadyFollowing as exc:
                    result = {"ok": True, "already": True,
                              "detail": str(exc)}
                except Exception as exc:  # noqa: BLE001 - one
                    # account's platform error is that account's
                    # failure (and the run's stop signal), never
                    # a crash that loses the journal mid-pool
                    result = {"ok": False, "error": str(exc)[:300]}
            entry["ok"] = bool(result.get("ok"))
            entry["error"] = str(result.get("error") or "")
            entry["detail"] = str(result.get("detail") or "")
            if result.get("already"):
                entry["detail"] = (entry["detail"]
                                    or "already following")
                entry["already"] = True
            run.entries.append(entry)
            if not entry["ok"] and stop_on_error:
                stopped = True
        run.finished_at = self._clock()
        if not dry_run:
            self.arms.note_live_run(tool)
        self.journal.append(run.to_journal())
        return run


def render_run(run: MassRun, *, tool_label: str = "") -> str:
    """The run report (DESIGN §14: a tool run's summary IS its
    deliverable).  Dry-runs render the plan; live runs render
    what each account actually did."""
    label = tool_label or run.tool
    if run.dry_run:
        lines = [f"🧪 DRY RUN — {label} → {run.target}",
                 f"platform: {run.platform} · "
                 f"{len(run.entries)} account(s), in order:"]
        for e in run.entries:
            lines.append(
                f"  +{e['offset_s']:.0f}s · {e['account']} — "
                f"{e['detail']}")
        lines.append(
            "nothing was sent. Run it again with “| live” "
            "appended to make it real.")
        return "\n".join(lines)
    ok = run.ok_count
    total = len(run.entries)
    mark = "✅" if ok == total else ("⚠️" if ok else "❌")
    lines = [f"{mark} {label} → {run.target}",
             f"{ok}/{total} account(s) acted:"]
    for e in run.entries:
        if e["ok"]:
            extra = f" ({e['detail']})" if e["detail"] else ""
            lines.append(f"  ✅ {e['account']}{extra}")
        else:
            why = e["error"] or e["detail"] or "failed"
            lines.append(f"  ❌ {e['account']} — {why}")
    lines.append(f"journal: run {run.run_id}")
    return "\n".join(lines)
