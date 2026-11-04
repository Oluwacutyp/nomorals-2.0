"""Telegram pairing into the §15 pool (slice 3).

WhatsApp pairs by an 8-character code typed into Linked Devices;
Telegram's MTProto equivalent is phone number → login code → (2FA
password where the account has one).  This module runs that flow
against the SAME durable registry the WhatsApp bridges live in
(:mod:`nomorals.social.pairing`): one pool, platform field apart.

Shape:

- ``TelegramPairer.start(phone)`` registers the account (platform
  ``telegram``, status ``pairing``), opens a Telethon client on the
  account's OWN session file (``<sessions_root>/tg-<digits>`` —
  never the live userbot's ``data/telegram.session``), and asks
  Telegram to send the login code.  The pending flow (client +
  phone-code hash) is held in memory only, with a TTL: a flow the
  owner never completes EXPIRES — the client is disconnected, the
  flow dropped, the record falls back to ``registered``.  A pairing
  code is a credential while it lives; nothing about it is written
  to disk.
- ``confirm_code`` signs in.  An account with two-factor auth moves
  the flow to ``password_needed``; ``confirm_password`` finishes it.
  The password passes through memory to Telethon and nowhere else —
  it is never stored, logged, or journaled.
- On success the record learns its numeric user id and flips to
  ``linked``.  The session FILE is then the credential, supervised
  by probe (a fresh client asking ``is_user_authorized``) — never
  by assumption.

The client is injectable (``client_factory``) so tests drive the
whole state machine without Telethon or a network; production
builds a real ``TelegramClient`` wrapper.  Missing api_id/api_hash
fails loudly at ``start`` — pairing without MTProto credentials is
impossible and the surface must say so, not improvise.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .pairing import AccountRecord, PairingRegistry, account_id_for

FLOW_TTL_S = 600.0


class PairingFlowError(ValueError):
    """The pairing flow cannot proceed (no flow, expired, bad state)."""


class TwoFactorRequired(Exception):
    """The account demands its 2FA password after the login code."""


@dataclass
class _Flow:
    account_id: str
    phone: str
    client: Any
    code_hash: str = ""
    state: str = "code_sent"      # code_sent | password_needed
    created_at: float = 0.0
    expires_at: float = 0.0
    # One client = one event loop for its whole life: Telethon
    # binds a client to the loop of its first connection, and a
    # second asyncio.run() loop makes every later call fail.  The
    # flow owns a loop thread; every client coroutine is submitted
    # to it (see _submit).
    loop: Any = field(default=None, repr=False)
    thread: Any = field(default=None, repr=False)

    def submit(self, coro: Any, timeout: float = 120.0) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)

    def close(self) -> None:
        try:
            self.submit(self.client.disconnect(), timeout=15.0)
        except Exception:  # noqa: BLE001 - a dead client is gone
            pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:  # noqa: BLE001
            pass
        if self.thread is not None:
            self.thread.join(timeout=5.0)


def _telethon_factory(session_path: str, api_id: int, api_hash: str) -> Any:
    """Production client: a thin async wrapper over Telethon with
    exactly the surface the pairer drives, translating Telethon's
    2FA error into :class:`TwoFactorRequired`."""
    from telethon import TelegramClient
    from telethon.errors import SessionPasswordNeededError

    client = TelegramClient(session_path, api_id, api_hash)

    class _Client:
        async def connect(self) -> None:
            await client.connect()

        async def disconnect(self) -> None:
            await client.disconnect()

        async def send_code(self, phone: str) -> str:
            sent = await client.send_code_request(phone)
            return str(getattr(sent, "phone_code_hash", "") or "")

        async def sign_in_code(self, phone: str, code: str,
                               code_hash: str) -> None:
            try:
                await client.sign_in(phone=phone, code=code,
                                     phone_code_hash=code_hash)
            except SessionPasswordNeededError as exc:
                raise TwoFactorRequired(
                    "account has two-factor auth on") from exc

        async def sign_in_password(self, password: str) -> None:
            await client.sign_in(password=password)

        async def is_authorized(self) -> bool:
            return bool(await client.is_user_authorized())

        async def me_id(self) -> str:
            me = await client.get_me()
            return str(getattr(me, "id", "") or "")

    return _Client()


def _run(coro: Any) -> Any:
    """Drive a STANDALONE coroutine (probe clients only — created,
    used and disconnected inside one call) on a private loop."""
    return asyncio.run(coro)


def _spawn_flow_loop() -> tuple[Any, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=loop.run_forever, name="tg-pair-flow", daemon=True)
    thread.start()
    return loop, thread


class TelegramPairer:
    """Phone + login-code pairing for Telegram accounts."""

    def __init__(self, registry: PairingRegistry, *,
                 sessions_root: str | Path,
                 api_id: int | str = "",
                 api_hash: str = "",
                 client_factory: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.time,
                 flow_ttl: float = FLOW_TTL_S) -> None:
        self.registry = registry
        self.sessions_root = Path(sessions_root)
        self.api_id = api_id
        self.api_hash = api_hash
        self._factory = client_factory or _telethon_factory
        self._clock = clock
        self.flow_ttl = float(flow_ttl)
        self._flows: dict[str, _Flow] = {}
        self._lock = threading.RLock()

    # ── helpers ─────────────────────────────────────────────────────
    def _credentials(self) -> tuple[int, str]:
        try:
            api_id = int(self.api_id)
        except (TypeError, ValueError):
            api_id = 0
        if not api_id or not self.api_hash:
            raise PairingFlowError(
                "Telegram pairing needs the MTProto api_id/api_hash "
                "(chat.telegram_api_id / chat.telegram_api_hash) — "
                "they are not configured, so no login code can be "
                "requested")
        return api_id, str(self.api_hash)

    def _session_stem(self, digits: str) -> str:
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        return str(self.sessions_root / f"tg-{digits}")

    def _drop_flow(self, account_id: str) -> _Flow | None:
        with self._lock:
            flow = self._flows.pop(account_id, None)
        if flow is not None:
            flow.close()
        return flow

    def _live_flow(self, account_id: str) -> _Flow:
        with self._lock:
            flow = self._flows.get(account_id)
        if flow is None:
            raise PairingFlowError(
                f"no live pairing flow for {account_id} — start one "
                "with /pair telegram <number>")
        if self._clock() > flow.expires_at:
            self._drop_flow(account_id)
            if self.registry.get(account_id) is not None:
                rec = self.registry.get(account_id)
                if rec is not None and rec.status == "pairing":
                    self.registry.set_status(account_id, "registered")
            raise PairingFlowError(
                "that pairing flow expired (codes live minutes, "
                "flows ten) — start again with /pair telegram "
                "<number>")
        return flow

    # ── the flow ────────────────────────────────────────────────────
    def start(self, phone: str, label: str = "") -> AccountRecord:
        digits = re.sub(r"\D", "", phone or "")
        if len(digits) < 7:
            raise PairingFlowError(
                "that doesn't look like a phone number — give the "
                "number in international format, e.g. 2348012345678")
        api_id, api_hash = self._credentials()
        rec_id = account_id_for("telegram", digits)
        now = self._clock()
        rec = self.registry.get(rec_id)
        stem = self._session_stem(digits)
        if rec is None:
            rec = AccountRecord(
                id=rec_id, platform="telegram", identity=digits,
                label=(label or "").strip(), session_path=stem,
                status="pairing", created_at=now)
        else:
            rec.status = "pairing"
            rec.session_path = stem
            if label:
                rec.label = label.strip()
        self.registry.upsert(rec)
        self._drop_flow(rec_id)  # a restarted flow replaces the old
        client = self._factory(stem, api_id, api_hash)
        loop, thread = _spawn_flow_loop()
        flow = _Flow(
            account_id=rec_id, phone=digits, client=client,
            state="code_sent", created_at=now,
            expires_at=now + self.flow_ttl, loop=loop, thread=thread)

        async def _open() -> str:
            await client.connect()
            return await client.send_code(digits)

        try:
            flow.code_hash = str(flow.submit(_open()) or "")
        except Exception:
            flow.close()
            self.registry.set_status(rec_id, "registered")
            raise
        with self._lock:
            self._flows[rec_id] = flow
        return rec

    def pending(self, account_id: str) -> dict[str, Any] | None:
        """Public flow state for surfaces (never the code hash)."""
        with self._lock:
            flow = self._flows.get(account_id)
        if flow is None:
            return None
        return {"state": flow.state,
                "expires_in_s": max(0.0, flow.expires_at
                                    - self._clock())}

    def confirm_code(self, account_id: str,
                     code: str) -> dict[str, Any]:
        flow = self._live_flow(account_id)
        code = re.sub(r"\D", "", code or "")
        if not code:
            raise PairingFlowError("give the login code Telegram sent")
        try:
            flow.submit(flow.client.sign_in_code(flow.phone, code,
                                                 flow.code_hash))
        except TwoFactorRequired:
            flow.state = "password_needed"
            return {"status": "password_needed",
                    "account": account_id}
        return self._finalize(flow)

    def confirm_password(self, account_id: str,
                         password: str) -> dict[str, Any]:
        flow = self._live_flow(account_id)
        if flow.state != "password_needed":
            raise PairingFlowError(
                "that account hasn't asked for a 2FA password — "
                "send the login code first (/pair tgcode)")
        if not password:
            raise PairingFlowError("give the account's 2FA password")
        flow.submit(flow.client.sign_in_password(password))
        return self._finalize(flow)

    def _finalize(self, flow: _Flow) -> dict[str, Any]:
        remote_id = ""
        try:
            remote_id = str(flow.submit(flow.client.me_id(),
                                        timeout=30.0) or "")
        except Exception:  # noqa: BLE001 - identity is a bonus; the
            remote_id = ""              # session file is the proof
        rec = self.registry.get(flow.account_id)
        if rec is not None:
            if remote_id:
                rec.remote_id = remote_id
            rec.status = "linked"
            rec.last_seen = self._clock()
            self.registry.upsert(rec)
        self._drop_flow(flow.account_id)
        return {"status": "linked", "account": flow.account_id,
                "remote_id": remote_id}

    # ── supervision ─────────────────────────────────────────────────
    def expire_flows(self) -> list[str]:
        """Drop every flow past its TTL; records fall back from
        ``pairing`` to ``registered``.  Returns expired account ids."""
        now = self._clock()
        with self._lock:
            stale = [fid for fid, f in self._flows.items()
                     if now > f.expires_at]
        out = []
        for account_id in stale:
            self._drop_flow(account_id)
            rec = self.registry.get(account_id)
            if rec is not None and rec.status == "pairing":
                self.registry.set_status(account_id, "registered")
            out.append(account_id)
        return out

    def probe(self, rec: AccountRecord) -> str:
        """Authorization truth for one Telegram account: a fresh
        client on the account's own session file asks MTProto
        whether the session is authorized.  No session file → the
        account was never linked → ``down``, honestly."""
        stem = rec.session_path or self._session_stem(rec.identity)
        if not Path(stem + ".session").exists() \
                and not Path(stem).exists():
            return "down"
        api_id, api_hash = self._credentials()
        client = self._factory(stem, api_id, api_hash)

        async def _check() -> bool:
            await client.connect()
            try:
                return bool(await client.is_authorized())
            finally:
                await client.disconnect()

        try:
            return "linked" if _run(_check()) else "down"
        except Exception:  # noqa: BLE001 - an unreachable network is
            return "down"                 # not a linked session
