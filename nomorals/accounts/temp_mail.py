"""Temporary email addresses for receiving verification emails.

Keyless temp-mail providers, no API keys needed. Used by the account
creator when a signup flow demands email verification: grab an address,
hand it to the site, then poll the inbox for the code/link.

Providers (all keyless):
- 1secmail — real REST API (getMessages/readMessage/genRandomMailbox),
  pure GET+JSON, no auth. The primary.
- GuerrillaMail — AJAX API with session token, no account needed.

The interface mirrors ``temp_sms`` so callers get the same shape:
grab an address, wait for a message, extract the verification code.
Use :func:`grab_address_cascade` to try providers in order.
"""

from __future__ import annotations

import json
import random
import re
import secrets
import string
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MailMessage",
    "TempAddress",
    "TempMailProvider",
    "OneSecMailProvider",
    "GuerrillaMailProvider",
    "MailTmProvider",
    "get_provider",
    "grab_address",
    "grab_address_cascade",
    "wait_message",
    "wait_code",
    "delete_message",
    "PROVIDERS",
    "CASCADE_PROVIDERS",
]


def _fetch_json(url: str, timeout: float = 20,
                headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, headers=headers or {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _fetch_text(url: str, timeout: float = 20,
                data: bytes | None = None) -> str:
    req = urllib.request.Request(
        url, data=data,
        headers={"User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


@dataclass
class MailMessage:
    """One received email."""
    id: str = ""
    sender: str = ""
    subject: str = ""
    date: str = ""
    body_text: str = ""
    body_html: str = ""
    code: str = ""  # extracted verification code, if any

    #: 4-8 digit runs are the usual OTP shape
    _CODE_RE = re.compile(r"(?<!\d)(\d{4,8})(?!\d)")

    def __post_init__(self) -> None:
        if not self.code:
            hay = f"{self.subject}\n{self.body_text}"
            m = self._CODE_RE.search(hay.replace(" ", "").replace("-", ""))
            if not m:
                m = self._CODE_RE.search(hay)
            if m:
                self.code = m.group(1)


@dataclass
class TempAddress:
    """A temporary receivable email address."""
    address: str = ""       # full address, e.g. user@1secmail.com
    login: str = ""         # local part
    domain: str = ""        # domain part
    provider: str = ""      # provider name
    token: str = ""         # provider session token (guerrillamail)


class TempMailProvider:
    """Interface every temp-mail provider implements."""

    name: str = ""

    def grab_address(self) -> TempAddress:
        raise NotImplementedError

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        raise NotImplementedError

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        """Fetch the full body for a message listing entry."""
        return msg

    def wait_for_message(self, addr: TempAddress, *,
                         timeout_s: float = 120,
                         poll_s: float = 8,
                         sender_hint: str = "",
                         subject_hint: str = "") -> MailMessage | None:
        deadline = time.time() + timeout_s
        seen: set[str] = set()
        while time.time() < deadline:
            try:
                msgs = self.get_messages(addr)
            except Exception as exc:  # noqa: BLE001
                _log.debug("%s poll failed: %s", self.name, exc)
                msgs = []
            for m in msgs:
                if m.id in seen:
                    continue
                seen.add(m.id)
                if sender_hint and sender_hint.lower() not in m.sender.lower():
                    continue
                if subject_hint and subject_hint.lower() not in m.subject.lower():
                    continue
                try:
                    m = self.read_message(addr, m)
                except Exception as exc:  # noqa: BLE001
                    _log.debug("%s read failed: %s", self.name, exc)
                return m
            time.sleep(poll_s)
        return None

    def delete_message(self, addr: TempAddress,
                       msg: MailMessage) -> bool:
        """Delete one message. Default: unsupported (returns False)."""
        return False

    def probe(self) -> bool:
        """True when the provider looks reachable. Default: try a cheap
        call. Never raises — a failed probe means "skip me"."""
        return True


class OneSecMailProvider(TempMailProvider):
    """1secmail.com — real REST API, keyless, pure GET+JSON."""

    name = "1secmail"
    BASE = "https://www.1secmail.com/api/v1/"

    def _get(self, **params: str) -> Any:
        qs = urllib.parse.urlencode(params)
        return _fetch_json(self.BASE + "?" + qs)

    def grab_address(self) -> TempAddress:
        # try random mailbox first (fastest), fall back to manual
        try:
            addrs = self._get(action="genRandomMailbox", count="1")
            if addrs:
                address = addrs[0]
                login, domain = address.split("@", 1)
                return TempAddress(address=address, login=login,
                                   domain=domain, provider=self.name)
        except Exception as exc:  # noqa: BLE001
            _log.debug("1secmail genRandomMailbox failed: %s", exc)
        domains = self._get(action="getDomainList")
        login = "".join(random.choices(string.ascii_lowercase + string.digits,
                                       k=12))
        domain = domains[0] if domains else "1secmail.com"
        return TempAddress(address=f"{login}@{domain}", login=login,
                           domain=domain, provider=self.name)

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        raw = self._get(action="getMessages", login=addr.login,
                        domain=addr.domain)
        out = []
        for m in raw or []:
            out.append(MailMessage(
                id=str(m.get("id", "")),
                sender=str(m.get("from", "")),
                subject=str(m.get("subject", "")),
                date=str(m.get("date", "")),
            ))
        return out

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        raw = self._get(action="readMessage", login=addr.login,
                        domain=addr.domain, id=msg.id)
        msg.body_text = str(raw.get("textBody", "") or "")
        msg.body_html = str(raw.get("htmlBody", "") or "")
        msg.sender = str(raw.get("from", "") or msg.sender)
        msg.subject = str(raw.get("subject", "") or msg.subject)
        # re-run code extraction now that we have the body
        msg.code = ""
        msg.__post_init__()
        return msg


class GuerrillaMailProvider(TempMailProvider):
    """GuerrillaMail — AJAX API, session token, no account."""

    name = "guerrillamail"
    BASE = "https://api.guerrillamail.com/ajax.php"

    def _call(self, **params: str) -> Any:
        qs = urllib.parse.urlencode(params)
        return _fetch_json(self.BASE + "?" + qs)

    def grab_address(self) -> TempAddress:
        raw = self._call(f="get_email_address", ip="127.0.0.1",
                         agent="Mozilla/5.0")
        address = str(raw.get("email_addr", ""))
        token = str(raw.get("sid_token", ""))
        login = address.split("@")[0] if "@" in address else address
        domain = address.split("@")[1] if "@" in address else ""
        return TempAddress(address=address, login=login, domain=domain,
                           provider=self.name, token=token)

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        raw = self._call(f="check_email", sid_token=addr.token, seq="0")
        out = []
        for m in (raw.get("list") or []):
            out.append(MailMessage(
                id=str(m.get("mail_id", "")),
                sender=str(m.get("mail_from", "")),
                subject=str(m.get("mail_subject", "")),
                date=str(m.get("mail_date", "")),
                body_text=str(m.get("mail_excerpt", "")),
            ))
        return out

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        raw = self._call(f="fetch_email", sid_token=addr.token,
                         email_id=msg.id)
        msg.body_text = str(raw.get("mail_body", "") or msg.body_text)
        # strip HTML tags for code extraction
        text = re.sub(r"<[^>]+>", " ", msg.body_text)
        msg.body_text = text
        msg.code = ""
        msg.__post_init__()
        return msg


class MailTmProvider(TempMailProvider):
    """mail.tm (and mail.gw — identical API) — the modern keyless temp-mail
    REST API and the primary of the cascade.

    Flow: GET /domains → POST /accounts {address, password} (201) →
    POST /token {address, password} → JWT → GET /messages (Bearer).
    Accounts are random-password JWT accounts; the JWT rides on
    ``TempAddress.token`` so ``wait_message`` works unchanged.
    """

    name = "mailtm"
    BASE = "https://api.mail.tm"
    _GW_BASE = "https://api.mail.gw"

    def __init__(self, base: str | None = None) -> None:
        self._base = base or self.BASE

    def _post_json(self, path: str, payload: dict[str, Any],
                   token: str = "") -> tuple[int, Any]:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "User-Agent": "Mozilla/5.0 (Linux; Android 10)"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self._base + path, data=data,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8", "replace")
                return resp.status, (json.loads(body) if body else {})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                return exc.code, (json.loads(body) if body else {})
            except ValueError:
                return exc.code, {}

    def _get_auth(self, path: str, token: str) -> Any:
        return _fetch_json(
            self._base + path, timeout=20,
            headers={"Authorization": f"Bearer {token}",
                     "User-Agent": "Mozilla/5.0 (Linux; Android 10)"},
        )

    def probe(self) -> bool:
        """True when the domain listing parses. Never raises."""
        try:
            domains = self.list_domains()
            return bool(domains)
        except Exception:  # noqa: BLE001
            _log.debug("mail.tm probe failed")
            return False

    def list_domains(self) -> list[str]:
        raw = _fetch_json(self._base + "/domains", timeout=20)
        members = raw.get("hydra:member") or []
        return [str(m.get("domain", "")) for m in members
                if m.get("domain")]

    def grab_address(self) -> TempAddress:
        domains = self.list_domains()
        domain = domains[0] if domains else "mail.tm"
        login = "".join(random.choices(string.ascii_lowercase +
                                       string.digits, k=12))
        address = f"{login}@{domain}"
        password = "".join(secrets.choice(
            string.ascii_letters + string.digits) for _ in range(20))
        status, _ = self._post_json("/accounts",
                                    {"address": address, "password": password})
        if status not in (200, 201):
            raise RuntimeError(
                f"mail.tm account creation failed (HTTP {status})")
        status, tok = self._post_json(
            "/token", {"address": address, "password": password})
        jwt = str(tok.get("token", "")) if status == 200 else ""
        if not jwt:
            raise RuntimeError("mail.tm token fetch failed")
        return TempAddress(address=address, login=login, domain=domain,
                           provider=self.name, token=jwt)

    def _sender(self, m: dict[str, Any]) -> str:
        frm = m.get("from") or {}
        addr = frm.get("address", "") if isinstance(frm, dict) else str(frm)
        name = frm.get("name", "") if isinstance(frm, dict) else ""
        return f"{name} <{addr}>".strip() if name else str(addr)

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        if not addr.token:
            return []
        raw = self._get_auth("/messages", addr.token)
        out = []
        for m in raw.get("hydra:member") or []:
            out.append(MailMessage(
                id=str(m.get("id", "")),
                sender=self._sender(m),
                subject=str(m.get("subject", "")),
                date=str(m.get("createdAt", "")),
                body_text=str(m.get("intro", "")),
            ))
        return out

    def read_message(self, addr: TempAddress,
                     msg: MailMessage) -> MailMessage:
        if not addr.token:
            return msg
        raw = self._get_auth(f"/messages/{msg.id}", addr.token)
        msg.body_text = str(raw.get("text", "") or msg.body_text)
        msg.body_html = str(raw.get("html", "") or "")
        msg.sender = self._sender(raw) or msg.sender
        msg.subject = str(raw.get("subject", "") or msg.subject)
        msg.code = ""
        msg.__post_init__()
        return msg

    def delete_message(self, addr: TempAddress,
                       msg: MailMessage) -> bool:
        """DELETE /messages/{id} — 204 on success."""
        if not addr.token:
            return False
        import urllib.request
        req = urllib.request.Request(
            f"{self._base}/messages/{msg.id}",
            headers={"Authorization": f"Bearer {addr.token}",
                     "User-Agent": "Mozilla/5.0 (Linux; Android 10)"},
            method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status in (200, 204)
        except Exception as exc:  # noqa: BLE001
            _log.debug("mail.tm delete failed: %s", exc)
            return False


PROVIDERS: dict[str, type[TempMailProvider]] = {
    "mailtm": MailTmProvider,
    "1secmail": OneSecMailProvider,
    "guerrillamail": GuerrillaMailProvider,
}

#: cascade order — mail.tm first (cleanest keyless REST API), then
#: 1secmail, then guerrilla. Providers whose probe() fails are skipped.
CASCADE_PROVIDERS = ["mailtm", "1secmail", "guerrillamail"]


def get_provider(name: str = "1secmail") -> TempMailProvider:
    cls = PROVIDERS.get(name.lower())
    if cls is None:
        raise ValueError(f"unknown temp-mail provider {name!r} "
                         f"(known: {sorted(PROVIDERS)})")
    return cls()


def grab_address(provider: str = "1secmail") -> TempAddress:
    return get_provider(provider).grab_address()


def grab_address_cascade(providers: list[str] | None = None) -> TempAddress:
    """Try providers in order until one yields a usable address.

    Providers whose ``probe()`` fails are skipped silently — a down
    provider is "skip me", never an exception to the caller.
    """
    last: Exception | None = None
    for name in providers or CASCADE_PROVIDERS:
        try:
            prov = get_provider(name)
        except Exception as exc:  # noqa: BLE001
            _log.debug("temp-mail provider %s unavailable: %s", name, exc)
            last = exc
            continue
        try:
            if not prov.probe():
                _log.debug("temp-mail provider %s probe failed — skipping",
                           name)
                continue
            addr = prov.grab_address()
            if addr.address:
                _log.info("temp-mail: got %s via %s", addr.address, name)
                return addr
        except Exception as exc:  # noqa: BLE001
            _log.debug("temp-mail provider %s failed: %s", name, exc)
            last = exc
    raise RuntimeError(f"all temp-mail providers failed (last: {last})")


def delete_message(addr: TempAddress, msg: MailMessage,
                   provider: str | None = None) -> bool:
    """Delete one message via its provider. Returns False when the
    provider doesn't support deletion."""
    prov = get_provider(provider or addr.provider or CASCADE_PROVIDERS[0])
    return prov.delete_message(addr, msg)


def wait_message(addr: TempAddress, *, timeout_s: float = 180,
                 poll_s: float = 8, sender_hint: str = "",
                 subject_hint: str = "",
                 provider: str | None = None) -> MailMessage | None:
    prov = get_provider(provider or addr.provider or "1secmail")
    return prov.wait_for_message(addr, timeout_s=timeout_s, poll_s=poll_s,
                                 sender_hint=sender_hint,
                                 subject_hint=subject_hint)


def wait_code(addr: TempAddress, *, timeout_s: float = 180,
              poll_s: float = 8, sender_hint: str = "",
              provider: str | None = None) -> str | None:
    """Wait for a verification email and return the extracted code."""
    msg = wait_message(addr, timeout_s=timeout_s, poll_s=poll_s,
                       sender_hint=sender_hint, provider=provider)
    return msg.code if msg else None
