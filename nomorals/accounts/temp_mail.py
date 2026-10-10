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
import string
import time
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
    "get_provider",
    "grab_address",
    "grab_address_cascade",
    "wait_message",
    "wait_code",
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


PROVIDERS: dict[str, type[TempMailProvider]] = {
    "1secmail": OneSecMailProvider,
    "guerrillamail": GuerrillaMailProvider,
}

#: cascade order — 1secmail first (simplest API), guerrilla as fallback
CASCADE_PROVIDERS = ["1secmail", "guerrillamail"]


def get_provider(name: str = "1secmail") -> TempMailProvider:
    cls = PROVIDERS.get(name.lower())
    if cls is None:
        raise ValueError(f"unknown temp-mail provider {name!r} "
                         f"(known: {sorted(PROVIDERS)})")
    return cls()


def grab_address(provider: str = "1secmail") -> TempAddress:
    return get_provider(provider).grab_address()


def grab_address_cascade(providers: list[str] | None = None) -> TempAddress:
    """Try providers in order until one yields a usable address."""
    last: Exception | None = None
    for name in providers or CASCADE_PROVIDERS:
        try:
            addr = grab_address(name)
            if addr.address:
                _log.info("temp-mail: got %s via %s", addr.address, name)
                return addr
        except Exception as exc:  # noqa: BLE001
            _log.debug("temp-mail provider %s failed: %s", name, exc)
            last = exc
    raise RuntimeError(f"all temp-mail providers failed (last: {last})")


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
