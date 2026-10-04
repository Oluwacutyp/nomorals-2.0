"""Temporary phone numbers for receiving SMS verification codes.

Free SMS-receive providers, no API keys needed.  Used by the account
creator when a signup flow demands phone verification: grab a number,
hand it to the site, then poll the inbox for the code.

Providers (all keyless):
- simcodes.net — country pages list numbers, per-number pages show the
  inbox as server-rendered HTML.  Some messages are login-gated, but
  most verification codes are fully visible.

The interface is provider-agnostic so new sources slot in without
touching callers.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "SmsMessage",
    "TempNumber",
    "TempSmsProvider",
    "SimcodesProvider",
    "get_provider",
    "grab_number",
    "wait_code",
    "PROVIDERS",
]


@dataclass
class SmsMessage:
    """One received SMS."""
    sender: str = ""
    body: str = ""
    received: str = ""  # human age string, e.g. "23 minutes ago"
    code: str = ""      # extracted verification code, if any

    #: 4-8 digit runs are the usual OTP shape
    _CODE_RE = re.compile(r"(?<!\d)(\d{4,8})(?!\d)")

    def __post_init__(self) -> None:
        if not self.code and self.body:
            m = self._CODE_RE.search(self.body.replace(" ", "").replace("-", ""))
            # also try with separators intact for spaced codes like "244 340"
            if not m:
                m = self._CODE_RE.search(self.body)
            if m:
                self.code = m.group(1)


@dataclass
class TempNumber:
    """A temporary receivable phone number."""
    number: str = ""        # full international format, e.g. +15304031584
    masked: str = ""        # as shown on listing pages, e.g. +1530403****
    country: str = ""       # ISO code, e.g. "us"
    country_name: str = ""
    provider: str = ""      # provider name
    inbox_id: str = ""      # provider-internal id for the inbox page
    online: bool = True


class TempSmsProvider:
    """Interface every temp-SMS provider implements."""

    name: str = "base"

    def list_numbers(self, country: str = "us",
                     limit: int = 20) -> list[TempNumber]:
        """Return currently-available numbers for a country."""
        raise NotImplementedError

    def get_messages(self, number: TempNumber,
                     limit: int = 20) -> list[SmsMessage]:
        """Poll the inbox for a number obtained from list_numbers."""
        raise NotImplementedError

    def wait_for_code(self, number: TempNumber, *,
                      sender_hint: str = "",
                      timeout: float = 120,
                      poll_every: float = 10) -> str:
        """Poll until a verification code arrives. Returns the code or ""."""
        deadline = time.time() + timeout
        seen: set[str] = set()
        while time.time() < deadline:
            try:
                msgs = self.get_messages(number)
            except Exception as exc:  # noqa: BLE001 - poll never crashes
                _log.debug("%s inbox poll failed: %s", self.name, exc)
                msgs = []
            for m in msgs:
                key = f"{m.sender}|{m.body}"
                if key in seen:
                    continue
                seen.add(key)
                if m.code and (not sender_hint or
                               sender_hint.lower() in m.sender.lower()
                               or sender_hint.lower() in m.body.lower()):
                    return m.code
            time.sleep(poll_every)
        return ""


def _fetch(url: str, timeout: float = 20) -> str:
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Devon/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


class SimcodesProvider(TempSmsProvider):
    """simcodes.net — free virtual numbers with public SMS inboxes.

    Listing pages are static HTML; inbox pages server-render the
    messages, so no JS emulation is needed.  A few messages are
    login-gated ("You must login to view this content") and are
    skipped.
    """

    name = "simcodes"
    _BASE = "https://simcodes.net"
    _LOGIN_WALL = "you must login to view this content"

    def list_numbers(self, country: str = "us",
                     limit: int = 20) -> list[TempNumber]:
        html = _fetch(f"{self._BASE}/virtual-phone-number/country/"
                      f"{country.lower()}")
        out: list[TempNumber] = []
        # cards: masked number + link to the inbox page
        for m in re.finditer(
                r'(\+\d[\d*]{5,})\s*</p>.*?'
                r'href="(/free-phone-number-sms/(\d+))"',
                html, re.S):
            masked, path, inbox_id = m.group(1), m.group(2), m.group(3)
            # country name from the card
            cm = re.search(r'alt="([^"]+)"', html[max(0, m.start() - 600):m.start()])
            out.append(TempNumber(
                masked=masked.strip(),
                country=country.lower(),
                country_name=(cm.group(1) if cm else ""),
                provider=self.name,
                inbox_id=inbox_id,
            ))
            if len(out) >= limit:
                break
        return out

    def get_messages(self, number: TempNumber,
                     limit: int = 20) -> list[SmsMessage]:
        if not number.inbox_id:
            return []
        html = _fetch(f"{self._BASE}/free-phone-number-sms/"
                      f"{number.inbox_id}")
        # full number lives in the Livewire snapshot data (HTML-escaped)
        if not number.number:
            m = re.search(r'phone_id[&quot;": ]+(\d{7,15})', html)
            if m:
                number.number = f"+{m.group(1)}"
        # strip scripts/styles, then find sender blocks
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        out: list[SmsMessage] = []
        # "Sender: <digits> <body> <age> ago"
        for m in re.finditer(
                r"Sender:\s*(\d{6,15})\s+(.*?)\s+"
                r"(\d+\s+(?:second|minute|hour|day)s?\s+ago)",
                text, re.I):
            sender, body, age = m.group(1), m.group(2).strip(), m.group(3)
            if self._LOGIN_WALL in body.lower():
                continue
            # drop trailing junk tokens (livewire artifacts)
            body = re.sub(r"\s+[A-Za-z0-9+/=]{8,}\s*$", "", body).strip()
            if not body:
                continue
            out.append(SmsMessage(sender=sender, body=body,
                                  received=age))
            if len(out) >= limit:
                break
        return out


PROVIDERS: dict[str, type[TempSmsProvider]] = {
    "simcodes": SimcodesProvider,
}


def get_provider(name: str = "simcodes") -> TempSmsProvider:
    """Instantiate a temp-SMS provider by name."""
    cls = PROVIDERS.get(name.lower())
    if cls is None:
        raise ValueError(
            f"unknown temp-sms provider {name!r} "
            f"(available: {', '.join(sorted(PROVIDERS))})")
    return cls()


def grab_number(country: str = "us",
                provider: str = "simcodes") -> dict[str, Any]:
    """Grab a free temporary phone number for SMS verification.

    Module-level so callers (chat, CLI) don't need an ``AccountCreator``
    instance — this path touches no vault.  Returns a plain dict
    (number/masked/country/inbox_id/provider) suitable for
    :func:`wait_code`.
    """
    prov = get_provider(provider)
    numbers = prov.list_numbers(country=country, limit=10)
    if not numbers:
        return {"status": "failed",
                "notes": f"no {provider} numbers for {country}"}
    n = numbers[0]
    return {
        "status": "ok",
        "number": n.number,
        "masked": n.masked,
        "country": n.country,
        "country_name": n.country_name,
        "provider": n.provider,
        "inbox_id": n.inbox_id,
    }


def wait_code(number_info: dict[str, Any], *,
              sender_hint: str = "",
              timeout: float = 180) -> str:
    """Wait for an SMS verification code on a grabbed number.

    ``number_info`` is the dict returned by :func:`grab_number`.
    Returns the code or "" on timeout.
    """
    prov = get_provider(str(number_info.get("provider", "simcodes")))
    num = TempNumber(
        number=str(number_info.get("number", "")),
        masked=str(number_info.get("masked", "")),
        country=str(number_info.get("country", "us")),
        country_name=str(number_info.get("country_name", "")),
        provider=str(number_info.get("provider", "simcodes")),
        inbox_id=str(number_info.get("inbox_id", "")),
    )
    return prov.wait_for_code(num, sender_hint=sender_hint,
                             timeout=timeout)
