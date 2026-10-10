"""Temporary phone numbers for receiving SMS verification codes.

Free SMS-receive providers, no API keys needed.  Used by the account
creator when a signup flow demands phone verification: grab a number,
hand it to the site, then poll the inbox for the code.

Providers (all keyless):
- simcodes.net — country pages list numbers, per-number pages show the
  inbox as server-rendered HTML.  Some messages are login-gated, but
  most verification codes are fully visible.
- 7sim.net — free, no-registration temp numbers from 50+ countries,
  real SIM-based (better deliverability than VoIP pools).  Listing
  structure is best-effort and probe-gated: the provider self-checks
  before the cascade uses it.

The interface is provider-agnostic so new sources slot in without
touching callers.  Use :func:`grab_number_cascade` to try providers in
order until one yields a usable number.
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
    "SevenSimProvider",
    "get_provider",
    "grab_number",
    "grab_number_cascade",
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
    _AGE_RE = re.compile(
        r"(\d+)\s*(second|minute|hour|day)s?\s*ago|just\s*now",
        re.I)

    def age_seconds(self) -> float | None:
        """Parse the human age string into seconds (None if unknown)."""
        m = self._AGE_RE.search(self.received or "")
        if not m:
            return None
        if m.group(0).lower().startswith("just"):
            return 0.0
        qty, unit = int(m.group(1)), m.group(2).lower()
        return qty * {"second": 1, "minute": 60,
                      "hour": 3600, "day": 86400}[unit]

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
    freshness: float = 0.0  # 0..1 activity hint — fresher numbers burn
                            # less wait time; providers set it when the
                            # listing shows recent activity


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

    def probe(self) -> bool:
        """True when the provider looks reachable. Never raises —
        a failed probe means "skip me" in the cascade."""
        return True

    def rank_numbers(self, numbers: list[TempNumber]) -> list[TempNumber]:
        """Freshest, most-usable numbers first.

        Online numbers beat offline ones; fully-resolved numbers beat
        masked-only ones; provider freshness hints break ties. Stale
        numbers burn signup time — ranking is the cheapest speedup.
        """
        def key(n: TempNumber) -> tuple:
            return (not n.online, not bool(n.number), -n.freshness)
        return sorted(numbers, key=key)

    def wait_for_code(self, number: TempNumber, *,
                      sender_hint: str = "",
                      timeout: float = 120,
                      poll_every: float = 10,
                      seen_store: dict | None = None) -> str:
        """Poll until a verification code arrives. Returns the code or "".

        ``seen_store`` is an optional persistent dict (e.g. the signup
        driver's KV) holding already-seen message keys — a restarted
        wait resumes instead of re-reading old messages as new.
        """
        deadline = time.time() + timeout
        store_key = f"sms_seen:{self.name}:{number.inbox_id or number.number}"
        seen: set[str] = set((seen_store or {}).get(store_key, ()))
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
                if seen_store is not None:
                    seen_store[store_key] = sorted(seen)
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


class SevenSimProvider(TempSmsProvider):
    """7sim.net — free no-registration temp numbers, 50+ countries.

    BEST-EFFORT / probe-gated: the listing HTML structure was researched
    (service confirmed live, free, no login, real SIM numbers) but the
    exact markup was not fully verified, so every parse is tolerant and
    :meth:`probe` must succeed before the cascade trusts this provider.
    A failed probe means "skip me" — never an exception to the caller.

    Flow: homepage → discover country/number listing links → number
    inbox pages with server-rendered messages.
    """

    name = "7sim"
    _BASE = "https://7sim.net"

    #: tolerant link patterns for number listing / inbox pages
    _LISTING_PATTERNS = (
        r'href="(/[^"]*(?:temporary-numbers|numbers|receive-sms)[^"]*)"',
        r'href="(/country/[^"]+)"',
        r'href="(/[^"]*united-states[^"]*)"',
    )
    _NUMBER_PATTERNS = (
        r'href="(/[^"]*(?:number|sms)/[^"]*)"[^>]*>\s*(\+\d[\d\s*\-]{5,})',
        r'(\+\d[\d\s*\-]{6,})\s*</a>',
    )

    def probe(self) -> bool:
        """True if the listing structure parses. Never raises."""
        try:
            return bool(self.list_numbers(limit=3))
        except Exception:  # noqa: BLE001 - probe failure = unavailable
            _log.debug("7sim probe failed")
            return False

    def _discover_listing_urls(self, html: str) -> list[str]:
        urls: list[str] = []
        for pattern in self._LISTING_PATTERNS:
            for m in re.finditer(pattern, html, re.I):
                url = m.group(1)
                if url.startswith("/"):
                    url = self._BASE + url
                if url not in urls:
                    urls.append(url)
        return urls[:5]

    def list_numbers(self, country: str = "us",
                     limit: int = 20) -> list[TempNumber]:
        try:
            home = _fetch(self._BASE + "/")
        except Exception:  # noqa: BLE001
            return []
        out: list[TempNumber] = []
        pages = [self._BASE + "/"] + self._discover_listing_urls(home)
        for page_url in pages:
            try:
                html = home if page_url == self._BASE + "/" else _fetch(
                    page_url)
            except Exception:  # noqa: BLE001
                continue
            text = re.sub(r"<script.*?</script>", " ", html,
                          flags=re.I | re.S)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
            for pattern in self._NUMBER_PATTERNS:
                for m in re.finditer(pattern, text, re.I):
                    raw_num = (m.group(2) if m.lastindex and m.lastindex >= 2
                               else m.group(1))
                    digits = re.sub(r"\D", "", raw_num or "")
                    if len(digits) < 7:
                        continue
                    link = ""
                    try:
                        link = m.group(1)
                        if link.startswith("/"):
                            link = self._BASE + link
                    except IndexError:
                        pass
                    out.append(TempNumber(
                        number=f"+{digits}",
                        masked=f"+{digits[:6]}****",
                        country=country.lower(),
                        provider=self.name,
                        inbox_id=link,
                    ))
                    if len(out) >= limit:
                        return out
            if out:
                break
        return out

    def get_messages(self, number: TempNumber,
                     limit: int = 20) -> list[SmsMessage]:
        inbox_url = number.inbox_id or ""
        if not inbox_url.startswith("http"):
            return []
        try:
            html = _fetch(inbox_url)
        except Exception:  # noqa: BLE001
            return []
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        out: list[SmsMessage] = []
        # sender blocks: "From: <sender> <body>" variants
        for m in re.finditer(
                r"(?:from|sender)[:\s]+([+\d][\d\s*\-]{5,20}?)\s+"
                r"(.{10,300}?)\s+(?:\d+\s+(?:second|minute|hour|day)s?\s+ago"
                r"|just now|today)",
                text, re.I):
            sender, body = m.group(1).strip(), m.group(2).strip()
            if not body or len(body) < 4:
                continue
            out.append(SmsMessage(sender=sender, body=body))
            if len(out) >= limit:
                break
        return out


PROVIDERS["7sim"] = SevenSimProvider


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
    n = prov.rank_numbers(numbers)[0]
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
              timeout: float = 180,
              seen_store: dict | None = None) -> str:
    """Wait for an SMS verification code on a grabbed number.

    ``number_info`` is the dict returned by :func:`grab_number` (or
    :func:`grab_number_cascade`).  Returns the code or "" on timeout.
    ``seen_store`` persists seen-message keys across restarts.
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
                             timeout=timeout, seen_store=seen_store)


#: cascade order — verified providers first; probe-gated ones after.
CASCADE_PROVIDERS = ("simcodes", "7sim")


def grab_number_cascade(
    country: str = "us",
    providers: tuple[str, ...] = CASCADE_PROVIDERS,
    *,
    extra_countries: tuple[str, ...] = ("uk",),
    numbers_per_provider: int = 3,
) -> dict[str, Any]:
    """Grab a temp number, trying providers (then countries) in order.

    Probe-gated providers (e.g. 7sim) self-check before use and are
    silently skipped when their structure doesn't parse.  Returns the
    first usable number dict, or ``{"status": "failed", "notes": ...}``
    when every source is exhausted — the signup driver treats that as
    "genuinely stuck" and hands to the owner.
    """
    notes: list[str] = []
    for prov_name in providers:
        try:
            prov = get_provider(prov_name)
        except ValueError as exc:
            notes.append(str(exc))
            continue
        probe = getattr(prov, "probe", None)
        if callable(probe):
            try:
                if not probe():
                    notes.append(f"{prov_name}: probe failed, skipped")
                    continue
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{prov_name}: probe error {exc}")
                continue
        for ctry in (country,) + tuple(
                c for c in extra_countries if c != country):
            try:
                numbers = prov.list_numbers(
                    country=ctry, limit=numbers_per_provider)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{prov_name}/{ctry}: {exc}")
                continue
            if not numbers:
                notes.append(f"{prov_name}/{ctry}: no numbers listed")
                continue
            n = prov.rank_numbers(numbers)[0]
            _log.info("temp-sms cascade: using %s number %s",
                      prov_name, n.masked or n.number)
            return {
                "status": "ok",
                "number": n.number,
                "masked": n.masked,
                "country": n.country,
                "country_name": n.country_name,
                "provider": n.provider,
                "inbox_id": n.inbox_id,
            }
    return {"status": "failed",
            "notes": "all temp-sms sources exhausted: "
                     + "; ".join(notes)}
