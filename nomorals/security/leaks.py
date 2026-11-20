"""Leak and credential-exposure sources.

KEYLESS and real: Hudson Rock Cavalier (stealer-log infections by email,
username or domain) and ProxyNova's combo lookup. Both answer anonymously
today.

KEYED and real, activated by env var: HIBP, DeHashed, LeakCheck, IntelX,
Snusbase. A missing key is reported in `sources_skipped`, never faked.

Result shape is uniform: `identities` (email/user), `passwords` (plaintext
where the source gives it), `sessions` (URL/login pairs from stealer
logs), `platforms`. A source that returns nothing contributes nothing.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["KEYS", "Exposure", "harvest_exposure", "register"]

_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.I)

#: env var -> provider, for the keyed adapters
KEYS = {
    "NM_HIBP_KEY": "hibp",
    "NM_DEHASHED_KEY": "dehashed",
    "NM_LEAKCHECK_KEY": "leakcheck",
    "NM_INTELX_KEY": "intelx",
    "NM_SNUSBASE_KEY": "snusbase",
}


@dataclass
class Exposure:
    subject: str
    identities: set[str] = field(default_factory=set)
    passwords: set[str] = field(default_factory=set)
    sessions: list[dict[str, str]] = field(default_factory=list)
    platforms: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    malwares: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {"subject": self.subject,
                "identities": sorted(self.identities)[:60],
                "passwords": sorted(self.passwords)[:60],
                "sessions": self.sessions[:40],
                "platforms": sorted(self.platforms),
                "sources": sorted(self.sources),
                "malwares": sorted(self.malwares),
                "exposed": bool(self.identities or self.passwords
                                or self.sessions)}


def _cavalier(email: str = "", username: str = "", domain: str = "",
              timeout: float = 25.0) -> Exposure:
    """Hudson Rock Cavalier — stealer-log infections. Keyless."""
    client = HttpClient(timeout=timeout)
    if email:
        subject, path = email, f"search-by-email?email={urllib.parse.quote(email)}"
    elif username:
        subject, path = username, f"search-by-username?username={urllib.parse.quote(username)}"
    elif domain:
        subject, path = domain, f"search-by-domain?domain={urllib.parse.quote(domain)}"
    else:
        return Exposure(subject="")
    out = Exposure(subject=subject, sources={"cavalier"})
    try:
        resp = client.get("https://cavalier.hudsonrock.com/api/json/v2/osint-tools/"
                          + path)
    except Exception as exc:  # noqa: BLE001
        _log.info("cavalier failed: %s", exc)
        return out
    if not resp.ok:
        return out
    try:
        data = json.loads(resp.text or "{}")
    except ValueError:
        return out
    message = str(data.get("message") or "")
    if "not found" in message.lower():
        return out
    for row in data.get("data", {}).get("stealers") or []:
        if row.get("computer_name"):
            out.sources.add("stealer-log")
        for key in ("top_logins", "top_passwords"):
            for value in (row.get(key) or []):
                if not isinstance(value, str):
                    continue
                if key == "top_logins" and "@" in value:
                    out.identities.add(value.lower())
                elif key == "top_logins":
                    out.identities.add(value)
                else:
                    out.passwords.add(value)
        for url in (row.get("top_logins") or []):
            if url.startswith("http"):
                host = urllib.parse.urlsplit(url).hostname or ""
                if host:
                    out.platforms.add(host)
    for mal in data.get("data", {}).get("antiviruses") or []:
        if isinstance(mal, str):
            out.malwares.add(mal)
    return out


def _proxynova(email: str, timeout: float = 25.0) -> Exposure:
    """ProxyNova combolist lookup. Keyless, returns email:password rows."""
    out = Exposure(subject=email, sources={"proxynova"})
    client = HttpClient(timeout=timeout)
    try:
        resp = client.get("https://www.proxynova.com/tools/comb/api/query",
                          params={"q": email, "limit": "20"})
    except Exception as exc:  # noqa: BLE001
        _log.info("proxynova failed: %s", exc)
        return out
    if not resp.ok:
        return out
    for line in (resp.text or "").splitlines():
        line = line.strip()
        if ":" not in line or "@" not in line:
            continue
        addr, _, secret = line.partition(":")
        if email.lower() in addr.lower():
            out.identities.add(addr.lower())
            if secret:
                out.passwords.add(secret)
    return out


def _hibp(email: str, key: str, timeout: float = 25.0) -> Exposure:
    out = Exposure(subject=email, sources={"hibp"})
    client = HttpClient(timeout=timeout)
    try:
        resp = client.get(f"https://haveibeenpwned.com/api/v3/breachedaccount/"
                          f"{urllib.parse.quote(email)}?truncateResponse=false",
                          headers={"hibp-api-key": key, "User-Agent": "nm-leaks"})
    except Exception:  # noqa: BLE001
        return out
    if not resp.ok:
        return out
    try:
        for breach in json.loads(resp.text or "[]"):
            out.platforms.add(str(breach.get("Name") or ""))
            out.sources.add("breach")
    except ValueError:
        pass
    return out


def _dehashed(term: str, key: str, timeout: float = 30.0) -> Exposure:
    out = Exposure(subject=term, sources={"dehashed"})
    client = HttpClient(timeout=timeout)
    try:
        resp = client.get("https://api.dehashed.com/search",
                          params={"query": term, "size": "20"},
                          headers={"Authorization": f"Basic {key}",
                                   "Accept": "application/json"})
    except Exception:  # noqa: BLE001
        return out
    if not resp.ok:
        return out
    try:
        data = json.loads(resp.text or "{}")
    except ValueError:
        return out
    for row in data.get("entries") or []:
        if row.get("email"):
            out.identities.add(str(row["email"]).lower())
        if row.get("password"):
            out.passwords.add(str(row["password"]))
        if row.get("domain"):
            out.platforms.add(str(row["domain"]))
    return out


def harvest_exposure(*, email: str = "", username: str = "", domain: str = "",
                     timeout: float = 30.0) -> dict[str, Any]:
    subject = email or username or domain
    merged = Exposure(subject=subject)
    skipped: list[str] = []
    ran: list[str] = []

    if email:
        for fn, name in ((_cavalier, "cavalier"), (_proxynova, "proxynova")):
            try:
                part = fn(email, timeout=timeout)
                ran.append(name)
                merged.identities |= part.identities
                merged.passwords |= part.passwords
                merged.sessions += part.sessions
                merged.platforms |= part.platforms
                merged.malwares |= part.malwares
                merged.sources |= part.sources
            except Exception as exc:  # noqa: BLE001
                _log.info("%s failed: %s", name,
