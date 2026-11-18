"""Social surface enumeration.

Two questions, both answered by a plain anonymous GET:

* is this username still free on platform X (registration spam target)?
* is this email already registered on platform X?

Availability is read from the platform's own profile URL: 404/redirect to
a signup page = free, 200 with a profile marker = taken. A site that
answers 200 for everything is marked INCONCLUSIVE rather than counted
free — the same discipline the exposure scanner uses.
"""
from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["SITES", "enum_username", "enum_email", "register"]

#: (id, profile URL template, taken-marker regex)
SITES: tuple[tuple[str, str, str], ...] = (
    ("github", "https://github.com/{}", r"<title>[^<]*\(@"),
    ("gitlab", "https://gitlab.com/{}", r"js-profile-header|profile-header"),
    ("reddit", "https://www.reddit.com/user/{}", r"<shreddit-app|user-profile"),
    ("pinterest", "https://www.pinterest.com/{}/", r"following|followers"),
    ("telegram", "https://t.me/{}", r"tgme_page_title"),
    ("tiktok", "https://www.tiktok.com/@{}", r'"uniqueId"'),
    ("instagram", "https://www.instagram.com/{}/", r"og:title"),
    ("twitter", "https://x.com/{}", r"profile|followers"),
    ("medium", "https://medium.com/@{}", r"@" ),
    ("devto", "https://dev.to/{}", r"profile-header|joined"),
    ("behance", "https://www.behance.net/{}", r"profile"),
    ("dribbble", "https://dribbble.com/{}", r"profile-name|shots"),
    ("twitch", "https://www.twitch.tv/{}", r"channel-info|followers"),
    ("soundcloud", "https://soundcloud.com/{}", r"soundCloudSection|profile"),
    ("vimeo", "https://vimeo.com/{}", r"profile_header|bio"),
    ("keybase", "https://keybase.io/{}", r"profile"),
    ("aboutme", "https://about.me/{}", r"profile"),
    ("producthunt", "https://www.producthunt.com/@{}", r"profile"),
    ("replit", "https://replit.com/@{}", r"profile"),
    ("codepen", "https://codepen.io/{}", r"profile"),
    ("gravatar", "https://gravatar.com/{}", r"profile"),
    ("linktr", "https://linktr.ee/{}", r"profile"),
    ("mastodon", "https://mastodon.social/@{}", r"profile"),
    ("threads", "https://www.threads.net/@{}", r"profile"),
)

_EMAIL_FREE = re.compile(r"not (?:found|registered)|no account|available", re.I)
_EMAIL_TAKEN = re.compile(r"already (?:registered|taken|in use)|"
                          r"sign in instead|log in instead", re.I)


@dataclass
class Hit:
    site: str
    target: str
    state: str            # free | taken | inconclusive | error
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"site": self.site, "target": self.target,
                "state": self.state, "evidence": self.evidence}


def _probe(site: str, url: str, marker: str, target: str,
           timeout: float, proxy: Optional[str]) -> Hit:
    client = HttpClient(timeout=timeout, proxy_url=proxy or "")
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return Hit(site, target, "error", f"{type(exc).__name__}: {str(exc)[:60]}")
    body = (resp.text or "")[:40_000]
    if resp.status == 404:
        return Hit(site, target, "free", "http 404")
    if resp.status == 200:
        if marker and re.search(marker, body, re.I):
            return Hit(site, target, "taken", "http 200 profile marker")
        return Hit(site, target, "inconclusive", "http 200 no marker")
    return Hit(site, target, "inconclusive", f"http {resp.status}")


def enum_username(username: str, *, sites: Optional[list[str]] = None,
                  workers: int = 12, timeout: float = 15.0,
                  proxy_factory: Optional[Callable[[], Optional[str]]] = None,
                  ) -> list[Hit]:
    wanted = [s for s in SITES if not sites or s[0] in sites]
    out: list[Hit] = []

    def job(row: tuple[str, str, str]) -> Hit:
        sid, tmpl, marker = row
        proxy = proxy_factory() if proxy_factory else None
        return _probe(sid, tmpl.format(username), marker, username, timeout, proxy)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        out.extend(pool.map(job, wanted))
    return out


def enum_email(email: str, *, timeout: float = 15.0,
               proxy: Optional[str] = None) -> list[Hit]:
    """Email-registered detection. Only sites that expose an anonymous
    answer are listed — the rest would be guesses."""
    local = email.split("@")[0]
    checks = (
        ("gravatar", f"https://gravatar.com/{local}", r"profile"),
        ("github-search", f"https://github.com/search?q={email}&type=users",
         r"user-list-item|search-title"),
    )
    out: list[Hit] = []
    for sid, url, marker in checks:
        out.append(_probe(sid, url, marker, email, timeout, proxy))
    return out


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "social_enum",
        description="Check a username's availability across social platforms "
                    "(registration-spam targeting) or an email's registration "
                    "status where a platform answers anonymously.",
        capability=Capability.NET_OUT,
        parameters={"target": "str — username or email",
                    "kind": "str — username | email (default username)",
                    "workers": "int (optional, 12)"},
    )
    def social_enum(target: str, *, kind: str = "username",
                    workers: str = "") -> dict[str, Any]:
        w = int(workers) if str(workers).strip() else 12
        if (kind or "username").lower() == "email":
            hits = enum_email(target)
        else:
            hits = enum_username(target, workers=w)
        free = [h.site for h in hits if h.state == "free"]
        return {"target": target, "free_on": free,
                "taken_on": [h.site for h in hits if h.state == "taken"],
                "inconclusive": [h.site for h in hits if h.state == "inconclusive"],
                "hits": [h.as_dict() for h in hits]}
