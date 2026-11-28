"""The public t.me page rung — no key, no login, no account.

A public @username's t.me page carries what Telegram itself
publishes for anyone: the display name, the verified badge,
the bio, the subscriber count, and the profile-photo URL.
It answers when every MTProto rung has missed and the
account is public — a public handle never returns an empty
dossier while this page answers.

Parsing is defensive on purpose: the page is read for its
Open Graph tags (og:title / og:description / og:image) plus
the verified-icon marker, and anything unparsable reads as
no answer (None), never a guessed field. The photo URL is a
link, never downloaded here — the dossier shows the photo
the same way the ledger path does.
"""
from __future__ import annotations

import html as _html
import re
import urllib.request
from collections.abc import Callable
from typing import Any

_OG_TITLE = re.compile(
    r'<meta\s+property="og:title"\s+content="([^"]*)"',
    re.IGNORECASE)
_OG_DESC = re.compile(
    r'<meta\s+property="og:description"\s+content="([^"]*)"',
    re.IGNORECASE)
_OG_IMAGE = re.compile(
    r'<meta\s+property="og:image"\s+content="([^"]*)"',
    re.IGNORECASE)
_SUBSCRIBERS = re.compile(r"([\d\s,\.]+)\s+subscribers",
                          re.IGNORECASE)


def _meta(pattern: re.Pattern[str], page: str) -> str:
    found = pattern.search(page)
    if not found:
        return ""
    return _html.unescape(found.group(1)).strip()


def fetch_tme(username: str, *,
              opener: Callable[..., Any] | None = None,
              timeout: float = 10.0) -> dict[str, Any] | None:
    """Read t.me/<username> for a public handle.

    Returns {"name", "verified", "bio", "photo_url",
    "subscribers"} or None when the page does not answer, is
    not a public profile, or cannot be parsed. Never raises.
    ``opener`` injects the fetch (tests); otherwise a plain
    urllib request with a browser user agent.
    """
    handle = str(username or "").strip().lstrip("@")
    if not handle or not handle.replace("_", "").isalnum():
        return None
    url = f"https://t.me/{handle}"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"})
        if opener is not None:
            response = opener(request)
        else:
            response = urllib.request.urlopen(
                request, timeout=timeout)
        with response as resp:
            page = resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a dead page reads None
        return None
    name = _meta(_OG_TITLE, page)
    if not name:
        return None
    description = _meta(_OG_DESC, page)
    bio = description
    subscribers = ""
    match = _SUBSCRIBERS.search(description)
    if match:
        subscribers = match.group(1).strip()
        bio = description[:match.start()].strip()
    return {
        "name": name,
        "verified": "verified-icon" in page,
        "bio": bio,
        "photo_url": _meta(_OG_IMAGE, page),
        "subscribers": subscribers,
    }
