"""Locale packs for family mode: N packs, Nigeria default, never a boundary.

A pack is DATA: country, default city, timezone label, languages,
currency, and the extra kid-safe blocked terms for that locale. Packs
never gate capability — every code path accepts every pack, and a
caller may pass an unknown pack code and still get the base behaviour
with that code recorded. The default is ``ng`` (Nigeria) because the
household this was built for lives there; it is a default, not a wall.
"""

from __future__ import annotations

from typing import Any

#: The default locale pack. Nigeria first, never Nigeria-only.
DEFAULT_LOCALE = "ng"

#: Base blocked terms applied in EVERY pack (English, word-boundary match).
BASE_BLOCKED_TERMS: tuple[str, ...] = (
    "porn", "pornography", "xxx", "nude", "naked", "sex video",
    "prostitute", "brothel", "strip club",
    "murder", "kill yourself", "suicide", "self-harm",
    "cocaine", "heroin", "marijuana", "weed", "cannabis", "meth",
    "casino", "betting", "gambling", "slot machine",
    "rape", "molest", "abuse",
)

LOCALE_PACKS: dict[str, dict[str, Any]] = {
    "ng": {
        "country": "Nigeria",
        "default_city": "Lagos",
        "timezone": "Africa/Lagos",
        "languages": ["en", "pcm", "yo", "ha", "ig"],
        "currency": "NGN",
        # Pidgin / local terms that trip the kid-safe filter in Nigeria.
        "extra_blocked_terms": ("ashawo", "runs girl", "yahoo boy", "juju", "ogogoro", "igbo smoke"),
    },
    "gh": {
        "country": "Ghana",
        "default_city": "Accra",
        "timezone": "Africa/Accra",
        "languages": ["en", "tw", "ga", "ee"],
        "currency": "GHS",
        "extra_blocked_terms": ("sakawa", "akpeteshie",),
    },
    "ke": {
        "country": "Kenya",
        "default_city": "Nairobi",
        "timezone": "Africa/Nairobi",
        "languages": ["en", "sw"],
        "currency": "KES",
        "extra_blocked_terms": ("changaa",),
    },
    "za": {
        "country": "South Africa",
        "default_city": "Johannesburg",
        "timezone": "Africa/Johannesburg",
        "languages": ["en", "zu", "xh", "af"],
        "currency": "ZAR",
        "extra_blocked_terms": (),
    },
    "us": {
        "country": "United States",
        "default_city": "New York",
        "timezone": "America/New_York",
        "languages": ["en", "es"],
        "currency": "USD",
        "extra_blocked_terms": ("hookup app",),
    },
    "gb": {
        "country": "United Kingdom",
        "default_city": "London",
        "timezone": "Europe/London",
        "languages": ["en"],
        "currency": "GBP",
        "extra_blocked_terms": ("fruit machine",),
    },
    "in": {
        "country": "India",
        "default_city": "Mumbai",
        "timezone": "Asia/Kolkata",
        "languages": ["en", "hi"],
        "currency": "INR",
        "extra_blocked_terms": ("satta", "gutka"),
    },
}


def get_locale_pack(code: str | None) -> dict[str, Any]:
    """Return the pack for ``code``; unknown codes get a generic pack.

    Never raises for an unknown locale: the pack system is a default,
    not a boundary. The returned dict always carries ``code``.
    """
    code = (code or DEFAULT_LOCALE).lower()
    pack = dict(LOCALE_PACKS.get(code, {}))
    if not pack:
        pack = {
            "country": "",
            "default_city": "",
            "timezone": "UTC",
            "languages": ["en"],
            "currency": "",
            "extra_blocked_terms": (),
        }
    pack["code"] = code
    return pack


def list_locale_packs() -> list[str]:
    """Sorted pack codes available out of the box."""
    return sorted(LOCALE_PACKS)


def blocked_terms_for(code: str | None) -> tuple[str, ...]:
    """Base terms + the pack's extra terms, de-duplicated, order kept."""
    pack = get_locale_pack(code)
    seen: set[str] = set()
    out: list[str] = []
    for term in (*BASE_BLOCKED_TERMS, *pack.get("extra_blocked_terms", ())):
        if term not in seen:
            seen.add(term)
            out.append(term)
    return tuple(out)
