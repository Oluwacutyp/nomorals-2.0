"""Bulk test-data generation for mock screens (FakeDetail
port Phase 5, census §29 + Family 5 §7).

A seeded generator of obviously synthetic personal and
identity test data — names, usernames, passwords, multi-
country phones, multi-country addresses, email-shaped
strings, company names, IMEI-shaped values, creative
names — in the shapes the screen models in :mod:`.schema`
consume, so one seed + count populates a whole screen
reproducibly.  A bulk screen builder per class
(:func:`bulk_list_screen`, :func:`bulk_call_history`,
:func:`bulk_chat_screen`, :func:`bulk_status_views`)
turns a seed into a populated ``ListScreen`` /
``CallHistoryScreen`` / ``Screen`` / ``StatusViewsScreen``
the renderers draw unchanged.

The safety boundary, stated once and enforced by
construction (census §17.6 + §29 risks):

- **Synthetic by shape.**  Generated people wear the
  house disposable surnames (``Trial`` / ``Demo`` /
  ``Test`` / ``Temp`` / ``Guest`` — the IdentityBank
  idiom), every record carries ``synthetic: True``, and
  the street names are demo streets.  Nothing generated
  here is, or can be mistaken for, a real person's
  details.
- **Phones are fictional by range.**  US numbers use the
  reserved fictional block ``+1 (555) 01xx``; UK numbers
  use Ofcom's reserved drama range ``+44 7700 900xxx``.
  Nigeria has no reserved fictional range, so Nigerian
  numbers (and the other countries') wear a zero
  subscriber block (``000 …``) — structurally valid,
  never an assigned number.
- **Email strings cannot receive mail.**  They are shaped
  ``<handle>@<provider>.example`` — the ``.example``
  domain can never resolve — and every record says so
  (``receives_mail: False``).  This is NOT temp mail:
  a real disposable inbox is a different organ
  (``accounts/temp_mail.py``) and is never used here.
- **IMEI values are format-valid, never assigned.**  A
  fixed test TAC (``35-000000``) plus a Luhn check digit;
  the record says ``assigned: False``.
- **No trial identities.**  This organ imports only the
  IdentityBank *name-pool constants* (census §29 reuse);
  it never imports, mints, stores or consumes a trial
  identity, and :meth:`BulkGenerator.identity` returns a
  plain dict — never a ``Persona`` — with no ``service``
  key and no storage behind it.
- **No generated faces.**  There is no face-generation
  backend wired (census §17.5).  :meth:`BulkGenerator.face`
  reports that honestly; generated people carry no
  avatar path, so the renderers draw the initials
  placeholder they already draw for avatar-less
  participants.
- **Counts are resource-scaled, not arbitrary.**
  Data-only generation caps at 999 records (FakeDetail's
  own address ceiling, census §7.3).  Populated screens
  cap at 200 rows: a list row is ~64–72 px tall, so 200
  rows is a ~14,000 px canvas — large but renderable;
  beyond that the PNG itself is the wall, and the error
  names the cap and the reason.

Determinism: every draw comes from one ``random.Random``
seeded at construction, and dates count from the fixed
:const:`REFERENCE_DATE` — the same seed gives identical
records on any day, on any machine.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import string
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any

from ....accounts.identity_bank import (
    DISPOSABLE_SURNAMES,
    INTERNATIONAL_FIRST_NAMES,
    LOCALE_FIRST_NAMES,
    NIGERIAN_FIRST_NAMES,
)
from .schema import (
    HISTORY_TYPES,
    LIST_PREVIEW_KINDS,
    CallHistoryEntry,
    CallHistoryScreen,
    DeviceChrome,
    FolderTab,
    ListRow,
    ListScreen,
    Message,
    Participant,
    Screen,
    SocialProfile,
    StatusViewer,
    StatusViewsScreen,
    StoryItem,
)

__all__ = [
    "BulkGenerator",
    "DemoDataError",
    "bulk_call_history",
    "bulk_chat_screen",
    "bulk_list_screen",
    "bulk_status_views",
    "export_records",
    "COUNTRIES",
    "CREATIVE_FAMILIES",
    "MAIL_PROVIDERS",
    "MAX_BULK_DATA",
    "MAX_BULK_SCREEN_ROWS",
    "REFERENCE_DATE",
]


class DemoDataError(ValueError):
    """A bulk-data request the organ cannot carry (bad
    kind / country / count) — the authoring layer turns
    this into the organ's usage error."""


#: The date bulk DOBs and time ladders count from — fixed
#: so the same seed reproduces identical records on any
#: day (a wall-clock "today" would not).
REFERENCE_DATE: date = date(2026, 11, 11)

#: Data-only generation ceiling (census §7.3: FakeDetail's
#: address generator offers up to 999 at once).
MAX_BULK_DATA = 999

#: Populated-screen ceiling — a list row is ~64–72 px
#: tall, so 200 rows is a ~14,000 px canvas; beyond that
#: the PNG's own memory is the wall.
MAX_BULK_SCREEN_ROWS = 200

#: Countries with a phone + address format in this organ.
COUNTRIES: tuple[str, ...] = ("ng", "us", "gb", "de", "fr", "in")

#: Mail variants the census names (§7.2) — labels on an
#: ``.example`` domain, never a real mailbox.
MAIL_PROVIDERS: tuple[str, ...] = (
    "gmail", "outlook", "hotmail", "yahoo", "mail",
    "rediffmail", "mailfence", "openmailbox", "10minutemail",
    "email")

#: Creative-name families the census names (§8) — the
#: seed list, built here from Devon's own word pools;
#: the census records no per-generator controls, so no
#: pool here claims to be FakeDetail's exact list.
CREATIVE_FAMILIES: tuple[str, ...] = (
    "nickname", "aesthetic", "band", "alien", "dj", "god",
    "horse", "orc", "planet", "dwarf", "warrior-cat",
    "personality", "insult", "bible-verse", "wow")

#: Company suffixes per country.
_COMPANY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "ng": ("Ltd", "Limited", "Enterprises", "Ventures"),
    "us": ("LLC", "Inc", "Co", "Studios"),
    "gb": ("Ltd", "Limited", "Group", "Studios"),
    "de": ("GmbH", "AG", "Studio"),
    "fr": ("SARL", "SAS", "Studio"),
    "in": ("Pvt Ltd", "Enterprises", "Studios"),
}

#: City pools per country (public geography — a city
#: alone is not a person's address).
_CITIES: dict[str, tuple[str, ...]] = {
    "ng": ("Ikeja", "Lekki", "Yaba", "Surulere", "Ikoyi",
           "Ajah", "Ogba", "Maryland", "Gbagada", "Ikorodu"),
    "us": ("New York", "Los Angeles", "Chicago", "Houston",
           "Phoenix", "Atlanta", "Seattle", "Austin"),
    "gb": ("London", "Manchester", "Birmingham", "Leeds",
           "Bristol", "Glasgow", "Liverpool"),
    "de": ("Berlin", "Munich", "Hamburg", "Cologne",
           "Frankfurt"),
    "fr": ("Paris", "Lyon", "Marseille", "Toulouse", "Nice"),
    "in": ("Mumbai", "Delhi", "Bengaluru", "Chennai",
           "Hyderabad", "Pune"),
}

#: State / region labels per country.
_STATES: dict[str, tuple[str, ...]] = {
    "ng": ("Lagos", "Oyo", "Ekiti", "Ogun", "Abuja FCT"),
    "us": ("NY", "CA", "IL", "TX", "AZ", "GA", "WA"),
    "gb": ("England", "Scotland", "Wales"),
    "de": ("Berlin", "Bavaria", "Hamburg", "NRW", "Hesse"),
    "fr": ("Île-de-France", "Auvergne-Rhône-Alpes",
           "Provence-Alpes-Côte d'Azur", "Occitanie"),
    "in": ("Maharashtra", "Delhi", "Karnataka", "Tamil Nadu",
           "Telangana"),
}

#: Street names — demo streets by construction, never a
#: real street a generated record could be tied to.
_DEMO_STREETS: tuple[str, ...] = (
    "Demo Close", "Test Avenue", "Sample Street",
    "Mock Road", "Trial Crescent", "Placeholder Lane",
    "Example Boulevard", "Sandbox Way")

#: Company-name word pools.
_COMPANY_FIRST: tuple[str, ...] = (
    "Golden", "Bright", "Swift", "Clear", "Prime", "Nova",
    "Urban", "Cedar", "Harbor", "Summit", "Willow", "Atlas",
    "Ember", "Juniper", "Marble", "Falcon")
_COMPANY_SECOND: tuple[str, ...] = (
    "Works", "Labs", "Studio", "Collective", "Goods",
    "Supply", "Media", "Craft", "Systems", "Trading",
    "Foods", "Textiles")

#: Neutral demo bios — generic test copy, never a fake
#: personal claim.
_BIOS: tuple[str, ...] = (
    "Demo account for screen tests.",
    "Test profile — not a real person.",
    "Mock contact for layout checks.",
    "Sample bio for a populated screen.",
    "Placeholder person, generated for testing.",
    "Demo identity — synthetic data only.")

#: Short preview/message lines for populated screens —
#: the demo idiom, neutral chat copy.
_PREVIEWS: tuple[str, ...] = (
    "the flyer is ready 🔥",
    "on my way",
    "see you at 10",
    "sending the files now",
    "that render came out clean",
    "morning! did you see the update",
    "the studio is booked for Saturday",
    "bringing the backdrops",
    "call me when you land",
    "half of the list, still typing",
    "okay this is the one. Post it.",
    "the invoice is in your inbox",
    "new drops on Friday",
    "guest list is done",
    "let me know when you are free",
    "the packing list draft",
)

#: Password alphabet — the creator.py shape
#: (``accounts/creator.py: generate_password``): letters
#: + digits, symbols optional.  The draws here come from
#: the seeded RNG (deterministic demo strings); the
#: crypto primitive stays the one for real credentials.
_PASSWORD_SYMBOLS = "!@#$%^&*()-_=+"

#: Deterministic avatar colours for generated people
#: (the renderers' initials placeholder wears these).
_AVATAR_COLORS: tuple[str, ...] = (
    "#E53935", "#8E24AA", "#3949AB", "#00897B", "#7CB342",
    "#FB8C00", "#6D4C41", "#546E7A", "#D81B60", "#1E88E5")

#: Locale -> first-name pool (the IdentityBank pools;
#: the owner's Nigerian scope first).
_LOCALES: dict[str, tuple[str, ...]] = dict(LOCALE_FIRST_NAMES)
_LOCALES.setdefault("ng", NIGERIAN_FIRST_NAMES)
_LOCALES.setdefault("intl", INTERNATIONAL_FIRST_NAMES)


def _luhn_check_digit(digits: str) -> str:
    """The Luhn check digit for a 14-digit IMEI body.
    In the full 15-digit number the check digit sits
    rightmost (never doubled); the body's rightmost
    digit is one step left, so it doubles — over the
    reversed body, the even positions double."""
    total = 0
    for pos, char in enumerate(reversed(digits)):
        value = int(char)
        if pos % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return str((10 - (total % 10)) % 10)


def _check_count(count: int, cap: int, what: str) -> int:
    """Validate a bulk count against its resource-scaled
    cap; the error names the cap and the reason."""
    try:
        n = int(count)
    except (TypeError, ValueError):
        raise DemoDataError(
            what + " count must be a whole number, got "
            + repr(count)) from None
    if n < 1:
        raise DemoDataError(
            what + " count must be at least 1, got "
            + str(n))
    if n > cap:
        raise DemoDataError(
            what + " count " + str(n) + " is over the "
            + str(cap) + " cap — "
            + ("a screen taller than that is a canvas "
               "the renderer cannot hold"
               if cap == MAX_BULK_SCREEN_ROWS
               else "999 is FakeDetail's own bulk ceiling "
               "(census §7.3)"))
    return n


class BulkGenerator:
    """Seeded bulk test-data generation.

    One instance = one seed: every draw comes from its
    own ``random.Random``, so the same seed + count
    reproduces identical records anywhere.  ``locale``
    picks the first-name pool (``ng_yoruba`` /
    ``ng_hausa`` / ``ng_igbo`` / ``ng`` / ``intl`` —
    the IdentityBank locale map).
    """

    def __init__(self, seed: int = 0, *,
                 locale: str = "ng") -> None:
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            raise DemoDataError(
                "seed must be a whole number, got "
                + repr(seed)) from None
        self.seed = seed_int
        self.rng = random.Random(seed_int)
        self.locale = str(locale or "ng").strip().lower()
        if self.locale not in _LOCALES:
            raise DemoDataError(
                "locale " + repr(locale) + " is not one of "
                + ", ".join(sorted(_LOCALES)))
        self._used_names: set[str] = set()

    # -- names ---------------------------------------------------

    def first_name(self) -> str:
        """A first name from the locale pool."""
        return self.rng.choice(_LOCALES[self.locale])

    def last_name(self) -> str:
        """A disposable surname — the house synthetic
        marker: a generated person is visibly test data,
        never a real person's name shape."""
        return self.rng.choice(DISPOSABLE_SURNAMES)

    def name(self) -> str:
        """A full synthetic name (unique within this
        generator — a repeat draw earns a numeral)."""
        for _ in range(20):
            full = self.first_name() + " " + self.last_name()
            if full not in self._used_names:
                self._used_names.add(full)
                return full
        base = self.first_name() + " " + self.last_name()
        n = 2
        while base + " " + str(n) in self._used_names:
            n += 1
        full = base + " " + str(n)
        self._used_names.add(full)
        return full

    def handle(self, first: str = "") -> str:
        """A social handle from a first name."""
        stem = re.sub(r"[^a-z0-9]+", "",
                      (first or self.first_name()).lower())
        stem = stem or "demo"
        return stem + "." + str(self.rng.randint(1, 999)) \
            if self.rng.random() < 0.5 else stem + "_" \
            + str(self.rng.randint(10, 99))

    def username(self, prefix: str = "demo") -> str:
        """A username in the creator.py shape (prefix +
        an 8-character lowercase/digit suffix) — drawn
        from the seeded RNG, so it reproduces."""
        suffix = "".join(self.rng.choice(
            string.ascii_lowercase + string.digits)
            for _ in range(8))
        return str(prefix or "demo") + "_" + suffix

    def bio(self) -> str:
        """A neutral demo bio."""
        return self.rng.choice(_BIOS)

    # -- contact shapes ------------------------------------------

    def email(self, provider: str = "gmail") -> dict[str, Any]:
        """An email-shaped TEST string: ``<handle>@
        <provider>.example``.  The ``.example`` domain can
        never resolve and the record says it cannot receive
        mail — this is not a mailbox (census §7.2)."""
        provider = str(provider or "gmail").strip().lower()
        if provider not in MAIL_PROVIDERS:
            raise DemoDataError(
                "mail provider " + repr(provider)
                + " is not one of "
                + ", ".join(MAIL_PROVIDERS))
        local = self.handle()
        return {"email": local + "@" + provider + ".example",
                "provider": provider,
                "receives_mail": False,
                "note": "test string — cannot receive mail"}

    def phone(self, country: str = "ng") -> dict[str, Any]:
        """A fictional phone number for ``country`` — the
        reserved fictional block where one exists (US
        555-01xx, UK 7700 900xxx), a zero subscriber
        block elsewhere.  ``fictional: True`` always."""
        country = str(country or "ng").strip().lower()
        if country not in COUNTRIES:
            raise DemoDataError(
                "country " + repr(country) + " is not one of "
                + ", ".join(COUNTRIES))
        if country == "us":
            number = "+1 (555) 01" + format(
                self.rng.randint(0, 99), "02d")
        elif country == "gb":
            number = "+44 7700 900" + format(
                self.rng.randint(0, 999), "03d")
        elif country == "ng":
            prefix = self.rng.choice(
                ("803", "805", "809", "810", "813", "816",
                 "703", "802", "812", "701", "902", "903"))
            number = "+234 " + prefix + " 000 " + format(
                self.rng.randint(0, 9999), "04d")
        elif country == "de":
            number = "+49 170 000 " + format(
                self.rng.randint(0, 9999), "04d")
        elif country == "fr":
            number = "+33 6 00 00 0" + format(
                self.rng.randint(0, 999), "03d")
        else:  # in
            number = "+91 98000 00" + format(
                self.rng.randint(0, 999), "03d")
        return {"number": number, "country": country,
                "fictional": True,
                "note": "test number — not attached to a "
                        "SIM or network"}

    def address(self, country: str = "ng") -> dict[str, Any]:
        """A synthetic address: a demo street (never a
        real street), a real city/state shape, and the
        country's null postal block — no real delivery
        point."""
        country = str(country or "ng").strip().lower()
        if country not in COUNTRIES:
            raise DemoDataError(
                "country " + repr(country) + " is not one of "
                + ", ".join(COUNTRIES))
        street = str(self.rng.randint(1, 120)) + " " \
            + self.rng.choice(_DEMO_STREETS)
        city = self.rng.choice(_CITIES[country])
        state = self.rng.choice(_STATES[country])
        if country == "us":
            postal = "00" + format(
                self.rng.randint(0, 999), "03d")
        elif country == "gb":
            postal = "ZZ" + str(self.rng.randint(1, 9)) \
                + " " + str(self.rng.randint(1, 9)) \
                + "ZZ"
        elif country == "ng":
            postal = "100000"
        elif country in ("de", "fr"):
            postal = "00000"
        else:  # in
            postal = "000000"
        line = street + ", " + city + ", " + state + " " \
            + postal + ", " + country.upper()
        return {"street": street, "city": city,
                "state": state, "postal": postal,
                "country": country, "line": line,
                "synthetic": True}

    # -- credentials / company / device --------------------------

    def password(self, length: int = 16, *,
                 symbols: bool = True) -> str:
        """A test password in the creator.py shape
        (letters + digits, symbols optional) — drawn from
        the seeded RNG.  A demo string, never a credential
        for a real account."""
        if int(length) < 4 or int(length) > 64:
            raise DemoDataError(
                "password length must be 4–64, got "
                + str(length))
        alphabet = string.ascii_letters + string.digits
        if symbols:
            alphabet += _PASSWORD_SYMBOLS
        return "".join(self.rng.choice(alphabet)
                       for _ in range(int(length)))

    def company(self, country: str = "ng") -> str:
        """A synthetic company name."""
        country = str(country or "ng").strip().lower()
        if country not in COUNTRIES:
            raise DemoDataError(
                "country " + repr(country) + " is not one of "
                + ", ".join(COUNTRIES))
        return (self.rng.choice(_COMPANY_FIRST) + " "
                + self.rng.choice(_COMPANY_SECOND) + " "
                + self.rng.choice(
                    _COMPANY_SUFFIXES[country]))

    def imei(self) -> dict[str, Any]:
        """An IMEI-shaped test value: fixed test TAC
        ``35-000000``, a random serial, and the Luhn
        check digit.  Format-valid, never an assigned
        device."""
        body = "35000000" + "".join(
            str(self.rng.randint(0, 9)) for _ in range(6))
        value = body + _luhn_check_digit(body)
        return {"imei": value, "format_valid": True,
                "assigned": False,
                "note": "test TAC — not a real device"}

    def dob(self) -> str:
        """An adult date of birth (21–45 years before the
        fixed reference date) as YYYY-MM-DD."""
        days_back = self.rng.randint(21 * 365, 45 * 365)
        when = REFERENCE_DATE - timedelta(days=days_back)
        return when.isoformat()

    # -- faces (honest availability) -----------------------------

    def face(self) -> dict[str, Any]:
        """The user-face verdict: no face-generation
        backend is wired (census §17.5), so there is no
        generated face to return — the record says so and
        names the fallback (the initials avatar the
        renderers already draw)."""
        return {"available": False,
                "reason": "no face-generation backend is "
                          "wired for demo data",
                "fallback": "initials avatar"}

    # -- creative names (census Family 6 seed list) -------------

    def creative_name(self, family: str) -> str:
        """A creative name from one of the census §8
        families — Devon's own word pools (the census
        records no per-generator controls, so no pool
        here claims to be FakeDetail's exact list).
        The insult pool is deliberately mild."""
        family = str(family or "").strip().lower()
        if family not in CREATIVE_FAMILIES:
            raise DemoDataError(
                "creative family " + repr(family)
                + " is not one of "
                + ", ".join(CREATIVE_FAMILIES))
        pick = self.rng.choice
        if family == "nickname":
            return pick(("Sunny", "Chief", "Big T", "Smally",
                         "Doc", "Boss Lady", "Captain", "Queen B",
                         "Smallie", "Odogwu")) + " " \
                + pick(("Vibes", "Moves", "Energy", "Waves",
                        "Sauce", "Magic"))
        if family == "aesthetic":
            return pick(("moonlit", "velvet", "golden", "hazy",
                         "silent", "cosmic", "ivory", "midnight")) \
                + " " + pick(("bloom", "aura", "dusk", "echo",
                              "muse", "halo"))
        if family == "band":
            return "The " + pick(("Velvet", "Electric",
                                  "Midnight", "Golden", "Neon",
                                  "Silent")) + " " \
                + pick(("Waves", "Echoes", "Horizon", "Foxes",
                        "Comets", "Rivers"))
        if family == "alien":
            return pick(("Zor", "Xyl", "Vex", "Qu", "Nyx",
                         "Blip")) + pick(("ar", "on", "ix",
                                          "ul", "ara", "eep"))
        if family == "dj":
            return "DJ " + pick(("Neon", "Velvet", "Turbo",
                                 "Cosmic", "Bass", "Solar")) \
                + " " + pick(("Wave", "Fox", "Knight", "Rider",
                              "Blaze", "Echo"))
        if family == "god":
            return pick(("Zeus", "Thor", "Anansi", "Shango",
                         "Odin", "Ra", "Amun", "Freya")) + " " \
                + pick(("of Thunder", "of Stories",
                        "of the Crossroads", "of Dawn",
                        "of the Deep", "of Embers"))
        if family == "horse":
            return pick(("Midnight", "Golden", "Silver",
                         "Copper", "Shadow", "Storm")) + " " \
                + pick(("Gallop", "Mane", "Hoof", "Star",
                        "Runner", "Spirit"))
        if family == "orc":
            return pick(("Grom", "Mok", "Zug", "Thrak",
                         "Borg", "Urz")) + pick(("ash", "gore",
                                                 "mok", "thak",
                                                 "zug", "nak"))
        if family == "planet":
            return pick(("Kepler", "Nova", "Zeta", "Lyra",
                         "Orion", "Vega")) + "-" \
                + str(self.rng.randint(100, 999)) \
                + pick(("b", "c", "d", "e"))
        if family == "dwarf":
            return pick(("Balin", "Dwalin", "Gimli", "Thorin",
                         "Borin", "Durin")) + " " \
                + pick(("Stonefist", "Ironhelm", "Goldbeard",
                        "Anvilborn", "Deepdelver"))
        if family == "warrior-cat":
            return pick(("Fire", "Shadow", "Storm", "Mist",
                         "Ember", "Frost")) + pick(("paw",
                                                    "tail",
                                                    "whisker",
                                                    "claw",
                                                    "heart",
                                                    "stripe"))
        if family == "personality":
            return pick(("The Planner", "The Dreamer",
                         "The Builder", "The Wanderer",
                         "The Listener", "The Spark")) + " — " \
                + pick(("steady hands, big ideas",
                        "always three projects deep",
                        "quiet until the idea lands",
                        "collects half-finished maps",
                        "first to arrive, last to leave",
                        "turns small talk into plans"))
        if family == "insult":
            return pick(("Captain Grump", "Sir Snores-a-Lot",
                         "Professor Slowpoke",
                         "Doctor Drama",
                         "Chief Overthinker",
                         "Madam Mumbles"))
        if family == "bible-verse":
            return pick((
                "Psalm 23:1 — The LORD is my shepherd; "
                "I shall not want.",
                "John 3:16 — For God so loved the world.",
                "Proverbs 3:5 — Trust in the LORD with "
                "all thine heart.",
                "Philippians 4:13 — I can do all things "
                "through Christ which strengtheneth me.",
                "Psalm 27:1 — The LORD is my light and "
                "my salvation.",
                "Joshua 1:9 — Be strong and of a good "
                "courage."))
        # wow
        return pick(("Thrall", "Jaina", "Sylvanas",
                     "Arthas", "Malfurion", "Tyrande")) \
            + " " + pick(("Stormrage", "Proudmoore",
                          "Windrunner", "Menethil",
                          "Whisperwind", "Hellscream"))

    # -- bundles ---------------------------------------------------

    def person(self, *, country: str = "ng") -> dict[str, Any]:
        """One synthetic person bundle — the Fake
        Identity Generator's shape as a plain dict (never
        a trial Persona, never stored)."""
        full = self.name()
        first = full.split(" ", 1)[0]
        handle = self.handle(first)
        mail = self.email()
        phone = self.phone(country)
        place = self.address(country)
        return {"synthetic": True,
                "kind": "demo-test-data",
                "first_name": first,
                "name": full,
                "handle": handle,
                "username": self.username(),
                "bio": self.bio(),
                "dob": self.dob(),
                "email": mail["email"],
                "email_provider": mail["provider"],
                "receives_mail": False,
                "phone": phone["number"],
                "phone_country": phone["country"],
                "address": place["line"],
                "password": self.password(),
                "company": self.company(country),
                "imei": self.imei()["imei"],
                "face": self.face()}

    def identity(self, *, country: str = "ng") -> dict[str, Any]:
        """The full fake-identity bundle — :meth:`person`
        under the census's tool name.  A plain dict, no
        service key, no storage, no trial machinery."""
        return self.person(country=country)

    def identities(self, count: int, *,
                   country: str = "ng") -> list[dict[str, Any]]:
        """``count`` identity bundles (cap 999, census
        §7.3)."""
        n = _check_count(count, MAX_BULK_DATA, "identity")
        return [self.identity(country=country)
                for _ in range(n)]

    # -- screen models ---------------------------------------------

    def participant(self, index: int) -> Participant:
        """A screen Participant from a generated name —
        deterministic colour, no avatar path (the
        renderer draws the initials placeholder)."""
        full = self.name()
        color = _AVATAR_COLORS[index % len(_AVATAR_COLORS)]
        return Participant("p" + str(index), full,
                           color=color)

    def social_profile(self, index: int) -> SocialProfile:
        """A social identity from a generated name."""
        full = self.name()
        first = full.split(" ", 1)[0]
        color = _AVATAR_COLORS[index % len(_AVATAR_COLORS)]
        return SocialProfile(
            "s" + str(index), full,
            handle=self.handle(first),
            verified=self.rng.random() < 0.15,
            color=color)

    def recent_time(self, index: int, total: int) -> str:
        """A descending time label for list row ``index``
        of ``total`` — newest first: now, minutes, hours,
        Yesterday, weekday, date.  A fixed ladder, so
        the same count always orders the same way."""
        if total <= 1:
            return "9:41 AM"
        if index == 0:
            return "now"
        if index <= 3:
            return str(index * 7) + "m"
        if index <= 8:
            return str(index - 3) + "h"
        if index == 9:
            return "Yesterday"
        if index <= 15:
            weekdays = ("Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday",
                        "Sunday")
            return weekdays[(index - 10) % 7]
        day = 10 - (index % 9)
        month = 11 - (index % 3)
        return format(month, "02d") + "/" \
            + format(max(1, day), "02d") + "/2026"

    def preview_text(self) -> str:
        """A short last-message preview line."""
        return self.rng.choice(_PREVIEWS)


def export_records(records: list[dict[str, Any]],
                   fmt: str = "json") -> str:
    """Export generated records as text / JSON / CSV /
    XML (census §29).  Every format round-trips through
    its standard parser — the exit proof.  Values are
    plain str/int/bool only, so nothing exotic can
    sneak into a format."""
    fmt = str(fmt or "json").strip().lower()
    if fmt not in ("text", "json", "csv", "xml"):
        raise DemoDataError(
            "export format " + repr(fmt) + " is not one of "
            "text · json · csv · xml")
    rows = [dict(r) for r in (records or [])]
    if fmt == "json":
        return json.dumps(rows, indent=2, sort_keys=True)
    if fmt == "text":
        lines = []
        for row in rows:
            parts = []
            for key in sorted(row):
                value = row[key]
                if isinstance(value, dict):
                    value = json.dumps(value, sort_keys=True)
                parts.append(key + ": " + str(value))
            lines.append(" · ".join(parts))
        return "\n".join(lines)
    if fmt == "csv":
        buf = io.StringIO()
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(buf, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {}
            for key, value in row.items():
                flat[key] = value if not isinstance(
                    value, dict) else json.dumps(
                        value, sort_keys=True)
            writer.writerow(flat)
        return buf.getvalue()
    root = ET.Element("demo-data")
    for row in rows:
        el = ET.SubElement(root, "record")
        for key in sorted(row):
            child = ET.SubElement(el, key)
            value = row[key]
            child.text = value if isinstance(value, str) \
                else json.dumps(value, sort_keys=True) \
                if isinstance(value, dict) \
                else str(value)
    return ET.tostring(root, encoding="unicode")


# ── bulk screens ─────────────────────────────────────────
#
# One seed + count -> a populated screen the renderers
# draw unchanged.  Rows are newest-first (the time
# ladder descends); the first rows carry the pinned /
# unread / ring states the census names, the rest vary
# by draw.

def bulk_list_screen(platform: str = "whatsapp", *,
                     count: int = 25, seed: int = 0,
                     flavor: str = "chats",
                     style: str = "android",
                     dark: bool = False) -> ListScreen:
    """A chat list (``flavor`` chats / calls / status)
    with ``count`` rows from one seed.  Cap 200 rows
    (the canvas-memory guard)."""
    n = _check_count(count, MAX_BULK_SCREEN_ROWS, "list")
    gen = BulkGenerator(seed)
    platform = str(platform or "whatsapp").strip().lower()
    if platform not in ("whatsapp", "instagram", "telegram",
                        "snapchat"):
        raise DemoDataError(
            "bulk list platform " + repr(platform)
            + " is not one of whatsapp · instagram · "
            "telegram · snapchat")
    flavor = str(flavor or "chats").strip().lower()
    if flavor not in ("chats", "calls", "status"):
        raise DemoDataError(
            "list flavor " + repr(flavor) + " is not one of "
            "chats · calls · status")
    chrome = DeviceChrome(style=style, clock="9:41",
                          battery=84, network="4G")
    people = [Participant("me", "You")]
    rows: list = []
    kinds = [k for k in LIST_PREVIEW_KINDS
             if k not in ("typing", "draft")]
    for idx in range(n):
        person = gen.participant(idx)
        people.append(person)
        kind = gen.rng.choice(kinds)
        if flavor == "calls":
            kind = gen.rng.choice((
                "call_in_voice", "call_out_voice",
                "call_missed_voice", "call_in_video",
                "call_out_video", "call_missed_video"))
        preview = ""
        if kind == "text":
            preview = gen.preview_text()
        elif kind.startswith("call_"):
            preview = "(2)" if gen.rng.random() < 0.2 else ""
        row = ListRow(
            "r" + str(idx), person.name,
            participant_id=person.id,
            preview=preview, preview_kind=kind,
            time=gen.recent_time(idx, n),
            unread=(gen.rng.randint(1, 4)
                    if idx < 3 and gen.rng.random() < 0.7
                    else 0),
            pinned=idx < 2,
            muted=gen.rng.random() < 0.1,
            ring=("unseen" if gen.rng.random() < 0.3 else "")
            if flavor != "calls" else "",
            ticks=("read" if gen.rng.random() < 0.4 else ""),
            row_type=("group" if gen.rng.random() < 0.12
                      else "person"),
            verified=(gen.rng.random() < 0.08),
            online=(gen.rng.random() < 0.2))
        if flavor == "status":
            row.section = ("Recent updates" if idx < n // 2
                           else "Viewed updates")
            row.segments = gen.rng.randint(1, 4)
            row.ring = ("unseen" if idx < n // 2 else "seen")
        rows.append(row)
    title = {"chats": "Chats", "calls": "Calls",
             "status": "Updates"}[flavor]
    if platform == "whatsapp":
        title = "WhatsApp" if flavor == "chats" else title
    elif platform == "instagram":
        title = "peacethefirst"
    elif platform == "telegram":
        title = "Chats"
    stories = []
    if platform == "instagram":
        stories = [StoryItem(people[i + 1].id,
                             note=gen.preview_text())
                   for i in range(min(3, n))]
    tabs = []
    if platform == "telegram":
        tabs = [FolderTab("All Chats",
                          sum(r.unread for r in rows))]
    return ListScreen(
        platform=platform, title=title, rows=rows,
        participants=people, chrome=chrome, flavor=flavor,
        dark=dark, tabs=tabs, stories=stories,
        nav_badge=1 if platform == "whatsapp" else 0)


def bulk_call_history(*, count: int = 25,
                      seed: int = 0) -> CallHistoryScreen:
    """An iPhone Call History with ``count`` entries
    from one seed.  Cap 200 rows."""
    n = _check_count(count, MAX_BULK_SCREEN_ROWS,
                     "call history")
    gen = BulkGenerator(seed)
    people = [Participant("me", "You")]
    entries: list = []
    labels = ("mobile", "home", "work", "FaceTime Audio")
    modes = ("default", "letter", "picture")
    for idx in range(n):
        person = gen.participant(idx)
        people.append(person)
        entries.append(CallHistoryEntry(
            "h" + str(idx), person.name,
            participant_id=person.id,
            type=gen.rng.choice(HISTORY_TYPES),
            avatar_mode=gen.rng.choice(modes),
            sim=("SIM 1" if gen.rng.random() < 0.15
                 else "SIM 2" if gen.rng.random() < 0.1
                 else ""),
            label=gen.rng.choice(labels),
            count=gen.rng.randint(1, 3),
            when=gen.recent_time(idx, n)))
    return CallHistoryScreen(
        entries=entries, participants=people,
        voicemail_count=gen.rng.randint(0, 3),
        chrome=DeviceChrome(style="iphone", clock="9:41",
                            battery=84, network="4G"))


def bulk_chat_screen(platform: str = "whatsapp", *,
                     count: int = 30, seed: int = 0,
                     style: str = "android", dark: bool = False,
                     group: bool = False) -> Screen:
    """A chat with ``count`` messages from one seed —
    the conversation builds over a steady clock, oldest
    first, outgoing messages ticked.  Cap 200 messages."""
    n = _check_count(count, MAX_BULK_SCREEN_ROWS, "chat")
    gen = BulkGenerator(seed)
    platform = str(platform or "whatsapp").strip().lower()
    if platform not in ("whatsapp", "instagram", "telegram",
                        "messenger", "snapchat", "imessage",
                        "sms"):
        raise DemoDataError(
            "bulk chat platform " + repr(platform)
            + " is not one of whatsapp · instagram · "
            "telegram · messenger · snapchat · imessage · sms")
    others = [gen.participant(i) for i in range(3)]
    people = [Participant("me", "You")] + others
    title = others[0].name
    subtitle = "online" if platform in (
        "whatsapp", "telegram") else "Active now"
    if group:
        title = "Demo Crew"
        subtitle = ", ".join(p.name for p in others) + ", you"
    messages: list = [Message(
        "e-day", "system", kind="divider", text="Today")]
    minutes = 0
    for idx in range(n):
        minutes += gen.rng.randint(1, 4)
        total = 9 * 60 + minutes
        hh, mm = divmod(total, 60)
        suffix = "AM" if hh < 12 else "PM"
        clock = str(hh if hh <= 12 else hh - 12) + ":" \
            + format(mm, "02d") + " " + suffix
        sender = "me" if gen.rng.random() < 0.45 \
            else gen.rng.choice(others).id
        messages.append(Message(
            "e" + str(idx), sender,
            text=gen.preview_text(), time=clock,
            ticks=("read" if sender == "me"
                   and gen.rng.random() < 0.7 else ""),
            reactions=(["🔥"] if gen.rng.random() < 0.08
                       else [])))
    return Screen(
        platform=platform, group=group, dark=dark,
        title=title, subtitle=subtitle,
        participants=people, messages=messages,
        chrome=DeviceChrome(style=style, clock="9:41",
                            battery=84, network="4G"))


def bulk_status_views(*, count: int = 25, seed: int = 0,
                      style: str = "android"
                      ) -> StatusViewsScreen:
    """A Status Views sheet with ``count`` viewers from
    one seed.  Cap 200 viewers."""
    n = _check_count(count, MAX_BULK_SCREEN_ROWS,
                     "status views")
    gen = BulkGenerator(seed)
    people = [Participant("me", "You")]
    viewers: list = []
    for idx in range(n):
        person = gen.participant(idx)
        people.append(person)
        viewers.append(StatusViewer(
            person.name, person.id,
            gen.recent_time(idx, n)))
    return StatusViewsScreen(
        owner=Participant("me", "You"),
        when="Today, 9:12 AM", viewers=viewers,
        participants=people,
        chrome=DeviceChrome(style=style, clock="9:41",
                            battery=84, network="4G"))
