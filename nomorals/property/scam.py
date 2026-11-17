"""Nigerian rental scam-detection engine (build-map #93).

Paste a Jiji / PropertyPro / Nigeria Property Centre link (or the
listing text) → a verdict in seconds.  Seven automated checks:

1. Price anomaly vs area norms (>25% below the area average → flag).
2. Duplicate detection: photo hashing + address normalization across
   listings the user has checked.
3. Reverse-image hook (injectable; Seer or an image-search backend).
4. Fee-language scan: "inspection / registration / caution /
   documentation fee" — LASRERA declared these illegal.
5. Address test: landmark-only, no street/number → flag.
6. Payment-channel check: personal account / irreversible route
   (crypto, gift cards, Western Union) + urgency language → hard stop.
7. Viewing check (Fredy pattern): landlord-abroad + keys-by-post, or
   money demanded before any viewing → flags.

Signal model follows orangecoding/fredy's scam-detection doc:
signals carry weights 1–3 and corroborate — a single weak signal
never fires a verdict alone; weight-3 signals (money before viewing,
irreversible payment routes, illegal fees) fire on their own.
Deliberately NOT signals: a refundable deposit, agency fees, and a
phone/email in the ad — all normal in honest Lagos listings
(Fredy's false-positive discipline).

Flag language is factual, never moralizing.  Every function never
raises.  ``check_listing`` is pure apart from the optional injectable
``fetcher`` / ``photo_analyzer`` seams, so tests run fully offline.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from .cross_listing import (
    Listing,
    PropertyPassportStore,
    cross_listing_score,
    group_listings,
)

__all__ = [
    "ScamReport", "ScamFlag", "AREA_NORMS", "ILLEGAL_FEE_TERMS",
    "ScamStore", "check_listing", "control_scamcheck",
]

_log = logging.getLogger(__name__)


#: Price tolerance for the cross-listing seam (audit R3-68): listing
#: prices for the same property across sites differ by small rounding
#: (₦2.5m vs ₦2.45m), so exact-kobo matching would almost never group.
#: 2% of the current price, in kobo. Address + photo rules from
#: ``same_property`` still apply, so two different properties never
#: merge on price alone.
_CROSS_LISTING_TOLERANCE_PCT = 0.02

_URL_RE = re.compile(r"https?://\S+")


def _source_from_url(url: str) -> str:
    """Domain of a listing URL (``jiji.ng``), "" when there is none.
    Pasted-text listings carry no source — never invent one."""
    try:
        host = urllib.parse.urlparse(url or "").netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:  # noqa: BLE001
        return ""


def _listing_source(raw: str, url: str) -> str:
    """Source for a listing: the checked URL's domain, else the first
    URL embedded in the listing text, else "" (unknown)."""
    source = _source_from_url(url)
    if source:
        return source
    m = _URL_RE.search(raw or "")
    return _source_from_url(m.group(0)) if m else ""


def _default_db() -> str:
    home = os.path.expanduser("~")
    return os.path.join(home, ".nomorals", "property", "scam.db")


# ── area norms ──────────────────────────────────────────────────────
# Seed annual rents (₦) for common Lagos rental types.  Conservative
# midpoints; refined from the user's own checks over time.  Flagging
# only fires at >25% BELOW these, so stale-high norms fail safe
# (no flag) rather than false-alarming.

AREA_NORMS: dict[str, dict[str, int]] = {
    "lekki": {"1br": 2_800_000, "2br": 4_500_000, "3br": 7_000_000,
              "room": 900_000, "studio": 1_800_000},
    "ajah": {"1br": 1_600_000, "2br": 2_500_000, "3br": 3_800_000,
             "room": 600_000, "studio": 1_100_000},
    "yaba": {"1br": 1_400_000, "2br": 2_200_000, "3br": 3_200_000,
             "room": 500_000, "studio": 950_000},
    "ikeja": {"1br": 1_800_000, "2br": 3_000_000, "3br": 4_500_000,
              "room": 700_000, "studio": 1_200_000},
    "surulere": {"1br": 1_300_000, "2br": 2_100_000, "3br": 3_000_000,
                 "room": 500_000, "studio": 900_000},
    "ikorodu": {"1br": 700_000, "2br": 1_100_000, "3br": 1_600_000,
                "room": 300_000, "studio": 500_000},
    "maryland": {"1br": 1_500_000, "2br": 2_400_000, "3br": 3_500_000,
                 "room": 550_000, "studio": 1_000_000},
    "ogba": {"1br": 900_000, "2br": 1_400_000, "3br": 2_000_000,
             "room": 350_000, "studio": 650_000},
    "victoria island": {"1br": 3_500_000, "2br": 6_000_000,
                        "3br": 9_000_000, "room": 1_200_000,
                        "studio": 2_400_000},
    "ikoyi": {"1br": 4_500_000, "2br": 8_000_000, "3br": 12_000_000,
              "room": 1_500_000, "studio": 3_000_000},
}

_AREA_ALIASES = {
    "vi": "victoria island", "lekki phase 1": "lekki",
    "lekki phase 2": "lekki", "ajah": "ajah",
}

ILLEGAL_FEE_TERMS = [
    "inspection fee", "registration fee", "caution fee",
    "documentation fee", "form fee", "allocation fee",
    "verification fee", "booking fee",
]

# Severity → score deduction.
_DEDUCT = {"info": 5, "warn": 15, "danger": 30, "hard_stop": 100}

# Signal weights (Fredy pattern): 1 = weak, 3 = fires on its own.
# Weak signals corroborate — one weight-2 signal alone never sinks a
# listing.  The weight is informational; the verdict still comes from
# the severity-deducted score so existing behavior is unchanged.
_WEIGHT = {
    "price_anomaly": 2,
    "duplicate": 2,
    "reverse_image": 2,
    "illegal_fee": 3,
    "vague_address": 2,
    "payment_channel": 3,
    "no_viewing": 3,
}


# ── data types ──────────────────────────────────────────────────────

@dataclass
class ScamFlag:
    code: str          # price_anomaly | duplicate | reverse_image |
                       # illegal_fee | vague_address | payment_channel |
                       # no_viewing
    severity: str      # info | warn | danger | hard_stop
    message: str       # factual, never moralizing
    evidence: str = ""
    weight: int = 0    # Fredy-style signal weight (1–3); 0 = unrated

    def __post_init__(self) -> None:
        if not self.weight:
            self.weight = _WEIGHT.get(self.code, 1)


@dataclass
class ScamReport:
    url: str = ""
    score: int = 100   # 0–100, higher = safer
    verdict: str = ""  # looks clean | check carefully | likely scam
    flags: list[ScamFlag] = field(default_factory=list)
    area: str = ""
    price_kobo: int = 0
    checked_at: float = 0.0
    #: Cross-listing signal (audit R3-68): the ``cross_listing_score``
    #: dict for the group this listing belongs to — only set when a
    #: prior sighting of the same property exists. Informational only;
    #: it never changes the score or the verdict.
    cross_listing: dict = field(default_factory=dict)

    def format(self) -> str:
        icon = "✅" if (self.score >= 70 and not any(
            f.severity in ("danger", "hard_stop") for f in self.flags)) \
            else ("⚠️" if self.score >= 40 else "🚨")
        lines = [f"{icon} Scam check — score {self.score}/100: {self.verdict}"]
        if self.area:
            lines.append(f"📍 area: {self.area.title()}")
        if self.price_kobo:
            lines.append(f"💰 listed price: ₦{self.price_kobo // 100:,}/yr")
        for f in self.flags:
            sev = {"info": "ℹ️", "warn": "⚠️",
                   "danger": "🔶", "hard_stop": "🛑"}.get(f.severity, "•")
            lines.append(f"{sev} {f.message}")
            if f.evidence:
                lines.append(f"   ↳ {f.evidence[:120]}")
        if not self.flags:
            lines.append("No red flags found in the seven automated checks.")
        if self.cross_listing:
            cl = self.cross_listing
            names = cl.get("source_names") or []
            n = cl.get("sources", 0)
            listings = cl.get("listings", 0)
            if n > 1:
                lines.append(
                    f"🔁 Cross-listed: same property on {n} sources "
                    f"({', '.join(names)}) — one property, not "
                    f"{listings} properties.")
            elif n == 1 and listings > 1:
                lines.append(
                    f"🔁 Seen {listings} time(s) on {names[0]} — "
                    "same listing seen before.")
            elif listings > 1:
                lines.append(
                    f"🔁 Seen {listings} time(s) before "
                    "(source unknown).")
        lines.append("")
        lines.append("Before paying anything:")
        lines.append("  1. Always inspect in person — never pay for a "
                     "flat you haven't entered.")
        lines.append("  2. Verify the address on a map; landmark-only "
                     "addresses can't be checked.")
        lines.append("  3. Reverse-image-search the photos — recycled photos "
                     "are a top scam tell.")
        lines.append("  4. Keep chat and payment on traceable channels; "
                     "irreversible routes (crypto, gift cards, wire) can't "
                     "be recovered.")
        lines.append("  5. Suspected fraud in Lagos can be reported to "
                     "LASRERA, the state regulator.")
        return "\n".join(lines)


# ── store (duplicate detection) ─────────────────────────────────────

#: The store's own DDL — the native path executes it and the seam
#: mapping is derived from it, so the two can never drift. (The
#: photo/address indexes below stay native-path DDL; the seam
#: reads are finds.)
_DDL = """CREATE TABLE IF NOT EXISTS seen_listings (
       id TEXT PRIMARY KEY, photo_hash TEXT, address_norm TEXT,
       url TEXT, price_kobo INTEGER, created_at REAL)"""


class ScamStore:
    """Seen listings (photo hashes + normalized addresses).

    Native path: a SQLite file at ``db_path``. When a document
    store is resolved (Mongo program P5 — an explicit ``store`` or
    the boot ambient), the same API runs over the seam in the
    ``property-scam__seen_listings`` collection instead
    (registered in storage.sidecars), and the file's existing
    rows are imported once, through the same mapping, at
    construction.
    """

    def __init__(self, db_path: str = "", *, store: Any = None) -> None:
        self._db: sqlite3.Connection | None = None
        self._port: Any = None
        try:
            path = db_path or _default_db()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            from ..storage.sidecars import sidecar_port

            port = sidecar_port("property-scam", _DDL,
                                ("seen_listings",), path, store=store)
            if port is not None:
                table = port.tables["seen_listings"]

                def write_seen(rows: list[dict[str, Any]]) -> int:
                    for row in rows:
                        table.put(row)
                    return len(rows)

                port.ensure_imported({"seen_listings": write_seen})
                self._port = port
                return
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(_DDL)
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS sl_photo ON seen_listings(photo_hash)")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS sl_addr ON seen_listings(address_norm)")
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB is an empty store
            _log.warning("scam: db unavailable, running empty", exc_info=True)
            self._db = None
            self._port = None

    def remember(self, photo_hash: str, address_norm: str,
                 url: str, price_kobo: int) -> None:
        try:
            listing_id = hashlib.md5(
                f"{photo_hash}|{address_norm}|{url}".encode()
            ).hexdigest()[:16]
            created = time.time()
            if self._port is not None:
                self._port.tables["seen_listings"].insert_ignore({
                    "id": listing_id, "photo_hash": photo_hash,
                    "address_norm": address_norm, "url": url,
                    "price_kobo": price_kobo, "created_at": created})
                return
            if self._db is None:
                return
            self._db.execute(
                "INSERT OR IGNORE INTO seen_listings VALUES (?, ?, ?, ?, ?, ?)",
                (listing_id, photo_hash, address_norm, url, price_kobo,
                 created))
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("scam remember failed", exc_info=True)

    def find_duplicate(self, photo_hash: str,
                       address_norm: str) -> list[dict]:
        """Prior listings with the same photos or the same address.

        The address branch used to exclude rows whose photo hash
        equalled the queried one — which, for the common photo-less
        listing (both hashes ""), excluded every address match and
        made same-address duplicates invisible. The id-based dedup
        below already prevents double-counting a row that matches
        both ways, so the exclusion is dropped (audit R3-68: the
        cross-listing seam needs those sightings).
        """
        try:
            if self._port is not None:
                table = self._port.tables["seen_listings"]
                rows = []
                if photo_hash:
                    rows += table.find({"photo_hash": photo_hash})
                if address_norm:
                    rows += table.find({"address_norm": address_norm})
                seen, out = set(), []
                for r in rows:
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        out.append(r)
                return out
            if self._db is None:
                return []
            rows = []
            if photo_hash:
                rows += self._db.execute(
                    "SELECT * FROM seen_listings WHERE photo_hash = ?",
                    (photo_hash,)).fetchall()
            if address_norm:
                rows += self._db.execute(
                    "SELECT * FROM seen_listings WHERE address_norm = ?",
                    (address_norm,)).fetchall()
            seen, out = set(), []
            for r in rows:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    out.append(dict(r))
            return out
        except Exception:  # noqa: BLE001
            _log.debug("scam duplicate lookup failed", exc_info=True)
            return []


# ── parsing helpers ─────────────────────────────────────────────────

def _extract_price_kobo(text: str) -> int:
    """₦2,500,000 / N2.5m / 2,500,000 naira → kobo per year."""
    t = (text or "").lower()
    m = re.search(r"[₦n]\s*([\d,.]+)\s*(m|million|k|thousand)?", t)
    if not m:
        m = re.search(r"([\d,]{5,})\s*(naira|ngn)?", t)
    if not m:
        return 0
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return 0
    mult = {"m": 1_000_000, "million": 1_000_000,
            "k": 1_000, "thousand": 1_000}.get((m.group(2) or "").lower(), 1)
    return int(num * mult * 100)


def _extract_area(text: str) -> str:
    t = (text or "").lower()
    for alias, canon in _AREA_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", t):
            return canon
    for area in AREA_NORMS:
        if re.search(r"\b" + re.escape(area) + r"\b", t):
            return area
    return ""


def _extract_bedrooms(text: str) -> str:
    t = (text or "").lower()
    m = re.search(r"(\d)\s*(bedroom|bed|br)\b", t)
    if m:
        n = m.group(1)
        return f"{n}br" if n in ("1", "2", "3") else "3br"
    if "studio" in t:
        return "studio"
    if re.search(r"\bself[\s-]?contain\b|\bsingle room\b", t):
        return "room"
    return ""


def _normalize_address(text: str) -> str:
    """Strip noise so the same address matches across listings."""
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    stop = {"no", "number", "street", "st", "road", "rd", "avenue", "ave",
            "close", "estate", "phase", "plot", "lagos", "nigeria"}
    return " ".join(w for w in t.split() if w not in stop and len(w) > 2)


def _photo_hash(photo_bytes: bytes) -> str:
    return hashlib.sha256(photo_bytes or b"").hexdigest()[:32]


def _is_url(s: str) -> bool:
    return bool(re.match(r"https?://", (s or "").strip(), re.I))


# ── the checks ────────────────────────────────────────────────────

def _check_price(text: str, area: str, bedrooms: str,
                 price_kobo: int) -> list[ScamFlag]:
    """Fredy note: a cheap flat is not a scam. Price alone only ever
    produces info/warn — the danger verdict needs corroboration."""
    flags: list[ScamFlag] = []
    if not area or not price_kobo:
        return flags
    norm = AREA_NORMS.get(area, {}).get(bedrooms or "2br")
    if not norm:
        return flags
    norm_kobo = norm * 100
    if price_kobo < norm_kobo * 0.75:
        pct = round((1 - price_kobo / norm_kobo) * 100)
        flags.append(ScamFlag(
            "price_anomaly", "danger",
            f"Price is {pct}% below the {area.title()} norm for this size — "
            "classic too-good-to-be-true bait.",
            f"listed ₦{price_kobo // 100:,} vs area norm ₦{norm:,}"))
    elif price_kobo < norm_kobo * 0.9:
        flags.append(ScamFlag(
            "price_anomaly", "info",
            "Price is below the area norm — verify it's genuine, not bait.",
            f"listed ₦{price_kobo // 100:,} vs area norm ₦{norm:,}"))
    return flags


def _check_duplicates(store: ScamStore | None, photo_hash: str,
                       address_norm: str) -> list[ScamFlag]:
    if store is None:
        return []
    dups = store.find_duplicate(photo_hash, address_norm)
    if not dups:
        return []
    return [ScamFlag(
        "duplicate", "warn",
        f"Same photos/address seen in {len(dups)} earlier listing(s) — "
        "scammers recycle one property across many ads.",
        f"first seen: {dups[0].get('url', 'unknown')[:80]}")]


def _check_reverse_image(photo_analyzer, photo_bytes: bytes) -> list[ScamFlag]:
    """Injectable hook: analyzer(photo_bytes) -> str describing findings."""
    if photo_analyzer is None or not photo_bytes:
        return []
    try:
        result = (photo_analyzer(photo_bytes) or "").lower()
    except Exception:  # noqa: BLE001
        return []
    if any(w in result for w in ("stock photo", "different property",
                                 "mismatch", "stolen", "other listing",
                                 "appears elsewhere")):
        return [ScamFlag(
            "reverse_image", "warn",
            "Listing photos appear to come from elsewhere — "
            "ask for a live video of the property.",
            result[:120])]
    return []


def _check_fees(text: str) -> list[ScamFlag]:
    t = (text or "").lower()
    found = [term for term in ILLEGAL_FEE_TERMS if term in t]
    if not found:
        return []
    return [ScamFlag(
        "illegal_fee", "danger",
        f"{found[0].title()} requested — LASRERA declared this illegal. "
        "No legitimate Lagos agent charges it.",
        "Lagos State Real Estate Regulatory Authority directive")]


def _check_address(text: str) -> list[ScamFlag]:
    t = (text or "").lower()
    has_number = bool(re.search(r"\bno\.?\s*\d+|\bplot\s*\d+|\d+\s+[a-z]+\s+(street|road|close|avenue)", t))
    has_street = bool(re.search(r"(street|road|close|avenue|estate)\b", t))
    landmark_only = (not has_number and not has_street
                     and bool(re.search(r"(near|opposite|behind|beside|around)\b", t)))
    if landmark_only:
        return [ScamFlag(
            "vague_address", "warn",
            "Address is landmark-only with no street or house number — "
            "you can't verify the property exists before visiting.",
            "insist on a full address before any inspection")]
    return []


_IRREVERSIBLE = re.compile(
    r"western\s*union|money\s*gram|gift\s*card|crypto|usdt|btc\b|bitcoin|"
    r"binance|usdc\b", re.I)

_URGENCY = ["urgent", "hurry", "asap", "first come", "limited",
            "going fast", "don't miss", "last chance", "today only"]


def _check_payment(text: str) -> list[ScamFlag]:
    t = (text or "").lower()
    personal = bool(re.search(
        r"(personal|private|my own)\s+account|pay\s+(me|directly)|"
        r"opay|palmpay|moniepoint.*(personal|my)|"
        r"account\s+(name|number).*(?!company|agency|ltd|limited)", t))
    irreversible = bool(_IRREVERSIBLE.search(t))
    urgent = any(w in t for w in _URGENCY)
    route = personal or irreversible
    if route and urgent:
        why = ("an irreversible route (crypto/gift card/wire — works like "
               "cash, can't be recovered)"
               if irreversible else "a personal account")
        return [ScamFlag(
            "payment_channel", "hard_stop",
            f"Payment to {why} under urgency pressure — "
            "do not send money. Inspect first, pay to a verifiable account.",
            "irreversible-payment + urgency is the #1 rental-scam pattern")]
    if irreversible:
        return [ScamFlag(
            "payment_channel", "danger",
            "Payment requested via an irreversible route (crypto, gift "
            "card, or wire transfer) — these work like cash and can't be "
            "recovered once sent. Legitimate agents accept traceable "
            "payment.",
            "FTC rental-scam guidance: wire/gift-card/crypto demands")]
    if personal:
        return [ScamFlag(
            "payment_channel", "warn",
            "Payment requested to a personal account — "
            "verify the recipient's identity before paying anything.",
            "")]
    return []


_NO_VIEWING_HARD = re.compile(
    r"pay\s+(?:me|us)?\s*(?:before|prior to)\s+(?:any\s+)?(?:viewing|inspection)|"
    r"money\s+before\s+(?:you\s+)?(?:view|inspect)|"
    r"payment\s+before\s+(?:viewing|inspection)|"
    r"keys?\s+(?:will\s+be\s+)?(?:sent\s+)?(?:by|via|through)\s+"
    r"(?:post|courier|dhl|fedex)|"
    r"send(?:ing)?\s+the\s+keys?", re.I)
_NO_VIEWING_SOFT = re.compile(
    r"\bi['’]?m\s+abroad\b|i\s+am\s+abroad|out\s+of\s+the\s+country|"
    r"not\s+in\s+nigeria|can['’]?t\s+view|cannot\s+view|"
    r"no\s+viewing|viewing\s+(?:is\s+)?(?:not|im)possible|"
    r"no\s+inspection\s+possible", re.I)


def _check_no_viewing(text: str) -> list[ScamFlag]:
    """Fredy pattern: the script is price-far-below-market + landlord
    can't do a viewing (abroad) + keys by post + money before viewing."""
    t = text or ""
    if _NO_VIEWING_HARD.search(t):
        return [ScamFlag(
            "no_viewing", "danger",
            "Money demanded before any viewing, or keys promised by post/"
            "courier — you would be paying for a flat nobody has shown "
            "you. Insist on an in-person viewing first.",
            "advance-payment-before-viewing is a classic scam script")]
    if _NO_VIEWING_SOFT.search(t):
        return [ScamFlag(
            "no_viewing", "warn",
            "The landlord says no viewing is possible (abroad / can't "
            "show the flat) — verify the property exists independently "
            "before any commitment.",
            "")]
    return []


# ── main entry ──────────────────────────────────────────────────────

def check_listing(url_or_text: str, *,
                  store: ScamStore | None = None,
                  fetcher=None,
                  photo_analyzer=None,
                  photo_bytes: bytes = b"",
                  passport_store=None) -> ScamReport:
    """Run the six checks. Never raises. Fully offline-capable.

    ``passport_store`` — audit R3-68: an optional
    :class:`PropertyPassportStore` that records each sighting, so a
    property's passport accumulates across re-listings. ``None``
    (the default) records nothing and keeps this function pure apart
    from its existing injectable seams.
    """
    report = ScamReport(checked_at=time.time())
    try:
        raw = (url_or_text or "").strip()
        if not raw:
            report.verdict = "nothing to check"
            return report

        text = raw
        if _is_url(raw):
            report.url = raw
            if fetcher is not None:
                try:
                    text = fetcher(raw) or raw
                except Exception:  # noqa: BLE001
                    _log.debug("scam fetcher failed", exc_info=True)

        area = _extract_area(text)
        bedrooms = _extract_bedrooms(text)
        price_kobo = _extract_price_kobo(text)
        # Strip URLs before extracting the address: a source link pasted
        # after "Address: ..." is not part of the address, and it made
        # the same property on two sites normalize to two different
        # addresses (so neither the duplicate check nor the
        # cross-listing grouping could see them as one property).
        addr_text = _URL_RE.sub(" ", text)
        addr_m = re.search(r"(?:address|location)[:\s]+(.{0,80})",
                           addr_text, re.I)
        raw_address = (addr_m.group(1).strip().rstrip(".").strip()
                       if addr_m else "")
        address_norm = _normalize_address(
            raw_address if raw_address else text[:80])
        phash = _photo_hash(photo_bytes) if photo_bytes else ""
        source = _listing_source(raw, report.url)

        report.area = area
        report.price_kobo = price_kobo

        flags: list[ScamFlag] = []
        flags += _check_price(text, area, bedrooms, price_kobo)
        flags += _check_duplicates(store, phash, address_norm)
        flags += _check_reverse_image(photo_analyzer, photo_bytes)
        flags += _check_fees(text)
        flags += _check_address(text)
        flags += _check_payment(text)
        flags += _check_no_viewing(text)

        score = 100
        for f in flags:
            score -= _DEDUCT.get(f.severity, 10)
        report.score = max(0, min(100, score))
        report.flags = flags
        has_serious = any(f.severity in ("danger", "hard_stop") for f in flags)
        report.verdict = (
            "looks clean" if report.score >= 70 and not has_serious else
            "check carefully" if report.score >= 40 else
            "likely scam — do not pay")

        # ── cross-listing signal (audit R3-68) ──────────────────
        # The current listing + its prior sightings in the ScamStore
        # form the group; group_listings/cross_listing_score say
        # whether the same property is marketed on several sources.
        # Informational only — it never changes the score or verdict.
        if store is not None:
            try:
                dups = store.find_duplicate(phash, address_norm)
            except Exception:  # noqa: BLE001
                dups = []
            if dups:
                items = [Listing(source=source, address=address_norm,
                                 price=float(price_kobo),
                                 photo_hashes=[phash] if phash else [],
                                 url=report.url or "")]
                for row in dups:
                    row = dict(row)
                    items.append(Listing(
                        source=_source_from_url(
                            str(row.get("url") or "")),
                        address=str(row.get("address_norm") or ""),
                        price=float(row.get("price_kobo") or 0),
                        photo_hashes=(
                            [str(row.get("photo_hash"))]
                            if row.get("photo_hash") else []),
                        url=str(row.get("url") or "")))
                tolerance = (price_kobo * _CROSS_LISTING_TOLERANCE_PCT
                             if price_kobo > 0 else 0.0)
                groups = group_listings(items, price_tolerance=tolerance)
                own = next((g for g in groups if items[0] in g), [])
                if len(own) > 1:
                    report.cross_listing = cross_listing_score(own)

        # ── property passport sighting (audit R3-68) ────────────
        # One property, one passport (the store dedupes on normalized
        # address); every sighting appends a "listed" event, so the
        # history accumulates across re-listings instead of resetting.
        if passport_store is not None and (raw_address or address_norm):
            try:
                passport = passport_store.create(
                    raw_address or address_norm)
                if passport is not None:
                    detail = (f"{source or 'text'}"
                              + (f" · ₦{price_kobo // 100:,}"
                                 if price_kobo > 0 else ""))
                    passport_store.add_event(
                        passport.passport_id, "listed", detail=detail,
                        source=source)
            except Exception:  # noqa: BLE001
                _log.debug("scam passport sighting failed", exc_info=True)

        if store is not None and (phash or address_norm):
            # Remember the source URL when the listing text carries one
            # (a pasted link), not the literal "text" — the stored URL
            # is what the cross-listing seam derives the source from,
            # and it is what the duplicate flag shows as "first seen".
            url_m = _URL_RE.search(raw)
            store_url = report.url or (url_m.group(0) if url_m else "text")
            store.remember(phash, address_norm, store_url, price_kobo)
        return report
    except Exception:  # noqa: BLE001 — never raises, by contract
        _log.warning("check_listing failed", exc_info=True)
        report.verdict = "check failed — inspect manually"
        return report


# ── chat ────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("/scamcheck <jiji/propertypro link or listing text> [photo:<path>] — "
            "seven automated scam checks, verdict in seconds. "
            "Example: /scamcheck https://jiji.ng/... ")


def control_scamcheck(tail: str, context=None, chat=None, **kwargs) -> str:
    """Chat entry: /scamcheck. Owner + community (not owner-gated)."""
    try:
        tail = (tail or "").strip()
        if not tail:
            return _usage()
        photo_bytes = b""
        photo_analyzer = None
        m = re.search(r"\bphoto:(\S+)", tail)
        if m:
            p = m.group(1)
            tail = (tail[:m.start()] + tail[m.end():]).strip()
            try:
                with open(os.path.expanduser(p), "rb") as fh:
                    photo_bytes = fh.read()
            except OSError:
                return f"couldn't read photo {p}"
            if context is not None:
                photo_analyzer = getattr(context, "photo_analyzer", None)
        store = ScamStore()
        # Audit R3-68: every sighting accumulates on the property's
        # passport (injectable via kwarg for tests; the default store
        # is the durable cross-listing passport file).
        pp_store = kwargs.get("property_passport_store")
        if pp_store is None:
            try:
                pp_store = PropertyPassportStore()
            except Exception:  # noqa: BLE001
                pp_store = None
        report = check_listing(tail, store=store,
                               photo_analyzer=photo_analyzer,
                               photo_bytes=photo_bytes,
                               passport_store=pp_store)
        return report.format() + "\n\n" + _general_signals_section(tail)
    except Exception:  # noqa: BLE001
        _log.warning("scamcheck failed", exc_info=True)
        return "scam check failed — inspect the listing manually."


def _general_signals_section(text: str) -> str:
    """AA-33: general scam-signal scoring (nomorals.scam.signals).

    The seven checks above are property-specific; this scores the
    same text against the general signal descriptors (urgency,
    advance-fee, impersonation, too-good pricing, off-platform
    pressure…). Honest failure: if the scorer itself fails, the
    section says "not checked" — an unscored text is never "safe".
    """
    try:
        from ..scam.signals import assess
        result = assess(text or "", kind="message")
    except Exception as exc:  # noqa: BLE001 — surface, never fake a pass
        _log.warning("general scam signals failed", exc_info=True)
        return (f"🧭 General scam signals: not checked "
                f"(scorer failed: {exc}).")
    verdict = result.get("verdict", "ok")
    flags = result.get("flags") or []
    icon = {"suspicious": "🚨", "review": "⚠️"}.get(verdict, "✅")
    lines = [f"{icon} General scam signals: {verdict} "
             f"(score {result.get('score', 0.0):.2f})"]
    if flags:
        lines.append("   flags: " + ", ".join(flags))
        for reason in (result.get("reasons") or [])[:4]:
            lines.append(f"   ↳ {reason[:160]}")
    else:
        lines.append("   no general scam patterns matched.")
    return "\n".join(lines)


def register(registry) -> None:
    """Tool-registry hook."""
    try:
        registry.register("scamcheck", control_scamcheck,
                          "Nigerian rental scam check — seven automated checks on any listing.")
    except Exception:  # noqa: BLE001
        _log.debug("scamcheck register failed", exc_info=True)
