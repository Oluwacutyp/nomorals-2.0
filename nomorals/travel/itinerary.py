"""TripIt-style email forwarding → auto itinerary (build-map #72).

Forward a booking confirmation (email, PDF, screenshot) → Devon parses
it → master itinerary with times, confirmation codes, calendar sync.

The best zero-effort onboarding in travel: the owner forwards what the
airline/hotel already sent them; Devon does the structuring.

Multi-format ingestion (Petrel Voyage pattern):
- email text (Gmail connector or pasted forward)
- PDF attachments (pypdf when available)
- screenshots/photos (Seer vision)

Document vault (MyTripx pattern): every trip's raw docs live in one
place, per trip: ``~/.nomorals/travel/vault/<trip_id>/``.

Chat:
    /trip                       list trips
    /trip <id>                  itinerary summary
    /trip add <text>            parse a pasted confirmation
    /trip calendar <id>         sync to Google Calendar (confirmation-gated)
    /trip docs <id>             list vault docs
Forwarded confirmation text is detected by the NL hook.

After adding a flight the reply offers #71 price tracking.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Flight", "HotelStay", "CarRental", "Trip",
    "parse_confirmation", "ItineraryBuilder",
    "confirmation_hook", "TRIP_VAULT_DIR",
]

TRIP_VAULT_DIR = os.path.expanduser("~/.nomorals/travel/vault")
TRIP_DB = os.path.expanduser("~/.nomorals/travel/trips.db")


# ── data model ──────────────────────────────────────────────────────────────

@dataclass
class Flight:
    airline: str = ""
    flight_number: str = ""      # "BA075"
    origin: str = ""             # IATA
    destination: str = ""        # IATA
    departs: str = ""            # ISO-ish "2026-12-01T22:45"
    arrives: str = ""            # ISO-ish
    pnr: str = ""                # 6-char booking reference
    seat: str = ""

    def one_line(self) -> str:
        route = f"{self.origin}→{self.destination}" if self.origin else ""
        fn = self.flight_number or "flight"
        t = f", departs {self.departs[11:16]}" if len(self.departs) > 12 else ""
        p = f", PNR {self.pnr}" if self.pnr else ""
        return f"{route}, {fn}{t}{p}".strip(", ")


@dataclass
class HotelStay:
    name: str = ""
    check_in: str = ""           # "2026-12-01"
    check_out: str = ""          # "2026-12-05"
    confirmation: str = ""
    address: str = ""

    def one_line(self) -> str:
        dates = ""
        if self.check_in:
            dates = f", {self.check_in}" + (f" → {self.check_out}"
                                            if self.check_out else "")
        c = f", conf {self.confirmation}" if self.confirmation else ""
        return f"{self.name}{dates}{c}".strip(", ")


@dataclass
class CarRental:
    company: str = ""
    pickup: str = ""
    dropoff: str = ""
    confirmation: str = ""

    def one_line(self) -> str:
        c = f", conf {self.confirmation}" if self.confirmation else ""
        return f"{self.company or 'car rental'}{c}".strip(", ")


@dataclass
class Trip:
    id: str
    name: str = ""
    flights: list[Flight] = field(default_factory=list)
    hotels: list[HotelStay] = field(default_factory=list)
    cars: list[CarRental] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)   # vault filenames
    created_at: float = 0.0

    def is_empty(self) -> bool:
        return not (self.flights or self.hotels or self.cars)


# ── parsing ─────────────────────────────────────────────────────────────────

_PNR_RE = re.compile(
    r"(?:pnr|booking\s+(?:reference|ref)|record\s+locator|confirmation\s+"
    r"(?:code|number|no)|ref(?:erence)?\s*(?:no|#)?)\s*[:#]?\s*([A-Z0-9]{6})\b",
    re.IGNORECASE,
)

_FLIGHT_RE = re.compile(
    r"\b([A-Z]{2})\s?(\d{2,4}[A-Z]?)\b"
)

_ROUTE_RE = re.compile(
    r"\b([A-Z]{3})\s*(?:\([^)]*\))?\s*(?:→|->|–|—|to)\s*"
    r"(?:\([^)]*\))?\s*([A-Z]{3})\b", re.IGNORECASE
)

_TIME_CTX_RE = re.compile(
    r"(?:depart(?:s|ure)?|dep\.?|leaves?|take-?off)\s*:?\s*"
    r"(\d{1,2}:\d{2})",
    re.IGNORECASE,
)
_ARR_CTX_RE = re.compile(
    r"(?:arriv(?:es|al)?|arr\.?|lands?)\s*:?\s*(\d{1,2}:\d{2})",
    re.IGNORECASE,
)

_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DMY_RE = re.compile(
    r"\b(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s*(\d{4})?\b", re.IGNORECASE
)
_DMY2_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")

_HOTEL_RE = re.compile(
    r"(?:hotel|accommodation|resort)\s*:\s*"
    r"([A-Z][\w .&'\-]{2,40}?)(?=\s*[,.\n]|$)",
    re.IGNORECASE,
)
_CHECKIN_RE = re.compile(
    r"check[\s-]?in\s*[:\-]?\s*([A-Za-z0-9 ,/-]{4,30})", re.IGNORECASE
)
_CHECKOUT_RE = re.compile(
    r"check[\s-]?out\s*[:\-]?\s*([A-Za-z0-9 ,/-]{4,30})", re.IGNORECASE,
)

_HOTEL_CONF_RE = re.compile(
    r"(?:hotel\s+)?confirmation\s*(?:code|number|no|#)?\s*[:#]?\s*"
    r"([A-Z0-9-]*\d[A-Z0-9-]{3,19})",
    re.IGNORECASE,
)

_CAR_RE = re.compile(
    r"(?:car\s+rental|rental\s+car|hire\s+car)\s*[:\-]?\s*"
    r"([A-Z][\w .,'&-]{2,40})?", re.IGNORECASE
)

_MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05",
           "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
           "nov": "11", "dec": "12"}


def _clean_pnr(raw: str) -> str:
    return (raw or "").strip().upper()


def parse_confirmation(text: str) -> dict[str, Any]:
    """Extract booking facts from confirmation text. Pure, never raises.

    Returns ``{"flights": [...], "hotels": [...], "cars": [...],
    "pnr": str, "raw": text}`` — dicts shaped for the dataclasses.
    """
    try:
        return _parse(text or "")
    except Exception:  # noqa: BLE001 — parsing never breaks the flow
        _log.debug("confirmation parse failed", exc_info=True)
        return {"flights": [], "hotels": [], "cars": [],
                "pnr": "", "raw": text or ""}


def _parse(text: str) -> dict[str, Any]:
    flights: list[dict[str, Any]] = []
    hotels: list[dict[str, Any]] = []
    cars: list[dict[str, Any]] = []

    # PNR / booking reference
    pnr_m = _PNR_RE.search(text)
    pnr = _clean_pnr(pnr_m.group(1)) if pnr_m else ""

    # Flights
    seen_fn: set[str] = set()
    for m in _FLIGHT_RE.finditer(text):
        code, num = m.group(1).upper(), m.group(2)
        # skip false positives: times ("22:45" won't match), pure codes
        if code in ("AM", "PM"):
            continue
        fn = f"{code}{num}"
        if fn in seen_fn:
            continue
        seen_fn.add(fn)
        flights.append({"airline": "", "flight_number": fn})

    # Routes → attach to flights in order
    routes = [(a.upper(), b.upper()) for a, b in _ROUTE_RE.findall(text)]
    for i, (o, d) in enumerate(routes):
        if i < len(flights):
            flights[i]["origin"] = o
            flights[i]["destination"] = d
        else:
            flights.append({"airline": "", "flight_number": "",
                            "origin": o, "destination": d})

    # Times → first flight (or standalone)
    dep_m = _TIME_CTX_RE.search(text)
    arr_m = _ARR_CTX_RE.search(text)
    if dep_m or arr_m:
        if not flights:
            flights.append({})
        f = flights[0]
        date = _first_date(text) or ""
        if dep_m:
            f["departs"] = f"{date}T{dep_m.group(1)}" if date else dep_m.group(1)
        if arr_m:
            f["arrives"] = f"{date}T{arr_m.group(1)}" if date else arr_m.group(1)

    if pnr and flights:
        # PNR belongs to the whole booking; attach to each flight
        for f in flights:
            f.setdefault("pnr", pnr)

    # Hotels
    hotel_m = _HOTEL_RE.search(text)
    if hotel_m:
        name = hotel_m.group(1).strip().rstrip(",.")
        ci = _CHECKIN_RE.search(text)
        co = _CHECKOUT_RE.search(text)
        hc = _HOTEL_CONF_RE.search(text)
        hotels.append({
            "name": name,
            "check_in": _norm_date(ci.group(1)) if ci else "",
            "check_out": _norm_date(co.group(1)) if co else "",
            "confirmation": (hc.group(1).strip() if hc else "") or pnr,
        })

    # Car rentals
    if _CAR_RE.search(text):
        cc = _HOTEL_CONF_RE.search(text)
        cars.append({
            "company": "",
            "confirmation": (cc.group(1).strip() if cc else "") or pnr,
        })

    return {"flights": flights, "hotels": hotels, "cars": cars,
            "pnr": pnr, "raw": text}


def _first_date(text: str) -> str:
    m = _ISO_DATE_RE.search(text)
    if m:
        return m.group(1)
    m = _DMY_RE.search(text)
    if m:
        day = m.group(1).zfill(2)
        mon = _MONTHS[m.group(2).lower()[:3]]
        year = m.group(3) or str(time.gmtime().tm_year)
        return f"{year}-{mon}-{day}"
    m = _DMY2_RE.search(text)
    if m:
        d, mo, y = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{mo}-{d}"
    return ""


def _norm_date(raw: str) -> str:
    raw = (raw or "").strip().rstrip(",.")
    iso = _first_date(raw)
    return iso or raw[:16]


# ── builder ─────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


class ItineraryBuilder:
    """Booking confirmations → structured trips + document vault.

    Never raises on ingestion problems — returns a (possibly empty) Trip
    and records what failed in the vault log.
    """

    def __init__(self, db_path: str = "", vault_dir: str = "") -> None:
        self.db_path = db_path or TRIP_DB
        self.vault_dir = vault_dir or TRIP_VAULT_DIR
        _ensure_dir(os.path.dirname(self.db_path))
        _ensure_dir(self.vault_dir)
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS trips
               (id TEXT PRIMARY KEY, name TEXT, data_json TEXT,
                created_at REAL)"""
        )
        self._db.commit()

    # ── ingestion ──

    def ingest_email(self, body: str, *, subject: str = "",
                     sender: str = "") -> Trip:
        """Parse a forwarded booking-confirmation email."""
        text = f"{subject}\n{body}" if subject else (body or "")
        parsed = parse_confirmation(text)
        trip = self._new_trip(parsed, name=self._trip_name(parsed))
        self._vault_text(trip, "email.txt", text,
                         meta={"subject": subject, "from": sender})
        return trip

    def ingest_pdf(self, pdf_path: str) -> Trip:
        """Parse a confirmation PDF. Fail-closed when no PDF tooling."""
        text = self._pdf_text(pdf_path)
        parsed = parse_confirmation(text)
        trip = self._new_trip(parsed, name=self._trip_name(parsed))
        try:
            import shutil
            dest = os.path.join(self._trip_dir(trip),
                                os.path.basename(pdf_path))
            shutil.copy2(pdf_path, dest)
            trip.docs.append(os.path.basename(pdf_path))
            self._save(trip)
        except Exception:  # noqa: BLE001
            _log.debug("pdf vault copy failed", exc_info=True)
        return trip

    def ingest_image(self, image_path: str, seer: Any = None) -> Trip:
        """Parse a confirmation screenshot/photo via Seer."""
        text = ""
        if seer is not None:
            try:
                text = seer.see(
                    image_path,
                    "Read this booking confirmation. List: flight numbers, "
                    "airline, origin/destination airport codes, departure and "
                    "arrival times and dates, booking reference / PNR, hotel "
                    "name with check-in and check-out dates, confirmation "
                    "numbers. Plain text, one fact per line.",
                ) or ""
            except Exception:  # noqa: BLE001
                _log.debug("seer confirmation read failed", exc_info=True)
        parsed = parse_confirmation(text)
        trip = self._new_trip(parsed, name=self._trip_name(parsed))
        try:
            import shutil
            dest = os.path.join(self._trip_dir(trip),
                                os.path.basename(image_path))
            shutil.copy2(image_path, dest)
            trip.docs.append(os.path.basename(image_path))
            self._save(trip)
        except Exception:  # noqa: BLE001
            _log.debug("image vault copy failed", exc_info=True)
        return trip

    def ingest_gmail(self, gmail: Any, query: str = "",
                     *, max_results: int = 10) -> list[Trip]:
        """Search Gmail for booking confirmations and ingest them."""
        trips: list[Trip] = []
        if gmail is None:
            return trips
        q = query or ("subject:(booking confirmation OR e-ticket OR "
                      "itinerary OR reservation)")
        try:
            listing = gmail.list_messages(query=q, max_results=max_results)
            for m in listing.get("messages", []):
                mid = m.get("id", "")
                try:
                    full = gmail.get_message(mid, format="full")
                    body = self._gmail_body(full)
                    subj = self._gmail_subject(full)
                    trips.append(self.ingest_email(body, subject=subj))
                except Exception:  # noqa: BLE001
                    _log.debug("gmail ingest failed for %s", mid,
                               exc_info=True)
        except Exception:  # noqa: BLE001
            _log.debug("gmail search failed", exc_info=True)
        return trips

    # ── trips ──

    def _new_trip(self, parsed: dict[str, Any], name: str = "") -> Trip:
        trip = Trip(
            id="trip_" + uuid.uuid4().hex[:10],
            name=name or self._trip_name(parsed),
            flights=[Flight(**{k: v for k, v in f.items()
                                if k in Flight.__dataclass_fields__})
                     for f in parsed.get("flights", [])],
            hotels=[HotelStay(**{k: v for k, v in h.items()
                                  if k in HotelStay.__dataclass_fields__})
                    for h in parsed.get("hotels", [])],
            cars=[CarRental(**{k: v for k, v in c.items()
                                if k in CarRental.__dataclass_fields__})
                  for c in parsed.get("cars", [])],
            created_at=time.time(),
        )
        self._save(trip)
        return trip

    @staticmethod
    def _trip_name(parsed: dict[str, Any]) -> str:
        fl = parsed.get("flights") or []
        if fl and fl[0].get("origin") and fl[0].get("destination"):
            return f"{fl[0]['origin']} → {fl[0]['destination']}"
        ht = parsed.get("hotels") or []
        if ht and ht[0].get("name"):
            return ht[0]["name"]
        return "trip"

    def get_trip(self, trip_id: str) -> Trip | None:
        try:
            row = self._db.execute(
                "SELECT data_json FROM trips WHERE id = ?", (trip_id,)
            ).fetchone()
            if not row:
                return None
            return self._from_row(trip_id, row["data_json"])
        except Exception:  # noqa: BLE001
            return None

    def list_trips(self) -> list[Trip]:
        try:
            rows = self._db.execute(
                "SELECT id, data_json FROM trips ORDER BY created_at DESC"
            ).fetchall()
            return [self._from_row(r["id"], r["data_json"]) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def _from_row(self, trip_id: str, data_json: str) -> Trip:
        d = json.loads(data_json or "{}")
        return Trip(
            id=trip_id,
            name=d.get("name", ""),
            flights=[Flight(**f) for f in d.get("flights", [])],
            hotels=[HotelStay(**h) for h in d.get("hotels", [])],
            cars=[CarRental(**c) for c in d.get("cars", [])],
            docs=d.get("docs", []),
            created_at=d.get("created_at", 0.0),
        )

    def _save(self, trip: Trip) -> None:
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO trips (id, name, data_json, "
                "created_at) VALUES (?, ?, ?, ?)",
                (trip.id, trip.name, json.dumps(asdict(trip)),
                 trip.created_at),
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("trip save failed", exc_info=True)

    # ── vault ──

    def _trip_dir(self, trip: Trip) -> str:
        return _ensure_dir(os.path.join(self.vault_dir, trip.id))

    def _vault_text(self, trip: Trip, filename: str, text: str,
                    meta: dict[str, Any] | None = None) -> None:
        try:
            path = os.path.join(self._trip_dir(trip), filename)
            with open(path, "w", encoding="utf-8") as fh:
                if meta:
                    fh.write(json.dumps(meta) + "\n---\n")
                fh.write(text or "")
            if filename not in trip.docs:
                trip.docs.append(filename)
                self._save(trip)
        except Exception:  # noqa: BLE001
            _log.debug("vault write failed", exc_info=True)

    def vault_docs(self, trip_id: str) -> list[str]:
        trip = self.get_trip(trip_id)
        return list(trip.docs) if trip else []

    # ── calendar ──

    def to_calendar(self, trip_id: str, gcal: Any, *,
                    confirmed: bool = False, db: Any = None,
                    context: Any = None) -> list[dict[str, Any]]:
        """Create calendar events for a trip. Confirmation-gated via gcal.

        Each flight/hotel becomes an event. Returns the created events.
        """
        trip = self.get_trip(trip_id)
        if trip is None or gcal is None:
            return []
        created: list[dict[str, Any]] = []
        for f in trip.flights:
            if not f.departs:
                continue
            try:
                ev = gcal.create_event(
                    summary=f"✈️ {f.flight_number or 'Flight'} "
                            f"{f.origin}→{f.destination}",
                    start={"dateTime": self._dt(f.departs)},
                    end={"dateTime": self._dt(f.arrives or f.departs)},
                    description=f"PNR {f.pnr}" if f.pnr else "",
                    location=f.origin or "",
                    confirmed=confirmed, db=db, context=context,
                )
                created.append(ev if isinstance(ev, dict) else {"ok": True})
            except Exception:  # noqa: BLE001
                _log.debug("calendar flight event failed", exc_info=True)
        for h in trip.hotels:
            if not h.check_in:
                continue
            try:
                ev = gcal.create_event(
                    summary=f"🏨 {h.name or 'Hotel'}",
                    start={"date": h.check_in[:10]},
                    end={"date": (h.check_out or h.check_in)[:10]},
                    description=(f"Confirmation {h.confirmation}"
                                 if h.confirmation else ""),
                    location=h.address or "",
                    confirmed=confirmed, db=db, context=context,
                )
                created.append(ev if isinstance(ev, dict) else {"ok": True})
            except Exception:  # noqa: BLE001
                _log.debug("calendar hotel event failed", exc_info=True)
        return created

    @staticmethod
    def _dt(raw: str) -> str:
        raw = (raw or "").strip()
        if "T" in raw and len(raw) >= 16:
            return raw[:16] + ":00"
        if re.fullmatch(r"\d{1,2}:\d{2}", raw):
            return raw + ":00"
        return raw

    # ── summary ──

    def summary(self, trip_id: str, *,
                total_kobo: int = 0,
                budget_kobo: int | None = None,
                programs: list | None = None) -> str:
        # #75: cost summary first when a total is known.
        trip = self.get_trip(trip_id)
        if trip is None:
            return "no such trip."
        lines = [f"🧳 {trip.name or trip.id}"]
        if total_kobo:
            from .display import _naira, points_vs_cash
            lines.append(f"💰 Total: {_naira(total_kobo)}")
            if budget_kobo:
                if total_kobo <= budget_kobo:
                    lines.append("   within budget ✅")
                else:
                    lines.append(
                        f"   over budget by "
                        f"{_naira(total_kobo - budget_kobo)} ⚠️")
            if programs:
                pvc = points_vs_cash(total_kobo, programs)
                if pvc:
                    lines.append(f"🎖️ {pvc}")
        for f in trip.flights:
            lines.append(f"  ✈️ {f.one_line()}")
        for h in trip.hotels:
            lines.append(f"  🏨 {h.one_line()}")
        for c in trip.cars:
            lines.append(f"  🚗 {c.one_line()}")
        if trip.is_empty():
            lines.append("  (nothing parsed yet — forward a confirmation)")
        if trip.docs:
            lines.append(f"  📎 {len(trip.docs)} doc(s) in the vault")
        return "\n".join(lines)

    # ── helpers ──

    @staticmethod
    def _pdf_text(pdf_path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            _log.debug("pypdf read failed", exc_info=True)
        # pdftotext fallback
        try:
            import subprocess
            out = subprocess.run(
                ["pdftotext", pdf_path, "-"], capture_output=True,
                text=True, timeout=30)
            if out.returncode == 0:
                return out.stdout
        except Exception:  # noqa: BLE001
            pass
        return ""

    @staticmethod
    def _gmail_body(full: dict[str, Any]) -> str:
        """Best-effort plain-text extraction from a Gmail message dict."""
        try:
            payload = full.get("payload", {}) or {}
            parts: list[str] = []

            def walk(p: dict[str, Any]) -> None:
                mime = p.get("mimeType", "")
                body = p.get("body", {}) or {}
                data = body.get("data", "")
                if data and mime.startswith("text/plain"):
                    try:
                        parts.append(base64.urlsafe_b64decode(
                            data + "=" * (-len(data) % 4)).decode(
                                "utf-8", "replace"))
                    except Exception:  # noqa: BLE001
                        pass
                for sub in p.get("parts", []) or []:
                    walk(sub)

            walk(payload)
            if parts:
                return "\n".join(parts)
            return full.get("snippet", "") or ""
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _gmail_subject(full: dict[str, Any]) -> str:
        try:
            for h in (full.get("payload", {}) or {}).get("headers", []):
                if h.get("name", "").lower() == "subject":
                    return h.get("value", "")
        except Exception:  # noqa: BLE001
            pass
        return ""


# ── chat: forwarded-confirmation detection ────────────────────────────────────

_CONFIRMATION_HINTS = (
    "booking confirmation", "e-ticket", "eticket", "itinerary",
    "reservation confirmed", "your trip", "pnr",
    "booking reference", "confirmation code",
)


def confirmation_hook(text: str) -> bool:
    """True when pasted text looks like a forwarded booking confirmation."""
    low = (text or "").lower()
    if len(low) < 60:
        return False
    hits = sum(1 for h in _CONFIRMATION_HINTS if h in low)
    has_pnr = bool(_PNR_RE.search(text or ""))
    has_flight = bool(_FLIGHT_RE.search(text or ""))
    return hits >= 1 and (has_pnr or has_flight or hits >= 2)


def added_message(trip: Trip) -> str:
    """'Added: Lagos→London, BA075, departs 22:45, PNR ABC123. ...'"""
    bits: list[str] = []
    for f in trip.flights[:2]:
        bits.append(f.one_line())
    for h in trip.hotels[:1]:
        bits.append(f"🏨 {h.one_line()}")
    detail = "; ".join(b for b in bits if b) or "booking"
    msg = f"Added: {detail}."
    if trip.flights:
        msg += " Want me to track it?"
    return msg
