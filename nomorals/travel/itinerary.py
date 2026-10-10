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
    "AIRLINE_CODES", "airline_for_flight_number",
    "trip_status", "trip_timeline", "detect_conflicts",
    "export_ics", "packing_list", "format_countdown",
]

TRIP_VAULT_DIR = os.path.expanduser("~/.nomorals/travel/vault")
TRIP_DB = os.path.expanduser("~/.nomorals/travel/trips.db")

#: Curated IATA flight-number prefix → airline. Nigerian carriers first,
#: then the majors — fills ``Flight.airline`` when the confirmation only
#: gives a flight number (TripIt-style enrichment).
AIRLINE_CODES = {
    # Nigeria
    "P4": "Air Peace", "W3": "Arik Air", "9J": "Dana Air",
    "QI": "Ibom Air", "VM": "Max Air", "OJ": "Overland Airways",
    "VK": "ValueJet", "Q9": "Green Africa", "R4": "Rano Air",
    # Africa / Middle East / Europe / Americas majors
    "BA": "British Airways", "VS": "Virgin Atlantic",
    "AF": "Air France", "KL": "KLM", "LH": "Lufthansa",
    "EK": "Emirates", "QR": "Qatar Airways", "ET": "Ethiopian Airlines",
    "TK": "Turkish Airlines", "MS": "EgyptAir", "AT": "Royal Air Maroc",
    "SA": "South African Airways", "KQ": "Kenya Airways",
    "DL": "Delta", "UA": "United Airlines", "AA": "American Airlines",
    "WN": "Southwest", "B6": "JetBlue", "AC": "Air Canada",
}


def airline_for_flight_number(flight_number: str) -> str:
    """'BA075' → 'British Airways'. '' when the prefix is unknown."""
    fn = (flight_number or "").upper().strip()
    m = re.match(r"^([A-Z]{2})", fn)
    if m:
        return AIRLINE_CODES.get(m.group(1), "")
    return ""


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

    # Airline names from flight-number prefixes
    for f in flights:
        if not f.get("airline") and f.get("flight_number"):
            f["airline"] = airline_for_flight_number(f["flight_number"])

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
                     sender: str = "", auto_merge: bool = True) -> Trip:
        """Parse a forwarded booking-confirmation email.

        TripIt-style merging: when the booking shares a PNR/booking
        reference with an existing trip, it folds into that trip instead
        of fragmenting the vault. Date-proximity matches are surfaced via
        find_merge_candidate() as suggestions, never silent merges.
        """
        text = f"{subject}\n{body}" if subject else (body or "")
        parsed = parse_confirmation(text)
        if auto_merge:
            candidate = self.find_merge_candidate(parsed)
            if (candidate is not None
                    and self._pnr_merge_ok(candidate, parsed)):
                trip = self._absorb(candidate, parsed)
                self._vault_text(trip, "email.txt", text,
                                 meta={"subject": subject, "from": sender,
                                       "merged": True})
                return trip
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

    # ── TripIt-style merging ──────────────────────────────────────────

    def delete_trip(self, trip_id: str) -> bool:
        """Remove a trip record (used by merges). Returns True on delete."""
        try:
            cur = self._db.execute("DELETE FROM trips WHERE id = ?",
                                   (trip_id,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def _parsed_dates(self, parsed: dict[str, Any]) -> set[str]:
        ds: set[str] = set()
        for f in parsed.get("flights", []):
            d = _parse_dt(f.get("departs", ""))
            if d:
                ds.add(d.strftime("%Y-%m-%d"))
        for h in parsed.get("hotels", []):
            d = _parse_dt(h.get("check_in", ""))
            if d:
                ds.add(d.strftime("%Y-%m-%d"))
        return ds

    def find_merge_candidate(self, parsed: dict[str, Any],
                             exclude_id: str = "") -> Trip | None:
        """Best existing trip this booking probably belongs to — for the
        'looks like your Lagos→London trip, merge?' suggestion (TripIt).

        PNR/booking-reference match first (precise), then date proximity
        within 3 days (the usual flight-lands / hotel-next-morning case).
        ``exclude_id`` skips the trip the booking just landed in.
        """
        import datetime as _dt
        trips = [t for t in self.list_trips() if t.id != exclude_id]
        pnr = (parsed.get("pnr") or "").upper()
        if pnr:
            for trip in trips:
                if pnr in self._trip_pnrs(trip):
                    return trip

        def _ord(d: str) -> int:
            return _dt.datetime.strptime(d, "%Y-%m-%d").toordinal()

        want = self._parsed_dates(parsed)
        if not want:
            return None
        wmin, wmax = min(want), max(want)
        for trip in trips:
            start, end = trip_dates(trip)
            if not start:
                continue
            if (_ord(wmin) <= _ord(end) + 3
                    and _ord(wmax) >= _ord(start) - 3):
                return trip
        return None

    @staticmethod
    def _trip_pnrs(trip: Trip) -> set[str]:
        """Every booking reference on a trip (flights + hotels + cars)."""
        codes: set[str] = set()
        for f in trip.flights:
            if f.pnr:
                codes.add(f.pnr.upper())
        for h in trip.hotels:
            if h.confirmation:
                codes.add(h.confirmation.upper())
        for c in trip.cars:
            if c.confirmation:
                codes.add(c.confirmation.upper())
        return codes

    def _pnr_merge_ok(self, trip: Trip, parsed: dict[str, Any]) -> bool:
        """Auto-merge only on a shared booking reference — precise, never a
        surprise. Date-proximity candidates are suggestions, not merges."""
        pnr = (parsed.get("pnr") or "").upper()
        return bool(pnr) and pnr in self._trip_pnrs(trip)

    def merge_trips(self, trip_ids: list[str]) -> Trip | None:
        """Fold several trips into the first one; delete the rest."""
        ids = [i for i in (trip_ids or []) if i]
        if not ids:
            return None
        base = self.get_trip(ids[0])
        if base is None:
            return None
        for tid in ids[1:]:
            other = self.get_trip(tid)
            if other is None or other.id == base.id:
                continue
            for f in other.flights:
                if asdict(f) not in [asdict(x) for x in base.flights]:
                    base.flights.append(f)
            for h in other.hotels:
                if asdict(h) not in [asdict(x) for x in base.hotels]:
                    base.hotels.append(h)
            for c in other.cars:
                if asdict(c) not in [asdict(x) for x in base.cars]:
                    base.cars.append(c)
            for doc in other.docs:
                if doc not in base.docs:
                    base.docs.append(doc)
            self.delete_trip(other.id)
        self._save(base)
        return base

    def _absorb(self, trip: Trip, parsed: dict[str, Any]) -> Trip:
        """Fold a freshly parsed booking into an existing trip."""
        have_f = [asdict(f) for f in trip.flights]
        for f in parsed.get("flights", []):
            clean = {k: v for k, v in f.items()
                     if k in Flight.__dataclass_fields__}
            if clean not in have_f:
                trip.flights.append(Flight(**clean))
                have_f.append(clean)
        have_h = [asdict(h) for h in trip.hotels]
        for h in parsed.get("hotels", []):
            clean = {k: v for k, v in h.items()
                     if k in HotelStay.__dataclass_fields__}
            if clean not in have_h:
                trip.hotels.append(HotelStay(**clean))
                have_h.append(clean)
        have_c = [asdict(c) for c in trip.cars]
        for c in parsed.get("cars", []):
            clean = {k: v for k, v in c.items()
                     if k in CarRental.__dataclass_fields__}
            if clean not in have_c:
                trip.cars.append(CarRental(**clean))
                have_c.append(clean)
        self._save(trip)
        return trip

    # ── trip intelligence wrappers ────────────────────────────────────

    def timeline(self, trip_id: str) -> str:
        trip = self.get_trip(trip_id)
        if trip is None:
            return "no such trip."
        events = trip_timeline(trip)
        if not events:
            return f"🧳 {trip.name or trip.id} — no timed events yet."
        lines = [f"🧳 {trip.name or trip.id} — travel timeline"]
        for e in events:
            cd = f" ({e['countdown']})" if e["countdown"] else ""
            det = f" — {e['detail']}" if e["detail"] else ""
            lines.append(f"  {e['text']}{cd}{det}")
        return "\n".join(lines)

    def conflicts(self, trip_id: str) -> list[str]:
        trip = self.get_trip(trip_id)
        return detect_conflicts(trip) if trip else []

    def export_ics(self, trip_id: str) -> str:
        """Write the trip's .ics calendar file into the vault. '' on miss."""
        trip = self.get_trip(trip_id)
        if trip is None:
            return ""
        path = os.path.join(self._trip_dir(trip), f"{trip.id}.ics")
        try:
            return export_ics(trip, path)
        except Exception:  # noqa: BLE001
            _log.debug("ics export failed", exc_info=True)
            return ""

    def packing(self, trip_id: str) -> str:
        trip = self.get_trip(trip_id)
        if trip is None:
            return "no such trip."
        items = packing_list(trip)
        if not items:
            return "Nothing to pack yet — add a flight or hotel first. 🎒"
        lines = [f"🎒 Packing for {trip.name or trip.id}:"]
        lines += [f"  • {item} — {reason}" for item, reason in items]
        return "\n".join(lines)

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
        status = trip_status(trip)
        lines = [f"{_STATUS_EMOJI.get(status, '🧳')} {trip.name or trip.id}"]
        # next event countdown — the travel-day card
        upcoming = [e for e in trip_timeline(trip)
                    if e["when"] and e["when"] >= time.time() - 3600]
        if upcoming:
            nxt = upcoming[0]
            lines.append(f"   next: {nxt['text']}"
                         + (f" — {nxt['countdown']}" if nxt["countdown"]
                            else ""))
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
            al = f" ({f.airline})" if f.airline else ""
            lines.append(f"  ✈️ {f.one_line()}{al}")
        for h in trip.hotels:
            lines.append(f"  🏨 {h.one_line()}")
        for c in trip.cars:
            lines.append(f"  🚗 {c.one_line()}")
        for w in detect_conflicts(trip)[:3]:
            lines.append(f"  {w}")
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


# ── trip intelligence: status, timeline, conflicts, export, packing ──────────

def _parse_dt(raw: str):
    """'2026-12-01T22:45' → datetime; '2026-12-01' → midnight; else None."""
    import datetime as _dt
    raw = (raw or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(raw[:16], fmt)
        except ValueError:
            continue
    return None


def format_countdown(ts: float, now: float | None = None) -> str:
    """'in 3d 4h' / 'tomorrow' / 'in 2h' / 'boarding soon' / '3d ago'."""
    import datetime as _dt
    now = now if now is not None else time.time()
    delta = ts - now
    if delta < -86400:
        return f"{int(-delta // 86400)}d ago"
    if delta < -3600:
        return f"{int(-delta // 3600)}h ago"
    if delta <= 0:
        return "now"
    if delta < 3600:
        return f"in {int(delta // 60)}m"
    if delta < 86400:
        return f"in {int(delta // 3600)}h"
    days = int(delta // 86400)
    if days == 1:
        return "tomorrow"
    return f"in {days}d"


def trip_dates(trip: Trip) -> tuple[str, str]:
    """(earliest, latest) YYYY-MM-DD across flights + hotels. ('','') if none."""
    ds: list[str] = []
    for f in trip.flights:
        for raw in (f.departs, f.arrives):
            d = _parse_dt(raw)
            if d:
                ds.append(d.strftime("%Y-%m-%d"))
    for h in trip.hotels:
        for raw in (h.check_in, h.check_out):
            d = _parse_dt(raw)
            if d:
                ds.append(d.strftime("%Y-%m-%d"))
    if not ds:
        return "", ""
    return min(ds), max(ds)


def trip_status(trip: Trip, now: float | None = None) -> str:
    """upcoming | in-progress | past | unknown — from booking dates."""
    import datetime as _dt
    now = now if now is not None else time.time()
    start, end = trip_dates(trip)
    if not start:
        return "unknown"
    s_ts = _dt.datetime.strptime(start, "%Y-%m-%d").timestamp()
    e_ts = (_dt.datetime.strptime(end, "%Y-%m-%d").timestamp()
            + 86399)
    if now < s_ts:
        return "upcoming"
    if now <= e_ts:
        return "in-progress"
    return "past"


_STATUS_EMOJI = {"upcoming": "🗓️", "in-progress": "✈️",
                 "past": "🗂️", "unknown": "🧳"}


def trip_timeline(trip: Trip,
                  now: float | None = None) -> list[dict[str, Any]]:
    """Chronological travel-day cards with countdowns. Pure."""
    import datetime as _dt
    now = now if now is not None else time.time()
    events: list[dict[str, Any]] = []
    for f in trip.flights:
        d = _parse_dt(f.departs)
        route = f"{f.origin}→{f.destination}" if f.origin else "flight"
        label = f"{f.airline + ' ' if f.airline else ''}{f.flight_number or ''}".strip()
        events.append({
            "kind": "flight", "when": d.timestamp() if d else 0.0,
            "text": f"🛫 {label} {route}".strip(),
            "countdown": format_countdown(d.timestamp(), now) if d else "",
            "detail": f"PNR {f.pnr}" if f.pnr else "",
        })
        da = _parse_dt(f.arrives)
        if da:
            events.append({
                "kind": "arrival", "when": da.timestamp(),
                "text": f"🛬 lands {f.destination or ''}".strip(),
                "countdown": format_countdown(da.timestamp(), now),
                "detail": "",
            })
    for h in trip.hotels:
        ci = _parse_dt(h.check_in)
        if ci:
            events.append({
                "kind": "hotel", "when": ci.timestamp(),
                "text": f"🏨 check in — {h.name or 'hotel'}",
                "countdown": format_countdown(ci.timestamp(), now),
                "detail": (f"conf {h.confirmation}"
                           if h.confirmation else ""),
            })
        co = _parse_dt(h.check_out)
        if co:
            events.append({
                "kind": "hotel-out", "when": co.timestamp() + 86399,
                "text": f"🏨 check out — {h.name or 'hotel'}",
                "countdown": format_countdown(co.timestamp() + 86399, now),
                "detail": "",
            })
    for c in trip.cars:
        events.append({
            "kind": "car", "when": 0.0,
            "text": f"🚗 {c.company or 'car rental'}",
            "countdown": "",
            "detail": (f"conf {c.confirmation}"
                       if c.confirmation else ""),
        })
    events.sort(key=lambda e: (e["when"] or float("inf")))
    return events


def detect_conflicts(trip: Trip) -> list[str]:
    """Real schedule problems: overlapping flights, hotel/check-in gaps."""
    warns: list[str] = []
    segs: list[tuple[float, float, str]] = []
    for f in trip.flights:
        d, a = _parse_dt(f.departs), _parse_dt(f.arrives)
        if d and a:
            if a <= d:
                warns.append(
                    f"⚠️ {f.flight_number or 'flight'} lands before it "
                    f"departs — check the dates.")
            segs.append((d.timestamp(), a.timestamp(),
                         f.flight_number or "flight"))
    for i, (s1, e1, n1) in enumerate(segs):
        for s2, e2, n2 in segs[i + 1:]:
            if s1 < e2 and s2 < e1:
                warns.append(
                    f"⚠️ {n1} and {n2} overlap — double-booked?")
    for h in trip.hotels:
        ci, co = _parse_dt(h.check_in), _parse_dt(h.check_out)
        if ci and co and co <= ci:
            warns.append(
                f"⚠️ {h.name or 'hotel'}: check-out is not after check-in.")
    # hotel check-in before the last flight lands?
    if segs and trip.hotels:
        last_land = max(e for _, e, _ in segs)
        for h in trip.hotels:
            ci = _parse_dt(h.check_in)
            if ci and ci.timestamp() < last_land - 86400:
                warns.append(
                    f"⚠️ {h.name or 'hotel'} check-in is a day+ before "
                    f"you land — confirm the date.")
    return warns


def _ics_escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace(";", "\\;") \
        .replace(",", "\\,").replace("\n", "\\n")


def export_ics(trip: Trip, path: str) -> str:
    """Write a TripIt-style .ics calendar file for the trip. Pure I/O."""
    import datetime as _dt
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0",
             "PRODID:-//Devon//Travel//EN"]
    uid_base = trip.id.replace(" ", "_")
    n = 0

    def event(uid: str, start: _dt.datetime, end: _dt.datetime,
              summary: str, desc: str = "", all_day: bool = False) -> None:
        nonlocal n
        n += 1
        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{uid}-{n}@devon.travel")
        lines.append("DTSTAMP:" + _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"))
        if all_day:
            lines.append("DTSTART;VALUE=DATE:" + start.strftime("%Y%m%d"))
            lines.append("DTEND;VALUE=DATE:" + end.strftime("%Y%m%d"))
        else:
            lines.append("DTSTART:" + start.strftime("%Y%m%dT%H%M%S"))
            lines.append("DTEND:" + end.strftime("%Y%m%dT%H%M%S"))
        lines.append("SUMMARY:" + _ics_escape(summary))
        if desc:
            lines.append("DESCRIPTION:" + _ics_escape(desc))
        lines.append("END:VEVENT")

    for f in trip.flights:
        d, a = _parse_dt(f.departs), _parse_dt(f.arrives)
        if not d:
            continue
        end = a if a and a > d else d + _dt.timedelta(hours=2)
        label = f"{f.airline + ' ' if f.airline else ''}" \
                f"{f.flight_number or 'Flight'}"
        route = f"{f.origin}→{f.destination}" if f.origin else ""
        event(uid_base, d, end, f"✈️ {label} {route}".strip(),
              f"PNR {f.pnr}" if f.pnr else "")
    for h in trip.hotels:
        ci, co = _parse_dt(h.check_in), _parse_dt(h.check_out)
        if not ci:
            continue
        end = co if co and co > ci else ci + _dt.timedelta(days=1)
        event(uid_base, ci, end, f"🏨 {h.name or 'Hotel'}",
              f"Confirmation {h.confirmation}" if h.confirmation else "",
              all_day=True)
    lines.append("END:VCALENDAR")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    return path


def packing_list(trip: Trip) -> list[tuple[str, str]]:
    """Rule-based packing checklist from trip facts. (item, reason)."""
    items: list[tuple[str, str]] = []
    intl = any(f.origin and f.destination and len(f.origin) == 3
               and len(f.destination) == 3 and f.origin[:2] != f.destination[:2]
               for f in trip.flights)
    # crude but honest: different country-ish routes → travel docs
    if trip.flights and not intl:
        intl = len({f.destination for f in trip.flights if f.destination}) > 1
    nights = 0
    for h in trip.hotels:
        ci, co = _parse_dt(h.check_in), _parse_dt(h.check_out)
        if ci and co and co > ci:
            nights = max(nights, (co - ci).days)
    if intl or trip.flights:
        items.append(("passport / ID", "flying — travel documents"))
    if trip.flights:
        items += [("phone charger + power bank", "long travel day"),
                  ("snacks + water", "airport time"),
                  ("boarding pass (offline copy)", "spotty airport wifi")]
    if nights:
        items.append((f"clothes for {nights} night{'s' if nights != 1 else ''}",
                      f"{nights}-night stay"))
        items += [("toiletries", f"{nights}-night stay"),
                  ("medications", "daily routine")]
    if trip.hotels:
        items.append(("hotel confirmation (offline copy)",
                      "front-desk check-in"))
    if trip.cars:
        items.append(("driver's licence", "car rental pickup"))
    return items
