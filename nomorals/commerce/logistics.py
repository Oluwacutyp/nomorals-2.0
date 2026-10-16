"""Logistics: GIG / Kwik / Sendbox descriptors, quotes, ETAs, tracking.

Nigeria-first delivery. Three real operators as descriptors:

- **GIG Logistics** — nationwide + international, terminal drop-off and
  doorstep, best for parcels that can wait a day or two.
- **Kwik Delivery** — Lagos (expanding) bike/courier, same-day /
  on-demand, best for intra-city speed.
- **Sendbox** — aggregator: compares couriers, handles pickups and
  international, best when the operator wants one dashboard.

Honesty rules:
- ``rate_quote()`` / ``eta()`` use an offline tariff table and are
  labelled ``estimated=True, source="offline tariff table"`` — they are
  planning numbers, NOT a live courier quote. The caller may pass a
  ``live_quote`` callable to get a real quote; without one, no live
  price is ever claimed.
- ``tracking()`` returns ``status="not_available"`` unless the caller
  supplies a real ``tracker`` callable (``tracker(carrier, waybill)``).
  A tracking number is never invented.
- ``validate_address()`` checks structure (Nigeria: state + 6-digit
  postal code + locality; generic: locality + country), never claims
  a real address exists — that needs a geocoder.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Carrier",
    "CARRIERS",
    "Address",
    "validate_address",
    "rate_quote",
    "eta",
    "tracking",
    "NIGERIAN_STATES",
]

NIGERIAN_STATES: tuple[str, ...] = (
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa",
    "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti",
    "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina",
    "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun",
    "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
    "Yobe", "Zamfara", "FCT",
)

_STATE_ALIASES = {s.lower(): s for s in NIGERIAN_STATES}
_STATE_ALIASES["abuja"] = "FCT"
_STATE_ALIASES["federal capital territory"] = "FCT"


@dataclass(frozen=True)
class Carrier:
    """Descriptor for one logistics operator."""
    name: str
    coverage: str
    speed: str
    best_for: str
    base_fee_minor: int      # minor units (kobo) — offline planning figure
    per_km_minor: int
    per_kg_minor: int
    supports_tracking: bool = True
    notes: str = ""


CARRIERS: dict[str, Carrier] = {
    "gig": Carrier(
        name="GIG Logistics", coverage="Nationwide NG + international",
        speed="Next-day interstate, 2-4 days international",
        best_for="Parcels that can wait a day; terminal drop-off",
        base_fee_minor=150_000, per_km_minor=50, per_kg_minor=50_000,
        notes="Terminal + doorstep options; book via GIG app / terminal.",
    ),
    "kwik": Carrier(
        name="Kwik Delivery", coverage="Lagos + expanding cities",
        speed="Same-day / on-demand (hours)",
        best_for="Intra-city speed; food, documents, small parcels",
        base_fee_minor=100_000, per_km_minor=120, per_kg_minor=20_000,
        notes="Bike + van fleet; on-demand via Kwik app.",
    ),
    "sendbox": Carrier(
        name="Sendbox", coverage="Nigeria + international (aggregator)",
        speed="Varies by underlying courier",
        best_for="One dashboard, pickups, international shipping",
        base_fee_minor=200_000, per_km_minor=60, per_kg_minor=60_000,
        notes="Aggregator — compares couriers; final price from Sendbox.",
    ),
}


@dataclass
class Address:
    """A delivery address. ``country`` is an ISO code (``NG`` default)."""
    line1: str = ""
    locality: str = ""   # area / town / city
    state: str = ""
    postal_code: str = ""
    country: str = "NG"
    phone: str = ""
    recipient: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "line1": self.line1, "locality": self.locality,
            "state": self.state, "postal_code": self.postal_code,
            "country": self.country, "phone": self.phone,
            "recipient": self.recipient,
        }


def validate_address(address: Address) -> dict[str, Any]:
    """Validate address *structure*. Returns ``{ok, errors, normalized}``.

    Nigeria (country NG):
    - line1 + locality required; state must be a real Nigerian state/FCT;
    - postal_code, when present, must be 6 digits; phone, when present,
      must look like a Nigerian number (+234 / 0 + 10 digits).

    Generic (other countries): line1 + locality + country required.
    This never claims the address exists — that needs a geocoder.
    """
    errors: list[str] = []
    addr = Address(
        line1=(address.line1 or "").strip(),
        locality=(address.locality or "").strip(),
        state=(address.state or "").strip(),
        postal_code=re.sub(r"\s+", "", address.postal_code or ""),
        country=(address.country or "NG").strip().upper(),
        phone=re.sub(r"[\s\-()]", "", address.phone or ""),
        recipient=(address.recipient or "").strip(),
    )
    if not addr.line1:
        errors.append("line1 (street address) is required")
    if not addr.locality:
        errors.append("locality (area/town/city) is required")
    if addr.country == "NG":
        canonical = _STATE_ALIASES.get(addr.state.lower())
        if not addr.state:
            errors.append("state is required for Nigerian addresses")
        elif canonical is None:
            errors.append(f"unknown Nigerian state: {addr.state!r}")
        else:
            addr.state = canonical
        if addr.postal_code and not re.fullmatch(r"\d{6}", addr.postal_code):
            errors.append("Nigerian postal codes are 6 digits")
        if addr.phone and not re.fullmatch(r"(\+234|0)\d{10}", addr.phone):
            errors.append("phone should look like +234XXXXXXXXXX or 0XXXXXXXXXX")
    elif not addr.country:
        errors.append("country is required")
    return {"ok": not errors, "errors": errors, "normalized": addr}


def _carrier(name: str) -> Carrier:
    key = (name or "").strip().lower()
    if key not in CARRIERS:
        raise ValueError(f"unknown carrier {name!r} — known: {sorted(CARRIERS)}")
    return CARRIERS[key]


def rate_quote(carrier_name: str, *, origin: Address, destination: Address,
               weight_kg: float, distance_km: float | None = None,
               live_quote: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Quote a delivery price.

    Without ``live_quote``: offline tariff-table estimate
    (``estimated=True``). With ``live_quote``: the callable's real result
    is returned verbatim with ``estimated=False`` — Devon never edits a
    live quote into a different number.
    """
    carrier = _carrier(carrier_name)
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if live_quote is not None:
        result = live_quote(carrier=carrier, origin=origin,
                            destination=destination, weight_kg=weight_kg,
                            distance_km=distance_km)
        if isinstance(result, dict):
            result.setdefault("estimated", False)
            result.setdefault("source", "live carrier quote")
            return result
        return {"estimated": False, "source": "live carrier quote",
                "quote": result}
    # Offline planning estimate. Distance unknown → intra-city default 15km.
    km = float(distance_km) if distance_km is not None else 15.0
    fee = (carrier.base_fee_minor
           + int(carrier.per_km_minor * km)
           + int(carrier.per_kg_minor * max(0.0, weight_kg - 1.0)))
    return {
        "carrier": carrier.name,
        "fee_minor": fee,
        "currency": "NGN",
        "weight_kg": weight_kg,
        "distance_km": km,
        "estimated": True,
        "source": "offline tariff table — planning figure, not a live quote",
    }


def eta(carrier_name: str, *, distance_km: float | None = None,
        same_city: bool = False) -> dict[str, Any]:
    """Estimated delivery window from distance bands (planning figure)."""
    carrier = _carrier(carrier_name)
    if carrier_name.lower() == "kwik" or same_city:
        window = "2–6 hours (same-day)"
    elif distance_km is None:
        window = "1–3 days"
    elif distance_km <= 50:
        window = "same-day to next-day"
    elif distance_km <= 500:
        window = "1–2 days"
    else:
        window = "2–4 days"
    return {
        "carrier": carrier.name,
        "eta": window,
        "estimated": True,
        "source": "offline distance bands — planning figure",
    }


def tracking(carrier_name: str, waybill: str, *,
             tracker: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Track a parcel. Honest not-available fallback without a tracker.

    ``tracker`` is ``tracker(carrier, waybill) -> dict`` — the real
    carrier lookup. Without it, returns ``status="not_available"`` with
    the reason, never an invented location or status.
    """
    carrier = _carrier(carrier_name)
    waybill = (waybill or "").strip()
    if not waybill:
        return {"carrier": carrier.name, "waybill": "",
                "status": "not_available",
                "reason": "no waybill number supplied"}
    if tracker is None:
        return {
            "carrier": carrier.name, "waybill": waybill,
            "status": "not_available",
            "reason": ("No live tracker wired for "
                       f"{carrier.name} — pass tracker=... to track for real. "
                       "Check the carrier app with this waybill."),
        }
    try:
        result = tracker(carrier=carrier, waybill=waybill)
    except Exception as exc:  # noqa: BLE001 — tracker failure is data
        return {"carrier": carrier.name, "waybill": waybill,
                "status": "not_available", "reason": f"tracker failed: {exc}"}
    if isinstance(result, dict):
        result.setdefault("carrier", carrier.name)
        result.setdefault("waybill", waybill)
        return result
    return {"carrier": carrier.name, "waybill": waybill,
            "status": str(result)}
