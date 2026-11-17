"""Airport → country mapping (IATA code → ISO 3166-1 alpha-2 country).

The data is NOT hand-written: ``data/airports_iata_country.csv`` was
generated from the OurAirports ``airports.csv`` dump (public domain —
"All data is released to the Public Domain, and comes with no guarantee
of accuracy or fitness for use", https://ourairports.com/data/).

- Canonical source: https://ourairports.com/data/airports.csv
- Fetched copy: https://raw.githubusercontent.com/davidmegginson/
  ourairports-data/main/airports.csv (the same file OurAirports
  publishes; upstream last modified Nov 6, 2026, fetched 2026-11-17)
- Source sha256:
  3d7eeb55961be0f1691e3420ec0e05d4a58be4597c61540511e3e838d7e486f9
- Coverage at generation: 86,371 dump rows → 9,053 distinct IATA codes
  → 239 countries, 0 rows dropped, 0 conflicting codes.
- Regeneration: ``generate_mapping.py`` in the lane verify dir
  (~/workspace/airportmap-verify/) — re-run it against a fresh dump
  and replace the CSV; every row stays traceable to a dataset row by
  its IATA code.

Why this module exists: the itinerary builder's flights carry IATA
codes (LOS→LHR) while the visa layer answers from ISO country codes
(NG→GB). Before this mapping the builder could not call
``visa_needed`` at all without guessing countries from airports —
which the visa layer's never-guess contract forbids. The mapping is
the only bridge, and it only ever reports what the dataset says:
an IATA code the dataset does not know maps to nothing, and the
caller must report that as unknown, never invent a country.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "AIRPORTS_DATA_PATH",
    "airport_country_count",
    "countries_for_route",
    "country_for_iata",
    "lookup_iata",
]

#: The generated mapping, next to this module (same pattern as
#: ``nomorals/wisdom/data/manifest.json``).
AIRPORTS_DATA_PATH = Path(__file__).parent / "data" / "airports_iata_country.csv"

_UNKNOWN_REASON = (
    "no airport with IATA code {code!r} in the airports data — "
    "the country is unknown, not guessed."
)

_cache: dict[str, str] | None = None


def _load() -> dict[str, str]:
    """Load the IATA→country mapping once. Empty on any failure —
    a missing file degrades to 'unknown for everything', never raises."""
    global _cache
    if _cache is not None:
        return _cache
    mapping: dict[str, str] = {}
    try:
        with open(AIRPORTS_DATA_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                code = (row.get("iata_code") or "").strip().upper()
                country = (row.get("iso_country") or "").strip().upper()
                if len(code) == 3 and code.isalpha() \
                        and len(country) == 2 and country.isalpha():
                    mapping.setdefault(code, country)
    except Exception:  # noqa: BLE001 — data problems degrade, never raise
        _log.debug("airport country mapping unavailable", exc_info=True)
    _cache = mapping
    return mapping


def _normalise(iata: Any) -> str:
    if not isinstance(iata, str):
        return ""
    return iata.strip().upper()


def country_for_iata(iata: Any) -> str | None:
    """IATA code → ISO country code, or None when the dataset does not
    know the code. Never raises, never guesses."""
    code = _normalise(iata)
    if len(code) != 3 or not code.isalpha():
        return None
    return _load().get(code)


def lookup_iata(iata: Any) -> dict[str, Any]:
    """Full descriptor: ``{"known", "country", "reason"}``.

    ``known`` is False for unknown AND malformed codes; the reason
    names the code that could not be mapped, so the caller can report
    it honestly instead of inventing a country.
    """
    code = _normalise(iata)
    country = country_for_iata(code)
    if country is None:
        shown = code if code else str(iata)
        return {"known": False, "country": None,
                "reason": _UNKNOWN_REASON.format(code=shown)}
    return {"known": True, "country": country, "reason": ""}


def countries_for_route(
        origin_iata: Any,
        destination_iata: Any) -> tuple[str | None, str | None, list[str]]:
    """Map both ends of a route at once.

    Returns ``(origin_country, destination_country, unmapped)`` where
    ``unmapped`` lists every IATA code the dataset does not know, in
    route order. Empty list means both ends mapped.
    """
    origin = country_for_iata(origin_iata)
    destination = country_for_iata(destination_iata)
    unmapped: list[str] = []
    if origin is None:
        unmapped.append(_normalise(origin_iata))
    if destination is None:
        unmapped.append(_normalise(destination_iata))
    return origin, destination, unmapped


def airport_country_count() -> int:
    """How many IATA codes the loaded mapping knows."""
    return len(_load())
