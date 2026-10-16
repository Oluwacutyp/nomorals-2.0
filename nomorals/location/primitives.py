"""Location primitives: coordinates, places, distance.

Stdlib only. Coordinates are validated on construction — a bad
latitude is a ``ValueError`` at the boundary, never a silent wrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "LatLon", "Place", "haversine_m", "NIGERIA_LAGOS",
    "DEFAULT_PLACE", "default_place", "DEFAULT_PLACES",
]

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class LatLon:
    """A point on Earth. Latitude -90..90, longitude -180..180."""

    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"latitude out of range: {self.lat!r}")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"longitude out of range: {self.lon!r}")

    def to_dict(self) -> dict[str, float]:
        return {"lat": self.lat, "lon": self.lon}

    @classmethod
    def from_dict(cls, data: dict) -> "LatLon":
        return cls(lat=float(data["lat"]), lon=float(data["lon"]))


@dataclass(frozen=True)
class Place:
    """A named place: a label over a coordinate, plus locale context."""

    name: str
    loc: LatLon
    address: str = ""
    locale: str = "ng"

    def to_dict(self) -> dict:
        return {"name": self.name, "loc": self.loc.to_dict(),
                "address": self.address, "locale": self.locale}

    @classmethod
    def from_dict(cls, data: dict) -> "Place":
        return cls(name=str(data["name"]), loc=LatLon.from_dict(data["loc"]),
                   address=str(data.get("address", "")),
                   locale=str(data.get("locale", "ng")))


def haversine_m(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in metres between two points."""
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


#: Lagos, Nigeria — the default place. A default, never a boundary:
#: every function here accepts any coordinate on Earth.
NIGERIA_LAGOS = LatLon(6.5244, 3.3792)

#: Default places per locale pack. Unknown locale → Lagos default.
DEFAULT_PLACES: dict[str, Place] = {
    "ng": Place("Lagos", NIGERIA_LAGOS, "Lagos, Nigeria", "ng"),
    "gh": Place("Accra", LatLon(5.6037, -0.1870), "Accra, Ghana", "gh"),
    "ke": Place("Nairobi", LatLon(-1.2921, 36.8219), "Nairobi, Kenya", "ke"),
    "za": Place("Johannesburg", LatLon(-26.2041, 28.0473),
                "Johannesburg, South Africa", "za"),
    "us": Place("New York", LatLon(40.7128, -74.0060),
                "New York, United States", "us"),
    "gb": Place("London", LatLon(51.5074, -0.1278),
                "London, United Kingdom", "gb"),
    "in": Place("Mumbai", LatLon(19.0760, 72.8777), "Mumbai, India", "in"),
}

DEFAULT_PLACE: Place = DEFAULT_PLACES["ng"]


def default_place(locale: str | None = None) -> Place:
    """Default place for a locale pack; unknown locale → Lagos, Nigeria.

    Never raises for an unknown locale: defaults are not boundaries.
    """
    return DEFAULT_PLACES.get((locale or "ng").lower(), DEFAULT_PLACE)
