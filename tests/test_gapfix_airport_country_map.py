"""Gapfix: airport→country mapping so the itinerary builder can answer
visa questions (R3-67 remainder).

Every IATA code asserted here comes from the OurAirports airports.csv
dump (public domain, fetched 2026-11-17) that generated
nomorals/travel/data/airports_iata_country.csv — nothing is asserted
from memory. Unknown codes stay unknown with the reason named; the
visa layer's never-guess contract holds end to end.
"""

from nomorals.travel.airports import (
    airport_country_count,
    countries_for_route,
    country_for_iata,
    lookup_iata,
)
from nomorals.travel.itinerary import ItineraryBuilder

SAMPLE_FLIGHT_EMAIL = """
Subject: Your booking confirmation - British Airways

Dear Passenger,

Booking reference: ABC123

Flight: BA075
From: LOS (Lagos) to LHR (London)
Departure: 2026-12-01 departs 22:45
Arrival: 2026-12-02 arrives 05:30
"""

SAMPLE_UNKNOWN_ROUTE_EMAIL = """
Booking reference: ZZ9Q2Q
Flight XY123, XXX → ZZZ, departs 08:00 on 2026-12-10, arrives 09:15.
"""


def _builder(tmp_path):
    return ItineraryBuilder(
        db_path=str(tmp_path / "trips.db"),
        vault_dir=str(tmp_path / "vault"))


# ── lookup ────────────────────────────────────────────────────────────────

def test_lookup_home_codes():
    assert country_for_iata("LOS") == "NG"
    assert country_for_iata("ABV") == "NG"
    assert country_for_iata("LHR") == "GB"


def test_lookup_spread_across_continents():
    # Every pair below is a row in the OurAirports dump.
    assert country_for_iata("JFK") == "US"   # North America
    assert country_for_iata("YYZ") == "CA"   # North America
    assert country_for_iata("GRU") == "BR"   # South America
    assert country_for_iata("CDG") == "FR"   # Europe
    assert country_for_iata("NBO") == "KE"   # Africa
    assert country_for_iata("ACC") == "GH"   # Africa
    assert country_for_iata("CAI") == "EG"   # Africa
    assert country_for_iata("DXB") == "AE"   # Asia
    assert country_for_iata("SIN") == "SG"   # Asia
    assert country_for_iata("SYD") == "AU"   # Oceania


def test_lookup_is_case_insensitive_and_strips():
    assert country_for_iata("los") == "NG"
    assert country_for_iata("  Lhr ") == "GB"


def test_lookup_unknown_returns_none_with_reason():
    assert country_for_iata("XXX") is None
    assert country_for_iata("ZZZ") is None
    result = lookup_iata("XXX")
    assert result["known"] is False
    assert result["country"] is None
    assert "XXX" in result["reason"]


def test_lookup_malformed_returns_none():
    assert country_for_iata("") is None
    assert country_for_iata("LO") is None
    assert country_for_iata("LOSS") is None
    assert country_for_iata("L0S") is None
    assert country_for_iata(None) is None


def test_lookup_known_carries_no_reason_to_doubt():
    result = lookup_iata("LOS")
    assert result == {"known": True, "country": "NG", "reason": ""}


def test_mapping_covers_the_whole_dataset():
    # 9,053 distinct IATA codes in the fetched dump; the shipped file
    # carries all of them. A floor, not an exact pin, so a future
    # regeneration from a newer dump does not break the test.
    assert airport_country_count() >= 9000


def test_countries_for_route_names_unmapped_codes():
    origin, dest, unmapped = countries_for_route("LOS", "LHR")
    assert (origin, dest) == ("NG", "GB")
    assert unmapped == []
    origin, dest, unmapped = countries_for_route("XXX", "LHR")
    assert origin is None and dest == "GB"
    assert unmapped == ["XXX"]
    origin, dest, unmapped = countries_for_route("XXX", "ZZZ")
    assert unmapped == ["XXX", "ZZZ"]


# ── builder wire ──────────────────────────────────────────────────────────

def test_builder_answers_visa_for_real_trip(tmp_path):
    builder = _builder(tmp_path)
    trip = builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    assert trip.flights and trip.flights[0].origin == "LOS"
    assert trip.flights[0].destination == "LHR"
    answer = builder.visa_for_trip(trip.id, "NG")
    assert answer["known"] is True
    assert answer["status"] == "visa_required"
    assert answer["needed"] is True
    assert answer["origin_country"] == "NG"
    assert answer["destination_country"] == "GB"
    assert answer["origin_iata"] == "LOS"
    assert answer["destination_iata"] == "LHR"


def test_builder_visa_unknown_airport_is_honest(tmp_path):
    builder = _builder(tmp_path)
    trip = builder.ingest_email(SAMPLE_UNKNOWN_ROUTE_EMAIL)
    assert trip.flights and trip.flights[0].origin == "XXX"
    answer = builder.visa_for_trip(trip.id, "NG")
    assert answer["known"] is False
    assert answer["needed"] is None
    assert "XXX" in answer["notes"]
    assert "ZZZ" in answer["notes"]


def test_builder_visa_missing_trip_is_honest(tmp_path):
    builder = _builder(tmp_path)
    answer = builder.visa_for_trip("trip_doesnotexist", "NG")
    assert answer["known"] is False
    assert answer["needed"] is None


def test_builder_visa_trip_without_flights_is_honest(tmp_path):
    builder = _builder(tmp_path)
    trip = builder.ingest_email(
        "Hotel: Eko Hotel & Suites\nCheck-in: 2026-12-02\n"
        "Check-out: 2026-12-05\nConfirmation number: HTL-99812\n")
    assert not trip.flights
    answer = builder.visa_for_trip(trip.id, "NG")
    assert answer["known"] is False
    assert answer["needed"] is None
    assert "flight" in answer["notes"].lower()


# ── /trip visa verb ───────────────────────────────────────────────────────

def test_trip_visa_explicit_codes_unchanged():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    out = mixin._control_trip("visa NG GH NG")
    assert "visa_free" in out
    assert "NG" in out and "GH" in out
    out = mixin._control_trip("visa NG JP NG")
    assert "unknown" in out.lower()
    assert "visa_free" not in out


def test_trip_visa_trip_form_answers_from_mapping(tmp_path):
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    mixin._trip_builder = _builder(tmp_path)
    trip = mixin._trip_builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    out = mixin._control_trip(f"visa {trip.id} NG")
    assert "visa_required" in out
    assert "LOS" in out and "LHR" in out
    assert "NG" in out and "GB" in out


def test_trip_visa_two_codes_still_usage():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    out = mixin._control_trip("visa NG GB")
    assert out.startswith("usage: /trip visa")


def test_trip_visa_unknown_trip_id_is_honest(tmp_path):
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin
    mixin = RuntimeMemoryMixin()
    mixin._trip_builder = _builder(tmp_path)
    out = mixin._control_trip("visa trip_nope1234 NG")
    assert "unknown" in out.lower()
