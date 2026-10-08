"""Tests for build-map #72: TripIt email-forwarding → auto itinerary.

All offline. Seer/Gmail/GCalendar are mocked.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from nomorals.travel.itinerary import (
    Flight, HotelStay, Trip,
    parse_confirmation, ItineraryBuilder,
    confirmation_hook, added_message,
)

SAMPLE_FLIGHT_EMAIL = """
Subject: Your booking confirmation - British Airways

Dear Passenger,

Thank you for booking with British Airways.

Booking reference: ABC123

Flight: BA075
From: LOS (Lagos) to LHR (London)
Departure: 2026-12-01 departs 22:45
Arrival: 2026-12-02 arrives 05:30

Have a pleasant journey.
"""

SAMPLE_HOTEL_EMAIL = """
Subject: Hotel reservation confirmed

Hotel: Eko Hotel & Suites
Check-in: 2026-12-02
Check-out: 2026-12-05
Confirmation number: HTL-99812

We look forward to hosting you.
"""

SAMPLE_COMBINED = """
Booking reference: XK7P2Q
Flight AP123, LOS → ABV, departs 08:00 on 2026-12-10, arrives 09:15.
Hotel: Transcorp Hilton, check-in 2026-12-10, check-out 2026-12-12.
"""


# ── parsing ─────────────────────────────────────────────────────────────────

def test_pnr_extraction():
    p = parse_confirmation(SAMPLE_FLIGHT_EMAIL)
    assert p["pnr"] == "ABC123"


def test_pnr_variants():
    assert parse_confirmation("PNR: X7K2P9").get("pnr") == "X7K2P9"
    assert parse_confirmation("record locator ABCDEF").get("pnr") == "ABCDEF"
    assert parse_confirmation("no booking here").get("pnr") == ""


def test_flight_number_extraction():
    p = parse_confirmation(SAMPLE_FLIGHT_EMAIL)
    assert p["flights"]
    assert p["flights"][0]["flight_number"] == "BA075"


def test_route_extraction():
    p = parse_confirmation(SAMPLE_FLIGHT_EMAIL)
    f = p["flights"][0]
    assert f["origin"] == "LOS"
    assert f["destination"] == "LHR"


def test_time_extraction():
    p = parse_confirmation(SAMPLE_FLIGHT_EMAIL)
    f = p["flights"][0]
    assert "22:45" in f["departs"]
    assert "05:30" in f["arrives"]


def test_hotel_extraction():
    p = parse_confirmation(SAMPLE_HOTEL_EMAIL)
    assert p["hotels"]
    h = p["hotels"][0]
    assert "Eko Hotel" in h["name"]
    assert h["check_in"] == "2026-12-02"
    assert h["check_out"] == "2026-12-05"
    assert h["confirmation"] == "HTL-99812"


def test_combined_booking():
    p = parse_confirmation(SAMPLE_COMBINED)
    assert p["pnr"] == "XK7P2Q"
    assert p["flights"][0]["flight_number"] == "AP123"
    assert p["hotels"][0]["name"] == "Transcorp Hilton"


def test_parse_never_raises():
    p = parse_confirmation(None)
    assert p["flights"] == []
    p2 = parse_confirmation("!@#$%^&*()")
    assert p2["pnr"] == ""


def test_no_false_positive_times():
    # "22:45" alone shouldn't create a flight
    p = parse_confirmation("Meet me at 22:45 for dinner.")
    assert p["flights"] == []


# ── builder ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def builder(tmp_path):
    return ItineraryBuilder(
        db_path=str(tmp_path / "trips.db"),
        vault_dir=str(tmp_path / "vault"),
    )


def test_ingest_email(builder):
    trip = builder.ingest_email(SAMPLE_FLIGHT_EMAIL,
                                subject="booking confirmation")
    assert trip.flights
    assert trip.flights[0].pnr == "ABC123"
    assert trip.flights[0].flight_number == "BA075"
    assert "email.txt" in trip.docs


def test_ingest_hotel_email(builder):
    trip = builder.ingest_email(SAMPLE_HOTEL_EMAIL)
    assert trip.hotels
    assert "Eko Hotel" in trip.hotels[0].name


def test_ingest_image_mocked_seer(builder, tmp_path):
    seer = MagicMock()
    seer.see.return_value = (
        "Flight BA075, LOS to LHR. Departs 22:45 on 2026-12-01. "
        "Booking reference ABC123."
    )
    img = tmp_path / "confirm.png"
    img.write_bytes(b"fake-png")
    trip = builder.ingest_image(str(img), seer=seer)
    assert trip.flights
    assert trip.flights[0].flight_number == "BA075"
    seer.see.assert_called_once()


def test_ingest_image_no_seer(builder, tmp_path):
    img = tmp_path / "confirm.png"
    img.write_bytes(b"fake-png")
    trip = builder.ingest_image(str(img), seer=None)
    assert trip.is_empty()  # no vision, no parse — honest


def test_trip_roundtrip(builder):
    trip = builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    loaded = builder.get_trip(trip.id)
    assert loaded is not None
    assert loaded.flights[0].pnr == "ABC123"
    assert builder.get_trip("nope") is None


def test_list_trips(builder):
    builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    builder.ingest_email(SAMPLE_HOTEL_EMAIL)
    trips = builder.list_trips()
    assert len(trips) == 2


def test_summary_format(builder):
    trip = builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    s = builder.summary(trip.id)
    assert "LOS→LHR" in s
    assert "BA075" in s
    assert "ABC123" in s


def test_vault_docs(builder):
    trip = builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    docs = builder.vault_docs(trip.id)
    assert "email.txt" in docs
    # raw text actually on disk
    path = os.path.join(builder.vault_dir, trip.id, "email.txt")
    assert os.path.exists(path)


def test_to_calendar_mocked_gcal(builder):
    trip = builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    gcal = MagicMock()
    gcal.create_event.return_value = {"id": "ev1"}
    created = builder.to_calendar(trip.id, gcal, confirmed=True)
    assert len(created) == 1
    gcal.create_event.assert_called_once()
    kwargs = gcal.create_event.call_args
    assert "BA075" in kwargs.kwargs.get("summary", "") or \
           "BA075" in str(kwargs)


def test_to_calendar_no_gcal(builder):
    trip = builder.ingest_email(SAMPLE_FLIGHT_EMAIL)
    assert builder.to_calendar(trip.id, None) == []


def test_to_calendar_no_trip(builder):
    gcal = MagicMock()
    assert builder.to_calendar("nope", gcal) == []


# ── chat helpers ────────────────────────────────────────────────────────────

def test_confirmation_hook_detects():
    assert confirmation_hook(SAMPLE_FLIGHT_EMAIL)
    assert confirmation_hook(SAMPLE_COMBINED)


def test_confirmation_hook_rejects():
    assert not confirmation_hook("hey how are you")
    assert not confirmation_hook("track LOS to LHR flights for me")
    assert not confirmation_hook("short")


def test_added_message():
    trip = Trip(id="trip_x", name="LOS → LHR",
                flights=[Flight(airline="", flight_number="BA075",
                                origin="LOS", destination="LHR",
                                departs="2026-12-01T22:45", pnr="ABC123")])
    msg = added_message(trip)
    assert "BA075" in msg
    assert "22:45" in msg
    assert "ABC123" in msg
    assert "track it" in msg


def test_added_message_hotel_only():
    trip = Trip(id="trip_y",
                hotels=[HotelStay(name="Eko Hotel",
                                  check_in="2026-12-02")])
    msg = added_message(trip)
    assert "Eko Hotel" in msg
    assert "track it" not in msg  # no flights → no track offer
